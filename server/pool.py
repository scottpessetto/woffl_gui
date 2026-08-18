"""The one process pool every CPU sweep shares.

Why a pool at all: the interactive sweeps (PF range, batch) are independent
per point and run on the request thread today, so they hold the GIL for
seconds. Measured on a 10-point PF sweep: the sweep itself is 4.03 s, and
while it runs an ordinary /context read goes from 33 ms to 1,252 ms. Three
concurrent sweeps cost 11.6 s each - the same as running them serially -
because the AnyIO threadpool interleaves GIL slices and buys no throughput.
Moving the work into processes fixes both halves: it uses the second vCPU,
and the request thread just waits on a future (GIL released), so ordinary
reads stay fast.

Why PERSISTENT: pool setup is most of the cost at this size. Same sweep,
WOFFL_MAX_WORKERS=2:

    serial (today)                    4.03 s   1.00x
    ProcessPoolExecutor per call      3.42 s   1.18x
    persistent pool                   2.01 s   2.00x

A per-call pool hands back most of the win. That is also why
``sensitivity._PARALLEL_MIN_RUNS`` has to exist - the threshold is there to
dodge a setup cost this module does not pay.

Why NOT plain fork: at first use the parent is a uvicorn process holding the
event loop, the AnyIO worker threads, the warm loop, the SWR refresh pool and
live Databricks sockets. A fork child inherits locks held by threads that do
not exist in it. Python 3.12 warns about fork() in a multi-threaded process
and 3.14 makes forkserver the Linux default, so this module uses forkserver
where it exists and spawn elsewhere (Windows). Both re-import in the worker,
which is the whole reason the pool is created ONCE at startup and primed
there: the import cost lands on boot instead of on the first engineer to run
a sweep.

Sizing is ``worker_ceiling()`` and nothing else - the same budget every other
ProcessPool in the tree obeys. app.yaml pins 2 for the 2-vCPU tier.

``submit_all`` is the only entry point. It is deliberately all-or-nothing:
any pool-level failure returns None and the caller reruns its own serial
loop, matching the fallback contract in ``network_optimizer`` and
``sensitivity._solve_parallel``. Nothing here retries a task - a task that
raises is the caller's error to surface, exactly as the serial path would.
"""

from __future__ import annotations

import logging
import multiprocessing
import threading
from concurrent.futures import Executor, ProcessPoolExecutor, as_completed
from typing import Any, Callable, Iterable, Optional, TypeVar

log = logging.getLogger("woffl.web.pool")

T = TypeVar("T")

# Test seam, mirroring sensitivity._EXECUTOR_CLS: tests swap in a
# ThreadPoolExecutor so monkeypatched solves stay visible (a real child
# process cannot see them). None = the real ProcessPoolExecutor.
_EXECUTOR_CLS: Optional[type] = None

_POOL: Optional[Executor] = None
_LOCK = threading.RLock()
# Bounds how many sweeps may be in the pool at once. Without it three
# concurrent sweeps oversubscribe two workers and every one of them slows
# down; with it they queue and each finishes at full speed.
_GATE: Optional[threading.Semaphore] = None
_WORKERS = 0


def _context():
    """forkserver where it exists (Linux), else spawn (Windows).

    Never plain fork - see the module docstring. forkserver starts one clean
    single-threaded server process and forks workers from THAT, so no worker
    inherits the uvicorn process's threads or their locks.
    """
    available = multiprocessing.get_all_start_methods()
    return multiprocessing.get_context("forkserver" if "forkserver" in available else "spawn")


def _prime() -> bool:
    """Priming task: import the sweep stack inside the worker.

    Booting the worker is not enough, and neither is importing. A child starts
    with cold imports AND a cold per-process profile cache, so the first sweep
    funds both. Measured, first PF sweep vs steady state on the same server:

        boot only                    3,949 ms  (steady 2,061 ms)
        + imports                    3,076 ms
        + one default solve          see below

    So the prime runs a real solve on Custom with default params - no
    Databricks, no well data, ~17 ms of actual work - which walks the whole
    path the sweep will take: factories, PVT, the preset WellProfile, scipy.

    Never raises: a worker that cannot prime is still a usable worker, and a
    failure here must not take the pool (or startup) down with it.
    """
    try:
        from server import schemas
        from server.services import solve

        solve.solve_single("Custom", schemas.SimParams())
    except Exception:  # noqa: BLE001 - priming is an optimization, not a gate
        log.debug("worker prime solve failed; worker still usable", exc_info=True)
    return True


def start() -> int:
    """Create and PRIME the pool. Called once from the app lifespan.

    Priming is not optional bookkeeping: a forkserver/spawn worker imports the
    woffl stack on its first task, and that import is ~1.3 s. Paying it here
    means the first sweep of the day is fast instead of being the one request
    that funds every worker's startup.

    Returns:
        int: worker count, or 0 when the pool could not start (the app runs
        fine without it - every caller has a serial fallback).
    """
    global _POOL, _GATE, _WORKERS
    from woffl.assembly.parallelism import worker_ceiling

    with _LOCK:
        if _POOL is not None:
            return _WORKERS
        workers = max(1, worker_ceiling())
        if workers == 1:
            # One worker is strictly worse than staying in-process: all the
            # IPC, none of the parallelism. Callers fall back to serial.
            log.info("process pool disabled (worker_ceiling=1)")
            return 0
        cls = _EXECUTOR_CLS or ProcessPoolExecutor
        try:
            if cls is ProcessPoolExecutor:
                _POOL = cls(max_workers=workers, mp_context=_context())
            else:  # test seam - ThreadPoolExecutor takes no mp_context
                _POOL = cls(max_workers=workers)
            # Prime every worker, not just one: each has its own imports.
            for fut in [_POOL.submit(_prime) for _ in range(workers)]:
                fut.result(timeout=180)
        except Exception as exc:  # noqa: BLE001 - never block startup
            log.warning("process pool unavailable, sweeps stay serial: %r", exc)
            _shutdown_locked()
            return 0
        _GATE = threading.Semaphore(workers)
        _WORKERS = workers
        log.info("process pool ready: %d workers (%s)", workers, _context().get_start_method())
        return workers


def _shutdown_locked() -> None:
    global _POOL, _GATE, _WORKERS
    pool, _POOL, _GATE, _WORKERS = _POOL, None, None, 0
    if pool is not None:
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except Exception:  # noqa: BLE001 - shutdown must not raise
            log.warning("process pool shutdown failed", exc_info=True)


def stop() -> None:
    """Tear the pool down (app lifespan shutdown, tests)."""
    with _LOCK:
        _shutdown_locked()


def workers() -> int:
    """Live worker count; 0 when the pool is down and callers must go serial."""
    with _LOCK:
        return _WORKERS


def submit_all(fn: Callable[..., T], jobs: Iterable[tuple]) -> Optional[list[T]]:
    """Run ``fn(*job)`` for every job, in parallel, results in job order.

    Args:
        fn: A module-level callable (picklable) doing one unit of work.
        jobs: Argument tuples, one per unit.

    Returns:
        list[T] in the order of ``jobs``, or None when the pool is
        unavailable or broke mid-run - the caller then reruns serially. A
        task that RAISES is re-raised here, because the serial path would
        have raised too; only pool-level failure returns None.
    """
    jobs = list(jobs)
    if not jobs:
        return []

    with _LOCK:
        pool, gate = _POOL, _GATE
    if pool is None or gate is None:
        return None

    # Queue behind other sweeps rather than oversubscribing the workers.
    gate.acquire()
    try:
        results: list[Any] = [None] * len(jobs)
        futures = {pool.submit(fn, *job): i for i, job in enumerate(jobs)}
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
        return results
    except Exception as exc:  # noqa: BLE001 - caller reruns serially
        from concurrent.futures.process import BrokenProcessPool

        if isinstance(exc, BrokenProcessPool):
            # A dead pool stays dead; drop it so later calls go straight to
            # their serial path instead of failing one by one.
            log.warning("process pool broke, falling back to serial for the process life")
            with _LOCK:
                _shutdown_locked()
            return None
        raise
    finally:
        gate.release()
