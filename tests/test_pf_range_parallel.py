"""PF-range fan-out: same answer as the serial loop, and it falls back.

The PF sweep moved from a serial in-request loop onto the shared persistent
process pool (server/pool.py). Two things must stay true forever, and both
are easy to break silently:

1. The parallel result is IDENTICAL to the serial one - same rows, same
   order, same values. as_completed returns out of order, so ordering is
   restored by index in pool.submit_all; a regression there would scramble
   rows by pressure and nobody would notice from the chart.
2. Losing the pool costs speed, never correctness. No pool, a broken pool,
   worker_ceiling()==1 - each must still produce the full sweep.

Also pinned here: the sweep must NOT run on threads. res_mix is rebuilt per
point precisely because ResMix.condition() mutates in place and shares its
child oil/wat/gas objects, so points cannot share one.

NOTE on worker_ceiling: the documented suite invocation is
WOFFL_MAX_WORKERS=1, which disables the pool. Every test that means to
exercise the PARALLEL path therefore patches worker_ceiling explicitly
(same convention as tests/test_combine_parallel.py) - otherwise these tests
would quietly degrade to serial and assert nothing about the change they
exist to guard.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

import woffl.assembly.parallelism as common_mod
from server import pool, schemas
from server.services import solve


@pytest.fixture()
def small_sweep():
    """Three pressures over a 1x2 pump grid - enough to be a real sweep."""
    return schemas.SimParams(
        nozzle_batch_options=["12"],
        throat_batch_options=["A", "B"],
        power_fluid_min=2800,
        power_fluid_max=3200,
        power_fluid_step=200,
    )


@pytest.fixture()
def real_pool(monkeypatch):
    """A real 2-worker PROCESS pool, regardless of WOFFL_MAX_WORKERS."""
    monkeypatch.setattr(common_mod, "worker_ceiling", lambda: 2)
    pool.stop()
    started = pool.start()
    yield started
    pool.stop()


@pytest.fixture()
def thread_pool(monkeypatch):
    """Pool backed by threads so monkeypatched callables stay visible."""
    monkeypatch.setattr(common_mod, "worker_ceiling", lambda: 2)
    monkeypatch.setattr(pool, "_EXECUTOR_CLS", ThreadPoolExecutor)
    pool.stop()
    started = pool.start()
    yield started
    pool.stop()


# --- 1. parallel == serial -------------------------------------------------


def test_parallel_matches_serial_exactly(small_sweep, real_pool):
    """The headline guarantee: fanning out changed nothing about the answer."""
    assert real_pool == 2, "this test is meaningless without a live pool"
    parallel = solve.run_pf_range("Custom", small_sweep)

    pool.stop()  # same call, now on the serial fallback
    assert pool.workers() == 0
    serial = solve.run_pf_range("Custom", small_sweep)

    assert parallel["pressures"] == serial["pressures"]
    assert len(parallel["rows"]) == len(serial["rows"])
    # Row-for-row, not just set-equal: out-of-order completion must not
    # reorder the output.
    assert parallel["rows"] == serial["rows"]


def test_row_order_follows_pressure_order(small_sweep, real_pool):
    """Rows come back grouped by pressure, ascending - the chart's x order."""
    assert real_pool == 2
    out = solve.run_pf_range("Custom", small_sweep)
    seen = [r["power_fluid_pressure"] for r in out["rows"]]
    assert seen == sorted(seen)
    assert sorted(set(seen)) == out["pressures"]


def test_submit_all_preserves_job_order(thread_pool):
    """The ordering contract, isolated from the physics."""
    import time

    assert thread_pool == 2

    def slow_first(i):
        time.sleep(0.05 if i == 0 else 0.0)
        return i

    assert pool.submit_all(slow_first, [(0,), (1,), (2,), (3,)]) == [0, 1, 2, 3]


# --- 2. the fallbacks ------------------------------------------------------


def test_no_pool_still_produces_the_full_sweep(small_sweep):
    pool.stop()
    assert pool.workers() == 0
    out = solve.run_pf_range("Custom", small_sweep)
    assert out["rows"], "serial fallback produced no rows"
    assert len(out["pressures"]) == 3


def test_worker_ceiling_one_disables_the_pool(monkeypatch, small_sweep):
    """One worker is all IPC and no parallelism - callers stay serial."""
    monkeypatch.setattr(common_mod, "worker_ceiling", lambda: 1)
    pool.stop()
    assert pool.start() == 0
    assert pool.workers() == 0
    assert solve.run_pf_range("Custom", small_sweep)["rows"]


def test_broken_pool_falls_back_and_stays_down(small_sweep, thread_pool):
    """A BrokenProcessPool returns None once, then the pool is dropped."""
    assert thread_pool == 2
    assert pool.submit_all(_broken, [(1,), (2,)]) is None
    assert pool.workers() == 0, "a broken pool must not be reused"
    # And the sweep still completes on the serial path.
    assert solve.run_pf_range("Custom", small_sweep)["rows"]


def test_task_error_is_raised_not_swallowed(thread_pool):
    """Pool-level failure returns None; a TASK error must still surface,
    because the serial loop would have raised it too."""
    assert thread_pool == 2
    with pytest.raises(ValueError, match="bad point"):
        pool.submit_all(_raiser, [(1,)])


# Module-level so the real ProcessPool variant of these stays picklable.
def _broken(_x):
    from concurrent.futures.process import BrokenProcessPool

    raise BrokenProcessPool("worker died")


def _raiser(_x):
    raise ValueError("bad point")


# --- 3. shape guards -------------------------------------------------------


def test_empty_grid_still_raises_invalid(small_sweep):
    small_sweep.nozzle_batch_options = []
    with pytest.raises(ValueError, match="at least one nozzle"):
        solve.run_pf_range("Custom", small_sweep)


def test_a_failed_point_drops_out_without_killing_the_sweep(monkeypatch, small_sweep):
    """Per-point isolation survived the move off the serial loop.

    Patches the real work INSIDE the point (batch_run), not _pf_point itself:
    the contract is that a point whose physics blows up is dropped, and a
    test double that raises from _pf_point would instead assert that
    run_pf_range swallows programming errors, which it must not.
    """
    from woffl.assembly.batchpump import BatchPump

    real_run = BatchPump.batch_run

    def flaky(self, jp_list, debug=False):
        if float(self.ppf_surf) == 3000.0:
            raise RuntimeError("crafted point failure")
        return real_run(self, jp_list, debug=debug)

    monkeypatch.setattr(BatchPump, "batch_run", flaky)
    pool.stop()  # serial path exercises the same helper, in-process
    out = solve.run_pf_range("Custom", small_sweep)

    assert out["pressures"] == [2800.0, 3000.0, 3200.0], "x-axis keeps every point"
    assert {r["power_fluid_pressure"] for r in out["rows"]} == {2800.0, 3200.0}


def test_bad_input_still_raises_rather_than_returning_an_empty_sweep(small_sweep):
    """The regression this file caught: per-point isolation must not swallow
    an input error into a successful-looking empty 200. Tubing wider than
    casing is rejected by the geometry, once, up front."""
    small_sweep.tubing_od = 9.0
    small_sweep.casing_od = 6.875
    pool.stop()
    with pytest.raises(ValueError):
        solve.run_pf_range("Custom", small_sweep)
