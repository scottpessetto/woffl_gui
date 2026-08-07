"""In-process background job registry.

One shared registry for every long compute the web app runs off the request
thread: pad / CFP optimization runs and the combined-permutations
sensitivity study. All of them are READ-ONLY compute - nothing here writes
anywhere - so a job is just a thread, a status envelope and a result held in
memory until the caller polls it.

Deliberately in-process, not a task queue. This app is one uvicorn worker in
front of a handful of engineers; a broker would add an operational surface
(a second process to run, a queue to drain, results to serialize) to buy
durability nobody asked for. The consequence is honest and bounded: a server
restart loses running jobs, the client's poll 404s, and the caller drops the
stale id and re-runs.

Settled jobs are pruned an hour after they finish, on the next start. That
is long enough for an engineer to come back from lunch to a finished run and
short enough that a day of runs is not still resident at 5pm.

The runner callable receives the job's own mutable dict, so it can publish
progress by assigning ``job["progress"]``. Nothing else in the dict is the
runner's to touch.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Optional

log = logging.getLogger("woffl.web.jobs")

# A settled job stays readable this long. Running jobs are never pruned.
_JOB_TTL_SECONDS = 3600.0
_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()

Runner = Callable[[dict[str, Any]], dict[str, Any]]


def _prune_jobs() -> None:
    now = time.monotonic()
    with _JOBS_LOCK:
        dead = [
            jid
            for jid, j in _JOBS.items()
            if j["status"] != "running" and now - j["settled_mono"] > _JOB_TTL_SECONDS
        ]
        for jid in dead:
            del _JOBS[jid]


def get(job_id: str, kinds: Optional[tuple[str, ...]] = None) -> Optional[dict[str, Any]]:
    """Poll envelope for one job, or None when it is unknown or expired.

    Args:
        job_id (str): Id handed back by :func:`start`.
        kinds (tuple, optional): Restrict to these job kinds. One registry
            backs several endpoints, and an id from another endpoint's
            namespace is "unknown" here, not a type error at the response
            model.

    Returns:
        dict | None: job_id, kind, status ("running" | "done" | "error"),
        progress, result, error, started_at (wall clock, ISO seconds) and
        seconds elapsed. None when there is no such job.
    """
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None or (kinds is not None and job["kind"] not in kinds):
            return None
        return {
            "job_id": job_id,
            "kind": job["kind"],
            "status": job["status"],
            "progress": job["progress"],
            "result": job["result"],
            "error": job["error"],
            "started_at": job["started_at"],
            "seconds": round(time.monotonic() - job["started_mono"], 1),
        }


def start(kind: str, run: Runner, progress: str = "starting...") -> str:
    """Spawn ``run`` on a daemon thread; returns the job id immediately.

    Args:
        kind (str): Job kind, echoed in the poll envelope and used to scope
            :func:`get`.
        run (Runner): Called with the job's mutable dict; its return value
            becomes ``result``. Assign ``job["progress"]`` to report
            progress.
        progress (str): First progress line, readable before the runner has
            published one of its own.

    Returns:
        str: The job id.
    """
    _prune_jobs()
    job_id = uuid.uuid4().hex[:12]
    job: dict[str, Any] = {
        "kind": kind,
        "status": "running",
        "progress": progress,
        "result": None,
        "error": None,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "started_mono": time.monotonic(),
        "settled_mono": 0.0,
    }
    with _JOBS_LOCK:
        _JOBS[job_id] = job

    def target() -> None:
        try:
            job["result"] = run(job)
            job["status"] = "done"
            job["progress"] = "done"
        except Exception as exc:  # noqa: BLE001 - job surface, never crash the server
            log.exception("%s job %s failed", kind, job_id)
            job["status"] = "error"
            job["error"] = str(exc)
        finally:
            job["settled_mono"] = time.monotonic()

    threading.Thread(target=target, daemon=True, name=f"job-{kind}-{job_id}").start()
    return job_id
