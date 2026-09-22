"""Cancelling optimize-tab jobs (DELETE /api/optimize/run/{id}): a long job
stops at its next progress step, polls as "cancelled" and frees its slot."""

import threading
import time

from fastapi.testclient import TestClient

from server import jobs
from server.main import app


def _wait(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_cancel_stops_a_running_pad_job_at_its_next_progress_step():
    started = threading.Event()

    def runner(job):
        started.set()
        for i in range(500):
            jobs.set_progress(job, f"trial {i}")
            time.sleep(0.01)
        return {"finished": True}

    job_id = jobs.start("pad", runner)
    assert started.wait(5)
    client = TestClient(app)
    assert client.delete(f"/api/optimize/run/{job_id}").json() == {"cancel_requested": True}
    assert _wait(lambda: client.get(f"/api/optimize/run/{job_id}").json()["status"] == "cancelled")
    body = client.get(f"/api/optimize/run/{job_id}").json()
    assert body["result"] is None and body["kind"] == "pad"


def test_cancel_is_scoped_to_optimize_job_kinds():
    job_id = jobs.start("sensitivity_combine", lambda job: {})
    client = TestClient(app)
    assert client.delete(f"/api/optimize/run/{job_id}").status_code == 404
    assert client.delete("/api/optimize/run/nope").status_code == 404


def test_set_progress_raises_only_after_cancel():
    job = {"cancel_event": threading.Event()}
    jobs.set_progress(job, "a")
    assert job["progress"] == "a"
    job["cancel_event"].set()
    try:
        jobs.set_progress(job, "b")
    except jobs.JobCancelled:
        pass
    else:
        raise AssertionError("expected JobCancelled")
    assert job["progress"] == "a"
    jobs.set_progress({}, "plain dicts from tests never cancel")
