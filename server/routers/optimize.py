"""Optimization pad board (read-only).

The redesigned optimization workflow: engineers match and save well fits on
the Single Well solver; this surface reports per-pad READINESS - which
wells have a saved fit, when, by whom, and what is missing. Offline flags
and future wells live client-side (the engineer's working config), so the
server stays stateless here.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from server import schemas
from server.services import ipr, optimizer_runs

router = APIRouter(prefix="/optimize", tags=["optimize"])


@router.get("/pad-status", response_model=schemas.PadFitStatusResponse)
def pad_status(
    pad: str = Query(..., min_length=1, max_length=8),
    extra: list[str] = Query([]),
) -> Any:
    """Saved-fit readiness for every well on ``pad``, plus any ``extra``
    donor wells (future wells may match a well on another pad)."""
    return ipr.pad_fit_status(pad, extra)


@router.post("/run", response_model=schemas.OptimizeRunStarted)
def start_run(req: schemas.OptimizeRunRequest) -> Any:
    """Start a pad or CFP optimization run as a background job.

    Read-only compute over saved fits - nothing is written anywhere. Runs
    take minutes (full batch simulation per trial header); poll
    GET /optimize/run/{job_id} for progress and the result.
    """
    if req.kind == "pad" and req.pad is None:
        raise HTTPException(
            status_code=422,
            detail={"error": "invalid", "message": "kind=pad requires pad (S, I or M)"},
        )
    return {"job_id": optimizer_runs.start_run(req)}


@router.get("/run/{job_id}", response_model=schemas.OptimizeJobStatus)
def run_status(job_id: str) -> Any:
    """Job status; `result` populates when status becomes done."""
    job = optimizer_runs.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "invalid", "message": f"unknown or expired job {job_id}"},
        )
    return job
