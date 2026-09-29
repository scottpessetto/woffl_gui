"""Header page: production-header pressure impact on every lift type.

Board and run are background jobs (read-only compute). Save writes the chosen
WHP/BHP relations and IPRs to mpu.wells.prop_hist through the one gated
writer: 403 when ALLOW_DATABRICKS_WRITES is off, and every row is stamped
with the acting engineer (identity.bind_entry_user).
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request

from server import jobs, schemas
from server.identity import bind_entry_user
from server.services import header_study

router = APIRouter(prefix="/header", tags=["header"])


def _invalid(exc: Exception) -> HTTPException:
    return HTTPException(status_code=422, detail={"error": "invalid", "message": str(exc)})


@router.get("/pads")
def get_pads() -> Any:
    """Pads with producers tested in the last six months."""
    return {"pads": header_study.list_pads()}


@router.get("/well/{well}")
def get_well(well: str, fit_days: int = 120, pads: Optional[str] = None) -> Any:
    """Per-well review data: trends, day-by-day fits and tests (read-only).

    ``pads`` (comma-separated) reuses the board's cached pad-wide historian
    pull instead of a per-well query.
    """
    import re as _re

    name = well.strip().upper()
    if not _re.fullmatch(r"MP[A-Z]-\d{1,3}[A-Z]?", name):
        raise _invalid(ValueError(f"invalid well name '{well}'"))
    if not 30 <= int(fit_days) <= 365:
        raise _invalid(ValueError("fit_days must be between 30 and 365"))
    try:
        pad_list = header_study.clean_pads(pads.split(",")) if pads else None
    except ValueError as exc:
        raise _invalid(exc) from exc
    return header_study.well_detail(name, int(fit_days), pad_list)


@router.post("/board", response_model=schemas.OptimizeRunStarted)
def start_board(req: schemas.HeaderBoardRequest) -> Any:
    """Load wells, relations, IPR candidates and lift-group correlations."""
    try:
        return {"job_id": header_study.start_board(req.pads, req.fit_days)}
    except ValueError as exc:
        raise _invalid(exc) from exc


@router.post("/run", response_model=schemas.OptimizeRunStarted)
def start_run(req: schemas.HeaderRunRequest) -> Any:
    """Estimate the oil/liquid response to a production-header change."""
    try:
        return {"job_id": header_study.start_run(req)}
    except ValueError as exc:
        raise _invalid(exc) from exc


@router.get("/job/{job_id}", response_model=schemas.HeaderJobStatus)
def job_status(job_id: str) -> Any:
    job = jobs.get(job_id, header_study.JOB_KINDS)
    if job is None:
        raise HTTPException(status_code=404, detail={"error": "invalid", "message": f"unknown or expired job {job_id}"})
    return job


@router.delete("/job/{job_id}")
def cancel_job(job_id: str) -> Any:
    if not jobs.cancel(job_id, header_study.JOB_KINDS):
        raise HTTPException(status_code=404, detail={"error": "invalid", "message": f"unknown or expired job {job_id}"})
    return {"cancel_requested": True}


@router.post("/save")
def save(req: schemas.HeaderSaveRequest, request: Request) -> Any:
    """Save chosen relations/IPRs (append-only prop_hist rows, one statement per well)."""
    from woffl.gui.ipr_anchor import writes_enabled

    if not writes_enabled():
        raise HTTPException(status_code=403, detail={
            "error": "writes_disabled",
            "message": "Saving requires ALLOW_DATABRICKS_WRITES=true in the app environment."})
    bind_entry_user(request)
    try:
        return header_study.save(req)
    except ValueError as exc:
        raise _invalid(exc) from exc
