"""JP install history endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from server import schemas
from server.services import history as history_service
from server.services import pump_match
from server import jobs

router = APIRouter(tags=["history"])


@router.get("/wells/{name}/jp-history", response_model=schemas.JpHistoryResponse)
def jp_history(name: str) -> Any:
    """Install rows + extended test/BHP window for the history chart.

    A well name that fails the SQL-guard shape check is unknown by
    definition (every real well passes); a valid name with no tracker rows
    returns an empty payload, matching the tab's "No JP history found".
    """
    from woffl.assembly.sql_guards import UnsafeSqlValueError, validate_well_name

    try:
        validate_well_name(name)
    except UnsafeSqlValueError:
        return JSONResponse(
            status_code=404,
            content={"error": "invalid", "message": f"Unknown well: {name!r}"},
        )
    return history_service.jp_history_payload(name)


@router.post("/wells/{name}/pump-match", response_model=schemas.OptimizeRunStarted)
def start_pump_match(name: str, req: schemas.PumpMatchRequest) -> Any:
    from woffl.assembly.sql_guards import UnsafeSqlValueError, validate_well_name
    try:
        validate_well_name(name)
    except UnsafeSqlValueError:
        raise HTTPException(404, "Unknown well")
    return {"job_id": pump_match.start(name, req)}


@router.get("/pump-match/{job_id}", response_model=schemas.PumpMatchJob)
def get_pump_match(job_id: str) -> Any:
    result = jobs.get(job_id, pump_match.KINDS)
    if result is None:
        raise HTTPException(404, "History job expired or unavailable. Run the comparison again.")
    return result


@router.delete("/pump-match/{job_id}")
def cancel_pump_match(job_id: str) -> Any:
    if not jobs.cancel(job_id, pump_match.KINDS):
        raise HTTPException(404, "History job expired or unavailable")
    return {"cancel_requested": True}
