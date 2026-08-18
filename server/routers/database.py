"""Well Database endpoints: chars table, aging pumps, prop_hist audit."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from server import schemas
from server.services import database as database_service

router = APIRouter(prefix="/database", tags=["database"])


@router.get("/wells", response_model=schemas.WellDatabaseResponse)
def database_wells() -> Any:
    """Chars table rows + wells lacking a deviation survey."""
    return database_service.database_rows()


@router.get("/aging-pumps", response_model=schemas.AgingPumpsResponse)
def aging_pumps(
    known_only: bool = Query(True, description="Only wells present in the chars table"),
    online_only: bool = Query(False, description="Only wells with a recent well test (allocated or info-only)"),
    online_days: int = Query(60, ge=1, le=3650, description="Recency window for online, days"),
    min_days: int = Query(365, ge=0, le=36500, description="Minimum days in hole"),
) -> Any:
    """Current-pump tenure per well, oldest first."""
    return database_service.aging_pumps(known_only, online_only, online_days, min_days)


@router.get("/prop-history/{well}", response_model=schemas.PropHistoryResponse)
def prop_history(well: str) -> Any:
    """prop_hist audit trail for one well: current stored state + full history."""
    from woffl.assembly.sql_guards import UnsafeSqlValueError, validate_well_name

    try:
        validate_well_name(well)
    except UnsafeSqlValueError:
        return JSONResponse(
            status_code=404,
            content={"error": "invalid", "message": f"Unknown well: {well!r}"},
        )
    payload = database_service.prop_history_payload(well)
    if payload is None:
        return JSONResponse(
            status_code=404,
            content={"error": "invalid", "message": f"Unknown well: {well!r} (no enthid in vw_well_header)"},
        )
    return payload
