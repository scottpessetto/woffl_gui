"""JP install history endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from server import schemas
from server.services import history as history_service

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
