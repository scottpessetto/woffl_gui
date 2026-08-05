"""Pump catalog endpoints - cross-brand equivalents lookup."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from server import schemas
from server.services import pumps

router = APIRouter(tags=["pumps"])


@router.get("/pumps/equivalents", response_model=schemas.EquivalentsResponse)
def get_equivalents(
    nozzle: str = Query(...), throat: str = Query(...)
) -> schemas.EquivalentsResponse:
    """Closest Guiberson/Kobe/Petrolift match for a National nozzle + throat."""
    try:
        return schemas.EquivalentsResponse(**pumps.equivalents(nozzle, throat))
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": "invalid", "message": str(exc), "suggested_gor": None},
        ) from exc
