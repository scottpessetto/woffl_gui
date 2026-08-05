"""Wells router - list, selection context, tests, and survey profile."""

from __future__ import annotations

from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException, Query

from server.schemas import (
    WellContext,
    WellProfileResponse,
    WellsResponse,
    WellTestsResponse,
)
from server.services import tests as tests_svc
from server.services import wells as wells_svc

router = APIRouter(tags=["wells"])


@router.get("/wells", response_model=WellsResponse)
def get_wells() -> dict[str, Any]:
    """All known wells from the characteristics frame."""
    return wells_svc.list_wells()


@router.get("/wells/{name}/context", response_model=WellContext)
def get_well_context(
    name: str,
    months: int = Query(6, ge=1, le=24),
    cap: int = Query(0, ge=0, le=50),
) -> dict[str, Any]:
    """Server-side replay of the sidebar seeding pipeline for one well."""
    try:
        return wells_svc.well_context(name, months, cap)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail={"error": "invalid", "message": f"unknown well {name}"},
        ) from None


@router.get("/wells/{name}/tests", response_model=WellTestsResponse)
def get_well_tests(
    name: str,
    months: int = Query(6, ge=1, le=24),
    cap: int = Query(0, ge=0, le=50),
) -> dict[str, Any]:
    """JSON-safe well-test rows, newest first ([] when the well has none)."""
    return {"well": name, "tests": tests_svc.tests_json(name, months, cap)}


@router.get("/wells/{name}/profile", response_model=WellProfileResponse)
def get_well_profile(
    name: str,
    jpump_tvd: Optional[float] = Query(None, ge=2500, le=8000),
    field_model: Optional[Literal["Schrader", "Kuparuk"]] = Query(None),
) -> dict[str, Any]:
    """Survey-based well profile; field-model preset fallback when no survey."""
    return wells_svc.well_profile_payload(name, jpump_tvd, field_model)
