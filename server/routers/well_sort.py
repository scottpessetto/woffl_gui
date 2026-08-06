"""Well Sort endpoints: field tables, down events, marginal WC, triage,
bench workbook. Port of woffl/gui/well_sort_page.py's three tabs.

POPs config travels as repeatable query params (?pops_pad=E&pops_pad=S...)
so results stay GET-cacheable and shareable; the SPA owns the selection
state (the old Streamlit session keys well_sort_pops_pads / _force_true).
"""

from __future__ import annotations

from datetime import date
from typing import Any, Optional

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse, Response

from server import schemas
from server.services import well_sort as service
from woffl.assembly.well_sort_engine import DEFAULT_POPS_PADS

router = APIRouter(prefix="/well-sort", tags=["well-sort"])

_XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _pops(pops_pad: Optional[list[str]]) -> list[str]:
    """None = caller sent nothing -> field defaults; [] must stay empty
    (an engineer deliberately clearing every POPs pad is a valid state,
    signalled by pops_pad=)."""
    if pops_pad is None:
        return list(DEFAULT_POPS_PADS)
    return [p for p in pops_pad if p]


@router.get("/tables", response_model=schemas.WellSortTablesResponse)
def tables(
    mode: str = Query("allocated", pattern="^(allocated|any)$", description="Display-test pick"),
    stale_days: int = Query(60, ge=14, le=180),
    pops_pad: Optional[list[str]] = Query(None, description="Pads with on-pad separation"),
    force_true: list[str] = Query([], description="Per-well PopsPad=True overrides"),
) -> Any:
    """Online / Offline / LTSI tables + field context in one round trip."""
    return service.tables_payload(mode, stale_days, _pops(pops_pad), force_true)


@router.get("/events", response_model=schemas.WellSortEventsResponse)
def events(
    window_days: int = Query(30, ge=7, le=60),
    down_hours: float = Query(8.0, ge=1.0, le=24.0),
) -> Any:
    """Shut-in events overlapping the window (the 30-Day Changes view)."""
    return service.events_payload(window_days, down_hours)


@router.get("/marginal-wc", response_model=schemas.MarginalWcResponse)
def marginal_wc(
    threshold_pct: float = Query(2.0, ge=0.0, le=150.0),
    stale_days: int = Query(60, ge=14, le=180),
    pops_pad: Optional[list[str]] = Query(None),
    force_true: list[str] = Query([]),
) -> Any:
    """Field-wide marginal WC via the cumulative-water threshold walk."""
    payload = service.marginal_payload(threshold_pct, stale_days, _pops(pops_pad), force_true)
    if payload is None:
        return JSONResponse(
            status_code=404,
            content={
                "error": "no_data",
                "message": "No online non-POPs wells with valid TotalWC / TotalWater.",
            },
        )
    return payload


@router.get("/pad-marginal-wc", response_model=schemas.PadMarginalWcResponse)
def pad_marginal_wc(
    pad: str = Query(..., min_length=1, max_length=8),
    pump_limit: float = Query(0.0, ge=0.0, le=200_000.0),
    stale_days: int = Query(60, ge=14, le=180),
    pops_pad: Optional[list[str]] = Query(None),
    force_true: list[str] = Query([]),
) -> Any:
    """Per-pad marginal WC + pump-limit headroom (POPs pads)."""
    payload = service.pad_marginal_payload(
        pad, pump_limit, stale_days, _pops(pops_pad), force_true
    )
    if payload is None:
        return JSONResponse(
            status_code=404,
            content={
                "error": "no_data",
                "message": f"No online wells on {pad}-Pad with a usable water/oil pair.",
            },
        )
    return payload


@router.get("/triage", response_model=schemas.TriageResponse)
def triage(
    threshold_pct: float = Query(2.0, ge=0.0, le=100.0),
    stale_days: int = Query(60, ge=14, le=180),
    pops_pad: Optional[list[str]] = Query(None),
    force_true: list[str] = Query([]),
) -> Any:
    """Keep / SI / BOL decisions for every well vs the field marginal WC."""
    payload = service.triage_payload(threshold_pct, stale_days, _pops(pops_pad), force_true)
    if payload is None:
        return JSONResponse(
            status_code=404,
            content={
                "error": "no_data",
                "message": "Can't compute the field marginal WC (no online non-POPs wells with valid water data).",
            },
        )
    return payload


@router.get("/bench.xlsx")
def bench(
    mode: str = Query("allocated", pattern="^(allocated|any)$"),
    stale_days: int = Query(60, ge=14, le=180),
    pops_pad: Optional[list[str]] = Query(None),
    force_true: list[str] = Query([]),
) -> Response:
    """3-sheet MPU_Well_Bench workbook (online / offline / ltsi)."""
    data = service.bench_xlsx(mode, stale_days, _pops(pops_pad), force_true)
    filename = f"MPU_Well_Bench_{date.today():%Y_%m_%d}.xlsx"
    return Response(
        content=data,
        media_type=_XLSX_MIME,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/refresh", response_model=schemas.WellSortRefreshResponse)
def refresh() -> Any:
    """Clear the Well Sort fetch caches (the page's Refresh button).

    Read-only cache invalidation - forces the next request to re-query
    Databricks; no data is written anywhere.
    """
    return {"cleared": service.refresh()}
