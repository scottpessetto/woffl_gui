"""Scott's Tools - the secret-menu engineering tools.

Ported from the Streamlit `scotts_tools` package. Every endpoint here is
READ-ONLY: these tools model, scan and compare, and none of them writes. JP
Calibration returns SQL as a PREVIEW for a human to run; it does not execute
it and never has.

"Hidden" is a UI affordance, not authorization - the React app keeps the menu
behind a typed word / the /scott route (web/src/lib/secretMenu.ts) exactly as
the Streamlit app did, and these routes are reachable by anyone who knows the
path. Do not put anything behind this prefix that needs a real permission
check.

Long tools run as background jobs (server.jobs), the same pattern the
optimizer endpoints use: a fleet scan or a header model is tens of seconds of
solving, which is not something to hold a socket open for.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from server import jobs, schemas
from server.services.tools import pad_watercut as pad_wc_svc

router = APIRouter(prefix="/tools", tags=["scotts-tools"])

# Job kinds this router owns; keeps one registry from leaking ids across
# endpoint namespaces.
_KINDS = (
    "tool_harness",
    "tool_washout",
    "tool_fric_trend",
    "tool_calibration",
    "tool_header_impact",
    "tool_pf_scenario",
)


def _invalid(exc: Exception) -> HTTPException:
    return HTTPException(status_code=422, detail={"error": "invalid", "message": str(exc)})


@router.get("/catalog", response_model=schemas.ToolCatalogResponse)
def catalog() -> Any:
    """Which tools this build actually serves.

    The React menu renders from this rather than a hardcoded list, so a tool
    that has not been ported cannot show up as a dead link.
    """
    return {"tools": schemas.TOOL_CATALOG}


@router.get("/job/{job_id}", response_model=schemas.ToolJobStatus)
def job_status(job_id: str) -> Any:
    """Poll any tool job. One envelope for all of them."""
    job = jobs.get(job_id, kinds=_KINDS)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "invalid", "message": "unknown or expired job"},
        )
    return job


# ── Pad Water Cut (fast; no job needed) ────────────────────────────────────


@router.get("/pad-watercut", response_model=schemas.PadWatercutResponse)
def pad_watercut(
    start: str = Query(..., description="YYYY-MM-DD inclusive"),
    end: str = Query(..., description="YYYY-MM-DD inclusive"),
) -> Any:
    """Daily pad-level water cut for pads G/H/I/J plus the combined series."""
    try:
        return pad_wc_svc.pad_watercut(start, end)
    except ValueError as exc:
        raise _invalid(exc) from None


@router.get("/pad-watercut/default-window", response_model=schemas.DateWindow)
def pad_watercut_window() -> Any:
    """The tab's default range (three years back to today)."""
    start, end = pad_wc_svc.default_window()
    return {"start": start, "end": end}


# ── Test Harness ───────────────────────────────────────────────────────────


@router.get("/harness/cases", response_model=schemas.HarnessCasesResponse)
def harness_cases() -> Any:
    """Registered cases, without running any."""
    from server.services.tools import harness

    return harness.list_cases()


@router.post("/harness/run", response_model=schemas.ToolJobStarted)
def harness_run() -> Any:
    """Run every case against today's data. ~1 min, so it is a job."""
    from server.services.tools import harness

    job_id = jobs.start(
        "tool_harness", lambda job: harness.run_all(), progress="running cases..."
    )
    return {"job_id": job_id}


# ── JP Wash-Out ────────────────────────────────────────────────────────────


@router.post("/washout/scan", response_model=schemas.ToolJobStarted)
def washout_scan(req: schemas.WashoutRequest) -> Any:
    """Scan the fleet for pumps that cannot make their measured lift water."""
    from server.services.tools import jp_washout

    job_id = jobs.start(
        "tool_washout",
        lambda job: jp_washout.scan(req.months_back, req.ppf_limit),
        progress="calibrating PF pressure per well...",
    )
    return {"job_id": job_id}


# ── JP Friction Trend ──────────────────────────────────────────────────────


@router.post("/fric-trend/run", response_model=schemas.ToolJobStarted)
def fric_trend_run(req: schemas.FricTrendRequest) -> Any:
    """Fit friction coefficients across each selected well's test history."""
    from server.services.tools import runs

    if not req.wells:
        raise _invalid(ValueError("Select at least one well."))
    job_id = jobs.start(
        "tool_fric_trend",
        lambda job: runs.fric_trend(list(req.wells), req.months_back),
        progress="calibrating tests...",
    )
    return {"job_id": job_id}


# ── JP Friction Calibration ────────────────────────────────────────────────


@router.get("/calibration/inputs", response_model=schemas.ToolRowsResponse)
def calibration_inputs(months_back: int = Query(6, ge=1, le=60)) -> Any:
    """The per-well calibration input table (no solving)."""
    from server.services.tools import runs

    return runs.calibration_inputs(months_back)


@router.post("/calibration/run", response_model=schemas.ToolJobStarted)
def calibration_run(req: schemas.CalibrationRequest) -> Any:
    """Fit ken/kth/kdi per well against measured BHP. Returns SQL to REVIEW."""
    from server.services.tools import runs

    job_id = jobs.start(
        "tool_calibration",
        lambda job: runs.run_calibration(list(req.wells or []), req.months_back),
        progress="calibrating friction coefficients...",
    )
    return {"job_id": job_id}


# ── Header Pressure Impact ─────────────────────────────────────────────────


@router.get("/header-impact/inputs", response_model=schemas.ToolRowsResponse)
def header_impact_inputs(
    pads: list[str] = Query(..., description="Pad letters"),
    months_back: int = Query(6, ge=1, le=60),
) -> Any:
    """The per-well input table for the selected pads, all lift types."""
    from server.services.tools import runs

    return runs.header_impact_inputs(list(pads), months_back)


@router.post("/header-impact/run", response_model=schemas.ToolJobStarted)
def header_impact_run(req: schemas.HeaderImpactRequest) -> Any:
    """Model a header pressure change across every producer on the pads."""
    from server.services.tools import runs

    if not req.pads:
        raise _invalid(ValueError("Select at least one pad."))
    job_id = jobs.start(
        "tool_header_impact",
        lambda job: runs.header_impact(
            list(req.pads), req.delta_p, req.months_back, req.pad_pf or None
        ),
        progress="solving wells at the new header pressure...",
    )
    return {"job_id": job_id}


# ── PF Scenario ────────────────────────────────────────────────────────────


@router.post("/pf-scenario/run", response_model=schemas.ToolJobStarted)
def pf_scenario_run(req: schemas.PfScenarioRequest) -> Any:
    """Compare two power-fluid pressures across the selected wells."""
    from server.services.tools import runs

    if not req.wells:
        raise _invalid(ValueError("Select at least one well."))
    job_id = jobs.start(
        "tool_pf_scenario",
        lambda job: runs.pf_scenario(
            list(req.wells), req.pf_a, req.pf_b, req.months_back
        ),
        progress="solving both scenarios...",
    )
    return {"job_id": job_id}
