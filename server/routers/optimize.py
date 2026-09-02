"""Optimization pad board (read-only).

The redesigned optimization workflow: engineers match and save well fits on
the Single Well solver; this surface reports per-pad READINESS - which
wells have a saved fit, when, by whom, and what is missing. Offline flags
and future wells live client-side (the engineer's working config), so the
server stays stateless here.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from fastapi import APIRouter, HTTPException, Query

from server import schemas
from server.services import (
    e_pad_curves,
    event_calibration,
    ipr,
    match_health,
    optimizer_runs,
    pad_curves,
)

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


@router.post("/match-health", response_model=schemas.OptimizeRunStarted)
def start_match_health(req: schemas.MatchHealthRequest) -> Any:
    """Start a match-health scorecard job for one pad: every active well
    modeled at its CURRENT pump vs its recent tests, plus fit provenance,
    field-evidence floors/betas and friction-rail flags, one verdict chip
    per well. Read-only compute; poll GET /optimize/run/{job_id}."""
    return {"job_id": match_health.start_match_health(req.pad)}


@router.post("/event-calibration", response_model=schemas.OptimizeRunStarted)
def start_event_calibration(req: schemas.EventCalibrationRequest) -> Any:
    """Start a multi-point event-calibration job for one well: hydrate its
    saved fit, gather every measured operating point in the current pump
    era and fit (ken, kth, kdi, fnz, mach_crit) against all of them at
    once. Read-only compute; poll GET /optimize/run/{job_id}."""
    return {"job_id": event_calibration.start_event_calibration(req.well)}


@router.get("/run/{job_id}", response_model=schemas.OptimizeJobStatus)
def run_status(job_id: str) -> Any:
    """Job status; `result` populates when status becomes done."""
    job = (
        optimizer_runs.get_job(job_id)
        or match_health.get_job(job_id)
        or event_calibration.get_job(job_id)
    )
    if job is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "invalid", "message": f"unknown or expired job {job_id}"},
        )
    return job


@router.get("/pump-curve", response_model=schemas.PumpCurveResponse)
def pump_curve(
    pad: Literal["S", "I", "M", "E"] = Query(...),
    n_pumps: Optional[int] = Query(None, ge=1, le=3),
    build: Optional[Literal["SM25000_26STG", "SN35000_18STG"]] = Query(None),
    suction_psi: Optional[float] = Query(None, ge=0.0, le=5000.0),
    hz_max: Optional[float] = Query(None, ge=30.0, le=60.0),
    max_header_psi: Optional[float] = Query(None, ge=1000.0, le=5000.0),
) -> Any:
    """Industry-format booster-pump curves for one pad's plant: the station
    family of delivered header pressure vs total flow plus each machine's
    head / BHP / efficiency curve, with BEP, the preferred and allowable
    operating regions and the capability frontier.

    Read-only static physics off the plant model and its data files - no run
    state - so it renders before a run and the engineer sees plant capability
    while configuring. n_pumps defaults to the plant's own.

    The four E-Pad knobs exist because none of them is a measured E-Pad tag
    (see server.schemas.OptimizeRunRequest) and the curve sheet beside a run
    form must show the booster the run will actually assume. They are rejected
    on the other pads rather than silently ignored.
    """
    e_knobs = {
        "build": build,
        "suction_psi": suction_psi,
        "hz_max": hz_max,
        "max_header_psi": max_header_psi,
    }
    passed = [k for k, v in e_knobs.items() if v is not None]
    if pad != "E" and passed:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid",
                "message": (
                    f"{', '.join(passed)} configure the E-Pad booster only; "
                    f"{pad}-Pad's plant is fixed vendor data"
                ),
            },
        )
    try:
        return pad_curves.pump_curve(pad, n_pumps, **e_knobs)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={"error": "invalid", "message": str(exc)},
        ) from exc


@router.post("/e-pad-booster", response_model=schemas.EPadBoosterResponse)
def e_pad_booster(req: schemas.EPadBoosterRequest) -> Any:
    """What each E-Pad booster candidate can deliver at a required dP: the
    flow window inside the vendor recommended operating range, the speed and
    amps it takes, and what caps the top of the window.

    The installed SM25000 26-stage build against the SN35000 18-stage
    alternative. Read-only static physics off the catalog stage curves - no
    run state, no Databricks - so POST here is a pure computation, not a
    write; the body only exists because the duty (dP, suction, SG, wear, speed
    cap, amps/BHP, motor amp limit) is seven scalars an engineer sweeps.
    """
    return e_pad_curves.capability(
        req.dp_psid,
        req.suction_psi,
        req.sg,
        req.condition,
        req.hz_max,
        req.amps_per_bhp,
        req.amp_limit_a,
    )
