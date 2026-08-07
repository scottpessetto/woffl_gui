"""Compute endpoints - solve, batch, PF-range, pressure profile, IPR,
knob sensitivity.

Error contract (SolveErrorDetail): every failure is an HTTPException 422
whose detail is {"error", "message", "suggested_gor"}. Typed solver failures
come from services.solve.SolveFailure; plain ValueError means bad inputs and
maps to "invalid".

Write endpoints (save-ipr / clear pin): gated on ALLOW_DATABRICKS_WRITES -
403 when off, mirroring the Streamlit pre-check that hides the button.
push_prop re-enforces the gate (and the prop_xref whitelist + the as-built
physical-property rejection) on the actual INSERT, so a race or a stale
client can never write past a closed gate. Every write stamps the acting
engineer via identity.bind_entry_user (X-Forwarded-Email).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from server import schemas
from server.identity import bind_entry_user
from server.services import ipr, sensitivity, solve

router = APIRouter(tags=["compute"])


def _invalid(exc: Exception) -> HTTPException:
    return HTTPException(
        status_code=422,
        detail={"error": "invalid", "message": str(exc), "suggested_gor": None},
    )


def _solver_error(exc: solve.SolveFailure) -> HTTPException:
    return HTTPException(status_code=422, detail=exc.detail())


@router.post("/solve", response_model=schemas.SolveResult)
def post_solve(req: schemas.SolveRequest) -> schemas.SolveResult:
    """Solve one well/pump operating point."""
    try:
        return schemas.SolveResult(**solve.solve_single(req.well, req.params))
    except solve.SolveFailure as exc:
        raise _solver_error(exc) from exc
    except ValueError as exc:
        raise _invalid(exc) from exc


@router.post("/batch", response_model=schemas.BatchResponse)
def post_batch(req: schemas.BatchRequest) -> schemas.BatchResponse:
    """Nozzle x throat batch sweep with recommendation + fit curve."""
    try:
        return schemas.BatchResponse(**solve.run_batch(req.well, req.params))
    except solve.SolveFailure as exc:
        raise _solver_error(exc) from exc
    except ValueError as exc:
        raise _invalid(exc) from exc


@router.post("/pf-range", response_model=schemas.PfRangeResponse)
def post_pf_range(req: schemas.PfRangeRequest) -> schemas.PfRangeResponse:
    """Batch sweep across a range of PF surface pressures."""
    try:
        return schemas.PfRangeResponse(**solve.run_pf_range(req.well, req.params))
    except solve.SolveFailure as exc:
        raise _solver_error(exc) from exc
    except ValueError as exc:
        raise _invalid(exc) from exc


@router.post("/pressure-profile", response_model=schemas.PressureProfileResponse)
def post_pressure_profile(req: schemas.PressureProfileRequest) -> schemas.PressureProfileResponse:
    """Production + PF pressure traverses and their differential."""
    try:
        return schemas.PressureProfileResponse(**solve.pressure_profile(req.well, req.params))
    except solve.SolveFailure as exc:
        raise _solver_error(exc) from exc
    except ValueError as exc:
        raise _invalid(exc) from exc


@router.post("/calibrate", response_model=schemas.CalibrateResponse)
def post_calibrate(req: schemas.CalibrateRequest) -> schemas.CalibrateResponse:
    """BHP friction calibration: fit ken/kth/kdi toward the test's measured
    BHP (read-only compute; nothing persisted - the client applies coefs to
    the sidebar, an explicit save keeps them)."""
    try:
        return schemas.CalibrateResponse(**solve.calibrate(req))
    except solve.SolveFailure as exc:
        raise _solver_error(exc) from exc
    except ValueError as exc:
        raise _invalid(exc) from exc


@router.post("/sensitivity", response_model=schemas.SensitivityResponse)
def post_sensitivity(req: schemas.SensitivityRequest) -> schemas.SensitivityResponse:
    """Per-knob sensitivity of the four match quantities (suction BHP, oil,
    liquid, power fluid) at the current operating point: every calibration
    knob swept across a defensible range, with its signed low/high excursion
    and an inert flag.

    Read-only diagnostic - no gate, no persistence, no physics change. It
    exists to show which knobs are actually live on THIS operating point:
    on a choked well the solver returns the choked-flow floor directly, so
    PF pressure, kth, kdi and wellhead pressure are provably inert and the
    engineer would otherwise turn them blind.
    """
    try:
        return schemas.SensitivityResponse(
            **sensitivity.run_sensitivity(
                req.well,
                req.params,
                {
                    "target_psu": req.target_psu,
                    "target_qoil": req.target_qoil,
                    "target_qliq": req.target_qliq,
                    "target_qpf": req.target_qpf,
                },
                req.bounds,
            )
        )
    except solve.SolveFailure as exc:
        raise _solver_error(exc) from exc
    except ValueError as exc:
        raise _invalid(exc) from exc


@router.post("/sensitivity/combine", response_model=schemas.CombineStarted)
def post_sensitivity_combine(req: schemas.CombineRequest) -> Any:
    """Start a combined-permutations study: vary the selected inputs TOGETHER
    over the engineer's own ranges and report what the combination can reach.

    This is the question a one-at-a-time sweep cannot answer. Where no single
    input closes the gap to the measured test, this solves the full factorial
    of the requested levels and returns the reachable [min, max] per match
    quantity, whether each supplied target falls inside it, and the
    permutation with the lowest RMS fractional error.

    The sizes that matter are minutes of serial solving (8 inputs at 3 levels
    is 6,561 solves), so it runs as a background job: this returns a job id
    and GET /sensitivity/combine/{job_id} polls status, progress and result.
    The request-level refusals still happen HERE, synchronously, so a typo is
    an immediate 422 rather than a job that fails a second later.

    The run cap (sensitivity.MAX_COMBINE_RUNS) is one of them, deliberately
    not a silent sample: a subset of a factorial would misrepresent what was
    explored, and the engineer reads "no combination reaches this" off the
    envelope.

    Read-only diagnostic - no gate, no persistence, no physics change.
    """
    try:
        return {"job_id": sensitivity.start_combine(req)}
    except ValueError as exc:
        raise _invalid(exc) from exc


@router.get("/sensitivity/combine/{job_id}", response_model=schemas.CombineJobStatus)
def get_sensitivity_combine(job_id: str) -> Any:
    """Combine study status; `result` populates when status becomes done.

    Jobs live in process memory and are pruned an hour after they settle, so
    an id from before a restart is a 404 and the client drops it.
    """
    job = sensitivity.get_combine_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "invalid", "message": f"unknown or expired job {job_id}"},
        )
    return job


@router.post("/ipr/fit", response_model=schemas.IprFitResponse)
def post_ipr_fit(req: schemas.IprFitRequest) -> schemas.IprFitResponse:
    """Vogel IPR fit for one well (recent / median / specific anchor)."""
    try:
        return schemas.IprFitResponse(**ipr.fit(req))
    except ValueError as exc:
        raise _invalid(exc) from exc


@router.get("/wells/{name}/ipr-pin", response_model=schemas.IprPinResponse)
def get_ipr_pin(name: str) -> schemas.IprPinResponse:
    """Saved IPR-anchor pin status for one well (fail-soft: none)."""
    return schemas.IprPinResponse(**ipr.pin(name))


def _writes_gate() -> None:
    """403 when the ALLOW_DATABRICKS_WRITES gate is off - the UI hides the
    save controls on /meta.writes_enabled, so hitting this means a stale
    client or a deliberate probe."""
    from woffl.gui.ipr_anchor import writes_enabled

    if not writes_enabled():
        raise HTTPException(
            status_code=403,
            detail={
                "error": "writes_disabled",
                "message": "Saving requires ALLOW_DATABRICKS_WRITES=true in the app environment.",
            },
        )


@router.post("/wells/{name}/save-ipr", response_model=schemas.SaveIprResponse)
def post_save_ipr(name: str, req: schemas.SaveIprRequest, request: Request) -> schemas.SaveIprResponse:
    """Pin the resolved anchor test AND save the sidebar's current IPR/fluid
    values as the well's defaults (mpu.wells.prop_hist, append-only)."""
    _writes_gate()
    bind_entry_user(request)
    return schemas.SaveIprResponse(**ipr.save(name, req))


@router.delete("/wells/{name}/ipr-pin", response_model=schemas.ClearIprPinResponse)
def delete_ipr_pin(name: str, request: Request) -> schemas.ClearIprPinResponse:
    """Clear the saved IPR default (appends the cleared-marker row - prop_hist
    is append-only, nothing is ever deleted)."""
    _writes_gate()
    bind_entry_user(request)
    return schemas.ClearIprPinResponse(**ipr.clear_pin(name))


@router.post("/wells/{name}/prop-lock", response_model=schemas.PropLockResponse)
def post_prop_lock(name: str, req: schemas.PropLockRequest, request: Request) -> schemas.PropLockResponse:
    """Toggle a WC/GOR/ResP field lock; locking pins the sent value in the
    same click (mpu.wells.prop_hist, append-only)."""
    _writes_gate()
    bind_entry_user(request)
    return schemas.PropLockResponse(**ipr.set_lock(name, req))
