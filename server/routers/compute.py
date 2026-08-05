"""Compute endpoints - solve, batch, PF-range, pressure profile, IPR.

Error contract (SolveErrorDetail): every failure is an HTTPException 422
whose detail is {"error", "message", "suggested_gor"}. Typed solver failures
come from services.solve.SolveFailure; plain ValueError means bad inputs and
maps to "invalid".
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from server import schemas
from server.services import ipr, solve

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
