"""Small WC-only scenario envelope using the same solve as the workbench.

The total-liquid IPR anchor and GOR stay fixed. Each scenario builds fresh
mutable PVT objects via solve_single. These sampled extrema are not statistical
confidence limits; failed samples remain visible and make the range incomplete.
"""

from __future__ import annotations

import math

from server import schemas
from server.services import solve


def run(req: schemas.WcUncertaintyRequest) -> schemas.WcUncertaintyResponse:
    """Evaluate up to nine WC scenarios, including the exact current input.

    Args:
        req: Well inputs and symmetric uncertainty in WC percentage points.

    Returns:
        Oil (BOPD) and suction BHP (psig) extrema over successful samples.
    """
    sp = req.params
    if sp.model_as_water or not 0 <= sp.form_wc <= .99:
        raise ValueError("WC uncertainty requires oil mode and formation WC from 0% to 99%.")
    width = req.uncertainty_points / 100.0
    raw_low, raw_high = sp.form_wc - width, sp.form_wc + width
    low, high = max(0.0, raw_low), min(.99, raw_high)
    # Include interior points: nonlinear oil/BHP extrema need not occur at
    # the endpoints. Deduplicate clipped endpoints and the zero-width case.
    cuts = sorted({min(.99, max(0.0, sp.form_wc + width * i / 4)) for i in range(-4, 5)})
    points = []
    for wc in cuts:
        try:
            candidate = sp.model_copy(deep=True, update={"form_wc": wc})
            result = solve.solve_single(req.well, candidate)
            if not all(math.isfinite(result[k]) for k in ("psu", "qoil_std", "fwat_bwpd", "qnz_bwpd")):
                raise ValueError("No finite prediction at this watercut.")
            points.append(schemas.WcUncertaintyPoint(wc=wc, oil=result["qoil_std"], bhp=result["psu"]))
        except (solve.SolveFailure, ValueError) as exc:
            points.append(schemas.WcUncertaintyPoint(wc=wc, error=str(exc) or "No solution at this watercut."))
    good = [p for p in points if p.error is None]
    base = next(p for p in points if p.wc == sp.form_wc)

    def bounds(metric: str):
        values = [getattr(p, metric) for p in good]
        return schemas.WcMetricRange(low=min(values), base=getattr(base, metric), high=max(values)) if values else None

    return schemas.WcUncertaintyResponse(
        well=req.well, uncertainty_points=req.uncertainty_points,
        wc_base=sp.form_wc, wc_low=low, wc_high=high,
        clipped=raw_low < 0 or raw_high > .99,
        complete=len(good) == len(points), base_solved=base.error is None,
        sample_count=len(points), solved_count=len(good),
        oil=bounds("oil"), bhp=bounds("bhp"), points=points,
    )
