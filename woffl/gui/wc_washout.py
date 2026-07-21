"""Water-cut washout detection + suggestion for low-rate jet pump wells.

Field lesson (MPE-19, 2026-07): on a low-rate well the power-fluid return
dominates the test separator stream — the TRUE formation water is less than
~10% of the PF volume — so the rate allocation nets formation water to ~0 and
the well test reports 0% water cut. Modeling with that WC makes BHP and oil
rate unmatchable (the suction mixture is far too light and dry); raising the
sidebar WC from 0% to 65% matched both on MPE-19.

Two pieces, both pure (no Streamlit) so they're unit-testable; rendering
lives in ``tabs/jetpump_solver.py``:

* :func:`detect_wc_washout` — flags the suspicious setup: near-zero WC on a
  well whose produced-fluid stream is small relative to the PF stream, while
  the model is having a hard time matching BHP and/or oil.
* :func:`suggest_water_cut` — sweeps WC through a caller-supplied solver
  closure and returns the WC that best matches the measured BHP + oil, as a
  STARTING POINT for the engineer (mirrors exactly what turning the sidebar
  Water Cut knob does; it never mutates the model itself).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

# ---------------------------------------------------------------------------
# Detection thresholds
# ---------------------------------------------------------------------------

# Sidebar/test WC at or below this is "reads as zero" — allocation noise, not
# a measured dry well.
WC_NEAR_ZERO = 0.05

# Washout exposure: reported produced fluid (total fluid, or oil when the
# test nets water to zero) divided by the PF rate. Below this the separator
# stream is PF-dominated and the allocation can't resolve formation water.
# MPE-19: 151 BOPD / 2,774 BWPD PF = 0.054. Even at its true 65% WC the
# formation water (~280 BWPD) is only ~10% of PF.
WASHOUT_FLUID_TO_PF_MAX = 0.15

# "Hard time matching" — either trips the flag.
BHP_MISMATCH_PSI = 100.0
OIL_MISMATCH_FRAC = 0.15

# ---------------------------------------------------------------------------
# Suggestion sweep bounds
# ---------------------------------------------------------------------------

SWEEP_LO = 0.05
SWEEP_HI = 0.90
SWEEP_COARSE_STEP = 0.05
SWEEP_FINE_STEP = 0.01
SWEEP_FINE_SPAN = 0.04
MIN_VALID_POINTS = 3


@dataclass
class WashoutFlag:
    """Why the near-zero WC looks like allocation washout, with the numbers
    the warning text quotes."""

    form_wc: float
    fluid_to_pf_ratio: float
    pf_rate: float
    bhp_delta: Optional[float]  # modeled psu - measured BHP (psi), if BHP known
    oil_delta_frac: Optional[float]  # (modeled - actual) / actual, if oil known


def detect_wc_washout(
    *,
    form_wc: float,
    pf_rate: Optional[float],
    produced_fluid: Optional[float],
    modeled_psu: Optional[float] = None,
    actual_bhp: Optional[float] = None,
    modeled_oil: Optional[float] = None,
    actual_oil: Optional[float] = None,
) -> Optional[WashoutFlag]:
    """Flag a near-zero WC that is probably allocation washout, not reality.

    Fires only when ALL three hold:
      1. the modeled WC is near zero (``form_wc <= WC_NEAR_ZERO``),
      2. the produced-fluid stream is small next to the PF stream
         (``produced_fluid / pf_rate <= WASHOUT_FLUID_TO_PF_MAX``), and
      3. the model is struggling: |modeled psu - measured BHP| >=
         BHP_MISMATCH_PSI, or the oil-rate error >= OIL_MISMATCH_FRAC.
         (With neither comparison available there is no evidence of a
         matching problem, so no flag.)

    ``produced_fluid`` should be the test's total produced fluid (oil + net
    water); on a washed-out test that's effectively the oil rate.
    Returns ``None`` when the setup doesn't fit.
    """
    if form_wc is None or form_wc > WC_NEAR_ZERO:
        return None
    if pf_rate is None or pf_rate <= 0:
        return None
    if produced_fluid is None or produced_fluid <= 0:
        return None

    ratio = float(produced_fluid) / float(pf_rate)
    if ratio > WASHOUT_FLUID_TO_PF_MAX:
        return None

    bhp_delta: Optional[float] = None
    if modeled_psu is not None and actual_bhp is not None and actual_bhp > 0:
        bhp_delta = float(modeled_psu) - float(actual_bhp)

    oil_delta_frac: Optional[float] = None
    if modeled_oil is not None and actual_oil is not None and actual_oil > 0:
        oil_delta_frac = (float(modeled_oil) - float(actual_oil)) / float(actual_oil)

    mismatch = (bhp_delta is not None and abs(bhp_delta) >= BHP_MISMATCH_PSI) or (
        oil_delta_frac is not None and abs(oil_delta_frac) >= OIL_MISMATCH_FRAC
    )
    if not mismatch:
        return None

    return WashoutFlag(
        form_wc=float(form_wc),
        fluid_to_pf_ratio=ratio,
        pf_rate=float(pf_rate),
        bhp_delta=bhp_delta,
        oil_delta_frac=oil_delta_frac,
    )


@dataclass
class WcSweepPoint:
    wc: float
    err: float
    psu: float
    qoil: float
    qnz: float


@dataclass
class WcSuggestion:
    """Result of the WC sweep — a starting point, not an answer.

    ``matched_on`` records what the objective could see ("BHP + oil" or
    "oil only"); an oil-only match is much weaker evidence. ``bounded``
    means the best WC sits at the sweep edge — treat with suspicion.
    """

    suggested_wc: float
    base_wc: float
    target_oil: float
    target_bhp: Optional[float]
    modeled_psu: float
    modeled_oil: float
    modeled_pf: float
    matched_on: str
    bounded: bool
    n_solved: int
    n_failed: int
    points: list[WcSweepPoint] = field(default_factory=list)


def suggest_water_cut(
    solve_at_wc: Callable[[float], Optional[tuple]],
    *,
    target_oil: float,
    target_bhp: Optional[float] = None,
    base_wc: float = 0.0,
) -> Optional[WcSuggestion]:
    """Sweep WC through the solver; return the best BHP+oil match.

    ``solve_at_wc(wc)`` must mirror the Solver hero exactly except for the
    ResMix water cut — returning the hero tuple
    ``(psu, sonic, qoil_std, fwat_bwpd, qnz_bwpd, mach_te)`` or ``None``
    for a failed solve (failed points are skipped, not fatal).

    Two passes: a coarse grid over [SWEEP_LO, SWEEP_HI], then a fine grid
    around the coarse winner. Objective = |Δoil|/oil + |Δpsu|/BHP (the BHP
    term drops out when no measured BHP exists — weaker, flagged via
    ``matched_on``). Returns ``None`` when fewer than MIN_VALID_POINTS
    points solve, or the targets are unusable.
    """
    if target_oil is None or target_oil <= 0:
        return None
    if target_bhp is not None and target_bhp <= 0:
        target_bhp = None

    def _err(res: tuple) -> float:
        psu, _sonic, qoil, _fwat, _qnz, _mach = res
        e = abs(float(qoil) - target_oil) / target_oil
        if target_bhp is not None:
            e += abs(float(psu) - target_bhp) / target_bhp
        return e

    points: dict[float, WcSweepPoint] = {}
    n_failed = 0

    def _eval(wc: float, allow_out_of_range: bool = False) -> None:
        nonlocal n_failed
        wc = round(float(wc), 2)
        if wc in points or not (0.0 < wc < 1.0):
            return
        # Keep every grid point inside the advertised sweep window — the fine
        # pass around a boundary winner must not creep past SWEEP_HI into the
        # numerically-dicey near-100% zone. base_wc alone may sit outside.
        if not allow_out_of_range and not (
            SWEEP_LO - 1e-9 <= wc <= SWEEP_HI + 1e-9
        ):
            return
        res = solve_at_wc(wc)
        if res is None:
            n_failed += 1
            return
        psu, _sonic, qoil, _fwat, qnz, _mach = res
        points[wc] = WcSweepPoint(
            wc=wc, err=_err(res), psu=float(psu), qoil=float(qoil), qnz=float(qnz)
        )

    # Coarse pass (include the current WC so the result can say "better than
    # where you are now", not just "best of the grid").
    wc = SWEEP_LO
    while wc <= SWEEP_HI + 1e-9:
        _eval(wc)
        wc += SWEEP_COARSE_STEP
    if 0.0 < base_wc < 1.0:
        _eval(base_wc, allow_out_of_range=True)

    if len(points) < MIN_VALID_POINTS:
        return None

    best = min(points.values(), key=lambda p: p.err)

    # Fine pass around the coarse winner.
    wc = best.wc - SWEEP_FINE_SPAN
    while wc <= best.wc + SWEEP_FINE_SPAN + 1e-9:
        _eval(wc)
        wc += SWEEP_FINE_STEP
    best = min(points.values(), key=lambda p: p.err)

    bounded = best.wc <= SWEEP_LO + 1e-9 or best.wc >= SWEEP_HI - 1e-9
    return WcSuggestion(
        suggested_wc=best.wc,
        base_wc=float(base_wc),
        target_oil=float(target_oil),
        target_bhp=float(target_bhp) if target_bhp is not None else None,
        modeled_psu=best.psu,
        modeled_oil=best.qoil,
        modeled_pf=best.qnz,
        matched_on="BHP + oil" if target_bhp is not None else "oil only",
        bounded=bounded,
        n_solved=len(points),
        n_failed=n_failed,
        points=sorted(points.values(), key=lambda p: p.wc),
    )
