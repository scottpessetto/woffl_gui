"""Power-fluid surface-pressure calibration — find ppf_surf needed to match observed lift_wat.

Inverts ``jetpump_solver``: given the observed nozzle/power-fluid rate
(``lift_wat`` from the well-test view), binary-search the power-fluid
surface pressure that makes the solver's modeled ``qnz`` match.

Used by:
  - JP Wash-Out Detection — flag wells whose required ppf_surf exceeds the
    surface infrastructure cap (~3400 psi).
  - JP Friction Trend Analysis — establish per-test ppf_surf before feeding
    the friction-coef calibrator.

Why lift_wat as the rate target: nozzle flow ``qnz`` is dominated by
``ppf_surf`` and nozzle area (Bernoulli), with minimal coupling to
ken/kth/kdi. This makes the search robust even when the well's friction
coefficients have drifted — which is exactly the property a wash-out
detector needs.

The search is structurally identical to ``pf_scenario._estimate_bhp``
(binary search with bound-handling); the axis is just swapped from psu to
ppf_surf.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from woffl.assembly.solopump import jetpump_solver
from woffl.geometry.jetpump import JetPump

PPF_LO_DEFAULT = 1000.0
PPF_HI_DEFAULT = 5000.0
TOL_PSI = 5.0
MAX_ITER = 30


def robust_bracket(
    f,
    lo: float,
    hi: float,
    *,
    lo_floor: float,
    hi_cap: float,
    expand_step: float = 500.0,
    max_iter: int = 16,
) -> dict:
    """Bracket a sign change of a (monotonic-ish) residual, robust to NaNs.

    ``f(x)`` returns a float, or NaN when the underlying solver fails at ``x``.
    The PF residual increases with ppf_surf (more PF pressure → more nozzle
    flow), so the search:

      1. Gets finite evaluations at the starting bounds, nudging a NaN bound
         *inward* toward the midpoint until the solver converges.
      2. Returns immediately if the start bounds already bracket a sign change.
      3. Otherwise expands the same-sign side *outward* toward its cap
         (``hi`` up when both residuals are negative, ``lo`` down when both are
         positive), re-evaluating until a sign change appears or a cap/NaN wall
         is hit.

    Returns ``{"status", "a", "b", "fa", "fb"}`` where status is:
      * ``"ok"``         — ``(a, b)`` brackets a sign change (finite, opposite
        signs); ``a == b`` marks an exact hit on a bound.
      * ``"nan"``        — couldn't get finite evaluations at usable bounds.
      * ``"no_bracket"`` — same sign even after expanding to the caps; inspect
        ``fa``/``fb`` to message under- vs over-shoot.
    """

    def feval(x: float) -> float:
        try:
            return float(f(x))
        except Exception:
            return float("nan")

    def finite_bound(x: float, mid: float) -> tuple[float, float]:
        v = feval(x)
        for _ in range(8):
            if not np.isnan(v):
                return x, v
            x = x + (mid - x) * 0.4
            if abs(mid - x) < 1.0:
                break
            v = feval(x)
        return x, v

    mid = (lo + hi) / 2.0
    lo, f_lo = finite_bound(lo, mid)
    hi, f_hi = finite_bound(hi, mid)

    if np.isnan(f_lo) or np.isnan(f_hi):
        return {"status": "nan", "a": lo, "b": hi, "fa": f_lo, "fb": f_hi}
    if f_lo == 0.0:
        return {"status": "ok", "a": lo, "b": lo, "fa": f_lo, "fb": f_lo}
    if f_hi == 0.0:
        return {"status": "ok", "a": hi, "b": hi, "fa": f_hi, "fb": f_hi}
    if f_lo * f_hi < 0:
        return {"status": "ok", "a": lo, "b": hi, "fa": f_lo, "fb": f_hi}

    for _ in range(max_iter):
        expanded = False
        if f_hi < 0 and hi < hi_cap:
            new_hi = min(hi_cap, hi + expand_step)
            v = feval(new_hi)
            if not np.isnan(v):
                hi, f_hi, expanded = new_hi, v, True
        elif f_lo > 0 and lo > lo_floor:
            new_lo = max(lo_floor, lo - expand_step)
            v = feval(new_lo)
            if not np.isnan(v):
                lo, f_lo, expanded = new_lo, v, True
        if f_lo * f_hi < 0:
            return {"status": "ok", "a": lo, "b": hi, "fa": f_lo, "fb": f_hi}
        if not expanded:
            break  # hit a cap or a NaN wall — can't widen further

    return {"status": "no_bracket", "a": lo, "b": hi, "fa": f_lo, "fb": f_hi}


@dataclass
class PfCalibrationResult:
    well_name: str
    target_lift: float       # observed lift_wat (BWPD)
    ppf_surf: float          # binary-search result (psi)
    modeled_qnz: float       # modeled nozzle PF rate at ppf_surf (BWPD)
    modeled_qoil: float      # diagnostic: oil rate at ppf_surf (STBOPD)
    modeled_bhp: float       # diagnostic: suction pressure at ppf_surf (psi)
    lift_residual: float     # modeled_qnz − target_lift; sign hints under/over
    converged: bool          # True iff bracket closed within tol AND no bound hit
    bounded: bool            # True iff search hit ppf_lo or ppf_hi limit
    sonic: bool              # solver reports sonic at the returned ppf_surf
    iterations: int


def _solve_with_pump(
    ppf_surf: float,
    jp: JetPump,
    *,
    pwh: float,
    tsu: float,
    wellbore,
    wellprof,
    ipr_su,
    prop_su,
    prop_pf,
    jpump_direction: str,
) -> tuple[float, float, float, bool, bool]:
    """Solve the jet pump once with a prebuilt JetPump object.

    Returns (qnz, qoil, psu, sonic, ok). All-NaN with ok=False on solver
    failure. Mirrors fric_calibration._solve_at_coefs's defensive wrap.
    Taking the pump object (rather than nozzle/throat strings) lets the
    nozzle-wear estimator pass a pump with a scaled effective nozzle.
    """
    try:
        psu, sonic, qoil, _fwat, qnz, _mach = jetpump_solver(
            pwh=pwh,
            tsu=tsu,
            ppf_surf=ppf_surf,
            jpump=jp,
            wellbore=wellbore,
            wellprof=wellprof,
            ipr_su=ipr_su,
            prop_su=prop_su,
            prop_pf=prop_pf,
            jpump_direction=jpump_direction,
        )
        if psu is None or np.isnan(psu):
            return np.nan, np.nan, np.nan, False, False
        return float(qnz), float(qoil), float(psu), bool(sonic), True
    except Exception:
        return np.nan, np.nan, np.nan, False, False


def _solve_at_ppf(
    ppf_surf: float,
    *,
    pwh: float,
    tsu: float,
    nozzle: str,
    throat: str,
    knz: float,
    ken: float,
    kth: float,
    kdi: float,
    wellbore,
    wellprof,
    ipr_su,
    prop_su,
    prop_pf,
    jpump_direction: str,
) -> tuple[float, float, float, bool, bool]:
    """Solve the jet pump once at a given ppf_surf (catalog pump)."""
    try:
        jp = JetPump(nozzle, throat, knz=knz, ken=ken, kth=kth, kdi=kdi)
    except Exception:
        return np.nan, np.nan, np.nan, False, False
    return _solve_with_pump(
        ppf_surf,
        jp,
        pwh=pwh,
        tsu=tsu,
        wellbore=wellbore,
        wellprof=wellprof,
        ipr_su=ipr_su,
        prop_su=prop_su,
        prop_pf=prop_pf,
        jpump_direction=jpump_direction,
    )


def calibrate_pf_for_lift(
    *,
    well_name: str,
    target_lift: float,
    pwh: float,
    tsu: float,
    nozzle: str,
    throat: str,
    knz: float,
    ken: float,
    kth: float,
    kdi: float,
    wellbore,
    wellprof,
    ipr_su,
    prop_su,
    prop_pf,
    jpump_direction: str = "reverse",
    ppf_lo: float = PPF_LO_DEFAULT,
    ppf_hi: float = PPF_HI_DEFAULT,
    tol_psi: float = TOL_PSI,
    max_iter: int = MAX_ITER,
) -> PfCalibrationResult:
    """Binary-search ppf_surf so jetpump_solver's qnz matches target_lift.

    Bound-handling mirrors ``pf_scenario._estimate_bhp``:
      - Both residuals < 0 (modeled_qnz < target at both lo and hi) →
        target unreachable; return bounded=True at ppf_hi.
      - Both residuals > 0 (modeled_qnz > target even at ppf_lo) →
        overshoot; return bounded=True at ppf_lo.
      - Otherwise bisect until |hi − lo| < tol_psi or max_iter is hit.

    The residual is monotonically increasing in ppf_surf (more PF pressure
    → more nozzle flow), so plain bisection converges in O(log2(range/tol))
    iterations — ~10 for the default [1000, 5000] / 5 psi.
    """

    def _residual(ppf: float):
        qnz, qoil, psu, sonic, ok = _solve_at_ppf(
            ppf,
            pwh=pwh, tsu=tsu,
            nozzle=nozzle, throat=throat,
            knz=knz, ken=ken, kth=kth, kdi=kdi,
            wellbore=wellbore, wellprof=wellprof,
            ipr_su=ipr_su, prop_su=prop_su, prop_pf=prop_pf,
            jpump_direction=jpump_direction,
        )
        if not ok:
            return None, qnz, qoil, psu, sonic
        return qnz - target_lift, qnz, qoil, psu, sonic

    res_lo, qnz_lo, qoil_lo, psu_lo_val, sonic_lo = _residual(ppf_lo)
    res_hi, qnz_hi, qoil_hi, psu_hi_val, sonic_hi = _residual(ppf_hi)

    if res_lo is None or res_hi is None:
        return PfCalibrationResult(
            well_name=well_name, target_lift=target_lift,
            ppf_surf=float("nan"), modeled_qnz=float("nan"),
            modeled_qoil=float("nan"), modeled_bhp=float("nan"),
            lift_residual=float("nan"),
            converged=False, bounded=False, sonic=False, iterations=2,
        )

    # Both negative — target unreachable even at ppf_hi (washing out hard).
    if res_lo < 0 and res_hi < 0:
        return PfCalibrationResult(
            well_name=well_name, target_lift=target_lift,
            ppf_surf=ppf_hi, modeled_qnz=qnz_hi, modeled_qoil=qoil_hi,
            modeled_bhp=psu_hi_val, lift_residual=res_hi,
            converged=False, bounded=True, sonic=sonic_hi, iterations=2,
        )
    # Both positive — modeled_qnz overshoots even at ppf_lo. Means lift_wat
    # is small relative to what this nozzle moves; not a washout, just
    # under-pumped.
    if res_lo > 0 and res_hi > 0:
        return PfCalibrationResult(
            well_name=well_name, target_lift=target_lift,
            ppf_surf=ppf_lo, modeled_qnz=qnz_lo, modeled_qoil=qoil_lo,
            modeled_bhp=psu_lo_val, lift_residual=res_lo,
            converged=False, bounded=True, sonic=sonic_lo, iterations=2,
        )

    lo, hi = ppf_lo, ppf_hi
    res_l = res_lo
    iters = 2
    qnz_m = qoil_m = psu_m = float("nan")
    sonic_m = False
    for _ in range(max_iter):
        if abs(hi - lo) < tol_psi:
            break
        mid = (lo + hi) / 2.0
        res_mid, qnz_m, qoil_m, psu_m, sonic_m = _residual(mid)
        iters += 1
        if res_mid is None:
            # Solver failed at mid — typically the high-pressure side gets
            # numerically uncomfortable; mirror _estimate_bhp and pull hi
            # down toward the failure boundary.
            hi = mid
            continue
        if (res_l < 0 and res_mid < 0) or (res_l > 0 and res_mid > 0):
            lo, res_l = mid, res_mid
        else:
            hi = mid

    mid_final = (lo + hi) / 2.0
    res_final, qnz_f, qoil_f, psu_f, sonic_f = _residual(mid_final)
    iters += 1
    if res_final is None:
        return PfCalibrationResult(
            well_name=well_name, target_lift=target_lift,
            ppf_surf=mid_final, modeled_qnz=qnz_m, modeled_qoil=qoil_m,
            modeled_bhp=psu_m, lift_residual=float("nan"),
            converged=False, bounded=False, sonic=sonic_m, iterations=iters,
        )
    return PfCalibrationResult(
        well_name=well_name, target_lift=target_lift,
        ppf_surf=mid_final, modeled_qnz=qnz_f, modeled_qoil=qoil_f,
        modeled_bhp=psu_f, lift_residual=res_final,
        converged=abs(hi - lo) < tol_psi, bounded=False,
        sonic=sonic_f, iterations=iters,
    )


# ---------------------------------------------------------------------------
# Nozzle-wear estimation (PF pressure MEASURED → area is the free variable)
# ---------------------------------------------------------------------------

# Effective nozzle-AREA factor bounds. >1 = washed out (bigger bore), <1 =
# partial plug / smaller pump than the tracker says. 1.6× area ≈ +26%
# diameter — beyond that the "same pump, worn" story isn't credible and the
# result should be read as "verify the pump identity".
WEAR_LO_DEFAULT = 0.7
WEAR_HI_DEFAULT = 1.6
WEAR_TOL = 0.005


@dataclass
class NozzleWearResult:
    well_name: str
    target_lift: float       # observed lift_wat (BWPD)
    ppf_surf: float          # MEASURED test-day PF pressure held fixed (psi)
    wear_factor: float       # effective nozzle-AREA multiple vs catalog
    dnz_catalog: float       # catalog nozzle diameter (in)
    dnz_effective: float     # dnz_catalog * sqrt(wear_factor) (in)
    equivalent_nozzle: str   # nearest catalog nozzle number to dnz_effective
    modeled_qnz: float       # modeled PF rate at the returned factor (BWPD)
    modeled_bhp: float       # diagnostic: suction pressure at the factor (psi)
    lift_residual: float     # modeled_qnz − target_lift
    converged: bool
    bounded: bool            # hit the wear-factor search bound
    sonic: bool
    iterations: int


def _nearest_nozzle_no(dnz: float) -> str:
    diffs = [abs(d - dnz) for d in JetPump.nozzle_dia]
    return str(diffs.index(min(diffs)) + 1)


def estimate_nozzle_wear(
    *,
    well_name: str,
    target_lift: float,
    ppf_surf: float,
    pwh: float,
    tsu: float,
    nozzle: str,
    throat: str,
    knz: float,
    ken: float,
    kth: float,
    kdi: float,
    wellbore,
    wellprof,
    ipr_su,
    prop_su,
    prop_pf,
    jpump_direction: str = "reverse",
    wear_lo: float = WEAR_LO_DEFAULT,
    wear_hi: float = WEAR_HI_DEFAULT,
    tol: float = WEAR_TOL,
    max_iter: int = MAX_ITER,
) -> NozzleWearResult:
    """Bisect the effective nozzle-AREA factor so modeled qnz matches the
    measured lift_wat at the MEASURED PF surface pressure.

    The inverse of :func:`calibrate_pf_for_lift` for the live-PF era: with
    ``ppf_surf`` measured (vw_pressure_daily test-day join), the pressure is
    no longer the free variable — the nozzle's effective flow area is. The
    factor is applied as ``dnz_eff = dnz_catalog * sqrt(factor)`` on a
    catalog pump (JetPump stores dnz as a plain attribute; anz/ate are
    derived properties, so ate = ath − anz shrinks self-consistently as the
    nozzle wears — physically what a wash-out does).

    Interpretation: ~1.0 = healthy; > ~1.10 = wash-out (compare
    ``equivalent_nozzle``: "your 12 is flowing like a 13"); < ~0.90 =
    partial plug or the tracker's pump identity is wrong. A ``bounded``
    result means even the search limits can't reproduce the measured rate —
    verify the pump identity / allocation before trusting any calibration.

    This is a DIAGNOSTIC — it does not mutate the model. qnz is monotonic
    in nozzle area, so plain bisection converges like the pressure search.
    """
    catalog = JetPump(nozzle, throat, knz=knz, ken=ken, kth=kth, kdi=kdi)
    dnz_catalog = float(catalog.dnz)

    def _residual(factor: float):
        jp = JetPump(nozzle, throat, knz=knz, ken=ken, kth=kth, kdi=kdi)
        jp.dnz = dnz_catalog * float(np.sqrt(factor))
        qnz, _qoil, psu, sonic, ok = _solve_with_pump(
            ppf_surf,
            jp,
            pwh=pwh, tsu=tsu,
            wellbore=wellbore, wellprof=wellprof,
            ipr_su=ipr_su, prop_su=prop_su, prop_pf=prop_pf,
            jpump_direction=jpump_direction,
        )
        if not ok:
            return None, qnz, psu, sonic
        return qnz - target_lift, qnz, psu, sonic

    def _result(factor, qnz, psu, res, converged, bounded, sonic, iters):
        dnz_eff = dnz_catalog * float(np.sqrt(factor))
        return NozzleWearResult(
            well_name=well_name, target_lift=target_lift, ppf_surf=ppf_surf,
            wear_factor=float(factor), dnz_catalog=dnz_catalog,
            dnz_effective=dnz_eff,
            equivalent_nozzle=_nearest_nozzle_no(dnz_eff),
            modeled_qnz=qnz, modeled_bhp=psu, lift_residual=res,
            converged=converged, bounded=bounded, sonic=sonic,
            iterations=iters,
        )

    res_lo, qnz_lo, psu_lo, sonic_lo = _residual(wear_lo)
    res_hi, qnz_hi, psu_hi, sonic_hi = _residual(wear_hi)

    if res_lo is None or res_hi is None:
        return _result(
            1.0, float("nan"), float("nan"), float("nan"),
            converged=False, bounded=False, sonic=False, iters=2,
        )
    # Both negative: even a 1.6×-area nozzle can't reach the measured rate.
    if res_lo < 0 and res_hi < 0:
        return _result(
            wear_hi, qnz_hi, psu_hi, res_hi,
            converged=False, bounded=True, sonic=sonic_hi, iters=2,
        )
    # Both positive: even a 0.7×-area nozzle overshoots the measured rate.
    if res_lo > 0 and res_hi > 0:
        return _result(
            wear_lo, qnz_lo, psu_lo, res_lo,
            converged=False, bounded=True, sonic=sonic_lo, iters=2,
        )

    lo, hi = wear_lo, wear_hi
    res_l = res_lo
    iters = 2
    for _ in range(max_iter):
        if abs(hi - lo) < tol:
            break
        mid = (lo + hi) / 2.0
        res_mid, _qnz_m, _psu_m, _sonic_m = _residual(mid)
        iters += 1
        if res_mid is None:
            hi = mid
            continue
        if (res_l < 0 and res_mid < 0) or (res_l > 0 and res_mid > 0):
            lo, res_l = mid, res_mid
        else:
            hi = mid

    mid_final = (lo + hi) / 2.0
    res_f, qnz_f, psu_f, sonic_f = _residual(mid_final)
    iters += 1
    if res_f is None:
        return _result(
            mid_final, float("nan"), float("nan"), float("nan"),
            converged=False, bounded=False, sonic=False, iters=iters,
        )
    return _result(
        mid_final, qnz_f, psu_f, res_f,
        converged=abs(hi - lo) < tol, bounded=False, sonic=sonic_f,
        iters=iters,
    )
