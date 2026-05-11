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
    """Solve the jet pump once at a given ppf_surf.

    Returns (qnz, qoil, psu, sonic, ok). All-NaN with ok=False on solver
    failure. Mirrors fric_calibration._solve_at_coefs's defensive wrap.
    """
    try:
        jp = JetPump(nozzle, throat, knz=knz, ken=ken, kth=kth, kdi=kdi)
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
