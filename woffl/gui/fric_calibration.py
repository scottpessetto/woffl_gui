"""Friction-coefficient auto-calibration for jet-pump BHP matching.

Sweeps the entrance (ken), throat (kth), and diffuser (kdi) friction
coefficients via ``scipy.optimize.minimize`` (Nelder-Mead) to find the
combination minimizing |modeled_BHP − target_BHP| at the latest-test
conditions for a single well.

The objective is BHP-only. knz (nozzle) is held fixed at 0.01 — varying it
trades off against PF rate match without improving BHP, and the field
typically sees PF rate match well at the default. Solver failures inside
the search are absorbed by returning a 1e6 penalty so the optimizer steps
away from bad regions rather than crashing.

Multi-start refinement kicks in when a single Nelder-Mead pass leaves more
than ``MULTISTART_THRESHOLD`` psi of error — the optimizer is re-seeded from
the corners of the bound box so we don't get stuck in the wrong basin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from woffl.assembly.solopump import jetpump_solver
from woffl.geometry.jetpump import JetPump

# Bounds for each varied parameter. ken (entrance loss) is geometry-dominated
# and usually small, but on hard-to-match wells it's allowed up to 0.40 so the
# optimizer can lean on it when kth/kdi alone can't reach the target BHP. Keep
# the sidebar ``ken`` widget max in sync (sidebar._render_loss_coefs) or a
# calibrated ken > the widget max gets silently reset.
KEN_BOUNDS = (0.005, 0.40)
KTH_BOUNDS = (0.05, 1.0)
KDI_BOUNDS = (0.05, 1.0)
ALL_BOUNDS = [KEN_BOUNDS, KTH_BOUNDS, KDI_BOUNDS]

NEUTRAL_KEN = 0.03
NEUTRAL_KTH = 0.30
NEUTRAL_KDI = 0.30
SOLVER_FAIL_PENALTY = 1e6
MULTISTART_THRESHOLD = 50.0  # psi — trigger alternate seeds if first pass is worse
BOUND_TOL = 0.01             # how close to a bound counts as "bounded"
GOOD_PSI = 25.0
FAIR_PSI = 75.0
# Strategic seed points spanning the 3D bound box — covers fully-clean,
# discharge-only-loss, suction-only-loss, and fully-degraded regimes so
# multi-start can escape local minima. The high-ken seeds (0.30) only get
# tried when the first pass misses by > MULTISTART_THRESHOLD, so ken is the
# last lever pulled — used to rescue a match when kth/kdi alone can't.
ALT_STARTS = [
    (0.01, 0.10, 0.10),  # all clean
    (0.10, 0.10, 0.10),  # suction loss only
    (0.01, 0.80, 0.80),  # discharge loss only
    (0.10, 0.80, 0.80),  # all degraded
    (0.30, 0.30, 0.30),  # high entrance loss, moderate discharge
    (0.30, 0.80, 0.80),  # high entrance loss + degraded discharge (last resort)
]


@dataclass
class FricCalibrationResult:
    well_name: str
    target_bhp: float
    knz: float            # held fixed at 0.01 throughout
    seed_ken: float       # ken value the optimizer was seeded with (caller's input)
    best_ken: float
    best_kth: float
    best_kdi: float
    best_modeled_bhp: float
    best_oil: float
    best_pf_rate: float
    bhp_error: float
    converged: bool
    iterations: Optional[int] = None
    method: str = "nelder_mead"
    # Diagnostics for understanding why a calibration is or isn't accurate
    match_quality: str = "unknown"  # "good" / "fair" / "poor" / "failed"
    bounded: bool = False           # True when an optimum coef is at the search bound
    sonic: bool = False             # True when modeled flow is sonic-pinned at optimum
    starts_tried: int = 1           # how many seed points were tried


def _solve_at_coefs(
    kth: float,
    kdi: float,
    *,
    pwh: float,
    tsu: float,
    ppf_surf: float,
    nozzle: str,
    throat: str,
    knz: float,
    ken: float,
    wellbore,
    wellprof,
    ipr_su,
    prop_su,
    prop_pf,
    jpump_direction: str,
):
    """Solve once at a given (kth, kdi).

    Returns (modeled_bhp, oil, pf_rate, sonic_status, ok).
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
        return float(psu), float(qoil), float(qnz), bool(sonic), True
    except Exception:
        return np.nan, np.nan, np.nan, False, False


def _classify_match(abs_err: float) -> str:
    if abs_err <= GOOD_PSI:
        return "good"
    if abs_err <= FAIR_PSI:
        return "fair"
    return "poor"


def _is_bounded(ken: float, kth: float, kdi: float) -> bool:
    """True when any of the three calibrated coefs sits on its search bound."""
    for value, (lo, hi) in zip([ken, kth, kdi], ALL_BOUNDS):
        if abs(value - lo) < BOUND_TOL or abs(value - hi) < BOUND_TOL:
            return True
    return False


def _clip_to_bounds(x: list[float]) -> tuple[float, float, float]:
    """Clip [ken, kth, kdi] to their respective bounds."""
    return tuple(float(np.clip(v, *b)) for v, b in zip(x, ALL_BOUNDS))


def _run_one_start(
    x0: tuple[float, float, float],
    target_bhp: float,
    solver_kwargs: dict,
):
    """Run a single Nelder-Mead pass from one seed point in (ken, kth, kdi).

    ``solver_kwargs`` must NOT include ``ken`` — it's varied by the optimizer
    and passed into ``_solve_at_coefs`` from the optimizer's x vector.

    Returns (ken, kth, kdi, modeled_bhp, oil, pf_rate, sonic, ok, iterations,
    abs_error). abs_error is +inf when the final solve fails.
    """

    def objective(x):
        for value, (lo, hi) in zip(x, ALL_BOUNDS):
            if not (lo <= value <= hi):
                return SOLVER_FAIL_PENALTY
        ken, kth, kdi = x
        psu, _oil, _pf, _sonic, ok = _solve_at_coefs(
            kth, kdi, ken=ken, **solver_kwargs
        )
        if not ok:
            return SOLVER_FAIL_PENALTY
        return abs(psu - target_bhp)

    # Higher maxiter for 3D vs 2D; tolerances unchanged
    result = minimize(
        objective,
        list(x0),
        method="Nelder-Mead",
        bounds=ALL_BOUNDS,
        options={"xatol": 0.001, "fatol": 0.5, "maxiter": 150},
    )

    ken_opt, kth_opt, kdi_opt = _clip_to_bounds(result.x)
    psu, oil, pf, sonic, ok = _solve_at_coefs(
        kth_opt, kdi_opt, ken=ken_opt, **solver_kwargs
    )
    abs_err = abs(psu - target_bhp) if ok else float("inf")
    iters = int(result.nit) if hasattr(result, "nit") else None
    return ken_opt, kth_opt, kdi_opt, psu, oil, pf, sonic, ok, iters, abs_err


def calibrate_friction_coefs(
    *,
    well_name: str,
    target_bhp: float,
    pwh: float,
    tsu: float,
    ppf_surf: float,
    nozzle: str,
    throat: str,
    knz: float,
    ken: float,
    wellbore,
    wellprof,
    ipr_su,
    prop_su,
    prop_pf,
    jpump_direction: str = "reverse",
) -> FricCalibrationResult:
    """Find (ken, kth, kdi) that drives modeled BHP toward ``target_bhp``.

    Runs Nelder-Mead from a neutral seed (NEUTRAL_KEN, 0.30, 0.30). If the
    residual is worse than ``MULTISTART_THRESHOLD`` psi, retries from the
    corners of the bound box and keeps the best result.

    Args:
        target_bhp: Actual BHP from the latest well test (psi).
        knz: Held fixed at the passed value (typically 0.01).
        ken: Seed value (e.g. from Databricks ``jpfric_entry`` or sidebar).
            The optimizer will vary ken within ``KEN_BOUNDS``.

    Returns:
        FricCalibrationResult with diagnostics:
          - ``converged``: final coefs produced a valid solve
          - ``match_quality``: "good"/"fair"/"poor" based on |bhp_error|
          - ``bounded``: any coef sitting on a search bound
          - ``sonic``: solver reports sonic flow at the optimum (BHP is
            choke-pinned by throat geometry, friction coefs cannot bring
            it down further)
    """
    # ken is varied — it's NOT in solver_kwargs and is passed via the x vector.
    solver_kwargs = dict(
        pwh=pwh,
        tsu=tsu,
        ppf_surf=ppf_surf,
        nozzle=nozzle,
        throat=throat,
        knz=knz,
        wellbore=wellbore,
        wellprof=wellprof,
        ipr_su=ipr_su,
        prop_su=prop_su,
        prop_pf=prop_pf,
        jpump_direction=jpump_direction,
    )

    # Seed with the caller's ken (clamped) so we start near current operating
    # value rather than always 0.03 — speeds convergence when Databricks has
    # a real jpfric_entry stored.
    seed_ken = float(np.clip(ken, *KEN_BOUNDS))
    neutral_seed = (seed_ken, NEUTRAL_KTH, NEUTRAL_KDI)

    best = _run_one_start(neutral_seed, target_bhp, solver_kwargs)
    starts_tried = 1
    total_iters = best[8] or 0

    if best[9] > MULTISTART_THRESHOLD:
        for x0 in ALT_STARTS:
            attempt = _run_one_start(x0, target_bhp, solver_kwargs)
            starts_tried += 1
            total_iters += attempt[8] or 0
            if attempt[9] < best[9]:
                best = attempt
            if best[9] <= GOOD_PSI:
                break

    ken_opt, kth_opt, kdi_opt, psu, oil, pf, sonic, ok, _, abs_err = best

    if not ok:
        return FricCalibrationResult(
            well_name=well_name,
            target_bhp=target_bhp,
            knz=knz,
            seed_ken=ken,
            best_ken=ken_opt,
            best_kth=kth_opt,
            best_kdi=kdi_opt,
            best_modeled_bhp=np.nan,
            best_oil=np.nan,
            best_pf_rate=np.nan,
            bhp_error=np.nan,
            converged=False,
            iterations=total_iters,
            match_quality="failed",
            bounded=_is_bounded(ken_opt, kth_opt, kdi_opt),
            sonic=False,
            starts_tried=starts_tried,
        )

    return FricCalibrationResult(
        well_name=well_name,
        target_bhp=target_bhp,
        knz=knz,
        seed_ken=ken,
        best_ken=ken_opt,
        best_kth=kth_opt,
        best_kdi=kdi_opt,
        best_modeled_bhp=psu,
        best_oil=oil,
        best_pf_rate=pf,
        bhp_error=psu - target_bhp,
        converged=True,
        iterations=total_iters,
        match_quality=_classify_match(abs_err),
        bounded=_is_bounded(ken_opt, kth_opt, kdi_opt),
        sonic=sonic,
        starts_tried=starts_tried,
    )


def compute_bhp_decomposition(
    cal_result: FricCalibrationResult,
    *,
    pwh: float,
    tsu: float,
    ppf_surf: float,
    wellbore,
    wellprof,
    prop_su,
    prop_pf,
    jpump_direction: str = "reverse",
) -> Optional[dict]:
    """Decompose the modeled BHP at the calibrated coefs into pressure
    components, for diagnosing why a well isn't matching well.

    Reuses the converged ``psu``, ``qoil``, and ``qpf`` from ``cal_result`` —
    no extra Nelder-Mead step. Re-runs the production-column integration
    (cheap) and the PF friction calc to extract the components.

    The production hydrostatic is approximated using a depth-averaged
    mixture density (single average density × column height), not the full
    Beggs per-segment integration. This is approximate — but it's
    comparable across wells, which is what we need for the diagnostic.

    Returns None if the cal_result didn't converge.
    """
    if not cal_result.converged:
        return None

    from woffl.flow import jetflow as jf
    from woffl.flow import outflow as of
    from woffl.flow import singlephase as sp
    from woffl.pvt.resmix import ResMix

    if jpump_direction == "reverse":
        production_flowpath = "tubing"
        powerfluid_flowpath = "annulus"
    else:
        production_flowpath = "annulus"
        powerfluid_flowpath = "tubing"

    psu = float(cal_result.best_modeled_bhp)
    qoil_std = float(cal_result.best_oil)
    qpf = float(cal_result.best_pf_rate)

    # ── PF side (single-phase water column) ─────────────────────────────
    # diff_press_static returns negative for downward direction; flip for the
    # positive hydrostatic gain going down.
    dp_stat_pf = sp.diff_press_static(prop_pf.density, -1 * wellprof.jetpump_vd)
    pf_hydrostatic = -float(dp_stat_pf)
    pf_friction = float(of.powerfluid_top_down_friction(
        ppf_surf, tsu, qpf, prop_pf, wellbore, wellprof, powerfluid_flowpath
    ))
    pni = ppf_surf - dp_stat_pf - pf_friction  # nozzle inlet pressure

    # ── Production column ──────────────────────────────────────────────
    # Build the throat-mixed fluid the production column actually carries
    wc_tm, _ = jf.throat_wc(qoil_std, prop_su.wc, qpf)
    prop_tm = ResMix(wc_tm, prop_su.fgor, prop_su.oil, prop_su.wat, prop_su.gas)

    _md_seg, prs_ray, _slh_ray = of.production_top_down_press(
        pwh, tsu, qoil_std, prop_tm, wellbore, wellprof, production_flowpath
    )
    pdi_of = float(prs_ray[-1])
    prod_total = pdi_of - pwh  # hydrostatic + friction (both positive going down)

    # Hydrostatic via depth-averaged mixture density. Uses suction temp as
    # the column temp (real column is cooler at the top — this slightly
    # under-counts hydrostatic, but consistently across wells so it's
    # useful for comparison).
    p_avg = (pwh + pdi_of) / 2.0
    prop_tm_avg = prop_tm.condition(p_avg, tsu)
    rho_prod_avg = float(prop_tm_avg.rho_mix())  # lbm/ft³
    prod_hydrostatic = float(sp.diff_press_static(rho_prod_avg, wellprof.jetpump_vd))
    prod_friction = prod_total - prod_hydrostatic

    pump_dp = pdi_of - psu  # pump pressure rise

    vd = float(wellprof.jetpump_vd)
    return {
        "pf_hydrostatic": pf_hydrostatic,
        "pf_friction": pf_friction,
        "pni": float(pni),
        "prod_hydrostatic": prod_hydrostatic,
        "prod_friction": prod_friction,
        "prod_total": prod_total,
        "pump_dp": pump_dp,
        "rho_prod_avg": rho_prod_avg,
        "prod_grad_psi_per_ft": prod_hydrostatic / vd if vd > 0 else None,
        "pf_grad_psi_per_ft": pf_hydrostatic / vd if vd > 0 else None,
        "jetpump_vd": vd,
    }
