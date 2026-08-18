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

import copy
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
    # Human-readable explanation for special outcomes. Set on "pinned" runs
    # (sonic well: coefficients left at their seeds); None otherwise.
    message: Optional[str] = None


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
          - ``match_quality == "pinned"``: the final solve was sonic, so the
            single-point match is degenerate (kth/kdi have zero psu gradient
            and ken only moves the cavitation floor). Coefficients are
            returned at their SEEDS, not the optimizer's railed values, and
            ``message`` explains the floor gap.
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
    if sonic:
        # Sonic-pinned: psu sits on the cavitation floor. On that branch a
        # single BHP point cannot identify friction - kth/kdi have zero
        # gradient on psu (they cannot move the floor) and ken only moves
        # the floor itself - so the optimizer rails coefficients without
        # learning anything (MPM-64 railed at ken=0.40/kth=0.05/kdi=0.05,
        # writing calibration-day gauge BHP into the floor). Return the
        # seeds instead of the railed values and flag the run "pinned".
        s_psu, s_oil, s_pf, _s_sonic, s_ok = _solve_at_coefs(
            NEUTRAL_KTH, NEUTRAL_KDI, ken=seed_ken, **solver_kwargs
        )
        if s_ok:
            gap = s_psu - target_bhp
            return FricCalibrationResult(
                well_name=well_name,
                target_bhp=target_bhp,
                knz=knz,
                seed_ken=ken,
                best_ken=seed_ken,
                best_kth=NEUTRAL_KTH,
                best_kdi=NEUTRAL_KDI,
                best_modeled_bhp=s_psu,
                best_oil=s_oil,
                best_pf_rate=s_pf,
                bhp_error=gap,
                converged=True,
                iterations=total_iters,
                match_quality="pinned",
                bounded=_is_bounded(seed_ken, NEUTRAL_KTH, NEUTRAL_KDI),
                sonic=True,
                starts_tried=starts_tried,
                message=(
                    "target BHP sits on the cavitation floor at these inputs "
                    "- a single BHP point cannot identify friction on a sonic "
                    "well (ken would only move the floor; kth/kdi cannot move "
                    "it at all). Left coefficients at their seeds. "
                    f"Floor gap: {gap:+.0f} psi."
                ),
            )
        # Seed solve failed even though the optimum solved (should not
        # happen) - fall through and report the optimizer's result as today.

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
    # Build prop_tm from COPIES of prop_su's child PVT objects. ResMix.condition
    # mutates its children in place, so sharing prop_su.oil/wat/gas here would
    # re-condition the CALLER's prop_su (stale derived props if it's reused after
    # this diagnostic).
    prop_tm = ResMix(
        wc_tm, prop_su.fgor,
        copy.deepcopy(prop_su.oil), copy.deepcopy(prop_su.wat), copy.deepcopy(prop_su.gas),
    )

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


# ---------------------------------------------------------------------------
# Multi-point event calibration (Pillar 1b)
# ---------------------------------------------------------------------------
# Fits (ken, kth, kdi, fnz, mach_crit) against EVERY measured operating point
# in the current pump era simultaneously. fnz is an effective nozzle-AREA
# factor (dnz_eff = dnz_catalog * sqrt(fnz), the pf_calibration wear pattern)
# so washout is a fitted parameter instead of a separate diagnostic.
# mach_crit relaxes the throat-entry cavitation floor (sonic cutoff moves
# from Mach 1.0 to mach_crit) so a well whose measured BHP level AND
# dBHP/dPpf response sit below today's floor stays reachable. Each point
# gets its own IPR anchor (oil-basis Vogel through that point's qtot/bhp) so
# IPR drift stays out of the friction fit.

FNZ_BOUNDS = (0.8, 1.3)
MACH_CRIT_BOUNDS = (1.0, 2.5)
MP_BOUNDS = [KEN_BOUNDS, KTH_BOUNDS, KDI_BOUNDS, FNZ_BOUNDS, MACH_CRIT_BOUNDS]
MP_PARAM_NAMES = ("ken", "kth", "kdi", "fnz", "mach_crit")
MP_KNZ = 0.01                 # nozzle loss held fixed, as in the 1-pt path
MP_SEED_KDI = 0.40            # library BatchPump default, not NEUTRAL_KDI
MP_BHP_SCALE_PSI = 50.0       # 1-sigma BHP mismatch in the objective
MP_PF_SCALE_FRAC = 0.05       # 1-sigma PF-rate mismatch (fraction of meas)
MP_PWF_MARGIN_PSI = 25.0      # point dropped unless bhp < res_pres - margin
MP_MAXITER = 100
MP_POOR_RMS_PF_PCT = 5.0      # with MULTISTART_THRESHOLD, gates the alt start
MP_BETA_DPPF = 300.0          # finite-difference step for implied beta, psi
# Paired-difference term: a flat fit (every point parked at the mean of the
# measured BHPs) scores WELL on pure levels, so nothing rewards the
# dBHP/dPpf RESPONSE and Nelder-Mead happily leaves mach_crit on the floor
# with implied_beta ~ 0. Differencing consecutive-in-ppf points (plus the
# full min-to-max span) cancels the level and scores the response directly.
MP_DBHP_SCALE = 25.0          # psi - 1-sigma mismatch on paired dBHP
MP_MIN_DPPF_PSI = 100.0       # pairs closer than this in ppf carry no signal
# Huber knee for every normalized residual (level BHP, level PF, paired
# dBHP), in 1-sigma units: quadratic inside +/- delta, linear outside.
# Median-like robustness - the miner's median over pairs beat the fitter's
# squared loss on noisy field days, so one wild gauge day pulls with a
# bounded slope instead of a squared one.
MP_HUBER_DELTA = 1.5
# One alternate seed at the woffl library defaults (JetPump ken/kth plus the
# BatchPump kdi) with a clean nozzle and the standard sonic cutoff - tried
# only when the first pass is poor.
MP_ALT_START = (0.03, 0.30, MP_SEED_KDI, 1.0, 1.0)


@dataclass
class MultipointResult:
    best_ken: float
    best_kth: float
    best_kdi: float
    best_fnz: float
    best_mach_crit: float
    rms_bhp_psi: float
    rms_pf_pct: float
    # RMS of (dpsu_model - dbhp_meas) over qualifying ppf pairs, psi.
    # None when no pair spans >= MP_MIN_DPPF_PSI (levels-only fit).
    rms_dbhp_psi: Optional[float]
    n_used: int
    n_dropped: int
    bounded: bool
    railed: list[str]
    implied_beta: Optional[float]
    per_point: list[dict]
    refusal: Optional[str]
    iterations: int
    message: Optional[str]


def _build_well_objects(well_config):
    """(wellbore, wellprofile, inflow, res_mix, prop_pf) for a WellConfig.

    Thin wrapper over the optimizer's single-source-of-truth factory so the
    fitter builds wells exactly like a batch run (and tests can stub it).
    """
    from woffl.assembly.network_optimizer import NetworkOptimizer

    return NetworkOptimizer._create_well_objects(well_config)


def _point_pvt_components(well_config):
    """PVT (oil, water, gas) via the same factory _create_well_objects uses."""
    from woffl.assembly.sim_factories import create_pvt_components

    return create_pvt_components(
        field_model=well_config.field_model,
        oil_api=well_config.oil_api,
        gas_sg=well_config.gas_sg,
        wat_sg=well_config.wat_sg,
        bubble_point=well_config.bubble_point,
    )


def _point_inflow(oil_rate: float, pwf: float, pres: float):
    """Oil-basis Vogel anchored on one point's own (rate, bhp)."""
    from woffl.flow.inflow import InFlow

    return InFlow(qwf=oil_rate, pwf=pwf, pres=pres)


def _point_res_mix(wc: float, fgor: float, pvt):
    from woffl.pvt.resmix import ResMix

    oil, water, gas = pvt
    return ResMix(wc=wc, fgor=fgor, oil=oil, wat=water, gas=gas)


def _mp_clip(x) -> tuple[float, float, float, float, float]:
    return tuple(float(np.clip(v, *b)) for v, b in zip(x, MP_BOUNDS))


def _huber(u: float) -> float:
    """Huber loss of a normalized residual: u^2 inside MP_HUBER_DELTA,
    linear continuation (matched value and slope) outside."""
    au = abs(u)
    if au <= MP_HUBER_DELTA:
        return au * au
    return MP_HUBER_DELTA * (2.0 * au - MP_HUBER_DELTA)


def _mp_pair_diffs(pts: list[tuple[float, float, float]]) -> list[tuple[float, float]]:
    """Paired differences over SOLVED points: [(dpsu_model, dbhp_meas), ...].

    ``pts`` is (ppf, bhp_meas, psu_model) per surviving point. ALL point
    combinations separated by at least MP_MIN_DPPF_PSI in ppf qualify -
    single day-pairs are noisy (gauge scatter, transients), and the miner's
    lesson applies here too: the response signal lives in the AGGREGATE of
    many pairs, not a handful (consecutive-only pairing left 2-3 pairs per
    well after the 20-point stratified cap, and the fit locked onto their
    noise). Differences are taken high-ppf minus low-ppf. O(n^2) on <= 20
    points is free - the solves are already done.
    """
    if len(pts) < 2:
        return []
    by_ppf = sorted(pts, key=lambda p: p[0])
    diffs = []
    for i in range(len(by_ppf) - 1):
        lo = by_ppf[i]
        for j in range(i + 1, len(by_ppf)):
            hi = by_ppf[j]
            if hi[0] - lo[0] >= MP_MIN_DPPF_PSI:
                diffs.append((hi[2] - lo[2], hi[1] - lo[1]))
    return diffs


def _mp_railed(x) -> list[str]:
    """Parameter names sitting within BOUND_TOL of a search bound.

    mach_crit's LOWER bound (1.0) is the no-op default - the physical sonic
    cutoff - so resting there is expected, not a rail; only the upper bound
    counts for it.
    """
    railed = []
    for name, v, (lo, hi) in zip(MP_PARAM_NAMES, x, MP_BOUNDS):
        if name != "mach_crit" and abs(v - lo) < BOUND_TOL:
            railed.append(name)
        elif abs(v - hi) < BOUND_TOL:
            railed.append(name)
    return railed


def _mp_refused(refusal: str, seed, n_dropped: int = 0) -> MultipointResult:
    ken, kth, kdi, fnz, mach_crit = seed
    return MultipointResult(
        best_ken=ken, best_kth=kth, best_kdi=kdi, best_fnz=fnz,
        best_mach_crit=mach_crit,
        rms_bhp_psi=float("nan"), rms_pf_pct=float("nan"), rms_dbhp_psi=None,
        n_used=0, n_dropped=n_dropped,
        bounded=False, railed=[], implied_beta=None, per_point=[],
        refusal=refusal, iterations=0,
        message=f"refused: {refusal}",
    )


def calibrate_multipoint(
    well_config,
    nozzle: str,
    throat: str,
    points: list[dict],
    *,
    seed: tuple = None,
) -> MultipointResult:
    """Fit (ken, kth, kdi, fnz, mach_crit) against many measured points.

    ``points`` is the builder's point-dict list ({date, kind, ppf, bhp,
    pf_rate, pwh, qtot, oil, wc, fgor, weight}); passing the whole builder
    result dict ({"points": [...], "refusal": ...}) also works — a builder
    refusal is mirrored straight into the result without fitting.

    Well objects are built ONCE from the config; each point then gets its
    own IPR anchor (oil-basis through that point's qtot/wc at pwf = bhp,
    pres = config res_pres) and its own ResMix at the point's wc/fgor (PVT
    components built once, mixes cached per unique wc/fgor). fnz scales the
    nozzle area: dnz_eff = dnz_catalog * sqrt(fnz). mach_crit is handed to
    every jetpump_solver call (and the implied-beta probes) as the throat-
    entry sonic cutoff; 1.0 reproduces today's physics exactly.

    ``seed`` may be a 5-tuple (ken, kth, kdi, fnz, mach_crit) or a legacy
    4-tuple, which gets mach_crit 1.0 appended.

    Objective: sum over surviving points of
        weight * [H((psu - bhp)/50) + H((qnz - pf_rate)/(0.05*pf_rate))]
    plus a paired-difference term over qualifying point pairs sorted by ppf
    (all combinations with |dppf| >= 100 psi):
        w_pair * H((dpsu_model - dbhp_meas)/25)
    where H is the Huber loss (_huber: quadratic within MP_HUBER_DELTA,
    linear outside) and w_pair = sum(level weights) / n_pairs, so the
    response carries the same total weight as the level. Oil is excluded
    (circular under per-point anchoring). A point whose solve fails
    contributes a flat penalty during the search, is excluded from
    pairing, and is dropped (counted in n_dropped) at the optimum; more
    than half the input points dropping is a refusal.
    """
    refusal = None
    if isinstance(points, dict):
        refusal = points.get("refusal")
        points = points.get("points") or []

    if seed is None:
        seed = (
            getattr(well_config, "ken_well", None) or NEUTRAL_KEN,
            getattr(well_config, "kth_well", None) or NEUTRAL_KTH,
            getattr(well_config, "kdi_well", None) or MP_SEED_KDI,
            1.0,
            1.0,
        )
    elif len(seed) == 4:
        seed = (*seed, 1.0)
    seed = _mp_clip(seed)

    if refusal:
        return _mp_refused(refusal, seed)
    if not points:
        return _mp_refused("no points supplied", seed)

    wellbore, wellprof, _base_inflow, _res_mix, prop_pf = _build_well_objects(
        well_config
    )
    res_pres = float(well_config.res_pres)
    tsu = float(well_config.form_temp)
    direction = getattr(well_config, "jpump_direction", "reverse")
    surf_pres = getattr(well_config, "surf_pres", None)

    # --- per-point prep: IPR anchor + ResMix, independent of the params ---
    pvt = _point_pvt_components(well_config)
    mix_cache: dict[tuple[float, float], object] = {}
    ctxs: list[dict] = []
    n_dropped = 0
    for pt in points:
        try:
            bhp = float(pt["bhp"])
            ppf = float(pt["ppf"])
            pf_rate = float(pt["pf_rate"])
            # A daily with no near test carries wc/fgor = None (builder
            # contract) - fall back to the config's saved formation values.
            wc_raw = pt.get("wc")
            wc = float(wc_raw) if wc_raw is not None else float(well_config.form_wc)
            fgor_raw = pt.get("fgor")
            fgor = (
                float(fgor_raw)
                if fgor_raw is not None
                else float(well_config.form_gor)
            )
        except (KeyError, TypeError, ValueError):
            n_dropped += 1
            continue
        pwh = pt.get("pwh")
        pwh = float(pwh) if pwh is not None else surf_pres
        oil = pt.get("oil")
        if oil is None and pt.get("qtot") is not None:
            oil = float(pt["qtot"]) * (1.0 - wc)
        # pwf must sit safely below res_pres or the Vogel anchor degenerates.
        if (
            pwh is None
            or oil is None or oil <= 0
            or pf_rate <= 0
            or not np.isfinite(bhp) or not np.isfinite(ppf)
            or bhp >= res_pres - MP_PWF_MARGIN_PSI
        ):
            n_dropped += 1
            continue
        key = (round(wc, 6), round(fgor, 3))
        if key not in mix_cache:
            mix_cache[key] = _point_res_mix(wc, fgor, pvt)
        ctxs.append(
            {
                "date": pt.get("date"),
                "kind": pt.get("kind"),
                "ppf": ppf,
                "bhp": bhp,
                "pf_rate": pf_rate,
                "pwh": float(pwh),
                "weight": float(pt.get("weight") or 1.0),
                "inflow": _point_inflow(float(oil), bhp, res_pres),
                "res_mix": mix_cache[key],
            }
        )

    n_total = len(points)
    if not ctxs or n_dropped > n_total / 2.0:
        return _mp_refused(
            "more than half the points dropped as unsolvable "
            f"({n_dropped}/{n_total})",
            seed,
            n_dropped=n_dropped,
        )

    dnz_catalog = float(JetPump(nozzle, throat, knz=MP_KNZ).dnz)

    def _solve_all(x, cs=None):
        """Solve every point at params x -> list of (psu, qnz) | None."""
        cs = ctxs if cs is None else cs
        ken, kth, kdi, fnz, mach_crit = x
        try:
            jp = JetPump(nozzle, throat, knz=MP_KNZ, ken=ken, kth=kth, kdi=kdi)
            jp.dnz = dnz_catalog * float(np.sqrt(fnz))
        except Exception:
            return [None] * len(cs)
        out = []
        for ctx in cs:
            try:
                psu, _sonic, _qoil, _fwat, qnz, _mach = jetpump_solver(
                    pwh=ctx["pwh"],
                    tsu=tsu,
                    ppf_surf=ctx["ppf"],
                    jpump=jp,
                    wellbore=wellbore,
                    wellprof=wellprof,
                    ipr_su=ctx["inflow"],
                    prop_su=ctx["res_mix"],
                    prop_pf=prop_pf,
                    jpump_direction=direction,
                    mach_crit=mach_crit,
                )
                if (
                    psu is None or qnz is None
                    or np.isnan(psu) or np.isnan(qnz)
                ):
                    out.append(None)
                else:
                    out.append((float(psu), float(qnz)))
            except Exception:
                out.append(None)
        return out

    def _cost(x):
        for v, (lo, hi) in zip(x, MP_BOUNDS):
            if not (lo <= v <= hi):
                return SOLVER_FAIL_PENALTY * len(ctxs)
        total = 0.0
        w_solved = 0.0
        solved: list[tuple[float, float, float]] = []
        for ctx, res in zip(ctxs, _solve_all(x)):
            if res is None:
                total += SOLVER_FAIL_PENALTY
                continue
            psu, qnz = res
            bhp_term = (psu - ctx["bhp"]) / MP_BHP_SCALE_PSI
            pf_term = (qnz - ctx["pf_rate"]) / (MP_PF_SCALE_FRAC * ctx["pf_rate"])
            total += ctx["weight"] * (_huber(bhp_term) + _huber(pf_term))
            w_solved += ctx["weight"]
            solved.append((ctx["ppf"], ctx["bhp"], psu))
        # Paired-difference term (see MP_DBHP_SCALE): the TOTAL pair weight
        # matches the total level weight so response and level pull equally.
        diffs = _mp_pair_diffs(solved)
        if diffs:
            w_pair = w_solved / max(1, len(diffs))
            for d_model, d_meas in diffs:
                total += w_pair * _huber((d_model - d_meas) / MP_DBHP_SCALE)
        return total

    def _run(x0):
        result = minimize(
            _cost,
            list(x0),
            method="Nelder-Mead",
            bounds=MP_BOUNDS,
            options={
                "xatol": 1e-4,
                # scaled to point count: ~0.15 psi of BHP mismatch per point
                "fatol": 1e-5 * len(ctxs),
                "maxiter": MP_MAXITER,
            },
        )
        return _mp_clip(result.x), int(getattr(result, "nit", 0) or 0)

    def _summarize(x):
        """Final eval at x: per-point rows, drops, RMS errors (level + pair)."""
        rows, drops = [], 0
        se_bhp, se_pf, n = 0.0, 0.0, 0
        used = []
        for ctx, res in zip(ctxs, _solve_all(x)):
            if res is None:
                drops += 1
                continue
            psu, qnz = res
            rows.append(
                {
                    "date": ctx["date"],
                    "kind": ctx["kind"],
                    "ppf": ctx["ppf"],
                    "bhp_meas": ctx["bhp"],
                    "bhp_model": psu,
                    "pf_meas": ctx["pf_rate"],
                    "pf_model": qnz,
                }
            )
            used.append(ctx)
            se_bhp += (psu - ctx["bhp"]) ** 2
            se_pf += (100.0 * (qnz - ctx["pf_rate"]) / ctx["pf_rate"]) ** 2
            n += 1
        rms_bhp = float(np.sqrt(se_bhp / n)) if n else float("nan")
        rms_pf = float(np.sqrt(se_pf / n)) if n else float("nan")
        diffs = _mp_pair_diffs(
            [(row["ppf"], row["bhp_meas"], row["bhp_model"]) for row in rows]
        )
        n_pairs = len(diffs)
        rms_dbhp = (
            float(np.sqrt(sum((dm - dq) ** 2 for dm, dq in diffs) / n_pairs))
            if n_pairs
            else None
        )
        return rows, used, drops, rms_bhp, rms_pf, rms_dbhp, n_pairs

    best_x, iters = _run(seed)
    rows, used, solve_drops, rms_bhp, rms_pf, rms_dbhp, n_pairs = _summarize(best_x)
    poor = (
        not used
        or rms_bhp > MULTISTART_THRESHOLD
        or rms_pf > MP_POOR_RMS_PF_PCT
    )
    if poor:
        alt_x, alt_iters = _run(MP_ALT_START)
        iters += alt_iters
        if _cost(alt_x) < _cost(best_x):
            best_x = alt_x
            rows, used, solve_drops, rms_bhp, rms_pf, rms_dbhp, n_pairs = _summarize(best_x)

    # Floor-escape restart: a fit resting on the mach_crit=1.0 default with
    # a poor BHP match is the P3 signature - Nelder-Mead cannot see past the
    # cavitation-floor kink from a mach_crit=1.0 seed (every simplex step
    # stays pinned, so the mach direction looks flat). Reseed from the
    # current optimum with mach_crit lifted to its upper bound and keep the
    # better of the two; wells that genuinely fit on the floor are untouched.
    # A POOR paired-difference residual is the same trap wearing a smaller
    # miss - the flat fit parks at the mean of the measured BHPs and scores
    # well on levels - so it fires the restart too (mean pair residual
    # > 1.0 <=> rms_dbhp > MP_DBHP_SCALE), and does so from ANY mach_crit:
    # a simplex that stalled partway up the mach direction is still stuck
    # on the kink, just not at the seed.
    diff_poor = rms_dbhp is not None and rms_dbhp > MP_DBHP_SCALE
    if used and (
        diff_poor
        or (best_x[4] - MP_BOUNDS[4][0] < BOUND_TOL and rms_bhp > GOOD_PSI)
    ):
        esc_x, esc_iters = _run((*best_x[:4], MP_BOUNDS[4][1]))
        iters += esc_iters
        if _cost(esc_x) < _cost(best_x):
            best_x = esc_x
            rows, used, solve_drops, rms_bhp, rms_pf, rms_dbhp, n_pairs = _summarize(best_x)

    # Polish restart: Nelder-Mead's simplex collapses as it converges, and a
    # 5-parameter fit against the level + all-pairs objective routinely hits
    # MP_MAXITER mid-refinement. One restart FROM the current optimum
    # rebuilds a fresh simplex around it and finishes the descent; kept only
    # when it actually improves the cost.
    if used:
        pol_x, pol_iters = _run(tuple(best_x))
        iters += pol_iters
        if _cost(pol_x) < _cost(best_x):
            best_x = pol_x
            rows, used, solve_drops, rms_bhp, rms_pf, rms_dbhp, n_pairs = _summarize(best_x)

    n_dropped += solve_drops
    ken, kth, kdi, fnz, mach_crit = best_x
    railed = _mp_railed(best_x)

    if n_dropped > n_total / 2.0:
        r = _mp_refused(
            "more than half the points dropped as unsolvable "
            f"({n_dropped}/{n_total})",
            best_x,
            n_dropped=n_dropped,
        )
        r.n_used = len(used)
        r.per_point = rows
        r.rms_bhp_psi = rms_bhp
        r.rms_pf_pct = rms_pf
        r.rms_dbhp_psi = rms_dbhp
        r.bounded = bool(railed)
        r.railed = railed
        r.iterations = iters
        return r

    # --- implied beta: -(dpsu/dppf) at the last used point's IPR anchor ---
    implied_beta = None
    if used:
        ppf_med = float(np.median([c["ppf"] for c in used]))
        anchor = used[-1]
        psus = []
        for ppf_probe in (ppf_med, ppf_med - MP_BETA_DPPF):
            probe = dict(anchor)
            probe["ppf"] = ppf_probe
            res = _solve_all(best_x, [probe])[0]
            psus.append(res[0] if res is not None else None)
        if psus[0] is not None and psus[1] is not None:
            implied_beta = float(-(psus[0] - psus[1]) / MP_BETA_DPPF)

    message = (
        f"fit {len(used)} points: RMS BHP {rms_bhp:.0f} psi, "
        f"PF {rms_pf:.1f}%"
    )
    if rms_dbhp is not None:
        message += f", dBHP {rms_dbhp:.0f} psi over {n_pairs} pairs"
    message += (
        f"; fnz {fnz:.2f} (washout {(fnz - 1.0) * 100.0:+.0f}%), "
        f"mach_crit {mach_crit:.2f}"
    )

    return MultipointResult(
        best_ken=ken,
        best_kth=kth,
        best_kdi=kdi,
        best_fnz=fnz,
        best_mach_crit=mach_crit,
        rms_bhp_psi=rms_bhp,
        rms_pf_pct=rms_pf,
        rms_dbhp_psi=rms_dbhp,
        n_used=len(used),
        n_dropped=n_dropped,
        bounded=bool(railed),
        railed=railed,
        implied_beta=implied_beta,
        per_point=rows,
        refusal=None,
        iterations=iters,
        message=message,
    )
