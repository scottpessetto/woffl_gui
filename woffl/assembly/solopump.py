"""System Operations

Code that has to do with running the entire system together. A combination of the IPR,
PVT, JetPump and Outflow. Used to create a final solution to compare.
"""

import math

import numpy as np

from woffl.flow import InFlow
from woffl.flow import jetflow as jf
from woffl.flow import outflow as of
from woffl.flow import singlephase as sp
from woffl.flow.errors import ConvergenceError, JetPumpError, ThroatEntryNoSolution
from woffl.flow.jetplot import JetBook, ThroatEntryChoked
from woffl.geometry import JetPump, PipeInPipe, WellProfile
from woffl.pvt import FormWater, ResMix


def powerfluid_residual(
    qpf_guess: float,
    pte: float,
    ppf_surf: float,
    tpf_surf: float,
    dp_stat: float,
    jpump: JetPump,
    wellbore: PipeInPipe,
    wellprof: WellProfile,
    prop_pf: FormWater,
    flowpath: str,
) -> tuple[float, float, float]:
    """Power Fluid Residual

    Solve for the power fluid residual, which is the difference between the amount of power fluid
    it was guessed the jet pump needs and the amount actually delivered. This will be thrown in a
    secant loop to drive the power fluid residual to zero

    Args:
        qpf_guess (float): Power Fluid Flowrate Guess, BWPD
        pte (float): Throat Entry Pressure, psig
        ppf_surf (float): Pressure Power Fluid Surface, psig
        tpf_surf (float): Temp Power Fluid Surface, deg F
        dp_stat (float): Static Pressure, psi
        jpump (JetPump): Jet Pump Class
        wellbore (PipeInPipe): Wellbore Geometry of Tubing and Casing
        wellprof (WellProfile): Well Profile Class
        prop_pf (FormWater): Power Fluid Properties, assumed to be the same as formation water
        flowpath (str): Where the flow is occuring, either "tubing" or "annulus"

    Returns:
        qpf_residual (float): Power Fluid Residual, Guess - Calc, BWPD
        vnz (float): Nozzle Velocity, ft/s
        pni (float): Nozzle Inlet Pressure, psig
    """
    dp_fric = of.powerfluid_top_down_friction(
        ppf_surf, tpf_surf, qpf_guess, prop_pf, wellbore, wellprof, flowpath
    )
    pni = ppf_surf - dp_stat - dp_fric
    vnz = jf.nozzle_velocity(pni, pte, jpump.knz, prop_pf.density)
    _, qpf_calc = jf.nozzle_rate(vnz, jpump.anz)  # bwpd, power fluid flowrate
    return qpf_guess - qpf_calc, vnz, pni


def qpf_secant(qpf1: float, qpf2: float, res1: float, res2: float) -> float:
    """Power Fluid Secant Method

    Uses the secant method to calculate the next powerfluid to find where the pressure drop
    across the wellbore and jetpump nozzle are the same. This probably could be simplified using
    Newton method, but finding the derivative of the Serghide equation seemed daunting...

    Args:
        qpf1 (float): Power Fluid Flow One, BWPD
        qpf2 (float): Power Fluid Flow Two, BWPD
        res1 (float): Residual Power Fluid One, BWPD
        res2 (float): Residual Power Fluid Two, BWPD

    Return:
        qpf3 (float): Power Fluid Flow Three, BWPD
    """
    if res1 == res2:
        raise ConvergenceError(
            "power fluid secant stalled, equal residuals at successive flowrates"
        )
    qpf3 = qpf2 - res2 * (qpf1 - qpf2) / (res1 - res2)
    return qpf3


def discharge_residual(
    psu: float,
    pwh: float,
    tsu: float,
    ppf_surf: float,
    jpump: JetPump,
    wellbore: PipeInPipe,
    wellprof: WellProfile,
    ipr_su: InFlow,
    prop_su: ResMix,
    prop_pf: FormWater,
    jpump_direction: str = "reverse",
    *,
    te_seed: JetBook | None = None,
) -> tuple[float, float, float, float, float]:
    """Discharge Residual

    Solve for the jet pump discharge residual, which is the difference between discharge pressure
    calculated by the jetpump and the discharge pressure from the outflow. Also iterates on the
    power fluid rate to account for pressure drop as the power fluid goes to the jet pump.

    Args:
        psu (float): Pressure Suction, psig
        pwh (float): Pressure Wellhead, psig
        tsu (float): Temperature Suction, deg F
        ppf_surf (float): Pressure Power Fluid Surface, psig
        jpump (JetPump): Jet Pump Class
        wellbore (PipeInPipe): Wellbore Geometry of Tubing and Casing
        wellprof (WellProfile): Well Profile Class
        ipr_su (InFlow): Inflow Performance Class
        prop_su (ResMix): Reservoir Mixture Conditions
        prop_pf (FormWater): Power Fluid Properties, assumed to be the same as formation water
        jpump_direction (str): Jet Pump Direction, "forward" or "reverse" Circulating
        te_seed (JetBook | None): Optional unscaled throat-entry book already
            swept from this psu (``psu_minimize``'s); reused instead of
            re-sweeping when its first pressure is exactly ``psu``. Keyword-only.
            [LIBRARY change -> upstream PR to kwellis/woffl] SOLV-F9.

    Returns:
        res_di (float): Jet Pump Discharge minus Out Flow Discharge, psid
        qoil_std (float): Oil Rate, STBOPD
        fwat_bpd (float): Formation Water Rate, BWPD
        qnz_bpd (float): Power Fluid Rate, BWPD
        mach_te (float): Throat Entry Mach, unitless
    """
    direction_list = ["forward", "reverse"]
    if jpump_direction not in direction_list:
        raise ValueError(
            f"{jpump_direction} not recognized, select from {direction_list}"
        )

    if jpump_direction == "reverse":
        production_flowpath = "tubing"
        powerfluid_flowpath = "annulus"
    else:
        production_flowpath = "annulus"
        powerfluid_flowpath = "tubing"

    # merge from jetpump_base_calcs into here to allow for power fluid flow iteration
    qoil_std, te_book = jf.throat_entry_zero_tde(
        psu=psu,
        tsu=tsu,
        ken=jpump.ken,
        ate=jpump.ate,
        ipr_su=ipr_su,
        prop_su=prop_su,
        seed_book=te_seed,
    )
    pte, vte, rho_te, mach_te = te_book.dete_zero()

    # iterate on powerfluid flowrate to account for annular differential pressure
    dp_stat = sp.diff_press_static(
        prop_pf.density, -1 * wellprof.jetpump_vd
    )  # static power fluid pressure
    qpf_list = [2000.0, 3000.0]  # bwpd, power fluid flowrate guess at 1 and 2
    res_list = []  # power fluid residual list

    for qpf in qpf_list:
        qpf_res, vnz, pni = powerfluid_residual(
            qpf,
            pte,
            ppf_surf,
            tsu,
            dp_stat,
            jpump,
            wellbore,
            wellprof,
            prop_pf,
            powerfluid_flowpath,
        )
        res_list.append(qpf_res)

    n = 0
    while abs(res_list[-1]) > 5:
        qpf = qpf_secant(qpf_list[-2], qpf_list[-1], res_list[-2], res_list[-1])
        if not math.isfinite(qpf) or qpf <= 0:
            raise ConvergenceError(
                f"power fluid rate iteration produced a non-physical rate ({qpf})"
            )
        qpf_res, vnz, pni = powerfluid_residual(
            qpf,
            pte,
            ppf_surf,
            tsu,
            dp_stat,
            jpump,
            wellbore,
            wellprof,
            prop_pf,
            powerfluid_flowpath,
        )
        qpf_list.append(qpf)
        res_list.append(qpf_res)

        n += 1
        if n == 20:
            raise ConvergenceError("power fluid rate did not converge")

    # print(f"Iterated PowerFluid and Residual {dict(zip(qpf_list, res_list))}")
    # print(f"Frictional Loss: {pni + dp_stat - ppf_surf:.1f} psi")

    # should I do something with pni???
    qnz_bwpd = qpf_list[-1]
    wc_tm, fwat_bwpd = jf.throat_wc(qoil_std, prop_su.wc, qnz_bwpd)

    # Propagate water-pump mode into the throat mixture so a 100%-water solve
    # stays water-anchored through the diffuser discharge.
    # [LIBRARY change -> upstream PR to kwellis/woffl]
    prop_tm = ResMix(
        wc_tm,
        prop_su.fgor,
        prop_su.oil,
        prop_su.wat,
        prop_su.gas,
        model_as_water=prop_su.model_as_water,
    )
    ptm = jf.throat_discharge(
        pte,
        tsu,
        jpump.kth,
        vnz,
        jpump.anz,
        prop_pf.density,
        vte,
        jpump.ate,
        rho_te,
        prop_tm,
    )
    # Water-pump mode: prop_tm flows are WATER-anchored, so the anchor must
    # include the nozzle water — qoil_std alone sized the diffuser and tubing
    # traverse on formation water only, dropping the (often dominant)
    # power-fluid volume. Oil-path solves are bit-identical.
    # [LIBRARY change -> upstream PR to kwellis/woffl]
    qtm_std = jf._throat_mixture_anchor(
        qoil_std, qnz_bwpd, wc_tm, prop_su.model_as_water
    )

    # diffuser area is assumed to be the same as the tubing area, whether forward or reverse jet pump
    vtm, pdi_jp = jf.diffuser_discharge(
        ptm, tsu, jpump.kdi, jpump.ath, wellbore.inn_pipe.inn_area, qtm_std, prop_tm
    )

    # out flow section
    md_seg, prs_ray, slh_ray = of.production_top_down_press(
        pwh, tsu, qtm_std, prop_tm, wellbore, wellprof, production_flowpath
    )

    pdi_of = prs_ray[-1]  # discharge pressure outflow
    res_di = pdi_jp - pdi_of  # what the jetpump puts out vs what is required
    return res_di, qoil_std, fwat_bwpd, qnz_bwpd, mach_te


def _residual_walk_inward(
    psu_start: float,
    psu_toward: float,
    pwh: float,
    tsu: float,
    ppf_surf: float,
    jpump: JetPump,
    wellbore: PipeInPipe,
    wellprof: WellProfile,
    ipr_su: InFlow,
    prop_su: ResMix,
    prop_pf: FormWater,
    jpump_direction: str,
    *,
    te_seed: JetBook | None = None,
) -> tuple[float, float, tuple[float, float, float, float]]:
    """Discharge residual at ``psu_start``, walking inward if it's infeasible.

    The inner throat-mixture solve has no solution at some suction pressures for
    marginal pumps (small throat ratio + high water cut). When ``psu_start`` (a
    bracket endpoint) is such a point, the outer solve would historically abort.
    Instead, step the suction toward ``psu_toward`` and return the FIRST feasible
    suction's residual, so the outer search retains a valid, bracketed root — the
    well does flow inside the range.

    The probe fractions are dense near the endpoint (most rescues are only a few
    psi in) and extend to ~95% of the way to ``psu_toward`` — some pumps are
    feasible only in a thin band right against the far suction bound (e.g. a 13B
    at 99% water cut feasible only in the top ~13 psi below reservoir pressure),
    and a half-range walk would miss them.

    The first probe is ``psu_start`` itself, so a feasible endpoint is returned
    unchanged and already-converging solves stay bit-identical. The
    throat-mixture :class:`ConvergenceError` AND any other bare
    :class:`JetPumpError` (e.g. ``nozzle_velocity`` raising when ``pni <=
    pte`` — the nozzle can't overcome throat-entry pressure at this suction)
    are walked past — both mean "infeasible at this suction, try the next
    one". ``ThroatEntryNoSolution`` is the one exception that still propagates
    unchanged, since it drives the GUI's GOR auto-recovery.
    [P1-8 fix -> upstream PR to kwellis/woffl] Previously only ConvergenceError
    was caught, so a bare JetPumpError from nozzle_velocity escaped the
    fallback and aborted a solve this rescue exists for. Raises
    :class:`ConvergenceError` if no feasible suction is found within range.

    ``te_seed`` (keyword-only) is the throat-entry book already swept at
    ``psu_start`` (SOLV-F9); it is only ever reused for the endpoint probe.

    SOLV-F2 (review 2026-09-01): the probe grid is coarse - its gaps reach
    14 % of the span, ~140 psi on a 1,000 psi bracket - so the first feasible
    probe can overshoot the feasibility edge by a lot. When it does AND its
    residual already has the sign the FAR end owns (positive walking up from
    psu_min, negative walking down from psu_max), the root sits in the
    unprobed gap and returning the probe would report it as the operating
    point (the marginal 11A fixture: floor 1877.1, probe 1885.0, residual
    +35 psid, reported as "sonic" at Mach 0.50). So bisect on FEASIBILITY
    between the last infeasible and the first feasible fraction until the
    residual flips sign (the root is then bracketed) or the edge is pinned
    to within ``_EDGE_TOL_PSI`` - 2-4 extra residual evaluations. A feasible
    endpoint still returns from the very first probe, bit-identical.

    Returns ``(psu, residual, (qoil_std, fwat_bwpd, qnz_bwpd, mach_te))``.
    [LIBRARY change -> upstream PR to kwellis/woffl]
    """
    span = psu_toward - psu_start
    fracs = (
        0.0,
        0.005,
        0.01,
        0.02,
        0.04,
        0.07,
        0.11,
        0.16,
        0.22,
        0.30,
        0.40,
        0.52,
        0.66,
        0.82,
        0.95,
    )
    last_infeasible: float | None = None  # fraction of the latest infeasible probe
    for fr in fracs:
        psu = psu_start + span * fr
        try:
            res, qoil, fwat, qnz, mach = discharge_residual(
                psu,
                pwh,
                tsu,
                ppf_surf,
                jpump,
                wellbore,
                wellprof,
                ipr_su,
                prop_su,
                prop_pf,
                jpump_direction,
                te_seed=te_seed if fr == 0.0 else None,
            )
        except ThroatEntryChoked:
            # FLOW-5: below the throat-entry choke floor by more than the
            # sweep tolerance - the feasible region is ABOVE, step inward.
            # (Caught before its parent ThroatEntryNoSolution on purpose.)
            # [LIBRARY change -> upstream PR to kwellis/woffl]
            last_infeasible = fr
            continue
        except ThroatEntryNoSolution:
            raise  # drives the GUI's GOR auto-recovery — must propagate unchanged
        except JetPumpError:
            # infeasible throat mixture (ConvergenceError) OR nozzle inlet
            # pressure below throat entry (bare JetPumpError from
            # nozzle_velocity) at this suction — step inward.
            # [P1-8 fix -> upstream PR to kwellis/woffl]
            last_infeasible = fr
            continue
        if last_infeasible is not None and res * span > 0:
            # SOLV-F2: walked past the feasibility edge AND the residual
            # already carries the far end's sign - the root is in the gap.
            # [LIBRARY change -> upstream PR to kwellis/woffl]
            return _refine_feasibility_edge(
                psu_start,
                span,
                last_infeasible,
                fr,
                (psu, res, (qoil, fwat, qnz, mach)),
                pwh,
                tsu,
                ppf_surf,
                jpump,
                wellbore,
                wellprof,
                ipr_su,
                prop_su,
                prop_pf,
                jpump_direction,
            )
        return psu, res, (qoil, fwat, qnz, mach)
    raise ConvergenceError("no feasible suction pressure for the inner throat solve")


_EDGE_TOL_PSI = 2.0  # SOLV-F2: how close to the feasibility edge the bisection lands
_EDGE_MAX_EVALS = 8  # 140 psi gap / 2^7 < 2 psi, so this cap is never the binding limit


def _refine_feasibility_edge(
    psu_start: float,
    span: float,
    lo_fr: float,
    hi_fr: float,
    hi_result: tuple[float, float, tuple[float, float, float, float]],
    pwh: float,
    tsu: float,
    ppf_surf: float,
    jpump: JetPump,
    wellbore: PipeInPipe,
    wellprof: WellProfile,
    ipr_su: InFlow,
    prop_su: ResMix,
    prop_pf: FormWater,
    jpump_direction: str,
) -> tuple[float, float, tuple[float, float, float, float]]:
    """Bisect the suction between an infeasible and a feasible probe fraction.

    [LIBRARY change -> upstream PR to kwellis/woffl] SOLV-F2 helper for
    :func:`_residual_walk_inward`. ``lo_fr`` is the last INFEASIBLE fraction
    and ``hi_fr`` the first FEASIBLE one (``hi_result`` is its evaluation).
    Halves the gap on feasibility, keeping the feasible side, and stops as
    soon as a feasible point's residual has the sign that brackets the root
    against the far end (``res * span <= 0``) or the gap is within
    ``_EDGE_TOL_PSI``. Returns the feasible evaluation closest to the edge -
    when the residual never flips, the pump is pinned by throat-mixture
    feasibility, NOT by the throat-entry choke, and the caller must not call
    it sonic. ``ThroatEntryNoSolution`` propagates unchanged, as in the walk.
    """
    lo, hi = lo_fr, hi_fr
    best = hi_result
    for _ in range(_EDGE_MAX_EVALS):
        if abs(span) * (hi - lo) <= _EDGE_TOL_PSI:
            break
        mid = 0.5 * (lo + hi)
        psu = psu_start + span * mid
        try:
            res, qoil, fwat, qnz, mach = discharge_residual(
                psu,
                pwh,
                tsu,
                ppf_surf,
                jpump,
                wellbore,
                wellprof,
                ipr_su,
                prop_su,
                prop_pf,
                jpump_direction,
            )
        except ThroatEntryChoked:
            lo = mid
            continue
        except ThroatEntryNoSolution:
            raise
        except JetPumpError:
            lo = mid
            continue
        hi = mid
        best = (psu, res, (qoil, fwat, qnz, mach))
        if res * span <= 0:
            break  # root bracketed between here and the far end
    return best


def jetpump_solver(
    pwh: float,
    tsu: float,
    ppf_surf: float,
    jpump: JetPump,
    wellbore: PipeInPipe,
    wellprof: WellProfile,
    ipr_su: InFlow,
    prop_su: ResMix,
    prop_pf: FormWater,
    jpump_direction: str = "reverse",
    *,
    mach_crit: float = 1.0,
) -> tuple[float, bool, float, float, float, float]:
    """JetPump Solver

    Find a solution for the jetpump system. Takes into account the reservoir conditions, the wellhead
    pressure, tubing and annulus geometry. The solver will move along the psu and dEte curves until a
    solution is found that satisfies the outflow tubing, annulus and pump conditions.

    Args:
        pwh (float): Pressure Wellhead, psig
        tsu (float): Temperature Suction, deg F
        ppf_surf (float): Pressure Power Fluid Surface, psig
        jpump (JetPump): Jet Pump Class
        wellbore (PipeInPipe): Wellbore Geometry of Tubing and Casing
        wellprof (WellProfile): Well Profile Class
        ipr_su (InFlow): Inflow Performance Class
        prop_su (ResMix): Reservoir Mixture Conditions
        prop_pf (FormWater): Power Fluid Properties, assumed to be the same as formation water
        jpump_direction (str): Jet Pump Direction, "forward" or "reverse" Circulating
        mach_crit (float): Critical Mach number where the throat entry chokes,
            unitless. Default 1.0 (historic behavior).
            [LIBRARY change -> upstream PR to kwellis/woffl] calibratable
            choking threshold, keyword-only; forwarded to jf.psu_minimize.

    Returns:
        psu (float): Suction Pressure, psig
        sonic_status (boolean): Is the throat entry at sonic velocity?
        qoil_std (float): Oil Rate, STBOPD
        fwat_bwpd (float): Formation Water Rate, BWPD
        qnz_bwpd (float): Power Fluid Rate, BWPD
        mach_te (float): Throat Entry Mach, unitless
    """
    psu_min, qoil_std, te_book = jf.psu_minimize(
        tsu=tsu,
        ken=jpump.ken,
        ate=jpump.ate,
        ipr_su=ipr_su,
        prop_su=prop_su,
        mach_crit=mach_crit,
    )
    psu_floor = psu_min  # the Mach = mach_crit throat-entry choke floor
    psu_max = ipr_su.pres - 10  # max suction pressure that can be used

    # Lower-bracket residual. The inner throat-mixture solve can be infeasible
    # right AT psu_min for marginal pumps (small throat ratio + high water cut).
    # The well still flows — the discharge residual crosses zero just inside the
    # feasible suction range — but historically the whole solve ABORTED here
    # because discharge_residual raised at the endpoint. Walk psu inward to the
    # nearest feasible suction so the outer search keeps a valid bracket. When
    # the endpoint is already feasible (the overwhelming majority) the first
    # probe returns it unchanged, so converged solves stay bit-identical.
    # [LIBRARY change -> upstream PR to kwellis/woffl]
    #
    # SOLV-F9: psu_minimize already swept the throat entry at psu_min; hand
    # that book to the endpoint probe so discharge_residual does not sweep it
    # again (~10 % of a sonic solve). Only at the historic mach_crit = 1.0 -
    # the Mach-one walk scales kde by 1/mach_crit^2, so any other threshold
    # must re-sweep unscaled. [LIBRARY change -> upstream PR to kwellis/woffl]
    psu_min, res_min, (qoil_std, fwat_bwpd, qnz_bwpd, mach_te) = _residual_walk_inward(
        psu_min,
        psu_max,
        pwh,
        tsu,
        ppf_surf,
        jpump,
        wellbore,
        wellprof,
        ipr_su,
        prop_su,
        prop_pf,
        jpump_direction,
        te_seed=te_book if mach_crit == 1.0 else None,
    )

    # if the jetpump (available) discharge is above the outflow (required) discharge at lowest suction
    # the well will flow, but at its critical limit
    if res_min > 0:
        # [LIBRARY change -> upstream PR to kwellis/woffl] SOLV-F2: "sonic"
        # means the operating point IS the throat-entry choke floor psu_minimize
        # solved for (tde at Mach = mach_crit is zero there by construction).
        # When the walk had to move the suction inward and the residual is
        # still positive at the feasibility edge, the pump is pinned by
        # throat-MIXTURE feasibility at a subsonic entry (11A: Mach 0.50) and
        # must not be reported as choked - fric_calibration would refuse the
        # well as "pinned" and Header Impact would call it sonic-decoupled.
        # NOTE a mach_te threshold is deliberately NOT used: at the floor the
        # reported mach_te is the last SUBSONIC point of the 25-psi sweep
        # (E-41 9X/10X/11X read 0.89/0.89/0.88 while choked), so any
        # threshold loose enough for them would mislabel marginal pumps.
        # "At the floor" allows the walk's edge tolerance: a floor probe that
        # FLOW-5 rejected (its discrete clamp just over tolerance) is refined
        # to within _EDGE_TOL_PSI of the floor and is still the choke.
        sonic_status = bool(psu_min - psu_floor <= _EDGE_TOL_PSI)
        return psu_min, sonic_status, qoil_std, fwat_bwpd, qnz_bwpd, mach_te

    psu_max, res_max, _ = _residual_walk_inward(
        psu_max,
        psu_min,
        pwh,
        tsu,
        ppf_surf,
        jpump,
        wellbore,
        wellprof,
        ipr_su,
        prop_su,
        prop_pf,
        jpump_direction,
    )

    # if the jetpump (available) discharge is below the outflow (required) discharge at highest suction
    # the well will not flow, need to pick different parameters
    if res_max < 0:
        # this isn't actually a value error, the code is working as intended
        # this provides a quick fix in the try statement in batch run
        raise ValueError("well cannot lift at max suction pressure")

    psu_diff = 5  # converged when successive psu guesses are this close, psi
    res_tol = 10  # and the discharge residual is driven this close to zero, psid

    # The root is bracketed in [psu_min, psu_max] (res_min <= 0, res_max >= 0).
    # Primary attempt uses the original bracket-end pair so every solve that
    # already converged returns the identical psu/oil it did before (additive
    # change only). If that secant stalls on a marginal well, the robustness
    # path below re-seeds and/or bisects.
    try:
        return _secant_solve(
            (psu_min, psu_max),
            res_min,
            res_max,
            psu_min,
            psu_max,
            pwh,
            tsu,
            ppf_surf,
            jpump,
            wellbore,
            wellprof,
            ipr_su,
            prop_su,
            prop_pf,
            jpump_direction,
            psu_diff,
            res_tol,
        )
    except ConvergenceError:
        pass

    # Fallback 1 — re-seed the secant from the well's known/measured flowing BHP
    # (ipr_su.pwf, clamped into the bracket). For a producing well the system
    # root sits in that neighborhood, so starting the iteration there (paired
    # with the upper bracket end for the residual sign) lands on/near the real
    # answer where the bracket-spanning pair stalled. Skip if pwf collapses onto
    # a bracket end (the primary attempt already used that pair).
    psu_seed = min(max(ipr_su.pwf, psu_min), psu_max)
    if abs(psu_seed - psu_max) > psu_diff and abs(psu_seed - psu_min) > psu_diff:
        try:
            return _secant_solve(
                (psu_max, psu_seed),
                res_min,
                res_max,
                psu_min,
                psu_max,
                pwh,
                tsu,
                ppf_surf,
                jpump,
                wellbore,
                wellprof,
                ipr_su,
                prop_su,
                prop_pf,
                jpump_direction,
                psu_diff,
                res_tol,
            )
        except ConvergenceError:
            pass

    # Fallback 2 — robust bisection on the bracketed root. Guaranteed to converge
    # when the residual changes sign across [psu_min, psu_max]; only raise
    # ConvergenceError if the root is somehow not bracketed.
    if res_min * res_max <= 0:
        return _bisection_solve(
            psu_min,
            psu_max,
            res_min,
            res_max,
            pwh,
            tsu,
            ppf_surf,
            jpump,
            wellbore,
            wellprof,
            ipr_su,
            prop_su,
            prop_pf,
            jpump_direction,
            res_tol,
        )
    raise ConvergenceError("Suction Pressure for Overall System did not converge")


def _secant_solve(
    seed_pair: tuple[float, float],
    res_min: float,
    res_max: float,
    psu_min: float,
    psu_max: float,
    pwh: float,
    tsu: float,
    ppf_surf: float,
    jpump: JetPump,
    wellbore: PipeInPipe,
    wellprof: WellProfile,
    ipr_su: InFlow,
    prop_su: ResMix,
    prop_pf: FormWater,
    jpump_direction: str,
    psu_diff: float,
    res_tol: float,
) -> tuple[float, bool, float, float, float, float]:
    """Secant Hunt for the Discharge Residual Root

    Drives the discharge residual to zero by secant iteration on suction
    pressure, clamping every overshoot back into the bracketed [psu_min,
    psu_max]. Raises ConvergenceError if it does not settle within the cap so
    jetpump_solver can fall back to bisection.

    Args:
        seed_pair (tuple): Starting (psu, psu) pair for the secant, psig
        res_min (float): Discharge residual at psu_min, psid
        res_max (float): Discharge residual at psu_max, psid
        psu_min (float): Lower suction-pressure bracket, psig
        psu_max (float): Upper suction-pressure bracket, psig
        pwh (float): Pressure Wellhead, psig
        tsu (float): Temperature Suction, deg F
        ppf_surf (float): Pressure Power Fluid Surface, psig
        jpump (JetPump): Jet Pump Class
        wellbore (PipeInPipe): Wellbore Geometry of Tubing and Casing
        wellprof (WellProfile): Well Profile Class
        ipr_su (InFlow): Inflow Performance Class
        prop_su (ResMix): Reservoir Mixture Conditions
        prop_pf (FormWater): Power Fluid Properties
        jpump_direction (str): Jet Pump Direction, "forward" or "reverse"
        psu_diff (float): Successive-psu convergence criterion, psi
        res_tol (float): Discharge residual convergence criterion, psid

    Returns:
        Same tuple as jetpump_solver: (psu, sonic_status, qoil_std, fwat_bwpd,
        qnz_bwpd, mach_te)
    """
    psu_list = list(seed_pair)
    # match each seed psu to its already-known residual where possible so a
    # bracket-end seed reuses res_min / res_max instead of re-evaluating
    res_lookup = {psu_min: res_min, psu_max: res_max}
    res_list: list[float] = []
    qoil_std = fwat_bwpd = qnz_bwpd = mach_te = 0.0
    # [LIBRARY change -> upstream PR to kwellis/woffl] track which suction the
    # cached rates correspond to, so the returned rates can never be the stale
    # 0.0 initializers (see the final guard before return).
    rates_at = None
    # [LIBRARY change -> upstream PR to kwellis/woffl] SOLV-P3: every suction
    # evaluated so far -> (residual, rates or None). discharge_residual is a
    # pure function of psu, so a clamp that lands the iterate back on a point
    # already evaluated (typically a bracket end) is served from here instead
    # of paying ~5 ms to recompute the identical numbers. Bit-identical by
    # construction. Guarded by TestSecantSolveReusesKnownPoints.
    known: dict[float, tuple[float, tuple[float, float, float, float] | None]] = {
        psu_min: (res_min, None),
        psu_max: (res_max, None),
    }
    for psu in psu_list:
        if psu in res_lookup:
            res_list.append(res_lookup[psu])
        else:
            res_seed, qoil_std, fwat_bwpd, qnz_bwpd, mach_te = discharge_residual(
                psu,
                pwh,
                tsu,
                ppf_surf,
                jpump,
                wellbore,
                wellprof,
                ipr_su,
                prop_su,
                prop_pf,
                jpump_direction,
            )
            res_list.append(res_seed)
            rates_at = psu
            known[psu] = (res_seed, (qoil_std, fwat_bwpd, qnz_bwpd, mach_te))

    n = 0  # loop counter
    while abs(psu_list[-2] - psu_list[-1]) > psu_diff or abs(res_list[-1]) > res_tol:
        # secant on the LAST TWO iterates; the root stays bracketed between
        # psu_min (res <= 0) and psu_max (res >= 0), so clamp overshoots back in
        psu_nxt = jf.psu_secant(psu_list[-2], psu_list[-1], res_list[-2], res_list[-1])
        psu_nxt = min(max(psu_nxt, psu_min), psu_max)
        if psu_nxt == psu_list[-1] and abs(res_list[-1]) > res_tol:
            # SOLV-P3: the clamp pinned the iterate on the point it is already
            # standing on. Re-evaluating returns the same residual, the loop
            # cannot exit (|res| > res_tol) and the next psu_secant raises on
            # equal residuals - so raise now and hand off to the fallbacks with
            # the identical outcome, minus the wasted evaluation.
            raise ConvergenceError(
                "Suction Pressure for Overall System did not converge "
                "(secant pinned on a bracket end)"
            )
        hit = known.get(psu_nxt)
        if hit is not None:
            res_nxt, rates = hit
            if rates is not None:
                qoil_std, fwat_bwpd, qnz_bwpd, mach_te = rates
                rates_at = psu_nxt
        else:
            res_nxt, qoil_std, fwat_bwpd, qnz_bwpd, mach_te = discharge_residual(
                psu_nxt,
                pwh,
                tsu,
                ppf_surf,
                jpump,
                wellbore,
                wellprof,
                ipr_su,
                prop_su,
                prop_pf,
                jpump_direction,
            )
            rates_at = psu_nxt
            known[psu_nxt] = (res_nxt, (qoil_std, fwat_bwpd, qnz_bwpd, mach_te))
        psu_list.append(psu_nxt)
        res_list.append(res_nxt)
        n += 1
        if n == 20:
            raise ConvergenceError(
                "Suction Pressure for Overall System did not converge"
            )
    # [LIBRARY change -> upstream PR to kwellis/woffl] Ensure the returned rates
    # correspond to the returned suction. When BOTH seeds are bracket ends (their
    # residuals come from res_lookup, so discharge_residual is never called for
    # them) AND the secant loop body never runs — a thin feasible band where the
    # bracket already satisfies psu_diff and res_tol — the rates would otherwise
    # stay at their 0.0 initializers, reporting a real flowing well as 0 BOPD
    # while returning normally (no ConvergenceError, so no fallback fires).
    # Evaluate once at the final psu in that case. In the normal converging path
    # rates_at already equals psu_final, so this is a no-op (bit-identical).
    psu_final = psu_list[-1]
    if rates_at != psu_final:
        _res, qoil_std, fwat_bwpd, qnz_bwpd, mach_te = discharge_residual(
            psu_final,
            pwh,
            tsu,
            ppf_surf,
            jpump,
            wellbore,
            wellprof,
            ipr_su,
            prop_su,
            prop_pf,
            jpump_direction,
        )
    return psu_final, False, qoil_std, fwat_bwpd, qnz_bwpd, mach_te


def _bisection_solve(
    psu_lo: float,
    psu_hi: float,
    res_lo: float,
    res_hi: float,
    pwh: float,
    tsu: float,
    ppf_surf: float,
    jpump: JetPump,
    wellbore: PipeInPipe,
    wellprof: WellProfile,
    ipr_su: InFlow,
    prop_su: ResMix,
    prop_pf: FormWater,
    jpump_direction: str,
    res_tol: float,
) -> tuple[float, bool, float, float, float, float]:
    """Bisection Fallback for the Discharge Residual Root

    Guaranteed-convergence fallback for jetpump_solver's suction-pressure solve.
    The discharge residual is bracketed on [psu_lo, psu_hi] (res_lo <= 0 at the
    sonic-choke floor, res_hi >= 0 at psu_max), so a bisection always narrows in
    on the root even when the secant overshoots/stalls on a marginal well.

    Args:
        psu_lo (float): Lower suction-pressure bracket (psu_min), psig
        psu_hi (float): Upper suction-pressure bracket (psu_max), psig
        res_lo (float): Discharge residual at psu_lo (<= 0), psid
        res_hi (float): Discharge residual at psu_hi (>= 0), psid
        pwh (float): Pressure Wellhead, psig
        tsu (float): Temperature Suction, deg F
        ppf_surf (float): Pressure Power Fluid Surface, psig
        jpump (JetPump): Jet Pump Class
        wellbore (PipeInPipe): Wellbore Geometry of Tubing and Casing
        wellprof (WellProfile): Well Profile Class
        ipr_su (InFlow): Inflow Performance Class
        prop_su (ResMix): Reservoir Mixture Conditions
        prop_pf (FormWater): Power Fluid Properties
        jpump_direction (str): Jet Pump Direction, "forward" or "reverse"
        res_tol (float): Residual tolerance to stop on, psid

    Returns:
        Same tuple as jetpump_solver: (psu, sonic_status, qoil_std, fwat_bwpd,
        qnz_bwpd, mach_te)
    """
    # orient so lo carries the negative residual and hi the positive one
    if res_lo > res_hi:
        psu_lo, psu_hi = psu_hi, psu_lo
        res_lo, res_hi = res_hi, res_lo

    psu_diff = 1.5  # tighter than the secant's 5 psi since bisection is cheap-ish
    psu_mid = (psu_lo + psu_hi) / 2
    # carry the most recent evaluation out so the final return uses the mid point
    qoil_std = fwat_bwpd = qnz_bwpd = mach_te = 0.0
    # [LIBRARY change -> upstream PR to kwellis/woffl] SOLV-F3: which suction
    # the cached rates belong to (mirrors _secant_solve). An infeasible probe
    # leaves the rates at the PREVIOUS midpoint, so on a width exit the
    # returned psu and rates could belong to different suctions (probe:
    # psu 469.5 with the rates of 470.3); the guard before return fixes that.
    rates_at: float | None = None
    try:
        res_mid, qoil_std, fwat_bwpd, qnz_bwpd, mach_te = discharge_residual(
            psu_mid,
            pwh,
            tsu,
            ppf_surf,
            jpump,
            wellbore,
            wellprof,
            ipr_su,
            prop_su,
            prop_pf,
            jpump_direction,
        )
        rates_at = psu_mid
    except JetPumpError:
        # SOLV-F3: the FIRST midpoint used to be evaluated outside the try,
        # so an infeasible first probe escaped as "solver failed" on a well
        # whose root is bracketed. Same too-low-side treatment as the loop.
        # [LIBRARY change -> upstream PR to kwellis/woffl]
        res_mid = -(res_tol + 1.0)

    n = 0
    while (psu_hi - psu_lo) / 2 > psu_diff and abs(res_mid) > res_tol:
        if res_mid < 0:
            psu_lo = psu_mid
        else:
            psu_hi = psu_mid
        psu_mid = (psu_lo + psu_hi) / 2
        try:
            res_mid, qoil_std, fwat_bwpd, qnz_bwpd, mach_te = discharge_residual(
                psu_mid,
                pwh,
                tsu,
                ppf_surf,
                jpump,
                wellbore,
                wellprof,
                ipr_su,
                prop_su,
                prop_pf,
                jpump_direction,
            )
            rates_at = psu_mid
        except JetPumpError:
            # Shouldn't occur inside [psu_min, psu_max] (psu_min is the
            # feasibility floor), but if a probe point has no throat-entry
            # solution (ThroatEntryNoSolution) it is below the floor — the
            # too-low side — so the root lies ABOVE it. [LIBRARY change ->
            # upstream PR to kwellis/woffl] Mark res_mid as definitively
            # negative so the NEXT iteration narrows psu_lo = psu_mid
            # consistently. Previously res_mid was left STALE (from an
            # earlier, different suction) and psu_mid was recomputed only to
            # be overwritten at the top of the loop — a possibly-wrong
            # side-pick. The bracket still shrank, so the final answer
            # converged; this just makes the narrowing direction correct.
            #
            # [P1-8 fix -> upstream PR to kwellis/woffl] Broadened from
            # ``except ThroatEntryNoSolution`` to the ``JetPumpError`` family
            # so a throat-mixture ConvergenceError OR nozzle_velocity's bare
            # JetPumpError (pni <= pte) at an interior probe get the same
            # "infeasible, too-low-side" treatment instead of escaping the
            # bisection fallback uncaught.
            res_mid = -(res_tol + 1.0)
        n += 1
        if n == 60:
            raise ConvergenceError(
                "Suction Pressure for Overall System did not converge (bisection)"
            )
    # [LIBRARY change -> upstream PR to kwellis/woffl] SOLV-F3: the returned
    # rates must come from the returned suction. On the normal path rates_at
    # already equals psu_mid (no-op, bit-identical). If the last probe was
    # infeasible, psu_mid itself has no throat solution - it sits on the
    # too-low side - so the answer is the feasible upper end psu_hi (its
    # residual is >= 0 from a real evaluation and the bracket is already
    # within 2 * psu_diff). Guarded by TestBisectionSolveConsistency.
    if rates_at != psu_mid:
        try:
            _res, qoil_std, fwat_bwpd, qnz_bwpd, mach_te = discharge_residual(
                psu_mid,
                pwh,
                tsu,
                ppf_surf,
                jpump,
                wellbore,
                wellprof,
                ipr_su,
                prop_su,
                prop_pf,
                jpump_direction,
            )
        except JetPumpError:
            psu_mid = psu_hi
            _res, qoil_std, fwat_bwpd, qnz_bwpd, mach_te = discharge_residual(
                psu_hi,
                pwh,
                tsu,
                ppf_surf,
                jpump,
                wellbore,
                wellprof,
                ipr_su,
                prop_su,
                prop_pf,
                jpump_direction,
            )
    return psu_mid, False, qoil_std, fwat_bwpd, qnz_bwpd, mach_te
