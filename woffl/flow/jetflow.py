"""Jet Flow Equations

Functions that are used for solving the fluid dynamics inside a jet pump. The actual
jet pump geometry is accomplished in a seperate module.
"""

from __future__ import annotations  # jetplot <-> jetflow import cycle: defer annotations

import math
from copy import deepcopy

import numpy as np
from scipy.integrate import trapezoid

from woffl.flow import jetplot as jp
from woffl.flow import singlephase as sp
from woffl.flow.errors import ConvergenceError, JetPumpError, ThroatEntryNoSolution
from woffl.flow.inflow import InFlow
from woffl.pvt.resmix import ResMix
from woffl.pvt.formwat import FormWater


# [LIBRARY change -> upstream PR to kwellis/woffl]
def water_nozzle(pni, pte, temp, knz, anz, prop_pf):
    """Water nozzle: (exit velocity ft/s, standard BWPD), consistent PVT work."""
    if pni <= pte:
        raise ThroatEntryNoSolution("nozzle inlet pressure must exceed throat entry")
    velocity = math.sqrt(2*prop_pf.pressure_work(pte, pni, temp)/(1+knz))
    prop_pf.condition(pte, temp)
    standard_rate = sp.ft3s_to_bpd(velocity*anz) * prop_pf.density/prop_pf.density_std
    return velocity, standard_rate


def throat_mixture(qoil_std, qnz_std, prop_su, prop_pf):
    """Mix formation and lift water while conserving their standard masses."""
    wc, fwat = throat_wc(qoil_std, prop_su.wc, qnz_std)
    water_total = fwat+qnz_std
    sg = ((fwat*prop_su.wat.wat_sg + qnz_std*prop_pf.wat_sg)/water_total
          if water_total > 0 else prop_su.wat.wat_sg)
    mix = ResMix(wc, prop_su.fgor, deepcopy(prop_su.oil), FormWater(sg),
                 deepcopy(prop_su.gas), model_as_water=prop_su.model_as_water)
    return wc, fwat, mix


def enterance_ke(ken: float, vte: float) -> float:
    """Throat Enterance Kinetic Energy

    Calculate the kinetic energy in the throat enterance.
    Add the energy lost due to friction to balance out with EE.

    Args:
        ken: Throat Entry Friction Factor, unitless
        vte: Velocity at Throat Entry, ft/s

    Returns:
        ke_te: Throat Enterance Kinetic Energy, ft2/s2
    """
    ke_te = (1 + ken) * (vte**2) / 2
    return ke_te


def incremental_ee(prs_ray: np.ndarray, rho_ray: np.ndarray) -> float:
    """Fluid Incremental Expansion Energy

    Calculate the incremental change in expansion energy for a fluid over a
    pressure change. Uses the trapezoid rule to sum the area under the density
    curve for a difference in pressure. Equal to Sum (dp/ρ). The length of the
    arrays must match and be equal or greater than length 2.

    Args:
        prs_ray (np.ndarray): Array of pressures, psig
        rho_ray (np.ndarray): Array of densitys, lbm/ft3

    Returns:
        ee_inc (float): Expansion Energy Incremental, ft2/s2
    """
    ee_inc = trapezoid(1 / rho_ray, 144 * 32.174 * prs_ray)
    return ee_inc


# [LIBRARY change -> upstream PR to kwellis/woffl] Entry energy v1:
# Both public walks use the same physical kinetic term and pressure integral.
# Function names and the deprecated Mach argument remain for source compatibility.
def throat_entry_zero_tde(
    psu: float, tsu: float, ken: float, ate: float, ipr_su: InFlow,
    prop_su: ResMix, *, seed_book: jp.JetBook | None = None,
) -> tuple[float, jp.JetBook]:
    """Operating throat-entry energy book (psig, degF, ft2, oil STB/day).

    The reachable energy minimum bounds the operating root. Wood Mach is
    diagnostic. Seed books are not trusted without all input provenance;
    the scoped material path supplies reuse instead.
    """
    from woffl.flow.entry_energy import entry_book
    return entry_book(psu, tsu, ken, ate, ipr_su, prop_su)


def throat_entry_mach_one(
    psu: float, tsu: float, ken: float, ate: float, ipr_su: InFlow,
    prop_su: ResMix, mach_crit: float = 1.0,
) -> tuple[float, float, jp.JetBook]:
    """Energy at the reachable entry limit (ft2/s2), rate and shared book.

    Despite the historical name, the limit is an energy turning point or
    pressure bound. mach_crit is retired and cannot scale energy or flow.
    """
    from woffl.flow.entry_energy import retired_mach
    retired_mach(mach_crit)
    rate, book = throat_entry_zero_tde(psu, tsu, ken, ate, ipr_su, prop_su)
    return book.minimum_energy, rate, book


def psu_minimize(
    tsu: float, ken: float, ate: float, ipr_su: InFlow,
    prop_su: ResMix, mach_crit: float = 1.0,
) -> tuple[float, float, jp.JetBook]:
    """Lowest feasible suction (psig), oil (STB/day), and shared entry book.

    Solves zero energy at the FIRST minimum reachable from suction. A
    pressure-bound limit carries that reason and must not be called sonic.
    mach_crit is a deprecated compatibility argument with no effect.
    """
    from woffl.flow.entry_energy import retired_mach, suction_limit
    retired_mach(mach_crit)
    return suction_limit(tsu, ken, ate, ipr_su, prop_su)


def psu_secant(psu1: float, psu2: float, dete1: float, dete2: float) -> float:
    """Next Suction Pressure with Secant Method

    Uses the secant method to calculate the next psu to use to find a zero dEte at Ma = 1.

    Args:
        psu1 (float): Suction Pressure One, psig
        psu2 (float): Suction Pressure Two, psig
        dete1 (float): Differential Energy at psu1 and Ma = 1, ft2/s2
        dete2 (float): Differential Energy at psu1 and Ma = 1, ft2/s2

    Return:
        psu3 (float): Suction Pressure Three, psig
    """
    if dete1 == dete2:
        raise ConvergenceError(
            "psu secant stalled, equal residuals at successive suction pressures"
        )
    psu3 = psu2 - dete2 * (psu1 - psu2) / (dete1 - dete2)
    return psu3


def ptm_secant(ptm1: float, ptm2: float, bal1: float, bal2: float) -> float:
    """Next Throat Pressure with Secant Method

    Uses the secant method to calculate the next ptm to use to find a zero momentum balance.

    Args:
        ptm1 (float): Throat Pressure One, psig
        ptm2 (float): Throat Pressure Two, psig
        bal1 (float): Throat Momentum Balance at One, psig
        bal2 (float): Throat Momentum Balance at Two, psig

    Return:
        ptm3 (float): Throat Pressure Three, psig
    """
    if bal1 == bal2:
        raise ConvergenceError(
            "ptm secant stalled, equal residuals at successive throat pressures"
        )
    ptm3 = ptm2 - bal2 * (ptm1 - ptm2) / (bal1 - bal2)
    return ptm3


def nozzle_velocity(pni: float, pte: float, knz: float, rho_nz: float) -> float:
    """Nozzle Velocity

    Solve Bernoulli's Equation to calculate the nozzle velocity in ft/s.

    Args:
        pni (float): Nozzle Inlet Pressure, psig
        pte (float): Throat Entry Pressure, psig
        knz (float): Friction of Nozzle, unitless
        rho_nz (float): Density of Nozzle Fluid, lbm/ft3

    Returns:
        vnz (float): Nozzle Velocity, ft/s
    """
    if pni <= pte:
        raise JetPumpError(
            f"Nozzle inlet pressure {pni:.0f} psig is below throat entry pressure "
            f"{pte:.0f} psig, power fluid cannot flow"
        )
    vnz = math.sqrt(2 * 32.174 * 144 * (pni - pte) / (rho_nz * (1 + knz)))
    return vnz


def nozzle_rate(vnz: float, anz: float) -> tuple[float, float]:
    """Nozzle Flow Rate

    Find Nozzle / Power Fluid Flowrate in ft3/s and BPD

    Args:
        vnz (float): Nozzle Velocity, ft/s
        anz (float): Area of Nozzle, ft2

    Returns:
        qnz_ft3s (float): Nozzle Flowrate ft3/s
        qnz_bpd (float): Nozzle Flowrate bpd
    """
    qnz_ft3s = anz * vnz
    qnz_bpd = sp.ft3s_to_bpd(qnz_ft3s)
    return qnz_ft3s, qnz_bpd


def throat_inlet_momentum(
    vnz: float, anz: float, rho_nz: float, vte: float, ate: float, rho_te: float
) -> tuple[float, float]:
    """Throat Inlet Momentum

    Calculate the inlet momentum of the throat in lbm/(s2*ft). The units
    are actually pressure, but referred to as momentum to help differentiate.

    Args:
        vnz (float): Velocity of Nozzle, ft/s
        anz (float): Area of Nozzle, ft2
        rho_nz (float): Density of Nozzle Fluid, lbm/ft3
        vte (float): Velocity of Throat Entry Mixture, ft/s
        ate (float): Area of Throat Entry, ft2
        rho_te (float): Density of Throat Entry Mixture, lbm/ft3

    Returns:
        mom_nz (float): Nozzle Momentum, lbm*ft/s2
        mom_te (float): Entry Momentum, lbm*ft/s2
    """
    mom_nz = sp.momentum(rho_nz, vnz, anz)  # momentum of the nozzle
    mom_te = sp.momentum(rho_te, vte, ate)  # momentum of the throat entry
    return mom_nz, mom_te


def throat_outlet_momentum(
    kth: float, vtm: float, ath: float, rho_tm: float
) -> tuple[float, float]:
    """Throat Outlet Momentum

    Calculate the outlet momentum of the throat in lbm/(s2*ft). The units
    are actually pressure, but referred to as momentum to help differentiate.

    Args:
        kth (float): Throat Friction Factor, unitless
        vtm (float): Velocity of Throat Mixture, ft/s
        ath (float): Area of the Throat, ft2
        rho_tm (float): Density of Throat Mixture, lbm/ft3

    Returns:
        mom_tm (float): Throat Mixting Momentum, lbm*ft/s2
        mom_fr (float): Throat Friction Momemum, lbm*ft/s2
    """
    mom_tm = sp.momentum(rho_tm, vtm, ath)
    mom_fr = (
        1 / 2 * kth * mom_tm
    )  # this is accurate, it is lumped into the mom to psi equation
    return mom_tm, mom_fr


def jet_velocity_head(vnz: float, rho_nz: float) -> float:
    """Cunningham Jet Velocity Head

    A parameter that Cunningham calculates in his papers called Jet Velocity Head
    The abbreviation used to denote this is typicall a capital Z. It can be thought
    as the pressure energy the jet is imparting on the fluid flowing in. Cunningham
    typically works in units of lbf/ft2 instead of lbf/in2

    Args:
        vnz (float): Velocity of Nozzle, ft/s
        rho_nz (float): Density of Nozzle Fluid, lbm/ft3

    Returns:
        cunn_z (float): Z Jet Velocity Head, lbf/in2
    """
    cunn_z = rho_nz * vnz**2 / (2 * 32.174 * 144)
    return cunn_z


def jet_pump_number(pte: float, cunn_z: float, anz: float, ate: float) -> float:
    """Cunningham Jet Pump Number

    A dimensionless number that Cunningham uses in the Liquid Jet Gas Compressor Paper
    Used to graph performance of the jet pump across the throat. The abbreviation in
    the Cunningham LJG paper is n

    Args:
        pte (float): Pressure of Throat Entry, psig
        cunn_z (float): Cunningham Z Jet Velocity Head, lbf/in2
        anz (float): Area of the nozzle, ft2
        ate (float): Area of the throat entry, ft2

    Returns:
        cunn_n (float): n Jet Pump Number, dimensionless
    """
    b = anz / (anz + ate)  # cunningham nozzle to throat ratio
    c = ate / anz  # cunningham area ratio
    cunn_n = 2 * cunn_z * c * b**2 / pte
    return cunn_n


def jet_velocity_ratio(vnz: float, vte: float) -> float:
    """Cunningham Jet Velocity Ratio

    Dimensionless Number used in Cunningham Paper on Liquid Jet Gas Compressors
    Used to estimate the throat exit pressure. The abbreviation in the Cunningham
    LJG paper is lower case vu.

    Args:
        vnz (float): Velocity of Nozzle, ft/s
        vte (float): Velocity of Throat Entry, ft/s

    Returns:
        cunn_v (float): v Jet Velocity Ratio, dimensionless
    """
    return vte / vnz


def throat_pressure_ratio(cunn_n: float, cunn_v: float) -> tuple[float, float]:
    """Cunningham Throat Pressure Ratio

    Dimensionless Number used in Cunningham Paper on Liquid Jet Gas Compressors
    Used to provide a theoretical pressure exit to inlet ratio, denoted as rto.
    These values will be used as a potential starting point for a throat exit pressure
    guess value. The value will then be refined with non-ideal iteration.

    Args:
        cunn_n (float): Cunningham Jet Pump Number, dimensionless
        cunn_v (float): Cunningham Jet Velocity Ratio, dimensionless

    Returns:
        rto_pos (float): Positive Throat Pressure Ratio, dimensionless
        rto_neg (float): Positive Throat Pressure Ratio, dimensionless
    """
    rto_pos = (1 + cunn_n + math.sqrt((1 + cunn_n) ** 2 - 4 * cunn_n * cunn_v)) / 2
    rto_neg = (1 + cunn_n - math.sqrt((1 + cunn_n) ** 2 - 4 * cunn_n * cunn_v)) / 2
    return rto_pos, rto_neg


def throat_momentum_balance(
    pte: float,
    ptm: float,
    mom_nz: float,
    mom_te: float,
    mom_tm: float,
    mom_fr: float,
    ath: float,
) -> float:
    """Throat Momentum Balance

    Momentum balance across the throat that should equal zero for the correct
    term of ptm. The output of this equation can be fed to a secant solver to
    calculate the next best guess of ptm. Hopefully this is more robust method.

    Args:
        pte (float): Pressure of Throat Entry, psig
        ptm (float): Throat Mixture Pressure, psig
        mom_nz (float): Nozzle Momentum, lbm*ft/s2
        mom_te (float): Entry Momentum, lbm*ft/s2
        mom_tm (float): Throat Mixting Momentum, lbm*ft/s2
        mom_fr (float): Throat Friction Momemum, lbm*ft/s2
        ath (float): Area of the Throat, ft2

    Returns:
        mom_bal (float): Balanced pressure or momentume, psig"""
    mom_in = sp.mom_to_psi(mom_nz + mom_te, ath)
    mom_out = sp.mom_to_psi(mom_fr + mom_tm, ath)
    mom_bal = pte + mom_in - ptm - mom_out
    return mom_bal


def throat_discharge(
    pte: float,
    tte: float,
    kth: float,
    vnz: float,
    anz: float,
    rho_nz: float,
    vte: float,
    ate: float,
    rho_te: float,
    prop_tm: ResMix,
):
    """Throat Discharge Pressure

    Solves the throat mixture equation of the jet pump. Calculates throat differntial pressure.
    Use the throat entry pressure and differential pressure to calculate throat mix pressure.
    Account for the discharge pressure is greater than the inlet pressure. Loops through the
    calculated discharge pressure until a converged answer occurs.

    Args:
        pte (float): Pressure of Throat Entry, psig
        tte (float): Temperature of Throat Entry, deg F
        kth (float): Friction of Throat Mix, Unitless
        vnz (float): Velocity of Nozzle, ft/s
        anz (float): Area of Nozzle, ft2
        rho_nz (float): Density of Nozzle Fluid, lbm/ft3
        vte (float): Velocity of Throat Entry Mixture, ft/s
        ate (float): Area of Throat Entry, ft2
        rho_te (float): Density of Throat Entry Mixture, lbm/ft3
        prop_tm (ResMix): Properties of the Throat Mixture

    Returns:
        ptm (float): Throat Discharge Pressure, psig
    """
    mom_nz, mom_te = throat_inlet_momentum(vnz, anz, rho_nz, vte, ate, rho_te)
    mnz = sp.massflow(rho_nz, vnz, anz)  # mass flow of the nozzle
    mte = sp.massflow(rho_te, vte, ate)  # mass flow of the throat entry
    ath = anz + ate  # area of the throat
    mtm = mnz + mte  # mass flow of total mixture

    # [LIBRARY change -> upstream PR to kwellis/woffl] Outlet momentum is
    # positive, so pressure cannot exceed inlet pressure plus inlet momentum.
    # Bound secant excursions before evaluating the finite-domain water PVT.
    from woffl.pvt.water_properties import PSI_MPA
    upper = min(pte + sp.mom_to_psi(mom_nz+mom_te, ath), 100/PSI_MPA-14.7)
    ptm_list = [min(3*pte, upper), min(2*pte, upper)]
    if ptm_list[0] == ptm_list[1]:
        ptm_list[1] = (15.+upper)/2
    bal_list = []

    # generate the first two guesses to work off of
    for ptm in ptm_list:
        rho_tm = prop_tm.condition(ptm, tte).rho_mix()  # density of total mixture
        vtm = sp.velocity(mtm / rho_tm, ath)
        mom_tm, mom_fr = throat_outlet_momentum(kth, vtm, ath, rho_tm)
        mom_bal = throat_momentum_balance(pte, ptm, mom_nz, mom_te, mom_tm, mom_fr, ath)
        bal_list.append(mom_bal)

    # converge on the LATEST residual so the returned ptm is the validated one
    # (checking bal_list[-2] returned a point whose own residual was never tested)
    n = 0
    while abs(bal_list[-1]) > 1:  # attempt to find ptm convergence
        ptm = min(upper, max(
            ptm_secant(ptm_list[-2], ptm_list[-1], bal_list[-2], bal_list[-1]), 15
        ))  # force ptm to never go below 15 psig

        rho_tm = prop_tm.condition(ptm, tte).rho_mix()  # density of total mixture
        vtm = sp.velocity(mtm / rho_tm, ath)
        mom_tm, mom_fr = throat_outlet_momentum(kth, vtm, ath, rho_tm)
        mom_bal = throat_momentum_balance(pte, ptm, mom_nz, mom_te, mom_tm, mom_fr, ath)

        ptm_list.append(ptm)
        bal_list.append(mom_bal)

        n += 1
        if n == 15:
            # Secant stalled — it oscillates on the compressible throat mixture
            # for marginal configs (small throat ratio + high water cut), where
            # rho_tm(ptm) is strongly nonlinear near the bubble point. Fall back
            # to a bracketed root-find on the SAME momentum-balance residual,
            # which is guaranteed to converge whenever the balance changes sign
            # over ptm (it virtually always does — the -ptm term dominates at
            # high pressure). Mirrors the solopump psu bisection fallback; the
            # secant fast-path above is untouched, so already-converging cases
            # are bit-identical. [LIBRARY change -> upstream PR to kwellis/woffl]
            def _bal(p):
                p = max(p, 15.0)
                rho = prop_tm.condition(p, tte).rho_mix()
                v = sp.velocity(mtm / rho, ath)
                m_tm, m_fr = throat_outlet_momentum(kth, v, ath, rho)
                return throat_momentum_balance(pte, p, mom_nz, mom_te, m_tm, m_fr, ath)

            return _throat_discharge_bracketed(_bal, pte, upper=upper)

    return ptm_list[-1]


def _throat_discharge_bracketed(bal_fn, pte: float, *, upper=None) -> float:
    """Bracketed fallback for :func:`throat_discharge` when the secant stalls.

    Scans the throat-discharge pressure ``ptm`` over a generous range for a sign
    change in the momentum-balance residual, then refines with Brent's method.
    Returns the validated ``ptm`` root. Raises :class:`ConvergenceError` only if
    no sign change exists in range (a genuinely non-physical configuration), so
    callers' existing failure handling is preserved.

    Root selection: the residual generically has TWO roots — the balance falls
    to -inf at both ends (gas expansion blows up the outlet momentum at low
    ptm; the -ptm term dominates at high ptm) with a positive hump between.
    The LOW root is the non-physical/choked branch; the HIGH root is the
    working discharge (the secant fast path, seeded at 2-3x pte, converges to
    it). Scan DOWNWARD from the top so the first bracketed sign change is the
    physical high root — the original upward scan locked onto the low root and
    reported an understated ptm/pdi, i.e. a false "pump can't lift" on exactly
    the marginal wells this fallback exists to save.
    [LIBRARY change -> upstream PR to kwellis/woffl]
    """
    from scipy.optimize import brentq

    lo = 15.0
    # The discharge sits a little above the throat-entry pressure for a working
    # pump (initial secant guesses were 2*pte, 3*pte); scan well past that.
    for hi in (max(6.0 * pte, 300.0), max(15.0 * pte, 1500.0)):
        hi = min(hi, upper) if upper is not None else hi
        grid = np.linspace(lo, hi, 60)
        prev_p = grid[-1]
        prev_v = bal_fn(prev_p)
        if prev_v > 0.0:
            # Still inside the positive hump — the high root is ABOVE this hi.
            # Expand the range rather than walk down into the low root.
            continue
        for p in grid[-2::-1]:
            v = bal_fn(p)
            if prev_v == 0.0:
                return float(prev_p)
            if (prev_v < 0.0) != (v < 0.0):  # sign change -> root bracketed
                return float(
                    brentq(bal_fn, p, prev_p, xtol=1e-2, rtol=1e-8, maxiter=200)
                )
            prev_p, prev_v = p, v

    # [LIBRARY change -> upstream PR to kwellis/woffl] FLOW-6: the 60-point
    # scan (step ~0.1 pte) steps clean over a positive hump narrower than one
    # step - exactly the marginal pumps this fallback exists for - and used to
    # raise here. Locate the hump's peak with a bounded scalar maximization
    # (~20 evaluations); if the peak is positive the physical HIGH root lies in
    # [peak, hi] where the balance is negative, and Brent finishes it.
    # Guarded by tests/test_jetflow_bracketed.py.
    from scipy.optimize import minimize_scalar

    hi = max(15.0 * pte, 1500.0)
    hi = min(hi, upper) if upper is not None else hi
    v_hi = bal_fn(hi)
    for _ in range(3):
        if v_hi < 0.0:
            break
        if upper is not None and hi >= upper:
            break
        hi = min(hi*4., upper) if upper is not None else hi*4.
        v_hi = bal_fn(hi)
    if v_hi < 0.0:
        peak = minimize_scalar(
            lambda p: -bal_fn(p),
            bounds=(lo, hi),
            method="bounded",
            options={"maxiter": 20, "xatol": 0.5},
        )
        p_peak = float(peak.x)
        if bal_fn(p_peak) > 0.0:
            return float(brentq(bal_fn, p_peak, hi, xtol=1e-2, rtol=1e-8, maxiter=200))
    raise ConvergenceError("throat mixture did not converge")


def throat_wc(qoil_std: float, wc_su: float, qwat_nz: float) -> tuple[float, float]:
    """Throat Watercut and Formation Water Rate

    Calculate watercut and formation water rate into the jet pump throat.
    New watercut after the power fluid and reservoir fluid have mixed together.

    Args:
        qoil_std (float): Oil Rate, STD BOPD
        wc_su (float): Watercut at pump suction, decimal
        qwat_nz (float): Powerfluid Flowrate, BWPD

    Returns:
        wc_tm (float): Watercut at throat, decimal
        qwat_su (float): Formation Water at Suction, bwpd"""

    # Water-pump mode: at 100% suction water cut the anchor rate IS the
    # formation water (no oil), so skip the oil-basis algebra (which would
    # divide by 1 - wc_su = 0). The throat mixture stays 100% water.
    # [LIBRARY change -> upstream PR to kwellis/woffl]
    if wc_su >= 1.0:
        qwat_su = qoil_std  # the suction "oil" slot carries the water rate
        return 1.0, qwat_su

    qwat_su = qoil_std * wc_su / (1 - wc_su)
    qwat_tot = qwat_nz + qwat_su
    wc_tm = qwat_tot / (qwat_tot + qoil_std)
    return wc_tm, qwat_su


def _throat_mixture_anchor(
    qoil_std: float, qnz_bwpd: float, wc_tm: float, water_mode: bool
) -> float:
    """Anchor rate for throat-mixture (prop_tm) flow calculations.

    Oil path: the anchor is the standard oil rate — wc_tm carries the
    power-fluid water into the mixture totals, so nothing more is needed.
    Water-pump mode: the anchor IS the water standard rate (see
    ResMix.insitu_volm_flow's water branch) and wc_tm = 1.0 carries no
    information about the power fluid — the nozzle water must be added
    explicitly, or the diffuser and tubing traverse are sized on formation
    water alone (e.g. 300 BWPD modeled instead of 2,800 with power fluid),
    inconsistent with the throat momentum balance that DOES include the
    nozzle mass flow. [LIBRARY change -> upstream PR to kwellis/woffl]
    """
    if water_mode and wc_tm >= 1.0:
        return qoil_std + qnz_bwpd
    return qoil_std


def diffuser_ke(kdi: float, vtm: float, vdi: float) -> float:
    """Diffuser Kinetic Energy

    Calculate the kinetic energy in the diffuser.
    Substract the energy lost due to friction from inlet fluid.

    Args:
        kdi: Diffuser Friction Factor, unitless
        vtm: Velocity at Throat Mixture, ft/s
        vdi: Velocity at Diffuser Discharge, ft/s

    Returns:
        ke_di: Diffuser Kinetic Energy, ft2/s2
    """
    ke_di = (vdi**2 - (1 - kdi) * vtm**2) / 2
    return ke_di


def diffuser_discharge(
    ptm: float,
    ttm: float,
    kdi: float,
    ath: float,
    adi: float,
    qoil_std: float,
    prop_tm: ResMix,
) -> tuple[float, float]:
    """Diffuser Discharge Pressure

    Directly calculate the diffuser discharge pressure. Only loops until the diffuser total
    energy is greater than 0. Then uses numpy interpolation for diffuser discharge pressure.

    Args:
        ptm (float): Throat Mixture Pressure, psig
        ttm (float): Throat Mixture Temp, deg F
        kdi (float): Diffuser Friction Factor, unitless
        ath (float): Throat Area, ft2
        adi (float): Diffuser / Tubing Area, ft2
        qoil_std (float): Oil Rate, STD BOPD
        prop_tm (ResMix): Properties of Throat Mixture

    Returns:
        vtm (float): Throat Mixture Velocity
        pdi (float): Diffuser Discharge Pressure, psig
    """
    prop_tm = prop_tm.condition(ptm, ttm)
    qtot = sum(prop_tm.insitu_volm_flow(qoil_std))
    vdi = sp.velocity(qtot, adi)
    vtm = sp.velocity(qtot, ath)

    di_book = jp.JetBook(
        ptm, vdi, prop_tm.rho_mix(), prop_tm.cmix(), diffuser_ke(kdi, vtm, vdi)
    )

    pinc = 100  # pressure increase

    n = 0
    while di_book.tde_ray[-1] < 0:
        pdi = di_book.prs_ray[-1] + pinc

        prop_tm = prop_tm.condition(pdi, ttm)
        qtot = sum(prop_tm.insitu_volm_flow(qoil_std))
        vdi = sp.velocity(qtot, adi)

        di_book.append(
            pdi, vdi, prop_tm.rho_mix(), prop_tm.cmix(), diffuser_ke(kdi, vtm, vdi)
        )

        n += 1
        if n == 500:  # 50,000 psi above ptm — physically impossible, bail out
            raise ConvergenceError(
                "diffuser discharge did not converge, tde never crossed zero"
            )

    pdi = di_book.dedi_zero()
    return vtm, pdi  # type: ignore


def jetpump_base_calcs(
    psu: float,
    tsu: float,
    pni: float,
    rho_ni: float,
    ken: float,
    knz: float,
    kth: float,
    kdi: float,
    ath: float,
    anz: float,
    adi: float,
    ipr_su: InFlow,
    prop_su: ResMix,
) -> tuple[float, float, float, float, float, float, float, ResMix]:
    """Jet Pump Overall Equations

    Solve the jetpump equations, calculating out the expected discharge conditions.
    Function dete_zero() will raise a ValueError if the selected psu is too low. This method is
    being depreciated because it assumes a fixed powerfluid pressure directly at the jet pump. In
    practice this is difficult. Part of the code is being moved into "solopump" file under the
    discharge residual code so a power fluid rate iteration can be added. Additionally, discharge
    residual will allow for defining if the jetpump is forward or reverse circulating.

    Args:
        psu (float): Suction Pressure, psig
        tsu (float): Suction Temp, deg F
        pni (float): Nozzle Inlet Pressure, psig
        rho_ni (float): Nozzle Inlet Density, lbm/ft3
        ken (float): Enterance Friction Factor, unitless
        knz (float): Nozzle Friction Factor, unitless
        kth (float): Throat Friction Factor, unitless
        kdi (float): Diffuser Friction Factor, unitless
        ath (float): Throat Area, ft2
        anz (float): Nozzle Area, ft2
        adi (float): Diffuser Area, ft2
        ipr_su (InFlow): IPR of Reservoir
        prop_su (ResMix): Properties of Suction Fluid

    Returns:
        pte (float): Throat Entry Pressure, psig
        ptm (float): Throat Mixture Pressure, psig
        pdi (float): Diffuser Discharge Pressure, psig
        qoil_std (float): Oil Rate, STBOPD
        fwat_bpd (float): Formation Water Rate, BWPD
        qnz_bpd (float): Power Fluid Rate, BWPD
        mach_te (float): Throat Entry Mach, unitless
        prop_tm (ResMix): Properties of Discharge Fluid
    """
    ate = ath - anz
    qoil_std, te_book = throat_entry_zero_tde(
        psu=psu, tsu=tsu, ken=ken, ate=ate, ipr_su=ipr_su, prop_su=prop_su
    )
    pte, vte, rho_te, mach_te = te_book.dete_zero()

    # rho_ni is supplied at nozzle inlet conditions. Recover its reference
    # density for standard-rate conversion and mass-conserving water mixing.
    prop_pf = FormWater(1.).condition(pni, tsu)
    prop_pf = FormWater(rho_ni/prop_pf.density).condition(pni, tsu)
    vnz, qnz_bwpd = water_nozzle(pni, pte, tsu, knz, anz, prop_pf)
    wc_tm, fwat_bwpd, prop_tm = throat_mixture(qoil_std, qnz_bwpd, prop_su, prop_pf)
    ptm = throat_discharge(pte, tsu, kth, vnz, anz, prop_pf.density, vte, ate, rho_te, prop_tm)
    qtm_std = _throat_mixture_anchor(qoil_std, qnz_bwpd, wc_tm, prop_su.model_as_water)
    vtm, pdi = diffuser_discharge(ptm, tsu, kdi, ath, adi, qtm_std, prop_tm)
    return pte, ptm, pdi, qoil_std, fwat_bwpd, qnz_bwpd, mach_te, prop_tm
