"""Selectable steady return-column hydraulics, with explicit model identities.

Beggs--Brill remains in outflow.py. Alternatives share its PVT, standard-rate
and geometry conventions; neither borrows its holdup or friction multiplier.
References and limitations: docs/hydraulics_models_2026-09-11.md.
"""

# [LIBRARY change -> upstream PR to kwellis/woffl]
import math
from typing import Literal

from scipy.optimize import brentq

from woffl.flow import singlephase as sp
from woffl.flow.errors import HydraulicsDomainError
from woffl.pvt import ResMix

HydraulicsModel = Literal["beggs", "hagedorn_brown", "drift_flux"]
HYDRAULICS_MODELS = {
    "beggs": ("Beggs-Brill + Payne", "beggs-payne-v1"),
    "hagedorn_brown": ("Hagedorn-Brown + Griffith", "hb-griffith-v1"),
    "drift_flux": ("Drift-flux (Shi / Pan)", "shi-pan-v1"),
}
GC = 32.174
PA_PER_PSI = 6894.757293168
KG_M3_PER_LBM_FT3 = 16.01846337396
N_M_PER_LBF_FT = 14.5939029372


def validate_model(model: str) -> str:
    """Return a supported model ID (str); reject unknown/unimplemented models."""
    if model not in HYDRAULICS_MODELS:
        raise HydraulicsDomainError(f"Unsupported hydraulics model: {model!r}")
    return model


def physics_model(model: str = "beggs") -> str:
    """Combined pump/return-model identity (str), including legacy BB fits."""
    from woffl.flow.entry_energy import MODEL_VERSION

    validate_model(model)
    return MODEL_VERSION if model == "beggs" else f"{MODEL_VERSION}+{HYDRAULICS_MODELS[model][1]}"


def kutateladze(bond: float) -> float:
    """Pan et al. (2011), Eq. 17; Bond number and result are dimensionless.

    Cku=142, Cw=0.008. Rationalization avoids cancellation at small diameter.
    """
    if not math.isfinite(bond) or bond <= 0:
        raise HydraulicsDomainError("Bond number must be positive and finite")
    return math.sqrt(math.sqrt(bond) / (142 * .008 * (math.sqrt(1 + bond / (142**2 * .008)) + 1)))


def drift_flux_holdup(
    vsl: float, vsg: float, diameter: float, rho_liq: float, rho_gas: float,
    tension: float, incline: float,
) -> float:
    """Shi gas/liquid slip with Pan's smoothing; solve the implicit gas flux.

    Args:
        vsl (float): Superficial liquid velocity, m/s, nonnegative.
        vsg (float): Superficial gas velocity, m/s, nonnegative.
        diameter (float): Hydraulic diameter, m.
        rho_liq (float): Liquid density, kg/m3.
        rho_gas (float): Gas density, kg/m3.
        tension (float): Gas/liquid surface tension, N/m.
        incline (float): Upward flow angle from horizontal, degrees, 0..90.

    Returns:
        holdup (float): In-situ liquid volume fraction, dimensionless.

    Pan et al., LBNL-4291E, Eqs. 7, 11-12, 15-25 and Table 1 (Cmax=1.2).
    Oil/water is a mixed liquid. This is not a three-phase slip/thermal solver.
    """
    _check_state(vsl, vsg, diameter, rho_liq, rho_gas, tension, incline)
    if vsg == 0:
        return 1.0
    if vsl == 0:
        return 0.0
    gravity = 9.80665
    bond = diameter**2 * gravity * (rho_liq - rho_gas) / tension
    ku = kutateladze(bond)
    uc = (gravity * tension * (rho_liq - rho_gas) / rho_liq**2)**.25
    flooding = ku * math.sqrt(rho_liq / rho_gas) * uc
    theta = math.radians(90 - incline)  # Pan uses angle from vertical
    m = 0.0 if incline == 0 else 1.27 * math.cos(theta)**.24 * (1 + math.sin(theta))**1.08
    mass_flux = rho_liq * vsl + rho_gas * vsg
    vmix = vsl + vsg
    b = 2 / 1.2 - 1.0667

    def residual(alpha):
        rho = alpha * rho_gas + (1 - alpha) * rho_liq
        um = mass_flux / rho  # mass-center velocity, not volumetric vmix
        beta = min(1.0, max(alpha, alpha * abs(um) / flooding))
        # B is explicitly the threshold where C0 starts to fall (Eq. 22 prose).
        eta = max(0.0, (beta - b) / (1 - b))
        c0 = 1.2 / (1 + .2 * eta**2)
        blend = min(1.0, max(0.0, (alpha - .06) / (.12 - .06)))
        k = 1.53 + (c0 * ku - 1.53) * .5 * (1 - math.cos(math.pi * blend))
        liquid_profile = 1 - c0 * alpha
        drift = liquid_profile * uc * k * m / (c0 * alpha * math.sqrt(rho_gas / rho_liq) + liquid_profile)
        return alpha * (c0 * vmix + drift) - vsg

    alpha = brentq(residual, 0.0, vsg / vmix, xtol=1e-12, rtol=1e-12, maxiter=60)
    return 1 - alpha


def _check_state(vsl, vsg, diameter, rho_liq, rho_gas, tension, incline):
    if (not all(math.isfinite(x) for x in (vsl, vsg, diameter, rho_liq, rho_gas, tension, incline))
            or vsl < 0 or vsg < 0 or diameter <= 0 or tension <= 0
            or not 0 < rho_gas < rho_liq or not 0 <= incline <= 90):
        raise HydraulicsDomainError("Alternative hydraulics requires finite co-current upward flow, positive diameter/tension and liquid density above gas density")


def hagedorn_brown_holdup(
    vsl: float, vsg: float, diameter: float, rho_liq: float, viscosity: float,
    tension: float, pressure: float,
) -> tuple[float, bool]:
    """Modified Hagedorn--Brown holdup, including the Griffith bubble branch.

    Args:
        vsl (float): Superficial liquid velocity, m/s, nonnegative.
        vsg (float): Superficial gas velocity, m/s, nonnegative.
        diameter (float): Hydraulic diameter, m.
        rho_liq (float): Liquid density, kg/m3.
        viscosity (float): Liquid dynamic viscosity, Pa s.
        tension (float): Gas/liquid surface tension, N/m.
        pressure (float): Absolute pressure, Pa.

    Returns:
        holdup (float): In-situ liquid fraction, dimensionless.
        bubble (bool): Whether the Griffith bubble-flow branch was used.

    Hagedorn & Brown (1965); PROMOD1 Sec. 2.5.2 rational chart fits. The
    secondary factor uses the liquid VISCOSITY number NL (not NLV).
    Inclined application projects gravity only; holdup is a vertical reference.
    """
    if (not all(math.isfinite(x) for x in (vsl, vsg, diameter, rho_liq, viscosity, tension, pressure))
            or min(vsl, vsg) < 0 or min(diameter, rho_liq, viscosity, tension, pressure) <= 0):
        raise HydraulicsDomainError("Invalid Hagedorn-Brown fluid/flow state")
    if vsg == 0:
        return 1.0, True
    if vsl == 0:
        return 0.0, False
    vmix = vsl + vsg
    lam = vsl / vmix
    # The numerical constants in this criterion require ft/s and feet.
    lb = max(1.071 - .2218 * (vmix / .3048)**2 / (diameter / .3048), .13)
    if 1 - lam < lb:
        vs = .8 * .3048
        term = 1 + vmix / vs
        alpha = (2 * vsg / vs) / (term + math.sqrt(term**2 - 4 * vsg / vs))
        return max(lam, 1 - alpha), True
    gravity = 9.80665
    nl = viscosity * (gravity / (rho_liq * tension**3))**.25
    if nl <= .002:
        cnl = .0019
    elif nl >= .4:
        cnl = .0115
    else:
        cnl = (.0019 + .0322 * nl - .6642 * nl**2 + 4.9951 * nl**3) / (1 - 10.0147 * nl + 33.8696 * nl**2 + 277.2817 * nl**3)
    velocity_number = (rho_liq / (gravity * tension))**.25
    nlv, ngv = vsl * velocity_number, vsg * velocity_number
    nd = diameter * math.sqrt(rho_liq * gravity / tension)
    h = nlv / ngv**.575 * (pressure / (14.7 * PA_PER_PSI))**.1 * cnl / nd
    base = math.sqrt((.0047 + 1123.32 * h + 729489.64 * h**2) / (1 + 1097.1566 * h + 722153.97 * h**2))
    x = ngv * nl**.38 / nd**2.14
    # Stay on the published secondary chart; a rational extrapolation has poles.
    if x <= .01:
        psi = 1.0
    else:
        x = min(x, .09)
        psi = max(1.0, (1.0886 - 69.9473 * x + 2334.3497 * x**2 - 12896.683 * x**3)
                  / (1 - 53.4401 * x + 1517.9369 * x**2 - 8419.8115 * x**3))
    return min(1.0, max(lam, base * psi)), False


def alternative_diff_press(
    pin: float, tin: float, hyd_dia: float, area: float, abs_ruff: float,
    length: float, height: float, qoil_std: float, prop: ResMix, model: str,
) -> tuple[float, float, float]:
    """Alternative segment pressure loss using outflow.beggs_diff_press units.

    Args:
        pin (float): Segment inlet pressure, psig.
        tin (float): Isothermal fluid temperature, degF.
        hyd_dia (float): Hydraulic diameter, inches.
        area (float): Actual flow area, ft2 (independent of hydraulic diameter).
        abs_ruff (float): Absolute roughness, inches.
        length (float): Signed distance, ft; positive along flow.
        height (float): Signed elevation change, ft; positive upward.
        qoil_std (float): Oil standard rate, BOPD; water BPD in dewatering mode.
        prop (ResMix): Mixture PVT, mutated and left at pin/tin.
        model (str): hagedorn_brown or drift_flux.

    Returns:
        dp_stat (float): Gravity pressure loss including acceleration, psid.
        dp_fric (float): Friction pressure loss including acceleration, psid.
        holdup (float): In-situ liquid fraction, dimensionless.
    """
    validate_model(model)
    if model == "beggs":
        raise HydraulicsDomainError("Beggs-Brill is evaluated by beggs_diff_press")
    if not all(math.isfinite(x) for x in (pin, tin, hyd_dia, area, abs_ruff, length, height, qoil_std)) or min(hyd_dia, area) <= 0 or abs_ruff < 0 or qoil_std < 0 or pin <= -14.7:
        raise HydraulicsDomainError("Invalid return-column inputs")
    if length == 0:
        if height != 0:
            raise HydraulicsDomainError("Nonzero elevation in a zero-length segment")
        return 0.0, 0.0, prop.condition(pin, tin).nslh()
    ratio = height / length
    if ratio < -1e-9 or ratio > 1 + 1e-6:
        raise HydraulicsDomainError("This alternative does not support downhill return-flow segments")
    incline = math.degrees(math.asin(min(1.0, max(0.0, ratio))))

    def state(pressure):
        prop.condition(pressure, tin)
        qo, qw, qg = prop.insitu_volm_flow(qoil_std)
        vsl, vsg = (qo + qw) / area, qg / area
        rl, rg = prop.rho_two()
        ml, mg = prop.visc_two()
        if qoil_std == 0:
            return prop.nslh(), rl, rg, ml, mg, vsl, vsg, False
        args = (vsl * .3048, vsg * .3048, hyd_dia * .0254, rl * KG_M3_PER_LBM_FT3)
        sigma = prop.tension * N_M_PER_LBF_FT
        if model == "drift_flux":
            hl = drift_flux_holdup(*args, rg * KG_M3_PER_LBM_FT3, sigma, incline)
            bubble = False
        else:
            hl, bubble = hagedorn_brown_holdup(*args, ml * .001, sigma, (pressure + 14.7) * PA_PER_PSI)
        return hl, rl, rg, ml, mg, vsl, vsg, bubble

    hl, rl, rg, ml, mg, vsl, vsg, bubble = state(pin)
    rho = hl * rl + (1 - hl) * rg
    vmix = vsl + vsg
    if vmix == 0:
        return sp.diff_press_static(rho, height), 0.0, hl
    mass_flux = rl * vsl + rg * vsg
    if model == "hagedorn_brown":
        visc = ml**hl * mg**(1 - hl)
        reynolds = sp.reynolds(rho, vmix, hyd_dia, visc)
        rho_f, vel_f = (mass_flux / vmix)**2 / rho, vmix
        if bubble:
            rho_f, vel_f = rl, vsl / hl
            reynolds = sp.reynolds(rl, vel_f, hyd_dia, ml)
        # H-B acceleration density is the slip density (Whitson pressure model).
        denominator = 1 - rho * vmix * vsg / (GC * 144 * (pin + 14.7))
    else:
        um = mass_flux / rho
        visc = hl * ml + (1 - hl) * mg
        reynolds = sp.reynolds(rho, um, hyd_dia, visc)
        rho_f, vel_f = rho, um

        def momentum(pressure):
            h, r_l, r_g, _, _, vl, vg, _ = state(pressure)
            liquid = r_l * vl**2 / h if h > 0 else 0.0
            gas = r_g * vg**2 / (1 - h) if h < 1 else 0.0
            return (liquid + gas) / (GC * 144)

        # Steady isothermal momentum balance: (1 + dJ/dP) dP/dL = -gravity-friction.
        # Includes PVT expansion/slip; does not reuse Beggs' acceleration closure.
        step = max(.01, (pin + 14.7) * 1e-5)
        step = min(step, (pin + 14.7) / 4)
        try:
            denominator = 1 + (momentum(pin + step) - momentum(pin - step)) / (2 * step)
        finally:
            prop.condition(pin, tin)
    if not math.isfinite(denominator) or denominator <= 0:
        raise HydraulicsDomainError(f"{model}: no subcritical return-pressure gradient (acceleration limit)")
    ff = sp.ffactor_darcy(reynolds, abs_ruff / hyd_dia)
    static = sp.diff_press_static(rho, height)
    friction = sp.diff_press_friction(ff, rho_f, vel_f, hyd_dia, length)
    return static / denominator, friction / denominator, hl
