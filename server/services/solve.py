"""Solver wrappers - single solve, batch sweep, PF-range sweep, traverse.

Server mirrors of the Streamlit solver call sites. Failure taxonomy is the
API's SolveErrorDetail contract:

- ThroatEntryNoSolution / IndexError -> "no_solution" + suggested_gor 250
- ConvergenceError                   -> "convergence"
- other solver ValueError            -> "convergence" (well cannot lift)
- form_wc >= 1 without water mode    -> "all_water" (pre-check, no solve)
- input problems (ValueError)        -> propagate; router maps to "invalid"
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import numpy as np
import pandas as pd

from server import schemas
from server.services import factories, frames

from woffl.gui.params import SimulationParams

log = logging.getLogger("woffl.web.solve")

# mirrors woffl/gui/utils.py:GOR_AUTO_RECOVERY_VALUE
GOR_AUTO_RECOVERY_VALUE = 250.0


class SolveFailure(Exception):
    """Typed solver failure carrying the SolveErrorDetail payload."""

    def __init__(self, error: str, message: str, suggested_gor: Optional[float] = None) -> None:
        super().__init__(message)
        self.error = error
        self.message = message
        self.suggested_gor = suggested_gor

    def detail(self) -> dict[str, Any]:
        """The HTTPException 422 detail body (SolveErrorDetail shape)."""
        return {
            "error": self.error,
            "message": self.message,
            "suggested_gor": self.suggested_gor,
        }


def _check_all_water(sp: schemas.SimParams) -> None:
    """All-water pre-check - a 100% WC well has no oil anchor to solve on.

    mirrors woffl/gui/tabs/jetpump_solver.py's watered-out dead-end guard:
    the engineer either enables dewatering (model as water) or lowers WC.
    """
    if sp.form_wc >= 1.0 and not sp.model_as_water:
        raise SolveFailure(
            "all_water",
            "Water cut is 100% and dewatering mode is off - there is no oil "
            "to anchor the IPR on. Enable 'model as water' or lower the "
            "water cut below 100%.",
        )


def _run_solver(
    p: SimulationParams,
    jetpump: Any,
    wellbore: Any,
    inflow: Any,
    res_mix: Any,
    wp: Any,
) -> tuple[float, bool, float, float, float, float]:
    """Run the assembly jetpump solver, mapping errors to SolveFailure.

    mirrors woffl/gui/utils.py:run_jetpump_solver - power fluid is the field
    model's FormWater preset conditioned (0, 60); errors are mapped instead
    of rendered.
    """
    from woffl.assembly.solopump import jetpump_solver
    from woffl.flow.errors import ConvergenceError, ThroatEntryNoSolution

    prop_pf = factories.power_fluid(p.field_model)
    try:
        return jetpump_solver(
            pwh=p.surf_pres,
            tsu=p.form_temp,
            ppf_surf=p.ppf_surf,
            jpump=jetpump,
            wellbore=wellbore,
            wellprof=wp,
            ipr_su=inflow,
            prop_su=res_mix,
            prop_pf=prop_pf,
            jpump_direction=p.jpump_direction,
        )
    except (ThroatEntryNoSolution, IndexError) as exc:
        # ThroatEntryNoSolution subclasses IndexError too; both mean no valid
        # throat-entry zero crossing - typically GOR too low. Suggest the GOR
        # auto-recovery value the Streamlit solver reseeds with.
        raise SolveFailure("no_solution", str(exc) or "No throat-entry solution", GOR_AUTO_RECOVERY_VALUE) from exc
    except ConvergenceError as exc:
        raise SolveFailure("convergence", str(exc)) from exc
    except ValueError as exc:
        # mirrors woffl/gui/utils.py:run_jetpump_solver's generic ValueError
        # branch (well cannot lift at max suction pressure).
        raise SolveFailure("convergence", str(exc)) from exc


def solve_single(well: str, sp: schemas.SimParams) -> dict[str, Any]:
    """Solve one well/pump operating point (SolveResult shape).

    Args:
        well: Selected well name ("Custom" allowed).
        sp: Simulation parameters (qwf = TOTAL LIQUID BLPD).

    Returns:
        SolveResult dict.

    Raises:
        SolveFailure: typed solver failure (all_water/no_solution/convergence).
        ValueError: invalid inputs (router maps to 422 "invalid").
    """
    _check_all_water(sp)
    p = sp.to_simulation_params(well)
    jetpump, wellbore, inflow, res_mix, wp = factories.build_sim_objects(sp, well)
    psu, sonic_status, qoil_std, fwat_bwpd, qnz_bwpd, mach_te = _run_solver(
        p, jetpump, wellbore, inflow, res_mix, wp
    )
    return {
        "psu": float(psu),
        "sonic_status": bool(sonic_status),
        "qoil_std": float(qoil_std),
        "fwat_bwpd": float(fwat_bwpd),
        "qnz_bwpd": float(qnz_bwpd),
        "mach_te": float(mach_te),
        "dewatering": bool(sp.model_as_water),
        "total_water": float(fwat_bwpd + qnz_bwpd),
    }


# ---------------------------------------------------------------------------
# Batch sweep
# ---------------------------------------------------------------------------


def _success_stats(df: pd.DataFrame) -> tuple[int, int, float]:
    """(total, successful, success_pct) for a batch sweep dataframe.

    mirrors woffl/gui/tabs/batch_run.py:batch_success_stats - "successful"
    means the solver converged, i.e. qoil_std is not NaN.
    """
    total = len(df)
    if total == 0:
        return 0, 0, 0.0
    successful = int((~df["qoil_std"].isna()).sum())
    return total, successful, (successful / total * 100)


def _augment_with_formation_marginals(batch: Any) -> None:
    """Compute mofwr and coeff_form for the formation-water axis.

    mirrors woffl/gui/tabs/batch_run.py:_augment_with_formation_marginals -
    the library's process_results only fits total/lift water; formation-water
    marginals are a GUI-side augmentation.
    """
    from woffl.assembly.batchpump import batch_curve_fit, gradient_back

    batch.df = batch.df.drop(columns=["mofwr"], errors="ignore")
    if "semi" not in batch.df.columns:
        batch.df["semi"] = False

    semi_df = batch.df[batch.df["semi"]].copy()
    if semi_df.empty:
        batch.df["mofwr"] = np.nan
        batch.coeff_form = None
        return

    semi_df = semi_df.sort_values(by="qoil_std", ascending=True)
    qoil_semi = semi_df["qoil_std"].to_numpy()
    fwat_semi = semi_df["form_wat"].to_numpy()

    semi_df["mofwr"] = gradient_back(qoil_semi, fwat_semi)

    batch.df = batch.df.merge(
        semi_df[["mofwr"]], left_index=True, right_index=True, how="left"
    )

    try:
        batch.coeff_form = batch_curve_fit(qoil_semi, fwat_semi, origin=False)
    except Exception:
        batch.coeff_form = None


def _recommend(batch: Any, marginal_watercut: float, water_type: str) -> Optional[dict[str, Any]]:
    """Recommend a jet pump by marginal watercut, or None when impossible.

    mirrors woffl/gui/utils.py:recommend_jetpump for the two axes the API
    exposes ("total" / "formation"; the GUI radio never offers "lift").
    Failures return None - same as the performance graph's try/except.
    """
    from woffl.assembly.batchpump import exp_model, rev_exp_deriv

    try:
        if not hasattr(batch, "df") or batch.df.empty:
            return None

        if water_type == "formation":
            water_col, marg_col = "form_wat", "mofwr"
            coeff = getattr(batch, "coeff_form", None)
        else:
            water_col, marg_col = "totl_wat", "motwr"
            coeff = getattr(batch, "coeff_totl", None)

        if marg_col not in batch.df.columns or "semi" not in batch.df.columns:
            return None

        semi_df = batch.df[batch.df["semi"]].copy()
        if semi_df.empty:
            return None
        semi_df = semi_df.sort_values(by="qoil_std", ascending=True)

        water_rates = semi_df[water_col].values
        original_ratios = semi_df[marg_col].values
        marginal_watercuts = 1 / (1 + original_ratios)

        below_threshold = marginal_watercuts <= marginal_watercut

        if not any(below_threshold):
            best_idx = int(np.argmin(marginal_watercuts))
            return {
                "nozzle": str(semi_df.iloc[best_idx]["nozzle"]),
                "throat": str(semi_df.iloc[best_idx]["throat"]),
                "qoil_std": float(semi_df.iloc[best_idx]["qoil_std"]),
                "water_rate": float(semi_df.iloc[best_idx][water_col]),
                "marginal_ratio": frames.opt_float(marginal_watercuts[best_idx]),
                "recommendation_type": "best_available",
            }

        valid_indices = np.where(below_threshold)[0]

        # No curve-fit coefficients (e.g. single semi-finalist): pick the
        # highest-oil pump still below the threshold.
        if coeff is None:
            valid_oil = [semi_df.iloc[int(idx)]["qoil_std"] for idx in valid_indices]
            best_idx = int(valid_indices[int(np.argmax(valid_oil))])
            return {
                "nozzle": str(semi_df.iloc[best_idx]["nozzle"]),
                "throat": str(semi_df.iloc[best_idx]["throat"]),
                "qoil_std": float(semi_df.iloc[best_idx]["qoil_std"]),
                "water_rate": float(semi_df.iloc[best_idx][water_col]),
                "marginal_ratio": frames.opt_float(marginal_watercuts[best_idx]),
                "recommendation_type": "optimal",
            }

        # Theoretical optimum where the curve's marginal WC hits the threshold.
        oil_water_ratio = (1 - marginal_watercut) / marginal_watercut
        a, b, c = coeff
        optimal_water_rate = rev_exp_deriv(oil_water_ratio, b, c)
        optimal_oil_rate = exp_model(optimal_water_rate, a, b, c)

        distances = []
        for idx in valid_indices:
            pump_water = water_rates[int(idx)]
            pump_oil = semi_df.iloc[int(idx)]["qoil_std"]
            distances.append(
                math.sqrt(
                    (pump_water - optimal_water_rate) ** 2
                    + (pump_oil - optimal_oil_rate) ** 2
                )
            )
        closest_idx = int(valid_indices[int(np.argmin(distances))])

        return {
            "nozzle": str(semi_df.iloc[closest_idx]["nozzle"]),
            "throat": str(semi_df.iloc[closest_idx]["throat"]),
            "qoil_std": float(semi_df.iloc[closest_idx]["qoil_std"]),
            "water_rate": float(semi_df.iloc[closest_idx][water_col]),
            "marginal_ratio": frames.opt_float(marginal_watercuts[closest_idx]),
            "recommendation_type": "optimal",
            "theoretical_water_rate": frames.opt_float(optimal_water_rate),
            "theoretical_oil_rate": frames.opt_float(optimal_oil_rate),
        }
    except Exception as exc:
        log.warning("jetpump recommendation failed: %s", exc)
        return None


def _fit_curve(batch: Any, water_type: str) -> Optional[dict[str, list[float]]]:
    """Sampled exponential fit curve {x, y}, or None when no fit exists.

    mirrors woffl/gui/tabs/batch_run.py:_render_performance_graph curve
    sampling: linspace(0, max water of the mode, 200), oil clipped >= 0.
    """
    from woffl.assembly.batchpump import exp_model

    coeff = getattr(batch, "coeff_form" if water_type == "formation" else "coeff_totl", None)
    if coeff is None:
        return None
    df = batch.df[~batch.df["qoil_std"].isna()]
    if df.empty:
        return None
    water_col = "form_wat" if water_type == "formation" else "totl_wat"
    max_water = frames.opt_float(df[water_col].max())
    if max_water is None or max_water <= 0:
        return None
    a, b, c = coeff
    xs = np.linspace(0.0, max_water, 200)
    ys = np.clip(np.array([exp_model(x, a, b, c) for x in xs]), 0.0, None)
    return {"x": [float(v) for v in xs], "y": [float(v) for v in ys]}


def run_batch(well: str, sp: schemas.SimParams) -> dict[str, Any]:
    """Full nozzle x throat batch sweep (BatchResponse shape).

    Args:
        well: Selected well name.
        sp: Simulation parameters carrying the batch grid + marginal WC.

    Returns:
        BatchResponse dict (rows keep the BatchPump df column names).

    Raises:
        ValueError: empty nozzle/throat grid (router maps to 422 "invalid").
    """
    from woffl.assembly.batchpump import BatchPump

    if not sp.nozzle_batch_options or not sp.throat_batch_options:
        raise ValueError(
            "Select at least one nozzle size and one throat ratio for batch analysis"
        )

    p = sp.to_simulation_params(well)
    _jetpump, wellbore, inflow, res_mix, wp = factories.build_sim_objects(sp, well)
    prop_pf = factories.power_fluid(p.field_model)

    # mirrors woffl/gui/utils.py:run_batch_pump
    jp_list = BatchPump.jetpump_list(
        list(sp.nozzle_batch_options),
        list(sp.throat_batch_options),
        knz=0.01,
        ken=p.ken,
        kth=p.kth,
        kdi=p.kdi,
    )
    batch = BatchPump(
        pwh=p.surf_pres,
        tsu=p.form_temp,
        ppf_surf=p.ppf_surf,
        wellbore=wellbore,
        wellprof=wp,
        ipr_su=inflow,
        prop_su=res_mix,
        prop_pf=prop_pf,
        jpump_direction=p.jpump_direction,
        wellname=f"{p.field_model} Well",
    )
    batch.batch_run(jp_list)

    # Guard curve-fit failures the way utils.run_batch_pump does: too few
    # converged points ("must not exceed the number of data points"), a
    # non-converging fit ("Optimal parameters not found"), or no
    # semi-finalists at all. The rows and stats are still valid - only the
    # fitted curve / recommendation degrade.
    try:
        batch.process_results()
    except (ValueError, RuntimeError, TypeError) as exc:
        log.warning("batch process_results degraded on %s: %s", well, exc)
        if "semi" not in getattr(batch, "df", pd.DataFrame()).columns:
            batch.df["semi"] = False
    if not hasattr(batch, "coeff_totl"):
        batch.coeff_totl = None
    if not hasattr(batch, "coeff_lift"):
        batch.coeff_lift = None

    _augment_with_formation_marginals(batch)

    total, successful, success_pct = _success_stats(batch.df)

    # The GUI radio offers total/formation and passes it straight through to
    # recommend_jetpump (see batch_run.py:_render_performance_graph); "lift"
    # is a library-only axis the app never uses.
    recommended = _recommend(batch, float(sp.marginal_watercut), sp.water_type)

    return {
        "rows": frames.records(batch.df),
        "stats": {"total": total, "successful": successful, "success_pct": float(success_pct)},
        "recommended": recommended,
        "fit_curve": _fit_curve(batch, sp.water_type),
        "x_mode": sp.water_type,
    }


# ---------------------------------------------------------------------------
# Power-fluid range sweep
# ---------------------------------------------------------------------------


def pressure_sweep_range(
    power_fluid_min: float, power_fluid_max: float, power_fluid_step: float
) -> np.ndarray:
    """PF-pressure sweep points, inclusive of an exact-multiple max.

    mirrors woffl/gui/utils.py:pressure_sweep_range - arange's exclusive stop
    would drop an exact-multiple max, and naively extending the stop can
    overshoot it, so clip back to power_fluid_max (+ epsilon for float
    rounding). A swept point can never land outside the requested range.
    """
    pressure_range = np.arange(
        power_fluid_min, power_fluid_max + power_fluid_step, power_fluid_step
    )
    return pressure_range[pressure_range <= power_fluid_max + 1e-9]


def run_pf_range(well: str, sp: schemas.SimParams) -> dict[str, Any]:
    """Batch sweep across a range of PF surface pressures (PfRangeResponse).

    Per-pressure isolation: one failing pressure logs and drops out; the
    rest of the sweep survives.

    Args:
        well: Selected well name.
        sp: Simulation parameters carrying the PF range + batch grid.

    Returns:
        PfRangeResponse dict (rows carry pump code + power_fluid_pressure).

    Raises:
        ValueError: empty nozzle/throat grid (router maps to 422 "invalid").
    """
    from woffl.assembly.batchpump import BatchPump

    if not sp.nozzle_batch_options or not sp.throat_batch_options:
        raise ValueError(
            "Select at least one nozzle size and one throat ratio for the PF sweep"
        )

    p = sp.to_simulation_params(well)
    _jetpump, wellbore, inflow, res_mix, wp = factories.build_sim_objects(sp, well)
    prop_pf = factories.power_fluid(p.field_model)

    pressures = pressure_sweep_range(
        p.power_fluid_min, p.power_fluid_max, p.power_fluid_step
    )
    jp_list = BatchPump.jetpump_list(
        list(sp.nozzle_batch_options),
        list(sp.throat_batch_options),
        knz=0.01,
        ken=p.ken,
        kth=p.kth,
        kdi=p.kdi,
    )

    # mirrors woffl/gui/utils.py:run_power_fluid_range_batch, with per-point
    # isolation added: the Streamlit loop let one bad pressure kill the whole
    # sweep; the API drops only that point.
    parts: list[pd.DataFrame] = []
    for pressure in pressures:
        try:
            batch = BatchPump(
                pwh=p.surf_pres,
                tsu=p.form_temp,
                ppf_surf=float(pressure),
                wellbore=wellbore,
                wellprof=wp,
                ipr_su=inflow,
                prop_su=res_mix,
                prop_pf=prop_pf,
                jpump_direction=p.jpump_direction,
                wellname=f"{p.field_model} Well",
            )
            batch.batch_run(jp_list, debug=False)
            df = batch.df
            df["power_fluid_pressure"] = float(pressure)
            parts.append(df)
        except Exception as exc:
            log.warning("PF sweep point %s psi failed on %s: %s", pressure, well, exc)

    rows: list[dict[str, Any]] = []
    if parts:
        comprehensive = pd.concat(parts, ignore_index=True)
        comprehensive["pump"] = comprehensive["nozzle"].astype(str) + comprehensive[
            "throat"
        ].astype(str)
        rows = frames.records(comprehensive)

    return {"rows": rows, "pressures": [float(v) for v in pressures]}


# ---------------------------------------------------------------------------
# Pressure profile (traverse)
# ---------------------------------------------------------------------------


def _powerfluid_pressure_profile(
    ppf_surf: float,
    tsu: float,
    qnz_bwpd: float,
    prop_pf: Any,
    wellbore: Any,
    wellprof: Any,
    flowpath: str,
) -> np.ndarray:
    """Segmented single-phase PF pressure column from surface to the JP.

    mirrors woffl/gui/tabs/pressure_profile.py:_powerfluid_pressure_profile -
    static head plus Darcy friction over the same outflow_spacing(100) grid
    the production traverse uses.
    """
    from woffl.flow import singlephase as sph

    if flowpath == "tubing":
        hyd_dia = wellbore.tube_hyd_dia
        area = wellbore.tube_area
        abs_ruff = wellbore.tube_abs_ruff
    else:
        hyd_dia = wellbore.ann_hyd_dia
        area = wellbore.ann_area
        abs_ruff = wellbore.ann_abs_ruff

    md_seg, vd_seg = wellprof.outflow_spacing(100)

    # Single-phase parameters (constant for incompressible fluid)
    prop_cond = prop_pf.condition(ppf_surf, tsu)
    rho = prop_cond.density
    visc = prop_cond.viscosity

    qwat_fts = sph.bpd_to_ft3s(qnz_bwpd)
    vel = sph.velocity(qwat_fts, area)
    n_re = sph.reynolds(rho, vel, hyd_dia, visc)
    rel_ruff = sph.relative_roughness(hyd_dia, abs_ruff)
    ff = sph.ffactor_darcy(n_re, rel_ruff)

    # Height convention matches production_top_down_press (negative = going
    # down). Length is NOT negated (positive = with flow, i.e. downward).
    vd_diff = np.diff(vd_seg) * -1
    md_diff = np.diff(md_seg)

    prs_list = [ppf_surf]
    for length, height in zip(md_diff, vd_diff):
        dp_stat = sph.diff_press_static(rho, height)
        dp_fric = sph.diff_press_friction(ff, rho, vel, hyd_dia, length)
        prs_list.append(prs_list[-1] - dp_stat - dp_fric)

    return np.array(prs_list)


def pressure_profile(well: str, sp: schemas.SimParams) -> dict[str, Any]:
    """Production and PF pressure traverses plus their differential.

    mirrors woffl/gui/tabs/pressure_profile.py:render_tab - solve the
    operating point first, then walk both strings top-down over the common
    outflow_spacing(100) MD grid.

    Args:
        well: Selected well name.
        sp: Simulation parameters.

    Returns:
        PressureProfileResponse dict.

    Raises:
        SolveFailure: the operating-point solve failed (typed contract).
        ValueError: invalid inputs (router maps to 422 "invalid").
    """
    from woffl.flow import jetflow as jf
    from woffl.flow import outflow as of
    from woffl.pvt.resmix import ResMix

    _check_all_water(sp)
    p = sp.to_simulation_params(well)
    jetpump, wellbore, inflow, res_mix, wp = factories.build_sim_objects(sp, well)
    psu, _sonic_status, qoil_std, _fwat_bwpd, qnz_bwpd, _mach_te = _run_solver(
        p, jetpump, wellbore, inflow, res_mix, wp
    )

    # Flow-path direction handling (reverse: produce up tubing, PF down the
    # annulus; forward: the opposite).
    if p.jpump_direction == "reverse":
        prod_path, pf_path = "tubing", "annulus"
    else:
        prod_path, pf_path = "annulus", "tubing"

    # Mixed production fluid (formation + power fluid) - same as
    # discharge_residual.
    prop_pf = factories.power_fluid(p.field_model)
    wc_tm, _ = jf.throat_wc(qoil_std, res_mix.wc, qnz_bwpd)
    prop_tm = ResMix(wc_tm, res_mix.fgor, res_mix.oil, res_mix.wat, res_mix.gas)

    # Production pressure profile (top-down from wellhead)
    md_seg, prod_prs, _slh = of.production_top_down_press(
        p.surf_pres, p.form_temp, qoil_std, prop_tm, wellbore, wp, prod_path
    )

    # Power-fluid pressure profile (top-down from PF surface pressure) over
    # the same outflow_spacing(100) grid.
    pf_prs = _powerfluid_pressure_profile(
        p.ppf_surf, p.form_temp, qnz_bwpd, prop_pf, wellbore, wp, pf_path
    )

    # Differential: power fluid minus production
    differential = pf_prs - prod_prs

    md = [float(v) for v in md_seg]
    return {
        "prod": {"md": md, "press": [float(v) for v in prod_prs]},
        "pf": {"md": md, "press": [float(v) for v in pf_prs]},
        "diff": {"md": md, "dp": [float(v) for v in differential]},
        "jpump_md": float(wp.jetpump_md),
        "metrics": {
            "psu": float(psu),
            "prod_at_jp": float(prod_prs[-1]),
            "pf_at_jp": float(pf_prs[-1]),
            "dp_at_jp": float(differential[-1]),
        },
    }
