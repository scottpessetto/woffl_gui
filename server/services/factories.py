"""Simulation object factories - server mirrors of the Streamlit GUI builders.

Each factory is a faithful copy of its ``woffl/gui/utils.py`` counterpart
(citation comments on every function) minus the Streamlit dependencies.
``build_sim_objects`` is the one entry point the solver wrappers use: it goes
through ``schemas.SimParams.to_simulation_params`` so the derived rates (the
qwf TOTAL-LIQUID convention, single-phase ``inflow_rate``) stay canonical.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from server import config, schemas
from server.cache import ttl_cache
from server.services import datasources

from woffl.flow.inflow import InFlow
from woffl.geometry import JetPump, Pipe, PipeInPipe, WellProfile
from woffl.pvt import BlackOil, FormGas, FormWater, ResMix

log = logging.getLogger("woffl.web.factories")


# mirrors woffl/gui/utils.py:create_jetpump
def create_jetpump(nozzle_no: str, area_ratio: str, ken: float, kth: float, kdi: float) -> JetPump:
    """Create a JetPump object with the given parameters."""
    return JetPump(nozzle_no=nozzle_no, area_ratio=area_ratio, ken=ken, kth=kth, kdi=kdi)


def apply_nozzle_area_factor(jetpump: JetPump, factor: float) -> JetPump:
    """Scale a pump's nozzle diameter so its flow area is ``factor`` x catalog.

    ``dnz_eff = dnz_catalog * sqrt(factor)`` - the pf_calibration wear
    mechanics. JetPump.anz/ate are derived properties, so they track the
    scaled dnz automatically. ``factor=1.0`` is a no-op (byte-identical
    solves). Mutates and returns the pump.
    """
    if factor != 1.0:
        jetpump.dnz = jetpump.dnz * float(np.sqrt(factor))
    return jetpump


# mirrors woffl/gui/utils.py:create_pipes
def create_pipes(
    tubing_od: float = 4.5,
    tubing_thickness: float = 0.5,
    casing_od: float = 6.875,
    casing_thickness: float = 0.5,
) -> tuple[Pipe, Pipe, PipeInPipe]:
    """Create tubing, casing, and wellbore (PipeInPipe) objects."""
    tube = Pipe(out_dia=tubing_od, thick=tubing_thickness)
    case = Pipe(out_dia=casing_od, thick=casing_thickness)
    wellbore = PipeInPipe(inn_pipe=tube, out_pipe=case)
    return tube, case, wellbore


# mirrors woffl/gui/utils.py:create_inflow
def create_inflow(qwf: float, pwf: float, pres: float) -> InFlow:
    """Create an InFlow object from a SINGLE-PHASE rate at ``pwf``.

    ``qwf`` keeps the ``InFlow`` library contract: ONE phase - the oil rate
    normally, the water rate in dewatering mode - NOT the total liquid rate
    that ``SimParams.qwf`` holds. Callers MUST pass
    ``SimulationParams.inflow_rate``, which picks the right phase.
    """
    return InFlow(qwf=qwf, pwf=pwf, pres=pres)


# mirrors woffl/gui/utils.py:create_pvt_components
def create_pvt_components(
    field_model: Optional[str] = None,
    oil_api: Optional[float] = None,
    gas_sg: Optional[float] = None,
    wat_sg: Optional[float] = None,
    bubble_point: Optional[float] = None,
) -> tuple[BlackOil, FormWater, FormGas]:
    """Create PVT components (oil, water, gas) for the given field model.

    Args:
        field_model: "Schrader" or "Kuparuk" (case-insensitive). Provides
            defaults for any override not explicitly supplied.
        oil_api: Optional per-well API gravity override.
        gas_sg: Optional gas specific gravity override. Sets BOTH the oil's
            solution-gas SG and the free-gas SG (same as the GUI).
        wat_sg: Optional water specific gravity override.
        bubble_point: Optional bubble point pressure override, psig.

    Returns:
        (BlackOil, FormWater, FormGas) instances.
    """
    if field_model is None:
        field_model = "schrader"
    field_model = field_model.lower()

    if field_model == "kuparuk":
        oil_default = BlackOil.kuparuk()
        wat_default = FormWater.kuparuk()
        gas_default = FormGas.kuparuk()
    else:
        oil_default = BlackOil.schrader()
        wat_default = FormWater.schrader()
        gas_default = FormGas.schrader()

    final_api = oil_api if oil_api is not None else oil_default.oil_api
    final_pbp = bubble_point if bubble_point is not None else oil_default.pbp
    final_oil_sg = gas_sg if gas_sg is not None else oil_default.gas_sg
    final_gas_sg = gas_sg if gas_sg is not None else gas_default.gas_sg
    final_wat_sg = wat_sg if wat_sg is not None else wat_default.wat_sg

    oil = BlackOil(oil_api=final_api, bubblepoint=final_pbp, gas_sg=final_oil_sg)
    water = FormWater(wat_sg=final_wat_sg)
    gas = FormGas(gas_sg=final_gas_sg)
    return oil, water, gas


# mirrors woffl/gui/utils.py:create_reservoir_mix
def create_reservoir_mix(
    wc: float,
    gor: float,
    temp: float,
    field_model: Optional[str] = None,
    oil_api: Optional[float] = None,
    gas_sg: Optional[float] = None,
    wat_sg: Optional[float] = None,
    bubble_point: Optional[float] = None,
    model_as_water: bool = False,
) -> ResMix:
    """Create a ResMix object with the given parameters.

    Args:
        wc: Water cut, fraction.
        gor: Formation GOR, scf/bbl.
        temp: Formation temperature, deg F (carried by callers, not ResMix).
        field_model: PVT preset selector.
        oil_api: Optional API gravity override.
        gas_sg: Optional gas SG override (sets oil AND gas SG).
        wat_sg: Optional water SG override.
        bubble_point: Optional bubble point override, psig.
        model_as_water: opt-in water-pump mode for a 100%-water (no-oil) well.

    Returns:
        ResMix instance.
    """
    oil, water, gas = create_pvt_components(
        field_model=field_model,
        oil_api=oil_api,
        gas_sg=gas_sg,
        wat_sg=wat_sg,
        bubble_point=bubble_point,
    )
    return ResMix(wc=wc, fgor=gor, oil=oil, wat=water, gas=gas, model_as_water=model_as_water)


# mirrors woffl/gui/utils.py:run_jetpump_solver (prop_pf construction) - the
# power fluid is always the field model's FormWater preset conditioned at
# 0 psig / 60 degF.
def power_fluid(field_model: Optional[str]) -> FormWater:
    """Power-fluid water properties for the field model, conditioned (0, 60)."""
    _, prop_pf, _ = create_pvt_components(field_model)
    return prop_pf.condition(0, 60)


# mirrors woffl/gui/utils.py:create_well_profile - the preset-model path used
# when no survey CSV exists (or the caller is Custom).
def _preset_well_profile(field_model: Optional[str], jpump_tvd: Optional[float]) -> WellProfile:
    """WellProfile from the field-model preset, rebuilt at ``jpump_tvd``."""
    model = (field_model or "schrader").lower()
    well_profile = WellProfile.kuparuk() if model == "kuparuk" else WellProfile.schrader()

    if jpump_tvd is not None:
        try:
            jpump_md = well_profile.md_interp(jpump_tvd)
            well_profile = WellProfile(
                md_list=well_profile.md_ray,
                vd_list=well_profile.vd_ray,
                jetpump_md=jpump_md,
            )
        except ValueError as exc:
            log.warning(
                "jetpump_tvd=%s is outside the %s profile's range (%s); using the default jetpump MD",
                jpump_tvd,
                model,
                exc,
            )
    return well_profile


# maxsize mirrors the Streamlit site: keyed on (well, jpump_tvd, field_model),
# a ~130-well fleet needs headroom for several TVD variants per well and the
# Nelder-Mead profile fit costs 12-412 ms per miss.
# A measured pump MD counts as "at" the requested TVD when the survey's TVD at
# that MD is within this many feet of it. Chars JP_TVD is itself interpolated
# from JP_MD along the same survey, so the normal case matches to well under a
# foot; a sidebar TVD override lands outside and falls back to the crossing.
_JP_MD_MATCH_FT = 5.0


def _chars_jp_md(well: str) -> Optional[float]:
    """The well's measured pump MD (chars ``JP_MD``), or None on any failure."""
    try:
        df, _source = datasources.well_chars_safe()
    except Exception:  # noqa: BLE001 - chars unavailable is a normal fallback
        return None
    if "JP_MD" not in df.columns:
        return None
    rows = df[df["Well"] == well]
    if rows.empty:
        return None
    val = rows.iloc[0].get("JP_MD")
    try:
        num = float(val)
    except (TypeError, ValueError):
        return None
    return num if np.isfinite(num) and num > 0 else None


def resolve_jetpump_md(
    well: Optional[str], jpump_tvd: float, md_list, vd_list
) -> Optional[float]:
    """Measured depth to place the jet pump on a survey (review 2026-09-01, #1).

    Truth first: when the chars frame carries the MEASURED ``JP_MD`` and the
    survey's TVD there agrees with the requested ``jpump_tvd`` (the normal
    case - chars JP_TVD was interpolated from that very MD), the measured MD is
    used verbatim. Otherwise (a sidebar TVD override, or no chars MD) the
    shallowest survey crossing of ``jpump_tvd`` is used. Never ``np.interp``
    over the full survey - see ``depth_interp.first_crossing_md``.

    Args:
        well: GUI well name, or None for Custom.
        jpump_tvd: Requested jetpump true vertical depth, ft.
        md_list: Survey station measured depths, ft.
        vd_list: Survey station true vertical depths, ft.

    Returns:
        Jetpump MD, ft, or None when the survey never reaches ``jpump_tvd``.
    """
    from server.services.depth_interp import first_crossing_md

    md = np.asarray(md_list, dtype=float)
    vd = np.asarray(vd_list, dtype=float)
    if well is not None:
        measured = _chars_jp_md(well)
        if measured is not None and md.size and md[0] <= measured <= md[-1]:
            order = np.argsort(md, kind="stable")
            vd_at_measured = float(np.interp(measured, md[order], vd[order]))
            if abs(vd_at_measured - float(jpump_tvd)) <= _JP_MD_MATCH_FT:
                return float(measured)
    return first_crossing_md(md, vd, float(jpump_tvd))


# mirrors woffl/gui/utils.py:create_well_profile_from_survey
@ttl_cache(config.TTL_PROFILES, maxsize=512)
def build_well_profile(well: Optional[str], jpump_tvd: float, field_model: str) -> WellProfile:
    """WellProfile from the well's deviation survey, else the preset model.

    Args:
        well: GUI well name, or None for Custom (always uses the preset).
        jpump_tvd: Jetpump true vertical depth, ft. Converted to MD by
            ``resolve_jetpump_md`` (measured chars JP_MD when it agrees, else
            the shallowest survey crossing - never np.interp on a toe-up
            survey).
        field_model: "Schrader" or "Kuparuk" preset used as fallback.

    Returns:
        WellProfile positioned at the requested jetpump TVD.
    """
    if well is not None:
        survey_data = datasources.survey(well)
        if survey_data is not None and not survey_data.empty:
            try:
                md_list = survey_data["meas_depth"].tolist()
                tvd_list = survey_data["tvd_depth"].tolist()
                jpump_md = resolve_jetpump_md(well, jpump_tvd, md_list, tvd_list)
                if jpump_md is None:
                    raise ValueError(
                        f"survey never reaches {jpump_tvd:.0f} ft TVD "
                        f"(max {max(tvd_list):.0f} ft)"
                    )
                return WellProfile(md_list=md_list, vd_list=tvd_list, jetpump_md=jpump_md)
            except Exception as exc:
                log.warning(
                    "Error creating well profile from survey data for %s: %s. Using default model.",
                    well,
                    exc,
                )
    return _preset_well_profile(field_model, jpump_tvd)


def build_sim_objects(
    sp: schemas.SimParams, well: str
) -> tuple[JetPump, PipeInPipe, InFlow, ResMix, WellProfile]:
    """Build every physics object one simulation needs, from one SimParams.

    Goes through ``sp.to_simulation_params(well)`` so derived rates stay
    canonical: ``inflow_rate`` is the SINGLE-PHASE rate (oil normally, water
    in dewatering mode) the InFlow library contract requires - never the
    total-liquid ``qwf`` the sidebar holds.

    Args:
        sp: API simulation parameters.
        well: Selected well name ("Custom" for no survey).

    Returns:
        (jetpump, wellbore, inflow, res_mix, well_profile).
    """
    p = sp.to_simulation_params(well)
    jetpump = create_jetpump(p.nozzle_no, p.area_ratio, p.ken, p.kth, p.kdi)
    apply_nozzle_area_factor(jetpump, sp.nozzle_area_factor)
    _tube, _case, wellbore = create_pipes(
        tubing_od=p.tubing_od,
        tubing_thickness=p.tubing_thickness,
        casing_od=p.casing_od,
        casing_thickness=p.casing_thickness,
    )
    inflow = create_inflow(p.inflow_rate, p.pwf, p.pres)
    res_mix = create_reservoir_mix(
        p.form_wc,
        p.form_gor,
        p.form_temp,
        field_model=p.field_model,
        oil_api=p.oil_api,
        gas_sg=p.gas_sg,
        wat_sg=p.wat_sg,
        bubble_point=p.bubble_point,
        model_as_water=p.model_as_water,
    )
    wp = build_well_profile(
        None if well == "Custom" else well, float(p.jpump_tvd), p.field_model
    )
    return jetpump, wellbore, inflow, res_mix, wp
