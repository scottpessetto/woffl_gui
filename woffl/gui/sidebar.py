"""Sidebar Input Widgets

Extracts all sidebar parameter collection from the monolithic app.py main()
into a dedicated module. Returns a SimulationParams dataclass and run button state.
"""

import streamlit as st

from woffl.gui.params import NOZZLE_OPTIONS, THROAT_OPTIONS, SimulationParams
from woffl.gui.utils import get_available_wells, get_well_data


# ---------------------------------------------------------------------------
# Helpers for two-tier session state (logical key + widget key)
# ---------------------------------------------------------------------------


def _set_param(key: str, value) -> None:
    """Set both the logical state key and its widget key in session state.

    Streamlit keyed widgets ignore the ``value`` param after first render and
    read from ``st.session_state[key]`` instead, so both must be kept in sync.
    """
    st.session_state[key] = value
    st.session_state[f"{key}_input"] = value


def _number_input(label: str, key: str, default, **kwargs):
    """Render a ``st.number_input`` with two-tier session state.

    On first render (or after a page switch clears the widget key), the widget
    is initialized from the persisted logical state rather than a hardcoded
    default.  After render, the logical state is updated from the widget value.
    """
    widget_key = f"{key}_input"
    if widget_key not in st.session_state:
        st.session_state[widget_key] = st.session_state.get(key, default)
    value = st.number_input(label, key=widget_key, **kwargs)
    st.session_state[key] = value
    return value


def _clear_ipr_state() -> None:
    """Remove cached IPR analysis results from session state."""
    st.session_state.pop("sw_ipr_info", None)
    st.session_state.pop("sw_vogel_coeffs", None)


# ---------------------------------------------------------------------------
# Well data → session state population
# ---------------------------------------------------------------------------


def _update_well_parameters_from_data(
    well_data: dict | None, selected_well: str
) -> None:
    """Update session state parameters when well selection changes."""
    if selected_well == "Custom" or not well_data:
        _clear_ipr_state()
        return

    is_new_well = selected_well != st.session_state.get(
        "last_selected_well_all", "Custom"
    )

    if is_new_well:
        _set_param("tubing_od", float(well_data.get("out_dia", 4.5)))
        _set_param("tubing_thickness", float(well_data.get("thick", 0.5)))
        _set_param("form_temp", int(well_data.get("form_temp", 70)))
        _set_param("jpump_tvd", int(well_data.get("JP_TVD", 4065)))
        _set_param("res_pres", int(well_data.get("res_pres", 1700)))
        st.session_state.field_model_index = (
            0 if well_data.get("is_sch", True) else 1
        )

        _auto_populate_from_ipr(selected_well)
        st.session_state.last_selected_well_all = selected_well


def _auto_populate_from_ipr(selected_well: str) -> None:
    """Auto-populate sidebar inflow/formation params from well test IPR data.

    Runs IPR analysis (estimate_reservoir_pressure + compute_vogel_coefficients)
    for the selected well and sets sidebar session state keys accordingly.
    Only called once per well selection (inside the is_new_well guard).
    """
    all_tests = st.session_state.get("all_well_tests_df")
    if all_tests is None or all_tests.empty:
        _clear_ipr_state()
        return

    well_tests = all_tests[all_tests["well"] == selected_well].copy()
    if well_tests.empty or len(well_tests) < 2:
        _clear_ipr_state()
        return

    try:
        from woffl.assembly.ipr_analyzer import (
            compute_vogel_coefficients,
            estimate_reservoir_pressure,
        )

        merged_with_rp = estimate_reservoir_pressure(well_tests)
        vogel_coeffs = compute_vogel_coefficients(merged_with_rp)

        well_coeffs = vogel_coeffs[vogel_coeffs["Well"] == selected_well]
        if well_coeffs.empty:
            _clear_ipr_state()
            return

        coeff_row = well_coeffs.iloc[0]

        # Cache Vogel coefficients for use by jetpump solver tab
        st.session_state["sw_vogel_coeffs"] = coeff_row.to_dict()

        # Auto-populate sidebar values
        _set_param("form_wc", round(float(coeff_row["form_wc"]), 2))
        _set_param("form_gor", int(coeff_row["fgor"]))
        _set_param("qwf", int(coeff_row["qwf"] * (1 - coeff_row["form_wc"])))
        _set_param("pwf", int(coeff_row["pwf"]))
        _set_param("res_pres", int(coeff_row["ResP"]))

        # Auto-populate surface pressure from most recent test
        import math

        recent = well_tests.sort_values("WtDate", ascending=False).iloc[0]
        whp = recent.get("whp")
        if whp is not None and not (isinstance(whp, float) and math.isnan(whp)):
            _set_param("surf_pres", int(whp))

        # Store info message for display in the Well Information expander
        num_tests = int(coeff_row["num_tests"])
        most_recent = coeff_row["most_recent_date"]
        date_str = (
            most_recent.strftime("%Y-%m-%d")
            if hasattr(most_recent, "strftime")
            else str(most_recent)
        )
        st.session_state["sw_ipr_info"] = (
            f"IPR values loaded from {num_tests} well tests (most recent: {date_str})"
        )

    except Exception:
        _clear_ipr_state()


# ---------------------------------------------------------------------------
# Widget rendering helpers
# ---------------------------------------------------------------------------


def _on_well_change() -> None:
    """Callback function when well selection changes."""
    st.session_state.selected_well = st.session_state.well_selector
    if hasattr(st.session_state, "well_data"):
        del st.session_state.well_data


def _render_well_selection() -> tuple[str, dict | None]:
    """Render well selection widgets and return selected well + data."""
    st.subheader("Well Selection")
    available_wells = get_available_wells()

    if "selected_well" not in st.session_state:
        st.session_state.selected_well = "Custom"

    selected_well = st.selectbox(
        "Select Well:",
        options=available_wells,
        index=(
            available_wells.index(st.session_state.selected_well)
            if st.session_state.selected_well in available_wells
            else 0
        ),
        help="Choose a well to auto-populate parameters, or select 'Custom' for manual entry",
        key="well_selector",
        on_change=_on_well_change,
    )

    well_data = None
    if selected_well != "Custom":
        if (
            "well_data" not in st.session_state
            or st.session_state.get("current_well") != selected_well
        ):
            st.session_state.well_data = get_well_data(selected_well)
            st.session_state.current_well = selected_well

        well_data = st.session_state.well_data
        _update_well_parameters_from_data(well_data, selected_well)

        if well_data:
            st.info(f"✅ Loaded data for {selected_well}")
            with st.expander("Well Information"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(
                        f"**Field Model:** {'Schrader' if well_data.get('is_sch', True) else 'Kuparuk'}"
                    )
                    st.write(f"**Tubing OD:** {well_data.get('out_dia', 'N/A')} inches")
                    st.write(
                        f"**Tubing Thickness:** {well_data.get('thick', 'N/A')} inches"
                    )
                    st.write(
                        f"**Reservoir Pressure:** {well_data.get('res_pres', 'N/A')} psi"
                    )
                with col2:
                    st.write(
                        f"**Formation Temp:** {well_data.get('form_temp', 'N/A')} °F"
                    )
                    st.write(f"**Jetpump TVD:** {well_data.get('JP_TVD', 'N/A')} ft")
                    st.write(f"**Jetpump MD:** {well_data.get('JP_MD', 'N/A')} ft")
                ipr_info = st.session_state.get("sw_ipr_info")
                if ipr_info:
                    st.info(ipr_info)
        else:
            st.warning(f"Could not load data for {selected_well}")

    return selected_well, well_data


def _render_jetpump_params() -> tuple[str, str, float, float, float, str]:
    """Render jetpump parameter widgets."""
    st.subheader("Jetpump Parameters")

    if "jpump_direction" not in st.session_state:
        st.session_state.jpump_direction = "Reverse"
    jpump_direction = st.radio(
        "Circulation Direction",
        options=["Reverse", "Forward"],
        index=["Reverse", "Forward"].index(st.session_state.jpump_direction),
        help="Reverse: power fluid down annulus, production up tubing. Forward: power fluid down tubing, production up annulus.",
        key="jpump_direction_input",
    )
    st.session_state.jpump_direction = jpump_direction

    if "nozzle_no" not in st.session_state:
        st.session_state.nozzle_no = "12"
    nozzle_no = st.selectbox(
        "Nozzle Size",
        NOZZLE_OPTIONS,
        index=NOZZLE_OPTIONS.index(st.session_state.nozzle_no),
        key="nozzle_no_input",
    )
    st.session_state.nozzle_no = nozzle_no

    if "area_ratio" not in st.session_state:
        st.session_state.area_ratio = "B"
    area_ratio = st.selectbox(
        "Area Ratio (Throat Size)",
        THROAT_OPTIONS,
        index=THROAT_OPTIONS.index(st.session_state.area_ratio),
        key="area_ratio_input",
    )
    st.session_state.area_ratio = area_ratio

    if "ken" not in st.session_state:
        st.session_state.ken = 0.03
    ken = st.slider(
        "Nozzle Loss Coefficient (ken)",
        0.01,
        0.10,
        st.session_state.ken,
        0.01,
        key="ken_input",
    )
    st.session_state.ken = ken

    if "kth" not in st.session_state:
        st.session_state.kth = 0.3
    kth = st.slider(
        "Throat Loss Coefficient (kth)",
        0.1,
        0.5,
        st.session_state.kth,
        0.1,
        key="kth_input",
    )
    st.session_state.kth = kth

    if "kdi" not in st.session_state:
        st.session_state.kdi = 0.4
    kdi = st.slider(
        "Diffuser Loss Coefficient (kdi)",
        0.1,
        0.5,
        st.session_state.kdi,
        0.1,
        key="kdi_input",
    )
    st.session_state.kdi = kdi

    return nozzle_no, area_ratio, ken, kth, kdi, jpump_direction.lower()


def _render_pipe_params(well_data: dict | None) -> tuple[float, float, float, float]:
    """Render pipe parameter widgets."""
    st.subheader("Pipe Parameters")
    auto_help = "Auto-populated from well data" if well_data else None

    tubing_od = _number_input(
        "Tubing Outer Diameter (inches)", "tubing_od", 4.5,
        min_value=2.0, max_value=9.0, step=0.1, format="%.3f", help=auto_help,
    )
    tubing_thickness = _number_input(
        "Tubing Wall Thickness (inches)", "tubing_thickness", 0.5,
        min_value=0.1, max_value=2.0, step=0.1, format="%.3f", help=auto_help,
    )
    casing_od = _number_input(
        "Casing Outer Diameter (inches)", "casing_od", 6.875,
        min_value=4.0, max_value=17.0, step=0.125, format="%.3f",
    )
    casing_thickness = _number_input(
        "Casing Wall Thickness (inches)", "casing_thickness", 0.5,
        min_value=0.1, max_value=2.0, step=0.1, format="%.3f",
    )
    return tubing_od, tubing_thickness, casing_od, casing_thickness


def _render_formation_params(well_data: dict | None) -> tuple[float, int, int]:
    """Render formation parameter widgets."""
    st.subheader("Formation Parameters")
    auto_help = "Auto-populated from well data" if well_data else None

    form_wc = _number_input(
        "Water Cut (form_wc)", "form_wc", 0.50,
        min_value=0.0, max_value=1.0, step=0.01, format="%.2f",
    )
    form_gor = _number_input(
        "Gas-Oil Ratio (form_gor)", "form_gor", 250,
        min_value=20, max_value=10000, step=25,
    )
    form_temp = _number_input(
        "Formation Temperature (form_temp, °F)", "form_temp", 70,
        min_value=32, max_value=350, step=1, help=auto_help,
    )
    return form_wc, form_gor, form_temp


def _render_well_params(well_data: dict | None) -> tuple[int, int, float, int]:
    """Render well parameter widgets."""
    st.subheader("Well Parameters")
    auto_help = "Auto-populated from well data" if well_data else None

    ppf_surf = _number_input(
        "Power Fluid Surface Pressure (psi)", "ppf_surf", 3168,
        min_value=1500, max_value=4000, step=10,
    )
    surf_pres = _number_input(
        "Surface Pressure (psi)", "surf_pres", 210,
        min_value=10, max_value=600, step=10,
    )
    jpump_tvd = _number_input(
        "Jetpump TVD (feet)", "jpump_tvd", 4065,
        min_value=2500, max_value=8000, step=10, help=auto_help,
    )
    rho_pf = _number_input(
        "Power Fluid Density (lbm/ft³)", "rho_pf", 62.4,
        min_value=50.0, max_value=70.0, step=0.1,
    )
    return surf_pres, jpump_tvd, rho_pf, ppf_surf


def _render_inflow_params(well_data: dict | None) -> tuple[int, int, int]:
    """Render inflow parameter widgets."""
    st.subheader("Inflow Parameters")
    auto_help = "Auto-populated from well data" if well_data else None

    qwf = _number_input(
        "Oil Rate at FBHP (qwf, BOPD)", "qwf", 750,
        min_value=100, max_value=6000, step=10,
    )
    pwf = _number_input(
        "Flowing Bottom Hole Pressure @ qwf (pwf, psi)", "pwf", 500,
        min_value=100, max_value=2500, step=10,
    )
    pres = _number_input(
        "Reservoir Pressure (pres, psi)", "res_pres", 1700,
        min_value=400, max_value=5000, step=10, help=auto_help,
    )
    return qwf, pwf, pres


def _render_batch_params() -> tuple[list[str], list[str], str]:
    """Render batch run parameter widgets."""
    st.subheader("Batch Run Parameters")
    nozzle_batch_options = st.multiselect(
        "Nozzle Sizes to Test",
        options=NOZZLE_OPTIONS,
        default=["9", "10", "11", "12", "13", "14", "15"],
    )

    throat_batch_options = st.multiselect(
        "Throat Ratios to Test", options=THROAT_OPTIONS, default=["A", "B", "C", "D"]
    )

    water_type = st.radio(
        "Water Type for Analysis",
        options=["lift", "total"],
        index=1,
        help="'Lift' shows power fluid water, 'Total' shows power fluid + formation water",
    )

    return nozzle_batch_options, throat_batch_options, water_type


def _render_power_fluid_range_params() -> tuple[int, int, int]:
    """Render power fluid range analysis parameter widgets."""
    st.subheader("Power Fluid Range Analysis")
    power_fluid_min = st.number_input(
        "Min Power Fluid Pressure (psi)",
        min_value=1000,
        max_value=5000,
        value=1800,
        step=100,
        help="Minimum power fluid pressure for range analysis",
    )
    power_fluid_max = st.number_input(
        "Max Power Fluid Pressure (psi)",
        min_value=1000,
        max_value=5000,
        value=3600,
        step=100,
        help="Maximum power fluid pressure for range analysis",
    )
    power_fluid_step = st.number_input(
        "Power Fluid Pressure Step (psi)",
        min_value=50,
        max_value=500,
        value=200,
        step=50,
        help="Step size for power fluid pressure range",
    )

    return power_fluid_min, power_fluid_max, power_fluid_step


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_sidebar() -> tuple[bool, SimulationParams]:
    """Render the complete sidebar and collect all parameters."""
    with st.sidebar:
        run_button = st.button("Run Simulation")
        st.sidebar.header("Parameters")

        # Well selection
        selected_well, well_data = _render_well_selection()

        # Marginal watercut
        marginal_watercut = _number_input(
            "Field Marginal Watercut", "marginal_watercut", 0.94,
            min_value=0.0, max_value=1.0, step=0.01, format="%.2f",
            help="Economic threshold for water handling in the field",
        )

        # Field model
        st.subheader("Field Model")
        if "field_model_index" not in st.session_state:
            st.session_state.field_model_index = 0
        field_model = st.radio(
            "Select Field Model:",
            options=["Schrader", "Kuparuk"],
            index=st.session_state.field_model_index,
            key="field_model_radio",
        )
        st.session_state.field_model_index = ["Schrader", "Kuparuk"].index(field_model)

        # Collect all parameter groups
        surf_pres, jpump_tvd, rho_pf, ppf_surf = _render_well_params(well_data)
        nozzle_no, area_ratio, ken, kth, kdi, jpump_direction = _render_jetpump_params()
        tubing_od, tubing_thickness, casing_od, casing_thickness = _render_pipe_params(
            well_data
        )
        form_wc, form_gor, form_temp = _render_formation_params(well_data)
        qwf, pwf, pres = _render_inflow_params(well_data)
        nozzle_batch_options, throat_batch_options, water_type = _render_batch_params()
        power_fluid_min, power_fluid_max, power_fluid_step = (
            _render_power_fluid_range_params()
        )

    params = SimulationParams(
        nozzle_no=nozzle_no,
        area_ratio=area_ratio,
        ken=ken,
        kth=kth,
        kdi=kdi,
        jpump_direction=jpump_direction,
        tubing_od=tubing_od,
        tubing_thickness=tubing_thickness,
        casing_od=casing_od,
        casing_thickness=casing_thickness,
        form_wc=form_wc,
        form_gor=form_gor,
        form_temp=form_temp,
        field_model=field_model,
        surf_pres=surf_pres,
        jpump_tvd=jpump_tvd,
        rho_pf=rho_pf,
        ppf_surf=ppf_surf,
        qwf=qwf,
        pwf=pwf,
        pres=pres,
        nozzle_batch_options=nozzle_batch_options,
        throat_batch_options=throat_batch_options,
        water_type=water_type,
        marginal_watercut=marginal_watercut,
        power_fluid_min=power_fluid_min,
        power_fluid_max=power_fluid_max,
        power_fluid_step=power_fluid_step,
        selected_well=selected_well,
        well_data=well_data,
    )

    return run_button, params
