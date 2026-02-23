"""Sidebar Input Widgets

Extracts all sidebar parameter collection from the monolithic app.py main()
into a dedicated module. Returns a SimulationParams dataclass and run button state.
"""

import streamlit as st
from woffl.gui.params import SimulationParams
from woffl.gui.utils import get_available_wells, get_well_data


def _update_well_parameters_from_data(well_data: dict | None, selected_well: str) -> None:
    """Update session state parameters when well selection changes.

    Args:
        well_data: Dictionary containing well characteristics from CSV
        selected_well: Name of the selected well
    """
    if selected_well == "Custom" or not well_data:
        return

    is_new_well = selected_well != st.session_state.get("last_selected_well_all", "Custom")

    if is_new_well:
        st.session_state.tubing_od = float(well_data.get("out_dia", 4.5))
        st.session_state.tubing_thickness = float(well_data.get("thick", 0.5))
        st.session_state.form_temp = int(well_data.get("form_temp", 70))
        st.session_state.jpump_tvd = int(well_data.get("JP_TVD", 4065))
        st.session_state.res_pres = int(well_data.get("res_pres", 1700))
        st.session_state.field_model_index = 0 if well_data.get("is_sch", True) else 1
        st.session_state.last_selected_well_all = selected_well


def _on_well_change() -> None:
    """Callback function when well selection changes."""
    st.session_state.selected_well = st.session_state.well_selector
    if hasattr(st.session_state, "well_data"):
        del st.session_state.well_data


def _render_well_selection() -> tuple[str, dict | None]:
    """Render well selection widgets and return selected well + data.

    Returns:
        Tuple of (selected_well_name, well_data_dict_or_None)
    """
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
        if "well_data" not in st.session_state or st.session_state.get("current_well") != selected_well:
            st.session_state.well_data = get_well_data(selected_well)
            st.session_state.current_well = selected_well

        well_data = st.session_state.well_data
        _update_well_parameters_from_data(well_data, selected_well)

        if well_data:
            st.info(f"✅ Loaded data for {selected_well}")
            with st.expander("Well Information"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Field Model:** {'Schrader' if well_data.get('is_sch', True) else 'Kuparuk'}")
                    st.write(f"**Tubing OD:** {well_data.get('out_dia', 'N/A')} inches")
                    st.write(f"**Tubing Thickness:** {well_data.get('thick', 'N/A')} inches")
                    st.write(f"**Reservoir Pressure:** {well_data.get('res_pres', 'N/A')} psi")
                with col2:
                    st.write(f"**Formation Temp:** {well_data.get('form_temp', 'N/A')} °F")
                    st.write(f"**Jetpump TVD:** {well_data.get('JP_TVD', 'N/A')} ft")
                    st.write(f"**Jetpump MD:** {well_data.get('JP_MD', 'N/A')} ft")
        else:
            st.warning(f"Could not load data for {selected_well}")

    return selected_well, well_data


def _render_jetpump_params() -> tuple[str, str, float, float, float]:
    """Render jetpump parameter widgets.

    Returns:
        Tuple of (nozzle_no, area_ratio, ken, kth, kdi)
    """
    st.subheader("Jetpump Parameters")
    nozzle_options = ["8", "9", "10", "11", "12", "13", "14", "15"]
    nozzle_no = st.selectbox("Nozzle Size", nozzle_options, index=nozzle_options.index("12"))

    area_ratio_options = ["X", "A", "B", "C", "D", "E"]
    area_ratio = st.selectbox("Area Ratio (Throat Size)", area_ratio_options, index=2)

    ken = st.slider("Nozzle Loss Coefficient (ken)", 0.01, 0.10, 0.03, 0.01)
    kth = st.slider("Throat Loss Coefficient (kth)", 0.1, 0.5, 0.3, 0.1)
    kdi = st.slider("Diffuser Loss Coefficient (kdi)", 0.1, 0.5, 0.4, 0.1)

    return nozzle_no, area_ratio, ken, kth, kdi


def _render_pipe_params(well_data: dict | None) -> tuple[float, float, float, float]:
    """Render pipe parameter widgets.

    Returns:
        Tuple of (tubing_od, tubing_thickness, casing_od, casing_thickness)
    """
    st.subheader("Pipe Parameters")

    if "tubing_od" not in st.session_state:
        st.session_state.tubing_od = 4.5
    if "tubing_thickness" not in st.session_state:
        st.session_state.tubing_thickness = 0.5

    tubing_od = st.number_input(
        "Tubing Outer Diameter (inches)",
        value=st.session_state.tubing_od,
        min_value=2.0,
        max_value=9.0,
        step=0.1,
        format="%.3f",
        help="Auto-populated from well data" if well_data else None,
        key="tubing_od_input",
    )
    tubing_thickness = st.number_input(
        "Tubing Wall Thickness (inches)",
        value=st.session_state.tubing_thickness,
        min_value=0.1,
        max_value=2.0,
        step=0.1,
        format="%.3f",
        help="Auto-populated from well data" if well_data else None,
        key="tubing_thickness_input",
    )

    st.session_state.tubing_od = tubing_od
    st.session_state.tubing_thickness = tubing_thickness

    casing_od = st.number_input(
        "Casing Outer Diameter (inches)", value=6.875, min_value=4.0, max_value=17.0, step=0.125, format="%.3f"
    )
    casing_thickness = st.number_input(
        "Casing Wall Thickness (inches)", value=0.5, min_value=0.1, max_value=2.0, step=0.1, format="%.3f"
    )

    return tubing_od, tubing_thickness, casing_od, casing_thickness


def _render_formation_params(well_data: dict | None) -> tuple[float, int, int]:
    """Render formation parameter widgets.

    Returns:
        Tuple of (form_wc, form_gor, form_temp)
    """
    st.subheader("Formation Parameters")
    form_wc = st.number_input("Water Cut (form_wc)", value=0.50, min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
    form_gor = st.number_input("Gas-Oil Ratio (form_gor)", value=250, min_value=20, max_value=10000, step=25)

    if "form_temp" not in st.session_state:
        st.session_state.form_temp = 70

    form_temp = st.number_input(
        "Formation Temperature (form_temp, °F)",
        value=st.session_state.form_temp,
        min_value=32,
        max_value=350,
        step=1,
        help="Auto-populated from well data" if well_data else None,
        key="form_temp_input",
    )
    st.session_state.form_temp = form_temp

    return form_wc, form_gor, form_temp


def _render_well_params(well_data: dict | None) -> tuple[int, int, float, int]:
    """Render well parameter widgets.

    Returns:
        Tuple of (surf_pres, jpump_tvd, rho_pf, ppf_surf)
    """
    st.subheader("Well Parameters")
    surf_pres = st.number_input("Surface Pressure (psi)", min_value=10, max_value=600, value=210, step=10)

    if "jpump_tvd" not in st.session_state:
        st.session_state.jpump_tvd = 4065

    jpump_tvd = st.number_input(
        "Jetpump TVD (feet)",
        value=st.session_state.jpump_tvd,
        min_value=2500,
        max_value=8000,
        step=10,
        help="Auto-populated from well data" if well_data else None,
        key="jpump_tvd_input",
    )
    st.session_state.jpump_tvd = jpump_tvd

    rho_pf = st.number_input("Power Fluid Density (lbm/ft³)", min_value=50.0, max_value=70.0, value=62.4, step=0.1)
    ppf_surf = st.number_input(
        "Power Fluid Surface Pressure (psi)", min_value=2000, max_value=4000, value=3168, step=10
    )

    return surf_pres, jpump_tvd, rho_pf, ppf_surf


def _render_inflow_params(well_data: dict | None) -> tuple[int, int, int]:
    """Render inflow parameter widgets.

    Returns:
        Tuple of (qwf, pwf, pres)
    """
    st.subheader("Inflow Parameters")
    qwf = st.number_input("Oil Rate at FBHP (qwf, BOPD)", min_value=100, max_value=6000, value=750, step=10)
    pwf = st.number_input(
        "Flowing Bottom Hole Pressure @ qwf (pwf, psi)", min_value=100, max_value=2500, value=500, step=10
    )

    if "res_pres" not in st.session_state:
        st.session_state.res_pres = 1700

    pres = st.number_input(
        "Reservoir Pressure (pres, psi)",
        value=st.session_state.res_pres,
        min_value=400,
        max_value=5000,
        step=10,
        help="Auto-populated from well data" if well_data else None,
        key="res_pres_input",
    )
    st.session_state.res_pres = pres

    return qwf, pwf, pres


def _render_batch_params() -> tuple[list[str], list[str], str]:
    """Render batch run parameter widgets.

    Returns:
        Tuple of (nozzle_batch_options, throat_batch_options, water_type)
    """
    st.subheader("Batch Run Parameters")
    nozzle_batch_options = st.multiselect(
        "Nozzle Sizes to Test",
        options=["8", "9", "10", "11", "12", "13", "14", "15"],
        default=["9", "10", "11", "12", "13", "14", "15"],
    )

    throat_batch_options = st.multiselect(
        "Throat Ratios to Test", options=["X", "A", "B", "C", "D", "E"], default=["A", "B", "C", "D"]
    )

    water_type = st.radio(
        "Water Type for Analysis",
        options=["lift", "total"],
        index=1,
        help="'Lift' shows power fluid water, 'Total' shows power fluid + formation water",
    )

    return nozzle_batch_options, throat_batch_options, water_type


def _render_power_fluid_range_params() -> tuple[int, int, int]:
    """Render power fluid range analysis parameter widgets.

    Returns:
        Tuple of (power_fluid_min, power_fluid_max, power_fluid_step)
    """
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


def render_sidebar() -> tuple[bool, SimulationParams]:
    """Render the complete sidebar and collect all parameters.

    Returns:
        Tuple of (run_button_pressed, simulation_params)
    """
    with st.sidebar:
        run_button = st.button("Run Simulation")
        st.sidebar.header("Parameters")

        # Well selection
        selected_well, well_data = _render_well_selection()

        # Marginal watercut
        marginal_watercut = st.number_input(
            "Field Marginal Watercut",
            min_value=0.0,
            max_value=1.0,
            value=0.94,
            step=0.01,
            format="%.2f",
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
        nozzle_no, area_ratio, ken, kth, kdi = _render_jetpump_params()
        tubing_od, tubing_thickness, casing_od, casing_thickness = _render_pipe_params(well_data)
        form_wc, form_gor, form_temp = _render_formation_params(well_data)
        surf_pres, jpump_tvd, rho_pf, ppf_surf = _render_well_params(well_data)
        qwf, pwf, pres = _render_inflow_params(well_data)
        nozzle_batch_options, throat_batch_options, water_type = _render_batch_params()
        power_fluid_min, power_fluid_max, power_fluid_step = _render_power_fluid_range_params()

    params = SimulationParams(
        nozzle_no=nozzle_no,
        area_ratio=area_ratio,
        ken=ken,
        kth=kth,
        kdi=kdi,
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
