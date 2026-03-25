"""Single-Well Analysis Page

Orchestrates the four analysis tabs for single-well jetpump simulation.
Creates simulation objects from parameters and delegates to tab renderers.
"""

import streamlit as st

from woffl.gui.params import SimulationParams
from woffl.gui.tabs import (
    batch_run,
    jetpump_solver,
    jp_history_tab,
    power_fluid_range,
    pressure_profile,
    pump_equivalent,
    well_profile,
)
from woffl.gui.utils import (
    create_inflow,
    create_jetpump,
    create_pipes,
    create_reservoir_mix,
    create_well_profile,
    create_well_profile_from_survey,
    get_well_survey_data,
)


def run_single_well_page(params: SimulationParams) -> None:
    """Run the single-well analysis page.

    Creates simulation objects from the collected parameters and renders
    the four analysis tabs.

    Args:
        params: Simulation parameters collected from the sidebar
    """
    with st.spinner("Running simulation..."):
        # Create simulation objects from params
        jetpump = create_jetpump(
            params.nozzle_no, params.area_ratio, params.ken, params.kth, params.kdi
        )
        tube, case, wellbore = create_pipes(
            params.tubing_od,
            params.tubing_thickness,
            params.casing_od,
            params.casing_thickness,
        )
        inflow = create_inflow(params.qwf, params.pwf, params.pres)
        res_mix = create_reservoir_mix(
            params.form_wc, params.form_gor, params.form_temp, params.field_model
        )

        # Use survey data for well profile if a specific well is selected
        if params.selected_well != "Custom":
            wp = create_well_profile_from_survey(
                params.selected_well, params.jpump_tvd, params.field_model
            )
            survey_data = get_well_survey_data(params.selected_well)
            if survey_data is not None and not survey_data.empty:
                st.info(f"✅ Using actual survey data for {params.selected_well}")
            else:
                st.info(
                    f"⚠️ Using default model for {params.selected_well} (survey data not available)"
                )
        else:
            wp = create_well_profile(params.field_model, params.jpump_tvd)

        # Build tab list — conditional tabs appended when data is available
        tab_labels = [
            "Jetpump Solution",
            "Batch Run",
            "Power Fluid Range Analysis",
            "Pressure Profile",
            "Well Profile",
            "Pump Equivalents",
        ]

        show_jp_tab = False
        jp_hist = st.session_state.get("jp_history_df")
        if jp_hist is not None and params.selected_well != "Custom":
            well_jp = jp_hist[jp_hist["Well Name"] == params.selected_well].dropna(
                subset=["Date Set"]
            )
            show_jp_tab = not well_jp.empty

        if show_jp_tab:
            tab_labels.append("JP History")

        tabs = st.tabs(tab_labels)

        with tabs[0]:
            jetpump_solver.render_tab(params, jetpump, wellbore, wp, inflow, res_mix)

        with tabs[1]:
            batch_run.render_tab(params, wellbore, wp, inflow, res_mix)

        with tabs[2]:
            power_fluid_range.render_tab(params, wellbore, wp, inflow, res_mix)

        with tabs[3]:
            pressure_profile.render_tab(
                params, jetpump, wellbore, wp, inflow, res_mix
            )

        with tabs[4]:
            well_profile.render_tab(params, wp)

        with tabs[5]:
            pump_equivalent.render_tab(params, jetpump)

        if show_jp_tab:
            with tabs[6]:
                jp_history_tab.render_tab(params)


def show_welcome_message() -> None:
    """Display the welcome/instructions message when no simulation is running."""
    st.info(
        "Adjust the parameters in the sidebar and click 'Run Simulation' to generate results."
    )

    st.markdown("""
    ## About WOFFL Jetpump Simulator
    
    This application provides a graphical user interface for the WOFFL package's jetpump functionality. 
    It allows you to simulate and visualize jetpump performance under various conditions.
    
    ### How to use:
    1. Select the field model (Schrader or Kuparuk)
    2. Adjust the parameters in the sidebar
    3. Click 'Run Simulation' to generate results
    4. View the results in the tabs
    
    ### Parameters:
    - **Field Model**: Choose between Schrader and Kuparuk models
    - **Jetpump Parameters**: Nozzle size, throat size, loss coefficients
    - **Pipe Parameters**: Tubing and casing dimensions
    - **Formation Parameters**: Water cut, gas-oil ratio, temperature
    - **Well Parameters**: Surface pressure, jetpump TVD, power fluid properties
    - **Inflow Parameters**: Flow rate, pressures
    - **Analysis Options**: Suction pressure range for multi-suction analysis
    """)
