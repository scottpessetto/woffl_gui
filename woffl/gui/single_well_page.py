"""Single-Well Analysis Page

Orchestrates the four analysis tabs for single-well jetpump simulation.
Creates simulation objects from parameters and delegates to tab renderers.
"""

import streamlit as st
from woffl.gui.params import SimulationParams
from woffl.gui.tabs import batch_run, jetpump_solver, power_fluid_range, well_profile
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
        jetpump = create_jetpump(params.nozzle_no, params.area_ratio, params.ken, params.kth, params.kdi)
        tube, case, ann = create_pipes(
            params.tubing_od, params.tubing_thickness, params.casing_od, params.casing_thickness
        )
        inflow = create_inflow(params.qwf, params.pwf, params.pres)
        res_mix = create_reservoir_mix(params.form_wc, params.form_gor, params.form_temp, params.field_model)

        # Use survey data for well profile if a specific well is selected
        if params.selected_well != "Custom":
            wp = create_well_profile_from_survey(params.selected_well, params.jpump_tvd, params.field_model)
            survey_data = get_well_survey_data(params.selected_well)
            if survey_data is not None and not survey_data.empty:
                st.info(f"✅ Using actual survey data for {params.selected_well}")
            else:
                st.info(f"⚠️ Using default model for {params.selected_well} (survey data not available)")
        else:
            wp = create_well_profile(params.field_model, params.jpump_tvd)

        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(
            ["Jetpump Solution", "Batch Run", "Power Fluid Range Analysis", "Well Profile"]
        )

        with tab1:
            jetpump_solver.render_tab(params, jetpump, tube, wp, inflow, res_mix)

        with tab2:
            batch_run.render_tab(params, tube, wp, inflow, res_mix)

        with tab3:
            power_fluid_range.render_tab(params, tube, wp, inflow, res_mix)

        with tab4:
            well_profile.render_tab(params, wp)


def show_welcome_message() -> None:
    """Display the welcome/instructions message when no simulation is running."""
    st.info("Adjust the parameters in the sidebar and click 'Run Simulation' to generate results.")

    st.markdown(
        """
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
    """
    )
