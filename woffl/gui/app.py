"""WOFFL Streamlit GUI Application

This is the main entry point for the WOFFL Streamlit GUI application.
It provides a web interface for interacting with the WOFFL package's jetpump functionality.
"""

import os
import sys

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Multi-well optimization imports
from woffl.assembly.network_optimizer import (
    NetworkOptimizer,
    PowerFluidConstraint,
    load_wells_from_csv,
)
from woffl.assembly.optimization_algorithms import optimize
from woffl.gui.optimization_utils import get_template_csv_content
from woffl.gui.optimization_viz import (
    create_efficiency_scatter,
    create_marginal_rate_chart,
    create_oil_rate_bar_chart,
    create_power_fluid_pie_chart,
    create_pump_config_chart,
    create_watercut_comparison,
)
from woffl.gui.utils import (
    create_inflow,
    create_jetpump,
    create_pipes,
    create_reservoir_mix,
    create_well_profile,
    create_well_profile_from_survey,
    generate_choked_figures,
    generate_discharge_check,
    generate_multi_suction_graphs,
    generate_multi_throat_entry_books,
    get_available_wells,
    get_well_data,
    highlight_recommended_pump,
    recommend_jetpump,
    run_batch_pump,
    run_jetpump_solver,
    run_power_fluid_range_batch,
)


def main():
    """Main function for the Streamlit application."""
    st.set_page_config(
        page_title="WOFFL Jetpump Simulator",
        page_icon="🛢️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("WOFFL Jetpump Simulator")

    # Mode selection
    app_mode = st.radio(
        "Select Analysis Mode:",
        ["Single Well Analysis", "Multi-Well Optimization"],
        horizontal=True,
        help="Single Well: Analyze one well in detail. Multi-Well: Optimize pump sizing across multiple wells.",
    )

    if app_mode == "Multi-Well Optimization":
        # Run multi-well optimization interface
        from woffl.gui.multi_well_page import run_multi_well_optimization_page

        run_multi_well_optimization_page()
        return  # Exit to avoid running single-well interface

    # Sidebar - Input Parameters (Single Well Mode)
    with st.sidebar:

        run_button = st.button("Run Simulation")

        st.sidebar.header("Parameters")

        # Well Selection Section
        st.subheader("Well Selection")
        available_wells = get_available_wells()

        # Initialize session state for well selection if not exists
        if "selected_well" not in st.session_state:
            st.session_state.selected_well = "Custom"

        def on_well_change():
            """Callback function when well selection changes"""
            st.session_state.selected_well = st.session_state.well_selector
            # Clear any cached well data to force reload
            if hasattr(st.session_state, "well_data"):
                del st.session_state.well_data

        def update_well_parameters_from_data(well_data, selected_well):
            """Update all session state parameters when well selection changes

            Args:
                well_data (dict): Dictionary containing well characteristics from CSV
                selected_well (str): Name of the selected well
            """
            if selected_well == "Custom" or not well_data:
                return

            # Track if this is a new well selection
            is_new_well = selected_well != st.session_state.get("last_selected_well_all", "Custom")

            if is_new_well:
                # Update all parameters from well data
                st.session_state.tubing_od = float(well_data.get("out_dia", 4.5))
                st.session_state.tubing_thickness = float(well_data.get("thick", 0.5))
                st.session_state.form_temp = int(well_data.get("form_temp", 70))
                st.session_state.jpump_tvd = int(well_data.get("JP_TVD", 4065))
                st.session_state.res_pres = int(well_data.get("res_pres", 1700))
                st.session_state.field_model_index = 0 if well_data.get("is_sch", True) else 1
                st.session_state.last_selected_well_all = selected_well

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
            on_change=on_well_change,
        )

        # Get well data if a specific well is selected
        well_data = None
        if selected_well != "Custom":
            # Cache well data in session state to avoid repeated loading
            if "well_data" not in st.session_state or st.session_state.get("current_well") != selected_well:
                st.session_state.well_data = get_well_data(selected_well)
                st.session_state.current_well = selected_well

            well_data = st.session_state.well_data

            # Update all parameters from well data
            update_well_parameters_from_data(well_data, selected_well)

            if well_data:
                st.info(f"✅ Loaded data for {selected_well}")
                # Display well summary
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

        marginal_watercut = st.number_input(
            "Field Marginal Watercut",
            min_value=0.0,
            max_value=1.0,
            value=0.94,
            step=0.01,
            format="%.2f",
            help="Economic threshold for water handling in the field",
        )

        st.subheader("Field Model")

        # Initialize session state for field model if not exists
        if "field_model_index" not in st.session_state:
            st.session_state.field_model_index = 0  # Default to Schrader

        field_model = st.radio(
            "Select Field Model:",
            options=["Schrader", "Kuparuk"],
            index=st.session_state.field_model_index,
            key="field_model_radio",
        )

        # Update session state with current selection
        st.session_state.field_model_index = ["Schrader", "Kuparuk"].index(field_model)

        st.subheader("Jetpump Parameters")
        nozzle_options = ["8", "9", "10", "11", "12", "13", "14", "15"]
        nozzle_no = st.selectbox("Nozzle Size", nozzle_options, index=nozzle_options.index("12"))

        area_ratio_options = ["X", "A", "B", "C", "D", "E"]
        area_ratio = st.selectbox("Area Ratio (Throat Size)", area_ratio_options, index=2)

        ken = st.slider("Nozzle Loss Coefficient (ken)", 0.01, 0.10, 0.03, 0.01)
        kth = st.slider("Throat Loss Coefficient (kth)", 0.1, 0.5, 0.3, 0.1)
        kdi = st.slider("Diffuser Loss Coefficient (kdi)", 0.1, 0.5, 0.4, 0.1)

        st.subheader("Pipe Parameters")

        # Initialize session state for parameters if not exists
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

        # Update session state with current values
        st.session_state.tubing_od = tubing_od
        st.session_state.tubing_thickness = tubing_thickness
        casing_od = st.number_input(
            "Casing Outer Diameter (inches)", value=6.875, min_value=4.0, max_value=17.0, step=0.125, format="%.3f"
        )
        casing_thickness = st.number_input(
            "Casing Wall Thickness (inches)", value=0.5, min_value=0.1, max_value=2.0, step=0.1, format="%.3f"
        )

        st.subheader("Formation Parameters")
        form_wc = st.number_input(
            "Water Cut (form_wc)", value=0.50, min_value=0.0, max_value=1.0, step=0.01, format="%.2f"
        )
        form_gor = st.number_input("Gas-Oil Ratio (form_gor)", value=250, min_value=20, max_value=10000, step=25)

        # Initialize session state for formation temperature
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

        # Update session state
        st.session_state.form_temp = form_temp

        st.subheader("Well Parameters")
        surf_pres = st.number_input("Surface Pressure (psi)", min_value=10, max_value=600, value=210, step=10)

        # Initialize session state for jetpump TVD
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

        # Update session state
        st.session_state.jpump_tvd = jpump_tvd

        rho_pf = st.number_input("Power Fluid Density (lbm/ft³)", min_value=50.0, max_value=70.0, value=62.4, step=0.1)
        ppf_surf = st.number_input(
            "Power Fluid Surface Pressure (psi)", min_value=2000, max_value=4000, value=3168, step=10
        )

        st.subheader("Inflow Parameters")
        qwf = st.number_input("Well Flow Rate (qwf, bbl/day)", min_value=100, max_value=6000, value=750, step=10)
        pwf = st.number_input(
            "Flowing Bottom Hole Pressure (pwf, psi)", min_value=100, max_value=2500, value=500, step=10
        )

        # Initialize session state for reservoir pressure
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

        # Update session state
        st.session_state.res_pres = pres

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

    # Main content area
    if run_button:
        with st.spinner("Running simulation..."):
            # Create objects
            jetpump = create_jetpump(nozzle_no, area_ratio, ken, kth, kdi)
            tube, case, ann = create_pipes(tubing_od, tubing_thickness, casing_od, casing_thickness)
            inflow = create_inflow(qwf, pwf, pres)
            res_mix = create_reservoir_mix(form_wc, form_gor, form_temp, field_model)

            # Use survey data for well profile if a specific well is selected
            if selected_well != "Custom":
                well_profile = create_well_profile_from_survey(selected_well, jpump_tvd, field_model)
                # Check if survey data was successfully loaded by looking for survey data
                from woffl.gui.utils import get_well_survey_data

                survey_data = get_well_survey_data(selected_well)
                if survey_data is not None and not survey_data.empty:
                    st.info(f"✅ Using actual survey data for {selected_well}")
                else:
                    st.info(f"⚠️ Using default model for {selected_well} (survey data not available)")
            else:
                well_profile = create_well_profile(field_model, jpump_tvd)

            # Create tabs for different visualizations
            (
                tab1,
                tab2,
                tab3,
                tab4,
            ) = st.tabs(
                [
                    "Jetpump Solution",
                    "Batch Run",
                    "Power Fluid Range Analysis",
                    "Well Profile",
                ]
            )

            with tab1:
                st.subheader("Jetpump Solver Results")

                # Run the solver
                with st.spinner("Running jetpump solver..."):
                    solver_results = run_jetpump_solver(
                        surf_pres, form_temp, rho_pf, ppf_surf, jetpump, tube, well_profile, inflow, res_mix
                    )

                if solver_results:
                    psu, sonic_status, qoil_std, fwat_bwpd, qnz_bwpd, mach_te = solver_results

                    # Create two columns for the results
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Suction Pressure", f"{psu:.1f} psig")
                        st.metric("Oil Flow Rate", f"{qoil_std:.1f} BOPD")
                        st.metric("Formation Water Rate", f"{fwat_bwpd:.1f} BWPD")

                    with col2:
                        st.metric("Power Fluid Rate", f"{qnz_bwpd:.1f} BWPD")
                        st.metric("Throat Entry Mach", f"{mach_te:.3f}")
                        st.metric("Sonic Flow", "Yes" if sonic_status else "No")

                    # Add some explanatory text
                    if sonic_status:
                        st.info("The well is operating at critical flow conditions (sonic velocity at throat entry).")
                    else:
                        st.success("The well is operating at stable flow conditions.")
                else:
                    st.warning(
                        "The solver could not find a solution with the current parameters. "
                        "Try adjusting the input values."
                    )

            with tab2:
                st.subheader("Batch Pump Analysis")

                if not nozzle_batch_options or not throat_batch_options:
                    st.warning("Please select at least one nozzle size and one throat ratio for batch analysis.")
                else:
                    # Run the batch pump simulation
                    with st.spinner("Running batch pump simulation..."):
                        batch_pump = run_batch_pump(
                            surf_pres,
                            form_temp,
                            rho_pf,
                            ppf_surf,
                            tube,
                            well_profile,
                            inflow,
                            res_mix,
                            nozzle_batch_options,
                            throat_batch_options,
                            wellname=f"{field_model} Well",
                        )

                    if batch_pump:
                        # Create tabs for different visualizations
                        batch_tab1, batch_tab2, batch_tab3 = st.tabs(
                            ["Performance Graph", "Derivative Graph", "Data Table"]
                        )

                        with batch_tab1:
                            # Ensure water_type is a string
                            water = water_type if water_type is not None else "lift"
                            st.subheader(f"Jet Pump Performance ({water.capitalize()} Water)")
                            fig, ax = plt.subplots(figsize=(10, 6))

                            # Check if curve fitting was successful
                            has_curve_fit = hasattr(batch_pump, "coeff_totl") and hasattr(batch_pump, "coeff_lift")
                            curve = True if has_curve_fit else False

                            try:
                                batch_pump.plot_data(water=water, curve=curve, ax=ax)

                                # Get recommended jet pump based on marginal watercut
                                try:
                                    recommendation = recommend_jetpump(batch_pump, marginal_watercut, water)
                                    highlight_recommended_pump(ax, recommendation, water)

                                    # Add a section for the recommendation details
                                    st.subheader("Recommended Jet Pump")
                                    rec_col1, rec_col2 = st.columns(2)

                                    with rec_col1:
                                        st.metric("Nozzle Size", recommendation["nozzle"])
                                        st.metric("Throat Ratio", recommendation["throat"])
                                        st.metric("Oil Rate", f"{recommendation['qoil_std']:.1f} BOPD")

                                    with rec_col2:
                                        water_label = "Lift Water" if water == "lift" else "Total Water"
                                        st.metric(water_label, f"{recommendation['water_rate']:.1f} BWPD")
                                        st.metric("Marginal Watercut", f"{recommendation['marginal_ratio']:.3f}")

                                        if recommendation["recommendation_type"] == "best_available":
                                            st.warning(
                                                "Note: No jet pump meets the specified marginal watercut threshold. "
                                                "This is the best available option."
                                            )
                                except Exception as e:
                                    st.warning(f"Could not determine recommended jet pump: {str(e)}")

                                st.pyplot(fig)
                            except Exception as e:
                                st.error(f"Error generating performance plot: {str(e)}")
                                st.info(
                                    "Could not generate the performance plot. "
                                    "Try selecting different nozzle sizes and throat ratios."
                                )

                            st.markdown(
                                """
                            **Graph Explanation:**
                            - **Red points**: Semi-finalist jet pumps (no other pump produces more oil with less water)
                            - **Blue points**: Eliminated jet pumps (another pump produces more oil with less water)
                            - **Red dashed line**: Exponential curve fit of the semi-finalist pumps
                            - **Gold star**: Recommended jet pump based on marginal watercut threshold
                            - **Labels**: Show nozzle size + throat ratio (e.g., "12B" = nozzle 12, throat B)
                            """
                            )

                        with batch_tab2:
                            # Ensure water_type is a string
                            water = water_type if water_type is not None else "lift"
                            st.subheader(f"Marginal Oil-Water Ratio ({water.capitalize()} Water)")

                            # Check if curve fitting was successful
                            has_curve_fit = hasattr(batch_pump, "coeff_totl") and hasattr(batch_pump, "coeff_lift")

                            if has_curve_fit:
                                # Create a figure and pass it to plot_derv
                                try:
                                    # Create a temporary file to save the figure
                                    import os
                                    import tempfile

                                    with tempfile.TemporaryDirectory() as tmpdirname:
                                        fig_path = os.path.join(tmpdirname, "derv_plot.png")
                                        batch_pump.plot_derv(water=water, fig_path=fig_path)
                                        st.image(fig_path)
                                except Exception as e:
                                    st.error(f"Error generating derivative plot: {str(e)}")
                                    st.info(
                                        "The derivative plot requires at least two semi-finalist jet pumps "
                                        "to calculate the marginal oil-water ratio."
                                    )
                            else:
                                st.warning("Curve fitting failed, so the derivative plot cannot be generated.")
                                st.info(
                                    "Try selecting more nozzle sizes and throat ratios to improve curve fitting. "
                                    "At least two semi-finalist jet pumps are required."
                                )

                            st.markdown(
                                """
                            **Graph Explanation:**
                            - **Points**: Marginal oil-water ratio for each semi-finalist jet pump
                            - **Line**: Analytical derivative of the exponential curve fit
                            - **Marginal Oil-Water Ratio**: Additional oil production per additional barrel of water
                            """
                            )

                        with batch_tab3:
                            st.subheader("Jet Pump Performance Data")

                            # Filter to show only successful runs
                            df_display = batch_pump.df.copy()
                            df_display = df_display[~df_display["qoil_std"].isna()]

                            # Format the dataframe for display
                            df_display = df_display[
                                [
                                    "nozzle",
                                    "throat",
                                    "qoil_std",
                                    "form_wat",
                                    "lift_wat",
                                    "totl_wat",
                                    "psu_solv",
                                    "mach_te",
                                    "sonic_status",
                                    "semi",
                                ]
                            ]

                            df_display = df_display.rename(
                                columns={
                                    "nozzle": "Nozzle",
                                    "throat": "Throat",
                                    "qoil_std": "Oil Rate (BOPD)",
                                    "form_wat": "Formation Water (BWPD)",
                                    "lift_wat": "Lift Water (BWPD)",
                                    "totl_wat": "Total Water (BWPD)",
                                    "psu_solv": "Suction Pressure (psig)",
                                    "mach_te": "Throat Entry Mach",
                                    "sonic_status": "Sonic Flow",
                                    "semi": "Semi-Finalist",
                                }
                            )

                            # Round numeric columns
                            numeric_cols = [
                                "Oil Rate (BOPD)",
                                "Formation Water (BWPD)",
                                "Lift Water (BWPD)",
                                "Total Water (BWPD)",
                                "Suction Pressure (psig)",
                                "Throat Entry Mach",
                            ]
                            df_display[numeric_cols] = df_display[numeric_cols].round(1)

                            # Display the dataframe
                            st.dataframe(df_display, use_container_width=True)

                            # Add a download button
                            csv = df_display.to_csv(index=False)
                            st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name="jetpump_batch_results.csv",
                                mime="text/csv",
                            )

                            # Add a separator
                            st.markdown("---")

                            # Add the Jet Pump Recommender Results table
                            st.subheader("Jet Pump Recommender Results")

                            # Get semi-finalist pumps
                            semi_df = batch_pump.df[batch_pump.df["semi"]].copy()

                            if not semi_df.empty:
                                # Sort by oil rate
                                semi_df = semi_df.sort_values(by="qoil_std", ascending=True)

                                # Get the appropriate water column based on water type
                                water_col = "lift_wat" if water == "lift" else "totl_wat"
                                marg_col = "molwr" if water == "lift" else "motwr"

                                # Convert marginal oil-water ratios to marginal watercuts
                                # Original ratios are (bbl oil / bbl water)
                                # We need (bbl water / (bbl water + bbl oil))
                                semi_df["marginal_watercut"] = 1 / (1 + semi_df[marg_col])

                                # Create a new dataframe for display
                                recommender_df = semi_df[
                                    ["nozzle", "throat", "qoil_std", water_col, marg_col, "marginal_watercut"]
                                ].copy()

                                # Rename columns for display
                                water_label = "Lift Water (BWPD)" if water == "lift" else "Total Water (BWPD)"
                                ratio_label = (
                                    "Marginal Oil/Lift Water Ratio"
                                    if water == "lift"
                                    else "Marginal Oil/Total Water Ratio"
                                )

                                recommender_df = recommender_df.rename(
                                    columns={
                                        "nozzle": "Nozzle",
                                        "throat": "Throat",
                                        "qoil_std": "Oil Rate (BOPD)",
                                        water_col: water_label,
                                        marg_col: ratio_label,
                                        "marginal_watercut": "Marginal Watercut",
                                    }
                                )

                                # Round numeric columns
                                numeric_cols = [
                                    "Oil Rate (BOPD)",
                                    water_label,
                                    ratio_label,
                                    "Marginal Watercut",
                                ]
                                recommender_df[numeric_cols] = recommender_df[numeric_cols].round(3)

                                # Try to get the recommended pump
                                try:
                                    recommendation = recommend_jetpump(batch_pump, marginal_watercut, water)

                                    # Add a column to highlight the recommended pump
                                    recommender_df["Recommended"] = False
                                    recommended_mask = (recommender_df["Nozzle"] == recommendation["nozzle"]) & (
                                        recommender_df["Throat"] == recommendation["throat"]
                                    )
                                    recommender_df.loc[recommended_mask, "Recommended"] = True

                                    # Display the dataframe with the recommended pump highlighted
                                    st.dataframe(recommender_df, use_container_width=True)

                                    # Add a note about the recommended pump
                                    st.info(
                                        f"The recommended pump is highlighted: Nozzle {recommendation['nozzle']}, "
                                        f"Throat {recommendation['throat']} with a marginal watercut of "
                                        f"{recommendation['marginal_ratio']:.3f}"
                                    )

                                    if recommendation["recommendation_type"] == "best_available":
                                        st.warning(
                                            "Note: No jet pump meets the specified marginal watercut threshold. "
                                            "This is the best available option."
                                        )
                                except Exception as e:
                                    # Just display the dataframe without highlighting
                                    st.dataframe(recommender_df, use_container_width=True)
                                    st.warning(f"Could not determine recommended jet pump: {str(e)}")

                                # Add a download button for the recommender results
                                csv_recommender = recommender_df.to_csv(index=False)
                                st.download_button(
                                    label="Download Recommender Results CSV",
                                    data=csv_recommender,
                                    file_name="jetpump_recommender_results.csv",
                                    mime="text/csv",
                                )
                            else:
                                st.warning("No semi-finalist jet pumps found. Cannot display recommender results.")

            with tab3:
                st.subheader("Power Fluid Range Analysis")

                if not nozzle_batch_options or not throat_batch_options:
                    st.warning(
                        "Please select at least one nozzle size and one throat ratio for power fluid range analysis."
                    )
                elif power_fluid_min >= power_fluid_max:
                    st.error("Minimum power fluid pressure must be less than maximum power fluid pressure.")
                else:
                    # Validate power fluid step
                    if power_fluid_step <= 0 or power_fluid_step >= (power_fluid_max - power_fluid_min):
                        st.error("Power fluid step must be positive and less than the pressure range.")
                    else:
                        # Calculate number of pressure points
                        pressure_points = int((power_fluid_max - power_fluid_min) / power_fluid_step) + 1
                        total_combinations = pressure_points * len(nozzle_batch_options) * len(throat_batch_options)

                        st.info(
                            f"This analysis will test {pressure_points} pressure points with "
                            f"{len(nozzle_batch_options)} nozzles and {len(throat_batch_options)} throat ratios "
                            f"for a total of {total_combinations} combinations."
                        )

                        # Run the comprehensive power fluid range analysis
                        with st.spinner("Running comprehensive power fluid range analysis..."):
                            comprehensive_df = run_power_fluid_range_batch(
                                surf_pres,
                                form_temp,
                                rho_pf,
                                power_fluid_min,
                                power_fluid_max,
                                power_fluid_step,
                                tube,
                                well_profile,
                                inflow,
                                res_mix,
                                nozzle_batch_options,
                                throat_batch_options,
                                wellname=f"{field_model} Well",
                            )

                        if comprehensive_df is not None and not comprehensive_df.empty:
                            # Filter to show only successful runs
                            successful_df = comprehensive_df[~comprehensive_df["qoil_std"].isna()].copy()

                            if not successful_df.empty:
                                # Create summary statistics
                                st.subheader("Analysis Summary")

                                col1, col2, col3, col4 = st.columns(4)

                                with col1:
                                    st.metric("Total Combinations", len(comprehensive_df))
                                    st.metric("Successful Runs", len(successful_df))

                                with col2:
                                    st.metric("Max Oil Rate", f"{successful_df['qoil_std'].max():.1f} BOPD")
                                    st.metric("Min Oil Rate", f"{successful_df['qoil_std'].min():.1f} BOPD")

                                with col3:
                                    st.metric("Max Total Water", f"{successful_df['totl_wat'].max():.1f} BWPD")
                                    st.metric("Min Total Water", f"{successful_df['totl_wat'].min():.1f} BWPD")

                                with col4:
                                    st.metric("Pressure Range", f"{power_fluid_min}-{power_fluid_max} psi")
                                    st.metric("Success Rate", f"{len(successful_df)/len(comprehensive_df)*100:.1f}%")

                                # Create visualization tabs
                                viz_tab1, viz_tab2, viz_tab3 = st.tabs(
                                    ["Performance vs Pressure", "Comprehensive Data", "Best Performers"]
                                )

                                with viz_tab1:
                                    st.subheader("Oil Rate vs Power Fluid Pressure")

                                    # Create a scatter plot showing oil rate vs power fluid pressure
                                    fig, ax = plt.subplots(figsize=(12, 8))

                                    # Plot each nozzle-throat combination with different colors
                                    # Get unique nozzle-throat combinations
                                    successful_df["pump_combo"] = successful_df["nozzle"] + successful_df["throat"]
                                    unique_combos = successful_df["pump_combo"].unique()

                                    colors = cm.get_cmap("tab20")(np.linspace(0, 1, len(unique_combos)))

                                    for i, combo in enumerate(unique_combos):
                                        combo_data = successful_df[successful_df["pump_combo"] == combo]
                                        ax.scatter(
                                            combo_data["power_fluid_pressure"],
                                            combo_data["qoil_std"],
                                            c=[colors[i]],
                                            label=combo,
                                            alpha=0.7,
                                            s=50,
                                        )

                                    ax.set_xlabel("Power Fluid Pressure (psi)")
                                    ax.set_ylabel("Oil Rate (BOPD)")
                                    ax.set_title(f"{field_model} Well - Oil Rate vs Power Fluid Pressure")
                                    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
                                    ax.grid(True, alpha=0.3)

                                    st.pyplot(fig)

                                    # Add a second plot for total water vs pressure
                                    st.subheader("Total Water vs Power Fluid Pressure")

                                    fig2, ax2 = plt.subplots(figsize=(12, 8))

                                    for i, combo in enumerate(unique_combos):
                                        combo_data = successful_df[successful_df["pump_combo"] == combo]
                                        ax2.scatter(
                                            combo_data["power_fluid_pressure"],
                                            combo_data["totl_wat"],
                                            c=[colors[i]],
                                            label=combo,
                                            alpha=0.7,
                                            s=50,
                                        )

                                    ax2.set_xlabel("Power Fluid Pressure (psi)")
                                    ax2.set_ylabel("Total Water (BWPD)")
                                    ax2.set_title(f"{field_model} Well - Total Water vs Power Fluid Pressure")
                                    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
                                    ax2.grid(True, alpha=0.3)

                                    st.pyplot(fig2)

                                with viz_tab2:
                                    st.subheader("Comprehensive Analysis Results")

                                    # Format the dataframe for display
                                    display_df = successful_df[
                                        [
                                            "power_fluid_pressure",
                                            "nozzle",
                                            "throat",
                                            "qoil_std",
                                            "form_wat",
                                            "lift_wat",
                                            "totl_wat",
                                            "psu_solv",
                                            "mach_te",
                                            "sonic_status",
                                        ]
                                    ].copy()

                                    display_df = display_df.rename(
                                        columns={
                                            "power_fluid_pressure": "Power Fluid Pressure (psi)",
                                            "nozzle": "Nozzle",
                                            "throat": "Throat",
                                            "qoil_std": "Oil Rate (BOPD)",
                                            "form_wat": "Formation Water (BWPD)",
                                            "lift_wat": "Lift Water (BWPD)",
                                            "totl_wat": "Total Water (BWPD)",
                                            "psu_solv": "Suction Pressure (psig)",
                                            "mach_te": "Throat Entry Mach",
                                            "sonic_status": "Sonic Flow",
                                        }
                                    )

                                    # Round numeric columns
                                    numeric_cols = [
                                        "Power Fluid Pressure (psi)",
                                        "Oil Rate (BOPD)",
                                        "Formation Water (BWPD)",
                                        "Lift Water (BWPD)",
                                        "Total Water (BWPD)",
                                        "Suction Pressure (psig)",
                                        "Throat Entry Mach",
                                    ]
                                    display_df[numeric_cols] = display_df[numeric_cols].round(1)

                                    # Sort by power fluid pressure, then by oil rate
                                    display_df = display_df.sort_values(
                                        by=["Power Fluid Pressure (psi)", "Oil Rate (BOPD)"], ascending=[True, False]
                                    )

                                    # Display the dataframe
                                    st.dataframe(display_df, use_container_width=True)

                                    # Add a download button for the comprehensive results
                                    csv_comprehensive = display_df.to_csv(index=False)
                                    st.download_button(
                                        label="Download Comprehensive Results CSV",
                                        data=csv_comprehensive,
                                        file_name="jetpump_power_fluid_range_analysis.csv",
                                        mime="text/csv",
                                    )

                                with viz_tab3:
                                    st.subheader("Best Performers at Each Pressure")

                                    # Find the best performer at each pressure point
                                    best_performers = []
                                    for pressure in successful_df["power_fluid_pressure"].unique():
                                        pressure_data = successful_df[successful_df["power_fluid_pressure"] == pressure]
                                        if not pressure_data.empty:
                                            best_performer = pressure_data.loc[pressure_data["qoil_std"].idxmax()]
                                            best_performers.append(best_performer)

                                    if best_performers:
                                        best_df = pd.DataFrame(best_performers)

                                        # Format for display
                                        best_display_df = best_df[
                                            [
                                                "power_fluid_pressure",
                                                "nozzle",
                                                "throat",
                                                "qoil_std",
                                                "totl_wat",
                                                "psu_solv",
                                                "sonic_status",
                                            ]
                                        ].copy()

                                        best_display_df = best_display_df.rename(
                                            columns={
                                                "power_fluid_pressure": "Power Fluid Pressure (psi)",
                                                "nozzle": "Nozzle",
                                                "throat": "Throat",
                                                "qoil_std": "Oil Rate (BOPD)",
                                                "totl_wat": "Total Water (BWPD)",
                                                "psu_solv": "Suction Pressure (psig)",
                                                "sonic_status": "Sonic Flow",
                                            }
                                        )

                                        # Round numeric columns
                                        numeric_cols = [
                                            "Power Fluid Pressure (psi)",
                                            "Oil Rate (BOPD)",
                                            "Total Water (BWPD)",
                                            "Suction Pressure (psig)",
                                        ]
                                        best_display_df[numeric_cols] = best_display_df[numeric_cols].round(1)

                                        # Sort by power fluid pressure
                                        best_display_df = best_display_df.sort_values(by="Power Fluid Pressure (psi)")

                                        st.dataframe(best_display_df, use_container_width=True)

                                        # Add a download button for the best performers
                                        csv_best = best_display_df.to_csv(index=False)
                                        st.download_button(
                                            label="Download Best Performers CSV",
                                            data=csv_best,
                                            file_name="jetpump_best_performers_by_pressure.csv",
                                            mime="text/csv",
                                        )

                                        # Show overall best performer
                                        overall_best = best_df.loc[best_df["qoil_std"].idxmax()]
                                        st.subheader("Overall Best Performer")

                                        best_col1, best_col2, best_col3 = st.columns(3)

                                        with best_col1:
                                            st.metric(
                                                "Best Configuration",
                                                f"Nozzle {overall_best['nozzle']}, Throat {overall_best['throat']}",
                                            )
                                            st.metric(
                                                "Power Fluid Pressure",
                                                f"{overall_best['power_fluid_pressure']:.0f} psi",
                                            )

                                        with best_col2:
                                            st.metric("Oil Rate", f"{overall_best['qoil_std']:.1f} BOPD")
                                            st.metric("Total Water", f"{overall_best['totl_wat']:.1f} BWPD")

                                        with best_col3:
                                            st.metric("Suction Pressure", f"{overall_best['psu_solv']:.1f} psig")
                                            st.metric("Sonic Flow", "Yes" if overall_best["sonic_status"] else "No")

                                    else:
                                        st.warning("No best performers found in the analysis.")

                                st.markdown(
                                    """
                                    **Analysis Explanation:**
                                    - This comprehensive analysis tests all selected nozzle and throat combinations across the specified power fluid pressure range
                                    - **Performance vs Pressure**: Shows how oil and water rates change with power fluid pressure for each pump configuration
                                    - **Comprehensive Data**: Complete results table with all successful combinations
                                    - **Best Performers**: Shows the highest oil-producing pump at each pressure point and identifies the overall best performer
                                    """
                                )
                            else:
                                st.error("No successful simulation runs found. Check your parameter settings.")
                        else:
                            st.error("Failed to run power fluid range analysis. Check your parameters and try again.")

            with tab4:
                st.subheader("Well Profile Visualization")

                # Display well profile information
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Well Profile Information:**")
                    st.write(f"- Well Name: {selected_well if selected_well != 'Custom' else 'Generic ' + field_model}")
                    st.write(f"- Number of Survey Points: {len(well_profile.md_ray)}")
                    st.write(f"- MD Range: {well_profile.md_ray[0]:.1f} to {well_profile.md_ray[-1]:.1f} ft")
                    st.write(f"- TVD Range: {well_profile.vd_ray[0]:.1f} to {well_profile.vd_ray[-1]:.1f} ft")
                    st.write(f"- Jetpump MD: {well_profile.jetpump_md:.1f} ft")
                    st.write(f"- Jetpump TVD: {jpump_tvd:.1f} ft")

                    # Calculate deviation
                    max_deviation = max(
                        abs(well_profile.md_ray[i] - well_profile.vd_ray[i]) for i in range(len(well_profile.md_ray))
                    )
                    st.write(f"- Max Deviation: {max_deviation:.1f} ft")

                with col2:
                    # Check if using survey data
                    from woffl.gui.utils import get_well_survey_data

                    survey_data = get_well_survey_data(selected_well) if selected_well != "Custom" else None

                    if survey_data is not None and not survey_data.empty:
                        st.success("✅ Using Actual Survey Data")
                        st.write(f"- Survey Points: {len(survey_data)}")
                        if "inclination" in survey_data.columns:
                            max_inc = survey_data["inclination"].max()
                            st.write(f"- Max Inclination: {max_inc:.2f}°")
                        if "azimuth" in survey_data.columns:
                            st.write(
                                f"- Azimuth Range: {survey_data['azimuth'].min():.1f}° to {survey_data['azimuth'].max():.1f}°"
                            )
                    else:
                        st.info(f"ℹ️ Using Default {field_model} Model")
                        st.write("- Generic well profile")
                        st.write("- Simplified trajectory")

                # Create well profile plot
                st.subheader("Well Trajectory")

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

                # Plot 1: TVD vs MD
                ax1.plot(well_profile.md_ray, well_profile.vd_ray, "b-", linewidth=2, label="Well Path")
                ax1.axhline(
                    y=jpump_tvd, color="r", linestyle="--", linewidth=2, label=f"Jetpump TVD ({jpump_tvd:.0f} ft)"
                )
                ax1.axvline(
                    x=well_profile.jetpump_md,
                    color="g",
                    linestyle="--",
                    linewidth=2,
                    label=f"Jetpump MD ({well_profile.jetpump_md:.0f} ft)",
                )
                ax1.scatter(
                    [well_profile.jetpump_md],
                    [jpump_tvd],
                    color="red",
                    s=100,
                    zorder=5,
                    marker="*",
                    label="Jetpump Location",
                )
                ax1.set_xlabel("Measured Depth (ft)", fontsize=12)
                ax1.set_ylabel("True Vertical Depth (ft)", fontsize=12)
                ax1.set_title("Well Profile: TVD vs MD", fontsize=14, fontweight="bold")
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                ax1.invert_yaxis()  # Invert Y-axis so depth increases downward

                # Plot 2: Deviation (MD - TVD) vs Depth
                deviation = [well_profile.md_ray[i] - well_profile.vd_ray[i] for i in range(len(well_profile.md_ray))]
                ax2.plot(deviation, well_profile.vd_ray, "g-", linewidth=2)
                ax2.axhline(
                    y=jpump_tvd, color="r", linestyle="--", linewidth=2, label=f"Jetpump TVD ({jpump_tvd:.0f} ft)"
                )
                ax2.set_xlabel("Horizontal Deviation (ft)", fontsize=12)
                ax2.set_ylabel("True Vertical Depth (ft)", fontsize=12)
                ax2.set_title("Well Deviation Profile", fontsize=14, fontweight="bold")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                ax2.invert_yaxis()  # Invert Y-axis so depth increases downward

                plt.tight_layout()
                st.pyplot(fig)

                # Add inclination plot if survey data is available
                if survey_data is not None and not survey_data.empty and "inclination" in survey_data.columns:
                    st.subheader("Well Inclination Profile")

                    fig3, ax3 = plt.subplots(figsize=(12, 6))
                    ax3.plot(survey_data["meas_depth"], survey_data["inclination"], "b-", linewidth=2)
                    ax3.axvline(
                        x=well_profile.jetpump_md,
                        color="r",
                        linestyle="--",
                        linewidth=2,
                        label=f"Jetpump MD ({well_profile.jetpump_md:.0f} ft)",
                    )
                    ax3.set_xlabel("Measured Depth (ft)", fontsize=12)
                    ax3.set_ylabel("Inclination (degrees)", fontsize=12)
                    ax3.set_title("Well Inclination vs Measured Depth", fontsize=14, fontweight="bold")
                    ax3.legend()
                    ax3.grid(True, alpha=0.3)

                    plt.tight_layout()
                    st.pyplot(fig3)

                # Add explanation
                st.markdown(
                    """
                **Well Profile Explanation:**
                - **TVD vs MD Plot**: Shows the well's path from surface to total depth
                - **Deviation Plot**: Shows horizontal offset from vertical at each depth
                - **Inclination Plot**: Shows well angle from vertical (0° = vertical, 90° = horizontal)
                - **Red star**: Jetpump location
                - **Dashed lines**: Jetpump MD and TVD reference lines
                
                A vertical well would show MD = TVD (45° line on first plot).
                Deviation indicates how far the well has moved horizontally from the surface location.
                """
                )

                """
                Optimize jet pump sizing across multiple wells to maximize field oil production
                given a constrained power fluid supply.
                """

                # Power Fluid Constraint Input
                st.write("### Power Fluid Constraint")
                col1, col2 = st.columns(2)
                with col1:
                    total_pf = st.number_input(
                        "Total Available Power Fluid (bbl/day)",
                        min_value=0,
                        max_value=50000,
                        value=10000,
                        step=500,
                        help="Total power fluid capacity available for all wells",
                    )
                with col2:
                    pf_pressure = st.number_input(
                        "Power Fluid Pressure (psi)",
                        min_value=1000,
                        max_value=5000,
                        value=ppf_surf,
                        step=100,
                        help="Surface power fluid pressure",
                    )

                # CSV Upload
                st.write("### Well Configuration")
                uploaded_file = st.file_uploader(
                    "Upload Well Configuration CSV",
                    type=["csv"],
                    help="Upload a CSV file with well configurations. Wells in jp_chars.csv will auto-populate.",
                )

                # Download template
                template_csv = get_template_csv_content()
                st.download_button(
                    label="Download CSV Template",
                    data=template_csv,
                    file_name="well_optimization_template.csv",
                    mime="text/csv",
                    help="Download a template CSV with example wells",
                )

                #  Optimization Settings
                st.write("### Optimization Settings")
                col1, col2 = st.columns(2)
                with col1:
                    opt_method = st.selectbox(
                        "Optimization Method",
                        ["greedy", "proportional"],
                        index=0,
                        help="Greedy: Iteratively allocate to highest marginal producer. Proportional: Allocate based on productivity.",
                    )
                with col2:
                    opt_marginal_wc = st.number_input(
                        "Marginal Watercut Threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=marginal_watercut,
                        step=0.01,
                        format="%.2f",
                    )

                # Run Optimization Button
                if st.button("Run Multi-Well Optimization", type="primary") and uploaded_file:
                    try:
                        # Save uploaded file temporarily
                        import tempfile

                        with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as f:
                            f.write(uploaded_file.getvalue())
                            temp_csv_path = f.name

                        # Load wells and create optimizer
                        with st.spinner("Loading well configurations..."):
                            wells = load_wells_from_csv(temp_csv_path)
                            st.success(f"✅ Loaded {len(wells)} wells from CSV")

                        pf_constraint = PowerFluidConstraint(total_rate=total_pf, pressure=pf_pressure, rho_pf=rho_pf)

                        optimizer = NetworkOptimizer(
                            wells=wells,
                            power_fluid=pf_constraint,
                            nozzle_options=nozzle_batch_options if nozzle_batch_options else ["10", "11", "12"],
                            throat_options=throat_batch_options if throat_batch_options else ["B", "C", "D"],
                            marginal_watercut=opt_marginal_wc,
                        )

                        # Run batch simulations
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        def progress_callback(current, total, well_name):
                            progress_bar.progress(current / total if total > 0 else 0)
                            status_text.text(f"Processing {well_name}... ({current}/{total})")

                        batch_results = optimizer.run_all_batch_simulations(progress_callback)
                        progress_bar.empty()
                        status_text.empty()

                        st.success(f"✅ Completed batch simulations for {len(wells)} wells")

                        # Run optimization
                        with st.spinner(f"Running {opt_method} optimization..."):
                            results = optimize(optimizer, method=opt_method)

                        if results:
                            st.success(f"✅ Optimization complete! Allocated pumps to {len(results)} wells")

                            # Display results in tabs
                            opt_tab1, opt_tab2, opt_tab3, opt_tab4 = st.tabs(
                                ["Summary", "Well Details", "Visualizations", "Export"]
                            )

                            with opt_tab1:
                                st.write("### Field-Level Metrics")
                                metrics = optimizer.calculate_field_metrics()

                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Total Oil Rate", f"{metrics['total_oil_rate']:.1f} BOPD")
                                    st.metric("Wells Optimized", metrics["num_wells"])
                                with col2:
                                    st.metric("Total Water Rate", f"{metrics['total_water_rate']:.1f} BWPD")
                                    st.metric("Field Watercut", f"{metrics['field_watercut']:.1%}")
                                with col3:
                                    st.metric("Power Fluid Used", f"{metrics['total_power_fluid']:.1f} BWPD")
                                    st.metric("PF Utilization", f"{metrics['power_fluid_utilization']:.1%}")
                                with col4:
                                    st.metric("Avg Marginal Oil", f"{metrics['average_marginal_oil']:.3f}")
                                    st.metric("Sonic Wells", f"{metrics['num_sonic']}/{metrics['num_wells']}")

                            with opt_tab2:
                                st.write("### Well-Level Results")
                                results_df = optimizer.to_dataframe()
                                st.dataframe(results_df, use_container_width=True)

                            with opt_tab3:
                                st.write("### Optimization Visualizations")

                                viz_col1, viz_col2 = st.columns(2)

                                with viz_col1:
                                    st.write("#### Power Fluid Allocation")
                                    fig_pie = create_power_fluid_pie_chart(results)
                                    st.pyplot(fig_pie)

                                    st.write("#### Oil Rate by Well")
                                    fig_oil = create_oil_rate_bar_chart(results)
                                    st.pyplot(fig_oil)

                                with viz_col2:
                                    st.write("#### Pump Configurations")
                                    fig_config = create_pump_config_chart(results)
                                    st.pyplot(fig_config)

                                    st.write("#### Watercut Comparison")
                                    fig_wc = create_watercut_comparison(results)
                                    st.pyplot(fig_wc)

                                st.write("#### Efficiency Analysis")
                                col_a, col_b = st.columns(2)

                                with col_a:
                                    st.write("##### Oil vs Power Fluid")
                                    fig_eff = create_efficiency_scatter(results)
                                    st.pyplot(fig_eff)

                                with col_b:
                                    st.write("##### Marginal Oil Rates")
                                    fig_marg = create_marginal_rate_chart(results)
                                    st.pyplot(fig_marg)

                            with opt_tab4:
                                st.write("### Export Results")
                                results_df = optimizer.to_dataframe()
                                csv_data = results_df.to_csv(index=False)
                                st.download_button(
                                    label="Download Optimization Results CSV",
                                    data=csv_data,
                                    file_name="multi_well_optimization_results.csv",
                                    mime="text/csv",
                                )
                        else:
                            st.warning("Optimization did not produce viable results. Try adjusting constraints.")

                        # Clean up temp file
                        import os

                        os.unlink(temp_csv_path)

                    except Exception as e:
                        st.error(f"Error during optimization: {str(e)}")
                        st.exception(e)

                elif not uploaded_file:
                    st.info("👆 Upload a CSV file to begin multi-well optimization")

    else:
        st.info("Adjust the parameters in the sidebar and click 'Run Simulation' to generate results.")

        # Display some information about the application
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


if __name__ == "__main__":
    main()
