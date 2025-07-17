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

from woffl.gui.utils import (
    create_inflow,
    create_jetpump,
    create_pipes,
    create_reservoir_mix,
    create_well_profile,
    generate_choked_figures,
    generate_discharge_check,
    generate_multi_suction_graphs,
    generate_multi_throat_entry_books,
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

    # Sidebar - Input Parameters
    with st.sidebar:

        run_button = st.button("Run Simulation")

        st.sidebar.header("Parameters")
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
        field_model = st.radio("Select Field Model:", options=["Schrader", "Kuparuk"], index=0)  # default selection

        st.subheader("Jetpump Parameters")
        nozzle_options = ["8", "9", "10", "11", "12", "13", "14", "15"]
        nozzle_no = st.selectbox("Nozzle Size", nozzle_options, index=nozzle_options.index("12"))

        area_ratio_options = ["X", "A", "B", "C", "D", "E"]
        area_ratio = st.selectbox("Area Ratio (Throat Size)", area_ratio_options, index=2)

        ken = st.slider("Nozzle Loss Coefficient (ken)", 0.01, 0.10, 0.03, 0.01)
        kth = st.slider("Throat Loss Coefficient (kth)", 0.1, 0.5, 0.3, 0.1)
        kdi = st.slider("Diffuser Loss Coefficient (kdi)", 0.1, 0.5, 0.4, 0.1)

        st.subheader("Pipe Parameters")
        tubing_od = st.number_input(
            "Tubing Outer Diameter (inches)", value=4.5, min_value=2.0, max_value=9.0, step=0.1, format="%.3f"
        )
        tubing_thickness = st.number_input(
            "Tubing Wall Thickness (inches)", value=0.5, min_value=0.1, max_value=2.0, step=0.1, format="%.3f"
        )
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
        form_temp = st.number_input(
            "Formation Temperature (form_temp, °F)", value=70, min_value=32, max_value=350, step=1
        )

        st.subheader("Well Parameters")
        surf_pres = st.number_input("Surface Pressure (psi)", min_value=10, max_value=600, value=210, step=10)
        jpump_tvd = st.number_input("Jetpump TVD (feet)", min_value=2500, max_value=8000, value=4065, step=10)
        rho_pf = st.number_input("Power Fluid Density (lbm/ft³)", min_value=50.0, max_value=70.0, value=62.4, step=0.1)
        ppf_surf = st.number_input(
            "Power Fluid Surface Pressure (psi)", min_value=2000, max_value=4000, value=3168, step=10
        )

        st.subheader("Inflow Parameters")
        qwf = st.number_input("Well Flow Rate (qwf, bbl/day)", min_value=100, max_value=6000, value=750, step=10)
        pwf = st.number_input(
            "Flowing Bottom Hole Pressure (pwf, psi)", min_value=100, max_value=2500, value=500, step=10
        )
        pres = st.number_input("Reservoir Pressure (pres, psi)", min_value=400, max_value=5000, value=1700, step=10)

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
            well_profile = create_well_profile(field_model, jpump_tvd)

            # Create tabs for different visualizations
            (
                tab1,
                tab2,
                tab3,
            ) = st.tabs(["Jetpump Solution", "Batch Run", "Power Fluid Range Analysis"])

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
