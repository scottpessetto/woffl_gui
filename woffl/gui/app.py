"""WOFFL Streamlit GUI Application

This is the main entry point for the WOFFL Streamlit GUI application.
It provides a web interface for interacting with the WOFFL package's jetpump functionality.
"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

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
    run_batch_pump,
    run_jetpump_solver,
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
    st.sidebar.header("Parameters")

    # Sidebar - Input Parameters
    with st.sidebar:

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
            index=0,
            help="'Lift' shows power fluid water, 'Total' shows power fluid + formation water",
        )

        run_button = st.button("Run Simulation")

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
            ) = st.tabs(["Jetpump Solution", "Batch Run"])

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
