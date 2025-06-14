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
        field_model = "schrader"
        if st.checkbox("Use Kuparuk Model", value=False):
            field_model = "kuparuk"

        st.subheader("Jetpump Parameters")
        nozzle_options = ["8", "9", "10", "11", "12", "13", "14", "15"]
        nozzle_no = st.selectbox("Nozzle Size", nozzle_options, index=nozzle_options.index("13"))

        area_ratio_options = ["A", "B", "C", "D", "E"]
        area_ratio = st.selectbox("Area Ratio (Throat Size)", area_ratio_options, index=0)

        ken = st.slider("Nozzle Loss Coefficient (ken)", 0.01, 0.10, 0.03, 0.01)
        kth = st.slider("Throat Loss Coefficient (kth)", 0.1, 0.5, 0.3, 0.1)
        kdi = st.slider("Diffuser Loss Coefficient (kdi)", 0.1, 0.5, 0.4, 0.1)

        st.subheader("Pipe Parameters")
        tubing_od = st.slider("Tubing Outer Diameter (inches)", 2.0, 7.0, 4.5, 0.1)
        tubing_thickness = st.slider("Tubing Wall Thickness (inches)", 0.1, 1.0, 0.5, 0.1)
        casing_od = st.slider("Casing Outer Diameter (inches)", 4.0, 10.0, 6.875, 0.125)
        casing_thickness = st.slider("Casing Wall Thickness (inches)", 0.1, 1.0, 0.5, 0.1)

        st.subheader("Formation Parameters")
        form_wc = st.slider("Water Cut (form_wc)", 0.0, 1.0, 0.894, 0.001)
        form_gor = st.slider("Gas-Oil Ratio (form_gor)", 100, 1000, 600, 50)
        form_temp = st.slider("Formation Temperature (form_temp, °F)", 80, 150, 111, 1)

        st.subheader("Well Parameters")
        surf_pres = st.slider("Surface Pressure (psi)", 100, 500, 210, 10)
        jpump_tvd = st.slider("Jetpump TVD (feet)", 3000, 5000, 4065, 50)
        rho_pf = st.slider("Power Fluid Density (lbm/ft³)", 50.0, 70.0, 62.4, 0.1)
        ppf_surf = st.slider("Power Fluid Surface Pressure (psi)", 2000, 4000, 3168, 50)

        st.subheader("Inflow Parameters")
        qwf = st.slider("Well Flow Rate (qwf, bbl/day)", 100, 500, 246, 10)
        pwf = st.slider("Flowing Bottom Hole Pressure (pwf, psi)", 800, 1500, 1049, 10)
        pres = st.slider("Reservoir Pressure (pres, psi)", 1000, 2000, 1400, 10)

        st.subheader("Analysis Options")
        psu_min = st.slider("Min Suction Pressure (psi)", 1000, 1200, 1106, 10)
        psu_max = st.slider("Max Suction Pressure (psi)", 1200, 1400, 1250, 10)
        psu_steps = st.slider("Number of Pressure Steps", 3, 10, 5, 1)

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
            tab1, tab2, tab3 = st.tabs(["Choked Figures", "Discharge Check", "Multi-Suction Analysis"])

            with tab1:
                st.subheader("Choked Figures")
                # Use matplotlib backend for the figures
                figs = generate_choked_figures(
                    form_temp, rho_pf, ppf_surf, jetpump, tube, well_profile, inflow, res_mix
                )
                if figs:  # Check if figs is not None
                    for i, fig in enumerate(figs):
                        st.pyplot(fig)
                else:
                    st.warning("No choked figures were generated.")

            with tab2:
                st.subheader("Discharge Check")
                figs = generate_discharge_check(
                    surf_pres, form_temp, rho_pf, ppf_surf, jetpump, tube, well_profile, inflow, res_mix
                )
                if figs:  # Check if figs is not None
                    for i, fig in enumerate(figs):
                        st.pyplot(fig)
                else:
                    st.warning("No discharge check figures were generated.")

            with tab3:
                st.subheader("Multi-Suction Analysis")
                psu_ray = np.linspace(psu_min, psu_max, psu_steps)
                qoil_list, book_list = generate_multi_throat_entry_books(
                    psu_ray, form_temp, ken, jetpump.ate, inflow, res_mix
                )
                figs = generate_multi_suction_graphs(qoil_list, book_list)
                if figs:  # Check if figs is not None
                    for i, fig in enumerate(figs):
                        st.pyplot(fig)
                else:
                    st.warning("No multi-suction graphs were generated.")

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
