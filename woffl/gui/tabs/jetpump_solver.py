"""Tab 1: Jetpump Solver Results

Renders the single-pump solution display showing suction pressure,
oil rate, water rate, power fluid rate, and sonic status.
"""

import streamlit as st
from woffl.gui.params import SimulationParams
from woffl.gui.utils import run_jetpump_solver


def render_tab(params: SimulationParams, jetpump, tube, well_profile, inflow, res_mix) -> None:
    """Render the Jetpump Solver Results tab.

    Args:
        params: Simulation parameters from sidebar
        jetpump: JetPump object
        tube: Tubing Pipe object
        well_profile: WellProfile object
        inflow: InFlow object
        res_mix: ResMix object
    """
    st.subheader("Jetpump Solver Results")

    with st.spinner("Running jetpump solver..."):
        solver_results = run_jetpump_solver(
            params.surf_pres,
            params.form_temp,
            params.rho_pf,
            params.ppf_surf,
            jetpump,
            tube,
            well_profile,
            inflow,
            res_mix,
        )

    if solver_results:
        psu, sonic_status, qoil_std, fwat_bwpd, qnz_bwpd, mach_te = solver_results

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Suction Pressure", f"{psu:.1f} psig")
            st.metric("Oil Flow Rate", f"{qoil_std:.1f} BOPD")
            st.metric("Formation Water Rate", f"{fwat_bwpd:.1f} BWPD")

        with col2:
            st.metric("Power Fluid Rate", f"{qnz_bwpd:.1f} BWPD")
            st.metric("Throat Entry Mach", f"{mach_te:.3f}")
            st.metric("Sonic Flow", "Yes" if sonic_status else "No")

        if sonic_status:
            st.info("The well is operating at critical flow conditions (sonic velocity at throat entry).")
        else:
            st.success("The well is operating at stable flow conditions.")
    else:
        st.warning(
            "The solver could not find a solution with the current parameters. " "Try adjusting the input values."
        )
