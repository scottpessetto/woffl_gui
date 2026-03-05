"""Tab 1: Jetpump Solver Results

Renders the single-pump solution display showing suction pressure,
oil rate, water rate, power fluid rate, and sonic status.

When JP history is uploaded and a non-Custom well is selected,
also shows a "Model vs Actual" comparison section with IPR chart
and modeled vs actual metrics.
"""

import math

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

    # Model vs Actual comparison (requires JP history + non-Custom well)
    _render_model_vs_actual(params, tube, well_profile)


@st.cache_data(ttl=86400, show_spinner=False)
def _cached_single_well_tests(well_name: str, months_back: int = 3):
    """Cache wrapper for Databricks single-well test query."""
    from woffl.assembly.restls_client import fetch_single_well_tests

    return fetch_single_well_tests(well_name, months_back)


def _render_model_vs_actual(params: SimulationParams, tube, well_profile) -> None:
    """Render the Model vs Actual comparison section.

    Only shown when JP history is uploaded and a non-Custom well is selected.
    """
    jp_hist = st.session_state.get("jp_history_df")
    if jp_hist is None or params.selected_well == "Custom":
        return

    from woffl.assembly.ipr_analyzer import (
        compute_vogel_coefficients,
        estimate_reservoir_pressure,
        generate_ipr_curves,
    )
    from woffl.assembly.jp_history import get_current_pump
    from woffl.gui.ipr_viz import create_ipr_plotly
    from woffl.gui.utils import (
        create_inflow,
        create_jetpump,
        create_reservoir_mix,
    )

    st.divider()
    st.subheader("Model vs Actual Comparison")

    # 1. Look up current JP from history
    current_pump = get_current_pump(jp_hist, params.selected_well)
    if current_pump is None:
        st.info(f"No JP history for {params.selected_well}")
        return

    nozzle = current_pump["nozzle_no"]
    throat = current_pump["throat_ratio"]
    if not nozzle or not throat:
        st.warning(f"JP history for {params.selected_well} is missing nozzle or throat data.")
        return

    st.caption(
        f"Current JP: Nozzle **{nozzle}**, Throat **{throat}** "
        f"(set {current_pump['date_set'].strftime('%Y-%m-%d') if current_pump['date_set'] is not None else 'N/A'})"
    )

    # 2. Query recent well tests from Databricks
    with st.spinner("Fetching recent well tests..."):
        try:
            test_df = _cached_single_well_tests(params.selected_well, months_back=3)
        except Exception as e:
            st.warning(f"Could not fetch well tests: {e}")
            return

    if test_df.empty or len(test_df) < 2:
        st.info(f"Not enough recent test data for {params.selected_well} (need 2+ tests with BHP).")
        return

    # 3. Estimate reservoir pressure + compute Vogel coefficients
    try:
        merged_with_rp = estimate_reservoir_pressure(test_df)
        vogel_coeffs = compute_vogel_coefficients(merged_with_rp)
    except Exception as e:
        st.warning(f"IPR analysis failed: {e}")
        return

    well_coeffs = vogel_coeffs[vogel_coeffs["Well"] == params.selected_well]
    if well_coeffs.empty:
        st.warning("Could not compute Vogel coefficients for this well.")
        return

    coeff_row = well_coeffs.iloc[0]

    # 4. Generate IPR curves and display chart with test points
    ipr_data = generate_ipr_curves(vogel_coeffs)
    if params.selected_well in ipr_data:
        fig = create_ipr_plotly(params.selected_well, ipr_data[params.selected_well], merged_with_rp)
        st.plotly_chart(fig, use_container_width=True)

    # 5. Run model with current JP + IPR-derived inflow
    # Use most recent test GOR by default, with sidebar override option
    recent_test = test_df.sort_values("WtDate", ascending=False).iloc[0]
    test_gor = recent_test.get("fgor", None)
    if test_gor is not None and not math.isnan(test_gor):
        test_gor = int(test_gor)
    else:
        test_gor = None

    override_gor = st.checkbox(
        "Override GOR from well test",
        value=False,
        help=f"Test GOR: {test_gor} scf/bbl. Check to use sidebar value ({params.form_gor}) instead.",
        key="mva_override_gor",
    )
    model_gor = params.form_gor if override_gor or test_gor is None else test_gor
    st.caption(f"Using GOR: **{model_gor}** scf/bbl ({'sidebar' if override_gor or test_gor is None else 'well test'})")

    ipr_inflow = create_inflow(coeff_row["qwf"], coeff_row["pwf"], coeff_row["ResP"])
    ipr_res_mix = create_reservoir_mix(
        coeff_row["form_wc"], model_gor, params.form_temp, params.field_model
    )
    current_jp = create_jetpump(nozzle, throat, params.ken, params.kth, params.kdi)

    model_results = run_jetpump_solver(
        params.surf_pres,
        params.form_temp,
        params.rho_pf,
        params.ppf_surf,
        current_jp,
        tube,
        well_profile,
        ipr_inflow,
        ipr_res_mix,
    )

    # 6. Display comparison metrics
    actual_oil = recent_test.get("WtOilVol", None)
    actual_bhp = recent_test.get("BHP", None)

    st.markdown("#### Modeled vs Actual (Most Recent Test)")

    if model_results:
        _psu, _sonic, modeled_oil, _fwat, _qnz, _mach = model_results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Modeled Oil Rate", f"{modeled_oil:.0f} BOPD")
        with col2:
            if actual_oil is not None and not math.isnan(actual_oil):
                st.metric("Actual Oil Rate", f"{actual_oil:.0f} BOPD")
            else:
                st.metric("Actual Oil Rate", "N/A")
        with col3:
            if actual_oil is not None and not math.isnan(actual_oil):
                delta = modeled_oil - actual_oil
                st.metric("Delta", f"{delta:+.0f} BOPD")
            else:
                st.metric("Delta", "N/A")

        if actual_bhp is not None and not math.isnan(actual_bhp):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Modeled BHP (suction)", f"{_psu:.0f} psi")
            with col2:
                st.metric("Actual BHP", f"{actual_bhp:.0f} psi")
            with col3:
                st.metric("Delta", f"{_psu - actual_bhp:+.0f} psi")
    else:
        st.warning("Model could not solve with the current JP and IPR-derived inflow.")

    # 7. Show well test data table
    import pandas as pd

    display_cols = {
        "WtDate": "Test Date",
        "WtOilVol": "Oil (BOPD)",
        "WtWaterVol": "Water (BWPD)",
        "WtTotalFluid": "Total Fluid (BPD)",
        "BHP": "BHP (psi)",
        "fgor": "GOR (scf/bbl)",
    }
    available = [c for c in display_cols if c in test_df.columns]
    table_df = test_df[available].copy()
    table_df = table_df.rename(columns=display_cols)
    if "Test Date" in table_df.columns:
        table_df["Test Date"] = pd.to_datetime(table_df["Test Date"]).dt.strftime("%Y-%m-%d")
    table_df = table_df.sort_values("Test Date", ascending=False).reset_index(drop=True)

    st.markdown(f"#### Well Test Data ({len(table_df)} tests)")
    st.dataframe(table_df, use_container_width=True, hide_index=True)
