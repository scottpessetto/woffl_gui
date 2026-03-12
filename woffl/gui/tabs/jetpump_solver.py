"""Tab 1: Jetpump Solver Results

Renders the single-pump solution display showing suction pressure,
oil rate, water rate, power fluid rate, and sonic status.

When JP history is uploaded and a non-Custom well is selected,
also shows a "Model vs Actual" comparison section with IPR chart
and modeled vs actual metrics.
"""

import streamlit as st
from woffl.gui.params import SimulationParams
from woffl.gui.utils import is_valid_number, run_jetpump_solver


def _render_input_summary(params: SimulationParams) -> None:
    """Render collapsible input summary showing current model parameters."""
    ipr_info = st.session_state.get("sw_ipr_info")
    label = f"Model Inputs (Nozzle {params.nozzle_no}, Throat {params.area_ratio})"

    with st.expander(label, expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Pump**")
            st.write(f"Nozzle: {params.nozzle_no}")
            st.write(f"Throat: {params.area_ratio}")
            st.write(f"ken: {params.ken}")
            st.write(f"kth: {params.kth}")
            st.write(f"kdi: {params.kdi}")
        with col2:
            st.markdown("**Well**")
            st.write(f"PF Pressure: {params.ppf_surf} psi")
            st.write(f"Surface Pressure: {params.surf_pres} psi")
            st.write(f"PF Density: {params.rho_pf} lbm/ft\u00b3")
            st.write(f"JP TVD: {params.jpump_tvd} ft")
        with col3:
            st.markdown("**Formation / IPR**")
            st.write(f"Reservoir Pressure: {params.pres} psi")
            st.write(f"Water Cut: {params.form_wc:.2f}")
            st.write(f"GOR: {params.form_gor} scf/bbl")
            st.write(f"Temperature: {params.form_temp} \u00b0F")
            st.write(f"qwf: {params.qwf} BOPD / pwf: {params.pwf} psi")
        if ipr_info:
            st.caption(f"*{ipr_info}*")


def _render_pump_identity_banner(params: SimulationParams) -> None:
    """Show banner indicating whether sidebar pump matches the installed pump."""
    jp_hist = st.session_state.get("jp_history_df")
    if jp_hist is None or params.selected_well == "Custom":
        st.info(f"Modeling: Nozzle {params.nozzle_no}, Throat {params.area_ratio}")
        return

    from woffl.assembly.jp_history import get_current_pump

    current_pump = get_current_pump(jp_hist, params.selected_well)
    if current_pump is None:
        st.info(f"Modeling: Nozzle {params.nozzle_no}, Throat {params.area_ratio}")
        return

    nozzle = current_pump["nozzle_no"]
    throat = current_pump["throat_ratio"]
    date_set = current_pump["date_set"]
    date_str = date_set.strftime("%Y-%m-%d") if date_set is not None else "N/A"

    if params.nozzle_no == nozzle and params.area_ratio == throat:
        st.success(
            f"Modeling: Nozzle {params.nozzle_no}, Throat {params.area_ratio} "
            f"\u2014 Matches current installed pump ({nozzle}{throat}, set {date_str})"
        )
    else:
        st.warning(
            f"Current installed pump is {nozzle}{throat}. "
            f"You are modeling a different configuration "
            f"(Nozzle {params.nozzle_no}, Throat {params.area_ratio})."
        )


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
    _render_input_summary(params)

    st.subheader("Jetpump Solver Results")
    _render_pump_identity_banner(params)

    # Clear stale calibration if well changed
    _cal = st.session_state.get("sw_calibration_result")
    if _cal and _cal.well_name != params.selected_well:
        st.session_state.pop("sw_calibration_result", None)

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

        # Calibration adjustment (uses result from previous run's _render_model_vs_actual)
        _cal = st.session_state.get("sw_calibration_result")
        if _cal and params.selected_well != "Custom":
            apply_cal = st.checkbox(
                f"Apply calibration factor ({_cal.calibration_factor:.2f})",
                value=False,
                key="sw_apply_calibration",
            )
            if apply_cal:
                if params.nozzle_no != _cal.current_nozzle or params.area_ratio != _cal.current_throat:
                    st.caption(
                        f"Factor derived from installed pump ({_cal.current_nozzle}{_cal.current_throat}) "
                        "\u2014 applying to a different pump is an approximation"
                    )
                cal_oil = qoil_std * _cal.calibration_factor
                cal_water = fwat_bwpd * _cal.calibration_factor
                st.markdown("**Calibrated Predictions:**")
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Calibrated Oil", f"{cal_oil:.1f} BOPD", delta=f"{cal_oil - qoil_std:+.1f}")
                with c2:
                    st.metric("Calibrated Water", f"{cal_water:.1f} BWPD", delta=f"{cal_water - fwat_bwpd:+.1f}")
    else:
        st.warning(
            "The solver could not find a solution with the current parameters. " "Try adjusting the input values."
        )

    # Model vs Actual comparison (requires JP history + non-Custom well)
    _render_model_vs_actual(params, tube, well_profile)


def _get_well_tests(well_name: str):
    """Get tests for a single well from the pre-fetched session state cache."""
    all_tests = st.session_state.get("all_well_tests_df")
    if all_tests is None or all_tests.empty:
        return None
    well_df = all_tests[all_tests["well"] == well_name].copy()
    return well_df if not well_df.empty else None


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

    st.info(
        f"Model vs Actual uses the CURRENT INSTALLED pump: "
        f"Nozzle {nozzle}, Throat {throat} "
        f"(set {current_pump['date_set'].strftime('%Y-%m-%d') if current_pump['date_set'] is not None else 'N/A'})"
    )

    # 2. Get well tests from pre-fetched cache
    test_df = _get_well_tests(params.selected_well)
    if test_df is None or len(test_df) < 2:
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
        fig = create_ipr_plotly(params.selected_well, ipr_data[params.selected_well], merged_with_rp, form_wc=params.form_wc)
        st.plotly_chart(fig, use_container_width=True)

    # 5. Run model with current JP + IPR-derived inflow
    # Use most recent test GOR by default, with sidebar override option
    recent_test = test_df.sort_values("WtDate", ascending=False).iloc[0]
    test_gor = recent_test.get("fgor", None)
    test_gor = int(test_gor) if is_valid_number(test_gor) else None

    test_whp = recent_test.get("whp", None)
    test_whp = float(test_whp) if is_valid_number(test_whp) else None

    override_gor = st.checkbox(
        "Override GOR from well test",
        value=False,
        help=f"Test GOR: {test_gor} scf/bbl. Check to use sidebar value ({params.form_gor}) instead.",
        key="mva_override_gor",
    )
    model_gor = params.form_gor if override_gor or test_gor is None else test_gor
    st.caption(f"Using GOR: **{model_gor}** scf/bbl ({'sidebar' if override_gor or test_gor is None else 'well test'})")

    model_surf_pres = test_whp if test_whp is not None else params.surf_pres
    st.caption(f"Using Surface Pressure: **{model_surf_pres:.0f}** psi ({'well test' if test_whp is not None else 'sidebar'})")

    oil_qwf = coeff_row["qwf"] * (1 - coeff_row["form_wc"])
    ipr_inflow = create_inflow(oil_qwf, coeff_row["pwf"], coeff_row["ResP"])
    ipr_res_mix = create_reservoir_mix(
        coeff_row["form_wc"], model_gor, params.form_temp, params.field_model
    )
    current_jp = create_jetpump(nozzle, throat, params.ken, params.kth, params.kdi)

    model_results = run_jetpump_solver(
        model_surf_pres,
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
    actual_pf = recent_test.get("lift_wat", None)
    actual_whp = recent_test.get("whp", None)

    st.markdown("#### Modeled vs Actual (Most Recent Test)")

    if model_results:
        _psu, _sonic, modeled_oil, _fwat, modeled_pf, _mach = model_results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Modeled Oil Rate", f"{modeled_oil:.0f} BOPD")
        with col2:
            st.metric("Actual Oil Rate", f"{actual_oil:.0f} BOPD" if is_valid_number(actual_oil) else "N/A")
        with col3:
            if is_valid_number(actual_oil):
                st.metric("Delta", f"{modeled_oil - actual_oil:+.0f} BOPD")
            else:
                st.metric("Delta", "N/A")

        if is_valid_number(actual_bhp):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Modeled BHP (suction)", f"{_psu:.0f} psi")
            with col2:
                st.metric("Actual BHP", f"{actual_bhp:.0f} psi")
            with col3:
                st.metric("Delta", f"{_psu - actual_bhp:+.0f} psi")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Modeled PF Rate", f"{modeled_pf:.0f} BWPD")
        with col2:
            st.metric("Actual PF Rate", f"{actual_pf:.0f} BWPD" if is_valid_number(actual_pf) else "N/A")
        with col3:
            if is_valid_number(actual_pf):
                st.metric("Delta", f"{modeled_pf - actual_pf:+.0f} BWPD")
            else:
                st.metric("Delta", "N/A")

        if is_valid_number(actual_pf) and abs(modeled_pf - actual_pf) > 100:
            st.warning(
                f"Modeled PF rate differs from actual by {abs(modeled_pf - actual_pf):.0f} BWPD. "
                "Check that the **Power Fluid Pressure** in the sidebar matches the actual PF pressure for this well."
            )

        if is_valid_number(actual_whp):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Test Surface Pressure", f"{actual_whp:.0f} psi")
            with col2:
                st.metric("Sidebar Surface Pressure", f"{params.surf_pres} psi")

        # Compute calibration factor
        if is_valid_number(actual_oil) and modeled_oil > 0:
            from woffl.assembly.calibration import CalibrationResult

            raw_factor = actual_oil / modeled_oil
            factor = max(0.3, min(2.0, raw_factor))
            cal_result = CalibrationResult(
                well_name=params.selected_well,
                current_nozzle=nozzle,
                current_throat=throat,
                model_oil=modeled_oil,
                actual_oil=actual_oil,
                model_pf=modeled_pf,
                actual_pf=actual_pf if is_valid_number(actual_pf) else None,
                model_bhp=_psu,
                actual_bhp=actual_bhp if is_valid_number(actual_bhp) else None,
                calibration_factor=factor,
            )
            st.session_state["sw_calibration_result"] = cal_result

            st.markdown("#### Model Calibration")
            grade = cal_result.quality_grade
            grade_color = {"good": "green", "fair": "orange", "poor": "red"}[grade]
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Calibration Factor", f"{cal_result.calibration_factor:.3f}")
            with c2:
                st.metric("Oil Error", f"{cal_result.oil_error_pct:.1f}%")
            with c3:
                st.markdown(f"**Quality:** :{grade_color}[{grade.upper()}]")
        else:
            st.session_state.pop("sw_calibration_result", None)
    else:
        st.session_state.pop("sw_calibration_result", None)
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
        "lift_wat": "PF Rate (BWPD)",
        "whp": "Surface Pres (psi)",
    }
    available = [c for c in display_cols if c in test_df.columns]
    table_df = test_df[available].copy()
    table_df = table_df.rename(columns=display_cols)
    if "Test Date" in table_df.columns:
        table_df["Test Date"] = pd.to_datetime(table_df["Test Date"]).dt.strftime("%Y-%m-%d")
    table_df = table_df.sort_values("Test Date", ascending=False).reset_index(drop=True)

    st.markdown(f"#### Well Test Data ({len(table_df)} tests)")
    st.dataframe(table_df, use_container_width=True, hide_index=True)
