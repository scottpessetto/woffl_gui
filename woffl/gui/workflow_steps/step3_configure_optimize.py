"""Step 3: Configure & Run Optimization — settings review, CSV override, run."""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from woffl.assembly.calibration import (
    compute_field_calibration_summary,
    run_calibration,
)
from woffl.assembly.network_optimizer import (
    NetworkOptimizer,
    PowerFluidConstraint,
    load_wells_from_dataframe,
)
from woffl.assembly.optimization_algorithms import optimize
from woffl.gui.optimization_viz import create_calibration_chart
from woffl.gui.utils import is_valid_number
from woffl.gui.workflow_page import _clear_downstream


def _build_actual_maps(opt_wells: list[str], test_df) -> tuple[dict, dict, dict]:
    """Build actual oil/PF/BHP maps from well test data."""
    actual_oil_map = {}
    actual_pf_map = {}
    actual_bhp_map = {}
    if test_df is None or test_df.empty:
        return actual_oil_map, actual_pf_map, actual_bhp_map

    well_tests = test_df[test_df["well"].isin(opt_wells)].copy()
    for well in opt_wells:
        wt = well_tests[well_tests["well"] == well].sort_values(
            "WtDate", ascending=False
        )
        if wt.empty:
            continue
        row_data = wt.iloc[0]
        if "WtOilVol" in wt.columns and is_valid_number(row_data["WtOilVol"]):
            actual_oil_map[well] = row_data["WtOilVol"]
        if "lift_wat" in wt.columns and is_valid_number(row_data["lift_wat"]):
            actual_pf_map[well] = row_data["lift_wat"]
        if "BHP" in wt.columns and is_valid_number(row_data["BHP"]):
            actual_bhp_map[well] = row_data["BHP"]

    return actual_oil_map, actual_pf_map, actual_bhp_map


def _build_current_jp_map(opt_wells: list[str], jp_hist) -> dict[str, tuple[str, str]]:
    """Build map of current JP configs from JP history."""
    from woffl.assembly.jp_history import get_current_pump

    current_jp_map = {}
    for well in opt_wells:
        current = get_current_pump(jp_hist, well)
        if current and current["nozzle_no"] and current["throat_ratio"]:
            current_jp_map[well] = (current["nozzle_no"], current["throat_ratio"])
    return current_jp_map


def render_step3():
    st.subheader("Step 3: Configure & Run Optimization")

    well_configs = st.session_state.get("uw_well_configs", [])
    if not well_configs:
        st.warning("No well configurations available. Go back to Step 2.")
        return

    # --- CSV override upload ---
    with st.expander("Upload Modified CSV (optional)"):
        st.caption("Override well parameters by uploading an edited optimization template.")
        override_file = st.file_uploader(
            "Upload CSV Override",
            type=["csv"],
            key="uw_csv_override",
        )
        if override_file is not None:
            try:
                override_df = pd.read_csv(override_file)
                new_configs = load_wells_from_dataframe(override_df)
                st.session_state["uw_well_configs"] = new_configs
                st.session_state["uw_template_df"] = override_df
                well_configs = new_configs
                st.success(f"Overrode with {len(new_configs)} wells from CSV")
            except Exception as e:
                st.error(f"Error parsing override CSV: {e}")

    # --- Wells summary ---
    st.write(f"**{len(well_configs)} wells** ready for optimization")
    with st.expander("Well Configurations"):
        rows = []
        for wc in well_configs:
            rows.append(
                {
                    "Well": wc.well_name,
                    "Res Pres": f"{wc.res_pres:.0f}",
                    "WC": f"{wc.form_wc:.0%}",
                    "Qwf (BLPD)": f"{wc.qwf:.0f}",
                    "Field": wc.field_model,
                    "JP TVD": f"{wc.jpump_tvd:.0f}",
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # --- Settings summary ---
    total_pf = st.session_state.get("uw_total_pf", 30000)
    pf_pressure = st.session_state.get("uw_pf_pressure", 3168)
    rho_pf = st.session_state.get("uw_rho_pf", 62.4)
    opt_method = st.session_state.get("uw_opt_method", "milp")
    marginal_wc = st.session_state.get("uw_marginal_wc", 0.94)
    use_calibration = st.session_state.get("uw_use_calibration", True)
    nozzle_opts = st.session_state.get(
        "uw_nozzle_opts", ["8", "9", "10", "11", "12", "13", "14"]
    )
    throat_opts = st.session_state.get("uw_throat_opts", ["X", "A", "B", "C", "D"])

    col1, col2, col3 = st.columns(3)
    col1.metric("PF Budget", f"{total_pf:,.0f} BWPD")
    col2.metric("PF Pressure", f"{pf_pressure:,.0f} psi")
    col3.metric("Algorithm", opt_method.upper())

    if not nozzle_opts or not throat_opts:
        st.warning("Select at least one nozzle size and one throat ratio in the sidebar.")
        return

    # --- Run optimization ---
    if st.button(
        "Run Optimization",
        type="primary",
        use_container_width=True,
        key="uw_run_optimization",
    ):
        try:
            wells = well_configs
            opt_well_names = [w.well_name for w in wells]

            # Build current JP map and inject into options
            effective_nozzles = list(nozzle_opts)
            effective_throats = list(throat_opts)
            jp_hist = st.session_state.get("jp_history_df")
            current_jp_map = {}
            has_jp_history = jp_hist is not None

            if has_jp_history and use_calibration:
                current_jp_map = _build_current_jp_map(opt_well_names, jp_hist)
                for nozzle, throat in current_jp_map.values():
                    if nozzle not in effective_nozzles:
                        effective_nozzles.append(nozzle)
                    if throat not in effective_throats:
                        effective_throats.append(throat)

            # Create optimizer
            pf_constraint = PowerFluidConstraint(
                total_rate=total_pf, pressure=pf_pressure, rho_pf=rho_pf
            )
            optimizer = NetworkOptimizer(
                wells=wells,
                power_fluid=pf_constraint,
                nozzle_options=effective_nozzles,
                throat_options=effective_throats,
                marginal_watercut=marginal_wc,
            )

            # Batch simulations
            st.write("### Running Batch Simulations")
            progress_bar = st.progress(0)
            status_text = st.empty()

            def progress_callback(current, total, well_name):
                if total > 0:
                    progress_bar.progress(current / total)
                    status_text.text(f"Processing {well_name}... ({current}/{total})")

            optimizer.run_all_batch_simulations(progress_callback)
            progress_bar.empty()
            status_text.empty()
            st.success(f"Completed batch simulations for {len(wells)} wells")

            # Calibration
            calibration_results = None
            if use_calibration and current_jp_map and has_jp_history:
                all_tests = st.session_state.get("all_well_tests_df")
                actual_oil_map, actual_pf_map, actual_bhp_map = _build_actual_maps(
                    opt_well_names, all_tests
                )

                if actual_oil_map:
                    calibration_results = run_calibration(
                        optimizer,
                        actual_oil_map,
                        actual_pf_map,
                        actual_bhp_map,
                        current_jp_map,
                    )
                    optimizer.set_calibration(calibration_results)

                    if calibration_results:
                        st.write("### Model Calibration")
                        summary = compute_field_calibration_summary(calibration_results)

                        cal_col1, cal_col2, cal_col3 = st.columns(3)
                        cal_col1.metric(
                            "Calibrated Wells",
                            f"{summary['num_calibrated']}/{len(wells)}",
                        )
                        cal_col2.metric(
                            "Median Factor", f"{summary['median_factor']:.2f}"
                        )
                        cal_col3.metric(
                            "Mean Factor", f"{summary['mean_factor']:.2f}"
                        )

                        fig_cal = create_calibration_chart(calibration_results)
                        st.pyplot(fig_cal)
                        plt.close()

            # Run optimization
            st.write("### Running Optimization")
            with st.spinner(f"Running {opt_method} optimization..."):
                results = optimize(optimizer, method=opt_method)

            if not results:
                st.warning(
                    "Optimization did not produce viable results. "
                    "Try adjusting constraints or pump options."
                )
                return

            st.success(f"Optimization complete! Allocated pumps to {len(results)} wells")

            # Store results and advance to Step 4
            _clear_downstream(4)
            st.session_state["uw_optimizer"] = optimizer
            st.session_state["uw_opt_results"] = results
            st.session_state["uw_calibration_results"] = calibration_results
            st.session_state["uw_current_jp_map"] = current_jp_map
            # Store actual maps for results page
            if use_calibration and current_jp_map and has_jp_history:
                all_tests = st.session_state.get("all_well_tests_df")
                oil_map, pf_map, bhp_map = _build_actual_maps(opt_well_names, all_tests)
                st.session_state["uw_actual_oil_map"] = oil_map
                st.session_state["uw_actual_pf_map"] = pf_map
                st.session_state["uw_actual_bhp_map"] = bhp_map

            st.session_state["uw_current_step"] = 4
            st.session_state["uw_max_step_reached"] = max(
                st.session_state.get("uw_max_step_reached", 3), 4
            )
            st.rerun()

        except Exception as e:
            st.error(f"Error during optimization: {str(e)}")
            st.exception(e)

    # Show previous results if they exist (user navigated back)
    if "uw_opt_results" in st.session_state:
        st.info("Previous optimization results available.")
        if st.button("View Results →", key="uw_step3_to_results"):
            st.session_state["uw_current_step"] = 4
            st.rerun()
