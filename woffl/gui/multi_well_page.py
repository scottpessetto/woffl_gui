"""Multi-Well Optimization Page

Standalone page for multi-well jet pump optimization.
"""

import os
import tempfile

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from woffl.assembly.calibration import (
    apply_calibration,
    compute_field_calibration_summary,
    run_calibration,
)
from woffl.assembly.ipr_analyzer import (
    compute_vogel_coefficients,
    estimate_reservoir_pressure,
    export_optimization_template,
)
from woffl.assembly.network_optimizer import (
    NetworkOptimizer,
    PowerFluidConstraint,
    load_wells_from_csv,
)
from woffl.assembly.optimization_algorithms import optimize
from woffl.assembly.well_test_client import filter_wells_by_pad, get_pad_names
from woffl.gui.optimization_utils import get_template_csv_content
from woffl.gui.optimization_viz import (
    create_calibration_chart,
    create_efficiency_scatter,
    create_ipr_comparison_pdf,
    create_marginal_rate_chart,
    create_oil_rate_bar_chart,
    create_power_fluid_pie_chart,
    create_pump_config_chart,
    create_watercut_comparison,
)
from woffl.gui.params import NOZZLE_OPTIONS, THROAT_OPTIONS
from woffl.gui.utils import is_valid_number, load_well_characteristics
from woffl.gui.well_test_page import _cached_mpu_well_names, _cached_well_test_query


def run_multi_well_optimization_page():
    """Render the multi-well optimization page"""

    st.title("🛢️ Multi-Well Jet Pump Optimization")

    st.markdown("""
    Optimize jet pump sizing across multiple wells to maximize field oil production 
    given a constrained power fluid supply.
    """)

    # Sidebar for parameters
    with st.sidebar:
        st.header("Optimization Parameters")

        # Data Source
        data_source = st.radio(
            "Data Source",
            ["CSV Upload", "Databricks"],
            horizontal=True,
            help="Load wells from CSV or pull directly from Databricks",
        )

        # Power Fluid Settings
        st.subheader("Power Fluid Constraint")
        total_pf = st.number_input(
            "Total Available Power Fluid (bbl/day)",
            min_value=0,
            value=30000,
            step=500,
            help="Total power fluid capacity available for all wells",
        )

        pf_pressure = st.number_input(
            "Power Fluid Pressure (psi)",
            min_value=1000,
            max_value=5000,
            value=3168,
            step=100,
            help="Surface power fluid pressure",
        )

        rho_pf = st.number_input(
            "Power Fluid Density (lbm/ft³)",
            min_value=50.0,
            max_value=70.0,
            value=62.4,
            step=0.1,
        )

        # Optimization Settings
        st.subheader("Optimization Settings")
        opt_method = st.selectbox(
            "Algorithm",
            ["milp", "mckp"],
            index=0,
            help="MILP: Optimal via scipy linear programming. MCKP: Multi-choice knapsack via OR-Tools CP-SAT.",
        )

        marginal_wc = st.number_input(
            "Marginal Watercut Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.94,
            step=0.01,
            format="%.2f",
        )

        has_jp_history_sidebar = "jp_history_df" in st.session_state
        use_calibration = st.checkbox(
            "Apply Model Calibration",
            value=True,
            help="Scale predictions using model-vs-actual on current pump configs",
            disabled=not has_jp_history_sidebar,
        )

        # Pump Options
        st.subheader("Pump Options to Test")
        nozzle_opts = st.multiselect(
            "Nozzle Sizes",
            NOZZLE_OPTIONS,
            default=["8", "9", "10", "11", "12", "13", "14"],
            help="Select nozzle sizes to test during optimization",
        )

        throat_opts = st.multiselect(
            "Throat Ratios",
            THROAT_OPTIONS,
            default=["X", "A", "B", "C", "D"],
            help="Select throat ratios to test during optimization",
        )

        # IPR Parameters (Databricks only)
        if data_source == "Databricks":
            st.subheader("IPR Parameters")
            max_rp_schrader = st.number_input(
                "Max Res Pressure — Schrader (psi)",
                min_value=800,
                max_value=3000,
                value=1800,
                step=50,
                key="mw_max_rp_sch",
            )
            max_rp_kuparuk = st.number_input(
                "Max Res Pressure — Kuparuk (psi)",
                min_value=1500,
                max_value=5000,
                value=3000,
                step=50,
                key="mw_max_rp_kup",
            )
            resp_modifier = st.number_input(
                "Res Pres Modifier (psi)",
                min_value=0,
                max_value=500,
                value=0,
                step=10,
                key="mw_resp_mod",
                help="Offset added to estimated reservoir pressure",
            )

        # Power Fluid Sensitivity
        st.subheader("Power Fluid Sensitivity")
        run_pf_sensitivity = st.checkbox(
            "Run PF Sensitivity",
            value=False,
            help="Sweep PF rate and pressure together to map the field oil production surface",
        )
        if run_pf_sensitivity:
            st.caption("Rate Range")
            pf_rate_min = st.number_input(
                "PF Rate Min (BWPD)",
                min_value=1000,
                max_value=60000,
                value=max(1000, total_pf - 6000),
                step=1000,
            )
            pf_rate_max = st.number_input(
                "PF Rate Max (BWPD)",
                min_value=1000,
                max_value=60000,
                value=min(60000, total_pf + 6000),
                step=1000,
            )
            pf_rate_step = st.number_input(
                "PF Rate Step (BWPD)",
                min_value=500,
                max_value=10000,
                value=2000,
                step=500,
            )
            st.caption("Pressure Range")
            pf_press_min = st.number_input(
                "PF Pressure Min (psi)",
                min_value=1000,
                max_value=5000,
                value=max(1000, pf_pressure - 500),
                step=100,
            )
            pf_press_max = st.number_input(
                "PF Pressure Max (psi)",
                min_value=1000,
                max_value=5000,
                value=min(5000, pf_pressure + 500),
                step=100,
            )
            pf_press_step = st.number_input(
                "PF Pressure Step (psi)",
                min_value=50,
                max_value=500,
                value=100,
                step=50,
            )

    # Main content
    st.write("## Well Configuration")

    uploaded_file = None
    has_wt_wells = "mw_wells_from_wt" in st.session_state
    has_db_wells = "mw_wells_from_db" in st.session_state

    if data_source == "Databricks":
        _render_databricks_loader(max_rp_schrader, max_rp_kuparuk, resp_modifier)
        has_db_wells = "mw_wells_from_db" in st.session_state

        if not has_db_wells and not has_wt_wells:
            return

    else:
        # --- CSV Upload path ---
        # Check for data sent from Well Test Analysis page
        if "wt_export_for_multiwell" in st.session_state:
            wt_df = st.session_state["wt_export_for_multiwell"]
            n_wells = len(wt_df)
            st.info(
                f"Well Test Analysis results available ({n_wells} wells). Load them or upload a CSV below."
            )
            if st.button(
                "Load Well Test Results", type="primary", use_container_width=True
            ):
                tmp = tempfile.NamedTemporaryFile(
                    suffix=".csv", delete=False, mode="w", newline=""
                )
                wt_df.to_csv(tmp, index=False)
                tmp.close()
                try:
                    wells = load_wells_from_csv(tmp.name)
                    st.session_state["mw_wells_from_wt"] = wells
                    has_wt_wells = True
                    st.success(f"Loaded {len(wells)} wells from Well Test Analysis")
                except Exception as e:
                    st.error(f"Error loading well test results: {e}")
                finally:
                    os.unlink(tmp.name)
                    del st.session_state["wt_export_for_multiwell"]

        col1, col2 = st.columns([2, 1])

        with col1:
            uploaded_file = st.file_uploader(
                "Upload Well Configuration CSV",
                type=["csv"],
                help="Upload a CSV file with well configurations. Wells in jp_chars.csv will auto-populate.",
            )

        with col2:
            st.write("")  # Spacer
            st.write("")  # Spacer
            template_csv = get_template_csv_content()
            st.download_button(
                label="📥 Download CSV Template",
                data=template_csv,
                file_name="well_optimization_template.csv",
                mime="text/csv",
                help="Download a template CSV with example wells",
                use_container_width=True,
            )

        if not uploaded_file and not has_wt_wells:
            st.info(
                "👆 Upload a CSV file to begin. Click 'Download CSV Template' to get started."
            )

            with st.expander("📖 How to Use", expanded=True):
                st.markdown("""
                ### Quick Start Guide

                1. **Download the CSV template** using the button above
                2. **Edit the CSV** to include your wells:
                   - For wells in the database (jp_chars.csv): Just list the well name
                   - For custom wells: Provide all required parameters
                3. **Upload the CSV** and configure power fluid constraints
                4. **Run optimization** to get pump recommendations

                ### Example CSV
                ```csv
                Well,res_pres,form_temp,JP_TVD
                MPB-28,,,
                MPE-35,,,
                CustomWell-1,1500,75,4000
                ```

                - MPB-28 and MPE-35 auto-load from database
                - CustomWell-1 uses specified custom parameters

                ### What You Get
                - Recommended jet pump size for each well
                - Total field oil production prediction
                - Power fluid allocation across wells
                - Comprehensive visualizations and metrics
                """)
            return

    # Validation check for pump options
    if not nozzle_opts or not throat_opts:
        st.warning(
            "⚠️ Please select at least one nozzle size and one throat ratio in the sidebar."
        )
        return

    # Run Optimization
    if st.button(
        "🚀 Run Multi-Well Optimization", type="primary", use_container_width=True
    ):
        try:
            # Load wells from the appropriate source
            if has_db_wells:
                wells = st.session_state["mw_wells_from_db"]
                st.success(f"✅ Using {len(wells)} wells from Databricks")
            elif has_wt_wells:
                wells = st.session_state["mw_wells_from_wt"]
                st.success(f"✅ Using {len(wells)} wells from Well Test Analysis")
            else:
                with tempfile.NamedTemporaryFile(
                    mode="wb", suffix=".csv", delete=False
                ) as f:
                    f.write(uploaded_file.getvalue())
                    temp_csv_path = f.name

                with st.spinner("Loading well configurations..."):
                    wells = load_wells_from_csv(temp_csv_path)
                    st.success(f"✅ Loaded {len(wells)} wells from CSV")
                os.unlink(temp_csv_path)

            # Show wells loaded
            with st.expander("Wells Loaded"):
                for well in wells:
                    st.write(f"- {well.well_name} ({well.field_model})")

            # Create optimizer
            pf_constraint = PowerFluidConstraint(
                total_rate=total_pf, pressure=pf_pressure, rho_pf=rho_pf
            )

            # Inject current JP configs into nozzle/throat options so batch sims include them
            effective_nozzles = list(nozzle_opts)
            effective_throats = list(throat_opts)
            jp_hist = st.session_state.get("jp_history_df")
            opt_well_names = [w.well_name for w in wells]
            current_jp_map = {}

            if jp_hist is not None and use_calibration:
                current_jp_map = _build_current_jp_map(opt_well_names, jp_hist)
                for nozzle, throat in current_jp_map.values():
                    if nozzle not in effective_nozzles:
                        effective_nozzles.append(nozzle)
                    if throat not in effective_throats:
                        effective_throats.append(throat)

            optimizer = NetworkOptimizer(
                wells=wells,
                power_fluid=pf_constraint,
                nozzle_options=effective_nozzles,
                throat_options=effective_throats,
                marginal_watercut=marginal_wc,
            )

            # Run batch simulations
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

            st.success(f"✅ Completed batch simulations for {len(wells)} wells")

            # Calibration step (Databricks path only)
            calibration_results = None
            if use_calibration and current_jp_map and has_jp_history_sidebar:
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

                    # Render calibration summary
                    if calibration_results:
                        st.write("### Model Calibration")
                        summary = compute_field_calibration_summary(calibration_results)
                        summary["num_skipped"] = len(wells) - summary["num_calibrated"]

                        cal_col1, cal_col2, cal_col3, cal_col4 = st.columns(4)
                        with cal_col1:
                            st.metric(
                                "Calibrated Wells",
                                f"{summary['num_calibrated']}/{len(wells)}",
                            )
                        with cal_col2:
                            st.metric(
                                "Median Factor", f"{summary['median_factor']:.2f}"
                            )
                        with cal_col3:
                            st.metric("Mean Factor", f"{summary['mean_factor']:.2f}")
                        with cal_col4:
                            if summary["worst_well"]:
                                worst_err = calibration_results[
                                    summary["worst_well"]
                                ].oil_error_pct
                                st.metric(
                                    "Worst Fit",
                                    f"{summary['worst_well']} ({worst_err:.0f}%)",
                                )

                        # Calibration chart
                        fig_cal = create_calibration_chart(calibration_results)
                        st.pyplot(fig_cal)
                        plt.close()

                        # Calibration detail table
                        with st.expander("Calibration Details"):
                            cal_rows = []
                            for name, c in calibration_results.items():
                                cal_rows.append(
                                    {
                                        "Well": name,
                                        "Current JP": f"{c.current_nozzle}{c.current_throat}",
                                        "Model Oil": f"{c.model_oil:.0f}",
                                        "Actual Oil": f"{c.actual_oil:.0f}",
                                        "Factor": f"{c.calibration_factor:.2f}",
                                        "Error %": f"{c.oil_error_pct:.1f}%",
                                        "Grade": c.quality_grade,
                                    }
                                )
                            st.dataframe(
                                pd.DataFrame(cal_rows),
                                use_container_width=True,
                                hide_index=True,
                            )

            # Run optimization
            st.write("### Running Optimization")
            with st.spinner(f"Running {opt_method} optimization..."):
                results = optimize(optimizer, method=opt_method)

            if not results:
                st.warning(
                    "⚠️ Optimization did not produce viable results. Try adjusting constraints or pump options."
                )
                return

            st.success(
                f"✅ Optimization complete! Allocated pumps to {len(results)} wells"
            )

            _render_result_tabs(results, optimizer, calibration_results)

            # Power Fluid Sensitivity
            if run_pf_sensitivity:
                st.divider()
                st.write("## Power Fluid Sensitivity")
                _render_pf_sensitivity(
                    optimizer,
                    opt_method,
                    original_rate=total_pf,
                    original_pressure=pf_pressure,
                    rate_min=pf_rate_min,
                    rate_max=pf_rate_max,
                    rate_step=pf_rate_step,
                    press_min=pf_press_min,
                    press_max=pf_press_max,
                    press_step=pf_press_step,
                )

        except Exception as e:
            st.error(f"❌ Error during optimization: {str(e)}")
            st.exception(e)


def _build_actual_maps(opt_wells: list[str], test_df) -> tuple[dict, dict, dict]:
    """Build actual oil/PF/BHP maps from well test data.

    Args:
        opt_wells: List of well names to include
        test_df: DataFrame of well test data (must have 'well' column)

    Returns:
        Tuple of (actual_oil_map, actual_pf_map, actual_bhp_map)
    """
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
    """Build map of current JP configs from JP history.

    Args:
        opt_wells: List of well names
        jp_hist: JP history DataFrame

    Returns:
        Dict mapping well_name to (nozzle, throat) tuple
    """
    from woffl.assembly.jp_history import get_current_pump

    current_jp_map = {}
    for well in opt_wells:
        current = get_current_pump(jp_hist, well)
        if current and current["nozzle_no"] and current["throat_ratio"]:
            current_jp_map[well] = (current["nozzle_no"], current["throat_ratio"])
    return current_jp_map


def _render_result_tabs(results, optimizer, calibration_results=None) -> None:
    """Render optimization result tabs: summary, details, visualizations, export."""
    st.write("## Optimization Results")

    # Determine which results to display for metrics/viz/export
    if calibration_results:
        display_results = apply_calibration(results, calibration_results)
        cal_label = " (Calibrated)"
    else:
        display_results = results
        cal_label = ""

    has_jp_history = "jp_history_df" in st.session_state
    tab_labels = ["📊 Summary", "📋 Well Details", "📈 Visualizations", "💾 Export"]
    if has_jp_history:
        tab_labels.append("🔄 Current vs Optimized")

    result_tabs = st.tabs(tab_labels)
    res_tab1, res_tab2, res_tab3, res_tab4 = result_tabs[:4]

    metrics = optimizer.calculate_field_metrics(display_results)

    with res_tab1:
        st.write(f"### Field-Level Metrics{cal_label}")
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

    with res_tab2:
        st.write("### Well-Level Results")
        results_df = optimizer.to_dataframe(display_results)
        if calibration_results:
            # Add Cal Factor column
            results_df["Cal Factor"] = results_df["Well"].map(
                lambda w: (
                    calibration_results[w].calibration_factor
                    if w in calibration_results
                    else 1.0
                )
            )
        st.dataframe(results_df, use_container_width=True, height=400)

    with res_tab3:
        st.write("### Optimization Visualizations")
        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            st.write("#### Power Fluid Allocation")
            fig_pie = create_power_fluid_pie_chart(display_results)
            st.pyplot(fig_pie)
            plt.close()

            st.write("#### Oil Rate by Well")
            fig_oil = create_oil_rate_bar_chart(display_results)
            st.pyplot(fig_oil)
            plt.close()

            st.write("#### Oil vs Power Fluid Efficiency")
            fig_eff = create_efficiency_scatter(display_results)
            st.pyplot(fig_eff)
            plt.close()

        with viz_col2:
            st.write("#### Pump Configurations")
            fig_config = create_pump_config_chart(display_results)
            st.pyplot(fig_config)
            plt.close()

            st.write("#### Watercut Comparison")
            fig_wc = create_watercut_comparison(display_results)
            st.pyplot(fig_wc)
            plt.close()

            st.write("#### Marginal Oil Rates")
            fig_marg = create_marginal_rate_chart(display_results)
            st.pyplot(fig_marg)
            plt.close()

    with res_tab4:
        st.write("### Export Results")
        export_df = optimizer.to_dataframe(display_results)
        if calibration_results:
            export_df["Cal Factor"] = export_df["Well"].map(
                lambda w: (
                    calibration_results[w].calibration_factor
                    if w in calibration_results
                    else 1.0
                )
            )
        csv_data = export_df.to_csv(index=False)

        st.download_button(
            label="📥 Download Optimization Results CSV",
            data=csv_data,
            file_name="multi_well_optimization_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.write("### Summary Statistics")
        st.write(f"- Total Wells: {len(display_results)}")
        st.write(f"- Total Oil: {metrics['total_oil_rate']:.1f} BOPD{cal_label}")
        st.write(f"- Total Power Fluid: {metrics['total_power_fluid']:.1f} BWPD")
        st.write(f"- Power Fluid Utilization: {metrics['power_fluid_utilization']:.1%}")

    if has_jp_history:
        with result_tabs[4]:
            _render_current_vs_optimized(results, optimizer, calibration_results)


def _render_current_vs_optimized(results, optimizer, calibration_results=None) -> None:
    """Render the Current vs Optimized comparison table."""
    from woffl.assembly.jp_history import get_current_pump

    jp_hist = st.session_state.get("jp_history_df")
    if jp_hist is None:
        return

    st.write("### Current JP vs Optimized Solution")

    opt_wells = [r.well_name for r in results]

    # Build actual maps from session state
    all_tests = st.session_state.get("all_well_tests_df")
    actual_oil_map, actual_pf_map, actual_bhp_map = _build_actual_maps(
        opt_wells, all_tests
    )

    # Get calibrated results for display
    if calibration_results:
        cal_results = apply_calibration(results, calibration_results)
    else:
        cal_results = results

    # Build comparison rows and current JP map
    rows = []
    current_jp_map = {}
    for r, cr in zip(results, cal_results):
        well = r.well_name
        current = get_current_pump(jp_hist, well)

        current_jp_str = "N/A"
        if current and current["nozzle_no"] and current["throat_ratio"]:
            current_jp_str = f"{current['nozzle_no']}{current['throat_ratio']}"

        current_jp_map[well] = current_jp_str

        opt_jp_str = f"{r.recommended_nozzle}{r.recommended_throat}"
        cal_oil = cr.predicted_oil_rate
        actual = actual_oil_map.get(well)
        actual_pf = actual_pf_map.get(well)

        row = {
            "Well": well,
            "Current JP": current_jp_str,
            "Actual Oil (BOPD)": f"{actual:.0f}" if actual is not None else "N/A",
            "Actual PF (BWPD)": f"{actual_pf:.0f}" if actual_pf is not None else "N/A",
            "Optimized JP": opt_jp_str,
            "Model Oil (BOPD)": f"{r.predicted_oil_rate:.0f}",
        }
        if calibration_results:
            row["Cal Oil (BOPD)"] = f"{cal_oil:.0f}"
            row["Cal Factor"] = (
                f"{calibration_results[well].calibration_factor:.2f}"
                if well in calibration_results
                else "1.00"
            )
        row["Opt PF (BWPD)"] = f"{r.allocated_power_fluid:.0f}"
        row["Delta Oil (BOPD)"] = (
            f"{cal_oil - actual:+.0f}" if actual is not None else "N/A"
        )
        rows.append(row)

    comp_df = pd.DataFrame(rows)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # Field totals — use calibrated oil for delta
    total_actual_oil = sum(v for v in actual_oil_map.values())
    total_actual_pf = sum(v for v in actual_pf_map.values())
    total_optimized = sum(cr.predicted_oil_rate for cr in cal_results)
    total_opt_pf = sum(r.allocated_power_fluid for r in results)
    uplift = total_optimized - total_actual_oil

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Total Current Oil",
            f"{total_actual_oil:.0f} BOPD" if total_actual_oil > 0 else "N/A",
        )
    with col2:
        label = (
            "Total Optimized Oil (Cal)"
            if calibration_results
            else "Total Optimized Oil"
        )
        st.metric(label, f"{total_optimized:.0f} BOPD")
    with col3:
        if total_actual_oil > 0:
            st.metric("Total Uplift", f"{uplift:+.0f} BOPD")
        else:
            st.metric("Total Uplift", "N/A")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Total Actual PF",
            f"{total_actual_pf:.0f} BWPD" if total_actual_pf > 0 else "N/A",
        )
    with col2:
        st.metric("Total Optimized PF", f"{total_opt_pf:.0f} BWPD")
    with col3:
        if total_actual_pf > 0:
            st.metric("Delta PF", f"{total_opt_pf - total_actual_pf:+.0f} BWPD")
        else:
            st.metric("Delta PF", "N/A")

    # IPR Comparison PDF download
    pdf_bytes = create_ipr_comparison_pdf(
        results,
        optimizer,
        actual_oil_map,
        actual_pf_map,
        actual_bhp_map,
        current_jp_map,
        calibration=calibration_results,
    )
    st.download_button(
        label="📥 Download IPR Comparison PDF",
        data=pdf_bytes,
        file_name="ipr_comparison.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


def _render_databricks_loader(max_rp_schrader, max_rp_kuparuk, resp_modifier):
    """Fetch well tests from Databricks, run IPR analysis, and build WellConfig objects."""
    from datetime import date, timedelta

    # Fetch well names (cached 24h)
    try:
        with st.spinner("Fetching well list from Databricks..."):
            all_well_names = _cached_mpu_well_names()
    except Exception as e:
        st.error(f"Could not fetch well names from Databricks: {e}")
        return

    if not all_well_names:
        st.warning("No well names returned from Databricks.")
        return

    all_pads = get_pad_names(all_well_names)

    # Pad selection checkboxes in sidebar
    with st.sidebar:
        st.subheader("Pad Selection")
        selected_pads = []
        for pad in all_pads:
            pad_wells = filter_wells_by_pad(all_well_names, [pad])
            if st.checkbox(
                f"Pad {pad} ({len(pad_wells)})", value=False, key=f"mw_pad_{pad}"
            ):
                selected_pads.append(pad)

    if not selected_pads:
        st.info("Select at least one pad in the sidebar to begin.")
        return

    filtered_well_names = filter_wells_by_pad(all_well_names, selected_pads)

    # Date range
    col_start, col_end = st.columns(2)
    with col_start:
        default_start = date.today() - timedelta(days=730)
        start_date = st.date_input("Start Date", value=default_start, key="mw_db_start")
    with col_end:
        end_date = st.date_input("End Date", value=date.today(), key="mw_db_end")

    if start_date > end_date:
        st.error("Start date must be before end date.")
        return

    st.caption(f"{len(filtered_well_names)} wells across {len(selected_pads)} pads")

    # Load & Analyze button
    if st.button(
        "🔄 Load & Analyze Wells from Databricks",
        type="primary",
        use_container_width=True,
    ):
        try:
            # Step 1: Fetch well tests
            st.write("### Step 1: Fetching Well Tests")
            with st.spinner("Querying Databricks for well tests (cached 24h)..."):
                df, dropped_wells = _cached_well_test_query(
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d"),
                    tuple(filtered_well_names),
                )

            if df.empty:
                st.error("No well test data returned for selected pads/dates.")
                return

            st.success(f"Loaded {len(df)} well tests for {df['well'].nunique()} wells")
            if dropped_wells:
                st.warning(
                    f"{len(dropped_wells)} wells dropped (no BHP/fluid data): {', '.join(dropped_wells)}"
                )

            # Step 2: Estimate reservoir pressure
            st.write("### Step 2: Estimating Reservoir Pressure")
            with st.spinner("Estimating optimal reservoir pressure per well..."):
                jp_chars = load_well_characteristics()
                merged_with_rp = estimate_reservoir_pressure(
                    df,
                    max_pres_schrader=max_rp_schrader,
                    max_pres_kuparuk=max_rp_kuparuk,
                    jp_chars=jp_chars,
                )

            # Step 3: Compute Vogel coefficients
            st.write("### Step 3: Computing IPR Parameters")
            with st.spinner("Computing Vogel IPR parameters..."):
                vogel_coeffs = compute_vogel_coefficients(
                    merged_with_rp, resp_modifier=resp_modifier
                )

            if vogel_coeffs.empty:
                st.error("Could not compute IPR parameters for any wells.")
                return

            st.success(f"Computed IPR parameters for {len(vogel_coeffs)} wells")

            # Step 4: Convert to WellConfig via optimization template
            st.write("### Step 4: Building Well Configurations")
            template_df = export_optimization_template(vogel_coeffs)

            tmp = tempfile.NamedTemporaryFile(
                suffix=".csv", delete=False, mode="w", newline=""
            )
            template_df.to_csv(tmp, index=False)
            tmp.close()
            try:
                wells = load_wells_from_csv(tmp.name)
                st.session_state["mw_wells_from_db"] = wells
                # Also store the well test data for the Current vs Optimized tab
                st.session_state["all_well_tests_df"] = df
                st.success(f"Ready to optimize {len(wells)} wells")
            finally:
                os.unlink(tmp.name)

            # Show loaded wells
            with st.expander(f"Wells Loaded ({len(wells)})"):
                summary_df = template_df[
                    ["Well", "res_pres", "form_wc", "qwf_bopd", "pwf", "field_model"]
                ].copy()
                summary_df.columns = [
                    "Well",
                    "Res Pres (psi)",
                    "WC",
                    "Qwf (BPD)",
                    "Pwf (psi)",
                    "Field",
                ]
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Error during Databricks load: {str(e)}")
            st.exception(e)
            return

    # Show status if wells already loaded
    if "mw_wells_from_db" in st.session_state:
        wells = st.session_state["mw_wells_from_db"]
        st.success(f"{len(wells)} wells loaded from Databricks — ready to optimize")
        with st.expander("Loaded Wells"):
            for well in wells:
                st.write(f"- {well.well_name} ({well.field_model})")


def _render_pf_sensitivity(
    optimizer,
    opt_method,
    original_rate,
    original_pressure,
    rate_min,
    rate_max,
    rate_step,
    press_min,
    press_max,
    press_step,
):
    """Run optimizer across a grid of PF rate x pressure and show results.

    Batch simulations re-run once per pressure level (hydraulics change).
    Within each pressure, rate allocation is swept without re-running batch sims.
    """
    import numpy as np

    if rate_min >= rate_max:
        st.warning("PF Rate Min must be less than Max.")
        return
    if press_min >= press_max:
        st.warning("PF Pressure Min must be less than Max.")
        return

    rate_range = list(
        range(int(rate_min), int(rate_max) + int(rate_step), int(rate_step))
    )
    press_range = list(
        range(int(press_min), int(press_max) + int(press_step), int(press_step))
    )
    total_runs = len(press_range) * len(rate_range)

    st.caption(
        f"Running {len(press_range)} pressure levels x {len(rate_range)} rate levels "
        f"= {total_runs} scenarios. Batch sims re-run per pressure level."
    )

    rows = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    run_idx = 0

    for pressure in press_range:
        # Batch sims re-run once per pressure level
        optimizer.power_fluid.pressure = pressure
        optimizer.run_all_batch_simulations()

        for rate in rate_range:
            run_idx += 1
            status_text.text(
                f"{pressure:,} psi / {rate:,} BWPD  ({run_idx}/{total_runs})"
            )
            progress_bar.progress(run_idx / total_runs)

            optimizer.power_fluid.total_rate = rate
            opt_results = optimize(optimizer, method=opt_method)
            if opt_results:
                metrics = optimizer.calculate_field_metrics()
                rows.append(
                    {
                        "PF Pressure (psi)": pressure,
                        "PF Rate (BWPD)": rate,
                        "Total Oil (BOPD)": metrics["total_oil_rate"],
                        "Total Water (BWPD)": metrics["total_water_rate"],
                        "Field WC": metrics["field_watercut"],
                        "PF Utilization": metrics["power_fluid_utilization"],
                        "Wells Allocated": metrics["num_wells"],
                    }
                )

    progress_bar.empty()
    status_text.empty()

    # Restore original state
    optimizer.power_fluid.pressure = original_pressure
    optimizer.power_fluid.total_rate = original_rate
    optimizer.run_all_batch_simulations()
    optimize(optimizer, method=opt_method)

    if not rows:
        st.warning("No viable results across the sensitivity grid.")
        return

    df = pd.DataFrame(rows)

    # --- Contour / heatmap of Total Oil ---
    pivot = df.pivot_table(
        index="PF Pressure (psi)", columns="PF Rate (BWPD)", values="Total Oil (BOPD)"
    )

    fig_contour = go.Figure(
        data=go.Contour(
            z=pivot.values,
            x=pivot.columns,  # rate
            y=pivot.index,  # pressure
            colorscale="YlGnBu",
            colorbar=dict(title="Oil (BOPD)"),
            contours=dict(showlabels=True, labelfont=dict(size=11, color="white")),
            hovertemplate="PF Rate: %{x:.0f} BWPD<br>PF Pressure: %{y:.0f} psi<br>Oil: %{z:.0f} BOPD<extra></extra>",
        )
    )

    # Mark the current operating point
    fig_contour.add_trace(
        go.Scatter(
            x=[original_rate],
            y=[original_pressure],
            mode="markers+text",
            name="Current",
            marker=dict(color="red", size=14, symbol="star"),
            text=["Current"],
            textposition="top center",
            textfont=dict(color="red", size=12),
            hovertemplate="PF Rate: %{x:.0f} BWPD<br>PF Pressure: %{y:.0f} psi<extra></extra>",
        )
    )

    fig_contour.update_layout(
        title="Total Oil Production (BOPD) — Rate x Pressure",
        xaxis_title="Power Fluid Rate (BWPD)",
        yaxis_title="Power Fluid Pressure (psi)",
    )

    st.plotly_chart(fig_contour, use_container_width=True)

    # --- Line chart: one line per pressure level ---
    fig_lines = go.Figure()
    colors = _pressure_colors(press_range)

    for pressure, color in zip(press_range, colors):
        pdf = df[df["PF Pressure (psi)"] == pressure]
        if pdf.empty:
            continue
        fig_lines.add_trace(
            go.Scatter(
                x=pdf["PF Rate (BWPD)"],
                y=pdf["Total Oil (BOPD)"],
                mode="lines+markers",
                name=f"{pressure} psi",
                line=dict(color=color, width=2),
                marker=dict(size=5),
                hovertemplate="PF Rate: %{x:.0f} BWPD<br>Oil: %{y:.0f} BOPD<extra></extra>",
            )
        )

    # Current operating point
    current_row = df[
        (df["PF Rate (BWPD)"] == original_rate)
        & (df["PF Pressure (psi)"] == original_pressure)
    ]
    if not current_row.empty:
        fig_lines.add_trace(
            go.Scatter(
                x=current_row["PF Rate (BWPD)"],
                y=current_row["Total Oil (BOPD)"],
                mode="markers",
                name="Current",
                marker=dict(color="red", size=14, symbol="star"),
                hovertemplate="PF Rate: %{x:.0f} BWPD<br>Oil: %{y:.0f} BOPD<extra></extra>",
            )
        )

    fig_lines.update_layout(
        title="Total Oil vs PF Rate by Pressure Level",
        xaxis_title="Power Fluid Rate (BWPD)",
        yaxis_title="Total Oil Rate (BOPD)",
        legend_title="PF Pressure",
        hovermode="x unified",
    )

    st.plotly_chart(fig_lines, use_container_width=True)

    # --- Summary table ---
    st.write("### Sensitivity Results")
    display_df = df.copy()
    display_df["Total Oil (BOPD)"] = display_df["Total Oil (BOPD)"].apply(
        lambda x: f"{x:,.0f}"
    )
    display_df["Total Water (BWPD)"] = display_df["Total Water (BWPD)"].apply(
        lambda x: f"{x:,.0f}"
    )
    display_df["Field WC"] = display_df["Field WC"].apply(lambda x: f"{x:.1%}")
    display_df["PF Utilization"] = display_df["PF Utilization"].apply(
        lambda x: f"{x:.1%}"
    )
    display_df["Wells Allocated"] = display_df["Wells Allocated"].astype(int)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # --- CSV export ---
    csv_data = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Sensitivity Results CSV",
        data=csv_data,
        file_name="pf_sensitivity_results.csv",
        mime="text/csv",
    )


def _pressure_colors(press_range):
    """Generate evenly-spaced hex colors across a blue-to-red gradient."""
    import numpy as np

    n = len(press_range)
    if n == 1:
        return ["#2E86AB"]
    # Interpolate from blue (#2E86AB) through teal to orange to red (#C1292E)
    cmap = [
        (0.18, 0.53, 0.67),  # #2E86AB
        (0.15, 0.68, 0.55),  # teal
        (0.93, 0.69, 0.13),  # golden
        (0.76, 0.16, 0.18),  # #C1292E
    ]
    positions = np.linspace(0, 1, len(cmap))
    result = []
    for i in range(n):
        t = i / (n - 1)
        # Find segment
        for j in range(len(positions) - 1):
            if t <= positions[j + 1]:
                seg_t = (t - positions[j]) / (positions[j + 1] - positions[j])
                r = cmap[j][0] + seg_t * (cmap[j + 1][0] - cmap[j][0])
                g = cmap[j][1] + seg_t * (cmap[j + 1][1] - cmap[j][1])
                b = cmap[j][2] + seg_t * (cmap[j + 1][2] - cmap[j][2])
                result.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
                break
    return result
