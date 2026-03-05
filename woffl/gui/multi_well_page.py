"""Multi-Well Optimization Page

Standalone page for multi-well jet pump optimization.
"""

import math
import os
import tempfile

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
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


def run_multi_well_optimization_page():
    """Render the multi-well optimization page"""

    st.title("🛢️ Multi-Well Jet Pump Optimization")

    st.markdown(
        """
    Optimize jet pump sizing across multiple wells to maximize field oil production 
    given a constrained power fluid supply.
    """
    )

    # Sidebar for parameters
    with st.sidebar:
        st.header("Optimization Parameters")

        # Power Fluid Settings
        st.subheader("Power Fluid Constraint")
        total_pf = st.number_input(
            "Total Available Power Fluid (bbl/day)",
            min_value=0,
            max_value=50000,
            value=10000,
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

        rho_pf = st.number_input("Power Fluid Density (lbm/ft³)", min_value=50.0, max_value=70.0, value=62.4, step=0.1)

        # Optimization Settings
        st.subheader("Optimization Settings")
        opt_method = st.selectbox(
            "Algorithm",
            ["greedy", "proportional"],
            index=0,
            help="Greedy: Iterative marginal allocation. Proportional: Productivity-based allocation.",
        )

        marginal_wc = st.number_input(
            "Marginal Watercut Threshold", min_value=0.0, max_value=1.0, value=0.94, step=0.01, format="%.2f"
        )

        # Pump Options
        st.subheader("Pump Options to Test")
        nozzle_opts = st.multiselect(
            "Nozzle Sizes",
            ["8", "9", "10", "11", "12", "13", "14", "15"],
            default=["8", "9", "10", "11", "12", "13", "14"],
            help="Select nozzle sizes to test during optimization",
        )

        throat_opts = st.multiselect(
            "Throat Ratios",
            ["X", "A", "B", "C", "D", "E"],
            default=["X", "A", "B", "C", "D"],
            help="Select throat ratios to test during optimization",
        )

    # Main content
    st.write("## Well Configuration")

    # Check for data sent from Well Test Analysis page
    if "wt_export_for_multiwell" in st.session_state:
        wt_df = st.session_state["wt_export_for_multiwell"]
        n_wells = len(wt_df)
        st.info(f"Well Test Analysis results available ({n_wells} wells). Load them or upload a CSV below.")
        if st.button("Load Well Test Results", type="primary", use_container_width=True):
            tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", newline="")
            wt_df.to_csv(tmp, index=False)
            tmp.close()
            try:
                wells = load_wells_from_csv(tmp.name)
                st.session_state["mw_wells_from_wt"] = wells
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

    has_wt_wells = "mw_wells_from_wt" in st.session_state

    if not uploaded_file and not has_wt_wells:
        st.info("👆 Upload a CSV file to begin. Click 'Download CSV Template' to get started.")

        # Show example and instructions
        with st.expander("📖 How to Use", expanded=True):
            st.markdown(
                """
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
            """
            )
        return

    # Validation check for pump options
    if not nozzle_opts or not throat_opts:
        st.warning("⚠️ Please select at least one nozzle size and one throat ratio in the sidebar.")
        return

    # Run Optimization
    if st.button("🚀 Run Multi-Well Optimization", type="primary", use_container_width=True):
        try:
            # Load wells from session state (Well Test Analysis) or uploaded CSV
            if has_wt_wells:
                wells = st.session_state["mw_wells_from_wt"]
                st.success(f"✅ Using {len(wells)} wells from Well Test Analysis")
            else:
                with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as f:
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
            pf_constraint = PowerFluidConstraint(total_rate=total_pf, pressure=pf_pressure, rho_pf=rho_pf)

            optimizer = NetworkOptimizer(
                wells=wells,
                power_fluid=pf_constraint,
                nozzle_options=nozzle_opts,
                throat_options=throat_opts,
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

            # Run optimization
            st.write("### Running Optimization")
            with st.spinner(f"Running {opt_method} optimization..."):
                results = optimize(optimizer, method=opt_method)

            if not results:
                st.warning("⚠️ Optimization did not produce viable results. Try adjusting constraints or pump options.")
                return

            st.success(f"✅ Optimization complete! Allocated pumps to {len(results)} wells")

            # Display results
            st.write("## Optimization Results")

            # Build comparison tab list dynamically
            has_jp_history = "jp_history_df" in st.session_state
            tab_labels = ["📊 Summary", "📋 Well Details", "📈 Visualizations", "💾 Export"]
            if has_jp_history:
                tab_labels.append("🔄 Current vs Optimized")

            result_tabs = st.tabs(tab_labels)
            res_tab1, res_tab2, res_tab3, res_tab4 = result_tabs[:4]

            with res_tab1:
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

            with res_tab2:
                st.write("### Well-Level Results")
                results_df = optimizer.to_dataframe()
                st.dataframe(results_df, use_container_width=True, height=400)

            with res_tab3:
                st.write("### Optimization Visualizations")

                viz_col1, viz_col2 = st.columns(2)

                with viz_col1:
                    st.write("#### Power Fluid Allocation")
                    fig_pie = create_power_fluid_pie_chart(results)
                    st.pyplot(fig_pie)
                    plt.close()

                    st.write("#### Oil Rate by Well")
                    fig_oil = create_oil_rate_bar_chart(results)
                    st.pyplot(fig_oil)
                    plt.close()

                    st.write("#### Oil vs Power Fluid Efficiency")
                    fig_eff = create_efficiency_scatter(results)
                    st.pyplot(fig_eff)
                    plt.close()

                with viz_col2:
                    st.write("#### Pump Configurations")
                    fig_config = create_pump_config_chart(results)
                    st.pyplot(fig_config)
                    plt.close()

                    st.write("#### Watercut Comparison")
                    fig_wc = create_watercut_comparison(results)
                    st.pyplot(fig_wc)
                    plt.close()

                    st.write("#### Marginal Oil Rates")
                    fig_marg = create_marginal_rate_chart(results)
                    st.pyplot(fig_marg)
                    plt.close()

            with res_tab4:
                st.write("### Export Results")
                results_df = optimizer.to_dataframe()
                csv_data = results_df.to_csv(index=False)

                st.download_button(
                    label="📥 Download Optimization Results CSV",
                    data=csv_data,
                    file_name="multi_well_optimization_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

                st.write("### Summary Statistics")
                st.write(f"- Total Wells: {len(results)}")
                st.write(f"- Total Oil: {metrics['total_oil_rate']:.1f} BOPD")
                st.write(f"- Total Power Fluid: {metrics['total_power_fluid']:.1f} BWPD")
                st.write(f"- Power Fluid Utilization: {metrics['power_fluid_utilization']:.1%}")

            # Current vs Optimized comparison tab
            if has_jp_history:
                with result_tabs[4]:
                    _render_current_vs_optimized(results)

            # Clean up temp file (only exists for CSV upload path)
            if not has_wt_wells and os.path.exists(temp_csv_path):
                os.unlink(temp_csv_path)

        except Exception as e:
            st.error(f"❌ Error during optimization: {str(e)}")
            st.exception(e)


@st.cache_data(ttl=86400, show_spinner=False)
def _cached_multi_well_tests(well_names: tuple, months_back: int = 3):
    """Cache wrapper for fetching recent tests for multiple wells."""
    from datetime import datetime

    from dateutil.relativedelta import relativedelta

    from woffl.assembly.restls_client import _denormalize_well_name, fetch_milne_well_tests

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - relativedelta(months=months_back)).strftime("%Y-%m-%d")
    db_names = [_denormalize_well_name(w) for w in well_names]
    df, _ = fetch_milne_well_tests(start_date, end_date, well_names=db_names)
    return df


def _render_current_vs_optimized(results) -> None:
    """Render the Current vs Optimized comparison table."""
    from woffl.assembly.jp_history import get_current_pump

    jp_hist = st.session_state.get("jp_history_df")
    if jp_hist is None:
        return

    st.write("### Current JP vs Optimized Solution")

    # Get optimized well names
    opt_wells = [r["well_name"] for r in results]

    # Fetch recent test data for actual oil rates
    with st.spinner("Fetching recent well tests for comparison..."):
        try:
            test_df = _cached_multi_well_tests(tuple(opt_wells), months_back=3)
        except Exception as e:
            st.warning(f"Could not fetch well tests for comparison: {e}")
            test_df = pd.DataFrame()

    # Build most recent actual oil per well
    actual_oil_map = {}
    if not test_df.empty and "WtOilVol" in test_df.columns:
        for well in opt_wells:
            well_tests = test_df[test_df["well"] == well].sort_values("WtDate", ascending=False)
            if not well_tests.empty:
                val = well_tests.iloc[0]["WtOilVol"]
                if not (isinstance(val, float) and math.isnan(val)):
                    actual_oil_map[well] = val

    # Build comparison rows
    rows = []
    for r in results:
        well = r["well_name"]
        current = get_current_pump(jp_hist, well)

        current_jp_str = "N/A"
        if current and current["nozzle_no"] and current["throat_ratio"]:
            current_jp_str = f"{current['nozzle_no']}{current['throat_ratio']}"

        opt_jp_str = f"{r['nozzle']}{r['throat']}"
        opt_oil = r.get("oil_rate", 0)
        actual = actual_oil_map.get(well)

        row = {
            "Well": well,
            "Current JP": current_jp_str,
            "Actual Oil (BOPD)": f"{actual:.0f}" if actual is not None else "N/A",
            "Optimized JP": opt_jp_str,
            "Optimized Oil (BOPD)": f"{opt_oil:.0f}",
            "Delta Oil (BOPD)": f"{opt_oil - actual:+.0f}" if actual is not None else "N/A",
        }
        rows.append(row)

    comp_df = pd.DataFrame(rows)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # Field totals
    total_actual = sum(v for v in actual_oil_map.values())
    total_optimized = sum(r.get("oil_rate", 0) for r in results)
    uplift = total_optimized - total_actual

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Current Oil", f"{total_actual:.0f} BOPD" if total_actual > 0 else "N/A")
    with col2:
        st.metric("Total Optimized Oil", f"{total_optimized:.0f} BOPD")
    with col3:
        if total_actual > 0:
            st.metric("Total Uplift", f"{uplift:+.0f} BOPD")
        else:
            st.metric("Total Uplift", "N/A")
