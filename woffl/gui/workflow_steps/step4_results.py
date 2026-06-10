"""Step 4: Results — comparison table, visualizations, exports."""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from woffl.assembly.calibration import apply_calibration
from woffl.assembly.jp_history import get_current_pump
from woffl.gui.optimization_viz import (
    create_efficiency_scatter,
    create_ipr_comparison_pdf,
    create_marginal_rate_chart,
    create_oil_rate_bar_chart,
    create_power_fluid_pie_chart,
    create_pump_config_chart,
    create_watercut_comparison,
)


def _render_pad_summary(pad: str, results) -> None:
    """Pad-mode summary panel: PF used vs limit, headroom, projected marginal WC.

    Rendered above the field-wide tabs when Pad scope is active. The
    "projected pad marginal WC" is computed from the OPTIMIZED pumps
    (predicted_lift_water / (predicted_lift_water + predicted_oil_rate)),
    so the engineer can read it as "if you install these pumps, here is
    your new marginal WC".
    """
    from woffl.gui.scotts_tools.well_sort import PUMP_LIMIT_PRESETS

    pad_limit = int(
        st.session_state.get(
            f"_pad_pump_limit_{pad}", PUMP_LIMIT_PRESETS.get(pad, 0)
        )
    )

    # Compare the stream the pad pump actually handles against its limit:
    # lift water only (S/H/I) or lift + formation (M/F/E full POPS).
    from woffl.gui.scotts_tools.well_sort import POPS_PUMP_HANDLES

    pad_stream = POPS_PUMP_HANDLES.get(pad, "lift")
    if pad_stream == "total":
        used_label = "Projected Pad Water"
        used_help = (
            "Sum of lift + formation water across all optimized wells — "
            "the full stream this pad's pump must handle."
        )
        total_pf_used = float(
            sum(r.predicted_lift_water + r.predicted_formation_water for r in results)
        )
    else:
        used_label = "Projected Total PF"
        used_help = (
            "Sum of allocated PF across all optimized wells. Formation "
            "water passes through to CFP and does not count against this "
            "pad's pump."
        )
        total_pf_used = float(sum(r.allocated_power_fluid for r in results))
    headroom = pad_limit - total_pf_used

    # Projected pad marginal WC = max(PFWC) where PFWC = lift / (lift + oil)
    pfwc_values: list[tuple[float, str]] = []
    for r in results:
        denom = r.predicted_lift_water + r.predicted_oil_rate
        if denom > 0:
            pfwc_values.append((r.predicted_lift_water / denom, r.well_name))
    if pfwc_values:
        marginal_pfwc, marginal_well = max(pfwc_values, key=lambda x: x[0])
    else:
        marginal_pfwc, marginal_well = None, None

    st.markdown(f"### Pad Summary: {pad}-Pad")
    cols = st.columns(4)
    cols[0].metric(
        "Pad Pump Limit",
        f"{pad_limit:,} BWPD",
        help=(
            "Capacity on TOTAL produced water (full POPS)."
            if pad_stream == "total"
            else "Capacity on lift (power-fluid) water only."
        ),
    )
    cols[1].metric(
        used_label,
        f"{total_pf_used:,.0f} BWPD",
        help=used_help,
    )
    if headroom >= 0:
        cols[2].metric(
            "Headroom",
            f"{headroom:+,.0f} BWPD",
            delta=f"{headroom:,.0f} BWPD pump capacity remaining",
            delta_color="normal",
        )
    else:
        cols[2].metric(
            "Headroom",
            f"{headroom:+,.0f} BWPD",
            delta=f"OVER by {abs(headroom):,.0f} BWPD",
            delta_color="inverse",
        )
    if marginal_pfwc is not None:
        cols[3].metric(
            "Projected Marginal WC",
            f"{marginal_pfwc:.3f}",
            delta=f"set by {marginal_well}",
            delta_color="off",
            help=(
                "Max PFWC across the pad's optimized pumps. "
                "PFWC = LiftWater / (LiftWater + Oil)."
            ),
        )
    else:
        cols[3].metric("Projected Marginal WC", "—")
    st.divider()


def render_step4():
    st.subheader("Step 4: Results")

    optimizer = st.session_state.get("uw_optimizer")
    results = st.session_state.get("uw_opt_results")
    calibration_results = st.session_state.get("uw_calibration_results")

    if not results or not optimizer:
        st.warning("No optimization results. Go back to Step 3 and run the optimizer.")
        return

    # Determine display results
    if calibration_results:
        display_results = apply_calibration(results, calibration_results)
        cal_label = " (Calibrated)"
    else:
        display_results = results
        cal_label = ""

    # Pad-mode summary (only when Step 1 set Pad scope). Read through the
    # shadow-aware helper — the Step-1 widget keys are GC'd by the time the
    # user is here, which made this panel unreachable.
    from woffl.gui.workflow_steps.step1_select_wells import _active_pad_scope

    pad_scope = _active_pad_scope()
    if pad_scope:
        _render_pad_summary(pad_scope, display_results)

    metrics = optimizer.calculate_field_metrics(display_results)

    has_jp_history = "jp_history_df" in st.session_state
    tab_labels = ["Summary", "Well Details", "Visualizations", "Export"]
    if has_jp_history:
        tab_labels.append("Current vs Optimized")

    result_tabs = st.tabs(tab_labels)

    # --- Summary tab ---
    with result_tabs[0]:
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

    # --- Well Details tab ---
    with result_tabs[1]:
        st.write("### Well-Level Results")
        results_df = optimizer.to_dataframe(display_results)
        if calibration_results:
            results_df["Cal Factor"] = results_df["Well"].map(
                lambda w: (
                    calibration_results[w].calibration_factor
                    if w in calibration_results
                    else 1.0
                )
            )
        st.dataframe(results_df, use_container_width=True, height=400)

    # --- Visualizations tab ---
    with result_tabs[2]:
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

    # --- Export tab ---
    with result_tabs[3]:
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
            label="Download Optimization Results CSV",
            data=csv_data,
            file_name="optimization_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.write("### Summary Statistics")
        st.write(f"- Total Wells: {len(display_results)}")
        st.write(f"- Total Oil: {metrics['total_oil_rate']:.1f} BOPD{cal_label}")
        st.write(f"- Total Power Fluid: {metrics['total_power_fluid']:.1f} BWPD")
        st.write(f"- Power Fluid Utilization: {metrics['power_fluid_utilization']:.1%}")

    # --- Current vs Optimized tab ---
    if has_jp_history:
        with result_tabs[4]:
            _render_current_vs_optimized(results, optimizer, calibration_results)

    # --- Navigation ---
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Back to Step 3", key="uw_step4_back"):
            st.session_state["uw_current_step"] = 3
            st.rerun()
    with col2:
        if st.button("Start Over", key="uw_step4_restart"):
            # Clear all uw_ state
            keys_to_clear = [k for k in st.session_state if k.startswith("uw_")]
            for k in keys_to_clear:
                del st.session_state[k]
            st.rerun()


def _render_current_vs_optimized(results, optimizer, calibration_results=None):
    """Render comparison table and IPR comparison PDF."""
    jp_hist = st.session_state.get("jp_history_df")
    if jp_hist is None:
        return

    st.write("### Current JP vs Optimized Solution")

    opt_wells = [r.well_name for r in results]

    actual_oil_map = st.session_state.get("uw_actual_oil_map", {})
    actual_pf_map = st.session_state.get("uw_actual_pf_map", {})
    actual_bhp_map = st.session_state.get("uw_actual_bhp_map", {})

    if calibration_results:
        cal_results = apply_calibration(results, calibration_results)
    else:
        cal_results = results

    # Build comparison rows. Values stay numeric so column_config can apply
    # consistent formatting + tooltips on top.
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
        opt_model_oil = float(r.predicted_oil_rate)
        cal_oil = float(cr.predicted_oil_rate)
        actual = actual_oil_map.get(well)
        actual_pf = actual_pf_map.get(well)

        # CalibrationResult.model_oil is the modeled oil for the CURRENT JP
        # — exactly the right baseline to subtract from opt_model_oil to
        # show the raw modeled uplift from switching pumps. Only available
        # when calibration was applied.
        current_model_oil = None
        if calibration_results and well in calibration_results:
            current_model_oil = float(calibration_results[well].model_oil)

        row = {
            "Well": well,
            "Current JP": current_jp_str,
            "Actual Oil": float(actual) if actual is not None else None,
            "Actual PF": float(actual_pf) if actual_pf is not None else None,
            "Current Model Oil": current_model_oil,
            "Optimized JP": opt_jp_str,
            "Opt Model Oil": opt_model_oil,
            "Δ Model": (
                opt_model_oil - current_model_oil
                if current_model_oil is not None
                else None
            ),
        }
        if calibration_results:
            row["Cal Oil"] = cal_oil
            row["Cal Factor"] = (
                float(calibration_results[well].calibration_factor)
                if well in calibration_results
                else 1.0
            )
        row["Opt PF"] = float(r.allocated_power_fluid)
        # "Δ Cal vs Actual" is the OLD "Delta Oil" — kept around because it
        # answers "what's the projected lift vs today" — but renamed and
        # given a tooltip so users know it assumes the calibration factor
        # transfers across pump configs (often the source of confusion).
        row["Δ Cal vs Actual"] = (
            cal_oil - actual if actual is not None else None
        )
        rows.append(row)

    comp_df = pd.DataFrame(rows)
    st.dataframe(
        comp_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Well": st.column_config.TextColumn("Well", pinned="left"),
            "Current JP": st.column_config.TextColumn(
                "Current JP",
                help="Pump installed today (from JP install history).",
            ),
            "Actual Oil": st.column_config.NumberColumn(
                "Actual Oil (BOPD)", format="%.0f",
                help="Observed oil rate from the latest qualifying well test.",
            ),
            "Actual PF": st.column_config.NumberColumn(
                "Actual PF (BWPD)", format="%.0f",
                help="Observed PF (lift water) rate from the latest test.",
            ),
            "Current Model Oil": st.column_config.NumberColumn(
                "Current Model Oil (BOPD)", format="%.0f",
                help=(
                    "What the physics model predicts for the **current** "
                    "installed pump. Compare to Actual Oil to see the "
                    "model bias on this well."
                ),
            ),
            "Optimized JP": st.column_config.TextColumn(
                "Optimized JP",
                help="Pump recommended by the optimizer (may equal Current).",
            ),
            "Opt Model Oil": st.column_config.NumberColumn(
                "Opt Model Oil (BOPD)", format="%.0f",
                help=(
                    "Model-predicted oil rate at the optimizer's recommended pump."
                ),
            ),
            "Δ Model": st.column_config.NumberColumn(
                "Δ Model (BOPD)", format="%+.0f",
                help=(
                    "Opt Model Oil − Current Model Oil. The **raw modeled "
                    "uplift** from switching to the recommended pump, "
                    "independent of any calibration scaling. This is the "
                    "honest 'what does the model think a pump swap buys you' "
                    "number."
                ),
            ),
            "Cal Oil": st.column_config.NumberColumn(
                "Cal Oil (BOPD)", format="%.0f",
                help=(
                    "Opt Model Oil × calibration factor. **Caveat**: the "
                    "factor was derived from the current pump's model vs "
                    "actual, so applying it to a different pump assumes "
                    "the model bias is the same for both configurations. "
                    "Often a fair approximation, sometimes not."
                ),
            ),
            "Cal Factor": st.column_config.NumberColumn(
                "Cal Factor", format="%.2f",
                help="actual_oil / current_model_oil. Clamped to 0.3–2.0.",
            ),
            "Opt PF": st.column_config.NumberColumn(
                "Opt PF (BWPD)", format="%.0f",
                help="PF allocated by the optimizer to this well.",
            ),
            "Δ Cal vs Actual": st.column_config.NumberColumn(
                "Δ Cal vs Actual (BOPD)", format="%+.0f",
                help=(
                    "Cal Oil − Actual Oil. **Projected** lift over today's "
                    "rate **assuming** the calibration factor transfers "
                    "across pump configurations. See Δ Model for the "
                    "calibration-free view."
                ),
            ),
        },
    )

    # Field totals — actuals restricted to wells the optimizer actually
    # allocated. The maps cover all CONFIGURED wells, so summing them whole
    # biased "Total Uplift" low whenever the optimizer dropped a well.
    opt_set = set(opt_wells)
    total_actual_oil = sum(v for w, v in actual_oil_map.items() if w in opt_set)
    total_actual_pf = sum(v for w, v in actual_pf_map.items() if w in opt_set)
    total_optimized = sum(cr.predicted_oil_rate for cr in cal_results)
    total_opt_pf = sum(r.allocated_power_fluid for r in results)
    uplift = total_optimized - total_actual_oil
    n_unallocated = len(set(actual_oil_map) - opt_set)
    if n_unallocated:
        st.caption(
            f"ℹ️ {n_unallocated} configured well(s) received no allocation — "
            "excluded from the totals below."
        )

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Total Current Oil",
        f"{total_actual_oil:.0f} BOPD" if total_actual_oil > 0 else "N/A",
    )
    label = (
        "Total Optimized Oil (Cal)" if calibration_results else "Total Optimized Oil"
    )
    col2.metric(label, f"{total_optimized:.0f} BOPD")
    col3.metric(
        "Total Uplift",
        f"{uplift:+.0f} BOPD" if total_actual_oil > 0 else "N/A",
    )

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Total Actual PF",
        f"{total_actual_pf:.0f} BWPD" if total_actual_pf > 0 else "N/A",
    )
    col2.metric("Total Optimized PF", f"{total_opt_pf:.0f} BWPD")
    col3.metric(
        "Delta PF",
        f"{total_opt_pf - total_actual_pf:+.0f} BWPD" if total_actual_pf > 0 else "N/A",
    )

    # IPR Comparison PDF — built ON CLICK (single click = auto-download).
    # The old eager build re-rendered an N-page matplotlib PDF on every
    # rerun of this tab just to feed a download button.
    if st.button(
        "Generate IPR Comparison PDF",
        key="uw_ipr_pdf_btn",
        use_container_width=True,
        help="Builds a one-page-per-well IPR comparison and downloads it.",
    ):
        from woffl.gui.workflow_steps.step2_review_ipr import (
            _trigger_browser_download,
        )

        with st.spinner("Building IPR comparison PDF..."):
            pdf_bytes = create_ipr_comparison_pdf(
                results,
                optimizer,
                actual_oil_map,
                actual_pf_map,
                actual_bhp_map,
                current_jp_map,
                calibration=calibration_results,
            )
        _trigger_browser_download(
            pdf_bytes, "ipr_comparison.pdf", "application/pdf"
        )
