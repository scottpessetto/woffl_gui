"""Step 4: Results — comparison table, visualizations, exports, PF sensitivity."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from woffl.assembly.calibration import apply_calibration
from woffl.assembly.jp_history import get_current_pump
from woffl.assembly.optimization_algorithms import optimize
from woffl.gui.optimization_viz import (
    create_efficiency_scatter,
    create_ipr_comparison_pdf,
    create_marginal_rate_chart,
    create_oil_rate_bar_chart,
    create_power_fluid_pie_chart,
    create_pump_config_chart,
    create_watercut_comparison,
)


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

    # --- PF Sensitivity (below tabs) ---
    if st.session_state.get("uw_run_pf_sensitivity", False):
        st.divider()
        st.write("## Power Fluid Sensitivity")
        _render_pf_sensitivity(optimizer)

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

    # Build comparison rows
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

    # Field totals
    total_actual_oil = sum(v for v in actual_oil_map.values())
    total_actual_pf = sum(v for v in actual_pf_map.values())
    total_optimized = sum(cr.predicted_oil_rate for cr in cal_results)
    total_opt_pf = sum(r.allocated_power_fluid for r in results)
    uplift = total_optimized - total_actual_oil

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

    # IPR Comparison PDF
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
        label="Download IPR Comparison PDF",
        data=pdf_bytes,
        file_name="ipr_comparison.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


def _render_pf_sensitivity(optimizer):
    """Run PF sensitivity sweep and display results."""
    opt_method = st.session_state.get("uw_opt_method", "milp")
    original_rate = st.session_state.get("uw_total_pf", 30000)
    original_pressure = st.session_state.get("uw_pf_pressure", 3168)
    rate_min = st.session_state.get("uw_pf_rate_min", max(1000, original_rate - 6000))
    rate_max = st.session_state.get("uw_pf_rate_max", min(60000, original_rate + 6000))
    rate_step = st.session_state.get("uw_pf_rate_step", 2000)
    press_min = st.session_state.get("uw_pf_press_min", max(1000, original_pressure - 500))
    press_max = st.session_state.get("uw_pf_press_max", min(5000, original_pressure + 500))
    press_step = st.session_state.get("uw_pf_press_step", 100)

    if rate_min >= rate_max or press_min >= press_max:
        st.warning("Check PF sensitivity range settings in the sidebar.")
        return

    rate_range = list(range(int(rate_min), int(rate_max) + int(rate_step), int(rate_step)))
    press_range = list(
        range(int(press_min), int(press_max) + int(press_step), int(press_step))
    )
    total_runs = len(press_range) * len(rate_range)

    st.caption(
        f"{len(press_range)} pressure levels x {len(rate_range)} rate levels "
        f"= {total_runs} scenarios"
    )

    if st.button("Run PF Sensitivity", type="primary", key="uw_run_pf_sens"):
        rows = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        run_idx = 0

        for pressure in press_range:
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
                    m = optimizer.calculate_field_metrics()
                    rows.append(
                        {
                            "PF Pressure (psi)": pressure,
                            "PF Rate (BWPD)": rate,
                            "Total Oil (BOPD)": m["total_oil_rate"],
                            "Total Water (BWPD)": m["total_water_rate"],
                            "Field WC": m["field_watercut"],
                            "PF Utilization": m["power_fluid_utilization"],
                            "Wells Allocated": m["num_wells"],
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

        # Contour plot
        pivot = df.pivot_table(
            index="PF Pressure (psi)",
            columns="PF Rate (BWPD)",
            values="Total Oil (BOPD)",
        )
        fig_contour = go.Figure(
            data=go.Contour(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale="YlGnBu",
                colorbar=dict(title="Oil (BOPD)"),
                contours=dict(showlabels=True, labelfont=dict(size=11, color="white")),
                hovertemplate="PF Rate: %{x:.0f} BWPD<br>PF Pressure: %{y:.0f} psi<br>Oil: %{z:.0f} BOPD<extra></extra>",
            )
        )
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
            )
        )
        fig_contour.update_layout(
            title="Total Oil Production (BOPD) — Rate x Pressure",
            xaxis_title="Power Fluid Rate (BWPD)",
            yaxis_title="Power Fluid Pressure (psi)",
        )
        st.plotly_chart(fig_contour, use_container_width=True)

        # Line chart by pressure
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

        # Summary table
        st.write("### Sensitivity Results")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            label="Download Sensitivity Results CSV",
            data=df.to_csv(index=False),
            file_name="pf_sensitivity_results.csv",
            mime="text/csv",
        )


def _pressure_colors(press_range):
    """Generate evenly-spaced hex colors across a blue-to-red gradient."""
    n = len(press_range)
    if n == 1:
        return ["#2E86AB"]
    cmap = [
        (0.18, 0.53, 0.67),
        (0.15, 0.68, 0.55),
        (0.93, 0.69, 0.13),
        (0.76, 0.16, 0.18),
    ]
    positions = np.linspace(0, 1, len(cmap))
    result = []
    for i in range(n):
        t = i / (n - 1)
        for j in range(len(positions) - 1):
            if t <= positions[j + 1]:
                seg_t = (t - positions[j]) / (positions[j + 1] - positions[j])
                r = cmap[j][0] + seg_t * (cmap[j + 1][0] - cmap[j][0])
                g = cmap[j][1] + seg_t * (cmap[j + 1][1] - cmap[j][1])
                b = cmap[j][2] + seg_t * (cmap[j + 1][2] - cmap[j][2])
                result.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
                break
    return result
