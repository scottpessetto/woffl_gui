"""Tab 2: Batch Pump Analysis

Renders the batch run analysis including performance graphs, derivative plots,
data tables, and jet pump recommender results.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from woffl.assembly.batchpump import exp_model
from woffl.assembly.calibration import CalibrationResult
from woffl.geometry.jetpump import JetPump
from woffl.gui.components.dataframe_display import display_results_table
from woffl.gui.params import SimulationParams
from woffl.gui.utils import (
    build_calibration_inputs,
    highlight_recommended_pump,
    is_valid_number,
    recommend_jetpump,
    render_bhp_calibration_warning,
    render_input_summary,
    render_pf_mismatch_warning,
    run_batch_pump,
)


def _render_performance_graph(batch_pump, params: SimulationParams) -> None:
    """Render the performance graph sub-tab using interactive Plotly."""
    water = params.water_type if params.water_type is not None else "total"
    st.subheader(
        f"Jet Pump Performance ({water.capitalize()} Water)",
        help=(
            "**Graph Legend:**\n\n"
            "- **Red diamonds**: Semi-finalist jet pumps (no other pump "
            "produces more oil with less water)\n"
            "- **Blue circles**: Eliminated jet pumps (another pump "
            "produces more oil with less water)\n"
            "- **Red dashed line**: Exponential curve fit of the "
            "semi-finalist pumps\n"
            "- **Gold star**: Recommended jet pump based on marginal "
            "watercut threshold\n"
            "- **Labels**: Show nozzle size + throat ratio "
            "(e.g., \"12B\" = nozzle 12, throat B)\n"
            "- **Hover** over any point for detailed metrics"
        ),
    )

    if water == "formation":
        has_curve_fit = getattr(batch_pump, "coeff_form", None) is not None
    else:
        has_curve_fit = getattr(batch_pump, "coeff_totl", None) is not None

    try:
        df = batch_pump.df.copy()
        df = df[~df["qoil_std"].isna()]

        water_col = "form_wat" if water == "formation" else "totl_wat"
        water_label = "Formation Water" if water == "formation" else "Total Water"

        # Max x for the axis. Computed up front so the curve-fit branch and
        # the axis-range config below both have it without re-computation.
        max_water_data = float(df[water_col].max()) if not df.empty else 0.0

        fig = go.Figure()

        # Eliminated points (non-semi-finalists)
        elim = df[~df["semi"]]
        if not elim.empty:
            jp_names_elim = [n + t for n, t in zip(elim["nozzle"], elim["throat"])]
            fig.add_trace(
                go.Scatter(
                    x=elim[water_col],
                    y=elim["qoil_std"],
                    mode="markers+text",
                    name="Eliminated",
                    marker=dict(
                        size=10, color="royalblue", line=dict(width=1, color="black")
                    ),
                    text=jp_names_elim,
                    textposition="top right",
                    textfont=dict(size=10),
                    customdata=np.stack(
                        [
                            elim["nozzle"],
                            elim["throat"],
                            elim["psu_solv"],
                            elim["mach_te"],
                            elim["form_wat"],
                        ],
                        axis=-1,
                    ),
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        f"Oil: %{{y:.0f}} BOPD<br>"
                        f"{water_label}: %{{x:.0f}} BWPD<br>"
                        "Suction P: %{customdata[2]:.0f} psi<br>"
                        "Mach: %{customdata[3]:.3f}<br>"
                        "Form Water: %{customdata[4]:.0f} BWPD"
                        "<extra></extra>"
                    ),
                )
            )

        # Semi-finalist points
        semi = df[df["semi"]]
        if not semi.empty:
            jp_names_semi = [n + t for n, t in zip(semi["nozzle"], semi["throat"])]
            fig.add_trace(
                go.Scatter(
                    x=semi[water_col],
                    y=semi["qoil_std"],
                    mode="markers+text",
                    name="Semi-Finalist",
                    marker=dict(
                        size=12,
                        color="crimson",
                        symbol="diamond",
                        line=dict(width=1, color="black"),
                    ),
                    text=jp_names_semi,
                    textposition="top right",
                    textfont=dict(size=10, color="crimson"),
                    customdata=np.stack(
                        [
                            semi["nozzle"],
                            semi["throat"],
                            semi["psu_solv"],
                            semi["mach_te"],
                            semi["form_wat"],
                        ],
                        axis=-1,
                    ),
                    hovertemplate=(
                        "<b>%{text}</b> (Semi-Finalist)<br>"
                        f"Oil: %{{y:.0f}} BOPD<br>"
                        f"{water_label}: %{{x:.0f}} BWPD<br>"
                        "Suction P: %{customdata[2]:.0f} psi<br>"
                        "Mach: %{customdata[3]:.3f}<br>"
                        "Form Water: %{customdata[4]:.0f} BWPD"
                        "<extra></extra>"
                    ),
                )
            )

        # Curve fit line. We track the x where the *un-clipped* curve crosses
        # zero so the x-axis can start near the curve's origin — saves the
        # plot from wasting the left third on the (clipped) negative tail.
        x_curve_zero = 0.0  # default: keep the x-axis floored at 0
        if has_curve_fit:
            coeff = batch_pump.coeff_form if water == "formation" else batch_pump.coeff_totl
            a, b, c = coeff
            fit_water = np.linspace(0, max_water_data, 200)
            # exp_model can dip below zero for some fit coefs at low water rates;
            # clip so the rendered curve never shows a negative oil rate.
            fit_oil_raw = np.array([exp_model(w, a, b, c) for w in fit_water])
            fit_oil = np.clip(fit_oil_raw, 0.0, None)
            # Smallest x where the raw (un-clipped) curve becomes positive
            # — that's the visual origin we want at the left of the plot.
            positive_mask = fit_oil_raw > 0
            if positive_mask.any():
                x_curve_zero = float(fit_water[int(np.argmax(positive_mask))])
            fig.add_trace(
                go.Scatter(
                    x=fit_water,
                    y=fit_oil,
                    mode="lines",
                    name="Exp. Curve Fit",
                    line=dict(color="crimson", width=2, dash="dash"),
                    hovertemplate=f"Oil: %{{y:.0f}} BOPD<br>{water_label}: %{{x:.0f}} BWPD<extra></extra>",
                )
            )

        # Recommended pump star
        recommendation = None
        try:
            recommendation = recommend_jetpump(
                batch_pump, params.marginal_watercut, water
            )
            rec_water = recommendation["water_rate"]
            rec_oil = recommendation["qoil_std"]
            fig.add_trace(
                go.Scatter(
                    x=[rec_water],
                    y=[rec_oil],
                    mode="markers",
                    name="Recommended",
                    marker=dict(
                        size=20,
                        color="gold",
                        symbol="star",
                        line=dict(width=2, color="black"),
                    ),
                    hovertemplate=(
                        f"<b>Recommended: {recommendation['nozzle']}{recommendation['throat']}</b><br>"
                        f"Oil: {rec_oil:.0f} BOPD<br>"
                        f"{water_label}: {rec_water:.0f} BWPD<br>"
                        f"Marginal WC: {recommendation['marginal_ratio']:.3f}"
                        "<extra></extra>"
                    ),
                )
            )
        except Exception:
            pass

        # Formation water varies in a narrow band across pumps (it's mostly
        # set by the reservoir IPR, not the pump). Anchoring the x-axis at
        # zero squashes all the points into a vertical sliver. For
        # "formation" let Plotly auto-fit the x-axis to the data.
        #
        # For "total" we use an EXPLICIT numeric range from the curve's
        # zero crossing to a little past the max data point. Plotly's
        # `range=[lo, None]` syntax falls back to autorange when one end
        # is None, so the curve's negative tail crept back into view —
        # forcing both ends with concrete numbers is what actually clips it.
        if water == "formation":
            xaxis_cfg = dict(autorange=True, gridcolor="lightgray")
        else:
            x_axis_upper = max(max_water_data * 1.05, x_curve_zero + 1.0)
            xaxis_cfg = dict(
                range=[x_curve_zero, x_axis_upper],
                gridcolor="lightgray",
            )

        fig.update_layout(
            title=dict(
                text=f"<b>{batch_pump.wellname}</b> Jet Pump Performance",
                font=dict(size=16),
            ),
            xaxis_title=f"{water.capitalize()} Water Rate (BWPD)",
            yaxis_title="Produced Oil Rate (BOPD)",
            xaxis=xaxis_cfg,
            # rangemode="nonnegative" forces the y-axis range into [0, ∞).
            # range=[0, None] doesn't always honor the zero floor when the
            # data (or curve fit) dips negative; rangemode is the canonical
            # Plotly way to clip the visible range without picking an
            # explicit upper bound.
            yaxis=dict(rangemode="nonnegative", gridcolor="lightgray"),
            plot_bgcolor="white",
            hovermode="closest",
            height=550,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        st.plotly_chart(fig, use_container_width=True)

        # Recommended pump metrics
        if recommendation is not None:
            st.subheader("Recommended Jet Pump")
            rec_col1, rec_col2 = st.columns(2)

            with rec_col1:
                st.metric("Nozzle Size", recommendation["nozzle"])
                st.metric("Throat Ratio", recommendation["throat"])
                st.metric("Oil Rate", f"{recommendation['qoil_std']:.1f} BOPD")

            with rec_col2:
                st.metric(water_label, f"{recommendation['water_rate']:.1f} BWPD")
                st.metric(
                    "Marginal Watercut", f"{recommendation['marginal_ratio']:.3f}"
                )

                if recommendation["recommendation_type"] == "best_available":
                    st.warning(
                        "Note: No jet pump meets the specified marginal watercut threshold. "
                        "This is the best available option."
                    )

    except Exception as e:
        st.error(f"Error generating performance plot: {str(e)}")
        st.info(
            "Could not generate the performance plot. Try selecting different nozzle sizes and throat ratios."
        )


def _render_data_table(batch_pump, params: SimulationParams) -> None:
    """Render the data table sub-tab."""
    water = params.water_type if params.water_type is not None else "lift"

    st.subheader("Jet Pump Performance Data")

    # Filter to show only successful runs
    df_display = batch_pump.df.copy()
    df_display = df_display[~df_display["qoil_std"].isna()]

    display_results_table(
        df=df_display,
        columns=[
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
        ],
        rename_map={
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
        },
        numeric_cols=[
            "Oil Rate (BOPD)",
            "Formation Water (BWPD)",
            "Lift Water (BWPD)",
            "Total Water (BWPD)",
            "Suction Pressure (psig)",
            "Throat Entry Mach",
        ],
        download_filename="jetpump_batch_results.csv",
    )

    # Separator
    st.markdown("---")

    # Jet Pump Recommender Results
    _render_recommender_table(batch_pump, params, water)


def _render_recommender_table(batch_pump, params: SimulationParams, water: str) -> None:
    """Render the recommender results table."""
    st.subheader("Jet Pump Recommender Results")

    semi_df = batch_pump.df[batch_pump.df["semi"]].copy()

    if semi_df.empty:
        st.warning(
            "No semi-finalist jet pumps found. Cannot display recommender results."
        )
        return

    semi_df = semi_df.sort_values(by="qoil_std", ascending=True)

    water_col = "form_wat" if water == "formation" else "totl_wat"
    marg_col = "mofwr" if water == "formation" else "motwr"

    # Convert marginal oil-water ratios to marginal watercuts
    semi_df["marginal_watercut"] = 1 / (1 + semi_df[marg_col])

    # Build display columns
    water_label = (
        "Formation Water (BWPD)" if water == "formation" else "Total Water (BWPD)"
    )
    ratio_label = (
        "Marginal Oil/Formation Water Ratio"
        if water == "formation"
        else "Marginal Oil/Total Water Ratio"
    )

    recommender_df = semi_df[
        ["nozzle", "throat", "qoil_std", water_col, marg_col, "marginal_watercut"]
    ].copy()

    recommender_df = recommender_df.rename(
        columns={
            "nozzle": "Nozzle",
            "throat": "Throat",
            "qoil_std": "Oil Rate (BOPD)",
            water_col: water_label,
            marg_col: ratio_label,
            "marginal_watercut": "Marginal Watercut",
        }
    )

    numeric_cols = ["Oil Rate (BOPD)", water_label, ratio_label, "Marginal Watercut"]
    recommender_df[numeric_cols] = recommender_df[numeric_cols].round(3)

    # Try to highlight the recommended pump
    try:
        recommendation = recommend_jetpump(batch_pump, params.marginal_watercut, water)

        recommender_df["Recommended"] = False
        recommended_mask = (recommender_df["Nozzle"] == recommendation["nozzle"]) & (
            recommender_df["Throat"] == recommendation["throat"]
        )
        recommender_df.loc[recommended_mask, "Recommended"] = True

        st.dataframe(recommender_df, use_container_width=True)

        st.info(
            f"The recommended pump is highlighted: Nozzle {recommendation['nozzle']}, "
            f"Throat {recommendation['throat']} with a marginal watercut of "
            f"{recommendation['marginal_ratio']:.3f}"
        )

        if recommendation["recommendation_type"] == "best_available":
            st.warning(
                "Note: No jet pump meets the specified marginal watercut threshold. "
                "This is the best available option."
            )
    except Exception as e:
        st.dataframe(recommender_df, use_container_width=True)
        st.warning(f"Could not determine recommended jet pump: {str(e)}")

    # Download button
    csv_recommender = recommender_df.to_csv(index=False)
    st.download_button(
        label="Download Recommender Results CSV",
        data=csv_recommender,
        file_name="jetpump_recommender_results.csv",
        mime="text/csv",
        key="dl_jetpump_recommender_results.csv",
    )


def _get_well_tests(well_name: str):
    """Get tests for a single well from the pre-fetched session state cache."""
    all_tests = st.session_state.get("all_well_tests_df")
    if all_tests is None or all_tests.empty:
        return None
    well_df = all_tests[all_tests["well"] == well_name].copy()
    return well_df if not well_df.empty else None


def _compute_batch_calibration(
    batch_pump, params: SimulationParams
) -> CalibrationResult | None:
    """Compute calibration factor by comparing the installed pump's batch result to actual well test.

    Returns CalibrationResult or None if data is insufficient.
    """
    jp_hist = st.session_state.get("jp_history_df")
    if jp_hist is None or params.selected_well == "Custom":
        return None

    from woffl.assembly.jp_history import get_current_pump

    current_pump = get_current_pump(jp_hist, params.selected_well)
    if current_pump is None:
        return None

    nozzle = current_pump["nozzle_no"]
    throat = current_pump["throat_ratio"]
    if not nozzle or not throat:
        return None

    # Find the installed pump in batch results
    df = batch_pump.df
    match = df[(df["nozzle"] == nozzle) & (df["throat"] == throat)]
    if match.empty or pd.isna(match.iloc[0]["qoil_std"]):
        return None

    model_row = match.iloc[0]
    model_oil = model_row["qoil_std"]
    model_pf = model_row["lift_wat"]
    model_bhp = model_row["psu_solv"]

    # Get actual data from well tests
    test_df = _get_well_tests(params.selected_well)
    if test_df is None or test_df.empty:
        return None

    recent_test = test_df.sort_values("WtDate", ascending=False).iloc[0]
    actual_oil = recent_test.get("WtOilVol", None)
    if not is_valid_number(actual_oil) or actual_oil <= 0:
        return None

    actual_pf = recent_test.get("lift_wat", None)
    actual_bhp = recent_test.get("BHP", None)

    raw_factor = actual_oil / model_oil
    factor = max(0.3, min(2.0, raw_factor))

    return CalibrationResult(
        well_name=params.selected_well,
        current_nozzle=nozzle,
        current_throat=throat,
        model_oil=model_oil,
        actual_oil=actual_oil,
        model_pf=model_pf,
        actual_pf=actual_pf if is_valid_number(actual_pf) else None,
        model_bhp=model_bhp,
        actual_bhp=actual_bhp if is_valid_number(actual_bhp) else None,
        calibration_factor=factor,
    )


def _render_batch_hero_strip(
    batch_pump, params: SimulationParams, wellbore, well_profile
) -> bool:
    """Hero metric strip — same shape as the Solver view.

    Renders the installed pump's batch row alongside the most recent well
    test, with vs-actual deltas on Oil, Power Fluid, and Suction Pressure.
    Doubles as the discoverability hook for the friction-coef calibration
    in the Solver view: when the deltas are large, the hint underneath
    points the user there.

    Side effect: stores the CalibrationResult in session_state so the
    "Apply calibration factor" toggle below the hero can use it. (We
    deliberately compute calibration here even when the user can't see
    the full breakdown, so the toggle behaves the same as before.)

    Returns True when PF rate mismatch is large enough that the rate-scalar
    calibration would encode a PF-pressure error rather than a real model
    correction — the caller disables the toggle in that case.
    """
    cal = _compute_batch_calibration(batch_pump, params)
    if cal is None:
        # No JP history / no actuals — show a minimal hero from the sidebar pump
        _render_batch_hero_no_actuals(batch_pump, params)
        return False

    st.session_state["batch_calibration_result"] = cal

    from woffl.assembly.jp_history import get_current_pump

    jp_hist = st.session_state.get("jp_history_df")
    current_pump = get_current_pump(jp_hist, params.selected_well)
    date_str = (
        current_pump["date_set"].strftime("%Y-%m-%d")
        if current_pump["date_set"] is not None
        else "N/A"
    )

    # Pull formation water for the installed pump from the batch DF
    # (CalibrationResult only carries oil/pf/bhp).
    match = batch_pump.df[
        (batch_pump.df["nozzle"] == cal.current_nozzle)
        & (batch_pump.df["throat"] == cal.current_throat)
    ]
    model_form_wat = (
        float(match.iloc[0]["form_wat"]) if not match.empty else None
    )

    st.caption(
        f"Showing the **current installed pump** "
        f"({cal.current_nozzle}{cal.current_throat}, set {date_str}). "
        f"Deltas compare modeled values to the most recent well test for "
        f"**{params.selected_well}**."
    )

    def _delta(modeled: float, actual: float | None, suffix: str) -> str | None:
        if actual is None:
            return None
        return f"{modeled - actual:+,.0f} {suffix}"

    h1, h2, h3, h4 = st.columns(4)
    with h1:
        d = _delta(cal.model_oil, cal.actual_oil, "vs actual")
        st.metric(
            "Oil Rate", f"{cal.model_oil:,.0f} BOPD",
            delta=d, delta_color="off" if d is None else "normal",
        )
    with h2:
        st.metric(
            "Formation Water",
            f"{model_form_wat:,.0f} BWPD" if model_form_wat is not None else "—",
        )
    with h3:
        d = _delta(cal.model_pf, cal.actual_pf, "vs actual")
        st.metric(
            "Power Fluid", f"{cal.model_pf:,.0f} BWPD",
            delta=d, delta_color="off" if d is None else "normal",
        )
    with h4:
        d = _delta(cal.model_bhp, cal.actual_bhp, "vs actual")
        st.metric(
            "Suction Pressure", f"{cal.model_bhp:,.0f} psig",
            delta=d, delta_color="off" if d is None else "normal",
        )

    # PF mismatch is the foundational check. If the sidebar PF pressure
    # doesn't match operating conditions, both the rate-scalar calibration
    # (here) and the BHP friction-cal (Solver) will encode the pressure
    # error rather than capturing a real model correction. The warning still
    # gates the calibration toggle below; the PF quickfix lives on the
    # Solver tab where users iterate on PF pressure tuning.
    cal_inputs = build_calibration_inputs(params, wellbore, well_profile)
    test_date_str = cal_inputs["test_date_str"] if cal_inputs else None
    _pf_warning_shown, pf_blocked = render_pf_mismatch_warning(
        cal.model_pf,
        cal.actual_pf,
        params.ppf_surf,
        test_date_str=test_date_str,
        well_name=params.selected_well,
    )

    # BHP red flag (only meaningful once PF is right). Falls back to the
    # softer one-line summary when the BHP match is acceptable.
    if not pf_blocked:
        bhp_warned = render_bhp_calibration_warning(cal.model_bhp, cal.actual_bhp)
        if not bhp_warned:
            grade = cal.quality_grade
            grade_color = {"good": "green", "fair": "orange", "poor": "red"}[grade]
            st.markdown(
                f"Oil error: **{cal.oil_error_pct:.1f}%** · "
                f"Quality: :{grade_color}[**{grade.upper()}**]  ·  "
                "Switch to the **Solver** view to fit ken/kth/kdi via "
                "*Friction-Coef Calibration* and drive the BHP delta toward zero."
            )

    return pf_blocked


_MARG_WC_QUICKFIX_KEY = "_batch_marg_wc_box"


def _on_batch_marg_wc_change() -> None:
    """on_change callback for the inline Marginal WC quickfix.

    Same widget-key dance as ``_on_pf_quickfix_change`` in utils.py — write
    the logical sidebar key (``marginal_watercut``) and pop the sidebar
    widget key (``marginal_watercut_input``) so the sidebar's _number_input
    helper re-initializes from the logical key on the next render. Avoids
    Streamlit's "cannot set widget state after render" error.
    """
    try:
        new_val = float(st.session_state[_MARG_WC_QUICKFIX_KEY])
    except (KeyError, ValueError, TypeError):
        return
    st.session_state["marginal_watercut"] = new_val
    st.session_state.pop("marginal_watercut_input", None)


def _do_batch_marg_import() -> None:
    """Compute the field-wide marginal WC and apply it to the sidebar.

    Single-well batch can only use a field-wide marginal — per-POPS-pad
    marginals depend on all the pad's wells together, so they don't drive
    a single-well pump choice meaningfully. Use the Pad Optimization
    workflow (Optimization Workflow → Scope: Pad) for that.
    """
    from woffl.gui.scotts_tools.well_sort import compute_field_marginal_wc

    try:
        threshold = float(st.session_state.get("marg_wc_threshold_pct", 2.0))
        result = compute_field_marginal_wc(threshold_pct=threshold)
        scope_descr = f"Field-wide ({threshold:.1f}% threshold)"
    except Exception as e:
        st.session_state["_batch_marg_import_err"] = (
            f"Could not compute marginal WC: {e}"
        )
        return

    if result is None:
        st.session_state["_batch_marg_import_err"] = (
            "Marginal WC unavailable — no online non-POPs wells with "
            "valid TotalWC / TotalWater. Check the Well Sort tab."
        )
        return

    new_wc = result["marginal_wc"]
    st.session_state["marginal_watercut"] = new_wc
    st.session_state.pop("marginal_watercut_input", None)
    st.session_state.pop(_MARG_WC_QUICKFIX_KEY, None)
    st.session_state["_batch_marg_import_msg"] = (
        f"Imported {new_wc:.3f} marginal WC — {scope_descr}, "
        f"set by {result['well']}."
    )


def _render_marg_wc_quickfix(params: SimulationParams) -> None:
    """Inline Marginal Watercut quick-entry rendered above the performance graph.

    Mirrors the PF quickfix pattern (utils.render_pf_quickfix_widget) so users
    can tune the recommendation threshold without scrolling back to the sidebar.
    The recommended-pump gold star and the recommender table both pick up the
    new value on the next rerun.

    Field-marginal-WC convention: this is the watercut the *worst well online*
    is producing at — pumps whose marginal WC sits below this still pay for
    their water handling, anything above stops being economic. POPS pads with
    headroom on water handling can be set to 1.00 to lift everything.
    """
    # Surface a success message from the previous-run Import click (the
    # click triggers a rerun, so st.success has to live across renders).
    import_msg = st.session_state.pop("_batch_marg_import_msg", None)
    if import_msg:
        st.success(import_msg)
    import_err = st.session_state.pop("_batch_marg_import_err", None)
    if import_err:
        st.warning(import_err)

    if st.session_state.get(_MARG_WC_QUICKFIX_KEY) != float(params.marginal_watercut):
        st.session_state[_MARG_WC_QUICKFIX_KEY] = float(params.marginal_watercut)

    col_input, col_import, col_water, col_status = st.columns([2, 2, 2, 2.5])
    with col_input:
        st.number_input(
            "Field Marginal Watercut",
            min_value=0.00,
            max_value=1.00,
            step=0.01,
            format="%.2f",
            key=_MARG_WC_QUICKFIX_KEY,
            on_change=_on_batch_marg_wc_change,
            help=(
                "Worst well online — for POPS pads put to 100% if room on "
                "pump for water handling. Updates the sidebar Marginal "
                "Watercut; the recommended pump (gold star) refreshes "
                "immediately."
            ),
        )
    with col_import:
        # Single-well batch can only meaningfully consume a field-wide
        # marginal WC. Per-POPS-pad marginals are built by all the pad's
        # wells together — using one to drive a single-well pump pick is
        # circular (changing this well's pump changes its contribution to
        # the marginal). For POPS-pad pump selection, the user goes to the
        # Pad Optimization workflow instead. Databricks fetchers behind
        # the field-wide calc are pre-warmed at app load by
        # _prefetch_well_sort_data, so the click hits cache.
        st.caption(" ")  # vertical alignment with the number_input above
        if st.button(
            "📊 Import Calc'd Marginal WC",
            key="batch_marg_import_btn",
            use_container_width=True,
            help=(
                "Pulls today's **field-wide** marginal WC from the Well Sort "
                "→ Marginal WC tab (cumulative-water threshold) and writes "
                "it to the sidebar. For per-POPS-pad pump selection, use "
                "**Optimization Workflow → Scope: Pad** — single-well batch "
                "can't see pad-wide tradeoffs."
            ),
        ):
            _do_batch_marg_import()
            st.rerun()
    with col_water:
        # Water-type selector lives here (not the sidebar) so the graph
        # axis + recommender ratio toggle right next to the threshold
        # that drives the recommendation.
        st.radio(
            "Water Type",
            options=["total", "formation"],
            index=0 if params.water_type != "formation" else 1,
            key="water_type",
            horizontal=True,
            help=(
                "**Total** = formation + power-fluid water (surface handling). "
                "**Formation** = reservoir-side water only. Drives the "
                "performance graph axis and recommender marginal ratio."
            ),
        )
    with col_status:
        st.caption(" ")
        st.caption(
            f"Recommendation threshold: pumps with marginal WC ≤ "
            f"**{float(params.marginal_watercut):.2f}** qualify."
        )


def _render_batch_hero_no_actuals(batch_pump, params: SimulationParams) -> None:
    """Hero strip variant for Custom mode / wells without test data.

    Shows the sidebar-selected pump's batch row with no deltas.
    """
    sb_nozzle = params.nozzle_no
    sb_throat = params.area_ratio
    match = batch_pump.df[
        (batch_pump.df["nozzle"] == sb_nozzle)
        & (batch_pump.df["throat"] == sb_throat)
    ]
    if match.empty or pd.isna(match.iloc[0].get("qoil_std")):
        return  # Nothing to show — sidebar pump wasn't in the batch grid
    row = match.iloc[0]

    st.caption(
        f"Showing the **sidebar pump** ({sb_nozzle}{sb_throat}). "
        "No well-test actuals available for vs-actual deltas."
    )
    h1, h2, h3, h4 = st.columns(4)
    with h1:
        st.metric("Oil Rate", f"{row['qoil_std']:,.0f} BOPD")
    with h2:
        st.metric("Formation Water", f"{row['form_wat']:,.0f} BWPD")
    with h3:
        st.metric("Power Fluid", f"{row['lift_wat']:,.0f} BWPD")
    with h4:
        st.metric("Suction Pressure", f"{row['psu_solv']:,.0f} psig")


def _render_nelder_mead_section(batch_pump, params: SimulationParams) -> None:
    """Render the Nelder-Mead continuous optimization section.

    Uses the best semi-finalist from the batch run as a seed for Nelder-Mead
    optimization to find the continuous optimal nozzle/throat diameters, then
    snaps to the nearest catalog pump.
    """
    semi_df = (
        batch_pump.df[batch_pump.df["semi"]].copy()
        if "semi" in batch_pump.df.columns
        else pd.DataFrame()
    )
    if semi_df.empty:
        return

    with st.expander("Nelder-Mead Continuous Optimization", expanded=False):
        st.markdown(
            "Use Nelder-Mead optimization to find the continuous optimal nozzle and throat "
            "diameters, then snap to the nearest catalog pump. The **lift cost** penalizes "
            "power fluid usage (bbl oil / bbl lift water). Higher values favor smaller pumps."
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            marg_wc_input = st.number_input(
                "Marginal Watercut",
                min_value=0.01,
                max_value=0.99,
                value=0.94,
                step=0.01,
                format="%.2f",
                key="nm_marginal_wc",
                help="Enter a marginal watercut to auto-calculate lift cost. OWR = (1 - WC) / WC.",
            )
            # Convert: WOR = wc / (1 - wc), OWR = 1/WOR = (1 - wc) / wc
            calculated_lift_cost = (1 - marg_wc_input) / marg_wc_input
            st.caption(
                f"WOR = {marg_wc_input / (1 - marg_wc_input):.2f} | "
                f"OWR (lift cost) = {calculated_lift_cost:.4f}"
            )
        with col2:
            lift_cost = st.slider(
                "Lift Cost (bbl oil / bbl lift water)",
                min_value=0.00,
                max_value=0.20,
                value=round(calculated_lift_cost, 2),
                step=0.01,
                format="%.2f",
                key="nm_lift_cost",
                help="Penalty for lift water usage. Auto-populated from marginal watercut, or override manually.",
            )
        with col3:
            # Let user pick the seed pump from semi-finalists
            semi_sorted = semi_df.sort_values("qoil_std", ascending=False)
            seed_options = [
                f"{row['nozzle']}{row['throat']} ({row['qoil_std']:.0f} BOPD)"
                for _, row in semi_sorted.iterrows()
            ]
            seed_choice = st.selectbox(
                "Seed Pump",
                options=seed_options,
                index=0,
                key="nm_seed_pump",
                help="Semi-finalist pump to seed the Nelder-Mead optimizer. Defaults to highest oil rate.",
            )
            seed_idx = seed_options.index(seed_choice)
            selected_semi = semi_sorted.iloc[seed_idx]

        run_nm = st.button("Run Nelder-Mead Optimization", key="run_nelder_mead")

        if run_nm:
            seed_jp = JetPump(
                nozzle_no=selected_semi["nozzle"],
                area_ratio=selected_semi["throat"],
                ken=params.ken,
                kth=params.kth,
                kdi=params.kdi,
            )

            with st.spinner("Running Nelder-Mead optimization..."):
                try:
                    result_df = batch_pump.search_run(seed_jp, lift_cost=lift_cost)
                except Exception as e:
                    st.error(f"Nelder-Mead optimization failed: {e}")
                    return

            if result_df is None or result_df.empty:
                st.warning("Optimization did not produce a result.")
                return

            row = result_df.iloc[0]
            st.markdown("#### Optimization Result")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**Catalog Pump (Snapped)**")
                st.metric("Pump", f"{row['nozzle']}{row['throat']}")
                if not pd.isna(row.get("qoil_std")):
                    st.metric("Oil Rate", f"{row['qoil_std']:.1f} BOPD")
                else:
                    st.metric("Oil Rate", "N/A (solver failed)")
            with col2:
                st.markdown("**Continuous Optimum**")
                st.metric("Nozzle Dia", f"{row['dnz_opt']:.4f} in")
                st.metric("Throat Dia", f"{row['dth_opt']:.4f} in")
            with col3:
                st.markdown("**Details**")
                if not pd.isna(row.get("lift_wat")):
                    st.metric("Lift Water", f"{row['lift_wat']:.0f} BWPD")
                if not pd.isna(row.get("psu_solv")):
                    st.metric("Suction P", f"{row['psu_solv']:.0f} psi")

            # Compare to seed pump
            if not pd.isna(row.get("qoil_std")):
                delta = row["qoil_std"] - selected_semi["qoil_std"]
                if abs(delta) > 0.5:
                    st.caption(
                        f"Nelder-Mead catalog pump produces **{delta:+.1f} BOPD** vs seed pump "
                        f"({selected_semi['nozzle']}{selected_semi['throat']} at {selected_semi['qoil_std']:.0f} BOPD)."
                    )
                else:
                    st.caption(
                        f"Nelder-Mead confirms the seed pump ({selected_semi['nozzle']}{selected_semi['throat']}) is already near-optimal."
                    )


def _augment_with_formation_marginals(batch_pump) -> None:
    """Compute mofwr and coeff_form for the formation-water axis.

    Mirrors what BatchPump.process_results does for motwr / molwr (and
    coeff_totl / coeff_lift), but for formation water. Done GUI-side so
    the library doesn't need a parallel change. Called after the batch
    run completes, and again after the calibration toggle re-runs
    process_results so mofwr stays in sync with the recomputed semi-
    finalists.
    """
    from woffl.assembly.batchpump import batch_curve_fit, gradient_back

    batch_pump.df = batch_pump.df.drop(columns=["mofwr"], errors="ignore")

    semi_df = batch_pump.df[batch_pump.df["semi"]].copy()
    if semi_df.empty:
        batch_pump.df["mofwr"] = np.nan
        batch_pump.coeff_form = None
        return

    semi_df = semi_df.sort_values(by="qoil_std", ascending=True)
    qoil_semi = semi_df["qoil_std"].to_numpy()
    fwat_semi = semi_df["form_wat"].to_numpy()

    semi_df["mofwr"] = gradient_back(qoil_semi, fwat_semi)

    batch_pump.df = batch_pump.df.merge(
        semi_df[["mofwr"]], left_index=True, right_index=True, how="left"
    )

    try:
        batch_pump.coeff_form = batch_curve_fit(qoil_semi, fwat_semi, origin=False)
    except Exception:
        batch_pump.coeff_form = None


def render_tab(
    params: SimulationParams, wellbore, well_profile, inflow, res_mix
) -> None:
    """Render the Batch Pump Analysis tab.

    Args:
        params: Simulation parameters from sidebar
        wellbore: PipeInPipe wellbore object
        well_profile: WellProfile object
        inflow: InFlow object
        res_mix: ResMix object
    """
    render_input_summary(params)

    # Surface any persisted message from a prior auto-recovery (e.g. GOR reset)
    msg = st.session_state.pop("_solver_gor_reset_msg", None)
    if msg:
        st.warning(msg)

    if not params.nozzle_batch_options or not params.throat_batch_options:
        st.warning(
            "Please select at least one nozzle size and one throat ratio for batch analysis."
        )
        return

    with st.spinner("Running batch pump simulation..."):
        batch_pump = run_batch_pump(
            params.surf_pres,
            params.form_temp,
            params.rho_pf,
            params.ppf_surf,
            wellbore,
            well_profile,
            inflow,
            res_mix,
            params.nozzle_batch_options,
            params.throat_batch_options,
            wellname=f"{params.field_model} Well",
            field_model=params.field_model,
            jpump_direction=params.jpump_direction,
            ken=params.ken,
            kth=params.kth,
            kdi=params.kdi,
        )

    if not batch_pump:
        return

    # Compute mofwr + coeff_form so the formation-water axis works without
    # a library change. Done before any downstream consumer (graph, table,
    # recommender) reads the dataframe.
    _augment_with_formation_marginals(batch_pump)

    # Inline marginal-WC quickfix — drives the recommended-pump gold star.
    # Lives just below the Model Inputs expander so the user can iterate
    # on the threshold without scrolling back to the sidebar.
    _render_marg_wc_quickfix(params)

    st.divider()

    # Performance graph — headline of the batch tab.
    _render_performance_graph(batch_pump, params)

    st.divider()

    # Hero strip — installed-pump batch row vs latest actuals. Lives below
    # the graph because single-pump model-vs-actual context is the Solver
    # tab's job; here it's secondary context for the recommendation table.
    # Side effect: computes + stashes the calibration so the toggle below
    # can use it. Returns True when PF rate mismatch is too large to trust
    # the rate-scalar calibration.
    pf_blocked = _render_batch_hero_strip(
        batch_pump, params, wellbore, well_profile
    )

    # Calibration toggle — scales modeled oil and formation water by the
    # installed-pump factor. The toggle sits below the performance graph,
    # so the scaling affects the data table and recommender below it
    # (not the graph above). Disabled while PF mismatch is too large to
    # trust the rate-scalar approach.
    cal = st.session_state.get("batch_calibration_result")
    if cal and cal.well_name == params.selected_well:
        cal_help = (
            "Disabled while PF rate mismatch is too large — fix the sidebar "
            "Power Fluid Surface Pressure first."
            if pf_blocked
            else (
                f"Scales oil and formation water rates by {cal.calibration_factor:.3f} "
                f"(derived from {cal.current_nozzle}{cal.current_throat} model vs actual). "
                "Affects the data table and recommender below."
            )
        )
        apply_cal = st.checkbox(
            f"Apply calibration factor ({cal.calibration_factor:.2f}) to results below",
            value=False,
            key="batch_apply_calibration",
            disabled=pf_blocked,
            help=cal_help,
        )
        # Streamlit preserves the underlying session_state value when a
        # checkbox is disabled — explicitly suppress the calibration when
        # PF is blocked so a stale-checked box doesn't slip through.
        if apply_cal and not pf_blocked:
            batch_pump.df["qoil_std_raw"] = batch_pump.df["qoil_std"]
            batch_pump.df["form_wat_raw"] = batch_pump.df["form_wat"]
            batch_pump.df["qoil_std"] = (
                batch_pump.df["qoil_std"] * cal.calibration_factor
            )
            batch_pump.df["form_wat"] = (
                batch_pump.df["form_wat"] * cal.calibration_factor
            )
            batch_pump.df["totl_wat"] = (
                batch_pump.df["form_wat"] + batch_pump.df["lift_wat"]
            )
            # Recompute semi-finalists, marginal ratios, curve fits from
            # calibrated data. Oil scales uniformly but total water does not
            # (lift water is unchanged), so the recommendation may shift to a
            # smaller pump when the model over-predicts.
            batch_pump.df.drop(
                columns=["semi", "motwr", "molwr", "mofwr"], errors="ignore", inplace=True
            )
            try:
                batch_pump.process_results()
            except ValueError:
                st.warning("No semi-finalist pumps found after applying calibration.")
                return
            # Re-derive mofwr / coeff_form from the calibrated semi-finalists
            # so the formation-water graph + recommender stay in sync.
            _augment_with_formation_marginals(batch_pump)
            st.caption(
                f"Showing **calibrated** results (factor {cal.calibration_factor:.3f} "
                f"from {cal.current_nozzle}{cal.current_throat}). "
                "Semi-finalists, marginal ratios, and recommendation in the "
                "table below have been recomputed from calibrated rates."
            )

    st.divider()

    # Data table + recommender (reflects calibration toggle above).
    _render_data_table(batch_pump, params)

    # Nelder-Mead continuous optimization
    _render_nelder_mead_section(batch_pump, params)
