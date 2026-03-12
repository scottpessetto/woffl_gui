"""Tab 2: Batch Pump Analysis

Renders the batch run analysis including performance graphs, derivative plots,
data tables, and jet pump recommender results.
"""

import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from woffl.assembly import curvefit as cf
from woffl.assembly.calibration import CalibrationResult
from woffl.gui.components.dataframe_display import display_results_table
from woffl.gui.params import SimulationParams
from woffl.gui.utils import (
    highlight_recommended_pump,
    is_valid_number,
    recommend_jetpump,
    run_batch_pump,
)


def _render_performance_graph(batch_pump, params: SimulationParams) -> None:
    """Render the performance graph sub-tab using interactive Plotly."""
    water = params.water_type if params.water_type is not None else "lift"
    st.subheader(f"Jet Pump Performance ({water.capitalize()} Water)")

    has_curve_fit = hasattr(batch_pump, "coeff_totl") and hasattr(batch_pump, "coeff_lift")

    try:
        df = batch_pump.df.copy()
        df = df[~df["qoil_std"].isna()]

        water_col = "lift_wat" if water == "lift" else "totl_wat"
        water_label = "Lift Water" if water == "lift" else "Total Water"

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
                    marker=dict(size=10, color="royalblue", line=dict(width=1, color="black")),
                    text=jp_names_elim,
                    textposition="top right",
                    textfont=dict(size=10),
                    customdata=np.stack(
                        [elim["nozzle"], elim["throat"], elim["psu_solv"], elim["mach_te"], elim["form_wat"]],
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
                    marker=dict(size=12, color="crimson", symbol="diamond", line=dict(width=1, color="black")),
                    text=jp_names_semi,
                    textposition="top right",
                    textfont=dict(size=10, color="crimson"),
                    customdata=np.stack(
                        [semi["nozzle"], semi["throat"], semi["psu_solv"], semi["mach_te"], semi["form_wat"]],
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

        # Curve fit line
        if has_curve_fit:
            coeff = batch_pump.coeff_lift if water == "lift" else batch_pump.coeff_totl
            a, b, c = coeff
            max_water = df[water_col].max()
            fit_water = np.linspace(0, max_water, 200)
            fit_oil = np.array([cf.exp_model(w, a, b, c) for w in fit_water])
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
            recommendation = recommend_jetpump(batch_pump, params.marginal_watercut, water)
            rec_water = recommendation["water_rate"]
            rec_oil = recommendation["qoil_std"]
            fig.add_trace(
                go.Scatter(
                    x=[rec_water],
                    y=[rec_oil],
                    mode="markers",
                    name="Recommended",
                    marker=dict(size=20, color="gold", symbol="star", line=dict(width=2, color="black")),
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

        fig.update_layout(
            title=dict(text=f"<b>{batch_pump.wellname}</b> Jet Pump Performance", font=dict(size=16)),
            xaxis_title=f"{water.capitalize()} Water Rate (BWPD)",
            yaxis_title="Produced Oil Rate (BOPD)",
            xaxis=dict(range=[0, None], gridcolor="lightgray"),
            yaxis=dict(range=[0, None], gridcolor="lightgray"),
            plot_bgcolor="white",
            hovermode="closest",
            height=550,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
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
                st.metric("Marginal Watercut", f"{recommendation['marginal_ratio']:.3f}")

                if recommendation["recommendation_type"] == "best_available":
                    st.warning(
                        "Note: No jet pump meets the specified marginal watercut threshold. "
                        "This is the best available option."
                    )

    except Exception as e:
        st.error(f"Error generating performance plot: {str(e)}")
        st.info("Could not generate the performance plot. Try selecting different nozzle sizes and throat ratios.")

    st.markdown(
        """
    **Graph Explanation:**
    - **Red diamonds**: Semi-finalist jet pumps (no other pump produces more oil with less water)
    - **Blue circles**: Eliminated jet pumps (another pump produces more oil with less water)
    - **Red dashed line**: Exponential curve fit of the semi-finalist pumps
    - **Gold star**: Recommended jet pump based on marginal watercut threshold
    - **Labels**: Show nozzle size + throat ratio (e.g., "12B" = nozzle 12, throat B)
    - **Hover** over any point for detailed metrics
    """
    )


def _render_derivative_graph(batch_pump, params: SimulationParams) -> None:
    """Render the derivative graph sub-tab."""
    water = params.water_type if params.water_type is not None else "lift"
    st.subheader(f"Marginal Oil-Water Ratio ({water.capitalize()} Water)")

    has_curve_fit = hasattr(batch_pump, "coeff_totl") and hasattr(batch_pump, "coeff_lift")

    if has_curve_fit:
        try:
            with tempfile.TemporaryDirectory() as tmpdirname:
                fig_path = os.path.join(tmpdirname, "derv_plot.png")
                batch_pump.plot_derv(water=water, fig_path=fig_path)
                st.image(fig_path)
        except Exception as e:
            st.error(f"Error generating derivative plot: {str(e)}")
            st.info(
                "The derivative plot requires at least two semi-finalist jet pumps "
                "to calculate the marginal oil-water ratio."
            )
    else:
        st.warning("Curve fitting failed, so the derivative plot cannot be generated.")
        st.info(
            "Try selecting more nozzle sizes and throat ratios to improve curve fitting. "
            "At least two semi-finalist jet pumps are required."
        )

    st.markdown(
        """
    **Graph Explanation:**
    - **Points**: Marginal oil-water ratio for each semi-finalist jet pump
    - **Line**: Analytical derivative of the exponential curve fit
    - **Marginal Oil-Water Ratio**: Additional oil production per additional barrel of water
    """
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
        st.warning("No semi-finalist jet pumps found. Cannot display recommender results.")
        return

    semi_df = semi_df.sort_values(by="qoil_std", ascending=True)

    water_col = "lift_wat" if water == "lift" else "totl_wat"
    marg_col = "molwr" if water == "lift" else "motwr"

    # Convert marginal oil-water ratios to marginal watercuts
    semi_df["marginal_watercut"] = 1 / (1 + semi_df[marg_col])

    # Build display columns
    water_label = "Lift Water (BWPD)" if water == "lift" else "Total Water (BWPD)"
    ratio_label = "Marginal Oil/Lift Water Ratio" if water == "lift" else "Marginal Oil/Total Water Ratio"

    recommender_df = semi_df[["nozzle", "throat", "qoil_std", water_col, marg_col, "marginal_watercut"]].copy()

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


def _compute_batch_calibration(batch_pump, params: SimulationParams) -> CalibrationResult | None:
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


def _render_model_vs_actual(batch_pump, params: SimulationParams) -> None:
    """Render Model vs Actual comparison and calibration for the batch run.

    Compares the currently installed pump's batch result to the most recent well test.
    Stores calibration in session state for optional application to all batch results.
    """
    cal = _compute_batch_calibration(batch_pump, params)
    if cal is None:
        return

    from woffl.assembly.jp_history import get_current_pump

    jp_hist = st.session_state.get("jp_history_df")
    current_pump = get_current_pump(jp_hist, params.selected_well)
    date_str = current_pump["date_set"].strftime("%Y-%m-%d") if current_pump["date_set"] is not None else "N/A"

    st.divider()
    st.subheader("Model vs Actual Comparison")
    st.info(
        f"Comparing batch result for the **current installed pump** "
        f"({cal.current_nozzle}{cal.current_throat}, set {date_str}) "
        f"to the most recent well test for **{params.selected_well}**."
    )

    # Side-by-side metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Modeled Oil Rate", f"{cal.model_oil:.0f} BOPD")
    with col2:
        st.metric("Actual Oil Rate", f"{cal.actual_oil:.0f} BOPD")
    with col3:
        st.metric("Delta", f"{cal.model_oil - cal.actual_oil:+.0f} BOPD")

    if cal.actual_bhp is not None and cal.model_bhp is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Modeled BHP (suction)", f"{cal.model_bhp:.0f} psi")
        with col2:
            st.metric("Actual BHP", f"{cal.actual_bhp:.0f} psi")
        with col3:
            st.metric("Delta", f"{cal.model_bhp - cal.actual_bhp:+.0f} psi")

    if cal.model_pf is not None and cal.actual_pf is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Modeled PF Rate", f"{cal.model_pf:.0f} BWPD")
        with col2:
            st.metric("Actual PF Rate", f"{cal.actual_pf:.0f} BWPD")
        with col3:
            st.metric("Delta", f"{cal.model_pf - cal.actual_pf:+.0f} BWPD")

        if abs(cal.model_pf - cal.actual_pf) > 100:
            st.warning(
                f"Modeled PF rate differs from actual by {abs(cal.model_pf - cal.actual_pf):.0f} BWPD. "
                "Check that the **Power Fluid Pressure** in the sidebar matches the actual PF pressure for this well."
            )

    # Calibration summary
    st.markdown("#### Model Calibration")
    grade = cal.quality_grade
    grade_color = {"good": "green", "fair": "orange", "poor": "red"}[grade]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Calibration Factor", f"{cal.calibration_factor:.3f}")
    with c2:
        st.metric("Oil Error", f"{cal.oil_error_pct:.1f}%")
    with c3:
        st.markdown(f"**Quality:** :{grade_color}[{grade.upper()}]")

    # Store for use in calibrated batch results
    st.session_state["batch_calibration_result"] = cal


def render_tab(params: SimulationParams, tube, well_profile, inflow, res_mix) -> None:
    """Render the Batch Pump Analysis tab.

    Args:
        params: Simulation parameters from sidebar
        tube: Tubing Pipe object
        well_profile: WellProfile object
        inflow: InFlow object
        res_mix: ResMix object
    """
    st.subheader("Batch Pump Analysis")

    if not params.nozzle_batch_options or not params.throat_batch_options:
        st.warning("Please select at least one nozzle size and one throat ratio for batch analysis.")
        return

    with st.spinner("Running batch pump simulation..."):
        batch_pump = run_batch_pump(
            params.surf_pres,
            params.form_temp,
            params.rho_pf,
            params.ppf_surf,
            tube,
            well_profile,
            inflow,
            res_mix,
            params.nozzle_batch_options,
            params.throat_batch_options,
            wellname=f"{params.field_model} Well",
        )

    if not batch_pump:
        return

    # Model vs Actual + calibration (before tabs so factor is available)
    _render_model_vs_actual(batch_pump, params)

    # Calibration toggle
    cal = st.session_state.get("batch_calibration_result")
    apply_cal = False
    if cal and cal.well_name == params.selected_well:
        apply_cal = st.checkbox(
            f"Apply calibration factor ({cal.calibration_factor:.2f}) to all batch results",
            value=False,
            key="batch_apply_calibration",
            help=(
                f"Scales all oil and formation water rates by {cal.calibration_factor:.3f} "
                f"(derived from {cal.current_nozzle}{cal.current_throat} model vs actual)."
            ),
        )
        if apply_cal:
            # Scale oil and formation water by calibration factor
            batch_pump.df["qoil_std_raw"] = batch_pump.df["qoil_std"]
            batch_pump.df["form_wat_raw"] = batch_pump.df["form_wat"]
            batch_pump.df["qoil_std"] = batch_pump.df["qoil_std"] * cal.calibration_factor
            batch_pump.df["form_wat"] = batch_pump.df["form_wat"] * cal.calibration_factor
            batch_pump.df["totl_wat"] = batch_pump.df["form_wat"] + batch_pump.df["lift_wat"]

            # Re-run process_results to recompute semi-finalists, marginal ratios,
            # and curve fits from calibrated data. Oil scales uniformly but total water
            # does not (lift water is unchanged), so the recommendation may shift to a
            # smaller pump when the model over-predicts.
            batch_pump.df.drop(columns=["semi", "motwr", "molwr"], errors="ignore", inplace=True)
            try:
                batch_pump.process_results()
            except ValueError:
                st.warning("No semi-finalist pumps found after applying calibration.")
                return

            st.caption(
                f"Showing **calibrated** results (factor {cal.calibration_factor:.3f} "
                f"from {cal.current_nozzle}{cal.current_throat}). "
                "Semi-finalists, marginal ratios, curve fit, and recommendation "
                "have been recomputed from calibrated rates."
            )

    batch_tab1, batch_tab2, batch_tab3 = st.tabs(["Performance Graph", "Derivative Graph", "Data Table"])

    with batch_tab1:
        _render_performance_graph(batch_pump, params)

    with batch_tab2:
        _render_derivative_graph(batch_pump, params)

    with batch_tab3:
        _render_data_table(batch_pump, params)
