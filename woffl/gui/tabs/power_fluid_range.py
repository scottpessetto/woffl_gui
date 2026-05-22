"""Tab 3: Power Fluid Range Analysis

Renders the comprehensive power fluid range analysis including scatter plots,
data tables, and best performer identification.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from woffl.gui.components.dataframe_display import display_results_table
from woffl.gui.params import SimulationParams
from woffl.gui.utils import (
    build_calibration_inputs,
    is_valid_number,
    render_bhp_calibration_warning,
    render_pf_mismatch_warning,
    render_pf_quickfix_widget,
    run_power_fluid_range_batch,
)

# Qualitative palette that stays legible on white
_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
]


def _render_performance_vs_pressure(
    successful_df: pd.DataFrame, field_model: str
) -> None:
    """Render the Performance vs Pressure sub-tab."""

    successful_df = successful_df.copy()
    successful_df["pump_combo"] = successful_df["nozzle"] + successful_df["throat"]
    combos = sorted(successful_df["pump_combo"].unique())

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Oil Rate vs PF Pressure", "Total Water vs PF Pressure"),
        horizontal_spacing=0.10,
    )

    for i, combo in enumerate(combos):
        cd = successful_df[successful_df["pump_combo"] == combo].sort_values(
            "power_fluid_pressure"
        )
        color = _PALETTE[i % len(_PALETTE)]

        custom = np.stack(
            [cd["psu_solv"], cd["totl_wat"], cd["form_wat"], cd["lift_wat"]], axis=-1
        )

        hover = (
            f"<b>{combo}</b><br>"
            "PF Pressure: %{x:.0f} psi<br>"
            "Oil: %{y:.0f} BOPD<br>"
            "Suction P: %{customdata[0]:.0f} psi<br>"
            "Total Water: %{customdata[1]:.0f} BWPD<br>"
            "Form Water: %{customdata[2]:.0f} BWPD<br>"
            "Lift Water: %{customdata[3]:.0f} BWPD"
            "<extra></extra>"
        )

        hover_water = (
            f"<b>{combo}</b><br>"
            "PF Pressure: %{x:.0f} psi<br>"
            "Total Water: %{y:.0f} BWPD<br>"
            "Oil: %{customdata[0]:.0f} BOPD"
            "<extra></extra>"
        )

        # Oil rate trace (left)
        fig.add_trace(
            go.Scatter(
                x=cd["power_fluid_pressure"],
                y=cd["qoil_std"],
                mode="lines+markers",
                name=combo,
                legendgroup=combo,
                line=dict(color=color, width=2),
                marker=dict(size=7, line=dict(width=0.5, color="black")),
                customdata=custom,
                hovertemplate=hover,
            ),
            row=1, col=1,
        )

        # Total water trace (right) — hide from legend (shared legendgroup)
        fig.add_trace(
            go.Scatter(
                x=cd["power_fluid_pressure"],
                y=cd["totl_wat"],
                mode="lines+markers",
                name=combo,
                legendgroup=combo,
                showlegend=False,
                line=dict(color=color, width=2),
                marker=dict(size=7, line=dict(width=0.5, color="black")),
                customdata=np.stack([cd["qoil_std"]], axis=-1),
                hovertemplate=hover_water,
            ),
            row=1, col=2,
        )

    fig.update_xaxes(title_text="Power Fluid Pressure (psi)", gridcolor="lightgray",
                     row=1, col=1)
    fig.update_xaxes(title_text="Power Fluid Pressure (psi)", gridcolor="lightgray",
                     row=1, col=2)
    fig.update_yaxes(title_text="Oil Rate (BOPD)", gridcolor="lightgray",
                     rangemode="tozero", row=1, col=1)
    fig.update_yaxes(title_text="Total Water (BWPD)", gridcolor="lightgray",
                     rangemode="tozero", row=1, col=2)

    fig.update_layout(
        title=dict(
            text=f"<b>{field_model} Well</b> — Power Fluid Sensitivity",
            font=dict(size=16),
        ),
        plot_bgcolor="white",
        hovermode="closest",
        height=600,
        legend=dict(
            title_text="Pump",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=10),
        ),
        margin=dict(r=100),
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_comprehensive_data(successful_df: pd.DataFrame) -> None:
    """Render the Comprehensive Data sub-tab."""
    st.subheader("Comprehensive Analysis Results")

    display_results_table(
        df=successful_df,
        columns=[
            "power_fluid_pressure",
            "nozzle",
            "throat",
            "qoil_std",
            "form_wat",
            "lift_wat",
            "totl_wat",
            "psu_solv",
            "mach_te",
            "sonic_status",
        ],
        rename_map={
            "power_fluid_pressure": "Power Fluid Pressure (psi)",
            "nozzle": "Nozzle",
            "throat": "Throat",
            "qoil_std": "Oil Rate (BOPD)",
            "form_wat": "Formation Water (BWPD)",
            "lift_wat": "Lift Water (BWPD)",
            "totl_wat": "Total Water (BWPD)",
            "psu_solv": "Suction Pressure (psig)",
            "mach_te": "Throat Entry Mach",
            "sonic_status": "Sonic Flow",
        },
        numeric_cols=[
            "Power Fluid Pressure (psi)",
            "Oil Rate (BOPD)",
            "Formation Water (BWPD)",
            "Lift Water (BWPD)",
            "Total Water (BWPD)",
            "Suction Pressure (psig)",
            "Throat Entry Mach",
        ],
        download_filename="jetpump_power_fluid_range_analysis.csv",
        download_label="Download Comprehensive Results CSV",
        sort_by=["Power Fluid Pressure (psi)", "Oil Rate (BOPD)"],
        sort_ascending=[True, False],
    )


def _render_best_performers(successful_df: pd.DataFrame) -> None:
    """Render the Best Performers sub-tab."""
    st.subheader("Best Performers at Each Pressure")

    best_performers = []
    for pressure in successful_df["power_fluid_pressure"].unique():
        pressure_data = successful_df[successful_df["power_fluid_pressure"] == pressure]
        if not pressure_data.empty:
            best_performer = pressure_data.loc[pressure_data["qoil_std"].idxmax()]
            best_performers.append(best_performer)

    if not best_performers:
        st.warning("No best performers found in the analysis.")
        return

    best_df = pd.DataFrame(best_performers)

    display_results_table(
        df=best_df,
        columns=[
            "power_fluid_pressure",
            "nozzle",
            "throat",
            "qoil_std",
            "totl_wat",
            "psu_solv",
            "sonic_status",
        ],
        rename_map={
            "power_fluid_pressure": "Power Fluid Pressure (psi)",
            "nozzle": "Nozzle",
            "throat": "Throat",
            "qoil_std": "Oil Rate (BOPD)",
            "totl_wat": "Total Water (BWPD)",
            "psu_solv": "Suction Pressure (psig)",
            "sonic_status": "Sonic Flow",
        },
        numeric_cols=[
            "Power Fluid Pressure (psi)",
            "Oil Rate (BOPD)",
            "Total Water (BWPD)",
            "Suction Pressure (psig)",
        ],
        download_filename="jetpump_best_performers_by_pressure.csv",
        download_label="Download Best Performers CSV",
        sort_by=["Power Fluid Pressure (psi)"],
        sort_ascending=[True],
    )

    # Overall best performer
    overall_best = best_df.loc[best_df["qoil_std"].idxmax()]
    st.subheader("Overall Best Performer")

    best_col1, best_col2, best_col3 = st.columns(3)

    with best_col1:
        st.metric(
            "Best Configuration",
            f"Nozzle {overall_best['nozzle']}, Throat {overall_best['throat']}",
        )
        st.metric(
            "Power Fluid Pressure",
            f"{overall_best['power_fluid_pressure']:.0f} psi",
        )

    with best_col2:
        st.metric("Oil Rate", f"{overall_best['qoil_std']:.1f} BOPD")
        st.metric("Total Water", f"{overall_best['totl_wat']:.1f} BWPD")

    with best_col3:
        st.metric("Suction Pressure", f"{overall_best['psu_solv']:.1f} psig")
        st.metric("Sonic Flow", "Yes" if bool(overall_best["sonic_status"]) else "No")


def _render_calibration_flags_for_installed_pump(
    successful_df: pd.DataFrame,
    params: SimulationParams,
    wellbore,
    well_profile,
) -> None:
    """Surface PF + BHP calibration flags for the installed pump.

    PF Range sweeps PF pressure across many pumps, but the only meaningful
    "vs actual" comparison is for the **currently installed pump** at the
    PF pressure closest to the sidebar value (the user's operating point).
    Same flag hierarchy as Solver + Batch Run: PF mismatch is foundational,
    BHP flag only meaningful once PF is right.
    """
    if params.selected_well == "Custom":
        return

    jp_hist = st.session_state.get("jp_history_df")
    if jp_hist is None:
        return

    from woffl.assembly.jp_history import get_current_pump

    current_pump = get_current_pump(jp_hist, params.selected_well)
    if current_pump is None:
        return
    nozzle = current_pump.get("nozzle_no")
    throat = current_pump.get("throat_ratio")
    if not nozzle or not throat:
        return

    # Find installed pump's row at the PF pressure closest to sidebar
    pump_rows = successful_df[
        (successful_df["nozzle"] == nozzle) & (successful_df["throat"] == throat)
    ]
    if pump_rows.empty:
        return  # Installed pump wasn't in this sweep
    closest_idx = (pump_rows["power_fluid_pressure"] - params.ppf_surf).abs().idxmin()
    modeled_bhp = pump_rows.loc[closest_idx, "psu_solv"]
    modeled_pf = pump_rows.loc[closest_idx, "lift_wat"]

    # Actuals from latest well test (memory-gauge override applied if active).
    from woffl.gui.utils import get_well_tests_for_well

    well_tests = get_well_tests_for_well(params.selected_well)
    if well_tests is None or well_tests.empty:
        return
    recent = well_tests.sort_values("WtDate", ascending=False).iloc[0]
    actual_bhp = recent.get("BHP") if "BHP" in well_tests.columns else None
    actual_pf = recent.get("lift_wat") if "lift_wat" in well_tests.columns else None

    # PF mismatch first — fix this before trusting any BHP comparison
    cal_inputs = build_calibration_inputs(params, wellbore, well_profile)
    test_date_str = cal_inputs["test_date_str"] if cal_inputs else None

    pf_blocked = False
    if is_valid_number(actual_pf):
        pf_warning_shown, pf_blocked = render_pf_mismatch_warning(
            float(modeled_pf),
            float(actual_pf),
            params.ppf_surf,
            test_date_str=test_date_str,
            well_name=params.selected_well,
        )
        if pf_warning_shown:
            render_pf_quickfix_widget(
                params,
                wellbore,
                well_profile,
                target_lift_wat=float(actual_pf),
            )

    # BHP flag only meaningful once PF is right
    if not pf_blocked and is_valid_number(actual_bhp):
        render_bhp_calibration_warning(float(modeled_bhp), float(actual_bhp))


def render_tab(
    params: SimulationParams, wellbore, well_profile, inflow, res_mix
) -> None:
    """Render the Power Fluid Range Analysis tab.

    Args:
        params: Simulation parameters from sidebar
        wellbore: PipeInPipe wellbore object
        well_profile: WellProfile object
        inflow: InFlow object
        res_mix: ResMix object
    """
    st.subheader(
        "Power Fluid Range Analysis",
        help=(
            "Tests every selected nozzle/throat combination across the "
            "configured PF pressure range.\n\n"
            "- **Performance vs Pressure** — oil and total water vs PF "
            "pressure for each combination.\n"
            "- **Comprehensive Data** — full results table for every "
            "successful combo.\n"
            "- **Best Performers** — highest-oil pump at each pressure "
            "point, plus the overall winner."
        ),
    )

    if not params.nozzle_batch_options or not params.throat_batch_options:
        st.warning(
            "Please select at least one nozzle size and one throat ratio for power fluid range analysis."
        )
        return

    if params.power_fluid_min >= params.power_fluid_max:
        st.error(
            "Minimum power fluid pressure must be less than maximum power fluid pressure."
        )
        return

    if params.power_fluid_step <= 0 or params.power_fluid_step >= (
        params.power_fluid_max - params.power_fluid_min
    ):
        st.error("Power fluid step must be positive and less than the pressure range.")
        return

    # Calculate number of pressure points
    pressure_points = (
        int((params.power_fluid_max - params.power_fluid_min) / params.power_fluid_step)
        + 1
    )
    total_combinations = (
        pressure_points
        * len(params.nozzle_batch_options)
        * len(params.throat_batch_options)
    )

    st.info(
        f"This analysis will test {pressure_points} pressure points with "
        f"{len(params.nozzle_batch_options)} nozzles and {len(params.throat_batch_options)} throat ratios "
        f"for a total of {total_combinations} combinations."
    )

    with st.spinner("Running comprehensive power fluid range analysis..."):
        comprehensive_df = run_power_fluid_range_batch(
            params.surf_pres,
            params.form_temp,
            params.rho_pf,
            params.power_fluid_min,
            params.power_fluid_max,
            params.power_fluid_step,
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

    if comprehensive_df is None or comprehensive_df.empty:
        st.error(
            "Failed to run power fluid range analysis. Check your parameters and try again."
        )
        return

    successful_df = comprehensive_df[~comprehensive_df["qoil_std"].isna()].copy()

    if successful_df.empty:
        st.error("No successful simulation runs found. Check your parameter settings.")
        return

    # Summary statistics
    st.subheader("Analysis Summary")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Combinations", len(comprehensive_df))
        st.metric("Successful Runs", len(successful_df))

    with col2:
        st.metric("Max Oil Rate", f"{successful_df['qoil_std'].max():.1f} BOPD")
        st.metric("Min Oil Rate", f"{successful_df['qoil_std'].min():.1f} BOPD")

    with col3:
        st.metric("Max Total Water", f"{successful_df['totl_wat'].max():.1f} BWPD")
        st.metric("Min Total Water", f"{successful_df['totl_wat'].min():.1f} BWPD")

    with col4:
        st.metric(
            "Pressure Range", f"{params.power_fluid_min}-{params.power_fluid_max} psi"
        )
        success_pct = len(successful_df) / len(comprehensive_df) * 100
        st.metric("Success Rate", f"{success_pct:.1f}%")

    # PF + BHP calibration flags for the installed pump at the sidebar's PF
    # pressure. Same flag hierarchy as Solver and Batch Run.
    _render_calibration_flags_for_installed_pump(
        successful_df, params, wellbore, well_profile
    )

    # Visualization tabs
    viz_tab1, viz_tab2, viz_tab3 = st.tabs(
        ["Performance vs Pressure", "Comprehensive Data", "Best Performers"]
    )

    with viz_tab1:
        _render_performance_vs_pressure(successful_df, params.field_model)

    with viz_tab2:
        _render_comprehensive_data(successful_df)

    with viz_tab3:
        _render_best_performers(successful_df)
