"""Tab 3: Power Fluid Range Analysis

Renders the comprehensive power fluid range analysis including scatter plots,
data tables, and best performer identification.
"""

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from woffl.gui.components.dataframe_display import display_results_table
from woffl.gui.params import SimulationParams
from woffl.gui.utils import run_power_fluid_range_batch


def _render_performance_vs_pressure(successful_df: pd.DataFrame, field_model: str) -> None:
    """Render the Performance vs Pressure sub-tab."""
    st.subheader("Oil Rate vs Power Fluid Pressure")

    successful_df = successful_df.copy()
    successful_df["pump_combo"] = successful_df["nozzle"] + successful_df["throat"]
    unique_combos = successful_df["pump_combo"].unique()
    colors = cm.get_cmap("tab20")(np.linspace(0, 1, len(unique_combos)))

    # Oil rate plot
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, combo in enumerate(unique_combos):
        combo_data = successful_df[successful_df["pump_combo"] == combo]
        ax.scatter(
            combo_data["power_fluid_pressure"],
            combo_data["qoil_std"],
            c=[colors[i]],
            label=combo,
            alpha=0.7,
            s=50,
        )
    ax.set_xlabel("Power Fluid Pressure (psi)")
    ax.set_ylabel("Oil Rate (BOPD)")
    ax.set_title(f"{field_model} Well - Oil Rate vs Power Fluid Pressure")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

    # Total water plot
    st.subheader("Total Water vs Power Fluid Pressure")
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    for i, combo in enumerate(unique_combos):
        combo_data = successful_df[successful_df["pump_combo"] == combo]
        ax2.scatter(
            combo_data["power_fluid_pressure"],
            combo_data["totl_wat"],
            c=[colors[i]],
            label=combo,
            alpha=0.7,
            s=50,
        )
    ax2.set_xlabel("Power Fluid Pressure (psi)")
    ax2.set_ylabel("Total Water (BWPD)")
    ax2.set_title(f"{field_model} Well - Total Water vs Power Fluid Pressure")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)
    plt.close(fig2)


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


def render_tab(params: SimulationParams, tube, well_profile, inflow, res_mix) -> None:
    """Render the Power Fluid Range Analysis tab.

    Args:
        params: Simulation parameters from sidebar
        tube: Tubing Pipe object
        well_profile: WellProfile object
        inflow: InFlow object
        res_mix: ResMix object
    """
    st.subheader("Power Fluid Range Analysis")

    if not params.nozzle_batch_options or not params.throat_batch_options:
        st.warning("Please select at least one nozzle size and one throat ratio for power fluid range analysis.")
        return

    if params.power_fluid_min >= params.power_fluid_max:
        st.error("Minimum power fluid pressure must be less than maximum power fluid pressure.")
        return

    if params.power_fluid_step <= 0 or params.power_fluid_step >= (params.power_fluid_max - params.power_fluid_min):
        st.error("Power fluid step must be positive and less than the pressure range.")
        return

    # Calculate number of pressure points
    pressure_points = int((params.power_fluid_max - params.power_fluid_min) / params.power_fluid_step) + 1
    total_combinations = pressure_points * len(params.nozzle_batch_options) * len(params.throat_batch_options)

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
            tube,
            well_profile,
            inflow,
            res_mix,
            params.nozzle_batch_options,
            params.throat_batch_options,
            wellname=f"{params.field_model} Well",
        )

    if comprehensive_df is None or comprehensive_df.empty:
        st.error("Failed to run power fluid range analysis. Check your parameters and try again.")
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
        st.metric("Pressure Range", f"{params.power_fluid_min}-{params.power_fluid_max} psi")
        success_pct = len(successful_df) / len(comprehensive_df) * 100
        st.metric("Success Rate", f"{success_pct:.1f}%")

    # Visualization tabs
    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Performance vs Pressure", "Comprehensive Data", "Best Performers"])

    with viz_tab1:
        _render_performance_vs_pressure(successful_df, params.field_model)

    with viz_tab2:
        _render_comprehensive_data(successful_df)

    with viz_tab3:
        _render_best_performers(successful_df)

    st.markdown(
        """
        **Analysis Explanation:**
        - This comprehensive analysis tests all selected nozzle and throat combinations across the specified power fluid pressure range
        - **Performance vs Pressure**: Shows how oil and water rates change with power fluid pressure for each pump configuration
        - **Comprehensive Data**: Complete results table with all successful combinations
        - **Best Performers**: Shows the highest oil-producing pump at each pressure point and identifies the overall best performer
        """
    )
