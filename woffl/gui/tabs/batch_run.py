"""Tab 2: Batch Pump Analysis

Renders the batch run analysis including performance graphs, derivative plots,
data tables, and jet pump recommender results.
"""

import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from woffl.gui.components.dataframe_display import display_results_table
from woffl.gui.params import SimulationParams
from woffl.gui.utils import (
    highlight_recommended_pump,
    recommend_jetpump,
    run_batch_pump,
)


def _render_performance_graph(batch_pump, params: SimulationParams) -> None:
    """Render the performance graph sub-tab."""
    water = params.water_type if params.water_type is not None else "lift"
    st.subheader(f"Jet Pump Performance ({water.capitalize()} Water)")
    fig, ax = plt.subplots(figsize=(10, 6))

    has_curve_fit = hasattr(batch_pump, "coeff_totl") and hasattr(batch_pump, "coeff_lift")
    curve = bool(has_curve_fit)

    try:
        batch_pump.plot_data(water=water, curve=curve, ax=ax)

        # Get recommended jet pump based on marginal watercut
        try:
            recommendation = recommend_jetpump(batch_pump, params.marginal_watercut, water)
            highlight_recommended_pump(ax, recommendation, water)

            st.subheader("Recommended Jet Pump")
            rec_col1, rec_col2 = st.columns(2)

            with rec_col1:
                st.metric("Nozzle Size", recommendation["nozzle"])
                st.metric("Throat Ratio", recommendation["throat"])
                st.metric("Oil Rate", f"{recommendation['qoil_std']:.1f} BOPD")

            with rec_col2:
                water_label = "Lift Water" if water == "lift" else "Total Water"
                st.metric(water_label, f"{recommendation['water_rate']:.1f} BWPD")
                st.metric("Marginal Watercut", f"{recommendation['marginal_ratio']:.3f}")

                if recommendation["recommendation_type"] == "best_available":
                    st.warning(
                        "Note: No jet pump meets the specified marginal watercut threshold. "
                        "This is the best available option."
                    )
        except Exception as e:
            st.warning(f"Could not determine recommended jet pump: {str(e)}")

        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error generating performance plot: {str(e)}")
        st.info("Could not generate the performance plot. " "Try selecting different nozzle sizes and throat ratios.")
    finally:
        plt.close(fig)

    st.markdown(
        """
    **Graph Explanation:**
    - **Red points**: Semi-finalist jet pumps (no other pump produces more oil with less water)
    - **Blue points**: Eliminated jet pumps (another pump produces more oil with less water)
    - **Red dashed line**: Exponential curve fit of the semi-finalist pumps
    - **Gold star**: Recommended jet pump based on marginal watercut threshold
    - **Labels**: Show nozzle size + throat ratio (e.g., "12B" = nozzle 12, throat B)
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

    batch_tab1, batch_tab2, batch_tab3 = st.tabs(["Performance Graph", "Derivative Graph", "Data Table"])

    with batch_tab1:
        _render_performance_graph(batch_pump, params)

    with batch_tab2:
        _render_derivative_graph(batch_pump, params)

    with batch_tab3:
        _render_data_table(batch_pump, params)
