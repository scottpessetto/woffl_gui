"""Jet Pump Recommender

Functions for recommending optimal jet pump configurations based on marginal watercut thresholds.
This module analyzes batch run results to identify the most economical jet pump configuration.
"""

import numpy as np
import pandas as pd

import woffl.assembly.curvefit as cf


def recommend_jetpump(batch_pump, marginal_watercut, water_type="lift"):
    """Recommend Jet Pump Based on Marginal Watercut

    Analyzes batch run results to recommend a jet pump configuration where the
    marginal watercut is closest to but still below the specified threshold.
    This represents the economic limit where additional water handling is justified.

    Args:
        batch_pump (BatchPump): BatchPump object with processed results
        marginal_watercut (float): Threshold for marginal watercut (bbl water / (bbl water + bbl oil))
        water_type (str): "lift" or "total" depending on the desired analysis

    Returns:
        dict: Recommended jet pump configuration and performance metrics
            {
                'nozzle': str,
                'throat': str,
                'qoil_std': float,
                'water_rate': float,
                'marginal_ratio': float,
                'recommendation_type': str
            }
    """
    # Validate inputs
    if not hasattr(batch_pump, "df") or batch_pump.df.empty:
        raise ValueError("Batch pump has no results to analyze")

    if not hasattr(batch_pump, "coeff_totl") or not hasattr(batch_pump, "coeff_lift"):
        raise ValueError("Batch pump curve fitting has not been performed")

    # Validate water type
    water_type = _validate_water_type(water_type)

    # Get semi-finalist pumps
    semi_df = batch_pump.df[batch_pump.df["semi"]].copy()
    if semi_df.empty:
        raise ValueError("No semi-finalist jet pumps found")

    # Sort by oil rate (which correlates with water rate for semi-finalists)
    semi_df = semi_df.sort_values(by="qoil_std", ascending=True)

    # Get the appropriate coefficients and water rates
    coeff = batch_pump.coeff_lift if water_type == "lift" else batch_pump.coeff_totl
    water_col = "lift_wat" if water_type == "lift" else "totl_wat"
    marg_col = "molwr" if water_type == "lift" else "motwr"

    # Get water rates and calculate marginal watercuts for semi-finalists
    water_rates = semi_df[water_col].values

    # Convert marginal oil-water ratios to marginal watercuts
    # Original ratios are (bbl oil / bbl water)
    # We need (bbl water / (bbl water + bbl oil))
    original_ratios = semi_df[marg_col].values
    marginal_watercuts = 1 / (1 + original_ratios)

    # Check if any pumps meet the threshold (below the watercut threshold)
    below_threshold = marginal_watercuts <= marginal_watercut

    # If no pumps meet the threshold, recommend the one with lowest marginal watercut
    if not any(below_threshold):
        best_idx = np.argmin(marginal_watercuts)
        recommendation = {
            "nozzle": semi_df.iloc[best_idx]["nozzle"],
            "throat": semi_df.iloc[best_idx]["throat"],
            "qoil_std": semi_df.iloc[best_idx]["qoil_std"],
            "water_rate": semi_df.iloc[best_idx][water_col],
            "marginal_ratio": marginal_watercuts[best_idx],
            "recommendation_type": "best_available",
        }
        return recommendation

    # Find theoretical optimal water rate using curve fit
    # This is where the marginal watercut equals the threshold
    # We need to convert the watercut threshold to an oil-water ratio first
    # watercut = water / (water + oil)
    # oil / water = (1 - watercut) / watercut
    oil_water_ratio = (1 - marginal_watercut) / marginal_watercut

    a, b, c = coeff
    optimal_water_rate = cf.rev_exp_deriv(oil_water_ratio, b, c)
    optimal_oil_rate = cf.exp_model(optimal_water_rate, a, b, c)

    # Find the closest semi-finalist pump just below the threshold
    # First, filter to only those below threshold
    valid_indices = np.where(below_threshold)[0]

    # If we have the theoretical point, find the closest actual pump
    if valid_indices.size > 0:
        # Calculate distances to the theoretical optimal point
        distances = []
        for idx in valid_indices:
            pump_water = water_rates[idx]
            pump_oil = semi_df.iloc[idx]["qoil_std"]
            # Calculate Euclidean distance in the oil-water space
            distance = np.sqrt((pump_water - optimal_water_rate) ** 2 + (pump_oil - optimal_oil_rate) ** 2)
            distances.append(distance)

        # Find the closest pump
        closest_idx = valid_indices[np.argmin(distances)]

        recommendation = {
            "nozzle": semi_df.iloc[closest_idx]["nozzle"],
            "throat": semi_df.iloc[closest_idx]["throat"],
            "qoil_std": semi_df.iloc[closest_idx]["qoil_std"],
            "water_rate": semi_df.iloc[closest_idx][water_col],
            "marginal_ratio": marginal_watercuts[closest_idx],
            "recommendation_type": "optimal",
            "theoretical_water_rate": optimal_water_rate,
            "theoretical_oil_rate": optimal_oil_rate,
        }
        return recommendation

    # This should never happen if above_threshold check is done correctly
    raise ValueError("Could not determine recommended jet pump")


def _validate_water_type(water_type):
    """Validate Type of Water String

    Checks that the string passed into a method or argument fits the required description.
    This is used when the water type wants to be defined as lift or total.

    Args:
        water_type (str): "lift" or "total" depending on the desired analysis

    Returns:
        str: Properly formatted as either "lift" or "total"
    """
    # Validate the 'water' argument
    if water_type not in {"lift", "total", "totl"}:
        raise ValueError(f"Invalid value for 'water_type': {water_type}. Expected 'lift', 'total', or 'totl'.")

    # Standardize "totl" to "total"
    if water_type == "totl":
        water_type = "total"
    return water_type


def highlight_recommended_pump(ax, recommendation, water_type="lift"):
    """Highlight the recommended pump on a performance plot

    Args:
        ax (matplotlib.axes.Axes): The axes object to plot on
        recommendation (dict): The recommendation dictionary from recommend_jetpump
        water_type (str): "lift" or "total" depending on the plot type
    """
    if recommendation is None:
        return

    water_type = _validate_water_type(water_type)

    # Extract values from recommendation
    water_rate = recommendation["water_rate"]
    oil_rate = recommendation["qoil_std"]
    pump_label = recommendation["nozzle"] + recommendation["throat"]

    # Highlight the recommended pump with a star marker
    ax.plot(
        water_rate,
        oil_rate,
        marker="*",
        markersize=15,
        markerfacecolor="gold",
        markeredgecolor="black",
        markeredgewidth=1.5,
        linestyle="none",
        label=f"Recommended: {pump_label}",
    )

    # Add annotation
    ax.annotate(
        f"Recommended: {pump_label}",
        xy=(water_rate, oil_rate),
        xytext=(10, 10),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", alpha=0.8),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
    )
