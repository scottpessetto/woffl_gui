"""Visualization Utilities for Multi-Well Optimization

Functions for creating charts and plots for optimization results.
"""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from woffl.assembly.network_optimizer import OptimizationResult


def create_power_fluid_pie_chart(results: list["OptimizationResult"]):
    """Create pie chart showing power fluid allocation across wells

    Args:
        results: List of optimization results

    Returns:
        matplotlib figure
    """
    labels = [r.well_name for r in results]
    sizes = [r.allocated_power_fluid for r in results]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create pie chart
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%", startangle=90, textprops={"fontsize": 10}
    )

    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")

    ax.axis("equal")
    plt.title("Power Fluid Allocation by Well", fontsize=14, fontweight="bold", pad=20)

    return fig


def create_oil_rate_bar_chart(results: list["OptimizationResult"], show_baseline=False, baseline_dict=None):
    """Create bar chart comparing well oil rates

    Args:
        results: List of optimization results
        show_baseline: Whether to show baseline comparison
        baseline_dict: Dictionary mapping well_name to baseline oil rate

    Returns:
        matplotlib figure
    """
    wells = [r.well_name for r in results]
    optimized = [r.predicted_oil_rate for r in results]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(wells))
    width = 0.35 if show_baseline and baseline_dict else 0.7

    if show_baseline and baseline_dict:
        baseline_rates = [baseline_dict.get(r.well_name, 0) for r in results]
        ax.bar(x - width / 2, baseline_rates, width, label="Baseline", alpha=0.8)
        ax.bar(x + width / 2, optimized, width, label="Optimized", alpha=0.8)
    else:
        ax.bar(x, optimized, width, label="Optimized", alpha=0.8)

    ax.set_xlabel("Well", fontsize=12)
    ax.set_ylabel("Oil Rate (BOPD)", fontsize=12)
    ax.set_title("Oil Rate by Well", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(wells, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


def create_pump_config_chart(results: list["OptimizationResult"]):
    """Create chart showing pump configurations across wells

    Args:
        results: List of optimization results

    Returns:
        matplotlib figure
    """
    wells = [r.well_name for r in results]
    configs = [f"{r.recommended_nozzle}{r.recommended_throat}" for r in results]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create bar chart with config labels
    y_pos = np.arange(len(wells))
    ax.barh(y_pos, [1] * len(wells), alpha=0.6)

    # Add config labels
    for i, (well, config) in enumerate(zip(wells, configs)):
        ax.text(0.5, i, config, ha="center", va="center", fontsize=12, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(wells)
    ax.set_xlabel("")
    ax.set_title("Recommended Jet Pump Configurations", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_xticks([])

    plt.tight_layout()
    return fig


def create_watercut_comparison(results: list["OptimizationResult"]):
    """Create chart comparing formation vs total watercut

    Args:
        results: List of optimization results

    Returns:
        matplotlib figure
    """
    wells = [r.well_name for r in results]
    total_watercuts = [r.total_watercut * 100 for r in results]

    # Calculate formation watercut
    form_wc = [
        (
            (r.predicted_formation_water / (r.predicted_oil_rate + r.predicted_formation_water) * 100)
            if (r.predicted_oil_rate + r.predicted_formation_water) > 0
            else 0
        )
        for r in results
    ]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(wells))
    width = 0.35

    ax.bar(x - width / 2, form_wc, width, label="Formation Watercut", alpha=0.8)
    ax.bar(x + width / 2, total_watercuts, width, label="Total Watercut", alpha=0.8)

    ax.set_xlabel("Well", fontsize=12)
    ax.set_ylabel("Watercut (%)", fontsize=12)
    ax.set_title("Watercut Comparison by Well", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(wells, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


def create_efficiency_scatter(results: list["OptimizationResult"]):
    """Create scatter plot of oil rate vs power fluid

    Args:
        results: List of optimization results

    Returns:
        matplotlib figure
    """
    pf_rates = [r.allocated_power_fluid for r in results]
    oil_rates = [r.predicted_oil_rate for r in results]
    wells = [r.well_name for r in results]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create scatter plot
    scatter = ax.scatter(pf_rates, oil_rates, s=200, alpha=0.6, edgecolors="black", linewidths=1.5)

    # Add well labels
    for i, well in enumerate(wells):
        ax.annotate(well, (pf_rates[i], oil_rates[i]), xytext=(5, 5), textcoords="offset points", fontsize=9)

    # Add efficiency lines (oil/pf ratio)
    max_pf = max(pf_rates) if pf_rates else 1
    max_oil = max(oil_rates) if oil_rates else 1

    efficiency_ratios = [0.05, 0.10, 0.15, 0.20]
    for ratio in efficiency_ratios:
        x_line = np.array([0, max_pf])
        y_line = x_line * ratio
        ax.plot(x_line, y_line, ":", color="gray", alpha=0.5, linewidth=1)
        # Label the line
        if max_pf * ratio < max_oil:
            ax.text(max_pf * 0.95, max_pf * ratio * 0.95, f"{ratio:.2f}", fontsize=8, color="gray", alpha=0.7)

    ax.set_xlabel("Power Fluid Rate (BWPD)", fontsize=12)
    ax.set_ylabel("Oil Rate (BOPD)", fontsize=12)
    ax.set_title("Oil Production vs Power Fluid Allocation", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_marginal_rate_chart(results: list["OptimizationResult"]):
    """Create bar chart of marginal oil rates

    Args:
        results: List of optimization results

    Returns:
        matplotlib figure
    """
    # Sort by marginal rate
    sorted_results = sorted(results, key=lambda r: r.marginal_oil_rate, reverse=True)

    wells = [r.well_name for r in sorted_results]
    marginal_rates = [r.marginal_oil_rate for r in sorted_results]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create bar chart with color gradient
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(wells)))
    bars = ax.bar(range(len(wells)), marginal_rates, color=colors, alpha=0.8)

    ax.set_xlabel("Well", fontsize=12)
    ax.set_ylabel("Marginal Oil Rate (bbl oil / bbl PF)", fontsize=12)
    ax.set_title("Marginal Oil Production Efficiency by Well", fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(wells)))
    ax.set_xticklabels(wells, rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, marginal_rates)):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    return fig
