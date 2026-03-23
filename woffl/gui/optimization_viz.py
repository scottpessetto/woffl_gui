"""Visualization Utilities for Multi-Well Optimization

Functions for creating charts and plots for optimization results.
"""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from woffl.assembly.calibration import CalibrationResult
    from woffl.assembly.network_optimizer import NetworkOptimizer, OptimizationResult


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
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 10},
    )

    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")

    ax.axis("equal")
    plt.title("Power Fluid Allocation by Well", fontsize=14, fontweight="bold", pad=20)

    return fig


def create_oil_rate_bar_chart(
    results: list["OptimizationResult"], show_baseline=False, baseline_dict=None
):
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
        ax.text(
            0.5, i, config, ha="center", va="center", fontsize=12, fontweight="bold"
        )

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
            (
                r.predicted_formation_water
                / (r.predicted_oil_rate + r.predicted_formation_water)
                * 100
            )
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
    scatter = ax.scatter(
        pf_rates, oil_rates, s=200, alpha=0.6, edgecolors="black", linewidths=1.5
    )

    # Add well labels
    for i, well in enumerate(wells):
        ax.annotate(
            well,
            (pf_rates[i], oil_rates[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

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
            ax.text(
                max_pf * 0.95,
                max_pf * ratio * 0.95,
                f"{ratio:.2f}",
                fontsize=8,
                color="gray",
                alpha=0.7,
            )

    ax.set_xlabel("Power Fluid Rate (BWPD)", fontsize=12)
    ax.set_ylabel("Oil Rate (BOPD)", fontsize=12)
    ax.set_title(
        "Oil Production vs Power Fluid Allocation", fontsize=14, fontweight="bold"
    )
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
    ax.set_title(
        "Marginal Oil Production Efficiency by Well", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(range(len(wells)))
    ax.set_xticklabels(wells, rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, marginal_rates)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    return fig


def create_calibration_chart(calibration_results: dict[str, "CalibrationResult"]):
    """Create horizontal bar chart of calibration factors per well.

    Args:
        calibration_results: Dict mapping well_name to CalibrationResult

    Returns:
        matplotlib figure
    """
    if not calibration_results:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No calibration data", ha="center", va="center", fontsize=14)
        return fig

    sorted_items = sorted(
        calibration_results.items(), key=lambda x: x[1].calibration_factor
    )
    wells = [name for name, _ in sorted_items]
    factors = [c.calibration_factor for _, c in sorted_items]
    errors = [c.oil_error_pct for _, c in sorted_items]
    grades = [c.quality_grade for _, c in sorted_items]

    colors = []
    for g in grades:
        if g == "good":
            colors.append("#2ca02c")
        elif g == "fair":
            colors.append("#ff7f0e")
        else:
            colors.append("#d62728")

    fig, ax = plt.subplots(figsize=(10, max(4, len(wells) * 0.5)))
    y_pos = np.arange(len(wells))
    bars = ax.barh(
        y_pos, factors, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5
    )

    # Vertical reference line at 1.0
    ax.axvline(x=1.0, color="black", linestyle="--", linewidth=1.5, alpha=0.7)

    # Bar labels
    for i, (bar, factor, error) in enumerate(zip(bars, factors, errors)):
        label = f"{factor:.2f} ({error:.0f}%)"
        x_pos = bar.get_width() + 0.02
        ax.text(
            x_pos, bar.get_y() + bar.get_height() / 2, label, va="center", fontsize=9
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(wells)
    ax.set_xlabel("Calibration Factor", fontsize=12)
    ax.set_title(
        "Model Calibration Factors (Actual / Model)", fontsize=14, fontweight="bold"
    )
    ax.set_xlim(0, max(factors) * 1.3 if factors else 2.5)
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    return fig


def create_ipr_comparison_pdf(
    results: list["OptimizationResult"],
    optimizer: "NetworkOptimizer",
    actual_oil_map: dict[str, float],
    actual_pf_map: dict[str, float],
    actual_bhp_map: dict[str, float],
    current_jp_map: dict[str, str],
    calibration: dict[str, "CalibrationResult"] | None = None,
) -> bytes:
    """Create a multi-page PDF with IPR comparison charts for each optimized well.

    Each page contains a Vogel IPR curve with current (red) and proposed (green)
    operating points, plus a text summary of both configurations.

    Args:
        results: List of OptimizationResult objects
        optimizer: NetworkOptimizer instance (provides WellConfig data)
        actual_oil_map: Dict mapping well_name to actual oil rate (BOPD)
        actual_pf_map: Dict mapping well_name to actual power fluid rate (BWPD)
        actual_bhp_map: Dict mapping well_name to measured BHP (psi)
        current_jp_map: Dict mapping well_name to current JP string (e.g. "12B")

    Returns:
        PDF file contents as bytes
    """
    import io
    import math

    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        for r in results:
            well_config = optimizer.get_well_by_name(r.well_name)
            if well_config is None:
                continue

            pres = well_config.res_pres
            qwf = well_config.qwf
            pwf = well_config.pwf

            # Vogel IPR curve
            qmax = qwf / (1 - 0.2 * (pwf / pres) - 0.8 * (pwf / pres) ** 2)
            pressures = np.linspace(0, pres, 200)
            oil_rates = qmax * (
                1 - 0.2 * (pressures / pres) - 0.8 * (pressures / pres) ** 2
            )

            # Optimized operating point
            opt_bhp = r.suction_pressure
            opt_oil = r.predicted_oil_rate

            # Current operating point
            actual_oil = actual_oil_map.get(r.well_name)
            actual_pf = actual_pf_map.get(r.well_name)
            current_jp = current_jp_map.get(r.well_name, "N/A")

            # Use measured BHP if available, otherwise inverse Vogel
            current_bhp = actual_bhp_map.get(r.well_name)
            if current_bhp is None and actual_oil is not None and 0 < actual_oil < qmax:
                disc = 0.04 + 3.2 * (1 - actual_oil / qmax)
                if disc >= 0:
                    current_bhp = (-0.2 + math.sqrt(disc)) / 1.6 * pres

            # --- Create figure (letter size) ---
            fig = plt.figure(figsize=(8.5, 11))

            # IPR chart — upper portion
            ax = fig.add_axes([0.12, 0.40, 0.80, 0.50])
            ax.plot(
                oil_rates, pressures, "b-", linewidth=2.5, label="Vogel IPR", zorder=2
            )

            # Current operating point
            if actual_oil is not None and current_bhp is not None:
                ax.plot(
                    actual_oil,
                    current_bhp,
                    "o",
                    color="#d62728",
                    markersize=14,
                    markeredgecolor="black",
                    markeredgewidth=1.5,
                    label=f"Current ({current_jp})",
                    zorder=5,
                )

            # Proposed operating point
            ax.plot(
                opt_oil,
                opt_bhp,
                "o",
                color="#2ca02c",
                markersize=14,
                markeredgecolor="black",
                markeredgewidth=1.5,
                label=f"Proposed ({r.recommended_nozzle}{r.recommended_throat})",
                zorder=5,
            )

            # Dashed arrow from current to proposed
            if actual_oil is not None and current_bhp is not None:
                ax.annotate(
                    "",
                    xy=(opt_oil, opt_bhp),
                    xytext=(actual_oil, current_bhp),
                    arrowprops=dict(
                        arrowstyle="->", color="gray", lw=1.5, linestyle="--"
                    ),
                    zorder=3,
                )

            ax.set_xlabel("Oil Rate (BOPD)", fontsize=12)
            ax.set_ylabel("Bottomhole Pressure (psi)", fontsize=12)
            ax.set_title(
                f"{r.well_name} — IPR Comparison",
                fontsize=15,
                fontweight="bold",
                pad=12,
            )
            ax.legend(fontsize=10, loc="upper right")
            ax.grid(True, alpha=0.3)
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0, top=pres * 1.05)

            # --- Separator line ---
            sep = Line2D(
                [0.08, 0.92],
                [0.37, 0.37],
                transform=fig.transFigure,
                color="gray",
                linewidth=0.5,
            )
            fig.add_artist(sep)

            # --- Text: Current (left column) ---
            fig.text(
                0.12, 0.33, "CURRENT", fontsize=13, fontweight="bold", color="#d62728"
            )
            cy = 0.30
            for line in [
                f"Jet Pump:    {current_jp}",
                (
                    f"Oil Rate:    {actual_oil:.0f} BOPD"
                    if actual_oil is not None
                    else "Oil Rate:    N/A"
                ),
                (
                    f"PF Rate:     {actual_pf:.0f} BWPD"
                    if actual_pf is not None
                    else "PF Rate:     N/A"
                ),
                (
                    f"Suction P:   {current_bhp:.0f} psi"
                    if current_bhp is not None
                    else "Suction P:   N/A"
                ),
            ]:
                fig.text(0.12, cy, line, fontsize=10, family="monospace")
                cy -= 0.025

            # --- Text: Proposed (right column) ---
            fig.text(
                0.55, 0.33, "PROPOSED", fontsize=13, fontweight="bold", color="#2ca02c"
            )
            py = 0.30
            proposed_lines = [
                f"Jet Pump:    {r.recommended_nozzle}{r.recommended_throat}",
                f"Oil Rate:    {opt_oil:.0f} BOPD",
                f"PF Rate:     {r.allocated_power_fluid:.0f} BWPD",
                f"Suction P:   {r.suction_pressure:.0f} psi",
            ]
            # Add calibration info if available
            if calibration and r.well_name in calibration:
                cal = calibration[r.well_name]
                cal_oil = opt_oil * cal.calibration_factor
                proposed_lines.append(f"Calibrated:  {cal_oil:.0f} BOPD")
                proposed_lines.append(f"Cal Factor:  {cal.calibration_factor:.2f}")
            for line in proposed_lines:
                fig.text(0.55, py, line, fontsize=10, family="monospace")
                py -= 0.025

            # --- Text: Delta (below both columns) ---
            if actual_oil is not None:
                delta_oil = opt_oil - actual_oil
                fig.text(0.12, 0.16, "CHANGE", fontsize=13, fontweight="bold")
                dy = 0.13
                fig.text(
                    0.12,
                    dy,
                    f"Oil:  {delta_oil:+.0f} BOPD",
                    fontsize=10,
                    family="monospace",
                )
                if actual_pf is not None:
                    delta_pf = r.allocated_power_fluid - actual_pf
                    dy -= 0.025
                    fig.text(
                        0.12,
                        dy,
                        f"PF:   {delta_pf:+.0f} BWPD",
                        fontsize=10,
                        family="monospace",
                    )

            # --- Footer with well metadata ---
            fig.text(
                0.12,
                0.04,
                f"Res Pres: {pres:.0f} psi  |  Form WC: {well_config.form_wc:.0%}  |  "
                f"GOR: {well_config.form_gor:.0f} scf/bbl  |  JP TVD: {well_config.jpump_tvd:.0f} ft",
                fontsize=8,
                color="gray",
            )

            pdf.savefig(fig)
            plt.close(fig)

    buf.seek(0)
    return buf.getvalue()
