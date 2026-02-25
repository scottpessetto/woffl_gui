"""IPR Visualization

Interactive Plotly charts for Vogel IPR curves (zoomable, hoverable)
plus matplotlib functions for PDF export of all IPR curves.
"""

import io
import math
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def create_ipr_plotly(
    well_name: str,
    ipr_data: Dict,
    merged_data: pd.DataFrame,
) -> "go.Figure":
    """Create an interactive Plotly IPR plot for a single well.

    Args:
        well_name: Well identifier
        ipr_data: IPR data dict for this well
        merged_data: Merged test+BHP DataFrame for scatter points

    Returns:
        plotly Figure (zoomable, hoverable)
    """
    fig = go.Figure()

    # Vogel IPR curve
    fig.add_trace(
        go.Scatter(
            x=ipr_data["fluid_recent"],
            y=list(ipr_data["bhp_array"]),
            mode="lines",
            name="Vogel IPR",
            line=dict(color="blue", width=3),
            hovertemplate="Fluid: %{x:.0f} BPD<br>BHP: %{y:.0f} psi<extra></extra>",
        )
    )

    # Test data scatter points
    well_test_data = merged_data[merged_data["well"] == well_name].copy()
    if not well_test_data.empty and "BHP" in well_test_data.columns and "WtTotalFluid" in well_test_data.columns:
        well_test_data = well_test_data.dropna(subset=["BHP", "WtTotalFluid"])
        if not well_test_data.empty:
            # Calculate days since test for color
            if "WtDate" in well_test_data.columns:
                well_test_data["date"] = pd.to_datetime(well_test_data["WtDate"])
                current_date = pd.to_datetime("today")
                well_test_data["days_since"] = (current_date - well_test_data["date"]).dt.days
                hover_text = [
                    f"Fluid: {row['WtTotalFluid']:.0f} BPD<br>"
                    f"BHP: {row['BHP']:.0f} psi<br>"
                    f"Date: {row['date'].strftime('%Y-%m-%d')}<br>"
                    f"Days ago: {row['days_since']}"
                    for _, row in well_test_data.iterrows()
                ]
                fig.add_trace(
                    go.Scatter(
                        x=well_test_data["WtTotalFluid"],
                        y=well_test_data["BHP"],
                        mode="markers",
                        name="Test Data",
                        marker=dict(
                            size=10,
                            color=well_test_data["days_since"],
                            colorscale="Viridis",
                            showscale=True,
                            colorbar=dict(title="Days Ago", thickness=15),
                            line=dict(width=1, color="black"),
                        ),
                        text=hover_text,
                        hoverinfo="text",
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=well_test_data["WtTotalFluid"],
                        y=well_test_data["BHP"],
                        mode="markers",
                        name="Test Data",
                        marker=dict(size=10, color="red", line=dict(width=1, color="black")),
                        hovertemplate="Fluid: %{x:.0f} BPD<br>BHP: %{y:.0f} psi<extra></extra>",
                    )
                )

    res_p = ipr_data["res_pres"]
    qmax = ipr_data["qmax_recent"]

    fig.update_layout(
        title=dict(text=f"<b>{well_name}</b> — Vogel IPR", font=dict(size=16)),
        xaxis_title="Total Fluid Rate (BPD)",
        yaxis_title="Bottom Hole Pressure (psi)",
        xaxis=dict(range=[0, None], gridcolor="lightgray"),
        yaxis=dict(range=[0, None], gridcolor="lightgray"),
        plot_bgcolor="white",
        hovermode="closest",
        height=500,
        annotations=[
            dict(
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                text=f"Res P: {res_p:.0f} psi<br>Qmax: {qmax:.0f} BPD",
                showarrow=False,
                font=dict(size=12),
                bgcolor="lightyellow",
                bordercolor="gray",
                borderwidth=1,
                borderpad=4,
                align="left",
            )
        ],
    )

    return fig


def create_ipr_grid_plotly(
    ipr_data: Dict[str, Dict],
    merged_data: pd.DataFrame,
) -> "go.Figure":
    """Create an interactive Plotly grid of IPR plots for all wells.

    Args:
        ipr_data: Dictionary from ipr_analyzer.generate_ipr_curves()
        merged_data: Merged test+BHP DataFrame for scatter points

    Returns:
        plotly Figure with subplots (zoomable, hoverable)
    """
    wells = sorted(ipr_data.keys())
    num_wells = len(wells)

    if num_wells == 0:
        fig = go.Figure()
        fig.add_annotation(text="No IPR data available", x=0.5, y=0.5, showarrow=False, font=dict(size=20))
        return fig

    # Scale columns based on well count to keep grid manageable
    if num_wells <= 6:
        num_columns = min(3, num_wells)
    elif num_wells <= 16:
        num_columns = 4
    else:
        num_columns = 5

    num_rows = max(1, math.ceil(num_wells / num_columns))

    # Plotly requires spacing < 1/(rows-1); calculate safe values
    if num_rows > 1:
        max_v_spacing = 1.0 / (num_rows - 1) - 0.001
        vertical_spacing = min(0.06, max_v_spacing)
    else:
        vertical_spacing = 0.06

    if num_columns > 1:
        max_h_spacing = 1.0 / (num_columns - 1) - 0.001
        horizontal_spacing = min(0.05, max_h_spacing)
    else:
        horizontal_spacing = 0.05

    fig = make_subplots(
        rows=num_rows,
        cols=num_columns,
        subplot_titles=[f"<b>{w}</b>" for w in wells],
        vertical_spacing=vertical_spacing,
        horizontal_spacing=horizontal_spacing,
    )

    # Prepare merged data with days_since
    df = merged_data.copy()
    if "WtDate" in df.columns:
        df["date"] = pd.to_datetime(df["WtDate"])
        current_date = pd.to_datetime("today")
        df["days_since"] = (current_date - df["date"]).dt.days

    for index, well in enumerate(wells):
        row = index // num_columns + 1
        col = index % num_columns + 1
        well_ipr = ipr_data[well]

        # Vogel IPR curve
        fig.add_trace(
            go.Scatter(
                x=well_ipr["fluid_recent"],
                y=list(well_ipr["bhp_array"]),
                mode="lines",
                name=f"{well} IPR",
                line=dict(color="blue", width=2),
                showlegend=False,
                hovertemplate=f"{well}<br>Fluid: %{{x:.0f}} BPD<br>BHP: %{{y:.0f}} psi<extra></extra>",
            ),
            row=row,
            col=col,
        )

        # Test data scatter
        well_test_data = df[df["well"] == well].copy()
        if not well_test_data.empty and "BHP" in well_test_data.columns and "WtTotalFluid" in well_test_data.columns:
            well_test_data = well_test_data.dropna(subset=["BHP", "WtTotalFluid"])
            if not well_test_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=well_test_data["WtTotalFluid"],
                        y=well_test_data["BHP"],
                        mode="markers",
                        name=f"{well} Tests",
                        showlegend=False,
                        marker=dict(
                            size=7,
                            color=well_test_data.get("days_since", "red"),
                            colorscale="Viridis" if "days_since" in well_test_data.columns else None,
                            line=dict(width=0.5, color="black"),
                        ),
                        hovertemplate=f"{well}<br>Fluid: %{{x:.0f}} BPD<br>BHP: %{{y:.0f}} psi<extra></extra>",
                    ),
                    row=row,
                    col=col,
                )

        # Add annotation with RP and Qmax
        res_p = well_ipr["res_pres"]
        qmax = well_ipr["qmax_recent"]
        fig.add_annotation(
            text=f"RP:{res_p:.0f} Qmax:{qmax:.0f}",
            x=0.02,
            y=0.95,
            xref=f"x{index + 1} domain" if index > 0 else "x domain",
            yref=f"y{index + 1} domain" if index > 0 else "y domain",
            showarrow=False,
            font=dict(size=9),
            bgcolor="lightyellow",
            bordercolor="gray",
            borderwidth=1,
        )

    fig.update_layout(
        height=max(400, num_rows * 350),
        plot_bgcolor="white",
        title_text="Vogel IPR Curves — All Wells",
        title_font_size=18,
    )

    # Set axis ranges
    for i in range(1, num_wells + 1):
        fig.update_xaxes(range=[0, None], row=(i - 1) // num_columns + 1, col=(i - 1) % num_columns + 1)
        fig.update_yaxes(range=[0, None], row=(i - 1) // num_columns + 1, col=(i - 1) % num_columns + 1)

    return fig


def create_ipr_pdf(
    ipr_data: Dict[str, Dict],
    merged_data: pd.DataFrame,
) -> bytes:
    """Generate a multi-page PDF with one IPR plot per page.

    Each page has a full-size, high-resolution IPR plot for one well.

    Args:
        ipr_data: Dictionary from ipr_analyzer.generate_ipr_curves()
        merged_data: Merged test+BHP DataFrame for scatter points

    Returns:
        PDF file as bytes
    """
    from matplotlib.backends.backend_pdf import PdfPages

    wells = sorted(ipr_data.keys())
    pdf_buffer = io.BytesIO()

    # Prepare merged data
    df = merged_data.copy()
    if "WtDate" in df.columns:
        df["date"] = pd.to_datetime(df["WtDate"])
        current_date = pd.to_datetime("today")
        df["days_since"] = (current_date - df["date"]).dt.days

    with PdfPages(pdf_buffer) as pdf:
        for well in wells:
            well_ipr = ipr_data[well]
            fig, ax = plt.subplots(1, 1, figsize=(11, 8.5))  # Letter landscape

            # Vogel IPR curve
            ax.plot(
                well_ipr["fluid_recent"],
                well_ipr["bhp_array"],
                color="blue",
                linewidth=3,
                label="Vogel IPR",
            )

            # Test data scatter
            well_test_data = df[df["well"] == well].copy()
            if not well_test_data.empty and "BHP" in well_test_data.columns:
                well_test_data = well_test_data.dropna(subset=["BHP", "WtTotalFluid"])
                if not well_test_data.empty:
                    if "days_since" in well_test_data.columns:
                        scatter = ax.scatter(
                            well_test_data["WtTotalFluid"],
                            well_test_data["BHP"],
                            c=well_test_data["days_since"],
                            alpha=0.7,
                            cmap="viridis",
                            edgecolors="black",
                            linewidth=0.5,
                            s=80,
                            zorder=5,
                            label="Test Data",
                        )
                        cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
                        cbar.set_label("Days Since Test", fontsize=11)
                    else:
                        ax.scatter(
                            well_test_data["WtTotalFluid"],
                            well_test_data["BHP"],
                            alpha=0.7,
                            color="red",
                            edgecolors="black",
                            linewidth=0.5,
                            s=80,
                            zorder=5,
                            label="Test Data",
                        )

            # Annotations
            res_p = well_ipr["res_pres"]
            qmax = well_ipr["qmax_recent"]
            ax.text(
                0.03,
                0.95,
                f"Reservoir Pressure: {res_p:.0f} psi\nQmax: {qmax:.0f} BPD",
                transform=ax.transAxes,
                fontsize=13,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9, edgecolor="gray"),
            )

            ax.set_xlabel("Total Fluid Rate (BPD)", fontsize=14)
            ax.set_ylabel("Bottom Hole Pressure (psi)", fontsize=14)
            ax.set_title(f"Vogel IPR — {well}", fontsize=16, fontweight="bold")
            ax.set_xlim(left=0)
            ax.set_ylim(bottom=0)
            ax.legend(fontsize=12, loc="upper right")
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=11)

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()


def create_ipr_grid_png(
    ipr_data: Dict[str, Dict],
    merged_data: pd.DataFrame,
    dpi: int = 200,
) -> bytes:
    """Generate a high-resolution PNG grid of all IPR plots.

    Creates a large matplotlib figure with all wells in a grid layout,
    rendered at high DPI so users can download and zoom in.

    Args:
        ipr_data: Dictionary from ipr_analyzer.generate_ipr_curves()
        merged_data: Merged test+BHP DataFrame for scatter points
        dpi: Resolution in dots per inch (default 200 for high-res)

    Returns:
        PNG image as bytes
    """
    wells = sorted(ipr_data.keys())
    num_wells = len(wells)

    if num_wells == 0:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, "No IPR data", ha="center", va="center", fontsize=14)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    num_columns = min(5, num_wells)
    num_rows = max(1, math.ceil(num_wells / num_columns))

    # Each subplot is 5x4 inches — large enough to read when zoomed
    fig_width = num_columns * 5
    fig_height = num_rows * 4
    fig, axs = plt.subplots(
        num_rows,
        num_columns,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )

    # Prepare merged data
    df = merged_data.copy()
    if "WtDate" in df.columns:
        df["date"] = pd.to_datetime(df["WtDate"])
        current_date = pd.to_datetime("today")
        df["days_since"] = (current_date - df["date"]).dt.days

    for index, well in enumerate(wells):
        row_idx = index // num_columns
        col_idx = index % num_columns
        ax = axs[row_idx][col_idx]
        well_ipr = ipr_data[well]

        # Vogel IPR curve
        ax.plot(
            well_ipr["fluid_recent"],
            well_ipr["bhp_array"],
            color="blue",
            linewidth=2,
            label="Vogel IPR",
        )

        # Test data scatter
        well_test_data = df[df["well"] == well].copy()
        if not well_test_data.empty and "BHP" in well_test_data.columns:
            well_test_data = well_test_data.dropna(subset=["BHP", "WtTotalFluid"])
            if not well_test_data.empty:
                if "days_since" in well_test_data.columns:
                    ax.scatter(
                        well_test_data["WtTotalFluid"],
                        well_test_data["BHP"],
                        c=well_test_data["days_since"],
                        alpha=0.7,
                        cmap="viridis",
                        edgecolors="black",
                        linewidth=0.3,
                        s=30,
                        zorder=5,
                    )
                else:
                    ax.scatter(
                        well_test_data["WtTotalFluid"],
                        well_test_data["BHP"],
                        alpha=0.7,
                        color="red",
                        edgecolors="black",
                        linewidth=0.3,
                        s=30,
                        zorder=5,
                    )

        res_p = well_ipr["res_pres"]
        qmax = well_ipr["qmax_recent"]
        ax.text(
            0.03,
            0.95,
            f"RP: {res_p:.0f}\nQmax: {qmax:.0f}",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.8),
        )

        ax.set_title(well, fontsize=10, fontweight="bold")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    # Hide unused subplots
    for i in range(num_wells, num_rows * num_columns):
        row_idx = i // num_columns
        col_idx = i % num_columns
        axs[row_idx][col_idx].set_visible(False)

    # Add shared axis labels
    fig.supxlabel("Total Fluid Rate (BPD)", fontsize=12)
    fig.supylabel("Bottom Hole Pressure (psi)", fontsize=12)
    fig.suptitle("Vogel IPR Curves — All Wells", fontsize=16, fontweight="bold", y=1.0)

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def create_rp_comparison_chart(vogel_coeffs: pd.DataFrame) -> plt.Figure:
    """Create a bar chart comparing estimated reservoir pressures across wells.

    Args:
        vogel_coeffs: DataFrame from compute_vogel_coefficients()

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(max(10, len(vogel_coeffs) * 0.6), 5))

    wells = vogel_coeffs["Well"].values
    res_pressures = vogel_coeffs["ResP"].values

    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(wells)))
    bars = ax.bar(range(len(wells)), res_pressures, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_xticks(range(len(wells)))
    ax.set_xticklabels(wells, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Estimated Reservoir Pressure (psi)", fontsize=11)
    ax.set_title("Reservoir Pressure Estimates by Well", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    for bar, val in zip(bars, res_pressures):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 20,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    return fig


def create_qmax_comparison_chart(vogel_coeffs: pd.DataFrame) -> plt.Figure:
    """Create a bar chart comparing Qmax values across wells.

    Args:
        vogel_coeffs: DataFrame from compute_vogel_coefficients()

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(max(10, len(vogel_coeffs) * 0.6), 5))

    wells = vogel_coeffs["Well"].values
    qmax_values = vogel_coeffs["QMax_recent"].values

    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(wells)))
    bars = ax.bar(range(len(wells)), qmax_values, color=colors, edgecolor="black", linewidth=0.5)

    ax.set_xticks(range(len(wells)))
    ax.set_xticklabels(wells, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Qmax (BPD)", fontsize=11)
    ax.set_title("Vogel Qmax by Well (Most Recent Test)", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    for bar, val in zip(bars, qmax_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 20,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    return fig
