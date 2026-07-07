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
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _pump_label_at_date(
    jp_history: Optional[pd.DataFrame],
    well_name: str,
    test_date,
) -> Optional[str]:
    """Pump installed on/before ``test_date`` formatted as e.g. ``"13C"``.

    Used by the IPR chart hover text so the user can see which pump was in
    the well at each test point. Returns None when no JP history is provided
    or no install record qualifies — caller falls through to a pump-less
    hover line.
    """
    if jp_history is None or test_date is None or pd.isna(test_date):
        return None
    well_df = jp_history[jp_history["Well Name"] == well_name].copy()
    if well_df.empty:
        return None
    well_df = well_df.dropna(subset=["Date Set"])
    well_df = well_df[well_df["Date Set"] <= pd.Timestamp(test_date)]
    if well_df.empty:
        return None
    latest = well_df.sort_values("Date Set", ascending=False).iloc[0]
    nozzle = latest.get("Nozzle Number")
    throat = latest.get("Throat Ratio")
    if not (pd.notna(nozzle) and pd.notna(throat)):
        return None
    try:
        return f"{int(nozzle)}{str(throat).strip()}"
    except (TypeError, ValueError):
        return None


def create_ipr_plotly(
    well_name: str,
    ipr_data: Dict,
    merged_data: pd.DataFrame,
    form_wc: Optional[float] = None,
    jp_history: Optional[pd.DataFrame] = None,
    show_jp_labels: bool = False,
) -> "go.Figure":
    """Create an interactive Plotly IPR plot for a single well.

    Args:
        well_name: Well identifier
        ipr_data: IPR data dict for this well
        merged_data: Merged test+BHP DataFrame for scatter points
        form_wc: Formation water cut (0-1) for calculating oil rate from total fluid

    Returns:
        plotly Figure (zoomable, hoverable)
    """
    fig = go.Figure()

    # Vogel IPR curve
    fluid_recent = ipr_data["fluid_recent"]
    if form_wc is not None:
        oil_rates = [f * (1 - form_wc) for f in fluid_recent]
        fig.add_trace(
            go.Scatter(
                x=fluid_recent,
                y=list(ipr_data["bhp_array"]),
                mode="lines",
                name="Vogel IPR",
                line=dict(color="blue", width=3),
                customdata=oil_rates,
                hovertemplate="Fluid: %{x:.0f} BPD<br>Oil: %{customdata:.0f} BOPD<br>BHP: %{y:.0f} psi<extra></extra>",
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=fluid_recent,
                y=list(ipr_data["bhp_array"]),
                mode="lines",
                name="Vogel IPR",
                line=dict(color="blue", width=3),
                hovertemplate="Fluid: %{x:.0f} BPD<br>BHP: %{y:.0f} psi<extra></extra>",
            )
        )

    # Test data scatter points — callers may pass an empty DataFrame for
    # wells without any tests in the lookback window (e.g. infrequently-
    # tested wells on the single-well page's 3-month cache). Guard the
    # 'well' column access so the IPR curve still renders without scatter.
    if not merged_data.empty and "well" in merged_data.columns:
        well_test_data = merged_data[merged_data["well"] == well_name].copy()
    else:
        well_test_data = pd.DataFrame()
    if (
        not well_test_data.empty
        and "BHP" in well_test_data.columns
        and "WtTotalFluid" in well_test_data.columns
    ):
        well_test_data = well_test_data.dropna(subset=["BHP", "WtTotalFluid"])
        if not well_test_data.empty:
            # Calculate days since test for color
            if "WtDate" in well_test_data.columns:
                well_test_data["date"] = pd.to_datetime(well_test_data["WtDate"])
                current_date = pd.to_datetime("today")
                well_test_data["days_since"] = (
                    current_date - well_test_data["date"]
                ).dt.days
                hover_text = []
                pump_labels: list[str] = []
                for _, row in well_test_data.iterrows():
                    oil_str = ""
                    if "WtOilVol" in well_test_data.columns and pd.notna(
                        row.get("WtOilVol")
                    ):
                        oil_str = f"Oil: {row['WtOilVol']:.0f} BOPD<br>"
                    pump_label = _pump_label_at_date(
                        jp_history, well_name, row.get("date")
                    )
                    pump_labels.append(pump_label or "")
                    pump_str = f"Pump: {pump_label}<br>" if pump_label else ""
                    hover_text.append(
                        f"Fluid: {row['WtTotalFluid']:.0f} BPD<br>"
                        f"{oil_str}"
                        f"BHP: {row['BHP']:.0f} psi<br>"
                        f"{pump_str}"
                        f"Date: {row['date'].strftime('%Y-%m-%d')}<br>"
                        f"Days ago: {row['days_since']}"
                    )

                # When the user opts in, render the JP label INSIDE the marker
                # (mode="markers+text", text at middle-center, marker enlarged
                # to hold ~3 chars). Otherwise stick with plain colored dots.
                if show_jp_labels and any(pump_labels):
                    fig.add_trace(
                        go.Scatter(
                            x=well_test_data["WtTotalFluid"],
                            y=well_test_data["BHP"],
                            mode="markers+text",
                            name="Test Data",
                            text=pump_labels,
                            textposition="middle center",
                            textfont=dict(size=10, color="white"),
                            marker=dict(
                                size=28,
                                color=well_test_data["days_since"],
                                colorscale="Viridis",
                                showscale=True,
                                colorbar=dict(
                                    title=dict(text="Days Ago", side="right"),
                                    thickness=15,
                                ),
                                line=dict(width=1, color="black"),
                            ),
                            customdata=hover_text,
                            hovertemplate="%{customdata}<extra></extra>",
                        )
                    )
                else:
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
                                colorbar=dict(
                                    # side="right" rotates the title 90° so it
                                    # runs vertically alongside the colorbar
                                    # instead of stacking on top and colliding
                                    # with the y-axis "Bottom Hole Pressure" label.
                                    title=dict(text="Days Ago", side="right"),
                                    thickness=15,
                                ),
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
                        marker=dict(
                            size=10, color="red", line=dict(width=1, color="black")
                        ),
                        hovertemplate="Fluid: %{x:.0f} BPD<br>BHP: %{y:.0f} psi<extra></extra>",
                    )
                )

    res_p = ipr_data["res_pres"]
    qmax = ipr_data["qmax_recent"]

    fig.update_layout(
        title=dict(text=f"<b>{well_name}</b> — Vogel IPR", font=dict(size=16)),
        xaxis_title="Total Fluid Rate (BPD)",
        yaxis_title="Bottom Hole Pressure (psi)",
        # rangemode="tozero" anchors the axis at 0 and autoscales the upper
        # bound; range=[0, None] silently falls back to full autorange (so the
        # axis did NOT start at 0 as intended — documented Plotly quirk).
        xaxis=dict(rangemode="tozero", gridcolor="lightgray"),
        yaxis=dict(rangemode="tozero", gridcolor="lightgray"),
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

            # Test data scatter — guard against an empty/no-column df
            if not df.empty and "well" in df.columns:
                well_test_data = df[df["well"] == well].copy()
            else:
                well_test_data = pd.DataFrame()
            if (
                not well_test_data.empty
                and "BHP" in well_test_data.columns
                and "WtTotalFluid" in well_test_data.columns
            ):
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
                bbox=dict(
                    boxstyle="round,pad=0.4",
                    facecolor="lightyellow",
                    alpha=0.9,
                    edgecolor="gray",
                ),
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

        # Test data scatter — guard against an empty/no-column df
        if not df.empty and "well" in df.columns:
            well_test_data = df[df["well"] == well].copy()
        else:
            well_test_data = pd.DataFrame()
        if (
            not well_test_data.empty
            and "BHP" in well_test_data.columns
            and "WtTotalFluid" in well_test_data.columns
        ):
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
