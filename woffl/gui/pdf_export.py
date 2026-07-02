"""PDF report export for the single-well analysis page.

Public entry point: :func:`generate_report`. Pulls the same inputs the
on-screen Solver and Batch Run views consume, runs the solver + batch sweep,
renders the IPR + batch performance figures to PNG via matplotlib (kaleido-free,
so it works in headless Databricks Apps), and assembles a multi-page PDF via ReportLab.

The figure-building helpers here intentionally duplicate a small amount of
logic from ``tabs/jetpump_solver.py`` and ``tabs/batch_run.py`` rather than
refactoring those modules — those renderers are heavily intertwined with
``st.*`` side effects and the duplication is preferable to a risky surgery
on production rendering paths.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional
from xml.sax.saxutils import escape as xml_escape

# Matplotlib uses the headless 'Agg' backend so we can render PNGs in a
# server-side Streamlit process (or a Databricks Apps container) without
# a display. Setting the backend must happen before pyplot is imported.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from woffl.assembly.batchpump import exp_model  # noqa: E402
from woffl.gui.params import SimulationParams  # noqa: E402
from woffl.gui.utils import (  # noqa: E402
    is_valid_number,
    recommend_jetpump,
    run_batch_pump,
    run_jetpump_solver,
)


# ---------------------------------------------------------------------------
# Figure rendering via matplotlib
# ---------------------------------------------------------------------------

# DPI for PNG rendering. 180 gives crisp text on Letter at 7" wide without
# blowing up the PDF file size. Higher (250+) is overkill for screen + print.
_FIG_DPI = 180


def _fig_to_png(fig: plt.Figure) -> bytes:
    """Render a matplotlib Figure to PNG bytes.

    Uses ``bbox_inches='tight'`` so the saved PNG crops cleanly to its
    content (no whitespace borders). Closes the figure after saving to
    release the matplotlib resources — important in a long-lived
    Streamlit process where many reports could otherwise leak figures.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Pump identity for the report
# ---------------------------------------------------------------------------


@dataclass
class _PumpIdentity:
    nozzle: str
    throat: str
    source: str  # "sidebar", "current installed", or "Custom"
    date_set: Optional[pd.Timestamp] = None
    # Set when the installed pump differs from the modeled (sidebar) pump,
    # so the cover line can call out the mismatch.
    installed_nozzle: Optional[str] = None
    installed_throat: Optional[str] = None
    installed_date_set: Optional[pd.Timestamp] = None


def _resolve_pump_identity(params: SimulationParams) -> _PumpIdentity:
    """Pick the pump the PDF should describe.

    The modeled pump is ALWAYS the sidebar pump — the hero solve and the
    batch sweep run with the JetPump built from ``params.nozzle_no`` /
    ``params.area_ratio``, so that's what the cover line must claim. (The
    old behavior labeled sidebar-solved numbers with the installed pump,
    which misattributed every what-if report.) JP history is consulted only
    to say whether the sidebar pump IS the installed one, or to note the
    installed pump it differs from.
    """
    if params.selected_well == "Custom":
        return _PumpIdentity(params.nozzle_no, params.area_ratio, "Custom")

    jp_hist = st.session_state.get("jp_history_df")
    current = None
    if jp_hist is not None:
        from woffl.assembly.jp_history import get_current_pump

        current = get_current_pump(jp_hist, params.selected_well)

    if not current or not current.get("nozzle_no") or not current.get("throat_ratio"):
        return _PumpIdentity(params.nozzle_no, params.area_ratio, "sidebar")

    same = str(current["nozzle_no"]) == str(params.nozzle_no) and str(
        current["throat_ratio"]
    ).strip().upper() == str(params.area_ratio).strip().upper()
    if same:
        return _PumpIdentity(
            params.nozzle_no,
            params.area_ratio,
            "current installed",
            date_set=current.get("date_set"),
        )
    return _PumpIdentity(
        params.nozzle_no,
        params.area_ratio,
        "sidebar",
        installed_nozzle=current["nozzle_no"],
        installed_throat=current["throat_ratio"],
        installed_date_set=current.get("date_set"),
    )


# ---------------------------------------------------------------------------
# Well-test actuals lookup
# ---------------------------------------------------------------------------


def _latest_test_row(well_name: str) -> Optional[pd.Series]:
    """Most recent well-test row for the given well, or None.

    Print captures the most-recent-test view (matches the Solver test picker
    default). Routes through ``get_well_tests_for_well`` so memory-gauge BHP
    overrides flow into the PDF's hero-strip deltas and selected-test card.
    """
    from woffl.gui.utils import get_well_tests_for_well

    well_tests = get_well_tests_for_well(well_name)
    if well_tests is None or well_tests.empty:
        return None
    return well_tests.sort_values("WtDate", ascending=False).iloc[0]


def _actuals_from_test(test_row: Optional[pd.Series]) -> dict[str, Optional[float]]:
    """Pull Oil / BHP / PF / WHP from a test row. Blank dict when None."""
    blank: dict[str, Optional[float]] = {
        "oil": None, "bhp": None, "pf": None, "whp": None, "date": None,
    }
    if test_row is None:
        return blank

    def _g(col: str) -> Optional[float]:
        v = test_row.get(col)
        return float(v) if is_valid_number(v) else None

    return {
        "oil": _g("WtOilVol"),
        "bhp": _g("BHP"),
        "pf": _g("lift_wat"),
        "whp": _g("whp"),
        "date": test_row.get("WtDate"),
    }


# ---------------------------------------------------------------------------
# IPR figure builder
# ---------------------------------------------------------------------------


def _build_ipr_figure(
    params: SimulationParams,
) -> Optional[plt.Figure]:
    """Build a matplotlib IPR figure mirroring the Solver tab's Model vs Actual.

    Same data-prep flow as before (Vogel-or-single-point fallback, sidebar
    reservoir pressure honored), then renders with matplotlib instead
    of Plotly. Matplotlib renders cleanly in any headless container —
    Kaleido isn't always reliable on Databricks Apps because its 1.x line
    fetches Chromium at runtime, which fails in restricted environments.

    Returns None for Custom mode or wells whose well-test cache is empty —
    no IPR section is added to the PDF in that case.
    """
    from woffl.assembly.ipr_analyzer import (
        compute_vogel_coefficients,
        estimate_reservoir_pressure,
        generate_ipr_curves,
    )
    from woffl.flow.inflow import InFlow

    from woffl.gui.utils import get_well_tests_for_well

    test_df = get_well_tests_for_well(params.selected_well)
    if test_df is None or test_df.empty:
        return None

    coeff_row = None
    merged_with_rp = None
    vogel_coeffs = None
    if len(test_df) >= 2:
        try:
            merged_with_rp = estimate_reservoir_pressure(test_df)
            vogel_coeffs = compute_vogel_coefficients(merged_with_rp)
            if (
                vogel_coeffs is not None
                and not vogel_coeffs.empty
                and "Well" in vogel_coeffs.columns
            ):
                well_coeffs = vogel_coeffs[vogel_coeffs["Well"] == params.selected_well]
                if not well_coeffs.empty:
                    coeff_row = well_coeffs.iloc[0]
        except Exception:
            coeff_row = None

    if coeff_row is not None:
        # ResP comes from the sidebar (seeded from the IPR-anchor test, editable)
        # so the PDF chart matches what the Solver / Batch Run use.
        model_res_p = float(params.pres)
        vogel_coeffs_plot = vogel_coeffs.copy()
        vogel_coeffs_plot.loc[
            vogel_coeffs_plot["Well"] == params.selected_well, "ResP"
        ] = model_res_p
        ipr_data = generate_ipr_curves(vogel_coeffs_plot)
        plot_points = merged_with_rp if merged_with_rp is not None else pd.DataFrame()
    else:
        # 0/1-test or Vogel-failed: synthesize a single-point IPR anchored
        # on the most recent test (or the sidebar values when the test
        # lacks total fluid / BHP).
        model_res_p = float(params.pres)
        recent = test_df.sort_values("WtDate", ascending=False).iloc[0]
        total = recent.get("WtTotalFluid")
        bhp = recent.get("BHP")
        if is_valid_number(total) and is_valid_number(bhp):
            anchor_qwf = float(total)
            anchor_pwf = float(bhp)
        else:
            wc = float(params.form_wc)
            if 0.0 <= wc < 1.0:
                anchor_qwf = float(params.qwf) / max(1e-6, 1.0 - wc)
            else:
                anchor_qwf = float(params.qwf)
            anchor_pwf = float(params.pwf)

        if not (anchor_qwf > 0 and 0 <= anchor_pwf < model_res_p and model_res_p > 0):
            return None  # Degenerate — skip IPR section in the PDF

        try:
            synth_qmax = InFlow.vogel_qmax(anchor_qwf, anchor_pwf, model_res_p)
        except Exception:
            return None
        synth_coeffs = pd.DataFrame(
            [{
                "Well": params.selected_well,
                "ResP": model_res_p,
                "qwf": anchor_qwf,
                "pwf": anchor_pwf,
                "QMax_recent": synth_qmax,
            }]
        )
        ipr_data = generate_ipr_curves(synth_coeffs)
        plot_points = test_df

    if params.selected_well not in ipr_data:
        return None

    well_ipr = ipr_data[params.selected_well]
    fig, ax = plt.subplots(figsize=(10.5, 6.5))

    # Vogel IPR curve.
    ax.plot(
        well_ipr["fluid_recent"],
        list(well_ipr["bhp_array"]),
        color="steelblue", linewidth=2.5, label="Vogel IPR", zorder=3,
    )

    # Test data scatter — colored by days_since to match the screen.
    if (
        plot_points is not None
        and not plot_points.empty
        and "well" in plot_points.columns
        and "BHP" in plot_points.columns
        and "WtTotalFluid" in plot_points.columns
    ):
        scatter_df = plot_points[plot_points["well"] == params.selected_well].copy()
        scatter_df = scatter_df.dropna(subset=["BHP", "WtTotalFluid"])
        if not scatter_df.empty and "WtDate" in scatter_df.columns:
            scatter_df["date"] = pd.to_datetime(scatter_df["WtDate"])
            today = pd.to_datetime("today")
            scatter_df["days_since"] = (today - scatter_df["date"]).dt.days
            sc = ax.scatter(
                scatter_df["WtTotalFluid"],
                scatter_df["BHP"],
                c=scatter_df["days_since"],
                cmap="viridis",
                s=80, edgecolors="black", linewidth=0.5,
                alpha=0.85, zorder=5, label="Test Data",
            )
            cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
            cbar.set_label("Days Ago", fontsize=11)

    res_p = well_ipr["res_pres"]
    qmax = well_ipr["qmax_recent"]
    ax.text(
        0.02, 0.97,
        f"Res P: {res_p:.0f} psi\nQmax: {qmax:.0f} BPD",
        transform=ax.transAxes, fontsize=11, va="top",
        bbox=dict(
            boxstyle="round,pad=0.4", facecolor="lightyellow",
            alpha=0.9, edgecolor="gray",
        ),
    )

    ax.set_xlabel("Total Fluid Rate (BPD)", fontsize=13)
    ax.set_ylabel("Bottom Hole Pressure (psi)", fontsize=13)
    ax.set_title(
        f"{params.selected_well} — Vogel IPR",
        fontsize=16, fontweight="bold",
    )
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=11)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Batch performance figure builder
# ---------------------------------------------------------------------------


def _build_batch_figure(
    batch_pump: Any, params: SimulationParams
) -> tuple[Optional[plt.Figure], Optional[dict]]:
    """Build a matplotlib batch-performance figure for the PDF.

    Mirrors ``_render_performance_graph`` in tabs/batch_run.py:
    eliminated (blue circles) + semi-finalist (red diamonds) scatter,
    dashed exp-fit curve through the semi-finalists, gold-star
    recommendation marker. Returns (fig, recommendation_dict); both may
    be None when the batch run didn't produce usable results.
    """
    water = params.water_type if params.water_type is not None else "total"

    df = batch_pump.df.copy()
    df = df[~df["qoil_std"].isna()]
    if df.empty:
        return None, None

    water_col = "form_wat" if water == "formation" else "totl_wat"
    max_water_data = float(df[water_col].max())

    fig, ax = plt.subplots(figsize=(10.5, 6.5))

    elim = df[~df["semi"]]
    if not elim.empty:
        ax.scatter(
            elim[water_col], elim["qoil_std"],
            color="royalblue", s=70, edgecolor="black", linewidth=0.6,
            label="Eliminated", zorder=4,
        )
        for x, y, n, t in zip(
            elim[water_col], elim["qoil_std"], elim["nozzle"], elim["throat"],
        ):
            ax.annotate(
                f"{n}{t}", (x, y),
                xytext=(5, 5), textcoords="offset points",
                fontsize=8, color="#333333",
            )

    semi = df[df["semi"]]
    if not semi.empty:
        ax.scatter(
            semi[water_col], semi["qoil_std"],
            color="crimson", marker="D", s=90, edgecolor="black", linewidth=0.6,
            label="Semi-Finalist", zorder=6,
        )
        for x, y, n, t in zip(
            semi[water_col], semi["qoil_std"], semi["nozzle"], semi["throat"],
        ):
            ax.annotate(
                f"{n}{t}", (x, y),
                xytext=(5, 5), textcoords="offset points",
                fontsize=8, color="crimson", fontweight="bold",
            )

    x_curve_zero = 0.0
    coeff = (
        getattr(batch_pump, "coeff_form", None) if water == "formation"
        else getattr(batch_pump, "coeff_totl", None)
    )
    if coeff is not None:
        a, b, c = coeff
        fit_water = np.linspace(0, max_water_data, 200)
        fit_oil_raw = np.array([exp_model(w, a, b, c) for w in fit_water])
        fit_oil = np.clip(fit_oil_raw, 0.0, None)
        positive_mask = fit_oil_raw > 0
        if positive_mask.any():
            x_curve_zero = float(fit_water[int(np.argmax(positive_mask))])
        ax.plot(
            fit_water, fit_oil,
            color="crimson", linewidth=2, linestyle="--",
            label="Exp. Curve Fit", zorder=5,
        )

    recommendation: Optional[dict] = None
    try:
        recommendation = recommend_jetpump(
            batch_pump, params.marginal_watercut, water
        )
        ax.scatter(
            [recommendation["water_rate"]], [recommendation["qoil_std"]],
            marker="*", s=500, color="gold", edgecolor="black", linewidth=1.2,
            label="Recommended", zorder=10,
        )
    except Exception:
        pass

    # x-axis range: same logic as the Plotly version. Formation water sits in a
    # narrow band, so autorange. Total water can start from the curve's
    # zero crossing to drop the empty left margin.
    if water == "formation":
        ax.set_xlim(left=df[water_col].min() * 0.95)
    else:
        x_axis_upper = max(max_water_data * 1.05, x_curve_zero + 1.0)
        ax.set_xlim(left=x_curve_zero, right=x_axis_upper)
    ax.set_ylim(bottom=0)

    wellname = getattr(batch_pump, "wellname", params.selected_well)
    ax.set_xlabel(f"{water.capitalize()} Water Rate (BWPD)", fontsize=13)
    ax.set_ylabel("Produced Oil Rate (BOPD)", fontsize=13)
    ax.set_title(
        f"{wellname} — Jet Pump Performance",
        fontsize=16, fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=11, framealpha=0.95)
    fig.tight_layout()
    return fig, recommendation


# ---------------------------------------------------------------------------
# Hero-strip values: re-run the solver so the PDF doesn't depend on stale
# session state. Cheap (<1 s for one solve).
# ---------------------------------------------------------------------------


def _solve_for_hero(
    params: SimulationParams, jetpump, wellbore, well_profile, inflow, res_mix
) -> Optional[tuple[float, bool, float, float, float, float]]:
    """Run the solver once and return (psu, sonic, qoil, fwat, qnz, mach_te)."""
    try:
        return run_jetpump_solver(
            params.surf_pres,
            params.form_temp,
            params.rho_pf,
            params.ppf_surf,
            jetpump,
            wellbore,
            well_profile,
            inflow,
            res_mix,
            field_model=params.field_model,
            jpump_direction=params.jpump_direction,
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# PDF assembly
# ---------------------------------------------------------------------------


def _styles():
    """ReportLab paragraph styles — cached as a fresh stylesheet per call.

    ReportLab styles aren't thread-safe across reuses; building a fresh
    StyleSheet keeps multiple concurrent Streamlit users from racing on
    style mutations.
    """
    from reportlab.lib.colors import HexColor
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet

    s = getSampleStyleSheet()
    s.add(ParagraphStyle(
        name="ReportTitle", parent=s["Title"],
        fontSize=22, spaceAfter=6, textColor=HexColor("#1a3a52"),
    ))
    s.add(ParagraphStyle(
        name="ReportSubtitle", parent=s["Normal"],
        fontSize=12, spaceAfter=14, textColor=HexColor("#555555"),
    ))
    s.add(ParagraphStyle(
        name="SectionHeader", parent=s["Heading2"],
        fontSize=14, spaceBefore=14, spaceAfter=6, textColor=HexColor("#1a3a52"),
    ))
    s.add(ParagraphStyle(
        name="NotesBody", parent=s["Normal"],
        fontSize=10, spaceAfter=8, leading=14,
        backColor=HexColor("#fff8d6"), borderPadding=8,
        borderColor=HexColor("#d4b800"), borderWidth=0.5,
    ))
    s.add(ParagraphStyle(
        name="Caption", parent=s["Normal"],
        fontSize=9, textColor=HexColor("#666666"),
    ))
    return s


def _cell(text: str, *, bold: bool = False, size: int = 9, align: str = "LEFT"):
    """Wrap cell text in a Paragraph so it wraps inside its column.

    Raw strings in ReportLab Tables overflow their column when they're
    wider than ``colWidth``. Paragraph cells auto-wrap, so this is the
    safety net for labels like "Reservoir Pressure" that might run long
    at certain font sizes.
    """
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph

    style = ParagraphStyle(
        f"_cell_{size}_{bold}_{align}",
        fontName="Helvetica-Bold" if bold else "Helvetica",
        fontSize=size,
        leading=size + 2,
        alignment={"LEFT": 0, "CENTER": 1, "RIGHT": 2}[align],
    )
    return Paragraph(str(text), style)


def _inputs_table(params: SimulationParams):
    """ReportLab Table mirroring the on-screen Model Inputs expander.

    Column widths sum to 7.2" — comfortably inside the 7.3" available
    width on Letter with 0.6" margins. Label columns are 1.35" each
    (room for "Reservoir Pressure" / "Surface Pressure" at 9pt bold);
    value columns are 1.05" each (room for "3168 scf/bbl" etc.). Cells
    are Paragraphs so anything still over-budget wraps cleanly instead
    of bleeding into the next column.
    """
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import Table, TableStyle

    def L(s):  # label cell
        return _cell(s, bold=True, size=9)

    def V(s):  # value cell
        return _cell(s, size=9)

    rows = [
        [_cell("Pump", bold=True, size=10), "",
         _cell("Well", bold=True, size=10), "",
         _cell("Formation / IPR", bold=True, size=10), ""],
        [L("Nozzle"), V(params.nozzle_no),
         L("PF Pressure"), V(f"{params.ppf_surf} psi"),
         L("Reservoir Pressure"), V(f"{params.pres} psi")],
        [L("Throat"), V(params.area_ratio),
         L("Surface Pressure"), V(f"{params.surf_pres} psi"),
         L("Water Cut"), V(f"{params.form_wc:.2f}")],
        [L("ken"), V(f"{params.ken:.3f}"),
         L("PF Density"), V(f"{params.rho_pf} lbm/ft³"),
         L("GOR"), V(f"{params.form_gor} scf/bbl")],
        [L("kth"), V(f"{params.kth:.3f}"),
         L("JP TVD"), V(f"{params.jpump_tvd} ft"),
         L("Temperature"), V(f"{params.form_temp} °F")],
        [L("kdi"), V(f"{params.kdi:.3f}"),
         L("Direction"), V(params.jpump_direction.capitalize()),
         L("qwf / pwf"), V(f"{params.qwf} / {params.pwf}")],
    ]
    tbl = Table(rows, colWidths=[1.35 * inch, 1.05 * inch] * 3)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef3")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
        ("SPAN", (0, 0), (1, 0)),
        ("SPAN", (2, 0), (3, 0)),
        ("SPAN", (4, 0), (5, 0)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return tbl


def _hero_table(
    modeled: tuple[float, float, float, float],
    actuals: dict[str, Optional[float]],
    deltas_valid: bool = True,
):
    """4-column hero strip: Oil / Formation Water / PF / Suction Pressure.

    Header text ("Formation Water", "Suction Pressure") is the widest
    content; columns are sized to fit it at 10pt bold plus a wrap cushion.
    Cells are Paragraphs so an unexpectedly wide value (e.g. five-digit
    delta) wraps instead of overflowing.

    ``deltas_valid=False`` blanks the Δ-vs-Actual row — used when the modeled
    (sidebar) pump differs from the pump installed at the test's date, where
    a delta would compare two different pumps (mirrors the on-screen hero).
    """
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import Table, TableStyle

    psu, qoil_std, fwat_bwpd, qnz_bwpd = modeled

    def _delta_str(modeled_val: float, actual: Optional[float], suffix: str) -> str:
        if actual is None:
            return "—"
        diff = modeled_val - actual
        sign = "+" if diff >= 0 else ""
        return f"{sign}{diff:,.0f} {suffix}"

    def H(s):  # header
        return _cell(s, bold=True, size=10, align="CENTER")

    def L(s):  # row label (leftmost column)
        return _cell(s, bold=True, size=9, align="LEFT")

    def V(s):  # value
        return _cell(s, size=9, align="CENTER")

    headers = [H("Oil Rate"), H("Formation Water"), H("Power Fluid"), H("Suction Pressure")]
    modeled_row = [
        V(f"{qoil_std:,.0f} BOPD"),
        V(f"{fwat_bwpd:,.0f} BWPD"),
        V(f"{qnz_bwpd:,.0f} BWPD"),
        V(f"{psu:,.0f} psig"),
    ]
    actual_row = [
        V(f"{actuals['oil']:,.0f} BOPD" if actuals.get("oil") is not None else "—"),
        V("—"),  # No actual for formation water
        V(f"{actuals['pf']:,.0f} BWPD" if actuals.get("pf") is not None else "—"),
        V(f"{actuals['bhp']:,.0f} psig" if actuals.get("bhp") is not None else "—"),
    ]
    if deltas_valid:
        delta_row = [
            V(_delta_str(qoil_std, actuals.get("oil"), "BOPD")),
            V("—"),
            V(_delta_str(qnz_bwpd, actuals.get("pf"), "BWPD")),
            V(_delta_str(psu, actuals.get("bhp"), "psig")),
        ]
    else:
        delta_row = [V("—"), V("—"), V("—"), V("—")]
    rows = [
        [V("")] + headers,
        [L("Modeled")] + modeled_row,
        [L("Actual")] + actual_row,
        [L("Δ vs Actual")] + delta_row,
    ]
    tbl = Table(
        rows,
        colWidths=[1.05 * inch, 1.55 * inch, 1.55 * inch, 1.55 * inch, 1.55 * inch],
    )
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef3")),
        ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#f5f5f5")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return tbl


def _calibration_block(well_name: str) -> list:
    """Build flowables for any calibration results present in session_state.

    Two possible blocks: rate-scalar (``sw_calibration_result``) and
    friction-coef (``sw_fric_calibration[well_name]``). Either or both may be
    absent — only render what's been computed.
    """
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

    flowables: list = []
    styles = _styles()

    rate_cal = st.session_state.get("sw_calibration_result")
    fric_cal_map = st.session_state.get("sw_fric_calibration", {})
    fric_cal = fric_cal_map.get(well_name)

    has_rate = (
        rate_cal is not None
        and getattr(rate_cal, "well_name", None) == well_name
    )
    has_fric = fric_cal is not None and getattr(fric_cal, "converged", False)

    if not has_rate and not has_fric:
        return flowables

    flowables.append(Paragraph("Calibration", styles["SectionHeader"]))

    # Two label+value pairs per row. Label cols are 1.6" (room for
    # "Match Quality" / "Derived from" at 9pt bold); value cols are 2.0".
    # Total 7.2" — fits inside the 7.3" available width.
    cal_widths = [1.6 * inch, 2.0 * inch, 1.6 * inch, 2.0 * inch]

    def _cal_style(title_color: str) -> TableStyle:
        return TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(title_color)),
            ("SPAN", (0, 0), (-1, 0)),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])

    def L(s):
        return _cell(s, bold=True, size=9)

    def V(s):
        return _cell(s, size=9)

    def T(s):
        return _cell(s, bold=True, size=10)

    if has_rate:
        rows = [
            [T("Rate-Scalar Calibration"), "", "", ""],
            [L("Factor"), V(f"{rate_cal.calibration_factor:.3f}"),
             L("Oil Error"), V(f"{rate_cal.oil_error_pct:.1f}%")],
            [L("Quality"), V(rate_cal.quality_grade.upper()),
             L("Derived from"),
             V(f"{rate_cal.current_nozzle}{rate_cal.current_throat}")],
        ]
        tbl = Table(rows, colWidths=cal_widths)
        tbl.setStyle(_cal_style("#e8eef3"))
        flowables.append(tbl)
        flowables.append(Spacer(1, 8))

    if has_fric:
        rows = [
            [T("BHP Friction-Coef Calibration"), "", "", ""],
            [L("Cal ken"), V(f"{fric_cal.best_ken:.3f}"),
             L("Cal kth"), V(f"{fric_cal.best_kth:.3f}")],
            [L("Cal kdi"), V(f"{fric_cal.best_kdi:.3f}"),
             L("BHP Error"), V(f"{fric_cal.bhp_error:+.0f} psi")],
            [L("Modeled BHP"), V(f"{fric_cal.best_modeled_bhp:.0f} psi"),
             L("Match Quality"),
             V(getattr(fric_cal, "match_quality", "unknown").upper())],
        ]
        tbl = Table(rows, colWidths=cal_widths)
        tbl.setStyle(_cal_style("#e8eef3"))
        flowables.append(tbl)

    return flowables


def _selected_test_card(actuals: dict[str, Optional[float]]):
    """Single-row card describing the test we're comparing against."""
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import Table, TableStyle

    test_date = actuals.get("date")
    if test_date is not None and hasattr(test_date, "strftime"):
        date_str = test_date.strftime("%Y-%m-%d")
    else:
        date_str = "—"

    def _fmt(val: Optional[float], suffix: str) -> str:
        return f"{val:,.0f} {suffix}" if val is not None else "—"

    def C(s, **kw):
        return _cell(s, align="CENTER", **kw)

    rows = [
        [_cell("Compared Against (Most Recent Test)", bold=True, size=10), "", "", "", ""],
        [C("Date", bold=True), C("Oil", bold=True), C("BHP", bold=True),
         C("PF Rate", bold=True), C("WHP", bold=True)],
        [
            C(date_str),
            C(_fmt(actuals.get("oil"), "BOPD")),
            C(_fmt(actuals.get("bhp"), "psi")),
            C(_fmt(actuals.get("pf"), "BWPD")),
            C(_fmt(actuals.get("whp"), "psi")),
        ],
    ]
    tbl = Table(rows, colWidths=[1.44 * inch] * 5)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef3")),
        ("SPAN", (0, 0), (-1, 0)),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return tbl


def _batch_data_table(batch_pump: Any):
    """Multi-page table of all successful batch-sweep rows.

    Mirrors the on-screen "Jet Pump Performance Data" table in
    ``tabs/batch_run.py::_render_data_table``. Sorted by oil rate
    descending so the most productive pumps land at the top of the first
    page. Semi-finalist rows are tinted yellow to match the on-screen
    visual hierarchy.

    Uses ``repeatRows=1`` so the header row reprints at the top of every
    page when the table spans multiple pages — important when batch
    sweeps return 30+ pumps and the table wraps.
    """
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import Table, TableStyle

    df = batch_pump.df.copy()
    df = df[~df["qoil_std"].isna()]
    if df.empty:
        return None

    df = df.sort_values("qoil_std", ascending=False).reset_index(drop=True)

    def H(s):  # 2-line column header: metric / unit
        return _cell(s, bold=True, size=8, align="CENTER")

    def V(s):  # cell value
        return _cell(s, size=8, align="CENTER")

    header = [
        H("Nozzle"), H("Throat"),
        H("Oil<br/>(BOPD)"), H("Form Water<br/>(BWPD)"),
        H("Lift Water<br/>(BWPD)"), H("Total Water<br/>(BWPD)"),
        H("Suction P<br/>(psig)"), H("Mach"), H("Sonic"), H("Semi"),
    ]
    rows = [header]
    semi_row_indices: list[int] = []  # 1-indexed (header is row 0)
    for idx, r in enumerate(df.itertuples(index=False), start=1):
        is_semi = bool(getattr(r, "semi", False))
        if is_semi:
            semi_row_indices.append(idx)
        sonic = "Yes" if bool(getattr(r, "sonic_status", False)) else "No"
        semi_str = "Yes" if is_semi else ""
        rows.append([
            V(str(r.nozzle)),
            V(str(r.throat)),
            V(f"{r.qoil_std:,.0f}"),
            V(f"{r.form_wat:,.0f}"),
            V(f"{r.lift_wat:,.0f}"),
            V(f"{r.totl_wat:,.0f}"),
            V(f"{r.psu_solv:,.0f}"),
            V(f"{r.mach_te:.3f}"),
            V(sonic),
            V(semi_str),
        ])

    col_widths = [
        0.45 * inch, 0.45 * inch,
        0.85 * inch, 0.95 * inch,
        0.85 * inch, 0.95 * inch,
        0.85 * inch, 0.55 * inch, 0.5 * inch, 0.5 * inch,
    ]
    tbl = Table(rows, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef3")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]
    # Tint semi-finalist rows so the visual hierarchy from the on-screen
    # diamond markers carries through to the PDF.
    for r_idx in semi_row_indices:
        style.append(("BACKGROUND", (0, r_idx), (-1, r_idx), colors.HexColor("#fff4cc")))
    tbl.setStyle(TableStyle(style))
    return tbl


def _recommended_pump_card(recommendation: Optional[dict], water_type: str):
    """Single-row card highlighting the recommended pump from the batch sweep."""
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import Table, TableStyle

    if recommendation is None:
        return None

    water_label = "Formation Water" if water_type == "formation" else "Total Water"

    def C(s, **kw):
        return _cell(s, align="CENTER", **kw)

    rows = [
        [_cell("Recommended Pump (Batch Sweep)", bold=True, size=11), "", "", ""],
        [C("Pump", bold=True, size=10), C("Oil Rate", bold=True, size=10),
         C(water_label, bold=True, size=10), C("Marginal WC", bold=True, size=10)],
        [
            C(f"{recommendation['nozzle']}{recommendation['throat']}", size=11),
            C(f"{recommendation['qoil_std']:.0f} BOPD", size=11),
            C(f"{recommendation['water_rate']:.0f} BWPD", size=11),
            C(f"{recommendation['marginal_ratio']:.3f}", size=11),
        ],
    ]
    tbl = Table(rows, colWidths=[1.8 * inch] * 4)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#fff4cc")),
        ("SPAN", (0, 0), (-1, 0)),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#aaaaaa")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return tbl


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_report(
    params: SimulationParams,
    jetpump,
    wellbore,
    well_profile,
    inflow,
    res_mix,
    *,
    engineer_name: str = "",
    notes: str = "",
) -> bytes:
    """Generate the multi-page PDF report and return its bytes.

    Runs the solver and a batch sweep, builds the IPR + batch performance
    figures, and lays out the report. The caller is responsible for handing
    the bytes to ``st.download_button``.
    """
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
    )

    styles = _styles()
    pump = _resolve_pump_identity(params)
    test_row = _latest_test_row(params.selected_well)
    actuals = _actuals_from_test(test_row)

    # Δ-vs-Actual is only meaningful when the modeled (sidebar) pump matches
    # the pump installed at the test's date — otherwise the delta compares
    # two different pumps (mirrors the on-screen hero's pump_differs logic).
    deltas_valid = True
    pump_at_test = None
    if test_row is not None and params.selected_well != "Custom":
        jp_hist = st.session_state.get("jp_history_df")
        if jp_hist is not None:
            from woffl.assembly.jp_history import get_pump_at_date

            pump_at_test = get_pump_at_date(
                jp_hist, params.selected_well, test_row.get("WtDate")
            )
        if pump_at_test and pump_at_test.get("nozzle_no") and pump_at_test.get("throat_ratio"):
            deltas_valid = str(pump_at_test["nozzle_no"]) == str(params.nozzle_no) and str(
                pump_at_test["throat_ratio"]
            ).strip().upper() == str(params.area_ratio).strip().upper()

    # Solver: fresh run from current params. The hero figures might differ
    # slightly from what's on screen when the user has been tweaking; that's
    # the intended behavior — print captures the current state.
    solver_results = _solve_for_hero(
        params, jetpump, wellbore, well_profile, inflow, res_mix
    )

    # Batch: always re-run for the PDF. The user clicked "Generate Report"
    # knowing this takes ~30 s; deterministic over caching stale state.
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
        wellname=params.selected_well if params.selected_well != "Custom" else f"{params.field_model} Well",
        field_model=params.field_model,
        jpump_direction=params.jpump_direction,
        ken=params.ken,
        kth=params.kth,
        kdi=params.kdi,
    )

    # Build Plotly figures and render to PNG.
    ipr_fig = _build_ipr_figure(params)
    batch_fig = None
    recommendation = None
    if batch_pump is not None:
        # ``coeff_form`` and ``mofwr`` are computed GUI-side, not by the
        # library — mirror the screen path so the formation-water axis works.
        from woffl.gui.tabs.batch_run import _augment_with_formation_marginals

        _augment_with_formation_marginals(batch_pump)
        batch_fig, recommendation = _build_batch_figure(batch_pump, params)

    ipr_png = _fig_to_png(ipr_fig) if ipr_fig is not None else None
    batch_png = _fig_to_png(batch_fig) if batch_fig is not None else None

    # ---- Document assembly ----
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=LETTER,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        title=f"Jet Pump Analysis — {params.selected_well}",
        author=engineer_name or "Woffl GUI",
    )

    story: list = []

    # Cover header
    story.append(Paragraph(
        f"Jet Pump Analysis — {params.selected_well}",
        styles["ReportTitle"],
    ))
    pump_line = f"Modeling pump <b>{pump.nozzle}{pump.throat}</b> ({pump.source})"
    if pump.date_set is not None and hasattr(pump.date_set, "strftime"):
        pump_line += f" — set {pump.date_set.strftime('%Y-%m-%d')}"
    if pump.installed_nozzle:
        pump_line += (
            f" — differs from installed "
            f"<b>{pump.installed_nozzle}{pump.installed_throat}</b>"
        )
        if pump.installed_date_set is not None and hasattr(
            pump.installed_date_set, "strftime"
        ):
            pump_line += f" (set {pump.installed_date_set.strftime('%Y-%m-%d')})"
    story.append(Paragraph(pump_line, styles["ReportSubtitle"]))

    meta_line = f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    if engineer_name.strip():
        # escape: ReportLab parses Paragraph text as markup — a stray <, >,
        # or & in free text aborts the whole doc.build.
        meta_line += f" by <b>{xml_escape(engineer_name.strip())}</b>"
    meta_line += f" · Field Model: {params.field_model}"
    story.append(Paragraph(meta_line, styles["Caption"]))
    story.append(Spacer(1, 10))

    if notes.strip():
        story.append(Paragraph("Engineer Notes", styles["SectionHeader"]))
        # Escape first (ReportLab parses markup — "WC <5% & PF >3000" would
        # abort doc.build), then preserve line breaks.
        notes_html = xml_escape(notes.strip()).replace("\n", "<br/>")
        story.append(Paragraph(notes_html, styles["NotesBody"]))

    # Model Inputs
    story.append(Paragraph("Model Inputs", styles["SectionHeader"]))
    story.append(_inputs_table(params))

    # Selected test card (skip for Custom / no-tests wells)
    if test_row is not None:
        story.append(Spacer(1, 10))
        story.append(_selected_test_card(actuals))

    # Hero strip
    if solver_results is not None:
        psu, _sonic, qoil_std, fwat_bwpd, qnz_bwpd, _mach = solver_results
        story.append(Paragraph("Solver Results", styles["SectionHeader"]))
        story.append(_hero_table(
            (psu, qoil_std, fwat_bwpd, qnz_bwpd), actuals, deltas_valid=deltas_valid,
        ))
        if not deltas_valid and pump_at_test is not None:
            story.append(Paragraph(
                f"Δ vs Actual suppressed — modeling "
                f"{params.nozzle_no}{params.area_ratio}, but the comparison test "
                f"ran pump {pump_at_test['nozzle_no']}{pump_at_test['throat_ratio']}.",
                styles["Caption"],
            ))
    else:
        story.append(Paragraph("Solver Results", styles["SectionHeader"]))
        story.append(Paragraph(
            "Solver did not converge with the current inputs.",
            styles["Normal"],
        ))

    # Calibration block (rate-scalar + friction-coef, when present)
    story.extend(_calibration_block(params.selected_well))

    # IPR chart on its own page (gives it room to breathe)
    if ipr_png is not None:
        story.append(PageBreak())
        story.append(Paragraph("Model vs Actual — IPR", styles["SectionHeader"]))
        story.append(Image(io.BytesIO(ipr_png), width=7.0 * inch, height=4.5 * inch))

    # Batch run on its own page
    if batch_png is not None:
        story.append(PageBreak())
        story.append(Paragraph("Batch Run — Jet Pump Performance", styles["SectionHeader"]))
        story.append(Image(io.BytesIO(batch_png), width=7.0 * inch, height=5.0 * inch))
        story.append(Spacer(1, 10))
        rec_card = _recommended_pump_card(recommendation, params.water_type or "total")
        if rec_card is not None:
            story.append(rec_card)

        # Full performance data table. Wraps across pages when the batch
        # sweep returns enough pumps to exceed a single page — the
        # ``repeatRows=1`` setting reprints the header on each new page.
        if batch_pump is not None:
            data_tbl = _batch_data_table(batch_pump)
            if data_tbl is not None:
                story.append(Spacer(1, 14))
                story.append(Paragraph(
                    "Jet Pump Performance Data", styles["SectionHeader"]
                ))
                story.append(data_tbl)

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()
