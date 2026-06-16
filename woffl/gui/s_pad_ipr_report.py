"""S-Pad "IPRs used" PDF report.

One page per reviewed well: the total-fluid Vogel IPR the optimization used,
with the actual well-test points overlaid. Points whose BHP came from an
uploaded memory gauge are drawn distinctly (orange diamonds), and a provenance
line records how each IPR was established (Vogel / single test / forced /
hypothetical) and, when gauge-backed, the gauge window + sample count.

Matplotlib (Agg) + PdfPages — kaleido-free, per the repo's export rule. Pure:
the caller gathers each well's tests + gauge-coverage dates and passes them in,
so this module imports no Streamlit / Databricks.
"""

import io

import matplotlib

matplotlib.use("Agg")  # headless — never fetches Chromium (no kaleido)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from woffl.flow.inflow import InFlow  # noqa: E402

_GAUGE_COLOR = "#ff7f0e"
_DB_COLOR = "#2ca02c"
_CURVE_COLOR = "#1f77b4"


def _vogel_curve(qwf_total: float, pwf: float, res_pres: float, n: int = 60):
    """(rate, pressure) arrays for the total-fluid Vogel IPR, or None if the
    anchor is degenerate (e.g. pwf ≥ reservoir pressure)."""
    try:
        inflow = InFlow(qwf=float(qwf_total), pwf=float(pwf), pres=float(res_pres))
    except (ValueError, ZeroDivisionError):
        return None
    ps = np.linspace(0.0, float(res_pres), n)
    rates = []
    for p in ps:
        try:
            rates.append(inflow.oil_flow(float(p)))
        except ValueError:
            rates.append(np.nan)
    return np.array(rates), ps


def _well_fig(item: dict):
    entry = item["entry"]
    tests = item.get("tests_df")
    gdates = item.get("gauge_dates") or set()
    well = entry["well_name"]

    fig, ax = plt.subplots(figsize=(8.5, 5.6))

    curve = _vogel_curve(entry["qwf"], entry["pwf"], entry["res_pres"])
    if curve is not None:
        rates, ps = curve
        ax.plot(rates, ps, "-", color=_CURVE_COLOR, lw=2, label="Vogel IPR (total fluid)")

    # IPR anchor (total-fluid qwf @ pwf)
    ax.scatter([entry["qwf"]], [entry["pwf"]], marker="*", s=200, color="black",
               zorder=6, label="IPR anchor")
    ax.axhline(entry["res_pres"], ls=":", color="grey", lw=1)
    ax.text(0, entry["res_pres"], f"  ResP {entry['res_pres']:.0f} psi",
            va="bottom", ha="left", fontsize=8, color="grey")

    # Test points — split gauge-backed BHP from Databricks BHP.
    if (tests is not None and not tests.empty
            and {"WtTotalFluid", "BHP", "WtDate"}.issubset(tests.columns)):
        t = tests.dropna(subset=["WtTotalFluid", "BHP"]).copy()
        if not t.empty:
            t["_d"] = pd.to_datetime(t["WtDate"]).dt.normalize()
            is_g = t["_d"].isin(gdates)
            tn, tg = t[~is_g], t[is_g]
            if not tn.empty:
                ax.scatter(tn["WtTotalFluid"], tn["BHP"], s=55, color=_DB_COLOR,
                           edgecolors="black", lw=0.5, zorder=5,
                           label="Test (Databricks BHP)")
            if not tg.empty:
                ax.scatter(tg["WtTotalFluid"], tg["BHP"], s=75, marker="D",
                           color=_GAUGE_COLOR, edgecolors="black", lw=0.5, zorder=5,
                           label="Test (memory-gauge BHP)")

    badge = f"{entry.get('ipr_source', '?')} IPR · {entry.get('bhp_source', '?')} BHP"
    if entry.get("offline"):
        badge += " · OFFLINE (excluded from run)"
    ax.set_title(f"{well}   —   {badge}", fontsize=11)
    ax.set_xlabel("Total fluid rate (BPD)")
    ax.set_ylabel("Flowing BHP (psi)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    note = entry.get("gauge_note") or ""
    if note:
        fig.text(0.5, 0.015, note, ha="center", fontsize=9, color=_GAUGE_COLOR)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def _cover_fig(pad: str, items: list[dict], stamp: str):
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    n_off = sum(1 for it in items if it["entry"].get("offline"))
    n_gauge = sum(1 for it in items if it["entry"].get("gauge_note"))
    header = [
        f"{pad}-Pad — IPRs Used for Optimization",
        "",
        stamp,
        f"{len(items)} reviewed well(s)  ·  {len(items) - n_off} active / {n_off} offline"
        f"  ·  {n_gauge} gauge-backed",
        "",
        "Legend:  ★ IPR anchor   ● Databricks BHP   ◆ memory-gauge BHP",
        "",
        "Wells:",
        "",
    ]
    rows = []
    for it in items:
        e = it["entry"]
        tags = [e.get("ipr_source", "?")]
        if e.get("gauge_note"):
            tags.append("gauge")
        if e.get("offline"):
            tags.append("OFFLINE")
        rows.append(f"  • {e['well_name']:<10}  ({', '.join(tags)})")
    ax.text(0.06, 0.96, "\n".join(header + rows), va="top", ha="left",
            fontsize=11, family="monospace")
    return fig


def build_ipr_pdf(pad: str, items: list[dict], stamp: str = "") -> bytes:
    """Build the IPRs-used PDF and return its bytes.

    Args:
        pad: pad letter (e.g. "S").
        items: one dict per well — ``{"entry": store_entry, "tests_df": DataFrame
            or None, "gauge_dates": set of normalized Timestamps the gauge covers}``.
        stamp: a caption for the cover (pass the timestamp in; keeps this pure).
    """
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        cover = _cover_fig(pad, items, stamp)
        pdf.savefig(cover)
        plt.close(cover)
        for it in items:
            try:
                fig = _well_fig(it)
            except Exception:
                continue  # one bad well shouldn't sink the whole report
            pdf.savefig(fig)
            plt.close(fig)
    return buf.getvalue()
