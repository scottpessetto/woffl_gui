"""PDF report export for the Header Pressure Impact tab.

Assembles a multi-page PDF (matplotlib, headless ``Agg`` backend) for a header-impact
run: field + per-pad sensitivity, the modeled-vs-reality sense check, the per-pad
optimism bias, the per-well WHP→BHP correlation summary, the response curve (if
built), and the Vogel IPR curves with each well's operating→scenario shift.

Offline-testable: ``build_report_pdf`` returns PDF bytes — no Streamlit or
Databricks runtime needed. The figure helpers reuse the same engine tables the
on-screen review computes, so the report and the app never disagree.
"""

from __future__ import annotations

import io
import math

import matplotlib

matplotlib.use("Agg")  # headless: render in a server/Databricks process, no display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from .header_engine import (  # noqa: E402
    aggregate_response_curve,
    bias_by_pad,
    sense_check_table,
    summarize_sensitivity,
)


def fig_to_png_bytes(fig, dpi: int = 150) -> bytes:
    """Render a matplotlib Figure to PNG bytes and close it (kaleido-free export)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def corr_grid_mpl(well_dfs: dict, fits: dict, wells, driver: str = "WHP", title=None,
                  results_df=None):
    """Matplotlib BHP-vs-driver scatter grid — hourly points colored by date with the
    good within-day fits in red. Kaleido-free replacement for the Plotly grid's PNG /
    PDF export. When ``results_df`` is given, EVERY well in ``wells`` gets a panel:
    its own trend, the donor's trend (titled "using <donor> corr"), or a labeled
    "using group correlation" note. Returns a Figure or None.
    """
    from . import header_engine as he
    from . import header_trend as ht

    key = f"BHP~{driver}"
    wells = list(wells)
    if not wells:
        return None
    plan = he.corr_display_plan(results_df, list((well_dfs or {}).keys())) if results_df is not None else {}
    cols = int(np.ceil(np.sqrt(len(wells))))
    rows = int(np.ceil(len(wells) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 8.5), squeeze=False)
    for i, w in enumerate(wells):
        ax = axes[i // cols][i % cols]
        p = plan.get(w, {})
        source = p.get("source", (w if w in (well_dfs or {}) else None))
        note = p.get("note", "")
        slope_hint = p.get("slope")
        src_df = (well_dfs or {}).get(source) if source else None
        d = None
        if src_df is not None and driver in src_df.columns and "BHP" in src_df.columns:
            d = src_df[[driver, "BHP"]].replace([np.inf, -np.inf], np.nan).dropna()
            d = d[d["BHP"] > 50]
        if d is not None and not d.empty:
            idx = pd.DatetimeIndex(d.index)
            ords = (idx.normalize() - idx.normalize().min()).days
            ax.scatter(d[driver], d["BHP"], c=ords, cmap="viridis", s=6)
            f = (fits.get(source) or {}).get(key)
            if f is not None and getattr(f, "daily", None) is not None and not f.daily.empty \
                    and "good" in f.daily.columns:
                for _, dr in f.daily[f.daily["good"]].iterrows():
                    if pd.isna(dr["slope"]):
                        continue
                    x0, x1 = dr["x_min"], dr["x_max"]
                    ax.plot([x0, x1],
                            [dr["slope"] * x0 + dr["intercept"], dr["slope"] * x1 + dr["intercept"]],
                            color="crimson", lw=0.8, alpha=0.7)
            cls = ht.classify_response(f) if f is not None else "n/a"
            slope = f.mean_slope if f is not None else float("nan")
            ttl = f"{w}  m={slope:.2f} · {cls}" + (f"\n{note}" if note else "")
            ax.set_title(ttl, fontsize=7)
            ax.tick_params(labelsize=6)
            try:
                xlo, xhi = np.nanpercentile(d[driver], [1, 99])
                ylo, yhi = np.nanpercentile(d["BHP"], [1, 99])
                ax.set_xlim(xlo - max((xhi - xlo) * 0.05, 1.0), xhi + max((xhi - xlo) * 0.05, 1.0))
                ax.set_ylim(ylo - max((yhi - ylo) * 0.05, 1.0), yhi + max((yhi - ylo) * 0.05, 1.0))
            except Exception:
                pass
        else:
            # No own/donor scatter — show a labeled note (e.g. "using group correlation").
            msg = note or "no correlation data"
            if pd.notna(slope_hint):
                msg += f"\n(m={float(slope_hint):.2f})"
            ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=8, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(w, fontsize=7)
    for j in range(len(wells), rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(title or f"BHP vs {driver} — good daily fits in red",
                 fontsize=13, fontweight="bold")
    try:
        fig.supxlabel(f"{driver} (psi)")
        fig.supylabel("BHP (psi)")
    except Exception:
        pass
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def well_fit_mpl(well: str, trend_df: pd.DataFrame, well_fits: dict):
    """Matplotlib per-well BHP-vs-WHP/HeaderP fit plot (kaleido-free PNG export).
    Returns a Figure or None."""
    from . import header_trend as ht

    avail = [(drv, f"BHP~{drv}") for drv in ("WHP", "HeaderP")
             if drv in trend_df.columns and "BHP" in trend_df.columns]
    if not avail:
        return None
    fig, axes = plt.subplots(1, len(avail), figsize=(11, 5), squeeze=False)
    for i, (drv, key) in enumerate(avail):
        ax = axes[0][i]
        d = trend_df[[drv, "BHP"]].replace([np.inf, -np.inf], np.nan).dropna()
        d = d[d["BHP"] > 50]
        if not d.empty:
            idx = pd.DatetimeIndex(d.index)
            ords = (idx.normalize() - idx.normalize().min()).days
            ax.scatter(d[drv], d["BHP"], c=ords, cmap="viridis", s=8)
        f = (well_fits or {}).get(key)
        if f is not None and getattr(f, "daily", None) is not None and not f.daily.empty:
            gd = f.daily[f.daily["good"]] if "good" in f.daily.columns else f.daily
            for _, dr in gd.iterrows():
                if pd.isna(dr["slope"]):
                    continue
                x0, x1 = dr["x_min"], dr["x_max"]
                ax.plot([x0, x1],
                        [dr["slope"] * x0 + dr["intercept"], dr["slope"] * x1 + dr["intercept"]],
                        color="crimson", lw=1.0, alpha=0.6)
        cls = ht.classify_response(f) if f is not None else "no fit"
        slope = f.mean_slope if f is not None else float("nan")
        ax.set_title(f"BHP vs {drv} — m={slope:.2f}, {cls}", fontsize=10)
        ax.set_xlabel(f"{drv} (psi)")
        ax.set_ylabel("BHP (psi)")
    fig.suptitle(f"{well} — within-day fits", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def _table_page(pdf: PdfPages, title: str, df: pd.DataFrame | None,
                fmt: dict | None = None, rows_per_page: int = 26) -> None:
    """Render a DataFrame as landscape-page matplotlib table(s), paginating rows
    across multiple pages so long tables don't run off the bottom of one page."""
    if df is None or df.empty:
        return
    df = df.copy()
    if fmt:
        for col, f in fmt.items():
            if col in df.columns:
                df[col] = df[col].map(lambda v, f=f: f.format(v) if pd.notna(v) else "—")
    df = df.astype(object).where(pd.notna(df), "—")
    pages = max(1, math.ceil(len(df) / rows_per_page))
    for p in range(pages):
        chunk = df.iloc[p * rows_per_page:(p + 1) * rows_per_page]
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        page_title = title if pages == 1 else f"{title}  (p. {p + 1}/{pages})"
        ax.set_title(page_title, fontsize=14, fontweight="bold", loc="left", pad=16)
        tbl = ax.table(cellText=chunk.values, colLabels=list(df.columns),
                       loc="upper center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7)
        tbl.scale(1, 1.4)
        pdf.savefig(fig)
        plt.close(fig)


def _curve_page(pdf: PdfPages, agg: dict, title: str) -> None:
    if not agg or not agg.get("deltas"):
        return
    deltas = agg["deltas"]
    fig, ax = plt.subplots(figsize=(11, 8.5))
    for pad, ys in sorted(agg.get("pads", {}).items()):
        ax.plot(deltas, ys, marker="o", label=f"Pad {pad}")
    if "ALL" in agg:
        ax.plot(deltas, agg["ALL"], marker="o", color="black", lw=2.5, ls=":", label="Field (ALL)")
        # Label each ALL point with its oil impact so it reads at a glance.
        for x, y in zip(deltas, agg["ALL"]):
            ax.annotate(f"{y:+.0f}", (x, y), textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=8, fontweight="bold", color="black")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Header change Δ (psi)")
    ax.set_ylabel("ΔOil vs current (BOPD)")
    ax.set_title(title, fontsize=14, fontweight="bold", loc="left")
    ax.legend()
    pdf.savefig(fig)
    plt.close(fig)


def _ipr_page(pdf: PdfPages, results_df: pd.DataFrame, ipr_rows: dict, pad: str,
              test_df=None) -> None:
    from woffl.flow.inflow import InFlow

    pad_df = results_df[results_df["Pad"] == pad]
    wells = [w for w in pad_df["Well"].tolist() if w in ipr_rows]
    if not wells:
        return
    cols = int(np.ceil(np.sqrt(len(wells))))
    rows = int(np.ceil(len(wells) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 8.5), squeeze=False)
    by_well = pad_df.set_index("Well")
    for i, w in enumerate(wells):
        ax = axes[i // cols][i % cols]
        ir = ipr_rows[w]
        try:
            rp = float(ir.get("res_pres", 1800.0))
            fwc = float(ir.get("form_wc", 0.5))
            inflow = InFlow(qwf=float(ir["qwf"]) * (1.0 - fwc), pwf=float(ir["pwf"]), pres=rp)
            bg = np.linspace(0.0, rp, 40)
            og = [float(inflow.oil_flow(float(b), "vogel")) for b in bg]
            ax.plot(og, bg, "-", color="#1f77b4")
        except Exception:
            pass
        wr = by_well.loc[w] if w in by_well.index else None
        if wr is not None:
            if pd.notna(wr.get("Oil now (BOPD)")) and pd.notna(wr.get("BHP now (psi)")):
                ax.plot([wr["Oil now (BOPD)"]], [wr["BHP now (psi)"]], "o", color="black")
            if pd.notna(wr.get("Oil scen (BOPD)")) and pd.notna(wr.get("BHP scen (psi)")):
                ax.plot([wr["Oil scen (BOPD)"]], [wr["BHP scen (psi)"]], "x", color="crimson", ms=8)
        if test_df is not None and not getattr(test_df, "empty", True) and "well" in test_df.columns:
            wt = test_df[test_df["well"] == w]
            if "BHP" in wt.columns and "WtOilVol" in wt.columns:
                wt = wt.dropna(subset=["BHP", "WtOilVol"])
                if not wt.empty:
                    ax.scatter(wt["WtOilVol"], wt["BHP"], s=14, facecolors="none",
                               edgecolors="#777", linewidths=0.8)
        ax.set_title(str(w), fontsize=8)
        ax.tick_params(labelsize=6)
    for j in range(len(wells), rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(f"Pad {pad} — Vogel IPR (● now, ✕ scenario)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig)
    plt.close(fig)


_SENSE_COLS = [
    "Well", "Pad", "Phys ΔOil (BOPD)", "Emp ΔOil (BOPD)", "Chosen ΔOil (BOPD)",
    "Oil now modeled (BOPD)", "Oil now measured (BOPD)", "Oil residual %",
]


def build_report_pdf(
    results_df: pd.DataFrame,
    delta_p: float,
    ipr_rows: dict | None = None,
    test_oil: dict | None = None,
    curve: dict | None = None,
    corr_table: pd.DataFrame | None = None,
    test_df=None,
    well_dfs: dict | None = None,
    fits: dict | None = None,
    stamp: str = "",
) -> bytes:
    """Assemble the header-impact report PDF and return its bytes.

    Always emits at least a cover page, so the result is a valid PDF even for an
    empty run. ``stamp`` (e.g. a timestamp) is printed on the cover; pass it in
    rather than reading the clock so the call stays deterministic.
    """
    ipr_rows = ipr_rows or {}
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # Cover
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.5, 0.62, "Header Pressure Impact Report", ha="center", fontsize=22, fontweight="bold")
        nwells = 0 if results_df is None else len(results_df)
        ax.text(0.5, 0.52, f"Header change Δ = {int(delta_p):+d} psi   ·   {nwells} wells",
                ha="center", fontsize=13)
        if stamp:
            ax.text(0.5, 0.46, stamp, ha="center", fontsize=10, color="gray")
        pdf.savefig(fig)
        plt.close(fig)

        if results_df is None or results_df.empty:
            return buf.getvalue()

        # Sensitivity (per-pad + overall)
        per_pad, overall = summarize_sensitivity(results_df, delta_p)
        if not per_pad.empty:
            table = pd.concat([per_pad, pd.DataFrame([overall])], ignore_index=True)
            _table_page(
                pdf, f"Sensitivity — ΔOil at {int(delta_p):+d} psi (BOPD per 100 psi normalized)",
                table,
                fmt={"Oil now (BOPD)": "{:,.0f}", "ΔOil (BOPD)": "{:+,.0f}",
                     "Oil after Δ (BOPD)": "{:,.0f}", "BOPD per 100 psi": "{:+,.1f}"},
            )

        # Response curve
        agg = aggregate_response_curve(curve)
        if agg:
            _curve_page(pdf, agg, "Oil response vs header change")

        # Modeled vs reality (trimmed to key columns so the table fits the page)
        sc = sense_check_table(results_df, test_oil or {})
        if not sc.empty:
            sc = sc[[c for c in _SENSE_COLS if c in sc.columns]]
            _table_page(
                pdf, "Modeled vs Reality", sc,
                fmt={"Phys ΔOil (BOPD)": "{:+,.0f}", "Emp ΔOil (BOPD)": "{:+,.0f}",
                     "Chosen ΔOil (BOPD)": "{:+,.0f}", "Oil now modeled (BOPD)": "{:,.0f}",
                     "Oil now measured (BOPD)": "{:,.0f}", "Oil residual %": "{:+,.0f}%"},
            )

        # Per-pad optimism bias
        _table_page(pdf, "Per-pad optimism bias (Emp ÷ Phys ΔOil, responsive wells)",
                    bias_by_pad(results_df), fmt={"Emp/Phys bias": "{:.2f}"})

        # Correlation summary
        if corr_table is not None and not corr_table.empty:
            _table_page(pdf, "WHP→BHP correlation summary", corr_table,
                        fmt={"BHP~WHP slope": "{:.2f}", "r²": "{:.2f}",
                             "BHP~HeaderP slope": "{:.2f}"})

        # Per-pad pages: IPR curves first, then the WHP→BHP correlation grid.
        for pad in sorted(results_df["Pad"].unique()):
            try:
                _ipr_page(pdf, results_df, ipr_rows, pad, test_df)
            except Exception:
                pass
            if well_dfs:
                try:
                    pad_df = results_df[results_df["Pad"] == pad]
                    cf = corr_grid_mpl(
                        well_dfs, fits or {}, pad_df["Well"].tolist(), "WHP",
                        title=f"Pad {pad} — WHP→BHP correlations (good daily fits in red)",
                        results_df=pad_df,
                    )
                    if cf is not None:
                        pdf.savefig(cf)
                        plt.close(cf)
                except Exception:
                    pass

    return buf.getvalue()


# ── per-well review PDF (one page per well: IPR | correlation) ────────────────


def _ipr_on_axis(ax, well, ir, wr, test_df, driver: str = "WHP") -> None:
    """Draw a well's Vogel IPR (+ operating/scenario points + test points) on ``ax``."""
    from woffl.flow.inflow import InFlow

    try:
        rp = float(ir.get("res_pres", 1800.0))
        fwc = float(ir.get("form_wc", 0.5))
        inflow = InFlow(qwf=float(ir["qwf"]) * (1.0 - fwc), pwf=float(ir["pwf"]), pres=rp)
        bg = np.linspace(0.0, rp, 60)
        og = [float(inflow.oil_flow(float(b), "vogel")) for b in bg]
        ax.plot(og, bg, "-", color="#1f77b4", lw=2, label="Vogel IPR")
    except Exception:
        ax.text(0.5, 0.5, "no IPR data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("IPR", fontsize=11)
        return
    if wr is not None:
        if pd.notna(wr.get("Oil now (BOPD)")) and pd.notna(wr.get("BHP now (psi)")):
            ax.plot([wr["Oil now (BOPD)"]], [wr["BHP now (psi)"]], "o", color="black",
                    ms=9, label="now")
        if pd.notna(wr.get("Oil scen (BOPD)")) and pd.notna(wr.get("BHP scen (psi)")):
            ax.plot([wr["Oil scen (BOPD)"]], [wr["BHP scen (psi)"]], "X", color="crimson",
                    ms=10, label="scenario")
    if test_df is not None and not getattr(test_df, "empty", True) and "well" in test_df.columns:
        wt = test_df[test_df["well"] == well]
        if "BHP" in wt.columns and "WtOilVol" in wt.columns:
            wt = wt.dropna(subset=["BHP", "WtOilVol"])
            if not wt.empty:
                ax.scatter(wt["WtOilVol"], wt["BHP"], s=22, facecolors="none",
                           edgecolors="#777", linewidths=0.8, label="tests")
    ax.set_xlabel("Oil (BOPD)")
    ax.set_ylabel("BHP (psi)")
    ax.set_title("IPR  (● now · ✕ scenario)", fontsize=11)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, loc="best")


def _corr_on_axis(ax, well, source, well_dfs, fits, note, driver: str = "WHP") -> None:
    """Draw a well's WHP→BHP correlation (own or donor) on ``ax`` — or a labeled note
    when it uses a group-average correlation / has no trend."""
    from . import header_trend as ht

    key = f"BHP~{driver}"
    src_df = (well_dfs or {}).get(source) if source else None
    d = None
    if src_df is not None and driver in src_df.columns and "BHP" in src_df.columns:
        d = src_df[[driver, "BHP"]].replace([np.inf, -np.inf], np.nan).dropna()
        d = d[d["BHP"] > 50]
    if d is None or d.empty:
        ax.text(0.5, 0.5, note or "no correlation data", ha="center", va="center",
                transform=ax.transAxes, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"WHP→BHP correlation", fontsize=11)
        return
    idx = pd.DatetimeIndex(d.index)
    ords = (idx.normalize() - idx.normalize().min()).days
    ax.scatter(d[driver], d["BHP"], c=ords, cmap="viridis", s=12)
    f = (fits.get(source) or {}).get(key)
    if f is not None and getattr(f, "daily", None) is not None and not f.daily.empty \
            and "good" in f.daily.columns:
        for _, dr in f.daily[f.daily["good"]].iterrows():
            if pd.isna(dr["slope"]):
                continue
            x0, x1 = dr["x_min"], dr["x_max"]
            ax.plot([x0, x1], [dr["slope"] * x0 + dr["intercept"], dr["slope"] * x1 + dr["intercept"]],
                    color="crimson", lw=1.2, alpha=0.7)
    cls = ht.classify_response(f) if f is not None else "n/a"
    slope = f.mean_slope if f is not None else float("nan")
    ttl = f"WHP→BHP  m={slope:.2f} · {cls}" + (f"  ({note})" if note else "")
    ax.set_title(ttl, fontsize=11)
    ax.set_xlabel(f"{driver} (psi)")
    ax.set_ylabel("BHP (psi)")
    try:
        xlo, xhi = np.nanpercentile(d[driver], [1, 99])
        ylo, yhi = np.nanpercentile(d["BHP"], [1, 99])
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
    except Exception:
        pass


def build_per_well_pdf(results_df, ipr_rows=None, well_dfs=None, fits=None,
                       test_df=None, stamp: str = "") -> bytes:
    """One page per well — its IPR (left) and WHP→BHP correlation (right) side by
    side, in the SAME order as the IPR & Correlation Coverage table — so the run can
    be reviewed well by well, then regenerated after a re-run. Returns PDF bytes.
    """
    from . import header_engine as he

    ipr_rows = ipr_rows or {}
    well_dfs = well_dfs or {}
    fits = fits or {}
    plan = he.corr_display_plan(results_df, list(well_dfs.keys()))
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.5, 0.6, "Per-Well Review — IPR + WHP→BHP", ha="center",
                fontsize=20, fontweight="bold")
        n = 0 if results_df is None else len(results_df)
        ax.text(0.5, 0.52, f"{n} wells · same order as the coverage table",
                ha="center", fontsize=12)
        if stamp:
            ax.text(0.5, 0.46, stamp, ha="center", fontsize=10, color="gray")
        pdf.savefig(fig)
        plt.close(fig)
        if results_df is None or results_df.empty:
            return buf.getvalue()
        for _, r in results_df.iterrows():
            well = r["Well"]
            fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 8.5))
            sub = f"Pad {r.get('Pad', '')} · {r.get('Lift', '')} · {r.get('Verdict', '')}"
            doil = r.get("Chosen ΔOil (BOPD)", r.get("ΔOil (BOPD)"))
            try:
                if pd.notna(doil):
                    sub += f" · ΔOil {float(doil):+.0f} BOPD"
            except Exception:
                pass
            fig.suptitle(f"{well}\n{sub}", fontsize=14, fontweight="bold")
            try:
                _ipr_on_axis(axL, well, ipr_rows.get(well, {}), r, test_df)
            except Exception:
                axL.text(0.5, 0.5, "IPR unavailable", ha="center", va="center", transform=axL.transAxes)
            try:
                p = plan.get(well, {})
                _corr_on_axis(axR, well, p.get("source"), well_dfs, fits, p.get("note", ""))
            except Exception:
                axR.text(0.5, 0.5, "correlation unavailable", ha="center", va="center", transform=axR.transAxes)
            fig.tight_layout(rect=(0, 0, 1, 0.92))
            pdf.savefig(fig)
            plt.close(fig)
    return buf.getvalue()
