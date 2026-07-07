"""Shared helpers for the pad-optimization pages and the pad review stage.

These previously lived in ``s_pad_page`` and were imported sideways by the I/M
pad pages — and silently NOT imported by ``step_review_wells`` (the batch
auto-match NameError). A dedicated leaf module removes the sibling-page
dependency and the ``s_pad_page ↔ step_review_wells`` import cycle that made
the direct import impossible.
"""

from __future__ import annotations

from typing import Optional


def parse_pump(label) -> Optional[tuple[str, str]]:
    """'12B' -> ('12','B'); 'Shut in'/blank -> None. Throat = trailing letter."""
    if not label or str(label).strip().lower() in ("shut in", "si", "—", ""):
        return None
    s = str(label).strip()
    i = len(s)
    while i > 0 and s[i - 1].isalpha():
        i -= 1
    return (s[:i], s[i:]) if s[:i] and s[i:] else None


def render_results_accounting(meta: dict, active: dict, si_wells: list[str]) -> bool:
    """Staleness banner + reasoned not-in-plan box for a pad Results page.

    ``meta`` carries ``store_sig`` (what the run was computed from) and
    ``reconciliation`` (per-well drop reasons), both stamped at run time.
    ``active`` is the live active-entries store; ``si_wells`` the active wells
    missing from the run's results. Returns True when the store has drifted
    since the run (results are stale). Replaces the blanket "Optimizer shut
    in — uneconomic at the marginal water cut" attribution, which was wrong
    for wells that failed simulation or were added after the run (P0-3).
    """
    import streamlit as st

    from woffl.gui.workflow_steps import well_review_store as wrs

    stale = meta.get("store_sig") is not None and meta[
        "store_sig"
    ] != wrs.store_signature(active)
    if stale:
        st.warning(
            "**Inputs changed since this run** — wells were added, edited, or "
            "toggled online/offline after these results were computed. The "
            "plan below reflects the OLD inputs; re-run the optimization to "
            "trust it."
        )

    if si_wells:
        recon = meta.get("reconciliation")
        reasons: dict[str, str] = {}
        if recon is not None and len(recon):
            reasons = dict(zip(recon["Well"], recon["Status"]))
        if stale:
            st.info(
                f"**{len(si_wells)} active well(s) are not in this run:** "
                + ", ".join(
                    f"{w} ({reasons.get(w, 'added/changed after the run')})"
                    for w in si_wells
                )
                + ". Re-run to include them."
            )
        else:
            st.info(
                f"**{len(si_wells)} well(s) not in the plan:** "
                + ", ".join(f"{w} ({reasons.get(w, 'not in run')})" for w in si_wells)
                + ". — 'not allocated' = the water budget bought more oil "
                "elsewhere (solver shut-in); 'failed simulation' = no pump "
                "combo converged; 'above marginal WC' = uneconomic at the "
                "marginal-watercut threshold."
            )
    return stale


def pump_size_change(cur_nz, cur_th, opt_nz, opt_th) -> str:
    """Direction of the pump-size move, current → optimized.

    Nozzle number is the size signal ("12B" → "13C" = bigger pump); a same-
    nozzle move is flagged on the throat letter (A < B < C …). Unparseable
    labels degrade to plain "new pump" rather than guessing a direction.
    """
    if not (cur_nz and cur_th and opt_nz and opt_th):
        return "—"
    if (cur_nz, cur_th) == (opt_nz, opt_th):
        return "same"
    try:
        n_cur, n_opt = int(str(cur_nz).strip()), int(str(opt_nz).strip())
    except (TypeError, ValueError):
        return "new pump"
    if n_opt > n_cur:
        return "▲ bigger"
    if n_opt < n_cur:
        return "▼ smaller"
    t_cur, t_opt = str(cur_th).strip().upper(), str(opt_th).strip().upper()
    if t_cur and t_opt and t_cur[0].isalpha() and t_opt[0].isalpha():
        return "▲ throat" if t_opt[0] > t_cur[0] else "▼ throat"
    return "new pump"


def build_comparison_rows(
    results,
    active: dict,
    si_wells: list[str],
    test_rates: dict,
    matchcheck_rows: Optional[dict] = None,
    header_psi: Optional[float] = None,
) -> list[dict]:
    """Per-well rows for the Results "current vs optimized" table.

    Pure (no Streamlit) so the table logic is unit-testable. ``results`` are
    the optimizer's :class:`OptimizationResult` rows; ``active`` the review
    store's active entries (``review_nozzle``/``review_throat`` = the pump in
    the hole, captured at review); ``si_wells`` the active wells the run left
    out of the plan; ``test_rates`` ``{well: (oil, pf)}`` measured medians from
    :func:`recent_test_rates`; ``matchcheck_rows`` ``{well: row}`` from the
    configure-stage match check (supplies Model÷Test at the CURRENT pump);
    ``header_psi`` the run's settled PF surface pressure
    (``meta["header_psi"]``) — a single PAD-LEVEL value applied to every
    optimized row (there's no per-well optimized header pressure).

    "Current oil"/"Current PF" are MEASURED test rates while "Opt oil"/"Opt PF"
    are MODELED — so "Δ oil" mixes model vs measured. The Model÷Test column is
    the per-well trust signal for exactly that mix. Rows sort by |Δ oil|.

    "Current BHP" and "Current PF psi" are the REVIEWED store values (the
    same ``pwf``/``ppf_surf_well`` the well's IPR anchor was built from — the
    per-well engineer-verified state, distinct from the measured test-rate
    columns above). "Opt BHP" is the optimizer's solved pump-intake suction
    pressure at jpump TVD (replaces the old detail-only "Suction (psi)"
    column). "Opt PF psi" is the pad-level settled header (``header_psi``),
    repeated on every optimized row by design.
    """
    matchcheck_rows = matchcheck_rows or {}
    rows = []
    opt_pf_psi = round(header_psi) if header_psi is not None else None

    def _current_pump(well: str) -> str:
        e = active.get(well, {})
        nz, th = e.get("review_nozzle") or "", e.get("review_throat") or ""
        return f"{nz}{th}" if nz and th else "—"

    def _base(well: str) -> dict:
        oil, pf = test_rates.get(well) or (None, None)
        mc = matchcheck_rows.get(well) or {}
        ratio = mc.get("oil_ratio")
        e = active.get(well, {})
        cur_bhp = e.get("pwf")
        cur_pf_psi = e.get("ppf_surf_well")
        return {
            "Well": well,
            "Current pump": _current_pump(well),
            "Current oil": round(oil) if oil else None,
            "Current PF": round(pf) if pf else None,
            "Current BHP": round(cur_bhp) if cur_bhp else None,
            "Current PF psi": round(cur_pf_psi) if cur_pf_psi else None,
            "Model÷Test": round(ratio, 2) if ratio else None,
        }

    for r in results:
        row = _base(r.well_name)
        opt_pump = f"{r.recommended_nozzle}{r.recommended_throat}"
        cur_oil = row["Current oil"] or 0
        e = active.get(r.well_name, {})
        row.update(
            {
                "Optimized pump": opt_pump,
                "Opt oil": round(r.predicted_oil_rate),
                "Opt PF": round(r.predicted_lift_water),
                "Opt BHP": round(r.suction_pressure),
                "Opt PF psi": opt_pf_psi,
                "Δ oil": round(r.predicted_oil_rate) - cur_oil,
                "Change": pump_size_change(
                    e.get("review_nozzle"),
                    e.get("review_throat"),
                    r.recommended_nozzle,
                    r.recommended_throat,
                ),
                "Status": "⚠ sonic" if r.sonic_status else "run",
                # detail columns (expander / CSV only)
                "Form. water (BPD)": round(r.predicted_formation_water),
                "Total WC": (
                    round(r.total_watercut, 3) if r.total_watercut is not None else None
                ),
            }
        )
        rows.append(row)

    for w in si_wells:
        row = _base(w)
        row.update(
            {
                "Optimized pump": "SHUT IN",
                "Opt oil": 0,
                "Opt PF": 0,
                "Opt BHP": None,
                "Opt PF psi": None,
                "Δ oil": -(row["Current oil"] or 0),
                "Change": "shut in",
                "Status": "SHUT IN",
                "Form. water (BPD)": 0,
                "Total WC": None,
            }
        )
        rows.append(row)

    rows.sort(key=lambda r: abs(r["Δ oil"]), reverse=True)
    return rows


def build_ipr_grid_figure(
    results,
    active: dict,
    si_wells: list[str],
):
    """Per-well IPR grid: the solver's IPR line + current vs optimized
    operating points, one subplot per well.

    Pure enough to unit test the returned figure's traces/annotations
    directly — the caller (``pad_page._render_results``) just wraps it in
    ``st.plotly_chart``. Panels: every optimized ``results`` well, plus every
    ``si_wells`` well that still has reviewed store data (``pwf``/``qwf``/
    ``res_pres`` — the same trio the well's IPR anchor was built from).
    Returns ``None`` when nobody in that combined set has the trio at all —
    there is nothing to draw.

    The IPR is drawn EXACTLY as the jet-pump solver evaluates it: the Vogel
    curve through the store anchor — see ``jetflow.py``'s
    ``ipr_su.oil_flow(psu, method="vogel")`` (restored ``ee3886e``; the
    woffl-2.0 sync had clobbered it back to the straight-line PI, see
    ``docs/upstream_sync.md`` #15). Because the solver now solves on this
    same Vogel curve, the "Optimized" marker lands ON it by construction.
    Guarded (skipped, never raised) per-panel when ``pwf >= res_pres``,
    ``res_pres <= 0``, ``qwf <= 0``, or a ``ZeroDivisionError`` out of
    ``vogel_qmax`` — a marginal/edge well still gets its operating-point
    markers even with no valid line.

    The "Optimized" marker's x is FORMATION liquid (oil + formation water) at
    the solved suction pressure — power-fluid/lift water is excluded because
    it never flows through the reservoir and would misrepresent the well's
    position on its own IPR. On the liquid axis the marker lands ON the Vogel
    curve by construction (the solver's oil rate is itself
    ``qmax * vogel_fraction(psu, res_pres)``, scaled by the same 1/(1-wc) as
    the curve). The pad-level settled header (``header_psi``) plays no role
    here: each optimized y is the well's own solved suction.
    """
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from woffl.gui.vogel import vogel_qmax, vogel_rate

    _CURRENT_COLOR = "#2ca02c"
    _OPT_COLOR = "#d62728"
    _CURVE_COLOR = "rgba(31, 119, 180, 0.55)"

    opt_by_well = {r.well_name: r for r in results}

    def _store(well: str):
        e = active.get(well, {})
        return e.get("pwf"), e.get("qwf"), e.get("res_pres")

    def _has_data(well: str) -> bool:
        pwf, qwf, res_pres = _store(well)
        return pwf is not None and qwf is not None and res_pres is not None

    def _cur_pump(well: str) -> str:
        e = active.get(well, {})
        nz, th = e.get("review_nozzle") or "", e.get("review_throat") or ""
        return f"{nz}{th}" if nz and th else "—"

    def _cur_oil(well: str, qwf) -> Optional[float]:
        """As-reviewed oil BOPD: the store's audit field, else liquid*(1-wc)."""
        e = active.get(well, {})
        oil = e.get("qwf_oil_review")
        if oil is not None and oil == oil:  # not NaN
            return float(oil)
        wc = e.get("form_wc")
        if qwf is not None and wc is not None and wc == wc:
            return float(qwf) * (1.0 - float(wc))
        return None

    si_with_data = [w for w in si_wells if _has_data(w)]
    wells = list(opt_by_well.keys()) + si_with_data
    if not wells or not any(_has_data(w) for w in wells):
        return None

    cols = int(np.ceil(np.sqrt(len(wells))))
    rows_n = int(np.ceil(len(wells) / cols))
    fig = make_subplots(
        rows=rows_n,
        cols=cols,
        subplot_titles=wells,
        vertical_spacing=0.16,
        horizontal_spacing=0.08,
        x_title="Total liquid (BPD)",
        y_title="BHP (psi)",
    )

    shown_current, shown_opt, shown_line = False, False, False
    for i, well in enumerate(wells):
        row_i, col_i = i // cols + 1, i % cols + 1
        pwf, qwf, res_pres = _store(well)
        r = opt_by_well.get(well)

        # (a) the solver's IPR: the Vogel curve through the store anchor
        # (jetflow evaluates oil_flow(psu, method="vogel") — restored
        # ee3886e — the optimized point sits on THIS curve).
        if (
            pwf is not None
            and qwf is not None
            and res_pres is not None
            and qwf > 0
            and res_pres > 0
            and pwf < res_pres
        ):
            try:
                qmax = vogel_qmax(qwf, pwf, res_pres)
            except ZeroDivisionError:
                qmax = None
            if qmax is not None:
                pwf_grid = np.linspace(res_pres, 0.0, 40)
                liq_curve = vogel_rate(pwf_grid, qmax, res_pres)
                fig.add_trace(
                    go.Scatter(
                        x=liq_curve,
                        y=pwf_grid,
                        mode="lines",
                        line=dict(color=_CURVE_COLOR, width=1.5),
                        name="IPR (Vogel)",
                        legendgroup="ipr",
                        showlegend=not shown_line,
                        hoverinfo="skip",
                    ),
                    row=row_i,
                    col=col_i,
                )
                shown_line = True

        # (b) current operating point (always plotted if we have it, even
        # when the curve above was skipped).
        have_current = qwf is not None and pwf is not None
        cur_oil = _cur_oil(well, qwf)
        if have_current:
            oil_line = f"<br>Oil {cur_oil:.0f} BOPD" if cur_oil is not None else ""
            fig.add_trace(
                go.Scatter(
                    x=[qwf],
                    y=[pwf],
                    mode="markers",
                    marker=dict(color=_CURRENT_COLOR, size=11, symbol="circle"),
                    name="Current",
                    legendgroup="current",
                    showlegend=not shown_current,
                    hovertemplate=(
                        f"{well} · {_cur_pump(well)}<br>Liquid %{{x:.0f}} BPD"
                        f"{oil_line}<br>BHP %{{y:.0f}} psi<extra></extra>"
                    ),
                ),
                row=row_i,
                col=col_i,
            )
            shown_current = True

        # (c) optimized operating point — formation liquid (oil + formation
        # water), NOT total liquid (lift/PF water excluded).
        opt_pump_label = "SHUT IN"
        opt_x = opt_y = None
        have_opt = r is not None
        if have_opt:
            opt_pump_label = f"{r.recommended_nozzle}{r.recommended_throat}"
            opt_x = r.predicted_oil_rate + r.predicted_formation_water
            opt_y = r.suction_pressure
            fig.add_trace(
                go.Scatter(
                    x=[opt_x],
                    y=[opt_y],
                    mode="markers",
                    marker=dict(color=_OPT_COLOR, size=13, symbol="star"),
                    name="Optimized",
                    legendgroup="optimized",
                    showlegend=not shown_opt,
                    hovertemplate=(
                        f"{well} · {opt_pump_label}<br>Liquid %{{x:.0f}} BPD"
                        f"<br>Oil {r.predicted_oil_rate:.0f} BOPD"
                        f"<br>BHP %{{y:.0f}} psi<extra></extra>"
                    ),
                ),
                row=row_i,
                col=col_i,
            )
            shown_opt = True

        # (d) dashed connector, current -> optimized (both points present)
        if have_current and have_opt:
            fig.add_trace(
                go.Scatter(
                    x=[qwf, opt_x],
                    y=[pwf, opt_y],
                    mode="lines",
                    line=dict(color="#888888", width=1, dash="dash"),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row_i,
                col=col_i,
            )

        # (e) per-panel pump-change annotation, top-right inside the axes:
        # pump change on line 1, oil rates current -> optimized on line 2.
        oil_from = f"{cur_oil:.0f}" if cur_oil is not None else "—"
        oil_to = f"{r.predicted_oil_rate:.0f}" if have_opt else "0"
        fig.add_annotation(
            text=(
                f"{_cur_pump(well)} → {opt_pump_label}"
                f"<br>{oil_from} → {oil_to} bopd"
            ),
            xref="x domain",
            yref="y domain",
            x=0.98,
            y=0.94,
            xanchor="right",
            yanchor="top",
            showarrow=False,
            font=dict(size=9),
            bgcolor="rgba(255,255,255,0.65)",
            row=row_i,
            col=col_i,
        )

    fig.update_layout(
        height=max(360, rows_n * 320),
        margin=dict(l=60, r=20, t=80, b=60),
        title_text="Per-well IPR — current vs optimized operating point",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def recent_test_rates(well: str, n_recent: int = 5):
    """(oil BOPD, pf BPD) — the MEDIAN of the well's recent valid tests.

    Robust to a single bad recent test: a low / shut-in / unallocated outlier
    won't drag the value down the way the single latest row did (e.g. S-03's bad
    last test with much higher priors, or MPS-69 reading 0). Takes up to
    ``n_recent`` most-recent tests with a positive value; oil and PF are taken
    independently. (None, None) when there are no valid tests.
    """
    import pandas as pd

    from woffl.gui.utils import get_well_tests_for_well

    try:
        t = get_well_tests_for_well(well)
        if t is None or t.empty:
            return None, None
        t = t.sort_values("WtDate", ascending=False)
        oils = (
            pd.to_numeric(t["WtOilVol"], errors="coerce")
            if "WtOilVol" in t.columns
            else pd.Series(dtype=float)
        )
        oils = oils[oils > 0].head(n_recent)
        pfs = (
            pd.to_numeric(t["lift_wat"], errors="coerce")
            if "lift_wat" in t.columns
            else pd.Series(dtype=float)
        )
        pfs = pfs[pfs > 0].head(n_recent)
        oil = float(oils.median()) if not oils.empty else None
        pf = float(pfs.median()) if not pfs.empty else None
        return oil, pf
    except Exception:
        return None, None
