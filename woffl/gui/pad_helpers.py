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
) -> list[dict]:
    """Per-well rows for the Results "current vs optimized" table.

    Pure (no Streamlit) so the table logic is unit-testable. ``results`` are
    the optimizer's :class:`OptimizationResult` rows; ``active`` the review
    store's active entries (``review_nozzle``/``review_throat`` = the pump in
    the hole, captured at review); ``si_wells`` the active wells the run left
    out of the plan; ``test_rates`` ``{well: (oil, pf)}`` measured medians from
    :func:`recent_test_rates`; ``matchcheck_rows`` ``{well: row}`` from the
    configure-stage match check (supplies Model÷Test at the CURRENT pump).

    "Current oil"/"Current PF" are MEASURED test rates while "Opt oil"/"Opt PF"
    are MODELED — so "Δ oil" mixes model vs measured. The Model÷Test column is
    the per-well trust signal for exactly that mix. Rows sort by |Δ oil|.
    """
    matchcheck_rows = matchcheck_rows or {}
    rows = []

    def _current_pump(well: str) -> str:
        e = active.get(well, {})
        nz, th = e.get("review_nozzle") or "", e.get("review_throat") or ""
        return f"{nz}{th}" if nz and th else "—"

    def _base(well: str) -> dict:
        oil, pf = test_rates.get(well) or (None, None)
        mc = matchcheck_rows.get(well) or {}
        ratio = mc.get("oil_ratio")
        return {
            "Well": well,
            "Current pump": _current_pump(well),
            "Current oil": round(oil) if oil else None,
            "Current PF": round(pf) if pf else None,
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
                "Suction (psi)": round(r.suction_pressure),
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
                "Δ oil": -(row["Current oil"] or 0),
                "Change": "shut in",
                "Status": "SHUT IN",
                "Form. water (BPD)": 0,
                "Total WC": None,
                "Suction (psi)": None,
            }
        )
        rows.append(row)

    rows.sort(key=lambda r: abs(r["Δ oil"]), reverse=True)
    return rows


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
