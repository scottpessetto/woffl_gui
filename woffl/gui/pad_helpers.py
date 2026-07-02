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
