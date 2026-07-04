"""Unified pad-optimization page (R-1 Phase C) — one render path, N pads.

The S/I/M pad pages were ~80% triplicated render code over the shared compute
core (:mod:`woffl.gui.pad_optimize`, Phase B) and the shared plant interface
(:mod:`woffl.gui.pad_plant_base`, Phase A). This module is the single copy of
the 3-stage flow (Review Wells → Configure & Run → Results, including the
match check, run, results accounting, and the scenario comparator), driven by
a :class:`PadSpec`. The pad modules (``s_pad_page`` / ``i_pad_page`` /
``m_pad_page``) shrink to their spec + entry function + the genuinely
pad-specific render hooks (curve/frontier plots, amps tables, thrust/recirc
warnings).

Every session-state and widget key is derived from ``spec.prefix`` exactly as
the pages spelled them (``sp_page_stage``, ``ip_run``, ``mp_scn_{well}_{n}``,
…), so the refactor is invisible to a live session.

HOW TO ADD A NEW PAD
--------------------
Create ``woffl/gui/x_pad_page.py`` with a :class:`PadSpec` and a no-arg entry
function, then add an app mode in ``app.py`` that calls it::

    from woffl.gui.pad_page import PadSpec, run_pad_page
    from woffl.gui.pad_plant_base import FixedHeaderPlant

    SPEC = PadSpec(
        pad="X",                     # review-store pad letter (well prefix MPX-)
        prefix="xp",                 # unique session/widget key prefix
        plant=FixedHeaderPlant(3200.0),   # or a real PadPlant subclass
        title="X-Pad Optimization 🛢",
        subtitle="Review each X-Pad well, then optimize the pad.",
        configure_caption="Delivered PF pressure is fixed at 3,200 psi.",
    )

    def run_x_pad_page() -> None:
        run_pad_page(SPEC)

That is a complete, working pad page: ``FixedHeaderPlant`` gives a constant
delivered header and an unbounded PF budget for pads with no booster model.
Everything else on the spec is optional — plant physics plots, per-pad
metrics, and warnings are added via the hook fields as the pad's booster
model matures (see the S/I/M modules for worked examples of every hook).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

from woffl.gui import pad_optimize
from woffl.gui.pad_helpers import parse_pump as _parse_pump
from woffl.gui.pad_helpers import recent_test_rates as _recent_test_rates
from woffl.gui.pad_helpers import (
    render_results_accounting as _render_results_accounting,
)
from woffl.gui.pad_plant_base import PadPlant
from woffl.gui.workflow_steps import well_review_store as wrs
from woffl.gui.workflow_steps.step_review_wells import render_review_stage, store_for

_STAGES = ["1 · Review Wells", "2 · Configure & Run", "3 · Results"]


@dataclass(frozen=True)
class PadSpec:
    """Everything pad-specific about a pad-optimization page.

    The required fields are the identity + the page copy that has no sensible
    generic default. Every other field defaults to the shared wording the
    S/I pages used, so a minimal spec (see the module docstring) renders a
    complete page. Callable fields are render hooks that stay in the pad's
    own module — they receive plain dicts (run/scenario meta) and draw with
    Streamlit themselves.
    """

    # -- identity ----------------------------------------------------------
    pad: str  # review-store pad letter ("S")
    prefix: str  # session/widget key prefix ("sp"); must be unique per pad
    plant: PadPlant  # booster model; plant.coupling picks the solve strategy

    # -- entry -------------------------------------------------------------
    title: str
    subtitle: str

    # -- configure stage ---------------------------------------------------
    configure_caption: str  # how this pad's boosters couple to the wells
    no_active_warning: str = (
        "No active reviewed wells. Go back to **Review Wells** "
        "(every well may be marked offline)."
    )
    n_pump_options: Optional[tuple[int, ...]] = None  # None = no pump-count radio
    n_pumps_label: str = "Booster pumps online"
    n_pumps_help: Optional[str] = None
    marginal_wc_help: Optional[str] = None
    flow_caption: Optional[Callable[[Optional[int]], str]] = None  # under multiselects
    plot_heading: Optional[str] = None  # "##### ..." above the capability plot
    render_plot: Optional[Callable[[Optional[int], Optional[list]], None]] = None
    matchcheck_caption: str = (
        "Before optimizing, model each well at its current pump + chosen IPR and "
        "compare to its recent tests (median). Fix the ✗ wells first."
    )
    matchcheck_note: str = "⚠ = off, ✓ = match. Worst matches first."
    run_bar_text: str = "Solving pump-curve coupling…"
    run_spinner_text: str = (
        "Running batch sweeps + optimization (this can take a minute)…"
    )
    n_steps: int = 11  # free_pressure sweep resolution (unused for fixed_curve)

    # -- results stage -----------------------------------------------------
    station_metric3: Optional[Callable[[dict], tuple[str, str]]] = None
    render_station_extras: Optional[Callable[[dict], None]] = None  # amps tables etc.
    results_plot_heading: Optional[str] = None  # "#### ..." above the results plot
    sweep_expander_label: str = "Pressure sweep explored"

    # -- scenario comparator -------------------------------------------------
    comparator_caption: str = (
        "Pick a pump (or shut-in) per well and see the resulting oil + header "
        "pressure next to a baseline."
    )
    cmp_base_help: str = (
        "Existing = each well's CURRENT measured oil/PF from recent tests. "
        "Optimized = the optimizer's picks."
    )
    scn_preset_help: Optional[str] = (
        "Presets use every well's current or optimized pump; Custom reads the "
        "per-well picks below."
    )
    custom_expander_label: str = "Per-well pump selection (edit individual wells)"
    filled_from_current_label: str = "latest installed pump"
    scenario_spinner_text: str = "Running scenario (coupled to the frontier)…"
    render_scenario_warnings: Optional[Callable[[dict, Optional[int]], None]] = None
    show_per_pump: bool = False  # per-pump flow row/metric (parallel fixed-speed pads)
    subs_caption: str = "⚠ Chosen pump infeasible — substituted a feasible one: "
    star_caption: str = (
        "★ Model couldn't solve — estimated from recent-test rate × the "
        "average ripple of the wells that did solve: "
    )
    infeas_caption: str = "✗ No feasible pump at all (counted as 0): "
    where_markdown: Optional[str] = None  # "**Where each option ...**" plot intro
    bias_note: str = " **Model÷Test** = model ÷ measured (>1 = loose IPR match)."
    # str.format template; placeholders: {base} (baseline label, lowercase),
    # {h0} / {h1} (baseline / scenario header psi — format specs allowed)
    delta_caption: str = (
        "**Δ oil** = per-well change ({base} → scenario). ★ = estimated. "
        "Sorted by biggest mover."
    )


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def _render_review(spec: PadSpec) -> None:
    render_review_stage(spec.pad)
    store = store_for(spec.pad)
    active = wrs.active_entries(store)
    st.divider()
    if active:
        n_off = len(store) - len(active)
        st.success(
            f"{len(active)} well(s) ready for optimization"
            + (f"  ·  {n_off} offline (excluded)" if n_off else "")
        )
        if st.button(
            "Next: Configure & Run →", type="primary", key=f"{spec.prefix}_to_config"
        ):
            st.session_state[f"{spec.prefix}_page_stage"] = 1
            st.rerun()
    elif store:
        st.warning(
            "Every reviewed well is marked offline — bring at least one online to continue."
        )
    else:
        st.info("Review at least one well (or add a hypothetical) to continue.")


def _render_configure(spec: PadSpec) -> None:
    from woffl.gui.params import NOZZLE_OPTIONS, THROAT_OPTIONS

    p = spec.prefix
    store = store_for(spec.pad)
    active = wrs.active_entries(store)
    if not active:
        st.warning(spec.no_active_warning)
        return

    offline = [w for w in store if store[w].get("offline")]
    st.markdown(
        f"**{len(active)} active well(s)** to optimize: " + ", ".join(active.keys())
    )
    if offline:
        st.caption("Offline (excluded): " + ", ".join(offline))
    st.caption(spec.configure_caption)

    if spec.n_pump_options:
        c1, c2, c3 = st.columns(3)
        with c1:
            n_pumps = st.radio(
                spec.n_pumps_label,
                list(spec.n_pump_options),
                horizontal=True,
                key=f"{p}_n_pumps",
                help=spec.n_pumps_help,
            )
        method_col, wc_col = c2, c3
    else:
        n_pumps = None  # fixed train — no pump-count choice
        method_col, wc_col = st.columns(2)
    with method_col:
        method = st.selectbox("Optimizer", ["milp", "mckp"], key=f"{p}_method")
    with wc_col:
        marginal_wc = st.number_input(
            "Marginal water cut",
            0.0,
            1.0,
            1.0,
            0.01,
            format="%.2f",
            key=f"{p}_marginal_wc",
            help=spec.marginal_wc_help,
        )

    c4, c5 = st.columns(2)
    with c4:
        nozzles = st.multiselect(
            "Nozzle sizes to test",
            NOZZLE_OPTIONS,
            default=["9", "10", "11", "12", "13", "14", "15"],
            key=f"{p}_nozzles",
        )
    with c5:
        throats = st.multiselect(
            "Throat ratios to test",
            THROAT_OPTIONS,
            default=["A", "B", "C", "D"],
            key=f"{p}_throats",
        )

    if spec.flow_caption is not None:
        st.caption(spec.flow_caption(n_pumps))

    # Where the pad sits on its capability curve/frontier. Before the first
    # run it shows just the curve; once a run for this pump count exists, the
    # last operating point is overlaid.
    if spec.render_plot is not None:
        if spec.plot_heading:
            st.markdown(spec.plot_heading)
        last_meta = st.session_state.get(f"{p}_run_meta")
        op = (
            [
                {
                    "label": "Last run",
                    "total_pf_bpd": last_meta["total_pf_bpd"],
                    "header_psi": last_meta["header_psi"],
                    "color": "#d62728",
                }
            ]
            if last_meta and last_meta.get("n_pumps") == n_pumps
            else None
        )
        if op:
            st.caption("✕ marks the operating point from the most recent run.")
        spec.render_plot(n_pumps, op)

    # ── Pre-flight: model-vs-test match check ────────────────────────────
    st.divider()
    st.markdown("##### Pre-flight: model match check")
    st.caption(spec.matchcheck_caption)
    current = {
        w: (active[w].get("review_nozzle") or "", active[w].get("review_throat") or "")
        for w in active
    }
    # Auto-run on landing; re-runs only when the config changes (pump count,
    # well set, current pumps, or IPRs) — keyed on a lightweight signature so
    # it doesn't re-compute on every interaction.
    sig = (
        n_pumps,
        tuple(
            sorted(
                (
                    w,
                    current[w][0],
                    current[w][1],
                    int(active[w].get("qwf", 0)),
                    int(active[w].get("pwf", 0)),
                    int(active[w].get("res_pres", 0)),
                )
                for w in active
            )
        ),
    )
    mc = st.session_state.get(f"{p}_matchcheck")
    refresh = st.button("↻ Re-run check", key=f"{p}_matchcheck_run")
    if refresh or not mc or mc.get("sig") != sig:
        cur_ch = {
            w: (current[w] if current[w][0] and current[w][1] else None) for w in active
        }
        tr = {w: _recent_test_rates(w) for w in active}
        try:
            with st.spinner("Modeling each well at its current pump…"):
                rows, hdr = pad_optimize.match_check(
                    wrs.store_to_well_configs(active),
                    spec.plant,
                    n_pumps,
                    cur_ch,
                    tr,
                )
            mc = {"rows": rows, "header": hdr, "sig": sig}
            st.session_state[f"{p}_matchcheck"] = mc
        except Exception as e:  # never block configuring/running on a check failure
            st.caption(f"Match check unavailable: {e}")
            mc = None

    if mc:
        import pandas as pd

        rows = mc["rows"]
        n_oil = sum(1 for r in rows if r["oil_flag"].startswith("✗"))
        n_pf = sum(1 for r in rows if r["pf_flag"].startswith("✗"))
        st.caption(
            f"Modeled at header ≈ {mc['header']:,.0f} psi.  "
            f"**{n_oil} oil mismatch · {n_pf} PF bust** (✗). " + spec.matchcheck_note
        )

        def _dev(r):
            ds = [abs(x - 1) for x in (r["oil_ratio"], r["pf_ratio"]) if x is not None]
            return max(ds) if ds else 0.0

        df = pd.DataFrame(
            [
                {
                    "Well": r["well"],
                    "Pump": r["pump"],
                    "Test oil": round(r["test_oil"]) if r["test_oil"] else None,
                    "Model oil": (
                        round(r["model_oil"]) if r["model_oil"] is not None else None
                    ),
                    "Oil×": round(r["oil_ratio"], 2) if r["oil_ratio"] else None,
                    "Oil match": r["oil_flag"],
                    "Test PF": round(r["test_pf"]) if r["test_pf"] else None,
                    "Model PF": (
                        round(r["model_pf"]) if r["model_pf"] is not None else None
                    ),
                    "PF×": round(r["pf_ratio"], 2) if r["pf_ratio"] else None,
                    "PF match": r["pf_flag"],
                }
                for r in sorted(rows, key=_dev, reverse=True)
            ]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)
    st.divider()

    if not nozzles or not throats:
        st.warning("Pick at least one nozzle and one throat.")
        return

    if st.button("▶ Run optimization", type="primary", key=f"{p}_run"):
        well_configs = wrs.store_to_well_configs(active)
        bar = st.progress(0.0, text=spec.run_bar_text)

        if spec.plant.coupling == "fixed_curve":

            def _progress(it, max_it, ppf, total_pf, new_ppf):
                bar.progress(
                    min(it / max_it, 1.0),
                    text=f"Iter {it}: header {ppf:,.0f} psi → {total_pf:,.0f} BPD "
                    f"→ curve {new_ppf:,.0f} psi",
                )

        else:

            def _progress(it, max_it, P, total_pf, total_oil):
                bar.progress(
                    min(it / max_it, 1.0),
                    text=f"Step {it}/{max_it}: header {P:,.0f} psi → "
                    f"{total_pf:,.0f} BPD PF → {total_oil:,.0f} BOPD",
                )

        try:
            with st.spinner(spec.run_spinner_text):
                results, _optimizer, meta = pad_optimize.run_optimization(
                    well_configs,
                    spec.plant,
                    n_pumps,
                    nozzles,
                    throats,
                    method,
                    marginal_wc,
                    n_steps=spec.n_steps,
                    progress=_progress,
                )
        except Exception as e:
            # A single bad entry (e.g. a hand-edited CSV field) must show an
            # error, not blow the page with a raw traceback (P0 fix, kept).
            bar.empty()
            st.error(f"Optimization failed: {e}")
            return
        bar.empty()
        # Stamp what this run was computed FROM, so Results can flag staleness
        # when wells are added/edited/toggled afterwards; meta already carries
        # the reconciliation (per-well drop reasons) from pad_optimize.
        meta["store_sig"] = wrs.store_signature(active)
        st.session_state[f"{p}_opt_results"] = results
        st.session_state[f"{p}_run_meta"] = meta
        # A fresh optimization invalidates any cached scenario / existing
        # baseline (union of the legacy pages' pop lists — the baseline keys
        # are S-only legacy, harmless no-ops elsewhere).
        for k in (
            f"{p}_scenario_meta",
            f"{p}_scenario_per_well",
            f"{p}_baseline_meta",
            f"{p}_baseline_per_well",
            f"{p}_scn_base_used",
        ):
            st.session_state.pop(k, None)
        st.session_state[f"{p}_page_stage"] = 2
        st.rerun()


def _download_results(spec: PadSpec, df) -> None:
    """Single-click CSV auto-download (per repo rule: no two-step downloads)."""
    if st.button("⬇ Download results CSV", key=f"{spec.prefix}_res_dl"):
        from woffl.gui.components.download import autodownload

        autodownload(
            df.to_csv(index=False).encode("utf-8"),
            f"{spec.pad}_pad_optimization_results.csv",
            "text/csv",
            f"{spec.prefix}_res_auto_dl",
        )


def _render_results(spec: PadSpec) -> None:
    import pandas as pd

    p = spec.prefix
    results = st.session_state.get(f"{p}_opt_results")
    meta = st.session_state.get(f"{p}_run_meta")
    if not results or not meta:
        st.warning("No results yet. Go to **Configure & Run**.")
        return

    store = store_for(spec.pad)
    active = wrs.active_entries(store)
    n_pumps = meta["n_pumps"]
    opt_choice = {
        r.well_name: (r.recommended_nozzle, r.recommended_throat) for r in results
    }
    # Active wells NOT in the results: solver shut-in, failed simulation,
    # marginal-WC exclusion — or added after the run. The accounting box
    # renders the per-well reason; the banner flags store drift.
    si_wells = [w for w in active if w not in opt_choice]
    _render_results_accounting(meta, active, si_wells)

    # Station operating point
    st.markdown("#### Station operating point")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Header pressure", f"{meta['header_psi']:,.0f} psi")
    m2.metric("Total power fluid", f"{meta['total_pf_bpd']:,.0f} BPD")
    if spec.station_metric3 is not None:
        label, value = spec.station_metric3(meta)
        m3.metric(label, value)
    m4.metric("Total oil", f"{meta['total_oil_bopd']:,.0f} BOPD")

    if spec.render_station_extras is not None:
        spec.render_station_extras(meta)

    # Per-well table — includes SHUT IN wells so the SI decision is visible.
    st.markdown("#### Per-well recommendations")
    rows = []
    for r in results:
        rows.append(
            {
                "Well": r.well_name,
                "Nozzle": r.recommended_nozzle,
                "Throat": r.recommended_throat,
                "Oil (BOPD)": round(r.predicted_oil_rate, 0),
                "Power fluid (BPD)": round(r.predicted_lift_water, 0),
                "Form. water (BPD)": round(r.predicted_formation_water, 0),
                "Total WC": round(r.total_watercut, 3),
                "Suction (psi)": round(r.suction_pressure, 0),
                "Status": "⚠ sonic" if r.sonic_status else "run",
            }
        )
    for w in si_wells:
        rows.append(
            {
                "Well": w,
                "Nozzle": "—",
                "Throat": "—",
                "Oil (BOPD)": 0,
                "Power fluid (BPD)": 0,
                "Form. water (BPD)": 0,
                "Total WC": None,
                "Suction (psi)": None,
                "Status": "SHUT IN",
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    op_points = [
        {
            "label": "Optimized",
            "total_pf_bpd": meta["total_pf_bpd"],
            "header_psi": meta["header_psi"],
            "color": "#d62728",
        }
    ]
    if spec.plant.coupling == "fixed_curve":
        # fixed-point solve: download next to the iteration trace, plot below
        c1, c2 = st.columns(2)
        with c1:
            _download_results(spec, df)
        with c2:
            with st.expander("Pump-curve coupling iterations"):
                st.dataframe(
                    pd.DataFrame(meta["history"]),
                    use_container_width=True,
                    hide_index=True,
                )
        if spec.render_plot is not None:
            if spec.results_plot_heading:
                st.markdown(spec.results_plot_heading)
            spec.render_plot(n_pumps, op_points)
    else:
        # pressure sweep: download, frontier plot, then the sweep explored
        _download_results(spec, df)
        if spec.render_plot is not None:
            if spec.results_plot_heading:
                st.markdown(spec.results_plot_heading)
            spec.render_plot(n_pumps, op_points)
        with st.expander(spec.sweep_expander_label):
            st.dataframe(
                pd.DataFrame(meta["sweep"]), use_container_width=True, hide_index=True
            )

    _render_scenario_comparator(spec, results, meta, active)


# ---------------------------------------------------------------------------
# Scenario comparator
# ---------------------------------------------------------------------------


def _render_scenario_comparator(spec: PadSpec, results, meta, active) -> None:
    import pandas as pd

    p = spec.prefix
    n = meta["n_pumps"]
    opt_choice = {
        r.well_name: (r.recommended_nozzle, r.recommended_throat) for r in results
    }
    opt_oil = {r.well_name: r.predicted_oil_rate for r in results}

    st.divider()
    st.markdown("#### Compare a manual scenario")
    st.caption(spec.comparator_caption)

    base_choice = st.radio(
        "Compare against",
        ["Existing (recent test rates)", "Optimized"],
        horizontal=True,
        key=f"{p}_cmp_base",
        help=spec.cmp_base_help,
    )
    is_existing = base_choice.startswith("Existing")
    # Switching the baseline changes how non-solving wells are handled, so a
    # scenario computed under the other baseline is stale — drop it.
    if st.session_state.get(f"{p}_scn_base_used") not in (None, base_choice):
        st.session_state.pop(f"{p}_scenario_meta", None)
        st.session_state.pop(f"{p}_scenario_per_well", None)
    st.session_state[f"{p}_scn_base_used"] = base_choice

    combos = [
        f"{nz}{t}" for nz in meta.get("nozzles", []) for t in meta.get("throats", [])
    ]
    current = {
        w: (active[w].get("review_nozzle") or "", active[w].get("review_throat") or "")
        for w in active
    }
    cur_labels = [f"{nz}{t}" for (nz, t) in current.values() if nz and t]
    pump_options = sorted(set(combos + cur_labels)) + ["Shut in"]

    preset = st.radio(
        "Scenario",
        ["All current pumps", "All optimized (baseline)", "Custom per-well"],
        horizontal=True,
        key=f"{p}_scn_preset",
        help=spec.scn_preset_help,
    )
    if preset == "Custom per-well":
        # Autofill basis for the per-well pickers. The selectbox keys carry a
        # nonce so an autofill click re-initializes them to the new defaults
        # (set logical default + bump nonce = fresh widgets that re-read index);
        # individual edits then persist until the next autofill.
        basis = st.session_state.get(f"{p}_scn_basis", "optimized")
        nonce = st.session_state.get(f"{p}_scn_nonce", 0)
        bc = st.columns([1.4, 1.4, 3])
        if bc[0].button("⤓ Fill all from latest installed", key=f"{p}_scn_fill_cur"):
            st.session_state[f"{p}_scn_basis"] = "current"
            st.session_state[f"{p}_scn_nonce"] = nonce + 1
            st.rerun()
        if bc[1].button("⤓ Fill all from optimized", key=f"{p}_scn_fill_opt"):
            st.session_state[f"{p}_scn_basis"] = "optimized"
            st.session_state[f"{p}_scn_nonce"] = nonce + 1
            st.rerun()
        bc[2].caption(
            f"Filled from **{spec.filled_from_current_label if basis == 'current' else 'optimized'}** "
            "— spot-edit any well below."
        )
        with st.expander(spec.custom_expander_label, expanded=True):
            cols = st.columns(3)
            for i, w in enumerate(sorted(active)):
                src = current.get(w) if basis == "current" else opt_choice.get(w)
                d_lbl = (
                    f"{src[0]}{src[1]}" if (src and src[0] and src[1]) else "Shut in"
                )
                if d_lbl not in pump_options:
                    d_lbl = pump_options[0]
                with cols[i % 3]:
                    st.selectbox(
                        w,
                        pump_options,
                        index=pump_options.index(d_lbl),
                        key=f"{p}_scn_{w}_{nonce}",
                    )

    if st.button("▶ Compute scenario & compare", type="primary", key=f"{p}_scn_run"):
        choices = {}
        for w in active:
            if preset == "All current pumps":
                nt = current.get(w)
                choices[w] = nt if (nt and nt[0] and nt[1]) else None
            elif preset == "All optimized (baseline)":
                choices[w] = opt_choice.get(w)  # None => SI (matches optimizer)
            else:
                _nonce = st.session_state.get(f"{p}_scn_nonce", 0)
                choices[w] = _parse_pump(st.session_state.get(f"{p}_scn_{w}_{_nonce}"))
        well_configs = wrs.store_to_well_configs(active)
        bar = st.progress(0.0, text="Solving scenario…")

        def _prog(it, mx, ppf, tpf, _npf):
            bar.progress(
                min(it / mx, 1.0),
                text=f"Iter {it}: header {ppf:,.0f} psi → {tpf:,.0f} BPD",
            )

        try:
            with st.spinner(spec.scenario_spinner_text):
                if is_existing:
                    cur_ch = {
                        w: (current[w] if current[w][0] and current[w][1] else None)
                        for w in active
                    }
                    tr = {w: _recent_test_rates(w) for w in active}
                    per_well, scn_meta = pad_optimize.evaluate_existing_scenario(
                        well_configs,
                        spec.plant,
                        n,
                        choices,
                        cur_ch,
                        test_rates=tr,
                        progress=_prog,
                    )
                else:
                    per_well, scn_meta = pad_optimize.evaluate_fixed_scenario(
                        well_configs,
                        spec.plant,
                        n,
                        choices,
                        fallback_choices=dict(opt_choice),
                        progress=_prog,
                    )
        except Exception as e:
            bar.empty()
            st.error(f"Scenario failed: {e}")
            return
        bar.empty()
        st.session_state[f"{p}_scenario_per_well"] = per_well
        st.session_state[f"{p}_scenario_meta"] = scn_meta
        st.rerun()

    # Baseline: "Existing" (each well's measured latest-test oil/PF — the
    # natural current-state reference) or "Optimized" (the optimizer's picks).
    base_label = "Optimized"
    base_meta = meta
    base_oil = dict(opt_oil)
    base_pf_map = None
    base_pump = {
        w: (f"{opt_choice[w][0]}{opt_choice[w][1]}" if opt_choice.get(w) else "SHUT IN")
        for w in active
    }
    if is_existing:
        base_label = "Current"
        rates = {w: _recent_test_rates(w) for w in active}
        base_oil = {w: (rates[w][0] or 0.0) for w in active}
        base_pf_map = {w: (rates[w][1] or 0.0) for w in active}
        base_pump = {
            w: (
                f"{current[w][0]}{current[w][1]}"
                if current[w][0] and current[w][1]
                else "—"
            )
            for w in active
        }
        tot_pf = sum(base_pf_map.values())
        hdr, _ = pad_optimize.settled_header(spec.plant, tot_pf, 0.0, n)
        base_meta = {
            "total_oil_bopd": sum(base_oil.values()),
            "total_pf_bpd": tot_pf,
            "header_psi": hdr,
        }
        if spec.show_per_pump:
            base_meta["per_pump_bpd"] = (tot_pf / n) if n else 0.0

    scn = st.session_state.get(f"{p}_scenario_meta")
    scn_pw = st.session_state.get(f"{p}_scenario_per_well")
    if scn and scn_pw:
        st.markdown(f"##### {base_label} vs your scenario")
        metrics = [
            "Total oil (BOPD)",
            "Total PF (BPD)",
            "Header PF pressure (psi)",
        ]
        base_vals = [
            base_meta["total_oil_bopd"],
            base_meta["total_pf_bpd"],
            base_meta["header_psi"],
        ]
        scn_vals = [
            scn["total_oil_bopd"],
            scn["total_pf_bpd"],
            scn["header_psi"],
        ]
        if spec.show_per_pump:
            metrics.append("Per-pump flow (BPD)")
            base_vals.append(base_meta["per_pump_bpd"])
            scn_vals.append(scn["per_pump_bpd"])
        cmp = pd.DataFrame(
            {"Metric": metrics, base_label: base_vals, "Your scenario": scn_vals}
        )
        cmp["Δ (scn − base)"] = cmp["Your scenario"] - cmp[base_label]
        for c in (base_label, "Your scenario", "Δ (scn − base)"):
            cmp[c] = cmp[c].round(0)
        st.dataframe(cmp, use_container_width=True, hide_index=True)

        d_oil = scn["total_oil_bopd"] - base_meta["total_oil_bopd"]
        if d_oil > 0:
            st.caption(
                f"Net: your scenario makes **{d_oil:,.0f} BOPD more** than {base_label.lower()}."
            )
        elif d_oil < 0:
            st.caption(
                f"Net: your scenario makes **{abs(d_oil):,.0f} BOPD less** than {base_label.lower()}."
            )
        if spec.render_scenario_warnings is not None:
            spec.render_scenario_warnings(scn, n)
        if scn.get("converged") is False:
            kind = "pump-curve" if spec.plant.coupling == "fixed_curve" else "frontier"
            st.warning(
                f"Scenario {kind} coupling did NOT converge in the iteration "
                "cap — the totals and header pressure are approximate; treat "
                "small deltas with caution."
            )

        subs = [
            r
            for r in scn_pw
            if r.get("note") and r["note"] not in ("infeasible", "star")
        ]
        starred = [r["well"] for r in scn_pw if r.get("note") == "star"]
        infeas = [r["well"] for r in scn_pw if r.get("note") == "infeasible"]
        if subs:
            st.caption(
                spec.subs_caption
                + ", ".join(f"{r['well']} ({r['pump']})" for r in subs)
            )
        if starred:
            st.caption(spec.star_caption + ", ".join(starred))
        if infeas:
            st.caption(spec.infeas_caption + ", ".join(infeas))

        if spec.render_plot is not None and spec.where_markdown:
            st.markdown(spec.where_markdown)
            spec.render_plot(
                n,
                [
                    {
                        "label": base_label,
                        "total_pf_bpd": base_meta["total_pf_bpd"],
                        "header_psi": base_meta["header_psi"],
                        "color": "#2ca02c",
                    },
                    {
                        "label": "Your scenario",
                        "total_pf_bpd": scn["total_pf_bpd"],
                        "header_psi": scn["header_psi"],
                        "color": "#d62728",
                    },
                ],
            )

        bias_note = spec.bias_note if is_existing else ""
        st.caption(
            spec.delta_caption.format(
                base=base_label.lower(),
                h0=base_meta["header_psi"],
                h1=scn["header_psi"],
            )
            + bias_note
        )
        scn_map = {r["well"]: r for r in scn_pw}
        mrows = []
        for w in sorted(active):
            srow = scn_map.get(w, {})
            b_oil = round(base_oil.get(w, 0))
            s_oil = round(srow.get("oil", 0))
            row = {
                "Well": w,
                f"{base_label} pump": base_pump.get(w, "—"),
                f"{base_label} oil": b_oil,
            }
            if base_pf_map is not None:
                row[f"{base_label} PF"] = round(base_pf_map.get(w, 0))
            row.update(
                {
                    "Scenario": srow.get("pump", "—"),
                    "Scn oil": s_oil,
                    "Δ oil": s_oil - b_oil,
                    "Scn PF": round(srow.get("pf", 0)),
                }
            )
            if base_pf_map is not None:  # Existing mode: show how off the model is
                bias = srow.get("bias")
                row["Model÷Test"] = round(bias, 2) if bias else None
            mrows.append(row)
        mdf = pd.DataFrame(mrows).sort_values(
            "Δ oil", key=lambda s: s.abs(), ascending=False
        )
        st.dataframe(mdf, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_pad_page(spec: PadSpec) -> None:
    st.title(spec.title)
    st.caption(spec.subtitle)

    stage_key = f"{spec.prefix}_page_stage"
    stage = st.session_state.setdefault(stage_key, 0)

    # Stage navigation. Forward stages stay locked until there's data for them.
    have_results = bool(st.session_state.get(f"{spec.prefix}_opt_results"))
    have_wells = bool(wrs.active_entries(store_for(spec.pad)))
    cols = st.columns(3)
    for i, label in enumerate(_STAGES):
        unlocked = i == 0 or (i == 1 and have_wells) or (i == 2 and have_results)
        with cols[i]:
            if i == stage:
                st.markdown(f"**:blue[{label}]**")
            elif unlocked:
                if st.button(
                    label, key=f"{spec.prefix}_nav_{i}", use_container_width=True
                ):
                    st.session_state[stage_key] = i
                    st.rerun()
            else:
                st.markdown(f":gray[{label}]")
    st.divider()

    if stage == 0:
        _render_review(spec)
    elif stage == 1:
        _render_configure(spec)
    else:
        _render_results(spec)
