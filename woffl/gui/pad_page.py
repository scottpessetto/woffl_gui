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
from woffl.gui.pad_helpers import build_comparison_rows as _build_comparison_rows
from woffl.gui.pad_helpers import build_ipr_grid_figure as _build_ipr_grid_figure
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


def _next_free_name(store: dict, base: str) -> str:
    """First unused '<base><n>' name — pure so it's testable."""
    n = 1
    while f"{base}{n}" in store:
        n += 1
    return f"{base}{n}"


def _next_placeholder_name(store: dict, src: str) -> str:
    """First unused '<src>-PH<n>' name — pure so it's testable."""
    return _next_free_name(store, f"{src}-PH")


def _render_placeholder_wells(spec: PadSpec, store: dict, active: dict) -> None:
    """Configure-screen placeholder / future wells.

    Two creation paths, both landing in the SAME review store flagged 🔵
    hypothetical (so they feed the optimizer, match check, PF what-if, and
    Base vs Future exactly like real wells, and staleness flagging covers
    adds/removes for free):

    * **Clone an existing well** — copies IPR, geometry, calibration, and
      review pump wholesale (``wrs.clone_entry``); for planned twins.
    * **New from scratch** — when the pad has NO analog: the engineer
      supplies IPR (ResP / oil / pwf / WC / GOR), geometry, and a pump
      (required for the fixed-pump tools), via ``wrs.hypothetical_entry``.
      Defaults to ⚫ offline — a FUTURE well ready for the Base vs Future
      picker, out of the base case and the optimization until brought online.
    """
    p = spec.prefix
    with st.expander("➕ Placeholder / future wells", expanded=False):
        tab_clone, tab_new = st.tabs(
            ["Clone an existing well", "New from scratch (no analog)"]
        )

        with tab_clone:
            st.caption(
                "Copy an existing reviewed well's IPR, geometry, and "
                "calibration — e.g. a planned twin of a producer."
            )
            sources = [w for w in active if not active[w].get("is_hypothetical")]
            if sources:
                c1, c2, c3 = st.columns([2, 2, 1])
                with c1:
                    src = st.selectbox("Clone from", sources, key=f"{p}_ph_src")
                with c2:
                    name = st.text_input(
                        "New well name",
                        value=_next_placeholder_name(store, src),
                        key=f"{p}_ph_name",
                    )
                with c3:
                    st.write("")  # aligns the button with the inputs
                    if st.button("Add", key=f"{p}_ph_add", use_container_width=True):
                        name = (name or "").strip()
                        if not name:
                            st.warning("Enter a name for the placeholder.")
                        elif name in store:
                            st.warning(f"{name} already exists in the review store.")
                        else:
                            store[name] = wrs.clone_entry(
                                store[src], name, source_well=src
                            )
                            # Drop the name widget's state so the next render
                            # suggests a fresh default instead of the used name.
                            st.session_state.pop(f"{p}_ph_name", None)
                            st.toast(
                                f"Added placeholder {name} (clone of {src})",
                                icon="🔵",
                            )
                            st.rerun()
            else:
                st.info("No reviewed real wells to clone yet.")

        with tab_new:
            from woffl.gui.params import NOZZLE_OPTIONS, THROAT_OPTIONS

            st.caption(
                "No analog on the pad? Specify the future well directly — "
                "expected IPR, geometry, and the planned pump. It starts "
                "⚫ offline (a FUTURE well): out of the base case and the "
                "optimization, pre-selected in *Base vs Future* below."
            )
            c1, c2, c3 = st.columns(3)
            with c1:
                nw_name = st.text_input(
                    "Well name",
                    value=_next_free_name(store, f"MP{spec.pad}-FUT"),
                    key=f"{p}_nw_name",
                )
                nw_fm = st.radio(
                    "Field model",
                    ["Schrader", "Kuparuk"],
                    horizontal=True,
                    key=f"{p}_nw_fm",
                )
                nw_resp = st.number_input(
                    "Reservoir P (psi)", 400, 5000, 1800, 10, key=f"{p}_nw_resp"
                )
            with c2:
                nw_oil = st.number_input(
                    "Expected oil (BOPD)", 0, 6000, 400, 10, key=f"{p}_nw_oil"
                )
                nw_pwf = st.number_input(
                    "Flowing BHP (psi)", 100, 2500, 900, 10, key=f"{p}_nw_pwf"
                )
                nw_wc = st.number_input(
                    "Water cut",
                    0.0,
                    0.99,
                    0.5,
                    0.01,
                    format="%.2f",
                    key=f"{p}_nw_wc",
                )
            with c3:
                nw_gor = st.number_input(
                    "GOR (scf/bbl)", 20, 10000, 250, 25, key=f"{p}_nw_gor"
                )
                nw_temp = st.number_input(
                    "Form temp (°F)", 32, 350, 160, 1, key=f"{p}_nw_temp"
                )
                nw_tvd = st.number_input(
                    "Jetpump TVD (ft)", 2500, 8000, 4200, 10, key=f"{p}_nw_tvd"
                )
            pc1, pc2, pc3, pc4 = st.columns([1, 1, 2, 1])
            with pc1:
                nw_noz = st.selectbox(
                    "Nozzle",
                    NOZZLE_OPTIONS,
                    index=NOZZLE_OPTIONS.index("12") if "12" in NOZZLE_OPTIONS else 0,
                    key=f"{p}_nw_noz",
                )
            with pc2:
                nw_thr = st.selectbox(
                    "Throat",
                    THROAT_OPTIONS,
                    index=THROAT_OPTIONS.index("B") if "B" in THROAT_OPTIONS else 0,
                    key=f"{p}_nw_thr",
                )
            with pc3:
                nw_off = st.checkbox(
                    "Start offline (future well)",
                    value=True,
                    key=f"{p}_nw_off",
                    help=(
                        "On (default): out of the base case / optimization, "
                        "available in Base vs Future. Off: participates "
                        "immediately like any active well."
                    ),
                )
            with pc4:
                st.write("")
                if st.button("Add", key=f"{p}_nw_add", use_container_width=True):
                    nw_name = (nw_name or "").strip()
                    if not nw_name:
                        st.warning("Enter a well name.")
                    elif nw_name in store:
                        st.warning(f"{nw_name} already exists in the review store.")
                    elif float(nw_pwf) >= float(nw_resp):
                        st.warning(
                            "Flowing BHP must be below reservoir pressure — "
                            "the IPR is undefined otherwise."
                        )
                    else:
                        store[nw_name] = wrs.hypothetical_entry(
                            nw_name,
                            field_model=nw_fm,
                            res_pres=nw_resp,
                            oil_bopd=nw_oil,
                            pwf=nw_pwf,
                            form_wc=nw_wc,
                            form_gor=nw_gor,
                            form_temp=nw_temp,
                            jpump_tvd=nw_tvd,
                            nozzle=nw_noz,
                            throat=nw_thr,
                            offline=bool(nw_off),
                            notes="hypothetical — future well (no analog)",
                        )
                        st.session_state.pop(f"{p}_nw_name", None)
                        st.toast(
                            f"Added future well {nw_name} ({nw_noz}{nw_thr})",
                            icon="🔵",
                        )
                        st.rerun()

        hypos = [w for w, e in store.items() if e.get("is_hypothetical")]
        if hypos:
            st.markdown("**Hypothetical / placeholder wells in this run:**")
            st.caption(
                "⚫ offline = out of the BASE case (and the optimization) — "
                "mark future wells offline, then add them back in the "
                "*Base vs Future* comparison below."
            )
            for w in sorted(hypos):
                note = store[w].get("notes") or "hypothetical"
                is_off = bool(store[w].get("offline"))
                stat = " · ⚫ offline (future)" if is_off else " · 🟢 online"
                row = st.columns([3, 1, 1])
                row[0].markdown(f"🔵 `{w}` — {note}{stat}")
                if row[1].button(
                    ("→ online" if is_off else "→ offline"),
                    key=f"{p}_ph_off_{w}",
                    use_container_width=True,
                    help=(
                        "Toggle whether this well participates in the base "
                        "case / optimization."
                    ),
                ):
                    store[w]["offline"] = not is_off
                    st.rerun()
                if row[2].button(
                    "Remove", key=f"{p}_ph_rm_{w}", use_container_width=True
                ):
                    store.pop(w, None)
                    st.rerun()


def _render_pf_what_if(spec: PadSpec, active: dict) -> None:
    """PF-pressure what-if: model all active wells at their reviewed pumps at
    two forced delivered PF pressures and show the oil impact.

    Baseline defaults to TODAY's live pad PF median (annulus gauges, via
    ``live_pad_pf_default``); both pressures are editable so the comparison
    can also be scenario-vs-scenario. Compute is
    ``pad_optimize.pf_pressure_what_if`` (bypasses the booster coupling —
    supply assumption stated in the caption); results persist in session
    state with a store signature so staleness is flagged.
    """
    import pandas as pd

    from woffl.gui.utils import live_pad_pf_default

    p = spec.prefix
    st.markdown("##### PF-pressure what-if (current pumps)")
    st.caption(
        "Model every active well at its **reviewed pump** at two delivered PF "
        "pressures and compare — the oil lever of moving pad PF, holding "
        "pumps fixed. Bypasses the booster coupling: it assumes the plant "
        "can deliver the chosen pressure at the resulting flow (sanity-check "
        "against the capability plot above). Baseline defaults to **today's "
        "live pad PF** (annulus-gauge median)."
    )
    try:
        live_pf = int(live_pad_pf_default(spec.pad))
    except Exception:
        live_pf = 3000
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        pf_base = st.number_input(
            "Baseline PF (psi)",
            min_value=500,
            max_value=5500,
            value=live_pf,
            step=25,
            key=f"{p}_pfw_base",
            help=(
                "Today's live pad PF by default — edit to compare from a "
                "different reference instead."
            ),
        )
    with c2:
        pf_scen = st.number_input(
            "Scenario PF (psi)",
            min_value=500,
            max_value=5500,
            value=min(live_pf + 200, 5500),
            step=25,
            key=f"{p}_pfw_scen",
        )
    with c3:
        st.write("")
        run_it = st.button(
            "Run PF what-if", key=f"{p}_pfw_run", use_container_width=True
        )

    result_key = f"{p}_pfw_result"
    if run_it:
        current = {
            w: (
                (active[w].get("review_nozzle") or "", active[w].get("review_throat") or "")
            )
            for w in active
        }
        current = {w: (c if c[0] and c[1] else None) for w, c in current.items()}
        tr = {w: _recent_test_rates(w) for w in active}
        try:
            with st.spinner(
                f"Modeling {len(active)} well(s) at {pf_base:,.0f} and "
                f"{pf_scen:,.0f} psi…"
            ):
                rows, totals = pad_optimize.pf_pressure_what_if(
                    wrs.store_to_well_configs(active), current, tr, pf_base, pf_scen
                )
            st.session_state[result_key] = {
                "rows": rows,
                "totals": totals,
                "pf_base": float(pf_base),
                "pf_scen": float(pf_scen),
                "store_sig": wrs.store_signature(active),
            }
        except Exception as e:
            st.session_state.pop(result_key, None)
            st.error(f"PF what-if failed: {e}")

    res = st.session_state.get(result_key)
    if not res:
        return
    if res.get("store_sig") != wrs.store_signature(active):
        st.caption(
            "⚠ Wells were added/edited since this what-if ran — re-run for "
            "the current state."
        )
    t = res["totals"]
    b_lbl = f"{res['pf_base']:,.0f}"
    s_lbl = f"{res['pf_scen']:,.0f}"
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"Oil @ {b_lbl} psi", f"{t['oil_base']:,.0f} BOPD")
    m2.metric(
        f"Oil @ {s_lbl} psi",
        f"{t['oil_scen']:,.0f} BOPD",
        delta=f"{t['d_oil']:+,.0f} BOPD",
    )
    m3.metric(
        "Projected Δ oil (test-anchored)",
        f"{t['projected_d_oil']:+,.0f} BOPD",
        help=(
            "Each well's measured test oil × the model's ratio between the "
            "two pressures — model bias cancels in the ratio, so this is the "
            "most trustworthy pad-level number. Wells with no test (e.g. "
            "placeholders) count in the modeled totals only."
        ),
    )
    m4.metric("Δ total PF", f"{t['pf_scen'] - t['pf_base']:+,.0f} BPD")

    def _r(v):
        return round(v) if v is not None else None

    df = pd.DataFrame(
        [
            {
                "Well": r["well"],
                "Pump": r["pump"],
                f"Oil @ {b_lbl}": _r(r["oil_base"]),
                f"Oil @ {s_lbl}": _r(r["oil_scen"]),
                "Δ oil": _r(r["d_oil"]),
                "Projected oil": _r(r["projected_oil"]),
                f"PF @ {b_lbl}": _r(r["pf_base"]),
                f"PF @ {s_lbl}": _r(r["pf_scen"]),
                "Δ PF": _r(r["d_pf"]),
            }
            for r in sorted(
                res["rows"],
                key=lambda r: abs(r["d_oil"]) if r["d_oil"] is not None else -1,
                reverse=True,
            )
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)
    if t["n_unsolved"]:
        st.caption(
            f"{t['n_unsolved']} well(s) didn't solve at one or both pressures "
            "(no reviewed pump, or no solution at that header) — excluded "
            "from the totals, shown blank above."
        )


def _render_base_vs_future(spec: PadSpec, store: dict, active: dict, n_pumps) -> None:
    """Base vs Future comparison — today's active wells vs the same pad plus
    selected offline ("future") wells, EXISTING pumps on both sides.

    Both cases settle on the plant coupling, so the added wells' PF demand
    droops the delivered header and today's producers feel it — the summary
    splits the net Δ into "future wells' oil" and "existing wells' Δ" so the
    real cost of adding the wells is visible, not just the gross add.
    Offline is the base-exclusion flag: mark a placeholder ⚫ offline above
    (or a real well on the Review stage), then pick it here as future.
    """
    import pandas as pd

    p = spec.prefix
    st.markdown("##### Base vs Future wells (existing pumps)")
    offline_wells = {w: e for w, e in store.items() if e.get("offline")}
    st.caption(
        "BASE = today's active wells at their reviewed pumps. FUTURE = base "
        "+ the wells picked below (typically placeholders marked ⚫ offline). "
        "Existing pumps everywhere — this isolates the effect of ADDING the "
        "wells, with the booster coupling settling both cases."
    )
    if not offline_wells:
        st.caption(
            "No offline wells to add — mark a placeholder offline above (or a "
            "real well on the Review stage) to enable the comparison."
        )
        return

    def _fmt(w: str) -> str:
        return f"🔵 {w}" if store[w].get("is_hypothetical") else w

    default_fut = [
        w for w in sorted(offline_wells) if offline_wells[w].get("is_hypothetical")
    ]
    c1, c2 = st.columns([3, 1])
    with c1:
        future_sel = st.multiselect(
            "Future wells (added to the base in the future case)",
            sorted(offline_wells),
            default=default_fut,
            format_func=_fmt,
            key=f"{p}_bvf_sel",
        )
    with c2:
        st.write("")
        run_it = st.button(
            "Run base vs future",
            key=f"{p}_bvf_run",
            use_container_width=True,
            disabled=not future_sel,
        )

    result_key = f"{p}_bvf_result"
    if run_it:

        def _choice(e: dict):
            n, t = e.get("review_nozzle"), e.get("review_throat")
            return (str(n), str(t)) if n and t else None

        base_entries = {w: e for w, e in active.items() if _choice(e)}
        no_pump = [w for w in active if w not in base_entries]
        fut_only = {
            w: offline_wells[w] for w in future_sel if _choice(offline_wells[w])
        }
        no_pump += [w for w in future_sel if w not in fut_only]
        if no_pump:
            st.caption("Excluded (no reviewed pump): " + ", ".join(sorted(no_pump)))
        if not base_entries or not fut_only:
            st.warning(
                "Need at least one base well and one future well with a "
                "reviewed pump."
            )
            return
        tr = {w: _recent_test_rates(w) for w in base_entries}
        base_choices = {w: _choice(e) for w, e in base_entries.items()}
        fut_entries = {**base_entries, **fut_only}
        fut_choices = {w: _choice(e) for w, e in fut_entries.items()}
        try:
            with st.spinner("Settling BASE case on the plant coupling…"):
                per_base, meta_base = pad_optimize.evaluate_fixed_scenario(
                    wrs.store_to_well_configs(base_entries),
                    spec.plant,
                    n_pumps,
                    base_choices,
                    test_rates=tr,
                    current_choices=base_choices,
                )
            with st.spinner(f"Settling FUTURE case (+{len(fut_only)} well(s))…"):
                per_fut, meta_fut = pad_optimize.evaluate_fixed_scenario(
                    wrs.store_to_well_configs(fut_entries),
                    spec.plant,
                    n_pumps,
                    fut_choices,
                    test_rates=tr,
                    current_choices=fut_choices,
                )
        except Exception as e:
            st.session_state.pop(result_key, None)
            st.error(f"Base-vs-future comparison failed: {e}")
            return
        rows = pad_optimize.base_vs_future_rows(per_base, per_fut, set(fut_only))
        totals = pad_optimize.base_vs_future_totals(rows, meta_base, meta_fut)
        st.session_state[result_key] = {
            "rows": rows,
            "totals": totals,
            "n_pumps": n_pumps,
            "future": sorted(fut_only),
            "store_sig": wrs.store_signature({**active, **fut_only}),
        }

    res = st.session_state.get(result_key)
    if not res:
        return
    sig_now = wrs.store_signature(
        {**active, **{w: store[w] for w in res.get("future", []) if w in store}}
    )
    if res.get("store_sig") != sig_now or res.get("n_pumps") != n_pumps:
        st.caption(
            "⚠ Wells or pump count changed since this comparison ran — re-run "
            "for the current state."
        )
    t = res["totals"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pad oil — base", f"{t['oil_base']:,.0f} BOPD")
    m2.metric(
        "Pad oil — future",
        f"{t['oil_future']:,.0f} BOPD",
        delta=f"{t['d_oil']:+,.0f} BOPD",
    )
    m3.metric(
        f"Future wells ({t['n_future']})",
        f"{t['future_oil']:,.0f} BOPD",
        help="Combined oil from the added wells in the future case.",
    )
    m4.metric(
        "Existing wells Δ",
        f"{t['existing_d_oil']:+,.0f} BOPD",
        help=(
            "What the added PF demand costs today's producers through the "
            "header droop — the future wells' oil net of this is the real "
            "pad gain."
        ),
    )
    st.caption(
        f"Header {t['header_base']:,.0f} → {t['header_future']:,.0f} psi · "
        f"Total PF {t['pf_base']:,.0f} → {t['pf_future']:,.0f} BPD."
    )

    def _r(v):
        return round(v) if v is not None else None

    df = pd.DataFrame(
        [
            {
                "Well": ("🔵 " + r["well"]) if r["future"] else r["well"],
                "Pump": r["pump"],
                "Oil base": _r(r["oil_base"]),
                "Oil future": _r(r["oil_future"]),
                "Δ oil": _r(r["d_oil"]),
                "PF base": _r(r["pf_base"]),
                "PF future": _r(r["pf_future"]),
            }
            for r in sorted(
                res["rows"],
                key=lambda r: (
                    not r["future"],
                    -(abs(r["d_oil"]) if r["d_oil"] is not None else 0),
                ),
            )
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)


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

    # Placeholder wells — clone an existing reviewed well into the store as a
    # hypothetical, without leaving the Configure screen.
    _render_placeholder_wells(spec, store, active)

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
        # Checkbox `value=` MUST read session_state first (CLAUDE.md gotcha) —
        # a hardcoded literal would clobber a programmatically-set state on
        # every rerun. Default True: the plant-derived gate is the new default
        # behavior; the manual number_input below is the override path.
        auto_wc = st.checkbox(
            "Auto marginal WC (from plant limits)",
            value=st.session_state.get(f"{p}_marginal_wc_auto", True),
            key=f"{p}_marginal_wc_auto",
            help=spec.marginal_wc_help,
        )
        if auto_wc:
            st.caption(
                "Gate computed from the booster curve / PF budget at each "
                "trial header; slack → parsimony tie-break."
            )
            marginal_wc = None
        else:
            # Same key as before the auto-WC feature shipped, so an old
            # session's manual value survives the upgrade untouched.
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
        parsimony_bopd = st.number_input(
            "Parsimony threshold (BOPD)",
            min_value=0.0,
            value=20.0,
            step=1.0,
            key=f"{p}_parsimony_bopd",
            help=(
                "A bigger pump must add at least this much oil to be chosen "
                "over a smaller one when PF has slack. 0 disables the "
                "tie-break."
            ),
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

    # PF-pressure what-if — the oil lever of moving pad PF at fixed pumps,
    # baseline = today's live pad PF.
    _render_pf_what_if(spec, active)
    st.divider()

    # Base vs Future — today's pad vs pad + flagged future wells, existing
    # pumps on both sides, coupled to the plant.
    _render_base_vs_future(spec, store, active, n_pumps)
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
                    parsimony_bopd=parsimony_bopd,
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


def _gate_caption(meta: dict) -> str:
    """One-line summary of the marginal-WC gate actually applied at the
    winning header, for the Results stage.

    Pure (no Streamlit) so it's unit-testable directly. Reads ``meta`` with
    ``.get`` throughout: a run stored in session BEFORE this feature shipped
    carries none of ``marginal_wc_used``/``marginal_wc_source``/``pf_slack``,
    and must render a sensible line instead of KeyError-ing.
    """
    used = meta.get("marginal_wc_used")
    if used is None:
        return (
            "Marginal WC gate: not recorded for this run — re-run to see gate detail."
        )
    source = meta.get("marginal_wc_source", "manual")
    if source == "manual":
        detail = "manual"
    elif meta.get("pf_slack"):
        detail = "auto — PF slack, parsimony active"
    else:
        detail = "auto, plant-derived — budget bound"
    return f"Marginal WC gate: {used:.2f} ({detail})"


def _parsimony_rows(swaps: list[dict]) -> list[dict]:
    """Display rows for the parsimony-swaps table — pure so it's testable
    without Streamlit. ``swaps`` is ``meta["parsimony_swaps"]``: a list of
    ``{well, from_pump, to_pump, oil_given_up, pf_saved}`` dicts."""
    return [
        {
            "Well": s.get("well"),
            "Pump": f"{s.get('from_pump')} → {s.get('to_pump')}",
            "Oil given up (BOPD)": s.get("oil_given_up"),
            "PF saved (BPD)": s.get("pf_saved"),
        }
        for s in swaps
    ]


def _render_parsimony_swaps(meta: dict) -> None:
    """Expander listing wells the parsimony tie-break kept on a smaller pump.
    Renders nothing when there's nothing to show (empty/missing/old meta)."""
    import pandas as pd

    swaps = meta.get("parsimony_swaps") or []
    if not swaps:
        return
    with st.expander(f"Parsimony: {len(swaps)} well(s) kept a smaller pump"):
        st.caption(
            "PF had slack at the winning header — these wells kept a smaller "
            "pump because the bigger one didn't add enough oil to clear the "
            "parsimony threshold."
        )
        st.dataframe(
            pd.DataFrame(_parsimony_rows(swaps)),
            use_container_width=True,
            hide_index=True,
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

    # Current state: measured recent-test medians + the pump captured at
    # review, with the configure-stage match check supplying Model÷Test.
    rates = {w: _recent_test_rates(w) for w in active}
    mc = st.session_state.get(f"{p}_matchcheck")
    mc_rows = {r["well"]: r for r in mc["rows"]} if mc else {}
    rows = _build_comparison_rows(
        results, active, si_wells, rates, mc_rows, header_psi=meta.get("header_psi")
    )
    cur_oil = sum(r["Current oil"] or 0 for r in rows)
    cur_pf = sum(r["Current PF"] or 0 for r in rows)
    cur_hdr = None
    if cur_pf > 0:
        hdr, over = pad_optimize.settled_header(spec.plant, cur_pf, 0.0, n_pumps)
        cur_hdr = None if over else hdr

    # Station operating point — optimized, with deltas vs today
    st.markdown("#### Station operating point — optimized vs current")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Header pressure",
        f"{meta['header_psi']:,.0f} psi",
        delta=(
            f"{meta['header_psi'] - cur_hdr:+,.0f} vs current"
            if cur_hdr is not None
            else None
        ),
        delta_color="off",
    )
    m2.metric(
        "Total power fluid",
        f"{meta['total_pf_bpd']:,.0f} BPD",
        delta=f"{meta['total_pf_bpd'] - cur_pf:+,.0f} vs current" if cur_pf else None,
        delta_color="off",
    )
    if spec.station_metric3 is not None:
        label, value = spec.station_metric3(meta)
        m3.metric(label, value)
    m4.metric(
        "Total oil",
        f"{meta['total_oil_bopd']:,.0f} BOPD",
        delta=(
            f"{meta['total_oil_bopd'] - cur_oil:+,.0f} vs current" if cur_oil else None
        ),
    )
    st.caption(
        f"**Current** = what the pad does today, measured: median of each well's "
        f"recent tests — {cur_oil:,.0f} BOPD oil · {cur_pf:,.0f} BPD PF"
        + (f" · header settles ≈ {cur_hdr:,.0f} psi at that draw" if cur_hdr else "")
        + ". **Optimized** = what the plan below is modeled to deliver."
    )
    st.caption(_gate_caption(meta))
    _render_parsimony_swaps(meta)

    if spec.render_station_extras is not None:
        spec.render_station_extras(meta)

    # Per-well table — current pump + measured rates beside the optimizer's
    # pick, SHUT IN wells included so the SI decision (and its cost) is visible.
    st.markdown("#### Per well — current vs optimized")
    _MAIN_COLS = [
        "Well",
        "Current pump",
        "Current oil",
        "Current PF",
        "Optimized pump",
        "Opt oil",
        "Opt PF",
        "Current BHP",
        "Opt BHP",
        "Current PF psi",
        "Opt PF psi",
        "Δ oil",
        "Change",
        "Model÷Test",
        "Status",
    ]
    _DETAIL_COLS = [
        "Well",
        "Optimized pump",
        "Form. water (BPD)",
        "Total WC",
        "Status",
    ]
    df = pd.DataFrame(rows)
    st.dataframe(df[_MAIN_COLS], use_container_width=True, hide_index=True)

    # Table totals — sums of the rows above (incl. SHUT IN zeros), so this
    # line always reconciles with the table even if it drifts from run meta.
    opt_oil_tot = sum(r["Opt oil"] or 0 for r in rows)
    opt_pf_tot = sum(r["Opt PF"] or 0 for r in rows)
    st.markdown(
        f"**Total power fluid** — current **{cur_pf:,.0f}** → optimized "
        f"**{opt_pf_tot:,.0f} BPD** ({opt_pf_tot - cur_pf:+,.0f})  ·  "
        f"**Total oil** — current **{cur_oil:,.0f}** → optimized "
        f"**{opt_oil_tot:,.0f} BOPD** ({opt_oil_tot - cur_oil:+,.0f})"
    )
    st.caption(
        "**Current oil/PF** = measured (median of recent tests); **Opt oil/PF** = "
        "model. **Δ oil** therefore mixes model vs measured — check **Model÷Test** "
        "(model ÷ test at the CURRENT pump, from the pre-flight match check): a well "
        "at 1.3 over-predicts by ~30%, so read its Δ as optimistic. **Current BHP** / "
        "**Current PF psi** are the REVIEWED store values (the well's IPR anchor); "
        "**Opt BHP** is the optimizer's solved pump-intake suction pressure; "
        "**Opt PF psi** is the pad-level settled header (same for every optimized "
        "row). **Change**: ▲ bigger / ▼ smaller = nozzle size up / down; ▲/▼ throat "
        "= same nozzle, larger / smaller throat. Sorted by biggest mover. Wells with "
        "no installed pump or no valid tests show —."
    )
    with st.expander("Full optimizer detail (formation water, WC)"):
        st.dataframe(df[_DETAIL_COLS], use_container_width=True, hide_index=True)

    with st.expander("Per-well IPR — current vs optimized", expanded=False):
        ipr_fig = _build_ipr_grid_figure(results, active, si_wells)
        if ipr_fig is None:
            st.caption(
                "No wells have enough reviewed IPR data (pwf/qwf/reservoir "
                "pressure) to draw a curve — review the wells in **Review "
                "Wells** first."
            )
        else:
            st.caption(
                "The curve is the well's Vogel IPR through the reviewed anchor "
                "(green ●) — both operating points sit on it because the "
                "jet-pump solver solves on this same curve."
            )
            st.plotly_chart(
                ipr_fig, use_container_width=True, key=f"{spec.prefix}_ipr_grid"
            )

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
