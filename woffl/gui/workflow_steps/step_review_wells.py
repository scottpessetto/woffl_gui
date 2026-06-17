"""Per-well "Review & Calibrate" stage for a pad (used by the dedicated S-Pad page).

The engineer steps through every well on the pad in the *reused* single-well
Solver, verifies/adjusts the match, and clicks **Save** — which snapshots the
reviewed sidebar state (+ calibration + provenance) into the per-pad store
(``well_review_store``). Saved wells drop off the pending list; hypothetical /
future wells can be added with a hand-entered IPR; the store round-trips to CSV
for reruns/edits.

Reuses, and does NOT fork, the Solver: it renders the single-well sidebar scoped
to the pad's wells (``render_sidebar(well_filter=...)``) and the exact
``single_well_page._build_simulation_objects`` + ``jetpump_solver.render_tab``
pair the Single-Well page uses.

Navigation (advancing past review) is owned by the host page — this module just
renders the review UI and maintains the store. ``store_for(pad)`` exposes it.
"""

import streamlit as st

from woffl.gui import memory_gauge
from woffl.gui.sidebar import clamp_seed, render_sidebar
from woffl.gui.utils import get_available_wells, get_well_tests_for_well, pad_from_mp_name
from woffl.gui.workflow_steps import well_review_store as wrs

_LAST_HYDRATED_KEY = "_sp_last_hydrated"  # which well we last pushed store->sidebar for


# ---------------------------------------------------------------------------
# Store helpers (per-pad)
# ---------------------------------------------------------------------------


def store_for(pad: str) -> dict:
    """The per-pad reviewed-well store ({well_name: entry})."""
    return st.session_state.setdefault(f"sp_well_store_{pad}", {})


def _select_well(well_name: str) -> None:
    """Point the sidebar well dropdown at ``well_name``: set the logical key and
    bump the selector nonce so the (freshly-keyed) selectbox re-reads its index.
    The nonce bump is what reliably advances the dropdown — popping the widget
    key alone didn't move it on Save & next."""
    st.session_state["selected_well"] = well_name
    st.session_state["_well_sel_nonce"] = st.session_state.get("_well_sel_nonce", 0) + 1
    st.session_state.pop(_LAST_HYDRATED_KEY, None)


def _pad_real_wells(pad: str) -> list[str]:
    """All characterized wells on the pad (the 'account for every well' set)."""
    return sorted(
        w for w in get_available_wells()
        if w != "Custom" and pad_from_mp_name(w) == pad
    )


# Wells that should default to OFFLINE (excluded from the run) when first reviewed/
# applied on a pad — the engineer can uncheck "Offline" to bring them online.
# Keyed by pad -> well NUMBERS within the pad, so it's robust to the "I-03" vs
# "MPI-3" naming variations (matched on the trailing number, leading zeros stripped).
_PAD_DEFAULT_OFFLINE: dict[str, set[str]] = {"I": {"3", "11", "15"}}


def _well_number(well: str) -> str | None:
    import re

    m = re.search(r"(\d+)\s*$", str(well))
    return str(int(m.group(1))) if m else None


def _is_default_offline(pad: str, well: str) -> bool:
    """True if this well defaults to offline on its pad (and isn't yet in the
    store with a user-chosen state — callers pass the stored value as the fallback)."""
    return _well_number(well) in _PAD_DEFAULT_OFFLINE.get(pad, set())


def _well_has_gauge(well: str) -> bool:
    """True if any test for the well carries a finite measured BHP."""
    try:
        tests = get_well_tests_for_well(well)
    except Exception:
        return False
    if tests is None or tests.empty or "BHP" not in tests.columns:
        return False
    import pandas as pd

    return bool(tests["BHP"].apply(lambda v: not pd.isna(v)).any())


def _infer_ipr_source(well: str) -> str:
    """Best-guess provenance tag for the IPR anchor (engineer can override)."""
    if st.session_state.get("sw_vogel_coeffs"):
        return "vogel"
    try:
        tests = get_well_tests_for_well(well)
    except Exception:
        tests = None
    if tests is not None and not tests.empty:
        return "single_test"
    return "forced"


# ---------------------------------------------------------------------------
# Sidebar hydration (store entry -> sidebar widgets)
# ---------------------------------------------------------------------------


def _set_sidebar(logical_key: str, value, *, as_int: bool) -> None:
    """Set a sidebar logical key + drop its widget key (re-seeds next render).

    Streamlit forbids writing a widget's state key after it renders, so we set
    the *logical* key and pop ``{key}_input``; ``_number_input`` re-initializes
    the widget from the logical value on the next run. Clamped to SEED_BOUNDS so
    an out-of-range value can't be silently reset to the widget minimum.
    """
    v = clamp_seed(logical_key, value)
    st.session_state[logical_key] = int(round(v)) if as_int else float(v)
    st.session_state.pop(f"{logical_key}_input", None)


def _hydrate_sidebar_from_entry(entry: dict) -> None:
    """Push a saved store entry back onto the sidebar so re-opening a well shows
    the *reviewed* values rather than a fresh IPR auto-populate."""
    # Sidebar qwf is OIL (BOPD); the store keeps the as-reviewed oil rate
    # alongside the total-liquid qwf — use the oil rate.
    oil = entry.get(wrs.OIL_RATE_FIELD)
    if oil is not None:
        _set_sidebar("qwf", oil, as_int=True)
    _set_sidebar("pwf", entry["pwf"], as_int=True)
    _set_sidebar("res_pres", entry["res_pres"], as_int=True)
    _set_sidebar("form_wc", round(float(entry["form_wc"]), 2), as_int=False)
    _set_sidebar("form_gor", entry["form_gor"], as_int=True)
    _set_sidebar("form_temp", entry["form_temp"], as_int=True)
    _set_sidebar("surf_pres", entry["surf_pres"], as_int=True)
    _set_sidebar("jpump_tvd", entry["jpump_tvd"], as_int=True)

    for k in ("tubing_od", "tubing_thickness", "casing_od", "casing_thickness"):
        _set_sidebar(k, float(entry[k]), as_int=False)

    for store_k, side_k in (("ken_well", "ken"), ("kth_well", "kth"), ("kdi_well", "kdi")):
        if entry.get(store_k) is not None:
            _set_sidebar(side_k, float(entry[store_k]), as_int=False)
    if entry.get("ppf_surf_well") is not None:
        _set_sidebar("ppf_surf", entry["ppf_surf_well"], as_int=True)

    for k in ("oil_api", "gas_sg", "wat_sg", "bubble_point"):
        if entry.get(k) is not None:
            _set_sidebar(k, float(entry[k]), as_int=False)

    if entry.get("review_nozzle"):
        st.session_state["nozzle_no"] = entry["review_nozzle"]
        st.session_state.pop("nozzle_no_input", None)
    if entry.get("review_throat"):
        st.session_state["area_ratio"] = entry["review_throat"]
        st.session_state.pop("area_ratio_input", None)

    fm = entry.get("field_model")
    if fm in ("Schrader", "Kuparuk"):
        st.session_state["field_model_index"] = ["Schrader", "Kuparuk"].index(fm)
        st.session_state.pop("field_model_radio", None)


# ---------------------------------------------------------------------------
# UI sections
# ---------------------------------------------------------------------------


def _render_progress(pad: str, real_wells: list[str]) -> None:
    store = store_for(pad)
    reviewed_real = [w for w in real_wells if w in store]
    pending = [w for w in real_wells if w not in store]
    hypo = [w for w, e in store.items() if e.get("is_hypothetical")]

    n_offline = sum(1 for w in store if store[w].get("offline"))
    n_active = len(store) - n_offline
    n_done, n_total = len(reviewed_real), len(real_wells)
    st.progress(
        n_done / n_total if n_total else 0.0,
        text=f"{pad}-Pad: {n_done} of {n_total} wells reviewed"
        + (f"  ·  +{len(hypo)} hypothetical" if hypo else "")
        + f"  ·  {n_active} active / {n_offline} offline",
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Pending**")
        st.write(", ".join(pending) if pending else "— none —")
    with c2:
        st.markdown("**Reviewed**")
        badges = []
        for w in reviewed_real:
            if store[w].get("offline"):
                mark = "⚫"
            else:
                mark = "🟢" if store[w].get("bhp_source") == "gauged" else "🟡"
            badges.append(f"{mark} {w}")
        badges += [
            ("⚫ " if store[w].get("offline") else "🔵 ") + f"{w} (hypo)" for w in hypo
        ]
        st.write("  ·  ".join(badges) if badges else "— none yet —")
    st.caption("🟢 gauged BHP  ·  🟡 assumed/no-gauge  ·  🔵 hypothetical  ·  ⚫ offline (excluded)")


def _render_modeling_status(pad: str) -> None:
    """Quick per-well 'offline for this modeling' toggles (pulled / down wells)."""
    store = store_for(pad)
    if not store:
        return
    with st.expander("🔌 Modeling status — exclude pulled / offline wells", expanded=False):
        st.caption(
            "Offline wells stay in the review set (accounted for) but are dropped "
            "from the optimization run. Toggle here without re-opening the Solver."
        )
        cols = st.columns(3)
        for i, w in enumerate(sorted(store)):
            with cols[i % 3]:
                # The checkbox's keyed state is the source of truth; write it
                # straight back to the store each run so the badge + the run
                # filter stay in sync.
                store[w]["offline"] = st.checkbox(
                    f"{w} offline", value=store[w].get("offline", False),
                    key=f"sp_status_off_{pad}_{w}",
                )


def _render_save_panel(params, real_wells: list[str], pad: str) -> None:
    well = params.selected_well
    store = store_for(pad)
    already = well in store

    st.markdown(f"#### Save review — **{well}**")

    # The Solver (rendered below) fills these (deferred) so the engineer sees the
    # well's history AND the model-vs-actual difference right under the title,
    # without scrolling into the Solver. Order matters: jp_box (the production /
    # BHP / JPCO chart + pump-in-well timeline) sits ABOVE hero_box (pump banner
    # + vs-actual metrics). Returned to render_review_stage, which hands them to
    # jetpump_solver.render_tab as jp_strip_container / hero_container.
    jp_box = st.container()
    hero_box = st.container()

    inferred_ipr = _infer_ipr_source(well)

    # Memory-gauge provenance: if a gauge is loaded for this well, record a note
    # ("created with gauge data") and default the match basis to gauged.
    gauge = memory_gauge.get_gauge(well)
    gauge_note = ""
    if gauge is not None:
        gauge_note = (
            f"Gauge-backed IPR — {gauge.start_date:%Y-%m-%d} to "
            f"{gauge.end_date:%Y-%m-%d}, {int(gauge.sample_count):,} samples"
        )
        st.caption(f"📈 {gauge_note}")
    default_gauged = _well_has_gauge(well) or gauge is not None

    c1, c2, c3 = st.columns([1.1, 1.1, 1.4])
    with c1:
        ipr_source = st.selectbox(
            "IPR basis", options=list(wrs.IPR_SOURCES),
            index=list(wrs.IPR_SOURCES).index(inferred_ipr),
            help="How the IPR anchor was established (auto-detected; override if needed).",
            key=f"sp_ipr_src_{well}",
        )
    with c2:
        bhp_source = st.selectbox(
            "Match basis", options=list(wrs.BHP_SOURCES),
            index=0 if default_gauged else 1,
            help="'gauged' = a measured BHP backed the match; 'assumed' = no gauge, pwf is an estimate.",
            key=f"sp_bhp_src_{well}",
        )
    with c3:
        notes = st.text_input("Note (optional)", key=f"sp_note_{well}",
                              value=store.get(well, {}).get("notes", ""))

    offline = st.checkbox(
        "🔌 Offline for this modeling (pulled / down — exclude from optimization)",
        # New wells on a pad's default-offline list start checked; once saved, the
        # stored value (the engineer's choice) wins.
        value=store.get(well, {}).get("offline", _is_default_offline(pad, well)),
        key=f"sp_offline_{well}",
    )

    jpump_md = params.well_data.get("JP_MD") if params.well_data else None

    btn_label = "💾 Update saved review" if already else "💾 Save & next well"
    if st.button(btn_label, type="primary", use_container_width=True, key=f"sp_save_{well}"):
        entry = wrs.snapshot_from_params(
            params, ipr_source=ipr_source, bhp_source=bhp_source, offline=offline,
            gauge_note=gauge_note,
            jpump_md=float(jpump_md) if jpump_md is not None else None,
            notes=notes,
        )
        store[well] = entry

        # Advance to the next UNREVIEWED well after this one (sequential), wrapping
        # to the first pending if we're at the end. Leave a one-shot target;
        # render_review_stage applies it right before the sidebar renders.
        if not already:
            i = real_wells.index(well) if well in real_wells else -1
            after = [w for w in real_wells[i + 1:] if w not in store]
            nxt = after[0] if after else next((w for w in real_wells if w not in store), None)
            if nxt:
                st.session_state[f"sp_review_target_{pad}"] = nxt
        st.toast(f"Saved {well}", icon="✅")
        st.rerun()

    if already and st.button("🗑 Remove from review set", key=f"sp_del_{well}"):
        store.pop(well, None)
        st.session_state.pop(_LAST_HYDRATED_KEY, None)
        st.rerun()

    return jp_box, hero_box


def _render_hypothetical_form(pad: str) -> None:
    with st.expander("➕ Add a hypothetical / future well", expanded=False):
        st.caption(
            "Model adding a well to the pad. Enter an expected IPR + geometry; "
            "it feeds the optimizer like a real well but is flagged hypothetical."
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            name = st.text_input("Well name", value=f"MP{pad}-NEW", key="sp_hypo_name")
            field_model = st.radio("Field model", ["Schrader", "Kuparuk"],
                                   horizontal=True, key="sp_hypo_fm")
            res_pres = st.number_input("Reservoir P (psi)", 400, 5000, 1800, 10, key="sp_hypo_resp")
        with c2:
            oil = st.number_input("Expected oil (BOPD)", 0, 6000, 400, 10, key="sp_hypo_oil")
            pwf = st.number_input("Flowing BHP (psi)", 100, 2500, 900, 10, key="sp_hypo_pwf")
            wc = st.number_input("Water cut", 0.0, 0.99, 0.5, 0.01, format="%.2f", key="sp_hypo_wc")
        with c3:
            gor = st.number_input("GOR (scf/bbl)", 20, 10000, 250, 25, key="sp_hypo_gor")
            temp = st.number_input("Form temp (°F)", 32, 350, 160, 1, key="sp_hypo_temp")
            tvd = st.number_input("Jetpump TVD (ft)", 2500, 8000, 4200, 10, key="sp_hypo_tvd")

        if st.button("Add hypothetical well", key="sp_hypo_add"):
            name = name.strip()
            if not name:
                st.warning("Enter a well name.")
                return
            oil_frac = max(1.0 - float(wc), wrs._MIN_OIL_FRACTION)
            store_for(pad)[name] = {
                "well_name": name,
                "res_pres": float(res_pres), "form_temp": float(temp),
                "jpump_tvd": float(tvd), "tubing_od": 4.5, "tubing_thickness": 0.5,
                "casing_od": 6.875, "casing_thickness": 0.5,
                "form_wc": float(wc), "form_gor": float(gor), "surf_pres": 210.0,
                "qwf": float(oil) / oil_frac, "pwf": float(pwf),
                wrs.OIL_RATE_FIELD: float(oil),
                "jpump_md": float(tvd), "oil_api": None, "gas_sg": None,
                "wat_sg": None, "bubble_point": None, "ppf_surf_well": None,
                "knz_well": None, "ken_well": None, "kth_well": None, "kdi_well": None,
                "field_model": field_model, "review_nozzle": "", "review_throat": "",
                "ipr_source": "hypothetical", "bhp_source": "assumed", "gauge_note": "",
                "is_hypothetical": True, "offline": False, "reviewed": True,
                "notes": "hypothetical",
            }
            st.toast(f"Added hypothetical {name}", icon="🔵")
            st.rerun()


def _render_ipr_pdf_button(pad: str) -> None:
    """Single-click download of the 'IPRs used' PDF — one IPR per reviewed well,
    with memory-gauge test points marked and a per-well provenance note."""
    store = store_for(pad)
    if not store:
        return
    if st.button("⬇ Download IPRs-used PDF", use_container_width=True, key="sp_ipr_pdf"):
        import datetime as _dt

        import pandas as pd

        from woffl.gui import s_pad_ipr_report
        from woffl.gui.components.download import autodownload

        items = []
        for well in sorted(store):
            entry = store[well]
            tests, gdates = None, set()
            if not entry.get("is_hypothetical"):
                try:
                    tests = get_well_tests_for_well(well)
                except Exception:
                    tests = None
                g = memory_gauge.get_gauge(well)
                if g is not None:
                    gdates = set(pd.to_datetime(g.daily_df["tag_date"]).dt.normalize())
            items.append({"entry": entry, "tests_df": tests, "gauge_dates": gdates})

        stamp = _dt.datetime.now().strftime("Generated %Y-%m-%d %H:%M")
        pdf_bytes = s_pad_ipr_report.build_ipr_pdf(pad, items, stamp)
        autodownload(pdf_bytes, f"{pad}_pad_IPRs_used.pdf", "application/pdf", "sp_ipr_pdf_dl")


def _render_csv_io(pad: str) -> None:
    store = store_for(pad)
    c1, c2 = st.columns(2)
    with c1:
        if store and st.button("⬇ Download review CSV", use_container_width=True, key="sp_csv_dl"):
            from woffl.gui.components.download import autodownload

            csv_bytes = wrs.store_to_dataframe(store).to_csv(index=False).encode("utf-8")
            autodownload(csv_bytes, f"{pad}_pad_review.csv", "text/csv", "sp_csv_auto_dl")
    with c2:
        up = st.file_uploader("⬆ Load review CSV", type=["csv"], key="sp_csv_up")
        if up is not None:
            sig = (up.name, up.size)
            if st.session_state.get("_sp_csv_sig") != sig:
                import pandas as pd

                # dtype=str so pandas can't turn nozzle '10' into the float 10.0
                # (which then stringifies to '10.0' and breaks NOZZLE_OPTIONS).
                loaded = wrs.dataframe_to_store(pd.read_csv(up, dtype=str))
                store.update(loaded)
                st.session_state["_sp_csv_sig"] = sig
                st.success(f"Loaded {len(loaded)} well(s) from CSV.")
                st.rerun()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _prefetch_pad_wells(pad: str, wells: list[str]) -> None:
    """Fire-and-forget warm of every pad well's cached props / tests / survey so
    swapping between wells in the review feels snappy (no cold per-well fetch +
    Nelder-Mead survey fit on first visit). Also warms the data the batch
    auto-match needs. Once per pad per session; a daemon thread fills the
    process-wide @st.cache_data entries while the engineer works the first well.
    """
    key = f"_pad_prefetched_{pad}"
    if st.session_state.get(key):
        return
    st.session_state[key] = True

    def _worker() -> None:
        from woffl.gui.utils import (
            create_well_profile_from_survey,
            get_well_data,
            get_well_tests_for_well,
        )

        for w in wells:
            wd = None
            try:
                wd = get_well_data(w)            # props (TVD / field / geometry)
            except Exception:
                pass
            try:
                get_well_tests_for_well(w)       # tests + gauge overlay + extended window
            except Exception:
                pass
            if wd:
                try:  # survey fit, keyed on the same (well, TVD, field) the review uses
                    tvd = wd.get("JP_TVD")
                    field = "schrader" if wd.get("is_sch", True) else "kuparuk"
                    create_well_profile_from_survey(
                        w, int(tvd) if tvd is not None else None, field
                    )
                except Exception:
                    pass

    import threading

    from streamlit.runtime.scriptrunner import add_script_run_ctx

    t = threading.Thread(target=_worker, daemon=True, name=f"pad-prefetch-{pad}")
    add_script_run_ctx(t)  # lets the thread call @st.cache_data fns cleanly
    t.start()


def _batch_automatch_inputs(well: str, jp_hist, ppf_surf: float, rho_pf: float = 62.4):
    """Assemble joint_match kwargs for one well from props + recent tests + the
    current pump — WITHOUT the sidebar (so the whole pad can be matched in one
    pass). Mirrors the sidebar's auto-populate field accesses. Returns
    (kwargs, None) or (None, reason) when data is missing.
    """
    import pandas as pd

    from woffl.assembly.ipr_analyzer import (
        compute_vogel_coefficients,
        estimate_reservoir_pressure,
    )
    from woffl.assembly.jp_history import get_current_pump
    from woffl.gui.utils import (
        create_pipes,
        create_well_profile_from_survey,
        get_well_data,
        get_well_tests_for_well,
    )

    wd = get_well_data(well)
    tests = get_well_tests_for_well(well)
    if not wd or tests is None or getattr(tests, "empty", True) or len(tests) < 2:
        return None, None, "insufficient props/tests"
    oil, pf = _recent_test_rates(well)
    if not oil or not pf:
        return None, None, "no recent oil/PF test"
    cp = get_current_pump(jp_hist, well) if jp_hist is not None else None
    if not (cp and cp.get("nozzle_no") and cp.get("throat_ratio")):
        return None, None, "no current pump"
    try:
        coeffs = compute_vogel_coefficients(estimate_reservoir_pressure(tests))
        row = coeffs[coeffs["Well"] == well].iloc[0]
        pres, wc, gor = float(row["ResP"]), float(row["form_wc"]), float(row["fgor"])
    except Exception:
        return None, None, "IPR fit failed"

    recent = tests.sort_values("WtDate", ascending=False).iloc[0]
    bhp = recent.get("BHP")
    bhp = float(bhp) if (bhp is not None and not pd.isna(bhp) and float(bhp) > 0) else None
    whp = recent.get("whp")
    surf = float(whp) if (whp is not None and not pd.isna(whp) and float(whp) > 0) else 250.0
    tvd = int(wd.get("JP_TVD") or 4065)
    field = "schrader" if wd.get("is_sch", True) else "kuparuk"
    try:
        _, _, wellbore = create_pipes(
            float(wd.get("out_dia") or 4.5), float(wd.get("thick") or 0.5), 6.875, 0.5
        )
        wp = create_well_profile_from_survey(well, tvd, field)
    except Exception:
        return None, None, "geometry/survey build failed"
    kwargs = dict(
        oil_target=oil, pf_target=pf, pres=pres,
        nozzle=str(cp["nozzle_no"]), throat=str(cp["throat_ratio"]),
        surf_pres=surf, form_temp=float(wd.get("form_temp") or 120.0),
        rho_pf=rho_pf, ppf_surf0=float(ppf_surf), wellbore=wellbore,
        well_profile=wp, form_wc=wc, form_gor=gor, field_model=field,
        bhp_target=bhp, pin_ppf=True,
    )
    # Raw scalars to rebuild a SimulationParams for "apply to store" (WellConfig
    # wants the capitalized field-model; the joint_match kwargs above keep the
    # lowercase the PVT helpers expect).
    raw = {
        "nozzle_no": str(cp["nozzle_no"]), "area_ratio": str(cp["throat_ratio"]),
        "jpump_direction": "reverse",
        "tubing_od": float(wd.get("out_dia") or 4.5),
        "tubing_thickness": float(wd.get("thick") or 0.5),
        "casing_od": 6.875, "casing_thickness": 0.5,
        "form_wc": float(wc), "form_gor": int(round(gor)),
        "form_temp": int(round(float(wd.get("form_temp") or 120.0))),
        "field_model": "Schrader" if wd.get("is_sch", True) else "Kuparuk",
        "oil_api": wd.get("oil_api"), "gas_sg": wd.get("gas_sg"),
        "wat_sg": wd.get("wat_sg"), "bubble_point": wd.get("bubble_point"),
        "surf_pres": int(round(surf)), "jpump_tvd": int(tvd), "has_bhp": bhp is not None,
    }
    return kwargs, raw, None


def _render_batch_automatch(pad: str, real_wells: list[str]) -> None:
    """One-push auto-match across the whole pad, then review. Holds each well's
    PF pressure, fits the IPR to its recent-test oil, reports the PF residual."""
    from woffl.gui.joint_match import batch_match, batch_summary

    with st.expander("🎯 Auto-match all wells (beta) — match the whole pad, then review",
                     expanded=False):
        import pandas as pd

        st.caption(
            "Pick the wells to auto-match (online wells start checked; default-offline "
            "ones unchecked). Each holds its PF pressure — the pad normal, overridable "
            "per well — fits the IPR to its recent-test oil, and reports the PF-rate "
            "residual. Review the table, then open the ones that need attention."
        )
        pad_ppf = st.number_input(
            "Pad PF header pressure (psi)", 800, 5500,
            {"M": 3500}.get(pad, 3400), 50, key=f"batch_ppf_{pad}",
            help="Seeds each well's held PF below. Edit a well's PF to override, or "
                 "hit Reset to re-seed everything to this value.",
        )

        # Stable selection frame (Well · Auto-match · PF psi). Rebuilt only when the
        # well set changes (NOT on every rerun — never feed the editor its own
        # return; that drops/dupes cells). Edits replay via the editor widget; we
        # read the return value. Default-offline wells start UNCHECKED.
        fkey = f"batch_sel_{pad}"
        if st.session_state.get(f"{fkey}_sig") != tuple(real_wells):
            st.session_state[fkey] = pd.DataFrame([
                {"Well": w, "Auto-match": not _is_default_offline(pad, w),
                 "PF psi": int(pad_ppf)} for w in real_wells
            ])
            st.session_state[f"{fkey}_sig"] = tuple(real_wells)

        edited = st.data_editor(
            st.session_state[fkey], key=f"{fkey}_editor", hide_index=True,
            use_container_width=True,
            column_config={
                "Well": st.column_config.TextColumn(disabled=True),
                "Auto-match": st.column_config.CheckboxColumn(default=True),
                "PF psi": st.column_config.NumberColumn(
                    min_value=800, max_value=5500, step=50, format="%d",
                    help="Held PF pressure for this well (defaults to the pad value)."),
            },
        )

        def _row_ppf(v):
            return float(v) if pd.notna(v) else float(pad_ppf)

        sel = [(str(r["Well"]), _row_ppf(r["PF psi"]))
               for _, r in edited.iterrows() if bool(r["Auto-match"])]

        b1, b2 = st.columns([2, 1])
        run = b1.button(f"🎯 Auto-match {len(sel)} selected well(s)", type="primary",
                        key=f"batch_run_{pad}", disabled=not sel, use_container_width=True)
        if b2.button("↻ Reset selection", key=f"batch_reset_{pad}", use_container_width=True):
            st.session_state.pop(fkey, None)
            st.session_state.pop(f"{fkey}_sig", None)
            st.session_state.pop(f"{fkey}_editor", None)
            st.rerun()

        if run:
            jp_hist = st.session_state.get("jp_history_df")
            wells_kwargs, raw_by_well, skipped = [], {}, []
            prog = st.progress(0.0, text="Building inputs…")
            for i, (w, ppf) in enumerate(sel):
                try:
                    kw, raw, why = _batch_automatch_inputs(w, jp_hist, ppf)
                except Exception as e:  # never let one well break the build loop
                    kw, raw, why = None, None, f"{type(e).__name__}"
                if kw is None:
                    skipped.append((w, why))
                else:
                    wells_kwargs.append((w, kw))
                    raw_by_well[w] = raw
                prog.progress((i + 1) / max(len(sel), 1),
                              text=f"Prepared {i + 1}/{len(sel)} wells…")
            with st.spinner(f"Matching {len(wells_kwargs)} wells…"):
                rows = batch_match(wells_kwargs)
            prog.empty()
            st.session_state[f"batch_rows_{pad}"] = rows
            st.session_state[f"batch_skipped_{pad}"] = skipped
            st.session_state[f"batch_raw_{pad}"] = raw_by_well

        rows = st.session_state.get(f"batch_rows_{pad}")
        skipped = st.session_state.get(f"batch_skipped_{pad}", [])
        if rows:
            import pandas as pd

            s = batch_summary(rows)
            st.caption(
                f"**matched {s['matched']} · partial {s['partial']} · "
                f"failed {s['failed']} · error {s['error']}**"
                + (f" · {len(skipped)} skipped (no data)" if skipped else "")
                + ".  'matched' = oil reproduced at the held PF pressure; "
                "**PF resid %** is how far modeled PF is from the test (nozzle/area)."
            )

            def _r(v):
                return round(v, 0) if (v is not None and v == v) else None

            df = pd.DataFrame([{
                "Well": r.well, "Status": r.status,
                "Oil err %": _r(r.oil_err_pct), "PF resid %": _r(r.pf_err_pct),
                "Note": (r.diagnostic or "")[:90],
            } for r in rows])
            st.dataframe(df, use_container_width=True, hide_index=True)
            if skipped:
                st.caption("Skipped: " + ", ".join(f"{w} ({why})" for w, why in skipped))

            # Apply the matched wells (oil reproduced at the held PF) straight into
            # the review store, so they're saved without re-doing each by hand.
            matched = [r for r in rows if r.status == "matched" and r.result is not None]
            raw_by_well = st.session_state.get(f"batch_raw_{pad}", {})
            if matched and st.button(
                f"💾 Apply {len(matched)} matched well(s) to the review store",
                key=f"batch_apply_{pad}",
                help="Saves each matched well's fitted IPR + friction (PF pressure held) "
                     "as a reviewed entry. Open any in the Solver below to adjust.",
            ):
                from woffl.gui.params import SimulationParams

                store = store_for(pad)
                applied = 0
                for r in matched:
                    raw, res = raw_by_well.get(r.well), r.result
                    if not raw or res is None:
                        continue
                    try:
                        p = SimulationParams(
                            selected_well=r.well,
                            nozzle_no=raw["nozzle_no"], area_ratio=raw["area_ratio"],
                            jpump_direction=raw["jpump_direction"],
                            ken=float(res.ken), kth=float(res.kth), kdi=float(res.kdi),
                            tubing_od=raw["tubing_od"], tubing_thickness=raw["tubing_thickness"],
                            casing_od=raw["casing_od"], casing_thickness=raw["casing_thickness"],
                            form_wc=round(float(raw["form_wc"]), 2), form_gor=int(raw["form_gor"]),
                            form_temp=int(raw["form_temp"]), field_model=raw["field_model"],
                            oil_api=raw["oil_api"], gas_sg=raw["gas_sg"], wat_sg=raw["wat_sg"],
                            bubble_point=raw["bubble_point"], surf_pres=int(raw["surf_pres"]),
                            jpump_tvd=int(raw["jpump_tvd"]), rho_pf=62.4,
                            ppf_surf=int(round(res.ppf_surf)),
                            qwf=int(round(res.qwf_oil)), pwf=int(round(res.pwf)),
                            pres=int(round(res.pres)),
                        )
                        off = store.get(r.well, {}).get(
                            "offline", _is_default_offline(pad, r.well))
                        store[r.well] = wrs.snapshot_from_params(
                            p, ipr_source="vogel",
                            bhp_source="gauged" if raw.get("has_bhp") else "assumed",
                            offline=off, pin_pf_pressure=True,
                        )
                        applied += 1
                    except Exception:
                        pass
                st.success(
                    f"Applied {applied} matched well(s) to the {pad}-Pad review store — "
                    "they're saved. Open any in the Solver below to fine-tune."
                )
                st.rerun()


def render_review_stage(pad: str) -> None:
    """Render the per-well review loop for ``pad``. Maintains ``store_for(pad)``."""
    real_wells = _pad_real_wells(pad)
    if not real_wells:
        st.warning(f"No characterized wells found for {pad}-Pad.")
        return

    # Warm every pad well's props/tests/survey in the background so swapping
    # between wells is snappy (and the batch auto-match hits warm caches).
    _prefetch_pad_wells(pad, real_wells)

    st.caption(
        "Open each well in the Solver, confirm the match, then **Save**. Saved "
        "wells drop off the pending list. Add future wells as hypotheticals. "
        "Download the review as a CSV to rerun or edit later."
    )

    # One-push pad-wide auto-match (review the table, then dig into problem wells).
    _render_batch_automatch(pad, real_wells)

    universe = real_wells

    # Apply the Save & next target (set by _render_save_panel) right before the
    # sidebar renders, so the Select Well dropdown follows to the next well.
    target = st.session_state.pop(f"sp_review_target_{pad}", None)
    if target in universe:
        _select_well(target)

    # Keep the sidebar selection inside the pad universe. A stale (non-pad) well
    # from a prior Single-Well session would otherwise render the wrong well.
    if st.session_state.get("selected_well") not in universe:
        nxt = next((w for w in real_wells if w not in store_for(pad)), real_wells[0])
        _select_well(nxt)

    _run_button, params = render_sidebar(well_filter=universe)

    # Hydrate sidebar from the saved entry the first time we land on a reviewed
    # well, so re-opening shows reviewed values (not a fresh IPR auto-populate).
    well = params.selected_well
    store = store_for(pad)
    if well in store and st.session_state.get(_LAST_HYDRATED_KEY) != well:
        _hydrate_sidebar_from_entry(store[well])
        st.session_state[_LAST_HYDRATED_KEY] = well
        st.rerun()
    if well not in store:
        st.session_state.pop(_LAST_HYDRATED_KEY, None)

    _render_progress(pad, real_wells)
    st.divider()
    jp_box, hero_box = _render_save_panel(params, real_wells, pad)
    _render_modeling_status(pad)
    _render_hypothetical_form(pad)
    _render_csv_io(pad)
    _render_ipr_pdf_button(pad)
    st.divider()

    # The reused Solver — same objects/renderer the Single-Well page uses.
    st.markdown("### Solver")
    from woffl.gui.single_well_page import _build_simulation_objects
    from woffl.gui.tabs import jetpump_solver

    try:
        jetpump, wellbore, inflow, res_mix, wp, _survey = _build_simulation_objects(params)
    except ValueError as e:
        st.error(
            f"Cannot build the model for {well}: {e}. Adjust the sidebar inputs "
            "(commonly flowing BHP must be below reservoir pressure)."
        )
        return

    # A single marginal well must never blank the whole page. Catch render
    # failures (e.g. a well the model can't converge) and degrade to a notice
    # instead of halting — but RE-RAISE Streamlit's control-flow exceptions
    # (st.rerun / st.stop), or we'd break the GOR auto-recovery rerun inside
    # render_tab and the page navigation.
    try:
        jetpump_solver.render_tab(
            params, jetpump, wellbore, wp, inflow, res_mix,
            hero_container=hero_box, jp_strip_container=jp_box,
        )
    except Exception as e:  # noqa: BLE001 — intentional last-resort guard
        if type(e).__name__ in ("RerunException", "StopException"):
            raise
        st.warning(
            f"Couldn't fully render the Solver for {well}: {e}. Your inputs "
            "above are still editable and saveable — this usually means the "
            "model couldn't converge for this well's current pump/IPR. Try the "
            "**Match test oil rate** button, nudge the pump, or mark the well "
            "offline if it's pulled."
        )
