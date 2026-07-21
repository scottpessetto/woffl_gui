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
from woffl.gui.pad_helpers import recent_test_rates as _recent_test_rates
from woffl.gui.sidebar import clamp_seed, render_sidebar
from woffl.gui.utils import (
    get_available_wells,
    get_well_tests_for_well,
    pad_from_mp_name,
)
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
        w for w in get_available_wells() if w != "Custom" and pad_from_mp_name(w) == pad
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
# Save-hook: pin the reviewed well's IPR anchor to Databricks (best-effort)
# ---------------------------------------------------------------------------


def _maybe_pin_saved_ipr(well: str) -> None:
    """Best-effort push of ``well``'s CURRENTLY-resolved IPR-anchor test as
    its saved default (``mpu.wells.prop_hist``'s ``ipr_wt_uid``), called from
    ``_save_and_advance`` AFTER the store save has already succeeded (W3 of
    the woffl-prop-hist-persistence plan).

    Shares the actual push mechanics with the Solver's "Save IPR as well
    default" button via ``woffl.gui.ipr_anchor.pin_ipr_anchor`` so the two
    push paths can never diverge. Never raises and never undoes the store
    save — a Databricks hiccup here must not lose the engineer's review:
      * gate off (``ALLOW_DATABRICKS_WRITES`` not set) → no push attempted,
        silent (this is the normal local-dev/CI state, not worth a caption
        on every save).
      * no resolvable test, or the anchor is a manual/provisional test
        (no ``wt_uid``) → skip with a one-line caption.
      * push raised (connection, enthid resolution) → ``st.warning`` with the
        message; the store save already succeeded.
      * success → clears the Solver's pin memo (so its next render re-queries
        instead of replaying the pre-save lookup) and shows a toast.

    ``well`` is always a REAL (non-hypothetical) well here: hypothetical
    wells are added directly to the store (see ``_render_hypothetical_form``)
    and never flow through ``_render_save_panel`` / ``_save_and_advance`` —
    ``render_review_stage`` restricts the sidebar's well selector to
    ``_pad_real_wells(pad)``.
    """
    from woffl.gui.ipr_anchor import PIN_SKIP_PREFIX, writes_enabled

    if not writes_enabled():
        return

    try:
        tests = get_well_tests_for_well(well)
    except Exception:
        tests = None

    anchor_row = None
    if tests is not None and not tests.empty:
        from woffl.gui.tabs.jetpump_solver import _resolve_anchor_test_row

        sorted_tests = tests.sort_values("WtDate", ascending=False).reset_index(
            drop=True
        )
        anchor_row = _resolve_anchor_test_row(well, sorted_tests)

    from woffl.gui.ipr_anchor import pin_ipr_anchor

    pushed, message = pin_ipr_anchor(well, anchor_row)
    if pushed:
        from woffl.gui.tabs.jetpump_solver import _clear_pin_cache

        _clear_pin_cache(well)
        st.toast(message, icon="📌")
    elif message.startswith(PIN_SKIP_PREFIX):
        st.caption(message)
    else:
        st.warning(message)


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

    for store_k, side_k in (
        ("ken_well", "ken"),
        ("kth_well", "kth"),
        ("kdi_well", "kdi"),
    ):
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

    # Circulation direction (store keeps lowercase; the radio uses title case).
    jd = str(entry.get("jpump_direction") or "").strip().lower()
    if jd in ("reverse", "forward"):
        st.session_state["jpump_direction"] = jd.title()
        st.session_state.pop("jpump_direction_input", None)


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
    st.caption(
        "🟢 gauged BHP  ·  🟡 assumed/no-gauge  ·  🔵 hypothetical  ·  ⚫ offline (excluded)"
    )


def _render_modeling_status(pad: str) -> None:
    """Quick per-well 'offline for this modeling' toggles (pulled / down wells)."""
    store = store_for(pad)
    if not store:
        return
    with st.expander(
        "🔌 Modeling status — exclude pulled / offline wells", expanded=False
    ):
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
                    f"{w} offline",
                    value=store[w].get("offline", False),
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
    # BHP / JPCO chart + pump-in-well timeline) sits ABOVE anchor_box (the IPR
    # anchor + comparison-test pickers — the *driving test* choice, hoisted here
    # because under the Solver heading nobody found it), which sits ABOVE
    # hero_box (pump banner + vs-actual metrics for that choice). Returned to
    # render_review_stage, which hands them to jetpump_solver.render_tab as
    # jp_strip_container / anchor_container / hero_container.
    jp_box = st.container()
    anchor_box = st.container()
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
            "IPR basis",
            options=list(wrs.IPR_SOURCES),
            index=list(wrs.IPR_SOURCES).index(inferred_ipr),
            help="How the IPR anchor was established (auto-detected; override if needed).",
            key=f"sp_ipr_src_{well}",
        )
    with c2:
        bhp_source = st.selectbox(
            "Match basis",
            options=list(wrs.BHP_SOURCES),
            index=0 if default_gauged else 1,
            help="'gauged' = a measured BHP backed the match; 'assumed' = no gauge, pwf is an estimate.",
            key=f"sp_bhp_src_{well}",
        )
    with c3:
        notes = st.text_input(
            "Note (optional)",
            key=f"sp_note_{well}",
            value=store.get(well, {}).get("notes", ""),
        )

    offline = st.checkbox(
        "🔌 Offline for this modeling (pulled / down — exclude from optimization)",
        # New wells on a pad's default-offline list start checked; once saved, the
        # stored value (the engineer's choice) wins.
        value=store.get(well, {}).get("offline", _is_default_offline(pad, well)),
        key=f"sp_offline_{well}",
    )

    jpump_md = params.well_data.get("JP_MD") if params.well_data else None

    def _save_and_advance(offline_flag: bool) -> None:
        """Snapshot → store → advance to the next unreviewed well. Shared by
        the Save button and the dedicated Offline button so the two paths
        can never drift."""
        try:
            entry = wrs.snapshot_from_params(
                params,
                ipr_source=ipr_source,
                bhp_source=bhp_source,
                offline=offline_flag,
                gauge_note=gauge_note,
                jpump_md=float(jpump_md) if jpump_md is not None else None,
                notes=notes,
            )
        except ValueError as e:
            # e.g. WC above what the oil optimizer can model — don't save, tell
            # the engineer exactly what to do instead.
            st.error(str(e))
            return
        store[well] = entry
        # The offline checkbox re-seeds from the stored value next render;
        # drop its widget state so a dedicated-button save shows through.
        st.session_state.pop(f"sp_offline_{well}", None)

        # Best-effort: pin this well's IPR anchor to Databricks so it becomes
        # the auto-default for every future session (W3 of the
        # woffl-prop-hist-persistence plan). AFTER the store write above —
        # this must never block or undo the save.
        _maybe_pin_saved_ipr(well)

        # Advance to the next UNREVIEWED well after this one (sequential), wrapping
        # to the first pending if we're at the end. Leave a one-shot target;
        # render_review_stage applies it right before the sidebar renders.
        if not already:
            i = real_wells.index(well) if well in real_wells else -1
            after = [w for w in real_wells[i + 1 :] if w not in store]
            nxt = (
                after[0]
                if after
                else next((w for w in real_wells if w not in store), None)
            )
            if nxt:
                st.session_state[f"sp_review_target_{pad}"] = nxt
        st.toast(
            f"Saved {well}" + (" — OFFLINE (excluded)" if offline_flag else ""),
            icon="🔌" if offline_flag else "✅",
        )
        st.rerun()

    # Two first-class actions side by side: the normal save (honors the
    # checkbox above) and a dedicated one-click "this well is down — exclude
    # it and move on" for pulled / dewatering wells, so marking a well
    # offline doesn't require spotting the checkbox first.
    btn_label = "💾 Update saved review" if already else "💾 Save & next well"
    off_label = "🔌 Offline — exclude" if already else "🔌 Offline & next well"
    c_save, c_off = st.columns([1.6, 1.0])
    with c_save:
        if st.button(
            btn_label,
            type="primary",
            use_container_width=True,
            key=f"sp_save_{well}",
        ):
            _save_and_advance(offline)
    with c_off:
        if st.button(
            off_label,
            use_container_width=True,
            key=f"sp_save_off_{well}",
            help="Save this well as OFFLINE (pulled / down / dewatering): it "
            "stays in the review set and pad accounting but is excluded from "
            "the optimization run. One click — no checkbox needed.",
        ):
            _save_and_advance(True)

    if already and st.button("🗑 Remove from review set", key=f"sp_del_{well}"):
        store.pop(well, None)
        st.session_state.pop(_LAST_HYDRATED_KEY, None)
        st.rerun()

    return jp_box, anchor_box, hero_box


def _render_hypothetical_form(pad: str) -> None:
    with st.expander("➕ Add a hypothetical / future well", expanded=False):
        st.caption(
            "Model adding a well to the pad. Enter an expected IPR + geometry; "
            "it feeds the optimizer like a real well but is flagged hypothetical."
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            name = st.text_input("Well name", value=f"MP{pad}-NEW", key="sp_hypo_name")
            field_model = st.radio(
                "Field model",
                ["Schrader", "Kuparuk"],
                horizontal=True,
                key="sp_hypo_fm",
            )
            res_pres = st.number_input(
                "Reservoir P (psi)", 400, 5000, 1800, 10, key="sp_hypo_resp"
            )
        with c2:
            oil = st.number_input(
                "Expected oil (BOPD)", 0, 6000, 400, 10, key="sp_hypo_oil"
            )
            pwf = st.number_input(
                "Flowing BHP (psi)", 100, 2500, 900, 10, key="sp_hypo_pwf"
            )
            wc = st.number_input(
                "Water cut", 0.0, 0.99, 0.5, 0.01, format="%.2f", key="sp_hypo_wc"
            )
        with c3:
            gor = st.number_input(
                "GOR (scf/bbl)", 20, 10000, 250, 25, key="sp_hypo_gor"
            )
            temp = st.number_input(
                "Form temp (°F)", 32, 350, 160, 1, key="sp_hypo_temp"
            )
            tvd = st.number_input(
                "Jetpump TVD (ft)", 2500, 8000, 4200, 10, key="sp_hypo_tvd"
            )

        if st.button("Add hypothetical well", key="sp_hypo_add"):
            name = name.strip()
            if not name:
                st.warning("Enter a well name.")
                return
            store_for(pad)[name] = wrs.hypothetical_entry(
                name,
                field_model=field_model,
                res_pres=res_pres,
                oil_bopd=oil,
                pwf=pwf,
                form_wc=wc,
                form_gor=gor,
                form_temp=temp,
                jpump_tvd=tvd,
            )
            st.toast(f"Added hypothetical {name}", icon="🔵")
            st.rerun()


def _render_ipr_pdf_button(pad: str) -> None:
    """Single-click download of the 'IPRs used' PDF — one IPR per reviewed well,
    with memory-gauge test points marked and a per-well provenance note."""
    store = store_for(pad)
    if not store:
        return
    if st.button(
        "⬇ Download IPRs-used PDF", use_container_width=True, key="sp_ipr_pdf"
    ):
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
        autodownload(
            pdf_bytes, f"{pad}_pad_IPRs_used.pdf", "application/pdf", "sp_ipr_pdf_dl"
        )


def _render_csv_io(pad: str) -> None:
    store = store_for(pad)
    c1, c2 = st.columns(2)
    with c1:
        if store and st.button(
            "⬇ Download review CSV", use_container_width=True, key=f"sp_csv_dl_{pad}"
        ):
            from woffl.gui.components.download import autodownload

            csv_bytes = (
                wrs.store_to_dataframe(store).to_csv(index=False).encode("utf-8")
            )
            autodownload(
                csv_bytes, f"{pad}_pad_review.csv", "text/csv", f"sp_csv_auto_dl_{pad}"
            )
    with c2:
        # Per-pad keys: this module backs the S/M/I pad pages, so a shared
        # uploader key/sig let one pad's upload carry over to (or be skipped on)
        # another pad when navigating between them.
        up = st.file_uploader("⬆ Load review CSV", type=["csv"], key=f"sp_csv_up_{pad}")
        if up is not None:
            sig = (up.name, up.size)
            if st.session_state.get(f"_sp_csv_sig_{pad}") != sig:
                import pandas as pd

                # dtype=str so pandas can't turn nozzle '10' into the float 10.0
                # (which then stringifies to '10.0' and breaks NOZZLE_OPTIONS).
                loaded = wrs.dataframe_to_store(pd.read_csv(up, dtype=str))
                store.update(loaded)
                # Loud validation: CSV holes coerce to 0.0 and then clamp to
                # plausible-looking defaults at run time — flag them per well
                # NOW (persisted across the rerun below).
                st.session_state[f"_sp_csv_issues_{pad}"] = wrs.validate_store(loaded)
                # Drop the offline-checkbox widget state for the loaded wells so
                # their value= re-seeds from the freshly loaded store. Otherwise
                # the checkbox returns its STALE persisted widget value and the
                # next _render_modeling_status pass writes it straight back over
                # the loaded "offline" flag (silently reverting it).
                for w in loaded:
                    st.session_state.pop(f"sp_status_off_{pad}_{w}", None)
                    st.session_state.pop(f"sp_offline_{w}", None)
                st.session_state[f"_sp_csv_sig_{pad}"] = sig
                st.success(f"Loaded {len(loaded)} well(s) from CSV.")
                st.rerun()

    issues = st.session_state.get(f"_sp_csv_issues_{pad}") or {}
    if issues:
        st.warning(
            f"**{len(issues)} uploaded well(s) have problems** — fix them (re-open "
            "in the Solver and re-save, or edit + re-upload the CSV) before running:"
        )
        for wn in sorted(issues):
            st.write(f"- **{wn}**: " + "; ".join(issues[wn]))
        if st.button("Dismiss", key=f"sp_csv_issues_dismiss_{pad}"):
            st.session_state.pop(f"_sp_csv_issues_{pad}", None)
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
                wd = get_well_data(w)  # props (TVD / field / geometry)
            except Exception:
                pass
            try:
                get_well_tests_for_well(w)  # tests + gauge overlay + extended window
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


# The Solver sidebar's default reservoir pressure — the anchor an unreviewed
# gaugeless well is ALREADY modeled with on the Solver page. Pump-limited
# wells barely feel ResP at the operating point, which is why that page
# matches most of them on pure defaults; the batch labels the assumption
# ("[IPR: assumed ResP 1700]") so it is never mistaken for a characterized
# pressure, and Apply stores it as ipr_source="forced".
_ASSUMED_RESP = 1700.0

# Skip reason shared between the input builder and the force-fit path, which
# converts these wells into OFFLINE store entries instead of leaving them
# dangling un-reviewed.
_NO_TESTS_REASON = "no well tests in cache"


def _offline_stub_entry(well: str) -> dict:
    """Minimal review-store entry for a well force-fit can't touch (no well
    tests → no oil/PF targets): saved OFFLINE so the pad reads fully
    processed and the optimizer excludes it. The placeholder physics only
    matter if the engineer later flips the well online — at which point
    they'd review it properly. Geometry/props from Databricks when available.
    """
    import pandas as pd

    from woffl.gui.utils import get_well_data

    wd = get_well_data(well) or {}

    def _opt(key: str):
        v = wd.get(key)
        try:
            if v is None or pd.isna(v):
                return None
        except (TypeError, ValueError):
            pass
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    tvd = float(wd.get("JP_TVD") or 4065)
    return {
        "well_name": well,
        "res_pres": 1700.0,
        "form_temp": float(wd.get("form_temp") or 120.0),
        "jpump_tvd": tvd,
        "tubing_od": float(wd.get("out_dia") or 4.5),
        "tubing_thickness": float(wd.get("thick") or 0.5),
        "casing_od": 6.875,
        "casing_thickness": 0.5,
        "form_wc": 0.5,
        "form_gor": 250.0,
        "surf_pres": 250.0,
        "qwf": 100.0,
        "pwf": 600.0,
        wrs.OIL_RATE_FIELD: 50.0,
        "jpump_md": tvd,
        "oil_api": _opt("oil_api"),
        "gas_sg": _opt("gas_sg"),
        "wat_sg": _opt("wat_sg"),
        "bubble_point": _opt("bubble_point"),
        "ppf_surf_well": None,
        "knz_well": None,
        "ken_well": None,
        "kth_well": None,
        "kdi_well": None,
        "jpump_direction": "reverse",
        "field_model": ("Schrader" if wd.get("is_sch", True) else "Kuparuk"),
        "review_nozzle": "",
        "review_throat": "",
        "ipr_source": "forced",
        "bhp_source": "assumed",
        "gauge_note": "",
        "is_hypothetical": False,
        "offline": True,
        "reviewed": True,
        "notes": "force-fit (offline — no well tests)",
    }


def _batch_automatch_inputs(
    well: str,
    jp_hist,
    ppf_surf: float,
    rho_pf: float = 62.4,
    store_entry: dict | None = None,
    fallback_pump: tuple[str, str] | None = None,
):
    """Assemble joint_match kwargs for one well from props + recent tests + the
    current pump — WITHOUT the sidebar (so the whole pad can be matched in one
    pass). Mirrors the sidebar's auto-populate field accesses. Returns
    (kwargs, raw, None) or (None, None, reason) when data is missing.

    ``store_entry`` (the well's saved review-store entry, if any) provides two
    fallbacks that make GAUGELESS wells matchable instead of skipped:

    * IPR anchor — ``estimate_reservoir_pressure`` drops wells with no valid
      BHP, so the Vogel fit has no row for them; the engineer's reviewed
      (res_pres, WC, GOR) stands in (they decided those in the Solver).
    * Friction seeds — reviewed ``ken_well``/``kth_well``/``kdi_well`` beat
      the Databricks ``jpfric_*`` defaults, which beat the generic library
      seeds (the single-well auto-match already got the sidebar's per-well
      coefs; the batch used to throw that calibration away).
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

    wd = get_well_data(well) or {}
    entry = store_entry or {}
    tests = get_well_tests_for_well(well)
    n_tests = 0 if tests is None or getattr(tests, "empty", True) else len(tests)
    # Targets are MEASURED (oil + PF from tests) — nothing to match without at
    # least one test. Everything else has a fallback tier.
    if n_tests == 0:
        return None, None, _NO_TESTS_REASON
    if not wd and not entry:
        # TVD/geometry can come from Databricks or a reviewed entry, but with
        # NEITHER, a generic-depth match would be garbage presented as truth.
        return None, None, "no well props (Databricks) and not reviewed"
    oil, pf = _recent_test_rates(well)
    if not oil or not pf:
        return None, None, "no recent oil/PF test"
    cp = get_current_pump(jp_hist, well) if jp_hist is not None else None
    if not (cp and cp.get("nozzle_no") and cp.get("throat_ratio")):
        # Tracker has no current install (S-Pad live run: MPS-17/25/54) — the
        # engineer's reviewed pump stands in; force-fit passes the sidebar
        # pump as a last resort so no well is left behind for pump identity.
        if entry.get("review_nozzle") and entry.get("review_throat"):
            cp = {
                "nozzle_no": entry["review_nozzle"],
                "throat_ratio": entry["review_throat"],
            }
        elif fallback_pump:
            cp = {"nozzle_no": fallback_pump[0], "throat_ratio": fallback_pump[1]}
        else:
            return None, None, "no current pump (tracker) and no reviewed pump"

    # ── IPR anchor, three tiers ──────────────────────────────────────────
    # 1. BHP-based Vogel fit (needs 2+ tests with BHP);
    # 2. the engineer's reviewed entry (they decided ResP/WC/GOR in the Solver);
    # 3. ASSUMED anchor — the same thing the Solver page itself does for an
    #    unreviewed gaugeless well (sidebar default ResP + test-derived WC/GOR),
    #    which models most of these pump-limited wells within ~10% (S-03 live:
    #    oil +11%, PF +7% on nothing but defaults). qmax is the matching knob,
    #    so the oil lands regardless; the label keeps the assumption visible
    #    and Apply stores it as ipr_source="forced", never "vogel".
    ipr_fallback = None
    fit_ok = False
    if n_tests >= 2:
        try:
            coeffs = compute_vogel_coefficients(estimate_reservoir_pressure(tests))
            row = coeffs[coeffs["Well"] == well].iloc[0]
            pres, wc, gor = (
                float(row["ResP"]), float(row["form_wc"]), float(row["fgor"])
            )
            fit_ok = True
        except Exception:
            pass  # gaugeless: no coeff row — fall through to the tiers below
    if not fit_ok:
        if entry.get("res_pres"):
            pres = float(entry["res_pres"])
            # Explicit None checks — `or` would turn a deliberately-reviewed
            # WC of 0.0 into 0.5 (0.0 is falsy), silently overriding the
            # engineer's decision.
            wc_v = entry.get("form_wc")
            wc = float(wc_v) if wc_v is not None else 0.5
            gor_v = entry.get("form_gor")
            gor = float(gor_v) if gor_v is not None else 250.0
            ipr_fallback = "reviewed entry"
        else:
            pres = _ASSUMED_RESP
            recent0 = tests.sort_values("WtDate", ascending=False).iloc[0]
            # WC/GOR derived from the recent test exactly the way
            # compute_vogel_coefficients derives them (water/total; fgor col).
            total = float(recent0.get("WtTotalFluid") or 0.0)
            water = float(recent0.get("WtWaterVol") or 0.0)
            wc = (water / total) if total > 0 else 0.5
            g = recent0.get("fgor")
            gor = float(g) if (g is not None and not pd.isna(g)) else 250.0
            ipr_fallback = f"assumed ResP {int(_ASSUMED_RESP)}"

    recent = tests.sort_values("WtDate", ascending=False).iloc[0]
    bhp = recent.get("BHP")
    bhp = (
        float(bhp)
        if (bhp is not None and not pd.isna(bhp) and float(bhp) > 0)
        else None
    )
    whp = recent.get("whp")
    if whp is not None and not pd.isna(whp) and float(whp) > 0:
        surf = float(whp)
    elif store_entry and store_entry.get("surf_pres"):
        surf = float(store_entry["surf_pres"])
    else:
        surf = 250.0
    # Geometry: Databricks props → reviewed entry → defaults (a reviewed well
    # with no Databricks row must still be matchable).
    tvd = int(wd.get("JP_TVD") or entry.get("jpump_tvd") or 4065)
    if "is_sch" in wd:
        field = "schrader" if wd.get("is_sch", True) else "kuparuk"
    elif entry.get("field_model") in ("Schrader", "Kuparuk"):
        field = entry["field_model"].lower()
    else:
        field = "schrader"
    tub_od = float(wd.get("out_dia") or entry.get("tubing_od") or 4.5)
    tub_th = float(wd.get("thick") or entry.get("tubing_thickness") or 0.5)
    try:
        _, _, wellbore = create_pipes(tub_od, tub_th, 6.875, 0.5)
        wp = create_well_profile_from_survey(well, tvd, field)
    except Exception:
        return None, None, "geometry/survey build failed"

    # Circulation direction: tracker Circulating (current install, enriched
    # at fetch) → live-pressure inference (PF on the tubing side ⇒ forward,
    # e.g. MPS-17) → reverse. Mirrors the sidebar's direction seed so batch
    # and single-well reviews agree.
    from woffl.gui.pump_identity import tracker_direction
    from woffl.gui.utils import live_pf_for_seed

    trk = tracker_direction(jp_hist, well)
    if trk:
        direction = trk
    else:
        live_pf = live_pf_for_seed(well)
        direction = (
            "forward"
            if (live_pf and live_pf.get("pf_source") == "tubing")
            else "reverse"
        )

    # Friction seeds: reviewed entry → Databricks jpfric_* → library defaults.
    # The single-well auto-match passes the sidebar's per-well coefs; the batch
    # used to fall back to generics for every well, discarding calibration.
    from woffl.gui.scotts_tools._common import friction_coefs_from_chars

    fc = friction_coefs_from_chars(wd)
    ken0 = kth0 = kdi0 = None
    if store_entry:
        ken0 = store_entry.get("ken_well")
        kth0 = store_entry.get("kth_well")
        kdi0 = store_entry.get("kdi_well")
    ken0 = float(ken0) if ken0 is not None else float(fc.get("ken", 0.03))
    kth0 = float(kth0) if kth0 is not None else float(fc.get("kth", 0.3))
    kdi0 = float(kdi0) if kdi0 is not None else float(fc.get("kdi", 0.4))

    form_temp = float(wd.get("form_temp") or entry.get("form_temp") or 120.0)

    def _pvt(wk: str):
        v = wd.get(wk)
        return v if v is not None else entry.get(wk)

    kwargs = dict(
        oil_target=oil,
        pf_target=pf,
        pres=pres,
        nozzle=str(cp["nozzle_no"]),
        throat=str(cp["throat_ratio"]),
        surf_pres=surf,
        form_temp=form_temp,
        rho_pf=rho_pf,
        ppf_surf0=float(ppf_surf),
        wellbore=wellbore,
        well_profile=wp,
        form_wc=wc,
        form_gor=gor,
        ken0=ken0,
        kth0=kth0,
        kdi0=kdi0,
        field_model=field,
        jpump_direction=direction,
        bhp_target=bhp,
        pin_ppf=True,
    )
    # Raw scalars to rebuild a SimulationParams for "apply to store" (WellConfig
    # wants the capitalized field-model; the joint_match kwargs above keep the
    # lowercase the PVT helpers expect).
    raw = {
        "nozzle_no": str(cp["nozzle_no"]),
        "area_ratio": str(cp["throat_ratio"]),
        "jpump_direction": direction,
        "tubing_od": tub_od,
        "tubing_thickness": tub_th,
        "casing_od": 6.875,
        "casing_thickness": 0.5,
        "form_wc": float(wc),
        "form_gor": int(round(gor)),
        "form_temp": int(round(form_temp)),
        "field_model": field.title(),
        "oil_api": _pvt("oil_api"),
        "gas_sg": _pvt("gas_sg"),
        "wat_sg": _pvt("wat_sg"),
        "bubble_point": _pvt("bubble_point"),
        "surf_pres": int(round(surf)),
        "jpump_tvd": int(tvd),
        "has_bhp": bhp is not None,
        "ipr_fallback": ipr_fallback,  # None, or "reviewed entry" (gaugeless)
    }
    return kwargs, raw, None


def _apply_batch_row(pad: str, store: dict, r, raw, *, note: str = "") -> str | None:
    """Write one batch-match row into the review store.

    Shared by the matched-only Apply button and the force-fit path (which
    passes a ``note`` marker like "force-fit (partial)" so hand-reviewed
    entries stay distinguishable from bulk saves). Returns an error string
    on failure, None on success — the caller aggregates.
    """
    from woffl.gui.params import SimulationParams

    res = r.result
    if not raw or res is None:
        return f"{r.well}: no result to apply"
    try:
        p = SimulationParams(
            selected_well=r.well,
            nozzle_no=raw["nozzle_no"],
            area_ratio=raw["area_ratio"],
            jpump_direction=raw["jpump_direction"],
            ken=float(res.ken),
            kth=float(res.kth),
            kdi=float(res.kdi),
            tubing_od=raw["tubing_od"],
            tubing_thickness=raw["tubing_thickness"],
            casing_od=raw["casing_od"],
            casing_thickness=raw["casing_thickness"],
            form_wc=round(float(raw["form_wc"]), 2),
            form_gor=int(raw["form_gor"]),
            form_temp=int(raw["form_temp"]),
            field_model=raw["field_model"],
            oil_api=raw["oil_api"],
            gas_sg=raw["gas_sg"],
            wat_sg=raw["wat_sg"],
            bubble_point=raw["bubble_point"],
            surf_pres=int(raw["surf_pres"]),
            jpump_tvd=int(raw["jpump_tvd"]),
            rho_pf=62.4,
            ppf_surf=int(round(res.ppf_surf)),
            qwf=int(round(res.qwf_oil)),
            pwf=int(round(res.pwf)),
            pres=int(round(res.pres)),
        )
        off = store.get(r.well, {}).get("offline", _is_default_offline(pad, r.well))
        store[r.well] = wrs.snapshot_from_params(
            p,
            # Fallback-anchored wells keep "forced" provenance — their ResP is
            # an engineer's decision or an assumption, never a Vogel fit.
            ipr_source=("forced" if raw.get("ipr_fallback") else "vogel"),
            bhp_source="gauged" if raw.get("has_bhp") else "assumed",
            offline=off,
            pin_pf_pressure=True,
            notes=note,
        )
        # Deliberately NOT calling _maybe_pin_saved_ipr here: batch auto-match
        # anchors are machine-fit, not user-reviewed, so this must not pin an
        # IPR default to Databricks (plan woffl-prop-hist-persistence W3(d) —
        # flagged for Scott to revisit once he trusts the batch match).
        return None
    except Exception as e:
        # Surface, don't swallow: a systematic failure (e.g. a bad field)
        # otherwise looked like a partial success.
        return f"{r.well}: {type(e).__name__}: {e}"


def _render_batch_automatch(pad: str, real_wells: list[str]) -> None:
    """One-push auto-match across the whole pad, then review. Holds each well's
    PF pressure, fits the IPR to its recent-test oil, reports the PF residual."""
    from woffl.gui.joint_match import batch_match, batch_summary

    with st.expander(
        "🎯 Auto-match all wells (beta) — match the whole pad, then review",
        # Default OPEN: the batch match / force-fit is the first thing Scott
        # reaches for on a pad — collapsing it in-session still sticks for
        # the rest of the visit.
        expanded=True,
    ):
        import pandas as pd

        st.caption(
            "Pick the wells to auto-match (online wells start checked; default-offline "
            "ones unchecked). Each holds its PF pressure — seeded per well from LIVE "
            "daily data (vw_pressure_daily, most-recent-test day), pad normal as "
            "fallback, overridable per well — fits the IPR to its recent-test oil, "
            "and reports the PF-rate residual. Review the table, then open the ones "
            "that need attention."
        )
        pad_ppf = st.number_input(
            "Pad PF header pressure (psi)",
            800,
            5500,
            {"M": 3500}.get(pad, 3400),
            50,
            key=f"batch_ppf_{pad}",
            help="Fallback for wells without a live PF reading. Wells with live "
            "data seed from their own measured PF; edit a well's PF to override, "
            "or hit Reset to re-seed everything.",
        )

        # Stable selection frame (Well · Auto-match · PF psi). Rebuilt only when the
        # well set changes (NOT on every rerun — never feed the editor its own
        # return; that drops/dupes cells). Edits replay via the editor widget; we
        # read the return value. Default-offline wells start UNCHECKED.
        # Per-well PF seeds from live daily data (test-day of the most recent
        # test, else latest daily reading), falling back to the pad value.
        from woffl.gui.utils import live_pf_for_seed

        def _live_or_pad_ppf(w: str) -> int:
            live = live_pf_for_seed(w)
            if live is not None:
                return int(round(live["pf_press"]))
            return int(pad_ppf)

        fkey = f"batch_sel_{pad}"
        if st.session_state.get(f"{fkey}_sig") != tuple(real_wells):
            st.session_state[fkey] = pd.DataFrame(
                [
                    {
                        "Well": w,
                        "Auto-match": not _is_default_offline(pad, w),
                        "PF psi": _live_or_pad_ppf(w),
                    }
                    for w in real_wells
                ]
            )
            st.session_state[f"{fkey}_sig"] = tuple(real_wells)

        edited = st.data_editor(
            st.session_state[fkey],
            key=f"{fkey}_editor",
            hide_index=True,
            use_container_width=True,
            column_config={
                "Well": st.column_config.TextColumn(disabled=True),
                "Auto-match": st.column_config.CheckboxColumn(default=True),
                "PF psi": st.column_config.NumberColumn(
                    min_value=800,
                    max_value=5500,
                    step=50,
                    format="%d",
                    help="Held PF pressure for this well — live daily reading "
                    "when available, else the pad value.",
                ),
            },
        )

        def _row_ppf(v):
            return float(v) if pd.notna(v) else float(pad_ppf)

        sel = [
            (str(r["Well"]), _row_ppf(r["PF psi"]))
            for _, r in edited.iterrows()
            if bool(r["Auto-match"])
        ]

        b1, b2, b3 = st.columns([2, 2, 1])
        run = b1.button(
            f"🎯 Auto-match {len(sel)} selected well(s)",
            type="primary",
            key=f"batch_run_{pad}",
            disabled=not sel,
            use_container_width=True,
        )
        force = b2.button(
            f"⚡ Force-fit + save {len(sel)} well(s)",
            key=f"batch_force_{pad}",
            disabled=not sel,
            use_container_width=True,
            help=(
                "One click to a fully-populated pad: auto-match every selected "
                "well and save EVERY result (matched, partial, or failed — "
                "best-found params) straight into the review store. Wells with "
                "no pump anywhere use the sidebar pump. Hand-reviewed wells "
                "are protected unless the overwrite box is ticked. Then go run "
                "the optimization and come back to fine-tune."
            ),
        )
        if b3.button(
            "↻ Reset", key=f"batch_reset_{pad}", use_container_width=True
        ):
            st.session_state.pop(fkey, None)
            st.session_state.pop(f"{fkey}_sig", None)
            st.session_state.pop(f"{fkey}_editor", None)
            st.rerun()
        force_overwrite = st.checkbox(
            "Force-fit may overwrite hand-reviewed wells",
            value=False,
            key=f"batch_force_ow_{pad}",
            help=(
                "Off (default): force-fit only fills wells that aren't in the "
                "review store yet (or were themselves force-fit before) — your "
                "hand-tuned reviews are never touched. On: every selected well "
                "is re-fit and overwritten."
            ),
        )

        # One-shot force-fit summary from the previous run (the rerun that
        # refreshed the review progress cleared anything rendered inline).
        _fmsg = st.session_state.pop(f"batch_force_msg_{pad}", None)
        if _fmsg:
            st.success(_fmsg["ok"])
            if _fmsg.get("detail"):
                st.caption(_fmsg["detail"])

        if force:
            jp_hist = st.session_state.get("jp_history_df")
            store = store_for(pad)
            side_pump = (
                str(st.session_state.get("nozzle_no", "12")),
                str(st.session_state.get("area_ratio", "B")),
            )
            wells_kwargs, raw_by_well = [], {}
            skipped: list[tuple[str, str]] = []
            protected: list[str] = []
            prog = st.progress(0.0, text="Building inputs…")
            for i, (w, ppf) in enumerate(sel):
                if (
                    not force_overwrite
                    and w in store
                    and not str(store[w].get("notes", "")).startswith("force-fit")
                ):
                    protected.append(w)
                else:
                    try:
                        kw, raw, why = _batch_automatch_inputs(
                            w,
                            jp_hist,
                            ppf,
                            store_entry=store.get(w),
                            fallback_pump=side_pump,
                        )
                    except Exception as e:
                        kw, raw, why = None, None, f"{type(e).__name__}"
                    if kw is None:
                        skipped.append((w, why))
                    else:
                        wells_kwargs.append((w, kw))
                        raw_by_well[w] = raw
                prog.progress(
                    (i + 1) / max(len(sel), 1),
                    text=f"Prepared {i + 1}/{len(sel)} wells…",
                )
            with st.spinner(f"Force-fitting {len(wells_kwargs)} wells…"):
                rows = batch_match(wells_kwargs)
            prog.empty()

            applied, errors_ = 0, []
            for r in rows:
                if r.result is None:
                    skipped.append((r.well, "match crashed"))
                    continue
                err = _apply_batch_row(
                    pad,
                    store,
                    r,
                    raw_by_well.get(r.well),
                    note=f"force-fit ({r.status})",
                )
                if err:
                    errors_.append(err)
                else:
                    applied += 1

            # No-test wells can't be matched (no targets) — default them to
            # OFFLINE store entries so the pad reads fully processed and the
            # optimizer excludes them, instead of leaving them dangling.
            defaulted_off: list[str] = []
            for w, why in skipped:
                if why != _NO_TESTS_REASON:
                    continue
                if w in store:
                    store[w]["offline"] = True
                else:
                    store[w] = _offline_stub_entry(w)
                defaulted_off.append(w)
            skipped = [(w, why) for w, why in skipped if w not in defaulted_off]

            st.session_state[f"batch_rows_{pad}"] = rows
            st.session_state[f"batch_skipped_{pad}"] = skipped
            st.session_state[f"batch_raw_{pad}"] = raw_by_well
            detail_bits = []
            if defaulted_off:
                detail_bits.append(
                    "defaulted OFFLINE (no well tests): " + ", ".join(defaulted_off)
                )
            if protected:
                detail_bits.append(
                    f"protected (hand-reviewed, untouched): {', '.join(protected)}"
                )
            if skipped:
                detail_bits.append(
                    "couldn't fit: "
                    + ", ".join(f"{w} ({why})" for w, why in skipped)
                )
            if errors_:
                detail_bits.append("errors: " + "; ".join(errors_))
            st.session_state[f"batch_force_msg_{pad}"] = {
                "ok": (
                    f"⚡ Force-fit saved {applied} well(s) to the {pad}-Pad "
                    "review store"
                    + (
                        f" (+{len(defaulted_off)} defaulted offline)"
                        if defaulted_off
                        else ""
                    )
                    + " — head to Configure & Run, then come back and "
                    "fine-tune the rough ones (see the table below; "
                    "force-fit entries are tagged in their notes)."
                ),
                "detail": "  ·  ".join(detail_bits),
            }
            st.rerun()

        if run:
            jp_hist = st.session_state.get("jp_history_df")
            wells_kwargs, raw_by_well, skipped = [], {}, []
            prog = st.progress(0.0, text="Building inputs…")
            for i, (w, ppf) in enumerate(sel):
                try:
                    kw, raw, why = _batch_automatch_inputs(
                        w, jp_hist, ppf, store_entry=store_for(pad).get(w)
                    )
                except Exception as e:  # never let one well break the build loop
                    kw, raw, why = None, None, f"{type(e).__name__}"
                if kw is None:
                    skipped.append((w, why))
                else:
                    wells_kwargs.append((w, kw))
                    raw_by_well[w] = raw
                prog.progress(
                    (i + 1) / max(len(sel), 1),
                    text=f"Prepared {i + 1}/{len(sel)} wells…",
                )
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

            _raw_map = st.session_state.get(f"batch_raw_{pad}", {})

            def _note(r):
                fb = (_raw_map.get(r.well) or {}).get("ipr_fallback")
                prefix = f"[IPR: {fb}] " if fb else ""
                # 200 chars keeps the model-ceiling tail (the actionable part
                # of an oil-short diagnostic) visible in the table.
                return (prefix + (r.diagnostic or ""))[:200]

            df = pd.DataFrame(
                [
                    {
                        "Well": r.well,
                        "Status": r.status,
                        "Oil err %": _r(r.oil_err_pct),
                        "PF resid %": _r(r.pf_err_pct),
                        "Note": _note(r),
                    }
                    for r in rows
                ]
            )
            st.dataframe(df, use_container_width=True, hide_index=True)
            if skipped:
                st.caption(
                    "Skipped: " + ", ".join(f"{w} ({why})" for w, why in skipped)
                )

            # Apply the matched wells (oil reproduced at the held PF) straight into
            # the review store, so they're saved without re-doing each by hand.
            matched = [
                r for r in rows if r.status == "matched" and r.result is not None
            ]
            raw_by_well = st.session_state.get(f"batch_raw_{pad}", {})
            if matched and st.button(
                f"💾 Apply {len(matched)} matched well(s) to the review store",
                key=f"batch_apply_{pad}",
                help="Saves each matched well's fitted IPR + friction (PF pressure held) "
                "as a reviewed entry. Open any in the Solver below to adjust.",
            ):
                store = store_for(pad)
                applied = 0
                skipped: list[str] = []
                for r in matched:
                    err = _apply_batch_row(pad, store, r, raw_by_well.get(r.well))
                    if err:
                        skipped.append(err)
                    else:
                        applied += 1
                st.success(
                    f"Applied {applied} matched well(s) to the {pad}-Pad review store — "
                    "they're saved. Open any in the Solver below to fine-tune."
                )
                if skipped:
                    with st.expander(
                        f"{len(skipped)} well(s) could not be applied", expanded=False
                    ):
                        for msg in skipped:
                            st.write(f"- {msg}")
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
    jp_box, anchor_box, hero_box = _render_save_panel(params, real_wells, pad)
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
        jetpump, wellbore, inflow, res_mix, wp, _survey = _build_simulation_objects(
            params
        )
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
            params,
            jetpump,
            wellbore,
            wp,
            inflow,
            res_mix,
            hero_container=hero_box,
            jp_strip_container=jp_box,
            anchor_container=anchor_box,
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
