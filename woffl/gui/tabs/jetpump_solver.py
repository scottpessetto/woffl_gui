"""Tab 1: Jetpump Solver Results

Renders the single-pump solution display showing suction pressure,
oil rate, water rate, power fluid rate, and sonic status.

When JP history is uploaded and a non-Custom well is selected,
also shows a "Model vs Actual" comparison section with IPR chart
and modeled vs actual metrics.
"""

import logging

import streamlit as st

from woffl.gui.fric_calibration import calibrate_friction_coefs
from woffl.gui.params import SimulationParams
from woffl.gui.utils import (
    _trigger_gor_reset,
    build_calibration_inputs,
    create_pvt_components,
    is_valid_number,
    render_bhp_calibration_warning,
    render_input_summary,
    render_pf_mismatch_warning,
    render_pf_quickfix_widget,
    run_jetpump_solver,
)

logger = logging.getLogger(__name__)


def _render_memory_gauge_section(well_name: str) -> None:
    """Memory-gauge upload + status block at the top of the Solver tab.

    Two surfaces:
      * **Persistent banner** when a gauge is active — coverage window,
        sample count, and tests matched. Visible without expanding anything.
      * **Collapsible expander** with the file uploader and preview. Stays
        collapsed by default so the page isn't cluttered for wells where
        gauge data isn't needed.

    Custom mode is skipped — gauge overrides are per-well, and Custom
    has no well-test data to override against.
    """
    if well_name == "Custom":
        return

    from woffl.gui.memory_gauge import (
        add_file_to_gauge,
        clear_extended_tests,
        clear_gauge,
        compute_databricks_vs_gauge_delta,
        coverage_summary,
        fetch_databricks_bhp_daily,
        fetch_extended_tests,
        get_gauge,
        get_pending_files,
        is_disregarding_databricks_bhp,
        parse_xlsx,
        remove_file_from_gauge,
        set_disregard_databricks_bhp,
        set_pending_files,
        store_extended_tests,
    )
    from woffl.gui.utils import get_well_tests_for_well

    # Surface any one-shot warning persisted from a prior Apply (extended-
    # fetch failure couldn't render mid-button-handler because the rerun
    # cleared the page).
    warn_msg = st.session_state.pop("_mg_apply_warning", None)
    if warn_msg:
        st.warning(warn_msg)

    gauge = get_gauge(well_name)
    disregard = is_disregarding_databricks_bhp(well_name)
    pending = get_pending_files(well_name)

    # Persistent status banner — visible without expanding the upload box.
    # Composes both states: gauge upload and "disregard Databricks BHP".
    if gauge is not None or disregard:
        # Surface a one-shot auto-divergence note (set on Apply) right
        # above the banner so users see WHY the disregard flag flipped on.
        auto_note = st.session_state.pop("_mg_auto_disregard_msg", None)
        if auto_note:
            st.warning(auto_note, icon="⚠️")

        bn_cols = st.columns([5, 1])

        with bn_cols[0]:
            if gauge is not None:
                well_tests_df = get_well_tests_for_well(well_name)
                cov = (
                    coverage_summary(well_tests_df, gauge)
                    if well_tests_df is not None
                    else None
                )
                file_count = len(gauge.files)
                file_str = f" ({file_count} file{'s' if file_count != 1 else ''})"
                date_range = (
                    f"{gauge.start_date.strftime('%Y-%m-%d')} → "
                    f"{gauge.end_date.strftime('%Y-%m-%d')}"
                )
                cov_str = (
                    f" · {cov['tests_matched']}/{cov['tests_total']} tests matched"
                    if cov
                    else ""
                )
                disregard_str = " · **Databricks BHP disregarded**" if disregard else ""
                st.info(
                    f"**Memory gauge active for {well_name}**{file_str} — "
                    f"{date_range} ({gauge.sample_count:,} samples)"
                    f"{cov_str}{disregard_str}. Gauge BHP replaces Databricks "
                    f"across the Solver, IPR fit, and JP history views.",
                    icon="📊",
                )
            else:
                # Disregard only — no gauge uploaded
                st.warning(
                    f"**Databricks BHP disregarded for {well_name}** — "
                    "no BHP data will be used in the Solver, IPR fit, or "
                    "Model vs Actual until you upload a memory gauge.",
                    icon="🚫",
                )

        with bn_cols[1]:
            if gauge is not None:
                if st.button(
                    "Clear gauge",
                    key=f"mg_clear_btn_{well_name}",
                    use_container_width=True,
                    help="Remove the memory-gauge override. The disregard flag is independent — uncheck it inside the expander to also restore Databricks BHP.",
                ):
                    clear_gauge(well_name)
                    clear_extended_tests(well_name)
                    st.session_state["_force_ipr_refresh"] = True
                    st.rerun()

    # Unmissable nudge when files were uploaded but never applied — the
    # pending stash survives a view detour (that's its whole point), so make
    # sure the user knows the files are waiting rather than in effect.
    if pending:
        n_pend = len(pending)
        st.warning(
            f"**{n_pend} gauge file{'s' if n_pend != 1 else ''} uploaded but "
            f"not yet applied** for {well_name} — nothing uses gauge BHP "
            "until you click **Apply** inside *Memory Gauge Data* below.",
            icon="📥",
        )

    if gauge is not None:
        expander_label = (
            f"Memory Gauge Data — {len(gauge.files)} file"
            f"{'s' if len(gauge.files) != 1 else ''} loaded"
        )
    else:
        expander_label = (
            "Memory Gauge Data — upload BHP for wells without a permanent gauge"
        )
    if pending:
        expander_label += (
            f" · {len(pending)} file{'s' if len(pending) != 1 else ''} pending Apply"
        )
    # Default open when uploads are pending, so a view detour (which resets
    # the expander's widget state to this default) reopens on the waiting
    # files instead of hiding them.
    with st.expander(expander_label, expanded=bool(pending)):
        st.caption(
            "Upload one or more XLSX files from downhole memory gauges (each "
            "must have a 'Date Time' column and a 'Pressure' column). Multiple "
            "files for the same well are combined into a single daily-median "
            "BHP series — useful when gauges get pulled and re-hung over time. "
            "Used in place of Databricks BHP across the Solver, IPR fit, and "
            "JP history views — for this well only, for this session."
        )

        # Manual disregard control. Independent of gauge upload: a well
        # with a known-bad Databricks feed can be flagged before (or even
        # without) uploading a gauge. The Apply handler may also auto-tick
        # this when the divergence check trips.
        disregard_widget_key = f"mg_disregard_cb_{well_name}"
        if disregard_widget_key not in st.session_state:
            st.session_state[disregard_widget_key] = disregard

        def _on_disregard_toggle() -> None:
            new_val = bool(st.session_state.get(disregard_widget_key, False))
            set_disregard_databricks_bhp(well_name, new_val)
            st.session_state["_force_ipr_refresh"] = True

        st.checkbox(
            "Disregard Databricks BHP for this well",
            key=disregard_widget_key,
            on_change=_on_disregard_toggle,
            help=(
                "Use this when the well has a Databricks BHP feed that is "
                "known to be wrong. The bad values are dropped before any "
                "memory-gauge data is applied, so the Solver and IPR fit "
                "only see gauge BHP (or no BHP at all if no gauge is loaded)."
            ),
        )

        # Loaded-files list — one row per file with a Remove button. Sorted
        # by start_date so the oldest is on top (matches how the user thinks
        # about gauge runs chronologically).
        if gauge is not None:
            st.markdown("**Loaded files:**")
            for f in sorted(gauge.files, key=lambda x: x.start_date):
                col_text, col_btn = st.columns([5, 1])
                with col_text:
                    st.markdown(
                        f"📄 `{f.source_filename}` — "
                        f"{f.start_date.strftime('%Y-%m-%d')} → "
                        f"{f.end_date.strftime('%Y-%m-%d')} · "
                        f"{f.sample_count:,} samples"
                    )
                with col_btn:
                    # Sanitize filename for the widget key — Streamlit keys
                    # don't tolerate '/', '\', or other special chars.
                    safe_name = "".join(
                        c if c.isalnum() else "_" for c in f.source_filename
                    )
                    if st.button(
                        "Remove",
                        key=f"mg_remove_{well_name}_{safe_name}",
                        use_container_width=True,
                    ):
                        remove_file_from_gauge(well_name, f.source_filename)
                        # Removing the last file clears the gauge; clear
                        # the extended tests + disregard flag too for clean
                        # revert to Databricks state.
                        if not get_gauge(well_name):
                            clear_extended_tests(well_name)
                        st.session_state["_force_ipr_refresh"] = True
                        st.rerun()
            st.divider()

        # The uploader uses a counter in its key so it can be "reset" by
        # incrementing the counter after its files are absorbed into the
        # pending stash (Streamlit has no API to programmatically clear a
        # file_uploader).
        counter_key = f"_mg_upload_counter_{well_name}"
        upload_counter = st.session_state.get(counter_key, 0)
        upload_label = (
            "Upload more gauge files"
            if gauge is not None
            else "Memory gauge XLSX file(s)"
        )
        uploaded = st.file_uploader(
            upload_label,
            type=["xlsx"],
            accept_multiple_files=True,
            key=f"mg_upload_{well_name}_{upload_counter}",
            help=(
                "Drop one or more files at once — each gauge run is combined "
                "into a single daily-median BHP history for the well."
            ),
        )

        # Absorb dropped files into the pending stash IMMEDIATELY, then reset
        # the uploader (bump its key counter) and rerun. Files living only in
        # the uploader's widget state were silently dropped by a view detour
        # (Solver → Batch Run → Solver): the segmented control runs only the
        # active view, and Streamlit garbage-collects the widget state of
        # anything that didn't render (see CLAUDE.md). The stash is a plain
        # session key, so pending files survive the detour — and each file is
        # parsed exactly once instead of on every rerun. Skip files already in
        # the gauge or the stash (by filename); collect per-file errors so one
        # bad file doesn't block the good ones.
        if uploaded:
            gauge_names = (
                {f.source_filename for f in gauge.files} if gauge is not None else set()
            )
            pending_names = {f.source_filename for f in pending}
            new_files: list = []
            notes: list[tuple[str, str]] = []
            skipped: list[str] = []
            for uf in uploaded:
                if uf.name in gauge_names:
                    skipped.append(f"`{uf.name}` (already loaded)")
                    continue
                if uf.name in pending_names:
                    skipped.append(f"`{uf.name}` (already pending)")
                    continue
                try:
                    new_files.append(parse_xlsx(uf.getvalue(), uf.name))
                    pending_names.add(uf.name)
                except Exception as e:
                    notes.append(("error", f"Could not parse `{uf.name}`: {e}"))
            if skipped:
                notes.append(("info", "Skipped " + ", ".join(skipped)))
            if new_files:
                set_pending_files(well_name, pending + new_files)
            if notes:
                st.session_state["_mg_upload_notes"] = notes
            st.session_state[counter_key] = upload_counter + 1
            st.rerun()

        # One-shot parse/skip messages from the absorption pass — the rerun
        # that reset the uploader wiped anything rendered inline there.
        for kind, msg in st.session_state.pop("_mg_upload_notes", []):
            (st.error if kind == "error" else st.info)(msg)

        if not pending:
            return

        # Pending files — parsed and held in session state, but NOT part of
        # the gauge until Apply. Mirrors the loaded-files list above, with
        # per-file Discard instead of Remove.
        st.markdown("**Uploaded — not yet applied:**")
        for f in sorted(pending, key=lambda x: x.start_date):
            col_text, col_btn = st.columns([5, 1])
            with col_text:
                st.markdown(
                    f"📄 `{f.source_filename}` — "
                    f"{f.start_date.strftime('%Y-%m-%d')} → "
                    f"{f.end_date.strftime('%Y-%m-%d')} · "
                    f"{f.sample_count:,} samples"
                )
            with col_btn:
                safe_name = "".join(
                    c if c.isalnum() else "_" for c in f.source_filename
                )
                if st.button(
                    "Discard",
                    key=f"mg_discard_{well_name}_{safe_name}",
                    use_container_width=True,
                ):
                    set_pending_files(
                        well_name,
                        [
                            p
                            for p in pending
                            if p.source_filename != f.source_filename
                        ],
                    )
                    st.rerun()

        # Combined preview across the pending files.
        total_samples = sum(f.sample_count for f in pending)
        new_start = min(f.start_date for f in pending)
        new_end = max(f.end_date for f in pending)
        c1, c2, c3 = st.columns(3)
        c1.metric("New files", f"{len(pending)}")
        c2.metric("Samples", f"{total_samples:,}")
        c3.metric(
            "Coverage",
            f"{new_start.strftime('%Y-%m-%d')} → {new_end.strftime('%Y-%m-%d')}",
        )

        n = len(pending)
        button_label = (
            f"Add {n} file{'s' if n != 1 else ''} to {well_name} gauge"
            if gauge is not None
            else f"Apply {n} file{'s' if n != 1 else ''} to {well_name}"
        )
        if st.button(
            button_label,
            type="primary",
            key=f"mg_apply_btn_{well_name}",
            use_container_width=True,
        ):
            # Combine every pending file into the well's gauge (creates one if
            # none exists). The resulting MemoryGaugeData carries the union
            # daily-median series across ALL files; new_gauge after the loop
            # is the final combined gauge. Clear the stash right after the
            # store so an interrupt mid-fetch below can't re-apply the files.
            new_gauge = None
            for pf in pending:
                new_gauge = add_file_to_gauge(well_name, pf)
            set_pending_files(well_name, [])

            # Network fetches keyed off the COMBINED window — done ONCE after
            # all files are added, so extended tests + the divergence check
            # cover the full coverage span, not each file's window.
            with st.spinner(
                f"Fetching well tests + Databricks BHP "
                f"{new_gauge.start_date.strftime('%Y-%m-%d')} → today…"
            ):
                ext_df = fetch_extended_tests(well_name, new_gauge.start_date)
                db_bhp_df = fetch_databricks_bhp_daily(
                    well_name,
                    new_gauge.start_date,
                    new_gauge.end_date,
                )

            if ext_df is not None and not ext_df.empty:
                store_extended_tests(well_name, ext_df)
            else:
                st.session_state["_mg_apply_warning"] = (
                    f"Could not fetch extended tests for {well_name} "
                    "(Databricks query returned nothing or failed). The "
                    "gauge is still active, but only tests in the shared "
                    "3-month cache will pick up gauge BHPs."
                )

            # Auto-divergence vs Databricks across the combined window. Auto-set
            # ON only; once on, the user owns it.
            delta = compute_databricks_vs_gauge_delta(db_bhp_df, new_gauge)
            if delta is not None and delta["divergent"]:
                set_disregard_databricks_bhp(well_name, True)
                # Pop the widget key (writing it directly raises a
                # StreamlitAPIException since the checkbox already
                # rendered this run) — see CLAUDE.md.
                st.session_state.pop(f"mg_disregard_cb_{well_name}", None)
                st.session_state["_mg_auto_disregard_msg"] = (
                    f"Auto-disabled Databricks BHP for {well_name} — "
                    f"the Databricks feed differs from your gauge by an "
                    f"average of **{delta['mean_abs_delta']:.0f} psi "
                    f"({delta['mean_pct_delta']:.0f}%)** over "
                    f"{delta['n_overlap']} overlapping days. "
                    f"Gauge mean: {delta['gauge_mean']:.0f} psi · "
                    f"Databricks mean: {delta['databricks_mean']:.0f} psi. "
                    f"Uncheck *Disregard Databricks BHP* below to override."
                )

            # No uploader reset needed here — the absorb step already bumped
            # the key counter when the files moved into the stash.
            st.session_state["_force_ipr_refresh"] = True
            st.rerun()


def _render_wc_washout_section(
    params: SimulationParams,
    *,
    actuals: dict,
    modeled_psu: float,
    modeled_oil: float,
    selected_test_row,
    effective_jetpump,
    wellbore,
    well_profile,
    inflow,
) -> None:
    """Near-zero-WC washout warning + WC suggestion sweep (the MPE-19 lesson).

    On a low-rate well the PF return swamps the test separator, so the rate
    allocation nets formation water to ~0 and the test's 0% WC is fiction —
    modeling with it leaves BHP and oil unmatchable. When that setup is
    detected (near-zero sidebar WC + PF-dominated stream + poor match), warn
    the engineer to consider raising WC, and offer a sweep that re-runs the
    solver across WC values to suggest a starting point (exactly what turning
    the sidebar Water Cut knob does — diagnostic; nothing applies without the
    explicit Apply click). Detection + sweep logic is pure in
    ``woffl/gui/wc_washout.py``.
    """
    from woffl.gui.sidebar import clamp_seed
    from woffl.gui.utils import create_reservoir_mix
    from woffl.gui.wc_washout import (
        BHP_MISMATCH_PSI,
        OIL_MISMATCH_FRAC,
        detect_wc_washout,
        suggest_water_cut,
    )

    if params.selected_well == "Custom":
        return

    # Produced fluid = the test's reported total fluid (oil + net water);
    # falls back to oil (on a washed-out test they're the same number).
    produced = None
    if selected_test_row is not None:
        tot = selected_test_row.get("WtTotalFluid")
        if is_valid_number(tot):
            produced = float(tot)
    if produced is None:
        produced = actuals.get("oil")

    flag = detect_wc_washout(
        form_wc=float(params.form_wc),
        pf_rate=actuals.get("pf"),
        produced_fluid=produced,
        modeled_psu=modeled_psu,
        actual_bhp=actuals.get("bhp"),
        modeled_oil=modeled_oil,
        actual_oil=actuals.get("oil"),
    )
    result_key = f"_wc_suggest_{params.selected_well}"
    if flag is None:
        # Setup no longer fits (e.g. the engineer already raised WC) — drop
        # any stale suggestion so it can't be applied out of context.
        st.session_state.pop(result_key, None)
        return

    mismatch_bits = []
    if flag.bhp_delta is not None and abs(flag.bhp_delta) >= BHP_MISMATCH_PSI:
        mismatch_bits.append(f"BHP is off by {flag.bhp_delta:+,.0f} psi")
    if (
        flag.oil_delta_frac is not None
        and abs(flag.oil_delta_frac) >= OIL_MISMATCH_FRAC
    ):
        mismatch_bits.append(f"oil is off by {flag.oil_delta_frac:+.0%}")
    st.warning(
        f"**Water cut reads {flag.form_wc:.0%} — don't trust it on this "
        f"well.** Produced fluid is only **{flag.fluid_to_pf_ratio:.0%}** of "
        f"the {flag.pf_rate:,.0f} BWPD power-fluid return, so the test "
        "allocation can wash formation water out to zero (low-rate wells: "
        "formation water gets lost in the PF stream). The model mismatch "
        f"({' and '.join(mismatch_bits)}) is consistent with a much higher "
        "real WC — consider raising the sidebar **Water Cut**, or run the "
        "sweep below for a suggested starting point.",
        icon="💧",
    )

    if st.button(
        "🔍 Suggest a water cut",
        key=f"wc_suggest_btn_{params.selected_well}",
        help=(
            "Re-runs the solver across WC values (5–90%) holding everything "
            "else at current sidebar settings, and picks the WC that best "
            "matches the comparison test's BHP and oil rate. Diagnostic — "
            "nothing changes unless you click Apply on the result."
        ),
    ):
        def _solve_at_wc(wc: float):
            rm = create_reservoir_mix(
                wc,
                params.form_gor,
                params.form_temp,
                params.field_model,
                oil_api=params.oil_api,
                gas_sg=params.gas_sg,
                wat_sg=params.wat_sg,
                bubble_point=params.bubble_point,
            )
            try:
                return run_jetpump_solver(
                    params.surf_pres,
                    params.form_temp,
                    params.rho_pf,
                    params.ppf_surf,
                    effective_jetpump,
                    wellbore,
                    well_profile,
                    inflow,
                    rm,
                    field_model=params.field_model,
                    jpump_direction=params.jpump_direction,
                    quiet=True,
                )
            except Exception:
                # Sweep points that fail (ThroatEntryNoSolution etc.) are
                # skipped, never fatal — the GOR auto-recovery must not fire
                # from a probe loop.
                return None

        with st.spinner("Sweeping water cut 5–90% through the solver…"):
            suggestion = suggest_water_cut(
                _solve_at_wc,
                target_oil=actuals.get("oil"),
                target_bhp=actuals.get("bhp"),
                base_wc=float(params.form_wc),
            )
        if suggestion is None:
            st.session_state.pop(result_key, None)
            st.warning(
                "The sweep couldn't converge enough WC points to make a "
                "suggestion — verify PF pressure and pump identity first."
            )
        else:
            st.session_state[result_key] = suggestion

    suggestion = st.session_state.get(result_key)
    if suggestion is None:
        return

    s1, s2, s3 = st.columns(3)
    s1.metric("Suggested Water Cut", f"{suggestion.suggested_wc:.0%}")
    s2.metric(
        "BHP at suggestion",
        f"{suggestion.modeled_psu:,.0f} psig",
        delta=(
            f"{suggestion.modeled_psu - suggestion.target_bhp:+,.0f} vs actual"
            if suggestion.target_bhp is not None
            else None
        ),
    )
    s3.metric(
        "Oil at suggestion",
        f"{suggestion.modeled_oil:,.0f} BOPD",
        delta=f"{suggestion.modeled_oil - suggestion.target_oil:+,.0f} vs actual",
    )
    caveats = [
        f"Matched on **{suggestion.matched_on}** across {suggestion.n_solved} "
        f"solved WC points ({suggestion.n_failed} failed)."
    ]
    if suggestion.matched_on == "oil only":
        caveats.append(
            "No measured BHP on the comparison test — an oil-only match is "
            "weak evidence; treat the suggestion as a rough starting point."
        )
    if suggestion.bounded:
        caveats.append(
            "The best match sits at the sweep boundary — the true WC may be "
            "outside 5–90%, or something else (PF pressure, pump identity) "
            "is off."
        )
    caveats.append(
        "A starting point, not an answer — sanity-check against offset wells "
        "and the pad's water cut."
    )
    st.caption(" ".join(caveats))

    if st.button(
        f"Apply WC {suggestion.suggested_wc:.2f} to sidebar",
        type="primary",
        key=f"wc_apply_btn_{params.selected_well}",
    ):
        # Logical-key + pop-widget-key pattern (see CLAUDE.md): the sidebar
        # widget already rendered this run, so write the logical key and let
        # _number_input re-seed the widget on the next run.
        st.session_state["form_wc"] = clamp_seed(
            "form_wc", round(float(suggestion.suggested_wc), 2)
        )
        st.session_state.pop("form_wc_input", None)
        st.session_state.pop(result_key, None)
        st.rerun()


def _render_pump_identity_banner(
    params: SimulationParams,
    *,
    effective_pump: tuple[str, str] | None = None,
    selected_test_date=None,
    test_pump: tuple[str, str] | None = None,
    pump_differs: bool = False,
) -> None:
    """Show which pump the hero strip is modeling and whether it matches the test.

    The hero models the **sidebar** pump (``effective_pump``). ``test_pump`` is
    the pump that was in the well at the selected test's date. When they differ,
    the test's actuals were measured on a different pump, so the hero's
    vs-actual deltas aren't a like-for-like comparison \u2014 this banner says so and
    the caller greys the deltas. When they match (or there's no test pump to
    compare against), the deltas are a valid comparison.
    """
    import pandas as pd

    if effective_pump is not None:
        model_n, model_t = effective_pump
    else:
        model_n, model_t = params.nozzle_no, params.area_ratio

    test_str = None
    if selected_test_date is not None and pd.notna(selected_test_date):
        test_str = selected_test_date.strftime("%Y-%m-%d")

    if params.selected_well == "Custom":
        st.info(f"Modeling **{model_n}{model_t}**.")
        return

    # Modeled pump differs from the pump in the well at the test's date \u2014 the
    # vs-actual deltas would compare different pumps, so they're not valid.
    if pump_differs and test_pump is not None:
        tn, tt = test_pump
        date_part = f" {test_str}" if test_str else ""
        st.warning(
            f"**Not a pump match.** Modeling **{model_n}{model_t}**, but the well "
            f"had **{tn}{tt}** at the{date_part} test you're comparing to \u2014 so the "
            "vs-actual deltas are **greyed out / paused** (a different pump against "
            f"that test isn't apples-to-apples). Set the sidebar pump to **{tn}{tt}** "
            "to compare against this test, or read the Model-vs-Actual section "
            f"below (it models {tn}{tt} for the proper comparison).",
            icon="\u26a0\ufe0f",
        )
        return

    # Modeled pump matches the test's pump \u2192 the deltas are like-for-like.
    if test_pump is not None and test_str:
        st.success(
            f"**Pump match.** Modeling **{model_n}{model_t}**, the same pump the "
            f"well had at the **{test_str}** test \u2014 so the vs-actual deltas below "
            "are a like-for-like comparison."
        )
        return

    # No test pump to compare against (no test selected, or no JP install record
    # at the test's date) \u2014 just report what's being modeled.
    st.info(f"Modeling **{model_n}{model_t}** (sidebar pump).")


def _is_valid_pump_code(nozzle, throat) -> bool:
    """True when (nozzle, throat) are recognized National pump codes.

    Guards against corrupt JP-history records — e.g. S-17 has a throat letter
    ('D') in the Nozzle Number column. Such a record must NOT be treated as a
    real "test pump" (it would blank the model-vs-actual deltas and crash
    ``JetPump('D', ...)``); callers fall back to the sidebar pump instead.
    """
    from woffl.gui.params import NOZZLE_OPTIONS, THROAT_OPTIONS

    return str(nozzle) in NOZZLE_OPTIONS and str(throat) in THROAT_OPTIONS


def _render_jp_history_strip(well_name: str) -> None:
    """The JP History tab's production/BHP/JPCO chart, pinned to the top of
    the Solver (same figure, reduced height), with a slim pump-install
    timeline bar underneath.

    Hidden behind a persistent toggle — the choice lives in a two-tier key
    so it survives the view switcher's widget-state GC. The Databricks
    fetch only fires while the section is shown.
    """
    import pandas as pd

    jp_hist = st.session_state.get("jp_history_df")
    if jp_hist is None or well_name == "Custom":
        return

    well_df = (
        jp_hist[jp_hist["Well Name"] == well_name]
        .dropna(subset=["Date Set"])
        .sort_values("Date Set")
        .reset_index(drop=True)
    )
    # No JP installs on record (e.g. S-67) is fine — we still show the test/BHP
    # time-series trend below, just without JP-change lines or an install bar.
    no_history = well_df.empty

    if "sw_jp_strip_input" not in st.session_state:
        st.session_state["sw_jp_strip_input"] = bool(
            st.session_state.get("sw_jp_strip", True)
        )
    show = st.toggle(
        "Pump history",
        key="sw_jp_strip_input",
        help=(
            "Production + BHP with jet-pump change lines (same chart as the "
            "JP History tab) and the install timeline."
        ),
    )
    st.session_state["sw_jp_strip"] = show
    if not show:
        return

    # SHARED figure builder with the JP History tab (jp_history_tab.
    # build_history_with_strip_figure) so the two charts are identical by
    # construction. The per-well queries are cached 24 h and pre-warmed by
    # the JP-history prefetch thread, so this is usually instant.
    from woffl.gui.tabs.jp_history_tab import (
        _fetch_extended_well_tests,
        build_history_with_strip_figure,
        render_current_pump_caption,
    )

    # No JP installs on record (S-67): the well may still have tests, so show the
    # oil/BHP time-series alone (empty JP-changes -> no install lines/timeline).
    if no_history:
        fallback_start = pd.Timestamp.today().normalize() - pd.DateOffset(years=5)
        t_df, bhp_d, bhp_o = _fetch_extended_well_tests(well_name, fallback_start)
        if t_df is None or t_df.empty:
            st.caption(f"No JP-install history or tests on record for {well_name}.")
            return
        fig, _tl = build_history_with_strip_figure(
            well_name, well_df, t_df, bhp_d, bhp_o
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "No JP-install history on record — showing the test / BHP trend only."
        )
        return

    earliest_date = well_df["Date Set"].min()
    test_df, bhp_daily_df, bhp_overlay_df = _fetch_extended_well_tests(
        well_name, earliest_date
    )

    fig, tl = build_history_with_strip_figure(
        well_name,
        well_df,
        test_df,
        bhp_daily_df,
        bhp_overlay_df,
        bhp_from_zero=True,
        height=470,
    )
    st.plotly_chart(fig, use_container_width=True, key="sw_jp_strip_chart")
    render_current_pump_caption(tl)


def render_tab(
    params: SimulationParams,
    jetpump,
    wellbore,
    well_profile,
    inflow,
    res_mix,
    *,
    hero_container=None,
    jp_strip_container=None,
    anchor_container=None,
) -> None:
    """Render the Jetpump Solver Results tab.

    Args:
        params: Simulation parameters from sidebar
        jetpump: JetPump object
        wellbore: PipeInPipe wellbore object
        well_profile: WellProfile object
        inflow: InFlow object
        res_mix: ResMix object
        hero_container: Optional Streamlit container into which the hero strip
            (pump-identity banner, the gauge caveat, and the vs-actual Oil /
            Water / Power-Fluid / Suction metrics) is rendered. The pad-review
            page passes a container placed right under the "Save review — WELL"
            title so the difference shows at the top. When ``None`` (the
            Single-Well Solver) the hero renders inline here, unchanged.
        jp_strip_container: Optional container into which the pump-history strip
            (production / BHP / JPCO chart + pump-install timeline) is rendered.
            The pad-review page passes a container placed ABOVE
            ``hero_container`` (both under the title) so the history plot sits
            right above the hero. When ``None`` it renders inline at the top of
            the Solver, unchanged.
        anchor_container: Optional container into which the IPR-anchor selector
            (+ the comparison-test decouple checkbox) is rendered. The
            pad-review page passes a container under the "Save review" title —
            without it the anchor control sat below the whole save panel under
            the "Solver" heading, where engineers never found it. Execution
            order is unchanged (the control still runs, and seeds the sidebar,
            before the solve); only where it DISPLAYS moves. ``None`` = inline
            at the top of the Solver, unchanged.
    """
    import pandas as pd

    render_input_summary(params)

    # Glanceable pump-install history (toggle to hide). A PLACEHOLDER is
    # reserved here and filled at the END of the tab render — Streamlit
    # paints elements progressively, so this keeps the strip's Databricks
    # fetch (cold first view of a well) from blocking everything below it;
    # the chart simply pops into place when ready. The pad-review page passes
    # its own container (placed above the hero, under the title) so the history
    # plot sits right above the hero; otherwise it renders inline here.
    _jp_strip_box = (
        jp_strip_container if jp_strip_container is not None else st.container()
    )

    # Memory-gauge upload + status. Lives near the top so the status banner
    # is unmissable when an override is active; the upload itself is in a
    # collapsed expander so it stays out of the way when not needed.
    _render_memory_gauge_section(params.selected_well)

    # Surface any persisted message from a prior auto-recovery (e.g. GOR reset)
    msg = st.session_state.pop("_solver_gor_reset_msg", None)
    if msg:
        st.warning(msg)

    # Surface the one-shot confirmation from a prior "Match test oil rate"
    # back-solve (it seeds the sidebar then reruns, so the note has to be
    # replayed on the following render).
    _bm_msg = st.session_state.pop("_oil_backmatch_msg", None)
    if _bm_msg:
        st.success(_bm_msg, icon="🎯")

    # Same one-shot replay for the joint oil + PF auto-match (it seeds several
    # sidebar fields then reruns). dict carries text + status so we can color it.
    _jm = st.session_state.pop("_joint_automatch_msg", None)
    if _jm:
        _icon = "🎯" if _jm.get("ok") else "⚠️"
        (st.success if _jm.get("ok") else st.warning)(_jm["text"], icon=_icon)

    # The IPR group renders as one unit — provisional-test entry, IPR-anchor
    # selector, comparison-test picker, and the IPR chart — so the control that
    # pins the IPR sits right next to the curve it moves (Scott: "the IPR test
    # selector needs to be right above the IPR graph, just below provisional
    # test"). The SEED still runs here at the TOP of the tab, before the solve,
    # so choosing the *driving test* reseeds the sidebar (qwf/pwf/ResP/WC/GOR)
    # and the solver re-runs against it — a watered-out most-recent test can't
    # dead-end the solve. Only the CHART is drawn later (in Model-vs-Actual, once
    # the fit is computed) into this same ``_anchor_box``, so it lands with the
    # group rather than far below. On the Single-Well page ``_anchor_box`` is a
    # container created here (group at the top of the tab); the pad-review page
    # passes its own (group under the "Save review" title).
    test_df = _get_well_tests(params.selected_well)

    # Same gate Model-vs-Actual uses for its anchor control: oil workflow, JP
    # history loaded, non-Custom, 2+ tests. Water mode is the explicit
    # watered-out fallback and has no oil anchor to seed; Custom has no tests.
    n_tests_now = 0 if test_df is None or test_df.empty else len(test_df)
    anchor_selector_shown = (
        st.session_state.get("jp_history_df") is not None
        and params.selected_well != "Custom"
        and not params.model_as_water
        and n_tests_now >= 2
    )
    _anchor_box = anchor_container if anchor_container is not None else st.container()
    with _anchor_box:
        # Provisional-test entry sits at the top of the group — "just below" it
        # comes the anchor selector. Always available (a well with no Databricks
        # tests can still be modeled once a manual test is added); gated only on
        # a real well. Moved up from Model-vs-Actual so the group reads
        # provisional → selector → chart.
        if params.selected_well != "Custom":
            _render_manual_test_entry(params.selected_well)
        if anchor_selector_shown:
            anchor_fit = _render_ipr_anchor_and_seed(params, test_df)
        else:
            anchor_fit = (None, None, None)

    # The comparison-test picker is slaved to the IPR anchor by default; the
    # "Use a different test for comparison" checkbox (rendered by the anchor
    # selector just above) frees it. That checkbox renders earlier in THIS run,
    # so its value is already committed when we read it here. Default = most
    # recent test, matching the pre-picker behaviour. Rendered into the same
    # box as the anchor selector — they're one control group.
    _decoupled = st.session_state.get(f"sw_ipr_decouple_{params.selected_well}", False)
    with _anchor_box:
        selected_test_row = _render_test_picker(
            params.selected_well, test_df, synced=not _decoupled
        )

    # The hero strip models the SIDEBAR pump — the pump the user picked to model.
    # We still look up the pump that was in the well at the selected test's date
    # so we can flag when the modeled pump differs from it: the test's actuals
    # were measured on that pump, so comparing a different pump's model to them
    # isn't valid (the vs-actual deltas get greyed out below). The
    # Model-vs-Actual section further down still models the test's own pump for
    # the proper historical comparison.
    jp_hist_for_pump = st.session_state.get("jp_history_df")
    test_date_for_pump = None
    test_pump = None  # (nozzle, throat) installed at the selected test's date
    effective_nozzle = params.nozzle_no
    effective_throat = params.area_ratio

    if selected_test_row is not None and jp_hist_for_pump is not None:
        td_raw = selected_test_row.get("WtDate")
        if pd.notna(td_raw):
            test_date_for_pump = td_raw
            pump_at_test = _pump_at_test_date(
                jp_hist_for_pump, params.selected_well, td_raw
            )
            if pump_at_test and _is_valid_pump_code(
                pump_at_test.get("nozzle_no"), pump_at_test.get("throat_ratio")
            ):
                test_pump = (
                    pump_at_test["nozzle_no"],
                    pump_at_test["throat_ratio"],
                )

    pump_differs = (
        test_pump is not None and (effective_nozzle, effective_throat) != test_pump
    )

    # The hero strip (pump-identity banner, gauge caveat, and the vs-actual
    # metrics) can be hoisted into a caller-supplied container so it renders at
    # the TOP of the page — the pad review places it right under the "Save
    # review — WELL" title so the engineer sees the difference first. With no
    # container (the Single-Well Solver) it renders inline here, as before.
    _hero = hero_container if hero_container is not None else st.container()

    with _hero:
        _render_pump_identity_banner(
            params,
            effective_pump=(effective_nozzle, effective_throat),
            selected_test_date=test_date_for_pump,
            test_pump=test_pump,
            pump_differs=pump_differs,
        )

    # The hero models the sidebar pump (effective == sidebar), so reuse the
    # JetPump already built from the sidebar inputs.
    effective_jetpump = jetpump

    # Clear stale calibration if well changed
    _cal = st.session_state.get("sw_calibration_result")
    if _cal and _cal.well_name != params.selected_well:
        st.session_state.pop("sw_calibration_result", None)

    # 100% water cut => zero oil volume fraction. The solver is oil-anchored
    # (formation water is carried as a multiple of the oil rate), so a no-oil
    # mixture is undefined and the library raises a ValueError. UNLESS water-pump
    # mode is on (model_as_water), in which case the solver is water-anchored and
    # we run it. Otherwise pre-empt the generic "Solver error: ..." box with a
    # clear, specific message and skip the solve.
    if params.form_wc >= 1.0 and not params.model_as_water:
        if anchor_selector_shown:
            _water_fix = (
                "Anchor the IPR on an earlier **oil-bearing** test using the "
                "**IPR anchor** selector (above the IPR chart — set it to "
                "*Specific test* and pick a test with oil) — the sidebar reseeds "
                "and the solver re-runs against it. Or lower the **Water Cut** "
                "below 100% in the sidebar."
            )
        else:
            _water_fix = (
                "Lower the **Water Cut** below 100% in the sidebar to model "
                "this well."
            )
        st.warning(
            "**100% water cut** — the selected test/input has no oil, and the "
            "jet pump **oil** model can't solve a zero-oil mixture. "
            f"{_water_fix} _(For a genuinely watered-out well with no oil test "
            'to fall back to, tick "Model as 100% water" in the sidebar.)_'
        )
        solver_results = None
    else:
        _spin = (
            "Running dewatering solve..."
            if params.model_as_water
            else "Running jetpump solver..."
        )
        with st.spinner(_spin):
            try:
                solver_results = run_jetpump_solver(
                    params.surf_pres,
                    params.form_temp,
                    params.rho_pf,
                    params.ppf_surf,
                    effective_jetpump,
                    wellbore,
                    well_profile,
                    inflow,
                    res_mix,
                    field_model=params.field_model,
                    jpump_direction=params.jpump_direction,
                )
            except IndexError:
                if params.model_as_water:
                    # Water mode has no GOR to recover (gas is zero at 100%
                    # water); surface a clear failure instead of GOR-resetting.
                    solver_results = None
                    st.error(
                        "Dewatering solve found no throat-entry solution — try a "
                        "higher power-fluid pressure or a different throat size."
                    )
                else:
                    # Throat-entry iteration produced no valid points — typically
                    # caused by an unrealistically low GOR for the well's PVT.
                    # Reset the sidebar GOR to GOR_AUTO_RECOVERY_VALUE (250) and
                    # remember a per-well GOR floor, so the re-solve — and every
                    # view, which all read the sidebar GOR now — uses the
                    # recovered value.
                    #
                    # Streamlit forbids writing to a widget's state key after the
                    # widget has rendered (and the sidebar already rendered above
                    # the tabs). So we set the logical key and DELETE the widget
                    # key — the _number_input helper will re-initialize the widget
                    # from the logical key on the next run.
                    _trigger_gor_reset(
                        params.selected_well,
                        params.form_gor,
                        reason="throat-entry iteration produced no valid points",
                    )

    # Water-pump (dewatering) mode renders its own compact result block — the
    # oil hero below (test-vs-actual deltas, calibration, pump identity) doesn't
    # apply to a no-oil well. Show what it takes to flow/dewater, then stop.
    if params.model_as_water:
        if solver_results:
            psu, sonic_status, qoil_std, fwat_bwpd, qnz_bwpd, mach_te = solver_results
            water_rate = fwat_bwpd  # formation water lifted (oil = 0 here)
            st.success(
                f"**Dewatering solve** — {params.selected_well} modeled as 100% "
                "water (no oil)."
            )
            w1, w2, w3, w4 = st.columns(4)
            w1.metric(
                "Suction Pressure",
                f"{psu:,.0f} psig",
                help="Bottomhole pressure the pump pulls — the dewatering drawdown.",
            )
            w2.metric(
                "Water Rate",
                f"{water_rate:,.0f} BWPD",
                help="Formation water lifted at the solved suction (from the IPR).",
            )
            w3.metric(
                "Power Fluid",
                f"{qnz_bwpd:,.0f} BWPD",
                help="Power-fluid (water) rate required to drive the pump.",
            )
            w4.metric(
                "PF Surface Pressure",
                f"{params.ppf_surf:,.0f} psig",
                help="Power-fluid pressure supplied at surface (sidebar).",
            )
            _sonic = "Sonic (choked)" if sonic_status else "Subsonic"
            st.caption(
                f"Throat: **{_sonic}** (Mach {mach_te:.2f}). Total water handled "
                f"≈ {water_rate + qnz_bwpd:,.0f} BWPD (formation + power fluid). "
                "Water is near-incompressible, so a water pump typically won't "
                "choke the way a gassy oil well does."
            )
        return

    if solver_results:
        psu, sonic_status, qoil_std, fwat_bwpd, qnz_bwpd, mach_te = solver_results

        # Hero strip — the four numbers a user actually came here to see.
        # Each modeled value is shown alongside its delta vs the SELECTED test
        # (the test picker above; defaults to most recent). The deltas double
        # as a visual nudge that these are the values the friction-coef
        # calibration in Model vs Actual can pull toward zero.
        actuals = _actuals_from_test(selected_test_row)

        # When the well has tests but no BHP-bearing test in the cache, the
        # sidebar's qwf/pwf/res_pres never got auto-populated from a Vogel
        # fit — so the modeled values below are driven by whatever defaults
        # are sitting in the sidebar, NOT calibrated to this well. Surface
        # that prominently so the user doesn't read "Suction Pressure"
        # as the well's actual operating BHP. When the oil delta is also
        # large, flag the specific value the engineer should tune toward.
        has_any_actual = any(v is not None for v in actuals.values())
        if (
            has_any_actual
            and actuals.get("bhp") is None
            and params.selected_well != "Custom"
        ):
            actual_oil = actuals.get("oil")
            oil_msg = ""
            if actual_oil is not None and actual_oil > 0:
                pct = (qoil_std - actual_oil) / actual_oil * 100
                if abs(pct) > 25:
                    direction = "higher" if pct > 0 else "lower"
                    oil_msg = (
                        f" **Modeled oil rate ({qoil_std:,.0f} BOPD) is "
                        f"{abs(pct):.0f}% {direction} than the latest test "
                        f"({actual_oil:,.0f} BOPD)** — tune sidebar Form WC, "
                        "Reservoir Pressure, or flowing BHP (pwf) until "
                        "the modeled oil rate aligns. That's the calibration "
                        "path for wells without a gauge."
                    )
            with _hero:
                st.warning(
                    f"**No BHP gauge data for {params.selected_well} in the "
                    "test-window cache.** Modeled values below reflect the "
                    "sidebar's reservoir/IPR inputs, not a Vogel fit to this "
                    "well. Treat the hero metrics as a sidebar-driven "
                    "what-if; the suction-pressure number in particular is "
                    "not an estimate of the well's actual BHP." + oil_msg
                )

        def _delta(modeled: float, actual: float | None, suffix: str) -> str | None:
            if actual is None:
                return None
            if pump_differs:
                # Modeled pump ≠ the test's pump, so a numeric delta would be
                # meaningless — blank the number but keep the "vs actual"
                # context (and the caller greys it).
                return suffix
            return f"{modeled - actual:+,.0f} {suffix}"

        def _label(base: str, actual) -> str:
            """Append '(modeled)' to a hero-strip label when no actual exists,
            so the user doesn't mistake a sidebar-driven prediction for a
            measured value."""
            return base if actual is not None else f"{base} (modeled)"

        # When the modeled (sidebar) pump differs from the pump the test was run
        # on, the vs-actual deltas compare different pumps — grey them out (the
        # banner above explains why). Streamlit's delta_color "off" renders the
        # delta in grey rather than red/green.
        _dcolor = "off" if pump_differs else "normal"

        with _hero:
            h1, h2, h3, h4 = st.columns(4)
            with h1:
                d = _delta(qoil_std, actuals["oil"], "vs actual")
                st.metric(
                    _label("Oil Rate", actuals["oil"]),
                    f"{qoil_std:,.0f} BOPD",
                    delta=d,
                    delta_color="off" if d is None else _dcolor,
                )
            with h2:
                # Formation Water has no actuals counterpart (we don't track it
                # in actuals dict), so it's always modeled — label accordingly.
                st.metric("Formation Water (modeled)", f"{fwat_bwpd:,.0f} BWPD")
            with h3:
                d = _delta(qnz_bwpd, actuals["pf"], "vs actual")
                st.metric(
                    _label("Power Fluid", actuals["pf"]),
                    f"{qnz_bwpd:,.0f} BWPD",
                    delta=d,
                    delta_color="off" if d is None else _dcolor,
                )
            with h4:
                d = _delta(psu, actuals["bhp"], "vs actual")
                st.metric(
                    _label("Suction Pressure", actuals["bhp"]),
                    f"{psu:,.0f} psig",
                    delta=d,
                    delta_color="off" if d is None else _dcolor,
                )

            # Footer the hero with the exact test the deltas are measured against
            # (date), so the difference is always attributable to a specific test
            # — and restate the pump-match status crisply next to the numbers
            # (the banner above carries the full explanation).
            _cmp_date = None
            if selected_test_row is not None:
                _wt = selected_test_row.get("WtDate")
                if pd.notna(_wt):
                    try:
                        _cmp_date = pd.Timestamp(_wt).strftime("%Y-%m-%d")
                    except (TypeError, ValueError):
                        _cmp_date = None
            if has_any_actual:
                _cmp = (
                    f"the **{_cmp_date}** well test"
                    if _cmp_date
                    else "the most recent well test"
                )
                if pump_differs:
                    st.caption(
                        f"Δ vs {_cmp} — **paused (not a pump match)**; see the note above."
                    )
                else:
                    st.caption(f"Δ vs {_cmp}.")

        # When the modeled (sidebar) pump differs from the test's pump, the
        # vs-actual checks below (PF-rate match, BHP calibration) would compare
        # different pumps — pause them and nudge the user back to a like-for-like
        # setup. (The banner above already explains; this is the actionable hint
        # under the metrics.)
        if pump_differs:
            tn, tt = test_pump
            st.caption(
                f"What-if mode — modeling **{effective_nozzle}{effective_throat}**, "
                f"not the test's pump (**{tn}{tt}**). PF-match and BHP "
                f"calibration are paused; set the sidebar pump to {tn}{tt} to "
                "compare against and calibrate to this test."
            )
        else:
            # PF check. With a MEASURED test-day PF pressure (vw_pressure_daily
            # join) a rate mismatch is a wear/identity diagnostic and never
            # gates calibration; only tests with no daily reading keep the old
            # "fix pressure first" gate. Anchor on the SELECTED comparison test
            # (not most-recent) so the displayed date matches the lift rate the
            # PF check / quickfix compares against.
            cal_inputs = build_calibration_inputs(
                params, wellbore, well_profile, selected_test_row=selected_test_row
            )
            test_date_str = cal_inputs["test_date_str"] if cal_inputs else None
            pf_warning_shown, pf_blocked = render_pf_mismatch_warning(
                qnz_bwpd,
                actuals["pf"],
                params.ppf_surf,
                test_date_str=test_date_str,
                well_name=params.selected_well,
                measured_pf=cal_inputs.get("test_pf_press") if cal_inputs else None,
            )
            # Render the quickfix whenever any warning fires (red OR yellow info)
            # so the user always has a one-click path to fine-tune. Calibration
            # gating (cal button disabled) is governed by `pf_blocked` only.
            if pf_warning_shown:
                render_pf_quickfix_widget(
                    params,
                    wellbore,
                    well_profile,
                    target_lift_wat=actuals["pf"],
                    selected_test_row=selected_test_row,
                )

            # BHP red flag (only meaningful once PF is right — otherwise the BHP
            # delta is mostly explained by the wrong PF, not friction).
            if not pf_blocked:
                # The hero footer (top, by the metrics) already states the
                # comparison test + date; just surface the BHP-calibration flag
                # here when it applies.
                render_bhp_calibration_warning(psu, actuals["bhp"], on_solver_view=True)

            # Compact calibration action bar — buttons live here so they're
            # visible without scrolling. Run is disabled while PF mismatch is
            # blocking; the Push button stays available so a prior result can
            # still be applied. Also disabled when the SELECTED test (from the
            # picker) has no measured BHP — there's nothing to target. Both
            # ``selected_test_row`` and ``bhp_missing`` reflect the picker so
            # the gating + calibration target stay in sync.
            _render_fric_cal_action_bar(
                params,
                wellbore,
                well_profile,
                selected_test_row=selected_test_row,
                pf_blocked=pf_blocked,
                bhp_missing=(actuals["bhp"] is None),
            )

            # Near-zero-WC washout check (MPE-19 lesson): on a low-rate well
            # the PF return swamps the separator and the allocation nets
            # formation water to ~0 — warn + offer a WC-sweep suggestion.
            _render_wc_washout_section(
                params,
                actuals=actuals,
                modeled_psu=psu,
                modeled_oil=qoil_std,
                selected_test_row=selected_test_row,
                effective_jetpump=effective_jetpump,
                wellbore=wellbore,
                well_profile=well_profile,
                inflow=inflow,
            )

        # Secondary diagnostics
        with st.expander("Throat diagnostics", expanded=False):
            d1, d2 = st.columns(2)
            with d1:
                st.metric("Throat Entry Mach", f"{mach_te:.3f}")
            with d2:
                st.metric("Sonic Flow", "Yes" if sonic_status else "No")
            if sonic_status:
                st.info("Critical flow conditions — sonic velocity at throat entry.")
            else:
                st.caption("Stable subsonic flow at the throat.")

        # Rate-scalar applied banner \u2014 full toggle + calibrated predictions
        # live next to the Calibration Factor metric in Model vs Actual below.
        _cal = st.session_state.get("sw_calibration_result")
        if (
            _cal
            and params.selected_well != "Custom"
            and st.session_state.get("sw_apply_calibration_persist", False)
        ):
            st.caption(
                f"Rate-scalar calibration **applied** "
                f"(factor {_cal.calibration_factor:.2f}). Calibrated rates "
                f"shown in *Model vs Actual* below."
            )
    else:
        st.warning(
            "The solver could not find a solution with the current parameters. "
            "Try adjusting the input values."
        )

    # Oil-rate back-match — rendered REGARDLESS of whether the current solve
    # converged. It's most useful precisely when it DIDN'T: for a gaugeless well
    # it searches for the flowing BHP / IPR that makes the pump reproduce the
    # known test oil rate (a consistent, converging operating point). Self-gates
    # to gaugeless wells with a known test oil rate, so it's a no-op otherwise.
    _render_oil_rate_backmatch(
        params, wellbore, well_profile, selected_test_row=selected_test_row
    )

    # Joint oil + PF auto-match — the headline calibration: find IPR + JP params
    # so the model reproduces BOTH the test oil AND power-fluid rate at once.
    # Self-gates to wells whose selected test has both a known oil and PF rate.
    _render_joint_automatch(
        params, wellbore, well_profile, selected_test_row=selected_test_row
    )

    # Model vs Actual comparison (requires JP history + non-Custom well). The
    # IPR-anchor fit was already computed + seeded at the top of the tab; pass
    # it down so MvA reuses it for its chart instead of recomputing.
    _render_model_vs_actual(
        params,
        wellbore,
        well_profile,
        selected_test_row=selected_test_row,
        anchor_fit=anchor_fit,
        # Draw the IPR chart into the same box as the provisional-test entry +
        # anchor pickers, so the curve sits right next to the control that moves
        # it. On Single-Well ``_anchor_box`` is the top-of-tab group; on pad
        # review it's the box under the "Save review" title. The comparison
        # TABLE + calibration below stay in the Model-vs-Actual section.
        ipr_chart_container=_anchor_box,
    )

    # Deferred fill of the pump-history placeholder reserved at the top —
    # everything above has already painted by the time this fetch runs.
    with _jp_strip_box:
        _render_jp_history_strip(params.selected_well)


def _can_run_fric_cal(params: SimulationParams) -> bool:
    """Cheap pre-flight check — do we have the prerequisites for calibration?

    Mirrors the early-exit logic in _render_model_vs_actual so the top action
    bar only renders when calibration is actually possible. Avoids triggering
    the heavier IPR analysis just to figure out whether to show a button.
    """
    if params.selected_well == "Custom":
        return False
    jp_hist = st.session_state.get("jp_history_df")
    if jp_hist is None:
        return False

    from woffl.assembly.jp_history import get_current_pump

    current_pump = get_current_pump(jp_hist, params.selected_well)
    if current_pump is None:
        return False
    if not current_pump.get("nozzle_no") or not current_pump.get("throat_ratio"):
        return False

    tests = _get_well_tests(params.selected_well)
    if tests is None or len(tests) < 1:
        return False
    # Calibration is possible if ANY test has measured BHP — the test
    # picker decides which specific test to target. Checking only the
    # most-recent test missed memory-gauge wells where the gauge covers
    # older tests but the most-recent test is outside coverage (and so
    # has NaN BHP after gauge merge).
    return any(is_valid_number(v) for v in tests["BHP"])


def _execute_fric_cal(
    params: SimulationParams,
    wellbore,
    well_profile,
    *,
    selected_test_row=None,
) -> tuple[bool, str | None]:
    """Build the calibration inputs and run calibrate_friction_coefs.

    Single source of truth for "how do we calibrate this well" — invoked from
    the top action bar. ``selected_test_row`` (from the Solver tab's test
    picker) is the target test; when omitted, most-recent test is used
    (preserves the pre-picker behaviour for direct callers).

    Result is stashed in session_state["sw_fric_calibration"][well] for the
    existing display code. Returns (success, error_message); the caller is
    responsible for surfacing the error.
    """
    from woffl.gui.utils import build_calibration_inputs

    inputs = build_calibration_inputs(
        params,
        wellbore,
        well_profile,
        selected_test_row=selected_test_row,
    )
    if inputs is None:
        return False, "Cannot calibrate — missing JP history or test data."
    if inputs["actual_bhp"] is None:
        return (
            False,
            "Cannot calibrate — the selected test has no measured BHP. "
            "Pick a different test in the test picker above, or upload "
            "memory-gauge data to fill in this test's BHP.",
        )

    _, water_obj, _ = create_pvt_components(params.field_model)
    prop_pf = water_obj.condition(0, 60)

    try:
        result = calibrate_friction_coefs(
            well_name=params.selected_well,
            target_bhp=inputs["actual_bhp"],
            pwh=inputs["model_surf_pres"],
            tsu=float(params.form_temp),
            ppf_surf=float(params.ppf_surf),
            nozzle=inputs["nozzle"],
            throat=inputs["throat"],
            knz=0.01,
            ken=float(params.ken),
            wellbore=wellbore,
            wellprof=well_profile,
            ipr_su=inputs["ipr_inflow"],
            prop_su=inputs["ipr_res_mix"],
            prop_pf=prop_pf,
            jpump_direction=params.jpump_direction,
        )
    except Exception as e:
        return False, f"Calibration failed: {e}"

    cal_state = st.session_state.setdefault("sw_fric_calibration", {})
    cal_state[params.selected_well] = result

    # Auto-push to sidebar so a successful calibration takes effect on the
    # next rerun without a second click. Widget keys (ken_input/…) are
    # popped — the sidebar's _number_input helper re-initializes them from
    # the logical keys on the next render. Writing widget keys directly
    # would raise after the sidebar already rendered this run.
    st.session_state["ken"] = float(result.best_ken)
    st.session_state["kth"] = float(result.best_kth)
    st.session_state["kdi"] = float(result.best_kdi)
    st.session_state.pop("ken_input", None)
    st.session_state.pop("kth_input", None)
    st.session_state.pop("kdi_input", None)
    st.session_state["_pushed_fric_msg"] = (
        f"Calibrated and applied: ken={result.best_ken:.3f}, "
        f"kth={result.best_kth:.3f}, kdi={result.best_kdi:.3f}"
    )
    return True, None


def _render_fric_cal_action_bar(
    params: SimulationParams,
    wellbore,
    well_profile,
    *,
    selected_test_row=None,
    pf_blocked: bool = False,
    bhp_missing: bool = False,
) -> None:
    """Compact calibration action bar rendered right below the hero strip.

    Single Run button + one-line status. Pulls the user toward calibration
    when the BHP red flag is showing — the button lives where the eyes
    already are. On success, the fitted ken/kth/kdi are automatically
    pushed to the sidebar so the next rerun uses them everywhere
    (Solver / Batch Run / PF Range). Detailed result metrics stay down
    in Model vs Actual; this strip is just the action surface.

    Disabled when:
      - ``pf_blocked`` — calibrating against a wrong PF pressure produces
        useless friction coefs.
      - ``bhp_missing`` — the BHP-match objective has no target. Common
        for S-pad wells whose recent tests lack a coincident gauge.
    """
    if not _can_run_fric_cal(params):
        return

    # Surface success message from the prior render's calibration push.
    pushed_msg = st.session_state.pop("_pushed_fric_msg", None)
    if pushed_msg:
        st.success(pushed_msg)

    cal_state = st.session_state.get("sw_fric_calibration", {})
    result = cal_state.get(params.selected_well)
    has_result = result is not None and getattr(result, "converged", False)

    col_run, col_status = st.columns([1.5, 4.5])

    disabled = pf_blocked or bhp_missing
    if bhp_missing:
        disable_help = (
            "Disabled — the selected test (test picker above) has no "
            "measured BHP. Friction-coef calibration fits ken/kth/kdi to "
            "drive modeled BHP toward measured BHP, so it needs a target. "
            "Pick a test that has a BHP value, or upload memory-gauge data."
        )
    elif pf_blocked:
        disable_help = (
            "Disabled while PF rate mismatch is too large — fix the "
            "sidebar Power Fluid Surface Pressure first."
        )
    else:
        disable_help = (
            "Fits ken/kth/kdi to drive modeled BHP toward measured "
            "BHP, then applies them to the sidebar in one click."
        )

    with col_run:
        run_label = "Re-run BHP Cal" if has_result else "Run BHP Calibration"
        run_clicked = st.button(
            run_label,
            type="primary",
            key="sw_run_fric_cal_top",
            use_container_width=True,
            disabled=disabled,
            help=disable_help,
        )

    with col_status:
        if bhp_missing:
            st.caption(
                "⚠️ No measured BHP for this well — calibration needs a "
                "gauge reading to target."
            )
        elif pf_blocked:
            st.caption("⚠️ Calibration blocked — fix the PF rate mismatch above first.")
        elif has_result:
            quality = getattr(result, "match_quality", "unknown")
            color = {"good": "green", "fair": "orange", "poor": "red"}.get(
                quality, "gray"
            )
            st.markdown(
                f"Last cal — match: :{color}[**{quality.upper()}**] · "
                f"BHP error {result.bhp_error:+.0f} psi · "
                f"applied: ken={result.best_ken:.3f}, "
                f"kth={result.best_kth:.3f}, kdi={result.best_kdi:.3f} · "
                f"see *Model vs Actual* below for details."
            )
        else:
            st.caption(
                "Fits ken/kth/kdi to drive modeled BHP toward measured BHP, "
                "then auto-applies to the sidebar."
            )

    # "What do these coefficients represent?" explainer — sits right below the
    # calibrate button so the user can read what ken/kth/kdi mean before running.
    from woffl.gui.explainers import render_kcoef_explainer

    render_kcoef_explainer()

    if run_clicked:
        with st.spinner("Calibrating (Nelder-Mead)..."):
            ok, err = _execute_fric_cal(
                params,
                wellbore,
                well_profile,
                selected_test_row=selected_test_row,
            )
        if not ok:
            st.error(err)
        else:
            st.rerun()


def _render_oil_rate_backmatch(
    params: SimulationParams,
    wellbore,
    well_profile,
    *,
    selected_test_row=None,
) -> None:
    """Button to infer flowing BHP from a KNOWN test oil rate (gaugeless wells).

    The engineer knows the test's oil rate and the installed pump but has no
    BHP gauge, so they can't anchor the IPR. Hand-tweaking qwf/pwf/ResP to make
    the modeled pump match the test rate is painful and often lands on a
    non-convergent IPR. This button does it numerically: it finds the pwf such
    that the pump installed at the test's date — at the test's conditions —
    reproduces the test oil rate, then seeds the sidebar IPR
    (qwf / pwf / res_pres) with that consistent solution.

    Only shown when the selected test has a known oil rate but no measured BHP.
    Uses :func:`woffl.gui.utils.build_calibration_inputs` to mirror the exact
    solver inputs the friction calibration builds (installed test pump, test
    WHP, ResMix from sidebar WC/GOR), but feeds a TRIAL IPR instead of the
    sidebar one. Reservoir pressure is held at the sidebar value (``params.pres``).
    """
    from woffl.gui.utils import build_calibration_inputs

    if params.selected_well == "Custom":
        return

    actuals = _actuals_from_test(selected_test_row)
    oil_test = actuals.get("oil")
    if oil_test is None or oil_test <= 0:
        return  # nothing to match against
    if actuals.get("bhp") is not None:
        return  # gauge present — the normal BHP calibration path applies

    # Build the test-conditioned solver inputs (installed pump + test WHP).
    inputs = build_calibration_inputs(
        params,
        wellbore,
        well_profile,
        selected_test_row=selected_test_row,
    )
    if inputs is None:
        return  # no JP history / test data — can't identify the pump
    nozzle = inputs["nozzle"]
    throat = inputs["throat"]
    # Corrupt / forward pump record (S-17, 'D' in the nozzle column): model the
    # sidebar pump instead of crashing on JetPump('D').
    if not _is_valid_pump_code(nozzle, throat):
        nozzle, throat = params.nozzle_no, params.area_ratio

    st.markdown("##### Match test oil rate (infer BHP)")
    st.caption(
        f"No BHP gauge on this test, but the oil rate is known "
        f"(**{oil_test:,.0f} BOPD**) and the pump installed at the test "
        f"(**{nozzle}{throat}**) is known. Infer the flowing BHP that makes "
        f"{nozzle}{throat} model that oil rate at PF "
        f"**{params.ppf_surf:,.0f} psi**, then seed the sidebar IPR with the "
        "consistent solution — no hand-tweaking."
    )

    if st.button(
        "🎯 Match test oil rate (infer BHP)",
        type="primary",
        key="sw_oil_backmatch_btn",
        use_container_width=True,
        help=(
            "Finds the flowing BHP (pwf) such that the test's installed pump, "
            "at the test's conditions and the sidebar reservoir pressure, "
            "reproduces the known test oil rate. Seeds qwf / pwf / Reservoir "
            "Pressure into the sidebar so every view stays consistent."
        ),
    ):
        from woffl.gui.ipr_backmatch import match_oil_rate
        from woffl.gui.sidebar import clamp_seed

        pres = float(params.pres)
        with st.spinner(
            f"Inferring flowing BHP so {nozzle}{throat} matches "
            f"{oil_test:,.0f} BOPD..."
        ):
            result = match_oil_rate(
                qoil_test=float(oil_test),
                pres=pres,
                nozzle=nozzle,
                throat=throat,
                surf_pres=inputs["model_surf_pres"],
                form_temp=float(params.form_temp),
                rho_pf=float(params.rho_pf),
                ppf_surf=float(params.ppf_surf),
                wellbore=wellbore,
                well_profile=well_profile,
                form_wc=float(params.form_wc),
                form_gor=float(params.form_gor),
                ken=float(params.ken),
                kth=float(params.kth),
                kdi=float(params.kdi),
                field_model=params.field_model,
                jpump_direction=params.jpump_direction,
            )

        if not result.ok or result.pwf is None:
            # Pump-limited / no-bracket: explain, do NOT touch the sidebar.
            st.warning(result.message, icon="⚠️")
            return

        # Seed the sidebar IPR with the consistent solution. Follow the
        # documented logical-key + pop-widget-key + rerun dance (the sidebar
        # already rendered this run), and clamp every seed into its widget's
        # bounds. Respect the per-well GOR floor if a marginal-well recovery
        # set one (we don't change GOR, but keep parity with the other seeders).
        seed_qwf = clamp_seed("qwf", int(round(result.qwf_oil)))
        seed_pwf = clamp_seed("pwf", int(round(result.pwf)))
        seed_res = clamp_seed("res_pres", int(round(result.pres)))
        for k, v in (("qwf", seed_qwf), ("pwf", seed_pwf), ("res_pres", seed_res)):
            st.session_state[k] = v
            st.session_state.pop(f"{k}_input", None)

        st.session_state["_oil_backmatch_msg"] = (
            f"Inferred flowing BHP = {seed_pwf:,} psi so {nozzle}{throat} "
            f"matches {result.qwf_oil:,.0f} BOPD (modeled "
            f"{result.modeled_oil:,.0f} BOPD). Seeded the sidebar IPR "
            f"(qwf {seed_qwf:,} BOPD · pwf {seed_pwf:,} psi · ResP "
            f"{seed_res:,} psi). Edit any field in the sidebar to override."
        )
        st.rerun()


def _render_joint_automatch(
    params: SimulationParams,
    wellbore,
    well_profile,
    *,
    selected_test_row=None,
) -> None:
    """Button: auto-match BOTH the test oil rate AND power-fluid rate.

    This is the headline per-well calibration. The engineer knows the test's oil
    rate, PF (lift-water) rate, and the installed pump, and wants the model to
    reproduce *both* before trusting it in the optimizer. Hand-tweaking the IPR +
    friction to hit two targets at once is impractical, so this does it
    numerically: it searches the IPR productivity, PF surface pressure, and
    friction coefs (ken/kth/kdi) for the combination that makes the installed
    pump model the test's oil AND PF, then seeds those into the sidebar.

    When the test also has a measured BHP, it's added as a third target (the
    friction coefs reconcile it). For a gaugeless well the flowing BHP falls out
    of the match (seeded as pwf). If the targets can't both be hit, the result
    explains WHY (PF-limited / pump-capacity-limited / etc.) instead of silently
    seeding a bad point — see :mod:`woffl.gui.joint_match`.
    """
    from woffl.gui.utils import build_calibration_inputs

    if params.selected_well == "Custom":
        return

    actuals = _actuals_from_test(selected_test_row)
    oil_test = actuals.get("oil")
    pf_test = actuals.get("pf")
    if not oil_test or oil_test <= 0 or not pf_test or pf_test <= 0:
        return  # need BOTH an oil and a PF target
    bhp_test = actuals.get("bhp")  # optional third target

    inputs = build_calibration_inputs(
        params,
        wellbore,
        well_profile,
        selected_test_row=selected_test_row,
    )
    if inputs is not None:
        nozzle, throat = inputs["nozzle"], inputs["throat"]
        model_surf_pres = inputs["model_surf_pres"]
        if not _is_valid_pump_code(nozzle, throat):
            nozzle, throat = params.nozzle_no, params.area_ratio
        pump_src = "installed"
    else:
        # No JP history (e.g. S-67: tests but no installs) — model the SIDEBAR
        # pump against the test. We can still match oil + PF; the engineer chose
        # the pump. Mirrors Model-vs-Actual's sidebar-pump fallback so the button
        # isn't hidden just because the well lacks install records.
        nozzle, throat = params.nozzle_no, params.area_ratio
        model_surf_pres = float(params.surf_pres)
        pump_src = "sidebar"

    st.markdown("##### Auto-match oil + power fluid")
    _bhp_note = f" and BHP **{bhp_test:,.0f} psi**" if bhp_test else " (BHP inferred)"
    st.caption(
        f"Hold the measured PF pressure (**{params.ppf_surf:,.0f} psi**) and match "
        f"the {pump_src} pump **{nozzle}{throat}** to this test's oil "
        f"**{oil_test:,.0f} BOPD**{_bhp_note} by fitting the IPR, then seed the "
        f"sidebar. The PF-rate gap vs the measured **{pf_test:,.0f} BWPD** is "
        "reported honestly (it reflects nozzle/area uncertainty), not hidden by "
        "dropping pressure."
    )
    float_pf = st.checkbox(
        "Let PF pressure float to also match the PF rate (advanced)",
        value=False,
        key="sw_joint_float_pf",
        help="Off (default): hold the real PF pressure, match oil, report the PF "
        "residual. On: let the matcher lower PF pressure to also hit the PF "
        "rate — it can land below the real header, so use only to explore.",
    )

    if st.button(
        "🎯 Auto-match oil + PF",
        type="primary",
        key="sw_joint_automatch_btn",
        use_container_width=True,
        help=(
            "Numerically searches IPR productivity, PF surface pressure, and "
            "friction coefs so the installed pump models BOTH the test oil and PF "
            "rate. Seeds qwf / pwf / ResP / ppf_surf / ken / kth / kdi into the "
            "sidebar. Edit any field afterward to override."
        ),
    ):
        from woffl.gui.joint_match import joint_match
        from woffl.gui.sidebar import clamp_seed

        with st.spinner(
            f"Searching IPR + JP params so {nozzle}{throat} matches "
            f"{oil_test:,.0f} BOPD oil and {pf_test:,.0f} BWPD PF…"
        ):
            r = joint_match(
                oil_target=float(oil_test),
                pf_target=float(pf_test),
                pres=float(params.pres),
                nozzle=nozzle,
                throat=throat,
                surf_pres=model_surf_pres,
                form_temp=float(params.form_temp),
                rho_pf=float(params.rho_pf),
                ppf_surf0=float(params.ppf_surf),
                wellbore=wellbore,
                well_profile=well_profile,
                form_wc=float(params.form_wc),
                form_gor=float(params.form_gor),
                ken0=float(params.ken),
                kth0=float(params.kth),
                kdi0=float(params.kdi),
                field_model=params.field_model,
                jpump_direction=params.jpump_direction,
                bhp_target=(float(bhp_test) if bhp_test else None),
                pin_ppf=not float_pf,
            )

        # Seed the sidebar with the matched params — even on a PARTIAL match we
        # seed (it's the closest consistent point) but flag it. Logical key +
        # pop widget key + rerun, every value clamped to its widget bounds.
        seeds = {
            "qwf": clamp_seed("qwf", int(round(r.qwf_oil))),
            "pwf": clamp_seed("pwf", int(round(r.pwf))),
            "res_pres": clamp_seed("res_pres", int(round(r.pres))),
            "ppf_surf": clamp_seed("ppf_surf", int(round(r.ppf_surf))),
            "ken": clamp_seed("ken", round(float(r.ken), 4)),
            "kth": clamp_seed("kth", round(float(r.kth), 4)),
            "kdi": clamp_seed("kdi", round(float(r.kdi), 4)),
        }
        for k, v in seeds.items():
            st.session_state[k] = v
            st.session_state.pop(f"{k}_input", None)

        pinned = not float_pf
        ppf_note = (
            ""
            if pinned
            else (
                f" ⚠️ PF pressure moved "
                f"{abs(r.ppf_surf - float(params.ppf_surf)):,.0f} psi to {r.ppf_surf:,.0f} "
                f"(from {params.ppf_surf:,.0f}) — sanity-check vs the known header."
                if abs(r.ppf_surf - float(params.ppf_surf)) > 500
                else ""
            )
        )
        seeded = (
            f"Seeded qwf {seeds['qwf']:,} · pwf {seeds['pwf']:,} · ResP {seeds['res_pres']:,}"
            + ("" if pinned else f" · PF {seeds['ppf_surf']:,} psi")
            + f" · ken {seeds['ken']:.3f}/kth {seeds['kth']:.3f}/kdi {seeds['kdi']:.3f}."
        )
        bhp_txt = (
            f" · BHP {r.modeled_bhp:,.0f} psi ({r.bhp_err_pct:+.0f}%)"
            if r.bhp_err_pct is not None
            else f" · BHP {r.modeled_bhp:,.0f} psi"
        )
        if pinned and r.ok:
            text = (
                f"**{nozzle}{throat} · PF pressure held at {seeds['ppf_surf']:,} psi.** "
                f"Oil matched {r.modeled_oil:,.0f} BOPD ({r.oil_err_pct:+.0f}%){bhp_txt}. "
                f"PF reads {r.modeled_pf:,.0f} vs measured {pf_test:,.0f} BWPD "
                f"({r.pf_err_pct:+.0f}%) — left as nozzle/area uncertainty. " + seeded
            )
        elif pinned:
            text = (
                f"PF held at {seeds['ppf_surf']:,} psi but couldn't match oil for "
                f"{nozzle}{throat}. **Why:** {r.diagnostic} " + seeded
            )
        elif r.ok:
            text = (
                f"Matched {nozzle}{throat}: oil {r.modeled_oil:,.0f} BOPD "
                f"({r.oil_err_pct:+.0f}%) · PF {r.modeled_pf:,.0f} BWPD "
                f"({r.pf_err_pct:+.0f}%){bhp_txt}. " + seeded + ppf_note
            )
        else:
            text = (
                f"Could not match both targets for {nozzle}{throat} "
                f"(closest: oil {r.modeled_oil:,.0f} BOPD {r.oil_err_pct:+.0f}%, "
                f"PF {r.modeled_pf:,.0f} BWPD {r.pf_err_pct:+.0f}%). "
                f"**Why:** {r.diagnostic} Seeded the closest point." + ppf_note
            )
        st.session_state["_joint_automatch_msg"] = {"text": text, "ok": r.ok}
        st.rerun()


def _render_fric_calibration_section(
    params: SimulationParams,
    wellbore,
    well_profile,
    ipr_inflow,
    ipr_res_mix,
    nozzle: str,
    throat: str,
    actual_bhp,
    model_surf_pres: float,
) -> None:
    """Render the Friction-Coef Calibration result display inside MvA.

    Buttons (Run + Push to sidebar) live in the action bar below the hero
    strip — see _render_fric_cal_action_bar. This section is read-only:
    just the metrics and convergence flags from the most recent run.
    """
    st.markdown("#### Friction-Coef Calibration (BHP-target)")

    if not is_valid_number(actual_bhp):
        st.caption(
            "Cannot calibrate — most recent test has no measured BHP. "
            "Friction-coef calibration requires an actual BHP to target."
        )
        return

    st.caption(
        f"Sweeps ken, kth, and kdi via Nelder-Mead to drive modeled BHP "
        f"toward actual ({actual_bhp:.0f} psi). knz is held fixed at 0.01. "
        f"ken seed is the sidebar value ({params.ken:.3f}). "
        "**Run / push controls live in the action bar at the top of this view.**"
    )

    cal_state = st.session_state.get("sw_fric_calibration", {})
    result = cal_state.get(params.selected_well)
    if result is None:
        st.info(
            "No calibration run yet for this well. Click **Run BHP Calibration** "
            "in the action bar above the Model vs Actual section."
        )
        return

    if not result.converged:
        st.warning(
            "Calibration did not converge — the optimizer could not find a "
            "(kth, kdi) combination that produced a successful solve. "
            "Consider checking the IPR or solver inputs."
        )
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Cal ken", f"{result.best_ken:.3f}")
    c2.metric("Cal kth", f"{result.best_kth:.3f}")
    c3.metric("Cal kdi", f"{result.best_kdi:.3f}")
    c4.metric(
        "Modeled BHP",
        f"{result.best_modeled_bhp:.0f} psi",
        delta=f"{result.bhp_error:+.0f}",
    )
    c5.metric("Modeled Oil", f"{result.best_oil:.0f} BOPD")

    iter_str = f", {result.iterations} iters" if result.iterations is not None else ""
    starts_str = (
        f", {result.starts_tried} seed" + ("s" if result.starts_tried != 1 else "")
        if hasattr(result, "starts_tried")
        else ""
    )
    flags = []
    if getattr(result, "bounded", False):
        flags.append("**bounded** (optimum at search edge — try varying ken)")
    if getattr(result, "sonic", False):
        flags.append("**sonic** (BHP choke-pinned by throat geometry)")
    flags_str = " · " + ", ".join(flags) if flags else ""

    quality_color = {"good": "green", "fair": "orange", "poor": "red"}.get(
        getattr(result, "match_quality", "unknown"), "gray"
    )
    st.markdown(
        f"Actual BHP: **{result.target_bhp:.0f}** psi · "
        f"Match: :{quality_color}[**{getattr(result, 'match_quality', 'unknown').upper()}**] "
        f"({iter_str}{starts_str}){flags_str}"
    )


def _get_well_tests(well_name: str):
    """Get tests for a single well, with memory-gauge BHP override applied."""
    from woffl.gui.utils import get_well_tests_for_well

    return get_well_tests_for_well(well_name)


def _render_ipr_rate_calculator(params: SimulationParams, res_pres: float) -> None:
    """Inline what-if under the IPR chart: type a flowing BHP, read the
    rates off the SAME Vogel curve the chart draws.

    Anchored exactly like everything else on this tab: sidebar qwf (OIL,
    BOPD) at sidebar pwf, sidebar/anchor reservoir pressure, sidebar WC for
    the oil↔liquid split. No solver run — pure IPR arithmetic.
    """

    def _vogel_frac(p: float, pr: float) -> float:
        r = max(0.0, min(p / pr, 1.0))
        return 1.0 - 0.2 * r - 0.8 * r * r

    pr = float(res_pres)
    oil_anchor = float(params.qwf)
    pwf_anchor = float(params.pwf)
    wc = min(max(float(params.form_wc), 0.0), 0.99)
    anchor_frac = _vogel_frac(pwf_anchor, pr)
    if pr <= 0 or oil_anchor <= 0 or anchor_frac <= 0:
        return

    st.markdown("##### Rate at a given BHP")
    c1, c2, c3, c4 = st.columns([1.3, 1, 1, 1])
    with c1:
        bhp_in = st.number_input(
            "Flowing BHP (psi)",
            min_value=0,
            value=int(round(pwf_anchor)),
            step=25,
            key="sw_ipr_calc_bhp",
            help=(
                "Reads liquid/oil/water off the Vogel curve above "
                "(sidebar-anchored IPR, sidebar WC)."
            ),
        )
    qmax_oil = oil_anchor / anchor_frac
    oil = qmax_oil * _vogel_frac(float(bhp_in), pr)
    liquid = oil / (1.0 - wc)
    water = liquid - oil
    c2.metric("Liquid", f"{liquid:,.0f} BLPD")
    c3.metric("Oil", f"{oil:,.0f} BOPD")
    c4.metric("Water", f"{water:,.0f} BWPD")
    if float(bhp_in) >= pr:
        st.caption(
            f"BHP ≥ reservoir pressure ({pr:.0f} psi) — no inflow at this drawdown."
        )
    else:
        st.caption(
            f"Vogel anchored at **{oil_anchor:,.0f} BOPD @ {pwf_anchor:,.0f} psi**, "
            f"ResP **{pr:,.0f} psi** · WC **{wc:.2f}** "
            f"(oil = liquid × {1.0 - wc:.2f})."
        )


def _actuals_from_test(test_row) -> dict[str, float | None]:
    """Extract Oil / BHP / PF actuals from a single well-test row.

    Returns all-None when ``test_row`` is None (e.g. no tests available or a
    Custom well).
    """
    blank = {"oil": None, "bhp": None, "pf": None}
    if test_row is None:
        return blank

    def _get(col: str) -> float | None:
        v = test_row.get(col)
        return float(v) if is_valid_number(v) else None

    return {
        "oil": _get("WtOilVol"),
        "bhp": _get("BHP"),
        "pf": _get("lift_wat"),
    }


def _pump_at_test_date(jp_hist, well_name: str, test_date):
    """Return the pump installed on or before ``test_date`` for the given well.

    Mirrors ``get_current_pump`` but with a date filter, so the Model vs Actual
    section can model with the pump that was *actually* in the well at the time
    of the selected test rather than the current installation.

    Returns a dict (same shape as ``get_current_pump``) or None when no install
    record qualifies.
    """
    import pandas as pd

    if jp_hist is None or test_date is None:
        return None
    if isinstance(test_date, float) and pd.isna(test_date):
        return None

    well_df = jp_hist[jp_hist["Well Name"] == well_name].copy()
    if well_df.empty:
        return None
    well_df = well_df.dropna(subset=["Date Set"])
    well_df = well_df[well_df["Date Set"] <= pd.Timestamp(test_date)]
    if well_df.empty:
        return None

    latest = well_df.sort_values("Date Set", ascending=False).iloc[0]
    nozzle = latest.get("Nozzle Number")
    throat = latest.get("Throat Ratio")
    tubing = latest.get("Tubing Diameter")
    date_set = latest.get("Date Set")

    nozzle_str = None
    if pd.notna(nozzle):
        try:
            nozzle_str = str(int(nozzle))
        except (TypeError, ValueError):
            nozzle_str = None

    tubing_val = None
    if pd.notna(tubing):
        try:
            tubing_val = float(tubing)
        except (TypeError, ValueError):
            tubing_val = None

    return {
        "nozzle_no": nozzle_str,
        "throat_ratio": str(throat).strip() if pd.notna(throat) else None,
        "tubing_od": tubing_val,
        "date_set": date_set,
    }


def _resolve_anchor_test_row(well_name: str, sorted_tests):
    """The well-test row the IPR anchor currently points to.

    Used to keep the 'Test to compare against' picker in sync with the IPR
    anchor. Delegates to the SAME resolver the anchor itself uses
    (ipr_anchor._resolve_anchor_row, via compute_anchored_vogel) so the two can
    never disagree: drop rows missing BHP *or* WtTotalFluid, then median
    (nearest-median-BHP) / specific (matching date, else most-recent fallback) /
    most-recent. Previously this re-derived the row with a BHP-only filter, a
    date-desc tie-break, and no specific→recent fallback, so for median / BHP-less
    'specific' anchors the comparison test could diverge from the test the IPR
    was built on. ``sorted_tests`` must be WtDate-desc.
    """
    import pandas as pd

    from woffl.gui.ipr_anchor import _resolve_anchor_row

    if sorted_tests is None or sorted_tests.empty:
        return None
    sig = st.session_state.get(f"sw_ipr_applied_sig_{well_name}")
    mode = sig[0] if sig else "recent"
    token = sig[1] if sig else None

    # Same fittable-row filter the anchor uses. If nothing is fittable the anchor
    # falls back to its synthetic path, so compare against the most-recent test.
    df = sorted_tests.dropna(subset=["BHP", "WtTotalFluid"]).copy()
    if df.empty:
        return sorted_tests.iloc[0]
    df["__date"] = pd.to_datetime(df.get("WtDate"), errors="coerce")
    anchor_date = token if mode == "specific" else None
    row, _label = _resolve_anchor_row(df, mode, anchor_date)
    return row.drop(labels="__date", errors="ignore")


def _anchor_test_pf(test_df, anchor_mode: str, anchor_date):
    """Test-day live PF of the test the anchor CURRENTLY points to.

    Same fittable-row filter + resolver as :func:`_resolve_anchor_test_row`,
    but takes the mode/date directly (the applied-sig key isn't written yet at
    seed time). Returns ``(pf_press, pf_source, wt_date)`` or
    ``(None, None, None)`` when the anchor test has no valid same-day reading
    (pre-2024-09-25 tests, manual tests, dead gauges).
    """
    import pandas as pd

    from woffl.gui.ipr_anchor import _resolve_anchor_row
    from woffl.gui.utils import pf_from_test_row

    if test_df is None or test_df.empty or "pf_press" not in test_df.columns:
        return None, None, None
    df = test_df.dropna(subset=["BHP", "WtTotalFluid"]).copy()
    if df.empty:
        return None, None, None
    df["__date"] = pd.to_datetime(df.get("WtDate"), errors="coerce")
    try:
        row, _label = _resolve_anchor_row(
            df, anchor_mode, anchor_date if anchor_mode == "specific" else None
        )
    except Exception:
        return None, None, None
    live = pf_from_test_row(row)
    if live is None:
        return None, None, None
    return live["pf_press"], live.get("pf_source"), live.get("pf_date")


def _render_test_picker(well_name: str, test_df, *, synced: bool = False):
    """Selectbox listing the well's tests (date-desc); returns the picked row.

    Default = most recent test, so behaviour matches the pre-picker version
    when the user doesn't interact. The picker drives:
      * the hero-strip vs-actual deltas (Oil / PF / BHP),
      * the Model vs Actual section's modeled pump (via :func:`_pump_at_test_date`)
        and its comparison row.

    When a memory gauge is active, the picker is filtered to tests on dates
    the gauge actually covers (the union of daily medians across all loaded
    files). Tests in gaps between multi-file uploads, or outside any file's
    window, are hidden — picking them would produce a meaningless comparison
    because their BHP is NaN.

    Selection persists per-well via session_state, so switching wells doesn't
    carry stale state. Returns None when no tests exist.
    """
    import pandas as pd

    from woffl.gui.memory_gauge import get_gauge

    if test_df is None or test_df.empty:
        return None

    gauge = get_gauge(well_name)
    if gauge is not None and not gauge.daily_df.empty:
        dates = pd.to_datetime(test_df["WtDate"]).dt.normalize()
        test_df = test_df[dates.isin(gauge.daily_df["tag_date"])]
        if test_df.empty:
            st.info(
                f"No {well_name} tests fall on dates the memory gauge "
                "covers. Upload an additional gauge file (or check the "
                "file's date range against your test history) to enable "
                "model-vs-actual comparisons."
            )
            return None

    sorted_tests = test_df.sort_values("WtDate", ascending=False).reset_index(drop=True)

    # Pump-at-test-date lookup needs JP history. Pull once, reuse per row.
    jp_hist = st.session_state.get("jp_history_df")

    def _fmt(row) -> str:
        date = row.get("WtDate")
        date_str = date.strftime("%Y-%m-%d") if pd.notna(date) else "N/A"
        parts = [date_str]
        # JP that was in the well at this test's date — same lookup the
        # solver uses, so the option label matches what's about to be modeled.
        if pd.notna(date) and jp_hist is not None:
            pump = _pump_at_test_date(jp_hist, well_name, date)
            if pump and pump.get("nozzle_no") and pump.get("throat_ratio"):
                parts.append(f"{pump['nozzle_no']}{pump['throat_ratio']}")
        oil = row.get("WtOilVol")
        if is_valid_number(oil):
            parts.append(f"Oil {float(oil):,.0f}")
        bhp = row.get("BHP")
        if is_valid_number(bhp):
            parts.append(f"BHP {float(bhp):,.0f}")
        pf = row.get("lift_wat")
        if is_valid_number(pf):
            parts.append(f"PF {float(pf):,.0f}")
        return "  ·  ".join(parts)

    options = [_fmt(row) for _, row in sorted_tests.iterrows()]
    key = f"sw_test_picker_{well_name}"
    shadow_key = f"sw_test_picker_date_{well_name}"

    # SYNCED (default): slave the comparison test to the IPR anchor so the model
    # is compared against the same test the IPR is built on. Only meaningful with
    # 2+ tests (i.e. when the IPR anchor selector is shown); with 0/1 tests there
    # is nothing to anchor, so fall through to the normal independent picker.
    if synced and len(sorted_tests) >= 2:
        target = _resolve_anchor_test_row(well_name, sorted_tests)
        tgt_idx = 0
        if target is not None and pd.notna(target.get("WtDate")):
            ttok = target["WtDate"].strftime("%Y-%m-%d")
            for i, (_, r) in enumerate(sorted_tests.iterrows()):
                d = r.get("WtDate")
                if pd.notna(d) and d.strftime("%Y-%m-%d") == ttok:
                    tgt_idx = i
                    break
        # Pop the key so `index` wins, then render disabled (visibly slaved).
        st.session_state.pop(key, None)
        st.selectbox(
            "Test to compare against",
            options=options,
            index=tgt_idx,
            key=key,
            disabled=True,
            help=(
                "Synced to the IPR anchor selector just above. Check 'Use a "
                "different test for comparison' under the IPR anchor to pick "
                "the comparison test independently."
            ),
        )
        return sorted_tests.iloc[tgt_idx]

    # DECOUPLED (or <2 tests): independent picker.
    # Clamp persisted selection to the current option list so a stale value
    # from a prior session doesn't crash the selectbox if the test cache
    # changed shape.
    if st.session_state.get(key) not in range(len(sorted_tests)):
        st.session_state.pop(key, None)

    # The view switcher renders only the active view, so Streamlit drops this
    # selectbox's widget state when the user detours through Batch Run / PF
    # Range. The shadow date key (a non-widget key) survives, so we re-seed the
    # default index from it on return rather than snapping back to most-recent.
    # When the widget state survived, Streamlit ignores `index` and keeps it.
    default_idx = 0
    token = st.session_state.get(shadow_key)
    if token:
        for i, (_, r) in enumerate(sorted_tests.iterrows()):
            d = r.get("WtDate")
            if pd.notna(d) and d.strftime("%Y-%m-%d") == token:
                default_idx = i
                break

    # Select by POSITION (options are integer indices rendered via format_func),
    # not by label: two tests can format to an identical label (same date + pump
    # + rates, now possible with manual provisional tests), and the old
    # options.index(selected) returned the FIRST match — operating on the wrong
    # row when the second was chosen.
    idx = st.selectbox(
        "Test to compare against",
        options=list(range(len(sorted_tests))),
        index=default_idx,
        format_func=lambda i: options[i],
        key=key,
        help=(
            "Pick a well test. The hero-strip vs-actual deltas, the Model vs "
            "Actual comparison, and the pump used for that comparison (from "
            "JP history at the test's date) all key off this selection. "
            "Default = most recent test."
        ),
    )

    row = sorted_tests.iloc[idx]
    # Persist the pick so a Batch/PF detour doesn't snap it back to most-recent.
    _d = row.get("WtDate")
    if pd.notna(_d):
        st.session_state[shadow_key] = _d.strftime("%Y-%m-%d")
    return row


# ── Saved IPR-anchor pin (mpu.wells.prop_hist, ipr_wt_uid) ──────────────────
# On a genuinely fresh well load (no sw_ipr_applied_sig_<well> yet this
# session), the anchor selector below defaults to a previously-saved pin
# instead of "Most recent" — see the woffl-prop-hist-persistence plan, W2.
# W3 (save/un-pin) calls _clear_pin_cache after a push so the next render
# re-queries instead of replaying a stale cached lookup.

_PIN_LOOKUP_PROP_ID = "ipr_wt_uid"
_PIN_CACHE_KEY = "_pin_cache"


def _clear_pin_cache(well_name: str) -> None:
    """Drop the memoized IPR-anchor pin lookup for one well.

    SEAM for W3: call this right after a successful save/un-pin push for
    ``well_name`` so the next render of :func:`_render_ipr_anchor_control`
    re-queries ``prop_hist`` instead of replaying the pre-save memo.
    """
    st.session_state.get(_PIN_CACHE_KEY, {}).pop(well_name, None)


def _load_pinned_anchor(well_name: str, test_df) -> dict | None:
    """Saved IPR-anchor pin for ``well_name``, memoized once per well per session.

    Reads ``mpu.wells.prop_hist`` (``ipr_wt_uid``) via
    ``prop_hist_client.fetch_latest_prop`` at most ONCE per well per Streamlit
    session — the memo lives in ``st.session_state[_PIN_CACHE_KEY]`` (a plain
    ``{well_name: result}`` dict) because :func:`_render_ipr_anchor_control`
    runs on every rerun and this must not re-hit Databricks each time.

    ANY failure (offline, missing grant, malformed row) degrades to "no pin"
    and is logged via ``logger.warning`` — NEVER raised, NEVER surfaced as
    ``st.error``: a saved-IPR lookup failing must never block the Solver. The
    failure itself is memoized too, so a dead connection doesn't retry every
    rerun.

    Returns:
      * ``None`` — no saved pin, an un-pinned NULL ``prop_value`` (see W3;
        there is no numeric sentinel — real ``wt_uid`` values are signed and
        span both positive and negative ranges), or the lookup failed.
      * ``{"status": "applied", "mode": "specific", "date_token":
        "YYYY-MM-DD", "wt_uid": int, "entry_user": str, "entry_datetime":
        ...}`` — a pin whose test is present in the CURRENT test frame.
      * ``{"status": "stale", "wt_uid": int, "entry_user": str,
        "entry_datetime": ...}`` — a pin whose test has aged out of the frame
        (tighter lookback, memory-gauge filtering, count cap).
    """
    cache = st.session_state.setdefault(_PIN_CACHE_KEY, {})
    if well_name in cache:
        return cache[well_name]

    import pandas as pd

    result = None
    try:
        from woffl.assembly.prop_hist_client import fetch_latest_prop
        from woffl.gui.ipr_anchor import find_test_row_by_wt_uid

        fetched = fetch_latest_prop(well_name, _PIN_LOOKUP_PROP_ID)
        if fetched is not None:
            value, entry_datetime, entry_user = fetched
            # A pin EXISTS iff prop_value is a real finite number (not
            # None/NaN) -- NO sign-based rule. Real wt_uid values are signed
            # and span both positive and negative ranges (observed roughly
            # -3.6M to +3.1M), so a negative uid is a perfectly valid pin.
            if value is not None and pd.notna(value):
                wt_uid = int(value)
                row = find_test_row_by_wt_uid(test_df, wt_uid)
                if row is not None and pd.notna(row.get("WtDate")):
                    result = {
                        "status": "applied",
                        "mode": "specific",
                        "date_token": row["WtDate"].strftime("%Y-%m-%d"),
                        "wt_uid": wt_uid,
                        "entry_user": entry_user,
                        "entry_datetime": entry_datetime,
                    }
                else:
                    result = {
                        "status": "stale",
                        "wt_uid": wt_uid,
                        "entry_user": entry_user,
                        "entry_datetime": entry_datetime,
                    }
    except Exception:
        logger.warning(
            "IPR-anchor pin lookup failed for %s; falling back to the "
            "default anchor.",
            well_name,
            exc_info=True,
        )
        result = None

    cache[well_name] = result
    return result


def _format_pin_date(value) -> str:
    """Best-effort YYYY-MM-DD formatting for a prop_hist ``entry_datetime``
    value (a python ``datetime``/``date``, a pandas ``Timestamp``, or a raw
    string depending on the connector) — never raises. ``entry_datetime`` is
    a full timestamp (the column was migrated off the old date-only
    ``entry_date`` -- see ``prop_hist_client``), but this still renders as a
    plain date, same as before the migration."""
    import pandas as pd

    try:
        if pd.isna(value):
            return str(value)
    except (TypeError, ValueError):
        pass
    try:
        return value.strftime("%Y-%m-%d")
    except (AttributeError, ValueError):
        return str(value)


def _render_pin_provenance_caption(
    pin_info: dict | None, mode: str, anchor_date
) -> None:
    """Provenance caption just under the anchor selector.

    Two cases, both informational and non-blocking:
      * ``pin_info["status"] == "applied"`` AND the anchor CURRENTLY resolves
        to that pin's test (mode + date match) — "Saved IPR: test ...". Shown
        regardless of whether this is the first fresh-load render or a later
        rerun after the seed-to-sidebar applied it, as long as the selection
        still matches the pin (the user hasn't picked something else).
      * ``pin_info["status"] == "stale"`` — the saved test aged out of the
        current frame. Only shown while the anchor is actually resolving to
        "most recent" (``mode == "recent"``) — the literal fallback this
        caption describes. If the user has since picked a DIFFERENT anchor
        (median / another specific test) on purpose, this stops narrating a
        fallback that no longer describes what's on screen.
    """
    if pin_info is None:
        return

    import pandas as pd

    if pin_info.get("status") == "applied":
        date_token = pin_info.get("date_token")
        current_token = (
            anchor_date.strftime("%Y-%m-%d")
            if anchor_date is not None and pd.notna(anchor_date)
            else None
        )
        if mode == "specific" and current_token == date_token:
            st.caption(
                f"📌 Saved IPR: test {date_token} "
                f"({pin_info.get('entry_user')}, "
                f"{_format_pin_date(pin_info.get('entry_datetime'))})"
            )
        return

    if pin_info.get("status") == "stale" and mode == "recent":
        st.caption(
            f"Saved IPR (test uid {pin_info.get('wt_uid')}) not in the "
            "current test window — using most recent."
        )


def _render_ipr_pin_controls(
    well_name: str, sorted_tests, pin_info: dict | None
) -> None:
    """ "📌 Save IPR as well default" / "🗑 Clear saved IPR" — the Solver's push
    UI for ``mpu.wells.prop_hist``'s ``ipr_wt_uid`` (W3 of the
    woffl-prop-hist-persistence plan). Renders right under the provenance
    caption, inside the same IPR-anchor group.

    Shares the actual push/clear mechanics with the pad-review save hook
    (``workflow_steps/step_review_wells.py::_save_and_advance``) via
    ``woffl.gui.ipr_anchor.pin_ipr_anchor`` / ``clear_ipr_pin`` — see that
    module for the full contract — so the two push paths can never diverge.

    The Save button is HIDDEN (not just disabled) with a one-line caption
    explaining why when: the ``ALLOW_DATABRICKS_WRITES`` gate is off, the
    anchor has no resolvable test, or the anchor is a manual/provisional test
    (no ``wt_uid`` — never pinnable). The Clear button only renders when a
    pin already exists for this well, per :func:`_load_pinned_anchor`.
    """
    import pandas as pd

    from woffl.gui.ipr_anchor import (
        PIN_SKIP_PREFIX,
        clear_ipr_pin,
        pin_ipr_anchor,
        writes_enabled,
    )

    if not writes_enabled():
        st.caption(
            "Saving an IPR default requires `ALLOW_DATABRICKS_WRITES=true` "
            "in the environment."
        )
        return

    anchor_row = _resolve_anchor_test_row(well_name, sorted_tests)
    raw_uid = anchor_row.get("wt_uid") if anchor_row is not None else None
    has_uid = raw_uid is not None and not pd.isna(raw_uid)

    date_label = "n/a"
    if anchor_row is not None:
        d = anchor_row.get("WtDate")
        if pd.notna(d):
            date_label = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)

    col_save, col_clear = st.columns([2, 1])
    with col_save:
        if not has_uid:
            reason = (
                "no resolvable test"
                if anchor_row is None
                else "this anchor is a manual/provisional test (no measured well-test ID)"
            )
            st.caption(f"📌 Save IPR as well default — unavailable: {reason}.")
        else:
            if st.button(
                "📌 Save IPR as well default",
                key=f"sw_save_ipr_pin_{well_name}",
                help=(
                    f"Saves test {date_label} as this well's default IPR "
                    "anchor (mpu.wells.prop_hist) so it auto-loads for every "
                    "future session."
                ),
            ):
                pushed, message = pin_ipr_anchor(well_name, anchor_row)
                if pushed:
                    _clear_pin_cache(well_name)
                    st.toast(message, icon="📌")
                    st.rerun()
                elif message.startswith(PIN_SKIP_PREFIX):
                    st.caption(message)
                else:
                    st.warning(message)

    with col_clear:
        if pin_info is not None and pin_info.get("status") in ("applied", "stale"):
            if st.button(
                "🗑 Clear saved IPR",
                key=f"sw_clear_ipr_pin_{well_name}",
                help=(
                    "Removes the saved default so this well falls back to "
                    "the most-recent test."
                ),
            ):
                cleared, message = clear_ipr_pin(well_name)
                if cleared:
                    _clear_pin_cache(well_name)
                    st.toast(message, icon="🗑")
                    st.rerun()
                else:
                    st.warning(message)


def _render_ipr_anchor_control(well_name: str, test_df):
    """Selector for which test anchors the Vogel IPR (separate from the
    comparison picker).

    Modes:
      * ``recent``   — anchor on the most-recent test; reservoir pressure fit
        to the whole test cloud (unchanged library behavior).
      * ``median``   — anchor on the median-BHP test; reservoir pressure re-fit
        for the best fit *through* that test.
      * ``specific`` — anchor on a user-picked test; reservoir pressure re-fit
        through it.

    Returns ``(anchor_mode, anchor_date)``; the caller passes these to
    :func:`_sync_chosen_ipr_to_sidebar`, which seeds the chosen test's IPR +
    fluid into the sidebar so every view (Batch Run, PF Range, top solver, and
    this section) uses the same curve.
    """
    import pandas as pd

    label_to_mode = {
        "Most recent": "recent",
        "Median test": "median",
        "Specific test": "specific",
    }
    mode_order = ["recent", "median", "specific"]

    # The view switcher renders only the active view, so Streamlit garbage-
    # collects this selectbox's widget state whenever the user detours through
    # Batch Run / PF Range. Restore the selection from the last-APPLIED anchor
    # (sw_ipr_applied_sig_<well>, a non-widget key that survives the detour) so
    # returning to the Solver tab doesn't snap back to "Most recent" — which
    # would also reseed the sidebar back to the recent fit, losing the user's
    # chosen IPR. When the widget state survived, Streamlit ignores `index` and
    # keeps the live value.
    applied_sig = st.session_state.get(f"sw_ipr_applied_sig_{well_name}")

    # Saved IPR-anchor pin (prop_hist's ipr_wt_uid, see the
    # woffl-prop-hist-persistence plan's W2). Looked up on EVERY render
    # (memoized — see _load_pinned_anchor) so the provenance caption below
    # can track it even after the sig gets set on the seed-triggered rerun.
    # Only substitutes into `applied_sig` for the DEFAULT-INDEX computation
    # when no sig exists yet — a genuinely fresh well load, never a session
    # where the user (or a prior rerun) already chose something.
    pin_info = _load_pinned_anchor(well_name, test_df)
    if (
        applied_sig is None
        and pin_info is not None
        and pin_info.get("status") == "applied"
    ):
        applied_sig = (pin_info["mode"], pin_info["date_token"])

    applied_mode = applied_sig[0] if applied_sig else "recent"
    default_mode_idx = (
        mode_order.index(applied_mode) if applied_mode in mode_order else 0
    )

    col_mode, col_pick = st.columns(2)
    with col_mode:
        sel = st.selectbox(
            "IPR anchor",
            options=list(label_to_mode),
            index=default_mode_idx,
            key=f"_sw_ipr_anchor_sel_{well_name}",
            help=(
                "Which test the Vogel IPR is anchored on. 'Most recent' fits "
                "reservoir pressure to the whole test cloud (default). 'Median "
                "test' / 'Specific test' anchor the curve on that test and "
                "re-fit reservoir pressure for the best fit through it. "
                "The 'Test to compare against' picker just below syncs to this "
                "by default (toggle below to unsync)."
            ),
        )
    mode = label_to_mode[sel]

    # Computed once (date-desc, as _resolve_anchor_test_row requires) so both
    # the specific-test picker below AND the save/clear pin controls (which
    # need to resolve the anchor row regardless of mode) share it instead of
    # each re-sorting.
    sorted_tests_desc = (
        test_df.sort_values("WtDate", ascending=False).reset_index(drop=True)
        if test_df is not None and not test_df.empty
        else test_df
    )

    anchor_date = None
    if mode == "specific":
        sorted_tests = sorted_tests_desc
        jp_hist = st.session_state.get("jp_history_df")

        def _opt(row) -> str:
            d = row.get("WtDate")
            parts = [d.strftime("%Y-%m-%d") if pd.notna(d) else "N/A"]
            # JP installed at this test's date — shown like the "Test to compare
            # against" dropdown so the pump is visible here too.
            if pd.notna(d) and jp_hist is not None:
                pump = _pump_at_test_date(jp_hist, well_name, d)
                if pump and pump.get("nozzle_no") and pump.get("throat_ratio"):
                    parts.append(f"{pump['nozzle_no']}{pump['throat_ratio']}")
            bhp = row.get("BHP")
            if is_valid_number(bhp):
                parts.append(f"BHP {float(bhp):,.0f}")
            oil = row.get("WtOilVol")
            if is_valid_number(oil):
                parts.append(f"Oil {float(oil):,.0f}")
            return "  ·  ".join(parts)

        date_opts = [_opt(r) for _, r in sorted_tests.iterrows()]
        anchor_key = f"_sw_ipr_anchor_pick_{well_name}"
        if st.session_state.get(anchor_key) not in range(len(sorted_tests)):
            st.session_state.pop(anchor_key, None)

        # Same restore for the specific-date picker: fall back to the applied
        # anchor date when the widget state was dropped on a tab detour.
        default_date_idx = 0
        if applied_sig and applied_sig[0] == "specific" and applied_sig[1]:
            token = applied_sig[1]
            for i, (_, r) in enumerate(sorted_tests.iterrows()):
                d = r.get("WtDate")
                if pd.notna(d) and d.strftime("%Y-%m-%d") == token:
                    default_date_idx = i
                    break

        with col_pick:
            # Select by POSITION (format_func renders the label) so two tests
            # with an identical label can't collapse onto the first via .index().
            picked_idx = st.selectbox(
                "Anchor test",
                options=list(range(len(sorted_tests))),
                index=default_date_idx,
                format_func=lambda i: date_opts[i],
                key=anchor_key,
                help="The test the Vogel curve is forced through.",
            )
        ad = sorted_tests.iloc[picked_idx].get("WtDate")
        anchor_date = ad if pd.notna(ad) else None

    # Saved-IPR provenance, right under the selector(s) just rendered.
    _render_pin_provenance_caption(pin_info, mode, anchor_date)

    # Save/clear the pin (W3 of the woffl-prop-hist-persistence plan) — right
    # next to the provenance caption above, inside the same anchor group.
    _render_ipr_pin_controls(well_name, sorted_tests_desc, pin_info)

    # Decouple toggle. By default the "Test to compare against" picker just below
    # is slaved to this IPR anchor (so the model is compared against the same
    # test the IPR is built on). Checking this frees the comparison picker to
    # select any test. This control renders at the top of the tab, just BEFORE
    # render_tab reads this key for the picker — so toggling takes effect on the
    # same run. (It resets to synced after a tab detour, which keeps the test
    # selection consistent since the anchor itself persists.)
    st.checkbox(
        "Use a different test for comparison (un-sync from the IPR anchor)",
        key=f"sw_ipr_decouple_{well_name}",
        help=(
            "By default the 'Test to compare against' picker just below matches "
            "the IPR anchor test selected here, so the model is compared against "
            "the same test the IPR is built on. Check this to choose the "
            "comparison test independently."
        ),
    )

    return mode, anchor_date


# Default IPR-anchor signature: most-recent anchor. The sidebar's auto-populate
# already seeds this exact operating point on well selection, so the default
# state must never push back (and never stomp a manual sidebar edit). Only an
# active change of the anchor TEST away from this writes anything.
_IPR_SIDEBAR_DEFAULT_SIG = ("recent", None)


def _sync_chosen_ipr_to_sidebar(
    well_name: str,
    *,
    anchor_mode: str,
    anchor_date,
    qwf_oil: float,
    pwf: float,
    res_p: float,
    form_wc: float,
    fgor: float,
    pf_press: float | None = None,
    pf_source: str | None = None,
    pf_date=None,
) -> None:
    """Seed the chosen test's IPR + fluid inputs into the sidebar.

    The InFlow / ResMix shared by Batch Run, PF Range, and the top Solver are
    built in ``single_well_page._build_simulation_objects`` from the sidebar's
    ``qwf`` / ``pwf`` / ``res_pres`` / ``form_wc`` / ``form_gor``. This makes the
    IPR-anchor test selection seed all five so every view uses the chosen test's
    curve AND fluid; the engineer then overrides any field by editing the
    sidebar (the edit persists for the session because it doesn't change the
    selection signature below). Before this, picking a non-default anchor only
    moved this Model-vs-Actual section — every other view reverted to the
    auto-populated most-recent fit (the reported bug).

    Writes only when the anchor TEST selection changes, tracked via a per-well
    signature. That keeps three things true:
      * the default state (most-recent anchor) never writes — auto-populate
        already seeded it, and manual sidebar edits survive untouched,
      * switching back to "Most recent" restores the recent-fit values,
      * it can't loop: after the push the signature matches and it stops.

    GOR is floored by ``_well_min_gor`` (set by the marginal-well solver
    auto-recovery in utils._trigger_gor_reset) so re-seeding can't undo a
    recovery that lifted GOR off an unsolvable value.

    Follows the documented logical-key + pop-widget-key + ``st.rerun`` dance:
    the sidebar already rendered this run, so its ``*_input`` widget keys can't
    be written directly (Streamlit raises after the widget renders).
    """
    import pandas as pd

    sig_key = f"sw_ipr_applied_sig_{well_name}"
    date_token = (
        anchor_date.strftime("%Y-%m-%d")
        if anchor_date is not None and pd.notna(anchor_date)
        else None
    )
    current_sig = (anchor_mode, date_token)
    if current_sig == st.session_state.get(sig_key, _IPR_SIDEBAR_DEFAULT_SIG):
        return

    # Match the int/round casting AND bounds-clamping the sidebar auto-populate
    # uses (sidebar.py _auto_populate_from_ipr / clamp_seed), so the recent-fit
    # case is a byte-for-byte no-op and a seed can never land outside its
    # widget's bounds (Streamlit silently resets out-of-range values to the
    # widget minimum).
    from woffl.gui.sidebar import clamp_seed

    gor_floor = st.session_state.get("_well_min_gor", {}).get(well_name, 0)
    new_vals = {
        "qwf": clamp_seed("qwf", int(qwf_oil)),
        "pwf": clamp_seed("pwf", int(pwf)),
        "res_pres": clamp_seed("res_pres", int(res_p)),
        "form_wc": clamp_seed("form_wc", round(float(form_wc), 2)),
        "form_gor": clamp_seed("form_gor", max(int(fgor), gor_floor)),
    }
    # PF follows the anchor test's DAY (live vw_pressure_daily reading joined
    # onto the test row) so the whole sidebar models the same day. Tests
    # without a valid same-day reading leave PF untouched.
    if pf_press is not None:
        new_vals["ppf_surf"] = clamp_seed("ppf_surf", int(round(pf_press)))

    # Record the selection as applied up front so we don't keep re-evaluating.
    st.session_state[sig_key] = current_sig

    if all(st.session_state.get(k) == v for k, v in new_vals.items()):
        return  # sidebar already matches — nothing to push

    for k, v in new_vals.items():
        st.session_state[k] = v
        # Drop the widget key so the sidebar's _number_input re-seeds it from
        # the logical key on the next run (writing *_input after render raises).
        st.session_state.pop(f"{k}_input", None)

    pf_note = ""
    if "ppf_surf" in new_vals:
        # Keep the circulation-direction radio consistent with the conduit
        # the live reading came from (tubing = forward circ). Sidebar already
        # rendered, so: logical key + pop widget key (rerun below re-inits it).
        if pf_source in ("annulus", "tubing"):
            st.session_state["jpump_direction"] = (
                "Reverse" if pf_source == "annulus" else "Forward"
            )
            st.session_state.pop("jpump_direction_input", None)
        st.session_state["_pf_live_info"] = {
            "well": well_name,
            "kind": "test day",
            "pf_press": float(new_vals["ppf_surf"]),
            "pf_source": pf_source,
            "pf_date": pf_date,
        }
        pf_note = f" · PF {new_vals['ppf_surf']:,} psi (test-day)"

    anchor_human = {
        "recent": "most-recent-test",
        "median": "median-test",
        "specific": "selected-test",
    }.get(anchor_mode, anchor_mode)
    st.session_state["_ipr_sync_msg"] = (
        f"Seeded the {anchor_human} IPR + fluid into the sidebar "
        f"(qwf {new_vals['qwf']:,} BOPD · pwf {new_vals['pwf']:,} psi · "
        f"ResP {new_vals['res_pres']:,} psi · WC {new_vals['form_wc']:.2f} · "
        f"GOR {new_vals['form_gor']:,}{pf_note}). Batch Run, PF Range, and "
        "the Solver now use this — edit any field in the sidebar to override."
    )
    st.rerun()


_WEAK_IPR_R2 = 0.2


def _render_weak_ipr_warning(params: SimulationParams, coeff_row, test_df) -> None:
    """Warn when the Vogel fit is too weakly determined to trust its ResP.

    Trigger: fit R² below ``_WEAK_IPR_R2`` (negative = worse than a flat
    mean — typical for flat test clouds where BHP barely moves and the rate
    scatter is allocation noise, so reservoir pressure is unidentifiable
    from tests alone). The fit is still shown and seeded — this deliberately
    does NOT substitute another value; it hands the decision to the
    engineer, with the characterized Databricks pressure as a reference.
    """
    import pandas as pd

    r2 = coeff_row.get("R2") if hasattr(coeff_row, "get") else None
    if r2 is None or pd.isna(r2) or float(r2) >= _WEAK_IPR_R2:
        return

    span_txt = ""
    if test_df is not None and "BHP" in test_df.columns:
        bhp = pd.to_numeric(test_df["BHP"], errors="coerce").dropna()
        if len(bhp) >= 2:
            span_txt = (
                f" The {len(bhp)} tests span only "
                f"**{bhp.max() - bhp.min():,.0f} psi of BHP**, so the cloud "
                "is nearly flat and reservoir pressure is not identifiable "
                "from tests alone."
            )

    # Deliberately NO reference value here: vw_prop_resvr.resvr_press was
    # itself populated from earlier fit output (circular — it has no more
    # basis in reality than the fit being warned about), and quoting any
    # number next to "decide for yourself" would anchor the decision.
    st.warning(
        f"⚠️ **Weak IPR fit** (R² = {float(r2):.2f}) — the fitted reservoir "
        f"pressure of **{float(coeff_row['ResP']):,.0f} psi** is a "
        f"best-effort estimate, not a measurement.{span_txt} "
        "Decide a reservoir pressure from real evidence (build-ups, "
        "shut-in gauge data, offset wells) and set **Reservoir Pressure** "
        "in the sidebar — your value persists for the session and drives "
        "the Solver, Batch Run, and PF Range."
    )


def _render_ipr_anchor_and_seed(params: SimulationParams, test_df):
    """Render the IPR-anchor test selector at the TOP of the Solver tab, fit the
    chosen test's Vogel curve, and seed the sidebar from it — BEFORE the solve.

    Hoisted out of Model-vs-Actual so the engineer picks the *driving test*
    right where the solve happens. The most-recent test is no longer a dead end:
    if it's watered-out or otherwise bad, pick an earlier oil-bearing test here
    and the sidebar (qwf/pwf/ResP/WC/GOR) reseeds via
    :func:`_sync_chosen_ipr_to_sidebar` and the solver re-runs against it.

    Mirrors the single-source-of-truth design — this only *seeds* the sidebar;
    the sidebar stays the truth the solve, Batch Run, PF Range, and
    Model-vs-Actual all read.

    The caller gates this on ``anchor_selector_shown`` (JP history loaded, oil
    workflow, non-Custom, 2+ tests), so by the time we're here the anchor
    control is appropriate to show.

    Returns ``(coeff_row, vogel_coeffs, merged_with_rp)`` so Model-vs-Actual can
    reuse the same fit for its chart without recomputing. ``coeff_row`` /
    ``vogel_coeffs`` are ``None`` only when the Vogel fit genuinely fails (e.g.
    every test row lacks BHP) — Model-vs-Actual then falls back to its
    synthetic-IPR path, exactly as before.
    """
    import pandas as pd

    anchor_mode, anchor_date = _render_ipr_anchor_control(params.selected_well, test_df)

    # One-shot confirmation from the prior run's sidebar seed. The seed reruns
    # the app, so the note replays on the following render — shown here at the
    # top, right next to the selector that triggered it.
    _ipr_sync_msg = st.session_state.pop("_ipr_sync_msg", None)
    if _ipr_sync_msg:
        st.success(_ipr_sync_msg, icon="✅")

    coeff_row = None
    vogel_coeffs = None
    merged_with_rp = None

    if anchor_mode in ("median", "specific"):
        # Anchored fit: hold the chosen test as the Vogel anchor and re-fit
        # reservoir pressure for the best fit through it (GUI-layer helper, no
        # upstream-library change).
        from woffl.gui.ipr_anchor import compute_anchored_vogel

        field_max_rp = 3000 if str(params.field_model).lower() == "kuparuk" else 1800
        anchored = compute_anchored_vogel(
            test_df,
            well_name=params.selected_well,
            anchor_mode=anchor_mode,
            anchor_date=anchor_date,
            field_max_rp=field_max_rp,
        )
        if anchored is not None:
            coeff_row = pd.Series(anchored)
            vogel_coeffs = pd.DataFrame([anchored])
            merged_with_rp = test_df
            st.caption(
                f"IPR anchored on **{anchored['anchor_label']}** · reservoir "
                f"pressure re-fit to **{anchored['ResP']:,} psi** for best "
                f"fit through that test (R² = {anchored['R2']})."
            )
        else:
            st.warning(
                "Anchored IPR fit unavailable (no test with both BHP and "
                "rate); falling back to the most-recent global fit."
            )

    if coeff_row is None:
        # Most-recent / global least-squares fit (unchanged library path).
        from woffl.assembly.ipr_analyzer import (
            compute_vogel_coefficients,
            estimate_reservoir_pressure,
        )

        try:
            merged_with_rp = estimate_reservoir_pressure(test_df)
            vogel_coeffs = compute_vogel_coefficients(merged_with_rp)
        except Exception as e:
            st.warning(
                f"Vogel IPR fit failed ({e}); falling back to a single-point "
                "IPR with sidebar reservoir pressure."
            )
            vogel_coeffs = None

        # Vogel may return an empty DataFrame when every test row for this well
        # is missing BHP (e.g. S-pad wells whose recent tests have no coincident
        # gauge). Fall through to the no-IPR path in MvA in that case.
        if (
            vogel_coeffs is not None
            and not vogel_coeffs.empty
            and "Well" in vogel_coeffs.columns
        ):
            well_coeffs = vogel_coeffs[vogel_coeffs["Well"] == params.selected_well]
            if not well_coeffs.empty:
                coeff_row = well_coeffs.iloc[0]

    # Seed the chosen test's IPR + fluid into the sidebar so Batch Run, PF Range,
    # and the Solver all use it (the engineer can then override any field in the
    # sidebar — see _sync_chosen_ipr_to_sidebar). No-op unless the anchor-test
    # selection actually changed; when it did, this reruns the app so the picker
    # + solve below run against the freshly-seeded sidebar values.
    if coeff_row is not None:
        pf_press, pf_source, pf_date = _anchor_test_pf(
            test_df, anchor_mode, anchor_date
        )
        _sync_chosen_ipr_to_sidebar(
            params.selected_well,
            anchor_mode=anchor_mode,
            anchor_date=anchor_date,
            qwf_oil=float(coeff_row["qwf"]) * (1 - float(coeff_row["form_wc"])),
            pwf=float(coeff_row["pwf"]),
            res_p=float(coeff_row["ResP"]),
            form_wc=float(coeff_row["form_wc"]),
            fgor=float(coeff_row["fgor"]),
            pf_press=pf_press,
            pf_source=pf_source,
            pf_date=pf_date,
        )

    return coeff_row, vogel_coeffs, merged_with_rp


def _render_manual_test_entry(well_name: str) -> None:
    """Expander to add provisional well tests not yet in Databricks.

    Stored per-well in ``st.session_state['sw_manual_tests'][well]`` for the
    session only (cleared on browser refresh / Streamlit restart). Injected
    into the test set by :func:`woffl.gui.utils.get_well_tests_for_well`, so a
    manual test flows into the IPR fit, the comparison picker, the anchor list,
    and the test table without any other call site changing.
    """
    from datetime import date

    import pandas as pd

    store = st.session_state.setdefault("sw_manual_tests", {})
    existing = store.get(well_name, [])

    label = "Add a provisional test (not yet in the system)"
    if existing:
        label += f" — {len(existing)} added this session"
    with st.expander(label, expanded=False):
        st.caption(
            "Enter a well test that isn't in Databricks yet. It's kept for this "
            "session only and feeds the IPR fit, comparison picker, anchor list, "
            "and test table for this well. Leave a field at 0 to mark it unknown."
        )
        with st.form(f"_manual_test_form_{well_name}", clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                t_date = st.date_input("Test date", value=date.today())
                oil = st.number_input("Oil (BOPD)", min_value=0.0, step=10.0)
                water = st.number_input("Water (BWPD)", min_value=0.0, step=10.0)
            with c2:
                bhp = st.number_input("BHP (psi)", min_value=0.0, step=10.0)
                whp = st.number_input("Surface / WHP (psi)", min_value=0.0, step=10.0)
            with c3:
                gor = st.number_input("GOR (scf/bbl)", min_value=0.0, step=10.0)
                pf = st.number_input(
                    "PF rate / lift water (BWPD)", min_value=0.0, step=10.0
                )
            submitted = st.form_submit_button("Add test")

        if submitted:
            row = {
                "well": well_name,
                "WtDate": pd.Timestamp(t_date),
                "WtOilVol": float(oil),
                "WtWaterVol": float(water),
                "WtTotalFluid": float(oil) + float(water),
                "BHP": float(bhp) if bhp > 0 else float("nan"),
                "fgor": float(gor) if gor > 0 else float("nan"),
                "lift_wat": float(pf) if pf > 0 else float("nan"),
                "whp": float(whp) if whp > 0 else float("nan"),
            }
            store.setdefault(well_name, []).append(row)
            st.success(
                f"Added provisional test "
                f"{pd.Timestamp(t_date).strftime('%Y-%m-%d')} for {well_name}."
            )
            st.rerun()

        if existing:
            st.markdown("**Provisional tests this session:**")
            for i, r in enumerate(existing):
                d = r.get("WtDate")
                ds = pd.Timestamp(d).strftime("%Y-%m-%d") if d is not None else "N/A"
                bhp_str = (
                    f"{float(r.get('BHP')):,.0f}"
                    if is_valid_number(r.get("BHP"))
                    else "n/a"
                )
                cols = st.columns([5, 1])
                with cols[0]:
                    st.caption(
                        f"{ds} · Oil {r.get('WtOilVol', 0):,.0f} · "
                        f"Water {r.get('WtWaterVol', 0):,.0f} · BHP {bhp_str}"
                    )
                with cols[1]:
                    if st.button("Remove", key=f"_mt_rm_{well_name}_{i}"):
                        store[well_name].pop(i)
                        if not store[well_name]:
                            store.pop(well_name, None)
                        st.rerun()


def _render_model_vs_actual(
    params: SimulationParams,
    wellbore,
    well_profile,
    *,
    selected_test_row=None,
    anchor_fit=(None, None, None),
    ipr_chart_container=None,
) -> None:
    """Render the Model vs Actual comparison section.

    Only shown when JP history is uploaded and a non-Custom well is selected.
    Renders the IPR chart for any test count (0 / 1 / 2+); the comparison
    table + calibration sections require at least 1 test.

    ``selected_test_row`` (from the Solver tab's test picker, defaults to
    the most recent test) drives two things:
      * the pump used to model the comparison — :func:`_pump_at_test_date`
        looks up the install on or before the selected test's date.
      * the row that gets compared — the metrics show modeled vs **that
        test's** Oil / BHP / PF, not the most recent test.

    ``anchor_fit`` is the ``(coeff_row, vogel_coeffs, merged_with_rp)`` tuple
    already computed + seeded at the TOP of the tab by
    :func:`_render_ipr_anchor_and_seed`. We reuse it here for the IPR chart +
    comparison rather than re-rendering the anchor control or re-running the
    Vogel fit. It's ``(None, None, None)`` for the 0/1-test or Vogel-failed
    cases, which fall through to the synthetic-IPR path below.

    ``ipr_chart_container``: optional container the IPR-chart block draws
    into. The pad-review page passes the same box as the IPR-anchor pickers
    (picker + curve belong together); ``None`` renders inline here.
    """
    jp_hist = st.session_state.get("jp_history_df")
    if jp_hist is None or params.selected_well == "Custom":
        return

    import pandas as pd

    from woffl.assembly.ipr_analyzer import generate_ipr_curves
    from woffl.assembly.jp_history import get_current_pump
    from woffl.flow.inflow import InFlow
    from woffl.gui.ipr_viz import create_ipr_plotly
    from woffl.gui.utils import create_inflow, create_jetpump, create_reservoir_mix

    st.divider()

    # Pump-at-test-date when a test is selected; fall back to current pump
    # otherwise (preserves existing behaviour when there's no test data).
    test_date = None
    if selected_test_row is not None:
        td = selected_test_row.get("WtDate")
        if pd.notna(td):
            test_date = td

    pump_info = None
    if test_date is not None:
        pump_info = _pump_at_test_date(jp_hist, params.selected_well, test_date)
    if pump_info is None:
        pump_info = get_current_pump(jp_hist, params.selected_well)

    # Pump to MODEL: prefer the install at the test's date. But a well with NO JP
    # history (S-67) or a corrupt / forward-circulating record (S-17, 'D' in the
    # nozzle column) must STILL show its tests, IPR trend, and BHP — so fall back
    # to the sidebar pump and note it, instead of bailing out of the whole section.
    valid_hist = (
        pump_info is not None
        and pump_info.get("nozzle_no")
        and pump_info.get("throat_ratio")
        and _is_valid_pump_code(pump_info["nozzle_no"], pump_info["throat_ratio"])
    )
    if valid_hist:
        nozzle = pump_info["nozzle_no"]
        throat = pump_info["throat_ratio"]
        install_date_str = (
            pump_info["date_set"].strftime("%Y-%m-%d")
            if pump_info.get("date_set") is not None
            else "N/A"
        )
    else:
        if pump_info is None:
            reason = "no JP history"
        else:
            reason = (
                f"an unrecognized pump record "
                f"(`{pump_info.get('nozzle_no')}{pump_info.get('throat_ratio')}`)"
            )
        st.info(
            f"**{params.selected_well}** has {reason} — modeling the sidebar pump "
            f"**{params.nozzle_no}{params.area_ratio}** against the tests "
            "(tests, IPR trend, and BHP still shown)."
        )
        nozzle, throat = params.nozzle_no, params.area_ratio
        install_date_str = "N/A"

    # 2. Get well tests from pre-fetched cache (includes any session-only
    # manual/provisional tests injected by get_well_tests_for_well).
    test_df = _get_well_tests(params.selected_well)
    n_tests = 0 if test_df is None or test_df.empty else len(test_df)

    # Provisional-test entry now renders at the TOP of the IPR group (with the
    # anchor selector + chart), not here — see render_tab's _anchor_box.

    # IPR anchor selection, Vogel fit, and the sidebar seed all happen at the TOP
    # of the tab now (see _render_ipr_anchor_and_seed), before the solve, so a
    # bad most-recent test can't dead-end the solver. Reuse that fit here for the
    # chart + comparison rather than re-rendering the control or recomputing.
    # coeff_row / vogel_coeffs / merged_with_rp are None for the 0/1-test or
    # Vogel-failed cases, which fall through to the synthetic-IPR path below
    # exactly as before.
    coeff_row, vogel_coeffs, merged_with_rp = anchor_fit

    # 4. Resolve the comparison test row + its surface pressure for the chart
    # and the modeled-vs-actual solver below. WC / GOR / ResP now come from the
    # sidebar (seeded from the anchor test above, editable to override), so
    # there are no per-field override widgets here anymore.
    recent_test = None
    test_whp: float | None = None
    if n_tests >= 1:
        if selected_test_row is not None:
            recent_test = selected_test_row
        else:
            recent_test = test_df.sort_values("WtDate", ascending=False).iloc[0]
        _w = recent_test.get("whp", None)
        test_whp = float(_w) if is_valid_number(_w) else None

    # The chosen anchor test seeded WC / GOR / Reservoir Pressure into the
    # sidebar (see _sync_chosen_ipr_to_sidebar). The sidebar is the single
    # source of truth from here on: this section, Batch Run, PF Range, and the
    # top Solver all read it, so they never disagree. The engineer overrides any
    # value by editing the sidebar — the edit persists for the session until a
    # different anchor test is picked.
    model_res_p = float(params.pres)
    model_gor = int(params.form_gor)

    # The whole IPR-chart block (seeded caption, JP-label toggle, curve,
    # weak-fit warning, rate calculator) draws into ``ipr_chart_container``
    # when the host page provides one — the pad-review page passes the same
    # box as the IPR-anchor pickers so the curve sits right next to the
    # control that moves it. ``None`` (Single-Well) renders inline here,
    # unchanged. Only display placement moves; the variables assigned in this
    # block stay visible to the comparison model below (with-blocks don't
    # scope).
    _ipr_box = (
        ipr_chart_container if ipr_chart_container is not None else st.container()
    )
    with _ipr_box:
        if coeff_row is not None:
            st.caption(
                f"IPR + fluid seeded from the anchor test into the sidebar: "
                f"Reservoir Pressure **{model_res_p:.0f}** psi · "
                f"GOR **{model_gor:,}** scf/bbl · WC **{float(params.form_wc):.2f}**. "
                "Edit any of them in the sidebar to override (persists for the session)."
            )
            # Draw the curve at the sidebar ResP (seeded from the anchor test,
            # overridable in the sidebar) so the chart matches the solver + Batch.
            vogel_coeffs_plot = vogel_coeffs.copy()
            vogel_coeffs_plot.loc[
                vogel_coeffs_plot["Well"] == params.selected_well, "ResP"
            ] = model_res_p
            ipr_data = generate_ipr_curves(vogel_coeffs_plot)
            plot_points = merged_with_rp
        else:
            # 0-test, 1-test, or Vogel-failed path.
            model_res_p = float(params.pres)

            anchor_src = None
            if n_tests >= 1:
                # Anchor on the SELECTED test (picker default = most recent), so
                # an older-test selection keeps the IPR-anchor + comparison row
                # in sync with what the user picked.
                if selected_test_row is not None:
                    anchor_row = selected_test_row
                else:
                    anchor_row = test_df.sort_values("WtDate", ascending=False).iloc[0]
                total = anchor_row.get("WtTotalFluid")
                bhp = anchor_row.get("BHP")
                if is_valid_number(total) and is_valid_number(bhp):
                    anchor_qwf = float(total)
                    anchor_pwf = float(bhp)
                    anchor_src = "well test"

            if anchor_src is None:
                # 0-test fallback (or 1-test missing total/BHP). Sidebar qwf is the
                # OIL rate; convert to total liquid for the chart's x-axis (BPD)
                # using sidebar form_wc. Falls back to oil rate when WC ≈ 1.
                wc = float(params.form_wc)
                if 0.0 <= wc < 1.0:
                    anchor_qwf = float(params.qwf) / max(1e-6, 1.0 - wc)
                else:
                    anchor_qwf = float(params.qwf)
                anchor_pwf = float(params.pwf)
                anchor_src = "sidebar"

            # Guard against degenerate sidebar values (pwf >= ResP would make
            # Vogel divide by zero / go negative; non-positive flow is meaningless).
            ipr_data = {}
            if anchor_qwf > 0 and 0 <= anchor_pwf < model_res_p and model_res_p > 0:
                try:
                    synth_qmax = InFlow.vogel_qmax(anchor_qwf, anchor_pwf, model_res_p)
                    synth_coeffs = pd.DataFrame(
                        [
                            {
                                "Well": params.selected_well,
                                "ResP": model_res_p,
                                "qwf": anchor_qwf,
                                "pwf": anchor_pwf,
                                "QMax_recent": synth_qmax,
                            }
                        ]
                    )
                    ipr_data = generate_ipr_curves(synth_coeffs)
                except Exception as e:
                    st.warning(
                        f"Could not build IPR curve from anchor "
                        f"(qwf={anchor_qwf:.0f}, pwf={anchor_pwf:.0f}, "
                        f"ResP={model_res_p:.0f}): {e}"
                    )
            plot_points = test_df if test_df is not None else pd.DataFrame()

            st.caption(
                f"Using Reservoir Pressure: **{model_res_p:.0f}** psi (sidebar) · "
                f"IPR anchor: {anchor_src} (qwf={anchor_qwf:.0f} BPD, "
                f"pwf={anchor_pwf:.0f} psi)"
            )
            if n_tests == 0:
                st.info(
                    f"No well tests for {params.selected_well} — chart shows the "
                    "IPR curve from sidebar values only. No actuals available, "
                    "so the modeled-vs-actual comparison and calibration are "
                    "unavailable for this well."
                )
            elif n_tests == 1:
                st.info(
                    f"Only 1 well test for {params.selected_well} — Vogel fit "
                    "needs 2+ tests, so the chart anchors on this single point + "
                    "sidebar reservoir pressure."
                )

        # JP-label toggle for the IPR scatter — sits just above the chart so
        # the checkbox is next to the picture it affects. Default off (plain
        # colored dots; the colorbar still encodes Days Ago).
        show_jp_labels = st.checkbox(
            "Show JP label inside each test point",
            value=False,
            key=f"mva_show_jp_labels_{params.selected_well}",
            help=(
                "Replace each test point's dot with the pump installed at that "
                'test\'s date (e.g. "12B"), drawn inside an enlarged colored '
                "marker. Useful for seeing pump changes alongside the IPR shape."
            ),
        )

        # 5. Always render the IPR chart (Vogel-fit or synthetic).
        if params.selected_well in ipr_data:
            plot_data = plot_points if plot_points is not None else pd.DataFrame()
            fig = create_ipr_plotly(
                params.selected_well,
                ipr_data[params.selected_well],
                plot_data,
                form_wc=params.form_wc,
                jp_history=jp_hist,
                show_jp_labels=show_jp_labels,
            )
            st.plotly_chart(fig, use_container_width=True)
            # Weak-fit banner directly under the IPR chart it's judging: the fit
            # is ALWAYS produced and seeded (never silently replaced), but when
            # the test cloud can't constrain reservoir pressure the engineer is
            # told to decide one themselves.
            _af_coeff_row, _, _af_tests = anchor_fit
            if _af_coeff_row is not None:
                _render_weak_ipr_warning(params, _af_coeff_row, _af_tests)
            _render_ipr_rate_calculator(params, model_res_p)

    # 6. Modeled vs Actual + calibration sections need at least 1 test.
    if n_tests == 0:
        return

    # 7. Run the comparison model. The IPR + fluid come from the SIDEBAR (which
    # the anchor test seeded above and the engineer may have overridden), so the
    # Model-vs-Actual solver, Batch Run, PF Range, and the top Solver all use
    # the same operating point. `model_res_p` / `model_gor` were resolved from
    # the sidebar above; `recent_test` / `test_whp` come from the selected test.
    model_surf_pres = test_whp if test_whp is not None else params.surf_pres
    st.caption(
        f"Using Surface Pressure: **{model_surf_pres:.0f}** psi ({'well test' if test_whp is not None else 'sidebar'})"
    )

    if coeff_row is not None:
        # Sidebar is the single source of truth (seeded from the anchor test;
        # editable). qwf is the OIL rate, matching create_inflow's contract.
        oil_qwf = float(params.qwf)
        pwf_for_inflow = float(params.pwf)
        wc_for_resmix = float(params.form_wc)
    else:
        # Single-test fallback — same logic as build_calibration_inputs.
        actual_oil_anchor = recent_test.get("WtOilVol")
        actual_bhp_anchor = recent_test.get("BHP")
        if is_valid_number(actual_oil_anchor) and is_valid_number(actual_bhp_anchor):
            oil_qwf = float(actual_oil_anchor)
            pwf_for_inflow = float(actual_bhp_anchor)
        else:
            oil_qwf = float(params.qwf)
            pwf_for_inflow = float(params.pwf)
        wc_for_resmix = float(params.form_wc)

    ipr_inflow = create_inflow(oil_qwf, pwf_for_inflow, model_res_p)
    ipr_res_mix = create_reservoir_mix(
        wc_for_resmix, model_gor, params.form_temp, params.field_model
    )
    current_jp = create_jetpump(nozzle, throat, params.ken, params.kth, params.kdi)

    # The MvA solve models the TEST's pump, which can hit a throat-entry
    # no-solution even when the top-of-tab solve (sidebar pump) succeeded —
    # guard it so a marginal well doesn't crash the whole page.
    try:
        model_results = run_jetpump_solver(
            model_surf_pres,
            params.form_temp,
            params.rho_pf,
            params.ppf_surf,
            current_jp,
            wellbore,
            well_profile,
            ipr_inflow,
            ipr_res_mix,
            field_model=params.field_model,
            jpump_direction=params.jpump_direction,
        )
    except IndexError:
        st.warning(
            f"Model vs Actual unavailable — the test's pump ({nozzle}{throat}) "
            "has no valid throat-entry solution at this operating point "
            "(GOR or suction pressure may be too low)."
        )
        model_results = None

    # 6. Display comparison metrics
    actual_oil = recent_test.get("WtOilVol", None)
    actual_bhp = recent_test.get("BHP", None)
    actual_pf = recent_test.get("lift_wat", None)
    actual_whp = recent_test.get("whp", None)

    # Heading reflects the SELECTED test's date (test picker default = most
    # recent). Falls back to a generic label if the test row has no WtDate.
    if recent_test is not None and pd.notna(recent_test.get("WtDate")):
        _test_date_label = f"{recent_test.get('WtDate').strftime('%Y-%m-%d')} Test"
    else:
        _test_date_label = "Most Recent Test"
    st.markdown(f"#### Modeled vs Actual ({_test_date_label})")

    # Pump-at-test-date callout sits BELOW the dynamic heading so the user
    # reads "Modeled vs Actual (2026-05-10)" → "uses 13C installed at that
    # date" → the actual comparison metrics. (Was previously rendered at the
    # very top of the MvA section, before the IPR chart.)
    if test_date is not None:
        st.info(
            f"Model vs Actual uses the pump installed at the time of the "
            f"**selected test ({test_date.strftime('%Y-%m-%d')})**: "
            f"Nozzle {nozzle}, Throat {throat} (set {install_date_str})."
        )
    else:
        st.info(
            f"Model vs Actual uses the CURRENT INSTALLED pump: "
            f"Nozzle {nozzle}, Throat {throat} (set {install_date_str})."
        )

    if model_results:
        _psu, _sonic, modeled_oil, _fwat, modeled_pf, _mach = model_results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Modeled Oil Rate", f"{modeled_oil:.0f} BOPD")
        with col2:
            st.metric(
                "Actual Oil Rate",
                f"{actual_oil:.0f} BOPD" if is_valid_number(actual_oil) else "N/A",
            )
        with col3:
            if is_valid_number(actual_oil):
                st.metric("Delta", f"{modeled_oil - actual_oil:+.0f} BOPD")
            else:
                st.metric("Delta", "N/A")

        if is_valid_number(actual_bhp):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Modeled BHP (suction)", f"{_psu:.0f} psi")
            with col2:
                st.metric("Actual BHP", f"{actual_bhp:.0f} psi")
            with col3:
                st.metric("Delta", f"{_psu - actual_bhp:+.0f} psi")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Modeled PF Rate", f"{modeled_pf:.0f} BWPD")
        with col2:
            st.metric(
                "Actual PF Rate",
                f"{actual_pf:.0f} BWPD" if is_valid_number(actual_pf) else "N/A",
            )
        with col3:
            if is_valid_number(actual_pf):
                st.metric("Delta", f"{modeled_pf - actual_pf:+.0f} BWPD")
            else:
                st.metric("Delta", "N/A")

        if is_valid_number(actual_pf) and abs(modeled_pf - actual_pf) > 100:
            st.warning(
                f"Modeled PF rate differs from actual by {abs(modeled_pf - actual_pf):.0f} BWPD. "
                "Check that the **Power Fluid Pressure** in the sidebar matches the actual PF pressure for this well."
            )

        if is_valid_number(actual_whp):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Test Surface Pressure", f"{actual_whp:.0f} psi")
            with col2:
                st.metric("Sidebar Surface Pressure", f"{params.surf_pres} psi")

        # Compute calibration factor
        if is_valid_number(actual_oil) and modeled_oil > 0:
            from woffl.assembly.calibration import CalibrationResult

            raw_factor = actual_oil / modeled_oil
            factor = max(0.3, min(2.0, raw_factor))
            cal_result = CalibrationResult(
                well_name=params.selected_well,
                current_nozzle=nozzle,
                current_throat=throat,
                model_oil=modeled_oil,
                actual_oil=actual_oil,
                model_pf=modeled_pf,
                actual_pf=actual_pf if is_valid_number(actual_pf) else None,
                model_bhp=_psu,
                actual_bhp=actual_bhp if is_valid_number(actual_bhp) else None,
                calibration_factor=factor,
            )
            st.session_state["sw_calibration_result"] = cal_result

            st.markdown("#### Model Calibration")
            grade = cal_result.quality_grade
            grade_color = {"good": "green", "fair": "orange", "poor": "red"}[grade]
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Calibration Factor", f"{cal_result.calibration_factor:.3f}")
            with c2:
                st.metric("Oil Error", f"{cal_result.oil_error_pct:.1f}%")
            with c3:
                st.markdown(f"**Quality:** :{grade_color}[{grade.upper()}]")

            # Rate-scalar apply toggle lives next to the metric it's derived
            # from. Toggling it scales the modeled rates everywhere else by
            # this factor (display-only — does not change BHP or PF rate).
            # Persist via a non-widget shadow so a Solver->Batch->Solver detour
            # (segmented_control GC drops this view's widget state) doesn't snap
            # the toggle back off and silently un-apply the rate scalar.
            apply_cal = st.checkbox(
                f"Apply rate-scalar calibration (×{cal_result.calibration_factor:.2f})",
                value=st.session_state.get("sw_apply_calibration_persist", False),
                key="sw_apply_calibration",
                help=(
                    "After the solver runs, scales modeled oil and water by a "
                    "constant so modeled oil matches actual oil from the most "
                    "recent test. Display-only — does not change BHP or PF "
                    "rate. Stacking this on top of the BHP friction "
                    "calibration double-corrects; prefer one or the other."
                ),
            )
            st.session_state["sw_apply_calibration_persist"] = apply_cal

            if apply_cal:
                cal_oil = cal_result.model_oil * cal_result.calibration_factor
                oil_delta_vs_actual = cal_oil - cal_result.actual_oil
                cc1, cc2 = st.columns(2)
                with cc1:
                    st.metric(
                        "Calibrated Oil",
                        f"{cal_oil:,.0f} BOPD",
                        delta=f"{oil_delta_vs_actual:+.0f} vs actual",
                        delta_color="off",
                        help="Modeled oil × calibration factor.",
                    )
                with cc2:
                    st.metric(
                        "Actual Oil",
                        f"{cal_result.actual_oil:,.0f} BOPD",
                    )

                if (
                    params.nozzle_no != cal_result.current_nozzle
                    or params.area_ratio != cal_result.current_throat
                ):
                    st.caption(
                        f"Factor derived from installed pump "
                        f"({cal_result.current_nozzle}{cal_result.current_throat}) "
                        "— applying to a different pump is an approximation."
                    )
        else:
            st.session_state.pop("sw_calibration_result", None)
            st.session_state.pop("sw_apply_calibration", None)
            st.session_state.pop("sw_apply_calibration_persist", None)

        # Friction-coef calibration (BHP-target). Stored in session_state for
        # the top jetpump-solver panel to apply via checkbox.
        _render_fric_calibration_section(
            params=params,
            wellbore=wellbore,
            well_profile=well_profile,
            ipr_inflow=ipr_inflow,
            ipr_res_mix=ipr_res_mix,
            nozzle=nozzle,
            throat=throat,
            actual_bhp=actual_bhp,
            model_surf_pres=model_surf_pres,
        )
    else:
        st.session_state.pop("sw_calibration_result", None)
        st.warning("Model could not solve with the current JP and IPR-derived inflow.")

    # 7. Show well test data table
    import pandas as pd

    display_cols = {
        "WtDate": "Test Date",
        "WtOilVol": "Oil (BOPD)",
        "WtWaterVol": "Water (BWPD)",
        "WtTotalFluid": "Total Fluid (BPD)",
        "BHP": "BHP (psi)",
        "fgor": "GOR (scf/bbl)",
        "lift_wat": "PF Rate (BWPD)",
        "whp": "Surface Pres (psi)",
    }
    available = [c for c in display_cols if c in test_df.columns]
    table_df = test_df[available].copy()
    table_df = table_df.rename(columns=display_cols)
    if "Test Date" in table_df.columns:
        table_df["Test Date"] = pd.to_datetime(table_df["Test Date"]).dt.strftime(
            "%Y-%m-%d"
        )
    table_df = table_df.sort_values("Test Date", ascending=False).reset_index(drop=True)

    st.markdown(f"#### Well Test Data ({len(table_df)} tests)")
    st.dataframe(table_df, use_container_width=True, hide_index=True)
