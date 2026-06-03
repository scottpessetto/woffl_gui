"""Tab 1: Jetpump Solver Results

Renders the single-pump solution display showing suction pressure,
oil rate, water rate, power fluid rate, and sonic status.

When JP history is uploaded and a non-Custom well is selected,
also shows a "Model vs Actual" comparison section with IPR chart
and modeled vs actual metrics.
"""

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
        is_disregarding_databricks_bhp,
        parse_xlsx,
        remove_file_from_gauge,
        set_disregard_databricks_bhp,
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
                    if well_tests_df is not None else None
                )
                file_count = len(gauge.files)
                file_str = (
                    f" ({file_count} file{'s' if file_count != 1 else ''})"
                )
                date_range = (
                    f"{gauge.start_date.strftime('%Y-%m-%d')} → "
                    f"{gauge.end_date.strftime('%Y-%m-%d')}"
                )
                cov_str = (
                    f" · {cov['tests_matched']}/{cov['tests_total']} tests matched"
                    if cov else ""
                )
                disregard_str = (
                    " · **Databricks BHP disregarded**" if disregard else ""
                )
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

    expander_label = (
        f"Memory Gauge Data — {len(gauge.files)} file"
        f"{'s' if len(gauge.files) != 1 else ''} loaded"
        if gauge is not None
        else "Memory Gauge Data — upload BHP for wells without a permanent gauge"
    )
    with st.expander(expander_label, expanded=False):
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
        # incrementing the counter after a successful Add (Streamlit has
        # no API to programmatically clear a file_uploader).
        upload_counter = st.session_state.get(
            f"_mg_upload_counter_{well_name}", 0
        )
        upload_label = (
            "Upload another gauge file"
            if gauge is not None
            else "Memory gauge XLSX"
        )
        uploaded = st.file_uploader(
            upload_label,
            type=["xlsx"],
            key=f"mg_upload_{well_name}_{upload_counter}",
        )
        if uploaded is None:
            return

        # Parse on every render while a file is in the uploader. Cheap
        # (~100 ms for ~9k rows). Each parse returns a single MemoryGaugeFile;
        # add_file_to_gauge combines it with any already-loaded files.
        try:
            preview = parse_xlsx(uploaded.getvalue(), uploaded.name)
        except Exception as e:
            st.error(f"Could not parse memory gauge file: {e}")
            return

        # Warn on duplicate filename — Streamlit's uploader can keep the
        # same file across reruns, and silently double-adding would inflate
        # the sample count. Filename match is good enough for now.
        if gauge is not None and any(
            f.source_filename == preview.source_filename for f in gauge.files
        ):
            st.warning(
                f"`{preview.source_filename}` is already loaded — "
                "click Remove on it above first if you want to replace it."
            )
            return

        c1, c2 = st.columns(2)
        c1.metric("Samples", f"{preview.sample_count:,}")
        pr_min = float(preview.raw_df["pressure"].min())
        pr_max = float(preview.raw_df["pressure"].max())
        c2.metric("Pressure range", f"{pr_min:,.0f} – {pr_max:,.0f} psi")
        st.caption(
            f"Coverage: **{preview.start_date.strftime('%Y-%m-%d')}** → "
            f"**{preview.end_date.strftime('%Y-%m-%d')}**"
        )

        button_label = (
            f"Add file to {well_name} gauge"
            if gauge is not None
            else f"Apply gauge to {well_name}"
        )
        if st.button(
            button_label,
            type="primary",
            key=f"mg_apply_btn_{well_name}",
            use_container_width=True,
        ):
            # Combine the new file into the well's gauge (creates one if
            # none exists). The resulting MemoryGaugeData carries the
            # union daily-median series across all files.
            new_gauge = add_file_to_gauge(well_name, preview)

            # Network fetches keyed off the COMBINED window — that way
            # extended tests + divergence checks cover the full coverage
            # span, not just the newly-added file's window.
            with st.spinner(
                f"Fetching well tests + Databricks BHP "
                f"{new_gauge.start_date.strftime('%Y-%m-%d')} → today…"
            ):
                ext_df = fetch_extended_tests(well_name, new_gauge.start_date)
                db_bhp_df = fetch_databricks_bhp_daily(
                    well_name, new_gauge.start_date, new_gauge.end_date,
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

            # Auto-divergence vs Databricks across the combined window.
            # Each Add re-checks because the wider gauge can shift the
            # comparison. Auto-set ON only; once on, the user owns it.
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

            # Reset the uploader by bumping its key counter — the next
            # render renders a fresh empty file_uploader, ready for the
            # next file.
            st.session_state[f"_mg_upload_counter_{well_name}"] = (
                upload_counter + 1
            )
            st.session_state["_force_ipr_refresh"] = True
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
            f"Modeling **{model_n}{model_t}** \u2014 **differs from the pump in the "
            f"well at the{date_part} test** you're matching to (**{tn}{tt}**). "
            "The vs-actual deltas below compare a different pump against that "
            f"test, so they're greyed out. Set the sidebar pump to **{tn}{tt}** "
            "to compare against this test, or use the Model-vs-Actual section "
            f"below (it models {tn}{tt} for the proper comparison).",
            icon="\u26a0\ufe0f",
        )
        return

    # Modeled pump matches the test's pump \u2192 the deltas are like-for-like.
    if test_pump is not None and test_str:
        st.success(
            f"Modeling **{model_n}{model_t}** \u2014 matches the pump in the well at "
            f"the **{test_str}** test, so the vs-actual deltas below are a "
            "like-for-like comparison."
        )
        return

    # No test pump to compare against (no test selected, or no JP install record
    # at the test's date) \u2014 just report what's being modeled.
    st.info(f"Modeling **{model_n}{model_t}** (sidebar pump).")


def render_tab(
    params: SimulationParams, jetpump, wellbore, well_profile, inflow, res_mix
) -> None:
    """Render the Jetpump Solver Results tab.

    Args:
        params: Simulation parameters from sidebar
        jetpump: JetPump object
        wellbore: PipeInPipe wellbore object
        well_profile: WellProfile object
        inflow: InFlow object
        res_mix: ResMix object
    """
    import pandas as pd

    render_input_summary(params)

    # Memory-gauge upload + status. Lives near the top so the status banner
    # is unmissable when an override is active; the upload itself is in a
    # collapsed expander so it stays out of the way when not needed.
    _render_memory_gauge_section(params.selected_well)

    # Surface any persisted message from a prior auto-recovery (e.g. GOR reset)
    msg = st.session_state.pop("_solver_gor_reset_msg", None)
    if msg:
        st.warning(msg)

    # Test picker FIRST — its selection determines which pump the solver
    # uses on this run (pump-at-test-date overrides sidebar when a test is
    # selected). Default = most recent test; matches the pre-picker
    # behaviour.
    test_df = _get_well_tests(params.selected_well)
    # By default the comparison test is synced to the IPR anchor (chosen in
    # Model vs Actual below). The decouple checkbox there sets sw_ipr_decouple_*;
    # Streamlit commits a widget's value before the rerun, so we can read it here
    # even though the checkbox renders later in the same run.
    _decoupled = st.session_state.get(
        f"sw_ipr_decouple_{params.selected_well}", False
    )
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
            if pump_at_test and pump_at_test.get("nozzle_no") and pump_at_test.get(
                "throat_ratio"
            ):
                test_pump = (
                    pump_at_test["nozzle_no"],
                    pump_at_test["throat_ratio"],
                )

    pump_differs = (
        test_pump is not None
        and (effective_nozzle, effective_throat) != test_pump
    )

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

    with st.spinner("Running jetpump solver..."):
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
            # Throat-entry iteration produced no valid points — typically caused
            # by an unrealistically low GOR for the well's PVT. Reset the sidebar
            # GOR to GOR_AUTO_RECOVERY_VALUE (250) and remember a per-well GOR
            # floor, so the re-solve — and every view, which all read the sidebar
            # GOR now — uses the recovered value.
            #
            # Streamlit forbids writing to a widget's state key after the widget
            # has rendered (and the sidebar already rendered above the tabs).
            # So we set the logical key and DELETE the widget key — the
            # _number_input helper will re-initialize the widget from the
            # logical key on the next run.
            _trigger_gor_reset(
                params.selected_well,
                params.form_gor,
                reason="throat-entry iteration produced no valid points",
            )

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

        h1, h2, h3, h4 = st.columns(4)
        with h1:
            d = _delta(qoil_std, actuals["oil"], "vs actual")
            st.metric(
                _label("Oil Rate", actuals["oil"]), f"{qoil_std:,.0f} BOPD",
                delta=d, delta_color="off" if d is None else _dcolor,
            )
        with h2:
            # Formation Water has no actuals counterpart (we don't track it
            # in actuals dict), so it's always modeled — label accordingly.
            st.metric("Formation Water (modeled)", f"{fwat_bwpd:,.0f} BWPD")
        with h3:
            d = _delta(qnz_bwpd, actuals["pf"], "vs actual")
            st.metric(
                _label("Power Fluid", actuals["pf"]), f"{qnz_bwpd:,.0f} BWPD",
                delta=d, delta_color="off" if d is None else _dcolor,
            )
        with h4:
            d = _delta(psu, actuals["bhp"], "vs actual")
            st.metric(
                _label("Suction Pressure", actuals["bhp"]), f"{psu:,.0f} psig",
                delta=d, delta_color="off" if d is None else _dcolor,
            )

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
            # PF mismatch is the foundational check — if the sidebar PF pressure
            # doesn't match operating conditions, friction calibration will fit
            # nonsense ken values to compensate. Gate calibration on this.
            # Pull the test date so the warning + quickfix can show it.
            cal_inputs = build_calibration_inputs(params, wellbore, well_profile)
            test_date_str = cal_inputs["test_date_str"] if cal_inputs else None
            pf_warning_shown, pf_blocked = render_pf_mismatch_warning(
                qnz_bwpd,
                actuals["pf"],
                params.ppf_surf,
                test_date_str=test_date_str,
                well_name=params.selected_well,
            )
            # Render the quickfix whenever any warning fires (red OR yellow info)
            # so the user always has a one-click path to fine-tune. Calibration
            # gating (cal button disabled) is governed by `pf_blocked` only.
            if pf_warning_shown:
                render_pf_quickfix_widget(
                    params, wellbore, well_profile, target_lift_wat=actuals["pf"]
                )

            # BHP red flag (only meaningful once PF is right — otherwise the BHP
            # delta is mostly explained by the wrong PF, not friction).
            if not pf_blocked:
                warned = render_bhp_calibration_warning(
                    psu, actuals["bhp"], on_solver_view=True
                )
                if not warned and any(v is not None for v in actuals.values()):
                    st.caption(
                        "Deltas compare modeled values to the most recent well test."
                    )

            # Compact calibration action bar — buttons live here so they're
            # visible without scrolling. Run is disabled while PF mismatch is
            # blocking; the Push button stays available so a prior result can
            # still be applied. Also disabled when the SELECTED test (from the
            # picker) has no measured BHP — there's nothing to target. Both
            # ``selected_test_row`` and ``bhp_missing`` reflect the picker so
            # the gating + calibration target stay in sync.
            _render_fric_cal_action_bar(
                params, wellbore, well_profile,
                selected_test_row=selected_test_row,
                pf_blocked=pf_blocked,
                bhp_missing=(actuals["bhp"] is None),
            )

        # Secondary diagnostics
        with st.expander("Throat diagnostics", expanded=False):
            d1, d2 = st.columns(2)
            with d1:
                st.metric("Throat Entry Mach", f"{mach_te:.3f}")
            with d2:
                st.metric("Sonic Flow", "Yes" if sonic_status else "No")
            if sonic_status:
                st.info(
                    "Critical flow conditions — sonic velocity at throat entry."
                )
            else:
                st.caption("Stable subsonic flow at the throat.")

        # Rate-scalar applied banner \u2014 full toggle + calibrated predictions
        # live next to the Calibration Factor metric in Model vs Actual below.
        _cal = st.session_state.get("sw_calibration_result")
        if (
            _cal
            and params.selected_well != "Custom"
            and st.session_state.get("sw_apply_calibration", False)
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

    # Model vs Actual comparison (requires JP history + non-Custom well)
    _render_model_vs_actual(
        params, wellbore, well_profile, selected_test_row=selected_test_row
    )


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
    params: SimulationParams, wellbore, well_profile, *, selected_test_row=None,
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
        params, wellbore, well_profile, selected_test_row=selected_test_row,
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
            st.caption(
                "⚠️ Calibration blocked — fix the PF rate mismatch above first."
            )
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
                params, wellbore, well_profile,
                selected_test_row=selected_test_row,
            )
        if not ok:
            st.error(err)
        else:
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
        if hasattr(result, "starts_tried") else ""
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


def _latest_actuals(well_name: str) -> dict[str, float | None]:
    """Most recent test values — kept for back-compat. New code should use
    :func:`_actuals_from_test` with a specific test row instead, so the test
    picker on the Solver tab can target any test (not just the latest).
    """
    blank = {"oil": None, "bhp": None, "pf": None}
    if well_name == "Custom":
        return blank
    tests = _get_well_tests(well_name)
    if tests is None or tests.empty:
        return blank
    recent = tests.sort_values("WtDate", ascending=False).iloc[0]
    return _actuals_from_test(recent)


def _actuals_from_test(test_row) -> dict[str, float | None]:
    """Extract Oil / BHP / PF actuals from a single well-test row.

    Generalises :func:`_latest_actuals` to any test, not just the most recent.
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
    anchor: reads the last-applied anchor signature (mode, date) and maps it to
    a row in ``sorted_tests`` the same way ipr_anchor._resolve_anchor_row does
    (median → nearest-median-BHP; specific → matching date; else most-recent).
    Falls back to the most-recent test. ``sorted_tests`` must be WtDate-desc.
    """
    import pandas as pd

    if sorted_tests is None or sorted_tests.empty:
        return None
    sig = st.session_state.get(f"sw_ipr_applied_sig_{well_name}")
    mode = sig[0] if sig else "recent"
    token = sig[1] if sig else None
    if mode == "median" and "BHP" in sorted_tests.columns:
        valid = sorted_tests[sorted_tests["BHP"].notna()]
        if not valid.empty:
            pos = int((valid["BHP"] - valid["BHP"].median()).abs().values.argmin())
            return valid.iloc[pos]
    elif mode == "specific" and token:
        d = pd.to_datetime(sorted_tests["WtDate"], errors="coerce")
        match = sorted_tests[d.dt.strftime("%Y-%m-%d") == token]
        if not match.empty:
            return match.iloc[0]
    return sorted_tests.iloc[0]  # most-recent (WtDate-desc) / fallback


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

    sorted_tests = test_df.sort_values(
        "WtDate", ascending=False
    ).reset_index(drop=True)

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
                "Synced to the IPR anchor (in Model vs Actual below). Check "
                "'Use a different test for comparison' under the IPR anchor to "
                "pick the comparison test independently."
            ),
        )
        return sorted_tests.iloc[tgt_idx]

    # DECOUPLED (or <2 tests): independent picker.
    # Clamp persisted selection to the current option list so a stale value
    # from a prior session doesn't crash the selectbox if the test cache
    # changed shape.
    if st.session_state.get(key) not in options:
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

    selected = st.selectbox(
        "Test to compare against",
        options=options,
        index=default_idx,
        key=key,
        help=(
            "Pick a well test. The hero-strip vs-actual deltas, the Model vs "
            "Actual comparison, and the pump used for that comparison (from "
            "JP history at the test's date) all key off this selection. "
            "Default = most recent test."
        ),
    )

    idx = options.index(selected)
    row = sorted_tests.iloc[idx]
    # Persist the pick so a Batch/PF detour doesn't snap it back to most-recent.
    _d = row.get("WtDate")
    if pd.notna(_d):
        st.session_state[shadow_key] = _d.strftime("%Y-%m-%d")
    return row


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
                "The 'Test to compare against' picker above syncs to this by "
                "default (toggle below to unsync)."
            ),
        )
    mode = label_to_mode[sel]

    anchor_date = None
    if mode == "specific":
        sorted_tests = test_df.sort_values(
            "WtDate", ascending=False
        ).reset_index(drop=True)
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
        if st.session_state.get(anchor_key) not in date_opts:
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
            picked = st.selectbox(
                "Anchor test",
                options=date_opts,
                index=default_date_idx,
                key=anchor_key,
                help="The test the Vogel curve is forced through.",
            )
        ad = sorted_tests.iloc[date_opts.index(picked)].get("WtDate")
        anchor_date = ad if pd.notna(ad) else None

    # Decouple toggle. By default the top "Test to compare against" picker is
    # slaved to this IPR anchor (so the model is compared against the same test
    # the IPR is built on). Checking this frees the comparison picker to select
    # any test. render_tab reads this key at the top of the run — Streamlit
    # commits it before the rerun, so toggling takes effect immediately. (It
    # resets to synced after a tab detour, which keeps the test selection
    # consistent since the anchor itself persists.)
    st.checkbox(
        "Use a different test for comparison (un-sync from the IPR anchor)",
        key=f"sw_ipr_decouple_{well_name}",
        help=(
            "By default the 'Test to compare against' picker at the top of the "
            "tab matches the IPR anchor test selected here, so the model is "
            "compared against the same test the IPR is built on. Check this to "
            "choose the comparison test independently."
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

    # Match the int/round casting the sidebar auto-populate uses (sidebar.py
    # _auto_populate_from_ipr), so the recent-fit case is a byte-for-byte no-op
    # and never reports a spurious change.
    gor_floor = st.session_state.get("_well_min_gor", {}).get(well_name, 0)
    new_vals = {
        "qwf": int(qwf_oil),
        "pwf": int(pwf),
        "res_pres": int(res_p),
        "form_wc": round(float(form_wc), 2),
        "form_gor": max(int(fgor), gor_floor),
    }

    # Record the selection as applied up front so we don't keep re-evaluating.
    st.session_state[sig_key] = current_sig

    if all(st.session_state.get(k) == v for k, v in new_vals.items()):
        return  # sidebar already matches — nothing to push

    for k, v in new_vals.items():
        st.session_state[k] = v
        # Drop the widget key so the sidebar's _number_input re-seeds it from
        # the logical key on the next run (writing *_input after render raises).
        st.session_state.pop(f"{k}_input", None)

    anchor_human = {
        "recent": "most-recent-test",
        "median": "median-test",
        "specific": "selected-test",
    }.get(anchor_mode, anchor_mode)
    st.session_state["_ipr_sync_msg"] = (
        f"Seeded the {anchor_human} IPR + fluid into the sidebar "
        f"(qwf {new_vals['qwf']:,} BOPD · pwf {new_vals['pwf']:,} psi · "
        f"ResP {new_vals['res_pres']:,} psi · WC {new_vals['form_wc']:.2f} · "
        f"GOR {new_vals['form_gor']:,}). Batch Run, PF Range, and the Solver "
        "now use this — edit any field in the sidebar to override."
    )
    st.rerun()


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
    """
    jp_hist = st.session_state.get("jp_history_df")
    if jp_hist is None or params.selected_well == "Custom":
        return

    import pandas as pd

    from woffl.assembly.ipr_analyzer import (
        compute_vogel_coefficients,
        estimate_reservoir_pressure,
        generate_ipr_curves,
    )
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

    if pump_info is None:
        st.info(f"No JP history for {params.selected_well}")
        return

    nozzle = pump_info["nozzle_no"]
    throat = pump_info["throat_ratio"]
    if not nozzle or not throat:
        st.warning(
            f"JP history for {params.selected_well} is missing nozzle or throat data."
        )
        return

    install_date_str = (
        pump_info["date_set"].strftime("%Y-%m-%d")
        if pump_info.get("date_set") is not None
        else "N/A"
    )

    # 2. Get well tests from pre-fetched cache (includes any session-only
    # manual/provisional tests injected by get_well_tests_for_well).
    test_df = _get_well_tests(params.selected_well)
    n_tests = 0 if test_df is None or test_df.empty else len(test_df)

    # Manual provisional-test entry — always available (a well with no
    # Databricks tests can still be modeled once a manual test is added).
    _render_manual_test_entry(params.selected_well)

    # IPR anchor selector (≥2 tests). Separate from the comparison picker: it
    # controls which test the Vogel curve is anchored on and re-fits reservoir
    # pressure through it. Writes session keys consumed by
    # build_calibration_inputs so calibration targets the same operating point.
    anchor_mode, anchor_date = "recent", None
    if n_tests >= 2:
        anchor_mode, anchor_date = _render_ipr_anchor_control(
            params.selected_well, test_df
        )

    # One-shot confirmation from the prior run's sidebar sync. The sync reruns
    # the app, so the note has to be surfaced on the following render.
    _ipr_sync_msg = st.session_state.pop("_ipr_sync_msg", None)
    if _ipr_sync_msg:
        st.success(_ipr_sync_msg, icon="✅")

    # 3. Try Vogel fit (≥2 tests). With 0 or 1 tests we fall through to a
    # single-point synthetic IPR anchored on sidebar ResP — chart still
    # renders so the user can see the model's expected operating envelope.
    coeff_row = None
    merged_with_rp = None
    vogel_coeffs = None
    if n_tests >= 2:
        if anchor_mode in ("median", "specific"):
            # Anchored fit: hold the chosen test as the Vogel anchor and re-fit
            # reservoir pressure for the best fit through it (GUI-layer helper,
            # no upstream-library change).
            from woffl.gui.ipr_anchor import compute_anchored_vogel

            field_max_rp = (
                3000 if str(params.field_model).lower() == "kuparuk" else 1800
            )
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
            try:
                merged_with_rp = estimate_reservoir_pressure(test_df)
                vogel_coeffs = compute_vogel_coefficients(merged_with_rp)
            except Exception as e:
                st.warning(
                    f"Vogel IPR fit failed ({e}); falling back to a single-point "
                    "IPR with sidebar reservoir pressure."
                )
                vogel_coeffs = None

            # Vogel may return an empty DataFrame when every test row for this
            # well is missing BHP (e.g. S-pad wells whose recent tests have no
            # coincident gauge). Fall through to the no-IPR path in that case.
            if (
                vogel_coeffs is not None
                and not vogel_coeffs.empty
                and "Well" in vogel_coeffs.columns
            ):
                well_coeffs = vogel_coeffs[
                    vogel_coeffs["Well"] == params.selected_well
                ]
                if not well_coeffs.empty:
                    coeff_row = well_coeffs.iloc[0]

    # Seed the chosen test's IPR + fluid into the sidebar so Batch Run, PF
    # Range, and the top Solver all use it (the engineer can then override any
    # field in the sidebar — see _sync_chosen_ipr_to_sidebar). No-op unless the
    # anchor-test selection actually changed. Done before the chart so a
    # selection change reruns immediately and the chart/solver below read the
    # freshly-seeded sidebar values.
    if coeff_row is not None:
        _sync_chosen_ipr_to_sidebar(
            params.selected_well,
            anchor_mode=anchor_mode,
            anchor_date=anchor_date,
            qwf_oil=float(coeff_row["qwf"]) * (1 - float(coeff_row["form_wc"])),
            pwf=float(coeff_row["pwf"]),
            res_p=float(coeff_row["ResP"]),
            form_wc=float(coeff_row["form_wc"]),
            fgor=float(coeff_row["fgor"]),
        )

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
        if (
            anchor_qwf > 0
            and 0 <= anchor_pwf < model_res_p
            and model_res_p > 0
        ):
            try:
                synth_qmax = InFlow.vogel_qmax(
                    anchor_qwf, anchor_pwf, model_res_p
                )
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
            "test's date (e.g. \"12B\"), drawn inside an enlarged colored "
            "marker. Useful for seeing pump changes alongside the IPR shape."
        ),
    )

    # 5. Always render the IPR chart (Vogel-fit or synthetic).
    if params.selected_well in ipr_data:
        plot_data = (
            plot_points if plot_points is not None else pd.DataFrame()
        )
        fig = create_ipr_plotly(
            params.selected_well,
            ipr_data[params.selected_well],
            plot_data,
            form_wc=params.form_wc,
            jp_history=jp_hist,
            show_jp_labels=show_jp_labels,
        )
        st.plotly_chart(fig, use_container_width=True)

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

    # 6. Display comparison metrics
    actual_oil = recent_test.get("WtOilVol", None)
    actual_bhp = recent_test.get("BHP", None)
    actual_pf = recent_test.get("lift_wat", None)
    actual_whp = recent_test.get("whp", None)

    # Heading reflects the SELECTED test's date (test picker default = most
    # recent). Falls back to a generic label if the test row has no WtDate.
    if recent_test is not None and pd.notna(recent_test.get("WtDate")):
        _test_date_label = (
            f"{recent_test.get('WtDate').strftime('%Y-%m-%d')} Test"
        )
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
            st.checkbox(
                f"Apply rate-scalar calibration (×{cal_result.calibration_factor:.2f})",
                value=False,
                key="sw_apply_calibration",
                help=(
                    "After the solver runs, scales modeled oil and water by a "
                    "constant so modeled oil matches actual oil from the most "
                    "recent test. Display-only — does not change BHP or PF "
                    "rate. Stacking this on top of the BHP friction "
                    "calibration double-corrects; prefer one or the other."
                ),
            )

            if st.session_state.get("sw_apply_calibration", False):
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
