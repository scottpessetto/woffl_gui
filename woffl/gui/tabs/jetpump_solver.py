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


def _render_pump_identity_banner(
    params: SimulationParams,
    *,
    effective_pump: tuple[str, str] | None = None,
    selected_test_date=None,
    fallback_reason: str | None = None,
) -> None:
    """Show what pump the solver is modeling vs what's in the well today.

    ``effective_pump`` is whatever the solver is actually using on this run.
    When a test is selected, this is the pump installed at that test's
    date (looked up via :func:`_pump_at_test_date`). Otherwise it falls
    back to the sidebar pump (``params.nozzle_no``/``area_ratio``).

    ``selected_test_date`` makes the banner clarify the test context
    ("Modeling X to match the YYYY-MM-DD test \u2014 current pump \u2026").

    ``fallback_reason`` flags the rare case where a test was selected but
    no JP install record qualifies \u2014 we fall back to the current pump and
    explain why.
    """
    import pandas as pd

    if effective_pump is not None:
        model_n, model_t = effective_pump
    else:
        model_n, model_t = params.nozzle_no, params.area_ratio

    jp_hist = st.session_state.get("jp_history_df")
    if jp_hist is None or params.selected_well == "Custom":
        st.info(f"Modeling: Nozzle {model_n}, Throat {model_t}")
        return

    from woffl.assembly.jp_history import get_current_pump

    current_pump = get_current_pump(jp_hist, params.selected_well)
    if current_pump is None:
        st.info(f"Modeling: Nozzle {model_n}, Throat {model_t}")
        return

    cur_n = current_pump["nozzle_no"]
    cur_t = current_pump["throat_ratio"]
    cur_date_str = (
        current_pump["date_set"].strftime("%Y-%m-%d")
        if current_pump.get("date_set") is not None
        else "N/A"
    )

    test_str = None
    if selected_test_date is not None and pd.notna(selected_test_date):
        test_str = selected_test_date.strftime("%Y-%m-%d")

    matches_current = (model_n == cur_n and model_t == cur_t)

    # Caveat path: the test selected has no qualifying JP install record,
    # so we fell back to the current pump.
    if fallback_reason:
        st.warning(
            f"Modeling **{model_n}{model_t}** (current pump in well). "
            f"{fallback_reason}"
        )
        return

    # Test-aware framing \u2014 "Modeling X to match the YYYY-MM-DD test".
    if test_str:
        if matches_current:
            st.success(
                f"Modeling **{model_n}{model_t}** to match the "
                f"**{test_str}** test \u2014 matches the current installed pump "
                f"(set {cur_date_str})."
            )
        else:
            st.info(
                f"Modeling **{model_n}{model_t}** to match the "
                f"**{test_str}** test. Current pump in well: "
                f"**{cur_n}{cur_t}** (set {cur_date_str})."
            )
        return

    # No test selected \u2014 original sidebar-vs-current framing.
    if matches_current:
        st.success(
            f"Modeling: Nozzle {model_n}, Throat {model_t} "
            f"\u2014 Matches current installed pump ({cur_n}{cur_t}, set {cur_date_str})"
        )
    else:
        st.warning(
            f"Current installed pump is {cur_n}{cur_t}. "
            f"You are modeling a different configuration "
            f"(Nozzle {model_n}, Throat {model_t})."
        )


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

    # Surface any persisted message from a prior auto-recovery (e.g. GOR reset)
    msg = st.session_state.pop("_solver_gor_reset_msg", None)
    if msg:
        st.warning(msg)

    # Test picker FIRST — its selection determines which pump the solver
    # uses on this run (pump-at-test-date overrides sidebar when a test is
    # selected). Default = most recent test; matches the pre-picker
    # behaviour.
    test_df = _get_well_tests(params.selected_well)
    selected_test_row = _render_test_picker(params.selected_well, test_df)

    # Resolve the EFFECTIVE pump for the solver. When a test is selected we
    # look up the JP install on/before that test's date — the test's PF rate
    # and BHP only make sense vs the pump that was actually in the well at
    # the time. When no JP record qualifies (or no test is selected), we
    # fall back to the sidebar/current pump.
    jp_hist_for_pump = st.session_state.get("jp_history_df")
    test_date_for_pump = None
    fallback_reason = None
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
                effective_nozzle = pump_at_test["nozzle_no"]
                effective_throat = pump_at_test["throat_ratio"]
            else:
                fallback_reason = (
                    f"No JP install record on or before "
                    f"{td_raw.strftime('%Y-%m-%d')} — using the current "
                    "pump for the model. Modeled vs actual will be misleading "
                    "if the pump differed at the test's date."
                )

    _render_pump_identity_banner(
        params,
        effective_pump=(effective_nozzle, effective_throat),
        selected_test_date=test_date_for_pump,
        fallback_reason=fallback_reason,
    )

    # Build the effective JetPump. When it matches the sidebar's pump
    # (no test override) we reuse the one passed in to avoid the extra
    # construction; when it differs we build a new one with the same
    # friction coefs.
    if (effective_nozzle, effective_throat) == (
        params.nozzle_no,
        params.area_ratio,
    ):
        effective_jetpump = jetpump
    else:
        from woffl.gui.utils import create_jetpump as _make_jp

        effective_jetpump = _make_jp(
            effective_nozzle,
            effective_throat,
            params.ken,
            params.kth,
            params.kdi,
        )

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
            # by an unrealistically low GOR for the well's PVT. Bump GOR to 250
            # and force the MvA override so a re-run uses sidebar GOR instead
            # of the (likely too-low) test GOR.
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
            return f"{modeled - actual:+,.0f} {suffix}"

        def _label(base: str, actual) -> str:
            """Append '(modeled)' to a hero-strip label when no actual exists,
            so the user doesn't mistake a sidebar-driven prediction for a
            measured value."""
            return base if actual is not None else f"{base} (modeled)"

        h1, h2, h3, h4 = st.columns(4)
        with h1:
            d = _delta(qoil_std, actuals["oil"], "vs actual")
            st.metric(
                _label("Oil Rate", actuals["oil"]), f"{qoil_std:,.0f} BOPD",
                delta=d, delta_color="off" if d is None else "normal",
            )
        with h2:
            # Formation Water has no actuals counterpart (we don't track it
            # in actuals dict), so it's always modeled — label accordingly.
            st.metric("Formation Water (modeled)", f"{fwat_bwpd:,.0f} BWPD")
        with h3:
            d = _delta(qnz_bwpd, actuals["pf"], "vs actual")
            st.metric(
                _label("Power Fluid", actuals["pf"]), f"{qnz_bwpd:,.0f} BWPD",
                delta=d, delta_color="off" if d is None else "normal",
            )
        with h4:
            d = _delta(psu, actuals["bhp"], "vs actual")
            st.metric(
                _label("Suction Pressure", actuals["bhp"]), f"{psu:,.0f} psig",
                delta=d, delta_color="off" if d is None else "normal",
            )

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
        # still be applied. Also disabled when the latest test has no
        # measured BHP (the calibration objective is BHP-match — without
        # an actual, there's nothing to fit toward).
        _render_fric_cal_action_bar(
            params, wellbore, well_profile,
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
    recent = tests.sort_values("WtDate", ascending=False).iloc[0]
    return is_valid_number(recent.get("BHP"))


def _execute_fric_cal(
    params: SimulationParams, wellbore, well_profile
) -> tuple[bool, str | None]:
    """Build the calibration inputs and run calibrate_friction_coefs.

    Single source of truth for "how do we calibrate this well" — invoked from
    the top action bar. Result is stashed in
    session_state["sw_fric_calibration"][well] for the existing display code.

    Returns (success, error_message). On failure, error_message is the
    caller's responsibility to surface.
    """
    from woffl.gui.utils import build_calibration_inputs

    inputs = build_calibration_inputs(params, wellbore, well_profile)
    if inputs is None:
        return False, "Cannot calibrate — missing JP history or test data."
    if inputs["actual_bhp"] is None:
        return False, "Cannot calibrate — most recent test has no measured BHP."

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
            "Disabled — most recent test has no measured BHP. Friction-coef "
            "calibration fits ken/kth/kdi to drive modeled BHP toward "
            "measured BHP, and needs a measured value to target."
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

    if run_clicked:
        with st.spinner("Calibrating (Nelder-Mead)..."):
            ok, err = _execute_fric_cal(params, wellbore, well_profile)
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
    """Get tests for a single well from the pre-fetched session state cache."""
    all_tests = st.session_state.get("all_well_tests_df")
    if all_tests is None or all_tests.empty:
        return None
    well_df = all_tests[all_tests["well"] == well_name].copy()
    return well_df if not well_df.empty else None


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


def _render_test_picker(well_name: str, test_df):
    """Selectbox listing the well's tests (date-desc); returns the picked row.

    Default = most recent test, so behaviour matches the pre-picker version
    when the user doesn't interact. The picker drives:
      * the hero-strip vs-actual deltas (Oil / PF / BHP),
      * the Model vs Actual section's modeled pump (via :func:`_pump_at_test_date`)
        and its comparison row.

    Selection persists per-well via session_state, so switching wells doesn't
    carry stale state. Returns None when no tests exist.
    """
    import pandas as pd

    if test_df is None or test_df.empty:
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

    # Clamp persisted selection to the current option list so a stale value
    # from a prior session doesn't crash the selectbox if the test cache
    # changed shape.
    if st.session_state.get(key) not in options:
        st.session_state.pop(key, None)

    selected = st.selectbox(
        "Test to compare against",
        options=options,
        index=0,
        key=key,
        help=(
            "Pick a well test. The hero-strip vs-actual deltas, the Model vs "
            "Actual comparison, and the pump used for that comparison (from "
            "JP history at the test's date) all key off this selection. "
            "Default = most recent test."
        ),
    )

    idx = options.index(selected)
    return sorted_tests.iloc[idx]


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

    # 2. Get well tests from pre-fetched cache
    test_df = _get_well_tests(params.selected_well)
    n_tests = 0 if test_df is None or test_df.empty else len(test_df)

    # 3. Try Vogel fit (≥2 tests). With 0 or 1 tests we fall through to a
    # single-point synthetic IPR anchored on sidebar ResP — chart still
    # renders so the user can see the model's expected operating envelope.
    coeff_row = None
    merged_with_rp = None
    vogel_coeffs = None
    if n_tests >= 2:
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
            well_coeffs = vogel_coeffs[vogel_coeffs["Well"] == params.selected_well]
            if not well_coeffs.empty:
                coeff_row = well_coeffs.iloc[0]

    # 4. Resolve the IPR anchor (qwf, pwf, ResP) and the points to overlay.
    # Compute test_gor / test_whp HERE (was: later, after the chart) so the
    # GOR override checkbox can sit next to the ResP override in a single
    # st.columns(2) row at the top of this section. Both checkboxes render
    # before the chart so the user sees them above the picture they affect.
    recent_test = None
    test_gor: int | None = None
    test_whp: float | None = None
    if n_tests >= 1:
        if selected_test_row is not None:
            recent_test = selected_test_row
        else:
            recent_test = test_df.sort_values("WtDate", ascending=False).iloc[0]
        _g = recent_test.get("fgor", None)
        test_gor = int(_g) if is_valid_number(_g) else None
        _w = recent_test.get("whp", None)
        test_whp = float(_w) if is_valid_number(_w) else None

    # Side-by-side overrides. Each column is independent: ResP only renders
    # when Vogel produced a coeff_row; GOR only renders when the selected
    # test has an fgor value. Captions stack below each column.
    if coeff_row is not None or test_gor is not None:
        col_resp, col_gor = st.columns(2)
        with col_resp:
            if coeff_row is not None:
                ipr_res_p = coeff_row["ResP"]
                override_res_p = st.checkbox(
                    "Override Reservoir Pressure from IPR analysis",
                    value=False,
                    help=(
                        f"IPR analysis ResP: {ipr_res_p:.0f} psi. Check to "
                        f"use sidebar value ({params.pres}) instead."
                    ),
                    key="mva_override_res_p",
                )
                model_res_p_local = (
                    params.pres if override_res_p else ipr_res_p
                )
                st.caption(
                    f"Using Reservoir Pressure: **{model_res_p_local:.0f}** "
                    f"psi ({'sidebar' if override_res_p else 'IPR analysis'})"
                )
            else:
                override_res_p = False
                model_res_p_local = float(params.pres)

        with col_gor:
            if test_gor is not None:
                override_gor = st.checkbox(
                    "Override GOR from well test",
                    value=st.session_state.get("mva_override_gor", False),
                    help=(
                        f"Test GOR: {test_gor} scf/bbl. Check to use "
                        f"sidebar value ({params.form_gor}) instead."
                    ),
                    key="mva_override_gor",
                )
                model_gor = (
                    params.form_gor if override_gor else test_gor
                )
                st.caption(
                    f"Using GOR: **{model_gor}** scf/bbl "
                    f"({'sidebar' if override_gor else 'well test'})"
                )
            else:
                override_gor = False
                model_gor = params.form_gor

    if coeff_row is not None:
        model_res_p = model_res_p_local

        if override_res_p:
            vogel_coeffs_plot = vogel_coeffs.copy()
            vogel_coeffs_plot.loc[
                vogel_coeffs_plot["Well"] == params.selected_well, "ResP"
            ] = model_res_p
            ipr_data = generate_ipr_curves(vogel_coeffs_plot)
        else:
            ipr_data = generate_ipr_curves(vogel_coeffs)
        plot_points = merged_with_rp
    else:
        # 0-test, 1-test, or Vogel-failed path. Clear any stale override flag
        # from a previously-selected multi-test well; the override only makes
        # sense when there's an IPR-derived ResP to override.
        st.session_state.pop("mva_override_res_p", None)
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

    # 7. Run model with current JP + IPR-derived inflow.
    # `recent_test`, `test_gor`, `test_whp`, and `model_gor` were all
    # computed earlier (next to the override checkboxes at the top of this
    # section). When test_gor is None we fall back to sidebar GOR — same as
    # the prior behaviour, just expressed up front.
    if test_gor is None:
        model_gor = params.form_gor

    model_surf_pres = test_whp if test_whp is not None else params.surf_pres
    st.caption(
        f"Using Surface Pressure: **{model_surf_pres:.0f}** psi ({'well test' if test_whp is not None else 'sidebar'})"
    )

    if coeff_row is not None:
        oil_qwf = coeff_row["qwf"] * (1 - coeff_row["form_wc"])
        pwf_for_inflow = coeff_row["pwf"]
        wc_for_resmix = coeff_row["form_wc"]
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
