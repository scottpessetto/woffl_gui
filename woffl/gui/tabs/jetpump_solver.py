"""Tab 1: Jetpump Solver Results

Renders the single-pump solution display showing suction pressure,
oil rate, water rate, power fluid rate, and sonic status.

When JP history is uploaded and a non-Custom well is selected,
also shows a "Model vs Actual" comparison section with IPR chart
and modeled vs actual metrics.
"""

import streamlit as st

from woffl.geometry.jetpump import JetPump
from woffl.gui.fric_calibration import calibrate_friction_coefs
from woffl.gui.params import SimulationParams
from woffl.gui.utils import (
    _trigger_gor_reset,
    build_calibration_inputs,
    create_pvt_components,
    is_valid_number,
    render_bhp_calibration_warning,
    render_pf_mismatch_warning,
    render_pf_quickfix_widget,
    run_jetpump_solver,
)


def _render_input_summary(params: SimulationParams) -> None:
    """Render collapsible input summary showing current model parameters."""
    ipr_info = st.session_state.get("sw_ipr_info")
    label = f"Model Inputs (Nozzle {params.nozzle_no}, Throat {params.area_ratio})"

    with st.expander(label, expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Pump**")
            st.write(f"Nozzle: {params.nozzle_no}")
            st.write(f"Throat: {params.area_ratio}")
            st.write(f"ken: {params.ken}")
            st.write(f"kth: {params.kth}")
            st.write(f"kdi: {params.kdi}")
        with col2:
            st.markdown("**Well**")
            st.write(f"PF Pressure: {params.ppf_surf} psi")
            st.write(f"Surface Pressure: {params.surf_pres} psi")
            st.write(f"PF Density: {params.rho_pf} lbm/ft\u00b3")
            st.write(f"JP TVD: {params.jpump_tvd} ft")
        with col3:
            st.markdown("**Formation / IPR**")
            st.write(f"Reservoir Pressure: {params.pres} psi")
            st.write(f"Water Cut: {params.form_wc:.2f}")
            st.write(f"GOR: {params.form_gor} scf/bbl")
            st.write(f"Temperature: {params.form_temp} \u00b0F")
            st.write(f"qwf: {params.qwf} BOPD / pwf: {params.pwf} psi")
        if ipr_info:
            st.caption(f"*{ipr_info}*")


def _render_pump_identity_banner(params: SimulationParams) -> None:
    """Show banner indicating whether sidebar pump matches the installed pump."""
    jp_hist = st.session_state.get("jp_history_df")
    if jp_hist is None or params.selected_well == "Custom":
        st.info(f"Modeling: Nozzle {params.nozzle_no}, Throat {params.area_ratio}")
        return

    from woffl.assembly.jp_history import get_current_pump

    current_pump = get_current_pump(jp_hist, params.selected_well)
    if current_pump is None:
        st.info(f"Modeling: Nozzle {params.nozzle_no}, Throat {params.area_ratio}")
        return

    nozzle = current_pump["nozzle_no"]
    throat = current_pump["throat_ratio"]
    date_set = current_pump["date_set"]
    date_str = date_set.strftime("%Y-%m-%d") if date_set is not None else "N/A"

    if params.nozzle_no == nozzle and params.area_ratio == throat:
        st.success(
            f"Modeling: Nozzle {params.nozzle_no}, Throat {params.area_ratio} "
            f"\u2014 Matches current installed pump ({nozzle}{throat}, set {date_str})"
        )
    else:
        st.warning(
            f"Current installed pump is {nozzle}{throat}. "
            f"You are modeling a different configuration "
            f"(Nozzle {params.nozzle_no}, Throat {params.area_ratio})."
        )


def _render_calibration_overrides(params: SimulationParams, jetpump):
    """Render the unified Calibration Overrides section.

    Two distinct calibrations can be applied to the Solver run, and historically
    each was a separate checkbox scattered around the page. They confused new
    users because the names ("friction-coef cal", "rate-scalar cal") give no
    clue about what they do or how they differ. This helper presents both in
    one place with a short plain-language explainer:

    - **Friction-coef** — sweeps ken/kth/kdi to drive *modeled BHP* toward
      *measured BHP*. Fixes a physical model mismatch. Applied **before** the
      solver (mutates the JetPump object).
    - **Rate-scalar** — multiplies modeled rates by a constant factor so
      modeled oil matches actual oil from the most recent test. Patches over
      residual error after friction-cal. Applied **after** the solver
      (display-only; results shown below the hero metrics).

    Returns the (possibly mutated) JetPump object so the caller can use it
    in the solver run.
    """
    if params.selected_well == "Custom":
        return jetpump

    fric_cal = (
        st.session_state.get("sw_fric_calibration", {}).get(params.selected_well)
    )
    rate_cal = st.session_state.get("sw_calibration_result")
    fric_available = fric_cal is not None and fric_cal.converged
    rate_available = rate_cal is not None

    if not fric_available and not rate_available:
        return jetpump

    use_fric = bool(st.session_state.get("sw_apply_fric_cal", False))
    use_rate = bool(st.session_state.get("sw_apply_calibration", False))
    expanded = use_fric or use_rate

    label = "Calibration Overrides"
    if use_fric and use_rate:
        label += "  ·  ⚠️ both applied"
    elif use_fric or use_rate:
        label += "  ·  ✓ active"

    with st.expander(label, expanded=expanded):
        st.caption(
            "Two ways to align the model with the well's last test. "
            "Both are optional. Run them from the Model vs Actual section "
            "below first, then come back and toggle here."
        )

        if fric_available:
            st.checkbox(
                f"**Physics calibration** — use ken={fric_cal.best_ken:.3f}, "
                f"kth={fric_cal.best_kth:.3f}, kdi={fric_cal.best_kdi:.3f}",
                value=False,
                key="sw_apply_fric_cal",
                help=(
                    "Replaces sidebar friction coefficients with values fit to "
                    "drive modeled BHP toward the well's measured BHP. Recomputes "
                    "the solver from scratch — affects every metric."
                ),
            )
        if rate_available:
            st.checkbox(
                f"**Rate-scalar calibration** — multiply modeled rates by "
                f"{rate_cal.calibration_factor:.2f}",
                value=False,
                key="sw_apply_calibration",
                help=(
                    "After the solver runs, scales modeled oil and water by a "
                    "constant so modeled oil matches actual oil from the most "
                    "recent test. Display-only — does not change BHP or PF rate."
                ),
            )

        # Refresh after rendering
        use_fric = bool(st.session_state.get("sw_apply_fric_cal", False))
        use_rate = bool(st.session_state.get("sw_apply_calibration", False))
        if use_fric and use_rate:
            st.warning(
                "Both calibrations are applied. The rate-scalar was fit against "
                "the *uncalibrated* model, so applying it on top of the physics "
                "calibration double-corrects. Prefer one or the other."
            )

    if use_fric:
        jetpump = JetPump(
            nozzle_no=params.nozzle_no,
            area_ratio=params.area_ratio,
            knz=jetpump.knz,
            ken=fric_cal.best_ken,
            kth=fric_cal.best_kth,
            kdi=fric_cal.best_kdi,
        )
    return jetpump


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
    _render_input_summary(params)

    st.subheader("Jetpump Solver Results")

    # Surface any persisted message from a prior auto-recovery (e.g. GOR reset)
    msg = st.session_state.pop("_solver_gor_reset_msg", None)
    if msg:
        st.warning(msg)

    _render_pump_identity_banner(params)

    # Clear stale calibration if well changed
    _cal = st.session_state.get("sw_calibration_result")
    if _cal and _cal.well_name != params.selected_well:
        st.session_state.pop("sw_calibration_result", None)

    jetpump = _render_calibration_overrides(params, jetpump)

    with st.spinner("Running jetpump solver..."):
        try:
            solver_results = run_jetpump_solver(
                params.surf_pres,
                params.form_temp,
                params.rho_pf,
                params.ppf_surf,
                jetpump,
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
        # Each modeled value is shown alongside its delta vs. the most recent
        # test (when available). The deltas double as a visual nudge that
        # these are the values the friction-coef calibration in Model vs
        # Actual can pull toward zero.
        actuals = _latest_actuals(params.selected_well)

        def _delta(modeled: float, actual: float | None, suffix: str) -> str | None:
            if actual is None:
                return None
            return f"{modeled - actual:+,.0f} {suffix}"

        h1, h2, h3, h4 = st.columns(4)
        with h1:
            d = _delta(qoil_std, actuals["oil"], "vs actual")
            st.metric(
                "Oil Rate", f"{qoil_std:,.0f} BOPD",
                delta=d, delta_color="off" if d is None else "normal",
            )
        with h2:
            st.metric("Formation Water", f"{fwat_bwpd:,.0f} BWPD")
        with h3:
            d = _delta(qnz_bwpd, actuals["pf"], "vs actual")
            st.metric(
                "Power Fluid", f"{qnz_bwpd:,.0f} BWPD",
                delta=d, delta_color="off" if d is None else "normal",
            )
        with h4:
            d = _delta(psu, actuals["bhp"], "vs actual")
            st.metric(
                "Suction Pressure", f"{psu:,.0f} psig",
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
        # still be applied.
        _render_fric_cal_action_bar(
            params, wellbore, well_profile, pf_blocked=pf_blocked
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

        # Show calibrated predictions when the rate-scalar checkbox is enabled
        # in the Calibration Overrides section above.
        _cal = st.session_state.get("sw_calibration_result")
        if (
            _cal
            and params.selected_well != "Custom"
            and st.session_state.get("sw_apply_calibration", False)
        ):
            cal_oil = qoil_std * _cal.calibration_factor
            cal_water = fwat_bwpd * _cal.calibration_factor
            st.markdown("**Calibrated predictions** (rate-scalar applied):")
            c1, c2 = st.columns(2)
            with c1:
                st.metric(
                    "Calibrated Oil",
                    f"{cal_oil:,.0f} BOPD",
                    delta=f"{cal_oil - qoil_std:+.0f}",
                )
            with c2:
                st.metric(
                    "Calibrated Water",
                    f"{cal_water:,.0f} BWPD",
                    delta=f"{cal_water - fwat_bwpd:+.0f}",
                )
            if (
                params.nozzle_no != _cal.current_nozzle
                or params.area_ratio != _cal.current_throat
            ):
                st.caption(
                    f"Factor derived from installed pump "
                    f"({_cal.current_nozzle}{_cal.current_throat}) \u2014 applying "
                    "to a different pump is an approximation."
                )
    else:
        st.warning(
            "The solver could not find a solution with the current parameters. "
            "Try adjusting the input values."
        )

    # Model vs Actual comparison (requires JP history + non-Custom well)
    _render_model_vs_actual(params, wellbore, well_profile)


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
    if tests is None or len(tests) < 2:
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
    return True, None


def _render_fric_cal_action_bar(
    params: SimulationParams,
    wellbore,
    well_profile,
    *,
    pf_blocked: bool = False,
) -> None:
    """Compact calibration action bar rendered right below the hero strip.

    Two buttons (Run + Push) plus a one-line status of the last result.
    Pulls the user toward calibration when the BHP red flag is showing —
    the buttons live where the eyes already are. Detailed result metrics
    stay down in Model vs Actual; this strip is just the action surface.

    When ``pf_blocked`` is True, the Run button is disabled — calibrating
    against a wrong PF pressure produces useless friction coefs. Push
    still works so a previously-good result remains applicable.
    """
    if not _can_run_fric_cal(params):
        return

    # Surface success message from a Push-to-sidebar click on the prior render
    pushed_msg = st.session_state.pop("_pushed_fric_msg", None)
    if pushed_msg:
        st.success(pushed_msg)

    cal_state = st.session_state.get("sw_fric_calibration", {})
    result = cal_state.get(params.selected_well)
    has_result = result is not None and getattr(result, "converged", False)

    col_run, col_push, col_status = st.columns([1.2, 2.5, 3])

    with col_run:
        run_label = "Re-run BHP Cal" if has_result else "Run BHP Calibration"
        run_clicked = st.button(
            run_label,
            type="primary",
            key="sw_run_fric_cal_top",
            use_container_width=True,
            disabled=pf_blocked,
            help=(
                "Disabled while PF rate mismatch is too large — fix the "
                "sidebar Power Fluid Surface Pressure first."
                if pf_blocked
                else None
            ),
        )

    with col_push:
        if has_result:
            st.button(
                f"Push to sidebar: ken={result.best_ken:.3f}, "
                f"kth={result.best_kth:.3f}, kdi={result.best_kdi:.3f}",
                key="sw_push_fric_to_sidebar_top",
                help=(
                    "Replace sidebar ken/kth/kdi with calibrated values. "
                    "Affects Batch Run / PF Range as well."
                ),
                on_click=_push_fric_to_sidebar_cb,
                args=(result.best_ken, result.best_kth, result.best_kdi),
                use_container_width=True,
            )

    with col_status:
        if pf_blocked:
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
                f"see *Model vs Actual* below for details."
            )
        else:
            st.caption(
                "Fits ken/kth/kdi to drive modeled BHP toward measured BHP. "
                "Detailed breakdown appears in *Model vs Actual* below."
            )

    if run_clicked:
        with st.spinner("Calibrating (Nelder-Mead)..."):
            ok, err = _execute_fric_cal(params, wellbore, well_profile)
        if not ok:
            st.error(err)
        else:
            st.rerun()


def _push_fric_to_sidebar_cb(ken: float, kth: float, kdi: float) -> None:
    """on_click callback for the 'Push to sidebar' button.

    Streamlit allows widget-key mutation inside on_click handlers (they run
    before the next render), but NOT in plain ``if button:`` blocks (which
    hit the post-render write protection). Both logical and widget keys are
    set so the number_input picks up the new value on the next render
    regardless of internal state caching.
    """
    st.session_state["ken"] = float(ken)
    st.session_state["kth"] = float(kth)
    st.session_state["kdi"] = float(kdi)
    st.session_state["ken_input"] = float(ken)
    st.session_state["kth_input"] = float(kth)
    st.session_state["kdi_input"] = float(kdi)
    # Top-panel "Use friction-coef calibration" checkbox is redundant once
    # the sidebar matches the calibration. Clear it.
    st.session_state.pop("sw_apply_fric_cal", None)
    st.session_state["_pushed_fric_msg"] = (
        f"Sidebar updated: ken={ken:.3f}, kth={kth:.3f}, kdi={kdi:.3f}"
    )


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
    """Most recent test values used by the Solver hero strip.

    Pulls allocated oil rate, measured BHP (suction), and PF rate from the
    most recent well test in the pre-fetched session cache. Returns None for
    any field the test row is missing. Surfaces these *up-front* in the hero
    so users immediately see where the model agrees vs. disagrees with reality
    — and which knobs (friction coefs in Model vs Actual) can pull each value.
    """
    blank = {"oil": None, "bhp": None, "pf": None}
    if well_name == "Custom":
        return blank
    tests = _get_well_tests(well_name)
    if tests is None or tests.empty:
        return blank
    recent = tests.sort_values("WtDate", ascending=False).iloc[0]

    def _get(col: str) -> float | None:
        if col not in tests.columns:
            return None
        v = recent.get(col)
        return float(v) if is_valid_number(v) else None

    return {
        "oil": _get("WtOilVol"),
        "bhp": _get("BHP"),
        "pf": _get("lift_wat"),
    }


def _render_model_vs_actual(params: SimulationParams, wellbore, well_profile) -> None:
    """Render the Model vs Actual comparison section.

    Only shown when JP history is uploaded and a non-Custom well is selected.
    """
    jp_hist = st.session_state.get("jp_history_df")
    if jp_hist is None or params.selected_well == "Custom":
        return

    from woffl.assembly.ipr_analyzer import (
        compute_vogel_coefficients,
        estimate_reservoir_pressure,
        generate_ipr_curves,
    )
    from woffl.assembly.jp_history import get_current_pump
    from woffl.gui.ipr_viz import create_ipr_plotly
    from woffl.gui.utils import create_inflow, create_jetpump, create_reservoir_mix

    st.divider()
    st.subheader("Model vs Actual Comparison")

    # 1. Look up current JP from history
    current_pump = get_current_pump(jp_hist, params.selected_well)
    if current_pump is None:
        st.info(f"No JP history for {params.selected_well}")
        return

    nozzle = current_pump["nozzle_no"]
    throat = current_pump["throat_ratio"]
    if not nozzle or not throat:
        st.warning(
            f"JP history for {params.selected_well} is missing nozzle or throat data."
        )
        return

    st.info(
        f"Model vs Actual uses the CURRENT INSTALLED pump: "
        f"Nozzle {nozzle}, Throat {throat} "
        f"(set {current_pump['date_set'].strftime('%Y-%m-%d') if current_pump['date_set'] is not None else 'N/A'})"
    )

    # 2. Get well tests from pre-fetched cache
    test_df = _get_well_tests(params.selected_well)
    if test_df is None or len(test_df) < 2:
        st.info(
            f"Not enough recent test data for {params.selected_well} (need 2+ tests with BHP)."
        )
        return

    # 3. Estimate reservoir pressure + compute Vogel coefficients
    try:
        merged_with_rp = estimate_reservoir_pressure(test_df)
        vogel_coeffs = compute_vogel_coefficients(merged_with_rp)
    except Exception as e:
        st.warning(f"IPR analysis failed: {e}")
        return

    well_coeffs = vogel_coeffs[vogel_coeffs["Well"] == params.selected_well]
    if well_coeffs.empty:
        st.warning("Could not compute Vogel coefficients for this well.")
        return

    coeff_row = well_coeffs.iloc[0]

    # Reservoir pressure override (same pattern as GOR override below)
    ipr_res_p = coeff_row["ResP"]
    override_res_p = st.checkbox(
        "Override Reservoir Pressure from IPR analysis",
        value=False,
        help=f"IPR analysis ResP: {ipr_res_p:.0f} psi. Check to use sidebar value ({params.pres}) instead.",
        key="mva_override_res_p",
    )
    model_res_p = params.pres if override_res_p else ipr_res_p
    st.caption(
        f"Using Reservoir Pressure: **{model_res_p:.0f}** psi "
        f"({'sidebar' if override_res_p else 'IPR analysis'})"
    )

    # 4. Generate IPR curves and display chart with test points
    if override_res_p:
        vogel_coeffs_plot = vogel_coeffs.copy()
        vogel_coeffs_plot.loc[
            vogel_coeffs_plot["Well"] == params.selected_well, "ResP"
        ] = model_res_p
        ipr_data = generate_ipr_curves(vogel_coeffs_plot)
    else:
        ipr_data = generate_ipr_curves(vogel_coeffs)
    if params.selected_well in ipr_data:
        fig = create_ipr_plotly(
            params.selected_well,
            ipr_data[params.selected_well],
            merged_with_rp,
            form_wc=params.form_wc,
        )
        st.plotly_chart(fig, use_container_width=True)

    # 5. Run model with current JP + IPR-derived inflow
    # Use most recent test GOR by default, with sidebar override option
    recent_test = test_df.sort_values("WtDate", ascending=False).iloc[0]
    test_gor = recent_test.get("fgor", None)
    test_gor = int(test_gor) if is_valid_number(test_gor) else None

    test_whp = recent_test.get("whp", None)
    test_whp = float(test_whp) if is_valid_number(test_whp) else None

    override_gor = st.checkbox(
        "Override GOR from well test",
        value=st.session_state.get("mva_override_gor", False),
        help=f"Test GOR: {test_gor} scf/bbl. Check to use sidebar value ({params.form_gor}) instead.",
        key="mva_override_gor",
    )
    model_gor = params.form_gor if override_gor or test_gor is None else test_gor
    st.caption(
        f"Using GOR: **{model_gor}** scf/bbl ({'sidebar' if override_gor or test_gor is None else 'well test'})"
    )

    model_surf_pres = test_whp if test_whp is not None else params.surf_pres
    st.caption(
        f"Using Surface Pressure: **{model_surf_pres:.0f}** psi ({'well test' if test_whp is not None else 'sidebar'})"
    )

    oil_qwf = coeff_row["qwf"] * (1 - coeff_row["form_wc"])
    ipr_inflow = create_inflow(oil_qwf, coeff_row["pwf"], model_res_p)
    ipr_res_mix = create_reservoir_mix(
        coeff_row["form_wc"], model_gor, params.form_temp, params.field_model
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

    st.markdown("#### Modeled vs Actual (Most Recent Test)")

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
        else:
            st.session_state.pop("sw_calibration_result", None)

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
