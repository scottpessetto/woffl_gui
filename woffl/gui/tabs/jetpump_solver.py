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
    Renders the IPR chart for any test count (0 / 1 / 2+); the comparison
    table + calibration sections require at least 1 test.
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
    if coeff_row is not None:
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
            recent = test_df.sort_values("WtDate", ascending=False).iloc[0]
            total = recent.get("WtTotalFluid")
            bhp = recent.get("BHP")
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
        )
        st.plotly_chart(fig, use_container_width=True)

    # 6. Modeled vs Actual + calibration sections need at least 1 test.
    if n_tests == 0:
        return

    # 7. Run model with current JP + IPR-derived inflow.
    # Use most recent test GOR by default, with sidebar override option.
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
