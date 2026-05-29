"""Utility functions for the WOFFL GUI

This module contains helper functions for the Streamlit GUI.
"""

import math
import os

import numpy as np
import pandas as pd
import streamlit as st

from woffl.assembly.batchpump import BatchPump, exp_model, rev_exp_deriv, validate_water
from woffl.assembly.solopump import jetpump_solver
from woffl.flow import jetgraphs as jg
from woffl.flow import jetplot as jplt
from woffl.flow.inflow import InFlow
from woffl.geometry.jetpump import JetPump
from woffl.geometry.pipe import Pipe, PipeInPipe
from woffl.geometry.wellprofile import WellProfile
from woffl.pvt.blackoil import BlackOil
from woffl.pvt.formgas import FormGas
from woffl.pvt.formwat import FormWater
from woffl.pvt.resmix import ResMix


def render_input_summary(params) -> None:
    """Render the collapsible Model Inputs expander.

    Shared between the Solver tab and the Batch Run tab so the user can
    see exactly what pump / well / formation inputs the solver and the
    nozzle-throat sweep are using, without scrolling back to the sidebar.
    """
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
            st.write(f"PF Density: {params.rho_pf} lbm/ft³")
            st.write(f"JP TVD: {params.jpump_tvd} ft")
        with col3:
            st.markdown("**Formation / IPR**")
            st.write(f"Reservoir Pressure: {params.pres} psi")
            st.write(f"Water Cut: {params.form_wc:.2f}")
            st.write(f"GOR: {params.form_gor} scf/bbl")
            st.write(f"Temperature: {params.form_temp} °F")
            st.write(f"qwf: {params.qwf} BOPD / pwf: {params.pwf} psi")
        if ipr_info:
            st.caption(f"*{ipr_info}*")


def is_valid_number(val) -> bool:
    """Check if a value is a valid (non-None, non-NaN) number.

    Uses ``pd.isna`` so it catches ``pd.NA`` in addition to None and NaN —
    pd.NA can appear in well-test rows after a "disregard Databricks BHP"
    toggle, and the older isinstance(float)+math.isnan check missed it.
    """
    if val is None:
        return False
    try:
        return not bool(pd.isna(val))
    except (TypeError, ValueError):
        return True  # non-NA-checkable scalars (strings, etc.)


# Default well-test lookback window (months). The single-well solver lets the
# user widen this via the sidebar; app startup pre-fetches this window into the
# shared ``all_well_tests_df`` cache.
DEFAULT_TEST_MONTHS = 6


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_all_well_tests(months_back: int = DEFAULT_TEST_MONTHS):
    """Fetch recent well tests for all MPU wells in one query. Cached 24h per window.

    Single source of truth for the well-test lookback window. Both app startup
    and per-well slicing in :func:`get_well_tests_for_well` call this, so a
    change to the sidebar lookback re-fetches (cached) for the new window
    without disturbing windows already cached or other consumers reading the
    startup-cached frame.
    """
    from datetime import datetime

    from dateutil.relativedelta import relativedelta

    from woffl.assembly.well_test_client import fetch_milne_well_tests

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - relativedelta(months=months_back)).strftime(
        "%Y-%m-%d"
    )
    df, _ = fetch_milne_well_tests(start_date, end_date)
    return df


def get_well_tests_for_well(well_name: str) -> pd.DataFrame | None:
    """Return well tests for a single well with memory-gauge BHP override applied.

    Single source of truth for all consumers — Solver MvA, Batch hero, sidebar
    IPR auto-populate, PDF export, calibration. Encapsulates three things:
      1. Per-well **extended** test window when a memory gauge sets one
         (tests from gauge.start_date to today, fetched on Apply). Takes
         precedence over the shared 3-month cache because the gauge's
         coverage typically extends further back than 3 months.
      2. Slicing the shared ``st.session_state["all_well_tests_df"]`` cache
         when there's no extended-window override.
      3. Applying the gauge's BHP daily medians to the resulting test rows.
      4. Appending session-only manual/provisional tests (Solver tab entry
         form) so a well with no Databricks tests can still be modeled.

    Returns ``None`` for Custom mode, or when neither Databricks nor manual
    tests exist for the well. Always returns a *copy* so callers can safely
    mutate (e.g. append columns).
    """
    if well_name == "Custom":
        return None

    # Lazy imports — keeps gui.utils importable at startup without depending
    # on the gauge module's session-state shape.
    from woffl.gui.memory_gauge import (
        apply_to_well_tests,
        get_extended_tests,
        get_gauge,
        is_disregarding_databricks_bhp,
    )

    well_df = None
    extended = get_extended_tests(well_name)
    if extended is not None and not extended.empty:
        # Extended fetch is a per-well full pull starting at the gauge's
        # start date — a superset of whatever this well has in the shared
        # lookback cache, so prefer it unconditionally.
        well_df = extended.copy()
    else:
        # Honor the sidebar lookback. fetch_all_well_tests is cached per
        # months_back, so the default window reuses the startup-cached frame
        # and a widened window fetches once then caches. Fall back to the
        # startup-cached frame if the re-fetch fails (e.g. Databricks down).
        months = int(st.session_state.get("sw_test_months", DEFAULT_TEST_MONTHS))
        try:
            all_tests = fetch_all_well_tests(months)
        except Exception:
            all_tests = None
        if all_tests is None or all_tests.empty:
            all_tests = st.session_state.get("all_well_tests_df")
        if all_tests is not None and not all_tests.empty:
            sliced = all_tests[all_tests["well"] == well_name].copy()
            if not sliced.empty:
                well_df = sliced

    # Memory-gauge handling applies only to the Databricks-sourced rows.
    if well_df is not None:
        # "Disregard Databricks BHP" — blank the column before applying the
        # gauge so anything OUTSIDE the gauge's coverage shows as missing
        # rather than inheriting the (known-bad) Databricks value. Using
        # ``np.nan`` (not ``pd.NA``) so the column stays float64 — pd.NA
        # would coerce to object dtype and break the Vogel fit downstream.
        if is_disregarding_databricks_bhp(well_name) and "BHP" in well_df.columns:
            well_df["BHP"] = np.nan

        gauge = get_gauge(well_name)
        if gauge is not None:
            well_df = apply_to_well_tests(well_df, gauge)

    # Inject session-only manual/provisional tests (Solver tab entry form).
    # Done after gauge handling so manual rows keep their own measured BHP,
    # and before the count cap so a recent manual test survives the cap.
    well_df = _append_manual_tests(well_df, well_name)

    if well_df is None or well_df.empty:
        return None

    # Optional cap: keep only the N most recent tests. 0 (default) = no cap.
    cap = int(st.session_state.get("sw_test_count_cap", 0) or 0)
    if cap > 0 and "WtDate" in well_df.columns and len(well_df) > cap:
        well_df = (
            well_df.sort_values("WtDate", ascending=False)
            .head(cap)
            .reset_index(drop=True)
        )

    return well_df


# Column schema for manually-entered provisional tests, aligned with the
# Databricks well-test frame so injected rows merge cleanly.
MANUAL_TEST_COLUMNS = [
    "well",
    "WtDate",
    "WtOilVol",
    "WtWaterVol",
    "WtTotalFluid",
    "BHP",
    "fgor",
    "lift_wat",
    "whp",
]


def _append_manual_tests(well_df, well_name: str):
    """Append session-cached manual tests for a well to its test frame.

    Manual tests live in ``st.session_state['sw_manual_tests'][well]`` (a list
    of row dicts written by the Solver tab's entry form). Returns the combined
    frame sorted newest-first, the manual rows alone when there are no
    Databricks tests, or the input unchanged when there are no manual rows.
    """
    manual = st.session_state.get("sw_manual_tests", {}).get(well_name)
    if not manual:
        return well_df

    manual_df = pd.DataFrame(manual)
    if "WtDate" in manual_df.columns:
        manual_df["WtDate"] = pd.to_datetime(manual_df["WtDate"], errors="coerce")

    if well_df is None or well_df.empty:
        combined = manual_df
    else:
        combined = pd.concat([well_df, manual_df], ignore_index=True)

    if "WtDate" in combined.columns:
        combined = combined.sort_values("WtDate", ascending=False)
    return combined.reset_index(drop=True)


GOR_AUTO_RECOVERY_VALUE = 250


# Pad-level default PF surface pressures (psi). C/E/H/I/M/S run at 3400,
# B/G/J at 2200 (booster pads), F at 2800. Pad K has no jet pumps.
# Wired into the sidebar's well auto-populate so MPB/MPG/MPF wells stop
# loading with the Schrader-pad default and triggering false PF mismatch
# warnings on the Solver / Batch tabs.
PAD_PF_DEFAULTS: dict[str, int] = {
    "B": 2200,
    "C": 3400,
    "E": 3400,
    "F": 2800,
    "G": 2200,
    "H": 3400,
    "I": 3400,
    "J": 2200,
    "M": 3400,
    "S": 3400,
}
PAD_PF_FALLBACK = 3400


def default_pad_pf(pad: str) -> int:
    """Default PF surface pressure (psi) for a given pad letter."""
    return PAD_PF_DEFAULTS.get(pad, PAD_PF_FALLBACK)


def pad_from_mp_name(mp_name: str) -> str:
    """MPB-30 → B, MPI-15 → I. Returns the input unchanged for unknown formats."""
    if not mp_name or "-" not in mp_name:
        return ""
    return mp_name.replace("MP", "").split("-")[0]


def _trigger_gor_reset(well_name: str, current_gor, reason: str) -> None:
    """Auto-recover from a solver failure caused by too-low GOR.

    Sets sidebar form_gor to GOR_AUTO_RECOVERY_VALUE, checks the MvA override,
    records a per-well GOR floor so the next well-selection cycle won't
    repopulate this well below the recovery value, queues a warning to display
    on the next render, then triggers a rerun.

    Streamlit forbids writing to a widget's state key after the widget rendered
    (the sidebar already rendered above the tabs). So we set the logical key
    and DELETE the widget key — the _number_input helper re-initializes the
    widget from the logical key on the next run.
    """
    st.session_state["form_gor"] = GOR_AUTO_RECOVERY_VALUE
    st.session_state.pop("form_gor_input", None)
    st.session_state["mva_override_gor"] = True

    # Remember per-well GOR floor — survives well-switches within the session
    # so we don't keep tripping the same failure for the same well.
    floor_map = st.session_state.setdefault("_well_min_gor", {})
    floor_map[well_name] = max(
        floor_map.get(well_name, 0), GOR_AUTO_RECOVERY_VALUE
    )

    st.session_state["_solver_gor_reset_msg"] = (
        f"Solver failed to converge for {well_name} at GOR={current_gor} scf/bbl "
        f"({reason}). GOR has been reset to **{GOR_AUTO_RECOVERY_VALUE} scf/bbl** "
        f"and the 'Override GOR from well test' box has been checked. "
        f"This floor is now remembered for {well_name} for the rest of the session."
    )
    st.rerun()


# BHP-match thresholds used by render_bhp_calibration_warning. Tuned to match
# the existing oil-rate quality grading on CalibrationResult (good <15%).
# Both absolute and percentage thresholds catch a real model mismatch:
#   - absolute trips first on low-pressure wells
#   - percentage trips first on high-pressure wells
_BHP_BAD_ABS_PSI = 100
_BHP_BAD_PCT = 15.0

# PF-rate-match thresholds. Two tiers: MODERATE shows a soft yellow info
# (calibration still allowed), SEVERE shows a red flag and gates calibration
# until the user acknowledges via the quickfix.
#
# Rationale: a perfectly tuned PF pressure is rarely achievable in practice
# (test values are noisy, operating PF drifts). Early versions blocked
# calibration on >100 BWPD / >15% which made the iterative cal-fix loop
# noisy. Once the user has interacted with the quickfix once for a well,
# we trust them and downgrade to the soft tier even at severe deltas.
_PF_BAD_ABS_BWPD = 100
_PF_BAD_PCT = 15.0
_PF_SEVERE_ABS_BWPD = 250
_PF_SEVERE_PCT = 25.0


def _pf_is_acknowledged(well_name: str | None) -> bool:
    """True once the user has interacted with the PF quickfix for this well."""
    if not well_name:
        return False
    return well_name in st.session_state.get("_pf_ack_wells", set())


def _ack_pf(well_name: str | None) -> None:
    """Mark that the user has interacted with the PF quickfix for this well.

    Called from the quickfix on_change callback and the Auto-match button.
    Persists for the rest of the session — once the user has shown they
    know about the mismatch, stop nagging them with red flags.
    """
    if not well_name:
        return
    st.session_state.setdefault("_pf_ack_wells", set()).add(well_name)


def render_bhp_calibration_warning(
    modeled_bhp, actual_bhp, *, on_solver_view: bool = False
) -> bool:
    """Render a red flag recommending Friction-Coef Calibration when BHP match is bad.

    Returns True when a warning was rendered, False otherwise (so callers can
    suppress duplicate hint text immediately afterward). Silently no-ops when
    either value is missing — calibration only makes sense with both sides.

    A "bad" match is |delta| > 100 psi OR > 15% of actual. Below that the
    physics calibration would just be fitting noise.

    Pass ``on_solver_view=True`` from the Solver view so the action text reads
    "scroll down to Friction-Coef Calibration" instead of "switch to the
    Solver view" — telling a user already on Solver to switch to Solver is
    confusing.
    """
    if not is_valid_number(modeled_bhp) or not is_valid_number(actual_bhp):
        return False
    abs_delta = abs(modeled_bhp - actual_bhp)
    pct_delta = (abs_delta / actual_bhp * 100) if actual_bhp > 0 else 0.0
    if abs_delta <= _BHP_BAD_ABS_PSI and pct_delta <= _BHP_BAD_PCT:
        return False

    direction = "above" if modeled_bhp > actual_bhp else "below"
    if on_solver_view:
        action = (
            "Scroll down to **Model vs Actual → Friction-Coef Calibration** "
            "to fit ken/kth/kdi against the measured BHP."
        )
    else:
        action = (
            "Switch to the **Solver** view and run *Friction-Coef Calibration* "
            "(in the Model vs Actual section) to fit ken/kth/kdi against the "
            "measured BHP."
        )
    st.error(
        f"🚩 **BHP match is poor** — modeled suction pressure is "
        f"{abs_delta:.0f} psi ({pct_delta:.1f}%) {direction} the measured BHP "
        f"of {actual_bhp:.0f} psi. {action}"
    )
    return True


def render_pf_mismatch_warning(
    modeled_pf,
    actual_pf,
    ppf_surf,
    *,
    test_date_str: str | None = None,
    well_name: str | None = None,
) -> tuple[bool, bool]:
    """Render a tiered PF mismatch warning. Returns ``(warning_shown, blocked)``.

    Three states based on severity + per-well acknowledgment:
      - **No warning** when |delta| ≤ moderate threshold → returns (False, False)
      - **Soft yellow info** when moderate-but-not-severe, OR when severe
        but the user has already acknowledged via the quickfix this session.
        Calibration is allowed → returns (True, False).
      - **Red error + gate** only when SEVERE and not yet acknowledged →
        returns (True, True). This is the only path that gates the cal button.

    The intent is to flag the issue once and then get out of the user's way.
    Iterating on PF + calibration becomes painful when a fresh red flag
    appears every time the modeled PF drifts >100 BWPD off the test value.
    Callers render the quickfix widget whenever ``warning_shown`` is True so
    the user can fine-tune from either tier.
    """
    if not is_valid_number(modeled_pf) or not is_valid_number(actual_pf):
        return False, False
    abs_delta = abs(modeled_pf - actual_pf)
    pct_delta = (abs_delta / actual_pf * 100) if actual_pf > 0 else 0.0
    if abs_delta <= _PF_BAD_ABS_BWPD and pct_delta <= _PF_BAD_PCT:
        return False, False

    is_severe = abs_delta > _PF_SEVERE_ABS_BWPD or pct_delta > _PF_SEVERE_PCT
    acknowledged = _pf_is_acknowledged(well_name)
    block = is_severe and not acknowledged

    direction = "above" if modeled_pf > actual_pf else "below"
    test_clause = (
        f" The matching well test was on **{test_date_str}** — look up the "
        f"actual PF surface pressure operating that day."
        if test_date_str
        else ""
    )

    if block:
        st.error(
            f"🚩 **PF rate mismatch — fix this before calibrating BHP** — "
            f"modeled PF rate is {abs_delta:.0f} BWPD ({pct_delta:.1f}%) "
            f"{direction} the measured actual ({actual_pf:.0f} BWPD). The "
            f"sidebar **Power Fluid Surface Pressure** ({ppf_surf} psi) "
            f"probably doesn't match the operating value.{test_clause} Adjust "
            "it until this delta is small, then run BHP calibration."
        )
    else:
        # Soft yellow info — small, easy to scan past once the user knows.
        st.info(
            f"ℹ️ PF rate is {abs_delta:.0f} BWPD ({pct_delta:.1f}%) "
            f"{direction} the test actual ({actual_pf:.0f} BWPD). "
            "Calibration is allowed but accuracy will suffer if PF is "
            "materially wrong — fine-tune via the quickfix below if needed."
        )

    return True, block


def build_calibration_inputs(
    params, wellbore, well_profile, selected_test_row=None,
) -> dict | None:
    """Build the IPR-derived simulation inputs for calibration / PF-search.

    Single source of truth for "given a selected well, what inputs do we feed
    the solver to compare against a specific well test." Used by both the
    BHP friction-cal executor and the PF auto-match solver — they need the
    exact same pump identity, inflow (IPR-derived), and surface-pressure
    choice (test value if present, else sidebar).

    The ``selected_test_row`` argument lets a caller target a specific test
    (driven by the Solver tab's test picker) instead of defaulting to the
    most-recent test. When provided, the pump used for calibration is the
    one that was installed in the well on that test's date — looked up
    via :func:`woffl.gui.tabs.jetpump_solver._pump_at_test_date`. When
    omitted, behavior matches the original most-recent-test default.

    Honors the override flags set in the Model vs Actual section:
      - mva_override_res_p → use sidebar reservoir pressure instead of IPR
      - mva_override_gor   → use sidebar GOR instead of test GOR

    Two test-count branches:
      - **2+ tests** → fit Vogel coefficients and use the IPR-derived
        (qwf, pwf, ResP) triple as the operating point.
      - **1 test** → no Vogel fit possible. Anchor the inflow with the
        single test's oil rate + BHP and complete the curve with the
        sidebar's reservoir pressure. Lets calibration run for new
        wells with only one historical test (e.g. H-31).

    Returns None when prerequisites are missing (Custom mode, no JP history,
    no well tests, etc.). Callers must handle None.
    """
    from woffl.assembly.ipr_analyzer import (
        compute_vogel_coefficients,
        estimate_reservoir_pressure,
    )
    from woffl.assembly.jp_history import get_current_pump

    if params.selected_well == "Custom":
        return None

    jp_hist = st.session_state.get("jp_history_df")
    if jp_hist is None:
        return None

    # Route through the central helper so memory-gauge BHP overrides feed
    # into both the BHP-target friction cal and the PF auto-match solver.
    well_tests = get_well_tests_for_well(params.selected_well)
    if well_tests is None or len(well_tests) < 1:
        return None

    # Pick the target test (and, when the user has selected one explicitly,
    # also the pump that was installed at that test's date). Falling back
    # to most-recent + current-pump preserves the original behaviour for
    # callers that don't pass a selection.
    if selected_test_row is not None:
        recent = selected_test_row
        # Lazy import to avoid utils ↔ tabs/jetpump_solver circular import
        # at module load. ``_pump_at_test_date`` is a pure pandas helper,
        # no UI side effects.
        from woffl.gui.tabs.jetpump_solver import _pump_at_test_date

        td = recent.get("WtDate")
        pump_at_date = (
            _pump_at_test_date(jp_hist, params.selected_well, td)
            if pd.notna(td) else None
        )
        if (
            pump_at_date is not None
            and pump_at_date.get("nozzle_no")
            and pump_at_date.get("throat_ratio")
        ):
            nozzle = pump_at_date["nozzle_no"]
            throat = pump_at_date["throat_ratio"]
        else:
            # Test predates any JP install record; fall back to current pump
            # rather than failing the whole calibration call.
            current_pump = get_current_pump(jp_hist, params.selected_well)
            if current_pump is None:
                return None
            nozzle = current_pump.get("nozzle_no")
            throat = current_pump.get("throat_ratio")
    else:
        current_pump = get_current_pump(jp_hist, params.selected_well)
        if current_pump is None:
            return None
        nozzle = current_pump.get("nozzle_no")
        throat = current_pump.get("throat_ratio")
        recent = well_tests.sort_values("WtDate", ascending=False).iloc[0]

    if not nozzle or not throat:
        return None

    coeff_row = None
    if len(well_tests) >= 2:
        # Honor the Solver tab's IPR-anchor selector so the calibration target
        # operating point matches the chart. Defaults to "recent" (the global
        # least-squares fit) when the user hasn't picked an anchor.
        anchor_mode = st.session_state.get(
            f"sw_ipr_anchor_mode_{params.selected_well}", "recent"
        )
        anchor_date = st.session_state.get(
            f"sw_ipr_anchor_date_{params.selected_well}"
        )
        if anchor_mode in ("median", "specific"):
            from woffl.gui.ipr_anchor import compute_anchored_vogel

            field_max_rp = (
                3000 if str(params.field_model).lower() == "kuparuk" else 1800
            )
            anchored = compute_anchored_vogel(
                well_tests,
                well_name=params.selected_well,
                anchor_mode=anchor_mode,
                anchor_date=anchor_date,
                field_max_rp=field_max_rp,
            )
            if anchored is not None:
                coeff_row = pd.Series(anchored)
        if coeff_row is None:
            try:
                merged = estimate_reservoir_pressure(well_tests)
                vogel = compute_vogel_coefficients(merged)
                well_coeffs = vogel[vogel["Well"] == params.selected_well]
                if not well_coeffs.empty:
                    coeff_row = well_coeffs.iloc[0]
            except Exception:
                coeff_row = None

    override_res_p = bool(st.session_state.get("mva_override_res_p", False))

    if coeff_row is not None:
        model_res_p = (
            float(params.pres) if override_res_p else float(coeff_row["ResP"])
        )
        oil_qwf = float(coeff_row["qwf"]) * (1 - float(coeff_row["form_wc"]))
        pwf_for_inflow = float(coeff_row["pwf"])
        wc_for_resmix = float(coeff_row["form_wc"])
    else:
        # 1-test (or Vogel-fit-failed) fallback: anchor IPR on the single test
        # point + sidebar ResP. WC pulls from sidebar since there's no Vogel-
        # averaged value available.
        model_res_p = float(params.pres)
        actual_oil = recent.get("WtOilVol")
        actual_bhp_val = recent.get("BHP")
        if is_valid_number(actual_oil) and is_valid_number(actual_bhp_val):
            oil_qwf = float(actual_oil)
            pwf_for_inflow = float(actual_bhp_val)
        else:
            oil_qwf = float(params.qwf)
            pwf_for_inflow = float(params.pwf)
        wc_for_resmix = float(params.form_wc)

    test_gor = recent.get("fgor")
    test_gor = int(test_gor) if is_valid_number(test_gor) else None
    override_gor = bool(st.session_state.get("mva_override_gor", False))
    model_gor = (
        params.form_gor if (override_gor or test_gor is None) else test_gor
    )

    test_whp = recent.get("whp")
    model_surf_pres = (
        float(test_whp) if is_valid_number(test_whp) else float(params.surf_pres)
    )

    ipr_inflow = create_inflow(oil_qwf, pwf_for_inflow, model_res_p)
    ipr_res_mix = create_reservoir_mix(
        wc_for_resmix, model_gor, params.form_temp, params.field_model
    )

    test_date = recent.get("WtDate")
    test_date_str = (
        test_date.strftime("%Y-%m-%d") if hasattr(test_date, "strftime") else None
    )

    actual_bhp = recent.get("BHP")
    actual_pf = recent.get("lift_wat")

    return {
        "nozzle": str(nozzle),
        "throat": str(throat),
        "model_surf_pres": model_surf_pres,
        "ipr_inflow": ipr_inflow,
        "ipr_res_mix": ipr_res_mix,
        "actual_bhp": float(actual_bhp) if is_valid_number(actual_bhp) else None,
        "actual_pf": float(actual_pf) if is_valid_number(actual_pf) else None,
        "test_date_str": test_date_str,
    }


_PF_QUICKFIX_KEY = "_pf_quickfix_box"


def _on_pf_quickfix_change() -> None:
    """on_change callback for the quickfix number input.

    Sets the logical sidebar key (``ppf_surf``) and POPs the sidebar widget
    key (``ppf_surf_input``). The sidebar's _number_input helper re-initializes
    the widget key from the logical key on the next render — see CLAUDE.md
    gotcha about writing to widget keys after a widget has already rendered
    in the same run.

    Also marks the PF mismatch as acknowledged for the current well so the
    next render downgrades the warning from red to yellow info. Reads the
    current well from a session_state hint set right before render.
    """
    try:
        new_val = int(st.session_state[_PF_QUICKFIX_KEY])
    except (KeyError, ValueError, TypeError):
        return
    st.session_state["ppf_surf"] = new_val
    st.session_state.pop("ppf_surf_input", None)
    _ack_pf(st.session_state.get("_pf_quickfix_well"))


def _solve_pf_for_actual_lift(
    target_lift_wat: float,
    params,
    wellbore,
    well_profile,
    inputs: dict,
) -> tuple[int | None, float | None, str | None]:
    """Search for the ppf_surf that produces target_lift_wat from the installed pump.

    Uses the installed pump (from JP history) and IPR-derived inflow — the
    same context calibration uses. Each evaluation runs the full jetpump
    solver once.

    The search starts at [1500, 4000] psi but uses a NaN-robust,
    self-expanding bracket finder (``pf_calibration.robust_bracket``) that
    nudges past solver-failure bounds and widens to the realistic surface
    envelope [800, 5500] psi before giving up — so a true bracket that lies
    outside the narrow start window is still found. brentq then converges
    within the bracket.

    Returns (solved_ppf, achieved_lift, error_message). On no-bracket or
    solver failure, ppf is None and error_message explains why.
    """
    from scipy.optimize import brentq

    from woffl.gui.pf_calibration import robust_bracket

    pump = create_jetpump(
        inputs["nozzle"], inputs["throat"], params.ken, params.kth, params.kdi
    )

    def lift_at(ppf: float) -> float:
        result = run_jetpump_solver(
            inputs["model_surf_pres"],
            float(params.form_temp),
            float(params.rho_pf),
            float(ppf),
            pump,
            wellbore,
            well_profile,
            inputs["ipr_inflow"],
            inputs["ipr_res_mix"],
            field_model=params.field_model,
            jpump_direction=params.jpump_direction,
        )
        if result is None:
            return float("nan")
        _psu, _sonic, _qoil, _fwat, qnz, _mach = result
        return float(qnz)

    def residual(ppf: float) -> float:
        # NaN (solver failure) propagates through the subtraction.
        return lift_at(ppf) - target_lift_wat

    # Realistic PF surface-pressure envelope. Start narrow; robust_bracket
    # expands outward to these caps as needed.
    LO_FLOOR, HI_CAP = 800.0, 5500.0
    br = robust_bracket(
        residual, 1500.0, 4000.0, lo_floor=LO_FLOOR, hi_cap=HI_CAP
    )

    if br["status"] == "nan":
        return None, None, (
            "Solver failed across the PF-pressure search range — the installed "
            "pump may not converge at these conditions. Check the IPR and pump "
            "inputs."
        )

    if br["status"] == "no_bracket":
        # Same sign even at the caps: fb<0 → can't reach target even at HI_CAP;
        # fa>0 → overshoots even at LO_FLOOR.
        if br["fb"] < 0:
            modeled = br["fb"] + target_lift_wat
            return None, None, (
                f"Even at {br['b']:.0f} psi PF, modeled lift "
                f"({modeled:.0f} BWPD) stays below the target "
                f"({target_lift_wat:.0f} BWPD). The pump can't reach the "
                "measured rate within the realistic surface envelope — likely "
                "a washout or the wrong nozzle/throat."
            )
        modeled = br["fa"] + target_lift_wat
        return None, None, (
            f"Even at {br['a']:.0f} psi PF, modeled lift "
            f"({modeled:.0f} BWPD) exceeds the target "
            f"({target_lift_wat:.0f} BWPD). The pump moves more than the "
            "measured rate at the lowest realistic PF pressure — check the "
            "pump size or the test's PF rate."
        )

    a, b = br["a"], br["b"]
    if a == b:  # exact hit on a bound
        return int(round(a)), float(br["fa"] + target_lift_wat), None

    try:
        solved = brentq(residual, a, b, xtol=2, maxiter=40)
    except Exception as e:
        return None, None, f"Auto-match search failed: {e}"

    achieved = lift_at(solved)
    return int(round(solved)), float(achieved), None


def render_pf_quickfix_widget(
    params, wellbore, well_profile, *, target_lift_wat: float | None
) -> None:
    """Inline PF-pressure quickfix rendered below a PF mismatch warning.

    Two affordances side-by-side:
      1. **Number input** — typing a value updates sidebar ppf_surf via
         on_change; Streamlit reruns and the simulation re-evaluates.
      2. **Auto-match button** — solves for the ppf_surf that produces the
         most recent test's measured lift_wat.

    The widget value is synced to current ``params.ppf_surf`` before render
    so external sidebar changes reflect here too. Skipped silently when the
    target lift isn't available (Auto-match would have nothing to match).
    """
    inputs = build_calibration_inputs(params, wellbore, well_profile)
    if inputs is None:
        return  # No JP/test context — quickfix wouldn't help anyway

    # Stash the current well so the on_change callback can mark it acked.
    # (on_change runs without arguments, so it has to read context from
    # session_state.)
    st.session_state["_pf_quickfix_well"] = params.selected_well

    # Sync widget value to current sidebar value before render
    if st.session_state.get(_PF_QUICKFIX_KEY) != int(params.ppf_surf):
        st.session_state[_PF_QUICKFIX_KEY] = int(params.ppf_surf)

    # Show any auto-match status from the previous render
    msg = st.session_state.pop("_pf_automatch_msg", None)
    err = st.session_state.pop("_pf_automatch_err", None)
    if msg:
        st.success(msg)
    if err:
        st.warning(err)

    col_input, col_auto, col_status = st.columns([2, 1, 3])
    with col_input:
        st.number_input(
            "Quickfix: actual PF surface pressure (psi)",
            min_value=1500, max_value=4000, step=10,
            key=_PF_QUICKFIX_KEY,
            on_change=_on_pf_quickfix_change,
            help=(
                "Updates the sidebar Power Fluid Surface Pressure. The "
                "simulation re-runs immediately and the lockout disappears "
                "once the modeled PF rate matches the test."
            ),
        )

    with col_auto:
        # Vertical alignment hack — caption above the button to push it down
        st.caption(" ")
        auto_clicked = st.button(
            "Auto-match",
            key="_pf_automatch_btn",
            help=(
                "Solve for the PF pressure that produces the test's actual "
                "lift water rate. Runs ~10 solver evaluations."
            ),
            disabled=target_lift_wat is None,
            use_container_width=True,
        )

    with col_status:
        st.caption(" ")
        if target_lift_wat is None:
            st.caption("Auto-match unavailable — test has no measured PF rate.")
        else:
            st.caption(
                f"Target: {target_lift_wat:.0f} BWPD "
                f"(from {inputs['test_date_str'] or 'most recent test'})"
            )

    if auto_clicked and target_lift_wat is not None:
        # User has engaged with PF — downgrade future warnings even if the
        # solve fails (they've shown they know about the mismatch).
        _ack_pf(params.selected_well)
        with st.spinner("Searching for matching PF pressure..."):
            solved, achieved, error = _solve_pf_for_actual_lift(
                target_lift_wat, params, wellbore, well_profile, inputs
            )
        if solved is None:
            st.session_state["_pf_automatch_err"] = error or "Auto-match failed."
        else:
            # Sidebar + quickfix widgets already rendered this run, so writing
            # to their session_state keys would raise. Set the logical key and
            # POP the widget keys; both widgets re-initialize from the logical
            # key on the next render (see CLAUDE.md gotcha).
            st.session_state["ppf_surf"] = solved
            st.session_state.pop("ppf_surf_input", None)
            st.session_state.pop(_PF_QUICKFIX_KEY, None)
            delta = achieved - target_lift_wat
            st.session_state["_pf_automatch_msg"] = (
                f"Auto-matched **PF = {solved} psi** → modeled lift "
                f"{achieved:.0f} BWPD vs target {target_lift_wat:.0f} "
                f"({delta:+.0f} BWPD)."
            )
        st.rerun()


def create_jetpump(nozzle_no, area_ratio, ken, kth, kdi):
    """Create a JetPump object with the given parameters."""
    return JetPump(
        nozzle_no=nozzle_no, area_ratio=area_ratio, ken=ken, kth=kth, kdi=kdi
    )


def create_pvt_components(
    field_model=None,
    oil_api=None,
    gas_sg=None,
    wat_sg=None,
    bubble_point=None,
):
    """Create PVT components (oil, water, gas) for the given field model.

    This is the single source of truth for Schrader/Kuparuk PVT model selection.
    Used by both create_reservoir_mix() and network_optimizer._create_well_objects().

    Args:
        field_model: "Schrader" or "Kuparuk" (case-insensitive). Provides defaults
            for any oil_api/gas_sg/wat_sg/bubble_point not explicitly supplied.
        oil_api, gas_sg, wat_sg, bubble_point: Optional per-well overrides
            (e.g., from Databricks vw_prop_resvr). When provided, these replace
            the field_model preset values.

    Returns:
        tuple: (BlackOil, FormWater, FormGas) instances
    """
    if field_model is None:
        field_model = "schrader"
    field_model = field_model.lower()

    if field_model == "kuparuk":
        oil_default = BlackOil.kuparuk()
        wat_default = FormWater.kuparuk()
        gas_default = FormGas.kuparuk()
    else:
        oil_default = BlackOil.schrader()
        wat_default = FormWater.schrader()
        gas_default = FormGas.schrader()

    final_api = oil_api if oil_api is not None else oil_default.oil_api
    final_pbp = bubble_point if bubble_point is not None else oil_default.pbp
    final_oil_sg = gas_sg if gas_sg is not None else oil_default.gas_sg
    final_gas_sg = gas_sg if gas_sg is not None else gas_default.gas_sg
    final_wat_sg = wat_sg if wat_sg is not None else wat_default.wat_sg

    oil = BlackOil(oil_api=final_api, bubblepoint=final_pbp, gas_sg=final_oil_sg)
    water = FormWater(wat_sg=final_wat_sg)
    gas = FormGas(gas_sg=final_gas_sg)
    return oil, water, gas


def create_reservoir_mix(
    wc,
    gor,
    temp,
    field_model=None,
    oil_api=None,
    gas_sg=None,
    wat_sg=None,
    bubble_point=None,
):
    """Create a ResMix object with the given parameters."""
    oil, water, gas = create_pvt_components(
        field_model=field_model,
        oil_api=oil_api,
        gas_sg=gas_sg,
        wat_sg=wat_sg,
        bubble_point=bubble_point,
    )
    return ResMix(wc=wc, fgor=gor, oil=oil, wat=water, gas=gas)


def create_well_profile(field_model=None, jpump_tvd=None):
    """Create a WellProfile object with the given field model and jetpump TVD.

    Args:
        field_model (str, optional): The field model to use ("schrader" or "kuparuk").
            If None, defaults to "schrader".
        jpump_tvd (float, optional): The jetpump true vertical depth in feet.
            If provided, the well profile will be adjusted to have this TVD.
            If not provided, the default jetpump MD from the field model will be used.

    Returns:
        WellProfile: A WellProfile object
    """
    # Default to schrader if field_model is None
    if field_model is None:
        field_model = "schrader"

    field_model = (
        field_model.lower()
    )  # Convert to lowercase for case-insensitive comparison

    # First create a well profile with the default jetpump MD
    if field_model == "schrader":
        well_profile = WellProfile.schrader()
    elif field_model == "kuparuk":
        well_profile = WellProfile.kuparuk()
    else:
        # Default to schrader if an unknown model is specified
        well_profile = WellProfile.schrader()

    # If jpump_tvd is provided, create a new well profile with the correct jetpump MD
    if jpump_tvd is not None:
        try:
            # Convert TVD to MD using the well profile's interpolation
            jpump_md = well_profile.md_interp(jpump_tvd)

            # Create a new well profile with the same MD/VD arrays but with the new jetpump MD
            well_profile = WellProfile(
                md_list=well_profile.md_ray,
                vd_list=well_profile.vd_ray,
                jetpump_md=jpump_md,
            )
        except ValueError as e:
            # If the TVD is outside the well profile's range, log a warning and use the default
            print(f"Warning: {e}. Using default jetpump MD.")

    return well_profile


def create_pipes(
    tubing_od=4.5, tubing_thickness=0.5, casing_od=6.875, casing_thickness=0.5
):
    """Create tubing, casing, and wellbore (PipeInPipe) objects."""
    tube = Pipe(out_dia=tubing_od, thick=tubing_thickness)
    case = Pipe(out_dia=casing_od, thick=casing_thickness)
    wellbore = PipeInPipe(inn_pipe=tube, out_pipe=case)
    return tube, case, wellbore


def create_inflow(qwf, pwf, pres):
    """Create an InFlow object with the given parameters."""
    return InFlow(qwf=qwf, pwf=pwf, pres=pres)


def generate_choked_figures(
    form_temp, rho_pf, ppf_surf, jetpump, tube, well_profile, inflow, res_mix
):
    """Generate choked figures using the jetgraphs module."""
    return jg.choked_figures(
        form_temp, rho_pf, ppf_surf, jetpump, tube, well_profile, inflow, res_mix
    )


def generate_discharge_check(
    surf_pres,
    form_temp,
    rho_pf,
    ppf_surf,
    jetpump,
    wellbore,
    well_profile,
    inflow,
    res_mix,
):
    """Generate discharge check using the jetgraphs module."""
    return jg.discharge_check(
        surf_pres,
        form_temp,
        rho_pf,
        ppf_surf,
        jetpump,
        wellbore,
        well_profile,
        inflow,
        res_mix,
    )


def generate_multi_throat_entry_books(psu_ray, form_temp, ken, ate, inflow, res_mix):
    """Generate multi throat entry books using the jetplot module."""
    return jplt.multi_throat_entry_books(psu_ray, form_temp, ken, ate, inflow, res_mix)


def generate_multi_suction_graphs(qoil_list, book_list):
    """Generate multi suction graphs using the jetplot module."""
    return jplt.multi_suction_graphs(qoil_list, book_list)


def run_jetpump_solver(
    surf_pres,
    form_temp,
    rho_pf,
    ppf_surf,
    jetpump,
    wellbore,
    well_profile,
    inflow,
    res_mix,
    field_model=None,
    jpump_direction="reverse",
):
    """Run the jetpump solver and return the results.

    This function uses the jetpump_solver from solopump to find a solution for the jetpump system
    that factors in the wellhead pressure and reservoir conditions.

    Returns:
        tuple or None: (psu, sonic_status, qoil_std, fwat_bwpd, qnz_bwpd, mach_te) if successful,
                       None if the solver fails
    """
    # Create power fluid properties from field model water
    _, prop_pf, _ = create_pvt_components(field_model)
    prop_pf = prop_pf.condition(0, 60)

    try:
        return jetpump_solver(
            pwh=surf_pres,
            tsu=form_temp,
            ppf_surf=ppf_surf,
            jpump=jetpump,
            wellbore=wellbore,
            wellprof=well_profile,
            ipr_su=inflow,
            prop_su=res_mix,
            prop_pf=prop_pf,
            jpump_direction=jpump_direction,
        )
    except ValueError as e:
        # Handle the case where the well cannot lift at max suction pressure
        st.error(f"Solver error: {str(e)}")
        return None


def recommend_jetpump(batch_pump, marginal_watercut, water_type="lift"):
    """Recommend Jet Pump Based on Marginal Watercut

    Analyzes batch run results to recommend a jet pump configuration where the
    marginal watercut is closest to but still below the specified threshold.
    This represents the economic limit where additional water handling is justified.

    Args:
        batch_pump (BatchPump): BatchPump object with processed results
        marginal_watercut (float): Threshold for marginal watercut (bbl water / (bbl water + bbl oil))
        water_type (str): "lift" or "total" depending on the desired analysis

    Returns:
        dict: Recommended jet pump configuration and performance metrics
            {
                'nozzle': str,
                'throat': str,
                'qoil_std': float,
                'water_rate': float,
                'marginal_ratio': float,
                'recommendation_type': str
            }
    """
    # Validate inputs
    if not hasattr(batch_pump, "df") or batch_pump.df.empty:
        raise ValueError("Batch pump has no results to analyze")

    # Validate water type
    water_type = _validate_water_type(water_type)

    # Map water type to its column / marginal-ratio / curve-fit coefficient.
    # "formation" uses the GUI-side mofwr / coeff_form computed by
    # _augment_with_formation_marginals (not part of the library).
    if water_type == "formation":
        water_col = "form_wat"
        marg_col = "mofwr"
        coeff = getattr(batch_pump, "coeff_form", None)
    elif water_type == "total":
        water_col = "totl_wat"
        marg_col = "motwr"
        coeff = getattr(batch_pump, "coeff_totl", None)
    else:  # "lift"
        water_col = "lift_wat"
        marg_col = "molwr"
        coeff = getattr(batch_pump, "coeff_lift", None)

    # coeff may be None when the curve fit didn't converge (often a degenerate
    # case — single semi-finalist, or all water rates near-equal). The
    # function still produces a usable recommendation by picking from the
    # semi-finalists directly; the curve-fit step is only used to compute
    # the theoretical optimal point for closest-match selection.
    if marg_col not in batch_pump.df.columns:
        raise ValueError(
            f"Marginal column {marg_col!r} not found in batch results"
        )

    # Get semi-finalist pumps
    semi_df = batch_pump.df[batch_pump.df["semi"]].copy()
    if semi_df.empty:
        raise ValueError("No semi-finalist jet pumps found")

    # Sort by oil rate (which correlates with water rate for semi-finalists)
    semi_df = semi_df.sort_values(by="qoil_std", ascending=True)

    # Get water rates and calculate marginal watercuts for semi-finalists
    water_rates = semi_df[water_col].values

    # Convert marginal oil-water ratios to marginal watercuts
    # Original ratios are (bbl oil / bbl water)
    # We need (bbl water / (bbl water + bbl oil))
    original_ratios = semi_df[marg_col].values
    marginal_watercuts = 1 / (1 + original_ratios)

    # Check if any pumps meet the threshold (below the watercut threshold)
    below_threshold = marginal_watercuts <= marginal_watercut

    # If no pumps meet the threshold, recommend the one with lowest marginal watercut
    if not any(below_threshold):
        best_idx = np.argmin(marginal_watercuts)
        recommendation = {
            "nozzle": semi_df.iloc[best_idx]["nozzle"],
            "throat": semi_df.iloc[best_idx]["throat"],
            "qoil_std": semi_df.iloc[best_idx]["qoil_std"],
            "water_rate": semi_df.iloc[best_idx][water_col],
            "marginal_ratio": marginal_watercuts[best_idx],
            "recommendation_type": "best_available",
        }
        return recommendation

    # First, filter to only those below threshold
    valid_indices = np.where(below_threshold)[0]

    # No curve-fit coefficients (e.g. single semi-finalist). Pick the
    # highest-oil pump that still sits below the threshold.
    if coeff is None:
        valid_oil = [semi_df.iloc[idx]["qoil_std"] for idx in valid_indices]
        best_idx = valid_indices[int(np.argmax(valid_oil))]
        return {
            "nozzle": semi_df.iloc[best_idx]["nozzle"],
            "throat": semi_df.iloc[best_idx]["throat"],
            "qoil_std": semi_df.iloc[best_idx]["qoil_std"],
            "water_rate": semi_df.iloc[best_idx][water_col],
            "marginal_ratio": marginal_watercuts[best_idx],
            "recommendation_type": "optimal",
        }

    # Find theoretical optimal water rate using curve fit
    # This is where the marginal watercut equals the threshold
    # We need to convert the watercut threshold to an oil-water ratio first
    # watercut = water / (water + oil)
    # oil / water = (1 - watercut) / watercut
    oil_water_ratio = (1 - marginal_watercut) / marginal_watercut

    a, b, c = coeff
    optimal_water_rate = rev_exp_deriv(oil_water_ratio, b, c)
    optimal_oil_rate = exp_model(optimal_water_rate, a, b, c)

    # If we have the theoretical point, find the closest actual pump
    if valid_indices.size > 0:
        # Calculate distances to the theoretical optimal point
        distances = []
        for idx in valid_indices:
            pump_water = water_rates[idx]
            pump_oil = semi_df.iloc[idx]["qoil_std"]
            # Calculate Euclidean distance in the oil-water space
            distance = np.sqrt(
                (pump_water - optimal_water_rate) ** 2
                + (pump_oil - optimal_oil_rate) ** 2
            )
            distances.append(distance)

        # Find the closest pump
        closest_idx = valid_indices[np.argmin(distances)]

        recommendation = {
            "nozzle": semi_df.iloc[closest_idx]["nozzle"],
            "throat": semi_df.iloc[closest_idx]["throat"],
            "qoil_std": semi_df.iloc[closest_idx]["qoil_std"],
            "water_rate": semi_df.iloc[closest_idx][water_col],
            "marginal_ratio": marginal_watercuts[closest_idx],
            "recommendation_type": "optimal",
            "theoretical_water_rate": optimal_water_rate,
            "theoretical_oil_rate": optimal_oil_rate,
        }
        return recommendation

    # This should never happen if below_threshold check is done correctly
    raise ValueError("Could not determine recommended jet pump")


def _validate_water_type(water_type):
    """Validate Type of Water String

    GUI-side validator that extends the library's validate_water() with
    "formation" — the library only accepts "lift" / "total" / "totl"; the
    GUI also offers the formation-only axis (mofwr / form_wat) which we
    keep out of the library to avoid an upstream PR.

    Args:
        water_type (str): "lift", "total", or "formation"

    Returns:
        str: One of "lift", "total", "formation".
    """
    if water_type in {"formation", "form"}:
        return "formation"
    return validate_water(water_type)


def highlight_recommended_pump(ax, recommendation, water_type="lift"):
    """Highlight the recommended pump on a performance plot

    Args:
        ax (matplotlib.axes.Axes): The axes object to plot on
        recommendation (dict): The recommendation dictionary from recommend_jetpump
        water_type (str): "lift" or "total" depending on the plot type
    """
    if recommendation is None:
        return

    water_type = _validate_water_type(water_type)

    # Extract values from recommendation
    water_rate = recommendation["water_rate"]
    oil_rate = recommendation["qoil_std"]
    pump_label = recommendation["nozzle"] + recommendation["throat"]

    # Highlight the recommended pump with a star marker
    ax.plot(
        water_rate,
        oil_rate,
        marker="*",
        markersize=15,
        markerfacecolor="gold",
        markeredgecolor="black",
        markeredgewidth=1.5,
        linestyle="none",
        label=f"Recommended: {pump_label}",
    )

    # Add annotation
    ax.annotate(
        f"Recommended: {pump_label}",
        xy=(water_rate, oil_rate),
        xytext=(10, 10),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", alpha=0.8),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
    )


def run_batch_pump(
    surf_pres,
    form_temp,
    rho_pf,
    ppf_surf,
    wellbore,
    well_profile,
    inflow,
    res_mix,
    nozzle_options,
    throat_options,
    wellname="Test Well",
    field_model=None,
    jpump_direction="reverse",
    ken=0.03,
    kth=0.3,
    kdi=0.4,
    knz=0.01,
):
    """Run a batch pump simulation with multiple nozzle and throat combinations.

    Args:
        surf_pres: Surface pressure (psi)
        form_temp: Formation temperature (°F)
        rho_pf: Power fluid density (lbm/ft³) — used for display only
        ppf_surf: Power fluid surface pressure (psi)
        wellbore: PipeInPipe wellbore object
        well_profile: Well profile object
        inflow: Inflow performance object
        res_mix: Reservoir mixture object
        nozzle_options: List of nozzle sizes to test
        throat_options: List of throat ratios to test
        wellname: Name of the well for display purposes
        field_model: Field model for power fluid properties
        jpump_direction: Circulation direction ("forward" or "reverse")

    Returns:
        BatchPump: A BatchPump object with results, or None if processing fails
    """
    # Create power fluid properties from field model water
    _, prop_pf, _ = create_pvt_components(field_model)
    prop_pf = prop_pf.condition(0, 60)

    # Create a list of jet pumps with all combinations of nozzles and throats
    jp_list = BatchPump.jetpump_list(
        nozzle_options, throat_options, knz=knz, ken=ken, kth=kth, kdi=kdi
    )

    # Create a BatchPump object
    batch_pump = BatchPump(
        pwh=surf_pres,
        tsu=form_temp,
        ppf_surf=ppf_surf,
        wellbore=wellbore,
        wellprof=well_profile,
        ipr_su=inflow,
        prop_su=res_mix,
        prop_pf=prop_pf,
        jpump_direction=jpump_direction,
        wellname=wellname,
    )

    # Run the batch simulation
    batch_pump.batch_run(jp_list)

    # Process the results
    try:
        batch_pump.process_results()
        return batch_pump
    except (ValueError, RuntimeError, TypeError) as e:
        error_msg = str(e)
        # Too few converged pump configurations to fit the 3-param exp curve —
        # almost always caused by too-low GOR for the well's PVT. Auto-recover
        # by bumping GOR and re-running.
        if "must not exceed the number of data points" in error_msg:
            current_gor = st.session_state.get("form_gor", "?")
            wellname = st.session_state.get("selected_well", "this well")
            _trigger_gor_reset(
                wellname,
                current_gor,
                reason=(
                    "only "
                    + error_msg.split("data points=")[-1].rstrip(".")
                    + " of N pump configurations converged — "
                    + "not enough to fit the 3-parameter response curve"
                ),
            )
        if "Optimal parameters not found" in error_msg:
            st.error(
                "Could not fit curve to the data. Try selecting more nozzle sizes and throat ratios."
            )
            st.info(
                "The batch run results are still available in the data table, but the curve fitting failed."
            )
            # Return the batch_pump object without curve fitting
            return batch_pump
        else:
            st.error(f"Error processing batch results: {error_msg}")
            return None


def run_power_fluid_range_batch(
    surf_pres,
    form_temp,
    rho_pf,
    power_fluid_min,
    power_fluid_max,
    power_fluid_step,
    wellbore,
    well_profile,
    inflow,
    res_mix,
    nozzle_options,
    throat_options,
    wellname="Test Well",
    field_model=None,
    jpump_direction="reverse",
    ken=0.03,
    kth=0.3,
    kdi=0.4,
    knz=0.01,
):
    """Run a comprehensive batch pump simulation across a range of power fluid pressures.

    Args:
        surf_pres: Surface pressure (psi)
        form_temp: Formation temperature (°F)
        rho_pf: Power fluid density (lbm/ft³) — used for display only
        power_fluid_min: Minimum power fluid pressure (psi)
        power_fluid_max: Maximum power fluid pressure (psi)
        power_fluid_step: Step size for power fluid pressure (psi)
        wellbore: PipeInPipe wellbore object
        well_profile: Well profile object
        inflow: Inflow performance object
        res_mix: Reservoir mixture object
        nozzle_options: List of nozzle sizes to test
        throat_options: List of throat ratios to test
        wellname: Name of the well for display purposes
        field_model: Field model for power fluid properties

    Returns:
        pandas.DataFrame: Comprehensive results across all power fluid pressures
    """
    # Create power fluid properties from field model water
    _, prop_pf, _ = create_pvt_components(field_model)
    prop_pf = prop_pf.condition(0, 60)

    # Create pressure range
    pressure_range = np.arange(
        power_fluid_min, power_fluid_max + power_fluid_step, power_fluid_step
    )

    # Create a list of jet pumps with all combinations of nozzles and throats
    jp_list = BatchPump.jetpump_list(
        nozzle_options, throat_options, knz=knz, ken=ken, kth=kth, kdi=kdi
    )

    all_results = []
    total_combinations = len(pressure_range) * len(jp_list)

    # Create a progress bar
    progress_bar = st.progress(0)
    current_combination = 0

    for pressure in pressure_range:
        # Create a BatchPump object for this pressure
        batch_pump = BatchPump(
            pwh=surf_pres,
            tsu=form_temp,
            ppf_surf=pressure,
            wellbore=wellbore,
            wellprof=well_profile,
            ipr_su=inflow,
            prop_su=res_mix,
            prop_pf=prop_pf,
            jpump_direction=jpump_direction,
            wellname=wellname,
        )

        # Run the batch simulation for this pressure
        batch_pump.batch_run(jp_list, debug=False)

        # Add power fluid pressure to the results
        batch_pump.df["power_fluid_pressure"] = pressure

        # Add the results to our comprehensive dataset
        all_results.append(batch_pump.df)

        # Update progress bar
        current_combination += len(jp_list)
        progress_bar.progress(current_combination / total_combinations)

    # Combine all results into a single dataframe
    comprehensive_df = pd.concat(all_results, ignore_index=True)

    # Clear progress bar
    progress_bar.empty()

    return comprehensive_df


# Well Data Management Functions
def _jp_chars_csv_path() -> str:
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "jp_data", "jp_chars.csv"
    )


@st.cache_data(ttl=3600, show_spinner="Loading well properties from Databricks…")
def load_well_characteristics() -> pd.DataFrame:
    """Load well characteristics from Databricks, with CSV fallback.

    Wraps the library helper fetch_well_props_enriched() (which joins
    vw_prop_mech + vw_prop_resvr and computes JP_TVD from local deviation
    surveys). Falls back to jp_chars.csv if Databricks is unreachable.

    Wells with estimated JP_TVD (no local survey) are flagged in
    st.session_state["wells_missing_surveys"] for the app-level warning.

    Cache: process-wide via @st.cache_data, TTL 1 hour. On Databricks Apps
    (single Streamlit process per container) this is shared across all users.
    """
    from woffl.assembly.databricks_client import fetch_well_props_enriched

    try:
        df, missing = fetch_well_props_enriched()
        if df.empty:
            raise RuntimeError("vw_prop_mech returned no rows")
    except Exception as e:
        st.warning(
            f"Databricks unavailable ({e}); falling back to jp_chars.csv. "
            "Properties may be stale."
        )
        try:
            return pd.read_csv(_jp_chars_csv_path())
        except Exception as csv_err:
            st.error(f"Both Databricks and jp_chars.csv failed: {csv_err}")
            return pd.DataFrame()

    st.session_state["wells_missing_surveys"] = missing
    return df


def get_available_wells():
    """Return list of available well names.

    Returns:
        list: List of well names from jp_chars.csv
    """
    well_chars = load_well_characteristics()
    if not well_chars.empty:
        return ["Custom"] + sorted(well_chars["Well"].tolist())
    return ["Custom"]


def get_well_data(well_name):
    """Get well data for a specific well.

    Args:
        well_name (str): Name of the well

    Returns:
        dict or None: Well data dictionary or None if not found
    """
    well_chars = load_well_characteristics()
    if well_chars.empty:
        return None

    well_data = well_chars[well_chars["Well"] == well_name]
    if well_data.empty:
        return None

    return well_data.iloc[0].to_dict()


def get_well_survey_data(well_name):
    """Load deviation survey CSV for specific well.

    Args:
        well_name (str): Name of the well

    Returns:
        pandas.DataFrame or None: Survey data or None if not found
    """
    try:
        # Get the path to the well survey file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        jp_data_dir = os.path.join(current_dir, "..", "jp_data")
        survey_dir = os.path.join(jp_data_dir, "well_surveys")
        survey_path = os.path.join(survey_dir, f"{well_name} Deviation Survey.csv")

        if os.path.exists(survey_path):
            survey_data = pd.read_csv(survey_path)
            return survey_data
        else:
            return None
    except Exception as e:
        st.warning(f"Could not load survey data for {well_name}: {str(e)}")
        return None


def create_well_profile_from_survey(well_name, jpump_tvd=None, field_model=None):
    """Create WellProfile using actual survey data instead of defaults.

    Args:
        well_name (str): Name of the well
        jpump_tvd (float, optional): Jetpump TVD override
        field_model (str, optional): Field model for fallback

    Returns:
        WellProfile: A WellProfile object using survey data or default model
    """
    # Try to load survey data
    survey_data = get_well_survey_data(well_name)

    if survey_data is not None and not survey_data.empty:
        try:
            # Extract MD and TVD arrays from survey data
            md_list = survey_data["meas_depth"].tolist()
            tvd_list = survey_data["tvd_depth"].tolist()

            # Get well data for jetpump MD if not provided
            if jpump_tvd is None:
                well_data = get_well_data(well_name)
                if well_data:
                    jpump_tvd = well_data.get("JP_TVD")

            # Calculate jetpump MD from TVD if available
            if jpump_tvd is not None:
                jpump_md = float(np.interp(jpump_tvd, tvd_list, md_list))
            else:
                # Use the last point as default
                jpump_md = float(md_list[-1]) if md_list else 5000.0

            # Create WellProfile with survey data
            well_profile = WellProfile(
                md_list=md_list, vd_list=tvd_list, jetpump_md=jpump_md
            )
            return well_profile

        except Exception as e:
            st.warning(
                f"Error creating well profile from survey data: {str(e)}. Using default model."
            )

    # Fallback to default model
    return create_well_profile(field_model=field_model, jpump_tvd=jpump_tvd)
