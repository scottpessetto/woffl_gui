"""Sidebar Input Widgets

Extracts all sidebar parameter collection from the monolithic app.py main()
into a dedicated module. Returns a SimulationParams dataclass and run button state.
"""

import streamlit as st

from woffl.gui.params import NOZZLE_OPTIONS, THROAT_OPTIONS, SimulationParams
from woffl.gui.utils import (
    DEFAULT_TEST_MONTHS,
    get_available_wells,
    get_well_data,
    is_valid_number,
)


# ---------------------------------------------------------------------------
# Helpers for two-tier session state (logical key + widget key)
# ---------------------------------------------------------------------------


def _set_param(key: str, value) -> None:
    """Set both the logical state key and its widget key in session state.

    Streamlit keyed widgets ignore the ``value`` param after first render and
    read from ``st.session_state[key]`` instead, so both must be kept in sync.
    """
    st.session_state[key] = value
    st.session_state[f"{key}_input"] = value


# Widget bounds for programmatically seeded fields, kept in sync with the
# matching _number_input calls below. A seed outside its widget's min/max is
# silently reset to the widget MINIMUM by Streamlit on the next frontend
# round-trip — so every programmatic seed must be clamped into these ranges.
SEED_BOUNDS = {
    "qwf": (10, 6000),
    "pwf": (100, 2500),
    "res_pres": (400, 5000),
    "form_wc": (0.0, 1.0),
    "form_gor": (20, 10000),
    "form_temp": (32, 350),
    "surf_pres": (10, 600),
    "ppf_surf": (800, 5500),
    "jpump_tvd": (2500, 8000),
    "oil_api": (11.0, 39.0),
    "bubble_point": (1001.0, 2999.0),
    "gas_sg": (0.51, 1.19),
    "wat_sg": (0.51, 1.49),
}


def clamp_seed(key: str, value):
    """Clamp a programmatic seed into its widget's bounds (see SEED_BOUNDS)."""
    lo, hi = SEED_BOUNDS.get(key, (None, None))
    if lo is not None:
        value = max(value, lo)
    if hi is not None:
        value = min(value, hi)
    return value


def _seed_param(key: str, raw, default, cast=float) -> None:
    """Seed a sidebar field from well data — NaN-safe and bounds-clamped.

    Databricks chars rows carry missing values as NaN under a *present* key,
    so a plain ``dict.get(key, default)`` never falls back — and ``int(nan)``
    raises. Falls back to ``default`` for None/NaN/non-numeric and clamps the
    result into the widget's bounds.
    """
    try:
        value = cast(raw) if is_valid_number(raw) else cast(default)
    except (TypeError, ValueError):
        value = cast(default)
    _set_param(key, clamp_seed(key, value))


def _number_input(label: str, key: str, default, **kwargs):
    """Render a ``st.number_input`` with two-tier session state.

    On first render (or after a page switch clears the widget key), the widget
    is initialized from the persisted logical state rather than a hardcoded
    default.  After render, the logical state is updated from the widget value.
    """
    widget_key = f"{key}_input"
    if widget_key not in st.session_state:
        st.session_state[widget_key] = st.session_state.get(key, default)
    value = st.number_input(label, key=widget_key, **kwargs)
    st.session_state[key] = value
    return value


def _clear_ipr_state() -> None:
    """Remove cached IPR analysis results from session state."""
    st.session_state.pop("sw_ipr_info", None)
    st.session_state.pop("sw_vogel_coeffs", None)


# ---------------------------------------------------------------------------
# Well data → session state population
# ---------------------------------------------------------------------------


def _populate_pump_from_history(selected_well: str) -> None:
    """Auto-populate sidebar nozzle/throat from current installed pump.

    Looks up JP history (loaded at app startup into ``jp_history_df``) for
    the selected well's most recent installed pump and writes it into the
    sidebar's nozzle/throat session-state keys via ``_set_param``. Silently
    no-ops when JP history is missing or the well has no current pump.
    """
    jp_hist = st.session_state.get("jp_history_df")
    if jp_hist is None:
        return

    from woffl.assembly.jp_history import get_current_pump

    current_pump = get_current_pump(jp_hist, selected_well)
    if current_pump is None:
        return

    nozzle = current_pump.get("nozzle_no")
    throat = current_pump.get("throat_ratio")
    if not nozzle or not throat:
        return

    nozzle_str = str(nozzle).strip()
    throat_str = str(throat).strip().upper()
    if nozzle_str in NOZZLE_OPTIONS:
        _set_param("nozzle_no", nozzle_str)
    if throat_str in THROAT_OPTIONS:
        _set_param("area_ratio", throat_str)


def _update_well_parameters_from_data(
    well_data: dict | None, selected_well: str
) -> None:
    """Update session state parameters when well selection changes."""
    if selected_well == "Custom" or not well_data:
        _clear_ipr_state()
        return

    is_new_well = selected_well != st.session_state.get(
        "last_selected_well_all", "Custom"
    )
    # Memory-gauge apply/clear sets this flag so the sidebar re-runs IPR
    # auto-populate (Vogel fit + pwf/res_pres seed) without also resetting
    # geometry / pump identity, which a full well re-selection would.
    force_ipr_refresh = bool(st.session_state.pop("_force_ipr_refresh", False))

    if is_new_well:
        _seed_param("tubing_od", well_data.get("out_dia"), 4.5)
        _seed_param("tubing_thickness", well_data.get("thick"), 0.5)
        _seed_param("form_temp", well_data.get("form_temp"), 70, cast=int)
        _seed_param("jpump_tvd", well_data.get("JP_TVD"), 4065, cast=int)
        _seed_param("res_pres", well_data.get("res_pres"), 1700, cast=int)

        # Casing from vw_prop_mech (casing_out_dia / casing_inn_dia) — the
        # same source Scott's Tools uses. Resets to the 6.875/0.5 default when
        # absent so the previous well's casing can't leak into this one.
        from woffl.gui.scotts_tools._common import casing_dims_from_chars

        casing_od, casing_thick = casing_dims_from_chars(well_data)
        _set_param("casing_od", round(casing_od, 3))
        _set_param("casing_thickness", round(casing_thick, 3))

        # Field model — write the radio's WIDGET key directly (the radio
        # renders later in this same run, so this is safe). Seeding only
        # field_model_index did nothing: keyed widgets ignore ``index`` once
        # they have state, so Kuparuk wells silently stayed on Schrader.
        is_sch_raw = well_data.get("is_sch")
        is_sch = True if not is_valid_number(is_sch_raw) else bool(is_sch_raw)
        st.session_state["field_model_radio"] = "Schrader" if is_sch else "Kuparuk"
        st.session_state.field_model_index = 0 if is_sch else 1

        # PVT from Databricks vw_prop_resvr. Missing/NaN values reset to the
        # selected field model's preset, so the previous well's PVT can't
        # leak across and the Kuparuk presets (24 API / 2250 Pb) actually
        # apply — these widgets always feed SimulationParams as overrides.
        api_default, pbp_default = (22.0, 1750.0) if is_sch else (24.0, 2250.0)
        _seed_param("oil_api", well_data.get("oil_api"), api_default)
        _seed_param("bubble_point", well_data.get("bubble_point"), pbp_default)
        _seed_param("gas_sg", well_data.get("gas_sg"), 0.65)
        _seed_param("wat_sg", well_data.get("wat_sg"), 1.02)

        # Pad-aware PF surface pressure default. B/G/J wells run on booster
        # pads at ~2200 psi; F is ~2800; the rest run on Schrader at ~3400.
        # Without this the GUI defaulted everyone to 3168 and triggered a
        # PF-mismatch warning on the Solver tab for every non-Schrader well.
        # Future-proof: replace with vw_power_fluid_header once populated.
        from woffl.gui.utils import default_pad_pf, pad_from_mp_name

        pad = pad_from_mp_name(selected_well)
        if pad:
            _set_param("ppf_surf", default_pad_pf(pad))

        _populate_pump_from_history(selected_well)
        _auto_populate_from_ipr(selected_well)
        st.session_state.last_selected_well_all = selected_well
    elif force_ipr_refresh:
        # Gauge override changed — refresh IPR-driven sidebar values
        # against the new BHPs without touching geometry/pump.
        _auto_populate_from_ipr(selected_well)


def _auto_populate_from_ipr(selected_well: str) -> None:
    """Auto-populate sidebar inflow/formation params from well test data.

    Two paths depending on test count:
      - **2+ tests** → run Vogel IPR analysis; populate sidebar from the
        averaged coefficients + most-recent test's surface pressure.
      - **1 test** → no Vogel fit possible. Populate sidebar directly from
        the single test's measured (oil, BHP, WC, GOR, WHP). Reservoir
        pressure stays at the vw_prop_resvr default already set by
        ``_update_well_parameters_from_data``. Lets new wells with one
        recorded test (e.g. H-31) still seed sensible solver inputs.

    Only called once per well selection (inside the is_new_well guard).
    """
    # The Solver tab's IPR-anchor seed tracks what it last applied via
    # sw_ipr_applied_sig_<well>, and the anchor selectbox restores its selection
    # from that marker so it survives a Batch/PF tab detour. Re-seeding here (new
    # well selection or memory-gauge refresh) resets the sidebar to the recent
    # fit, so clear the marker too — that way the anchor selector, the sidebar,
    # and the seed all start consistently at "Most recent" for the (re)seeded
    # well, rather than the sidebar showing recent while the selector still reads
    # "specific".
    st.session_state.pop(f"sw_ipr_applied_sig_{selected_well}", None)

    import pandas as _pd

    def _is_finite(v) -> bool:
        # Use pd.isna so pd.NA, NaN, NaT, and None all return False — the
        # BHP column may carry pd.NA after a "disregard Databricks BHP"
        # toggle, and the older isinstance(float)+math.isnan check misses it.
        if v is None:
            return False
        try:
            return not bool(_pd.isna(v))
        except (TypeError, ValueError):
            return True  # non-NA-checkable scalars (strings, etc.)

    # Route through the central helper so memory-gauge BHP overrides feed
    # into the IPR auto-populate (Vogel fit + sidebar pwf/res_pres seed).
    from woffl.gui.utils import get_well_tests_for_well

    well_tests = get_well_tests_for_well(selected_well)
    if well_tests is None or well_tests.empty:
        _clear_ipr_state()
        return

    floor_map = st.session_state.get("_well_min_gor", {})
    gor_floor = floor_map.get(selected_well, 0)

    if len(well_tests) >= 2:
        try:
            from woffl.assembly.ipr_analyzer import (
                compute_vogel_coefficients,
                estimate_reservoir_pressure,
            )

            merged_with_rp = estimate_reservoir_pressure(well_tests)
            vogel_coeffs = compute_vogel_coefficients(merged_with_rp)

            vogel_usable = (
                vogel_coeffs is not None
                and not vogel_coeffs.empty
                and "Well" in vogel_coeffs.columns
            )
            well_coeffs = (
                vogel_coeffs[vogel_coeffs["Well"] == selected_well]
                if vogel_usable else None
            )
            if well_coeffs is not None and not well_coeffs.empty:
                coeff_row = well_coeffs.iloc[0]
                st.session_state["sw_vogel_coeffs"] = coeff_row.to_dict()

                # Seeds clamped to widget bounds — out-of-range session values
                # get silently reset to the widget minimum by Streamlit.
                _set_param(
                    "form_wc", clamp_seed("form_wc", round(float(coeff_row["form_wc"]), 2))
                )
                _set_param(
                    "form_gor",
                    clamp_seed("form_gor", max(int(coeff_row["fgor"]), gor_floor)),
                )
                _set_param(
                    "qwf",
                    clamp_seed("qwf", int(coeff_row["qwf"] * (1 - coeff_row["form_wc"]))),
                )
                _set_param("pwf", clamp_seed("pwf", int(coeff_row["pwf"])))
                _set_param("res_pres", clamp_seed("res_pres", int(coeff_row["ResP"])))

                recent = well_tests.sort_values("WtDate", ascending=False).iloc[0]
                whp = recent.get("whp")
                if _is_finite(whp):
                    _set_param("surf_pres", clamp_seed("surf_pres", int(whp)))

                num_tests = int(coeff_row["num_tests"])
                most_recent = coeff_row["most_recent_date"]
                date_str = (
                    most_recent.strftime("%Y-%m-%d")
                    if hasattr(most_recent, "strftime")
                    else str(most_recent)
                )
                st.session_state["sw_ipr_info"] = (
                    f"IPR values loaded from {num_tests} well tests "
                    f"(most recent: {date_str})"
                )
                return
            # Vogel returned empty (typically: every test row lacks BHP, e.g.
            # S-pad wells with no coincident gauge). Fall through to the
            # direct-from-test seed path below so the sidebar still picks up
            # WC/GOR/oil/WHP from the latest test — the engineer can then
            # manually tune pwf/res_pres against the modeled-vs-actual gap.
        except Exception:
            # Vogel exploded — fall through to direct-from-test seed too.
            pass

    # Direct-from-test seed path: runs when there's 1 test, or when 2+ tests
    # exist but Vogel couldn't fit (most often because no test has BHP). The
    # `_is_finite(bhp)` guard at the pwf line means no-BHP wells skip pwf
    # cleanly; the engineer keeps the sidebar's prior pwf/res_pres and tunes
    # against the oil-rate-mismatch flag in the hero strip.
    _clear_ipr_state()
    recent = well_tests.sort_values("WtDate", ascending=False).iloc[0]

    oil = recent.get("WtOilVol")
    water = recent.get("WtWaterVol")
    total = recent.get("WtTotalFluid")
    bhp = recent.get("BHP")
    fgor = recent.get("fgor")
    whp = recent.get("whp")

    if _is_finite(water) and _is_finite(total) and float(total) > 0:
        wc = max(0.0, min(1.0, float(water) / float(total)))
        _set_param("form_wc", round(wc, 2))
    if _is_finite(oil):
        _set_param("qwf", clamp_seed("qwf", int(float(oil))))
    if _is_finite(bhp):
        _set_param("pwf", clamp_seed("pwf", int(float(bhp))))
    if _is_finite(fgor):
        _set_param("form_gor", clamp_seed("form_gor", max(int(float(fgor)), gor_floor)))
    if _is_finite(whp):
        _set_param("surf_pres", clamp_seed("surf_pres", int(float(whp))))

    test_date = recent.get("WtDate")
    date_str = (
        test_date.strftime("%Y-%m-%d")
        if hasattr(test_date, "strftime")
        else str(test_date)
    )
    st.session_state["sw_ipr_info"] = (
        f"Sidebar seeded from 1 well test (most recent: {date_str}) — "
        "Vogel IPR fit unavailable until a second test is recorded."
    )


# ---------------------------------------------------------------------------
# Widget rendering helpers
# ---------------------------------------------------------------------------


def _well_selector_key() -> str:
    """Versioned key for the well dropdown. Bumping ``_well_sel_nonce`` makes the
    selectbox a FRESH widget that re-reads its ``index`` from ``selected_well`` —
    the reliable way to advance the dropdown programmatically (the S-Pad review
    queue's Save & next). Popping the widget key alone proved unreliable."""
    return f"well_selector_{st.session_state.get('_well_sel_nonce', 0)}"


def _on_well_change() -> None:
    """Callback function when well selection changes.

    Auto-activates the simulation for any non-Custom well so the user sees
    results immediately without scrolling back up to click Run. Custom mode
    still requires the explicit Run button (sidebar values aren't yet set).
    """
    new_well = st.session_state[_well_selector_key()]
    st.session_state.selected_well = new_well
    if hasattr(st.session_state, "well_data"):
        del st.session_state.well_data
    if new_well != "Custom":
        st.session_state.sw_sim_active = True


def _render_test_lookback_controls() -> None:
    """Well-test lookback window + optional count cap.

    Drives how much test history feeds the IPR fit, the comparison picker, and
    the Vogel auto-populate — everything reads through
    ``utils.get_well_tests_for_well``, which honors these two keys. Defaults:
    6 months, no count cap.

    Note: changing these updates the Solver tab's Model-vs-Actual chart/table
    live (it re-reads tests each render), but does not re-seed the sidebar
    qwf/pwf/ResP fields — those refresh on the next well selection.
    """
    with st.expander("Well Test History", expanded=False):
        _number_input(
            "Lookback (months)",
            "sw_test_months",
            default=DEFAULT_TEST_MONTHS,
            min_value=1,
            max_value=24,
            step=1,
            help=(
                "How far back to pull well tests for this well. Widen this when "
                "a well has too few recent tests to fit a sensible IPR. "
                "Re-fetches from Databricks (cached per window)."
            ),
        )
        _number_input(
            "Max tests (0 = all)",
            "sw_test_count_cap",
            default=0,
            min_value=0,
            max_value=50,
            step=1,
            help=(
                "Cap the number of most-recent tests used. 0 keeps every test "
                "in the lookback window."
            ),
        )


def _render_well_selection(well_filter: list[str] | None = None) -> tuple[str, dict | None]:
    """Render well selection widgets and return selected well + data.

    When ``well_filter`` is provided (the pad-scoped optimization review loop),
    the dropdown is restricted to that list of well names — no "Custom" entry,
    so the review flow always operates on a real, characterized well.
    """
    st.subheader("Well Selection")
    available_wells = list(well_filter) if well_filter else get_available_wells()

    if "selected_well" not in st.session_state:
        st.session_state.selected_well = "Custom"

    selected_well = st.selectbox(
        "Select Well:",
        options=available_wells,
        index=(
            available_wells.index(st.session_state.selected_well)
            if st.session_state.selected_well in available_wells
            else 0
        ),
        help="Choose a well to auto-populate parameters, or select 'Custom' for manual entry",
        key=_well_selector_key(),
        on_change=_on_well_change,
    )

    well_data = None
    if selected_well != "Custom":
        if (
            "well_data" not in st.session_state
            or st.session_state.get("current_well") != selected_well
        ):
            st.session_state.well_data = get_well_data(selected_well)
            st.session_state.current_well = selected_well

        well_data = st.session_state.well_data
        _update_well_parameters_from_data(well_data, selected_well)

        if well_data:
            st.info(f"✅ Loaded data for {selected_well}")
            with st.expander("Well Information"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(
                        f"**Field Model:** {'Schrader' if well_data.get('is_sch', True) else 'Kuparuk'}"
                    )
                    st.write(f"**Tubing OD:** {well_data.get('out_dia', 'N/A')} inches")
                    st.write(
                        f"**Tubing Thickness:** {well_data.get('thick', 'N/A')} inches"
                    )
                    st.write(
                        f"**Reservoir Pressure:** {well_data.get('res_pres', 'N/A')} psi"
                    )
                with col2:
                    st.write(
                        f"**Formation Temp:** {well_data.get('form_temp', 'N/A')} °F"
                    )
                    st.write(f"**Jetpump TVD:** {well_data.get('JP_TVD', 'N/A')} ft")
                    st.write(f"**Jetpump MD:** {well_data.get('JP_MD', 'N/A')} ft")
                ipr_info = st.session_state.get("sw_ipr_info")
                if ipr_info:
                    st.info(ipr_info)
        else:
            st.warning(f"Could not load data for {selected_well}")

        _render_test_lookback_controls()

    return selected_well, well_data


def _render_jetpump_params() -> tuple[str, str, str]:
    """Render the always-visible pump selection (direction, nozzle, throat)."""
    st.subheader("Pump")

    if "jpump_direction" not in st.session_state:
        st.session_state.jpump_direction = "Reverse"
    jpump_direction = st.radio(
        "Circulation Direction",
        options=["Reverse", "Forward"],
        index=["Reverse", "Forward"].index(st.session_state.jpump_direction),
        help="Reverse: power fluid down annulus, production up tubing. Forward: power fluid down tubing, production up annulus.",
        key="jpump_direction_input",
    )
    st.session_state.jpump_direction = jpump_direction

    col1, col2 = st.columns(2)
    with col1:
        # Coerce/validate so a stray value (e.g. a nozzle read back from CSV as
        # the float '10.0') can't crash NOZZLE_OPTIONS.index(...). Tolerate the
        # '10.0' form, else fall back to the default.
        nz = str(st.session_state.get("nozzle_no", "12"))
        if nz not in NOZZLE_OPTIONS:
            try:
                nz = str(int(float(nz)))
            except (TypeError, ValueError):
                pass
            if nz not in NOZZLE_OPTIONS:
                nz = "12"
        st.session_state.nozzle_no = nz
        nozzle_no = st.selectbox(
            "Nozzle",
            NOZZLE_OPTIONS,
            index=NOZZLE_OPTIONS.index(nz),
            key="nozzle_no_input",
        )
        st.session_state.nozzle_no = nozzle_no
    with col2:
        ar = str(st.session_state.get("area_ratio", "B"))
        if ar not in THROAT_OPTIONS:
            ar = "B"
        st.session_state.area_ratio = ar
        area_ratio = st.selectbox(
            "Throat",
            THROAT_OPTIONS,
            index=THROAT_OPTIONS.index(ar),
            key="area_ratio_input",
        )
        st.session_state.area_ratio = area_ratio

    return nozzle_no, area_ratio, jpump_direction.lower()


def _render_loss_coefs() -> tuple[float, float, float]:
    """Render the friction loss coefficients (advanced)."""
    ken = _number_input(
        "Nozzle Loss Coefficient (ken)",
        key="ken",
        default=0.03,
        min_value=0.001,
        max_value=0.40,  # widened to match fric_calibration.KEN_BOUNDS upper
        step=0.005,
        format="%.3f",
    )
    kth = _number_input(
        "Throat Loss Coefficient (kth)",
        key="kth",
        default=0.3,
        min_value=0.05,
        max_value=1.0,
        step=0.005,
        format="%.3f",
    )
    kdi = _number_input(
        "Diffuser Loss Coefficient (kdi)",
        key="kdi",
        default=0.4,
        min_value=0.05,
        max_value=1.0,
        step=0.005,
        format="%.3f",
    )
    return ken, kth, kdi


def _render_pipe_params(well_data: dict | None) -> tuple[float, float, float, float]:
    """Render pipe parameter widgets."""
    auto_help = "Auto-populated from well data" if well_data else None

    tubing_od = _number_input(
        "Tubing Outer Diameter (inches)", "tubing_od", 4.5,
        min_value=2.0, max_value=9.0, step=0.1, format="%.3f", help=auto_help,
    )
    tubing_thickness = _number_input(
        "Tubing Wall Thickness (inches)", "tubing_thickness", 0.5,
        min_value=0.1, max_value=2.0, step=0.1, format="%.3f", help=auto_help,
    )
    casing_od = _number_input(
        "Casing Outer Diameter (inches)", "casing_od", 6.875,
        min_value=4.0, max_value=17.0, step=0.125, format="%.3f",
    )
    casing_thickness = _number_input(
        "Casing Wall Thickness (inches)", "casing_thickness", 0.5,
        min_value=0.1, max_value=2.0, step=0.1, format="%.3f",
    )
    return tubing_od, tubing_thickness, casing_od, casing_thickness


def _render_formation_inflow(
    well_data: dict | None,
) -> tuple[float, int, int, int, int, int]:
    """Render the core inflow + formation widgets (the values most often edited)."""
    auto_help = "Auto-populated from well data" if well_data else None
    ipr_help = (
        "Auto-populated from recent well tests (IPR analysis)" if well_data else None
    )

    qwf = _number_input(
        "Oil Rate at FBHP (qwf, BOPD)", "qwf", 750,
        min_value=10, max_value=6000, step=10, help=ipr_help,
    )
    pwf = _number_input(
        "Flowing BHP @ qwf (pwf, psi)", "pwf", 500,
        min_value=100, max_value=2500, step=10, help=ipr_help,
    )
    pres = _number_input(
        "Reservoir Pressure (psi)", "res_pres", 1700,
        min_value=400, max_value=5000, step=10, help=ipr_help,
    )
    form_wc = _number_input(
        "Water Cut", "form_wc", 0.50,
        min_value=0.0, max_value=1.0, step=0.01, format="%.2f", help=ipr_help,
    )
    form_gor = _number_input(
        "Gas-Oil Ratio (scf/bbl)", "form_gor", 250,
        min_value=20, max_value=10000, step=25, help=ipr_help,
    )
    form_temp = _number_input(
        "Formation Temperature (°F)", "form_temp", 70,
        min_value=32, max_value=350, step=1, help=auto_help,
    )
    return form_wc, form_gor, form_temp, qwf, pwf, pres


def _render_pvt_overrides(
    well_data: dict | None,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Render PVT override widgets (advanced — fall through to field-model preset)."""
    has_pvt = bool(well_data and well_data.get("oil_api") is not None)
    st.caption(
        "Per-well values from Databricks vw_prop_resvr override the field-model "
        "presets (Schrader: 22 API, 1750 Pb, 0.65 gas SG, 1.02 wat SG)."
        if has_pvt
        else "No per-well PVT data — using field-model preset."
    )
    oil_api = _number_input(
        "Oil API", "oil_api", 22.0,
        min_value=11.0, max_value=39.0, step=0.1, format="%.1f",
    )
    bubble_point = _number_input(
        "Bubble Point (psig)", "bubble_point", 1750.0,
        min_value=1001.0, max_value=2999.0, step=10.0, format="%.0f",
    )
    gas_sg = _number_input(
        "Gas Specific Gravity", "gas_sg", 0.65,
        min_value=0.51, max_value=1.19, step=0.01, format="%.2f",
    )
    wat_sg = _number_input(
        "Water Specific Gravity", "wat_sg", 1.02,
        min_value=0.51, max_value=1.49, step=0.01, format="%.2f",
    )
    return oil_api, gas_sg, wat_sg, bubble_point


def _render_pressures() -> tuple[int, int]:
    """Render the two always-visible pressure knobs (PF + surface)."""
    # Bounds match the PF Auto-match search envelope [800, 5500] — narrower
    # widget bounds made Streamlit silently reset an out-of-range solved value
    # to the widget minimum (user saw "Auto-matched 4300 psi", solver ran 1500).
    ppf_surf = _number_input(
        "Power Fluid Surface Pressure (psi)", "ppf_surf", 3168,
        min_value=800, max_value=5500, step=10,
    )
    surf_pres = _number_input(
        "Wellhead Surface Pressure (psi)", "surf_pres", 210,
        min_value=10, max_value=600, step=10,
    )
    return surf_pres, ppf_surf


def _render_geometry(well_data: dict | None) -> tuple[int, float]:
    """Render geometry widgets that mostly auto-populate (TVD + PF density)."""
    auto_help = "Auto-populated from well data" if well_data else None
    jpump_tvd = _number_input(
        "Jetpump TVD (feet)", "jpump_tvd", 4065,
        min_value=2500, max_value=8000, step=10, help=auto_help,
    )
    rho_pf = _number_input(
        "Power Fluid Density (lbm/ft³)", "rho_pf", 62.4,
        min_value=50.0, max_value=70.0, step=0.1,
    )
    return jpump_tvd, rho_pf


def _multiselect(label: str, key: str, options: list[str], default: list[str]) -> list[str]:
    """Two-tier multiselect (logical key + widget key) — same pattern as
    ``_number_input``, so selections survive the widget-state GC when the
    sidebar unmounts (mode switches) instead of snapping back to defaults."""
    widget_key = f"{key}_input"
    if widget_key not in st.session_state:
        st.session_state[widget_key] = [
            v for v in st.session_state.get(key, default) if v in options
        ]
    value = st.multiselect(label, options=options, key=widget_key)
    st.session_state[key] = value
    return value


def _render_batch_params() -> tuple[list[str], list[str], str]:
    """Render batch-run parameters (only used by the Batch Run view)."""
    st.caption("Used by the Batch Run view")
    nozzle_batch_options = _multiselect(
        "Nozzle Sizes to Test",
        "batch_nozzles",
        NOZZLE_OPTIONS,
        ["9", "10", "11", "12", "13", "14", "15"],
    )
    throat_batch_options = _multiselect(
        "Throat Ratios to Test", "batch_throats", THROAT_OPTIONS, ["A", "B", "C", "D"]
    )
    # water_type lives next to the Marginal WC quickfix on the Batch Run
    # page (more discoverable than buried in the sidebar Advanced section).
    # We just read the current value here so SimulationParams stays correct.
    water_type = st.session_state.get("water_type", "total")
    return nozzle_batch_options, throat_batch_options, water_type


def _render_print_section(selected_well: str) -> None:
    """Bottom-of-sidebar PDF report controls.

    Adds engineer name + notes inputs and the Generate Report button. On
    click, sets ``_print_pdf_pending`` so ``run_single_well_page`` can do the
    heavy work (it has the simulation objects already built); the next rerun
    surfaces the download button via the same helper.

    The two-phase flow (set flag → main page generates → rerun → download
    button appears) is needed because the sidebar renders before the main
    page is built, so the sim objects aren't available here. Keeping
    generation in the main page also lets the spinner live in the central
    panel where the user is looking.
    """
    st.divider()
    st.subheader("Print Report")

    st.text_input(
        "Engineer Name (optional)",
        key="print_engineer_name",
        placeholder="Your name",
    )
    st.text_area(
        "Notes (optional)",
        key="print_notes",
        placeholder="Context for the reader (what / why)",
        height=100,
    )

    is_custom = selected_well == "Custom"
    gen_help = (
        "Disabled in Custom mode — select a real well to generate a report."
        if is_custom
        else (
            "Builds a PDF with inputs, solver results, the IPR chart, and the "
            "batch performance plot. Runs the batch sweep, so the first "
            "generation takes ~30 seconds."
        )
    )
    if st.button(
        "Generate Report",
        key="print_generate_btn",
        type="primary",
        use_container_width=True,
        disabled=is_custom,
        help=gen_help,
    ):
        st.session_state["_print_pdf_pending"] = True
        # Wipe stale output so the download button doesn't briefly show
        # last run's file while the new one is generating. Clearing the
        # downloaded flag lets the auto-download iframe re-fire after the
        # new generation completes.
        st.session_state.pop("_print_pdf_bytes", None)
        st.session_state.pop("_print_pdf_filename", None)
        st.session_state.pop("_print_pdf_downloaded", None)

    pdf_bytes = st.session_state.get("_print_pdf_bytes")
    pdf_filename = st.session_state.get("_print_pdf_filename")
    if pdf_bytes and pdf_filename:
        # Auto-trigger the browser download on the first render after
        # generation. ``st.download_button`` requires a user click — we
        # mimic the click via a tiny embedded iframe whose <script> tag
        # clicks a hidden <a download> link. Only fires once per
        # generation (guarded by ``_print_pdf_downloaded``), so changing
        # an unrelated sidebar slider later doesn't re-download.
        if not st.session_state.get("_print_pdf_downloaded"):
            _auto_download(pdf_bytes, pdf_filename)
            st.session_state["_print_pdf_downloaded"] = True

        # Manual fallback — useful when the auto-download is blocked by
        # the browser, or when the user wants to grab the file again.
        st.download_button(
            "↓ Download Again",
            data=pdf_bytes,
            file_name=pdf_filename,
            mime="application/pdf",
            use_container_width=True,
            key="print_download_btn",
        )


def _auto_download(pdf_bytes: bytes, filename: str) -> None:
    """Single-click PDF download — delegates to the shared implementation
    in woffl.gui.components.download."""
    from woffl.gui.components.download import autodownload

    autodownload(pdf_bytes, filename, "application/pdf", "woffl_auto_dl")


def _render_power_fluid_range_params() -> tuple[int, int, int]:
    """Render PF-range parameters (only used by the Power Fluid Range view).

    Two-tier keys so edits survive sidebar unmounts (mode switches) —
    keyless widgets reset to their ``value=`` defaults on every remount.
    """
    st.caption("Used by the Power Fluid Range view")
    power_fluid_min = _number_input(
        "Min Power Fluid Pressure (psi)", "power_fluid_min", 1800,
        min_value=1000, max_value=5000, step=100,
    )
    power_fluid_max = _number_input(
        "Max Power Fluid Pressure (psi)", "power_fluid_max", 3600,
        min_value=1000, max_value=5000, step=100,
    )
    power_fluid_step = _number_input(
        "Pressure Step (psi)", "power_fluid_step", 200,
        min_value=50, max_value=500, step=50,
    )
    return power_fluid_min, power_fluid_max, power_fluid_step


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_sidebar(well_filter: list[str] | None = None) -> tuple[bool, SimulationParams]:
    """Render the complete sidebar and collect all parameters.

    Inputs are grouped from most-edited (top) to least-edited (Advanced expander):
      1. Well selection — auto-runs the simulation on change
      2. Field model + Pump (always visible)
      3. Pressures — the main "what-if" knobs
      4. Inflow & Formation (expanded by default)
      5. Geometry (collapsed; auto-populated)
      6. Advanced — loss coefs, PVT overrides, batch ranges, PF range (collapsed)

    ``well_filter`` (optional) restricts the well dropdown to a subset — used by
    the pad-scoped optimization review loop to scope the picker to one pad.
    """
    with st.sidebar:
        run_button = st.button(
            "Re-run Simulation",
            help=(
                "Selecting a well runs automatically. Use this for Custom mode "
                "or to force a re-run after editing inputs."
            ),
            use_container_width=True,
        )

        # Well selection — auto-run hooks fire from its on_change
        selected_well, well_data = _render_well_selection(well_filter)

        # Field model
        if "field_model_index" not in st.session_state:
            st.session_state.field_model_index = 0
        field_model = st.radio(
            "Field Model",
            options=["Schrader", "Kuparuk"],
            index=st.session_state.field_model_index,
            horizontal=True,
            key="field_model_radio",
        )
        st.session_state.field_model_index = ["Schrader", "Kuparuk"].index(field_model)

        # Pump selection — direction, nozzle, throat
        nozzle_no, area_ratio, jpump_direction = _render_jetpump_params()

        # Pressures — most-tweaked sidebar values
        st.subheader("Pressures")
        surf_pres, ppf_surf = _render_pressures()

        # Inflow & Formation — auto-populated from IPR but commonly reviewed
        with st.expander("Inflow & Formation", expanded=True):
            (
                form_wc,
                form_gor,
                form_temp,
                qwf,
                pwf,
                pres,
            ) = _render_formation_inflow(well_data)

        # Geometry — auto-populated, rarely edited
        with st.expander("Geometry", expanded=False):
            jpump_tvd, rho_pf = _render_geometry(well_data)
            (
                tubing_od,
                tubing_thickness,
                casing_od,
                casing_thickness,
            ) = _render_pipe_params(well_data)

        # Advanced — loss coefficients, PVT, marginal WC, view-specific knobs
        with st.expander("Advanced", expanded=False):
            st.markdown("**Loss Coefficients**")
            ken, kth, kdi = _render_loss_coefs()

            st.markdown("**PVT Overrides**")
            oil_api, gas_sg, wat_sg, bubble_point = _render_pvt_overrides(well_data)

            st.markdown("**Field**")
            marginal_watercut = _number_input(
                "Marginal Watercut", "marginal_watercut", 0.94,
                min_value=0.0, max_value=1.0, step=0.01, format="%.2f",
                help="Economic threshold for water handling in the field",
            )

            st.markdown("**Batch Run**")
            nozzle_batch_options, throat_batch_options, water_type = (
                _render_batch_params()
            )

            st.markdown("**Power Fluid Range**")
            power_fluid_min, power_fluid_max, power_fluid_step = (
                _render_power_fluid_range_params()
            )

        # Print Report section — bottom of the sidebar, below all inputs.
        _render_print_section(selected_well)

    params = SimulationParams(
        nozzle_no=nozzle_no,
        area_ratio=area_ratio,
        ken=ken,
        kth=kth,
        kdi=kdi,
        jpump_direction=jpump_direction,
        tubing_od=tubing_od,
        tubing_thickness=tubing_thickness,
        casing_od=casing_od,
        casing_thickness=casing_thickness,
        form_wc=form_wc,
        form_gor=form_gor,
        form_temp=form_temp,
        field_model=field_model,
        oil_api=oil_api,
        gas_sg=gas_sg,
        wat_sg=wat_sg,
        bubble_point=bubble_point,
        surf_pres=surf_pres,
        jpump_tvd=jpump_tvd,
        rho_pf=rho_pf,
        ppf_surf=ppf_surf,
        qwf=qwf,
        pwf=pwf,
        pres=pres,
        nozzle_batch_options=nozzle_batch_options,
        throat_batch_options=throat_batch_options,
        water_type=water_type,
        marginal_watercut=marginal_watercut,
        power_fluid_min=power_fluid_min,
        power_fluid_max=power_fluid_max,
        power_fluid_step=power_fluid_step,
        selected_well=selected_well,
        well_data=well_data,
    )

    return run_button, params
