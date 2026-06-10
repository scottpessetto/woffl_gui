"""Step 3: Configure & Run Optimization — settings review, CSV override, run."""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from woffl.assembly.calibration import (
    compute_field_calibration_summary,
    run_calibration,
)
from woffl.assembly.network_optimizer import (
    NetworkOptimizer,
    PowerFluidConstraint,
    load_wells_from_dataframe,
)
from woffl.assembly.optimization_algorithms import optimize
from woffl.gui.optimization_viz import create_calibration_chart
from woffl.gui.params import NOZZLE_OPTIONS, THROAT_OPTIONS
from woffl.gui.utils import is_valid_number
from woffl.gui.workflow_page import _clear_downstream


_DEFAULT_POPS_PADS = ("E", "F", "H", "I", "M", "S")


@st.cache_data(ttl=3600, show_spinner=False)
def _auto_marginal_wc(pops_tuple: tuple[str, ...],
                       overrides_tuple: tuple[str, ...]) -> dict | None:
    """Compute the field marginal WC from current Well-Sort data, cached 1 h.

    Hash key is the POPS list + per-well overrides so the cached value
    invalidates correctly when the user changes the POPS settings on the
    Well Sort tab. Returns None on any data-fetch failure so the caller
    can fall back to the existing 0.94 default.
    """
    try:
        from woffl.assembly.well_sort_client import (
            build_online_table,
            classify_wells,
            compute_field_marginal_wc,
        )

        # Use the CACHED Well-Sort fetchers (warmed by the app's startup
        # prefetch thread) — the raw well_sort_client functions re-fired
        # five sequential warehouse queries on every cache expiry.
        from woffl.gui.scotts_tools.well_sort import (
            _cached_producer_catalog,
            _cached_producers,
            _cached_recent_tests,
            _cached_shut_in_history,
            _cached_xv_status,
        )

        producers = _cached_producers()
        if not producers:
            return None
        shut_in = _cached_shut_in_history()
        tests = _cached_recent_tests(days=180)
        catalog = _cached_producer_catalog()
        xv = _cached_xv_status()
        online_set, _ = classify_wells(
            producers, shut_in, xv_df=xv, trust_xv=True,
        )
        online_df = build_online_table(
            tests, shut_in, producers, mode="allocated",
            xv_df=xv, online_wells=online_set, catalog_df=catalog,
        )
        return compute_field_marginal_wc(
            online_df,
            set(pops_tuple),
            pops_overrides={w: True for w in overrides_tuple},
        )
    except Exception:
        return None


def _on_marginal_wc_change() -> None:
    """Mark marginal_watercut as user-touched so the auto-fill stops firing."""
    st.session_state["uw_marginal_wc_touched"] = True


def _build_actual_maps(opt_wells: list[str], test_df) -> tuple[dict, dict, dict]:
    """Build actual oil/PF/BHP maps from well test data."""
    actual_oil_map = {}
    actual_pf_map = {}
    actual_bhp_map = {}
    if test_df is None or test_df.empty:
        return actual_oil_map, actual_pf_map, actual_bhp_map

    well_tests = test_df[test_df["well"].isin(opt_wells)].copy()
    for well in opt_wells:
        wt = well_tests[well_tests["well"] == well].sort_values(
            "WtDate", ascending=False
        )
        if wt.empty:
            continue
        row_data = wt.iloc[0]
        if "WtOilVol" in wt.columns and is_valid_number(row_data["WtOilVol"]):
            actual_oil_map[well] = row_data["WtOilVol"]
        if "lift_wat" in wt.columns and is_valid_number(row_data["lift_wat"]):
            actual_pf_map[well] = row_data["lift_wat"]
        if "BHP" in wt.columns and is_valid_number(row_data["BHP"]):
            actual_bhp_map[well] = row_data["BHP"]

    return actual_oil_map, actual_pf_map, actual_bhp_map


def _build_current_jp_map(opt_wells: list[str], jp_hist) -> dict[str, tuple[str, str]]:
    """Build map of current JP configs from JP history."""
    from woffl.assembly.jp_history import get_current_pump

    current_jp_map = {}
    for well in opt_wells:
        current = get_current_pump(jp_hist, well)
        if current and current["nozzle_no"] and current["throat_ratio"]:
            current_jp_map[well] = (current["nozzle_no"], current["throat_ratio"])
    return current_jp_map


def render_step3():
    st.subheader("Step 3: Configure & Run Optimization")

    well_configs = st.session_state.get("uw_well_configs", [])
    if not well_configs:
        st.warning("No well configurations available. Go back to Step 2.")
        return

    # --- CSV override upload ---
    with st.expander("Upload Modified CSV (optional)"):
        st.caption("Override well parameters by uploading an edited optimization template.")
        override_file = st.file_uploader(
            "Upload CSV Override",
            type=["csv"],
            key="uw_csv_override",
        )
        if override_file is not None:
            # Process the file ONCE (keyed on its identity). Re-processing on
            # every rerun re-queried Databricks each interaction and silently
            # rebuilt the configs — discarding accepted pre-calibration.
            file_sig = f"{override_file.name}:{override_file.size}"
            if st.session_state.get("uw_csv_override_sig") != file_sig:
                try:
                    override_df = pd.read_csv(override_file)
                    new_configs = load_wells_from_dataframe(override_df)
                    # New configs invalidate pre-calibration AND any results.
                    _clear_downstream(2.5)
                    st.session_state["uw_well_configs"] = new_configs
                    st.session_state["uw_template_df"] = override_df
                    st.session_state["uw_csv_override_sig"] = file_sig
                    well_configs = new_configs
                    st.success(f"Overrode with {len(new_configs)} wells from CSV")
                except Exception as e:
                    st.error(f"Error parsing override CSV: {e}")
            else:
                st.caption(
                    f"Override CSV **{override_file.name}** is loaded "
                    f"({len(well_configs)} wells). Remove the file to revert."
                )

    # --- Wells summary ---
    st.write(f"**{len(well_configs)} wells** ready for optimization")
    with st.expander("Well Configurations"):
        rows = []
        for wc in well_configs:
            rows.append(
                {
                    "Well": wc.well_name,
                    "Res Pres": f"{wc.res_pres:.0f}",
                    "WC": f"{wc.form_wc:.0%}",
                    "Qwf (BLPD)": f"{wc.qwf:.0f}",
                    "Field": wc.field_model,
                    "JP TVD": f"{wc.jpump_tvd:.0f}",
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # --- Inline configuration ---
    # All optimization knobs live here so the user sees what's about to run.
    # Defaults are seeded into session_state on first paint so the Run button
    # below can read them without a separate fallback.
    st.markdown("### Configuration")

    # When Pad scope is active (set on Step 1), pre-fill Total PF from the
    # pad's pump limit. Source of truth is the editable value the user can
    # set on the Marginal WC tab (_pad_pump_limit_<pad>), falling back to
    # PUMP_LIMIT_PRESETS so the field works even without a Marginal-WC visit.
    # We seed `uw_total_pf` BEFORE rendering the number_input — Streamlit
    # forbids writes to a widget's key after it renders. Re-seed only when
    # the pad changes so the user can still hand-tune the value within a
    # single pad-mode session.
    # Read the scope through the shadow-aware helper — the Step-1 widget keys
    # (uw_scope / uw_pad_scope_pad) are GC'd the moment Step 1 stops
    # rendering, which made Pad mode silently degrade to Field-wide here.
    from woffl.gui.workflow_steps.step1_select_wells import _active_pad_scope

    pad_scope = _active_pad_scope()
    pad_limit_for_banner = None
    if pad_scope:
        from woffl.gui.scotts_tools.well_sort import PUMP_LIMIT_PRESETS

        pad_limit_for_banner = int(
            st.session_state.get(
                f"_pad_pump_limit_{pad_scope}",
                PUMP_LIMIT_PRESETS.get(pad_scope, 0),
            )
        )
        last_seeded = st.session_state.get("_uw_pad_scope_seeded_pad")
        if last_seeded != pad_scope:
            st.session_state["uw_total_pf"] = pad_limit_for_banner
            st.session_state["uw_total_pf_input"] = pad_limit_for_banner
            st.session_state["_uw_pad_scope_seeded_pad"] = pad_scope
        st.info(
            f"**Pad mode: {pad_scope}-Pad** — PF constraint pre-filled from "
            f"the pad's pump limit (**{pad_limit_for_banner:,} BWPD**). "
            "Edit below to override for this run; change the preset on the "
            "Marginal WC tab to make it stick."
        )
    else:
        st.session_state.pop("_uw_pad_scope_seeded_pad", None)

    # Two-tier state (logical uw_* + widget uw_*_input) for every config
    # widget: the old self-referential `value=get(key), key=key` pattern
    # did NOT survive widget-state GC — a detour to Step 4 and back reset
    # Total PF / pressure / density to defaults while the on-screen results
    # were still computed from the user's settings.
    def _two_tier_number(label: str, key: str, default, cast, **kwargs):
        widget_key = f"{key}_input"
        if widget_key not in st.session_state:
            st.session_state[widget_key] = cast(st.session_state.get(key, default))
        val = st.number_input(label, key=widget_key, **kwargs)
        st.session_state[key] = val
        return val

    pf_col1, pf_col2, pf_col3 = st.columns(3)
    with pf_col1:
        total_pf = _two_tier_number(
            "Total PF (bbl/day)",
            "uw_total_pf",
            30000,
            int,
            min_value=0,
            step=500,
            help=(
                f"PF capacity for {pad_scope}-Pad pump (pre-filled from preset)."
                if pad_scope
                else "Total power fluid available across all wells."
            ),
        )
    with pf_col2:
        pf_pressure = _two_tier_number(
            "PF Pressure (psi)",
            "uw_pf_pressure",
            3168,
            int,
            min_value=1000,
            max_value=5000,
            step=100,
        )
    with pf_col3:
        rho_pf = _two_tier_number(
            "PF Density (lbm/ft³)",
            "uw_rho_pf",
            62.4,
            float,
            min_value=50.0,
            max_value=70.0,
            step=0.1,
        )

    # Auto-fill marginal_watercut from Well Sort data (POPS-aware MAX of
    # online wells). Must run BEFORE the number_input widget — Streamlit
    # forbids writing to a widget's key after the widget renders.
    pops_pads = tuple(sorted(
        st.session_state.get("well_sort_pops_pads", _DEFAULT_POPS_PADS) or ()
    ))
    pops_overrides = tuple(sorted(
        st.session_state.get("well_sort_pops_force_true", []) or ()
    ))
    auto = _auto_marginal_wc(pops_pads, pops_overrides)
    if auto and not st.session_state.get("uw_marginal_wc_touched", False):
        st.session_state["uw_marginal_wc"] = float(auto["wc"])
        st.session_state["uw_marginal_wc_input"] = float(auto["wc"])

    algo_col1, algo_col2, algo_col3 = st.columns(3)
    with algo_col1:
        # index restores from the shadow after widget-state GC (Streamlit
        # ignores index when the widget state survived).
        opt_method = st.selectbox(
            "Algorithm",
            ["milp", "mckp"],
            index=1 if st.session_state.get("_uw_opt_method") == "mckp" else 0,
            key="uw_opt_method",
            help=(
                "MILP: Optimal via scipy linear programming. "
                "MCKP: Multi-choice knapsack via OR-Tools CP-SAT."
            ),
        )
        st.session_state["_uw_opt_method"] = opt_method
    with algo_col2:
        marginal_wc = _two_tier_number(
            "Marginal Watercut",
            "uw_marginal_wc",
            0.94,
            float,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            format="%.2f",
            on_change=_on_marginal_wc_change,
            help=(
                "Worst well online — for POPS pads put to 100% if room "
                "on pump for water handling. NOTE: currently informational "
                "— the optimizer does not yet enforce this threshold."
            ),
        )
        if auto:
            tag = "POPS" if auto["is_pops"] else "non-POPS"
            st.caption(
                f"Auto: {auto['wc']:.3f} from {auto['well']} "
                f"({auto['pad']} pad, {tag})"
            )
            if st.session_state.get("uw_marginal_wc_touched", False):
                if st.button(
                    "Re-auto-fill from Well Sort",
                    key="uw_marginal_wc_reauto",
                    help="Discards your manual edit and refills from Well Sort.",
                ):
                    st.session_state.pop("uw_marginal_wc_touched", None)
                    st.session_state["uw_marginal_wc"] = float(auto["wc"])
                    st.session_state.pop("uw_marginal_wc_input", None)
                    st.rerun()
        else:
            st.caption("Auto-fill unavailable — using default 0.94.")
    with algo_col3:
        has_jp_history = "jp_history_df" in st.session_state
        use_calibration = st.checkbox(
            "Apply Calibration",
            value=bool(
                st.session_state.get(
                    "uw_use_calibration",
                    st.session_state.get("_uw_use_calibration", True),
                )
            ),
            key="uw_use_calibration",
            help=(
                "Scale predictions using model-vs-actual on current pump "
                "configs. Disabled when no JP history is loaded."
            ),
            disabled=not has_jp_history,
        )
        st.session_state["_uw_use_calibration"] = use_calibration

    with st.expander("Pump Options to Test", expanded=False):
        # default= restores from the shadow keys after widget-state GC; the
        # old self-referential default (reading the widget's own key) fell
        # back to hardcoded defaults on every step detour.
        nozzle_opts = st.multiselect(
            "Nozzle Sizes",
            NOZZLE_OPTIONS,
            default=st.session_state.get(
                "uw_nozzle_opts",
                st.session_state.get(
                    "_uw_nozzle_opts", ["8", "9", "10", "11", "12", "13", "14"]
                ),
            ),
            key="uw_nozzle_opts",
        )
        st.session_state["_uw_nozzle_opts"] = list(nozzle_opts)
        throat_opts = st.multiselect(
            "Throat Ratios",
            THROAT_OPTIONS,
            default=st.session_state.get(
                "uw_throat_opts",
                st.session_state.get("_uw_throat_opts", ["X", "A", "B", "C", "D"]),
            ),
            key="uw_throat_opts",
        )
        st.session_state["_uw_throat_opts"] = list(throat_opts)

    if not nozzle_opts or not throat_opts:
        st.warning(
            "Select at least one nozzle size and one throat ratio above."
        )
        return

    # --- Run optimization ---
    if st.button(
        "Run Optimization",
        type="primary",
        use_container_width=True,
        key="uw_run_optimization",
    ):
        try:
            wells = well_configs
            opt_well_names = [w.well_name for w in wells]

            # Build current JP map and inject into options
            effective_nozzles = list(nozzle_opts)
            effective_throats = list(throat_opts)
            jp_hist = st.session_state.get("jp_history_df")
            current_jp_map = {}
            has_jp_history = jp_hist is not None

            if has_jp_history and use_calibration:
                current_jp_map = _build_current_jp_map(opt_well_names, jp_hist)
                for nozzle, throat in current_jp_map.values():
                    if nozzle not in effective_nozzles:
                        effective_nozzles.append(nozzle)
                    if throat not in effective_throats:
                        effective_throats.append(throat)

            # Create optimizer
            pf_constraint = PowerFluidConstraint(
                total_rate=total_pf, pressure=pf_pressure, rho_pf=rho_pf
            )
            optimizer = NetworkOptimizer(
                wells=wells,
                power_fluid=pf_constraint,
                nozzle_options=effective_nozzles,
                throat_options=effective_throats,
                marginal_watercut=marginal_wc,
            )

            # Batch simulations
            st.write("### Running Batch Simulations")
            progress_bar = st.progress(0)
            status_text = st.empty()

            def progress_callback(current, total, well_name):
                if total > 0:
                    progress_bar.progress(current / total)
                    status_text.text(f"Processing {well_name}... ({current}/{total})")

            # Per-well parallelism, capped by WOFFL_MAX_WORKERS (2 on
            # Databricks, higher locally) — this loop is the workflow's
            # dominant runtime and the wells are independent.
            from woffl.gui.scotts_tools._common import worker_ceiling

            optimizer.run_all_batch_simulations(
                progress_callback, max_workers=worker_ceiling()
            )
            progress_bar.empty()
            status_text.empty()
            st.success(f"Completed batch simulations for {len(wells)} wells")

            # Actual maps — built ONCE and stored unconditionally below, so a
            # re-run with calibration off can't leave a previous run's maps
            # feeding Step 4's "Current vs Optimized" comparison.
            all_tests = st.session_state.get("all_well_tests_df")
            actual_oil_map, actual_pf_map, actual_bhp_map = _build_actual_maps(
                opt_well_names, all_tests
            )

            # Calibration
            calibration_results = None
            if use_calibration and current_jp_map and has_jp_history:
                if actual_oil_map:
                    calibration_results = run_calibration(
                        optimizer,
                        actual_oil_map,
                        actual_pf_map,
                        actual_bhp_map,
                        current_jp_map,
                    )
                    optimizer.set_calibration(calibration_results)

                    if calibration_results:
                        st.write("### Model Calibration")
                        summary = compute_field_calibration_summary(calibration_results)

                        cal_col1, cal_col2, cal_col3 = st.columns(3)
                        cal_col1.metric(
                            "Calibrated Wells",
                            f"{summary['num_calibrated']}/{len(wells)}",
                        )
                        cal_col2.metric(
                            "Median Factor", f"{summary['median_factor']:.2f}"
                        )
                        cal_col3.metric(
                            "Mean Factor", f"{summary['mean_factor']:.2f}"
                        )

                        fig_cal = create_calibration_chart(calibration_results)
                        st.pyplot(fig_cal)
                        plt.close()

            # Run optimization
            st.write("### Running Optimization")
            with st.spinner(f"Running {opt_method} optimization..."):
                results = optimize(optimizer, method=opt_method)

            if not results:
                # Drop the PREVIOUS run's results too — leaving them made
                # "Previous optimization results available" below present
                # stale numbers as if they matched the inputs on screen.
                for k in ("uw_opt_results", "uw_optimizer", "uw_calibration_results"):
                    st.session_state.pop(k, None)
                st.warning(
                    "Optimization did not produce viable results. "
                    "Try adjusting constraints or pump options."
                )
                return

            st.success(f"Optimization complete! Allocated pumps to {len(results)} wells")

            # Store results and advance to Step 4
            _clear_downstream(4)
            st.session_state["uw_optimizer"] = optimizer
            st.session_state["uw_opt_results"] = results
            st.session_state["uw_calibration_results"] = calibration_results
            st.session_state["uw_current_jp_map"] = current_jp_map
            # Actual maps stored unconditionally (computed above) — keyed to
            # THIS run's configured wells.
            st.session_state["uw_actual_oil_map"] = actual_oil_map
            st.session_state["uw_actual_pf_map"] = actual_pf_map
            st.session_state["uw_actual_bhp_map"] = actual_bhp_map

            st.session_state["uw_current_step"] = 4
            st.session_state["uw_max_step_reached"] = max(
                st.session_state.get("uw_max_step_reached", 3), 4
            )
            st.rerun()

        except Exception as e:
            st.error(f"Error during optimization: {str(e)}")
            st.exception(e)

    # Show previous results if they exist (user navigated back)
    if "uw_opt_results" in st.session_state:
        st.info("Previous optimization results available.")
        if st.button("View Results →", key="uw_step3_to_results"):
            st.session_state["uw_current_step"] = 4
            st.rerun()
