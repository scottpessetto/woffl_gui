"""Unified Optimization Workflow — linear 4-step flow from well selection to results."""

import streamlit as st

# Step labels. Key 2.5 is the optional pre-calibration step inserted
# between IPR review and Configure & Run. Using a float key avoids
# renumbering downstream step state.
STEPS = {
    1: "Select Wells",
    2: "Review IPR",
    2.5: "Pre-Calibrate",
    3: "Configure & Run",
    4: "Results",
}


def _clear_downstream(from_step):
    """Clear all uw_ session state keys for steps >= from_step.

    Accepts ints or 2.5 to match the STEPS dict.
    """
    step_keys = {
        2: [
            "uw_vogel_coeffs",
            "uw_ipr_curves",
            "uw_merged_with_rp",
            "uw_excluded_wells",
            "uw_template_df",
            "uw_well_configs",
            # CSV-override state must die with the configs it overrode —
            # otherwise a Step-1 reload + same-file re-upload is silently
            # skipped by the signature check.
            "uw_csv_override_sig",
            "uw_csv_pristine",
        ],
        2.5: [
            "uw_precal_results",
            "uw_precal_skipped",
        ],
        3: [
            "uw_optimizer",
            "uw_opt_results",
            "uw_calibration_results",
            "uw_reconciliation",
            "uw_actual_oil_map",
            "uw_actual_pf_map",
            "uw_actual_bhp_map",
            "uw_current_jp_map",
        ],
        4: [],
    }
    # Sort numerically; sorted([1, 2, 2.5, 3, 4]) handles mixed ints/floats.
    for step in sorted(step_keys):
        if step >= from_step:
            for key in step_keys.get(step, []):
                st.session_state.pop(key, None)


def _render_step_indicator(current_step, max_reached):
    """Render a horizontal step indicator.

    Labels are rendered by 1-based position in STEPS so the user sees
    "Step 1...Step 5" even though internal keys are 1, 2, 2.5, 3, 4.
    """
    cols = st.columns(len(STEPS))
    for i, (step_num, label) in enumerate(STEPS.items()):
        display_idx = i + 1  # 1-based UI label
        with cols[i]:
            if step_num == current_step:
                st.markdown(f"**:blue[Step {display_idx}: {label}]**")
            elif step_num <= max_reached:
                if st.button(
                    f"Step {display_idx}: {label}",
                    key=f"uw_nav_{step_num}",
                    use_container_width=True,
                ):
                    st.session_state["uw_current_step"] = step_num
                    st.rerun()
            else:
                st.markdown(f":gray[Step {display_idx}: {label}]")


def _render_sidebar():
    """Render the (now minimal) workflow sidebar.

    Per-step config — IPR knobs, power fluid constraints, algorithm choice,
    pump-option multiselects, PF sensitivity ranges — moved into the steps
    themselves so the controls sit next to the actions they affect. The
    sidebar is now just app-wide cache management.
    """
    with st.sidebar:
        st.header("Workflow")
        st.caption(
            "Step-specific parameters live inline within each step's main " "panel."
        )

        st.divider()
        st.subheader("Data Cache")
        # A BUTTON, not a checkbox: the old checkbox cleared the shared
        # process-wide caches on EVERY rerun while ticked, degrading every
        # user of the app to cold Databricks queries until it was unticked.
        if st.button(
            "Refresh Databricks data",
            key="uw_force_refresh_btn",
            help=(
                "Clear all cached well-test / BHP / well-property data and "
                "re-query Databricks on the next action."
            ),
            use_container_width=True,
        ):
            from woffl.gui.utils import fetch_all_well_tests, load_well_characteristics
            from woffl.gui.well_test_cache import (
                _cached_bhp_query,
                _cached_mpu_well_names,
                _cached_well_test_query,
            )

            _cached_bhp_query.clear()
            _cached_well_test_query.clear()
            _cached_mpu_well_names.clear()
            fetch_all_well_tests.clear()
            load_well_characteristics.clear()
            try:
                from woffl.gui.scotts_tools._common import (
                    fetch_well_tests,
                    fetch_well_tests_raw,
                )

                fetch_well_tests.clear()
                fetch_well_tests_raw.clear()
            except Exception:
                pass
            st.success("Caches cleared — data refreshes on the next action.")


def run_workflow_page():
    """Main entry point for the unified optimization workflow."""
    st.title("Optimization Workflow")
    st.caption("Select wells, review IPR, configure optimization, view results.")

    # Initialize step state
    if "uw_current_step" not in st.session_state:
        st.session_state["uw_current_step"] = 1
    if "uw_max_step_reached" not in st.session_state:
        st.session_state["uw_max_step_reached"] = 1

    current_step = st.session_state["uw_current_step"]
    max_reached = st.session_state["uw_max_step_reached"]

    # Render sidebar and step indicator
    _render_sidebar()
    _render_step_indicator(current_step, max_reached)
    st.divider()

    # Route to the current step
    if current_step == 1:
        from woffl.gui.workflow_steps.step1_select_wells import render_step1

        render_step1()
    elif current_step == 2:
        from woffl.gui.workflow_steps.step2_review_ipr import render_step2

        render_step2()
    elif current_step == 2.5:
        from woffl.gui.workflow_steps.step2_5_precalibrate import render_step2_5

        render_step2_5()
    elif current_step == 3:
        from woffl.gui.workflow_steps.step3_configure_optimize import render_step3

        render_step3()
    elif current_step == 4:
        from woffl.gui.workflow_steps.step4_results import render_step4

        render_step4()
