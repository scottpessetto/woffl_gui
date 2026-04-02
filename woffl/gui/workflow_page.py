"""Unified Optimization Workflow — linear 4-step flow from well selection to results."""

import streamlit as st

from woffl.gui.params import NOZZLE_OPTIONS, THROAT_OPTIONS

# Step labels
STEPS = {
    1: "Select Wells",
    2: "Review IPR",
    3: "Configure & Run",
    4: "Results",
}


def _clear_downstream(from_step: int):
    """Clear all uw_ session state keys for steps >= from_step."""
    step_keys = {
        2: [
            "uw_vogel_coeffs",
            "uw_ipr_curves",
            "uw_merged_with_rp",
            "uw_excluded_wells",
            "uw_template_df",
            "uw_well_configs",
        ],
        3: [
            "uw_optimizer",
            "uw_opt_results",
            "uw_calibration_results",
            "uw_actual_oil_map",
            "uw_actual_pf_map",
            "uw_actual_bhp_map",
            "uw_current_jp_map",
        ],
        4: [],
    }
    for step in range(from_step, 5):
        for key in step_keys.get(step, []):
            st.session_state.pop(key, None)


def _render_step_indicator(current_step: int, max_reached: int):
    """Render a horizontal step indicator."""
    cols = st.columns(len(STEPS))
    for i, (step_num, label) in enumerate(STEPS.items()):
        with cols[i]:
            if step_num == current_step:
                st.markdown(
                    f"**:blue[Step {step_num}: {label}]**"
                )
            elif step_num <= max_reached:
                if st.button(
                    f"Step {step_num}: {label}",
                    key=f"uw_nav_{step_num}",
                    use_container_width=True,
                ):
                    st.session_state["uw_current_step"] = step_num
                    st.rerun()
            else:
                st.markdown(
                    f":gray[Step {step_num}: {label}]"
                )


def _render_sidebar():
    """Render unified sidebar with IPR and optimization parameters."""
    with st.sidebar:
        st.header("Workflow Parameters")

        # IPR Parameters
        st.subheader("IPR Parameters")
        max_rp_schrader = st.number_input(
            "Max Res Pressure — Schrader (psi)",
            min_value=800,
            max_value=3000,
            value=1800,
            step=50,
            key="uw_max_rp_sch",
        )
        max_rp_kuparuk = st.number_input(
            "Max Res Pressure — Kuparuk (psi)",
            min_value=1500,
            max_value=5000,
            value=3000,
            step=50,
            key="uw_max_rp_kup",
        )
        resp_modifier = st.number_input(
            "Res Pres Modifier (psi)",
            min_value=0,
            max_value=500,
            value=0,
            step=10,
            key="uw_resp_mod",
            help="Offset added to estimated reservoir pressure",
        )

        # Power Fluid Settings
        st.subheader("Power Fluid Constraint")
        total_pf = st.number_input(
            "Total Available Power Fluid (bbl/day)",
            min_value=0,
            value=30000,
            step=500,
            key="uw_total_pf",
        )
        pf_pressure = st.number_input(
            "Power Fluid Pressure (psi)",
            min_value=1000,
            max_value=5000,
            value=3168,
            step=100,
            key="uw_pf_pressure",
        )
        rho_pf = st.number_input(
            "Power Fluid Density (lbm/ft\u00b3)",
            min_value=50.0,
            max_value=70.0,
            value=62.4,
            step=0.1,
            key="uw_rho_pf",
        )

        # Optimization Settings
        st.subheader("Optimization Settings")
        opt_method = st.selectbox(
            "Algorithm",
            ["milp", "mckp"],
            index=0,
            key="uw_opt_method",
            help="MILP: Optimal via scipy linear programming. MCKP: Multi-choice knapsack via OR-Tools CP-SAT.",
        )
        marginal_wc = st.number_input(
            "Marginal Watercut Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.94,
            step=0.01,
            format="%.2f",
            key="uw_marginal_wc",
        )

        has_jp_history = "jp_history_df" in st.session_state
        use_calibration = st.checkbox(
            "Apply Model Calibration",
            value=True,
            key="uw_use_calibration",
            help="Scale predictions using model-vs-actual on current pump configs",
            disabled=not has_jp_history,
        )

        # Pump Options
        st.subheader("Pump Options to Test")
        nozzle_opts = st.multiselect(
            "Nozzle Sizes",
            NOZZLE_OPTIONS,
            default=["8", "9", "10", "11", "12", "13", "14"],
            key="uw_nozzle_opts",
        )
        throat_opts = st.multiselect(
            "Throat Ratios",
            THROAT_OPTIONS,
            default=["X", "A", "B", "C", "D"],
            key="uw_throat_opts",
        )

        # Power Fluid Sensitivity
        st.subheader("Power Fluid Sensitivity")
        run_pf_sensitivity = st.checkbox(
            "Run PF Sensitivity",
            value=False,
            key="uw_run_pf_sensitivity",
            help="Sweep PF rate and pressure to map field oil production surface",
        )
        if run_pf_sensitivity:
            st.caption("Rate Range")
            st.number_input(
                "PF Rate Min (BWPD)",
                min_value=1000,
                max_value=60000,
                value=max(1000, total_pf - 6000),
                step=1000,
                key="uw_pf_rate_min",
            )
            st.number_input(
                "PF Rate Max (BWPD)",
                min_value=1000,
                max_value=60000,
                value=min(60000, total_pf + 6000),
                step=1000,
                key="uw_pf_rate_max",
            )
            st.number_input(
                "PF Rate Step (BWPD)",
                min_value=500,
                max_value=10000,
                value=2000,
                step=500,
                key="uw_pf_rate_step",
            )
            st.caption("Pressure Range")
            st.number_input(
                "PF Pressure Min (psi)",
                min_value=1000,
                max_value=5000,
                value=max(1000, pf_pressure - 500),
                step=100,
                key="uw_pf_press_min",
            )
            st.number_input(
                "PF Pressure Max (psi)",
                min_value=1000,
                max_value=5000,
                value=min(5000, pf_pressure + 500),
                step=100,
                key="uw_pf_press_max",
            )
            st.number_input(
                "PF Pressure Step (psi)",
                min_value=50,
                max_value=500,
                value=100,
                step=50,
                key="uw_pf_press_step",
            )

        # Cache management
        st.divider()
        st.subheader("Data Cache")
        force_refresh = st.checkbox(
            "Force fresh Databricks query",
            value=False,
            key="uw_force_refresh",
            help="Clear cached data and re-query Databricks",
        )
        if force_refresh:
            from woffl.gui.well_test_page import (
                _cached_bhp_query,
                _cached_mpu_well_names,
                _cached_well_test_query,
            )

            _cached_bhp_query.clear()
            _cached_well_test_query.clear()
            _cached_mpu_well_names.clear()
            st.success("Cache cleared")


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
    elif current_step == 3:
        from woffl.gui.workflow_steps.step3_configure_optimize import render_step3

        render_step3()
    elif current_step == 4:
        from woffl.gui.workflow_steps.step4_results import render_step4

        render_step4()
