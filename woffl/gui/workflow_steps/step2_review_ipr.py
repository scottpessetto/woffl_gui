"""Step 2: Review IPR — run Vogel analysis, review fits, exclude wells, download CSV."""

import pandas as pd
import streamlit as st

from woffl.assembly.ipr_analyzer import (
    compute_vogel_coefficients,
    estimate_reservoir_pressure,
    export_optimization_template,
    generate_ipr_curves,
)
from woffl.assembly.network_optimizer import load_wells_from_dataframe
from woffl.gui.ipr_viz import (
    create_ipr_grid_png,
    create_ipr_pdf,
    create_ipr_plotly,
)
from woffl.gui.utils import load_well_characteristics
from woffl.gui.workflow_page import _clear_downstream


def render_step2():
    st.subheader("Step 2: Review IPR")

    csv_shortcut = st.session_state.get("uw_csv_shortcut", False)

    if csv_shortcut:
        _render_csv_review()
    else:
        _render_ipr_review()


def _render_csv_review():
    """Show well configs loaded from CSV upload — no IPR analysis needed."""
    well_configs = st.session_state.get("uw_well_configs", [])
    template_df = st.session_state.get("uw_template_df")

    if not well_configs:
        st.warning("No well configurations loaded. Go back to Step 1.")
        return

    st.info(f"**{len(well_configs)} wells** loaded from CSV — IPR analysis skipped.")

    # Show the loaded parameters
    rows = []
    for wc in well_configs:
        rows.append(
            {
                "Well": wc.well_name,
                "Res Pres (psi)": f"{wc.res_pres:.0f}",
                "WC": f"{wc.form_wc:.0%}",
                "Qwf (BLPD)": f"{wc.qwf:.0f}",
                "Pwf (psi)": f"{wc.pwf:.0f}",
                "Field": wc.field_model,
                "JP TVD (ft)": f"{wc.jpump_tvd:.0f}",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Download the template back out
    if template_df is not None:
        csv_data = template_df.to_csv(index=False)
        st.download_button(
            label="Download CSV Template",
            data=csv_data,
            file_name="optimization_template.csv",
            mime="text/csv",
        )

    if st.button(
        "Proceed to Optimization →",
        type="primary",
        use_container_width=True,
        key="uw_step2_csv_proceed",
    ):
        st.session_state["uw_current_step"] = 3
        st.session_state["uw_max_step_reached"] = max(
            st.session_state.get("uw_max_step_reached", 2), 3
        )
        st.rerun()


def _render_ipr_review():
    """Run IPR analysis on Databricks well tests, show results, allow exclusions."""
    df = st.session_state.get("uw_well_test_df")
    if df is None:
        st.warning("No well test data loaded. Go back to Step 1.")
        return

    dropped_wells = st.session_state.get("uw_dropped_wells", [])
    if dropped_wells:
        st.warning(
            f"{len(dropped_wells)} wells dropped (no BHP or fluid rate): "
            f"{', '.join(dropped_wells)}"
        )

    # Run IPR analysis if not already done
    if "uw_vogel_coeffs" not in st.session_state:
        max_rp_sch = st.session_state.get("uw_max_rp_sch", 1800)
        max_rp_kup = st.session_state.get("uw_max_rp_kup", 3000)
        resp_mod = st.session_state.get("uw_resp_mod", 0)

        with st.spinner("Estimating reservoir pressure..."):
            jp_chars = load_well_characteristics()
            merged_with_rp = estimate_reservoir_pressure(
                df,
                max_pres_schrader=max_rp_sch,
                max_pres_kuparuk=max_rp_kup,
                jp_chars=jp_chars,
            )

        with st.spinner("Computing Vogel IPR coefficients..."):
            vogel_coeffs = compute_vogel_coefficients(
                merged_with_rp, resp_modifier=resp_mod
            )

        if vogel_coeffs.empty:
            st.error("Could not compute Vogel coefficients for any wells.")
            return

        ipr_curves = generate_ipr_curves(vogel_coeffs)

        st.session_state["uw_vogel_coeffs"] = vogel_coeffs
        st.session_state["uw_ipr_curves"] = ipr_curves
        st.session_state["uw_merged_with_rp"] = merged_with_rp

    vogel_coeffs = st.session_state["uw_vogel_coeffs"]
    ipr_curves = st.session_state["uw_ipr_curves"]
    merged_with_rp = st.session_state["uw_merged_with_rp"]

    # --- Summary metrics ---
    st.write("### Field Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Wells Analyzed", len(vogel_coeffs))
    col2.metric("Avg Res Pressure", f"{vogel_coeffs['ResP'].mean():.0f} psi")
    col3.metric("Avg Qmax", f"{vogel_coeffs['QMax_recent'].mean():.0f} BPD")
    col4.metric("Wells < 3 Tests", int((vogel_coeffs["num_tests"] < 3).sum()))

    # --- Tabs: IPR Curves, Well Details ---
    tabs = st.tabs(["IPR Curves", "Well Details"])

    with tabs[0]:
        _render_ipr_curves(ipr_curves, merged_with_rp, vogel_coeffs)

    with tabs[1]:
        display_df = vogel_coeffs.copy()
        display_df = display_df.rename(
            columns={
                "ResP": "Res Pressure (psi)",
                "QMax_recent": "Qmax Recent (BPD)",
                "qwf": "Test Fluid Rate (BPD)",
                "pwf": "Test BHP (psi)",
                "form_wc": "Watercut",
                "num_tests": "# Tests",
            }
        )
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # --- Well exclusion ---
    st.write("### Select Wells to Optimize")
    st.caption("Uncheck wells to exclude them from optimization.")

    exclude_df = vogel_coeffs[["Well", "num_tests", "ResP", "QMax_recent"]].copy()
    exclude_df.insert(0, "Include", True)
    # Flag wells with few tests
    exclude_df["Note"] = exclude_df["num_tests"].apply(
        lambda n: "Few tests" if n < 3 else ""
    )

    edited = st.data_editor(
        exclude_df,
        use_container_width=True,
        hide_index=True,
        disabled=["Well", "num_tests", "ResP", "QMax_recent", "Note"],
        column_config={
            "Include": st.column_config.CheckboxColumn("Include", default=True),
            "ResP": st.column_config.NumberColumn("Res Pres (psi)", format="%.0f"),
            "QMax_recent": st.column_config.NumberColumn("Qmax (BPD)", format="%.0f"),
        },
        key="uw_well_exclusion_editor",
    )

    included_wells = edited[edited["Include"]]["Well"].tolist()
    excluded_count = len(edited) - len(included_wells)
    st.caption(
        f"{len(included_wells)} wells included"
        + (f", {excluded_count} excluded" if excluded_count else "")
    )

    # --- CSV download ---
    st.write("### Export")
    filtered_coeffs = vogel_coeffs[vogel_coeffs["Well"].isin(included_wells)]
    template_df = export_optimization_template(filtered_coeffs)

    csv_data = template_df.to_csv(index=False)
    st.download_button(
        label="Download Optimization Template CSV",
        data=csv_data,
        file_name="optimization_template.csv",
        mime="text/csv",
    )

    # --- Proceed ---
    if st.button(
        f"Proceed to Optimization with {len(included_wells)} wells →",
        type="primary",
        use_container_width=True,
        key="uw_step2_proceed",
    ):
        if not included_wells:
            st.error("Select at least one well.")
            return

        # Build WellConfigs from template
        try:
            well_configs = load_wells_from_dataframe(template_df)
        except Exception as e:
            st.error(f"Error building well configurations: {e}")
            return

        _clear_downstream(3)
        st.session_state["uw_well_configs"] = well_configs
        st.session_state["uw_template_df"] = template_df
        st.session_state["uw_excluded_wells"] = set(
            edited[~edited["Include"]]["Well"].tolist()
        )
        st.session_state["uw_current_step"] = 3
        st.session_state["uw_max_step_reached"] = max(
            st.session_state.get("uw_max_step_reached", 2), 3
        )
        st.rerun()


def _render_ipr_curves(ipr_curves, merged_data, vogel_coeffs):
    """Render IPR curves with well selector."""
    if not ipr_curves:
        st.warning("No IPR curves to display.")
        return

    wells = sorted(ipr_curves.keys())

    col_sel, col_nav = st.columns([3, 1])
    with col_sel:
        selected_well = st.selectbox(
            "Select Well:",
            wells,
            key="uw_ipr_well_selector",
        )
    with col_nav:
        st.write("")
        st.write(f"**{wells.index(selected_well) + 1}** of **{len(wells)}** wells")

    if selected_well and selected_well in ipr_curves:
        fig = create_ipr_plotly(
            selected_well,
            ipr_curves[selected_well],
            merged_data,
        )
        st.plotly_chart(fig, use_container_width=True)

        well_coeffs = vogel_coeffs[vogel_coeffs["Well"] == selected_well]
        if not well_coeffs.empty:
            row = well_coeffs.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Res Pressure", f"{row['ResP']:.0f} psi")
            c2.metric("Qmax (Recent)", f"{row['QMax_recent']:.0f} BPD")
            c3.metric("Test BHP", f"{row['pwf']:.0f} psi")
            c4.metric("# Tests", int(row["num_tests"]))

    # Export all IPR curves
    st.write("---")
    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        if st.button("Generate Grid PNG", use_container_width=True, key="uw_gen_png"):
            with st.spinner(f"Rendering {len(wells)} wells..."):
                png_bytes = create_ipr_grid_png(ipr_curves, merged_data, dpi=200)
            st.download_button(
                label="Download Grid PNG",
                data=png_bytes,
                file_name="vogel_ipr_grid.png",
                mime="image/png",
                use_container_width=True,
            )
    with exp_col2:
        if st.button("Generate PDF", use_container_width=True, key="uw_gen_pdf"):
            with st.spinner(f"Generating PDF for {len(wells)} wells..."):
                pdf_bytes = create_ipr_pdf(ipr_curves, merged_data)
            st.download_button(
                label="Download IPR Curves PDF",
                data=pdf_bytes,
                file_name="vogel_ipr_curves.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
