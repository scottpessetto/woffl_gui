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


def _trigger_browser_download(data: bytes, filename: str, mime: str) -> None:
    """Single-click download — delegates to the shared implementation in
    woffl.gui.components.download (kept under this name because step4 also
    imports it from here)."""
    from woffl.gui.components.download import autodownload

    autodownload(data, filename, mime)


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
        "Proceed to Pre-Calibration →",
        type="primary",
        use_container_width=True,
        key="uw_step2_csv_proceed",
    ):
        # Clear 2.5+ so a previous run's calibration can't be "Accepted"
        # onto configs it wasn't fit against.
        _clear_downstream(2.5)
        st.session_state["uw_current_step"] = 2.5
        st.session_state["uw_max_step_reached"] = max(
            st.session_state.get("uw_max_step_reached", 2), 2.5
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
            f"{len(dropped_wells)} wells dropped (no fluid-rate test in window): "
            f"{', '.join(dropped_wells)}"
        )

    # IPR settings (live inline so the user can re-run with different
    # caps/offsets without hopping back to a sidebar). Editing any of these
    # widgets and clicking the re-run button below clears the Vogel cache.
    with st.expander("IPR Settings", expanded=False):
        st.number_input(
            "Max Res Pressure — Schrader (psi)",
            min_value=800,
            max_value=3000,
            value=1800,
            step=50,
            key="uw_max_rp_sch",
            help="Cap on estimated reservoir pressure for Schrader-field wells.",
        )
        st.number_input(
            "Max Res Pressure — Kuparuk (psi)",
            min_value=1500,
            max_value=5000,
            value=3000,
            step=50,
            key="uw_max_rp_kup",
            help="Cap on estimated reservoir pressure for Kuparuk-field wells.",
        )
        st.number_input(
            "Res Pres Modifier (psi)",
            min_value=0,
            max_value=500,
            value=0,
            step=10,
            key="uw_resp_mod",
            help="Constant offset added to estimated reservoir pressure.",
        )
        if st.button(
            "Re-run IPR Analysis",
            key="uw_rerun_ipr",
            help="Discard the cached Vogel fit and re-run with the values above.",
        ):
            for k in ("uw_vogel_coeffs", "uw_ipr_curves", "uw_merged_with_rp"):
                st.session_state.pop(k, None)
            st.rerun()

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
    # Restore previous exclusions — the editor's widget state is GC'd by any
    # step detour, which silently re-included every well. uw_excluded_wells
    # is saved at Proceed time, so the base reflects the last applied set.
    prev_excluded = st.session_state.get("uw_excluded_wells") or set()
    exclude_df.insert(0, "Include", ~exclude_df["Well"].isin(prev_excluded))
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
    # Pass the live Databricks chars (module-level import) — without them the
    # template was built from legacy jp_chars.csv, whose stale/empty values
    # clobbered the real JP_TVD etc. when loaded back into WellConfigs.
    template_df = export_optimization_template(
        filtered_coeffs, jp_chars=load_well_characteristics()
    )

    csv_data = template_df.to_csv(index=False)
    st.download_button(
        label="Download Optimization Template CSV",
        data=csv_data,
        file_name="optimization_template.csv",
        mime="text/csv",
    )

    # IPR review exports — same artifacts as the buttons at the bottom of the
    # IPR Curves tab, surfaced HERE where the review/export decision is made
    # (buried in the tab they were effectively undiscoverable; reviewing the
    # fits well-by-well before optimizing is the whole point of this step).
    exp1, exp2 = st.columns(2)
    with exp1:
        if st.button(
            f"Download IPR Review PDF — all {len(ipr_curves)} wells",
            key="uw_export_ipr_pdf",
            use_container_width=True,
            help=(
                "One full-page IPR per well (Vogel fit + test points) for "
                "reviewing the fits before optimizing."
            ),
        ):
            with st.spinner(f"Generating PDF for {len(ipr_curves)} wells..."):
                pdf_bytes = create_ipr_pdf(ipr_curves, merged_with_rp)
            _trigger_browser_download(
                pdf_bytes, "ipr_review_all_wells.pdf", "application/pdf"
            )
    with exp2:
        if st.button(
            "Download IPR Grid PNG (all wells)",
            key="uw_export_ipr_png",
            use_container_width=True,
            help="All wells on one high-resolution grid image.",
        ):
            with st.spinner(f"Rendering {len(ipr_curves)} wells..."):
                png_bytes = create_ipr_grid_png(ipr_curves, merged_with_rp, dpi=200)
            _trigger_browser_download(png_bytes, "ipr_review_grid.png", "image/png")

    # --- Upload-edited-template re-visualizer ---
    _render_edited_template_viewer(merged_with_rp)

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

        # Clear 2.5+ (not just 3+): stale pre-calibration results fit against
        # the OLD IPR anchors could otherwise be "Accepted" onto these new
        # WellConfigs by name match.
        _clear_downstream(2.5)
        st.session_state["uw_well_configs"] = well_configs
        st.session_state["uw_template_df"] = template_df
        st.session_state["uw_excluded_wells"] = set(
            edited[~edited["Include"]]["Well"].tolist()
        )
        st.session_state["uw_current_step"] = 2.5
        st.session_state["uw_max_step_reached"] = max(
            st.session_state.get("uw_max_step_reached", 2), 2.5
        )
        st.rerun()


def _render_ipr_curves(
    ipr_curves,
    merged_data,
    vogel_coeffs,
    *,
    key_suffix: str = "",
    png_filename: str = "vogel_ipr_grid.png",
    pdf_filename: str = "vogel_ipr_curves.pdf",
):
    """Render IPR curves with a well selector and PNG/PDF download buttons.

    ``key_suffix`` lets the same UI be rendered twice on one page without
    Streamlit key collisions — used by the "View Edited IPRs" section below
    to mirror the original IPR Curves tab against an uploaded template.
    """
    if not ipr_curves:
        st.warning("No IPR curves to display.")
        return

    wells = sorted(ipr_curves.keys())

    col_sel, col_nav, col_lbl = st.columns([3, 1, 2])
    with col_sel:
        selected_well = st.selectbox(
            "Select Well:",
            wells,
            key=f"uw_ipr_well_selector{key_suffix}",
        )
    with col_nav:
        st.write("")
        st.write(f"**{wells.index(selected_well) + 1}** of **{len(wells)}** wells")
    with col_lbl:
        st.write("")
        show_jp_labels = st.checkbox(
            "Show JP labels on points",
            value=False,
            key=f"uw_ipr_show_jp_labels{key_suffix}",
            help=(
                "Render each test point as the pump installed at that "
                "test's date (e.g. \"12B\") inside an enlarged colored marker."
            ),
        )

    if selected_well and selected_well in ipr_curves:
        fig = create_ipr_plotly(
            selected_well,
            ipr_curves[selected_well],
            merged_data,
            jp_history=st.session_state.get("jp_history_df"),
            show_jp_labels=show_jp_labels,
        )
        st.plotly_chart(fig, use_container_width=True)

        well_coeffs = vogel_coeffs[vogel_coeffs["Well"] == selected_well]
        if not well_coeffs.empty:
            row = well_coeffs.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Res Pressure", f"{row['ResP']:.0f} psi")
            c2.metric("Qmax (Recent)", f"{row['QMax_recent']:.0f} BPD")
            c3.metric("Test BHP", f"{row['pwf']:.0f} psi")
            n_tests = row.get("num_tests")
            c4.metric(
                "# Tests",
                int(n_tests) if pd.notna(n_tests) else "—",
            )

    # Export all IPR curves — single-click: build the file then auto-trigger
    # the browser download via a hidden anchor + JS, skipping the separate
    # download-button step.
    st.write("---")
    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        if st.button(
            "Download Grid PNG",
            use_container_width=True,
            key=f"uw_gen_png{key_suffix}",
        ):
            with st.spinner(f"Rendering {len(wells)} wells..."):
                png_bytes = create_ipr_grid_png(ipr_curves, merged_data, dpi=200)
            _trigger_browser_download(png_bytes, png_filename, "image/png")
    with exp_col2:
        if st.button(
            "Download PDF",
            use_container_width=True,
            key=f"uw_gen_pdf{key_suffix}",
        ):
            with st.spinner(f"Generating PDF for {len(wells)} wells..."):
                pdf_bytes = create_ipr_pdf(ipr_curves, merged_data)
            _trigger_browser_download(pdf_bytes, pdf_filename, "application/pdf")


def _template_to_vogel_coeffs(df: pd.DataFrame) -> pd.DataFrame:
    """Map an optimization-template CSV back to the schema generate_ipr_curves expects.

    The download (``export_optimization_template``) flattens ``vogel_coeffs``
    + ``jp_chars`` into a "ready to optimize" shape. To re-visualize the
    IPRs from an edited copy we just need the columns generate_ipr_curves
    actually reads: Well, ResP, qwf, pwf, QMax_recent. QMax_recent isn't in
    the template so we re-derive it via the Vogel formula from the edited
    ResP / qwf / pwf.
    """
    coeffs = df[["Well"]].copy()
    coeffs["ResP"] = pd.to_numeric(df["res_pres"], errors="coerce")
    coeffs["qwf"] = pd.to_numeric(df["qwf_blpd"], errors="coerce")
    coeffs["pwf"] = pd.to_numeric(df["pwf"], errors="coerce")
    if "form_wc" in df.columns:
        coeffs["form_wc"] = pd.to_numeric(df["form_wc"], errors="coerce").fillna(0.5)
    else:
        coeffs["form_wc"] = 0.5
    if "form_gor" in df.columns:
        coeffs["fgor"] = pd.to_numeric(df["form_gor"], errors="coerce").fillna(250)
    else:
        coeffs["fgor"] = 250
    coeffs["num_tests"] = float("nan")  # not carried in template
    coeffs["most_recent_date"] = pd.NaT

    # QMax_recent via Vogel: qmax = qwf / (1 − 0.2·x − 0.8·x²)  where x = pwf/ResP
    x = coeffs["pwf"] / coeffs["ResP"]
    vogel_frac = 1 - 0.2 * x - 0.8 * x * x
    coeffs["QMax_recent"] = coeffs["qwf"] / vogel_frac.where(vogel_frac > 0, 1.0)

    # Drop rows that can't produce a curve (missing or non-positive ResP)
    coeffs = coeffs[coeffs["ResP"] > 0].reset_index(drop=True)
    return coeffs


def _render_edited_template_viewer(merged_with_rp: pd.DataFrame) -> None:
    """Upload-edited-template re-visualizer.

    Sits below the optimization-template download in the Review IPR step.
    The user downloads the template, hand-edits ResP / qwf / pwf / form_wc,
    re-uploads here, and sees the same IPR Curves view (well selector +
    PNG / PDF download) against the edited values. View-only — does NOT
    feed downstream optimization; to actually run with edited values, the
    user uploads via Step 1's CSV-upload path.
    """
    st.write("### View Edited IPRs")
    st.caption(
        "Download the **Optimization Template CSV** above, hand-edit values "
        "(ResP, qwf, pwf, form_wc), and upload it here to visualize the IPRs "
        "from your edits. View-only — to actually optimize against the edited "
        "values, restart Step 1 with the **Upload CSV Template** path."
    )

    uploaded = st.file_uploader(
        "Upload edited optimization template",
        type=["csv"],
        key="uw_ipr_edited_upload",
        help=(
            "Same column shape as the template download above: "
            "Well / res_pres / qwf_blpd / pwf / form_wc (others ignored)."
        ),
    )
    if uploaded is None:
        return

    try:
        edited_df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read the uploaded CSV: {e}")
        return

    required = {"Well", "res_pres", "qwf_blpd", "pwf"}
    missing = required - set(edited_df.columns)
    if missing:
        st.error(
            f"Upload is missing required columns: {sorted(missing)}. "
            "Expected the same schema as the template download above."
        )
        return

    try:
        edited_coeffs = _template_to_vogel_coeffs(edited_df)
    except Exception as e:
        st.error(f"Could not parse edited template: {e}")
        return

    if edited_coeffs.empty:
        st.warning(
            "No wells in the upload have a positive ResP — nothing to plot. "
            "Check the res_pres column."
        )
        return

    edited_curves = generate_ipr_curves(edited_coeffs)
    if not edited_curves:
        st.warning(
            "Could not generate IPR curves from the edited values — Vogel "
            "couldn't fit any well. Check pwf < ResP for each row."
        )
        return

    st.success(
        f"Loaded **{len(edited_curves)}** wells from the edited template."
    )

    _render_ipr_curves(
        edited_curves,
        merged_with_rp,
        edited_coeffs,
        key_suffix="_edited",
        png_filename="vogel_ipr_grid_edited.png",
        pdf_filename="vogel_ipr_curves_edited.pdf",
    )
