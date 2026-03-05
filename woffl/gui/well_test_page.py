"""Well Test Analysis Page

Third mode in the WOFFL GUI. Allows users to:
1. Upload FDC well test CSV
2. Select wells to analyze
3. Query Databricks for BHP data (cached for 24 hours)
4. Compute Vogel IPR parameters
5. Visualize IPR curves
6. Download optimization template CSV
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from woffl.assembly.databricks_client import (
    get_tags_for_wells,
    load_tag_dict,
    query_bhp_for_well_tests,
)
from woffl.assembly.restls_client import (
    fetch_milne_well_tests,
    filter_wells_by_pad,
    get_mpu_well_names,
    get_pad_names,
)


@st.cache_data(ttl=86400, show_spinner=False)  # Cache for 24 hours
def _cached_bhp_query(
    tag_dict_frozen: tuple,
    wells_tuple: tuple,
) -> dict:
    """Cached wrapper around query_bhp_for_well_tests.

    Uses @st.cache_data which persists in memory across user sessions
    as long as the Databricks App compute is running. Cache TTL is 24 hours.

    Args:
        tag_dict_frozen: Tuple of (well, bhp_tag, headerP_tag, whp_tag) for hashability
        wells_tuple: Tuple of well names (hashable)

    Returns:
        Dictionary mapping well name to BHP DataFrame
    """
    # Reconstruct the tag_dict from the frozen tuple
    tag_dict = {well: (bhp, hp, whp) for well, bhp, hp, whp in tag_dict_frozen}
    return query_bhp_for_well_tests(tag_dict, list(wells_tuple))


@st.cache_data(ttl=86400, show_spinner=False)  # Cache for 24 hours
def _cached_mpu_well_names() -> list[str]:
    """Cached list of all MPU well names from Databricks."""
    return get_mpu_well_names()


@st.cache_data(ttl=86400, show_spinner=False)  # Cache for 24 hours
def _cached_well_test_query(
    start_date: str, end_date: str, well_names_tuple: tuple
) -> tuple[pd.DataFrame, list[str]]:
    """Cached wrapper around fetch_milne_well_tests.

    Args:
        start_date: Start date string 'YYYY-MM-DD' (used as cache key).
        end_date: End date string 'YYYY-MM-DD' (used as cache key).
        well_names_tuple: Tuple of well names to query (hashable for cache key).

    Returns tuple of (DataFrame, dropped_wells). DataFrame has columns matching
    ipr_analyzer expectations: well, WtDate, BHP, WtTotalFluid, WtOilVol, etc.
    dropped_wells lists wells removed due to missing BHP or fluid rate data.
    """
    return fetch_milne_well_tests(start_date, end_date, list(well_names_tuple))


from woffl.assembly.ipr_analyzer import (
    compute_vogel_coefficients,
    estimate_reservoir_pressure,
    export_optimization_template,
    generate_ipr_curves,
)
from woffl.assembly.well_test_processor import WellTestProcessor, merge_tests_with_bhp
from woffl.gui.ipr_viz import (
    create_ipr_grid_png,
    create_ipr_pdf,
    create_ipr_plotly,
    create_qmax_comparison_chart,
    create_rp_comparison_chart,
)


def _load_jp_chars() -> pd.DataFrame:
    """Load jp_chars.csv for well characteristics lookup."""
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        jp_chars_path = os.path.join(current_dir, "..", "jp_data", "jp_chars.csv")
        return pd.read_csv(jp_chars_path)
    except FileNotFoundError:
        return pd.DataFrame()


def run_well_test_analysis_page():
    """Render the Well Test Analysis page."""

    st.title("Well Test Analysis")

    st.markdown(
        """
    Upload FDC well test data, query Databricks for BHP gauge data,
    and generate Vogel IPR parameters for use in multi-well optimization.
    """
    )

    # --- Sidebar parameters ---
    with st.sidebar:
        st.header("IPR Analysis Parameters")

        st.subheader("Reservoir Pressure Limits")
        max_rp_schrader = st.number_input(
            "Max Res Pressure — Schrader (psi)",
            min_value=800,
            max_value=3000,
            value=1800,
            step=50,
            help="Maximum reservoir pressure for Schrader wells during RP estimation",
        )
        max_rp_kuparuk = st.number_input(
            "Max Res Pressure — Kuparuk (psi)",
            min_value=1500,
            max_value=5000,
            value=3000,
            step=50,
            help="Maximum reservoir pressure for Kuparuk wells during RP estimation",
        )
        resp_modifier = st.number_input(
            "Res Pres Modifier (psi)",
            min_value=0,
            max_value=500,
            value=0,
            step=10,
            help="Offset added to estimated Res Pres for Vogel curve fitting. "
            "Set to 0 for tightest fit to data. Increase to shift curve higher.",
        )

        st.divider()
        st.subheader("BHP Data Cache")
        st.caption(
            "BHP query results are cached for 24 hours. " "The cache persists across users while the app is running."
        )
        force_refresh = st.checkbox(
            "Force fresh Databricks query",
            value=False,
            help="Clear the cache and re-query Databricks for BHP data",
        )
        if force_refresh:
            _cached_bhp_query.clear()
            _cached_well_test_query.clear()
            _cached_mpu_well_names.clear()
            st.success("Cache cleared — next run will re-query Databricks")

    # --- Data Source Selection ---
    data_source = st.radio(
        "Data Source",
        options=["Databricks", "FDC CSV Upload"],
        horizontal=True,
        help="Choose how to load well test data",
    )

    if data_source == "Databricks":
        _run_restls_path(max_rp_schrader, max_rp_kuparuk, resp_modifier)
    else:
        _run_fdc_csv_path(max_rp_schrader, max_rp_kuparuk, resp_modifier)

    # --- Display Results (shared by both paths) ---
    if st.session_state.get("wt_analysis_complete", False):
        vogel_coeffs = st.session_state["wt_vogel_coeffs"]
        ipr_curves = st.session_state["wt_ipr_curves"]
        merged_with_rp = st.session_state["wt_merged_data"]

        st.write("## Results")

        tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary", "📈 IPR Curves", "📋 Well Details", "💾 Export"])

        with tab1:
            _render_summary_tab(vogel_coeffs, merged_with_rp)

        with tab2:
            _render_ipr_curves_tab(ipr_curves, merged_with_rp, vogel_coeffs)

        with tab3:
            _render_well_details_tab(vogel_coeffs, merged_with_rp)

        with tab4:
            _render_export_tab(vogel_coeffs)


def _run_restls_path(max_rp_schrader, max_rp_kuparuk, resp_modifier):
    """Databricks data source path — loads well tests directly."""
    from datetime import date, timedelta

    # --- Pad & well selection (before query) ---
    # Fetch well names (cached) to build pad list
    try:
        with st.spinner("Fetching well list from Databricks..."):
            all_well_names = _cached_mpu_well_names()
    except Exception as e:
        st.error(f"Could not fetch well names from Databricks: {e}")
        return

    if not all_well_names:
        st.warning("No well names returned from Databricks. Check connectivity and try 'Force fresh Databricks query' in the sidebar.")
        return

    all_pads = get_pad_names(all_well_names)

    with st.sidebar:
        st.divider()
        st.subheader("Pad Selection")
        selected_pads = []
        for pad in all_pads:
            pad_wells = filter_wells_by_pad(all_well_names, [pad])
            if st.checkbox(f"Pad {pad} ({len(pad_wells)})", value=False, key=f"pad_{pad}"):
                selected_pads.append(pad)

    if not selected_pads:
        st.warning("Select at least one pad in the sidebar.")
        return

    filtered_well_names = filter_wells_by_pad(all_well_names, selected_pads)

    # Date range pickers
    col_start, col_end = st.columns(2)
    with col_start:
        default_start = date.today() - timedelta(days=730)
        start_date = st.date_input("Start Date", value=default_start, key="db_start_date")
    with col_end:
        end_date = st.date_input("End Date", value=date.today(), key="db_end_date")

    if start_date > end_date:
        st.error("Start date must be before end date.")
        return

    st.caption(f"{len(filtered_well_names)} wells across {len(selected_pads)} pads")

    if st.button("🔄 Load Well Tests from Databricks", type="primary", use_container_width=True):
        try:
            with st.spinner("Querying Databricks for well tests (cached 24h)..."):
                df, dropped_wells = _cached_well_test_query(
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d"),
                    tuple(filtered_well_names),
                )

            if df.empty:
                st.error("No well test data returned for selected pads/dates.")
                return

            st.session_state["restls_well_tests"] = df
            st.success(f"Loaded {len(df)} well tests for {df['well'].nunique()} wells")

            if dropped_wells:
                st.warning(
                    f"{len(dropped_wells)} wells dropped (no BHP or fluid rate data): "
                    f"{', '.join(dropped_wells)}"
                )

        except Exception as e:
            st.error(f"Error querying Databricks: {str(e)}")
            st.exception(e)
            return

    if "restls_well_tests" not in st.session_state:
        st.info("Select pads in the sidebar, then click the button above to load well tests.")
        return

    df = st.session_state["restls_well_tests"]
    all_wells = sorted(df["well"].unique().tolist())

    # Show test count summary
    test_counts = df.groupby("well").size().sort_values(ascending=False)
    with st.expander(f"📊 Test Counts ({len(all_wells)} wells)"):
        st.dataframe(
            test_counts.reset_index().rename(columns={"well": "Well", 0: "Tests"}),
            use_container_width=True,
            hide_index=True,
        )

    # --- Run Analysis ---
    if st.button("🚀 Run Well Test Analysis", type="primary", use_container_width=True, key="restls_run"):
        try:
            merged_data = df.copy()

            if merged_data.empty:
                st.error("❌ No test data for selected wells.")
                return

            wells_in_data = merged_data["well"].nunique()
            st.success(f"✅ {len(merged_data)} test points across {wells_in_data} wells (BHP included)")

            # Estimate reservoir pressure
            st.write("### Step 1: Estimating Reservoir Pressure")
            with st.spinner("Estimating optimal reservoir pressure per well..."):
                jp_chars = _load_jp_chars()
                merged_with_rp = estimate_reservoir_pressure(
                    merged_data,
                    max_pres_schrader=max_rp_schrader,
                    max_pres_kuparuk=max_rp_kuparuk,
                    jp_chars=jp_chars,
                )

            st.success("✅ Reservoir pressure estimation complete")

            # Compute Vogel coefficients
            st.write("### Step 2: Computing Vogel IPR Coefficients")
            with st.spinner("Computing Vogel IPR parameters..."):
                vogel_coeffs = compute_vogel_coefficients(merged_with_rp, resp_modifier=resp_modifier)

            if vogel_coeffs.empty:
                st.error("❌ Could not compute Vogel coefficients for any wells.")
                return

            st.success(f"✅ Computed IPR parameters for {len(vogel_coeffs)} wells")

            # Generate IPR curves
            ipr_curves = generate_ipr_curves(vogel_coeffs)

            # Store results in session state
            st.session_state["wt_vogel_coeffs"] = vogel_coeffs
            st.session_state["wt_ipr_curves"] = ipr_curves
            st.session_state["wt_merged_data"] = merged_with_rp
            st.session_state["wt_analysis_complete"] = True

        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            st.exception(e)
            return


def _run_fdc_csv_path(max_rp_schrader, max_rp_kuparuk, resp_modifier):
    """FDC CSV upload data source path — original workflow."""

    # --- File uploads ---
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_fdc = st.file_uploader(
            "Upload FDC Well Test CSV",
            type=["csv"],
            help="Upload the FDC well test export CSV file",
        )

    with col2:
        uploaded_tags = st.file_uploader(
            "Custom Tag File (optional)",
            type=["csv"],
            help="Upload a custom bhp_dict.csv if the bundled one doesn't have your wells",
        )

    if not uploaded_fdc:
        st.info("👆 Upload an FDC well test CSV file to begin.")

        with st.expander("📖 How to Use", expanded=True):
            st.markdown(
                """
            ### Quick Start Guide

            1. **Export well tests** from FDC as a CSV file
            2. **Upload the CSV** using the file uploader above
            3. **Select wells** to analyze (or use all)
            4. **Configure parameters** in the sidebar (max RP, modifier)
            5. **Run analysis** to compute Vogel IPR parameters
            6. **Download results** as a CSV template for Multi-Well Optimization

            ### What You Get
            - Vogel IPR curves for each well
            - Estimated reservoir pressure per well
            - Qmax estimates (recent, lowest BHP, median)
            - Downloadable CSV matching the multi-well optimization template

            ### Requirements
            - Wells must have BHP gauge tags in bhp_dict.csv
            - Databricks connectivity for BHP data queries
            - At least 2 well tests per well for reliable IPR estimation
            """
            )
        return

    # --- Parse FDC CSV ---
    try:
        processor = WellTestProcessor(uploaded_fdc)
        processor.parse()
        all_wells = processor.get_unique_wells()
        test_counts = processor.get_test_count_per_well()
        date_range = processor.get_test_date_range()
    except Exception as e:
        st.error(f"❌ Error parsing FDC CSV: {str(e)}")
        st.exception(e)
        return

    date_min = date_range[0].strftime("%Y-%m-%d")
    date_max = date_range[1].strftime("%Y-%m-%d")
    st.success(f"✅ Parsed {len(all_wells)} wells from CSV ({date_min} to {date_max})")

    # --- Well selection ---
    st.subheader("Select Wells")

    use_all = st.checkbox("Use All Wells", value=True)

    if use_all:
        selected_wells = all_wells
    else:
        selected_wells = st.multiselect(
            "Select wells to analyze:",
            options=all_wells,
            default=all_wells,
            help="Choose which wells to include in the IPR analysis",
        )

    if not selected_wells:
        st.warning("⚠️ Please select at least one well.")
        return

    # Show test count summary
    with st.expander(f"📊 Test Counts ({len(selected_wells)} wells selected)"):
        filtered_counts = test_counts[test_counts.index.isin(selected_wells)]
        st.dataframe(
            filtered_counts.reset_index().rename(columns={"index": "Well", 0: "Tests"}),
            use_container_width=True,
            hide_index=True,
        )

    # --- Load tag dictionary ---
    try:
        if uploaded_tags is not None:
            tag_dict = load_tag_dict(uploaded_tags)
            st.info("📋 Using custom tag file")
        else:
            tag_dict = load_tag_dict()
            st.info("📋 Using bundled bhp_dict.csv")

        found_tags, missing_wells = get_tags_for_wells(selected_wells, tag_dict)

        if missing_wells:
            st.warning(
                f"⚠️ {len(missing_wells)} wells not found in tag dictionary and will be skipped: "
                f"{', '.join(missing_wells)}"
            )

        wells_with_tags = list(found_tags.keys())
        if not wells_with_tags:
            st.error("❌ No selected wells have SCADA tag mappings. Cannot proceed.")
            return

    except Exception as e:
        st.error(f"❌ Error loading tag dictionary: {str(e)}")
        return

    # --- Run Analysis ---
    if st.button("🚀 Run Well Test Analysis", type="primary", use_container_width=True):
        try:
            # Step 1: Query Databricks for BHP data (cached for 24 hours)
            st.write("### Step 1: Loading BHP Data")

            # Build hashable inputs for cache key
            tag_dict_frozen = tuple((well, *tags) for well, tags in found_tags.items())
            wells_tuple = tuple(sorted(wells_with_tags))

            with st.spinner(
                "Loading BHP data (using cache if available, " "otherwise querying Databricks — may take 1-2 min)..."
            ):
                bhp_data = _cached_bhp_query(tag_dict_frozen, wells_tuple)

            wells_with_bhp = list(bhp_data.keys())
            if not wells_with_bhp:
                st.error("❌ No BHP data returned. Check tag mappings and Databricks connectivity.")
                return

            st.success(
                f"✅ BHP data loaded for {len(wells_with_bhp)} wells "
                f"(cached for 24h — use sidebar 'Force fresh query' to clear)"
            )

            # Step 2: Merge test data with BHP
            st.write("### Step 2: Merging Test Data with BHP")
            with st.spinner("Merging well test data with BHP gauge data..."):
                well_tests = processor.get_well_tests(wells_with_bhp)
                merged_data = merge_tests_with_bhp(wells_with_bhp, bhp_data, well_tests)

            if merged_data.empty:
                st.error("❌ No matching dates found between well tests and BHP data.")
                return

            wells_in_merged = merged_data["well"].nunique()
            st.success(f"✅ Merged data: {len(merged_data)} test points across {wells_in_merged} wells")

            # Step 3: Estimate reservoir pressure
            st.write("### Step 3: Estimating Reservoir Pressure")
            with st.spinner("Estimating optimal reservoir pressure per well..."):
                jp_chars = _load_jp_chars()
                merged_with_rp = estimate_reservoir_pressure(
                    merged_data,
                    max_pres_schrader=max_rp_schrader,
                    max_pres_kuparuk=max_rp_kuparuk,
                    jp_chars=jp_chars,
                )

            st.success("✅ Reservoir pressure estimation complete")

            # Step 4: Compute Vogel coefficients
            st.write("### Step 4: Computing Vogel IPR Coefficients")
            with st.spinner("Computing Vogel IPR parameters..."):
                vogel_coeffs = compute_vogel_coefficients(merged_with_rp, resp_modifier=resp_modifier)

            if vogel_coeffs.empty:
                st.error("❌ Could not compute Vogel coefficients for any wells.")
                return

            st.success(f"✅ Computed IPR parameters for {len(vogel_coeffs)} wells")

            # Step 5: Generate IPR curves
            ipr_curves = generate_ipr_curves(vogel_coeffs)

            # Store results in session state
            st.session_state["wt_vogel_coeffs"] = vogel_coeffs
            st.session_state["wt_ipr_curves"] = ipr_curves
            st.session_state["wt_merged_data"] = merged_with_rp
            st.session_state["wt_analysis_complete"] = True

        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            st.exception(e)
            return

def _render_summary_tab(vogel_coeffs: pd.DataFrame, merged_data: pd.DataFrame):
    """Render the Summary tab."""
    st.write("### Field-Level Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Wells Analyzed", len(vogel_coeffs))
        avg_rp = vogel_coeffs["ResP"].mean()
        st.metric("Avg Reservoir Pressure", f"{avg_rp:.0f} psi")

    with col2:
        avg_qmax = vogel_coeffs["QMax_recent"].mean()
        st.metric("Avg Qmax (Recent)", f"{avg_qmax:.0f} BPD")
        total_tests = merged_data["well"].value_counts().sum()
        st.metric("Total Test Points", int(total_tests))

    with col3:
        min_rp = vogel_coeffs["ResP"].min()
        st.metric("Min Reservoir Pressure", f"{min_rp:.0f} psi")
        max_rp = vogel_coeffs["ResP"].max()
        st.metric("Max Reservoir Pressure", f"{max_rp:.0f} psi")

    with col4:
        avg_wc = vogel_coeffs["form_wc"].mean()
        st.metric("Avg Watercut", f"{avg_wc:.1%}")
        wells_few_tests = (vogel_coeffs["num_tests"] < 3).sum()
        st.metric("Wells < 3 Tests", int(wells_few_tests))

    # Reservoir pressure comparison chart
    st.write("### Reservoir Pressure Comparison")
    fig_rp = create_rp_comparison_chart(vogel_coeffs)
    st.pyplot(fig_rp)
    plt.close()

    # Qmax comparison chart
    st.write("### Qmax Comparison")
    fig_qmax = create_qmax_comparison_chart(vogel_coeffs)
    st.pyplot(fig_qmax)
    plt.close()


def _render_ipr_curves_tab(ipr_curves, merged_data, vogel_coeffs):
    """Render the IPR Curves tab with interactive Plotly charts.

    Shows one well at a time with a dropdown selector.
    Each chart is full-size, zoomable, and hoverable.
    """
    st.write("### Vogel IPR Curves")
    st.caption("Interactive charts — zoom, pan, and hover for details. Use the dropdown to switch wells.")

    if not ipr_curves:
        st.warning("No IPR curves to display.")
        return

    wells = sorted(ipr_curves.keys())

    # Well selector
    col_sel, col_nav = st.columns([3, 1])
    with col_sel:
        selected_well = st.selectbox(
            "Select Well:",
            wells,
            key="ipr_well_selector",
            help="Choose a well to view its IPR curve",
        )
    with col_nav:
        st.write("")  # spacer
        st.write(f"**{wells.index(selected_well) + 1}** of **{len(wells)}** wells")

    # Render the selected well's IPR chart
    if selected_well and selected_well in ipr_curves:
        fig_single = create_ipr_plotly(selected_well, ipr_curves[selected_well], merged_data)
        st.plotly_chart(fig_single, use_container_width=True)

        # Show well-specific stats below the chart
        well_coeffs = vogel_coeffs[vogel_coeffs["Well"] == selected_well]
        if not well_coeffs.empty:
            row = well_coeffs.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Res Pressure", f"{row['ResP']:.0f} psi")
            c2.metric("Qmax (Recent)", f"{row['QMax_recent']:.0f} BPD")
            c3.metric("Test BHP", f"{row['pwf']:.0f} psi")
            c4.metric("# Tests", int(row["num_tests"]))

    # Export section
    st.write("---")
    st.write("### 📄 Export All IPR Curves")

    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        st.caption("**Grid PNG** — High-res image of all wells in a grid. Download and zoom in.")
        if st.button("🖼️ Generate Grid PNG", use_container_width=True):
            with st.spinner(f"Rendering {len(wells)} wells at 200 DPI..."):
                png_bytes = create_ipr_grid_png(ipr_curves, merged_data, dpi=200)
            st.download_button(
                label="📥 Download Grid PNG",
                data=png_bytes,
                file_name="vogel_ipr_grid.png",
                mime="image/png",
                use_container_width=True,
            )

    with exp_col2:
        st.caption("**Multi-page PDF** — One full-page plot per well, suitable for printing.")
        if st.button("📄 Generate PDF", use_container_width=True):
            with st.spinner(f"Generating PDF for {len(wells)} wells..."):
                pdf_bytes = create_ipr_pdf(ipr_curves, merged_data)
            st.download_button(
                label="📥 Download IPR Curves PDF",
                data=pdf_bytes,
                file_name="vogel_ipr_curves.pdf",
                mime="application/pdf",
                use_container_width=True,
            )


def _render_well_details_tab(vogel_coeffs, merged_data):
    """Render the Well Details tab."""
    st.write("### Per-Well Results")

    # Format the display DataFrame
    display_df = vogel_coeffs.copy()
    display_df = display_df.rename(
        columns={
            "ResP": "Res Pressure (psi)",
            "QMax_recent": "Qmax Recent (BPD)",
            "QMax_lowest_bhp": "Qmax Low BHP (BPD)",
            "QMax_median": "Qmax Median (BPD)",
            "qwf": "Test Fluid Rate (BPD)",
            "pwf": "Test BHP (psi)",
            "form_wc": "Watercut",
            "num_tests": "# Tests",
            "most_recent_date": "Most Recent Test",
        }
    )

    st.dataframe(display_df, use_container_width=True, height=400, hide_index=True)

    # Show merged raw data
    with st.expander("📋 Raw Merged Data"):
        st.dataframe(merged_data, use_container_width=True, height=300)


def _render_export_tab(vogel_coeffs):
    """Render the Export tab with editable table, CSV download, and send-to-multi-well."""
    st.write("### Export for Multi-Well Optimization")

    st.markdown(
        "Edit values below, then download CSV or send directly to the "
        "**Multi-Well Optimization** page."
    )

    # Generate template
    template_df = export_optimization_template(vogel_coeffs)

    # Editable table
    edited_df = st.data_editor(
        template_df,
        use_container_width=True,
        hide_index=True,
        disabled=["Well"],
        key="export_editor",
    )

    # Action buttons
    col_send, col_download = st.columns(2)

    with col_send:
        if st.button("Send to Multi-Well Optimization", type="primary", use_container_width=True):
            st.session_state["wt_export_for_multiwell"] = edited_df
            st.success(f"Sent {len(edited_df)} wells. Switch to the Multi-Well Optimization page to load them.")

    with col_download:
        csv_data = edited_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="well_test_optimization_template.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Raw Vogel coefficients
    with st.expander("Raw Vogel Coefficients"):
        st.dataframe(vogel_coeffs, use_container_width=True, hide_index=True)
        vogel_csv = vogel_coeffs.to_csv(index=False)
        st.download_button(
            label="Download Vogel Coefficients CSV",
            data=vogel_csv,
            file_name="vogel_coefficients.csv",
            mime="text/csv",
            use_container_width=True,
        )
