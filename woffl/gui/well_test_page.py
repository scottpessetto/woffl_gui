"""Well Test Analysis Page

Third mode in the WOFFL GUI. Allows users to:
1. Upload FDC well test CSV
2. Select wells to analyze
3. Query Databricks for BHP data (cached for 24 hours)
4. Compute Vogel IPR parameters
5. Visualize IPR curves
6. Download optimization template CSV
"""

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from woffl.assembly.databricks_client import (
    get_tags_for_wells,
    load_tag_dict,
    query_bhp_for_well_tests,
)
from woffl.assembly.well_test_client import (
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
from woffl.gui.utils import load_well_characteristics


def run_well_test_analysis_page():
    """Render the Well Test Analysis page."""

    st.title("Well Test Analysis")

    st.markdown("""
    Upload FDC well test data, query Databricks for BHP gauge data,
    and generate Vogel IPR parameters for use in multi-well optimization.
    """)

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
            "BHP query results are cached for 24 hours. "
            "The cache persists across users while the app is running."
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

        tab_labels = ["📊 Summary", "📈 IPR Curves", "📋 Well Details", "💾 Export"]
        has_jp_history = "jp_history_df" in st.session_state
        if has_jp_history:
            tab_labels.append("🔍 Model Check")

        tabs = st.tabs(tab_labels)

        with tabs[0]:
            _render_summary_tab(vogel_coeffs, merged_with_rp)

        with tabs[1]:
            _render_ipr_curves_tab(ipr_curves, merged_with_rp, vogel_coeffs)

        with tabs[2]:
            _render_well_details_tab(vogel_coeffs, merged_with_rp)

        with tabs[3]:
            _render_export_tab(vogel_coeffs)

        if has_jp_history:
            with tabs[4]:
                _render_model_check_tab(vogel_coeffs, merged_with_rp)


def _run_restls_path(max_rp_schrader, max_rp_kuparuk, resp_modifier):
    """Databricks data source path — loads well tests directly."""
    from datetime import date, timedelta

    # --- Pad & well selection (before query) ---
    # Fetch well names (cached) to build pad list
    try:
        with st.spinner("Fetching well list from Databricks..."):
            db_well_names = _cached_mpu_well_names()
    except Exception as e:
        st.error(f"Could not fetch well names from Databricks: {e}")
        return

    # Filter to only JP wells (those in jp_chars.csv) — ESP wells lack JP TVDs
    from woffl.assembly.well_test_client import _normalize_well_name
    from woffl.gui.utils import load_well_characteristics

    jp_wells = set(load_well_characteristics()["Well"].tolist())
    all_well_names = [w for w in db_well_names if _normalize_well_name(w) in jp_wells]

    if not all_well_names:
        st.warning(
            "No well names returned from Databricks. Check connectivity and try 'Force fresh Databricks query' in the sidebar."
        )
        return

    all_pads = get_pad_names(all_well_names)

    with st.sidebar:
        st.divider()
        st.subheader("Pad Selection")
        selected_pads = []
        for pad in all_pads:
            pad_wells = filter_wells_by_pad(all_well_names, [pad])
            if st.checkbox(
                f"Pad {pad} ({len(pad_wells)})", value=False, key=f"pad_{pad}"
            ):
                selected_pads.append(pad)

    if not selected_pads:
        st.warning("Select at least one pad in the sidebar.")
        return

    filtered_well_names = filter_wells_by_pad(all_well_names, selected_pads)

    # Date range pickers
    col_start, col_end = st.columns(2)
    with col_start:
        default_start = date.today() - timedelta(days=90)
        start_date = st.date_input(
            "Start Date", value=default_start, key="db_start_date"
        )
    with col_end:
        end_date = st.date_input("End Date", value=date.today(), key="db_end_date")

    if start_date > end_date:
        st.error("Start date must be before end date.")
        return

    st.caption(f"{len(filtered_well_names)} wells across {len(selected_pads)} pads")

    if st.button(
        "🔄 Load Well Tests from Databricks", type="primary", use_container_width=True
    ):
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
        st.info(
            "Select pads in the sidebar, then click the button above to load well tests."
        )
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
    if st.button(
        "🚀 Run Well Test Analysis",
        type="primary",
        use_container_width=True,
        key="restls_run",
    ):
        try:
            merged_data = df.copy()

            if merged_data.empty:
                st.error("❌ No test data for selected wells.")
                return

            wells_in_data = merged_data["well"].nunique()
            st.success(
                f"✅ {len(merged_data)} test points across {wells_in_data} wells (BHP included)"
            )

            if not _run_ipr_analysis(
                merged_data, max_rp_schrader, max_rp_kuparuk, resp_modifier
            ):
                return

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
            st.markdown("""
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
            """)
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
                "Loading BHP data (using cache if available, "
                "otherwise querying Databricks — may take 1-2 min)..."
            ):
                bhp_data = _cached_bhp_query(tag_dict_frozen, wells_tuple)

            wells_with_bhp = list(bhp_data.keys())
            if not wells_with_bhp:
                st.error(
                    "❌ No BHP data returned. Check tag mappings and Databricks connectivity."
                )
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
            st.success(
                f"✅ Merged data: {len(merged_data)} test points across {wells_in_merged} wells"
            )

            # Step 3-5: Run IPR analysis pipeline
            if not _run_ipr_analysis(
                merged_data, max_rp_schrader, max_rp_kuparuk, resp_modifier
            ):
                return

        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            st.exception(e)
            return


def _run_ipr_analysis(
    merged_data, max_rp_schrader, max_rp_kuparuk, resp_modifier
) -> bool:
    """Run the shared IPR analysis pipeline: estimate RP, compute Vogel, generate curves.

    Stores results in session state on success.

    Returns:
        True on success, False on failure.
    """
    st.write("### Estimating Reservoir Pressure")
    with st.spinner("Estimating optimal reservoir pressure per well..."):
        jp_chars = load_well_characteristics()
        merged_with_rp = estimate_reservoir_pressure(
            merged_data,
            max_pres_schrader=max_rp_schrader,
            max_pres_kuparuk=max_rp_kuparuk,
            jp_chars=jp_chars,
        )

    st.success("✅ Reservoir pressure estimation complete")

    st.write("### Computing Vogel IPR Coefficients")
    with st.spinner("Computing Vogel IPR parameters..."):
        vogel_coeffs = compute_vogel_coefficients(
            merged_with_rp, resp_modifier=resp_modifier
        )

    if vogel_coeffs.empty:
        st.error("❌ Could not compute Vogel coefficients for any wells.")
        return False

    st.success(f"✅ Computed IPR parameters for {len(vogel_coeffs)} wells")

    ipr_curves = generate_ipr_curves(vogel_coeffs)

    st.session_state["wt_vogel_coeffs"] = vogel_coeffs
    st.session_state["wt_ipr_curves"] = ipr_curves
    st.session_state["wt_merged_data"] = merged_with_rp
    st.session_state["wt_analysis_complete"] = True
    return True


def _render_model_check_tab(vogel_coeffs: pd.DataFrame, merged_with_rp: pd.DataFrame):
    """Render the Model Check tab.

    Runs the jetpump solver for each well using its current JP from history
    and IPR-derived inflow, then compares modeled vs actual production.
    """
    from woffl.assembly.jp_history import get_current_pump
    from woffl.gui.utils import (
        create_inflow,
        create_jetpump,
        create_pipes,
        create_reservoir_mix,
        create_well_profile,
        is_valid_number,
        load_well_characteristics,
        run_jetpump_solver,
    )

    st.write("### Model Check")
    st.caption(
        "Runs the WOFFL solver for each well using its current JP (from JP History) "
        "and IPR-derived inflow parameters, then compares modeled vs actual production."
    )

    jp_hist = st.session_state.get("jp_history_df")
    if jp_hist is None:
        st.warning(
            "JP History not loaded. Upload a JP History file on the main page to use Model Check."
        )
        return

    jp_chars = load_well_characteristics()

    # --- Configuration ---
    with st.expander("Model Parameters", expanded=False):
        st.caption("These apply to all wells during the model check.")
        cfg_col1, cfg_col2, cfg_col3 = st.columns(3)
        with cfg_col1:
            mc_surf_pres = st.number_input(
                "Surface Pressure (psi)",
                value=210,
                min_value=10,
                max_value=600,
                step=10,
                key="mc_surf_pres",
            )
            mc_rho_pf = st.number_input(
                "PF Density (lbm/ft³)",
                value=62.4,
                min_value=50.0,
                max_value=70.0,
                step=0.1,
                key="mc_rho_pf",
            )
        with cfg_col2:
            mc_ppf_surf = st.number_input(
                "PF Pressure (psi)",
                value=3168,
                min_value=1500,
                max_value=4000,
                step=10,
                key="mc_ppf_surf",
            )
            mc_default_gor = st.number_input(
                "Default GOR (scf/bbl)",
                value=250,
                min_value=20,
                max_value=10000,
                step=25,
                key="mc_default_gor",
                help="Used when GOR is not available from well test data",
            )
        with cfg_col3:
            mc_ken = st.number_input(
                "ken",
                value=0.03,
                min_value=0.01,
                max_value=0.10,
                step=0.01,
                format="%.2f",
                key="mc_ken",
            )
            mc_kth = st.number_input(
                "kth",
                value=0.30,
                min_value=0.10,
                max_value=0.50,
                step=0.10,
                format="%.1f",
                key="mc_kth",
            )
            mc_kdi = st.number_input(
                "kdi",
                value=0.40,
                min_value=0.10,
                max_value=0.50,
                step=0.10,
                format="%.1f",
                key="mc_kdi",
            )

    if not st.button(
        "Run Model Check", type="primary", use_container_width=True, key="mc_run"
    ):
        st.info(
            "Click the button above to run the solver for all wells and compare modeled vs actual."
        )
        return

    wells = sorted(vogel_coeffs["Well"].tolist())
    rows = []
    skipped = []
    progress_bar = st.progress(0)
    status = st.empty()

    for i, well_name in enumerate(wells):
        status.text(f"Solving {well_name}... ({i + 1}/{len(wells)})")
        progress_bar.progress((i + 1) / len(wells))

        # 1. Look up current JP from history
        current_pump = get_current_pump(jp_hist, well_name)
        if (
            current_pump is None
            or not current_pump["nozzle_no"]
            or not current_pump["throat_ratio"]
        ):
            skipped.append((well_name, "No JP in history"))
            continue

        nozzle = current_pump["nozzle_no"]
        throat = current_pump["throat_ratio"]

        # 2. Look up well characteristics from jp_chars
        well_row = (
            jp_chars[jp_chars["Well"] == well_name]
            if not jp_chars.empty
            else pd.DataFrame()
        )
        if not well_row.empty:
            wr = well_row.iloc[0]
            form_temp = int(wr.get("form_temp", 70))
            field_model = "Schrader" if wr.get("is_sch", True) else "Kuparuk"
            jp_tvd = int(wr.get("JP_TVD", 4065))
            tubing_od = float(wr.get("out_dia", 4.5))
            tubing_thick = float(wr.get("thick", 0.5))
        else:
            form_temp = 70
            field_model = "Schrader"
            jp_tvd = 4065
            tubing_od = 4.5
            tubing_thick = 0.5

        # 3. Get IPR parameters from vogel_coeffs
        coeff_row = vogel_coeffs[vogel_coeffs["Well"] == well_name].iloc[0]
        qwf = coeff_row["qwf"]
        pwf = coeff_row["pwf"]
        res_pres = coeff_row["ResP"]
        form_wc = coeff_row["form_wc"]

        # 4. Get GOR from most recent well test (if available)
        well_tests = merged_with_rp[merged_with_rp["well"] == well_name].sort_values(
            "WtDate", ascending=False
        )
        recent_test = well_tests.iloc[0] if not well_tests.empty else None

        gor = mc_default_gor
        if recent_test is not None and "fgor" in well_tests.columns:
            test_gor = recent_test.get("fgor", None)
            if is_valid_number(test_gor) and test_gor > 0:
                gor = int(test_gor)

        # 5. Create simulation objects
        try:
            jp = create_jetpump(nozzle, throat, mc_ken, mc_kth, mc_kdi)
            _, _, wellbore = create_pipes(tubing_od, tubing_thick)
            wp = create_well_profile(field_model, jp_tvd)
            oil_qwf = qwf * (1 - form_wc)
            ipr = create_inflow(oil_qwf, pwf, res_pres)
            rm = create_reservoir_mix(form_wc, gor, form_temp, field_model)
        except Exception as e:
            skipped.append((well_name, f"Object creation: {e}"))
            continue

        # 6. Run solver
        try:
            result = run_jetpump_solver(
                mc_surf_pres,
                form_temp,
                mc_rho_pf,
                mc_ppf_surf,
                jp,
                wellbore,
                wp,
                ipr,
                rm,
                field_model=field_model,
            )
            if result is None:
                skipped.append((well_name, "Solver returned None"))
                continue
            psu, sonic, modeled_oil, fwat, modeled_pf, mach = result
        except Exception:
            skipped.append((well_name, "Solver failed"))
            continue

        # 7. Get actual values from most recent test
        actual_oil = (
            recent_test.get("WtOilVol", None) if recent_test is not None else None
        )
        actual_bhp = recent_test.get("BHP", None) if recent_test is not None else None
        actual_pf = (
            recent_test.get("lift_wat", None) if recent_test is not None else None
        )

        row = {
            "Well": well_name,
            "JP": f"{nozzle}{throat}",
            "Field": field_model[:3],
            "GOR": gor,
            "Modeled Oil": round(modeled_oil),
            "Actual Oil": round(actual_oil) if is_valid_number(actual_oil) else None,
            "Modeled BHP": round(psu),
            "Actual BHP": round(actual_bhp) if is_valid_number(actual_bhp) else None,
            "Modeled PF": round(modeled_pf),
            "Actual PF": round(actual_pf) if is_valid_number(actual_pf) else None,
            "Sonic": sonic,
        }
        rows.append(row)

    progress_bar.empty()
    status.empty()

    if not rows:
        st.warning(
            "No wells could be modeled. Check that JP History has matching wells."
        )
        if skipped:
            st.caption(f"Skipped: {', '.join(f'{w} ({r})' for w, r in skipped)}")
        return

    # --- Build results DataFrame ---
    df = pd.DataFrame(rows)

    # Compute deltas
    df["Delta Oil"] = df.apply(
        lambda r: (
            r["Modeled Oil"] - r["Actual Oil"] if r["Actual Oil"] is not None else None
        ),
        axis=1,
    )
    df["Delta BHP"] = df.apply(
        lambda r: (
            r["Modeled BHP"] - r["Actual BHP"] if r["Actual BHP"] is not None else None
        ),
        axis=1,
    )
    df["Delta PF"] = df.apply(
        lambda r: (
            r["Modeled PF"] - r["Actual PF"] if r["Actual PF"] is not None else None
        ),
        axis=1,
    )

    # --- Field totals ---
    st.write("### Field Totals")
    valid_oil = df.dropna(subset=["Actual Oil"])
    total_modeled_oil = df["Modeled Oil"].sum()
    total_actual_oil = valid_oil["Actual Oil"].sum() if not valid_oil.empty else 0
    total_modeled_pf = df["Modeled PF"].sum()
    valid_pf = df.dropna(subset=["Actual PF"])
    total_actual_pf = valid_pf["Actual PF"].sum() if not valid_pf.empty else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Wells Modeled", len(df))
    with col2:
        st.metric("Total Modeled Oil", f"{total_modeled_oil:,.0f} BOPD")
    with col3:
        st.metric(
            "Total Actual Oil",
            f"{total_actual_oil:,.0f} BOPD" if total_actual_oil > 0 else "N/A",
        )
    with col4:
        if total_actual_oil > 0:
            st.metric(
                "Total Delta Oil", f"{total_modeled_oil - total_actual_oil:+,.0f} BOPD"
            )
        else:
            st.metric("Total Delta Oil", "N/A")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Sonic Wells", f"{df['Sonic'].sum()}/{len(df)}")
    with col2:
        st.metric("Total Modeled PF", f"{total_modeled_pf:,.0f} BWPD")
    with col3:
        st.metric(
            "Total Actual PF",
            f"{total_actual_pf:,.0f} BWPD" if total_actual_pf > 0 else "N/A",
        )
    with col4:
        if total_actual_pf > 0:
            st.metric(
                "Total Delta PF", f"{total_modeled_pf - total_actual_pf:+,.0f} BWPD"
            )
        else:
            st.metric("Total Delta PF", "N/A")

    # --- Results table ---
    st.write("### Per-Well Results")

    # Format for display
    display_df = df.copy()
    display_cols = [
        "Well",
        "JP",
        "Field",
        "GOR",
        "Modeled Oil",
        "Actual Oil",
        "Delta Oil",
        "Modeled BHP",
        "Actual BHP",
        "Delta BHP",
        "Modeled PF",
        "Actual PF",
        "Delta PF",
        "Sonic",
    ]
    display_df = display_df[display_cols]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # --- Skipped wells ---
    if skipped:
        with st.expander(f"Skipped Wells ({len(skipped)})"):
            for well_name, reason in skipped:
                st.write(f"- **{well_name}**: {reason}")

    # --- Download ---
    csv_data = df.to_csv(index=False)
    st.download_button(
        label="Download Model Check CSV",
        data=csv_data,
        file_name="model_check_results.csv",
        mime="text/csv",
        use_container_width=True,
        key="mc_download",
    )


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
    st.caption(
        "Interactive charts — zoom, pan, and hover for details. Use the dropdown to switch wells."
    )

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
        fig_single = create_ipr_plotly(
            selected_well,
            ipr_curves[selected_well],
            merged_data,
            form_wc=st.session_state.get("form_wc"),
        )
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
        st.caption(
            "**Grid PNG** — High-res image of all wells in a grid. Download and zoom in."
        )
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
        st.caption(
            "**Multi-page PDF** — One full-page plot per well, suitable for printing."
        )
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
        if st.button(
            "Send to Multi-Well Optimization", type="primary", use_container_width=True
        ):
            st.session_state["wt_export_for_multiwell"] = edited_df
            st.success(
                f"Sent {len(edited_df)} wells. Switch to the Multi-Well Optimization page to load them."
            )

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
