"""Step 1: Select Wells — pad/date selection from Databricks or CSV upload."""

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from woffl.assembly.well_test_client import (
    filter_wells_by_pad,
    get_pad_names,
)
from woffl.gui.workflow_page import _clear_downstream


def render_step1():
    st.subheader("Step 1: Select Wells")

    data_source = st.radio(
        "Data Source",
        ["Databricks", "Upload CSV Template"],
        horizontal=True,
        key="uw_data_source_radio",
    )

    if data_source == "Databricks":
        _render_databricks_path()
    else:
        _render_csv_upload_path()


def _render_databricks_path():
    """Pad selection + date range → load well tests from Databricks."""
    from woffl.assembly.well_test_client import _normalize_well_name
    from woffl.gui.utils import load_well_characteristics
    from woffl.gui.well_test_page import _cached_mpu_well_names, _cached_well_test_query

    # Fetch well names and filter to JP wells
    try:
        with st.spinner("Fetching well list from Databricks..."):
            db_well_names = _cached_mpu_well_names()
    except Exception as e:
        st.error(f"Could not fetch well names from Databricks: {e}")
        return

    jp_wells = set(load_well_characteristics()["Well"].tolist())
    all_well_names = [w for w in db_well_names if _normalize_well_name(w) in jp_wells]

    if not all_well_names:
        st.warning("No JP wells found. Check Databricks connectivity and jp_chars.csv.")
        return

    all_pads = get_pad_names(all_well_names)

    # Pad selection
    st.write("### Select Pads")
    pad_cols = st.columns(min(len(all_pads), 6))
    selected_pads = []
    for i, pad in enumerate(all_pads):
        pad_wells = filter_wells_by_pad(all_well_names, [pad])
        with pad_cols[i % len(pad_cols)]:
            if st.checkbox(
                f"Pad {pad} ({len(pad_wells)})",
                value=False,
                key=f"uw_pad_{pad}",
            ):
                selected_pads.append(pad)

    if not selected_pads:
        st.info("Select at least one pad above.")
        return

    filtered_well_names = filter_wells_by_pad(all_well_names, selected_pads)

    # Date range
    col_start, col_end = st.columns(2)
    with col_start:
        default_start = date.today() - timedelta(days=90)
        start_date = st.date_input(
            "Start Date", value=default_start, key="uw_start_date"
        )
    with col_end:
        end_date = st.date_input("End Date", value=date.today(), key="uw_end_date")

    if start_date > end_date:
        st.error("Start date must be before end date.")
        return

    st.caption(f"{len(filtered_well_names)} wells across {len(selected_pads)} pads")

    if st.button(
        "Load Well Tests from Databricks",
        type="primary",
        use_container_width=True,
        key="uw_load_tests",
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

            # Clear downstream state and store results
            _clear_downstream(2)
            st.session_state["uw_well_test_df"] = df
            st.session_state["uw_dropped_wells"] = dropped_wells
            st.session_state["uw_csv_shortcut"] = False
            st.session_state["uw_current_step"] = 2
            st.session_state["uw_max_step_reached"] = max(
                st.session_state.get("uw_max_step_reached", 1), 2
            )
            st.rerun()

        except Exception as e:
            st.error(f"Error querying Databricks: {str(e)}")
            st.exception(e)

    # Show summary if data already loaded
    if "uw_well_test_df" in st.session_state and not st.session_state.get(
        "uw_csv_shortcut", False
    ):
        df = st.session_state["uw_well_test_df"]
        st.success(
            f"Well tests loaded: {len(df)} tests for {df['well'].nunique()} wells"
        )
        if st.button("Proceed to Review IPR →", key="uw_step1_proceed"):
            st.session_state["uw_current_step"] = 2
            st.session_state["uw_max_step_reached"] = max(
                st.session_state.get("uw_max_step_reached", 1), 2
            )
            st.rerun()


def _render_csv_upload_path():
    """Upload an optimization template CSV to skip Databricks + IPR analysis."""
    from woffl.assembly.network_optimizer import load_wells_from_dataframe

    uploaded = st.file_uploader(
        "Upload Optimization Template CSV",
        type=["csv"],
        key="uw_csv_upload",
        help="Upload a CSV with well configurations (same format as the optimization template). "
        "This skips Databricks loading and IPR analysis.",
    )

    if uploaded is not None:
        try:
            input_df = pd.read_csv(uploaded)
            well_configs = load_wells_from_dataframe(input_df)

            _clear_downstream(2)
            st.session_state["uw_well_configs"] = well_configs
            st.session_state["uw_template_df"] = input_df
            st.session_state["uw_csv_shortcut"] = True
            st.session_state["uw_current_step"] = 2
            st.session_state["uw_max_step_reached"] = max(
                st.session_state.get("uw_max_step_reached", 1), 2
            )

            st.success(f"Loaded {len(well_configs)} well configurations from CSV")
            st.rerun()

        except Exception as e:
            st.error(f"Error parsing CSV: {str(e)}")
