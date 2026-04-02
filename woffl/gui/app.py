"""WOFFL Streamlit GUI Application

This is the main entry point for the WOFFL Streamlit GUI application.
It provides a web interface for interacting with the WOFFL package's jetpump functionality.

The application supports three modes:
- Single Well Analysis: Detailed analysis of one well with multiple visualization tabs
- Multi-Well Optimization: Optimize pump sizing across multiple wells
- Well Test Analysis: Generate Vogel IPR from well tests + Databricks BHP data
"""

import os
import sys
from pathlib import Path

import streamlit as st

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from woffl.assembly.jp_history import parse_jp_history
from woffl.gui.sidebar import render_sidebar
from woffl.gui.single_well_page import run_single_well_page, show_welcome_message

_JP_HISTORY_PATH = (
    Path(__file__).resolve().parent.parent.parent / "data" / "jetpump_history.xlsx"
)


@st.cache_data(ttl=86400, show_spinner=False)
def _cached_jp_history():
    """Fetch JP history from Databricks mpu_tracker. Cached 24h."""
    from woffl.assembly.databricks_client import fetch_jp_history

    return fetch_jp_history()


@st.cache_data(ttl=86400, show_spinner=False)
def _cached_all_well_tests(months_back: int = 3):
    """Fetch recent well tests for all MPU wells in one query. Cached 24h."""
    from datetime import datetime

    from dateutil.relativedelta import relativedelta

    from woffl.assembly.well_test_client import fetch_milne_well_tests

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - relativedelta(months=months_back)).strftime(
        "%Y-%m-%d"
    )
    df, _ = fetch_milne_well_tests(start_date, end_date)
    return df


def main():
    """Main function for the Streamlit application."""
    st.set_page_config(
        page_title="WOFFL Jetpump Simulator",
        page_icon="🛢️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("WOFFL Haus")
    st.caption("*Built on Kaelin Ellis's WOFFL Jet Pump Model*")

    # Global JP history — fetch from Databricks, fall back to bundled Excel
    with st.sidebar:
        if "jp_history_df" not in st.session_state:
            with st.spinner("Loading JP history from Databricks..."):
                try:
                    st.session_state["jp_history_df"] = _cached_jp_history()
                    st.session_state["jp_history_source"] = "Databricks"
                except Exception as e:
                    if _JP_HISTORY_PATH.exists():
                        st.session_state["jp_history_df"] = parse_jp_history(
                            str(_JP_HISTORY_PATH)
                        )
                        st.session_state["jp_history_source"] = "Excel (fallback)"
                        st.warning(f"Databricks unavailable, using bundled Excel: {e}")
                    else:
                        st.warning(f"Could not load JP history: {e}")

        if "jp_history_df" in st.session_state:
            source = st.session_state.get("jp_history_source", "")
            st.caption(
                f"JP History: {len(st.session_state['jp_history_df'])} records ({source})"
            )

        jp_file = st.file_uploader(
            "Upload JP History override (xlsx)",
            type=["xlsx"],
            key="jp_history_upload",
            help="Upload an Excel file to override the Databricks data for this session.",
        )
        if jp_file:
            st.session_state["jp_history_df"] = parse_jp_history(jp_file)
            st.session_state["jp_history_source"] = "Excel (uploaded)"
            st.success(
                f"Using uploaded JP History ({len(st.session_state['jp_history_df'])} records)"
            )

        if "jp_history_df" in st.session_state:
            # Pre-fetch all well tests once (cached 24h for all users)
            if "all_well_tests_df" not in st.session_state:
                with st.spinner("Fetching recent well tests from Databricks..."):
                    try:
                        st.session_state["all_well_tests_df"] = _cached_all_well_tests()
                        n = len(st.session_state["all_well_tests_df"])
                        st.caption(f"Loaded {n} well test records")
                    except Exception as e:
                        st.warning(f"Could not fetch well tests: {e}")
                        st.session_state["all_well_tests_df"] = None

    modes = ["Single Well Analysis", "Multi-Well Optimization", "Well Test Analysis", "Well Database"]
    if st.session_state.get("_scotts_tools", False):
        modes.append("Scott's Tools")

    # Mode selection
    app_mode = st.radio(
        "Select Analysis Mode:",
        modes,
        horizontal=True,
        help=(
            "Single Well: Analyze one well in detail. "
            "Multi-Well: Optimize pump sizing across multiple wells. "
            "Well Test: Generate Vogel IPR from well tests + Databricks BHP data."
        ),
    )

    if app_mode == "Well Database":
        from woffl.gui.well_database_page import run_well_database_page

        run_well_database_page()

    elif app_mode == "Well Test Analysis":
        from woffl.gui.well_test_page import run_well_test_analysis_page

        run_well_test_analysis_page()

    elif app_mode == "Multi-Well Optimization":
        from woffl.gui.multi_well_page import run_multi_well_optimization_page

        run_multi_well_optimization_page()

    elif app_mode == "Scott's Tools":
        from woffl.gui.scotts_tools_page import run_scotts_tools_page

        run_scotts_tools_page()

    else:
        # Single Well Analysis mode
        run_button, params = render_sidebar()

        if run_button:
            st.session_state.sw_sim_active = True

        if st.session_state.get("sw_sim_active", False):
            run_single_well_page(params)
        else:
            show_welcome_message()

    # Easter egg — renders at the very bottom of the sidebar (after all page-specific content)
    if not st.session_state.get("_scotts_tools", False):
        with st.sidebar:
            st.divider()
            code = st.text_input(
                "", placeholder="", label_visibility="collapsed", key="_egg_input"
            )
            if code.strip().lower() == "scott":
                st.session_state["_scotts_tools"] = True
                st.rerun()


if __name__ == "__main__":
    main()
