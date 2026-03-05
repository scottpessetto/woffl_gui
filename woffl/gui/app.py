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

import streamlit as st

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from woffl.assembly.jp_history import parse_jp_history
from woffl.gui.sidebar import render_sidebar
from woffl.gui.single_well_page import run_single_well_page, show_welcome_message


@st.cache_data(ttl=86400, show_spinner=False)
def _cached_all_well_tests(months_back: int = 3):
    """Fetch recent well tests for all MPU wells in one query. Cached 24h."""
    from datetime import datetime

    from dateutil.relativedelta import relativedelta

    from woffl.assembly.restls_client import fetch_milne_well_tests

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - relativedelta(months=months_back)).strftime("%Y-%m-%d")
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

    st.title("WOFFL Jetpump Simulator")

    # Global JP history uploader (available across all modes)
    with st.sidebar:
        jp_file = st.file_uploader("JP History (xlsx)", type=["xlsx"], key="jp_history_upload")
        if jp_file:
            st.session_state["jp_history_df"] = parse_jp_history(jp_file)
            st.caption(f"Loaded {len(st.session_state['jp_history_df'])} JP records")

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

    # Mode selection
    app_mode = st.radio(
        "Select Analysis Mode:",
        ["Single Well Analysis", "Multi-Well Optimization", "Well Test Analysis"],
        horizontal=True,
        help=(
            "Single Well: Analyze one well in detail. "
            "Multi-Well: Optimize pump sizing across multiple wells. "
            "Well Test: Generate Vogel IPR from well tests + Databricks BHP data."
        ),
    )

    if app_mode == "Well Test Analysis":
        from woffl.gui.well_test_page import run_well_test_analysis_page

        run_well_test_analysis_page()
        return

    if app_mode == "Multi-Well Optimization":
        from woffl.gui.multi_well_page import run_multi_well_optimization_page

        run_multi_well_optimization_page()
        return

    # Single Well Analysis mode
    run_button, params = render_sidebar()

    if run_button:
        run_single_well_page(params)
    else:
        show_welcome_message()


if __name__ == "__main__":
    main()
