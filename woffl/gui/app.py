"""WOFFL Streamlit GUI Application

This is the main entry point for the WOFFL Streamlit GUI application.
It provides a web interface for interacting with the WOFFL package's jetpump functionality.

The application supports two modes:
- Single Well Analysis: Detailed analysis of one well with multiple visualization tabs
- Multi-Well Optimization: Optimize pump sizing across multiple wells
"""

import os
import sys

import streamlit as st

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from woffl.gui.sidebar import render_sidebar
from woffl.gui.single_well_page import run_single_well_page, show_welcome_message


def main():
    """Main function for the Streamlit application."""
    st.set_page_config(
        page_title="WOFFL Jetpump Simulator",
        page_icon="🛢️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("WOFFL Jetpump Simulator")

    # Mode selection
    app_mode = st.radio(
        "Select Analysis Mode:",
        ["Single Well Analysis", "Multi-Well Optimization"],
        horizontal=True,
        help="Single Well: Analyze one well in detail. Multi-Well: Optimize pump sizing across multiple wells.",
    )

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
