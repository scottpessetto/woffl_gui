"""Scott's Tools page orchestrator — hidden utilities behind the easter egg.

Routes the four sub-tabs to their respective render functions.
"""

import streamlit as st

from . import (
    jp_calibration,
    jp_fric_trend,
    jp_washout,
    pad_watercut,
    pf_scenario,
    well_sort,
)


def run_scotts_tools_page():
    """Render the Scott's Tools page."""
    st.title("Scott's Tools")
    st.caption("You found the secret menu.")

    tab_labels = [
        "PF Scenario Analysis",
        "JP Friction Calibration",
        "JP Fric Trend",
        "JP Wash-Out",
        "Well Sort",
        "Pad Water Cut",
    ]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        pf_scenario.render_tab()

    with tabs[1]:
        jp_calibration.render_tab()

    with tabs[2]:
        jp_fric_trend.render_tab()

    with tabs[3]:
        jp_washout.render_tab()

    with tabs[4]:
        well_sort.render_tab()

    with tabs[5]:
        pad_watercut.render_tab()
