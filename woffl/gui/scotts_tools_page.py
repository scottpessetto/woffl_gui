"""Scott's Tools — hidden utilities that run on the Databricks server.

Unlocked via the ?scott query param easter egg.
"""

import streamlit as st


def run_scotts_tools_page():
    """Render the Scott's Tools page."""
    st.title("Scott's Tools")
    st.caption("You found the secret menu.")

    tab_labels = ["Tool 1", "Tool 2", "Tool 3"]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        st.header("Tool 1")
        st.info("Add your first tool here.")

    with tabs[1]:
        st.header("Tool 2")
        st.info("Add your second tool here.")

    with tabs[2]:
        st.header("Tool 3")
        st.info("Add your third tool here.")
