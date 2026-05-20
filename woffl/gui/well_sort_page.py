"""Top-level Well Sort page.

Hosts two sub-tabs:
  - **Wells** — the full online / offline / LTSI / 30-day-change view
    (the existing tool, formerly under Scott's Tools).
  - **Marginal WC** — cumulative-water-threshold marginal water-cut
    calculator with an "import to sidebar" affordance.

Both tabs share the same Databricks-cached helpers in
``woffl.gui.scotts_tools.well_sort``.
"""

import streamlit as st

from woffl.gui.scotts_tools import well_sort


def run_well_sort_page() -> None:
    """Render the Well Sort top-level page with its two inner tabs."""
    tab_wells, tab_marginal = st.tabs(["Wells", "Marginal WC"])

    with tab_wells:
        well_sort.render_tab()

    with tab_marginal:
        well_sort.render_marginal_wc_tab()
