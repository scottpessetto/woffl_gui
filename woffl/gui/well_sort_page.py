"""Top-level Well Sort page.

Hosts three sub-tabs:
  - **Wells** — the full online / offline / LTSI / 30-day-change view
    (the existing tool, formerly under Scott's Tools).
  - **Triage (beta)** — experimental keep / SI / BOL decision view driven by
    each well's water cut vs the field marginal WC. Runs alongside Wells
    (which is untouched) for back-to-back comparison.
  - **Marginal WC** — cumulative-water-threshold marginal water-cut
    calculator with an "import to sidebar" affordance.

All tabs share the same Databricks-cached helpers in
``woffl.gui.scotts_tools.well_sort``.
"""

import streamlit as st

from woffl.gui.scotts_tools import well_sort


def run_well_sort_page() -> None:
    """Render the Well Sort top-level page with its three inner tabs."""
    tab_wells, tab_triage, tab_marginal = st.tabs(
        ["Wells", "Triage (beta)", "Marginal WC"]
    )

    with tab_wells:
        well_sort.render_tab()

    with tab_triage:
        well_sort.render_triage_tab()

    with tab_marginal:
        well_sort.render_marginal_wc_tab()
