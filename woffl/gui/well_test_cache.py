"""Cached Databricks well-test query helpers.

Extracted from the now-deleted ``well_test_page.py`` module so the
workflow pages can keep their @st.cache_data wrappers without dragging
the legacy standalone "Well Test Analysis" page along with them.

The 24-hour TTL means the cache survives most user sessions on the
Databricks Apps single-process runtime — see CLAUDE.md "Caching scope".
"""

import pandas as pd
import streamlit as st

from woffl.assembly.well_test_client import fetch_milne_well_tests, get_mpu_well_names


@st.cache_data(ttl=86400, max_entries=4, show_spinner=False)
def _cached_mpu_well_names() -> list[str]:
    """Cached list of all MPU well names from Databricks."""
    return get_mpu_well_names()


@st.cache_data(ttl=86400, max_entries=64, show_spinner=False)
def _cached_well_test_query(
    start_date: str, end_date: str, well_names_tuple: tuple
) -> tuple[pd.DataFrame, list[str]]:
    """Cached wrapper around fetch_milne_well_tests.

    Returns ``(DataFrame, dropped_wells)``. DataFrame columns match
    ipr_analyzer expectations: well, WtDate, BHP, WtTotalFluid, WtOilVol, etc.
    dropped_wells lists wells removed due to missing BHP or fluid rate data.
    """
    return fetch_milne_well_tests(start_date, end_date, list(well_names_tuple))
