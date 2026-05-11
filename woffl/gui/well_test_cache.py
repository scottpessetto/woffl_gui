"""Cached Databricks well-test query helpers.

Extracted from the now-deleted ``well_test_page.py`` module so the
workflow pages can keep their @st.cache_data wrappers without dragging
the legacy standalone "Well Test Analysis" page along with them.

The 24-hour TTL means the cache survives most user sessions on the
Databricks Apps single-process runtime — see CLAUDE.md "Caching scope".
"""

import pandas as pd
import streamlit as st

from woffl.assembly.databricks_client import query_bhp_for_well_tests
from woffl.assembly.well_test_client import (
    fetch_milne_well_tests,
    get_mpu_well_names,
)


@st.cache_data(ttl=86400, show_spinner=False)
def _cached_bhp_query(
    tag_dict_frozen: tuple,
    wells_tuple: tuple,
) -> dict:
    """Cached wrapper around query_bhp_for_well_tests.

    Args:
        tag_dict_frozen: Tuple of (well, bhp_tag, headerP_tag, whp_tag) for
            hashability.
        wells_tuple: Tuple of well names (hashable).

    Returns:
        Dictionary mapping well name to BHP DataFrame.
    """
    tag_dict = {well: (bhp, hp, whp) for well, bhp, hp, whp in tag_dict_frozen}
    return query_bhp_for_well_tests(tag_dict, list(wells_tuple))


@st.cache_data(ttl=86400, show_spinner=False)
def _cached_mpu_well_names() -> list[str]:
    """Cached list of all MPU well names from Databricks."""
    return get_mpu_well_names()


@st.cache_data(ttl=86400, show_spinner=False)
def _cached_well_test_query(
    start_date: str, end_date: str, well_names_tuple: tuple
) -> tuple[pd.DataFrame, list[str]]:
    """Cached wrapper around fetch_milne_well_tests.

    Returns ``(DataFrame, dropped_wells)``. DataFrame columns match
    ipr_analyzer expectations: well, WtDate, BHP, WtTotalFluid, WtOilVol, etc.
    dropped_wells lists wells removed due to missing BHP or fluid rate data.
    """
    return fetch_milne_well_tests(start_date, end_date, list(well_names_tuple))
