"""Well-test fetch, per-well slicing, and JSON projection.

qwf-convention reminder: WtTotalFluid is TOTAL LIQUID (BLPD) - the rate the
sidebar/SimParams ``qwf`` holds. v1 has no memory-gauge or manual-test
layers, so this is the shared-cache slice path of the Streamlit helper only.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import pandas as pd

from server import config
from server.cache import ttl_cache
from server.services import frames


# 8 windows cached - each entry is the FULL fleet's history for one lookback
# window (small: ~90 wells x tens of tests). Two windows are live in-tree (the
# 6-month default and evidence._min_test_bhp's 12), and /wells/{name}/tests
# accepts months 1..60, so a maxsize of 4 let a handful of ad-hoc requests
# evict a 24 h fleet query and force a full refetch.
# mirrors woffl/gui/utils.py:fetch_all_well_tests
@ttl_cache(config.TTL_WELL_TESTS, maxsize=8)
def fetch_all_well_tests(months: int) -> pd.DataFrame:
    """Fleet-wide well tests for the trailing ``months`` window.

    Raises on Databricks failure so a blip is never cached.

    Args:
        months: lookback window in calendar months (relativedelta).

    Returns:
        Frame with well, wt_uid, WtDate, WtOilVol, WtWaterVol, WtGasVol,
        WtTotalFluid, form_wc, BHP, fgor, lift_wat, whp, pf_press, pf_source.
    """
    from dateutil.relativedelta import relativedelta

    from woffl.assembly.well_test_client import fetch_milne_well_tests

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - relativedelta(months=months)).strftime("%Y-%m-%d")
    df, _dropped = fetch_milne_well_tests(start_date, end_date)
    return df


def tests_for_well(well: str, months: int, cap: int = 0) -> Optional[pd.DataFrame]:
    """One well's tests from the shared fleet cache, newest kept under a cap.

    Soft-fail: Databricks down or no rows -> None (v1 drops the gauge and
    manual-test layers of the Streamlit helper).
    # mirrors woffl/gui/utils.py:get_well_tests_for_well

    Args:
        well: GUI well name, e.g. "MPB-28".
        months: lookback window in months.
        cap: keep only the N most recent tests; 0 = no cap.

    Returns:
        Copy of the sliced frame, or None when the well has no tests.
    """
    if well == "Custom":
        return None
    try:
        all_tests = fetch_all_well_tests(months)
    except Exception:
        return None
    if all_tests is None or all_tests.empty:
        return None
    sliced = all_tests[all_tests["well"] == well].copy()
    if sliced.empty:
        return None
    if cap > 0 and "WtDate" in sliced.columns and len(sliced) > cap:
        sliced = (
            sliced.sort_values("WtDate", ascending=False)
            .head(cap)
            .reset_index(drop=True)
        )
    return sliced


# DataFrame column -> JSON key (the WellTestsResponse row contract).
_TEST_COLUMNS: dict[str, str] = {
    "wt_uid": "wt_uid",
    "WtDate": "date",
    "WtOilVol": "oil",
    "WtWaterVol": "water",
    "WtGasVol": "gas",
    "WtTotalFluid": "total_fluid",
    "form_wc": "form_wc",
    "BHP": "bhp",
    "fgor": "fgor",
    "lift_wat": "lift_wat",
    "whp": "whp",
    "pf_press": "pf_press",
    "pf_source": "pf_source",
}


def tests_json(well: str, months: int, cap: int = 0) -> list[dict[str, Any]]:
    """JSON-safe test rows, newest first. [] when the well has none."""
    df = tests_for_well(well, months, cap)
    if df is None or df.empty:
        return []
    if "WtDate" in df.columns:
        df = df.sort_values("WtDate", ascending=False)
    return frames.records(df, _TEST_COLUMNS)
