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
@ttl_cache(config.TTL_WELL_TESTS, maxsize=8)
def fetch_all_well_tests(months: int) -> pd.DataFrame:
    """Fleet-wide well tests for the trailing ``months`` window.

    EVERY test, allocated and info-only (``allocated`` column). Only what a
    person picks from reads this frame whole; anything that selects or fits
    tests on its own takes ``allocated_only`` of it, as ``tests_for_well``
    does by default.

    Raises on Databricks failure so a blip is never cached.

    Args:
        months: lookback window in calendar months (relativedelta).

    Returns:
        Frame with well, wt_uid, WtDate, allocated, WtOilVol, WtWaterVol,
        WtGasVol, WtTotalFluid, form_wc, BHP, fgor, lift_wat, whp, pf_press,
        pf_source, dup_wt_uids.
    """
    from dateutil.relativedelta import relativedelta

    # A shorter window is a SLICE of a longer one already in cache - never a
    # second fleet query. The warm loop used to run "the biggest query in
    # the app" three times for the nested 6/12/24-month windows (review
    # 2026-09-01, DATA-10); now the longest warmed window feeds the rest.
    longest = _longest_cached_window(months)
    if longest is not None:
        cutoff = datetime.now() - relativedelta(months=months)
        big = fetch_all_well_tests(longest)  # cache hit by construction
        if big is not None and "WtDate" in big.columns:
            return big[pd.to_datetime(big["WtDate"]) >= pd.Timestamp(cutoff)].copy()

    return _fetch_window(months, datetime.now())


def _fetch_window(months: int, now: datetime) -> pd.DataFrame:
    from dateutil.relativedelta import relativedelta
    from woffl.assembly.well_test_client import fetch_milne_well_tests

    end_date = now.strftime("%Y-%m-%d")
    start_date = (now - relativedelta(months=months)).strftime("%Y-%m-%d")
    df, _dropped = fetch_milne_well_tests(start_date, end_date)
    return df


def warm_test_windows() -> None:
    """Prime every live lookback from one new fleet snapshot, never old cache data."""
    from dateutil.relativedelta import relativedelta

    windows = sorted(set(config.WARM_TEST_MONTHS), reverse=True)
    if not windows:
        return
    version = fetch_all_well_tests.cache_version()
    now = datetime.now()
    big = _fetch_window(windows[0], now)
    for months in windows:
        frame = big
        if months != windows[0] and "WtDate" in big.columns:
            cutoff = pd.Timestamp(now - relativedelta(months=months))
            frame = big[pd.to_datetime(big["WtDate"]) >= cutoff].copy()
        fetch_all_well_tests.cache_prime(frame, months, version=version)


def _longest_cached_window(months: int) -> Optional[int]:
    """The longest warmed window that covers ``months`` and is ALREADY in the
    fleet-tests cache, else None (then the caller queries the warehouse)."""
    try:
        from server.config import WARM_TEST_MONTHS

        candidates = sorted((m for m in WARM_TEST_MONTHS if m > months), reverse=True)
        for m in candidates:
            if fetch_all_well_tests.cache_has(m):  # type: ignore[attr-defined]
                return m
    except Exception:  # noqa: BLE001 - the cache probe is an optimization only
        return None
    return None


def allocated_only(df: pd.DataFrame) -> pd.DataFrame:
    """The allocated tests of a fleet/well frame (see well_test_client)."""
    from woffl.assembly.well_test_client import allocated_only as _allocated_only

    return _allocated_only(df)


def tests_for_well(
    well: str, months: int, cap: int = 0, include_info: bool = False
) -> Optional[pd.DataFrame]:
    """One well's tests from the shared fleet cache, newest kept under a cap.

    Soft-fail: Databricks down or no rows -> None (v1 drops the gauge and
    manual-test layers of the Streamlit helper).

    Args:
        well: GUI well name, e.g. "MPB-28".
        months: lookback window in months.
        cap: keep only the N most recent tests; 0 = no cap.
        include_info: also return info-only (unallocated) tests. Off by
            default: only a list the engineer picks from should carry them.

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
    sliced = all_tests[all_tests["well"] == well]
    if not include_info:
        sliced = allocated_only(sliced)
    sliced = sliced.copy()
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
    "allocated": "allocated",
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


def tests_json(
    well: str, months: int, cap: int = 0, include_info: bool = False
) -> list[dict[str, Any]]:
    """JSON-safe test rows, newest first. [] when the well has none.

    ``allocated`` is false on an info-only row and null on a frame that never
    carried the flag.
    """
    df = tests_for_well(well, months, cap, include_info)
    if df is None or df.empty:
        return []
    if "WtDate" in df.columns:
        df = df.sort_values("WtDate", ascending=False)
    return frames.records(df, _TEST_COLUMNS)
