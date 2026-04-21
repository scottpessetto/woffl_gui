"""Pad Water Cut Client

Builds a daily per-pad water-cut time series for pads G, H, I, J by:
1. Pulling allocated well tests from mpu.wells.vw_well_test (form_oil,
   form_wat, lift_wat).
2. Forward-filling each well's test values across every day in the range
   (a test value is the best estimate until the next test replaces it).
3. Pulling daily shut-in hours from mpu.wells.vw_shut_in and zeroing out
   any well-day where SUM(down_hours) > SI_HOURS_THRESHOLD — the well
   didn't contribute to pad flow that day.
4. Aggregating per pad per day, with the power-fluid recycle rule:
       G, J  (return to plant):  pad_water = form_wat + lift_wat
       H, I  (on-pad recycle):   pad_water = form_wat
   Combined ("All"): same rule applied per-well, then summed.

Returned DataFrame is long-format: date, pad, oil, water, wc.
"""

from __future__ import annotations

import pandas as pd

from woffl.assembly.databricks_client import execute_query

PADS = ("G", "H", "I", "J")
RECYCLE_PADS = {"H", "I"}
SI_HOURS_THRESHOLD = 6.0
TEST_LOOKBACK_DAYS = 365

_PAD_LIKE = " OR ".join(f"h.well_name LIKE '{p}-%'" for p in PADS)

TESTS_QUERY = f"""\
SELECT
    vwt.well_name,
    vwt.wt_date,
    vwt.form_oil AS oil_rate,
    vwt.form_wat AS fwat_rate,
    vwt.lift_wat AS lwat_rate
FROM mpu.wells.vw_well_test vwt
JOIN mpu.wells.vw_well_header h
    ON vwt.enthid = h.enthid
WHERE h.field = 'MPU'
    AND h.well_type = 'prod'
    AND ({_PAD_LIKE})
    AND vwt.allocated = True
    AND vwt.wt_date BETWEEN '{{tests_start}}' AND '{{end_date}}'
ORDER BY vwt.well_name, vwt.wt_date
"""

SHUT_IN_QUERY = f"""\
WITH pad_producers AS (
    SELECT enthid, well_name
    FROM mpu.wells.vw_well_header h
    WHERE h.field = 'MPU'
        AND h.well_type = 'prod'
        AND ({_PAD_LIKE})
)
SELECT
    s.well_name,
    s.dtdate,
    SUM(CAST(s.down_hours AS DOUBLE)) AS hrs
FROM mpu.wells.vw_shut_in s
JOIN pad_producers p ON s.dthid = p.enthid
WHERE s.dtdate BETWEEN '{{start_date}}' AND '{{end_date}}'
GROUP BY s.well_name, s.dtdate
"""


def _pad_of(well_name: str) -> str:
    return well_name[0] if well_name else ""


def _fetch_tests(start_date: str, end_date: str) -> pd.DataFrame:
    tests_start = (pd.Timestamp(start_date) - pd.Timedelta(days=TEST_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    query = TESTS_QUERY.format(tests_start=tests_start, end_date=end_date)
    df = execute_query(query)
    if df.empty:
        return df
    df["wt_date"] = pd.to_datetime(df["wt_date"], utc=True).dt.tz_localize(None).dt.normalize()
    for col in ("oil_rate", "fwat_rate", "lwat_rate"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df = df.sort_values(["well_name", "wt_date"])
    return df


def _fetch_shut_in(start_date: str, end_date: str) -> pd.DataFrame:
    query = SHUT_IN_QUERY.format(start_date=start_date, end_date=end_date)
    df = execute_query(query)
    if df.empty:
        return df
    df["dtdate"] = pd.to_datetime(df["dtdate"], utc=True).dt.tz_localize(None).dt.normalize()
    df["hrs"] = pd.to_numeric(df["hrs"], errors="coerce").fillna(0.0)
    return df


def _forward_fill_tests(tests_df: pd.DataFrame, date_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Build a wide daily table: one row per (well, date) with last-known rates."""
    if tests_df.empty:
        return pd.DataFrame(columns=["well_name", "date", "oil_rate", "fwat_rate", "lwat_rate"])

    frames = []
    for well, grp in tests_df.groupby("well_name", sort=False):
        grp = grp.drop_duplicates(subset="wt_date", keep="last").set_index("wt_date")
        # Reindex to every day in the target range, forward-fill from the
        # most recent prior test (lookback window is included in grp).
        daily = grp[["oil_rate", "fwat_rate", "lwat_rate"]].reindex(
            date_index.union(grp.index)
        ).ffill().reindex(date_index)
        daily = daily.dropna(how="all")
        if daily.empty:
            continue
        daily = daily.fillna(0.0)
        daily["well_name"] = well
        daily = daily.reset_index().rename(columns={"index": "date"})
        frames.append(daily)

    if not frames:
        return pd.DataFrame(columns=["well_name", "date", "oil_rate", "fwat_rate", "lwat_rate"])
    return pd.concat(frames, ignore_index=True)


def _apply_shut_in_mask(daily: pd.DataFrame, si_df: pd.DataFrame) -> pd.DataFrame:
    """Zero out any well-day where SI hours exceed the threshold."""
    if daily.empty or si_df.empty:
        return daily
    si = si_df.rename(columns={"dtdate": "date"}).copy()
    merged = daily.merge(si[["well_name", "date", "hrs"]], on=["well_name", "date"], how="left")
    mask = merged["hrs"].fillna(0.0) > SI_HOURS_THRESHOLD
    for col in ("oil_rate", "fwat_rate", "lwat_rate"):
        merged.loc[mask, col] = 0.0
    return merged.drop(columns=["hrs"])


def _aggregate_pads(daily: pd.DataFrame) -> pd.DataFrame:
    """Per-pad and combined daily oil/water/WC with recycle rule."""
    if daily.empty:
        return pd.DataFrame(columns=["date", "pad", "oil", "water", "wc"])

    daily = daily.copy()
    daily["pad"] = daily["well_name"].map(_pad_of)
    # Per-row water contribution depends on pad (H/I exclude lift water).
    lift_factor = (~daily["pad"].isin(RECYCLE_PADS)).astype(float)
    daily["water_contrib"] = daily["fwat_rate"] + lift_factor * daily["lwat_rate"]

    per_pad = (
        daily.groupby(["date", "pad"], as_index=False)
        .agg(oil=("oil_rate", "sum"), water=("water_contrib", "sum"))
    )
    combined = (
        daily.groupby("date", as_index=False)
        .agg(oil=("oil_rate", "sum"), water=("water_contrib", "sum"))
    )
    combined["pad"] = "All"
    out = pd.concat([per_pad, combined[["date", "pad", "oil", "water"]]], ignore_index=True)

    total = out["oil"] + out["water"]
    out["wc"] = (out["water"] / total).where(total > 0, 0.0)
    return out.sort_values(["pad", "date"]).reset_index(drop=True)


def fetch_pad_watercut(start_date: str, end_date: str) -> pd.DataFrame:
    """Build the daily pad-level water-cut time series.

    Args:
        start_date: 'YYYY-MM-DD' inclusive.
        end_date:   'YYYY-MM-DD' inclusive.

    Returns:
        DataFrame with columns [date, pad, oil, water, wc]. Pads include
        'G', 'H', 'I', 'J', and 'All' (combined across the four pads).
    """
    date_index = pd.date_range(start=start_date, end=end_date, freq="D", name="date")
    if len(date_index) == 0:
        return pd.DataFrame(columns=["date", "pad", "oil", "water", "wc"])

    tests_df = _fetch_tests(start_date, end_date)
    si_df = _fetch_shut_in(start_date, end_date)

    daily = _forward_fill_tests(tests_df, date_index)
    daily = _apply_shut_in_mask(daily, si_df)
    return _aggregate_pads(daily)
