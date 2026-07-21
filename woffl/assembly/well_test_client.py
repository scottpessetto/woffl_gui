"""Well Test Client

Queries Databricks for Milne Point well tests via direct SQL against
mpu.wells views, then maps result columns to the format expected by
ipr_analyzer.py (matching the FDC CSV column names).

Uses execute_query() from databricks_client.py for all Databricks auth.
"""

import re

import pandas as pd

from woffl.assembly.databricks_client import execute_query
from woffl.assembly.sql_guards import validate_iso_date, validate_well_name

WELL_HEADER_QUERY = """\
SELECT well_name
FROM mpu.wells.vw_well_header
WHERE field = 'MPU'
"""

WELL_TEST_QUERY = """\
SELECT
    vwt.well_name,
    vwt.wt_uid,
    vwt.wt_date,
    vwt.whp,
    vwt.form_oil AS oil_rate,
    vwt.form_wat AS fwat_rate,
    vwt.form_gas AS fgas_rate,
    vwt.form_wc,
    vwt.form_gor AS fgor,
    vwt.lift_wat,
    round(vbdc.bhp_cln_value, 2) AS bhp,
    vpd.tubing_prs AS pf_tubing_prs,
    vpd.inn_ann_prs AS pf_inn_ann_prs
FROM mpu.wells.vw_well_test vwt
LEFT JOIN mpu.wells.vw_bhp_daily_clean vbdc
    ON vwt.enthid = vbdc.enthid
    AND to_date(vwt.wt_date) = vbdc.tag_date
LEFT JOIN (
    -- vw_pressure_daily can carry multiple rows per well+day (repeated
    -- samples); max() collapses them and prefers an operating reading over a
    -- same-day shut-in 0. Resolved into pf_press/pf_source in Python — see
    -- woffl.assembly.pf_pressure.resolve_pf_pressure.
    SELECT
        enthid,
        sample_date,
        max(tubing_prs) AS tubing_prs,
        max(inn_ann_prs) AS inn_ann_prs
    FROM mpu.wells.vw_pressure_daily
    GROUP BY enthid, sample_date
) vpd
    ON vwt.enthid = vpd.enthid
    AND to_date(vwt.wt_date) = vpd.sample_date
WHERE vwt.well_name IN ({well_list})
    AND vwt.wt_date BETWEEN '{start_date}' AND '{end_date}'
    AND vwt.allocated = True
ORDER BY vwt.well_name, vwt.wt_date
"""


def _denormalize_well_name(name: str) -> str:
    """Reverse normalize: MPx-## -> x-0## (e.g., 'MPB-28' -> 'B-028').

    Converts GUI/jp_chars format back to Databricks vw_well_header format.
    """
    well = name
    # Strip MP prefix
    if well.startswith("MP"):
        well = well[2:]

    # Zero-pad the number portion to 3 digits
    match = re.match(r"([A-Z]+)-(\d+)", well)
    if match:
        prefix = match.group(1)
        number = match.group(2).zfill(3)
        well = f"{prefix}-{number}"

    return well


def _normalize_well_name(name: str) -> str:
    """Normalize well names to MPx-## format (e.g., 'B-028' -> 'MPB-28').

    Databricks vw_well_header returns names like 'B-028'.
    jp_chars.csv and the optimizer expect 'MPB-28'.

    This is the CANONICAL implementation — well_sort_client imports it
    rather than keeping its own copy (see R-10 / P2-7 dedup). Non-string
    input (e.g. a stray NaN from a DataFrame column) is returned unchanged
    rather than raising, matching the more defensive well_sort_client
    version this superseded.
    """
    if not isinstance(name, str):
        return name
    match = re.search(r"(\w+-\d+)", name)
    if not match:
        return name

    well = match.group(1)
    # DB names are 3-digit zero-padded (B-028, B-008). Strip ONE leading zero so
    # the GUI form is 2-digit zero-padded to match jp_chars (B-028 -> MPB-28,
    # B-008 -> MPB-08). Do NOT strip all zeros — jp_chars keys single-digit wells
    # as e.g. MPH-08, so MPB-8 would not join.
    well = re.sub(r"-(0)(?=\d+)", "-", well)
    # Add MP prefix if not already present
    if not well.startswith("MP"):
        well = "MP" + well
    return well


def get_mpu_well_names() -> list[str]:
    """Fetch all MPU well names from vw_well_header."""
    df = execute_query(WELL_HEADER_QUERY)
    if df.empty:
        return []
    return df["well_name"].tolist()


def get_pad_names(well_names: list[str]) -> list[str]:
    """Extract unique pad prefixes from well names (e.g., 'B-028' -> 'B')."""
    pads = set()
    for name in well_names:
        match = re.match(r"([A-Z]+)-", name)
        if match:
            pads.add(match.group(1))
    return sorted(pads)


def filter_wells_by_pad(well_names: list[str], pads: list[str]) -> list[str]:
    """Filter well names to only those matching the given pad prefixes."""
    pad_prefixes = tuple(f"{p}-" for p in pads)
    return [w for w in well_names if w.startswith(pad_prefixes)]


def fetch_milne_well_tests(
    start_date: str, end_date: str, well_names: list[str] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Query Databricks for Milne Point production well tests with BHP.

    Args:
        start_date: Start date string in 'YYYY-MM-DD' format.
        end_date: End date string in 'YYYY-MM-DD' format.
        well_names: List of well names to query. If None, queries all MPU wells.

    Returns:
        Tuple of (DataFrame, dropped_wells) where DataFrame has columns matching
        the FDC CSV format expected by ipr_analyzer.py, and dropped_wells is a
        list of well names that had no usable BHP or fluid rate data.
    """
    # Step 1: get well names if not provided
    if well_names is None:
        well_names = get_mpu_well_names()
    if not well_names:
        return pd.DataFrame(), []

    # Format well list for SQL IN clause. validate_well_name raises
    # UnsafeSqlValueError before any of these reach the query text.
    well_list = ", ".join(f"'{validate_well_name(w)}'" for w in well_names)

    # Step 2: query well tests with BHP
    query = WELL_TEST_QUERY.format(
        well_list=well_list,
        start_date=validate_iso_date(start_date),
        end_date=validate_iso_date(end_date),
    )
    df = execute_query(query)

    if df.empty:
        return df, []

    # Map query columns to the names expected by ipr_analyzer.py
    column_map = {
        "well_name": "well",
        "wt_date": "WtDate",
        "bhp": "BHP",
        "oil_rate": "WtOilVol",
        "fwat_rate": "WtWaterVol",
        "fgas_rate": "WtGasVol",
    }

    rename_map = {k: v for k, v in column_map.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # Compute WtTotalFluid = oil + formation water
    if "WtOilVol" in df.columns and "WtWaterVol" in df.columns:
        df["WtTotalFluid"] = df["WtOilVol"] + df["WtWaterVol"]

    # Normalize well names to MPx-## format
    if "well" in df.columns:
        df["well"] = df["well"].apply(_normalize_well_name)

    # Ensure WtDate is tz-naive datetime (Databricks returns tz-aware)
    if "WtDate" in df.columns:
        df["WtDate"] = pd.to_datetime(df["WtDate"], utc=True).dt.tz_localize(None)

    # Ensure numeric columns. wt_uid (mpu.wells.vw_well_test's well-test unique
    # ID) is the IPR-anchor pin key (see woffl.assembly.prop_hist_client /
    # ipr_wt_uid) -- kept as a nullable float here (not int) so a manual/
    # provisional test row (which has no wt_uid) can carry NaN rather than
    # forcing an int column to object dtype.
    for col in [
        "BHP",
        "WtOilVol",
        "WtWaterVol",
        "WtTotalFluid",
        "WtGasVol",
        "lift_wat",
        "whp",
        "pf_tubing_prs",
        "pf_inn_ann_prs",
        "wt_uid",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Resolve the test-day power-fluid pressure from the vw_pressure_daily
    # join (annulus for reverse circ, tubing when tubing > annulus = forward
    # circ). Adds nullable pf_press/pf_source columns; rows without a valid
    # same-day reading carry NaN/None and consumers fall back themselves.
    from woffl.assembly.pf_pressure import add_pf_columns

    df = add_pf_columns(df)

    # Track wells before dropping incomplete rows
    all_wells = set(df["well"].unique()) if "well" in df.columns else set()

    # Drop rows missing well/date/fluid rate. BHP is intentionally kept as a
    # nullable column so tests without a coincident BHP gauge measurement
    # still appear in the test history (they just can't drive Vogel IPR).
    # Downstream consumers that require BHP (compute_vogel_coefficients,
    # Model vs Actual, etc.) filter BHP themselves.
    required = ["well", "WtDate", "WtTotalFluid"]
    existing_required = [c for c in required if c in df.columns]
    df = df.dropna(subset=existing_required)

    # Identify wells that were completely lost
    remaining_wells = set(df["well"].unique()) if "well" in df.columns else set()
    dropped_wells = sorted(all_wells - remaining_wells)

    # Sort by well and date
    df = df.sort_values(by=["well", "WtDate"])

    return df, dropped_wells


def fetch_single_well_tests(well_name: str, months_back: int = 3) -> pd.DataFrame:
    """Query last N months of tests for one well.

    Args:
        well_name: MP-prefixed name (e.g., 'MPB-28').
        months_back: Number of months of history to fetch.

    Returns:
        DataFrame with ipr_analyzer-compatible columns, or empty DataFrame.
    """
    from datetime import datetime

    from dateutil.relativedelta import relativedelta

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - relativedelta(months=months_back)).strftime(
        "%Y-%m-%d"
    )

    db_name = _denormalize_well_name(well_name)
    df, _ = fetch_milne_well_tests(start_date, end_date, well_names=[db_name])
    return df
