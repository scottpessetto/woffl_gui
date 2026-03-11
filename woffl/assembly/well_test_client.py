"""Well Test Client

Queries Databricks for Milne Point well tests via direct SQL against
mpu.wells views, then maps result columns to the format expected by
ipr_analyzer.py (matching the FDC CSV column names).

Uses execute_query() from databricks_client.py for all Databricks auth.
"""

import re

import pandas as pd

from woffl.assembly.databricks_client import execute_query

WELL_HEADER_QUERY = """\
SELECT well_name
FROM mpu.wells.vw_well_header
WHERE field = 'MPU'
"""

WELL_TEST_QUERY = """\
SELECT
    vwt.well_name,
    vwt.wt_date,
    vwt.whp,
    vwt.form_oil AS oil_rate,
    vwt.form_wat AS fwat_rate,
    vwt.form_gas AS fgas_rate,
    vwt.form_wc,
    vwt.form_gor AS fgor,
    vwt.lift_wat,
    round(vbdc.bhp_cln_value, 2) AS bhp
FROM mpu.wells.vw_well_test vwt
LEFT JOIN mpu.wells.vw_bhp_daily_clean vbdc
    ON vwt.enthid = vbdc.enthid
    AND to_date(vwt.wt_date) = vbdc.tag_date
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
    """
    match = re.search(r"(\w+-\d+)", name)
    if not match:
        return name

    well = match.group(1)
    # Remove leading zeros in the number portion
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

    # Format well list for SQL IN clause
    well_list = ", ".join(f"'{w}'" for w in well_names)

    # Step 2: query well tests with BHP
    query = WELL_TEST_QUERY.format(
        well_list=well_list,
        start_date=start_date,
        end_date=end_date,
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

    # Ensure numeric columns
    for col in ["BHP", "WtOilVol", "WtWaterVol", "WtTotalFluid", "WtGasVol", "lift_wat", "whp"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Track wells before dropping incomplete rows
    all_wells = set(df["well"].unique()) if "well" in df.columns else set()

    # Drop rows without BHP or fluid rate (required for IPR analysis)
    required = ["well", "WtDate", "BHP", "WtTotalFluid"]
    existing_required = [c for c in required if c in df.columns]
    df = df.dropna(subset=existing_required)

    # Identify wells that were completely lost
    remaining_wells = set(df["well"].unique()) if "well" in df.columns else set()
    dropped_wells = sorted(all_wells - remaining_wells)

    # Sort by well and date
    df = df.sort_values(by=["well", "WtDate"])

    return df, dropped_wells


def fetch_single_well_tests(
    well_name: str, months_back: int = 3
) -> pd.DataFrame:
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
    start_date = (datetime.now() - relativedelta(months=months_back)).strftime("%Y-%m-%d")

    db_name = _denormalize_well_name(well_name)
    df, _ = fetch_milne_well_tests(start_date, end_date, well_names=[db_name])
    return df
