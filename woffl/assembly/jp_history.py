"""JP History Parser

Reads jet pump installation history from Excel files and identifies
the current (most recently installed) pump for each well.
"""

import pandas as pd


def parse_jp_history(file) -> pd.DataFrame:
    """Read JP history xlsx from a file uploader or path.

    Args:
        file: Streamlit UploadedFile or file path string.

    Returns:
        Cleaned DataFrame with standardized column names.
    """
    df = pd.read_excel(file)

    # Standardize column names: strip whitespace, collapse internal spaces
    df.columns = df.columns.str.strip().str.replace(r"\s+", " ", regex=True)

    # Ensure Date Set is datetime
    if "Date Set" in df.columns:
        df["Date Set"] = pd.to_datetime(df["Date Set"], errors="coerce")

    if "Date Pulled" in df.columns:
        df["Date Pulled"] = pd.to_datetime(df["Date Pulled"], errors="coerce")

    # Strip well name whitespace
    if "Well Name" in df.columns:
        df["Well Name"] = df["Well Name"].astype(str).str.strip()

    return df


def get_current_pump(jp_hist: pd.DataFrame, well_name: str) -> dict | None:
    """Return the current pump for a well (latest Date Set).

    Args:
        jp_hist: DataFrame from parse_jp_history().
        well_name: Well identifier (e.g., "MPB-37").

    Returns:
        Dict with nozzle_no, throat_ratio, tubing_od, date_set,
        or None if well not found.
    """
    well_df = jp_hist[jp_hist["Well Name"] == well_name].copy()
    if well_df.empty:
        return None

    # Drop rows without a Date Set
    well_df = well_df.dropna(subset=["Date Set"])
    if well_df.empty:
        return None

    # Latest Date Set = current pump
    latest = well_df.sort_values("Date Set", ascending=False).iloc[0]

    nozzle = latest.get("Nozzle Number")
    throat = latest.get("Throat Ratio")
    tubing = latest.get("Tubing Diameter")
    date_set = latest.get("Date Set")

    return {
        "nozzle_no": str(int(nozzle)) if pd.notna(nozzle) else None,
        "throat_ratio": str(throat).strip() if pd.notna(throat) else None,
        "tubing_od": float(tubing) if pd.notna(tubing) else None,
        "date_set": date_set,
    }


def get_all_current_pumps(jp_hist: pd.DataFrame) -> pd.DataFrame:
    """Return the current pump for every well (latest Date Set each).

    Args:
        jp_hist: DataFrame from parse_jp_history().

    Returns:
        DataFrame with one row per well showing the most recent pump.
    """
    df = jp_hist.dropna(subset=["Date Set"]).copy()
    if df.empty:
        return pd.DataFrame()

    # Keep the row with the latest Date Set per well
    idx = df.groupby("Well Name")["Date Set"].idxmax()
    return df.loc[idx].reset_index(drop=True)
