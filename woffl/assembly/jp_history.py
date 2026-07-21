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


def _pump_dict_from_row(latest: pd.Series) -> dict:
    """Package a JP-history row into the standard pump dict."""
    nozzle = latest.get("Nozzle Number")
    throat = latest.get("Throat Ratio")
    tubing = latest.get("Tubing Diameter")
    date_set = latest.get("Date Set")

    # Nozzle number must be coercible to int — JetPump expects "1"-"20". A
    # non-numeric value (e.g. 'G') means the row isn't really a jet-pump
    # install (legacy ESP/wireline data sometimes sits in this column);
    # return None so callers skip the well.
    nozzle_str = None
    if pd.notna(nozzle):
        try:
            nozzle_str = str(int(nozzle))
        except (TypeError, ValueError):
            nozzle_str = None

    tubing_val = None
    if pd.notna(tubing):
        try:
            tubing_val = float(tubing)
        except (TypeError, ValueError):
            tubing_val = None

    # Optional enrichment columns (pump_identity.enrich_jp_history) — passed
    # through additively when present so GUI consumers can show direction /
    # brand provenance. Plain tracker or xlsx frames simply yield None here.
    def _opt_str(key: str):
        v = latest.get(key)
        return v if isinstance(v, str) and v else None

    return {
        "nozzle_no": nozzle_str,
        "throat_ratio": str(throat).strip() if pd.notna(throat) else None,
        "tubing_od": tubing_val,
        "date_set": date_set,
        "circ_direction": _opt_str("Circ Direction"),
        "manufacturer": _opt_str("Manufacturer"),
        "raw_pump": _opt_str("Raw Pump"),
    }


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
    return _pump_dict_from_row(latest)


def get_pump_at_date(jp_hist: pd.DataFrame, well_name: str, date) -> dict | None:
    """Return the pump installed on a well at a given date, or None.

    The pump at a date is the install with the LATEST Date Set on or before
    that date. Date Pulled is deliberately ignored: jet-pump changeouts are
    same-day slickline runs (pull + set in one visit per the AKIMS well
    events), so pumps are contiguous set-to-set in reality — but the
    tracker's Date Pulled column lags/shifts by days-to-weeks, and honoring
    it created phantom "no pump in hole" windows that wrongly returned None
    for tests taken in them.

    Used to pair historical well tests with the pump that was actually in
    the hole at test time — calibrating an old test against today's pump
    geometry makes the friction coefficients absorb the nozzle/throat area
    difference.

    Args:
        jp_hist: DataFrame from parse_jp_history().
        well_name: Well identifier (e.g., "MPB-37").
        date: Date of interest (anything pd.Timestamp accepts).

    Returns:
        Dict with nozzle_no, throat_ratio, tubing_od, date_set, or None when
        the date precedes the well's first recorded install.
    """
    if jp_hist is None or date is None or pd.isna(date):
        return None

    well_df = jp_hist[jp_hist["Well Name"] == well_name].dropna(subset=["Date Set"])
    if well_df.empty:
        return None

    when = pd.Timestamp(date)
    candidates = well_df[well_df["Date Set"] <= when]
    if candidates.empty:
        return None

    latest = candidates.sort_values("Date Set", ascending=False).iloc[0]
    return _pump_dict_from_row(latest)


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
