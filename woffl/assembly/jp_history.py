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


def order_installs(df: pd.DataFrame, ascending: bool = False) -> pd.DataFrame:
    """Installs ordered by ``Date Set`` with a DETERMINISTIC same-day tie-break.

    A JPCO is a same-day slickline pull + set, so two rows on one ``Date Set``
    is the normal record of a changeout. Ordering by ``Date Set`` alone with
    pandas' default unstable sort left the winner to chance, so the current
    pump, the pump-at-test-date and the era sequence could flip between
    fetches (review 2026-09-01, DATA-2). Tie-break: the row still in the
    hole (``Date Pulled`` NaT) is the later install; among rows that both
    carry a pull date the later pull is the later install; then the input
    order, kept by a stable sort.

    Args:
        df: JP-history rows (any subset) with ``Date Set`` and optionally
            ``Date Pulled``.
        ascending: True for oldest-first (era building), False for
            latest-first (current pump).

    Returns:
        The same rows, reordered; index preserved.
    """
    if df is None or df.empty:
        return df
    work = df.copy()
    if "Date Pulled" in work.columns:
        pulled = pd.to_datetime(work["Date Pulled"], errors="coerce")
    else:
        pulled = pd.Series(pd.NaT, index=work.index)
    # NaT (still in hole) must sort as the LATEST pull regardless of direction.
    work["_pull_key"] = pulled.fillna(pd.Timestamp.max)
    work["_row_key"] = range(len(work))
    work = work.sort_values(
        ["Date Set", "_pull_key", "_row_key"],
        ascending=[ascending, ascending, ascending],
        kind="stable",
    )
    return work.drop(columns=["_pull_key", "_row_key"])


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

    # Latest Date Set = current pump (same-day ties resolved by order_installs)
    latest = order_installs(well_df, ascending=False).iloc[0]
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

    latest = order_installs(candidates, ascending=False).iloc[0]
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

    # Keep the latest install per well; same-day ties via order_installs
    # (idxmax on Date Set alone picked an arbitrary row on a changeout day).
    ordered = order_installs(df, ascending=False)
    return ordered.drop_duplicates(subset=["Well Name"], keep="first").reset_index(drop=True)


def pump_ages(jp_hist: pd.DataFrame, today=None) -> pd.DataFrame:
    """Current-pump age per well — who's overdue for a jet pump change.

    Tenure follows the house rule (see AGENTS.md / the JPCO gotcha): the
    current pump has been in hole since its **Date Set**, and the tracker's
    ``Date Pulled`` is never consulted. Age = ``today − latest Date Set``.

    Returns one row per well, oldest pump first:
    ``Well Name · Nozzle Number · Throat Ratio · Date Set · Days In Hole ·
    Installs`` (installs = rows with a valid Date Set on record — frequent
    changers vs never-touched wells). Empty frame in → empty frame out.
    """
    if jp_hist is None or jp_hist.empty:
        return pd.DataFrame()
    df = jp_hist.dropna(subset=["Date Set"]).copy()
    if df.empty:
        return pd.DataFrame()

    today = pd.Timestamp(today) if today is not None else pd.Timestamp.today()
    today = today.normalize()

    current = df.loc[df.groupby("Well Name")["Date Set"].idxmax()].copy()
    counts = df.groupby("Well Name")["Date Set"].size()
    current["Installs"] = current["Well Name"].map(counts).astype(int)
    current["Days In Hole"] = (
        (today - pd.to_datetime(current["Date Set"]).dt.normalize()).dt.days
    ).astype(int)

    keep = [
        c
        for c in (
            "Well Name",
            "Nozzle Number",
            "Throat Ratio",
            "Date Set",
            "Days In Hole",
            "Installs",
        )
        if c in current.columns
    ]
    return (
        current[keep]
        .sort_values("Days In Hole", ascending=False)
        .reset_index(drop=True)
    )


def filter_recently_online(
    ages: pd.DataFrame, last_test: dict, days: int, today=None
) -> pd.DataFrame:
    """Keep aging-pump rows whose well has produced recently.

    "Online" evidence = the well's latest well test (``last_test``:
    {well: timestamp}) falling within ``days`` of ``today``. EVERY test
    counts, allocated or info-only: allocation is a monthly accounting pass,
    so a producing well routinely has no allocated test for ~30 days and an
    allocated-only proxy declared it offline (MPS-05, 2026-08-18: tested
    2026-08-16, last allocated 2026-07-17).

    A ``Last Test`` column is added to every surviving row. Wells with NO
    known test are dropped (no evidence of production). Empty in → empty out;
    an empty ``last_test`` map returns the frame unchanged (source
    unavailable → don't silently drop everything).
    """
    if ages is None or ages.empty:
        return pd.DataFrame() if ages is None else ages
    if not last_test:
        return ages
    today = (
        pd.Timestamp(today) if today is not None else pd.Timestamp.today()
    ).normalize()
    out = ages.copy()
    # vw_well_test dates arrive TZ-AWARE (Etc/UTC) while `today` is naive —
    # comparing them raises TypeError. Coerce everything through UTC and strip
    # the tz so aware, naive and missing values all compare cleanly.
    out["Last Test"] = pd.to_datetime(
        out["Well Name"].map(last_test), utc=True, errors="coerce"
    ).dt.tz_localize(None)
    cutoff = today - pd.Timedelta(days=int(days))
    out = out[out["Last Test"].notna() & (out["Last Test"] >= cutoff)]
    return out.reset_index(drop=True)
