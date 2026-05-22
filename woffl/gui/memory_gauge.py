"""Memory gauge data ingestion for wells without permanent BHP gauges.

Some wells have well tests but no Databricks BHP feed — engineers hang
temporary memory gauges below the jet pump, pull them periodically, and
download the pressure log as an XLSX. This module:

1. Parses the XLSX into a normalized (timestamp, pressure) DataFrame.
2. Resamples to daily medians so the data matches the shape of
   ``_cached_bhp_daily`` from Databricks.
3. Stores per-well overrides in session state so downstream consumers
   (Solver Model vs Actual, IPR analyzer, JP history, sidebar
   auto-populate) can pick up the gauge BHP instead of Databricks.

Session-only persistence: gauge data is lost on browser refresh. A v2
could write to ``data/memory_gauges/<well>.parquet`` to survive refreshes.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# openpyxl PageMargins compatibility shim
# ---------------------------------------------------------------------------

_OPENPYXL_PATCHED = False


def _patch_openpyxl_pagemargins() -> None:
    """Allow openpyxl to read XLSX files whose chart sheets use legacy 'l/r/t/b' margins.

    The downhole gauge tool exports XLSXs with a Chartsheet whose PageMargins
    element uses the older single-letter shorthand (l, r, t, b). openpyxl
    expects (left, right, top, bottom) and raises TypeError otherwise. We
    monkey-patch the ``__init__`` once at first parse to accept either form.
    """
    global _OPENPYXL_PATCHED
    if _OPENPYXL_PATCHED:
        return
    from openpyxl.worksheet.page import PageMargins

    orig_init = PageMargins.__init__

    def patched(self, l=None, r=None, t=None, b=None, **kwargs):
        if l is not None:
            kwargs.setdefault("left", l)
        if r is not None:
            kwargs.setdefault("right", r)
        if t is not None:
            kwargs.setdefault("top", t)
        if b is not None:
            kwargs.setdefault("bottom", b)
        orig_init(self, **kwargs)

    PageMargins.__init__ = patched
    _OPENPYXL_PATCHED = True


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class MemoryGaugeFile:
    """A single uploaded gauge file's parsed contents.

    A well can have multiple of these (gauges get pulled and re-hung over
    months/years) — they're combined into a single ``MemoryGaugeData`` for
    downstream consumers.
    """

    source_filename: str
    raw_df: pd.DataFrame  # columns: timestamp, pressure
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    sample_count: int
    uploaded_at: datetime


@dataclass
class MemoryGaugeData:
    """Combined gauge data for a single well across one or more files.

    The combined fields (``daily_df``, ``start_date``, ``end_date``,
    ``sample_count``) are computed in ``__post_init__`` from the file list,
    so re-creating an instance with a different ``files`` list automatically
    re-aggregates. Overlapping samples (rare — typically the gauge pull/
    install instant) are deduplicated by timestamp.
    """

    well_name: str
    files: list[MemoryGaugeFile]
    daily_df: pd.DataFrame = field(init=False)
    start_date: pd.Timestamp = field(init=False)
    end_date: pd.Timestamp = field(init=False)
    sample_count: int = field(init=False)

    def __post_init__(self) -> None:
        if not self.files:
            raise ValueError("MemoryGaugeData requires at least one file.")
        all_samples = pd.concat(
            [f.raw_df for f in self.files], ignore_index=True,
        )
        all_samples = all_samples.sort_values("timestamp").drop_duplicates(
            subset=["timestamp"]
        )
        all_samples["tag_date"] = all_samples["timestamp"].dt.normalize()
        self.daily_df = (
            all_samples.groupby("tag_date", as_index=False)["pressure"]
            .median()
            .rename(columns={"pressure": "bhp"})
        )
        self.start_date = min(f.start_date for f in self.files)
        self.end_date = max(f.end_date for f in self.files)
        # Sample count is the SUM of file samples (pre-dedupe), so the user
        # sees how many raw points they uploaded. Daily-median dedup is an
        # internal detail of the aggregation.
        self.sample_count = sum(f.sample_count for f in self.files)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_xlsx(file_bytes: bytes, source_filename: str) -> MemoryGaugeFile:
    """Parse one memory-gauge XLSX into a ``MemoryGaugeFile``.

    The exporter writes a "Data" sheet with columns
    ``Line No., Date Time, Time, Pressure, Temperature, dPressure``.
    Row 1 (after the header) is a units descriptor (psi, hr, degF, etc.)
    that must be skipped. Subsequent rows are 5-second samples.

    Returns a single-file dataclass. Use :func:`add_file_to_gauge` to
    combine multiple files into one well's gauge. Raises ``ValueError``
    on any parsing failure with a user-readable message.
    """
    _patch_openpyxl_pagemargins()

    try:
        xls = pd.ExcelFile(io.BytesIO(file_bytes))
    except Exception as e:
        raise ValueError(f"Could not open XLSX: {e}") from e

    # The exporter uses a single 'Data' sheet; tolerate variant names too.
    candidate_names = ["Data", "data", "Sheet1"]
    sheet = next((s for s in candidate_names if s in xls.sheet_names), xls.sheet_names[0])

    try:
        df = pd.read_excel(xls, sheet_name=sheet)
    except Exception as e:
        raise ValueError(f"Could not read sheet {sheet!r}: {e}") from e

    if df.empty:
        raise ValueError("Memory gauge sheet is empty.")

    # Detect required columns (case-insensitive). The exporter uses
    # "Date Time" and "Pressure"; allow common variants.
    cols_lower = {c.lower(): c for c in df.columns}
    ts_col = cols_lower.get("date time") or cols_lower.get("datetime") or cols_lower.get("timestamp")
    pr_col = cols_lower.get("pressure")
    if ts_col is None or pr_col is None:
        raise ValueError(
            f"Expected 'Date Time' and 'Pressure' columns; got {list(df.columns)}."
        )

    df = df[[ts_col, pr_col]].rename(columns={ts_col: "timestamp", pr_col: "pressure"})

    # Drop the units-descriptor row (row 0). Its 'timestamp' value is a
    # format string like "M/d/yyyy HH:mm:ss" and 'pressure' is "psi".
    # Coercing both columns and dropping NaNs filters it out cleanly.
    # ``format='mixed'`` silences the "no inferable format" UserWarning when
    # row 0 is a literal format string and rows 1+ are real timestamps.
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", format="mixed")
    df["pressure"] = pd.to_numeric(df["pressure"], errors="coerce")
    df = df.dropna(subset=["timestamp", "pressure"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid (timestamp, pressure) rows after parsing.")

    # Sort and dedupe (rare but possible at gauge restart points)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])

    return MemoryGaugeFile(
        source_filename=source_filename,
        raw_df=df[["timestamp", "pressure"]].copy(),
        start_date=df["timestamp"].min(),
        end_date=df["timestamp"].max(),
        sample_count=len(df),
        uploaded_at=datetime.now(),
    )


# ---------------------------------------------------------------------------
# Session-state CRUD
# ---------------------------------------------------------------------------

_STATE_KEY = "_memory_gauge"


def get_gauge(well_name: str) -> Optional[MemoryGaugeData]:
    """Return the memory gauge for the given well, or None."""
    return st.session_state.get(_STATE_KEY, {}).get(well_name)


def store_gauge(data: MemoryGaugeData) -> None:
    """Persist gauge data for a well in session state, replacing any prior upload."""
    st.session_state.setdefault(_STATE_KEY, {})[data.well_name] = data


def clear_gauge(well_name: str) -> None:
    """Remove the gauge override for a well, if present."""
    state = st.session_state.get(_STATE_KEY)
    if state is not None:
        state.pop(well_name, None)


def has_gauge(well_name: str) -> bool:
    return get_gauge(well_name) is not None


def add_file_to_gauge(
    well_name: str, new_file: MemoryGaugeFile
) -> MemoryGaugeData:
    """Append a parsed file to the well's gauge (creating one if needed).

    Returns the new combined ``MemoryGaugeData`` and stores it under the
    well's key — overwriting any prior instance — so subsequent
    ``get_gauge`` calls see the wider window.
    """
    existing = get_gauge(well_name)
    files = (existing.files if existing is not None else []) + [new_file]
    new_gauge = MemoryGaugeData(well_name=well_name, files=files)
    store_gauge(new_gauge)
    return new_gauge


def remove_file_from_gauge(well_name: str, source_filename: str) -> bool:
    """Remove a file from a well's gauge by filename. Returns True if removed.

    If the removed file was the last one, the gauge is cleared entirely
    (no empty-files MemoryGaugeData is left in session state).
    """
    existing = get_gauge(well_name)
    if existing is None:
        return False
    remaining = [
        f for f in existing.files if f.source_filename != source_filename
    ]
    if len(remaining) == len(existing.files):
        return False
    if not remaining:
        clear_gauge(well_name)
    else:
        store_gauge(MemoryGaugeData(well_name=well_name, files=remaining))
    return True


# ---------------------------------------------------------------------------
# "Disregard Databricks BHP" per-well flag
# ---------------------------------------------------------------------------
# Some wells (e.g. MPB-35) have a Databricks BHP feed that is known to be
# wrong. The user can flag the well so the central read helper drops the
# Databricks BHP entirely. Works independently of gauge upload — with a
# gauge, the gauge fills in covered dates; without one, the well simply has
# no BHP data (and downstream code already handles missing BHP gracefully).

_DISREGARD_KEY = "_disregard_databricks_bhp"


def is_disregarding_databricks_bhp(well_name: str) -> bool:
    return bool(st.session_state.get(_DISREGARD_KEY, {}).get(well_name, False))


def set_disregard_databricks_bhp(well_name: str, value: bool) -> None:
    """Persist the disregard flag for a well. Setting False removes the entry."""
    state = st.session_state.setdefault(_DISREGARD_KEY, {})
    if value:
        state[well_name] = True
    else:
        state.pop(well_name, None)


# ---------------------------------------------------------------------------
# Divergence detection: Databricks BHP vs memory-gauge daily medians
# ---------------------------------------------------------------------------
# When both sources exist for overlapping dates we compute the difference.
# Crossing either of the divergence thresholds auto-enables the disregard
# flag on Apply — the assumption is that if the user took the trouble to
# upload a memory gauge, they trust it over the Databricks feed.

_DIVERGENCE_MEAN_ABS_PSI = 100  # avg |delta| over the overlap window
_DIVERGENCE_MEAN_PCT = 20.0     # avg |delta| as % of gauge median


def compute_databricks_vs_gauge_delta(
    databricks_bhp_df: pd.DataFrame | None, gauge: MemoryGaugeData,
) -> dict | None:
    """Compare Databricks daily BHP to gauge daily medians on overlapping dates.

    Returns a dict with diagnostics + a ``divergent`` boolean, or None when
    there's no overlap to compare (one source is empty / their windows
    don't intersect).

    The Databricks daily feed is expected in the same shape as
    ``_cached_bhp_daily``: columns ``tag_date`` (datetime, midnight) and
    ``bhp`` (psi).
    """
    if databricks_bhp_df is None or databricks_bhp_df.empty:
        return None
    if "tag_date" not in databricks_bhp_df.columns or "bhp" not in databricks_bhp_df.columns:
        return None

    db = databricks_bhp_df[["tag_date", "bhp"]].copy()
    db["tag_date"] = pd.to_datetime(db["tag_date"]).dt.normalize()
    db = db.dropna().drop_duplicates(subset=["tag_date"])

    merged = db.merge(
        gauge.daily_df.rename(columns={"bhp": "gauge_bhp"}),
        on="tag_date", how="inner",
    )
    if merged.empty:
        return None

    merged["delta"] = merged["bhp"] - merged["gauge_bhp"]
    abs_delta = merged["delta"].abs()
    gauge_mean = merged["gauge_bhp"].mean()
    pct_delta = (abs_delta / gauge_mean * 100) if gauge_mean > 0 else pd.Series([0.0])

    mean_abs = float(abs_delta.mean())
    mean_pct = float(pct_delta.mean())
    max_abs = float(abs_delta.max())

    divergent = (
        mean_abs >= _DIVERGENCE_MEAN_ABS_PSI
        or mean_pct >= _DIVERGENCE_MEAN_PCT
    )

    return {
        "n_overlap": int(len(merged)),
        "mean_abs_delta": mean_abs,
        "mean_pct_delta": mean_pct,
        "max_abs_delta": max_abs,
        "gauge_mean": float(gauge_mean),
        "databricks_mean": float(merged["bhp"].mean()),
        "divergent": divergent,
    }


def fetch_databricks_bhp_daily(
    well_name: str, start_date: pd.Timestamp, end_date: pd.Timestamp,
) -> pd.DataFrame | None:
    """Fetch Databricks daily BHP for one well over a date range.

    Returns None on failure or empty result. Used for both the divergence
    check on Apply and the JP history overlay when disregard is active.
    The underlying query is the same one ``_cached_bhp_daily`` uses on the
    JP history tab — refactoring it into a shared helper would be churn
    for one extra call site, so we just route through the existing cache.
    """
    from woffl.assembly.well_test_client import _denormalize_well_name
    from woffl.gui.tabs.jp_history_tab import _cached_bhp_daily

    db_name = _denormalize_well_name(well_name)
    start = start_date.strftime("%Y-%m-%d")
    end = end_date.strftime("%Y-%m-%d")
    try:
        df = _cached_bhp_daily(db_name, start, end)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    return df


# ---------------------------------------------------------------------------
# Extended well-tests fetch — covers the gauge's window which is typically
# wider than the app-wide 3-month cache (e.g., a gauge dropped in October
# that's pulled in January falls outside a May 3-month window entirely).
# ---------------------------------------------------------------------------

_EXT_TESTS_KEY = "_extended_well_tests"


@st.cache_data(ttl=86400, show_spinner=False)
def _cached_extended_tests_for_well(
    db_well_name: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """Fetch well tests for one well over an arbitrary date range. Cached 24h.

    Args are the Databricks-format well name (e.g. 'B-035') and ISO date
    strings. Returned columns match ``fetch_milne_well_tests`` output
    (well, WtDate, BHP, WtOilVol, WtWaterVol, etc.).
    """
    from woffl.assembly.well_test_client import fetch_milne_well_tests

    df, _ = fetch_milne_well_tests(start_date, end_date, well_names=[db_well_name])
    return df if df is not None else pd.DataFrame()


def fetch_extended_tests(
    well_name: str, start_date: pd.Timestamp
) -> pd.DataFrame | None:
    """Fetch tests for one well from ``start_date`` to today.

    Returns None on Databricks failure or empty result; callers fall back
    to the shared 3-month ``all_well_tests_df`` cache.
    """
    from woffl.assembly.well_test_client import _denormalize_well_name

    db_name = _denormalize_well_name(well_name)
    start = start_date.strftime("%Y-%m-%d")
    end = datetime.now().strftime("%Y-%m-%d")
    try:
        df = _cached_extended_tests_for_well(db_name, start, end)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    return df


def store_extended_tests(well_name: str, df: pd.DataFrame) -> None:
    """Persist the extended-window test DataFrame in session state."""
    st.session_state.setdefault(_EXT_TESTS_KEY, {})[well_name] = df


def get_extended_tests(well_name: str) -> pd.DataFrame | None:
    return st.session_state.get(_EXT_TESTS_KEY, {}).get(well_name)


def clear_extended_tests(well_name: str) -> None:
    state = st.session_state.get(_EXT_TESTS_KEY)
    if state is not None:
        state.pop(well_name, None)


# ---------------------------------------------------------------------------
# Application: override BHP on test rows / daily feed
# ---------------------------------------------------------------------------


def apply_to_well_tests(well_df: pd.DataFrame, gauge: MemoryGaugeData) -> pd.DataFrame:
    """Return a copy of ``well_df`` with the BHP column overridden from gauge data.

    For each test whose WtDate falls inside the gauge's coverage window,
    BHP is replaced with the gauge's daily median for that date. Tests
    outside the window keep their existing BHP (typically NaN for wells
    that don't have a Databricks feed).

    The well_df is assumed to be a single-well slice (no 'well' filter
    applied here). Caller is responsible for filtering first.
    """
    if "WtDate" not in well_df.columns:
        return well_df

    out = well_df.copy()
    # Ensure BHP exists as a numeric column before the merge. An all-None
    # object column triggers a pandas FutureWarning during ``combine_first``;
    # coercing upfront also handles the case where BHP came back from
    # Databricks with mixed dtypes.
    if "BHP" not in out.columns:
        out["BHP"] = pd.NA
    out["BHP"] = pd.to_numeric(out["BHP"], errors="coerce")

    # Normalize test dates to date-only for the join key (matches daily_df).
    out["_test_date"] = pd.to_datetime(out["WtDate"]).dt.normalize()
    lookup = gauge.daily_df.set_index("tag_date")["bhp"]
    matched_bhp = out["_test_date"].map(lookup)

    # Gauge wins wherever it has coverage. Outside coverage, keep existing
    # BHP — which lets users with partial Databricks coverage still see
    # their old values for dates the gauge didn't span.
    out["BHP"] = matched_bhp.combine_first(out["BHP"])
    out = out.drop(columns=["_test_date"])

    return out


def daily_bhp_from_gauge(gauge: MemoryGaugeData) -> pd.DataFrame:
    """Return the gauge's daily BHP in ``_cached_bhp_daily`` format.

    Drop-in replacement for the Databricks daily BHP fetch used by the
    JP history tab. Columns: ``tag_date`` (datetime, midnight), ``bhp``.
    """
    return gauge.daily_df.copy()


def coverage_summary(well_df: pd.DataFrame, gauge: MemoryGaugeData) -> dict:
    """Return diagnostics for the UI status banner.

    Keys: ``tests_total``, ``tests_in_window`` (count of WtDates inside the
    gauge's [start, end] window), ``tests_matched`` (count whose date has
    a same-day median in the daily_df — should equal tests_in_window in
    practice, but kept distinct for defensive UI messaging).
    """
    if well_df is None or well_df.empty:
        return {"tests_total": 0, "tests_in_window": 0, "tests_matched": 0}

    dates = pd.to_datetime(well_df["WtDate"]).dt.normalize()
    in_window = dates.between(
        gauge.start_date.normalize(), gauge.end_date.normalize()
    ).sum()
    matched = dates.isin(gauge.daily_df["tag_date"]).sum()
    return {
        "tests_total": int(len(well_df)),
        "tests_in_window": int(in_window),
        "tests_matched": int(matched),
    }
