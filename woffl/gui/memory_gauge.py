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
    # columns: timestamp, pressure — MINUTE-MEDIAN samples (downsampled at
    # parse time; see parse_xlsx). Only daily medians are consumed downstream.
    raw_df: pd.DataFrame
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    sample_count: int  # RAW (pre-downsample) points in the uploaded file
    uploaded_at: datetime
    # RAW pressure extremes, captured before downsampling — the upload
    # preview shows these so sub-minute spikes/dips (gauge pull/install
    # transients) stay visible to the engineer sanity-checking a file.
    pressure_min: float = float("nan")
    pressure_max: float = float("nan")


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

    raw_count = len(df)
    start_date = df["timestamp"].min()
    end_date = df["timestamp"].max()
    pressure_min = float(df["pressure"].min())
    pressure_max = float(df["pressure"].max())

    # Downsample to 1-minute medians: gauges sample every ~5 s, so a 90-day
    # file is ~1.5M rows (~37 MB) that used to sit in session_state for the
    # whole session — per file, per well, per user, on a shared 6 GB box.
    # Only the DAILY median is consumed downstream, so minute medians keep
    # the aggregation effectively identical at ~1/12th the memory.
    df = (
        df.set_index("timestamp")["pressure"]
        .resample("1min")
        .median()
        .dropna()
        .reset_index()
    )

    return MemoryGaugeFile(
        source_filename=source_filename,
        raw_df=df[["timestamp", "pressure"]],
        start_date=start_date,
        end_date=end_date,
        sample_count=raw_count,
        uploaded_at=datetime.now(),
        pressure_min=pressure_min,
        pressure_max=pressure_max,
    )


# ---------------------------------------------------------------------------
# Session-state CRUD
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Pending uploads — parsed but not yet applied
# ---------------------------------------------------------------------------
# The Solver's uploader parses dropped files immediately and stashes them
# here (a NON-widget session key) until the user clicks Apply. A
# file_uploader's widget state is garbage-collected whenever its view isn't
# rendered (the Single-Well page's segmented control runs only the active
# view), so files living only in the uploader were silently dropped by a
# Solver → Batch Run → Solver detour. The stash survives the detour.


# ---------------------------------------------------------------------------
# "Disregard Databricks BHP" per-well flag
# ---------------------------------------------------------------------------
# Some wells (e.g. MPB-35) have a Databricks BHP feed that is known to be
# wrong. The user can flag the well so the central read helper drops the
# Databricks BHP entirely. Works independently of gauge upload — with a
# gauge, the gauge fills in covered dates; without one, the well simply has
# no BHP data (and downstream code already handles missing BHP gracefully).


# ---------------------------------------------------------------------------
# Divergence detection: Databricks BHP vs memory-gauge daily medians
# ---------------------------------------------------------------------------
# When both sources exist for overlapping dates we compute the difference.
# Crossing either of the divergence thresholds auto-enables the disregard
# flag on Apply — the assumption is that if the user took the trouble to
# upload a memory gauge, they trust it over the Databricks feed.


# ---------------------------------------------------------------------------
# Extended well-tests fetch — covers the gauge's window which is typically
# wider than the app-wide 3-month cache (e.g., a gauge dropped in October
# that's pulled in January falls outside a May 3-month window entirely).
# ---------------------------------------------------------------------------


# max_entries: keyed on (well, start_date, end_date), so one well holds several
# entries as the gauge window moves. A miss here is a Databricks round trip
# (~3.8 s: this is the fleet query narrowed to a 1-well IN list), so with a
# ~130-well fleet 64 was far too small — walking the fleet's gauges evicted the
# earlier wells and re-paid the query for each one. 512 gives ~4 windows per
# well; each entry is one well's test rows, not a fleet frame.


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


