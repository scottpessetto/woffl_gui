"""Operator grab-sample OIW - the CFP sample log, rolled up to field days.

The operators keep a spreadsheet of manual oil-in-water grab samples off the
CFP water system: one row per sample, a date, a clock time, a sample point,
a lab ppm and the sampler's name. This module turns one uploaded copy of that
workbook into a daily sampled oil rate so it can be drawn beside the
calculated first-stage loss band (services/tools/sep_oil_loss.py).

**The math.** A grab sample is a concentration, so the rate it implies is

    oil_bopd = ppm * water_rate_bpd / 1e6

``water_rate_bpd`` is a CALLER input, echoed back in the response, because
the workbook's own ``(BOPD)`` column is a hardcoded 95,000 BWPD basis that
nobody maintains - it is blank on most recent rows and wrong whenever the
plant is not at 95,000. Nothing here reads that column.

**The roll-up.** Samples are irregular manual grabs, a handful a day at best,
so the day's rate is the plain unweighted mean of its samples' rates. Time
weighting a set of grabs would invent a duty cycle the log does not carry.
Days are Alaska calendar days, the same day definition the loss tool uses;
the workbook's Date column already IS the operator's local calendar date, so
no zone conversion is applied to it.

**The caveat, which is the whole reason to be careful with this number.**
The only location still being sampled, ``P-5417C``, sits DOWNSTREAM of the
deoilers (V-5419 / V-5421 / V-5422 / V-5425). The calculated band is the
first-stage water leg UPSTREAM of them, off ``MPU_AI_5317``. A baseline of
1,000-2,500 ppm at 71,000 BPD is about 106 BOPD, far below the calculated
lower bound - and that difference is mostly deoiler RECOVERY, not
measurement error. ``V-5317`` is the one location that samples the same
stream the calculated band describes. Every response carries that caveat in
``notes`` and the page renders it, so the two series are never presented as
directly comparable.

Read-only and stateless: the upload is parsed in memory and returned. Nothing
is written to Databricks or to disk.
"""

from __future__ import annotations

import io
import logging
from typing import Any, Optional

import pandas as pd

log = logging.getLogger("woffl.web.tools.oiw_samples")

# The live sheet. Same layout on the V-5317 sheet, which stopped in 2023 and
# is kept only because it is the one upstream sample point.
DEFAULT_SHEET = "OIW Daily"
DEFAULT_LOCATION = "P-5417C"
# The one sample point on the SAME stream as the calculated band.
UPSTREAM_LOCATION = "V-5317"
DEFAULT_WATER_RATE_BPD = 95_000.0
WATER_RATE_MIN, WATER_RATE_MAX = 1_000.0, 300_000.0

# Both sheets of interest carry two header rows; row 1 holds the names.
HEADER_ROW = 1

# Column names, normalized (stripped, lowercased). The live sheet writes
# "Date " with a trailing space and the V-5317 sheet spells its BOPD column
# out as a sentence, so every lookup goes through the normalized map.
DATE_KEYS = ("date", "date ", "sample date")
LOCATION_KEYS = ("location", "sample point")
PPM_KEYS = ("ppm", "oiw ppm")

# Junk guards. The log has blank rows, text typed into numeric cells and at
# least one 2107 date, and none of it may raise or reach the client.
EARLIEST_DATE = pd.Timestamp("2000-01-01")
# 1e6 ppm is pure oil; anything at or above it is a typo, not a sample.
PPM_CEIL = 1_000_000.0

FIELD_TZ = "America/Anchorage"


# ---------------------------------------------------------------------------
# Workbook
# ---------------------------------------------------------------------------


def _resolve_sheet(names: list[str], requested: str) -> str:
    """The workbook's own spelling of a requested sheet.

    Args:
        names (list): Sheet names (str) as the workbook spells them.
        requested (str): Sheet asked for, matched case-insensitively.

    Returns:
        sheet (str): The workbook's spelling.

    Raises:
        ValueError: No sheet matches, with the available names listed.
    """
    if requested in names:
        return requested
    wanted = requested.strip().lower()
    for name in names:
        if name.strip().lower() == wanted:
            return name
    raise ValueError(f"no sheet named {requested!r}; this workbook has {names}")


def _column(frame: pd.DataFrame, keys: tuple[str, ...]) -> Optional[Any]:
    """First column whose normalized header matches one of ``keys``.

    Args:
        frame (DataFrame): The sheet as read.
        keys (tuple): Normalized header names (str) to accept, in order of
            preference.

    Returns:
        column (Any | None): The workbook's own column label, or None.
    """
    normalized: dict[str, Any] = {}
    for col in frame.columns:
        normalized.setdefault(str(col).strip().lower(), col)
    for key in keys:
        if key in normalized:
            return normalized[key]
    return None


def _clean(raw: pd.DataFrame, today: pd.Timestamp) -> tuple[pd.DataFrame, int]:
    """Date / location / ppm rows that survive every junk filter.

    A row is kept only when its date parses into a plausible window AND its
    ppm is a finite positive number below pure oil. Everything else - blank
    spacer rows, text in a numeric cell, the stray 2107 date - is counted and
    dropped.

    Args:
        raw (DataFrame): The sheet as read, header row already applied.
        today (pd.Timestamp): Field-local today, naive; the newest date a
            sample may carry.

    Returns:
        rows (tuple): ``(frame, dropped)`` - a DataFrame with columns ``day``
            (Timestamp, midnight), ``location`` (str) and ``ppm`` (float,
            ppm), plus the number of rows (int) dropped as unparseable.

    Raises:
        ValueError: The sheet has no Location or PPM column, so it is not a
            grab-sample sheet in this layout.
    """
    date_col = _column(raw, DATE_KEYS)
    loc_col = _column(raw, LOCATION_KEYS)
    ppm_col = _column(raw, PPM_KEYS)
    missing = [
        label
        for label, col in (("Date", date_col), ("Location", loc_col), ("PPM", ppm_col))
        if col is None
    ]
    if missing:
        raise ValueError(
            f"sheet is not a grab-sample log in this layout: no {', '.join(missing)} "
            f"column among {[str(c) for c in raw.columns[:12]]}"
        )

    out = pd.DataFrame(
        {
            "day": pd.to_datetime(raw[date_col], errors="coerce"),
            "location": raw[loc_col].astype(str).str.strip(),
            "ppm": pd.to_numeric(raw[ppm_col], errors="coerce"),
        }
    )
    total = len(out)
    out["day"] = out["day"].dt.normalize()
    keep = (
        out["day"].notna()
        & (out["day"] >= EARLIEST_DATE)
        & (out["day"] <= today)
        & out["ppm"].notna()
        & (out["ppm"] > 0.0)
        & (out["ppm"] < PPM_CEIL)
        & (out["location"] != "")
        & (out["location"].str.lower() != "nan")
    )
    kept = out.loc[keep].reset_index(drop=True)
    return kept, int(total - len(kept))


# ---------------------------------------------------------------------------
# Roll-up
# ---------------------------------------------------------------------------


def _daily(frame: pd.DataFrame, water_rate_bpd: float, location: str) -> list[dict[str, Any]]:
    """One row per Alaska calendar day the samples touch.

    ``bopd_mean`` and ``bbl`` are the same number: a daily rate held for one
    day is that many barrels. Both are the time-UNWEIGHTED mean of the day's
    per-sample rates, because a handful of manual grabs carries no duty cycle
    to weight with.

    Args:
        frame (DataFrame): Cleaned rows for one location, columns ``day``,
            ``location``, ``ppm`` (ppm).
        water_rate_bpd (float): Water rate the concentrations act on, BPD.
        location (str): Sample point, echoed onto every row.

    Returns:
        rows (list): Chronological dicts with ``date`` (str, YYYY-MM-DD),
            ``samples`` (int), ``ppm_mean`` / ``ppm_min`` / ``ppm_max``
            (float, ppm), ``bopd_mean`` (float, BOPD), ``bbl`` (float, bbl)
            and ``location`` (str).
    """
    if frame.empty:
        return []

    rates = frame["ppm"] * water_rate_bpd / 1.0e6
    grouped = (
        frame.assign(bopd=rates)
        .groupby("day", sort=True)
        .agg(
            samples=("ppm", "size"),
            ppm_mean=("ppm", "mean"),
            ppm_min=("ppm", "min"),
            ppm_max=("ppm", "max"),
            bopd_mean=("bopd", "mean"),
        )
    )
    return [
        {
            "date": day.strftime("%Y-%m-%d"),
            "samples": int(row.samples),
            "ppm_mean": round(float(row.ppm_mean), 1),
            "ppm_min": round(float(row.ppm_min), 1),
            "ppm_max": round(float(row.ppm_max), 1),
            "bopd_mean": round(float(row.bopd_mean), 2),
            "bbl": round(float(row.bopd_mean), 2),
            "location": location,
        }
        for day, row in zip(grouped.index, grouped.itertuples(index=False))
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def oiw_samples(
    blob: bytes,
    filename: str,
    location: str = DEFAULT_LOCATION,
    water_rate_bpd: float = DEFAULT_WATER_RATE_BPD,
    sheet: str = DEFAULT_SHEET,
) -> dict[str, Any]:
    """Parse one grab-sample workbook into a daily sampled oil rate.

    Every failure mode is a ``ValueError`` with a message an engineer can
    act on - a wrong workbook, a missing sheet, a sheet in another layout, a
    sheet with nothing parseable in it. A location that simply has no samples
    is NOT an error: the response comes back with an empty ``daily``, the
    locations that do have samples, and a note saying so.

    Args:
        blob (bytes): The uploaded .xlsx bytes.
        filename (str): Name to echo back, for the UI.
        location (str): Sample point to roll up, matched case-insensitively.
        water_rate_bpd (float): Water rate the concentrations act on, BPD.
            The workbook's own ``(BOPD)`` column is ignored.
        sheet (str): Worksheet to read, matched case-insensitively.

    Returns:
        payload (dict): ``filename`` (str), ``sheet`` (str), ``location``
            (str), ``water_rate_bpd`` (float, BPD), ``locations_available``
            (list of str), ``first_date`` / ``last_date`` (str | None,
            YYYY-MM-DD), ``sample_count`` (int, samples at ``location``),
            ``daily`` (list, see :func:`_daily`) and ``notes`` (list of str).

    Raises:
        ValueError: The bytes are not a readable workbook, the sheet does not
            exist, the sheet is in another layout, or nothing in it parses.
    """
    if not (WATER_RATE_MIN <= water_rate_bpd <= WATER_RATE_MAX):
        raise ValueError(
            f"water rate must be {WATER_RATE_MIN:,.0f} - {WATER_RATE_MAX:,.0f} BPD"
        )

    try:
        book = pd.ExcelFile(io.BytesIO(blob))
    except Exception as exc:  # openpyxl raises a zoo of its own types
        raise ValueError(f"could not open {filename} as an XLSX workbook: {exc}") from exc

    resolved = _resolve_sheet(list(book.sheet_names), sheet)
    try:
        raw = pd.read_excel(book, sheet_name=resolved, header=HEADER_ROW)
    except Exception as exc:
        raise ValueError(f"could not read sheet {resolved!r}: {exc}") from exc
    if raw.empty:
        raise ValueError(f"sheet {resolved!r} has no rows below its header")

    today = pd.Timestamp.now(tz=FIELD_TZ).tz_localize(None).normalize()
    try:
        frame, dropped = _clean(raw, today)
    except ValueError:
        raise
    except Exception as exc:
        # A sheet shaped like nothing seen here is bad input, not a bug worth
        # a 500: name it and let the engineer pick another sheet.
        log.exception("oiw samples: unreadable sheet %r in %s", resolved, filename)
        raise ValueError(f"could not read sheet {resolved!r} as a grab-sample log: {exc}") from exc
    if frame.empty:
        raise ValueError(
            f"sheet {resolved!r} has no rows with both a parseable date and a "
            f"positive ppm ({dropped} rows dropped)"
        )

    # Case is not a sample point: the log carries "P-5417C" and "p-5417C" for
    # the same tap, so locations are grouped case-insensitively and each group
    # is offered under its most-used spelling. Anything beyond case - "P5417C",
    # "P-5417-C" - is left alone, because merging those would be a guess.
    spellings = frame["location"].value_counts()
    canonical: dict[str, str] = {}
    for name in spellings.index:
        canonical.setdefault(str(name).lower(), str(name))
    locations = sorted(canonical.values())

    wanted = location.strip().lower()
    picked = canonical.get(wanted)
    at_location = frame.loc[frame["location"].str.lower() == wanted]

    resolved_location = picked if picked is not None else location.strip()
    daily = _daily(at_location, water_rate_bpd, resolved_location)

    notes = [
        f"Sampled rate is ppm x {water_rate_bpd:,.0f} BPD / 1e6, one unweighted mean "
        "per Alaska calendar day. The workbook's own (BOPD) column assumes a fixed "
        "95,000 BWPD and is not used."
    ]
    if picked is None:
        notes.append(
            f"No samples at {location.strip()!r} on sheet {resolved!r}. "
            f"Sampled locations there: {', '.join(locations)}."
        )
    if dropped:
        notes.append(
            f"{dropped} of {len(raw)} rows on sheet {resolved!r} were dropped as "
            "unparseable (blank, non-numeric ppm, or an out-of-range date)."
        )
    if resolved_location.upper() != UPSTREAM_LOCATION:
        notes.append(
            f"{resolved_location} is sampled DOWNSTREAM of the deoilers, while the "
            "calculated band is the first-stage water leg upstream of them. The gap "
            f"between the two is deoiler recovery, not error. Only {UPSTREAM_LOCATION} "
            "samples the same stream as the calculated band."
        )

    log.info(
        "oiw samples: %s sheet=%r location=%r rows=%d dropped=%d days=%d",
        filename,
        resolved,
        resolved_location,
        len(at_location),
        dropped,
        len(daily),
    )
    return {
        "filename": filename,
        "sheet": resolved,
        "location": resolved_location,
        "water_rate_bpd": float(water_rate_bpd),
        "locations_available": locations,
        "first_date": daily[0]["date"] if daily else None,
        "last_date": daily[-1]["date"] if daily else None,
        "sample_count": int(len(at_location)),
        "daily": daily,
        "notes": notes,
    }
