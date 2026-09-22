"""Stateless CSV/XLSX OIW log parsing and upstream sample comparisons.

Preserve sample time, point, lab method, sampler and notes. Units must be
declared: ppmv is volume/volume; mg/L requires oil density. Unknown units
retain concentration but withhold oil fractions and rates. Point grabs do
not establish daily volumes. Daily concentration/rate means describe only
the collected samples; the entered flow is an explicit hypothetical basis.

V-5317 is the documented first-stage stream; P-5417C is downstream of the
deoilers. Other sample locations remain unverified. Comparison uses only
same-stream samples with confirmed units and recent backward historian
pairs. No calibration or process settings are changed, and uploads are not
stored server-side. See docs/separator_sample_workflow_2026-09-14.md.
"""

from __future__ import annotations

import io
import logging
from typing import Any, Optional

import pandas as pd

from server.services.tools.oiw_validation import oil_fraction, sample_timestamp, compare_samples

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
PPM_KEYS = ("ppm", "oiw ppm", "concentration", "result", "mg/l")

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
    concentration is a finite nonnegative number below pure oil. Everything else - blank
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
            "day": pd.to_datetime(raw[date_col], format="mixed", dayfirst=False, errors="coerce"),
            "location": raw[loc_col].astype(str).str.strip(),
            "ppm": pd.to_numeric(raw[ppm_col], errors="coerce"),
        }
    )
    total = len(out)
    out["day"] = out["day"].dt.normalize()
    time_col = _column(raw, ("time", "sample time"))
    out["timestamp"] = [
        sample_timestamp(day, raw.loc[idx, time_col]) if time_col is not None and pd.notna(day) else None
        for idx, day in out["day"].items()
    ]
    out["source_row"] = out.index + HEADER_ROW + 2
    for name, keys in (("sampler", ("sampler", "operator")), ("method", ("method", "lab method")), ("notes", ("notes", "comment"))):
        column = _column(raw, keys)
        out[name] = raw[column].fillna("").astype(str) if column is not None else ""
    keep = (
        out["day"].notna()
        & (out["day"] >= EARLIEST_DATE)
        & (out["day"] <= today)
        & out["ppm"].notna()
        & (out["ppm"] >= 0.0)
        & (out["ppm"] < PPM_CEIL)
        & (out["location"] != "")
        & (out["location"].str.lower() != "nan")
    )
    kept = out.loc[keep].reset_index(drop=True)
    return kept, int(total - len(kept))


# ---------------------------------------------------------------------------
# Roll-up
# ---------------------------------------------------------------------------


def _daily(frame: pd.DataFrame, water_rate_bpd: float, location: str, units: str = "unknown", oil_density: float | None = None, rate_basis: str = "liquid") -> list[dict[str, Any]]:
    """Unweighted sample-day means; never extrapolate grabs into daily barrels.

    Fraction is oil volume / total liquid volume. For a liquid flow use Q*f;
    for a water-only flow use Qw*f/(1-f). Units must be confirmed first.
    """
    if frame.empty:
        return []

    fractions = frame["ppm"].map(lambda v: oil_fraction(v, units, oil_density)).astype(float)
    rates = water_rate_bpd * fractions if rate_basis == "liquid" else water_rate_bpd * fractions / (1.0 - fractions)
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
            "bopd_mean": None if pd.isna(row.bopd_mean) else round(float(row.bopd_mean), 2),
            "bbl": None,  # Point grabs do not establish a daily volume.
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
    units: str = "unknown",
    oil_density_kgm3: float | None = None,
    rate_basis: str = "liquid",
    days: int | None = None,
    lag_minutes: float = 0.0,
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
    oil_fraction(0, units, oil_density_kgm3)
    if rate_basis not in ("liquid", "water"):
        raise ValueError("flow basis must be liquid or water")
    if days is not None and not 1 <= days <= 90:
        raise ValueError("comparison days must be 1-90")
    if not 0 <= lag_minutes <= 120:
        raise ValueError("sample transport delay must be 0-120 minutes")

    if filename.lower().endswith(".csv"):
        resolved = "CSV"
        first_data_row = 2
        try:
            raw = pd.read_csv(io.BytesIO(blob), encoding="utf-8-sig")
        except Exception as exc:
            raise ValueError("could not read sample CSV; use the downloadable template") from exc
    else:
        try:
            book = pd.ExcelFile(io.BytesIO(blob))
        except Exception as exc:
            raise ValueError(f"could not open {filename} as an XLSX workbook") from exc
        with book:
            resolved = _resolve_sheet(list(book.sheet_names), sheet)
            raw = pd.read_excel(book, sheet_name=resolved, header=HEADER_ROW)
            first_data_row = 3
            # New logs can use an ordinary first-row header, as in the CSV.
            if _column(raw, DATE_KEYS) is None:
                raw = pd.read_excel(book, sheet_name=resolved, header=0)
                first_data_row = 2
    if raw.empty:
        raise ValueError(f"sheet {resolved!r} has no rows below its header")

    today = pd.Timestamp.now(tz=FIELD_TZ).tz_localize(None).normalize()
    try:
        frame, dropped = _clean(raw, today)
        frame["source_row"] += first_data_row - 3
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
            f"nonnegative concentration ({dropped} rows dropped)"
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
    daily = _daily(at_location, water_rate_bpd, resolved_location, units, oil_density_kgm3, rate_basis)

    notes = [
        f"Sample rates use the entered {water_rate_bpd:,.0f} BPD {rate_basis} basis. "
        "Daily means describe only the collected grabs; daily barrels are withheld. "
        "The workbook's own (BOPD) column is not used."
    ]
    if units == "unknown":
        notes.append("Confirm whether the lab reports ppm by volume or mg/L before calculating rates. Mass ppm (mg/kg) is a different basis and needs a lab conversion.")
    if units == "mg/L":
        notes.append(f"Oil volume fraction = mg/L / ({oil_density_kgm3:g} kg/m3 x 1,000). Confirm density at the sample basis and that the lab method represents the oil measured by Red Eye.")
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
    if resolved_location.upper() == DEFAULT_LOCATION:
        notes.append(
            f"{resolved_location} is sampled DOWNSTREAM of the deoilers, while the "
            "calculated scenarios are upstream. Differences can include recovery, "
            f"timing and measurement error. Only {UPSTREAM_LOCATION} "
            "samples the same stream as the calculated band."
        )
    elif resolved_location.upper() != UPSTREAM_LOCATION:
        notes.append("This sample point's process location is unverified; it cannot validate Red Eye.")

    records = []
    for row in at_location.itertuples():
        fraction = oil_fraction(row.ppm, units, oil_density_kgm3)
        records.append({
            "source_row": int(row.source_row), "date": row.day.strftime("%Y-%m-%d"),
            "timestamp": row.timestamp, "location": row.location, "concentration": float(row.ppm),
            "oil_pct": None if fraction is None else fraction * 100.0,
            "sampler": row.sampler, "method": row.method, "notes": row.notes,
            "status": "comparison not requested",
        })
    comparison = {"paired_count": 0, "stable_pair_count": 0, "median_error_pts": None}
    if days is not None:
        from server.services.tools import sep_oil_loss as loss

        raw_history = pd.DataFrame(columns=["tag", "t", "value"])
        if resolved_location.upper() == UPSTREAM_LOCATION and units != "unknown" and any(r["timestamp"] is not None for r in records):
            try:
                raw_history = loss._raw(days)
            except Exception:
                log.exception("OIW sample historian comparison unavailable")
                notes.append("Historian comparison unavailable. Samples parsed successfully; retry the comparison when historian access returns.")
        comparison = compare_samples(records, raw_history, lag_minutes)
        notes.append("Pairs use V-5317 only and the last flow/WC reports within 15 minutes before sample time minus transport delay. Old reports remain unpaired even when a quiet exception-reported signal may still be valid.")
        notes.append("Positive meter-minus-sample oil error means the meter reads more oil. The median excludes pairs with a preceding 2-minute WC range above 5 points. It is a bias diagnostic, not a saved calibration or an independent accuracy test.")

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
        "units": units,
        "oil_density_kgm3": oil_density_kgm3,
        "rate_basis": rate_basis,
        "comparison_days": days,
        "lag_minutes": lag_minutes,
        "samples": records,
        **comparison,
        "locations_available": locations,
        "first_date": daily[0]["date"] if daily else None,
        "last_date": daily[-1]["date"] if daily else None,
        "sample_count": int(len(at_location)),
        "daily": daily,
        "notes": notes,
    }
