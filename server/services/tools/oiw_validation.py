"""Unit conversion and time-matched grab-sample checks, without calibration writes."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from server.services.tools import sep_oil_loss as loss


def oil_fraction(value: float, units: str, oil_density: float | None) -> float | None:
    """Oil volume / liquid volume. mg/L requires oil density in kg/m3."""
    if units == "unknown":
        return None
    if units == "ppmv":
        fraction = value / 1e6
    elif units == "mg/L":
        if oil_density is None or not math.isfinite(oil_density) or not 500 <= oil_density <= 1200:
            raise ValueError("mg/L conversion requires oil density of 500-1,200 kg/m3 at the sample basis")
        fraction = value / (oil_density * 1000.0)
    else:
        raise ValueError("sample units must be unknown, ppmv, or mg/L; mass ppm must first be converted by the lab")
    if not math.isfinite(fraction) or not 0 <= fraction < 1:
        raise ValueError("sample concentration converts outside 0-100% oil by volume")
    return fraction


def sample_timestamp(day: pd.Timestamp, value: Any) -> str | None:
    """Preserve field clock time; missing/ambiguous/nonexistent times stay unpaired."""
    try:
        if value is None or pd.isna(value) or str(value).strip() == "":
            return None
        if isinstance(value, (float, int, np.number)):
            if not 0 <= value < 1:
                return None
            seconds = round(float(value) * 86400)
        elif hasattr(value, "hour"):
            seconds = value.hour * 3600 + value.minute * 60 + value.second
        else:
            import re

            match = re.fullmatch(r"(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(AM|PM)?", str(value).strip(), re.I)
            if not match:
                return None
            hh, mm, ss = int(match[1]), int(match[2]), int(match[3] or 0)
            if mm >= 60 or ss >= 60 or hh >= 24:
                return None
            if match[4]:
                if not 1 <= hh <= 12:
                    return None
                hh = hh % 12 + (12 if match[4].upper() == "PM" else 0)
            seconds = hh * 3600 + mm * 60 + ss
        stamp = (day.normalize() + pd.Timedelta(seconds=seconds)).tz_localize(
            loss.FIELD_TZ, ambiguous="NaT", nonexistent="NaT"
        )
        return None if pd.isna(stamp) else stamp.isoformat()
    except (ValueError, TypeError):
        return None


def compare_samples(records: list[dict], raw: pd.DataFrame, lag_minutes: float) -> dict:
    """Backward pair each upstream grab with actual reports no more than 15 min old.

    Reporting age is a conservative pairing criterion, not a claim that a
    held exception-reported value is faulty. Never use a future meter reading.
    """
    traces: dict[str, pd.DataFrame] = {}
    for tag, key in ((loss.WC_TAG, "wc"), (loss.FLOW_TAG, "flow")):
        series = raw.loc[raw["tag"] == tag, ["t", "value"]].copy()
        series["t"] = pd.to_datetime(series["t"], utc=True)
        traces[key] = series.sort_values("t").drop_duplicates("t", keep="last")
    errors = []
    for row in records:
        row["status"] = "different stream"
        if row["location"].strip().upper() != "V-5317":
            continue
        if row["timestamp"] is None:
            row["status"] = "missing or ambiguous time"
            continue
        if row["oil_pct"] is None:
            row["status"] = "confirm units"
            continue
        target = pd.Timestamp(row["timestamp"]) - pd.Timedelta(minutes=lag_minutes)
        row["comparison_time"] = target.isoformat()
        matched = {}
        for key, series in traces.items():
            prior = series[series["t"] <= target]
            if prior.empty:
                break
            hit = prior.iloc[-1]
            age = (target - hit["t"]).total_seconds() / 60.0
            if age > 15 or not np.isfinite(hit["value"]):
                break
            matched[key] = float(hit["value"])
            row[f"{key}_age_minutes"] = round(age, 2)
        if len(matched) < 2:
            row["status"] = "no recent historian pair"
            continue
        wc, flow = matched["wc"], matched["flow"]
        if not 0 <= wc <= 100 or flow <= loss.FLOW_MIN_BPD:
            row["status"] = "invalid reading or low flow"
            continue
        row.update(
            status="matched",
            meter_wc_pct=wc,
            meter_oil_pct=100.0 - wc,
            flow_bpd=flow,
            sample_oil_bopd=flow * row["oil_pct"] / 100.0,
            meter_oil_bopd=flow * (1.0 - wc / 100.0),
            error_pts=100.0 - wc - row["oil_pct"],
        )
        # Show local variability; a changing interface makes a point grab a
        # weak offset reference. This diagnostic only uses the preceding 2 min.
        recent = traces["wc"]
        recent = recent[(recent["t"] > target - pd.Timedelta(minutes=2)) & (recent["t"] <= target)]
        prior = traces["wc"][traces["wc"]["t"] <= target - pd.Timedelta(minutes=2)].tail(1)
        values = pd.concat([prior, recent])["value"]
        values = values[np.isfinite(values) & values.between(0, 100)]
        row["wc_range_pts"] = float(values.max() - values.min()) if len(values) else None
        if row["wc_range_pts"] is not None and row["wc_range_pts"] > 5:
            row["status"] = "matched during changing WC"
        else:
            errors.append(row["error_pts"])
    return {
        "paired_count": sum(r["status"].startswith("matched") for r in records),
        "stable_pair_count": len(errors),
        "median_error_pts": float(np.median(errors)) if errors else None,
    }
