"""First-stage water-leg carryover scenarios from four historian tags.

MPU_FI_5365 is assumed to be total liquid BPD on the water outlet;
MPU_AI_5317 supplies volume water cut (%). MPU_LIC_5365CV1 and
MPU_LC5365SP1 are the controlled level indication and its setpoint.

The raw integral is Q * (1 - WC/100). The excursion integral references
WC to a causal trailing 24-hour p95 on a fixed minute clock (80-100%).
Its subtraction may remove instrument bias OR sustained real carryover;
the meter alone cannot distinguish them. Always expose the raw volume and
reference subtraction. The two capped scenarios are sensitivity cases,
not confidence bounds or physical bounds during inventory transients.

Use every tag transition and split intervals at Alaska midnight. Exclude
invalid WC/flow, flow <=1,000 BPD and flow report ages >=15 minutes. Hold
exception-reported WC; without instrument quality tags this cannot establish
meter health. Event level labels describe indications, not root causes.

Reads are TTL-cached. No production writes. See
docs/separator_sample_workflow_2026-09-14.md for validation and limitations.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from server import config
from server.cache import ttl_cache

log = logging.getLogger("woffl.web.tools.sep_oil_loss")

FLOW_TAG = "MPU_FI_5365"
WC_TAG = "MPU_AI_5317"
# The CONTROLLED level indication - the level the controller is acting on,
# whichever measurement method is currently selected. Verified against the
# setpoint over 35 days: mean |LIC - SP| = 3.98 points and corr 0.71, versus
# 14.53 / 0.35 for the older MPU_LI_5365A transmitter, which the loop is not
# using. Reading LI_5365A called 34 of 50 events "level loss"; on the channel
# that is actually in control only 5.5 h of 718 sit below 20%, so most upsets
# are separation failing at a NORMAL, held level - a different problem.
LEVEL_TAG = "MPU_LIC_5365CV1"
LEVEL_SP_TAG = "MPU_LC5365SP1"

FIELD_TZ = "America/Anchorage"

# Below this the separator is down or the meter is drifting around zero; the
# hours are excluded from every average and integral. Normal flow is 40k-120k.
FLOW_MIN_BPD = 1_000.0
# Film baseline: trailing window and quantile. 24 h spans a full operating
# cycle; p95 rides the top of the band without chasing a single spike.
BASELINE_WINDOW = "24h"
BASELINE_QUANTILE = 0.95
BASELINE_FLOOR, BASELINE_CEIL = 80.0, 100.0
# An upset is water cut this many points below the meter's own plateau.
UPSET_DROP_PTS = 5.0
# Event stitching: ignore blips shorter than this, and treat two dips inside
# the merge gap as one event (the interface oscillates on a ~5-10 min cycle).
EVENT_MIN_MINUTES = 2.0
EVENT_MERGE_MINUTES = 10.0
# A level this low means the vessel lost its water inventory outright.
LEVEL_LOSS_PCT = 20.0
# Sitting this far below setpoint means the loop is calling for level it
# cannot hold - losing control without losing the vessel.
LEVEL_SP_BAND_PTS = 10.0
# Longest gap a single sample may represent, so one historian dropout cannot
# smear a stale value across hours of integral.
MAX_SAMPLE_HOURS = 0.25

MAX_EVENTS = 50
MAX_SERIES_POINTS = 1_500

PERIODS: tuple[tuple[str, int], ...] = (("Last 24 h", 1), ("Last 7 d", 7), ("Last 30 d", 30))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@ttl_cache(config.TTL_HISTORIAN, maxsize=8)
def _raw(days: int) -> pd.DataFrame:
    """Historian rows for the four tags over a trailing window (one query)."""
    from woffl.assembly.historian_client import fetch_tag_history

    return fetch_tag_history([FLOW_TAG, WC_TAG, LEVEL_TAG, LEVEL_SP_TAG], days)


def _grid(raw: pd.DataFrame) -> pd.DataFrame:
    """Step-hold every tag onto the union clock, retaining missing coverage.

    Returns a frame of local timestamps, flow (BPD), WC/level/setpoint (%),
    valid flags and forward integration durations (hours).
    """
    columns = ["t", "flow", "wc", "level", "level_sp", "dt_h", "valid"]
    if raw.empty:
        return pd.DataFrame(columns=columns)

    def one(tag: str, name: str) -> pd.DataFrame:
        sub = raw[raw["tag"] == tag][["t", "value"]].rename(columns={"value": name})
        sub = sub.sort_values("t").drop_duplicates("t").reset_index(drop=True)
        sub["t"] = pd.to_datetime(sub["t"], utc=True).dt.tz_convert(FIELD_TZ)
        return sub

    flow = one(FLOW_TAG, "flow")
    if flow.empty:
        return pd.DataFrame(columns=columns)
    # Integrate on every reported change, including a WC excursion entirely
    # between two flow reports. Split at field midnight for exact daily totals.
    lo, hi = flow["t"].min(), flow["t"].max()
    times = pd.DatetimeIndex(pd.to_datetime(raw["t"], utc=True)).tz_convert(FIELD_TZ)
    midnights = pd.date_range(lo.normalize(), hi.normalize() + pd.DateOffset(days=1), freq="D")
    expires = pd.DatetimeIndex(flow["t"] + pd.Timedelta(hours=MAX_SAMPLE_HOURS))
    times = times.union(midnights).union(expires).unique().sort_values()
    grid = pd.DataFrame({"t": times[(times >= lo) & (times <= hi)]})
    flow["flow_reported"] = flow["t"]
    grid = pd.merge_asof(grid, flow, on="t", direction="backward")
    for tag, name in ((WC_TAG, "wc"), (LEVEL_TAG, "level"), (LEVEL_SP_TAG, "level_sp")):
        other = one(tag, name)
        if other.empty:
            grid[name] = np.nan
        else:
            grid = pd.merge_asof(grid, other, on="t", direction="backward")

    span = grid["t"].diff().dt.total_seconds().div(3600.0).shift(-1)
    grid["dt_h"] = span.fillna(0.0).clip(0.0, MAX_SAMPLE_HOURS)
    flow_age = (grid["t"] - grid["flow_reported"]).dt.total_seconds() / 3600.0
    grid["valid"] = (
        np.isfinite(grid["flow"]) & (grid["flow"] > FLOW_MIN_BPD)
        & np.isfinite(grid["wc"]) & grid["wc"].between(0.0, 100.0)
        & (flow_age < MAX_SAMPLE_HOURS)
    )
    for name in ("level", "level_sp"):
        grid[name] = grid[name].where(grid[name].between(0.0, 100.0))
    grid.loc[~grid["valid"], "dt_h"] = 0.0
    return grid


def _film_baseline(grid: pd.DataFrame) -> np.ndarray:
    """Causal rolling excursion reference (%); an oil-free plateau is unproven."""
    if grid.empty:
        return np.zeros(0)
    indexed = grid.set_index("t")
    # Fixed minute support keeps exception-report frequency from weighting
    # the reference. Only prior observations contribute; never backfill it
    # from future clean readings. This is an excursion reference, not proof
    # that the plateau is oil-free or that a change was caused by filming.
    clock = pd.date_range(indexed.index[0], indexed.index[-1], freq="1min")
    sampled = indexed.reindex(clock, method="ffill")
    valid_wc = sampled["wc"].where(sampled["valid"])
    rolled = (
        valid_wc.rolling(BASELINE_WINDOW, min_periods=1)
        .quantile(BASELINE_QUANTILE)
        .clip(BASELINE_FLOOR, BASELINE_CEIL)
    )
    return rolled.reindex(indexed.index, method="ffill").fillna(BASELINE_CEIL).to_numpy()


# ---------------------------------------------------------------------------
# Loss model
# ---------------------------------------------------------------------------


def _oil_rates(
    grid: pd.DataFrame, field_oil_bopd: float, max_oil_frac: float
) -> pd.DataFrame:
    """Add raw indication, reference subtraction and ordered scenarios (BOPD)."""
    out = grid.copy()
    out["base"] = _film_baseline(out)
    # Excursion fraction relative to the assumed reference.
    deficit = ((out["base"] - out["wc"]) / 100.0).clip(lower=0.0)
    out["oil_upper"] = (out["flow"] * deficit).clip(upper=field_oil_bopd)
    out["oil_lower"] = out["flow"] * deficit.clip(upper=max_oil_frac)
    out["oil_lower"] = np.minimum(out["oil_lower"], out["oil_upper"])
    out["oil_raw"] = out["flow"] * (1.0 - out["wc"] / 100.0)
    out["oil_reference_removed"] = out["oil_raw"] - out["flow"] * deficit
    out.loc[~out["valid"], ["oil_upper", "oil_lower", "oil_raw", "oil_reference_removed"]] = 0.0
    return out


def _tw_mean(values: pd.Series, weights: pd.Series) -> Optional[float]:
    """Time-weighted mean, or None when the window carries no valid time."""
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = values[keep], weights[keep]
    total = float(weights.sum())
    if total <= 0:
        return None
    return float(np.average(values.to_numpy(dtype=float), weights=weights.to_numpy(dtype=float)))


def _barrels(rate_bopd: pd.Series, dt_h: pd.Series) -> float:
    """Integrate a BOPD rate over hours into barrels."""
    return float((rate_bopd * dt_h / 24.0).sum())


def _events(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Contiguous upsets, worst barrels first.

    An upset is water cut at least ``UPSET_DROP_PTS`` below the meter's own
    plateau while the separator is running. Dips closer together than
    ``EVENT_MERGE_MINUTES`` are one event - the interface oscillates, so the
    raw mask alone shatters a single upset into hundreds of slivers.

    Each event is classified against the CONTROLLED level and its setpoint,
    which is the whole diagnostic value of the level channel:

    * ``level loss`` - the vessel lost its water inventory outright.
    * ``off setpoint`` - the loop is holding well under what it is calling
      for, so it is losing the level without losing the vessel.
    * ``at setpoint`` - level held where the operator asked and the stream
      STILL went oil. This signature alone does not establish the cause.
    """
    if frame.empty:
        return []
    mask = ((frame["wc"] < frame["base"] - UPSET_DROP_PTS) & frame["valid"]).to_numpy()
    hits = np.flatnonzero(mask)
    if hits.size == 0:
        return []

    stamps = frame["t"].to_numpy()
    gap = np.timedelta64(int(EVENT_MERGE_MINUTES * 60), "s")
    splits = np.flatnonzero((stamps[hits][1:] - stamps[hits][:-1]) > gap)

    events: list[dict[str, Any]] = []
    for chunk in np.split(hits, splits + 1):
        block = frame.iloc[chunk[0] : chunk[-1] + 1]
        hours = float(block["dt_h"].sum())
        if hours * 60.0 < EVENT_MIN_MINUTES:
            continue
        weights = block["dt_h"].clip(lower=1e-9)
        level_min = float(block["level"].min()) if block["level"].notna().any() else None
        level_avg = (
            None
            if block["level"].isna().all()
            else float(_tw_mean(block["level"].ffill().bfill(), weights) or 0.0)
        )
        sp_avg = (
            None
            if block["level_sp"].isna().all()
            else float(_tw_mean(block["level_sp"].ffill().bfill(), weights) or 0.0)
        )
        deviation = None if (level_avg is None or sp_avg is None) else level_avg - sp_avg

        if level_min is not None and level_min < LEVEL_LOSS_PCT:
            kind = "level loss"
        elif deviation is None or block[["level", "level_sp"]].isna().any().any():
            kind = "unknown"
        elif abs(deviation) > LEVEL_SP_BAND_PTS:
            kind = "off setpoint"
        else:
            kind = "at setpoint"

        events.append(
            {
                "start": block["t"].iloc[0].isoformat(),
                "end": (block["t"].iloc[-1] + pd.Timedelta(hours=block["dt_h"].iloc[-1])).isoformat(),
                "hours": round(hours, 3),
                "wc_min": round(float(block["wc"].min()), 2),
                "wc_avg": round(float(_tw_mean(block["wc"], weights) or 0.0), 2),
                "flow_avg": round(float(_tw_mean(block["flow"], weights) or 0.0), 1),
                "bbl_upper": round(_barrels(block["oil_upper"], block["dt_h"]), 1),
                "bbl_lower": round(_barrels(block["oil_lower"], block["dt_h"]), 1),
                "level_min": None if level_min is None else round(level_min, 1),
                "level_avg": None if level_avg is None else round(level_avg, 1),
                "level_sp_avg": None if sp_avg is None else round(sp_avg, 1),
                "level_dev_avg": None if deviation is None else round(deviation, 1),
                "kind": kind,
            }
        )
    events.sort(key=lambda e: e["bbl_upper"], reverse=True)
    return events


def _clip_intervals(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Intersect integration intervals with a requested half-open window."""
    out = frame.copy()
    left = out["t"].clip(lower=start)
    right = (out["t"] + pd.to_timedelta(out["dt_h"], unit="h")).clip(upper=end)
    out["dt_h"] = ((right - left).dt.total_seconds() / 3600.0).clip(lower=0.0)
    return out[out["dt_h"] > 0].copy()


def _period_rows(
    frame: pd.DataFrame, events: list[dict[str, Any]], field_oil_bopd: float, window_days: int
) -> list[dict[str, Any]]:
    """Roll-ups for each standard look-back that fits inside the window."""
    if frame.empty:
        return []
    end = frame["t"].max()
    starts = {e["start"]: pd.Timestamp(e["start"]) for e in events}

    rows: list[dict[str, Any]] = []
    for label, days in PERIODS:
        if days > window_days:
            continue
        cut = end - pd.Timedelta(days=days)
        block = _clip_intervals(frame, cut, end)
        hours = float(block["dt_h"].sum())
        if hours <= 0:
            continue
        field_bbl = field_oil_bopd * hours / 24.0
        upper = _barrels(block["oil_upper"], block["dt_h"])
        lower = _barrels(block["oil_lower"], block["dt_h"])
        upset = float(block.loc[block["wc"] < block["base"] - UPSET_DROP_PTS, "dt_h"].sum())
        rows.append(
            {
                "label": label,
                "days": days,
                "hours": round(hours, 2),
                "downtime_hours": round(float(days * 24.0 - hours), 2),
                "flow_avg": round(float(_tw_mean(block["flow"], block["dt_h"]) or 0.0), 1),
                "wc_avg": round(float(_tw_mean(block["wc"], block["dt_h"]) or 0.0), 2),
                "base_avg": round(float(_tw_mean(block["base"], block["dt_h"]) or 0.0), 2),
                "bbl_upper": round(upper, 1),
                "bbl_lower": round(lower, 1),
                "bbl_raw": round(_barrels(block["oil_raw"], block["dt_h"]), 1),
                "bbl_reference_removed": round(_barrels(block["oil_reference_removed"], block["dt_h"]), 1),
                "bopd_upper": round(upper / (hours / 24.0), 1),
                "bopd_lower": round(lower / (hours / 24.0), 1),
                "pct_field_upper": round(100.0 * upper / field_bbl, 2) if field_bbl > 0 else None,
                "pct_field_lower": round(100.0 * lower / field_bbl, 2) if field_bbl > 0 else None,
                "upset_hours": round(upset, 2),
                "events": sum(1 for s in starts.values() if s > cut),
            }
        )
    return rows


def _daily_rows(
    frame: pd.DataFrame, events: list[dict[str, Any]], field_oil_bopd: float
) -> list[dict[str, Any]]:
    """One row per FIELD calendar day the window touches.

    Days are Alaska local, not UTC: an operator asking "what did we lose
    yesterday" means their own midnight, and a UTC cut would slice every
    night shift's upset across two bars.

    ``covered_hours`` is how much of the day the window actually spans, so a
    clipped first or last day is not read as a quiet one. ``hours`` is the
    running time inside that; the difference includes downtime and missing/invalid data.

    Args:
        frame (DataFrame): The rate-annotated grid.
        events (list): Already-built events, counted into their start day.
        field_oil_bopd (float): Denominator for the percent-of-field columns.

    Returns:
        rows (list): Chronological dicts, one per day.
    """
    if frame.empty:
        return []

    stamps = frame["t"]
    window_start, window_end = stamps.min(), stamps.max()
    day = stamps.dt.normalize()

    per_day_events: dict[str, int] = {}
    for event in events:
        key = pd.Timestamp(event["start"]).strftime("%Y-%m-%d")
        per_day_events[key] = per_day_events.get(key, 0) + 1

    upset = frame["wc"] < frame["base"] - UPSET_DROP_PTS
    rows: list[dict[str, Any]] = []
    for stamp, block in frame.groupby(day, sort=True):
        hours = float(block["dt_h"].sum())
        span_lo = max(stamp, window_start)
        next_midnight = stamp + pd.DateOffset(days=1)
        day_hours = (next_midnight - stamp).total_seconds() / 3600.0
        span_hi = min(next_midnight, window_end)
        covered = max((span_hi - span_lo).total_seconds() / 3600.0, 0.0)
        upper = _barrels(block["oil_upper"], block["dt_h"])
        lower = _barrels(block["oil_lower"], block["dt_h"])
        # A day the separator barely ran has a denominator near zero, which
        # turns a handful of barrels into a double-digit percentage. Report
        # the barrels and leave the share blank.
        field_bbl = field_oil_bopd * hours / 24.0 if hours >= 1.0 else 0.0
        key = stamp.strftime("%Y-%m-%d")
        rows.append(
            {
                "date": key,
                "hours": round(hours, 2),
                "covered_hours": round(covered, 2),
                "bbl_upper": round(upper, 1),
                "bbl_lower": round(lower, 1),
                "bbl_raw": round(_barrels(block["oil_raw"], block["dt_h"]), 1),
                "bbl_reference_removed": round(_barrels(block["oil_reference_removed"], block["dt_h"]), 1),
                "pct_field_upper": round(100.0 * upper / field_bbl, 2) if field_bbl > 0 else None,
                "pct_field_lower": round(100.0 * lower / field_bbl, 2) if field_bbl > 0 else None,
                "upset_hours": round(float(block.loc[upset.loc[block.index], "dt_h"].sum()), 2),
                "events": per_day_events.get(key, 0),
                # A day the window only clips, or one the separator spent down,
                # cannot be compared bar-for-bar against a full running day.
                "partial": bool(hours < covered - 0.5 or covered < day_hours - 0.5),
            }
        )
    return rows


def _series(frame: pd.DataFrame) -> dict[str, list]:
    """Even-stride downsample for the chart, plus the cumulative loss curve."""
    if frame.empty:
        return {
            k: []
            for k in (
                "t", "flow", "wc", "base", "level", "level_sp",
                "oil_upper", "cum_upper", "cum_lower",
            )
        }

    work = frame.copy()
    work["cum_upper"] = (work["oil_upper"] * work["dt_h"] / 24.0).cumsum()
    work["cum_lower"] = (work["oil_lower"] * work["dt_h"] / 24.0).cumsum()

    size = len(work)
    if size > MAX_SERIES_POINTS:
        keep = np.unique(np.linspace(0, size - 1, MAX_SERIES_POINTS).round().astype(int))
        work = work.iloc[keep]

    def col(name: str, dp: int) -> list[Optional[float]]:
        return [None if not np.isfinite(v) else round(float(v), dp) for v in work[name]]

    return {
        "t": [ts.isoformat() for ts in work["t"]],
        "flow": col("flow", 1),
        "wc": col("wc", 2),
        "base": col("base", 2),
        "level": col("level", 2),
        "level_sp": col("level_sp", 2),
        "oil_upper": col("oil_upper", 1),
        "cum_upper": col("cum_upper", 1),
        "cum_lower": col("cum_lower", 1),
    }


def sep_oil_loss_day(
    date: str,
    days: int = 14,
    field_oil_bopd: float = 65_000.0,
    max_oil_frac: float = 0.10,
) -> dict[str, Any]:
    """One FIELD calendar day at full resolution, for the drill-down.

    The window view downsamples the whole span to ``MAX_SERIES_POINTS``, which
    over 14 days is about 13 minutes per point - too coarse to read the
    interface oscillation that drives an upset. Re-slicing to one day spends
    the same point budget on 86,400 s, so the trace lands near a minute.

    Costs no extra historian round trip: ``_raw`` is already cached for this
    window, and the day is a slice of that same frame.

    Args:
        date (str): Field calendar day, ``YYYY-MM-DD``.
        days (int): The window the day was picked from, 1-90.
        field_oil_bopd (float): Same ceiling and denominator as the window.
        max_oil_frac (float): Same conservative-bound cap as the window.

    Returns:
        payload (dict): Matching schemas.SepOilLossDayResponse.

    Raises:
        ValueError: Bad date, an argument out of range, or a day the window
            does not cover.
    """
    try:
        stamp = pd.Timestamp(date)
    except ValueError as exc:
        raise ValueError(f"{date!r} is not a YYYY-MM-DD date") from exc
    if stamp.tzinfo is not None or stamp != stamp.normalize():
        raise ValueError(f"{date!r} must be a bare YYYY-MM-DD date")

    window = int(days)
    if not 1 <= window <= 90:
        raise ValueError(f"days must be 1-90, got {days}")
    if not 1_000.0 <= field_oil_bopd <= 200_000.0:
        raise ValueError(f"field_oil_bopd must be 1,000-200,000, got {field_oil_bopd}")
    if not 0.0 < max_oil_frac <= 1.0:
        raise ValueError(f"max_oil_frac must be in (0, 1], got {max_oil_frac}")

    grid = _grid(_raw(window))
    if grid.empty:
        raise ValueError("the historian returned no rows for this window")

    # Everything except the series comes from the SAME pass the window view
    # makes, then is filtered to the day. Re-running the event walk on a
    # one-day slice would re-split an upset that straddles midnight and the
    # drill-down would disagree with the bar it was opened from.
    frame = _oil_rates(grid, field_oil_bopd, max_oil_frac)
    events = _events(frame)
    summary = next(
        (r for r in _daily_rows(frame, events, field_oil_bopd) if r["date"] == date),
        None,
    )
    if summary is None:
        raise ValueError(f"{date} is outside the {window} day window")

    # The film baseline is a trailing 24 h quantile, so it too has to come
    # from the whole window; only the plotted slice is per day.
    day = frame[frame["t"].dt.normalize() == stamp.tz_localize(FIELD_TZ)]
    return {
        "date": date,
        "days": window,
        "summary": summary,
        "events": [e for e in events if e["start"][:10] == date],
        "series": _series(day),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def sep_oil_loss(
    days: int = 14,
    field_oil_bopd: float = 65_000.0,
    max_oil_frac: float = 0.10,
) -> dict[str, Any]:
    """Oil leaving with the first-stage separator water leg.

    Args:
        days (int): Trailing window, 1-90.
        field_oil_bopd (float): Field oil production used as the scenario
            cap on the water leg and as the denominator for the
            percent-of-field columns. Milne sells 50,000-65,000 BOPD; the
            default takes the top of that range, which raises the rate cap
            (a larger upper bound) and lowers every percent-of-field share.
        max_oil_frac (float): Ceiling on the oil FRACTION of the water leg for
            the conservative bound, 0-1. Defaults to 0.10 - a leg running more
            than a tenth oil is already an upset, so a looser cap approaches the field-capped scenario. Neither is a proven bound.

    Returns:
        payload (dict): Matching schemas.SepOilLossResponse.

    Raises:
        ValueError: Argument outside its range.
    """
    window = int(days)
    if not 1 <= window <= 90:
        raise ValueError(f"days must be 1-90, got {days}")
    if not 1_000.0 <= field_oil_bopd <= 200_000.0:
        raise ValueError(f"field_oil_bopd must be 1,000-200,000, got {field_oil_bopd}")
    if not 0.0 < max_oil_frac <= 1.0:
        raise ValueError(f"max_oil_frac must be in (0, 1], got {max_oil_frac}")

    grid = _grid(_raw(window))
    if grid.empty:
        return {
            "flow_tag": FLOW_TAG,
            "wc_tag": WC_TAG,
            "level_tag": LEVEL_TAG,
            "level_sp_tag": LEVEL_SP_TAG,
            "days": window,
            "start": None,
            "end": None,
            "field_oil_bopd": field_oil_bopd,
            "max_oil_frac": max_oil_frac,
            "flow_min_bpd": FLOW_MIN_BPD,
            "upset_drop_pts": UPSET_DROP_PTS,
            "valid_hours": 0.0,
            "excluded_hours": 0.0,
            "periods": [],
            "daily": [],
            "events": [],
            "series": _series(grid),
        }

    frame = _oil_rates(grid, field_oil_bopd, max_oil_frac)
    events = _events(frame)
    span_h = float(
        (frame["t"].max() - frame["t"].min()).total_seconds() / 3600.0
    )
    valid_h = float(frame["dt_h"].sum())

    return {
        "flow_tag": FLOW_TAG,
        "wc_tag": WC_TAG,
        "level_tag": LEVEL_TAG,
        "level_sp_tag": LEVEL_SP_TAG,
        "days": window,
        "start": frame["t"].min().isoformat(),
        "end": frame["t"].max().isoformat(),
        "field_oil_bopd": field_oil_bopd,
        "max_oil_frac": max_oil_frac,
        "flow_min_bpd": FLOW_MIN_BPD,
        "upset_drop_pts": UPSET_DROP_PTS,
        "valid_hours": round(valid_h, 2),
        "excluded_hours": round(max(span_h - valid_h, 0.0), 2),
        "periods": _period_rows(frame, events, field_oil_bopd, window),
        "daily": _daily_rows(frame, events, field_oil_bopd),
        "events": events[:MAX_EVENTS],
        "series": _series(frame),
    }
