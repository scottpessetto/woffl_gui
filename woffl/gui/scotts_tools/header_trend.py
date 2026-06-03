"""Empirical header/WHP → BHP coupling from historian trends.

GUI-side support for the Header Pressure Impact tab's empirical engine. Two
concerns:

1. **Data access** — a *date-bounded*, *hourly* pull of BHP / header / WHP from
   the SCADA historian. Tags come from the authoritative **`mpu.wells.vw_bhp_tags`**
   (the same mapping the app already uses), NOT the hand-maintained
   `bhp_dict.csv`. That view gives two BHP candidates per well (`bhp_esp` = the
   "3" series, `bhp_other` = the "7" series); we pull both and keep whichever
   actually returns data — which auto-resolves the per-well 3-vs-7 split and
   flags genuinely gaugeless wells. WHP is derived (`MPU_PI_<pad_number>2<well>`)
   and the header is a pad-level tag. Covers every well, including ones the old
   CSV was missing.

2. **Slope fitting — within-day, then mean** (the method validated in Scott's
   original tool). For each calendar day we fit BHP-vs-header (and BHP-vs-WHP)
   from that day's hourly points, then average the daily slopes. Each fit is
   intraday, so it isolates the pressure response from slow reservoir drift and
   keeps good signal-to-noise (the intraday swing is large vs gauge noise).

   A responsive-vs-slugging classifier operationalizes the field insight that a
   near-unity, well-correlated slope means BHP genuinely tracks surface pressure,
   while a flat/uncorrelated slope is slugging that won't respond to a header move.

The fit/classify functions are pure (no Streamlit, no Databricks) so they're
unit-testable on synthetic hourly series.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import theilslopes


# ── data access ──────────────────────────────────────────────────────────────

# Pad-level header (manifold) pressure tags — one per pad, stable. vw_bhp_tags
# carries per-well BHP candidates but not the pad header, so these are kept here
# (the header tags historically used in the field; verified against the pads in
# vw_bhp_tags). A pad absent from this map simply gets no HeaderP series.
PAD_HEADER_TAGS: dict[str, str] = {
    "B": "MPU_PI_2019A", "C": "MPU_PI_2619A", "E": "MPU_PI_4540",
    "F": "MPU_PI_2440",  "G": "MPU_PI_3001",  "H": "MPU_PI_3110",
    "I": "MPU_PI_3205",  "J": "MPU_PI_3305",  "K": "MPU_PI_4440",
    "L": "MPU_PI_2540",  "M": "MPU_PI_4287A", "S": "MPU_PI_8006",
}

# Hourly-binned, date-bounded historian pull. Averages into time bins WITHOUT a
# daily-max collapse so each day keeps multiple points for within-day fitting.
# Columns (LocalTime / tag / value / MeasureDate) verified against live data.
_HISTORIAN_TREND_QUERY = """
WITH BinAverages AS (
    SELECT
        CAST(FLOOR(CAST(LocalTime AS BIGINT) / {bin_secs}) * {bin_secs} AS TIMESTAMP) AS time_bin,
        tag,
        AVG(value) AS avg_value
    FROM reporting.historian.vw_mpu_measurements
    WHERE tag IN ({tag_list})
      AND MeasureDate BETWEEN '{start_date}' AND '{end_date}'
    GROUP BY time_bin, tag
)
SELECT time_bin, tag, avg_value
FROM BinAverages
ORDER BY time_bin, tag;
"""


def _clean_tag(t) -> str | None:
    if t is None:
        return None
    s = str(t).strip()
    return s if s and s.lower() != "nan" else None


def _clean_pressure(s: pd.Series) -> pd.Series:
    """Coerce to numeric and drop non-physical readings (<=0 or absurd spikes)."""
    s = pd.to_numeric(s, errors="coerce")
    return s.where((s > 0) & (s < 10000))


def _parse_mp_name(mp: str) -> tuple[str, int] | None:
    """'MPI-22' -> ('I', 22). Returns None if it doesn't parse."""
    m = re.match(r"MP([A-Z]+)-(\d+)", str(mp).strip().upper())
    return (m.group(1), int(m.group(2))) if m else None


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_bhp_tag_map() -> dict[tuple[str, int], dict]:
    """Authoritative well→tag map from mpu.wells.vw_bhp_tags (replaces bhp_dict.csv).

    Keyed by (pad_letter, well_number_int) to dodge leading-zero name mismatches.
    Each value carries both BHP candidates (``bhp_esp`` = '3' series, ``bhp_other``
    = '7' series), a derived WHP tag (``MPU_PI_<pad_number>2<well_number>``), and
    the pad header tag. The live BHP is resolved later by which candidate has data.
    """
    from woffl.assembly.databricks_client import execute_query

    df = execute_query(
        "SELECT well_pad, pad_number, well_number, bhp_esp, bhp_other "
        "FROM mpu.wells.vw_bhp_tags"
    )
    out: dict[tuple[str, int], dict] = {}
    for _, r in df.iterrows():
        pad = str(r["well_pad"]).strip().upper()
        wellnum = str(r["well_number"]).strip()
        try:
            key = (pad, int(wellnum))
        except (TypeError, ValueError):
            continue
        padnum = r["pad_number"]
        whp = f"MPU_PI_{int(padnum)}2{wellnum}" if pd.notna(padnum) else None
        out[key] = {
            "bhp_esp": _clean_tag(r["bhp_esp"]),
            "bhp_other": _clean_tag(r["bhp_other"]),
            "whp": whp,
            "header": PAD_HEADER_TAGS.get(pad),
        }
    return out


# Latest valid reading per tag over a date window. ROW_NUMBER picks the most
# recent cleaned (>0, <10000) value, so a trailing null/spike doesn't win.
_LATEST_HEADER_QUERY = """
WITH recent AS (
    SELECT tag, value,
           ROW_NUMBER() OVER (PARTITION BY tag ORDER BY LocalTime DESC) AS rn
    FROM reporting.historian.vw_mpu_measurements
    WHERE tag IN ({tag_list})
      AND MeasureDate BETWEEN '{start_date}' AND '{end_date}'
      AND value > 0 AND value < 10000
)
SELECT tag, value FROM recent WHERE rn = 1;
"""


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_latest_pad_header(
    pads: tuple[str, ...], start_date: str, end_date: str
) -> dict[str, float]:
    """Latest valid manifold (header) pressure per pad, from the historian.

    Reads each pad's header tag (PAD_HEADER_TAGS) and returns the most recent
    cleaned reading over [start_date, end_date]. Pads without a header tag, or
    with no data in the window, are simply omitted. Keyed by pad letter ('G').

    ``pads`` is a tuple (not list) so the result is cacheable.
    """
    from woffl.assembly.databricks_client import execute_query

    tag_to_pad = {PAD_HEADER_TAGS[p]: p for p in pads if p in PAD_HEADER_TAGS}
    if not tag_to_pad:
        return {}
    tag_list = ", ".join(f"'{t}'" for t in tag_to_pad)
    df = execute_query(
        _LATEST_HEADER_QUERY.format(
            tag_list=tag_list, start_date=start_date, end_date=end_date
        )
    )
    out: dict[str, float] = {}
    if df is None or df.empty:
        return out
    for _, r in df.iterrows():
        pad = tag_to_pad.get(str(r["tag"]).strip())
        if pad is not None and pd.notna(r["value"]):
            out[pad] = float(r["value"])
    return out


def _resolve_bhp(wide: pd.DataFrame, esp_tag: str | None, other_tag: str | None) -> pd.Series | None:
    """Pick the BHP candidate with the most cleaned data (auto-resolves 3 vs 7).

    Returns None when neither candidate has data — i.e. the well is effectively
    gaugeless for the window.
    """
    best = None
    for tag in (esp_tag, other_tag):
        if tag and tag in wide.columns:
            s = _clean_pressure(wide[tag])
            n = int(s.notna().sum())
            if n > 0 and (best is None or n > best[0]):
                best = (n, s)
    return best[1] if best else None


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_header_trends(
    well_names: tuple[str, ...],
    start_date: str,
    end_date: str,
    bin_secs: int = 3600,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Pull date-bounded, sub-daily BHP / HeaderP / WHP per well from the historian.

    Tags come from mpu.wells.vw_bhp_tags (not bhp_dict.csv). For each well the
    live BHP tag is whichever of bhp_esp / bhp_other returns data; WHP is derived;
    header is the pad-level tag.

    Args:
        well_names: tuple of normalized MP well names (e.g. ("MPI-22", "MPB-28")).
            Tuple (not list) so the result is cacheable.
        start_date / end_date: 'YYYY-MM-DD'.
        bin_secs: averaging bin width in seconds (3600 = hourly).

    Returns:
        (well_dfs, missing) where well_dfs maps each well to a timestamp-indexed
        DataFrame with whatever of BHP / HeaderP / WHP resolved, and missing lists
        wells absent from vw_bhp_tags.
    """
    from woffl.assembly.databricks_client import execute_query

    tagmap = fetch_bhp_tag_map()
    per_well: dict[str, dict] = {}
    missing: list[str] = []
    all_tags: set[str] = set()
    for wn in well_names:
        key = _parse_mp_name(wn)
        info = tagmap.get(key) if key else None
        if not info:
            missing.append(wn)
            continue
        per_well[wn] = info
        for t in (info["bhp_esp"], info["bhp_other"], info["whp"], info["header"]):
            if t:
                all_tags.add(t)

    if not all_tags:
        return {}, list(well_names)

    query = _HISTORIAN_TREND_QUERY.format(
        bin_secs=int(bin_secs),
        tag_list=", ".join(f"'{t}'" for t in sorted(all_tags)),
        start_date=start_date,
        end_date=end_date,
    )
    # Let SQL/connection errors propagate — the (uncached) caller surfaces them.
    raw = execute_query(query)
    if raw is None or raw.empty:
        return {}, missing

    raw["time_bin"] = pd.to_datetime(raw["time_bin"])
    wide = raw.pivot(index="time_bin", columns="tag", values="avg_value").sort_index()

    well_dfs: dict[str, pd.DataFrame] = {}
    for wn, info in per_well.items():
        cols: dict[str, pd.Series] = {}
        bhp = _resolve_bhp(wide, info["bhp_esp"], info["bhp_other"])
        if bhp is not None:
            cols["BHP"] = bhp
        if info["whp"] and info["whp"] in wide.columns:
            cols["WHP"] = _clean_pressure(wide[info["whp"]])
        if info["header"] and info["header"] in wide.columns:
            cols["HeaderP"] = _clean_pressure(wide[info["header"]])
        if cols:
            df = pd.DataFrame(cols).dropna(how="all")
            if not df.empty:
                well_dfs[wn] = df

    return well_dfs, missing


# ── within-day slope fitting (pure) ───────────────────────────────────────────


@dataclass
class WithinDayFit:
    """Within-day slopes of ``y`` vs ``x`` for one well, averaged across days.

    ``mean_slope`` is the headline dY/dX (e.g. dBHP/dHeaderP), averaged over the
    days that passed the intraday quality filters. ``daily`` holds the per-day
    fits (day, slope, intercept, r2, n, x_min, x_max) for diagnostics / PNG.
    """

    y_name: str
    x_name: str
    mean_slope: float
    median_slope: float
    slope_std: float
    n_days: int       # fittable days: enough points + real intraday driver movement
    n_good_days: int  # of those, how many are well-correlated (r² ≥ r2_day_min)
    mean_r2: float    # average within-day r² over the fittable days
    daily: pd.DataFrame


def _empty_fit(y_name: str, x_name: str) -> WithinDayFit:
    return WithinDayFit(
        y_name, x_name, np.nan, np.nan, np.nan, 0, 0, np.nan,
        pd.DataFrame(columns=["day", "slope", "intercept", "r2", "n", "x_min", "x_max", "good"]),
    )


def fit_within_day(
    df: pd.DataFrame,
    *,
    y_name: str,
    x_name: str,
    min_pts_per_day: int = 6,
    min_x_range: float = 2.0,
    min_bhp: float = 50.0,
    r2_day_min: float = 0.5,
    slope_lo: float = 0.2,
    slope_hi: float = 1.5,
    robust: bool = False,
) -> WithinDayFit | None:
    """Fit ``y_name`` vs ``x_name`` within each day, then average daily slopes.

    Args:
        df: timestamp-indexed sub-daily DataFrame with the two columns.
        min_pts_per_day: skip days with fewer usable points.
        min_x_range: skip days where the driver's intraday span (psi) is below
            this — a flat driver carries no within-day slope information.
        min_bhp: drop rows with BHP at/below this (sentinel / shut-in garbage).
        r2_day_min, slope_lo, slope_hi: a day is "good" if its within-day r² is
            at least r2_day_min AND its slope is in [slope_lo, slope_hi]. The mean
            slope is taken over GOOD days only — dropping the bad/negative/flat
            daily fits (as the original tool did) keeps slug/noise days from
            diluting the slope toward zero. n_days still counts all fittable days,
            so an all-bad well reads as slugging, not insufficient.
        robust: Theil-Sen per day when True, else OLS (matches the original tool).

    Returns:
        WithinDayFit, or None if either column is absent.
    """
    if y_name not in df.columns or x_name not in df.columns:
        return None

    d = df[[x_name, y_name]].replace([np.inf, -np.inf], np.nan).dropna()
    if "BHP" in (x_name, y_name):
        bhp_col = "BHP" if "BHP" in d.columns else None
        if bhp_col:
            d = d[d[bhp_col] > min_bhp]
    if d.empty:
        return _empty_fit(y_name, x_name)

    days = pd.DatetimeIndex(d.index).normalize()
    recs: list[dict] = []
    for day, g in d.groupby(days):
        x = g[x_name].to_numpy(dtype=float)
        y = g[y_name].to_numpy(dtype=float)
        if x.size < min_pts_per_day:
            continue
        if (x.max() - x.min()) < min_x_range:
            continue  # driver flat within the day — no information
        if robust:
            res = theilslopes(y, x)
            slope, intercept = float(res[0]), float(res[1])
        else:
            slope, intercept = (float(v) for v in np.polyfit(x, y, 1))
        if np.std(x) > 0 and np.std(y) > 0:
            r = float(np.corrcoef(x, y)[0, 1])
            r2 = r * r
        else:
            r2 = np.nan
        recs.append(
            {
                "day": day, "slope": slope, "intercept": intercept, "r2": r2,
                "n": int(x.size), "x_min": float(x.min()), "x_max": float(x.max()),
            }
        )

    daily = pd.DataFrame(recs)
    if daily.empty:
        return _empty_fit(y_name, x_name)

    # "Good" days: a clean, physically-signed fit (good r² AND slope in band).
    # Averaging over good days only — dropping the bad/negative/flat daily fits,
    # as the original tool did — is what keeps slug/noise days from diluting the
    # slope toward zero. n_days counts all fittable days so an all-bad (e.g.
    # sonic-decoupled) well reads as slugging, not insufficient.
    daily["good"] = (
        (daily["r2"] >= r2_day_min)
        & (daily["slope"] >= slope_lo)
        & (daily["slope"] <= slope_hi)
    )
    good = daily[daily["good"]]
    if good.empty:
        return WithinDayFit(
            y_name, x_name, np.nan, np.nan, np.nan, int(len(daily)), 0, np.nan, daily
        )
    return WithinDayFit(
        y_name=y_name,
        x_name=x_name,
        mean_slope=float(good["slope"].mean()),
        median_slope=float(good["slope"].median()),
        slope_std=float(good["slope"].std(ddof=0)),
        n_days=int(len(daily)),
        n_good_days=int(len(good)),
        mean_r2=float(good["r2"].mean()),
        daily=daily,
    )


# Pairs we fit: (response, driver). BHP~WHP is comparable to the physics solve
# (which sweeps WHP); BHP~HeaderP is the direct header→BHP lever; WHP~HeaderP is
# the flowline transmission factor (physics assumes ~1:1).
_FIT_PAIRS = (("BHP", "HeaderP"), ("BHP", "WHP"), ("WHP", "HeaderP"))


def fit_well(df: pd.DataFrame, **kwargs) -> dict[str, WithinDayFit]:
    """Fit every available (response, driver) pair for one well's sub-daily trend.

    Returns {"BHP~HeaderP": WithinDayFit, ...} for whichever pairs both columns
    exist. ``kwargs`` pass through to :func:`fit_within_day`.
    """
    out: dict[str, WithinDayFit] = {}
    for y_name, x_name in _FIT_PAIRS:
        fit = fit_within_day(df, y_name=y_name, x_name=x_name, **kwargs)
        if fit is not None:
            out[f"{y_name}~{x_name}"] = fit
    return out


def classify_response(
    fit: WithinDayFit | None,
    *,
    min_fittable: int = 5,
    min_good: int = 5,
    good_frac: float = 0.25,
) -> str:
    """Label a fit 'responsive', 'slugging', or 'insufficient'.

    insufficient = too few fittable days (sparse data / driver never moved).
    responsive   = a clean coupling on enough days (≥min_good good days AND
                   ≥good_frac of fittable days good) → BHP genuinely tracks
                   surface pressure; the good-day mean slope is meaningful.
    slugging     = fittable days exist but few are clean/positive → no consistent
                   response (e.g. a sonic-decoupled jet pump). Defaults are
                   tunable and should be validated against wells you know.
    """
    if fit is None or fit.n_days < min_fittable:
        return "insufficient"
    if fit.n_good_days < min_good or (fit.n_good_days / max(fit.n_days, 1)) < good_frac:
        return "slugging"
    return "responsive"
