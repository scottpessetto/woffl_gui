"""Pure math for the Header page: WHP/BHP relations, lift-group correlations,
Vogel inflow and the per-well response to a production-header change.

No IO here, so every rule is unit-testable offline. ``header_study`` supplies
the data and owns the jobs.

The chain for a well that is not a jet pump::

    dWHP = r * dHeader        r: within-day WHP~Header slope (default 1.0)
    dBHP = s * dWHP           s: CLOSED-LOOP within-day BHP~WHP slope
    dLiq = IPR(BHP + dBHP) - IPR(BHP)
    dOil = dLiq * (1 - WC)    total-liquid IPR, oil derived once

``s`` is measured while the well is producing, so it already contains the rate
falling as BHP rises (for an ESP at fixed speed, ``s = 1 / (1 + PI * k)`` with
``k`` the pump curve's head loss per barrel). It must never be coupled to the
IPR a second time. High-PI ESPs therefore show small slopes and still lose the
most liquid: ``dLiq = -PI * s * dWHP``.

Jet pumps do not use this chain; ``header_study`` solves them with the WOFFL
model at both wellhead pressures.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import pandas as pd

# Lift groups that carry an empirical relation. JP wells are solved physically.
LIFT_GROUPS = ("ESP", "gas-lift", "flowing")

# Documented reservoir-pressure fallbacks (psig) when nothing is saved or
# fitted - the same numbers the JP IPR estimator caps at (ipr_analyzer).
RES_DEFAULT_PRES = {"schrader": 1800.0, "kuparuk": 3000.0, "sag": 3000.0}
RES_DEFAULT_FALLBACK = 1800.0

# Upper bound for a pseudo reservoir pressure backed out of test points.
RES_PR_MAX = {"schrader": 2200.0, "kuparuk": 4200.0, "sag": 4200.0}
PR_MAX_FALLBACK = 3500.0

# Highest BHP a gauge can plausibly read, shut in or flowing. Deliberately
# separate from the fit cap: shut-in Schrader gauges read ~2,450 (MPR-111)
# and ~2,620 psi (MPL-46) on 2026-09-29, above the 2,200 fit cap, and are
# real. MPL-20's tag (PF pressure at the pump entry, ~4,630) and MPR-110's
# stuck 4,188 stay above these.
GAUGE_MAX = {"schrader": 3200.0, "kuparuk": 4500.0, "sag": 4500.0}
GAUGE_MAX_FALLBACK = 4500.0

# A within-day fit day counts when its r2 reaches this.
R2_DAY_MIN = 0.5
# A well's relation is "measured" with at least this many fit days, making up
# at least this fraction of the days on which the driver moved.
MEASURED_MIN_DAYS = 5
MEASURED_MIN_FRAC = 0.25
# Physical clip for any slope used in a prediction.
SLOPE_CLIP = (0.0, 1.2)
# An IPR needs this much drawdown to be usable (EVID-F17 precedent).
MIN_DRAWDOWN_PSI = 300.0

# prop_hist encodings (value_type double in prop_xref).
REL_SOURCE_CODE = {"measured": 1.0, "correlation": 2.0, "manual": 3.0}
# 2 was labelled "assumed" in prop_xref; it is the reservoir correlation.
IPR_SOURCE_CODE = {"fit": 1.0, "correlation": 2.0, "manual": 3.0}
REL_SOURCE_NAME = {int(v): k for k, v in REL_SOURCE_CODE.items()}
IPR_SOURCE_NAME = {int(v): k for k, v in IPR_SOURCE_CODE.items()}


def _finite(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def reservoir_key(reservoir: Any) -> str:
    """Lower-cased reservoir name ('schrader', 'kuparuk', ...) or ''."""
    return str(reservoir or "").strip().lower()


def default_pres(reservoir: Any) -> float:
    """Documented fallback reservoir pressure (psig) for a reservoir name."""
    return RES_DEFAULT_PRES.get(reservoir_key(reservoir), RES_DEFAULT_FALLBACK)


def pr_cap(reservoir: Any) -> float:
    """Upper bound (psig) for a pseudo-Pr backed out of test points."""
    return RES_PR_MAX.get(reservoir_key(reservoir), PR_MAX_FALLBACK)


# ── within-day relation summary ──────────────────────────────────────────────


def summarize_daily(daily: Optional[pd.DataFrame], r2_min: float = R2_DAY_MIN) -> dict[str, Any]:
    """Summarize per-day fits (``header_trend.fit_within_day().daily``).

    Unlike the old tool's classifier, no slope band is applied: a high-rate
    ESP genuinely couples at 0.05-0.3, and the old [0.2, 1.5] "good day" band
    called those wells slugging. A day counts when its own r2 reaches
    ``r2_min``; the slope is the median over those days with its IQR.

    Args:
        daily: frame with ``slope`` and ``r2`` per day (may be None/empty).
        r2_min: per-day r2 needed for a day to count.

    Returns:
        dict: slope, q25, q75 (psi/psi), r2 (mean over fit days), n_fit,
        n_days, status ("measured" | "weak" | "no_data").
    """
    out: dict[str, Any] = {"slope": None, "q25": None, "q75": None, "r2": None,
                           "n_fit": 0, "n_days": 0, "status": "no_data"}
    if daily is None or len(daily) == 0 or "slope" not in daily or "r2" not in daily:
        return out
    n_days = int(len(daily))
    fit = daily[pd.to_numeric(daily["r2"], errors="coerce") >= r2_min]
    out["n_days"] = n_days
    out["n_fit"] = int(len(fit))
    if fit.empty:
        out["status"] = "weak"
        return out
    slopes = pd.to_numeric(fit["slope"], errors="coerce").dropna()
    if slopes.empty:
        out["status"] = "weak"
        return out
    out["slope"] = float(slopes.median())
    out["q25"] = float(slopes.quantile(0.25))
    out["q75"] = float(slopes.quantile(0.75))
    out["r2"] = float(pd.to_numeric(fit["r2"], errors="coerce").mean())
    measured = len(slopes) >= MEASURED_MIN_DAYS and len(slopes) / max(n_days, 1) >= MEASURED_MIN_FRAC
    out["status"] = "measured" if measured else "weak"
    return out


def clip_slope(s: Optional[float]) -> Optional[float]:
    v = _finite(s)
    if v is None:
        return None
    return min(max(v, SLOPE_CLIP[0]), SLOPE_CLIP[1])


# ── lift-group correlation ───────────────────────────────────────────────────


def fit_correlation(points: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Correlate measured closed-loop slopes with liquid rate for one lift group.

    ``s = a + b * ln(q_liq)`` by Theil-Sen, because the slope falls as the
    well's deliverability rises (``s = 1/(1 + PI*k)``) and rate is the one
    deliverability proxy every well has, gauge or not. With fewer than five
    wells, or a rate range under 2x, the trend is not identifiable and the
    group median is used instead (b = 0).

    Args:
        points: ``{"well", "q_liq" (BLPD), "slope" (psi/psi)}`` for wells whose
            relation is measured.

    Returns:
        dict with kind ("trend" | "median"), a, b, n, q_min, q_max, resid_mad,
        wells; None when no point is usable.
    """
    pts = [(p["well"], _finite(p.get("q_liq")), _finite(p.get("slope"))) for p in points]
    pts = [(w, q, s) for w, q, s in pts if q is not None and q > 0 and s is not None]
    if not pts:
        return None
    wells = [w for w, _, _ in pts]
    q = np.array([p[1] for p in pts], dtype=float)
    s = np.array([p[2] for p in pts], dtype=float)
    x = np.log(q)
    kind, a, b = "median", float(np.median(s)), 0.0
    if len(pts) >= 5 and q.max() / q.min() >= 2.0:
        from scipy.stats import theilslopes

        b, a = (float(v) for v in theilslopes(s, x)[:2])
        kind = "trend"
    resid = s - (a + b * x)
    return {
        "kind": kind, "a": a, "b": b, "n": len(pts),
        "q_min": float(q.min()), "q_max": float(q.max()),
        "resid_mad": float(np.median(np.abs(resid - np.median(resid)))),
        "wells": wells,
    }


def predict_slope(corr: Optional[dict[str, Any]], q_liq: Any) -> Optional[float]:
    """Slope from a group correlation at a liquid rate (clamped to its range)."""
    if not corr:
        return None
    if corr["kind"] == "median" or corr["b"] == 0.0:
        return clip_slope(corr["a"])
    q = _finite(q_liq)
    if q is None or q <= 0:
        return None
    q = min(max(q, corr["q_min"]), corr["q_max"])
    return clip_slope(corr["a"] + corr["b"] * math.log(q))


# ── Vogel inflow ─────────────────────────────────────────────────────────────


def vogel_factor(r: float) -> float:
    r = min(max(r, 0.0), 1.0)
    return 1.0 - 0.2 * r - 0.8 * r * r


def ipr_valid(ipr: Optional[dict[str, Any]]) -> Optional[str]:
    """None when the IPR anchor is usable, else the reason it is not."""
    if not ipr:
        return "no IPR"
    qwf, pwf, pres = (_finite(ipr.get(k)) for k in ("qwf", "pwf", "pres"))
    if qwf is None or pwf is None or pres is None:
        return "IPR anchor incomplete"
    if qwf <= 0:
        return "IPR rate must be positive"
    if pwf <= 0:
        return "IPR flowing pressure must be positive"
    if pres - pwf < MIN_DRAWDOWN_PSI:
        return f"reservoir pressure within {MIN_DRAWDOWN_PSI:.0f} psi of BHP"
    return None


def ipr_rate(ipr: dict[str, Any], pwf: float) -> float:
    """Total liquid (BLPD) at ``pwf`` on the Vogel curve through the anchor."""
    pres = float(ipr["pres"])
    qmax = float(ipr["qwf"]) / vogel_factor(float(ipr["pwf"]) / pres)
    return qmax * vogel_factor(float(pwf) / pres)


def ipr_pi(ipr: dict[str, Any], pwf: float) -> float:
    """Local productivity index -dq/dpwf (BLPD/psi) at ``pwf``."""
    pres = float(ipr["pres"])
    qmax = float(ipr["qwf"]) / vogel_factor(float(ipr["pwf"]) / pres)
    r = min(max(float(pwf) / pres, 0.0), 1.0)
    return qmax * (0.2 + 1.6 * r) / pres


def fit_pseudo_pr(pwf: list[float], q: list[float], pr_hi: float) -> Optional[dict[str, Any]]:
    """Vogel pseudo reservoir pressure from a well's (BHP, liquid) test points.

    Wraps ``header_engine.fit_vogel_ipr``. Returns its dict plus ``usable``:
    at least four points, 100 psi of BHP spread, a pr off the cap, and an
    rmse under a quarter of the mean rate. An unusable fit is still reported
    so the page can say why it fell back.
    """
    from server.services.tools.header_engine import fit_vogel_ipr

    fit = fit_vogel_ipr(pwf, q, pr_hi=pr_hi)
    if fit is None:
        return None
    qmean = float(np.mean([v for v in q if v and v > 0])) if q else float("nan")
    reasons = []
    if fit["n"] < 4:
        reasons.append("fewer than 4 gauged tests")
    if fit["pwf_spread"] < 100.0:
        reasons.append(f"BHP spread {fit['pwf_spread']:.0f} psi < 100")
    if fit["pr_at_bound"]:
        reasons.append("pressure pinned at the reservoir cap")
    if math.isfinite(qmean) and qmean > 0 and fit["rmse"] > 0.25 * qmean:
        reasons.append("rate scatter > 25%")
    fit["usable"] = not reasons
    fit["why_not"] = "; ".join(reasons) or None
    return fit


# ── per-well response ────────────────────────────────────────────────────────


def nonjp_delta(
    d_header: float,
    whp_hdr: Optional[float],
    slope: float,
    ipr: dict[str, Any],
    bhp_now: float,
    wc: float,
) -> dict[str, float]:
    """Response of a non-JP well to a header change.

    Args:
        d_header: header change (psi).
        whp_hdr: dWHP/dHeader (None -> 1.0).
        slope: closed-loop dBHP/dWHP.
        ipr: usable Vogel anchor {qwf, pwf, pres}.
        bhp_now: current flowing BHP (psig) the change starts from.
        wc: water cut fraction applied to the liquid change.

    Returns:
        dict: d_whp, d_bhp (psi), liq_now, d_liq (BLPD), d_oil (BOPD), pi.
    """
    r = 1.0 if whp_hdr is None else float(whp_hdr)
    d_whp = r * float(d_header)
    d_bhp = float(slope) * d_whp
    liq_now = ipr_rate(ipr, bhp_now)
    d_liq = ipr_rate(ipr, bhp_now + d_bhp) - liq_now
    return {
        "d_whp": d_whp, "d_bhp": d_bhp, "liq_now": liq_now, "d_liq": d_liq,
        "d_oil": d_liq * (1.0 - float(wc)), "pi": ipr_pi(ipr, bhp_now),
    }


# ── observed events ──────────────────────────────────────────────────────────


def event_windows(event_time: str, pre_hours: int, post_hours: int, gap_hours: int) -> dict[str, pd.Timestamp]:
    """Pre and post windows around an event, excluding ``gap_hours`` each side.

    Hours, not days: a bring-online is often observed a day later, and the
    post window has to fit between the event and now. With event
    2026-09-28 12:00, gap 6, pre 72 and post 18: pre is Sep 25 06:00 to
    Sep 28 06:00 and post is Sep 28 18:00 to Sep 29 12:00. Times are the
    historian's local timestamps.
    """
    ev = pd.Timestamp(event_time)
    if ev.tzinfo is not None:
        ev = ev.tz_convert(None)
    pre_end = ev - pd.Timedelta(hours=gap_hours)
    post_start = ev + pd.Timedelta(hours=gap_hours)
    return {
        "pre_start": pre_end - pd.Timedelta(hours=pre_hours),
        "pre_end": pre_end,
        "post_start": post_start,
        "post_end": post_start + pd.Timedelta(hours=post_hours),
    }


def window_delta(series: Optional[pd.Series], win: dict[str, pd.Timestamp]) -> dict[str, Any]:
    """Median-after-minus-median-before for a timestamp-indexed series.

    Medians, not means, so a half-day upset inside a window does not move the
    answer. Returns pre, post, delta and point counts (None when a window is
    empty).
    """
    out: dict[str, Any] = {"pre": None, "post": None, "delta": None, "n_pre": 0, "n_post": 0}
    if series is None or len(series) == 0:
        return out
    s = pd.to_numeric(series, errors="coerce").dropna()
    idx = pd.DatetimeIndex(s.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    s.index = idx
    pre = s[(s.index >= win["pre_start"]) & (s.index < win["pre_end"])]
    post = s[(s.index >= win["post_start"]) & (s.index < win["post_end"])]
    out["n_pre"], out["n_post"] = int(len(pre)), int(len(post))
    if len(pre):
        out["pre"] = float(pre.median())
    if len(post):
        out["post"] = float(post.median())
    if len(pre) and len(post):
        out["delta"] = out["post"] - out["pre"]
    return out


# ── run status ───────────────────────────────────────────────────────────────

# Firm = rests on the well's own data or on an engineer's saved review
# (user decision 2026-09-29), so "conditional" means "some wells nobody has
# reviewed yet" and clears as wells are saved.
#   relation: the pump model, the well's own measured slope, or ANY saved relation
#   IPR:      the jet pump's Solver IPR, ANY saved IPR, a USABLE fit of the
#             well's own gauged tests, or the well's own SAVED ResP
# Unsaved correlations, default ResP (1,800 / 3,000), flagged gauge fits and
# unsaved manual values stay conditional.
FIRM_RELATIONS = {"physics", "measured"}


def relation_firm(row: dict[str, Any]) -> bool:
    return row.get("relation_source") in FIRM_RELATIONS or bool(row.get("relation_saved"))


def ipr_firm(row: dict[str, Any]) -> bool:
    if row.get("ipr_source") == "jp" or row.get("ipr_saved"):
        return True
    if row.get("ipr_source") == "fit":
        return bool(row.get("ipr_fit_usable"))
    return row.get("ipr_source") == "correlation" and row.get("pres_basis") == "saved"


def is_firm(row: dict[str, Any]) -> bool:
    return relation_firm(row) and ipr_firm(row)


def run_status(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Coverage status of a run from its per-well rows.

    ``complete``: every online well is modeled and firm (its own data or a
    saved review, see :func:`is_firm`).
    ``conditional``: every online well modeled, some not yet reviewed
    (borrowed correlation, default ResP, unsaved fit or manual value).
    ``incomplete``: at least one online well has no estimate.
    """
    online = [r for r in rows if r.get("online")]
    missing = [r["well"] for r in online if r.get("outcome") != "modeled"]
    soft = [r["well"] for r in online if r.get("outcome") == "modeled" and not is_firm(r)]
    if missing:
        status = "incomplete"
    elif soft:
        status = "conditional"
    else:
        status = "complete"
    return {"status": status, "missing": missing, "soft": soft, "online": len(online)}


def spread(values: list[float]) -> Optional[dict[str, float]]:
    """Median and interquartile range of a sample, or None when empty."""
    v = [x for x in (_finite(x) for x in values) if x is not None]
    if not v:
        return None
    a = np.asarray(v, dtype=float)
    return {"med": float(np.median(a)), "q25": float(np.quantile(a, 0.25)),
            "q75": float(np.quantile(a, 0.75)), "n": len(v)}


def gauge_problem(bhp_now: Optional[float], whp_now: Optional[float], reservoir: Any,
                  recent: Optional[pd.Series] = None) -> Optional[str]:
    """Why a BHP gauge reading is not a flowing BHP, or None when plausible.

    Catches the MPL-20 case (the "BHP" tag reads power-fluid pressure at the
    pump entry, far above any reservoir pressure) and flat-lined gauges. A
    pump intake BELOW wellhead pressure is normal for a pumped well (the
    pump supplies the head), so it is not a fault.
    """
    if bhp_now is None:
        return None
    limit = GAUGE_MAX.get(reservoir_key(reservoir), GAUGE_MAX_FALLBACK)
    if bhp_now > limit:
        return f"reads {bhp_now:.0f} psi, above any {reservoir or 'reservoir'} pressure - not a flowing BHP"
    if recent is not None:
        s = pd.to_numeric(recent, errors="coerce").dropna()
        if len(s) >= 24 and float(s.max() - s.min()) < 1.0:
            return "flat-lined (under 1 psi of movement in 72 h)"
    return None


def range_delta(d_header: float, whp_hdr: Optional[float], ipr: dict[str, Any], bhp_now: float, wc: float,
                slope_lo: float, slope_hi: float, pres_lo: float, pres_hi: float) -> tuple[float, float]:
    """(smaller, larger) oil-change magnitude bounds from slope and ResP ranges.

    Lower ResP through the same anchor means a steeper IPR, so the largest
    loss pairs the high slope with the low ResP. Both ends keep the 300 psi
    drawdown rule.
    """
    floor = float(ipr["pwf"]) + MIN_DRAWDOWN_PSI
    a = nonjp_delta(d_header, whp_hdr, slope_lo, {**ipr, "pres": max(pres_hi, floor)}, bhp_now, wc)["d_oil"]
    b = nonjp_delta(d_header, whp_hdr, slope_hi, {**ipr, "pres": max(pres_lo, floor)}, bhp_now, wc)["d_oil"]
    return (a, b) if abs(a) <= abs(b) else (b, a)
