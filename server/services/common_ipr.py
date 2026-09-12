"""One explicit oil IPR candidate across history, with fixed reservoir pressure.

Only observed oil/BHP fit the common curve; no pump losses, measurements or
saved values are changed. Chronological holdouts are conditional on measured
BHP, so forward production replay is still required for optimization decisions.
"""

from bisect import bisect_right
from collections import Counter
from datetime import datetime, timezone
import math

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from server import schemas
from server.services import datasources, tests
from server.services.pump_match import code, number, stamp


def term(bhp, pres):
    ratio = np.asarray(bhp) / pres
    return 1 - .2 * ratio - .8 * ratio * ratio


def fit_frame(req, tracker, frame, as_of, source="fixture"):
    """Pure fitting boundary for frozen data; no warehouse or persistence calls."""
    p = req.params
    if p.model_as_water or not 0 <= p.form_wc < .99 or p.pwf >= p.pres:
        raise ValueError("A common oil IPR requires oil mode, WC below 99%, and anchor BHP below reservoir pressure.")
    end = pd.Timestamp(as_of).tz_convert("UTC").normalize()
    cutoff = end - pd.DateOffset(months=req.months)
    installs = [(stamp(r.get("Date Set")), r) for r in tracker.to_dict("records")]
    unknown_install = any(at is None for at, _ in installs)
    installs = sorted(((at, r) for at, r in installs if at is not None and at <= end + pd.Timedelta(days=1)), key=lambda r: r[0])
    starts = [at for at, _ in installs]
    duplicates = Counter(starts)
    change_days = {at.normalize() for at in starts}
    eras = [dict(id=at.isoformat(), date_set=at.isoformat(), pump=f"{code(r.get('Nozzle Number')) or '?'}{code(r.get('Throat Ratio')) or '?'}") for at, r in installs]
    rows = []
    missing_dates, future_dates = 0, 0
    for raw in frame.to_dict("records"):
        at = stamp(raw.get("WtDate"))
        if at is None:
            missing_dates += 1
            continue
        if at.normalize() > end:
            future_dates += 1
            continue
        if at.normalize() < cutoff:
            continue
        day = at.normalize()
        idx = bisect_right(starts, day) - 1
        era = eras[idx] if idx >= 0 else None
        ident = code(raw.get("wt_uid"))
        rows.append(dict(test_id=ident or f"missing:{at.isoformat()}", date=day.isoformat(),
            era_id=era["id"] if era else None, pump=era["pump"] if era else "Unknown",
            bhp=number(raw.get("BHP")), oil=number(raw.get("WtOilVol")), wc=number(raw.get("form_wc")),
            gor=number(raw.get("fgor")), liquid=number(raw.get("WtTotalFluid")),
            reason=None, phase=None, candidate_oil=None, baseline_oil=None))
    rows.sort(key=lambda r: (r["date"], r["test_id"]))
    if len(rows) > 1000:
        raise ValueError("More than 1,000 tests in this window; choose a shorter history.")
    # Freeze the chronological split on RAW dated observations before any
    # engineering selection, physical cuts or robust residual weighting.
    days = sorted({r["date"] for r in rows})
    split = days[min(len(days)-1, max(1, math.floor(len(days)*(1-req.holdout_fraction))))] if len(days) > 1 else None
    embargo = pd.Timestamp(split) - pd.Timedelta(days=3) if split else None
    ids = Counter(r["test_id"] for r in rows)
    for r in rows:
        day = pd.Timestamp(r["date"])
        r["phase"] = "holdout" if split and r["date"] >= split else "training"
        if r["phase"] == "training" and embargo is not None and day >= embargo:
            r["phase"] = "embargo"
            r["reason"] = "Three-day embargo before holdout."
        if r["test_id"].startswith("missing:") or ids[r["test_id"]] != 1:
            r["reason"] = "Missing or duplicate test identity."
        elif day in change_days:
            r["reason"] = "Installation day."
        elif unknown_install or r["era_id"] is None:
            r["reason"] = "Historical installation is unknown."
        elif duplicates[pd.Timestamp(r["era_id"])] > 1:
            r["reason"] = "Ambiguous installation timestamp."
        elif r["oil"] is None or r["oil"] <= 0 or r["bhp"] is None or not 50 < r["bhp"] < p.pres - 10:
            r["reason"] = "Requires positive flowing oil and BHP below reservoir pressure by more than 10 psi."
        elif r["wc"] is None or not 0 <= r["wc"] < .99 or r["gor"] is None or r["gor"] < 0:
            r["reason"] = "Missing or invalid measured WC/GOR."
        elif r["liquid"] is None or r["liquid"] <= 0 or r["oil"] > r["liquid"]:
            r["reason"] = "Invalid formation liquid/oil rates."
        if not r["reason"] and r["phase"] == "training" and (r["test_id"] in req.exclude_tests or r["era_id"] in req.exclude_eras):
            r["reason"] = "Excluded from training by user."
    train = [r for r in rows if r["phase"] == "training" and not r["reason"]]
    notes = [
        "One Vogel oil IPR is fitted across selected training dates/pumps. Reservoir pressure, anchor BHP, WC and GOR stay at the current inputs.",
        "Equal weight per training date; a robust loss limits outlier influence. No measured value is changed and nothing is saved.",
        "Holdout oil is evaluated at measured BHP, not predicted operating BHP. Use forward BHP/oil/PF replay and independent pressure-response checks before optimization.",
        "The split precedes exclusions and fitting. Holdout tests cannot be manually removed here; changing the setup after inspecting holdouts makes validation exploratory.",
        "Measured BHP is assumed to represent pump suction; verify its datum and test quality.",
    ]
    if source != "databricks":
        notes.append("Installation source is unverified; confirm tracker history before using this candidate.")
    if missing_dates or future_dates:
        notes.append(f"Skipped {missing_dates} tests with missing/invalid dates and {future_dates} future-dated tests; these cannot enter the chronological split.")
    qmax = None
    seeds = None
    if len({r["date"] for r in train}) < 3:
        notes.append("At least three usable training dates are required. Review exclusions or widen the history.")
    else:
        x = term([r["bhp"] for r in train], p.pres)
        y = np.array([r["oil"] for r in train])
        date_counts = Counter(r["date"] for r in train)
        weight = np.array([1/date_counts[r["date"]] for r in train])
        initial = float(np.sum(weight*x*y) / np.sum(weight*x*x))
        residual = initial*x-y
        day_residuals = np.array([np.median([residual[i] for i, r in enumerate(train) if r["date"] == day]) for day in date_counts])
        scale = max(1., 1.4826*float(np.median(np.abs(day_residuals-np.median(day_residuals)))))
        def derivative(q):
            residual = q*x-y
            return float(np.sum(weight*x*residual/np.sqrt(scale*scale+residual*residual)))
        lo, hi = float(np.min(y/x)), float(np.max(y/x))
        qmax = (lo+hi)/2 if math.isclose(lo, hi, rel_tol=1e-12) else float(brentq(derivative, lo, hi))
        qwf = qmax * float(term(p.pwf, p.pres)) / (1-p.form_wc)
        if 10 <= qwf <= 20000:
            seeds = {"qwf": qwf}
        else:
            notes.append("Fitted anchor lies outside the supported 10-20,000 BLPD input range; candidate cannot be applied.")
        if max(r["bhp"] for r in train) - min(r["bhp"] for r in train) < 100:
            notes.append("Training BHP spans less than 100 psi; the curve response is weakly tested. Reservoir pressure was held fixed.")
    baseline_qmax = p.qwf*(1-p.form_wc)/float(term(p.pwf, p.pres))
    for r in rows:
        if r["reason"] or r["phase"] == "embargo":
            continue
        r["baseline_oil"] = float(baseline_qmax*term(r["bhp"], p.pres))
        r["candidate_oil"] = float(qmax*term(r["bhp"], p.pres)) if qmax is not None else None
    def summary(phase):
        selected = [r for r in rows if r["phase"] == phase and not r["reason"]]
        def mae(key):
            errors = [abs(r[key]-r["oil"]) for r in selected if r[key] is not None]
            return float(np.mean(errors)) if errors else None
        return dict(tests=len(selected), dates=len({r["date"] for r in selected}),
                    installations=len({r["era_id"] for r in selected}),
                    candidate_mae=mae("candidate_oil"), baseline_mae=mae("baseline_oil"))
    return schemas.CommonOilIprResult(request=req, as_of=str(as_of), source=source,
        qmax_oil=qmax, seeds=seeds, training=summary("training"), holdout=summary("holdout"),
        split_date=split, rows=rows, eras=eras, notes=notes)


def run(req: schemas.CommonOilIprRequest):
    tracker, source = datasources.jp_history()
    tracker = tracker[tracker["Well Name"] == req.well].copy()
    frame = tests.fetch_all_well_tests(max(24, req.months))
    frame = frame[frame["well"] == req.well].copy()
    return fit_frame(req, tracker, frame, datetime.now(timezone.utc).isoformat(), source)
