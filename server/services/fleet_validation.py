"""Offline fleet actuality audit using a frozen, read-only source snapshot.

Current-config comparisons are retrospective reproductions, not independent
validation: saved or automatically fitted IPR may use the scored observations.
No observed BHP or oil enters the forward prediction interface.
"""
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from woffl.assembly.well_test_client import _normalize_well_name
from woffl.assembly.pf_pressure import resolve_pf_pressure
from woffl.flow.entry_energy import MODEL_VERSION
from woffl.flow.hydraulics import physics_model


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def dates(series):
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()


def saved_cutoff(context, saved):
    """Latest known IPR/loss save date; unknown for automatically seeded IPR."""
    if context.get("ipr_source") != "saved" or not saved or not saved.get("saved_at"):
        return None
    stamps = [saved["saved_at"], *(saved.get("friction_at") or {}).values()]
    usable = pd.to_datetime(pd.Series(stamps), errors="coerce", utc=True).dropna()
    return usable.max().date().isoformat() if len(usable) else None


def prepare(snapshot, lookback_days=90, fresh_days=14):
    """Select current-pump observations and retain every exclusion in coverage."""
    asof = pd.Timestamp(snapshot["captured_at"]).tz_localize(None).normalize()
    start = asof-pd.Timedelta(days=lookback_days)
    frames = snapshot["frames"]
    bhp = frames["bhp"].copy()
    bhp["well"] = bhp.well_name.map(_normalize_well_name)
    bhp["date"] = dates(bhp.tag_date)
    bhp["observed_bhp"] = pd.to_numeric(bhp.bhp_cln_value, errors="coerce")
    # A conflicting well-day is not a gauge observation that can be scored.
    duplicates = bhp.duplicated(["well", "date"], keep=False)
    conflict_counts = bhp[duplicates].groupby("well").size().to_dict()
    bhp = bhp[~duplicates]
    tests = frames["tests"].copy()
    tests["date"] = dates(tests.WtDate)
    pressure = frames["pressure"].copy()
    pressure["date"] = dates(pressure.sample_date)
    volume = frames["pf_volume"].copy()
    volume["date"] = dates(volume.pfdate)
    daily = bhp.merge(pressure[["well", "date", "tubing_prs", "inn_ann_prs", "btmhole_prs"]],
                     on=["well", "date"], how="left", validate="one_to_one")
    daily = daily.merge(volume[["well", "date", "pwr_fld_net"]],
                        on=["well", "date"], how="left", validate="one_to_one")
    result = []
    for item in snapshot["universe"]["wells"]:
        name = item["name"]
        cfg = snapshot["configs"].get(name)
        ctx = snapshot["contexts"].get(name) or {}
        wt = tests[(tests.well == name) & (tests.date >= start) & (tests.date < asof)].copy()
        gauge = bhp[(bhp.well == name) & (bhp.observed_bhp > 50) & (bhp.date < asof)]
        latest = gauge.date.max()
        cutoff = saved_cutoff(ctx, snapshot["saved"].get(name))
        record = dict(well=name, pad=item.get("pad"), config=deepcopy(cfg),
                      ipr_source=ctx.get("ipr_source") or "default", ipr_r2=ctx.get("ipr_r2"),
                      clamped=ctx.get("clamped", []), pump=ctx.get("pump"),
                      saved_cutoff=cutoff, recent_test_count=len(wt),
                      latest_gauge_date=latest.date().isoformat() if pd.notna(latest) else None,
                      fresh_gauge=bool(pd.notna(latest) and latest >= asof-pd.Timedelta(days=fresh_days)),
                      duplicate_gauge_rows=conflict_counts.get(name, 0),
                      observations=[], rejected_tests={}, rejected_daily={})
        result.append(record)
        if not record["fresh_gauge"]:
            record["exclusion"] = "no credible BHP gauge reading in last 14 days"
            continue
        if not cfg:
            record["exclusion"] = snapshot["errors"].get(name, "no model inputs")
            continue
        pump = ctx.get("pump") or {}
        if not pump.get("date_set") or not cfg.get("installed_nozzle") or not cfg.get("installed_throat"):
            record["exclusion"] = "no dated current pump identity"
            continue
        if pump.get("source") != "databricks":
            record["exclusion"] = "current pump comes from fallback data"
            continue
        installed = pd.Timestamp(pump["date_set"]).normalize()
        selected_tests = []
        rejected = Counter()
        for raw in wt.sort_values("date").to_dict("records"):
            if raw["date"] <= installed:
                rejected["before current pump or installation day"] += 1
                continue
            values = {key: number(raw.get(col)) for key, col in
                      (("observed_bhp", "BHP"), ("observed_oil", "WtOilVol"),
                       ("observed_pf", "lift_wat"), ("ppf", "pf_press"), ("pwh", "whp"),
                       ("wc", "form_wc"), ("fgor", "fgor"), ("qtot", "WtTotalFluid"))}
            if values["observed_bhp"] is None or values["observed_bhp"] <= 50:
                rejected["invalid BHP"] += 1
                continue
            if values["observed_oil"] is None or values["observed_oil"] <= 0:
                rejected["no positive measured oil"] += 1
                continue
            # Missing/zero allocated lift volume does not prevent a BHP/oil
            # comparison; the forward solve needs pressure, not measured PF rate.
            if values["observed_pf"] is not None and values["observed_pf"] <= 0:
                values["observed_pf"] = None
            if values["ppf"] is None or not 800 <= values["ppf"] <= 5500:
                rejected["no plausible test-day PF pressure"] += 1
                continue
            if values["pwh"] is None or not 10 <= values["pwh"] <= 600:
                rejected["no plausible measured test-day wellhead pressure"] += 1
                continue
            source = raw.get("pf_source")
            direction = "forward" if source == "tubing" else "reverse" if source == "annulus" else None
            if direction != cfg["jpump_direction"]:
                rejected["test circulation differs from current model"] += 1
                continue
            stamp = raw["date"].date().isoformat()
            selected_tests.append(dict(date=stamp, kind="test", wt_uid=str(raw.get("wt_uid")),
                                       after_saved=bool(cutoff and stamp > cutoff), **values))
        record["rejected_tests"] = dict(rejected)
        if not selected_tests:
            record["exclusion"] = "no usable BHP/oil/pressure test in current pump era within 90 days"
            continue
        record["observations"].extend(selected_tests)
        rejected = Counter()
        dd = daily[(daily.well == name) & (daily.date >= start) & (daily.date < asof)]
        record["daily_available"] = len(dd)
        for raw in dd.sort_values("date").to_dict("records"):
            if raw["date"] <= installed:
                rejected["before current pump or installation day"] += 1
                continue
            observed_bhp, pf = number(raw["observed_bhp"]), number(raw.get("pwr_fld_net"))
            ppf, source = resolve_pf_pressure(raw.get("tubing_prs"), raw.get("inn_ann_prs"))
            direction = "forward" if source == "tubing" else "reverse" if source == "annulus" else None
            pwh = number(raw.get("inn_ann_prs") if direction == "forward" else raw.get("tubing_prs"))
            reason = None
            if observed_bhp is None or observed_bhp <= 50:
                reason = "invalid BHP"
            elif pf is None or pf <= 500:
                reason = "no operating PF rate above 500 BPD"
            elif ppf is None or not 800 <= ppf <= 5500:
                reason = "no plausible PF pressure"
            elif pwh is None or not 10 <= pwh <= 600:
                reason = "no plausible measured production pressure"
            elif direction != cfg["jpump_direction"]:
                reason = "daily circulation differs from current model"
            if reason:
                rejected[reason] += 1
                continue
            stamp = raw["date"].date().isoformat()
            record["observations"].append(dict(date=stamp, kind="daily", pwh=pwh, ppf=ppf,
                observed_bhp=observed_bhp, observed_pf=pf,
                pressure_view_bhp=number(raw.get("btmhole_prs")),
                after_saved=bool(cutoff and stamp > cutoff)))
        record["rejected_daily"] = dict(rejected)
    return result


def predict_well(record, predict_function=None):
    """Replay every observation using unchanged well inputs and actual pressures."""
    from woffl.assembly.network_optimizer import WellConfig, NetworkOptimizer
    from woffl.assembly.solopump import jetpump_solver
    from woffl.geometry import JetPump
    result = deepcopy(record)
    if record.get("exclusion"):
        return result
    for row in result["observations"]:
        for key in list(row):
            if key.startswith("predicted_") or key in {
                "error", "bhp_error", "pf_error_pct", "oil_error_bopd", "oil_error_pct", "sonic"
            }:
                row.pop(key)
    cfg = WellConfig(**record["config"])
    result.update(physics_model=physics_model(cfg.hydraulics_model), hydraulics_model=cfg.hydraulics_model)
    if predict_function is None:
        try:
            wellbore, profile, inflow, mixture, power = NetworkOptimizer._create_well_objects(cfg)
            pump = JetPump(cfg.installed_nozzle, cfg.installed_throat,
                           ken=cfg.ken_well if cfg.ken_well is not None else .03,
                           kth=cfg.kth_well if cfg.kth_well is not None else .3,
                           kdi=cfg.kdi_well if cfg.kdi_well is not None else .4)
            pump.dnz *= math.sqrt(cfg.fnz_well if cfg.fnz_well is not None else 1.)
        except Exception as exc:
            for row in result["observations"]:
                row["error"] = f"{type(exc).__name__}: {exc}"
            return result
        def predict_function(inputs):
            return jetpump_solver(inputs["pwh"], cfg.form_temp, inputs["ppf"], pump,
                                  wellbore, profile, inflow, mixture, power, cfg.jpump_direction,
                                  hydraulics_model=cfg.hydraulics_model)
    memo = {}
    for row in result["observations"]:
        try:
            inputs = {key: row[key] for key in ("ppf", "pwh")}
            key = tuple(inputs.values())
            if key not in memo:
                memo[key] = predict_function(inputs)
            bhp, sonic, oil, _fw, pf, _mach = memo[key]
            if not all(math.isfinite(v) for v in (bhp, oil, pf)):
                raise ValueError("nonfinite forward prediction")
            row.update(predicted_bhp=float(bhp), predicted_oil=float(oil), predicted_pf=float(pf),
                       sonic=bool(sonic), bhp_error=float(bhp-row["observed_bhp"]))
            if row.get("observed_pf") is not None:
                row["pf_error_pct"] = float(100*(pf/row["observed_pf"]-1))
            if row["kind"] == "test":
                row.update(oil_error_bopd=float(oil-row["observed_oil"]),
                           oil_error_pct=float(100*(oil/row["observed_oil"]-1)))
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
    return result


def metrics(rows):
    good = [r for r in rows if "error" not in r and "bhp_error" in r]
    result = dict(observations=len(rows), solved=len(good), failed=len(rows)-len(good))
    for key in ("bhp_error", "pf_error_pct", "oil_error_bopd", "oil_error_pct"):
        values = np.asarray([r[key] for r in good if key in r], dtype=float)
        if len(values):
            result[key] = dict(n=len(values), bias=float(values.mean()),
                rms=float(np.sqrt(np.mean(values**2))), median_abs=float(np.median(abs(values))),
                p90_abs=float(np.quantile(abs(values), .9)))
    result["bhp_within_50psi"] = sum(abs(r["bhp_error"]) <= 50 for r in good)
    result["pf_within_10pct"] = sum(abs(r["pf_error_pct"]) <= 10 for r in good if "pf_error_pct" in r)
    result["oil_within_20pct"] = sum(abs(r["oil_error_pct"]) <= 20 for r in good if "oil_error_pct" in r)
    return result


def summarize(report):
    active = [w for w in report["wells"] if not w.get("exclusion")]
    tests = [r for w in active for r in w["observations"] if r["kind"] == "test"]
    daily = [r for w in active for r in w["observations"] if r["kind"] == "daily"]
    latest = []
    for well in active:
        wt = [r for r in well["observations"] if r["kind"] == "test"]
        last = max(wt, key=lambda r: (r["date"], r["wt_uid"]))
        well["latest_test"] = last
        latest.append(last)
        well["test_metrics"] = metrics(wt)
        well["daily_metrics"] = metrics([r for r in well["observations"] if r["kind"] == "daily"])
    report["coverage"] = dict(app_wells=len(report["wells"]),
        fresh_gauge_wells=sum(w["fresh_gauge"] for w in report["wells"]),
        fresh_gauge_with_recent_tests=sum(w["fresh_gauge"] and w["recent_test_count"] > 0 for w in report["wells"]),
        eligible_wells=len(active), pads=sorted({w["pad"] for w in active}),
        exclusions=dict(Counter(w["exclusion"] for w in report["wells"] if w.get("exclusion"))))
    report["metrics"] = dict(latest_test=metrics(latest), all_tests=metrics(tests), daily=metrics(daily),
        tests_after_saved=metrics([r for r in tests if r["after_saved"]]),
        daily_after_saved=metrics([r for r in daily if r["after_saved"]]))
    report["pads"] = {pad: metrics([w["latest_test"] for w in active if w["pad"] == pad])
                      for pad in report["coverage"]["pads"]}
    return report


def diagnose(report):
    """Label input drift and add explicitly conditional hydraulic comparisons.

    The secondary solve is anchored on measured BHP/oil and composition, so
    it is a diagnostic only and never included in the primary accuracy scores.
    """
    asof = pd.Timestamp(report["snapshot_time"]).tz_localize(None).normalize()
    for well in report["wells"]:
        if well.get("exclusion"):
            continue
        last, cfg = well["latest_test"], well["config"]
        flags = []
        well["test_age_days"] = int((asof-pd.Timestamp(last["date"])).days)
        if well["test_age_days"] > 30:
            flags.append("latest usable test is over 30 days old")
        wc = last.get("wc")
        if wc is not None:
            well["watercut_difference_points"] = 100*(cfg["form_wc"]-wc)
            if abs(cfg["form_wc"]-wc) > .10:
                flags.append("model and test watercut differ by over 10 percentage points")
        if well.get("saved_cutoff") and well["saved_cutoff"] < well["pump"]["date_set"]:
            flags.append("saved coefficients/IPR predate current pump")
        if last.get("observed_pf") is None:
            flags.append("test PF rate unavailable; BHP and oil still scored")
        elif last["observed_pf"] > 20000:
            flags.append("reported test PF exceeds 20000 BPD; retained in raw score")
        if "error" in last:
            flags.append("current-config forward solve failed")
        well["input_flags"] = flags
        if (wc is not None and 0 <= wc < .99 and last.get("qtot") and
                0 < last["observed_bhp"] < cfg["res_pres"]):
            conditional = deepcopy(well)
            conditional["observations"] = [deepcopy(last)]
            conditional["observations"][0].pop("error", None)
            conditional["config"].update(form_wc=wc,
                qwf=last["observed_oil"]/(1-wc), pwf=last["observed_bhp"])
            if last.get("fgor") is not None:
                conditional["config"]["form_gor"] = last["fgor"]
            checked = predict_well(conditional)["observations"][0]
            well["test_conditioned_diagnostic"] = {k:v for k,v in checked.items()
                if k.startswith("predicted_") or k in {"error", "bhp_error", "pf_error_pct", "sonic"}}
        day = [r for r in well["observations"] if r["kind"] == "daily" and "error" not in r]
        observed, modeled = [], []
        for i, a in enumerate(day):
            for b in day[i+1:]:
                span = (pd.Timestamp(b["date"])-pd.Timestamp(a["date"])).days
                dp = b["ppf"]-a["ppf"]
                if 3 <= span <= 30 and abs(dp) >= 100 and abs(b["pwh"]-a["pwh"]) <= 25:
                    observed.append(-(b["observed_bhp"]-a["observed_bhp"])/dp)
                    modeled.append(-(b["predicted_bhp"]-a["predicted_bhp"])/dp)
        well["response"] = dict(pairs=len(observed),
            observed_beta=float(np.median(observed)) if len(observed) >= 5 else None,
            model_beta=float(np.median(modeled)) if len(modeled) >= 5 else None)
    return report


def run_audit(snapshot, progress=print):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from woffl.assembly.parallelism import worker_ceiling
    records = prepare(snapshot)
    eligible = [r for r in records if not r.get("exclusion")]
    progress(f"Scoring {len(eligible)} wells and {sum(len(w['observations']) for w in eligible)} observations")
    workers = min(2, worker_ceiling())
    report = dict(model_version=MODEL_VERSION, snapshot_time=snapshot["captured_at"],
        read_only=True, refitted=False, field_validated=False, workers=workers,
        source_query_count=len(snapshot.get("query_stats", [])),
        source_rows={k:len(v) for k,v in snapshot["frames"].items()},
        scope="Current-config retrospective reproduction; no refit. After-save subsets remain conditional on current geometry and inputs.",
        wells=[r for r in records if r.get("exclusion")])
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(predict_well, r): r for r in eligible}
        for future in as_completed(pending):
            record = future.result()
            report["wells"].append(record)
            progress(f"{record['well']}: {len(record['observations'])} observations, "
                     f"{sum('error' in r for r in record['observations'])} failed solves")
    report["wells"].sort(key=lambda r: r["well"])
    return diagnose(summarize(report))


def write_report(report, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    rows, observations = [], []
    for w in report["wells"]:
        row = {k: w.get(k) for k in ("well", "pad", "ipr_source", "ipr_r2", "latest_gauge_date", "saved_cutoff", "exclusion", "test_age_days", "watercut_difference_points")}
        row["input_flags"] = "; ".join(w.get("input_flags", []))
        row["rejected_tests"] = "; ".join(f"{n}: {reason}" for reason,n in w["rejected_tests"].items())
        row["recent_test_count"] = w["recent_test_count"]
        row["pump_date"] = (w.get("pump") or {}).get("date_set")
        for key in ("installed_nozzle", "installed_throat", "form_wc", "form_gor", "res_pres", "jpump_direction"):
            row["model_"+key] = (w.get("config") or {}).get(key)
        row.update({k: v for k, v in w.get("latest_test", {}).items() if k not in {"kind", "after_saved"}})
        rows.append(row)
        observations.extend(dict(well=w["well"], pad=w["pad"], **r) for r in w["observations"])
    pd.DataFrame(rows).to_csv(output.with_suffix(".csv"), index=False)
    pd.DataFrame(observations).to_csv(output.with_name(output.name+"_observations.csv"), index=False)
    print(json.dumps(dict(coverage=report["coverage"], metrics=report["metrics"]), indent=2))
