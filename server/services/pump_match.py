"""Read-only saved-model and chronological production-history comparisons.

Pump losses stay at clean reference. This is the app counterpart to the
chronological benchmark, not a shared multi-installation pump-loss fit.
Historical outcomes never enter a held-out prediction's solver inputs. The
saved-model replay is retrospective and does not claim held-out validation.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import math
import pickle
from pathlib import Path
from statistics import mean, median

import pandas as pd

from server import jobs, pool, schemas, surface_cache
from server.services import datasources, tests, wells
from server.services.field_validation import training_config
from server.services.optimizer_runs import _config_from_seeds
from woffl.flow.hydraulics import physics_model

KINDS = ("pump-match",)
MAX_TESTS = 1000
CHUNK_SIZE = 8
EMBARGO_DAYS = 3


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def stamp(value):
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    return parsed if pd.notna(parsed) else None


def code(value):
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text or None


def scores(rows):
    attempted = [r for r in rows if r["status"] in {"replay", "fit", "prediction", "failed"}]
    solved = [r for r in attempted if r["status"] != "failed"]
    bhp = [r["predicted_bhp"] - r["bhp"] for r in solved if r["bhp"] is not None and r["bhp"] > 0]
    oil = [r["predicted_oil"] - r["oil"] for r in solved if r["oil"] is not None and r["oil"] >= 0]
    oil_pct = [100 * abs(r["predicted_oil"] / r["oil"] - 1) for r in solved if r["oil"] is not None and r["oil"] > 0]
    pf_pct = [100 * abs(r["predicted_pf"] / r["pf"] - 1) for r in solved if r["pf"] is not None and 0 < r["pf"] <= 20000]
    return dict(attempted=len(attempted), solved=len(solved), failed=len(attempted)-len(solved),
                bhp_count=len(bhp), bhp_bias=mean(bhp) if bhp else None,
                bhp_rms=math.sqrt(mean(x*x for x in bhp)) if bhp else None,
                oil_count=len(oil), oil_mae=mean(abs(x) for x in oil) if oil else None,
                oil_median_abs_pct=median(oil_pct) if oil_pct else None,
                pf_count=len(pf_pct), pf_median_abs_pct=median(pf_pct) if pf_pct else None)


def assemble(config, tracker, test_frame, request, as_of):
    """Freeze installation assignments and split tests without outcome-based scoring cuts."""
    from woffl.geometry import JetPump

    end_day = pd.Timestamp(as_of).tz_convert("UTC").normalize()
    cutoff = end_day - pd.DateOffset(months=request.months)
    # Reuse the warmed history when a shorter chart window is requested.
    # Moving the left chart edge must not discard an otherwise usable fit.
    training_cutoff = end_day - pd.DateOffset(months=max(24, request.months))
    installs, undated = [], 0
    for raw in tracker.to_dict("records"):
        at = stamp(raw.get("Date Set"))
        if at is None:
            undated += 1
        elif at <= end_day + pd.Timedelta(days=1):
            installs.append((at, raw))
    installs.sort(key=lambda pair: pair[0])
    duplicates = Counter(at for at, _ in installs)
    starts = [at for at, _ in installs]
    change_days = {at.normalize() for at in starts}
    eras, specs = [], {}
    for i, (at, raw) in enumerate(installs):
        ident = f"{config.well_name}:{at.isoformat()}" + (f":duplicate:{i}" if duplicates[at] > 1 else "")
        nz, th, direction = code(raw.get("Nozzle Number")), code(raw.get("Throat Ratio")), code(raw.get("Circ Direction"))
        era = dict(installation_id=ident, date_set=at.isoformat(),
                   end=starts[i+1].isoformat() if i+1 < len(starts) else None,
                   pump=f"{nz or '?'}{th or '?'}", nozzle=nz, throat=th, direction=direction,
                   manufacturer=code(raw.get("Manufacturer")), flags=[], unavailable=None)
        if undated:
            era["unavailable"] = "Undated tracker installations prevent reliable historical assignment."
        elif duplicates[at] > 1:
            era["unavailable"] = "Ambiguous installation timestamp."
        elif direction not in {"forward", "reverse"}:
            era["unavailable"] = "Unknown historical circulation direction."
        try:
            pump = JetPump(nz, th)
            for col, expected in (("Nozzle Diameter", pump.dnz), ("Throat Diameter", pump.dth)):
                actual = number(raw.get(col))
                if actual is not None and abs(actual-expected) > .00011:
                    era["flags"].append(f"Nominal {col.lower()} conflicts with the modeled catalog label.")
        except (ValueError, TypeError, KeyError):
            era["unavailable"] = "Unsupported historical nozzle/throat."
        if code(raw.get("Pump Converted")) in {"True", "true", "1"}:
            era["flags"].append("Tracker pump converted to its National equivalent.")
        eras.append(era)
        specs[ident] = dict(tubing_od=number(raw.get("Tubing Diameter")))

    observations, missing_dates = [], 0
    for raw in test_frame.to_dict("records"):
        at = stamp(raw.get("WtDate"))
        if at is None:
            missing_dates += 1
            continue
        day = at.normalize()
        if not training_cutoff <= day <= end_day:
            continue
        idx = bisect_right(starts, day) - 1
        era = eras[idx] if idx >= 0 else None
        row = dict(date=day.isoformat(), wt_uid=code(raw.get("wt_uid")) or f"missing-id:{at.isoformat()}",
                   installation_id=era["installation_id"] if era else None,
                   status="missing", phase=None, message=None,
                   bhp=number(raw.get("BHP")), oil=number(raw.get("WtOilVol")),
                   pf=number(raw.get("lift_wat")), liquid=number(raw.get("WtTotalFluid")),
                   ppf=number(raw.get("pf_press")), pwh=number(raw.get("whp")),
                   wc=number(raw.get("form_wc")), gor=number(raw.get("fgor")),
                   pf_source=code(raw.get("pf_source")))
        if day in change_days:
            row.update(status="excluded", installation_id=None,
                       message="Installation day: daily pressures cannot identify which pump was operating.")
        elif era is None:
            row.update(status="excluded", message="No dated installation covers this test.")
        elif era["unavailable"]:
            row["message"] = era["unavailable"]
        elif not (row["ppf"] is not None and 800 <= row["ppf"] <= 5500 and row["pwh"] is not None and 10 <= row["pwh"] <= 600):
            row["message"] = "Missing or out-of-range test-day PF/WHP pressure."
        elif {"annulus": "reverse", "tubing": "forward"}.get(row["pf_source"]) != era["direction"]:
            row["message"] = "Test-day PF source and historical circulation disagree."
        observations.append(row)
    observations.sort(key=lambda r: (r["date"], r["wt_uid"]))
    repeated = Counter(r["wt_uid"] for r in observations)
    for row in observations:
        if repeated[row["wt_uid"]] > 1 or row["wt_uid"].startswith("missing-id:"):
            row.update(status="excluded", message="Missing or duplicate test identity.")
    rows = [r for r in observations if pd.Timestamp(r["date"]) >= cutoff]
    if len(rows) > MAX_TESTS:
        raise ValueError(f"History has {len(rows)} tests; choose a shorter window (limit {MAX_TESTS}).")

    def trainable(row):
        return (row["message"] is None and row["status"] != "excluded" and
                row["oil"] is not None and row["oil"] > 0 and row["bhp"] is not None and
                50 < row["bhp"] < config.res_pres - 10 and row["wc"] is not None and
                0 <= row["wc"] < .99 and row["gor"] is not None and row["gor"] >= 0)

    # Split from immutable raw candidates. A period lacking its own previous
    # pump can still train the following installation; result labels must not
    # change the eligibility of that raw training data.
    hardware_errors = {e["installation_id"]: e["unavailable"] for e in eras}
    training_rows = {e["installation_id"]: [deepcopy(r) for r in observations
                     if r["installation_id"] == e["installation_id"] and trainable(r)] for e in eras}

    def installation_config(base, era):
        cfg = deepcopy(base)
        cfg.installed_nozzle, cfg.installed_throat = era["nozzle"], era["throat"]
        cfg.jpump_direction = era["direction"]
        cfg.ken_well, cfg.kth_well, cfg.kdi_well, cfg.fnz_well = .03, .3, .4, 1.
        if specs[era["installation_id"]]["tubing_od"] is not None:
            cfg.tubing_od = specs[era["installation_id"]]["tubing_od"]
        return cfg

    work = []
    for i, era in enumerate(eras):
        own = [r for r in rows if r["installation_id"] == era["installation_id"]]
        if not own or era["unavailable"]:
            continue
        if request.mode == "all_tests":
            # Apply ONE saved well fit to every usable historical test. No
            # minimum test count, per-test inflow re-anchor or training embargo.
            cfg = installation_config(config, era)
            era.update(prediction_config=deepcopy(vars(cfg)))
            # WellConfig stores TOTAL liquid, while InFlow takes oil. Preserve
            # the saved oil IPR when using a test's water cut: changing WC alone
            # would silently re-anchor the oil curve. Never use measured test
            # oil or BHP here. Only the accompanying water/gas mixture varies.
            oil_anchor = cfg.qwf * (1 - cfg.form_wc)
            valid_anchor = math.isfinite(oil_anchor) and oil_anchor > 0 and 0 <= cfg.form_wc < 1
            for row in own:
                if row["message"] is None and row["status"] != "excluded":
                    if not valid_anchor:
                        row["message"] = "Saved inputs do not define a positive oil IPR."
                    elif row["wc"] is None or not 0 <= row["wc"] < 1:
                        row["message"] = "Missing or invalid measured test WC; oil replay requires 0% to below 100%."
                    elif row["gor"] is None or row["gor"] < 0:
                        row["message"] = "Missing or invalid measured test GOR; a nonnegative value is required."
                    else:
                        at = deepcopy(cfg)
                        at.form_wc, at.form_gor = row["wc"], row["gor"]
                        at.qwf = oil_anchor / (1 - at.form_wc)
                        row.update(phase="replay", input_wc=at.form_wc, input_gor=at.form_gor)
                        work.append((at, {"date": row["date"], "ppf": row["ppf"], "pwh": row["pwh"]}, rows.index(row)))
            continue
        previous = eras[i-1] if i else None
        training_era = era if request.mode == "same_pump" else previous
        if training_era is None or hardware_errors[training_era["installation_id"]]:
            era["unavailable"] = "No usable immediately preceding installation for training."
        else:
            candidates = training_rows[training_era["installation_id"]]
            if request.mode == "previous_pump":
                boundary = pd.Timestamp(era["date_set"]).normalize() - pd.Timedelta(days=EMBARGO_DAYS)
                candidates = [r for r in candidates if pd.Timestamp(r["date"]) < boundary][-request.training_tests:]
            else:
                candidates = candidates[:request.training_tests]
            era.update(training_installation_id=training_era["installation_id"],
                       training_count=len(candidates), training_test_ids=[r["wt_uid"] for r in candidates],
                       training_start=candidates[0]["date"] if candidates else None,
                       training_end=candidates[-1]["date"] if candidates else None)
            dates = len({r["date"] for r in candidates})
            if dates < 3:
                origin = "Immediately preceding pump" if request.mode == "previous_pump" else "This pump"
                era["unavailable"] = (
                    f"{origin} {training_era['pump']} (set {training_era['date_set'][:10]}) has "
                    f"{dates} usable training dates; at least 3 with measured oil, BHP and composition are required."
                )
            else:
                training = [dict(r, kind="test", qtot=r["oil"]/(1-r["wc"]), fgor=r["gor"]) for r in candidates]
                cfg = installation_config(training_config(config, training), era)
                era.update(prediction_config=deepcopy(vars(cfg)),
                           training_ppf_span=max(r["ppf"] for r in training)-min(r["ppf"] for r in training),
                           input_wc=cfg.form_wc, input_gor=cfg.form_gor)
                train_ids = {r["wt_uid"] for r in candidates}
                held_start = pd.Timestamp(candidates[-1]["date"]) + pd.Timedelta(days=EMBARGO_DAYS)
                for row in own:
                    if row["message"] is not None or row["status"] == "excluded":
                        continue
                    if request.mode == "same_pump" and row["wt_uid"] in train_ids:
                        phase = "fit"
                    elif pd.Timestamp(row["date"]) <= held_start:
                        row.update(status="excluded", message="Before the forecast cutoff or within the three-day training embargo.")
                        continue
                    else:
                        phase = "prediction"
                    row.update(phase=phase, input_wc=cfg.form_wc, input_gor=cfg.form_gor)
                    # No observed oil/BHP/PF/composition reaches a prediction task.
                    work.append((cfg, {"date": row["date"], "ppf": row["ppf"], "pwh": row["pwh"]}, rows.index(row)))
        if era["unavailable"]:
            for row in own:
                if row["message"] is None:
                    row["message"] = era["unavailable"]
    notes = [
        ("One saved oil-rate-versus-BHP IPR is held fixed across all test points and pumps. Each test's measured WC/GOR supplies the water/gas mixture. This is a retrospective comparison using known composition, not held-out prediction validation."
         if request.mode == "all_tests" else
         "Inflow is fitted to earlier tests; WC/GOR are frozen from those tests."),
        "Pump losses remain clean reference, not a multi-pump loss calibration. Fitted wear is never transferred across installations.",
        "Uses saved/current geometry, PVT and reservoir-pressure priors; their historical values and signed gauge offsets are unverified.",
        "Predictions use test-day PF pressure and WHP. Each test's oil, BHP and PF rate are comparisons, not per-test solver anchors.",
        "Installation days excluded. Oil is scored only against actual tests. PF above 20,000 BPD is displayed but excluded from percentage scores.",
    ]
    if request.mode != "all_tests":
        notes.append(f"Three-day training embargo. Training can use recorded tests back to {training_cutoff.date()}, including before the displayed {request.months}-month window. Missing tests and unsupported installations remain gaps.")
    if missing_dates:
        notes.append(f"{missing_dates} tests have no usable date and cannot be plotted.")
    return eras, rows, work, notes


def predict_chunk(tasks):
    """Pure worker task; fresh well/PVT objects for every observation."""
    from woffl.assembly.network_optimizer import NetworkOptimizer
    from woffl.assembly.solopump import jetpump_solver
    from woffl.geometry import JetPump

    out = []
    for cfg, controls, index in tasks:
        try:
            at = deepcopy(cfg)
            bore, profile, inflow, mixture, pf = NetworkOptimizer._create_well_objects(at)
            pump = JetPump(at.installed_nozzle, at.installed_throat, ken=.03, kth=.3, kdi=.4)
            bhp, sonic, oil, fwat, qpf, _mach = jetpump_solver(
                controls["pwh"], at.form_temp, controls["ppf"], pump, bore, profile,
                inflow, mixture, pf, at.jpump_direction, hydraulics_model=at.hydraulics_model)
            if not all(math.isfinite(float(v)) for v in (bhp, oil, fwat, qpf)):
                raise ValueError("Non-finite prediction")
            out.append((index, dict(predicted_bhp=float(bhp), predicted_oil=float(oil),
                                    predicted_pf=float(qpf), predicted_liquid=float(oil+fwat), sonic=bool(sonic))))
        except Exception as exc:
            out.append((index, dict(message=f"{type(exc).__name__}: {exc}")))
    return out


def run(job, well, request):
    from concurrent.futures.process import BrokenProcessPool

    jobs.check_cancelled(job)
    job["progress"] = "loading saved well inputs and historical tests"
    context = wells.well_context(well, 6, 0)
    seeds = dict(context["seeds"])
    if context.get("jpump_md") is not None:
        seeds["jpump_md"] = context["jpump_md"]
    # Well-fit replay retains the saved oil IPR, using measured test WC/GOR.
    # Chronological validation derives inputs only from training. Neither carries
    # fitted pump losses or wear into another installation.
    if request.mode != "all_tests":
        seeds.update(form_wc=.5, form_gor=250, qwf=750, pwf=500)
    if request.edited_inputs is not None:
        aliases = {"qwf_liq": "qwf", "res_pres": "pres"}
        seeds.update({aliases.get(k, k): v for k, v in request.edited_inputs.model_dump(exclude_none=True).items()})
    seeds.update(ken=.03, kth=.3, kdi=.4, nozzle_area_factor=1., mach_crit=1.,
                 hydraulics_model=request.hydraulics_model)
    cfg = _config_from_seeds(well, "", seeds)
    tracker, source = datasources.jp_history()
    tracker = tracker[tracker["Well Name"] == well].copy()
    if tracker.empty:
        raise ValueError("No dated pump history for this well.")
    fleet_tests = tests.fetch_all_well_tests(max(24, request.months))
    if fleet_tests is None or fleet_tests.empty:
        raise ValueError("No recorded well tests are available in this history window.")
    test_frame = fleet_tests[fleet_tests["well"] == well].copy()
    as_of = datetime.now(timezone.utc).isoformat()
    jobs.check_cancelled(job)
    eras, rows, work, notes = assemble(cfg, tracker, test_frame, request, as_of)
    if request.edited_inputs is not None:
        notes[0] = notes[0].replace("One saved oil-rate-versus-BHP IPR", "One edited oil-rate-versus-BHP IPR")
        notes.insert(0, "Preview of current well-input edits. Nothing is saved; optimization uses database inputs until Save well inputs succeeds.")
    if source != "databricks":
        notes.append("Installation history uses the Excel fallback; tracker identity is unverified.")
    replay_source = sha256(Path(__file__).read_bytes() + Path(__file__).with_name("field_validation.py").read_bytes()).hexdigest()
    snapshot_inputs = (replay_source, request.model_dump(), source, as_of[:10], eras, rows, [(vars(c), p, i) for c, p, i in work])
    key, cached = surface_cache.snapshot(cfg, "pump-match-v1", snapshot_inputs)
    if cached is not None:
        return cached
    for start in range(0, len(work), CHUNK_SIZE):
        jobs.check_cancelled(job)
        job["progress"] = f"predicting history: {start}/{len(work)} tests"
        chunk = work[start:start+CHUNK_SIZE]
        future = pool.submit(predict_chunk, chunk)
        values = None
        if future is not None:
            try:
                values = future.result()
            except BrokenProcessPool:
                pool.stop()
        if values is None:
            with pool.cpu_slot():
                jobs.check_cancelled(job)
                values = predict_chunk(chunk)
        jobs.check_cancelled(job)
        for index, predicted in values:
            rows[index].update(predicted)
            rows[index]["status"] = "failed" if predicted.get("message") else rows[index]["phase"]
    for era in eras:
        own = [r for r in rows if r["installation_id"] == era["installation_id"]]
        era["fit_scores"] = scores([r for r in own if r["phase"] == "fit"])
        era["prediction_scores"] = scores([r for r in own if r["phase"] == "prediction"])
        era["replay_scores"] = scores([r for r in own if r["phase"] == "replay"])
    result = schemas.PumpMatchResult(well=well, request=request, physics_model=physics_model(request.hydraulics_model),
        snapshot_id=(key or sha256(pickle.dumps(snapshot_inputs)).digest()).hex(), as_of=as_of,
        source=source, notes=notes, well_inputs=deepcopy(vars(cfg)), eras=eras, rows=rows).model_dump()
    jobs.check_cancelled(job)
    surface_cache.store_snapshot(key, result)
    return result


def start(well, request):
    return jobs.start(KINDS[0], lambda job: run(job, well, request))
