"""Pure chronological cross-pump reference benchmark; no data access or saves.

This is a baseline for improving calibration, not an approved lifetime fit.
Geometry/PVT/reservoir-pressure priors may come from today's well context.
Only earlier tests estimate oil productivity; pump losses stay at the clean
reference. Unknown historical inputs and nominal-spec conflicts remain visible.
"""
from copy import deepcopy
from dataclasses import asdict
from datetime import date, timedelta
import math

from server.services.field_validation import training_config
from server.services.fleet_validation import metrics


def chronological_challenges(eras, *, min_tests=3, max_train_tests=10, embargo_days=3):
    """Adjacent installations only; choose splits by time and data coverage.

    Never pool across an intervening unobserved installation, choose dates
    by model errors, or reinterpret a repeat size as the same physical pump.
    """
    ordered = sorted(eras, key=lambda e: e["start"])
    challenges = []
    for earlier, later in zip(ordered, ordered[1:]):
        # The preflight may omit a tracker row with no usable geometry or
        # observations. Its set date still closed the previous interval.
        if earlier["end_exclusive"] != later["start"]:
            continue
        cutoff = (date.fromisoformat(later["start"])-timedelta(days=embargo_days)).isoformat()
        train = sorted((deepcopy(t) for t in earlier["tests"] if t["date"] < cutoff),
                       key=lambda t: (t["date"], t["wt_uid"]))[-max_train_tests:]
        held = sorted((deepcopy(t) for t in later["tests"] if t["date"] > later["start"]),
                      key=lambda t: (t["date"], t["wt_uid"]))
        if len({t["date"] for t in train}) < min_tests or len({t["date"] for t in held}) < min_tests:
            continue
        challenges.append(dict(training_installation=deepcopy(earlier),
                               prediction_installation=deepcopy(later), train=train, held=held))
    return challenges


def evaluate_challenge(config, challenge, *, predict_function=None):
    """Forecast a complete later installation with the earlier oil IPR frozen.

    Primary forecast also freezes WC/GOR from training. A second, conditional
    replay uses measured WC/GOR as operating inputs while preserving that oil
    IPR. Neither predictor receives held-out oil, BHP or measured PF rate.
    """
    from woffl.assembly.network_optimizer import NetworkOptimizer
    from woffl.assembly.solopump import jetpump_solver
    from woffl.flow.hydraulics import physics_model
    from woffl.geometry import JetPump

    earlier, later = challenge["training_installation"], challenge["prediction_installation"]
    train = [dict(p, kind="test", qtot=p["oil"]/(1-p["wc"]), fgor=p["gor"])
             for p in challenge["train"]]
    cfg = training_config(config, train)
    cfg.installed_nozzle, cfg.installed_throat = later["nozzle"], later["throat"]
    cfg.jpump_direction = later["direction"]
    if later.get("tubing_od"):
        cfg.tubing_od = later["tubing_od"]
    cfg.pump_calibration_scoped = True
    frozen_oil_anchor = cfg.qwf * (1-cfg.form_wc)

    def predict(inputs):
        at = deepcopy(cfg)
        at.form_wc, at.form_gor = inputs["wc"], inputs["gor"]
        at.qwf = frozen_oil_anchor / (1-at.form_wc)
        bore, profile, inflow, mixture, pf = NetworkOptimizer._create_well_objects(at)
        pump = JetPump(at.installed_nozzle, at.installed_throat, ken=.03, kth=.3, kdi=.4)
        return jetpump_solver(inputs["pwh"], at.form_temp, inputs["ppf"], pump,
                              bore, profile, inflow, mixture, pf, at.jpump_direction,
                              hydraulics_model=at.hydraulics_model)

    rows = []
    for observed in challenge["held"]:
        row = dict(date=observed["date"], wt_uid=observed["wt_uid"],
                   observed_bhp=observed["bhp"], observed_oil=observed["oil"], observed_pf=observed["pf"],
                   ppf=observed["ppf"], pwh=observed["pwh"], wc=observed["wc"], gor=observed["gor"], predictions={})
        for mode in ("frozen_composition", "measured_composition"):
            inputs = {k: observed[k] for k in ("date", "ppf", "pwh")}
            inputs.update(wc=cfg.form_wc if mode == "frozen_composition" else observed["wc"],
                          gor=cfg.form_gor if mode == "frozen_composition" else observed["gor"])
            try:
                bhp, sonic, oil, _fw, qpf, _mach = (predict_function or predict)(inputs)
                if not all(math.isfinite(float(v)) for v in (bhp, oil, qpf)):
                    raise ValueError("non-finite prediction")
                prediction = dict(predicted_bhp=bhp, predicted_oil=oil, predicted_pf=qpf, sonic=bool(sonic),
                                  bhp_error=bhp-observed["bhp"], oil_error_bopd=oil-observed["oil"],
                                  oil_error_pct=100*(oil/observed["oil"]-1))
                if observed["pf"] is not None and 0 < observed["pf"] <= 20000:
                    prediction["pf_error_pct"] = 100*(qpf/observed["pf"]-1)
                else:
                    prediction["pf_scoring_exclusion"] = "missing/nonpositive PF or allocation above 20000 BPD"
            except Exception as exc:
                prediction = {"error": f"{type(exc).__name__}: {exc}"}
            row["predictions"][mode] = prediction
        rows.append(row)
    return dict(well=cfg.well_name, physics_model=physics_model(cfg.hydraulics_model),
        hydraulics_model=cfg.hydraulics_model,
        training_pump=earlier["nozzle"]+earlier["throat"], training_start=earlier["start"],
        training_test_count=len(train), training_last_test=max(p["date"] for p in train),
        prediction_pump=later["nozzle"]+later["throat"], prediction_start=later["start"],
        prediction_end=later["end_exclusive"],
        different_size=(earlier["nozzle"], earlier["throat"]) != (later["nozzle"], later["throat"]),
        geometry_flags=earlier.get("geometry_flags", [])+later.get("geometry_flags", []),
        converted_pump=bool(earlier.get("converted") or later.get("converted")),
        prediction_config=asdict(cfg), train=deepcopy(challenge["train"]), rows=rows,
        metrics={mode: metrics([r["predictions"][mode] for r in rows])
                 for mode in ("frozen_composition", "measured_composition")},
        validated_for_sizing=False)
