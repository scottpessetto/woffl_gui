"""Offline event holdout validation. Inputs are snapshots; this module does no I/O."""
from copy import deepcopy
from dataclasses import asdict
from datetime import date, timedelta
from statistics import median
import math

from woffl.flow.hydraulics import physics_model


def split_events(points, *, step_psi=100., gap_days=7, embargo_days=3):
    """Hold out the latest whole pressure event with at least three dates.

    No random-day split. Embargo training near the boundary and exclude daily
    records borrowing a rate anchor from the held-out period. Events depend
    only on dates and PF pressure, never on prediction errors or BHP.
    """
    ordered = sorted((deepcopy(p) for p in points), key=lambda p: (p["date"], p["kind"]))
    by_day = {}
    for p in ordered:
        by_day.setdefault(p["date"], []).append(float(p["ppf"]))
    blocks = []
    previous = None
    for stamp, pressures in sorted(by_day.items()):
        day, pressure = date.fromisoformat(stamp), median(pressures)
        if previous is None or (day-previous[0]).days > gap_days or abs(pressure-previous[1]) >= step_psi:
            blocks.append([])
        blocks[-1].append(stamp)
        previous = day, pressure
    eligible = [b for b in blocks[1:] if len(b) >= 3]
    if not eligible:
        return [], [], "no separate pressure event with at least three observed days"
    cutoff = eligible[-1][0]
    end_train = (date.fromisoformat(cutoff)-timedelta(days=embargo_days)).isoformat()
    train = [p for p in ordered if p["date"] < end_train and
             (p["kind"] == "test" or (p.get("anchor_date") and p["anchor_date"] < end_train))]
    held = [p for p in ordered if p["date"] >= cutoff]
    if len(train) < 10 or max(p["ppf"] for p in train)-min(p["ppf"] for p in train) < 200:
        return train, held, "training history needs 10 usable points and 200 psi pressure spread"
    return train, held, None


def training_config(config, train):
    """Freeze a Vogel IPR from actual training tests, never held-out labels."""
    cfg = deepcopy(config)
    tests = [p for p in train if p["kind"] == "test" and p.get("qtot") and p.get("wc") is not None
             and 0 < p["bhp"] < cfg.res_pres and 0 <= p["wc"] < .99]
    if not tests:
        raise ValueError("no actual training-period well test for an independent IPR")
    qmax = []
    for p in tests:
        x = p["bhp"]/cfg.res_pres
        oil = p["oil"] if p.get("oil") is not None else p["qtot"]*(1-p["wc"])
        qmax.append(oil/(1-.2*x-.8*x*x))
    cfg.form_wc = median(p["wc"] for p in tests)
    cfg.form_gor = median(p["fgor"] for p in tests if p.get("fgor") is not None) if any(p.get("fgor") is not None for p in tests) else cfg.form_gor
    cfg.pwf = .5*cfg.res_pres
    cfg.qwf = .7*median(qmax)/(1-cfg.form_wc)
    cfg.ken_well, cfg.kth_well, cfg.kdi_well = .03, .3, .4
    cfg.fnz_well, cfg.mach_crit_well = 1., 1.
    return cfg


def _rms(values):
    return math.sqrt(sum(v*v for v in values)/len(values)) if values else None


def evaluate_events(config, train, held, *, fit_function=None, predict_function=None, progress=None):
    """Fit training data, then predict held-out BHP/PF/oil with a fixed IPR.

    Only actual tests contribute oil errors. Older snapshots contain daily oil
    inferred from BHP; those values are ignored. The current fitter holds one
    training-derived oil IPR fixed, so a rerun of an old snapshot is a new
    experiment, not reproduction of the former per-point-IPR calibration.
    Every failed held-out solve stays in the report.
    """
    from woffl.gui.fric_calibration import calibrate_multipoint
    from woffl.assembly.network_optimizer import NetworkOptimizer
    from woffl.assembly.solopump import jetpump_solver
    from woffl.geometry import JetPump
    cfg = training_config(config, train)
    nozzle, throat = cfg.installed_nozzle, cfg.installed_throat
    if not nozzle or not throat:
        raise ValueError("no installed pump identity")
    # Cap compute using training PF coverage, without consulting held-out data.
    ranked = sorted(train, key=lambda p: p["ppf"])
    fit_points = ranked if len(ranked) <= 20 else [ranked[round(i*(len(ranked)-1)/19)] for i in range(20)]
    fit = (fit_function or calibrate_multipoint)(cfg, nozzle, throat, fit_points,
                                               seed=(.03, .3, .4, 1., 1.), progress=progress)
    report = dict(physics_model=physics_model(cfg.hydraulics_model), hydraulics_model=cfg.hydraulics_model,
                  calibration_contract="fixed-oil-ipr-v1",
                  well=cfg.well_name, train_points=len(train),
                  fit_points=len(fit_points), held_points=len(held), train_end=max(p["date"] for p in train),
                  held_start=min(p["date"] for p in held), refusal=fit.refusal,
                  training_rms_bhp_psi=fit.rms_bhp_psi if math.isfinite(fit.rms_bhp_psi) else None,
                  train_test_count=sum(p["kind"] == "test" for p in train),
                  field_validated=False)
    if fit.refusal:
        return report
    cfg.ken_well, cfg.kth_well, cfg.kdi_well = fit.best_ken, fit.best_kth, fit.best_kdi
    cfg.fnz_well = fit.best_fnz
    # Keep the exact frozen model and fit observations for offline replay.
    report.update(prediction_config=asdict(cfg), fit_observations=deepcopy(fit_points))
    held_wc = [p["wc"] for p in held if p["kind"] == "test" and p.get("wc") is not None]
    report.update(training_watercut=cfg.form_wc,
                  held_test_median_watercut=median(held_wc) if held_wc else None)
    wellbore, profile, inflow, mixture, pf = NetworkOptimizer._create_well_objects(cfg)
    pump = JetPump(nozzle, throat, ken=fit.best_ken, kth=fit.best_kth, kdi=fit.best_kdi)
    pump.dnz *= math.sqrt(fit.best_fnz)
    def predict(point):
        return jetpump_solver(point["pwh"], cfg.form_temp, point["ppf"], pump,
                              wellbore, profile, inflow, mixture, pf, cfg.jpump_direction,
                              hydraulics_model=cfg.hydraulics_model)
    rows = []
    for point in held:
        row = dict(date=point["date"], kind=point["kind"], ppf=point["ppf"], observed_bhp=point["bhp"], observed_pf=point["pf_rate"])
        try:
            # Prediction interface deliberately receives no held-out outcomes.
            inputs = {k: point[k] for k in ("date", "ppf", "pwh")}
            psu, sonic, oil, _fw, qpf, _mach = (predict_function or predict)(inputs)
            if not all(math.isfinite(value) for value in (psu, oil, qpf)):
                raise ValueError("nonfinite held-out prediction")
            row.update(predicted_bhp=psu, predicted_pf=qpf, predicted_oil=oil, sonic=bool(sonic),
                       bhp_error=psu-point["bhp"], pf_error_pct=100*(qpf/point["pf_rate"]-1))
            if point["kind"] == "test" and point.get("oil") is not None:
                row.update(observed_oil=point["oil"], oil_error_bopd=oil-point["oil"])
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    solved = [r for r in rows if "error" not in r]
    pairs = [((b["predicted_bhp"]-a["predicted_bhp"])-(b["observed_bhp"]-a["observed_bhp"]))
             for i, a in enumerate(solved) for b in solved[i+1:] if abs(b["ppf"]-a["ppf"]) >= 100]
    report.update(rows=rows, failed_solves=len(rows)-len(solved), railed=fit.railed,
                  held_rms_bhp_psi=_rms([r["bhp_error"] for r in solved]),
                  held_rms_pf_pct=_rms([r["pf_error_pct"] for r in solved]),
                  held_rms_oil_bopd=_rms([r["oil_error_bopd"] for r in solved if "oil_error_bopd" in r]),
                  held_oil_tests=sum("oil_error_bopd" in r for r in solved),
                  held_rms_dbhp_psi=_rms(pairs), held_response_pairs=len(pairs),
                  limitation="Retrospective event holdout under supplied geometry/reservoir pressure; engineering review required.")
    return report
