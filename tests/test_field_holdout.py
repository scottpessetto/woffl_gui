from copy import deepcopy
from datetime import date, timedelta
from types import SimpleNamespace

import pytest

from server.services.field_validation import split_events, training_config, evaluate_events
from server.services.optimizer_runs import _config_from_seeds


def points():
    rows = []
    for i in range(35):
        stamp = (date(2026, 1, 1)+timedelta(days=i)).isoformat()
        rows.append(dict(date=stamp, kind="test" if i % 5 == 0 else "daily",
            anchor_date="2026-01-01" if i < 30 else "2026-01-31", ppf=3000+200*(i//10),
            bhp=500., pf_rate=2000., pwh=210., qtot=800., oil=160., wc=.8, fgor=600., weight=1.))
    return rows


def test_holdout_uses_whole_event_and_excludes_future_rate_anchors():
    rows = points()
    rows[20]["kind"] = "daily"
    rows[20]["anchor_date"] = "2026-02-01"
    train, held, refusal = split_events(rows)
    assert refusal is None
    assert {p["date"] for p in train}.isdisjoint(p["date"] for p in held)
    assert held[0]["date"] == "2026-01-31"
    assert max(p["date"] for p in train) < "2026-01-28"
    assert all(p["date"] != rows[20]["date"] for p in train)
    assert len(held) == 5


def test_held_out_labels_never_reach_fit_or_prediction():
    config = _config_from_seeds("Custom", "M", dict(nozzle_no="12", area_ratio="B", pres=1700))
    train, held, _ = split_events(points())
    before = deepcopy(config)
    seen = []
    def fit(cfg, nozzle, throat, fit_points, **kwargs):
        seen.append((cfg.qwf, deepcopy(fit_points)))
        return SimpleNamespace(refusal=None, best_ken=.03, best_kth=.3, best_kdi=.4,
                               best_fnz=1., rms_bhp_psi=0., railed=[])
    def predict(inputs):
        assert set(inputs) == {"date", "ppf", "pwh"}
        return 510., False, 150., 600., 2050., .5
    a = evaluate_events(config, train, held, fit_function=fit, predict_function=predict)
    changed = deepcopy(held)
    for p in changed:
        p.update(bhp=800., oil=10000., qtot=50000.)
    b = evaluate_events(config, train, changed, fit_function=fit, predict_function=predict)
    assert seen[0] == seen[1]
    assert a["held_rms_bhp_psi"] == 10
    assert b["held_rms_bhp_psi"] == 290
    assert a["held_oil_tests"] == 1  # derived daily oil never counts as a measurement
    assert a["prediction_config"]["qwf"] == seen[0][0]
    assert a["fit_observations"] == seen[0][1]
    assert config == before


def test_training_ipr_needs_an_actual_training_test():
    cfg = _config_from_seeds("Custom", "M", {})
    with pytest.raises(ValueError, match="actual training"):
        training_config(cfg, [p for p in points() if p["kind"] == "daily"])


@pytest.mark.parametrize("failure", ["exception", "nonfinite"])
def test_holdout_solve_failures_are_counted(failure):
    cfg = _config_from_seeds("Custom", "M", dict(nozzle_no="12", area_ratio="B"))
    train, held, _ = split_events(points())
    fit = lambda *a, **kw: SimpleNamespace(refusal=None, best_ken=.03, best_kth=.3,
        best_kdi=.4, best_fnz=1., rms_bhp_psi=0., railed=[])
    def predict(_):
        if failure == "nonfinite":
            return float("nan"), False, 150., 600., 2050., .5
        raise ValueError("cannot lift")
    report = evaluate_events(cfg, train, held, fit_function=fit, predict_function=predict)
    assert report["failed_solves"] == len(held)
    assert report["held_rms_bhp_psi"] is None
    assert not report["field_validated"]
