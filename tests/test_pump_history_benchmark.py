"""Historical pump-size forecasts must not anchor on the held-out outcomes."""
from copy import deepcopy

from server.services.pump_history_benchmark import chronological_challenges, evaluate_challenge
from woffl.assembly.network_optimizer import WellConfig


def _test(day, oil=100, bhp=500):
    return dict(date=day, wt_uid=day, oil=oil, bhp=bhp, pf=2000.,
                wc=.5, gor=250., ppf=3000., pwh=200.)


def _eras():
    return [dict(start="2026-01-01", end_exclusive="2026-02-01", nozzle="12", throat="B",
                 direction="reverse", tests=[_test(f"2026-01-{d:02d}") for d in (10, 17, 24, 31)]),
            dict(start="2026-02-01", end_exclusive="2026-03-01", nozzle="13", throat="C",
                 direction="reverse", tests=[_test(f"2026-02-{d:02d}") for d in (1, 8, 15, 22)])]


def test_split_excludes_changeout_day_and_training_embargo():
    challenge, = chronological_challenges(_eras())
    assert [r["date"] for r in challenge["train"]] == ["2026-01-10", "2026-01-17", "2026-01-24"]
    assert [r["date"] for r in challenge["held"]] == ["2026-02-08", "2026-02-15", "2026-02-22"]


def test_repeat_size_is_a_distinct_installation():
    eras = _eras()
    eras[1].update(nozzle="12", throat="B")
    assert len(chronological_challenges(eras)) == 1


def test_no_leap_over_an_unobserved_installation():
    eras = _eras()
    eras.insert(1, dict(eras[0], start="2026-01-29", tests=[]))
    assert chronological_challenges(eras) == []


def test_omitted_tracker_interval_still_blocks_a_cross_pump_challenge():
    eras = _eras()
    eras[0]["end_exclusive"] = "2026-01-29"
    assert chronological_challenges(eras) == []


def test_held_out_oil_bhp_pf_cannot_change_fit_or_prediction_inputs():
    cfg = WellConfig(well_name="W1", res_pres=1500, form_temp=80, jpump_tvd=4000)
    challenge, = chronological_challenges(_eras())
    calls = []
    def predict(inputs):
        calls.append(deepcopy(inputs))
        assert set(inputs) == {"date", "ppf", "pwh", "wc", "gor"}
        return 500., False, 100., 100., 2000., .1
    first = evaluate_challenge(cfg, challenge, predict_function=predict)
    original_calls = deepcopy(calls)
    for row in challenge["held"]:
        row.update(oil=5000, bhp=1000, pf=18000.)
    calls.clear()
    second = evaluate_challenge(cfg, challenge, predict_function=predict)
    assert first["prediction_config"] == second["prediction_config"]
    assert calls == original_calls
    assert first["rows"][0]["predictions"]["frozen_composition"]["bhp_error"] == 0
    assert second["rows"][0]["predictions"]["frozen_composition"]["bhp_error"] == -500
    assert first["prediction_config"]["ken_well"] == .03
    assert first["prediction_config"]["fnz_well"] == 1.


def test_failed_predictions_stay_in_coverage():
    def fail(_):
        raise ValueError("cannot lift")
    challenge, = chronological_challenges(_eras())
    cfg = WellConfig(well_name="W1", res_pres=1500, form_temp=80, jpump_tvd=4000)
    result = evaluate_challenge(cfg, challenge, predict_function=fail)
    assert len(result["rows"]) == 3
    assert all("error" in r["predictions"]["frozen_composition"] for r in result["rows"])
    assert result["metrics"]["frozen_composition"]["solved"] == 0
    assert result["validated_for_sizing"] is False
