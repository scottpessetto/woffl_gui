from copy import deepcopy
from dataclasses import asdict

import pandas as pd
import pytest

from server.services.fleet_validation import prepare, predict_well, metrics, saved_cutoff
from server.services.optimizer_runs import _config_from_seeds


def snapshot():
    name = "MPM-64"
    cfg = _config_from_seeds(name, "M", dict(nozzle_no="12", area_ratio="B", pres=1700))
    tests = [dict(well=name, wt_uid=i, WtDate=day, BHP=500., WtOilVol=200.,
                  WtTotalFluid=1000., form_wc=.8, fgor=600., lift_wat=2500.,
                  pf_press=3000., pf_source="annulus", whp=210.)
             for i, day in enumerate(["2026-07-31", "2026-08-01", "2026-08-17", "2026-09-05"])]
    return dict(captured_at="2026-09-08T17:00:00+00:00",
        universe=dict(wells=[dict(name=name, pad="M")]), configs={name: asdict(cfg)},
        contexts={name: dict(ipr_source="saved", pump=dict(date_set="2026-08-01", source="databricks"))},
        saved={name: dict(saved_at="2026-08-10T13:00:00Z", friction_at={"kdi":"2026-08-20T13:00:00Z"})}, errors={},
        frames=dict(tests=pd.DataFrame(tests),
            bhp=pd.DataFrame([dict(well_name="M-064", tag_date="2026-09-07", bhp_cln_value=510.)]),
            pressure=pd.DataFrame([dict(well=name,sample_date="2026-09-07",tubing_prs=210.,inn_ann_prs=3000.,btmhole_prs=510.)]),
            pf_volume=pd.DataFrame([dict(well=name,pfdate="2026-09-07",pwr_fld_net=2500.)])))


def test_fleet_excludes_prior_pump_and_uses_actual_observation_pressures():
    source = snapshot()
    record = prepare(source)[0]
    assert not record.get("exclusion")
    assert [r["date"] for r in record["observations"]] == ["2026-08-17", "2026-09-05", "2026-09-07"]
    assert record["rejected_tests"] == {"before current pump or installation day": 2}
    assert [r["after_saved"] for r in record["observations"]] == [False, True, True]
    assert all(r["pwh"] == 210. and r["ppf"] == 3000. for r in record["observations"])


def test_missing_pf_volume_keeps_bhp_and_oil_with_separate_metric_denominators():
    source = snapshot()
    source["frames"]["tests"].loc[3,"lift_wat"] = 0.
    record = prepare(source)[0]
    result = predict_well(record, lambda inputs: (520., False, 180., 720., 2450., .5))
    wt = [r for r in result["observations"] if r["kind"] == "test"]
    assert len(wt) == 2
    assert wt[-1]["observed_pf"] is None and "pf_error_pct" not in wt[-1]
    scores = metrics(wt)
    assert scores["bhp_error"]["n"] == 2
    assert scores["oil_error_pct"]["n"] == 2
    assert scores["pf_error_pct"]["n"] == 1


def test_fleet_outcomes_never_enter_forward_solver_and_inputs_are_unchanged():
    record = prepare(snapshot())[0]
    before = deepcopy(record)
    def predictor(inputs):
        assert set(inputs) == {"ppf", "pwh"}
        return 520., False, 180., 720., 2450., .5
    first = predict_well(record, predictor)
    changed = deepcopy(record)
    changed["observations"][0]["observed_bhp"] = 1000.
    changed["observations"][0]["observed_oil"] = 2000.
    second = predict_well(changed, predictor)
    assert first["observations"][0]["predicted_bhp"] == second["observations"][0]["predicted_bhp"]
    assert record == before


@pytest.mark.parametrize("mode", ["exception", "nonfinite"])
def test_fleet_failed_predictions_stay_in_denominator(mode):
    record = prepare(snapshot())[0]
    record["observations"][0].update(predicted_bhp=500., bhp_error=0.)
    def predictor(inputs):
        if mode == "exception":
            raise ValueError("cannot lift")
        return float("nan"), False, 180., 720., 2450., .5
    rows = predict_well(record, predictor)["observations"]
    assert metrics(rows)["failed"] == len(rows)
    assert "bhp_error" not in metrics(rows)
    assert "predicted_bhp" not in rows[0]


def test_stale_gauge_is_reported_in_coverage():
    source = snapshot()
    source["frames"]["bhp"]["tag_date"] = "2026-08-01"
    record = prepare(source)[0]
    assert not record["fresh_gauge"]
    assert record["exclusion"] == "no credible BHP gauge reading in last 14 days"


def test_observation_pressure_direction_conflict_is_not_silently_reinterpreted():
    source = snapshot()
    source["frames"]["tests"]["pf_source"] = "tubing"
    record = prepare(source)[0]
    assert record.get("exclusion")
    assert record["rejected_tests"]["test circulation differs from current model"] == 2


def test_auto_ipr_has_no_claimed_after_save_holdout():
    assert saved_cutoff(dict(ipr_source="vogel"),dict(saved_at="2026-08-01")) is None
