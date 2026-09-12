"""Offline fixed-plan stress tests: immutable source, fixed oil IPR, gaps."""
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from server import schemas
from server.main import app
from server.services import optimizer_runs as runs, plan_robustness as stress
from woffl.flow.entry_energy import MODEL_VERSION


def snapshot():
    cfg = runs._config_from_seeds("MPM-01", "M", {"pres": 1700., "pwf": 600., "qwf": 1000.,
        "form_wc": .7, "form_gor": 700., "nozzle_no": "12", "area_ratio": "B", "ken": .11})
    return {"version": 1, "physics_model": MODEL_VERSION,
            "source_fingerprint": stress.source_fingerprint([asdict(cfg)]),
            "request": schemas.OptimizeRunRequest(kind="pad", pad="M").model_dump(),
            "configs": [asdict(cfg)], "header_psi": 2600., "lambda_used": .02,
            "current": {cfg.well_name: ["12", "B", "installed"]},
            "proposed": {cfg.well_name: ["13", "C", "replacement"]}}


class Plant:
    water_key = "totl_wat"
    def budget_at_pressure(self, pressure, n_pumps):
        return 6000. - (pressure - 2600.) * 5.
    def clamp_window(self, n_pumps):
        return 2000., 3500.
    def flags(self, water, n_pumps):
        return {"over_capacity": False, "recirc": False}
    def delivered_header(self, water, pressure, n_pumps):
        return pressure, water > self.budget_at_pressure(pressure, n_pumps)


def test_stress_cases_are_bounded_and_joint_assumptions_explicit():
    req = schemas.PadRobustnessRequest(source_job_id="fixture", joint_cases=True)
    cases = stress.cases(req)
    assert len(cases) == 9
    assert cases[-1]["wc_offset"] > 0 and cases[-1]["gor_factor"] > 1 and cases[-1]["header_offset"] < 0
    zero = req.model_copy(update={"wc_points": 0., "gor_percent": 0., "header_psi": 0.})
    assert len(stress.cases(zero)) == 1


def test_scenarios_hold_one_oil_ipr_and_do_not_mutate_source():
    snap = snapshot()
    original = deepcopy(snap)
    for case in stress.cases(schemas.PadRobustnessRequest(source_job_id="fixture", joint_cases=True)):
        cfg, = stress.scenario_configs(snap, case)
        assert cfg.qwf * (1 - cfg.form_wc) == pytest.approx(300.)
        assert cfg.pwf == 600. and cfg.res_pres == 1700.
        assert cfg.form_gor == pytest.approx(700. * case["gor_factor"])
        assert cfg.ken_well == .11 and cfg.pump_calibration_scoped
    assert snap == original


def test_invalid_wc_is_an_explicit_gap_not_clipped():
    snap = snapshot()
    snap["configs"][0]["form_wc"] = .98
    with pytest.raises(ValueError, match="not clipped"):
        stress.scenario_configs(snap, {"wc_offset": .03, "gor_factor": 1.})


@pytest.mark.parametrize("patch", [{"wc_points": 11}, {"gor_percent": 51}, {"header_psi": 251},
                                     {"wc_points": float("nan")}, {"configs": []}])
def test_api_refuses_unbounded_or_client_supplied_model_inputs(patch):
    response = TestClient(app).post("/api/optimize/robustness", json={"source_job_id": "fixture", **patch}) if not any(isinstance(v, float) and v != v for v in patch.values()) else None
    if response is not None:
        assert response.status_code == 422
    else:
        with pytest.raises(ValueError):
            schemas.PadRobustnessRequest(source_job_id="fixture", **patch)


def test_source_requires_complete_server_job_and_is_copied(monkeypatch):
    snap = snapshot()
    result = {"robustness_available": True, "_plan_snapshot": snap}
    monkeypatch.setattr(stress.jobs, "get", lambda job_id, kinds: {"status": "done", "result": result})
    resolved = stress.source_snapshot("job")
    resolved["configs"][0]["qwf"] = 9000.
    assert snap["configs"][0]["qwf"] == 1000.
    saved_hash = snap["source_fingerprint"]
    snap["source_fingerprint"] = "stale"
    with pytest.raises(ValueError, match="source, plant inputs, or surveys changed"):
        stress.source_snapshot("job")
    snap["source_fingerprint"] = saved_hash
    result["robustness_available"] = False
    with pytest.raises(ValueError, match="incomplete coverage"):
        stress.source_snapshot("job")


def test_robustness_endpoint_uses_only_the_completed_server_snapshot(monkeypatch):
    snap = snapshot()
    source = {"status": "done", "result": {"robustness_available": True, "_plan_snapshot": snap}}
    monkeypatch.setattr(stress.jobs, "get", lambda job_id, kinds: source if job_id == "source" and kinds == ("pad",) else None)
    captured = {}
    def start(kind, runner, progress):
        captured["kind"] = kind
        captured["runner"] = runner
        return "stress-job"
    def run(job, req, copied):
        assert copied["configs"][0]["qwf"] == 1000.
        assert copied["proposed"] == {"MPM-01": ["13", "C", "replacement"]}
        return {"request": req.model_dump()}
    monkeypatch.setattr(stress.jobs, "start", start)
    monkeypatch.setattr(stress, "run", run)
    client = TestClient(app)
    response = client.post("/api/optimize/robustness", json={"source_job_id": "source"})
    assert response.status_code == 200 and response.json() == {"job_id": "stress-job"}
    snap["configs"][0]["qwf"] = 9999.
    captured["runner"]({})
    assert captured["kind"] == "pad_robustness"
    assert client.post("/api/optimize/robustness", json={"source_job_id": "missing"}).status_code == 422


def test_same_fixed_plan_has_zero_gain_and_no_input_save(monkeypatch):
    snap = snapshot()
    snap["proposed"] = deepcopy(snap["current"])
    original = deepcopy(snap)
    monkeypatch.setattr(runs, "_pad_plant_for_run", lambda *a: Plant())
    seen = []
    def solve(configs, plans, pressure, job):
        seen.append((plans, pressure, configs[0].qwf * (1 - configs[0].form_wc)))
        return {("MPM-01", "12", "B", "installed"):
                {"oil_rate": 250., "lift_water": 3000., "formation_water": 900.}}
    monkeypatch.setattr(stress, "solve_options", solve)
    result = stress.run({}, schemas.PadRobustnessRequest(source_job_id="fixture", joint_cases=True), snap)
    assert len(seen) == 9
    assert all(r["proposed_oil_delta"] == 0. for r in result["cases"])
    assert all(r["preferred"] == ["Current", "Proposed"] for r in result["cases"])
    assert all(v[2] == pytest.approx(300.) for v in seen)
    assert result["min_comparable_oil_gain"] == result["max_comparable_oil_gain"] == 0.
    assert snap == original


def test_failures_and_plant_infeasibility_do_not_get_rank_or_regret(monkeypatch):
    snap = snapshot()
    monkeypatch.setattr(runs, "_pad_plant_for_run", lambda *a: Plant())
    def solve(configs, plans, pressure, job):
        return {("MPM-01", "12", "B", "installed"):
                {"oil_rate": 250., "lift_water": 3000., "formation_water": 900.},
                ("MPM-01", "13", "C", "replacement"):
                None if pressure < 2600. else {"oil_rate": 300., "lift_water": 5000., "formation_water": 1500.}}
    monkeypatch.setattr(stress, "solve_options", solve)
    result = stress.run({}, schemas.PadRobustnessRequest(source_job_id="fixture", wc_points=0., gor_percent=0.), snap)
    assert len(result["cases"]) == 3
    for row in result["cases"]:
        current, proposed = row["plans"]
        assert current["feasible"] is True and current["regret"] is None
        assert proposed["feasible"] is not True and proposed["regret"] is None
        assert row["preferred"] == []
    assert result["min_comparable_oil_gain"] is None
    assert result["plans"][1]["failed_cases"] == 1
    assert result["plans"][1]["infeasible_cases"] == 2


def test_strict_candidate_lookup_never_substitutes_clean_for_installed(monkeypatch):
    import woffl.assembly.network_optimizer as network
    snap = snapshot()
    configs = stress.scenario_configs(snap, {"wc_offset": 0., "gor_factor": 1.})
    looked_up = []
    class Batch:
        def __init__(self, configs, constraint, nozzles, throats, **kw):
            assert len(nozzles) <= 2 and len(throats) <= 2
            assert configs[0].ppf_surf_well == constraint.pressure == 2600.
        def run_all_batch_simulations(self, **kw):
            pass
        def get_pump_performance(self, well, nozzle, throat, *, pump_state):
            looked_up.append((well, nozzle, throat, pump_state))
            return None if pump_state == "installed" else {"oil_rate": 300., "lift_water": 4000., "formation_water": 1000.}
    monkeypatch.setattr(network, "NetworkOptimizer", Batch)
    plans = {"Current": snap["current"], "Proposed": snap["proposed"]}
    options = stress.solve_options(configs, plans, 2600., {})
    assert looked_up == [("MPM-01", "12", "B", "installed"), ("MPM-01", "13", "C", "replacement")]
    score = stress.score_plan("Current", snap["current"], configs, options, Plant(), 3, 2600., .02)
    assert score["feasible"] is None and score["oil"] is None


def test_absent_plan_mapping_is_unknown_but_explicit_shut_in_is_zero():
    snap = snapshot()
    configs = stress.scenario_configs(snap, {"wc_offset": 0., "gor_factor": 1.})
    missing = stress.score_plan("Current", {}, configs, {}, Plant(), 3, 2600., .02)
    assert missing["feasible"] is None and missing["oil"] is None
    assert missing["failed_wells"] == ["MPM-01"]
    explicit = stress.score_plan("Current", {"MPM-01": None}, configs, {}, Plant(), 3, 2600., .02)
    assert explicit["feasible"] is True and explicit["oil"] == 0.


def test_capacity_alone_does_not_certify_low_flow_pressure_delivery():
    snap = snapshot()
    configs = stress.scenario_configs(snap, {"wc_offset": 0., "gor_factor": 1.})
    plant = Plant()
    plant.delivered_header = lambda water, pressure, n: (pressure - 100., False)
    options = {("MPM-01", "12", "B", "installed"):
               {"oil_rate": 250., "lift_water": 3000., "formation_water": 900.}}
    score = stress.score_plan("Current", snap["current"], configs, options, plant, 3, 2600., .02)
    assert score["machine_water"] < score["budget"]
    assert score["feasible"] is False and score["objective"] is None
    assert score["coupling_residual_psi"] == -100.
