"""Sensitivity scenarios must reproduce Apply and preserve explicit IPR scope."""

import pytest

from server import schemas
from server.services import sensitivity, solve
from server.services.scenarios import scenario_params, scenario_patch
from woffl.assembly.pump_candidates import CLEAN_PUMP


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    from woffl.assembly import databricks_client
    def forbidden(*args, **kwargs):
        pytest.fail("Sensitivity scenarios cannot access Databricks")
    monkeypatch.setattr(databricks_client, "execute_query", forbidden)
    monkeypatch.setattr(databricks_client, "execute_write", forbidden)


@pytest.mark.parametrize("update", [{"nozzle_no": "13"}, {"area_ratio": "C"}, {"pump_state": "replacement"}])
def test_changed_hardware_has_clean_coefficients_and_reproduces_apply(update):
    base = schemas.SimParams(ken=.2, kth=.7, kdi=.8, nozzle_area_factor=1.15)
    candidate = scenario_params(base, update)
    assert candidate.pump_state == "replacement"
    assert all(getattr(candidate, key) == value for key, value in CLEAN_PUMP.items())
    applied = schemas.SimParams.model_validate({**base.model_dump(), **scenario_patch(base, candidate)})
    assert candidate == applied
    result = sensitivity._solve_chunk("Custom", base, [update])[0]
    expected = sensitivity._metrics(solve.solve_single("Custom", applied))
    assert {key: result[key] for key in expected} == expected
    assert base.ken == .2 and base.nozzle_area_factor == 1.15


def test_installed_loss_study_remains_available_and_replacement_stays_clean():
    installed = schemas.SimParams()
    assert scenario_params(installed, {"kth": .8}).kth == .8
    clean = schemas.SimParams(pump_state="replacement")
    assert scenario_params(clean, {"kth": .8}).kth == CLEAN_PUMP["kth"]
    assert sensitivity._solve_chunk("Custom", clean, [{"kth": .8}])[0]["qoil"] == solve.solve_single("Custom", clean)["qoil_std"]


@pytest.mark.parametrize("wc", [.55, .8, .92])
def test_composition_changes_preserve_one_oil_curve_at_fractional_inputs(wc):
    base = schemas.SimParams(qwf=1000.25, pwf=512.75, pres=1700.5, form_wc=.713)
    candidate = scenario_params(base, {"form_wc": wc, "form_gor": 534.25})
    p = candidate.to_simulation_params("Custom")
    assert p.inflow_rate == pytest.approx(base.qwf * (1 - base.form_wc))
    assert p.pwf == 512.75 and p.pres == 1700.5
    assert p.form_gor == 534.25
    assert scenario_patch(base, candidate)["qwf"] == candidate.qwf


def test_anchor_measurement_and_deliberate_curve_edits_are_explicit():
    base = schemas.SimParams(qwf=1000, form_wc=.8)
    measured = scenario_params(base, {"form_wc": .85}, "anchor_measurement")
    assert measured.qwf == 1000
    assert measured.to_simulation_params("Custom").inflow_rate == pytest.approx(150)
    edited = scenario_params(base, {"qwf": 1200, "form_wc": .85})
    assert edited.to_simulation_params("Custom").inflow_rate == pytest.approx(240)


@pytest.mark.parametrize("update", [{"pres": 400}, {"qwf": float("nan")}, {"form_wc": 1.}])
def test_invalid_scenario_is_retained_as_failure(update):
    got = sensitivity._solve_chunk("Custom", schemas.SimParams(), [update])[0]
    assert got.get("error")
    assert "applied_inputs" not in got


def test_single_and_combined_wc_scenarios_use_same_resolved_inputs():
    base = schemas.SimParams(form_wc=.8, qwf=1000.25)
    one = sensitivity._params_for(sensitivity._BY_ID["form_wc"], base, .85)
    got = sensitivity._solve_chunk("Custom", base, [{"form_wc": .85}])[0]
    assert got["applied_inputs"] == scenario_patch(base, one)
    assert got["qoil"] == solve.solve_single("Custom", one)["qoil_std"]


def test_combined_result_freezes_request_and_resolved_inputs():
    params = schemas.SimParams()
    knobs = [schemas.CombineKnob(id="form_wc", low=.45, high=.55, levels=2)]
    result = sensitivity.run_combine("Custom", params, {"target_qoil": 400}, knobs,
        test_key="2026-07-15|42", installation_key="set-2026-07-01")
    snapshot = result["request"]
    params.qwf = 1500
    knobs[0].high = .9
    assert snapshot["params"]["qwf"] == 750
    assert snapshot["knobs"][0]["high"] == .55
    assert snapshot["test_key"] == "2026-07-15|42"
    assert snapshot["installation_key"] == "set-2026-07-01"
    assert snapshot["wc_basis"] == "fixed_oil_ipr"
    for row in result["runs"]:
        assert row.get("applied_inputs")
        applied = schemas.SimParams.model_validate({**snapshot["params"], **row["applied_inputs"]})
        assert row["qoil"] == solve.solve_single("Custom", applied)["qoil_std"]


def test_score_does_not_count_oil_and_liquid_twice():
    target = {"target_psu": 1000, "target_qoil": 100, "target_qliq": 500, "target_qpf": 2000}
    got = {"psu": 1000, "qoil": 110, "qliq": 750, "qpf": 2000}
    assert sensitivity._score(got, target) == pytest.approx((.01 / 3) ** .5)
    target["target_qoil"] = None
    assert sensitivity._score(got, target) == pytest.approx((.25 / 3) ** .5)


def test_fractional_inputs_are_preserved_through_simulation_conversion():
    inputs = dict(qwf=987.25, pwf=612.5, pres=1589.75, form_gor=734.25,
                  surf_pres=212.5, form_temp=110.25, ppf_surf=3100.75, jpump_tvd=4500.25)
    p = schemas.SimParams(**inputs).to_simulation_params("Custom")
    assert {key: getattr(p, key) for key in inputs} == inputs
