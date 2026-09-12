"""History replay must preserve installation boundaries and independent forecasts."""
from copy import deepcopy
from datetime import datetime, timezone
import pickle
import threading
import time

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from server import jobs, schemas, surface_cache
from server.main import app
from server.services import pump_match as pm
from woffl.assembly.network_optimizer import WellConfig


def inputs():
    tracker = pd.DataFrame([
        {"Well Name": "MPE-42", "Date Set": "2026-01-01T06:00:00Z", "Nozzle Number": "11",
         "Throat Ratio": "C", "Circ Direction": "reverse", "Tubing Diameter": 4.5},
        {"Well Name": "MPE-42", "Date Set": "2026-02-01T18:00:00Z", "Nozzle Number": "13",
         "Throat Ratio": "C", "Circ Direction": "reverse", "Tubing Diameter": 4.5},
    ])
    rows = []
    for month in (1, 2):
        for day in (1, 5, 10, 15, 20, 25):
            rows.append(dict(well="MPE-42", wt_uid=f"{month}-{day}", WtDate=f"2026-{month:02d}-{day:02d}",
                             BHP=500., WtOilVol=100., WtTotalFluid=200., form_wc=.5, fgor=250.,
                             lift_wat=2000., pf_press=3000.+day, whp=200., pf_source="annulus"))
    cfg = WellConfig(well_name="MPE-42", res_pres=1500, form_temp=80, jpump_tvd=4000)
    return cfg, tracker, pd.DataFrame(rows)


def assemble(mode="previous_pump", training_tests=3, **changes):
    cfg, tracker, tests = inputs()
    for key, value in changes.items():
        if key == "tracker": tracker = value
        elif key == "tests": tests = value
        else: setattr(cfg, key, value)
    return pm.assemble(cfg, tracker, tests, schemas.PumpMatchRequest(mode=mode, training_tests=training_tests),
                       "2026-03-01T00:00:00+00:00")


def test_previous_pump_can_train_even_without_its_own_forecast():
    eras, rows, work, _ = assemble()
    assert eras[0]["unavailable"]
    assert eras[1]["unavailable"] is None
    assert eras[1]["training_count"] == 3
    assert len(work) == 5
    assert all(rows[index]["phase"] == "prediction" for _, _, index in work)


def test_held_out_outcomes_and_composition_never_change_predictions():
    cfg, tracker, tests = inputs()
    first = assemble()[2]
    tests.loc[tests.WtDate.str.startswith("2026-02"), ["BHP", "WtOilVol", "lift_wat", "form_wc", "fgor"]] = [1000, 9999, 25000, .9, 1000]
    second = assemble(tests=tests)[2]
    assert pickle.dumps(first) == pickle.dumps(second)
    assert all(set(controls) == {"date", "ppf", "pwh"} for _, controls, _ in second)
    assert all(c.ken_well == .03 and c.fnz_well == 1 for c, _, _ in second)


def test_missing_held_out_outcomes_keep_the_prediction_in_coverage():
    _, _, tests = inputs()
    tests.loc[tests.WtDate.str.startswith("2026-02"), ["BHP", "WtOilVol", "lift_wat", "form_wc", "fgor"]] = None
    assert len(assemble(tests=tests)[2]) == 5


def test_same_pump_marks_training_and_embargo_separately():
    eras, rows, work, _ = assemble(mode="same_pump")
    assert {rows[i]["phase"] for _, _, i in work} == {"fit", "prediction"}
    assert len([1 for _, _, i in work if rows[i]["phase"] == "fit"]) == 6
    assert all(r["status"] == "excluded" for r in rows if r["date"].startswith(("2026-01-01", "2026-02-01")))
    assert eras[0]["training_end"].startswith("2026-01-15")
    _, _, tests = inputs()
    new = dict(tests.iloc[3], WtDate="2026-01-18", wt_uid="embargo")
    tests = pd.concat([tests, pd.DataFrame([new])], ignore_index=True)
    rows = assemble(mode="same_pump", tests=tests)[1]
    assert next(r for r in rows if r["wt_uid"] == "embargo")["status"] == "excluded"


def test_exact_timestamps_and_same_size_changeouts_remain_distinct():
    _, tracker, _ = inputs()
    tracker.loc[1, "Nozzle Number"] = "11"
    eras, _, work, _ = assemble(tracker=tracker)
    assert eras[0]["pump"] == eras[1]["pump"]
    assert eras[0]["installation_id"] != eras[1]["installation_id"]
    assert "18:00:00+00:00" in eras[1]["date_set"]
    assert len(work) == 5


def test_unobserved_or_unsupported_intervening_pump_blocks_transfer():
    _, tracker, _ = inputs()
    extra = dict(tracker.iloc[0], **{"Date Set": "2026-01-29", "Nozzle Number": "bad"})
    tracker = pd.concat([tracker, pd.DataFrame([extra])], ignore_index=True)
    assert assemble(tracker=tracker)[2] == []
    tracker.loc[2, "Nozzle Number"] = "12"
    eras, rows, work, _ = assemble(tracker=tracker)
    assert work == []
    assert eras[-1]["training_installation_id"] == eras[1]["installation_id"]
    assert "Immediately preceding pump 12C (set 2026-01-29) has 0 usable training dates" in eras[-1]["unavailable"]
    assert all(r["message"] == eras[-1]["unavailable"] for r in rows if r["installation_id"] == eras[-1]["installation_id"])


@pytest.mark.parametrize("mode", ["previous_pump", "same_pump"])
def test_insufficient_training_keeps_the_candidate_count_and_dates(mode):
    _, _, tests = inputs()
    tests.loc[~tests.WtDate.str.endswith(("05", "10")), "BHP"] = None
    eras, _, work, _ = assemble(mode=mode, tests=tests)
    era = eras[-1]
    assert work == []
    assert era["training_count"] == 2
    assert len(era["training_test_ids"]) == 2
    month = "01" if mode == "previous_pump" else "02"
    assert era["training_start"].startswith(f"2026-{month}-05")
    assert era["training_end"].startswith(f"2026-{month}-10")
    assert "2 usable training dates; at least 3" in era["unavailable"]
    assert "prediction_config" not in era


@pytest.mark.parametrize("mode", ["previous_pump", "same_pump"])
def test_shorter_plot_window_keeps_earlier_training_and_predictions(mode):
    cfg, tracker, tests = inputs()
    tracker.loc[0, "Date Set"] = "2025-01-01T06:00:00Z"
    tracker.loc[1, "Date Set"] = "2025-09-01T18:00:00Z"
    if mode == "same_pump":
        tracker = tracker.iloc[:1].copy()
    tests["WtDate"] = tests.WtDate.str.replace("2026-01", "2025-01").str.replace("2026-02", "2025-09")
    outputs = []
    for months in (24, 6):
        req = schemas.PumpMatchRequest(mode=mode, months=months, training_tests=3)
        outputs.append(pm.assemble(cfg, tracker, tests, req, "2026-03-01T00:00:00Z"))
    long, short = outputs
    assert all(r["date"] >= "2025-09-01" for r in short[1])
    assert short[0][-1]["training_test_ids"] == long[0][-1]["training_test_ids"]
    assert short[0][-1]["training_start"].startswith("2025-01")
    assert len(short[2]) > 0
    tasks = [{rows[i]["wt_uid"]: (vars(cfg), controls) for cfg, controls, i in work}
             for _eras, rows, work, _notes in outputs]
    for uid, task in tasks[1].items():
        assert task == tasks[0][uid]
    assert all(short[1][i]["phase"] == "prediction" for _cfg, _controls, i in short[2])


def test_every_test_replay_covers_single_test_installations_without_reanchoring():
    _, _, tests = inputs()
    tests = tests[tests.WtDate.str.endswith("05")].copy()
    eras, rows, work, _ = assemble(mode="all_tests", tests=tests, qwf=321., pwf=450., form_wc=.4, form_gor=444.)
    assert len(work) == 2
    assert {rows[i]["phase"] for _cfg, _controls, i in work} == {"replay"}
    assert all((cfg.qwf, cfg.pwf, cfg.form_wc, cfg.form_gor) == (385.2, 450., .5, 250.) for cfg, _, _ in work)
    assert {cfg.installed_nozzle for cfg, _, _ in work} == {"11", "13"}
    assert all(e.get("training_count", 0) == 0 for e in eras)
    # Outcomes never become hidden inflow anchors. Measured WC/GOR are inputs.
    tests.loc[:, ["BHP", "WtOilVol", "lift_wat"]] = [1000., 9999., 25000.]
    altered = assemble(mode="all_tests", tests=tests, qwf=321., pwf=450., form_wc=.4, form_gor=444.)[2]
    assert pickle.dumps(work) == pickle.dumps(altered)


def test_replay_test_wc_gor_preserve_one_saved_oil_ipr_across_all_pumps():
    from woffl.assembly.network_optimizer import NetworkOptimizer

    cfg, tracker, tests = inputs()
    cfg.qwf, cfg.pwf, cfg.form_wc, cfg.form_gor = 321., 450., .4, 444.
    before = deepcopy(vars(cfg))
    reference = NetworkOptimizer._create_well_objects(cfg)[2]
    tests["form_wc"] = [.1, 0., .2, .3, .4, .5, .6, .7, .8, .9, .99, .999]
    tests["fgor"] = [100. * i for i in range(len(tests))]
    tests.loc[1, "fgor"] = 0.
    eras, rows, work, _ = pm.assemble(cfg, tracker, tests, schemas.PumpMatchRequest(), "2026-03-01T00:00:00Z")
    assert len(work) == 10
    assert len({id(at) for at, _, _ in work}) == 10
    for at, _controls, index in work:
        row = rows[index]
        assert at.form_wc == row["input_wc"] == row["wc"]
        assert at.form_gor == row["input_gor"] == row["gor"]
        _bore, _profile, ipr, mix, _pf = NetworkOptimizer._create_well_objects(at)
        assert mix.wc == row["wc"] and mix.fgor == row["gor"]
        assert at.pwf == cfg.pwf and at.res_pres == cfg.res_pres
        for pressure in (50., 450., 800., 1200., 1490.):
            assert ipr.oil_flow(pressure, method="vogel") == pytest.approx(
                reference.oil_flow(pressure, method="vogel"), rel=1e-14)
    assert vars(cfg) == before
    assert all(e["prediction_config"]["qwf"] == cfg.qwf for e in eras)
    assert all(e["prediction_config"]["form_wc"] == cfg.form_wc for e in eras)


@pytest.mark.parametrize("column,value,reason", [
    ("form_wc", None, "test WC"), ("form_wc", 1., "test WC"),
    ("form_wc", -.01, "test WC"), ("fgor", None, "test GOR"),
    ("fgor", -1., "test GOR"), ("fgor", float("inf"), "test GOR"),
])
def test_replay_missing_composition_is_a_visible_gap_not_a_saved_value_fallback(column, value, reason):
    _, _, tests = inputs()
    tests.loc[1, column] = value
    _, rows, work, _ = assemble(mode="all_tests", tests=tests)
    row = next(r for r in rows if r["wt_uid"] == "1-5")
    assert row["status"] == "missing" and row["phase"] is None
    assert reason in row["message"]
    assert len(work) == 9
    assert all(rows[i]["wt_uid"] != row["wt_uid"] for _, _, i in work)


def test_duplicate_and_undated_tracker_records_never_supply_a_fit():
    _, tracker, _ = inputs()
    duplicate = pd.concat([tracker, tracker.iloc[[0]]], ignore_index=True)
    assert assemble(tracker=duplicate)[2] == []
    tracker.loc[0, "Date Set"] = None
    assert assemble(tracker=tracker)[2] == []


def test_pressure_source_conflict_and_duplicate_tests_stay_visible():
    _, _, tests = inputs()
    tests.loc[7, "pf_source"] = "tubing"
    tests.loc[8, "wt_uid"] = tests.loc[9, "wt_uid"]
    _, rows, work, _ = assemble(tests=tests)
    assert len(work) == 2
    assert any("circulation disagree" in (r["message"] or "") for r in rows)
    assert sum("duplicate" in (r["message"] or "") for r in rows) == 2


@pytest.fixture
def runtime(monkeypatch):
    cfg, tracker, tests = inputs()
    monkeypatch.setattr(pm.wells, "well_context", lambda *_: {"seeds": dict(pres=1500, form_wc=.999, ken=99.)})
    monkeypatch.setattr(pm.datasources, "jp_history", lambda: (tracker, "databricks"))
    monkeypatch.setattr(pm.tests, "fetch_all_well_tests", lambda *_: tests)
    monkeypatch.setattr(pm.pool, "submit", lambda *_: None)
    class Clock:
        @staticmethod
        def now(_): return datetime(2026, 3, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(pm, "datetime", Clock)
    surface_cache.clear()
    yield tracker, tests
    surface_cache.clear()


def fake_predict(tasks):
    return [(i, dict(predicted_bhp=510., predicted_oil=105., predicted_pf=2100.,
                     predicted_liquid=210., sonic=False)) for _cfg, _point, i in tasks]


def test_replay_snapshot_cache_and_scores(runtime, monkeypatch):
    calls = []
    fetch_windows = []
    def fetch(months):
        fetch_windows.append(months)
        return runtime[1]
    monkeypatch.setattr(pm.tests, "fetch_all_well_tests", fetch)
    def predict(tasks):
        calls.append(tasks)
        return fake_predict(tasks)
    monkeypatch.setattr(pm, "predict_chunk", predict)
    req = schemas.PumpMatchRequest(mode="previous_pump", training_tests=3, months=6)
    result = pm.run({}, "MPE-42", req)
    assert fetch_windows == [24]
    assert result["eras"][1]["prediction_scores"]["solved"] == 5
    assert result["eras"][1]["prediction_scores"]["oil_mae"] == 5
    assert result["validated_for_sizing"] is False
    again = pm.run({}, "MPE-42", req)
    assert len(calls) == 1 and result == again
    again["rows"][0]["oil"] = -999
    assert pm.run({}, "MPE-42", req)["rows"][0]["oil"] != -999
    runtime[1].loc[7, "WtOilVol"] = 120.
    changed = pm.run({}, "MPE-42", req)
    assert changed["snapshot_id"] != result["snapshot_id"]
    assert len(calls) == 2
    different = pm.run({}, "MPE-42", req.model_copy(update={"hydraulics_model": "hagedorn_brown"}))
    assert different["snapshot_id"] != changed["snapshot_id"]


def test_failed_solves_remain_rows_with_no_zero_prediction(runtime, monkeypatch):
    monkeypatch.setattr(pm, "predict_chunk", lambda tasks: [(i, {"message": "cannot lift"}) for _, _, i in tasks])
    result = pm.run({}, "MPE-42", schemas.PumpMatchRequest(mode="previous_pump", training_tests=3))
    s = result["eras"][1]["prediction_scores"]
    assert s["attempted"] == s["failed"] == 5 and s["solved"] == 0
    failed = [r for r in result["rows"] if r["status"] == "failed"]
    assert len(failed) == 5 and all(r["predicted_oil"] is None for r in failed)


def test_default_replay_uses_saved_well_inputs_and_keeps_failure_coverage(runtime, monkeypatch):
    seeds = dict(pres=1500., qwf=321., pwf=450., form_wc=.4, form_gor=444., ken=99., nozzle_area_factor=2.)
    monkeypatch.setattr(pm.wells, "well_context", lambda *_: {"seeds": seeds})
    calls = []
    def predict(tasks):
        calls.extend(tasks)
        return [(i, dict(message="cannot lift") if point["date"][8:10] == "10" else fake_predict([(cfg, point, i)])[0][1])
                for cfg, point, i in tasks]
    monkeypatch.setattr(pm, "predict_chunk", predict)
    client = TestClient(app)
    jid = client.post("/api/wells/MPE-42/pump-match", json={}).json()["job_id"]
    job = wait_job(client, jid)
    assert job["status"] == "done", job
    result = job["result"]
    assert result["request"]["mode"] == "all_tests"
    assert len(calls) == 10  # Every test other than the two installation days.
    assert all((cfg.qwf, cfg.pwf, cfg.form_wc, cfg.form_gor) == (385.2, 450., .5, 250.) for cfg, _, _ in calls)
    assert result["well_inputs"]["qwf"] == 321.
    assert result["well_inputs"]["form_wc"] == .4
    assert result["well_inputs"]["form_gor"] == 444.
    assert all(cfg.ken_well == .03 and cfg.fnz_well == 1. for cfg, _, _ in calls)
    assert all(set(point) == {"date", "ppf", "pwh"} for _, point, _ in calls)
    assert sum(e["replay_scores"]["failed"] for e in result["eras"]) == 2
    assert sum(e["replay_scores"]["solved"] for e in result["eras"]) == 8
    assert sum(e["prediction_scores"]["attempted"] + e["fit_scores"]["attempted"] for e in result["eras"]) == 0
    assert all(r["predicted_oil"] is None for r in result["rows"] if r["status"] == "failed")
    runtime[1].loc[1, ["form_wc", "fgor"]] = [.75, 700.]
    revised_test = pm.run({}, "MPE-42", schemas.PumpMatchRequest())
    assert revised_test["snapshot_id"] != result["snapshot_id"]
    changed_row = next(r for r in revised_test["rows"] if r["wt_uid"] == "1-5")
    assert (changed_row["input_wc"], changed_row["input_gor"]) == (.75, 700.)
    assert revised_test["well_inputs"] == result["well_inputs"]
    seeds["qwf"] = 400.
    changed = pm.run({}, "MPE-42", schemas.PumpMatchRequest())
    assert changed["snapshot_id"] != result["snapshot_id"]
    assert changed["well_inputs"]["qwf"] == 400.


def test_worker_forwards_model_and_scoped_hardware(monkeypatch):
    from woffl.assembly import solopump
    cfg, _, _ = inputs()
    cfg.hydraulics_model = "hagedorn_brown"
    cfg.installed_nozzle, cfg.installed_throat = "13", "C"
    cfg.ken_well, cfg.kth_well, cfg.kdi_well, cfg.fnz_well = .12, .35, .42, 1.1
    monkeypatch.setattr("woffl.assembly.network_optimizer.NetworkOptimizer._create_well_objects", lambda *_: (1, 2, 3, 4, 5))
    calls = []
    def solve(*args, **kwargs):
        calls.append((args, kwargs))
        return 500., False, 100., 100., 2000., .1
    monkeypatch.setattr(solopump, "jetpump_solver", solve)
    result = pm.predict_chunk([(cfg, {"date": "2026-02-15", "ppf": 3000., "pwh": 200.}, 7)])
    assert result[0][1]["predicted_liquid"] == 200.
    assert calls[0][1]["hydraulics_model"] == "hagedorn_brown"
    assert calls[0][0][3].ken == .12
    from woffl.geometry import JetPump
    assert calls[0][0][3].dnz == pytest.approx(JetPump("13", "C").dnz * 1.1**.5)
    assert (calls[0][0][3].kth, calls[0][0][3].kdi) == (.35, .42)


def scoped_history():
    from server.services.well_model import describe
    cfg, tracker, tests = inputs()
    # Two installations of the SAME catalog size must retain distinct scope.
    tracker.loc[0, "Nozzle Number"] = "13"
    coefs = dict(ken=.005, kth=.386, kdi=.072, nozzle_area_factor=1.01)
    fit = dict(status="active", pump="13C", date_set=tracker.iloc[-1]["Date Set"],
               well_model_fingerprint=describe(cfg)["fingerprint"], coefficients=coefs)
    return cfg, tracker, tests, fit


def test_saved_history_fit_applies_only_to_its_exact_installation_with_measured_composition():
    cfg, tracker, tests, fit = scoped_history()
    tests.loc[tests.WtDate.str.startswith("2026-02"), ["form_wc", "fgor"]] = [.82, 700.]
    eras, rows, work, _ = pm.assemble(cfg, tracker, tests, schemas.PumpMatchRequest(),
                                     "2026-03-01T00:00:00Z", fit)
    assert [e["pump_losses"] for e in eras] == ["clean_reference", "saved_calibration"]
    assert eras[0]["pump"] == eras[1]["pump"] == "13C"
    for at, controls, index in work:
        if rows[index]["date"].startswith("2026-02"):
            assert (at.ken_well, at.kth_well, at.kdi_well, at.fnz_well) == (.005, .386, .072, 1.01)
            assert (at.form_wc, at.form_gor) == (.82, 700.)
        else:
            assert (at.ken_well, at.kth_well, at.kdi_well, at.fnz_well) == (.03, .3, .4, 1.)
        assert at.qwf*(1-at.form_wc) == pytest.approx(cfg.qwf*(1-cfg.form_wc))
        assert set(controls) == {"date", "ppf", "pwh"}


@pytest.mark.parametrize("change", ["status", "time", "pump", "hash", "curve", "geometry", "hydraulics", "clean", "chronological"])
def test_saved_history_fit_never_crosses_unsupported_scope(change):
    cfg, tracker, tests, fit = scoped_history()
    req = schemas.PumpMatchRequest(training_tests=3)
    if change == "status": fit["status"] = "stale"
    elif change == "time": fit["date_set"] = "2026-02-01T17:00:00Z"
    elif change == "pump": fit["pump"] = "12C"
    elif change == "hash": fit["well_model_fingerprint"] = "0"*32
    elif change == "curve": cfg.qwf += 10.
    elif change == "geometry": tracker.loc[1, "Tubing Diameter"] = 3.5
    elif change == "hydraulics": cfg.hydraulics_model = "drift_flux"
    elif change == "clean": req.pump_losses = "clean_reference"
    elif change == "chronological": req.mode = "same_pump"
    eras, _, work, _ = pm.assemble(cfg, tracker, tests, req, "2026-03-01T00:00:00Z", fit)
    assert work
    assert all(e["pump_losses"] == "clean_reference" for e in eras)
    assert all((at.ken_well, at.kth_well, at.kdi_well, at.fnz_well) == (.03, .3, .4, 1.) for at, _, _ in work)


@pytest.mark.parametrize("model", ["beggs", "hagedorn_brown", "drift_flux"])
def test_replay_worker_matches_real_installed_single_and_batch_predictions(model):
    from server.services.optimizer_runs import _config_from_seeds
    from server.services.solve import solve_single, run_batch
    sp = schemas.SimParams(nozzle_no="13", area_ratio="C", hydraulics_model=model,
        ken=.005, kth=.386, kdi=.072, nozzle_area_factor=1.01,
        nozzle_batch_options=["13"], throat_batch_options=["C"])
    cfg = _config_from_seeds("Custom", "", sp.model_dump())
    before = deepcopy(vars(cfg))
    result = pm.predict_chunk([(cfg, {"date": "2026-02-15", "ppf": sp.ppf_surf, "pwh": sp.surf_pres}, 7)])
    index, replay = result[0]
    assert index == 7 and "message" not in replay
    single = solve_single("Custom", sp)
    installed = next(r for r in run_batch("Custom", sp)["rows"] if r["pump_state"] == "installed")
    for key, single_key, batch_key in (("predicted_bhp", "psu", "psu_solv"),
            ("predicted_oil", "qoil_std", "qoil_std"), ("predicted_pf", "qnz_bwpd", "lift_wat")):
        assert replay[key] == pytest.approx(single[single_key], abs=1e-8)
        assert replay[key] == pytest.approx(installed[batch_key], abs=1e-8)
    assert replay["predicted_liquid"] == pytest.approx(single["qoil_std"]+single["fwat_bwpd"], abs=1e-8)
    assert vars(cfg) == before


def test_edited_input_preview_keeps_test_composition_and_hardware_without_saving(runtime, monkeypatch):
    saved = dict(pres=1500., qwf=321., pwf=450., form_wc=.4, form_gor=444., form_temp=80., bubble_point=1750.)
    monkeypatch.setattr(pm.wells, "well_context", lambda *_: {"seeds": saved})
    from woffl.gui import ipr_anchor
    monkeypatch.setattr(ipr_anchor, "save_ipr_values", lambda *a, **k: pytest.fail("preview must not save"))
    calls = []
    def predict(work):
        calls.extend(work)
        return fake_predict(work)
    monkeypatch.setattr(pm, "predict_chunk", predict)
    payload = dict(qwf_liq=400.25, pwf=480.5, res_pres=1600.75, form_wc=.3, form_gor=900.,
                   surf_pres=225., form_temp=95., bubble_point=1900.)
    request = schemas.PumpMatchRequest(edited_inputs=payload)
    preview = pm.run({}, "MPE-42", request)
    assert preview["well_inputs"]["qwf"] == 400.25
    assert preview["well_inputs"]["pwf"] == 480.5
    assert preview["well_inputs"]["res_pres"] == 1600.75
    assert len(calls) == 10
    for cfg, controls, _ in calls:
        assert cfg.qwf * (1 - cfg.form_wc) == pytest.approx(400.25 * .7)
        assert cfg.form_wc == .5 and cfg.form_gor == 250.
        assert cfg.form_temp == 95. and cfg.bubble_point == 1900.
        assert cfg.installed_nozzle in {"11", "13"} and cfg.ken_well == .03
        assert controls["pwh"] == 200.  # Recorded pressure, not sidebar WHP.
    assert "Nothing is saved" in preview["notes"][0]
    assert saved["qwf"] == 321. and saved["form_temp"] == 80.
    baseline = pm.run({}, "MPE-42", schemas.PumpMatchRequest())
    assert baseline["snapshot_id"] != preview["snapshot_id"]
    assert baseline["well_inputs"]["qwf"] == 321.


@pytest.mark.parametrize("extra", [{"ken": .1}, {"nozzle_no": "15"}, {"oil_api": 30.}, {"form_wc": 1.}, {"pwf": 1800.}])
def test_preview_rejects_unsavable_or_invalid_inputs(extra):
    payload = dict(qwf_liq=400., pwf=500., res_pres=1500., form_wc=.5, form_gor=250.)
    with pytest.raises(ValueError):
        schemas.PumpMatchRequest(edited_inputs={**payload, **extra})


def test_preview_cannot_enter_chronological_refitting():
    with pytest.raises(ValueError, match="every-test"):
        schemas.PumpMatchRequest(mode="previous_pump", edited_inputs=dict(
            qwf_liq=400., pwf=500., res_pres=1500., form_wc=.5, form_gor=250.))


def wait_job(client, jid):
    deadline = time.monotonic()+5
    while time.monotonic() < deadline:
        value = client.get(f"/api/pump-match/{jid}").json()
        if value["status"] != "running": return value
        time.sleep(.01)
    raise AssertionError("job did not settle")


def test_api_lifecycle_and_request_validation(runtime, monkeypatch):
    monkeypatch.setattr(pm, "predict_chunk", fake_predict)
    client = TestClient(app)
    assert client.post("/api/wells/MPE-42/pump-match", json={"hydraulics_model": "tulsa"}).status_code == 422
    assert client.post("/api/wells/MPE-42/pump-match", json={"training_tests": 2}).status_code == 422
    assert client.get("/api/pump-match/missing").status_code == 404
    assert client.delete("/api/pump-match/missing").status_code == 404
    start = client.post("/api/wells/MPE-42/pump-match", json={"mode": "previous_pump", "training_tests": 3})
    assert start.status_code == 200
    value = wait_job(client, start.json()["job_id"])
    assert value["status"] == "done", value
    assert value["result"]["well"] == "MPE-42"
    assert client.get(f"/api/optimize/run/{value['job_id']}").status_code == 404


def test_cancelled_queued_job_does_no_data_access(monkeypatch):
    slots = threading.BoundedSemaphore(1)
    slots.acquire()
    monkeypatch.setattr(jobs, "_JOB_SLOTS", slots)
    called = []
    monkeypatch.setattr(pm, "run", lambda *_: called.append(True))
    client = TestClient(app)
    jid = client.post("/api/wells/MPE-42/pump-match", json={}).json()["job_id"]
    try:
        assert client.delete(f"/api/pump-match/{jid}").status_code == 200
        assert wait_job(client, jid)["status"] == "cancelled"
        assert not called
    finally:
        slots.release()


def test_cancellation_waits_for_chunk_and_discards_result(runtime, monkeypatch):
    begun, release = threading.Event(), threading.Event()
    def predict(tasks):
        begun.set()
        release.wait(4)
        return fake_predict(tasks)
    monkeypatch.setattr(pm, "predict_chunk", predict)
    client = TestClient(app)
    jid = client.post("/api/wells/MPE-42/pump-match", json={"mode": "previous_pump", "training_tests": 3}).json()["job_id"]
    try:
        assert begun.wait(3)
        client.delete(f"/api/pump-match/{jid}")
        assert client.get(f"/api/pump-match/{jid}").json()["status"] == "running"
    finally:
        release.set()
    value = wait_job(client, jid)
    assert value["status"] == "cancelled" and value["result"] is None
