"""Installed-hardware persistence and selection; every warehouse boundary mocked."""
from types import SimpleNamespace
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from server import schemas
from server.cache import clear_all_caches
from server.services import pump_calibration as pc
from woffl.assembly.pump_candidates import CLEAN_PUMP, scoped_pumps
from woffl.assembly.network_optimizer import NetworkOptimizer, WellConfig, PowerFluidConstraint
from woffl.assembly.optimization_algorithms import milp_optimization, mckp_optimization
from woffl.flow.entry_energy import MODEL_VERSION
from woffl.geometry.jetpump import JetPump

WELL = "MPE-42"
PUMP = {"nozzle_no": "13", "throat_ratio": "C", "date_set": "2026-08-10", "source": "databricks"}
COEFS = {"ken": .005, "kth": .386, "kdi": .072, "nozzle_area_factor": 1.01}


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    clear_all_caches()
    monkeypatch.delenv("ALLOW_DATABRICKS_WRITES", raising=False)
    def blocked(*a, **kw):
        raise AssertionError("Unexpected warehouse access")
    monkeypatch.setattr(pc.history, "execute_query", blocked)
    monkeypatch.setattr(pc.history, "execute_write", blocked)
    yield
    clear_all_caches()


def record():
    return {"v": 1, "n": "13", "t": "C", "i": pc.installation(PUMP["date_set"]),
            "m": MODEL_VERSION, "k": list(COEFS.values()),
            "q": {"bhp": 76., "pf": 61.5, "n": 19, "bounds": ["ken"]}}


def install_record(monkeypatch, rec):
    monkeypatch.setattr(pc, "snapshot", lambda: {WELL: {"comment_text": json.dumps(rec),
        "entry_datetime": "2026-09-08", "entry_user": "engineer@example.com"}})


def test_fit_reloads_only_for_its_installation_and_model(monkeypatch):
    install_record(monkeypatch, record())
    result = pc.resolve(WELL, PUMP)
    assert result["status"] == "active"
    assert result["coefficients"] == COEFS
    assert result["quality"]["pf"] == 61.5
    # Same size, later installation is a different physical pump.
    for changed in ({**PUMP, "date_set": "2026-09-01"}, {**PUMP, "nozzle_no": "14"},
                    {**PUMP, "source": "excel_fallback"}, None):
        result = pc.resolve(WELL, changed)
        assert result["status"] == "stale" and result["coefficients"] == {}
    rec = record(); rec["m"] = "old-model"
    install_record(monkeypatch, rec)
    assert pc.resolve(WELL, PUMP)["status"] == "stale"


@pytest.mark.parametrize("model", ["hagedorn_brown", "drift_flux"])
def test_alternative_fit_reloads_only_with_matching_hydraulics(monkeypatch, model):
    from woffl.flow.hydraulics import physics_model
    rec = {**record(), "h": model, "m": physics_model(model)}
    install_record(monkeypatch, rec)
    active = pc.resolve(WELL, PUMP)
    assert active["hydraulics_model"] == model and active["coefficients"] == COEFS
    assert pc.resolve(WELL, PUMP, hydraulics_model=model)["status"] == "active"
    assert not pc.resolve(WELL, PUMP, hydraulics_model="beggs")["coefficients"]
    later = pc.resolve(WELL, {**PUMP, "date_set": "2026-09-10"})
    assert later["status"] == "stale" and not later["coefficients"]
    assert later["hydraulics_model"] == model  # keep well model, discard installation losses
    install_record(monkeypatch, record())
    assert not pc.resolve(WELL, PUMP, hydraulics_model=model)["coefficients"]


def test_model_version_and_hydraulics_pair_must_agree(monkeypatch):
    install_record(monkeypatch, {**record(), "h": "drift_flux"})
    assert pc.resolve(WELL, PUMP)["status"] == "stale"
    install_record(monkeypatch, {**record(), "h": "tulsa"})
    assert pc.resolve(WELL, PUMP)["status"] == "unavailable"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -.1, 999.])
def test_corrupt_coefficients_never_become_active(monkeypatch, bad):
    rec = record(); rec["k"][0] = bad
    install_record(monkeypatch, rec)
    result = pc.resolve(WELL, PUMP)
    assert result["status"] == "unavailable" and not result["coefficients"]


def test_fleet_calibration_read_is_one_select(monkeypatch):
    calls = []
    monkeypatch.setattr(pc.history, "execute_query", lambda sql: calls.append(sql) or pd.DataFrame())
    pc.resolve(WELL, PUMP)
    pc.resolve("MPB-28", PUMP)
    assert len(calls) == 1 and calls[0].startswith("SELECT ")


@pytest.fixture()
def save_case(monkeypatch):
    from server.services import datasources, ipr
    from woffl.assembly import databricks_client
    fit = {**COEFS, "fnz": COEFS["nozzle_area_factor"], "rms_bhp_psi": 76.,
           "rms_pf_pct": 61.5, "n_used": 19, "railed": ["ken"], "implied_beta": .268}
    result = {"well": WELL, "pump": "13C", "era_start": PUMP["date_set"],
              "physics_model": MODEL_VERSION, "fit": fit, "mined_beta": .062}
    monkeypatch.setattr(pc.jobs, "get", lambda *a, **kw: {"status": "done", "result": result})
    monkeypatch.setattr(databricks_client, "fetch_jp_history", lambda: pd.DataFrame([{
        "Well Name": WELL, "Nozzle Number": 13, "Throat Ratio": "C",
        "Date Set": pd.Timestamp(PUMP["date_set"]), "Tubing Diameter": 4.5,
    }]))
    monkeypatch.setattr(pc.history, "resolve_entry_user", lambda: "engineer@example.com")
    rows, evicted = [], []
    monkeypatch.setattr(pc.history, "push_eng_comment", lambda *a, **kw: rows.append((a, kw)) or 1)
    monkeypatch.setattr(ipr, "_invalidate_after_write", lambda well: evicted.append(well))
    return result, rows, evicted


def test_save_is_one_atomic_installation_record_with_quality(save_case):
    result, rows, evicted = save_case
    response = pc.save_fit(WELL, "job")
    assert len(rows) == 1 and evicted == [WELL]
    args, kwargs = rows[0]
    assert args[0] == WELL and args[2] == "engineer@example.com"
    assert kwargs == {"context": pc.CONTEXT}
    assert len(args[3]) <= 500
    rec = pc.decode(args[3])
    assert rec["k"] == list(COEFS.values())  # full precision
    assert rec["q"]["pf"] == 61.5 and rec["q"]["bounds"] == ["ken"]
    assert "installation only" in response["message"]


@pytest.mark.parametrize("model", ["hagedorn_brown", "drift_flux"])
def test_alternative_model_saves_atomically_with_precise_coefficients(save_case, model):
    from woffl.flow.hydraulics import physics_model
    result, rows, evicted = save_case
    result.update(hydraulics_model=model, physics_model=physics_model(model))
    result["fit"].update(ken=.005123456789012345, kth=.38612345678901234,
                         kdi=.07212345678901234, fnz=1.0123456789012345,
                         rms_dbhp_psi=72.12345, railed=["ken", "kth", "kdi", "fnz"])
    pc.save_fit(WELL, "job")
    assert len(rows) == 1 and evicted == [WELL]
    text = rows[0][0][3]
    rec = pc.decode(text)
    assert len(text) <= 500
    assert (rec["h"], rec["m"]) == (model, physics_model(model))
    assert rec["k"] == [result["fit"][k] for k in ("ken", "kth", "kdi", "fnz")]


def test_save_rejects_mismatched_hydraulics_version_before_writing(save_case):
    result, rows, evicted = save_case
    result["hydraulics_model"] = "drift_flux"
    with pytest.raises(ValueError, match="physics model"):
        pc.save_fit(WELL, "job")
    assert rows == [] and evicted == []


def test_save_reads_fresh_tracker_while_cached_refresh_is_in_flight(save_case):
    from server.services import datasources

    _, rows, _ = save_case
    cached = datasources._jp_history_databricks
    cached.cache_prime(pd.DataFrame())  # deliberately unusable old cache data
    key = ((), ())
    assert cached._cache.try_begin_refresh(key)
    try:
        pc.save_fit(WELL, "job")
        assert len(rows) == 1
    finally:
        cached._cache.end_refresh(key)


def test_save_fresh_tracker_failure_never_uses_old_cache(save_case, monkeypatch):
    from server.services import datasources
    from woffl.assembly import databricks_client

    _, rows, evicted = save_case
    datasources._jp_history_databricks()  # populate a valid but now old snapshot
    def fail(): raise RuntimeError("tracker unavailable")
    monkeypatch.setattr(databricks_client, "fetch_jp_history", fail)
    with pytest.raises(ValueError, match="Could not verify"):
        pc.save_fit(WELL, "job")
    assert not rows and not evicted


@pytest.mark.parametrize("key,value", [("well", "MPB-28"), ("era_start", "2026-09-01"),
                                      ("pump", "14C"), ("physics_model", "old")])
def test_save_refuses_stale_or_foreign_fit_before_any_write(save_case, key, value):
    result, rows, evicted = save_case
    result[key] = value
    with pytest.raises(ValueError): pc.save_fit(WELL, "job")
    assert rows == [] and evicted == []


def test_pump_save_gate_blocks_before_job_or_warehouse(monkeypatch):
    from server.main import app
    def blocked(*a, **kw): raise AssertionError("Save gate failed")
    monkeypatch.setattr(pc, "save_fit", blocked)
    response = TestClient(app).post(f"/api/wells/{WELL}/pump-calibration", json={"job_id": "job"})
    assert response.status_code == 403


def test_catalog_replacements_do_not_inherit_any_fitted_pump_coefficient():
    pumps = scoped_pumps(["13", "14"], ["C"], ("13", "C"), COEFS)
    assert [(p.noz_no, p.pump_state) for p in pumps] == [("13", "installed"), ("13", "replacement"), ("14", "replacement")]
    assert pumps[0].ken == .005
    assert pumps[0].anz / JetPump("13", "C").anz == pytest.approx(1.01)
    for p in pumps[1:]:
        assert (p.ken, p.kth, p.kdi) == (.03, .3, .4)
        assert p.anz == JetPump(p.noz_no, p.rat_ar).anz
    assert all(p.pump_state == "replacement" for p in scoped_pumps(["13"], ["C"], None, COEFS))


def allocation(installed_oil=100., replacement_oil=200.):
    cfg = WellConfig(WELL, 1700., 70., 4065., pump_calibration_scoped=True,
                     installed_nozzle="13", installed_throat="C")
    opt = NetworkOptimizer([cfg], PowerFluidConstraint(2000., 3000.), ["13"], ["C"], marginal_watercut=1.)
    opt.water_price = 0.
    rows = []
    for state, oil in [("installed", installed_oil), ("replacement", replacement_oil)]:
        rows.append(dict(nozzle="13", throat="C", pump_state=state, qoil_std=oil, lift_wat=1000.,
            form_wat=100., totl_wat=1100., psu_solv=500., sonic_status=False, mach_te=.2,
            molwr=.1, motwr=.1, semi=True, error="na"))
    opt.batch_results[WELL] = SimpleNamespace(wellname=WELL, df=pd.DataFrame(rows))
    return opt


@pytest.mark.parametrize("engine", [milp_optimization, mckp_optimization])
@pytest.mark.parametrize("installed,replacement,state", [(100., 200., "replacement"), (200., 100., "installed"), (100., 100., "installed")])
def test_optimizer_distinguishes_same_size_hardware_and_keeps_identical_pump(engine, installed, replacement, state):
    opt = allocation(installed, replacement)
    result = engine(opt)
    assert len(result) == 1
    assert result[0].pump_state == state
    assert result[0].predicted_oil_rate == max(installed, replacement)
    assert opt.get_pump_performance(WELL, "13", "C")["oil_rate"] == installed
    assert opt.get_pump_performance(WELL, "13", "C", "replacement")["oil_rate"] == replacement


def test_explicit_replacement_api_uses_reference_pump_preserving_well_inputs():
    sp = schemas.SimParams(pump_state="replacement", **COEFS, form_wc=.74, qwf=1521, pres=1054)
    assert (sp.ken, sp.kth, sp.kdi, sp.nozzle_area_factor) == (.03, .3, .4, 1.)
    assert (sp.form_wc, sp.qwf, sp.pres) == (.74, 1521, 1054)


def test_fixed_scenario_same_size_replacement_uses_its_own_performance():
    from woffl.gui.pad_optimize import _score_fixed_choices, _same_hardware
    opt = allocation()
    rows, pf, oil = _score_fixed_choices(opt, opt.wells, {WELL: ("13", "C", "replacement")}, {}, {})
    assert oil == 200. and rows[0]["pump_state"] == "replacement"
    assert not _same_hardware(("13", "C"), ("13", "C", "replacement"))
    assert _same_hardware(("13", "C"), ("13", "C", "installed"))


def test_single_point_save_preserves_the_area_used_in_its_fit(save_case):
    result, rows, _ = save_case
    result.update(fit=None, single={"ken": .04, "kth": .35, "kdi": .4, "match_quality": "good"},
                  current={"nozzle_area_factor": 1.12})
    pc.save_fit(WELL, "job")
    rec = pc.decode(rows[0][0][3])
    assert rec["k"][3] == 1.12 and rec["q"]["provisional"] is True


def test_unidentified_single_point_fit_cannot_be_saved(save_case):
    result, rows, _ = save_case
    result.update(fit=None, single={"ken": .04, "kth": .35, "kdi": .4, "match_quality": "pinned"})
    with pytest.raises(ValueError): pc.save_fit(WELL, "job")
    assert not rows


def test_installation_time_is_not_lost_when_matching_same_day_changeouts(save_case, monkeypatch):
    import woffl.assembly.jp_history as tracker
    result, rows, _ = save_case
    monkeypatch.setattr(tracker, "get_current_pump", lambda *a: {**PUMP, "date_set": "2026-08-10T15:30:00"})
    result["installation_date_set"] = "2026-08-10T10:30:00"
    with pytest.raises(ValueError): pc.save_fit(WELL, "job")
    assert not rows
    result["installation_date_set"] = "2026-08-10T15:30:00"
    pc.save_fit(WELL, "job")
    assert len(rows) == 1


@pytest.mark.parametrize("model", ["beggs", "hagedorn_brown", "drift_flux"])
def test_real_batch_installed_and_clean_predictions_match_single_solves(model):
    from server.services.solve import run_batch, solve_single, _pf_point
    sp = schemas.SimParams(nozzle_no="13", area_ratio="C", hydraulics_model=model, **COEFS,
                          nozzle_batch_options=["13"], throat_batch_options=["C"])
    rows = run_batch("Custom", sp)["rows"]
    pf_rows = _pf_point("Custom", sp.model_dump_json(), sp.ppf_surf)
    assert len(rows) == 2 and len(pf_rows) == 2
    for state in ("installed", "replacement"):
        params = schemas.SimParams(**{**sp.model_dump(), "pump_state": state})
        single = solve_single("Custom", params)
        row = next(r for r in rows if r["pump_state"] == state)
        pressure_row = pf_rows[pf_rows.pump_state == state].iloc[0]
        assert row["psu_solv"] == pytest.approx(single["psu"], abs=1e-8)
        assert row["qoil_std"] == pytest.approx(single["qoil_std"], abs=1e-8)
        assert pressure_row.psu_solv == pytest.approx(single["psu"], abs=1e-8)


def test_cfp_surface_retains_clean_same_size_changeout_separately(monkeypatch):
    from woffl.gui import cfp_moves, cfp_optimize
    opt = allocation()
    cfg = opt.wells[0]
    cfg.pad = "B"
    def batch(self, **kw): self.batch_results = opt.batch_results
    monkeypatch.setattr(NetworkOptimizer, "run_all_batch_simulations", batch)
    monkeypatch.setattr(cfp_optimize, "delivered_by_pad", lambda *a, **kw: ({"B": 3000.}, []))
    surfaces = cfp_moves.build_response_surfaces({"B": [cfg]}, {WELL: True}, {WELL: ("13", "C")},
        object(), p_grid=[2500., 2700.], nozzles=["13"], throats=["C"], p0=2600., c_pad_pf_psi=3000.)
    ws = surfaces.wells[WELL]
    assert ws.current == "13C"
    assert ws.options["13C"]["oil"] == [100., 100.]
    assert ws.options["13C (clean)"]["oil"] == [200., 200.]
    assert surfaces.baseline_choices()[WELL] == "13C"


def test_save_endpoint_attributes_the_engineer_and_keeps_pump_record_separate(save_case, monkeypatch):
    from server.main import app
    from server import identity
    from woffl.gui import ipr_anchor
    _, rows, _ = save_case
    monkeypatch.setattr(ipr_anchor, "writes_enabled", lambda: True)
    monkeypatch.setattr(pc.history, "resolve_entry_user", identity._provider)
    response = TestClient(app).post(f"/api/wells/{WELL}/pump-calibration", json={"job_id": "job"},
                                   headers={"X-Forwarded-Email": "reviewer@example.com"})
    assert response.status_code == 200
    assert rows[0][0][2] == "reviewer@example.com"


def test_failed_calibration_save_does_not_invalidate_or_claim_success(save_case, monkeypatch):
    _, rows, evicted = save_case
    def fail(*a, **kw): raise RuntimeError("write did not complete")
    monkeypatch.setattr(pc.history, "push_eng_comment", fail)
    with pytest.raises(RuntimeError): pc.save_fit(WELL, "job")
    assert not rows and not evicted


def test_readiness_board_uses_installation_scope_instead_of_legacy_rows(monkeypatch):
    from server.services import ipr, wells
    from woffl.gui import ipr_anchor
    monkeypatch.setattr(wells, "list_wells", lambda: {"wells": [{"name": WELL, "pad": "E"}]})
    monkeypatch.setattr(ipr_anchor, "warm_saved_ipr_cache", lambda: 0)
    monkeypatch.setattr(ipr_anchor, "load_saved_ipr", lambda well: {"friction": COEFS})
    monkeypatch.setattr(pc, "resolve_current", lambda *a: {"status": "legacy", "coefficients": {}})
    row = ipr._pad_fit("E", ())["wells"][0]
    assert not row["has_friction"] and row["friction_keys"] == []
    ipr._pad_fit.cache_clear()
    monkeypatch.setattr(pc, "resolve_current", lambda *a: {"status": "active", "coefficients": COEFS})
    row = ipr._pad_fit("E", ())["wells"][0]
    assert row["has_friction"] and set(row["friction_keys"]) == set(COEFS)
