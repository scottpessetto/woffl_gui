"""Event-calibration job: payload assembly, refusal chain, fail-soft mined
beta, and kind/status plumbing through the optimize run registry.

All synthetic - hydration, points builder, fitter and evidence are
monkeypatched; nothing here touches a warehouse.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import server.services.calibration_points as cal_points
import server.services.event_calibration as ec
import server.services.optimizer_runs as runs
from server.main import app

WELL = "MPM-01"

# ---------------------------------------------------------------------------
# Synthetic building blocks
# ---------------------------------------------------------------------------


def _cfg(well=WELL, ken=0.05, kth=0.30, kdi=0.30, res_pres=1700.0, surf_pres=210.0):
    return SimpleNamespace(
        well_name=well,
        ken_well=ken,
        kth_well=kth,
        kdi_well=kdi,
        res_pres=res_pres,
        surf_pres=surf_pres,
        form_temp=140.0,
    )


def _built(well=WELL, refusal=None, n_points=12):
    return {
        "well": well,
        "pump": {"nozzle": "12", "throat": "B", "date_set": "2026-06-01"},
        "era_start": "2026-06-01",
        "points": [{"ppf": 3000.0 + 40.0 * i} for i in range(n_points)],
        "ppf_spread": 440.0,
        "n_daily": 9,
        "n_test": 3,
        "refusal": refusal,
    }


def _fit_result(refusal=None):
    return SimpleNamespace(
        best_ken=0.08,
        best_kth=0.22,
        best_kdi=0.31,
        best_fnz=1.12,
        best_mach_crit=1.0,
        rms_bhp_psi=28.0,
        rms_pf_pct=3.1,
        rms_dbhp_psi=14.0,
        n_used=11,
        n_dropped=1,
        bounded=False,
        railed=["ken"],
        implied_beta=0.041,
        per_point=[],
        refusal=refusal,
        iterations=200,
        message="ok" if refusal is None else f"refused: {refusal}",
    )


def _ev_row(beta=0.062, source="well"):
    return {
        "floor": 380.0,
        "psu_ref": 520.0,
        "beta": beta,
        "beta_source": source,
        "n_days": 90,
        "n_pairs": 12,
        "window": ["2026-05-01", "2026-08-01"],
    }


def _test_row(bhp=800.0, whp=95.0, pf_press=3100.0, date="2026-08-01"):
    """One tests_json row - only the keys the fallback leg reads."""
    return {"date": date, "bhp": bhp, "whp": whp, "pf_press": pf_press}


def _single_result(match_quality="good", message=None):
    """FricCalibrationResult stand-in - only the single_payload fields."""
    return SimpleNamespace(
        best_ken=0.07,
        best_kth=0.25,
        best_kdi=0.33,
        best_modeled_bhp=812.0,
        target_bhp=800.0,
        match_quality=match_quality,
        message=message,
    )


@pytest.fixture()
def client(monkeypatch) -> TestClient:
    """Happy-path wiring; individual tests override the piece they break."""
    import woffl.gui.fric_calibration as fc

    monkeypatch.setattr(
        runs, "_build_configs", lambda pads, offline, future, note, prov=None: [_cfg()]
    )
    monkeypatch.setattr(
        runs, "_current_and_tests", lambda wells: ({WELL: ("12", "B")}, {})
    )
    monkeypatch.setattr(
        cal_points, "pad_points", lambda wells, **kw: {WELL: _built()}
    )
    monkeypatch.setattr(
        fc, "calibrate_multipoint", lambda cfg, nz, th, built, **kw: _fit_result()
    )
    monkeypatch.setattr(
        ec.evidence_svc, "pad_evidence", lambda names, res_pres=None: {WELL: _ev_row()}
    )
    # No test with a measured BHP by default: the single-point fallback is
    # impossible unless a test opts in, so refusal tests keep old behavior.
    monkeypatch.setattr(ec.tests_svc, "tests_json", lambda well, months, cap=0: [])
    return TestClient(app)


def _start(client: TestClient, well=WELL) -> str:
    resp = client.post("/api/optimize/event-calibration", json={"well": well})
    assert resp.status_code == 200
    return resp.json()["job_id"]


def _wait_done(client: TestClient, job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/api/optimize/run/{job_id}").json()
        if body["status"] != "running":
            return body
        time.sleep(0.05)
    raise AssertionError("job did not settle in time")


# ---------------------------------------------------------------------------
# Happy path - the contract payload, verbatim
# ---------------------------------------------------------------------------


def test_happy_path_payload_and_kind(client):
    body = _wait_done(client, _start(client))
    assert body["kind"] == "event_cal"
    assert body["status"] == "done"

    r = body["result"]
    assert r["well"] == WELL
    assert r["pump"] == "12B"
    assert r["era_start"] == "2026-06-01"
    assert r["n_daily"] == 9
    assert r["n_test"] == 3
    assert r["ppf_spread"] == 440.0
    assert r["refusal"] is None
    assert r["method"] == "event"
    assert r["fallback_reason"] is None
    assert r["single"] is None
    assert r["mined_beta"] == 0.062
    assert r["mined_beta_source"] == "well"
    assert r["current"] == {"ken": 0.05, "kth": 0.30, "kdi": 0.30}

    fit = r["fit"]
    assert fit == {
        "ken": 0.08,
        "kth": 0.22,
        "kdi": 0.31,
        "fnz": 1.12,
        "mach_crit": 1.0,
        "rms_bhp_psi": 28.0,
        "rms_pf_pct": 3.1,
        "rms_dbhp_psi": 14.0,
        "n_used": 11,
        "n_dropped": 1,
        "railed": ["ken"],
        "implied_beta": 0.041,
        "message": "ok",
    }


# ---------------------------------------------------------------------------
# Refusal chain
# ---------------------------------------------------------------------------


def test_builder_refusal_skips_fitter(client, monkeypatch):
    import woffl.gui.fric_calibration as fc

    refusal = "young pump era - not identifiable yet"
    monkeypatch.setattr(
        cal_points, "pad_points", lambda wells, **kw: {WELL: _built(refusal=refusal)}
    )

    def boom(*a, **kw):
        raise AssertionError("fitter must not run on a builder refusal")

    monkeypatch.setattr(fc, "calibrate_multipoint", boom)

    body = _wait_done(client, _start(client))
    assert body["status"] == "done"
    r = body["result"]
    # Both legs impossible (fixture has no test with a BHP): the honest
    # event refusal stands, exactly as before the unified button.
    assert r["method"] == "event"
    assert r["refusal"] == refusal
    assert r["fallback_reason"] is None
    assert r["single"] is None
    assert r["fit"] is None
    # Builder metadata still reports - the engineer sees WHY it refused.
    assert r["n_daily"] == 9 and r["n_test"] == 3


def test_fitter_refusal_yields_null_fit(client, monkeypatch):
    import woffl.gui.fric_calibration as fc

    refusal = "6 of 12 points failed to solve"
    monkeypatch.setattr(
        fc, "calibrate_multipoint", lambda cfg, nz, th, built, **kw: _fit_result(refusal)
    )
    # A test BHP exists, but a FITTER refusal is not a young-era signal -
    # the fallback only rides builder refusals.
    monkeypatch.setattr(
        ec.tests_svc, "tests_json", lambda well, months, cap=0: [_test_row()]
    )

    body = _wait_done(client, _start(client))
    assert body["status"] == "done"
    r = body["result"]
    assert r["refusal"] == refusal
    assert r["fit"] is None
    assert r["method"] == "event"
    assert r["single"] is None
    assert r["mined_beta"] == 0.062  # evidence still mined


def test_missing_builder_result_is_no_calibration_data(client, monkeypatch):
    monkeypatch.setattr(cal_points, "pad_points", lambda wells, **kw: {})
    body = _wait_done(client, _start(client))
    assert body["status"] == "done"
    r = body["result"]
    assert r["refusal"] == "no calibration data"
    assert r["fit"] is None
    assert r["method"] == "event" and r["single"] is None
    assert r["era_start"] is None
    assert r["n_daily"] == 0 and r["n_test"] == 0 and r["ppf_spread"] == 0.0


# ---------------------------------------------------------------------------
# Single-point fallback (young era)
# ---------------------------------------------------------------------------


def test_builder_refusal_falls_back_to_single_point(client, monkeypatch):
    import woffl.gui.fric_calibration as fc

    refusal = "young pump era - not identifiable yet"
    monkeypatch.setattr(
        cal_points, "pad_points", lambda wells, **kw: {WELL: _built(refusal=refusal)}
    )
    monkeypatch.setattr(
        ec.tests_svc, "tests_json", lambda well, months, cap=0: [_test_row()]
    )
    monkeypatch.setattr(
        fc, "_build_well_objects", lambda cfg: ("wb", "wp", "ipr", "mix", "pf")
    )
    seen: dict = {}

    def fake_single(**kw):
        seen.update(kw)
        return _single_result()

    monkeypatch.setattr(fc, "calibrate_friction_coefs", fake_single)

    body = _wait_done(client, _start(client))
    assert body["status"] == "done"
    r = body["result"]
    assert r["method"] == "single_point"
    assert r["fallback_reason"] == refusal
    assert r["refusal"] is None
    assert r["fit"] is None
    assert r["single"] == {
        "ken": 0.07,
        "kth": 0.25,
        "kdi": 0.33,
        "modeled_bhp": 812.0,
        "target_bhp": 800.0,
        "match_quality": "good",
        "message": None,
    }
    # The fallback fed the /calibrate mechanics from the hydrated config and
    # the latest test: its measured BHP/WHP/PF pressure, the saved ken seed.
    assert seen["target_bhp"] == 800.0
    assert seen["pwh"] == 95.0
    assert seen["ppf_surf"] == 3100.0
    assert seen["ken"] == 0.05
    assert seen["nozzle"] == "12" and seen["throat"] == "B"
    assert seen["knz"] == 0.01


def test_fallback_skips_tests_without_bhp(client, monkeypatch):
    """Rows without a measured BHP never qualify as the fallback target."""
    monkeypatch.setattr(
        cal_points,
        "pad_points",
        lambda wells, **kw: {WELL: _built(refusal="young pump era")},
    )
    monkeypatch.setattr(
        ec.tests_svc,
        "tests_json",
        lambda well, months, cap=0: [_test_row(bhp=None), _test_row(bhp=None)],
    )
    body = _wait_done(client, _start(client))
    r = body["result"]
    assert r["method"] == "event"
    assert r["refusal"] == "young pump era"
    assert r["single"] is None


def test_fallback_crash_keeps_event_refusal(client, monkeypatch):
    """A blown-up fallback leg is fail-soft: the refusal reports as before."""
    import woffl.gui.fric_calibration as fc

    refusal = "young pump era - not identifiable yet"
    monkeypatch.setattr(
        cal_points, "pad_points", lambda wells, **kw: {WELL: _built(refusal=refusal)}
    )
    monkeypatch.setattr(
        ec.tests_svc, "tests_json", lambda well, months, cap=0: [_test_row()]
    )

    def boom(cfg):
        raise RuntimeError("survey fetch died")

    monkeypatch.setattr(fc, "_build_well_objects", boom)

    body = _wait_done(client, _start(client))
    assert body["status"] == "done"
    r = body["result"]
    assert r["method"] == "event"
    assert r["refusal"] == refusal
    assert r["fallback_reason"] is None and r["single"] is None


# ---------------------------------------------------------------------------
# Missing well - hydration produced no config
# ---------------------------------------------------------------------------


def test_missing_well_errors(client, monkeypatch):
    monkeypatch.setattr(
        runs, "_build_configs", lambda pads, offline, future, note, prov=None: []
    )
    body = _wait_done(client, _start(client))
    assert body["status"] == "error"
    assert "no usable saved fit" in body["error"]
    assert body["result"] is None


# ---------------------------------------------------------------------------
# Fail-soft mined beta
# ---------------------------------------------------------------------------


def test_evidence_failure_leaves_mined_beta_null(client, monkeypatch):
    def dead_warehouse(names, res_pres=None):
        raise RuntimeError("warehouse offline")

    monkeypatch.setattr(ec.evidence_svc, "pad_evidence", dead_warehouse)
    body = _wait_done(client, _start(client))
    assert body["status"] == "done"
    r = body["result"]
    assert r["mined_beta"] is None
    assert r["mined_beta_source"] is None
    assert r["fit"] is not None  # the fit itself is unaffected


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


def test_empty_well_is_422(client):
    assert (
        client.post("/api/optimize/event-calibration", json={"well": ""}).status_code
        == 422
    )


def test_job_streams_fitter_progress_into_envelope(client, monkeypatch):
    """The fitter's per-pass line must land in job["progress"] as it runs -
    the poller showed one frozen string for the whole 3-minute MPE-35 fit."""
    import woffl.gui.fric_calibration as fc

    job = {"progress": "start"}
    seen = {}

    def fake_mp(cfg, nz, th, built, progress=None, **kw):
        assert callable(progress)
        progress("fitting 24 points - pass 1 (evaluation 10)")
        seen["after_callback"] = job["progress"]
        return _fit_result()

    monkeypatch.setattr(fc, "calibrate_multipoint", fake_mp)
    payload = ec._run_event_calibration_job(job, WELL)

    assert seen["after_callback"] == "fitting 24 points - pass 1 (evaluation 10)"
    assert payload["method"] == "event"
    # After the fit the envelope names the evidence step, not the last pass.
    assert "field evidence" in job["progress"]
