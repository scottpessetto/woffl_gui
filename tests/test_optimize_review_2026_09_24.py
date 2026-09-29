"""Optimization review 2026-09-24: server contract fixes.

Pump identity normalization, request validation, job queue/prune safety,
readable job errors, E-Pad result provenance, donor provenance for planned
wells and CFP exclusion of wells that cannot anchor. Engines are faked; the
engines' own fixes are pinned by their suites.
"""

from __future__ import annotations

import math
import threading
import time
from decimal import Decimal
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import server.services.optimizer_runs as runs
import server.services.wells as wells_svc
from server import jobs, schemas
from server.main import app

_UNIVERSE = {
    "wells": [{"name": "MPM-01", "pad": "M"}, {"name": "MPM-02", "pad": "M"}, {"name": "MPB-28", "pad": "B"}],
    "source": "databricks",
}
_SEEDS = {"pres": 1700.0, "qwf": 900.0, "pwf": 600.0, "form_wc": 0.7}


@pytest.fixture(autouse=True)
def _offline_inputs(monkeypatch):
    from server.services import ipr as ipr_svc

    monkeypatch.setattr(ipr_svc, "prime_saved_ipr", lambda: None)
    monkeypatch.setattr(wells_svc, "list_wells", lambda: _UNIVERSE)


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


# -- pump identity --------------------------------------------------------------


@pytest.mark.parametrize("raw, expected", [
    (("12", "B"), ("12", "B")),
    (("12", "b"), ("12", "B")),        # tracker lowercase throat
    ((" 7 ", "B"), ("7", "B")),         # catalog size outside the sidebar list
    (("12.0", "C"), ("12", "C")),       # float-formatted tracker nozzle
    ((12, "a"), ("12", "A")),
    (("12", "0.39"), None),             # legacy numeric throat entry
    (("99", "B"), None),                # beyond the catalog
    (("", "B"), None),
    ((None, None), None),
    ((float("nan"), "B"), None),
])
def test_one_pump_identity_for_every_read(raw, expected):
    assert runs._pump_identity(*raw) == expected


def test_tracker_and_model_identity_agree_for_lowercase_and_size_seven(monkeypatch):
    """Both reads of the installed pump normalize the same way, and the
    model's own identity wins, so the current-pump baseline is not dropped."""
    from woffl.assembly import jp_history
    from server.services import datasources, tests as tests_svc

    monkeypatch.setattr(datasources, "jp_history_safe", lambda: (pd.DataFrame(), "excel_fallback"))
    monkeypatch.setattr(jp_history, "get_current_pump",
                        lambda df, well: {"nozzle_no": "7", "throat_ratio": "b"})
    monkeypatch.setattr(tests_svc, "tests_for_well", lambda well, months, cap: None)
    current, _rates = runs._current_and_tests(["MPI-15"])
    assert current == {"MPI-15": ("7", "B")}
    cfg = SimpleNamespace(well_name="MPI-15", installed_nozzle="7", installed_throat="B")
    runs._installed_current([cfg], current)
    assert current["MPI-15"] == (cfg.installed_nozzle, cfg.installed_throat)


def test_hydration_keeps_an_out_of_list_installed_pump(monkeypatch):
    ctx = {"seeds": dict(_SEEDS), "pump": {"nozzle_no": "7", "throat_ratio": "B"}}
    monkeypatch.setattr(wells_svc, "well_context", lambda well, months, cap: ctx)
    notes: list[str] = []
    configs = runs._build_configs(["M"], set(), [], notes, only={"MPM-01"})
    assert [(c.installed_nozzle, c.installed_throat) for c in configs] == [("7", "B")]


# -- readable failures ----------------------------------------------------------


def test_unknown_donor_is_named_not_a_raw_key_error(monkeypatch):
    def ctx(well, months, cap):
        if well not in {"MPM-01", "MPM-02"}:
            raise KeyError(well)
        return {"seeds": dict(_SEEDS)}

    monkeypatch.setattr(wells_svc, "well_context", ctx)
    notes: list[str] = []
    runs._build_configs(["M"], set(), [schemas.FutureWellSpec(name="NEW", match="MPI-9O1")], notes)
    assert "MPI-9O1: unknown well (not in the well characteristics list)" in notes
    assert not any("('MPI-9O1')" in n for n in notes)


def test_water_cut_refusal_prints_three_decimals():
    with pytest.raises(ValueError, match=r"water cut 0\.995 >= 0\.99"):
        runs._config_from_seeds("MPM-01", "M", {**_SEEDS, "form_wc": 0.995})


def test_job_failure_carries_the_per_well_reasons():
    notes = ["MPB-28: invalid model (water cut 0.995 >= 0.99 - not modelable)", "Manual reference discharge: 2,792 psi."]
    err = runs._failure("no active wells with usable saved fits on pads B", notes)
    assert "MPB-28: invalid model (water cut 0.995" in str(err)
    assert "Manual reference" not in str(err)
    assert str(runs._failure("plain", [])) == "plain"


# -- request validation ---------------------------------------------------------


@pytest.mark.parametrize("body, message", [
    ({"kind": "pad", "pad": "I", "nozzles": []}, "at least one nozzle"),
    ({"kind": "pad", "pad": "I", "nozzles": ["99"]}, "Unknown pump sizes: 99"),
    ({"kind": "pad", "pad": "I", "throats": ["Z"]}, "Unknown pump sizes: Z"),
    ({"kind": "pad", "pad": "S", "strategy": "choke"}, "no hold-pumps choke plan"),
    ({"kind": "pad", "pad": "M", "future": [{"name": "N1", "match": "MPM-01", "pad": "I"}]}, "outside this run"),
    ({"kind": "cfp", "cfp_pad_pf_psi": {"B": 3500}}, "cannot exceed the reference discharge"),
    ({"kind": "pad", "pad": "I", "n_pumps": 2}, "fixed pump train"),
    ({"kind": "pad", "pad": "S", "n_pumps": 1}, "offers 3, 2 pumps online"),
    ({"kind": "pad", "pad": "M", "future": [{"name": "N1", "match": "MPX-404"}]}, "Unknown donor well: MPX-404"),
])
def test_requests_that_cannot_run_are_rejected_before_a_job_starts(client, body, message):
    r = client.post("/api/optimize/run", json=body)
    assert r.status_code == 422
    assert message in str(r.json())


def test_choke_runs_ignore_the_replacement_grid():
    req = schemas.OptimizeRunRequest(kind="pad", pad="M", strategy="choke", nozzles=[])
    assert req.nozzles == []


# -- job registry ---------------------------------------------------------------


def test_a_job_that_is_settling_is_never_pruned(monkeypatch):
    """Status flips before the settle time lands; a 0.0 settle time must not
    read as 'settled at boot' (a just-finished result was pruned)."""
    job = {"kind": "pad", "status": "done", "settled_mono": 0.0, "started_mono": 0.0}
    monkeypatch.setitem(jobs._JOBS, "settling", job)
    jobs._prune_jobs()
    assert "settling" in jobs._JOBS


def test_the_wait_queue_is_bounded_and_reports_its_position(monkeypatch):
    monkeypatch.setattr(jobs, "_JOB_SLOTS", threading.BoundedSemaphore(1))
    monkeypatch.setattr(jobs, "_MAX_QUEUED", 2)
    release = threading.Event()
    ids = []
    try:
        ids.append(jobs.start("pad", lambda job: release.wait(5) and {}))
        time.sleep(0.2)
        ids.append(jobs.start("pad", lambda job: {}))
        ids.append(jobs.start("pad", lambda job: {}))
        time.sleep(0.4)
        assert jobs.get(ids[1])["progress"].startswith("queued - 1 job ahead")
        assert jobs.get(ids[2])["progress"].startswith("queued - 2 jobs ahead")
        with pytest.raises(jobs.JobQueueFull):
            jobs.start("pad", lambda job: {})
    finally:
        release.set()
        for jid in ids:
            deadline = time.monotonic() + 5
            while jobs.get(jid)["status"] == "running" and time.monotonic() < deadline:
                time.sleep(0.05)


def test_a_full_queue_is_a_429_with_a_reason(client, monkeypatch):
    def full(*args, **kwargs):
        raise jobs.JobQueueFull("6 jobs are already waiting for the compute slot")

    monkeypatch.setattr(runs, "start_run", full)
    r = client.post("/api/optimize/run", json={"kind": "pad", "pad": "M"})
    assert r.status_code == 429
    assert "already waiting" in r.json()["detail"]["message"]


# -- serialization --------------------------------------------------------------


def test_plain_turns_arrays_and_scalars_into_json_values():
    out = runs._plain({
        "arr": np.array([1.0, np.nan, 3.0]), "series": pd.Series([1, 2]), "dec": Decimal("1.25"),
        "nat": pd.NaT, "na": pd.NA, "dt64": np.datetime64("2026-09-24"), "inf": math.inf,
    })
    assert out["arr"] == [1.0, None, 3.0]
    assert out["series"] == [1, 2]
    assert out["dec"] == 1.25
    assert out["nat"] is None and out["na"] is None and out["inf"] is None
    assert out["dt64"].startswith("2026-09-24")


# -- provenance -----------------------------------------------------------------


def test_planned_well_rows_name_their_donor(monkeypatch):
    monkeypatch.setattr(wells_svc, "well_context", lambda well, months, cap: {
        "seeds": dict(_SEEDS), "ipr_source": "saved", "ipr_r2": 0.9,
        "pump_calibration": {"status": "active"}})
    notes: list[str] = []
    prov: dict = {}
    runs._build_configs(["M"], set(), [schemas.FutureWellSpec(name="NEW", match="MPM-01")], notes, prov)
    assert prov["NEW"]["donor"] == "MPM-01"
    assert prov["NEW"]["ipr_source"] == "saved"
    # A planned well gets a clean pump, never the donor's installed fit.
    assert prov["NEW"]["has_friction"] is False and prov["NEW"]["pump_calibration"] is None


def test_e_pad_results_record_the_booster_they_ran(monkeypatch):
    import woffl.gui.pad_optimize as pad_optimize

    monkeypatch.setattr(wells_svc, "well_context", lambda well, months, cap: {"seeds": dict(_SEEDS)})
    monkeypatch.setattr(wells_svc, "list_wells", lambda: {"wells": [{"name": "MPE-01", "pad": "E"}], "source": "x"})
    monkeypatch.setattr(runs, "_current_and_tests", lambda names: ({}, {}))

    def fake_run(configs, plant, *args, **kw):
        return [], object(), {"header_psi": None, "feasible": True}

    monkeypatch.setattr(pad_optimize, "run_optimization", fake_run)
    req = schemas.OptimizeRunRequest(kind="pad", pad="E", e_pad_build="SN35000_18STG", e_pad_amp_limit_a=120.0)
    result = runs._run_pad_job({"cancel_event": None}, req)
    assert result["meta"]["e_pad"] == {"build": "SN35000_18STG", "suction_psi": 2704.0, "hz_max": 60.0,
                                       "max_header_psi": 3500.0, "amp_limit_a": 120.0}


# -- CFP --------------------------------------------------------------------------


def test_cfp_excludes_an_unanchorable_online_well_instead_of_failing(monkeypatch):
    import woffl.gui.cfp_moves as cfp_moves
    from server.services import datasources

    monkeypatch.setattr(wells_svc, "list_wells", lambda: {"wells": [
        {"name": "MPB-01", "pad": "B"}, {"name": "MPB-02", "pad": "B"}], "source": "x"})
    monkeypatch.setattr(wells_svc, "well_context", lambda well, months, cap: {"seeds": dict(_SEEDS)})
    monkeypatch.setattr(runs, "_current_and_tests", lambda names: ({n: ("12", "B") for n in names}, {}))
    monkeypatch.setattr(datasources, "pf_latest_safe", lambda: pd.DataFrame())

    def surfaces(pad_configs, online, current, plant, **kw):
        # MPB-02's current pump never converged at P0.
        return SimpleNamespace(p0=2792.0, wells={
            "MPB-01": SimpleNamespace(pad="B", online=True, current="12B", options={"12B": {}}),
            "MPB-02": SimpleNamespace(pad="B", online=True, current="12B", options={})})

    anchored = {}

    def anchor(s, psi_per_kbpd):
        anchored["wells"] = sorted(s.wells)
        return object()

    monkeypatch.setattr(cfp_moves, "build_response_surfaces", surfaces)
    monkeypatch.setattr(cfp_moves, "anchor", anchor)
    monkeypatch.setattr(cfp_moves, "moves_summary", lambda s, p, **kw: {
        "today": {"pressure": 2792.0}, "baseline": {"MPB-01": "12B"}, "plan": None, "singles": []})
    monkeypatch.setattr(cfp_moves, "option_at", lambda ws, label, p: (10.0, 100.0))
    result = runs._run_cfp_job({"cancel_event": None}, schemas.OptimizeRunRequest(kind="cfp", cfp_pads=["B"]))
    assert anchored["wells"] == ["MPB-01"]
    assert result["coverage"]["complete"] is False
    assert "MPB-02" in result["coverage"]["unaccounted_wells"]
    assert any("excluded" in n and "MPB-02" in n for n in result["notes"])
