"""POST /optimize/run job contract: request gating, saved-fit hydration
(offline exclusion + future-well donor cloning), job lifecycle, and result
serialization. The engines themselves (pad_optimize, cfp_moves) are pinned
by their own suites and by a live E2E; they are faked here.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

import server.services.optimizer_runs as runs
import server.services.wells as wells_svc
from server import schemas
from server.main import app

_UNIVERSE = {
    "wells": [
        {"name": "MPM-01", "pad": "M"},
        {"name": "MPM-02", "pad": "M"},
        {"name": "MPB-28", "pad": "B"},
    ],
    "source": "databricks",
}

_SEEDS = {
    "MPM-01": {"pres": 1700.0, "qwf": 900.0, "pwf": 600.0, "form_wc": 0.7, "ken": 0.05},
    "MPM-02": {"pres": 1600.0, "qwf": 800.0, "pwf": 500.0, "form_wc": 0.6},
    "MPB-28": {"pres": 1429.0, "qwf": 1731.0, "pwf": 1175.0, "form_wc": 0.82, "kth": 0.497},
}


class _FakeResult:
    def __init__(self, well: str):
        self.well_name = well
        self.recommended_nozzle = "12"
        self.recommended_throat = "B"
        self.allocated_power_fluid = 3000.0
        self.predicted_oil_rate = 250.0
        self.predicted_formation_water = 900.0
        self.suction_pressure = 1100.0
        self.marginal_oil_rate = 0.08
        self.sonic_status = False


@pytest.fixture()
def client(monkeypatch) -> TestClient:
    monkeypatch.setattr(wells_svc, "list_wells", lambda: _UNIVERSE)
    monkeypatch.setattr(
        wells_svc, "well_context", lambda well, months, cap: {"seeds": dict(_SEEDS[well])}
    )
    # current pumps + tests are cosmetic for the contract - keep them empty
    monkeypatch.setattr(runs, "_current_and_tests", lambda wells: ({}, {}))
    return TestClient(app)


def _wait_done(client: TestClient, job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/api/optimize/run/{job_id}").json()
        if body["status"] != "running":
            return body
        time.sleep(0.05)
    raise AssertionError("job did not settle in time")


def test_pad_kind_requires_pad(client):
    r = client.post("/api/optimize/run", json={"kind": "pad"})
    assert r.status_code == 422


def test_unknown_job_is_404(client):
    assert client.get("/api/optimize/run/nope").status_code == 404


@pytest.mark.parametrize("payload", [
    {"kind": "cfp", "p0_psi": 2890},
    {"kind": "cfp", "cfp_pad_pf_psi": {"B": 999}},
    {"kind": "pad", "pad": "E", "e_pad_suction_psi": 3600, "e_pad_max_header_psi": 3500},
    {"kind": "pad", "pad": "M", "required_wells": ["MPM-01"], "offline": ["MPM-01"]},
    {"kind": "pad", "pad": "M", "strategy": "choke", "future": [{"name": "NEW", "match": "MPM-01"}]},
    {"kind": "cfp", "future": [{"name": "NEW", "match": "MPM-01", "pad": "B"}, {"name": "new", "match": "MPM-01", "pad": "G"}]},
])
def test_incoherent_capacity_study_rejected_before_job(client, payload):
    assert client.post("/api/optimize/run", json=payload).status_code == 422


def test_future_name_cannot_replace_an_existing_well(client):
    req = schemas.OptimizeRunRequest(kind="pad", pad="M", future=[{"name": "mpm-01", "match": "MPB-28"}])
    with pytest.raises(ValueError, match="existing"):
        runs._run_pad_job({}, req)


def test_missing_required_model_is_not_silently_excluded(client):
    req = schemas.OptimizeRunRequest(kind="pad", pad="M", required_wells=["UNKNOWN"])
    with pytest.raises(ValueError, match="Required online.*UNKNOWN"):
        runs._run_pad_job({}, req)


@pytest.mark.parametrize("strategy", ["jpco", "choke"])
def test_required_future_and_planned_pump_reach_pad_engine(client, monkeypatch, strategy):
    import woffl.gui.pad_optimize as pad_optimize
    captured = {}

    def capture(configs, plant, *args, **kw):
        captured.update(kw)
        if strategy == "choke":
            captured["current"] = args[1]
        raise RuntimeError("captured engine request")

    monkeypatch.setattr(runs.evidence_svc, "pad_evidence", lambda *args: {})
    monkeypatch.setattr(pad_optimize, "run_optimization", capture)
    monkeypatch.setattr(pad_optimize, "run_choke_optimization", capture)
    req = schemas.OptimizeRunRequest(kind="pad", pad="M", strategy=strategy,
        required_wells=["MPM-01"], future=[{"name": "NEW", "match": "MPB-28",
        "require_online": True, "nozzle": "11", "throat": "c"}])
    with pytest.raises(RuntimeError, match="captured engine request"):
        runs._run_pad_job({}, req)
    assert captured["required_wells"] == {"MPM-01", "NEW"}
    if strategy == "choke":
        assert captured["current"]["NEW"] == ("11", "C")


def test_cfp_reference_grid_and_water_delta_use_one_baseline(client, monkeypatch):
    """The old pump may not solve at the new pressure; its before-water is
    evaluated at the reference, and no asynchronous live PF is substituted."""
    from types import SimpleNamespace
    import woffl.gui.cfp_moves as cfp_moves
    from server.services import datasources
    captured = {}
    monkeypatch.setattr(runs, "_current_and_tests", lambda names: ({n: ("12", "B") for n in names}, {}))

    def forbidden_live_pf():
        raise AssertionError("manual reference must not mix in live pad pressures")

    monkeypatch.setattr(datasources, "pf_latest_safe", forbidden_live_pf)

    def surfaces(pad_configs, online, current, plant, **kw):
        captured.update(kw)
        return SimpleNamespace(wells={"MPB-28": SimpleNamespace(pad="B", online=True)})

    def summary(surfaces, plant, **kw):
        return {"today": {"pressure": 2793., "oil": 100., "water": 500.},
            "baseline": {"MPB-28": "12B"}, "plan": {"choices": {"MPB-28": "SI"}, "pressure": 2802.},
            "singles": [{"well": "MPB-28", "from": "12B", "to": "SI", "pressure_after": 2802.}]}

    def option(ws, label, pressure):
        if label == "SI":
            return 0., 0.
        return (100., 500.) if pressure == 2793. else None

    monkeypatch.setattr(cfp_moves, "build_response_surfaces", surfaces)
    monkeypatch.setattr(cfp_moves, "anchor", lambda *args, **kw: object())
    monkeypatch.setattr(cfp_moves, "moves_summary", summary)
    monkeypatch.setattr(cfp_moves, "option_at", option)
    req = schemas.OptimizeRunRequest(kind="cfp", cfp_pads=["B"], p0_psi=2793., cfp_pad_pf_psi={"B": 2675.})
    result = runs._run_cfp_job({}, req)
    assert 2793. in captured["p_grid"] and max(captured["p_grid"]) == 2880.
    assert captured["p0"] == 2793. and captured["measured_pad_pf"] == {"B": 2675.}
    assert result["anchor_basis"] == "manual_reference"
    assert result["summary"]["singles"][0]["own_water_delta"] == -500.


def test_pad_run_lifecycle_and_hydration(client, monkeypatch):
    captured: dict = {}

    def fake_run(configs, plant, n_pumps, nozzles, throats, method, marginal_wc, **kw):
        captured["wells"] = [c.well_name for c in configs]
        captured["pads"] = [c.pad for c in configs]
        captured["ken"] = {c.well_name: c.ken_well for c in configs}
        captured["method"] = method
        captured["n_steps"] = kw.get("n_steps")
        captured["water_price"] = kw.get("water_price")
        captured["marginal_wc"] = marginal_wc
        # A skipped plant-curve point has no header; progress must still work.
        kw["progress"](1, 3, None, 0., 0.)
        kw["progress"](2, 3, 2450., 9000., 750.)
        return [_FakeResult(c.well_name) for c in configs], object(), {
            "header_psi": 2450.0,
            "total_pf_bpd": 9000.0,
            "total_oil_bopd": 750.0,
            "converged": True,
            "marginal_wc_used": 0.94,
            "marginal_wc_source": "auto",
            "parsimony_swaps": [],
        }

    import woffl.gui.pad_optimize as pad_optimize

    monkeypatch.setattr(pad_optimize, "run_optimization", fake_run)

    r = client.post(
        "/api/optimize/run",
        json={
            "kind": "pad",
            "pad": "M",
            "offline": ["MPM-02"],
            "future": [{"name": "MPM-99", "match": "MPB-28"}],
            "method": "mckp",
            "n_steps": 3,
        },
    )
    assert r.status_code == 200
    body = _wait_done(client, r.json()["job_id"])
    assert body["status"] == "done"

    # Hydration: offline well excluded; future well cloned from its donor
    # (carries the donor's calibration) but runs under its own name on M.
    assert captured["wells"] == ["MPM-01", "MPM-99"]
    assert captured["pads"] == ["M", "M"]
    assert captured["ken"]["MPM-99"] is None  # donor MPB-28 has kth, not ken
    assert captured["method"] == "mckp"
    assert captured["n_steps"] == 3
    # no price given -> auto (None) and the legacy gate untouched
    assert captured["water_price"] is None and captured["marginal_wc"] is None

    result = body["result"]
    assert result["pad"] == "M"
    assert {row["well"] for row in result["rows"]} == {"MPM-01", "MPM-99"}
    assert result["rows"][0]["pump"] == "12B"
    assert result["meta"]["header_psi"] == 2450.0
    assert any("MPM-99" in n for n in result["notes"])  # future-well provenance note


def test_choke_strategy_routes_to_the_choke_engine(client, monkeypatch):
    """strategy="choke" must run run_choke_optimization (never the JPCO
    engine), pass the reduced pump count through, and return a `plan`
    payload (not `rows`) with fit provenance merged onto every well."""
    import woffl.gui.pad_optimize as pad_optimize

    captured: dict = {}

    def fake_choke(
        configs, plant, n_pumps, current, test_rates, *, n_levels, progress=None, evidence=None
    ):
        captured["n_pumps"] = n_pumps
        captured["n_levels"] = n_levels
        captured["wells"] = sorted(c.well_name for c in configs)
        captured["evidence"] = evidence
        rows = [
            {
                "well": c.well_name,
                "pump": "12B",
                "basis": "model",
                "action": "full",
                "delivered_psi": 3000.0,
                "choke_dp_psi": 0.0,
                "pf": 1000.0,
                "oil": 100.0,
                "d_oil_vs_full": 0.0,
                "d_pf_vs_full": 0.0,
                "test_oil": None,
                "test_pf": None,
                "projected_oil": None,
                "next_trim_bopd_per_bpd": None,
            }
            for c in configs
        ]
        return rows, {"mode": "choke", "header_psi": 3000.0, "n_pumps": n_pumps}

    def boom(*a, **k):
        raise AssertionError("JPCO engine must not run for strategy=choke")

    # keep the suction-evidence pull off the network - {} means "no wells
    # with usable data", which the wiring passes through as evidence=None
    monkeypatch.setattr(runs.evidence_svc, "pad_evidence", lambda names, res_pres=None: {})
    monkeypatch.setattr(pad_optimize, "run_choke_optimization", fake_choke)
    monkeypatch.setattr(pad_optimize, "run_optimization", boom)

    r = client.post(
        "/api/optimize/run",
        json={"kind": "pad", "pad": "M", "strategy": "choke", "n_pumps": 2},
    )
    assert r.status_code == 200
    body = _wait_done(client, r.json()["job_id"])
    assert body["status"] == "done"
    result = body["result"]
    assert "plan" in result and "rows" not in result
    assert captured["n_pumps"] == 2
    assert captured["n_levels"] == 10  # the choke default when n_steps is unset
    assert captured["wells"] == ["MPM-01", "MPM-02"]
    assert captured["evidence"] is None  # empty evidence collapses to None
    assert result["meta"]["mode"] == "choke"
    # fit provenance rides on every plan row, like pad rows
    assert all(
        {"ipr_source", "ipr_r2", "has_friction"} <= set(row) for row in result["plan"]
    )

def test_build_configs_plumbs_event_calibration_fields(monkeypatch):
    """The saved event-cal fit reaches the engine: mach_crit /
    nozzle_area_factor seeds land on mach_crit_well / fnz_well, and the
    context's current-pump seed (JP history) lands on installed_nozzle /
    installed_throat so fnz wear stays with the installed pump."""
    universe = {"wells": [{"name": "MPM-01", "pad": "M"}], "source": "databricks"}
    seeds = {
        "pres": 1700.0, "qwf": 900.0, "pwf": 600.0, "form_wc": 0.7,
        "ken": 0.05, "mach_crit": 1.6, "nozzle_area_factor": 1.12,
        "nozzle_no": "12", "area_ratio": "B",
    }
    monkeypatch.setattr(wells_svc, "list_wells", lambda: universe)
    monkeypatch.setattr(
        wells_svc, "well_context", lambda well, months, cap: {"seeds": dict(seeds)}
    )
    (cfg,) = runs._build_configs(["M"], set(), [], [])
    assert cfg.mach_crit_well == 1.6
    assert cfg.fnz_well == 1.12
    assert cfg.installed_nozzle == "12"
    assert cfg.installed_throat == "B"


def test_build_configs_event_cal_fields_fail_soft_none(monkeypatch):
    """No saved event-cal fit and no current pump -> all four fields None
    (byte-identical legacy behavior downstream)."""
    universe = {"wells": [{"name": "MPM-01", "pad": "M"}], "source": "databricks"}
    seeds = {"pres": 1700.0, "qwf": 900.0, "pwf": 600.0, "form_wc": 0.7}
    monkeypatch.setattr(wells_svc, "list_wells", lambda: universe)
    monkeypatch.setattr(
        wells_svc, "well_context", lambda well, months, cap: {"seeds": dict(seeds)}
    )
    (cfg,) = runs._build_configs(["M"], set(), [], [])
    assert cfg.mach_crit_well is None
    assert cfg.fnz_well is None
    assert cfg.installed_nozzle is None
    assert cfg.installed_throat is None


def test_choke_run_survives_evidence_fetch_failure(client, monkeypatch):
    """A dead warehouse during the evidence pull must degrade to the
    uncorrected model-only run (fail-soft note), never fail the job."""
    import woffl.gui.pad_optimize as pad_optimize

    captured: dict = {}

    def fake_choke(
        configs, plant, n_pumps, current, test_rates, *, n_levels, progress=None, evidence=None
    ):
        captured["evidence"] = evidence
        rows = [
            {
                "well": c.well_name,
                "pump": "12B",
                "basis": "model",
                "action": "full",
                "delivered_psi": 3000.0,
                "choke_dp_psi": 0.0,
                "pf": 1000.0,
                "oil": 100.0,
            }
            for c in configs
        ]
        return rows, {"mode": "choke", "header_psi": 3000.0, "n_pumps": n_pumps}

    def dead_warehouse(names, res_pres=None):
        raise RuntimeError("warehouse unreachable")

    monkeypatch.setattr(pad_optimize, "run_choke_optimization", fake_choke)
    monkeypatch.setattr(runs.evidence_svc, "pad_evidence", dead_warehouse)

    r = client.post(
        "/api/optimize/run",
        json={"kind": "pad", "pad": "M", "strategy": "choke"},
    )
    assert r.status_code == 200
    body = _wait_done(client, r.json()["job_id"])
    assert body["status"] == "done"
    result = body["result"]
    assert "plan" in result and len(result["plan"]) == 2
    assert captured["evidence"] is None  # the engine ran model-only
    assert any(
        "suction evidence unavailable" in n and "model-only run" in n
        for n in result["notes"]
    )



def test_run_failure_surfaces_as_error(client, monkeypatch):
    import woffl.gui.pad_optimize as pad_optimize

    def boom(*a, **k):
        raise RuntimeError("infeasible sweep")

    monkeypatch.setattr(pad_optimize, "run_optimization", boom)
    r = client.post("/api/optimize/run", json={"kind": "pad", "pad": "M"})
    body = _wait_done(client, r.json()["job_id"])
    assert body["status"] == "error"
    assert "infeasible sweep" in body["error"]


def test_plain_flattens_engine_payloads():
    import numpy as np
    import pandas as pd

    out = runs._plain(
        {
            "f": np.float64(1.5),
            "nan": float("nan"),
            "ts": pd.Timestamp("2026-08-06"),
            "df": pd.DataFrame({"Well": ["A"], "Status": ["ok"]}),
            "nested": [(np.int32(2), {"x": np.bool_(True)})],
        }
    )
    assert out == {
        "f": 1.5,
        "nan": None,
        "ts": "2026-08-06 00:00:00",
        "df": [{"Well": "A", "Status": "ok"}],
        "nested": [[2, {"x": True}]],
    }


def test_future_donor_seeding_failure_is_noted(client, monkeypatch):
    def flaky_context(well, months, cap):
        if well == "MPB-28":
            raise RuntimeError("databricks blip")
        return {"seeds": dict(_SEEDS[well])}

    monkeypatch.setattr(wells_svc, "well_context", flaky_context)

    import woffl.gui.pad_optimize as pad_optimize

    monkeypatch.setattr(
        pad_optimize,
        "run_optimization",
        lambda configs, *a, **k: ([_FakeResult(c.well_name) for c in configs], object(), {}),
    )
    r = client.post(
        "/api/optimize/run",
        json={"kind": "pad", "pad": "M", "future": [{"name": "MPM-99", "match": "MPB-28"}]},
    )
    body = _wait_done(client, r.json()["job_id"])
    assert body["status"] == "done"
    assert {row["well"] for row in body["result"]["rows"]} == {"MPM-01", "MPM-02", "MPM-99"}
    missing = next(row for row in body["result"]["rows"] if row["well"] == "MPM-99")
    assert missing["outcome"] == "missing_inputs" and missing["oil"] is None
    assert body["result"]["coverage"]["complete"] is False
    assert any("MPM-99" in n and "skipped" in n for n in body["result"]["notes"])


def test_failed_model_is_not_economic_shut_in_and_leaves_load_unaccounted(client, monkeypatch):
    """Identical missing selections have different outcomes when physics failed."""
    import woffl.gui.pad_optimize as pad_optimize
    monkeypatch.setattr(pad_optimize, "run_optimization", lambda *a, **kw: ([], object(), {
        "feasible": True, "header_psi": 2600.0,
        "reconciliation": [
            {"Well": "MPM-01", "Status": "failed simulation", "Configs OK": 0, "Detail": "No lift solution."},
            {"Well": "MPM-02", "Status": "simulated", "Configs OK": 3, "Detail": ""},
        ],
    }))
    body = runs._run_pad_job({}, schemas.OptimizeRunRequest(kind="pad", pad="M"))
    rows = {r["well"]: r for r in body["rows"]}
    assert rows["MPM-01"]["outcome"] == "failed_model"
    assert rows["MPM-02"]["outcome"] == "economic_shut_in"
    assert rows["MPM-01"]["oil"] is None
    assert body["coverage"]["unaccounted_wells"] == ["MPM-01"]
    assert body["coverage"]["accounted_online"] == 1
    assert body["meta"]["feasible"] is None
    assert body["meta"]["modeled_subset_feasible"] is True
    assert body["meta"]["recommendation_status"] == "incomplete_exploratory"
    assert body["meta"]["modeled_hardware_gain_bopd"] is None


def test_hydration_failure_stays_visible_in_expected_pad_coverage(client, monkeypatch):
    import woffl.gui.pad_optimize as pad_optimize
    def context(well, *a):
        if well == "MPM-02":
            raise RuntimeError("source unavailable")
        return {"seeds": dict(_SEEDS[well])}
    monkeypatch.setattr(wells_svc, "well_context", context)
    monkeypatch.setattr(pad_optimize, "run_optimization", lambda configs, *a, **kw: (
        [_FakeResult(c.well_name) for c in configs], object(), {"feasible": True}))
    body = runs._run_pad_job({}, schemas.OptimizeRunRequest(kind="pad", pad="M"))
    assert body["n_wells"] == 1
    assert body["coverage"]["expected_online"] == 2
    assert body["coverage"]["accounted_online"] == 1
    assert len(body["rows"]) == 2
    missing = next(r for r in body["rows"] if r["well"] == "MPM-02")
    assert missing["outcome"] == "missing_inputs"
    assert missing["pf"] is None and missing["oil"] is None
    assert body["meta"]["feasible"] is None


@pytest.mark.parametrize("feasible", [True, False])
def test_hardware_counterfactual_keeps_measured_bias_out_of_gain(client, monkeypatch, feasible):
    import woffl.gui.pad_optimize as pad_optimize
    monkeypatch.setattr(runs, "_current_and_tests", lambda names: (
        {w: ("12", "B") for w in names}, {w: (9999.0, None) for w in names}))
    monkeypatch.setattr(runs, "_modeled_current", lambda configs, current, header, opt: {
        c.well_name: {"oil": 250., "pf": 3000., "form_water": 900., "ppf": header} for c in configs})
    monkeypatch.setattr(pad_optimize, "run_optimization", lambda configs, *a, **kw: (
        [_FakeResult(c.well_name) for c in configs], object(), {"header_psi": 2600., "feasible": feasible,
        "min_total_flow": 9000., "hydraulically_feasible": True}))
    body = runs._run_pad_job({}, schemas.OptimizeRunRequest(kind="pad", pad="M"))
    assert body["coverage"]["complete"] is True
    assert body["meta"]["current_model_oil_bopd"] == 500.
    assert body["meta"]["modeled_hardware_gain_bopd"] == (0. if feasible else None)
    assert all(r["modeled_hardware_gain"] == (0. if feasible else None) for r in body["rows"])
    assert body["meta"]["min_total_flow"] == 9000.
    if not feasible:
        assert body["meta"]["recommendation_status"] == "conditional_operating_limits"
        assert any("gains withheld" in note for note in body["notes"])
    assert all(r["test_oil"] == 9999. and r["test_pf"] is None for r in body["rows"])
    assert "same plan header" in body["meta"]["comparison_basis"]


def test_current_counterfactual_reuses_winner_and_preserves_inputs(monkeypatch):
    """No extra batch for cached installed candidates; controls are identical."""
    import woffl.assembly.network_optimizer as network
    cfg = runs._config_from_seeds("MPM-01", "M", {
        **_SEEDS["MPM-01"], "nozzle_no": "12", "area_ratio": "B", "ppf_surf": 2300., "form_gor": 777.})
    original = vars(cfg).copy()
    class Winner:
        def get_pump_performance(self, well, n, t, *, pump_state):
            assert (well, n, t, pump_state) == ("MPM-01", "12", "B", "installed")
            return {"oil_rate": 250., "lift_water": 3000., "formation_water": 900.}
    monkeypatch.setattr(network, "NetworkOptimizer", lambda *a, **kw: pytest.fail("winner already has this current candidate"))
    result = runs._modeled_current([cfg], {"MPM-01": ("12", "B")}, 2600., Winner())
    assert result["MPM-01"]["ppf"] == 2600.
    assert result["MPM-01"]["oil"] == 250.
    assert vars(cfg) == original


def test_current_counterfactual_missing_candidate_uses_same_header_and_oil_ipr(monkeypatch):
    import woffl.assembly.network_optimizer as network
    cfg = runs._config_from_seeds("MPM-01", "M", {
        **_SEEDS["MPM-01"], "nozzle_no": "12", "area_ratio": "B", "ppf_surf": 2300., "form_gor": 777.})
    class Batch:
        def __init__(self, configs, constraint, nozzles, throats, **kw):
            assert len(configs) == 1
            clone = configs[0]
            assert clone is not cfg and clone.ppf_surf_well == constraint.pressure == 2600.
            assert clone.qwf * (1 - clone.form_wc) == cfg.qwf * (1 - cfg.form_wc)
            assert clone.form_gor == 777. and clone.pwf == cfg.pwf and clone.res_pres == cfg.res_pres
        def run_all_batch_simulations(self, **kw):
            pass
        def get_pump_performance(self, *a, **kw):
            return {"oil_rate": 251., "lift_water": 3001., "formation_water": 901.}
    monkeypatch.setattr(network, "NetworkOptimizer", Batch)
    result = runs._modeled_current([cfg], {"MPM-01": ("12", "B")}, 2600.)
    assert result["MPM-01"]["oil"] == 251.
    assert cfg.ppf_surf_well == 2300.


@pytest.mark.parametrize("invalid", [{"form_wc": .99}, {"qwf": 0.}, {"pwf": 1700.}])
def test_invalid_oil_model_is_accounted_without_entering_pad_sweep(client, monkeypatch, invalid):
    import woffl.gui.pad_optimize as pad_optimize
    monkeypatch.setattr(wells_svc, "well_context", lambda well, *a: {
        "seeds": {**_SEEDS[well], **(invalid if well == "MPM-01" else {})}})
    def run(configs, *a, **kw):
        assert [c.well_name for c in configs] == ["MPM-02"]
        return [_FakeResult("MPM-02")], object(), {"feasible": True}
    monkeypatch.setattr(pad_optimize, "run_optimization", run)
    result = runs._run_pad_job({}, schemas.OptimizeRunRequest(kind="pad", pad="M"))
    row = next(r for r in result["rows"] if r["well"] == "MPM-01")
    assert row["outcome"] == "unsupported_model"
    assert row["oil"] is None and row["pf"] is None
    assert result["coverage"]["unaccounted_wells"] == ["MPM-01"]
    assert result["meta"]["feasible"] is None


def test_recent_test_context_retains_missing_pf_as_unknown(monkeypatch):
    import pandas as pd
    monkeypatch.setattr(runs.datasources, "jp_history_safe", lambda: (None, "fixture"))
    monkeypatch.setattr(runs.tests_svc, "tests_for_well", lambda *a: pd.DataFrame({
        "WtDate": [pd.Timestamp("2026-09-10")], "WtOilVol": [123.]}))
    pumps, rates = runs._current_and_tests(["MPM-01"])
    assert pumps == {} and rates == {"MPM-01": (123., None)}


def test_no_usable_wells_returns_missing_rows_without_running_optimizer(client, monkeypatch):
    import woffl.gui.pad_optimize as pad_optimize
    monkeypatch.setattr(wells_svc, "well_context", lambda *a: {"seeds": {"form_wc": .999}})
    monkeypatch.setattr(pad_optimize, "run_optimization", lambda *a, **kw: pytest.fail("no inputs to simulate"))
    result = runs._run_pad_job({}, schemas.OptimizeRunRequest(kind="pad", pad="M"))
    assert len(result["rows"]) == 2 and result["n_wells"] == 0
    assert all(r["outcome"] == "unsupported_model" and r["oil"] is None for r in result["rows"])
    assert result["coverage"]["expected_online"] == 2
    assert result["coverage"]["accounted_online"] == 0
    assert result["meta"]["feasible"] is None


def test_cfp_pads_filter_and_water_enrichment(client, monkeypatch):
    """cfp_pads scopes the run to the selected pads, and every single move
    is enriched with own_water_delta (the SI/BOL ladder's PW column)."""
    from types import SimpleNamespace

    universe = {
        "wells": [
            {"name": "MPB-28", "pad": "B"},
            {"name": "MPG-01", "pad": "G"},
            {"name": "MPC-01", "pad": "C"},
            {"name": "MPL-01", "pad": "L"},
        ],
        "source": "databricks",
    }
    seeds = {"pres": 1700.0, "qwf": 900.0, "pwf": 600.0, "form_wc": 0.7}
    monkeypatch.setattr(wells_svc, "list_wells", lambda: universe)
    monkeypatch.setattr(wells_svc, "well_context", lambda well, months, cap: {"seeds": dict(seeds)})
    monkeypatch.setattr(
        runs, "_current_and_tests", lambda names: ({n: "12B" for n in names}, {})
    )

    import pandas as pd

    import woffl.assembly.pf_pressure as pf_pressure
    import woffl.gui.cfp_moves as cfp_moves
    from server.services import datasources

    # No warehouse in unit tests: the fleet PF frame is empty and the cluster
    # resolver (now called with that FRAME, not a pad letter) finds nothing.
    monkeypatch.setattr(datasources, "pf_latest_safe", lambda: pd.DataFrame())
    monkeypatch.setattr(pf_pressure, "pad_pf_cluster", lambda df, **kw: {})

    captured: dict = {}

    def fake_surfaces(pad_configs, online, current, plant, **kw):
        captured["pads"] = sorted(pad_configs)
        wells = {
            c.well_name: SimpleNamespace(pad=c.pad, online=online[c.well_name])
            for ws in pad_configs.values()
            for c in ws
        }
        return SimpleNamespace(wells=wells)

    def fake_summary(surfaces, plant):
        return {
            "today": {"pressure": 2800.0, "oil": 100.0, "water": 1000.0, "n_online": 2, "n_bol_candidates": 0},
            "lambda_bopd_per_psi": 1.0,
            "singles": [
                {
                    "well": "MPB-28",
                    "pad": "B",
                    "type": "shut_in",
                    "from": "12B",
                    "to": None,
                    "fleet_oil_delta": -5.0,
                    "own_oil_delta": -10.0,
                    "pressure_delta": 2.0,
                    "pressure_after": 2802.0,
                    "at_trip": False,
                }
            ],
            "n_positive_singles": 0,
            "pairs": [],
            "frontier": [],
            "plan": None,
            "plan_gain": None,
            "baseline": {"MPB-28": "12B", "MPG-01": "12B"},
        }

    # option_at prices a label at a pressure: OFF (None/"SI") makes nothing.
    def fake_option_at(ws, label, pressure):
        return (0.0, 0.0) if label in (None, "SI", "OFF") else (10.0, 500.0)

    monkeypatch.setattr(cfp_moves, "build_response_surfaces", fake_surfaces)
    monkeypatch.setattr(cfp_moves, "anchor", lambda surfaces, psi_per_kbpd: object())
    monkeypatch.setattr(cfp_moves, "moves_summary", fake_summary)
    monkeypatch.setattr(cfp_moves, "option_at", fake_option_at)

    r = client.post("/api/optimize/run", json={"kind": "cfp", "cfp_pads": ["B", "G", "L"]})
    assert r.status_code == 200
    body = _wait_done(client, r.json()["job_id"])
    assert body["status"] == "done"

    # C-pad well excluded from hydration; result echoes the run's pads with
    # the canonical four first, then extras. The extra pad's PF assumption
    # is surfaced as a note.
    assert captured["pads"] == ["B", "G", "L"]
    result = body["result"]
    assert result["pads"] == ["B", "G", "L"]
    assert result["n_wells"] == 3
    assert any("L-Pad" in n and "boosted on-pad" in n for n in result["notes"])

    # Enrichment: shutting in from 12B frees the 500 BWPD that pump made.
    single = result["summary"]["singles"][0]
    assert single["own_water_delta"] == -500.0

    # POPs pads separate water on-pad - they never load the CFP machines,
    # so the schema rejects them with the reason.
    r2 = client.post("/api/optimize/run", json={"kind": "cfp", "cfp_pads": ["M"]})
    assert r2.status_code == 422
    assert "POPs" in r2.text

    # A non-POPs pad with no wells is not a schema error - it hydrates zero
    # wells and the job reports that honestly.
    r3 = client.post("/api/optimize/run", json={"kind": "cfp", "cfp_pads": ["Z"]})
    body3 = _wait_done(client, r3.json()["job_id"])
    assert body3["status"] == "error"
    assert "no active wells" in body3["error"]


@pytest.mark.parametrize("with_future", [False, True])
def test_cfp_offline_wells_are_bring_online_candidates(client, monkeypatch, with_future):
    """SRV-4 / OPT-A10 (review 2026-09-01): a board-offline well must reach
    the CFP engine with online=False - it IS the bring-online candidate.
    Until the fix it was dropped before hydration, so the SI/BOL ladder
    could never price bringing a shut-in well back on."""
    from types import SimpleNamespace

    import pandas as pd

    import woffl.assembly.pf_pressure as pf_pressure
    import woffl.gui.cfp_moves as cfp_moves
    from server.services import datasources

    universe = {
        "wells": [{"name": "MPB-28", "pad": "B"}, {"name": "MPG-01", "pad": "G"}],
        "source": "databricks",
    }
    seeds = {"pres": 1700.0, "qwf": 900.0, "pwf": 600.0, "form_wc": 0.7}
    monkeypatch.setattr(wells_svc, "list_wells", lambda: universe)
    monkeypatch.setattr(wells_svc, "well_context", lambda well, months, cap: {"seeds": dict(seeds)})
    monkeypatch.setattr(
        runs, "_current_and_tests",
        lambda names: ({n: ("12", "B") for n in names if n != "FUTURE-G"}, {}),
    )
    monkeypatch.setattr(datasources, "pf_latest_safe", lambda: pd.DataFrame())
    monkeypatch.setattr(pf_pressure, "pad_pf_cluster", lambda df, **kw: {})

    captured: dict = {}

    def fake_surfaces(pad_configs, online, current, plant, **kw):
        captured["online"] = dict(online)
        captured["current"] = dict(current)
        captured["pads"] = {c.well_name: c.pad for cs in pad_configs.values() for c in cs}
        wells = {
            c.well_name: SimpleNamespace(pad=c.pad, online=online[c.well_name])
            for ws in pad_configs.values()
            for c in ws
        }
        return SimpleNamespace(wells=wells)

    def fake_summary(surfaces, plant):
        return {
            "today": {"pressure": 2800.0, "oil": 100.0, "water": 1000.0, "n_online": 1, "n_bol_candidates": 1},
            "lambda_bopd_per_psi": 1.0,
            "singles": [],
            "n_positive_singles": 0,
            "pairs": [],
            "frontier": [],
            "plan": None,
            "plan_gain": None,
            "baseline": {"MPB-28": "12B", "MPG-01": "OFF"},
        }

    monkeypatch.setattr(cfp_moves, "build_response_surfaces", fake_surfaces)
    monkeypatch.setattr(cfp_moves, "anchor", lambda surfaces, psi_per_kbpd: object())
    monkeypatch.setattr(cfp_moves, "moves_summary", fake_summary)
    monkeypatch.setattr(cfp_moves, "option_at", lambda ws, label, pressure: (0.0, 0.0))

    request = {"kind": "cfp", "cfp_pads": ["B", "G"], "offline": ["MPG-01"]}
    if with_future:
        request["future"] = [{"name": "FUTURE-G", "match": "MPM-01", "pad": "G"}]
    r = client.post("/api/optimize/run", json=request)
    assert r.status_code == 200
    body = _wait_done(client, r.json()["job_id"])
    assert body["status"] == "done", body.get("error")
    expected = {"MPB-28": True, "MPG-01": False}
    if with_future:
        expected["FUTURE-G"] = False
        assert captured["pads"]["FUTURE-G"] == "G"
        assert captured["current"]["FUTURE-G"] == ("12", "B")
    assert captured["online"] == expected


def test_pad_run_forwards_the_header_setpoint(client, monkeypatch):
    """setpoint_psi pins a free-pressure pad's header to ONE trial. The
    engine decides what to do with it (fixed_curve S-Pad ignores it), so the
    server contract is only that the kwarg is forwarded verbatim and that
    the clamped value the engine reports rides back out in meta."""
    captured: dict = {}

    def fake_run(configs, plant, n_pumps, nozzles, throats, method, marginal_wc, **kw):
        captured["setpoint_psi"] = kw.get("setpoint_psi")
        captured["kw"] = set(kw)
        return [_FakeResult(c.well_name) for c in configs], object(), {
            "header_psi": 3200.0,
            "total_pf_bpd": 9000.0,
            "total_oil_bopd": 750.0,
            "converged": True,
            "setpoint_psi": 3200.0,
        }

    import woffl.gui.pad_optimize as pad_optimize

    monkeypatch.setattr(pad_optimize, "run_optimization", fake_run)
    r = client.post(
        "/api/optimize/run",
        json={"kind": "pad", "pad": "M", "setpoint_psi": 3200.0},
    )
    assert r.status_code == 200
    body = _wait_done(client, r.json()["job_id"])
    assert body["status"] == "done"
    assert captured["setpoint_psi"] == 3200.0
    # the pinned header is part of the meta contract the page reads
    assert body["result"]["meta"]["setpoint_psi"] == 3200.0


def test_pad_run_without_a_setpoint_sweeps(client, monkeypatch):
    """No setpoint given -> None forwarded (a swept run), and the S-Pad is
    handed the same None: the fixed-curve station has no header to pin."""
    captured: dict = {}

    def fake_run(configs, plant, n_pumps, nozzles, throats, method, marginal_wc, **kw):
        captured["setpoint_psi"] = kw.get("setpoint_psi")
        return [_FakeResult(c.well_name) for c in configs], object(), {
            "header_psi": 2450.0,
            "total_pf_bpd": 9000.0,
            "total_oil_bopd": 750.0,
            "converged": True,
            "setpoint_psi": None,
        }

    import woffl.gui.pad_optimize as pad_optimize

    monkeypatch.setattr(pad_optimize, "run_optimization", fake_run)
    r = client.post("/api/optimize/run", json={"kind": "pad", "pad": "M"})
    assert r.status_code == 200
    body = _wait_done(client, r.json()["job_id"])
    assert body["status"] == "done"
    assert captured["setpoint_psi"] is None
    assert body["result"]["meta"]["setpoint_psi"] is None


def test_setpoint_outside_the_band_is_rejected(client):
    r = client.post("/api/optimize/run", json={"kind": "pad", "pad": "M", "setpoint_psi": 900.0})
    assert r.status_code == 422


def test_pad_run_forwards_the_water_price(client, monkeypatch):
    captured: dict = {}

    def fake_run(configs, plant, n_pumps, nozzles, throats, method, marginal_wc, **kw):
        captured["water_price"] = kw.get("water_price")
        captured["kw"] = set(kw)
        return [_FakeResult(c.well_name) for c in configs], object(), {
            "header_psi": 2450.0,
            "total_pf_bpd": 9000.0,
            "total_oil_bopd": 750.0,
            "converged": True,
            "lambda_used": 0.02,
            "lambda_source": "manual",
            "objective_bopd_equiv": 570.0,
            "solver_agreement": {"agree": True, "mckp_objective": 570.0, "milp_objective": 570.0},
        }

    import woffl.gui.pad_optimize as pad_optimize

    monkeypatch.setattr(pad_optimize, "run_optimization", fake_run)
    r = client.post(
        "/api/optimize/run",
        json={"kind": "pad", "pad": "M", "method": "mckp", "lambda_bopd_per_bpd": 0.02, "parsimony_bopd": 50},
    )
    assert r.status_code == 200
    body = _wait_done(client, r.json()["job_id"])
    assert body["status"] == "done"
    assert captured["water_price"] == 0.02
    assert "parsimony_bopd" not in captured["kw"]  # retired: never forwarded
    meta = body["result"]["meta"]
    assert meta["lambda_used"] == 0.02 and meta["lambda_source"] == "manual"
    assert meta["objective_bopd_equiv"] == 570.0
    assert meta["solver_agreement"]["agree"] is True
