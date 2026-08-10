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


def test_pad_run_lifecycle_and_hydration(client, monkeypatch):
    captured: dict = {}

    def fake_run(configs, plant, n_pumps, nozzles, throats, method, marginal_wc, **kw):
        captured["wells"] = [c.well_name for c in configs]
        captured["pads"] = [c.pad for c in configs]
        captured["ken"] = {c.well_name: c.ken_well for c in configs}
        captured["method"] = method
        captured["n_steps"] = kw.get("n_steps")
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

    def fake_choke(configs, plant, n_pumps, current, test_rates, *, n_levels, progress=None):
        captured["n_pumps"] = n_pumps
        captured["n_levels"] = n_levels
        captured["wells"] = sorted(c.well_name for c in configs)
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
    assert result["meta"]["mode"] == "choke"
    # fit provenance rides on every plan row, like pad rows
    assert all(
        {"ipr_source", "ipr_r2", "has_friction"} <= set(row) for row in result["plan"]
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
    assert {row["well"] for row in body["result"]["rows"]} == {"MPM-01", "MPM-02"}
    assert any("MPM-99" in n and "skipped" in n for n in body["result"]["notes"])


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

    import woffl.assembly.pf_pressure as pf_pressure
    import woffl.gui.cfp_moves as cfp_moves

    monkeypatch.setattr(pf_pressure, "pad_pf_cluster", lambda pad: None)

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
