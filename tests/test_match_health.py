"""Match-health scorecard: row assembly, verdict precedence, friction-rail
detection tolerances, and the fail-soft path when the evidence pull raises.

All synthetic - hydration, match_check and evidence are monkeypatched; no
Databricks, no batch simulation.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import server.services.match_health as mh
import server.services.optimizer_runs as runs
import server.services.wells as wells_svc
from server.main import app

# ---------------------------------------------------------------------------
# Synthetic building blocks
# ---------------------------------------------------------------------------


def _cfg(well: str, ken=0.05, kth=0.30, kdi=0.30, res_pres=1700.0):
    return SimpleNamespace(
        well_name=well, ken_well=ken, kth_well=kth, kdi_well=kdi, res_pres=res_pres
    )


def _check_row(well: str, **overrides):
    row = {
        "well": well,
        "pump": "12B",
        "test_oil": 500.0,
        "model_oil": 520.0,
        "oil_ratio": 1.04,
        "oil_flag": "match",
        "test_pf": 3000.0,
        "model_pf": 2900.0,
        "pf_ratio": 0.97,
        "pf_flag": "match",
        "model_psu": 400.0,
        "sonic": False,
    }
    row.update(overrides)
    return row


def _ev(**overrides):
    ev = {
        "floor": 380.0,
        "psu_ref": 450.0,
        "beta": 0.01,
        "beta_source": "well",
        "n_days": 200,
        "n_pairs": 1500,
        "window": ("2025-08-10", "2026-08-10"),
    }
    ev.update(overrides)
    return ev


def _assemble_one(check_row, cfg, ev, prov=None, last_test=None):
    evidence = {check_row["well"]: ev} if ev is not None else None
    rows = mh.assemble_rows(
        [check_row],
        prov or {},
        evidence,
        [cfg],
        last_test or {},
    )
    assert len(rows) == 1
    return rows[0]


# ---------------------------------------------------------------------------
# Row assembly
# ---------------------------------------------------------------------------


def test_row_carries_every_column():
    prov = {"W1": {"ipr_source": "saved", "ipr_r2": 0.93, "has_friction": True}}
    row = _assemble_one(
        _check_row("W1"),
        _cfg("W1", ken=0.05, kth=0.30, kdi=0.30),
        _ev(),
        prov=prov,
        last_test={"W1": "2026-08-01"},
    )
    assert row["well"] == "W1"
    assert row["pump"] == "12B"
    assert row["ipr_source"] == "saved"
    assert row["ipr_r2"] == 0.93
    assert row["model_test_oil_ratio"] == 1.04
    assert row["model_test_pf_ratio"] == 0.97
    assert row["pf_flag"] == "match"
    assert row["model_psu"] == 400.0
    assert row["sonic"] is False
    assert row["evidence_floor"] == 380.0
    assert row["floor_violation"] == pytest.approx(20.0)  # 400 - 380
    assert row["beta"] == 0.01
    assert row["beta_source"] == "well"
    assert row["n_pairs"] == 1500
    assert row["ken"] == 0.05 and row["kth"] == 0.30 and row["kdi"] == 0.30
    assert not (row["ken_railed"] or row["kth_railed"] or row["kdi_railed"])
    assert row["last_test_date"] == "2026-08-01"
    assert row["verdict"] == "ok"  # violation 20 <= 25 confirms the model


def test_floor_violation_needs_both_sides():
    # No model psu -> no violation, even with a measured floor.
    row = _assemble_one(_check_row("W1", model_psu=None), _cfg("W1"), _ev())
    assert row["floor_violation"] is None
    # No measured floor -> no violation, even with a model psu.
    row = _assemble_one(_check_row("W1"), _cfg("W1"), _ev(floor=None))
    assert row["floor_violation"] is None


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


def test_contradicted_on_floor_violation():
    # Model floor 145 psi above the measured floor - the MPM-64 signature.
    # The model must CLAIM the floor (sonic) for the measured floor to
    # falsify it.
    row = _assemble_one(
        _check_row("W1", model_psu=525.0, sonic=True), _cfg("W1"), _ev(floor=380.0)
    )
    assert row["floor_violation"] == pytest.approx(145.0)
    assert row["verdict"] == "contradicted"


def test_subsonic_floor_violation_is_not_contradicted():
    """EVID-F3 (review 2026-09-01): a subsonic well whose modeled psu today
    sits above the lowest BHP it reached in the last year is not
    contradicted - it is simply not at its floor today. Only a SONIC claim
    (zero suction response) is falsified by the measured floor."""
    row = _assemble_one(
        _check_row("W1", model_psu=525.0, sonic=False), _cfg("W1"), _ev(floor=380.0)
    )
    assert row["floor_violation"] == pytest.approx(145.0)
    assert row["verdict"] == "ok"


def test_floor_violation_at_threshold_is_not_contradicted():
    row = _assemble_one(
        _check_row("W1", model_psu=405.0), _cfg("W1"), _ev(floor=380.0)
    )
    assert row["floor_violation"] == pytest.approx(25.0)
    assert row["verdict"] == "ok"


def test_contradicted_on_responsive_beta_vs_sonic_claim():
    # The MPM-28 case: floor CONFIRMED (317 measured vs 291 model) but the
    # well-earned beta 0.077 falsifies the model's sonic-pinned claim.
    row = _assemble_one(
        _check_row("W1", model_psu=291.0, sonic=True),
        _cfg("W1"),
        _ev(floor=317.0, beta=0.077, beta_source="well"),
    )
    assert row["floor_violation"] < 0  # no floor violation
    assert row["verdict"] == "contradicted"


def test_responsive_beta_needs_well_source_and_sonic_model():
    # Pad-inherited beta is not the well's own testimony.
    row = _assemble_one(
        _check_row("W1", sonic=True),
        _cfg("W1"),
        _ev(floor=380.0, beta=0.077, beta_source="pad"),
    )
    assert row["verdict"] == "ok"
    # Model not sonic-pinned -> nothing to falsify.
    row = _assemble_one(
        _check_row("W1", sonic=False),
        _cfg("W1"),
        _ev(floor=380.0, beta=0.077, beta_source="well"),
    )
    assert row["verdict"] == "ok"
    # Insensitive well (beta below 0.03) agrees with a sonic model.
    row = _assemble_one(
        _check_row("W1", sonic=True),
        _cfg("W1"),
        _ev(floor=380.0, beta=0.01, beta_source="well"),
    )
    assert row["verdict"] == "ok"


def test_railed_cal_verdict():
    row = _assemble_one(
        _check_row("W1"), _cfg("W1", ken=0.40, kth=0.05, kdi=0.05), _ev()
    )
    assert row["ken_railed"] and row["kth_railed"] and row["kdi_railed"]
    assert row["verdict"] == "railed-cal"


def test_weak_fit_verdict():
    prov = {"W1": {"ipr_source": "auto", "ipr_r2": 0.41, "has_friction": False}}
    row = _assemble_one(_check_row("W1"), _cfg("W1"), _ev(), prov=prov)
    assert row["verdict"] == "weak-fit"


def test_missing_r2_is_not_weak():
    prov = {"W1": {"ipr_source": "defaults", "ipr_r2": None, "has_friction": False}}
    row = _assemble_one(_check_row("W1"), _cfg("W1"), _ev(), prov=prov)
    assert row["verdict"] == "ok"


def test_verdict_precedence_contradicted_beats_railed_beats_weak():
    prov = {"W1": {"ipr_source": "auto", "ipr_r2": 0.2, "has_friction": True}}
    # All three fire -> contradicted wins.
    row = _assemble_one(
        _check_row("W1", model_psu=600.0, sonic=True),
        _cfg("W1", ken=0.40, kth=0.05, kdi=0.05),
        _ev(floor=380.0),
        prov=prov,
    )
    assert row["verdict"] == "contradicted"
    # Railed + weak -> railed wins.
    row = _assemble_one(
        _check_row("W1"),
        _cfg("W1", ken=0.40, kth=0.05, kdi=0.05),
        _ev(),
        prov=prov,
    )
    assert row["verdict"] == "railed-cal"
    # Weak only.
    row = _assemble_one(_check_row("W1"), _cfg("W1"), _ev(), prov=prov)
    assert row["verdict"] == "weak-fit"


# ---------------------------------------------------------------------------
# Rail detection tolerances
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ken, railed",
    [
        (0.40, True),
        (0.395, True),   # |0.395 - 0.40| = 0.005 < 0.01
        (0.391, True),
        (0.385, False),  # 0.015 away - a legitimate high-ken fit
        (0.30, False),
        (0.005, False),  # the LOWER ken bound is not the degenerate corner
        (None, False),
    ],
)
def test_ken_rail_tolerance(ken, railed):
    got, _, _ = mh.friction_rails(ken, 0.30, 0.30)
    assert got is railed


@pytest.mark.parametrize(
    "k, railed",
    [
        (0.05, True),
        (0.0505, True),   # within 0.001 of the 0.05 floor
        (0.051, True),
        (0.052, False),
        (0.30, False),
        (None, False),
    ],
)
def test_kth_kdi_rail_tolerance(k, railed):
    _, kth_railed, kdi_railed = mh.friction_rails(0.05, k, k)
    assert kth_railed is railed
    assert kdi_railed is railed


# ---------------------------------------------------------------------------
# Fail-soft: evidence raise leaves evidence columns None, rows still built
# ---------------------------------------------------------------------------


def test_rows_survive_missing_evidence():
    row = _assemble_one(_check_row("W1"), _cfg("W1"), None)
    assert row["evidence_floor"] is None
    assert row["floor_violation"] is None
    assert row["beta"] is None
    assert row["beta_source"] is None
    assert row["n_pairs"] is None
    assert row["verdict"] == "ok"


# ---------------------------------------------------------------------------
# Job lifecycle through the router (POST /optimize/match-health + polling)
# ---------------------------------------------------------------------------

_UNIVERSE = {
    "wells": [{"name": "MPM-01", "pad": "M"}, {"name": "MPM-02", "pad": "M"}],
    "source": "databricks",
}


@pytest.fixture()
def client(monkeypatch) -> TestClient:
    def fake_build_configs(pads, offline, future, note, prov=None):
        if prov is not None:
            prov["MPM-01"] = {"ipr_source": "saved", "ipr_r2": 0.9, "has_friction": True}
            prov["MPM-02"] = {"ipr_source": "auto", "ipr_r2": 0.3, "has_friction": True}
        return [
            _cfg("MPM-01", ken=0.05, kth=0.30, kdi=0.30),
            _cfg("MPM-02", ken=0.40, kth=0.05, kdi=0.05),
        ]

    def fake_match_check(configs, plant, n_pumps, current, test_rates):
        rows = [
            _check_row("MPM-01", model_psu=600.0, sonic=True),
            _check_row("MPM-02", model_psu=390.0, sonic=False),
        ]
        return rows, 2450.0

    import woffl.gui.pad_optimize as pad_optimize

    monkeypatch.setattr(wells_svc, "list_wells", lambda: _UNIVERSE)
    monkeypatch.setattr(runs, "_build_configs", fake_build_configs)
    monkeypatch.setattr(runs, "_current_and_tests", lambda wells: ({}, {}))
    monkeypatch.setattr(runs, "_pad_plant", lambda pad: object())
    monkeypatch.setattr(pad_optimize, "match_check", fake_match_check)
    monkeypatch.setattr(mh, "_last_test_dates", lambda wells: {"MPM-01": "2026-08-01"})
    return TestClient(app)


def _wait_done(client: TestClient, job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/api/optimize/run/{job_id}").json()
        if body["status"] != "running":
            return body
        time.sleep(0.05)
    raise AssertionError("job did not settle in time")


def test_job_lifecycle_and_assembly(client, monkeypatch):
    monkeypatch.setattr(
        mh.evidence_svc,
        "pad_evidence",
        lambda names, res_pres=None: {
            "MPM-01": _ev(floor=380.0),
            "MPM-02": _ev(floor=380.0, beta=0.079, beta_source="well"),
        },
    )
    r = client.post("/api/optimize/match-health", json={"pad": "M"})
    assert r.status_code == 200
    body = _wait_done(client, r.json()["job_id"])
    assert body["status"] == "done"
    assert body["kind"] == "match_health"

    result = body["result"]
    assert result["pad"] == "M"
    assert result["header_psi"] == 2450.0
    assert result["n_wells"] == 2
    by_well = {row["well"]: row for row in result["rows"]}
    # MPM-01: floor violation 220 psi -> contradicted despite the good fit.
    assert by_well["MPM-01"]["floor_violation"] == pytest.approx(220.0)
    assert by_well["MPM-01"]["verdict"] == "contradicted"
    assert by_well["MPM-01"]["last_test_date"] == "2026-08-01"
    # MPM-02: no violation, responsive beta but model NOT sonic -> the
    # railed calibration is the loudest remaining problem.
    assert by_well["MPM-02"]["ken_railed"] is True
    assert by_well["MPM-02"]["verdict"] == "railed-cal"
    assert by_well["MPM-02"]["last_test_date"] is None


def test_job_survives_evidence_fetch_failure(client, monkeypatch):
    def dead_warehouse(names, res_pres=None):
        raise RuntimeError("warehouse unreachable")

    monkeypatch.setattr(mh.evidence_svc, "pad_evidence", dead_warehouse)
    r = client.post("/api/optimize/match-health", json={"pad": "M"})
    assert r.status_code == 200
    body = _wait_done(client, r.json()["job_id"])
    assert body["status"] == "done"
    result = body["result"]
    assert len(result["rows"]) == 2
    assert all(row["evidence_floor"] is None for row in result["rows"])
    assert all(row["beta"] is None for row in result["rows"])
    assert any("suction evidence unavailable" in n for n in result["notes"])


def test_bad_pad_is_422(client):
    assert client.post("/api/optimize/match-health", json={"pad": "Z"}).status_code == 422
