"""POST /calibrate contract: argument mapping and result serialization.

The optimizer itself (Nelder-Mead multi-start, bounds, match grading) lives
in woffl.gui.fric_calibration and is exercised by its own tests + the E2E
run; these tests pin the SERVER contract: the request params feed the same
sim-object factories as a solve, test-day WHP wins over the sidebar surface
pressure (build_calibration_inputs' rule), knz stays at the Streamlit
constant 0.01, only ken/kth/kdi ever reach the optimizer as free variables,
and NaN diagnostics serialize as JSON null.
"""

from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

import woffl.gui.fric_calibration as fc
from woffl.gui.fric_calibration import FricCalibrationResult

from server.main import app

WELL = "MPB-28"


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture()
def captured(monkeypatch):
    calls: list[dict] = []

    def fake_calibrate(**kwargs):
        calls.append(kwargs)
        return FricCalibrationResult(
            well_name=kwargs["well_name"],
            target_bhp=kwargs["target_bhp"],
            knz=kwargs["knz"],
            seed_ken=kwargs["ken"],
            best_ken=0.05,
            best_kth=0.45,
            best_kdi=0.62,
            best_modeled_bhp=1180.0,
            best_oil=300.0,
            best_pf_rate=3000.0,
            bhp_error=5.0,
            converged=True,
            iterations=42,
            match_quality="good",
            bounded=False,
            sonic=False,
            starts_tried=1,
        )

    monkeypatch.setattr(fc, "calibrate_friction_coefs", fake_calibrate)
    return calls


def _params(**over) -> dict:
    """A minimal valid SimParams payload (schema defaults fill the rest)."""
    base: dict = {"ken": 0.08, "surf_pres": 210.0, "ppf_surf": 3168.0, "form_temp": 120.0}
    base.update(over)
    return base


def test_calibrate_maps_args(client, captured):
    r = client.post(
        "/api/calibrate",
        json={"well": WELL, "params": _params(), "target_bhp": 1175.0, "test_whp": 464.0},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["converged"] is True
    assert body["match_quality"] == "good"
    assert (body["ken"], body["kth"], body["kdi"]) == (0.05, 0.45, 0.62)

    call = captured[0]
    assert call["well_name"] == WELL
    assert call["target_bhp"] == 1175.0
    assert call["pwh"] == 464.0  # test-day WHP wins
    assert call["knz"] == 0.01  # Streamlit constant, held fixed
    assert call["ken"] == 0.08  # sidebar seed
    # The free variables are ONLY the friction coefs: geometry objects are
    # prebuilt from params (as-built) and passed opaque - no depth/casing
    # dimension appears among the optimizer's arguments.
    assert {"jpump_md", "casing_out_dia", "tubing_od"}.isdisjoint(call.keys())


def test_calibrate_whp_falls_back_to_sidebar(client, captured):
    r = client.post(
        "/api/calibrate",
        json={"well": WELL, "params": _params(surf_pres=222.0), "target_bhp": 1175.0, "test_whp": None},
    )
    assert r.status_code == 200
    assert captured[0]["pwh"] == 222.0


def test_calibrate_nan_serializes_null(client, monkeypatch):
    def failed(**kwargs):
        return FricCalibrationResult(
            well_name=kwargs["well_name"],
            target_bhp=kwargs["target_bhp"],
            knz=kwargs["knz"],
            seed_ken=kwargs["ken"],
            best_ken=0.03,
            best_kth=0.3,
            best_kdi=0.3,
            best_modeled_bhp=math.nan,
            best_oil=math.nan,
            best_pf_rate=math.nan,
            bhp_error=math.nan,
            converged=False,
            iterations=7,
            match_quality="failed",
            bounded=False,
            sonic=False,
            starts_tried=5,
        )

    monkeypatch.setattr(fc, "calibrate_friction_coefs", failed)
    r = client.post(
        "/api/calibrate",
        json={"well": WELL, "params": _params(), "target_bhp": 1175.0, "test_whp": None},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["converged"] is False
    assert body["modeled_bhp"] is None
    assert body["bhp_error"] is None
