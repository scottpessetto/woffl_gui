"""POST /match-test contract: argument mapping and result serialization.

The fit itself (seed scan, scaled Nelder-Mead, sonic / failed branches)
lives in woffl.gui.gaugeless_match and has its own tests; these pin the
SERVER contract: the request params feed the same sim-object factories as
a solve, the inflow is rebuilt per trial on the OIL basis through
factories.create_inflow at the sidebar's reservoir pressure, test-day WHP
and PF pressure win over the sidebar, knz stays at 0.01, ken is held at the
sidebar seed, water mode is refused, and NaN diagnostics serialize as null.
"""

from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

import woffl.gui.gaugeless_match as gm
from woffl.gui.gaugeless_match import GaugelessMatchResult

from server.main import app

WELL = "MPB-28"


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture()
def captured(monkeypatch):
    calls: list[dict] = []

    def fake_match(**kwargs):
        calls.append(kwargs)
        return GaugelessMatchResult(
            well_name=kwargs["well_name"],
            oil_test=kwargs["oil_test"],
            water_test=kwargs["water_test"],
            pf_test=kwargs["pf_test"],
            pwf=912.0,
            qwf_liq=kwargs["oil_test"] + kwargs["water_test"],
            form_wc=kwargs["water_test"] / (kwargs["oil_test"] + kwargs["water_test"]),
            kth=0.41,
            kdi=0.52,
            ken=kwargs["ken"],
            knz=kwargs["knz"],
            modeled_bhp=915.0,
            modeled_oil=170.0,
            modeled_water=486.0,
            modeled_pf=2870.0,
            score=0.005,
            oil_error_pct=-0.6,
            pf_error_pct=0.2,
            match_quality="good",
            converged=True,
            seed_pwf=930.0,
            scan=[{"pwf": 360.0, "psu": 400.0, "oil": 300.0, "pf": 3100.0, "sonic": False}],
            iterations=80,
            starts_tried=1,
            pf_reachable=False,
            pf_model_min=2250.0,
            pf_model_max=2390.0,
            area_factor_needed=1.42,
        )

    monkeypatch.setattr(gm, "match_test", fake_match)
    return calls


def _params(**over) -> dict:
    base: dict = {
        "ken": 0.08,
        "kth": 0.33,
        "kdi": 0.44,
        "surf_pres": 210.0,
        "ppf_surf": 3168.0,
        "form_temp": 120.0,
        "pres": 1800.0,
    }
    base.update(over)
    return base


def test_match_test_maps_args(client, captured):
    r = client.post(
        "/api/match-test",
        json={
            "well": WELL,
            "params": _params(),
            "test_oil": 171.0,
            "test_water": 489.0,
            "test_pf": 2863.0,
            "test_whp": 464.0,
            "test_pf_press": 3050.0,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["match_quality"] == "good"
    assert body["pwf"] == 912.0
    assert body["qwf_liq"] == 660.0
    assert body["form_wc"] == pytest.approx(489.0 / 660.0)
    assert (body["kth"], body["kdi"], body["ken"]) == (0.41, 0.52, 0.08)
    assert body["pwh_used"] == 464.0 and body["ppf_surf_used"] == 3050.0
    assert body["scan"][0]["pf"] == 3100.0
    assert "nozzle" in body["caveat"]
    assert body["pf_reachable"] is False
    assert (body["pf_model_min"], body["pf_model_max"]) == (2250.0, 2390.0)
    # the unreachable branch's one hardware number rides through to the client
    assert body["area_factor_needed"] == 1.42

    call = captured[0]
    assert call["well_name"] == WELL
    assert (call["oil_test"], call["water_test"], call["pf_test"]) == (171.0, 489.0, 2863.0)
    assert call["pwh"] == 464.0  # test-day WHP wins
    assert call["ppf_surf"] == 3050.0  # test-day PF pressure wins
    assert call["knz"] == 0.01  # held constant
    assert call["ken"] == 0.08  # sidebar seed, held
    assert (call["seed_kth"], call["seed_kdi"]) == (0.33, 0.44)
    assert call["pres"] == 1800.0
    # the inflow factory anchors on the OIL basis at the sidebar's pres
    ipr = call["make_inflow"](171.0, 912.0)
    assert ipr.qwf == pytest.approx(171.0)
    assert ipr.pwf == pytest.approx(912.0)
    assert ipr.pres == pytest.approx(1800.0)
    # geometry is prebuilt and passed opaque - never a free variable
    assert {"jpump_md", "casing_out_dia", "tubing_od"}.isdisjoint(call.keys())


def test_sidebar_conditions_when_test_day_values_missing(client, captured):
    r = client.post(
        "/api/match-test",
        json={"well": WELL, "params": _params(), "test_oil": 171.0, "test_water": 489.0, "test_pf": 2863.0},
    )
    assert r.status_code == 200
    call = captured[0]
    assert call["pwh"] == 210.0 and call["ppf_surf"] == 3168.0


def test_water_mode_is_refused(client, captured):
    r = client.post(
        "/api/match-test",
        json={
            "well": WELL,
            "params": _params(model_as_water=True, form_wc=1.0),
            "test_oil": 171.0,
            "test_water": 489.0,
            "test_pf": 2863.0,
        },
    )
    assert r.status_code in (400, 422)
    assert captured == []


def test_nan_diagnostics_serialize_as_null(client, monkeypatch):
    def failed(**kwargs):
        return GaugelessMatchResult(
            well_name=kwargs["well_name"],
            oil_test=kwargs["oil_test"],
            water_test=kwargs["water_test"],
            pf_test=kwargs["pf_test"],
            pwf=math.nan,
            qwf_liq=660.0,
            form_wc=0.74,
            kth=0.33,
            kdi=0.44,
            ken=0.08,
            knz=0.01,
            modeled_bhp=math.nan,
            modeled_oil=math.nan,
            modeled_water=math.nan,
            modeled_pf=math.nan,
            score=math.inf,
            oil_error_pct=math.nan,
            pf_error_pct=math.nan,
            match_quality="failed",
            converged=False,
            message="the pump model found no operating point anywhere in the BHP window",
        )

    monkeypatch.setattr(gm, "match_test", failed)
    r = client.post(
        "/api/match-test",
        json={"well": WELL, "params": _params(), "test_oil": 171.0, "test_water": 489.0, "test_pf": 2863.0},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["match_quality"] == "failed"
    assert body["pwf"] is None and body["modeled_bhp"] is None and body["score"] is None
    assert body["area_factor_needed"] is None  # nothing to explain when nothing solved
    assert "no operating point" in body["message"]


def test_rejects_a_test_without_oil(client, captured):
    r = client.post(
        "/api/match-test",
        json={"well": WELL, "params": _params(), "test_oil": 0.0, "test_water": 489.0, "test_pf": 2863.0},
    )
    assert r.status_code == 422
    assert captured == []
