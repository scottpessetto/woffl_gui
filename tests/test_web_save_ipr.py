"""Web save-ipr endpoints: gate chain, attribution, and the prop payload.

The write RULES (prop_xref whitelist, as-built rejection, WC cap, friction
only-when-changed) live in woffl.gui.ipr_anchor / prop_hist_client and are
unit-tested there. These tests pin the SERVER contract layered on top:

* the 403 gate pre-check (no push ever happens with the gate off),
* per-request X-Forwarded-Email attribution via the contextvar provider
  registered in server/identity.py,
* the exact prop_ids one save writes - and that none is an as-built
  physical property,
* the single shared batch stamp + the engineer comment joined on it,
* the values-only save (no pinnable anchor) and the cleared-marker un-pin.

push_prop / push_eng_comment / load_saved_ipr are monkeypatched at the
ipr_anchor namespace (the names its save functions actually call), so no
test can reach Databricks. resolve_entry_user runs REAL to exercise the
provider chain - the forwarded header satisfies it before the SQL fallback.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import woffl.gui.ipr_anchor as ipr_anchor
from woffl.assembly import prop_hist_client as phc
from woffl.assembly.prop_hist_client import AS_BUILT_PROP_IDS

import server.identity as identity
import server.services.ipr as ipr_svc
from server.main import app

WELL = "MPL-20"
HEADERS = {"X-Forwarded-Email": "engineer@example.com"}

PAYLOAD = {
    "qwf_liq": 1940.0,
    "pwf": 812.0,
    "res_pres": 1559.0,
    "form_wc": 0.83,
    "form_gor": 320.0,
    "surf_pres": 210.0,
    # sidebar defaults: never materialized as rows when nothing is stored
    "ken": 0.03,
    "kth": 0.3,
    "kdi": 0.4,
    "comment": "anchored on the 7/25 test",
    "pin_wt_uid": 123456.0,
    "pin_date": "2026-07-25",
}


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture()
def recorder(monkeypatch):
    """Capture every would-be prop_hist write; block the Databricks reads."""
    pushes: list[dict] = []
    comments: list[dict] = []

    def fake_push(well_name, prop_id, value, entry_user, entry_datetime=None):
        pushes.append(
            {
                "well": well_name,
                "prop_id": prop_id,
                "value": value,
                "entry_user": entry_user,
                "entry_datetime": entry_datetime,
            }
        )
        return 1

    def fake_comment(well_name, entry_datetime, entry_user, note):
        comments.append(
            {
                "well": well_name,
                "entry_datetime": entry_datetime,
                "entry_user": entry_user,
                "note": note,
            }
        )
        return 1

    monkeypatch.setattr(ipr_anchor, "push_prop", fake_push)
    monkeypatch.setattr(ipr_anchor, "push_eng_comment", fake_comment)
    # "nothing stored yet" - the friction only-when-changed rule reads this
    monkeypatch.setattr(ipr_anchor, "load_saved_ipr", lambda well: None)
    monkeypatch.delenv("WOFFL_ENTRY_USER", raising=False)
    # The provider slot is process-global and other suites (the
    # prop_hist_client provider tests) reset it to None and leave the
    # current_user cache populated - restore OUR provider and drop the cache
    # so attribution comes from the forwarded header, order-independent.
    identity.register_entry_user_provider()
    monkeypatch.setitem(phc._entry_user_cache, "value", None)
    ipr_svc._saved_ipr.cache_clear()
    return pushes, comments


@pytest.fixture()
def gate_on(monkeypatch):
    monkeypatch.setenv("ALLOW_DATABRICKS_WRITES", "true")


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


def test_save_requires_write_gate(client, recorder, monkeypatch):
    monkeypatch.delenv("ALLOW_DATABRICKS_WRITES", raising=False)
    r = client.post(f"/api/wells/{WELL}/save-ipr", json=PAYLOAD, headers=HEADERS)
    assert r.status_code == 403
    assert r.json()["detail"]["error"] == "writes_disabled"
    pushes, comments = recorder
    assert pushes == [] and comments == []


def test_clear_pin_requires_write_gate(client, recorder, monkeypatch):
    monkeypatch.delenv("ALLOW_DATABRICKS_WRITES", raising=False)
    r = client.delete(f"/api/wells/{WELL}/ipr-pin", headers=HEADERS)
    assert r.status_code == 403
    pushes, _ = recorder
    assert pushes == []


# ---------------------------------------------------------------------------
# The full save: pin + values + comment
# ---------------------------------------------------------------------------


def test_full_save_payload(client, recorder, gate_on):
    r = client.post(f"/api/wells/{WELL}/save-ipr", json=PAYLOAD, headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["pinned"] is True
    assert body["pin_skipped"] is False
    assert body["n_values"] == 6  # curve + rate + fluids; default friction skipped

    pushes, comments = recorder
    by_id = {p["prop_id"]: p for p in pushes}

    # Exactly the IPR/value ids plus the anchor pin - never a physical prop.
    assert set(by_id) == {
        "ipr_wt_uid",
        "ipr_qwf_liq",
        "ipr_pwf",
        "form_wc",
        "form_gor",
        "surf_press",
        "resvr_press",
    }
    assert not set(by_id) & AS_BUILT_PROP_IDS

    assert by_id["ipr_wt_uid"]["value"] == 123456.0
    assert by_id["ipr_qwf_liq"]["value"] == 1940.0  # liquid, verbatim
    assert by_id["resvr_press"]["value"] == 1559.0

    # Attribution: every row stamped with the forwarded engineer, not the SP.
    assert {p["entry_user"] for p in pushes} == {"engineer@example.com"}

    # One shared batch stamp across all VALUE rows; the comment joins on it.
    stamps = {p["entry_datetime"] for p in pushes if p["prop_id"] != "ipr_wt_uid"}
    assert len(stamps) == 1 and None not in stamps
    assert len(comments) == 1
    assert comments[0]["entry_datetime"] == stamps.pop()
    assert comments[0]["note"] == "anchored on the 7/25 test"


def test_changed_friction_rides_along(client, recorder, gate_on):
    payload = dict(PAYLOAD, ken=0.055, comment=None)
    r = client.post(f"/api/wells/{WELL}/save-ipr", json=payload, headers=HEADERS)
    assert r.status_code == 200
    pushes, comments = recorder
    by_id = {p["prop_id"]: p for p in pushes}
    assert by_id["jpfric_entry"]["value"] == 0.055
    # kth/kdi still sit at the uncalibrated defaults with nothing stored: skipped
    assert "jpfric_throat" not in by_id and "jpfric_diffuser" not in by_id
    assert comments == []  # no note supplied


def test_wc_capped_at_099(client, recorder, gate_on):
    r = client.post(
        f"/api/wells/{WELL}/save-ipr", json=dict(PAYLOAD, form_wc=1.0), headers=HEADERS
    )
    assert r.status_code == 200
    pushes, _ = recorder
    by_id = {p["prop_id"]: p for p in pushes}
    assert by_id["form_wc"]["value"] == 0.99


def test_values_only_save_without_pin(client, recorder, gate_on):
    payload = dict(PAYLOAD, pin_wt_uid=None, pin_date=None)
    r = client.post(f"/api/wells/{WELL}/save-ipr", json=payload, headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["pinned"] is False
    assert body["pin_message"] is None
    pushes, _ = recorder
    assert "ipr_wt_uid" not in {p["prop_id"] for p in pushes}


# ---------------------------------------------------------------------------
# Un-pin
# ---------------------------------------------------------------------------


def test_clear_pin_writes_cleared_marker(client, recorder, gate_on):
    r = client.delete(f"/api/wells/{WELL}/ipr-pin", headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["cleared"] is True
    pushes, _ = recorder
    assert len(pushes) == 1
    assert pushes[0]["prop_id"] == "ipr_wt_uid"
    assert pushes[0]["value"] == ipr_anchor.PIN_CLEARED_VALUE
    assert pushes[0]["entry_user"] == "engineer@example.com"


# ---------------------------------------------------------------------------
# WC/GOR/ResP lock toggles
# ---------------------------------------------------------------------------


def test_prop_lock_requires_write_gate(client, recorder, monkeypatch):
    monkeypatch.delenv("ALLOW_DATABRICKS_WRITES", raising=False)
    r = client.post(
        f"/api/wells/{WELL}/prop-lock",
        json={"field": "form_wc", "locked": True, "value": 0.83},
        headers=HEADERS,
    )
    assert r.status_code == 403
    pushes, _ = recorder
    assert pushes == []


def test_lock_pins_value_then_lock_row(client, recorder, gate_on):
    r = client.post(
        f"/api/wells/{WELL}/prop-lock",
        json={"field": "form_wc", "locked": True, "value": 1.0},
        headers=HEADERS,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["locked"] is True
    assert body["value"] == 0.99  # echoed as stored: WC re-capped

    pushes, _ = recorder
    # Value row FIRST (capped), then the 1.0 lock row - both attributed.
    assert [(p["prop_id"], p["value"]) for p in pushes] == [
        ("form_wc", 0.99),
        ("form_wc_lock", 1.0),
    ]
    assert {p["entry_user"] for p in pushes} == {"engineer@example.com"}
    assert not {p["prop_id"] for p in pushes} & AS_BUILT_PROP_IDS


def test_lock_resp_uses_registry_prop_ids(client, recorder, gate_on):
    r = client.post(
        f"/api/wells/{WELL}/prop-lock",
        json={"field": "res_pres", "locked": True, "value": 1800.0},
        headers=HEADERS,
    )
    assert r.status_code == 200
    pushes, _ = recorder
    assert [(p["prop_id"], p["value"]) for p in pushes] == [
        ("resvr_press", 1800.0),
        ("resvr_press_lock", 1.0),
    ]


def test_unlock_writes_unlocked_marker_only(client, recorder, gate_on):
    r = client.post(
        f"/api/wells/{WELL}/prop-lock",
        json={"field": "form_gor", "locked": False, "value": None},
        headers=HEADERS,
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["locked"] is False and body["value"] is None
    pushes, _ = recorder
    assert [(p["prop_id"], p["value"]) for p in pushes] == [
        ("form_gor_lock", ipr_anchor.LOCK_UNLOCKED_VALUE),
    ]


def test_lock_rejects_unknown_field(client, recorder, gate_on):
    r = client.post(
        f"/api/wells/{WELL}/prop-lock",
        json={"field": "jpump_tvd", "locked": True, "value": 4000.0},
        headers=HEADERS,
    )
    assert r.status_code == 422  # Literal validation - physical props unreachable
    pushes, _ = recorder
    assert pushes == []
