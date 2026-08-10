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

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import woffl.gui.ipr_anchor as ipr_anchor
from woffl.assembly import databricks_client
from woffl.assembly import prop_hist_client as phc
from woffl.assembly.prop_hist_client import AS_BUILT_PROP_IDS

import server.identity as identity
import server.services.database as db_svc
import server.services.ipr as ipr_svc
import server.services.wells as wells_svc
from server.main import app

WELL = "MPL-20"
_ENTHID = 4711  # fake enthid the pinned well_enthid_map hands back
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

    def fake_push_many(well_name, values, entry_user, entry_datetime=None):
        for prop_id, value in values.items():
            fake_push(well_name, prop_id, value, entry_user, entry_datetime)
        return len(values)

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
    monkeypatch.setattr(ipr_anchor, "push_props", fake_push_many)
    monkeypatch.setattr(ipr_anchor, "push_eng_comment", fake_comment)
    # "nothing stored yet" - the friction only-when-changed rule reads this
    monkeypatch.setattr(ipr_anchor, "load_saved_ipr", lambda well: None)
    # evict_prop_history resolves well -> enthid through this map; the real
    # one opens a Databricks connection (whose load_dotenv flips the write
    # gate on - the AGENTS.md section-3 landmine), so pin it to a fake id.
    monkeypatch.setattr(phc, "well_enthid_map", lambda force_refresh=False: {WELL: _ENTHID})
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


def test_characterization_rides_along_only_when_sent(client, recorder, gate_on):
    """bubble_point / form_temp are canonical props (resvr_bubb / resvr_temp).
    The Solver sends one only after the engineer moved it off the seeded
    value - a sensitivity permutation varies both - so an ordinary save must
    write neither, and a changed one must land under the same batch stamp."""
    r = client.post(f"/api/wells/{WELL}/save-ipr", json=PAYLOAD, headers=HEADERS)
    assert r.status_code == 200
    pushes, _ = recorder
    assert {"resvr_bubb", "resvr_temp"} & {p["prop_id"] for p in pushes} == set()

    pushes.clear()
    payload = dict(PAYLOAD, bubble_point=2100.0, form_temp=185.0, comment=None)
    r = client.post(f"/api/wells/{WELL}/save-ipr", json=payload, headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["n_values"] == 8  # the six curve rows plus these two
    by_id = {p["prop_id"]: p for p in pushes}
    assert by_id["resvr_bubb"]["value"] == 2100.0
    assert by_id["resvr_temp"]["value"] == 185.0
    stamps = {p["entry_datetime"] for p in pushes if p["prop_id"] != "ipr_wt_uid"}
    assert len(stamps) == 1


def test_characterization_bounds_are_enforced(client, recorder, gate_on):
    """BlackOil only validates a bubble point in 1001-2999; the request must
    refuse an out-of-range value rather than push it into the pivots."""
    r = client.post(
        f"/api/wells/{WELL}/save-ipr", json=dict(PAYLOAD, bubble_point=3500.0), headers=HEADERS
    )
    assert r.status_code == 422
    pushes, _ = recorder
    assert pushes == []


def test_manual_point_save_clears_the_pin_instead_of_setting_one(client, recorder, gate_on):
    """A manual anchor is NOT a well test. Saving one must drop any pinned
    test - a surviving pin makes the next open read the curve as
    test-anchored (server.services.wells labels it "saved" vs "manual") and
    flips the anchor selector back to that test."""
    payload = dict(PAYLOAD, unpin=True, pin_wt_uid=None, pin_date=None, comment=None)
    r = client.post(f"/api/wells/{WELL}/save-ipr", json=payload, headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["pinned"] is False
    assert body["n_values"] == 6

    pushes, _ = recorder
    pin_rows = [p for p in pushes if p["prop_id"] == "ipr_wt_uid"]
    assert len(pin_rows) == 1
    # The cleared marker, never a real wt_uid and never a negative sentinel.
    assert pin_rows[0]["value"] == ipr_anchor.PIN_CLEARED_VALUE
    # Cleared FIRST, so the values carry the later stamp and win
    # ipr_anchor.saved_wins - otherwise the curve would read as test-anchored.
    # Stamps are strictly increasing per process (next_entry_datetime, pinned
    # by test_prop_hist_client), so push ORDER is the observable contract:
    # clear_ipr_pin lets push_prop allocate its own stamp, which is why the
    # recorder sees entry_datetime=None on that row.
    assert pushes[0]["prop_id"] == "ipr_wt_uid"
    assert len({p["entry_datetime"] for p in pushes if p["prop_id"] != "ipr_wt_uid"}) == 1


def test_unpin_and_a_pin_request_cannot_both_happen(client, recorder, gate_on):
    """unpin wins: the client only sets it for a manual point, where any
    pin_wt_uid it still carries is a leftover from the anchor selector."""
    payload = dict(PAYLOAD, unpin=True, comment=None)  # pin_wt_uid still 123456
    r = client.post(f"/api/wells/{WELL}/save-ipr", json=payload, headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["pinned"] is False
    pushes, _ = recorder
    pin_values = [p["value"] for p in pushes if p["prop_id"] == "ipr_wt_uid"]
    assert pin_values == [ipr_anchor.PIN_CLEARED_VALUE]


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


# ---------------------------------------------------------------------------
# Cache invalidation - a write must show on the NEXT poll, not after the TTL
# ---------------------------------------------------------------------------
#
# Review sessions run TWO browser clients: the one that saves invalidates its
# own react-query cache, but the other client's next pad-status poll hits the
# SERVER cache - only server-side eviction can keep that poll truthful.

_UNIVERSE = {"wells": [{"name": WELL, "pad": "L", "is_sch": False}], "source": "databricks"}


def _fit(qwf_liq: float) -> dict:
    """Minimal _assemble_saved_ipr record - enough for ipr_svc._fit_row."""
    return {
        "values": {"qwf_liq": qwf_liq, "pwf": 500.0},
        "friction": {},
        "locks": {},
        "lock_values": {},
        "saved_at": None,
        "saved_by": None,
        "pin_at": None,
        "pin_value": None,
        "pin_user": None,
    }


@pytest.fixture()
def pad_board(monkeypatch):
    """Pad-status board backed by a mutable store, plus a recompute counter.

    `saved` plays prop_hist: a test mutates it to make a write "land", then
    asserts whether the next pad-status poll re-reads it (cache dropped) or
    replays the cached payload (cache kept). `computes` counts _pad_fit
    recomputes via its warm_saved_ipr_cache call.
    """
    saved: dict[str, dict] = {}
    computes: list[int] = []
    monkeypatch.setattr(wells_svc, "list_wells", lambda: _UNIVERSE)
    monkeypatch.setattr(
        ipr_anchor, "warm_saved_ipr_cache", lambda force=False: computes.append(1) or 0
    )
    monkeypatch.setattr(ipr_anchor, "load_saved_ipr", lambda well: saved.get(well))
    ipr_svc._saved_ipr.cache_clear()
    ipr_svc._pad_fit.cache_clear()
    return saved, computes


def _board_row(client) -> dict:
    r = client.get("/api/optimize/pad-status?pad=L")
    assert r.status_code == 200
    return r.json()["wells"][0]


def test_save_refreshes_pad_status_on_next_poll(client, recorder, pad_board, gate_on):
    """A successful save drops the board's TTL entry: the next poll serves
    the new fit instead of replaying the pre-save payload for 5 minutes."""
    saved, computes = pad_board
    assert _board_row(client)["has_curve"] is False  # primes the cache
    assert computes == [1]

    saved[WELL] = _fit(1940.0)  # the write lands in "prop_hist"
    r = client.post(f"/api/wells/{WELL}/save-ipr", json=PAYLOAD, headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["n_values"] > 0

    assert _board_row(client)["has_curve"] is True  # fresh, same process, no TTL wait
    assert len(computes) == 2  # exactly one recompute, forced by the save


def test_clear_pin_refreshes_pad_status_on_next_poll(client, recorder, pad_board, gate_on):
    saved, computes = pad_board
    saved[WELL] = dict(
        _fit(1940.0),
        pin_at=pd.Timestamp("2026-07-25 12:00:00"),
        pin_value=123456.0,
        pin_user="engineer@example.com",
    )
    assert _board_row(client)["pin_at"] is not None

    saved[WELL] = _fit(1940.0)  # the cleared-marker row lands: pin gone
    r = client.delete(f"/api/wells/{WELL}/ipr-pin", headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["cleared"] is True

    assert _board_row(client)["pin_at"] is None
    assert len(computes) == 2


def test_prop_lock_refreshes_pad_status_on_next_poll(client, recorder, pad_board, gate_on):
    saved, computes = pad_board
    assert _board_row(client)["locks"] == {}

    saved[WELL] = dict(_fit(1940.0), locks={"form_wc": True})
    r = client.post(
        f"/api/wells/{WELL}/prop-lock",
        json={"field": "form_wc", "locked": True, "value": 0.83},
        headers=HEADERS,
    )
    assert r.status_code == 200
    assert r.json()["ok"] is True

    assert _board_row(client)["locks"] == {"form_wc": True}
    assert len(computes) == 2


def test_failed_save_keeps_the_cached_board(client, recorder, pad_board, gate_on, monkeypatch):
    """A failed write changed nothing - it must NOT cost the board (or any
    other reader) its cache entry."""
    saved, computes = pad_board
    assert _board_row(client)["has_curve"] is False

    def refuse(*args, **kwargs):
        raise RuntimeError("warehouse down")

    monkeypatch.setattr(ipr_anchor, "push_prop", refuse)
    monkeypatch.setattr(ipr_anchor, "push_props", refuse)

    r = client.post(f"/api/wells/{WELL}/save-ipr", json=PAYLOAD, headers=HEADERS)
    assert r.status_code == 200  # per-part failures ride in the messages
    body = r.json()
    assert body["pinned"] is False
    assert body["n_values"] == 0

    r = client.delete(f"/api/wells/{WELL}/ipr-pin", headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["cleared"] is False

    assert _board_row(client)["has_curve"] is False
    assert computes == [1]  # both polls served from cache: nothing evicted


def test_save_evicts_prop_history_audit_entry(client, recorder, gate_on, monkeypatch):
    """The audit page's per-enthid TTL entry drops with the save too."""
    queries: list[str] = []
    monkeypatch.setattr(
        databricks_client, "execute_query", lambda sql: queries.append(sql) or pd.DataFrame()
    )
    db_svc._prop_history.cache_clear()
    db_svc._prop_history(_ENTHID)
    db_svc._prop_history(_ENTHID)
    assert len(queries) == 1  # cached

    r = client.post(f"/api/wells/{WELL}/save-ipr", json=PAYLOAD, headers=HEADERS)
    assert r.status_code == 200

    db_svc._prop_history(_ENTHID)
    assert len(queries) == 2  # evicted: the next audit read re-SELECTs


def test_save_survives_enthid_lookup_failure(client, recorder, gate_on, monkeypatch):
    """Eviction must never fail a landed write: an enthid-map failure
    degrades to clearing the whole prop-history cache, not a 500."""
    queries: list[str] = []
    monkeypatch.setattr(
        databricks_client, "execute_query", lambda sql: queries.append(sql) or pd.DataFrame()
    )
    db_svc._prop_history.cache_clear()
    db_svc._prop_history(_ENTHID)
    assert len(queries) == 1

    def refuse(force_refresh=False):
        raise RuntimeError("no warehouse")

    monkeypatch.setattr(phc, "well_enthid_map", refuse)
    r = client.post(f"/api/wells/{WELL}/save-ipr", json=PAYLOAD, headers=HEADERS)
    assert r.status_code == 200

    db_svc._prop_history(_ENTHID)
    assert len(queries) == 2  # fallback cleared the whole cache
