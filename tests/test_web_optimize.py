"""GET /optimize/pad-status contract: readiness rows per pad + donor extras.

The saved-IPR assembly itself (_assemble_saved_ipr) is unit-tested with the
prop_hist client suites; these tests pin the board's shape: which wells
appear, how a saved/missing fit serializes, and that donor wells from OTHER
pads ride in via `extra`.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import woffl.gui.ipr_anchor as ipr_anchor

import server.services.ipr as ipr_svc
import server.services.wells as wells_svc
from server.main import app

_UNIVERSE = {
    "wells": [
        {"name": "MPL-01", "pad": "L", "is_sch": False},
        {"name": "MPL-20", "pad": "L", "is_sch": False},
        {"name": "MPB-28", "pad": "B", "is_sch": True},
    ],
    "source": "databricks",
}

_SAVED = {
    "MPL-20": {
        "values": {"qwf_liq": 1068.0, "pwf": 500.0},
        "friction": {"ken": 0.014, "kth": 0.497},
        "locks": {"form_wc": True, "form_gor": False, "res_pres": False},
        "lock_values": {},
        "saved_at": pd.Timestamp("2026-08-05 17:00:00"),
        "saved_by": "engineer@example.com",
        "pin_at": pd.Timestamp("2026-07-25 12:00:00"),
        "pin_value": 123.0,
        "pin_user": "engineer@example.com",
    },
    # MPB-28: friction-only characterization, no saved curve
    "MPB-28": {
        "values": {},
        "friction": {"kdi": 0.62},
        "locks": {"form_wc": False, "form_gor": False, "res_pres": False},
        "lock_values": {},
        "saved_at": None,
        "saved_by": None,
        "pin_at": None,
        "pin_value": None,
        "pin_user": None,
    },
}


@pytest.fixture()
def client(monkeypatch) -> TestClient:
    monkeypatch.setattr(wells_svc, "list_wells", lambda: _UNIVERSE)
    monkeypatch.setattr(ipr_anchor, "warm_saved_ipr_cache", lambda force=False: 0)
    monkeypatch.setattr(ipr_anchor, "load_saved_ipr", lambda well: _SAVED.get(well))
    ipr_svc._pad_fit.cache_clear()
    return TestClient(app)


def test_pad_rows_and_readiness(client):
    r = client.get("/api/optimize/pad-status?pad=L")
    assert r.status_code == 200
    body = r.json()
    assert body["pad"] == "L"
    rows = {w["well"]: w for w in body["wells"]}
    assert set(rows) == {"MPL-01", "MPL-20"}  # only L-pad wells

    fitted = rows["MPL-20"]
    assert fitted["has_curve"] is True
    assert fitted["saved_at"].startswith("2026-08-05")
    assert fitted["saved_by"] == "engineer@example.com"
    assert fitted["has_friction"] is True
    assert fitted["friction_keys"] == ["ken", "kth"]
    assert fitted["locks"]["form_wc"] is True
    assert fitted["pin_at"].startswith("2026-07-25")

    bare = rows["MPL-01"]  # never saved anything
    assert bare["has_curve"] is False
    assert bare["saved_at"] is None
    assert bare["has_friction"] is False


def test_extra_donor_from_other_pad(client):
    r = client.get("/api/optimize/pad-status?pad=L&extra=MPB-28")
    assert r.status_code == 200
    body = r.json()
    extras = {w["well"]: w for w in body["extras"]}
    assert set(extras) == {"MPB-28"}
    donor = extras["MPB-28"]
    assert donor["pad"] == "B"  # donors keep their own pad
    assert donor["has_curve"] is False  # friction-only characterization
    assert donor["has_friction"] is True
    assert donor["friction_keys"] == ["kdi"]
