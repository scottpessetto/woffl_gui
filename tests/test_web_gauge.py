"""Memory-gauge web endpoints: XLSX parse/combine and the fit BHP override.

The parse math itself (units-row skip, minute-median downsample, daily
medians, multi-file dedupe) lives in woffl.gui.memory_gauge and runs
UNCHANGED under the endpoint - these tests feed synthetic gauge exports
through the real pipeline and pin the HTTP contract plus the one new rule:
IprFit applies gauge daily medians over test BHP before fitting.
"""

from __future__ import annotations

import io
from datetime import datetime, timedelta

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from server import schemas
from server.main import app
from server.services import ipr as ipr_svc


def _gauge_xlsx(start: datetime, days: int, base_psi: float, step_s: int = 300) -> bytes:
    """Synthetic exporter file: 'Data' sheet, units row, 5-min samples.

    Pressure = base + day index (so each day's median is base + i, exactly).
    """
    stamps: list[object] = ["M/d/yyyy HH:mm:ss"]  # units-descriptor row
    pressures: list[object] = ["psi"]
    for i in range(days):
        day = start + timedelta(days=i)
        for s in range(0, 24 * 3600, step_s):
            stamps.append(day + timedelta(seconds=s))
            pressures.append(base_psi + i)
    df = pd.DataFrame({"Date Time": stamps, "Pressure": pressures, "Temperature": 0})
    buf = io.BytesIO()
    df.to_excel(buf, sheet_name="Data", index=False)
    return buf.getvalue()


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def test_parse_single_file(client):
    blob = _gauge_xlsx(datetime(2026, 6, 1), days=3, base_psi=1100.0)
    r = client.post("/api/gauge/parse", files=[("files", ("g1.xlsx", blob))])
    assert r.status_code == 200
    body = r.json()
    assert body["start_date"] == "2026-06-01"
    assert body["end_date"] == "2026-06-03"
    assert [d["bhp"] for d in body["daily"]] == [1100.0, 1101.0, 1102.0]
    assert body["files"][0]["filename"] == "g1.xlsx"
    assert body["files"][0]["pressure_min"] == 1100.0
    assert body["files"][0]["pressure_max"] == 1102.0
    # raw sample count is pre-downsample: 3 days of 5-min samples
    assert body["sample_count"] == 3 * (24 * 3600 // 300)


def test_parse_combines_files(client):
    b1 = _gauge_xlsx(datetime(2026, 6, 1), days=2, base_psi=1100.0)
    b2 = _gauge_xlsx(datetime(2026, 6, 3), days=2, base_psi=1200.0)
    r = client.post(
        "/api/gauge/parse",
        files=[("files", ("a.xlsx", b1)), ("files", ("b.xlsx", b2))],
    )
    assert r.status_code == 200
    body = r.json()
    assert body["start_date"] == "2026-06-01"
    assert body["end_date"] == "2026-06-04"
    assert [d["date"] for d in body["daily"]] == [
        "2026-06-01",
        "2026-06-02",
        "2026-06-03",
        "2026-06-04",
    ]
    assert len(body["files"]) == 2


def test_parse_rejects_garbage(client):
    r = client.post("/api/gauge/parse", files=[("files", ("junk.xlsx", b"not an xlsx"))])
    assert r.status_code == 422
    assert "junk.xlsx" in r.json()["detail"]["message"]


# ---------------------------------------------------------------------------
# Fit override
# ---------------------------------------------------------------------------


def _test_frame() -> pd.DataFrame:
    """Two tests WITHOUT BHP (the gauge-well case) + one with."""
    return pd.DataFrame(
        {
            "well": ["MPX-01"] * 3,
            "wt_uid": [1.0, 2.0, 3.0],
            "WtDate": pd.to_datetime(["2026-06-01", "2026-06-02", "2026-06-04"]),
            "WtTotalFluid": [1500.0, 1400.0, 1300.0],
            "BHP": [None, None, 900.0],
            "form_wc": [0.8, 0.8, 0.8],
            "fgor": [300.0, 300.0, 300.0],
            "whp": [200.0, 200.0, 200.0],
        }
    )


def test_fit_uses_gauge_bhp(monkeypatch):
    """Without overrides the frame has ONE usable test (fit impossible);
    gauge coverage makes all three usable and anchors on gauge BHP."""
    monkeypatch.setattr(ipr_svc.tests, "tests_for_well", lambda well, months, cap: _test_frame())

    req = schemas.IprFitRequest(well="MPX-01", anchor_mode="specific", anchor_date="2026-06-02")
    with pytest.raises(ValueError):
        ipr_svc.fit(req)

    req_gauge = schemas.IprFitRequest(
        well="MPX-01",
        anchor_mode="specific",
        anchor_date="2026-06-02",
        bhp_overrides=[
            schemas.GaugeDay(date="2026-06-01", bhp=1050.0),
            schemas.GaugeDay(date="2026-06-02", bhp=1000.0),
        ],
    )
    out = ipr_svc.fit(req_gauge)
    # The anchored test's BHP is the GAUGE median, not the (missing) feed.
    assert out["coeffs"]["pwf"] == 1000.0
    assert out["coeffs"]["qwf"] == 1400.0


def test_fit_gauge_wins_inside_coverage(monkeypatch):
    """A test WITH a feed BHP still takes the gauge value when covered."""
    monkeypatch.setattr(ipr_svc.tests, "tests_for_well", lambda well, months, cap: _test_frame())
    req = schemas.IprFitRequest(
        well="MPX-01",
        anchor_mode="specific",
        anchor_date="2026-06-04",
        bhp_overrides=[
            schemas.GaugeDay(date="2026-06-02", bhp=1000.0),
            schemas.GaugeDay(date="2026-06-04", bhp=950.0),
        ],
    )
    out = ipr_svc.fit(req)
    assert out["coeffs"]["pwf"] == 950.0  # gauge 950 beats feed 900
