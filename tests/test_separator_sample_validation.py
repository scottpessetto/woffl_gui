"""Independent volume, timing and field-sample validation examples. Offline only."""

import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from server.main import app
from server.services.tools import sep_oil_loss as loss
from server.services.tools.oiw_validation import oil_fraction, sample_timestamp

T0 = pd.Timestamp("2026-08-17T08:00:00-08:00")


def history(rows):
    return pd.DataFrame([(tag, T0 + pd.Timedelta(seconds=sec), value) for tag, sec, value in rows], columns=["tag", "t", "value"])


def test_excursion_between_flow_reports_is_integrated():
    raw = history([(loss.FLOW_TAG, 0, 72000), (loss.FLOW_TAG, 600, 72000),
                   (loss.WC_TAG, 0, 100), (loss.WC_TAG, 120, 50), (loss.WC_TAG, 240, 100)])
    frame = loss._oil_rates(loss._grid(raw), 65000, 1.0)
    # Exactly two minutes of half-oil liquid at 72,000 BPD = 50 barrels.
    assert loss._barrels(frame.oil_upper, frame.dt_h) == pytest.approx(50)
    assert loss._barrels(frame.oil_raw, frame.dt_h) == pytest.approx(50)


def test_fraction_scenario_cannot_exceed_field_scenario():
    raw = history([(loss.FLOW_TAG, 0, 150000), (loss.FLOW_TAG, 300, 150000),
                   (loss.WC_TAG, 0, 100), (loss.WC_TAG, 60, 0)])
    frame = loss._oil_rates(loss._grid(raw), 1000, 1.0)
    assert frame.oil_lower.max() == frame.oil_upper.max() == 1000


def test_sustained_four_percent_indication_stays_visible_as_raw_volume(monkeypatch):
    raw = history([(tag, sec, val) for sec in range(0, 86401, 600)
                   for tag, val in ((loss.FLOW_TAG, 72000), (loss.WC_TAG, 96))])
    monkeypatch.setattr(loss, "_raw", lambda days: raw)
    p = loss.sep_oil_loss(days=1)["periods"][0]
    assert p["bbl_upper"] == 0
    assert p["bbl_raw"] == p["bbl_reference_removed"] == 2880


def test_no_future_reference_backfill():
    raw = history([(tag, sec, val) for sec in range(0, 3601, 60)
                   for tag, val in ((loss.FLOW_TAG, 72000), (loss.WC_TAG, 90 if sec < 600 else 100))])
    frame = loss._oil_rates(loss._grid(raw), 65000, .1)
    assert frame.base.iloc[0] == 90
    assert frame.base.iloc[-1] == 100


def test_midnight_splits_interval_without_creating_hours():
    raw = history([(loss.FLOW_TAG, 0, 72000), (loss.FLOW_TAG, 600, 72000), (loss.WC_TAG, 0, 50)])
    raw["t"] += pd.Timedelta(hours=15, minutes=55)
    frame = loss._oil_rates(loss._grid(raw), 65000, 1)
    daily = loss._daily_rows(frame, [], 65000)
    assert [d["bbl_raw"] for d in daily] == [125, 125]
    assert sum(d["hours"] for d in daily) == pytest.approx(1 / 6, abs=.01)


def test_invalid_wc_is_excluded_but_zero_is_preserved():
    raw = history([(loss.FLOW_TAG, sec, 72000) for sec in range(0, 301, 60)] +
                  [(loss.WC_TAG, 0, 100), (loss.WC_TAG, 60, -1), (loss.WC_TAG, 120, 101),
                   (loss.WC_TAG, 180, np.inf), (loss.WC_TAG, 240, 0)])
    grid = loss._grid(raw)
    assert grid.dt_h.sum() == pytest.approx(2 / 60)
    assert grid.loc[grid.wc == 0, "valid"].all()


def test_level_unknown_and_positive_deviation_are_not_at_setpoint():
    raw = history([(tag, sec, val) for sec in range(0, 1201, 60)
                   for tag, val in ((loss.FLOW_TAG, 72000), (loss.WC_TAG, 100 if sec < 600 else 50))])
    assert loss._events(loss._oil_rates(loss._grid(raw), 65000, .1))[0]["kind"] == "unknown"
    raw = pd.concat([raw, history([(loss.LEVEL_TAG, 0, 80), (loss.LEVEL_SP_TAG, 0, 50)])])
    assert loss._events(loss._oil_rates(loss._grid(raw), 65000, .1))[0]["kind"] == "off setpoint"


def test_mg_l_volume_conversion_and_density_requirement():
    # 850 mg oil in 1 litre, with oil density 850 kg/m3 = 1 mL = 0.1%.
    assert oil_fraction(850, "mg/L", 850) == pytest.approx(.001)
    assert oil_fraction(1000, "ppmv", None) == pytest.approx(.001)
    assert oil_fraction(1000, "unknown", None) is None
    with pytest.raises(ValueError, match="density"):
        oil_fraction(850, "mg/L", None)
    with pytest.raises(ValueError, match="outside"):
        oil_fraction(999999, "mg/L", 500)


@pytest.mark.parametrize("clock, expected", [("08:00", "08:00:00"), (.5, "12:00:00"), ("8:30 PM", "20:30:00"), (None, None), ("99:00", None)])
def test_excel_and_text_times(clock, expected):
    actual = sample_timestamp(pd.Timestamp("2026-08-17"), clock)
    assert (actual[11:19] if actual else None) == expected


def test_dst_times_are_not_guessed():
    assert sample_timestamp(pd.Timestamp("2026-03-08"), "02:30") is None
    assert sample_timestamp(pd.Timestamp("2026-11-01"), "01:30") is None


def post_log(monkeypatch, rows, **params):
    def forbidden(*args):
        raise AssertionError("Unexpected historian read")
    monkeypatch.setattr(loss, "_raw", forbidden)
    frame = pd.DataFrame(rows, columns=["Date", "Time", "Location", "PPM", "Method", "Notes"])
    response = TestClient(app).post("/api/tools/sep-oil-loss/samples", files={"file": ("samples.csv", frame.to_csv(index=False).encode())}, params=params)
    assert response.status_code == 200, response.text
    return response.json()


def test_units_must_be_confirmed_and_zero_grab_is_retained(monkeypatch):
    body = post_log(monkeypatch, [["2026-08-17", "08:00", "V-5317", 0, "Lab", "clean"]], location="V-5317", days=14)
    assert body["sample_count"] == 1
    assert body["daily"][0]["bbl"] is None
    assert body["daily"][0]["bopd_mean"] is None
    assert body["samples"][0]["status"] == "confirm units"
    assert body["samples"][0]["source_row"] == 2


def test_downstream_and_unknown_locations_never_validate_meter(monkeypatch):
    for tap in ("P-5417C", "V-9999"):
        body = post_log(monkeypatch, [["2026-08-17", "08:00", tap, 1000, "Lab", ""]], location=tap, units="ppmv", days=14)
        assert body["paired_count"] == 0
        assert body["samples"][0]["status"] == "different stream"
        if tap == "V-9999":
            assert any("unverified" in note for note in body["notes"])


def test_water_only_flow_uses_oil_water_ratio(monkeypatch):
    body = post_log(monkeypatch, [["2026-08-17", "08:00", "P-5417C", 100000, "Lab", ""]], units="ppmv", rate_basis="water", water_rate_bpd=9000)
    assert body["daily"][0]["bopd_mean"] == 1000


def test_pair_uses_backward_time_and_matched_flow_not_fallback(monkeypatch):
    raw = history([(loss.FLOW_TAG, 0, 72000), (loss.WC_TAG, 0, 99), (loss.WC_TAG, 180, 50)])
    monkeypatch.setattr(loss, "_raw", lambda days: raw)
    csv = b"Date,Time,Location,PPM,Method,Notes\n2026-08-17,08:04,V-5317,850,lab,after cleaning\n2026-08-17,09:00,V-5317,850,lab,late\n"
    response = TestClient(app).post("/api/tools/sep-oil-loss/samples", files={"file": ("samples.csv", csv)}, params={"location": "V-5317", "units": "mg/L", "oil_density_kgm3": 850, "days": 14, "lag_minutes": 2, "water_rate_bpd": 95000})
    assert response.status_code == 200, response.text
    body = response.json()
    row = body["samples"][0]
    assert row["sample_oil_bopd"] == pytest.approx(72)
    assert row["meter_oil_bopd"] == pytest.approx(720)
    assert row["error_pts"] == pytest.approx(.9)
    assert row["method"] == "lab" and row["notes"] == "after cleaning"
    assert row["wc_age_minutes"] == 2
    assert body["paired_count"] == body["stable_pair_count"] == 1
    assert body["samples"][1]["status"] == "no recent historian pair"
    assert body["median_error_pts"] == pytest.approx(.9)


def test_historian_failure_retains_sample_log(monkeypatch):
    def fail(days):
        raise RuntimeError("offline")
    monkeypatch.setattr(loss, "_raw", fail)
    response = TestClient(app).post("/api/tools/sep-oil-loss/samples", files={"file": ("samples.csv", b"Date,Time,Location,PPM\n2026-08-17,08:00,V-5317,1000\n")}, params={"location": "V-5317", "units": "ppmv", "days": 14})
    assert response.status_code == 200
    assert response.json()["sample_count"] == 1
    assert any("comparison unavailable" in n for n in response.json()["notes"])
