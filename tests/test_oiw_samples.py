"""Operator OIW grab-sample upload: parse, roll-up and the HTTP contract.

The sample log is a hand-kept spreadsheet, so these tests feed the real
endpoint synthetic workbooks in that exact shape - two header rows, blank
spacer rows, text in numeric cells, a nonsense far-future date - and pin
three things: the daily math (ppm x water rate / 1e6, unweighted mean per
Alaska calendar day), the fact that no junk row can raise, and that the
DOWNSTREAM-of-the-deoilers caveat rides along with every location except
V-5317, which is the one tap on the same stream as the calculated band.

Nothing here touches Databricks: the endpoint parses the upload and returns
it, storing nothing (the same contract as POST /api/gauge/parse).
"""

from __future__ import annotations

import io
from typing import Any, Optional

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from server.main import app

PATH = "/api/tools/sep-oil-loss/samples"


def _workbook(rows: list[list[Any]], sheet: str = "OIW Daily") -> bytes:
    """One sample-log sheet: a title row, then the header row, then rows.

    Args:
        rows (list): Row lists of ``[date, time, location, ppm, sampler]``.
        sheet (str): Worksheet name to write.

    Returns:
        blob (bytes): The .xlsx bytes.
    """
    frame = pd.DataFrame(rows, columns=["Date ", "Time", "Location", "PPM", "Sampler"])
    buf = io.BytesIO()
    # startrow=1 leaves the title row the operators keep above the headers,
    # which is why the parser reads with header=1.
    frame.to_excel(buf, sheet_name=sheet, index=False, startrow=1)
    return buf.getvalue()


def _post(
    client: TestClient,
    blob: bytes,
    name: str = "samples.xlsx",
    params: Optional[dict[str, Any]] = None,
):
    return client.post(PATH, files={"file": (name, blob)}, params=params or {})


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def test_daily_rollup_is_the_unweighted_mean_of_the_days_samples(client):
    blob = _workbook(
        [
            ["2026-08-17", "08:00", "P-5417C", 1000, "Jim"],
            ["2026-08-17", "14:00", "P-5417C", 3000, "Jim"],
            ["2026-08-18", "09:00", "P-5417C", 2000, "Jared"],
        ]
    )
    body = _post(client, blob).json()

    assert body["location"] == "P-5417C"
    assert body["sheet"] == "OIW Daily"
    assert body["water_rate_bpd"] == 95000.0
    assert body["sample_count"] == 3
    assert body["first_date"] == "2026-08-17"
    assert body["last_date"] == "2026-08-18"

    day = body["daily"][0]
    assert day["date"] == "2026-08-17"
    assert day["samples"] == 2
    assert day["ppm_mean"] == 2000.0
    assert day["ppm_min"] == 1000.0
    assert day["ppm_max"] == 3000.0
    # (1000 + 3000) / 2 ppm on 95,000 BPD = 190 BOPD, and a daily rate held
    # for one day is that many barrels.
    assert day["bopd_mean"] == 190.0
    assert day["bbl"] == 190.0
    assert day["location"] == "P-5417C"


def test_water_rate_is_the_caller_basis_and_is_echoed_back(client):
    """The sheet's own (BOPD) column assumes 95,000 BWPD and is not read."""
    blob = _workbook([["2026-08-17", "08:00", "P-5417C", 2000, "Jim"]])
    body = _post(client, blob, params={"water_rate_bpd": 71000}).json()

    assert body["water_rate_bpd"] == 71000.0
    assert body["daily"][0]["bopd_mean"] == 142.0
    assert any("71,000 BPD" in note for note in body["notes"])


def test_junk_rows_are_dropped_and_counted_not_raised(client):
    blob = _workbook(
        [
            ["2026-08-17", "08:00", "P-5417C", 1200, "Jim"],
            [None, None, None, None, None],
            ["not a date", "08:00", "P-5417C", 1500, "Jim"],
            ["2107-01-01", "08:00", "P-5417C", 1500, "Jim"],
            ["2026-08-17", "12:00", "P-5417C", "no reading", "Jim"],
            ["2026-08-17", "13:00", "P-5417C", -5, "Jim"],
        ]
    )
    r = _post(client, blob)

    assert r.status_code == 200
    body = r.json()
    assert body["sample_count"] == 1
    assert [d["date"] for d in body["daily"]] == ["2026-08-17"]
    assert any("dropped as unparseable" in note for note in body["notes"])


def test_case_variants_of_a_tap_are_one_location(client):
    """"P-5417C" and "p-5417c" are the same tap typed twice, not two points."""
    blob = _workbook(
        [
            ["2026-08-17", "08:00", "P-5417C", 1000, "Jim"],
            ["2026-08-17", "14:00", "P-5417C", 1000, "Jim"],
            ["2026-08-17", "20:00", "p-5417c", 1000, "Jim"],
            ["2026-08-17", "22:00", "V-5412", 900, "Jim"],
        ]
    )
    body = _post(client, blob).json()

    assert body["locations_available"] == ["P-5417C", "V-5412"]
    assert body["sample_count"] == 3
    assert body["daily"][0]["samples"] == 3


def test_a_location_with_no_samples_is_a_note_not_an_error(client):
    blob = _workbook([["2026-08-17", "08:00", "P-5417C", 1000, "Jim"]])
    r = _post(client, blob, params={"location": "V-9999"})

    assert r.status_code == 200
    body = r.json()
    assert body["daily"] == []
    assert body["sample_count"] == 0
    assert body["locations_available"] == ["P-5417C"]
    assert any("No samples at 'V-9999'" in note for note in body["notes"])


def test_downstream_caveat_rides_on_every_tap_but_v_5317(client):
    rows = [
        ["2026-08-17", "08:00", "P-5417C", 1000, "Jim"],
        ["2022-05-04", "08:00", "V-5317", 200, "Jim"],
    ]
    downstream = _post(client, _workbook(rows)).json()
    upstream = _post(client, _workbook(rows), params={"location": "V-5317"}).json()

    assert any("DOWNSTREAM of the deoilers" in note for note in downstream["notes"])
    # V-5317 samples the stream the calculated band describes, so there is
    # nothing to warn about.
    assert not any("DOWNSTREAM" in note for note in upstream["notes"])


def test_sheet_in_another_layout_is_a_422_not_a_500(client):
    """The Deoiler OIW / LP Desander sheets are a different shape entirely."""
    frame = pd.DataFrame(
        [["2026-08-17", "08:00", 229, 386, 5520]],
        columns=["Date ", "Time", "V-5425 (New)", "V-5422 (Old)", "OIW"],
    )
    buf = io.BytesIO()
    frame.to_excel(buf, sheet_name="OIW Daily", index=False, startrow=1)
    r = _post(client, buf.getvalue())

    assert r.status_code == 422
    assert r.json()["detail"]["error"] == "invalid"
    assert "grab-sample log" in r.json()["detail"]["message"]


def test_empty_sheet_and_all_junk_sheet_are_clean_422s(client):
    empty = _post(client, _workbook([]))
    assert empty.status_code == 422
    assert "no rows" in empty.json()["detail"]["message"]

    junk = _post(client, _workbook([["nope", None, "P-5417C", "nope", None]]))
    assert junk.status_code == 422
    assert "parseable date" in junk.json()["detail"]["message"]


def test_missing_sheet_names_the_ones_the_workbook_has(client):
    blob = _workbook([["2026-08-17", "08:00", "P-5417C", 1000, "Jim"]])
    r = _post(client, blob, params={"sheet": "Nope"})

    assert r.status_code == 422
    assert "OIW Daily" in r.json()["detail"]["message"]


def test_non_xlsx_upload_and_garbage_bytes_are_rejected(client):
    csv = _post(client, b"date,ppm\n2026-08-17,900\n", name="samples.csv")
    assert csv.status_code == 422
    assert csv.json()["detail"]["message"] == "samples.csv: expected an .xlsx workbook"

    garbage = _post(client, b"this is not a workbook", name="samples.xlsx")
    assert garbage.status_code == 422
    assert garbage.json()["detail"]["error"] == "invalid"
    assert "XLSX workbook" in garbage.json()["detail"]["message"]


def test_the_v_5317_sheet_is_read_with_the_same_parser(client):
    """The V-5317 sheet stopped in 2023; it must still parse, not 500."""
    blob = _workbook(
        [["2023-05-29", "08:00", "V-5317", 227, "Jim"]],
        sheet="V-5317",
    )
    body = _post(client, blob, params={"sheet": "V-5317", "location": "V-5317"}).json()

    assert body["sheet"] == "V-5317"
    assert body["sample_count"] == 1
    # 227 ppm on 95,000 BPD = 21.565 BOPD, reported to 2 dp.
    assert body["daily"][0]["bopd_mean"] == pytest.approx(21.565, abs=0.01)
