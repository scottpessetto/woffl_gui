"""Shut-in windows on the production-history chart (server side).

The chart interpolates between sparse well tests, so a well shut in for two
years (MPL-06, casing leak 2024-01-10 .. 2026-06-20) used to render as a
continuous producing ramp. /wells/{name}/jp-history now carries the downtime
log's shut-in windows; the SPA steps the rates to zero across them.

No Databricks: the query layer is monkeypatched throughout.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from server.cache import clear_all_caches
from server.services import history as history_svc


def _days(rows: list[tuple[str, float]], code: str = "SI", reason: str = "Shut In") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dtdate": pd.to_datetime([d for d, _h in rows]).date,
            "hrs": [h for _d, h in rows],
            "down_code": [code] * len(rows),
            "down_reason": [reason] * len(rows),
        }
    )


def _spans(frame: pd.DataFrame) -> list[tuple[str, str, int]]:
    return [
        (r.start.strftime("%Y-%m-%d"), r.end.strftime("%Y-%m-%d"), int(r.days))
        for r in frame.itertuples(index=False)
    ]


@pytest.fixture(autouse=True)
def _clean_caches():
    clear_all_caches()
    yield
    clear_all_caches()


# ---------------------------------------------------------------------------
# _shape_shut_in - day rows -> windows
# ---------------------------------------------------------------------------


def test_consecutive_full_down_days_collapse_into_one_window():
    out = history_svc._shape_shut_in(
        _days([("2024-01-10", 24), ("2024-01-11", 24), ("2024-01-12", 21)], "D62", "Casing Leak")
    )
    assert _spans(out) == [("2024-01-10", "2024-01-12", 3)]
    assert list(out.columns) == ["start", "end", "days", "code", "reason"]
    assert out.loc[0, "code"] == "D62" and out.loc[0, "reason"] == "Casing Leak"


def test_short_partial_runs_bridge_but_do_not_start_or_end_a_window():
    # MPL-06's real start: 10.1 h, 24, 19.7, 14.6, 24 ... -> one window from
    # the first FULL day; the partial lead-in day is not shut in.
    rows = [("2024-01-09", 10.09), ("2024-01-10", 24), ("2024-01-11", 19.65),
            ("2024-01-12", 14.59), ("2024-01-13", 24), ("2024-01-14", 9.1)]
    assert _spans(history_svc._shape_shut_in(_days(rows))) == [("2024-01-10", "2024-01-13", 4)]


def test_more_than_the_bridge_limit_of_partial_days_splits_the_window():
    partial = [(f"2025-03-{d:02d}", 12.0) for d in range(2, 2 + history_svc.SHUT_IN_BRIDGE_DAYS + 1)]
    rows = [("2025-03-01", 24)] + partial + [("2025-03-06", 24)]
    assert _spans(history_svc._shape_shut_in(_days(rows))) == [
        ("2025-03-01", "2025-03-01", 1), ("2025-03-06", "2025-03-06", 1)]


def test_a_day_missing_from_the_log_ends_the_run_and_partial_only_days_are_ignored():
    rows = [("2025-05-01", 24), ("2025-05-03", 24), ("2025-06-01", 15), ("2025-06-02", 12)]
    assert _spans(history_svc._shape_shut_in(_days(rows))) == [
        ("2025-05-01", "2025-05-01", 1), ("2025-05-03", "2025-05-03", 1)]


def test_duplicate_log_rows_and_empty_input_are_harmless():
    rows = [("2025-07-01", 48), ("2025-07-01", 48), ("2025-07-02", 24)]  # SUM over duplicate rows
    assert _spans(history_svc._shape_shut_in(_days(rows))) == [("2025-07-01", "2025-07-02", 2)]
    assert history_svc._shape_shut_in(pd.DataFrame()).empty
    assert list(history_svc._shape_shut_in(pd.DataFrame()).columns) == ["start", "end", "days", "code", "reason"]


def test_the_per_well_query_is_guarded_and_keyed_to_the_producer_enthid(monkeypatch):
    from woffl.assembly import databricks_client
    from woffl.assembly.sql_guards import UnsafeSqlValueError

    seen: list[str] = []
    monkeypatch.setattr(databricks_client, "execute_query", lambda sql: seen.append(sql) or _days([]))
    assert history_svc.shut_in_windows("L-006", "2023-06-12", "2026-09-22").empty
    sql = seen[0]
    assert "mpu.wells.vw_shut_in" in sql and "well_name = 'L-006'" in sql
    assert "s.dthid IN" in sql and "vw_well_test" in sql
    assert "BETWEEN '2023-06-12' AND '2026-09-22'" in sql
    assert ">= 8.0" in sql, "only notable downtime days travel"
    with pytest.raises(UnsafeSqlValueError):
        history_svc.shut_in_windows("L-006' OR 1=1 --", "2023-06-12", "2026-09-22")


# ---------------------------------------------------------------------------
# jp_history_payload + the endpoint
# ---------------------------------------------------------------------------


def _jp_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Well Name": ["MPL-06", "MPL-06"],
            "Date Set": pd.to_datetime(["2023-06-12", "2023-06-24"]),
            "Nozzle Number": [11.0, 11.0],
            "Throat Ratio": ["D", "B"],
        }
    )


@pytest.fixture
def mpl06(monkeypatch):
    state: dict = {"shut_in_boom": False}
    tests = pd.DataFrame({"WtDate": pd.to_datetime(["2023-12-18", "2026-06-24"]),
                          "WtOilVol": [246.7, 1158.3], "WtWaterVol": [443.4, 1075.9]})

    def _shut_in(db_name, start, end):
        if state["shut_in_boom"]:
            raise RuntimeError("vw_shut_in unavailable")
        return history_svc._shape_shut_in(
            _days([("2024-01-10", 24), ("2024-01-11", 24), ("2026-06-20", 24)], "06WeU", "Casing Leak"))

    monkeypatch.setattr(history_svc.datasources, "jp_history_safe", lambda: (_jp_frame(), "databricks"))
    monkeypatch.setattr(history_svc, "extended_tests", lambda *a: tests.copy())
    monkeypatch.setattr(history_svc, "bhp_daily", lambda *a: pd.DataFrame())
    monkeypatch.setattr(history_svc, "shut_in_windows", _shut_in)
    return state


def test_payload_carries_json_safe_shut_in_windows(mpl06):
    payload = history_svc.jp_history_payload("MPL-06")
    assert payload["shut_in"] == [
        {"start": "2024-01-10", "end": "2024-01-11", "days": 2, "code": "06WeU", "reason": "Casing Leak"},
        {"start": "2026-06-20", "end": "2026-06-20", "days": 1, "code": "06WeU", "reason": "Casing Leak"},
    ]
    assert len(payload["tests"]) == 2


def test_a_missing_shut_in_log_fails_soft_to_todays_chart(mpl06):
    mpl06["shut_in_boom"] = True
    payload = history_svc.jp_history_payload("MPL-06")
    assert payload["shut_in"] == []
    assert len(payload["tests"]) == 2, "tests still render"


def test_the_endpoint_response_model_keeps_the_windows(mpl06):
    from server.main import app

    # No context manager: entering it would run the app lifespan (the warm loop).
    body = TestClient(app).get("/api/wells/MPL-06/jp-history").json()
    assert [w["start"] for w in body["shut_in"]] == ["2024-01-10", "2026-06-20"]


def test_a_well_without_installs_returns_an_empty_shut_in_list(mpl06):
    assert history_svc.jp_history_payload("MPX-99")["shut_in"] == []
