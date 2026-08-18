"""The Well Database "Aging jet pumps" list: which wells count as ONLINE.

Reported 2026-08-18: MPS-05, a well with 334 days on its current pump and a
well test two days old, was missing from the list and badged "Online: No".

Cause: the online proxy only counted ALLOCATED tests. Allocation is a monthly
accounting pass, so a producing well routinely has no allocated test for ~30
days - MPS-05 measured against live Databricks that day was tested 2026-08-16
with its last allocated test on 2026-07-17 (32 days, outside the window). At a
30-day window the allocated-only proxy called 37 of 118 producing wells offline.

Databricks is monkeypatched here; the query shape itself is exercised by the
live probe recorded above.
"""

from __future__ import annotations

import pandas as pd
import pytest

from server.services import database as dbsvc

# The reported well: tested two days ago, allocated a month ago.
_TESTS = pd.DataFrame(
    {
        "well_name": ["S-005", "S-054", "F-079"],
        "last_test": pd.to_datetime(
            ["2026-08-16 05:18:32", "2026-08-15 15:15:35", "2026-05-25 04:35:00"],
            utc=True,
        ),
        # F-079 has never been allocated at all.
        "last_allocated": pd.to_datetime(
            ["2026-07-17", "2026-07-14", None], utc=True
        ),
    }
)

_JP_HISTORY = pd.DataFrame(
    {
        "Well Name": ["MPS-05", "MPS-54", "MPF-79"],
        "Date Set": pd.to_datetime(["2025-09-18", "2024-01-05", "2025-01-01"]),
        "Nozzle Number": [9.0, 12.0, 11.0],
        "Throat Ratio": ["C", "B", "A"],
    }
)

_CHARS = pd.DataFrame({"Well": ["MPS-05", "MPS-54", "MPF-79"]})


@pytest.fixture()
def aging(monkeypatch):
    """dbsvc.aging_pumps over a stubbed warehouse, frozen at 2026-08-18."""
    monkeypatch.setattr(
        dbsvc.datasources, "jp_history_safe", lambda: (_JP_HISTORY, "test")
    )
    monkeypatch.setattr(
        dbsvc.datasources, "well_chars_safe", lambda: (_CHARS, "test")
    )
    monkeypatch.setattr(
        dbsvc.pd.Timestamp, "today", staticmethod(lambda: pd.Timestamp("2026-08-18"))
    )

    def _query(_sql):
        return _TESTS

    import woffl.assembly.databricks_client as dbc

    monkeypatch.setattr(dbc, "execute_query", _query)
    dbsvc.latest_test_dates.cache_clear()
    yield lambda **kw: dbsvc.aging_pumps(
        kw.get("known_only", True),
        kw.get("online_only", True),
        kw.get("online_days", 60),
        kw.get("min_days", 0),
    )["rows"]
    dbsvc.latest_test_dates.cache_clear()


def _row(rows, well):
    hits = [r for r in rows if r["well"] == well]
    return hits[0] if hits else None


def test_a_well_tested_recently_is_online_even_with_no_recent_allocation(aging):
    """The reported bug, at the window that hid it."""
    row = _row(aging(online_days=30), "MPS-05")
    assert row is not None, "a well tested 2 days ago must survive the online filter"
    assert row["online"] is True
    assert row["last_test"] == "2026-08-16"
    assert row["last_allocated"] == "2026-07-17"


def test_the_row_carries_both_dates_so_the_lag_is_visible(aging):
    """"Online, just not allocated yet" has to read off the row - that is the
    question the badge alone could not answer."""
    row = _row(aging(online_only=False), "MPS-05")
    assert (row["last_test"], row["last_allocated"]) == ("2026-08-16", "2026-07-17")


def test_a_well_never_allocated_still_counts_as_online(aging):
    """Allocation is not a precondition for producing - F-079 has no allocated
    test on record at all, so an allocated-only proxy dropped it forever."""
    row = _row(aging(online_days=120), "MPF-79")
    assert row is not None and row["online"] is True
    assert row["last_allocated"] is None


def test_a_well_outside_the_window_is_still_offline(aging):
    """The filter must keep doing its job: a well SI'd for years is exactly the
    row this list is meant to suppress."""
    rows = aging(online_days=30)
    assert _row(rows, "MPF-79") is None  # last test 2026-05-25, 85 days back


def test_an_unavailable_test_source_disables_the_filter(monkeypatch):
    """Fail-soft: no test dates must not mean an empty aging list."""
    monkeypatch.setattr(
        dbsvc.datasources, "jp_history_safe", lambda: (_JP_HISTORY, "test")
    )
    monkeypatch.setattr(
        dbsvc.datasources, "well_chars_safe", lambda: (_CHARS, "test")
    )
    monkeypatch.setattr(dbsvc, "latest_test_dates", lambda: ({}, {}))

    rows = dbsvc.aging_pumps(True, True, 30, 0)["rows"]

    assert len(rows) == 3
    assert all(r["last_test"] is None for r in rows)


def test_the_query_asks_for_every_test_not_just_allocated_ones():
    """Guard against the fix being undone in SQL: an `allocated = True`
    predicate in the WHERE clause is what caused the report."""
    sql = dbsvc._LATEST_TEST_QUERY.lower()
    assert "where" not in sql, "no row filter - allocation is a column, not a gate"
    assert "max(wt_date) as last_test" in sql
    assert "case when allocated then wt_date end" in sql
