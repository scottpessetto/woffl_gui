"""Tests for Databricks client with mocked database calls."""

import threading
import time
from unittest.mock import patch

import pandas as pd
import pytest

import woffl.assembly.databricks_client as databricks_client
from woffl.assembly.databricks_client import (
    _CONN_LOCAL,
    _TOKEN_CACHE,
    _oauth_token,
    _query_via_connector,
    fetch_jp_history,
    get_tags_for_wells,
)

# ── _query_via_connector: first-attempt connect failure still retries ──────
# Regression for P2-6: `_new_connection()` used to be called OUTSIDE the
# try/except in `_query_via_connector`, so a failure on the very first
# connection attempt raised immediately and skipped the retry entirely.


class _FakeCursor:
    def __init__(self, rows, columns):
        self._rows = rows
        self.description = [(c,) for c in columns]

    def execute(self, query):
        pass

    def fetchall(self):
        return self._rows

    def close(self):
        pass


class _FakeConnection:
    def __init__(self, rows=None, columns=None):
        self._rows = rows if rows is not None else [(1,)]
        self._columns = columns if columns is not None else ["value"]
        self.closed = False

    def cursor(self):
        return _FakeCursor(self._rows, self._columns)

    def close(self):
        self.closed = True


class TestQueryViaConnectorRetriesFirstAttempt:
    def setup_method(self):
        # Per-thread connection cache -- start every test with no cached conn.
        _CONN_LOCAL.conn = None

    def teardown_method(self):
        _CONN_LOCAL.conn = None

    @patch("woffl.assembly.databricks_client._new_connection")
    def test_first_connect_failure_retries_and_succeeds(self, mock_new_conn):
        good_conn = _FakeConnection(rows=[(42,)], columns=["answer"])
        # Attempt 1: connecting itself raises. Attempt 2: succeeds.
        mock_new_conn.side_effect = [RuntimeError("connect failed"), good_conn]

        result = _query_via_connector("SELECT 42")

        assert mock_new_conn.call_count == 2
        assert isinstance(result, pd.DataFrame)
        assert result["answer"].iloc[0] == 42

    @patch("woffl.assembly.databricks_client._new_connection")
    def test_first_connect_failure_forces_token_refresh_and_clears_cache(
        self, mock_new_conn
    ):
        _TOKEN_CACHE["token"] = "stale-token"
        _TOKEN_CACHE["expires_at"] = time.time() + 3600

        mock_new_conn.side_effect = [RuntimeError("connect failed"), _FakeConnection()]

        _query_via_connector("SELECT 1")

        # Existing retry behavior (must survive the fix): a failed attempt
        # forces a token refresh on the next attempt.
        assert _TOKEN_CACHE["token"] is None

    @patch("woffl.assembly.databricks_client._new_connection")
    def test_two_connect_failures_raise_the_last_error(self, mock_new_conn):
        mock_new_conn.side_effect = [
            RuntimeError("first failure"),
            RuntimeError("second failure"),
        ]

        with pytest.raises(RuntimeError, match="second failure"):
            _query_via_connector("SELECT 1")

        assert mock_new_conn.call_count == 2


# ── _oauth_token: HTTP fetch must not hold _TOKEN_LOCK ─────────────────────
# Regression for P2-6: _oauth_token used to hold _TOKEN_LOCK across the ~30 s
# HTTP token request, serializing every thread behind one network call.


class TestOauthTokenReleasesLockDuringFetch:
    def setup_method(self):
        _TOKEN_CACHE["token"] = None
        _TOKEN_CACHE["expires_at"] = 0.0

    def teardown_method(self):
        _TOKEN_CACHE["token"] = None
        _TOKEN_CACHE["expires_at"] = 0.0

    def test_lock_is_not_held_during_the_http_call(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_HOST", "example.databricks.com")
        monkeypatch.setenv("DATABRICKS_CLIENT_ID", "client-id")
        monkeypatch.setenv("DATABRICKS_CLIENT_SECRET", "client-secret")

        fetch_started = threading.Event()
        release_fetch = threading.Event()

        class _FakeResponse:
            def __enter__(self):
                fetch_started.set()
                # Stand-in for the ~30 s HTTP round trip -- deterministic
                # (event-gated), not a timed sleep.
                assert release_fetch.wait(
                    timeout=5
                ), "test never released the fake fetch"
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return b'{"access_token": "fresh-token", "expires_in": 3600}'

        def fake_urlopen(req, timeout=30):
            return _FakeResponse()

        monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

        result_holder = {}

        def call_oauth_token():
            result_holder["token"] = _oauth_token()

        worker = threading.Thread(target=call_oauth_token)
        worker.start()

        assert fetch_started.wait(timeout=5), "fetch never started"

        # While the fake HTTP call is still "in flight", the lock must be
        # free. If _oauth_token held _TOKEN_LOCK across the fetch, this
        # acquire would fail/time out.
        acquired = databricks_client._TOKEN_LOCK.acquire(timeout=2)
        try:
            assert acquired, "_TOKEN_LOCK was held during the HTTP fetch"
        finally:
            if acquired:
                databricks_client._TOKEN_LOCK.release()

        release_fetch.set()
        worker.join(timeout=5)
        assert not worker.is_alive()
        assert result_holder["token"] == "fresh-token"

    def test_returns_fresh_token_and_populates_cache(self, monkeypatch):
        monkeypatch.setenv("DATABRICKS_HOST", "example.databricks.com")
        monkeypatch.setenv("DATABRICKS_CLIENT_ID", "client-id")
        monkeypatch.setenv("DATABRICKS_CLIENT_SECRET", "client-secret")

        class _FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return b'{"access_token": "brand-new-token", "expires_in": 3600}'

        monkeypatch.setattr(
            "urllib.request.urlopen", lambda req, timeout=30: _FakeResponse()
        )

        token = _oauth_token()

        assert token == "brand-new-token"
        assert _TOKEN_CACHE["token"] == "brand-new-token"
        assert _TOKEN_CACHE["expires_at"] > time.time()


# ── get_tags_for_wells ──────────────────────────────────────────────────────


class TestGetTagsForWells:
    @staticmethod
    def _tag_dict():
        return {
            "MPB-28": ("bhp_b28", "hp_b28", "whp_b28"),
            "MPE-41": ("bhp_e41", "hp_e41", "whp_e41"),
        }

    def test_all_found(self):
        found, missing = get_tags_for_wells(["MPB-28", "MPE-41"], self._tag_dict())
        assert len(found) == 2
        assert missing == []

    def test_some_missing(self):
        found, missing = get_tags_for_wells(["MPB-28", "FAKE-99"], self._tag_dict())
        assert "MPB-28" in found
        assert "FAKE-99" in missing

    def test_all_missing(self):
        found, missing = get_tags_for_wells(["FAKE-1", "FAKE-2"], self._tag_dict())
        assert found == {}
        assert len(missing) == 2

    def test_empty_wells(self):
        found, missing = get_tags_for_wells([], self._tag_dict())
        assert found == {}
        assert missing == []

    def test_tag_tuple_structure(self):
        found, _ = get_tags_for_wells(["MPB-28"], self._tag_dict())
        bhp, hp, whp = found["MPB-28"]
        assert bhp == "bhp_b28"
        assert hp == "hp_b28"
        assert whp == "whp_b28"


# ── fetch_jp_history (mocked) ──────────────────────────────────────────────


class TestFetchJPHistory:
    @patch("woffl.assembly.databricks_client.execute_query")
    def test_date_columns_are_datetime(self, mock_query):
        mock_query.return_value = pd.DataFrame(
            {
                "Well Name": ["MPB-28 ", "MPE-41"],
                "Date Set": ["2024-01-15", "2024-02-20"],
                "Date Pulled": ["2024-06-01", None],
                "Nozzle Number": [12, 13],
                "Throat Ratio": ["A", "B"],
                "Tubing Diameter": [4.5, 4.5],
            }
        )
        result = fetch_jp_history()
        assert pd.api.types.is_datetime64_any_dtype(result["Date Set"])

    @patch("woffl.assembly.databricks_client.execute_query")
    def test_well_name_stripped(self, mock_query):
        mock_query.return_value = pd.DataFrame(
            {
                "Well Name": ["  MPB-28  ", "MPE-41 "],
                "Date Set": ["2024-01-15", "2024-02-20"],
                "Nozzle Number": [12, 13],
                "Throat Ratio": ["A", "B"],
            }
        )
        result = fetch_jp_history()
        assert result["Well Name"].iloc[0] == "MPB-28"
        assert result["Well Name"].iloc[1] == "MPE-41"

    @patch("woffl.assembly.databricks_client.execute_query")
    def test_returns_dataframe(self, mock_query):
        mock_query.return_value = pd.DataFrame(
            {
                "Well Name": ["MPB-28"],
                "Date Set": ["2024-01-15"],
                "Nozzle Number": [12],
                "Throat Ratio": ["A"],
            }
        )
        result = fetch_jp_history()
        assert isinstance(result, pd.DataFrame)
