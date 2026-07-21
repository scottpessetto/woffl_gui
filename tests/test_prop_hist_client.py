"""Tests for prop_hist_client -- fully mocked, zero live Databricks calls.

Covers: prop_xref whitelist rejection, enthid 0-match/multi-match guards,
the write-gate-off short-circuit, INSERT parameter shapes, fetch_latest_prop
None/newest behavior, and resolve_entry_user's env-override precedence.
"""

import os
from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd
import pytest

import woffl.assembly.prop_hist_client as phc
from woffl.assembly.databricks_client import WritesDisabledError
from woffl.assembly.prop_hist_client import (
    EnthidResolutionError,
    UnknownPropIdError,
    fetch_latest_prop,
    fetch_prop_xref,
    push_prop,
    resolve_entry_user,
    well_enthid_map,
)


def _reset_caches():
    phc._xref_cache["value"] = None
    phc._xref_cache["expires_at"] = 0.0
    phc._enthid_cache["value"] = None
    phc._enthid_cache["expires_at"] = 0.0
    phc._entry_user_cache["value"] = None


def _query_router(xref=None, enthid=None, current_user=None, prop_hist=None):
    """Return an execute_query stand-in that answers based on which table the
    query text touches -- lets a single test drive multiple distinct reads
    (xref whitelist, enthid map, prop_hist itself) through one mock."""

    def _execute_query(query: str):
        q = query.lower()
        if "prop_xref" in q:
            return xref if xref is not None else pd.DataFrame({"prop_id": []})
        if "vw_well_header" in q:
            return (
                enthid
                if enthid is not None
                else pd.DataFrame({"enthid": [], "well_name": []})
            )
        if "current_user" in q:
            return (
                current_user
                if current_user is not None
                else pd.DataFrame({"current_user": []})
            )
        if "prop_hist" in q:
            return prop_hist if prop_hist is not None else pd.DataFrame()
        raise AssertionError(f"test router got an unexpected query: {query!r}")

    return _execute_query


class _CacheResetMixin:
    def setup_method(self):
        _reset_caches()

    def teardown_method(self):
        _reset_caches()
        os.environ.pop("WOFFL_ENTRY_USER", None)
        os.environ.pop("ALLOW_DATABRICKS_WRITES", None)


# ── fetch_prop_xref ──────────────────────────────────────────────────────────


class TestFetchPropXref(_CacheResetMixin):
    @patch("woffl.assembly.prop_hist_client.execute_query")
    def test_returns_set_of_prop_ids(self, mock_query):
        mock_query.return_value = pd.DataFrame(
            {"prop_id": ["ipr_wt_uid", "jpfric_entry", "jpfric_throat"]}
        )
        result = fetch_prop_xref()
        assert result == {"ipr_wt_uid", "jpfric_entry", "jpfric_throat"}

    @patch("woffl.assembly.prop_hist_client.execute_query")
    def test_caches_across_calls(self, mock_query):
        mock_query.return_value = pd.DataFrame({"prop_id": ["ipr_wt_uid"]})
        fetch_prop_xref()
        fetch_prop_xref()
        assert mock_query.call_count == 1

    @patch("woffl.assembly.prop_hist_client.execute_query")
    def test_force_refresh_bypasses_cache(self, mock_query):
        mock_query.return_value = pd.DataFrame({"prop_id": ["ipr_wt_uid"]})
        fetch_prop_xref()
        fetch_prop_xref(force_refresh=True)
        assert mock_query.call_count == 2

    @patch("woffl.assembly.prop_hist_client.execute_query")
    def test_empty_table_returns_empty_set(self, mock_query):
        mock_query.return_value = pd.DataFrame({"prop_id": []})
        assert fetch_prop_xref() == set()


# ── well_enthid_map ──────────────────────────────────────────────────────────


class TestWellEnthidMap(_CacheResetMixin):
    @patch("woffl.assembly.prop_hist_client.execute_query")
    def test_normalizes_db_names_to_gui_format(self, mock_query):
        mock_query.return_value = pd.DataFrame(
            {"enthid": [111, 222], "well_name": ["B-028", "E-041"]}
        )
        result = well_enthid_map()
        assert result == {"MPB-28": 111, "MPE-41": 222}

    @patch("woffl.assembly.prop_hist_client.execute_query")
    def test_caches_across_calls(self, mock_query):
        mock_query.return_value = pd.DataFrame(
            {"enthid": [111], "well_name": ["B-028"]}
        )
        well_enthid_map()
        well_enthid_map()
        assert mock_query.call_count == 1

    @patch("woffl.assembly.prop_hist_client.execute_query")
    def test_ambiguous_name_is_omitted_from_the_single_valued_map(self, mock_query):
        # Two rows sharing a well_name -- a data-quality issue the map
        # doesn't silently resolve by picking one; push_prop's guard (below)
        # is where this actually raises.
        mock_query.return_value = pd.DataFrame(
            {"enthid": [111, 999], "well_name": ["B-028", "B-028"]}
        )
        result = well_enthid_map()
        assert "MPB-28" not in result

    @patch("woffl.assembly.prop_hist_client.execute_query")
    def test_drops_null_well_names(self, mock_query):
        mock_query.return_value = pd.DataFrame(
            {"enthid": [111, 222], "well_name": ["B-028", None]}
        )
        result = well_enthid_map()
        assert result == {"MPB-28": 111}


# ── push_prop: whitelist guard ───────────────────────────────────────────────


class TestPushPropWhitelist(_CacheResetMixin):
    @patch("woffl.assembly.prop_hist_client.execute_write")
    @patch("woffl.assembly.prop_hist_client.execute_query")
    def test_rejects_prop_id_not_in_xref(self, mock_query, mock_write):
        mock_query.side_effect = _query_router(
            xref=pd.DataFrame({"prop_id": ["ipr_wt_uid", "jpfric_entry"]})
        )

        with pytest.raises(UnknownPropIdError) as exc_info:
            push_prop("MPB-28", "not_a_real_prop", 5.0, "scott")

        message = str(exc_info.value)
        assert "not_a_real_prop" in message
        # Whitelist rejection message lists the valid keys.
        assert "ipr_wt_uid" in message
        assert "jpfric_entry" in message
        mock_write.assert_not_called()


# ── push_prop: enthid resolution guards ──────────────────────────────────────


class TestPushPropEnthidGuards(_CacheResetMixin):
    @patch("woffl.assembly.prop_hist_client.execute_write")
    @patch("woffl.assembly.prop_hist_client.execute_query")
    def test_zero_match_raises(self, mock_query, mock_write):
        mock_query.side_effect = _query_router(
            xref=pd.DataFrame({"prop_id": ["ipr_wt_uid"]}),
            enthid=pd.DataFrame({"enthid": [], "well_name": []}),
        )

        with pytest.raises(EnthidResolutionError, match="No enthid found"):
            push_prop("MPB-28", "ipr_wt_uid", 5.0, "scott")

        mock_write.assert_not_called()

    @patch("woffl.assembly.prop_hist_client.execute_write")
    @patch("woffl.assembly.prop_hist_client.execute_query")
    def test_multi_match_raises(self, mock_query, mock_write):
        mock_query.side_effect = _query_router(
            xref=pd.DataFrame({"prop_id": ["ipr_wt_uid"]}),
            enthid=pd.DataFrame(
                {"enthid": [111, 999], "well_name": ["B-028", "B-028"]}
            ),
        )

        with pytest.raises(EnthidResolutionError, match="Multiple enthids"):
            push_prop("MPB-28", "ipr_wt_uid", 5.0, "scott")

        mock_write.assert_not_called()


# ── push_prop: write gate ────────────────────────────────────────────────────


class TestPushPropWriteGate(_CacheResetMixin):
    @patch("woffl.assembly.databricks_client._new_connection")
    @patch("woffl.assembly.prop_hist_client.execute_query")
    def test_gate_off_raises_before_any_connection_attempt(
        self, mock_query, mock_new_conn
    ):
        os.environ.pop("ALLOW_DATABRICKS_WRITES", None)
        mock_query.side_effect = _query_router(
            xref=pd.DataFrame({"prop_id": ["ipr_wt_uid"]}),
            enthid=pd.DataFrame({"enthid": [111], "well_name": ["B-028"]}),
        )

        # Real execute_write (not mocked) -- proves the gate closes the door
        # before prop_hist_client's own logic ever reaches a connection.
        with pytest.raises(WritesDisabledError):
            push_prop("MPB-28", "ipr_wt_uid", 5.0, "scott")

        mock_new_conn.assert_not_called()


# ── push_prop: INSERT parameter shapes ───────────────────────────────────────


class TestPushPropInsertParameters(_CacheResetMixin):
    @patch("woffl.assembly.prop_hist_client.execute_write")
    @patch("woffl.assembly.prop_hist_client.execute_query")
    def test_insert_called_with_exact_parameters(self, mock_query, mock_write):
        mock_query.side_effect = _query_router(
            xref=pd.DataFrame({"prop_id": ["ipr_wt_uid"]}),
            enthid=pd.DataFrame({"enthid": [12345], "well_name": ["B-028"]}),
        )
        mock_write.return_value = 1

        before = datetime.now(timezone.utc)
        result = push_prop("MPB-28", "ipr_wt_uid", 987654, "scott")
        after = datetime.now(timezone.utc)

        assert result == 1
        assert mock_write.call_count == 1
        sql_arg, params_arg = mock_write.call_args[0]
        assert sql_arg.strip().upper().startswith("INSERT")
        assert "mpu.wells.prop_hist" in sql_arg
        assert "entry_datetime" in sql_arg

        # entry_datetime is bound as a real timezone-aware UTC datetime (not
        # a date string) -- assert type/awareness/recency rather than an
        # exact value, since "now" isn't reproducible.
        entry_dt = params_arg.pop("entry_datetime")
        assert isinstance(entry_dt, datetime)
        assert entry_dt.tzinfo is not None
        assert before <= entry_dt <= after

        assert params_arg == {
            "enthid": 12345,
            "prop_id": "ipr_wt_uid",
            "prop_value": 987654.0,
            "entry_user": "scott",
        }

    @patch("woffl.assembly.prop_hist_client.execute_write")
    @patch("woffl.assembly.prop_hist_client.execute_query")
    def test_prop_value_is_coerced_to_float(self, mock_query, mock_write):
        mock_query.side_effect = _query_router(
            xref=pd.DataFrame({"prop_id": ["ipr_wt_uid"]}),
            enthid=pd.DataFrame({"enthid": [1], "well_name": ["B-001"]}),
        )
        mock_write.return_value = 1

        push_prop("MPB-01", "ipr_wt_uid", "42", "scott")

        _, params_arg = mock_write.call_args[0]
        assert params_arg["prop_value"] == 42.0
        assert isinstance(params_arg["prop_value"], float)

    @patch("woffl.assembly.prop_hist_client.execute_write")
    @patch("woffl.assembly.prop_hist_client.execute_query")
    def test_negative_wt_uid_is_pushed_verbatim(self, mock_query, mock_write):
        # Real wt_uid values in vw_well_test are signed and span roughly
        # -3.6M to +3.1M -- almost all negative in practice (e.g. C-045's
        # real saved pin, prop_value=-3576674). push_prop must not special-
        # case sign in any way.
        mock_query.side_effect = _query_router(
            xref=pd.DataFrame({"prop_id": ["ipr_wt_uid"]}),
            enthid=pd.DataFrame({"enthid": [12345], "well_name": ["C-045"]}),
        )
        mock_write.return_value = 1

        push_prop("MPC-45", "ipr_wt_uid", -3576674, "scott")

        _, params_arg = mock_write.call_args[0]
        assert params_arg["prop_value"] == -3576674.0

    @patch("woffl.assembly.prop_hist_client.execute_write")
    @patch("woffl.assembly.prop_hist_client.execute_query")
    def test_none_value_pushes_sql_null(self, mock_query, mock_write):
        # None is the un-pin marker (W3): it must bind as a SQL NULL, not be
        # coerced through float() -- a negative-number sentinel isn't safe
        # since real wt_uid values are signed and can be negative themselves.
        mock_query.side_effect = _query_router(
            xref=pd.DataFrame({"prop_id": ["ipr_wt_uid"]}),
            enthid=pd.DataFrame({"enthid": [12345], "well_name": ["B-028"]}),
        )
        mock_write.return_value = 1

        result = push_prop("MPB-28", "ipr_wt_uid", None, "scott")

        assert result == 1
        _, params_arg = mock_write.call_args[0]
        assert params_arg["prop_value"] is None

        entry_dt = params_arg.pop("entry_datetime")
        assert isinstance(entry_dt, datetime)
        assert entry_dt.tzinfo is not None

        assert params_arg == {
            "enthid": 12345,
            "prop_id": "ipr_wt_uid",
            "prop_value": None,
            "entry_user": "scott",
        }

    @patch("woffl.assembly.prop_hist_client.execute_write")
    @patch("woffl.assembly.prop_hist_client.execute_query")
    def test_non_finite_value_raises(self, mock_query, mock_write):
        mock_query.side_effect = _query_router(
            xref=pd.DataFrame({"prop_id": ["ipr_wt_uid"]}),
            enthid=pd.DataFrame({"enthid": [1], "well_name": ["B-001"]}),
        )

        with pytest.raises(phc.PropHistError):
            push_prop("MPB-01", "ipr_wt_uid", float("nan"), "scott")

        mock_write.assert_not_called()


# ── fetch_latest_prop ─────────────────────────────────────────────────────────


class TestFetchLatestProp(_CacheResetMixin):
    @patch("woffl.assembly.prop_hist_client.execute_query")
    def test_returns_none_on_empty(self, mock_query):
        mock_query.side_effect = _query_router(
            enthid=pd.DataFrame({"enthid": [111], "well_name": ["B-028"]}),
            prop_hist=pd.DataFrame(
                {"prop_value": [], "entry_datetime": [], "entry_user": []}
            ),
        )
        assert fetch_latest_prop("MPB-28", "ipr_wt_uid") is None

    @patch("woffl.assembly.prop_hist_client.execute_query")
    def test_returns_newest_row_when_multiple_present(self, mock_query):
        # Deliberately unsorted / not what the SQL's ORDER BY would produce --
        # the function must re-sort defensively rather than trust row order.
        mock_query.side_effect = _query_router(
            enthid=pd.DataFrame({"enthid": [111], "well_name": ["B-028"]}),
            prop_hist=pd.DataFrame(
                {
                    "prop_value": [100.0, 200.0, 150.0],
                    "entry_datetime": pd.to_datetime(
                        [
                            "2026-01-01T08:00:00Z",
                            "2026-07-01T14:30:00Z",
                            "2026-04-01T00:00:00Z",
                        ]
                    ),
                    "entry_user": ["alice", "scott", "bob"],
                }
            ),
        )

        result = fetch_latest_prop("MPB-28", "ipr_wt_uid")

        assert result == (200.0, pd.Timestamp("2026-07-01T14:30:00Z"), "scott")

    @patch("woffl.assembly.prop_hist_client.execute_query")
    def test_same_day_rows_return_the_later_timestamp(self, mock_query):
        # entry_datetime is a full timestamp (the column was migrated off the
        # old date-only entry_date), so two same-day pushes resolve
        # deterministically to the genuinely later one -- the capability the
        # rename buys (previously same-day rows were unordered).
        mock_query.side_effect = _query_router(
            enthid=pd.DataFrame({"enthid": [111], "well_name": ["B-028"]}),
            prop_hist=pd.DataFrame(
                {
                    "prop_value": [100.0, 200.0],
                    "entry_datetime": pd.to_datetime(
                        ["2026-07-08T09:00:00Z", "2026-07-08T15:45:00Z"]
                    ),
                    "entry_user": ["alice", "scott"],
                }
            ),
        )

        result = fetch_latest_prop("MPB-28", "ipr_wt_uid")

        assert result == (200.0, pd.Timestamp("2026-07-08T15:45:00Z"), "scott")

    @patch("woffl.assembly.prop_hist_client.execute_query")
    def test_rejects_unsafe_prop_id_shape_without_querying(self, mock_query):
        mock_query.side_effect = AssertionError(
            "execute_query should not be called for an unsafe prop_id"
        )

        with pytest.raises(UnknownPropIdError):
            fetch_latest_prop("MPB-28", "bad; DROP TABLE prop_hist")

    @patch("woffl.assembly.prop_hist_client.execute_query")
    def test_null_prop_value_returns_none_not_error(self, mock_query):
        # An un-pinned well's latest row has prop_value = SQL NULL, which
        # pandas may surface as None (object dtype) or NaN (float64 dtype)
        # depending on the connector. Either way this must return a `None`
        # value in the tuple, never raise, and never be confused with a
        # real (possibly negative) wt_uid.
        mock_query.side_effect = _query_router(
            enthid=pd.DataFrame({"enthid": [111], "well_name": ["B-028"]}),
            prop_hist=pd.DataFrame(
                {
                    "prop_value": [None],
                    "entry_datetime": ["2026-07-06"],
                    "entry_user": ["scott"],
                }
            ),
        )

        result = fetch_latest_prop("MPB-28", "ipr_wt_uid")

        assert result is not None
        value, entry_datetime, entry_user = result
        assert value is None
        assert entry_datetime == "2026-07-06"
        assert entry_user == "scott"

    @patch("woffl.assembly.prop_hist_client.execute_query")
    def test_nan_prop_value_returns_none(self, mock_query):
        mock_query.side_effect = _query_router(
            enthid=pd.DataFrame({"enthid": [111], "well_name": ["B-028"]}),
            prop_hist=pd.DataFrame(
                {
                    "prop_value": [float("nan")],
                    "entry_datetime": ["2026-07-06"],
                    "entry_user": ["scott"],
                }
            ),
        )

        value, _, _ = fetch_latest_prop("MPB-28", "ipr_wt_uid")
        assert value is None

    @patch("woffl.assembly.prop_hist_client.execute_query")
    def test_negative_prop_value_is_returned_as_a_valid_pin(self, mock_query):
        # Real wt_uid values are signed and span roughly -3.6M to +3.1M --
        # almost all negative in practice. A negative prop_value is a REAL
        # value, not "no pin" -- confirms fetch_latest_prop applies no
        # sign-based rule.
        mock_query.side_effect = _query_router(
            enthid=pd.DataFrame({"enthid": [111], "well_name": ["C-045"]}),
            prop_hist=pd.DataFrame(
                {
                    "prop_value": [-3576674.0],
                    "entry_datetime": ["2026-07-06"],
                    "entry_user": ["scott"],
                }
            ),
        )

        value, _, _ = fetch_latest_prop("MPC-45", "ipr_wt_uid")
        assert value == -3576674.0


# ── resolve_entry_user ────────────────────────────────────────────────────────


class TestResolveEntryUser(_CacheResetMixin):
    @patch("woffl.assembly.prop_hist_client.execute_query")
    def test_env_override_wins_without_querying(self, mock_query, monkeypatch):
        monkeypatch.setenv("WOFFL_ENTRY_USER", "scott.pessetto")
        mock_query.side_effect = AssertionError(
            "execute_query should not be called when the env override is set"
        )

        assert resolve_entry_user() == "scott.pessetto"

    @patch("woffl.assembly.prop_hist_client.execute_query")
    def test_falls_back_to_current_user_and_caches(self, mock_query, monkeypatch):
        monkeypatch.delenv("WOFFL_ENTRY_USER", raising=False)
        mock_query.return_value = pd.DataFrame({"current_user": ["svc_principal"]})

        first = resolve_entry_user()
        second = resolve_entry_user()

        assert first == "svc_principal"
        assert second == "svc_principal"
        assert mock_query.call_count == 1

    @patch("woffl.assembly.prop_hist_client.execute_query")
    def test_env_override_takes_precedence_even_after_caching_current_user(
        self, mock_query, monkeypatch
    ):
        monkeypatch.delenv("WOFFL_ENTRY_USER", raising=False)
        mock_query.return_value = pd.DataFrame({"current_user": ["svc_principal"]})
        resolve_entry_user()  # populates the current_user cache

        monkeypatch.setenv("WOFFL_ENTRY_USER", "override_user")
        assert resolve_entry_user() == "override_user"
