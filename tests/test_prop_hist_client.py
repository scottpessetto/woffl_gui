"""Tests for prop_hist_client -- fully mocked, zero live Databricks calls.

Covers: prop_xref whitelist rejection, enthid 0-match/multi-match guards,
the write-gate-off short-circuit, INSERT parameter shapes, fetch_latest_prop
None/newest behavior, and resolve_entry_user's env-override precedence.
"""

import os
import threading
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pandas as pd
import pytest

import woffl.assembly.prop_hist_client as phc
from woffl.assembly.databricks_client import WritesDisabledError
from woffl.assembly.prop_hist_client import (
    AS_BUILT_PROP_IDS,
    AsBuiltPropError,
    EnthidResolutionError,
    UnknownPropIdError,
    fetch_latest_prop,
    fetch_prop_xref,
    next_entry_datetime,
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
    # Monotonic stamp allocator: a stamp left over from a prior test would make
    # the next one bump off it instead of reading the clock.
    phc._last_stamp = None


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


# ── push_prop: as-built guard ────────────────────────────────────────────────


class TestPushPropAsBuiltGuard(_CacheResetMixin):
    """2026-08-03 incident: the pad review write-through pushed jpump_md and
    casing_out_dia, replacing eight wells' MEASURED pump depth with the
    interpolated JP_TVD (C-002: 7688 → 6270.223 ft) and their casing OD with
    the 6.875 UI fallback. As-built dimensions are read-only from woffl; this
    is the chokepoint that makes that class of bug impossible."""

    @pytest.mark.parametrize("prop_id", sorted(AS_BUILT_PROP_IDS))
    @patch("woffl.assembly.prop_hist_client.execute_write")
    @patch("woffl.assembly.prop_hist_client.execute_query")
    def test_every_as_built_id_is_rejected(self, mock_query, mock_write, prop_id):
        mock_query.side_effect = _query_router(
            xref=pd.DataFrame({"prop_id": sorted(AS_BUILT_PROP_IDS)}),
            enthid=pd.DataFrame({"enthid": [111], "well_name": ["B-028"]}),
        )

        # Rejected even though the id IS a valid prop_xref key with a
        # resolvable well — the guard is about authorship, not validity.
        with pytest.raises(AsBuiltPropError, match=prop_id):
            push_prop("MPB-28", prop_id, 6270.2230992, "scott")

        mock_write.assert_not_called()

    def test_jpump_md_and_casing_out_dia_are_covered(self):
        """The two ids the incident actually corrupted."""
        assert {"jpump_md", "casing_out_dia"} <= AS_BUILT_PROP_IDS


# ── entry_datetime: strictly monotonic stamps ────────────────────────────────


class TestEntryDatetimeAllocation(_CacheResetMixin):
    """``entry_datetime`` decides BOTH which row a read resolves to
    (``ROW_NUMBER() ... ORDER BY entry_datetime DESC``) and which rows belong to
    one save (the ``woffl_eng_comment`` join key). The Windows system clock has
    15.625 ms granularity — 2000 back-to-back ``datetime.now(timezone.utc)``
    calls return ONE value — so two saves in one tick used to collide: merged
    comments, and an arbitrary winner on read.
    """

    def test_back_to_back_stamps_strictly_increase(self):
        stamps = [next_entry_datetime() for _ in range(2000)]
        assert len(set(stamps)) == 2000, "collision inside one clock tick"
        assert stamps == sorted(stamps)

    def test_a_bare_now_would_have_collided(self):
        """Pins the premise, so this suite still means something on a platform
        with a finer clock (there the allocator is simply a pass-through)."""
        now_stamps = {datetime.now(timezone.utc) for _ in range(2000)}
        alloc_stamps = {next_entry_datetime() for _ in range(2000)}
        assert len(alloc_stamps) == 2000
        if len(now_stamps) < 2000:
            # Coarse clock (Windows): the allocator is doing real work here.
            assert len(alloc_stamps) > len(now_stamps)

    def test_stamps_stay_utc_aware_and_track_the_clock(self):
        before = datetime.now(timezone.utc)
        stamp = next_entry_datetime()
        after = datetime.now(timezone.utc)
        assert stamp.tzinfo is timezone.utc
        # Bumping may push a stamp past `after` by microseconds when the clock
        # is coarse; it must never drift backwards or run away.
        assert before - timedelta(seconds=1) <= stamp <= after + timedelta(seconds=1)

    def test_concurrent_callers_never_share_a_stamp(self):
        """Streamlit runs script runs on threads and app.py warms caches on
        another, so the allocator has to hold under contention."""
        out: list = []
        lock = threading.Lock()

        def _worker():
            mine = [next_entry_datetime() for _ in range(200)]
            with lock:
                out.extend(mine)

        threads = [threading.Thread(target=_worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(out) == 1600
        assert len(set(out)) == 1600

    @patch("woffl.assembly.prop_hist_client.execute_write")
    @patch("woffl.assembly.prop_hist_client.execute_query")
    def test_two_quick_pushes_land_on_distinct_stamps(self, mock_query, mock_write):
        mock_query.side_effect = _query_router(
            xref=pd.DataFrame({"prop_id": ["ipr_wt_uid"]}),
            enthid=pd.DataFrame({"enthid": [111], "well_name": ["B-028"]}),
        )

        push_prop("MPB-28", "ipr_wt_uid", 1.0, "scott")
        push_prop("MPB-28", "ipr_wt_uid", 2.0, "scott")

        first, second = (c[0][1]["entry_datetime"] for c in mock_write.call_args_list)
        # Without this the second value could lose the DESC tie-break and the
        # well would reopen on the FIRST push's value.
        assert second > first

    @patch("woffl.assembly.prop_hist_client.execute_write")
    @patch("woffl.assembly.prop_hist_client.execute_query")
    def test_an_explicit_batch_stamp_is_used_verbatim(self, mock_query, mock_write):
        """A save's rows MUST share their stamp — re-allocating per row would
        destroy the batch identity the comment hangs off."""
        mock_query.side_effect = _query_router(
            xref=pd.DataFrame({"prop_id": ["ipr_wt_uid", "ipr_pwf"]}),
            enthid=pd.DataFrame({"enthid": [111], "well_name": ["B-028"]}),
        )
        batch = next_entry_datetime()

        push_prop("MPB-28", "ipr_wt_uid", 1.0, "scott", entry_datetime=batch)
        push_prop("MPB-28", "ipr_pwf", 900.0, "scott", entry_datetime=batch)

        stamps = [c[0][1]["entry_datetime"] for c in mock_write.call_args_list]
        assert stamps == [batch, batch]

    @patch("woffl.assembly.prop_hist_client.execute_write")
    @patch("woffl.assembly.prop_hist_client.execute_query")
    def test_a_later_lone_push_cannot_reuse_a_batch_stamp(self, mock_query, mock_write):
        """The allocator records handed-out batch stamps, so a subsequent
        default-stamped push can't land back inside that save."""
        mock_query.side_effect = _query_router(
            xref=pd.DataFrame({"prop_id": ["ipr_wt_uid"]}),
            enthid=pd.DataFrame({"enthid": [111], "well_name": ["B-028"]}),
        )
        batch = next_entry_datetime()
        push_prop("MPB-28", "ipr_wt_uid", 1.0, "scott", entry_datetime=batch)
        push_prop("MPB-28", "ipr_wt_uid", 2.0, "scott")

        stamps = [c[0][1]["entry_datetime"] for c in mock_write.call_args_list]
        assert stamps[1] > stamps[0] == batch


# ── rendering stamps in Alaska time ─────────────────────────────────────────


class TestAlaskaRendering(_CacheResetMixin):
    """Kaelin, 2026-08-03: "have a 19:22 timestamp, which I don't know what
    that means." Stamps are STORED as UTC instants — the column is an ordering
    key — and rendered in Alaska time wherever a person reads one."""

    def test_the_timestamp_kaelin_could_not_read(self):
        utc = datetime(2026, 8, 3, 19, 22, tzinfo=timezone.utc)
        assert phc.format_alaska(utc) == "2026-08-03 11:22 AKDT"

    def test_winter_stamps_render_akst(self):
        utc = datetime(2026, 12, 3, 19, 22, tzinfo=timezone.utc)
        assert phc.format_alaska(utc) == "2026-12-03 10:22 AKST"

    def test_evening_ak_save_keeps_its_own_date(self):
        """21:00 AKDT on the 3rd is 05:00 UTC on the 4th — a raw UTC date shows
        the engineer the wrong day, which is why the captions convert."""
        utc = datetime(2026, 8, 4, 5, 0, tzinfo=timezone.utc)
        assert phc.format_alaska(utc, "%Y-%m-%d") == "2026-08-03"

    def test_naive_conversion_drops_the_offset_for_widgets(self):
        utc = datetime(2026, 8, 3, 19, 22, tzinfo=timezone.utc)
        local = phc.to_alaska(utc)
        assert local.tzinfo is None and (local.hour, local.minute) == (11, 22)

    def test_migrated_date_only_rows_are_not_shifted(self):
        """ka9612's 2026-04-16 DART bulk load was an ``entry_date DATE`` before
        the 2026-07-08 migration. Converting midnight UTC would render it
        '2026-04-15 16:00' — wrong day, and a time of day it never had."""
        bulk = datetime(2026, 4, 16, 0, 0, tzinfo=timezone.utc)
        assert phc.format_alaska(bulk, "%Y-%m-%d %H:%M") == "2026-04-16 00:00"
        assert phc.to_alaska(bulk).day == 16

    def test_a_real_write_at_almost_midnight_still_converts(self):
        """The sentinel is EXACT midnight — one microsecond past is a genuine
        app write and must render in AK like any other."""
        real = datetime(2026, 4, 16, 0, 0, 0, 1, tzinfo=timezone.utc)
        assert phc.to_alaska(real).day == 15

    def test_storage_stays_utc(self):
        """The allocator must never start handing out local time: two rows an
        hour apart would collide in the November fold and the DESC tie-break
        that resolves every read would pick arbitrarily."""
        assert phc.next_entry_datetime().tzinfo is timezone.utc

    def test_the_dst_fold_that_makes_local_storage_unsafe(self):
        """Documents WHY storage is UTC: 2026-11-01 01:30 AK happens twice."""
        from zoneinfo import ZoneInfo

        ak = ZoneInfo(phc.ALASKA_TZ)
        first = datetime(2026, 11, 1, 1, 30, tzinfo=ak, fold=0)
        second = datetime(2026, 11, 1, 1, 30, tzinfo=ak, fold=1)
        assert first.astimezone(timezone.utc) != second.astimezone(timezone.utc)
        # …yet identical as wall clock, so a naive-local column loses the order.
        assert first.replace(tzinfo=None) == second.replace(tzinfo=None)

    def test_conversion_is_order_preserving(self):
        stamps = [phc.next_entry_datetime() for _ in range(50)]
        local = [phc.to_alaska(s) for s in stamps]
        assert local == sorted(local)

    def test_bad_input_is_returned_untouched_not_raised(self):
        """A caption is never worth crashing a page over."""
        assert phc.to_alaska(None) is None
        assert phc.to_alaska("not a timestamp") == "not a timestamp"
        assert phc.format_alaska("not a timestamp") == "not a timestamp"


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
    def test_none_value_is_refused_not_bound_as_null(self, mock_query, mock_write):
        """prop_value is DOUBLE **NOT NULL** in mpu.wells.prop_hist.

        This test used to assert None bound as a SQL NULL "un-pin marker" —
        it passed only because the write was mocked. Against the real table
        every such push died with DELTA_NOT_NULL_CONSTRAINT_VIOLATED, and the
        callers' broad `except` turned that into a silent no-op: the 🗑 Clear
        saved IPR button and the 🔒 lock checkboxes never persisted anything,
        and the table holds ZERO NULL prop_value rows (found 2026-08-04).
        Clearing is an explicit per-prop sentinel now."""
        mock_query.side_effect = _query_router(
            xref=pd.DataFrame({"prop_id": ["ipr_wt_uid"]}),
            enthid=pd.DataFrame({"enthid": [12345], "well_name": ["B-028"]}),
        )

        with pytest.raises(phc.PropHistError, match="NOT NULL"):
            push_prop("MPB-28", "ipr_wt_uid", None, "scott")

        mock_write.assert_not_called()

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


# ── entry-user PROVIDER (the hosted-app attribution fix, 2026-07-30) ────────
# On Databricks Apps every session shares one container and current_user() is
# the SERVICE PRINCIPAL — app.py registers a provider that reads the
# forwarded-user header so each save is stamped with the real engineer.


class TestEntryUserProvider:
    @pytest.fixture(autouse=True)
    def _reset_provider(self, monkeypatch):
        monkeypatch.delenv("WOFFL_ENTRY_USER", raising=False)
        _reset_caches()  # a cached current_user from a prior test must not leak
        yield
        phc.set_entry_user_provider(None)

    @patch("woffl.assembly.prop_hist_client.execute_query")
    def test_provider_beats_current_user(self, mock_query):
        mock_query.return_value = pd.DataFrame({"current_user": ["svc_principal"]})
        phc.set_entry_user_provider(lambda: "scott@hilcorp.com")
        assert resolve_entry_user() == "scott@hilcorp.com"
        assert mock_query.call_count == 0

    def test_env_override_still_beats_the_provider(self, monkeypatch):
        monkeypatch.setenv("WOFFL_ENTRY_USER", "override_user")
        phc.set_entry_user_provider(lambda: "scott@hilcorp.com")
        assert resolve_entry_user() == "override_user"

    @patch("woffl.assembly.prop_hist_client.execute_query")
    def test_provider_called_per_resolve_never_cached(self, mock_query):
        """Identity is per-session on a shared host — caching one user's name
        would stamp it onto everyone else's saves."""
        users = iter(["engineer_a", "engineer_b"])
        phc.set_entry_user_provider(lambda: next(users))
        assert resolve_entry_user() == "engineer_a"
        assert resolve_entry_user() == "engineer_b"
        assert mock_query.call_count == 0

    @patch("woffl.assembly.prop_hist_client.execute_query")
    def test_provider_none_falls_through_to_current_user(self, mock_query):
        """Local run: no forwarded headers → provider returns None → the
        normal current_user() path (cached) takes over."""
        mock_query.return_value = pd.DataFrame({"current_user": ["scott_local"]})
        phc.set_entry_user_provider(lambda: None)
        assert resolve_entry_user() == "scott_local"
        assert mock_query.call_count == 1

    @patch("woffl.assembly.prop_hist_client.execute_query")
    def test_provider_exception_never_blocks_a_save(self, mock_query):
        mock_query.return_value = pd.DataFrame({"current_user": ["svc_principal"]})

        def boom():
            raise RuntimeError("st.context unavailable")

        phc.set_entry_user_provider(boom)
        assert resolve_entry_user() == "svc_principal"
