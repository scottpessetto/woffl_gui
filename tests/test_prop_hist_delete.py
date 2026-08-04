"""The approval-gated prop_hist delete (`prop_hist_client.delete_props`).

prop_hist is append-only by default and that default is load-bearing — the
trail is how Kaelin caught the 2026-08-03 as-built incident. Deletion exists
only because append cannot retract rows that should never have been written:
Scott's call, 2026-08-04, backed by his MODIFY grant.

These tests pin the awkwardness that makes it safe. Every one of them describes
a way someone could delete well data by accident, and asserts we don't.
"""

import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

import woffl.assembly.prop_hist_client as phc
from woffl.assembly.databricks_client import WritesDisabledError
from woffl.assembly.prop_hist_client import (
    DELETE_GATE_ENV,
    DeleteNotApprovedError,
    DeleteTarget,
    delete_props,
)

STAMP = datetime(2026, 8, 3, 19, 23, 53, 154819, tzinfo=timezone.utc)
TARGET = DeleteTarget("MPC-02", "jpump_md", 6270.2230992, STAMP)
REASON = "2026-08-03 as-built incident: app-authored pump depth"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(DELETE_GATE_ENV, raising=False)
    monkeypatch.delenv("ALLOW_DATABRICKS_WRITES", raising=False)
    phc._xref_cache["value"] = None
    phc._xref_cache["expires_at"] = 0.0
    phc._enthid_cache["value"] = None
    phc._enthid_cache["expires_at"] = 0.0
    yield


@pytest.fixture
def wired(monkeypatch):
    """xref + enthid + a single matching row + a Delta version, all faked."""
    monkeypatch.setattr(phc, "fetch_prop_xref", lambda: {"jpump_md", "ipr_pwf"})
    monkeypatch.setattr(phc, "_resolve_enthid", lambda w: 36511486)
    monkeypatch.setattr(phc, "_table_version", lambda: 727)
    state = {"matches": 1, "executed": []}

    def _query(sql):
        if "count(*)" in sql:
            return pd.DataFrame({"n": [state["matches"]]})
        raise AssertionError(f"unexpected query: {sql[:60]}")

    def _via_connector(runner):
        class _Cur:
            def execute(self, sql, params):
                state["executed"].append((sql, params))

            rowcount = -1

        return runner(_Cur())

    monkeypatch.setattr(phc, "execute_query", _query)
    monkeypatch.setattr(phc, "_execute_via_connector", _via_connector)
    return state


class TestRefusals:
    def test_dry_run_is_the_default(self, wired):
        out = delete_props([TARGET], reason=REASON, expect=1)
        assert out["applied"] is False and out["deleted"] == 0
        assert out["planned"] == 1
        assert wired["executed"] == [], "a dry run must not touch the table"

    def test_apply_without_the_delete_gate_raises(self, wired, monkeypatch):
        """The normal write gate alone is NOT enough — that's the whole point of
        a second env var."""
        monkeypatch.setenv("ALLOW_DATABRICKS_WRITES", "true")
        with pytest.raises(DeleteNotApprovedError, match=DELETE_GATE_ENV):
            delete_props([TARGET], reason=REASON, expect=1, apply=True)
        assert wired["executed"] == []

    def test_apply_without_the_write_gate_raises(self, wired, monkeypatch):
        monkeypatch.setenv(DELETE_GATE_ENV, "true")
        with pytest.raises(WritesDisabledError):
            delete_props([TARGET], reason=REASON, expect=1, apply=True)
        assert wired["executed"] == []

    def test_wrong_expect_count_aborts(self, wired):
        """A manifest that grew since the caller last counted it must not run."""
        with pytest.raises(DeleteNotApprovedError, match="expect=2"):
            delete_props([TARGET], reason=REASON, expect=2, apply=True)
        assert wired["executed"] == []

    def test_empty_reason_aborts(self, wired):
        with pytest.raises(DeleteNotApprovedError, match="reason"):
            delete_props([TARGET], reason="   ", expect=1, apply=True)

    def test_unknown_prop_id_aborts(self, wired):
        bad = DeleteTarget("MPC-02", "not_a_prop", 1.0, STAMP)
        with pytest.raises(phc.UnknownPropIdError):
            delete_props([bad], reason=REASON, expect=1)

    @pytest.mark.parametrize("matches", [0, 2])
    def test_target_must_match_exactly_one_row(self, wired, matches):
        """0 = wrong stamp or already gone. 2 = the manifest is not specific
        enough, and a blind delete would take a row nobody reviewed."""
        wired["matches"] = matches
        with pytest.raises(DeleteNotApprovedError, match=f"matches {matches} rows"):
            delete_props([TARGET], reason=REASON, expect=1)
        assert wired["executed"] == []

    def test_one_bad_target_aborts_the_whole_batch(self, wired, monkeypatch):
        """Validation is all-or-nothing: a partial delete is the worst outcome
        because it leaves the operator unsure what actually happened."""
        monkeypatch.setenv("ALLOW_DATABRICKS_WRITES", "true")
        monkeypatch.setenv(DELETE_GATE_ENV, "true")
        wired["matches"] = 0
        good = DeleteTarget("MPC-02", "ipr_pwf", 1141.0, STAMP)
        with pytest.raises(DeleteNotApprovedError):
            delete_props([good, TARGET], reason=REASON, expect=2, apply=True)
        assert wired["executed"] == []


class TestApply:
    @pytest.fixture(autouse=True)
    def _gates(self, monkeypatch):
        monkeypatch.setenv("ALLOW_DATABRICKS_WRITES", "true")
        monkeypatch.setenv(DELETE_GATE_ENV, "true")

    def test_deletes_by_exact_four_column_match(self, wired):
        out = delete_props([TARGET], reason=REASON, expect=1, apply=True)
        assert out["applied"] is True and out["deleted"] == 1
        (sql, params), = wired["executed"]
        assert sql.startswith("DELETE FROM mpu.wells.prop_hist")
        # Every column named — never a date-granularity or predicate-only match.
        for col in ("enthid", "prop_id", "prop_value", "entry_datetime"):
            assert f":{col}" in sql and col in params
        assert params["entry_datetime"] == STAMP
        assert params["enthid"] == 36511486

    def test_uses_entry_datetime_not_the_migrated_away_entry_date(self, wired):
        """Kaelin's DART delete_prop filters `entry_date`, which stopped
        existing when the column became `entry_datetime TIMESTAMP` on
        2026-07-08 — that predicate raises column-not-found here."""
        delete_props([TARGET], reason=REASON, expect=1, apply=True)
        (sql, _), = wired["executed"]
        assert "entry_datetime" in sql
        assert "entry_date " not in sql and ":entry_date " not in sql

    def test_report_carries_the_time_travel_undo(self, wired):
        out = delete_props([TARGET], reason=REASON, expect=1, apply=True)
        assert out["version_before"] == 727
        assert "RESTORE TABLE mpu.wells.prop_hist TO VERSION AS OF 727" in out["undo"]
        assert out["reason"] == REASON

    def test_as_built_ids_are_deletable_even_though_unwritable(self, wired):
        """The asymmetry we want: push_prop can never AUTHOR an as-built value,
        but woffl must be able to retract the bad ones it already wrote."""
        assert "jpump_md" in phc.AS_BUILT_PROP_IDS
        with pytest.raises(phc.AsBuiltPropError):
            phc.push_prop("MPC-02", "jpump_md", 6270.0, "scott")
        out = delete_props([TARGET], reason=REASON, expect=1, apply=True)
        assert out["deleted"] == 1


class TestNoAppPath:
    def test_no_gui_module_calls_delete_props(self):
        """Deleting well data is a console act with a human reading the dry
        run, never a button. If this fails, someone wired it into a page."""
        gui = Path(__file__).resolve().parent.parent / "woffl" / "gui"
        offenders = [
            p.relative_to(gui).as_posix()
            for p in gui.rglob("*.py")
            if "delete_props" in p.read_text(encoding="utf-8")
        ]
        assert offenders == [], f"delete_props reached from the GUI: {offenders}"

    def test_the_gate_is_off_by_default_in_this_environment(self):
        assert os.environ.get(DELETE_GATE_ENV) is None
        assert phc.delete_gate_enabled() is False
