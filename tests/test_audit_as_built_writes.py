"""The as-built overwrite audit (woffl/assembly/audit_as_built_writes.py).

Built from the real 2026-08-03 rows Kaelin pulled out of prop_hist: five wells
whose jpump_md was replaced by the interpolated JP_TVD (C-002 = enthid
36511486, 7688 ft -> 6270.2230992 ft), alongside a well nobody touched and a
well whose only row the app ever wrote. The audit has to separate those three
cases — a false positive sends someone to "fix" a correct depth.
"""

import pandas as pd
import pytest

from woffl.assembly import audit_as_built_writes as audit

SINCE = pd.Timestamp("2026-07-30", tz="UTC")
OLD = pd.Timestamp("2026-04-16 00:00:00", tz="UTC")
BAD = pd.Timestamp("2026-08-03 19:23:53", tz="UTC")


def _row(well, enthid, prop_id, value, at, user):
    return {
        "well_name": well,
        "enthid": enthid,
        "prop_id": prop_id,
        "prop_value": value,
        "entry_datetime": at,
        "entry_user": user,
    }


@pytest.fixture
def history():
    return pd.DataFrame(
        [
            # C-002: the tally depth, then the app's interpolated TVD.
            _row("C-002", 36511486, "jpump_md", 7688.0, OLD, "ka9612"),
            _row("C-002", 36511486, "jpump_md", 6270.2230992, BAD, "Scott.Pessetto@hilcorp.com"),
            # …and its casing OD flattened to the 6.875 UI fallback.
            _row("C-002", 36511486, "casing_out_dia", 9.625, OLD, "ka9612"),
            _row("C-002", 36511486, "casing_out_dia", 6.875, BAD, "Scott.Pessetto@hilcorp.com"),
            # B-028: never reviewed on the bad build — must not be flagged.
            _row("B-028", 32795728, "jpump_md", 4974.0, OLD, "ka9612"),
            # G-016: the app authored the ONLY row it has — nothing to restore.
            _row("G-016", 36536208, "jpump_md", 4168.9681920858, BAD, "Scott.Pessetto@hilcorp.com"),
        ]
    )


def test_only_post_cutoff_current_values_are_flagged(history):
    hits = audit.find_overwrites(history, SINCE)
    assert set(zip(hits["well_name"], hits["prop_id"])) == {
        ("C-002", "jpump_md"),
        ("C-002", "casing_out_dia"),
        ("G-016", "jpump_md"),
    }


def test_restore_value_is_the_last_pre_cutoff_row(history):
    hits = audit.find_overwrites(history, SINCE).set_index(["well_name", "prop_id"])
    md = hits.loc[("C-002", "jpump_md")]
    assert md["current_value"] == pytest.approx(6270.2230992)
    assert md["restore_value"] == pytest.approx(7688.0)
    assert md["restore_by"] == "ka9612"
    assert hits.loc[("C-002", "casing_out_dia"), "restore_value"] == pytest.approx(9.625)


def test_app_authored_orphan_has_no_restore_value(history):
    """No pre-incident row ⇒ the audit must say so rather than invent one."""
    hits = audit.find_overwrites(history, SINCE).set_index(["well_name", "prop_id"])
    assert pd.isna(hits.loc[("G-016", "jpump_md"), "restore_value"])


def test_untouched_well_is_absent(history):
    hits = audit.find_overwrites(history, SINCE)
    assert "B-028" not in set(hits["well_name"])


def test_restore_statements_are_insert_only_and_skip_orphans(history):
    hits = audit.find_overwrites(history, SINCE)
    stmts = audit.restore_statements(hits, "scott.pessetto@hilcorp.com")

    assert len(stmts) == 2  # G-016 has nothing to restore
    assert all(s.startswith("INSERT INTO mpu.wells.prop_hist ") for s in stmts)
    assert all(";" in s and " DELETE" not in s.upper() for s in stmts)
    md = next(s for s in stmts if "'jpump_md'" in s)
    assert "36511486" in md and "7688.0" in md
    assert "scott.pessetto@hilcorp.com" in md
    # The repair is attributable to whoever runs it, not backdated.
    assert "current_timestamp()" in md


def test_a_clean_history_flags_nothing():
    clean = pd.DataFrame([_row("B-028", 32795728, "jpump_md", 4974.0, OLD, "ka9612")])
    assert audit.find_overwrites(clean, SINCE).empty
