"""Find (and generate the undo for) app-authored writes to as-built props.

Between 2026-07-30 and 2026-08-03 the pad-review write-through
(`woffl.gui.review_persistence.FIELD_MAP`) carried `jpump_md` and
`casing_out_dia`. A review store entry always holds SOME number for those —
the Databricks value when present, a UI or force-fit substitute otherwise —
so the write-through could not tell a decision from a default, and it pushed
the defaults: measured pump depths were replaced by the locally interpolated
JP_TVD (C-002: 7688 ft -> 6270.2230992 ft) and casing ODs by the 6.875
fallback. Kaelin caught it in `vw_prop_mech` — the giveaway was jpump_md
values carrying ten decimal places.

The code paths are fixed (`prop_hist_client.AS_BUILT_PROP_IDS` now rejects
these ids at the single write chokepoint). This script is the other half:
locating the damage and printing exactly what to put back.

prop_hist is append-only, and `execute_write` only ever runs a single INSERT,
so "restore" means inserting a NEW row carrying the last pre-incident value.
This module deliberately does NOT write. It emits the INSERT statements for
whoever owns the well file to review and run — the guard it would have to
punch through exists precisely so no automated path can author these ids.

Usage:
    python -m woffl.assembly.audit_as_built_writes
    python -m woffl.assembly.audit_as_built_writes --since 2026-07-30 --sql
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone

import pandas as pd

from woffl.assembly.databricks_client import execute_query
from woffl.assembly.prop_hist_client import AS_BUILT_PROP_IDS, format_alaska

# The day `jpump_md` / `casing_out_dia` entered FIELD_MAP. Every as-built row
# stamped on or after it is suspect; the newest row BEFORE it is the value to
# restore.
DEFAULT_SINCE = "2026-07-30"

_HISTORY_QUERY = """
SELECT h.well_name, p.enthid, p.prop_id, p.prop_value,
       p.entry_datetime, p.entry_user
FROM mpu.wells.prop_hist p
JOIN mpu.wells.vw_well_header h ON p.enthid = h.enthid
WHERE p.prop_id IN ({ids})
ORDER BY h.well_name, p.prop_id, p.entry_datetime
"""


def fetch_history() -> pd.DataFrame:
    """Full prop_hist history for every as-built prop id, oldest row first."""
    ids = ",".join(f"'{p}'" for p in sorted(AS_BUILT_PROP_IDS))  # our own constant
    df = execute_query(_HISTORY_QUERY.format(ids=ids))
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "well_name", "enthid", "prop_id", "prop_value",
                "entry_datetime", "entry_user",
            ]
        )
    df = df.copy()
    df["entry_datetime"] = pd.to_datetime(df["entry_datetime"], utc=True)
    return df


def find_overwrites(history: pd.DataFrame, since: datetime) -> pd.DataFrame:
    """One row per (well, prop) whose CURRENT value was written on/after
    ``since``, with the last value that preceded it.

    ``restore_value`` is NaN when nothing preceded it — the app invented the
    only row that (well, prop) has ever had, so there is nothing to roll back
    to and the data team has to supply the real measurement.
    """
    rows = []
    for (well, prop_id), grp in history.groupby(["well_name", "prop_id"], sort=True):
        grp = grp.sort_values("entry_datetime")
        latest = grp.iloc[-1]
        if latest["entry_datetime"] < since:
            continue  # current value predates the incident — untouched
        prior = grp[grp["entry_datetime"] < since]
        good = prior.iloc[-1] if not prior.empty else None
        rows.append(
            {
                "well_name": well,
                "enthid": int(latest["enthid"]),
                "prop_id": prop_id,
                "current_value": float(latest["prop_value"]),
                "current_at": latest["entry_datetime"],
                "current_by": str(latest["entry_user"]),
                "restore_value": (
                    float(good["prop_value"]) if good is not None else float("nan")
                ),
                "restore_from": (good["entry_datetime"] if good is not None else pd.NaT),
                "restore_by": str(good["entry_user"]) if good is not None else "",
            }
        )
    return pd.DataFrame(rows)


def restore_statements(overwrites: pd.DataFrame, entry_user: str) -> list[str]:
    """Copy-pasteable INSERTs re-asserting each last-known-good value.

    Stamped with the person running the repair, not the original author: the
    restore is a new, attributable edit, and pretending otherwise would put a
    fake row in the audit trail.
    """
    out = []
    for _, r in overwrites.iterrows():
        if pd.isna(r["restore_value"]):
            continue
        out.append(
            "INSERT INTO mpu.wells.prop_hist "
            "(enthid, prop_id, prop_value, entry_datetime, entry_user) VALUES "
            f"({r['enthid']}, '{r['prop_id']}', {r['restore_value']!r}, "
            f"current_timestamp(), '{entry_user}');"
            f"  -- {r['well_name']}: undo {r['current_value']!r}"
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--since",
        default=DEFAULT_SINCE,
        help=f"ISO date; rows stamped on/after it are suspect (default {DEFAULT_SINCE})",
    )
    ap.add_argument(
        "--sql",
        action="store_true",
        help="also print the restore INSERT statements",
    )
    ap.add_argument(
        "--user",
        default="<your.email@hilcorp.com>",
        help="entry_user to stamp on the restore rows",
    )
    args = ap.parse_args()

    since = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)

    history = fetch_history()
    if history.empty:
        print("prop_hist has no rows for any as-built prop id.", file=sys.stderr)
        return 2

    hits = find_overwrites(history, since)
    print(f"As-built props audited: {', '.join(sorted(AS_BUILT_PROP_IDS))}")
    print(f"Suspect cutoff:         {since:%Y-%m-%d} (UTC); times shown in AK")
    print(f"(well, prop) pairs whose current value lands after the cutoff: {len(hits)}")
    if hits.empty:
        return 0

    print()
    for _, r in hits.sort_values(["well_name", "prop_id"]).iterrows():
        restore = (
            "NO PRIOR VALUE — the data team must supply the measurement"
            if pd.isna(r["restore_value"])
            else f"{r['restore_value']!r} ({r['restore_by']}, "
            f"{format_alaska(r['restore_from'], '%Y-%m-%d')})"
        )
        print(f"  {r['well_name']:<10} {r['prop_id']:<16}")
        print(f"      now:     {r['current_value']!r} ({r['current_by']}, "
              f"{format_alaska(r['current_at'])})")
        print(f"      restore: {restore}")

    orphans = int(hits["restore_value"].isna().sum())
    if orphans:
        print(f"\n{orphans} pair(s) have no pre-cutoff row to restore.")

    if args.sql:
        print("\n-- Review before running. prop_hist is append-only: these add")
        print("-- new latest rows, they do not delete the bad ones.")
        for stmt in restore_statements(hits, args.user):
            print(stmt)
    else:
        print("\nRe-run with --sql --user you@hilcorp.com for the restore INSERTs.")

    return 1


if __name__ == "__main__":
    sys.exit(main())
