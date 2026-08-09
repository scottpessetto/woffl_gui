"""prop_hist Client

Minimal read/write client for `mpu.wells.prop_hist` -- the append-only
property-history table (enthid, prop_id, prop_value, entry_datetime,
entry_user). Phase 1 use case: pinning a well's chosen IPR anchor test
(`ipr_wt_uid`) so it survives across sessions/users (see docs/prop_hist_asks.md
and the woffl-prop-hist-persistence plan).

Pattern adapted from Kyle's `dart/datapush/mppush.py` (reviewed, not
imported) -- see the plan's "DART review" section for what was adopted vs
deliberately dropped (`os.getlogin()` as entry_user, `delete_prop`,
sqlalchemy). This module is Hilcorp/fork-specific plumbing like
`databricks_client.py`, not upstream `woffl` library code -- no
`upstream_sync.md` entry needed.

This module has NO Streamlit dependency (`woffl.gui` may not be importable /
running) -- caching is a plain module-level TTL dict, mirroring
`databricks_client._TOKEN_CACHE`, not `st.cache_data`.

All writes go through `databricks_client.execute_write`, which enforces the
`ALLOW_DATABRICKS_WRITES` env gate and refuses anything but a single
parameterized INSERT. Every push here is the well-known
`push_prop`/`fetch_latest_prop` shape; there is no delete/update in this
module (corrections are new rows; un-pinning writes a SQL NULL prop_value --
see W3). NOTE: `wt_uid` values in `vw_well_test` are signed and span both
positive and negative ranges (observed roughly -3.6M to +3.1M) -- prop_value
must NEVER be interpreted with a sign-based rule. NULL is the only safe
"no value" marker.
"""

from __future__ import annotations

import math
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence, Tuple

import pandas as pd

from woffl.assembly.databricks_client import (
    WritesDisabledError,
    _execute_via_connector,
    execute_query,
    execute_write,
)
from woffl.assembly.well_test_client import _normalize_well_name

PROP_XREF_QUERY = "SELECT prop_id FROM mpu.wells.prop_xref"

WELL_ENTHID_QUERY = """\
SELECT enthid, well_name
FROM mpu.wells.vw_well_header
WHERE well_type = 'prod'
"""

PROP_HIST_INSERT_SQL = (
    "INSERT INTO mpu.wells.prop_hist "
    "(enthid, prop_id, prop_value, entry_datetime, entry_user) "
    "VALUES (:enthid, :prop_id, :prop_value, :entry_datetime, :entry_user)"
)

# Batched form of the SAME insert (see `push_props`). One logical save used to
# be one statement PER property: 6-9 serialized Delta commits at ~0.2-1 s each,
# which is what made "Save IPR as well default" hang for seconds. Delta takes a
# multi-row VALUES list happily, and `_validate_single_insert` already permits
# it (it only forbids ';' chaining), so a save is now ONE statement.
#
# Every marker is numbered — including enthid/stamp/user, which are identical
# on every row — so no parameter name is ever repeated in the statement. That
# is deliberate: repeated named markers are a connector-behaviour bet, and the
# write path is the one place in this app that cannot be smoke-tested against
# the warehouse (AGENTS.md section 3).
PROP_HIST_INSERT_HEAD = (
    "INSERT INTO mpu.wells.prop_hist "
    "(enthid, prop_id, prop_value, entry_datetime, entry_user) VALUES "
)

# A save is at most nine rows (six IPR values + three friction coefficients).
# The cap keeps one statement's text bounded no matter what a future caller
# hands in; anything larger wants a different design, not a longer string.
_MAX_BATCH_ROWS = 32

# Engineer comments live in their own table: prop_hist.prop_value is
# DOUBLE NOT NULL, so free text physically cannot go on a prop row. A comment
# is bound to the SAVE, not to any single property — one click of "Save IPR as
# well default" writes up to nine prop rows — so the join key is the batch
# stamp that every row of that save shares. See `push_prop(entry_datetime=...)`.
ENG_COMMENT_INSERT_SQL = (
    "INSERT INTO mpu.wells.woffl_eng_comment "
    "(enthid, entry_datetime, entry_user, context, comment_text) "
    "VALUES (:enthid, :entry_datetime, :entry_user, :context, :comment_text)"
)

CURRENT_USER_QUERY = "SELECT current_user() AS current_user"

_CACHE_TTL_SECONDS = 3600.0

# Module-level TTL caches -- deliberately NOT st.cache_data (this module must
# work without Streamlit, e.g. from a plain script or a pytest process).
_xref_cache: dict = {"value": None, "expires_at": 0.0}
_enthid_cache: dict = {"value": None, "expires_at": 0.0}  # {name: [enthid, ...]}
_entry_user_cache: dict = {"value": None}

# ── entry_datetime allocation ───────────────────────────────────────────────
#
# entry_datetime is load-bearing twice over: every read resolves a property by
# "latest row per (enthid, prop_id) ORDER BY entry_datetime DESC", and it is the
# ONLY thing tying one logical save's rows together (prop_hist has no batch id),
# which is what `mpu.wells.woffl_eng_comment` joins on to attach the engineer's
# note. Both properties need stamps that are strictly INCREASING per save.
#
# `datetime.now()` does not provide that. The Windows system clock has 15.625 ms
# granularity — measured on this workstation, 2000 back-to-back
# `datetime.now(timezone.utc)` calls returned ONE distinct value. Two saves
# inside one tick therefore collided outright: the comments merged (the
# comment join is `GROUP BY enthid, entry_datetime`, so one note won), and worse,
# the tie made `ROW_NUMBER() ... ORDER BY entry_datetime DESC` arbitrary — the
# well could reopen on the FIRST save's values, silently discarding the second.
#
# So stamps are allocated here instead: wall-clock when the clock has moved,
# else the previous stamp plus one microsecond (the column round-trips to the
# microsecond, verified 2026-07-08). Strictly monotonic per process, which is
# the failure mode that actually occurs — a double-click or two quick saves in
# one session. It does NOT coordinate across processes; on Databricks Apps that
# is one Streamlit process per container, and two containers landing in the same
# MICROsecond is a different order of unlikely from the 15.6 ms window this
# closes. Bumping can run the stamp ahead of the true clock by 1 µs per row
# issued inside a single tick; a save is at most nine rows, so the skew is
# bounded far below the granularity it is compensating for.
_STAMP_LOCK = threading.Lock()
_STAMP_TICK = timedelta(microseconds=1)
_last_stamp: Optional[datetime] = None


def next_entry_datetime() -> datetime:
    """A UTC stamp strictly greater than every stamp this process has issued.

    Use for ANY new prop_hist write. A caller writing several props as ONE
    logical save takes a single stamp from here and passes it to every
    ``push_prop`` (see :func:`push_prop`'s ``entry_datetime``) — that shared
    value is the save's batch identity, so it must not be re-allocated per row.
    """
    global _last_stamp
    with _STAMP_LOCK:
        stamp = datetime.now(timezone.utc)
        if _last_stamp is not None and stamp <= _last_stamp:
            stamp = _last_stamp + _STAMP_TICK
        _last_stamp = stamp
        return stamp


# ── rendering stamps for humans ─────────────────────────────────────────────
#
# Stamps are STORED as UTC instants and that is not negotiable: entry_datetime
# is an ordering key before it is a readable field. Every read resolves a
# property by "latest row per (enthid, prop_id) ORDER BY entry_datetime DESC",
# so the column has to be monotonic and unambiguous.
#
# Storing Alaska wall time instead would break that twice a year. On
# 2026-11-01 local 01:30 occurs TWICE — 09:30 and 10:30 UTC — so two rows an
# hour apart would carry the identical stamp and the later one could lose the
# tie-break. That is the same class of bug as the 15.6 ms clock collision
# fixed in next_entry_datetime, except a naive-local column makes it
# unfixable rather than merely subtle.
#
# The actual complaint (Kaelin, 2026-08-03: "have a 19:22 timestamp, which I
# don't know what that means") is a DISPLAY problem — 19:22 UTC is 11:22 AKDT.
# So convert at the edge, everywhere a person reads a stamp, and label the
# zone. Storage stays UTC.
ALASKA_TZ = "America/Anchorage"


def to_alaska(ts, naive: bool = True):
    """A UTC stamp rendered in Alaska local time (AKDT/AKST per the date).

    ``naive=True`` (default) drops the offset once converted, which is what
    display widgets want — a bare wall-clock reading under a column header
    that names the zone. ``naive=False`` keeps it tz-aware for ``%Z``.

    Rows at EXACTLY midnight UTC are left alone. Those are the pre-2026-07-08
    ``entry_date DATE`` rows the migration widened into timestamps — the DART
    bulk load of 2026-04-16 and friends. They never carried a time of day, so
    shifting them by the UTC offset would both invent a precision they don't
    have and move them to the previous DATE (2026-04-16 -> "2026-04-15 16:00"),
    which reads as a real evening edit that never happened. A genuine app write
    has microsecond precision, so exact midnight is a safe sentinel for "this
    is a date, not an instant".

    Fail-soft: an unparseable value, or a runtime with no tz database, returns
    the input untouched. A caption is never worth crashing a page over.
    """
    if ts is None:
        return ts
    try:
        from zoneinfo import ZoneInfo

        import pandas as _pd

        stamp = _pd.Timestamp(ts)
        stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp
        if (stamp.hour, stamp.minute, stamp.second, stamp.microsecond) == (0, 0, 0, 0):
            return stamp.tz_localize(None) if naive else stamp
        local = stamp.tz_convert(ZoneInfo(ALASKA_TZ))
        return local.tz_localize(None) if naive else local
    except Exception:
        return ts


def format_alaska(ts, fmt: str = "%Y-%m-%d %H:%M %Z") -> str:
    """``to_alaska`` as a labelled string, e.g. '2026-08-03 11:22 AKDT'."""
    local = to_alaska(ts, naive=False)
    try:
        return local.strftime(fmt)
    except Exception:
        return str(ts)


_PROP_ID_SHAPE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

# As-built completion facts. These describe steel in the ground — they come
# off the wellbore diagram / tubing tally and are owned by the data team, NOT
# by anything this app fits, solves, or defaults. woffl only ever CONSUMES
# them (vw_prop_mech -> JP_MD / casing dims), so any push at this id is a bug
# by construction, and a silent one: the solver keeps running on a fabricated
# depth while the well file now disagrees with the diagram.
#
# 2026-08-03 incident: the pad review write-through pushed jpump_md and
# casing_out_dia. Eight wells (C-02, C-05, C-015, C-026, C-40, G-16, J-9,
# J-26) had their measured depth replaced by the locally interpolated JP_TVD
# (e.g. C-002: 7688 ft MD -> 6270.223 ft) and their casing OD by the 6.875
# fallback. The two fabrication sites are fixed at the source
# (step_review_wells) and the ids are off the write-through map
# (review_persistence.FIELD_MAP); this is the chokepoint backstop so no
# future caller can reintroduce it.
AS_BUILT_PROP_IDS = frozenset(
    {
        "jpump_md",
        "casing_out_dia",
        "casing_inn_dia",
        "tubing_out_dia",
        "tubing_inn_dia",
    }
)


class PropHistError(ValueError):
    """Base error for prop_hist_client operations."""


class UnknownPropIdError(PropHistError):
    """Raised when prop_id isn't a valid key in mpu.wells.prop_xref (or is
    shaped unsafely for SQL interpolation in the read path)."""


class AsBuiltPropError(PropHistError):
    """Raised when a write targets an as-built completion property
    (``AS_BUILT_PROP_IDS``). Read-only from woffl: the data team owns them."""


class EnthidResolutionError(PropHistError):
    """Raised when a well_name resolves to zero or more than one enthid in
    mpu.wells.vw_well_header (DART's push_prop guard, ported)."""


def fetch_prop_xref(force_refresh: bool = False) -> set[str]:
    """Cached whitelist of valid prop_id values from mpu.wells.prop_xref.

    Cached for `_CACHE_TTL_SECONDS` (module-level TTL dict, not st.cache --
    this module must work without Streamlit). Pass force_refresh=True to
    bypass a live cache entry.
    """
    now = time.time()
    if (
        not force_refresh
        and _xref_cache["value"] is not None
        and now < _xref_cache["expires_at"]
    ):
        return _xref_cache["value"]

    df = execute_query(PROP_XREF_QUERY)
    valid = set(df["prop_id"].astype(str)) if not df.empty else set()

    _xref_cache["value"] = valid
    _xref_cache["expires_at"] = now + _CACHE_TTL_SECONDS
    return valid


def _fetch_enthid_groups(force_refresh: bool = False) -> dict[str, list[int]]:
    """Cached {normalized_well_name: [enthid, ...]} grouping.

    Grouped rather than collapsed to a single value so the 0-match /
    multi-match guards below can tell "well not found" apart from "well name
    is ambiguous in vw_well_header" -- both raise, but with different typed
    messages. `well_enthid_map()` is the public single-valued view.
    """
    now = time.time()
    if (
        not force_refresh
        and _enthid_cache["value"] is not None
        and now < _enthid_cache["expires_at"]
    ):
        return _enthid_cache["value"]

    df = execute_query(WELL_ENTHID_QUERY)
    groups: dict[str, list[int]] = {}
    if not df.empty:
        for _, row in df.iterrows():
            raw_name = row.get("well_name")
            if raw_name is None or (isinstance(raw_name, float) and pd.isna(raw_name)):
                continue
            normalized = _normalize_well_name(str(raw_name).strip())
            groups.setdefault(normalized, []).append(int(row["enthid"]))

    _enthid_cache["value"] = groups
    _enthid_cache["expires_at"] = now + _CACHE_TTL_SECONDS
    return groups


def well_enthid_map(force_refresh: bool = False) -> dict[str, int]:
    """Cached {normalized_well_name: enthid} for producing wells.

    Built from mpu.wells.vw_well_header (well_type='prod'), keyed by the
    canonical GUI name (`well_test_client._normalize_well_name`). Names that
    resolve to more than one enthid (a data-quality issue, not expected in
    practice) are OMITTED here rather than picking one silently -- callers
    that need the explicit 0-vs-multiple distinction (i.e. push_prop) use
    `_resolve_enthid`, which raises instead of dropping the entry.
    """
    groups = _fetch_enthid_groups(force_refresh=force_refresh)
    return {name: ids[0] for name, ids in groups.items() if len(ids) == 1}


def _resolve_enthid(well_name: str) -> int:
    """Resolve well_name -> enthid via `well_enthid_map`'s grouping.

    Tries `well_name` as-is first -- the canonical GUI form used everywhere
    else in the app (e.g. 'MPB-28', the output of
    `well_test_client._normalize_well_name`) -- and only falls back to
    `_normalize_well_name` (which expects the raw Databricks form, e.g.
    'B-028') if that direct lookup misses. Normalizing unconditionally would
    corrupt an already-normalized single-digit well number:
    `_normalize_well_name` strips ONE leading zero, so re-applying it to
    'MPB-01' yields 'MPB-1', not 'MPB-01' -- i.e. it is a one-way DB->GUI
    conversion, not idempotent on GUI input.

    Raises EnthidResolutionError on zero or multiple matches (DART's
    `_resolve_enthid` guard, ported to the cached bulk map instead of a
    live per-well query).
    """
    groups = _fetch_enthid_groups()
    matches = groups.get(well_name)
    if matches is None:
        matches = groups.get(_normalize_well_name(well_name), [])

    if len(matches) == 0:
        raise EnthidResolutionError(
            f"No enthid found for well '{well_name}' in mpu.wells.vw_well_header."
        )
    if len(matches) > 1:
        raise EnthidResolutionError(
            f"Multiple enthids found for well '{well_name}': {sorted(matches)}."
        )
    return matches[0]


# Optional per-call identity provider, registered by the GUI at startup
# (`set_entry_user_provider`). Exists for Databricks Apps, where every session
# shares one container and `current_user()` resolves to the app's SERVICE
# PRINCIPAL — without this, every hosted save is stamped with the same UUID
# and the entry_user audit trail is lost. A provider (e.g. reading the
# X-Forwarded-Email request header via st.context) recovers the real engineer
# PER SESSION, which an env var cannot (it's process-global). This module
# stays Streamlit-free: the provider is injected, never imported.
_entry_user_provider = None


def set_entry_user_provider(provider) -> None:
    """Register a zero-arg callable returning the acting user (or None).

    Called on EVERY resolve (never cached — identity is per-session on a
    shared host). Exceptions and falsy returns fall through to the next
    precedence tier, so a broken provider can never block a save.
    """
    global _entry_user_provider
    _entry_user_provider = provider


def resolve_entry_user(force_refresh: bool = False) -> str:
    """Resolve the identity to stamp on prop_hist writes.

    Precedence:
    1. `WOFFL_ENTRY_USER` env override, if set -- checked on every call,
       never cached, so a test/session override always wins.
    2. The registered entry-user provider (the hosted app registers one that
       reads the forwarded-user request header) -- per call, never cached.
    3. The SQL session's `SELECT current_user()`, cached per process
       (it doesn't change mid-session). On Databricks Apps this is the
       service principal -- the fallback of last resort.

    Deliberately NEVER `os.getlogin()` -- wrong identity on Databricks Apps,
    where every user runs as the service principal / container user (see the
    plan's DART review: "DO NOT ADOPT").
    """
    env_user = os.environ.get("WOFFL_ENTRY_USER")
    if env_user:
        return env_user

    if _entry_user_provider is not None:
        try:
            provided = _entry_user_provider()
        except Exception:
            provided = None
        if provided:
            return str(provided)

    if not force_refresh and _entry_user_cache["value"] is not None:
        return _entry_user_cache["value"]

    df = execute_query(CURRENT_USER_QUERY)
    if df.empty:
        raise PropHistError("SELECT current_user() returned no rows.")
    user = str(df["current_user"].iloc[0])
    _entry_user_cache["value"] = user
    return user


def push_prop(
    well_name: str,
    prop_id: str,
    value: Optional[float],
    entry_user: str,
    entry_datetime: Optional[datetime] = None,
) -> int:
    """Insert one row into mpu.wells.prop_hist.

    DART pattern, ported: whitelist prop_id against `fetch_prop_xref()`,
    resolve well_name to an enthid via `_resolve_enthid` (raises on 0 or >1
    matches), then a parameterized INSERT (entry_datetime=now, UTC,
    timezone-aware) through `databricks_client.execute_write` -- which
    enforces the ALLOW_DATABRICKS_WRITES gate and the INSERT-only/no-chaining
    guard.

    Args:
        well_name: any well-name spelling the app uses (GUI 'MPB-28' or DB
            'B-028') -- normalized internally.
        prop_id: must be a valid key in mpu.wells.prop_xref, and must not be
            an as-built completion fact (``AS_BUILT_PROP_IDS``).
        value: numeric prop_value, or ``None`` to write a SQL NULL. ``None``
            is the un-pin/"no value" marker (see `ipr_anchor.clear_ipr_pin`)
            -- NEVER a negative sentinel, since real values (e.g. `wt_uid`)
            can themselves be negative. Non-``None`` values are coerced to
            `float` and must be finite (raises `PropHistError` on NaN/inf).
        entry_user: identity to stamp -- callers pass `resolve_entry_user()`
            (kept explicit here rather than defaulted, so a push's identity
            is always visible at the call site).
        entry_datetime: UTC stamp to write. Defaults to ``now``, which is what
            a lone push wants. A caller writing several props as ONE logical
            save (e.g. ``ipr_anchor.save_ipr_values``, up to nine rows) should
            generate one stamp and pass it to every push: that shared value is
            the only batch identity prop_hist has, and it's what
            ``mpu.wells.woffl_eng_comment`` joins on to attach the engineer's
            comment to the save. Verified to round-trip to the microsecond.

    Raises:
        AsBuiltPropError: prop_id is an as-built completion fact. woffl reads
            those; the data team writes them.

    Returns:
        Rowcount from execute_write. The Databricks connector reports ``-1``
        for INSERT rather than an affected-row count, so treat any non-raising
        return as success — do NOT assert ``== 1``.
    """
    prop_value = _validated_prop_value(prop_id, value)
    enthid = _resolve_enthid(well_name)

    parameters = {
        "enthid": enthid,
        "prop_id": prop_id,
        "prop_value": prop_value,
        # An explicit stamp is a caller's BATCH identity — used verbatim, never
        # re-allocated, or the save's rows would stop sharing a key.
        "entry_datetime": entry_datetime or next_entry_datetime(),
        "entry_user": entry_user,
    }
    return execute_write(PROP_HIST_INSERT_SQL, parameters)


def _validated_prop_value(prop_id: str, value: Optional[float]) -> float:
    """Whitelist `prop_id` and coerce `value`, or raise. Shared by both write
    entry points so a batched save can never be validated more loosely than a
    lone push. `fetch_prop_xref` is TTL-cached, so calling this per row costs
    one dict lookup after the first; the as-built check stays ahead of it, so
    a forbidden prop_id is refused without touching the warehouse."""
    if prop_id in AS_BUILT_PROP_IDS:
        raise AsBuiltPropError(
            f"prop_id '{prop_id}' is an as-built completion property — woffl "
            "reads it from vw_prop_mech and must never write it. Solver output "
            "and UI defaults are not measurements; correct the well file "
            "instead."
        )
    valid_ids = fetch_prop_xref()
    if prop_id not in valid_ids:
        raise UnknownPropIdError(
            f"prop_id '{prop_id}' is not in mpu.wells.prop_xref. "
            f"Valid keys: {sorted(valid_ids)}"
        )
    if value is None:
        # prop_value is DOUBLE **NOT NULL**. A None here reaches Databricks and
        # comes back as DELTA_NOT_NULL_CONSTRAINT_VIOLATED after a full round
        # trip — and callers that wrapped the push in a broad `except` turned
        # that into a silent no-op. The 🗑 Clear-saved-IPR button and the 🔒
        # lock checkboxes both did exactly that, for months, with zero NULL
        # rows ever written (found 2026-08-04). Fail here instead, naming the
        # convention: "cleared" is an explicit sentinel chosen per prop_id
        # (see ipr_anchor.PIN_CLEARED_VALUE / LOCK_UNLOCKED_VALUE).
        raise PropHistError(
            f"prop_value is NOT NULL in mpu.wells.prop_hist, so prop_id "
            f"'{prop_id}' cannot be cleared by writing None. Push an explicit "
            "sentinel the readers understand instead."
        )
    prop_value = float(value)
    if not math.isfinite(prop_value):
        raise PropHistError(
            f"prop_value must be finite (got {prop_value!r})."
        )
    return prop_value


def push_props(
    well_name: str,
    values: dict[str, Optional[float]],
    entry_user: str,
    entry_datetime: Optional[datetime] = None,
) -> int:
    """Insert SEVERAL prop_hist rows for one well in ONE statement.

    Same guards as :func:`push_prop`, applied to every row BEFORE anything is
    sent: as-built rejection, prop_xref whitelist, NOT-NULL refusal, finite
    coercion, single-enthid resolution. Validation is all-or-nothing, which is
    the point of batching — prop_hist has no transaction, so the old per-prop
    loop could leave a save half-written when row four was rejected.

    Args:
        well_name: any spelling the app uses; normalized internally.
        values: ``{prop_id: value}``. Insertion order is preserved (it only
            affects the SQL text; readers resolve by ``entry_datetime``).
        entry_user: identity to stamp on every row.
        entry_datetime: the save's BATCH stamp, shared by every row and by the
            engineer's comment. Defaults to one freshly allocated stamp for the
            whole batch — never one per row.

    Returns:
        The number of rows sent (``execute_write`` reports ``-1`` for INSERT,
        so the rowcount is useless to a caller that needs a count).
    """
    if not values:
        return 0
    if len(values) > _MAX_BATCH_ROWS:
        raise PropHistError(
            f"push_props takes at most {_MAX_BATCH_ROWS} rows per statement "
            f"(got {len(values)})."
        )

    rows = [
        (prop_id, _validated_prop_value(prop_id, value))
        for prop_id, value in values.items()
    ]
    enthid = _resolve_enthid(well_name)
    stamp = entry_datetime or next_entry_datetime()

    tuples: list[str] = []
    parameters: dict = {}
    for i, (prop_id, prop_value) in enumerate(rows):
        parameters[f"enthid_{i}"] = enthid
        parameters[f"prop_id_{i}"] = prop_id
        parameters[f"prop_value_{i}"] = prop_value
        parameters[f"entry_datetime_{i}"] = stamp
        parameters[f"entry_user_{i}"] = entry_user
        tuples.append(
            f"(:enthid_{i}, :prop_id_{i}, :prop_value_{i}, "
            f":entry_datetime_{i}, :entry_user_{i})"
        )

    execute_write(PROP_HIST_INSERT_HEAD + ", ".join(tuples), parameters)
    return len(rows)


# ── deleting rows: the sanctioned escape hatch, approval-gated ──────────────
#
# prop_hist is append-only BY DEFAULT and that is the right default: the trail
# is how the 2026-08-03 as-built incident was found at all. But append cannot
# undo everything — when the app wrote rows that should never have existed,
# Scott's call (2026-08-04) was to remove them outright rather than bury them
# under a correction, and he has MODIFY on the table.
#
# So deletion lives here, next to push_prop, instead of being rediscovered and
# rewritten as a throwaway script every time. It is deliberately awkward:
#
#   1. ALLOW_PROP_HIST_DELETE must be set — SEPARATE from
#      ALLOW_DATABRICKS_WRITES, so turning normal saves on never turns deletes
#      on. Set it for one invocation; never in .env, app.yaml, or a Databricks
#      App config.
#   2. ALLOW_DATABRICKS_WRITES must ALSO be set (the existing write gate).
#   3. `apply=True` must be passed; the default is a dry run.
#   4. `expect` must equal the manifest length — a caller states the row count
#      up front and a mismatch aborts, so a broadened manifest can't quietly
#      take more than intended.
#   5. Every row is named EXPLICITLY (well, prop_id, prop_value,
#      entry_datetime) and must match exactly one live row. There is no
#      predicate/bulk form: no "delete everything since <date>".
#   6. `reason` is required and echoed into the report.
#
# NO GUI PATH MAY CALL THIS — enforced by tests/test_prop_hist_delete.py, which
# greps woffl/gui for the symbol. Deleting well data is a deliberate act at a
# console with a human reading the dry run, not a button.
#
# Unlike push_prop, as-built prop_ids are ALLOWED here: removing bad as-built
# rows is precisely what this exists for. push_prop still refuses to author
# them, which is the asymmetry we want — woffl can retract its own bad writes,
# never originate one.

DELETE_GATE_ENV = "ALLOW_PROP_HIST_DELETE"

PROP_HIST_DELETE_SQL = (
    "DELETE FROM mpu.wells.prop_hist WHERE enthid = :enthid "
    "AND prop_id = :prop_id AND prop_value = :prop_value "
    "AND entry_datetime = :entry_datetime"
)


class DeleteNotApprovedError(PropHistError):
    """Raised when a delete is attempted without the explicit approval gate,
    the row-count assertion, or a reason."""


@dataclass(frozen=True)
class DeleteTarget:
    """Exactly one prop_hist row, named in full.

    ``entry_datetime`` is the microsecond-precision stamp from the row itself
    (read it back from prop_hist — do NOT reconstruct it). Note that Kaelin's
    DART ``delete_prop`` matches on an ``entry_date`` DATE column; that column
    was migrated to ``entry_datetime TIMESTAMP`` on 2026-07-08 and no longer
    exists, so that predicate raises column-not-found against this table.
    """

    well_name: str
    prop_id: str
    prop_value: float
    entry_datetime: datetime


def delete_gate_enabled() -> bool:
    """Whether the delete-specific approval gate is set for this process."""
    return os.environ.get(DELETE_GATE_ENV, "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _resolve_targets(targets: Sequence[DeleteTarget]) -> list[dict]:
    """Manifest -> bind-parameter dicts, with every row validated.

    Raises before touching anything if a prop_id isn't a prop_xref key, a well
    doesn't resolve to exactly one enthid, or a row doesn't match exactly one
    live prop_hist row (0 = already gone or a wrong stamp; >1 = the manifest is
    not specific enough and a blind delete would take a row nobody reviewed).
    """
    valid = fetch_prop_xref()
    resolved = []
    for t in targets:
        if t.prop_id not in valid:
            raise UnknownPropIdError(
                f"prop_id '{t.prop_id}' is not in mpu.wells.prop_xref."
            )
        params = {
            "enthid": _resolve_enthid(t.well_name),
            "prop_id": t.prop_id,
            "prop_value": float(t.prop_value),
            "entry_datetime": t.entry_datetime,
        }
        n = int(
            execute_query(
                "SELECT count(*) AS n FROM mpu.wells.prop_hist "
                f"WHERE enthid = {params['enthid']} "
                f"AND prop_id = '{t.prop_id}' "
                f"AND prop_value = {params['prop_value']!r} "
                f"AND entry_datetime = '{t.entry_datetime.isoformat()}'"
            )["n"].iloc[0]
        )
        if n != 1:
            raise DeleteNotApprovedError(
                f"{t.well_name}/{t.prop_id} at {t.entry_datetime.isoformat()} "
                f"matches {n} rows, expected exactly 1 — refusing the batch."
            )
        resolved.append(params)
    return resolved


def delete_props(
    targets: Sequence[DeleteTarget],
    *,
    reason: str,
    expect: int,
    apply: bool = False,
) -> dict:
    """Delete explicitly-named prop_hist rows. Dry run unless ``apply=True``.

    THIS IS NOT PART OF ANY APP FLOW. See the block comment above for the
    approval model. Recipe (the 2026-08-03 as-built cleanup, verbatim):

        from woffl.assembly.audit_as_built_writes import (
            fetch_history, find_overwrites)
        hits = find_overwrites(fetch_history(), since)   # what went wrong
        targets = [DeleteTarget(r.well_name, r.prop_id,
                                r.current_value, r.current_at)
                   for _, r in hits.iterrows()]
        delete_props(targets, reason="...", expect=len(targets))          # dry
        delete_props(targets, reason="...", expect=len(targets), apply=True)

    Run the dry pass first and READ IT. Then, for the real pass only:
        ALLOW_DATABRICKS_WRITES=true ALLOW_PROP_HIST_DELETE=true python ...

    Args:
        targets: the rows to remove, each named in full.
        reason: why — required, non-empty, echoed into the report.
        expect: must equal ``len(targets)``; the caller's own count assertion.
        apply: False (default) validates and reports, touching nothing.

    Returns:
        ``{"planned", "deleted", "applied", "reason", "version_before",
        "version_after", "undo"}``. ``undo`` is a ready ``RESTORE TABLE``
        statement — Delta time travel is the only way back, so it is surfaced
        rather than left to be looked up under pressure.

    Raises:
        DeleteNotApprovedError: a gate, the count assertion, or the reason is
            missing, or a target doesn't match exactly one row.
        WritesDisabledError: ALLOW_DATABRICKS_WRITES is unset.
    """
    if not (reason or "").strip():
        raise DeleteNotApprovedError("A non-empty `reason` is required.")
    if expect != len(targets):
        raise DeleteNotApprovedError(
            f"expect={expect} but the manifest has {len(targets)} row(s) — "
            "refusing. State the count you intend to delete."
        )

    resolved = _resolve_targets(targets)
    version_before = _table_version()
    report = {
        "planned": len(resolved),
        "deleted": 0,
        "applied": False,
        "reason": reason.strip(),
        "version_before": version_before,
        "version_after": version_before,
        "undo": None,
    }
    if not apply:
        return report

    if not delete_gate_enabled():
        raise DeleteNotApprovedError(
            f"{DELETE_GATE_ENV} is not set. Deleting prop_hist rows needs "
            "explicit approval, separate from the normal write gate."
        )
    from woffl.assembly.databricks_client import _write_gate_enabled

    if not _write_gate_enabled():
        raise WritesDisabledError(
            "Databricks writes are disabled. Set ALLOW_DATABRICKS_WRITES=true."
        )

    deleted = 0
    for params in resolved:
        def _run(cursor, _p=params):
            # execute_write is INSERT-only by design (_validate_single_insert),
            # so a delete goes to the connector directly. Still parameterized.
            cursor.execute(PROP_HIST_DELETE_SQL, _p)
            return cursor.rowcount

        _execute_via_connector(_run)  # connector reports -1 for DML; verify below
        deleted += 1

    version_after = _table_version()
    report.update(
        applied=True,
        deleted=deleted,
        version_after=version_after,
        undo=(
            "RESTORE TABLE mpu.wells.prop_hist TO VERSION AS OF "
            f"{version_before};"
        ),
    )
    return report


def _table_version() -> int:
    """Current Delta version of prop_hist — the handle for a time-travel undo."""
    return int(
        execute_query(
            "SELECT max(version) AS v FROM (DESCRIBE HISTORY mpu.wells.prop_hist)"
        )["v"].iloc[0]
    )


_MAX_COMMENT_CHARS = 500


def push_eng_comment(
    well_name: str,
    entry_datetime: datetime,
    entry_user: str,
    comment_text: str,
    context: str = "ipr_save",
) -> int:
    """Attach one engineer comment to a prop_hist save batch.

    ``entry_datetime`` MUST be the same stamp handed to every ``push_prop`` of
    that save — it is the join key, and prop_hist has no other batch identity.
    Two saves for one well on the same day therefore keep separate comments:
    the grain is the timestamp, not the date.

    Call this AFTER the property pushes have succeeded. A comment describing
    edits that never landed is worse than no comment, so the ordering is not
    incidental.

    Raises `PropHistError` on empty text, and whatever `_resolve_enthid` /
    `execute_write` raise (unknown well, writes disabled). Callers treat it as
    best-effort and must not let a comment failure undo a successful save.
    """
    text = (comment_text or "").strip()
    if not text:
        raise PropHistError("comment_text is empty — nothing to attach.")
    if len(text) > _MAX_COMMENT_CHARS:
        text = text[:_MAX_COMMENT_CHARS].rstrip()

    return execute_write(
        ENG_COMMENT_INSERT_SQL,
        {
            "enthid": _resolve_enthid(well_name),
            "entry_datetime": entry_datetime,
            "entry_user": entry_user,
            "context": context,
            "comment_text": text,
        },
    )


def fetch_eng_comments(well_name: str) -> pd.DataFrame:
    """Every engineer comment for a well, newest first.

    Returns columns ``entry_datetime, entry_user, context, comment_text``;
    an EMPTY frame with those columns when the well has none, so callers can
    merge without a None check.

    ``enthid`` is int-coerced into the SQL text because `execute_query` has no
    bind support — same guard every other read in this module uses.
    """
    cols = ["entry_datetime", "entry_user", "context", "comment_text"]
    enthid = _resolve_enthid(well_name)
    df = execute_query(
        "SELECT entry_datetime, entry_user, context, comment_text "
        f"FROM mpu.wells.woffl_eng_comment WHERE enthid = {int(enthid)} "
        "ORDER BY entry_datetime DESC"
    )
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    # A retried write can land the identical row twice (execute_write makes two
    # attempts); collapse exact duplicates rather than showing the note twice.
    return df[cols].drop_duplicates()


def fetch_latest_prop(
    well_name: str, prop_id: str
) -> Optional[Tuple[Optional[float], object, str]]:
    """Latest (prop_value, entry_datetime, entry_user) for (well_name, prop_id).

    Reads mpu.wells.prop_hist directly (not a pivot view -- `ipr_wt_uid`
    isn't pivoted into vw_prop_mech/vw_prop_resvr). Returns None when there
    is no row yet for this well+prop.

    ``entry_datetime`` orders deterministically -- including same-day rows,
    since the column is a full timestamp (not a bare date) -- so two pushes
    on the same calendar day resolve to the genuinely later one rather than
    an arbitrary tie-break.

    ``prop_value`` in the returned tuple is ``None`` when the latest row's
    prop_value is SQL NULL (the un-pin/"no value" marker -- see
    `ipr_anchor.clear_ipr_pin`) or NaN (however the connector represents a
    NULL numeric column), else a `float`. Callers must treat `None` as "no
    value" and must NOT apply any sign-based rule -- real prop_value data
    (e.g. `wt_uid`) can be negative.

    prop_id is validated for SQL-safe shape (not whitelist-checked against
    prop_xref -- an unrecognized-but-shape-safe prop_id just reads back
    zero rows) before being spliced into the query text, since
    `execute_query` (read path) has no native parameter binding -- only
    `execute_write` does.
    """
    if not isinstance(prop_id, str) or not _PROP_ID_SHAPE_RE.match(prop_id):
        raise UnknownPropIdError(f"Invalid prop_id shape for SQL: {prop_id!r}")

    enthid = _resolve_enthid(well_name)

    query = (
        "SELECT prop_value, entry_datetime, entry_user "
        "FROM mpu.wells.prop_hist "
        f"WHERE enthid = {enthid} AND prop_id = '{prop_id}' "
        "ORDER BY entry_datetime DESC LIMIT 1"
    )
    df = execute_query(query)
    if df.empty:
        return None

    # Defensive re-sort in Python: correct even if the caller's mock (or a
    # future connector quirk) hands back more than one row despite the
    # LIMIT 1 above. entry_datetime is a full timestamp, so this also
    # resolves same-day rows to the genuinely later one (deterministic,
    # unlike the old date-only ordering).
    df = df.sort_values("entry_datetime", ascending=False)
    row = df.iloc[0]

    raw_value = row["prop_value"]
    if raw_value is None or pd.isna(raw_value):
        value: Optional[float] = None
    else:
        value = float(raw_value)

    return value, row["entry_datetime"], str(row["entry_user"])
