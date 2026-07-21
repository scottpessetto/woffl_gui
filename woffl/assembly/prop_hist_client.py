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
import time
from datetime import datetime, timezone
from typing import Optional, Tuple

import pandas as pd

from woffl.assembly.databricks_client import execute_query, execute_write
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

CURRENT_USER_QUERY = "SELECT current_user() AS current_user"

_CACHE_TTL_SECONDS = 3600.0

# Module-level TTL caches -- deliberately NOT st.cache_data (this module must
# work without Streamlit, e.g. from a plain script or a pytest process).
_xref_cache: dict = {"value": None, "expires_at": 0.0}
_enthid_cache: dict = {"value": None, "expires_at": 0.0}  # {name: [enthid, ...]}
_entry_user_cache: dict = {"value": None}

_PROP_ID_SHAPE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


class PropHistError(ValueError):
    """Base error for prop_hist_client operations."""


class UnknownPropIdError(PropHistError):
    """Raised when prop_id isn't a valid key in mpu.wells.prop_xref (or is
    shaped unsafely for SQL interpolation in the read path)."""


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
    else in the app (e.g. 'MPB-28', per well_test_client.fetch_single_well_tests
    and friends) -- and only falls back to `_normalize_well_name` (which
    expects the raw Databricks form, e.g. 'B-028') if that direct lookup
    misses. Normalizing unconditionally would corrupt an already-normalized
    single-digit well number: `_normalize_well_name` strips ONE leading
    zero, so re-applying it to 'MPB-01' yields 'MPB-1', not 'MPB-01' -- i.e.
    it is a one-way DB->GUI conversion, not idempotent on GUI input.

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


def resolve_entry_user(force_refresh: bool = False) -> str:
    """Resolve the identity to stamp on prop_hist writes.

    Precedence:
    1. `WOFFL_ENTRY_USER` env override, if set (e.g. once the deployed app
       threads through a real Streamlit-user identity) -- checked on every
       call, never cached, so a test/session override always wins.
    2. The local SQL session's `SELECT current_user()`, cached per process
       (it doesn't change mid-session).

    Deliberately NEVER `os.getlogin()` -- wrong identity on Databricks Apps,
    where every user runs as the service principal / container user (see the
    plan's DART review: "DO NOT ADOPT").
    """
    env_user = os.environ.get("WOFFL_ENTRY_USER")
    if env_user:
        return env_user

    if not force_refresh and _entry_user_cache["value"] is not None:
        return _entry_user_cache["value"]

    df = execute_query(CURRENT_USER_QUERY)
    if df.empty:
        raise PropHistError("SELECT current_user() returned no rows.")
    user = str(df["current_user"].iloc[0])
    _entry_user_cache["value"] = user
    return user


def push_prop(
    well_name: str, prop_id: str, value: Optional[float], entry_user: str
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
        prop_id: must be a valid key in mpu.wells.prop_xref.
        value: numeric prop_value, or ``None`` to write a SQL NULL. ``None``
            is the un-pin/"no value" marker (see `ipr_anchor.clear_ipr_pin`)
            -- NEVER a negative sentinel, since real values (e.g. `wt_uid`)
            can themselves be negative. Non-``None`` values are coerced to
            `float` and must be finite (raises `PropHistError` on NaN/inf).
        entry_user: identity to stamp -- callers pass `resolve_entry_user()`
            (kept explicit here rather than defaulted, so a push's identity
            is always visible at the call site).

    Returns:
        Rowcount from execute_write (1 on a normal single-row insert).
    """
    valid_ids = fetch_prop_xref()
    if prop_id not in valid_ids:
        raise UnknownPropIdError(
            f"prop_id '{prop_id}' is not in mpu.wells.prop_xref. "
            f"Valid keys: {sorted(valid_ids)}"
        )

    enthid = _resolve_enthid(well_name)

    if value is None:
        prop_value: Optional[float] = None
    else:
        prop_value = float(value)
        if not math.isfinite(prop_value):
            raise PropHistError(
                f"prop_value must be finite (got {prop_value!r}) or None for NULL."
            )

    parameters = {
        "enthid": enthid,
        "prop_id": prop_id,
        "prop_value": prop_value,
        "entry_datetime": datetime.now(timezone.utc),
        "entry_user": entry_user,
    }
    return execute_write(PROP_HIST_INSERT_SQL, parameters)


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
