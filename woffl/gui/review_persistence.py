"""Review persistence — property write-through into ``mpu.wells.prop_hist``.

Kaelin's design, chosen by Scott 2026-07-30 over a CSV draft and a new-table
draft: review state is timestamped property rows in the existing append-only
EAV history table, latest (enthid, prop_id) by ``entry_datetime`` wins. The
goal, in Scott's words: *"when a user opens a well in single review or
optimization it has the latest edit so that it keeps the curve and rate the
engineer saw fit."*

HOW THE LOOP CLOSES (this module is only the pad-review WRITE half)
-------------------------------------------------------------------
* **Write, pad review**: one :func:`sync_pad` call at the top of the pad/CFP
  pages pushes CHANGED properties of saved review entries (the store only
  mutates on deliberate actions, so this runs at review cadence).
* **Write, single well**: the Solver's "📌 Save IPR as well default" pushes the
  anchor pin AND the sidebar's current values (``ipr_anchor.save_ipr_values``).
* **Read**: there is deliberately NO store-restore here. Values flow back on
  well-open through the app's NORMAL paths — canonical ids via the
  ``vw_prop_mech``/``vw_prop_resvr`` pivots, and the saved IPR curve via
  ``sidebar._seed_saved_ipr`` (latest-timestamp-wins against the anchor pin).

WHAT PERSISTS (Scott's selection, 2026-07-30)
---------------------------------------------
Canonical characterization ids that already existed, plus the five IPR ids he
self-added to ``prop_xref`` the same day (``ipr_qwf_liq``, ``ipr_pwf``,
``form_wc``, ``form_gor``, ``surf_press``). Deliberately NOT persisted, per
Scott: the reviewed pump (``jp_nozzle``/``jp_throat_ratio`` — pump truth stays
with the JP tracker), ``well_reviewed``/``well_offline`` (workflow state, not
well characteristics), direction / field model / PF pin (live-detected or
derivable on open).

Also NOT persisted, as of 2026-08-03: the as-built completion dimensions
(``jpump_md``, ``casing_out_dia``, ``tubing_out_dia`` — see
``prop_hist_client.AS_BUILT_PROP_IDS``). Reviews carry model values; the pump
depth and pipe sizes are measurements woffl reads and never authors. They were
in this map for four days and overwrote eight wells' measured depths with
interpolated TVD and their casing OD with the 6.875 fallback.

WRITE DISCIPLINE
----------------
* Per-(well, prop) baseline BULK-READ from prop_hist at first sync — only
  values that genuinely differ from the stored latest are pushed; re-saving an
  unchanged well writes nothing.
* NEVER push NULL — review saves assert values; un-setting canonical data is
  not this module's business (a NULL ``resvr_press`` would blank the pivots).
* Hypothetical wells have no enthid → session + manual CSV only.
* Every write goes through ``prop_hist_client.push_prop`` (xref-validated,
  gate-checked, parameterized, ``resolve_entry_user()``-stamped). Fail-soft
  throughout — persistence may never break a page.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Optional

import streamlit as st


def _enc_float(v):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(f) or math.isinf(f)) else f


class Field:
    """One persistable store field → prop_id. ``canon`` marks ids serving the
    vw_prop_mech/vw_prop_resvr pivots (canonical characterization)."""

    def __init__(self, store_key: str, prop_id: str, canon: bool,
                 enc: Callable = _enc_float):
        self.store_key = store_key
        self.prop_id = prop_id
        self.canon = canon
        self.enc = enc


def _enc_qwf_liq(entry_qwf):
    # The store's qwf is ALREADY total liquid (store_to_well_configs feeds the
    # optimizer with it) — no conversion here, unlike the sidebar's oil-based
    # qwf handled by ipr_anchor.save_ipr_values.
    return _enc_float(entry_qwf)


FIELD_MAP: list = [
    # canonical characterization (existing ids; a saved review IS the well's
    # latest characterization through the pivots)
    Field("res_pres", "resvr_press", True),
    Field("form_temp", "resvr_temp", True),
    Field("bubble_point", "resvr_bubb", True),
    Field("oil_api", "form_oil_api", True),
    Field("gas_sg", "form_gas_sg", True),
    Field("wat_sg", "form_wat_sg", True),
    Field("ken_well", "jpfric_entry", True),
    Field("kth_well", "jpfric_throat", True),
    Field("kdi_well", "jpfric_diffuser", True),
    Field("knz_well", "jpfric_nozzle", True),
    # the saved IPR curve + rate (added to prop_xref 2026-07-30)
    Field("qwf", "ipr_qwf_liq", False, enc=_enc_qwf_liq),
    Field("pwf", "ipr_pwf", False),
    Field("form_wc", "form_wc", False),
    Field("form_gor", "form_gor", False),
    Field("surf_pres", "surf_press", False),
]

# NOT persisted, deliberately: jpump_md / casing_out_dia / tubing_out_dia and
# friends (``prop_hist_client.AS_BUILT_PROP_IDS``). Everything above is a
# reviewed MODEL value — a pressure, a PVT property, a friction coefficient,
# an IPR anchor — that the engineer decided and the pivots should follow. The
# as-built dimensions are the opposite: measurements off the wellbore diagram
# that woffl only reads. A store entry always carries SOME number for them
# (a Databricks value when present, a UI/force-fit default otherwise), so
# write-through could not tell "the engineer changed the pump depth" from
# "no row existed and 6.875 got substituted" — and it pushed the substitute.
# See the 2026-08-03 incident note on AS_BUILT_PROP_IDS. push_prop rejects
# these ids outright now; keeping them out of FIELD_MAP is the first gate.

_BASELINE_KEY = "_rp_baseline_{pad}"
_SYNCED_FLAG = "_rp_synced_{pad}"
_XREF_CACHE = {"ids": None, "at": 0.0}
_XREF_TTL = 3600.0


def available_prop_ids(force: bool = False) -> set:
    """Live prop_xref ids, TTL-cached; empty set on failure (fail-soft)."""
    now = time.monotonic()
    if (
        not force
        and _XREF_CACHE["ids"] is not None
        and now - _XREF_CACHE["at"] < _XREF_TTL
    ):
        return _XREF_CACHE["ids"]
    try:
        from woffl.assembly.databricks_client import execute_query

        df = execute_query("SELECT prop_id FROM mpu.wells.prop_xref")
        ids = {str(p).strip() for p in df["prop_id"].dropna()}
    except Exception:
        ids = _XREF_CACHE["ids"] or set()
    _XREF_CACHE["ids"] = ids
    _XREF_CACHE["at"] = now
    return ids


def active_fields() -> list:
    ids = available_prop_ids()
    return [f for f in FIELD_MAP if f.prop_id in ids]


def writes_enabled() -> bool:
    from woffl.gui.ipr_anchor import writes_enabled as _we

    return _we()


def encode_entry(entry: dict, fields: Optional[list] = None) -> dict:
    """{prop_id: float|None} for one store entry over the active fields."""
    out = {}
    for f in fields if fields is not None else active_fields():
        out[f.prop_id] = f.enc(entry.get(f.store_key))
    return out


def _latest_props(prop_ids, pad: Optional[str] = None) -> dict:
    """Bulk latest-per-(well, prop) read → {well: {prop_id: value|None}}."""
    ids = sorted({str(p) for p in prop_ids})
    if not ids:
        return {}
    from woffl.assembly.databricks_client import execute_query
    from woffl.assembly.well_test_client import _normalize_well_name

    id_list = ",".join(f"'{i}'" for i in ids)  # ids come from OUR whitelist
    df = execute_query(
        f"""
        SELECT h.well_name, p.prop_id, p.prop_value FROM (
            SELECT enthid, prop_id, prop_value,
                   ROW_NUMBER() OVER (
                       PARTITION BY enthid, prop_id ORDER BY entry_datetime DESC
                   ) AS rn
            FROM mpu.wells.prop_hist
            WHERE prop_id IN ({id_list})
        ) p
        JOIN mpu.wells.vw_well_header h ON p.enthid = h.enthid
        WHERE p.rn = 1
        """
    )
    out: dict = {}
    if df is None or df.empty:
        return out
    for _, r in df.iterrows():
        well = _normalize_well_name(str(r["well_name"]).strip())
        if pad and not str(well).replace("MP", "", 1).startswith(str(pad).upper()):
            continue
        v = r["prop_value"]
        try:
            v = (
                None
                if v is None or (isinstance(v, float) and math.isnan(v))
                else float(v)
            )
        except (TypeError, ValueError):
            v = None
        out.setdefault(well, {})[str(r["prop_id"])] = v
    return out


def _push(well: str, prop_id: str, value: float) -> None:
    from woffl.assembly.prop_hist_client import push_prop, resolve_entry_user

    push_prop(well, prop_id, value, resolve_entry_user())


def wrs_store(pad: str) -> dict:
    """The live session store dict (same object ``store_for`` hands the pages)."""
    return st.session_state.setdefault(f"sp_well_store_{pad}", {})


def sync_pad(pad: str) -> dict:
    """Write-through for one pad; call at the top of the page each rerun.

    First call seeds the per-(well, prop) baseline from prop_hist's stored
    latest. Later calls push only genuinely changed values.

    Returns {"saved": n, "disabled": bool, "error": str|None,
    "skipped_hypotheticals": n}.
    """
    out = {"saved": 0, "disabled": False, "error": None, "skipped_hypotheticals": 0}
    try:
        store = wrs_store(pad)
        flag = _SYNCED_FLAG.format(pad=pad)
        bkey = _BASELINE_KEY.format(pad=pad)
        fields = active_fields()

        if not st.session_state.get(flag):
            st.session_state[flag] = True
            latest = _latest_props([f.prop_id for f in fields], pad=pad)
            st.session_state[bkey] = {
                (w, pid): v for w, props in latest.items() for pid, v in props.items()
            }
            return out

        if not writes_enabled():
            out["disabled"] = True
            return out

        baseline = st.session_state.get(bkey) or {}
        n_pushed = 0
        for well, entry in store.items():
            if entry.get("is_hypothetical"):
                out["skipped_hypotheticals"] += 1
                continue  # no enthid — cannot live in prop_hist
            for pid, val in encode_entry(entry, fields).items():
                if val is None:
                    continue  # never push NULL (canon safety)
                prev = baseline.get((well, pid))
                if prev is not None and abs(prev - val) < 1e-6:
                    continue
                _push(well, pid, val)
                baseline[(well, pid)] = val
                n_pushed += 1
        st.session_state[bkey] = baseline
        out["saved"] = n_pushed
    except Exception as e:  # persistence must never break the page
        out["error"] = str(e)
    return out


def render_caption(results: dict) -> None:
    errors = {p: r["error"] for p, r in results.items() if r.get("error")}
    disabled = any(r.get("disabled") for r in results.values())
    n_hypo = sum(r.get("skipped_hypotheticals") or 0 for r in results.values())

    bits = [
        "💾 Saved reviews write property edits to `mpu.wells.prop_hist` "
        "(timestamped, latest wins — wells open with them everywhere)"
    ]
    if disabled:
        bits.append("**writes OFF** (`ALLOW_DATABRICKS_WRITES`)")
    if n_hypo:
        bits.append(f"{n_hypo} hypothetical well(s) session-only (no enthid)")
    st.caption(" — ".join(bits) + ".")
    if errors:
        st.warning(
            "Persistence problem (reviews still work this session, but are NOT "
            "being saved): " + "; ".join(f"{p}: {e}" for p, e in errors.items())
        )
