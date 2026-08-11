"""Suction-response history for the Single Well advanced panel.

Daily (PF pressure, BHP) field scatter for one well - the measured points an
advanced user eyeballs against the model's psu(Ppf) response curve (the
client overlays the curve from the existing POST /api/compute/pf-range; this
module never solves). Productized from scripts/response_eyeball.py.

Data assembly (everything fail-soft except the fleet fetch, which the router
maps to the standard error path):

* days: evidence._fleet_pressure_daily() (365 days of vw_pressure_daily,
  TTL-cached fleet-wide) sliced to the well; a day survives when
  resolve_pf_pressure() yields a real PF header (>= 800 psi built into the
  resolver, <= PPF_MAX_PSI here) and BHP clears the dead-gauge floor.
* era: jp_history current pump Date Set. Days on/after it are "current",
  before it "prior" (a prior pump's response is a different curve - the
  panel greys those points). No tracker -> era_start null, all "current".
* res_pres: saved-fit reservoir pressure when one exists (ipr_anchor.
  load_saved_ipr - memoized per well, warmed fleet-wide at startup, so this
  is the cheapest per-well source of the SAVED value; a locked ResP lock
  outranks saved values, mirroring the sidebar seeding precedence), else the
  characteristics row's res_pres. Used only to flag buildup days
  (BHP >= res_pres is shut-in buildup, not a flowing response point).
* evidence: the mined floor/psu_ref/beta summary for the same well via
  evidence.pad_evidence, so the panel can draw the field-evidence band next
  to the scatter. Absent evidence -> null, never fatal.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

from server.services import datasources, evidence

log = logging.getLogger("woffl.web.response_history")

# Resolved PF above this is not a real header (same guard as evidence.py).
PPF_MAX_PSI = evidence.PPF_MAX_PSI
# Daily BHP at/below this is a dead/glitching gauge (same guard as evidence.py).
BHP_GLITCH_PSI = evidence.BHP_GLITCH_PSI

# Evidence keys surfaced to the panel (subset of the well_evidence dict).
_EVIDENCE_KEYS = ("floor", "psu_ref", "beta", "beta_source", "n_pairs")


# ---------------------------------------------------------------------------
# Pure day assembly (no I/O - the unit-test surface)
# ---------------------------------------------------------------------------


def build_days(
    daily: "pd.DataFrame | list[dict[str, Any]]",
    era_start: Optional[pd.Timestamp] = None,
    res_pres: Optional[float] = None,
) -> list[dict[str, Any]]:
    """Panel day rows from one well's daily pressure rows. PURE - no I/O.

    Args:
        daily: rows with sample_date, tubing_prs, inn_ann_prs, btmhole_prs
            (one row per day - the fleet query's max() aggregation).
        era_start: current pump Date Set; None means no tracker -> every
            surviving day is "current".
        res_pres: reservoir pressure for the buildup flag; None/<=0 means
            unknown -> buildup is False everywhere.

    Returns:
        [{"date", "ppf", "bhp", "era", "buildup"}, ...] sorted by date. A day
        survives only with a valid resolved PF (resolve_pf_pressure enforces
        >= 800 psi; <= PPF_MAX_PSI here) and BHP > BHP_GLITCH_PSI.
    """
    from woffl.assembly.pf_pressure import resolve_pf_pressure

    df = daily if isinstance(daily, pd.DataFrame) else pd.DataFrame(list(daily))
    needed = {"sample_date", "btmhole_prs"}
    if df is None or df.empty or not needed <= set(df.columns):
        return []

    df = df.copy()
    df["sample_date"] = pd.to_datetime(df["sample_date"], errors="coerce")
    df["btmhole_prs"] = pd.to_numeric(df["btmhole_prs"], errors="coerce")
    df = df.dropna(subset=["sample_date", "btmhole_prs"]).sort_values("sample_date")
    df = df[df["btmhole_prs"] > BHP_GLITCH_PSI]

    ppf = [
        resolve_pf_pressure(t, a)[0]
        for t, a in zip(df.get("tubing_prs", pd.Series(index=df.index, dtype=float)),
                        df.get("inn_ann_prs", pd.Series(index=df.index, dtype=float)))
    ]
    df["ppf"] = pd.to_numeric(pd.Series(ppf, index=df.index), errors="coerce")
    df = df[df["ppf"].notna() & (df["ppf"] <= PPF_MAX_PSI)]
    if df.empty:
        return []

    resp = float(res_pres) if res_pres is not None and float(res_pres) > 0 else None
    era = None
    if era_start is not None:
        stamp = pd.to_datetime(era_start, errors="coerce")
        if not pd.isna(stamp):
            era = stamp

    return [
        {
            "date": when.date().isoformat(),
            "ppf": float(p),
            "bhp": float(b),
            "era": "current" if era is None or when >= era else "prior",
            "buildup": bool(resp is not None and float(b) >= resp),
        }
        for when, p, b in zip(df["sample_date"], df["ppf"], df["btmhole_prs"])
    ]


# ---------------------------------------------------------------------------
# Fail-soft per-well side inputs
# ---------------------------------------------------------------------------


def _era_start_and_pump(well: str) -> tuple[Optional[pd.Timestamp], Optional[str]]:
    """(current pump Date Set, "14B"-style pump label); (None, None) fail-soft."""
    try:
        from woffl.assembly.jp_history import get_current_pump

        jp_hist, source = datasources.jp_history_safe()
        if jp_hist is None or source is None:
            return None, None
        current = get_current_pump(jp_hist, well)
        if current is None:
            return None, None
        stamp = pd.to_datetime(current.get("date_set"), errors="coerce")
        era = None if pd.isna(stamp) else stamp.normalize()
        nozzle = current.get("nozzle_no")
        throat = current.get("throat_ratio")
        pump = f"{nozzle}{throat}" if nozzle and throat else None
        return era, pump
    except Exception:
        log.warning("jp_history unavailable for %s", well, exc_info=True)
        return None, None


def _res_pres(well: str) -> Optional[float]:
    """Reservoir pressure for the buildup flag; None fail-soft.

    Saved fit first: load_saved_ipr is memoized per well (warmed fleet-wide
    at startup), making it the cheapest source of the SAVED value. A locked
    ResP outranks saved values - the sidebar seeding precedence. Falls back
    to the characteristics row when nothing is saved.
    """
    try:
        from server.services import frames
        from woffl.gui.ipr_anchor import load_saved_ipr

        info = load_saved_ipr(well)
        if info:
            locks = info.get("locks") or {}
            lock_val = frames.opt_float((info.get("lock_values") or {}).get("res_pres"))
            if bool(locks.get("res_pres")) and lock_val is not None:
                return lock_val
            saved = frames.opt_float((info.get("values") or {}).get("res_pres"))
            if saved is not None:
                return saved
    except Exception:
        log.warning("saved-fit res_pres lookup failed for %s", well, exc_info=True)

    try:
        from server.services import frames

        chars_df, _source = datasources.well_chars_safe()
        matches = chars_df[chars_df["Well"] == well]
        if matches.empty:
            return None
        return frames.opt_float(matches.iloc[0].get("res_pres"))
    except Exception:
        log.warning("chars res_pres lookup failed for %s", well, exc_info=True)
        return None


def _well_evidence(well: str, res_pres: Optional[float]) -> Optional[dict[str, Any]]:
    """Panel subset of the mined evidence dict; None fail-soft."""
    try:
        res_map = {well: res_pres} if res_pres is not None else None
        row = evidence.pad_evidence([well], res_map).get(well)
    except Exception:
        log.warning("evidence assembly failed for %s", well, exc_info=True)
        return None
    if row is None:
        return None
    return {key: row.get(key) for key in _EVIDENCE_KEYS}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def response_history(well: str) -> dict[str, Any]:
    """ResponseHistoryResponse payload for one well.

    Raises only when the fleet daily fetch itself fails (Databricks down) -
    the router owns that error path. Every side input fails soft.
    """
    fleet = evidence._fleet_pressure_daily()
    sub = fleet[fleet["well"] == well] if not fleet.empty else fleet

    era, pump = _era_start_and_pump(well)
    res_pres = _res_pres(well)

    return {
        "days": build_days(sub, era_start=era, res_pres=res_pres),
        "era_start": era.date().isoformat() if era is not None else None,
        "pump": pump,
        "evidence": _well_evidence(well, res_pres),
        "res_pres": res_pres,
    }
