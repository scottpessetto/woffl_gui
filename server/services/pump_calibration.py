"""Versioned installed-pump fits, stored atomically in the existing comment ledger.

The dedicated context holds one self-contained JSON record (under 500 chars).
No schema changes, numeric well-property writes, or best-effort metadata joins.
"""
from __future__ import annotations

import json
import logging
import math

import pandas as pd

from server import jobs
from server.cache import ttl_cache
from woffl.assembly import prop_hist_client as history
from woffl.assembly.pump_candidates import CLEAN_PUMP
from woffl.assembly.well_test_client import _normalize_well_name
from woffl.flow.entry_energy import MODEL_VERSION

CONTEXT = "pump_calibration_v1"
log = logging.getLogger(__name__)


def installation(value):
    stamp = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(stamp) else stamp.isoformat()


@ttl_cache(300, maxsize=1)
def snapshot():
    """One fleet SELECT, including only the latest record per well."""
    df = history.execute_query(
        "SELECT h.well_name, c.entry_datetime, c.entry_user, c.comment_text "
        "FROM mpu.wells.woffl_eng_comment c "
        "JOIN mpu.wells.vw_well_header h ON h.enthid = c.enthid "
        "WHERE h.well_type = 'prod' AND c.context = 'pump_calibration_v1' "
        "QUALIFY ROW_NUMBER() OVER (PARTITION BY c.enthid ORDER BY c.entry_datetime DESC) = 1"
    )
    return {_normalize_well_name(str(r["well_name"])): r for r in df.to_dict("records")}


def decode(text):
    rec = json.loads(text)
    if rec.get("v") != 1 or len(rec["k"]) != 4 or not installation(rec["i"]):
        raise ValueError("Invalid calibration record")
    for value, (lo, hi) in zip(rec["k"], [(0.001, .4), (.05, 1), (.05, 1), (.8, 1.3)]):
        if not isinstance(value, (float, int)) or not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError("Invalid calibration coefficient")
    return rec


def resolve(well, pump, legacy=None):
    status = {"status": "legacy" if legacy else "none", "coefficients": {}, "quality": None}
    if legacy:
        status["message"] = "Older unscoped calibration retained in history; refit for this installation."
    try:
        row = snapshot().get(well)
        if row is None:
            return status
        rec = decode(row["comment_text"])
        status.update(status="stale", pump=f'{rec["n"]}{rec["t"]}', date_set=rec["i"],
                      physics_model=rec["m"], saved_at=str(row["entry_datetime"]),
                      saved_by=str(row["entry_user"]), quality=rec.get("q"),
                      message="Saved fit belongs to a different installation or physics model; refit before use.")
        if (pump and pump.get("source") == "databricks" and
            (pump.get("nozzle_no"), pump.get("throat_ratio"), installation(pump.get("date_set"))) ==
            (rec["n"], rec["t"], installation(rec["i"])) and rec["m"] == MODEL_VERSION):
            status.update(status="active", coefficients=dict(zip(CLEAN_PUMP, rec["k"])),
                          message="Saved calibration applies only to this installed pump.")
        return status
    except Exception:
        log.warning("Scoped calibration unavailable for %s", well, exc_info=True)
        return {**status, "status": "unavailable", "message": "Calibration could not be verified; using reference coefficients."}


def resolve_current(well, legacy=None):
    """Readiness board: resolve scope without re-running well/IPR hydration."""
    from server.services import datasources
    from woffl.assembly.jp_history import get_current_pump
    try:
        if well not in snapshot():
            return resolve(well, None, legacy)
        df, source = datasources.jp_history_safe()
        pump = get_current_pump(df, well) if df is not None else None
        if pump is not None:
            pump = {**dict(pump), "source": source}
        return resolve(well, pump, legacy)
    except Exception:
        return {"status": "unavailable", "coefficients": {}, "quality": None,
                "message": "Calibration could not be verified; using reference coefficients."}


def save_fit(well, job_id):
    """Save a completed server fit after checking the current tracker installation.

    Quality and full-precision coefficients come from the server job, never a
    client claim. This action accepts a provisional fit; it does not qualify it.
    """
    from server.services import datasources, ipr
    from woffl.assembly.jp_history import get_current_pump

    job = jobs.get(job_id, kinds=("event_cal",))
    result = (job or {}).get("result") or {}
    if not job or job.get("status") != "done" or result.get("well") != well:
        raise ValueError("Calibration expired or belongs to another well. Run calibration again.")
    # A save must verify against a fresh tracker read, not an hour-old/SWR
    # installation. Failure is explicit; never save against the Excel fallback.
    try:
        df = datasources._jp_history_databricks.cache_refresh()
        source = "databricks"
    except Exception as exc:
        raise ValueError("Could not verify the current tracker installation; nothing was saved.") from exc
    pump = get_current_pump(df, well) if df is not None else None
    if pump is None or source != "databricks":
        raise ValueError("A current tracker installation is required to save a pump calibration.")
    nozzle, throat = str(pump.get("nozzle_no")), str(pump.get("throat_ratio"))
    stamp = installation(pump.get("date_set"))
    if (not stamp or stamp != installation(result.get("installation_date_set") or result.get("era_start")) or
        f"{nozzle}{throat}" != result.get("pump") or result.get("physics_model") != MODEL_VERSION):
        raise ValueError("Pump installation or physics model changed since this fit. Run calibration again.")
    fit = result.get("fit")
    single = result.get("single")
    if fit and not result.get("refusal"):
        coefs = [fit[k] for k in ("ken", "kth", "kdi", "fnz")]
        quality = {"bhp": round(fit["rms_bhp_psi"], 1), "pf": round(fit["rms_pf_pct"], 1),
                   "n": fit["n_used"], "bounds": fit.get("railed", []),
                   "beta": fit.get("implied_beta"), "measured_beta": result.get("mined_beta"),
                   "dbhp": fit.get("rms_dbhp_psi")}
        beta, measured = quality["beta"], quality["measured_beta"]
        quality["provisional"] = bool(quality["bounds"] or quality["pf"] > 10 or quality["bhp"] > 50 or
            (beta is not None and measured is not None and abs(beta - measured) > .03))
        quality = {k: round(v, 4) if isinstance(v, float) else v for k, v in quality.items()}
    elif single and single.get("match_quality") not in ("failed", "pinned"):
        # A one-point fit does not identify nozzle area: preserve only the
        # area used by that fit, reported with its server input snapshot.
        coefs = [single[k] for k in ("ken", "kth", "kdi")] + [result.get("current", {}).get("nozzle_area_factor", 1.)]
        quality = {"n": 1, "provisional": True}
    else:
        raise ValueError("This calibration has no valid fitted coefficients to save.")
    rec = {"v": 1, "n": nozzle, "t": throat, "i": stamp, "m": MODEL_VERSION, "k": coefs, "q": quality}
    text = json.dumps(rec, separators=(",", ":"), allow_nan=False)
    decode(text)
    if len(text) > 500:
        raise ValueError("Calibration record exceeds the storage limit; nothing was saved.")
    saved_at = history.next_entry_datetime()
    who = history.resolve_entry_user()
    history.push_eng_comment(well, saved_at, who, text, context=CONTEXT)
    snapshot.cache_clear()
    ipr._invalidate_after_write(well)
    return {"message": f"Saved calibration for installed {nozzle}{throat} ({stamp[:10]}). New optimization runs will use it for this installation only."}
