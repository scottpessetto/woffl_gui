"""Versioned installed-pump fits, stored atomically in the existing comment ledger.

The dedicated context holds one self-contained JSON record (under 500 chars).
No schema changes, numeric well-property writes, or best-effort metadata joins.
"""
from __future__ import annotations

import json
import logging
import math
import re

import pandas as pd

from server import jobs
from server.cache import ttl_cache
from woffl.assembly import prop_hist_client as history
from woffl.assembly.pump_candidates import CLEAN_PUMP
from woffl.assembly.well_test_client import _normalize_well_name
from woffl.flow.entry_energy import MODEL_VERSION
from woffl.flow.hydraulics import physics_model, validate_model

CONTEXT = "pump_calibration_v1"
log = logging.getLogger(__name__)


def installation(value):
    """Exact UTC identity; naive tracker stamps use the ledger's UTC convention."""
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
    validate_model(rec.get("h", "beggs"))
    if rec.get("v") not in (1, 2) or len(rec["k"]) != 4 or not installation(rec["i"]):
        raise ValueError("Invalid calibration record")
    if rec.get("v") == 2 and not re.fullmatch(r"[0-9a-f]{32}", str(rec.get("u", ""))):
        raise ValueError("Invalid well-model fingerprint")
    for value, (lo, hi) in zip(rec["k"], [(0.001, .4), (.05, 1), (.05, 1), (.8, 1.3)]):
        if not isinstance(value, (float, int)) or not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError("Invalid calibration coefficient")
    return rec


def resolve(well, pump, legacy=None, hydraulics_model=None, well_model_fingerprint=None):
    if hydraulics_model is not None:
        validate_model(hydraulics_model)
    status = {"status": "legacy" if legacy else "none", "coefficients": {}, "quality": None,
              "hydraulics_model": hydraulics_model or "beggs"}
    if legacy:
        status["message"] = "Older unscoped calibration retained in history; refit for this installation."
    try:
        row = snapshot().get(well)
        if row is None:
            return status
        rec = decode(row["comment_text"])
        selected = hydraulics_model or rec.get("h", "beggs")
        status["hydraulics_model"] = selected
        status.update(status="stale", pump=f'{rec["n"]}{rec["t"]}', date_set=rec["i"],
                      physics_model=rec["m"], saved_at=str(row["entry_datetime"]),
                      saved_by=str(row["entry_user"]), quality=rec.get("q"),
                      well_model_fingerprint=rec.get("u"),
                      message="Saved fit belongs to a different installation or physics model; refit before use.")
        if (pump and pump.get("source") == "databricks" and
            (pump.get("nozzle_no"), pump.get("throat_ratio"), installation(pump.get("date_set"))) ==
            (rec["n"], rec["t"], installation(rec["i"])) and
            rec.get("h", "beggs") == selected and rec["m"] == physics_model(selected)):
            if rec.get("v") != 2:
                status["message"] = "Saved fit predates the fixed well-IPR contract. Refit using the approved well inputs."
            elif rec["u"] != well_model_fingerprint:
                status["message"] = "Well IPR, fluid or geometry inputs differ from this calibration. Refit before use."
            else:
                status.update(status="active", coefficients=dict(zip(CLEAN_PUMP, rec["k"])),
                              message="Saved calibration matches this installation and well model.")
        return status
    except Exception:
        log.warning("Scoped calibration unavailable for %s", well, exc_info=True)
        return {**status, "status": "unavailable", "message": "Calibration could not be verified; using reference coefficients."}


def resolve_current(well, legacy=None):
    """Readiness uses the same well-model dependency check as optimization."""
    from server.services import wells
    try:
        if well not in snapshot():
            return resolve(well, None, legacy)
        return wells.well_context(well, 6, 0)["pump_calibration"]
    except Exception:
        return {"status": "unavailable", "coefficients": {}, "quality": None,
                "message": "Calibration could not be verified; using reference coefficients."}


def save_fit(well, job_id):
    """Save a completed server fit after checking the current tracker installation.

    Quality and full-precision coefficients come from the server job, never a
    client claim. This action accepts a provisional fit; it does not qualify it.
    """
    from server.services import datasources, ipr, well_model, wells
    from woffl.assembly.jp_history import get_current_pump

    job = jobs.get(job_id, kinds=("event_cal",))
    result = (job or {}).get("result") or {}
    if not job or job.get("status") != "done" or result.get("well") != well:
        raise ValueError("Calibration expired or belongs to another well. Run calibration again.")
    # A save must verify against a fresh tracker read, not an hour-old/SWR
    # installation. Failure is explicit; never save against the Excel fallback.
    try:
        df = datasources.jp_history_fresh()
        source = "databricks"
    except Exception as exc:
        raise ValueError("Could not verify the current tracker installation; nothing was saved.") from exc
    pump = get_current_pump(df, well) if df is not None else None
    if pump is None or source != "databricks":
        raise ValueError("A current tracker installation is required to save a pump calibration.")
    nozzle, throat = str(pump.get("nozzle_no")), str(pump.get("throat_ratio"))
    stamp = installation(pump.get("date_set"))
    selected = validate_model(result.get("hydraulics_model", "beggs"))
    if (not stamp or stamp != installation(result.get("installation_date_set") or result.get("era_start")) or
        f"{nozzle}{throat}" != result.get("pump") or result.get("physics_model") != physics_model(selected)):
        raise ValueError("Pump installation or physics model changed since this fit. Run calibration again.")
    expected = result.get("well_model_fingerprint")
    if not expected:
        raise ValueError("Calibration predates the fixed well-IPR contract. Run calibration again.")
    try:
        current_model = well_model.from_context(wells.well_context(well, 6, 0, fresh=True, tracker=df), selected)
    except Exception as exc:
        raise ValueError("Could not verify fresh well inputs; nothing was saved. Try again when well data is available.") from exc
    if current_model["fingerprint"] != expected:
        raise ValueError("Well inputs changed since this calibration. Run calibration again before saving.")
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
            fit.get("rms_oil_pct") is None or fit.get("n_oil", 0) < 3 or fit.get("rms_oil_pct", 0) > 10 or
            (beta is not None and measured is not None and abs(beta - measured) > .03))
        quality = {k: round(v, 4) if isinstance(v, float) else v for k, v in quality.items()}
        if fit.get("rms_oil_bopd") is not None:
            quality["oil"] = round(fit["rms_oil_bopd"], 1)
        if fit.get("rms_oil_pct") is not None:
            quality["oil_pct"] = round(fit["rms_oil_pct"], 1)
        # Ledger capacity is fixed. Omit absent diagnostics; retain all fitted
        # coefficients at full precision and the explicit review flag.
        quality = {k: v for k, v in quality.items() if v is not None}
    elif single and single.get("match_quality") not in ("failed", "pinned"):
        # A one-point fit does not identify nozzle area: preserve only the
        # area used by that fit, reported with its server input snapshot.
        coefs = [single[k] for k in ("ken", "kth", "kdi")] + [result.get("current", {}).get("nozzle_area_factor", 1.)]
        quality = {"n": 1, "provisional": True}
    else:
        raise ValueError("This calibration has no valid fitted coefficients to save.")
    rec = {"v": 2, "n": nozzle, "t": throat, "i": stamp, "u": expected,
           "m": physics_model(selected), "h": selected, "k": coefs, "q": quality}
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
