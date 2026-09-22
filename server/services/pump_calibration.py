"""Versioned installed-pump fits, stored atomically in the existing comment ledger.

The dedicated context holds one self-contained JSON record (under 500 chars).
No schema changes, numeric well-property writes, or best-effort metadata joins.
"""
from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
import uuid

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


# ---------------------------------------------------------------------------
# Gaugeless test-match fits (POST /match-test) as installed-pump calibrations
# ---------------------------------------------------------------------------
#
# A Match test fits kth/kdi (ken held) together with the IPR anchor it infers.
# Save well inputs persists the anchor but, by the installation-scoped pump
# contract, never loose coefficients - so a reopened well lost the match and
# looked reset (user report 2026-09-22). The fit is kept HERE, server-side,
# for an hour under a token: what gets saved is the server's own result, never
# a client claim, exactly like save_fit's job result.

_MATCH_TTL_S = 3600.0
_MATCH_FITS: dict[str, dict] = {}
_MATCH_LOCK = threading.Lock()
# Context seeds that are not well-model inputs: pump hardware and losses.
_NOT_MODEL_INPUTS = {"nozzle_no", "area_ratio", "ken", "kth", "kdi", "nozzle_area_factor", "mach_crit",
                     "ppf_surf", "hydraulics_model"}


def remember_match_fit(well: str, params: dict, result: dict) -> str | None:
    """Keep a successful match for a later explicit save; returns its token.
    ``params`` are the sidebar inputs the match ran on; the matched anchor
    (qwf, pwf, WC) is laid over them the way "Apply to inputs" does."""
    # A failed match, or a closest point whose PF was unreachable (BHP not
    # identified), is not a calibration and gets no save token.
    if result.get("match_quality") == "failed" or result.get("pwf") is None or result.get("pf_reachable") is False:
        return None
    inputs = {k: v for k, v in params.items() if k not in _NOT_MODEL_INPUTS}
    inputs.update(qwf=round(float(result["qwf_liq"])), pwf=round(float(result["pwf"])),
                  form_wc=float(f"{float(result['form_wc']):.3f}"))
    record = {
        "well": well, "at": time.monotonic(), "inputs": inputs,
        "pump": f"{params.get('nozzle_no')}{params.get('area_ratio')}",
        "hydraulics_model": params.get("hydraulics_model", "beggs"),
        "coefs": [float(result["ken"]), float(result["kth"]), float(result["kdi"]),
                  float(params.get("nozzle_area_factor", 1.0))],
        "quality": {k: v for k, v in {
            "n": 1, "provisional": True, "src": "match",
            "pf": abs(round(result["pf_error_pct"], 1)) if result.get("pf_error_pct") is not None else None,
            "oil_pct": abs(round(result["oil_error_pct"], 1)) if result.get("oil_error_pct") is not None else None,
            "mq": result.get("match_quality"),
        }.items() if v is not None},
    }
    token = uuid.uuid4().hex
    with _MATCH_LOCK:
        now = time.monotonic()
        for key in [k for k, v in _MATCH_FITS.items() if now - v["at"] > _MATCH_TTL_S]:
            del _MATCH_FITS[key]
        _MATCH_FITS[token] = record
    return token


def save_match_fit(well: str, token: str) -> dict:
    """Save a remembered Match test fit as this installation's calibration.

    Same record and checks as save_fit: a fresh tracker read must show the
    pump the match ran on, and the FRESH saved well inputs must equal the
    inputs the match ran on (so save the matched well inputs first). The fit
    is one test, so it is always marked provisional.
    """
    from server.services import datasources, ipr, well_model, wells
    from woffl.assembly.jp_history import get_current_pump

    with _MATCH_LOCK:
        rec = _MATCH_FITS.get(token)
    if rec is None or rec["well"] != well or time.monotonic() - rec["at"] > _MATCH_TTL_S:
        raise ValueError("This match expired or belongs to another well. Match the test again.")
    try:
        df = datasources.jp_history_fresh()
    except Exception as exc:
        raise ValueError("Could not verify the current tracker installation; nothing was saved.") from exc
    pump = get_current_pump(df, well) if df is not None else None
    if pump is None:
        raise ValueError("A current tracker installation is required to save a pump calibration.")
    nozzle, throat = str(pump.get("nozzle_no")), str(pump.get("throat_ratio"))
    stamp = installation(pump.get("date_set"))
    if not stamp or f"{nozzle}{throat}" != rec["pump"]:
        raise ValueError(f"The match ran on {rec['pump']} but the tracker shows {nozzle}{throat} installed; nothing was saved.")
    selected = validate_model(rec["hydraulics_model"])
    try:
        ctx = wells.well_context(well, 6, 0, fresh=True, tracker=df)
        saved = well_model.from_context(ctx, selected)
        overlay = {k: v for k, v in rec["inputs"].items() if k in ctx["seeds"]}
        matched = well_model.from_context({**ctx, "seeds": {**ctx["seeds"], **overlay}}, selected)
    except Exception as exc:
        raise ValueError("Could not verify fresh well inputs; nothing was saved. Try again when well data is available.") from exc
    if saved["fingerprint"] != matched["fingerprint"]:
        diff = [k for k in matched["inputs"] if matched["inputs"][k] != saved["inputs"].get(k)]
        raise ValueError("Save the matched well inputs first: the saved " + ", ".join(diff or ["well model"]) +
                         " differ from what this match used. Nothing was saved.")
    record = {"v": 2, "n": nozzle, "t": throat, "i": stamp, "u": saved["fingerprint"],
              "m": physics_model(selected), "h": selected, "k": rec["coefs"], "q": rec["quality"]}
    text = json.dumps(record, separators=(",", ":"), allow_nan=False)
    try:
        decode(text)
    except ValueError as exc:
        raise ValueError(f"The matched coefficients are outside the saveable range ({exc}); nothing was saved.") from exc
    if len(text) > 500:
        raise ValueError("Calibration record exceeds the storage limit; nothing was saved.")
    saved_at = history.next_entry_datetime()
    who = history.resolve_entry_user()
    history.push_eng_comment(well, saved_at, who, text, context=CONTEXT)
    snapshot.cache_clear()
    ipr._invalidate_after_write(well)
    with _MATCH_LOCK:
        _MATCH_FITS.pop(token, None)
    k = rec["coefs"]
    return {"message": f"Saved the matched pump fit for installed {nozzle}{throat} ({stamp[:10]}): kth {k[1]:.3f}, "
                       f"kdi {k[2]:.3f}, ken {k[0]:.3f}. It reloads with this well and is used by new optimization runs "
                       "for this installation only (one test, so marked provisional)."}
