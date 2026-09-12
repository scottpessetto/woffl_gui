"""Multi-point event calibration - one background job per well (Pillar 1b).

The engineer's ONE calibration button ("Calibrate to field data"): hydrate
the well exactly as an optimization run would (saved fit -> WellConfig),
gather every measured operating point in the CURRENT pump era
(calibration_points), and fit (ken, kth, kdi, fnz) against all
of them at once (fric_calibration.calibrate_multipoint). Identifiability
comes from data spread; when the builder refuses (young era / no data /
no spread) the job falls back to the single-point latest-test BHP match
(the /calibrate mechanics) and says so via method="single_point" +
fallback_reason. Only when the fallback is impossible too does the well
get the honest event refusal string.

READ-ONLY compute, same posture as match_health: field evidence (the mined
beta the fit is judged against) is strictly fail-soft - a dead warehouse
leaves mined_beta None, never fails the job. Nothing here writes anywhere;
persisting an accepted fit is the save path's job.
"""

from __future__ import annotations
from copy import deepcopy

from woffl.flow.entry_energy import MODEL_VERSION
from woffl.flow.hydraulics import physics_model, validate_model

import logging
import math
import os
import tempfile
import pandas as pd
from typing import Any, Optional

from server import jobs, pool
from server.services import calibration_points, evidence as evidence_svc, optimizer_runs
from server.services import tests as tests_svc

log = logging.getLogger("woffl.web.event_calibration")

_KIND = "event_cal"


def get_job(job_id: str) -> Optional[dict[str, Any]]:
    """Poll envelope for one event-calibration job; None when unknown/expired."""
    return jobs.get(job_id, (_KIND,))


def start_event_calibration(well: str, hydraulics_model: str | None = None) -> str:
    """Spawn the event-calibration thread for one well; returns the job id."""
    return jobs.start(
        _KIND,
        lambda job: _run_event_calibration_job(job, well, hydraulics_model),
        progress="hydrating saved fit...",
    )


# ---------------------------------------------------------------------------
# Pure payload assembly (the unit-test surface - no I/O)
# ---------------------------------------------------------------------------


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def fit_payload(fit: Any) -> dict[str, Any]:
    """The contract's `fit` object from a non-refused MultipointResult."""
    return {
        "ken": float(fit.best_ken),
        "kth": float(fit.best_kth),
        "kdi": float(fit.best_kdi),
        "fnz": float(fit.best_fnz),
        "mach_crit": float(fit.best_mach_crit),
        "rms_bhp_psi": float(fit.rms_bhp_psi),
        "rms_pf_pct": float(fit.rms_pf_pct),
        "rms_dbhp_psi": _num(fit.rms_dbhp_psi),
        "rms_oil_bopd": _num(getattr(fit, "rms_oil_bopd", None)),
        "rms_oil_pct": _num(getattr(fit, "rms_oil_pct", None)),
        "n_oil": int(getattr(fit, "n_oil", 0)),
        "per_point": getattr(fit, "per_point", []),
        "n_used": int(fit.n_used),
        "n_dropped": int(fit.n_dropped),
        "railed": list(fit.railed),
        "implied_beta": _num(fit.implied_beta),
        "message": fit.message,
    }


def single_payload(res: Any) -> dict[str, Any]:
    """The contract's `single` object from a FricCalibrationResult (the
    single-point fallback leg)."""
    return {
        "ken": float(res.best_ken),
        "kth": float(res.best_kth),
        "kdi": float(res.best_kdi),
        "modeled_bhp": _num(res.best_modeled_bhp),
        "target_bhp": _num(res.target_bhp),
        "match_quality": str(res.match_quality),
        "message": res.message,
    }


def _pump_label(nozzle: Any, throat: Any) -> Optional[str]:
    """Display convention shared with optimizer_runs rows: '12B'."""
    if nozzle and throat:
        return f"{nozzle}{throat}"
    return None


# ---------------------------------------------------------------------------
# Single-point fallback (young era) - the Auto-match BHP path, server-side
# ---------------------------------------------------------------------------


def _latest_test_target(well: str, era_start=None, direction=None) -> Optional[dict[str, Any]]:
    """Newest test row carrying a measured BHP - the same row the web
    client's test picker defaults to (tests_json is newest-first, 6-month
    window like GET /wells/{name}/tests). None when no test has a BHP."""
    try:
        rows = tests_svc.tests_json(well, 6)
    except Exception as exc:  # noqa: BLE001 - fail-soft, refusal stands
        log.warning("well tests unavailable for %s: %s", well, exc)
        return None
    for row in rows:
        if era_start is not None:
            date = pd.to_datetime(row.get("date"), utc=True, errors="coerce")
            start = pd.to_datetime(era_start, utc=True, errors="coerce")
            if pd.isna(date) or pd.isna(start) or date.normalize() <= start.normalize():
                continue
        wc = calibration_points._test_wc({"form_wc": row.get("form_wc"), "WtOilVol": row.get("oil"), "WtWaterVol": row.get("water")}, None)
        gor = _num(row.get("fgor"))
        pf = _num(row.get("pf_press"))
        source_direction = {"annulus": "reverse", "tubing": "forward"}.get(row.get("pf_source"))
        if direction and source_direction and source_direction != direction:
            continue
        if (_num(row.get("bhp")) or 0) > 50 and wc is not None and gor is not None and gor >= 0 and pf is not None and 800 <= pf <= 5500:
            return row
    return None


def _single_point_fallback(
    job: dict[str, Any], well: str, config: Any, nozzle: str, throat: str,
    era_start=None,
) -> Optional[dict[str, Any]]:
    """Run the /calibrate single-point path from the hydrated config.

    Mirrors solve.calibrate's assembly: sim objects from the saved fit
    (the same factory calibrate_multipoint uses), test-day WHP when
    measured else the config surface pressure, test-day PF pressure when
    measured else the saved per-well PF pressure. None when no test with
    a measured BHP exists - the caller keeps the event refusal.
    """
    from woffl.gui import fric_calibration

    job["progress"] = "young era - matching latest test BHP instead..."
    test = _latest_test_target(well, era_start, getattr(config, "jpump_direction", None))
    if test is None:
        return None

    wc = calibration_points._test_wc({"form_wc": test.get("form_wc"), "WtOilVol": test.get("oil"), "WtWaterVol": test.get("water")}, None)
    gor = _num(test.get("fgor"))
    if wc is None or gor is None or gor < 0:
        return None
    at = deepcopy(config)
    # Change the measured water/gas mixture while preserving the approved oil curve.
    at.qwf = float(config.qwf)*(1-float(config.form_wc))/(1-wc)
    at.form_wc, at.form_gor = wc, gor
    wellbore, wellprof, inflow, res_mix, prop_pf = fric_calibration._build_well_objects(at)
    whp = _num(test.get("whp"))
    pwh = whp if whp is not None and whp > 0 else float(config.surf_pres)
    pf_press = _num(test.get("pf_press"))
    ppf_surf = (
        pf_press
        if pf_press is not None and pf_press > 0
        else _num(getattr(config, "ppf_surf_well", None)) or 3168.0
    )
    result = fric_calibration.calibrate_friction_coefs(
        well_name=well,
        target_bhp=float(test["bhp"]),
        pwh=float(pwh),
        tsu=float(config.form_temp),
        ppf_surf=float(ppf_surf),
        nozzle=str(nozzle),
        throat=str(throat),
        knz=0.01,
        ken=_num(getattr(config, "ken_well", None)) or fric_calibration.NEUTRAL_KEN,
        wellbore=wellbore,
        wellprof=wellprof,
        ipr_su=inflow,
        prop_su=res_mix,
        prop_pf=prop_pf,
        seed_kth=_num(getattr(config, "kth_well", None)) or fric_calibration.NEUTRAL_KTH,
        seed_kdi=_num(getattr(config, "kdi_well", None)) or fric_calibration.NEUTRAL_KDI,
        nozzle_area_factor=_num(getattr(config, "fnz_well", None)) or 1.0,
        mach_crit=_num(getattr(config, "mach_crit_well", None)) or 1.0,
        jpump_direction=getattr(config, "jpump_direction", "reverse"),
        hydraulics_model=getattr(config, "hydraulics_model", "beggs"),
    )
    return single_payload(result)


# ---------------------------------------------------------------------------
# The job
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The fit itself: one pool worker, progress relayed through a file
# ---------------------------------------------------------------------------

# How often the job thread looks for a new progress line while it waits on
# the worker. The fitter reports every MP_PROGRESS_EVERY cost evaluations
# (a few seconds apart), so half a second is plenty and costs nothing.
PROGRESS_POLL_S = 0.5


def _write_progress(path: str, msg: str) -> None:
    """Atomic overwrite (write-then-replace) so the reader never sees a
    half-written line."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(msg)
    os.replace(tmp, path)


def _multipoint_worker(config: Any, nozzle: str, throat: str, built: Any, progress_path: str):
    """Run calibrate_multipoint in a pool CHILD.

    Picklable arguments only (the WellConfig dataclass, the builder's point
    dicts, a file path) - the worker builds its own physics objects, exactly
    as the in-thread fit does. Progress cannot reach the parent's job dict
    from here, so each line is written to ``progress_path`` and the parent
    polls it (see _fit_multipoint). A write failure is swallowed: progress
    is cosmetic, the fit is not.
    """
    from woffl.gui import fric_calibration

    def _progress(msg: str) -> None:
        try:
            _write_progress(progress_path, msg)
        except OSError:
            pass

    return fric_calibration.calibrate_multipoint(config, nozzle, throat, built, progress=_progress)


def _fit_multipoint(
    job: dict[str, Any], config: Any, nozzle: str, throat: str, built: Any, progress: Any
):
    """The multipoint fit, on the process pool when it is up, else in-thread.

    Why: the fit is 2-3 minutes of pure solving (MPE-35: 24 points, four
    Nelder-Mead passes). On the job thread it held the GIL the whole time
    and every other request crawled (a 33 ms /context read measured at
    1.25 s under a 4 s sweep; a 3-minute fit froze the app for every
    engineer on the 2-vCPU tier). In a worker the job thread just waits on
    a Future, relaying the worker's progress lines into the envelope.

    Hydration and the field-evidence query stay on the job thread: they
    are Databricks I/O, and the worker would only pay for its own session.

    Fallback: pool down, pool broken, or submit refused -> the in-thread fit
    with the direct progress callback, exactly as before. A fit that RAISES
    raises here either way (the job records the error).
    """
    from concurrent.futures import TimeoutError as FutureTimeout
    from concurrent.futures.process import BrokenProcessPool

    from woffl.gui import fric_calibration

    if pool.workers() > 0:
        fd, path = tempfile.mkstemp(prefix="woffl_cal_", suffix=".txt")
        os.close(fd)
        try:
            fut = pool.submit(_multipoint_worker, config, nozzle, throat, built, path)
            if fut is not None:
                job["progress"] = f"fitting {len((built or {}).get('points') or []) if isinstance(built, dict) else len(built or [])} points (pool worker)..."
                last: Optional[str] = None
                while True:
                    try:
                        return fut.result(timeout=PROGRESS_POLL_S)
                    except FutureTimeout:
                        try:
                            with open(path, encoding="utf-8") as fh:
                                msg = fh.read().strip()
                        except OSError:
                            msg = ""
                        if msg and msg != last:
                            job["progress"] = msg
                            last = msg
                    except BrokenProcessPool:
                        log.warning("process pool broke mid-fit for %s; refitting in-thread", getattr(config, "well_name", "?"))
                        break
        finally:
            for name in (path, path + ".tmp"):
                try:
                    os.remove(name)
                except OSError:
                    pass
    return fric_calibration.calibrate_multipoint(config, nozzle, throat, built, progress=progress)


def _run_event_calibration_job(job: dict[str, Any], well: str, hydraulics_model: str | None = None) -> dict[str, Any]:
    from woffl.gui import fric_calibration
    from server.services.wells import _pad_from_mp_name as pad_from_mp_name

    pad = pad_from_mp_name(well)
    notes: list[str] = []
    prov: dict[str, dict[str, Any]] = {}
    configs = optimizer_runs._build_configs([pad], set(), [], notes, prov)
    config = next((c for c in configs if c.well_name == well), None)
    if config is None:
        raise ValueError(f"no usable saved fit for {well}")
    selected = validate_model(hydraulics_model or getattr(config, "hydraulics_model", "beggs"))
    if selected != getattr(config, "hydraulics_model", "beggs"):
        # A different return model needs its own fitted pump coefficients.
        config.ken_well, config.kth_well, config.kdi_well, config.fnz_well = .03, .3, .4, 1.
    config.hydraulics_model = selected
    from server.services.well_model import describe
    well_model = describe(config)

    current, _rates = optimizer_runs._current_and_tests([well])
    nozzle, throat = current.get(well, (None, None))

    job["progress"] = "building calibration points..."
    res_pres = _num(getattr(config, "res_pres", None))
    surf_pres = _num(getattr(config, "surf_pres", None))
    built = calibration_points.pad_points(
        [well],
        res_pres={well: res_pres} if res_pres is not None else None,
        surf_pres={well: surf_pres} if surf_pres is not None else None,
        directions={well: getattr(config, "jpump_direction", "reverse")},
    ).get(well)

    # The installed pump: the JP tracker's word first, else the era pump the
    # points builder resolved (same jp_history row, different fetch path).
    built_pump = (built or {}).get("pump") or {}
    if not (nozzle and throat):
        nozzle, throat = built_pump.get("nozzle"), built_pump.get("throat")

    refusal: Optional[str] = None
    fit: Optional[dict[str, Any]] = None
    builder_refused = False
    if built is None:
        refusal = "no calibration data"
        builder_refused = True
    elif built.get("refusal"):
        refusal = str(built["refusal"])
        builder_refused = True
    elif not (nozzle and throat):
        refusal = "no current pump installed"
    else:
        n_points = len(built.get("points") or [])
        job["progress"] = f"fitting {n_points} points..."

        # The fit is up to four Nelder-Mead passes of ~100 iterations, each
        # solving every point ~170 times - 25-62 s PER PASS at 24 points
        # (MPE-35, 2026-09-01: 3.2 min end to end). Stream the fitter's
        # pass/evaluation line into the envelope so the poller shows
        # movement instead of one frozen string for minutes.
        def _progress(msg: str) -> None:
            job["progress"] = msg

        result = _fit_multipoint(job, config, str(nozzle), str(throat), built, _progress)
        if result.refusal:
            refusal = str(result.refusal)
        else:
            fit = fit_payload(result)

    # Builder refusal (young era / no data / no spread): do NOT stop - fall
    # back to the single-point BHP match the standalone /calibrate endpoint
    # runs, fed from the same hydrated config. Strictly fail-soft: if the
    # fallback is impossible too (no test BHP, no pump) or blows up, the
    # honest event refusal stands exactly as before.
    method = "event"
    fallback_reason: Optional[str] = None
    single: Optional[dict[str, Any]] = None
    if builder_refused and nozzle and throat and (built or {}).get("era_start"):
        try:
            single = _single_point_fallback(
                job, well, config, str(nozzle), str(throat),
                era_start=(built or {}).get("era_start"),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("single-point fallback failed for %s: %s", well, exc)
            single = None
        if single is not None:
            method = "single_point"
            fallback_reason = refusal
            refusal = None

    # Field-evidence beta: strictly fail-soft - a dead warehouse leaves the
    # mined columns None; the fit report still builds.
    mined_beta: Optional[float] = None
    mined_beta_source: Optional[str] = None
    # A Databricks round trip (or stall) here used to hide behind the last
    # fitter line; label it so a slow warehouse reads as what it is.
    job["progress"] = "fit done - checking the measured suction response (field evidence)..."
    try:
        ev = evidence_svc.pad_evidence(
            [well], {well: res_pres} if res_pres is not None else None
        )
        row = (ev or {}).get(well)
        if row is not None:
            mined_beta = _num(row.get("beta"))
            mined_beta_source = row.get("beta_source")
    except Exception as exc:  # noqa: BLE001
        log.warning("mined-beta evidence unavailable for %s: %s", well, exc)

    from server.services.pump_calibration import installation

    return optimizer_runs._plain(
        {
            "physics_model": physics_model(selected),
            "well_model_fingerprint": well_model["fingerprint"],
            "well_model_inputs": well_model["inputs"],
            "calibration_contract": "fixed-oil-ipr-v1",
            "hydraulics_model": selected,
            "well": well,
            "pump": _pump_label(nozzle, throat),
            "era_start": (built or {}).get("era_start"),
            "installation_date_set": installation(((built or {}).get("pump") or {}).get("date_set")),
            "n_daily": int((built or {}).get("n_daily") or 0),
            "n_test": int((built or {}).get("n_test") or 0),
            "ppf_spread": float((built or {}).get("ppf_spread") or 0.0),
            "refusal": refusal,
            "method": method,
            "fallback_reason": fallback_reason,
            "single": single,
            "fit": fit,
            "mined_beta": mined_beta,
            "mined_beta_source": mined_beta_source,
            "mined_beta_scope": "well_history" if mined_beta_source == "well" else mined_beta_source,
            "response_validation": "diagnostic_not_holdout",
            "data_exclusions": (built or {}).get("excluded", []),
            "composition_policy": (built or {}).get("composition_policy"),
            "current": {
                "nozzle_area_factor": _num(getattr(config, "fnz_well", None)) or 1.0,
                "ken": _num(getattr(config, "ken_well", None)),
                "kth": _num(getattr(config, "kth_well", None)),
                "kdi": _num(getattr(config, "kdi_well", None)),
            },
        }
    )
