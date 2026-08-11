"""Multi-point event calibration - one background job per well (Pillar 1b).

The engineer's ONE calibration button ("Calibrate to field data"): hydrate
the well exactly as an optimization run would (saved fit -> WellConfig),
gather every measured operating point in the CURRENT pump era
(calibration_points), and fit (ken, kth, kdi, fnz, mach_crit) against all
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

import logging
from typing import Any, Optional

from server import jobs
from server.services import calibration_points, evidence as evidence_svc, optimizer_runs
from server.services import tests as tests_svc

log = logging.getLogger("woffl.web.event_calibration")

_KIND = "event_cal"


def get_job(job_id: str) -> Optional[dict[str, Any]]:
    """Poll envelope for one event-calibration job; None when unknown/expired."""
    return jobs.get(job_id, (_KIND,))


def start_event_calibration(well: str) -> str:
    """Spawn the event-calibration thread for one well; returns the job id."""
    return jobs.start(
        _KIND,
        lambda job: _run_event_calibration_job(job, well),
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
    return f if f == f else None  # NaN-safe


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


def _latest_test_target(well: str) -> Optional[dict[str, Any]]:
    """Newest test row carrying a measured BHP - the same row the web
    client's test picker defaults to (tests_json is newest-first, 6-month
    window like GET /wells/{name}/tests). None when no test has a BHP."""
    try:
        rows = tests_svc.tests_json(well, 6)
    except Exception as exc:  # noqa: BLE001 - fail-soft, refusal stands
        log.warning("well tests unavailable for %s: %s", well, exc)
        return None
    for row in rows:
        if _num(row.get("bhp")) is not None:
            return row
    return None


def _single_point_fallback(
    job: dict[str, Any], well: str, config: Any, nozzle: str, throat: str
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
    test = _latest_test_target(well)
    if test is None:
        return None

    wellbore, wellprof, inflow, res_mix, prop_pf = fric_calibration._build_well_objects(
        config
    )
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
        jpump_direction=getattr(config, "jpump_direction", "reverse"),
    )
    return single_payload(result)


# ---------------------------------------------------------------------------
# The job
# ---------------------------------------------------------------------------


def _run_event_calibration_job(job: dict[str, Any], well: str) -> dict[str, Any]:
    from woffl.gui import fric_calibration
    from woffl.gui.utils import pad_from_mp_name

    pad = pad_from_mp_name(well)
    notes: list[str] = []
    prov: dict[str, dict[str, Any]] = {}
    configs = optimizer_runs._build_configs([pad], set(), [], notes, prov)
    config = next((c for c in configs if c.well_name == well), None)
    if config is None:
        raise ValueError(f"no usable saved fit for {well}")

    current, _rates = optimizer_runs._current_and_tests([well])
    nozzle, throat = current.get(well, (None, None))

    job["progress"] = "building calibration points..."
    res_pres = _num(getattr(config, "res_pres", None))
    surf_pres = _num(getattr(config, "surf_pres", None))
    built = calibration_points.pad_points(
        [well],
        res_pres={well: res_pres} if res_pres is not None else None,
        surf_pres={well: surf_pres} if surf_pres is not None else None,
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
        result = fric_calibration.calibrate_multipoint(
            config, str(nozzle), str(throat), built
        )
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
    if builder_refused and nozzle and throat:
        try:
            single = _single_point_fallback(job, well, config, str(nozzle), str(throat))
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

    return optimizer_runs._plain(
        {
            "well": well,
            "pump": _pump_label(nozzle, throat),
            "era_start": (built or {}).get("era_start"),
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
            "current": {
                "ken": _num(getattr(config, "ken_well", None)),
                "kth": _num(getattr(config, "kth_well", None)),
                "kdi": _num(getattr(config, "kdi_well", None)),
            },
        }
    )
