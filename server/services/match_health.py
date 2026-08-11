"""Match-health scorecard - the model-vs-field picture per well, one pad.

One background job per pad: hydrate every active well from its saved fit
(exactly as an optimization run does), model each at its CURRENT pump via
pad_optimize.match_check, pull the field-evidence suction rows, and fold it
all into one row per well an engineer can read across:

  fit provenance -> is the inflow curve trusted?
  model / test ratios -> does the model reproduce the latest tests?
  model floor vs measured floor -> is the cavitation claim contradicted?
  measured beta -> does the field say the well responds to PF?
  friction rails -> did calibration degenerate to the bound box corner?

The verdict chip compresses that into one word per well, first match wins:
"contradicted" beats "railed-cal" beats "weak-fit" beats "ok".

READ-ONLY compute, same fail-soft posture as optimizer runs: a dead
warehouse degrades the evidence columns to None, never fails the job.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

from server import jobs
from server.services import evidence as evidence_svc, optimizer_runs, tests as tests_svc

log = logging.getLogger("woffl.web.match_health")

_KIND = "match_health"

# ---------------------------------------------------------------------------
# Verdict tunables (one block)
# ---------------------------------------------------------------------------

FLOOR_VIOLATION_MIN_PSI = 25.0  # model floor this far above measured = contradicted
BETA_RESPONSIVE = 0.03          # measured -dBHP/dPpf at/above this = well responds
WEAK_R2 = 0.5                   # saved-fit r2 below this = weak fit
KEN_RAIL_TOL = 0.01             # |ken - upper bound| under this = railed
KTHDI_RAIL_TOL = 0.001          # |kth/kdi - lower bound| within this = railed


def get_job(job_id: str) -> Optional[dict[str, Any]]:
    """Poll envelope for one scorecard job; None when unknown/expired."""
    return jobs.get(job_id, (_KIND,))


def start_match_health(pad: str) -> str:
    """Spawn the scorecard thread for one pad; returns the job id."""
    return jobs.start(
        _KIND,
        lambda job: _run_match_health_job(job, pad),
        progress="building well models from saved fits...",
    )


# ---------------------------------------------------------------------------
# Pure row assembly (the unit-test surface - no I/O)
# ---------------------------------------------------------------------------


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None  # NaN-safe


def friction_rails(
    ken: Optional[float], kth: Optional[float], kdi: Optional[float]
) -> tuple[bool, bool, bool]:
    """Which loss coefficients sit on the degenerate-fit corner of the
    calibration bound box (ken at its ceiling, kth/kdi on their floors) -
    the Nelder-Mead signature of a sonic-pinned well where the optimizer
    wrote the calibration-day gauge BHP into the cavitation floor."""
    from woffl.gui.fric_calibration import KDI_BOUNDS, KEN_BOUNDS, KTH_BOUNDS

    ken_railed = ken is not None and abs(ken - KEN_BOUNDS[1]) < KEN_RAIL_TOL
    kth_railed = kth is not None and abs(kth - KTH_BOUNDS[0]) <= KTHDI_RAIL_TOL
    kdi_railed = kdi is not None and abs(kdi - KDI_BOUNDS[0]) <= KTHDI_RAIL_TOL
    return ken_railed, kth_railed, kdi_railed


def _verdict(row: dict[str, Any]) -> str:
    """First match wins: contradicted > railed-cal > weak-fit > ok."""
    fv = row.get("floor_violation")
    if fv is not None and fv > FLOOR_VIOLATION_MIN_PSI:
        return "contradicted"
    if (
        row.get("beta_source") == "well"
        and row.get("beta") is not None
        and row["beta"] >= BETA_RESPONSIVE
        and row.get("sonic") is True
    ):
        # The model claims the well is pinned at the cavitation floor
        # (zero suction response) while the well's own measured event
        # pairs show it responding to PF cuts.
        return "contradicted"
    if row.get("ken_railed") or row.get("kth_railed") or row.get("kdi_railed"):
        return "railed-cal"
    r2 = row.get("ipr_r2")
    if r2 is not None and r2 < WEAK_R2:
        return "weak-fit"
    return "ok"


def assemble_rows(
    check_rows: list[dict[str, Any]],
    prov: dict[str, dict[str, Any]],
    evidence: Optional[dict[str, dict[str, Any]]],
    configs: list[Any],
    last_test: dict[str, str],
) -> list[dict[str, Any]]:
    """One scorecard row per match_check row. PURE - no I/O.

    ``evidence`` None (warehouse down) leaves every evidence column None;
    the rest of the row still builds.
    """
    cfg_by_well = {c.well_name: c for c in configs}
    rows: list[dict[str, Any]] = []
    for cr in check_rows:
        w = cr.get("well")
        p = prov.get(w) or {}
        ev = (evidence or {}).get(w) or {}
        cfg = cfg_by_well.get(w)

        ken = _num(getattr(cfg, "ken_well", None))
        kth = _num(getattr(cfg, "kth_well", None))
        kdi = _num(getattr(cfg, "kdi_well", None))
        ken_railed, kth_railed, kdi_railed = friction_rails(ken, kth, kdi)

        model_psu = _num(cr.get("model_psu"))
        floor = _num(ev.get("floor"))
        violation = (
            model_psu - floor if (model_psu is not None and floor is not None) else None
        )

        row: dict[str, Any] = {
            "well": w,
            "pump": cr.get("pump"),
            "ipr_source": p.get("ipr_source"),
            "ipr_r2": _num(p.get("ipr_r2")),
            "test_oil": _num(cr.get("test_oil")),
            "model_oil": _num(cr.get("model_oil")),
            "model_test_oil_ratio": _num(cr.get("oil_ratio")),
            "oil_flag": cr.get("oil_flag"),
            "test_pf": _num(cr.get("test_pf")),
            "model_pf": _num(cr.get("model_pf")),
            "model_test_pf_ratio": _num(cr.get("pf_ratio")),
            "pf_flag": cr.get("pf_flag"),
            "model_psu": model_psu,
            "sonic": cr.get("sonic"),
            "evidence_floor": floor,
            "floor_violation": violation,
            "beta": _num(ev.get("beta")),
            "beta_source": ev.get("beta_source"),
            "n_pairs": ev.get("n_pairs"),
            "ken": ken,
            "kth": kth,
            "kdi": kdi,
            "ken_railed": ken_railed,
            "kth_railed": kth_railed,
            "kdi_railed": kdi_railed,
            "last_test_date": last_test.get(w),
        }
        row["verdict"] = _verdict(row)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Side inputs (fail-soft)
# ---------------------------------------------------------------------------


def _last_test_dates(wells: list[str]) -> dict[str, str]:
    """ISO date of each well's most recent test; missing wells omitted."""
    out: dict[str, str] = {}
    for w in wells:
        try:
            df = tests_svc.tests_for_well(w, 6, 0)
            if df is not None and not df.empty and "WtDate" in df.columns:
                d = pd.to_datetime(df["WtDate"], errors="coerce").max()
                if pd.notna(d):
                    out[w] = d.date().isoformat()
        except Exception:  # noqa: BLE001 - cosmetic column, never fatal
            pass
    return out


# ---------------------------------------------------------------------------
# The job
# ---------------------------------------------------------------------------


def _run_match_health_job(job: dict[str, Any], pad: str) -> dict[str, Any]:
    from woffl.gui.pad_optimize import match_check

    notes: list[str] = []
    prov: dict[str, dict[str, Any]] = {}
    configs = optimizer_runs._build_configs([pad], set(), [], notes, prov)
    if len(configs) == 0:
        raise ValueError(f"no active wells with usable saved fits on {pad}-Pad")

    names = [c.well_name for c in configs]
    job["progress"] = "reading current pumps + tests..."
    current, test_rates = optimizer_runs._current_and_tests(names)
    last_test = _last_test_dates(names)

    job["progress"] = f"modeling {len(configs)} wells at current pumps..."
    n_pumps = optimizer_runs._PAD_DEFAULTS[pad]["n_pumps"]
    check_rows, header = match_check(
        configs, optimizer_runs._pad_plant(pad), n_pumps, current, test_rates
    )

    # Field-evidence suction rows: strictly fail-soft. A dead warehouse
    # leaves the evidence columns None; the scorecard still builds.
    job["progress"] = "reading pressure history..."
    res_pres_map = {
        c.well_name: float(c.res_pres)
        for c in configs
        if getattr(c, "res_pres", None) is not None
    }
    evidence: Optional[dict[str, dict[str, Any]]]
    try:
        evidence = evidence_svc.pad_evidence(names, res_pres_map)
    except Exception as exc:  # noqa: BLE001
        evidence = None
        notes.append(
            f"suction evidence unavailable ({exc}); scorecard is model-vs-test only"
        )

    job["progress"] = "assembling scorecard..."
    rows = assemble_rows(check_rows, prov, evidence, configs, last_test)
    return optimizer_runs._plain(
        {
            "pad": pad,
            "rows": rows,
            "header_psi": header,
            "notes": notes,
            "n_wells": len(rows),
        }
    )
