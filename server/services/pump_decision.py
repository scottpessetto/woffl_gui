"""Cost of PF on S-Pad's 60 Hz curve, and a single-well pump decision.

The engineer's question: if one well takes another 1,000 BPD of PF - or
gives 1,000 BPD back - how many barrels of oil does that cost (or give) the
OTHER wells? And for one well to act on - a JP replacement, a restart or a
new well - which pump size is worth it once that cost is paid? The compute
is in ``woffl.gui.pad_marginal``; this module models the wells.

Two passes on the curve:

1. Coarse - every producing well at its installed pump across the curve's
   plausible header range, to find today's operating point ``H0``.
2. Fine - the same wells, and the target at every catalog size, at headers
   around ``H0``, so the +/-1,000 BPD sensitivity and every candidate's
   resettled header are interpolated from nearby modeled points.

Well models hydrate from saved fits exactly as an optimization run does
(``optimizer_runs._build_configs``). Installed pumps keep their fit; every
replacement uses clean reference coefficients. READ-ONLY compute on the
shared job registry; poll GET /optimize/run/{job_id}.
"""

from __future__ import annotations

import logging
from copy import copy
from typing import Any, Optional

from woffl.flow.entry_energy import MODEL_VERSION

from server import jobs, schemas
from server.services import optimizer_runs

log = logging.getLogger("woffl.web.pump_decision")

KIND = "pump_decision"
_COARSE_LEVELS = 6


def get_job(job_id: str) -> Optional[dict[str, Any]]:
    return jobs.get(job_id, (KIND,))


def start(req: schemas.PumpDecisionRequest) -> str:
    return jobs.start(KIND, lambda job: _run(job, req), progress="building well models from saved fits...")


def _optimizer(configs: list[Any], header: float, nozzles: list[str], throats: list[str], rho_pf: Any):
    from woffl.assembly.network_optimizer import NetworkOptimizer, PowerFluidConstraint
    from woffl.assembly.parallelism import worker_ceiling
    from woffl.gui.pad_optimize import _EVAL_CAP_FALLBACK_BPD

    clones = []
    for cfg in configs:
        clone = copy(cfg)
        clone.ppf_surf_well = header
        clones.append(clone)
    opt = NetworkOptimizer(clones, PowerFluidConstraint(total_rate=_EVAL_CAP_FALLBACK_BPD, pressure=header, rho_pf=rho_pf),
                           nozzles, throats, marginal_watercut=1.0)
    opt.run_all_batch_simulations(max_workers=worker_ceiling())
    return opt


def _installed_at(configs: list[Any], header: float, rho_pf: Any) -> dict[str, Optional[tuple[float, float]]]:
    """Every well's INSTALLED pump (saved fit) at one forced header. The
    empty catalog lists keep the batch to that one pump per well."""
    opt = _optimizer(configs, header, [], [], rho_pf)
    out: dict[str, Optional[tuple[float, float]]] = {}
    for cfg in configs:
        perf = opt.get_pump_performance(cfg.well_name, cfg.installed_nozzle, cfg.installed_throat, pump_state="installed")
        out[cfg.well_name] = (perf["oil_rate"], perf["lift_water"]) if perf else None
    return out


def _target_at(cfg: Any, header: float, nozzles: list[str], throats: list[str], rho_pf: Any) -> dict[tuple[str, str], tuple[float, float]]:
    """The target at every catalog size (clean reference) plus its installed
    pump (saved fit) at one header: (pump, state) -> (oil, PF)."""
    opt = _optimizer([cfg], header, nozzles, throats, rho_pf)
    df = getattr(opt.batch_results.get(cfg.well_name), "df", None)
    out: dict[tuple[str, str], tuple[float, float]] = {}
    if df is None or df.empty:
        return out
    for rec in df.to_dict("records"):
        state = rec.get("pump_state") or "replacement"
        pump = f"{rec.get('nozzle')}{rec.get('throat')}"
        if (pump, state) in out:
            continue
        perf = opt.get_pump_performance(cfg.well_name, str(rec.get("nozzle")), str(rec.get("throat")), pump_state=state)
        if perf is not None and perf["oil_rate"] > 0 and perf["lift_water"] > 0:
            out[(pump, state)] = (perf["oil_rate"], perf["lift_water"])
    return out


def _with_identity(cfg: Any, current: dict[str, tuple[str, str]]) -> Optional[Any]:
    """A config that knows its installed pump, from the saved seeds or the
    JP tracker; None when neither knows it."""
    if cfg.installed_nozzle and cfg.installed_throat:
        return cfg
    pump = current.get(cfg.well_name)
    if not pump:
        return None
    clone = copy(cfg)
    clone.installed_nozzle, clone.installed_throat = pump
    return clone


def _run(job: dict[str, Any], req: schemas.PumpDecisionRequest) -> dict[str, Any]:
    from woffl.gui import pad_marginal as pm
    from woffl.gui.pad_plant_base import power_fluid_density

    pad = req.pad
    plant = optimizer_runs._pad_plant(pad)
    if plant.coupling != "fixed_curve":
        raise ValueError(f"{pad}-Pad has no fixed pump curve; this cost is only defined for fixed-speed pads")
    target = req.target.strip()
    future_names = {f.name for f in req.future}
    role = "future" if target in future_names else "offline" if target in set(req.offline) else "online"

    notes: list[str] = []
    prov: dict[str, dict[str, Any]] = {}
    configs = optimizer_runs._build_configs([pad], set(req.offline) - {target}, req.future, notes, prov)
    by_name = {c.well_name: c for c in configs}
    if target not in by_name:
        raise ValueError(f"{target} has no usable well model on {pad}-Pad; save its fit (or its donor's) first")

    n_pumps = req.n_pumps if req.n_pumps is not None else optimizer_runs._PAD_DEFAULTS[pad]["n_pumps"]
    lo, hi = plant.clamp_window(n_pumps)
    rho = power_fluid_density(plant)

    def header_of_flow(q: float) -> Optional[float]:
        return plant.header_at_flow(q, n_pumps)

    job["progress"] = "reading current pumps and tests..."
    current, test_rates = optimizer_runs._current_and_tests([c.well_name for c in configs])
    # The other producing wells: existing and online. Planned future wells
    # other than the target are not producing today.
    producing: list[Any] = []
    for c in configs:
        if c.well_name == target or c.well_name in future_names:
            continue
        ident = _with_identity(c, current)
        if ident is None:
            notes.append(f"{c.well_name}: installed pump unknown - its PF draw is missing from the curve")
        else:
            producing.append(ident)
    target_cfg = by_name[target]
    target_installed = _with_identity(target_cfg, current) if role == "online" else None
    if role == "online" and target_installed is None:
        notes.append(f"{target}: installed pump unknown - candidates are compared with an empty slot")

    base_wells = producing + ([target_installed] if target_installed is not None else [])
    points: dict[str, list[pm.Point]] = {c.well_name: [] for c in base_wells}

    def model_installed(levels: list[float], stage: str) -> None:
        for k, level in enumerate(levels):
            jobs.check_cancelled(job)
            job["progress"] = f"{stage} {k + 1}/{len(levels)}: {len(base_wells)} installed pumps at {level:,.0f} psi"
            for w, v in _installed_at(base_wells, level, rho).items():
                if v is not None:
                    points[w].append((level, v[0], v[1]))

    # 1) coarse: the header range this curve can deliver
    cap = plant.flow_window(n_pumps)[1]
    top = header_of_flow(0.05 * cap) or hi
    bottom = header_of_flow(cap) or lo
    c_lo, c_hi = max(lo, min(bottom, top) - 200.0), min(hi, top)
    coarse = [c_lo + (c_hi - c_lo) * i / (_COARSE_LEVELS - 1) for i in range(_COARSE_LEVELS)]
    model_installed(coarse, "finding today's header")
    curves = {w: pm.clean_curve(p) for w, p in points.items()}
    h0 = pm.settle(curves, 0.0, header_of_flow, c_lo, c_hi)
    if h0 is None:
        raise ValueError("The pump curve has no head for today's modeled PF demand")

    # 2) fine: around today's header
    fine = sorted({min(hi, max(lo, round(h0 + d, 1))) for d in pm.FINE_OFFSETS_PSI})
    model_installed(fine, "pricing the header response")
    curves = {w: pm.clean_curve(p) for w, p in points.items()}
    for w in [w for w, c in curves.items() if len(c) < 2]:
        notes.append(f"{w}: installed pump does not solve across the header range - its PF draw is missing from the curve")
        curves.pop(w)
    h0 = pm.settle(curves, 0.0, header_of_flow, lo, hi) or h0

    t_points: dict[tuple[str, str], list[pm.Point]] = {}
    sized = target_installed if target_installed is not None else target_cfg
    for k, level in enumerate(fine):
        jobs.check_cancelled(job)
        job["progress"] = f"sizing {target} {k + 1}/{len(fine)} at {level:,.0f} psi"
        for key, (oil, pf) in _target_at(sized, level, req.nozzles, req.throats, rho).items():
            t_points.setdefault(key, []).append((level, oil, pf))

    others = {w: c for w, c in curves.items() if w != target}
    target_base = curves.get(target) if target_installed is not None else None
    installed_key = ((f"{target_installed.installed_nozzle}{target_installed.installed_throat}", "installed")
                     if target_installed is not None else None)
    candidates = []
    for (pump, state), pts in t_points.items():
        curve = pm.clean_curve(pts)
        if (pump, state) == installed_key and target_base:
            curve = target_base  # the same saved-fit pump, with the coarse points too
        if len(curve) >= 2:
            candidates.append({"pump": pump, "pump_state": state, "curve": curve})
    if not candidates:
        notes.append(f"{target}: no catalog pump solves near {h0:,.0f} psi")

    sens = pm.pf_sensitivity(others, curves, header_of_flow, h0, lo, hi, req.delta_pf_bpd)
    sweep = pm.pf_sweep(others, curves, header_of_flow, h0, lo, hi, max(5000.0, 5.0 * req.delta_pf_bpd))
    rows = pm.score_candidates(others, target_base, candidates, header_of_flow, h0, lo, hi, sens["lambda"])

    today = pm.well_rates(curves, h0)
    wells_today = [{"well": w, "oil": o, "pf": p,
                    "test_oil": (test_rates.get(w) or (None, None))[0],
                    "test_pf": (test_rates.get(w) or (None, None))[1]} for w, (o, p) in sorted(today.items())]
    tr = test_rates.get(target)
    base_t = pm.at_header(target_base, h0) if target_base else None

    return optimizer_runs._plain({
        "pad": pad,
        "target": target,
        "target_role": role,
        "physics_model": MODEL_VERSION,
        "n_pumps": n_pumps,
        "header_psi": h0,
        "model_pf_bpd": sum(p for _o, p in today.values()),
        "model_oil_bopd": sum(o for o, _p in today.values()),
        "test_pf_bpd": sum(float(r["test_pf"] or 0.0) for r in wells_today),
        "levels_psi": sorted(set(coarse) | set(fine)),
        "sensitivity": sens,
        "sweep": sweep,
        "baseline": ({"pump": installed_key[0], "oil": base_t[0], "pf": base_t[1],
                      "test_oil": tr[0] if tr else None, "test_pf": tr[1] if tr else None}
                     if installed_key and base_t else None),
        "current_pump": "".join(current[target]) if target in current else None,
        "candidates": rows,
        "wells_today": wells_today,
        "provenance": prov.get(target),
        "notes": notes,
    })
