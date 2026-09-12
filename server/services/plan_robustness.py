"""Bounded fixed-plan stress cases at controlled headers (I/M/E only).

No allocation, fitted-input shift, candidate substitution or measurement scaling.
Both plans retain the source run's exact well inputs and hardware states. The
server's settled source job is the only source of plans/configuration.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import math
from pathlib import Path
from typing import Any

from server import jobs, schemas
from server.services import optimizer_runs
from woffl.flow.entry_energy import MODEL_VERSION

KIND = "pad_robustness"


def source_fingerprint(configs: list[dict[str, Any]]) -> str:
    """Bounded local source/input identity, refreshed at capture and replay.

    This catches same-process source edits as well as changing surveys. No
    warehouse read; absent surveys retain the named generic-profile fallback.
    """
    root = Path(__file__).resolve().parents[2]
    paths = []
    for folder in ("pvt", "flow", "geometry", "assembly", "gui"):
        paths.extend((root / "woffl" / folder).rglob("*.py"))
    paths.extend((root / "woffl" / "jp_data").rglob("*.json"))
    paths.extend(p for p in (root / "woffl" / "jp_data").rglob("*.csv") if "well_surveys" not in p.parts)
    paths.extend((root / "data").glob("*.json"))
    paths.extend((root / "server" / "services" / name) for name in ("plan_robustness.py", "factories.py"))
    paths.extend(root / "woffl" / "jp_data" / "well_surveys" / f"{c['well_name']} Deviation Survey.csv" for c in configs)
    digest = sha256()
    for path in sorted(set(paths)):
        digest.update(path.relative_to(root).as_posix().encode())
        try:
            digest.update(path.read_bytes())
        except FileNotFoundError:
            digest.update(b"<absent>")
    return digest.hexdigest()


def get_job(job_id: str):
    return jobs.get(job_id, (KIND,))


def source_snapshot(job_id: str) -> dict[str, Any]:
    source = jobs.get(job_id, ("pad",))
    if source is None or source["status"] != "done":
        raise ValueError("Run a completed I/M/E JPCO comparison first; this source job is unavailable.")
    result = source.get("result") or {}
    snap = result.get("_plan_snapshot")
    if not result.get("robustness_available") or not snap:
        raise ValueError("This run has incomplete coverage/current hardware, unsupported plant control, or no saved scenario snapshot. Run I/M/E JPCO again after resolving coverage.")
    if snap.get("version") != 1 or snap.get("physics_model") != MODEL_VERSION:
        raise ValueError("Source model version changed; run the pad comparison again.")
    if snap.get("source_fingerprint") != source_fingerprint(snap.get("configs", [])):
        raise ValueError("Model source, plant inputs, or surveys changed; run the pad comparison again.")
    if not 0 < len(snap.get("configs", [])) <= 100:
        raise ValueError("The comparison supports 1-100 modeled wells.")
    return deepcopy(snap)


def start(req: schemas.PadRobustnessRequest) -> str:
    snapshot = source_snapshot(req.source_job_id)
    return jobs.start(KIND, lambda job: run(job, req, snapshot), progress="checking fixed-plan scenarios...")


def cases(req: schemas.PadRobustnessRequest) -> list[dict[str, Any]]:
    """At most nine explicit, equally unweighted engineering stress cases."""
    out = [{"name": "Base", "wc_offset": 0., "gor_factor": 1., "header_offset": 0.}]
    for field, span, title, scale, center in (
        ("wc_offset", req.wc_points, "All-well WC", .01, 0.),
        ("gor_factor", req.gor_percent, "All-well GOR", .01, 1.),
        ("header_offset", req.header_psi, "Shared header", 1., 0.),
    ):
        if span == 0:
            continue
        for sign in (-1, 1):
            out.append({**out[0], "name": f"{title} {sign * span:+g}" +
                        (" pp" if field == "wc_offset" else " %" if field == "gor_factor" else " psi"),
                        field: center + sign * span * scale})
    if req.joint_cases and any((req.wc_points, req.gor_percent, req.header_psi)):
        for sign in (-1, 1):
            out.append({"name": "Joint: " + ("higher WC/GOR, lower header" if sign > 0 else "lower WC/GOR, higher header"),
                        "wc_offset": sign * req.wc_points / 100., "gor_factor": 1 + sign * req.gor_percent / 100.,
                        "header_offset": -sign * req.header_psi})
    # Zero ranges can make a joint case duplicate a one-at-a-time case.
    unique = {}
    for case in out:
        unique.setdefault((case["wc_offset"], case["gor_factor"], case["header_offset"]), case)
    return list(unique.values())


def scenario_configs(snapshot: dict, case: dict) -> list[Any]:
    """Hold the oil IPR, anchor pressure and reservoir pressure exactly fixed."""
    from woffl.assembly.network_optimizer import WellConfig
    configs = []
    for raw in snapshot["configs"]:
        values = dict(raw)
        wc = float(raw["form_wc"]) + case["wc_offset"]
        if not 0 <= wc < .99:
            raise ValueError(f"{raw['well_name']}: scenario WC {wc:.4f} is outside the supported oil-model range; case not clipped.")
        values["form_wc"] = wc
        values["qwf"] = float(raw["qwf"]) * (1 - float(raw["form_wc"])) / (1 - wc)
        values["form_gor"] = float(raw["form_gor"]) * case["gor_factor"]
        configs.append(WellConfig(**values))
    return configs


def solve_options(configs: list[Any], plans: dict, pressure: float, job: dict) -> dict:
    """At most two catalog choices per well, through the shared batch runtime."""
    from woffl.assembly.network_optimizer import NetworkOptimizer, PowerFluidConstraint
    from woffl.assembly.parallelism import worker_ceiling
    groups = {}
    for cfg in configs:
        wanted = [p[cfg.well_name] for p in plans.values() if p.get(cfg.well_name)]
        if not wanted:
            continue
        key = (tuple(sorted({c[0] for c in wanted})), tuple(sorted({c[1] for c in wanted})))
        cfg.ppf_surf_well = pressure
        groups.setdefault(key, []).append(cfg)
    found = {}
    for (nozzles, throats), group in groups.items():
        jobs.check_cancelled(job)
        optimizer = NetworkOptimizer(group,
            PowerFluidConstraint(total_rate=500000., pressure=pressure, rho_pf=None),
            list(nozzles), list(throats), marginal_watercut=1.)
        optimizer.run_all_batch_simulations(max_workers=worker_ceiling())
        for cfg in group:
            for plan in plans.values():
                choice = plan.get(cfg.well_name)
                if choice:
                    key = (cfg.well_name, *choice)
                    if key not in found:
                        found[key] = optimizer.get_pump_performance(cfg.well_name, *choice[:2], pump_state=choice[2])
    return found


def score_plan(name: str, choices: dict, configs: list[Any], options: dict,
               plant: Any, n_pumps: Any, pressure: float, lam: float) -> dict:
    """Strict selected-candidate totals. Missing physics is unknown, not zero."""
    oil = pf = form_water = 0.
    failures = []
    for cfg in configs:
        if cfg.well_name not in choices:
            failures.append(cfg.well_name)
            continue
        choice = choices[cfg.well_name]
        if choice is None:  # explicit plan shut-in/offline
            continue
        perf = options.get((cfg.well_name, *choice))
        if perf is None or any(not _valid_rate(perf.get(k)) for k in ("oil_rate", "lift_water", "formation_water")):
            failures.append(cfg.well_name)
            continue
        oil += perf["oil_rate"]
        pf += perf["lift_water"]
        form_water += perf["formation_water"]
    empty = {"plan": name, "feasible": None, "oil": None, "pf": None, "machine_water": None,
             "budget": None, "objective": None, "regret": None, "reason": "", "failed_wells": failures}
    if failures:
        return {**empty, "reason": "Selected pump did not solve: " + ", ".join(failures)}
    machine_water = pf + form_water if plant.water_key == "totl_wat" else pf
    budget = plant.budget_at_pressure(pressure, n_pumps)
    lo, hi = plant.clamp_window(n_pumps)
    flags = plant.flags(machine_water, n_pumps)
    delivered, over = plant.delivered_header(machine_water, pressure, n_pumps)
    residual = delivered - pressure if delivered is not None else None
    feasible = (lo <= pressure <= hi and budget is not None and machine_water <= budget + 1e-6
                and not flags.get("over_capacity", False) and not over
                and residual is not None and math.isfinite(residual) and abs(residual) <= 10.)
    return {**empty, "feasible": feasible, "oil": oil, "pf": pf, "machine_water": machine_water,
            "budget": budget, "objective": oil - lam * machine_water if feasible else None,
            "reason": "Within modeled plant capacity." if feasible else "Fixed plan cannot hold this header within modeled plant capacity.",
            "delivered_header_psi": delivered, "coupling_residual_psi": residual,
            "recirculation": bool(flags.get("recirc", False))}


def _valid_rate(value: Any) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) >= 0
    except (TypeError, ValueError):
        return False


def run(job: dict, req: schemas.PadRobustnessRequest, snapshot: dict) -> dict:
    source_req = schemas.OptimizeRunRequest(**snapshot["request"])
    if source_req.pad not in {"I", "M", "E"} or source_req.strategy != "jpco":
        raise ValueError("Fixed-plan stress cases support controlled-header I/M/E JPCO runs only.")
    plant = optimizer_runs._pad_plant_for_run(source_req.pad, source_req)
    n_pumps = source_req.n_pumps if source_req.n_pumps is not None else optimizer_runs._PAD_DEFAULTS[source_req.pad]["n_pumps"]
    plans = {"Current": snapshot["current"], "Proposed": snapshot["proposed"]}
    lam = float(snapshot.get("lambda_used") or 0.)
    results = []
    scenario_list = cases(req)
    for i, case in enumerate(scenario_list):
        jobs.check_cancelled(job)
        job["progress"] = f"Scenario {i + 1}/{len(scenario_list)}: {case['name']}"
        pressure = float(snapshot["header_psi"]) + case["header_offset"]
        error = None
        try:
            if not 1000. <= pressure <= 5000.:
                raise ValueError("Scenario header is outside the supported pressure range; case not clipped.")
            configs = scenario_configs(snapshot, case)
            options = solve_options(configs, plans, pressure, job)
            scores = [score_plan(name, choices, configs, options, plant, n_pumps, pressure, lam)
                      for name, choices in plans.items()]
        except jobs.JobCancelled:
            raise
        except Exception as exc:
            error = str(exc)
            scores = [{"plan": name, "feasible": None, "oil": None, "pf": None, "machine_water": None,
                       "budget": None, "objective": None, "regret": None, "reason": error,
                       "failed_wells": []} for name in plans]
        feasible = [s for s in scores if s["feasible"] is True]
        if len(feasible) == len(plans):
            best = max(s["objective"] for s in feasible)
            for s in feasible:
                s["regret"] = max(0., best - s["objective"])
        a, b = scores
        gain = b["oil"] - a["oil"] if a["oil"] is not None and b["oil"] is not None else None
        gain = 0. if gain is not None and abs(gain) < 1e-8 else gain
        results.append({"case": case, "header_psi": pressure, "plans": scores, "error": error,
                        "proposed_oil_delta": gain,
                        "preferred": [s["plan"] for s in feasible if s["regret"] is not None and s["regret"] <= 1e-8]})
    summaries = []
    for name in plans:
        scores = [next(s for s in row["plans"] if s["plan"] == name) for row in results]
        comparable = [r for r in results if all(s["feasible"] is True for s in r["plans"])]
        summaries.append({"plan": name, "feasible_cases": sum(s["feasible"] is True for s in scores),
                          "failed_cases": sum(s["feasible"] is None for s in scores),
                          "infeasible_cases": sum(s["feasible"] is False for s in scores),
                          "preferred_when_both_feasible": sum(name in r["preferred"] for r in comparable),
                          "comparable_cases": len(comparable),
                          "max_regret": max((s["regret"] for s in scores if s["regret"] is not None), default=None)})
    deltas = [r["proposed_oil_delta"] for r in results if all(s["feasible"] is True for s in r["plans"])]
    return optimizer_runs._plain({"source_job_id": req.source_job_id, "request": req.model_dump(),
        "pad": source_req.pad, "water_key": plant.water_key, "lambda_used": lam,
        "cases": results, "plans": summaries, "min_comparable_oil_gain": min(deltas) if deltas else None,
        "max_comparable_oil_gain": max(deltas) if deltas else None,
        "assumptions": "Nine or fewer user-selected engineering stress cases, not probabilities or uncertainty bounds. WC/GOR offsets move all wells together; each oil IPR stays fixed. Optional joint corners are stated assumptions about co-movement. Headers are controlled operating scenarios, not fitted plant uncertainty. Both fixed plans use the same case and the source run's water price. No well input is saved, no pump is substituted, and no allocation is rerun. Preference/regret compare only these two plans; unknown and infeasible cases are retained."})
