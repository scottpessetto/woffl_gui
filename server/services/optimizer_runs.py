"""Optimization runs - M/S/I pad and CFP - as in-process background jobs.

The ENGINES are the same pure modules the Streamlit pad pages call:
``woffl.gui.pad_optimize`` (MILP/MCKP allocation against a PadPlant, S =
fixed-curve fixed point, I/M = free-pressure sweep) and
``woffl.gui.cfp_moves`` (anchored-delta moves over response surfaces for the
B/G/C/J CFP plant). What changed is the WELL-MODEL SOURCE: WellConfigs
hydrate from each well's SAVED FIT via the same context-seeding pipeline
the sidebar uses (chars/as-built + prop_hist saved IPR + calibrated
friction + locks) instead of the Streamlit session review store. That is
the redesigned workflow: fits are matched and saved on the Single Well
solver; runs consume them.

Board config: wells checked OFFLINE on the Optimization page are excluded
from a run; FUTURE wells run under their own name with their donor well's
seeds (generic well profile when no survey exists for the new name).

Every trial header re-simulates every well x nozzle x throat, so a run
takes minutes. POST /optimize/run starts a daemon thread and returns a job
id; GET /optimize/run/{id} polls {status, progress, result}. The registry
behind that is ``server.jobs`` - in process memory (single-worker
deployment), pruned an hour after a job settles - shared with the
sensitivity combine study.

Engine mutation hazard: the run loops write the trial header into
``WellConfig.ppf_surf_well`` IN PLACE, so configs are built fresh per run
and never shared between jobs.
"""

from __future__ import annotations

from woffl.flow.entry_energy import MODEL_VERSION
from woffl.flow.hydraulics import physics_model

import logging

import math
from copy import copy
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

from server import jobs, schemas
from server.services import datasources, evidence as evidence_svc, tests as tests_svc, wells as wells_svc

# ---------------------------------------------------------------------------
# Job registry
# ---------------------------------------------------------------------------
#
# The registry itself is server.jobs, shared with the sensitivity combine
# study. These two are the optimizer's typed door onto it.

_KINDS = ("pad", "cfp")


def get_job(job_id: str) -> Optional[dict[str, Any]]:
    """Poll envelope for one optimization run; None when unknown/expired."""
    return jobs.get(job_id, _KINDS)


def start_run(req: schemas.OptimizeRunRequest) -> str:
    """Spawn the run thread; returns the job id immediately."""
    runner = _run_pad_job if req.kind == "pad" else _run_cfp_job
    return jobs.start(
        req.kind,
        lambda job: runner(job, req),
        progress="building well models from saved fits...",
    )


# ---------------------------------------------------------------------------
# JSON flattening (engine results carry numpy / pandas / dataclasses)
# ---------------------------------------------------------------------------


def _plain(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_plain(x) for x in v]
    if isinstance(v, pd.DataFrame):
        return [_plain(r) for r in v.to_dict("records")]
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if is_dataclass(v) and not isinstance(v, type):
        return _plain(asdict(v))
    if isinstance(v, (pd.Timestamp, datetime)):
        return str(v)
    if v is None or isinstance(v, (str, int, float, bool)):
        return v
    return str(v)


# ---------------------------------------------------------------------------
# Well-model hydration from saved fits
# ---------------------------------------------------------------------------

log = logging.getLogger("woffl.web.optimizer_runs")

# Above this the oil rate collapses to allocation noise and the solver's oil
# IPR is meaningless. Such a well is REFUSED here (the caller records it as an
# invalid model), never silently capped: capping to 0.99 entered a dewatering
# well as a 1%-oil producer and handed it a pump recommendation (review
# 2026-09-01, SRV-3; AGENTS.md §8 "silently zeroing is the worst option").
_MAX_MODELABLE_WC = 0.99


def _config_from_seeds(name: str, pad: str, seeds: dict[str, Any]):
    """A fresh WellConfig from one well's context seeds (sidebar-identical)."""
    from woffl.assembly.network_optimizer import WellConfig

    def f(key: str, default: Optional[float] = None) -> Optional[float]:
        v = seeds.get(key)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return default
        return v if math.isfinite(v) else default

    def s(key: str) -> Optional[str]:
        v = seeds.get(key)
        return str(v) if v else None

    wc_seed = f("form_wc", 0.5)
    if wc_seed >= _MAX_MODELABLE_WC:
        raise ValueError(
            f"water cut {wc_seed:.2f} >= {_MAX_MODELABLE_WC} - not modelable as an "
            "oil producer; mark the well offline (bring-online candidate) instead"
        )

    if f("qwf", 750.0) <= 0:
        raise ValueError("inflow liquid rate must be positive")
    if f("pwf", 500.0) >= f("pres", 1700.0):
        raise ValueError("flowing pressure must be below reservoir pressure")

    from server.services.factories import well_geometry
    profile, _geometry_source = well_geometry(
        name, f("jpump_tvd", 4065.0), str(seeds.get("field_model") or "Schrader"), f("jpump_md")
    )

    return WellConfig(
        well_name=name,
        res_pres=f("pres", 1700.0),
        form_temp=f("form_temp", 120.0),
        jpump_tvd=f("jpump_tvd", 4065.0),
        # Preserve measured MD; otherwise use the same survey/estimated-field
        # TVD crossing as Solver. Never let WellConfig assume MD equals TVD.
        jpump_md=profile.jetpump_md,
        tubing_od=f("tubing_od", 4.5),
        tubing_thickness=f("tubing_thickness", 0.271),
        casing_od=f("casing_od", 6.875),
        casing_thickness=f("casing_thickness", 0.5),
        form_wc=wc_seed,
        form_gor=f("form_gor", 250.0),
        field_model=str(seeds.get("field_model") or "Schrader"),
        hydraulics_model=str(seeds.get("hydraulics_model") or "beggs"),
        surf_pres=f("surf_pres", 210.0),
        qwf=f("qwf", 750.0),
        pwf=f("pwf", 500.0),
        oil_api=f("oil_api"),
        gas_sg=f("gas_sg"),
        wat_sg=f("wat_sg"),
        rho_pf=f("rho_pf", 63.648),
        bubble_point=f("bubble_point"),
        ppf_surf_well=f("ppf_surf"),
        ken_well=f("ken"),
        kth_well=f("kth"),
        kdi_well=f("kdi"),
        mach_crit_well=f("mach_crit"),
        fnz_well=f("nozzle_area_factor"),
        # Installed pump identity from the context's JP-history seed (the
        # cheapest source already in this flow; fail-soft None). fnz_well is
        # wear on THIS pump - _simulate_single_well scales only the matching
        # candidate, never the whole JPCO catalog.
        pump_calibration_scoped=True,
        installed_nozzle=s("nozzle_no"),
        installed_throat=s("area_ratio"),
        jpump_direction=str(seeds.get("jpump_direction") or "reverse"),
        pad=pad,
    )


def _build_configs(
    pads: list[str],
    offline: set[str],
    future: list[schemas.FutureWellSpec],
    note: list[str],
    prov: Optional[dict[str, dict[str, Any]]] = None,
    include_offline: bool = False,
    coverage: Optional[dict[str, dict[str, Any]]] = None,
) -> list[Any]:
    """WellConfigs for every ACTIVE well on ``pads`` + the future wells.

    Seeds come from the context pipeline per well (chars + saved-fit
    overlay), so a run models each well exactly as the Solver opens it.
    Hydration failures are skipped with a note, never fatal.

    ``include_offline=True`` also hydrates the board-offline wells (the CFP
    engine needs them: a store entry flagged offline IS a bring-online
    candidate - review 2026-09-01, SRV-4/OPT-A10). The caller marks them
    ``online=False``; the pad runs keep the default and exclude them.
    """
    universe = wells_svc.list_wells()["wells"]
    by_pad = [w["name"] for w in universe if w.get("pad") in pads]
    donors = {fw.match for fw in future}
    if coverage is not None:
        for row in universe:
            name = row["name"]
            if name in by_pad:
                coverage[name] = {
                    "well": name, "pad": row.get("pad", ""),
                    "role": "offline" if name in offline else "online",
                    "outcome": "offline" if name in offline else "missing_inputs",
                    "reason": ("Offline today; included as a bring-online candidate." if include_offline else "Excluded by the run's offline selection.") if name in offline else "Well inputs have not been loaded.",
                }
        for fw in future:
            coverage[fw.name] = {"well": fw.name, "pad": fw.pad or pads[0],
                "role": "future", "outcome": "missing_inputs", "reason": "Donor inputs unavailable."}

    seeds_by_well: dict[str, dict[str, Any]] = {}
    for name in sorted(set(by_pad) | donors):
        if name in offline and name not in donors and not include_offline:
            continue
        try:
            ctx = wells_svc.well_context(name, 6, 0)
            if ctx.get("geometry_issue"):
                raise ValueError(ctx["geometry_issue"])
            seeds = dict(ctx["seeds"])
            # Measured pump MD rides beside the seeds (not a SimParams field);
            # without it WellConfig models MD = TVD (review 2026-09-01, #2).
            if ctx.get("jpump_md") is not None:
                seeds["jpump_md"] = ctx["jpump_md"]
            seeds_by_well[name] = seeds
            if prov is not None:
                # Where this well's inflow curve came from. The pump the
                # optimizer picks is only as trustworthy as this.
                prov[name] = {
                    "hydraulics_model": seeds.get("hydraulics_model", "beggs"),
                    "physics_model": physics_model(seeds.get("hydraulics_model", "beggs")),
                    "pump_calibration": ctx.get("pump_calibration"),
                    "ipr_source": ctx.get("ipr_source"),
                    "ipr_r2": ctx.get("ipr_r2"),
                    "has_friction": (ctx.get("pump_calibration") or {}).get("status") == "active",
                }
        except Exception as exc:  # noqa: BLE001 - fail-soft per well
            note.append(f"{name}: seeding failed ({exc})")
            if coverage is not None and name in coverage and name not in offline:
                coverage[name].update(outcome="missing_inputs", reason=f"Well inputs unavailable: {exc}")

    configs: list[Any] = []
    pad_of = {w["name"]: w.get("pad", "") for w in universe}
    for name in by_pad:
        if name in offline and not include_offline:
            continue
        seeds = seeds_by_well.get(name)
        if seeds is None:
            continue
        try:
            configs.append(_config_from_seeds(name, pad_of.get(name, ""), seeds))
            if coverage is not None and name not in offline:
                coverage[name].update(outcome="not_evaluated", reason="Model inputs loaded; no operating result yet.")
        except Exception as exc:  # noqa: BLE001
            note.append(f"{name}: invalid model ({exc})")
            if coverage is not None and name not in offline:
                coverage[name].update(outcome="unsupported_model", reason=str(exc))

    for fw in future:
        target_pad = fw.pad.strip().upper() if fw.pad is not None else pads[0]
        if target_pad not in pads:
            note.append(f"{fw.name}: target pad {target_pad} is outside this run - skipped")
            continue
        seeds = seeds_by_well.get(fw.match)
        if seeds is None:
            note.append(f"{fw.name}: donor {fw.match} could not be seeded - skipped")
            continue
        try:
            # A hypothetical well has no survey: it runs on the field preset
            # profile, so the donor's MEASURED MD does not transfer.
            cfg = _config_from_seeds(
                fw.name, target_pad, {k: v for k, v in seeds.items() if k not in {"jpump_md", "nozzle_no", "area_ratio", "ken", "kth", "kdi", "nozzle_area_factor"}}
            )
            configs.append(cfg)
            if coverage is not None:
                coverage[fw.name].update(outcome="not_evaluated", reason="Donor model loaded; no operating result yet.")
            note.append(f"{fw.name}: future well modeled on {fw.match}'s saved fit")
        except Exception as exc:  # noqa: BLE001
            note.append(f"{fw.name}: invalid model ({exc})")
            if coverage is not None:
                coverage[fw.name].update(outcome="unsupported_model", reason=str(exc))
    return configs


def _coverage_summary(ledger: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Account for requested wells without inventing missing rates or loads."""
    accounted = {"modeled", "economic_shut_in", "held_measured", "offline"}
    rows = list(ledger.values())
    unaccounted = [r["well"] for r in rows if r["role"] != "offline" and r["outcome"] not in accounted]
    return {
        "complete": not unaccounted,
        "expected_online": sum(r["role"] == "online" for r in rows),
        "accounted_online": sum(r["role"] == "online" and r["outcome"] in accounted for r in rows),
        "unaccounted_wells": unaccounted,
        "rows": rows,
    }


def _qualify_coverage(meta: dict[str, Any], coverage: dict[str, Any]) -> None:
    """Subset feasibility does not establish whole-pad feasibility."""
    meta["recommendation_status"] = "complete_model_coverage" if coverage["complete"] else "incomplete_exploratory"
    if not coverage["complete"]:
        meta["modeled_subset_feasible"] = meta.get("feasible")
        meta["feasible"] = None


def _unmodeled_pad_row(entry: dict[str, Any]) -> dict[str, Any]:
    """A requested well stays visible even when no WellConfig could be built."""
    empty = ("current_pump", "test_oil", "test_pf", "pump", "pump_state", "oil", "pf",
             "form_water", "suction", "sonic", "marginal_oil", "ipr_source", "ipr_r2",
             "current_model_oil", "current_model_pf", "modeled_hardware_gain")
    return {"well": entry["well"], "outcome": entry["outcome"], "outcome_reason": entry["reason"],
            **dict.fromkeys(empty), "has_friction": False}


def _modeled_current(configs: list[Any], current: dict[str, tuple[str, str]],
                     header: Optional[float], optimizer: Any = None) -> dict[str, dict[str, float]]:
    """Current hardware at the proposal's pressure and saved well inputs.

    Uses the same frozen saved well inputs as the proposal. Measured oil/PF
    never scale this prediction. This isolates the hardware decision; it is
    not a claim about today's production or pressure-change uplift. Unknown
    current hardware is a gap. Reuse the winning grid, then one pooled batch
    for current candidates absent from that grid.
    """
    from woffl.assembly.network_optimizer import NetworkOptimizer, PowerFluidConstraint
    from woffl.assembly.parallelism import worker_ceiling

    if header is None or not math.isfinite(float(header)) or float(header) <= 0:
        return {}
    eligible = []
    for cfg in configs:
        pump = current.get(cfg.well_name)
        if not pump:
            continue
        if (cfg.installed_nozzle, cfg.installed_throat) != pump:
            continue
        clone = copy(cfg)
        clone.ppf_surf_well = float(header)
        eligible.append(clone)
    if not eligible:
        return {}
    get_perf = getattr(optimizer, "get_pump_performance", None)
    perfs = {c.well_name: get_perf(c.well_name, *current[c.well_name], pump_state="installed")
             for c in eligible} if get_perf else {}
    missing = [c for c in eligible if perfs.get(c.well_name) is None]
    if missing:
        pumps = [current[c.well_name] for c in missing]
        opt = NetworkOptimizer(missing,
            PowerFluidConstraint(total_rate=500000.0, pressure=float(header), rho_pf=None),
            sorted({p[0] for p in pumps}), sorted({p[1] for p in pumps}), marginal_watercut=1.0)
        opt.run_all_batch_simulations(max_workers=worker_ceiling())
        perfs.update({c.well_name: opt.get_pump_performance(c.well_name, *current[c.well_name], pump_state="installed") for c in missing})
    out = {}
    for cfg in eligible:
        perf = perfs.get(cfg.well_name)
        if perf is not None:
            out[cfg.well_name] = {"oil": float(perf["oil_rate"]), "pf": float(perf["lift_water"]),
                "form_water": float(perf["formation_water"]), "ppf": float(cfg.ppf_surf_well)}
    return out


def _current_and_tests(
    wells: list[str],
) -> tuple[dict[str, tuple[str, str]], dict[str, tuple[float, Optional[float]]]]:
    """Current pump per well (JP tracker) + median recent test (oil, PF).

    Mirrors pad_helpers.recent_test_rates: the median of up to 5 recent
    positive-oil tests. Fail-soft per well.
    """
    from woffl.assembly.jp_history import get_current_pump

    current: dict[str, tuple[str, str]] = {}
    rates: dict[str, tuple[float, Optional[float]]] = {}

    jp_hist, _source = datasources.jp_history_safe()
    for well in wells:
        if jp_hist is not None:
            try:
                pump = get_current_pump(jp_hist, well)
                nz = str(pump.get("nozzle_no") or "") if pump else ""
                th = str(pump.get("throat_ratio") or "") if pump else ""
                if nz and th:
                    current[well] = (nz, th)
            except Exception:  # noqa: BLE001
                pass
        try:
            df = tests_svc.tests_for_well(well, 6, 0)
            if df is not None and not df.empty and "WtOilVol" in df.columns:
                recent = df.sort_values("WtDate", ascending=False)
                recent = recent[pd.to_numeric(recent["WtOilVol"], errors="coerce") > 0].head(5)
                if not recent.empty:
                    oil = float(pd.to_numeric(recent["WtOilVol"], errors="coerce").median())
                    pf = float(pd.to_numeric(recent["lift_wat"], errors="coerce").median()) if "lift_wat" in recent else float("nan")
                    rates[well] = (oil, pf if math.isfinite(pf) else None)
        except Exception:  # noqa: BLE001
            pass
    return current, rates


# ---------------------------------------------------------------------------
# Pad run (S / I / M)
# ---------------------------------------------------------------------------

# Per-pad engine defaults, mirroring the PadSpec modules
# (s_pad_page n_pump_options=(3,2); m_pad_page (3,2,1), n_steps=9;
# i_pad_page n_pumps=None fixed train, n_steps=11).
_PAD_DEFAULTS = {
    "S": {"n_pumps": 3, "n_steps": 11},
    "I": {"n_pumps": None, "n_steps": 11},
    "M": {"n_pumps": 3, "n_steps": 9},
    # E: single VFD machine, no pump-count choice - the I-Pad shape.
    "E": {"n_pumps": None, "n_steps": 11},
}


def _pad_plant(pad: str):
    """The pad's plant at its DEFAULT configuration.

    Used where there is no run request to configure from (the match-health
    scorecard, the static curve sheet). Pad runs go through
    ``_pad_plant_for_run``, which honours the E-Pad knobs.
    """
    if pad == "S":
        from woffl.gui.s_pad_plant import PLANT
    elif pad == "I":
        from woffl.gui.i_pad_plant import PLANT
    elif pad == "M":
        from woffl.gui.m_pad_plant import PLANT
    elif pad == "E":
        from woffl.gui.e_pad_plant import PLANT
    else:
        raise ValueError(f"unknown pad run '{pad}' - expected S, I, M or E")
    return PLANT


def _pad_plant_for_run(pad: str, req: schemas.OptimizeRunRequest):
    """The plant a RUN uses. Identical to ``_pad_plant`` except on E-Pad,
    whose booster is configured per run: which build is in the ground, its
    suction, its speed cap and the operational header cap are all things the
    engineer sets, because none of them is a measured E-Pad tag (see
    ``woffl/gui/e_pad_plant`` and the E_Pad_Pumps README)."""
    if pad != "E":
        return _pad_plant(pad)
    from woffl.gui.e_pad_plant import EPadPlant

    return EPadPlant(
        req.e_pad_build,
        suction_psi=req.e_pad_suction_psi,
        hz_max=req.e_pad_hz_max,
        max_header_psi=req.e_pad_max_header_psi,
        amp_limit=req.e_pad_amp_limit_a,
    )


def _run_pad_job(job: dict[str, Any], req: schemas.OptimizeRunRequest) -> dict[str, Any]:
    from woffl.gui.pad_optimize import run_choke_optimization, run_optimization

    pad = req.pad or "S"
    defaults = _PAD_DEFAULTS[pad]
    notes: list[str] = []
    prov: dict[str, dict[str, Any]] = {}
    ledger: dict[str, dict[str, Any]] = {}
    configs = _build_configs([pad], set(req.offline), req.future, notes, prov, coverage=ledger)
    if len(configs) == 0:
        coverage = _coverage_summary(ledger)
        meta = {"feasible": None}
        _qualify_coverage(meta, coverage)
        if coverage["complete"]:
            meta["recommendation_status"] = "no_online_wells"
        notes.append(f"No active wells with usable inputs on {pad}-Pad; no optimization was performed.")
        return {"pad": pad, "physics_model": MODEL_VERSION, "n_wells": 0, "meta": meta,
                "notes": notes, "coverage": coverage,
                **({"plan": []} if req.strategy == "choke" else
                   {"rows": [_unmodeled_pad_row(r) for r in ledger.values() if r["role"] != "offline"]})}

    job["progress"] = f"simulating {len(configs)} wells..."

    def cb(step: int, total: int, header: float | None, pf: float, oil: float) -> None:
        header_text = f"header {header:,.0f} psi" if header is not None else "header unavailable"
        job["progress"] = (
            f"trial {step}/{total} - {header_text}"
            + (f", oil {oil:,.0f} BOPD" if oil else "")
        )

    if req.strategy == "choke":
        # Short-term plan: HOLD every installed pump (no JPCO), choke back /
        # shut in wells to fit the (possibly reduced) bank's PF budget. Rows
        # come sorted action-first; provenance rides along like pad rows.
        job["progress"] = "reading current pumps + tests..."
        current, test_rates = _current_and_tests([c.well_name for c in configs])
        # Field-measured suction response (floor/psu_ref/beta per well) -
        # corrects the model's cavitation floor where the gauges contradict
        # it. Strictly fail-soft: an unreachable warehouse degrades to the
        # uncorrected (model-only) run, never to a failed job.
        job["progress"] = "reading pressure history..."
        names = [c.well_name for c in configs]
        res_pres_map = {
            c.well_name: float(c.res_pres)
            for c in configs
            if getattr(c, "res_pres", None) is not None
        }
        try:
            ev = evidence_svc.pad_evidence(names, res_pres_map)
        except Exception as exc:
            ev = None
            notes.append(f"suction evidence unavailable ({exc}); model-only run")
        job["progress"] = f"pricing {len(configs)} wells at ladder pressures..."
        plan, meta = run_choke_optimization(
            configs,
            _pad_plant_for_run(pad, req),
            req.n_pumps if req.n_pumps is not None else defaults["n_pumps"],
            current,
            test_rates,
            n_levels=req.n_steps if req.n_steps is not None else 10,
            progress=cb,
            evidence=ev or None,
        )
        for row in plan:
            row.update(
                prov.get(row["well"])
                or {"ipr_source": None, "ipr_r2": None, "has_friction": False}
            )
            if row["well"] in ledger:
                outcome = ("unsupported_model" if row.get("basis") == "none" else
                           "economic_shut_in" if row.get("action") == "shut" else
                           "held_measured" if row.get("basis") == "test" else "modeled")
                if row.get("basis") == "test" and (test_rates.get(row["well"]) or (None, None))[1] is None:
                    outcome = "unsupported_model"
                ledger[row["well"]].update(outcome=outcome,
                    reason="No modeled or measured operating contribution." if outcome == "unsupported_model" else "Included in the choke plan.")
        for row in plan:
            if row.get("suction_basis") != "evidence":
                continue
            w = row.get("well")
            beta = row.get("response_beta")
            source = row.get("beta_source") or "default"
            n_pairs = (ev or {}).get(w, {}).get("n_pairs", 0)
            floor = row.get("evidence_floor_psi")
            notes.append(
                f"{w}: suction from field data (beta {beta:.2f} {source}, "
                f"{n_pairs} events; floor {floor:.0f} measured vs model)"
                if beta is not None and floor is not None
                else f"{w}: suction from field data"
            )
        coverage = _coverage_summary(ledger)
        _qualify_coverage(meta, coverage)
        return _plain(
            {
                "pad": pad,
                "plan": plan,
                "physics_model": MODEL_VERSION,
                "meta": meta,
                "notes": notes,
                "n_wells": len(configs),
                "coverage": coverage,
            }
        )

    results, _optimizer, meta = run_optimization(
        configs,
        _pad_plant_for_run(pad, req),
        req.n_pumps if req.n_pumps is not None else defaults["n_pumps"],
        req.nozzles,
        req.throats,
        req.method,
        req.marginal_wc,
        n_steps=req.n_steps if req.n_steps is not None else defaults["n_steps"],
        water_price=req.lambda_bopd_per_bpd,
        setpoint_psi=req.setpoint_psi,
        progress=cb,
    )

    job["progress"] = "assembling results..."
    names = [c.well_name for c in configs]
    current, test_rates = _current_and_tests(names)
    job["progress"] = "checking installed pumps at the plan header..."
    try:
        current_model = _modeled_current(configs, current, meta.get("header_psi"), _optimizer)
    except Exception as exc:
        current_model = {}
        notes.append(f"Current-pump counterfactual unavailable ({exc}); no modeled hardware gain reported.")

    chosen = {r.well_name: r for r in results}
    reconciliation = _plain(meta.get("reconciliation")) or []
    reconciled = {r["Well"]: r for r in reconciliation}
    rows: list[dict[str, Any]] = []
    for cfg in configs:
        r = chosen.get(cfg.well_name)
        cur = current.get(cfg.well_name)
        tr = test_rates.get(cfg.well_name)
        rc = reconciled.get(cfg.well_name, {})
        outcome = "modeled" if r else "economic_shut_in" if rc.get("Configs OK", 0) > 0 else "failed_model"
        reason = ("Selected modeled operating point." if r else
                  "Viable pump candidates were not allocated under this run's objective and water budget." if outcome == "economic_shut_in" else
                  rc.get("Detail") or "No usable candidate result; this is not a shut-in recommendation.")
        ledger[cfg.well_name].update(outcome=outcome, reason=reason)
        base = current_model.get(cfg.well_name)
        if ledger[cfg.well_name]["role"] == "future":
            base = {"oil": 0.0, "pf": 0.0, "form_water": 0.0, "ppf": None}
        proposed_oil = r.predicted_oil_rate if r else 0.0 if outcome == "economic_shut_in" else None
        delta = proposed_oil - base["oil"] if base is not None and proposed_oil is not None else None
        if delta is not None and abs(delta) < 1e-8:
            delta = 0.0
        rows.append(
            {
                "well": cfg.well_name,
                "current_pump": f"{cur[0]}{cur[1]}" if cur else None,
                "test_oil": tr[0] if tr else None,
                "test_pf": tr[1] if tr else None,
                "outcome": outcome,
                "outcome_reason": reason,
                "current_model_oil": base["oil"] if base else None,
                "current_model_pf": base["pf"] if base else None,
                "modeled_hardware_gain": delta,
                "pump": f"{r.recommended_nozzle}{r.recommended_throat}" if r else None,
                "pump_state": getattr(r, "pump_state", None) if r else None,
                "oil": proposed_oil,
                "pf": r.allocated_power_fluid if r else 0.0 if outcome == "economic_shut_in" else None,
                "form_water": r.predicted_formation_water if r else 0.0 if outcome == "economic_shut_in" else None,
                "suction": r.suction_pressure if r else None,
                "sonic": bool(r.sonic_status) if r else None,
                "marginal_oil": r.marginal_oil_rate if r else None,
                # Fit provenance: which inflow curve this pump was chosen
                # against, so a saved fit is visibly not a weak auto-fit.
                **(
                    prov.get(cfg.well_name)
                    or {"ipr_source": None, "ipr_r2": None, "has_friction": False}
                ),
            }
        )

    shown = {r["well"] for r in rows}
    for entry in ledger.values():
        if entry["role"] == "offline" or entry["well"] in shown:
            continue
        rows.append(_unmodeled_pad_row(entry))
    coverage = _coverage_summary(ledger)
    _qualify_coverage(meta, coverage)
    comparison_complete = coverage["complete"] and all(r["modeled_hardware_gain"] is not None for r in rows)
    meta["current_model_oil_bopd"] = sum(r["current_model_oil"] for r in rows) if comparison_complete else None
    meta["modeled_hardware_gain_bopd"] = sum(r["modeled_hardware_gain"] for r in rows) if comparison_complete else None
    meta["comparison_basis"] = "Installed and proposed pumps at the same plan header and saved well inputs. This isolates the hardware decision; recent test oil is context, not the modeled baseline. Future wells start offline. Current hardware at this header is a counterfactual, not a separate plant-feasible plan."
    can_stress = (pad in {"I", "M", "E"} and coverage["complete"] and len(configs) <= 100 and
                  meta.get("header_psi") is not None and all(
                      ledger[c.well_name]["role"] == "future" or
                      current.get(c.well_name) == (c.installed_nozzle, c.installed_throat)
                      for c in configs))
    snapshot = None
    if can_stress:
        from server.services.plan_robustness import source_fingerprint
        snapshot = {
            "version": 1, "physics_model": MODEL_VERSION,
            "request": req.model_dump(mode="json"), "configs": [asdict(c) for c in configs],
            "header_psi": meta["header_psi"], "lambda_used": meta.get("lambda_used", 0.0),
            "current": {c.well_name: None if ledger[c.well_name]["role"] == "future" else
                        [*current[c.well_name], "installed"] for c in configs},
            "proposed": {r["well"]: [chosen[r["well"]].recommended_nozzle,
                         chosen[r["well"]].recommended_throat, getattr(chosen[r["well"]], "pump_state", None) or "installed"]
                         if r["well"] in chosen else None for r in rows},
        }
        snapshot["source_fingerprint"] = source_fingerprint(snapshot["configs"])

    keep = (
        "header_psi", "total_pf_bpd", "total_machine_water_bpd", "total_oil_bopd", "n_pumps", "converged",
        "in_range", "recirc", "over_capacity", "feasible", "sweep", "history",
        "marginal_wc_used", "marginal_wc_source", "pf_slack", "parsimony_swaps",
        "lambda_used", "lambda_source", "objective_bopd_equiv", "water_key",
        "solver_agreement",
        "reconciliation", "per_pump_bpd", "station_cap_bpd", "frontier_cap_bpd",
        "amp_limited", "setpoint_psi",
        "curve_header_psi", "coupling_residual_psi", "search_header_psi",
        "qualified_selections", "rejected_selections", "search_scope",
        "recommendation_status", "modeled_subset_feasible", "current_model_oil_bopd", "modeled_hardware_gain_bopd", "comparison_basis",
    )
    return _plain(
        {
            "pad": pad,
            "rows": rows,
            "physics_model": MODEL_VERSION,
            "meta": {k: meta.get(k) for k in keep if k in meta},
            "notes": notes,
            "n_wells": len(configs),
            "coverage": coverage,
            "robustness_available": can_stress,
            "robustness_unavailable_reason": None if can_stress else (
                "S-Pad requires a coupled fixed-curve stress study; controlled-header stress cases support I/M/E only."
                if pad == "S" else "Resolve incomplete coverage/current pump identity, then run again to capture the two fixed plans."),
            "_plan_snapshot": snapshot,
        }
    )


# ---------------------------------------------------------------------------
# CFP run (B / G / C / J against the produced-water plant)
# ---------------------------------------------------------------------------

_CFP_PADS = ["B", "G", "C", "J"]


def _run_cfp_job(job: dict[str, Any], req: schemas.OptimizeRunRequest) -> dict[str, Any]:
    from woffl.gui.cfp_moves import anchor, build_response_surfaces, moves_summary
    from woffl.gui.cfp_pad_plant import PLANT

    notes: list[str] = []
    ledger: dict[str, dict[str, Any]] = {}
    offline = set(req.offline)
    # Stable order: the canonical CFP four first, then any extra non-POPs
    # pads (L, R, ...) in the order given. The schema already rejected POPs
    # pads - their water separates on-pad and never rides the CFP machines.
    sel = list(dict.fromkeys(req.cfp_pads)) or list(_CFP_PADS)
    run_pads = [p for p in _CFP_PADS if p in sel] + [p for p in sel if p not in _CFP_PADS]
    for extra in (p for p in run_pads if p not in _CFP_PADS):
        notes.append(
            f"{extra}-Pad is not plant-supplied (no line-dP entry): PF modeled as "
            f"boosted on-pad at the C-Pad booster knob ({req.c_pad_pf_psi:,.0f} psi); "
            "its produced water still loads the CFP machines"
        )
    # Offline wells are hydrated too: they are the bring-online candidates
    # (marked online=False below). Until 2026-09-01 they were dropped here,
    # so the SI/BOL ladder could never price bringing a shut-in well back on.
    configs = _build_configs(run_pads, offline, req.future, notes, include_offline=True, coverage=ledger)
    if len(configs) == 0:
        raise ValueError(f"no active wells with usable saved fits on pads {', '.join(run_pads)}")

    # Pre-flight the physics invariants WOFFL enforces at model build time -
    # a well with an inconsistent saved fit (pwf >= ResP, or no rate) must be
    # skipped with a note, not allowed to raise mid-sweep and kill the run.
    bad: list[str] = []
    usable: list[Any] = []
    for cfg in configs:
        if cfg.pwf >= cfg.res_pres:
            bad.append(f"{cfg.well_name} (pwf {cfg.pwf:,.0f} >= ResP {cfg.res_pres:,.0f})")
            ledger[cfg.well_name].update(outcome="unsupported_model", reason="Flowing pressure is not below reservoir pressure.")
        elif not cfg.qwf or cfg.qwf <= 0:
            bad.append(f"{cfg.well_name} (no usable test rate)")
            ledger[cfg.well_name].update(outcome="unsupported_model", reason="No usable inflow rate.")
        else:
            usable.append(cfg)
    if bad:
        notes.append("inconsistent saved fit (skipped): " + ", ".join(sorted(bad)))
    configs = usable
    if len(configs) == 0:
        raise ValueError(f"no wells with a consistent saved fit on pads {', '.join(run_pads)}")

    names = [c.well_name for c in configs]
    job["progress"] = "reading current pumps + tests..."
    # Donors may be on a pad outside this run; fetch their tracked pumps as
    # well so future wells can enter the bring-online response surfaces.
    donor_names = [fw.match for fw in req.future]
    current, _rates = _current_and_tests(list(dict.fromkeys(names + donor_names)))

    # Existing wells need a tracked pump to anchor the delta model. Future
    # wells borrow the donor's pump and enter with an offline baseline.
    donors_of_future = {fw.name: fw.match for fw in req.future}
    pad_configs: dict[str, list[Any]] = {}
    online: dict[str, bool] = {}
    skipped: list[str] = []
    for cfg in configs:
        cur = current.get(cfg.well_name) or (
            current.get(donors_of_future.get(cfg.well_name, "")) or None
        )
        if cur is None:
            skipped.append(cfg.well_name)
            ledger[cfg.well_name].update(outcome="missing_inputs", reason="No tracked current pump; no response surface was built.")
            continue
        current[cfg.well_name] = cur
        pad_configs.setdefault(cfg.pad, []).append(cfg)
        # Future wells and board-offline wells enter as bring-online
        # candidates; everything else is online at its current pump.
        online[cfg.well_name] = (
            cfg.well_name not in donors_of_future and cfg.well_name not in offline
        )
    if skipped:
        notes.append("no tracked pump (skipped): " + ", ".join(sorted(skipped)))
    if not pad_configs:
        raise ValueError("no CFP wells with a tracked current pump")

    measured_pad_pf = None
    try:
        from woffl.assembly.pf_pressure import pad_pf_cluster

        from server.services import datasources

        # pad_pf_cluster takes the fleet pf_latest FRAME and returns
        # {pad: {"psi", "n_cluster", ...}}. It used to be called with a pad
        # LETTER, which raised inside this try and was swallowed, so the CFP
        # run never saw a measured header and always fell back to the
        # PAD_LINE_DP table (review 2026-09-01, SRV-15).
        clusters = pad_pf_cluster(datasources.pf_latest_safe())
        measured_pad_pf = {
            pad: float(clusters[pad]["psi"])
            for pad in pad_configs
            if pad != "C" and pad in clusters
        } or None
        if measured_pad_pf:
            notes.append(
                "measured pad PF: "
                + ", ".join(
                    f"{p} {v:,.0f} psi (n={clusters[p]['n_cluster']})"
                    for p, v in sorted(measured_pad_pf.items())
                )
            )
    except Exception as exc:  # noqa: BLE001 - fallback to PAD_LINE_DP inside the engine
        log.warning("measured pad PF unavailable, using PAD_LINE_DP: %s", exc)
        measured_pad_pf = None

    p0 = req.p0_psi
    grid = [float(p) for p in np.linspace(max(p0 - 300.0, 1800.0), 2880.0, 7)]

    def cb(step: int, total: int, pressure: float) -> None:
        job["progress"] = f"response surfaces {step}/{total} - discharge {pressure:,.0f} psi"

    included = {c.well_name for ws in pad_configs.values() for c in ws}
    surfaces = build_response_surfaces(
        pad_configs,
        online,
        {w: current[w] for w in included},
        PLANT,
        p_grid=grid,
        nozzles=req.nozzles,
        throats=req.throats,
        p0=p0,
        c_pad_pf_psi=req.c_pad_pf_psi,
        measured_pad_pf=measured_pad_pf,
        progress=cb,
    )

    job["progress"] = "pricing moves..."
    plant = anchor(surfaces, psi_per_kbpd=req.psi_per_kbpd)
    summary = moves_summary(surfaces, plant)

    # Enrich every single move with its OWN water delta (BWPD at its settled
    # discharge) - the SI/BOL board prices moves in produced water, not just
    # oil, since freeing PW for jet pumps is the whole point of a shut-in.
    from woffl.gui.cfp_moves import option_at

    for m in summary["singles"]:
        ws = surfaces.wells.get(m["well"])
        if ws is None:
            m["own_water_delta"] = None
            continue
        p_after = m["pressure_after"]
        m["own_water_delta"] = (
            option_at(ws, m["to"], p_after)[1] - option_at(ws, m["from"], p_after)[1]
        )

    # Per-well today-vs-plan rows for the results charts (dumbbell + bridge):
    # both states read off the SAME response surfaces at their settled
    # pressures, so the chart can never disagree with the plan numbers.

    baseline = summary["baseline"]
    today_p = summary["today"]["pressure"]
    plan = summary.get("plan") or {}
    plan_choices = plan.get("choices") or baseline
    plan_p = plan.get("pressure", today_p)
    well_rows = []
    for w, ws in surfaces.wells.items():
        b_lab = baseline.get(w)
        p_lab = plan_choices.get(w, b_lab)
        b_oil, b_wat = option_at(ws, b_lab, today_p)
        p_oil, p_wat = option_at(ws, p_lab, plan_p)
        well_rows.append(
            {
                "well": w,
                "pad": ws.pad,
                "online": bool(ws.online),
                "baseline_label": b_lab,
                "plan_label": p_lab,
                "baseline_oil": b_oil,
                "plan_oil": p_oil,
                "baseline_water": b_wat,
                "plan_water": p_wat,
                "changed": p_lab != b_lab,
            }
        )

    for w in surfaces.wells:
        ledger[w].update(outcome="modeled", reason=(
            "Offline baseline; included as a bring-online candidate in the measured-anchor delta model."
            if ledger[w]["role"] == "offline" else "Included in the measured-anchor delta model."))
    coverage = _coverage_summary(ledger)
    if not coverage["complete"]:
        notes.append("Incomplete exploratory CFP comparison: omitted wells' pressure/oil response and possible changes are unknown. The measured pressure anchor is preserved; this is not a complete field recommendation.")

    return _plain(
        {
            "pads": sorted(pad_configs),
            "physics_model": MODEL_VERSION,
            "notes": notes,
            "n_wells": sum(len(v) for v in pad_configs.values()),
            "p0_psi": p0,
            "summary": summary,
            "wells": well_rows,
            "coverage": coverage,
        }
    )
