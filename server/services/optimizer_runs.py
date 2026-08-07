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

import math
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

from server import jobs, schemas
from server.services import datasources, tests as tests_svc, wells as wells_svc

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

# WellConfig hard-raises above this (well_review_store.MAX_MODELABLE_WC).
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

    return WellConfig(
        well_name=name,
        res_pres=f("pres", 1700.0),
        form_temp=f("form_temp", 120.0),
        jpump_tvd=f("jpump_tvd", 4065.0),
        tubing_od=f("tubing_od", 4.5),
        tubing_thickness=f("tubing_thickness", 0.271),
        casing_od=f("casing_od", 6.875),
        casing_thickness=f("casing_thickness", 0.5),
        form_wc=min(f("form_wc", 0.5), _MAX_MODELABLE_WC),
        form_gor=f("form_gor", 250.0),
        field_model=str(seeds.get("field_model") or "Schrader"),
        surf_pres=f("surf_pres", 210.0),
        qwf=f("qwf", 750.0),
        pwf=f("pwf", 500.0),
        oil_api=f("oil_api"),
        gas_sg=f("gas_sg"),
        wat_sg=f("wat_sg"),
        bubble_point=f("bubble_point"),
        ppf_surf_well=f("ppf_surf"),
        ken_well=f("ken"),
        kth_well=f("kth"),
        kdi_well=f("kdi"),
        jpump_direction=str(seeds.get("jpump_direction") or "reverse"),
        pad=pad,
    )


def _build_configs(
    pads: list[str],
    offline: set[str],
    future: list[schemas.FutureWellSpec],
    note: list[str],
    prov: Optional[dict[str, dict[str, Any]]] = None,
) -> list[Any]:
    """WellConfigs for every ACTIVE well on ``pads`` + the future wells.

    Seeds come from the context pipeline per well (chars + saved-fit
    overlay), so a run models each well exactly as the Solver opens it.
    Hydration failures are skipped with a note, never fatal.
    """
    universe = wells_svc.list_wells()["wells"]
    by_pad = [w["name"] for w in universe if w.get("pad") in pads]
    donors = {fw.match for fw in future}

    seeds_by_well: dict[str, dict[str, Any]] = {}
    for name in sorted(set(by_pad) | donors):
        if name in offline and name not in donors:
            continue
        try:
            ctx = wells_svc.well_context(name, 6, 0)
            seeds_by_well[name] = ctx["seeds"]
            if prov is not None:
                # Where this well's inflow curve came from. The pump the
                # optimizer picks is only as trustworthy as this.
                prov[name] = {
                    "ipr_source": ctx.get("ipr_source"),
                    "ipr_r2": ctx.get("ipr_r2"),
                    "has_friction": any(
                        ctx["seeds"].get(k) is not None for k in ("ken", "kth", "kdi")
                    ),
                }
        except Exception as exc:  # noqa: BLE001 - fail-soft per well
            note.append(f"{name}: seeding failed ({exc})")

    configs: list[Any] = []
    pad_of = {w["name"]: w.get("pad", "") for w in universe}
    for name in by_pad:
        if name in offline:
            continue
        seeds = seeds_by_well.get(name)
        if seeds is None:
            continue
        try:
            configs.append(_config_from_seeds(name, pad_of.get(name, ""), seeds))
        except Exception as exc:  # noqa: BLE001
            note.append(f"{name}: invalid model ({exc})")

    for fw in future:
        seeds = seeds_by_well.get(fw.match)
        if seeds is None:
            note.append(f"{fw.name}: donor {fw.match} could not be seeded - skipped")
            continue
        try:
            cfg = _config_from_seeds(fw.name, pads[0], seeds)
            configs.append(cfg)
            note.append(f"{fw.name}: future well modeled on {fw.match}'s saved fit")
        except Exception as exc:  # noqa: BLE001
            note.append(f"{fw.name}: invalid model ({exc})")
    return configs


def _current_and_tests(
    wells: list[str],
) -> tuple[dict[str, tuple[str, str]], dict[str, tuple[float, float]]]:
    """Current pump per well (JP tracker) + median recent test (oil, PF).

    Mirrors pad_helpers.recent_test_rates: the median of up to 5 recent
    positive-oil tests. Fail-soft per well.
    """
    from woffl.assembly.jp_history import get_current_pump

    current: dict[str, tuple[str, str]] = {}
    rates: dict[str, tuple[float, float]] = {}

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
                    pf = float(pd.to_numeric(recent.get("lift_wat"), errors="coerce").median())
                    rates[well] = (oil, pf if math.isfinite(pf) else 0.0)
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
}


def _pad_plant(pad: str):
    if pad == "S":
        from woffl.gui.s_pad_plant import PLANT
    elif pad == "I":
        from woffl.gui.i_pad_plant import PLANT
    elif pad == "M":
        from woffl.gui.m_pad_plant import PLANT
    else:
        raise ValueError(f"unknown pad run '{pad}' - expected S, I or M")
    return PLANT


def _run_pad_job(job: dict[str, Any], req: schemas.OptimizeRunRequest) -> dict[str, Any]:
    from woffl.gui.pad_optimize import run_optimization

    pad = req.pad or "S"
    defaults = _PAD_DEFAULTS[pad]
    notes: list[str] = []
    prov: dict[str, dict[str, Any]] = {}
    configs = _build_configs([pad], set(req.offline), req.future, notes, prov)
    if len(configs) == 0:
        raise ValueError(f"no active wells with usable saved fits on {pad}-Pad")

    job["progress"] = f"simulating {len(configs)} wells..."

    def cb(step: int, total: int, header: float, pf: float, oil: float) -> None:
        job["progress"] = (
            f"trial {step}/{total} - header {header:,.0f} psi"
            + (f", oil {oil:,.0f} BOPD" if oil else "")
        )

    results, _optimizer, meta = run_optimization(
        configs,
        _pad_plant(pad),
        req.n_pumps if req.n_pumps is not None else defaults["n_pumps"],
        req.nozzles,
        req.throats,
        req.method,
        req.marginal_wc,
        n_steps=req.n_steps if req.n_steps is not None else defaults["n_steps"],
        parsimony_bopd=req.parsimony_bopd,
        progress=cb,
    )

    job["progress"] = "assembling results..."
    names = [c.well_name for c in configs]
    current, test_rates = _current_and_tests(names)

    chosen = {r.well_name: r for r in results}
    rows: list[dict[str, Any]] = []
    for cfg in configs:
        r = chosen.get(cfg.well_name)
        cur = current.get(cfg.well_name)
        tr = test_rates.get(cfg.well_name)
        rows.append(
            {
                "well": cfg.well_name,
                "current_pump": f"{cur[0]}{cur[1]}" if cur else None,
                "test_oil": tr[0] if tr else None,
                "test_pf": tr[1] if tr else None,
                "pump": f"{r.recommended_nozzle}{r.recommended_throat}" if r else None,
                "oil": r.predicted_oil_rate if r else None,
                "pf": r.allocated_power_fluid if r else None,
                "form_water": r.predicted_formation_water if r else None,
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

    keep = (
        "header_psi", "total_pf_bpd", "total_oil_bopd", "n_pumps", "converged",
        "in_range", "recirc", "over_capacity", "feasible", "sweep", "history",
        "marginal_wc_used", "marginal_wc_source", "pf_slack", "parsimony_swaps",
        "reconciliation", "per_pump_bpd", "station_cap_bpd", "frontier_cap_bpd",
        "amp_limited",
    )
    return _plain(
        {
            "pad": pad,
            "rows": rows,
            "meta": {k: meta.get(k) for k in keep if k in meta},
            "notes": notes,
            "n_wells": len(configs),
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
    configs = _build_configs(run_pads, offline, req.future, notes)
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
        elif not cfg.qwf or cfg.qwf <= 0:
            bad.append(f"{cfg.well_name} (no usable test rate)")
        else:
            usable.append(cfg)
    if bad:
        notes.append("inconsistent saved fit (skipped): " + ", ".join(sorted(bad)))
    configs = usable
    if len(configs) == 0:
        raise ValueError(f"no wells with a consistent saved fit on pads {', '.join(run_pads)}")

    names = [c.well_name for c in configs]
    job["progress"] = "reading current pumps + tests..."
    current, _rates = _current_and_tests(names)

    # Wells without a tracked current pump cannot anchor a delta model -
    # mirror of the Streamlit page skipping unreviewed pumps. Future wells
    # take their donor's... no: future wells have no current pump either;
    # they enter as bring-online candidates (online=False) with a nominal
    # current size = their donor-derived first candidate is not defined, so
    # they are skipped here too and noted. Bring-online planning for future
    # wells lands with the donor-pump enhancement.
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
            continue
        current[cfg.well_name] = cur
        pad_configs.setdefault(cfg.pad, []).append(cfg)
        # Future wells and board-offline wells enter as bring-online
        # candidates; everything else is online at its current pump.
        online[cfg.well_name] = cfg.well_name not in donors_of_future
    if skipped:
        notes.append("no tracked pump (skipped): " + ", ".join(sorted(skipped)))
    if not pad_configs:
        raise ValueError("no CFP wells with a tracked current pump")

    measured_pad_pf = None
    try:
        from woffl.assembly.pf_pressure import pad_pf_cluster

        measured_pad_pf = {
            pad: pad_pf_cluster(pad) for pad in pad_configs if pad != "C"
        }
        measured_pad_pf = {p: v for p, v in measured_pad_pf.items() if v}
    except Exception:  # noqa: BLE001 - fallback to PAD_LINE_DP inside the engine
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

    return _plain(
        {
            "pads": sorted(pad_configs),
            "notes": notes,
            "n_wells": sum(len(v) for v in pad_configs.values()),
            "p0_psi": p0,
            "summary": summary,
            "wells": well_rows,
        }
    )
