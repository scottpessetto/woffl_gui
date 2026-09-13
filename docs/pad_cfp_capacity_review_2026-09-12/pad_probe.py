"""Offline audit: real pad curves/allocators, deterministic synthetic well responses.

Only batch simulation is patched. No database, pool, saved inputs or source edits.
Run with PYTHONPATH=. and WOFFL_MAX_WORKERS=1 from repository root.
"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json

import pandas as pd

from woffl.assembly.network_optimizer import NetworkOptimizer, WellConfig
from woffl.gui.pad_optimize import run_optimization
from woffl.gui.pad_plant_base import SPadPlant, IPadPlant, MPadPlant
from woffl.gui.e_pad_plant import EPadPlant

ROOT = Path(__file__).resolve().parent


def synthetic_case(plant, count, water_each, *, pin=None, n_pumps=None, increasing_oil=False,
                   upper_solving_pressure=None):
    configs = [WellConfig(well_name=f"SYN-{i}", res_pres=1500, form_temp=100,
                          jpump_tvd=4000, qwf=500, pwf=500, form_wc=.5,
                          use_survey=False) for i in range(count)]

    def run_batch(opt, max_workers=None):
        opt.batch_results = {}
        for i, wc in enumerate(opt.wells):
            oil = (400 + (opt.power_fluid.pressure - 3000) * .2 if increasing_oil
                   else 400 - i * 10)
            form = oil * wc.form_wc / (1-wc.form_wc)
            lift = water_each - form if plant.water_key == "totl_wat" else water_each
            row = dict(nozzle="12", throat="B", error="na", qoil_std=oil,
                       lift_wat=lift, form_wat=form, totl_wat=lift+form,
                       psu_solv=500., sonic_status=False, mach_te=.5)
            if upper_solving_pressure is not None and opt.power_fluid.pressure > upper_solving_pressure:
                row.update(error="synthetic high-pressure failed solve", qoil_std=float("nan"),
                           lift_wat=float("nan"), form_wat=float("nan"), totl_wat=float("nan"))
            opt.batch_results[wc.well_name] = SimpleNamespace(df=pd.DataFrame([row]))
        return opt.batch_results

    with patch.object(NetworkOptimizer, "run_all_batch_simulations", run_batch):
        results, _, meta = run_optimization(
            configs, plant, n_pumps, ["12"], ["B"], "milp", None,
            water_price=0., n_steps=3, refine_rounds=0, setpoint_psi=pin)
    header = meta["header_psi"]
    draw = meta["total_machine_water_bpd"]
    delivered, over = plant.delivered_header(draw, header, n_pumps)
    return dict(expected=count, selected=len(results), selected_names=[r.well_name for r in results],
                header_psi=header, machine_water=draw, lift_water=meta["total_pf_bpd"],
                budget=meta.get("frontier_cap_bpd", meta.get("station_cap_bpd")),
                oil=meta["total_oil_bopd"], feasible=meta.get("feasible"),
                qualified=meta.get("qualified_selections"), rejected=meta.get("rejected_selections"),
                converged=meta["converged"], in_range=meta["in_range"],
                actual_delivered_header=delivered, actual_over_capacity=over,
                closure_error_psi=None if delivered is None else delivered-header,
                sweep=meta["sweep"])


def main():
    plants = {"S": (SPadPlant(), 3), "I": (IPadPlant(), None),
              "M": (MPadPlant(), 3), "E": (EPadPlant(), None)}
    out = {"assumptions": "Real plant curves and real MILP; synthetic constant machine-water well responses; no field qualification.",
           "plant_windows": {}, "cases": {}}
    for name, (plant, count) in plants.items():
        lo, hi = plant.pressure_window(count)
        out["plant_windows"][name] = dict(flow=plant.flow_window(count), pressure=[lo, hi],
            budget_at_floor=plant.budget_at_pressure(lo, count),
            budget_at_ceiling=plant.budget_at_pressure(hi, count), water_key=plant.water_key)
        out["cases"][name+"_over_capacity"] = synthetic_case(plant, 6, 20000., n_pumps=count)
    out["cases"]["E_low_branch_pinned"] = synthetic_case(EPadPlant(), 1, 4000., pin=3500.)
    out["cases"]["E_low_branch_swept"] = synthetic_case(EPadPlant(), 1, 4000., increasing_oil=True)
    out["cases"]["S_below_minimum"] = synthetic_case(SPadPlant(), 1, 20000., n_pumps=3)
    out["cases"]["S_failed_far_bracket"] = synthetic_case(SPadPlant(), 1, 40000., n_pumps=3,
                                                           upper_solving_pressure=3600.)
    out["cases"]["E_suction_above_cap"] = synthetic_case(EPadPlant(suction_psi=3600.), 1, 20000., pin=4000.)
    e = EPadPlant(amp_limit=50)
    q = 18000.
    out["E_amp_report"] = dict(flow=q, actual_frontier=e.max_discharge_pressure(q),
                               envelope=e.envelope([q], at_pressure=3400)[0])
    (ROOT / "pad_probe.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
