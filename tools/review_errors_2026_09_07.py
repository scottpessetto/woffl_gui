"""Offline reproductions for docs/code_review_2026-09-07.md.

Run from the repository root with PYTHONPATH=. using the project venv.
No production writes or warehouse reads. These observations document bugs;
they are not acceptance tests asserting that the current behavior is correct.
"""

import ast
import argparse
import builtins
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import pandas as pd
import pytest


def mixture_conservation():
    from woffl.flow.singlephase import ft3s_to_bpd
    from woffl.pvt import BlackOil, FormGas, FormWater, ResMix

    mix = ResMix(.8, 250, BlackOil.schrader(), FormWater.schrader(), FormGas.schrader())
    rows = []
    for p in (0., 500., 1000., 2000., 3000.):
        mix.condition(p, 100.)
        oil, water, gas = mix.insitu_volm_flow(100.)
        rows.append(dict(pressure=p, water_bpd=ft3s_to_bpd(water),
                         expected_water_bpd=400., oil_rbpd=ft3s_to_bpd(oil),
                         oil_from_fvf=100. * mix.oil.oil_fvf(),
                         mass_lbm_s=sum((oil, water, gas)) * mix.rho_mix()))
    return rows


def saved_watercut():
    from tests.test_web_wells_context import _saved, context
    from server.services.optimizer_runs import _config_from_seeds

    rows = []
    with pytest.MonkeyPatch.context() as mp:
        saved, run = context.__wrapped__(mp)
        for wc in (.974, .986):
            info = _saved({})
            info.update(values=dict(qwf_liq=1000., pwf=500., res_pres=1700., form_wc=wc),
                        saved_at=pd.Timestamp("2026-09-07"))
            saved["MPB-28"] = info
            ctx = run()
            seeds = ctx["seeds"]
            try:
                _config_from_seeds("MPB-28", "B", seeds)
                accepted = True
            except ValueError:
                accepted = False
            rows.append(dict(saved_wc=wc, restored_wc=seeds["form_wc"],
                             saved_oil=1000*(1-wc), restored_oil=seeds["qwf"]*(1-seeds["form_wc"]),
                             optimizer_accepts=accepted, reported_clamps=ctx.get("clamped")))
    return rows


def batch_wear():
    from server import schemas
    from server.services.solve import run_batch
    from woffl.assembly.batchpump import BatchPump
    from woffl.assembly.network_optimizer import WellConfig, _simulate_single_well
    from woffl.geometry.jetpump import JetPump

    captured = []

    class Captured(Exception):
        pass

    def capture(self, pumps, *args, **kwargs):
        captured.append({jp.noz_no + jp.rat_ar + ":" + getattr(jp, "pump_state", "installed"): jp.anz / JetPump(jp.noz_no, jp.rat_ar).anz
                         for jp in pumps})
        raise Captured()

    with patch.object(BatchPump, "batch_run", capture):
        try:
            run_batch("Custom", schemas.SimParams(nozzle_area_factor=1.2,
                      nozzle_batch_options=["12", "13"], throat_batch_options=["B"]))
        except Captured:
            pass
        cfg = WellConfig(well_name="Custom", res_pres=1700., form_temp=70., jpump_tvd=4065.,
                         fnz_well=1.2, installed_nozzle="12", installed_throat="B", pump_calibration_scoped=True)
        try:
            _simulate_single_well(cfg, 3168., ["12", "13"], ["B"])
        except Captured:
            pass
    return dict(single_well_batch=captured[0], network_batch=captured[1])


def calibration_fallback():
    from server.services import event_calibration as ec
    from server.services.optimizer_runs import _config_from_seeds
    from woffl.gui import fric_calibration as fc

    cfg = _config_from_seeds("Custom", "M", dict(ken=.07, kth=.6, kdi=.7,
                            nozzle_area_factor=1.2, mach_crit=1.5))
    got = {}

    def capture(**kwargs):
        got.update(kwargs)
        return NS(best_ken=.07, best_kth=.6, best_kdi=.7, best_modeled_bhp=500.,
                  target_bhp=500., match_quality="good", message="probe")

    with patch.object(ec, "_latest_test_target", return_value=dict(bhp=500., date="2020-01-01")), \
         patch.object(fc, "_build_well_objects", return_value=(None,)*5), \
         patch.object(fc, "calibrate_friction_coefs", capture):
        ec._single_point_fallback({}, "Custom", cfg, "12", "B")
    return {k: got.get(k, "not forwarded") for k in
            ("seed_kth", "seed_kdi", "nozzle_area_factor", "mach_crit")}


def cfp_settling():
    from woffl.gui.cfp_moves import AnchoredPlant, Surfaces, WellSurface, settle

    grid = [2500., 2880.]
    ws = WellSurface("W", "B", True, "12B", {
        "12B": dict(_grid=grid, oil=[100., 150.], water=[31000., 69000.])})
    surfaces = Surfaces(grid, 2600., {"W": ws})
    plant = AnchoredPlant(2600., 40000., p_floor=2500.)
    state = settle({"W": "12B"}, surfaces, plant)
    actual_pressure, _ = plant.pressure_at(state["water"])
    return dict(reported_pressure=state["pressure"], feasible=state["feasible"],
                pressure_from_reported_water=actual_pressure,
                residual_psi=actual_pressure-state["pressure"])


def auto_price():
    from woffl.assembly.optimization_algorithms import derive_lambda, milp_optimization

    df = pd.DataFrame(dict(nozzle=["12", "13"], throat=["B", "B"],
                           qoil_std=[80., 100.], lift_wat=[80., 100.]))
    opt = NS(wells=[NS(well_name="W")], batch_results={"W": NS(df=df)},
             power_fluid=NS(total_rate=90.))

    def perf(well, nozzle, throat):
        rate = 80. if nozzle == "12" else 100.
        return dict(oil_rate=rate, lift_water=rate, formation_water=0., total_water=rate,
                    suction_pressure=900., sonic_status=False, mach_te=.5,
                    marginal_oil_lift_water=1.)

    opt.get_pump_performance = perf
    opt.water_price = derive_lambda(opt.batch_results, 90.)[0]
    auto_oil = sum(r.predicted_oil_rate for r in milp_optimization(opt))
    price = opt.water_price
    opt.water_price = 0.
    return dict(derived_lambda=price, auto_oil=auto_oil,
                unpriced_oil=sum(r.predicted_oil_rate for r in milp_optimization(opt)))


def geometry():
    from server import schemas
    from woffl.geometry.pipe import Pipe, PipeInPipe
    try:
        sp = schemas.SimParams(tubing_od=4.5, casing_od=5.5, casing_thickness=.5)
    except ValueError:
        return dict(api_accepts=False)
    p = PipeInPipe(Pipe(sp.tubing_od, sp.tubing_thickness),
                   Pipe(sp.casing_od, sp.casing_thickness))
    return dict(api_accepts=True, annulus_area=p.ann_area, annulus_hyd_dia=p.ann_hyd_dia)


def evidence_reference():
    from woffl.assembly.network_optimizer import WellConfig
    from woffl.gui.pad_optimize import _apply_suction_evidence

    cfg = WellConfig(well_name="W", res_pres=1700., form_temp=70., jpump_tvd=4065.)
    ev = {"W": dict(floor=450., floor_source="era", psu_ref=500., ppf_ref=3200., beta=.1, beta_source="well")}
    rows = []
    for top in (3200., 3500.):
        grid = [{"W": (100., 1000., 600., True)}, {"W": (100., 1200., 600., True)}]
        _apply_suction_evidence(grid, [3000., top], ["W"], ev, {"W": cfg})
        rows.append(dict(sweep_top=top, evaluated_pressure=3000., corrected_bhp=grid[0]["W"][2]))
    return rows


def retry_after_commit():
    from woffl.assembly import databricks_client as dc

    calls = []

    class Cursor:
        def close(self):
            if len(calls) == 1:
                raise ConnectionError("synthetic close failure after successful execution")

    class Connection:
        def cursor(self):
            return Cursor()

        def close(self):
            pass

    def committed(cursor):
        calls.append("simulated commit")
        return 1

    # Exercise the common retry helper with fake connections only. Neither
    # execute_write nor a production write gate is used or enabled.
    with patch.object(dc, "_CONN_LOCAL", NS(conn=None)), \
         patch.object(dc, "_new_connection", return_value=Connection()), \
         patch.object(dc, "_TOKEN_CACHE", {"token": None}):
        dc._execute_via_connector(committed)
    return dict(successful_executions=len(calls), expected_executions=1)


def wear_after_replacement():
    from tests.test_web_wells_context import context
    from woffl.assembly import jp_history
    from woffl.gui.ipr_anchor import _assemble_saved_ipr
    from server.services import wells

    old = _assemble_saved_ipr({"jpfric_nozzle_area": dict(prop_value=1.2,
                              entry_datetime=pd.Timestamp("2025-01-01"), entry_user="probe")})
    with pytest.MonkeyPatch.context() as mp:
        saved, run = context.__wrapped__(mp)
        saved["MPB-28"] = old
        mp.setattr(wells.datasources, "jp_history_safe", lambda: (pd.DataFrame(), "probe"))
        mp.setattr(jp_history, "get_current_pump", lambda *args: dict(nozzle_no="13", throat_ratio="B",
                   date_set=pd.Timestamp("2026-09-01")))
        ctx = run()
    return dict(wear_saved="2025-01-01", new_pump_set="2026-09-01",
                new_pump=ctx["seeds"].get("nozzle_no"),
                restored_area_factor=ctx["seeds"].get("nozzle_area_factor"))


def syntax_inventory():
    root = Path(__file__).resolve().parents[1]
    paths = [p for folder in ("woffl", "server", "tests", "tools", "scripts")
             for p in (root / folder).rglob("*.py")]
    for path in paths:
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    return dict(parsed_python_files=len(paths),
                python_lines=sum(len(p.read_text(encoding="utf-8-sig").splitlines()) for p in paths))


def multipart_dependency():
    from fastapi.dependencies.utils import ensure_multipart_is_installed

    original_import = builtins.__import__

    def without_multipart(name, *args, **kwargs):
        if name.split(".")[0] in ("python_multipart", "multipart"):
            raise ImportError("simulate clean environment without undeclared multipart")
        return original_import(name, *args, **kwargs)

    requirements = (Path(__file__).resolve().parents[1] / "requirements.txt").read_text()
    with patch("builtins.__import__", without_multipart):
        try:
            ensure_multipart_is_installed()
        except RuntimeError as exc:
            return dict(declared="python-multipart" in requirements, startup_guard=str(exc))
    return dict(declared="python-multipart" in requirements, startup_guard="passed")


def choke_machine_budget():
    from tests.test_pad_optimize import FreePlant
    from woffl.gui import pad_optimize as po
    from woffl.assembly.network_optimizer import WellConfig

    class TotalWaterPlant(FreePlant):
        water_key = "totl_wat"

        def budget_at_pressure(self, pressure, n_pumps=None):
            return 1000.

    cfg = WellConfig(well_name="W", res_pres=1700., form_temp=70.,
                     jpump_tvd=4065., form_wc=.9)
    with patch.object(po, "_model_at_forced_header", return_value={"W": (100., 900., 500., False)}):
        rows, meta = po.run_choke_optimization([cfg], TotalWaterPlant(), None,
                                             {"W": ("12", "B")}, {"W": (100., 900.)}, n_levels=2)
    oil = sum(r["oil"] for r in rows)
    pf = sum(r["pf"] for r in rows)
    return dict(capacity_bpd=1000., selected_pf_bpd=pf,
                selected_total_water_bpd=pf + oil * cfg.form_wc / (1-cfg.form_wc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    observations = {}
    for probe in (syntax_inventory, mixture_conservation, saved_watercut, batch_wear,
                  calibration_fallback, cfp_settling, auto_price, geometry,
                  evidence_reference, retry_after_commit, wear_after_replacement,
                  multipart_dependency, choke_machine_budget):
        try:
            observations[probe.__name__] = probe()
        except Exception as exc:
            observations[probe.__name__] = dict(probe_error=f"{type(exc).__name__}: {exc}")
    rendered = json.dumps(observations, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
