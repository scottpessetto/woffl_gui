"""Offline investigation of a recorded BHP plot, with pump losses held fixed.

Run from the repository with PYTHONPATH=.; uses JSON and local surveys only.
Same-test conditioning and hypothetical perturbations are diagnostics, never
independent accuracy scores. No app/physics defaults or saved fits are changed.
"""
import argparse
from contextlib import ExitStack
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
from unittest.mock import patch

import numpy as np

from server.services.fleet_validation import metrics
from woffl.assembly.network_optimizer import NetworkOptimizer, WellConfig
from woffl.assembly import solopump
from woffl.flow import jetflow, singlephase
from woffl.flow.entry_energy import MODEL_VERSION
from woffl.geometry import JetPump


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "baseline": "Replay original frozen config at measured surface pressures",
    "test_composition": "Measured WC/GOR; preserve original oil IPR exactly",
    "test_oil_inflow": "Measured oil/BHP IPR anchor; original WC/GOR",
    "test_state": "Measured oil/BHP anchor and WC/GOR; diagnostic only",
    "gor_minus20": "Test state with GOR multiplied by 0.8; hypothetical",
    "gor_plus20": "Test state with GOR multiplied by 1.2; hypothetical",
    "temp_minus20": "Test state with uniform temperature reduced 20 F; hypothetical",
    "temp_plus20": "Test state with uniform temperature increased 20 F; hypothetical",
    "pf_density_minus2": "Test state with standard PF density reduced 2%; hypothetical",
    "pf_density_plus2": "Test state with standard PF density increased 2%; hypothetical",
    "return_mesh25": "Test state with 25 ft return-column grid instead of 100 ft",
    "segmented_pf": "Test state with isothermal, pressure-dependent PF traverse",
    "return_no_slip": "Test state with BB holdup forced to no-slip; structural probe, not a qualified correlation",
    "return_no_payne": "Test state without Payne holdup correction; structural probe",
}


def config_for(well, variant):
    cfg = deepcopy(well["config"])
    obs = well["latest_test"]
    old_oil_anchor = cfg["qwf"] * (1-cfg["form_wc"])
    if variant not in {"baseline", "test_oil_inflow"}:
        cfg["form_wc"] = obs["wc"]
        cfg["form_gor"] = obs["fgor"]
        cfg["qwf"] = old_oil_anchor / (1-cfg["form_wc"])
    if variant not in {"baseline", "test_composition"}:
        cfg["qwf"] = obs["observed_oil"] / (1-cfg["form_wc"])
        cfg["pwf"] = obs["observed_bhp"]
    if variant.startswith("gor_"):
        cfg["form_gor"] *= .8 if variant == "gor_minus20" else 1.2
    if variant.startswith("temp_"):
        cfg["form_temp"] += -20 if variant == "temp_minus20" else 20
    if variant.startswith("pf_density_"):
        cfg["rho_pf"] *= .98 if variant == "pf_density_minus2" else 1.02
    return WellConfig(**cfg)


def objects(cfg):
    bore, profile, ipr, mix, pf = NetworkOptimizer._create_well_objects(cfg)
    pump = JetPump(cfg.installed_nozzle, cfg.installed_throat,
                   ken=.03 if cfg.ken_well is None else cfg.ken_well,
                   kth=.3 if cfg.kth_well is None else cfg.kth_well,
                   kdi=.4 if cfg.kdi_well is None else cfg.kdi_well)
    pump.dnz *= math.sqrt(1. if cfg.fnz_well is None else cfg.fnz_well)
    return pump, bore, profile, ipr, mix, pf


def segmented_pf(qguess, pte, ptop, temp, _static, pump, bore, profile, pf, flowpath):
    """Investigation-only midpoint PF traverse; retain nozzle and loss equations."""
    if flowpath == "tubing":
        diameter, area, rough = bore.tube_hyd_dia, bore.tube_area, bore.tube_abs_ruff
    else:
        diameter, area, rough = bore.ann_hyd_dia, bore.ann_area, bore.ann_abs_ruff
    md, tvd = profile.outflow_spacing(100.)
    pressure = ptop
    for length, height in zip(np.diff(md), np.diff(tvd)):
        next_pressure = pressure
        for _ in range(3):
            pf.condition((pressure+next_pressure)/2, temp)
            velocity = singlephase.velocity(singlephase.bpd_to_ft3s(qguess)*pf.volume_factor(), area)
            reynolds = singlephase.reynolds(pf.density, velocity, diameter, pf.viscosity)
            factor = singlephase.ffactor_darcy(reynolds, singlephase.relative_roughness(diameter, rough))
            friction = singlephase.diff_press_friction(factor, pf.density, velocity, diameter, length)
            next_pressure = pressure + pf.density*height/144 - friction
        pressure = next_pressure
    velocity, rate = jetflow.water_nozzle(pressure, pte, temp, pump.knz, pump.anz, pf)
    return qguess-rate, velocity, pressure


def scored(result, obs):
    bhp, sonic, oil, water, pf, mach = result
    row = dict(predicted_bhp=float(bhp), sonic=bool(sonic), predicted_oil=float(oil),
               predicted_formation_water=float(water), predicted_pf=float(pf), mach=float(mach),
               bhp_error=float(bhp-obs["observed_bhp"]),
               oil_error_bopd=float(oil-obs["observed_oil"]),
               oil_error_pct=float(100*(oil/obs["observed_oil"]-1)))
    if obs.get("observed_pf"):
        row["pf_error_pct"] = float(100*(pf/obs["observed_pf"]-1))
    return row


def run_variant(well, variant):
    try:
        cfg = config_for(well, variant)
        pump, bore, profile, ipr, mix, pf = objects(cfg)
        obs = well["latest_test"]
        with ExitStack() as stack:
            if variant == "return_mesh25":
                spacing = profile.outflow_spacing
                stack.enter_context(patch.object(profile, "outflow_spacing", lambda _: spacing(25.)))
            if variant == "segmented_pf":
                stack.enter_context(patch.object(solopump, "powerfluid_residual", segmented_pf))
            if variant == "return_no_slip":
                stack.enter_context(patch.object(solopump.of.tp, "beggs_holdup_inc", lambda nslh, *_: nslh))
            if variant == "return_no_payne":
                stack.enter_context(patch.object(solopump.of.tp, "payne_correction", lambda holdup, _: holdup))
            result = solopump.jetpump_solver(obs["pwh"], cfg.form_temp, obs["ppf"],
                                            pump, bore, profile, ipr, mix, pf, cfg.jpump_direction)
        return scored(result, obs)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def trace_at(well, suction):
    """Expose component balances with the measured oil/BHP anchor, no refit."""
    cfg = config_for(well, "test_state")
    pump, bore, profile, ipr, mix, pf = objects(cfg)
    obs = well["latest_test"]
    trace = dict(suction=float(suction), pump_tvd=float(profile.jetpump_vd),
                 configured_tvd=cfg.jpump_tvd, nozzle_area=pump.anz, entry_area=pump.ate,
                 return_static_with_acceleration=0., return_friction_with_acceleration=0.)
    original_pf = solopump.powerfluid_residual
    original_throat = jetflow.throat_discharge
    original_diffuser = jetflow.diffuser_discharge
    original_segment = solopump.of.beggs_diff_press
    original_return = solopump.of.production_top_down_press

    def traced_pf(*args):
        result = original_pf(*args)
        trace.update(pte=float(args[1]), pni=float(result[2]),
                     pf_static=float(-args[4]), pf_friction=float(args[2]-args[4]-result[2]),
                     nozzle_rate_residual=float(result[0]))
        return result

    def traced_throat(*args):
        value = original_throat(*args)
        trace["ptm"] = float(value)
        return value

    def traced_diffuser(*args):
        result = original_diffuser(*args)
        trace["pdi_pump"] = float(result[1])
        return result

    def traced_segment(*args):
        result = original_segment(*args)
        trace["return_static_with_acceleration"] -= float(result[0])
        trace["return_friction_with_acceleration"] -= float(result[1])
        return result

    def traced_return(*args):
        result = original_return(*args)
        trace["pdi_return"] = float(result[1][-1])
        return result

    try:
        with patch.object(solopump, "powerfluid_residual", traced_pf), \
             patch.object(jetflow, "throat_discharge", traced_throat), \
             patch.object(jetflow, "diffuser_discharge", traced_diffuser), \
             patch.object(solopump.of, "beggs_diff_press", traced_segment), \
             patch.object(solopump.of, "production_top_down_press", traced_return):
            result = solopump.discharge_residual(suction, obs["pwh"], cfg.form_temp,
                         obs["ppf"], pump, bore, profile, ipr, mix, pf, cfg.jpump_direction)
        trace.update(discharge_residual=float(result[0]), oil=float(result[1]),
                     formation_water=float(result[2]), pf=float(result[3]), mach=float(result[4]))
    except Exception as exc:
        trace["error"] = f"{type(exc).__name__}: {exc}"
    return trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT/"docs/fleet_actuality_2026-09-08.json")
    parser.add_argument("--output", type=Path, default=ROOT/"docs/bhp_model_diagnostic_2026-09-11.json")
    args = parser.parse_args()
    source = json.loads(args.source.read_text(encoding="utf-8"))
    report = dict(model_version=MODEL_VERSION, source=args.source.name,
        source_sha256=sha256(args.source.read_bytes()).hexdigest(), snapshot_time=source["snapshot_time"],
        live_queries=0, refitted=False, scope="Frozen retrospective and same-test diagnostics, not field validation",
        variants=VARIANTS, wells=[])
    for well in source["wells"]:
        if not well.get("latest_test"):
            continue
        row = dict(well=well["well"], observed=well["latest_test"], results={})
        for variant in VARIANTS:
            row["results"][variant] = run_variant(well, variant)
        state = row["results"]["test_state"]
        row["at_observed_bhp"] = trace_at(well, well["latest_test"]["observed_bhp"])
        if "predicted_bhp" in state:
            row["at_test_conditioned_prediction"] = trace_at(well, state["predicted_bhp"])
        report["wells"].append(row)
        print(f"{well['well']}: {len(row['results'])} diagnostic variants", flush=True)
    report["metrics"] = {variant: metrics([r["results"][variant] for r in report["wells"]]) for variant in VARIANTS}
    differences = [r["results"]["baseline"]["predicted_bhp"]-r["observed"]["predicted_bhp"]
                   for r in report["wells"] if "predicted_bhp" in r["results"]["baseline"] and "predicted_bhp" in r["observed"]]
    report["baseline_replay_max_abs_difference_psi"] = max(map(abs, differences), default=None)
    report["code_sha256"] = {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p.read_bytes()).hexdigest()
        for folder in ("woffl/flow", "woffl/pvt", "woffl/geometry") for p in sorted((ROOT/folder).glob("*.py"))}
    for rel in ("woffl/assembly/solopump.py", "woffl/assembly/network_optimizer.py", "woffl/assembly/sim_factories.py", "tools/bhp_model_diagnostic.py"):
        report["code_sha256"][rel] = sha256((ROOT/rel).read_bytes()).hexdigest()
    report["survey_sha256"] = {r["well"]: sha256(p.read_bytes()).hexdigest()
        for r in report["wells"] if (p := ROOT/"woffl/jp_data/well_surveys"/f"{r['well']} Deviation Survey.csv").exists()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    plot(report, args.output.with_suffix(".png"))
    for variant, values in report["metrics"].items():
        error = values.get("bhp_error", {})
        print(f"{variant}: solved={values['solved']}, median_abs={error.get('median_abs', 0):.2f}, rms={error.get('rms', 0):.2f}")


def plot(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    good = [w for w in report["wells"] if "predicted_bhp" in w["results"]["baseline"]]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.8), gridspec_kw={"width_ratios": [1, .85, 1.15]})
    fig.suptitle("BHP investigation: pump loss coefficients held fixed", fontsize=17, fontweight="bold", y=.98)
    fig.text(.5, .93, "September 8 frozen observations | 34 of 35 wells solved | same-test conditioning is diagnostic, not independent validation", ha="center", fontsize=10)
    ax = axes[0]
    for w in good:
        observed = w["observed"]["observed_bhp"]
        baseline = w["results"]["baseline"]["predicted_bhp"]
        conditioned = w["results"]["test_state"]["predicted_bhp"]
        ax.plot([observed, observed], [baseline, conditioned], color="#b6bbc4", lw=1)
        ax.scatter(observed, baseline, s=28, c="#e78b37", zorder=3)
        ax.scatter(observed, conditioned, s=30, marker="x", c="#2369a4", zorder=4)
        if w["well"] in {"MPB-35", "MPJ-29", "MPE-48", "MPE-42", "MPM-62"}:
            ax.annotate(w["well"].replace("MP", ""), (observed, baseline), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.plot([0, 2100], [0, 2100], color="#737b87", ls="--", lw=1)
    ax.scatter([], [], c="#e78b37", s=28, label="Original inputs")
    ax.scatter([], [], c="#2369a4", s=30, marker="x", label="Test inflow + WC/GOR")
    ax.set(xlim=(0, 2100), ylim=(0, 2100), xlabel="Measured BHP, psi", ylabel="Modeled BHP, psi", title="Updating inputs does not remove the bias")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    selected = [w for w in good if w["well"] in {"MPB-35", "MPJ-29", "MPE-48", "MPI-22", "MPM-62", "MPE-42"}]
    selected.sort(key=lambda w: w["results"]["baseline"]["bhp_error"], reverse=True)
    y = np.arange(len(selected))
    ax.barh(y-.17, [w["results"]["baseline"]["bhp_error"] for w in selected], .32, color="#e78b37")
    ax.barh(y+.17, [w["results"]["test_state"]["bhp_error"] for w in selected], .32, color="#2369a4")
    ax.set_yticks(y, [w["well"] for w in selected])
    ax.invert_yaxis()
    ax.set(xlabel="Modeled minus measured BHP, psi", title="Large residuals remain")

    ax = axes[2]
    probes = [("gor_plus20", "GOR +20%"), ("temp_plus20", "Uniform T +20 F"),
              ("pf_density_minus2", "PF density -2%"), ("return_mesh25", "Return grid: 25 ft"),
              ("segmented_pf", "Segmented PF"), ("return_no_slip", "No-slip holdup probe"),
              ("return_no_payne", "Without Payne")]
    for i, (variant, _) in enumerate(probes):
        deltas = [w["results"][variant]["predicted_bhp"]-w["results"]["test_state"]["predicted_bhp"]
                  for w in good if "predicted_bhp" in w["results"][variant]]
        ax.scatter(deltas, i+np.linspace(-.16, .16, len(deltas)), s=14, alpha=.55, color="#2369a4")
        ax.scatter(np.median(deltas), i, marker="|", s=180, color="black", zorder=5)
    ax.axvline(0, color="#737b87", lw=1)
    ax.set_yticks(range(len(probes)), [label for _, label in probes])
    ax.invert_yaxis()
    ax.set(xlabel="BHP change from test-conditioned case, psi", title="Different wells respond differently")
    for ax in axes:
        ax.grid(axis="x", alpha=.15)
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(.02, .035, "Hypothetical input changes are not measured uncertainty. No-slip/Payne probes isolate assumptions; they are not approved replacement models.\nOne failed well (MPF-73) remains in the JSON and coverage counts. No live queries, refitting, or changes to app physics.", fontsize=9, color="#4f5663")
    fig.tight_layout(rect=(0, .10, 1, .88), w_pad=2.5)
    fig.savefig(path, dpi=155)
    plt.close(fig)


if __name__ == "__main__":
    main()
