"""Offline comparison of proposed throat-entry closures; never imported by the app.

Run in a separate process. Temporary patches are restored after every case.
All wells are explicit synthetic/reference fixtures; no database access.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.optimize import brentq

from woffl.assembly import solopump as so
from woffl.flow import jetflow as jf
from woffl.flow.jetplot import JetBook
from woffl.flow.inflow import InFlow
from woffl.geometry import JetPump, Pipe, PipeInPipe, WellProfile
from woffl.pvt import BlackOil, FormGas, FormWater, ResMix

GC_PSI = 32.174 * 144
CASES = {
    "E41_reference": dict(wc=.894, gor=600., q_oil=246., pwf=1049., pres=1400., temp=80.),
    "high_watercut": dict(wc=.974, gor=600., q_oil=50., pwf=700., pres=1700., temp=100.),
    "gassy": dict(wc=.8, gor=1200., q_oil=100., pwf=500., pres=1700., temp=100.),
    "low_gor": dict(wc=.8, gor=50., q_oil=100., pwf=500., pres=1700., temp=100.),
}
MACH_VALUES = (1., 1.1, 1.5, 2., 2.5)
ORIGINAL_CHOKE = jf.throat_entry_mach_one
ORIGINAL_ZERO = jf.throat_entry_zero_tde
ORIGINAL_FLOOR = jf.psu_minimize


def inputs(case):
    ipr = InFlow(case["q_oil"], case["pwf"], case["pres"])
    fluid = ResMix(case["wc"], case["gor"], BlackOil.schrader(), FormWater.schrader(), FormGas.schrader())
    return ipr, fluid


def full_book(psu, temp, ken, area, ipr, fluid, step=2.5, scale=1.):
    """Sample the actual density/flow energy path, including its turning point."""
    q = ipr.oil_flow(psu, method="vogel")
    book = None
    for p in np.linspace(psu, 50., max(2, int(np.ceil((psu-50.)/step))+1)):
        fluid.condition(float(p), temp)
        v = sum(fluid.insitu_volm_flow(q)) / area
        args = (float(p), v, fluid.rho_mix(), fluid.cmix(), jf.enterance_ke(ken, v)*scale)
        if book is None:
            book = JetBook(*args)
        else:
            book.append(*args)
    return q, book


def energy_minimum_floor(tsu, ken, ate, ipr_su, prop_su, mach_crit=1., *, step=2.5):
    """Research prototype: zero of min(E(p)), using the same PVT/energy path.

    A boundary minimum is a numerical pressure-floor limit, not sonic flow.
    This is internal barotropic consistency, not a thermodynamic validation.
    """
    def residual(psu):
        _, book = full_book(psu, tsu, ken, ate, ipr_su, prop_su, step)
        return float(min(book.tde))
    lo, hi = 50.01, ipr_su.pres-10.
    if residual(lo) <= 0:
        psu = lo
    else:
        psu = brentq(residual, lo, hi, xtol=.001)
    q, book = full_book(psu, tsu, ken, ate, ipr_su, prop_su, step)
    return psu, q, book


@contextmanager
def variant(name, mach):
    """Only alter the two entry walks inside this standalone experiment."""
    if name == "threshold_only":
        # Algebraically cancels the existing KE divisor. This effective ken
        # is an experiment implementation detail, never a fitted coefficient.
        def choke(psu, tsu, ken, ate, ipr_su, prop_su, mach_crit=1.):
            return ORIGINAL_CHOKE(psu, tsu, (1+ken)*mach_crit**2-1,
                                  ate, ipr_su, prop_su, mach_crit)
        with patch.object(jf, "throat_entry_mach_one", choke):
            yield
    elif name == "scale_both":
        def zero(psu, tsu, ken, ate, ipr_su, prop_su, *, seed_book=None):
            effective_ken = ken if mach == 1. else (1+ken)/mach**2-1
            return ORIGINAL_ZERO(psu, tsu, effective_ken, ate, ipr_su, prop_su,
                                 seed_book=seed_book if mach == 1. else None)
        with patch.object(jf, "throat_entry_zero_tde", zero):
            yield
    elif name == "energy_minimum":
        # Its book uses a finer grid; do not seed the production 25-psi walk.
        def zero(psu, tsu, ken, ate, ipr_su, prop_su, *, seed_book=None):
            return ORIGINAL_ZERO(psu, tsu, ken, ate, ipr_su, prop_su)
        with patch.object(jf, "psu_minimize", energy_minimum_floor), patch.object(jf, "throat_entry_zero_tde", zero):
            yield
    else:
        yield


def floor_probe(case, name, mach):
    ipr, fluid = inputs(case)
    pump = JetPump("12", "B")
    with variant(name, mach):
        floor, _, choke = jf.psu_minimize(case["temp"], pump.ken, pump.ate, ipr, deepcopy(fluid), mach)
        _, book = jf.throat_entry_zero_tde(floor, case["temp"], pump.ken, pump.ate, ipr, deepcopy(fluid))
        try:
            pte, _, _, mte = book.dete_zero()
            accepted = dict(accepted=True, pte=float(pte), mach_te=float(mte))
        except Exception as exc:
            accepted = dict(accepted=False, error=f"{type(exc).__name__}: {exc}")
    _, dense = full_book(floor, case["temp"], pump.ken, pump.ate, ipr, deepcopy(fluid))
    idx = int(np.argmin(dense.tde))
    return dict(psu_floor=floor,
                physical_energy_min_fraction=float(dense.tde[idx]/dense.kde[0]),
                physical_energy_min_mach=float(dense.mach[idx]),
                pressure_floor_limited=idx == len(dense.tde)-1,
                walk_ke_ratio=float(choke.kde[0]/book.kde[0]), **accepted)


def solve_probe(case, name, mach, ppf):
    ipr, fluid = inputs(case)
    pump = JetPump("12", "B")
    wb = PipeInPipe(Pipe(4.5, .5), Pipe(6.875, .5))
    with variant(name, mach):
        answer = so.jetpump_solver(210., case["temp"], ppf, pump, wb,
                WellProfile.schrader(), ipr, fluid, FormWater.schrader().condition(0., 60.),
                "reverse", mach_crit=mach)
    return dict(zip(("psu", "sonic", "oil", "formation_water", "power_fluid", "mach_te"), answer))


def sound_probe(case, pressure):
    _, fluid = inputs(case)
    delta = .01
    density = [fluid.condition(pressure+d, case["temp"]).rho_mix() for d in (-delta, 0., delta)]
    drho_dp = (density[2]-density[0])/(2*delta)
    fluid.condition(pressure, case["temp"])
    cwood = fluid.cmix()
    cpath = float(np.sqrt(GC_PSI/drho_dp)) if drho_dp > 0 else None
    return dict(pressure=pressure, wood_ft_s=cwood, density_path_ft_s=cpath,
                path_over_wood=cpath/cwood if cpath else None,
                gas_volume_fraction=fluid.volm_fract()[2], drho_dp=drho_dp)


def run():
    report = {"scope": "Offline fixtures; no fitted or live well predictions. Alternative closures are research probes only.",
              "inputs": CASES, "floors": [], "solves": [], "sound": [], "refinement": []}
    for case_name, case in CASES.items():
        for pressure in (200., 500., 1000.):
            report["sound"].append(dict(case=case_name, **sound_probe(case, pressure)))
        for mode in ("legacy", "threshold_only", "scale_both", "energy_minimum"):
            for mach in ((1.,) if mode == "energy_minimum" else MACH_VALUES):
                row = dict(case=case_name, mode=mode, mach_crit=mach)
                try:
                    row.update(floor_probe(case, mode, mach))
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                report["floors"].append(row)
                for ppf in (2500., 3168.):
                    row = dict(case=case_name, mode=mode, mach_crit=mach, ppf=ppf)
                    try:
                        row.update(solve_probe(case, mode, mach, ppf))
                    except Exception as exc:
                        row["error"] = f"{type(exc).__name__}: {exc}"
                    report["solves"].append(row)
        for step in (10., 5., 2.5, 1.25):
            ipr, fluid = inputs(case)
            pump = JetPump("12", "B")
            psu, _, book = energy_minimum_floor(case["temp"], pump.ken, pump.ate, ipr, fluid, step=step)
            idx = int(np.argmin(book.tde))
            report["refinement"].append(dict(case=case_name, step_psi=step, floor=psu,
                energy_min_mach=float(book.mach[idx]), boundary_minimum=idx == len(book.tde)-1))
    return report


def plot(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for mode, label in (("legacy", "Current model"), ("threshold_only", "Remove KE scaling"),
                        ("scale_both", "Scale both walks (empirical)")):
        floor = [r for r in report["floors"] if r["case"] == "E41_reference" and r["mode"] == mode and "psu_floor" in r]
        solves = [r for r in report["solves"] if r["case"] == "E41_reference" and r["mode"] == mode and r["ppf"] == 3168. and "psu" in r]
        axes[0].plot([r["mach_crit"] for r in floor], [r["psu_floor"] for r in floor], "o-", label=label)
        axes[1].plot([r["mach_crit"] for r in solves], [r["psu"] for r in solves], "o-", label=label)
    for ax, title in zip(axes, ("Calculated choke floor", "Returned operating suction at PF 3,168 psig")):
        ax.set(title=title, xlabel="Critical-Mach setting", ylabel="Suction pressure (psig)")
        ax.grid(alpha=.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("E-41 reference fixture — model experiments, not field validation")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    # These counterfactual edits described the pre-entry-energy-v1 engine.
    # Preserve the original JSON/figure; do not relabel new physics as legacy.
    if hasattr(jf, "psu_minimize") and not hasattr(jf, "_TE_PDEC"):
        raise SystemExit("Historical study requires the pre-entry-energy-v1 engine. Use tools/physics_qualification.py and tests/test_entry_energy.py for the current model.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("critical-mach-study.json"))
    args = parser.parse_args()
    report = run()
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    plot(report, args.output.with_suffix(".png"))
    print(json.dumps({"floors": len(report["floors"]), "solves": len(report["solves"]),
                      "solve_errors": sum("error" in r for r in report["solves"]),
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
