"""Offline model qualification, separate from compatibility regressions.

--strict exits nonzero while an energy-closure gap remains. The default
report makes that gap visible in CI without calling it validated physics.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import warnings

from woffl.flow.entry_energy import MODEL_VERSION


def energy_closure(mach_crit):
    from woffl.flow import jetflow as jf
    from woffl.flow.inflow import InFlow
    from woffl.geometry.jetpump import JetPump
    from woffl.pvt import BlackOil, FormGas, FormWater, ResMix

    pump = JetPump("12", "B")
    ipr = InFlow(100., 500., 1700.)
    fluid = ResMix(.8, 250., BlackOil.schrader(), FormWater.schrader(), FormGas.schrader())
    args = (1200., 100., pump.ken, pump.ate, ipr)
    _, operating = jf.throat_entry_zero_tde(*args, deepcopy(fluid))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        _, _, choking = jf.throat_entry_mach_one(*args, deepcopy(fluid), mach_crit=mach_crit)
    # Same inlet state must have the same kinetic term in both energy walks.
    reference = float(operating.kde_ray[0])
    actual = float(choking.kde_ray[0])
    mismatch = abs(actual-reference) / max(abs(reference), 1e-12)
    return dict(mach_crit=mach_crit, operating_ke=reference, choking_ke=actual,
                relative_energy_mismatch=mismatch, qualified=mismatch <= 1e-10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("physics-qualification.json"))
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    rows = [energy_closure(mc) for mc in (1., 1.5, 2., 2.5)]
    report = {"qualified": all(r["qualified"] for r in rows), "energy_closure": rows,
              "model_version": MODEL_VERSION, "field_validated": False,
              "qualification_scope": "agreement of the two entry energy calculations only",
              "limitation": "Mach multiplier retired. Shared energy checks do not establish field validity; isothermal PVT and phase-release assumptions still require validation."}
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = ["## Entry energy consistency", "", report["qualification_scope"], "", report["limitation"], "",
             "| Legacy Mach input (ignored) | Energy mismatch | Consistent |", "|---:|---:|:---:|"]
    lines += [f"| {r['mach_crit']} | {100*r['relative_energy_mismatch']:.2f}% | {r['qualified']} |" for r in rows]
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.strict and not report["qualified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
