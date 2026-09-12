"""Small offline reproductions of model-plan findings; no field simulation."""
from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import patch

from tests.test_pad_optimize import CurvePlant, FakeOptimizer, _perf, _result, _wells
from server.services.match_health import _verdict
from woffl.gui.pad_optimize import run_optimization


ROOT = Path(__file__).resolve().parents[1]


def synthetic_allocation(opt, **_):
    # A bounded, choked-oil response: more header increases PF draw while
    # oil stays level. Respect the allocation budget at every trial.
    draw = 10000. + 2*(opt.power_fluid.pressure-2500.)
    return [_result("W1", draw, 100.)] if draw <= opt.power_fluid.total_rate else []


def main():
    plant = CurvePlant()
    with patch("woffl.assembly.network_optimizer.NetworkOptimizer", FakeOptimizer), \
         patch.object(FakeOptimizer, "perf_table", {("W1", "12", "B"): lambda p: _perf(10000 + 2*(p-2500), 100)}), \
         patch("woffl.assembly.network_optimizer.reconcile_wells", lambda *_: {}), \
         patch("woffl.assembly.optimization_algorithms.optimize", synthetic_allocation):
        _, _, meta = run_optimization(_wells("W1"), plant, 3, ["12"], ["B"],
                                      "milp", None, water_price=.001, refine_rounds=0)
    curve_pressure = plant.header_at_flow(meta["total_pf_bpd"], 3)
    report = dict(scope="Synthetic bounded response and toy plant; no field forecast or live reads",
        response="Oil 100 BOPD; PF draw = 10000 + 2*(header - 2500); each trial respects its budget",
        fixed_curve=dict(reported_header=meta["header_psi"], selected_flow=meta["total_pf_bpd"],
                         curve_header_at_selected_flow=curve_pressure,
                         pressure_gap=curve_pressure-meta["header_psi"],
                         reported_converged=meta["converged"], selected_oil=meta["total_oil_bopd"],
                         water_price=.001),
        empty_match_health_verdict=_verdict({}))
    report["code_sha256"] = {rel: sha256((ROOT/rel).read_bytes()).hexdigest() for rel in
        ["woffl/gui/pad_optimize.py", "server/services/match_health.py", "tools/model_plan_optimization_probes.py"]}
    path = ROOT/"docs/model_plan_optimization_probes_after_2026-09-11.json"
    path.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
