"""Run the chronological pump-history reference benchmark on frozen local data."""
from hashlib import sha256
import json
from pathlib import Path
import pickle

from server.services.fleet_validation import metrics
from server.services.pump_history_benchmark import chronological_challenges, evaluate_challenge
from server.services.optimizer_runs import _plain
from woffl.assembly.network_optimizer import WellConfig

ROOT = Path(__file__).resolve().parents[1]
WELLS = ("MPB-30", "MPB-37", "MPB-39", "MPF-107", "MPE-42", "MPF-73")


def main():
    preflight_path = ROOT/"docs/pump_lifetime_preflight_v2_2026-09-11.json"
    snapshot_path = ROOT/"build/fleet-actuality-snapshot.pkl"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    snapshot = pickle.loads(snapshot_path.read_bytes())
    configs = snapshot["configs"]
    if isinstance(configs, list):
        configs = {c.well_name: c for c in configs}
    reports, exclusions = [], []
    for well in preflight["wells"]:
        name = well["well"]
        if name not in WELLS:
            continue
        base = configs.get(name)
        if base is None:
            exclusions.append(dict(well=name, reason="no frozen model config"))
            continue
        if isinstance(base, dict):
            base = WellConfig(**base)
        for challenge in chronological_challenges(well["eras"]):
            try:
                report = evaluate_challenge(base, challenge)
                reports.append(report)
                print(f"{name}: {report['training_pump']} -> {report['prediction_pump']}, "
                      f"{len(report['rows'])} later tests", flush=True)
            except Exception as exc:
                exclusions.append(dict(well=name, start=challenge["prediction_installation"]["start"],
                                       reason=f"{type(exc).__name__}: {exc}"))
    report = dict(scope="Chronological cross-pump reference challenge; earlier-test Vogel productivity, clean pump losses; no app calibration saves",
        snapshot_time=snapshot["captured_at"], snapshot_sha256=sha256(snapshot_path.read_bytes()).hexdigest(),
        preflight_sha256=sha256(preflight_path.read_bytes()).hexdigest(), live_queries=0,
        limitations=["Present-day geometry/PVT/reservoir-pressure priors are not verified historical inputs.",
                     "No fitted pump-loss parameters are transferred across installations. This establishes the reference to beat.",
                     "Frozen-composition forecast uses only earlier tests for IPR, WC and GOR; measured-composition replay is conditional on later WC/GOR.",
                     "Gauge depth is assumed at suction; typical 40-ft offset has not been individually corrected.",
                     "Nominal catalog conflicts and Guiberson equivalence conversions remain flagged, not silently repaired.",
                     "These are observational changeouts, not controlled trials; this does not validate alternative-size ranking or marginal response."],
        challenges=reports, exclusions=exclusions)
    modes = ("frozen_composition", "measured_composition")
    report["metrics"] = {mode: metrics([r["predictions"][mode] for c in reports for r in c["rows"]]) for mode in modes}
    report["different_size_metrics"] = {mode: metrics([r["predictions"][mode] for c in reports if c["different_size"] for r in c["rows"]]) for mode in modes}
    report["code_sha256"] = {rel: sha256((ROOT/rel).read_bytes()).hexdigest() for rel in
        ("tools/pump_history_benchmark.py", "server/services/pump_history_benchmark.py",
         "server/services/field_validation.py", "woffl/assembly/solopump.py", "woffl/geometry/jetpump.py")}
    path = ROOT/"docs/pump_history_benchmark_2026-09-11.json"
    report = _plain(report)
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    plot(report, path.with_suffix(".png"))
    print(json.dumps(dict(challenges=len(reports), exclusions=exclusions, metrics=report["metrics"]), indent=2))


def plot(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from datetime import date
    fig, axes = plt.subplots(len(WELLS), 3, figsize=(16, 18), squeeze=False)
    for i, name in enumerate(WELLS):
        eras = [c for c in report["challenges"] if c["well"] == name]
        for j, (quantity, unit) in enumerate((("bhp", "psi"), ("oil", "BOPD"), ("pf", "BPD"))):
            ax = axes[i][j]
            for c in eras:
                xs = [date.fromisoformat(r["date"]) for r in c["rows"]]
                actual = [r[f"observed_{quantity}"] for r in c["rows"]]
                if quantity == "pf":
                    actual = [v if v is not None and 0 < v <= 20000 else float("nan") for v in actual]
                predicted = [r["predictions"]["frozen_composition"].get(f"predicted_{quantity}", float("nan")) for r in c["rows"]]
                ax.scatter(xs, actual, s=11, color="#263d55", label="Observed")
                ax.plot(xs, predicted, color="#da8335", lw=1.5, marker=".", ms=3, label="Earlier-test forecast")
                start = date.fromisoformat(c["prediction_start"])
                ax.axvline(start, color="#86939d", lw=.6, ls="--")
                ax.text(start, .98, c["prediction_pump"], transform=ax.get_xaxis_transform(), fontsize=7, va="top")
            ax.set_title(f"{name} | {quantity.upper()} ({unit})", fontsize=10)
            locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
            ax.tick_params(labelsize=8)
            ax.grid(alpha=.15)
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Pump match over time: chronological reference forecast", fontsize=17, fontweight="bold")
    fig.text(.5, .955, "Forecasts at later test dates use earlier oil productivity, WC and GOR. Pump losses remain at clean reference.", ha="center", fontsize=10)
    fig.text(.5, .015, "Dark points: observed | Orange: earlier-test forecast | Dashed lines: installation changes | Gaps retain failed predictions\nPresent-day geometry/PVT/pressure priors; historical specs need reconciliation. Baseline for improvement, not validated pump-size recommendations.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .04, 1, .945))
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
