"""Compare return hydraulics on identical chronological, frozen local challenges.

No calibration, live reads, saves or automatic model selection. Failed predictions
remain in coverage counts; common-case scores prevent survivor selection bias.
"""
from copy import deepcopy
from datetime import date
from hashlib import sha256
import json
from pathlib import Path
import pickle
from time import perf_counter

from server.services.fleet_validation import metrics
from server.services.optimizer_runs import _plain
from server.services.pump_history_benchmark import chronological_challenges, evaluate_challenge
from tools.pump_history_benchmark import WELLS
from woffl.assembly.network_optimizer import WellConfig
from woffl.flow.hydraulics import HYDRAULICS_MODELS

ROOT = Path(__file__).resolve().parents[1]
MODES = ("frozen_composition", "measured_composition")


def summarize(reports):
    """Both all-attempt and paired common-success metrics, retaining failures."""
    summary = {}
    for mode in MODES:
        indexed = {model: {(c["well"], c["prediction_start"], r["date"], r["wt_uid"]):
                           (r["predictions"][mode], c["different_size"])
                           for c in challenges for r in c["rows"]}
                   for model, challenges in reports.items()}
        all_keys = set.union(*(set(rows) for rows in indexed.values()))
        if any(set(rows) != all_keys for rows in indexed.values()):
            raise ValueError("Models were not evaluated on identical observations")
        common = {key for key in all_keys if all("bhp_error" in rows[key][0] for rows in indexed.values())}
        summary[mode] = {
            model: dict(all=metrics([p for p, _ in rows.values()]),
                        different_size=metrics([p for p, different in rows.values() if different]),
                        common=metrics([rows[key][0] for key in sorted(common)]),
                        common_different_size=metrics([rows[key][0] for key in sorted(common) if rows[key][1]]))
            for model, rows in indexed.items()}
    return summary


def main():
    preflight_path = ROOT / "docs/pump_lifetime_preflight_v2_2026-09-11.json"
    snapshot_path = ROOT / "build/fleet-actuality-snapshot.pkl"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    snapshot = pickle.loads(snapshot_path.read_bytes())  # trusted local project snapshot only
    configs = snapshot["configs"]
    if isinstance(configs, list):
        configs = {c.well_name: c for c in configs}
    reports, timing = {}, {}
    for model in HYDRAULICS_MODELS:
        started = perf_counter()
        reports[model] = []
        for well in preflight["wells"]:
            if well["well"] not in WELLS:
                continue
            base = deepcopy(configs[well["well"]])
            if isinstance(base, dict):
                base = WellConfig(**base)
            base.hydraulics_model = model
            for challenge in chronological_challenges(well["eras"]):
                report = evaluate_challenge(base, challenge)
                reports[model].append(report)
                primary = report["metrics"]["frozen_composition"]
                print(f"{model}: {well['well']} {report['training_pump']} -> {report['prediction_pump']}: "
                      f"{primary['solved']}/{primary['observations']} solved", flush=True)
        timing[model] = round(perf_counter()-started, 3)
    report = _plain(dict(
        scope="Same earlier-test oil productivity, WC/GOR and clean pump losses; only return hydraulics differs",
        snapshot_time=snapshot["captured_at"], live_queries=0, fitted_parameters=0,
        snapshot_sha256=sha256(snapshot_path.read_bytes()).hexdigest(),
        preflight_sha256=sha256(preflight_path.read_bytes()).hexdigest(),
        elapsed_seconds=timing, validated_for_sizing=False,
        limitations=[
            "Present-day geometry, PVT and reservoir-pressure priors are not verified historical inputs.",
            "Nominal specification conflicts remain unresolved; tracker dimensions are not measured wear.",
            "Gauge-to-pump offset is assumed zero; typical within 40 ft is not an individual correction.",
            "Observational changeouts do not prove causal size ranking or marginal-WC response accuracy.",
            "Model comparison uses no retuning; choosing a winner on these observations needs another later holdout.",
            "Hagedorn-Brown holdup is a vertical reference; inclined use projects gravity only.",
            "Shi/Pan uses one mixed liquid, steady isothermal flow and hydraulic diameter for annuli.",
            "Downhill return segments and non-subcritical gradients are explicit alternative-model failures.",
            "Tulsa is unavailable pending complete primary closure equations and independent verification.",
        ],
        metrics=summarize(reports), challenges=reports,
        code_sha256={rel: sha256((ROOT/rel).read_bytes()).hexdigest() for rel in (
            "tools/hydraulics_benchmark.py", "server/services/pump_history_benchmark.py",
            "server/services/field_validation.py", "woffl/flow/hydraulics.py", "woffl/flow/outflow.py",
            "woffl/assembly/solopump.py", "woffl/geometry/jetpump.py")},
    ))
    path = ROOT / "docs/hydraulics_benchmark_2026-09-11.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    plot(report, path.with_suffix(".png"))
    print(json.dumps(dict(metrics=report["metrics"], seconds=timing), indent=2), flush=True)


def plot(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    colors = {"beggs": "#c97514", "hagedorn_brown": "#3272a8", "drift_flux": "#258a62"}
    labels = {"beggs": "Beggs-Brill + Payne", "hagedorn_brown": "Hagedorn-Brown + Griffith", "drift_flux": "Shi / Pan drift-flux"}
    fig, axes = plt.subplots(len(WELLS), 2, figsize=(15, 18), squeeze=False)
    for i, name in enumerate(WELLS):
        for j, (quantity, unit) in enumerate((("bhp", "psi"), ("oil", "BOPD"))):
            ax = axes[i][j]
            for model, challenges in report["challenges"].items():
                for c in (c for c in challenges if c["well"] == name):
                    xs = [date.fromisoformat(r["date"]) for r in c["rows"]]
                    if model == "beggs":
                        ax.scatter(xs, [r[f"observed_{quantity}"] for r in c["rows"]], s=12, color="#242b36", label="Observed", zorder=5)
                        start = date.fromisoformat(c["prediction_start"])
                        ax.axvline(start, color="#959ba3", lw=.6, ls="--")
                        ax.text(start, .98, c["prediction_pump"], transform=ax.get_xaxis_transform(), fontsize=8, va="top")
                    predicted = [r["predictions"]["frozen_composition"].get(f"predicted_{quantity}", float("nan")) for r in c["rows"]]
                    ax.plot(xs, predicted, color=colors[model], lw=1.4, marker=".", ms=3, label=labels[model])
            ax.set_title(f"{name} | {quantity.upper()} ({unit})", fontsize=11)
            locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
            ax.tick_params(labelsize=8)
            ax.grid(alpha=.15)
            ax.spines[["top", "right"]].set_visible(False)
    handles, names = axes[0][0].get_legend_handles_labels()
    unique = dict(zip(names, handles))
    fig.legend(unique.values(), unique.keys(), loc="upper center", bbox_to_anchor=(.5, .955), ncol=4, frameon=False)
    fig.suptitle("Return hydraulics across pump changes: same inputs, no loss-coefficient retuning", fontsize=15, fontweight="bold")
    fig.text(.5, .025, "Earlier tests set oil productivity, WC and GOR. Dashed lines mark later installations. Missing predictions remain failures.\nCurrent geometry/PVT/pressure priors; unresolved catalog flags. Comparison does not validate pump-size optimization.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .055, 1, .93))
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
