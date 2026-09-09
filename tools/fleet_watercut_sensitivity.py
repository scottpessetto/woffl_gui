"""Offline WC scenarios for a frozen fleet report; no fitting or warehouse reads.

Run from the repository root with the repository venv. Scenario bands are
illustrative input perturbations, not estimated measurement confidence intervals.
"""
import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("WOFFL_MAX_WORKERS", "2")
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

OFFSETS = (-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10)
MODES = ("fixed_gor", "fixed_gas_per_liquid")


def forbid_query(*args, **kwargs):
    raise RuntimeError("Watercut sensitivity must use only the frozen report")


def sweep_well(well):
    from server.services.fleet_validation import predict_well
    from woffl.assembly import databricks_client

    databricks_client.execute_query = forbid_query
    config = well["config"]
    current_wc = config["form_wc"]
    last = well["latest_test"]
    base = {"well": well["well"], "config": config, "observations": [last]}
    cases = []
    memo = {}
    for mode in MODES:
        scenarios = [("offset", delta, min(.99, max(0., current_wc + delta/100)))
                     for delta in OFFSETS]
        if last.get("wc") is not None and 0 <= last["wc"] <= .99:
            scenarios.append(("test_wc", None, last["wc"]))
        for kind, delta, wc in scenarios:
            gor = config["form_gor"]
            if mode == "fixed_gas_per_liquid" and wc != current_wc:
                gor *= (1-current_wc)/(1-wc)
            key = (wc, gor)
            if key not in memo:
                candidate = deepcopy(base)
                candidate["config"].update(form_wc=wc, form_gor=gor)
                memo[key] = predict_well(candidate)["observations"][0]
            prediction = memo[key]
            cases.append(dict(mode=mode, kind=kind, requested_offset_points=delta,
                actual_offset_points=100*(wc-current_wc), wc=wc, gor=gor,
                **{k: v for k, v in prediction.items() if k.startswith("predicted_")
                   or k in {"error", "bhp_error", "oil_error_pct", "pf_error_pct", "sonic"}}))
    # A frozen-input replay must reproduce the source before its scenarios can be used.
    baseline = next(c for c in cases if c["mode"] == "fixed_gor"
                    and c["requested_offset_points"] == 0)
    if "error" in last:
        if "error" not in baseline:
            raise AssertionError(f"{well['well']}: original failure no longer reproduced")
    else:
        for key in ("predicted_bhp", "predicted_oil", "predicted_pf"):
            if abs(baseline[key]-last[key]) > 1e-8:
                raise AssertionError(f"{well['well']}: baseline changed for {key}")
    return dict(well=well["well"], pad=well["pad"], model_wc=current_wc,
                model_gor=config["form_gor"], qwf_liquid=config["qwf"],
                latest_test=last, baseline=baseline, cases=cases)


def band(well, mode, width):
    return [c for c in well["cases"] if c["mode"] == mode and c["kind"] == "offset"
            and abs(c["requested_offset_points"]) <= width and "error" not in c]


def summarize(wells):
    import numpy as np

    solved = [w for w in wells if "error" not in w["baseline"]]
    summaries = {}
    for mode in MODES:
        for width in (5, 10):
            bhp_shift, oil_shift, bhp_remainder = [], [], []
            for w in solved:
                cases = band(w, mode, width)
                b = w["baseline"]
                bhp_shift.append(max(abs(c["predicted_bhp"]-b["predicted_bhp"]) for c in cases))
                oil_shift.append(max(abs(100*(c["predicted_oil"]/b["predicted_oil"]-1)) for c in cases))
                bhp_remainder.append(min(abs(c["bhp_error"]) for c in cases))
            summaries[f"{mode}_{width}points"] = dict(
                n=len(solved), median_max_bhp_shift_psi=float(np.median(bhp_shift)),
                largest_bhp_shift_psi=float(max(bhp_shift)),
                median_max_oil_shift_pct=float(np.median(oil_shift)),
                bhp_still_over_50psi=sum(e > 50 for e in bhp_remainder),
                bhp_still_over_100psi=sum(e > 100 for e in bhp_remainder),
                failed_scenarios=sum("error" in c for w in wells for c in w["cases"]
                    if c["mode"] == mode and c["kind"] == "offset"
                    and abs(c["requested_offset_points"]) <= width))
    return summaries


def render(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    wells = [w for w in report["wells"] if "error" not in w["baseline"]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), layout="constrained")
    fig.suptitle("Watercut uncertainty changes the model comparison\n"
                 f"{report['snapshot_time'][:10]} | {len(wells)} baseline solves; "
                 f"{len(report['wells'])-len(wells)} baseline failure", fontsize=16)
    for ax, key, label in zip(axes, ("bhp", "oil"), ("BHP (psi)", "Oil (BOPD)")):
        values = []
        for w in wells:
            x = w["latest_test"]["observed_"+key]
            b = w["baseline"]["predicted_"+key]
            values.extend((x, b))
            for width, color, lw in ((10, "#cbd5e1", 3), (5, "#2563eb", 1.6)):
                ys = [c["predicted_"+key] for c in band(w, "fixed_gor", width)]
                values.extend(ys)
                ax.vlines(x, min(ys), max(ys), color=color, lw=lw, zorder=2)
            ax.scatter([x], [b], s=24, c="#0f172a", zorder=3)
            if (key == "bhp" and abs(w["baseline"]["bhp_error"]) > 300) or (
                    key == "oil" and w["well"] in {"MPH-08", "MPB-28"}):
                ax.annotate(w["well"], (x, b), xytext=(5, 6),
                            textcoords="offset points", fontsize=8)
        lim = max(values)*1.1
        ax.plot([0, lim], [0, lim], color="#64748b", ls="--", lw=1)
        ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="Reported test "+label,
               ylabel="Model "+label)
        ax.grid(alpha=.2)
    axes[0].legend(handles=[Line2D([0], [0], marker="o", color="none", markerfacecolor="#0f172a",
                       label="Current inputs"),
        Line2D([0], [0], color="#2563eb", lw=2, label="WC ±5 percentage points"),
        Line2D([0], [0], color="#cbd5e1", lw=3, label="WC ±10 percentage points")],
        loc="upper left", fontsize=9)
    fig.supxlabel("Sampled scenarios, not confidence intervals. WC bounded to 0–99%; fixed total-liquid IPR and GOR.\n"
                  "Other inputs and reported measurements held fixed. Failed scenarios are listed in the data; "
                  "no coefficients were fitted.", fontsize=9)
    fig.savefig(output.with_suffix(".png"), dpi=180)
    plt.close(fig)

    summary = report["summary"]
    text = [f"# Fleet watercut sensitivity — {report['snapshot_time'][:10]}", "",
        f"This is an offline sensitivity experiment on the same {len(report['wells'])} wells in the fleet "
        f"comparison. It is not a new accuracy score or a calibration. All {len(report['wells'])} unperturbed "
        "replays reproduced the source success/failure status; successful BHP, oil and PF "
        "outputs matched within 1e-8 in their original units. No warehouse reads, saved "
        "input changes or application changes were required.", "",
        f"![WC scenario ranges]({output.name}.png)", "", "## Assumptions", "",
        "- WC is formation-water fraction, excluding returned power fluid. Sweep ±10 "
        "percentage points in 2.5-point steps around each configured WC, bounded to 0–99%. "
        "The ±5-point band uses its five interior samples. Near 0/99%, bands are truncated.",
        "- Hold total-liquid IPR reference rate (`qwf`), reference BHP, reservoir pressure, "
        "pump, losses, fluid properties and the latest test's operating pressures fixed. "
        "Changing WC changes the derived oil IPR once, using `oil = liquid * (1 - WC)`. "
        "The resulting operating liquid rate can change through the hydraulic solve.",
        "- `fixed_gor` holds gas per stock-tank oil fixed (the ordinary WC-only model input "
        "change). `fixed_gas_per_liquid` also adjusts GOR so `GOR * (1 - WC)` is constant: "
        "gas per barrel of formation liquid, and therefore gas at the unchanged IPR "
        "reference rate, stays fixed. This is not a fixed operating gas-rate constraint.",
        "- Test WC substitution changes WC only under each gas convention; it does not "
        "re-anchor IPR on measured BHP/oil. No WC was selected or saved to improve fit.",
        "- These are illustrative ranges, not measured WC errors or statistical confidence "
        "intervals. Each reported test coordinate remains fixed in the figure. Real WC "
        "measurement error can move the reported oil coordinate too, and total-liquid/PF "
        "errors can be correlated. Thus this is only one part of the uncertainty.", "",
        "## Quantified sensitivity", "",
        "For each well, take the largest absolute change from its current prediction among "
        f"the successful sampled scenarios. Then take the median across {len(wells)} baseline successes. "
        "Failed scenarios remain recorded and do not contribute invented bounds.", "",
        "| Gas assumption | WC band, points | Median max BHP change, psi | Largest BHP change, psi | "
        "Median max oil change, % | Wells still >50 psi off at every successful sample | Failed scenarios |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for mode in MODES:
        for width in (5, 10):
            s = summary[f"{mode}_{width}points"]
            text.append(f"| {mode} | ±{width} | {s['median_max_bhp_shift_psi']:.1f} | "
                        f"{s['largest_bhp_shift_psi']:.1f} | {s['median_max_oil_shift_pct']:.1f} | "
                        f"{s['bhp_still_over_50psi']}/{s['n']} | {s['failed_scenarios']} |")
    failed = [(w["well"], c) for w in report["wells"] for c in w["cases"] if "error" in c]
    text.extend(["", f"There are {len(failed)} failed requested scenarios across the full "
                 "experiment (including the test-WC substitutions and repeated clipped "
                 "endpoints). A missing prediction is not evidence of a narrow range. "
                 "The JSON retains each failure and its inputs.", "",
                 "The count of residuals above 50 psi is an optimistic diagnostic that "
                 "allows choosing a different WC for each well after seeing its BHP. It is "
                 "not an improvement claim, a fit or an independent validation.", "",
                 "## Selected wells", "",
                 "Ranges below use fixed GOR and ±5 WC points. Observed BHP does not change.", "",
                 "| Well | Model / test WC, % | Observed BHP, psi | Current BHP, psi | "
                 "Sampled BHP range, psi | Current oil, BOPD | Sampled oil range, BOPD |",
                 "|---|---:|---:|---:|---:|---:|---:|"])
    for w in wells:
        if w["well"] not in {"MPB-28", "MPB-35", "MPE-24", "MPE-48", "MPH-08", "MPI-22", "MPJ-29"}:
            continue
        c = band(w, "fixed_gor", 5)
        br = [v["predicted_bhp"] for v in c]
        oil = [v["predicted_oil"] for v in c]
        t, b = w["latest_test"], w["baseline"]
        text.append(f"| {w['well']} | {100*w['model_wc']:.1f} / {100*t['wc']:.1f} | "
                    f"{t['observed_bhp']:.0f} | {b['predicted_bhp']:.0f} | "
                    f"{min(br):.0f}–{max(br):.0f} | {b['predicted_oil']:.0f} | "
                    f"{min(oil):.0f}–{max(oil):.0f} |")
    closure = report["rate_closure"]
    text.extend(["", "## Interpretation", "",
        "WC can move the oil prediction substantially, especially at high WC. In this "
        "model, the typical BHP response to an isolated WC change is much smaller than "
        "the existing fleet BHP error. Holding gas per liquid fixed makes BHP sensitivity "
        "smaller still. This result does not establish the true field sensitivity or rule "
        "out larger BHP errors caused by correlated liquid-rate, gas, PF or IPR errors.", "",
        "The prior fleet report's test-conditioned diagnostic changed WC, GOR and the IPR "
        "anchor together. Improvements there cannot be attributed to WC alone.", "",
        "## How to improve the comparison and calibration", "",
        "1. Establish a realistic WC uncertainty per test from repeat samples and test "
        "conditions. A five-point error at 80% WC changes derived oil by 25% at fixed "
        "formation-liquid rate: 1,000 BLPD gives 200 BOPD at 80% versus 150 at 85%.",
        "2. Compare BHP, formation liquid, PF and oil together. Comparing WC to "
        f"`1 - oil / formation_liquid` across {closure['n']} latest tests gives a maximum "
        f"difference of {closure['max_difference_points']:.2g} percentage points. "
        "Internal consistency is not independent verification of those measurements. Reconcile "
        "returned PF before using surface water to estimate formation WC.",
        "3. The multipoint fitter already uses Huber losses and point weights for BHP/PF "
        "residuals. Its oil test enters the per-point IPR anchor, while WC/GOR are fixed "
        "inputs. Extend that existing framework to vary composition and dependent rate "
        "inputs coherently within evidence-based bounds, with a penalty for departing "
        "from the test estimate. BHP/PF weights also need "
        "gauge-datum and metering checks; those observations are not automatically correct.",
        "4. Validate on later, stable operating periods within the same pump installation, "
        "with training inputs frozen. Check pressure response as well as level match. "
        "Keep large unresolved BHP errors and failed solves visible.",
        "5. Evaluate optimization moves at low/base/high credible WC and show the gain "
        "range. Prefer moves that stay beneficial across those scenarios; flag cases whose "
        "recommendation changes with WC. Keep existing physics conservation tests separate "
        "from uncertain field fit scores.", "",
        "These are proposed follow-ups. This experiment does not modify the live optimizer, "
        "calibration, Medium deployment or physics acceptance thresholds.", "",
        "## Reproduction", "", "```powershell",
        "./venv/Scripts/python.exe tools/fleet_watercut_sensitivity.py", "```", "",
        f"Source JSON SHA-256: `{report['source_sha256']}`. "
        f"[All scenarios]({output.name}.json). Two-worker ceiling; source queries blocked.", ""])
    output.with_suffix(".md").write_text("\n".join(text), encoding="utf-8")


def main():
    from woffl.assembly.parallelism import worker_ceiling

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT/"docs/fleet_actuality_2026-09-08.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source_bytes = args.source.read_bytes()
    source = json.loads(source_bytes.decode("utf-8"))
    eligible = [w for w in source["wells"] if not w.get("exclusion")]
    if not eligible:
        raise ValueError("Source report has no eligible wells")
    closure = [100*abs(1-w["latest_test"]["observed_oil"]/w["latest_test"]["qtot"]
                      -w["latest_test"]["wc"]) for w in eligible
               if w["latest_test"].get("qtot", 0) > 0 and w["latest_test"].get("wc") is not None]
    wells = []
    workers = min(2, worker_ceiling())
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(sweep_well, w) for w in eligible]
        for future in as_completed(futures):
            result = future.result()
            wells.append(result)
            print(f"{len(wells)}/{len(eligible)} {result['well']}", flush=True)
    wells.sort(key=lambda w: w["well"])
    report = dict(snapshot_time=source["snapshot_time"], model_version=source["model_version"],
        source_sha256=hashlib.sha256(source_bytes).hexdigest(), workers=workers,
        read_only=True, refitted=False, confidence_intervals=False,
        source_queries=0, offsets_points=OFFSETS, baseline_replays_verified=len(wells),
        rate_closure=dict(n=len(closure), max_difference_points=max(closure) if closure else None),
        summary=summarize(wells), wells=wells)
    output = args.output or ROOT/f"docs/fleet_watercut_sensitivity_{source['snapshot_time'][:10]}"
    output.with_suffix(".json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    render(report, output)
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
