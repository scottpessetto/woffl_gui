"""Render the September 8, 2026 fleet study, including its reviewed findings."""
import argparse
import html
import json
from pathlib import Path

import numpy as np


def render(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = json.loads(path.read_text(encoding="utf-8"))
    if report["snapshot_time"][:10] != "2026-09-08":
        raise ValueError("Review and update the dated findings before rendering a different snapshot")
    selected = [w for w in report["wells"] if not w.get("exclusion")]
    good = [w for w in selected if "error" not in w["latest_test"]]
    coverage, scores = report["coverage"], report["metrics"]
    pads = coverage["pads"]
    colors = dict(zip(pads, plt.colormaps["tab10"].colors))
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), layout="constrained")
    fig.suptitle("WOFFL: current model versus measured well tests\n"
                 f"Snapshot {report['snapshot_time'][:10]} | {len(selected)} wells, {len(pads)} pads | {report['model_version']}", fontsize=16)
    for ax, (key, label, tolerance) in zip(axes.flat, [("bhp", "BHP (psi)", 50),
            ("oil", "Oil (BOPD)", .2), ("pf", "Power fluid (BPD, log scale)", .1)]):
        valid = [w for w in good if w["latest_test"].get("observed_"+key) is not None]
        values = [w["latest_test"][prefix+key] for w in valid for prefix in ("observed_", "predicted_")]
        lo, hi = (max(.1, min(values)*.7), max(values)*1.25) if key == "pf" else (0, max(values)*1.12)
        x = np.geomspace(lo, hi, 150) if key == "pf" else np.linspace(lo, hi, 150)
        ax.plot(x, x, color="#475569", lw=1, label="Exact agreement")
        ax.fill_between(x, x-tolerance if key == "bhp" else x*(1-tolerance),
                        x+tolerance if key == "bhp" else x*(1+tolerance), color="#e2e8f0")
        for pad in pads:
            subset = [w["latest_test"] for w in valid if w["pad"] == pad]
            ax.scatter([w["observed_"+key] for w in subset], [w["predicted_"+key] for w in subset],
                       s=36, color=colors[pad], edgecolor="white", linewidth=.5, label=pad, zorder=3)
        if key == "pf":
            ax.set_xscale("log"); ax.set_yscale("log")
        ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Measured "+label, ylabel="Modeled "+label)
        ax.set_title(f"{len(valid)} solved wells; shaded band ±{tolerance:g} psi" if key == "bhp" else
                     f"{len(valid)} solved wells; shaded band ±{100*tolerance:g}%", fontsize=11)
        ax.grid(alpha=.2)
        for well in valid:
            if (key == "bhp" and abs(well["latest_test"]["bhp_error"]) > 300) or (key == "pf" and well["well"] == "MPB-35"):
                t = well["latest_test"]
                ax.annotate(well["well"], (t["observed_"+key], t["predicted_"+key]),
                            xytext=(-5, 7) if key == "pf" else (5, 7),
                            ha="right" if key == "pf" else "left", textcoords="offset points", fontsize=8)
    ax = axes[1,1]
    for i, pad in enumerate(pads):
        vals = [abs(w["latest_test"]["bhp_error"]) for w in good if w["pad"] == pad]
        ax.scatter(np.full(len(vals), i), vals, color=colors[pad], s=32, alpha=.75)
        if vals:
            ax.plot([i-.22, i+.22], [np.median(vals)]*2, color="#0f172a", lw=2)
    ax.axhline(50, color="#64748b", ls="--", lw=1)
    ax.set_xticks(range(len(pads)), [f"{p}\n(n={sum(w['pad']==p for w in good)})" for p in pads])
    ax.set(ylabel="Absolute BHP error (psi)", xlabel="Pad", title="Latest usable test per well; black marks = pad medians")
    ax.grid(axis="y", alpha=.2)
    fig.supxlabel("Retrospective reproduction with unchanged model inputs. Failed solves remain in the report.\n"
                  "Daily gauge records and test-day BHP can share the same source; they are not independent measurements.", fontsize=10)
    fig.savefig(path.with_suffix(".png"), dpi=180)
    plt.close(fig)

    latest = scores["latest_test"]
    def metric(score, key, stat, digits=1):
        value = score.get(key, {}).get(stat)
        return "—" if value is None else f"{value:,.{digits}f}"
    lines = [f"# Fleet model actuality — {report['snapshot_time'][:10]}", "",
        f"Databricks access worked. The audit captured current inputs and observations in **{report['source_query_count']} bulk SELECT queries**, "
        "then ran offline with **two workers**. No calibration was refitted or saved, and no production data or deployment was changed.", "",
        f"The model currently reproduces BHP unevenly across the fleet: **{metric(latest, 'bhp_error', 'median_abs')} psi median absolute error** "
        f"on {latest['solved']} solved latest tests, with **{latest['failed']} failed solve**. "
        f"PF and oil median absolute percentage errors are **{metric(latest,'pf_error_pct','median_abs')}%** and "
        f"**{metric(latest,'oil_error_pct','median_abs')}%**, respectively.", "",
        "## Coverage and scope", "",
        f"- {coverage['app_wells']} wells in the app's supported model universe. The source gauge registry covers 487 MPU well records; this is not a model audit of all 487.",
        f"- {coverage['fresh_gauge_wells']} app wells have credible BHP (>50 psi) within 14 days; {coverage['fresh_gauge_with_recent_tests']} also have tests within 90 days.",
        f"- **{coverage['eligible_wells']} wells across {len(pads)} pads** have usable tests after their current pump installation: {', '.join(pads)}.",
        f"- {scores['all_tests']['observations']} test observations and {scores['daily']['observations']} daily operating observations scored. These overlap on some dates and should not be added as independent measurements.",
        "- Each test uses its measured PF pressure and wellhead pressure. IPR, watercut, GOR, geometry and loss coefficients remain as the app hydrates them today. Missing/zero test PF volume leaves PF accuracy unscored while BHP/oil remain usable.",
        "- Daily comparisons require credible BHP, PF pressure, production pressure and net PF >500 BPD. No centered BHP smoothing or exclusion based on model error was applied.",
        "- Prior-pump tests, installation-day tests, and circulation conflicts are excluded and listed in the CSV. Current gauge readings do not guarantee that the latest rate test is recent.", "",
        "These are **retrospective reproductions**, not an independent qualification. Saved/automatically fitted inputs may incorporate scored tests. "
        "The after-save subset is reported separately but still uses current geometry, fluid assumptions and model v2. "
        "The earlier three-well report used newly trained event fits and frozen training IPR; its 23–59 psi RMS is a different experiment. "
        "Comparisons assume the cleaned gauge BHP represents pump suction pressure; gauge calibration and pressure-datum equivalence were not independently verified.", "",
        "## Errors", "",
        "Errors are model minus measured. Latest-test statistics give each well one observation. Full-history statistics weight wells by their number of observations. "
        "Numerical error statistics cover successful solves; failures remain explicitly counted.", "",
        "| Comparison | Observations / failed | BHP median absolute / RMS, psi | PF median absolute / RMS, % | Oil median absolute / RMS, % |",
        "|---|---:|---:|---:|---:|"]
    for label, key in [("Latest usable test per well","latest_test"),("All current-era tests","all_tests"),
                       ("Daily operating observations","daily"),("Tests after known saved inputs","tests_after_saved")]:
        v=scores[key]
        lines.append(f"| {label} | {v['observations']} / {v['failed']} | {metric(v,'bhp_error','median_abs')} / {metric(v,'bhp_error','rms')} | "
                     f"{metric(v,'pf_error_pct','median_abs')} / {metric(v,'pf_error_pct','rms')} | {metric(v,'oil_error_pct','median_abs')} / {metric(v,'oil_error_pct','rms')} |")
    lines += ["", f"Among latest solved tests: **{latest['bhp_within_50psi']}/{latest['solved']}** are within 50 psi BHP; "
              f"**{latest['pf_within_10pct']}/{latest['pf_error_pct']['n']}** within 10% PF; "
              f"**{latest['oil_within_20pct']}/{latest['solved']}** within 20% oil. These are reporting bands, not acceptance standards.", "",
              "## Findings that need attention", "",
              "1. **BHP errors persist beyond stale inputs.** Updating the diagnostic IPR anchor, WC and GOR to the measured latest test reduces some errors, but large differences remain on MPB-35, MPJ-29 and MPE-48. This diagnostic explicitly uses measured BHP/oil as inputs and is not independent validation.",
              f"2. **Composition inputs often differ from tests.** {sum(abs(w.get('watercut_difference_points',0))>10 for w in selected)}/{len(selected)} wells differ by over 10 watercut percentage points. That can materially affect IPR and multiphase flow; it does not by itself identify which input is correct.",
              "3. **MPB-35's latest test reports 82,133.56 BPD PF.** It remains in the raw error statistics and strongly affects PF RMS. This record needs reconciliation with PF metering/allocation and pump identity; no value was silently corrected or discarded.",
              "4. **MPI-24 has a circulation conflict.** The current tracker/model says forward; the test pressure signals resolve as annulus PF (reverse). It is excluded pending reconciliation, rather than solving the wrong flow path.",
              "5. **MPF-73 is producing in the observations but the configured model cannot lift at maximum suction.** This is a substantive model/input failure, retained in the scorecard.",
              "6. **A good level match can hide a response mismatch.** MPB-37, MPH-19, MPJ-27 and MPM-16 have observational pressure-response slopes around 0.07–0.14 psi/psi while their frozen model is pinned. Pair statistics restrict time separation to 3–30 days, PF separation to at least 100 psi and WHP change to at most 25 psi. They are correlated and can include changing reservoir/test conditions; they do not establish causal PF response.", "",
              "Do not loosen physics acceptance tests to fit this dataset. First reconcile the highlighted measurement/configuration conflicts, then revisit the worst well models with explicit event holdouts. The shared-energy consistency tests remain a separate requirement.", "",
              "## Largest latest-test BHP errors", "",
              "| Well | Test date | Measured / modeled BHP, psi | Signed error, psi | Oil error, % | PF error, % |",
              "|---|---|---:|---:|---:|---:|"]
    for w in sorted(good,key=lambda w:abs(w["latest_test"]["bhp_error"]),reverse=True)[:10]:
        t=w["latest_test"]
        pf = f"{t['pf_error_pct']:+.1f}" if "pf_error_pct" in t else "unavailable"
        lines.append(f"| {w['well']} | {t['date']} | {t['observed_bhp']:.0f} / {t['predicted_bhp']:.0f} | {t['bhp_error']:+.0f} | {t['oil_error_pct']:+.1f} | {pf} |")
    lines += ["", "## Pad coverage", "", "Small pad samples are shown as coverage, not pad rankings.", "",
              "| Pad | Wells / failures | Median absolute BHP error, psi | Median absolute oil error, % |", "|---|---:|---:|---:|"]
    for p,v in report["pads"].items():
        lines.append(f"| {p} | {v['observations']} / {v['failed']} | {metric(v,'bhp_error','median_abs')} | {metric(v,'oil_error_pct','median_abs')} |")
    lines += ["", "## Artifacts and replay", "",
              f"![Observed versus modeled results]({path.with_suffix('.png').name})", "",
              f"- [Searchable well scorecard]({path.with_suffix('.html').name})",
              f"- [Well-level CSV, including exclusions]({path.with_suffix('.csv').name})",
              f"- [Observation-level CSV]({path.stem}_observations.csv)",
              f"- [Full metrics, fixed inputs and conditional diagnostics]({path.name})", "",
              "The raw snapshot stays in ignored `build/fleet-actuality-snapshot.pkl`. "
              "Source tables are `vw_bhp_tags`, `vw_bhp_daily_clean`, `vw_well_test`, `vw_pressure_daily`, "
              "`vw_power_fluid_volume`, well characteristics, JP tracker and saved property history. "
              "The cleaned daily gauge and pressure-view BHP agree exactly on 5,385 credible paired records in the app's 90-day window; they are the same evidence source, not a second validation.", "",
              "Replay without additional warehouse reads:", "", "```powershell", "$env:PYTHONPATH='.'",
              "./venv/Scripts/python.exe tools/fleet_actuality.py",
              f"./venv/Scripts/python.exe tools/render_fleet_actuality.py {path.as_posix()}", "```", "",
              "Refresh the snapshot explicitly with `tools/fleet_actuality.py --live --fetch-only`. "
              "Neither script writes to Databricks. Offline audit regression tests cover chronology, missing PF, direction conflicts, label isolation and failure accounting."]
    lines += ["", "Validation: eight new fleet-audit regressions pass. The final full offline suite passed 1,817 tests. "
              "The figure was visually inspected; HTML scorecard sorting/filtering and artifact links were checked locally."]
    path.with_suffix(".md").write_text("\n".join(lines)+"\n",encoding="utf-8")

    data=[]
    for w in report["wells"]:
        t=w.get("latest_test",{})
        pump=w.get("pump") or {}
        data.append(dict(well=w["well"],pad=w["pad"],pump=str(pump.get("nozzle_no") or "")+str(pump.get("throat_ratio") or ""),
            date=t.get("date"), measured=t.get("observed_bhp"), modeled=t.get("predicted_bhp"),
            error=t.get("bhp_error"), oil=t.get("oil_error_pct"),pf=t.get("pf_error_pct"),source=w["ipr_source"],
            status=w.get("exclusion") or t.get("error") or "Scored",
            flags="; ".join(w.get("input_flags",[]) + [f"{n}: {reason}" for reason,n in w["rejected_tests"].items()])))
    payload=json.dumps(data).replace("<","\\u003c")
    title=f"Fleet actuality — {report['snapshot_time'][:10]}"
    document="""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TITLE</title><style>body{font:15px system-ui;margin:32px;color:#172033;background:#f8fafc}h1{margin-bottom:8px}
p{max-width:1000px;line-height:1.5}.cards{display:flex;gap:16px;flex-wrap:wrap}.card{padding:18px;background:white;border:1px solid #cbd5e1;border-radius:8px;min-width:180px}.card b{font-size:26px;display:block}
input,select{padding:10px;margin:12px 8px 12px 0;border:1px solid #94a3b8;border-radius:4px}table{border-collapse:collapse;background:white;width:100%;font-size:13px}th,td{padding:9px;border-bottom:1px solid #e2e8f0;text-align:right}th{cursor:pointer;background:#e2e8f0;position:sticky;top:0}td:first-child,td:nth-last-child(-n+3){text-align:left}tr.bad{background:#fff1f2}img{max-width:100%;background:white}a{color:#0369a1}.scroll{overflow:auto}.muted{color:#475569}</style>
<h1>TITLE</h1><p>Current model inputs compared with actual gauge and test observations. No refitting. Local report; no external connections. Medium remains at two workers.</p>
<div class="cards"><div class="card"><b>ELIGIBLE / 90</b>wells with usable current-pump tests</div><div class="card"><b>BHP psi</b>median absolute BHP error</div><div class="card"><b>PF%</b>median absolute PF error</div><div class="card"><b>OIL%</b>median absolute oil error</div></div>
<p class="muted">One latest usable test per well. Failed solves and exclusions remain visible. Missing PF volume does not exclude BHP/oil comparisons. Historical reproduction is not independent qualification.</p>
<p><a href="STEM.md">Full report</a> · <a href="STEM.csv">Download well CSV</a> · <a href="STEM_observations.csv">Download observation CSV</a></p>
<input id="query" placeholder="Search wells, pads, flags…" aria-label="Search wells"><select id="mode" aria-label="Filter status"><option value="all">All wells</option><option value="scored">Scored wells</option><option value="excluded">Failures and exclusions</option></select><span id="count"></span>
<div class="scroll"><table><thead><tr id="head"></tr></thead><tbody id="body"></tbody></table></div>
<h2>Measured versus modeled</h2><img src="STEM.png" alt="BHP, oil and PF prediction comparisons across pads">
<script type="application/json" id="rows">PAYLOAD</script><script>
const rows=JSON.parse(document.getElementById('rows').textContent);const columns=[['well','Well'],['pad','Pad'],['pump','Pump'],['date','Test date'],['measured','Measured BHP'],['modeled','Modeled BHP'],['error','BHP error'],['oil','Oil error %'],['pf','PF error %'],['source','IPR source'],['status','Status'],['flags','Notes']];let sort='well',ascending=true;
for(const [key,label] of columns){const th=document.createElement('th');th.textContent=label;th.onclick=()=>{ascending=sort===key?!ascending:true;sort=key;draw()};document.getElementById('head').appendChild(th)}
function draw(){const query=document.getElementById('query').value.toLowerCase(),mode=document.getElementById('mode').value;const shown=rows.filter(r=>JSON.stringify(r).toLowerCase().includes(query)&&(mode==='all'||(mode==='scored'?r.status==='Scored':r.status!=='Scored'))).sort((a,b)=>{let x=a[sort],y=b[sort];if(x==null)return 1;if(y==null)return -1;let v=typeof x==='number'?x-y:String(x).localeCompare(String(y));return ascending?v:-v});const body=document.getElementById('body');body.replaceChildren();for(const r of shown){const tr=document.createElement('tr');if(Math.abs(r.error||0)>100||r.status.includes('Error'))tr.className='bad';for(const [key] of columns){const td=document.createElement('td');const v=r[key];td.textContent=v==null?'—':typeof v==='number'?v.toFixed(1):String(v);tr.appendChild(td)}body.appendChild(tr)}document.getElementById('count').textContent=shown.length+' wells'}document.getElementById('query').oninput=draw;document.getElementById('mode').onchange=draw;draw();
</script></html>"""
    for key,value in [("TITLE",html.escape(title)),("ELIGIBLE",str(len(selected))),
                      ("BHP",metric(latest,"bhp_error","median_abs")),("PF%",metric(latest,"pf_error_pct","median_abs")+"%"),
                      ("OIL%",metric(latest,"oil_error_pct","median_abs")+"%"),("STEM",html.escape(path.stem)),("PAYLOAD",payload)]:
        # Card substitutions must not replace BHP labels throughout the document.
        document=document.replace("<b>BHP psi</b>",f"<b>{value} psi</b>") if key=="BHP" else document.replace(key,value)
    path.with_suffix(".html").write_text(document,encoding="utf-8")


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("report",type=Path)
    render(parser.parse_args().report)
