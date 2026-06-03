"""Headless back-test probe — runs the IPR-fit + ΔOil parity machinery on REAL
Databricks well-test data, with NO Streamlit, so we (Claude + Scott) can see real
numbers and iterate on the back-test knobs together.

The windowed pseudo-Pr fits, the then/now anchors, and the depletion diagnosis are
the EXACT engine functions the app uses. The ΔOil prediction here is an anchored
approximation of the app's (it pins each well's IPR at its NOW operating point),
enough to compare database-Pr vs fitted-pseudo-Pr parity.

Usage (from the inner repo root):
    MSYS_NO_PATHCONV=1 venv/Scripts/python.exe tools/hpi_backtest_probe.py --pad G --months 12
    ... --pad ALL          # every pad
    ... --json out.json     # also dump structured results
"""
import argparse
import json
import os
import sys

# Run from the inner repo root so the LOCAL woffl (with woffl.gui) wins over any
# stale pip-installed woffl — see CLAUDE.md "Library imports resolve locally".
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.pop("bricks_http", None)  # MSYS guard so databricks_client reads .env cleanly

from datetime import datetime

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from woffl.flow.inflow import InFlow
from woffl.gui.scotts_tools import header_engine as he
from woffl.gui.scotts_tools import header_trend as ht
from woffl.gui.scotts_tools._common import fetch_well_tests_raw, pad_from_mp_name
from woffl.gui.utils import load_well_characteristics


def _db_doil(bhp_then, bhp_now, oil_now, form_wc, res_pres):
    """Approx database-Pr ΔOil pred: a Vogel IPR pinned at the NOW operating point
    (oil_now @ bhp_now) with the database reservoir pressure."""
    try:
        if any(pd.isna(v) for v in (bhp_then, bhp_now, oil_now, res_pres)) or oil_now <= 0:
            return np.nan
        wc = float(form_wc) if pd.notna(form_wc) else 0.0
        inflow = InFlow(qwf=float(oil_now), pwf=float(bhp_now), pres=float(res_pres))
        return float(inflow.oil_flow(float(bhp_now), "vogel")
                     - inflow.oil_flow(float(bhp_then), "vogel"))
    except Exception:
        return np.nan


def _vogel_slope(bhp, qmax, pr):
    """Local IPR slope dOil/dBHP (BOPD per psi, negative) of a Vogel curve at ``bhp``
    — the BLUE fit's rate sensitivity at the operating point."""
    if pr is None or pd.isna(pr) or pr <= 0 or pd.isna(qmax) or pd.isna(bhp):
        return np.nan
    return -float(qmax) / float(pr) * (0.2 + 1.6 * min(max(float(bhp), 0.0), float(pr)) / float(pr))


def run(pads, months):
    ch = load_well_characteristics()
    ch["pad"] = ch["Well"].map(pad_from_mp_name)
    tr = fetch_well_tests_raw(months)
    res_map = dict(zip(ch["Well"], ch["res_pres"]))
    sch_map = dict(zip(ch["Well"], ch["is_sch"]))
    want = None if (len(pads) == 1 and pads[0] == "ALL") else set(pads)
    wells = sorted(ch["Well"]) if want is None else sorted(ch[ch["pad"].isin(want)]["Well"])

    rows = []
    for w in wells:
        tw = tr[tr["well"] == w]
        if tw.empty:
            continue
        formation = "Schrader" if bool(sch_map.get(w, True)) else "Kuparuk"
        a = he.backtest_anchors(tw)
        if not a:
            continue
        # BLUE line: one Vogel fit to all the well's points (the rate-impact curve).
        fit = he.fit_well_ipr(tw, pr_hi=he.pr_hi_for_formation(formation))
        pr1 = fit["pr"] if fit else np.nan
        bnd1 = bool(fit["pr_at_bound"]) if fit else False
        sig = he.depletion_signature(tw)
        geo = sig["verdict"]
        wc_recent = pd.to_numeric(tw.sort_values("WtDate")["form_wc"], errors="coerce").dropna()
        wc = float(wc_recent.iloc[-1]) if not wc_recent.empty else 0.0
        # ROBUST rate sensitivity: linear liquid-vs-BHP slope (BLPD/100psi), then ×(1−WC)
        # → OIL sensitivity (BOPD per 100 psi BHP). This is the units-correct lever.
        liq100 = sig["rate_bhp_slope"] * 100.0 if pd.notna(sig["rate_bhp_slope"]) else np.nan
        oil100 = liq100 * (1.0 - wc) if pd.notna(liq100) else np.nan

        doil_actual = a["d_oil"]
        obs100 = (doil_actual / a["d_whp"] * 100.0) if abs(a["d_whp"]) >= 1e-6 else np.nan
        rows.append({
            "well": w, "form": formation, "pad": pad_from_mp_name(w), "n": a["n_tests"],
            "res_db": res_map.get(w), "pr1": pr1, "bnd1": bnd1, "corr": sig["corr"], "geo": geo,
            "wc": wc, "dWHP": a["d_whp"], "dBHP": a["d_bhp"], "dOil": doil_actual,
            "liq100": liq100, "oil100": oil100, "obs100": obs100,
        })
    return pd.DataFrame(rows)


def estimate_chain(pads, months, d_whp):
    """The full deterministic per-well chain on REAL data:
       ΔWHP → ①correlation(own/group) → ΔBHP → ②liquid IPR(own/donor) → ΔLiquid → ③×(1-WC) → ΔOil
    Resolves donors exactly as the app will, then rolls up to a field estimate."""
    ch = load_well_characteristics()
    ch["pad"] = ch["Well"].map(pad_from_mp_name)
    tr = fetch_well_tests_raw(months)
    sch_map = dict(zip(ch["Well"], ch["is_sch"]))
    want = None if (len(pads) == 1 and pads[0] == "ALL") else set(pads)
    wells = sorted(ch["Well"]) if want is None else sorted(ch[ch["pad"].isin(want)]["Well"])

    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - relativedelta(months=months)).strftime("%Y-%m-%d")
    try:
        well_dfs, _ = ht.fetch_header_trends(tuple(sorted(wells)), start, end)
    except Exception as e:
        print(f"empirical fetch failed: {e}"); well_dfs = {}

    base = {}
    for w in wells:
        tw = tr[tr["well"] == w]
        if tw.empty:
            continue
        formation = "Schrader" if bool(sch_map.get(w, True)) else "Kuparuk"
        a = he.backtest_anchors(tw)
        fit = he.fit_well_ipr(tw, pr_hi=he.pr_hi_for_formation(formation))
        wcs = pd.to_numeric(tw.sort_values("WtDate")["form_wc"], errors="coerce").dropna()
        wc = float(wcs.iloc[-1]) if not wcs.empty else 0.0
        # ① own correlation = within-day BHP~WHP slope IF responsive
        cf = (ht.fit_well(well_dfs[w]).get("BHP~WHP")) if w in well_dfs else None
        responsive = cf is not None and ht.classify_response(cf) == "responsive" and pd.notna(cf.mean_slope)
        base[w] = {
            "formation": formation, "pad": pad_from_mp_name(w),
            "bhp_now": (a or {}).get("bhp_now"), "wc": wc,
            "qmax": fit["qmax"] if fit else np.nan, "pr": fit["pr"] if fit else np.nan,
            "corr_own": float(cf.mean_slope) if responsive else np.nan,
            "corr_cls": ht.classify_response(cf) if cf is not None else "no fit",
        }
    if not base:
        return pd.DataFrame(), 0.0, 0.0, 0.0

    # Build the wells dict and run the REAL engine orchestrator (same as the app; no
    # sonic here since the probe doesn't run physics → upper bound on the loss).
    wells = {
        w: {"own_corr": (b["corr_own"] if pd.notna(b["corr_own"]) else None),
            "sonic": False, "lift": "JP",
            "qmax": (b["qmax"] if pd.notna(b["qmax"]) else None),
            "pr": (b["pr"] if pd.notna(b["pr"]) else None),
            "pad": b["pad"], "formation": b["formation"],
            "bhp_now": b["bhp_now"], "wc": b["wc"]}
        for w, b in base.items()
    }
    est = he.estimate_header_impacts(wells, d_whp)
    gstats = he.group_correlation_stats(wells)
    lo_ovr = {k: max(0.0, m - s) for k, (m, s) in gstats.items()}
    hi_ovr = {k: m + s for k, (m, s) in gstats.items()}
    est_lo = he.estimate_header_impacts(wells, d_whp, lo_ovr)
    est_hi = he.estimate_header_impacts(wells, d_whp, hi_ovr)

    def _fsum(e):
        return sum(v["doil"] for v in e.values() if pd.notna(v["doil"]))

    rows = [{
        "well": w, "pad": base[w]["pad"], "form": base[w]["formation"],
        "corr_cls": base[w]["corr_cls"], "corr": e["corr"], "csrc": e["corr_src"],
        "isrc": e["ipr_src"], "wc": round(base[w]["wc"], 2), "bhp_now": base[w]["bhp_now"],
        "dBHP": e["dbhp"], "dLiq": e["dliquid"], "dOil": e["doil"],
        "dOil_lo": est_lo[w]["doil"], "dOil_hi": est_hi[w]["doil"], "conf": e["conf"],
    } for w, e in est.items()]
    return pd.DataFrame(rows), _fsum(est), _fsum(est_lo), _fsum(est_hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pad", default="G", help="comma list e.g. G,H,I,J  or  ALL")
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--minwhp", type=float, default=15.0, help="min |ΔWHP| to count as a real header move")
    ap.add_argument("--dwhp", type=float, default=None, help="run the full ΔOil chain for this header rise (psi)")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    pads = [p.strip().upper() for p in args.pad.split(",") if p.strip()]

    if args.dwhp is not None:
        df, field, flo, fhi = estimate_chain(pads, args.months, args.dwhp)
        if df.empty:
            print("no wells"); return
        pd.set_option("display.width", 240); pd.set_option("display.max_columns", 40)
        show = df.copy()
        for c in ["corr", "bhp_now", "dBHP", "dLiq", "dOil"]:
            show[c] = show[c].round(1)
        print(f"\n=== FULL CHAIN: +{args.dwhp:.0f} psi header on pads {','.join(pads)}, {args.months} mo ===")
        print("(corr=dBHP/dWHP used; csrc/isrc = correlation/IPR source own|group|donor; dOil per well)")
        print(show.to_string(index=False))
        tot = df[df["dOil"].notna()]
        blo, bhi = sorted([flo, fhi])
        print("\n-- PER-PAD ROLLUP (ΔOil + 1σ CI from group corr ±σ) --")
        pr_rows = []
        for pad, g in df.groupby("pad"):
            gv = g[g["dOil"].notna()]
            lo, hi = sorted([gv["dOil_lo"].sum(), gv["dOil_hi"].sum()])
            pr_rows.append({"pad": pad, "wells": len(g), "dOil": round(gv["dOil"].sum()),
                            "CI_lo": round(lo), "CI_hi": round(hi),
                            "own/own": int((g["conf"] == "high").sum())})
        print(pd.DataFrame(pr_rows).to_string(index=False))
        print(f"\nFIELD ΔOil for +{args.dwhp:.0f} psi: {field:+,.0f} BOPD  ·  1σ CI [{blo:+,.0f}, {bhi:+,.0f}] "
              f"·  {len(tot)} of {len(df)} wells")
        print(f"  by confidence: " + ", ".join(
            f"{k} {g['dOil'].sum():+,.0f} ({len(g)}w)" for k, g in tot.groupby('conf')))
        print("  NOTE: probe has no physics → no sonic wells zeroed; the app number will be lower.")
        if args.json:
            with open(args.json, "w", encoding="utf-8") as f:
                json.dump({"pads": pads, "d_whp": args.dwhp,
                           "rows": json.loads(df.to_json(orient="records"))}, f, indent=2)
            print(f"wrote {args.json}")
        return

    df = run(pads, args.months)
    if df.empty:
        print(f"No back-testable wells on pad(s) {args.pad} ({args.months} mo).")
        return

    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 40)
    show = df[["well", "form", "n", "wc", "pr1", "bnd1", "corr", "geo",
               "dWHP", "dBHP", "dOil", "liq100", "oil100", "obs100"]].copy()
    for c in ["pr1", "dWHP", "dBHP", "dOil", "liq100", "oil100", "obs100"]:
        show[c] = show[c].round(0)
    show["corr"] = show["corr"].round(2)
    show["wc"] = show["wc"].round(2)
    print(f"\n=== Rate impact + depletion (IPR on TOTAL LIQUID): pads {','.join(pads)}, {args.months} mo, {len(df)} wells ===")
    print("(liq100 = liquid-vs-BHP slope; oil100 = liq100×(1−WC) = OIL per 100 psi BHP; obs100 = ΔOil/ΔWHP)")
    print(show.to_string(index=False))

    # ── CONFIDENCE INTERVAL on header-pressure impact ────────────────────────
    # The robust sensitivity is the liquid-vs-BHP LINEAR slope × (1−WC) = OIL/100psi BHP,
    # and it equals the IPR response ONLY on back-pressure wells (corr<0). Depleting
    # wells' slope is the depletion line, not a header response → excluded. (obs100 is
    # shown but unusable: the real header moves were only 15–56 psi, so it's all noise.)
    bp = df[(df["geo"] == "back-pressure") & df["oil100"].notna()]
    print(f"\n-- HEADER-PRESSURE IMPACT (BOPD oil per 100 psi BHP, ≈ per 100 psi header at 1:1) --")
    print(f"   responsive population: {len(bp)} back-pressure wells "
          f"(depleting {int((df['geo']=='depleting').sum())} excluded; "
          f"flat/mixed {int((df['geo']=='flat/mixed').sum())} = no clear response)")
    if len(bp) >= 3:
        o = bp["oil100"]
        med, mean, sd = o.median(), o.mean(), o.std()
        sem = sd / np.sqrt(len(o))
        print(f"   OIL-vs-BHP impact: median {med:+,.0f} · mean {mean:+,.0f} ± {sd:,.0f} (1σ) "
              f"· range [{o.min():+,.0f}, {o.max():+,.0f}]")
        print(f"   ≈95% CI on the mean: [{mean-1.96*sem:+,.0f}, {mean+1.96*sem:+,.0f}] BOPD per 100 psi")
    else:
        print(f"   only {len(bp)} responsive wells — per-well oil: "
              + ", ".join(f"{r.well} {r.oil100:+,.0f}" for r in bp.itertuples()))
    print(f"\n-- depletion mix -- {dict(df['geo'].value_counts())}")
    print(f"-- blue fit pinned at cap: {int(df['bnd1'].sum())}/{int(df['pr1'].notna().sum())} fittable --")

    if args.json:
        out = {"pads": pads, "months": args.months,
               "rows": json.loads(df.to_json(orient="records"))}
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
