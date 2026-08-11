"""Eyeball diagnostic: measured daily (PF pressure, BHP) vs the model's
suction response curve at old (single-point) and new (event-fit) parameters.

Wells: MPM-64, MPM-28, MPM-45 (the beta-gate verdict wells). Fitted params
hardcoded from the 2026-08-10 multipoint harness run (scripts/
multipoint_validation.py output) - rerunning the fits here would add 3 min
for identical numbers.

Run:
  cd C:/dev/woffl_gui/woffl_gui
  PYTHONPATH=. venv/Scripts/python.exe scripts/response_eyeball.py
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# (well, new (ken, kth, kdi, fnz, mach_crit)) - old params come from the config
CASES = [
    ("MPM-64", (0.008, 0.22, 0.21, 1.03, 1.09)),
    ("MPM-28", (0.033, 0.05, 0.10, 1.03, 1.00)),
    ("MPM-45", (0.024, 0.24, 0.37, 1.08, 1.32)),
]
PPF_GRID = [1800.0 + 100.0 * i for i in range(19)]  # 1800..3600


def model_curve(wc, nozzle, throat, ken, kth, kdi, fnz, mach_crit):
    from woffl.assembly.network_optimizer import NetworkOptimizer
    from woffl.assembly import solopump as so
    from woffl.geometry.jetpump import JetPump

    wellbore, wellprof, inflow, res_mix, prop_pf = (
        NetworkOptimizer._create_well_objects(wc)
    )
    jp = JetPump(nozzle, throat, knz=0.01, ken=ken, kth=kth, kdi=kdi)
    jp.dnz = jp.dnz * math.sqrt(fnz)
    xs, ys = [], []
    for ppf in PPF_GRID:
        try:
            psu, _s, _q, _f, _n, _m = so.jetpump_solver(
                wc.surf_pres, wc.form_temp, ppf, jp, wellbore, wellprof,
                inflow, res_mix, prop_pf, "reverse", mach_crit=mach_crit,
            )
            xs.append(ppf)
            ys.append(psu)
        except Exception:
            continue
    return xs, ys


def main() -> None:
    from server.services.evidence import _fleet_pressure_daily
    from server.services.optimizer_runs import _build_configs, _current_and_tests
    from woffl.assembly.pf_pressure import resolve_pf_pressure
    from woffl.assembly.jp_history import get_current_pump
    from server.services import datasources

    notes: list = []
    prov: dict = {}
    configs = _build_configs(["M"], set(), [], notes, prov)
    by_name = {c.well_name: c for c in configs}
    current, _ = _current_and_tests(list(by_name))
    jp_hist, _src = datasources.jp_history_safe()
    daily = _fleet_pressure_daily()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (well, new) in zip(axes, CASES):
        wc = by_name[well]
        cc = current[well]
        cur = get_current_pump(jp_hist, well)
        era = pd.to_datetime(cur["date_set"]) if cur else None

        d = daily[daily["well"] == well].copy()
        d["ppf"] = [
            resolve_pf_pressure(t, a)[0]
            for t, a in zip(d["tubing_prs"], d["inn_ann_prs"])
        ]
        d["bhp"] = pd.to_numeric(d["btmhole_prs"], errors="coerce")
        d = d.dropna(subset=["ppf", "bhp"])
        d = d[(d["bhp"] > 50) & d["ppf"].between(800, 5500)]
        in_era = d[d["sample_date"] >= era] if era is not None else d
        prior = d[d["sample_date"] < era] if era is not None else d.iloc[0:0]

        ax.scatter(prior["ppf"], prior["bhp"], s=8, c="tab:gray", alpha=0.35,
                   label="daily, prior pump")
        ax.scatter(in_era["ppf"], in_era["bhp"], s=12, c="tab:blue",
                   label="daily, current era")

        old_k = (wc.ken_well or 0.03, wc.kth_well or 0.30, wc.kdi_well or 0.40)
        xs, ys = model_curve(wc, cc[0], cc[1], *old_k, 1.0, 1.0)
        ax.plot(xs, ys, c="tab:red", lw=1.5, ls="--", label="model, old cal")
        xs, ys = model_curve(wc, cc[0], cc[1], *new)
        ax.plot(xs, ys, c="tab:green", lw=1.8, label="model, event fit")

        era_s = era.date() if era is not None else "?"
        ax.set_title(f"{well} ({cc[0]}{cc[1]}, era {era_s})", fontsize=10)
        ax.set_xlabel("PF surface pressure (psi)")
        ax.set_xlim(1750, 3650)
    axes[0].set_ylabel("BHP / model suction (psi)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(
        "Suction response: daily field data vs model, old vs event calibration",
        fontsize=11,
    )
    fig.text(
        0.99, 0.01,
        "scripts/response_eyeball.py | vw_pressure_daily 12 mo | 2026-08-10",
        ha="right", fontsize=7, color="gray",
    )
    fig.tight_layout()
    out = "scripts/response_eyeball.png"
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
