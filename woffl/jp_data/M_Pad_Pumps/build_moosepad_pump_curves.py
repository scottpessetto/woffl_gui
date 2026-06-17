#!/usr/bin/env python3
"""
build_moosepad_pump_curves.py
-------------------------------------------------------------------------------
Regenerates the Moose Pad (Mod 42) injection-pump curve files.

Moose Pad runs SIX Schlumberger REDA HPS pumps in a two-stage parallel/series train:

    LP bank: 3x "A" pump  P-4220A/B/C  REDA N1400N-A, 13 stages  (1000/10in housing)
    HP bank: 3x "B" pump  P-4230A/B/C  REDA M675-A,  41 stages  (862/8.62in)

The LP bank (parallel) lifts produced water to a ~1,400 psig common header that
feeds the Disposal + Kuparuk injection headers AND the suction of the HP bank.
The HP bank (parallel) takes that ~1,400 psig and boosts to ~3,500 psig Power Fluid.
So the two banks are in SERIES (LP -> HP) while each bank is 3 pumps in PARALLEL.

The head/power polynomials below were fit (this script just evaluates them) to the
Schlumberger data-sheet + "Moosepad HPS set up sheet" tabular points (60 Hz, SG 1.05):
shut-in, BEP, max-rate, and the constant-pressure scenario.  Fit reproduces the
manufacturer's shut-in / BEP / power points to <0.5%, and the live SCADA HP point
(16,480 BPD, 3,495 psig) to within 4 psi at ~54 Hz.

Head is stored in FEET (specific-gravity independent); boost psi = head_ft * SG / 2.31.
Standard library only.
"""

import csv
import os

# SG: Schlumberger DESIGN value was 1.05; field produced water is ~1.02-1.04 (operator).
# head_ft below is SG-INDEPENDENT (the real pump curve) -> boost_psi = head_ft * SG / 2.31.
SG = 1.03          # field produced-water specific gravity (mid of 1.02-1.04)
REF_RPM = 3569     # 60 Hz reference speed (both pumps)

# FIELD WEAR DERATE: live SCADA (2026-06-16) shows the installed pumps make ~0.91x the
# as-new head at the displayed flow/speed (range ~0.87-0.96 at SG 1.03) while drawing
# on-curve power -> efficiency loss consistent with solids/erosive stage wear. Columns
# below are the AS-NEW (clean) reference; multiply by FIELD_HEAD_FACTOR for current perf.
FIELD_HEAD_FACTOR = 1.00   # set to ~0.91 to emit field-representative head instead of as-new

# ----- 60 Hz curve fits, coeffs HIGH->LOW order: c3*Q^3 + c2*Q^2 + c1*Q + c0 -----
PUMPS = {
    "A_LP_4220": {
        "model": "REDA N1400N-A", "stages": 13, "tags": "P-4220A/B/C",
        "head_ft": [-3.166675e-12, -2.796207e-08, -5.711101e-03, 3.493744e+03],
        "bhp":     [-2.789893e-12,  3.245156e-07,  1.626390e-03, 8.106112e+02],
        "qmin": 14276, "qbep": 49791, "qmax": 65262, "qgrid_max": 68000, "qstep": 2000,
        "design_suction_psig": 400,
    },
    "B_HP_4230": {
        "model": "REDA M675-A", "stages": 41, "tags": "P-4230A/B/C",
        "head_ft": [-2.859001e-11, -1.786390e-07, -3.232443e-02, 6.535052e+03],
        "bhp":     [-2.789838e-11,  1.481445e-06,  1.260881e-03, 7.564934e+02],
        "qmin": 8351, "qbep": 27708, "qmax": 34798, "qgrid_max": 37000, "qstep": 1000,
        "design_suction_psig": 1500,
    },
}

# Motor current: amps ~= k * BHP, calibrated to the Schlumberger design point
# (183 A at design HP).  Both are 1500 HP / 4160 V Toshiba; VFD trip 195.8 A, FLA 178 A.
AMP_PER_HP = {"A_LP_4220": 183.0 / 1350.7, "B_HP_4230": 183.0 / 1396.8}
AMP_TRIP = 195.8
AMP_FLA = 178.0


def poly(c, q):                      # c in high->low order
    r = 0.0
    for k in c:
        r = r * q + k
    return r


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    for key, p in PUMPS.items():
        path = os.path.join(out_dir, f"MoosePad_{key}_curve_for_repo.csv")
        k_amp = AMP_PER_HP[key]
        rows = []
        q = 0
        while q <= p["qgrid_max"]:
            h_ft = poly(p["head_ft"], q) * FIELD_HEAD_FACTOR
            bhp = poly(p["bhp"], q)
            boost = h_ft * SG / 2.31
            in_range = 1 if p["qmin"] <= q <= p["qmax"] else 0
            rows.append([
                q,
                round(h_ft, 1),
                round(boost, 1),
                round(boost + p["design_suction_psig"], 1),
                round(bhp, 1),
                round(bhp * k_amp, 1),
                in_range,
            ])
            q += p["qstep"]
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "flow_bpd_60hz", "head_ft", f"boost_psi_SG{SG}",
                f"discharge_psig_atDesignSuction{p['design_suction_psig']}",
                "bhp_hp", "amps_60hz_est", "in_recommended_range",
            ])
            w.writerows(rows)
        print(f"wrote {os.path.basename(path)} ({len(rows)} rows)  "
              f"{p['model']} {p['stages']}stg  shutoff boost {poly(p['head_ft'],0)*SG/2.31:.0f} psi  "
              f"amps@trip->BHP {AMP_TRIP/k_amp:.0f}hp")


if __name__ == "__main__":
    main()
