# I-Pad Booster Pump Curves — Summary & How to Use

**Prepared:** 2026-06-16 · **Validated against live SCADA (6/16/2026), Summit's HPS selection sheet & 2021 factory tests**
**Source folders:** `P-0901 HP Booster\`, `P-1021 LP Booser\` (Summit/Halliburton HPS data sheets, GA drawings, factory test reports, coefficient file) + I-Pad SCADA screens

---

## 1. What the pumps are

I-Pad runs a **two-stage produced-water / power-fluid injection train**. Both units are Summit
(Halliburton) HPS horizontal pumping systems built from the **same pump stage**
(`SN35000 XRC CCW`, 950 series, 10 in housing), driven by **VFDs** — so one stage curve drives both;
you just multiply by the stage count.

| Item | **P-0901 "HP Booster"** (PUMP-32-0901) | **P-1021 "LP Booser"** (PUMP-32-1021) |
|---|---|---|
| Role in train | High-pressure stage (final) | Low-pressure stage (first) |
| Pump stage | Summit SN35000CCW, 950/10 | Summit SN35000CCW, 950/10 |
| **Stages** | **17 (single section)** | **26 — TANDEM 17 + 9 on one shaft** |
| Motor | **1250 HP**, 4160 V, TMEIC, VFD | **1500 HP**, 4160 V, 5812USS, VSD |
| **Motor current limit** | **154 A** (TMEIC trip) | **192 A** (VFD drive); trip 209 A; FLA 216 A |
| Intake / discharge | 8" CL1500 / **6" CL2500** RTJ | 8" CL1500 / 6" CL1500 RTJ |
| Thrust chamber | S2001/S3001 Inconel, oil-cooled | S3001 Inconel, oil-cooled |
| Seal / coupling | Flowserve UHTW/DHTW 2250 / PSC1358 | Flowserve UHTW/DHTW 2250 / PSC1358 |

> **The key facts:**
> 1. **Both pumps use the identical SN35000 stage** — one per-stage curve (below), scaled ×17 (HP) or ×26 (LP).
> 2. **P-1021 LP is a 26-stage tandem (17 + 9)**, two pump sections in series on one shaft (GA item 9 + item 10).
>    Same tandem architecture as the S-Pad boosters (41 + 38 = 79).
> 3. **The two units run in series** — confirmed by live SCADA:
>    `Power Fluid Separator (~217 psig) → P-1021 LP (→2104 psig) → P-0901 HP (→3408 psig) → Power Fluid Wells`.
>    The LP's 2104 psig discharge **is** the HP's 2082 psig intake.
> 4. **These run on VFDs and are constrained by motor amps** — and the **LP is currently pinned at its
>    192 A drive limit** (see §3). The pump curve tells you what it *can* make; the amp limit caps how hard it runs.

---

## 2. Validation against live SCADA (6/16/2026 ~08:30)

The model is Summit's published stage coefficients (`SN35000 Coeff.xlsx`, 3500 RPM / 60 Hz), with
**SG tuned to 1.04** (the value the live differentials imply; Summit's sheet used 1.01).

| | **P-0901 HP** | **P-1021 LP** |
|---|---|---|
| Speed | 59.3 Hz | 57.3 Hz |
| Flow | 32,363 BPD | 32,869 BPD |
| Suction → Discharge | 2,082 → 3,408 psig | 217 → 2,104 psig |
| **Live dP** | **1,326 psid** | **1,887 psid** |
| **Model dP** | **1,336 psid** ✓ | **1,874 psid** ✓ |
| Live motor amps | 140.6 A | 190.9 A |
| Model amps | 140.6 A (calibrated) | 190.9 A (calibrated) |

Other anchors: shut-off head 234.5 ft/stage vs 2021 factory tests (242 & 228, avg **235**) ✓ · BEP 41,250 BPD ✓ ·
recommended range 33,000–49,500 BPD ✓ · reproduces Summit's selection sheet (57 Hz/34,000 BPD → 1,156 vs 1,155 psi).

---

## 3. Motor amp limits — the real operating constraint

Both pumps are VFD-driven and **limited by motor current, not just by the pump curve.** Amps track shaft
power (`amps ≈ k × BHP`), so they rise with flow and with speed (`BHP ∝ (Hz/60)³`).

| | **P-0901 HP** | **P-1021 LP** |
|---|---|---|
| VFD/FLA limit | **154 A** (= trip) | **192 A** (drive limit) |
| Trip | 154 A, short delay | 209 A, short delay |
| Motor FLA / iFix temp limit | — | 216 A |
| **Now** | 140.6 A | **190.9 A** |
| Headroom | ~13 A (~9%) | **~1 A — essentially none** |
| BHP ceiling at limit | ~1,036 hp | ~1,338 hp |

- **P-1021 LP is amp-limited.** At 190.9 A it is sitting on the 192 A drive limit, which is why SCADA shows
  **"PUMP OUT OF CURVE"** and the drive holds it at **57.3 Hz**. To run the same point at 60 Hz it would draw
  **~220 A** — far over the limit. **It cannot make more flow or head without exceeding 192 A**; the only ways
  to gain are to lower its discharge pressure (its load) or accept the speed cap.
- **P-0901 HP has ~9% headroom** (140.6 of 154 A). It is alarming **DISCHARGE PRESS HIHI** (PIC-32-0903 SP 3,400 psig),
  so it is pressure-limited at the top end, not amp-limited yet — but its 1,250 HP motor reaches 154 A around **~1,036 BHP**.

---

## 4. The files

| File | What it is |
|---|---|
| `I-Pad_pump_curves_for_repo.csv` | **The main table.** Per-stage head/BHP/eff at 60 Hz, plus ready-made dP, BHP **and estimated-amps** columns for the 17-stage HP and 26-stage LP pumps. |
| `I-Pad_pump_curve_meta.json` | Machine-readable: fitted polynomials, both pump configs, motor-amp limits, live operating points, affinity rules, provenance. |
| `build_ipad_pump_curves.py` | Generator (standard-library Python). Edit `SG`, stage counts, amp calibration, or flow grid and re-run. |
| `README_I-Pad_Booster_Pumps.md` | This file. |

---

## 5. How to estimate pressure & amps

Each unit is a **single series train** (all stages see the same flow — no parallel split):

```
head_ft   = n_stages * head_per_stage(Q)        (n = 17 HP, 26 LP; Q in BPD at 60 Hz)
dP_psi    = head_ft * SG / 2.31                  (SG ~ 1.04 produced water)
discharge = intake_psi + dP_psi
BHP       = n_stages * bhp_per_stage(Q)          (water; x~1.04 for actual shaft power)
amps      ~ k * BHP                              (k = 0.149 HP unit, 0.144 LP unit)  -> compare to limit
```

**Per-stage head curve (60 Hz, head in feet, SG-independent):**

```
head_ft_per_stage =  234.547 + 4.33098e-4*Q - 2.23911e-7*Q^2
                  +  8.68018e-12*Q^3 - 1.41535e-16*Q^4 + 7.45683e-22*Q^5   (Q in BPD, 0–60,000)
```

**On a VFD at other speeds:** `Q ∝ Hz/60`, `head ∝ (Hz/60)²`, `BHP & amps ∝ (Hz/60)³`.

### P-0901 HP — at 59.3 Hz, intake 2,082 psig (amp limit 154 A)

| Flow (BPD) | dP (psi) | Discharge (psig) | BHP | Amps |
|---|---|---|---|---|
| 28,000 | 1,397 | 3,479 | 894 | 133 |
| 30,000 | 1,370 | 3,452 | 918 | 137 |
| **32,363** *(live)* | **1,336** | **3,418** | **945** | **141** |
| 34,000 | 1,312 | 3,394 | 964 | 143 |
| 36,000 | 1,281 | 3,363 | 985 | 147 |

### P-1021 LP — at 57.3 Hz, intake 217 psig (amp limit 192 A drive / 209 A trip)

| Flow (BPD) | dP (psi) | Discharge (psig) | BHP | Amps |
|---|---|---|---|---|
| 28,000 | 1,976 | 2,193 | 1,250 | 179 |
| 30,000 | 1,935 | 2,152 | 1,284 | 184 |
| **32,869** *(live — at limit)* | **1,874** | **2,091** | **1,330** | **191** |
| 34,000 | 1,848 | 2,065 | 1,348 | 193 ⚠ over 192 |
| 36,000 | 1,800 | 2,017 | 1,377 | 198 ⚠ |

---

## 6. Assumptions & caveats

- **Series train, not parallel.** P-1021 (LP) → P-0901 (HP); each pump's stages are in series, heads add.
- **Amp limit governs the LP.** Tables show what the curve *can* make; the LP can't actually exceed 192 A
  (≈ 32–33k BPD at 57.3 Hz). Treat the LP rows above ~191 A as unreachable at this speed/suction.
- **SG = 1.04** (live-calibrated). Head in feet is SG-independent; rescale dP by `SG/1.04` for other fluids.
  Pump *efficiency* is computed on water (SG 1.0), per the catalog — peaks ~77.5% near BEP.
- **Amps model** `amps ≈ k·BHP` is calibrated to the 6/16/2026 point (HP k=0.149, LP k=0.144). It is accurate
  near the operating region; power factor / efficiency drift as you move far from it. Use it for trend, not protection.
- **Speed.** Curve is the 60 Hz (3500 RPM) curve; HP runs ~59.3 Hz, LP ~57.3 Hz. Both VFD.
- **Pump capability vs operating point.** The curve gives what a unit makes at a flow; actual flow is set by
  where the pump curve crosses the system curve (and, for the LP, by the amp cap).
- **Data basis:** Summit `SN35000 Coeff.xlsx` (3500 RPM stage fit), cross-checked vs the two 2021 factory
  section tests (WO 107012584 = 17-stg, 107012585 = 9-stg), the Summit HPS differential-pressure selection
  sheet, and live I-Pad SCADA (6/16/2026).

---

## 7. Underlying factory test data (per stage, SG 1.0, Summit test bench, 2021-08-27)

Head/power are **per stage** (the catalog basis); near-identical curves for a 17-stage and a 9-stage build
confirm these are normalized per-stage values.

| Flow (BPD) | Head ft/stg — 17-stg test | Head ft/stg — 9-stg test | BHP/stg | Eff % |
|---|---|---|---|---|
| 0 | 242.06 | 227.78 | ~38 | 0 |
| ~10,000 | 220.22 | 217.78 | ~41 | ~40 |
| ~19,000 | 197.03 | 200.94 | ~45 | ~62 |
| ~29,000 | 177.06 | 182.37 | ~52 | ~73 |
| ~36,000 | 162.76 | 168.88 | ~58 | ~77 |
| ~43,000 | 144.55 | 148.73 | ~61 | ~78 |
| ~50,000 | 122.73 | 121.39 | ~62 | ~73 |

Per-stage head × stage count (17 or 26) = pump head. dP (psi) = pump head × SG / 2.31. Discharge = intake + dP.

---

## 8. Maintenance notes (April 2026 LP failure — `Lessons Learned …docx`)

- **Coupling design flaw:** the PSC1358 Inconel pins bend, losing pump clearances and causing rubbing.
  At rated the pump produces **~18,800 lb of shaft thrust**. Summit does not stock this coupling.
- **⚠ P-0901 HP uses the same coupling/pin design with similar run hours — flagged as a likely repeat risk.**
- Thrust-chamber shim-bolt and new-TC issues (no dog-point grub-screw holes; bent shaft corrected to 0.0045" TIR);
  Hilcorp had no shims on hand. **Action items:** stock shims/couplings; add a PM to check glycol levels and
  rotate the shaft on spare pumps; investigate startup/process-upset hammering.
