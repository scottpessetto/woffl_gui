# S-Pad Booster Pump Curves — Summary & How to Use

**Prepared:** 2026-06-15 · **Validated against plant SCADA**
**Source folder:** `S-Pad Booster Pumps\` (Weatherford/Borets test reports, HPS data sheets)

---

## 1. What the pumps are

All three S-Pad booster pumps (**P-800 B-1, B-2, B-3**) are identical:

| Item | Value |
|---|---|
| Manufacturer | Weatherford / Borets |
| Pump model | **ESPi675TJ‑12000** ("675‑12000", 6.75 in series) |
| Type | Multistage vertical centrifugal (ESP-style) booster |
| **Configuration** | **TANDEM — two sections on one shaft: 41‑stage + 38‑stage = 79 stages in series** |
| Operating speed | **3500 RPM** (motor nameplate 3570 RPM) |
| Motor | Reliance 1000 HP, 3570 RPM, 4160 V, frame E5810 |
| Recommended flow range | **7,650 – 18,360 BPD per pump** (matches SCADA thrust boundaries 7,800 / 19,000) |
| Best efficiency point | ≈ 12,000 BPD per pump, ~80% |
| Typical suction (intake) | ~220 psi (SCADA) |

> **The key fact: each unit is a 79‑stage tandem (41 + 38).** The HPS data sheet lists
> *both* a 41‑stage and a 38‑stage pump per unit, and the GA drawing shows them as
> "PUMP 1" + "PUMP 2" stacked on one shaft. They are in **series**, so the heads add:
> 79 stages total. (An earlier draft of these files treated 38/41 as alternative builds —
> that was wrong and made the pressure come out ~½. Corrected here.)

---

## 2. Validation against SCADA (2026-06-15)

| Quantity | This model (79‑stg) | SCADA / field |
|---|---|---|
| Differential at ~10,200 BPD/pump | 3,235 psi | 3,461 − 220 = **3,242 psi** ✓ |
| Discharge, 3 pumps @ ~30,700 BPD total | 3,455 psi | **3,460–3,463 psi** ✓ |
| Discharge, 3 pumps @ 30,000 BPD total | 3,474 psi | ~**3,475 psi** (your reading) ✓ |
| Shut‑off head | ~3,670–3,800 psi | red curve ~**3,775 psi** ✓ |
| BEP flow | ~12,000 BPD | yellow peak ~**12,000** ✓ |
| Thrust boundaries | 7,650 / 18,360 | purple **7,800** / blue **19,000** ✓ |

The fitted head curve overlays the SCADA red "HEAD CURVE" across the whole range.

---

## 3. The files

| File | What it is |
|---|---|
| `S-Pad_3-Pump_Parallel_Lookup.csv` | **All 3 online** → total station flow → dP and discharge pressure. **This is the file for your question.** |
| `S-Pad_Booster_Pump_Curve_SINGLE.csv` | One unit (79‑stage): flow → head, dP, discharge, BHP, efficiency. |
| `pump_curve_for_repo.csv` | Clean, comment‑free table (250‑BPD steps, both flow axes) for another repo to load/interpolate at runtime. |
| `pump_curve_meta.json` | Machine‑readable parameters (stages, RPM, SG, suction) + the fitted polynomial + provenance, to accompany the repo CSV. |
| `build_pump_curves.py` | Raw test data + curve fit. Edit `N_STAGES`, `SG`, `SUCTION_PSI`, speed, etc. and re‑run; regenerates all of the above. |

---

## 4. How to estimate pressure with all 3 pumps online

The three units run in **parallel** into a common discharge header. For parallel pumps:
**they all make the same differential pressure, and flows add** → each pump carries **⅓**
of the total station flow.

1. Open `S-Pad_3-Pump_Parallel_Lookup.csv`.
2. Find your **total station flow** (left column).
3. Read **`dP_psi`** (boost) and **`discharge_psi`** (= 220 psi suction + dP).

**Quick reference (all 3 online, 79‑stage tandem, SG 1.0, 220 psi suction):**

| Total flow (BPD) | Per pump | Boost dP (psi) | Discharge (psi) |
|---|---|---|---|
| 24,000 | 8,000 | 3,400 | 3,620 |
| 30,000 | 10,000 | 3,254 | 3,474 |
| 30,700 *(current)* | 10,233 | 3,235 | 3,455 |
| 36,000 | 12,000 | 3,065 | 3,285 |
| 45,000 | 15,000 | 2,683 | 2,903 |
| 54,000 | 18,000 | 2,167 | 2,387 |

**Direct formula (SG≈1.0):**

```
Q_pp = total_flow / 3                       (BPD per pump)
head_per_stage = 107.221
               - 3.6333e-4  * Q_pp
               - 4.5572e-8  * Q_pp^2
               - 3.8676e-12 * Q_pp^3        (ft)
dP_psi    = head_per_stage * 79 * SG / 2.31
discharge = suction_psi + dP_psi            (suction ~220 psi)
```

---

## 5. Assumptions & caveats

- **Parallel operation** (shared header, flows add). Confirmed by SCADA: 3 units at ~equal
  flow and equal discharge.
- **Pump capability vs. operating point.** The curve gives what the unit makes at a given
  flow. The actual flow is set by where this curve crosses the **system/injection curve**.
- **SG = 1.00** in the tables; S‑Pad produced water ≈ 1.01 → multiply dP by ~1.01 (~+1%).
  This 1% closes most of the small gap to the SCADA discharge.
- **Speed 3500 RPM.** On VFD at other speeds: flow ∝ (RPM/3500), head ∝ (RPM/3500)².
- **Suction 220 psi** taken from SCADA intake; use the live intake for best accuracy.
- **Data basis:** fit of 2012 Borets section tests (WO 13579 = 38‑stg, WO 13580 = 41‑stg,
  both at 3500 RPM), per stage, then ×79. Cross‑checked vs 2004 Weatherford shop tests
  (2500 RPM, affinity‑scaled) — agree ~1% in the operating range.

---

## 6. Underlying test data (per stage, 3500 RPM, SG 1.0)

| Flow/pump (BPD) | Head ft/stg — WO13579 (38) | Head ft/stg — WO13580 (41) | Eff % |
|---|---|---|---|
| 0 | 107.8 | 105.3 | 0 |
| ~4,000 | 108.2 | 106.6 | ~44 |
| ~7,400 | 99.0 | 96.6 | ~66 |
| ~12,000 | 90.8 | 89.2 | ~80 (BEP) |
| ~14,900 | 79.6 | 79.1 | ~79 |
| ~18,000 | 63.9 | 63.5 | ~71 |
| ~20,900 | 43.5 | 44.3 | ~55 |

Unit head = (ft/stage) × **79**. Differential psi = unit head × SG / 2.31. Discharge = suction + dP.
