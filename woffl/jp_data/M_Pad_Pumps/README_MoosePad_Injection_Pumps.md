# Moose Pad (Mod 42) Injection Pump Curves — Summary & How to Use

**Prepared:** 2026-06-16 · **Validated against Schlumberger data/set-up sheets & a 2026 SCADA snapshot**
**Source folder:** `Schlumberger pumps\` (REDA HPS data sheets, VSD curves, "Moosepad HPS set up" sheets, GA drawings, BOMs)

---

## 1. What the pumps are

Moose Pad runs **six Schlumberger REDA HPS pumps** in two banks of three. Both banks are
**VFD-driven**; all six share the same 1500 HP Toshiba motor.

| | **"A" pump — LP bank** | **"B" pump — HP bank** |
|---|---|---|
| Tags | **P-4220A / B / C** (3 parallel) | **P-4230A / B / C** (3 parallel) |
| Pump | REDA **N1400N-A** | REDA **M675-A** |
| **Stages** | **13** | **41** |
| Housing | 1000 series (10 in), mixed-flow | 862 series (8.62 in) |
| Shaft | higher-strength Monel | Inconel 718 |
| Motor | Toshiba 1500 HP, 4160 V (3569 RPM @ 60 Hz) | same |
| **Motor amp limit** | FLA 178 A · design 183 A · **trip 195.8 A** | same |
| Nominal / range (Hz) | 61 / 55–62 | 60 / 51–61 |
| Intake / discharge | 8" CL600 / 8" CL1500 RTJ | 8" CL1500 / **6" CL2500** RTJ |
| Seal | 48V, max 850 psig | 8610, max 2900 psig |
| Design point (60 Hz, SG 1.05) | 50,000 bpd, 400→1,654 psig (1,254 boost), 1,351 HP | 32,000 bpd, 1,500→3,502 psig (2,002 boost), 1,397 HP |
| Shut-in pressure | 1,988 psig | 4,471 psig (set Pd trip < ~4,000) |

---

## 2. Architecture — parallel **and** series (different from S-Pad and I-Pad)

```
  produced water ──► [ P-4220 A/B/C ]  3 LP pumps in PARALLEL
                          │  boost to ~1,400 psig common header
                          ├───────────────► Disposal header (~1,380 psig)
                          ├───────────────► Kuparuk injection header (~1,391 psig)
                          └──► [ P-4230 A/B/C ]  3 HP pumps in PARALLEL
                                   suction ~1,400 psig ─► boost to ~3,500 psig
                                   └───────► Power Fluid header (~3,489 psig)
```

> **The key facts:**
> 1. **Within each bank the 3 pumps run in PARALLEL** (same discharge header, flows add) — like the S-Pad boosters.
> 2. **The two banks run in SERIES** (LP → HP) — the LP discharge (~1,400 psig) is the HP suction — like the I-Pad LP→HP pair.
>    So Moose Pad is a **hybrid: parallel within a bank, series between banks.**
> 3. **Both pump types use a different REDA pump** (N1400N 13-stage vs M675 41-stage) — *not* one shared stage. Two curves.
> 4. The HP pump alone tops out near ~2,800 psi boost; it reaches the ~3,500 psig power-fluid pressure **only because it is
>    fed ~1,400 psig by the LP bank.**

Compare the three pads:

| Pad | Pumps | Arrangement |
|---|---|---|
| **S-Pad** | 3 identical boosters | parallel only |
| **I-Pad** | 1 LP + 1 HP | series only (LP→HP) |
| **Moose Pad** | 3 LP + 3 HP | **parallel within each bank, series between banks** |

---

## 3. Validation & live SCADA (6/16/2026 ~10:09)

**Curve vs manufacturer** — the polynomial model reproduces Schlumberger's own data-sheet / set-up-sheet
points to <0.5%: A shut-in 1,588 vs 1,587.8 psi · A BEP boost 1,250 vs 1,255 · A power 1,352 vs 1,350.7 HP ·
B shut-in 2,970 vs 2,970.7 · B BEP boost 2,225 vs 2,228 · B power 1,397 vs 1,397 HP.

**Live operating points (all six pumps):**

| Pump | Hz | Flow (bpd) | Suction | Discharge | dP (psid) | Amps | First-in-alarm |
|---|---|---|---|---|---|---|---|
| P-4220A (LP) | 55.0 | 18,456 | 289 | 1,399 | 1,109 | 98.4 | PUMP HIHI DIFF PRESS CURVE S/D |
| P-4220B (LP) | 54.7 | 19,509 | 287 | 1,400 | 1,113 | 92.2 | DISCHARGE FLOW LOLO |
| P-4220C (LP) | 53.1 | 20,643 | 304 | 1,398 | 1,092 | 99.3 | DISCHARGE FLOW LOLO |
| P-4230A (HP) | 57.9 | 16,512 | 1,392 | 3,495 | 2,102 | 125.0 | PSD LEVEL 3 |
| P-4230B (HP) | 57.6 | 18,047 | 1,392 | 3,498 | 2,106 | 123.3 | PSD LEVEL 3 |
| P-4230C (HP) | 55.9 | 17,588 | 1,391 | 3,499 | 2,108 | 119.2 | PSD LEVEL 3 |

- **Series staging confirmed:** LP discharge header (1,398–1,400 psig) **is** the HP suction (1,391–1,392 psig). PIC-4221 holds the LP header at 1,400; PIC-4231 holds the HP/power-fluid header at 3,500.
- **Amps match the as-new curve:** motor current tracks catalog BHP within ~1–7 A (`amps ≈ 0.13 × BHP`). Lots of headroom — 92–125 A against **178 A FLA / 195.8 A trip** (~52–70% of FLA). Amps are *not* the limiting factor here.
- **Operating far left of BEP:** both banks run at 53–58 Hz and well below their 60 Hz BEPs (LP ~18–21k vs 49,791; HP ~16–18k vs 27,708). That low-flow/high-diff corner is why **P-4220A is at its HIHI diff-pressure curve shutdown** and B/C show discharge-flow LOLO.

### ⚠ Finding: pumps are ~10% below their head curves (possible wear)

At the **displayed flow and speed**, every pump makes about **10% less differential than its as-new curve**
(field ÷ curve dP = mean **0.895**, range 0.86–0.94) — **while drawing on-curve power** (amps match catalog BHP):

| Pump | Field dP | As-new-curve dP | Field/curve |
|---|---|---|---|
| P-4220A | 1,109 | 1,276 | 0.87 |
| P-4220B | 1,113 | 1,257 | 0.89 |
| P-4220C | 1,092 | 1,177 | 0.93 |
| P-4230A | 2,102 | 2,449 | 0.86 |
| P-4230B | 2,106 | 2,377 | 0.89 |
| P-4230C | 2,108 | 2,237 | 0.94 |

(Ratios above are at the **field SG ~1.03**; at SG 1.02–1.04 the mean gap is ~8–10%, so the deficit holds across the range.)

**Lower head at on-curve power = reduced hydraulic efficiency, the classic signature of stage WEAR.** The operator
confirms Moose Pad **handles heavy solids**, so **erosive stage wear is the likely cause** — and these are 2017-vintage
units with documented thrust-chamber/alignment history (§7). It is **not** an SG effect (SG scales head and power
together) and **not** a simple flow-meter offset (that would move head and power in opposite directions on the curve).
**Before finalizing the wear number, verify:** the exact produced-water SG, discharge/suction transmitter calibration,
and (ideally) a witnessed single-pump performance check. The CSV/JSON curves are the **as-new reference**; multiply
head/boost by **`FIELD_HEAD_FACTOR ≈ 0.91`** for field-representative performance at today's condition.

---

## 4. The files

| File | What it is |
|---|---|
| `MoosePad_A_LP_4220_curve_for_repo.csv` | LP "A" pump (N1400N, 13 stg): flow → head, boost, discharge, BHP, est-amps at 60 Hz. |
| `MoosePad_B_HP_4230_curve_for_repo.csv` | HP "B" pump (M675, 41 stg): same columns. |
| `MoosePad_pump_curve_meta.json` | Machine-readable: both curve polynomials, configs, motor-amp limits, design points, SCADA snapshot, provenance. |
| `build_moosepad_pump_curves.py` | Generator (standard-library Python). Edit `SG` / grid and re-run. |
| `README_MoosePad_Injection_Pumps.md` | This file. |

---

## 5. How to estimate pressure & amps

Each pump is a single series train of stages (all stages see the same flow). For a **bank of 3 in parallel**, the
pumps share a discharge pressure and **flows add** → each pump carries ⅓ of the bank total.

```
head_ft   = c3*Q^3 + c2*Q^2 + c1*Q + c0          (Q = BPD per pump at 60 Hz; coeffs in JSON)
boost_psi = head_ft * SG / 2.31                   (field SG ~1.02-1.04; CSV uses 1.03)
discharge = suction_psi + boost_psi               (LP suction ~ feed; HP suction = LP header ~1,400)
BHP       = c3*Q^3 + c2*Q^2 + c1*Q + c0           (power polynomial)
amps      ~ k * BHP                               (k = 0.135 A-pump, 0.131 B-pump)  -> compare to 195.8 A trip
```

**On the VFD at other speeds:** `Q ∝ Hz/60`, `head ∝ (Hz/60)²`, `BHP & amps ∝ (Hz/60)³`.
Both banks currently run turned-down (LP 53–55 Hz, HP 56–58 Hz), well left of their 60 Hz BEP, so amps run low (~92–125 A).

### A (LP) pump — per pump, 60 Hz, suction 400 psig

| Flow (bpd) | Boost (psi) | Discharge (psig) | BHP | Amps |
|---|---|---|---|---|
| 20,000 | 1,520 | 1,920 | 951 | 129 |
| 30,000 | 1,460 | 1,860 | 1,076 | 146 |
| 40,000 | 1,372 | 1,772 | 1,216 | 165 |
| 49,791 *(BEP)* | 1,250 | 1,650 | 1,352 | 183 |
| 60,000 | 1,076 | 1,476 | 1,474 | 200 ⚠ over 195.8 |
| 65,262 *(max)* | 964 | 1,364 | 1,523 | 206 ⚠ |

### B (HP) pump — per pump, 60 Hz, suction 1,500 psig

| Flow (bpd) | Boost (psi) | Discharge (psig) | BHP | Amps |
|---|---|---|---|---|
| 10,000 | 2,802 | 4,302 | 889 | 117 |
| 20,000 | 2,540 | 4,040 | 1,151 | 151 |
| 27,708 *(BEP)* | 2,225 | 3,725 | 1,335 | 175 |
| 31,750 *(design)* | 2,006 | 3,506 | 1,397 | 183 |
| 34,798 *(max)* | 1,813 | 3,313 | 1,419 | 186 |

> Note: at 60 Hz the **A/LP pump becomes amp-limited above ~57,000 bpd** (195.8 A trip ≈ 1,445 BHP). The B/HP pump has
> amp headroom across its whole range at 60 Hz. In current turned-down operation neither bank is near the amp trip.

---

## 6. Assumptions & caveats

- **Parallel within bank, series between banks** (see §2). For a bank, divide the header flow by the number of
  running pumps to get per-pump flow before reading the curve.
- **SG:** Schlumberger design used 1.05; **field produced water is ~1.02–1.04** (operator). CSV `boost_psi` uses 1.03.
  Head in feet is SG-independent — convert with your own SG for the optimizer.
- **Field runs turned-down and ~10% below curve.** SCADA (all 6 pumps) shows ~16,000–21,000 bpd per pump at 53–58 Hz —
  well below the 60 Hz BEPs (50,000 / 27,700) — and ~10% under the as-new head curve at on-curve power (see §3 finding).
  The CSV/JSON are the **as-new reference**; multiply boost by ~0.89 for today's field-representative values, or better,
  re-derive the factor once SG and transmitter calibration are confirmed.
- **Amps model** `amps ≈ k·BHP` (k ≈ 0.13) is calibrated to the design point and confirmed by live data (within ~1–7 A);
  for trend/comparison, not protection. Amps are well below the 178 A FLA / 195.8 A trip in current operation.
- **HP discharge trip.** B-pump shut-in (4,471 psig) exceeds the 4,400 psig rating — the discharge trip should be set
  to keep Pd below ~4,000 psig (per the data sheet note).
- **Data basis:** `M04 A/B PUMP Data Sheet`, `A/B-Pumps Moosepad HPS set up sheet`, `A/B PUMP Curves` PDFs, GA drawings
  102865153 (A) / 102866716 (B), and a 2026 Moose Pad SCADA snapshot.

---

## 6.5 Notes for the optimizer / building system curves

When porting these into a system-curve / optimizer model:

1. **Ingest `head_ft(Q)` (per pump, 60 Hz)** from the CSV — it is **SG-independent** and is the true pump curve.
   The polynomials are in the JSON (`head_ft_poly_60hz`).
2. **Affinity-scale to actual speed:** `Q ∝ Hz/60`, `head ∝ (Hz/60)²`, `BHP & amps ∝ (Hz/60)³`.
3. **Apply the wear derate:** multiply head by **`FIELD_HEAD_FACTOR ≈ 0.91`** (range 0.87–0.96) for the *current* worn
   curve. The CSV columns are **as-new**; the build script has a `FIELD_HEAD_FACTOR` knob if you want it pre-applied.
4. **Convert to psi with field SG (1.02–1.04):** `boost_psi = head_ft × SG / 2.31`.
5. **Bank topology:** within a bank the 3 pumps are **parallel** (same dP, flows add); the **LP bank discharge is the
   HP bank suction** (series). So the station head is LP-boost + HP-boost in series, with each bank's flow split 3 ways.
6. **System curve** = static head + friction(Q²). Its intersection with the **derated** pump curve sets the operating
   point. Today's live points (§3) are good anchors to fit the friction term — and they already include the ~9% wear,
   so if you fit the system curve to live data **and** use the as-new pump curve, you'd double-count; use one or the other.
7. **Re-tune the factor** once you confirm SG and transmitter calibration — and re-pull it per pump if you want
   per-unit wear (4230A/4220A are the most degraded at ~0.87; 4220C/4230C the least at ~0.95).

## 7. Maintenance notes (from the `Reliability` and skid folders)

- **P-4230C thrust-chamber high vibration** investigated Apr 2023 (`230422 4230C Thrust Chamber High Vibe\`).
- **P-4230B skid realignment** performed Apr 2023 (`4230B\230401 4320B Skid Realignment\`) — multiple skid moves / motor shimming.
- Coupling is **GA-161615 / 100152071** (see drawing). Thrust chambers are 3-bearing, 18,000 lbf, common to A and B.
- These REDA units share the same monitored failure modes as other Milne HPS pumps (thrust-chamber bearings, seal,
  coupling) — track thrust-chamber vibration and oil temperature against the set-up-sheet limits (alarm 0.25 in/s, trip 0.35 in/s; oil 200/210 °F).
