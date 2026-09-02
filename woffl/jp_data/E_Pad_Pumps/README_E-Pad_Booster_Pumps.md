# E-Pad Booster Pump Candidates - Summary & How to Use

**Prepared:** 2026-08-27
**Source:** `Summit E Pad Booster (1).xlsx` and `Summit E Pad Booster as SN35000.xlsx` - the stage
tables on sheet `Current` plus the Summit ESP catalog performance page embedded in each workbook
(`SM25000 Pump`, halliburton.com p. 250; `SN35000 Pump`, p. 256).
**Not validated against live E-Pad SCADA.** No operating point, and no motor nameplate, came with
the workbooks. Everything below is catalog physics plus the workbooks' own affinity sheet.

---

## 1. What the candidates are

Two Summit ESP mixed-flow builds for the E-Pad power-fluid booster, both VFD driven.

| Item | **SM25000 - 26 stg** (in well) | **SN35000 - 18 stg** (alternative) |
|---|---|---|
| Catalog stage | SM25000, mixed flow, XRC, CCW | SN35000, mixed flow, XRC, CCW |
| Series / housing | 875 series, 8.75 in (10.75 in min casing) | 950 series, 9.5 in (11.75 in min casing) |
| **Stages** | **26** | **18** |
| **XRC operating range at 60 Hz** | **8,100 - 32,400 BPD** | **12,400 - 49,500 BPD** |
| BEP at 60 Hz | 27,000 BPD | 41,250 BPD |
| Shut-off head per stage at 60 Hz | 154 ft (peaks 156 ft at 5,000 BPD) | 235 ft |
| Standard shaft limit | 1,094 HP (high-strength 1,795) | 1,440 HP (high-strength 2,370) |
| Catalog housing pressure limit | 4,400 psi | 2,800 psi - see §5 |
| Motor | not stated in the sheet | not stated in the sheet |
| Workbook scenario | 55 Hz, SG 1.02, 2,800 psi suction | 55 Hz, SG 1.00, 2,600 psi suction |

Both are **mixed-flow stages, so head RISES off shut-off** before falling (SM25000: 154 ft at 0,
156 ft at 5,000). That is the real curve shape on the catalog page, not a digitization artifact -
do not "fix" it with a monotone assertion.

The same **SN35000 stage runs on I-Pad today** (17-stage HP + 26-stage tandem LP, `I_Pad_Pumps/`),
live-validated to ~1 pct against SCADA. That is the closest thing to field calibration this stage
has, and it is where the default amps/BHP below comes from.

---

## 2. The model

Exactly the affinity sheet the two workbooks implement, **inverted**: the workbooks fix a speed and
read off dP, while the screen fixes the required dP and solves the SPEED, because that is the
decision a VFD actually makes.

```
Q60   = Q * 60 / Hz                                  index the 60 Hz stage table
head  = condition * n_stages * head_stg(Q60) * (Hz/60)^2
dP    = head * SG / 2.31
disc  = suction + dP
BHP   = SG * n_stages * bhp_stg(Q60) * (Hz/60)^3     the workbooks' "HP load"
amps  = amps_per_bhp * BHP
ROR   = [xrc_lo, xrc_hi] * Hz/60                     the range moves with speed
```

`head_stg` / `bhp_stg` are **linearly interpolated** between the digitized rows. The rows *are* the
digitization; a 5th-order fit through 10-15 of them would add wiggle the catalog page does not have.
Efficiency uses the repo's `PadPlant.hydraulic_efficiency` (`Q_gpm * H * SG / (3960 * BHP)`), which
is the workbooks' formula with the exact constant instead of their rounded 136,000 - a 0.17 pct
difference, always in the same direction.

Bit-identical to both workbooks at their own scenario cells (pinned in
`tests/test_e_pad_booster.py`): SM25000 26 stg / 55 Hz / SG 1.02 reads 1,485.611 psid at shut-off
and 721.078 BHP at 41,000 BPD-at-60-Hz; SN35000 18 stg / 55 Hz / SG 1.00 reads 1,538.690 psid and
930.314 BHP.

---

## 3. The answer at the 3,400 psig header

E-Pad's power-fluid header runs at **3,400 psig** (`server/services/wells.py:_PAD_PF_DEFAULTS`).
From 2,800 psi suction that is a **600 psid** duty. At SG 1.02, as-new, 60 Hz cap, no amp cap:

| | **SM25000 - 26 stg** | **SN35000 - 18 stg** |
|---|---|---|
| Flow window inside ROR | **4,726 - 22,867 BPD** | **7,260 - 37,293 BPD** |
| Deliverable rate at 3,400 psig | **22,867 BPD** | **37,293 BPD** |
| Speed there | 42.3 Hz | 45.2 Hz |
| Shaft power there | 313 BHP | 525 BHP |
| Amps there (k = 0.1435) | 45 A | 75 A |
| Efficiency there | 74.6 pct | 72.5 pct |
| What caps it | recommended range (high) | recommended range (high) |

Neither build is anywhere near pressure-limited at 600 psid - both are **range-limited**, running in
the low 40s Hz. The alternative moves **63 pct more water** at the same header for 68 pct more
shaft power.

### SM25000 - 26 stg, holding 600 psid

| Flow (BPD) | Hz | BHP | Amps | Eff pct | ROR at that Hz | In ROR |
|---|---|---|---|---|---|---|
| 5,000 | 35.0 | 110 | 16 | 46.4 | 4,732 - 18,927 | yes |
| 10,000 | 36.4 | 150 | 22 | 67.9 | 4,910 - 19,642 | yes |
| 15,000 | 38.0 | 199 | 29 | 76.8 | 5,124 - 20,498 | yes |
| 20,000 | 40.3 | 263 | 38 | 77.5 | 5,445 - 21,781 | yes |
| **22,867** | **42.3** | **313** | **45** | **74.6** | 5,717 - 22,867 | **at the edge** |
| 25,000 | 43.9 | 355 | 51 | 71.9 | 5,929 - 23,715 | NO |
| 30,000 | 48.2 | 479 | 69 | 63.9 | 6,509 - 26,036 | NO |

### SN35000 - 18 stg, holding 600 psid (SG 1.02)

| Flow (BPD) | Hz | BHP | Amps | Eff pct | ROR at that Hz | In ROR |
|---|---|---|---|---|---|---|
| 5,000 | 34.6 | 154 | 22 | 33.2 | 7,145 - 28,522 | NO |
| 10,000 | 35.9 | 188 | 27 | 54.2 | 7,413 - 29,594 | yes |
| 15,000 | 37.3 | 232 | 33 | 66.0 | 7,716 - 30,801 | yes |
| 20,000 | 38.7 | 283 | 41 | 72.2 | 7,988 - 31,888 | yes |
| 25,000 | 40.1 | 336 | 48 | 75.9 | 8,284 - 33,068 | yes |
| 30,000 | 41.8 | 401 | 58 | 76.3 | 8,628 - 34,444 | yes |
| **37,293** | **45.2** | **525** | **75** | **72.5** | 9,342 - 37,293 | **at the edge** |

The ROR high edge lands at exactly **120 pct of BEP** on both stages - the catalog XRC top *is*
1.2x BEP for each (32,400 / 27,000 and 49,500 / 41,250). That is the arithmetic check on the
digitized ranges.

### The same question has three honest answers - do not conflate them

"What flow do I get?" depends on how you intend to run the drive.

| Policy | SM25000 26 stg at 600 psid | What it costs |
|---|---|---|
| **Slow to match** - hold 600 psid exactly | **22,867 BPD at 42.3 Hz** | 313 BHP, 45 A, eff 74.6 pct. Nothing wasted. Least water. |
| **Flat out + choke** at a 55 Hz cap - pass the range ceiling, burn the surplus | **29,700 BPD at 55 Hz** (+6,833) | 686 BHP, 98 A. Pump makes 1,012 psid, so **412 psi choked off = 208 hydraulic HP burned**. |
| **Pin 55 Hz and force 600 psid** | 37,248 BPD | **125 pct of the range ceiling**, eff collapses to 52.7 pct, 721 BHP. Off curve - not an operating point. |

The third row is why the screen used to look like it was "capping" for no
reason. At a fixed dP the crossing flow rises with speed FASTER than the
recommended range does, so past the duty speed every rung is over the range
top. The duty row is simply the last rung still inside it - exactly 100 pct of
its own range ceiling.

The sweet spot for a 600 psid duty is around **40 Hz**: 19,336 BPD at 77.7 pct
efficiency for 254 BHP, 10 pct inside the range. Both builds are ~1,500-1,900
psid machines being asked for 600, which is the real finding here - **the
installed 26-stage build is oversized for this duty** and the 18-stage
SN35000 makes even more head, not less.

---

## 3b. As a pad plant (the optimizer's view)

`woffl/gui/e_pad_plant.EPadPlant` is the `PadPlant` face on this data, so
E-Pad runs through the pad optimizer like S/I/M. Its capability frontier is
**unimodal in flow**, unlike the I and M frontiers which only fall:

| Total PF (BPD) | Max deliverable header (psi) | What binds |
|---|---|---|
| below ~2,700 | none | drive cannot slow far enough to stay in range |
| 4,000 | 3,230 | range FLOOR - drive held at 29.6 Hz |
| **8,100** | **4,562** | the knee: range floor stops binding at 60 Hz |
| 16,000 | 4,439 | 60 Hz, falling head |
| 32,400 | 4,005 | range CEILING at 60 Hz - throughput limit |
| above 32,400 | none | no speed keeps the flow on the curve |

Two consequences, and they are the pad's actual answer:

- **The booster is not the pressure constraint at 3,400 psi.** Its frontier
  sits above 4,000 psi across the whole range, so the PF budget is a flat
  32,400 BPD over the entire sweep band and the **operational header cap**
  (3,500 psi, adopted from I-Pad pending an E-Pad piping number) is what
  limits the sweep. Raise that cap and the optimizer gets more PF pressure
  for free - which is exactly why it is a run knob.
- **The booster's real limit is throughput.** 32,400 BPD installed against
  **49,500 BPD** on the SN35000. That +17,100 BPD is what a changeout buys if
  E-Pad's PF demand grows past the installed ceiling - e.g. adding wells.

Every inverse in the plant scans before it bisects. A monotone bisection from
zero flow (the shape the I/M inverses assume, because their frontiers are
monotone) tests the collapsed low-flow branch first, fails, and reports a
budget of 0.0 at every header. `tests/test_e_pad_plant.py` pins that.

---

## 4. Amps

`amps = k * BHP` - the convention the I-Pad and M-Pad plant models already use.

**No E-Pad motor data came with the curve sheets.** So:

- `k` defaults to **0.1435 A/BHP**, the live-calibrated value for the I-Pad 26-stage SN35000 HPS
  unit on a **4160 V** motor (SCADA 2026-06-16). It is a **transferred estimate**. Scale it by
  `4160 / V` for another voltage, and replace it on screen the moment the E-Pad nameplate is known.
- The **amp limit defaults to unset**: the screen reports amps always and enforces a cap only when
  the engineer types the motor limit in. Nothing is invented.

Amps are for **trend, not protection** - power factor and motor efficiency drift away from the
calibration point, and `k * BHP` makes amps scale as `(Hz/60)^3` where constant-V/Hz theory says
`(Hz/60)^2`. Below ~46 Hz, where both candidates sit at the 600 psid duty, that is the model's
largest single uncertainty. It is kept because it is the repo's one validated amps convention;
introducing a second one for one pad would be worse.

---

## 5. What is deliberately NOT enforced

- **Catalog housing pressure limit.** The SN35000 page says 2,800 psi, below the 3,400 psig header.
  That is a **downhole ESP housing** number, and the identical stage runs at **3,408 psig discharge
  on I-Pad today** in an HPS barrel with its own pressure-rated discharge head. Enforcing the
  catalog figure here would raise a false alarm on the better candidate. It is displayed, not gated.
  **Chase the actual HPS barrel / discharge-head rating before committing a build.**
- **Shaft HP limit.** Carried for reference. Neither build approaches it - the SM25000 26 stg peaks
  near 721 BHP at 55 Hz against a 1,094 HP standard shaft.
- **NPSH / suction requirements.** Not published on the supplied catalog pages.
- **Speed floor.** The model's 20 Hz floor is numerical (below it the 60-Hz-equivalent flow walks
  off the digitized table), not an operating limit. The recommended range binds far above it.

---

## 6. The files

| File | What it is |
|---|---|
| `E-Pad_booster_pump_meta.json` | Both candidates: digitized 60 Hz stage tables, XRC ranges, BEP, catalog limits, screen defaults, provenance. |
| `README_E-Pad_Booster_Pumps.md` | This file. |

Physics: `woffl/gui/e_pad_booster.py` (the candidate screen) and
`woffl/gui/e_pad_plant.py` (the `PadPlant` face the optimizer runs).

APIs: `POST /api/optimize/e-pad-booster` (`server/services/e_pad_curves.py`) for
the candidate comparison; `GET /api/optimize/pump-curve?pad=E` + the four
booster knobs (`server/services/pad_curves.py`) for the curve sheet;
`POST /api/optimize/run` with `pad=E` and the `e_pad_*` knobs for the pad run.

Screens: the **E-Pad booster** tab (candidate comparison,
`web/src/pages/optimize/EPadBoosterPanel.tsx`) and the **E-Pad** run tab
(readiness board + optimizer, `RunPanel` + `PadCharts`), both on the
Optimization page.

Tests: `tests/test_e_pad_booster.py` (workbook pins, the three views) and
`tests/test_e_pad_plant.py` (the unimodal frontier, the run knobs, the wiring).
