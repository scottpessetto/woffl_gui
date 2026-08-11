# Model trust and event calibration - session reference (2026-08-10)

Everything built in the 2026-08-10 session: what it does, where it lives, which
knobs tune it, and how to re-run the validation harnesses. Written for the
engineer who wants to play with it and tune it later.

---

## 1. Why this exists - the findings

1. **The cavitation floor was fiction on most of M-Pad.** The solver
   (`solopump.jetpump_solver`) early-returns a sonic point whose suction
   (`psu_min` from `jetflow.psu_minimize`) is independent of PF pressure.
   8 of 15 comparable M-Pad wells had 12-month measured BHPs 50-145 psi BELOW
   their modeled floors. MPM-64 ran at 343 psi in Nov 2025 against a modeled
   floor of 430 ("even with infinite power fluid you could not get below this
   psu" - falsified by a gauge).
2. **Single-point BHP calibration manufactured it.** Auto-match BHP fits
   ken/kth/kdi to ONE gauge value. On a sonic-pinned well kth/kdi cannot move
   suction at all, so Nelder-Mead railed ken at 0.40 (dragging the floor up to
   calibration-day BHP) and parked kth/kdi at their 0.05 floors (deepening the
   fake "free choke" region). Every 14B on the pad had this signature.
3. **The PF-rate hydraulics are excellent.** Across a measured 3,429 -> 3,030
   psi cut, model dPF -4.9% vs measured -4.9%. Trust the nozzle side.
4. **The field response is measurable from data we already pull.**
   `vw_pressure_daily` (daily PF pressure + BHP since 2024-09) yields per-well
   response slopes: beta = -dBHP/dPpf ~ 0.08-0.19 on responsive wells,
   ~0.00-0.02 on genuinely pinned ones. The miner reproduced a hand-read PI
   slope (0.087) at 0.079-0.085 from 1,652 event pairs.

Design consequence: keep the physics engine untouched, discipline what feeds
it (calibration), correct what is claimed from it where measurement disagrees
(evidence layer), and show provenance everywhere.

---

## 2. Map of the pieces

| Piece | Backend | Frontend | Purpose |
|---|---|---|---|
| Evidence layer | `server/services/evidence.py`, `woffl/gui/pad_optimize.py` | choke plan tables/badges | measured floors + response correct the choke plan |
| Decision ladder | `pad_optimize.run_choke_optimization` meta `ladder` | `RunPanel.tsx` HeaderDropLadder | "header sags X -> best action -> gain" |
| Match scorecard | `server/services/match_health.py` | `MatchHealthPanel.tsx` (Optimization page) | per-well model-vs-field trust board |
| mach_crit + fnz | `woffl/flow/jetflow.py`, `solopump`, `batchpump`, `network_optimizer`, `schemas.SimParams` | sidebar params (applied via calibration) | slip choking closure + nozzle washout factor |
| Points builder | `server/services/calibration_points.py` | - | era-gated daily (Ppf, BHP, PF rate) fit set |
| Multipoint fitter | `woffl/gui/fric_calibration.py` `calibrate_multipoint` | - | fits ken/kth/kdi/fnz/mach_crit to the era history |
| Unified calibrate | `server/services/event_calibration.py` | `EventCalibration.tsx` in `CalibrateBar.tsx` | ONE button; event fit with single-point fallback |
| Single-point guard | `fric_calibration.calibrate_friction_coefs` | CalibrateBar messaging | refuses degenerate fits on sonic-pinned wells |
| Response diagnostic | `server/services/response_history.py` | `ResponseDiagnostic.tsx` (Solver page, bottom) | field dots vs model curve, the eyeball judge |
| Persistence | `ipr_anchor.FRICTION_PROPS` + prop_hist | IprControls save | event-cal params flow into every consumer |

---

## 3. Evidence layer (choke plan corrections)

**Mining** (`evidence.py`): one cached fleet query on `vw_pressure_daily`
(365 d). Per well:
- `floor` = min(p5 of flowing daily BHP, min test BHP). Flowing = valid PF
  pressure in [800, 5500], BHP > 50, BHP < saved res_pres.
- `psu_ref` = median BHP of the last 14 flowing days.
- `beta` = clamp(-median(dBHP/dPpf), 0, 0.5) over day pairs 3-30 days apart
  with |dPpf| >= 100 psi, never spanning a pump change. `beta_source` =
  "well" (>= 5 pairs) -> "pad" (median of siblings) -> "default" (0.09).

**Gates** (`pad_optimize._apply_suction_evidence`) - a well is corrected only
when the model claims sonic at the top ladder level AND either:
- floor gate: model floor exceeds measured floor by > 25 psi
  (`_EVIDENCE_VIOLATION_MIN_PSI`), or
- response gate: `beta_source == "well"` and beta >= 0.03
  (`_EVIDENCE_BETA_MIN`; field separation: insensitive wells measure <= 0.022,
  responsive >= 0.04). Pad/default betas never trigger it.

**Correction**: `psu_e(level) = psu_ref + beta * (P_full - level)`; oil scales
by the saved oil-basis Vogel ratio; PF rates stay MODEL (validated). Rows
carry `suction_basis`, `evidence_gate` ("floor"/"response"), floor/violation/
beta provenance; the landing table badges corrected wells "field".
Wells whose evidence CONFIRMS the model keep their free chokes - the gate
cuts both ways. `evidence=None` (Streamlit path) is byte-identical to
pre-evidence behavior.

---

## 4. Header-drop decision ladder

`meta["ladder"]` on choke runs: per ladder level below the winning header,
scale the bank frontier so all-run settles there, re-run the sweep + trim,
report best action / pad oil / gain vs doing nothing. Rendered as a collapsed
dropdown in RunPanel. Inherits evidence corrections automatically (with them,
"do nothing" at a 400 psi sag costs ~550 BOPD instead of ~115).

## 5. Match health scorecard

`POST /api/optimize/match-health {pad}` -> job -> per-well rows: fit source +
r2, model/test oil and PF ratios, model floor vs measured floor, beta with
provenance, ken/kth/kdi with railed flags, verdict chip:
`contradicted` > `railed-cal` > `weak-fit` > `ok`. Panel below RunPanel on
each pad tab. Use it the way the Prosper email did: fix the worst rows first.

## 6. New physics parameters

- **`mach_crit`** (default 1.0, bounds 1.0-2.5): slip closure on the choking
  criterion. Implemented by scaling the throat-entry kinetic differential
  energy by 1/mach_crit^2, so choking lands at homogeneous-computed
  Mach = mach_crit (effective sonic velocity = mach_crit x Wood speed).
  Floors DROP monotonically as it rises. A WELL/FLUID property - transfers to
  hypothetical pumps, which is what makes JPCO evaluation honest.
- **`nozzle_area_factor` (fnz)** (default 1.0, bounds 0.8-1.3): effective
  nozzle area ratio, `dnz_eff = dnz * sqrt(fnz)`. Identified by PF-rate
  residuals; fnz > 1 = washout, continuously (fleet fits ran +3% to +12%).
  A property of the INSTALLED pump: applies only to the installed
  (nozzle, throat) in batch sweeps (`WellConfig.installed_nozzle/throat`),
  JPCO candidates always solve at 1.0, resets on pump change.

Defaults reproduce pre-change behavior bit-identically. Library edits carry
`[LIBRARY change -> upstream PR to kwellis/woffl]` markers.

## 7. Event calibration (multi-point)

**Dataset** (`calibration_points.py`): current pump era only (jp_history
date-set tenure). Daily triplets: Ppf (`vw_pressure_daily` via
resolve_pf_pressure), BHP (`btmhole_prs`), PF rate
(`vw_power_fluid_volume.pwr_fld_net` > 500). Steady-state filter drops days
where BHP deviates > 60 psi from a centered 5-day rolling median. In-era
tests ride along at weight 3. Capped at 20 points, stratified across the Ppf
range, tests always kept. wc/fgor/IPR anchor per point from the nearest
in-era test (30 d) else the saved fit.
Refusals: < 10 usable points ("young pump era"), Ppf spread < 200 psi
("not identifiable"), > half the points unsolvable.

**Fitter** (`calibrate_multipoint`): Nelder-Mead over (ken, kth, kdi, fnz,
mach_crit), per-point Vogel anchored on that point's own test. Objective =
Huber(level BHP /50) + Huber(level PF /5%) + Huber(pair dBHP /25) over ALL
point pairs with dPpf >= 100 psi (total pair weight = total level weight).
Escapes: alt start at library seeds when poor; floor-escape reseed with
mach_crit at 2.5 when pinned or pair residual poor; polish restart from the
optimum. Returns RMS BHP / PF% / dBHP, railed list, implied_beta, per-point
rows, message.

**Unified button** (Solver page CalibrateBar): "Calibrate to field data" ->
`POST /api/optimize/event-calibration {well}` (job, poll via the optimize
run status route, kind "event_cal"). Era has data -> event fit, summary-first
result ("Matched 20 days of this pump's history ... nozzle ~5% washed out,
model tracks measured BHP within 18 psi") + response check vs mined beta
(green when within 0.03, amber "treat suction sensitivity as evidence-layer"
otherwise). Young era -> SERVER falls back to the single-point latest-test
BHP match and says so; Apply then writes ken/kth/kdi only. The standalone
Auto-match button is gone.

**Persist / inherit**: Save as well default pushes
`jpfric_nozzle_area`/`jp_mach_crit` (prop_xref rows added 2026-08-10) with
the same skip-default conventions as ken; hydration restores them into the
solver sidebar and `WellConfig.fnz_well/mach_crit_well`, so the pad
optimizer, choke plan, and scorecard all consume event-calibrated wells.

## 8. Response diagnostic (the eyeball judge)

Solver page, bottom, collapsed "Suction response diagnostic (advanced)".
Daily (Ppf, BHP) dots (current era blue, prior pumps faded, buildup days
toggleable), measured-floor line, and "Overlay model response" which sweeps
the CURRENT sidebar inputs and draws the model psu(Ppf) curve (grays out when
inputs change). If the green line does not move like the dots, the fit will
not predict pressure changes. Data: `GET /api/wells/{well}/response-history`.

## 9. Single-point calibration guard

`calibrate_friction_coefs` returns `match_quality="pinned"` with seed
coefficients and an explanation whenever the final solve is sonic - a single
BHP cannot identify friction there (ken would only move the floor). Both UIs
show the message and do not apply. Subsonic behavior unchanged; `bounded`
results carry a low-confidence caution.

---

## 10. Tuning knob reference

`server/services/evidence.py`:
| knob | value | effect |
|---|---|---|
| FLOOR_PCTL | 5 | lower -> floor closer to the absolute min (more aggressive violations) |
| MIN_PAIRS | 5 | pairs needed before a well earns its own beta |
| PAIR_WINDOW_DAYS | (3, 30) | wider -> more pairs, more IPR-drift contamination |
| DPF_MIN_PSI | 100 | pair qualification threshold |
| BETA_CLAMP | (0.0, 0.5) | response slope bounds |
| BETA_DEFAULT | 0.09 | fallback slope (measured MPM-64 Nov event) |
| PSU_REF_DAYS | 14 | window for the measured operating suction |

`woffl/gui/pad_optimize.py`:
| knob | value | effect |
|---|---|---|
| _EVIDENCE_VIOLATION_MIN_PSI | 25.0 | floor-gate trigger margin |
| _EVIDENCE_BETA_MIN | 0.03 | response-gate trigger (field separation 0.022 / 0.04) |

`server/services/calibration_points.py`:
| knob | value | effect |
|---|---|---|
| MAX_FIT_POINTS | 20 | fit cost vs coverage (solves scale linearly) |
| STEADY_STATE_TOL_PSI | 60.0 | transient-day rejection; lower = stricter |
| refusal thresholds | 10 pts / 200 psi spread | identifiability floor |

`woffl/gui/fric_calibration.py` (multipoint block):
| knob | value | effect |
|---|---|---|
| MP_DBHP_SCALE | 25.0 | pair-residual normalization; lower -> response weighs more |
| MP_MIN_DPPF_PSI | 100.0 | pair qualification inside the fit |
| MP_HUBER_DELTA | 1.5 | outlier robustness; lower = more median-like |
| MP_MAXITER | 100 (x starts + polish) | fit budget |
| KEN/KTH/KDI/FNZ/MACH_CRIT bounds | (.005-.40)/(.05-1)/(.05-1)/(.8-1.3)/(1-2.5) | parameter ranges |
| GOOD_PSI / MULTISTART_THRESHOLD | 25 / 50 | escape/alt-start triggers |

## 11. Validation harnesses (scripts/, all read-only, run from repo root)

```bash
# fleet evidence table + choke plan with/without corrections + ladder diff
PYTHONPATH=. venv/Scripts/python.exe scripts/evidence_validation.py

# multipoint fits for every eligible M-Pad well: old vs new params, RMS,
# implied beta vs mined beta, refusals (THE judgment run after fitter tuning)
PYTHONPATH=. venv/Scripts/python.exe scripts/multipoint_validation.py

# field dots vs model curves PNG for MPM-64/28/45 (update the hardcoded
# fitted params from the harness output first)
PYTHONPATH=. venv/Scripts/python.exe scripts/response_eyeball.py

# older probes: knee ladders, floor sensitivity, ken decomposition
PYTHONPATH=. WOFFL_MAX_WORKERS=8 venv/Scripts/python.exe scripts/mpad_knee_probe.py
PYTHONPATH=. venv/Scripts/python.exe scripts/mpm64_floor_probe.py
PYTHONPATH=. venv/Scripts/python.exe scripts/mpm64_ken_decompose.py
```

## 12. Known limitations / open items

- **Era-pure fits vs mixed-era betas**: the mined beta blends pump eras (the
  Nov 2025 MPM-64 event was the PRIOR pump); event fits are era-pure. Ambers
  on the response check often mean "the eras disagree", not "the fit failed".
  Judge with the response diagnostic. Possible refinements: era-pure beta in
  the evidence miner, or probing implied_beta where the era data has support
  instead of at median ppf.
- **implied_beta probe location**: computed at (median ppf, median - 300);
  a fit with a knee below the median legitimately reads ~0 there.
- **Railed old calibrations**: wells with ken = 0.40 / kth = kdi = 0.05 saved
  fits (most 14Bs) should be event-recalibrated and re-saved as data allows.
- **MPM-10 / MPM-60 refusals**: every fit point unsolvable at their configs -
  investigate the configs (probably IPR/geometry inconsistency).
- **Young eras (post-JPCO wave)**: 7 M-Pad wells refuse until ~2 weeks of
  daily history accumulates; the fallback single-point match covers levels.
- **psu_minimize criterion**: mach_crit is a calibratable closure over a
  homogeneous Mach-1 criterion that is known-conservative; if fits keep
  wanting mach_crit at bounds, the criterion itself is next (upstream PR).
- **Grand-plan pillars not yet built**: decision -> outcome loop (pillar 5);
  scorecard does not yet auto-refresh or persist history.

## 13. Data reference

Views: `mpu.wells.vw_pressure_daily` (daily tubing/annulus/bottomhole psi,
since 2024-09-25; names like "M-064"), `mpu.wells.vw_power_fluid_volume`
(daily PF bpd, `pwr_fld_net`), `mpu.wells.vw_bhp_daily_clean`,
`mpu.wells.vw_well_test` (with test-day PF join), `mpu.wells.prop_hist` /
`prop_xref`.
prop_xref rows added 2026-08-10: `jpfric_nozzle_area` ("Jet Pump Nozzle Area
Factor (washout)"), `jp_mach_crit` ("Jet Pump Critical Mach (slip closure)"),
both unitless/mechanical/double.

## 14. Other changes this session

- Choke plan UI: per-well IPR curve grid + dumbbell fixes + sonic/cavitation
  markers on the landing table.
- IPR anchor: "Median - BHP" vs "Median - Liquid rate" split; UI always
  follows the fit's own anchor (fixed the median-anchor mismatch and the
  wrong-test pinning on save).
- M-Pad pump curves: operating cap at 60 Hz (datasheet 61); frontier and
  regression pins re-derived.
- `worker_ceiling()`: unset-local default capped at min(cores, 8) after a
  14-worker OOM.
- Full suite at session end: 2082 passed / 1 skipped; web typecheck clean.
