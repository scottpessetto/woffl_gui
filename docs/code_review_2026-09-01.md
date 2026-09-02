# Codebase Review — 2026-09-01

> **STATUS (2026-09-02, fix run):** the P0/P1 backlog below is largely CLOSED in the working
> tree (uncommitted, on `main` beside the pre-existing E-Pad work). Suite: **~1,600 passed**
> (was 1,491). Library patches are registered as `docs/upstream_sync.md` **#16–#28** (+ the
> solver-core entries that follow), each tagged and guarded by a red-if-lost test.
>
> **Fixed — physics:** FLOW-1 Payne re-floor (#16); PVT-F1 acoustic compressibility in `cmix`
> with a dead-oil floor (#17); PVT-F2 V-B floor (#18); PVT-F3 DAK z-factor (#19); PVT-F4/F5
> ResMix guards (#20/#21); FLOW-2 raw-survey TVD in the traverse (#23; fleet p90 111 ft → 0);
> FLOW-3 survey validation (MPC-05 and four more corrupt CSVs now raise → preset + warning:
> **re-pull MPC-05, MPC-40, MPI-11, MPL-20, MPM-64 from Oracle PDB**); FLOW-10/11/12 and the
> `Pipe` wall guard; server JP-MD from measured chars `JP_MD` / first crossing (rows 1–2 of the
> table); SRV-6 `/calibrate` fits the same pump as the page; SRV-7 water-mode flag; SOLV-F4 R²
> of the returned curve; SOLV-F5 RP floor-fallback flagged + weak, field cap from chars, anchored
> path clamped; SOLV-F7 pinned branch returns the caller's coefficients; EVID-F6 per-point IPR
> re-anchoring along the test's Vogel curve; EVID-F7 measured WHP on daily points; EVID-F13/F14
> header-trend median estimator + 15 psi driver floor; EVID-F17 generic-IPR refusal within 300
> psi; EVID-F22 circulation direction in every tool; EVID-F23/F24 Fric Trend measured PF /
> Wash-Out ratio flag; EVID-F25 gaugeless back-calc returns None at the wall; OPT-A1 honest
> surface gaps + infeasible states; OPT-A2 anchor on live p0 (with SRV-15); OPT-A3 dominated
> configs gated with their semi; OPT-A4 MCKP shut-in allowed + infeasible trial = empty, never a
> raise; OPT-A6 match check keeps measured PF; OPT-A9 unanchorable online wells refused.
> **Solver core (#29–#35):** SOLV-F2 walk-inward now bisects on feasibility to the edge and
> `sonic_status` means "within 2 psi of the choke floor" (11A: reported sonic at Mach 0.50 with
> residual +35 → bracketed, residual −8, subsonic); SOLV-F3 bisection first-probe guard + rates
> re-evaluated at the returned psu; FLOW-5 the throat-entry clamp is accepted only within 2 %
> of the kinetic head (converged floors clamp at ≤ 0.5 %, fabricated states at 26–98 %) and
> raises `ThroatEntryChoked` otherwise; FLOW-9 `psu_minimize` interpolates tde at `mach_crit`
> and requires |tee| ≤ 1 % (cannot-be-fed pumps now raise instead of "sonic, 30 BOPD"); FLOW-6
> bounded hump search; FLOW-7 list-based `JetBook` (bit-identical); SOLV-F9 throat-entry sweep
> reused; SOLV-P3 known-point secant. Timings: E-41 9X 16.6 → 7.5 ms, marginal 12B 81 → 43 ms,
> 48-pump batch 1245 → 918 ms. Suite after all of it: **1,616 passed**.
>
> **Databricks verification (run live 2026-09-02, read-only; the write gate stayed off):**
> DATA-14 CLOSED — `vw_bhp_daily_clean` has zero duplicate (enthid, tag_date) keys. DATA-12
> CLOSED as immaterial — 191 of 178,768 well-days in `vw_pressure_daily` have more than one
> sample (max 2). DATA-13 OPEN and REAL: the warehouse session is UTC, 102,725 of 165,398
> `wt_date` stamps carry a time of day spread UNIFORMLY over all 24 UTC hours (4.0–4.6k per
> hour — not a workday pattern), and 16,094 of the 31,782 tests since PF coverage (51 %) fall
> on a DIFFERENT calendar day in Alaska than in UTC. The daily views key on DATE columns. Which
> day is right depends on what `wt_date` actually is (a UTC instant, or a local wall-clock
> stored as UTC) — a question for the data owner before anything is changed.
>
> **Fixed — truth/provenance:** DATA-1 `.env` never exports the write gates; DATA-2 same-day
> install tie-break; DATA-3 fleet tests survive a bad well name; DATA-5 pump source badge
> (bundled-xlsx fallback flagged); DATA-6 `is_sch` null-safe + `is_sch_estimated`; DATA-7
> override rows never lock as as-built; DATA-8 beyond-survey depth → estimated; DATA-11 retry
> only on session errors; DATA-15 CSV fallback never cached; EVID-F1 distinct PF events;
> EVID-F2 per-era floor (`floor_source`); EVID-F3 scorecard `contradicted` needs the sonic
> claim and an era floor; EVID-F5 `beta_raw`; EVID-F26 SQL preview is append-only INSERT;
> SRV-3 WC ≥ 0.99 refused; SRV-4 offline wells are CFP bring-online candidates; SRV-5 anchor
> test's PF + direction seeded; SRV-9 clamped seeds reported (`clamped`) and shown; SRV-11
> (partial: WC/offline); WEB-1 Alaska timestamps; WEB-5 nozzle 15; WEB-15 sensitivity targets
> the Solver's comparison test; WEB-16 (partial: the RP-fallback wording); WEB-17 PF caption;
> AGENTS.md rate convention + stale §3/§8 notes corrected.
>
> **Fixed — performance:** SOLV-P1 lazy matplotlib/scipy (#22); DATA-9 (partial: warm loop
> still per-well — see below); DATA-10 nested test windows sliced from the 24-month frame;
> SRV-8 combine study on the server pool; SRV-12 job semaphore (`WOFFL_MAX_JOBS`); WEB-2
> ChartPanel memo; WEB-6 context keyed on well; WEB-7 background polling; WEB-8 job ids
> persisted per pad / per well.
>
> **Still open:** DATA-9 warm-loop per-well fan-out (needs the fleet daily-BHP frame);
> DATA-13 `wt_date` day semantics (see above); OPT-A5 parsimony ratio test, OPT-A7 E-Pad budget/knee,
> OPT-A11 water basis, OPT-A13/A14/A15 per-well pump lists / shared executor / surface cache,
> and the FLOW-4 `mach_crit` energy-balance decision — all deferred to the optimization
> redesign pass (§8); WEB-14 server-owned client defaults; EVID-F8 mach_crit profile
> likelihood; EVID-F28/F29 separator baseline and OIW units; PVT-F6 water PVT(T); SRV-13/14.

**Scope:** entire working tree — `woffl/` library (pvt, flow, geometry, assembly), `woffl/gui/` plants and optimizers, `server/` FastAPI services and tools, `web/src/` React SPA. Reviewed by eight parallel deep-read agents (PVT, flow+geometry, solver+IPR+calibration, optimization engines, Databricks data layer, server core, evidence/calibration/tools, web SPA). Findings deduped; the highest-impact claims were re-verified by hand against the code and, where cheap, by a numeric probe.

**Baseline:** `1491 passed` in 28 s (`tests/test_joint_match_sweep.py` deselected). Line numbers reference the 2026-09-01 working tree, which carries uncommitted E-Pad / event-calibration edits (reviewed as-is; nothing half-finished found beyond WEB-5 and two stale comments).

**Brief:** performance, physics busts, and places that assume or guess an answer and present it as truth. The optimization schemes were audited for busts only; a state-of-the-schemes note at the end seeds the redesign pass.

**Verification key:** ✓ = orchestrator re-read the code and/or reproduced the number. Everything else is agent-reported with the stated evidence; check the line before fixing if it looks off.

**No files were modified.** Probe scripts are in the session scratchpad.

---

## 0. The ten that change answers today

| # | Finding | Where | Effect |
|---|---|---|---|
| 1 ✓ | Jet-pump measured depth from `np.interp` over a toe-up survey | `server/services/factories.py:215`, `wells.py:688` | 8 wells (MPH-31/32, MPM-20/22/24/43/45, MPS-17) solved with the pump at the lateral toe. MPH-31: 21,180 ft vs measured 5,144 |
| 2 ✓ | Optimizer never sets `jpump_md`, so MD = TVD for every well | `server/services/optimizer_runs.py:117-150` | Every pad / CFP / match-health / event-calibration run traverses a vertical well to the pump |
| 3 ✓ | Payne correction applied after the no-slip floor, never re-floored | `woffl/flow/outflow.py:67-68` | Gas-free tubing modeled at HL 0.924. Low-GOR wells −7 to −20 % oil; friction calibration has been absorbing it |
| 4 ✓ | Wood's-equation sound speed uses McCain material-balance oil compressibility | `woffl/pvt/resmix.py:331-341`, `blackoil.py:155-177` | Mixture sonic velocity 10–50 % low below Pb; +33 to +78 % jump across a 2-psi step at Pb. `mach_te` biased high → early "sonic" verdicts everywhere |
| 5 ✓ | `mach_crit` scales the choke floor but not the throat-entry zero the discharge residual uses | `solopump.py:141`, `jetflow.py:57`, `jetplot.py:267-284` | With a calibrated `mach_crit > 1` the operating point violates the solver's own energy equation (12B at 1.5: +30 % oil, ~400 psi-equivalent unaccounted). The fitter persists these to `prop_hist` |
| 6 ✓ | CFP response surface fills non-converged grid points with the nearest converged value | `woffl/gui/cfp_moves.py:105-135` | A pump that only converges at high PF is scored as feasible at low PF; feeds the λ-sweep and every move board |
| 7 ✓ | Header Impact reads `"EmpClass"` while the row key is `"Emp class"` | `server/services/tools/header_impact.py:967`, `runs.py:243` | The physics-vs-field disagreement verdict can never fire; the page's "Responsive" count is always 0 |
| 8 ✓ | `default_pad_pf(pad)` is passed a pad letter; it takes a well name | `header_impact.py:301` | Every pad without a live reading defaults to 3,400 psi (G/J/B are ~2,200) |
| 9 ✓ | `pad_pf_cluster(pad)` is passed a string; it takes a DataFrame | `optimizer_runs.py:534-541` | Always swallowed → CFP never uses measured header pressures. This currently MASKS OPT-A2 (July 2,792 anchor constant); fix both together |
| 10 | Survey piecewise fit under-fits; production hydrostatic uses the fitted TVD, PF side uses the raw TVD | `woffl/geometry/wellprofile.py:376-441, 196` | Fleet p90 111 ft / ~40 psi, max 240 ft; MPE-48 stops at rmse 154 ft when 12 ft is attainable |

---

## 1. Physics busts

### Library — PVT (`woffl/pvt/`)

- **PVT-F1 (P0) ✓** — `ResMix.cmix` is Wood's equation, but below Pb `BlackOil.compress` returns McCain-Rollins-Villena co, which includes liberated-gas volume. Pure oil "sound speed" is 111–877 ft/s below Pb vs ~4,000 real, then 4.4× across Pb. Reproduced: Schrader 100 °F, WC 0.8 / GOR 400 → 1,244 ft/s at 1,749 psig, 1,657 at 1,751. Consumers: `mach_te`, `sonic_status`, the throat-entry subsonic mask, Header Impact "sonic-decoupled", the optimizer's "never swap for a sonic one". Fix: an acoustic co (Vasquez-Beggs form at Rsb — what is already used above Pb) in the `cmix` path; keep McCain under its own name. `mach_te` pins in `tests/batch_test.py` will move. Library change: tag + `upstream_sync.md` + regression test.
- **PVT-F2 (P1)** — Vasquez-Beggs above-Pb co numerator has no floor and goes negative for 10–14 API at 60–80 °F (Ugnu range): Bo rises with pressure; `cmix` can hit a `sqrt` domain error as an untyped `ValueError`. Fix: floor with a warning, or `compressibility_kartoatmodjo_above` (already implemented, `blackoil.py:486`).
- **PVT-F3 (P1)** — Cubic z-factor drifts −10 % at ppr 4.5 (γg 0.65, 80 °F), −20 % at ppr 6, and returns 0.143 vs DAK 0.686 for γg 1.0 — inside the accepted range and above the clamp. Wrong sign of dz/dp beyond ppr ≈ 3 → cg overstated → `cmix` understated (compounds F1). Fix: flip the commented one-liner at `formgas.py:107-110` to `_zfactor_dak` (verified correct).
- **PVT-F4 (P1)** — `ResMix` accepts wc ∉ [0,1] and fgor < 0 silently (wc 1.05 → oil fraction −0.055). `WellConfig`, CSV stores and `prop_hist.form_wc` reach it unguarded. Add the range check every child class already has.
- **PVT-F5 (P1)** — When fgor < Rs(p) free gas is clamped to zero but the oil is still evaluated at Rs(p): 48 scf/stb of phantom dissolved gas at 1,400 psig (Schrader, fgor 150). ~0.5 lbm/ft³ on density, 5–10 % on viscosity. Fix: `min(Rs(p), fgor)`.
- **PVT-F6 (P2)** — Water density `sg × 62.4` and viscosity 0.75 cP are temperature-free constants ("come back later"). ~50 psi on a warm PF column, 1.7× on Reynolds at the ends of the range. All four field presets are the same 1.02 SG.
- **PVT-F9 (doc)** — Isothermal cg in Wood's equation is defensible (Himr); resolve the "adiabatic?" docstring so nobody "fixes" it.

### Library — flow and geometry (`woffl/flow/`, `woffl/geometry/`)

- **FLOW-1 (P0/P1) ✓** — see table row 3. One-line fix: `slh = max(slh, nslh)` after `payne_correction`, and HL = 1 when there is no free gas. Library change.
- **FLOW-2 (P1)** — see table row 10. Best fix: interpolate `vd_seg` from the raw survey in `_outflow_spacing` (the 100-ft segmentation already bounds node count, so the fit buys nothing and the PF/production inconsistency disappears). Also removes the 11–161 ms per-well profile cost (FLOW-8).
- **FLOW-3 (P1)** — `WellProfile` accepts a corrupt survey. `MPC-05 Deviation Survey.csv` has 179 duplicate MDs with alternating TVDs (6090 → 0 / 4547); the `_horz_dist` clip at `wellprofile.py:253` turns the impossible steps into silent zeros. TVD at 0.85·TD is +281 ft (~100 psi). Fix: reject duplicate MDs with disagreeing TVDs; re-pull the CSV.
- **FLOW-4 / SOLV-F1 (P1) ✓** — see table row 5. `throat_entry_zero_tde` takes no `mach_crit`; with it > 1 `_dete_zero` never finds a crossing and clamps to the tde minimum. Reproduced by reviewer probe: 12B at `mach_crit` 1.15 / 1.3 / 1.5 leaves 41 / 65 / 83 % of the kinetic energy unaccounted. `mach_crit = 2.5` (the multipoint fitter's upper bound and escape seed) raises on every E-41 pump. A consistent slip closure changes the density walk / effective sound speed for BOTH the stop criterion and the walk, not kde. Until then treat `mach_crit > 1` as diagnostic and stop persisting it (see EVID-F8).
- **FLOW-5 (P1)** — `_dete_zero`'s "tde stays positive → clamp to min" also fabricates a solution for pumps that are choked at every suction (tee +24 % of kde at psu = pres − 10 → reported "sonic, 30 BOPD"). Bound the clamp: accept only when the masked minimum ≤ 1–2 % of kde, else `ThroatEntryNoSolution` so BatchPump marks the combo infeasible rather than choked.
- **FLOW-6 (P2)** — `_throat_discharge_bracketed`'s 60-point scan (step ≈ 0.1·pte) misses a positive hump narrower than one step — exactly the marginal pumps the fallback exists for — and raises `ConvergenceError`. Fix: bounded scalar maximization of the balance, then brentq on [peak, hi].
- **FLOW-9 (P2)** — `psu_minimize` exits on |Δpsu| ≤ 5 only; two clamps at the bounds exit "converged" silently. Also require |tee| ≤ ~1 % of kde.
- **FLOW-11 (P3)** — `froude == 0` raises a bare `ZeroDivisionError` in holdup (escapes every `except ValueError`); the zero-flow guard was added to friction only.
- **FLOW-12 (P3)** — Laminar/turbulent step at Re 4000 (f 0.016 → 0.041). Standard is ~2300 with a blend.

### Solver assembly (`woffl/assembly/solopump.py`, `batchpump.py`)

- **SOLV-F2 (P1)** — After a suction walk-inward, the first feasible probe on the fixed fraction grid with a positive residual is returned as `sonic_status=True`. Marginal fixture, 12A: floor 1,844.5, walked to 1,854.7, residual +54.5, reported sonic at Mach 0.53; the true root lies in the unprobed gap. Grid gaps reach 14 % of span → up to ~140 psi high on a 1,000-psi span. Cascades: `fric_calibration` refuses the well as "pinned", Header Impact tags it "sonic-decoupled". Fix: bisect on feasibility between the last infeasible and first feasible fraction; derive `sonic_status` from `mach_te ≥ mach_crit − ε`.
- **SOLV-F3 (P1)** — `_bisection_solve`: the first midpoint is evaluated outside the `try`, so a `JetPumpError` there surfaces as "solver failed" on a bracketed well; an interior exception leaves the rates from the previous midpoint, so the returned psu and rates can belong to different suctions (469.5 vs 470.3 in probe).
- **SOLV-F9 (P2)** — `discharge_residual` re-sweeps the throat entry that `psu_minimize` just computed (`te_book` discarded). ~10 % of a sonic solve.

### IPR fitting (`ipr_analyzer.py`, `gui/ipr_anchor.py`)

- **SOLV-F4 (P1)** — `compute_vogel_coefficients` picks RP as the minimum over all anchors, returns the recent-test `qwf/pwf`, and reports R² computed with the median anchor. Probe: reported 0.844, R² of the returned curve 0.920. The server's weak-IPR gate reads this number.
- **SOLV-F5 (P1)** — Floor fallback RP = max BHP + 50 manufactures a hyper-productive curve (Vogel qmax/qtest 11–28× at pwf/pres 0.95–0.98) with no weak-fit signal; the `ipr_anchor` copy is not clamped to the field cap (1,840 > 1,800); the server calls `estimate_reservoir_pressure` without `jp_chars` so every well is capped at the Schrader 1,800 — a Kuparuk well with any test above 1,790 hits the fallback. Fix: `rp_source` flag that forces the weak-fit path; pass the field model; one fallback helper.

### Optimization engines (`woffl/gui/pad_optimize.py`, `cfp_*`, `optimization_algorithms.py`)

- **OPT-A1 (P0) ✓** — see table row 6. Return `None` outside `[first_valid, last_valid]` and inside interior gaps; make `_best_option` / `settle` treat that option as unavailable. Add a test with a surface that is `None` at the low grid points.
- **OPT-A2 (P1) ✓** — CFP delivered PF = measured pad PF + (P − 2,792), the 120-day July constant, while the server passes today's live `p0`. At P0 = 2,850 B-pad is modeled 58 psi above what its wells see. Masked today by SRV-15 (row 9); fix together with `anchor_disch_p=p0`.
- **OPT-A3 (P1)** — MILP marginal-WC gate reads `molwr/motwr`, which `process_results` writes only on `semi` rows; NaN fails open, so the gate removes a semi config and keeps the non-semi config it dominates.
- **OPT-A4 (P1)** — MILP and MCKP solve different problems: candidate sets (all converged rows vs semi-only after "best throat per nozzle"), shut-in (at-most-one vs `add_exactly_one`, `allow_shutin=False` hard-coded at `optimization_algorithms.py:599`), and failure mode — MCKP's `RuntimeError` at a tight budget (`network.py:80-85`) is uncaught and kills an entire I/M/E run. Minimum: `allow_shutin=True` or catch and mark the trial skipped.
- **OPT-A5 (P1)** — `apply_parsimony` has no ratio test (gives up 19 BOPD to save 50 BPD) and never reallocates the freed PF. Up to ~400 BOPD/pad discarded under the label "parsimony". Gate on the run's own λ.
- **OPT-A6 (P1)** — Match check and every scenario evaluator overwrite each well's measured `ppf_surf_well` with the plant header (`pad_optimize.py:922, 996, 652, 823`, `cfp_optimize.py:427`). The pre-flight trust check therefore compares wells to tests at the wrong PF (E-Pad: 3,472 vs setpoint 3,400).
- **OPT-A7 (P1)** — E-Pad `budget_at_pressure` is 32,400 BPD for every P in 3,000–3,500 (frontier at the flow ceiling is 4,005), so the sweep is degenerate and returns the 3,500 cap "adopted from I-Pad"; suction 2,800 is a workbook cell. Below the knee the frontier collapses and the fixed-point evaluators drive the header down, where a real plant recirculates and holds setpoint.
- **OPT-A8/A9 (P1)** — `AnchoredPlant.pressure_at` silently floors at `min(p_grid)` (BOL over-recommended); an online well whose current pump never converged at any grid point is treated as idle in W0, so its "bring online" shows as a gain.
- **OPT-A11 (P1)** — `water_key="lift_wat"` is hard-coded for all pads while `well_sort_engine.POPS_PUMP_HANDLES` says E/F/M pumps handle total water. Basis mismatch between the pad optimizer and Well Sort on E/M.
- **OPT-A12 (method)** — Model bias does not cancel for the well being moved (its absolute modeled water enters the delta). The methodology doc overstates robustness; the trust gate should weight action-list wells.
- **OPT-A18/A25 (P3)** — `FixedHeaderPlant.budget` = inf → MCKP `OverflowError`; S-Pad discharge cubic extrapolates to −400 psi at 55,080 BPD.

### Evidence, calibration and tools (`server/services/`)

- **EVID-F6 (P1)** — Multipoint calibration attaches the nearest test's oil (≤ 30 d) to each daily point but keeps the day's own BHP, then re-anchors `InFlow` per point at (oil_test, bhp_day). Over a month sharing one test every point asserts the wrong rate at its drawdown; the residual becomes systematically signed with PF, and the fit rewards a LESS responsive model, leaning on `mach_crit`/`fnz` to reconcile. The single biggest question over `implied_beta` and fitted `mach_crit`. Fix: one IPR per era anchored at the test, or `oil_k = Vogel_test(bhp_k)`.
- **EVID-F8 (P1)** — `mach_crit` (1.0–2.5) is the only knob that lowers `psu_min`, so any reason the gauge sits below the modeled floor (GOR, Pb, IPR anchor, allocation, gauge datum) is fitted as "slip" and persisted as a well property that transfers to hypothetical pumps. The floor probe shows fgor×0.5 or throat+1 move the floor as much. Fix: profile likelihood; refuse to persist when the cost is flat; cross-check before accepting > ~1.3.
- **EVID-F2 (P1)** — The measured floor is min over 365 days across pump eras and fluids, then used to falsify the CURRENT pump's floor — and the floor gate is the one that "wins". Compute per era (`Date Set` is already fetched).
- **EVID-F3 (P1)** — `match_health._verdict` returns `contradicted` whenever model psu − floor > 25 with no sonic condition, on a psu modeled at a synthetic pad header. Paints well-behaved subsonic wells red.
- **EVID-F1 (P1)** — `n_pairs` counts overlapping day pairs; one PF step with three days each side is nine pairs and promotes the well to `beta_source="well"` — the only source allowed to trigger the response gate. Count distinct PF moves, require ≥ 2.
- **EVID-F13/F14 (P1)** — Header Impact's good-day filter (r² ≥ 0.5, slope ∈ [0.2, 1.5]) is a truncated estimator: it cannot report a coupling below 0.2 and a true 0.12 with daily σ 0.15 reads ~0.3; the group donor default inherits the floor. `min_x_range = 2 psi` admits days with no slope information. Keep the band for classification; estimate with Theil-Sen (already wired) over all days.
- **EVID-F22 (P1)** — Forward-circulation wells (MPS-17, MPE-17, MPL-20, F-pad) are modeled REVERSE in every tool: `jp_calibration:404` hard-codes it, `jp_fric_trend`, `jp_washout`, `pf_scenario` omit it, `_common.build_well_config` never sets it.
- **EVID-F23/F24 (P1)** — Fric Trend infers a PF pressure from allocated lift water at seed coefficients when the test row carries the measured `pf_press`; Wash-Out flags on a fixed 3,400 psi so a 2,200-pad pump needing 3,000 (+36 %) is not flagged while an M-pad pump needing 3,450 (+1.5 %) is.
- **EVID-F25 (P1)** — Gaugeless BHP back-calc returns the bracket edge (100 psi) as "sonic" and seeds a synthetic Vogel at 100 psi; Header Impact imports it and nothing distinguishes solved from hit-the-wall.
- **EVID-F17 (P1)** — ESP wells: assumed 1,800 ResP plus `pres = max(res, bhp + 100)` manufactures a 100-psi-drawdown PI; a 30 psi ΔBHP then predicts ~30 % of the well's oil, rolled into the pad total with an unlabeled integer ResP column.
- **EVID-F28 (P1)** — Separator oil-loss film baseline references each reading to the trailing-24 h p95 of the analyzer's own WC, so sustained real carryover longer than a day becomes the plateau and is billed as zero; both bounds share it. Add the raw `100 − wc` integral as a ceiling or anchor to lab OIW.
- **EVID-F29 (P2)** — OIW grab samples: ppm (mg/L) treated as volume fraction (÷ oil density ≈ +15 %); one constant water rate across the whole log while the daily meter is already cached.
- **SOLV-F7 (P1) ✓** — Single-point friction calibration fits three coefficients to one scalar (non-identifiable), with bounds several times the published ranges (ken ≤ 0.40 vs 0–0.03), and `NEUTRAL_KDI = 0.30` ≠ the 0.40 default, so the refused "pinned" branch hands back kdi 0.30 and `save_ipr_values` pushes a `jpfric_diffuser=0.3` row for a calibration that was refused.
- **SOLV-F8 (P1)** — `calibrate_pf_for_lift` / `estimate_nozzle_wear` report `converged=True` on a bracket that collapsed by failures (probe: converged, residual −1,000 BWPD = 40 % of target). Wash-Out displays the flag verbatim.
- **SRV-6 (P1)** — `/calibrate` ignores `nozzle_area_factor` and `mach_crit`, so the coefficients it returns were fit against a different pump than the page then solves with.
- **SRV-7 (P1)** — `pressure_profile` builds the throat-mixture `ResMix` without `model_as_water` — a third construction site the settled decision does not cover.

---

## 2. Guessed values presented as truth

- **DATA-1 (P1, safety) ✓** — The server warm loop opens a Databricks connection seconds after startup; `_new_connection` calls `load_dotenv()` unconditionally and `.env` carries `ALLOW_DATABRICKS_WRITES`, so the local production write gate flips ON with no user action. `writes_enabled()` reports False on the first `/meta` call and True on the next. Fix: read credentials with `dotenv_values()` and copy only the `bricks_*` keys.
- **DATA-2 (P1)** — Same-day pull+set (the JPCO pattern) resolves to an arbitrary pump: `get_current_pump`, `get_pump_at_date`, `build_pump_eras` and `JP_HISTORY_QUERY` all order by `Date Set` alone with an unstable sort. Current-pump identity can flip between fetches. Add a secondary key.
- **DATA-3 (P1)** — Fleet well tests are fetched through an IN-list of all ~487 header names, each passed through a strict name regex; one odd name (a sidetrack suffix, lowercase) raises and every well's tests become `None`. Use the enthid join the other two bulk readers already use.
- **DATA-5 (P1)** — Tracker failure falls back to `data/jetpump_history.xlsx` dated 2026-03-05 and seeds nozzle/throat as installed; the source is discarded before it reaches the context payload.
- **DATA-6/7/8 (P1)** — `is_sch` is True for a NULL pump depth (`fillna(0) < 5500`) and inherits estimated TVD without provenance; `local_well_overrides.csv` placeholders (friction 0.2 ×4, ResP 1,500, API 21, casing/tubing) render as measured, locked as-built; `np.interp` clamps a pump below the surveyed interval to the last station with `tvd_estimated=False`.
- **DATA-12 (P2)** — Four copies of the pressure-daily aggregate take `max()` per column independently with no ceiling; a tubing spike on a reverse-circ well flips it to forward and seeds the sidebar. Aggregate the pair from one sample; add `PF_MAX_VALID`; one SQL constant.
- **DATA-13/14 (verify)** — `WtDate` is made UTC-naive in pandas while the SQL joins use `to_date(wt_date)` in the warehouse session zone; and the daily-BHP join is unaggregated (if the view is per-bore, test rows duplicate). Two one-line queries settle both: `SELECT current_timezone(), min(hour(wt_date)), max(hour(wt_date)) FROM vw_well_test` and `SELECT enthid, tag_date, count(*) FROM vw_bhp_daily_clean GROUP BY 1,2 HAVING count(*) > 1`.
- **DATA-15 (P2)** — `list_wells` caches the CSV fallback for an hour (12 h under warm retention); after one blip every well's casing is the 6.875/0.5 default until expiry.
- **DATA-16/17/18/19 (P2)** — Well Sort: `FIRST(down_code)` without ORDER BY; multi-bore reservoir = alphabetically first; lift type from a single test (Header Impact's `classify_lift` is the better rule); pad-WC `fillna(0)` turns a NULL lift-water allocation into zero PF.
- **SRV-3 (P1)** — Optimizer caps WC ≥ 0.99 to 0.99 instead of refusing (stale comment cites a deleted module). A dewatering well enters as a 1 %-oil well and gets a pump recommendation.
- **SRV-4 / OPT-A10 (P1)** — CFP drops board-offline wells entirely, so they are never bring-online candidates; the comment and the project notes say the opposite.
- **SRV-5 (P1)** — The IPR-anchor seed omits the anchor test's PF pressure although the row carries it, so `/calibrate` fits to one day's BHP at another day's PF.
- **SRV-9/11 (P1)** — Seeds are silently clamped to widget bounds with no provenance field; the optimizer carries invented defaults (ResP 1,700 / qwf 750 / pwf 500 / WC 0.5 / GOR 250) that are live for a well with no tests, and the CFP path collects no provenance at all.
- **SRV-16 (P2)** — `to_simulation_params` `int()`-truncates GOR, temperature, surface pressure, TVD, PF, qwf, pwf and pres; sensitivity labels `round()` while the solve sees `int()`.
- **EVID-F18/F19 (P1)** — Every tool seeds from chars plus the AUTOMATIC Vogel fit (24 of 32 have R² ≤ 0 per `docs/ipr_model_review.md`) or the 0.5/250/750/500 defaults, with preset PVT — the saved anchor, well PVT, `jp_mach_crit` and wear factor never reach them. Tools therefore disagree with the Solver on the same well, unlabeled. Build tool configs through `optimizer_runs._config_from_seeds(well_context(...))` and add an "IPR src" column.
- **EVID-F7 (P1)** — Daily calibration points set `pwh` to the default (210) while the same row carries `tubing_prs`, the production WHP on a reverse-circ well; 50–150 psi of real variance is dumped into the friction coefficients.
- **EVID-F26 (P1)** — The calibration SQL preview emits `UPDATE mpu.wells.prop_hist SET jpfric_entry = CASE well_name …` against an append-only EAV table with no such columns.
- **WEB-1 (P1)** — Save-history timestamps display raw UTC unlabelled; the "19:22 that I don't know what it means" complaint the Streamlit page fixed regressed in the port. Format server-side with the existing `format_alaska`.
- **WEB-5 (P1) ✓** — Client pad-run nozzle default is 9–14; the server default and the single-well Batch default include 15. Pad runs never consider nozzle 15.
- **WEB-14/16/17/18 (P2)** — A dozen server defaults hand-copied into the client (p0 2,792, slope 13.69, C-pad PF 3,400, E-Pad 2,800/60/3,500, parsimony 20); `matchGrade` / washout thresholds claim to mirror deleted Streamlit code and differ from `match_health.py`; the "seeded pad default" caption shows the engineer's edited value; typed values silently clamp to duplicated bounds.
- **WEB-15 (P2)** — Sensitivity targets are always the most recent test while the Solver may be on the median or a picked test.
- **SOLV-F6 / OPT-A19 (P1)** — `rho_pf` is accepted, validated and shown, and never used; PF density is the field preset 1.02 in single-well and the formation `wat_sg` in pad runs. Decision: remove the field.

---

## 3. Performance

**Where the time goes.** A sonic single-well solve is 7.5–8.4 ms (one residual evaluation, ~220 `ResMix.condition` calls); a marginal secant-path solve is 28–41 ms; a 28-pump batch ≈ 0.35 s per well. A Databricks round trip is ≥ 150 ms warm. Compute is cheap; queries, process spawns and full cross-product sweeps are not.

| ID | Finding | Where | Cost / fix |
|---|---|---|---|
| SOLV-P1 | `batchpump.py` imports `matplotlib.pyplot` and `scipy.optimize` at module scope | `batchpump.py:11-16` | 0.98 s per ProcessPool worker spawn; a 20-well pad on 2 workers spends ~2 s spawning to save ~3 s. Lazy-import |
| FLOW-7 | `JetBook.append` does 8 `np.append` + a scipy `trapezoid` on a 2-point slice per step | `jetplot.py:82-110` | ~37 % of every solve. Python lists + inline trapezoid ≈ −30 % |
| OPT-A13 | "Installed pump only" passes simulate the full nozzle × throat product | `network_optimizer.py:628`; `pad_optimize.py:917, 993, 641, 811`; `cfp_optimize.py:429` | 24 solves per well where 1 is needed; choke plan does it 11× → ~5,000 solves vs ~220 per run. Accept a per-well `jp_list` |
| OPT-A14 | Fresh `ProcessPoolExecutor` per batch call, 8–11× per run | `network_optimizer.py:396-401` | Server has a primed persistent pool the optimizer cannot use. Accept an executor |
| OPT-A15 | No response-surface cache; `_fleet_signature` does not exist (project notes are wrong) | `cfp_moves` / `optimizer_runs` | Every CFP job rebuilds 7 batches |
| FLOW-8 | `WellProfile` fit 11–161 ms per well, uncached | `network_optimizer.py:710` | Vanishes with FLOW-2(a) |
| SOLV-P3 | Secant bound-clamp stall re-evaluates the identical point up to 20× | `solopump.py:611-636` | ~60 ms on exactly the marginal wells |
| SOLV-F9 | Throat entry re-swept after `psu_minimize` | `solopump.py:383, 141` | ~10 % of a sonic solve |
| EVID-F11 | Event calibration (~3 min): `_cost` re-evaluated 3× per run, polish always runs, `fatol` unreachable so every pass hits `maxiter`, `fnz` is nearly separable, whole pad hydrated for one well | `fric_calibration.py:938-975`, `event_calibration.py:182` | Algorithmic cuts first; the pool buys ≤ 2× on the 2-vCPU tier |
| EVID-F27 | Fric Trend: bisection + 7 Nelder-Mead starts × 150 iters per test | `fric_calibration.py:264-276` | O(10⁴) solves per well. Warm-start from the previous test |
| DATA-9 | Warm loop is a per-well fan-out: two queries per well × fleet × 6 workers every 6 h; new thread pools per pass leave connections open | `history.py:293`, `warmup.py:355, 395` | 180–260 queries where 2 fleet frames would do |
| DATA-10 | `WARM_TEST_MONTHS = (6, 12, 24)` runs the biggest query three times | `config.py:37` | Fetch 24 once, slice |
| DATA-11 | Any failing query closes the connection and nulls the shared token for all threads; the XV status query fails permanently every 5 min | `databricks_client.py:169-178` | Retry only connection-class errors |
| SRV-8 | Combine study forks a raw `ProcessPoolExecutor` (default start method) inside uvicorn, bypassing the pool gate | `sensitivity.py:816-826` | Route through `pool.submit_all` |
| SRV-12/13/14 | Unbounded concurrent jobs (each pad run spawns its own pools on 2 vCPU); `/batch` sweep on the request thread; jp-history payload = every daily BHP since first install | `jobs.py:100`, `solve.py:315`, `history.py:245` | Job semaphore; fan out batch; `?since=` |
| WEB-2 | `armedOption` memo keyed on inline `zoom` arrays → every chart tears down and re-inits (`notMerge: true`) on every parent render; Solver re-inits the IPR chart per keystroke and loses the zoom | `ChartPanel.tsx:420-455` | Stable zoom key + `replaceMerge` — single biggest UI fix |
| WEB-6/7/8 | Context refetched (and discarded) on window change; tool-job poller freezes in background tabs; event-calibration / match-health job ids die on navigation | `Layout.tsx:36`, `hooks.ts:682`, `EventCalibration.tsx:437` | Key on well only; `refetchIntervalInBackground`; persist job ids |
| WEB-9/10/11/12 | IPR chart memo on whole `params`; sensitivity store serialized to localStorage per keystroke; E-Pad panel builds 5 chart options inline; 6 pages subscribe the whole params object | various | `useMemo` / debounce |
| PVT-F14 | PVT hot path | — | Nothing material (~2 ms per 100-point sweep). Do not micro-optimize |

---

## 4. Safety and hygiene

- **DATA-20** — `_validate_single_insert` admits `INSERT OVERWRITE` (no live caller, but the guard's promise is weaker than its name).
- **DATA-4** — `WELL_ENTHID_QUERY` has no `field = 'MPU'` filter; a same-named producer elsewhere silently drops the well from the saved-IPR map.
- **DATA-22** — Four copies of the write-gate truthy check (AGENTS.md cites a fifth in a deleted file).
- **SRV-P3** — The 500 handler echoes the exception text (Databricks error strings, paths) to the client.
- **Chart rule** — `CfpCharts.tsx:181-197` custom `renderItem` dumbbell on an x-zoomed chart; `FricTrendPage.tsx:54` default tooltip on a time axis.

## 5. Dead code and stale documentation

- Zero callers: `well_test_processor.py`; `assembly/calibration.py` (+ `NetworkOptimizer.set_calibration` — a bare actual/model multiplier on oil AND water, good that it is dead); `pf_calibration.robust_bracket`; `cfp_optimize.run_joint_optimization` (the exogenous formulation the rules say never to reintroduce); `databricks_client.get_tags_for_wells`; `jetflow.py:404-480` Cunningham helpers; `deadoil.py`; eight `header_impact._*` helpers; the whole `header_report.py`; `jp_fric_trend._add_jpco_overlays` (builds a Plotly figure); `fric_calibration.compute_bhp_decomposition`.
- Duplicates: `server/services/factories.py:29-155` is a verbatim copy of `sim_factories` (AGENTS.md says the latter is THE copy); five pad-letter helpers; four pressure-daily aggregates; two `_SEED_BOUNDS`; two `WellConfig` builders with different truth; `_resolve_pump_for_test` ×2; `PARAM_BOUNDS` mirrors `schemas` by hand.
- Diffuser default is 0.30 in `JetPump`/`continuous_jetpump`/`NEUTRAL_KDI` and 0.40 in `params`/`jetpump_list`.
- `pipe.py:20` accepts `thick` up to `out_dia` (inner diameter goes negative); `pump_identity._LETTER_OFFSETS` omits X.
- **Docs:** `CLAUDE.md` describes the deleted Streamlit app throughout. `AGENTS.md §5` "sidebar qwf is OIL … snapshots convert" is inverted versus the code (total liquid everywhere, converted once) and cites a deleted module. `AGENTS.md §8` cites `scotts_tools/jp_calibration.py:930` and an "unguarded tag f-string" that no longer exist. `pad_page.py:1594` comment cited in §8 is in a deleted file. `_fleet_signature` caching described in the project notes does not exist.

---

## 6. Verified correct — do not re-flag

Kartoatmodjo Rs/Bo/μ, McCain co formula, Vasquez-Beggs formula, live-oil density, Abdul-Majeed tension, Lee-Gonzalez-Eakin, Sutton, DAK (unused but right), mass-fraction algebra, Wood units. Throat-entry energy balance and sign; sonic criterion self-consistent with the density walk (c_eff/cmix 0.98–1.02; tde minimum at Mach 1/√(1+Ken)); nozzle velocity; throat momentum balance (197 calls, zero low-root landings); diffuser convention; every Beggs-Brill constant, pattern boundary, holdup triplet, C-factor, ψ, transition, S and Ek; outflow signs; Serghide; Vogel; the vogel-not-pidx divergence. Rate convention is TOTAL LIQUID end-to-end with one downward conversion per site; WC bases are not mixed in `throat_wc`. Water-pump mode keyed on the flag. Walk-inward is bit-identical for already-converging wells. All three write guards, numbered bind markers, the six-condition delete gate, no second connect path, sentinel semantics, UTC ordering, token cache, thread-local connections, cache sizing. Pump-at-test is set-to-set everywhere. `classify_lift`. Header trend fits on hourly bins (not daily averages). Debouncing, explicit-submit sweeps, lazy routes and chart mounting in the SPA.

---

## 7. Recommended order

**Phase 0 — one-line fixes with large effect (hours).** Table rows 1, 2, 3, 7, 8, 9 + OPT-A2 together, OPT-A1, WEB-5, DATA-1, `NEUTRAL_KDI`, SRV-7. Each needs a pinned regression (MPH-31 → 5,144 ft; a gas-free segment at HL 1; a surface with `None` low points).

**Phase 1 — library physics (tagged, registered in `upstream_sync.md`, red-if-lost tests).** PVT-F1 acoustic co, PVT-F3 DAK, PVT-F2 floor, PVT-F4/F5 input guards; FLOW-2(a) raw-survey TVD + FLOW-3 survey validation; FLOW-5 bounded clamp; SOLV-F2 feasibility bisection + sonic from Mach; SOLV-F3 guards; FLOW-4 `mach_crit` — either thread it through the zero finder consistently or gate `> 1` as diagnostic and stop persisting it (EVID-F8). Expect `mach_te` and some `psu` pins to move; that is the point.

**Phase 2 — truth and provenance.** Seed-clamp and RP-fallback flags (SRV-9, SOLV-F5, SOLV-F4); tools built from `well_context` seeds with well PVT and circulation direction (EVID-F18/F19/F22); anchor-test PF in the seed (SRV-5); xlsx / CSV-fallback / estimated-TVD badges (DATA-5/6/15); Alaska timestamps (WEB-1); per-era floor, distinct-event count and the sonic gate in the scorecard (EVID-F1/F2/F3); per-point IPR in calibration (EVID-F6/F7); the three Databricks verification queries.

**Phase 3 — performance.** Lazy imports; `JetBook` lists; reuse `te_book`; per-well `jp_list`; shared executor + surface cache; warm loop on two fleet frames; retry policy; `ChartPanel` memo; background polling and job persistence.

**Phase 4 — optimization schemes (next conversation).** See §8.

---

## 8. State of the optimization schemes (seed for the redesign)

- **S-Pad** runs a knapsack at a fixed header iterated to a damped fixed point. That is a best-response equilibrium, not an optimum — each iterate ignores that its own PF draw lowers the header for every well — and the damping can oscillate between two discrete allocations (`converged` reports it; nothing acts on it). Because the header is a deterministic function of total flow, the right shape is the one-dimensional sweep the free-pressure pads already use.
- **I/M/E** sweep 11 (9) uniform header points with a knapsack at each. Sound structure; coarse near the optimum and wasteful far from it; E-Pad's budget is flat across the sweep (OPT-A7); "delivers exactly P" throttles any frontier surplus without pricing it.
- **MILP vs MCKP** are not the same problem (OPT-A4). MILP is the more complete one. Keep one; if CP-SAT stays, feed it the MILP candidate set with shut-in allowed.
- **CFP moves** is an anchored-delta λ-sweep plus exhaustive single/pair boards. Consistent only where per-well curves are concave; the boards guarantee the plan is never worse than the best single/pair move. The exposure is upstream: surface fill-in (A1), the anchor offset (A2), absolute-water bias on moved wells (A12).
- **Well Sort** is a WC-threshold triage, not an optimizer, on a different water basis than the pad optimizer for E/M (A11).
- **Divergence from the operational problem:** only water is priced (no amps, throttling loss, or JPCO cost); the economics cutoff exists three contradictory ways (hand `marginal_wc`, `derive_pad_marginal_wc`, `apply_parsimony`); a single pad header everywhere except CFP and the loops overwrite per-well measured PF; plant models are maximum-deliverable frontiers where operators run setpoints with bypass.
- **Recommendation:** one formulation — per-well (pump, delivered-P) response tables with explicit infeasible marks; a single MCKP/MILP over a P-grid (flow grid for S) with objective `oil − λ·water`, shut-in allowed, per-well measured PF offsets, setpoint semantics below the knee. The λ-sweep becomes a reporting view, not a second solver.
