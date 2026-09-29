# Optimization runs: review, fixes and speed (2026-09-24)

A full review of the optimization runs (pad resize and choke plans, CFP moves,
the MILP/CP-SAT allocators, the job/server layer and the Optimize page),
followed by fixes for every finding that was reproduced. Six parallel reviews
each reproduced their findings with scripts or exact code traces; unverified
suspicions were discarded. Local implementation only: nothing was committed,
deployed or written to Databricks.

Read this with [the September 12 capacity delivery](pad_cfp_capacity_delivery_2026-09-12.md),
whose formulation it keeps: maximize oil within capacity by default, one saved
oil IPR per well, installed versus clean replacement hardware, and explicit
qualification of plant limits and model coverage.

## What an engineer sees now

- **The answer first.** A pad run opens with one card: a status chip
  (Qualified / Conditional / Exploratory), "Change N pumps - modeled gain
  +X BOPD", the header and how much of the plant's water capacity the plan
  uses, the list of changes (well, from, to, gain), and a CSV export. CFP runs
  get the same card ("Make 2 changes - modeled +200 BOPD at 2,800 psi
  discharge") and choke runs say "Choke N, shut in M at a X psi header -
  projected +Y BOPD vs today". Results sit above the plant curves.
- **Fewer decisions before Run.** Pump grid, solver, objective and header
  setpoint are folded under **Advanced settings** with a one-line summary of
  what is non-default, plus **Reset to defaults**. The default path is: pick
  the pad tab, press Run, read the card.
- **Settings stay with their pad.** Each run tab (S/I/M/E/CFP) keeps its own
  persisted form; a pump count chosen on S no longer rides into an M or I run.
  The active tab is in the URL (`?tab=M`), so Back from a well link returns to
  the run.
- **Stale and missing results are named.** When an input changes after a run
  (offline or required wells, future wells, pump count, grid, booster, CFP
  pressures), the result says which, and asks for a rerun. An expired job says
  it is gone instead of vanishing.
- **Problems before sending.** Offline-and-required wells, choke runs whose
  future wells have no planned pump, E-Pad suction at or above the header cap,
  out-of-range numbers and CFP pad PF above the reference discharge are listed
  under the Run button. Server rejections (422) show their reason instead of
  "HTTP 422".
- **Progress you can read.** A progress bar with a time-left estimate from the
  run's own steps, and a queue position ("queued - 1 job ahead") when the
  single Medium job slot is busy.
- **Plain units.** Every pad table column carries its unit; "Current model
  oil" is "If unchanged"; the marginal column is "Marginal oil per 1,000 BPD";
  the manual water price is entered in BOPD per 1,000 BPD, the unit the results
  use (it was BOPD/BPD, 1,000 times off from the display).

## Defects fixed

Severity is the reviewer's. "Before" is the reproduced failure.

### Pad engine and choke plan (`woffl/gui/pad_optimize.py`, plants)

| Sev. | Defect (before) | Fix |
|---|---|---|
| High | Choke plan held a well at its **measured test rates** at headers below where the model can lift it, so a low header won on oil the well cannot make (450 vs a consistent 430 BOPD). | A well that solves somewhere on the ladder but not at or below a header is shut in there; only a never-solvable well is held at its test. |
| High | Choke "today" collapsed to plant suction when a reduced bank could not carry today's measured draw: M-Pad 1-of-3 pumps showed +2,850 BOPD "vs today"; I-Pad crashed at 217 psi. | When the settle is over capacity, today is modeled at each well's measured PF pressure (fallback: installed-bank settle, then the search window). `today_basis` is reported. |
| Med | The evidence (field suction) correction re-gated the lone "today" point, producing a +6 BOPD artifact for a well left exactly at today's header. | The today point takes the ladder's verdict per well. |
| Med | E-Pad stress cases called a low-flow point feasible that the run rejects (delivered header held the setpoint with no modeled recycle). | Stress scoring uses the run's own `_plant_operating_check`. |
| Med | All-trials-failed errors blamed amp limits even when required wells could not be served. | The message names the allocation status and cause. |
| Med | Solver errors at a header were silently skipped; the winner came from another header with status "optimal". | `sweep_complete=false` and the failed headers are shown. |
| Low | Refinement never bisected toward failed/zero-budget neighbours, so optima at a capacity edge were missed. | Refinement brackets against the nearest tried point. |
| Low | A degenerate pressure window solved one header 13 times. | Each decision point is solved once. |
| Low | `_settle_scenario_coupling` reported converged across a demand jump (205 psi residual). | Only a closed residual converges. |
| Low | E-Pad sweep ceiling read the nominal knee, not the amp-limited peak (3,870 vs 3,928 psi). | Always scans for the peak. |
| Low | `match_check` fallback header ignored formation water on M/E. | Adds it. |
| Low | Booster screen labelled a low-flow block "Recommended range (high)". | New "(low)" label. |
| Med | The choke landing-table IPR curve raised for about 6% of non-integer reservoir pressures (`pres * 24 / 24` rounding above `pres`), crashing the run. | Curve points are capped at reservoir pressure. |

### Allocation (`woffl/assembly/network.py`, `optimization_algorithms.py`, upstream patches 47-50)

| Sev. | Defect (before) | Fix |
|---|---|---|
| High | No wall-clock limit on the resize MILP, its refinements, tie-break or CP-SAT; a 30x50 correlated instance ran over 300 s while holding the shared CPU slot. | One deadline per allocation (20 s, `WOFFL_ALLOC_TIME_LIMIT_S`); on timeout a qualified incumbent is `feasible` with its gap. Pad sweeps share a run budget (120 s, `WOFFL_RUN_ALLOC_BUDGET_S`, 2 s floor per trial). |
| Med | `threads: 1` passed to HiGHS: after any default-thread HiGHS call in the process, every later MILP returned "HiGHS Status 0: Not Set" for the life of the process. | Option removed (network and choke MILPs); a subprocess test pins the order. |
| Med | Priced tie-break ran a second full MILP (4-7x the primary), changing 0 of 12 plans; at zero price, equal-oil plans used arbitrary extra water. | One solve with bounded lexicographic tie-break terms (least water at price 0, keep installed on ties); the bound is reported. |
| Med | Candidate building scanned each well's frame per row (quadratic). | Vectorized (0.41 s to 0.07 s for 30x51). |
| Low | Malformed installed identity aborted the whole batch; deduplicated installed/clean twins counted as "excluded". | Skipped with a reason; counts separated. |
| Low | CP-SAT requests on fractional water were "cross-checked" by re-solving the same MILP. | Skipped and said so. |

### CFP (`woffl/gui/cfp_moves.py`)

| Sev. | Defect (before) | Fix |
|---|---|---|
| High | A required new well that solves only above P0 produced a plan shutting in 35 of 37 wells (-2,967 BOPD) while a 2-change plan made +485. | Required wells are seeded (today plus each required size, alone and with each single offset) before the bounded search: +486.6 BOPD, 2 changes. Variant: +74 (39 changes) to +156 (5 changes). |
| Med | Required wells emptied the shut-in/bring-online board (11 rows, 0 shut-ins). | Boards stay relative to today with `meets_required`; the UI marks rows that break a requirement and keeps them out of "top moves". |
| Med | The pair budget covered only the first 1-3 bring-online options (a +337.6 BOPD pair never tried). | Round-robin across options; swaps spread across well pairs. |
| Low-Med | Offsets scored water at different pressures; every well's catalog included every other well's size (+25% simulations). | Same-pressure scoring; per-well catalogs. |
| Med (server) | One online well without a converged current pump at P0 aborted the whole CFP study; offline/future wells without a tracked pump were dropped. | The well is excluded with a note and coverage reads incomplete; idle candidates need no tracked pump. |

### Server and jobs (`server/`)

| Sev. | Defect (before) | Fix |
|---|---|---|
| Med | Tracker and saved-input pump identities were normalized differently (lowercase throat, size 7), silently blanking the pad's current-pump baseline, gains and stress test. | One `_pump_identity` normalizer; the model's own installed identity is used; wells without a baseline are named in the notes. |
| Low-Med | Job errors dropped per-well reasons ("no active wells..."), unknown donors read `('MPI-9O1')`, water cut printed at 2 decimals. | Reasons attached; "unknown well"; 3 decimals. |
| Low-Med | Empty or unknown pump grids, choke on S, pump counts a plant does not offer, future wells on another pad, unknown donors and pad PF above P0 started jobs that failed or degraded. | Rejected with a 422 and a reason. |
| Low | Unbounded job queue; a just-settled job could be pruned before its first poll. | Queue cap (429 with reason) and queue position; settle time recorded before status. |
| Low | E-Pad curve sheet ignored the run's amp limit; E-Pad results did not record their booster. | Amp limit passed through; `meta.e_pad` echoed and used by the chart. |
| Low | `_plain` stringified arrays/Series/Decimal. | JSON values. |

### Optimize page (`web/src/pages/optimize/`)

Fixed: 422s shown as "HTTP 422"; run settings shared across pad tabs; Cancel
stuck on "Cancelling..." after one cancel; water price unit 1,000x off;
E-Pad chart redrawing the form's booster under an old result; future wells
flagged "defaults - a guess" (they use the donor's saved fit); incomplete-and-
infeasible runs shown without a warning or with a projected gain; job ids lost
when leaving the tab during start (also match health and PF cost); silent
result expiry; CFP connector lines drifting after zoom; "Replace ? clean
reference"; CFP "in plan" matched by well only; "2,900 psi trip" where the
flag means the 2,880 psi control limit; a tab bar that overflowed narrow
windows.

## Speed

Local synthetic measurements; not hosted latency (Medium runs two workers).

| Where | Before | After |
|---|---|---|
| CFP `moves_summary` (reviewer's timing fleet) | 7.4 s | 0.5 s, same plan |
| CFP settle, 7,155 choice sets | 6.16 s | 0.27 s, 0 mismatches |
| CFP C-Pad well simulations per study | 8 | 1 |
| Allocation, 30x51, price 0.05 | 1.20 s | 0.06 s |
| Allocation, cloned wells, price 0.05 | 57.5 s | 0.61 s, same objective |
| Candidate building, 30x51 | 0.41 s | 0.07 s |
| Hard 30x50 allocation | >300 s | bounded (1 s limit: gap 2e-6) |
| Choke ladder pricing, 8 wells, per level | 256 solves | 8 solves |

**Pad header sweep.** Profiling 18-well S and I fixtures (real surveys,
synthetic Schrader inputs, 2 workers, the deployed pool and response cache):
a run is 7,830 physics solves in 15 trials; 93% of wall time is physics
(23.7 ms per S solve, 17.4 ms per I solve), 75% of that in the Beggs-Brill
return-flow march and its PVT. There are no exact repeat solves to cache and
almost no infeasible pumps, but only about 10 of 29 candidates per well are on
the oil/water frontier at any header. Two exact changes were made
(patch 51):

- **Bracket-dominance pruning.** Physics runs in bracketing order (ends and
  middle on the full grid); at an interior header a pump is skipped when one
  rival beats it by 1% on both oil and water at the solved headers on both
  sides. The leading trial is then completed on the full grid and
  re-allocated, and the sweep re-ranked until the winner is a full-grid trial,
  so every reported number comes from full grids. `WOFFL_SWEEP_PRUNE=0`
  restores the unpruned path; `meta.sweep_pruning` reports what was skipped.
- **One solve per identical twin.** An installed pump with no fitted
  coefficients is identical to its clean same-size replacement; it is solved
  once. Single-pump jobs (S settle, choke ladder, scenario evaluators) solve
  only the row they read.

| Pad run (18 wells, 2 workers) | Unpruned | Pruned | Speedup |
|---|---|---|---|
| S, maximize oil | 85.5 / 84.0 s | 58.8 / 63.3 s | 1.39x |
| I, maximize oil | 66.7 / 67.6 s | 43.7 / 44.1 s | 1.53x |
| S / I, water price 0.05 | 113.9 / 102.7; 105.2 s | 60.7 / 63.4; 63.7 s | 1.75x / 1.65x |
| S / I, CP-SAT | 143.9 / 106.5 s | 95.3 / 71.4 s | 1.51x / 1.49x |
| S / I, required well | 150.7 / 92.7 s | 111.1 / 67.5 s | 1.36x / 1.37x |
| Choke, I, 10 levels | 396 rows, 9.2 s | 198 rows, 7.1-7.7 s | 1.25x |

Solves per sweep fall from 7,830 to about 4,600. Every case gave the
identical plan with and without pruning (per-well pump and state, rates,
header, totals, winning frames and meta). The host was shared, so compare
ratios, not absolute times; a hosted S-Pad run is projected, not measured,
to fall from about 116 s to about 70 s. The 1% margin is an engineering rule,
not a proof: a pump skipped at a non-winning header could in principle have
mattered there. A response-surface sweep was 2.1x faster but changed the
S-Pad plan (-8.5 BOPD) because the top of the sweep is nearly flat, so it
was not adopted.

## Verification

- Full Python suite: **2,376 passed** (the September 12 record was 2,198);
  frontend: **52 passed** (was 45). `git diff --check` is clean.
- New regression files: `tests/test_optimize_review_2026_09_24.py` (server,
  32), `tests/test_choke_review_2026_09_24.py` (engine, 16),
  `tests/test_allocation_2026_09_24.py` (allocation, 21),
  `tests/test_cfp_moves_2026_09_24.py` (CFP, 15),
  `tests/test_sweep_prune_2026_09_24.py` (pruning and twins, 21) and
  `web/tests/optimizeRunSummary.test.mjs` (7).
  46 of the 47 server/engine regression tests fail against the committed code
  (checked in a HEAD worktree); the other is a compatibility check.
- Browser (built SPA, every `/api` call intercepted, no writes):
  `tools/check_capacity_optimization_ui.py` passed (4 mocked runs, no browser
  errors) and `tools/check_pad_decision_ui.py` passed (3 stress starts,
  cancel). Desktop and 960-pixel captures were inspected.
- TypeScript and the production build pass (existing large-chart-chunk
  warning).

## Changed contracts

- `OptimizeRunRequest` rejects: empty/unknown pump sizes on resize and CFP
  runs, choke on S, future wells outside the run's pads, CFP pad PF above P0.
  `/optimize/run` rejects pump counts the plant does not offer and unknown
  donors. Full queues return 429.
- Pad meta adds `sweep_complete`, `sweep_pruning`, `e_pad`; choke meta adds `today_basis`;
  allocation status adds `time_limit_reached`, `tie_break_bound`,
  `deduplicated_candidates`; CFP singles/pairs add `meets_required`; rows of
  planned wells add `donor`.
- `_model_at_forced_header` accepts per-well pressures and runs each well on
  its own held pump.
- Env: `WOFFL_ALLOC_TIME_LIMIT_S` (20), `WOFFL_RUN_ALLOC_BUDGET_S` (120),
  `WOFFL_MAX_QUEUED_JOBS` (6), `WOFFL_SWEEP_PRUNE` (on; `0` disables pruning).
- `_simulate_single_well` / `well_grids` accept an optional candidate subset;
  response-cache keys include it.
- Upstream register: patches 47-51 (see [upstream_sync](upstream_sync.md)).

## Not done

- Field qualification is unchanged: these fixes change search, allocation,
  reporting and usability, not the physics or any fitted input.
- The Cost-of-PF panel still auto-starts a job on S/I tab open; with one job
  slot, a Run clicked in the first ~30 s queues behind it (now shown).
- "Today" still differs between the pad run, match health and the PF panel.
- Hydration is not shared across panels.
- The marginal-oil column still reads 0 for a selected pump that is not the
  best throat for its nozzle: the library maps the missing derivative to 0.0
  (often a "keep installed" choice). Treat 0 there as "not computed".

## Addendum: E-Pad booster calibrated to the E-41 rate test

Later the same day, the E-Pad booster rate test (E-41 surface kit) was applied
to the E-Pad plant and screens. Details and numbers are in
[the E-Pad pump README](../woffl/jp_data/E_Pad_Pumps/README_E-Pad_Booster_Pumps.md#0-field-rate-test-e-41-surface-kit);
the raw points are in the meta `field_test`, and
`e_pad_booster.field_calibration` derives the calibration from them.

- The model's motor current was about ten times low (0.1435 A/BHP carried from
  I-Pad's 4,160 V motor): 89 A where the drive read 889 A. It is now
  1.4281 A/BHP with the drive's 889 A current limit enforced by default.
- The default E-Pad plant overstated PF capacity at 3,400 psi by about
  2,900 BWPD (32,400 catalog). It now delivers the tested 29,491 BWPD there,
  current-limited at 53.1 Hz, and about 27,100 BWPD at the 3,500 psi cap.
- Suction defaults to 2,704 psi (measured at the limit; 2,725 at baseline).
- The installed unit carries a head condition of 0.766 and the upper range it
  ran at (33,323 BPD at 60 Hz); the SN35000 alternative and an as-new
  replacement (`field_calibrated=False`) stay on catalog curves. The booster
  comparison screen compares builds as new on the E-41 motor.
- One test at one speed: values away from 3,400 psi are the catalog shape
  through that point. The header cap is still I-Pad's 3,500 psi.
- Tests: `tests/test_e_pad_field_2026_09_24.py` (8); frontier mechanics tests
  now run on the explicit catalog plant. Full suite after the addendum:
  2,384 passed; frontend 52; TypeScript and build pass.
