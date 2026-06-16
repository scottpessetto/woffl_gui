# S-Pad Joint Oil + Power-Fluid Auto-Match

**Built overnight 2026-06-15 → 16. All verified OFFLINE (synthetic + 543-test
suite). Real-well validation is the first thing to do at 7am — see "Verify at
7am" below. Nothing committed yet.**

## What you asked for

> "find a way to get a match on oil and power fluid by altering IPR and JP params
> for each S-Pad well … a button the user can push on each well. If you cant get
> a match … research deeply why and work a solution."

Done — plus the deeper root-cause fix for *"the pump is in the well and working,
but not converging in the model."*

## The button

On every well in the Solver (home **and** the S-Pad review — same reused
component), under *Model vs Actual*, there's now:

> **🎯 Auto-match oil + PF**

It appears whenever the selected test has both a known **oil** rate and **power
fluid** (lift-water) rate. One click:

1. Reads the test's oil + PF (and BHP, if the test has a gauge).
2. Searches the **IPR productivity, PF surface pressure, and friction coefs
   (ken/kth/kdi)** for the combination that makes the **installed pump** model
   that test's oil **and** PF at once (and BHP, when present, as a 3rd target).
3. Seeds the sidebar with the consistent solution (`qwf`, `pwf`, `ResP`,
   `ppf_surf`, `ken`, `kth`, `kdi`) — every downstream view (Batch, PF Range,
   optimizer) then reads it.
4. If it **can't** hit both targets, it does **not** silently seed a bad point —
   it tells you the physical reason (see Diagnostics).

For a gaugeless well (S-03, S-05) the flowing BHP falls out of the match and is
seeded as `pwf` — exactly your "starting IPR = the solution that lets the test's
jet pump reproduce the test" ask. (The older oil-only "🎯 Match test oil rate"
button is still there for the simplest gaugeless case.)

**No JP history?** The button still works (e.g. **S-67**: tests but no install
records). When the installed pump can't be looked up it models the **sidebar**
pump against the test (the caption says "sidebar pump" instead of "installed"),
so a well missing install history isn't locked out of the match.

## The diagnostics (when it can't match)

Instead of guessing, the matcher **re-solves at the pump's physical ceilings**
(max IPR, max PF pressure) and reports the real limit:

- **PF-limited** — "the nozzle delivers at most X BPD even at 5,500 psi vs the
  target — nozzle too small, or the real delivered PF pressure is higher than
  modeled." → check pump size / header pressure.
- **Pump-capacity-limited** — "with unlimited reservoir AND max PF pressure the
  12B pump still tops out at X BOPD < target — this pump physically can't make
  the oil; fit a bigger nozzle/throat."
- **PF-pressure-limited on oil** — reaches the oil target only at higher PF
  pressure → raise the delivered header pressure.
- **PF too high at the floor**, **sonic-choked**, etc.

## The deeper fix — "works in reality, not in the model" (LIBRARY)

While stress-testing I found **29% of a realistic S-Pad parameter grid failed to
solve** with `ConvergenceError: throat mixture did not converge`. Researching it:

- **Not** a GOR problem, **not** secant oscillation.
- The outer `jetpump_solver` computes the lower suction-bracket residual right at
  `psu_min`, where the inner throat-mixture solve is infeasible for marginal
  pumps (small throat ratio + high water cut). It then **aborted the whole
  solve** — even though **29 of 30 suction pressures in range are feasible and
  the discharge residual clearly crosses zero** (a real, flowing solution).

**Fix** (`woffl/assembly/solopump.py::_residual_walk_inward`): when a bracket
endpoint is infeasible, walk the suction inward (probe fractions dense near the
endpoint, extending to ~95% of the span — some pumps are feasible only in a thin
band against reservoir pressure, e.g. a 13B at 99% WC) to the nearest feasible
point, so the outer search keeps a valid bracket. Endpoint-feasible cases are
bit-identical (no regression). Added a defensive Brent fallback in the
throat-mixture solver too (`woffl/flow/jetflow.py::_throat_discharge_bracketed`).

**Characterized across 6,048 combos** (nozzles 9–18 × throats X–E × WC 0.5–0.99 ×
GOR 150–2000 × pres 1200–2600 × both circulation directions): the realistic grid
went **51/72 → 72/72**; in the extreme grid, **every case with a real solution
(residual crosses zero) now solves**. The only remaining `ConvergenceError`s are
genuinely infeasible geometries — tiny throat (X/A) + 99% water cut demanding
2000–3000 BOPD — which physically cannot produce and correctly fail.

⚠️ **These two are LIBRARY changes → need an upstream PR to `kwellis/woffl`.**
Regression guard: `tests/test_asm_solopump.py::TestMarginalConvergence`.

## Files

| File | What |
|---|---|
| `woffl/gui/joint_match.py` | The matcher + diagnostics (new) |
| `woffl/gui/tabs/jetpump_solver.py` | `_render_joint_automatch` button + message replay |
| `woffl/gui/sidebar.py` | `SEED_BOUNDS` gained ken/kth/kdi |
| `woffl/assembly/solopump.py` | `_residual_walk_inward` (LIBRARY) |
| `woffl/flow/jetflow.py` | `_throat_discharge_bracketed` (LIBRARY) |
| `tests/test_joint_match.py` | 6 pytest guards (round-trip, forward, +BHP, marginal, diagnostics) |
| `tests/test_joint_match_sweep.py` | 72-combo sweep (manual) |
| `tests/test_asm_solopump.py` | `TestMarginalConvergence` regression guard |

## Validation done offline

- Round-trip recovery: **matched to 0.0%** on oil + PF across 72 realistic combos.
- Edge cases matched: forward-circulating, gauged (3-target), WC 0.97, the
  marginal 12B/94%-WC regime, low-rate wells.
- Diagnostics correctly flag unreachable targets as `partial` (no fake matches).
- ~0.5 s/well (interactive-fast).
- **Full suite: 543 passed** (534 baseline + 3 solopump + 6 joint-match).

## Verify at 7am (needs live Databricks)

1. Open S-Pad review, pick a well with a good recent test (e.g. one that was
   busting in the pre-flight). Click **🎯 Auto-match oil + PF**. Confirm the hero
   oil + PF now sit on the test, and the seeded sidebar values look sane.
2. Try a **gaugeless** well (S-03 / S-05): confirm it infers a sane BHP and the
   IPR reproduces the test pump's rates.
3. Try the well(s) that **wouldn't converge** before (S-204, S-23, S-17). They
   should now solve; if one still won't, the diagnostic will say why — send me
   the message.
4. Sanity-check the seeded **PF pressure** against the known header; if you see
   the ⚠️ "PF pressure moved …" note, that well's modeled PF pressure differs
   from reality (worth a look).

## Known limits / next

- `pwf` seed is clamped to (100, 2500) psi — fine for S-Pad (BHP < 2000), but a
  higher-pressure pad would need the bound widened.
- PF surface pressure floats freely to get the best match; a "pin PF pressure
  (match via friction only)" mode is a natural follow-up for when the header
  pressure is known exactly.
- A **batch "auto-match all wells"** button (one click → match-health table for
  the whole pad) is the next workflow win.

## Batch button — recipe for the live build (do this together with real data)

The per-well matcher is the tested core; the batch button is a thin loop over it.
It needs each well's **measured test (oil + PF)**, which lives in Databricks
(`get_well_tests_for_well`) — that's why it can't be built/verified blind. The
review store (`well_review_store`) already carries everything else. Recipe:

1. For each pad well (saved store entries via `store_for(pad)`, or all
   `_pad_real_wells(pad)` for an un-reviewed pre-pass):
   - **Geometry/conditions** from the store entry (or, for un-reviewed wells,
     from `load_well_characteristics` + the IPR auto-populate): `review_nozzle`/
     `review_throat`, `res_pres`, `form_temp`, `jpump_tvd`, tubing/casing dims,
     `form_wc`, `form_gor`, `surf_pres`, `ppf_surf_well`, `ken/kth/kdi_well`,
     `field_model`, `oil_api/gas_sg/wat_sg/bubble_point`.
   - **Targets**: `oil, pf = _recent_test_rates(well)` (median of recent tests;
     PF = `lift_wat`). Skip the well if either is missing.
   - Build `wellbore = create_pipes(...)`,
     `well_profile = create_well_profile_from_survey(well, jpump_tvd, field_model)`.
2. Call `joint_match(...)` **wrapped in `try/except`** (one bad well must not
   break the table) — collect `{well, status, oil_err_pct, pf_err_pct, ppf_surf,
   diagnostic}`.
3. Render a health table: ✓ matched / ⚠ partial (+ diagnostic) / ✗ error, sorted
   worst-first, so the engineer hand-reviews only the problem wells. Optional
   "apply" per row that seeds that well's store entry with the matched params.

Keep it **beta + fully wrapped** so it can never blank the review page. The
per-well loop itself is already validated (the 72-combo sweep IS a batch); only
the store→kwargs + test-fetch glue is new, so verify it live on 2–3 wells first.
