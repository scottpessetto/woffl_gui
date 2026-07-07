# Upstream Sync — Local Library Patches (don't lose these)

This repo (`github.com/scottpessetto/woffl_gui`) is a **fork** of the upstream
`woffl` library (`github.com/kwellis/woffl`). The GUI (`woffl/gui/`) is ours
alone, but the four **library** packages — `woffl/pvt/`, `woffl/geometry/`,
`woffl/flow/`, `woffl/assembly/` — are shared with upstream and published to PyPI.

We carry a few **local patches inside the library** that are **not yet
upstreamed**. When the upstream owner ships changes and we sync them in, those
patches must survive the merge. This file is the authoritative record so they
can't be silently overwritten — and the regression tests below are the tripwire
if one ever is.

> Every patch site is tagged `# [LIBRARY change -> upstream PR to kwellis/woffl]`
> in the code. `grep -rn "upstream PR" woffl/` finds them all.

---

## The load-bearing local patches (pending upstream PR)

### 1. `woffl/assembly/solopump.py` — marginal-well convergence
New helpers `_secant_solve`, `_bisection_solve`, `_residual_walk_inward`, plus the
`jetpump_solver` body that orchestrates them. Two distinct fixes:

- **Secant → bisection fallback.** When the primary secant on the discharge
  residual stalls, re-seed from the well's measured flowing BHP (`ipr_su.pwf`),
  then fall back to a robust bisection over the bracketed root.
- **Suction walk-inward.** When a suction bracket endpoint (`psu_min` / `psu_max`)
  is itself infeasible — the inner throat solve has no solution *at that exact
  point* — walk the suction inward to the nearest feasible point so the outer
  search keeps a valid, bracketed root.

**Why:** marginal wells (small throat ratio + high water cut) physically flow,
but the legacy solver aborted at the infeasible endpoint — *"the pump is in the
well and working, but won't converge in the model."* Characterized across 6,048
combos: every well whose discharge residual crosses zero now solves; only
physically-impossible geometries fail. **Endpoint-feasible / already-converging
solves are bit-identical** (additive change only — no behavior change where it
already worked).

**Guarded by:** `tests/test_asm_solopump.py::TestMarginalConvergence`
(`test_marginal_pump_converges`, `test_marginal_endpoint_is_actually_infeasible`,
`test_several_marginal_pumps_converge`, `test_thin_upper_band_feasibility_converges`)
plus the existing solopump smoke tests.

### 2. `woffl/flow/jetflow.py` — throat-mixture discharge bracketing
New helper `_throat_discharge_bracketed`. `throat_discharge` now calls it where it
used to `raise ConvergenceError` at the 15-iteration secant cap: it Brent-brackets
the momentum-balance root and only raises when there is genuinely **no sign change**
(no solution). Defensive robustness for the compressible throat mixture near the
bubble point; already-converging solves are untouched.

**Guarded by:** the marginal-convergence tests above + the full solver/regression
suite (it rescues secant stalls around a real root; the walk-inward in #1 handles
infeasible endpoints).

### 3. `woffl/assembly/network_optimizer.py` — process-pool serial fallback
`run_all_batch_simulations` wraps the `ProcessPoolExecutor` path in a
`BrokenProcessPool` guard: if a worker dies abruptly (an OOM kill on the 2-vCPU/
6 GB Databricks app, or spawn/resource flakiness with `WOFFL_MAX_WORKERS=10`
locally), it falls back to **serial in-process execution** so the match check,
optimizer, and scenario comparator complete instead of failing. Additive — the
normal serial/parallel paths are unchanged; the fallback only fires on a broken
pool. (⚠ Verify whether `network_optimizer.py` actually ships in upstream
`kwellis/woffl` — it may be a fork-added multi-well optimizer, in which case this
is a local-only change with no merge-clobber risk and no PR needed.)

### 4. `woffl/pvt/resmix.py` — zero-oil (100% water cut) guard
`_static_insitu_volm_flow` computes `qtot = qoil / yoil` to recover the total
insitu flow from the oil rate. At **100% water cut** the oil volume fraction
`yoil` is exactly zero, so this raised a bare **`ZeroDivisionError`** — which is
**not** a `ValueError`, so it escaped every `except ValueError` solver handler
(including `run_jetpump_solver`) and crashed the Streamlit page. (Observed
running **S-03**, a ~100% WC well.) The guard raises a typed **`ValueError`**
with a clear message instead, so the existing GUI / batch handlers catch it and
show a normal solver-error box. Additive — the `yoil > 0` path is bit-identical.

**Guarded by:** `tests/test_pvt_resmix.py::test_full_watercut_raises_valueerror`
(+ `test_near_full_watercut_still_solves` confirms the guard is specific to the
degenerate case). Goes red if a sync drops the guard — `ZeroDivisionError` is
not a `ValueError`, so `pytest.raises(ValueError)` fails.

### 5. Water-pump (dewatering) mode — opt-in 100%-water solve
Spans three shared files; **additive and gated by an explicit
`ResMix(model_as_water=...)` flag (default `False`)**, so the oil path is
bit-identical and the #4 guard still raises when the flag is off. Reuses the IPR
as the well's water deliverability. Lets a 100%-water (watered-out / source) well
be modeled to see what suction + power fluid it takes to flow it; water is
~incompressible, so the throat stays subsonic (won't choke).

- `woffl/pvt/resmix.py` — `model_as_water` ctor flag + `_static_insitu_volm_flow_water`
  (anchors total insitu flow on **water** when there's no oil); `insitu_volm_flow`
  branches to it.
- `woffl/flow/jetflow.py` — `throat_wc` 100%-WC branch (water rate is the anchor,
  no ÷(1−wc)); `jetpump_base_calcs` propagates the flag into its internal `prop_tm`.
- `woffl/assembly/solopump.py` — `discharge_residual` propagates the flag into
  **its** `prop_tm` (a **second** throat-mixture site — missing it makes the solve
  raise the #4 guard mid-solve; the e2e test below catches that).

GUI wiring (sidebar toggle, params, dedicated result block) is under `woffl/gui/`
and is **not** upstreamed. Full design: `docs/water_pump_mode_plan.md`.

**Guarded by:** `tests/test_asm_solopump.py::TestWaterPumpMode`
(`test_water_pump_solve_converges`, `test_water_mode_off_still_raises_at_full_wc`),
`tests/test_asm_solopump.py::test_throat_wc_water_branch`, and
`tests/test_pvt_resmix.py::test_water_mode_anchors_on_water`. The
"off still raises" test is the tripwire that the oil path / #4 guard survived.

### 6. `woffl/assembly/solopump.py` — `_secant_solve` returns real rates on a skipped loop
The primary `jetpump_solver` call passes `seed_pair=(psu_min, psu_max)`, whose
residuals come from `res_lookup`, so `discharge_residual` is never called for the
seeds. If the secant `while` loop then never runs — a **thin feasible band** where
the bracket already satisfies `psu_diff` and `res_tol` — the returned
`qoil_std/fwat_bwpd/qnz_bwpd/mach_te` stayed at their `0.0` initializers: a real
flowing well reported as **0 BOPD while returning normally** (no `ConvergenceError`,
so the bisection/walk-inward fallbacks never fired). The patch tracks which suction
the cached rates correspond to (`rates_at`) and re-evaluates `discharge_residual`
at the returned `psu` only when they don't already match — so the normal converging
path is **bit-identical**.

**Guarded by:** `tests/test_asm_solopump.py::TestSecantSolveRatesPopulated`
(stubs `discharge_residual`, drives a degenerate already-converged bracket, asserts
real rates + exactly one final eval). Goes red if the final evaluation is dropped.

### 7. `woffl/assembly/batchpump.py` — `update_press("reservoir")` + idempotent `process_results`
Two additive fixes:
- **`update_press` dotted path.** `setattr(self, "ipr_su.pres", psig)` does not
  traverse a dotted path — it created a junk attribute literally named
  `"ipr_su.pres"` and left the real `self.ipr_su.pres` untouched, so
  `update_press("reservoir", …)` was a **silent no-op** (ran at the original
  reservoir pressure). Now walks the path; flat keys (`wellhead`/`powerfluid`) are
  unchanged.
- **`process_results` idempotency.** The `motwr`/`molwr` merge wasn't idempotent:
  a second call merged those names into a df that already had them, so pandas
  suffixed them `_x`/`_y` and the plain columns vanished
  (`get_pump_performance`'s `row.get("molwr")` then silently returned `None`).
  Now drops any prior `motwr`/`molwr` before re-merging.

**Guarded by:** `tests/test_asm_batchpump.py::TestUpdatePress` and
`::TestProcessResultsIdempotent`.

### 8. `woffl/pvt/blackoil.py` — McCain below-bubble compressibility takes Rsb
`_compute_compressibility` fed `compressibility_mccain_below` `rs = gas_solubility()`,
which **below the bubble point** is `Rs` at the *current* pressure. McCain-Rollins-
Villena (1988) Eq. 5 is defined with **Rsb** — the solution GOR *at the bubble
point* (a fixed property of the oil). Passing `Rs(p) << Rsb` systematically
understated sub-bubble oil compressibility. The patch evaluates solubility at the
bubble point explicitly (`solubility_kartoatmodjo(self.pbp, …)`). This is a real
physics change: it nudges `cmix` (mixture speed of sound) and hence `mach_te`, so
the `batch_test.py` 9X/12B reference `mach_te` values were re-baselined (the
operating point — oil/water/psu — is unchanged).

**Guarded by:** `tests/test_pvt_blackoil.py::test_oil_compressibility_below`
(asserts ~2.40e-4 psi⁻¹; reverts to ~2.16e-4 if the Rsb input is lost).

### 9. `woffl/flow/twophase.py` — Beggs-Brill L3 exponent
`beggs_flow_pattern` used `l3 = 0.1 * nslh**-1.468`; the canonical Beggs-Brill L3
boundary is `0.10 * λL**-1.4516` (the surrounding `l2 ≈ -2.4684` and `l4 = -6.738`
match canonical — `-1.468` looks like a transcription slip copying the `.468` from
L2). L3 separates the intermittent / distributed / transition regimes, so a wrong
exponent picks the wrong holdup correlation near that boundary → wrong slip holdup
→ wrong static ΔP in the two-phase outflow path.

**Guarded by:** `tests/test_multiphase.py::test_beggs_l3_exponent_is_canonical`
(at `nslh=0.5`, `froude=0.275` classifies as `intermittent` with `-1.4516` but
`transition` with the old `-1.468`).

### 10. Water-pump mode — outflow anchored on TOTAL water (P1-1, 2026-07-02)
Extends #5. In water mode the `prop_tm` anchor is a **water** rate (see
`ResMix.insitu_volm_flow`'s water branch), and `wc_tm = 1.0` carries no
information about the power fluid — so anchoring the diffuser discharge and the
tubing traverse on `qoil_std` (= formation water only) dropped the nozzle
volume: a well moving 300 BWPD formation + 2,500 BWPD PF was modeled at 300
BWPD in the tubing, inconsistent with the throat momentum balance (which DOES
include the nozzle mass flow). New helper `jetflow._throat_mixture_anchor`
returns `qoil_std + qnz_bwpd` only when `water_mode and wc_tm >= 1.0`; both
call sites (`jetflow.jetpump_base_calcs`, `solopump.discharge_residual` — the
diffuser AND `production_top_down_press`) use it. **Oil-path solves are
bit-identical** (anchor unchanged unless the water branch is engaged). At the
reference water fixture the operating point moves psu 912 → 747 psig and
formation water 1,142 → 1,316 BWPD — the broken model materially understated
the well.

**Guarded by:** `tests/test_asm_solopump.py::test_throat_mixture_anchor`,
`TestWaterPumpMode::test_water_solve_outflow_includes_power_fluid` (pinned
post-fix values; the legacy anchor lands ~22% off) and
`TestWaterPumpMode::test_anchor_is_live_in_the_solve_path` (monkeypatches the
anchor back to formation-only and asserts the solve changes — proves the
helper is on the live path, not dead code).

### 11. `_throat_discharge_bracketed` — take the physical HIGH root (P1-3, 2026-07-02)
Extends #2. The momentum-balance residual generically has **two** roots (it
falls to −∞ at both ends with a positive hump between): the low root is the
non-physical/choked branch, the high root is the working discharge the secant
fast path (seeded at 2–3× pte) converges to. The fallback's original upward
scan from 15 psig locked onto the **low** root — an understated ptm/pdi, i.e. a
false "pump can't lift" on exactly the marginal wells the fallback exists to
save. Now scans **downward** from the top of the range (expanding the range
first when `bal(hi) > 0`, meaning the scan started inside the hump), so the
first bracketed sign change is the physical high root. Secant fast path
untouched; already-converging solves bit-identical.

**Guarded by:** `tests/test_asm_solopump.py::test_bracketed_throat_discharge_takes_physical_high_root`
(synthetic two-root residual with roots at 100 and 2,000; asserts 2,000 —
the upward scan returns 100 and goes red).

### 12. Low-severity robustness guards (Low-tier review sweep)
A batch of small additive guards across the shared library — all bit-identical on
the normal path, each tagged in-code with `upstream PR`:

- **`solopump.py` `_bisection_solve`** — on a `ThroatEntryNoSolution` probe inside
  the bracket, mark `res_mid` negative (too-low side) instead of leaving it stale,
  so the next narrowing direction is correct. (Covered by `TestMarginalConvergence`.)
- **`batchpump.py`** — `form_wor`/`totl_wor` guard `qoil_std == 0` (all-water solve
  → NaN, not inf/ZeroDivisionError that mislabels a valid 0-oil pump as failed);
  `gradient_back` guards equal successive water rates (zero denominator → NaN).
  Guarded by `tests/test_asm_batchpump.py::test_gradient_back_equal_water_is_nan`.
- **`jetplot.py`** (diagnostic plot path) — clamp `pidx` from `searchsorted(...,1)`
  so an all-subsonic window can't IndexError; floor `throat_entry_book`'s pressure
  sweep below `psu` so a low-pressure well (`psu<=200`) doesn't sweep the wrong
  direction. `psu>200` is bit-identical.
- **`blackoil.py`** — replace the dead `np.errstate(invalid="raise")` (a no-op on
  Python-float math) in `solubility_kartoatmodjo` with a real `pabs<=0` guard;
  make the BlackOil range validations INCLUSIVE to match the docstrings. Guarded by
  `tests/test_pvt_blackoil.py::{test_validation_bounds_inclusive,
  test_validation_rejects_out_of_range, test_solubility_negative_abs_pressure_raises}`.
- **`formgas.py` / `formwat.py`** — inclusive SG range validations (match docstrings).
  Guarded by `test_pvt_formgas.py::test_gas_sg_bounds_inclusive` and
  `test_pvt_formwater.py::test_wat_sg_bounds_inclusive`.

### 13. `woffl/assembly/databricks_client.py` — connection retry + OAuth lock scope (P2-6, 2026-07-06)
Two additive robustness fixes in `_query_via_connector` / `_oauth_token`:

- **First-attempt connect failure now retries.** `_new_connection()` was called
  OUTSIDE the try/except in `_query_via_connector`'s retry loop, so a failure on
  the very first connection attempt (attempt 0) raised immediately and never
  reached attempt 2. It's now called INSIDE the try, so a first-connect failure
  (warehouse cold-start, transient network blip) takes the same
  reset-conn/force-token-refresh/retry path a mid-query failure already did.
  Already-successful first connections are unaffected.
- **`_oauth_token` no longer holds `_TOKEN_LOCK` across the ~30 s HTTP token
  fetch.** It now: checks the cache under the lock → if stale, releases the
  lock → does the HTTP fetch unlocked → re-acquires the lock and keeps
  whichever token is newer (another thread may have refreshed meanwhile)
  before writing. Previously every thread serialized behind one thread's
  network call (e.g. the app's concurrent startup warm queries in
  `CLAUDE.md`'s "databricks_client design" note). A cache hit (common case)
  is unaffected — it still returns under the lock without any fetch.

**Guarded by:** `tests/test_databricks_client.py::TestQueryViaConnectorRetriesFirstAttempt`
(`test_first_connect_failure_retries_and_succeeds`,
`test_first_connect_failure_forces_token_refresh_and_clears_cache`,
`test_two_connect_failures_raise_the_last_error`) and
`::TestOauthTokenReleasesLockDuringFetch`
(`test_lock_is_not_held_during_the_http_call` — event-gated, not sleep-based —
`test_returns_fresh_token_and_populates_cache`).

### 14. P1-8/P1-9/P1-10 — fallback exception widening, zero-flow guard, z-factor clamp (2026-07-06)
Three additive robustness fixes from the 2026-07-01 review, all bit-identical on
the normal/in-range path:

- **`woffl/assembly/solopump.py` (P1-8)** — `_residual_walk_inward` previously
  only caught `ConvergenceError`; `_bisection_solve`'s midpoint retry previously
  only caught `ThroatEntryNoSolution`. Neither walked past a **bare
  `JetPumpError`** raised by `jetflow.nozzle_velocity` when `pni <= pte` (nozzle
  inlet pressure below throat entry pressure) — exactly the "works in the well,
  not in the model" failure class #1 exists to rescue, just from a different
  inner exception type. `_residual_walk_inward` now catches the `JetPumpError`
  family broadly but still explicitly re-raises `ThroatEntryNoSolution`
  unchanged (it drives the GUI's GOR auto-recovery — must not be swallowed);
  `_bisection_solve`'s midpoint retry is broadened from
  `except ThroatEntryNoSolution` to `except JetPumpError` (now also covers
  `ConvergenceError` and the bare base class), keeping the same
  too-low-side/`res_mid` narrowing logic.
- **`woffl/flow/singlephase.py` (P1-9)** — `ffactor_darcy` computed
  `64 / reynolds` with no guard; `reynolds == 0` (zero flow, e.g. a shut-in
  segment) raised a bare, untyped `ZeroDivisionError`. Every caller pairs `ff`
  with a velocity term (`dp_fric ~ ff * vel**2`), and `reynolds == 0` implies
  `vel == 0` too, so the physically correct friction contribution in that limit
  is zero. Now returns `0.0` for `reynolds <= 0` before the laminar branch;
  `reynolds > 0` path unchanged.
- **`woffl/pvt/formgas.py` (P1-10)** — `_zfactor_grad_school`'s cubic
  correlation is unguarded outside its documented validity range (very high
  `ppr` / very low `tpr`) and could return `z <= 0` or an implausibly large `z`,
  silently poisoning `_compute_density` (division by `zfactor`) and
  `ResMix.cmix`'s `math.sqrt(...)` downstream with a domain error or a
  negative/huge density. The raw output is now clamped to
  `[FormGas._ZFACTOR_MIN, FormGas._ZFACTOR_MAX] = [0.05, 3.0]` — a no-op for
  every realistic natural-gas z-factor (well within that band).

**Guarded by:** `tests/test_asm_solopump.py::TestFallbackWalksPastBareJetPumpError`
(`test_walk_inward_passes_bare_jetpumperror`,
`test_walk_inward_still_reraises_throat_entry_no_solution`,
`test_bisection_midpoint_passes_bare_jetpumperror`);
`tests/test_multiphase.py::test_ffactor_darcy_zero_reynolds_does_not_raise`
(+ `test_ffactor_darcy_in_range_unchanged` precondition guard);
`tests/test_pvt_formgas.py::test_zfactor_gradschool_clamped_outside_correlation_range`,
`::test_zfactor_property_clamped_and_finite`
(+ `::test_zfactor_gradschool_in_range_unchanged` precondition guard).

### 15. `woffl/flow/jetflow.py` + `woffl/flow/jetplot.py` — solve the IPR on Vogel, not straight-line PI (restores ee3886e, 2026-07-06)
Commit `ee3886e` (2026-03-11, "change woffl to solve on ipr not straightline PI.
t'isnt right") flipped the three throat-entry IPR evaluations —
`jetflow.throat_entry_zero_tde`, `jetflow.throat_entry_mach_one`, and
`jetplot.throat_entry_book` — from `ipr_su.oil_flow(psu, method="pidx")` to
`method="vogel"`, and re-pinned `tests/batch_test.py` accordingly. A later
upstream sync (`0f147fb`, "incorporate woffl 2.0") **silently reverted all
three sites back to `"pidx"`** — this is exactly the clobber scenario this
document's sync protocol exists to catch, and it slipped through because the
`batch_test.py` pins were loose enough (1% rel tolerance) that only 3 of the 4
reference tests actually went red, and the discrepancy was small enough to
read as ordinary re-baselining noise from unrelated solver/outflow patches.
**Upstream `kwellis/woffl` still uses `"pidx"`** — this divergence is
intentional and permanent, not a bug to reconcile away on the next sync.

Both curves pass through `(pwf, qwf)` and `(pres, 0)`; strictly BETWEEN those
two points the Vogel curve sits ABOVE the straight-line PI chord (more oil at
a given psu), and BELOW `pwf` it sits below the chord (less oil/water at a
given psu). At the E-41 batch fixture (`pwf=1049`, `pres=1400`), all four
reference `psu_solv` values (1117–1323 psig) land strictly between `pwf` and
`pres`, so `qoil_std`/`totl_wat` moved up; re-pinned in `tests/batch_test.py`
(9X, 9D, 16E — 12B stayed within the existing 1% tolerance).
`tests/test_asm_solopump.py::TestWaterPumpMode::test_water_solve_outflow_includes_power_fluid`
solves BELOW its water-IPR's anchor `pwf` (`psu≈698` vs `pwf=1000`), so
`fwat` moved down — also re-pinned.

**Guarded by:** `tests/test_asm_solopump.py::TestSolverUsesVogelIPR`
(`test_solver_qoil_matches_vogel_not_pidx` — a single cheap E-41 solve
asserting `qoil_std == ipr.oil_flow(psu, "vogel")` AND
`qoil_std != ipr.oil_flow(psu, "pidx")` at a psu strictly between `pwf` and
`pres`). Goes red immediately if any of the three call sites reverts to
`method="pidx"`.

---

## Dead-code deletions from `woffl/assembly/` (R-10, 2026-07-06)

Not patches — these are GUI-fork-only removals of confirmed-zero-caller code
inside the shared library dirs. Recorded here so a future upstream merge
doesn't silently resurrect them (if `kwellis/woffl` still carries these
symbols, decide deliberately whether to re-delete the merged-in copy rather
than assume the merge "restored" something we need):

- `woffl/assembly/optimization_analyzer.py` — entire file deleted (zero
  callers anywhere in the tree; `compare_scenarios` and friends were never
  imported outside their own module).
- `woffl/assembly/network_optimizer.py` — deleted `validate_allocation`
  (dead + superseded), `create_well_template_csv`, `load_wells_from_csv`
  (zero callers incl. tests). Deleted `get_calibrated_results` (dead
  duplicate of `calibration.apply_calibration`, which `step4_results.py`
  actually uses). **Kept** `validate_well_config` (still unwired — see B-2 —
  but not dead, it's exercised by `tests/test_network_optimizer.py`).
- `woffl/assembly/network.py` — deleted the `WellNetwork` class (dead;
  `woffl/assembly/__init__.py`'s `from .network import WellNetwork` was its
  only "caller", also removed). **Kept** the module-level `optimize_jet_pumps`
  function + `SCALE` constant — these are live (used by
  `optimization_algorithms.mckp_optimization` and covered by
  `tests/test_asm_network.py`).
- `woffl/assembly/well_test_processor.py` — deleted `merge_tests_with_bhp`
  (zero callers outside its own test file; the app never merges test data
  with BHP this way anymore — `vw_bhp_daily_clean` / live PF replaced this
  path). **Kept** the `WellTestProcessor` class itself (out of R-10's
  explicit scope, though note under NOTES in the W11 handoff: it too has no
  callers outside its own test file — worth a follow-up look).
- Legacy BHP chain in `woffl/assembly/databricks_client.py`
  (`load_tag_dict`, `query_bhp_for_well_tests`) and the `bhp_dict.csv` data
  file it reads — **NOT deleted this pass** (file was excluded from this
  work item to avoid clobbering concurrent edits); still dead per the same
  review finding, deferred to a follow-up pass.
  **Follow-up (2026-07-06): deleted** — both functions, their dedicated test
  classes (`TestLoadTagDict`, `TestQueryBhpForWellTests`), and
  `woffl/jp_data/bhp_dict.csv` are gone; zero callers confirmed repo-wide
  (`vw_bhp_tags` / `vw_bhp_daily_clean` fully replaced this path). **Kept**
  `get_tags_for_wells` (out of this item's explicit scope) even though it's
  now orphaned with `query_bhp_for_well_tests` gone — worth a follow-up look.

---

## NOT upstream — safe to change freely
The joint oil + power-fluid auto-match (`woffl/gui/joint_match.py`), its per-well
**🎯 Auto-match oil + PF** button, the batch core, and everything else under
`woffl/gui/` are **GUI** — ours, never upstreamed. The bulk of the auto-match work
lives there; only the solver + PVT files above (`solopump.py`, `jetflow.py`,
`resmix.py`) touch the shared library.

---

## Sync protocol — run this EVERY time you pull upstream

1. **Before merging,** know these patches exist (this file).
2. **Merge upstream into a branch** (never straight onto a release branch). Watch
   for conflicts specifically in `solopump.py` and `jetflow.py`.
3. **After merging, run the full suite:**
   ```bash
   WOFFL_MAX_WORKERS=1 PYTHONPATH=. ./venv/Scripts/python.exe -m pytest tests/ -q
   ```
   If `TestMarginalConvergence` (or any solopump test) **goes red**, an upstream
   merge dropped or altered a local solver fix — re-apply it from this file / git
   history before shipping. **The tests are the safety net: a silently-lost patch
   turns red.** (Baseline: the suite is fully green — 563 tests as of 2026-06-29.)
4. **See the full divergence set** any time:
   ```bash
   git diff <upstream-remote>/main -- woffl/pvt woffl/geometry woffl/flow woffl/assembly
   ```
   Run from the repo root. Anything that shows up there is a local library
   divergence; confirm it's either listed here or genuinely intended.

---

## The real fix
Get these two patches **merged upstream** — your buddy owns `kwellis/woffl`. Once
they land there, the divergence (and this whole risk) disappears: the next sync
just brings them back as upstream code. This file + the regression tests give him
everything he needs to review and accept them. Until then, treat #1 and #2 as
**load-bearing local patches** and never let a sync clobber them.
