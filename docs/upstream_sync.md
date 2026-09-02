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
  `AGENTS.md`'s "databricks_client design" note). A cache hit (common case)
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

### 16. `woffl/flow/outflow.py` — re-floor slip holdup AFTER the Payne correction (FLOW-1, 2026-09-01)
`beggs_diff_press` applies the canonical Beggs-Brill restriction `HL >= lambda_L`
inside `beggs_holdup_base`, then multiplies by the Payne (1979) inclination factor
(0.924 uphill) with no floor afterwards. A gas-free segment (`fgor < Rs(p)`,
`lambda_L = 1`) was therefore modeled at `HL = 0.924` — 7.6 % of the liquid
column replaced by gas density that is not there. Measured on the E-41 fixture at
PF 2,300: fgor 300 moved 12B from 240 to 208 BOPD (−13 %) and 11C from 212 to
170 (−20 %); fgor 250 shifted psu by 101–125 psi; fgor 800 wells moved 0 to −7 %.
The model was optimistic on low-GOR wells and friction calibration had been
absorbing the error. One line: `slh = min(max(slh, nslh), 1.0)` after the Payne
call. Gassy segments where Payne leaves `HL` above `lambda_L` are bit-identical.

**Guarded by:** `tests/outflow_test.py::test_gas_free_segment_holds_full_liquid_holdup`
(a gas-free segment must return `slh == 1.0`) and
`::test_slip_holdup_never_below_no_slip_after_payne`. `test_bottom_pressure` was
re-pinned for the 600-scf/stb fixture if it moved.

### 17. `woffl/pvt/blackoil.py` + `resmix.py` — ACOUSTIC oil compressibility in the sound-speed path (PVT-F1, 2026-09-02)
`ResMix.cmix` is Wood's (1930) equation, but it consumed `BlackOil.compress`, which
BELOW the bubble point is the McCain-Rollins-Villena total compressibility — a
material-balance quantity that includes the volume of gas liberated as pressure
drops, 1-2 orders of magnitude above the liquid-phase value. Fed into Wood's
equation it gave pure-oil "sound speeds" of 344-877 ft/s below Pb (real: ~4,000)
and a 4.4x jump across Pb: Schrader 100 °F, WC 0.8 / GOR 400 → cmix 1,244 ft/s at
1,749 psig vs 1,657 at 1,751 (+33 % across two psi); WC 0.3 / GOR 300 → 895 vs
1,597 (+78 %). Consumers: `mach_te`, `sonic_status`, the throat-entry subsonic
mask, Header Impact "sonic-decoupled", the optimizer's never-swap-for-sonic rule.

The patch adds `BlackOil.compress_acoustic` — Vasquez-Beggs evaluated with the gas
actually in solution at the current pressure (`gas_solubility()`, i.e. Rs(p) below
Pb and Rsb above) — and `cmix` reads that instead of `compress`. Above Pb the two
are the SAME call, so above-Pb cmix is bit-identical and the value is continuous
across Pb (0.14 % / 0.20 % on the two cases above). `compress` and `comp_comp`
keep their material-balance meaning and names; nothing else consumed them. After
the patch pure-oil c is 3,816-5,211 ft/s below Pb (V-B extrapolates low on co at
low Rs / low pressure, so the low-pressure end runs somewhat HIGH; the gas term
dominates cmix there whenever free gas exists). `mach_te` in `batch_test.py`
dropped 7-11 % on every row; rates, psu and the sonic set did not move. The cmix
docstring now states that the isothermal gas compressibility is deliberate
(Wood's bubbly-liquid heat-bath argument, what Himr 2009 fits) — do not "fix" it
to adiabatic.

**Guarded by:** `tests/test_pvt_resmix.py::test_cmix_continuous_across_bubble_point`
(both cases within 1 % across Pb; red at +33 % / +78 % if cmix reverts),
`::test_cmix_above_bubble_point_unchanged_by_acoustic_patch`,
`tests/test_pvt_blackoil.py::test_compress_acoustic_continuous_across_bubble_point`
and `::test_pure_oil_acoustic_sound_speed_is_physical_below_pb`. `batch_test.py`
`mach_te` pins re-baselined (dated comment there).

### 18. `woffl/pvt/blackoil.py` — floor on the Vasquez-Beggs compressibility (PVT-F2, 2026-09-02)
The V-B numerator `5 Rs + 17.2 T − 1180 γg + 12.61 API − 1433` goes negative for
~10-14 API at 60-80 °F (Ugnu): 14 API / 0.65 / Pb 1,500 / 60 °F returned
−1.6e-6 psi⁻¹ above Pb, so `Bo` ROSE with pressure and, with no free gas, `cmix`
put a negative bulk modulus into `sqrt` (untyped `ValueError`). It also dips
below 1e-6 for mid-API oil at 80 °F below ~900 psig with little gas in solution.
`BlackOil._floor_compressibility` clamps both the material-balance above-Pb branch
and the acoustic helper at `_CO_FLOOR = 1e-6 psi⁻¹` (below any real oil; water is
3.1e-6) and warns ONCE per process (`RuntimeWarning`, class flag
`_co_floor_warned`). No-op on every in-range input.

**Guarded by:** `tests/test_pvt_blackoil.py::test_compressibility_floor_heavy_oil_low_temp`
(co ≥ 1e-6 on both paths, warning fires once, Bo falls with pressure),
`::test_compressibility_floor_is_noop_in_range`, and
`tests/test_pvt_resmix.py::test_cmix_does_not_raise_for_heavy_oil_at_low_temp`.

### 19. `woffl/pvt/formgas.py` — z-factor is Dranchuk-Abu-Kassem, not the "grad school" cubic (PVT-F3, 2026-09-02)
`_compute_zfactor` used the unattributed cubic, which drifts −10 % at ppr 4.5
(γg 0.65, 80 °F: 0.700 vs DAK 0.779), −20 % at ppr 6, and returned 0.143 vs DAK
0.686 for γg 1.0 at 3,015 psia — inside the accepted input range and above the
clamp — with the WRONG SIGN of dz/dp beyond ppr ≈ 3, so `compress` (cg) was
overstated and `cmix` understated (compounding #17). The already-implemented
`_zfactor_dak` was verified to converge over tpr 1.05-3 / ppr 0-15 (0 failures on
a 0.05 × 0.25 grid). New `FormGas._zfactor` = DAK, clamped to
`[_ZFACTOR_MIN, _ZFACTOR_MAX]` (the P1-10 crash guard stays), with the cubic as
the fallback ONLY if Newton raises; `compress` differentiates the same
correlation; DAK's Newton tolerance tightened 1e-3 → 1e-7 so the +10 psi forward
difference is not dominated by stopping error. Against the 42-point Peng-Robinson
sweep DAK is closer than the cubic (mean |z err| 2.55 % vs 3.53 %, max 2.95 % vs
4.78 %); the single 2,500-psig end point moved the other way and two hysys
comparison tolerances were widened by one point (`test_gas_viscosity` 3 → 4 %,
`test_mixture_density` 4 → 5 %, documented in-test). `batch_test.py` pins moved
< 1 % (`mach_te` 9X −0.8 %, rates −0.04 %, psu +0.01 %) and were refreshed.
`_zfactor_grad_school` itself is untouched.

**Guarded by:** `tests/test_pvt_formgas.py::test_zfactor_property_is_dak` (pins
0.779 / 0.686 at 3,015 psia / 80 °F, asserts the cubic's 0.143 no longer reaches
the property, and that cg < 1/p where dz/dp > 0) and
`::test_zfactor_dak_converges_over_validity_range`.

### 20. `woffl/pvt/resmix.py` — `ResMix.__init__` validates wc and fgor (PVT-F4, 2026-09-02)
Every child class range-checks its inputs; the mixture did not. wc 1.05 gave an
oil volume fraction of −0.055 silently; a negative fgor a negative gas mass
fraction. `WellConfig`, the CSV stores and `prop_hist.form_wc` reach this
constructor unguarded. Now `ValueError` for wc ∉ [0, 1] or fgor < 0 (inclusive:
0 / 1 / 0 remain legal — dewatering and dead oil).

**Guarded by:** `tests/test_pvt_resmix.py::test_watercut_out_of_range_raises`,
`::test_negative_fgor_raises`, `::test_watercut_and_fgor_boundaries_accepted`.

### 21. `woffl/pvt/blackoil.py` + `resmix.py` — an undersaturated stream's oil carries only fgor (PVT-F5, 2026-09-02)
When `fgor < Rs(p)` the mass balance (`_owg_mass_fraction`) clamps free gas to
zero, but the oil was still evaluated at the correlation's Rs(p): Schrader at
fgor 150 / 1,400 psig / 100 °F carried 48 scf/stb of phantom dissolved gas
(density 54.65 vs 55.03 lbm/ft³, viscosity 13.6 vs 16.5 cP). `BlackOil.condition`
gains an optional `rs_max` (default None — standalone API unchanged, and a later
plain `condition()` drops the cap); `_compute_gas_solubility` returns
`min(Rs(p), rs_max)`; `ResMix.condition` passes `rs_max=self.fgor`. Every
derived oil property (Bo, density, viscosity, tension, acoustic co, above-Pb co)
reads `gas_solubility()`, so one cap fixes them all, and the mass balance's
`xrs` now lands on exactly zero free gas instead of clipping a negative. The
standard-condition densities in `__init__` are deliberately NOT capped (stock-
tank oil is a property of the oil, not the stream), so every saturated stream
(`fgor ≥ Rs(p)`, the normal case, including the E-41 fixture) is bit-identical.

**Guarded by:** `tests/test_pvt_resmix.py::test_undersaturated_stream_oil_carries_only_fgor`
(oil density/viscosity equal the Rs = 150 evaluation exactly, xgas == 0, fractions
close), `::test_saturated_stream_oil_is_bit_identical`,
`::test_standalone_blackoil_condition_unchanged`.

Housekeeping in the same pass: `woffl/pvt/deadoil.py` (a 17-line `DeadOil` stub
with zero imports anywhere in `woffl/`, `server/` or `tests/`) was deleted. If an
upstream sync brings it back, it is safe to delete again.

---

### 22. Lazy matplotlib / scipy.optimize imports on the solver import chain (SOLV-P1, 2026-09-01)
`woffl/assembly/batchpump.py`, `woffl/flow/jetplot.py`, `woffl/geometry/wellprofile.py`
and `woffl/flow/jetgraphs.py` imported `matplotlib` (and `batchpump`/`jetplot`
`scipy.optimize`) at module scope although only their plotting / curve-fit /
annotation helpers use them. `jetplot` sits on the solver's hot import chain
(`jetflow -> jetplot`), so every ProcessPool worker spawn and the FastAPI process
paid ~0.5-1.0 s for code `batch_run` never runs; a 20-well pad on the 2-worker tier
spent ~2 s spawning to save ~3 s of solves. `batchpump` imports lazily inside the
methods; `jetplot` / `jetgraphs` / `wellprofile` go through a `_LazyModule` /
`_LazyPyplot` proxy so `plt.`/`mpl.`/`opt.` attribute access is unchanged.
`scipy.optimize` stayed eager in `wellprofile` until patch 23 (FLOW-2) moved the
survey fit off the physics path; it is now imported inside `segments_fit`.
Behaviour is bit-identical; only import time moves.

**Guarded by:** `tests/test_batchpump_lazy_imports.py` (a fresh interpreter imports
`batchpump`, `network_optimizer`, `solopump` and asserts no `matplotlib*` module is
loaded; a second test exercises the lazy curve-fit path).

---

### 23. `woffl/geometry/wellprofile.py` — the production traverse reads the RAW survey (FLOW-2 / FLOW-8, 2026-09-02)
`WellProfile.outflow_spacing` interpolated the traverse's node TVDs from
`md_fit`/`vd_fit`, the greedy AIC/BIC piecewise fit `segments_fit` builds in
`__init__`. The fit's loop breaks at the first count that fails to improve, and
Nelder-Mead reports `success=False` for every count >= 4 on real surveys, so the
fitted TVD at the pump sat off the raw survey by a fleet median of 14 ft, p90 111
ft, max 240 ft (MPE-48; 22 of 91 surveys > 60 ft) — while the power-fluid
hydrostatic on the other side of the pump (`solopump.discharge_residual` via
`wellprof.jetpump_vd`) used the RAW survey. The two sides of the pump now see the
same TVD: `_outflow_spacing` takes `md_ray`/`vd_ray`, evenly spaced surface ->
pump with the pre-existing `max(ceil(L/seg_len), 3)` node rule (a straight-line
profile, whose fit was one segment, is bit-identical — `tests/outflow_test.py`
did not move). Fleet probe after: |vd_traverse − vd_raw| at the pump = 0.0 for
all 86 constructible surveys. `segments_fit` / `filter()` / `md_fit` / `vd_fit` /
`hd_fit` stay (upstream API, used by the profile plot) but are now LAZY
properties computed on first access, so construction no longer pays 11-161 ms
(measured median 83 ms, max 620 ms -> 0.1 ms) and `scipy.optimize` is imported
inside `segments_fit` rather than at module scope. Pins: `tests/batch_test.py`
9D/12B/16E moved (the Schrader preset's fit sat 6.65 ft deep at the 6693 ft pump
-> ~2.9 psi of production hydrostatic; 16E 36.45 -> 38.37 BOPD).

**Guarded by:** `tests/test_wellprofile_validation.py`
(`test_traverse_pump_tvd_equals_raw_survey_tvd_fleetwide`,
`test_outflow_spacing_follows_survey_kinks_not_fit`,
`test_outflow_spacing_node_rule_unchanged`,
`test_segments_fit_not_run_by_construction_or_traverse`,
`test_schrader_preset_traverse_uses_raw_pump_tvd`,
`test_wellprofile_imports_scipy_optimize_lazily`) and the re-pinned
`tests/batch_test.py` reference rows.

---

### 24. `woffl/geometry/wellprofile.py` — reject corrupt surveys at construction (FLOW-3, 2026-09-02)
`WellProfile.__init__` accepted anything. `MPC-05 Deviation Survey.csv` carries
179 duplicated MDs whose TVDs alternate (6090 -> 0 / 4547 ft); `sort_profile`'s
unstable argsort interleaved them and the `_horz_dist` |dvd| <= |dmd| clip turned
the impossible steps into silent zeros, putting the pump 281 ft too deep (~100
psi). New `validate_survey(md_ray, vd_ray, tol_ft=SURVEY_STEP_TOL_FT)` runs after
a now-STABLE sort: exact duplicate rows are dropped; duplicate MDs whose TVDs
agree within `tol_ft` coalesce to the first-seen row (survey files that merge two
runs are common — MPH-31 has 157); duplicate MDs that disagree by more, or any
|dTVD| > |dMD| + `tol_ft`, raise `ValueError` naming the station. `tol_ft` = 5.0,
calibrated on the 91 local surveys: run-merge disagreement tops out at 3.9 ft
(MPR-111), the corrupt files sit at 9.5 (MPI-11, one 2.4 ft step at 84 deg
inclination), 17.9 (MPM-64), 32.6 (MPL-20), 49 (MPC-40, subsea datum in the TVD
column) and 4,547 ft (MPC-05). Those five now raise and every caller
(`network_optimizer.load_well_profile`, `server/services/factories.build_well_profile`,
`server/services/wells.py` profile endpoint) already falls back to the field
preset with a logged warning — re-pull the five CSVs from Oracle PDB. The
`_horz_dist` clip (patch 12) stays for sub-tolerance noise. Clean surveys are
bit-identical (`validate_survey` is an identity on them).

**Guarded by:** `tests/test_wellprofile_validation.py`
(`test_mpc05_corrupt_survey_raises_naming_station`,
`test_mph31_toe_up_survey_constructs_and_coalesces_duplicates`,
`test_duplicate_md_conflicting_tvd_raises`,
`test_impossible_step_raises_naming_stations`,
`test_fleet_only_the_known_corrupt_surveys_raise`, plus the exact-duplicate /
within-tolerance / identity cases).

---

### 25. `woffl/geometry/wellprofile.py` — depth lookups: a range check that raises, shallowest-crossing `md_interp` (FLOW-10, 2026-09-02)
`_depth_interp`'s guard was `(min < x < max) is False` — a numpy bool is never
the `False` singleton, so it never raised and `vd_interp(TD + 5000)` returned the
clamped end value. Now `if not (lo <= x <= hi): raise ValueError` (inclusive, so a
pump AT total depth no longer raises). `md_interp` fed a non-monotonic `vd_ray`
as `xp` to `np.interp` (77 of 91 surveys are toe-up; MPH-31's pump TVD mapped to
the 21,180 ft toe instead of 5,144 ft) — it now takes the SHALLOWEST crossing via
the new `first_crossing` helper (same logic as
`server/services/depth_interp.first_crossing_md`). Monotonic surveys are
bit-identical. Every caller in `woffl/` and `server/` already wraps these in
`try/except ValueError`.

**Guarded by:** `tests/test_wellprofile_validation.py`
(`test_vd_interp_out_of_range_raises`, `test_interp_inclusive_at_both_ends`,
`test_md_interp_toe_up_takes_shallowest_crossing`, `test_first_crossing_helper`,
`test_mph31_md_interp_matches_measured_pump_depth`).

---

### 26. `woffl/flow/twophase.py` — zero-flow holdup returns the no-slip holdup (FLOW-11, 2026-09-02)
`vmix == 0` gives `froude == 0`; `beggs_holdup_base` divides by `froude**c` and
`beggs_cf_base` takes `log(froude**h)`, raising a bare `ZeroDivisionError` /
`ValueError("math domain error")` that escaped every `except ValueError` in the
solvers (patch 14 only guarded the friction factor). `beggs_holdup_inc` now
short-circuits `froude <= 0 -> nslh` (no flow, no slip). `froude > 0` is
bit-identical.

**Guarded by:** `tests/test_multiphase.py::test_beggs_holdup_inc_zero_flow_returns_no_slip`
and `tests/outflow_test.py::test_zero_flow_traverse_does_not_raise`.

---

### 27. `woffl/flow/singlephase.py` — laminar cutoff at Re 2300 with a linear blend to Serghide at 4000 (FLOW-12, 2026-09-02)
`ffactor_darcy` ran `64/Re` all the way to Re 4000 and then stepped to Serghide
(f 0.016 -> 0.041, a 2.5x jump in friction at one Reynolds number). Laminar now
ends at `RE_LAMINAR = 2300`; between 2300 and `RE_TURBULENT = 4000` the factor is
blended linearly from `64/2300` to `serghide(4000, rel_ruff)`, continuous at both
ends. `Re < 2300` and `Re >= 4000` are bit-identical; the E-41 / Schrader
fixtures are fully turbulent, so no pin moved.

**Guarded by:** `tests/test_multiphase.py::test_ffactor_darcy_laminar_ends_at_2300_and_blends_to_4000`.

---

### 28. `woffl/geometry/pipe.py` — a wall of half the OD or more is rejected (review section 5, 2026-09-02)
`Pipe.__init__` checked `thick > out_dia`, so a 0.6 in wall on a 1.0 in OD was
accepted and `inn_dia` came out negative. Now `2 * thick >= out_dia` raises.

**Guarded by:** `tests/outflow_test.py::test_pipe_wall_must_leave_an_inner_diameter`.

---

### 29. `woffl/assembly/solopump.py` — suction walk bisects to the feasibility edge; "sonic" = at the choke floor (SOLV-F2, 2026-09-02)
`_residual_walk_inward` probes a fixed fraction grid whose gaps reach 14 % of
the bracket span (~140 psi on a 1,000 psi bracket). When the first FEASIBLE
probe already carried the far end's residual sign it was returned as-is and
`jetpump_solver` reported it as `sonic_status=True` — the marginal 11A fixture
(94 % WC, 2,000 psi): floor 1,877.1, first feasible probe 1,885.0, residual
+35 psid, "sonic" at Mach 0.50, the true root in the unprobed gap. Downstream,
`fric_calibration` refused such wells as "pinned" and Header Impact called them
sonic-decoupled. Now `_refine_feasibility_edge` bisects on FEASIBILITY between
the last infeasible and first feasible fraction until the residual flips sign
(root bracketed) or the edge is pinned to within `_EDGE_TOL_PSI` (2 psi), 2-4
extra evaluations; 11A solves to 1,883.3 psig, residual −8 psid, `sonic=False`.
`sonic_status` is now "the returned suction is (within 2 psi of) the Mach =
mach_crit floor `psu_minimize` solved for", not "which branch returned"; a
mach_te threshold was rejected because at the floor the reported `mach_te` is
the last discrete subsonic sweep point (E-41 9X/10X/11X read 0.85-0.89 while
choked). A feasible endpoint still returns from the first probe — the E-41 48
rows and the other marginal fixtures are bit-identical.

**Guarded by:** `tests/test_asm_solopump.py::TestWalkInwardFeasibilityEdge`
(`test_marginal_11a_is_bracketed_not_sonic` is the real case).

---

### 30. `woffl/assembly/solopump.py` — `_bisection_solve` guards its first probe and returns rates from the returned suction (SOLV-F3, 2026-09-02)
The initial midpoint's `discharge_residual` sat outside the `try`, so a
`JetPumpError` there escaped as "solver failed" on a well whose root IS
bracketed; and an interior exception set `res_mid` negative but left the rates
from the previous midpoint, so a width exit could return psu 469.5 with the
rates of 470.3. The first probe now gets the loop's "too-low side" treatment
and a `rates_at` guard (mirroring `_secant_solve`) re-evaluates at the returned
psu — falling back to the feasible upper end `psu_hi` when the final midpoint
itself has no throat solution. Normal-path solves are unchanged.

**Guarded by:** `tests/test_asm_solopump.py::TestBisectionSolveConsistency`.

---

### 31. `woffl/flow/jetplot.py` — `_dete_zero` bounds the positive-tde clamp (FLOW-5, 2026-09-02)
When tde stays positive along the subsonic branch `_dete_zero` clamps to the
minimum-tde point. That is right only when the minimum is (nearly) zero — the
Mach-1 choke that `psu_minimize` places the floor on. Accepting ANY positive
minimum fabricated a throat-entry state for pumps that cannot be fed at that
suction (a 9X on a 2,000 BOPD / 95 % WC well bottoms out at 26 % of the entry
kinetic energy and reported "sonic, 30 BOPD"). The clamp is now accepted only
when the minimum is within `_CLAMP_TOL_FRAC` = 2 % of `kde_ray[0]`; otherwise
`ThroatEntryChoked` (a `ThroatEntryNoSolution` subclass, defined next to the
raise) is raised. Tolerance justified by measurement: converged floors clamp at
<= 0.5 % (E-41 sweep, marginal, thin-band fixtures) after #32; fabricated ones
sit at 26-98 %. `_residual_walk_inward` steps PAST `ThroatEntryChoked` (the
remedy is a higher suction — the feasible region is above), so at
`mach_crit = 1.0` nothing changes and `mach_crit > 1` diagnostic solves, whose
scaled floor sits below the unscaled Mach-1 choke, walk up to the first real
zero crossing instead of dying (E-41 12B at 1.15 / 1.5 still converges).
Plain `ThroatEntryNoSolution` still propagates for the GOR auto-recovery.

**Guarded by:** `tests/test_jetplot_book.py` (`test_clamp_rejected_when_the_branch_never_closes`,
`test_clamp_accepted_when_minimum_is_within_tolerance`) and
`tests/test_asm_solopump.py::TestWalkInwardFeasibilityEdge::test_throat_entry_choked_is_walked_past`.

---

### 32. `woffl/flow/jetflow.py` — `psu_minimize` checks the choke residual and interpolates tde at Mach = mach_crit (FLOW-9, 2026-09-02)
`psu_minimize` exited on |Δpsu| <= 5 alone, so two clamps at a bound exited
"converged" with a residual nowhere near zero, and `throat_entry_mach_one`
returned the NEAREST subsonic sweep point (`tde_ray[-2]`), which sits above the
true Mach-crit minimum by the 25-psi step's discretization gap. Now: (a) tde is
linearly interpolated AT Mach = mach_crit between the two bracketing sweep
points (`_tde_at_mach`); (b) convergence also requires |tee| <= `_TEE_TOL_FRAC`
= 1 % of `kde_ray[0]` — but ONLY when the sweep actually crossed mach_crit; a
sweep that hit the 50-psig floor first (pure water: Mach 0.04) returns tde at
the floor, a numerical guard whose floor pressure jumps between iterates, so
those keep the historic |Δpsu| exit bit-identically (the water-pump fixtures);
(c) two clamps at the `pres − 10` bound with the choke residual still positive
raise `ThroatEntryNoSolution` ("cannot be fed at any suction") instead of
letting the solver fabricate a choked point there; two clamps at the 50-psig
floor keep returning 50 as before. Every choke floor moved by < 1.2 psi on the
48 E-41 pumps and < 0.2 psi on the marginal fixture; `tests/batch_test.py` 9X
re-pinned (see its dated comment — the mach_te pin is quantized at the choke).

**Guarded by:** `tests/test_asm_solopump.py::TestPsuMinimizeChokeResidual`,
`tests/test_jetflow_bracketed.py::test_throat_entry_mach_one_returns_the_interpolated_value`,
and `tests/test_asm_solopump.py::TestWaterPumpMode` (the non-crossing branch).

---

### 33. `woffl/flow/jetflow.py` — `_throat_discharge_bracketed` finds a hump narrower than its scan step (FLOW-6, 2026-09-02)
The 60-point downward scan (step ~0.1·pte) stepped clean over a positive
momentum-balance hump narrower than one step and raised `ConvergenceError` for
exactly the marginal pumps the fallback (#2, #11) exists for. When the scan
sees no sign change it now maximizes the balance over [15, hi] with a bounded
`scipy.optimize.minimize_scalar` (~20 evaluations; hi pushed out ×4 up to three
times if the balance is still positive there); a positive peak brackets the
physical HIGH root in [peak, hi] for `brentq`. Only the previously-raising path
changes.

**Guarded by:** `tests/test_jetflow_bracketed.py::test_narrow_hump_missed_by_the_scan_is_still_found`.

---

### 34. `woffl/flow/jetplot.py` — `JetBook` sweeps on Python lists with an inline trapezoid (FLOW-7, 2026-09-02)
`JetBook.append` did eight `np.append` copies plus a scipy `trapezoid` on a
two-element slice per sweep step — ~37 % of every solve. The columns are now
lists (`book.prs`, `book.tde`, ...) with the `*_ray` numpy views materialized
lazily by properties (cached until the next append; setter kept for the tests
that assign whole arrays); the expansion-energy increment is computed inline in
scipy's exact operation order (`d * (y[1:] + y[:-1]) / 2.0`, `x = (144·32.174)·p`).
The sweep loops in `jetflow` deliberately keep reading `prs_ray[-1]` /
`tde_ray[-1]`: the SCALAR TYPE fed back into the PVT (np.float64 / np.int64,
not a Python float) is part of bit-identity — the PVT's `**` / `exp` paths round
differently for Python floats, which shifted three E-41 rows by one ULP when the
loops read the lists directly. Verified bit-identical on all 48 E-41 rows and
the marginal fixtures. E-41 sonic solve 16.6 → 7.5 ms, secant 36 → 20 ms,
28-pump batch 1.25 → 0.92 s (with #35).

**Guarded by:** `tests/test_jetplot_book.py` (`test_append_expansion_energy_bit_identical_to_incremental_ee`,
`test_integer_psu_keeps_the_integer_element_type`, `test_zero_tde_walk_bit_identical_to_manual_sweep`).

---

### 35. `woffl/assembly/solopump.py` + `woffl/flow/jetflow.py` — throat-entry seed reuse and secant known-point reuse (SOLV-F9 / SOLV-P3, 2026-09-02)
(a) `discharge_residual` re-swept the throat entry `psu_minimize` had just
computed. `throat_entry_zero_tde(..., seed_book=)` now accepts the Mach-one
book: both walks start at the same psu with the same 25-psi steps, so the seed
is truncated where the zero-tde stop rule would have fired (`_seed_zero_tde_book`,
on a copy) and extended only if needed — the same points, bit-identical.
`jetpump_solver` passes it only for the endpoint probe and only at
`mach_crit = 1.0` (the Mach-one walk scales kde otherwise). (b) `_secant_solve`
keeps every evaluated suction in `known`; a clamp back onto an evaluated point
(typically a bracket end) is served from memory, and a clamp that pins the
iterate on the point it already stands on with |res| > res_tol raises at once —
the next `psu_secant` would have raised on equal residuals anyway. Both
bit-identical by construction (`discharge_residual` is a pure function of psu).

**Guarded by:** `tests/test_asm_solopump.py::TestSeedBookReuse` and
`tests/test_asm_solopump.py::TestSecantSolveReusesKnownPoints`.

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
