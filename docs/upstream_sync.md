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
