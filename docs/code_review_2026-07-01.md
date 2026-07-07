# Codebase Review — 2026-07-01

> **STATUS (2026-07-06, orchestrated fix run):** The P1/P2/R backlog below is
> now closed in the working tree. Fixed this run: P1-8/9/10 (physics
> robustness, upstream-tagged), P1-14/15/16/17/19/20 + P1-13's safe half (GUI
> core), P1-21/22/25 (dead header_engine helpers wired), P1-23/26 (header-impact
> verdicts/lift), P1-29 + P1-35 (pump tenure + same-test coupling), P1-18
> (startup-failure retry), P1-31/32/33/34 (Well Sort UI); P2-2..P2-7; R-8
> (canonical marginal WC — Step-3's auto-fill changes 1.00 → 0.97), R-10 + R-11
> (dead code/hygiene), Vogel consolidation (`woffl/gui/vogel.py`); the 4 red
> physics tests re-pinned to commit 9b20c65's Beggs-Brill corrections
> (attribution bisect-verified). Full per-finding evidence:
> `docs/review_status_2026-07-06.md`. Suite: 1186 passed / 1 skipped (opt-in
> live marker). Still open: R-7 (blocked on the uncommitted jetpump_solver.py
> edit), R-9 context-builder consolidation (deferred to the file-splits plan),
> R-2..R-5 file splits, the rho_pf wire-or-remove decision (P1-13's
> behavior-changing half), and new findings from this run — the dead
> out-of-range check in `wellprofile._depth_interp` (`is False` on numpy bool),
> the now-orphaned `databricks_client.get_tags_for_wells`, the unguarded
> tag-list f-string in databricks_client, and the zero-caller
> `WellTestProcessor` class.

> **STATUS (2026-07-01, same day):** All P0 findings below are FIXED in the
> working tree, plus the safety-net tests (pad plants pinned values, review-store
> round-trip, GUI smoke imports). Suite: 724 passed (was 565). P1/P2 and the
> refactor plan (R-x) remain open. Fix details per finding are noted inline in
> code comments referencing these IDs.

**Scope:** entire working tree (~39k lines in `woffl/` + ~4k tests), reviewed by nine parallel
deep-read agents (library physics, assembly/optimizers, GUI core, solver cluster, tabs/viz/export,
multi-well workflow, pad pages, Header Impact cluster, Scott's Tools, data/tests/hygiene), findings
deduped and the highest-impact claims spot-verified by hand.

**Baseline:** 565/565 tests pass (41 s). `compileall` clean. Line numbers reference the
2026-07-01 working tree (38 modified files uncommitted at review time).

**Verified by hand** (marked ✓ below): P0-1, P0-2, P0-4, P0-6, P1-4, P2-1. Everything else is
agent-reported with the stated confidence; verify the specific line before fixing if it looks off.

**Purpose:** feed the cleanup/refactor pass, then the "bulletproof pad optimization for field
engineers" push. Priorities are ordered for that goal.

---

## P0 — Wrong answers on the pad-optimization path

These are the findings that can put a wrong pad plan in front of an engineer today.

### P0-1 ✓ `step_review_wells.py:509` — batch auto-match is 100% broken (NameError)
`_recent_test_rates` is called but never imported in this module (it lives in `s_pad_page.py:97`;
i/m pad pages import it, this file doesn't). The call site is inside `_batch_automatch_inputs`,
wrapped in `except Exception` → "🎯 Auto-match all wells (beta)" silently skips **every well on
every pad** with caption "Skipped: MPS-xx (NameError)". The flagship one-push pad-match feature
does nothing and reports it as a data problem. Fix: import it (better: move `_parse_pump` +
`_recent_test_rates` to a shared helpers module — see R-1). Add a smoke test.

### P0-2 ✓ `well_review_store.py:118,234-237` + `network_optimizer.py:293` — WC ≥ 0.99 round-trip corruption (the S-03 bug)
Snapshot converts `qwf_liquid = qwf_oil / max(1-wc, 0.01)`; the optimizer converts back
`oil = qwf_liquid × (1-wc)`. The clamp breaks the inverse:
- WC = 1.00 → oil = 0 → all configs NaN → well contributes nothing → Results says
  "**Optimizer shut in** S-03 — their power fluid bought more oil elsewhere", false on both counts.
- WC = 0.995 / 0.999 → modeled oil silently 0.5× / 0.1× the reviewed oil.

Commit `89efbe6` added water-pump mode to the single-well Solver but the pad path
(`well_review_store → WellConfig → NetworkOptimizer`) never learned it. Fix options: propagate
water-pump mode into `WellConfig`/batch, or hard-stop at review save for `form_wc > 0.99` with a
clear message. Silently zeroing is the worst option. Also remove the round-trip asymmetry (store
oil directly or clamp identically on both sides).

### P0-3 Pad pages: stale results vs live store → false "Optimizer shut in"
`s_pad_page.py:673-677`, `i_pad_page.py:599-600`, `m_pad_page.py:560-561`. The SI list is
`active_wells_now − optimized_wells_then`. Add/online a well after a run and it renders as
"Optimizer shut in" without the optimizer ever seeing it. Nothing invalidates results on store
change. Fix: stamp a run signature (well set + per-well physical inputs + method) into `meta`;
banner + suppress SI attribution when current store signature ≠ run signature. (Same fix class as
the Batch-Run `_batch_sweep_cache` pattern.) Related: I/M pages don't pop the stored scenario on a
fresh optimizer run (S does — `s_pad_page.py:643-646`); copy that.

### P0-4 ✓ `network_optimizer.py:806-844` (`load_wells_from_dataframe`) — NaN-truthy fallbacks = the silent well-drop mechanism
`base_config.get("out_dia") or default` keeps **NaN** (NaN is truthy). Databricks nulls flow into
tubing/casing dims, qwf, pwf, GOR, surf_pres; `WellConfig.__post_init__` doesn't validate them;
every combo for that well NaNs; the optimizer then drops the well silently (see P0-5). Fix:
`pd.notna(x)` guards + validate the physical fields at load with a named-well error.

### P0-5 Workflow: partial failure renders as success; MILP and MCKP solve different problems
- A well whose whole nozzle×throat sweep NaN'd still shows in "Completed batch simulations for
  N wells" (`step3_configure_optimize.py:477`); failure detail exists in `df["error"]` but nothing
  reads it; `_simulate_single_well` prints to stdout (invisible on Databricks Apps).
- MILP per-well constraint is ≤1 (`optimization_algorithms.py:119-122`): implicit shut-in, wells
  silently omitted. MCKP is exactly-one (`network.py:155-157`): infeasible budget raises instead.
  MCKP also only searches `df["semi"]` (total-water Pareto set, `batchpump.py:321-324`) even when
  the budget is lift-water — so the two methods differ in semantics and search space, not just
  solver. Step 4's only unallocated-well caption is buried and conditional
  (`step4_results.py:434-439`).
- Fix (see B-1): per-well reconciliation table after batch + in Step 4; make MCKP's skip list and
  MILP's omissions explicit; document/unify the semantics; either give MCKP full config access
  under lift-water budgets or state the restriction.

### P0-6 ✓ `marginal_watercut` is a dead knob in every optimizer path
`NetworkOptimizer` stores it (`network_optimizer.py:211`); neither `milp_optimization` nor
`mckp_optimization` reads it. Step 3 collects it (even auto-fills from Well Sort); all three pad
pages pass it; the pad Results SI info-box attributes shut-ins to "uneconomic at the marginal
water cut" — an impossible outcome. Fix: enforce (filter configs whose marginal/total WC exceeds
threshold — one list-comp in MILP; pass through to MCKP) or remove the widget everywhere.

### P0-7 `i_pad_page.py:187,211,262,332,349` — over-capacity I-Pad scenario crashes instead of warning
`_frontier_header` can return suction ≈ 217 psi; damped iterate drops below 1000;
`PowerFluidConstraint.__post_init__` raises "pressure must be between 1000-5000". The engineer
sizing pumps up gets a crash instead of the designed `over_capacity` message. S clamps to
[1000, 5000]; M floors at 1400; only I is exposed. Also `_match_check_ipad` (`i_pad_page.py:128-130`)
models wells at an uncapped header (≈4,400-4,750 psi vs the 3,500 cap used everywhere else) →
spurious "✗ BUST" verdicts.

### P0-8 Workflow stale-results traps
- `step3_configure_optimize.py:561-563`: a failed re-run leaves the **previous** run's results
  live (the `not results` branch cleans up; the exception branch doesn't) → "View Results" shows a
  plan from different inputs.
- `step2_review_ipr.py:140-147`: "Re-run IPR Analysis" doesn't invalidate downstream configs →
  optimizer runs against configs built from old fits.
- `step3:129-158`: CSV override can't be reverted (pristine configs overwritten in place; caption
  says removing the file reverts — it doesn't) and a same-name/size re-upload is silently skipped
  by the signature check.
- Fix class: `_clear_downstream()` on every input-changing action + run-provenance manifest (B-3).

### P0-9 Scenario evaluators: fixed-point non-convergence rendered as success
All six `_evaluate_fixed_scenario` / `_evaluate_existing_scenario` copies (S/I/M) lack the
`converged` tracking that `_run_coupled_optimization` has. After 8 oscillating iterations the
per-well rows come from the last trial header while `meta["header_psi"]` is the damped
extrapolation — internally inconsistent, no caveat. Fix once in the unified evaluator (R-1).

### P0-10 Pad review-store CSV holes become silent plausible garbage
`well_review_store.py:369`: missing floats coerce to 0.0, then `to_well_config` clamps into
library bounds (res_pres 0→400 psi, jpump_tvd 0→2500 ft, form_temp 0→32 °F). A truncated CSV
optimizes without warning. Lowercase `field_model` ("schrader") passes the default check and
raises at run time — uncaught on S-Pad (`s_pad_page.py:635-639` has no try/except around Run,
unlike I/M). Fix: loud per-row validation on upload (B-5) + wrap S-Pad's run handler.

---

## P1 — Wrong answers elsewhere (single-well, physics, decision tools)

### Physics library (all fixable locally; flag for upstream PR per CLAUDE.md rules)

- **P1-1 [HIGH]** `jetflow.py:611-617,766` + `resmix.py:422-459` — **water-pump mode drops the PF
  volume from every flow anchored on `prop_tm`**: at `wc_su ≥ 1.0`, `diffuser_discharge` and
  `production_top_down_press` see formation water only (300 BWPD instead of 2,800 with PF), while
  `throat_discharge` includes nozzle mass — internally inconsistent chain. This sits in the local
  `model_as_water` patch (no upstream coordination needed) and is directly relevant to S-03-class
  wells. Fix: anchor downstream calls on `qoil_std + qnz_bwpd` in water mode.
- **P1-2 [HIGH]** `twophase.py:272-313` — Beggs & Brill missing the canonical **C ≥ 0 clamp** (and
  the `HL(0) ≥ λ` floor). Negative C is routine for intermittent flow → holdup understated ~30% in
  realistic cases; extreme cases give negative slip density (pressure decreasing with depth)
  silently. Affects every BHP/discharge traverse.
- **P1-3 [MEDIUM]** `jetflow.py:562-588` — `_throat_discharge_bracketed` scans from 15 psig up and
  takes the **first** sign change; the residual generically has two roots and the low one is the
  non-physical/choked branch → understated ptm/pdi → false "pump can't lift" on exactly the
  marginal wells the patch exists to save. Fix: scan downward from `hi` / restrict bracket to
  ~[pte, hi] / take the largest root.
- **P1-4 ✓ [MEDIUM]** `batch_run.py:1007-1016` + `power_fluid_range.py:405-415` — sweep-cache
  signatures **missing `params.model_as_water`** (neither file references it at all). At
  `form_wc == 1.0` — exactly the dewatering case — toggling the mode produces an identical
  signature; the cache serves results from the other mode. One-line fix each + see R-6 (shared
  signature builder).
- **P1-5 [MEDIUM]** `twophase.py:454-472` / `outflow.py` — B&B acceleration term (`beggs_ek`,
  `1/(1−Ek)`) implemented but never wired into `beggs_diff_press`. Understates gradient where gas
  velocity is high (upper wellbore).
- **P1-6 [MEDIUM]** `wellprofile.py:238-249` — survey noise with `|Δvd| > |Δmd|` → silent NaN →
  `hpat="unknown"` → **bare KeyError** in `beggs_holdup_inc` that no `except ValueError` catches.
  Clamp vd_diff to md_diff at construction; make `beggs_flow_pattern` raise typed.
- **P1-7 [MEDIUM]** `jetplot.py:232-258` — `_dete_zero` interpolates over a possibly non-monotonic
  masked array (non-contiguous `dtdp ≥ 0` mask) → silent garbage pte/mach. Keep the leading
  contiguous run of the mask. (Repo-wide pattern: `np.interp` on assumed-monotonic xp also in
  `md_interp`, `dedi_zero`, `utils.py:1559` — consider one guarded-interp helper.)
- **P1-8 [MEDIUM]** `solopump.py:664-689` + `:285` — bisection midpoint catches
  `ThroatEntryNoSolution` but not `ConvergenceError`; walk-inward doesn't walk past a bare
  `JetPumpError` from `nozzle_velocity` (pni ≤ pte). Both re-create "works in well, not in model"
  on the fallback path (loud failure, not silent).
- **P1-9 [LOW]** `singlephase.py:179-180` — `64/reynolds` with no Re=0 guard → bare
  ZeroDivisionError escapes untyped on zero-flow inputs.
- **P1-10 [LOW]** `formgas.py:230-247` — `_zfactor_grad_school` cubic unguarded, can go ≤ 0 at
  high ppr/low tpr → sqrt domain error in `cmix`. Clamp or switch to the adjacent `_zfactor_dak`.

### Single-well tabs & GUI core

- **P1-11 [HIGH]** `batch_run.py:423-483` — batch calibration pairs the **current pump's model row
  with the most recent test's actuals without checking pump-at-test-date**. In the JPCO→next-test
  window (exactly when engineers open Batch Run) the ratio is bogus, clamped to [0.3, 2.0], graded,
  and offered as a toggle that rescales the whole table. The Solver guards this (`pump_differs`);
  pdf_export guards it; Batch Run doesn't. Same gap in
  `power_fluid_range._render_calibration_flags_for_installed_pump` (warnings only). Fix via one
  shared "installed-pump vs latest-test" helper using `get_pump_at_date` (R-7).
- **P1-12 [HIGH]** `utils.py:1414-1440` — `_load_well_characteristics_cached` returns (and
  st.cache_data caches) fallback/empty frames **process-wide for 1 h**: a Databricks blip at fill
  time serves stale jp_chars.csv to every user; the double-failure branch caches an **empty well
  list** (dropdown collapses to "Custom", Well Database blank, no retry). Fix: raise on
  empty/failed instead of returning, so the cache stays unfilled; render status outside the cached
  core; then inspect `errors["chars"]` in `app.py:129-137` (currently ignored).
- **P1-13 [MEDIUM]** Sidebar "Power Fluid Density" is a **dead knob**: `run_jetpump_solver` /
  `run_batch_pump` / `run_power_fluid_range_batch` accept `rho_pf` and never use it (PF
  hydrostatics come from FormWater); it's echoed in the input summary and PDF as if live. Pad
  pages separately hardcode 62.4 while I/M plant SG ≈ 1.03-1.04 (~2% nozzle-flow bias). Wire it or
  remove it everywhere.
- **P1-14 [MEDIUM]** `batch_run.py:895-937` — with calibration ON, the Nelder-Mead comparison
  mixes calibrated seed vs raw NM output → phantom ~40% "uplift" at factor 0.7.
- **P1-15 [MEDIUM]** `well_profile.py:60-64,145-148` — "Horizontal Deviation" plotted as
  `md − vd`, not the Pythagorean `hd_ray` the profile already carries; >2× error on slant sections.
- **P1-16 [MEDIUM]** `power_fluid_range.py:383-387` vs `utils.py:1354-1356` — `np.arange(min,
  max+step, step)` overshoots the stated max when the range isn't step-divisible; Best Performers
  can crown a pressure outside the requested range.
- **P1-17 [MEDIUM]** `batch_run.py:55-56,285-286` — partially failed sweep renders as complete
  (failed combos silently filtered; `df["error"]` never surfaced). PF Range does this right;
  copy its Successful-Runs metrics.
- **P1-18 [MEDIUM]** `app.py:177-184` — one startup well-test failure pins
  `all_well_tests_df=None` for the session; Step 3/4 `_build_actual_maps` and
  `scotts_tools/_common.py:126` read it directly → calibration/actuals silently vanish even after
  Databricks recovers (single-well flows self-heal; these don't).
- **P1-19 [LOW]** `batch_run.py:1099-1105` — calibration toggle is a bare widget key
  (`value=False`): a Solver detour GC's it and the table silently reverts to uncalibrated. Same
  class: `pe_nozzle_no`/`pe_area_ratio`, `nm_*` keys, `jp_hist_bhp_zero`,
  `single_well_page.sw_active_view`.
- **P1-20 [LOW]** `utils.py:953-955` — out-of-range `jpump_tvd` silently falls back to default
  pump MD with only a `print()` (invisible on Databricks Apps).

### Header Impact

- **P1-21 [HIGH]** `header_impact.py:97-136,1401-1403` — **failed JP solves rendered as "no
  response"**: `_solve_at_whp` never reads `df["error"]`; a non-converging well lands in the
  "header move won't help" bucket — exactly the marginal-well population. Surface solver errors as
  "model failed", not "no response".
- **P1-22 [MEDIUM]** `header_impact.py:1319-1333,1361-1377` — WHP-floor clamp: displayed scenario
  WHP can be negative, and `phys_slope` divides by the **unclamped** Δ (≈2.9× understatement in
  the example) → biased Physics-vs-Empirical scatter and sense-check verdicts.
- **P1-23 [MEDIUM]** `header_impact.py:611-616` — "no data"/"insufficient" non-JP wells booked as
  confident ΔOil = 0 labeled "slugging" → systematically understates field uplift; the row
  contradicts its own "Emp class" column.
- **P1-24 [MEDIUM]** `header_impact.py:1268-1278` — a JP well's manual ResP edit is silently
  absorbed into the base (and ignored) after any Apply button or scenario reload, while still
  displaying the edited value. Saved scenarios reproduce different results than the original run.
- **P1-25 [MEDIUM]** `header_impact.py:597-602` — non-JP `bhp_now` anchor = raw last hourly bin
  (no shut-in/sentinel filter) → a 12-psi tail anchors the whole generic Vogel. Use median of last
  N cleaned readings with the >50 psi filter.
- **P1-26 [MEDIUM]** `header_impact.py:179-189` — `_classify_lift`: JP classification wins if
  *any* install exists, with no recency bound → a JP→ESP conversion years ago still models as JP
  forever. Needs a terminator (e.g., ESP amps in recent tests override stale installs).

### Well Sort / Triage / calibration tools

- **P1-27 [HIGH]** `well_sort.py:1230-1258` — Triage BOL-trial compares **formation-water** WC
  history against a **total-WC** marginal line → systematically over-recommends bringing
  high-water JP wells back online. (`_effective_wc` also silently falls back form-basis when
  `totl_wc` is NaN.)
- **P1-28 [HIGH]** `well_sort.py:1182-1197` — outlier protection only checks `OilDev`; a single
  fluky high-water test (normal oil) → straight to "🔴 SI candidate" with "the latest test looks
  real". `WatDev` is already computed; use it.
- **P1-29 [HIGH]** `jp_calibration.py:94,400` — fits friction coefs with **today's pump** against
  a historical BHP test (no `get_pump_at_date`, no pump-changed guard) — and the SQL preview
  pushes those coefs to `vw_prop_mech` for every downstream consumer. jp_fric_trend/jp_washout
  already do this right; copy them. Same tenure violation in `pf_scenario.py:253,563`.
- **P1-30 [MEDIUM]** `well_sort.py:612` vs `well_sort_client.py:943` — **two live
  `compute_field_marginal_wc` implementations with different math** (POPS inclusion, basis,
  algorithm). Step-3 auto-fill uses one; Batch-Run import/Triage the other. Unify (R-8).
- **P1-31 [MEDIUM]** `well_sort.py:128-163,585-589,934-936` — POPS config + pad pump limits live
  only in widget keys → GC'd on app-mode switch → Batch-Run import silently uses hard-coded
  default POPS set; edited pump limits revert.
- **P1-32 [MEDIUM]** `jp_washout.py:80-91,136-163` — `PumpChangedSinceTest` computed then thrown
  away → wash-out flags recommend changeouts for pumps already out of the hole.
- **P1-33 [MEDIUM]** `well_sort.py:1523-1528` — Triage shows no XV-empty warning (Wells tab does);
  on the hosted app XV is *always* empty → decisions silently degrade to the lagging shut-in log.
- **P1-34 [MEDIUM]** `well_sort.py:896-904` — pad-marginal caption describes capacity-shedding
  the code doesn't do (`compute_pad_marginal_wc` is a plain max).
- **P1-35 [MEDIUM]** `jp_calibration.py:80-83` — BHP target and WHP boundary can come from
  different tests (independent latest-row picks); mismatch absorbed into pushed coefs.

---

## P2 — Deployment & foundation

- **P2-1 ✓ [HIGH]** `requirements.txt:8` — bare `woffl` self-pin: Databricks installs **unpatched
  upstream woffl** into site-packages next to the vendored patched tree; works today only by
  path-precedence luck; any sys.path change silently swaps the solver. Remove the line. Also drop
  `databricks-sdk` (imported nowhere).
- **P2-2 [MEDIUM]** `.github/workflows/publish-pypi.yml` — fork carries a live PyPI publish
  workflow on every tag, using dead `actions/*-artifact@v3`. Delete or gate on
  `github.repository == 'kwellis/woffl'`.
- **P2-3 [MEDIUM]** `tests/e41_test.py`, `tests/jpump_test.py` — demo scripts, zero asserts, run
  full solver sweeps + `plt.show()` at pytest **collection**; `tests/test_joint_match_sweep.py`
  likewise collects and mutates `WOFFL_MAX_WORKERS` process-wide. Move to `scripts?/tools/` or
  guard under `__main__`. Add `[tool.pytest.ini_options] testpaths=["tests"]` (a bare `pytest` at
  root would also collect `scotts_tools/test_harness.py`). NOTE: `batch_test.py`/`outflow_test.py`
  rely on the `*_test.py` collection pattern — don't narrow `python_files`.
- **P2-4 [MEDIUM]** `tests/test_utils.py:715-855` — live-Databricks-coupled asserts
  (`test_mpb28_values` pins `res_pres == 1900.0` from the live view) → suite goes red on a data
  update. Mock or mark.
- **P2-5 [MEDIUM]** `pull_surveys.py:76,86` — legacy script writes `jp_chars.csv` back with an
  extra index column every run (damage already visible in the tracked file) and over-aggressive
  `dropna()`. Delete it; point `check_missing_surveys.py`'s message at `pull_missing_surveys.py`.
- **P2-6 [LOW]** `databricks_client.py:132-135` — first-attempt connection failure skips the
  retry path; `_oauth_token` holds the lock across a 30 s HTTP call (serializes startup warm
  threads).
- **P2-7 [LOW]** SQL built by f-string interpolation of well names/dates throughout
  `well_test_client` / `pad_watercut_client` (internal source, low risk, but it's the whole query
  surface).

---

## Bulletproofing plan (pad + multi-well workflows)

Ranked; these close the "engineer acts on a wrong plan" class structurally.

- **B-1 Well accounting, end to end.** One `reconcile_wells()` helper: "selected 24 → IPR-fit 22 →
  simulated OK 20 → in plan 18, per-well reasons", rendered in Step 3, top of Step 4, and the pad
  Results pages. Today drops happen at four layers with four visibilities (fetch banner, stdout
  print, logger.warning, silent MILP omission). Includes: read `df["error"]`, make
  `mckp_optimization` return its skip list, distinguish "failed" / "shut-in" / "not run".
- **B-2 Pre-run validation gate.** `validate_well_config` exists (`network_optimizer.py:929`) and
  is never called. Wire it + NaN-proofing (P0-4) + degenerate-IPR checks (pwf ≥ res_pres, qwf ≤ 0,
  zero budget) into Step-2 Proceed / Step-3 Run / pad Run, blocking or requiring acknowledgment.
- **B-3 Run-provenance manifest + signatures.** Store inputs (budget, PF pressure, method,
  water_key, calibration flag, well-set signature, timestamp) with every result set; render at top
  of Results; embed in CSV/PDF exports; banner "inputs changed since this run". Fixes P0-3/P0-8
  structurally.
- **B-4 Honest constrained-stream labeling.** `power_fluid_utilization` uses lift-water ÷ budget
  even when the budget is total-water (`network_optimizer.py:553`, `step4_results.py:167`); Step-4
  pad summary re-reads the *current preset* instead of the run's budget (`step4_results.py:29-34`);
  MCKP hardcodes `marginal_oil_lift_water` while MILP maps by water_key
  (`optimization_algorithms.py:245`).
- **B-5 Loud CSV validation** (workflow override + pad review store): per-row accept/reject table,
  required columns, bounds, `field_model` normalization, content-hash signatures.
- **B-6 Convergence surfaced everywhere** — `converged` in all six pad scenario evaluators + the
  header actually used for final rows.
- **B-7 Calibration honesty.** "Calibration requested but not applied: <reason>" caption (Step 3
  silently skips when tests/JP map missing); calibration toggle must not change the candidate pump
  set (`step3:437-443` — inject current pumps unconditionally); pad-page equivalents.
- **B-8 Test-date on actuals** — CvO tables/PDF show "Actual Oil" with no date; add date + age
  warning.
- **B-9 Memoize the workflow batch sweep** on a config signature (single-well pattern) so
  method/budget changes don't re-run the multi-minute ProcessPool sweep and widen the
  stale-results window. Same for the pad match-check failure memo (currently retries every rerun).
- **B-10 Post-run sanity check** — auto-rerun match-check at the converged header for wells whose
  recommended pump == current pump; flag model ≠ measured > 2×.

---

## Refactor plan

### R-1 Unify the three pad pages (the marquee item)
Measured: **~75-80% of ~2,900 lines is triplicated** (six byte-identical ~30-line ★/ripple blocks,
six identical fallback blocks, near-identical evaluators/configure/results/nav). Genuinely
pad-specific: ~150-250 lines each. Proposed:
1. `pad_plant_base.py` — `PadPlant` ABC (poly/meta loader, affinity math, bisection inverse,
   `header_at_flow`, `budget_at_pressure`, `pressure_window`, `flow_window`, `envelope(at_pressure)`,
   `flags`); S/I/M as thin data subclasses. Fixes P0-7/P0-9/amps-at-wrong-point structurally.
2. `pad_optimize.py` — pure, no Streamlit, unit-testable: one `run_optimization`, one evaluator
   pair, one match-check, uniform meta `{header_psi, total_pf, total_oil, converged, flags}`.
3. `pad_page.py` — `PadSpec(pad, plant, title, captions, defaults)` + `run_pad_page(spec)`; one
   "Pad Optimization" app mode with a pad selector. **A new pad = one PadSpec + a plant subclass,
   or `FixedHeaderPlant(psi)` for pads with no booster model — this is the direct path to "any
   engineer optimizes any pad".**
4. Move `_parse_pump`/`_recent_test_rates` to shared helpers (kills the sibling imports and P0-1
   at the root).

### R-2 Split `header_impact.py` (3,257 lines) into a package
Cut along its existing banners: `data / controls / solve / results / review / coverage / exports /
diagnostics / figures`, `__init__` re-exports `render_tab`. Move pure solve math into
`header_engine.py` (its docstring already queues this; tests exist to receive it). Consolidate the
six BHP-scatter implementations and three IPR-figure implementations; move
`_estimate_gaugeless_ipr` out of pf_scenario into `_common`. **Use the Python cutter pattern, not
PowerShell line-slicing (mojibake lesson).**

### R-3 Split `utils.py` (~1,576 lines) into five modules
`sim_objects / well_data / well_tests / sim_runners / calibration_ui` with `utils.py` as a
re-export shim (~30 import sites untouched). Kills the utils↔tabs lazy-import cycle. Drop dead
params (`build_calibration_inputs`' unused `wellbore`/`well_profile`; `rho_pf` black hole per
P1-13).

### R-4 Split `well_sort.py` (~1,666 lines)
`well_sort_data.py` (the seven cached fetchers — app.py's prefetch thread imports them) / Wells
tab / marginal-WC engine + tab / `triage.py`. The two decision functions are pure DataFrame→
DataFrame — unit-test them when extracted (highest-consequence untested logic in the file).

### R-5 Split `jetpump_solver.py` (2,783 lines), `batch_run.py` (1,161), `pdf_export.py` (1,139)
batch_run's sections are cleanly separable (graph/recommender, hero/calibration, marginal-WC
quickfix, NM expander). pdf_export: extract the IPR data-prep shared with jetpump_solver into one
pure function (screen and PDF can't drift).

### R-6 Shared `physical_sweep_signature(params)` builder
One source for the sweep-cache signature + per-tab extras — enforces the "ADD IT TO THE SIGNATURE"
gotcha (P1-4 exists because the tuple lives in two places).

### R-7 One "installed-pump vs latest-test actuals" helper
Three copies (batch_run ×2, power_fluid_range) + two local pump-at-date re-derivations
(`jetpump_solver._pump_at_test_date`, `ipr_viz._pump_label_at_date`). Use the library's
`get_pump_at_date`; fixes P1-11 once.

### R-8 One `compute_field_marginal_wc`
Resolve the two-implementations split (P1-30) deliberately: pick semantics, delete the other, keep
the name unique.

### R-9 Shared solver-context builder in `scotts_tools/_common`
pf_scenario / jp_calibration / jp_fric_trend / jp_washout each repeat pump→chars→vogel→config→
objects→coefs with different fallbacks (~100 lines collapse; makes tenure handling uniform,
fixing P1-29 in one place). Also: `chars_num`/`chars_is_sch` helpers (3 copies),
`lookback_window(months)` (5 copies), pad-value-grid renderer (2 copies).
`pf_scenario._discharge_residual_fixed_rate` is a hand-copied solopump mirror that already lacks
the bracketing fallback — consider a `fixed_rate=` hook in the library (upstream-flagged).

### R-10 Dead code deletions (~1,000+ lines)
- `assembly/optimization_analyzer.py` — entirely dead (~304 lines, zero callers) **but modified in
  the current working tree — confirm intent before deleting**.
- `network_optimizer.py`: `validate_allocation` (dead + wrong), `create_well_template_csv`,
  `load_wells_from_csv`, `get_calibrated_results` (duplicate of `calibration.apply_calibration` —
  keep one). Keep `validate_well_config` and WIRE it (B-2).
- `network.WellNetwork` class (only module-level `optimize_jet_pumps` is used).
- Legacy BHP chain: `databricks_client.load_tag_dict` / `query_bhp_for_well_tests` /
  `well_test_cache._cached_bhp_query` / `bhp_dict.csv` + their test classes (delete together);
  `well_test_processor.merge_tests_with_bhp` (no app callers).
- `ipr_viz.py`: `create_ipr_grid_plotly`, `create_rp_comparison_chart`,
  `create_qmax_comparison_chart` (~290 lines), unused `HAS_PLOTLY` flag.
- flow: dead Cunningham helpers (`jetflow.py:359-435`), dead Ros pattern machinery
  (`twophase.py`), `pvt/deadoil.py` stub.
- GUI: `hpi_pdf_bytes` remnant, `_trigger_browser_download` shim (step2→step4), `sidebar._is_finite`
  (duplicate of `utils.is_valid_number`), `uw_water_key`/`uw_current_jp_map` write-only keys.
- `_normalize_well_name` triplication (well_test_client / well_sort_client / inline in
  well_test_processor) → one canonical helper.
- Vogel math triplication (ipr_analyzer fit / step2 re-derivation / optimization_viz inverse) →
  one `vogel.py`.

### R-11 Repo hygiene (delete/fix list)
- Delete: `_ul` (0 bytes), `temp/` one-shot smoke scripts + `hpi_backtest_latest.json`,
  `tools/hpi_probe_*.json` + `hpi_chain_GHIJ.json` (keep `hpi_backtest_probe.py`),
  `data/18Jan26 MPB-35...xlsx` (554 KB), `data/well_data_grid_plot*.png` (1 MB),
  `JP_rec_flow.png` (root, unreferenced).
- Keep `data/`: `hysys_*.json` (tests), `jetpump_dimensions.json` (pump_equivalent),
  `jetpump_history.xlsx` (app.py fallback), `excel_into_json.py` (their generator).
- `pyproject.toml`: `[tool.pytest.ini_options] testpaths`; fix `packages.find` excludes when
  packaging is next touched (tests/ currently ships in the wheel; `woffl/jp_data` wouldn't).
- **Do not move/rename `tests/harness_cases.py`** — the deployed app imports it
  (`scotts_tools/test_harness.py`).
- `well_test_cache.py`: add `max_entries` to the 24 h caches (6 GB shared box).
- `tests/test_well_sort_client.py` (untracked): good — commit it alongside the well_sort_client
  soft-fail change it verifies.

---

## Test coverage priorities (pre-refactor safety net)

Coverage map summary: pvt/geometry/flow/assembly-solvers/optimizers **good**; GUI glue + solver
seed-sync + header engine **good/partial**; **pad plant models, pad pages, all workflow_steps,
well_review_store, Triage decision logic, pf_scenario/jp_calibration/jp_fric_trend/jp_washout
compute cores: ZERO tests.**

Highest-value additions, in order:
1. **Pinned-value tests for `s/i/m_pad_plant.py`** in the style of `test_cfp_plant.py` (anchor to
   the SCADA-validated points + datasheet values in `build_moosepad_pump_curves.py`'s docstring),
   plus a regen test asserting the build script reproduces the committed M-Pad CSVs.
2. **`well_review_store.py`** round-trip tests (the `_FakeSt` pattern from `test_solver_ipr_sync`)
   — would have caught P0-2 — including CSV load/save and `to_well_config` bounds.
3. **`pad_optimize.py`** (post-R-1): fixed-point convergence, over-capacity, marginal-WC
   enforcement, run-signature invalidation.
4. **Triage decision functions** (post-R-4 extraction): WC-basis consistency, outlier protection
   on both oil and water deviations.
5. **Import/smoke test for every GUI page module** + a stub call of `_batch_automatch_inputs`
   (would have caught P0-1 — the 565-test suite couldn't see it).
6. `reconcile_wells` (post-B-1) end-to-end drop accounting.

---

## Verified-clean list (hunted for, not found)

No kaleido/`to_image`/`write_image` anywhere; downloads are single-click via the shared
`autodownload`; all `st.data_editor`s follow the stable-frame pattern; JPCO set-to-set tenure
correct in `jp_history`, the Solver strip, jp_fric_trend, jp_washout (violations only in
jp_calibration/pf_scenario/batch-run-calibration, flagged above); `run_jetpump_solver` still
re-raises `ThroatEntryNoSolution`; sidebar single-source-of-truth holds across Solver/Batch/PF
Range/pressure profile/pdf_export; SEED_BOUNDS clamping honored on all seed paths except
`sidebar.py:209` (`default_pad_pf`, currently in-bounds) and header-impact scenario restore;
ProdXV symmetry consistent; `worker_ceiling()` honored by all parallel paths; databricks_client
thread-local connections + token cache sound; solopump local patches otherwise correct (bracket
re-orientation, bit-identical fast paths, stale-rates guard); oil-vs-liquid qwf chain consistent
in header-impact and (except the P0-2 clamp region) the pad store; `.env` never committed; no
hardcoded secrets.
