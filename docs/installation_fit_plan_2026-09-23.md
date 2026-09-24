# Fitting a well across all its pump installations: plan (2026-09-23)

User request: fit each well across all of its pump installations and the
tests that belong to each, so the user can see how the model chose its fits,
and use current math and optimization methods to make it fast.

**Status:** phases 0-3 were built the same day. See the
[delivery record](installation_fit_delivery_2026-09-23.md) for what was
built, where it differs from this plan, measured timings and the benchmark.
This plan is kept as the design rationale.

"Measured" below means timed on live data on 2026-09-23 (read-only,
`WOFFL_MAX_WORKERS=1`, this laptop). "Estimated" is arithmetic from those
timings. Phase 1 must replace every estimate with a measurement.

## 1. What the data and evidence allow

These facts set the design. The sources are in the linked documents.

- **Most installations cannot be fitted on their own.** Over the two-year
  preflight window ([preflight v2](pump_lifetime_preflight_v2_2026-09-11.json)):
  - the median well has 3 installations with tests and 11.5 eligible tests;
  - the median installation has **2** eligible tests;
  - the median PF-pressure span within an installation is **1.7 psi**;
  - only 143 installations have at least 3 eligible tests, and 29 wells have
    at least two configurations.

  The current single-installation fitter refuses anything with fewer than
  10 points or less than 200 psi of PF-pressure spread.
- **Pump losses are second order; the IPR is first order.** In the PETEX
  comparison, switching to reference losses moved the pump intake by a mean
  of 14 psi, while the IPR alone accounted for a median 13 of the 23 points
  of oil error ([PETEX findings](petex_findings.md)). On B-30 and B-39 the
  curve misses oil at measured BHP by 25% and 46%
  ([fixed-IPR investigation](well_match_diagnostic_2026-09-12.md)). No pump
  coefficient can make up for that.
- **The discharge deficit is structural.** E-42 is 195 psi short, J-29
  350 psi and E-48 398 psi, and return friction explains little of it. Both
  engines share the deficit. Loss coefficients fitted to hide it would only
  mislead.
- **Changeout responses are the decision signal, and today they are wrong.**
  Both engines get the direction of the BHP/oil change wrong on 2 of 3 size
  changes (E-42 11C→13C, B-28 13B→13C). The user's stated objective is
  reliable pump-size decisions at the marginal WC, so these events matter
  more than level fit.
- **Constraints the fitter must respect** (AGENTS.md, standing decisions):
  - One oil IPR per well, with no automatic shifts. Refitting that one
    curve is an explicit user action.
  - No Mach multiplier.
  - Tracker diameters are nominal, not wear.
  - No unbounded gauge offsets.
  - Replacements use reference coefficients.
  - Scoped identity: well, nozzle/throat, Date Set, physics model and
    well-model fingerprint.
  - One `pump_calibration_v1` record of 500 characters or fewer per well.
  - Medium compute.

**Consequence:** the fitter must pool information across installations,
check identifiability instead of assuming it, test the IPR before it fits
anything to the pump, and be scored on changeout responses as well as
levels.

## 2. The statistical model

### Observations

For test *t* in installation *i*, the measured outputs are
`y_t = (BHP, oil, PF rate)`. Each is optional, and a missing value
contributes nothing.

The inputs come from the same test: WHP, PF surface pressure, WC and GOR.
Historical replay already uses measured WC/GOR with the fixed oil IPR
(`pump_match.py:195-220`).

The steady daily points already built for the current installation
(`calibration_points.points_for_well`, 5-day median filter, WC/GOR from the
nearest test) are extended to past installations. They carry BHP and PF
only. They are the only source of PF-pressure spread inside most
installations.

### Parameters

| Level | Parameters | Prior / treatment |
|---|---|---|
| Well | Vogel `qmax` of the one oil IPR (ResP held) | Held at the saved curve by default; refit only by explicit toggle (decision D1) |
| Well | Shared pump behaviour `phi = (kth, kdi)` | Weak prior centred on reference (.30, .40), bounded as today |
| Installation *i* | `log fnz_i` (effective nozzle area) | Shrunk toward 0: `N(0, tau_f^2)` |
| Installation *i* (model M3 only) | Offsets `(dkth_i, dkdi_i)` | Shrunk toward 0: `N(0, tau_k^2)` |
| Fixed | ken = .03, knz = .01, `mach_crit` = 1 | Not fitted: ken rails on pinned wells, and knz and fnz separate only through the entry area |

The nozzle-area term is per installation because PETEX found 27 of 39
measured PF rates above ideal-nozzle capacity (median +2.1%), and PF rate is
the best-identified channel. The prior widths `tau` come from the data by
empirical Bayes (below), not from hand tuning. This is **partial pooling**:
an installation with 2 tests stays close to the well-level behaviour, and
one with 40 tests and a wide PF span is allowed to differ.

### Objective

The estimate is the MAP of the posterior: a robust, regularized nonlinear
least-squares problem.

```
min over theta:  sum_t sum_c rho( (m_c(theta; u_t) - y_tc) / sigma_c )        level terms
               + sum_(a,b) rho( (dm(theta; a,b) - dy(a,b)) / sigma_d )           response terms
               + sum_i ||delta_i / tau||^2  +  ||(phi - phi_ref) / s_phi||^2     priors
```

- `rho` is a Huber loss, as in `calibrate_multipoint`.
- The `sigma_c` are measurement uncertainties agreed before fitting
  (starting points: BHP 50 psi, which includes the ±15 psi from the 40 ft
  gauge datum; PF 5%; oil max(10 BOPD, 10%)). These are the multipoint
  fitter's existing scales, so results are comparable.
- **Response terms** are paired differences:
  - within an installation, the existing dBHP pairs at least 100 psi apart
    in PF pressure;
  - **across each changeout**, the last steady tests before the Date Set
    against the first after, excluding the installation day and a 3-day
    embargo.

  Differences cancel most gauge-datum and IPR-level bias, and they measure
  directly what a pump decision depends on.

### Model ladder

The same machinery fits four nested structures. Every rung shares one IPR.

| Model | Pump parameters | Answers |
|---|---|---|
| M0 | Reference losses everywhere | How far does the IPR alone get? |
| M1 | One well-level `phi` for every installation | Is there a stable well/pump-family behaviour? |
| M2 | M1 plus a shrunk `fnz_i` per installation | Do installations differ in effective nozzle area? |
| M3 | M2 plus shrunk kth/kdi offsets | Do installations differ in wear/mixing? |

**Selection uses forward-chaining grouped cross-validation: never training
error.**
- Train on installations 1..k and predict installation k+1 in full,
  including its changeout response, with the 3-day embargo.
- Held-out predictions of a new installation use the prediction the
  standing decision allows: reference losses plus the well IPR, plus the
  model's well-level `phi` only in the experimental comparison (D2).
- The scoring rule is fixed before any run: summed standardized held-out
  error over BHP, oil, PF and changeout deltas.
- AICc/BIC are reported but do not decide.
- The simplest model within one standard error of the best CV score wins
  (the 1-SE rule), because small samples favour parsimony.
- Selection is nested: the scored benchmark wells never tune the rule
  (as [optimization fitting status](optimization_fitting_status_2026-09-12.md)
  requires).

### Gates before any pump parameter is fitted

1. **IPR consistency.** Compare oil at measured BHP against the curve. If
   the median miss is above 20% (to be tuned in Phase 1), pump parameters
   are not fitted. The UI says the IPR is the problem and offers the
   explicit one-curve refit. Otherwise the pump terms absorb IPR error and
   look like wear, which is the B-39 trap.
2. **Identifiability.**
   - Scale the Jacobian and take its SVD. Compute the collinearity index
     (Brun et al., 2001) for each candidate parameter subset.
   - A parameter whose direction is weak or collinear (index above about
     15, or its posterior barely narrower than its prior) is frozen at its
     prior. The UI shows it as "not identified" rather than as a number.
   - This subset selection replaces today's hard 10-point / 200-psi refusal.
     Well-supported installations still get individual parameters; thin
     ones stay at the well level.
3. **Regime.** A sonic or pinned result (`psu = psu_min`) has zero
   derivative in kth/kdi. Such tests enter as **censored observations**:
   the model only has to predict BHP at or above the measured value. They
   are never plotted as fits. Feasibility-edge returns with a large closure
   residual (the L-06 class) are flagged, not averaged in.

### Uncertainty

A Laplace approximation at the MAP gives the covariance
`(J^T W J + P)^(-1)`, with P the prior precision. It is almost free,
because the Jacobian already exists. It provides:
- parameter intervals;
- prediction bands, by the delta method on the same Jacobian;
- the empirical-Bayes estimate of `tau`, by maximizing the Laplace marginal
  likelihood over one or two scalars.

Intervals are labeled "model-conditional". Structural error (the discharge
deficit) is outside them, and the UI says so.

## 3. Speed

**Measured cost:**
- One `jetpump_solver` call takes 10-40 ms: about 5 `discharge_residual`
  evaluations at about 5 ms each (MPB-28 13C, MPS-05 9C, MPE-42). Building
  the input objects takes 0.1 ms.
- In one residual, about 57% of the time is the Beggs-Brill return-column
  march and about 19% is gas Z-factor (DAK).
- Today's multipoint fit (Nelder-Mead over 4 variables, up to 3 passes)
  takes about **3.2 min for one installation of about 24 points**.

### 3.1 Algorithms, highest gain first

1. **Trust-region Gauss-Newton / Levenberg-Marquardt instead of
   Nelder-Mead.** Use `scipy.optimize.least_squares(method="trf")` with
   bounds, `x_scale="jac"` and a Huber loss, and pass the prior rows as
   extra residuals. It uses the least-squares structure. Problems of this
   size converge in about 5-15 Jacobian evaluations, where Nelder-Mead
   takes hundreds of function evaluations and scales badly with
   dimension.
2. **Jacobians by implicit differentiation, not by re-solving.** The
   forward model is the root of `R(psu; theta, u) = pdi_jp - pdi_of = 0`.
   At the converged root, the implicit function theorem gives

   ```
   d psu / d theta = -(dR/dtheta) / (dR/dpsu)
   d oil / d theta = IPR'(psu) * d psu/d theta + (direct oil term, zero for pump params)
   d qpf / d theta = partials of the nozzle equation at pte(psu)
   ```

   Each partial is one residual evaluation at a **fixed** psu. Nothing
   re-solves. Per test and iteration:

   | Approach | Residual evaluations per test and iteration |
   |---|---|
   | Finite differences over 5 parameters | 6 solves, about 30 |
   | Implicit differentiation | 1 solve (5) + dR/dpsu (1) + parameter groups (2-3), about 8-9 |

   That is roughly **3.5x fewer**, before the caching below.
3. **Block sparsity and coloring.** An installation's parameters touch only
   that installation's tests. Perturbing "fnz for every installation"
   together gives all their columns in one pass (Curtis-Powell-Reid
   grouping). Per-test Jacobian cost therefore does not grow with the
   number of installations. `least_squares(jac_sparsity=...)` receives the
   block-arrow pattern.
4. **Newton polish removes solver jitter.** The root stops at 5 psi /
   10 psid, which would put psi-scale noise into the residuals and stall
   LM. The IFT step already computes `dR/dpsu`, so one Newton step
   `psu <- psu - R/(dR/dpsu)`, confirmed by one residual, lands the root
   to well under 1 psi. The shared library is not changed.
5. **Equation-error start, output-error finish.** At the *measured* BHP and
   rates the pump balance needs no root solve at all:
   - `discharge_residual(psu_measured)` is one about 5 ms evaluation;
   - the return column at measured rates is computed once per test and
     cached, since it contains no coefficient;
   - each coefficient enters its equation affinely at a fixed state
     (`jetflow.py:24-31, 257, 610`), so this start-up problem is nearly
     linear least squares.

   It solves in milliseconds and starts LM close to the optimum
   (typically 2-4 iterations instead of about 10). It also gives an
   instant first view of each installation's implied losses.
6. **Reuse the missed cache.** `calibrate_multipoint._solve_all` does not
   run inside `entry_energy.scoped_paths`, so the entry
   `MaterialPath`, about (pres − 60)/10 PVT condition calls, is rebuilt on
   every solve. Run the whole fit inside one scope with one `ResMix` per
   (WC, GOR). This is a quick win for today's event calibration as well
   (Phase 0).
7. **Staged residual (only if profiling calls for it).** kth and kdi leave
   the entry, nozzle and return column unchanged at fixed psu. A
   `discharge_residual` that can return its stages would make those two
   columns nearly free. That is a shared-library change: tag it, register
   it in `upstream_sync.md` and name a regression test. Do it only if
   Phase 1 profiling shows the kth/kdi columns matter.
8. **Cross-validation without full refits.** Warm-start each fold from the
   full-data MAP, which takes 1-3 iterations. Screen rungs with
   approximate leave-group-out: one Newton step from the full fit with
   installation k's rows removed, reusing the same `J`. Run exact refits
   only for the selected rung and for what is displayed.
9. **Parallelism and caching within Medium.**
   - One `server.pool` batch per LM iteration, holding every test of the
     well, split into chunks. Use `worker_ceiling()` and never a hardcoded
     pool size.
   - The fit is one heavy background job (`WOFFL_MAX_JOBS=1`) with
     `jobs.set_progress` cancel points between iterations.
   - Results are cached immutably, keyed by test `wt_uid`s, tracker rows,
     well-model fingerprint, physics model and request. Nothing is
     pre-warmed: it runs on demand, which keeps warehouse and CPU cost
     where it is today.

### 3.2 Budget (estimated; to be measured in Phase 1)

Assumptions:
- Per observation and iteration, about 9 residual evaluations × 5 ms, so
  about 45 ms.
- Two workers.
- A median well has about 12 tests plus about 25 steady days, so about
  40 observations.
- A p90 well has about 57 tests and 8 installations, so about 150
  observations.

| Work | Median well | p90 well |
|---|---|---|
| Equation-error start (all rungs) | < 1 s | about 2 s |
| One LM iteration | about 1 s | about 3.5 s |
| M0-M3, about 4 iterations each after the start | about 16 s | about 1 min |
| Forward-chaining CV (warm, about 2 iterations × folds × 4 rungs, approximate screen) | about 20 s | about 2 min |
| **Total, one well, all rungs and CV** | **under 1 min** | **about 3 min** |

For comparison, today one installation takes about 3.2 min on the same
kind of well. If the budget is missed, trim in this order: approximate
leave-group-out only for M3, drop M3 on wells whose identifiability gate
already rejects it, and cap the daily points per installation.

## 4. Data assembly (reuse, do not rebuild)

- **Installations and assignment of tests:** reuse `pump_match.assemble`.
  It already provides set-to-set tenure, same-size changeouts as separate
  installations, the installation-day exclusion, duplicate/undated
  blocking, direction and `pf_source` agreement, and nominal-spec conflict
  flags. `common_ipr.py` and the client `buildTimeline` duplicate this
  logic. The fitter should not add a fourth copy.
- **Tests:** use the 24-month warmed window from `tests.fetch_all_well_tests`
  (60 months as an explicit option), with the `trainable` eligibility rule.
- **Daily points:** generalize `calibration_points.points_for_well` from
  "current era" to "any era", with the same filters and caps.
- **Excluded tests:** the Solver's per-well exclusions live in the browser.
  The request carries `exclude_wt_uids`, the same field the IPR fit already
  honours, so the fitter and the chart agree.
- **Stable test IDs:** the extended-test payload lacks `wt_uid`, WC, GOR and
  WHP (`optimization_fitting_status_2026-09-12.md:86-89`). Use the
  `fetch_all_well_tests` rows, which have them.

## 5. Persistence (no new schema)

- **Only the current installation's fitted values are saveable**, through
  the existing `pump_calibration_v1` save:
  - the same scope checks, fresh tracker read and fingerprint;
  - the same 500-character budget;
  - `q` gains a compact tag giving the rung and number of installations,
    for example `"mi":[2,4]`.

  A reviewed save is still an explicit user action on a completed job ID.
  Client coefficients are never trusted.
- **Historical installations' and well-level parameters are display-only.**
  They are recomputed from the cache and never written. That keeps "a saved
  pump fit is active only for its exact installation" intact.
- **Replacements keep reference coefficients** (standing decision). The CV
  measures whether well-level `phi` predicts a *new* installation better
  than reference. That is evidence for decision D2, not a silent change.
- **An IPR refit is saved only through the existing well-input save.**

## 6. What the user sees

The fit lives in the existing Production History strip (Solver and JP
History) as a new mode, **"Fit across installations"**, next to Every test,
Refit next pump and Refit same pump.

1. **Timeline.** Installation bands as today. Solid lines are in-sample
   fitted BHP, oil and PF; dashed lines are the held-out prediction each
   installation received before it was seen. Pinned and edge tests use a
   distinct marker. Failed solves are gaps, never zeros. The chart uses
   `ChartPanel`, SVG and markArea/markLine carriers, and follows the chart
   rule.
2. **Installation table.**
   - One row per installation: pump, dates, test and daily counts, PF span;
     fitted `fnz_i` (and kth/kdi under M3) with intervals, or "not
     identified — using well level".
   - A **shrinkage bar**, showing how far that installation's own data
     moved it from the well level.
   - Fit and held-out scores kept separate.
3. **"Why this model" card.**
   - The ladder M0-M3 with training scores, held-out CV scores and AICc.
   - The winner and the 1-SE rule that picked it.
   - The gates that fired, for example "IPR misses oil by 31% at measured
     BHP; pump terms not fitted."
4. **Changeout panel.** For each changeout, the measured ΔBHP/Δoil against
   the model prediction, with a direction-correct tick. This is the
   decision check.
5. **Identifiability detail**, collapsed: the parameter correlation matrix,
   collinearity indices, and an optional on-demand profile-likelihood
   sparkline for any parameter (a bounded extra job).
6. **Actions:**
   - "Save current installation fit" (existing endpoint path; disabled with
     the reason when a gate fails or the rung is M0);
   - "Refit the one IPR" (explicit, goes to the existing save flow);
   - links from optimizer rows showing whether their recommendation is
     backed by this well's validated installations.

Every view is labeled retrospective or held-out, and model-conditional.
`validated_for_sizing` stays false until the acceptance criteria in section
7 pass on the benchmark set.

## 7. Verification and acceptance

- **Recovery on synthetic data.** Generate multi-installation data with the
  *real* solver from known parameters plus noise at the agreed sigma, then
  check:
  - parameter recovery;
  - interval coverage close to nominal;
  - that the identifiability gate freezes what the design cannot identify
    (for example, a 2-test installation with no PF span);
  - that the IPR gate fires when an IPR bias is injected.
- **Derivative checks.** Implicit-differentiation Jacobian against central
  differences with polished roots, per regime (non-sonic, sonic, edge).
- **Determinism.** The same inputs give the same fit, and pooled and serial
  runs are bit-identical (the pool rule).
- **Benchmarks.** The six frozen benchmark wells (B-30, B-37, B-39, E-42,
  F-107, F-73) and then the 29 multi-pump candidates, compared with:
  - reference losses;
  - today's current-installation multipoint fit;
  - the existing chronological challenges
    (`pump_history_benchmark.chronological_challenges`).
- **Acceptance (fixed before any benchmark run, proposed; user to confirm):**
  - held-out BHP RMS no worse than reference on the frozen set;
  - changeout direction correct on at least as many events as reference;
  - held-out oil error no worse than the IPR-only M0;
  - no rung chosen whose CV gain is within 1 SE of a simpler one.

  Failing wells are reported, not tuned away.
- **Offline suite and fixtures.** Mock the data layer as in
  `test_web_optimizer_runs.py`. No live reads in CI, and never a
  production write.

## 8. Phases

| Phase | Deliverable | Exit check |
|---|---|---|
| 0 | `scoped_paths` around the existing multipoint fit; measure before/after on MPE-35 | Same coefficients within tolerance, measured speedup recorded |
| 1 | Pure compute core `server/services/installation_fit.py`: data assembly (reusing `assemble`), equation-error start, IFT Jacobian + Newton polish, LM/TRF with priors, M0-M2, gates, Laplace uncertainty; offline tests incl. synthetic recovery and derivative checks | Section 7 synthetic checks pass; timings measured on MPB-28 / MPE-42 / MPS-05 |
| 2 | Job + API (`POST /wells/{name}/installation-fit`, poll/cancel), pool batching, result cache, forward-chaining CV and 1-SE selection, M3 | Benchmark table on the six frozen wells; budget met or trimmed as in section 3.2 |
| 3 | UI mode, installation table, "why this model", changeout panel; browser QA with fixtures | Frontend tests and build; fixture screenshots |
| 4 | Save path for the current installation (`q.mi`), optimizer evidence links; 29-well report | Acceptance section 7 on the benchmark; user decision D2 |
| 5 (optional) | Staged residual library change, errors-in-variables for WC/GOR, profile likelihood | Only if phases 1-4 show the need |

Each phase is shippable on its own and changes no production data.

## 9. Out of scope, or ruled out

- A Mach multiplier, per-test or per-installation IPR anchors, automatic IPR
  shifts, and reservoir-pressure trends.
- Gauge offsets, unless a documented datum exists.
- Treating tracker diameters as wear.
- Choosing a hydraulics model per date from its error. The hydraulics model
  is a fixed input of the fit identity; comparing models means one fit per
  model.
- Fitting loss coefficients to hide the discharge deficit. Deficit
  diagnosis (component closure, datum, P1 vs wellhead) is separate work,
  and the fit reports the residual it cannot explain.
- Hierarchical pooling *across wells*, for example a fleet prior per pump
  size. It is the natural next step once single-well pooling is validated,
  and the model extends to it without redesign.

## 10. Decisions for the user

- **D1. IPR in the fit:** hold the saved curve by default and offer the
  one-curve refit as an explicit toggle (proposed), or refit the one curve
  by default. Both keep one IPR.
- **D2. Replacements:** keep reference coefficients (standing decision) and
  only *report* whether the well-level behaviour predicts new
  installations better; or allow a validated well-level `phi` to seed
  replacement candidates on that well.
- **D3. History window:** 24 months by default (proposed), with 60 months
  available on request.
- **D4. Acceptance thresholds** in section 7: confirm or change them before
  the benchmark runs.
