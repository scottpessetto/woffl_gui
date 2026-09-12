# Well fitting, sensitivities and pad decisions: review and next implementation

Reviewed September 12, 2026, against local revision `ad96e7a` and the recorded
September 8-12 investigations. This is a source review with bounded offline
diagnostics. The findings below remain open; no application behavior, production
data or deployment was changed during this review.

The main recommendation is to make fitting, historical comparison, sensitivity
studies and optimization evaluate the **same versioned well model**. Then assess
whether proposed changes remain worthwhile across credible model and operating
uncertainty. Improving a BHP level match alone does not establish an accurate oil
response or a reliable pump choice.

## Preserve the user's modeling contract

- One oil IPR describes each well across the selected history and pump changes.
  Keep it locked during ordinary pump calibration and composition sensitivity.
  An explicit **Refit well IPR** action may estimate one new common curve; time-
  or installation-specific IPR shifts require an explicit separate choice.
- Historical tests use their measured WC/GOR and contemporaneous operating
  controls. Actual oil, BHP and PF remain observations to compare with the model.
  Preserve raw values and explain exclusions; do not alter observations to
  improve the match or invent daily oil measurements from modeled BHP.
- Keep well properties separate from the fitted condition of an installation.
  Same-size replacements remain distinct; new hardware uses clean-reference
  assumptions unless independent evidence supports a different prior.
- Preserve the existing physics, numerical closure checks, scoped pump identity,
  pad water-stream definitions and Medium compute limits while repairing the
  workflow. Additional physics should follow a demonstrated component error.

## What works and where the paths disagree

The saved-input workflow, historical overlay, per-test composition, installed/
clean candidate distinction, shared job pool/cache and coupled S-Pad pressure
checks provide useful foundations. The existing numerical tests establish code
contracts; they do not establish field prediction accuracy.

| Current path | Well inflow/composition | Pump assumptions | Consequence |
|---|---|---|---|
| Every-test history | One saved/edited oil IPR, actual test WC/GOR | Clean reference for every installation | Faithful conditional replay, but cannot yet show the saved installed-pump calibration |
| Multi-point event calibration | Each point gets a local oil/BHP IPR; daily points borrow test information | Fitted installed losses/area | The training model differs from the model later used by optimization |
| Single-point fallback | Saved oil IPR and saved composition, test-day pressures | BHP-only fit | Retrospective composition can differ from the chosen test; remains provisional |
| Match sensitivities | Current sidebar model; WC changes also change oil inflow; one comparison test | Catalog/loss sweeps can bypass pump-scope rules | Best-looking rows need not reproduce after Apply or demonstrate a transferable fit |
| Pad optimization | One saved curve and current saved composition per included well | Installed fit or clean replacement | Requires consistent fitting, complete pad accounting and response qualification |

### Priority 1: repair model and result consistency

**1. Calibration must use the approved common oil IPR.**
`woffl/gui/fric_calibration.py:739` builds `_base_inflow` but the multipoint loop
constructs individual point inflows at `:797`. Its objective explicitly excludes
oil (`:711`) because measured oil/BHP already determine each local curve. With
point composition and other dependencies unchanged, changing the saved anchor
does not change that objective. Applying these coefficients to the saved well
curve in optimization answers a different question.

Use the fixed well curve with each observation's composition. Score independent
actual test oil alongside BHP and PF; daily observations should only constrain
what was measured. The single-point path already uses saved inflow
(`server/services/event_calibration.py:152`), but needs consistent test composition
and must retain its provisional status and pinned-fit refusal.

**2. Sensitivity scenarios must obey the same hardware rules as Apply and optimization.**
`server/services/sensitivity.py:486` and `:792` use `model_copy` without
cross-field validation. Changing nozzle/throat can carry the old installed loss
coefficients and area. A loss sweep can also change a clean replacement's losses.
Frontend Apply correctly resets those values (`web/src/state/params.ts:82`), so
the displayed sensitivity and applied model can differ materially.

The [offline probe](workflow_sensitivity_probe_2026-09-12.json) records all inputs,
predictions and source hashes. For a synthetic Custom well with installed
`ken=.2, kth=.7, kdi=.8, fnz=1.15`, changing nozzle 12 to 13 produced:

| Case | Sensitivity BHP / oil | Equivalent Solver inputs after Apply: BHP / oil |
|---|---:|---:|
| Catalog size change | 846.4 psi / 301.95 BOPD | 277.0 psi / 406.91 BOPD |
| Clean replacement, then sweep kth to .8 | 706.7 psi / 334.84 BOPD | 356.95 psi / 396.83 BOPD |

Both individual and combined sensitivity workers reproduce the scope bypass.
The after-Apply case was reconstructed from the frontend's enforced clean-pump
transition, not clicked in a browser during this audit. These are reproducible
software inconsistencies, not measured field errors. Use one canonical scenario
builder for sweeps, Apply and optimizer candidates; require equivalent outputs
for equivalent inputs and pump state.

**3. Freeze sensitivity results with their full inputs and comparison identity.**
`web/src/state/sensitivity.ts:31` stores the fired study's knob labels/count but
not its complete inputs or target test. The server result lacks that snapshot
(`server/services/sensitivity.py:1028`). `CombinePanel.tsx:272` renders old results
against current targets, while old scores/best-index remain. `TopRunsTable.tsx:42`
applies only swept fields to whatever sidebar values are current.

Retain the complete submitted model, test ID/date/conditions, source/model
revision and assumption mode. Display stale studies as such; Apply must either
restore the whole reviewed scenario explicitly or require a rerun. Preserve the
comparison test through Solver navigation. Do not turn a comparison target into
an IPR anchor automatically. `SensitivityPage.tsx:93` currently selects Solver's
comparison test or latest available, but takes its outcomes without assembling
all corresponding historical controls/hardware.

**4. Fit identity must include the well inputs used to obtain it.**
`server/services/pump_calibration.py:75` activates a fit by installation and
physics/hydraulics identity. The save check (`:127`) and compact record (`:152`)
have no well-input dependency fingerprint. Saving a new IPR/PVT value can leave
an old calibration active; a completed job using older well inputs remains
saveable. The new Save bar correctly persists inputs, but cannot itself solve
this model-version gap.

Fingerprint the normalized oil IPR and stable PVT/geometry/survey/PF-density
dependencies actually used, plus physics identity. Retain training-data and
control provenance. A changed live PF pressure alone should not invalidate a
historical fit. Reject stale jobs on save and visibly mark incompatible saved
coefficients stale on hydration. Keep a reviewable full model snapshot and use
compact hashes within the existing 500-character calibration ledger constraint;
do not introduce an unreviewed production schema change.

**5. Unavailable models must not become operating recommendations or spare capacity.**
`server/services/optimizer_runs.py:457` returns no pump for nonallocated wells;
`web/src/pages/optimize/RunPanel.tsx:275` renders every such case as **SHUT IN**.
The backend already distinguishes failed simulation from budget exclusion
(`woffl/assembly/network_optimizer.py:854`) but the result UI ignores its
reconciliation. Separately, hydration/invalid-config failures are skipped
(`optimizer_runs.py:226,239`), leaving pad plant accounting to the surviving
configs (`woffl/gui/pad_optimize.py:486,493,528`).

Create an expected-well manifest with explicit outcomes: modeled, held at
measured conditions, economically shut in, already offline, unsupported or
missing inputs. A producing unmodeled well must still have its load accounted
for, or the run must remain explicitly incomplete/exploratory. For S-Pad held
water participates in station flow/pressure closure; for I/M/E it consumes the
appropriate machine-water budget. A fixed measured contribution is only a stated
approximation over a changed header: use supported response bounds or withhold
qualification. Unknown PF cannot become known zero.

CFP already uses measured-anchor water deltas (`woffl/gui/cfp_moves.py:182,219`).
A constant omitted background load cancels from that pressure equation; retain
that method. Nevertheless, omitted wells lose their pressure/oil response and
possible decisions, so their assumed unchanged contribution must be explicit.

### Priority 2: make the well fit physically interpretable

**Establish observation quality before freeing fitting parameters.** Test
pressures are joined by day and PF uses daily maxima
(`woffl/assembly/well_test_client.py:45,54`). This avoids some dead readings but
does not prove that oil, composition, PF, WHP and BHP describe one steady operating
point. Review source timestamps, stability, allocation quality, circulation,
nominal hardware and gauge datum. Reuse a shared audited observation assembler.

Event calibration currently includes installation-day daily values
(`server/services/calibration_points.py:248,260,325`) where replay excludes the
ambiguous day. It can clamp WC and attach the nearest test from either time
direction (`:147,298`). Preserve raw composition, disclose attachments and their
uncertainty, and split training/holdout periods before nearest-test attachment or
centered filtering. Treat overlapping daily/test rows and pressure pairs as
correlated evidence, not independent extra observations.

**Use an oil-IPR consistency check before adjusting pump losses.** Plot actual
oil versus datum-consistent measured BHP with the approved oil curve, colored by
installation/time/WC. A fixed oil curve determines oil at any given BHP, so pump
losses cannot independently fix incompatible oil and BHP observations. This
diagnostic is computationally cheap and uses no full pump solve.

The current explicit IPR tools fit total liquid across tests and then select one
test's WC (`woffl/gui/ipr_anchor.py:413,437`; `woffl/assembly/ipr_analyzer.py:230`).
For a new user-requested common **oil** IPR, fit measured oil/BHP with reservoir
pressure independently supported and fixed by default. Release Pr only when the
user chooses and the data constrain it. Store the resulting curve through the
existing equivalent total-liquid reference anchor; never silently reinterpret
old saved curves or force every date onto its own curve.

**Diagnose which part of the model is inconsistent.** At selected misses, show
nozzle/PF delivery, available pump discharge, required return pressure, static
head, friction, entry/mixing/diffuser contributions, pressure-balance closure and
operating regime. Conditioning on measured BHP/rate is allowed for diagnosis but
must not be scored as an independent prediction. Retain the existing interior-
root numerical checks; branch stability/feasibility-edge behavior remain separate
questions from whether an equation solver returned a finite number.

**Fit the smallest identifiable parameter set across installations.** Share the
well's approved IPR/PVT/geometry assumptions. Keep installation parameters
separate and bounded around physically supported values. Use sensitivity
signatures, parameter profiles and multiple starting points to detect parameters
that trade off against one another. Do not release all pump losses, area, Pr,
WC/GOR, temperature and gauge offset simultaneously just because it lowers error.
Keep real geometry and measured fluid information locked unless explicitly
reviewed. Related wells can support common physics or uncertainty priors; they
must retain their own IPR and installation condition.

Weight BHP, oil and PF discrepancies by defensible measurement/model uncertainty.
Avoid double-counting liquid and oil or WC/GOR and their underlying phase-rate
measurements. Use robust residual handling with visible exclusions and retained
failures. NIST's [errors-in-variables calibration study](https://www.nist.gov/publications/errors-variables-calibration-dark-uncertainty)
supports accounting for uncertainty in calibration inputs and excess dispersion;
the well-specific weighting and covariance model here remain proposed work.

**Replay the actual model that will be used.** Add clearly selected clean-reference
and saved-calibrated-installation overlays. `pump_match.py:173,283` currently
forces clean losses everywhere, so saving a pump calibration cannot change this
overlay. Reconstruct historical calibration only for its exact installation and
compatible well-model revision; retain earlier records without transferring wear.

### Priority 3: make sensitivities answer engineering and decision questions

Provide distinct purposes with explicit invariants:

| Purpose | What varies | What stays fixed / what the result means |
|---|---|---|
| Explain a miss | A bounded, physically supported parameter or small parameter set | Same model revision and matched test conditions; joint BHP/oil/PF signatures identify possible causes |
| Composition response | Test/current WC and GOR scenarios | One oil IPR: adjust equivalent liquid anchor so oil deliverability does not move |
| Measurement uncertainty | Plausible oil/water/gas/pressure measurement errors | Raw data unchanged; derived WC/GOR remain coherent; any alternate well curve is explicit and common across dates |
| Operating decisions | Pump, well PF pressure/WHP and shared plant conditions | Installed/clean semantics, full pad coupling and the same well scenario for current and proposed plans |

The present WC panel intentionally holds liquid qwf fixed
(`server/services/wc_uncertainty.py:3,37`), as does the WC knob in Match
Sensitivities. That changes the oil IPR: at a 1,000 BLPD anchor, moving WC from
80% to 85% lowers anchor oil from 200 to 150 BOPD, before hydraulic response.
This can be useful as an explicit anchor-measurement scenario, but is not a
composition-only change under the user's fixed-oil-IPR constraint. Make the
distinction visible and use fixed oil IPR for the default composition response.

The current combined-study score is RMS fractional error across available
BHP/oil/liquid/PF targets (`server/services/sensitivity.py:711`). It is a
one-test ranking, not parameter identification. Where oil and liquid are linked
by the same fixed WC, their fractional residuals duplicate one another. Replace
implicit equal weighting with disclosed observation uncertainty/correlation and
multi-observation validation when using this as a well-fitting tool.

The current reachable flag independently compares each target with sampled
min/max (`sensitivity.py:1003`). `EnvelopeChart.tsx:221` then says that no
combination gets there when a target is outside the sampled range. A finite
grid cannot establish that claim for all unsampled combinations, and separate
metric ranges do not establish one jointly matching case. Label this a sampled
scenario envelope and separately report whether any single solved scenario
meets all selected tolerances. Retain failed cases and incomplete coverage.

Include uncertainty in credible WC/GOR, oil deliverability if explicitly enabled,
Pr if supported, current pump condition, hydraulic model, PF density, WHP,
header-to-well pressure loss and plant capacity/slope. Use correlated scenarios
when quantities share a measurement source. The existing +/-5-point WC sweep is
a sensitivity range, not a confidence interval.

A lower-priority parity fix is also needed: `SimParams.to_simulation_params`
still truncates fractional qwf/pwf/Pr/GOR/temperature/pressures
(`server/schemas.py:130-144`), while the optimizer/history WellConfig path retains
floats. Remove that divergence with a fractional-input parity regression. It
does not explain the large field errors recorded below.

### Priority 4: qualify pad choices, including uncertainty

**Use a matched current-model counterfactual.** Show measured current production,
modeled current hardware/conditions, and modeled proposed hardware/conditions.
Compute modeled gain from proposed minus modeled current under identical well
assumptions. The current UI juxtaposes optimized model oil with current test oil
(`RunPanel.tsx:204`); test oil is the median of up to five positive-oil tests over
six months, without installation/condition alignment (`optimizer_runs.py:288`).
Baseline model bias must not appear as a pump-change benefit. A no-change plan
must give zero modeled gain. Any measured-anchored projection requires explicit
response assumptions; bias cancellation is not guaranteed across hardware changes.

**Separate saved, matched and qualified.** The current green Ready state checks
saved inputs and an active/nonprovisional installed fit (`OptimizePage.tsx:35`).
Match Health is separate and is not consumed by ordinary JPCO runs. Report input
completeness, match quality, independent pressure-response evidence, cross-pump
evidence and applicable operating range separately. Missing evidence is unknown.
Pad/default beta must not be labeled as an independently reproduced well response.

**Validate whole changes, not randomly selected neighboring days.** Freeze model
selection and fitting inputs; hold out later pressure events, entire different-
size installations and then wells. Retain all eligible failures in coverage.
Score BHP/oil/PF levels and incremental oil/PF response. Measured-composition
lookback remains conditional retrospective validation; future WC/GOR must be
forecast or explicitly treated as uncertain inputs. A historically correct WC/GOR
does not make a forecast independently blind.

**Assess the same candidate plans across credible pad scenarios.** Show base and
downside gain, constraint violations, alternative pump ranking and regret from
choosing one plan when another scenario is true. A scenario win fraction is a
stability measure, not a calibrated probability unless scenario weights have a
defensible statistical basis. Reconcile all producing-well loads and re-solve
plant pressure/capacity for each case. Preserve I/S lift-water versus M/E total-
water economics. Keep the existing measured-anchor CFP semantics.

Start on Medium with cached finalist plans and a small representative scenario
set, then refine around unstable choices. This is an application design inference
from ensemble decision methods, not a validated jet-pump algorithm: see SINTEF's
[work on representative ensemble selection](https://www.sintef.no/en/projects/2018/digital-subsurface/).
Verify CFP finalists directly at their reported settled pressure to bound
interpolation error. Benchmark discrete search separately against small exhaustive
references; optimizing a biased model more exactly does not improve field accuracy.

**Use uncertain choices to prioritize measurements.** Rank wells by how much
their plausible uncertainty changes the pad plan/value and shared water demand,
not just their worst BHP error. Recommend the missing measurement or an engineer-
reviewed stable pressure-response test likely to distinguish the competing
models. Gaugeless wells can participate with broader uncertainty and explicit
assumptions; inferred BHP must remain identified as modeled evidence.

## Evidence already available

The [fixed-IPR investigation](well_match_diagnostic_2026-09-12.md) preserved a
seven-well September 8 snapshot with 392 tests, 384 eligible and 351 solved using
measured test WC/GOR. Its common-case comparisons after the numerical fix show:

| Well | Saved-composition to measured-composition BHP RMS | Oil median absolute error |
|---|---:|---:|
| B-28 | 168.7 to 215.3 psi | 39.6% to 17.2% |
| B-30 | 150.6 to 92.6 psi | 10.8% to 23.2% |
| B-39 | 158.8 to 177.0 psi | 140.5% to 70.9% |

These frozen retrospective results are not today's live well fits. They show why
improving BHP alone is insufficient and why actual composition is necessary even
when it increases error. No new fleet accuracy claim or rerun is made here.

NIST's [model validation guidance](https://www.itl.nist.gov/div898/handbook/pmd/section4/pmd44.htm)
also emphasizes residual patterns over reliance on a single R-squared statistic.
For this app, residuals versus date, pump, WC, GOR and PF pressure should accompany
aggregate metrics. Field acceptance thresholds should reflect measurement
repeatability and the size/cost of the proposed decision, not arbitrary universal
BHP or percent-error cutoffs.

## Concrete implementation order and acceptance checks

| Order | Deliverable | Required proof |
|---|---|---|
| 1 | Scenario/result consistency and honest optimizer outcomes | Size/loss sensitivity reproduces Apply and direct solver; stale studies cannot mix cases; failed-model and economic-shut-in labels differ; fractional inputs agree across paths |
| 2 | Fixed-IPR calibration, shared observation assembly and fit dependency identity | Every training solve uses one oil curve with actual composition; test oil is withheld from prediction inputs; changed well inputs invalidate old jobs/fits; installation days and invalid observations stay explicit |
| 3 | Well Fit workspace with explicit common-IPR refit, component and identification diagnostics | User chooses held/free parameters; actual data remain unchanged; preview and saved-calibration replay reproduce the version used in optimization; correlated/railed alternatives are visible |
| 4 | Complete pad baseline, uncertainty and finalist comparison | Every online well is accounted for; no-change gain is zero; held loads affect capacity/pressure correctly; scenario failures remain visible; finalists close and rankings are reported across scenarios |
| 5 | Blind event/pump/well validation and operating feedback | Predicted oil/PF changes meet decision-specific targets on data unused for model selection; observed outcomes update a versioned evidence record |

The proposed user sequence is **Review data -> Fit/lock the well -> Check history
and response -> Save the model revision -> Reconcile the whole pad -> Compare
plans and uncertainty -> Measure the outcome**. Save remains an explicit user
action; a completed calibration is not automatically a qualified recommendation.

## Review scope and artifacts

No runtime source changes, full test rerun, frontend build, live data access,
production write or deployment was performed in this review. The preceding
2,006-Python/18-frontend green baseline remains the latest full-suite result;
the newly identified gaps are not covered by that claim.

The bounded sensitivity probe used only synthetic Custom-well solves. Its local
runner is `build/workflow-review/sensitivity_scope_probe.py`; invocation from the
repository is `PYTHONPATH=.` followed by
`venv/Scripts/python.exe build/workflow-review/sensitivity_scope_probe.py`.
The durable [JSON artifact](workflow_sensitivity_probe_2026-09-12.json) records
complete reproducible inputs, outputs and code hashes. No existing field evidence
or frozen benchmark was overwritten. Findings and the implementation sequence
are proposed next work, not fixes already delivered.
