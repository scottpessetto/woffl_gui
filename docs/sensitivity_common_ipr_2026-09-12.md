# Sensitivity scenarios and explicit common oil IPR

This delivery fixes sensitivity/Apply consistency and adds an explicit common
oil IPR candidate fit. Source and browser fixtures were verified locally; this
record does not establish deployment or any production database write.

## One oil curve, explicit uncertainty assumptions

Sensitivity and the Solver's WC uncertainty card default to **Composition
(keep oil IPR)**. Changing measured WC converts the liquid anchor as
`qwf_case = qwf_base * (1 - WC_base) / (1 - WC_case)`, preserving the oil-rate
versus BHP curve. GOR remains independently fixed unless the engineer varies it.
This is not an assumption of fixed operating gas rate.

**Anchor measurement (vary oil IPR)** retains the measured liquid anchor and
lets WC uncertainty change the inferred oil anchor. This is an explicit
alternative. The general sensitivity page also retains qwf, anchor BHP and
reservoir-pressure knobs, identified as deliberate curve changes. No scenario
automatically changes session inputs or saved values.

`server/services/scenarios.py` supplies the same resolved cases to one-at-a-time
and combined sweeps. Catalog changes and replacement cases use clean reference
coefficients/area. Installed-pump losses may still be explored on their own
installation. The previous code carried installed losses into catalog changes,
then Apply reset them and produced a different answer; this is corrected.
Fractional qwf/BHP/Pr/GOR/temperature/pressures/depth now survive conversion to
SimulationParams, matching the float convention in optimizer/history configs.

## Reproducible studies

A combined study returns its complete submitted request, test identity,
installation identity and WC basis. Each successful row includes the actual
resolved input delta, including automatic oil-anchor and clean-hardware changes.
Charts/error columns retain submitted targets. Old results cannot be applied
after inputs, test, installation or WC basis change. A legacy result lacking its
request snapshot must be rerun. Apply restores the same comparison test in
Solver; a comparison test is not automatically made an IPR anchor.

Ranking uses fractional BHP, oil and PF error; liquid substitutes when oil is
missing. Liquid remains visible as a diagnostic, without counting the same
fixed-WC rate error twice. This remains a single-test engineering shortlist,
not measurement-weighted calibration or independent model validation.

The sampled scenario envelope covers successful samples only. Separate metric
intervals do not prove a joint match, and failed or unsampled settings are not
bounded by the displayed extrema. Measurement-informed covariance, identifiable
parameter directions and joint statistical acceptance remain further work.

## Fit one oil IPR across pump history

The shared production-history control in Solver and JP History contains a
collapsed **Fit one oil IPR across pump history** panel. It performs no work
until **Fit candidate oil IPR** is clicked.

The candidate fits one positive Vogel oil Qmax from recorded oil/BHP across
training dates and installations. Reservoir pressure is fixed at the displayed
value. The displayed anchor BHP, WC and GOR are preserved; only qwf is proposed
for Apply. User edits to reservoir pressure require another explicit fit.

- A chronological holdout split is established on raw dated observations
  before physical cuts, user selections or fitting. A three-day embargo
  separates training from holdout.
- Each training date has equal total weight. A smooth robust loss limits
  outlier influence; its scale is derived from training dates only.
- Missing/duplicate identities, installation days, unknown installations,
  invalid composition/rates and BHP outside flowing conditions are excluded
  visibly. Missing/invalid and future dates are counted in notes.
- Training tests and installations can be excluded in the review table.
  These choices cannot remove held-out tests or move the split. Changing the
  window/split after inspecting holdouts makes the exercise exploratory.
- Candidate and current-curve oil errors are shown separately for training
  and held-out dates. They are conditional on **measured BHP**, not a forecast
  of operating oil/BHP. Forward history replay and independent pressure-response
  validation remain necessary before optimization.

**Apply candidate IPR** changes the session qwf only, selects edited-input
history preview, and establishes explicit common-curve/manual-anchor intent.
Automatic single-test fitting cannot overwrite it. The persistent Save bar
saves that curve and clears an existing single-test anchor pin, including when
Save is clicked from JP History or after navigating to Solver. A new well,
explicit context reseed, or explicit single-test fit/anchor choice clears that
session intent. Existing database write gates and append-only persistence are
unchanged. New optimization runs consume saved inputs through the existing path.
Solver initially shows **Oil only** after a common curve is applied, including
after navigation. Historical tests with different WC can then be compared on
the same oil basis; the user can still switch back to total liquid.

## Validation

Focused Python checks: **51 passed**, covering scenario hardware parity,
fractional inputs, fixed-oil versus anchor-measurement semantics, pool ordering,
failed cases, common-curve recovery across changing composition/pumps, raw split
isolation, holdout outcome isolation, date weighting, outlier influence,
invalid/future dates and excluded observations.

Browser fixture checks (all API traffic intercepted, no Databricks access):

- `tools/check_sensitivity_workflow_ui.py`: actual server scenario versus
  Apply/Solver agreement, immutable targets, WC modes, stale Apply prevention,
  and comparison-test handoff; zero browser errors.
- `tools/check_wc_uncertainty_ui.py`: both WC modes, collapse/cache, late
  response protection, incomplete ranges, clipping and narrow layout;
  zero browser errors.
- `tools/check_common_oil_ipr_ui.py`: explicit fit/Apply/Save, fixed reservoir
  pressure, stale-candidate blocking, training selections, and two intercepted
  saves clearing a single-test anchor from JP History and Solver; zero browser
  errors. Screenshots under `build/common-oil-ipr-*.png` are fixture illustrations.

TypeScript checking and focused Node state/study tests passed. The integrated
suite/build and deployment status are recorded in the final workflow handoff.
