# Well fitting and pad decision workflow — September 12, 2026

This implements the first workflow improvements from the
[well-fit review](well_fit_pad_workflow_review_2026-09-12.md). It does not establish
new field accuracy or deployment. Existing frozen field comparisons remain
historical evidence with their original inputs and model contracts.

## Engineer workflow

1. Open the well in Solver or JP History. Review the dated hardware, measured
   production tests and saved well inputs. The persistent **Save well inputs**
   panel identifies changed values and explicitly reports a read-only app.
2. Run **Every test (well fit)**. One oil IPR describes every installation;
   each test supplies its actual WC, GOR, PF pressure and WHP. Actual oil/BHP/PF
   are comparison observations, never individual inflow anchors.
3. If the oil curve needs revision, open **Fit one oil IPR across pump history**.
   The explicit candidate fit holds reservoir pressure fixed, fits one positive
   Vogel oil Qmax to measured oil/BHP, weights dates equally and limits outlier
   influence. Review training exclusions and later held-out dates. Apply changes
   the session's liquid anchor representation, preserving the chosen anchor BHP,
   WC and GOR. Run **Current edits (preview)** before **Save well inputs**.
4. Run **Calibrate to field data** after saving the stable well inputs. Pump
   calibration holds that same oil IPR fixed and changes only installed-pump
   parameters. Review BHP, actual-test oil and PF errors, bounds and response
   diagnostics. Apply previews the fit; **Save installed-pump calibration** is
   a separate explicit database action.
5. Replay with **Saved fit where valid**. The exact matching installation uses
   its saved coefficients and area. Other installations use clean reference
   losses. The installation table identifies which was actually used; a
   changed model or conflicting historical geometry produces a visible reason
   for using reference losses. **Clean reference** remains an explicit comparison.
6. Start a new optimization run. Review every expected well's outcome and the
   coverage manifest. Missing/failed models cannot count as zero-load shut-ins.
   Compare **Current pumps at plan header** with the proposed hardware; recent
   measured production remains separate context.
7. On a complete I/M/E JPCO run, open **Stress-test current and proposed plans**.
   Choose engineering ranges for WC, GOR and delivered header, optionally two
   joint cases. Compare the same fixed plans, feasibility, oil gain and regret
   against each other. No inputs are saved or pump choices substituted.

## Fixed oil curve and observations

Calibration previously rebuilt inflow from individual observations, which could
hide well-model errors inside point-specific oil curves. The multipoint and
single-test paths now preserve the approved oil curve. Per-test WC changes its
total-liquid representation by `qwf = approved_anchor_oil / (1 - test_wc)`;
the oil anchor and reservoir pressure do not move. GOR follows the test.

Actual tests contribute oil residuals and reported RMS oil error. Daily pressure
rows have no invented oil rates and contribute no oil score. Invalid composition,
installation days, conflicting circulation and unusable inputs remain exclusions.
Daily composition is attached to a measured test with its identity, date,
lag and future/past provenance. Tests replace daily rows on the same date;
repeated dates share level and response weight.

The common-IPR candidate partitions raw dated observations chronologically before
training exclusions, with a three-day embargo. Held-out observations cannot be
manually removed from that fit. Its oil error is conditional on **measured BHP**,
not a forecast of operating BHP. Tuning after seeing holdouts is exploratory.
Forward history replay and independent pressure/hardware changes remain necessary.

Historical response beta is labeled a well-history, pad or default diagnostic.
Agreement with a pooled reference is not independent validation. The installed
fit's review flag also covers missing oil evidence (fewer than three scored oil
tests) or oil RMS above 10%, alongside the existing BHP/PF/bound checks. These
are review thresholds, not universal field-acceptance standards.

## Saving and model identity

Version 2 calibration records retain the existing append-only comment context,
one gated INSERT and 500-character limit. Full-precision coefficients are stored
alongside a compact dependency fingerprint. The identity includes the normalized
oil curve, reservoir pressure, stable PVT/geometry, circulation, return model and
local survey contents. Installation identity still includes exact Date Set.

Measured test WC/GOR, live PF pressure and fitted pump coefficients are not stable
well dependencies. A composition change that preserves the same oil curve does
not invalidate it. A changed oil curve or stable input does. Old version 1 fits
remain in history but need explicit refitting under the fixed-IPR contract.

Start/Apply/Save reject mismatched fit inputs. Save checks a fresh tracker and
fresh characterization/saved-IPR reads; failed verification performs no write.
Same-process saves invalidate the existing caches. Reload and new optimization
runs use a fit only if its model and installation still match. No property IDs,
tables, write gates, warehouse size, worker limits or native thread limits change.

Save well inputs persists the displayed IPR plus supported composition, WHP,
temperature and bubble point. Other edited geometry/PVT/mode settings are named
explicitly. **Restore session-only settings** restores just those source values,
retaining the intended oil curve. Permanent changes to these settings require
updating their characterization/as-built source. A common-IPR candidate is saved
as a common curve and clears any single-test anchor pin, with that action shown
before the user saves.

Solver, replay and optimization also share the same geometry resolution. Missing
MD uses a survey TVD crossing, or an explicitly estimated field profile when no
survey exists. Measured MD is retained only if it agrees with the profile's TVD
within the existing five-foot tolerance. Conflicts, unreadable surveys and
unreachable TVD are explicit model failures; context and save controls remain
available. A deliberate Solver TVD edit remains a preview. This closes a prior
case where Solver used a deviated preset while optimization assumed MD = TVD.
Estimated geometry is still an assumption that needs verification.

## Sensitivities and pad decisions

WC sensitivities default to **fixed oil IPR**. The separate anchor-measurement
basis deliberately changes inferred deliverability and is named accordingly.
Catalog replacements always use clean coefficients, including same-size
replacements. Combined results carry their original request, targets and exact
scenario changes; Apply reproduces those inputs and refuses stale studies.
The comparison-test handoff is consumed once and does not pin a database anchor.
Fractional inputs are retained through the single-well conversion.

Combined ranking uses disclosed fractional BHP/oil/PF error; liquid replaces oil
when oil is unavailable. It does not count oil and its dependent liquid rate
twice. The finite grid is a sampled envelope, not a proof that no other combination
can match or a statistical confidence region.

Pad coverage distinguishes missing inputs, unsupported models, failed solves,
explicit offline wells and modeled economic shut-ins. Unaccounted online load
makes a pad result incomplete and exploratory, with whole-pad feasibility
withheld. The matched current-hardware counterfactual isolates modeled hardware
gain at the proposed header; it is not total operating uplift or a bias correction.

The stress tool compares two immutable server-job plans in at most nine stated
cases, with bounded wells and the existing pool/cache. WC/GOR changes hold each
oil IPR fixed. Plant constraints use I-Pad lift water and M/E total machine water.
Unknown and infeasible cases remain visible and are excluded from numeric gain
ranges. Case preference counts are stability diagnostics, not probabilities.
Regret is relative only to the other feasible plan and the source water price.
S-Pad and CFP need a coupled scenario evaluator and are explicitly unsupported
by this new control. CFP's existing measured-anchor delta formulation is retained.

## Remaining work and evidence limits

This is not a simultaneous multi-installation loss fit. Only the currently
verified installed fit is persisted/replayed; older installations use reference
hardware. A future shared fit should identify one well model plus adequately
supported installation-specific losses without transferring wear or using
per-point IPR shifts. Correlated/weakly identified parameter families still need
explicit diagnostics beyond boundary and residual checks.

Independent event/pump/well validation must split raw data before composition
attachment and centered filtering. Current daily calibration is retrospective.
Future WC/GOR, pressure losses, capacities and measurement uncertainty require
defensible ranges; the stress defaults are engineering assumptions. A common-date
measured pad anchor, coupled S/CFP finalist studies, multiple alternative plans,
and measured operating outcomes remain follow-up work. No new frozen field
benchmark was substituted for independent qualification.

## Verification

- Full Python suite: **2,116 passed**. Two older fallback fixtures were updated
  to include the newly required measured WC/GOR/PF inputs. New save/identity
  regressions mock database boundaries; legacy calibration test fixtures were
  tightened to reject unexpected warehouse reads.
- Frontend: **29 tests passed**, TypeScript check passed, production Vite build
  passed and `web/dist` regenerated after stopping the Vite QA server. The
  existing chart-vendor chunk-size notice remains.
- Fixture browser checks passed for production history, edit/preview/save,
  installed-pump scope, sensitivity Apply, both WC bases, common-IPR fit/Apply/
  unpin-save, and pad coverage/stress cases. API requests were intercepted;
  checks included stale results, cancellation, narrow layouts, and no page errors.
- Real synthetic numerical solves verify BHP/oil/PF/liquid parity across
  Solver, replay and batch for all three hydraulic models. This is consistency
  evidence, not field qualification.
- A read of the bundled `jp_chars.csv` found no geometry conflicts in its 70
  rows carrying TVD. This is a local snapshot check, not a current fleet audit.

No production property record or deployment was changed by this implementation
session. The preceding frozen field benchmark was not rerun or overwritten.
