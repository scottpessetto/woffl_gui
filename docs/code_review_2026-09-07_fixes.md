# Review fixes — 2026-09-07

Implemented R01–R11 and O01–O02 from [the review](code_review_2026-09-07.md).
The earlier optimization fixes are also retained. Changes are local; nothing
has been deployed and no production properties have been written.

## Changes and reproduced outcomes

| Finding | Implemented behavior | Verification |
|---|---|---|
| R01, component conservation | Stock-tank oil mass, dissolved gas, phase density/Bo and volumetric/mass flow interfaces use one consistent component balance. | Specified 400 BPD water stays 400 across pressure; total mass stays 2.042793659 lbm/s. 48 WC/GOR/temperature cases cover eight pressures each. |
| R02, watercut precision | Hydration preserves full watercut precision for test, locked and saved values. | 0.974 reloads as 0.974 and retains 26 BOPD; 0.986 remains optimizer-eligible. |
| R03, candidate wear | Single-well sizing applies wear only to the selected baseline size, consistent with network sizing. | Installed 12B keeps factor 1.2; alternative 13B uses 1.0 in both paths. |
| R04, replacement wear | Saved parameter timestamps survive hydration; old/undated wear resets when an installation date establishes a new pump. | New 13B no longer inherits the old factor 1.2. Well-specific Mach calibration remains saved. |
| R05, fallback calibration | Saved wear, critical Mach and throat/diffuser seeds reach the fitter. Job fallback requires a known era and only selects in-era tests. | Nondefault parameters reach the actual fitter call; prior-pump tests are rejected. |
| R06, CFP convergence | Validate final pressure residual; use bracketed roots within available surface intervals if fixed-point iteration fails. Unconverged states cannot rank as feasible. | The steep synthetic surface now balances at 2,594.221 psi, residual below 1e-9 psi. |
| R07, evidence reference | Use paired observation pressure/BHP references, not the sweep ceiling; scale oil from each point's own model BHP and correct today's projection reference too. | Shared 3,000 psi evaluation returns 520 psi BHP for both search ceilings. Missing pressure references leave the model uncorrected. |
| R08, deployment dependency | Declare `python-multipart>=0.0.27`. | Installed environment supports application imports/upload routes; dependency probe confirms declaration. A fresh hosted deployment was not attempted. |
| R09, geometry | Reject zero annular clearance in both API and library validation. | Zero-area input rejected; positive clearance accepted. |
| R10, duplicate write retry | Write execution errors are not replayed; cursor cleanup errors cannot repeat completed work. Reads retain retry; writes can retry connection setup before execution. | Fake committed operation executes once; lost execution response also executes once. No database involved. |
| R11, polling recovery | Keep job IDs through transient errors; bounded retry/backoff and continued polling. Only confirmed 404/410 clears the ID. | QueryObserver integration test recovers a failed response and retrieves the original result. |
| O01, choke water budget | Allocate plant machine water throughout pressure anchoring, option ranking, trimming and contingency ladders. Report PF separately; UI budget/value labels follow water basis. | The option demanding 1,800 BPD against a 1,000 BPD cap is shut in instead of accepted. |
| O02, pricing tie | Preserve the economic optimum, then maximize oil among ties in MILP and CP-SAT. | Both choose feasible 80 BOPD rather than zero at lambda 1. |

The [original observations](code_review_2026-09-07_probes.json) remain unchanged.
[Post-fix observations](code_review_2026-09-07_after.json) record the corrected
behavior. The multipart probe deliberately blocks imports to demonstrate the
requirement; its expected error text is not an application startup failure.

## Validation

- Full Python suite: **1,746 passed**, two existing warnings.
- TypeScript checking and the production frontend build passed. Tracked
  `web/dist` was rebuilt. The existing large chart-bundle warning remains.
- Two Node polling tests passed: `node --test tests/jobPolling.test.mjs` from `web`.
- Independently recomputed discharge residuals for all 44 non-sonic E-41 batch
  cases remain within the existing 10 psid solver tolerance (maximum 9.067).
- Named regressions are in `tests/test_review_2026_09_07.py` and
  `tests/batch_test.py`; shared-library changes are registered in
  [upstream_sync.md](upstream_sync.md), patches 37–39.

The conservation change intentionally moves model outputs. In the existing
E-41 fixture, 16E oil moves from 38.37 to 34.795 BOPD, and 12X joins the sonic
set. Affected numerical reference assertions were updated after checking the
component invariants and discharge residuals. This does not establish field
accuracy, and stored fits should be reassessed against observations before
relying on their revised forecasts.

## Boundaries

The older critical-Mach energy-closure inconsistency and simplified constant
water PVT remain separate model-development work; this patch does not claim to
resolve or validate them. It also does not claim a measured runtime speedup.
The automatic water-price policy is unchanged; the fix resolves ties, not every
possible distinction between economic optimization and maximizing oil. CP-SAT
retains its documented integer quantization.

A same-size baseline represents keeping the installed pump. Evaluate a clean
replacement of that same size with nozzle area factor 1.0. Existing saved
friction/Mach values were not rewritten or recalibrated in production.
