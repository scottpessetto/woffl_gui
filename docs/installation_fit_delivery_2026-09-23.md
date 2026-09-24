# Fit across installations: delivery (2026-09-23)

Built from the [plan](installation_fit_plan_2026-09-23.md), phases 0-3,
checked locally only. Nothing is deployed, and nothing writes to production.
Benchmark data: [installation_fit_benchmark_2026-09-23.json](installation_fit_benchmark_2026-09-23.json).

## What was built

| Piece | Where |
|---|---|
| Compute core and job | `server/services/installation_fit.py` |
| Endpoints | `POST /api/wells/{name}/installation-fit`, `GET`/`DELETE /api/installation-fit/{job_id}` (`server/routers/history.py`) |
| Schemas | `InstallationFitRequest`, `InstallationFitJob` (`server/schemas.py`) |
| Shared history loader | `pump_match.load_history`, split out of `pump_match.run` |
| UI | Production History -> Compare -> **Fit across installations** (`web/src/components/InstallationFit.tsx`, adapter `web/src/lib/installationFit.ts`) |
| Tests | `tests/test_installation_fit.py` (13), `web/tests/installationFit.test.mjs` (7), plus a scope test in `tests/test_fric_calibration.py` |

**Phase 0.** `calibrate_friction_coefs`, `calibrate_multipoint` and
`gaugeless_match.match_test` now run inside one `entry_energy.scoped_paths`
scope, so a fit builds each fluid's throat-entry path once instead of on
every solve. Live MPE-35 multipoint fit, frozen inputs: 93.0 s before,
78.9 s after, with identical coefficients and RMS.

**Model.**
- One oil IPR per well. The saved curve is held by default. `refit_ipr`
  explicitly refits its single scale.
- `ken`, `knz` and `mach_crit` are held.
- The model ladder:
  - M0: reference losses.
  - M1: one well-level kth (and kdi, if separable).
  - M2: M1 plus a per-installation nozzle-area factor, shrunk toward 1
    (prior sd 5%, log scale).
  - M3: M2 plus per-installation kth offsets (prior sd 0.10).
- The objective is the MAP of a Huber-robust, prior-regularized least-squares
  problem, using the multipoint fitter's scales: BHP 50 psi, oil
  max(10 BOPD, 10%), PF 5%.
- A failed test costs a 3-sigma residual.

**Gates, in order.**
1. **IPR.** Pump terms are not fitted if the curve misses test oil at
   measured BHP by a median above 20%, unless the one curve is refitted.
2. **Information.** kth is held when its Fisher information over one prior
   variance is below 1 (sonic-pinned histories).
3. **Collinearity.** kdi is held when the kth/kdi Brun index is above 15,
   judged at the fitted point, not at reference. kth then carries the
   combined loss, and the UI says so.

**Selection.**
- Rolling-origin held-out refits split at each pump change and at the middle
  test of long installations, with a 3-day embargo, up to 6 folds.
- A new pump is predicted at the prior nozzle factor.
- The rule is the simplest rung within one standard error of the best
  held-out loss. With no held-out tests, the lowest AICc wins, labeled
  unvalidated.
- Each pump-change fold reports measured vs predicted BHP and oil change,
  with direction.

**Speed.**
- Levenberg-Marquardt with IRLS Huber on data rows.
- Implicit-differentiation Jacobians: d psu/d p = -(dR/dp)/(dR/dpsu), one
  `discharge_residual` per parameter at the converged root, and only for the
  parameters the rung uses.
- Newton polish of each root.
- Predictor-corrector continuation for LM trial steps. The optimum always
  gets a full solve.
- Pool batching, with a serial fallback.
- The implicit-differentiation Jacobian matches polished full-solve central
  differences within about 4-8% (tested).

## Differences from the plan

- **kth and kdi are not separable** from BHP, oil and PF in practice. The
  collinearity index was 280 on MPE-42 and 52 on MPB-39, and 23 at the
  fitted point on a synthetic well. The collinearity gate usually leaves one
  combined loss (kth).
- **M3 offsets are kth only** whenever kdi is held.
- **Not built yet:**
  - empirical-Bayes prior widths (the priors above are fixed and
    provisional);
  - daily pressure points for past installations;
  - censored treatment of sonic tests (they are fitted as ordinary
    predictions and flagged);
  - held-out oil scoring in the acceptance summary (per-fold oil error is
    shown in the UI);
  - the save path for the current installation (phase 4).
- **Budget.** A job takes 1-3 minutes on most wells that pass the gate. The
  worst case was 700 s: MPF-73 with the IPR refitted.

## Benchmark: 29 multi-pump candidates, live, read-only

These are the 29 candidates from the September 11 preflight, over a 24-month
window with Beggs-Brill and 2 workers. "Held-out BHP RMS" pools every fold's
held-out tests. Direction counts only clear changes (at least 25 psi or
10 BOPD).

| | Saved IPR held (default) | One IPR refitted |
|---|---|---|
| IPR gate fails (median oil miss > 20%) | 16 of 29 | 16 of 29 (the curve is then refitted) |
| Chosen model | M0 20, M1 7, M2 2 | M0 11, M1 10, M2 6, M3 2 |
| Pump rung beats M0 on held-out loss by more than 1 SE | 9 wells | 18 wells |
| Held-out BHP RMS, chosen vs M0, where a pump rung was chosen | better on 9/9; median ratio 0.77 (0.67-1.00) | better on 16/18, worse on 2 (MPS-45 144 vs 137, MPS-54 123 vs 93 psi); median 0.82 (0.47-1.31) |
| Pump-change direction, chosen vs M0, same wells | 32/54 vs 31/56 | 63/103 vs 58/99 |
| Job time | median 8 s (gated wells stop early), max 130 s, 16 min total | median 86 s, max 700 s, 53 min total |

**Reading it.**
- **Level prediction improves, pump-change direction does not.** Pooled
  losses predict later BHP better on the wells where they are chosen. The
  direction of the response to a pump change barely moves (about 58-61%
  right, against reference's 55-59%).
- **kth rails at its 0.05 floor on most chosen fits** (MPI-22, MPI-27,
  MPI-36, MPM-14, MPM-16, MPI-29, MPF-73, MPM-62). Railed kth means "more
  lift than reference". That matches the known fleet-wide high BHP bias
  (September 8: +82 psi mean). The loss is acting as a per-well bias
  correction, not as identified wear. The UI marks railed values and says
  so on the chosen model.
- **The saved IPR is the first-order problem on 16 of 29 wells.** MPB-32's
  saved curve misses by a median of about 600%, and MPB-39's by 44%. Refitting
  the one curve scale moves some wells far from 1 (MPB-32 0.30, at the
  bound; MPI-33 2.03; MPF-73 0.65).
- **Against the proposed acceptance criteria** (plan section 7):
  - Held-out BHP no worse than reference: met on the held default. In refit
    mode 16/18 improve and 2 do not.
  - Pump-change direction at least as good as reference: met, but only
    barely, and not by a meaningful margin.
  - Oil: not yet summarized.
  - `validated_for_sizing` stays false. The standing decision that
    replacements use reference losses (D2) is supported by this evidence.

## Verification

- Python: 2,270 passed (full offline suite).
- Frontend: 45 tests passed, type check clean, production build.
- A network-blocked run of the full suite made no new real-query attempts
  from these changes.
- Browser QA against the local server with live read-only data: JP History,
  MPF-107, installation-fit endpoints answered by a real result.
  - Fitted and held-out chart views, all tables, no page errors.
  - QA caught a wording bug (IPR text followed the checkbox instead of the
    result's request), which is fixed.

## Next

1. Reconcile the saved IPRs the gate flags, starting with MPB-32 and MPB-39,
   before any pump-term work on those wells.
2. Replace the fixed priors with empirical Bayes, and add held-out oil to
   the acceptance summary.
3. Only if a rung later improves pump-change direction by a real margin:
   the phase 4 save path (current installation only, existing
   `pump_calibration_v1` gate, 500-character record).
4. Investigate the shared high-BHP bias that kth is absorbing (discharge
   deficit, datum, P1 vs wellhead). Pump losses should not be used to hide
   it.
