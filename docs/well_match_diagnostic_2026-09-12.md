# One well IPR, measured test composition — September 12, 2026

The user clarified that historical lookback must use each test's WC and GOR,
and that the IPR must stay fixed unless the user explicitly requests a shift.
The goal is one IPR describing the well across pumps. This revision implements
that contract and fixes a numerical failure found while investigating the
remaining matches. It does not implement the shared multi-installation fitter.

## Implemented behavior

**Every test (saved well fit)** holds the saved Vogel **oil-rate-versus-BHP**
curve, anchor pressure and reservoir pressure fixed. Historical hardware,
measured WC/GOR and test-day PF/WHP determine the operating point. Test oil,
BHP and PF rate remain outcomes for comparison, never per-test IPR anchors.
The source reads `vw_well_test.form_wc` and `form_gor` for each test; composition
does not come from today's sidebar values. Invalid/missing WC or GOR leaves
an explained gap. No zero prediction or composition substitution fills it.

WellConfig stores a total-liquid anchor, but the solver's InFlow takes oil:

```text
saved_oil_anchor = saved_qwf * (1 - saved_wc)
test_qwf = saved_oil_anchor / (1 - test_wc)
```

This changes only the total-liquid representation. The oil IPR is identical
at every suction pressure; GOR changes the gas mixture. Fresh PVT objects
prevent one test's conditioning from contaminating another. The saved inputs
are never mutated. Tests cover WC from zero through 99.9%, GOR zero and higher,
multiple pumps, invalid composition, curve preservation and cache invalidation.

The plot tooltip and test inspector show the WC/GOR actually used; installation
details show the fixed oil anchor in BOPD. The two optional chronological
comparisons are now labeled **Refit earlier tests: same pump / next pump**.
Choosing one explicitly invokes an earlier-test fit for that comparison; neither
mode changes the saved IPR. They retain frozen training composition for the
later-test forecast. Measured-composition replay is labeled retrospective.

## Numerical defect and recovery

The prior solver rejected B-30 when discharge residuals were negative at both
ends of the feasible suction range. Direct interior evaluations show two
pressure-balance roots. The lower-suction crossing already exists in the same
equations and inputs; the endpoint assumption hid it.

[Residual curve](well_match_diagnostic_2026-09-12.png) and
[recorded inputs/results](well_match_diagnostic_2026-09-12.json) show the
September 2 example: 1149.62 psig, 338.39 BOPD, with discharge residual about
2.3e-12 psi. Both endpoints remain negative. No observed BHP/rate is used to
choose the root. The first negative-to-positive crossing follows the ordinary
solver's bracket orientation; transient branch stability remains unverified.

The bounded fallback runs only after that endpoint failure, scans 64 intervals,
refines with Brent, and accepts only finite, nonnegative rates with an actual
discharge residual within 10 psi. It does not bridge known infeasible probes
or accept an unclosed sign jump. The cap is 192 residual evaluations. Very
narrow unprobed feasible regions can still be missed. Existing successful and
entry-limited paths are unchanged. No energy, PVT, inflow or loss equation was
altered. This is [upstream patch 45](upstream_sync.md).

## Frozen-data comparison

`tools/well_match_diagnostic.py` uses only the trusted September 8 snapshot,
captured at 2026-09-08T22:06:01.207507+00:00. It records inputs, source hashes,
all rows, failures and common-observation scores. B-30 uses 12 months to match
the screenshot investigation; the other six wells use 24 months. All cases use
one saved IPR per well, BB/Payne and clean catalog pump losses. No live queries
or writes were made. The comparison took 60.3 seconds locally, including the
residual trace; that is not hosted latency.

There are **392 tests, 384 eligible**. Disabling just the new fallback reproduces
the earlier solver on **329/384**. Enabling it with identical saved composition
solves **352/384**, recovering all **23** missed B-30 tests. All **329** already
successful predictions are exactly unchanged. Using measured WC/GOR solves
**351/384**; F-73 loses one previously solved case and still needs diagnosis.

The table compares saved versus measured composition **after** the numerical
fix, on the identical observations solved by both variants. Oil error is median
absolute percent error; failures remain counted separately above.

| Well | Common tests | BHP RMS: saved → test composition (psi) | Oil error: saved → test composition |
|---|---:|---:|---:|
| B-28 | 69 | 168.7 → 215.3 | 39.6% → 17.2% |
| B-30 | 44 | 150.6 → 92.6 | 10.8% → 23.2% |
| B-37 | 60 | 107.9 → 115.3 | 13.0% → 12.0% |
| B-39 | 48 | 158.8 → 177.0 | 140.5% → 70.9% |
| F-107 | 44 | 167.6 → 187.9 | 21.6% → 21.2% |
| E-42 | 75 | 77.8 → 85.4 | 14.7% → 16.7% |
| F-73 | 11 | 227.6 → 229.3 | 88.3% → 89.6% |

Measured composition makes the retrospective question meaningful; it does not
guarantee a lower error. B-30 now covers **44/44** tests; its original 21-test
survivor-only score must not be compared as though it were the same cohort.
The separate chronological adapter check solves **140/166**, preserves all
**121** comparable September 11 predictions exactly and still fails all 26
F-73 target tests. That artifact is
`build/pump-match-benchmark-after-interior-search-2026-09-12.json`. Earlier
September 11/12 benchmark artifacts remain preserved.

## How to improve the remaining matches without hiding the cause

1. **Check oil-IPR consistency before adjusting pump losses.** Evaluate the
   saved oil curve at measured BHP and compare it with measured oil. If those
   two observations do not lie on the curve, no pump-loss coefficient can
   reproduce both exactly while the IPR stays fixed. For example, B-30 has
   25.3% median absolute oil discrepancy at measured BHP across 43 in-range
   observations; B-39 has 46.3% across 50. This direct curve check requires no
   pump/return calculation and is diagnostic, not forecast validation. A joint
   well fit should use one curve across the selected history, with any IPR
   refit an explicit user choice; no automatic per-date or per-pump shifts.
2. **Align measured conditions.** The current PF source reduces daily readings
   with `max`, whereas BHP is daily cleaned data and oil/WC/GOR are test values.
   Inspect stable operating windows and pressure timing before attributing a
   mismatch to pump physics. Keep raw records and explain unsupported periods.
   B-30 has one observed BHP at/above saved reservoir pressure; B-37 has one,
   F-73 three. Verify timing/datum/source before changing pressure assumptions.
3. **Separate pump and return losses.** Diagnose available versus required
   discharge at measured suction, and inspect nozzle/PF, entry, mixing,
   diffuser, static-head and return-friction contributions. Measured-outcome
   conditioning is a diagnostic only. Verify hardware and flow direction;
   tracker diameters are nominal specs, not wear measurements. The user's
   typical gauge offset within 40 ft cannot justify arbitrary large offsets.
4. **Fit only parameters the observations constrain.** Use bounded shared
   hydraulic parameters where physically transferable and installation-specific
   losses where justified. Keep the IPR fixed by default. Require simultaneous
   BHP/oil/PF and pressure-response checks; a better BHP level alone can worsen
   oil or PF. Assess parameter sensitivity, correlation and bound hits before
   exposing extra degrees of freedom. Account for uncertainty in inputs as
   well as outcomes; the statistical basis is illustrated by
   [NIST's errors-in-variables calibration research](https://www.nist.gov/publications/errors-variables-calibration-dark-uncertainty).
   Adapting that approach to these wells remains proposed work.
5. **Validate decisions on later whole events/installations.** Keep fit and
   prediction scores distinct, use whole-event/size holdouts, retain failures,
   and check predicted changes in oil and PF under pressure/pump changes.
   Use the pad/field marginal-water objective and check recommendation rank
   stability. The existing hydraulics comparison does not establish one
   alternative as a universal improvement. A lower historical error does not
   yet establish reliable optimization gains.

The shared multi-installation fitter, component diagnostics, identifiability
and blind optimization-gain validation remain the next engineering blocks.
This revision supplies a more faithful replay and numerical coverage, with
the remaining misses visible.

## Verification

**1,997 Python tests passed**, including the eight interior-root regressions
and seven new composition cases; **17 frontend tests passed**; TypeScript and
Vite build passed. `web/dist` was rebuilt. Fixture-only Playwright passed with
zero browser errors, all comparison modes, per-test WC/GOR, fixed oil-IPR
display, four synchronized SVG axes, gaps and cancellation. Source snapshots
in the numerical reports predate only a subsequent explanatory comment edit
in solopump; executable behavior is unchanged.

No deployment or production-data write was performed.
