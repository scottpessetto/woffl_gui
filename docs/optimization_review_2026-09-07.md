# Optimization workflow review - 2026-09-07

> **Documentation status, 2026-09-08:** Historical review. The choke water-basis and price-tie findings were subsequently addressed in the [fix record](code_review_2026-09-07_fixes.md); persistent workers and winner-only payload retention are in [Medium performance](medium_performance_2026-09-07.md). S-Pad progress labels and installed/clean candidates were also corrected. Use the [handoff](session_learnings_2026-09-08.md) for remaining work; do not treat the open-finding headings below as current status.

Follow-up: the choke water-budget and priced-tie findings below are now fixed,
along with CFP convergence and evidence-reference defects from the broader
review. See [implementation and validation](code_review_2026-09-07_fixes.md).
The findings below are retained as the original review record.

Scope: traced the React optimization board and run submission through saved-fit
hydration, background jobs, pad JPCO allocation, CFP future-well handling, and
plant/result reporting. Inspected the choke allocation path and solver economics.
This was a local code review with synthetic reproductions, not a validation of
field predictions against current operating data.

**Fixed in this working tree**

- E/M JPCO plant flags and operating envelopes used lift water even though the
  allocation budgets total water. They now use the winning machine-water total;
  the envelope also receives the winning header instead of defaulting to the
  plant header cap. The API exposes `total_machine_water_bpd` for the winner and
  sweep trials. Station charts use it for duty/trial points, and results show
  machine water separately from PF. Existing PF totals retain their meaning.
  Guard: `TestPressureSweepRun.test_plant_reporting_uses_machine_water_and_winning_pressure`.
- Every trial shared mutable WellConfigs, so the returned winning optimizer
  carried the last trial's well pressures. Each trial now takes its own shallow
  copies before setting pressure. This changes input ownership, not pump physics.
  Guard: `TestPressureSweepRun.test_winning_well_configs_retain_winning_pressure`.
- The browser flattened future wells across pads without their destination;
  hydration assigned all of them to the first pad. Submission now includes the
  destination pad, and hydration honors it. Out-of-run destinations are skipped
  with a note. Older requests without a pad retain the first-pad fallback.
- CFP looked up tracked pumps only for modeled wells, so donors outside the
  selected pads were absent and their future wells were skipped. Pump lookup
  now includes donors; future wells retain an offline baseline.
  Guard for both future-well fixes:
  `test_cfp_offline_wells_are_bring_online_candidates[True]`.

**High-priority findings still open**

1. Choke allocation does not honor the plant's water basis.
   `woffl/gui/pad_optimize.py::run_choke_optimization` builds options with PF,
   passes them to `_trim_to_budget`, and compares their summed PF against
   `plant.budget_at_pressure`. It never branches on `plant.water_key`.
   E/M declare `totl_wat`, so a plan fitting the PF budget can exceed the actual
   machine-water budget once formation water is included. The JPCO reporting
   fix above does not repair this separate allocation path. The repair needs
   formation water carried through modeled and test-fallback options, evidence
   adjustments, trim slopes, and contingency ladders, with explicit handling
   when a test fallback has no usable formation-water estimate.

2. Automatic price can choose shut-in when a producing pump fits.
   Reproduced using the actual `derive_lambda` and `milp_optimization` functions:
   one well with candidates (water, oil) = (80, 80) and (100, 100), budget 90.
   Derived lambda is 1. Both candidates have zero priced objective; the solver
   returns no pump and zero oil. At lambda zero it selects the feasible 80-oil
   pump. The priced objective allows this tie, but the result is surprising for
   a default mode described as automatic allocation. A secondary objective
   maximizing oil among equally priced solutions would resolve this example;
   it must be implemented consistently in both solvers and checked against
   integer quantization. That alone would not establish that auto pricing
   maximizes oil under every discrete budget. The current documented manual/
   auto objective policy was preserved in this patch.

**Performance opportunities to benchmark next**

- Each pad sweep point calls `run_all_batch_simulations`, which creates and
  tears down a process pool when parallel execution is enabled. A run-scoped
  pool could amortize Windows process startup and imports; retain worker caps
  and the existing broken-pool serial fallback. No speedup has been measured.
- The sweep retains every trial's optimizer and complete batch results until
  completion, although only the winner needs those objects after scoring.
  Keep scalar sweep records plus the current winning optimizer to reduce peak
  memory. Refinement still needs the scalar decision coordinates and scores.
- Progress callbacks currently receive the flow decision as the purported
  header on S-Pad. The API then labels that value psi. Correcting this is a
  small usability improvement independent of the allocation formulation.

**Verification**

Baseline: 1,681 tests passed. New regression cases were run before the fixes
and failed on pressure ownership, water-basis reporting, and the external
future donor. After the initial fixes, all 66 targeted pad/API tests passed.
Full final-suite and frontend build results are reported in the delivery message.
No production writes, deployment, or live warehouse validation were performed.
