# Pad optimization redesign — one formulation (2026-09-02)

Decisions taken with Scott on 2026-09-02, following the 2026-09-01 review (§8):

| Decision | Choice |
|---|---|
| Objective | **oil − λ·water**, one λ (BOPD per BPD of machine water) shared by every engine. Replaces the hand marginal-WC gate, the auto-derived gate and the parsimony pass. |
| Solvers | **Both kept, made identical**: MILP (scipy) and MCKP (CP-SAT) see the same candidate set (every converged pump), the same shut-in rule (at most one pump per well) and the same objective. `method` is a cross-check that must agree on the objective value. |
| Plant | **Setpoint below the knee, frontier above**: the delivered header is the operator setpoint whenever the frontier can deliver more at that flow; capacity binds only on the frontier. |
| S-Pad | **One sweep for every pad**: a fixed-speed station's header is a function of total flow, so S-Pad sweeps total flow with the knapsack inside, exactly as I/M/E sweep header. The damped fixed point is retired. |

## 1. The problem every engine solves

For a pad with wells `w`, each with converged candidate pumps `k` (a pump is a
(nozzle, throat) pair with modeled `oil_wk` BOPD and `water_wk` BPD of the water
stream the pad's machines handle):

```
maximize   Σ_w Σ_k x_wk · (oil_wk − λ · water_wk)
subject to Σ_k x_wk ≤ 1                    for every well  (shut-in allowed)
           Σ_w Σ_k x_wk · water_wk ≤ B(P)   the plant budget at the delivered header P
           x_wk ∈ {0, 1}
```

`P` is the decision the outer sweep walks: the header for a free-pressure plant
(I/M/E), the total flow for a fixed-curve station (S, where `P = header_at_flow(Q)`
and `B = Q`).

### Why λ subsumes the three old cutoffs

- A marginal water-cut gate `w` excluded a pump whose incremental barrel was more
  than `w` water, i.e. whose oil-per-water ratio `r < (1−w)/w`. In the objective the
  same pump carries a negative increment when `r < λ`. Same rule, `λ = (1−w)/w`.
- The auto-derived gate read the budget's own shadow price off the pooled Pareto
  frontier. `derive_lambda` returns that price directly as `λ*` (bbl/bbl) instead of
  converting it to a water cut and back.
- Parsimony swapped a pump down when it gave up ≤ 20 BOPD for any water saving,
  with no ratio test and no re-spend of the freed water. With λ in the objective a
  swap is taken exactly when `Δoil / Δwater < λ`, and the freed water is re-spent by
  the same solve.

### λ sources

- **manual** — the engineer's price (slider). Trials are compared on the objective.
- **auto** (default) — `λ*` from `derive_lambda` at each trial: the ratio of the
  frontier segment that exhausts the budget, 0 when the budget never binds. Because
  `λ*` varies by trial, trials are compared on OIL (as today), and the winner
  reports its `λ*`. This is the equal-slope allocation of Kanu (1981) the CFP engine
  already uses; the pad engines now speak the same language.

## 2. Both solvers, one candidate set

`milp_optimization` and `mckp_optimization` both take every converged row of each
well's batch frame (`error == "na"`), not only the semifinalists, and both allow
shut-in. `network.optimize_jet_pumps` gained `water_price` and `all_configs`
parameters (defaults keep the upstream signature). CP-SAT works on integers, so the
objective is scaled by 100 and rounded; the agreement test asserts the two
objectives match within that quantization.

## 3. Plant semantics

`PadPlant.delivered_header(q_total, setpoint, n_pumps)` returns
`min(setpoint, header_at_flow(q_total))` and `over_capacity` when the frontier cannot
carry the flow. Free-pressure sweeps treat the header as the setpoint decision with
`budget_at_pressure(P)` as before (that already IS "setpoint below the knee");
the scenario evaluators (`evaluate_fixed_scenario`, `evaluate_existing_scenario`)
now settle on the delivered header instead of the frontier, which is what made
E-Pad collapse below its knee.

## 4. One sweep

`run_optimization` walks the pad's decision variable over a coarse grid of
`n_steps`, then refines around the best point (two rounds of bracket halving), and
returns the winner with the usual `meta` plus `lambda_used`, `lambda_source`,
`objective_bopd_equiv` and `solver_agreement` (when `method="mckp"` the MILP is run
at the winning point and the objectives compared).

## 5. What is NOT changed here

- The CFP moves engine keeps its λ-sweep and boards (same λ meaning).
- The choke plan keeps its ladder (already an equal-slope trim).
- Water basis per pad (`PadPlant.water_key`): **decided 2026-09-02** — E and M pumps
  handle formation AND lift water (Scott; matches `well_sort_engine.POPS_PUMP_HANDLES`),
  so `EPadPlant` / `MPadPlant` budget and price `totl_wat`; I and S stay on `lift_wat`
  (PF-only pumps). Guarded by `tests/test_pad_plants.py::test_water_key_follows_what_each_pad_pump_handles`.
- Energy and JPCO costs are not priced (decision: oil − λ·water only).

## 6. Compatibility

`OptimizeRunRequest.marginal_wc` and `parsimony_bopd` are still accepted:
`marginal_wc` maps to `λ = (1−w)/w`, `parsimony_bopd` is ignored with a note in
`meta`. New knob: `lambda_bopd_per_bpd` (None = auto).

## 7. Status — shipped 2026-09-02

| Piece | Where | Guarded by |
|---|---|---|
| Priced objective in both solvers, same candidate set (every converged row, shut-in allowed) | `optimization_algorithms.milp_optimization` / `mckp_optimization`, `network.optimize_jet_pumps(water_price=, all_configs=)` | `tests/test_optimization_algorithms.py::TestWaterPriceObjective`, `tests/test_marginal_wc_enforcement.py` (incl. `TestSolverAgreement`) |
| λ sources: manual / legacy wc / auto (`derive_lambda` = crossing-segment slope of the pooled Pareto frontier at the budget) | `optimization_algorithms.marginal_wc_to_lambda`, `water_price`, `derive_lambda` | `TestWaterPriceObjective`, `TestPriceFromGate` |
| One sweep for every pad; S-Pad sweeps total flow (header = curve at that flow); coarse grid + `refine_rounds` bracket halving; fixed point retired (`converged` always True, `history` = []) | `pad_optimize.run_optimization` | `tests/test_pad_optimize.py::TestFixedCurveSweep`, `TestPressureSweepRun` (refinement, setpoint pin), `TestMarginalWcAutoDeriveAndParsimony` |
| Setpoint below the knee, frontier above; recirculation holds the setpoint | `PadPlant.delivered_header`, `pad_optimize.settled_header` (duck-typed plants fall back to the base rule) | `tests/test_pad_plants.py`, `tests/test_choke_plan.py` |
| `setpoint_psi` pins a free-pressure header (one trial, no refinement) | `run_optimization(setpoint_psi=)` | `test_setpoint_pins_the_header_instead_of_sweeping` |
| MCKP runs re-solve the winner with MILP → `meta["solver_agreement"]` | `run_optimization` | `test_mckp_reports_solver_agreement` |
| Server: `OptimizeRunRequest.lambda_bopd_per_bpd` (None = auto; wins over `marginal_wc`); `parsimony_bopd` accepted, never forwarded; meta exposes `lambda_used`, `lambda_source`, `objective_bopd_equiv`, `water_key`, `solver_agreement` | `server/schemas.py`, `server/services/optimizer_runs.py` | `tests/test_web_optimizer_runs.py::test_pad_run_forwards_the_water_price` |
| Client: "Water price λ (BOPD/BPD)" auto/manual knob replaces the marginal-WC gate + parsimony inputs; results show the price (BOPD/MBPD, equivalent WC and source in the tooltip) and the MILP cross-check | `web/src/pages/optimize/RunPanel.tsx`, `web/src/api/types.ts` | `tsc --noEmit` |
| Server + client: `OptimizeRunRequest.setpoint_psi` (1,000-5,000 psi, None = sweep) forwarded to `run_optimization`; the run reports the CLAMPED pin as `meta["setpoint_psi"]` (None when swept or on a fixed-curve pad); RunPanel shows a "Header setpoint (psi)" auto/manual knob for I/M/E JPCO runs and PadResults says when the header was pinned | `server/schemas.py`, `server/services/optimizer_runs.py`, `pad_optimize.run_optimization`, `web/src/pages/optimize/RunPanel.tsx` | `tests/test_web_optimizer_runs.py::test_pad_run_forwards_the_header_setpoint`, `test_pad_run_without_a_setpoint_sweeps`, `test_no_setpoint_reports_none_and_a_fixed_curve_pad_ignores_one` |
| Scenario evaluators settle the plan <-> header coupling by BRACKETED BISECTION on `g(P) = delivered_header(draw(P)) - P` over the clamp window (2 probes + <=12 halvings); the damped fixed point survives only as the no-sign-change fallback, keeping `converged=False` semantics | `pad_optimize._settle_scenario_coupling`, `evaluate_fixed_scenario`, `evaluate_existing_scenario` | `tests/test_pad_optimize.py::TestScenarioHeaderCoupling` |

The optimizer's `marginal_watercut` attribute is kept as the EQUIVALENT gate
`1/(1+λ)` of the price the trial solved at (pages and `reconcile_wells` still
label with it); `mwc_excluded` / `mwc_excluded_wells` are always empty — nothing
is pruned, the knapsack chooses shut-in.

Closed 2026-09-02: the scenario evaluators
(`evaluate_fixed_scenario` / `evaluate_existing_scenario`) no longer iterate a
damped fixed point. A fixed plan has no decision to sweep, but the delivered
header and the plan's own PF draw still determine each other, so
`_settle_scenario_coupling` solves that crossing by bisection over the plant's
clamp window (bounded work, no oscillation) and only falls back to the damped
loop when the band carries no sign change - including the degenerate
free-pressure case where nothing draws and every trial is trivially "held".
They still settle on `delivered_header`, so setpoint stays below the knee.
The trade the bisection buys: guaranteed convergence for roughly 2-4x the
plan evaluations of a converging damped loop (each one is a full pad batch
solve), bounded at 14.

