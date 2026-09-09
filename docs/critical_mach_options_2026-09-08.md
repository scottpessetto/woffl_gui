# Critical-Mach investigation and resolution options — 2026-09-08

**Follow-up:** the shared unscaled balance has now been implemented. This
file records the pre-change investigation; see
[the implementation and validation record](entry_energy_implementation_2026-09-08.md)
for current behavior and replay instructions.

The inconsistency is confirmed and has observable effects on convergence and
sonic classification. The recommended direction is one shared throat-entry
energy model, with the limiting flow derived from that model. The present
Mach multiplier is not a validated slip model. Simply moving or deleting its
energy divisor is insufficient.

This investigation adds an offline experiment, analytic tests and artifacts.
It does not change the application's physics, persisted calibrations, deployed
compute tier or production data. All examples below are explicit fixtures,
not new measurements from the named field well.

## 1. Direct cause and consequences

`woffl/flow/jetflow.py::throat_entry_mach_one` computes

    E_choke(p) = (1 + ken) * v(p)^2 / (2 * mach_crit^2)
                 + integral[psu -> p] (144 * gc / rho) dp

`throat_entry_zero_tde` computes

    E_operating(p) = (1 + ken) * v(p)^2 / 2
                     + integral[psu -> p] (144 * gc / rho) dp

Here gc = 32.174 in the library's field-unit convention; E has units ft²/s².
Both walks use the same fluid state, area, IPR rate and physical velocity.
At mach_crit = 2, the choke walk carries only 25% of the kinetic/loss term
used by the operating walk. Pressure work is not rescaled.

`solopump.jetpump_solver` first finds a floor with the first equation, then
`discharge_residual` tries to obtain a throat state using the second equation.
It deliberately refuses to reuse the scaled book when mach_crit != 1. The
following outcomes occur:

- A lowered floor is infeasible under the operating equation. The inward
  search moves back to a higher suction, undoing much of the intended change.
- `sonic_status` compares the returned suction with the original, lowered
  floor. A point still limited by the unscaled entry balance can therefore
  lose its sonic flag. Calibration and evidence gates use this flag.
- A sufficiently low starting floor can produce no non-negative energy
  gradient. That is `ThroatEntryNoSolution`, which intentionally propagates
  through `_residual_walk_inward` to the existing GOR recovery path. It is
  different from the recoverable `ThroatEntryChoked` exception.

Gas-rich fixture: WC 0.8, GOR 1,200 scf/STB, oil IPR anchor 100 BOPD at 500
psig, reservoir 1,700 psig, 12B pump, PF 3,168 psig, suction temperature 100°F.

| Current mach_crit | Calculated floor, psig | Returned suction, psig | Sonic flag |
|---:|---:|---:|---|
| 1.0 | 328.65 | 328.65 | true |
| 1.1 | 298.62 | 328.18 | false |
| 1.5 | 215.78 | 328.19 | false |
| 2.0 | 157.85 | solve fails | — |
| 2.5 | 125.93 | solve fails | — |

The E-41 reference fixture's 12B operating point is subsonic: its suction
stays approximately 1,105 psig across settings while the calculated floor
drops from 1,076 to 711 psig. A subsonic well being insensitive to a choking
threshold is not itself a bug. The rejected floor, sonic misclassification
on the gas-rich case, and new failures establish the practical mismatch.

## 2. A second assumption also needs correction

The code comment claims dividing kinetic energy by Mcrit² moves the energy
minimum exactly to homogeneous Mach Mcrit. That is not generally true for
the actual PVT path and loss model.

With conserved mass flow, fixed area, and the current isothermal density path,
v = mass_flow / (area * rho). Differentiating the implemented energy equation
with respect to pressure gives

    dE/dp = (144 * gc / rho) * [1 - (1+ken) * v² / (Mcrit² * c_path²)]
    c_path² = 144 * gc / (d rho / dp)

Therefore its stationary point satisfies

    v / c_Wood = Mcrit * c_path / [c_Wood * sqrt(1+ken)]

It equals Mcrit only if the last ratio is one. `c_path` is a derivative of
the chosen fixed-temperature material path, not a claim about the measured
acoustic sound speed. The entry loss coefficient also appears explicitly.

`ResMix.cmix()` uses volume-weighted component acoustic compressibilities.
The expansion walk reconditions the fluid and changes dissolved/free gas
partition with pressure. Those are different responses. In the E-41 fixture,
c_path/c_Wood is 0.975, 0.946 and 0.907 at 200, 500 and 1,000 psig. The
unscaled energy minimum at the limiting suction is near Wood Mach 0.94.

The low-GOR fixture exposes a stronger density-model limitation: at 500 and
1,000 psig the finite-difference density derivative is zero, while `cmix()`
is about 4,748 ft/s. Its dissolved gas is capped and the current below-nominal-
bubblepoint Bo correlation is then pressure-independent; water density is
also constant. An internally consistent derivative-based limit cannot by
itself repair that property model. A minimum at the 50-psig numerical boundary
must be labeled a pressure-bound limit, not physical sonic choking.

Research supports distinguishing closures: Benjelloun and Ghidaglia derive
different wave speeds for shared-velocity, separate-velocity and thermal
equilibrium models. Slip is represented through governing equations, rather
than just a multiplier on one energy term. Chung et al. likewise derive a
two-phase critical-flow criterion from a two-fluid model and assess it against
nozzle experiments. These support the modeling approach; neither validates
WOFFL's coefficients or a particular correction for these wells.
[Benjelloun & Ghidaglia, 2021](https://arxiv.org/html/2110.05215v1),
[Chung et al., 2004](https://doi.org/10.1016/j.jsv.2003.07.003).

## 3. Options compared

| Option | What it resolves | Limitation / judgment |
|---|---|---|
| Use Mcrit = 1 for qualified model comparisons; keep measured response corrections separate | Removes the two-walk scaling mismatch immediately | Containment only. Does not repair the acoustic/property-path distinction or field response bias. Existing fitted parameters must be reviewed/refit. |
| Remove the divisor but keep stopping at Mach Mcrit | Makes both walks carry the same kinetic term | Reject as a standalone fix. The threshold may lie beyond the energy minimum and create an artificial, higher floor and false sonic classification. |
| Apply the divisor to both energy walks | Makes the two equations agree and permits lower suction | Empirical model change, not established slip physics. Changes subsonic performance too; requires new interpretation, model version and full refitting. Not recommended as the physical fix. |
| Derive the limit from one shared, unscaled energy path | Aligns feasibility, operating state and limiting flow for the existing material path | Recommended engineering foundation. Must distinguish interior/boundary minima and resolve PVT assumptions; alone it may not remove measured field bias. |
| Implement a physically coupled slip / finite gas-release model | Can represent a different critical velocity through phase mass, momentum and energy behavior | Longer-term candidate. Requires additional assumptions, data and validation. Use only if the shared homogeneous model still fails held-out field response. |

The two simple edits were tested, not just assessed qualitatively:

- **Remove divisor:** on the E-41 fixture, Mcrit = 2 returned suction 1,133.74
  psig and `sonic=true`, despite reported Wood Mach only 0.374. At Mcrit = 2.5,
  suction rose to 1,178.30 psig. This is an artificial constraint at a point
  past the actual energy minimum, not the desired relaxation.
- **Scale both walks:** at PF 3,168 psig, E-41 oil changed from 210.97 BOPD
  at Mcrit = 1 to 276.35 BOPD at Mcrit = 2 (about +31%). Its operating Mach
  stayed near 0.50. This is changing the operating energy budget even away
  from choking. Matching two altered equations would pass the existing
  mismatch test without proving a physical energy balance.
- **Shared energy minimum prototype:** E-41 floor 1,075.72 psig; gas-rich floor
  327.71 psig. All eight fixture/pressure combinations solved. This prototype
  removes the arbitrary Mach setting; it does not reproduce the empirical
  multiplier's lowered floors or establish field agreement.

The prototype minimizes a densely sampled energy path and roots its minimum
against the IPR-fed suction. It uses the real downstream solver. It is a
research probe, with a global minimum search; production must handle the
physically reachable branch, multiple minima, the pressure boundary, and
property discontinuities explicitly. It is not an application feature.

## 4. Recommended implementation sequence

1. Introduce an explicit model version and one throat-entry state/energy
   builder consumed by floor finding, discharge residuals and diagnostics.
   Return the limit reason separately: energy turning point, pressure bound,
   mixture feasibility or discharge balance. Classify sonic flow from that
   shared solution, rather than proximity to a differently computed floor.
2. For the first consistent model, retain physical kinetic energy and obtain
   the limiting state from the reachable energy branch. Report the existing
   Wood Mach as a diagnostic. Keep the current model available for comparison;
   do not silently reinterpret saved `jp_mach_crit` values.
3. Make phase assumptions explicit. Correct density/compressibility consistency
   where a capped solution gas ratio makes the current oil path incompressible.
   Compare an equilibrium gas-release path with a frozen/finite-release path;
   add a slip model only with the corresponding phase transport equations.
4. Refit against pressure/BHP/PF observations within a single pump era, then
   evaluate on whole held-out pressure events. Check BHP levels, response
   slopes, oil and PF together. A fitting error reduction alone is not enough.
   An empirical choking parameter is not proven transferable to a different
   throat/nozzle just because it is stored as a well property.
5. Include model version/closure choices in cached response keys and fit
   provenance. Recheck optimizer recommendations and their sensitivity to
   model choice before relying on differences between proposed pump sizes.

This work does not require a larger Databricks tier. The production algorithm
should locate/refine the minimum adaptively and reuse PVT evaluations; the
dense offline prototype is designed for inspection, not runtime performance.
Benchmark it on the existing two-worker budget before enabling it.

## 5. Tests and reproducibility

The investigation ran 64 floor probes and 128 full-well solves across four
fixtures, five Mach settings, two PF pressures and four model variants (the
energy-minimum variant has no adjustable Mach setting). Eight solves failed:
six under the current model and two under the scale-both candidate. Errors
are retained explicitly in the JSON; they are not dropped from a success rate.

Pressure-step refinement from 10 to 1.25 psi moved the final two computed
floors by less than 0.002 psi in all four fixtures. The low-GOR minimum lies
on the numerical pressure boundary, so it is not a sonic qualification.

Four new tests validate the investigation itself. Three compare the minimum
against an independent isothermal ideal-gas result for ken = 0, 0.03 and 0.4:
Pthroat/Psuction = exp(-1/2), using **absolute** pressures, and
v/c = 1/sqrt(1+ken). The fourth verifies trial patches restore the original
functions even if a probe raises. Existing expected failures remain visible.

Full offline regression result after adding the investigation tests:
**1,770 passed, 3 expected failures, 2 warnings**. The three expected failures
are still the unresolved production energy-scaling checks; the experiment
does not replace or suppress them.

Production acceptance needs more than the current inlet-KE comparison:

- Same-state agreement of the full energy curve and independent conservation.
- Returned sonic state at an interior energy limit; no sonic claim at a
  pressure bound or mixture-feasibility edge.
- Agreement between analytic derivatives and energy-curve turning points.
- Finite, continuous behavior through bubblepoint, zero-free-gas, high-watercut
  and low-pressure limits; pressure-grid convergence.
- Full Solver/Batch/Network equivalence with the same versioned inputs.
- Held-out event prediction and optimizer decision stability after refitting.

Reproduce offline from the repository root:

```powershell
$env:PYTHONPATH='.'
$env:WOFFL_MAX_WORKERS='1'
./venv/Scripts/python.exe tools/critical_mach_study.py --output docs/critical_mach_study_2026-09-08.json
./venv/Scripts/python.exe -m pytest tests/test_critical_mach_study.py -q
```

Artifacts: [raw probes](critical_mach_study_2026-09-08.json),
[comparison plot](critical_mach_study_2026-09-08.png),
[experiment code](../tools/critical_mach_study.py).
