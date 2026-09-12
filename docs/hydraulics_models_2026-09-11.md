# Selectable return hydraulics — September 11, 2026

The solver now offers **Beggs–Brill + Payne, Hagedorn–Brown + Griffith, and
Shi/Pan drift-flux**. Beggs–Brill stays the default. Tulsa unified remains a
disabled, explicitly unavailable choice: its complete primary closure equations
were not accessible. This is two new alternatives, not three completed ones.

The first historical comparison does **not** support replacing Beggs–Brill
fleet-wide. Model selection is useful for diagnosis, but a lower BHP error can
coincide with a worse oil forecast—the quantity that matters for pump sizing.

## Workflow and fit identity

Select **Return hydraulics model** in the sidebar. The choice follows single
solves, pressure profiles, batch sizing, PF sweeps, WC sensitivity and calibration.
Changing models resets the four pump coefficients to the clean reference.
Other well inputs remain in place.

For a named well, save supported well inputs before calibration. Run **Calibrate
to field data** using the selected hydraulics, then preview with **Apply to
inputs**. **Save installed-pump calibration** saves that model choice with the
completed fit. Subsequent pad/CFP optimizations hydrate it per well. The sidebar
selector alone is a session preview; **Save well inputs** does not persist it.
Optimizer rows identify their return model.

The existing compact comment record includes `h` (hydraulics ID) and `m`
(combined physics version). Model, installation, coefficients and quality are
one gated INSERT, checked against the 500-character limit. No new table or
property ID is required. No live save was performed during development.

| ID | Combined physics version | Interpretation |
|---|---|---|
| `beggs` | `entry-energy-v2` | Existing BB + Payne path, unchanged |
| `hagedorn_brown` | `entry-energy-v2+hb-griffith-v1` | Modified H-B with Griffith bubble branch |
| `drift_flux` | `entry-energy-v2+shi-pan-v1` | Steady isothermal Shi/Pan gas/liquid closure |

Old records without `h` mean Beggs–Brill. They cannot supply another model's
coefficients. A later installation invalidates fitted pump losses; the well's
selected return model remains. Replacement pumps use reference coefficients,
including clean replacements of the same size. The cache includes model choice
and physics source. Calibration job handles and apply controls distinguish models.

These choices govern the main simulation/calibration/optimization workflow.
Standalone Header Impact, PF Scenario, washout and friction-trend tools still
build their own Beggs–Brill inputs from their selected tests. They do not hydrate
the sidebar's selected model or the main optimizer's saved configuration.
Their calculation helpers now accept the model where they receive a WellConfig,
but their separate input builders still default to BB. Unifying those input
paths is follow-up work; historical evidence also needs an explicit date/model basis.

## Equations and implementation boundaries

`woffl/flow/hydraulics.py` owns alternative holdup and segment gradients.
`outflow.production_top_down_press` dispatches explicitly. Unknown model IDs
raise an error; they never silently select BB. All alternatives use the existing
PVT, conserved standard rates, actual return area, survey and 100-ft integration
grid. No throat-entry equation, Mach criterion or friction fitting bound changed.

**Hagedorn–Brown + Griffith.** The vertical H-B liquid-holdup correlation uses
the PROMOD1 rational approximations to the viscosity, primary holdup and secondary
charts. The secondary argument uses the liquid **viscosity** number `NL`, not
the liquid velocity number `NLV`. Griffith's bubble branch uses 0.8 ft/s relative
gas/liquid velocity. Non-bubble friction uses no-slip density squared divided by
slip density; the Reynolds viscosity is weighted geometrically by holdup. The
bubble branch uses actual liquid velocity and liquid properties. Acceleration
uses slip density and gas superficial velocity. These implementation choices
follow the [PROMOD1 manual, section 2.5.2](https://www.bsee.gov/sites/bsee.gov/files/tap-technical-assessment-program/300ae.pdf)
and [Whitson's H-B implementation documentation](https://wiki.whitson.com/pipeflow/correlations/hagedorn_brown/).

The model originated in vertical, small-diameter tubing experiments, not this
fleet's complete range of geometry and fluids. Inclined application here projects
gravity along the survey; holdup remains a vertical reference. Viscosity-number
chart endpoints are held at `NL=.002/.4`; the secondary chart factor is held
at its upper endpoint `x=.09` to avoid poles in rational extrapolation. These
are bounded chart approximations, not evidence of applicability outside the
experimental range. [Hagedorn and Brown, 1965](https://doi.org/10.2118/940-PA).

**Shi/Pan drift-flux.** Gas fraction is found by closing gas superficial flux
against `ug = C0*j + ud`. Pan's equations 15–25 supply gas-fraction smoothing,
inclination, flooding velocity and the distribution parameter. The chosen
Table 1 branch has `Cmax=1.2`, `a1=.06`, `a2=.12`, `m0=1.27`, `n1=.24`,
`n2=1.08`, `Fv=1`, `Cku=142`, and `Cw=.008`. The threshold expression is
clamped below zero so `C0` stays at `Cmax` below the stated threshold. Pan uses
inclination from vertical; the interface converts from the survey's upward
angle from horizontal. Mass-center velocity and volumetric mixture velocity
remain distinct. [Pan et al., LBNL-4291E, 2011](https://escholarship.org/content/qt0k35m2vg/qt0k35m2vg.pdf).

Wall friction uses mixture density and mass-center velocity, with linearly
holdup-weighted phase viscosity and the existing Darcy factor. For the steady
isothermal pressure equation, the implementation differentiates the conserved
phase momentum flux `J = rhoL*vsl²/HL + rhoG*vsg²/(1-HL)` with respect to
pressure. The total gradient carries `1/(1+dJ/dP)` after unit conversion.
PVT is restored after each derivative evaluation. This derives a steady pressure
traverse from the published closure; it does not reproduce the full transient,
thermal T2Well simulator.

Oil and water form one mixed liquid. Neither alternative models separate oil/water
slip, emulsion transitions or eccentric annular phase geometry. Annuli use their
actual area and hydraulic diameter, an approximation requiring field validation.
The alternatives support upward/horizontal co-current return segments; downhill
segments and non-subcritical acceleration states fail explicitly. They do not
borrow BB holdup, its friction multiplier or its acceleration clamp.

The pressure-profile plot also now constructs the same PF/formation mixture and
water-mode rate as the solver. Its former separate mixture construction could
ignore independent PF density and omit returned PF volume in dewatering mode.
This corrects the plotted column; it does not alter BB operating-point physics.

## Same-input chronological comparison

The [benchmark tool](../tools/hydraulics_benchmark.py) replayed six wells,
12 later installations and 163 observations. Earlier tests set oil productivity,
WC and GOR; clean pump losses remain fixed. Only return hydraulics changed.
All attempted predictions remain in the [JSON report](hydraulics_benchmark_2026-09-11.json).
The [time plot](hydraulics_benchmark_2026-09-11.png) shows BHP and oil across installations.

The following errors use the **same 130 observations solved by all models**
(126 have scoreable measured PF). Coverage refers to all 163 attempts.

| Model | Solved / attempted | BHP RMS, psi | Median absolute oil error | Median absolute PF error |
|---|---:|---:|---:|---:|
| Beggs–Brill + Payne | 135 / 163 | 130.7 | 26.8% | 9.9% |
| Hagedorn–Brown + Griffith | 137 / 163 | 127.6 | 28.9% | 10.1% |
| Shi/Pan drift-flux | 130 / 163 | 146.5 | 34.9% | 7.4% |

On the 63 common observations after an actual size change, BHP RMS is
87.1 / 81.5 / 88.1 psi respectively; median oil error is 24.0 / 28.7 / 40.5%.
A BHP-only winner is therefore insufficient for the user's sizing objective.

- **MPE-42 11C → 13C:** BHP RMS remains about 128 psi with every model.
  Its return-model change barely moves the forecast. Investigate entry-limited
  behavior, independent inflow/composition and pump recovery before further
  coefficient fitting.
- **MPF-73:** all 26 predictions still fail across two target installations.
  Alternatives did not resolve its inability to lift. Preserve the nominal
  catalog discrepancy and component-balance investigation.
- **MPB-39 12B → 10C:** BHP RMS improves from 92.4 to 69.0 to 52.4 psi,
  while median oil error worsens from 49.5 to 56.2 to 67.0%. This is a concrete
  example of a better BHP plot giving a worse sizing basis.
- **MPB-30:** drift-flux fails all seven target observations; H-B solves all
  seven, including two BB failures, but predicts a poor level match. Do not
  compare each model's successful subset without exposing those failures.

The final recorded serial run, including both frozen and measured-composition
replays, took 18.9 s for BB, 16.1 s for H-B and 83.7 s for drift-flux. Earlier
verification runs took approximately 6.0 / 5.5 / 15.9 s with identical prediction
metrics. Local timings varied substantially; these are not hosted calibration
latencies. Medium workers, job limits and bounded caches are unchanged.
Drift-flux's extra PVT/holdup evaluations cost more in every recorded run.

Present-day geometry/PVT/reservoir-pressure priors are not verified historical
inputs. Tracker dimensions are nominal specifications, never wear measurements;
several catalog conflicts remain. Gauge offset is assumed zero, despite gauges
usually being within 40 ft of the JP. These observations cannot validate
alternative pump ranking, shared parameter transfer or marginal-water response.
Choosing a model on this comparison requires another later holdout.

## Verification and next work

Final local checks: **1,962 Python tests and 11 frontend tests passed**;
TypeScript/Vite production build passed. Python reported four existing warnings;
the build retained its existing chunk-size advisory. The plotted artifact was
visually inspected. No new browser QA, production writes or deployment was run.

Named regression guards cover liquid hydrostatics/Hagen–Poiseuille friction,
ideal-gas acceleration, Griffith relative velocity, Pan's published Ku figure
and large-diameter limit, phase bounds, invalid domains, unchanged BB, real
solver discharge closure, model-specific cache keys, identical single/batch/PF
results, water-mode plot consistency, fit save/hydration and UI reset behavior.
See [upstream patch 44](upstream_sync.md#44-selectable-return-hydraulics-2026-09-11)
and `tests/test_hydraulics_models.py`. Numerical consistency is not field qualification.

Tulsa still needs the complete flow-pattern and closure equations from Zhang
et al. (2003), plus independent reference cases. Publisher access to the
[unified-model paper](https://doi.org/10.1115/1.1615246) was blocked; public
Tulsa/BSEE progress reports describe the framework but omit required closures.
Do not fill those gaps with guessed equations or relabel another correlation.

Next, keep the [multi-installation plan](jp_model_improvement_plan_2026-09-11.md)
focused on oil/PF/BHP together: verified catalog/completion inputs, reservoir
evolution separated from pump effects, response to pressure changes, then
prediction of complete later installations and other wells. Judge proposed
changes by `delta_oil - lambda*delta_machine_water`, with
`lambda=(1-marginal_wc)/marginal_wc`, using the correct constrained water stream.

**Pump Match Over Time remains an in-app screen to build.** The current PNG is
an offline comparison. The planned production/history screen should show
installation boundaries, BHP/oil/PF, model identity, fitted versus held-out
predictions, failed/missing periods and evidence linked to optimizer decisions.
