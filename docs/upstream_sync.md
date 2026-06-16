# Upstream Sync — Local Library Patches (don't lose these)

This repo (`github.com/scottpessetto/woffl_gui`) is a **fork** of the upstream
`woffl` library (`github.com/kwellis/woffl`). The GUI (`woffl/gui/`) is ours
alone, but the four **library** packages — `woffl/pvt/`, `woffl/geometry/`,
`woffl/flow/`, `woffl/assembly/` — are shared with upstream and published to PyPI.

We carry a few **local patches inside the library** that are **not yet
upstreamed**. When the upstream owner ships changes and we sync them in, those
patches must survive the merge. This file is the authoritative record so they
can't be silently overwritten — and the regression tests below are the tripwire
if one ever is.

> Every patch site is tagged `# [LIBRARY change -> upstream PR to kwellis/woffl]`
> in the code. `grep -rn "upstream PR" woffl/` finds them all.

---

## The load-bearing local patches (pending upstream PR)

### 1. `woffl/assembly/solopump.py` — marginal-well convergence
New helpers `_secant_solve`, `_bisection_solve`, `_residual_walk_inward`, plus the
`jetpump_solver` body that orchestrates them. Two distinct fixes:

- **Secant → bisection fallback.** When the primary secant on the discharge
  residual stalls, re-seed from the well's measured flowing BHP (`ipr_su.pwf`),
  then fall back to a robust bisection over the bracketed root.
- **Suction walk-inward.** When a suction bracket endpoint (`psu_min` / `psu_max`)
  is itself infeasible — the inner throat solve has no solution *at that exact
  point* — walk the suction inward to the nearest feasible point so the outer
  search keeps a valid, bracketed root.

**Why:** marginal wells (small throat ratio + high water cut) physically flow,
but the legacy solver aborted at the infeasible endpoint — *"the pump is in the
well and working, but won't converge in the model."* Characterized across 6,048
combos: every well whose discharge residual crosses zero now solves; only
physically-impossible geometries fail. **Endpoint-feasible / already-converging
solves are bit-identical** (additive change only — no behavior change where it
already worked).

**Guarded by:** `tests/test_asm_solopump.py::TestMarginalConvergence`
(`test_marginal_pump_converges`, `test_marginal_endpoint_is_actually_infeasible`,
`test_several_marginal_pumps_converge`, `test_thin_upper_band_feasibility_converges`)
plus the existing solopump smoke tests.

### 2. `woffl/flow/jetflow.py` — throat-mixture discharge bracketing
New helper `_throat_discharge_bracketed`. `throat_discharge` now calls it where it
used to `raise ConvergenceError` at the 15-iteration secant cap: it Brent-brackets
the momentum-balance root and only raises when there is genuinely **no sign change**
(no solution). Defensive robustness for the compressible throat mixture near the
bubble point; already-converging solves are untouched.

**Guarded by:** the marginal-convergence tests above + the full solver/regression
suite (it rescues secant stalls around a real root; the walk-inward in #1 handles
infeasible endpoints).

---

## NOT upstream — safe to change freely
The joint oil + power-fluid auto-match (`woffl/gui/joint_match.py`), its per-well
**🎯 Auto-match oil + PF** button, the batch core, and everything else under
`woffl/gui/` are **GUI** — ours, never upstreamed. The bulk of the auto-match work
lives there; only the **two** solver files above touch the shared library.

---

## Sync protocol — run this EVERY time you pull upstream

1. **Before merging,** know these patches exist (this file).
2. **Merge upstream into a branch** (never straight onto a release branch). Watch
   for conflicts specifically in `solopump.py` and `jetflow.py`.
3. **After merging, run the full suite:**
   ```bash
   WOFFL_MAX_WORKERS=1 PYTHONPATH=. ./venv/Scripts/python.exe -m pytest tests/ -q
   ```
   If `TestMarginalConvergence` (or any solopump test) **goes red**, an upstream
   merge dropped or altered a local solver fix — re-apply it from this file / git
   history before shipping. **The tests are the safety net: a silently-lost patch
   turns red.** (Baseline: the suite is fully green — 545 tests as of 2026-06-16.)
4. **See the full divergence set** any time:
   ```bash
   git diff <upstream-remote>/main -- woffl/pvt woffl/geometry woffl/flow woffl/assembly
   ```
   Run from the repo root. Anything that shows up there is a local library
   divergence; confirm it's either listed here or genuinely intended.

---

## The real fix
Get these two patches **merged upstream** — your buddy owns `kwellis/woffl`. Once
they land there, the divergence (and this whole risk) disappears: the next sync
just brings them back as upstream code. This file + the regression tests give him
everything he needs to review and accept them. Until then, treat #1 and #2 as
**load-bearing local patches** and never let a sync clobber them.
