# Design Plan — 100% Water Pump Mode

*Status: **IMPLEMENTED 2026-06-22** — all recommended defaults confirmed; full
suite 551 green with the oil path bit-identical. Solver view only for v1.*
*Decision locked: water deliverability comes from the **IPR** (`qwf` = water rate
at `pwf`, same Vogel/PI math).*

> **Implementation note (caught by the e2e test):** the throat mixture is rebuilt
> in **two** places, not one — `jetflow.jetpump_base_calcs` **and**
> `solopump.discharge_residual`. Both must propagate `model_as_water`, or the #4
> zero-oil guard fires mid-solve. Verified numbers (12B, 2000 psi reservoir):
> psu ≈ 913 psig, water ≈ 1142 BWPD, PF ≈ 2792 BWPD, Mach 0.027 (subsonic).
> Final patch list: `upstream_sync.md` #5.

---

## 1. Goal

Let the jet pump solver model a well producing **100% water, no oil** — a
watered-out producer, or a water source / disposal well — and report the suction
pressure, water rate, power-fluid rate, and sonic status, exactly like the oil
workflow does today.

## 2. Why the current model can't do it

The solver is **oil-anchored end to end**: it takes an oil rate from the IPR and
scales every other phase off it. Three spots break at water cut = 1.0:

| Location | Code | Fails at WC=1.0 because |
|---|---|---|
| `pvt/resmix.py::_static_insitu_volm_flow` | `qtot = qoil / yoil` | `yoil = 0` → ÷0 (already guarded → raises a clear `ValueError`) |
| `flow/jetflow.py::throat_wc` (~591) | `qwat_su = qoil_std * wc_su / (1 - wc_su)` | `1 - wc_su = 0` → ÷0 |
| `flow/jetflow.py::throat_wc` | `wc_tm = qwat_tot / (qwat_tot + qoil_std)` | fine numerically, but built on the oil basis |

The model carries **formation water as a multiple of oil rate**, so a finite oil
rate at WC=1.0 implies infinite water — "water, no oil" is genuinely
*inexpressible*, not just a divide-by-zero. Supporting it = **re-anchor the flow
on water rate** when there's no oil.

## 3. Design principles

1. **Additive / bit-identical for oil wells.** Everything keys off an explicit
   opt-in flag; WC < 1.0 takes the exact path it does today. This matches the
   house style for the other library patches (see `upstream_sync.md`).
2. **Safe default unchanged.** With the flag off, WC=1.0 still raises the clear
   `ValueError` (the guard already shipped) and the GUI shows the friendly
   "100% water cut" message + skip. Water mode is deliberate, never silent.
3. **Reuse the IPR** (Scott's call): in water mode the inflow curve *is* the
   water deliverability — `qwf` = water rate at `pwf`, same Vogel/PI. Zero new
   inflow math; just a relabel.

## 4. The switch mechanism (the one subtle part)

`jetpump_base_calcs` (jetflow.py:690) builds an **internal** throat mixture at
line 750:

```python
prop_tm = ResMix(wc_tm, prop_su.fgor, prop_su.oil, prop_su.wat, prop_su.gas)
```

In water mode `wc_tm = 1.0`, so this internal mixture is *also* pure water and
*also* needs the water anchor. That kills the "just key on `wc == 1.0` inside
ResMix" shortcut's safety (it would make WC=1.0 silently water-model for **every**
caller, including batch/optimizer). So:

> **Use an explicit `model_as_water` flag on `ResMix` (default `False`), and
> propagate it into the one internally-constructed `prop_tm`.**

That's a single extra kwarg on line 750. The flag — not the WC value — decides
whether `insitu_volm_flow` water-anchors or raises. Oil callers never set it →
bit-identical; batch/optimizer at WC=1.0 still raise unless explicitly opted in.

## 5. Changes, file by file

### Library (shared → upstream PR)

**`woffl/pvt/resmix.py`**
- `__init__`: add `model_as_water: bool = False`, store it.
- `insitu_volm_flow`: branch —
  - `model_as_water and yoil == 0` → water anchor (new static helper):
    `qwat = mwat/rho_wat`, `qtot = qwat / ywat`, `qoil = 0`, `qgas = ygas*qtot`
    (the input rate is the **water** std rate; use `rho_wat_std`, `wat.density`).
  - `yoil == 0` (flag off) → raise the existing clear `ValueError` (unchanged).
  - else → today's oil anchor (unchanged, bit-identical).

**`woffl/flow/jetflow.py`**
- `throat_wc`: add a `wc_su >= 1.0` branch → `qwat_su = qoil_std` (the anchor
  rate is already the water rate), `wc_tm = 1.0`. Oil path (`wc_su < 1`)
  untouched.
- `jetpump_base_calcs` (line 750): propagate the flag —
  `ResMix(wc_tm, …, model_as_water=prop_su.model_as_water)`.

**`woffl/assembly/solopump.py`**
- `jetpump_solver` output mapping: in water mode the IPR rate lives in the
  `qoil_std` slot but is physically water. Return **oil = 0** and surface the
  rate as formation water (`fwat_bwpd`, which already equals it via the
  `throat_wc` water branch). One small mapping block; no solver-loop changes.

### GUI (ours — free to change)

- **`sidebar.py`**: toggle **"Model as 100% water (no oil)"**. When on: set/lock
  WC = 1.00, relabel the rate input **"Water Rate (qwf)"**, carry
  `model_as_water` on `SimulationParams`.
- **`utils.py`**: pass `model_as_water=params.model_as_water` into the `ResMix`
  build in `create_pvt_components` / the solver call.
- **`tabs/jetpump_solver.py`**: when `model_as_water`, **bypass** the new WC≥1.0
  warn-and-skip and run the solver; relabel the hero (Oil → 0 / **Water** = the
  rate); turn the "100% water cut" warning into an info note.

## 6. Output semantics (water mode)

- Oil = 0
- Formation water = `fwat_bwpd` (the IPR water deliverability at solved suction)
- Power fluid = `qnz_bwpd` (unchanged; PF is water)
- Total water = formation + power fluid
- **Sonic status:** water is ~incompressible → high mixture sound speed → throat
  Mach stays low, so a water pump essentially **won't choke**. The model reports
  this naturally (no sonic decoupling). Physically correct and a useful contrast
  to gassy oil wells — worth surfacing in the hero.

## 7. Test plan

- **New:** `insitu_volm_flow` water-anchor branch returns `qoil==0`, `qwat>0`,
  `qgas==0` at WC=1.0 with `model_as_water=True`; `throat_wc` water branch
  (`wc_su=1.0` → `qwat_su==qoil_std`, `wc_tm==1.0`, no raise); an end-to-end
  `jetpump_solver` water solve on a representative MPU water well converges to a
  sane psu + water rate.
- **Unchanged (tripwire):** the full suite stays green and **bit-identical** for
  oil wells — `tests/test_pvt_resmix.py` (incl. the WC=1.0 *raise* test, which
  still holds with the flag off) and
  `tests/test_asm_solopump.py::TestMarginalConvergence`.

## 8. Upstream impact

Touches three shared library files (`resmix.py`, `jetflow.py`, `solopump.py`),
all **additive**. Becomes **load-bearing patch #5** in `upstream_sync.md`, tagged
`# [LIBRARY change -> upstream PR to kwellis/woffl]`, with the regression tests
as the tripwire. `network_optimizer.py` is not touched.

## 9. Effort / risk

**Moderate, contained.** The physics is well-understood (re-anchor on water); the
real care points are (a) propagating the flag into the internal `prop_tm` and
(b) the output mapping + GUI relabels. Low risk to the oil path because the flag
gates everything and oil wells never set it.

## 10. Open questions for Scott

1. **Toggle UX:** auto-set & lock WC=100% when the toggle is on (recommended), or
   leave WC editable and just require 100%?
2. **Gas:** assume none (GOR is gas-per-oil → 0 at 100% water — recommended), or
   do you ever need free/dissolved gas in the water leg?
3. **Scope:** Solver tab only for v1, or also wire water mode into Batch Run / PF
   Range / the multi-well pages? (Recommend Solver-only first, then extend.)
4. **Power fluid:** still treated as water at `ppf_surf` (unchanged) — correct?
