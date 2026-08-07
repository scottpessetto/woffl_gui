/**
 * Engineer-set sweep bounds: the one shape shared by the knob table, where
 * the ranges get typed, and the combined-permutations panel, which explores
 * them. A knob id absent from the map means "no override, sweep the default
 * range", so an untouched page posts an empty object.
 *
 * Values are in the knob's OWN units, exactly as the server reads them:
 * absolute field values for continuous knobs (form_gor in scf/bbl, ken
 * unitless, pressures in psi) and 0-based catalog indices for the pump
 * identity knobs. Which one a knob is is `kind`, not a hardcoded id list.
 */

import type { KnobBounds, SensitivityKnob } from "../../api/types";

/** Knob id to the engineer's override. Missing id = that knob's default. */
export type BoundsMap = Record<string, KnobBounds>;

/** True when the knob walks a parts catalog, so its bounds are indices. */
export function isCatalogKnob(knob: SensitivityKnob): boolean {
  return knob.kind === "catalog";
}

/**
 * Decimals the server formats this knob's labels with, read back off
 * `baseline_label` - the knob model carries the labels, not the precision.
 * Drives the input step, so a pressure moves by 1 psi and ken by 0.001.
 */
export function knobDecimals(knob: SensitivityKnob): number {
  const dot = knob.baseline_label.indexOf(".");
  return dot < 0 ? 0 : knob.baseline_label.length - dot - 1;
}

/** Spinner step for one knob's inputs: one unit in its last decimal place. */
export function knobStep(knob: SensitivityKnob): number {
  const dp = knobDecimals(knob);
  return dp <= 0 ? 1 : Math.pow(10, -dp);
}

/**
 * The default range as the editor should show it. `default_low` and
 * `default_high` come back PRE-clamp - the frozen table range resolved to
 * absolute values - so bubble_point off a 2600 psi base offers a 3120 psi
 * high against a 2999 psi limit. Clamping here keeps the box inside the
 * limits it advertises and matches what the server actually swept.
 */
export function defaultRange(knob: SensitivityKnob): { low: number; high: number } {
  return {
    low: clampKnobValue(knob, knob.default_low),
    high: clampKnobValue(knob, knob.default_high),
  };
}

/** The range the editor shows: the override when there is one, else default. */
export function effectiveRange(
  knob: SensitivityKnob,
  bounds: BoundsMap,
): { low: number; high: number } {
  if (knob.id in bounds) {
    const override = bounds[knob.id];
    return { low: override.low, high: override.high };
  }
  return defaultRange(knob);
}

/**
 * Hold a value inside the hard limits the sidebar and the model enforce, so
 * the state can never carry a range the solver would refuse. The server
 * clamps too; this keeps the number in the box honest about it.
 */
export function clampKnobValue(knob: SensitivityKnob, value: number): number {
  let out = value;
  if (knob.clamp_low !== null && out < knob.clamp_low) out = knob.clamp_low;
  if (knob.clamp_high !== null && out > knob.clamp_high) out = knob.clamp_high;
  // A catalog bound is an index into the option list, and an index off the
  // end of the catalog is not a pump.
  const options = knob.options;
  if (isCatalogKnob(knob) && options !== null && options.length > 0) {
    out = Math.min(Math.max(Math.round(out), 0), options.length - 1);
  }
  return out;
}

/** Catalog option at an index, e.g. 3 to "14C"; the bare index if adrift. */
export function catalogLabel(knob: SensitivityKnob, value: number): string {
  const index = Math.round(value);
  const options = knob.options;
  if (options === null || index < 0 || index >= options.length) return String(index);
  return options[index];
}

/**
 * One bound as text: the catalog option, else fixed to the knob's decimals.
 * No thousands separators - the string goes straight into a number input.
 */
export function fmtBound(knob: SensitivityKnob, value: number): string {
  return isCatalogKnob(knob) ? catalogLabel(knob, value) : value.toFixed(knobDecimals(knob));
}

/** Same bound to the precision the knob is displayed at. */
function sameBound(knob: SensitivityKnob, a: number, b: number): boolean {
  if (isCatalogKnob(knob)) return Math.round(a) === Math.round(b);
  return Math.abs(a - b) < knobStep(knob) / 2;
}

/** The map without one knob's override, back to its default range. */
function withoutKnob(bounds: BoundsMap, id: string): BoundsMap {
  const out: BoundsMap = {};
  for (const key of Object.keys(bounds)) {
    if (key !== id) out[key] = bounds[key];
  }
  return out;
}

/**
 * One edited endpoint folded into the map, clamped to the knob's hard limits.
 * A pair that lands back on the default range drops out entirely rather than
 * riding along as a no-op override, so the row stops reading as edited.
 *
 * A low above the high is left as typed: the server swaps a crossed pair, and
 * reordering under the engineer mid-edit would move the box they are in.
 */
export function withBound(
  bounds: BoundsMap,
  knob: SensitivityKnob,
  side: "low" | "high",
  value: number,
): BoundsMap {
  const current = effectiveRange(knob, bounds);
  const edited = clampKnobValue(knob, value);
  const next: KnobBounds = {
    low: side === "low" ? edited : current.low,
    high: side === "high" ? edited : current.high,
    steps: bounds[knob.id]?.steps ?? null,
  };
  const fallback = defaultRange(knob);
  if (
    next.steps === null &&
    sameBound(knob, next.low, fallback.low) &&
    sameBound(knob, next.high, fallback.high)
  ) {
    return withoutKnob(bounds, knob.id);
  }
  return { ...bounds, [knob.id]: next };
}
