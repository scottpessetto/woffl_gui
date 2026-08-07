/**
 * Shared math for the combined-permutation study: how many solves the
 * current picker asks for, what one run reads on one metric, and the axis
 * padding the envelope bars need so a target that sits outside the bar is
 * still on screen.
 *
 * The four match quantities themselves come from ./metrics - the combined
 * study and the tornado score the same numbers in the same units, and two
 * copies of that table would drift.
 */

import type { CombineRun, SensitivityKnob, SensitivityPoint } from "../../api/types";
import { METRICS, type MetricId, type MetricSpec } from "./metrics";

/** Server cap on one factorial. Mirrors services.sensitivity.MAX_COMBINE_RUNS;
 *  the picker refuses to fire past it rather than collecting a 422. */
export const MAX_COMBINE_RUNS = 10000;

/** Measured single solve, server side. Only used for the picker's estimate. */
export const SOLVE_MS = 15;

/** Levels offered per knob. Two is the corners, five is as fine as a
 *  factorial stays affordable past three knobs. */
export const LEVEL_CHOICES = [2, 3, 5] as const;

/** Measured test values the study is scored against, as the page holds them. */
export interface CombineTargets {
  target_psu: number | null;
  target_qoil: number | null;
  target_qliq: number | null;
  target_qpf: number | null;
}

const TARGET_KEY: Record<MetricId, keyof CombineTargets> = {
  psu: "target_psu",
  qoil: "target_qoil",
  qliq: "target_qliq",
  qpf: "target_qpf",
};

/** The measured value for one metric, null when the test carried none. */
export function targetOf(targets: CombineTargets, metric: MetricId): number | null {
  const v = targets[TARGET_KEY[metric]];
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

/** One quantity of one run against its measured value. */
export interface RunReading {
  spec: MetricSpec;
  /** modeled value, null when the solve failed */
  value: number | null;
  /** measured test value, null when the test carried none */
  target: number | null;
  /** modeled minus target; null when either side is missing */
  err: number | null;
}

/** All four quantities of one run, in metric-table order. `run` takes a
 *  CombineRun or the study baseline - both carry the same four fields. */
export function runReadings(
  run: CombineRun | SensitivityPoint,
  targets: CombineTargets,
): RunReading[] {
  return METRICS.map((spec) => {
    const raw = run[spec.id];
    const value = typeof raw === "number" && Number.isFinite(raw) ? raw : null;
    const target = targetOf(targets, spec.id);
    return { spec, value, target, err: value !== null && target !== null ? value - target : null };
  });
}

/**
 * Levels actually sent for one knob. A catalog knob has no values between
 * its options, so asking for five levels across two nozzle sizes would
 * inflate the quoted run count against a factorial the server is going to
 * dedupe anyway. Continuous knobs take the picker's choice unchanged.
 *
 * The count the picker shows has to be the count the server checks against
 * its cap, so whatever this returns is what goes on the request.
 */
export function knobLevels(knob: SensitivityKnob, low: number, high: number, levels: number): number {
  if (knob.kind !== "catalog") return levels;
  const options = Math.round(Math.abs(high - low)) + 1;
  return Math.max(2, Math.min(levels, options, 7));
}

/** Time the factorial will take, in words. */
export function estimateLabel(runs: number): string {
  const secs = (runs * SOLVE_MS) / 1000;
  if (secs < 1) return "under a second";
  if (secs < 60) return `about ${secs.toFixed(secs < 10 ? 1 : 0)} s`;
  return `about ${(secs / 60).toFixed(1)} min`;
}

/**
 * Axis ends for a range bar, padded off the data and rounded outward to a
 * readable step. Pass the target in `lo`/`hi` when there is one: the whole
 * point of the bar is whether the target falls inside it, so an axis that
 * crops the target off the end answers nothing.
 *
 * `floorAtZero` keeps a rate axis off negative numbers, which no rate has.
 */
export function niceRange(lo: number, hi: number, floorAtZero: boolean): [number, number] {
  const span = hi - lo;
  const pad = span > 0 ? span * 0.12 : Math.max(Math.abs(hi) * 0.05, 1);
  const rawLo = floorAtZero ? Math.max(0, lo - pad) : lo - pad;
  const rawHi = hi + pad;
  const step = Math.pow(10, Math.floor(Math.log10(Math.max(rawHi - rawLo, 1e-9)))) / 2;
  // Dividing by a decimal step leaves binary dust, and ECharts prints the
  // axis end verbatim - a 0.7000000000000001 tick is nobody's idea of nice.
  const snap = (v: number): number => Number((Math.round(v / step) * step).toFixed(6));
  return [snap(Math.floor(rawLo / step) * step), snap(Math.ceil(rawHi / step) * step)];
}
