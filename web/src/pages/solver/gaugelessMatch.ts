/**
 * Gates and cross-checks for "Match the test (no gauge)". Kept free of
 * runtime imports so the node tests can load it directly.
 *
 * The match infers BHP from the power-fluid rate with the nozzle held at its
 * catalog area, so a worn nozzle reads as a LOW BHP - and a low BHP makes the
 * IPR look pumped-off, which is how a worn 12 ended up "recommending" a 9B
 * (user report 2026-09-29). Two consequences:
 *   - a test that carries a gauge BHP never gets the match: the gauge is the
 *     better measurement, and Calibrate to field data already uses it;
 *   - a gaugeless match is cross-checked against any gauge reading on the
 *     same pump, and steered toward a test early in the pump's life, when the
 *     nozzle is closest to catalog.
 */

import type { WellTestRow } from "../../api/types";

/** A test this far into its pump's life is "early" - little nozzle wear yet. */
export const EARLY_LIFE_DAYS = 90;
/** Inferred-vs-gauge BHP gap that is flagged (floored by the match's own resolution). */
export const GAUGE_GAP_PSI = 150;
/** Without a pump set date, only gauge readings this close to the test count. */
const NO_SET_DATE_WINDOW_DAYS = 90;

const DAY_MS = 86_400_000;

function days(a: string, b: string): number {
  return (Date.parse(a.slice(0, 10)) - Date.parse(b.slice(0, 10))) / DAY_MS;
}

/** The test carries a downhole BHP (tracker gauge or an uploaded memory gauge). */
export function hasGaugeBhp(t: WellTestRow | null): boolean {
  return t !== null && t.bhp !== null && t.bhp > 0;
}

/** A test the gaugeless match can run on: oil and PF present, no gauge BHP. */
export function isGaugelessMatchable(t: WellTestRow): boolean {
  return t.oil !== null && t.oil > 0 && t.lift_wat !== null && t.lift_wat > 0 && !hasGaugeBhp(t);
}

/** Days from the pump's set date to the test; null when either is unknown. */
export function daysIntoPump(test: WellTestRow, dateSet: string | null): number | null {
  if (!dateSet) return null;
  const d = days(test.date, dateSet);
  return Number.isFinite(d) ? d : null;
}

/**
 * The earliest matchable test on the SAME pump when the selected test is past
 * the early-life window - the better test to infer BHP from. Null when the
 * selected test is already early, the set date is unknown, or nothing earlier
 * is matchable.
 */
export function earlierTestInPumpLife(
  tests: WellTestRow[],
  selected: WellTestRow,
  dateSet: string | null,
): WellTestRow | null {
  const age = daysIntoPump(selected, dateSet);
  if (age === null || age <= EARLY_LIFE_DAYS || !dateSet) return null;
  const set = dateSet.slice(0, 10);
  let best: WellTestRow | null = null;
  for (const t of tests) {
    const d = t.date.slice(0, 10);
    if (d < set || d >= selected.date.slice(0, 10) || !isGaugelessMatchable(t)) continue;
    if (best === null || d < best.date.slice(0, 10)) best = t;
  }
  return best;
}

/**
 * The gauge BHP nearest in time to the selected test on the same pump (on or
 * after its set date). Without a set date, only readings within 90 days count.
 */
export function nearestGaugeBhp(
  tests: WellTestRow[],
  selected: WellTestRow,
  dateSet: string | null,
): { bhp: number; date: string } | null {
  const set = dateSet ? dateSet.slice(0, 10) : null;
  let best: { bhp: number; date: string; gap: number } | null = null;
  for (const t of tests) {
    if (!hasGaugeBhp(t)) continue;
    if (set !== null && t.date.slice(0, 10) < set) continue;
    const gap = Math.abs(days(t.date, selected.date));
    if (!Number.isFinite(gap) || (set === null && gap > NO_SET_DATE_WINDOW_DAYS)) continue;
    if (best === null || gap < best.gap) best = { bhp: t.bhp as number, date: t.date, gap };
  }
  return best && { bhp: best.bhp, date: best.date };
}

/** Whether an inferred BHP disagrees with a gauge reading by more than the flag threshold. */
export function gaugeGapFlagged(inferred: number, gauge: number, resolutionPsi: number | null): boolean {
  return Math.abs(inferred - gauge) > Math.max(GAUGE_GAP_PSI, resolutionPsi ?? 0);
}
