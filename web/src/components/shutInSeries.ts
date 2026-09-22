/**
 * Shut-in handling for the production-history chart.
 *
 * Well tests are sparse, and a line through them reads as continuous
 * production. The server sends the downtime log's shut-in windows
 * (`shut_in`, inclusive YYYY-MM-DD days); these pure helpers turn them into
 * explicit zero-rate segments for the stacked Oil/Form Water areas and into
 * null breaks for pressure lines (a shut-in BHP is not zero psi, and a line
 * drawn across the shut-in would be invented). Kept free of app imports so
 * web/tests can load it directly.
 */

export interface ShutInWindow {
  start: string;
  end: string;
  days?: number | null;
  code?: string | null;
  reason?: string | null;
}

/** A shut-in span in epoch ms, half-open: [start, end). */
export interface ShutInSpan {
  start: number;
  end: number;
  window: ShutInWindow;
}

export interface RatePoint {
  x: number;
  oil: number;
  water: number;
}

const DAY_MS = 86_400_000;

function dayStart(value: unknown): number | null {
  if (typeof value !== "string" || value.length < 10) return null;
  const t = Date.parse(`${value.slice(0, 10)}T00:00:00Z`);
  return Number.isNaN(t) ? null : t;
}

/** The payload's windows, tolerating an older server that omits the field. */
export function shutInWindowsOf(data: unknown): ShutInWindow[] {
  const raw = (data as { shut_in?: unknown } | null)?.shut_in;
  return Array.isArray(raw) ? (raw as ShutInWindow[]) : [];
}

/**
 * Windows -> sorted, non-overlapping spans from the start of the first shut-in
 * day to the end of the last. A day that carries a well test is measured
 * production and wins: the span is split around that day, so a zero segment
 * never contradicts a plotted test.
 */
export function shutInSpans(windows: readonly ShutInWindow[], testTimes: readonly number[] = []): ShutInSpan[] {
  const testDays = new Set(testTimes.filter(Number.isFinite).map((t) => Math.floor(t / DAY_MS) * DAY_MS));
  const spans: ShutInSpan[] = [];
  const sorted = windows
    .map((w) => ({ w, s: dayStart(w.start), e: dayStart(w.end) }))
    .filter((r): r is { w: ShutInWindow; s: number; e: number } => r.s !== null && r.e !== null && r.e >= r.s)
    .sort((a, b) => a.s - b.s);
  for (const { w, s, e } of sorted) {
    let from = s;
    for (let day = s; day <= e; day += DAY_MS) {
      if (!testDays.has(day)) continue;
      if (day > from) spans.push({ start: from, end: day, window: w });
      from = day + DAY_MS;
    }
    if (e + DAY_MS > from) spans.push({ start: from, end: e + DAY_MS, window: w });
  }
  // Merge anything touching or overlapping (defensive: the server collapses).
  const merged: ShutInSpan[] = [];
  for (const span of spans) {
    const last = merged[merged.length - 1];
    if (last && span.start <= last.end) last.end = Math.max(last.end, span.end);
    else merged.push({ ...span });
  }
  return merged;
}

/** The span containing x, if any. */
export function shutInAt(spans: readonly ShutInSpan[], x: number): ShutInSpan | undefined {
  return spans.find((s) => x >= s.start && x < s.end);
}

/** Linear test-to-test value at x; holds the one neighbour that exists. */
function rateAt(points: readonly RatePoint[], x: number): RatePoint {
  let before: RatePoint | undefined;
  let after: RatePoint | undefined;
  for (const p of points) {
    if (p.x <= x) before = p;
    else { after = p; break; }
  }
  if (!before || !after) {
    const held = before ?? after;
    return { x, oil: held?.oil ?? 0, water: held?.water ?? 0 };
  }
  const f = after.x === before.x ? 1 : (x - before.x) / (after.x - before.x);
  return { x, oil: before.oil + (after.oil - before.oil) * f, water: before.water + (after.water - before.water) * f };
}

/**
 * Oil and Form Water points (one shared x sequence, so the ECharts stack
 * stays index-aligned) with each shut-in drawn as a step to zero: the
 * test-to-test interpolation runs up to the shut-in, drops vertically to 0,
 * holds 0 to the end of the last shut-in day, then steps back up to the
 * interpolated value and continues to the next test. A shut-in before the
 * first test or after the last one starts or ends at zero instead of
 * inventing a rate on the far side.
 */
export function rateSeriesWithShutIns(tests: readonly RatePoint[], spans: readonly ShutInSpan[]):
  { oil: [number, number][]; water: [number, number][] } {
  const sorted = [...tests].sort((a, b) => a.x - b.x);
  const rows: RatePoint[] = [];
  let i = 0;
  for (const span of spans) {
    while (i < sorted.length && sorted[i].x < span.start) rows.push(sorted[i++]);
    if (sorted.some((p) => p.x < span.start)) rows.push(rateAt(sorted, span.start));
    rows.push({ x: span.start, oil: 0, water: 0 });
    // Tests inside a span cannot occur when shutInSpans saw the same tests
    // (their days split the span); a test stamped exactly at the span end
    // belongs after the zero segment.
    while (i < sorted.length && sorted[i].x < span.end) i++;
    rows.push({ x: span.end, oil: 0, water: 0 });
    if (sorted.some((p) => p.x >= span.end)) rows.push(rateAt(sorted, span.end));
  }
  while (i < sorted.length) rows.push(sorted[i++]);
  return { oil: rows.map((r) => [r.x, r.oil]), water: rows.map((r) => [r.x, r.water]) };
}

/**
 * Pressure (or any measured) points with a null at each span boundary, so an
 * ECharts line (connectNulls false) breaks across the shut-in. Real samples
 * recorded inside the span are kept - they form their own segment and the
 * tooltip labels them as shut-in readings.
 */
export function breakAtShutIns(points: readonly (readonly [number, number | null])[], spans: readonly ShutInSpan[]):
  [number, number | null][] {
  const out: [number, number | null][] = [];
  const breaks = spans.flatMap((s) => [s.start, s.end]);
  let b = 0;
  const sorted = [...points].sort((a, c) => a[0] - c[0]);
  for (const [x, y] of sorted) {
    while (b < breaks.length && breaks[b] <= x) out.push([breaks[b++], null]);
    out.push([x, y]);
  }
  // Boundaries after the last sample have nothing to break.
  return out;
}
