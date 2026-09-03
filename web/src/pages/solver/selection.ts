/**
 * Test-row selection helpers shared by the Solver workbench components:
 * stable row keys, anchor-mode resolution, pump-at-date lookup and the
 * picker label format (date + pump + BHP + Liq).
 */

import type { AnchorMode, JpInstallRow, WellTestRow } from "../../api/types";
import { fmtDate, fmtNum } from "../../lib/format";

/** Stable identity for a test row (wt_uid when present, else the date). */
export function testKey(t: WellTestRow): string {
  return t.wt_uid !== null ? `uid:${t.wt_uid}` : `date:${t.date}`;
}

/**
 * The test row the IPR anchor currently points to.
 *
 * `fitAnchorDate` is the anchor date the FIT RESPONSE reported
 * (coeffs.anchor_date) - the server's own resolution, preferred whenever it
 * matches a row so the UI can never disagree with the fit it is displaying.
 * Before the fit lands, the local mirror applies: recent = newest, median /
 * median_liq = the test whose BHP / total fluid sits nearest the window's
 * median of that value (the server's statistic in
 * ipr_anchor._resolve_anchor_row - NOT the middle of the date-sorted list),
 * specific = exact date (falling back to newest when the date left the
 * window).
 *
 * "manual" resolves to NULL on purpose: the anchor is the sidebar's own
 * qwf/pwf, so there is no test to pin and the save must not claim one.
 * Callers that need a test for COMPARISON (model vs actual) pick their own
 * fallback rather than borrowing the anchor's.
 */
export function resolveAnchorTest(
  sorted: WellTestRow[],
  mode: AnchorMode,
  anchorDate: string | null,
  fitAnchorDate?: string | null,
): WellTestRow | null {
  if (sorted.length === 0 || mode === "manual") return null;
  if (fitAnchorDate != null) {
    const d = fitAnchorDate.slice(0, 10);
    const hit = sorted.find((t) => t.date.slice(0, 10) === d);
    if (hit) return hit;
  }
  if (mode === "median") return medianTest(sorted, (t) => t.bhp as number);
  if (mode === "median_liq") return medianTest(sorted, (t) => t.total_fluid as number);
  if (mode === "specific" && anchorDate) {
    return sorted.find((t) => t.date === anchorDate) ?? sorted[0];
  }
  return sorted[0];
}

/** The server's median-anchor statistic (ipr_anchor._resolve_anchor_row):
 *  over the FIT-ELIGIBLE rows (BHP and total fluid both present - the fit
 *  drops the rest), the test whose `value` is nearest the median value
 *  (BHP for "median", total fluid for "median_liq"). Pandas median: an
 *  even count averages the two middle values. Ties keep the first row in
 *  newest-first order, matching the server frame. */
function medianTest(
  sorted: WellTestRow[],
  value: (t: WellTestRow) => number,
): WellTestRow | null {
  const rows = sorted.filter((t) => t.bhp != null && t.total_fluid != null);
  if (rows.length === 0) return null;
  const values = rows.map(value).sort((a, b) => a - b);
  const mid = Math.floor(values.length / 2);
  const median =
    values.length % 2 === 1 ? values[mid] : (values[mid - 1] + values[mid]) / 2;
  let best = rows[0];
  let bestD = Math.abs(value(rows[0]) - median);
  for (const t of rows.slice(1)) {
    const d = Math.abs(value(t) - median);
    if (d < bestD) {
      best = t;
      bestD = d;
    }
  }
  return best;
}

/**
 * Pump installed on or before `date`. Tenure is set-to-set - a pump runs
 * until the NEXT Date Set, and Date Pulled is never consulted.
 * ISO strings compare lexicographically, so no Date parsing is needed.
 */
export function pumpAt(
  installs: JpInstallRow[],
  date: string,
): { nozzle: string; throat: string } | null {
  let best: JpInstallRow | null = null;
  for (const row of installs) {
    if (row.date_set === null || row.date_set.slice(0, 10) > date) continue;
    if (best === null || best.date_set === null || row.date_set > best.date_set) {
      best = row;
    }
  }
  if (!best || best.nozzle === null || best.throat === null) return null;
  return { nozzle: best.nozzle, throat: best.throat };
}

/** ``pumpAt`` as a "13C" code. */
export function pumpLabelAt(installs: JpInstallRow[], date: string): string | null {
  const p = pumpAt(installs, date);
  return p && `${p.nozzle}${p.throat}`;
}

/** Picker option label: "2026-05-14 | 13C | BHP 812 | Liq 1,940". */
export function testLabel(t: WellTestRow, pump: string | null): string {
  const parts = [fmtDate(t.date)];
  if (pump) parts.push(pump);
  if (t.bhp !== null) parts.push(`BHP ${fmtNum(t.bhp)}`);
  if (t.total_fluid !== null) parts.push(`Liq ${fmtNum(t.total_fluid)}`);
  return parts.join(" | ");
}
