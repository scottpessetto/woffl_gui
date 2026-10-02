/**
 * Test-row selection helpers shared by the Solver workbench components:
 * stable row keys, anchor-mode resolution, pump-at-date lookup and the
 * picker label format (date + pump + BHP + Liq).
 */

import type { AnchorMode, JpInstallRow, WellTestRow } from "../../api/types";
import { fmtDate, fmtNum } from "../../lib/format";

/** Stable identity for a test row (wt_uid when present, else the date; the
 *  engineer's manual test has its own fixed key - state/manualTest). */
export function testKey(t: WellTestRow): string {
  if (t.manual === true) return "manual";
  return t.wt_uid !== null ? `uid:${t.wt_uid}` : `date:${t.date}`;
}

/** An info-only test: FDC has not accepted it for allocation, so nobody has
 *  reviewed it. Only an explicit `allocated: false` counts - rows from a
 *  server that never sent the flag are all allocated tests. */
export function isInfoOnly(t: WellTestRow): boolean {
  return t.allocated === false;
}

/** The tests an automatic choice may use (server: ipr._fit_tests): never an
 *  info-only test. The engineer's own test counts - they put it there. */
export function allocatedTests(tests: WellTestRow[]): WellTestRow[] {
  return tests.filter((t) => !isInfoOnly(t));
}

/** "Allocated" | "Info only" | "Manual" | "-" for the test table and chart. */
export function testKind(t: WellTestRow): string {
  if (t.manual === true) return "Manual";
  if (isInfoOnly(t)) return "Info only";
  return t.allocated === true ? "Allocated" : "-";
}

/** A specific anchor: the test's wt_uid when it has one, plus its date - or
 *  the engineer's own test, which has neither a wt_uid nor a unique date. */
export interface AnchorPick {
  date: string | null;
  uid: number | null;
  manual?: boolean;
}

/** The pick that names `t` (the anchor picker's option value, as state). */
export function pickOf(t: WellTestRow | undefined): AnchorPick | null {
  return t ? { date: t.date, uid: t.wt_uid, manual: t.manual === true } : null;
}

/**
 * The test row the IPR anchor currently points to.
 *
 * `fitAnchor` is the anchor the FIT RESPONSE reported (coeffs.anchor_wt_uid /
 * anchor_date) - the server's own resolution, preferred whenever it matches a
 * row so the UI can never disagree with the fit it is displaying. Before the
 * fit lands, the local mirror applies over the ALLOCATED tests, the only ones
 * an automatic mode may pick: recent = newest, median / median_liq = the test
 * whose BHP / total fluid sits nearest the window's median of that value (the
 * server's statistic in ipr_anchor._resolve_anchor_row - NOT the middle of
 * the date-sorted list). specific = the picked test, info-only or not, by
 * wt_uid and then by date (falling back to newest when it left the window).
 *
 * "manual" resolves to NULL on purpose: the anchor is the sidebar's own
 * qwf/pwf, so there is no test to pin and the save must not claim one.
 * Callers that need a test for COMPARISON (model vs actual) pick their own
 * fallback rather than borrowing the anchor's.
 */
export function resolveAnchorTest(
  sorted: WellTestRow[],
  mode: AnchorMode,
  pick: AnchorPick | null,
  fitAnchor?: AnchorPick | null,
  /** The engineer switched info-only tests into the fit, so the automatic
   *  modes choose among them too (IprFitRequest.include_info_only). */
  infoInFit = false,
): WellTestRow | null {
  if (sorted.length === 0 || mode === "manual") return null;
  const fitHit = fitAnchor ? findPick(sorted, fitAnchor) : null;
  if (fitHit) return fitHit;
  const auto = infoInFit ? sorted : allocatedTests(sorted);
  if (mode === "median") return medianTest(auto, (t) => t.bhp as number);
  if (mode === "median_liq") return medianTest(auto, (t) => t.total_fluid as number);
  if (mode === "specific" && pick) return findPick(sorted, pick) ?? sorted[0];
  return auto[0] ?? sorted[0];
}

/** The row a pick names: its wt_uid first, else the first row on its date,
 *  an allocated one ahead of an info-only one (the server's date rule). */
function findPick(sorted: WellTestRow[], pick: AnchorPick): WellTestRow | null {
  if (pick.manual) return sorted.find((t) => t.manual === true) ?? null;
  if (pick.uid != null) {
    const hit = sorted.find((t) => t.wt_uid === pick.uid);
    if (hit) return hit;
  }
  if (pick.date == null) return null;
  const day = pick.date.slice(0, 10);
  const onDay = sorted.filter((t) => t.date.slice(0, 10) === day && t.manual !== true);
  return onDay.find((t) => !isInfoOnly(t)) ?? onDay[0] ?? null;
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

/** Picker option label: "2026-05-14 | 13C | BHP 812 | Liq 1,940", with
 *  " | info only" on a test FDC has not allocated and " | your test" on the
 *  engineer's own (LRS / typed) test. */
export function testLabel(t: WellTestRow, pump: string | null): string {
  const parts = [fmtDate(t.date)];
  if (pump) parts.push(pump);
  if (t.bhp !== null) parts.push(`BHP ${fmtNum(t.bhp)}`);
  if (t.total_fluid !== null) parts.push(`Liq ${fmtNum(t.total_fluid)}`);
  if (isInfoOnly(t)) parts.push("info only");
  if (t.manual === true) parts.push("your test");
  return parts.join(" | ");
}
