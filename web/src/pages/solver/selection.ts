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
 * The test row the IPR anchor currently points to. Mirrors the server's
 * anchor semantics: recent = newest, median = middle of the date-sorted
 * window, specific = exact date (falling back to newest when the date
 * left the window).
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
): WellTestRow | null {
  if (sorted.length === 0 || mode === "manual") return null;
  if (mode === "median") return sorted[Math.floor((sorted.length - 1) / 2)];
  if (mode === "specific" && anchorDate) {
    return sorted.find((t) => t.date === anchorDate) ?? sorted[0];
  }
  return sorted[0];
}

/**
 * Pump installed on or before `date`. Tenure is set-to-set - a pump runs
 * until the NEXT Date Set, and Date Pulled is never consulted.
 * Mirror of woffl/gui/ipr_viz.py:_pump_label_at_date (ISO strings compare
 * lexicographically, so no Date parsing is needed).
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
