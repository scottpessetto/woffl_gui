/**
 * A loaded LRS test sheet as the engineer's own test - pure, so web/tests can
 * load it directly. The sheet gives rates, water cut, wellhead and
 * power-fluid pressure; it has NO bottom-hole pressure, so the test's BHP
 * comes from the well's daily gauge feed when it covers the test date and is
 * otherwise left for the engineer (the test cannot anchor without one).
 */

import type { LrsTestResponse, SimParams } from "../../api/types";
import type { ManualTest } from "../../state/manualTest";

type Bounds = Partial<Record<keyof SimParams, [number, number]>>;

export interface LrsLoad {
  test: ManualTest;
  /** What was held back or needs the engineer, in plain words. */
  notes: string[];
}

/** The daily gauge BHP on `date`, else the latest one in the `maxDays` before it. */
export function gaugeBhpNear(
  daily: { date: string; bhp: number }[],
  date: string,
  maxDays = 3,
): { date: string; bhp: number } | null {
  const target = Date.parse(`${date.slice(0, 10)}T00:00:00Z`);
  if (Number.isNaN(target)) return null;
  let best: { date: string; bhp: number } | null = null;
  for (const d of daily) {
    if (!(d.bhp > 0)) continue;
    const t = Date.parse(`${d.date.slice(0, 10)}T00:00:00Z`);
    if (Number.isNaN(t) || t > target || target - t > maxDays * 86_400_000) continue;
    if (best === null || d.date > best.date) best = d;
  }
  return best;
}

const fmt = (v: number, dp = 0) => v.toLocaleString("en-US", { maximumFractionDigits: dp });

/**
 * The sheet as a manual test. `bounds` are the sidebar's input bounds: when
 * the test anchors the IPR its values are laid over the sidebar, so a value
 * outside them is reported here rather than clamped silently later.
 */
export function lrsManualTest(
  sheet: LrsTestResponse,
  date: string,
  bounds: Bounds,
  gauge: { date: string; bhp: number } | null,
): LrsLoad {
  const notes: string[] = [];
  const check = (key: keyof SimParams, value: number | null, label: string, unit: string) => {
    const range = bounds[key];
    if (value !== null && range && (value < range[0] || value > range[1])) {
      notes.push(`${label} ${fmt(value, 1)} ${unit} is outside the sidebar's ${fmt(range[0])}-${fmt(range[1])} range; as the anchor it is clamped there.`);
    }
  };
  check("qwf", sheet.total_fluid, "Formation fluid rate", "BLPD");
  check("surf_pres", sheet.whp, "Wellhead pressure", "psi");
  check("ppf_surf", sheet.pf_press, "Power-fluid pressure", "psi");

  // A GOR below the model's floor is the sheet reporting no gas rate, not a
  // measured ratio: the test carries no GOR and the well's own is kept.
  let gor = sheet.gor;
  const gorRange = bounds.form_gor;
  if (gor !== null && gorRange && (gor < gorRange[0] || gor > gorRange[1])) {
    notes.push(`GOR ${fmt(gor, 1)} scf/stb on the sheet is outside the model's ${fmt(gorRange[0])}-${fmt(gorRange[1])} range (no gas rate measured?), so the test carries no GOR and the well's GOR is kept.`);
    gor = null;
  }

  if (gauge) {
    notes.push(`The sheet has no BHP: the test uses the daily gauge reading of ${fmt(gauge.bhp)} psi on ${gauge.date.slice(0, 10)}. Change it below if the test saw something else.`);
  } else {
    notes.push("The sheet has no BHP and the daily gauge has no reading near the test date. Enter the BHP below - the test cannot anchor the IPR without one.");
  }
  if (sheet.pf_rate === null) notes.push("No power-fluid rate on the sheet - enter it below.");

  return {
    test: {
      date,
      oil: sheet.oil,
      water: sheet.water,
      bhp: gauge ? gauge.bhp : null,
      gor,
      whp: sheet.whp,
      pfRate: sheet.pf_rate,
      pfPress: sheet.pf_press,
      source: `${sheet.location ?? "LRS"} (${sheet.filename})`,
    },
    notes,
  };
}
