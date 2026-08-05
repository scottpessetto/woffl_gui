/**
 * Model-vs-actual grading and washout detection - mirrors of
 * tabs/jetpump_solver._match_grade (15%/30% bands) and
 * wc_washout.detect_wc_washout thresholds.
 */

import type { SolveResult, WellTestRow } from "../api/types";

export type Grade = "good" | "fair" | "poor" | "na";

export const GRADE_COLORS: Record<Grade, string> = {
  good: "#1a7f37",
  fair: "#bf8700",
  poor: "#c9252d",
  na: "#64748b",
};

export interface MatchGrade {
  grade: Grade;
  errPct: number | null;
}

export function matchGrade(modeled: number | null, actual: number | null): MatchGrade {
  if (
    modeled === null ||
    actual === null ||
    !Number.isFinite(modeled) ||
    !Number.isFinite(actual) ||
    actual === 0
  ) {
    return { grade: "na", errPct: null };
  }
  const errPct = (Math.abs(modeled - actual) / Math.abs(actual)) * 100;
  return { grade: errPct < 15 ? "good" : errPct < 30 ? "fair" : "poor", errPct };
}

export interface ComparisonRow {
  label: string;
  unit: string;
  modeled: number | null;
  actual: number | null;
  delta: number | null;
  grade: MatchGrade;
  dp: number; // display decimals
}

/** Actuals from the comparison test row (mirror of _actuals_from_test). */
export function actualsFromTest(test: WellTestRow | null): {
  oil: number | null;
  bhp: number | null;
  pf: number | null;
} {
  if (!test) return { oil: null, bhp: null, pf: null };
  return {
    oil: test.oil ?? null,
    bhp: test.bhp ?? null,
    pf: test.lift_wat ?? null,
  };
}

export function buildComparisonRows(solve: SolveResult, test: WellTestRow | null): ComparisonRow[] {
  const actuals = actualsFromTest(test);
  const water = test?.water ?? null;
  const rows: Array<[string, string, number | null, number | null, number]> = [
    ["Oil", "BOPD", solve.qoil_std, actuals.oil, 0],
    ["Formation water", "BWPD", solve.fwat_bwpd, water, 0],
    ["Power fluid", "BWPD", solve.qnz_bwpd, actuals.pf, 0],
    ["Suction BHP", "psi", solve.psu, actuals.bhp, 0],
  ];
  return rows.map(([label, unit, modeled, actual, dp]) => ({
    label,
    unit,
    modeled,
    actual,
    delta: modeled !== null && actual !== null ? modeled - actual : null,
    grade: matchGrade(modeled, actual),
    dp,
  }));
}

export interface WashoutFlag {
  reason: string;
}

/**
 * PF-dominated allocation washout (the MPE-19 lesson): reported WC near zero
 * while produced fluid is a sliver of the PF rate and the model misses BHP
 * or oil badly. Mirror of wc_washout.detect_wc_washout.
 */
export function detectWcWashout(args: {
  formWc: number;
  pfRate: number | null;
  producedFluid: number | null;
  modeledPsu: number;
  actualBhp: number | null;
  modeledOil: number;
  actualOil: number | null;
}): WashoutFlag | null {
  const { formWc, pfRate, producedFluid, modeledPsu, actualBhp, modeledOil, actualOil } = args;
  if (formWc > 0.05) return null;
  if (pfRate === null || producedFluid === null || pfRate <= 0) return null;
  if (producedFluid / pfRate > 0.15) return null;
  const bhpMiss = actualBhp !== null && Math.abs(modeledPsu - actualBhp) >= 100;
  const oilMiss =
    actualOil !== null && actualOil !== 0 && (Math.abs(modeledOil - actualOil) / Math.abs(actualOil)) * 100 >= 15;
  if (!bhpMiss && !oilMiss) return null;
  return {
    reason:
      "Reported water cut is ~0% while produced fluid is a small fraction of the power-fluid rate, " +
      "and the model misses the test badly. The allocated WC may be washed out by PF returns - " +
      "treat the reported WC as suspect.",
  };
}
