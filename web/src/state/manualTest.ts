/**
 * The engineer's own well test - an LRS portable-separator test loaded from
 * its summary sheet, or numbers typed in. Per well and SESSION-ONLY, like the
 * memory gauge: it has no Databricks row, so it is gone on refresh.
 *
 * The test holds its OWN measured numbers. It is a test like any other in the
 * Solver: listed, plotted, selectable as the IPR anchor (the fit then writes
 * the anchor into the sidebar), and part of the reservoir-pressure fit. It is
 * never the sidebar's values by reference - measurements must not move when a
 * model input does.
 */

import { create } from "zustand";

import type { ManualTestFitRow, SimParams, WellTestRow } from "../api/types";

export interface ManualTest {
  date: string; // YYYY-MM-DD
  oil: number | null; // BOPD
  water: number | null; // BWPD formation
  /** Flowing BHP, psi. An LRS sheet has none: it comes from the daily gauge
   *  or the engineer, and the test cannot anchor without it. */
  bhp: number | null;
  gor: number | null; // scf/stb; null = not measured
  whp: number | null; // psi
  /** Measured power-fluid rate, BWPD. */
  pfRate: number | null;
  pfPress: number | null; // psi
  /** Where it came from, e.g. "LRS Unit 6 (E-48 test.xlsx)"; null when typed in. */
  source: string | null;
}

/** testKey of the manual row - it has no wt_uid and must never collide with
 *  a measured test on the same date. */
export const MANUAL_TEST_KEY = "manual";

interface ManualTestState {
  byWell: Record<string, ManualTest>;
  setManualTest: (well: string, test: ManualTest) => void;
  clearManualTest: (well: string) => void;
}

export const useManualTestStore = create<ManualTestState>((set) => ({
  byWell: {},
  setManualTest: (well, test) => set((s) => ({ byWell: { ...s.byWell, [well]: test } })),
  clearManualTest: (well) =>
    set((s) => {
      const next = { ...s.byWell };
      delete next[well];
      return { byWell: next };
    }),
}));

/** Today as YYYY-MM-DD in local time (the default manual-test date). */
export function todayIso(): string {
  const d = new Date();
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

/** A hand-entered test starts from the sidebar's inflow point, so an engineer
 *  who typed the test there only has to add the PF rate. `qwf` is TOTAL
 *  LIQUID: oil and water derive from it at the sidebar water cut. */
export function manualTestFromSidebar(params: SimParams): ManualTest {
  const round1 = (v: number) => Math.round(v * 10) / 10;
  return {
    date: todayIso(),
    oil: round1(params.qwf * (1 - params.form_wc)),
    water: round1(params.qwf * params.form_wc),
    bhp: params.pwf,
    gor: params.form_gor,
    whp: params.surf_pres,
    pfRate: null,
    pfPress: params.ppf_surf,
    source: null,
  };
}

/** The manual test as a test row (null rates stay null: unmeasured). */
export function manualTestRow(test: ManualTest): WellTestRow {
  const total = test.oil !== null && test.water !== null ? test.oil + test.water : null;
  return {
    wt_uid: null,
    manual: true,
    allocated: null,
    date: test.date,
    oil: test.oil,
    water: test.water,
    gas: null,
    total_fluid: total,
    form_wc: total !== null && total > 0 && test.water !== null ? test.water / total : null,
    bhp: test.bhp,
    fgor: test.gor,
    lift_wat: test.pfRate,
    whp: test.whp,
    pf_press: test.pfPress,
    pf_source: null,
  };
}

/** The test as the fit request carries it, or null while it cannot take part
 *  in a fit (no rate or no BHP yet). */
export function manualFitRow(test: ManualTest): ManualTestFitRow | null {
  if (test.oil === null || test.water === null || test.bhp === null) return null;
  const total = test.oil + test.water;
  if (!(total > 0) || !(test.bhp > 0)) return null;
  return {
    date: test.date,
    total_fluid: total,
    water: test.water,
    bhp: test.bhp,
    fgor: test.gor,
    whp: test.whp,
    pf_press: test.pfPress,
  };
}

/** Save-comment provenance: the saved curve came from this test, which is in
 *  no database. */
export function manualTestNote(test: ManualTest): string {
  const n = (v: number | null, unit: string) =>
    v === null ? null : `${v.toLocaleString("en-US", { maximumFractionDigits: 0 })} ${unit}`;
  const parts = [
    n(test.oil, "BOPD oil"),
    n(test.water, "BWPD water"),
    n(test.bhp, "psi BHP"),
    test.pfRate === null ? null : `PF ${n(test.pfRate, "BWPD")}${test.pfPress === null ? "" : ` at ${n(test.pfPress, "psi")}`}`,
  ].filter((p): p is string => p !== null);
  return `${test.source ?? "Manual test"} ${test.date}: ${parts.join(", ")}`;
}
