/**
 * Memory-gauge state - the SPA equivalent of the Streamlit session store
 * (woffl/gui/memory_gauge.py's _STATE_KEY). Per-well, SESSION-ONLY by
 * design: gauge data is gone on refresh, exactly like the Streamlit app
 * (its docstring floats a parquet-backed v2; same here).
 *
 * The raw File handles are kept so that adding/removing one file re-sends
 * EVERY file to POST /gauge/parse - the multi-file combination (timestamp
 * dedupe -> daily medians) always runs server-side in
 * memory_gauge.MemoryGaugeData, so the SPA and Streamlit can never
 * aggregate differently.
 */

import { create } from "zustand";

import type { GaugeParseResponse } from "../api/types";

export interface WellGauge {
  /** Raw uploads, re-sent wholesale on any add/remove. */
  fileObjects: File[];
  /** Server-combined parse result (daily medians, window, per-file meta). */
  meta: GaugeParseResponse;
  /** date -> bhp lookup built once from meta.daily. */
  dailyByDate: Record<string, number>;
}

interface GaugeState {
  byWell: Record<string, WellGauge>;
  setGauge: (well: string, fileObjects: File[], meta: GaugeParseResponse) => void;
  clearGauge: (well: string) => void;
}

export const useGaugeStore = create<GaugeState>((set) => ({
  byWell: {},

  setGauge: (well, fileObjects, meta) =>
    set((s) => ({
      byWell: {
        ...s.byWell,
        [well]: {
          fileObjects,
          meta,
          dailyByDate: Object.fromEntries(meta.daily.map((d) => [d.date, d.bhp])),
        },
      },
    })),

  clearGauge: (well) =>
    set((s) => {
      const next = { ...s.byWell };
      delete next[well];
      return { byWell: next };
    }),
}));

/** Months of lookback needed to cover the gauge window, rounded UP to a
 * 6-month step (coarse steps keep the fleet-tests TTL cache from being
 * evicted by one-off month values), capped at the server's 60-month max. */
export function gaugeMonths(meta: GaugeParseResponse, baseMonths: number): number {
  const start = new Date(meta.start_date).getTime();
  const needed = Math.ceil((Date.now() - start) / (30 * 24 * 3600 * 1000));
  const stepped = Math.ceil(needed / 6) * 6;
  return Math.min(60, Math.max(baseMonths, stepped));
}
