/**
 * The single source of truth for simulation inputs - the replacement for
 * Streamlit's session_state two-tier key system. One store, one params
 * object, no widget-GC mirrors, no clamp-on-rerun surprises.
 *
 * Seeding: when a well is selected the server replays the sidebar seeding
 * pipeline (chars -> pump history -> IPR fit -> saved-IPR overlay -> live PF)
 * and returns a complete `seeds` partial. `applyContext` lays it over the
 * defaults ONCE per (well, context fetch); manual edits afterwards persist
 * until the well changes or seeds are re-applied explicitly.
 */

import { create } from "zustand";

import type { PropLock, SimParams, WellContext } from "../api/types";
import { DEFAULT_PARAMS, PARAM_BOUNDS } from "../api/types";

export interface AsBuiltLocks {
  tubing: boolean;
  casing: boolean;
  jpump_tvd: boolean;
}

export interface PropLocks {
  form_wc: PropLock;
  form_gor: PropLock;
  res_pres: PropLock;
}

const NO_AS_BUILT: AsBuiltLocks = { tubing: false, casing: false, jpump_tvd: false };
const NO_PROP_LOCKS: PropLocks = {
  form_wc: { locked: false, value: null },
  form_gor: { locked: false, value: null },
  res_pres: { locked: false, value: null },
};

export function clampToBounds<K extends keyof SimParams>(key: K, value: SimParams[K]): SimParams[K] {
  const bounds = PARAM_BOUNDS[key];
  if (!bounds || typeof value !== "number" || Number.isNaN(value)) return value;
  const [lo, hi] = bounds;
  return Math.min(hi, Math.max(lo, value)) as SimParams[K];
}

/**
 * Lay a partial over a params object, clamping each numeric field to its
 * widget bounds. The single localized cast bridges TypeScript's correlated
 * union limitation (generic per-key assignment inside a keyed loop).
 */
export function mergeClamped(base: SimParams, partial: Partial<SimParams>): SimParams {
  const next = { ...base };
  for (const key of Object.keys(partial) as Array<keyof SimParams>) {
    const value = partial[key];
    if (value === null || value === undefined) continue;
    (next[key] as SimParams[typeof key]) = clampToBounds(key, value as SimParams[typeof key]);
  }
  return next;
}

interface ParamsState {
  well: string; // "Custom" | "MPB-28" ...
  params: SimParams;
  /** true once a well is selected or Run pressed - gates auto-solve */
  simActive: boolean;
  /** test window (sidebar Well Test History expander) */
  months: number;
  cap: number;
  asBuiltLocks: AsBuiltLocks;
  propLocks: PropLocks;
  context: WellContext | null;
  /** well name the current context seeds were applied for (dedupe) */
  seededFor: string | null;

  set: <K extends keyof SimParams>(key: K, value: SimParams[K]) => void;
  setMany: (partial: Partial<SimParams>) => void;
  setWindow: (months: number, cap: number) => void;
  selectWell: (name: string) => void;
  applyContext: (ctx: WellContext) => void;
  run: () => void;
}

const WELL_STORAGE_KEY = "woffl.well";

function persistedWell(): string {
  try {
    return localStorage.getItem(WELL_STORAGE_KEY) || "Custom";
  } catch {
    return "Custom";
  }
}

const initialWell = persistedWell();

export const useParamsStore = create<ParamsState>((set) => ({
  // Restore the last-selected well across reloads; context fetch + seeding
  // re-run exactly as they do for a fresh selection.
  well: initialWell,
  params: { ...DEFAULT_PARAMS },
  simActive: initialWell !== "Custom",
  months: 6,
  cap: 0,
  asBuiltLocks: NO_AS_BUILT,
  propLocks: NO_PROP_LOCKS,
  context: null,
  seededFor: null,
  set: (key, value) =>
    set((s) => ({ params: { ...s.params, [key]: clampToBounds(key, value) } })),

  setMany: (partial) => set((s) => ({ params: mergeClamped(s.params, partial) })),

  setWindow: (months, cap) => set({ months, cap, seededFor: null }),

  selectWell: (name) => {
    try {
      localStorage.setItem(WELL_STORAGE_KEY, name);
    } catch {
      // storage unavailable (private mode) - selection still works in-memory
    }
    return set(() => ({
      well: name,
      // Custom = clean bench; a named well seeds on context arrival.
      params: { ...DEFAULT_PARAMS },
      simActive: name !== "Custom",
      asBuiltLocks: NO_AS_BUILT,
      propLocks: NO_PROP_LOCKS,
      context: null,
      seededFor: null,
    }));
  },

  applyContext: (ctx) =>
    set((s) => {
      if (s.well !== ctx.well) return s; // stale response for a superseded selection
      const seeded = mergeClamped({ ...DEFAULT_PARAMS }, ctx.seeds);
      // Preserve sweep selections across seed application: they are user
      // workflow state, not well data.
      seeded.nozzle_batch_options = s.params.nozzle_batch_options;
      seeded.throat_batch_options = s.params.throat_batch_options;
      seeded.water_type = s.params.water_type;
      return {
        params: seeded,
        context: ctx,
        seededFor: ctx.well,
        simActive: true,
        asBuiltLocks: { ...NO_AS_BUILT, ...ctx.as_built_locks },
        propLocks: { ...NO_PROP_LOCKS, ...ctx.prop_locks },
      };
    }),

  run: () => set({ simActive: true }),
}));

/** Effective params sent to the solver: dewatering forces form_wc = 1. */
export function effectiveParams(params: SimParams): SimParams {
  return params.model_as_water ? { ...params, form_wc: 1.0 } : params;
}
