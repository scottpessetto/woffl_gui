/**
 * Sensitivity working state - the ranges the engineer typed into the input
 * table, which metric and which input they were reading, and the combined
 * permutation picker along with the id of the last study they fired.
 *
 * Persisted to localStorage per browser, like the optimization pad board:
 * this is one engineer's scratch config, not shared field truth. Keyed BY
 * WELL throughout, because a range that is honest on one well is noise on
 * the next - one well's judgement must never show up under another.
 *
 * The job id is the part actually worth keeping. A combined study runs on
 * the server and outlives this page: walking over to the Solver unmounts
 * the view but the job keeps solving, so without the id a finished result
 * sits unreachable in the registry. Server jobs expire after about an hour;
 * a 404 clears the id.
 *
 * Well SELECTION is not ours - params.ts deliberately does not restore it,
 * and this store only surfaces a well's state when that well is selected
 * again by hand.
 */

import { create } from "zustand";

import type { BoundsMap } from "../pages/sensitivity/bounds";

const STORAGE_KEY = "woffl.sensitivity";

/** What the last fired study varied, captured at Run so the results keep
 *  their own columns when the picker moves on - or when the page is left
 *  and comes back to a job that finished while it was gone. */
export interface FiredStudy {
  ids: string[];
  labels: Record<string, string>;
  count: number;
}

/** Which tornado metric is showing and which input row is expanded. */
export interface ViewState {
  /** MetricId, held loosely: storage is untrusted, so the page resolves it
   *  against METRICS and falls back to the first entry. */
  metricId: string;
  selectedId: string | null;
}

/** The combined-permutation panel for one well. */
export interface CombineState {
  /** input ids ticked in the picker */
  picked: string[];
  levels: number;
  /** re-attach target for the server job; a 404 clears it */
  jobId: string | null;
  /** what was actually submitted, for the result columns */
  fired: FiredStudy | null;
}

interface Persisted {
  bounds: Record<string, BoundsMap>; // well -> the ranges they typed
  view: Record<string, ViewState>; // well -> metric and selected input
  combine: Record<string, CombineState>; // well -> picker + last job
}

/** No overrides: one shared object, so an empty map is identity-stable. */
export const NO_BOUNDS: BoundsMap = {};

/** Untouched view, shared for the same reason. */
export const DEFAULT_VIEW: ViewState = { metricId: "psu", selectedId: null };

/** Untouched picker: nothing ticked, three levels a side. */
export const DEFAULT_COMBINE: CombineState = {
  picked: [],
  levels: 3,
  jobId: null,
  fired: null,
};

function restore(): Persisted {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const p = JSON.parse(raw) as Partial<Persisted>;
      return {
        bounds: p.bounds && typeof p.bounds === "object" ? p.bounds : {},
        view: p.view && typeof p.view === "object" ? p.view : {},
        combine: p.combine && typeof p.combine === "object" ? p.combine : {},
      };
    }
  } catch {
    // storage unavailable - defaults still work in-memory
  }
  return { bounds: {}, view: {}, combine: {} };
}

function persist(state: SensitivityState): void {
  try {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        bounds: state.bounds,
        view: state.view,
        combine: state.combine,
      }),
    );
  } catch {
    // ignore - private mode
  }
}

interface SensitivityState extends Persisted {
  setBounds: (well: string, map: BoundsMap) => void;
  /** Back to the default sweep on one well, without touching the others. */
  resetBounds: (well: string) => void;
  setView: (well: string, view: ViewState) => void;
  setCombine: (well: string, combine: CombineState) => void;
  /** The one field that moves on its own: the job started, or expired. */
  setCombineJob: (well: string, jobId: string | null) => void;
}

const initial = restore();

export const useSensitivityStore = create<SensitivityState>((set, get) => ({
  ...initial,

  setBounds: (well, map) => {
    set((s) => ({ bounds: { ...s.bounds, [well]: map } }));
    persist(get());
  },

  resetBounds: (well) => {
    set((s) => {
      const next = { ...s.bounds };
      delete next[well];
      return { bounds: next };
    });
    persist(get());
  },

  setView: (well, view) => {
    set((s) => ({ view: { ...s.view, [well]: view } }));
    persist(get());
  },

  setCombine: (well, combine) => {
    set((s) => ({ combine: { ...s.combine, [well]: combine } }));
    persist(get());
  },

  setCombineJob: (well, jobId) => {
    set((s) => ({
      combine: { ...s.combine, [well]: { ...(s.combine[well] ?? DEFAULT_COMBINE), jobId } },
    }));
    persist(get());
  },
}));
