/**
 * Optimization pad-board configuration - the engineer's working state for
 * an upcoming optimization run: which wells to treat as offline, planned
 * FUTURE wells that borrow ("match") an existing well's saved fit (the
 * donor may live on any pad), and the last run job id per tab.
 *
 * Persisted to localStorage per browser, like the Well Sort config: this is
 * one engineer's scratch config, not shared field truth. The saved-fit
 * status itself always comes live from prop_hist via /optimize/pad-status.
 */

import { create } from "zustand";

const STORAGE_KEY = "woffl.optimize";

export interface FutureWell {
  name: string; // engineer-chosen label, e.g. "MPL-88"
  match: string; // donor well whose saved fit it copies
}

interface Persisted {
  pad: string | null; // last-viewed pad on the board
  offline: Record<string, string[]>; // pad -> wells checked offline
  future: Record<string, FutureWell[]>; // pad -> planned wells
  /** Run-tab key ("S"|"I"|"M"|"CFP") -> last job id, to re-attach after tab
   * switches/reloads. Server jobs expire after ~1 h; a 404 clears it. */
  lastJob: Record<string, string | null>;
}

function restore(): Persisted {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const p = JSON.parse(raw) as Partial<Persisted>;
      return {
        pad: typeof p.pad === "string" ? p.pad : null,
        offline: p.offline && typeof p.offline === "object" ? p.offline : {},
        future: p.future && typeof p.future === "object" ? p.future : {},
        lastJob: p.lastJob && typeof p.lastJob === "object" ? p.lastJob : {},
      };
    }
  } catch {
    // storage unavailable - defaults still work in-memory
  }
  return { pad: null, offline: {}, future: {}, lastJob: {} };
}

function persist(state: OptimizeState): void {
  try {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        pad: state.pad,
        offline: state.offline,
        future: state.future,
        lastJob: state.lastJob,
      }),
    );
  } catch {
    // ignore - private mode
  }
}

interface OptimizeState extends Persisted {
  setPad: (pad: string) => void;
  toggleOffline: (pad: string, well: string) => void;
  addFuture: (pad: string, fw: FutureWell) => void;
  removeFuture: (pad: string, name: string) => void;
  setLastJob: (runKey: string, jobId: string | null) => void;
}

const initial = restore();

export const useOptimizeStore = create<OptimizeState>((set, get) => ({
  ...initial,

  setPad: (pad) => {
    set({ pad });
    persist(get());
  },

  toggleOffline: (pad, well) => {
    set((s) => {
      const cur = s.offline[pad] ?? [];
      const next = cur.includes(well) ? cur.filter((w) => w !== well) : [...cur, well].sort();
      return { offline: { ...s.offline, [pad]: next } };
    });
    persist(get());
  },

  addFuture: (pad, fw) => {
    set((s) => {
      const cur = s.future[pad] ?? [];
      if (cur.some((f) => f.name === fw.name)) return s; // names are row identity
      return { future: { ...s.future, [pad]: [...cur, fw] } };
    });
    persist(get());
  },

  removeFuture: (pad, name) => {
    set((s) => ({
      future: { ...s.future, [pad]: (s.future[pad] ?? []).filter((f) => f.name !== name) },
    }));
    persist(get());
  },

  setLastJob: (runKey, jobId) => {
    set((s) => ({ lastJob: { ...s.lastJob, [runKey]: jobId } }));
    persist(get());
  },
}));
