/**
 * Well Sort shared configuration - the replacement for the Streamlit
 * session keys `well_sort_pops_pads` / `well_sort_pops_force_true` and the
 * per-pad pump-limit edits. All three Well Sort views (Wells, Triage,
 * Marginal WC) read the same POPs selection, exactly like the old page.
 *
 * Persisted to localStorage so an engineer's field configuration survives
 * reloads (an upgrade over session-scoped Streamlit state; the old
 * mirror-key GC machinery is unnecessary here).
 */

import { create } from "zustand";

/** Static default until /well-sort/tables echoes the server's list. */
export const DEFAULT_POPS_PADS = ["E", "F", "H", "I", "M", "S"];

const STORAGE_KEY = "woffl.wellSort";

interface Persisted {
  popsPads: string[];
  forceTrue: string[];
  padLimits: Record<string, number>;
}

function restore(): Persisted {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const p = JSON.parse(raw) as Partial<Persisted>;
      return {
        popsPads: Array.isArray(p.popsPads) ? p.popsPads : [...DEFAULT_POPS_PADS],
        forceTrue: Array.isArray(p.forceTrue) ? p.forceTrue : [],
        padLimits: p.padLimits && typeof p.padLimits === "object" ? p.padLimits : {},
      };
    }
  } catch {
    // storage unavailable - defaults still work in-memory
  }
  return { popsPads: [...DEFAULT_POPS_PADS], forceTrue: [], padLimits: {} };
}

function persist(state: WellSortState): void {
  try {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        popsPads: state.popsPads,
        forceTrue: state.forceTrue,
        padLimits: state.padLimits,
      }),
    );
  } catch {
    // ignore - private mode
  }
}

interface WellSortState {
  /** Pads with on-pad production separation (PopsPad=True for their wells). */
  popsPads: string[];
  /** Per-well PopsPad=True overrides, applied after the pad-level flags. */
  forceTrue: string[];
  /** Edited per-pad pump limits (BWPD); missing key = use the preset. */
  padLimits: Record<string, number>;

  setPopsPads: (pads: string[]) => void;
  setForceTrue: (wells: string[]) => void;
  setPadLimit: (pad: string, limit: number) => void;
  resetPadLimit: (pad: string) => void;
}

const initial = restore();

export const useWellSortStore = create<WellSortState>((set, get) => ({
  ...initial,

  setPopsPads: (pads) => {
    set({ popsPads: [...pads].sort() });
    persist(get());
  },
  setForceTrue: (wells) => {
    set({ forceTrue: [...wells].sort() });
    persist(get());
  },
  setPadLimit: (pad, limit) => {
    set((s) => ({ padLimits: { ...s.padLimits, [pad]: limit } }));
    persist(get());
  },
  resetPadLimit: (pad) => {
    set((s) => {
      const next = { ...s.padLimits };
      delete next[pad];
      return { padLimits: next };
    });
    persist(get());
  },
}));
