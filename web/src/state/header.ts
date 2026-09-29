/**
 * Header page working state: the run form, per-well choices (online,
 * relation, IPR) and the last board/run job ids.
 *
 * Persisted per browser like the Optimize board - one engineer's scratch
 * configuration. Saved relations and IPRs always come from prop_hist through
 * the board job.
 */

import { create } from "zustand";

import type { HeaderWellChoice } from "../api/types";
import { DEFAULT_HEADER_FORM, type HeaderForm } from "../pages/header/model";

const STORAGE_KEY = "woffl.header";

interface Persisted {
  form: HeaderForm;
  choices: Record<string, HeaderWellChoice>;
  boardJob: string | null;
  runJob: string | null;
  /** Stable key of the request behind runJob, to flag a stale result. */
  runKey: string | null;
  /** Pads loaded on the Wells tab (independent of the Impact tab's pads). */
  wellsPads: string[];
}

function restore(): Persisted {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const p = JSON.parse(raw) as Partial<Persisted>;
      return {
        form: { ...DEFAULT_HEADER_FORM, ...(p.form && typeof p.form === "object" ? p.form : {}) },
        choices: p.choices && typeof p.choices === "object" ? p.choices : {},
        boardJob: typeof p.boardJob === "string" ? p.boardJob : null,
        runJob: typeof p.runJob === "string" ? p.runJob : null,
        runKey: typeof p.runKey === "string" ? p.runKey : null,
        wellsPads: Array.isArray(p.wellsPads) ? p.wellsPads.filter((x) => typeof x === "string") : [],
      };
    }
  } catch {
    // storage unavailable - defaults still work in-memory
  }
  return { form: DEFAULT_HEADER_FORM, choices: {}, boardJob: null, runJob: null, runKey: null, wellsPads: [] };
}

function persist(s: Persisted): void {
  try {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ form: s.form, choices: s.choices, boardJob: s.boardJob, runJob: s.runJob, runKey: s.runKey, wellsPads: s.wellsPads }),
    );
  } catch {
    // ignore - private mode
  }
}

interface HeaderState extends Persisted {
  setForm: (patch: Partial<HeaderForm>) => void;
  setChoice: (well: string, patch: Partial<HeaderWellChoice>) => void;
  resetChoices: () => void;
  setBoardJob: (id: string | null) => void;
  setRunJob: (id: string | null, key?: string | null) => void;
  setWellsPads: (pads: string[]) => void;
}

export const useHeaderStore = create<HeaderState>((set, get) => ({
  ...restore(),
  setForm: (patch) => {
    set({ form: { ...get().form, ...patch } });
    persist(get());
  },
  setChoice: (well, patch) => {
    const prev = get().choices[well] ?? { well };
    set({ choices: { ...get().choices, [well]: { ...prev, ...patch, well } } });
    persist(get());
  },
  resetChoices: () => {
    set({ choices: {} });
    persist(get());
  },
  setBoardJob: (id) => {
    set({ boardJob: id });
    persist(get());
  },
  setRunJob: (id, key = null) => {
    set({ runJob: id, runKey: id ? key : null });
    persist(get());
  },
  setWellsPads: (pads) => {
    set({ wellsPads: [...new Set(pads)].sort() });
    persist(get());
  },
}));
