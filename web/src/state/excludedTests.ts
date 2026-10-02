/**
 * Well tests the engineer has excluded as bad data, per well.
 *
 * An excluded test leaves the anchor and comparison dropdowns, the IPR fit
 * (the server drops it by wt_uid, so "Most recent" and the medians never
 * pick it) and the chart. The tests table still lists it, greyed, so it can
 * be brought back. Persisted in localStorage: one engineer's data-quality
 * call, not shared field truth (there is no database home for it yet).
 */

import { create } from "zustand";

const STORAGE_KEY = "woffl.excludedTests";

type ByWell = Record<string, string[]>; // well -> test keys (selection.testKey)

function restore(): ByWell {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    const parsed = raw ? (JSON.parse(raw) as unknown) : null;
    return parsed && typeof parsed === "object" ? (parsed as ByWell) : {};
  } catch {
    return {};
  }
}

interface ExcludedTestsState {
  byWell: ByWell;
  setExcluded: (well: string, key: string, excluded: boolean) => void;
}

export const useExcludedTests = create<ExcludedTestsState>((set) => ({
  byWell: restore(),
  setExcluded: (well, key, excluded) =>
    set((s) => {
      const current = new Set(s.byWell[well] ?? []);
      if (excluded) current.add(key);
      else current.delete(key);
      const byWell = { ...s.byWell, [well]: [...current].sort() };
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(byWell));
      } catch {
        // storage unavailable: the exclusion still applies for this session
      }
      return { byWell };
    }),
}));

// --- info-only tests in the IPR fit -----------------------------------------
//
// Off by default: info-only tests are unscreened SCADA tests, and the
// optimizer seeds a well from its allocated tests, so the Solver opens on the
// same curve. Switching it on for a well is the same kind of call as an
// exclusion - one engineer's judgement about that well's data - and persists
// the same way.

const INFO_FIT_KEY = "woffl.infoOnlyInFit";

function restoreInfoFit(): Record<string, boolean> {
  try {
    const raw = localStorage.getItem(INFO_FIT_KEY);
    const parsed = raw ? (JSON.parse(raw) as unknown) : null;
    return parsed && typeof parsed === "object" ? (parsed as Record<string, boolean>) : {};
  } catch {
    return {};
  }
}

interface InfoFitState {
  byWell: Record<string, boolean>;
  setInfoInFit: (well: string, on: boolean) => void;
}

export const useInfoOnlyInFit = create<InfoFitState>((set) => ({
  byWell: restoreInfoFit(),
  setInfoInFit: (well, on) =>
    set((s) => {
      const byWell = { ...s.byWell };
      if (on) byWell[well] = true;
      else delete byWell[well];
      try {
        localStorage.setItem(INFO_FIT_KEY, JSON.stringify(byWell));
      } catch {
        // storage unavailable: the choice still applies for this session
      }
      return { byWell };
    }),
}));

const EMPTY: string[] = [];

/** The excluded test keys for one well (stable empty array when none). */
export function useExcludedKeys(well: string): string[] {
  return useExcludedTests((s) => s.byWell[well] ?? EMPTY);
}
