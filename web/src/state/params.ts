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

/** Shared empty ownership set - a fresh bench has no hand-set inputs. */
const NO_MANUAL: ReadonlySet<keyof SimParams> = new Set();

/** Ownership set plus `keys`, or the same set when nothing is new (so a
 *  no-op edit does not re-render every field that reads it). */
function withManual(
  current: ReadonlySet<keyof SimParams>,
  keys: Array<keyof SimParams>,
): ReadonlySet<keyof SimParams> {
  if (keys.every((key) => current.has(key))) return current;
  const next = new Set(current);
  for (const key of keys) next.add(key);
  return next;
}

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
  /** Well the open-time IPR fit has already been laid over. Per WELL, not
   *  per SolverPage mount: the latch used to live in component state, so
   *  every return to the Solver re-applied the fit and silently discarded
   *  manual sidebar edits to WC / GOR / qwf / pwf - including a permutation
   *  applied from Match Sensitivities. */
  fitAppliedFor: string | null;
  markFitApplied: (well: string) => void;
  /** Inputs the ENGINEER set on this well - by typing in the sidebar, by
   *  applying a sensitivity permutation, or by Auto-match BHP. The open-time
   *  fit and the "Apply IPR to inputs" button both skip these, the same way
   *  they skip locked WC/GOR/ResP. Without this the fit landed on top of a
   *  hand-matched well and the engineer saved the fit back over their own
   *  numbers (observed on MPC-45, 2026-08-08). Cleared when the well or its
   *  context changes, and released deliberately by "Apply IPR to inputs". */
  manualFields: ReadonlySet<keyof SimParams>;
  /** Where the current hand-set state came from, when it came from a study:
   *  "permutation 2 of 27, score 0.0571". Rides into the save comment so the
   *  next engineer sees WHY these numbers, not just the numbers. */
  matchNote: string | null;
  setMatchNote: (note: string | null) => void;

  set: <K extends keyof SimParams>(key: K, value: SimParams[K]) => void;
  setMany: (partial: Partial<SimParams>) => void;
  setWindow: (months: number, cap: number) => void;
  selectWell: (name: string) => void;
  applyContext: (ctx: WellContext) => void;
  /** Mirror a server-confirmed lock toggle (usePropLock response). */
  setPropLock: (field: keyof PropLocks, lock: PropLock) => void;
  /** Lay IPR fit seeds over the params, skipping LOCKED fields - the
   * Streamlit contract: an anchor re-seed touches everything EXCEPT the
   * locked WC/GOR/ResP (jetpump_solver.py's locked_fields filter) - and
   * skipping fields the engineer set by hand. `release` is the explicit
   * "take the fit anyway" of the Apply IPR button: it drops ownership of the
   * seeded fields first, so the fit wins and keeps winning until the next
   * hand edit. */
  applyIprSeeds: (seeds: Partial<SimParams>, release?: boolean) => void;
  run: () => void;
}

export const useParamsStore = create<ParamsState>((set) => ({
  // Every fresh load starts at the Welcome screen (well = Custom): opening
  // on a remembered well invites working yesterday's problem by accident.
  // Deliberately NO localStorage restore (removed 2026-08-06, Scott).
  well: "Custom",
  params: { ...DEFAULT_PARAMS },
  simActive: false,
  months: 6,
  cap: 0,
  asBuiltLocks: NO_AS_BUILT,
  propLocks: NO_PROP_LOCKS,
  context: null,
  seededFor: null,
  fitAppliedFor: null,
  markFitApplied: (name) => set({ fitAppliedFor: name }),
  manualFields: NO_MANUAL,
  matchNote: null,
  setMatchNote: (note) => set({ matchNote: note }),

  set: (key, value) =>
    set((s) => ({
      params: { ...s.params, [key]: clampToBounds(key, value) },
      manualFields: withManual(s.manualFields, [key]),
    })),

  setMany: (partial) =>
    set((s) => ({
      params: mergeClamped(s.params, partial),
      manualFields: withManual(s.manualFields, Object.keys(partial) as Array<keyof SimParams>),
    })),

  // Window changes refetch the tests and re-run the fit; they are NOT a
  // reason to rebuild the bench. Nulling seededFor used to re-open the
  // Layout seeding gate, and applyContext then rebuilt params from
  // DEFAULT_PARAMS - so widening the lookback silently discarded every
  // hand-set input, including an applied permutation. Only the fit latch
  // resets: the new window can legitimately produce a new fit, and that fit
  // still skips locked and hand-set fields.
  setWindow: (months, cap) => set({ months, cap, fitAppliedFor: null }),

  selectWell: (name) =>
    set(() => ({
      well: name,
      // Custom = clean bench; a named well seeds on context arrival.
      params: { ...DEFAULT_PARAMS },
      simActive: name !== "Custom",
      asBuiltLocks: NO_AS_BUILT,
      propLocks: NO_PROP_LOCKS,
      context: null,
      seededFor: null,
      fitAppliedFor: null,
      manualFields: NO_MANUAL,
      matchNote: null,
    })),

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
        // The seeds ARE the well's state as the server assembled it; nothing
        // on the bench is hand-set any more.
        manualFields: NO_MANUAL,
        matchNote: null,
      };
    }),

  setPropLock: (field, lock) => set((s) => ({ propLocks: { ...s.propLocks, [field]: lock } })),

  applyIprSeeds: (seeds, release = false) =>
    set((s) => {
      const filtered: Partial<SimParams> = { ...seeds };
      if (s.propLocks.form_wc.locked) delete filtered.form_wc;
      if (s.propLocks.form_gor.locked) delete filtered.form_gor;
      if (s.propLocks.res_pres.locked) delete filtered.pres;
      const keys = Object.keys(filtered) as Array<keyof SimParams>;
      if (!release) {
        for (const key of keys) {
          if (s.manualFields.has(key)) delete filtered[key];
        }
        return { params: mergeClamped(s.params, filtered) };
      }
      // Explicit "take the fit": ownership of exactly the seeded fields is
      // handed back, and a study's provenance no longer describes the bench.
      const manual = new Set(s.manualFields);
      for (const key of keys) manual.delete(key);
      return {
        params: mergeClamped(s.params, filtered),
        manualFields: manual,
        matchNote: null,
      };
    }),

  run: () => set({ simActive: true }),
}));

/** Effective params sent to the solver: dewatering forces form_wc = 1. */
export function effectiveParams(params: SimParams): SimParams {
  return params.model_as_water ? { ...params, form_wc: 1.0 } : params;
}
