/**
 * Submitted sweep snapshots, per well - session-scoped (in-memory, no
 * localStorage). The expensive sweeps (Batch Run, ...) run on explicit
 * submit against a SNAPSHOT of the params; keeping the snapshot here
 * instead of component state means navigating away and back re-attaches to
 * the last run (the query cache still holds its result) instead of
 * presenting an empty page. A reload starts clean, consistent with the
 * app's fresh-start behavior.
 */

import { create } from "zustand";

import type { SimParams } from "../api/types";

interface SweepsState {
  /** well -> the params snapshot the last batch sweep was submitted with. */
  batch: Record<string, SimParams>;
  setBatchSnapshot: (well: string, snapshot: SimParams) => void;
}

export const useSweepsStore = create<SweepsState>((set) => ({
  batch: {},
  setBatchSnapshot: (well, snapshot) =>
    set((s) => ({ batch: { ...s.batch, [well]: snapshot } })),
}));
