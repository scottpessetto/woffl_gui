/**
 * Well Sort - field-wide online/offline triage workspace. Port of
 * woffl/gui/well_sort_page.py (Wells | Triage | Marginal WC) on the web
 * stack: one shared POPs configuration (zustand, localStorage) drives all
 * three views, replacing the Streamlit session keys.
 */

import { Settings2 } from "lucide-react";
import { useState } from "react";

import { useWellSortTables } from "../api/hooks";
import { useWellSortStore } from "../state/wellSort";
import { ChipToggles, WellPicker } from "./well-sort/shared";
import { MarginalView } from "./well-sort/MarginalView";
import { TriageView } from "./well-sort/TriageView";
import { WellsView } from "./well-sort/WellsView";

type View = "wells" | "triage" | "marginal";

const VIEWS: Array<{ id: View; label: string; caption: string }> = [
  {
    id: "wells",
    label: "Wells",
    caption: "Online / offline / LTSI status for every MPU producer, plus 30-day shut-in events.",
  },
  {
    id: "triage",
    label: "Triage",
    caption: "Keep / SI / BOL decisions - each well's water cut vs the field marginal WC.",
  },
  {
    id: "marginal",
    label: "Marginal WC",
    caption: "Cumulative-water-threshold marginal water cut, field-wide and per POPs pad.",
  },
];

export default function WellSortPage() {
  const [view, setView] = useState<View>("wells");
  const [configOpen, setConfigOpen] = useState(false);

  const popsPads = useWellSortStore((s) => s.popsPads);
  const forceTrue = useWellSortStore((s) => s.forceTrue);
  const setPopsPads = useWellSortStore((s) => s.setPopsPads);
  const setForceTrue = useWellSortStore((s) => s.setForceTrue);

  // Any tables fetch echoes the live pad/producer lists for the config UI.
  const tables = useWellSortTables("allocated", 60, popsPads, forceTrue);
  const allPads = tables.data?.all_pads ?? [];
  const producers = tables.data?.producers ?? [];

  const active = VIEWS.find((v) => v.id === view) ?? VIEWS[0];

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-lg font-semibold text-slate-800">Well Sort</h1>
          <p className="text-xs text-slate-500">{active.caption}</p>
        </div>
        <button
          type="button"
          onClick={() => setConfigOpen((v) => !v)}
          className="inline-flex items-center gap-1.5 rounded-md border border-slate-300 bg-white px-2.5 py-1.5 text-xs font-medium text-slate-700 hover:border-slate-400"
          title="Pads with on-pad production separation + per-well overrides (shared by all three views)"
        >
          <Settings2 className="h-3.5 w-3.5" />
          POPs config
          <span className="text-slate-400">
            {popsPads.length} pads{forceTrue.length > 0 ? `, ${forceTrue.length} overrides` : ""}
          </span>
        </button>
      </div>

      {configOpen && (
        <div className="rounded-lg border border-slate-200 bg-white p-3">
          <div className="grid gap-4 md:grid-cols-2">
            <div>
              <p className="mb-1 text-xs font-semibold text-slate-600">
                Pads with on-pad production separation
              </p>
              <p className="mb-2 text-xs text-slate-400">
                Wells on these pads get PopsPad=True and are excluded from the field marginal WC.
              </p>
              <ChipToggles
                options={allPads.length > 0 ? allPads : popsPads}
                selected={popsPads}
                onChange={setPopsPads}
              />
            </div>
            <div>
              <p className="mb-1 text-xs font-semibold text-slate-600">
                Per-well PopsPad=True overrides
              </p>
              <p className="mb-2 text-xs text-slate-400">
                Treated as having on-pad separation regardless of the pad-level setting.
              </p>
              <WellPicker options={producers} selected={forceTrue} onChange={setForceTrue} />
            </div>
          </div>
        </div>
      )}

      <div className="flex flex-wrap gap-1 rounded-lg border border-slate-200 bg-white p-1 w-fit shadow-[0_1px_2px_rgba(15,23,42,0.05)]">
        {VIEWS.map((v) => (
          <button
            key={v.id}
            type="button"
            onClick={() => setView(v.id)}
            className={
              view === v.id
                ? "rounded-md bg-blue-600 px-3 py-1 text-sm font-medium text-white"
                : "rounded-md px-3 py-1 text-sm text-slate-600 hover:bg-slate-100"
            }
          >
            {v.label}
          </button>
        ))}
      </div>

      {view === "wells" && <WellsView />}
      {view === "triage" && <TriageView />}
      {view === "marginal" && <MarginalView />}
    </div>
  );
}
