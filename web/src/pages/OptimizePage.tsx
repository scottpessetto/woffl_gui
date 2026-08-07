/**
 * Optimization pad board - the redesigned optimization entry point.
 *
 * Engineers match + save well fits on the Single Well solver; this page
 * reports per-pad READINESS: every well on the pad, whether a fit is saved
 * (IPR curve + calibrated friction), when and by whom, and what is
 * missing. Wells can be checked OFFLINE (excluded from a future run), and
 * planned FUTURE wells can be appended, each matching an existing well's
 * saved fit - the donor may live on any pad. Offline flags and future
 * wells persist in localStorage (one engineer's working config); fit
 * status always comes live from prop_hist.
 */

import clsx from "clsx";
import { Check, Plus, Trash2 } from "lucide-react";
import { useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";

import { usePadFitStatus, useWells } from "../api/hooks";
import type { PadFitWell } from "../api/types";
import { Card, ErrorNote, Spinner } from "../components/ui";
import { useOptimizeStore } from "../state/optimize";
import { useParamsStore } from "../state/params";

import { RunPanel } from "./optimize/RunPanel";

const INPUT_CLS =
  "h-8 rounded-md border border-slate-300 bg-white px-2 text-sm text-slate-800 " +
  "outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-200";

/** Missing-parts chips, or the green Ready check. */
function FitStatus({ row }: { row: PadFitWell }) {
  const missing: string[] = [];
  if (!row.has_curve) missing.push("IPR");
  if (!row.has_friction) missing.push("friction");
  if (missing.length === 0) {
    return (
      <span className="inline-flex items-center gap-1 text-emerald-700">
        <Check className="h-3.5 w-3.5" /> Ready
      </span>
    );
  }
  return (
    <span className="inline-flex flex-wrap gap-1">
      {missing.map((m) => (
        <span key={m} className="rounded bg-amber-50 px-1.5 py-px text-[11px] font-medium text-amber-700">
          no {m}
        </span>
      ))}
    </span>
  );
}

/** One pad's readiness board: every well's saved-fit status, offline
 * toggles, and planned future wells. Rendered by the Pad review tab (with
 * the pad selector) AND under each pad run tab, scoped to that pad. */
function PadReadiness({ pad }: { pad: string }) {
  const wells = useWells();
  const offlineByPad = useOptimizeStore((s) => s.offline);
  const futureByPad = useOptimizeStore((s) => s.future);
  const toggleOffline = useOptimizeStore((s) => s.toggleOffline);
  const addFuture = useOptimizeStore((s) => s.addFuture);
  const removeFuture = useOptimizeStore((s) => s.removeFuture);

  // Well name -> Single Well solver, same selection flow as the sidebar
  // picker (context fetch seeds the params on arrival).
  const navigate = useNavigate();
  const selectWell = useParamsStore((s) => s.selectWell);
  const openWell = (name: string) => {
    selectWell(name);
    navigate("/solver");
  };

  const [newName, setNewName] = useState("");
  const [newMatch, setNewMatch] = useState("");

  const future = futureByPad[pad] ?? [];
  const offline = offlineByPad[pad] ?? [];
  const donors = useMemo(() => future.map((f) => f.match), [future]);

  const statusQ = usePadFitStatus(pad, donors);
  const byWell = useMemo(() => {
    const m = new Map<string, PadFitWell>();
    for (const w of statusQ.data?.wells ?? []) m.set(w.well, w);
    for (const w of statusQ.data?.extras ?? []) m.set(w.well, w);
    return m;
  }, [statusQ.data]);

  const wellNames = useMemo(() => (wells.data?.wells ?? []).map((w) => w.name), [wells.data]);

  const addRow = () => {
    const name = newName.trim();
    if (!name || !wellNames.includes(newMatch)) return;
    addFuture(pad, { name, match: newMatch });
    setNewName("");
    setNewMatch("");
  };

  return (
    <>
      {statusQ.isError && <ErrorNote error={statusQ.error} />}
      {statusQ.isLoading && <Spinner label={`Loading ${pad}-Pad fits`} />}

      {statusQ.data && (
        <Card padded={false} className="overflow-hidden">
          <table className="w-full border-collapse text-[13px]">
            <thead>
              <tr className="border-b border-slate-200 bg-slate-50 text-left text-slate-600">
                <th className="px-3 py-2 font-semibold">Well</th>
                <th className="px-3 py-2 font-semibold">IPR saved</th>
                <th className="px-3 py-2 font-semibold">By</th>
                <th className="px-3 py-2 font-semibold">Friction</th>
                <th className="px-3 py-2 font-semibold">Fit status</th>
                <th className="px-3 py-2 text-center font-semibold">Offline</th>
                <th className="w-8 px-2 py-2"></th>
              </tr>
            </thead>
            <tbody>
              {statusQ.data.wells.map((row) => {
                const isOffline = offline.includes(row.well);
                return (
                  <tr
                    key={row.well}
                    className={clsx(
                      "border-b border-slate-100 last:border-b-0",
                      isOffline && "opacity-50",
                    )}
                  >
                    <td className="px-3 py-1.5">
                      <button
                        type="button"
                        onClick={() => openWell(row.well)}
                        title="Open this well on the Single Well solver"
                        className="font-medium text-slate-700 underline decoration-slate-300 decoration-dotted underline-offset-2 hover:text-blue-700 hover:decoration-blue-400"
                      >
                        {row.well}
                      </button>
                    </td>
                    <td className="px-3 py-1.5 tabular-nums text-slate-600">{row.saved_at?.slice(0, 10) ?? "-"}</td>
                    <td className="max-w-44 truncate px-3 py-1.5 text-slate-500" title={row.saved_by ?? undefined}>
                      {row.saved_by ? row.saved_by.split("@")[0] : "-"}
                    </td>
                    <td className="px-3 py-1.5 text-slate-600">
                      {row.has_friction ? row.friction_keys.join(" ") : "-"}
                    </td>
                    <td className="px-3 py-1.5">
                      <FitStatus row={row} />
                    </td>
                    <td className="px-3 py-1.5 text-center">
                      <input
                        type="checkbox"
                        checked={isOffline}
                        onChange={() => toggleOffline(pad, row.well)}
                        title="Exclude this well from the optimization run"
                        className="h-4 w-4 rounded border-slate-300 accent-blue-600"
                      />
                    </td>
                    <td></td>
                  </tr>
                );
              })}

              {future.map((f) => {
                const donor = byWell.get(f.match);
                return (
                  <tr key={`future-${f.name}`} className="border-b border-slate-100 bg-indigo-50/40 last:border-b-0">
                    <td className="px-3 py-1.5 font-medium text-slate-700">
                      {f.name}
                      <span className="ml-2 rounded bg-indigo-100 px-1.5 py-px text-[10px] font-medium text-indigo-700">
                        future - matches {f.match}
                      </span>
                    </td>
                    <td className="px-3 py-1.5 tabular-nums text-slate-600">
                      {donor?.saved_at?.slice(0, 10) ?? "-"}
                    </td>
                    <td className="max-w-44 truncate px-3 py-1.5 text-slate-500">
                      {donor?.saved_by ? donor.saved_by.split("@")[0] : "-"}
                    </td>
                    <td className="px-3 py-1.5 text-slate-600">
                      {donor?.has_friction ? donor.friction_keys.join(" ") : "-"}
                    </td>
                    <td className="px-3 py-1.5">{donor ? <FitStatus row={donor} /> : "-"}</td>
                    <td className="px-3 py-1.5 text-center text-[11px] text-slate-400">planned</td>
                    <td className="px-2 py-1.5">
                      <button
                        type="button"
                        aria-label={`Remove ${f.name}`}
                        className="text-slate-400 hover:text-red-600"
                        onClick={() => removeFuture(pad, f.name)}
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </td>
                  </tr>
                );
              })}

              <tr className="bg-slate-50/60">
                <td className="px-3 py-2" colSpan={4}>
                  <div className="flex flex-wrap items-center gap-2">
                    <input
                      value={newName}
                      onChange={(e) => setNewName(e.target.value)}
                      placeholder="Future well name"
                      className={clsx(INPUT_CLS, "w-40")}
                    />
                    <span className="text-xs text-slate-500">matches</span>
                    <input
                      value={newMatch}
                      onChange={(e) => setNewMatch(e.target.value)}
                      list={`optimize-donor-wells-${pad}`}
                      placeholder="existing well (any pad)"
                      className={clsx(INPUT_CLS, "w-48")}
                    />
                    <datalist id={`optimize-donor-wells-${pad}`}>
                      {wellNames.map((w) => (
                        <option key={w} value={w} />
                      ))}
                    </datalist>
                  </div>
                </td>
                <td className="px-3 py-2" colSpan={3}>
                  <button
                    type="button"
                    disabled={!newName.trim() || !wellNames.includes(newMatch)}
                    onClick={addRow}
                    title={
                      !newName.trim()
                        ? "Name the planned well first"
                        : !wellNames.includes(newMatch)
                          ? "Pick the existing well whose saved fit it should use"
                          : `Add ${newName.trim()} to ${pad}-Pad, using ${newMatch}'s saved fit`
                    }
                    className="flex items-center gap-1 rounded-md border border-slate-300 bg-white px-2 py-1 text-xs font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-40"
                  >
                    <Plus className="h-3.5 w-3.5" /> Add future well
                  </button>
                </td>
              </tr>
            </tbody>
          </table>
        </Card>
      )}
    </>
  );
}

export default function OptimizePage() {
  const wells = useWells();
  const pad = useOptimizeStore((s) => s.pad);
  const setPad = useOptimizeStore((s) => s.setPad);

  const [view, setView] = useState<"board" | "S" | "I" | "M" | "CFP">("board");

  const pads = useMemo(() => {
    const uniq = new Set((wells.data?.wells ?? []).map((w) => w.pad).filter(Boolean));
    return [...uniq].sort();
  }, [wells.data]);

  const activePad = pad && pads.includes(pad) ? pad : (pads[0] ?? null);

  const VIEW_TABS: { key: "board" | "S" | "I" | "M" | "CFP"; label: string }[] = [
    { key: "board", label: "Pad review" },
    { key: "S", label: "S-Pad" },
    { key: "I", label: "I-Pad" },
    { key: "M", label: "M-Pad" },
    { key: "CFP", label: "CFP run" },
  ];

  return (
    <div className="mx-auto max-w-6xl space-y-4 p-4">
      <div>
        <h1 className="text-xl font-semibold tracking-tight text-slate-800">Optimization</h1>
        <p className="text-sm text-slate-500">
          Match and save each well's fit on the Single Well solver; review readiness here,
          then run the pad and CFP optimizations against those saved fits.
        </p>
      </div>

      <div className="flex gap-1 rounded-lg bg-white p-1 shadow-sm ring-1 ring-slate-200 w-fit">
        {VIEW_TABS.map((t) => (
          <button
            key={t.key}
            type="button"
            onClick={() => setView(t.key)}
            className={
              "rounded-md px-3 py-1 text-sm font-medium transition-colors " +
              (view === t.key ? "bg-blue-600 text-white" : "text-slate-600 hover:bg-slate-50")
            }
          >
            {t.label}
          </button>
        ))}
      </div>

      {view !== "board" && (
        <RunPanel kind={view === "CFP" ? "cfp" : "pad"} pad={view === "CFP" ? null : view} />
      )}

      {/* Pad run tabs carry their own readiness board, scoped to the pad -
          the same table the Pad review tab shows, minus the pad selector. */}
      {(view === "S" || view === "I" || view === "M") && (
        <div className="space-y-2">
          <h2 className="text-sm font-semibold tracking-tight text-slate-700">
            {view}-Pad readiness
          </h2>
          <PadReadiness pad={view} />
        </div>
      )}

      {view === "board" && (
        <>
          {pads.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {pads.map((p) => (
                <button
                  key={p}
                  type="button"
                  onClick={() => setPad(p)}
                  className={clsx(
                    "rounded-md px-2.5 py-1 text-sm font-medium transition-colors",
                    p === activePad
                      ? "bg-blue-600 text-white"
                      : "bg-white text-slate-600 ring-1 ring-slate-200 hover:bg-slate-50",
                  )}
                >
                  {p}
                </button>
              ))}
            </div>
          )}
          {activePad && <PadReadiness pad={activePad} />}
        </>
      )}
    </div>
  );
}
