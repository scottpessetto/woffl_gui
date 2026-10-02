/**
 * Wells tab: review and save how every well is represented, pad by pad.
 *
 * Picking pads starts the wells board in the background (no button); the
 * board is cached server-side for 15 minutes, so switching back and forth is
 * quick. Choices are shared with the Impact tab, so an edit here is what the
 * next Estimate uses. Save writes the chosen relation / IPR to prop_hist.
 */

import clsx from "clsx";
import { useEffect, useRef, useState } from "react";

import { isMissingJob } from "../../api/client";
import { useCancelHeaderJob, useHeaderJob, useHeaderPads, useSaveHeader, useStartHeaderBoard } from "../../api/hooks";
import type { HeaderBoard, HeaderSaveResponse, HeaderSaveWell } from "../../api/types";
import { ChartPanel } from "../../charts/ChartPanel";
import { Button, Card, ErrorNote, InfoNote, Section } from "../../components/ui";
import { fmtNum } from "../../lib/format";
import { useHeaderStore } from "../../state/header";
import { BoardTable } from "./BoardTable";
import { correlationChart } from "./charts";
import { JobError, JobLine } from "./JobStatus";
import { savePlan } from "./model";
import { WellCards } from "./WellCards";

const FIELD = "h-8 rounded-md border border-slate-300 bg-white px-2 text-sm tabular-nums";

function IprGroupsTable({ board }: { board: HeaderBoard }) {
  const groups = Object.entries(board.ipr_groups);
  if (!groups.length) return null;
  const users = (key: string) => board.rows.filter((r) => r.ipr_group === key && r.lift !== "JP").length;
  return (
    <Card>
      <div className="mb-1 text-sm font-medium text-slate-700">Reservoir pressure and BHP for wells without a gauge</div>
      <p className="mb-2 max-w-4xl text-xs text-slate-500">
        Every well runs on its own reservoir pressure: the value saved in prop_hist, else the documented default (range
        +/-20%). A well without a working gauge takes its BHP from the typical BHP/ResP of gauged, flowing wells in the same
        pad and reservoir. Shut-in gauges are shown as evidence for anyone saving a ResP; they are not used.
      </p>
      <div className="overflow-auto">
        <table className="w-full text-[12.5px] tabular-nums">
          <thead className="text-left text-slate-500">
            <tr>
              {["Group", "Default ResP", "ResP saved", "BHP / ResP (median [IQR])", "From gauged wells", "Default for", "Shut-in gauges (ResP at least)", "Note"].map((h) => (
                <th key={h} className="px-2 py-1 font-medium whitespace-nowrap">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {groups.map(([key, g]) => (
              <tr key={key} className="border-t border-slate-100 align-top">
                <td className="px-2 py-1 font-medium whitespace-nowrap">{key}</td>
                <td className="px-2 py-1">{fmtNum(g.default_pres, 0)}</td>
                <td className="px-2 py-1">{g.n_saved_pres} of {g.n_members} wells</td>
                <td className="px-2 py-1 whitespace-nowrap">{fmtNum(g.ratio.med, 2)} [{fmtNum(g.ratio.q25, 2)}-{fmtNum(g.ratio.q75, 2)}]</td>
                <td className="px-2 py-1 text-xs">{g.wells.length ? `${g.wells.length}: ${g.wells.join(", ")}` : "-"}</td>
                <td className="px-2 py-1">{users(key)} wells</td>
                <td className="px-2 py-1 text-xs">{(g.shut_in ?? []).map((x) => `${x.well} ${fmtNum(x.bhp, 0)}`).join(", ") || "-"}</td>
                <td className="px-2 py-1 text-xs text-amber-800">{g.note ?? ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function Correlations({ board }: { board: HeaderBoard }) {
  return (
    <Section title="Correlations wells can borrow">
      <div className="space-y-4">
        <IprGroupsTable board={board} />
        {Object.keys(board.correlations).length > 0 && (
          <>
            <p className="max-w-4xl text-xs text-slate-500">
              BHP~WHP: measured closed-loop slopes by lift type and reservoir. At fixed speed an ESP's slope is about
              1/(1 + PI x pump-curve steepness), so high-rate wells barely move BHP yet lose the most liquid. Wells without a
              usable gauge take their group's line at their test rate (hollow markers). The JP group is used only by jet pumps
              set to the BHP~WHP relation. Groups are built from the wells on the pads loaded here.
            </p>
            <div className="grid gap-4 lg:grid-cols-2">
              {Object.entries(board.correlations)
                .sort(([, a], [, b]) => a.lift.localeCompare(b.lift) || Number(a.reservoir === null) - Number(b.reservoir === null))
                .map(([key, corr]) => {
                  const opt = correlationChart(key, corr, board.rows);
                  const c = corr.correlation;
                  const users = board.rows.filter((r) => r.corr_group === key).length;
                  return (
                    <Card key={key}>
                      <div className="mb-1 text-sm font-medium text-slate-700">
                        {corr.lift} - {corr.reservoir ?? "all reservoirs"}:{" "}
                        {c ? `${c.n} measured wells, ${c.kind === "trend" ? "trend with rate" : "group median"}, scatter +/-${fmtNum(c.resid_mad, 2)}` : "no measured wells"}
                        <span className="ml-1 text-xs font-normal text-slate-400">
                          {users ? `(default group for ${users} well${users > 1 ? "s" : ""})` : "(fallback only)"}
                        </span>
                      </div>
                      {opt ? <ChartPanel option={opt} height={300} zoom={{ xAxisIndex: [0], yAxisIndex: [0] }} /> :
                        <div className="py-6 text-center text-sm text-slate-400">No measured wells in this group.</div>}
                    </Card>
                  );
                })}
            </div>
          </>
        )}
      </div>
    </Section>
  );
}

export function WellsTab({ dFor, writesOn }: { dFor: (pad: string) => number; writesOn: boolean }) {
  const padsQ = useHeaderPads();
  const wellsPads = useHeaderStore((s) => s.wellsPads);
  const setWellsPads = useHeaderStore((s) => s.setWellsPads);
  const fitDays = useHeaderStore((s) => s.form.fitDays);
  const setForm = useHeaderStore((s) => s.setForm);
  const choices = useHeaderStore((s) => s.choices);
  const setChoice = useHeaderStore((s) => s.setChoice);
  const resetChoices = useHeaderStore((s) => s.resetChoices);
  const boardJob = useHeaderStore((s) => s.boardJob);
  const setBoardJob = useHeaderStore((s) => s.setBoardJob);

  const startBoard = useStartHeaderBoard();
  const cancel = useCancelHeaderJob();
  const save = useSaveHeader();
  const boardQ = useHeaderJob(boardJob);
  const [view, setView] = useState<"cards" | "table">("cards");
  const [saveSel, setSaveSel] = useState<Set<string>>(new Set());
  const [savingWell, setSavingWell] = useState<string | null>(null);
  const [saveResult, setSaveResult] = useState<HeaderSaveResponse | null>(null);

  const pads = padsQ.data?.pads ?? [];
  const want = [...wellsPads].sort();
  const wantKey = `${want.join(",")}|${fitDays}`;
  // Keep showing the last finished board (and save against ITS job) while a
  // reload runs - after a save, or when pads change - so the cards never
  // blank out mid-review.
  const [shown, setShown] = useState<{ board: HeaderBoard; jobId: string } | null>(null);
  useEffect(() => {
    if (boardQ.data?.status === "done" && boardJob && boardQ.data.job_id === boardJob) {
      setShown({ board: boardQ.data.result as HeaderBoard, jobId: boardJob });
    }
  }, [boardQ.data, boardJob]);
  const board = shown?.board ?? null;
  const shownJob = shown?.jobId ?? null;
  const boardKey = board ? `${board.pads.join(",")}|${board.fit_days}` : null;
  const running = boardQ.data?.status === "running";
  // Which request the current job belongs to (survives re-renders, not reloads).
  const requested = useRef<string | null>(boardKey);

  useEffect(() => {
    if (isMissingJob(boardQ.error)) setBoardJob(null);
  }, [boardQ.error, setBoardJob]);

  // Background load: whenever the wanted pads/window differ from what is
  // loaded (or loading), start a board job - after a short pause, so clicking
  // through several pads starts one job, not five. A job for pads no longer
  // wanted is cancelled.
  useEffect(() => {
    if (!want.length || startBoard.isPending) return;
    if (boardKey === wantKey || (running && requested.current === wantKey)) return;
    const t = setTimeout(() => {
      if (running && boardJob) cancel.mutate(boardJob);
      requested.current = wantKey;
      setSaveSel(new Set());
      setSaveResult(null);
      startBoard
        .mutateAsync({ pads: want, fit_days: fitDays })
        .then(({ job_id }) => setBoardJob(job_id))
        .catch(() => { requested.current = null; });
    }, 600);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wantKey, boardKey, running, startBoard.isPending]);

  const reload = async () => {
    if (!want.length) return;
    requested.current = wantKey;
    const { job_id } = await startBoard.mutateAsync({ pads: want, fit_days: fitDays });
    setBoardJob(job_id);
  };

  const afterSave = async (res: HeaderSaveResponse) => {
    setSaveResult(res);
    // A saved well now defaults to what was saved: drop its overrides,
    // including the gauge verdict (saved too). Online stays a session choice.
    for (const r of res.results) {
      if (r.saved) {
        setChoice(r.well, { gauge_bad: null, relation: null, corr_group: null, slope: null, ipr: null, ipr_group: null, qwf: null, pwf: null, pres: null });
      }
    }
    if (res.saved_wells > 0 && board) {
      // Saved values become the board defaults: reload so the cards show them.
      requested.current = `${board.pads.join(",")}|${board.fit_days}`;
      const { job_id } = await startBoard.mutateAsync({ pads: board.pads, fit_days: board.fit_days });
      setBoardJob(job_id);
    }
  };
  const doSave = async () => {
    if (!board || !shownJob) return;
    const wells = board.rows
      .filter((r) => saveSel.has(r.well))
      .map((r) => savePlan(r, choices[r.well]).entry)
      .filter((e): e is NonNullable<typeof e> => e !== null);
    if (!wells.length) return;
    const res = await save.mutateAsync({ board_job_id: shownJob!, wells });
    setSaveSel(new Set());
    await afterSave(res);
  };
  const doSaveOne = async (entry: HeaderSaveWell) => {
    if (!board || !shownJob) return;
    setSavingWell(entry.well);
    try {
      await afterSave(await save.mutateAsync({ board_job_id: shownJob, wells: [entry] }));
    } finally {
      setSavingWell(null);
    }
  };
  const saveable = board ? board.rows.filter((r) => savePlan(r, choices[r.well]).entry !== null) : [];

  return (
    <div className="space-y-5">
      <Section title="Wells - how each well is represented">
        <p className="mb-3 max-w-4xl text-sm text-slate-600">
          Pick one or more pads; the wells load in the background. Check each well's gauge, BHP~WHP relation and IPR, change
          what is wrong and Save - saved values become every user's defaults and the Impact tab uses them.
        </p>
        <Card>
          <div className="flex flex-wrap items-end gap-4">
            <div>
              <div className="text-xs text-slate-500">Pads</div>
              <div className="mt-1 flex flex-wrap gap-1">
                {pads.map((p) => (
                  <button
                    key={p}
                    type="button"
                    onClick={() => setWellsPads(wellsPads.includes(p) ? wellsPads.filter((x) => x !== p) : [...wellsPads, p].sort())}
                    title="Click to add or remove a pad"
                    className={clsx(
                      "h-8 min-w-8 rounded-md border px-2 text-sm transition-colors",
                      wellsPads.includes(p) ? "border-blue-500 bg-blue-50 font-medium text-blue-700" : "border-slate-300 bg-white text-slate-600 hover:bg-slate-50",
                    )}
                  >
                    {p}
                  </button>
                ))}
                {wellsPads.length > 1 && (
                  <button type="button" className="h-8 rounded-md px-2 text-xs text-slate-500 hover:bg-slate-100" onClick={() => setWellsPads([])}>
                    clear
                  </button>
                )}
              </div>
            </div>
            <label className="text-xs text-slate-500">
              Fit window
              <select className={clsx(FIELD, "mt-1 block")} value={fitDays} onChange={(e) => setForm({ fitDays: Number(e.target.value) })}>
                {[60, 90, 120, 180, 365].map((d) => <option key={d} value={d}>{d} days</option>)}
              </select>
            </label>
            <div className="ml-auto flex flex-wrap items-center gap-2">
              {board && <span className="text-xs text-slate-500">Loaded {board.pads.join(", ")} at {board.built_at.slice(11, 16)}</span>}
              <Button size="sm" onClick={() => void reload()} disabled={!want.length || running} busy={startBoard.isPending}
                title="Reload now (for example after someone else saved)">
                Reload
              </Button>
            </div>
          </div>
          <div className="mt-2 space-y-2">
            <JobLine job={boardQ.data} jobId={boardJob} label="Loading wells" />
            <JobError job={boardQ.data} />
            {startBoard.error && <ErrorNote error={startBoard.error} />}
          </div>
        </Card>
      </Section>

      {!want.length && <div className="py-6 text-center text-sm text-slate-400">Pick a pad to load its wells.</div>}

      {board && (
        <Section
          title={`${board.rows.length} wells on ${board.pads.join(", ")}`}
          actions={
            <div className="flex flex-wrap items-center justify-end gap-2">
              <div className="inline-flex overflow-hidden rounded-md border border-slate-300 text-xs">
                {(["cards", "table"] as const).map((v) => (
                  <button key={v} type="button" onClick={() => setView(v)}
                    className={clsx("px-3 py-1", view === v ? "bg-slate-700 text-white" : "bg-white text-slate-700 hover:bg-slate-50")}>
                    {v === "cards" ? "Review cards" : "Compact table"}
                  </button>
                ))}
              </div>
              <Button size="sm" variant="ghost" onClick={() => resetChoices()} title="Clear every unsaved per-well choice">
                Reset choices
              </Button>
              {view === "table" && (
                <>
                  <Button size="sm" onClick={() => setSaveSel(new Set(saveable.map((r) => r.well)))} disabled={!saveable.length}>
                    Select all saveable ({saveable.length})
                  </Button>
                  <Button size="sm" variant="primary" onClick={() => void doSave()} disabled={!writesOn || saveSel.size === 0}
                    busy={save.isPending}
                    title={writesOn ? "Append the chosen relations/IPRs to prop_hist" : "Saving is disabled in this environment (read-only)"}>
                    Save selected ({saveSel.size})
                  </Button>
                </>
              )}
            </div>
          }
        >
          <div className="space-y-2">
            {save.error && <ErrorNote error={save.error} />}
            {saveResult && (
              <InfoNote>
                Saved {saveResult.saved_wells} well{saveResult.saved_wells === 1 ? "" : "s"} as {saveResult.entry_user}.
                {saveResult.results.filter((r) => r.error).map((r) => (
                  <div key={r.well} className="text-xs text-red-700">{r.well}: {r.error}</div>
                ))}
              </InfoNote>
            )}
            {running && (
              <div className="text-xs text-slate-500">
                {boardKey !== wantKey ? `Showing ${board.pads.join(", ")} while ${want.join(", ")} loads...` : "Refreshing in the background - keep working."}
              </div>
            )}
            {view === "cards" ? (
              <WellCards board={board} dFor={dFor} onSave={(e) => void doSaveOne(e)} saving={savingWell} writesOn={writesOn} />
            ) : (
              <BoardTable
                board={board}
                selected={saveSel}
                onToggleSave={(well, on) =>
                  setSaveSel((s) => {
                    const n = new Set(s);
                    if (on) n.add(well);
                    else n.delete(well);
                    return n;
                  })
                }
              />
            )}
            <p className="text-xs text-slate-500">
              Save writes the relation (slope, WHP~header slope, fit r2/days, source) and the IPR anchor (liquid rate, flowing
              BHP, reservoir pressure) as new prop_hist rows, one statement per well. Jet-pump IPRs are saved in Solver.
            </p>
          </div>
        </Section>
      )}

      {board && <Correlations board={board} />}
    </div>
  );
}
