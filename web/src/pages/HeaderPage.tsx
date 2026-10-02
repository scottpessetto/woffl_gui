/**
 * Header - what a production-header pressure change costs each pad.
 *
 * The first question is one step: pick pads, type the change (or pick an
 * observed event), Estimate. The answer comes back with a range and a
 * response curve. Below it the wells board shows how every well was
 * represented; engineers refine it as time allows - mark bad gauges, assign
 * BHP~WHP and reservoir IPR correlations to wells without gauges, enter
 * manual values - and Save makes those the next run's defaults.
 *
 * "Header" is the PRODUCTION header (wellhead back-pressure), not the
 * power-fluid header the pad optimizer sweeps.
 */

import clsx from "clsx";
import { useEffect, useMemo } from "react";
import { useSearchParams } from "react-router-dom";

import { isMissingJob, stableStringify } from "../api/client";
import { useHeaderJob, useHeaderPads, useMeta, useStartHeaderRun } from "../api/hooks";
import type { HeaderBoard, HeaderRunResult } from "../api/types";
import { Button, Card, ErrorNote, InfoNote, Section, WarnNote } from "../components/ui";
import { fmtNum } from "../lib/format";
import { useHeaderStore } from "../state/header";
import { JobError, JobLine } from "./header/JobStatus";
import { DEFAULT_DELTA_PSI, buildRunRequest, deltaFor, runBlockers } from "./header/model";
import { ResultPanel } from "./header/ResultPanel";
import { WellsTab } from "./header/WellsTab";

const FIELD = "h-8 rounded-md border border-slate-300 bg-white px-2 text-sm tabular-nums";

type Tab = "impact" | "wells";

export default function HeaderPage() {
  const [params, setParams] = useSearchParams();
  const tab: Tab = params.get("tab") === "wells" ? "wells" : "impact";
  const setTab = (t: Tab) => setParams(t === "impact" ? {} : { tab: t }, { replace: true });

  const padsQ = useHeaderPads();
  const meta = useMeta();
  const form = useHeaderStore((s) => s.form);
  const setForm = useHeaderStore((s) => s.setForm);
  const choices = useHeaderStore((s) => s.choices);
  const runJob = useHeaderStore((s) => s.runJob);
  const runKey = useHeaderStore((s) => s.runKey);
  const setRunJob = useHeaderStore((s) => s.setRunJob);
  const wellsPads = useHeaderStore((s) => s.wellsPads);
  const setWellsPads = useHeaderStore((s) => s.setWellsPads);

  const startRun = useStartHeaderRun();
  const runQ = useHeaderJob(runJob);

  // A server restart or the 1 h job TTL forgets jobs: drop the stale id.
  useEffect(() => {
    if (isMissingJob(runQ.error)) setRunJob(null);
  }, [runQ.error, setRunJob]);
  // First visit to the Wells tab starts from the Impact tab's pads.
  useEffect(() => {
    if (tab === "wells" && !wellsPads.length && form.pads.length) setWellsPads(form.pads);
  }, [tab, wellsPads.length, form.pads, setWellsPads]);

  const run = runQ.data?.status === "done" ? (runQ.data.result as HeaderRunResult) : null;
  const pads = padsQ.data?.pads ?? [];
  const selectedPads = [...form.pads].sort();
  // The last Estimate's wells (for the "now" header labels and to trim
  // choices to the ones that differ from its defaults).
  const board: HeaderBoard | null = run?.board && run.board.pads.join(",") === selectedPads.join(",") ? run.board : null;

  const request = useMemo(() => buildRunRequest(form, board, choices), [board, form, choices]);
  const blockers = useMemo(() => runBlockers(form, board, choices), [form, board, choices]);
  const stale = Boolean(run && runKey && runKey !== stableStringify(request));
  const busy = runQ.data?.status === "running";

  const doRun = async () => {
    const { job_id } = await startRun.mutateAsync(request);
    setRunJob(job_id, stableStringify(request));
  };
  // Each card's impact line uses the change on screen: the run's measured
  // change after an event run, otherwise the typed change for that pad.
  const dFor = (pad: string) => {
    if (form.mode === "event" && run?.mode === "event") {
      const p = run.pads.find((x) => x.pad === pad);
      if (p && typeof p.d_header === "number") return p.d_header;
    }
    return deltaFor(form, pad);
  };
  const writesOn = meta.data?.writes_enabled === true;

  return (
    <div className="mx-auto max-w-[1500px] space-y-5 p-4">
      <div className="flex items-center gap-1 border-b border-slate-200">
        {([["impact", "Impact"], ["wells", "Wells"]] as const).map(([t, label]) => (
          <button
            key={t}
            type="button"
            onClick={() => setTab(t)}
            className={clsx(
              "-mb-px border-b-2 px-4 py-2 text-sm font-medium",
              tab === t ? "border-blue-600 text-blue-700" : "border-transparent text-slate-500 hover:text-slate-800",
            )}
          >
            {label}
          </button>
        ))}
      </div>

      {tab === "wells" ? (
        <WellsTab dFor={dFor} writesOn={writesOn} />
      ) : (
      <>
      <Section title="Header pressure impact">
        <p className="mb-3 max-w-4xl text-sm text-slate-600">
          What a production-header pressure change costs each well and pad. Jet pumps use their BHP~WHP relation (saved,
          measured, else the jet-pump group correlation) on their pump model's IPR; one with no relation - or every jet
          pump, if you pick it - is solved with the WOFFL model at both wellhead pressures. ESP, gas-lift and flowing wells use their measured closed-loop BHP~WHP slope (the rate
          response is already in it) and their IPR; wells without a working gauge borrow their lift type's and reservoir's
          correlations.
        </p>
        <Card>
          <div className="flex flex-wrap items-end gap-5">
            <div>
              <div className="text-xs text-slate-500">Pads</div>
              <div className="mt-1 flex flex-wrap gap-1">
                {pads.map((p) => (
                  <button
                    key={p}
                    type="button"
                    onClick={() =>
                      setForm({ pads: form.pads.includes(p) ? form.pads.filter((x) => x !== p) : [...form.pads, p].sort() })
                    }
                    className={clsx(
                      "h-8 min-w-8 rounded-md border px-2 text-sm transition-colors",
                      form.pads.includes(p)
                        ? "border-blue-500 bg-blue-50 font-medium text-blue-700"
                        : "border-slate-300 bg-white text-slate-600 hover:bg-slate-50",
                    )}
                  >
                    {p}
                  </button>
                ))}
                {padsQ.isLoading && <span className="text-xs text-slate-400">loading pads...</span>}
              </div>
            </div>
            <div>
              <div className="text-xs text-slate-500">Change</div>
              <div className="mt-1 inline-flex overflow-hidden rounded-md border border-slate-300">
                {(["scenario", "event"] as const).map((m) => (
                  <button
                    key={m}
                    type="button"
                    onClick={() => setForm({ mode: m })}
                    className={clsx("px-3 py-1.5 text-sm", form.mode === m ? "bg-blue-600 text-white" : "bg-white text-slate-700 hover:bg-slate-50")}
                  >
                    {m === "scenario" ? "Header change" : "Observed event"}
                  </button>
                ))}
              </div>
            </div>
            <div title="How jet pumps turn the wellhead-pressure change into a BHP change. The BHP~WHP relation uses each well's saved or measured closed-loop slope, else the jet-pump group correlation, and falls back to the WOFFL model for a well with none. The WOFFL model solves the installed pump at both pressures with PF held. Both use the pump model's IPR. Single wells can be set otherwise on the Wells tab.">
              <div className="text-xs text-slate-500">Jet pumps</div>
              <div className="mt-1 inline-flex overflow-hidden rounded-md border border-slate-300">
                {([["empirical", "BHP~WHP relation"], ["model", "WOFFL model"]] as const).map(([m, label]) => (
                  <button
                    key={m}
                    type="button"
                    onClick={() => setForm({ jpMethod: m })}
                    className={clsx("px-3 py-1.5 text-sm", (form.jpMethod ?? "empirical") === m ? "bg-blue-600 text-white" : "bg-white text-slate-700 hover:bg-slate-50")}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>
            <div className="ml-auto flex items-center gap-2">
              <Button
                variant="primary"
                onClick={() => void doRun()}
                disabled={blockers.length > 0 || busy}
                busy={startRun.isPending}
                title="Load the pads' wells if needed and estimate the impact"
              >
                Estimate
              </Button>
            </div>
          </div>

          {form.pads.length > 0 && (
            <div className="mt-4 flex flex-wrap items-end gap-3">
              {form.mode === "scenario" ? (
                <>
                  {selectedPads.map((p) => (
                    <label key={p} className="text-xs text-slate-500">
                      {p}-Pad header change (psi)
                      <input
                        type="number"
                        step={5}
                        className={clsx(FIELD, "mt-1 block w-28")}
                        value={deltaFor(form, p)}
                        onChange={(e) => setForm({ deltas: { ...form.deltas, [p]: Number(e.target.value) } })}
                      />
                      <span className="text-[11px] text-slate-400">
                        {board?.header_now[p] != null ? `now ${fmtNum(board.header_now[p], 0)} psi` : " "}
                      </span>
                    </label>
                  ))}
                  {selectedPads.length > 1 && (
                    <Button
                      size="sm"
                      onClick={() => {
                        const v = deltaFor(form, selectedPads[0]);
                        setForm({ deltas: Object.fromEntries(selectedPads.map((p) => [p, v])) });
                      }}
                      title="Copy the first pad's change to every pad"
                    >
                      Same for all
                    </Button>
                  )}
                </>
              ) : (
                <>
                  <label className="text-xs text-slate-500">
                    Event time (historian local)
                    <input
                      type="datetime-local"
                      className={clsx(FIELD, "mt-1 block")}
                      value={form.eventTime}
                      onChange={(e) => setForm({ eventTime: e.target.value })}
                    />
                  </label>
                  {([["preHours", "Hours before"], ["postHours", "Hours after"], ["gapHours", "Skip either side"]] as const).map(([k, label]) => (
                    <label key={k} className="text-xs text-slate-500">
                      {label}
                      <input
                        type="number"
                        min={0}
                        className={clsx(FIELD, "mt-1 block w-24")}
                        value={form[k]}
                        onChange={(e) => setForm({ [k]: Number(e.target.value) })}
                      />
                    </label>
                  ))}
                  <label className="text-xs text-slate-500">
                    Event well (excluded)
                    <input
                      className={clsx(FIELD, "mt-1 block w-28")}
                      list="header-wells"
                      placeholder="e.g. MPL-20"
                      value={form.eventWell}
                      onChange={(e) => setForm({ eventWell: e.target.value.toUpperCase() })}
                    />
                    <datalist id="header-wells">
                      {(board?.rows ?? []).map((r) => <option key={r.well} value={r.well} />)}
                    </datalist>
                  </label>
                  <label className="text-xs text-slate-500" title="Optional: the event well's new oil rate, to report the net">
                    Its oil (BOPD)
                    <input
                      type="number"
                      min={0}
                      className={clsx(FIELD, "mt-1 block w-24")}
                      value={form.eventWellOil ?? ""}
                      onChange={(e) => setForm({ eventWellOil: e.target.value === "" ? null : Number(e.target.value) })}
                    />
                  </label>
                </>
              )}
            </div>
          )}
          {blockers.length > 0 && (
            <ul className="mt-2 list-disc pl-5 text-xs text-amber-800">
              {blockers.map((b) => <li key={b}>{b}</li>)}
            </ul>
          )}
          <div className="mt-2 space-y-2">
            <JobLine job={runQ.data} jobId={runJob} label="Estimating" />
            <JobError job={runQ.data} />
            {startRun.error && <ErrorNote error={startRun.error} />}
          </div>
          {form.mode === "scenario" && form.pads.length > 0 && (
            <div className="mt-2 text-xs text-slate-400">
              Positive = header rises (the usual cost of bringing a well on). Pads without an entry use {DEFAULT_DELTA_PSI} psi.
            </div>
          )}
        </Card>
      </Section>

      {run && stale && <WarnNote>Inputs changed since this result. Estimate again to update it.</WarnNote>}
      {run && <ResultPanel result={run} />}

      {!run && (
        <InfoNote>
          Estimate uses each well's saved relation and IPR, or its defaults. Review or change them on the{" "}
          <button type="button" className="font-medium underline" onClick={() => setTab("wells")}>Wells tab</button>.
        </InfoNote>
      )}
      </>
      )}
    </div>
  );
}
