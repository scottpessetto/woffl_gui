import { useEffect, useMemo, useRef, useState } from "react";

import { isMissingJob, stableStringify } from "../api/client";
import { useCancelInstallationFit, useInstallationFitJob, useStartInstallationFit } from "../api/hooks";
import {
  HYDRAULICS_LABELS,
  type InstallationFitParam,
  type InstallationFitRequest,
  type InstallationFitResult,
  type InstallationModel,
  type PumpMatchResult,
} from "../api/types";
import { fmtDate, fmtNum } from "../lib/format";
import {
  changeouts, directionScore, excludedUids, fitAsMatch, paramsFor, shrinkageSigmas, type FitView,
} from "../lib/installationFit";
import { useParamsStore } from "../state/params";
import { useExcludedKeys } from "../state/excludedTests";
import { Badge, Button, ErrorNote } from "./ui";

const CHECKBOX = "h-4 w-4 rounded border-slate-300 accent-blue-600";
const SELECT = "rounded border border-slate-300 bg-white px-2 py-1 text-xs";
const MODELS: InstallationModel[] = ["M0", "M1", "M2", "M3"];
const PARAM_LABEL: Record<InstallationFitParam["physical"], string> = {
  ipr: "IPR scale", kth: "kth", kdi: "kdi", fnz: "Nozzle area factor",
};

/** A railed value is a bound hit, not an identified loss: say so everywhere. */
function paramText(p: InstallationFitParam): string {
  return `${fmtNum(p.value, 3)}${p.sd !== null ? ` ± ${fmtNum(p.sd, 3)}` : ""}${p.at_bound ? " (at bound)" : ""}`;
}

/** How far an installation's own data pulled it from the well level. */
function ShrinkBar({ p }: { p: InstallationFitParam }) {
  const s = shrinkageSigmas(p);
  if (s === null) return <span>-</span>;
  const width = Math.min(Math.abs(s), 3) / 3 * 100;
  return <span className="inline-flex items-center gap-2" title={`${fmtNum(s, 2)} prior sd from the well level`}>
    <span className="relative inline-block h-2 w-16 rounded bg-slate-100">
      <span className={`absolute top-0 h-2 rounded ${s >= 0 ? "left-1/2 bg-blue-500" : "right-1/2 bg-amber-500"}`}
        style={{ width: `${width / 2}%` }} />
      <span className="absolute left-1/2 top-[-2px] h-3 w-px bg-slate-400" />
    </span>
    {fmtNum(s, 1)} sd
  </span>;
}

function DeltaCell({ d, unit }: { d: { measured: number; predicted: number; clear: boolean; direction_correct: boolean | null } | null; unit: string }) {
  if (!d) return <td className="px-2 py-1.5">-</td>;
  const mark = !d.clear ? <Badge>no clear change</Badge>
    : d.direction_correct ? <Badge tone="good">direction right</Badge> : <Badge tone="poor">direction wrong</Badge>;
  return <td className="px-2 py-1.5">{fmtNum(d.measured)} / {fmtNum(d.predicted)} {unit} {mark}</td>;
}

/**
 * Fit the well across all its pump installations (read-only job) and show
 * how the model was chosen. Hands the chosen view to the history chart.
 */
export function InstallationFit({ well, contextReady, onMatch }: {
  well: string; contextReady: boolean; onMatch: (match: PumpMatchResult | null) => void;
}) {
  const hydraulics = useParamsStore((s) => s.params.hydraulics_model);
  const excludedKeys = useExcludedKeys(well);
  const [months, setMonths] = useState<InstallationFitRequest["months"]>(24);
  const [refitIpr, setRefitIpr] = useState(false);
  const [includeM3, setIncludeM3] = useState(true);
  const [viewModel, setViewModel] = useState<InstallationModel | null>(null);
  const [view, setView] = useState<FitView>("fit");
  const [handle, setHandle] = useState<{ id: string; key: string } | null>(null);
  const start = useStartInstallationFit();
  const cancel = useCancelInstallationFit();
  const cancelJob = cancel.mutate;
  const request = useMemo<InstallationFitRequest>(() => ({ hydraulics_model: hydraulics, months, refit_ipr: refitIpr,
    include_m3: includeM3, exclude_wt_uids: excludedUids(excludedKeys) }), [hydraulics, months, refitIpr, includeM3, excludedKeys]);
  const key = stableStringify({ well, request });
  const latestKey = useRef(key);
  const currentId = handle?.key === key ? handle.id : null;
  const query = useInstallationFitJob(currentId);
  const result: InstallationFitResult | null = handle?.key === key && query.data?.status === "done" &&
    query.data.result?.well === well ? query.data.result : null;
  const running = !!currentId && (!query.data || query.data.status === "running") && !isMissingJob(query.error);
  const shown: InstallationModel | null = result
    ? (viewModel && result.models[viewModel] ? viewModel : result.selected.model) : null;
  const match = useMemo(() => result && shown ? fitAsMatch(result, shown, view) : null, [result, shown, view]);

  useEffect(() => { onMatch(match); }, [match, onMatch]);
  useEffect(() => () => onMatch(null), [onMatch]);
  useEffect(() => {
    if (handle && handle.key !== key) setHandle(null);
  }, [key, handle]);
  useEffect(() => () => { if (handle) cancelJob(handle.id); }, [handle, cancelJob]);
  useEffect(() => {
    latestKey.current = key;
    return () => { latestKey.current = "unmounted"; };
  }, [key]);

  const run = async () => {
    const requestedKey = key;
    setViewModel(null);
    try {
      const { job_id } = await start.mutateAsync({ well, request });
      if (latestKey.current !== requestedKey) cancelJob(job_id);
      else setHandle({ id: job_id, key: requestedKey });
    } catch { /* rendered below */ }
  };

  const shownModel = result && shown ? result.models[shown] : undefined;
  const wellParams = shownModel ? paramsFor(shownModel.params, null) : [];
  const shownChangeouts = result && shown ? changeouts(result.cv.folds, shown) : [];
  const eras = result?.eras.filter((e) => e.n_tests > 0) ?? [];

  return <div className="space-y-3 text-xs text-slate-600">
    <div className="flex flex-wrap items-center gap-4">
      <label>History <select aria-label="Installation fit window" className={`${SELECT} ml-1`} value={months}
        onChange={(e) => setMonths(Number(e.target.value) as InstallationFitRequest["months"])}>
        <option value={24}>24 months</option><option value={60}>60 months</option>
      </select></label>
      <label className="flex cursor-pointer items-center gap-2"
        title="Refits the scale of the ONE oil IPR across every test. Nothing is saved.">
        <input className={CHECKBOX} type="checkbox" checked={refitIpr} onChange={(e) => setRefitIpr(e.target.checked)} />
        Refit the one IPR
      </label>
      <label className="flex cursor-pointer items-center gap-2">
        <input className={CHECKBOX} type="checkbox" checked={includeM3} onChange={(e) => setIncludeM3(e.target.checked)} />
        Include per-installation loss offsets (M3)
      </label>
      <Button size="sm" variant="secondary" busy={start.isPending} disabled={!contextReady || start.isPending || running} onClick={run}>
        {result ? "Fit again" : "Fit across installations"}
      </Button>
      {running && <Button size="sm" variant="secondary" disabled={cancel.isPending} onClick={() => currentId && cancelJob(currentId)}>Cancel</Button>}
    </div>
    <div className="space-y-1 text-slate-500" aria-live="polite">
      <p>{HYDRAULICS_LABELS[hydraulics]}. Fits every pump installation together: one oil IPR for the well
        ({refitIpr ? "its single scale refitted" : "the saved curve held"}), shared well-level pump losses, and each
        installation's own nozzle-area factor pulled toward the well level unless its tests say otherwise. Then refits
        on earlier tests only to score predictions of later ones, and picks the simplest model that predicts about as
        well as the best. Read-only; nothing is saved.</p>
      {request.exclude_wt_uids.length > 0 && <p>{request.exclude_wt_uids.length} test{request.exclude_wt_uids.length === 1 ? "" : "s"} excluded in the Solver are left out.</p>}
      {!contextReady && <p>Loading saved well inputs...</p>}
      {running && <p>{query.data?.progress ?? "Starting fit..."} This can take a few minutes.</p>}
      {query.data?.status === "cancelled" && <p>Fit cancelled.</p>}
      {query.data?.status === "error" && <p className="text-amber-700">{query.data.error}</p>}
      {start.isError && <ErrorNote error={start.error} />}
      {query.isError && <ErrorNote error={query.error} />}
    </div>

    {result && <>
      <div className="space-y-2 rounded border border-slate-200 bg-slate-50 p-3">
        <p className="flex flex-wrap items-center gap-2 text-sm font-semibold text-slate-800">
          Chosen: {result.selected.model} - {result.selected.model && result.models[result.selected.model]?.label}
          {result.selected.basis === "held_out_1se" ? <Badge tone="info">held-out, one-standard-error rule</Badge>
            : <Badge tone="fair">not tested on unseen data</Badge>}
        </p>
        <p>{result.selected.reason}</p>
        {(() => {
          const chosen = result.selected.model ? result.models[result.selected.model] : undefined;
          const railed = chosen?.params.filter((p) => p.at_bound && p.installation_id === null) ?? [];
          return railed.length > 0 && <p className="text-amber-700">
            {railed.map((p) => PARAM_LABEL[p.physical]).join(", ")} sits at its bound. The better held-out fit may come from
            absorbing a steady model bias (for example BHP modelled high), not from identified pump wear. Treat it as a level
            correction for this well's history, not as evidence for a pump-size change.</p>;
        })()}
        <p>IPR check: the saved curve misses test oil at measured BHP by a median{" "}
          {fmtNum(result.ipr_gate.median_abs === null ? null : 100 * result.ipr_gate.median_abs, 0)}%
          ({fmtNum(result.ipr_gate.median_signed === null ? null : 100 * result.ipr_gate.median_signed, 0)}% signed, {result.ipr_gate.count} tests).
          {result.ipr_gate.passes ? " Within the 20% gate."
            : result.request.refit_ipr ? " Above the 20% gate; this fit refitted the one curve's scale, so pump terms were fitted against the refitted curve."
              : " Above the 20% gate, so pump terms were not fitted. Tick Refit the one IPR to fit them against a refitted curve."}</p>
        {result.losses?.reason && <p>{result.losses.reason}</p>}
        {Object.entries(result.skipped).map(([m, why]) => <p key={m} className="text-amber-700">{m} not fitted: {why}</p>)}
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left">
          <caption className="mb-2 text-left font-semibold text-slate-700">Models compared - select one to show it on the chart</caption>
          <thead className="border-b text-slate-500"><tr>
            {["Model", "Well-level values", "Fit (std RMS)", "AICc", "Held-out loss ± SE", "Changeout direction", ""].map((h) => <th key={h} className="px-2 py-1.5 font-medium">{h}</th>)}
          </tr></thead>
          <tbody>{MODELS.filter((m) => result.models[m] || result.skipped[m]).map((m) => {
            const s = result.models[m];
            const cv = result.cv.scores[m];
            const dirs = directionScore(changeouts(result.cv.folds, m));
            return <tr key={m} className={`border-b border-slate-100 ${shown === m ? "bg-blue-50" : ""}`}>
              <td className="px-2 py-1.5">{m} {result.selected.model === m && <Badge tone="good">chosen</Badge>}<br />
                <span className="text-slate-500">{s?.label ?? "not fitted"}</span></td>
              <td className="px-2 py-1.5">{s ? paramsFor(s.params, null).map((p) => `${PARAM_LABEL[p.physical]} ${paramText(p)}`).join("; ") || "reference" : "-"}</td>
              <td className="px-2 py-1.5">{fmtNum(s?.rms_standardized, 2)}</td>
              <td className="px-2 py-1.5">{fmtNum(s?.aicc, 1)}</td>
              <td className="px-2 py-1.5">{cv ? `${fmtNum(cv.mean_loss, 2)} ± ${fmtNum(cv.se, 2)} (${cv.n})` : "-"}</td>
              <td className="px-2 py-1.5">{dirs.clear ? `${dirs.correct}/${dirs.clear} right` : "-"}</td>
              <td className="px-2 py-1.5">{s && <button className="text-blue-700 underline" onClick={() => setViewModel(m)}>{shown === m ? "shown" : "show"}</button>}</td>
            </tr>;
          })}</tbody>
        </table>
      </div>
      <p className="text-slate-500">Fit: RMS of standardized misses (BHP / 50 psi, oil / 10%, PF / 5%). Held-out loss: mean robust loss on tests
        predicted by refits on earlier tests only ({result.cv.embargo_days}-day embargo); lower is better. Changeout direction: whether the held-out
        prediction got the sign of each clear BHP/oil change across a pump change right.</p>

      <div className="flex flex-wrap items-center gap-4">
        <span className="font-medium text-slate-700">Chart ({shown}):</span>
        {(["fit", "held_out"] as const).map((v) => <label key={v} className="flex cursor-pointer items-center gap-1.5">
          <input type="radio" name={`fit-view-${well}`} checked={view === v} onChange={() => setView(v)} />
          {v === "fit" ? "Fitted history (dotted)" : "Held-out predictions (dashed)"}
        </label>)}
      </div>

      {shownModel && <div className="overflow-x-auto">
        <table className="w-full text-left">
          <caption className="mb-2 text-left font-semibold text-slate-700">{shown} by installation</caption>
          <thead className="border-b text-slate-500"><tr>
            {["Pump / set date", "Tests", "PF span (psi)", "Own parameters", "Pulled from well level", "Identified by its tests"].map((h) => <th key={h} className="px-2 py-1.5 font-medium">{h}</th>)}
          </tr></thead>
          <tbody>
            <tr className="border-b border-slate-100 bg-slate-50">
              <td className="px-2 py-1.5 font-medium">Well level (all pumps)</td><td className="px-2 py-1.5" colSpan={2}>-</td>
              <td className="px-2 py-1.5">{wellParams.map((p) => `${PARAM_LABEL[p.physical]} ${paramText(p)}`).join("; ") || "reference losses"}</td>
              <td className="px-2 py-1.5">-</td>
              <td className="px-2 py-1.5">{wellParams.map((p) => <span key={p.name} className="mr-1">{p.identified ? <Badge tone="good">{p.physical}</Badge> : <Badge tone="fair">{p.physical}: prior</Badge>}</span>)}</td>
            </tr>
            {eras.map((e) => {
              const own = paramsFor(shownModel.params, e.installation_id);
              return <tr key={e.installation_id} className="border-b border-slate-100">
                <td className="px-2 py-1.5">{e.pump} / {fmtDate(e.date_set)}{e.flags.length > 0 && <span className="text-amber-700" title={e.flags.join(" ")}> *</span>}</td>
                <td className="px-2 py-1.5">{e.n_tests}</td>
                <td className="px-2 py-1.5">{fmtNum(e.ppf_span)}</td>
                <td className="px-2 py-1.5">{own.length ? own.map((p) => <div key={p.name}>{PARAM_LABEL[p.physical]}{p.name.startsWith("d") ? " offset" : ""} {paramText(p)}</div>) : "well level only"}</td>
                <td className="px-2 py-1.5">{own.map((p) => <div key={p.name}><ShrinkBar p={p} /></div>)}</td>
                <td className="px-2 py-1.5">{own.map((p) => <div key={p.name}>{p.identified ? <Badge tone="good">yes</Badge> : <Badge tone="fair">mostly prior</Badge>}</div>)}</td>
              </tr>;
            })}
          </tbody>
        </table>
        <p className="mt-1 text-slate-500">Pulled from well level: how far this pump's own tests moved its value, in prior standard deviations. Mostly prior: its
          uncertainty is barely narrower than the prior, so the tests say little about it. ± values are model-conditional (Laplace), inflated by misfit.</p>
      </div>}

      {shownChangeouts.length > 0 && <div className="overflow-x-auto">
        <table className="w-full text-left">
          <caption className="mb-2 text-left font-semibold text-slate-700">Pump changes: measured vs held-out predicted change ({shown})</caption>
          <thead className="border-b text-slate-500"><tr>
            {["First test on new pump", "BHP change measured / predicted", "Oil change measured / predicted"].map((h) => <th key={h} className="px-2 py-1.5 font-medium">{h}</th>)}
          </tr></thead>
          <tbody>{shownChangeouts.map((c) => <tr key={c.origin} className="border-b border-slate-100">
            <td className="px-2 py-1.5">{fmtDate(c.origin)}</td>
            <DeltaCell d={c.bhp} unit="psi" /><DeltaCell d={c.oil} unit="BOPD" />
          </tr>)}</tbody>
        </table>
        <p className="mt-1 text-slate-500">Medians of the last 3 tests before and the first 3 after each change. This is the quantity a pump-size decision relies on.</p>
      </div>}

      <details>
        <summary className="cursor-pointer">Held-out checks by split ({result.cv.folds.length})</summary>
        <table className="mt-2 w-full text-left">
          <thead className="border-b text-slate-500"><tr>
            {["Split", "Kind", "Train / test", "BHP RMS (psi)", "Oil error", "PF error", "Loss"].map((h) => <th key={h} className="px-2 py-1 font-medium">{h}</th>)}
          </tr></thead>
          <tbody>{result.cv.folds.map((f) => {
            const s = shown ? f.models[shown] : undefined;
            return <tr key={f.origin} className="border-b border-slate-100">
              <td className="px-2 py-1">{fmtDate(f.origin)}</td>
              <td className="px-2 py-1">{f.kind === "changeout" ? "new pump" : "later, same pump"}</td>
              <td className="px-2 py-1">{f.n_train} / {f.n_test}</td>
              {f.skipped ? <td className="px-2 py-1" colSpan={4}>{f.skipped}</td> : <>
                <td className="px-2 py-1">{fmtNum(s?.bhp_rms)}</td>
                <td className="px-2 py-1">{s?.oil_median_abs_pct == null ? "-" : `${fmtNum(s.oil_median_abs_pct, 0)}%`}</td>
                <td className="px-2 py-1">{s?.pf_median_abs_pct == null ? "-" : `${fmtNum(s.pf_median_abs_pct, 1)}%`}</td>
                <td className="px-2 py-1">{fmtNum(s?.mean_loss, 2)}</td></>}
            </tr>;
          })}</tbody>
        </table>
      </details>
      <details className="text-slate-500"><summary className="cursor-pointer">Fit assumptions and source</summary>
        <div className="mt-2 space-y-1">{result.notes.map((n) => <p key={n}>{n}</p>)}
          <p>Model {result.physics_model}; source {result.source}; captured {result.as_of}; {fmtNum(result.seconds, 0)} s.</p>
          <p>Not validated for sizing: replacements still use reference losses.</p>
        </div>
      </details>
    </>}
  </div>;
}
