import { useMutation } from "@tanstack/react-query";
import { useMemo, useState } from "react";

import { post, stableStringify } from "../api/client";
import type { CommonOilIprRequest, CommonOilIprResult } from "../api/types";
import { fmtDate, fmtNum } from "../lib/format";
import { effectiveParams, useParamsStore } from "../state/params";
import { Button, ErrorNote } from "./ui";

/** An explicit candidate; fitting, Apply and database Save are separate actions. */
export function CommonOilIprFit({ well, onApply }: { well: string; onApply?: () => void }) {
  const params = useParamsStore((s) => s.params);
  const context = useParamsStore((s) => s.context);
  const [open, setOpen] = useState(false);
  const [months, setMonths] = useState<CommonOilIprRequest["months"]>(24);
  const [fraction, setFraction] = useState(.25);
  const [excluded, setExcluded] = useState<string[]>([]);
  const [eras, setEras] = useState<string[]>([]);
  const [applied, setApplied] = useState(false);
  const request = useMemo<CommonOilIprRequest>(() => ({ well, params: effectiveParams(params), months,
    holdout_fraction: fraction, exclude_tests: [...excluded].sort(), exclude_eras: [...eras].sort() }),
    [well, params, months, fraction, excluded, eras]);
  const fit = useMutation({ mutationFn: (req: CommonOilIprRequest) => post<CommonOilIprResult>("/common-ipr-fit", req) });
  const result = fit.data?.request.well === well ? fit.data : null;
  const current = !!result && stableStringify(result.request) === stableStringify(request);
  const reflectsApplied = applied && !!result?.seeds && stableStringify({ ...result.request,
    params: { ...result.request.params, ...result.seeds } }) === stableStringify(request);
  const supported = context?.well === well && !params.model_as_water && params.form_wc < .99 && params.pwf < params.pres;
  const toggleTest = (id: string) => { setApplied(false); setExcluded((prev) => prev.includes(id) ? prev.filter((v) => v !== id) : [...prev, id]); };
  const toggleEra = (id: string) => { setApplied(false); setEras((prev) => prev.includes(id) ? prev.filter((v) => v !== id) : [...prev, id]); };
  const apply = () => {
    const live = useParamsStore.getState();
    if (!current || !result?.seeds || live.well !== well || stableStringify(effectiveParams(live.params)) !== stableStringify(result.request.params)) return;
    live.setMany(result.seeds);
    live.setCommonIprIntent(true);
    live.markFitApplied(well);
    live.setMatchNote(`Common oil IPR: fixed Pr ${result.request.params.pres} psi; ${result.training.dates} training dates across ${result.training.installations} installations; holdout from ${result.split_date?.slice(0, 10) ?? "none"}. Conditional on measured BHP; forward replay still required.`);
    setApplied(true);
    onApply?.();
  };
  return <div className="rounded-md border border-slate-200 bg-slate-50/50 p-3 text-xs text-slate-600">
    <button type="button" aria-expanded={open} className="text-left font-semibold text-slate-700" onClick={() => setOpen(!open)}>
      {open ? "Hide" : "Fit"} one oil IPR across pump history
    </button>
    {open && <div className="mt-3 space-y-3">
      <p>Fit a candidate oil curve from measured oil/BHP. Reservoir pressure stays at {fmtNum(params.pres)} psi; anchor BHP, WC and GOR stay unchanged. Edit reservoir pressure deliberately before fitting if needed.</p>
      <div className="flex flex-wrap items-center gap-3">
        <label>History <select aria-label="Common IPR history" className="rounded border border-slate-300 bg-white px-2 py-1" value={months}
          onChange={(e) => { setMonths(Number(e.target.value) as CommonOilIprRequest["months"]); setApplied(false); }}>
          {[6, 12, 24, 60].map((n) => <option key={n} value={n}>{n} months</option>)}
        </select></label>
        <label>Latest dates held out <select aria-label="Common IPR holdout" className="rounded border border-slate-300 bg-white px-2 py-1" value={fraction}
          onChange={(e) => { setFraction(Number(e.target.value)); setApplied(false); }}>
          {[.2, .25, .33, .5].map((n) => <option key={n} value={n}>{Math.round(n*100)}%</option>)}
        </select></label>
        <Button size="sm" variant="secondary" disabled={!supported || fit.isPending} busy={fit.isPending}
          onClick={() => { setApplied(false); fit.mutate(request); }}>Fit candidate oil IPR</Button>
        <Button size="sm" disabled={!current || !result?.seeds || fit.isPending} onClick={apply}>Apply candidate IPR</Button>
      </div>
      {!supported && <p>Load an oil well with WC below 99% and anchor BHP below reservoir pressure.</p>}
      <p>Holdout oil uses measured BHP; it is a curve check, not an operating-rate forecast. Forward history replay and pressure-response validation remain necessary.</p>
      {fit.isError && <ErrorNote error={fit.error} />}
      {reflectsApplied && <p role="status" className="text-emerald-700">Candidate applied to the session. Run the history comparison using Current edits (preview), then Save well inputs when satisfied.</p>}
      {result && !current && !reflectsApplied && <p role="status" className="text-amber-700">Inputs or training selections changed. Fit again before applying; the table below belongs to the previous fit.</p>}
      {result && <>
        <div className="flex flex-wrap gap-x-5 gap-y-1 text-slate-700">
          <span>Training: {result.training.dates} dates / {result.training.installations} pumps</span>
          <span>Holdout: {result.holdout.dates} dates from {fmtDate(result.split_date)}</span>
          <span>Oil Qmax: {fmtNum(result.qmax_oil)} BOPD</span>
        </div>
        <table className="w-full text-left"><thead><tr><th>Oil error at measured BHP</th><th>Current curve</th><th>Candidate</th></tr></thead>
          <tbody>{(["training", "holdout"] as const).map((kind) => <tr key={kind}><td className="capitalize">{kind} MAE (BOPD)</td>
            <td>{fmtNum(result[kind].baseline_mae)}</td><td>{fmtNum(result[kind].candidate_mae)}</td></tr>)}</tbody>
        </table>
        <details><summary className="cursor-pointer">Review training selections and excluded tests</summary>
          <p className="my-2">Selections affect training only. Held-out dates stay fixed for this window and split. Repeated tuning after viewing holdouts makes this comparison exploratory.</p>
          <div className="mb-2 flex flex-wrap gap-3">{result.eras.map((era) => <label key={era.id} className="flex items-center gap-1">
            <input type="checkbox" checked={!eras.includes(era.id)} onChange={() => toggleEra(era.id)} /> {era.pump} set {fmtDate(era.date_set)}
          </label>)}</div>
          <div className="max-h-72 overflow-auto"><table className="w-full text-left"><thead><tr><th>Train</th><th>Date / pump</th><th>BHP</th><th>Oil</th><th>WC / GOR</th><th>Use / reason</th></tr></thead>
            <tbody>{result.rows.map((r, i) => <tr key={`${r.test_id}:${i}`} className="border-t border-slate-100">
              <td>{r.phase === "training" && (!r.reason || r.reason === "Excluded from training by user.") && <input type="checkbox"
                aria-label={`Include training test ${r.test_id}`} checked={!excluded.includes(r.test_id) && !eras.includes(r.era_id ?? "")} onChange={() => toggleTest(r.test_id)} />}</td>
              <td>{fmtDate(r.date)} / {r.pump}</td><td>{fmtNum(r.bhp)}</td><td>{fmtNum(r.oil)}</td>
              <td>{fmtNum(r.wc === null ? null : 100*r.wc, 1)}% / {fmtNum(r.gor)}</td><td>{r.reason ?? r.phase}</td>
            </tr>)}</tbody></table></div>
        </details>
        <details><summary className="cursor-pointer">Fit assumptions and limitations</summary>{result.notes.map((n) => <p key={n} className="mt-1">{n}</p>)}</details>
      </>}
    </div>}
  </div>;
}
