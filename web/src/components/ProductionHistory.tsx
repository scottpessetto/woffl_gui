import { useEffect, useMemo, useRef, useState } from "react";

import { stableStringify, isMissingJob } from "../api/client";
import { useCancelPumpMatch, usePumpMatchJob, useStartPumpMatch } from "../api/hooks";
import { HYDRAULICS_LABELS, type JpHistoryResponse, type PumpMatchRequest } from "../api/types";
import { fmtDate, fmtNum } from "../lib/format";
import { changedWellInputs, wellInputProblem, wellInputValues } from "../lib/wellInputs";
import { useParamsStore } from "../state/params";
import { HistoryStrip } from "./HistoryStrip";
import { Button, ErrorNote } from "./ui";

const CHECKBOX = "h-4 w-4 rounded border-slate-300 accent-blue-600";
const SELECT = "rounded border border-slate-300 bg-white px-2 py-1 text-xs";
const errorPercent = (value: number | null) => value === null ? "-" : `${fmtNum(value, 1)}%`;

/** Shared production plot and replay controls for Solver and JP History. */
export function ProductionHistory(props: {
  data: JpHistoryResponse; bhpFromZero?: boolean; showPf?: boolean; height?: number; gaugePreview?: boolean; initialShow?: boolean;
}) {
  const { data } = props;
  const params = useParamsStore((s) => s.params);
  const model = params.hydraulics_model;
  const context = useParamsStore((s) => s.context);
  const contextReady = context?.well === data.well;
  const [show, setShow] = useState(props.initialShow ?? false);
  const [months, setMonths] = useState<PumpMatchRequest["months"]>(24);
  const [mode, setMode] = useState<PumpMatchRequest["mode"]>("all_tests");
  const allTests = mode === "all_tests";
  const [useEdits, setUseEdits] = useState(false);
  const preview = allTests && useEdits;
  const inputProblem = preview ? wellInputProblem(params) : null;
  const changed = changedWellInputs(params, context?.seeds).length;
  const previewValues = useMemo(() => wellInputValues(params, context?.seeds), [params, context?.seeds]);
  const [training, setTraining] = useState(10);
  const [oilDetail, setOilDetail] = useState(true);
  const [pfDetail, setPfDetail] = useState(false);
  const [selectedEra, setSelectedEra] = useState<string | null>(null);
  const [testId, setTestId] = useState<string | null>(null);
  const [handle, setHandle] = useState<{ id: string; key: string } | null>(null);
  const start = useStartPumpMatch();
  const cancel = useCancelPumpMatch();
  const cancelJob = cancel.mutate;
  const request = useMemo<PumpMatchRequest>(() => ({ hydraulics_model: model, months, mode, training_tests: training,
    edited_inputs: preview ? previewValues : null }), [model, months, mode, training, preview, previewValues]);
  // A new source snapshot or saved context immediately hides the earlier curve.
  const key = stableStringify({ well: data.well, request, data, seeds: context?.seeds, md: context?.jpump_md });
  const latestKey = useRef(key);
  latestKey.current = key;
  const currentId = handle?.key === key ? handle.id : null;
  const query = usePumpMatchJob(currentId);
  const result = show && handle?.key === key && query.data?.status === "done" &&
    query.data.result?.well === data.well && stableStringify(query.data.result.request) === stableStringify(request)
    ? query.data.result : null;
  const running = !!currentId && (!query.data || query.data.status === "running") && !isMissingJob(query.error);

  useEffect(() => {
    if (handle && handle.key !== key) {
      setHandle(null);
      setSelectedEra(null);
      setTestId(null);
    }
  }, [key, handle, cancelJob]);
  useEffect(() => () => { if (handle) cancelJob(handle.id); }, [handle, cancelJob]);
  useEffect(() => {
    latestKey.current = key;
    return () => { latestKey.current = "unmounted"; };
  }, [key]);

  const run = async () => {
    const requestedKey = key;
    setSelectedEra(null);
    setTestId(null);
    try {
      // Awaiting also handles a late start response after this view unmounts.
      const { job_id } = await start.mutateAsync({ well: data.well, request });
      if (latestKey.current !== requestedKey) cancelJob(job_id);
      else setHandle({ id: job_id, key: requestedKey });
    } catch { /* The mutation's error is rendered below. */ }
  };
  const row = result?.rows.find((r) => r.wt_uid === testId);
  const era = result?.eras.find((e) => e.installation_id === (row?.installation_id ?? selectedEra));
  const visibleEras = result?.eras.filter((e) => result.rows.some((r) => r.installation_id === e.installation_id)) ?? [];
  const solved = visibleEras.reduce((total, e) => total + (allTests ? e.replay_scores : e.prediction_scores).solved, 0);
  const attempted = visibleEras.reduce((total, e) => total + (allTests ? e.replay_scores : e.prediction_scores).attempted, 0);
  const unscored = result?.rows.filter((r) => r.status === "excluded" || r.status === "missing") ?? [];
  const reasons = new Map<string, number>();
  for (const r of unscored) {
    const reason = r.message ?? "No eligible model inputs for this test.";
    reasons.set(reason, (reasons.get(reason) ?? 0) + 1);
  }

  return <div className="space-y-3">
    <div className="flex flex-wrap items-center gap-4 text-xs text-slate-600">
      <label className="flex cursor-pointer items-center gap-2">
        <input className={CHECKBOX} type="checkbox" checked={show} onChange={(e) => setShow(e.target.checked)} />
        Show model match
      </label>
      {show && <>
        <label>Compare <select aria-label="History comparison" className={`${SELECT} ml-1`} value={mode}
          onChange={(e) => setMode(e.target.value as PumpMatchRequest["mode"])}>
          <option value="all_tests">Every test (well fit)</option>
          <option value="same_pump">Refit earlier tests: same pump</option>
          <option value="previous_pump">Refit earlier tests: next pump</option>
        </select></label>
        {allTests && <label>Well inputs <select aria-label="History well inputs" className={`${SELECT} ml-1`}
          value={useEdits ? "edited" : "saved"} onChange={(e) => setUseEdits(e.target.value === "edited")}>
          <option value="saved">Saved in database</option>
          <option value="edited">Current edits (preview)</option>
        </select></label>}
        <label>History <select aria-label="History window" className={`${SELECT} ml-1`} value={months}
          onChange={(e) => setMonths(Number(e.target.value) as PumpMatchRequest["months"])}>
          {[6, 12, 24, 60].map((n) => <option key={n} value={n}>{n} months</option>)}
        </select></label>
        {!allTests && <label>Training tests <select aria-label="Training tests" className={`${SELECT} ml-1`} value={training}
          onChange={(e) => setTraining(Number(e.target.value))}>
          {[3, 5, 10, 20].map((n) => <option key={n} value={n}>{n}</option>)}
        </select></label>}
        <Button size="sm" variant="secondary" busy={start.isPending} disabled={!contextReady || !!inputProblem || start.isPending || running} onClick={run}>
          {result ? "Run again" : "Run comparison"}
        </Button>
        {running && <Button size="sm" variant="secondary" disabled={cancel.isPending} onClick={() => currentId && cancelJob(currentId)}>Cancel</Button>}
      </>}
    </div>
    {show && <div className="space-y-1 text-xs text-slate-500" aria-live="polite">
      <p>{HYDRAULICS_LABELS[model]}. {allTests
        ? `Holds one ${preview ? "edited" : "saved"} oil IPR across every test and pump, using each test's measured WC, GOR and PF/WHP pressures.`
        : mode === "previous_pump"
          ? "Predicts the next installation using the preceding pump's last training tests."
          : "Fits the first training tests on each installation, then predicts later tests."}</p>
      <p>{preview ? "Uses the preview well inputs with saved well geometry" : "Uses saved well geometry, fluid properties and reservoir pressure"}; pump losses stay at clean reference.
        {allTests ? " The IPR stays fixed. This comparison uses known test composition." : " This mode refits inflow from earlier tests and freezes their WC/GOR for later predictions; it does not change the saved IPR."}</p>
      <p>History sets the chart window. {allTests ? "Every test with usable inputs is attempted, including early tests and short pump runs." : "Earlier tests can still train the model."} Gaps have an explanation below.</p>
      {allTests && <p>{preview
        ? "Preview uses the IPR and supported well inputs currently in the sidebar. Save well inputs above to use them in new optimization runs. Historical pumps, test WC/GOR and operating pressures still come from the recorded tests."
        : changed ? "Sidebar edits are not in this comparison. Select Current edits (preview) and run again to review them before saving."
          : "Edit the sidebar, select Current edits (preview), and run the comparison before saving."}</p>}
      {inputProblem && <p className="text-amber-700">{inputProblem}</p>}
      {!contextReady && <p>Loading saved well inputs...</p>}
      {props.gaugePreview && <p>Replay scores use recorded well-test BHP; the uploaded gauge preview is not part of this run.</p>}
      {running && <p>{query.data?.progress ?? "Starting history comparison..."}</p>}
      {query.data?.status === "cancelled" && <p>Comparison cancelled.</p>}
      {query.data?.status === "error" && <p className="text-amber-700">{query.data.error}</p>}
      {start.isError && <ErrorNote error={start.error} />}
      {query.isError && <ErrorNote error={query.error} />}
      {cancel.isError && <ErrorNote error={cancel.error} />}
      {result && <p>{allTests
        ? `${solved}/${result.rows.length} tests have predictions; ${attempted - solved} failed solves; `
        : `${solved}/${attempted} held-out predictions solved; `}{unscored.length} tests excluded or unsupported.
        {attempted === 0 && " No predictions are available in this window; inspect installation details below."}</p>}
      {result && unscored.length > 0 && <details open={unscored.length > result.rows.length / 2}>
        <summary className="cursor-pointer">Why {unscored.length} tests have no prediction</summary>
        <ul className="ml-5 list-disc space-y-1 py-1">
          {[...reasons].sort((a, b) => b[1] - a[1]).map(([reason, count]) => <li key={reason}>{count} tests: {reason}</li>)}
        </ul>
        {mode === "previous_pump" && visibleEras.some((e) => e.unavailable) && <p>
          To compare the well fit at every usable test, choose <strong>Every test (well fit)</strong> and run again.
        </p>}
      </details>}
    </div>}
    {result && <div className="flex flex-wrap items-center gap-4 text-xs text-slate-600">
      <label className="flex items-center gap-2"><input className={CHECKBOX} type="checkbox" checked={oilDetail} onChange={(e) => setOilDetail(e.target.checked)} />Oil detail</label>
      <label className="flex items-center gap-2"><input className={CHECKBOX} type="checkbox" checked={pfDetail} onChange={(e) => setPfDetail(e.target.checked)} />PF rate detail</label>
      <span>{allTests ? "Open circles: actual tests. Dashed lines/filled circles: model predictions." : "Dots: actual tests. Diamonds/dotted: fitted history. Circles/dashed: held-out predictions."}</span>
      {selectedEra && <button className="text-blue-700 underline" onClick={() => { setSelectedEra(null); setTestId(null); }}>Show all installations</button>}
    </div>}
    <HistoryStrip {...props} match={result} selectedEra={selectedEra} oilDetail={oilDetail} pfDetail={pfDetail} onSelectTest={setTestId} />
    {result && <>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <caption className="mb-2 text-left font-semibold text-slate-700">{allTests ? "Historical model results by installation - select a pump to inspect its inputs" : "Held-out results by installation - select a pump to inspect its fitting window"}</caption>
          <thead className="border-b text-slate-500"><tr>
            {["Pump / set date", allTests ? "Well fit" : "Training", "Solved", "BHP RMS / bias (psi)", "Oil MAE (BOPD)", "Oil error", "PF error", "Unscored"].map((h) => <th className="px-2 py-2 font-medium" key={h}>{h}</th>)}
          </tr></thead>
          <tbody>{visibleEras.map((e) => {
            const s = allTests ? e.replay_scores : e.prediction_scores;
            return <tr key={e.installation_id} className={`border-b border-slate-100 ${selectedEra === e.installation_id ? "bg-blue-50" : ""}`}>
              <td className="px-2 py-2"><button className="text-left text-blue-700 underline" onClick={() => { setSelectedEra(e.installation_id); setTestId(null); }}>{e.pump} / {fmtDate(e.date_set)}</button></td>
              <td className="px-2 py-2">{allTests ? preview ? "Edited inflow" : "Saved inflow" : `${e.training_count} tests`}</td><td className="px-2 py-2">{s.solved}/{s.attempted}</td>
              <td className="px-2 py-2">{fmtNum(s.bhp_rms)} / {fmtNum(s.bhp_bias)}</td>
              <td className="px-2 py-2">{fmtNum(s.oil_mae)}</td><td className="px-2 py-2">{errorPercent(s.oil_median_abs_pct)}</td>
              <td className="px-2 py-2">{errorPercent(s.pf_median_abs_pct)}</td>
              <td className="px-2 py-2">{result.rows.filter((r) => r.installation_id === e.installation_id && ["missing", "excluded"].includes(r.status)).length}</td>
            </tr>;
          })}</tbody>
        </table>
      </div>
      <p className="text-xs text-slate-500">Oil/PF percentages are median absolute errors on measured tests. Failed solves count in coverage. These scores do not establish a reliable sizing gain.</p>
      {era && <div className="space-y-1 rounded border border-slate-200 bg-slate-50 p-3 text-xs text-slate-600">
        <p className="font-semibold">{era.pump}, set {era.date_set} - {era.manufacturer ?? "Unknown manufacturer"}, {era.direction ?? "unknown circulation"}</p>
        {era.unavailable && <p className="text-amber-700">{era.unavailable}</p>}
        {!allTests && <p>Training: {fmtDate(era.training_start)} to {fmtDate(era.training_end)}, {era.training_count} tests,
          PF pressure span {fmtNum(era.training_ppf_span)} psi.</p>}
        {allTests ? <p>WC and GOR follow each test. The {preview ? "edited" : "saved"} oil IPR stays fixed across all installations.</p>
          : <p>Fixed training WC {fmtNum(era.input_wc === null ? null : 100 * era.input_wc, 1)}%, GOR {fmtNum(era.input_gor)} scf/STB.</p>}
        {era.prediction_config && <p>Fixed oil IPR: {fmtNum(Number(era.prediction_config.qwf) * (1 - Number(era.prediction_config.form_wc)))} BOPD at {fmtNum(Number(era.prediction_config.pwf))} psi;
          reservoir pressure {fmtNum(Number(era.prediction_config.res_pres))} psi.</p>}
        {era.fit_scores.attempted > 0 && <p>Fitted history: {era.fit_scores.solved}/{era.fit_scores.attempted} solved;
          BHP RMS {fmtNum(era.fit_scores.bhp_rms)} psi; oil MAE {fmtNum(era.fit_scores.oil_mae)} BOPD.</p>}
        {era.flags.map((f) => <p key={f} className="text-amber-700">{f}</p>)}
      </div>}
      <details className="text-xs text-slate-600" open={!!row}>
        <summary className="cursor-pointer">Inspect a test or model miss</summary>
        <select aria-label="Inspect history test" className={`${SELECT} my-2 max-w-full`} value={testId ?? ""} onChange={(e) => setTestId(e.target.value || null)}>
          <option value="">Select a test</option>
          {result.rows.filter((r) => !selectedEra || r.installation_id === selectedEra).map((r, i) => <option key={`${r.wt_uid}-${i}`} value={r.wt_uid}>{fmtDate(r.date)} - {r.status} - {r.wt_uid}</option>)}
        </select>
        {row && <div className="space-y-1 rounded bg-slate-50 p-3">
          <p>{fmtDate(row.date)}: {row.status}{row.message ? ` - ${row.message}` : ""}</p>
          <p>BHP actual / model: {fmtNum(row.bhp)} / {fmtNum(row.predicted_bhp)} psi.
            Oil: {fmtNum(row.oil)} / {fmtNum(row.predicted_oil)} BOPD. PF: {fmtNum(row.pf)} / {fmtNum(row.predicted_pf)} BPD.</p>
          <p>Formation liquid: {fmtNum(row.liquid)} / {fmtNum(row.predicted_liquid)} BLPD.</p>
          <p>Model inputs: WC {fmtNum(row.input_wc === null ? null : 100 * row.input_wc, 1)}%; GOR {fmtNum(row.input_gor)} scf/STB.</p>
          <p>Test-day PF pressure {fmtNum(row.ppf)} psi; WHP {fmtNum(row.pwh)} psi.
            {row.sonic !== null && ` Solver entry-limited flag: ${row.sonic ? "yes" : "no"}.`}</p>
        </div>}
      </details>
      <details className="text-xs text-slate-500"><summary className="cursor-pointer">Comparison assumptions and source</summary>
        <div className="mt-2 space-y-1">{result.notes.map((note) => <p key={note}>{note}</p>)}
          <p>Model {result.physics_model}; source {result.source}; captured {result.as_of}.</p>
          <p className="break-all">Replay {result.snapshot_id.slice(0, 16)}</p>
        </div>
      </details>
    </>}
  </div>;
}
