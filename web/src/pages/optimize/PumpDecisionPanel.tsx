import { isMissingJob } from "../../api/client";
/**
 * Cost of power fluid on S-Pad, and a single-well pump decision.
 *
 * S-Pad's boosters always run at 60 Hz, so the header sits where the pump
 * curve meets the wells' total PF demand. If one well takes another
 * 1,000 BPD, the header walks down the curve and every other well loses oil;
 * give 1,000 BPD back and they gain. This panel shows that cost in barrels
 * (and the equivalent marginal PF water cut), then prices every pump size for
 * one well - a JP replacement, a restart or a new well - with the header
 * resettled on the curve. Compute: server/services/pump_decision.py over
 * woffl/gui/pad_marginal.py. Read-only; nothing is saved.
 */

import clsx from "clsx";
import { Scale } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { useOptimizeJob, useStartPumpDecision, useWells } from "../../api/hooks";
import type { PumpDecisionCandidate, PumpDecisionResult, PumpDecisionStep } from "../../api/types";
import { Badge, Card, Spinner, WarnNote } from "../../components/ui";
import { fmtNum, fmtPct, fmtSigned } from "../../lib/format";
import { useOptimizeStore } from "../../state/optimize";
import { usePadOffline } from "./offline";
import { CancelJobButton, CancelledNote } from "./CancelJob";
import { PfCostChart } from "./PfCostChart";

const TH_CLS = "px-2 py-1.5 text-left font-semibold";
const TD_CLS = "px-2 py-1 tabular-nums";
// Mirrors server/schemas.py PumpDecisionRequest (and the pad run defaults).
const NOZZLES = ["9", "10", "11", "12", "13", "14", "15"];
const THROATS = ["A", "B", "C", "D"];

function Verdict({ row }: { row: PumpDecisionCandidate }) {
  if (row.beats_marginal === null) return <span className="text-slate-400">-</span>;
  if (row.beats_marginal && row.d_pf <= 0)
    return <Badge tone="good" title="At least as much oil for no more PF">no extra PF</Badge>;
  return row.beats_marginal
    ? <Badge tone="good" title="This pump's extra oil comes at a PF water cut below the pad marginal">below marginal</Badge>
    : <Badge tone="fair" title="This pump's extra PF costs the other wells more oil than it adds">above marginal</Badge>;
}

function StepTile({ label, step, dq, padWide = false }: { label: string; step: PumpDecisionStep | null; dq: number; padWide?: boolean }) {
  if (!step) return <div className="rounded-md bg-white p-2 ring-1 ring-slate-200 text-sm text-slate-500">{label}: curve has no solution</div>;
  const adding = label.startsWith("+");
  return (
    <div className="rounded-md bg-white p-2 ring-1 ring-slate-200">
      <p className="text-xs font-medium text-slate-500">{label}</p>
      <p className={clsx("text-xl font-semibold tabular-nums", adding ? "text-rose-700" : "text-emerald-700")}>
        {fmtSigned(step.others_d_oil, 1)} BOPD
      </p>
      <p className="text-xs text-slate-600">
        {padWide ? "across all wells" : "to the other wells"}; header {fmtSigned(step.d_header_psi, 0)} psi. Their PF moves {fmtSigned(step.others_d_pf, 0)} BPD,
        so the station flow changes {fmtSigned((adding ? dq : -dq) + step.others_d_pf, 0)} BPD.
      </p>
    </div>
  );
}

function Results({ result, dq }: { result: PumpDecisionResult; dq: number }) {
  const s = result.sensitivity;
  const best = result.candidates.find((c) => c.net_oil !== null) ?? null;
  const [showWells, setShowWells] = useState(false);
  const impact = useMemo(() => {
    const add = new Map((s.add?.wells ?? []).map((w) => [w.well, w]));
    const rem = new Map((s.remove?.wells ?? []).map((w) => [w.well, w]));
    return result.wells_today.filter((w) => w.well !== result.target)
      .map((w) => ({ ...w, add: add.get(w.well), rem: rem.get(w.well) }))
      .sort((a, b) => (a.add?.d_oil ?? 0) - (b.add?.d_oil ?? 0));
  }, [s, result]);
  const baseLabel = result.baseline
    ? `today's ${result.baseline.pump} (saved fit)`
    : "an empty slot (no pump today)";

  return (
    <div className="space-y-3">
      <div className="space-y-2 rounded-lg bg-slate-50 p-3 ring-1 ring-slate-200">
        <p className="text-sm text-slate-600">
          Cost of PF on {result.pad}-Pad ({result.coupling === "free_pressure"
            ? `booster held at a ${fmtNum(result.setpoint_psi ?? null)} psi setpoint until its frontier cannot carry the flow`
            : `${result.n_pumps} boosters at 60 Hz`}). Modeled today: header{" "}
          <strong>{fmtNum(result.header_psi)} psi</strong>, {fmtNum(result.model_pf_bpd)} BPD PF, {fmtNum(result.model_oil_bopd)} BOPD
          {result.test_pf_bpd > 0 && <> (recent tests: {fmtNum(result.test_pf_bpd)} BPD PF)</>}.
        </p>
        {result.coupling === "free_pressure" && (
          <p className="text-sm text-slate-700">
            {(result.free_headroom_bpd ?? 0) > 0
              ? <>The booster holds {fmtNum(result.setpoint_psi ?? null)} psi with <strong>{fmtNum(result.free_headroom_bpd ?? null)} BPD</strong> of free headroom: extra PF inside it costs the other wells nothing. Past it the header falls along the frontier.</>
              : <>The booster is <strong>at its frontier</strong> ({fmtNum(result.header_psi)} psi, below the {fmtNum(result.setpoint_psi ?? null)} psi setpoint), so every extra barrel of PF lowers the header for all wells.</>}
          </p>
        )}
        <div className="grid gap-2 md:grid-cols-2">
          <StepTile label={`+${fmtNum(dq)} BPD PF drawn by ${result.target ?? "any well"}`} step={s.add} dq={dq} padWide={result.target === null} />
          <StepTile label={`-${fmtNum(dq)} BPD PF given back by ${result.target ?? "any well"}`} step={s.remove} dq={dq} padWide={result.target === null} />
        </div>
        <PfCostChart result={result} />
        {s.pfwc !== null && (
          <p className="text-sm text-slate-700">
            Marginal PF water cut <strong>{fmtPct(s.pfwc, 1)}</strong>: extra oil from a pump in {result.target ?? "any well"} must come at a PF
            water cut below {fmtPct(s.pfwc, 1)}, i.e. more than <strong>{fmtNum((s.lambda ?? 0) * 1000, 1)} BOPD per 1,000 BPD</strong> of
            extra PF, or the pad loses oil overall.
          </p>
        )}
        <button type="button" onClick={() => setShowWells((v) => !v)} className="text-xs text-blue-700 hover:underline">
          {showWells ? "Hide" : "Show"} the impact on each {result.target === null ? "" : "other "}well ({impact.length})
        </button>
        {showWells && (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[36rem] border-collapse text-[13px]">
              <thead>
                <tr className="border-b border-slate-200 text-xs text-slate-500">
                  <th className={TH_CLS}>Well</th>
                  <th className={TH_CLS} title="Modeled at today's header, BOPD / BPD">Oil / PF today</th>
                  <th className={TH_CLS}>Oil at +{fmtNum(dq)} BPD</th>
                  <th className={TH_CLS}>Oil at -{fmtNum(dq)} BPD</th>
                  <th className={TH_CLS} title="Median recent test, BOPD / BPD">Test oil / PF</th>
                </tr>
              </thead>
              <tbody>
                {impact.map((w) => (
                  <tr key={w.well} className="border-b border-slate-100 text-slate-700">
                    <td className={clsx(TD_CLS, "font-medium")}>{w.well}</td>
                    <td className={TD_CLS}>{fmtNum(w.oil)} / {fmtNum(w.pf)}</td>
                    <td className={clsx(TD_CLS, "text-rose-700")}>{fmtSigned(w.add?.d_oil, 1)}</td>
                    <td className={clsx(TD_CLS, "text-emerald-700")}>{fmtSigned(w.rem?.d_oil, 1)}</td>
                    <td className={clsx(TD_CLS, "text-slate-500")}>{fmtNum(w.test_oil)} / {fmtNum(w.test_pf)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {result.target !== null && (<>
      {best && (
        <p className="text-sm text-slate-700">
          Best modeled choice for {result.target}: <strong>{best.pump}</strong>
          {best.pump_state === "installed" ? " (keep the installed pump)" : ""}, net {fmtSigned(best.net_oil, 0)} BOPD to the pad
          versus {baseLabel}, after the other wells' {fmtSigned(best.others_d_oil, 0)} BOPD.
        </p>
      )}

      <div className="overflow-x-auto">
        <table className="w-full min-w-[60rem] border-collapse text-[13px]">
          <thead>
            <tr className="border-b border-slate-200 text-xs text-slate-500">
              <th className={TH_CLS}>Pump</th>
              <th className={TH_CLS} title="The target's oil / PF at the resettled header">Oil / PF</th>
              <th className={TH_CLS} title="Header after the pad resettles on the curve (change from today)">Header</th>
              <th className={TH_CLS} title={`The target's change versus ${baseLabel}`}>Extra oil / PF</th>
              <th className={TH_CLS} title="Extra PF / (extra PF + extra oil)">Incremental PFWC</th>
              <th className={TH_CLS}>vs marginal</th>
              <th className={TH_CLS} title="Every other well's oil change at the new header">Other wells</th>
              <th className={TH_CLS} title="Target change + other wells' change">Net pad oil</th>
            </tr>
          </thead>
          <tbody>
            {result.candidates.map((c) => (
              <tr key={`${c.pump}-${c.pump_state}`}
                className={clsx("border-b border-slate-100 text-slate-700", best && c === best && "bg-emerald-50")}>
                <td className={clsx(TD_CLS, "font-medium")}>
                  {c.pump}
                  <span className="ml-1 text-[11px] font-normal text-slate-400">
                    {c.pump_state === "installed" ? "installed (fit)" : "clean"}
                  </span>
                </td>
                <td className={TD_CLS}>{fmtNum(c.oil)} / {fmtNum(c.pf)}</td>
                <td className={TD_CLS} title={c.extrapolated ? "Beyond the modeled header range; rates held at the nearest modeled point" : undefined}>
                  {fmtNum(c.header_psi)} ({fmtSigned(c.d_header_psi)}){c.extrapolated ? " *" : ""}
                </td>
                <td className={TD_CLS}>{fmtSigned(c.d_oil)} / {fmtSigned(c.d_pf)}</td>
                <td className={TD_CLS}>{fmtPct(c.inc_pfwc, 1)}</td>
                <td className={TD_CLS}><Verdict row={c} /></td>
                <td className={clsx(TD_CLS, "text-slate-600")}>{fmtSigned(c.others_d_oil, 1)}</td>
                <td className={clsx(TD_CLS, "font-medium", (c.net_oil ?? 0) > 0 ? "text-emerald-700" : "text-slate-600")}>
                  {c.net_oil === null ? "-" : fmtSigned(c.net_oil, 1)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      </>)}

      {result.notes.length > 0 && (
        <ul className="space-y-0.5 text-xs text-slate-500">
          {result.notes.map((n) => <li key={n}>{n}</li>)}
        </ul>
      )}
      <p className="text-xs text-slate-400">
        Every other well stays on its installed pump with its saved fit; the target's replacements use clean reference
        coefficients. Model rates, not test-scaled: check the pad's match health before acting. * = header outside the
        modeled range.
      </p>
    </div>
  );
}

export function PumpDecisionPanel({ pad }: { pad: "S" | "I" }) {
  const wells = useWells();
  const futureByPad = useOptimizeStore((s) => s.future);
  const jobKey = `pump_decision:${pad}`;
  const jobId = useOptimizeStore((s) => s.lastJob[jobKey] ?? null);
  const setLastJob = useOptimizeStore((s) => s.setLastJob);
  const start = useStartPumpDecision();
  const job = useOptimizeJob(jobId);

  const padList = useMemo(() => [pad], [pad]);
  const { offline: offlineSet, ready: offlineReady, failed: offlineFailed } = usePadOffline(padList);
  const future = useMemo(() => (futureByPad[pad] ?? []).map((f) => ({ ...f, pad })), [futureByPad, pad]);
  const options = useMemo(() => {
    const names = (wells.data?.wells ?? []).filter((w) => w.pad === pad).map((w) => w.name).sort();
    return [...names.map((n) => ({ name: n, tag: offlineSet.has(n) ? "offline" : "" })),
            ...future.map((f) => ({ name: f.name, tag: "future" }))];
  }, [wells.data, pad, offlineSet, future]);

  const [target, setTarget] = useState("");
  // Pad-wide: no well is sized; the step is extra draw anywhere on the pad.
  const [allWells, setAllWells] = useState(false);
  const [nPumps, setNPumps] = useState<number>(3);
  const [dq, setDq] = useState(1000);
  const [setpoint, setSetpoint] = useState(3500);
  const chosen = options.some((o) => o.name === target) ? target : "";

  useEffect(() => {
    if (jobId && isMissingJob(job.error)) setLastJob(jobKey, null);
  }, [jobId, job.error, jobKey, setLastJob]);

  const running = job.data?.status === "running" || start.isPending;
  const result = job.data?.status === "done" && job.data.kind === "pump_decision"
    ? (job.data.result as PumpDecisionResult | null) : null;
  const dqValid = Number.isFinite(dq) && dq > 0 && dq <= 10000 &&
    (pad === "S" || (Number.isFinite(setpoint) && setpoint >= 1000 && setpoint <= 5000));

  const run = () => {
    if ((!chosen && !allWells) || !dqValid) return;
    start.mutate({
      pad, target: allWells ? null : chosen, offline: [...offlineSet].sort(), future,
      n_pumps: pad === "S" ? nPumps : null, nozzles: NOZZLES, throats: THROATS, delta_pf_bpd: dq,
      setpoint_psi: pad === "S" ? null : setpoint,
    }, { onSuccess: (r) => setLastJob(jobKey, r.job_id) });
  };

  return (
    <div className="space-y-2">
      <h2 className="text-sm font-semibold tracking-tight text-slate-700">{pad}-Pad cost of PF and pump decision</h2>
      <Card className="space-y-3">
        <div className="flex flex-wrap items-center gap-3">
          <label className="flex items-center gap-1.5 text-xs text-slate-600"
            title="Show the cost of adding or giving back PF across every well on the pad, without sizing a pump">
            <input type="checkbox" checked={allWells} onChange={(e) => setAllWells(e.target.checked)} className="h-4 w-4 accent-blue-600" />
            All wells
          </label>
          <select aria-label="Well to size" value={chosen} onChange={(e) => setTarget(e.target.value)} disabled={allWells}
            className="h-8 rounded-md border border-slate-300 bg-white px-2 text-sm text-slate-800 disabled:opacity-50">
            <option value="">Well to size...</option>
            {options.map((o) => <option key={o.name} value={o.name}>{o.name}{o.tag ? ` (${o.tag})` : ""}</option>)}
          </select>
          {pad === "S" ? (
            <label className="flex items-center gap-1 text-xs text-slate-600">
              Boosters online
              <select aria-label="Booster pumps online" value={nPumps} onChange={(e) => setNPumps(Number(e.target.value))}
                className="h-7 rounded border border-slate-300 bg-white text-xs">
                <option value={3}>3</option>
                <option value={2}>2</option>
              </select>
            </label>
          ) : (
            <label className="flex items-center gap-1 text-xs text-slate-600"
              title="The header setpoint the booster holds while its frontier allows (defaults to the 3,500 psi operational cap)">
              Setpoint
              <input type="number" aria-label="Header setpoint, psi" min={1000} max={5000} step={50} value={setpoint}
                onChange={(e) => setSetpoint(Number(e.target.value))}
                className="h-7 w-20 rounded border border-slate-300 bg-white px-1 text-xs tabular-nums" />
              psi
            </label>
          )}
          <label className="flex items-center gap-1 text-xs text-slate-600">
            PF change +/-
            <input type="number" aria-label="PF change, BPD" min={100} max={10000} step={100} value={dq}
              onChange={(e) => setDq(Number(e.target.value))}
              className="h-7 w-20 rounded border border-slate-300 bg-white px-1 text-xs tabular-nums" />
            BPD
          </label>
          <button type="button" disabled={running || (!chosen && !allWells) || !dqValid || (!offlineReady && !offlineFailed)} onClick={run}
            title={!offlineReady && !offlineFailed ? "Loading the downtime log so shut-in wells are excluded" : undefined}
            className="flex items-center gap-1.5 rounded-md bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50">
            <Scale className="h-3.5 w-3.5" />
            {running ? "Pricing..." : allWells ? "Price PF across the pad" : "Price PF and pump sizes"}
          </button>
          <CancelJobButton jobId={jobId} running={job.data?.status === "running"} />
        </div>
        <p className="text-xs text-slate-500">
          {pad === "S"
            ? "The boosters run at 60 Hz, so more PF draw lowers the header for every well."
            : "The booster holds its setpoint until its frontier cannot carry the flow; past that, more PF draw lowers the header for every well."}{" "}
          Shows what adding or giving back PF
          costs the other wells in barrels, then every pump size for the chosen well with the header resettled. Tick All
          wells to see the cost across the whole pad without sizing a pump. Offline ticks
          and future wells come from the readiness board.
        </p>
        {running && job.data?.progress && <p className="text-xs text-slate-500">{job.data.progress} ({fmtNum(job.data.seconds)}s)</p>}
        {running && !job.data?.progress && <Spinner label="Starting" />}
        {job.data?.status === "error" && <WarnNote>Pricing failed: {job.data.error}</WarnNote>}
        <CancelledNote job={job.data} />
        {offlineFailed && <WarnNote>The downtime log did not load, so shut-in wells are not excluded automatically. Tick them offline on the readiness board first.</WarnNote>}
        {start.isError && <WarnNote>Could not start: {start.error.message}</WarnNote>}
        {result !== null && <Results result={result} dq={result.sensitivity.d_q} />}
      </Card>
    </div>
  );
}
