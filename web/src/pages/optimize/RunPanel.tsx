/**
 * One optimization run tab (S / I / M pad or CFP) - constraints form, Run
 * button, live job progress, and results. The engines are the Streamlit
 * pad pages' own compute cores running server-side over SAVED fits; the
 * board tab's offline flags and future wells feed straight into the run.
 *
 * Job ids persist in the optimize store, so switching tabs (or reloading)
 * re-attaches to a still-running or recently finished job; an expired job
 * (server restart) clears itself silently.
 */

import clsx from "clsx";
import { Play } from "lucide-react";
import { useEffect, useMemo, useState, type ReactNode } from "react";

import { useOptimizeJob, useStartOptimizeRun, useWells } from "../../api/hooks";
import type {
  CfpMoveRow,
  CfpRunResult,
  OptimizeRunRequest,
  PadRunResult,
  PadRunRow,
} from "../../api/types";
import { Card, Spinner, WarnNote } from "../../components/ui";
import { fmtNum } from "../../lib/format";
import { DEFAULT_POPS_PADS } from "../../state/wellSort";
import { useOptimizeStore } from "../../state/optimize";

import { CfpResultCharts } from "./CfpCharts";
import { usePadOffline } from "./offline";
import { PadCharts } from "./PadCharts";

const NOZZLE_OPTIONS = ["8", "9", "10", "11", "12", "13", "14", "15"];
const THROAT_OPTIONS = ["X", "A", "B", "C", "D", "E"];
const CFP_PADS = ["B", "G", "C", "J"];

const INPUT_CLS =
  "h-8 w-24 rounded-md border border-slate-300 bg-white px-2 text-sm tabular-nums " +
  "text-slate-800 outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-200";

/** meta values arrive JSON-flattened; narrow numerics defensively. */
function metaNum(meta: Record<string, unknown>, key: string): number | null {
  const v = meta[key];
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function ChipToggle({
  options,
  selected,
  onChange,
}: {
  options: string[];
  selected: string[];
  onChange: (next: string[]) => void;
}) {
  return (
    <div className="flex flex-wrap gap-1">
      {options.map((o) => {
        const on = selected.includes(o);
        return (
          <button
            key={o}
            type="button"
            onClick={() => onChange(on ? selected.filter((s) => s !== o) : [...selected, o])}
            className={clsx(
              "rounded px-1.5 py-0.5 text-xs font-medium transition-colors",
              on ? "bg-blue-600 text-white" : "bg-white text-slate-500 ring-1 ring-slate-200 hover:bg-slate-50",
            )}
          >
            {o}
          </button>
        );
      })}
    </div>
  );
}

function Metric({ label, value, title }: { label: string; value: string; title?: string }) {
  return (
    <div title={title} className={title ? "cursor-help" : undefined}>
      <p className="text-[10px] font-semibold uppercase tracking-wide text-slate-400">{label}</p>
      <p className="text-lg font-semibold tabular-nums text-slate-800">{value}</p>
    </div>
  );
}

const TH_CLS = "px-2 py-1.5 text-right font-semibold";
const TD_CLS = "px-2 py-1 text-right tabular-nums";

/** Which inflow curve the well's pump was picked against. A reviewed save is
 *  the point of the whole save-fits workflow; a weak auto-fit or generic
 *  defaults mean the recommended pump is only as good as a sketch. */
function FitSource({ row }: { row: PadRunRow }) {
  const r2 = row.ipr_r2;
  // R2 <= 0 means the Vogel curve tracks the tests WORSE than a flat line -
  // the pump picked against it is noise, so it reads as loud as defaults.
  const broken = r2 !== null && r2 <= 0;
  const weak = r2 !== null && r2 > 0 && r2 < 0.5;
  const [label, tone, hint] =
    row.ipr_source === "saved"
      ? ["saved", "text-emerald-700", "Engineer-reviewed IPR from prop_hist."]
      : row.ipr_source === "vogel"
        ? [
            `auto R2 ${r2 === null ? "-" : r2.toFixed(2)}`,
            broken ? "text-rose-700" : weak ? "text-amber-700" : "text-slate-500",
            broken
              ? "The Vogel fit is worse than a flat line - this well's pump pick is noise until someone reviews it."
              : weak
                ? "Automatic Vogel fit, and a weak one - review this well before trusting its pump."
                : "Automatic Vogel fit over recent tests; not reviewed.",
          ]
        : row.ipr_source === "single_test"
          ? ["1 test", "text-amber-700", "Seeded from one well test - no fit, so no curvature."]
          : ["defaults", "text-rose-700", "No usable tests: generic IPR (qwf 750 / pwf 500 / ResP 1700). The pump pick is a guess."];
  return (
    <span className={clsx("text-[11px] font-medium", tone)} title={hint}>
      {label}
      {!row.has_friction && (
        <span className="ml-1 text-slate-400" title="Library friction coefficients - no BHP calibration saved.">
          nofric
        </span>
      )}
    </span>
  );
}

function PadResults({ result }: { result: PadRunResult }) {
  const meta = result.meta;
  const swaps = Array.isArray(meta.parsimony_swaps) ? meta.parsimony_swaps.length : 0;
  const totalTestOil = result.rows.reduce((a, r) => a + (r.test_oil ?? 0), 0);
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
        <Metric label="Header" value={`${fmtNum(metaNum(meta, "header_psi"))} psi`} />
        <Metric label="Total PF" value={`${fmtNum(metaNum(meta, "total_pf_bpd"))} BPD`} />
        <Metric label="Optimized oil" value={`${fmtNum(metaNum(meta, "total_oil_bopd"))} BOPD`} />
        <Metric label="Current test oil" value={`${fmtNum(totalTestOil)} BOPD`} />
        <Metric
          label="Marginal WC gate"
          value={
            metaNum(meta, "marginal_wc_used") !== null
              ? `${(metaNum(meta, "marginal_wc_used")! * 100).toFixed(1)}%`
              : "-"
          }
        />
      </div>
      {meta.converged === false && (
        <WarnNote>Plant coupling did not converge - treat the header and totals as approximate.</WarnNote>
      )}
      {meta.over_capacity === true && <WarnNote>Plan exceeds plant capacity.</WarnNote>}

      <Card padded={false} className="overflow-x-auto">
        <table className="w-full border-collapse text-[13px]">
          <thead>
            <tr className="border-b border-slate-200 bg-slate-50 text-slate-600">
              <th className="px-2 py-1.5 text-left font-semibold">Well</th>
              <th
                className="px-2 py-1.5 text-left font-semibold"
                title="The inflow curve this well's pump was chosen against. saved = an engineer-reviewed fit; auto = a Vogel fit over recent tests, with its R2; 1 test = a single test; defaults = no tests, so generic values were used and the pump pick is a guess."
              >
                Fit
              </th>
              <th className="px-2 py-1.5 text-left font-semibold">Current</th>
              <th className={TH_CLS}>Test oil</th>
              <th className={TH_CLS}>Test PF</th>
              <th className="px-2 py-1.5 text-left font-semibold">Plan</th>
              <th className={TH_CLS}>Oil</th>
              <th className={TH_CLS}>PF</th>
              <th className={TH_CLS}>Suction</th>
              <th className={TH_CLS}>Marginal</th>
            </tr>
          </thead>
          <tbody>
            {result.rows.map((r) => {
              const change = r.pump !== null && r.current_pump !== null && r.pump !== r.current_pump;
              return (
                <tr key={r.well} className="border-b border-slate-100 last:border-b-0">
                  <td className="px-2 py-1 text-left font-medium text-slate-700">{r.well}</td>
                  <td className="px-2 py-1 text-left">
                    <FitSource row={r} />
                  </td>
                  <td className="px-2 py-1 text-left text-slate-600">{r.current_pump ?? "-"}</td>
                  <td className={clsx(TD_CLS, "text-slate-600")}>{fmtNum(r.test_oil)}</td>
                  <td className={clsx(TD_CLS, "text-slate-500")}>{fmtNum(r.test_pf)}</td>
                  <td className="px-2 py-1 text-left">
                    {r.pump === null ? (
                      <span className="font-medium text-amber-700">SHUT IN</span>
                    ) : (
                      <span className={clsx("font-medium", change ? "text-blue-700" : "text-slate-700")}>
                        {r.pump}
                        {r.sonic && <span title="sonic throat"> *</span>}
                      </span>
                    )}
                  </td>
                  <td className={clsx(TD_CLS, "text-slate-700")}>{fmtNum(r.oil)}</td>
                  <td className={clsx(TD_CLS, "text-slate-600")}>{fmtNum(r.pf)}</td>
                  <td className={clsx(TD_CLS, "text-slate-500")}>{fmtNum(r.suction)}</td>
                  <td className={clsx(TD_CLS, "text-slate-500")}>{fmtNum(r.marginal_oil, 2)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </Card>

      {swaps > 0 && (
        <p className="text-xs text-slate-500">
          Parsimony: {swaps} well{swaps > 1 ? "s" : ""} swapped down to a smaller pump for
          near-zero oil given up.
        </p>
      )}
      {result.notes.length > 0 && (
        <div className="space-y-0.5 text-xs text-slate-500">
          {result.notes.map((n) => (
            <p key={n}>{n}</p>
          ))}
        </div>
      )}
    </div>
  );
}

const MOVE_LABELS: Record<string, string> = {
  resize: "Resize",
  shut_in: "Shut in",
  bring_online: "Bring online",
};

/** Format a signed water delta as engineer-speak: SI frees PW, BOL adds it. */
function pwDelta(v: number | null): string {
  if (v === null) return "-";
  if (v < 0) return `frees ${fmtNum(-v)}`;
  if (v > 0) return `adds ${fmtNum(v)}`;
  return "0";
}

function CfpResults({ result }: { result: CfpRunResult }) {
  const s = result.summary;
  const planActions = s.plan?.actions ?? [];
  const singles = s.singles.filter((m) => m.fleet_oil_delta > 0).slice(0, 12);
  // The on/off ladder: every priced shut-in and bring-online, best first -
  // the "which wells free PW for jet pumps, and at what oil cost" view.
  // Bring-online singles enumerate every candidate pump size; keep only the
  // best option per well so the ladder reads one decision per row.
  const bestOnOff = new Map<string, CfpMoveRow>();
  for (const m of s.singles) {
    if (m.type !== "shut_in" && m.type !== "bring_online") continue;
    const k = `${m.type}-${m.well}`;
    const prev = bestOnOff.get(k);
    if (!prev || m.fleet_oil_delta > prev.fleet_oil_delta) bestOnOff.set(k, m);
  }
  const onOffMoves = [...bestOnOff.values()].sort((a, b) => b.fleet_oil_delta - a.fleet_oil_delta);
  const inPlan = new Set(planActions.map((a) => `${a.type}-${a.well}`));
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
        <Metric label="Discharge today" value={`${fmtNum(s.today.pressure)} psi`} />
        <Metric
          label={`Modeled oil (${result.n_wells} run wells)`}
          value={`${fmtNum(s.today.oil)} BOPD`}
          title="Total MODELED oil across the wells in this run, at today's discharge pressure - the optimizer's baseline, not field production. Plan gains are measured against this number."
        />
        <Metric label="Plant water" value={`${fmtNum(s.today.water)} BWPD`} />
        <Metric
          label="Shadow price"
          value={s.lambda_bopd_per_psi !== null ? `${fmtNum(s.lambda_bopd_per_psi, 2)} BOPD/psi` : "-"}
        />
        <Metric
          label="Best plan gain"
          value={s.plan_gain !== null ? `+${fmtNum(s.plan_gain)} BOPD` : "-"}
        />
      </div>

      <CfpResultCharts result={result} />

      {planActions.length > 0 && (
        <Card padded={false} className="overflow-x-auto">
          <p className="px-3 pt-2 text-xs font-semibold text-slate-600">Best plan</p>
          <table className="w-full border-collapse text-[13px]">
            <thead>
              <tr className="border-b border-slate-200 text-slate-600">
                <th className="px-2 py-1.5 text-left font-semibold">Action</th>
                <th className="px-2 py-1.5 text-left font-semibold">Well</th>
                <th className="px-2 py-1.5 text-left font-semibold">From</th>
                <th className="px-2 py-1.5 text-left font-semibold">To</th>
                <th className={TH_CLS}>Own oil</th>
                <th className={TH_CLS} title="the well's own water change, BWPD - negative frees PW">PW (BWPD)</th>
              </tr>
            </thead>
            <tbody>
              {planActions.map((a) => (
                <tr key={`${a.well}-${a.to ?? ""}`} className="border-b border-slate-100 last:border-b-0">
                  <td className="px-2 py-1 text-left font-medium text-slate-700">
                    {MOVE_LABELS[a.type] ?? a.type}
                  </td>
                  <td className="px-2 py-1 text-left text-slate-700">{a.well}</td>
                  <td className="px-2 py-1 text-left text-slate-500">{a.from ?? "-"}</td>
                  <td className="px-2 py-1 text-left text-slate-700">{a.to ?? "-"}</td>
                  <td className={TD_CLS}>{fmtNum(a.own_oil_delta)}</td>
                  <td className={TD_CLS}>{pwDelta(a.own_water_delta)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}

      {onOffMoves.length > 0 && (
        <Card padded={false} className="overflow-x-auto">
          <p className="px-3 pt-2 text-xs font-semibold text-slate-600">
            Shut in / bring online ladder
            <span className="ml-2 font-normal text-slate-400">
              every on/off move priced - PW freed vs oil cost, net of the discharge-pressure change on the rest of the fleet
            </span>
          </p>
          <table className="w-full border-collapse text-[13px]">
            <thead>
              <tr className="border-b border-slate-200 text-slate-600">
                <th className="px-2 py-1.5 text-left font-semibold">Move</th>
                <th className="px-2 py-1.5 text-left font-semibold">Well</th>
                <th className="px-2 py-1.5 text-left font-semibold" title="the pump involved: shut in FROM this pump / brought online TO the best candidate">Pump</th>
                <th className={TH_CLS} title="the well's own oil change, BOPD">Own oil</th>
                <th className={TH_CLS} title="the well's own water change, BWPD - shutting in frees PW for jet pumps">PW (BWPD)</th>
                <th className={TH_CLS} title="total fleet oil change: own oil + what the discharge-pressure change does to every other well">Net oil</th>
                <th className={TH_CLS}>Discharge after</th>
                <th className="px-2 py-1.5 text-left font-semibold" title="part of the best plan?">Plan</th>
              </tr>
            </thead>
            <tbody>
              {onOffMoves.map((m) => (
                <tr key={`${m.type}-${m.well}`} className="border-b border-slate-100 last:border-b-0">
                  <td className="px-2 py-1 text-left font-medium text-slate-700">{MOVE_LABELS[m.type]}</td>
                  <td className="px-2 py-1 text-left text-slate-700">{m.well}</td>
                  <td className="px-2 py-1 text-left text-slate-500">
                    {m.type === "shut_in" ? (m.from ?? "-") : (m.to ?? "-")}
                  </td>
                  <td className={TD_CLS}>{fmtNum(m.own_oil_delta)}</td>
                  <td className={TD_CLS}>{pwDelta(m.own_water_delta)}</td>
                  <td className={clsx(TD_CLS, m.fleet_oil_delta > 0 ? "text-emerald-700" : "text-slate-600")}>
                    {m.fleet_oil_delta > 0 ? "+" : ""}
                    {fmtNum(m.fleet_oil_delta)}
                  </td>
                  <td className={TD_CLS}>
                    {fmtNum(m.pressure_after)}
                    {m.at_trip && <span title="at the 2,900 psi trip"> !</span>}
                  </td>
                  <td className="px-2 py-1 text-left">
                    {inPlan.has(`${m.type}-${m.well}`) ? (
                      <span className="font-semibold text-emerald-700">yes</span>
                    ) : (
                      <span className="text-slate-400">-</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}

      {singles.length > 0 && (
        <Card padded={false} className="overflow-x-auto">
          <p className="px-3 pt-2 text-xs font-semibold text-slate-600">
            Every knob, priced (top positive moves)
          </p>
          <table className="w-full border-collapse text-[13px]">
            <thead>
              <tr className="border-b border-slate-200 text-slate-600">
                <th className="px-2 py-1.5 text-left font-semibold">Move</th>
                <th className="px-2 py-1.5 text-left font-semibold">Well</th>
                <th className={TH_CLS} title="total fleet oil change: own oil + what the discharge-pressure change does to every other well">Net oil</th>
                <th className={TH_CLS}>Own oil</th>
                <th className={TH_CLS}>Pressure after</th>
              </tr>
            </thead>
            <tbody>
              {singles.map((m) => (
                <tr key={`${m.type}-${m.well}-${m.to ?? ""}`} className="border-b border-slate-100 last:border-b-0">
                  <td className="px-2 py-1 text-left text-slate-600">
                    {MOVE_LABELS[m.type]}
                    {m.type === "resize" && ` ${m.from ?? ""} to ${m.to ?? ""}`}
                  </td>
                  <td className="px-2 py-1 text-left font-medium text-slate-700">{m.well}</td>
                  <td className={clsx(TD_CLS, m.fleet_oil_delta > 0 ? "text-emerald-700" : "text-slate-600")}>
                    {m.fleet_oil_delta > 0 ? "+" : ""}
                    {fmtNum(m.fleet_oil_delta)}
                  </td>
                  <td className={TD_CLS}>{fmtNum(m.own_oil_delta)}</td>
                  <td className={TD_CLS}>
                    {fmtNum(m.pressure_after)}
                    {m.at_trip && <span title="at the 2,900 psi trip"> !</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}

      {result.notes.length > 0 && (
        <div className="space-y-0.5 text-xs text-slate-500">
          {result.notes.map((n) => (
            <p key={n}>{n}</p>
          ))}
        </div>
      )}
    </div>
  );
}

/** ``aside`` rides beside the pump curves (the pad readiness board). Pad
 *  runs only - CFP has no per-pad board and keeps its own full-width charts. */
export function RunPanel({
  kind,
  pad,
  aside,
}: {
  kind: "pad" | "cfp";
  pad: "S" | "I" | "M" | null;
  aside?: ReactNode;
}) {
  const runKey = kind === "cfp" ? "CFP" : (pad as string);
  const wells = useWells();
  const futureByPad = useOptimizeStore((s) => s.future);
  const lastJob = useOptimizeStore((s) => s.lastJob);
  const setLastJob = useOptimizeStore((s) => s.setLastJob);

  const [nozzles, setNozzles] = useState(["9", "10", "11", "12", "13", "14"]);
  const [throats, setThroats] = useState(["A", "B", "C", "D"]);
  const [method, setMethod] = useState<"milp" | "mckp">("milp");
  const [autoWc, setAutoWc] = useState(true);
  const [manualWc, setManualWc] = useState(0.95);
  const [parsimony, setParsimony] = useState(20);
  const [p0, setP0] = useState(2792);
  const [slope, setSlope] = useState(13.69);
  const [cPadPf, setCPadPf] = useState(3400);
  const [cfpPads, setCfpPads] = useState<string[]>([...CFP_PADS]);
  // Selectable CFP pads: the canonical four plus any non-POPs pad found in
  // the well universe (L, R, ...). Non-POPs water rides the CFP machines,
  // so those pads may legitimately join; PF for pads beyond B/G/J is
  // modeled at the C-Pad booster knob (the server notes this on the run).
  // POPs pads separate water on-pad and are never offered.
  const cfpPadOptions = useMemo(() => {
    const extras = [
      ...new Set(
        (wells.data?.wells ?? [])
          .map((w) => w.pad)
          .filter((p) => p && !CFP_PADS.includes(p) && !DEFAULT_POPS_PADS.includes(p)),
      ),
    ].sort();
    return [...CFP_PADS, ...extras];
  }, [wells.data]);

  const runPads = kind === "cfp" ? cfpPads : [pad ?? "S"];
  // Manual ticks plus long-term shut-in wells the downtime log knows about,
  // minus anything the engineer explicitly kept online.
  const { offline: offlineSet, autoCount } = usePadOffline(runPads);
  const offline = useMemo(() => [...offlineSet].sort(), [offlineSet]);
  const future = useMemo(
    () => runPads.flatMap((p) => futureByPad[p] ?? []),
    [runPads, futureByPad],
  );
  const activeCount = useMemo(() => {
    const names = (wells.data?.wells ?? []).filter((w) => runPads.includes(w.pad)).map((w) => w.name);
    return names.filter((n) => !offlineSet.has(n)).length;
  }, [wells.data, runPads, offlineSet]);

  const start = useStartOptimizeRun();
  const jobId = lastJob[runKey] ?? null;
  const job = useOptimizeJob(jobId);

  // Expired job (server restart): drop the stale id quietly.
  useEffect(() => {
    if (jobId && job.isError) setLastJob(runKey, null);
  }, [jobId, job.isError, runKey, setLastJob]);

  const running = job.data?.status === "running" || start.isPending;

  const run = () => {
    const req: OptimizeRunRequest = {
      kind,
      pad,
      offline,
      future,
      nozzles,
      throats,
      method,
      marginal_wc: autoWc ? null : manualWc,
      parsimony_bopd: parsimony,
      n_pumps: null,
      n_steps: null,
      p0_psi: p0,
      psi_per_kbpd: slope,
      c_pad_pf_psi: cPadPf,
      cfp_pads: cfpPads,
    };
    start.mutate(req, { onSuccess: (r) => setLastJob(runKey, r.job_id) });
  };

  const result = job.data?.status === "done" ? job.data.result : null;
  const padResult = result !== null && "rows" in result ? result : null;

  return (
    <div className="space-y-3">
      <Card className="space-y-3">
        <div className="flex flex-wrap items-start gap-x-6 gap-y-3">
          <div>
            <p className="mb-1 text-xs font-medium text-slate-500">Nozzles</p>
            <ChipToggle options={NOZZLE_OPTIONS} selected={nozzles} onChange={setNozzles} />
          </div>
          <div>
            <p className="mb-1 text-xs font-medium text-slate-500">Throats</p>
            <ChipToggle options={THROAT_OPTIONS} selected={throats} onChange={setThroats} />
          </div>
          {kind === "pad" ? (
            <>
              <label className="block">
                <span className="text-xs font-medium text-slate-500">Solver</span>
                <select
                  value={method}
                  onChange={(e) => setMethod(e.target.value === "mckp" ? "mckp" : "milp")}
                  className="mt-1 block h-8 rounded-md border border-slate-300 bg-white px-2 text-sm"
                >
                  <option value="milp">MILP (exact)</option>
                  <option value="mckp">MCKP (CP-SAT)</option>
                </select>
              </label>
              <div>
                <span className="text-xs font-medium text-slate-500">Marginal WC gate</span>
                <div className="mt-1 flex h-8 items-center gap-2">
                  <label className="flex cursor-pointer items-center gap-1 text-xs text-slate-600">
                    <input
                      type="checkbox"
                      checked={autoWc}
                      onChange={(e) => setAutoWc(e.target.checked)}
                      className="h-4 w-4 rounded border-slate-300 accent-blue-600"
                    />
                    auto
                  </label>
                  {!autoWc && (
                    <input
                      type="number"
                      value={manualWc}
                      min={0}
                      max={1}
                      step={0.01}
                      onChange={(e) => setManualWc(Number(e.target.value))}
                      className={INPUT_CLS}
                    />
                  )}
                </div>
              </div>
              <label className="block">
                <span className="text-xs font-medium text-slate-500">Parsimony (BOPD)</span>
                <input
                  type="number"
                  value={parsimony}
                  min={0}
                  step={5}
                  onChange={(e) => setParsimony(Number(e.target.value))}
                  className={clsx(INPUT_CLS, "mt-1 block")}
                />
              </label>
            </>
          ) : (
            <>
              <div title="B/G/C/J are the CFP pads. L, R, ... also send their produced water through the CFP machines and may join; their PF is modeled at the C-Pad booster pressure. POPs pads separate water on-pad and are not offered.">
                <p className="mb-1 text-xs font-medium text-slate-500">CFP pads in the run</p>
                <ChipToggle options={cfpPadOptions} selected={cfpPads} onChange={setCfpPads} />
              </div>
              <label className="block">
                <span className="text-xs font-medium text-slate-500">PW discharge today (psi)</span>
                <input type="number" value={p0} min={2300} max={2900} step={5} onChange={(e) => setP0(Number(e.target.value))} className={clsx(INPUT_CLS, "mt-1 block")} />
              </label>
              <label className="block">
                <span className="text-xs font-medium text-slate-500">Machine slope (psi/kBPD)</span>
                <input type="number" value={slope} min={9} max={17.5} step={0.25} onChange={(e) => setSlope(Number(e.target.value))} className={clsx(INPUT_CLS, "mt-1 block")} />
              </label>
              <label className="block">
                <span className="text-xs font-medium text-slate-500">C-Pad booster PF (psi)</span>
                <input type="number" value={cPadPf} min={1000} max={5000} step={25} onChange={(e) => setCPadPf(Number(e.target.value))} className={clsx(INPUT_CLS, "mt-1 block")} />
              </label>
            </>
          )}
        </div>

        <div className="flex items-center gap-3 border-t border-slate-100 pt-2.5">
          <button
            type="button"
            disabled={running || nozzles.length === 0 || throats.length === 0 || (kind === "cfp" && cfpPads.length === 0)}
            onClick={run}
            className="flex items-center gap-1.5 rounded-md bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
          >
            <Play className="h-3.5 w-3.5" />
            {running ? "Running..." : `Run ${runKey} optimization`}
          </button>
          <span className="text-xs text-slate-500">
            {activeCount} active well{activeCount === 1 ? "" : "s"}
            {offline.length > 0 &&
              ` - ${offline.length} offline` +
                (autoCount > 0 ? ` (${autoCount} long-term shut-in)` : "")}
            {future.length > 0 && ` - ${future.length} future`}
            {" - models from saved fits (set them on the Single Well solver)"}
          </span>
        </div>
        {running && job.data?.progress && (
          <p className="text-xs text-slate-500">
            {job.data.progress} ({fmtNum(job.data.seconds)}s)
          </p>
        )}
        {job.data?.status === "error" && <WarnNote>Run failed: {job.data.error}</WarnNote>}
        {start.isError && <WarnNote>Could not start the run: {start.error.message}</WarnNote>}
      </Card>

      {running && !job.data?.progress && <Spinner label="Starting run" />}
      {kind === "pad" && pad && (
        // Curves left, readiness right, 50/50: the plant and the wells
        // feeding it read together instead of a full screen apart.
        // grid-cols-2 is repeat(2, minmax(0, 1fr)), so the chart half can
        // actually shrink; min-w-0 does the same for the flex-less children.
        <div className={clsx("grid gap-4", aside && "xl:grid-cols-2")}>
          <div className="min-w-0">
            <PadCharts pad={pad} result={padResult} />
          </div>
          {aside && <div className="min-w-0">{aside}</div>}
        </div>
      )}
      {padResult !== null && <PadResults result={padResult} />}
      {result !== null && "summary" in result && <CfpResults result={result} />}
    </div>
  );
}
