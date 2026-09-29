/**
 * One optimization run tab (S / I / M / E pad or CFP) - constraints form,
 * Run button, live job progress, and results. The engines run server-side
 * over SAVED fits; the board tab's offline flags and future wells feed
 * straight into the run.
 *
 * The form and the job id persist per tab in the optimize store, so
 * switching tabs (or reloading) re-attaches to a still-running or recently
 * finished job with the settings that produced it. A result whose inputs no
 * longer match the form says so; an expired job says it is gone.
 */

import clsx from "clsx";
import { Download, Play } from "lucide-react";
import { useEffect, useMemo, useState, type ReactNode } from "react";

import { isMissingJob, stableStringify } from "../../api/client";

import { useOptimizeJob, usePumpCurve, useStartOptimizeRun, useWells } from "../../api/hooks";
import { HYDRAULICS_LABELS } from "../../api/types";
import type {
  CfpMoveRow,
  CfpRunResult,
  ChokeLadderAction,
  ChokeLadderRung,
  ChokePlanResult,
  ChokePlanRow,
  EPadBuild,
  OptimizeRunRequest,
  PadRunResult,
  PadRunRow,
  RunPad,
  RunCoverage,
} from "../../api/types";
import { Card, Spinner, WarnNote } from "../../components/ui";
import { WellHistoryLink } from "../../components/WellHistoryLink";
import { downloadCsv } from "../../lib/csv";
import { fmtNum, fmtSigned } from "../../lib/format";
import { DEFAULT_POPS_PADS } from "../../state/wellSort";
import { formFor, useOptimizeStore, type RunForm } from "../../state/optimize";

import { CancelJobButton, CancelledNote } from "./CancelJob";
import { CfpResultCharts } from "./CfpCharts";
import { ChokeDumbbell, IprLandingTable } from "./ChokeCharts";
import { usePadOffline } from "./offline";
import { PadCharts } from "./PadCharts";
import { unselectedOutcomeLabel } from "./outcomes";
import { PlanRobustness } from "./PlanRobustness";
import {
  changeText,
  fmtDuration,
  planChanges,
  requestBlockers,
  requestChanges,
  runProgress,
  runStatus,
  type RunStatus,
} from "./runSummary";

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

/** Single-select pump-count chips: how many booster pumps are online for
 *  the run. null (untouched) = the plant's own default, its first option. */
function PumpCountChips({
  options,
  selected,
  onChange,
}: {
  options: number[];
  selected: number | null;
  onChange: (next: number) => void;
}) {
  const active = selected ?? options[0];
  return (
    <div className="flex flex-wrap gap-1">
      {options.map((o) => (
        <button
          key={o}
          type="button"
          onClick={() => onChange(o)}
          className={clsx(
            "rounded px-1.5 py-0.5 text-xs font-medium transition-colors",
            o === active
              ? "bg-blue-600 text-white"
              : "bg-white text-slate-500 ring-1 ring-slate-200 hover:bg-slate-50",
          )}
        >
          {o}
        </button>
      ))}
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
function FitSource({ row }: { row: Pick<PadRunRow, "ipr_source" | "ipr_r2" | "has_friction" | "pump_calibration" | "hydraulics_model" | "donor"> }) {
  const r2 = row.ipr_r2;
  if (row.donor) {
    // A planned well runs on its donor's saved inputs with a clean pump.
    return (
      <span className="text-[11px] font-medium text-indigo-700" title={`Planned well: well inputs copied from ${row.donor}'s ${row.ipr_source === "saved" ? "saved fit" : "current inputs"}; clean reference pump.`}>
        donor {row.donor}
        <span className="block text-[10px] font-normal text-slate-500">{HYDRAULICS_LABELS[row.hydraulics_model ?? "beggs"]}</span>
      </span>
    );
  }
  const hydraulics = row.hydraulics_model ?? row.pump_calibration?.hydraulics_model ?? "beggs";
  // R2 <= 0 means the Vogel curve tracks the tests WORSE than a flat line -
  // the pump picked against it is noise, so it reads as loud as defaults.
  const broken = r2 !== null && r2 <= 0;
  const weak = r2 !== null && r2 > 0 && r2 < 0.5;
  const [label, tone, hint] =
    row.ipr_source === "saved"
      ? ["saved", "text-emerald-700", "Engineer-reviewed IPR from prop_hist, anchored on a pinned well test."]
      : row.ipr_source === "manual"
      ? [
          "manual pt",
          "text-sky-700",
          "Engineer-chosen operating point with NO well test behind it (a joint match, a backmatched BHP, an applied permutation). Reviewed, but not measured - the curve away from that point is an assumption.",
        ]
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
      {row.has_friction && <span className={clsx("ml-1", row.pump_calibration?.quality?.provisional ? "text-amber-700" : "text-slate-500")} title={row.pump_calibration?.message}>{row.pump_calibration?.quality?.provisional ? "pump fit - provisional" : "installed fit"}</span>}
      {!row.has_friction && (
        <span className="ml-1 text-slate-400" title="Reference pump coefficients - no verified installation calibration.">
          reference
        </span>
      )}
      <span className="block text-[10px] font-normal text-slate-500">{HYDRAULICS_LABELS[hydraulics]}</span>
    </span>
  );
}

/** Sweep points the allocator could not settle. The plan is the best of
 *  the points that did; a solver error (not an infeasible header) means a
 *  better header may have been missed. */
function FailedTrialsNote({ meta }: { meta: Record<string, unknown> }) {
  const failed = Array.isArray(meta.failed_trials) ? (meta.failed_trials as Record<string, unknown>[]) : [];
  if (failed.length === 0) return null;
  const statuses = [...new Set(failed.map((f) => String(f.status ?? "failed")))];
  const solverTrouble = statuses.some((s) => s !== "infeasible" && s !== "unsupported");
  const text = `${failed.length} sweep point${failed.length === 1 ? "" : "s"} had no allocation (${statuses.join(", ")}).`;
  return solverTrouble ? (
    <WarnNote>{text} The solver did not finish at those headers, so a better plan there cannot be ruled out.</WarnNote>
  ) : (
    <p className="text-xs text-slate-500">{text} No plan meets the constraints at those headers.</p>
  );
}

function CoverageNotice({ coverage, cfp = false }: { coverage?: RunCoverage; cfp?: boolean }) {
  if (!coverage) return <WarnNote>This older result has no complete well accounting. Run again before reviewing a pad recommendation.</WarnNote>;
  return (
    <div className="space-y-2">
      {!coverage.complete && <WarnNote>
        Incomplete exploratory run: {coverage.unaccounted_wells.join(", ")} have no accounted operating model.
        {cfp ? " The measured pressure anchor is preserved; these wells' response and possible changes are unknown." : " Their online water loads are unaccounted for; these totals do not establish a feasible whole-pad plan."}
      </WarnNote>}
      <details className="text-xs text-slate-600">
        <summary className="cursor-pointer">Well accounting: {coverage.accounted_online}/{coverage.expected_online} online wells accounted for</summary>
        <div className="mt-2 space-y-1">{coverage.rows.map((r) => <p key={r.well}><strong>{r.well}</strong> ({r.role}): {r.reason}</p>)}</div>
      </details>
    </div>
  );
}

const STATUS_CLS: Record<RunStatus["tone"], string> = {
  good: "bg-emerald-50 text-emerald-800 ring-emerald-200",
  warn: "bg-amber-50 text-amber-800 ring-amber-200",
  bad: "bg-rose-50 text-rose-800 ring-rose-200",
};

function StatusChip({ status }: { status: RunStatus }) {
  return (
    <span title={status.detail} className={clsx("rounded px-2 py-0.5 text-xs font-semibold ring-1", STATUS_CLS[status.tone])}>
      {status.label}
    </span>
  );
}

const PAD_CSV_COLUMNS = [
  { key: "well", label: "Well" },
  { key: "current_pump", label: "Current pump" },
  { key: "pump", label: "Plan pump" },
  { key: "pump_state", label: "Plan pump state" },
  { key: "oil", label: "Plan oil (BOPD)" },
  { key: "current_model_oil", label: "Current pump at plan header (BOPD)" },
  { key: "modeled_hardware_gain", label: "Modeled hardware gain (BOPD)" },
  { key: "pf", label: "Plan PF (BPD)" },
  { key: "form_water", label: "Plan formation water (BPD)" },
  { key: "suction", label: "Plan suction (psi)" },
  { key: "test_oil", label: "Recent test oil (BOPD)" },
  { key: "test_pf", label: "Recent test PF (BPD)" },
  { key: "outcome", label: "Outcome" },
  { key: "outcome_reason", label: "Outcome reason" },
];

/** The answer first: what to change, what it is worth, and how far to
 *  trust it. Everything below it is the supporting detail. */
function PadRecommendation({ result }: { result: PadRunResult }) {
  const meta = result.meta;
  const status = runStatus(meta, result.coverage);
  const changes = planChanges(result.rows);
  const gain = metaNum(meta, "modeled_hardware_gain_bopd");
  const header = metaNum(meta, "header_psi");
  const totalWater = meta.water_key === "totl_wat";
  const used = totalWater ? metaNum(meta, "total_machine_water_bpd") : metaNum(meta, "total_pf_bpd");
  const cap = metaNum(meta, "frontier_cap_bpd") ?? metaNum(meta, "station_cap_bpd");
  const shown = changes.slice(0, 10);
  return (
    <Card className="space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        <StatusChip status={status} />
        <p className="text-sm font-semibold text-slate-800">
          {changes.length === 0 ? "Keep every installed pump" : `Change ${changes.length} pump${changes.length === 1 ? "" : "s"}`}
          {gain !== null
            ? ` - modeled gain ${fmtSigned(gain)} BOPD`
            : " - modeled gain withheld"}
        </p>
        <button
          type="button"
          onClick={() => downloadCsv(`${result.pad}-pad-plan.csv`, PAD_CSV_COLUMNS, result.rows as unknown as Record<string, unknown>[])}
          className="ml-auto flex items-center gap-1 rounded-md border border-slate-300 bg-white px-2 py-1 text-xs font-medium text-slate-700 hover:bg-slate-50"
        >
          <Download className="h-3.5 w-3.5" /> CSV
        </button>
      </div>
      <p className="text-xs text-slate-600">
        {status.detail} Header {fmtNum(header)} psi
        {used !== null
          ? `; the plan uses ${fmtNum(used)}${cap !== null ? ` of ${fmtNum(cap)}` : ""} BPD of ${totalWater ? "machine water (lift plus formation)" : "power fluid"}.`
          : "."}
        {gain === null && " The gain is shown only when every well is modeled and the plan meets the plant limits."}
      </p>
      {shown.length > 0 && (
        <ul className="grid gap-x-6 gap-y-0.5 text-[13px] sm:grid-cols-2">
          {shown.map((c) => (
            <li key={c.well} className="flex items-baseline gap-2">
              <span className="font-medium text-slate-700"><WellHistoryLink well={c.well} /></span>
              <span className="text-blue-700">{changeText(c)}</span>
              {c.gain !== null && (
                <span className={clsx("ml-auto tabular-nums", c.gain >= 0 ? "text-emerald-700" : "text-rose-700")}>
                  {fmtSigned(c.gain)} BOPD
                </span>
              )}
            </li>
          ))}
        </ul>
      )}
      {changes.length > shown.length && (
        <p className="text-xs text-slate-500">+ {changes.length - shown.length} more changes in the table below.</p>
      )}
    </Card>
  );
}

function PadResults({ result }: { result: PadRunResult }) {
  const meta = result.meta;
  const nPumpsUsed = metaNum(meta, "n_pumps");
  // a number only when the engineer pinned the header; null on a swept run
  const setpoint = metaNum(meta, "setpoint_psi");
  const lam = metaNum(meta, "lambda_used");
  const lamSource = typeof meta.lambda_source === "string" ? meta.lambda_source : null;
  const wcEquiv = metaNum(meta, "marginal_wc_used");
  const allocation = meta.allocation_status != null && typeof meta.allocation_status === "object"
    ? meta.allocation_status as Record<string, unknown> : null;
  const agreement =
    meta.solver_agreement && typeof meta.solver_agreement === "object"
      ? (meta.solver_agreement as { agree?: boolean; mckp_objective?: number; milp_objective?: number; error?: string; skipped?: string })
      : null;
  const totalTestOil = result.rows.reduce((a, r) => a + (r.test_oil ?? 0), 0);
  const infeasible = meta.feasible === false || meta.modeled_subset_feasible === false;
  return (
    <div className="space-y-3">
      <PadRecommendation result={result} />
      <CoverageNotice coverage={result.coverage} />
      <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
        <Metric label="Header" value={`${fmtNum(metaNum(meta, "header_psi"))} psi`} />
        <Metric label="Total PF" value={`${fmtNum(metaNum(meta, "total_pf_bpd"))} BPD`} />
        {meta.water_key === "totl_wat" && (
          <Metric
            label="Machine water"
            value={`${fmtNum(metaNum(meta, "total_machine_water_bpd"))} BPD`}
            title="Formation water plus lift water handled by the pad pumps"
          />
        )}
        <Metric label={result.coverage?.complete ? "Proposed modeled oil" : "Subset modeled oil"} value={`${fmtNum(metaNum(meta, "total_oil_bopd"))} BOPD`} />
        <Metric
          label="If pumps unchanged"
          value={`${fmtNum(metaNum(meta, "current_model_oil_bopd"))} BOPD`}
          title="Installed pumps modeled at the plan header with the same saved well inputs"
        />
        <Metric label="Modeled hardware gain" value={`${fmtSigned(metaNum(meta, "modeled_hardware_gain_bopd"))} BOPD`} />
        <Metric label="Recent test oil (context)" value={`${fmtNum(totalTestOil)} BOPD`} title="Sum of recent positive-test medians; test dates can differ. This is not the modeled baseline or an optimization gain." />
        <Metric
          label="Water price"
          value={lam === null ? "-" : lam === 0 ? "none" : `${fmtNum(lam * 1000, 1)} BOPD/MBPD`}
          title={
            lam === null
              ? undefined
              : `Oil given up per MBPD of ${meta.water_key === "totl_wat" ? "formation plus lift" : "lift"} water in the objective (oil - λ·water)${
                  wcEquiv !== null ? `; equivalent marginal WC ${(wcEquiv * 100).toFixed(1)}%` : ""
                }${lamSource ? `; source: ${lamSource}` : ""}`
          }
        />
      </div>
      {typeof meta.comparison_basis === "string" && <p className="text-xs text-slate-500">{meta.comparison_basis}</p>}
      {lam !== null && <p className="text-xs text-slate-600">{lam === 0
        ? "Objective: maximize oil within plant capacity."
        : `Objective: oil minus a manual water price of ${fmtNum(lam * 1000, 1)} BOPD/MBPD.`}</p>}
      {metaNum(meta, "diagnostic_lambda") !== null && <p className="text-xs text-slate-500">
        Estimated value of capacity: {fmtNum((metaNum(meta, "diagnostic_lambda") ?? 0) * 1000, 1)} BOPD/MBPD.
        {" "}This is a frontier estimate, separate from the allocation objective.
      </p>}
      {nPumpsUsed !== null && (
        <p className="text-xs text-slate-500">
          Plant modeled with {nPumpsUsed} booster pump{nPumpsUsed === 1 ? "" : "s"} online.
        </p>
      )}
      {setpoint !== null && (
        <p className="text-xs text-slate-500">
          Header pinned at {fmtNum(setpoint)} psi by the engineer - the pressure was not swept.
        </p>
      )}
      {meta.converged === false && (
        <WarnNote>Plant coupling did not converge - treat the header and totals as approximate.</WarnNote>
      )}
      {meta.over_capacity === true && <WarnNote>Plan exceeds plant capacity.</WarnNote>}
      {infeasible && <WarnNote>This plan does not meet all modeled operating limits. Resolve the conditions below before treating it as an operating recommendation.</WarnNote>}
      {(meta.in_range === false || meta.recirc === true) && <WarnNote>Machine flow is outside the modeled operating range or requires recirculation. Net well demand alone does not establish a valid machine operating point.</WarnNote>}
      {Array.isArray(meta.operating_assumptions) && meta.operating_assumptions.map((note, i) =>
        <WarnNote key={i}>{String(note)}</WarnNote>)}
      <FailedTrialsNote meta={meta} />
      {meta.sweep_pruning != null && typeof meta.sweep_pruning === "object" && (() => {
        const p = meta.sweep_pruning as { enabled?: boolean; skipped_solves?: number; total_solves?: number };
        return p.enabled && (p.skipped_solves ?? 0) > 0 ? (
          <p className="text-xs text-slate-500">
            Header search skipped {fmtNum(p.skipped_solves ?? 0)} of {fmtNum((p.skipped_solves ?? 0) + (p.total_solves ?? 0))} pump
            solves that were clearly beaten on both sides; the winning header was re-solved on the full pump grid.
          </p>
        ) : null;
      })()}
      {allocation && <p className="text-xs text-slate-500">
        Allocation status: {String(allocation.status ?? "unknown")} ({String(allocation.solver ?? "unspecified solver")}).
        {allocation.time_limit_reached === true && typeof allocation.gap === "number" &&
          ` Stopped at the time limit within ${fmtNum(allocation.gap * 100, 4)}% of the best possible plan.`}
        {typeof allocation.refinement_reason === "string" && ` Precision refinement: ${allocation.refinement_reason}.`}
        {" "}This applies to the sampled pump choices; pressure search and model support are checked separately.
      </p>}
      {typeof meta.search_scope === "string" && (
        <p className="text-xs text-slate-500">
          Pump choices were checked at their coupled station pressure. Pressure-balance error:
          {" "}{fmtNum(metaNum(meta, "coupling_residual_psi"), 1)} psi.
          {" "}{fmtNum(metaNum(meta, "rejected_selections"))} selections could not be qualified.
          {" "}This is the best settled plan found in the search.
        </p>
      )}

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
              <th className="px-2 py-1.5 text-left font-semibold">Current pump</th>
              <th className={TH_CLS} title="Median of up to five recent positive-oil tests - context, not the modeled baseline">Test oil (BOPD)</th>
              <th className={TH_CLS}>Test PF (BPD)</th>
              <th className={TH_CLS} title="Installed pump modeled at the plan header with the same saved well inputs">If unchanged (BOPD)</th>
              <th className="px-2 py-1.5 text-left font-semibold">Plan pump</th>
              <th className={TH_CLS}>Plan oil (BOPD)</th>
              <th className={TH_CLS} title="Plan oil minus the installed pump at the same header">Gain (BOPD)</th>
              <th className={TH_CLS}>Plan PF (BPD)</th>
              <th className={TH_CLS} title="Pump suction (flowing BHP at the pump)">Suction (psi)</th>
              <th className={TH_CLS} title="Extra oil this well makes per 1,000 BPD more water through its plan pump (the local slope of its pump curve)">Marginal oil per 1,000 BPD</th>
            </tr>
          </thead>
          <tbody>
            {result.rows.map((r) => {
              const change = r.pump_state === "replacement" || (r.pump !== null && r.current_pump !== null && r.pump !== r.current_pump);
              return (
                <tr key={r.well} className="border-b border-slate-100 last:border-b-0">
                  <td className="px-2 py-1 text-left font-medium text-slate-700"><WellHistoryLink well={r.well} /></td>
                  <td className="px-2 py-1 text-left">
                    <FitSource row={r} />
                  </td>
                  <td className="px-2 py-1 text-left text-slate-600">{r.current_pump ?? "-"}</td>
                  <td className={clsx(TD_CLS, "text-slate-600")}>{fmtNum(r.test_oil)}</td>
                  <td className={clsx(TD_CLS, "text-slate-500")}>{fmtNum(r.test_pf)}</td>
                  <td className={TD_CLS}>{fmtNum(r.current_model_oil ?? null)}</td>
                  <td className="px-2 py-1 text-left">
                    {r.pump === null ? (
                      <span className="font-medium text-amber-700" title={r.outcome_reason}>
                        {unselectedOutcomeLabel(r)}
                      </span>
                    ) : (
                      <span className={clsx("font-medium", change ? "text-blue-700" : "text-slate-700")}>
                        {r.pump}
                        {r.pump_state && <span className="block text-[10px] font-normal">{r.pump_state === "replacement" ? "Replace (clean reference)" : "Keep installed"}</span>}
                        {r.sonic && <span title="sonic throat"> *</span>}
                      </span>
                    )}
                  </td>
                  <td className={clsx(TD_CLS, "text-slate-700")}>{fmtNum(r.oil)}</td>
                  <td className={TD_CLS}>{fmtSigned(r.modeled_hardware_gain ?? null)}</td>
                  <td className={clsx(TD_CLS, "text-slate-600")}>{fmtNum(r.pf)}</td>
                  <td className={clsx(TD_CLS, "text-slate-500")}>{fmtNum(r.suction)}</td>
                  <td className={clsx(TD_CLS, "text-slate-500")}>{fmtNum(r.marginal_oil === null ? null : r.marginal_oil * 1000, 1)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </Card>

      {agreement && agreement.error && (
        <WarnNote>MILP cross-check failed: {agreement.error}</WarnNote>
      )}
      {agreement && agreement.skipped && <p className="text-xs text-slate-500">{agreement.skipped}</p>}
      {agreement && !agreement.error && agreement.agree === false && (
        <WarnNote>
          The two solvers disagree at the winning header (MCKP {fmtNum(agreement.mckp_objective ?? null, 1)} vs MILP{" "}
          {fmtNum(agreement.milp_objective ?? null, 1)} BOPD-equivalent). Treat the plan as approximate.
        </WarnNote>
      )}
      {agreement && !agreement.error && agreement.agree === true && (
        <p className="text-xs text-slate-500">
          {allocation?.requested_solver === "cp-sat" && typeof allocation.refinement_reason === "string"
            ? "The CP-SAT request used original-unit MILP refinement. The repeated MILP result agrees; this is not an independent solver cross-check."
            : "MILP and CP-SAT agree on the allocation objective at the search header."}
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

const ACTION_META: Record<
  ChokePlanRow["action"],
  { label: string; cls: string; hint: string }
> = {
  shut: { label: "SHUT IN", cls: "text-amber-700", hint: "Close the well in for the outage." },
  choke: {
    label: "CHOKE",
    cls: "text-blue-700",
    hint: "Pinch the wellhead PF throttle down to the delivered pressure / PF rate shown.",
  },
  hold: {
    label: "HOLD",
    cls: "text-slate-600",
    hint: "Model would not solve this well - held at its measured test rates; only shut-in was considered.",
  },
  full: { label: "FULL", cls: "text-slate-600", hint: "Leave full open at the header." },
  excluded: {
    label: "n/a",
    cls: "text-slate-400",
    hint: "No model solution and no recent test - contributes nothing to the plan.",
  },
};

/** strategy="choke" results: every installed pump HELD, the plan is per-well
 *  PF settings. Rows arrive sorted action-first (shut, choke, hold, full). */
function ChokePlanResults({ result }: { result: ChokePlanResult }) {
  const meta = result.meta;
  const lam = metaNum(meta, "lambda_bopd_per_bpd");
  // An incomplete run moves the subset's verdict out of ``feasible``.
  const infeasible = meta.feasible === false || meta.modeled_subset_feasible === false;
  const projD = infeasible ? null : metaNum(meta, "projected_d_oil_bopd");
  const headerToday = metaNum(meta, "header_today_psi");
  const status = runStatus(meta, result.coverage);
  const nChoke = metaNum(meta, "n_choked") ?? 0;
  const nShut = metaNum(meta, "n_shut") ?? 0;
  return (
    <div className="space-y-3">
      <Card className="flex flex-wrap items-center gap-2">
        <StatusChip status={status} />
        <p className="text-sm font-semibold text-slate-800">
          {nChoke + nShut === 0 ? "Run every well full open" : `Choke ${nChoke}, shut in ${nShut}`}
          {` at a ${fmtNum(metaNum(meta, "header_psi"))} psi header`}
          {projD !== null && ` - projected ${fmtSigned(projD)} BOPD vs today`}
        </p>
        <p className="w-full text-xs text-slate-600">{status.detail} Installed pumps are held; the actions are wellhead PF settings.</p>
      </Card>
      <CoverageNotice coverage={result.coverage} />
      <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
        <Metric
          label="Header"
          value={`${fmtNum(metaNum(meta, "header_psi"))} psi`}
          title={headerToday !== null ? `Today's settled header: ${fmtNum(headerToday)} psi` : undefined}
        />
        <Metric
          label={meta.water_key === "totl_wat" ? "Machine water / budget" : "PF / budget"}
          value={`${fmtNum(metaNum(meta, "total_machine_water_bpd") ?? metaNum(meta, "total_pf_bpd"))} / ${fmtNum(metaNum(meta, "frontier_cap_bpd"))}`}
          title="BPD - the budget is the bank's capability frontier at this header and pump count"
        />
        <Metric label="Model oil" value={`${fmtNum(metaNum(meta, "total_oil_bopd"))} BOPD`} />
        <Metric
          label="Proj. vs today"
          value={projD === null ? "-" : `${projD >= 0 ? "+" : ""}${fmtNum(projD)} BOPD`}
          title="Test-anchored projection using the modeled rate ratio. It assumes the model's relative pressure response is accurate."
        />
        <Metric
          label="Selected reduction tradeoff"
          value={lam === null ? "no trims" : `${fmtNum(lam * 1000)} BOPD/MBPD`}
          title="Largest average oil loss per MBPD freed among selected reductions; a diagnostic, separate from the maximum-oil objective"
        />
      </div>
      <p className="text-xs text-slate-500">
        Existing and planned pumps held. {fmtNum(metaNum(meta, "n_choked"))} choked,{" "}
        {fmtNum(metaNum(meta, "n_shut"))} shut in, {fmtNum(metaNum(meta, "n_full"))} full open.
      </p>
      {meta.recirc === true && (
        <WarnNote>
          Machine demand sits below the {fmtNum(metaNum(meta, "min_total_flow"))} BPD minimum
          for this pump count. A qualified recycle or operating arrangement is needed.
        </WarnNote>
      )}
      {meta.over_capacity === true && <WarnNote>Plan exceeds plant capacity.</WarnNote>}

      {infeasible && <WarnNote>This choke plan does not meet all modeled operating limits. Its projected gain is withheld.</WarnNote>}
      {Array.isArray(meta.operating_assumptions) && meta.operating_assumptions.map((note, i) =>
        <WarnNote key={i}>{String(note)}</WarnNote>)}
      <FailedTrialsNote meta={meta} />
      {meta.allocation_status != null && typeof meta.allocation_status === "object" && <p className="text-xs text-slate-500">
        Allocation status: {String((meta.allocation_status as Record<string, unknown>).status ?? "unknown")} on the sampled choke settings.
      </p>}

      <ChokeDumbbell plan={result.plan} />

      <Card padded={false} className="overflow-x-auto">
        <table className="w-full border-collapse text-[13px]">
          <thead>
            <tr className="border-b border-slate-200 bg-slate-50 text-slate-600">
              <th rowSpan={2} className="px-2 py-1 text-left font-semibold align-bottom">
                Well
              </th>
              <th rowSpan={2} className="px-2 py-1 text-left font-semibold align-bottom">
                Fit
              </th>
              <th
                rowSpan={2}
                className="px-2 py-1 text-left font-semibold align-bottom"
                title="Installed pump - this plan never changes it"
              >
                Pump (held)
              </th>
              <th
                colSpan={4}
                className="border-l border-slate-200 px-2 py-1 text-center font-semibold"
                title="What to DO at each wellhead: the PF setting this plan asks for"
              >
                Plan setting
              </th>
              <th
                colSpan={2}
                className="border-l border-slate-200 px-2 py-1 text-center font-semibold"
                title="What that setting saves and costs vs running this well wide open at the plan header"
              >
                vs full open
              </th>
              <th
                rowSpan={2}
                className={clsx(TH_CLS, "border-l border-slate-200 align-bottom")}
                title="Measured test oil times the modeled rate ratio; accuracy depends on the relative pressure response"
              >
                Proj. oil (BOPD)
              </th>
              <th
                rowSpan={2}
                className={clsx(TH_CLS, "align-bottom")}
                title="Oil lost per 1,000 BPD of PF freed if this well is trimmed ONE more step - who gives up the least next"
              >
                Next trim (BOPD/MBPD)
              </th>
            </tr>
            <tr className="border-b border-slate-200 bg-slate-50 text-slate-600">
              <th className="border-l border-slate-200 px-2 py-1 text-left font-semibold">
                Action
              </th>
              <th
                className={TH_CLS}
                title="Pinch the wellhead PF throttle until the delivered gauge reads this. FULL = leave wide open at the header."
              >
                Set PF psi to
              </th>
              <th className={TH_CLS} title="Expected PF rate at that setting - the number to pinch to on the well's PF meter">
                PF rate (BPD)
              </th>
              <th className={TH_CLS} title="Model oil at the plan setting">
                Oil (BOPD)
              </th>
              <th
                className={clsx(TH_CLS, "border-l border-slate-200")}
                title="PF handed back to the bank by this setting (negative = freed)"
              >
                PF freed (BPD)
              </th>
              <th className={TH_CLS} title="Oil given up for that PF. 0 = a free choke (sonic-flat well); + = choking GAINS oil">
                Oil cost (BOPD)
              </th>
            </tr>
          </thead>
          <tbody>
            {result.plan.map((r) => {
              const a = ACTION_META[r.action];
              return (
                <tr key={r.well} className="border-b border-slate-100 last:border-b-0">
                  <td className="px-2 py-1 text-left font-medium text-slate-700"><WellHistoryLink well={r.well} /></td>
                  <td className="px-2 py-1 text-left">
                    <FitSource row={r} />
                  </td>
                  <td className="px-2 py-1 text-left text-slate-600">{r.pump ?? "-"}</td>
                  <td className="px-2 py-1 text-left">
                    <span className={clsx("font-medium", a.cls)} title={a.hint}>
                      {a.label}
                    </span>
                  </td>
                  <td className={clsx(TD_CLS, "border-l border-slate-100 text-slate-600")}>
                    {r.action === "choke" || r.action === "full" ? fmtNum(r.delivered_psi) : "-"}
                  </td>
                  <td className={clsx(TD_CLS, "text-slate-700")}>{fmtNum(r.pf)}</td>
                  <td className={clsx(TD_CLS, "text-slate-700")}>{fmtNum(r.oil)}</td>
                  <td className={clsx(TD_CLS, "border-l border-slate-100 text-slate-500")}>
                    {r.d_pf_vs_full !== null && r.d_pf_vs_full !== 0 ? fmtNum(r.d_pf_vs_full) : "-"}
                  </td>
                  <td className={clsx(TD_CLS, "text-slate-500")}>
                    {r.d_oil_vs_full !== null && r.d_oil_vs_full !== 0
                      ? `${r.d_oil_vs_full > 0 ? "+" : ""}${fmtNum(r.d_oil_vs_full)}`
                      : "-"}
                  </td>
                  <td className={clsx(TD_CLS, "border-l border-slate-100 text-slate-500")}>
                    {fmtNum(r.projected_oil)}
                  </td>
                  <td className={clsx(TD_CLS, "text-slate-400")}>
                    {r.next_trim_bopd_per_bpd === null ? "-" : fmtNum(r.next_trim_bopd_per_bpd * 1000)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </Card>

      <HeaderDropLadder rungs={meta.ladder} />
      <IprLandingTable plan={result.plan} />
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

/** "choke MPM-64 @ 1,700" / "shut MPM-22" / "hold MPM-31". */
function ladderAction(a: ChokeLadderAction): string {
  return a.action === "choke" ? `choke ${a.well} @ ${fmtNum(a.set_psi)}` : `${a.action} ${a.well}`;
}

/** Contingency ladder: if the PF bank sags below the plan header, what is
 *  the best response and what does it gain over doing nothing. Collapsed by
 *  default; renders nothing on runs made before the feature (no ladder). */
function HeaderDropLadder({ rungs }: { rungs: ChokeLadderRung[] | undefined }) {
  if (rungs == null || rungs.length === 0) return null;
  return (
    <Card padded={false} className="overflow-x-auto">
      <details>
        <summary className="cursor-pointer select-none px-2 py-2 text-xs font-semibold text-slate-600 hover:text-slate-800">
          Header-drop decision ladder
        </summary>
        <p className="px-2 pb-1 text-[11px] text-slate-500">
          If the PF bank degrades until the all-run header settles this far below the plan
          header: the best response, and what it gains over doing nothing.
        </p>
        <table className="w-full border-collapse text-[13px]">
          <thead>
            <tr className="border-b border-slate-200 bg-slate-50 text-slate-600">
              <th className={TH_CLS} title="How far the all-run header settles below the plan header">
                Drop (psi)
              </th>
              <th className={TH_CLS}>Settles at (psi)</th>
              <th className={TH_CLS} title="Pad oil if every well just runs at the sagged header">
                Do nothing (BOPD)
              </th>
              <th className={TH_CLS} title="Header the best response holds instead">
                Hold header at (psi)
              </th>
              <th className="px-2 py-1.5 text-left font-semibold">Best response</th>
              <th className={TH_CLS}>Pad oil (BOPD)</th>
              <th className={TH_CLS} title="Best-response pad oil minus do-nothing pad oil">
                Gain (BOPD)
              </th>
            </tr>
          </thead>
          <tbody>
            {rungs.map((r) => {
              const labels = r.actions.map(ladderAction);
              const full = labels.join(", ");
              const shown =
                labels.length > 3
                  ? `${labels.slice(0, 3).join(", ")} + ${labels.length - 3} more`
                  : full;
              return (
                <tr key={r.drop_psi} className="border-b border-slate-100 last:border-b-0">
                  <td className={clsx(TD_CLS, "font-medium text-slate-700")}>
                    -{fmtNum(r.drop_psi)}
                  </td>
                  <td className={clsx(TD_CLS, "text-slate-600")}>{fmtNum(r.settles_psi)}</td>
                  <td className={clsx(TD_CLS, "text-slate-500")}>{fmtNum(r.run_all_oil_bopd)}</td>
                  <td className={clsx(TD_CLS, "text-slate-600")}>{fmtNum(r.best_header_psi)}</td>
                  <td
                    className="px-2 py-1 text-left text-slate-600"
                    title={labels.length > 3 ? full : undefined}
                  >
                    {labels.length === 0 ? <span className="text-slate-400">no change</span> : shown}
                  </td>
                  <td className={clsx(TD_CLS, "text-slate-700")}>{fmtNum(r.plan_oil_bopd)}</td>
                  <td className={clsx(TD_CLS, r.gain_bopd > 0 ? "text-emerald-700" : "text-slate-600")}>
                    {r.gain_bopd > 0 ? "+" : ""}
                    {fmtNum(r.gain_bopd)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </details>
    </Card>
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
  // Top moves are ones the plan could make: a move that switches off a
  // Required-online well stays on the ladder below, marked, but is not
  // offered as a gain.
  const singles = s.singles.filter((m) => m.fleet_oil_delta > 0 && m.meets_required !== false).slice(0, 12);
  const breaksRequired = (m: { meets_required?: boolean }) => m.meets_required === false;
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
  // A bring-online row names one pump size; "in plan" must match that size.
  const planKey = (m: { type: string; well: string; to: string | null }) =>
    m.type === "bring_online" ? `${m.type}-${m.well}-${m.to ?? ""}` : `${m.type}-${m.well}`;
  const inPlan = new Set(planActions.map(planKey));
  const cfpStatus: RunStatus = !result.coverage || !result.coverage.complete
    ? { tone: "bad", label: "Exploratory", detail: "Some wells have no usable response table; their changes are unknown." }
    : s.plan_status === "no_feasible_plan"
      ? { tone: "warn", label: "No supported plan", detail: "No combination meets the pressure and required-online constraints." }
      : { tone: "good", label: "Supported", detail: "Every run well has a response table; rates are interpolated, not re-solved at the final pressure." };
  const actionText = (a: (typeof planActions)[number]) =>
    a.type === "shut_in" ? `shut in ${a.well}` : a.type === "bring_online" ? `bring ${a.well} online on ${a.to}` : `${a.well} ${a.from} to ${a.to}`;
  return (
    <div className="space-y-3">
      <Card className="space-y-1">
        <div className="flex flex-wrap items-center gap-2">
          <StatusChip status={cfpStatus} />
          <p className="text-sm font-semibold text-slate-800">
            {planActions.length === 0 ? "Keep today's configuration" : `Make ${planActions.length} change${planActions.length === 1 ? "" : "s"}`}
            {s.plan_gain !== null && ` - modeled ${fmtSigned(s.plan_gain)} BOPD`}
            {s.plan && ` at ${fmtNum(s.plan.pressure)} psi discharge`}
          </p>
        </div>
        {planActions.length > 0 && <p className="text-[13px] text-blue-700">{planActions.map(actionText).join("; ")}</p>}
        <p className="text-xs text-slate-600">{cfpStatus.detail}</p>
      </Card>
      <CoverageNotice coverage={result.coverage} cfp />
      <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
        <Metric label="Reference discharge" value={`${fmtNum(s.today.pressure)} psi`} />
        <Metric
          label={`Modeled oil (${result.n_wells} run wells)`}
          value={`${fmtNum(s.today.oil)} BOPD`}
          title="Modeled oil across these run wells at the reference discharge. Plan gains are measured against this baseline."
        />
        <Metric label="Modeled run-well water" value={`${fmtNum(s.today.water)} BWPD`} title="Machine-water contribution from these modeled wells; not measured total CFP throughput." />
        <Metric
          label="Shadow price"
          value={s.lambda_bopd_per_psi !== null ? `${fmtNum(s.lambda_bopd_per_psi, 2)} BOPD/psi` : "-"}
        />
        <Metric
          label="Best supported plan gain"
          value={s.plan_gain !== null ? `${s.plan_gain > 0 ? "+" : ""}${fmtNum(s.plan_gain)} BOPD` : "-"}
        />
      </div>

      <CfpResultCharts result={result} />
      {s.plan_status === "no_feasible_plan" && <WarnNote>No supported plan meets the pressure and required-online constraints.</WarnNote>}
      {s.search_scope && <p className="text-xs text-slate-500">{typeof s.search_scope === "string"
        ? s.search_scope : s.search_scope.global_optimum_on_surfaces === true
          ? "All combinations were evaluated on the sampled response tables. The result remains conditional on those tables and plant assumptions."
          : "The plan is the best supported combination found in a bounded search; global optimality is not established."}
        {typeof s.search_scope === "object" && s.search_scope.direct_solver_validated === false
          && " Final pressure rates are interpolated from the tables; a direct final-pressure well solve has not been performed."}
      </p>}
      {(s.required_wells?.length ?? 0) > 0 && <p className="text-xs text-slate-600">Required online: {s.required_wells?.join(", ")}</p>}

      {planActions.length > 0 && (
        <Card padded={false} className="overflow-x-auto">
          <p className="px-3 pt-2 text-xs font-semibold text-slate-600">Best supported plan</p>
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
                <tr key={`${m.type}-${m.well}`} className={clsx("border-b border-slate-100 last:border-b-0", breaksRequired(m) && "opacity-50")}
                  title={breaksRequired(m) ? "Switches off a Required-online well; the plan cannot make this move." : undefined}>
                  <td className="px-2 py-1 text-left font-medium text-slate-700">
                    {MOVE_LABELS[m.type]}
                    {breaksRequired(m) && <span className="ml-1 text-[10px] font-normal text-amber-700">breaks required</span>}
                  </td>
                  <td className="px-2 py-1 text-left text-slate-700"><WellHistoryLink well={m.well} /></td>
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
                    {m.at_trip && <span title="Held at the 2,880 psi upper control limit (2,900 psi trip less a 20 psi margin)"> !</span>}
                  </td>
                  <td className="px-2 py-1 text-left">
                    {inPlan.has(planKey(m)) ? (
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

      {s.pairs.length > 0 && <Card padded={false} className="overflow-x-auto">
        <p className="px-3 pt-2 text-xs font-semibold text-slate-600">Bring online with an offset</p>
        <p className="px-3 py-1 text-xs text-slate-500">Jointly evaluated changes, including wells that cannot operate alone. Net oil includes the pressure response of existing wells.</p>
        <table className="w-full border-collapse text-[13px]"><thead><tr className="border-b border-slate-200">
          <th className="px-3 py-2 text-left">Bring online</th><th className="px-3 py-2 text-left">Offset</th>
          <th className={TH_CLS}>Net oil (BOPD)</th><th className={TH_CLS}>Discharge after</th>
        </tr></thead><tbody>{s.pairs.map((pair, i) => <tr key={`${pair.bring_on.well}-${pair.offset.well}-${i}`}
          className={clsx("border-b border-slate-100", breaksRequired(pair) && "opacity-50")}
          title={breaksRequired(pair) ? "Switches off a Required-online well; the plan cannot make this pair." : undefined}>
          <td className="px-3 py-1.5">{pair.bring_on.well}: {pair.bring_on.to}</td>
          <td className="px-3 py-1.5">{pair.offset.well}: {pair.offset.from} to {pair.offset.to}</td>
          <td className={TD_CLS}>{pair.fleet_oil_delta > 0 ? "+" : ""}{fmtNum(pair.fleet_oil_delta)}</td>
          <td className={TD_CLS}>{fmtNum(pair.pressure_after)} psi</td>
        </tr>)}</tbody></table>
      </Card>}

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
                  <td className="px-2 py-1 text-left font-medium text-slate-700"><WellHistoryLink well={m.well} /></td>
                  <td className={clsx(TD_CLS, m.fleet_oil_delta > 0 ? "text-emerald-700" : "text-slate-600")}>
                    {m.fleet_oil_delta > 0 ? "+" : ""}
                    {fmtNum(m.fleet_oil_delta)}
                  </td>
                  <td className={TD_CLS}>{fmtNum(m.own_oil_delta)}</td>
                  <td className={TD_CLS}>
                    {fmtNum(m.pressure_after)}
                    {m.at_trip && <span title="Held at the 2,880 psi upper control limit (2,900 psi trip less a 20 psi margin)"> !</span>}
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
  pad: RunPad | null;
  aside?: ReactNode;
}) {
  const runKey = kind === "cfp" ? "CFP" : (pad as string);
  const wells = useWells();
  const futureByPad = useOptimizeStore((s) => s.future);
  const requiredByPad = useOptimizeStore((s) => s.requiredOnline);
  const lastJob = useOptimizeStore((s) => s.lastJob);
  const lastJobKey = useOptimizeStore((s) => s.lastJobKey);
  const setLastJob = useOptimizeStore((s) => s.setLastJob);
  const forms = useOptimizeStore((s) => s.forms);
  const setForm = useOptimizeStore((s) => s.setForm);
  const resetForm = useOptimizeStore((s) => s.resetForm);

  // Per-tab form, persisted: one pad's pump count or booster settings never
  // ride into another pad's run, and survive leaving the page.
  const form = useMemo(() => formFor(forms, runKey), [forms, runKey]);
  const edit = (patch: Partial<RunForm>) => setForm(runKey, patch);
  const {
    nozzles, throats, method, autoLam, waterPricePerK, autoSetpoint, manualSetpoint,
    p0, referencePadPf, slope, cPadPf, cfpPads,
    ePadBuild, ePadSuction, ePadHzMax, ePadHeaderCap, ePadAmpLimit,
  } = form;
  // S-Pad's header follows its flow, so it has no hold-pumps choke plan.
  const strategy = pad === "S" ? "jpco" : form.strategy;
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

  const runPads = useMemo(() => (kind === "cfp" ? cfpPads : [pad ?? "S"]), [kind, cfpPads, pad]);
  // Manual ticks plus default-offline wells (LTSI, SI shut-ins, recycle),
  // minus anything the engineer explicitly kept online.
  const { offline: offlineSet, autoCount, ready: offlineReady, failed: offlineFailed } = usePadOffline(runPads);
  const offlineSettled = offlineReady || offlineFailed;
  const offline = useMemo(() => [...offlineSet].sort(), [offlineSet]);
  const future = useMemo(
    () => runPads.flatMap((p) => (futureByPad[p] ?? []).map((fw) => ({ ...fw, pad: p }))),
    [runPads, futureByPad],
  );
  const activeCount = useMemo(() => {
    const names = (wells.data?.wells ?? []).filter((w) => runPads.includes(w.pad)).map((w) => w.name);
    return names.filter((n) => !offlineSet.has(n)).length;
  }, [wells.data, runPads, offlineSet]);

  const ePadAmpLimitNum = Number(ePadAmpLimit);
  const ePadAmpLimitReq =
    ePadAmpLimit.trim() === "" || !Number.isFinite(ePadAmpLimitNum) || ePadAmpLimitNum <= 0
      ? null
      : ePadAmpLimitNum;
  // The booster configuration the E-Pad curve sheet and the run must agree
  // on. Ignored (and server-rejected) on every other pad.
  const ePadKnobs = useMemo(
    () => ({
      build: ePadBuild,
      suctionPsi: ePadSuction,
      hzMax: ePadHzMax,
      maxHeaderPsi: ePadHeaderCap,
      ampLimitA: ePadAmpLimitReq,
    }),
    [ePadBuild, ePadSuction, ePadHzMax, ePadHeaderCap, ePadAmpLimitReq],
  );

  // Only the free-pressure pads (I/M/E) have a header to pin, and only a
  // JPCO run sweeps one. Everything else sends null - a swept run.
  const setpointPinnable = kind === "pad" && strategy === "jpco" && pad !== "S";
  const setpointReq = setpointPinnable && !autoSetpoint ? manualSetpoint : null;

  // The plant's selectable online-pump counts, off the (hard-cached) curve
  // payload; [] = fixed train (I/E-Pad) or a CFP run - no control rendered.
  const pumpCurve = usePumpCurve(kind === "pad" ? pad : null, null, ePadKnobs);
  const pumpOptions = pumpCurve.data?.n_pump_options ?? [];
  // A stored count this plant does not offer means "the plant default".
  const nPumps = form.nPumps !== null && pumpOptions.includes(form.nPumps) ? form.nPumps : null;

  const start = useStartOptimizeRun();
  const jobId = lastJob[runKey] ?? null;
  const job = useOptimizeJob(jobId);

  // Expired job (server restart, or an hour after it settled): drop the id
  // and say so, instead of the result silently vanishing.
  const [expired, setExpired] = useState(false);
  useEffect(() => {
    if (jobId && isMissingJob(job.error)) {
      setLastJob(runKey, null);
      setExpired(true);
    }
  }, [jobId, job.error, runKey, setLastJob]);

  const running = job.data?.status === "running" || start.isPending;

  const req: OptimizeRunRequest = useMemo(() => ({
    kind,
    pad,
    offline,
    future,
    required_wells: runPads.flatMap((p) => requiredByPad[p] ?? []),
    nozzles,
    throats,
    method,
    strategy,
    // The form holds BOPD per 1,000 BPD (what the results show); the API
    // takes BOPD per BPD.
    lambda_bopd_per_bpd: autoLam ? null : waterPricePerK / 1000,
    marginal_wc: null,
    parsimony_bopd: 0,
    n_pumps: nPumps,
    n_steps: null,
    setpoint_psi: setpointReq,
    p0_psi: p0,
    psi_per_kbpd: slope,
    c_pad_pf_psi: cPadPf,
    cfp_pad_pf_psi: Object.fromEntries(Object.entries(referencePadPf)
      .filter(([p, value]) => value.trim() !== "" && cfpPads.includes(p)).map(([p, value]) => [p, Number(value)])),
    cfp_pads: cfpPads,
    e_pad_build: ePadBuild,
    e_pad_suction_psi: ePadSuction,
    e_pad_hz_max: ePadHzMax,
    e_pad_max_header_psi: ePadHeaderCap,
    e_pad_amp_limit_a: ePadAmpLimitReq,
  }), [kind, pad, offline, future, runPads, requiredByPad, nozzles, throats, method, strategy, autoLam,
    waterPricePerK, nPumps, setpointReq, p0, slope, cPadPf, referencePadPf, cfpPads, ePadBuild, ePadSuction,
    ePadHzMax, ePadHeaderCap, ePadAmpLimitReq]);
  const blockers = requestBlockers(req);

  const run = () => {
    const key = stableStringify(req);
    setExpired(false);
    // mutateAsync, not mutate(..., {onSuccess}): per-call callbacks are
    // dropped if the engineer leaves the tab before the start returns, which
    // orphaned a running job that then held the only job slot.
    start.mutateAsync(req).then((r) => setLastJob(runKey, r.job_id, key)).catch(() => undefined);
  };

  const result = job.data?.status === "done" ? job.data.result : null;
  // "meta" too: the match-health scorecard payload also carries "rows".
  const padResult = result !== null && "rows" in result && "meta" in result ? result : null;
  const chokeResult = result !== null && "plan" in result ? result : null;
  const cfpResult = result !== null && "summary" in result ? result : null;
  const chokeMode = kind === "pad" && strategy === "choke";

  // What changed since the shown result was requested (after the offline
  // set has settled, so the downtime log loading is not a "change").
  const shownKey = jobId ? lastJobKey[runKey] ?? null : null;
  const changed = useMemo(() => {
    if (result === null || shownKey === null || !offlineSettled) return [];
    try {
      return requestChanges(JSON.parse(shownKey) as OptimizeRunRequest, req);
    } catch {
      return [];
    }
  }, [result, shownKey, offlineSettled, req]);

  const progress = runProgress(job.data?.progress, job.data?.seconds);
  const advancedSummary = [
    strategy === "choke" ? null : `nozzles ${nozzles.join(",") || "none"}`,
    strategy === "choke" ? null : `throats ${throats.join(",") || "none"}`,
    strategy === "choke" ? null : method === "mckp" ? "CP-SAT" : "MILP",
    strategy === "choke" ? null : autoLam ? "maximize oil" : `water ${fmtNum(waterPricePerK, 1)} BOPD per 1,000 BPD`,
    setpointPinnable ? (autoSetpoint ? "header swept" : `header ${fmtNum(manualSetpoint)} psi`) : null,
  ].filter(Boolean).join(" - ");

  const hasResult = padResult !== null || chokeResult !== null || cfpResult !== null;

  return (
    <div className="space-y-3">
      <Card className="space-y-3">
        <div className="flex flex-wrap items-start gap-x-6 gap-y-3">
          {kind === "pad" && (
            <label
              className="block"
              title="Resize picks new nozzle/throat per well (a JPCO costs about a day per pump). Choke / shut in HOLDS every installed pump and only re-allocates power fluid - the short-term plan when a PF booster pump is down."
            >
              <span className="text-xs font-medium text-slate-500">Strategy</span>
              <select
                value={strategy}
                onChange={(e) => edit({ strategy: e.target.value === "choke" ? "choke" : "jpco" })}
                className="mt-1 block h-8 rounded-md border border-slate-300 bg-white px-2 text-sm"
              >
                <option value="jpco">Resize pumps (JPCO)</option>
                <option value="choke" disabled={pad === "S"}>Choke / shut in (hold pumps){pad === "S" ? " - unavailable on S" : ""}</option>
              </select>
            </label>
          )}
          {kind === "pad" && pumpOptions.length > 0 && (
            <div title="Booster pumps online for this run - drop it when a machine is down (e.g. one M-Pad HP pump out). The PF budget, capability frontier and min-flow floor all follow the selected count.">
              <p className="mb-1 text-xs font-medium text-slate-500">Pumps online</p>
              <PumpCountChips options={pumpOptions} selected={nPumps} onChange={(n) => edit({ nPumps: n })} />
            </div>
          )}
          {kind === "cfp" && (
            <>
              <div title="B/G/C/J use the CFP response model. Additional pads assume locally boosted PF; incremental routing from POPS pads is not qualified here.">
                <p className="mb-1 text-xs font-medium text-slate-500">CFP pads in the run</p>
                <ChipToggle options={cfpPadOptions} selected={cfpPads} onChange={(v) => edit({ cfpPads: v })} />
              </div>
              <label className="block">
                <span className="text-xs font-medium text-slate-500">Reference PW discharge (psi)</span>
                <input type="number" value={p0} min={2300} max={2880} step={5} onChange={(e) => edit({ p0: Number(e.target.value) })} className={clsx(INPUT_CLS, "mt-1 block")} />
              </label>
              <label className="block">
                <span className="text-xs font-medium text-slate-500">Machine slope (psi per 1,000 BPD)</span>
                <input type="number" value={slope} min={9} max={17.5} step={0.25} onChange={(e) => edit({ slope: Number(e.target.value) })} className={clsx(INPUT_CLS, "mt-1 block")} />
              </label>
              <label className="block">
                <span className="text-xs font-medium text-slate-500">C-Pad booster PF (psi)</span>
                <input type="number" value={cPadPf} min={1000} max={5000} step={25} onChange={(e) => edit({ cPadPf: Number(e.target.value) })} className={clsx(INPUT_CLS, "mt-1 block")} />
              </label>
            </>
          )}
        </div>

        {kind === "cfp" && <details className="border-t border-slate-100 pt-2 text-xs text-slate-600">
          <summary className="cursor-pointer">Pad PF at the same reference conditions</summary>
          <p className="my-2">Enter pressures corresponding to the reference discharge and online configuration. Blank fields use fixed line-loss assumptions. The reference discharge is a manual scenario value.</p>
          <div className="flex flex-wrap gap-3">{["B", "G", "J"].filter((p) => cfpPads.includes(p)).map((p) => <label key={p}>
            {p}-Pad PF (psi)<input type="number" min={1000} max={5000} placeholder="assumed" value={referencePadPf[p] ?? ""}
              onChange={(e) => edit({ referencePadPf: { ...referencePadPf, [p]: e.target.value } })}
              className={clsx(INPUT_CLS, "ml-2")} />
          </label>)}</div>
        </details>}

        {kind === "pad" && pad === "E" && (
          // E-Pad's booster is the one plant whose configuration is not a
          // measured tag, so the run form has to ask. Build especially: this
          // is how "would the SN35000 make more oil?" gets answered - run the
          // pad on each build and compare the fleet total.
          <div className="flex flex-wrap items-end gap-x-4 gap-y-3 border-t border-slate-100 pt-2.5">
            <label className="block" title="Which build the run assumes is in the ground. Run both and compare the fleet oil to price a changeout.">
              <span className="text-xs font-medium text-slate-500">Booster build</span>
              <select
                value={ePadBuild}
                onChange={(e) => edit({ ePadBuild: e.target.value as EPadBuild })}
                className={clsx(INPUT_CLS, "mt-1 block w-56")}
              >
                <option value="SM25000_26STG">SM25000 - 26 stg (in well)</option>
                <option value="SN35000_18STG">SN35000 - 18 stg (alternative)</option>
              </select>
            </label>
            <label className="block" title="Booster suction (psig) from the CFP water header. 2,704 was measured at the current-limited maximum rate in the E-41 rate test (2,725 at the 27,789 BWPD baseline); rate depends on the header holding it.">
              <span className="text-xs font-medium text-slate-500">Suction (psig)</span>
              <input type="number" value={ePadSuction} min={0} max={5000} step={25} onChange={(e) => edit({ ePadSuction: Number(e.target.value) })} className={clsx(INPUT_CLS, "mt-1 block")} />
            </label>
            <label className="block" title="VFD speed cap. The stage curve is the 60 Hz catalog curve, so it cannot exceed 60.">
              <span className="text-xs font-medium text-slate-500">Max speed (Hz)</span>
              <input type="number" value={ePadHzMax} min={30} max={60} step={1} onChange={(e) => edit({ ePadHzMax: Number(e.target.value) })} className={clsx(INPUT_CLS, "mt-1 block")} />
            </label>
            <label className="block" title="Operational discharge cap on the PF header - piping/wellhead, not the pump. 3,500 is adopted from I-Pad pending an E-Pad number; the booster's own frontier peaks near 4,560 psi, so this cap is what limits the sweep.">
              <span className="text-xs font-medium text-slate-500">Header cap (psi)</span>
              <input type="number" value={ePadHeaderCap} min={1000} max={5000} step={25} onChange={(e) => edit({ ePadHeaderCap: Number(e.target.value) })} className={clsx(INPUT_CLS, "mt-1 block")} />
            </label>
            <label className="block" title="Motor current limit. 889 A is the E-41 drive's I-limit reached in the rate test (29,491 BWPD at 3,400 psi). Blank enforces no cap.">
              <span className="text-xs font-medium text-slate-500">Amp limit (A)</span>
              <input type="number" value={ePadAmpLimit} min={1} max={5000} step={1} placeholder="none" onChange={(e) => edit({ ePadAmpLimit: e.target.value })} className={clsx(INPUT_CLS, "mt-1 block")} />
            </label>
          </div>
        )}

        {kind === "pad" && pad === "E" && (
          <p className="text-xs text-slate-600">
            The installed SM25000 is calibrated to the E-41 rate test: it current-limits
            at 889 A at 29,491 BWPD and 3,400 psi from 2,704 psi suction, so that is its
            capacity at the 3,400 psi header. One test, one speed: rates at other headers
            are the catalog curve derated to that point. The 3,500 psi header cap is still
            I-Pad&apos;s number, not an E-Pad measurement. The alternative build runs as new on
            the same motor. See the E-Pad booster tab for the candidate comparison.
          </p>
        )}

        {kind === "pad" &&
          nPumps !== null &&
          pumpOptions.length > 0 &&
          nPumps < Math.max(...pumpOptions) && (
            <p className="text-xs text-amber-700">
              Reduced bank: optimizing on the {nPumps}-pump frontier - the PF budget and
              deliverable header drop, and the min-flow (recirc) floor drops with them.
            </p>
          )}

        {(kind === "cfp" || !chokeMode) && (
          // Defaults are sound for a first run; these knobs are for studies.
          <details className="border-t border-slate-100 pt-2">
            <summary className="cursor-pointer text-xs text-slate-600">
              Advanced settings <span className="text-slate-400">({advancedSummary || "defaults"})</span>
            </summary>
            <div className="mt-2 flex flex-wrap items-start gap-x-6 gap-y-3">
              <div>
                <p className="mb-1 text-xs font-medium text-slate-500">Nozzles</p>
                <ChipToggle options={NOZZLE_OPTIONS} selected={nozzles} onChange={(v) => edit({ nozzles: v })} />
              </div>
              <div>
                <p className="mb-1 text-xs font-medium text-slate-500">Throats</p>
                <ChipToggle options={THROAT_OPTIONS} selected={throats} onChange={(v) => edit({ throats: v })} />
              </div>
              {kind === "pad" && (
                <>
                  <label className="block" title="Both solve the same discrete allocation. CP-SAT is cross-checked against MILP at the winning header.">
                    <span className="text-xs font-medium text-slate-500">Solver</span>
                    <select
                      value={method}
                      onChange={(e) => edit({ method: e.target.value === "mckp" ? "mckp" : "milp" })}
                      className="mt-1 block h-8 rounded-md border border-slate-300 bg-white px-2 text-sm"
                    >
                      <option value="milp">MILP</option>
                      <option value="mckp">CP-SAT (cross-checked)</option>
                    </select>
                  </label>
                  <div title="Default: maximize oil within the plant's machine-water capacity. Untick to charge every barrel of machine water a deliberate oil-equivalent cost.">
                    <span className="text-xs font-medium text-slate-500">Objective</span>
                    <div className="mt-1 flex h-8 items-center gap-2">
                      <label className="flex cursor-pointer items-center gap-1 text-xs text-slate-600">
                        <input
                          type="checkbox"
                          checked={autoLam}
                          onChange={(e) => edit({ autoLam: e.target.checked })}
                          className="h-4 w-4 rounded border-slate-300 accent-blue-600"
                        />
                        Maximize oil within capacity
                      </label>
                      {!autoLam && (
                        <label className="flex items-center gap-1 text-xs text-slate-600">
                          Charge
                          <input
                            aria-label="Water price, BOPD per 1,000 BPD"
                            type="number"
                            value={waterPricePerK}
                            min={0}
                            max={10000}
                            step={1}
                            onChange={(e) => edit({ waterPricePerK: Number(e.target.value) })}
                            className={clsx(INPUT_CLS, "w-20")}
                          />
                          BOPD per 1,000 BPD water
                        </label>
                      )}
                    </div>
                  </div>
                  {setpointPinnable && (
                    <div title="The booster header the run is planned at. Swept by default: the run tries the plant's pressure window and keeps the best. Pin it when the operator is already holding a setpoint; the run then evaluates that ONE pressure.">
                      <span className="text-xs font-medium text-slate-500">Header setpoint (psi)</span>
                      <div className="mt-1 flex h-8 items-center gap-2">
                        <label className="flex cursor-pointer items-center gap-1 text-xs text-slate-600">
                          <input
                            type="checkbox"
                            checked={autoSetpoint}
                            onChange={(e) => edit({ autoSetpoint: e.target.checked })}
                            className="h-4 w-4 rounded border-slate-300 accent-blue-600"
                          />
                          sweep
                        </label>
                        {!autoSetpoint && (
                          <input
                            aria-label="Header setpoint (psi)"
                            type="number"
                            value={manualSetpoint}
                            min={1000}
                            max={5000}
                            step={25}
                            onChange={(e) => edit({ manualSetpoint: Number(e.target.value) })}
                            className={INPUT_CLS}
                          />
                        )}
                      </div>
                    </div>
                  )}
                </>
              )}
              <button
                type="button"
                onClick={() => resetForm(runKey)}
                className="self-end rounded-md border border-slate-300 bg-white px-2 py-1 text-xs text-slate-600 hover:bg-slate-50"
              >
                Reset to defaults
              </button>
            </div>
          </details>
        )}

        <div className="flex flex-wrap items-center gap-3 border-t border-slate-100 pt-2.5">
          <button
            type="button"
            disabled={running || !offlineSettled || blockers.length > 0}
            onClick={run}
            className="flex items-center gap-1.5 rounded-md bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
          >
            <Play className="h-3.5 w-3.5" />
            {running ? "Running..." : `Run ${runKey} optimization`}
          </button>
          <CancelJobButton jobId={jobId} running={job.data?.status === "running"} />
          <span className="text-xs text-slate-500">
            {activeCount} active well{activeCount === 1 ? "" : "s"}
            {offline.length > 0 &&
              ` - ${offline.length} offline` +
                (autoCount > 0 ? ` (${autoCount} by default: shut in or recycle)` : "")}
            {future.length > 0 && ` - ${future.length} future`}
            {" - models from saved fits (set them on the Single Well solver)"}
          </span>
        </div>
        {!offlineSettled && <p className="text-xs text-slate-500">Loading the downtime log so shut-in wells are excluded...</p>}
        {blockers.map((b) => <WarnNote key={b}>{b}</WarnNote>)}
        {running && job.data && (
          <div className="space-y-1">
            <div className="h-1.5 w-full overflow-hidden rounded bg-slate-100">
              <div
                className={clsx("h-full rounded bg-blue-500 transition-all", progress.fraction === null && "w-1/3 animate-pulse")}
                style={progress.fraction !== null ? { width: `${Math.max(3, progress.fraction * 100)}%` } : undefined}
              />
            </div>
            <p className="text-xs text-slate-500">
              {job.data.progress} ({fmtNum(job.data.seconds)} s
              {progress.etaSeconds !== null && `, about ${fmtDuration(progress.etaSeconds)} left`})
            </p>
          </div>
        )}
        {job.data?.status === "error" && <WarnNote>Run failed: {job.data.error}</WarnNote>}
        <CancelledNote job={job.data} />
        {expired && <WarnNote>The previous run is no longer on the server (restarted, or finished over an hour ago). Run again to see a result.</WarnNote>}
        {offlineFailed && <WarnNote>The downtime log did not load, so shut-in wells are not excluded automatically. Tick them offline on the readiness board before running.</WarnNote>}
        {start.isError && <WarnNote>Could not start the run: {start.error.message}</WarnNote>}
        {changed.length > 0 && !running && (
          <WarnNote>
            Inputs changed since this result was run ({changed.join("; ")}). The result below
            describes the earlier inputs - run again to update it.
          </WarnNote>
        )}
        {job.data?.started_at && hasResult && (
          <p className="text-xs text-slate-400">
            Result from the run started {job.data.started_at.replace("T", " ")} ({fmtNum(job.data.seconds)} s).
          </p>
        )}
      </Card>

      {running && !job.data?.progress && <Spinner label="Starting run" />}
      {padResult !== null && <PadResults result={padResult} />}
      {padResult !== null && jobId && <PlanRobustness key={jobId} sourceJobId={jobId} available={padResult.robustness_available === true} unavailableReason={padResult.robustness_unavailable_reason} />}
      {chokeResult !== null && <ChokePlanResults result={chokeResult} />}
      {cfpResult !== null && <CfpResults result={cfpResult} />}
      {kind === "pad" && pad && (
        // Plant curves left, readiness right (below the answer once there
        // is one): the plant and the wells feeding it read together, and
        // the board stays editable for the next run. min-w-0 lets the chart
        // half shrink.
        <div className={clsx("grid gap-4", aside && "xl:grid-cols-2")}>
          <div className="min-w-0">
            <PadCharts
              pad={pad}
              result={padResult ?? chokeResult}
              nPumps={nPumps}
              ePad={ePadKnobsOf(padResult ?? chokeResult) ?? ePadKnobs}
            />
          </div>
          {aside && <div className="min-w-0">{aside}</div>}
        </div>
      )}
    </div>
  );
}

/** The booster a result actually ran on (E-Pad echoes it in meta), so the
 *  chart under an old result never redraws the form's current booster. */
function ePadKnobsOf(result: { meta: Record<string, unknown> } | null) {
  const cfg = result?.meta?.e_pad;
  if (cfg === null || typeof cfg !== "object") return null;
  const c = cfg as Record<string, unknown>;
  if (typeof c.build !== "string" || typeof c.suction_psi !== "number" || typeof c.hz_max !== "number"
    || typeof c.max_header_psi !== "number") return null;
  return {
    build: c.build as EPadBuild,
    suctionPsi: c.suction_psi,
    hzMax: c.hz_max,
    maxHeaderPsi: c.max_header_psi,
    ampLimitA: typeof c.amp_limit_a === "number" ? c.amp_limit_a : null,
  };
}
