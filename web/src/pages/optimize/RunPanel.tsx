import { isMissingJob } from "../../api/client";
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
import { fmtNum } from "../../lib/format";
import { DEFAULT_POPS_PADS } from "../../state/wellSort";
import { useOptimizeStore } from "../../state/optimize";

import { CfpResultCharts } from "./CfpCharts";
import { ChokeDumbbell, IprLandingTable } from "./ChokeCharts";
import { usePadOffline } from "./offline";
import { PadCharts } from "./PadCharts";
import { unselectedOutcomeLabel } from "./outcomes";
import { PlanRobustness } from "./PlanRobustness";

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
function FitSource({ row }: { row: Pick<PadRunRow, "ipr_source" | "ipr_r2" | "has_friction" | "pump_calibration" | "hydraulics_model"> }) {
  const r2 = row.ipr_r2;
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
      ? (meta.solver_agreement as { agree?: boolean; mckp_objective?: number; milp_objective?: number; error?: string })
      : null;
  const totalTestOil = result.rows.reduce((a, r) => a + (r.test_oil ?? 0), 0);
  return (
    <div className="space-y-3">
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
        <Metric label="Current pumps at plan header" value={`${fmtNum(metaNum(meta, "current_model_oil_bopd"))} BOPD`} />
        <Metric label="Modeled hardware gain" value={`${fmtNum(metaNum(meta, "modeled_hardware_gain_bopd"))} BOPD`} />
        <Metric label="Recent test oil (context)" value={`${fmtNum(totalTestOil)} BOPD`} title="Sum of recent positive-test medians; test dates can differ. This is not the modeled baseline or an optimization gain." />
        <Metric
          label="Water price"
          value={lam === null ? "-" : `${fmtNum(lam * 1000, 1)} BOPD/MBPD`}
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
      {meta.feasible === false && <WarnNote>This plan does not meet all modeled operating limits. Resolve the conditions below before treating it as an operating recommendation.</WarnNote>}
      {(meta.in_range === false || meta.recirc === true) && <WarnNote>Machine flow is outside the modeled operating range or requires recirculation. Net well demand alone does not establish a valid machine operating point.</WarnNote>}
      {Array.isArray(meta.operating_assumptions) && meta.operating_assumptions.map((note, i) =>
        <WarnNote key={i}>{String(note)}</WarnNote>)}
      {allocation && <p className="text-xs text-slate-500">
        Allocation status: {String(allocation.status ?? "unknown")} ({String(allocation.solver ?? "unspecified solver")}).
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
              <th className="px-2 py-1.5 text-left font-semibold">Current</th>
              <th className={TH_CLS}>Test oil</th>
              <th className={TH_CLS}>Test PF</th>
              <th className={TH_CLS} title="Installed pump modeled at the proposal header with the same saved well inputs">Current model oil</th>
              <th className="px-2 py-1.5 text-left font-semibold">Plan</th>
              <th className={TH_CLS}>Oil</th>
              <th className={TH_CLS}>Hardware gain</th>
              <th className={TH_CLS}>PF</th>
              <th className={TH_CLS}>Suction</th>
              <th className={TH_CLS}>Marginal</th>
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
                        {r.pump_state && <span className="block text-[10px] font-normal">{r.pump_state === "replacement" ? "Replace ? clean reference" : "Keep installed"}</span>}
                        {r.sonic && <span title="sonic throat"> *</span>}
                      </span>
                    )}
                  </td>
                  <td className={clsx(TD_CLS, "text-slate-700")}>{fmtNum(r.oil)}</td>
                  <td className={TD_CLS}>{fmtNum(r.modeled_hardware_gain ?? null)}</td>
                  <td className={clsx(TD_CLS, "text-slate-600")}>{fmtNum(r.pf)}</td>
                  <td className={clsx(TD_CLS, "text-slate-500")}>{fmtNum(r.suction)}</td>
                  <td className={clsx(TD_CLS, "text-slate-500")}>{fmtNum(r.marginal_oil, 2)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </Card>

      {agreement && agreement.error && (
        <WarnNote>MILP cross-check failed: {agreement.error}</WarnNote>
      )}
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
  const projD = meta.feasible === false ? null : metaNum(meta, "projected_d_oil_bopd");
  const headerToday = metaNum(meta, "header_today_psi");
  return (
    <div className="space-y-3">
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

      {meta.feasible === false && <WarnNote>This choke plan does not meet all modeled operating limits. Its projected gain is withheld.</WarnNote>}
      {Array.isArray(meta.operating_assumptions) && meta.operating_assumptions.map((note, i) =>
        <WarnNote key={i}>{String(note)}</WarnNote>)}
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
                <tr key={`${m.type}-${m.well}`} className="border-b border-slate-100 last:border-b-0">
                  <td className="px-2 py-1 text-left font-medium text-slate-700">{MOVE_LABELS[m.type]}</td>
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

      {s.pairs.length > 0 && <Card padded={false} className="overflow-x-auto">
        <p className="px-3 pt-2 text-xs font-semibold text-slate-600">Bring online with an offset</p>
        <p className="px-3 py-1 text-xs text-slate-500">Jointly evaluated changes, including wells that cannot operate alone. Net oil includes the pressure response of existing wells.</p>
        <table className="w-full border-collapse text-[13px]"><thead><tr className="border-b border-slate-200">
          <th className="px-3 py-2 text-left">Bring online</th><th className="px-3 py-2 text-left">Offset</th>
          <th className={TH_CLS}>Net oil (BOPD)</th><th className={TH_CLS}>Discharge after</th>
        </tr></thead><tbody>{s.pairs.map((pair, i) => <tr key={`${pair.bring_on.well}-${pair.offset.well}-${i}`} className="border-b border-slate-100">
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
  pad: RunPad | null;
  aside?: ReactNode;
}) {
  const runKey = kind === "cfp" ? "CFP" : (pad as string);
  const wells = useWells();
  const futureByPad = useOptimizeStore((s) => s.future);
  const requiredByPad = useOptimizeStore((s) => s.requiredOnline);
  const lastJob = useOptimizeStore((s) => s.lastJob);
  const setLastJob = useOptimizeStore((s) => s.setLastJob);

  // Mirrors server/schemas.py OptimizeRunRequest.nozzles - the client always
  // sends the list, so an omission here silently drops a size from every run.
  const [nozzles, setNozzles] = useState(["9", "10", "11", "12", "13", "14", "15"]);
  const [throats, setThroats] = useState(["A", "B", "C", "D"]);
  const [method, setMethod] = useState<"milp" | "mckp">("milp");
  const [strategy, setStrategy] = useState<"jpco" | "choke">("jpco");
  // The default maximizes oil within capacity. A manual water price changes
  // that objective; its starting value is 0.02 BOPD/BPD = 20 BOPD/MBPD.
  const [autoLam, setAutoLam] = useState(true);
  const [manualLam, setManualLam] = useState(0.02);
  // Header setpoint for the free-pressure pads (I/M/E). auto = sweep the
  // pressure window and let the run pick; unchecked pins it to one trial.
  // The S-Pad's header is a function of flow, so it has no setpoint to pin
  // and the control is never rendered there.
  const [autoSetpoint, setAutoSetpoint] = useState(true);
  const [manualSetpoint, setManualSetpoint] = useState(3200);
  const [nPumps, setNPumps] = useState<number | null>(null);
  const [p0, setP0] = useState(2792);
  const [referencePadPf, setReferencePadPf] = useState<Record<string, string>>({});
  const [slope, setSlope] = useState(13.69);
  const [cPadPf, setCPadPf] = useState(3400);
  const [cfpPads, setCfpPads] = useState<string[]>([...CFP_PADS]);
  useEffect(() => { if (pad === "S") setStrategy("jpco"); }, [pad]);
  // E-Pad booster configuration. Defaults mirror server.schemas
  // OptimizeRunRequest: the Summit workbook's suction cell and I-Pad's
  // operational header cap, neither of them an E-Pad measurement.
  const [ePadBuild, setEPadBuild] = useState<EPadBuild>("SM25000_26STG");
  const [ePadSuction, setEPadSuction] = useState(2800);
  const [ePadHzMax, setEPadHzMax] = useState(60);
  const [ePadHeaderCap, setEPadHeaderCap] = useState(3500);
  const [ePadAmpLimit, setEPadAmpLimit] = useState("");
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
    () => runPads.flatMap((p) => (futureByPad[p] ?? []).map((fw) => ({ ...fw, pad: p }))),
    [runPads, futureByPad],
  );
  const activeCount = useMemo(() => {
    const names = (wells.data?.wells ?? []).filter((w) => runPads.includes(w.pad)).map((w) => w.name);
    return names.filter((n) => !offlineSet.has(n)).length;
  }, [wells.data, runPads, offlineSet]);

  // The booster configuration the E-Pad curve sheet and the run must agree
  // on. Ignored (and server-rejected) on every other pad.
  const ePadKnobs = useMemo(
    () => ({
      build: ePadBuild,
      suctionPsi: ePadSuction,
      hzMax: ePadHzMax,
      maxHeaderPsi: ePadHeaderCap,
    }),
    [ePadBuild, ePadSuction, ePadHzMax, ePadHeaderCap],
  );
  const ePadAmpLimitNum = Number(ePadAmpLimit);
  const ePadAmpLimitReq =
    ePadAmpLimit.trim() === "" || !Number.isFinite(ePadAmpLimitNum) || ePadAmpLimitNum <= 0
      ? null
      : ePadAmpLimitNum;

  // Only the free-pressure pads (I/M/E) have a header to pin, and only a
  // JPCO run sweeps one. Everything else sends null - a swept run.
  const setpointPinnable = kind === "pad" && strategy === "jpco" && pad !== "S";
  const setpointReq = setpointPinnable && !autoSetpoint ? manualSetpoint : null;

  // The plant's selectable online-pump counts, off the (hard-cached) curve
  // payload; [] = fixed train (I/E-Pad) or a CFP run - no control rendered.
  const pumpCurve = usePumpCurve(kind === "pad" ? pad : null, null, ePadKnobs);
  const pumpOptions = pumpCurve.data?.n_pump_options ?? [];

  const start = useStartOptimizeRun();
  const jobId = lastJob[runKey] ?? null;
  const job = useOptimizeJob(jobId);

  // Expired job (server restart): drop the stale id quietly.
  useEffect(() => {
    if (jobId && isMissingJob(job.error)) setLastJob(runKey, null);
  }, [jobId, job.error, runKey, setLastJob]);

  const running = job.data?.status === "running" || start.isPending;

  const run = () => {
    const req: OptimizeRunRequest = {
      kind,
      pad,
      offline,
      future,
      required_wells: runPads.flatMap((p) => requiredByPad[p] ?? []),
      nozzles,
      throats,
      method,
      strategy,
      lambda_bopd_per_bpd: autoLam ? null : manualLam,
      marginal_wc: null,
      parsimony_bopd: 0,
      n_pumps: nPumps,
      n_steps: null,
      setpoint_psi: setpointReq,
      p0_psi: p0,
      psi_per_kbpd: slope,
      c_pad_pf_psi: cPadPf,
      cfp_pad_pf_psi: Object.fromEntries(Object.entries(referencePadPf).filter(([, value]) => value.trim() !== "").map(([p, value]) => [p, Number(value)])),
      cfp_pads: cfpPads,
      e_pad_build: ePadBuild,
      e_pad_suction_psi: ePadSuction,
      e_pad_hz_max: ePadHzMax,
      e_pad_max_header_psi: ePadHeaderCap,
      e_pad_amp_limit_a: ePadAmpLimitReq,
    };
    start.mutate(req, { onSuccess: (r) => setLastJob(runKey, r.job_id) });
  };

  const result = job.data?.status === "done" ? job.data.result : null;
  // "meta" too: the match-health scorecard payload also carries "rows".
  const padResult = result !== null && "rows" in result && "meta" in result ? result : null;
  const chokeResult = result !== null && "plan" in result ? result : null;
  const chokeMode = kind === "pad" && strategy === "choke";

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
                onChange={(e) => setStrategy(e.target.value === "choke" ? "choke" : "jpco")}
                className="mt-1 block h-8 rounded-md border border-slate-300 bg-white px-2 text-sm"
              >
                <option value="jpco">Resize pumps (JPCO)</option>
                <option value="choke" disabled={pad === "S"}>Choke / shut in (hold pumps){pad === "S" ? " - unavailable on S" : ""}</option>
              </select>
            </label>
          )}
          {!chokeMode && (
            <>
              <div>
                <p className="mb-1 text-xs font-medium text-slate-500">Nozzles</p>
                <ChipToggle options={NOZZLE_OPTIONS} selected={nozzles} onChange={setNozzles} />
              </div>
              <div>
                <p className="mb-1 text-xs font-medium text-slate-500">Throats</p>
                <ChipToggle options={THROAT_OPTIONS} selected={throats} onChange={setThroats} />
              </div>
            </>
          )}
          {kind === "pad" && pumpOptions.length > 0 && (
            <div title="Booster pumps online for this run - drop it when a machine is down (e.g. one M-Pad HP pump out). The PF budget, capability frontier and min-flow floor all follow the selected count.">
              <p className="mb-1 text-xs font-medium text-slate-500">Pumps online</p>
              <PumpCountChips options={pumpOptions} selected={nPumps} onChange={setNPumps} />
            </div>
          )}
          {kind === "pad" ? (
            strategy === "jpco" && (
            <>
              <label className="block">
                <span className="text-xs font-medium text-slate-500">Solver</span>
                <select
                  value={method}
                  onChange={(e) => setMethod(e.target.value === "mckp" ? "mckp" : "milp")}
                  className="mt-1 block h-8 rounded-md border border-slate-300 bg-white px-2 text-sm"
                >
                  <option value="milp">MILP</option>
                  <option value="mckp">MCKP (CP-SAT)</option>
                </select>
              </label>
              <div title="Default: maximize oil within the plant's machine-water capacity. Uncheck to apply an additional deliberate oil-equivalent water cost.">
                <span className="text-xs font-medium text-slate-500">Allocation objective</span>
                <div className="mt-1 flex h-8 items-center gap-2">
                  <label className="flex cursor-pointer items-center gap-1 text-xs text-slate-600">
                    <input
                      type="checkbox"
                      checked={autoLam}
                      onChange={(e) => setAutoLam(e.target.checked)}
                      className="h-4 w-4 rounded border-slate-300 accent-blue-600"
                    />
                    Maximize oil within capacity
                  </label>
                  {!autoLam && (
                    <input
                      aria-label="Manual water price (BOPD/BPD)"
                      type="number"
                      value={manualLam}
                      min={0}
                      max={10}
                      step={0.005}
                      onChange={(e) => setManualLam(Number(e.target.value))}
                      className={INPUT_CLS}
                    />
                  )}
                </div>
              </div>
              {setpointPinnable && (
                <div title="The booster header the run is planned at. auto sweeps the plant's pressure window and keeps the best trial; pin it when the header is already decided (a setpoint the operator is holding) and the run evaluates that ONE pressure instead. S-Pad is a fixed-curve station - its header follows the flow - so it has no setpoint to pin.">
                  <span className="text-xs font-medium text-slate-500">Header setpoint (psi)</span>
                  <div className="mt-1 flex h-8 items-center gap-2">
                    <label className="flex cursor-pointer items-center gap-1 text-xs text-slate-600">
                      <input
                        type="checkbox"
                        checked={autoSetpoint}
                        onChange={(e) => setAutoSetpoint(e.target.checked)}
                        className="h-4 w-4 rounded border-slate-300 accent-blue-600"
                      />
                      auto (sweep)
                    </label>
                    {!autoSetpoint && (
                      <input
                        type="number"
                        value={manualSetpoint}
                        min={1000}
                        max={5000}
                        step={25}
                        onChange={(e) => setManualSetpoint(Number(e.target.value))}
                        className={INPUT_CLS}
                      />
                    )}
                  </div>
                </div>
              )}
            </>
            )
          ) : (
            <>
              <div title="B/G/C/J use the CFP response model. Additional pads assume locally boosted PF; incremental routing from POPS pads is not qualified here.">
                <p className="mb-1 text-xs font-medium text-slate-500">CFP pads in the run</p>
                <ChipToggle options={cfpPadOptions} selected={cfpPads} onChange={setCfpPads} />
              </div>
              <label className="block">
                <span className="text-xs font-medium text-slate-500">Reference PW discharge (psi)</span>
                <input type="number" value={p0} min={2300} max={2880} step={5} onChange={(e) => setP0(Number(e.target.value))} className={clsx(INPUT_CLS, "mt-1 block")} />
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

        {kind === "cfp" && <details className="border-t border-slate-100 pt-2 text-xs text-slate-600">
          <summary className="cursor-pointer">Pad PF at the same reference conditions</summary>
          <p className="my-2">Enter pressures corresponding to the reference discharge and online configuration. Blank fields use fixed line-loss assumptions. The reference discharge is a manual scenario value.</p>
          <div className="flex flex-wrap gap-3">{["B", "G", "J"].filter((p) => cfpPads.includes(p)).map((p) => <label key={p}>
            {p}-Pad PF (psi)<input type="number" min={1000} max={5000} placeholder="assumed" value={referencePadPf[p] ?? ""}
              onChange={(e) => setReferencePadPf((prev) => ({ ...prev, [p]: e.target.value }))}
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
                onChange={(e) => setEPadBuild(e.target.value as EPadBuild)}
                className={clsx(INPUT_CLS, "mt-1 block w-56")}
              >
                <option value="SM25000_26STG">SM25000 - 26 stg (in well)</option>
                <option value="SN35000_18STG">SN35000 - 18 stg (alternative)</option>
              </select>
            </label>
            <label className="block" title="Booster suction (psig) - the upstream stage's discharge. 2,800 is the Summit workbook's cell, not a measured E-Pad tag.">
              <span className="text-xs font-medium text-slate-500">Suction (psig)</span>
              <input type="number" value={ePadSuction} min={0} max={5000} step={25} onChange={(e) => setEPadSuction(Number(e.target.value))} className={clsx(INPUT_CLS, "mt-1 block")} />
            </label>
            <label className="block" title="VFD speed cap. The stage curve is the 60 Hz catalog curve, so it cannot exceed 60.">
              <span className="text-xs font-medium text-slate-500">Max speed (Hz)</span>
              <input type="number" value={ePadHzMax} min={30} max={60} step={1} onChange={(e) => setEPadHzMax(Number(e.target.value))} className={clsx(INPUT_CLS, "mt-1 block")} />
            </label>
            <label className="block" title="Operational discharge cap on the PF header - piping/wellhead, not the pump. 3,500 is adopted from I-Pad pending an E-Pad number; the booster's own frontier peaks near 4,560 psi, so this cap is what limits the sweep.">
              <span className="text-xs font-medium text-slate-500">Header cap (psi)</span>
              <input type="number" value={ePadHeaderCap} min={1000} max={5000} step={25} onChange={(e) => setEPadHeaderCap(Number(e.target.value))} className={clsx(INPUT_CLS, "mt-1 block")} />
            </label>
            <label className="block" title="Motor amp cap. Blank enforces nothing - no E-Pad motor nameplate came with the vendor curve sheets.">
              <span className="text-xs font-medium text-slate-500">Amp limit (A)</span>
              <input type="number" value={ePadAmpLimit} min={1} max={5000} step={1} placeholder="none" onChange={(e) => setEPadAmpLimit(e.target.value)} className={clsx(INPUT_CLS, "mt-1 block")} />
            </label>
          </div>
        )}

        {kind === "pad" && pad === "E" && (
          <p className="text-xs text-amber-700">
            E-Pad&apos;s booster model is NOT validated against live SCADA - catalog stage
            curve plus the Summit workbook&apos;s affinity sheet. Suction and the header cap
            above are assumptions, and the header cap (not the pump) is what limits the
            pressure sweep. See the E-Pad booster tab for the candidate comparison.
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

        <div className="flex items-center gap-3 border-t border-slate-100 pt-2.5">
          <button
            type="button"
            disabled={running || (!chokeMode && (nozzles.length === 0 || throats.length === 0)) || (kind === "cfp" && cfpPads.length === 0)}
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
            <PadCharts
              pad={pad}
              result={padResult ?? chokeResult}
              nPumps={nPumps}
              ePad={ePadKnobs}
            />
          </div>
          {aside && <div className="min-w-0">{aside}</div>}
        </div>
      )}
      {padResult !== null && <PadResults result={padResult} />}
      {padResult !== null && jobId && <PlanRobustness key={jobId} sourceJobId={jobId} available={padResult.robustness_available === true} unavailableReason={padResult.robustness_unavailable_reason} />}
      {chokeResult !== null && <ChokePlanResults result={chokeResult} />}
      {result !== null && "summary" in result && <CfpResults result={result} />}
    </div>
  );
}
