/**
 * Match-health scorecard - the model-vs-field picture per well, one pad.
 *
 * One background job (same registry and polling as optimization runs)
 * models every active well at its CURRENT pump and lays the model against
 * everything the field measured: recent tests, the mined suction floor and
 * response slope, the fit provenance and the friction-calibration rails.
 * The verdict chip compresses each well to one word so an engineer can see
 * at a glance which matches to fix before trusting an optimizer run.
 */

import { HeartPulse } from "lucide-react";
import { useEffect } from "react";

import { useOptimizeJob, useStartMatchHealth } from "../../api/hooks";
import type { MatchHealthResult, MatchHealthRow, MatchHealthVerdict, RunPad } from "../../api/types";
import { Badge, Card, Spinner, WarnNote } from "../../components/ui";
import { fmtNum } from "../../lib/format";
import { useOptimizeStore } from "../../state/optimize";

const TH_CLS = "px-2 py-1.5 text-left font-semibold";
const TD_CLS = "px-2 py-1 tabular-nums";

const VERDICT_META: Record<
  MatchHealthVerdict,
  { tone: "neutral" | "good" | "fair" | "poor" | "info"; hint: string }
> = {
  contradicted: {
    tone: "fair",
    hint:
      "Field data contradicts the model: the modeled cavitation floor sits above " +
      "the measured BHP floor, or the well's own measured response slope shows it " +
      "responding to PF while the model claims it is sonic-pinned.",
  },
  "railed-cal": {
    tone: "fair",
    hint:
      "Friction calibration railed at the bound box corner (ken at max, kth/kdi at min) - " +
      "the degenerate fit that writes the calibration-day gauge BHP into the floor.",
  },
  "weak-fit": {
    tone: "neutral",
    hint: "Saved inflow fit has r2 below 0.5 - the model is only as good as a sketch.",
  },
  ok: { tone: "good", hint: "No contradiction, rail or weak fit detected." },
};

function VerdictChip({ verdict }: { verdict: MatchHealthVerdict }) {
  const meta = VERDICT_META[verdict] ?? VERDICT_META.ok;
  return (
    <Badge tone={meta.tone} title={meta.hint}>
      {verdict}
    </Badge>
  );
}

/** "saved 0.93" / "auto 0.41" / "-" - the inflow curve's provenance. */
function fitLabel(row: MatchHealthRow): string {
  if (!row.ipr_source && row.ipr_r2 === null) return "-";
  const r2 = row.ipr_r2 !== null ? ` ${fmtNum(row.ipr_r2, 2)}` : "";
  return `${row.ipr_source ?? "?"}${r2}`;
}

/** "1,020 / 940 (1.09)" - model vs test with the ratio the flags band on. */
function ratioCell(model: number | null, test: number | null, ratio: number | null): string {
  if (model === null && test === null) return "-";
  const pair = `${fmtNum(model)} / ${fmtNum(test)}`;
  return ratio !== null ? `${pair} (${fmtNum(ratio, 2)})` : pair;
}

/** Model floor vs measured floor; the violation is what the verdict bands on. */
function floorCell(row: MatchHealthRow): string {
  if (row.model_psu === null && row.evidence_floor === null) return "-";
  const pair = `${fmtNum(row.model_psu)} / ${fmtNum(row.evidence_floor)}`;
  return row.floor_violation !== null
    ? `${pair} (${row.floor_violation > 0 ? "+" : ""}${fmtNum(row.floor_violation)})`
    : pair;
}

function betaCell(row: MatchHealthRow): string {
  if (row.beta === null) return "-";
  const src = row.beta_source ?? "?";
  const pairs = row.n_pairs !== null ? `, ${fmtNum(row.n_pairs)} pairs` : "";
  return `${fmtNum(row.beta, 3)} (${src}${pairs})`;
}

function FrictionCell({ row }: { row: MatchHealthRow }) {
  if (row.ken === null && row.kth === null && row.kdi === null) {
    return <span className="text-slate-400">-</span>;
  }
  const railed = row.ken_railed || row.kth_railed || row.kdi_railed;
  return (
    <span className="inline-flex items-center gap-1.5">
      <span>
        {fmtNum(row.ken, 3)} / {fmtNum(row.kth, 2)} / {fmtNum(row.kdi, 2)}
      </span>
      {railed && (
        <Badge
          tone="fair"
          title={
            "Calibration sits on its bounds: " +
            [
              row.ken_railed ? "ken at max" : null,
              row.kth_railed ? "kth at min" : null,
              row.kdi_railed ? "kdi at min" : null,
            ]
              .filter(Boolean)
              .join(", ")
          }
        >
          railed
        </Badge>
      )}
    </span>
  );
}

export function MatchHealthPanel({ pad }: { pad: RunPad }) {
  // The job id lives in the persisted optimize store, keyed PER PAD, so a
  // tab switch or a page change mid-scorecard no longer orphans a running
  // job and its result (review 2026-09-01, WEB-8). Per-pad keys also mean a
  // stale M scorecard can never render under S.
  const jobKey = `match_health:${pad}`;
  const jobId = useOptimizeStore((s) => s.lastJob[jobKey] ?? null);
  const setLastJob = useOptimizeStore((s) => s.setLastJob);
  const setJobId = (id: string | null) => setLastJob(jobKey, id);
  const start = useStartMatchHealth();
  const job = useOptimizeJob(jobId);

  // Expired job (server restart): drop the stale id quietly.
  useEffect(() => {
    if (jobId && job.isError) setLastJob(jobKey, null);
  }, [jobId, job.isError, jobKey, setLastJob]);

  const running = job.data?.status === "running" || start.isPending;
  const result =
    job.data?.status === "done" && job.data.kind === "match_health"
      ? (job.data.result as MatchHealthResult | null)
      : null;

  return (
    <div className="space-y-2">
      <h2 className="text-sm font-semibold tracking-tight text-slate-700">
        {pad}-Pad match health
      </h2>
      <Card className="space-y-3">
        <div className="flex items-center gap-3">
          <button
            type="button"
            disabled={running}
            onClick={() =>
              start.mutate({ pad }, { onSuccess: (r) => setJobId(r.job_id) })
            }
            className="flex items-center gap-1.5 rounded-md bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
          >
            <HeartPulse className="h-3.5 w-3.5" />
            {running ? "Checking..." : `Check ${pad}-Pad match health`}
          </button>
          <span className="text-xs text-slate-500">
            Every active well at its current pump: model vs tests, measured floors and
            response slopes, calibration rails - the matches to fix before trusting a run.
          </span>
        </div>
        {running && job.data?.progress && (
          <p className="text-xs text-slate-500">
            {job.data.progress} ({fmtNum(job.data.seconds)}s)
          </p>
        )}
        {running && !job.data?.progress && <Spinner label="Starting scorecard" />}
        {job.data?.status === "error" && (
          <WarnNote>Scorecard failed: {job.data.error}</WarnNote>
        )}
        {start.isError && (
          <WarnNote>Could not start the scorecard: {start.error.message}</WarnNote>
        )}

        {result !== null && (
          <>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[56rem] border-collapse text-[13px]">
                <thead>
                  <tr className="border-b border-slate-200 text-xs text-slate-500">
                    <th className={TH_CLS}>Well</th>
                    <th className={TH_CLS}>Pump</th>
                    <th className={TH_CLS} title="Inflow-curve provenance + r2 of the saved fit">
                      Fit
                    </th>
                    <th className={TH_CLS} title="Model oil vs median recent test oil, BOPD (ratio)">
                      Model/Test oil
                    </th>
                    <th className={TH_CLS} title="Model PF vs median recent test PF, BPD (ratio)">
                      Model/Test PF
                    </th>
                    <th
                      className={TH_CLS}
                      title="Modeled suction pressure vs measured BHP floor, psi (violation = model - measured; above +25 the model's floor claim is contradicted)"
                    >
                      Floor model/meas
                    </th>
                    <th
                      className={TH_CLS}
                      title="Measured suction response -dBHP/dPpf (source, event pairs behind it)"
                    >
                      beta
                    </th>
                    <th className={TH_CLS} title="Calibrated loss coefficients ken / kth / kdi">
                      Friction
                    </th>
                    <th className={TH_CLS}>Last test</th>
                    <th className={TH_CLS}>Verdict</th>
                  </tr>
                </thead>
                <tbody>
                  {result.rows.map((row) => (
                    <tr key={row.well} className="border-b border-slate-100 text-slate-700">
                      <td className={`${TD_CLS} font-medium`}>{row.well}</td>
                      <td className={TD_CLS}>{row.pump ?? "-"}</td>
                      <td className={TD_CLS}>{fitLabel(row)}</td>
                      <td className={TD_CLS} title={row.oil_flag ?? undefined}>
                        {ratioCell(row.model_oil, row.test_oil, row.model_test_oil_ratio)}
                      </td>
                      <td className={TD_CLS} title={row.pf_flag ?? undefined}>
                        {ratioCell(row.model_pf, row.test_pf, row.model_test_pf_ratio)}
                      </td>
                      <td className={TD_CLS}>
                        {floorCell(row)}
                        {row.sonic === true && (
                          <span
                            className="ml-1 text-slate-400"
                            title="Model reports this well sonic-pinned at its cavitation floor"
                          >
                            *
                          </span>
                        )}
                      </td>
                      <td className={TD_CLS}>{betaCell(row)}</td>
                      <td className={TD_CLS}>
                        <FrictionCell row={row} />
                      </td>
                      <td className={TD_CLS}>{row.last_test_date ?? "-"}</td>
                      <td className={TD_CLS}>
                        <VerdictChip verdict={row.verdict} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {result.notes.length > 0 && (
              <ul className="space-y-0.5 text-xs text-slate-500">
                {result.notes.map((n) => (
                  <li key={n}>{n}</li>
                ))}
              </ul>
            )}
            <p className="text-xs text-slate-400">
              Modeled at the plant-derived header {fmtNum(result.header_psi)} psi. * = model
              reports the well sonic-pinned. Floors and betas come from a year of daily gauge
              history; "-" means no data, never a passing grade.
            </p>
          </>
        )}
      </Card>
    </div>
  );
}
