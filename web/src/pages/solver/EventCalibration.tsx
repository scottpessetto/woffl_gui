/**
 * "Calibrate to field data" - the solver page's ONE calibration action.
 * The server tries the multi-point era fit first: the full knob set
 * (ken/kth/kdi + nozzle washout + choking Mach) against the installed pump
 * era's daily (PF pressure, BHP, PF rate) history, as a background job
 * (POST /optimize/event-calibration, polled through the shared
 * /optimize/run/{job_id} monitor like match-health). When the era is too
 * young to identify anything (builder refusal), the server falls back to
 * the single-point latest-test BHP match (the old Auto-match BHP
 * mechanics) and the payload says so: method "single_point" +
 * fallback_reason. The result renders summary-first in plain language.
 *
 * Nothing lands on the sidebar automatically: "Apply to inputs" lays the
 * result over the params store via setMany, so the applied fields become
 * engineer-owned (manualFields) and the open-time IPR fit stops
 * overwriting them. An event fit applies all five knobs; a single-point
 * match applies ken/kth/kdi ONLY - one BHP point cannot see nozzle wear or
 * the choking Mach, so those stay untouched.
 */

import { Activity } from "lucide-react";
import { useEffect, useState } from "react";

import { useOptimizeJob, useStartEventCalibration } from "../../api/hooks";
import type { EventCalibrationResult } from "../../api/types";
import { Button } from "../../components/ui";
import { fmtNum } from "../../lib/format";
import { useParamsStore } from "../../state/params";

/** Coefficient in the scorecard's shorthand: 3 decimals, trailing zeros and
 *  the leading "0" dropped - 0.024 -> ".024", 0.240 -> ".24". */
function coef(v: number): string {
  let s = v.toFixed(3).replace(/(\.\d*?)0+$/, "$1").replace(/\.$/, "");
  if (s.startsWith("0.")) s = s.slice(1);
  return s;
}

/** |model - measured| within 0.03 psi/psi counts as reproduced. */
const BETA_TOL = 0.03;

/** The young-era fallback: the server matched the latest test's BHP
 * (single-point) because the era fit was impossible. Amber-tinted so it
 * reads as "provisional", with the unlock condition spelled out. */
function SinglePointBlock({ result }: { result: EventCalibrationResult }) {
  const setMany = useParamsStore((s) => s.setMany);
  const single = result.single;

  if (!single) {
    return (
      <p className="basis-full text-xs text-amber-700">
        Fallback calibration returned no result - check the server logs.
      </p>
    );
  }

  const pinned = single.match_quality === "pinned";
  const failed = single.match_quality === "failed";
  const headline =
    `Young pump era - ${result.fallback_reason ?? "not enough era history"}. ` +
    `Matched the latest test BHP instead (single-point): modeled ` +
    `${fmtNum(single.modeled_bhp)} vs target ${fmtNum(single.target_bhp)} psi ` +
    `(${single.match_quality}). Full field calibration unlocks as this pump ` +
    "accumulates daily history.";

  const applyTitle = pinned
    ? "Nothing was fitted - the coefficients came back at their seeds, so there is nothing to apply."
    : failed
      ? "The solver found no valid operating point, so there is nothing to apply."
      : "Lay the matched ken/kth/kdi over the sidebar. Nozzle wear and choking Mach " +
        "are untouched - a single-point match cannot see them. Save as well default to keep them.";

  return (
    <div className="basis-full space-y-1 rounded-md border border-amber-200 bg-amber-50 px-2.5 py-2">
      <p className="text-xs text-amber-800">{headline}</p>
      {pinned && single.message && (
        <p className="text-xs text-amber-700">Not calibrated - {single.message}</p>
      )}
      {failed && (
        <p className="text-xs text-amber-700">
          Calibration failed - the solver found no valid operating point at any friction setting.
        </p>
      )}
      <p className="font-mono text-[11px] text-slate-500">
        {`ken ${coef(single.ken)} | kth ${coef(single.kth)} | kdi ${coef(single.kdi)}`}
      </p>
      <Button
        variant="secondary"
        size="sm"
        disabled={pinned || failed}
        title={applyTitle}
        onClick={() => setMany({ ken: single.ken, kth: single.kth, kdi: single.kdi })}
      >
        Apply to inputs
      </Button>
    </div>
  );
}

function ResultBlock({ result }: { result: EventCalibrationResult }) {
  const setMany = useParamsStore((s) => s.setMany);
  const fit = result.fit;

  if (result.method === "single_point") return <SinglePointBlock result={result} />;

  if (result.refusal) {
    return <p className="basis-full text-xs text-amber-700">Not calibrated - {result.refusal}</p>;
  }
  if (!fit) {
    return (
      <p className="basis-full text-xs text-amber-700">
        Calibration returned no fit and no reason - check the server logs.
      </p>
    );
  }

  const eraStart = result.era_start ? result.era_start.slice(0, 10) : "era start";
  const wearPct = (fit.fnz - 1) * 100;
  const wearPhrase =
    wearPct >= 0.5
      ? `nozzle ${wearPct.toFixed(0)}% washed out`
      : wearPct <= -0.5
        ? `nozzle ${Math.abs(wearPct).toFixed(0)}% restricted`
        : "nozzle at catalog size";
  const headline =
    `Matched ${fit.n_used} days of this pump's history (${eraStart} - today): ` +
    `${wearPhrase}, model tracks measured BHP within ${Math.round(fit.rms_bhp_psi)} psi.`;

  const paramsLine =
    `ken ${coef(fit.ken)} | kth ${coef(fit.kth)} | kdi ${coef(fit.kdi)} | ` +
    `nozzle area ${fit.fnz.toFixed(2)} | mach crit ${fit.mach_crit.toFixed(2)}`;

  const qualityLine =
    `RMS BHP ${Math.round(fit.rms_bhp_psi)} psi | PF ${fit.rms_pf_pct.toFixed(1)}%` +
    (fit.rms_dbhp_psi !== null ? ` | dBHP ${Math.round(fit.rms_dbhp_psi)} psi` : "") +
    ` | ${fit.n_used} points (${result.n_daily} daily / ${result.n_test} tests, ` +
    `spread ${Math.round(result.ppf_spread)} psi)` +
    (fit.n_dropped > 0 ? ` - ${fit.n_dropped} dropped` : "");

  const modelBeta = fit.implied_beta;
  const minedBeta = result.mined_beta;
  const betaKnown = modelBeta !== null && minedBeta !== null;
  const betaOk = betaKnown && Math.abs(modelBeta - minedBeta) <= BETA_TOL;

  const cur = result.current;
  const curParts = (["ken", "kth", "kdi"] as const)
    .filter((k) => cur[k] !== null)
    .map((k) => `${k} ${coef(cur[k] as number)}`);
  const applyTitle =
    "Lay the fitted coefficients, nozzle area factor and critical Mach over the sidebar" +
    (curParts.length ? ` (replaces ${curParts.join(" / ")})` : "") +
    ". Save as well default to keep them.";

  return (
    <div className="basis-full space-y-1 rounded-md border border-slate-200 bg-slate-50 px-2.5 py-2">
      <p className="text-xs text-slate-700">{headline}</p>
      {(modelBeta !== null || minedBeta !== null) && (
        <p className="text-xs text-slate-700">
          response: model {modelBeta !== null ? modelBeta.toFixed(3) : "n/a"} vs measured{" "}
          {minedBeta !== null ? minedBeta.toFixed(3) : "n/a"}
          {result.mined_beta_source ? ` (${result.mined_beta_source})` : ""}
          {betaOk && <span className="ml-1 text-emerald-700">{"\u2713"} reproduced</span>}
        </p>
      )}
      {betaKnown && !betaOk && (
        <p className="text-xs text-amber-700">
          response not reproduced - treat suction sensitivity as evidence-layer
        </p>
      )}
      <p className="font-mono text-[11px] text-slate-500">{paramsLine}</p>
      <p className="font-mono text-[11px] text-slate-500">{qualityLine}</p>
      {fit.railed.length > 0 && (
        <p className="text-xs text-amber-700">
          railed on a search bound: {fit.railed.join(", ")} - treat as low confidence
        </p>
      )}
      {fit.message && <p className="text-xs text-slate-500">{fit.message}</p>}
      <Button
        variant="secondary"
        size="sm"
        title={applyTitle}
        onClick={() =>
          setMany({
            ken: fit.ken,
            kth: fit.kth,
            kdi: fit.kdi,
            nozzle_area_factor: fit.fnz,
            mach_crit: fit.mach_crit,
          })
        }
      >
        Apply to inputs
      </Button>
    </div>
  );
}

export function EventCalibration({ well }: { well: string }) {
  const [jobId, setJobId] = useState<string | null>(null);
  const start = useStartEventCalibration();
  const job = useOptimizeJob(jobId);

  // Expired job (server restart): drop the stale id quietly.
  useEffect(() => {
    if (jobId && job.isError) setJobId(null);
  }, [jobId, job.isError]);

  // Fresh state per well - one well's fit must not render under another.
  useEffect(() => {
    setJobId(null);
  }, [well]);

  // A bench with no named well has no era history to fit against.
  if (well === "Custom") return null;

  const running = start.isPending || job.data?.status === "running";
  const result =
    job.data?.status === "done" && job.data.kind === "event_cal"
      ? (job.data.result as EventCalibrationResult | null)
      : null;

  return (
    <>
      <Button
        variant="primary"
        size="sm"
        disabled={running}
        busy={running}
        title={
          "Fits the pump model to this pump era's daily field history; " +
          "young eras fall back to matching the latest test BHP."
        }
        onClick={() => {
          start.mutate({ well }, { onSuccess: (r) => setJobId(r.job_id) });
        }}
      >
        <span className="flex items-center gap-1.5">
          <Activity className="h-3.5 w-3.5" />
          {running ? "Calibrating..." : "Calibrate to field data"}
        </span>
      </Button>
      {running && (
        <span className="text-xs text-slate-500">
          {job.data?.progress ?? "Starting calibration..."}
        </span>
      )}
      {start.isError && (
        <span className="basis-full text-xs text-amber-700">
          Could not start calibration: {start.error.message}
        </span>
      )}
      {job.data?.status === "error" && (
        <span className="basis-full text-xs text-amber-700">
          Calibration failed: {job.data.error}
        </span>
      )}
      {result !== null && <ResultBlock result={result} />}
    </>
  );
}
