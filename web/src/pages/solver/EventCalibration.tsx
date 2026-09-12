import { isMissingJob } from "../../api/client";
/**
 * "Calibrate to field data" - the solver page's ONE calibration action.
 * The server tries the multi-point era fit first: the full knob set
 * (ken/kth/kdi + estimated nozzle area) against the installed pump
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
 * overwriting them. An event fit applies four coefficients and resets the
 * retired Mach field to 1. A single-point match applies ken/kth/kdi only;
 * one BHP observation cannot determine nozzle area.
 */

import { Activity } from "lucide-react";
import { useEffect } from "react";

import { useMeta, useSavePumpCalibration, useOptimizeJob, useStartEventCalibration } from "../../api/hooks";
import { HYDRAULICS_LABELS, type HydraulicsModel, type EventCalibrationResult, type WellContext } from "../../api/types";
import { Button } from "../../components/ui";
import { fmtNum } from "../../lib/format";
import { useOptimizeStore } from "../../state/optimize";
import { useParamsStore } from "../../state/params";

function matchesInstallation(result: EventCalibrationResult, pump: WellContext["pump"] | undefined, model: HydraulicsModel) {
  return (result.hydraulics_model ?? "beggs") === model && !!pump?.date_set && pump.source === "databricks" &&
    `${pump.nozzle_no}${pump.throat_ratio}` === result.pump &&
    new Date(pump.date_set).getTime() === new Date(result.installation_date_set ?? result.era_start ?? "").getTime();
}

function SaveFit({ result, jobId }: { result: EventCalibrationResult; jobId: string }) {
  const context = useParamsStore((s) => s.context);
  const model = useParamsStore((s) => s.params.hydraulics_model);
  const meta = useMeta();
  const save = useSavePumpCalibration(result.well);
  const valid = matchesInstallation(result, context?.pump, model);
  const hasFit = !result.refusal && (result.fit || (result.single && !["pinned", "failed"].includes(result.single.match_quality)));
  if (!hasFit) return null;
  return <div className="basis-full space-y-1 text-xs text-slate-500">
    <Button size="sm" variant="secondary" disabled={!valid || !meta.data?.writes_enabled || save.isPending}
      busy={save.isPending} onClick={() => save.mutate(jobId)}>Save installed-pump calibration</Button>
    <p>Calibration uses saved well inputs and in-era tests. Save changed well inputs before refitting.</p>
    <p>{HYDRAULICS_LABELS[result.hydraulics_model ?? "beggs"]}. Saves this fit for {result.pump}, installed {result.era_start?.slice(0, 10)}. Save well inputs using the bar at the top of this page.</p>
    {!valid && <p className="text-amber-700">The installation or hydraulics differs from this fit. Refresh the well and calibrate again.</p>}
    {!meta.data?.writes_enabled && <p>Saving is unavailable while this app is in read-only mode. Apply remains available for this session.</p>}
    {save.data && <p className="text-emerald-700">{save.data.message}</p>}
    {save.isError && <p className="text-amber-700">{save.error.message}</p>}
  </div>;
}

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
  const applyFit = useParamsStore((s) => s.applyPumpFit);
  const context = useParamsStore((s) => s.context);
  const model = useParamsStore((s) => s.params.hydraulics_model);
  const validScope = matchesInstallation(result, context?.pump, model);
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
    `(${single.match_quality}). Event calibration becomes available as this pump ` +
    "accumulates daily history.";

  const applyTitle = pinned
    ? "Nothing was fitted - the coefficients came back at their seeds, so there is nothing to apply."
    : failed
      ? "The solver found no valid operating point, so there is nothing to apply."
      : "Apply the matched ken/kth/kdi to the sidebar. Nozzle area stays at its current estimate; " +
        "a single BHP observation cannot determine it. Use Save installed-pump calibration to keep this fit.";

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
        disabled={pinned || failed || !validScope}
        title={applyTitle}
        onClick={() => applyFit(result, { ken: single.ken, kth: single.kth, kdi: single.kdi, nozzle_area_factor: result.current.nozzle_area_factor ?? 1 })}
      >
        Apply to inputs
      </Button>
    </div>
  );
}

function ResultBlock({ result }: { result: EventCalibrationResult }) {
  const applyFit = useParamsStore((s) => s.applyPumpFit);
  const context = useParamsStore((s) => s.context);
  const model = useParamsStore((s) => s.params.hydraulics_model);
  const validScope = matchesInstallation(result, context?.pump, model);
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
      ? `estimated nozzle area ${wearPct.toFixed(0)}% above catalog`
      : wearPct <= -0.5
        ? `estimated nozzle area ${Math.abs(wearPct).toFixed(0)}% below catalog`
        : "nozzle at catalog size";
  const headline =
    `Matched ${fit.n_used} points from this pump's history (${eraStart} - today): ` +
    `${wearPhrase}, BHP fit RMS error ${Math.round(fit.rms_bhp_psi)} psi.`;

  const paramsLine =
    `ken ${coef(fit.ken)} | kth ${coef(fit.kth)} | kdi ${coef(fit.kdi)} | ` +
    `nozzle area ${fit.fnz.toFixed(2)}`;

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
    "Apply the fitted coefficients and nozzle area factor to the sidebar" +
    (curParts.length ? ` (replaces ${curParts.join(" / ")})` : "") +
    ". Use Save installed-pump calibration to keep this fit.";

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
        disabled={!validScope}
        onClick={() =>
          applyFit(result, {
            ken: fit.ken,
            kth: fit.kth,
            kdi: fit.kdi,
            nozzle_area_factor: fit.fnz,
            mach_crit: 1.0,
          })
        }
      >
        Apply to inputs
      </Button>
    </div>
  );
}

export function EventCalibration({ well }: { well: string }) {
  // Persisted per WELL in the optimize store (it already persists run job
  // ids): event calibration is a 1-3 minute job, and a Solver -> Batch ->
  // Solver detour used to drop the id and orphan the fit (review
  // 2026-09-01, WEB-8). Per-well keys keep one well's fit off another.
  const model = useParamsStore((s) => s.params.hydraulics_model);
  const jobKey = model === "beggs" ? `event_cal:${well}` : `event_cal:${well}:${model}`;
  const jobId = useOptimizeStore((s) => s.lastJob[jobKey] ?? null);
  const setLastJob = useOptimizeStore((s) => s.setLastJob);
  const setJobId = (id: string | null) => setLastJob(jobKey, id);
  const start = useStartEventCalibration();
  const job = useOptimizeJob(jobId);

  // Expired job (server restart): drop the stale id quietly.
  useEffect(() => {
    if (jobId && isMissingJob(job.error)) setLastJob(jobKey, null);
  }, [jobId, job.error, jobKey, setLastJob]);

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
          "young eras fall back to matching the latest test BHP. " +
          "A full fit is several passes over every point and typically takes 1-3 minutes - " +
          "the line beside the button shows the pass and evaluation count."
        }
        onClick={() => {
          start.mutate({ well, hydraulics_model: model }, { onSuccess: (r) => setJobId(r.job_id) });
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
          {job.data?.seconds !== undefined && (
            <span className="ml-1.5 tabular-nums text-slate-400">
              {Math.round(job.data.seconds)}s
            </span>
          )}
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
      {result !== null && jobId && <SaveFit key={jobId} result={result} jobId={jobId} />}
    </>
  );
}
