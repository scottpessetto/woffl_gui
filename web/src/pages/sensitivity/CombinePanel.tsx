/**
 * Combined permutations - the question single-knob sensitivity cannot
 * answer: can ANY combination inside the engineer's ranges reach the
 * measured test?
 *
 * The tornado ranks knobs one at a time, so on a well where no single knob
 * closes the gap it goes quiet exactly when the engineer most needs an
 * answer. Two knobs together often do close it, and the only honest way to
 * know is to solve the factorial. Pick the inputs that are genuinely shaky
 * on this well, pick how finely to cut their ranges, and the server runs
 * every permutation.
 *
 * Ranges are whatever the engineer set in the knob table above, not the
 * defaults - what they tuned is what gets combined. Explicitly triggered:
 * a 5-level study over four inputs is 625 solves and must never fire off a
 * render.
 *
 * The study outlives this panel. It runs on the server, so navigating away
 * mid-solve does not stop it: the picker, what was fired and the job id all
 * live in the sensitivity store keyed by well, and a remount re-attaches to
 * the job rather than orphaning it. An expired id clears itself.
 */

import { ChevronDown, ChevronRight } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { useCombineJob, useSensitivityCombine } from "../../api/hooks";
import type { CombineKnob, SensitivityKnob, SimParams } from "../../api/types";
import { Button, Card, ErrorNote, Spinner } from "../../components/ui";
import {
  type CombineState,
  DEFAULT_COMBINE,
  useSensitivityStore,
} from "../../state/sensitivity";
import { type BoundsMap, effectiveRange, fmtBound } from "./bounds";
import {
  type CombineTargets,
  estimateLabel,
  knobLevels,
  LEVEL_CHOICES,
  MAX_COMBINE_RUNS,
} from "./combine";
import { EnvelopeChart } from "./EnvelopeChart";
import { PermutationScatter } from "./PermutationScatter";
import { TopRunsTable } from "./TopRunsTable";

const HELP =
  "Solves every combination of the selected inputs across the ranges set above, then reports " +
  "the span each match quantity can reach and which combination lands closest to the test.";

export function CombinePanel({
  well,
  params,
  targets,
  knobs,
  bounds,
}: {
  well: string;
  params: SimParams;
  targets: CombineTargets;
  knobs: SensitivityKnob[];
  /** the same edited bounds the tornado is sweeping */
  bounds: BoundsMap;
}) {
  // Everything the engineer chose here belongs to the well it was chosen on,
  // and outlives the page: keying by well is what keeps MPI-31's envelope
  // off the next well, and what lets a walk to the Solver come back to a
  // study that kept solving while the view was unmounted.
  const state = useSensitivityStore((s) => s.combine[well] ?? DEFAULT_COMBINE);
  const setCombine = useSensitivityStore((s) => s.setCombine);
  const setCombineJob = useSensitivityStore((s) => s.setCombineJob);
  const { picked, levels, jobId, fired } = state;
  // Open on arrival when there is a study to come back to, so a re-attached
  // run is not hiding behind a collapsed header.
  const [open, setOpen] = useState(jobId !== null);
  // The job this mount started itself. Anything else on screen came from
  // before the engineer navigated away, and says so.
  const startedHere = useRef<string | null>(null);

  const combine = useSensitivityCombine();
  const { reset } = combine;
  const job = useCombineJob(jobId);
  const reattached = jobId !== null && startedHere.current !== jobId;

  // The POST's own pending/error state is this component's, not the well's,
  // so a refusal on one well must not greet the engineer on the next.
  useEffect(() => {
    reset();
  }, [well, reset]);

  // Expired job (server restart, cleared registry): drop the stale id
  // quietly, exactly as the optimization run panel does.
  useEffect(() => {
    if (jobId !== null && job.isError) setCombineJob(well, null);
  }, [well, jobId, job.isError, setCombineJob]);

  const plan = useMemo(() => {
    const entries: { knob: SensitivityKnob; req: CombineKnob }[] = [];
    for (const knob of knobs) {
      if (!picked.includes(knob.id)) continue;
      const { low, high } = effectiveRange(knob, bounds);
      entries.push({
        knob,
        req: { id: knob.id, low, high, levels: knobLevels(knob, low, high, levels) },
      });
    }
    const count = entries.reduce((n, e) => n * e.req.levels, 1);
    return { entries, count: entries.length === 0 ? 0 : count };
  }, [knobs, picked, bounds, levels]);

  // Busy from the click until the job settles, including the gap between the
  // start call returning an id and the first poll answering.
  const settled = job.data?.status === "done" || job.data?.status === "error";
  const running = combine.isPending || (jobId !== null && !settled);
  const overCap = plan.count > MAX_COMBINE_RUNS;
  const canRun = plan.entries.length > 0 && !overCap && !running;

  const runStudy = () => {
    const labels: Record<string, string> = {};
    for (const e of plan.entries) labels[e.knob.id] = e.knob.label;
    const next: CombineState = {
      picked,
      levels,
      jobId: null,
      fired: { ids: plan.entries.map((e) => e.knob.id), labels, count: plan.count },
    };
    setCombine(well, next);
    combine.mutate(
      {
        well,
        params,
        target_psu: targets.target_psu,
        target_qoil: targets.target_qoil,
        target_qliq: targets.target_qliq,
        target_qpf: targets.target_qpf,
        knobs: plan.entries.map((e) => e.req),
      },
      {
        onSuccess: (started) => {
          startedHere.current = started.job_id;
          setCombineJob(well, started.job_id);
        },
      },
    );
  };

  // The POST only refuses bad requests; everything the solve can hit lands on
  // the job as an error status, so both have to reach the same note.
  const failed = job.data?.status === "error" ? new Error(job.data.error ?? "study failed") : null;
  const data = job.data?.status === "done" ? (job.data.result ?? undefined) : undefined;
  const caption =
    data === undefined
      ? ""
      : data.n_failed === 0
        ? `${data.n_runs} permutations, all solved.`
        : `${data.n_runs} permutations, ${data.n_failed} failed to solve and are left off the charts.`;

  return (
    <Card padded={false} className="p-3">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        title={HELP}
        className="flex w-full items-center gap-1.5 text-left text-sm font-semibold text-slate-700"
      >
        {open ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
        Combined Permutations
        <span className="pl-1 text-xs font-normal text-slate-500">
          Can any combination inside these ranges reach the test?
        </span>
      </button>

      {open && (
        <div className="space-y-3 pt-3">
          <div className="grid gap-x-5 gap-y-1 sm:grid-cols-2 lg:grid-cols-3">
            {knobs.map((knob) => {
              const { low, high } = effectiveRange(knob, bounds);
              return (
                <label
                  key={knob.id}
                  className="flex cursor-pointer items-center gap-2 rounded px-1 py-0.5 text-xs text-slate-700 hover:bg-slate-50"
                >
                  <input
                    type="checkbox"
                    checked={picked.includes(knob.id)}
                    onChange={(e) =>
                      setCombine(well, {
                        ...state,
                        picked: e.target.checked
                          ? [...picked, knob.id]
                          : picked.filter((id) => id !== knob.id),
                      })
                    }
                    className="h-3.5 w-3.5 rounded border-slate-300"
                  />
                  <span className="font-medium">{knob.label}</span>
                  {knob.inert && (
                    <span className="rounded bg-slate-100 px-1 py-px text-[10px] text-slate-500">
                      inert
                    </span>
                  )}
                  <span className="ml-auto tabular-nums text-slate-500">
                    {fmtBound(knob, low)} to {fmtBound(knob, high)}
                  </span>
                </label>
              );
            })}
          </div>

          <div className="flex flex-wrap items-center gap-3 border-t border-slate-100 pt-3">
            <span className="text-xs text-slate-500">Levels per input</span>
            <div className="flex gap-1 rounded-lg border border-slate-200 bg-white p-0.5">
              {LEVEL_CHOICES.map((n) => (
                <button
                  key={n}
                  type="button"
                  onClick={() => setCombine(well, { ...state, levels: n })}
                  className={
                    levels === n
                      ? "rounded-md bg-blue-600 px-2.5 py-0.5 text-xs font-medium text-white"
                      : "rounded-md px-2.5 py-0.5 text-xs text-slate-600 hover:bg-slate-100"
                  }
                >
                  {n}
                </button>
              ))}
            </div>

            <span className="text-xs tabular-nums text-slate-600">
              {plan.entries.length === 0
                ? "Pick the inputs you are not sure about."
                : `${plan.count} runs, ${estimateLabel(plan.count)}`}
            </span>

            <Button variant="primary" size="sm" onClick={runStudy} disabled={!canRun} busy={running}>
              Run combination
            </Button>

            {reattached && (
              <span className="text-xs text-slate-500">
                Picked up from the study you started on this well before leaving the page.
              </span>
            )}

            {overCap && (
              <span className="text-xs font-medium text-red-700">
                {plan.count} runs is past the {MAX_COMBINE_RUNS} run cap. Deselect an input or
                drop to 2 levels.
              </span>
            )}
          </div>

          {running && (
            <div className="flex items-center justify-center gap-3">
              <Spinner label={`Solving ${fired?.count ?? plan.count} permutations`} />
              {job.data?.progress != null && (
                <span className="text-xs tabular-nums text-slate-500">{job.data.progress}</span>
              )}
            </div>
          )}
          {combine.isError && <ErrorNote error={combine.error} />}
          {failed !== null && <ErrorNote error={failed} />}

          {data !== undefined && fired !== null && !running && (
            <div className="space-y-3">
              {data.notes.map((note) => (
                <p key={note} className="px-1 text-xs text-slate-500">
                  {note}
                </p>
              ))}
              <EnvelopeChart
                envelope={data.envelope}
                reachable={data.reachable}
                targets={targets}
                caption={caption}
              />
              <PermutationScatter
                runs={data.runs}
                baseline={data.baseline}
                bestIndex={data.best_index}
                targets={targets}
                knobIds={fired.ids}
                knobLabels={fired.labels}
              />
              <TopRunsTable
                runs={data.runs}
                bestIndex={data.best_index}
                targets={targets}
                knobIds={fired.ids}
                knobLabels={fired.labels}
                knobs={knobs}
              />
            </div>
          )}
        </div>
      )}
    </Card>
  );
}
