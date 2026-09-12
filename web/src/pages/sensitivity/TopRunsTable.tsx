/**
 * The ten permutations that come closest to the measured test, with the
 * knob settings that got there. The scatter says a match exists somewhere
 * in the cloud; this says what to type into the sidebar to reproduce it.
 *
 * Ranked by the server's score - RMS fractional error across whatever
 * targets the test carried - so a run is only "best" against the quantities
 * that were actually measured. With no targets at all there is nothing to
 * rank by and the table falls back to factorial order.
 */

import { useMemo } from "react";
import { useNavigate } from "react-router-dom";

import type { CombineRequest, CombineRun, SensitivityKnob } from "../../api/types";
import { CRIMSON, SLATE } from "../../charts/theme";
import { Button, Card, type Column, DataTable } from "../../components/ui";
import { fmtNum } from "../../lib/format";
import { effectiveParams, useParamsStore } from "../../state/params";
import { useSensitivityStore } from "../../state/sensitivity";
import { appliedStudyParams, matchingStudy } from "./study";
import { type CombineTargets, runReadings } from "./combine";
import { METRICS, signed } from "./metrics";

const SCORE_HELP =
  "RMS fractional error in measured BHP, oil and PF; liquid replaces oil only if oil is unavailable. " +
  "Lower is closer on this test. This is not an uncertainty-weighted fit or independent validation.";

type Row = Record<string, unknown>;

/** DataTable rows are loosely keyed, so every numeric cell narrows here. */
function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

/**
 * One line of provenance for an applied permutation, e.g.
 * "Sensitivity permutation 2 of 27 (score 0.0571): PF surface pressure 3,910,
 * Throat loss (kth) 0.525, IPR anchor rate 439".
 *
 * It lands in the Solver header and prefills the save comment, so a saved
 * curve carries the study that produced it instead of arriving as bare
 * numbers nobody can re-derive.
 */
function matchNote(
  row: Row,
  run: CombineRun,
  total: number,
  knobLabels: Record<string, string>,
): string {
  const rank = num(row.rank);
  const score = num(row.score);
  const settings = Object.entries(run.labels)
    .map(([id, label]) => `${knobLabels[id] ?? id} ${label}`)
    .join(", ");
  const head = `Sensitivity permutation ${rank ?? "?"} of ${total}`;
  const scored = score === null ? head : `${head} (score ${fmtNum(score, 4)})`;
  return settings ? `${scored}: ${settings}` : scored;
}

export function TopRunsTable({
  runs,
  bestIndex,
  targets,
  knobIds,
  knobLabels,
  knobs,
  request, currentRequest, canApply,
}: {
  runs: CombineRun[];
  bestIndex: number | null;
  targets: CombineTargets;
  /** varied knob ids, in picker order - one column each */
  knobIds: string[];
  knobLabels: Record<string, string>;
  /** the full input list, for the field name and kind behind each column */
  knobs: SensitivityKnob[];
  request: CombineRequest | null;
  currentRequest: CombineRequest;
  canApply: boolean;
}) {
  const setMany = useParamsStore((s) => s.setMany);
  const setMatchNote = useParamsStore((s) => s.setMatchNote);
  const navigate = useNavigate();
  const rows = useMemo<Row[]>(() => {
    const solved = runs
      .map((run, idx) => ({ run, idx }))
      .filter((r) => r.run.error === null);
    const scored = solved.filter((r) => r.run.score !== null);
    const ordered = scored.length > 0
      ? scored.sort((a, b) => (a.run.score ?? 0) - (b.run.score ?? 0))
      : solved;

    return ordered.slice(0, 10).map((entry, i) => {
      const row: Row = { rank: i + 1, idx: entry.idx, score: entry.run.score, run: entry.run };
      for (const id of knobIds) row[`k_${id}`] = entry.run.labels[id] ?? "-";
      for (const r of runReadings(entry.run, targets)) {
        row[`m_${r.spec.id}`] = r.value;
        row[`e_${r.spec.id}`] = r.err;
      }
      return row;
    });
  }, [runs, targets, knobIds]);

  const columns = useMemo<Column<Row>[]>(() => {
    const cols: Column<Row>[] = [
      { key: "rank", label: "#", align: "right", width: "3rem" },
    ];
    for (const id of knobIds) {
      cols.push({
        key: `k_${id}`,
        label: knobLabels[id] ?? id,
        align: "right",
        render: (row) => String(row[`k_${id}`] ?? "-"),
      });
    }
    for (const spec of METRICS) {
      cols.push({
        key: `m_${spec.id}`,
        label: `${spec.label} (${spec.unit})`,
        align: "right",
        render: (row) => {
          const value = num(row[`m_${spec.id}`]);
          const err = num(row[`e_${spec.id}`]);
          return (
            <span>
              {fmtNum(value, spec.dp)}
              {err !== null && (
                <span className="pl-1.5 text-[11px]" style={{ color: SLATE }}>
                  {signed(err, spec.dp)}
                </span>
              )}
            </span>
          );
        },
      });
    }
    cols.push({
      key: "score",
      label: "Score",
      align: "right",
      help: SCORE_HELP,
      render: (row) => {
        const score = num(row.score);
        const best = num(row.idx) === bestIndex;
        return (
          <span style={best ? { color: CRIMSON, fontWeight: 600 } : undefined}>
            {fmtNum(score, 4)}
          </span>
        );
      },
    });
    cols.push({
      key: "apply",
      label: "",
      align: "right",
      render: (row) => {
        const run = row.run as CombineRun | undefined;
        if (run === undefined) return null;
        const patch = run.applied_inputs ?? {};
        const fields = Object.keys(patch);
        if (fields.length === 0) return null;
        return (
          <Button
            variant="secondary"
            size="sm"
            disabled={!canApply || !request || !run.applied_inputs}
            title={`Apply this submitted scenario (${fields.join(", ")}) and open Solver with the same comparison test. Saving well inputs is a separate action.`}
            onClick={() => {
              if (!canApply || !request) return;
              const live = useParamsStore.getState();
              if (!matchingStudy({ ...currentRequest, well: live.well, params: effectiveParams(live.params), installation_key: JSON.stringify(live.context?.pump ?? null) }, request)) return;
              const candidate = appliedStudyParams(request, run);
              if (!candidate) return;
              setMany(candidate);
              if (request.test_key) useSensitivityStore.getState().queueComparison(request.well, request.test_key);
              // Provenance travels with the numbers: the Solver shows it and
              // the save comment prefills with it, so prop_hist records WHY
              // this curve rather than just what it was.
              setMatchNote(`${matchNote(row, run, runs.length, knobLabels)}. Comparison ${request.test_key ?? "none"}; ${request.wc_basis ?? "fixed_oil_ipr"}.`);
              navigate("/solver");
            }}
          >
            Apply
          </Button>
        );
      },
    });
    return cols;
  }, [knobIds, knobLabels, bestIndex, knobs, setMany, setMatchNote, navigate, runs.length, request, currentRequest, canApply]);

  if (rows.length === 0) return null;

  const unscored = rows.every((r) => num(r.score) === null);

  return (
    <Card padded={false} className="p-2">
      <p className="px-2 pt-1 text-xs font-semibold text-slate-600">Closest Permutations</p>
      <p className="px-2 pb-1.5 text-[11px] text-slate-500">
        {unscored
          ? "No measured test on this well, so these are the first ten permutations in factorial order."
          : "Ranked by BHP, oil and PF fractional error (liquid substitutes for missing oil). Liquid remains visible for diagnosis. The highlighted score is closest on this test."}
      </p>
      <DataTable
        columns={columns}
        rows={rows}
        rowKey={(row) => String(row.idx)}
        maxHeight="24rem"
        sortable
        pinFirst
        highlightRow={(row) => num(row.idx) === bestIndex}
      />
    </Card>
  );
}
