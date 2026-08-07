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

import type { CombineRun, SensitivityKnob, SimParams } from "../../api/types";
import { CRIMSON, SLATE } from "../../charts/theme";
import { Button, Card, type Column, DataTable } from "../../components/ui";
import { fmtNum } from "../../lib/format";
import { useParamsStore } from "../../state/params";
import { type CombineTargets, runReadings } from "./combine";
import { METRICS, signed } from "./metrics";

const SCORE_HELP =
  "Root-mean-square fractional error across the measured quantities only. Lower is better; " +
  "zero would match every measured number exactly.";

type Row = Record<string, unknown>;

/** DataTable rows are loosely keyed, so every numeric cell narrows here. */
function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

/**
 * The sidebar patch that reproduces one permutation.
 *
 * Catalog inputs carry an option INDEX in `values` and the option string in
 * `labels`; `nozzle_no` and `area_ratio` are string fields on SimParams, so
 * the label is the value to write. Everything else is already in the field's
 * own units. Inputs that were not varied are left alone.
 */
function runPatch(run: CombineRun, knobs: SensitivityKnob[]): Partial<SimParams> {
  const patch: Record<string, string | number> = {};
  for (const k of knobs) {
    if (!(k.id in run.values)) continue;
    const label = run.labels[k.id];
    if (k.kind === "catalog") {
      if (label !== undefined) patch[k.field] = label;
    } else {
      patch[k.field] = run.values[k.id];
    }
  }
  return patch as Partial<SimParams>;
}

export function TopRunsTable({
  runs,
  bestIndex,
  targets,
  knobIds,
  knobLabels,
  knobs,
}: {
  runs: CombineRun[];
  bestIndex: number | null;
  targets: CombineTargets;
  /** varied knob ids, in picker order - one column each */
  knobIds: string[];
  knobLabels: Record<string, string>;
  /** the full input list, for the field name and kind behind each column */
  knobs: SensitivityKnob[];
}) {
  const setMany = useParamsStore((s) => s.setMany);
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
        const patch = runPatch(run, knobs);
        const fields = Object.keys(patch);
        if (fields.length === 0) return null;
        return (
          <Button
            variant="secondary"
            size="sm"
            title={`Write this permutation into the sidebar (${fields.join(", ")}) and open the Solver. Nothing is saved until you save the well default there.`}
            onClick={() => {
              setMany(patch);
              navigate("/solver");
            }}
          >
            Apply
          </Button>
        );
      },
    });
    return cols;
  }, [knobIds, knobLabels, bestIndex, knobs, setMany, navigate]);

  if (rows.length === 0) return null;

  const unscored = rows.every((r) => num(r.score) === null);

  return (
    <Card padded={false} className="p-2">
      <p className="px-2 pt-1 text-xs font-semibold text-slate-600">Closest Permutations</p>
      <p className="px-2 pb-1.5 text-[11px] text-slate-500">
        {unscored
          ? "No measured test on this well, so these are the first ten permutations in factorial order."
          : "Ranked by RMS fractional error against the measured test. The red score is the best run."}
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
