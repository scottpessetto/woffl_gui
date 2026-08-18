/**
 * Shared plumbing for the tool pages.
 *
 * Every tool but Pad Water Cut is a background job: start it, poll it, render
 * rows. Rather than repeat that in six pages, `useToolRun` owns the job id and
 * `<RunStatus>` owns the running/error/idle states, so each page is left with
 * its own controls, columns and charts.
 *
 * `AutoTable` exists because these tools return WIDE, tool-specific frames
 * (the calibration table alone is ~25 columns) whose shape is defined
 * server-side. Hand-writing a column list per tool would guarantee they drift
 * apart; instead the columns come from the data, with a `prefer` list to pin
 * the few that should lead.
 */

import { useMemo, useState } from "react";

import { useStartToolJob, useToolJob } from "../../api/hooks";
import type { ToolRow } from "../../api/types";
import { Badge, type Column, DataTable, ErrorNote, InfoNote, Spinner } from "../../components/ui";
import { fmtNum } from "../../lib/format";

/** Start/poll one tool job; `result` is whatever that tool returns. */
export function useToolRun<Req>(path: string) {
  const [jobId, setJobId] = useState<string | null>(null);
  const start = useStartToolJob<Req>(path);
  const job = useToolJob(jobId);

  return {
    run: (req: Req) =>
      start.mutate(req, { onSuccess: (r) => setJobId(r.job_id) }),
    reset: () => setJobId(null),
    starting: start.isPending,
    startError: start.error,
    job: job.data ?? null,
    running: Boolean(jobId) && (!job.data || job.data.status === "running"),
    result: job.data?.status === "done" ? (job.data.result ?? null) : null,
    error: job.data?.status === "error" ? job.data.error : null,
  };
}

export function RunStatus({
  run,
  idle,
}: {
  run: ReturnType<typeof useToolRun<never>>;
  idle: string;
}) {
  if (run.startError) return <ErrorNote error={run.startError} />;
  if (run.running) {
    return (
      <div className="flex items-center gap-3">
        <Spinner label={run.job?.progress ?? "Starting"} />
        {run.job?.seconds != null && (
          <span className="text-xs text-slate-500 tabular-nums">
            {fmtNum(run.job.seconds, 0)}s
          </span>
        )}
      </div>
    );
  }
  if (run.error) {
    return (
      <ErrorNote
        error={{ error: "internal", message: run.error } as unknown as Error}
      />
    );
  }
  if (!run.result) return <InfoNote>{idle}</InfoNote>;
  return null;
}

/** Cell formatter: numbers get thousands separators, booleans a check. */
function renderCell(v: unknown) {
  if (v === null || v === undefined || v === "") return <span className="text-slate-300">-</span>;
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "number") {
    return <span className="tabular-nums">{fmtNum(v, Number.isInteger(v) ? 0 : 2)}</span>;
  }
  return String(v);
}

/** Table over a tool's rows, columns derived from the data. */
export function AutoTable({
  rows,
  prefer = [],
  hide = [],
  emptyNote = "No rows.",
}: {
  rows: ToolRow[];
  /** Columns to pin, in order, ahead of whatever else the rows carry. */
  prefer?: string[];
  hide?: string[];
  emptyNote?: string;
}) {
  const columns = useMemo<Column<ToolRow>[]>(() => {
    if (!rows.length) return [];
    const seen = new Set<string>();
    for (const r of rows) for (const k of Object.keys(r)) seen.add(k);
    for (const h of hide) seen.delete(h);
    // Underscore-prefixed keys are engine internals, never display columns.
    const keys = [...prefer.filter((k) => seen.has(k)),
                  ...[...seen].filter((k) => !prefer.includes(k) && !k.startsWith("_"))];
    return keys.map((k) => ({
      key: k,
      label: k,
      align: typeof rows[0]?.[k] === "number" ? ("right" as const) : undefined,
      render: (r: ToolRow) => renderCell(r[k]),
    }));
  }, [rows, prefer, hide]);

  if (!rows.length) return <InfoNote>{emptyNote}</InfoNote>;
  return (
    <DataTable
      rows={rows}
      columns={columns}
      rowKey={(r, i) => String(r.Well ?? r.name ?? i)}
      sortable
      pinFirst
      maxHeight="32rem"
    />
  );
}

/** Small labelled number input, matching the sidebar's field styling. */
export function NumField({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
  width = "w-28",
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
  width?: string;
}) {
  return (
    <label className="block">
      <span className="text-xs text-slate-500">{label}</span>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => {
          const n = Number(e.target.value);
          if (!Number.isNaN(n)) onChange(n);
        }}
        className={`mt-1 block h-8 ${width} rounded-md border border-slate-300 bg-white px-2 text-sm text-slate-800 outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-200`}
      />
    </label>
  );
}

/** Verdict / status pill with the app's tone vocabulary. */
export function VerdictBadge({ value }: { value: unknown }) {
  const v = String(value ?? "");
  // Tone vocabulary is the app's: good / fair / poor / info / neutral.
  const tone = v.startsWith("responsive")
    ? "good"
    : v.includes("sonic")
      ? "fair"
      : v === "slugging"
        ? "poor"
        : v === "n/a" || !v
          ? "neutral"
          : "info";
  return <Badge tone={tone}>{v || "-"}</Badge>;
}
