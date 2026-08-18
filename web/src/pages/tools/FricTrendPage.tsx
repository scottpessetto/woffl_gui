/**
 * JP Fric Trend - fitted friction coefficients over a well's test history.
 *
 * Port of woffl/gui/scotts_tools/jp_fric_trend.py. Two-step per test:
 * calibrate the PF pressure that reproduces the measured lift water, then fit
 * the coefficients that reproduce the measured BHP at that pressure.
 *
 * Each point uses the pump installed AT THAT TEST and the test's own WC/GOR,
 * which is what makes the trend readable: before that fix every historical
 * point was fitted with today's geometry, so a "trend" across a pump change
 * was a geometry artifact rather than wear.
 */

import { useMemo, useState } from "react";

import { useWells } from "../../api/hooks";
import type { ToolRow } from "../../api/types";
import { ChartPanel } from "../../charts/ChartPanel";
import type { EChartsOption } from "../../charts/echarts";
import { CATEGORY20, axis, houseOption } from "../../charts/theme";
import { Button, InfoNote, Section, WarnNote } from "../../components/ui";
import { AutoTable, NumField, RunStatus, useToolRun } from "./ToolRun";

interface Req {
  wells: string[];
  months_back: number;
}

const COEFS = ["Cal ken", "Cal kth", "Cal kdi"] as const;
const LEAD = ["Well", "WtDate", "Pump", "Cal ken", "Cal kth", "Cal kdi",
  "PpfRequired", "BHP target", "BHP modeled", "Match", "Status"];

/** One series per (well, coefficient) over test date. */
function trendChart(rows: ToolRow[]): EChartsOption | null {
  const dated = rows.filter((r) => r.WtDate);
  if (!dated.length) return null;
  const wells = [...new Set(dated.map((r) => String(r.Well)))];

  const series = wells.flatMap((w, wi) =>
    COEFS.map((c, ci) => ({
      name: `${w} ${c.replace("Cal ", "")}`,
      type: "line" as const,
      symbolSize: 6,
      lineStyle: { color: CATEGORY20[(wi * 3 + ci) % CATEGORY20.length], type: ci === 0 ? "solid" as const : ci === 1 ? "dashed" as const : "dotted" as const },
      itemStyle: { color: CATEGORY20[(wi * 3 + ci) % CATEGORY20.length] },
      data: dated
        .filter((r) => String(r.Well) === w && typeof r[c] === "number")
        .map((r) => [String(r.WtDate), r[c] as number] as [string, number]),
    })),
  ).filter((s) => s.data.length > 0);

  if (!series.length) return null;
  return houseOption({
    tooltip: { trigger: "axis" },
    legend: { top: 0, type: "scroll" },
    grid: { top: 40, left: 60, right: 20, bottom: 40 },
    xAxis: { type: "time", ...axis("Test date") },
    yAxis: { type: "value", ...axis("Fitted coefficient") },
    series,
  });
}

export default function FricTrendPage() {
  const wellsQ = useWells();
  const all = useMemo(() => (wellsQ.data?.wells ?? []).map((w) => w.name).sort(), [wellsQ.data]);

  const [selected, setSelected] = useState<string[]>([]);
  const [months, setMonths] = useState(12);
  const run = useToolRun<Req>("/tools/fric-trend/run");

  const result = run.result as
    | { rows?: ToolRow[]; wells?: string[]; skipped?: Record<string, string> }
    | null;
  const rows = result?.rows ?? [];
  const chart = useMemo(() => trendChart(rows), [rows]);
  const skipped = Object.entries(result?.skipped ?? {});

  return (
    <div className="space-y-4">
      <Section
        title="JP Fric Trend"
        actions={
          <Button
            onClick={() => run.run({ wells: selected, months_back: months })}
            disabled={!selected.length || run.running}
          >
            {run.running ? "Calibrating..." : "Run"}
          </Button>
        }
      >
        <p className="mb-3 text-sm text-slate-600">
          Fits ken, kth and kdi against every usable test in the window, one point per test.
          A coefficient drifting up across a pump&apos;s tenure is the wear signal; a step at a
          pump change is not.
        </p>
        <div className="flex flex-wrap items-end gap-3">
          <NumField label="Lookback (months)" value={months} onChange={setMonths} min={1} max={60} />
          <Button size="sm" variant="ghost" onClick={() => setSelected([])}>Clear</Button>
        </div>
        <div className="mt-3">
          <span className="text-xs text-slate-500">Wells ({selected.length} selected)</span>
          <div className="mt-1 flex max-h-40 flex-wrap gap-1 overflow-y-auto rounded-md border border-slate-200 p-2">
            {all.map((w) => (
              <button
                key={w}
                type="button"
                onClick={() =>
                  setSelected((s) => (s.includes(w) ? s.filter((x) => x !== w) : [...s, w]))
                }
                className={
                  "rounded border px-1.5 py-0.5 text-xs transition-colors " +
                  (selected.includes(w)
                    ? "border-blue-500 bg-blue-50 font-medium text-blue-700"
                    : "border-slate-200 bg-white text-slate-500 hover:bg-slate-50")
                }
              >
                {w}
              </button>
            ))}
          </div>
        </div>
      </Section>

      <RunStatus run={run as never} idle="Select one or more wells and press Run." />

      {skipped.length > 0 && (
        <WarnNote>
          {skipped.length} well{skipped.length === 1 ? "" : "s"} skipped:{" "}
          {skipped.map(([w, why]) => `${w} (${why})`).join(", ")}
        </WarnNote>
      )}

      {chart && (
        <Section title="Fitted coefficients over time">
          <ChartPanel option={chart} height={420} zoom={{ xAxisIndex: [0], yAxisIndex: [0] }} />
        </Section>
      )}

      {rows.length > 0 ? (
        <Section title="Per-test detail">
          <AutoTable rows={rows} prefer={LEAD} />
        </Section>
      ) : (
        result && <InfoNote>No tests in the window had everything the two-step fit needs.</InfoNote>
      )}
    </div>
  );
}
