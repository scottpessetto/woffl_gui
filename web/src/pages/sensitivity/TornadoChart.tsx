/**
 * Tornado - which input moves the selected match quantity, and by how far.
 * One row per knob sorted by total travel, both bars anchored at the
 * baseline (x = 0) and coloured by direction rather than by input.
 *
 * Inert inputs keep their row and carry a grey chip in the axis label. On a
 * well sitting on the choked-flow floor that empty row IS the finding - the
 * solver returns the choked suction pressure directly, so the power-fluid
 * and loss-coefficient knobs are bit-identical across their whole range -
 * and filtering it out would hide exactly what the engineer came for.
 *
 * With a measured test loaded, the dashed line is the distance the match
 * still has to travel. A bar that stops short of it cannot close the gap on
 * its own, whatever the engineer does to that input.
 */

import { useMemo } from "react";

import type { EChartsOption } from "../../charts/echarts";
import { ACCENT, axis, baseGrid, baseTooltip, CRIMSON, GOLD, houseOption, SLATE, TEXT, ttHeader, ttRow } from "../../charts/theme";
import { ChartPanel } from "../../charts/ChartPanel";
import { Card } from "../../components/ui";
import { fmtNum } from "../../lib/format";
import { type MetricSpec, signed, type TornadoRow } from "./metrics";

const HELP =
  "Signed change in the selected match quantity across each input's full swept range, " +
  "measured from the current sidebar case. Rows marked inert did not move any of the " +
  "four quantities anywhere in their range.";

/** The one field the row tooltip needs off an axis-trigger param. */
function axisCategory(raw: unknown): string | null {
  const list = Array.isArray(raw) ? raw : [raw];
  for (const p of list) {
    if (p !== null && typeof p === "object" && "axisValue" in p) {
      const v = p.axisValue;
      if (typeof v === "string") return v;
    }
  }
  return null;
}

/** Round an axis bound outward to a readable step, so a bound forced by the
 *  target gap does not print as a raw float tick. `dir` is -1 for the min
 *  side and +1 for the max. */
function niceBound(v: number, dir: -1 | 1): number {
  if (v === 0 || !Number.isFinite(v)) return 0;
  const mag = Math.abs(v);
  const step = Math.pow(10, Math.floor(Math.log10(mag))) / 2;
  return dir < 0 ? Math.floor(v / step) * step : Math.ceil(v / step) * step;
}

function tornadoOption(
  rows: TornadoRow[],
  spec: MetricSpec,
  gap: number | null,
  selectedId: string | null,
): EChartsOption {
  const byLabel: Record<string, TornadoRow | undefined> = {};
  for (const r of rows) byLabel[r.label] = r;

  const carrier: Record<string, unknown> = {
    name: "",
    type: "line",
    data: [],
    silent: true,
    ...(gap !== null
      ? {
          markLine: {
            silent: true,
            symbol: "none",
            animation: false,
            data: [
              {
                xAxis: gap,
                lineStyle: { color: GOLD, width: 1.5, type: "dashed" },
                label: {
                  show: true,
                  formatter: `${signed(gap, spec.dp)} ${spec.unit} to test`,
                  position: "start",
                  rotate: 0,
                  distance: 6,
                  color: GOLD,
                  fontSize: 11,
                },
              },
            ],
          },
        }
      : {}),
  };

  return houseOption({
    grid: { ...baseGrid, left: 178, right: 32, top: 34, bottom: 46 },
    tooltip: {
      ...baseTooltip,
      trigger: "axis",
      axisPointer: { type: "shadow" },
      formatter: (raw: unknown): string => {
        const label = axisCategory(raw);
        const row = label !== null ? byLabel[label] : undefined;
        if (!row) return "";
        const out = [ttHeader(row.label)];
        if (row.inert) {
          out.push(ttRow(SLATE, "Inert", "no measurable change"));
        } else {
          if (row.down !== null) {
            out.push(ttRow(CRIMSON, "Decrease", `${signed(row.down, spec.dp)} ${spec.unit}`));
          }
          if (row.up !== null) {
            out.push(ttRow(ACCENT, "Increase", `${signed(row.up, spec.dp)} ${spec.unit}`));
          }
        }
        out.push(
          `<div style="margin-top:4px;max-width:260px;color:${SLATE};font-size:11px">${row.basis}</div>`,
        );
        return out.join("");
      },
    },
    legend: { data: ["Decrease", "Increase"], top: 4, itemWidth: 14, textStyle: { fontSize: 11 } },
    // The target gap is the whole point of the chart, so the axis must reach
    // it even when no single knob comes close - on a choked well the bars sit
    // on one side of zero and the gap is far off the other. Round outward, or
    // ECharts prints the raw float as a tick label.
    xAxis: {
      type: "value",
      ...axis(`Change from Baseline (${spec.unit})`),
      min: (v: { min: number }) => niceBound(Math.min(v.min, gap ?? v.min), -1),
      max: (v: { max: number }) => niceBound(Math.max(v.max, gap ?? v.max), 1),
    },
    yAxis: {
      type: "category",
      ...axis(""),
      // Row 0 is the biggest mover, so the axis runs top-down.
      inverse: true,
      data: rows.map((r) => r.label),
      splitLine: { show: false },
      axisTick: { show: false },
      axisLabel: {
        color: SLATE,
        fontSize: 11,
        margin: 8,
        formatter: (value: string): string => {
          const row = byLabel[value];
          if (!row) return value;
          const head = row.id === selectedId ? `{sel|${value}}` : value;
          return row.inert ? `${head} {chip|inert}` : head;
        },
        rich: {
          sel: { color: TEXT, fontSize: 11, fontWeight: 600 },
          chip: {
            color: SLATE,
            backgroundColor: "#f1f5f9",
            borderRadius: 3,
            padding: [2, 4],
            fontSize: 10,
          },
        },
      },
    },
    series: [
      {
        name: "Decrease",
        type: "bar",
        stack: "excursion",
        barWidth: "55%",
        itemStyle: { color: CRIMSON },
        data: rows.map((r) => r.down),
      },
      {
        name: "Increase",
        type: "bar",
        stack: "excursion",
        itemStyle: { color: ACCENT },
        data: rows.map((r) => r.up),
      },
      carrier,
    ],
  });
}

export function TornadoChart({
  rows,
  spec,
  baseline,
  target,
  selectedId,
  onSelect,
}: {
  rows: TornadoRow[];
  spec: MetricSpec;
  /** baseline reading of the selected metric, null when the solve had none */
  baseline: number | null;
  /** measured test value for the selected metric */
  target: number | null;
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  const gap = baseline !== null && target !== null ? target - baseline : null;

  const option = useMemo(
    () => tornadoOption(rows, spec, gap, selectedId),
    [rows, spec, gap, selectedId],
  );

  const handleSelect = useMemo(() => {
    const idByLabel: Record<string, string | undefined> = {};
    for (const r of rows) idByLabel[r.label] = r.id;
    return (name: string) => {
      const id = idByLabel[name];
      if (id !== undefined) onSelect(id);
    };
  }, [rows, onSelect]);

  if (rows.length === 0) return null;

  return (
    <Card padded={false} className="p-2">
      <p className="px-2 pt-1 text-xs font-semibold text-slate-600" title={HELP}>
        Match Sensitivity, {spec.label}
      </p>
      <p className="px-2 text-[11px] text-slate-500">
        {baseline !== null && `Baseline ${fmtNum(baseline, spec.dp)} ${spec.unit}. `}
        Click a row for its sweep curve.
      </p>
      {/* Nothing on a tornado is worth zooming, and a disarmed brush leaves
          the click free to select a row. */}
      <ChartPanel
        option={option}
        height={Math.max(260, 26 * rows.length + 96)}
        zoom={{ xAxisIndex: "none", yAxisIndex: "none" }}
        onSelect={handleSelect}
      />
    </Card>
  );
}
