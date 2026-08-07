/**
 * Reachable envelope - the span each match quantity can be pushed to by ANY
 * combination inside the engineer's ranges, one horizontal bar per quantity
 * with the measured test as a gold reference line.
 *
 * The bar is the answer to the question the tornado cannot answer. A test
 * line sitting outside its bar means no permutation of these knobs, at any
 * setting the engineer said was believable, reaches that number - the model
 * is wrong somewhere the sweep is not looking. A line inside the bar means
 * the match exists and the scatter below says which run gets closest.
 *
 * Each quantity keeps its own axis and its own units. Four rates and
 * pressures normalised onto one axis would be a prettier chart and a lie.
 */

import { useMemo } from "react";

import type { EChartsOption } from "../../charts/echarts";
import { ACCENT, axis, baseTooltip, CRIMSON, GOLD, GRID_LINE, houseOption, SLATE, ttHeader, ttRow } from "../../charts/theme";
import { ChartPanel } from "../../charts/ChartPanel";
import { Card } from "../../components/ui";
import { fmtNum } from "../../lib/format";
import { type CombineTargets, niceRange, targetOf } from "./combine";
import { METRICS, type MetricSpec } from "./metrics";

const HELP =
  "The lowest and highest each quantity reached across every solved permutation. " +
  "The gold line is the measured test. A test line outside the bar cannot be matched " +
  "by any combination of the selected inputs inside their ranges.";

/** Vertical pitch of one metric row: bar, its tick labels, its axis name. */
const ROW_PX = 80;
const TOP_PX = 30;
const BAR_PX = 30;

interface EnvelopeRow {
  spec: MetricSpec;
  min: number;
  max: number;
  target: number | null;
  /** null when no target was supplied for this quantity */
  reachable: boolean | null;
  color: string;
}

function buildRows(
  envelope: Record<string, number[]>,
  reachable: Record<string, boolean>,
  targets: CombineTargets,
): EnvelopeRow[] {
  const rows: EnvelopeRow[] = [];
  for (const spec of METRICS) {
    const span = envelope[spec.id];
    if (!Array.isArray(span) || span.length < 2) continue;
    const [min, max] = span;
    if (!Number.isFinite(min) || !Number.isFinite(max)) continue;
    const target = targetOf(targets, spec.id);
    const reach = target === null ? null : (reachable[spec.id] ?? false);
    rows.push({
      spec,
      min,
      max,
      target,
      reachable: reach,
      color: reach === null ? SLATE : reach ? ACCENT : CRIMSON,
    });
  }
  return rows;
}

function envelopeOption(rows: EnvelopeRow[]): EChartsOption {
  const grids: Record<string, unknown>[] = [];
  const xAxes: Record<string, unknown>[] = [];
  const yAxes: Record<string, unknown>[] = [];
  const series: Record<string, unknown>[] = [];

  rows.forEach((row, i) => {
    const { spec } = row;
    // A rate cannot go negative, so its axis should not either; suction
    // pressure is gauge and stays positive on any solve that converged.
    const lo = row.target !== null ? Math.min(row.min, row.target) : row.min;
    const hi = row.target !== null ? Math.max(row.max, row.target) : row.max;
    const [axMin, axMax] = niceRange(lo, hi, true);

    grids.push({ left: 84, right: 148, top: TOP_PX + i * ROW_PX, height: BAR_PX });
    xAxes.push({
      type: "value",
      gridIndex: i,
      ...axis(spec.axisName, { min: axMin, max: axMax }),
      nameGap: 26,
      splitLine: { show: true, lineStyle: { color: GRID_LINE } },
    });
    yAxes.push({
      type: "category",
      gridIndex: i,
      data: [""],
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { show: false },
      splitLine: { show: false },
    });

    // Stacked pair: an invisible bar carries the row up to its minimum, the
    // visible one draws min-to-max. ECharts bars start at zero, and a range
    // bar drawn with a custom renderItem would break the house rule.
    series.push({
      name: `${spec.id}-base`,
      type: "bar",
      stack: spec.id,
      xAxisIndex: i,
      yAxisIndex: i,
      barWidth: "58%",
      silent: true,
      itemStyle: { color: "transparent" },
      data: [row.min],
    });
    series.push({
      name: spec.label,
      type: "bar",
      stack: spec.id,
      xAxisIndex: i,
      yAxisIndex: i,
      itemStyle: { color: row.color },
      data: [row.max - row.min],
      label: {
        show: true,
        position: "right",
        distance: 8,
        color: row.color,
        fontSize: 11,
        fontWeight: 500,
        formatter: `${fmtNum(row.min, spec.dp)} to ${fmtNum(row.max, spec.dp)}`,
      },
      tooltip: {
        formatter: (): string => {
          const out = [ttHeader(spec.label)];
          out.push(ttRow(row.color, "Reachable", `${fmtNum(row.min, spec.dp)} to ${fmtNum(row.max, spec.dp)} ${spec.unit}`));
          if (row.target !== null) {
            out.push(ttRow(GOLD, "Measured test", `${fmtNum(row.target, spec.dp)} ${spec.unit}`));
            out.push(
              `<div style="margin-top:4px;color:${row.color};font-size:11px">` +
                `${row.reachable === true ? "Inside the envelope" : "Outside the envelope"}</div>`,
            );
          }
          return out.join("");
        },
      },
    });

    if (row.target !== null) {
      series.push({
        name: "",
        type: "line",
        xAxisIndex: i,
        yAxisIndex: i,
        data: [],
        silent: true,
        markLine: {
          silent: true,
          symbol: "none",
          animation: false,
          data: [
            {
              xAxis: row.target,
              lineStyle: { color: GOLD, width: 1.5, type: "dashed" },
              label: {
                show: true,
                position: "end",
                distance: 3,
                formatter: `Test ${fmtNum(row.target, spec.dp)}`,
                color: GOLD,
                fontSize: 11,
              },
            },
          ],
        },
      });
    }
  });

  return houseOption({
    tooltip: { ...baseTooltip, trigger: "item" },
    grid: grids,
    xAxis: xAxes,
    yAxis: yAxes,
    series,
  });
}

export function EnvelopeChart({
  envelope,
  reachable,
  targets,
  caption,
}: {
  envelope: Record<string, number[]>;
  reachable: Record<string, boolean>;
  targets: CombineTargets;
  /** run and failure counts, composed by the panel */
  caption: string;
}) {
  const rows = useMemo(
    () => buildRows(envelope, reachable, targets),
    [envelope, reachable, targets],
  );
  const option = useMemo(() => envelopeOption(rows), [rows]);

  if (rows.length === 0) return null;

  const missed = rows.filter((r) => r.reachable === false).map((r) => r.spec.label);
  const scored = rows.filter((r) => r.reachable !== null);

  return (
    <Card padded={false} className="p-2">
      <p className="px-2 pt-1 text-xs font-semibold text-slate-600" title={HELP}>
        Reachable Envelope
      </p>
      <p className="px-2 text-[11px] text-slate-500">{caption}</p>
      {missed.length > 0 ? (
        <p className="px-2 pt-0.5 text-[11px] font-medium text-red-700">
          Out of reach in these ranges: {missed.join(", ")}. No combination of the selected inputs
          gets there.
        </p>
      ) : scored.length > 0 ? (
        <p className="px-2 pt-0.5 text-[11px] font-medium text-blue-700">
          Every measured quantity falls inside the reachable range.
        </p>
      ) : null}
      {/* A four-row range bar has nothing to zoom into, and a disarmed brush
          keeps the page scrolling under the cursor. */}
      <ChartPanel
        option={option}
        height={TOP_PX + rows.length * ROW_PX}
        zoom={{ xAxisIndex: "none", yAxisIndex: "none" }}
      />
    </Card>
  );
}
