/**
 * CFP run result charts - the three views the moves methodology earns:
 *
 * 1. Oil vs discharge frontier - the equal-slope efficiency frontier with
 *    today and the best plan marked, and the 2,900 psi trip line. Where we
 *    sit vs where we could be; the slope near today IS the shadow price.
 * 2. Plan bridge (waterfall) - today's oil, one bar per plan action (its
 *    OWN oil delta at the plan pressure), then the plant-pressure feedback
 *    residual, landing on the plan total. The residual is the honest
 *    encoding of the anchored plant's pressure endogeneity - the actions'
 *    own deltas deliberately do NOT sum to the fleet delta.
 * 3. Today vs plan per well (dumbbell, sorted by delta) - where the barrels
 *    actually move. Solid dot = today (measured state), hollow ring = plan
 *    (house fill semantics); connector green for gains, crimson for losses.
 */

import { useMemo } from "react";

import type { CfpRunResult } from "../../api/types";
import type { EChartsOption } from "../../charts/echarts";
import { ACCENT, axis, baseGrid, baseTooltip, CRIMSON, houseOption, SLATE } from "../../charts/theme";
import { ChartPanel } from "../../charts/ChartPanel";
import { Card } from "../../components/ui";
import { fmtNum } from "../../lib/format";

// Petroleum phase convention, matching the pump-history strip.
const OIL_GREEN = "#2E7D32";

function frontierOption(result: CfpRunResult): EChartsOption | null {
  const s = result.summary;
  const pts = [...s.frontier].sort((a, b) => a.pressure - b.pressure);
  if (pts.length === 0) return null;
  return houseOption({
    tooltip: {
      ...baseTooltip,
      trigger: "item",
      formatter: (raw: unknown): string => {
        const p = raw as { value: [number, number]; seriesName: string };
        return `${p.seriesName}<br/>Discharge: ${fmtNum(p.value[0])} psi<br/>Oil: ${fmtNum(p.value[1])} BOPD`;
      },
    },
    legend: { top: 4, right: 8, textStyle: { fontSize: 12 } },
    grid: { ...baseGrid, top: 40 },
    xAxis: { type: "value", ...axis("PW discharge (psi)", { min: "dataMin", max: 2950 }) },
    yAxis: { type: "value", ...axis("Modeled oil, run wells (BOPD)") },
    series: [
      {
        name: "Frontier",
        type: "line",
        showSymbol: true,
        symbolSize: 5,
        data: pts.map((p) => [p.pressure, p.oil]),
        lineStyle: { color: ACCENT, width: 2 },
        itemStyle: { color: ACCENT },
        markLine: {
          silent: true,
          symbol: "none",
          lineStyle: { color: CRIMSON, type: "dashed", width: 1 },
          label: { formatter: "2,900 trip", color: CRIMSON, fontSize: 11 },
          data: [{ xAxis: 2900 }],
        },
      },
      {
        name: "Today",
        type: "scatter",
        symbolSize: 12,
        data: [[s.today.pressure, s.today.oil]],
        itemStyle: { color: SLATE, borderColor: "#0f172a", borderWidth: 1 },
      },
      ...(s.plan
        ? [
            {
              name: "Plan",
              type: "scatter" as const,
              symbol: "diamond",
              symbolSize: 14,
              data: [[s.plan.pressure, s.plan.oil]],
              itemStyle: { color: CRIMSON, borderColor: "#0f172a", borderWidth: 1 },
            },
          ]
        : []),
    ],
  });
}

function bridgeOption(result: CfpRunResult): EChartsOption | null {
  const s = result.summary;
  const plan = s.plan;
  if (!plan || plan.actions.length === 0) return null;

  // Largest own-oil movers first; everything past 8 folds into one bar so
  // the bridge stays readable on a busy plan.
  const sorted = [...plan.actions].sort(
    (a, b) => Math.abs(b.own_oil_delta) - Math.abs(a.own_oil_delta),
  );
  const shown = sorted.slice(0, 8);
  const restDelta = sorted.slice(8).reduce((a, x) => a + x.own_oil_delta, 0);
  const ownSum = sorted.reduce((a, x) => a + x.own_oil_delta, 0);
  const feedback = plan.oil - s.today.oil - ownSum;

  const steps: { label: string; delta: number }[] = [
    ...shown.map((a) => ({ label: a.well, delta: a.own_oil_delta })),
    ...(sorted.length > 8 ? [{ label: `${sorted.length - 8} more`, delta: restDelta }] : []),
    { label: "pressure feedback", delta: feedback },
  ];

  const categories = ["Today", ...steps.map((x) => x.label), "Plan"];
  const base: (number | null)[] = [0];
  const rise: { value: number | null; itemStyle?: { color: string } }[] = [
    { value: s.today.oil, itemStyle: { color: SLATE } },
  ];
  let level = s.today.oil;
  for (const st of steps) {
    base.push(Math.min(level, level + st.delta));
    rise.push({
      value: Math.abs(st.delta),
      itemStyle: { color: st.delta >= 0 ? OIL_GREEN : CRIMSON },
    });
    level += st.delta;
  }
  base.push(0);
  rise.push({ value: plan.oil, itemStyle: { color: ACCENT } });

  return houseOption({
    tooltip: {
      ...baseTooltip,
      trigger: "axis",
      formatter: (raw: unknown): string => {
        const arr = raw as { dataIndex: number; axisValue: string }[];
        const i = arr[0]?.dataIndex ?? 0;
        if (i === 0) return `Today: ${fmtNum(s.today.oil)} BOPD`;
        if (i === categories.length - 1) return `Plan: ${fmtNum(plan.oil)} BOPD`;
        const st = steps[i - 1];
        return `${st.label}: ${st.delta >= 0 ? "+" : ""}${fmtNum(st.delta)} BOPD`;
      },
    },
    grid: { ...baseGrid, top: 24, bottom: 48 },
    xAxis: {
      type: "category",
      data: categories,
      axisLabel: { rotate: 30, fontSize: 11 },
    },
    yAxis: { type: "value", ...axis("Modeled oil, run wells (BOPD)") },
    series: [
      { type: "bar", stack: "bridge", itemStyle: { color: "transparent" }, emphasis: { disabled: true }, data: base, silent: true },
      { type: "bar", stack: "bridge", barWidth: "55%", data: rise },
    ],
  });
}

function dumbbellOption(result: CfpRunResult): { option: EChartsOption; height: number } | null {
  const rows = [...result.wells]
    .filter((w) => w.baseline_oil > 0 || w.plan_oil > 0)
    .sort((a, b) => (b.plan_oil - b.baseline_oil) - (a.plan_oil - a.baseline_oil));
  if (rows.length === 0) return null;

  const wells = rows.map((r) => r.well);
  const height = Math.max(260, rows.length * 26 + 90);

  return {
    option: houseOption({
      tooltip: {
        ...baseTooltip,
        trigger: "item",
        formatter: (raw: unknown): string => {
          const p = raw as { dataIndex: number };
          const r = rows[p.dataIndex];
          return [
            `<b>${r.well}</b> (${r.pad}-Pad)`,
            `Today: ${r.baseline_label} - ${fmtNum(r.baseline_oil)} BOPD`,
            `Plan: ${r.plan_label} - ${fmtNum(r.plan_oil)} BOPD`,
            `Delta: ${r.plan_oil - r.baseline_oil >= 0 ? "+" : ""}${fmtNum(r.plan_oil - r.baseline_oil)} BOPD`,
          ].join("<br/>");
        },
      },
      legend: { top: 4, right: 8, textStyle: { fontSize: 12 } },
      grid: { ...baseGrid, top: 36, left: 76 },
      xAxis: { type: "value", ...axis("Oil (BOPD)", { min: 0 }) },
      yAxis: { type: "category", data: wells, inverse: true, axisLabel: { fontSize: 11 } },
      series: [
        {
          // connector segments, colored by direction
          type: "custom",
          silent: true,
          renderItem: (params: { dataIndex: number }, api: { coord: (v: [number, number]) => [number, number] }) => {
            const r = rows[params.dataIndex];
            const p0 = api.coord([r.baseline_oil, params.dataIndex]);
            const p1 = api.coord([r.plan_oil, params.dataIndex]);
            const gain = r.plan_oil >= r.baseline_oil;
            return {
              type: "line",
              shape: { x1: p0[0], y1: p0[1], x2: p1[0], y2: p1[1] },
              style: { stroke: gain ? OIL_GREEN : CRIMSON, lineWidth: 2 },
            };
          },
          data: rows.map((r) => [r.baseline_oil, r.plan_oil]),
        },
        {
          name: "Today",
          type: "scatter",
          symbolSize: 8,
          data: rows.map((r, i) => [r.baseline_oil, i]),
          itemStyle: { color: SLATE },
        },
        {
          name: "Plan",
          type: "scatter",
          symbolSize: 9,
          data: rows.map((r, i) => [r.plan_oil, i]),
          // hollow ring = plan (fictitious until executed), house semantics
          itemStyle: { color: "#ffffff", borderColor: ACCENT, borderWidth: 2 },
        },
      ],
    }),
    height,
  };
}

export function CfpResultCharts({ result }: { result: CfpRunResult }) {
  const frontier = useMemo(() => frontierOption(result), [result]);
  const bridge = useMemo(() => bridgeOption(result), [result]);
  const dumbbell = useMemo(() => dumbbellOption(result), [result]);

  return (
    <div className="space-y-3">
      <div className="grid items-start gap-3 xl:grid-cols-2">
        {frontier && (
          <Card padded={false} className="p-2">
            <p
              className="px-2 pt-1 text-xs font-semibold text-slate-600"
              title="Total modeled oil across the run's wells at each PW discharge pressure - the efficiency frontier. Slate dot = today, crimson diamond = best plan, dashed line = 2,900 psi trip."
            >
              Modeled oil vs PW discharge
            </p>
            <ChartPanel option={frontier} height={300} zoom={{ xAxisIndex: [0], yAxisIndex: [0] }} />
          </Card>
        )}
        {bridge && (
          <Card padded={false} className="p-2">
            <p className="px-2 pt-1 text-xs font-semibold text-slate-600">
              Today to plan, by action
            </p>
            <ChartPanel option={bridge} height={300} zoom={{ xAxisIndex: "none", yAxisIndex: "none" }} />
          </Card>
        )}
      </div>
      {dumbbell && (
        <Card padded={false} className="p-2">
          <p className="px-2 pt-1 text-xs font-semibold text-slate-600">
            Today vs plan by well
          </p>
          <ChartPanel option={dumbbell.option} height={dumbbell.height} zoom={{ xAxisIndex: [0], yAxisIndex: "none" }} />
        </Card>
      )}
    </div>
  );
}
