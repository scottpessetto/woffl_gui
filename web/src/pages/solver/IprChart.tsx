/**
 * Vogel IPR chart: fitted curve + test scatter colored by recency + the
 * modeled operating point. Mirror of woffl/gui/ipr_viz.py:create_ipr_plotly
 * (axes, series, Res P / Qmax annotation) rebuilt on the house ECharts
 * theme. The curve is recomputed CLIENT-SIDE from the anchor at the SIDEBAR
 * reservoir pressure, so dragging ResP redraws instantly.
 */

import { useMemo, useState } from "react";

import type { IprFitResponse, JpInstallRow, SimParams, SolveResult, WellTestRow } from "../../api/types";
import type { EChartsOption } from "../../charts/echarts";
import { ACCENT, axis, baseGrid, baseTooltip, CRIMSON, houseOption, SLATE, TEXT, VIRIDIS } from "../../charts/theme";
import { ChartPanel } from "../../charts/ChartPanel";
import { Card, WarnNote } from "../../components/ui";
import { fmtDate, fmtNum, daysAgo } from "../../lib/format";
import { iprCurveFromAnchor, vogelQmax } from "../../lib/vogel";

import { pumpLabelAt } from "./selection";

/** [total_fluid, bhp, daysAgo, date, oil, pumpLabel] */
type TestPoint = [number, number, number, string, number | null, string | null];

export function IprChart({
  tests,
  fit,
  params,
  solve,
  compareTest,
  installs,
}: {
  tests: WellTestRow[];
  fit: IprFitResponse | null;
  params: SimParams;
  solve: SolveResult | null;
  compareTest: WellTestRow | null;
  installs: JpInstallRow[];
}) {
  // Old GUI: checkbox "Show JP label inside each test point"
  // (mva_show_jp_labels_{well}); per-well because the workbench remounts.
  const [showJpLabels, setShowJpLabels] = useState(false);
  const option = useMemo<EChartsOption>(() => {
    // Anchor precedence: server fit > selected comparison test > raw sidebar
    // inflow - always re-evaluated at the SIDEBAR reservoir pressure.
    let anchorQwf = params.qwf;
    let anchorPwf = params.pwf;
    if (fit) {
      anchorQwf = fit.coeffs.qwf;
      anchorPwf = fit.coeffs.pwf;
    } else if (compareTest && compareTest.total_fluid !== null && compareTest.bhp !== null) {
      anchorQwf = compareTest.total_fluid;
      anchorPwf = compareTest.bhp;
    }
    const curve = iprCurveFromAnchor(anchorQwf, anchorPwf, params.pres);
    const qmax = vogelQmax(anchorQwf, anchorPwf, params.pres);
    const formWc = params.form_wc;

    const points: TestPoint[] = [];
    for (const t of tests) {
      if (t.total_fluid === null || t.bhp === null) continue;
      points.push([t.total_fluid, t.bhp, daysAgo(t.date) ?? 0, t.date, t.oil, pumpLabelAt(installs, t.date)]);
    }
    const maxDays = Math.max(1, ...points.map((p) => p[2]));

    return houseOption({
      tooltip: { ...baseTooltip, trigger: "item" },
      legend: { top: 4, right: 8, textStyle: { fontSize: 12 } },
      grid: { ...baseGrid, top: 48, right: 96 },
      xAxis: { type: "value", ...axis("Total Fluid Rate (BPD)", { min: 0 }) },
      yAxis: { type: "value", ...axis("Bottom Hole Pressure (psi)", { min: 0 }) },
      visualMap:
        points.length > 0
          ? [
              {
                type: "continuous",
                dimension: 2,
                seriesIndex: 1,
                min: 0,
                max: maxDays,
                // VIRIDIS is dark-to-bright; reversed so RECENT tests (low
                // days-ago) render bright yellow and old ones fade to purple.
                inRange: { color: [...VIRIDIS].reverse() },
                right: 0,
                top: "middle",
                itemHeight: 130,
                calculable: false,
                text: ["Days Ago", ""],
                textStyle: { color: SLATE, fontSize: 11 },
              },
            ]
          : [],
      graphic: [
        {
          type: "text",
          left: 64,
          top: 8,
          style: {
            text: `Res P: ${fmtNum(params.pres)} psi   Qmax: ${fmtNum(qmax)} BPD`,
            fill: TEXT,
            fontSize: 12,
            fontWeight: 500,
          },
        },
      ],
      series: [
        {
          name: "Vogel IPR",
          type: "line",
          showSymbol: false,
          lineStyle: { color: ACCENT, width: 3 },
          itemStyle: { color: ACCENT },
          data: curve ? curve.fluid.map((f, i) => [f, curve.bhp[i]]) : [],
          tooltip: {
            formatter: (raw: unknown): string => {
              // ECharts item-tooltip param: value is this series' [fluid, bhp] pair.
              const p = raw as { value: [number, number] };
              const v = p.value;
              return [
                `Fluid: ${fmtNum(v[0])} BPD`,
                `Oil: ${fmtNum(v[0] * (1 - formWc))} BOPD`,
                `BHP: ${fmtNum(v[1])} psi`,
              ].join("<br/>");
            },
          },
        },
        {
          name: "Test Data",
          type: "scatter",
          symbolSize: showJpLabels ? 26 : 11,
          itemStyle: { borderColor: "#0f172a", borderWidth: 1 },
          // No animation: zoom filtering animates leaving points, and label
          // formatters run on those transitional elements with value
          // undefined - one throw mid-render blanks the whole canvas.
          animation: false,
          label: showJpLabels
            ? {
                show: true,
                position: "inside",
                fontSize: 9,
                fontWeight: 600,
                color: "#ffffff",
                textBorderColor: "rgba(15,23,42,0.85)",
                textBorderWidth: 2,
                formatter: (raw: unknown): string => {
                  // Defensive narrowing: during zoom transitions ECharts can
                  // invoke this for filtered-out points with value undefined.
                  if (raw && typeof raw === "object" && "value" in raw) {
                    const v = raw.value;
                    if (Array.isArray(v) && typeof v[5] === "string") return v[5];
                  }
                  return "";
                },
              }
            : { show: false },
          data: points,
          tooltip: {
            formatter: (raw: unknown): string => {
              // ECharts item-tooltip param: value is this series' TestPoint datum.
              const p = raw as { value: TestPoint };
              const v = p.value;
              return [
                `<b>${fmtDate(v[3])}</b>`,
                `Fluid: ${fmtNum(v[0])} BPD`,
                `Oil: ${v[4] !== null ? `${fmtNum(v[4])} BOPD` : "-"}`,
                `BHP: ${fmtNum(v[1])} psi`,
                `Pump: ${v[5] ?? "-"}`,
                `Days ago: ${fmtNum(v[2])}`,
              ].join("<br/>");
            },
          },
        },
        {
          name: "Model",
          type: "scatter",
          symbol: "diamond",
          symbolSize: 14,
          itemStyle: { color: CRIMSON, borderColor: "#0f172a", borderWidth: 1 },
          data: solve ? [[solve.qoil_std + solve.fwat_bwpd, solve.psu]] : [],
          tooltip: {
            formatter: (raw: unknown): string => {
              // ECharts item-tooltip param: value is the single [fluid, psu] point.
              const p = raw as { value: [number, number] };
              const v = p.value;
              return `<b>Modeled operating point</b><br/>Fluid: ${fmtNum(v[0])} BPD<br/>Suction: ${fmtNum(v[1])} psi`;
            },
          },
        },
      ],
    });
  }, [tests, fit, params, solve, compareTest, installs, showJpLabels]);


  return (
    <Card>
      <div className="mb-1 flex justify-end">
        <label className="flex cursor-pointer items-center gap-2 text-xs text-slate-600">
          <input
            type="checkbox"
            checked={showJpLabels}
            onChange={(e) => setShowJpLabels(e.target.checked)}
            className="h-4 w-4 rounded border-slate-300 accent-blue-600"
          />
          Show JP label inside each test point
        </label>
      </div>
      <ChartPanel option={option} height={520} zoom={{ xAxisIndex: [0], yAxisIndex: [0] }} />
      {fit?.weak && (
        <WarnNote className="mt-2">
          IPR fit is weak (R2 {fmtNum(fit.coeffs.r2, 2)}) - treat the curve as a sketch
        </WarnNote>
      )}
    </Card>
  );
}
