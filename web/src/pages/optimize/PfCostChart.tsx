/**
 * Cost of PF on S-Pad, as a curve: extra PF drawn by the modeled well
 * (negative = given back) against the oil change at every OTHER well, each
 * point resettled on the 60 Hz pump curve (pump_decision.sweep). Candidate
 * pumps for the modeled well sit on the same axes at their own extra PF, so
 * the engineer can see which sizes land on or off the general curve.
 *
 * Chart rule: mounted through ChartPanel only, tooltips via theme helpers.
 * One quantity on one axis; the header change rides in the tooltip, not on
 * a second y-scale.
 */

import { useMemo } from "react";

import type { PumpDecisionResult } from "../../api/types";
import { ChartPanel } from "../../charts/ChartPanel";
import type { EChartsOption } from "../../charts/echarts";
import {
  ACCENT,
  AXIS_LINE,
  axis,
  baseGrid,
  baseTooltip,
  GOLD,
  GRID_LINE,
  houseOption,
  nearestByX,
  SLATE,
  TEXT,
  ttHeader,
  ttNote,
  ttRow,
} from "../../charts/theme";
import { fmtNum, fmtSigned } from "../../lib/format";

type SweepRow = [number, number, number, number, boolean]; // d_q, others oil, header, d_header, extrapolated
type CandRow = [number, number, string, number | null]; // target extra PF, others oil, pump, net

/** The chart option; exported so a preview harness can render it standalone. */
export function pfCostOption(result: PumpDecisionResult): EChartsOption {
  const sweep: SweepRow[] = result.sweep.map((p) => [p.d_q, p.others_d_oil, p.header_psi, p.d_header_psi, p.extrapolated]);
  const byQ = new Map(result.sweep.map((p) => [p.d_q, p]));
  const cands: CandRow[] = result.candidates
    .filter((c) => c.net_oil !== null && Number.isFinite(c.d_pf))
    .map((c) => [c.d_pf, c.others_d_oil, c.pump_state === "installed" ? `${c.pump} (today)` : c.pump, c.net_oil]);
  const span = Math.max(...result.sweep.map((p) => Math.abs(p.d_q)), 1);
  const dq = result.sensitivity.d_q;
  const marks = [dq, -dq]
    .map((q) => byQ.get(q) ?? null)
    .filter((p): p is NonNullable<typeof p> => p !== null)
    .map((p) => ({
      coord: [p.d_q, p.others_d_oil],
      value: `${fmtSigned(p.d_q)} BPD PF: ${fmtSigned(p.others_d_oil, 1)} BOPD`,
      // given back: label above-left; drawn: below-right - clear of the
      // candidate labels, which sit to the right of their own markers
      label: p.d_q < 0 ? { position: "top", offset: [-40, -6] } : { position: "bottom", offset: [-50, 10] },
    }));
  const padWide = result.target === null;
  const lineName = padWide ? "All wells (extra PF on the curve)" : "Other wells (extra PF on the curve)";
  const candName = padWide ? "" : `Pump sizes for ${result.target}`;
  const who = padWide ? "the pad" : result.target;

  return houseOption({
    grid: { ...baseGrid, top: 44, left: 64, right: 32, bottom: 52 },
    // One series in pad-wide mode: the title names it, so no legend box.
    legend: padWide ? { show: false } : { top: 4, left: 8, itemWidth: 14, textStyle: { color: TEXT, fontSize: 12 }, data: [lineName, candName] },
    tooltip: {
      ...baseTooltip,
      trigger: "axis",
      // Pad-wide lists up to 8 wells: keep the box inside the chart.
      confine: true,
      axisPointer: { type: "cross", lineStyle: { color: AXIS_LINE }, crossStyle: { color: AXIS_LINE }, label: { precision: 0 } },
      formatter: (raw: unknown) => {
        const list = (Array.isArray(raw) ? raw : [raw]) as { axisValue?: unknown; axisDim?: string }[];
        const x = list.find((p) => p.axisDim === "x" && typeof p.axisValue === "number")?.axisValue as number | undefined;
        if (x === undefined) return "";
        const pt = nearestByX(sweep, x);
        const out: string[] = [];
        if (pt) {
          const p = byQ.get(pt[0]);
          out.push(ttHeader(`${fmtSigned(pt[0])} BPD PF ${pt[0] >= 0 ? "drawn" : "given back"}`));
          out.push(ttRow(ACCENT, padWide ? "Pad oil" : "Other wells' oil", `${fmtSigned(pt[1], 1)} BOPD`));
          out.push(ttRow(AXIS_LINE, "Header", `${fmtNum(pt[2])} psi (${fmtSigned(pt[3])})`));
          const top = Object.entries(p?.wells ?? {})
            .filter(([, v]) => Math.abs(v) >= 0.05)
            .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
            .slice(0, padWide ? 8 : 4);
          for (const [w, v] of top) out.push(ttRow(SLATE, `  ${w}`, `${fmtSigned(v, 1)} BOPD`));
          if (pt[4]) out.push(ttNote("Header beyond the modeled range: rates held at the nearest modeled point."));
        }
        const near = cands.filter((c) => Math.abs(c[0] - x) <= span / 40);
        for (const c of near) {
          out.push(ttRow(GOLD, c[2], `others ${fmtSigned(c[1], 1)}, net ${fmtSigned(c[3], 1)} BOPD`));
        }
        return out.join("");
      },
    },
    xAxis: { ...axis(`Extra PF drawn by ${who}, BPD (negative = given back)`, { min: -span, max: span }), type: "value" },
    yAxis: { ...axis(padWide ? "Oil change across all wells, BOPD" : "Oil change at the other wells, BOPD"), type: "value", nameGap: 44 },
    series: [
      {
        name: lineName,
        type: "line",
        data: sweep,
        showSymbol: false,
        lineStyle: { color: ACCENT, width: 2 },
        itemStyle: { color: ACCENT },
        z: 3,
        markPoint: {
          symbol: "circle",
          symbolSize: 8,
          itemStyle: { color: ACCENT, borderColor: "#ffffff", borderWidth: 2 },
          label: {
            show: true, color: TEXT, fontSize: 11, fontWeight: 500,
            backgroundColor: "#ffffff", borderColor: GRID_LINE, borderWidth: 1, borderRadius: 4, padding: [2, 5],
            formatter: (p: { value?: unknown }) => String(p.value ?? ""),
          },
          data: marks,
        },
        markLine: {
          silent: true,
          symbol: "none",
          label: { show: false },
          lineStyle: { color: AXIS_LINE, type: "dashed", width: 1 },
          data: [{ xAxis: 0 }, { yAxis: 0 }],
        },
      },
      {
        name: candName,
        type: "scatter",
        data: cands,
        symbol: "diamond",
        symbolSize: 10,
        itemStyle: { color: GOLD, borderColor: "#ffffff", borderWidth: 2 },
        label: { show: true, position: "right", color: TEXT, fontSize: 10, formatter: (p: { value?: unknown }) => String((p.value as CandRow)[2]) },
        labelLayout: { hideOverlap: true },
        z: 4,
      },
    ],
  });
}

export function PfCostChart({ result }: { result: PumpDecisionResult }) {
  const opt = useMemo(() => (result.sweep?.length ? pfCostOption(result) : null), [result]);
  if (!opt) return null;
  return (
    <div className="rounded-md bg-white p-1 ring-1 ring-slate-200">
      <ChartPanel option={opt} height={320} zoom={{ xAxisIndex: [0], yAxisIndex: [0] }} />
    </div>
  );
}
