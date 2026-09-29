/**
 * Header page chart options (ECharts, rendered through ChartPanel).
 */

import type { EChartsOption } from "../../charts/echarts";
import { ACCENT, CATEGORY20, CRIMSON, GOLD, SLATE, axis, houseOption, ttHeader, ttNote, ttRow } from "../../charts/theme";
import type {
  HeaderBoardRow, HeaderCorrelation, HeaderRunResult, HeaderRunRow, HeaderValidationRow, HeaderWellDetail,
} from "../../api/types";
import { fmtNum } from "../../lib/format";

const LIFT_COLOR: Record<string, string> = { ESP: ACCENT, JP: GOLD, "gas-lift": CATEGORY20[4], flowing: SLATE };

export const liftColor = (lift: string): string => LIFT_COLOR[lift] ?? SLATE;

/** Oil change per well, sorted, coloured by lift type. */
export function impactBars(rows: HeaderRunRow[]): EChartsOption | null {
  const pts = rows
    .filter((r) => r.outcome === "modeled" && typeof r.d_oil === "number")
    .map((r) => ({ well: r.well, d: r.d_oil as number, lift: r.lift, liq: r.d_liq ?? null, bhp: r.d_bhp ?? null }))
    .sort((a, b) => a.d - b.d);
  if (!pts.length) return null;
  return houseOption({
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "shadow" },
      formatter: (raw: unknown) => {
        const p = pts[(raw as { dataIndex: number }[])[0].dataIndex];
        return (
          ttHeader(`${p.well} (${p.lift})`) +
          ttRow(liftColor(p.lift), "Oil change", `${fmtNum(p.d, 1)} BOPD`) +
          ttRow(SLATE, "Liquid change", `${fmtNum(p.liq, 1)} BLPD`) +
          ttRow(SLATE, "BHP change", `${fmtNum(p.bhp, 1)} psi`)
        );
      },
    },
    grid: { top: 16, left: 64, right: 20, bottom: 72 },
    xAxis: { type: "category", data: pts.map((p) => p.well), ...axis(""), axisLabel: { rotate: 60, fontSize: 10 } },
    yAxis: { type: "value", ...axis("Oil change (BOPD)") },
    series: [{
      type: "bar",
      data: pts.map((p) => ({ value: p.d, itemStyle: { color: liftColor(p.lift) } })),
    }],
  });
}

/** Measured closed-loop slope vs liquid rate for one lift group, with the
 *  fitted correlation and the wells that borrow it. */
export function correlationChart(
  key: string,
  corr: HeaderCorrelation,
  rows: HeaderBoardRow[],
): EChartsOption | null {
  if (!corr.points.length) return null;
  const c = corr.correlation;
  const borrowers = rows
    .filter((r) => r.corr_group === key && (r.gauge_auto_bad || r.measured.status !== "measured") && r.corr_options[key] && r.liquid)
    .map((r) => ({ well: r.well, q: r.liquid as number, s: r.corr_options[key].slope }));
  const reservoirs = [...new Set(corr.points.map((p) => p.reservoir || "unknown"))].sort();
  const line: [number, number][] = [];
  if (c) {
    const n = 40;
    for (let i = 0; i <= n; i++) {
      const q = c.q_min * Math.pow(c.q_max / c.q_min, i / n);
      line.push([q, Math.min(1.2, Math.max(0, c.a + c.b * Math.log(q)))]);
    }
  }
  return houseOption({
    tooltip: {
      trigger: "item",
      formatter: (raw: unknown) => {
        const p = raw as { seriesName: string; data: [number, number, string?] };
        if (!p.data[2]) return ttHeader("Correlation") + ttRow(ACCENT, "slope", fmtNum(p.data[1], 2));
        return ttHeader(p.data[2]) + ttRow(SLATE, "Liquid", `${fmtNum(p.data[0], 0)} BLPD`) +
          ttRow(SLATE, "dBHP/dWHP", fmtNum(p.data[1], 2)) + ttNote(p.seriesName);
      },
    },
    legend: { top: 0, textStyle: { fontSize: 11 } },
    grid: { top: 32, left: 56, right: 20, bottom: 44 },
    xAxis: { type: "log", ...axis("Liquid rate (BLPD, log)"), min: "dataMin", max: "dataMax" },
    yAxis: { type: "value", ...axis("BHP / WHP slope (psi/psi)"), min: 0 },
    series: [
      ...reservoirs.map((res, i) => ({
        name: `Measured - ${res}`,
        type: "scatter" as const,
        symbolSize: 9,
        itemStyle: { color: CATEGORY20[(i * 2) % CATEGORY20.length] },
        data: corr.points.filter((p) => (p.reservoir || "unknown") === res).map((p) => [p.q_liq, p.slope, p.well]),
      })),
      {
        name: "Borrows the correlation",
        type: "scatter" as const,
        symbol: "emptyCircle",
        symbolSize: 10,
        itemStyle: { color: CRIMSON },
        data: borrowers.map((b) => [b.q, b.s, b.well]),
      },
      ...(c ? [{
        name: c.kind === "trend" ? `s = ${c.a.toFixed(2)} ${c.b < 0 ? "-" : "+"} ${Math.abs(c.b).toFixed(3)} ln(q)` : `median ${c.a.toFixed(2)}`,
        type: "line" as const,
        showSymbol: false,
        lineStyle: { color: SLATE, type: "dashed" as const, width: 2 },
        data: line,
      }] : []),
    ],
  });
}

/** Predicted vs measured BHP change around an observed event. */
export function validationChart(rows: HeaderValidationRow[]): EChartsOption | null {
  const pts = rows.filter((r) => typeof r.d_bhp_pred === "number" && typeof r.d_bhp_meas === "number");
  if (!pts.length) return null;
  const used = pts.filter((r) => !r.operational);
  const vals = used.flatMap((r) => [r.d_bhp_pred as number, r.d_bhp_meas as number]);
  const lo = Math.min(0, ...vals) - 2;
  const hi = Math.max(0, ...vals) + 2;
  return houseOption({
    tooltip: {
      trigger: "item",
      formatter: (raw: unknown) => {
        const d = (raw as { data: [number, number, string] }).data;
        return ttHeader(d[2]) + ttRow(ACCENT, "Predicted", `${fmtNum(d[0], 1)} psi`) + ttRow(SLATE, "Measured", `${fmtNum(d[1], 1)} psi`);
      },
    },
    legend: { top: 0, textStyle: { fontSize: 11 } },
    grid: { top: 32, left: 56, right: 20, bottom: 44 },
    xAxis: { type: "value", ...axis("Predicted BHP change (psi)"), min: lo, max: hi },
    yAxis: { type: "value", ...axis("Measured BHP change (psi)"), min: lo, max: hi },
    series: [
      { name: "Gauged wells", type: "scatter", symbolSize: 9, itemStyle: { color: ACCENT },
        data: used.map((r) => [r.d_bhp_pred, r.d_bhp_meas, r.well]) },
      { name: "1:1", type: "line", showSymbol: false, lineStyle: { color: SLATE, type: "dashed" }, data: [[lo, lo], [hi, hi]] },
    ],
  });
}

/** Oil change vs a uniform header change on every selected pad: total with
 *  its low/high envelope, one line per pad, and a marker at the run's change. */
export function responseCurve(result: HeaderRunResult, marker: number | null): EChartsOption | null {
  const c = result.curve;
  if (!c || !c.grid.length) return null;
  const pads = Object.keys(c.pads);
  const band = (ys: number[]) => c.grid.map((x, i) => [x, ys[i]]);
  return houseOption({
    tooltip: {
      trigger: "axis",
      formatter: (raw: unknown) => {
        const i = (raw as { dataIndex: number }[])[0].dataIndex;
        return (
          ttHeader(`${c.grid[i] > 0 ? "+" : ""}${c.grid[i]} psi header`) +
          ttRow(ACCENT, "All pads", `${fmtNum(c.total.base[i], 1)} BOPD`) +
          ttRow(SLATE, "Range", `${fmtNum(c.total.lo[i], 1)} to ${fmtNum(c.total.hi[i], 1)}`) +
          pads.map((p, k) => ttRow(CATEGORY20[(k * 2 + 2) % CATEGORY20.length], `${p}-Pad`, `${fmtNum(c.pads[p].base[i], 1)} BOPD`)).join("")
        );
      },
    },
    legend: { top: 0, textStyle: { fontSize: 11 }, data: ["All pads", "Range", ...pads.map((p) => `${p}-Pad`)] },
    grid: { top: 32, left: 64, right: 24, bottom: 44 },
    xAxis: { type: "value", ...axis("Header change on every selected pad (psi)"), min: c.grid[0], max: c.grid[c.grid.length - 1] },
    yAxis: { type: "value", ...axis("Oil change (BOPD)") },
    series: [
      // Envelope as a stacked pair: lower edge invisible, band = hi - lo.
      { name: "lo-edge", type: "line", data: band(c.grid.map((_, i) => Math.min(c.total.lo[i], c.total.hi[i]))),
        showSymbol: false, lineStyle: { opacity: 0 }, stack: "range", stackStrategy: "all", silent: true, tooltip: { show: false } },
      { name: "Range", type: "line", data: band(c.grid.map((_, i) => Math.abs(c.total.hi[i] - c.total.lo[i]))),
        showSymbol: false, lineStyle: { opacity: 0 }, stack: "range", stackStrategy: "all", areaStyle: { color: ACCENT, opacity: 0.12 },
        itemStyle: { color: ACCENT, opacity: 0.35 }, silent: true },
      { name: "All pads", type: "line", data: band(c.total.base), showSymbol: false, lineStyle: { color: ACCENT, width: 2.5 },
        itemStyle: { color: ACCENT },
        markLine: marker === null ? undefined : {
          symbol: "none", silent: true, lineStyle: { color: CRIMSON, type: "dashed" },
          label: { formatter: `${marker > 0 ? "+" : ""}${marker} psi`, color: CRIMSON }, data: [{ xAxis: marker }],
        } },
      ...pads.map((p, k) => ({
        name: `${p}-Pad`, type: "line" as const, data: band(c.pads[p].base), showSymbol: false,
        lineStyle: { width: 1.5, color: CATEGORY20[(k * 2 + 2) % CATEGORY20.length] },
        itemStyle: { color: CATEGORY20[(k * 2 + 2) % CATEGORY20.length] },
      })),
    ],
  });
}

// ── review cards: compact per-well charts ────────────────────────────────────

const CARD_GRID = { top: 26, left: 44, right: 44, bottom: 28 };
const ms = (t: string) => new Date(t).getTime();

/** BHP (left) and WHP / header (right) over the fit window - gauge health at a glance. */
export function cardTrend(detail: HeaderWellDetail): EChartsOption | null {
  if (!detail.trend.length) return null;
  const pts = (k: "bhp" | "whp" | "headerp") =>
    detail.trend.filter((p) => typeof p[k] === "number").map((p) => [ms(p.t), p[k] as number]);
  return houseOption({
    tooltip: {
      trigger: "axis",
      formatter: (raw: unknown) => {
        const arr = raw as { seriesName: string; value: [number, number]; color: string }[];
        if (!arr.length) return "";
        return ttHeader(new Date(arr[0].value[0]).toLocaleString()) +
          arr.map((a) => ttRow(a.color, a.seriesName, `${fmtNum(a.value[1], 0)} psi`)).join("");
      },
    },
    legend: { top: 0, itemWidth: 12, textStyle: { fontSize: 10 } },
    grid: CARD_GRID,
    xAxis: { type: "time", ...axis(""), axisLabel: { fontSize: 9 } },
    yAxis: [
      { type: "value", ...axis(""), scale: true, axisLabel: { fontSize: 9 } },
      { type: "value", ...axis(""), scale: true, axisLabel: { fontSize: 9 }, splitLine: { show: false } },
    ],
    series: [
      { name: "BHP", type: "line", showSymbol: false, data: pts("bhp"), lineStyle: { width: 1.2, color: ACCENT }, itemStyle: { color: ACCENT } },
      { name: "WHP", type: "line", yAxisIndex: 1, showSymbol: false, data: pts("whp"), lineStyle: { width: 1, color: GOLD }, itemStyle: { color: GOLD } },
      { name: "Header", type: "line", yAxisIndex: 1, showSymbol: false, data: pts("headerp"), lineStyle: { width: 1, color: SLATE }, itemStyle: { color: SLATE } },
    ],
  });
}

export interface SlopeLine {
  label: string;
  value: number;
  color: string;
  dashed?: boolean;
}

/** Each day's within-day BHP~WHP slope (filled when its r2 counts), with the
 *  candidate relations as horizontal lines. */
export function cardSlopes(detail: HeaderWellDetail, lines: SlopeLine[]): EChartsOption | null {
  if (!detail.daily.length && !lines.length) return null;
  const good = detail.daily.filter((d) => d.slope !== null && (d.r2 ?? 0) >= detail.r2_day_min);
  const weak = detail.daily.filter((d) => d.slope !== null && (d.r2 ?? 0) < detail.r2_day_min);
  const clampY = (v: number) => Math.max(-0.5, Math.min(1.8, v));
  const toPts = (arr: typeof good) => arr.map((d) => [ms(d.day), clampY(d.slope as number), d.r2 ?? 0]);
  return houseOption({
    tooltip: {
      trigger: "item",
      formatter: (raw: unknown) => {
        const p = raw as { seriesName: string; value: [number, number, number] };
        if (p.seriesName.startsWith("line:")) return "";
        return ttHeader(new Date(p.value[0]).toLocaleDateString()) + ttRow(ACCENT, "dBHP/dWHP", fmtNum(p.value[1], 2)) +
          ttRow(SLATE, "r2", fmtNum(p.value[2], 2)) + ttNote(p.seriesName);
      },
    },
    grid: { ...CARD_GRID, right: 12 },
    xAxis: { type: "time", ...axis(""), axisLabel: { fontSize: 9 } },
    yAxis: { type: "value", ...axis(""), min: -0.5, max: 1.8, axisLabel: { fontSize: 9 } },
    series: [
      { name: "Day fit counts (r2 above the bar)", type: "scatter", symbolSize: 6, itemStyle: { color: ACCENT }, data: toPts(good) },
      { name: "Weak day (not counted)", type: "scatter", symbolSize: 5, itemStyle: { color: "transparent", borderColor: SLATE, borderWidth: 1 }, data: toPts(weak) },
      {
        name: "line:refs", type: "line", data: [], silent: true,
        markLine: {
          symbol: "none", silent: true,
          // Alternate label corners so near-equal lines stay readable.
          data: lines.map((l, i) => ({
            yAxis: l.value, lineStyle: { color: l.color, type: l.dashed ? "dashed" : "solid", width: 1.5 },
            label: {
              formatter: `${l.label} ${l.value.toFixed(2)}`, fontSize: 9, color: l.color,
              position: (["insideEndTop", "insideStartTop", "insideEndBottom", "insideStartBottom"] as const)[i % 4],
            },
          })),
        },
      },
    ],
  });
}

export interface IprCurve {
  label: string;
  points: [number, number][];
  color: string;
  dashed?: boolean;
  width?: number;
}

/** IPR: tests (liquid vs BHP, older = paler), today's operating point, and the candidate curves. */
export function cardIpr(detail: HeaderWellDetail | undefined, curves: IprCurve[], now: [number, number] | null): EChartsOption | null {
  const tests = (detail?.tests ?? []).filter((t) => typeof t.liquid === "number" && typeof t.bhp === "number" && (t.bhp as number) > 50);
  if (!tests.length && !curves.some((c) => c.points.length)) return null;
  const n = tests.length;
  return houseOption({
    tooltip: {
      trigger: "item",
      formatter: (raw: unknown) => {
        const p = raw as { seriesName: string; value: [number, number, string?] };
        return ttHeader(p.value[2] ?? p.seriesName) + ttRow(SLATE, "Liquid", `${fmtNum(p.value[0], 0)} BLPD`) +
          ttRow(SLATE, "BHP", `${fmtNum(p.value[1], 0)} psi`);
      },
    },
    legend: { top: 0, itemWidth: 12, textStyle: { fontSize: 10 } },
    grid: { ...CARD_GRID, right: 12 },
    xAxis: { type: "value", ...axis(""), min: 0, axisLabel: { fontSize: 9 } },
    yAxis: { type: "value", ...axis(""), min: 0, axisLabel: { fontSize: 9 } },
    series: [
      {
        name: "Tests", type: "scatter", symbolSize: 6,
        data: tests.map((t, i) => ({
          value: [t.liquid, t.bhp, t.date],
          itemStyle: { color: SLATE, opacity: 0.25 + 0.75 * ((i + 1) / n) },
        })),
      },
      ...(now ? [{ name: "Now", type: "scatter" as const, symbol: "diamond", symbolSize: 11, itemStyle: { color: CRIMSON }, data: [[now[0], now[1], "Now (latest test rate, gauge BHP)"]] }] : []),
      ...curves.filter((c) => c.points.length).map((c) => ({
        name: c.label, type: "line" as const, showSymbol: false, data: c.points,
        lineStyle: { color: c.color, width: c.width ?? 1.5, type: (c.dashed ? "dashed" : "solid") as "dashed" | "solid" },
        itemStyle: { color: c.color },
      })),
    ],
  });
}
