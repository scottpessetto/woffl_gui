/**
 * Shared pump-curve chart primitives.
 *
 * The pad booster charts (S / I / M, PadCharts) and the E-Pad booster
 * candidate screen (EPadBoosterPanel) draw the same industry curve sheet, so
 * the annotation carrier, the tooltip row builders and the machine
 * head / BHP / efficiency panel live here once. Extracted verbatim from
 * PadCharts when E-Pad joined - if a number moves in one place it moves in
 * both, which is the point.
 */

import type { PumpMachineCurve } from "../../api/types";
import type { EChartsOption } from "../../charts/echarts";
import {
  ACCENT,
  axis,
  baseGrid,
  baseTooltip,
  CRIMSON,
  GOLD,
  houseOption,
  SLATE,
  ttHeader,
  ttRow,
} from "../../charts/theme";
import { fmtNum } from "../../lib/format";

// Petroleum phase convention, matching the pump-history strip.
export const OIL_GREEN = "#2E7D32";

// Operating-region shading, both drawn exactly as the plant reports them and
// left to composite where they overlap. POR is NOT always inside AOR: on
// I-Pad the vendor range starts above the preferred band, so POR hangs out to
// the left of it. That mismatch is information; never clip it away.
export const AOR_FILL = "rgba(37,99,235,0.06)";
export const POR_FILL = "rgba(37,99,235,0.12)";

export interface Duty {
  headerPsi: number | null;
  totalBpd: number | null;
  perPumpBpd: number | null;
}

/** [lo, hi] when the pair is a usable span, else null. */
export function span(v: number[] | null): [number, number] | null {
  if (v === null || v.length < 2) return null;
  const lo = v[0];
  const hi = v[1];
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi <= lo) return null;
  return [lo, hi];
}

/** [flow, column] pairs out of a machine points table, dropping bad rows. */
export function xy(points: number[][], col: number): [number, number][] {
  const out: [number, number][] = [];
  for (const p of points) {
    const q = p[0];
    const v = p[col];
    if (Number.isFinite(q) && Number.isFinite(v)) out.push([q, v]);
  }
  return out;
}

/* ------------------------------------------------------------ annotations */

export function refLabel(text: string, color: string): Record<string, unknown> {
  return { show: true, formatter: text, position: "insideEndTop", color, fontSize: 11 };
}

export interface Marks {
  aor: number[] | null;
  por: number[] | null;
  bep: number | null;
  minFlow: number | null;
  /** horizontal discharge limit (psi) */
  cap: number | null;
  /** vertical duty flow */
  duty: number | null;
}

/**
 * Bands and reference lines parked on an unnamed, silent carrier series.
 * markArea/markLine relayout with the axes on every dataZoom; a
 * custom-series renderItem does not, and drifts off the axes after a zoom.
 */
export function carrierSeries(m: Marks): Record<string, unknown> {
  const areas: Record<string, unknown>[][] = [];
  const aor = span(m.aor);
  const por = span(m.por);
  if (aor !== null) {
    areas.push([{ xAxis: aor[0], itemStyle: { color: AOR_FILL } }, { xAxis: aor[1] }]);
  }
  if (por !== null) {
    areas.push([{ xAxis: por[0], itemStyle: { color: POR_FILL } }, { xAxis: por[1] }]);
  }

  const lines: Record<string, unknown>[] = [];
  if (m.bep !== null) {
    lines.push({
      xAxis: m.bep,
      lineStyle: { color: SLATE, width: 1, type: "dashed" },
      label: refLabel("BEP", SLATE),
    });
  }
  if (m.minFlow !== null) {
    lines.push({
      xAxis: m.minFlow,
      lineStyle: { color: CRIMSON, width: 1, type: "dashed" },
      label: refLabel("min flow", CRIMSON),
    });
  }
  if (m.cap !== null) {
    lines.push({
      yAxis: m.cap,
      lineStyle: { color: GOLD, width: 1, type: "dashed" },
      label: refLabel("discharge cap", GOLD),
    });
  }
  if (m.duty !== null) {
    lines.push({
      xAxis: m.duty,
      lineStyle: { color: CRIMSON, width: 1.5 },
      label: refLabel("duty", CRIMSON),
    });
  }

  return {
    name: "",
    type: "line",
    data: [],
    silent: true,
    ...(areas.length > 0 ? { markArea: { silent: true, animation: false, data: areas } } : {}),
    ...(lines.length > 0
      ? { markLine: { silent: true, symbol: "none", animation: false, data: lines } }
      : {}),
  };
}

/* ---------------------------------------------------------------- tooltips */

/** The axis-trigger tooltip fields these charts read. */
export interface TipParam {
  seriesName?: string;
  color?: unknown;
  axisValue?: unknown;
  value?: unknown;
}

export function tipParams(raw: unknown): TipParam[] {
  return (Array.isArray(raw) ? raw : [raw]) as TipParam[];
}

function tipColor(p: TipParam): string {
  const c = typeof p.color === "string" ? p.color : SLATE;
  // Hollow markers fill white, and a white dot on a white tooltip is nothing.
  return c.toLowerCase() === "#ffffff" ? SLATE : c;
}

/** The y reading of an axis-trigger datum, whether it is [x, y] or a bare y. */
function tipValue(p: TipParam): number | null {
  const v = Array.isArray(p.value) ? p.value[1] : p.value;
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

export function tipAxisNum(list: TipParam[]): number | null {
  for (const p of list) {
    if (typeof p.axisValue === "number" && Number.isFinite(p.axisValue)) return p.axisValue;
  }
  return null;
}

export function tipAxisText(list: TipParam[]): string {
  for (const p of list) {
    if (typeof p.axisValue === "string") return p.axisValue;
  }
  return "";
}

export interface UnitSpec {
  unit: string;
  dp: number;
}

/** One row per series, unit and precision keyed by series name. */
export function tipRows(list: TipParam[], units: Record<string, UnitSpec>): string[] {
  const out: string[] = [];
  for (const p of list) {
    const name = typeof p.seriesName === "string" ? p.seriesName : "";
    const spec = units[name];
    const v = tipValue(p);
    if (spec === undefined || v === null) continue;
    out.push(ttRow(tipColor(p), name, `${fmtNum(v, spec.dp)} ${spec.unit}`));
  }
  return out;
}

/* --------------------------------------------------- machine curve panel */

export const MACHINE_HELP =
  "Pump performance: head, BHP and efficiency vs flow per pump. Shading is the " +
  "allowable and preferred operating regions as the vendor states them; the solid " +
  "crimson line is the optimized duty flow.";

export function machineOption(pump: PumpMachineCurve, duty: Duty): EChartsOption {
  const derateName = pump.derate_note ?? "Field derated head";
  const units: Record<string, UnitSpec> = {
    Head: { unit: "ft", dp: 0 },
    BHP: { unit: "BHP", dp: 0 },
    Efficiency: { unit: "%", dp: 1 },
  };
  // Derated head sits next to head in the legend, where it gets compared.
  const legend =
    pump.head_derated !== null
      ? ["Head", derateName, "BHP", "Efficiency"]
      : ["Head", "BHP", "Efficiency"];

  const series: Record<string, unknown>[] = [
    {
      name: "Head",
      type: "line",
      yAxisIndex: 0,
      showSymbol: false,
      data: xy(pump.points, 1),
      lineStyle: { color: ACCENT, width: 2.2 },
      itemStyle: { color: ACCENT },
      z: 5,
    },
    {
      name: "BHP",
      type: "line",
      yAxisIndex: 1,
      showSymbol: false,
      data: xy(pump.points, 2),
      lineStyle: { color: GOLD, width: 1.8 },
      itemStyle: { color: GOLD },
      z: 4,
    },
    {
      name: "Efficiency",
      type: "line",
      yAxisIndex: 2,
      showSymbol: false,
      data: xy(pump.points, 3),
      lineStyle: { color: OIL_GREEN, width: 1.8 },
      itemStyle: { color: OIL_GREEN },
      z: 4,
    },
  ];

  if (pump.head_derated !== null) {
    units[derateName] = { unit: "ft", dp: 0 };
    series.push({
      name: derateName,
      type: "line",
      yAxisIndex: 0,
      showSymbol: false,
      data: xy(pump.head_derated, 1),
      lineStyle: { color: ACCENT, width: 1.6, type: "dashed" },
      itemStyle: { color: ACCENT },
      z: 5,
    });
  }

  series.push(
    carrierSeries({
      aor: pump.aor,
      por: pump.por,
      bep: pump.bep,
      minFlow: pump.min_flow,
      cap: null,
      duty: duty.perPumpBpd,
    }),
  );

  return houseOption({
    tooltip: {
      ...baseTooltip,
      trigger: "axis",
      formatter: (raw: unknown): string => {
        const list = tipParams(raw);
        const q = tipAxisNum(list);
        const out = q !== null ? [ttHeader(`${fmtNum(q)} BPD per pump`)] : [];
        return [...out, ...tipRows(list, units)].join("");
      },
    },
    legend: { top: 4, right: 8, itemWidth: 18, textStyle: { fontSize: 12 }, data: legend },
    // Wide right margin: the efficiency axis is offset off the BHP axis.
    grid: { ...baseGrid, top: 56, right: 96 },
    xAxis: { type: "value", ...axis("Flow per pump (BPD)", { min: 0 }) },
    yAxis: [
      { type: "value", ...axis("Head (ft)") },
      {
        type: "value",
        position: "right",
        ...axis("BHP"),
        nameGap: 40,
        splitLine: { show: false },
      },
      {
        type: "value",
        position: "right",
        offset: 52,
        ...axis("Efficiency (%)", { min: 0, max: 100 }),
        nameGap: 40,
        splitLine: { show: false },
      },
    ],
    series,
  });
}
