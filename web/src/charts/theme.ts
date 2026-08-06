/**
 * House chart style: light, dense, engineer-grade. Shared option fragments
 * so every chart on the site reads as one family.
 */

import type { EChartsOption } from "./echarts";

export const ACCENT = "#2563eb"; // blue-600
export const CRIMSON = "#c9252d";
export const GOLD = "#d4a017";
export const SLATE = "#64748b";
export const GRID_LINE = "#e2e8f0";
export const AXIS_LINE = "#94a3b8";
export const TEXT = "#334155";

/** Viridis stops for the days-since-test color scale (matplotlib order). */
export const VIRIDIS = [
  "#440154",
  "#482878",
  "#3e4989",
  "#31688e",
  "#26828e",
  "#1f9e89",
  "#35b779",
  "#6ece58",
  "#b5de2b",
  "#fde725",
];

/** 20-color categorical palette for multi-pump sweeps (tab20-like). */
export const CATEGORY20 = [
  "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
  "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
  "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
  "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5",
];

export const baseTextStyle = {
  fontFamily:
    "ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
  color: TEXT,
} as const;

export const baseTooltip = {
  backgroundColor: "#ffffff",
  borderColor: GRID_LINE,
  borderWidth: 1,
  padding: [8, 10],
  textStyle: { color: TEXT, fontSize: 12 },
  extraCssText: "box-shadow: 0 4px 12px rgba(15, 23, 42, 0.12); border-radius: 6px;",
} as const;

export const baseGrid = {
  left: 56,
  right: 24,
  top: 40,
  bottom: 44,
  containLabel: false,
} as const;

export function axis(name: string, opts?: { min?: number | "dataMin"; max?: number | "dataMax"; inverse?: boolean }): Record<string, unknown> {
  return {
    name,
    nameLocation: "middle",
    nameGap: 32,
    nameTextStyle: { color: SLATE, fontSize: 12, fontWeight: 500 },
    axisLine: { lineStyle: { color: AXIS_LINE } },
    axisTick: { lineStyle: { color: AXIS_LINE } },
    axisLabel: { color: SLATE, fontSize: 11 },
    splitLine: { lineStyle: { color: GRID_LINE } },
    ...(opts?.min !== undefined ? { min: opts.min } : {}),
    ...(opts?.max !== undefined ? { max: opts.max } : {}),
    ...(opts?.inverse ? { inverse: true } : {}),
  };
}

/** Merge the house base into a chart option. */
export function houseOption(option: EChartsOption): EChartsOption {
  return {
    textStyle: baseTextStyle,
    animationDuration: 200,
    ...option,
  };
}

/* ---------------------------------------------------------------- tooltip
 * Shared building blocks so every chart's tooltip reads the same way:
 * bold date/x header, one row per series with the series' marker color,
 * label left, formatted value right. Custom series (era bands, marker
 * overlays) never dump raw data into tooltips - they either render through
 * these helpers or are excluded by the chart's formatter.
 */

/** Nearest point by x in an x-sorted [x, y, ...] array; null if empty. */
export function nearestByX<T extends readonly [number, ...unknown[]]>(
  pts: readonly T[],
  x: number,
): T | null {
  if (pts.length === 0) return null;
  let lo = 0;
  let hi = pts.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (pts[mid][0] < x) lo = mid;
    else hi = mid;
  }
  return x - pts[lo][0] <= pts[hi][0] - x ? pts[lo] : pts[hi];
}

/** Bold header line (date or x value) for a tooltip. */
export function ttHeader(text: string): string {
  return `<div style="font-weight:600;margin-bottom:4px">${text}</div>`;
}

/** One tooltip row: colored dot, label, right-aligned tabular value. */
export function ttRow(color: string, label: string, value: string): string {
  return (
    `<div style="display:flex;align-items:center;gap:6px;line-height:1.7">` +
    `<span style="width:8px;height:8px;border-radius:9999px;background:${color};flex:none"></span>` +
    `<span>${label}</span>` +
    `<span style="margin-left:auto;padding-left:16px;font-variant-numeric:tabular-nums;font-weight:500">${value}</span>` +
    `</div>`
  );
}

/** Minimal shape of an axis-trigger tooltip param this app relies on. */
interface AxisTooltipParam {
  seriesName?: string;
  color?: unknown;
  axisValue?: unknown;
  value?: unknown;
}

export interface AxisTooltipSpec {
  /** unit appended to the header (the axis-pointer reading), e.g. "ft MD" */
  headerUnit?: string;
  headerDp?: number;
  /** unit appended to each series value, e.g. "psi" */
  unit: string;
  dp?: number;
  /** datum dimension holding the reading: 1 = y (default), 0 = x */
  valueDim?: 0 | 1;
}

/**
 * House axis-trigger tooltip for single-quantity charts: formatted header
 * with unit, one ttRow per series. Charts mixing units (rates + pressures)
 * write their own formatter from ttHeader/ttRow instead.
 */
export function axisTooltip(spec: AxisTooltipSpec): (raw: unknown) => string {
  const { headerUnit = "", headerDp = 0, unit, dp = 0, valueDim = 1 } = spec;
  const nf = new Intl.NumberFormat("en-US", {
    minimumFractionDigits: 0,
    maximumFractionDigits: dp,
  });
  const hf = new Intl.NumberFormat("en-US", {
    minimumFractionDigits: 0,
    maximumFractionDigits: headerDp,
  });
  return (raw: unknown): string => {
    const list = (Array.isArray(raw) ? raw : [raw]) as AxisTooltipParam[];
    if (list.length === 0) return "";
    const head = list.find((p) => typeof p.axisValue === "number");
    const out: string[] = [];
    if (head) {
      out.push(ttHeader(`${hf.format(head.axisValue as number)}${headerUnit ? ` ${headerUnit}` : ""}`));
    }
    for (const p of list) {
      const v = Array.isArray(p.value) ? p.value[valueDim] : p.value;
      if (typeof v !== "number" || !Number.isFinite(v)) continue;
      const color = typeof p.color === "string" ? p.color : SLATE;
      out.push(ttRow(color, p.seriesName ?? "", `${nf.format(v)} ${unit}`));
    }
    return out.join("");
  };
}
