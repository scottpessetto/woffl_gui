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
