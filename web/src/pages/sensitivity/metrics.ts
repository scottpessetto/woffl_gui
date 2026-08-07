/**
 * The four match quantities and the per-knob row math the Sensitivity panels
 * share: units for the axes and tooltips, defensive readers for the nullable
 * server fields, and the sorted tornado rows the chart and the table both
 * render from.
 */

import type { SensitivityKnob, SensitivityPoint, SensitivityResponse } from "../../api/types";
import { fmtNum } from "../../lib/format";

export type MetricId = "psu" | "qoil" | "qliq" | "qpf";

export interface MetricSpec {
  id: MetricId;
  /** segmented-control label */
  label: string;
  /** y axis name, unit included */
  axisName: string;
  /** bare unit for tooltips, reference lines and table headers */
  unit: string;
  dp: number;
}

export const METRICS: MetricSpec[] = [
  { id: "psu", label: "Suction BHP", axisName: "Suction BHP (psig)", unit: "psig", dp: 0 },
  { id: "qoil", label: "Oil", axisName: "Oil Rate (BOPD)", unit: "BOPD", dp: 0 },
  { id: "qliq", label: "Liquid", axisName: "Produced Liquid Rate (BLPD)", unit: "BLPD", dp: 0 },
  { id: "qpf", label: "Power fluid", axisName: "Power Fluid Rate (BWPD)", unit: "BWPD", dp: 0 },
];

/** One metric off one solved point; null covers a failed solve. */
export function pointMetric(p: SensitivityPoint, metric: MetricId): number | null {
  const v = p[metric];
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

/** Signed excursion on one side; low/high arrive as loose keyed records. */
export function excursion(side: Record<string, number | null>, metric: MetricId): number | null {
  const v = side[metric];
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

/** Response field holding the measured test value for each metric. */
const TARGET_KEY: Record<MetricId, "target_psu" | "target_qoil" | "target_qliq" | "target_qpf"> = {
  psu: "target_psu",
  qoil: "target_qoil",
  qliq: "target_qliq",
  qpf: "target_qpf",
};

/** The measured test value for a metric, when the well test carried one. */
export function targetFor(res: SensitivityResponse, metric: MetricId): number | null {
  const v = res[TARGET_KEY[metric]];
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

/** Value with an explicit sign, so a delta column never reads as a level. */
export function signed(v: number | null, dp = 0): string {
  if (v === null || !Number.isFinite(v)) return "-";
  return v > 0 ? `+${fmtNum(v, dp)}` : fmtNum(v, dp);
}

/**
 * One tornado row. `down` and `up` are the furthest excursion in each
 * direction rather than the low-case and high-case ends, so a knob that
 * pushes the metric the same way at both ends draws one honest bar instead
 * of two bars fighting over the same side of zero.
 */
export interface TornadoRow {
  id: string;
  label: string;
  basis: string;
  inert: boolean;
  /** most negative excursion; null when the knob never moved it down */
  down: number | null;
  /** most positive excursion; null when the knob never moved it up */
  up: number | null;
  /** sort key: total travel across the whole sweep */
  span: number;
}

export function tornadoRows(knobs: SensitivityKnob[], metric: MetricId): TornadoRow[] {
  const rows = knobs.map((k) => {
    const ends: number[] = [];
    for (const side of [k.low, k.high]) {
      const v = excursion(side, metric);
      if (v !== null) ends.push(v);
    }
    const negs = ends.filter((v) => v < 0);
    const poss = ends.filter((v) => v > 0);
    const down = negs.length > 0 ? Math.min(...negs) : null;
    const up = poss.length > 0 ? Math.max(...poss) : null;
    return {
      id: k.id,
      label: k.label,
      basis: k.basis,
      inert: k.inert,
      down,
      up,
      span: Math.abs(down ?? 0) + (up ?? 0),
    };
  });
  // Biggest mover first; the tornado axis is inverted so row 0 sits on top.
  return rows.sort((a, b) => b.span - a.span);
}
