/**
 * Permutation scatter - every solved combination plotted as its miss against
 * the measured test: BHP error across, oil error up, crosshair at the origin.
 * A run on the crosshair matches the test on both quantities.
 *
 * The colour is the third dimension. A cloud can sit right on the origin and
 * still be wrong, because the combination that lands BHP and oil often does
 * it by making far too much water; the visual map on absolute liquid miss
 * separates the runs that match everything from the runs that match the two
 * quantities on the axes and blow the third.
 *
 * Needs a measured BHP and a measured oil rate. Without both there is no
 * origin to plot against and the envelope is the whole answer.
 */

import { useMemo } from "react";

import type { CombineRun, SensitivityPoint } from "../../api/types";
import type { EChartsOption } from "../../charts/echarts";
import { ACCENT, axis, baseGrid, baseTooltip, CRIMSON, GOLD, houseOption, SLATE, ttHeader, ttRow } from "../../charts/theme";
import { ChartPanel } from "../../charts/ChartPanel";
import { Card } from "../../components/ui";
import { fmtNum } from "../../lib/format";
import { type CombineTargets, runReadings, targetOf } from "./combine";
import { signed } from "./metrics";

const HELP =
  "Each point is one permutation, placed by how far it misses the measured test on BHP " +
  "and on oil. The crosshair is a perfect match on both. Colour is how far the same run " +
  "misses on produced liquid.";

/** [bhp error psi, oil error BOPD, absolute liquid error BLPD, run index]. */
type Point = [number, number, number, number];

interface ScatterInput {
  runs: CombineRun[];
  baseline: SensitivityPoint;
  cloud: Point[];
  basePoint: Point | null;
  bestPoint: Point | null;
  knobIds: string[];
  knobLabels: Record<string, string>;
  targets: CombineTargets;
  /** upper end of the liquid-miss colour scale; null hides the visual map */
  liqScale: number | null;
}

function scatterOption(inp: ScatterInput): EChartsOption {
  const { runs, baseline, cloud, basePoint, bestPoint, knobIds, knobLabels, targets, liqScale } = inp;
  const tooltipFor = (raw: unknown): string => {
    // Item-trigger param: `value` is this point's own tuple, so the run it
    // came from is the fourth slot rather than anything ECharts hands back.
    const p = raw as { value?: unknown };
    const value: unknown[] = Array.isArray(p.value) ? p.value : [];
    const slot = value[3];
    const idx = typeof slot === "number" ? slot : -1;
    const source = idx >= 0 && idx < runs.length ? runs[idx] : null;
    const run: CombineRun | SensitivityPoint = source ?? baseline;
    const varied = source === null ? null : source.labels;

    const head =
      varied === null
        ? "Current case"
        : knobIds.map((id) => `${knobLabels[id] ?? id} ${varied[id] ?? "-"}`).join(", ");
    const out = [`<div style="max-width:280px">${ttHeader(head)}</div>`];

    for (const r of runReadings(run, targets)) {
      if (r.value === null) continue;
      const dot = r.spec.id === "psu" || r.spec.id === "qoil" ? ACCENT : SLATE;
      const miss = r.err === null ? "" : ` (${signed(r.err, r.spec.dp)})`;
      out.push(ttRow(dot, r.spec.label, `${fmtNum(r.value, r.spec.dp)} ${r.spec.unit}${miss}`));
    }
    if ("score" in run && run.score !== null) {
      out.push(ttRow(CRIMSON, "Score", fmtNum(run.score, 4)));
    }
    if (run.sonic === true) {
      out.push(`<div style="margin-top:4px;color:${SLATE};font-size:11px">Sonic at the throat</div>`);
    }
    return out.join("");
  };

  const series: Record<string, unknown>[] = [
    {
      name: "",
      type: "line",
      data: [],
      silent: true,
      markLine: {
        silent: true,
        symbol: "none",
        animation: false,
        label: { show: false },
        lineStyle: { color: GOLD, width: 1, type: "dashed" },
        data: [{ xAxis: 0 }, { yAxis: 0 }],
      },
    },
    {
      name: "Permutations",
      type: "scatter",
      symbolSize: 7,
      // A full factorial is up to 1200 points; animating them on every
      // option swap costs more than it communicates.
      animation: false,
      itemStyle: { color: ACCENT, opacity: 0.85 },
      data: cloud,
      tooltip: { formatter: tooltipFor },
    },
  ];
  if (basePoint !== null) {
    series.push({
      name: "Current case",
      type: "scatter",
      symbolSize: 13,
      z: 5,
      itemStyle: { color: SLATE, borderColor: "#ffffff", borderWidth: 1.5 },
      data: [basePoint],
      tooltip: { formatter: tooltipFor },
    });
  }
  if (bestPoint !== null) {
    series.push({
      name: "Best match",
      type: "scatter",
      symbol: "diamond",
      symbolSize: 18,
      z: 6,
      itemStyle: { color: CRIMSON, borderColor: "#ffffff", borderWidth: 1.5 },
      data: [bestPoint],
      tooltip: { formatter: tooltipFor },
    });
  }

  const legend = ["Permutations"];
  if (basePoint !== null) legend.push("Current case");
  if (bestPoint !== null) legend.push("Best match");

  return houseOption({
    tooltip: { ...baseTooltip, trigger: "item" },
    legend: { data: legend, top: 4, itemWidth: 14, textStyle: { fontSize: 11 } },
    grid: { ...baseGrid, left: 68, right: 112, top: 44, bottom: 52 },
    xAxis: { type: "value", ...axis("BHP Error, Model minus Test (psi)") },
    yAxis: { type: "value", ...axis("Oil Error, Model minus Test (BOPD)") },
    visualMap:
      liqScale !== null
        ? [
            {
              type: "continuous",
              dimension: 2,
              // Index 1 is the permutation cloud; the carrier, the current
              // case and the best run keep their own fixed colours.
              seriesIndex: 1,
              min: 0,
              max: liqScale,
              inRange: { color: [ACCENT, GOLD, CRIMSON] },
              right: 0,
              top: "middle",
              itemHeight: 110,
              calculable: false,
              text: ["Liquid Miss (BLPD)", ""],
              textStyle: { color: SLATE, fontSize: 10 },
            },
          ]
        : [],
    series,
  });
}

export function PermutationScatter({
  runs,
  baseline,
  bestIndex,
  targets,
  knobIds,
  knobLabels,
}: {
  runs: CombineRun[];
  baseline: SensitivityPoint;
  bestIndex: number | null;
  targets: CombineTargets;
  /** varied knob ids, in picker order - drives the tooltip header */
  knobIds: string[];
  knobLabels: Record<string, string>;
}) {
  const built = useMemo(() => {
    const tPsu = targetOf(targets, "psu");
    const tOil = targetOf(targets, "qoil");
    if (tPsu === null || tOil === null) return null;
    const tLiq = targetOf(targets, "qliq");

    const at = (r: CombineRun | SensitivityPoint, i: number): Point | null => {
      if (r.psu === null || r.qoil === null) return null;
      const liq = tLiq !== null && r.qliq !== null ? Math.abs(r.qliq - tLiq) : 0;
      return [r.psu - tPsu, r.qoil - tOil, liq, i];
    };

    const cloud: Point[] = [];
    runs.forEach((run, i) => {
      if (run.error !== null || i === bestIndex) return;
      const pt = at(run, i);
      if (pt !== null) cloud.push(pt);
    });

    const best = bestIndex !== null && bestIndex < runs.length ? at(runs[bestIndex], bestIndex) : null;
    const base = at(baseline, -1);
    const maxLiq = cloud.reduce((m, p) => Math.max(m, p[2]), 0);
    return { cloud, best, base, liqScale: tLiq !== null && maxLiq > 0 ? maxLiq : null };
  }, [runs, baseline, bestIndex, targets]);

  const option = useMemo(
    () =>
      built === null
        ? null
        : scatterOption({
            runs,
            baseline,
            cloud: built.cloud,
            basePoint: built.base,
            bestPoint: built.best,
            knobIds,
            knobLabels,
            targets,
            liqScale: built.liqScale,
          }),
    [built, runs, baseline, knobIds, knobLabels, targets],
  );

  if (built === null || option === null || built.cloud.length === 0) return null;

  return (
    <Card padded={false} className="p-2">
      <p className="px-2 pt-1 text-xs font-semibold text-slate-600" title={HELP}>
        Permutation Error Against the Test
      </p>
      <p className="px-2 text-[11px] text-slate-500">
        {built.cloud.length} solved permutations. Crosshair is a match on both quantities.
      </p>
      <ChartPanel option={option} height={420} zoom={{ xAxisIndex: [0], yAxisIndex: [0] }} />
    </Card>
  );
}
