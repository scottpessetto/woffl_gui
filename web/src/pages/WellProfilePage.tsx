/**
 * Well Profile - wellbore trajectory visualization. Plots horizontal
 * departure vs TVD and MD vs TVD from the deviation survey (or the field
 * preset trajectory when no survey exists), plus inclination when available.
 * Carries the MD <-> TVD depth interpolator, whose hit is marked on both
 * depth charts.
 */

import { useCallback, useMemo, useState } from "react";

import { useWellProfile } from "../api/hooks";
import type { DepthLookupResponse } from "../api/types";
import type { EChartsOption } from "../charts/echarts";
import { ACCENT, axis, axisTooltip, baseGrid, baseTooltip, CRIMSON, GOLD, houseOption, SLATE } from "../charts/theme";
import { ChartPanel } from "../charts/ChartPanel";
import { Badge, Card, ErrorNote, InfoNote, Spinner } from "../components/ui";
import { fmtNum } from "../lib/format";
import { useParamsStore } from "../state/params";
import { DepthInterpolator } from "./well-profile/DepthInterpolator";

/** Index of the value in `values` closest to `target`. */
function nearestIndex(values: number[], target: number): number {
  let best = 0;
  let bestDist = Infinity;
  for (let i = 0; i < values.length; i++) {
    const d = Math.abs(values[i] - target);
    if (d < bestDist) {
      bestDist = d;
      best = i;
    }
  }
  return best;
}

/** Linear read of `ys` at `x` on the monotone `xs` - marker placement only. */
function interpAt(xs: number[], ys: number[], x: number): number {
  if (xs.length === 0) return 0;
  let hi = 1;
  while (hi < xs.length - 1 && xs[hi] < x) hi++;
  const lo = hi - 1;
  const span = xs[hi] - xs[lo];
  if (span <= 0) return ys[lo];
  return ys[lo] + ((x - xs[lo]) / span) * (ys[hi] - ys[lo]);
}

/** Crosshair + dot marking one looked-up depth on a depth chart. */
function markerSeries(name: string, x: number, y: number): Record<string, unknown> {
  return {
    name,
    type: "scatter",
    data: [[x, y]],
    symbol: "circle",
    symbolSize: 11,
    itemStyle: { color: CRIMSON },
    z: 10,
    markLine: {
      silent: true,
      symbol: "none",
      data: [
        { xAxis: x, lineStyle: { color: CRIMSON, type: "dotted", width: 1 }, label: { show: false } },
        { yAxis: y, lineStyle: { color: CRIMSON, type: "dotted", width: 1 }, label: { show: false } },
      ],
    },
  };
}

export default function WellProfilePage() {
  const well = useParamsStore((s) => s.well);
  const params = useParamsStore((s) => s.params);

  const query = useWellProfile(well, params.jpump_tvd, params.field_model);
  const data = query.data;
  const [hit, setHit] = useState<DepthLookupResponse | null>(null);
  // Stable identity: DepthInterpolator pushes through an effect keyed on it.
  const onResult = useCallback((next: DepthLookupResponse | null) => setHit(next), []);

  const profileOption = useMemo<EChartsOption | null>(() => {
    if (!data) return null;
    const pts = data.hd.map((h, i) => [h, data.vd[i]]);
    const series: Record<string, unknown>[] = [
      {
        name: "Trajectory",
        type: "line",
        data: pts,
        showSymbol: false,
        lineStyle: { color: SLATE, width: 1.5 },
        itemStyle: { color: SLATE },
      },
    ];
    if (data.jetpump_md !== null && data.md.length > 0) {
      const i = nearestIndex(data.md, data.jetpump_md);
      series.push({
        name: "Jet Pump",
        type: "scatter",
        data: [[data.hd[i], data.vd[i]]],
        symbol: "diamond",
        symbolSize: 14,
        itemStyle: { color: GOLD },
      });
    }
    if (hit && data.md.length > 0) {
      series.push(
        markerSeries("Lookup", interpAt(data.md, data.hd, hit.md), hit.tvd),
      );
    }
    return houseOption({
      title: { text: "Wellbore Profile", left: 8, textStyle: { fontSize: 13, fontWeight: 600 } },
      tooltip: { ...baseTooltip, trigger: "axis", formatter: axisTooltip({ headerUnit: "ft out", unit: "ft TVD" }) },
      grid: { ...baseGrid },
      xAxis: { type: "value", ...axis("Horizontal Departure (ft)") },
      yAxis: { type: "value", ...axis("True Vertical Depth (ft)"), inverse: true },
      series,
    });
  }, [data, hit]);

  const mdTvdOption = useMemo<EChartsOption | null>(() => {
    if (!data) return null;
    const rawPts = data.md.map((m, i) => [m, data.vd[i]]);
    const filteredPts = data.md_filtered.map((m, i) => [m, data.vd_filtered[i]]);
    const series: Record<string, unknown>[] = [
      {
        name: "Survey",
        type: "line",
        data: rawPts,
        showSymbol: false,
        lineStyle: { color: ACCENT, width: 2 },
        itemStyle: { color: ACCENT },
        markLine:
          data.jetpump_md !== null && data.jetpump_vd !== null
            ? {
                silent: true,
                symbol: "none",
                data: [
                  {
                    xAxis: data.jetpump_md,
                    lineStyle: { color: GOLD, type: "dashed", width: 1 },
                    label: { show: false },
                  },
                  {
                    yAxis: data.jetpump_vd,
                    lineStyle: { color: GOLD, type: "dashed", width: 1 },
                    label: { show: false },
                  },
                ],
              }
            : undefined,
      },
      {
        name: "Filtered",
        type: "line",
        data: filteredPts,
        showSymbol: false,
        lineStyle: { color: SLATE, width: 1.5, type: "dashed" },
        itemStyle: { color: SLATE },
      },
    ];
    if (data.jetpump_md !== null && data.jetpump_vd !== null) {
      series.push({
        name: "Jet Pump",
        type: "scatter",
        data: [[data.jetpump_md, data.jetpump_vd]],
        symbol: "diamond",
        symbolSize: 14,
        itemStyle: { color: GOLD },
      });
    }
    if (hit) {
      series.push(markerSeries("Lookup", hit.md, hit.tvd));
    }
    return houseOption({
      title: { text: "MD vs TVD", left: 8, textStyle: { fontSize: 13, fontWeight: 600 } },
      tooltip: { ...baseTooltip, trigger: "axis", formatter: axisTooltip({ headerUnit: "ft MD", unit: "ft TVD" }) },
      legend: { bottom: 0, itemWidth: 16, textStyle: { fontSize: 11 } },
      grid: { ...baseGrid, bottom: 68 },
      xAxis: { type: "value", ...axis("Measured Depth (ft)") },
      yAxis: { type: "value", ...axis("True Vertical Depth (ft)"), inverse: true },
      series,
    });
  }, [data, hit]);

  const inclinationOption = useMemo<EChartsOption | null>(() => {
    const inc = data?.inclination;
    if (!inc) return null;
    const pts = inc.md.map((m, i) => [m, inc.deg[i]]);
    return houseOption({
      title: { text: "Inclination", left: 8, textStyle: { fontSize: 13, fontWeight: 600 } },
      tooltip: { ...baseTooltip, trigger: "axis", formatter: axisTooltip({ headerUnit: "ft MD", unit: "deg", dp: 1 }) },
      grid: { ...baseGrid },
      xAxis: { type: "value", ...axis("Measured Depth (ft)") },
      yAxis: { type: "value", ...axis("Inclination (deg)") },
      series: [
        {
          name: "Inclination",
          type: "line",
          data: pts,
          showSymbol: false,
          lineStyle: { color: ACCENT, width: 1.5 },
          itemStyle: { color: ACCENT },
        },
      ],
    });
  }, [data]);


  if (query.isError) {
    return <ErrorNote error={query.error} />;
  }
  if (!data) {
    return <Spinner label="Loading well profile" />;
  }

  return (
    <div className="space-y-4">
      {query.isFetching && <Spinner label="Updating" />}
      <div className="flex flex-wrap items-center gap-2">
        {data.has_survey ? (
          <Badge tone="info">{fmtNum(data.md.length)} survey points</Badge>
        ) : (
          <Badge tone="fair">No survey - {params.field_model} default</Badge>
        )}
        {data.jetpump_md !== null && <Badge>JP MD: {fmtNum(data.jetpump_md)} ft</Badge>}
        {data.jetpump_vd !== null && <Badge>JP TVD: {fmtNum(data.jetpump_vd)} ft</Badge>}
      </div>
      {!data.has_survey && (
        <InfoNote>
          No deviation survey on record for {well} - showing the {params.field_model} preset
          trajectory scaled to the jet pump TVD.
        </InfoNote>
      )}
      <DepthInterpolator
        well={well}
        fieldModel={params.field_model}
        mdRange={[data.md[0] ?? 0, data.md[data.md.length - 1] ?? 0]}
        onResult={onResult}
      />
      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <ChartPanel option={profileOption} height={480} zoom={{ xAxisIndex: [0], yAxisIndex: [0] }} />
        </Card>
        <Card>
          <ChartPanel option={mdTvdOption} height={480} zoom={{ xAxisIndex: [0], yAxisIndex: [0] }} />
        </Card>
      </div>
      {inclinationOption && (
        <Card>
          <ChartPanel option={inclinationOption} height={320} zoom={{ xAxisIndex: [0], yAxisIndex: [0] }} />
        </Card>
      )}
    </div>
  );
}
