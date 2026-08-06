/**
 * Pressure Profile view - production vs power-fluid pressure traverse and
 * their differential, both plotted against depth (surface on top). Auto-runs
 * off the debounced sidebar params exactly like the Solver view.
 */

import { useMemo } from "react";

import { usePressureProfile } from "../api/hooks";
import type { EChartsOption } from "../charts/echarts";
import { ACCENT, axis, axisTooltip, baseGrid, baseTooltip, CRIMSON, GOLD, houseOption, SLATE } from "../charts/theme";
import { ChartPanel } from "../charts/ChartPanel";
import { Card, ErrorNote, InfoNote, Metric, Spinner } from "../components/ui";
import { fmtNum } from "../lib/format";
import { useDebounced } from "../lib/useDebounced";
import { effectiveParams, useParamsStore } from "../state/params";

/** Depth markLine at the jetpump, shared by both charts. */
function jpMarkLine(jpumpMd: number, labeled: boolean): Record<string, unknown> {
  return {
    silent: true,
    symbol: "none",
    data: [
      {
        yAxis: jpumpMd,
        lineStyle: { color: GOLD, type: "dashed", width: 1.5 },
        label: labeled
          ? {
              formatter: `JP @ ${fmtNum(jpumpMd)} ft MD`,
              position: "insideStartTop",
              color: GOLD,
              fontSize: 11,
            }
          : { show: false },
      },
    ],
  };
}

export default function PressureProfilePage() {
  const well = useParamsStore((s) => s.well);
  const params = useParamsStore((s) => s.params);
  const simActive = useParamsStore((s) => s.simActive);

  const effective = useMemo(() => effectiveParams(params), [params]);
  const debounced = useDebounced(effective);
  const query = usePressureProfile(well, debounced, simActive);
  const data = query.data;

  const profileOption = useMemo<EChartsOption | null>(() => {
    if (!data) return null;
    const prodPts = data.prod.md.map((md, i) => [data.prod.press[i], md]);
    const pfPts = data.pf.md.map((md, i) => [data.pf.press[i], md]);
    return houseOption({
      title: { text: "Pressure vs Depth", left: 8, textStyle: { fontSize: 13, fontWeight: 600 } },
      // Depth is the independent variable: the pointer tracks the y axis,
      // rows show each string's pressure (datum dim 0) at that depth.
      tooltip: { ...baseTooltip, trigger: "axis", axisPointer: { axis: "y" }, formatter: axisTooltip({ headerUnit: "ft MD", unit: "psi", valueDim: 0 }) },
      legend: { bottom: 0, itemWidth: 16, textStyle: { fontSize: 11 } },
      grid: { ...baseGrid, bottom: 68 },
      xAxis: { type: "value", ...axis("Pressure (psi)") },
      yAxis: { type: "value", ...axis("Depth (ft MD)"), inverse: true },
      series: [
        {
          name: "Production",
          type: "line",
          data: prodPts,
          showSymbol: false,
          lineStyle: { color: ACCENT, width: 2 },
          itemStyle: { color: ACCENT },
          markLine: jpMarkLine(data.jpump_md, true),
        },
        {
          name: "Power Fluid",
          type: "line",
          data: pfPts,
          showSymbol: false,
          lineStyle: { color: SLATE, width: 2 },
          itemStyle: { color: SLATE },
        },
        {
          name: "Suction",
          type: "scatter",
          data: [[data.metrics.psu, data.jpump_md]],
          symbol: "diamond",
          symbolSize: 13,
          itemStyle: { color: CRIMSON },
        },
      ],
    });
  }, [data]);

  const diffOption = useMemo<EChartsOption | null>(() => {
    if (!data) return null;
    const diffPts = data.diff.md.map((md, i) => [data.diff.dp[i], md]);
    return houseOption({
      title: {
        text: "Differential (PF - Production)",
        left: 8,
        textStyle: { fontSize: 13, fontWeight: 600 },
      },
      tooltip: { ...baseTooltip, trigger: "axis", axisPointer: { axis: "y" }, formatter: axisTooltip({ headerUnit: "ft MD", unit: "psi", valueDim: 0 }) },
      grid: { ...baseGrid, bottom: 68 },
      xAxis: { type: "value", ...axis("Differential (psi)") },
      yAxis: { type: "value", ...axis("Depth (ft MD)"), inverse: true },
      series: [
        {
          name: "PF - Production",
          type: "line",
          data: diffPts,
          showSymbol: false,
          lineStyle: { color: GOLD, width: 2 },
          itemStyle: { color: GOLD },
          markLine: {
            silent: true,
            symbol: "none",
            data: [
              { xAxis: 0, lineStyle: { color: SLATE, type: "dashed" }, label: { show: false } },
              ...(jpMarkLine(data.jpump_md, false).data as Record<string, unknown>[]),
            ],
          },
        },
      ],
    });
  }, [data]);


  if (!simActive) {
    return (
      <InfoNote>
        Select a well in the sidebar, or press Run with Custom inputs, to compute the
        production and power-fluid pressure traverses.
      </InfoNote>
    );
  }
  if (query.isError) {
    return <ErrorNote error={query.error} />;
  }
  if (!data) {
    return <Spinner label="Computing pressure profile" />;
  }

  const m = data.metrics;
  return (
    <div className="space-y-4">
      {query.isFetching && <Spinner label="Updating" />}
      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <ChartPanel option={profileOption} height={520} zoom={{ xAxisIndex: [0], yAxisIndex: [0] }} />
        </Card>
        <Card>
          <ChartPanel option={diffOption} height={520} zoom={{ xAxisIndex: [0], yAxisIndex: [0] }} />
        </Card>
      </div>
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <Metric label="Suction P" value={fmtNum(m.psu)} sub="psi" />
        <Metric label="Production @ JP" value={fmtNum(m.prod_at_jp)} sub="psi" />
        <Metric label="Power Fluid @ JP" value={fmtNum(m.pf_at_jp)} sub="psi" />
        <Metric label="Differential @ JP" value={fmtNum(m.dp_at_jp)} sub="psi" />
      </div>
      <p className="text-xs text-slate-500">
        Production pressure at JP depth is the discharge the pump must produce to lift mixed
        fluid to surface; suction is where formation fluid enters below the pump. Power fluid
        is single-phase water; production uses Beggs and Brill.
      </p>
    </div>
  );
}
