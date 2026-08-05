/**
 * Well Profile - wellbore trajectory visualization. Plots horizontal
 * departure vs TVD and MD vs TVD from the deviation survey (or the field
 * preset trajectory when no survey exists), plus inclination when available.
 * Mirrors woffl/gui/tabs/well_profile.py.
 */

import { useMemo } from "react";

import { useWellProfile } from "../api/hooks";
import type { EChartsOption } from "../charts/echarts";
import { ACCENT, axis, baseGrid, baseTooltip, GOLD, houseOption, SLATE } from "../charts/theme";
import { useEChart } from "../charts/useEChart";
import { Badge, Card, ErrorNote, InfoNote, Spinner } from "../components/ui";
import { fmtNum } from "../lib/format";
import { useParamsStore } from "../state/params";

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

export default function WellProfilePage() {
  const well = useParamsStore((s) => s.well);
  const params = useParamsStore((s) => s.params);

  const query = useWellProfile(well, params.jpump_tvd, params.field_model);
  const data = query.data;

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
    return houseOption({
      title: { text: "Wellbore Profile", left: 8, textStyle: { fontSize: 13, fontWeight: 600 } },
      tooltip: { ...baseTooltip, trigger: "axis" },
      grid: { ...baseGrid },
      xAxis: { type: "value", ...axis("Horizontal Departure (ft)") },
      yAxis: { type: "value", ...axis("True Vertical Depth (ft)"), inverse: true },
      series,
    });
  }, [data]);

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
    return houseOption({
      title: { text: "MD vs TVD", left: 8, textStyle: { fontSize: 13, fontWeight: 600 } },
      tooltip: { ...baseTooltip, trigger: "axis" },
      legend: { bottom: 0, itemWidth: 16, textStyle: { fontSize: 11 } },
      grid: { ...baseGrid, bottom: 68 },
      xAxis: { type: "value", ...axis("Measured Depth (ft)") },
      yAxis: { type: "value", ...axis("True Vertical Depth (ft)"), inverse: true },
      series,
    });
  }, [data]);

  const inclinationOption = useMemo<EChartsOption | null>(() => {
    const inc = data?.inclination;
    if (!inc) return null;
    const pts = inc.md.map((m, i) => [m, inc.deg[i]]);
    return houseOption({
      title: { text: "Inclination", left: 8, textStyle: { fontSize: 13, fontWeight: 600 } },
      tooltip: { ...baseTooltip, trigger: "axis" },
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

  const profileRef = useEChart(profileOption);
  const mdTvdRef = useEChart(mdTvdOption);
  const inclinationRef = useEChart(inclinationOption);

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
      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <div ref={profileRef} className="h-[480px]" />
        </Card>
        <Card>
          <div ref={mdTvdRef} className="h-[480px]" />
        </Card>
      </div>
      {inclinationOption && (
        <Card>
          <div ref={inclinationRef} className="h-[320px]" />
        </Card>
      )}
    </div>
  );
}
