/**
 * Suction response diagnostic (advanced) - a collapsed-by-default panel at
 * the bottom of the Solver workbench for engineers judging whether their
 * fit reproduces the well's REAL response: 12 months of daily field
 * (PF pressure, BHP) pairs as a scatter, with the current sidebar model's
 * psu(Ppf) response curve overlaid on demand via the EXISTING /pf-range
 * sweep restricted to the current nozzle/throat. If the green curve does
 * not move like the dots, the fit will not predict pressure changes.
 *
 * Renders nothing while loading, when the well has no usable days, or when
 * the server predates GET /wells/{well}/response-history (404s/errors).
 */

import { useMemo, useState } from "react";

import { stableStringify } from "../../api/client";
import { usePfRange, useResponseHistory } from "../../api/hooks";
import type { ResponseHistoryDay, SimParams } from "../../api/types";
import { ChartPanel } from "../../charts/ChartPanel";
import type { EChartsOption } from "../../charts/echarts";
import {
  ACCENT,
  axis,
  baseGrid,
  baseTooltip,
  GOLD,
  houseOption,
  SLATE,
  ttHeader,
  ttRow,
} from "../../charts/theme";
import { Button, Card } from "../../components/ui";
import { fmtNum } from "../../lib/format";
import { effectiveParams, useParamsStore } from "../../state/params";

// Petroleum phase convention, matching the pump-history strip.
const OIL_GREEN = "#2E7D32";

/** The pf-range snapshot restricted to the CURRENT nozzle/throat: one pump
 * at the sidebar's inputs, swept over the sidebar's PF pressure window - so
 * the overlay is exactly the model the rest of the workbench is solving. */
function overlaySnapshot(params: SimParams): SimParams {
  return {
    ...effectiveParams(params),
    nozzle_batch_options: [params.nozzle_no],
    throat_batch_options: [params.area_ratio],
  };
}

/** [ppf, bhp, date, era, buildup] tuples - extras ride along for the tooltip. */
type Pt = [number, number, string, string, boolean];

function toPoints(days: ResponseHistoryDay[]): Pt[] {
  return days
    .filter((d) => Number.isFinite(d?.ppf) && Number.isFinite(d?.bhp))
    .map((d) => [d.ppf, d.bhp, d.date ?? "", d.era ?? "current", d.buildup === true] as Pt);
}

function chartOption(
  pts: Pt[],
  showBuildup: boolean,
  floor: number | null,
  overlay: [number, number][] | null,
  stale: boolean,
): EChartsOption {
  const current = pts.filter((p) => !p[4] && p[3] === "current");
  const prior = pts.filter((p) => !p[4] && p[3] !== "current");
  const buildup = showBuildup ? pts.filter((p) => p[4]) : [];

  const tip = (raw: unknown): string => {
    // ECharts hands the formatter an untyped param object; the scatter data
    // rows are our own Pt tuples, so narrow at runtime before trusting them.
    if (raw == null || typeof raw !== "object" || !("value" in raw)) return "";
    const v = raw.value;
    if (!Array.isArray(v) || v.length < 5) return "";
    const [ppf, bhp, date, era, isBuildup] = v as Pt; // shape proven above
    const tag = isBuildup ? " (buildup)" : era !== "current" ? " (prior pump)" : "";
    return (
      ttHeader(`${date}${tag}`) +
      ttRow(ACCENT, "PF pressure (psi)", fmtNum(ppf)) +
      ttRow(SLATE, "BHP (psi)", fmtNum(bhp))
    );
  };

  const series: Record<string, unknown>[] = [
    {
      name: "prior pump",
      type: "scatter",
      symbolSize: 6,
      itemStyle: { color: SLATE, opacity: 0.35 },
      data: prior,
    },
    {
      name: "current pump",
      type: "scatter",
      symbolSize: 7,
      itemStyle: { color: ACCENT },
      data: current,
      ...(floor != null
        ? {
            markLine: {
              silent: true,
              symbol: "none",
              lineStyle: { type: "dashed", color: SLATE, width: 1.2 },
              label: {
                formatter: "measured floor",
                position: "insideEndTop",
                color: SLATE,
                fontSize: 10,
              },
              data: [{ yAxis: floor }],
            },
          }
        : {}),
    },
  ];
  if (buildup.length > 0)
    series.push({
      name: "buildup",
      type: "scatter",
      symbolSize: 6,
      itemStyle: { color: GOLD, opacity: 0.8 },
      data: buildup,
    });
  if (overlay != null && overlay.length > 0)
    series.push({
      name: "model (current inputs)",
      type: "line",
      showSymbol: false,
      z: 5,
      lineStyle: {
        color: stale ? SLATE : OIL_GREEN,
        width: 2,
        opacity: stale ? 0.45 : 1,
      },
      tooltip: { show: false },
      data: overlay,
    });

  return houseOption({
    tooltip: { ...baseTooltip, trigger: "item", formatter: tip },
    legend: { top: 2, right: 8, itemWidth: 14, textStyle: { color: SLATE, fontSize: 11 } },
    grid: { ...baseGrid, top: 28 },
    // scale: pressures live far from zero - a forced zero origin would crush
    // the response into the top-right corner.
    xAxis: { ...axis("PF pressure (psi)"), scale: true },
    yAxis: { ...axis("BHP (psi)"), nameGap: 42, scale: true },
    series,
  } as EChartsOption);
}

/** Advanced suction-response panel; `well` keys both fetches. */
export function ResponseDiagnostic({ well }: { well: string }) {
  const histQ = useResponseHistory(well);
  const params = useParamsStore((s) => s.params);
  const [showBuildup, setShowBuildup] = useState(false);
  // Explicit-submit overlay, PfRangePage-style: snapshot the params on click
  // and let the shared /pf-range query cache dedupe repeat runs.
  const [snapshot, setSnapshot] = useState<SimParams | null>(null);
  const pfQ = usePfRange(well, snapshot);

  // Overlay staleness: the run's snapshot no longer matches what the same
  // click would send today (any effective-param or pump-selection change).
  const stale =
    snapshot !== null && stableStringify(overlaySnapshot(params)) !== stableStringify(snapshot);

  const overlay = useMemo(() => {
    const rows = pfQ.data?.rows;
    if (!rows) return null;
    return rows
      .filter((r) => Number.isFinite(r?.power_fluid_pressure) && Number.isFinite(r?.psu_solv))
      .map((r) => [r.power_fluid_pressure, r.psu_solv] as [number, number])
      .sort((a, b) => a[0] - b[0]);
  }, [pfQ.data]);

  const data = histQ.data;
  const pts = useMemo(() => toPoints(data?.days ?? []), [data]);
  const floor = data?.evidence?.floor ?? null;
  const option = useMemo(
    () => (pts.length > 0 ? chartOption(pts, showBuildup, floor, overlay, stale) : null),
    [pts, showBuildup, floor, overlay, stale],
  );

  const provenance = useMemo(() => {
    const parts: string[] = [];
    const ev = data?.evidence ?? null;
    if (ev != null && ev.beta != null)
      parts.push(`beta ${ev.beta.toFixed(3)} (${ev.beta_source ?? "?"}, ${ev.n_pairs ?? 0} pairs)`);
    if (ev != null && ev.floor != null) parts.push(`floor ${fmtNum(ev.floor)} psi`);
    if (data?.era_start != null)
      parts.push(`era ${data.era_start.slice(0, 10)}${data.pump != null ? ` (${data.pump})` : ""}`);
    return parts.join(" | ");
  }, [data]);

  const buildupCount = useMemo(() => pts.filter((p) => p[4]).length, [pts]);

  // Old servers 404 the endpoint; a well without usable days has nothing to
  // judge. Either way the panel simply does not exist.
  if (histQ.isError || data == null || pts.length === 0 || option == null) return null;

  return (
    <Card padded={false}>
      <details>
        <summary className="cursor-pointer select-none px-2 py-2 text-xs font-semibold text-slate-600 hover:text-slate-800">
          Suction response diagnostic (advanced)
        </summary>
        <div className="space-y-2 px-3 pb-3">
          <p className="text-[11px] text-slate-500">
            Daily field (PF pressure, BHP) vs the CURRENT sidebar model response - if the green
            curve does not move like the dots, the fit will not predict pressure changes.
          </p>
          <div className="flex flex-wrap items-center gap-3">
            <Button
              size="sm"
              busy={pfQ.isFetching}
              onClick={() => setSnapshot(overlaySnapshot(params))}
            >
              Overlay model response
            </Button>
            <label className="flex cursor-pointer items-center gap-1.5 text-xs text-slate-600">
              <input
                type="checkbox"
                checked={showBuildup}
                onChange={(e) => setShowBuildup(e.target.checked)}
              />
              show buildup days{buildupCount > 0 ? ` (${buildupCount})` : ""}
            </label>
            {overlay != null && stale && !pfQ.isFetching && (
              <span className="text-xs text-amber-700">stale - inputs changed</span>
            )}
            {pfQ.isError && <span className="text-xs text-red-700">model overlay failed</span>}
          </div>
          <ChartPanel option={option} height={340} zoom={{ xAxisIndex: "all", yAxisIndex: "all" }} />
          {provenance !== "" && <p className="text-[11px] text-slate-500">{provenance}</p>}
        </div>
      </details>
    </Card>
  );
}
