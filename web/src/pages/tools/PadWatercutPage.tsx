/**
 * Pad Water Cut - daily pad-level WC for G/H/I/J.
 *
 * Port of the retired Streamlit Pad Water Cut tool. Same method, same series,
 * same default three-year window; the aggregation is untouched server-side.
 *
 * Two charts, as the tab had: water cut on a fixed 0-100% axis (a WC chart
 * that rescales to its own data hides how close a pad is to the limit), and
 * the oil/water rates behind it.
 */

import { useMemo, useState } from "react";

import { usePadWatercut, usePadWatercutWindow } from "../../api/hooks";
import type { PadWatercutResponse } from "../../api/types";
import { ChartPanel } from "../../charts/ChartPanel";
import type { EChartsOption } from "../../charts/echarts";
import { axis, axisTooltip, houseOption } from "../../charts/theme";
import { Button, Card, ErrorNote, InfoNote, Section, Spinner } from "../../components/ui";
import { downloadCsv } from "../../lib/csv";

// The tab's colours, kept so a screenshot of either app reads the same.
const PAD_COLOR: Record<string, string> = {
  G: "#1f77b4",
  H: "#2ca02c",
  I: "#ff7f0e",
  J: "#d62728",
  All: "#555555",
};
const PADS = ["G", "H", "I", "J", "All"] as const;

const DATE_CLS =
  "h-8 rounded-md border border-slate-300 bg-white px-2 text-sm text-slate-800 " +
  "outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-200";

function wcOption(data: PadWatercutResponse, shown: Set<string>): EChartsOption {
  const series = data.series
    .filter((s) => shown.has(s.pad))
    .map((s) => ({
      name: s.pad,
      type: "line" as const,
      showSymbol: false,
      // "All" is the combined stream - dashed and heavier so it reads as the
      // aggregate rather than a fifth pad.
      lineStyle: {
        color: PAD_COLOR[s.pad] ?? "#888",
        width: s.pad === "All" ? 3 : 1.8,
        type: s.pad === "All" ? ("dashed" as const) : ("solid" as const),
      },
      itemStyle: { color: PAD_COLOR[s.pad] ?? "#888" },
      data: s.points
        .filter((p) => p.date && p.wc != null)
        .map((p) => [p.date as string, (p.wc as number) * 100] as [string, number]),
    }));

  return houseOption({
    tooltip: axisTooltip({ unit: "%", dp: 0 }),
    legend: { top: 0 },
    grid: { top: 34, left: 56, right: 20, bottom: 40 },
    xAxis: { type: "time", ...axis("Date") },
    // Fixed 0-100: the point is distance to the water limit, not autoscale.
    yAxis: { type: "value", min: 0, max: 100, ...axis("Water Cut (%)") },
    series,
  });
}

function rateOption(data: PadWatercutResponse, shown: Set<string>): EChartsOption {
  const series = data.series
    .filter((s) => shown.has(s.pad))
    .flatMap((s) => {
      const color = PAD_COLOR[s.pad] ?? "#888";
      return [
        {
          name: `${s.pad} oil`,
          type: "line" as const,
          showSymbol: false,
          lineStyle: { color, width: 2 },
          itemStyle: { color },
          data: s.points
            .filter((p) => p.date && p.oil != null)
            .map((p) => [p.date as string, p.oil as number] as [string, number]),
        },
        {
          name: `${s.pad} water`,
          type: "line" as const,
          showSymbol: false,
          lineStyle: { color, width: 2, type: "dotted" as const },
          itemStyle: { color },
          data: s.points
            .filter((p) => p.date && p.water != null)
            .map((p) => [p.date as string, p.water as number] as [string, number]),
        },
      ];
    });

  return houseOption({
    tooltip: axisTooltip({ unit: " BPD", dp: 0 }),
    legend: { top: 0, type: "scroll" },
    grid: { top: 34, left: 64, right: 20, bottom: 40 },
    xAxis: { type: "time", ...axis("Date") },
    yAxis: { type: "value", ...axis("Rate (BPD)") },
    series,
  });
}

export default function PadWatercutPage() {
  const windowQ = usePadWatercutWindow(true);
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");
  const [shown, setShown] = useState<Set<string>>(new Set(PADS));
  const [showRates, setShowRates] = useState(false);

  // Seed the range from the server default once it arrives; edits stick.
  const effStart = start || windowQ.data?.start || "";
  const effEnd = end || windowQ.data?.end || "";

  const badRange = Boolean(effStart && effEnd && effStart >= effEnd);
  const query = usePadWatercut(effStart, effEnd, !badRange);

  const wcOpt = useMemo(
    () => (query.data ? wcOption(query.data, shown) : null),
    [query.data, shown],
  );
  const rateOpt = useMemo(
    () => (query.data && showRates ? rateOption(query.data, shown) : null),
    [query.data, shown, showRates],
  );

  function toggle(pad: string) {
    setShown((prev) => {
      const next = new Set(prev);
      if (next.has(pad)) next.delete(pad);
      else next.add(pad);
      return next;
    });
  }

  function exportCsv() {
    if (!query.data) return;
    const rows = query.data.series.flatMap((s) =>
      s.points.map((p) => ({
        pad: s.pad,
        date: p.date ?? "",
        wc: p.wc ?? "",
        oil: p.oil ?? "",
        water: p.water ?? "",
      })),
    );
    downloadCsv(
      `pad_watercut_${effStart}_${effEnd}.csv`,
      [
        { key: "pad", label: "pad" },
        { key: "date", label: "date" },
        { key: "wc", label: "wc" },
        { key: "oil", label: "oil" },
        { key: "water", label: "water" },
      ],
      rows,
    );
  }

  return (
    <div className="space-y-4">
      <Section
        title="Pad Water Cut"
        actions={
          <Button size="sm" variant="ghost" onClick={exportCsv} disabled={!query.data}>
            Download CSV
          </Button>
        }
      >
        <p className="mb-3 text-sm text-slate-600">
          Daily pad-level WC for pads G, H, I and J. Each well&apos;s last allocated test is
          forward-filled; well-days with more than 6 h of shut-in are excluded. H and I are
          treated as on-pad PF recycle (lift water stays); G and J ship lift water back to
          the plant.
        </p>

        <div className="flex flex-wrap items-end gap-3">
          <label className="block">
            <span className="text-xs text-slate-500">Start</span>
            <input
              type="date"
              value={effStart}
              onChange={(e) => setStart(e.target.value)}
              className={`${DATE_CLS} mt-1 block`}
            />
          </label>
          <label className="block">
            <span className="text-xs text-slate-500">End</span>
            <input
              type="date"
              value={effEnd}
              onChange={(e) => setEnd(e.target.value)}
              className={`${DATE_CLS} mt-1 block`}
            />
          </label>

          <div className="flex items-end gap-1">
            {PADS.map((p) => (
              <button
                key={p}
                type="button"
                onClick={() => toggle(p)}
                className={
                  "rounded-md border px-2.5 py-1 text-sm transition-colors " +
                  (shown.has(p)
                    ? "border-slate-300 bg-white text-slate-800"
                    : "border-slate-200 bg-slate-100 text-slate-400")
                }
                style={shown.has(p) ? { borderLeft: `3px solid ${PAD_COLOR[p]}` } : undefined}
              >
                {p}
              </button>
            ))}
          </div>
        </div>

        {badRange && (
          <InfoNote className="mt-3">Start date must be before end date.</InfoNote>
        )}
      </Section>

      {query.isLoading && <Spinner label="Building pad water-cut series" />}
      {query.isError && <ErrorNote error={query.error} />}

      {query.data && query.data.series.length === 0 && (
        <InfoNote>No data returned for the selected date range.</InfoNote>
      )}

      {wcOpt && (
        <Card>
          <ChartPanel option={wcOpt} height={480} zoom={{ xAxisIndex: [0], yAxisIndex: [0] }} />
        </Card>
      )}

      {query.data && query.data.series.length > 0 && (
        <Section
          title="Pad oil &amp; water rates"
          actions={
            <Button size="sm" variant="ghost" onClick={() => setShowRates((v) => !v)}>
              {showRates ? "Hide" : "Show"}
            </Button>
          }
        >
          {rateOpt && <ChartPanel option={rateOpt} height={420} zoom={{ xAxisIndex: [0], yAxisIndex: [0] }} />}
        </Section>
      )}
    </div>
  );
}
