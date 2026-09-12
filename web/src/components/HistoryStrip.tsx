/**
 * Production & JP change history - THE shared stacked figure for the JP
 * History view and the Solver's pump-history strip ("one builder so the two
 * can never drift").
 *
 * Top grid: Oil (green) stacked under Form Water (blue) as areas, BHP
 * (orange) and optional PF pressure (purple dotted) on the right axis,
 * JPCO dashed red change lines with rotated labels that pass straight
 * through the strip. Bottom grid: pumps-in-hole timeline, one colored band
 * per era (tenure = Date Set -> next Date Set; the tracker's Date Pulled
 * produces phantom gaps), same pump code = same color.
 */

import { useMemo } from "react";

import type { JpHistoryResponse, PumpMatchResult } from "../api/types";
import { historyTimestamp, matchAt, matchLines } from "./historyMatchSeries";
import type { EChartsOption } from "../charts/echarts";
import { axis, baseTooltip, houseOption, nearestByX, SLATE, ttHeader, ttNote, ttRow } from "../charts/theme";
import { ChartPanel } from "../charts/ChartPanel";
import { fmtDate, fmtNum, pumpCode } from "../lib/format";

// Colors lifted from the retired Streamlit app's plotly original.
const OIL_LINE = "#2E7D32";
const OIL_FILL = "rgba(46,125,50,0.4)";
const WAT_LINE = "#1565C0";
const WAT_FILL = "rgba(21,101,192,0.3)";
const BHP_COLOR = "#E65100";
const PF_COLOR = "#6A1B9A";
const JPCO_LINE = "rgba(211,47,47,0.7)";
const JPCO_TEXT = "#D32F2F";

/** plotly px.colors.qualitative.Set2 - the strip's pump palette. */
const SET2 = [
  "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3",
  "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3",
];

const DAY_MS = 86_400_000;

function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function ms(v: unknown): number | null {
  if (typeof v !== "string" || v.length === 0) return null;
  const t = historyTimestamp(v);
  return Number.isNaN(t) ? null : t;
}

interface Era {
  code: string;
  start: number;
  end: number;
  color: string;
}

interface JpChange {
  x: number;
  label: string;
}

/**
 * Eras + change markers from the install rows. Exported for reuse in tests
 * or captions; pure data shaping.
 */
export function buildTimeline(data: JpHistoryResponse): { eras: Era[]; changes: JpChange[] } {
  const installs = data.installs
    .map((row) => ({ set: ms(row.date_set), code: pumpCode(row.nozzle, row.throat) }))
    .filter((r): r is { set: number; code: string } => r.set !== null)
    .sort((a, b) => a.set - b.set);
  if (installs.length === 0) return { eras: [], changes: [] };

  const today = Date.now();
  const colorOf = new Map<string, string>();
  for (const ins of installs) {
    if (!colorOf.has(ins.code)) colorOf.set(ins.code, SET2[colorOf.size % SET2.length]);
  }

  const eras: Era[] = installs.map((ins, i) => ({
    code: ins.code,
    start: ins.set,
    // Tenure runs Date Set -> next Date Set (JPCO same-day rule).
    end: i + 1 < installs.length ? installs[i + 1].set : today,
    color: colorOf.get(ins.code) ?? SET2[0],
  }));

  const changes: JpChange[] = installs.map((ins, i) => {
    let label: string;
    if (i === 0) {
      label = `Set ${ins.code}`;
    } else if (installs[i - 1].code === ins.code) {
      label = `JPCO ${ins.code} (same)`;
    } else {
      label = `JPCO ${installs[i - 1].code} to ${ins.code}`;
    }
    return { x: ins.set, label };
  });

  return { eras, changes };
}

export function HistoryStrip({
  data,
  bhpFromZero = true,
  showPf = false,
  height = 520,
  match = null,
  selectedEra = null,
  oilDetail = false,
  pfDetail = false,
  onSelectTest,
}: {
  data: JpHistoryResponse;
  bhpFromZero?: boolean;
  showPf?: boolean;
  height?: number;
  match?: PumpMatchResult | null;
  selectedEra?: string | null;
  oilDetail?: boolean;
  pfDetail?: boolean;
  onSelectTest?: (id: string) => void;
}) {
  const panels = match ? [...(oilDetail ? ["oil" as const] : []), ...(pfDetail ? ["pf" as const] : [])] : [];
  const chartHeight = height + panels.length * 175;
  const option = useMemo<EChartsOption | null>(() => {
    const { eras, changes } = buildTimeline(data);
    if (eras.length === 0) return null;
    const detailPanels = match ? [...(oilDetail ? ["oil" as const] : []), ...(pfDetail ? ["pf" as const] : [])] : [];
    const bandX = 1 + detailPanels.length;
    const bandY = 2 + detailPanels.length;
    const matchRows = match?.rows.filter((r) => !selectedEra || r.installation_id === selectedEra) ?? [];

    // Oil and Form Water come from the SAME test rows in the same order -
    // index alignment is what makes the ECharts stack correct, so the rows
    // are sorted ONCE by date and every per-series array is built from that
    // order. Missing values coerce to 0 so the stacked fill never breaks.
    const tests = [...data.tests]
      .map((t) => ({ x: ms(t.date), t }))
      .filter((r): r is { x: number; t: (typeof data.tests)[number] } => r.x !== null)
      .sort((a, b) => a.x - b.x);
    const oilPts: [number, number][] = [];
    const fwatPts: [number, number][] = [];
    const pfPts: [number, number][] = [];
    const bhpTestPts: [number, number][] = [];
    for (const { x, t } of tests) {
      oilPts.push([x, num(t.oil_rate) ?? 0]);
      fwatPts.push([x, num(t.fwat_rate) ?? 0]);
      const pf = num(t.pf_press);
      if (pf !== null) pfPts.push([x, pf]);
      const bhp = num(t.bhp);
      if (bhp !== null) bhpTestPts.push([x, bhp]);
    }
    const bhpDailyPts: [number, number][] = [];
    for (const d of data.bhp_daily) {
      const x = ms(d.date);
      if (x !== null && Number.isFinite(d.bhp)) bhpDailyPts.push([x, d.bhp]);
    }
    bhpDailyPts.sort((a, b) => a[0] - b[0]);
    // Prefer the daily BHP series; fall back to test-date BHP (original).
    const bhpPts = bhpDailyPts.length > 0 ? bhpDailyPts : bhpTestPts;

    /**
     * Unified tooltip, plotly x-unified style: ONE date header, then the
     * nearest point of EVERY series - the default axis tooltip drops series
     * whose x grid differs from the snapped value (tests are ~weekly, BHP
     * daily) and dumps custom-series data raw (the era band's epoch-ms
     * datum rendered as a 13-digit "JP" number). Era membership becomes a
     * readable "Pump in hole" row instead.
     */
    const tooltipFormatter = (raw: unknown): string => {
      const list = (Array.isArray(raw) ? raw : [raw]) as { axisValue?: unknown }[];
      const at = list.find((p) => typeof p.axisValue === "number");
      if (!at) return "";
      const ms0 = at.axisValue as number;
      const rows: string[] = [ttHeader(fmtDate(new Date(ms0)))];
      const oil = nearestByX(oilPts, ms0);
      if (oil) rows.push(ttRow(OIL_LINE, "Oil", `${fmtNum(oil[1])} BOPD`));
      const fwat = nearestByX(fwatPts, ms0);
      if (fwat) rows.push(ttRow(WAT_LINE, "Form Water", `${fmtNum(fwat[1])} BWPD`));
      const bhp = nearestByX(bhpPts, ms0);
      if (bhp) rows.push(ttRow(BHP_COLOR, "BHP", `${fmtNum(bhp[1])} psi`));
      if (showPf) {
        const pf = nearestByX(pfPts, ms0);
        if (pf) rows.push(ttRow(PF_COLOR, "PF pressure", `${fmtNum(pf[1])} psi`));
      }
      const modeled = matchAt(matchRows, ms0);
      if (modeled && ["replay", "fit", "prediction"].includes(modeled.status)) {
        const phase = modeled.phase === "replay" ? "model" : modeled.phase === "fit" ? "fit" : "prediction";
        rows.push(ttRow(BHP_COLOR, `BHP ${phase}`, `${fmtNum(modeled.predicted_bhp)} psi`));
        rows.push(ttRow(OIL_LINE, `Oil ${phase}`, `${fmtNum(modeled.predicted_oil)} BOPD`));
        if (pfDetail) rows.push(ttRow(PF_COLOR, `PF rate ${phase}`, `${fmtNum(modeled.predicted_pf)} BPD`));
      } else if (modeled) {
        rows.push(ttRow(SLATE, "Model", modeled.status === "failed" ? "Solve failed" : "No prediction"));
        rows.push(ttNote(modeled.message ?? "This test has no model prediction."));
      } else if (match) {
        rows.push(ttNote("No modeled test on this date. Predictions are calculated at recorded well tests."));
      }
      if (modeled?.input_wc != null && modeled.input_gor != null) {
        rows.push(ttRow(SLATE, "Model WC / GOR", `${fmtNum(100 * modeled.input_wc, 1)}% / ${fmtNum(modeled.input_gor)} scf/STB`));
      }
      const era = eras.find((e) => ms0 >= e.start && ms0 <= e.end);
      rows.push(
        ttRow(
          era?.color ?? "#e2e8f0",
          "Pump in hole",
          era ? `${era.code} (set ${fmtDate(new Date(era.start))})` : "-",
        ),
      );
      return rows.join("");
    };

    // Shared x-range: earliest install - 15d .. today + 15d (original).
    const minMs = (match && match.rows.length ? Math.min(...match.rows.map((r) => Date.parse(r.date))) : eras[0].start) - 15 * DAY_MS;
    const maxMs = (match ? Date.parse(match.as_of) : Date.now()) + 15 * DAY_MS;
    const span = maxMs - minMs;

    // --- bottom strip: one markArea rect per era. markArea (unlike a
    // custom-series renderItem) relayouts natively on every dataZoom, so
    // the bands can never drift from the date axis while zooming.
    const bandAreas = eras.map((e) => [
      {
        xAxis: e.start,
        itemStyle: { color: e.color, borderColor: "#ffffff", borderWidth: 1 },
        // Only label segments wide enough to carry text (original: >3% of span).
        label:
          (e.end - e.start) / span > 0.03
            ? {
                show: true,
                position: "inside",
                formatter: e.code,
                color: "#1a1a1a",
                fontSize: 12,
              }
            : { show: false },
      },
      { xAxis: e.end },
    ]);

    // --- JPCO change lines: dashed red verticals with rotated labels on
    // the main grid, mirrored (unlabeled) through the strip below. Two
    // markLine sets because marks clip to their own grid.
    const changeLineStyle = { color: JPCO_LINE, width: 1.5, type: [5, 4] };
    const changeLines = (labeled: boolean) =>
      changes.map((c) => ({
        xAxis: c.x,
        lineStyle: changeLineStyle,
        label: labeled
          ? {
              show: true,
              formatter: c.label,
              position: "insideEndTop",
              rotate: 90,
              align: "right",
              verticalAlign: "middle",
              color: JPCO_TEXT,
              fontSize: 10,
              distance: 6,
            }
          : { show: false },
      }));

    const series: Record<string, unknown>[] = [
      {
        name: "Oil (BOPD)",
        type: "line",
        xAxisIndex: 0,
        yAxisIndex: 0,
        stack: "production",
        data: oilPts,
        showSymbol: false,
        lineStyle: { color: OIL_LINE, width: 1.5 },
        itemStyle: { color: OIL_LINE },
        areaStyle: { color: OIL_FILL },
      },
      {
        name: "Form Water (BWPD)",
        type: "line",
        xAxisIndex: 0,
        yAxisIndex: 0,
        stack: "production",
        data: fwatPts,
        showSymbol: false,
        lineStyle: { color: WAT_LINE, width: 1.5 },
        itemStyle: { color: WAT_LINE },
        areaStyle: { color: WAT_FILL },
      },
      {
        name: "BHP (psi)",
        type: "line",
        xAxisIndex: 0,
        yAxisIndex: 1,
        data: bhpPts,
        ...(bhpPts === bhpDailyPts
          ? { showSymbol: false, lineStyle: { color: BHP_COLOR, width: 1.5 } }
          : { symbolSize: 4, lineStyle: { color: BHP_COLOR, width: 2 } }),
        itemStyle: { color: BHP_COLOR },
      },
      // Invisible carriers for the marks: unnamed so legend toggles can
      // never hide them, one per grid because marks clip to their grid.
      {
        type: "line",
        xAxisIndex: 0,
        yAxisIndex: 0,
        data: [],
        silent: true,
        markLine: {
          silent: true,
          symbol: "none",
          animation: false,
          data: changeLines(true),
          z: 20,
        },
      },
      {
        type: "line",
        xAxisIndex: bandX,
        yAxisIndex: bandY,
        data: [],
        silent: true,
        markArea: {
          silent: true,
          animation: false,
          data: bandAreas,
        },
        markLine: {
          silent: true,
          symbol: "none",
          animation: false,
          data: changeLines(false),
          z: 20,
        },
      },
    ];
    if (showPf) {
      series.push({
        name: "PF pressure (psi)",
        type: "line",
        xAxisIndex: 0,
        yAxisIndex: 1,
        data: pfPts,
        symbolSize: 3,
        lineStyle: { color: PF_COLOR, width: 1, type: "dotted" },
        itemStyle: { color: PF_COLOR },
      });
    }

    if (match) {
      series.push(...matchLines(match, "bhp", BHP_COLOR, 0, 1, selectedEra),
        ...matchLines(match, "oil", OIL_LINE, 0, 0, selectedEra));
      // Actual replay test markers remain selectable when a prediction fails.
      for (const [key, color, yAxisIndex] of [["oil", OIL_LINE, 0], ["bhp", BHP_COLOR, 1]] as const) {
        series.push({ type: "line", xAxisIndex: 0, yAxisIndex, lineStyle: { opacity: 0 },
          symbol: "emptyCircle", symbolSize: 6, itemStyle: { color }, z: 10,
          data: matchRows.map((r) => ({ value: [Date.parse(r.date), r[key]], name: r.wt_uid })) });
      }
      const selected = match.eras.find((e) => e.installation_id === selectedEra);
      if (selected?.training_start && selected.training_end) {
        series.push({ type: "line", data: [], silent: true, xAxisIndex: 0, yAxisIndex: 0,
          markArea: { silent: true, itemStyle: { color: "rgba(100,116,139,0.10)" },
            label: { position: "insideTopLeft", fontSize: 10, color: SLATE },
            data: [[{ name: "Inflow fitting window", xAxis: Date.parse(selected.training_start) },
              { xAxis: Date.parse(selected.training_end) }]] } });
      }
      detailPanels.forEach((quantity, i) => {
        const xAxisIndex = i + 1, yAxisIndex = i + 2;
        const color = quantity === "oil" ? OIL_LINE : PF_COLOR;
        const actual = quantity === "oil" ? oilPts : tests.map(({ x, t }) => [x, num(t.lift_wat)]);
        series.push({ name: quantity === "oil" ? "Oil (BOPD)" : "Actual PF rate", type: "line",
          xAxisIndex, yAxisIndex, data: actual, showSymbol: true, symbolSize: 4,
          lineStyle: { color, width: 1.4 }, itemStyle: { color }, connectNulls: false });
        series.push(...matchLines(match, quantity, color, xAxisIndex, yAxisIndex, selectedEra));
        series.push({ type: "line", data: [], silent: true, xAxisIndex, yAxisIndex,
          markLine: { silent: true, symbol: "none", data: changeLines(false) } });
      });
    }

    return houseOption({
      ...(match ? { animation: false } : {}),
      tooltip: {
        ...baseTooltip,
        trigger: "axis",
        axisPointer: { type: "line" },
        formatter: tooltipFormatter,
      },
      axisPointer: { link: [{ xAxisIndex: "all" }] },
      legend: {
        top: 0,
        right: 8,
        itemWidth: 18,
        textStyle: { fontSize: 11 },
        data: [
          ...(showPf ? ["PF pressure (psi)"] : []),
          "BHP (psi)",
          "Form Water (BWPD)",
          "Oil (BOPD)",
          ...(match ? ["BHP model", "BHP fit", "BHP prediction", "Oil model", "Oil fit", "Oil prediction"].filter((name) => series.some((s) => s.name === name)) : []),
          ...(match && pfDetail ? ["Actual PF rate", "PF rate model", "PF rate fit", "PF rate prediction"].filter((name) => series.some((s) => s.name === name)) : []),
        ],
        type: "scroll",
      },
      grid: [
        { left: 64, right: 64, top: 28, height: height - 130 },
        ...detailPanels.map((_q, i) => ({ left: 64, right: 64, top: height - 55 + i * 175, height: 115 })),
        { left: 64, right: 64, top: chartHeight - 75, bottom: 18 },
      ],
      xAxis: [
        {
          type: "time",
          gridIndex: 0,
          min: minMs,
          max: maxMs,
          axisLine: { lineStyle: { color: "#94a3b8" } },
          axisLabel: { color: SLATE, fontSize: 11 },
        },
        ...detailPanels.map((_q, i) => ({ type: "time" as const, gridIndex: i + 1,
          min: minMs, max: maxMs, axisLabel: { color: SLATE, fontSize: 11 } })),
        {
          type: "time",
          gridIndex: bandX,
          min: minMs,
          max: maxMs,
          axisLabel: { show: false },
          axisTick: { show: false },
          axisLine: { show: false },
        },
      ],
      yAxis: [
        { type: "value", gridIndex: 0, ...axis("Rate (BPD)"), nameGap: 44 },
        {
          type: "value",
          gridIndex: 0,
          position: "right",
          ...axis("BHP (psi)"),
          nameGap: 44,
          min: bhpFromZero ? 0 : "dataMin",
          splitLine: { show: false },
        },
        ...detailPanels.map((q, i) => ({ type: "value" as const, gridIndex: i + 1,
          ...axis(q === "oil" ? "Oil (BOPD)" : "PF rate (BPD)"), nameGap: 44 })),
        { type: "value", gridIndex: bandX, min: 0, max: 1, show: false },
      ],
      series,
    });
  }, [data, bhpFromZero, showPf, match, selectedEra, oilDetail, pfDetail, height, chartHeight]);

  if (option === null) return null;
  // Brush zooms the main grid (x0 + both rate/BHP axes). Listing BOTH x
  // axes keeps the pump-era strip natively window-synced with the main
  // grid: ECharts links dataZoom components that share an axis, and both
  // time axes span the identical min/max range.
  return (
    <ChartPanel
      option={option}
      height={chartHeight}
      zoom={{ xAxisIndex: Array.from({ length: 2 + panels.length }, (_, i) => i),
        yAxisIndex: Array.from({ length: 2 + panels.length }, (_, i) => i) }}
      onSelect={onSelectTest}
    />
  );
}
