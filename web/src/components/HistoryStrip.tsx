/**
 * Production & JP change history - THE shared stacked figure for the JP
 * History view and the Solver's pump-history strip, mirror of
 * woffl/gui/tabs/jp_history_tab.py:build_history_with_strip_figure
 * ("one builder so the two can never drift").
 *
 * Top grid: Oil (green) stacked under Form Water (blue) as areas, BHP
 * (orange) and optional PF pressure (purple dotted) on the right axis,
 * JPCO dashed red change lines with rotated labels that pass straight
 * through the strip. Bottom grid: pumps-in-hole timeline, one colored band
 * per era (tenure = Date Set -> next Date Set; the tracker's Date Pulled
 * produces phantom gaps), same pump code = same color.
 */

import { useMemo } from "react";

import type { JpHistoryResponse } from "../api/types";
import type { EChartsOption } from "../charts/echarts";
import { axis, baseTooltip, houseOption, SLATE } from "../charts/theme";
import { useEChart } from "../charts/useEChart";
import { fmtDate, fmtNum, pumpCode } from "../lib/format";

// Colors lifted from the plotly original (jp_history_tab.py).
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
  const t = new Date(v).getTime();
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
  /** alternate label offset to reduce overlap (original y_frac 0.95/0.85) */
  tier: number;
}

/** Cartesian rect handed to custom renderItem. */
interface CoordSys {
  x: number;
  y: number;
  width: number;
  height: number;
}

interface RenderParams {
  coordSys: CoordSys;
}

interface RenderApi {
  value: (dim: number) => number;
  coord: (point: [number, number]) => [number, number];
  getHeight: () => number;
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
    return { x: ins.set, label, tier: i % 2 };
  });

  return { eras, changes };
}

export function HistoryStrip({
  data,
  bhpFromZero = true,
  showPf = false,
  height = 520,
}: {
  data: JpHistoryResponse;
  bhpFromZero?: boolean;
  showPf?: boolean;
  height?: number;
}) {
  const option = useMemo<EChartsOption | null>(() => {
    const { eras, changes } = buildTimeline(data);
    if (eras.length === 0) return null;

    // Oil and Form Water come from the SAME test rows in the same order -
    // index alignment is what makes the ECharts stack correct. Missing
    // values coerce to 0 so the stacked fill never breaks.
    const oilPts: [number, number][] = [];
    const fwatPts: [number, number][] = [];
    const pfPts: [number, number][] = [];
    const bhpTestPts: [number, number][] = [];
    for (const t of data.tests) {
      const x = ms(t.date);
      if (x === null) continue;
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

    // Shared x-range: earliest install - 15d .. today + 15d (original).
    const minMs = eras[0].start - 15 * DAY_MS;
    const maxMs = Date.now() + 15 * DAY_MS;
    const span = maxMs - minMs;

    // --- bottom strip: one rect per era ------------------------------------
    const bandData: [number, number, number][] = eras.map((e, i) => [e.start, e.end, i]);
    const renderBand = (p: RenderParams, api: RenderApi): Record<string, unknown> => {
      const era = eras[api.value(2)];
      const cs = p.coordSys;
      const x0 = Math.max(api.coord([api.value(0), 0])[0], cs.x);
      const x1 = Math.min(api.coord([api.value(1), 0])[0], cs.x + cs.width);
      if (x1 <= x0) return { type: "group", children: [] };
      const children: Record<string, unknown>[] = [
        {
          type: "rect",
          shape: { x: x0, y: cs.y, width: x1 - x0, height: cs.height },
          style: { fill: era.color, stroke: "#ffffff", lineWidth: 1 },
        },
      ];
      // Only label segments wide enough to carry text (original: >3% of span).
      if ((era.end - era.start) / span > 0.03) {
        children.push({
          type: "text",
          style: {
            x: (x0 + x1) / 2,
            y: cs.y + cs.height / 2,
            text: era.code,
            align: "center",
            verticalAlign: "middle",
            fill: "#1a1a1a",
            fontSize: 12,
          },
        });
      }
      return { type: "group", children };
    };

    // --- JPCO change lines: dashed red verticals spanning BOTH grids, with
    // rotated labels at the top (original: paper-referenced shapes) ---------
    const changeData: [number, number, number][] = changes.map((c, i) => [c.x, c.tier, i]);
    const renderChange = (p: RenderParams, api: RenderApi): Record<string, unknown> => {
      const change = changes[api.value(2)];
      const cs = p.coordSys;
      const x = api.coord([api.value(0), 0])[0];
      if (x < cs.x || x > cs.x + cs.width) return { type: "group", children: [] };
      const yTop = cs.y;
      // Extend through the gap and the strip below (clip:false); the strip
      // grid ends ~18px above the container bottom.
      const yBottom = api.getHeight() - 20;
      return {
        type: "group",
        children: [
          {
            type: "line",
            shape: { x1: x, y1: yTop, x2: x, y2: yBottom },
            style: { stroke: JPCO_LINE, lineWidth: 1.5, lineDash: [5, 4] },
          },
          {
            type: "text",
            rotation: Math.PI / 2,
            origin: [x - 6, yTop + 4],
            style: {
              x: x - 6,
              y: yTop + 4,
              text: change.label,
              align: "right",
              verticalAlign: "middle",
              fill: JPCO_TEXT,
              fontSize: 10,
            },
          },
        ],
      };
    };

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
      // Prefer the daily BHP series; fall back to test-date BHP (original).
      bhpDailyPts.length > 0
        ? {
            name: "BHP (psi)",
            type: "line",
            xAxisIndex: 0,
            yAxisIndex: 1,
            data: bhpDailyPts,
            showSymbol: false,
            lineStyle: { color: BHP_COLOR, width: 1.5 },
            itemStyle: { color: BHP_COLOR },
          }
        : {
            name: "BHP (psi)",
            type: "line",
            xAxisIndex: 0,
            yAxisIndex: 1,
            data: bhpTestPts,
            symbolSize: 4,
            lineStyle: { color: BHP_COLOR, width: 2 },
            itemStyle: { color: BHP_COLOR },
          },
      {
        name: "JP changes",
        type: "custom",
        xAxisIndex: 0,
        yAxisIndex: 0,
        renderItem: renderChange,
        data: changeData,
        clip: false,
        silent: true,
        z: 20,
        tooltip: { show: false },
      },
      {
        name: "Pumps in hole",
        type: "custom",
        xAxisIndex: 1,
        yAxisIndex: 2,
        renderItem: renderBand,
        data: bandData,
        tooltip: {
          formatter: (raw: unknown): string => {
            // Custom-series tooltip param: value is this band's datum.
            const p = raw as { value: [number, number, number] };
            const era = eras[p.value[2]];
            const days = Math.round((era.end - era.start) / DAY_MS);
            return [
              `<b>${era.code}</b>`,
              `Set ${fmtDate(new Date(era.start))} to ${fmtDate(new Date(era.end))}`,
              `${fmtNum(days)} days`,
            ].join("<br/>");
          },
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

    return houseOption({
      tooltip: {
        ...baseTooltip,
        trigger: "axis",
        axisPointer: { type: "line" },
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
        ],
      },
      grid: [
        { left: 64, right: 64, top: 28, bottom: "24%" },
        { left: 64, right: 64, top: "84%", bottom: 18 },
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
        {
          type: "time",
          gridIndex: 1,
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
        { type: "value", gridIndex: 1, min: 0, max: 1, show: false },
      ],
      series,
    });
  }, [data, bhpFromZero, showPf]);

  const ref = useEChart(option);

  if (option === null) return null;
  return <div ref={ref} style={{ height }} />;
}
