/**
 * Separator Oil Loss - oil leaving with the first-stage separator water leg.
 *
 * Two SCADA tags carry the whole question: MPU_FI_5365 (water-leg flow, BPD,
 * essentially all of the field's produced water) and MPU_AI_5317 (the Red Eye
 * water-cut analyzer on that same stream). Oil out the water leg is
 * flow x (1 - wc) integrated in time; the server applies the three
 * corrections that make that integral honest - the analyzer's own trailing
 * plateau instead of a hard 100%, gating on FLOW so a real deep water-cut
 * excursion is never thrown away as an artifact, and the field oil rate as a
 * physical ceiling.
 *
 * The page exists to make the BAND unavoidable. Every barrel figure reads
 * "lower - upper" with its percent of field production beside it, so an
 * implausible as-read number announces itself instead of being quoted.
 *
 * The method itself sits behind two HOVER explainers rather than a standing
 * paragraph: what the band means, and how the upper and lower bound are
 * found, step by step. Both open on the same point, because it is the one
 * every reader assumes wrongly: the number is NOT flow x (1 - wc) off the Red
 * Eye. The method popover names the five corrections that separate the two
 * and closes with a worked day whose arithmetic is recomputed from the live
 * knobs (`workedExample`), so the prose can never drift from the payload.
 *
 * An operator OIW grab-sample workbook can be uploaded to overlay the SAMPLED
 * loss on the daily chart (POST /tools/sep-oil-loss/samples - parsed and
 * returned, held in page state, never stored). It is a different stream at
 * every sample point but V-5317: P-5417C sits downstream of the deoilers, so
 * the gap to the calculated band is deoiler recovery, not error, and the page
 * says so under the chart rather than letting the marks read as a check.
 */

import { useCallback, useEffect, useMemo, useState } from "react";

import { useOiwSamples, useSepOilLoss, useSepOilLossDay } from "../../api/hooks";
import type {
  OiwSampleDay,
  OiwSamplesResponse,
  SepLossDay,
  SepLossEvent,
  SepLossPeriod,
  SepOilLossResponse,
} from "../../api/types";
import { ChartPanel } from "../../charts/ChartPanel";
import type { EChartsOption } from "../../charts/echarts";
import {
  ACCENT,
  CRIMSON,
  GOLD,
  SLATE,
  axis,
  baseGrid,
  baseTooltip,
  houseOption,
  nearestByX,
  ttHeader,
  ttRow,
} from "../../charts/theme";
import {
  Badge,
  Button,
  Card,
  type Column,
  DataTable,
  ErrorNote,
  HelpPopover,
  InfoNote,
  Metric,
  Section,
  Spinner,
  WarnNote,
} from "../../components/ui";
import { downloadCsv } from "../../lib/csv";
import { fmtNum, fmtPct, fmtSigned } from "../../lib/format";
import { useDebounced } from "../../lib/useDebounced";

import { NumField } from "./ToolRun";

const WINDOWS = [7, 14, 30, 60, 90] as const;

/** Server ranges (routers/tools.py). Clamped so a half-typed value never 422s. */
const FIELD_OIL_MIN = 1_000;
const FIELD_OIL_MAX = 200_000;
const OIL_PCT_MIN = 1;
const OIL_PCT_MAX = 100;

/** Above this share of field oil the as-read number is not quotable. */
const PLAUSIBLE_PCT = 10;

/**
 * One typical day, used in the method explainer to show what the film
 * correction actually costs: a Red Eye plateaued at EX_PLATEAU on a
 * steady EX_FLOW leg, with one EX_MINUTES carry-under to EX_DIP.
 * Numbers match services/tools/sep_oil_loss.py `_oil_rates` / `_barrels`.
 */
const EX_FLOW = 70_000;
const EX_PLATEAU = 96;
const EX_DIP = 60;
const EX_MINUTES = 30;

/** Naive `flow x (1 - wc)` against both bounds for that day, recomputed from
 *  the live knobs so the arithmetic on screen is the arithmetic being run. */
function workedExample(
  fieldOil: number,
  oilPct: number,
): { naive: number; film: number; upper: number; lower: number; rate: number; capped: number } {
  const dipH = EX_MINUTES / 60;
  const film = EX_FLOW * ((100 - EX_PLATEAU) / 100) * ((24 - dipH) / 24);
  const naive = film + EX_FLOW * ((100 - EX_DIP) / 100) * (dipH / 24);
  const deficit = (EX_PLATEAU - EX_DIP) / 100;
  const rate = Math.min(EX_FLOW * deficit, fieldOil);
  const capped = EX_FLOW * Math.min(deficit, oilPct / 100);
  return { naive, film, upper: rate * (dipH / 24), lower: capped * (dipH / 24), rate, capped };
}

/** The leg flow is context behind the reading, not the reading: SLATE, faded. */
const FLOW_LINE = "rgba(100,116,139,0.45)";
/** ACCENT at 10% - the band between the two bounds. */
const BAND_FILL = "rgba(37,99,235,0.10)";
/** Same hue at 30% - the day the top chart is currently showing. */
const BAND_FOCUS = "rgba(37,99,235,0.30)";

/** Operator grab-sample defaults, matching services/tools/oiw_samples.py. */
const SAMPLE_SHEET = "OIW Daily";
const SAMPLE_LOCATION = "P-5417C";
/** The ONE sample point on the same stream as the calculated band. Every
 *  other tap is downstream of the deoilers, so the two are not comparable. */
const UPSTREAM_LOCATION = "V-5317";
const WATER_RATE_MIN = 1_000;
const WATER_RATE_MAX = 300_000;

const SELECT_CLS =
  "mt-1 block h-8 rounded-md border border-slate-300 bg-white px-2 text-sm " +
  "text-slate-800 outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-200";

const clamp = (v: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, v));

/** "2026-08-16T11:17:40-08:00" -> "2026-08-16 11:17", field offset kept. */
function fmtStamp(iso: string | null): string {
  if (!iso || iso.length < 16) return iso ?? "-";
  return `${iso.slice(0, 10)} ${iso.slice(11, 16)}`;
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/** "2026-08-09" -> "9-Aug". */
function dayMon(date: string): string {
  const month = Number(date.slice(5, 7));
  if (!Number.isFinite(month) || month < 1 || month > 12) return date;
  return `${Number(date.slice(8, 10))}-${MONTHS[month - 1]}`;
}

/** Minutes east of UTC carried by the server's own stamps, e.g. -480.
 *  Read from the data rather than assumed: Alaska is -09:00 in winter, and
 *  a browser in another zone must still tick on FIELD midnight. */
function fieldOffsetMinutes(iso: string): number {
  const sign = iso.slice(-6, -5);
  if (sign !== "+" && sign !== "-") return 0;
  const hours = Number(iso.slice(-5, -3));
  const minutes = Number(iso.slice(-2));
  if (!Number.isFinite(hours) || !Number.isFinite(minutes)) return 0;
  return (sign === "-" ? -1 : 1) * (hours * 60 + minutes);
}

/** Epoch ms -> "9-Aug" on the field's calendar, not the browser's. */
function dayMonFromMs(ms: number, offsetMinutes: number): string {
  const shifted = new Date(ms + offsetMinutes * 60_000);
  return `${shifted.getUTCDate()}-${MONTHS[shifted.getUTCMonth()]}`;
}

/** Epoch ms -> "14:30" on the field's clock. Used once the span is short
 *  enough that every tick would otherwise carry the same date. */
function clockFromMs(ms: number, offsetMinutes: number): string {
  const shifted = new Date(ms + offsetMinutes * 60_000);
  const hh = String(shifted.getUTCHours()).padStart(2, "0");
  const mm = String(shifted.getUTCMinutes()).padStart(2, "0");
  return `${hh}:${mm}`;
}

const band = (lo: number, hi: number, dp = 0): string => `${fmtNum(lo, dp)} - ${fmtNum(hi, dp)}`;

/** Percent-of-field band; "-" when the server had no denominator. */
function pctBand(p: SepLossPeriod): string {
  if (p.pct_field_lower === null || p.pct_field_upper === null) return "-";
  return `${band(p.pct_field_lower, p.pct_field_upper, 1)}%`;
}

// ---------------------------------------------------------------------------
// Series
// ---------------------------------------------------------------------------

type Pt = [number, number | null];

interface Traces {
  /** [epoch ms, the server's own ISO stamp] for tooltip headers. */
  stamps: [number, string][];
  wc: Pt[];
  base: Pt[];
  level: Pt[];
  levelSp: Pt[];
  flow: Pt[];
  cumLower: Pt[];
  cumUpper: Pt[];
  /** Stacked band carriers: lower bound, then the span up to the upper. */
  bandBase: [number, number][];
  bandSpan: [number, number][];
  /** Minutes east of UTC on the server's stamps; ticks label field days. */
  offsetMinutes: number;
  n: number;
}

/** The uploaded grab samples, indexed for the daily chart. `location` is the
 *  workbook's own spelling of the sample point, so the series label and the
 *  caveat always name the tap the marks actually came from. */
interface SampleOverlay {
  location: string;
  byDate: Record<string, OiwSampleDay | undefined>;
}

/** The `t` column of the loose series map. */
function timeCol(series: SepOilLossResponse["series"]): string[] {
  const raw = series.t;
  const out: string[] = [];
  if (!Array.isArray(raw)) return out;
  for (const v of raw) if (typeof v === "string") out.push(v);
  return out;
}

/** One numeric column of the loose series map, padded to `t`'s length. */
function numCol(series: SepOilLossResponse["series"], key: string, n: number): (number | null)[] {
  const out = new Array<number | null>(n).fill(null);
  const raw = series[key];
  if (!Array.isArray(raw)) return out;
  const len = Math.min(n, raw.length);
  for (let i = 0; i < len; i += 1) {
    const v: string | number | null = raw[i];
    if (typeof v === "number" && Number.isFinite(v)) out[i] = v;
  }
  return out;
}

/** Parallel arrays -> [x, y] traces on one epoch-ms clock. Takes the window
 *  payload or a single-day one; only `series` is read. */
function buildTraces(data: { series: SepOilLossResponse["series"] }): Traces {
  const iso = timeCol(data.series);
  const n = iso.length;
  const wc = numCol(data.series, "wc", n);
  const base = numCol(data.series, "base", n);
  const level = numCol(data.series, "level", n);
  const levelSp = numCol(data.series, "level_sp", n);
  const flow = numCol(data.series, "flow", n);
  const cumLower = numCol(data.series, "cum_lower", n);
  const cumUpper = numCol(data.series, "cum_upper", n);

  const tr: Traces = {
    stamps: [],
    wc: [],
    base: [],
    level: [],
    levelSp: [],
    flow: [],
    cumLower: [],
    cumUpper: [],
    bandBase: [],
    bandSpan: [],
    offsetMinutes: n > 0 ? fieldOffsetMinutes(iso[0]) : 0,
    n: 0,
  };
  for (let i = 0; i < n; i += 1) {
    const ms = Date.parse(iso[i]);
    if (!Number.isFinite(ms)) continue;
    tr.stamps.push([ms, iso[i]]);
    tr.wc.push([ms, wc[i]]);
    tr.base.push([ms, base[i]]);
    tr.level.push([ms, level[i]]);
    tr.levelSp.push([ms, levelSp[i]]);
    tr.flow.push([ms, flow[i]]);
    tr.cumLower.push([ms, cumLower[i]]);
    tr.cumUpper.push([ms, cumUpper[i]]);
    const lo = cumLower[i];
    const hi = cumUpper[i];
    // Both carriers get the same index filter, so the stack stays aligned.
    if (lo !== null && hi !== null) {
      tr.bandBase.push([ms, lo]);
      tr.bandSpan.push([ms, Math.max(hi - lo, 0)]);
    }
    tr.n += 1;
  }
  return tr;
}

// ---------------------------------------------------------------------------
// Charts
// ---------------------------------------------------------------------------

/** The axis-trigger tooltip fields these charts read. */
interface TipParam {
  seriesName?: string;
  color?: unknown;
  axisValue?: unknown;
  value?: unknown;
  dataIndex?: unknown;
}

interface UnitSpec {
  unit: string;
  dp: number;
}

/** Ticks on a time axis, on the FIELD calendar. ECharts would otherwise
 *  print a bare day number and silently use the browser's zone. Inside a
 *  36 h span every tick would carry the same date, so the drill-down ticks
 *  on the clock instead. */
function timeTicks(tr: Traces): Record<string, unknown> {
  const span =
    tr.stamps.length > 1 ? tr.stamps[tr.stamps.length - 1][0] - tr.stamps[0][0] : 0;
  const intraday = span > 0 && span <= 36 * 3_600_000;
  return {
    axisLabel: {
      color: SLATE,
      fontSize: 11,
      formatter: (value: number) =>
        intraday
          ? clockFromMs(value, tr.offsetMinutes)
          : dayMonFromMs(value, tr.offsetMinutes),
    },
  };
}

/**
 * Axis tooltip for a time chart carrying mixed units: the field's own stamp
 * as the header, one row per NAMED series. Series absent from `units` (the
 * band carriers) are skipped, so a carrier can never leak an epoch-ms datum.
 */
function timeTip(
  units: Record<string, UnitSpec>,
  stamps: [number, string][],
): (raw: unknown) => string {
  return (raw: unknown): string => {
    const list = (Array.isArray(raw) ? raw : [raw]) as TipParam[];
    const out: string[] = [];
    for (const p of list) {
      if (typeof p.axisValue === "number" && Number.isFinite(p.axisValue)) {
        const hit = nearestByX(stamps, p.axisValue);
        if (hit !== null) out.push(ttHeader(fmtStamp(hit[1])));
        break;
      }
    }
    for (const p of list) {
      const name = typeof p.seriesName === "string" ? p.seriesName : "";
      const spec = units[name];
      // Annotated unknown: Array.isArray widens an indexed datum to any.
      const v: unknown = Array.isArray(p.value) ? p.value[1] : p.value;
      if (spec === undefined || typeof v !== "number" || !Number.isFinite(v)) continue;
      const color = typeof p.color === "string" ? p.color : SLATE;
      out.push(ttRow(color, name, `${fmtNum(v, spec.dp)} ${spec.unit}`));
    }
    return out.join("");
  };
}

/**
 * Diagnostic chart: water cut against the analyzer's own plateau, with the
 * CONTROLLED level and its setpoint on the same percent axis. Carry-under
 * reads as a WC departure below the plateau; whether the level went with it
 * is what separates a level-control problem from a separation problem. Level
 * tracking setpoint through a deep WC drop is the interesting case.
 */
function wcOption(tr: Traces): EChartsOption {
  const hasLevel = tr.level.some((p) => p[1] !== null);
  const hasLevelSp = tr.levelSp.some((p) => p[1] !== null);
  const units: Record<string, UnitSpec> = {
    "Water cut": { unit: "%", dp: 1 },
    "Analyzer plateau": { unit: "%", dp: 1 },
    "Water leg flow": { unit: "BPD", dp: 0 },
  };
  const legend = ["Water cut", "Analyzer plateau"];
  const series: Record<string, unknown>[] = [
    {
      name: "Water cut",
      type: "line",
      yAxisIndex: 0,
      showSymbol: false,
      data: tr.wc,
      lineStyle: { color: ACCENT, width: 1.8 },
      itemStyle: { color: ACCENT },
      z: 6,
    },
    {
      name: "Analyzer plateau",
      type: "line",
      yAxisIndex: 0,
      showSymbol: false,
      data: tr.base,
      lineStyle: { color: SLATE, width: 1.4, type: "dashed" },
      itemStyle: { color: SLATE },
      z: 5,
    },
  ];
  if (hasLevel) {
    units["Controlled level"] = { unit: "%", dp: 1 };
    legend.push("Controlled level");
    series.push({
      name: "Controlled level",
      type: "line",
      yAxisIndex: 0,
      showSymbol: false,
      data: tr.level,
      lineStyle: { color: GOLD, width: 1.4 },
      itemStyle: { color: GOLD },
      z: 4,
    });
  }
  if (hasLevelSp) {
    units["Level setpoint"] = { unit: "%", dp: 1 };
    legend.push("Level setpoint");
    series.push({
      name: "Level setpoint",
      type: "line",
      yAxisIndex: 0,
      showSymbol: false,
      data: tr.levelSp,
      lineStyle: { color: GOLD, width: 1, type: "dotted" },
      itemStyle: { color: GOLD },
      z: 3,
    });
  }
  legend.push("Water leg flow");
  series.push({
    name: "Water leg flow",
    type: "line",
    yAxisIndex: 1,
    showSymbol: false,
    data: tr.flow,
    lineStyle: { color: FLOW_LINE, width: 1 },
    itemStyle: { color: FLOW_LINE },
    z: 2,
  });

  return houseOption({
    tooltip: { ...baseTooltip, trigger: "axis", formatter: timeTip(units, tr.stamps) },
    legend: { top: 4, right: 8, itemWidth: 18, textStyle: { fontSize: 12 }, data: legend },
    grid: { ...baseGrid, top: 52, right: 76, bottom: 28 },
    xAxis: { type: "time", ...axis(""), nameGap: 0, ...timeTicks(tr) },
    // Fixed 0-100: the reading is distance below the plateau, not autoscale.
    yAxis: [
      { type: "value", ...axis("Water cut / level (%)", { min: 0, max: 100 }) },
      {
        type: "value",
        position: "right",
        ...axis("Water leg flow (BPD)", { min: 0 }),
        nameGap: 44,
        splitLine: { show: false },
      },
    ],
    series,
  });
}

/**
 * The answer over the window, as a band. The shading between the bounds is a
 * transparent stacked carrier plus the span - markArea would not follow a
 * curve and a custom renderItem would drift off the axes on zoom.
 */
function cumOption(tr: Traces): EChartsOption {
  const units: Record<string, UnitSpec> = {
    "Lower bound": { unit: "bbl", dp: 0 },
    "Upper bound": { unit: "bbl", dp: 0 },
  };
  return houseOption({
    tooltip: { ...baseTooltip, trigger: "axis", formatter: timeTip(units, tr.stamps) },
    legend: {
      top: 4,
      right: 8,
      itemWidth: 18,
      textStyle: { fontSize: 12 },
      data: ["Lower bound", "Upper bound"],
    },
    grid: { ...baseGrid, top: 52, left: 76, bottom: 28 },
    xAxis: { type: "time", ...axis(""), nameGap: 0, ...timeTicks(tr) },
    yAxis: { type: "value", ...axis("Cumulative oil (bbl)", { min: 0 }) },
    series: [
      {
        type: "line",
        stack: "band",
        data: tr.bandBase,
        showSymbol: false,
        silent: true,
        lineStyle: { width: 0 },
        itemStyle: { color: "transparent" },
        areaStyle: { color: "transparent" },
        z: 1,
      },
      {
        type: "line",
        stack: "band",
        data: tr.bandSpan,
        showSymbol: false,
        silent: true,
        lineStyle: { width: 0 },
        itemStyle: { color: "transparent" },
        areaStyle: { color: BAND_FILL },
        z: 1,
      },
      {
        name: "Lower bound",
        type: "line",
        data: tr.cumLower,
        showSymbol: false,
        lineStyle: { color: ACCENT, width: 2 },
        itemStyle: { color: ACCENT },
        z: 5,
      },
      {
        name: "Upper bound",
        type: "line",
        data: tr.cumUpper,
        showSymbol: false,
        lineStyle: { color: CRIMSON, width: 1.8, type: "dashed" },
        itemStyle: { color: CRIMSON },
        z: 4,
      },
    ],
  });
}

/**
 * Per calendar day, the same band as a floating bar: the bar spans lower to
 * upper, so its height is the uncertainty and its position is the answer. A
 * transparent stacked base carries it off the axis, the same idiom as the
 * cumulative chart. Partial days - clipped by the window, or cut short by
 * separator downtime - are hatched, because a short day always looks quiet.
 * Clicking a bar drills the top chart into that day; the focused bar fills
 * solid so it is obvious which day the trace above belongs to.
 *
 * `samples` overlays the operators' grab-sample rate for the days it covers,
 * on the same barrels axis. It is a DIFFERENT stream at every location but
 * V-5317 - downstream of the deoilers - which is why the caption under the
 * chart says so rather than letting the marks read as a check on the band.
 */
function dailyOption(
  days: SepLossDay[],
  focus: string | null,
  samples: SampleOverlay | null,
): EChartsOption {
  const labels = days.map((d) => d.date);
  const base = days.map((d) => d.bbl_lower);
  const span = days.map((d) => ({
    value: Math.max(d.bbl_upper - d.bbl_lower, 0),
    itemStyle: {
      color: d.date === focus ? BAND_FOCUS : BAND_FILL,
      borderColor: CRIMSON,
      borderWidth: d.date === focus ? 1.5 : 1,
      ...(d.partial ? { borderType: "dashed" as const } : {}),
    },
  }));
  const lower = days.map((d) => d.bbl_lower);
  const sampleName = samples === null ? null : `Sampled (${samples.location})`;

  const series: Record<string, unknown>[] = [
    {
      type: "bar",
      stack: "day",
      data: base,
      silent: true,
      itemStyle: { color: "transparent" },
      z: 1,
    },
    {
      type: "bar",
      stack: "day",
      data: span,
      barMaxWidth: 26,
      z: 2,
    },
    {
      // The lower bound as a tick on top of the transparent base, so the
      // number to quote is a mark and not just where the shading starts.
      type: "scatter",
      data: lower,
      symbol: "rect",
      symbolSize: [18, 2],
      itemStyle: { color: ACCENT },
      silent: true,
      z: 5,
    },
  ];
  if (samples !== null && sampleName !== null) {
    // Hoisted: narrowing of a parameter does not survive into the callback.
    const byDate = samples.byDate;
    series.push({
      name: sampleName,
      type: "scatter",
      // Nulls on days with no grab sample: the marks are the sample record,
      // not a curve to interpolate across a day nobody walked out to.
      data: labels.map((d) => byDate[d]?.bbl ?? null),
      symbol: "diamond",
      symbolSize: 9,
      itemStyle: { color: GOLD },
      z: 6,
    });
  }

  return houseOption({
    tooltip: {
      ...baseTooltip,
      trigger: "axis",
      axisPointer: { type: "shadow" },
      formatter: (raw: unknown): string => {
        const list = (Array.isArray(raw) ? raw : [raw]) as TipParam[];
        const idx = list.length > 0 ? list[0].dataIndex : undefined;
        const day = typeof idx === "number" ? days[idx] : undefined;
        if (day === undefined) return "";
        const rows = [
          ttRow(ACCENT, "Lower bound", `${fmtNum(day.bbl_lower)} bbl`),
          ttRow(CRIMSON, "Upper bound", `${fmtNum(day.bbl_upper)} bbl`),
          ttRow(SLATE, "Upset", `${fmtNum(day.upset_hours, 1)} h, ${day.events} events`),
          ttRow(SLATE, "Running", `${fmtNum(day.hours, 1)} of ${fmtNum(day.covered_hours, 1)} h`),
        ];
        if (day.pct_field_upper !== null && day.pct_field_lower !== null) {
          rows.push(
            ttRow(
              SLATE,
              "Share of field oil",
              `${fmtNum(day.pct_field_lower, 1)} - ${fmtNum(day.pct_field_upper, 1)}%`,
            ),
          );
        }
        const hit = samples === null ? undefined : samples.byDate[day.date];
        if (hit !== undefined && sampleName !== null) {
          rows.push(
            ttRow(
              GOLD,
              sampleName,
              `${fmtNum(hit.bbl)} bbl at ${fmtNum(hit.ppm_mean)} ppm, ` +
                `${hit.samples} ${hit.samples === 1 ? "sample" : "samples"}`,
            ),
          );
        }
        return ttHeader(day.date) + rows.join("");
      },
    },
    // No legend without the overlay: the band's two carriers are unnamed, so
    // there would be nothing to list.
    legend:
      sampleName === null
        ? { show: false }
        : {
            top: 4,
            right: 8,
            itemWidth: 18,
            textStyle: { fontSize: 12 },
            data: [sampleName],
          },
    grid: { ...baseGrid, top: sampleName === null ? 24 : 44, left: 76, bottom: 28 },
    xAxis: {
      type: "category",
      data: labels,
      ...axis(""),
      nameGap: 0,
      // Short labels, so let ECharts drop the ones that will not fit rather
      // than rotating every one of them.
      axisLabel: { color: SLATE, fontSize: 11, formatter: dayMon },
    },
    yAxis: { type: "value", ...axis("Oil to the water leg (bbl)", { min: 0 }) },
    series,
  });
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

/** Worst to least: a lost vessel outranks a loop that is merely under. */
const KIND_TONE: Record<SepLossEvent["kind"], "poor" | "fair" | "info"> = {
  "level loss": "poor",
  "off setpoint": "fair",
  "at setpoint": "info",
};

const KIND_HELP: Record<SepLossEvent["kind"], string> = {
  "level loss": "Vessel lost its water inventory during the upset.",
  "off setpoint": "Loop held well under the level it was calling for.",
  "at setpoint": "Level was held as asked and the water leg still ran oil.",
};

const EVENT_COLUMNS: Column<SepLossEvent>[] = [
  { key: "start", label: "Start", render: (r) => fmtStamp(r.start) },
  { key: "hours", label: "Hours", align: "right", render: (r) => fmtNum(r.hours, 1) },
  {
    key: "wc_min",
    label: "WC min (%)",
    align: "right",
    help: "Deepest analyzer reading in the excursion. A real oil sweep, not a dropout.",
    render: (r) => fmtNum(r.wc_min, 1),
  },
  { key: "wc_avg", label: "WC avg (%)", align: "right", render: (r) => fmtNum(r.wc_avg, 1) },
  {
    key: "flow_avg",
    label: "Leg flow (BPD)",
    align: "right",
    render: (r) => fmtNum(r.flow_avg, 0),
  },
  {
    key: "bbl_lower",
    label: "Lower (bbl)",
    align: "right",
    help: "Oil fraction of the leg capped at the oil-fraction cap.",
    render: (r) => fmtNum(r.bbl_lower, 0),
  },
  {
    key: "bbl_upper",
    label: "Upper (bbl)",
    align: "right",
    help: "Analyzer as read, film-corrected, capped at the field oil rate.",
    render: (r) => fmtNum(r.bbl_upper, 0),
  },
  {
    key: "level_min",
    label: "Level min (%)",
    align: "right",
    help: "Lowest controlled level (MPU_LIC_5365CV1) during the upset.",
    render: (r) => fmtNum(r.level_min, 1),
  },
  {
    key: "level_sp_avg",
    label: "Setpoint (%)",
    align: "right",
    help: "Level the loop was calling for during the upset.",
    render: (r) => fmtNum(r.level_sp_avg, 1),
  },
  {
    key: "level_dev_avg",
    label: "Dev (pts)",
    align: "right",
    help: "Controlled level minus setpoint. Negative means the loop is under.",
    render: (r) => fmtSigned(r.level_dev_avg, 1),
  },
  {
    key: "kind",
    label: "Level",
    help:
      "Level loss: the vessel dropped below 20%. Off setpoint: held more than " +
      "10 points under what the loop called for. At setpoint: level was where " +
      "it was asked and the leg ran oil anyway - separation, not level control.",
    render: (r) => (
      <Badge tone={KIND_TONE[r.kind]} title={KIND_HELP[r.kind]}>
        {r.kind}
      </Badge>
    ),
  },
];

const EVENT_CSV = EVENT_COLUMNS.map((c) => ({ key: c.key, label: c.label }));

function PeriodCard({ p }: { p: SepLossPeriod }) {
  const implausible = p.pct_field_upper !== null && p.pct_field_upper > PLAUSIBLE_PCT;
  return (
    <Card>
      <div className="flex items-baseline justify-between gap-2">
        <h4 className="text-sm font-semibold tracking-tight text-slate-700">{p.label}</h4>
        <Badge tone="neutral">{fmtNum(p.events)} events</Badge>
      </div>
      <div className="mt-2 text-2xl font-semibold tabular-nums text-slate-800">
        {band(p.bbl_lower, p.bbl_upper)}
        <span className="ml-1.5 text-sm font-normal text-slate-500">bbl</span>
      </div>
      <div className="mt-1 text-xs text-slate-500">
        {band(p.bopd_lower, p.bopd_upper)} BOPD, {pctBand(p)} of field oil
      </div>
      <dl className="mt-3 grid grid-cols-[1fr_auto] gap-x-4 gap-y-1 text-xs">
        <dt className="text-slate-500">Upset hours</dt>
        <dd className="text-right tabular-nums text-slate-700">{fmtNum(p.upset_hours, 1)}</dd>
        <dt className="text-slate-500">Valid hours</dt>
        <dd className="text-right tabular-nums text-slate-700">{fmtNum(p.hours, 1)}</dd>
        <dt className="text-slate-500">Separator down (h)</dt>
        <dd className="text-right tabular-nums text-slate-700">
          {fmtNum(p.downtime_hours, 1)}
        </dd>
        <dt className="text-slate-500">Leg WC / plateau</dt>
        <dd className="text-right tabular-nums text-slate-700">
          {fmtNum(p.wc_avg, 1)} / {fmtNum(p.base_avg, 1)}%
        </dd>
      </dl>
      {implausible && (
        <WarnNote className="mt-3">
          The analyzer as read puts {fmtNum(p.pct_field_upper, 1)}% of field oil production out
          the water leg, more than a plausible share. Quote the lower bound.
        </WarnNote>
      )}
    </Card>
  );
}

export default function SepOilLossPage() {
  const [days, setDays] = useState(14);
  const [fieldOilIn, setFieldOilIn] = useState(65_000);
  // 10% of the leg is the conservative cap an engineer will actually defend:
  // above that the "lower" bound stops being a floor and just tracks the
  // as-read meter, which is the number this page exists to bracket.
  const [oilPctIn, setOilPctIn] = useState(10);
  // A daily bar drills the top chart into that field day at full resolution.
  const [focusDay, setFocusDay] = useState<string | null>(null);

  // Debounced so typing a five-digit rate is one request, not five.
  const fieldOilTyped = useDebounced(fieldOilIn, 400);
  const oilPctTyped = useDebounced(oilPctIn, 400);
  const fieldOil = clamp(fieldOilTyped, FIELD_OIL_MIN, FIELD_OIL_MAX);
  const oilPct = clamp(oilPctTyped, OIL_PCT_MIN, OIL_PCT_MAX);
  const outOfRange = fieldOilTyped !== fieldOil || oilPctTyped !== oilPct;
  // The method explainer's arithmetic, on the knobs currently in force.
  const example = useMemo(() => workedExample(fieldOil, oilPct), [fieldOil, oilPct]);

  const query = useSepOilLoss(days, fieldOil, oilPct / 100);
  const data = query.data ?? null;
  const dayQuery = useSepOilLossDay(focusDay, days, fieldOil, oilPct / 100);
  const dayData = dayQuery.data && dayQuery.data.date === focusDay ? dayQuery.data : null;

  // The uploaded workbook lives in page state and nowhere else: the endpoint
  // parses and returns, exactly like the gauge tool, so a reload dropping the
  // overlay is the correct behaviour and not lost work.
  const [sampleFile, setSampleFile] = useState<File | null>(null);
  const [samples, setSamples] = useState<OiwSamplesResponse | null>(null);
  const [sampleLoc, setSampleLoc] = useState(SAMPLE_LOCATION);
  const [waterRateIn, setWaterRateIn] = useState(95_000);
  const waterRate = clamp(useDebounced(waterRateIn, 500), WATER_RATE_MIN, WATER_RATE_MAX);
  const { mutate: parseSamples, isPending: parsing, error: parseError, reset: resetParse } =
    useOiwSamples();

  // One trigger for every input: picking a file, switching sample point and
  // changing the water-rate basis all re-parse the same File, because the
  // server kept nothing to re-roll and the basis is part of the answer.
  useEffect(() => {
    if (sampleFile === null) return;
    parseSamples(
      { file: sampleFile, location: sampleLoc, waterRateBpd: waterRate, sheet: SAMPLE_SHEET },
      {
        onSuccess: (parsed) => {
          setSamples(parsed);
          // The workbook's own spelling of the tap wins, so the picker and
          // the chart label read the way the log does.
          if (parsed.location !== sampleLoc) setSampleLoc(parsed.location);
        },
      },
    );
  }, [sampleFile, sampleLoc, waterRate, parseSamples]);

  const clearSamples = useCallback(() => {
    setSampleFile(null);
    setSamples(null);
    resetParse();
  }, [resetParse]);

  const overlay = useMemo<SampleOverlay | null>(() => {
    if (samples === null || samples.daily.length === 0) return null;
    const byDate: Record<string, OiwSampleDay | undefined> = {};
    for (const day of samples.daily) byDate[day.date] = day;
    return { location: samples.location, byDate };
  }, [samples]);
  const downstream = overlay !== null && overlay.location.toUpperCase() !== UPSTREAM_LOCATION;

  // Clicking a bar swaps the top chart's source; everything else is shared,
  // so the same builder draws both and the axes cannot drift apart.
  const tr = useMemo(() => (data ? buildTraces(data) : null), [data]);
  const dayTr = useMemo(() => (dayData ? buildTraces(dayData) : null), [dayData]);
  const wcSource = dayTr && dayTr.n > 0 ? dayTr : tr;
  const wcOpt = useMemo(
    () => (wcSource && wcSource.n > 0 ? wcOption(wcSource) : null),
    [wcSource],
  );
  const cumOpt = useMemo(() => (tr && tr.n > 0 ? cumOption(tr) : null), [tr]);
  const dailyOpt = useMemo(
    () => (data && data.daily.length > 0 ? dailyOption(data.daily, focusDay, overlay) : null),
    [data, focusDay, overlay],
  );

  const pickDay = useCallback((date: string) => {
    setFocusDay((cur) => (cur === date ? null : date));
  }, []);

  function exportCsv() {
    if (!data) return;
    downloadCsv(`sep_oil_loss_events_${data.days}d.csv`, EVENT_CSV, data.events);
  }

  return (
    <div className="space-y-4">
      <Section title="Separator Oil Loss">
        <div className="flex flex-wrap items-end gap-3">
          <div>
            <span className="text-xs text-slate-500">Window (days)</span>
            <div className="mt-1 flex flex-wrap gap-1">
              {WINDOWS.map((d) => (
                <button
                  key={d}
                  type="button"
                  // A shorter window may not cover the focused day; drop it
                  // rather than let the drill-down 400 on a stale date.
                  onClick={() => {
                    setDays(d);
                    setFocusDay(null);
                  }}
                  className={
                    "h-8 min-w-10 rounded-md border px-2 text-sm transition-colors " +
                    (days === d
                      ? "border-blue-500 bg-blue-50 font-medium text-blue-700"
                      : "border-slate-300 bg-white text-slate-600 hover:bg-slate-50")
                  }
                >
                  {d}
                </button>
              ))}
            </div>
          </div>
          <NumField
            label="Field oil (BOPD)"
            value={fieldOilIn}
            onChange={setFieldOilIn}
            min={FIELD_OIL_MIN}
            max={FIELD_OIL_MAX}
            step={1000}
          />
          <NumField
            label="Max oil in water leg (%)"
            value={oilPctIn}
            onChange={setOilPctIn}
            min={OIL_PCT_MIN}
            max={OIL_PCT_MAX}
            step={5}
          />
          <div>
            <span className="text-xs text-slate-500">OIW samples (.xlsx)</span>
            <input
              type="file"
              accept=".xlsx"
              aria-label="OIW samples (.xlsx)"
              onChange={(e) => {
                const picked = e.target.files;
                if (picked && picked.length > 0) setSampleFile(picked[0]);
                e.target.value = ""; // same file re-pickable after a Clear
              }}
              className={
                "mt-1 block h-8 w-56 text-xs text-slate-600 file:mr-2 file:h-8 " +
                "file:rounded-md file:border file:border-slate-300 file:bg-white " +
                "file:px-2 file:text-xs file:font-medium file:text-slate-700 " +
                "hover:file:bg-slate-50"
              }
            />
          </div>
          {samples !== null && samples.locations_available.length > 0 && (
            <label className="block">
              <span className="text-xs text-slate-500">Sample point</span>
              <select
                value={sampleLoc}
                onChange={(e) => setSampleLoc(e.target.value)}
                className={SELECT_CLS}
              >
                {samples.locations_available.map((loc) => (
                  <option key={loc} value={loc}>
                    {loc}
                  </option>
                ))}
              </select>
            </label>
          )}
          <NumField
            label="Sample water rate (BPD)"
            value={waterRateIn}
            onChange={setWaterRateIn}
            min={WATER_RATE_MIN}
            max={WATER_RATE_MAX}
            step={5000}
            width="w-32"
          />
          {sampleFile !== null && (
            <Button size="sm" variant="ghost" onClick={clearSamples}>
              Clear samples
            </Button>
          )}
        </div>

        {outOfRange && (
          <WarnNote className="mt-3">
            Field oil is limited to {fmtNum(FIELD_OIL_MIN)} - {fmtNum(FIELD_OIL_MAX)} BOPD and the
            oil cap to {OIL_PCT_MIN} - {OIL_PCT_MAX}%. Showing the nearest valid value
            ({fmtNum(fieldOil)} BOPD, {fmtNum(oilPct)}%).
          </WarnNote>
        )}

        {parsing && <Spinner label="Parsing OIW samples" />}
        {parseError !== null && <ErrorNote error={parseError} className="mt-3" />}
        {samples !== null && samples.sample_count > 0 && (
          <div className="mt-3 flex flex-wrap items-center gap-2 text-xs text-slate-500">
            <Badge
              tone="info"
              title={`${samples.filename}, sheet ${samples.sheet}. Held in this page only - a reload clears it.`}
            >
              {fmtNum(samples.sample_count)} samples at {samples.location}
              {samples.first_date !== null && samples.last_date !== null
                ? `, ${samples.first_date} to ${samples.last_date}`
                : ""}
            </Badge>
            <span>
              Sampled rate is ppm x {fmtNum(samples.water_rate_bpd)} BPD / 1e6, one unweighted
              mean per Alaska day
            </span>
          </div>
        )}
        {samples !== null && samples.sample_count === 0 && (
          <WarnNote className="mt-3">
            No samples at {samples.location} on sheet {samples.sheet}. Pick another sample point
            from the list.
          </WarnNote>
        )}
      </Section>

      {/* The method is a hover away rather than a standing paragraph: it is
          read once and then only gets between the engineer and the charts. */}
      <div className="flex flex-wrap items-center gap-2">
        <HelpPopover label="What the band means" title="Quote the band, never one end">
          <p>
            Neither end is the raw meter arithmetic. Flow x (1 - wc) off the Red Eye is NOT the
            answer: it bills the analyzer&apos;s film as oil around the clock, counts every hour
            the separator is down as 100% oil, and can imply more oil out the water leg than the
            field produces. Hover &quot;How the bounds are found&quot; for the five corrections.
          </p>
          <p className="mt-2">
            What you get instead is a band. Every barrel figure reads lower - upper with its
            share of field oil beside it. The UPPER bound is the analyzer as read AFTER the film
            correction and the flow gate, with the instantaneous oil rate held to{" "}
            {fmtNum(fieldOil)} BOPD - all of the field&apos;s oil short-circuiting the vessel is
            the absolute physical ceiling. The LOWER bound is the same integral with the oil
            fraction of the leg capped at {fmtNum(oilPct)}%.
          </p>
          <p className="mt-2">
            The two ends can differ several-fold during a bad excursion, because the meter can
            imply 70,000-87,000 BOPD out the water leg, which is more oil than the field makes.
            Quoting the upper end alone invents barrels; quoting the lower end alone buries the
            problem.
          </p>
        </HelpPopover>
        <HelpPopover
          label="How the bounds are found"
          title="Upper and lower bound, step by step"
          width="w-[34rem]"
        >
          <p className="mb-2">
            Start from flow x (1 - wc) and understand why it is wrong. On a filmed meter it
            charges oil that is not there, on a down separator it charges the whole leg, and on a
            deep excursion it charges more oil than the field makes. Five corrections, in order:
          </p>
          <ol className="list-decimal space-y-1.5 pl-4">
            <li>
              <span className="font-medium text-slate-700">Clock.</span> The water-leg flow meter
              is the fastest and most regular tag, so it sets the integration clock; water cut,
              level and setpoint step-hold onto it, which is what an exception-reported historian
              means. No single sample stands for more than 15 minutes, so one dropout cannot
              smear a stale value across hours.
            </li>
            <li>
              <span className="font-medium text-slate-700">Gate.</span> Samples at or below the
              flow gate are dropped outright - the separator is down or the meter is drifting
              around zero. The gate is on FLOW only, never on the water-cut value: the deep
              90 - 60 - 30 - 5 sweeps are real carry-under, and screening on the analyzer would
              delete the events this page exists to find.
            </li>
            <li>
              <span className="font-medium text-slate-700">Datum.</span> Each reading is
              referenced to the analyzer&apos;s OWN trailing 24 h p95 water cut, clipped to
              80-100%. A filmed Red Eye stops reaching 100%, so a straight 100 - wc bills the
              film as oil forever; only departures below the meter&apos;s own plateau are
              charged.
            </li>
            <li>
              <span className="font-medium text-slate-700">Rate.</span> Implied oil fraction is
              plateau minus reading, never negative. UPPER rate is leg flow x that fraction,
              clipped at {fmtNum(fieldOil)} BOPD. LOWER rate is leg flow x the same fraction with
              the FRACTION capped at {fmtNum(oilPct)}%, so the cap bites only on the excursions
              that imply more oil in the leg than it could plausibly carry.
            </li>
            <li>
              <span className="font-medium text-slate-700">Barrels.</span> Rate x hours / 24,
              summed over the valid samples in the look-back. Percent of field divides that by{" "}
              {fmtNum(fieldOil)} BOPD over the same valid hours, so downtime never inflates the
              share.
            </li>
          </ol>
          <p className="mt-2">
            <span className="font-medium text-slate-700">Worked example.</span> A meter plateaued
            at {EX_PLATEAU}% on a steady {fmtNum(EX_FLOW)} BPD leg, with one {EX_MINUTES} minute
            carry-under to {EX_DIP}%. Flow x (1 - wc) charges {fmtNum(example.naive)} bbl that
            day, {fmtNum(example.film)} bbl of it film the meter can no longer read off. This
            page charges NOTHING while the reading sits on its own plateau, then{" "}
            {fmtNum(example.upper)} bbl upper and {fmtNum(example.lower)} bbl lower for the dip -
            {" "}{fmtNum(example.rate)} BOPD as read against {fmtNum(example.capped)} BOPD at
            the {fmtNum(oilPct)}% cap.
          </p>
          <p className="mt-2">
            Both knobs are yours: field oil {fmtNum(FIELD_OIL_MIN)} - {fmtNum(FIELD_OIL_MAX)}{" "}
            BOPD, oil cap {OIL_PCT_MIN} - {OIL_PCT_MAX}%. Raising field oil loosens the ceiling
            and lowers every percent-of-field share; tightening the cap pulls the lower bound
            down. Above {PLAUSIBLE_PCT}% of field oil the as-read end is not quotable.
          </p>
        </HelpPopover>
      </div>

      {query.isError && <ErrorNote error={query.error} />}
      {query.isLoading && <Spinner label="Reading the separator historian" />}
      {data && query.isFetching && <Spinner label="Updating" />}

      {data && (
        <div className="flex flex-wrap items-center gap-2 text-xs text-slate-500">
          <Badge tone="info" title="Water-leg flow meter, BPD">
            {data.flow_tag}
          </Badge>
          <Badge tone="info" title="Red Eye water cut on the water leg, %">
            {data.wc_tag}
          </Badge>
          {data.level_tag && (
            <Badge tone="info" title="Controlled level indication - the level the loop acts on, %">
              {data.level_tag}
            </Badge>
          )}
          {data.level_sp_tag && (
            <Badge tone="info" title="Level setpoint for that loop, %">
              {data.level_sp_tag}
            </Badge>
          )}
          {data.start !== null && data.end !== null && (
            <span>
              Window {fmtStamp(data.start)} to {fmtStamp(data.end)}
            </span>
          )}
          <span>
            Flow gate {fmtNum(data.flow_min_bpd)} BPD, upset {fmtNum(data.upset_drop_pts, 1)}{" "}
            points below plateau
          </span>
        </div>
      )}

      {data && data.periods.length === 0 && (
        <InfoNote>
          The historian returned no hours above the {fmtNum(data.flow_min_bpd)} BPD flow gate in
          this window, so there is nothing to integrate. Widen the window or check that
          {" "}{data.flow_tag} is reporting.
        </InfoNote>
      )}

      {data && data.periods.length > 0 && (
        <>
          <Card>
            <div className="flex flex-wrap gap-3">
              <Metric
                label="Valid hours"
                value={fmtNum(data.valid_hours, 1)}
                sub={`${fmtNum(data.excluded_hours, 1)} h excluded below the flow gate`}
              />
              <Metric
                label="Field oil basis"
                value={`${fmtNum(data.field_oil_bopd)} BOPD`}
                title="Ceiling on the instantaneous rate and the percent-of-field denominator."
                sub="ceiling on the instantaneous rate"
              />
              <Metric
                label="Oil-fraction cap"
                value={fmtPct(data.max_oil_frac, 0)}
                sub="sets the lower bound"
              />
            </div>
          </Card>

          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {data.periods.map((p) => (
              <PeriodCard key={p.label} p={p} />
            ))}
          </div>
        </>
      )}

      {wcOpt && (
        <Section
          title={
            focusDay === null
              ? "Water cut, controlled level and setpoint"
              : `Water cut, controlled level and setpoint - ${dayMon(focusDay)}`
          }
          actions={
            focusDay !== null && (
              <Button size="sm" variant="ghost" onClick={() => setFocusDay(null)}>
                Show full window
              </Button>
            )
          }
        >
          <Card>
            {focusDay !== null && dayData === null && !dayQuery.isError && (
              <Spinner label={`Loading ${focusDay}`} />
            )}
            {dayQuery.isError && <ErrorNote error={dayQuery.error} />}
            {/* Only the percent axis zooms: the BPD axis carries context. */}
            <ChartPanel option={wcOpt} height={420} zoom={{ xAxisIndex: [0], yAxisIndex: [0] }} />
            {dayData?.summary && (
              <p className="mt-2 px-1 text-xs text-slate-500">
                {dayMon(dayData.date)}: {band(dayData.summary.bbl_lower, dayData.summary.bbl_upper)}{" "}
                bbl over {fmtNum(dayData.summary.upset_hours, 1)} upset hours in{" "}
                {dayData.events.length} events. One point per minute here, against roughly one
                per quarter hour on the full window.
              </p>
            )}
          </Card>
        </Section>
      )}

      {dailyOpt && (
        <Section title="Oil to the water leg by field day">
          <Card>
            <ChartPanel
              option={dailyOpt}
              height={340}
              zoom={{ xAxisIndex: [0], yAxisIndex: [0] }}
              onSelect={pickDay}
            />
            <p className="mt-2 px-1 text-xs text-slate-500">
              Click a bar to drill the chart above into that day; click it again to come back.
              Bar spans the lower to upper bound. Dashed outline marks a day the window only
              clips or the separator spent partly down, so it is not comparable bar for bar.
              Days are Alaska local and cover the whole window, so they will not sum to a
              rolling look-back card.
            </p>
            {overlay !== null && samples !== null && (
              <p className="mt-1 px-1 text-xs text-slate-500">
                Gold diamonds are the operators&apos; grab samples at {overlay.location}: the
                day&apos;s unweighted mean of ppm x {fmtNum(samples.water_rate_bpd)} BPD / 1e6,
                on the days somebody pulled a sample. Held in this page only.
              </p>
            )}
            {downstream && overlay !== null && (
              <WarnNote className="mt-2">
                Sampled OIW at {overlay.location} is taken DOWNSTREAM of the deoilers
                (V-5419 / V-5421 / V-5422 / V-5425), while the calculated band is the
                first-stage water leg UPSTREAM of them. The gap between the diamonds and the
                bars is deoiler recovery, not measurement error - the two are not measuring
                the same stream. Only {UPSTREAM_LOCATION} samples the stream the band
                describes.
              </WarnNote>
            )}
          </Card>
        </Section>
      )}

      {cumOpt && (
        <Section title="Cumulative oil to the water leg">
          <Card>
            <ChartPanel option={cumOpt} height={360} zoom={{ xAxisIndex: [0], yAxisIndex: [0] }} />
          </Card>
        </Section>
      )}

      {data && data.events.length > 0 && (
        <Section
          title={`Carry-under events (${data.events.length}, worst first)`}
          actions={
            <Button size="sm" variant="ghost" onClick={exportCsv}>
              Download CSV
            </Button>
          }
        >
          <DataTable
            columns={EVENT_COLUMNS}
            rows={data.events}
            rowKey={(r) => r.start}
            maxHeight="30rem"
            sortable
            emptyLabel="No carry-under events in this window"
          />
        </Section>
      )}
    </div>
  );
}
