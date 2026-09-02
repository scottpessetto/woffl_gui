/**
 * Pad booster-plant charts (S / I / M) - the three views the pad
 * optimization earns:
 *
 * 1. Station curve - delivered header pressure vs total flow: the
 *    curve family (one line per pump count on fixed speed, iso-speed lines
 *    on VFD), the capability frontier where one exists, the allowable and
 *    preferred operating bands, and the duty point the optimizer landed on.
 *    One picture answers whether the plant can hold this header at this rate.
 * 2. Pump performance, one card per machine - the vendor curve sheet: head,
 *    BHP and efficiency vs per-pump flow on a single grid with the duty flow
 *    marked. This is the sheet the vendor ships, so it is the sheet the plan
 *    gets checked against.
 * 3. Optimizer trace - where the header came from: the pressure sweep with
 *    its oil and power-fluid response (free_pressure pads), or the
 *    fixed-point iteration walking onto the pump curve (fixed_curve pads).
 *    Without it the header is a number with no provenance.
 *
 * The curves are static plant physics from GET /api/optimize/pump-curve, so
 * panels 1 and 2 render before any run; the duty point and panel 3 come from
 * the run meta.
 */

import clsx from "clsx";
import { useMemo } from "react";

import { usePumpCurve } from "../../api/hooks";
import type { EPadBuild, PumpCurveResponse } from "../../api/types";
import type { EChartsOption } from "../../charts/echarts";
import {
  ACCENT,
  axis,
  baseGrid,
  baseTooltip,
  CRIMSON,
  houseOption,
  nearestByX,
  SLATE,
  ttHeader,
} from "../../charts/theme";
import { ChartPanel } from "../../charts/ChartPanel";
import { Card } from "../../components/ui";
import { fmtNum } from "../../lib/format";

import {
  carrierSeries,
  MACHINE_HELP,
  machineOption,
  refLabel,
  tipAxisNum,
  tipAxisText,
  tipParams,
  tipRows,
  type Duty,
  type UnitSpec,
} from "./curveChart";

/** meta values arrive JSON-flattened; narrow numerics defensively. */
function metaNum(meta: Record<string, unknown>, key: string): number | null {
  const v = meta[key];
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

interface HistoryRow {
  iter: number;
  trialPsi: number;
  curvePsi: number | null;
  totalBpd: number | null;
}

/** Sweep trial as [header_psi, oil_bopd, total_pf_bpd]. */
type SweepPoint = [number, number, number];

const NO_DUTY: Duty = { headerPsi: null, totalBpd: null, perPumpBpd: null };

/** The point the optimizer settled on. Fixed-curve pads carry the per-pump
 *  split in the meta; the rest divide the station total. */
function readDuty(meta: Record<string, unknown> | null, nPumps: number | null): Duty {
  if (meta === null) return NO_DUTY;
  const totalBpd = metaNum(meta, "total_pf_bpd");
  const perPump = metaNum(meta, "per_pump_bpd");
  return {
    headerPsi: metaNum(meta, "header_psi"),
    totalBpd,
    perPumpBpd: perPump ?? (totalBpd !== null ? totalBpd / (nPumps ?? 1) : null),
  };
}

/** Pressure-sweep trials, sorted by header so nearestByX can index them. */
function readSweep(meta: Record<string, unknown> | null): SweepPoint[] {
  const raw = meta === null ? null : meta.sweep;
  if (!Array.isArray(raw)) return [];
  const out: SweepPoint[] = [];
  for (const row of raw) {
    if (typeof row !== "object" || row === null) continue;
    const r = row as Record<string, unknown>;
    const psi = metaNum(r, "header_psi");
    const oil = metaNum(r, "total_oil_bopd");
    const pf = metaNum(r, "total_pf_bpd");
    if (psi === null || oil === null || pf === null) continue;
    out.push([psi, oil, pf]);
  }
  return out.sort((a, b) => a[0] - b[0]);
}

/** Fixed-point iterations in order. The settled column is curve_psi on
 *  fixed_curve pads and frontier_psi on free_pressure ones. */
function readHistory(meta: Record<string, unknown> | null): HistoryRow[] {
  const raw = meta === null ? null : meta.history;
  if (!Array.isArray(raw)) return [];
  const out: HistoryRow[] = [];
  for (const row of raw) {
    if (typeof row !== "object" || row === null) continue;
    const r = row as Record<string, unknown>;
    const iter = metaNum(r, "iter");
    const trial = metaNum(r, "trial_psi");
    if (iter === null || trial === null) continue;
    out.push({
      iter,
      trialPsi: trial,
      curvePsi: metaNum(r, "curve_psi") ?? metaNum(r, "frontier_psi"),
      totalBpd: metaNum(r, "total_pf_bpd"),
    });
  }
  return out.sort((a, b) => a.iter - b.iter);
}

/* ------------------------------------------------------------ panel 1 */

function stationOption(
  curve: PumpCurveResponse,
  duty: Duty,
  sweep: SweepPoint[],
  history: HistoryRow[],
): EChartsOption {
  const st = curve.station;
  const series: Record<string, unknown>[] = [];
  const units: Record<string, UnitSpec> = {};
  const legend: string[] = [];

  const addPsi = (name: string) => {
    legend.push(name);
    units[name] = { unit: "psi", dp: 0 };
  };

  for (const line of st.curves) {
    addPsi(line.label);
    series.push({
      name: line.label,
      type: "line",
      showSymbol: false,
      data: line.points,
      lineStyle: line.active
        ? { color: ACCENT, width: 2.4 }
        : { color: SLATE, width: 1.2, type: "dashed" },
      itemStyle: { color: line.active ? ACCENT : SLATE },
      z: line.active ? 5 : 3,
    });
  }

  if (st.frontier !== null) {
    addPsi(st.frontier.label);
    series.push({
      name: st.frontier.label,
      type: "line",
      showSymbol: false,
      data: st.frontier.points,
      lineStyle: { color: CRIMSON, width: 2, type: "dashed" },
      itemStyle: { color: CRIMSON },
      z: 4,
    });
  }

  series.push(
    carrierSeries({
      aor: st.aor,
      por: st.por,
      bep: st.bep,
      minFlow: st.min_flow,
      cap: st.header_cap,
      duty: null,
    }),
  );

  // Iteration order, so the dotted path reads as the walk it was.
  const path: [number, number][] = [];
  for (const h of history) {
    if (h.totalBpd !== null) path.push([h.totalBpd, h.trialPsi]);
  }
  if (path.length > 0) {
    addPsi("Fixed-point path");
    series.push({
      name: "Fixed-point path",
      type: "line",
      data: path,
      symbolSize: 5,
      lineStyle: { color: SLATE, width: 1, type: "dotted" },
      itemStyle: { color: SLATE },
      z: 6,
    });
  }

  if (sweep.length > 0) {
    addPsi("Sweep trials");
    series.push({
      name: "Sweep trials",
      type: "scatter",
      symbolSize: 6,
      data: sweep.map((s) => [s[2], s[0]]),
      itemStyle: { color: "#ffffff", borderColor: SLATE, borderWidth: 1.2 },
      z: 6,
    });
  }

  if (duty.totalBpd !== null && duty.headerPsi !== null) {
    addPsi("Optimized");
    series.push({
      name: "Optimized",
      type: "scatter",
      symbol: "diamond",
      symbolSize: 14,
      data: [[duty.totalBpd, duty.headerPsi]],
      itemStyle: { color: CRIMSON, borderColor: "#0f172a", borderWidth: 1 },
      z: 10,
    });
  }

  return houseOption({
    tooltip: {
      ...baseTooltip,
      trigger: "axis",
      formatter: (raw: unknown): string => {
        const list = tipParams(raw);
        const q = tipAxisNum(list);
        const out = q !== null ? [ttHeader(`${fmtNum(q)} BPD total`)] : [];
        return [...out, ...tipRows(list, units)].join("");
      },
    },
    legend: { top: 4, right: 8, itemWidth: 18, textStyle: { fontSize: 12 }, data: legend },
    grid: { ...baseGrid, top: 56 },
    xAxis: { type: "value", ...axis("Total Flow (BPD)", { min: 0 }) },
    yAxis: { type: "value", ...axis("Header pressure (psi)") },
    series,
  });
}

/* ---------------------------------- panel 2 (the machine sheet is panel 3,
 * built by curveChart.machineOption) */

interface Trace {
  heading: string;
  option: EChartsOption;
  /** category x axis - brush zoom does not apply */
  category: boolean;
}

function sweepTrace(sweep: SweepPoint[], duty: Duty): Trace {
  const units: Record<string, UnitSpec> = {
    Oil: { unit: "BOPD", dp: 0 },
    "Total PF": { unit: "BPD", dp: 0 },
    Optimized: { unit: "BOPD", dp: 0 },
  };
  const legend = ["Oil", "Total PF"];

  const series: Record<string, unknown>[] = [
    {
      name: "Oil",
      type: "line",
      yAxisIndex: 0,
      symbolSize: 5,
      data: sweep.map((s) => [s[0], s[1]]),
      lineStyle: { color: ACCENT, width: 2 },
      itemStyle: { color: ACCENT },
      z: 4,
    },
    {
      name: "Total PF",
      type: "line",
      yAxisIndex: 1,
      showSymbol: false,
      data: sweep.map((s) => [s[0], s[2]]),
      lineStyle: { color: SLATE, width: 1.4, type: "dashed" },
      itemStyle: { color: SLATE },
      z: 3,
    },
  ];

  const won = duty.headerPsi !== null ? nearestByX(sweep, duty.headerPsi) : null;
  if (won !== null) {
    legend.push("Optimized");
    series.push({
      name: "Optimized",
      type: "scatter",
      yAxisIndex: 0,
      symbol: "diamond",
      symbolSize: 14,
      data: [[won[0], won[1]]],
      itemStyle: { color: CRIMSON, borderColor: "#0f172a", borderWidth: 1 },
      z: 10,
    });
  }

  return {
    heading: "Pad oil and power fluid vs header pressure",
    category: false,
    option: houseOption({
      tooltip: {
        ...baseTooltip,
        trigger: "axis",
        formatter: (raw: unknown): string => {
          const list = tipParams(raw);
          const psi = tipAxisNum(list);
          const out = psi !== null ? [ttHeader(`${fmtNum(psi)} psi header`)] : [];
          return [...out, ...tipRows(list, units)].join("");
        },
      },
      legend: { top: 4, right: 8, itemWidth: 18, textStyle: { fontSize: 12 }, data: legend },
      grid: { ...baseGrid, top: 56, right: 64 },
      xAxis: { type: "value", ...axis("Header pressure (psi)") },
      yAxis: [
        { type: "value", ...axis("Oil (BOPD)") },
        {
          type: "value",
          position: "right",
          ...axis("Total PF (BPD)"),
          nameGap: 44,
          splitLine: { show: false },
        },
      ],
      series,
    }),
  };
}

function historyTrace(history: HistoryRow[], duty: Duty): Trace {
  const units: Record<string, UnitSpec> = {
    "Trial header": { unit: "psi", dp: 0 },
    "Curve header": { unit: "psi", dp: 0 },
  };

  const series: Record<string, unknown>[] = [
    {
      name: "Trial header",
      type: "line",
      symbolSize: 5,
      data: history.map((h) => h.trialPsi),
      lineStyle: { color: SLATE, width: 1.4, type: "dashed" },
      itemStyle: { color: SLATE },
      z: 3,
    },
    {
      name: "Curve header",
      type: "line",
      symbolSize: 6,
      data: history.map((h) => h.curvePsi),
      lineStyle: { color: ACCENT, width: 2 },
      itemStyle: { color: ACCENT },
      z: 4,
    },
  ];

  if (duty.headerPsi !== null) {
    series.push({
      name: "",
      type: "line",
      data: [],
      silent: true,
      markLine: {
        silent: true,
        symbol: "none",
        animation: false,
        data: [
          {
            yAxis: duty.headerPsi,
            lineStyle: { color: CRIMSON, width: 1.5 },
            label: refLabel("settled", CRIMSON),
          },
        ],
      },
    });
  }

  return {
    heading: "Header fixed-point convergence",
    category: true,
    option: houseOption({
      tooltip: {
        ...baseTooltip,
        trigger: "axis",
        formatter: (raw: unknown): string => {
          const list = tipParams(raw);
          const it = tipAxisText(list);
          const out = it !== "" ? [ttHeader(`Iteration ${it}`)] : [];
          return [...out, ...tipRows(list, units)].join("");
        },
      },
      legend: {
        top: 4,
        right: 8,
        itemWidth: 18,
        textStyle: { fontSize: 12 },
        data: ["Trial header", "Curve header"],
      },
      grid: { ...baseGrid, top: 56 },
      xAxis: {
        type: "category",
        data: history.map((h) => String(h.iter)),
        ...axis("Iteration"),
        splitLine: { show: false },
      },
      yAxis: { type: "value", ...axis("Header pressure (psi)") },
      series,
    }),
  };
}

function traceOption(
  curve: PumpCurveResponse,
  duty: Duty,
  sweep: SweepPoint[],
  history: HistoryRow[],
): Trace | null {
  if (curve.coupling === "free_pressure" && sweep.length > 0) return sweepTrace(sweep, duty);
  if (curve.coupling === "fixed_curve" && history.length > 0) return historyTrace(history, duty);
  return null;
}

/* ------------------------------------------------------------ component */

/** ``nPumps`` is the run form's pumps-online selection: it drives the
 *  pre-run curves so the frontier shown is the bank the run will assume.
 *  Once a result exists its meta n_pumps wins - that is what actually ran.
 *  ``ePad`` is the E-Pad booster configuration the run form holds; it must
 *  ride along or the sheet draws a different booster than the run assumes. */
export function PadCharts({
  pad,
  result,
  nPumps: nPumpsSelected = null,
  ePad,
}: {
  pad: string;
  result: { meta: Record<string, unknown> } | null;
  nPumps?: number | null;
  ePad?: {
    build: EPadBuild;
    suctionPsi: number;
    hzMax: number;
    maxHeaderPsi: number;
  };
}) {
  const meta = result !== null ? result.meta : null;
  const nPumps = meta !== null ? metaNum(meta, "n_pumps") : nPumpsSelected;
  const curve = usePumpCurve(pad, nPumps, ePad).data ?? null;

  const duty = useMemo(() => readDuty(meta, nPumps), [meta, nPumps]);
  const sweep = useMemo(() => readSweep(meta), [meta]);
  const history = useMemo(() => readHistory(meta), [meta]);

  const station = useMemo(
    () => (curve !== null ? stationOption(curve, duty, sweep, history) : null),
    [curve, duty, sweep, history],
  );
  const machines = useMemo(
    () =>
      curve !== null
        ? curve.pumps.map((p) => ({ label: p.label, option: machineOption(p, duty) }))
        : [],
    [curve, duty],
  );
  const trace = useMemo(
    () => (curve !== null ? traceOption(curve, duty, sweep, history) : null),
    [curve, duty, sweep, history],
  );

  // Loading or failed: render nothing rather than a box pretending to be a
  // chart.
  if (curve === null || station === null) return null;
  const np = curve.nameplate;

  return (
    <div className="@container space-y-3">
      {/* Container queries, not viewport ones: these panels sit in a column
          beside the readiness board, so they must pair up on the width they
          actually get, not the width of the window. And two columns only
          when there is a second panel - a lone card in a 2-col grid renders
          half-width beside dead space. */}
      <div className={clsx("grid items-start gap-3", trace !== null && "@4xl:grid-cols-2")}>
        <Card padded={false} className="p-2">
          <p
            className="px-2 pt-1 text-xs font-semibold text-slate-600"
            title={`Delivered header pressure vs total flow at ${np.speed}. Source: ${np.source}. Validated: ${np.validated}.`}
          >
            {curve.pad}-Pad Booster
          </p>
          <p className="px-2 text-[11px] text-slate-500">
            {np.equipment} - {np.model} - {np.arrangement}
          </p>
          <ChartPanel option={station} height={320} zoom={{ xAxisIndex: [0], yAxisIndex: [0] }} />
        </Card>
        {trace !== null && (
          <Card padded={false} className="p-2">
            <p className="px-2 pt-1 text-xs font-semibold text-slate-600">{trace.heading}</p>
            <ChartPanel
              option={trace.option}
              height={320}
              zoom={
                trace.category
                  ? { xAxisIndex: "none", yAxisIndex: "none" }
                  : { xAxisIndex: [0], yAxisIndex: [0] }
              }
            />
          </Card>
        )}
      </div>
      {machines.length > 0 && (
        <div
          className={clsx(
            "grid items-start gap-3",
            machines.length > 1 && "@4xl:grid-cols-2",
          )}
        >
          {machines.map((m) => (
            <Card key={m.label} padded={false} className="p-2">
              <p className="px-2 pt-1 text-xs font-semibold text-slate-600" title={MACHINE_HELP}>
                {m.label}
              </p>
              <ChartPanel
                option={m.option}
                height={300}
                zoom={{ xAxisIndex: [0], yAxisIndex: [0] }}
              />
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
