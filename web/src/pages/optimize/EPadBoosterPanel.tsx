/**
 * E-Pad booster candidate screen.
 *
 * One question: at a required dP across the booster, how much water can each
 * candidate build push into the 3,400 psig E-Pad power-fluid header, inside
 * the vendor recommended operating range, and what amps does that pull?
 *
 * The installed Summit SM25000 26-stage build against the SN35000 18-stage
 * alternative. The engineer sets the dP (or clicks the header button, which
 * sets it from the suction), and optionally caps motor amps; the server solves
 * the SPEED each build needs at every flow - the decision a VFD actually
 * makes - and returns the feasible flow window plus the whole constant-dP
 * locus.
 *
 * Three chart layers:
 *  1. the answer - speed and amps needed across flow, both candidates, with
 *     the feasible window shaded and the amp / speed ceilings drawn;
 *  2. per candidate, the iso-speed dP family with the required dP as a
 *     reference line and the duty point on it - the sheet you check the answer
 *     against;
 *  3. per candidate, the catalog head / BHP / efficiency page.
 *
 * Static physics, no run state: this renders with no saved fits and no
 * optimization run.
 */

import { useMemo, useState } from "react";

import { useEPadBooster } from "../../api/hooks";
import type { EPadBoosterResponse, EPadCandidate, EPadPoint } from "../../api/types";
import { ChartPanel } from "../../charts/ChartPanel";
import type { EChartsOption } from "../../charts/echarts";
import {
  ACCENT,
  axis,
  baseGrid,
  baseTooltip,
  CRIMSON,
  houseOption,
  SLATE,
  ttHeader,
} from "../../charts/theme";
import {
  Badge,
  Card,
  DataTable,
  ErrorNote,
  HelpPopover,
  InfoNote,
  Metric,
  Spinner,
  WarnNote,
  type Column,
} from "../../components/ui";
import { fmtNum } from "../../lib/format";
import { useDebounced } from "../../lib/useDebounced";

import {
  carrierSeries,
  MACHINE_HELP,
  machineOption,
  refLabel,
  tipAxisNum,
  tipParams,
  tipRows,
  xy,
  type UnitSpec,
} from "./curveChart";

const INPUT_CLS =
  "h-8 w-24 rounded-md border border-slate-300 bg-white px-2 text-sm tabular-nums " +
  "text-slate-800 outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-200";

// Installed build vs alternative, one hue each, held across every panel so a
// line never has to be re-identified between charts.
const INSTALLED_COLOR = ACCENT;
const ALTERNATIVE_COLOR = CRIMSON;

/* --------------------------------------------------------------- controls */

interface Duty {
  dpPsid: number;
  suctionPsi: number;
  sg: number;
  condition: number;
  hzMax: number;
  ampsPerBhp: number;
  /** "" = no cap: report amps, enforce nothing */
  ampLimit: string;
}

const SEED: Duty = {
  dpPsid: 600,
  suctionPsi: 2800,
  sg: 1.02,
  condition: 1.0,
  hzMax: 60,
  ampsPerBhp: 0.1435,
  ampLimit: "",
};

interface NumFieldSpec {
  key: keyof Omit<Duty, "ampLimit">;
  label: string;
  min: number;
  max: number;
  step: number;
  help?: string;
}

const FIELDS: NumFieldSpec[] = [
  {
    key: "dpPsid",
    label: "Required dP (psid)",
    min: 50,
    max: 4000,
    step: 25,
    help: "Differential the booster has to make. Discharge = suction + dP.",
  },
  { key: "suctionPsi", label: "Suction (psig)", min: 0, max: 5000, step: 25 },
  {
    key: "sg",
    label: "SG",
    min: 0.9,
    max: 1.3,
    step: 0.01,
    help: "Pumped-fluid specific gravity. Head in feet is SG-independent; only the psi conversion and the shaft power use it.",
  },
  {
    key: "condition",
    label: "Condition",
    min: 0.6,
    max: 1.0,
    step: 0.01,
    help: "Head-only wear derate, the workbooks' Condition cell. 1.00 = as-new. Shaft power stays on the as-new curve, so efficiency falls with it.",
  },
  { key: "hzMax", label: "Max speed (Hz)", min: 30, max: 60, step: 1 },
  {
    key: "ampsPerBhp",
    label: "Amps / BHP",
    min: 0.01,
    max: 20,
    step: 0.001,
    help: "amps = k x shaft BHP. The 0.1435 default is the I-Pad SN35000 live calibration on a 4160 V motor - a transferred estimate, not E-Pad data. Scale by 4160/V for another voltage.",
  },
];

/* ------------------------------------------------- fixed-speed ladder */

interface SpeedRow extends Record<string, unknown> {
  key: string;
  hz: number;
  isDuty: boolean;
  q: number | null;
  rorLo: number;
  rorHi: number;
  pct: number | null;
  bhp: number | null;
  amps: number | null;
  eff: number | null;
  verdict: string;
  inRor: boolean;
  ampOk: boolean;
}

const SPEED_COLUMNS: Column<SpeedRow>[] = [
  {
    key: "hz",
    label: "Run at (Hz)",
    align: "right",
    render: (r) => (
      <span className={r.isDuty ? "font-semibold text-blue-700" : undefined}>
        {fmtNum(r.hz, 1)}
        {r.isDuty && " *"}
      </span>
    ),
  },
  {
    key: "q",
    label: "Flow out (BPD)",
    align: "right",
    help: "Where that speed's curve crosses the required dP - the water you actually pass with the drive pinned there.",
    render: (r) => <span className="font-semibold">{fmtNum(r.q)}</span>,
  },
  {
    key: "rorLo",
    label: "Range at that speed",
    align: "right",
    help: "The vendor recommended range scales with speed, so it moves row to row.",
    render: (r) => `${fmtNum(r.rorLo)} - ${fmtNum(r.rorHi)}`,
  },
  {
    key: "pct",
    label: "% of range top",
    align: "right",
    render: (r) => (
      <span className={r.pct !== null && r.pct > 100 ? "text-red-700" : undefined}>
        {fmtNum(r.pct)}
      </span>
    ),
  },
  { key: "bhp", label: "BHP", align: "right", render: (r) => fmtNum(r.bhp) },
  { key: "amps", label: "Amps", align: "right", render: (r) => fmtNum(r.amps, 1) },
  { key: "eff", label: "Eff (%)", align: "right", render: (r) => fmtNum(r.eff, 1) },
  {
    key: "verdict",
    label: "Verdict",
    render: (r) =>
      r.inRor && r.ampOk ? (
        <Badge tone="good">in range</Badge>
      ) : (
        <Badge tone="poor">{r.verdict}</Badge>
      ),
  },
];

function speedRows(c: EPadCandidate): SpeedRow[] {
  return c.speed_table.map((r) => ({
    key: `${c.nameplate.key}-${r.hz}`,
    hz: r.hz,
    isDuty: r.is_duty,
    q: r.q_bpd,
    rorLo: r.ror_lo,
    rorHi: r.ror_hi,
    pct: r.pct_of_ror_hi,
    bhp: r.bhp,
    amps: r.amps,
    eff: r.eff_pct,
    verdict: r.blocked_by ?? "in range",
    inRor: r.in_ror,
    ampOk: r.amp_ok,
  }));
}

/* ----------------------------------------------------------------- table */

interface Row extends Record<string, unknown> {
  key: string;
  build: string;
  installed: boolean;
  q: number | null;
  hz: number | null;
  bhp: number | null;
  amps: number | null;
  headroom: number | null;
  eff: number | null;
  pctBep: number | null;
  windowLo: number | null;
  windowHi: number | null;
  limit: string;
  reason: string | null;
}

const COLUMNS: Column<Row>[] = [
  {
    key: "build",
    label: "Build",
    width: "16rem",
    render: (r) => (
      <span className="flex items-center gap-1.5">
        <span
          className="inline-block h-2.5 w-2.5 rounded-full"
          style={{ background: r.installed ? INSTALLED_COLOR : ALTERNATIVE_COLOR }}
        />
        {r.build}
      </span>
    ),
  },
  {
    key: "q",
    label: "Deliverable (BPD)",
    align: "right",
    help: "Most water this build can move at the required dP while staying inside the recommended range and under the amp cap.",
    render: (r) => <span className="font-semibold">{fmtNum(r.q)}</span>,
  },
  {
    key: "hz",
    label: "Speed (Hz)",
    align: "right",
    render: (r) => fmtNum(r.hz, 1),
  },
  { key: "bhp", label: "BHP", align: "right", render: (r) => fmtNum(r.bhp) },
  { key: "amps", label: "Amps", align: "right", render: (r) => fmtNum(r.amps, 1) },
  {
    key: "headroom",
    label: "Amp headroom",
    align: "right",
    help: "Motor limit minus amps at the duty point. Blank when no limit is set.",
    render: (r) => fmtNum(r.headroom, 1),
  },
  { key: "eff", label: "Eff (%)", align: "right", render: (r) => fmtNum(r.eff, 1) },
  {
    key: "pctBep",
    label: "% of BEP",
    align: "right",
    help: "Duty flow over the best-efficiency flow at that speed.",
    render: (r) => fmtNum(r.pctBep),
  },
  {
    key: "windowLo",
    label: "Flow window (BPD)",
    align: "right",
    help: "Every flow between these two holds the required dP inside the recommended range and under the amp cap. The low end is the turndown.",
    render: (r) =>
      r.windowLo === null || r.windowHi === null
        ? "-"
        : `${fmtNum(r.windowLo)} - ${fmtNum(r.windowHi)}`,
  },
  {
    key: "limit",
    label: "Capped by",
    render: (r) =>
      r.reason !== null ? (
        <Badge tone="poor" title={r.reason}>
          no duty
        </Badge>
      ) : (
        r.limit
      ),
  },
];

/* ---------------------------------------------------------------- panel 1 */

/** [flow, value] pairs off the locus, skipping flows with no solution. */
function locusXy(locus: EPadPoint[], pick: (p: EPadPoint) => number | null): [number, number][] {
  const out: [number, number][] = [];
  for (const p of locus) {
    const v = pick(p);
    if (v !== null && Number.isFinite(v)) out.push([p.q_bpd, v]);
  }
  return out;
}

// Window shading, one faint hue per build: with two candidates a single
// shared fill composites into a third shade where they overlap and stops
// telling you whose window is whose.
const WINDOW_FILL_INSTALLED = "rgba(37,99,235,0.09)";
const WINDOW_FILL_ALTERNATIVE = "rgba(201,37,45,0.07)";

/**
 * The answer chart: the speed each build needs to hold the required dP across
 * flow, and the amps that speed pulls. Speed on the left axis, amps on the
 * right; the feasible flow window is shaded per build and the ceilings the
 * engineer set (max speed, motor amps) are drawn as reference lines.
 */
function dutyOption(rep: EPadBoosterResponse): EChartsOption {
  const series: Record<string, unknown>[] = [];
  const units: Record<string, UnitSpec> = {};
  const legend: string[] = [];
  const areas: Record<string, unknown>[][] = [];
  const lines: Record<string, unknown>[] = [];

  for (const c of rep.candidates) {
    const color = c.nameplate.installed ? INSTALLED_COLOR : ALTERNATIVE_COLOR;
    const fill = c.nameplate.installed ? WINDOW_FILL_INSTALLED : WINDOW_FILL_ALTERNATIVE;
    const hzName = `${c.nameplate.label} - speed`;
    const ampName = `${c.nameplate.label} - amps`;
    legend.push(hzName, ampName);
    units[hzName] = { unit: "Hz", dp: 1 };
    units[ampName] = { unit: "A", dp: 1 };

    series.push({
      name: hzName,
      type: "line",
      yAxisIndex: 0,
      showSymbol: false,
      data: locusXy(c.locus, (p) => p.hz),
      lineStyle: { color, width: 2.2 },
      itemStyle: { color },
      z: 5,
    });
    series.push({
      name: ampName,
      type: "line",
      yAxisIndex: 1,
      showSymbol: false,
      data: locusXy(c.locus, (p) => p.amps),
      lineStyle: { color, width: 1.6, type: "dashed" },
      itemStyle: { color },
      z: 4,
    });

    if (c.window !== null) {
      areas.push([
        { xAxis: c.window[0], itemStyle: { color: fill } },
        { xAxis: c.window[1] },
      ]);
      lines.push({
        xAxis: c.window[1],
        lineStyle: { color, width: 1.5 },
        label: refLabel(`${fmtNum(c.window[1])} BPD`, color),
      });
    }
  }

  lines.push({
    yAxis: rep.target.hz_max,
    lineStyle: { color: SLATE, width: 1, type: "dashed" },
    label: refLabel(`${fmtNum(rep.target.hz_max)} Hz max`, SLATE),
  });

  // A markLine resolves `yAxis` against the axis of the series that CARRIES
  // it - a per-item yAxisIndex is ignored. So the amp limit needs its own
  // carrier bound to the amps axis; on the speed carrier a 60 A limit would
  // draw at 60 Hz, i.e. pinned to the top of the wrong scale.
  series.push({
    name: "",
    type: "line",
    yAxisIndex: 0,
    data: [],
    silent: true,
    markArea: { silent: true, animation: false, data: areas },
    markLine: { silent: true, symbol: "none", animation: false, data: lines },
  });
  if (rep.target.amp_limit_a !== null) {
    series.push({
      name: "",
      type: "line",
      yAxisIndex: 1,
      data: [],
      silent: true,
      markLine: {
        silent: true,
        symbol: "none",
        animation: false,
        data: [
          {
            yAxis: rep.target.amp_limit_a,
            lineStyle: { color: CRIMSON, width: 1.4, type: "dashed" },
            label: refLabel(`${fmtNum(rep.target.amp_limit_a)} A limit`, CRIMSON),
          },
        ],
      },
    });
  }

  return houseOption({
    tooltip: {
      ...baseTooltip,
      trigger: "axis",
      formatter: (raw: unknown): string => {
        const list = tipParams(raw);
        const q = tipAxisNum(list);
        const head = q !== null ? [ttHeader(`${fmtNum(q)} BPD`)] : [];
        return [...head, ...tipRows(list, units)].join("");
      },
    },
    legend: { top: 4, left: 8, itemWidth: 18, textStyle: { fontSize: 11 }, data: legend },
    grid: { ...baseGrid, top: 74, right: 72 },
    xAxis: { type: "value", ...axis("Flow (BPD)", { min: 0 }) },
    yAxis: [
      { type: "value", ...axis("Speed to hold dP (Hz)", { min: 0 }) },
      {
        type: "value",
        position: "right",
        ...axis("Amps (A)", { min: 0 }),
        nameGap: 40,
        splitLine: { show: false },
      },
    ],
    series,
  });
}

/* ---------------------------------------------------------------- panel 2 */

/**
 * The iso-speed dP family for one build, with the required dP as a horizontal
 * reference and the duty point sitting on it. The shaded band is the vendor
 * recommended range AT 60 Hz - the range the duty actually respects moves with
 * speed, so the crimson duty line is the number to read, not the band edge.
 */
function familyOption(c: EPadCandidate, dpTarget: number): EChartsOption {
  const series: Record<string, unknown>[] = [];
  const units: Record<string, UnitSpec> = {};
  const legend: string[] = [];
  const topHz = c.curves.length > 0 ? c.curves[c.curves.length - 1].hz : 60;
  // Keep the required-dP reference line on scale even when no speed reaches
  // it: a markLine past the data range does not stretch the axis, so an
  // impossible duty would silently vanish and the sheet would look like it
  // agreed with the ask.
  let peak = 0;
  for (const line of c.curves) {
    for (const p of line.points) if (p[1] > peak) peak = p[1];
  }
  const yMax = dpTarget > peak ? Math.ceil((dpTarget * 1.05) / 100) * 100 : undefined;

  for (const line of c.curves) {
    const active = line.hz === topHz;
    legend.push(line.label);
    units[line.label] = { unit: "psid", dp: 0 };
    series.push({
      name: line.label,
      type: "line",
      showSymbol: false,
      data: xy(line.points, 1),
      lineStyle: active
        ? { color: ACCENT, width: 2.4 }
        : { color: SLATE, width: 1.2, type: "dashed" },
      itemStyle: { color: active ? ACCENT : SLATE },
      z: active ? 5 : 3,
    });
  }

  if (c.duty !== null) {
    const name = "Duty";
    legend.push(name);
    units[name] = { unit: "psid", dp: 0 };
    series.push({
      name,
      type: "scatter",
      symbol: "diamond",
      symbolSize: 14,
      data: [[c.duty.q_bpd, dpTarget]],
      itemStyle: { color: CRIMSON, borderColor: "#0f172a", borderWidth: 1 },
      z: 10,
    });
  }

  series.push(
    carrierSeries({
      aor: c.ror_60hz,
      por: null,
      bep: c.bep_60hz,
      minFlow: null,
      cap: null,
      duty: c.duty !== null ? c.duty.q_bpd : null,
    }),
  );
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
          yAxis: dpTarget,
          lineStyle: { color: CRIMSON, width: 1.4, type: "dashed" },
          label: refLabel(`${fmtNum(dpTarget)} psid required`, CRIMSON),
        },
      ],
    },
  });

  return houseOption({
    tooltip: {
      ...baseTooltip,
      trigger: "axis",
      formatter: (raw: unknown): string => {
        const list = tipParams(raw);
        const q = tipAxisNum(list);
        const head = q !== null ? [ttHeader(`${fmtNum(q)} BPD`)] : [];
        return [...head, ...tipRows(list, units)].join("");
      },
    },
    legend: { top: 4, right: 8, itemWidth: 18, textStyle: { fontSize: 11 }, data: legend },
    grid: { ...baseGrid, top: 56 },
    xAxis: { type: "value", ...axis("Flow (BPD)", { min: 0 }) },
    yAxis: { type: "value", ...axis("Differential (psid)", { min: 0, max: yMax }) },
    series,
  });
}

/* ------------------------------------------------------------- component */

const DUTY_HELP =
  "Speed each build needs to hold the required dP at every flow (solid, left axis) " +
  "and the amps that speed pulls (dashed, right axis). Shaded spans are the flows " +
  "where the build stays inside its recommended operating range and under the amp " +
  "cap; the vertical line on each is the most water it can deliver at this dP.";

const FAMILY_HELP =
  "Vendor curve sheet, differential vs flow at each speed. The dashed crimson line " +
  "is the dP asked for; the diamond is the duty point. Shading is the recommended " +
  "operating range as the catalog states it at 60 Hz - the range the solve respects " +
  "scales with speed, so it is tighter than the band at the reduced speeds below.";

export function EPadBoosterPanel() {
  const [duty, setDuty] = useState<Duty>(SEED);
  const debounced = useDebounced(duty, 350);

  const ampLimit = Number(debounced.ampLimit);
  const request = useMemo(
    () => ({
      dp_psid: debounced.dpPsid,
      suction_psi: debounced.suctionPsi,
      sg: debounced.sg,
      condition: debounced.condition,
      hz_max: debounced.hzMax,
      amps_per_bhp: debounced.ampsPerBhp,
      amp_limit_a:
        debounced.ampLimit.trim() === "" || !Number.isFinite(ampLimit) || ampLimit <= 0
          ? null
          : ampLimit,
    }),
    [debounced, ampLimit],
  );

  const query = useEPadBooster(request);
  const rep = query.data ?? null;

  const rows: Row[] = useMemo(() => {
    if (rep === null) return [];
    return rep.candidates.map((c) => ({
      key: c.nameplate.key,
      build: c.nameplate.label,
      installed: c.nameplate.installed,
      q: c.duty?.q_bpd ?? null,
      hz: c.duty?.hz ?? null,
      bhp: c.duty?.bhp ?? null,
      amps: c.duty?.amps ?? null,
      headroom: c.duty?.amp_headroom_a ?? null,
      eff: c.duty?.eff_pct ?? null,
      pctBep: c.duty?.pct_of_bep ?? null,
      windowLo: c.window?.[0] ?? null,
      windowHi: c.window?.[1] ?? null,
      limit: c.limited_by,
      reason: c.infeasible_reason,
    }));
  }, [rep]);

  const blocked = rep === null ? [] : rep.candidates.filter((c) => c.duty === null);

  const dischargePsi = duty.suctionPsi + duty.dpPsid;
  const headerPsi = rep?.target.header_default_psi ?? 3400;
  const onHeader = Math.abs(dischargePsi - headerPsi) < 0.5;

  return (
    <div className="@container space-y-4">
      <div>
        <h2 className="text-sm font-semibold tracking-tight text-slate-700">
          E-Pad booster - candidate capability at a required dP
        </h2>
        <p className="text-xs text-slate-500">
          The Summit SM25000 26-stage build in the well against the SN35000 18-stage
          alternative. Static catalog physics: no saved fits and no optimization run needed.
        </p>
      </div>

      <Card>
        <div className="flex flex-wrap items-end gap-x-4 gap-y-3">
          {FIELDS.map((f) => (
            <label key={f.key} className="block" title={f.help}>
              <span className="text-xs font-medium text-slate-500">{f.label}</span>
              <input
                type="number"
                value={duty[f.key]}
                min={f.min}
                max={f.max}
                step={f.step}
                onChange={(e) =>
                  setDuty((d) => ({ ...d, [f.key]: Number(e.target.value) }))
                }
                className={`${INPUT_CLS} mt-1 block`}
              />
            </label>
          ))}
          <label
            className="block"
            title="Motor current cap. Leave blank to report amps without enforcing anything - no E-Pad motor nameplate came with the vendor curve sheets, so nothing is assumed."
          >
            <span className="text-xs font-medium text-slate-500">Amp limit (A)</span>
            <input
              type="number"
              value={duty.ampLimit}
              min={1}
              max={5000}
              step={1}
              placeholder="none"
              onChange={(e) => setDuty((d) => ({ ...d, ampLimit: e.target.value }))}
              className={`${INPUT_CLS} mt-1 block`}
            />
          </label>

          <Metric
            label="Discharge"
            value={`${fmtNum(dischargePsi)} psig`}
            sub={`${fmtNum(duty.suctionPsi)} suction + ${fmtNum(duty.dpPsid)} psid`}
            tone={onHeader ? "good" : "neutral"}
            title="Suction plus the required dP - the pressure the booster would actually deliver at."
          />
          <button
            type="button"
            disabled={onHeader}
            onClick={() =>
              setDuty((d) => ({ ...d, dpPsid: Math.max(50, headerPsi - d.suctionPsi) }))
            }
            className="h-8 rounded-md bg-blue-600 px-3 text-sm font-medium text-white transition-colors hover:bg-blue-700 disabled:bg-slate-200 disabled:text-slate-500"
          >
            {onHeader
              ? `On the ${fmtNum(headerPsi)} psi header`
              : `Set dP for ${fmtNum(headerPsi)} psi`}
          </button>
        </div>
      </Card>

      {query.isError && <ErrorNote error={query.error} />}
      {rep === null && !query.isError && <Spinner label="Solving booster capability" />}

      {rep !== null && (
        <>
          <Card padded={false} className="p-2">
            <div className="flex items-center gap-2 px-2 pt-1">
              <p className="text-xs font-semibold text-slate-600">
                Deliverable at {fmtNum(rep.target.discharge_psi)} psig
                {rep.target.amp_limit_a !== null &&
                  ` under ${fmtNum(rep.target.amp_limit_a)} A`}
              </p>
              <HelpPopover label="how this is solved" title="Required-dP capability">
                <p>
                  At a fixed dP the speed a centrifugal booster needs rises with flow, so
                  each build&apos;s answer is a flow WINDOW, not a single point. Below the
                  window the pump is under its recommended range - too slow to hold the dP
                  at that little flow. Above it, the recommended range, the amp cap, or the
                  60 Hz capability wall cuts in, and the &quot;capped by&quot; column says
                  which.
                </p>
                <p className="mt-2">{rep.notes.stage_table}</p>
              </HelpPopover>
            </div>
            <DataTable columns={COLUMNS} rows={rows} rowKey={(r) => r.key} maxHeight="12rem" />
          </Card>

          <div className="grid items-start gap-3 @4xl:grid-cols-2">
            {rep.candidates.map((c) => (
              <Card key={`${c.nameplate.key}-speed`} padded={false} className="p-2">
                <div className="flex items-center gap-2 px-2 pt-1">
                  <p className="text-xs font-semibold text-slate-600">
                    {c.nameplate.label} - pin the drive at a speed
                  </p>
                  <HelpPopover label="two ways to run it" title="Slow down, or run flat out and choke">
                    <p>
                      The ladder is the direct answer to &quot;I set the drive to X Hz, what
                      comes out?&quot; - the flow where that speed&apos;s curve crosses the
                      required dP. The recommended range scales WITH speed, so it moves row
                      to row, and above the duty speed the crossing flow runs off the right
                      end of the range faster than the range grows. That is what caps the
                      deliverable rate.
                    </p>
                    <p className="mt-2">
                      Which leaves two real operating policies. SLOW the drive until the
                      pump makes exactly the dP asked for - the starred row, in range, no
                      wasted pressure, least water. Or run at the speed cap, pass the most
                      the range allows there, and burn the surplus across a choke - more
                      water, more shaft power, a throttling loss. Both are priced below.
                    </p>
                  </HelpPopover>
                </div>
                <DataTable
                  columns={SPEED_COLUMNS}
                  rows={speedRows(c)}
                  rowKey={(r) => r.key}
                  highlightRow={(r) => r.isDuty}
                  maxHeight="16rem"
                />
                {c.duty !== null && (
                  <div className="mt-2 grid gap-2 px-1 pb-1 sm:grid-cols-2">
                    <div className="rounded-md border border-slate-200 bg-white px-3 py-2">
                      <p className="text-[11px] font-semibold tracking-wide text-slate-500 uppercase">
                        Slow to match ({fmtNum(c.duty.hz, 1)} Hz)
                      </p>
                      <p className="mt-0.5 text-lg font-semibold tabular-nums text-slate-800">
                        {fmtNum(c.duty.q_bpd)} BPD
                      </p>
                      <p className="text-[11px] text-slate-500">
                        {fmtNum(c.duty.bhp)} BHP - {fmtNum(c.duty.amps, 1)} A - eff{" "}
                        {fmtNum(c.duty.eff_pct, 1)}% - no throttling loss
                      </p>
                    </div>
                    {c.throttled !== null ? (
                      <div className="rounded-md border border-slate-200 bg-white px-3 py-2">
                        <p className="text-[11px] font-semibold tracking-wide text-slate-500 uppercase">
                          Flat out + choke ({fmtNum(c.throttled.hz, 0)} Hz)
                        </p>
                        <p className="mt-0.5 text-lg font-semibold tabular-nums text-slate-800">
                          {fmtNum(c.throttled.q_bpd)} BPD
                          <span className="ml-1.5 text-xs font-medium text-green-700">
                            +{fmtNum(c.throttled.q_bpd - c.duty.q_bpd)}
                          </span>
                        </p>
                        <p className="text-[11px] text-slate-500">
                          {fmtNum(c.throttled.bhp)} BHP - {fmtNum(c.throttled.amps, 1)} A -
                          makes {fmtNum(c.throttled.dp_made_psid)} psid, so{" "}
                          {fmtNum(c.throttled.throttle_psid)} psi choked off (
                          {fmtNum(c.throttled.throttle_hhp)} hhp burned)
                        </p>
                      </div>
                    ) : (
                      <div className="rounded-md border border-slate-200 bg-slate-50 px-3 py-2">
                        <p className="text-[11px] font-semibold tracking-wide text-slate-500 uppercase">
                          Flat out + choke
                        </p>
                        <p className="mt-0.5 text-sm text-slate-500">
                          Not available: at {fmtNum(rep.target.hz_max, 0)} Hz the range
                          ceiling flow already makes less than the required dP.
                        </p>
                      </div>
                    )}
                  </div>
                )}
              </Card>
            ))}
          </div>

          {blocked.length > 0 && (
            <WarnNote>
              <p className="font-medium">
                {blocked.length === rep.candidates.length
                  ? "Neither build can hold this duty."
                  : "One build cannot hold this duty."}
              </p>
              <ul className="mt-1 list-disc space-y-0.5 pl-5">
                {blocked.map((c) => (
                  <li key={c.nameplate.key}>
                    {c.nameplate.label}: {c.infeasible_reason}
                  </li>
                ))}
              </ul>
              <p className="mt-1">
                The differential-vs-flow sheets below still draw each build&apos;s speed
                family, so you can see how far short it falls.
              </p>
            </WarnNote>
          )}

          {/* No locus anywhere means no line to draw: an empty axis frame reads
              as a broken chart, and the reasons above say more than it could. */}
          {rep.candidates.some((c) => c.locus.length > 0) && (
            <Card padded={false} className="p-2">
              <p className="px-2 pt-1 text-xs font-semibold text-slate-600" title={DUTY_HELP}>
                Speed and amps to hold {fmtNum(rep.target.dp_psid)} psid
              </p>
              <ChartPanel
                option={dutyOption(rep)}
                height={340}
                zoom={{ xAxisIndex: [0], yAxisIndex: [0] }}
              />
            </Card>
          )}

          <div className="grid items-start gap-3 @4xl:grid-cols-2">
            {rep.candidates.map((c) => (
              <Card key={c.nameplate.key} padded={false} className="p-2">
                <p
                  className="px-2 pt-1 text-xs font-semibold text-slate-600"
                  title={FAMILY_HELP}
                >
                  {c.nameplate.label} - differential vs flow
                </p>
                <p className="px-2 text-[11px] text-slate-500">
                  {c.nameplate.model} - {c.nameplate.series_housing} - {c.nameplate.n_stages}{" "}
                  stg - BEP {fmtNum(c.bep_60hz)} BPD - range {fmtNum(c.ror_60hz[0])} to{" "}
                  {fmtNum(c.ror_60hz[1])} BPD at 60 Hz
                </p>
                <ChartPanel
                  option={familyOption(c, rep.target.dp_psid)}
                  height={300}
                  zoom={{ xAxisIndex: [0], yAxisIndex: [0] }}
                />
              </Card>
            ))}
          </div>

          <div className="grid items-start gap-3 @4xl:grid-cols-2">
            {rep.candidates.map((c) => (
              <Card key={`${c.nameplate.key}-machine`} padded={false} className="p-2">
                <p
                  className="px-2 pt-1 text-xs font-semibold text-slate-600"
                  title={MACHINE_HELP}
                >
                  {c.machine.label}
                </p>
                <ChartPanel
                  option={machineOption(c.machine, {
                    headerPsi: null,
                    totalBpd: null,
                    perPumpBpd: c.duty !== null ? c.duty.q_bpd : null,
                  })}
                  height={300}
                  zoom={{ xAxisIndex: [0], yAxisIndex: [0] }}
                />
              </Card>
            ))}
          </div>

          <WarnNote>
            <p className="font-medium">Housing pressure rating is not modeled.</p>
            <p className="mt-1">{rep.notes.housing_pressure}</p>
          </WarnNote>

          <InfoNote>
            <p className="font-medium">Amps are an estimate, for trend not protection.</p>
            <p className="mt-1">{rep.notes.amps}</p>
            <p className="mt-2 font-medium">Also not enforced</p>
            <ul className="mt-1 list-disc space-y-0.5 pl-5">
              {rep.notes.not_enforced.map((n) => (
                <li key={n}>{n}</li>
              ))}
            </ul>
            <p className="mt-2 font-medium">Curve provenance</p>
            <ul className="mt-1 list-disc space-y-0.5 pl-5">
              {rep.candidates.map((c) => (
                <li key={c.nameplate.key}>
                  {c.nameplate.label}: {c.nameplate.source}
                </li>
              ))}
            </ul>
          </InfoNote>
        </>
      )}
    </div>
  );
}
