/**
 * Choke-plan visuals: the per-well dumbbell chart (where each well moves
 * between full open and the plan) and the IPR landing table (where each
 * well ends up on its inflow curve at max PF vs the choked setting).
 *
 * Chart rule: mounted through ChartPanel only, SVG renderer, tooltips via
 * theme helpers. No custom-series renderItem - the dumbbell connectors are
 * plain 2-point line series, which relayout correctly under dataZoom.
 */

import clsx from "clsx";
import { useMemo, useState } from "react";

import type { ChokePlanRow } from "../../api/types";
import { ChartPanel } from "../../charts/ChartPanel";
import type { EChartsOption } from "../../charts/echarts";
import {
  ACCENT,
  AXIS_LINE,
  SLATE,
  TEXT,
  axis,
  houseOption,
  ttHeader,
  ttRow,
} from "../../charts/theme";
import { Card } from "../../components/ui";
import { fmtNum } from "../../lib/format";

type Metric = "oil" | "pf" | "psi";
type Baseline = "full" | "today";

const METRIC_LABEL: Record<Metric, string> = {
  oil: "Oil (BOPD)",
  pf: "PF rate (BPD)",
  psi: "PF pressure delivered (psi)",
};

const BASELINE_LABEL: Record<Baseline, string> = {
  full: "Full open (model)",
  today: "Today (test)",
};

/** One dumbbell row: baseline and plan values for the selected metric, or
 *  null when that well has nothing comparable under this view. */
function pick(r: ChokePlanRow, metric: Metric, baseline: Baseline): [number, number] | null {
  if (r.action === "excluded") return null;
  if (baseline === "full") {
    // model-vs-model: matches the plan table's dPF / dOil deltas
    if (r.basis !== "model") return null;
    if (metric === "oil") return r.oil_full != null && r.oil != null ? [r.oil_full, r.oil] : null;
    if (metric === "pf") return r.pf_full != null && r.pf != null ? [r.pf_full, r.pf] : null;
    // pressure: a shut well delivers nothing - no meaningful second dot
    return r.delivered_full_psi != null && r.delivered_psi != null
      ? [r.delivered_full_psi, r.delivered_psi]
      : null;
  }
  // today: measured test as the first dot
  if (metric === "oil")
    return r.test_oil != null && r.projected_oil != null ? [r.test_oil, r.projected_oil] : null;
  if (metric === "pf") return r.test_pf != null && r.pf != null ? [r.test_pf, r.pf] : null;
  return null; // tests do not record delivered PF pressure
}

function dumbbellOption(
  rows: { well: string; from: number; to: number }[],
  metric: Metric,
  baseline: Baseline,
): EChartsOption {
  // first plan row (biggest action) at the TOP of the chart
  const names = rows.map((r) => r.well).reverse();
  const byWell = new Map(rows.map((r) => [r.well, r]));

  const connectors = rows.map((r) => ({
    type: "line" as const,
    silent: true,
    showSymbol: false,
    lineStyle: { color: AXIS_LINE, width: 1.5 },
    tooltip: { show: false },
    data: [
      [r.from, r.well],
      [r.to, r.well],
    ],
  }));

  const tip = (params: unknown): string => {
    const p = params as { value?: [number, string] };
    const well = p.value?.[1] ?? "";
    const r = byWell.get(well);
    if (!r) return well;
    return (
      ttHeader(well) +
      ttRow(SLATE, BASELINE_LABEL[baseline], fmtNum(r.from)) +
      ttRow(ACCENT, "Plan", fmtNum(r.to)) +
      ttRow(r.to >= r.from ? "#2E7D32" : "#c9252d", "Delta", fmtNum(r.to - r.from))
    );
  };

  return houseOption({
    tooltip: { trigger: "item", formatter: tip },
    legend: { top: 0, textStyle: { color: TEXT, fontSize: 11 } },
    xAxis: axis(METRIC_LABEL[metric]),
    yAxis: {
      type: "category",
      data: names,
      axisLabel: { color: TEXT, fontSize: 11 },
      axisLine: { lineStyle: { color: AXIS_LINE } },
      axisTick: { show: false },
    },
    series: [
      ...connectors,
      {
        name: BASELINE_LABEL[baseline],
        type: "scatter",
        symbolSize: 11,
        itemStyle: { color: "#ffffff", borderColor: SLATE, borderWidth: 1.5 },
        data: rows.map((r) => [r.from, r.well]),
      },
      {
        name: "Plan",
        type: "scatter",
        symbolSize: 7,
        itemStyle: { color: ACCENT },
        data: rows.map((r) => [r.to, r.well]),
      },
    ],
  } as EChartsOption);
}

function Seg<T extends string>({
  value,
  options,
  onChange,
  disabled,
}: {
  value: T;
  options: { id: T; label: string; title?: string; off?: boolean }[];
  onChange: (v: T) => void;
  disabled?: boolean;
}) {
  return (
    <div className="flex gap-1">
      {options.map((o) => (
        <button
          key={o.id}
          type="button"
          disabled={disabled || o.off}
          title={o.title}
          onClick={() => onChange(o.id)}
          className={clsx(
            "rounded px-1.5 py-0.5 text-xs font-medium transition-colors",
            o.id === value
              ? "bg-blue-600 text-white"
              : "bg-white text-slate-500 ring-1 ring-slate-200 hover:bg-slate-50",
            (disabled || o.off) && "cursor-not-allowed opacity-40",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

/** Per-well movement between a baseline and the plan, one dumbbell per well.
 *  Metric switch: oil / PF rate / delivered PF pressure. Baseline switch:
 *  model full-open (matches the table deltas) or today's measured test. */
export function ChokeDumbbell({ plan }: { plan: ChokePlanRow[] }) {
  const [metric, setMetric] = useState<Metric>("oil");
  const [baseline, setBaseline] = useState<Baseline>("full");

  const rows = useMemo(
    () =>
      plan
        .map((r) => {
          const v = pick(r, metric, baseline);
          return v === null ? null : { well: r.well, from: v[0], to: v[1] };
        })
        .filter((r): r is { well: string; from: number; to: number } => r !== null),
    [plan, metric, baseline],
  );

  const option = useMemo(() => dumbbellOption(rows, metric, baseline), [rows, metric, baseline]);
  const skipped = plan.filter((r) => r.action !== "excluded").length - rows.length;

  return (
    <Card padded={false} className="p-2">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 px-2 pt-1">
        <p className="text-xs font-semibold text-slate-600">
          Per-well movement: {BASELINE_LABEL[baseline]} to plan
        </p>
        <Seg
          value={metric}
          onChange={setMetric}
          options={[
            { id: "oil", label: "Oil" },
            { id: "pf", label: "PF rate" },
            {
              id: "psi",
              label: "PF pressure",
              off: baseline === "today",
              title:
                baseline === "today"
                  ? "Well tests do not record delivered PF pressure - switch the baseline to Full open."
                  : "Delivered PF pressure at the wellhead (psi)",
            },
          ]}
        />
        <Seg
          value={baseline}
          onChange={(b) => {
            setBaseline(b);
            if (b === "today" && metric === "psi") setMetric("oil");
          }}
          options={[
            {
              id: "full",
              label: "vs full open",
              title: "Model at full open vs the choked plan - matches the table's dPF/dOil.",
            },
            {
              id: "today",
              label: "vs today",
              title: "Latest measured test vs the test-anchored projection (model bias cancels).",
            },
          ]}
        />
      </div>
      {rows.length > 0 ? (
        <ChartPanel
          option={option}
          height={90 + rows.length * 26}
          zoom={{ xAxisIndex: [0], yAxisIndex: "none" }}
        />
      ) : (
        <p className="px-2 py-4 text-xs text-slate-500">
          Nothing comparable under this view (no wells carry both values).
        </p>
      )}
      {skipped > 0 && rows.length > 0 && (
        <p className="px-2 pb-1 text-[11px] text-slate-400">
          {skipped} well{skipped === 1 ? "" : "s"} without both values under this view omitted.
        </p>
      )}
    </Card>
  );
}

/* ---------------------------------------------------------- IPR landing */

const TH = "px-2 py-1.5 text-right font-semibold";
const TD = "px-2 py-1 text-right tabular-nums";

/** Where each well ends up on its inflow curve: suction pressure (the pump's
 *  flowing BHP) and drawdown at max PF (full open at the plan header) vs the
 *  choked setting. Model-basis wells only - held/excluded wells have no
 *  solved suction pressure. */
export function IprLandingTable({ plan }: { plan: ChokePlanRow[] }) {
  const rows = plan.filter((r) => r.basis === "model");
  if (rows.length === 0) return null;
  return (
    <Card padded={false} className="overflow-x-auto">
      <p className="px-2 pt-2 text-xs font-semibold text-slate-600">
        IPR landing - where each well sits on its inflow curve
      </p>
      <p className="px-2 pb-1 text-[11px] text-slate-500">
        Suction pressure is the flowing BHP at the pump; drawdown = reservoir P minus suction.
        Choking backs the well up its IPR: suction rises, drawdown and oil drop. An asterisk
        marks a suction pinned at the cavitation floor (sonic throat entry) - there the choke
        only sheds power fluid, and suction and oil hold until the well leaves sonic.
        Wells marked field use measured suction response mined from PF-pressure history -
        their modeled floors were contradicted by measured BHPs.
      </p>
      <table className="w-full border-collapse text-[13px]">
        <thead>
          <tr className="border-b border-slate-200 bg-slate-50 text-slate-600">
            <th rowSpan={2} className="px-2 py-1.5 text-left font-semibold align-bottom">
              Well
            </th>
            <th rowSpan={2} className={clsx(TH, "align-bottom")} title="Reservoir pressure from the saved fit">
              Res P (psi)
            </th>
            <th colSpan={3} className="border-l border-slate-200 px-2 py-1 text-center font-semibold">
              Max PF (full open @ header)
            </th>
            <th colSpan={3} className="border-l border-slate-200 px-2 py-1 text-center font-semibold">
              Choked setting
            </th>
            <th
              rowSpan={2}
              className={clsx(TH, "border-l border-slate-200 align-bottom")}
              title="How far the choke backs the well up its IPR: suction rise, psi"
            >
              Suction rise (psi)
            </th>
          </tr>
          <tr className="border-b border-slate-200 bg-slate-50 text-slate-600">
            <th className={clsx(TH, "border-l border-slate-200")}>Suction (psi)</th>
            <th className={TH}>Drawdown (psi)</th>
            <th className={TH}>Oil (BOPD)</th>
            <th className={clsx(TH, "border-l border-slate-200")}>Suction (psi)</th>
            <th className={TH}>Drawdown (psi)</th>
            <th className={TH}>Oil (BOPD)</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const shut = r.action === "shut";
            const rise =
              r.psu != null && r.psu_full != null && !shut ? r.psu - r.psu_full : null;
            const fieldBasis = r.suction_basis === "evidence";
            const fieldTitle = fieldBasis
              ? `Suction response from field data: beta ${
                  r.response_beta != null ? r.response_beta.toFixed(2) : "?"
                } (${r.beta_source ?? "?"}), floor ${fmtNum(
                  r.evidence_floor_psi,
                )} psi measured vs model (violation +${fmtNum(r.floor_violation_psi)} psi)`
              : undefined;
            return (
              <tr key={r.well} className="border-b border-slate-100 last:border-b-0">
                <td className="px-2 py-1 text-left font-medium text-slate-700">{r.well}</td>
                <td className={clsx(TD, "text-slate-500")}>{fmtNum(r.res_pres)}</td>
                <td
                  className={clsx(TD, "border-l border-slate-100 text-slate-700")}
                  title={
                    fieldBasis
                      ? fieldTitle
                      : r.sonic_full
                        ? "Pinned at the cavitation floor (sonic throat entry)"
                        : undefined
                  }
                >
                  {fmtNum(r.psu_full)}
                  {fieldBasis ? (
                    <span className="text-[10px] text-slate-400"> field</span>
                  ) : r.sonic_full ? (
                    <span className="text-slate-400"> *</span>
                  ) : null}
                </td>
                <td className={clsx(TD, "text-slate-600")}>
                  {fmtNum(r.res_pres != null && r.psu_full != null ? r.res_pres - r.psu_full : null)}
                </td>
                <td className={clsx(TD, "text-slate-700")}>{fmtNum(r.oil_full)}</td>
                {shut ? (
                  <td colSpan={3} className="border-l border-slate-100 px-2 py-1 text-center font-medium text-amber-700">
                    SHUT IN - builds to static
                  </td>
                ) : (
                  <>
                    <td
                      className={clsx(TD, "border-l border-slate-100 text-slate-700")}
                      title={
                        fieldBasis
                          ? fieldTitle
                          : r.sonic
                            ? "Pinned at the cavitation floor (sonic throat entry)"
                            : undefined
                      }
                    >
                      {fmtNum(r.psu)}
                      {fieldBasis ? (
                        <span className="text-[10px] text-slate-400"> field</span>
                      ) : r.sonic ? (
                        <span className="text-slate-400"> *</span>
                      ) : null}
                    </td>
                    <td className={clsx(TD, "text-slate-600")}>
                      {fmtNum(r.res_pres != null && r.psu != null ? r.res_pres - r.psu : null)}
                    </td>
                    <td className={clsx(TD, "text-slate-700")}>{fmtNum(r.oil)}</td>
                  </>
                )}
                <td className={clsx(TD, "border-l border-slate-100 text-slate-500")}>
                  {rise === null ? "-" : `+${fmtNum(rise)}`}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <IprCurveGrid rows={rows} />
    </Card>
  );
}

/* ------------------------------------------------------- IPR curve grid */

function iprPanelOption(r: ChokePlanRow, xMax: number, yMax: number): EChartsOption {
  const curve = r.ipr_curve ?? [];
  const tip = (params: unknown): string => {
    const p = params as { seriesName?: string; value?: [number, number] };
    const oil = p.value?.[0] ?? null;
    const psu = p.value?.[1] ?? null;
    let html =
      ttHeader(r.well) +
      ttRow(p.seriesName === "Plan" ? ACCENT : SLATE, "Suction (psi)", fmtNum(psu, 1)) +
      ttRow(p.seriesName === "Plan" ? ACCENT : SLATE, "Oil (BOPD)", fmtNum(oil, 1));
    if (p.seriesName === "Plan" && r.psu != null && r.psu_full != null)
      html += ttRow(AXIS_LINE, "Suction rise (psi)", `+${fmtNum(r.psu - r.psu_full, 1)}`);
    return html;
  };

  const series: Record<string, unknown>[] = [
    {
      type: "line",
      silent: true,
      showSymbol: false,
      lineStyle: { color: SLATE, width: 1.5 },
      tooltip: { show: false },
      data: curve,
    },
  ];
  if (r.oil_full != null && r.psu_full != null)
    series.push({
      name: "Full open",
      type: "scatter",
      symbolSize: 11,
      itemStyle: { color: "#ffffff", borderColor: SLATE, borderWidth: 1.5 },
      data: [[r.oil_full, r.psu_full]],
    });
  // a shut well has no choked operating point - only the full-open ring
  if (r.action !== "shut" && r.oil != null && r.psu != null)
    series.push({
      name: "Plan",
      type: "scatter",
      symbolSize: 7,
      itemStyle: { color: ACCENT },
      data: [[r.oil, r.psu]],
    });

  return houseOption({
    tooltip: { trigger: "item", formatter: tip },
    grid: { left: 46, right: 10, top: 8, bottom: 34 },
    xAxis: {
      ...axis("Oil (BOPD)", { min: 0, max: xMax }),
      nameGap: 20,
      nameTextStyle: { color: SLATE, fontSize: 10, fontWeight: 500 },
      axisLabel: { color: SLATE, fontSize: 10 },
    },
    yAxis: {
      ...axis("BHP (psi)", { min: 0, max: yMax }),
      nameGap: 32,
      nameTextStyle: { color: SLATE, fontSize: 10, fontWeight: 500 },
      axisLabel: { color: SLATE, fontSize: 10 },
    },
    series,
  } as EChartsOption);
}

/** Mini IPR chart per model-basis well: the Vogel inflow curve with the
 *  full-open and choked-plan operating points. Shared axis limits across
 *  all panels so they are honestly comparable. Renders nothing when no
 *  shown row carries a usable curve (older run payloads). */
function IprCurveGrid({ rows }: { rows: ChokePlanRow[] }) {
  const shown = rows.filter((r) => r.ipr_curve != null && r.ipr_curve.length > 0);
  if (shown.length === 0) return null;

  let xMax = 0;
  let yMax = 0;
  for (const r of shown) {
    const c = r.ipr_curve as [number, number][];
    xMax = Math.max(xMax, c[c.length - 1][0]);
    yMax = Math.max(yMax, r.res_pres != null ? r.res_pres : c[0][1]);
  }

  return (
    <div>
      <p className="px-2 pb-1 pt-2 text-[11px] text-slate-500">
        Operating points on each well's inflow curve: ring = full open, dot = choked plan.
      </p>
      <div className="grid grid-cols-[repeat(auto-fill,minmax(240px,1fr))] gap-2 px-2 pb-2">
        {shown.map((r) => (
          <div key={r.well}>
            <p className="text-[11px] font-medium text-slate-600">{r.well}</p>
            <ChartPanel
              option={iprPanelOption(r, xMax, yMax)}
              height={170}
              zoom={{ xAxisIndex: "none", yAxisIndex: "none" }}
            />
          </div>
        ))}
      </div>
    </div>
  );
}
