/**
 * Batch Pump Analysis - the nozzle x throat grid sweep.
 * Mirrors woffl/gui/tabs/batch_run.py: success metrics, performance graph
 * (eliminated / semi-finalist / curve fit / recommended star), recommended
 * pump block, results table, recommender table.
 *
 * Expensive: runs ONLY on explicit submit. The page snapshots the params on
 * click and hands the snapshot to useBatch (null = not requested yet).
 */

import { useMemo, useState } from "react";

import { stableStringify } from "../api/client";
import { useBatch } from "../api/hooks";
import type { BatchRecommendation, BatchRow, SimParams, WaterType } from "../api/types";
import { NOZZLE_OPTIONS, THROAT_OPTIONS } from "../api/types";
import type { EChartsOption } from "../charts/echarts";
import { ACCENT, axis, baseGrid, baseTooltip, CRIMSON, GOLD, houseOption } from "../charts/theme";
import { ChartPanel } from "../charts/ChartPanel";
import {
  Badge,
  Button,
  Card,
  type Column,
  DataTable,
  ErrorNote,
  InfoNote,
  Metric,
  Section,
  Spinner,
  WarnNote,
} from "../components/ui";
import { MultiChipSelect, RadioRow } from "../layout/ParamFields";
import { downloadCsv } from "../lib/csv";
import { fmtNum, fmtPct, pumpCode } from "../lib/format";
import { effectiveParams, useParamsStore } from "../state/params";

/** Five-point star (ECharts has no built-in star symbol). */
const STAR_SYMBOL =
  "path://M150,25 L179,111 L269,111 L197,165 L223,251 L150,200 L77,251 L103,165 L31,111 L121,111 Z";

interface ScatterDatum {
  value: [number, number];
  name: string;
  row?: BatchRow;
  rec?: BatchRecommendation;
}

function performanceOption(
  rows: BatchRow[],
  fitCurve: { x: number[]; y: number[] } | null,
  recommended: BatchRecommendation | null,
  xMode: WaterType,
): EChartsOption {
  const waterKey: "form_wat" | "totl_wat" = xMode === "formation" ? "form_wat" : "totl_wat";
  const waterLabel = xMode === "formation" ? "Formation Water" : "Total Water";
  const xTitle = `${xMode === "formation" ? "Formation" : "Total"} Water Rate (BWPD)`;

  const tooltipFormatter = (raw: unknown): string => {
    const p = raw as { seriesName?: string; data?: unknown; value?: unknown };
    const d = p.data as Partial<ScatterDatum> | undefined;
    if (d?.row) {
      const r = d.row;
      const suffix = p.seriesName === "Semi-Finalist" ? " (Semi-Finalist)" : "";
      return [
        `<b>${d.name ?? ""}</b>${suffix}`,
        `Oil: ${fmtNum(r.qoil_std)} BOPD`,
        `${waterLabel}: ${fmtNum(r[waterKey])} BWPD`,
        `Suction P: ${fmtNum(r.psu_solv)} psig`,
        `Mach: ${fmtNum(r.mach_te, 3)}`,
        `Form Water: ${fmtNum(r.form_wat)} BWPD`,
      ].join("<br/>");
    }
    if (d?.rec) {
      const rec = d.rec;
      return [
        `<b>Recommended: ${pumpCode(rec.nozzle, rec.throat)}</b>`,
        `Oil: ${fmtNum(rec.qoil_std)} BOPD`,
        `${waterLabel}: ${fmtNum(rec.water_rate)} BWPD`,
        `Marginal WC: ${fmtNum(rec.marginal_ratio, 3)}`,
      ].join("<br/>");
    }
    const v = Array.isArray(p.value) ? (p.value as number[]) : null;
    if (v) return `Oil: ${fmtNum(v[1])} BOPD<br/>${waterLabel}: ${fmtNum(v[0])} BWPD`;
    return "";
  };

  const elimData: ScatterDatum[] = [];
  const semiData: ScatterDatum[] = [];
  for (const r of rows) {
    (r.semi ? semiData : elimData).push({
      value: [r[waterKey], r.qoil_std],
      name: pumpCode(r.nozzle, r.throat),
      row: r,
    });
  }

  const series: Record<string, unknown>[] = [
    {
      name: "Eliminated",
      type: "scatter",
      symbolSize: 9,
      itemStyle: { color: ACCENT, borderColor: "#0f172a", borderWidth: 1 },
      label: {
        show: true,
        position: "inside",
        formatter: "{b}",
        fontSize: 9,
        color: "#1e293b",
      },
      labelLayout: { hideOverlap: true },
      data: elimData,
    },
    {
      name: "Semi-Finalist",
      type: "scatter",
      symbol: "diamond",
      symbolSize: 12,
      itemStyle: { color: CRIMSON, borderColor: "#0f172a", borderWidth: 1 },
      label: {
        show: true,
        position: "top",
        formatter: "{b}",
        fontSize: 10,
        fontWeight: 600,
        color: CRIMSON,
      },
      labelLayout: { hideOverlap: true },
      data: semiData,
    },
  ];

  if (fitCurve && fitCurve.x.length > 0) {
    series.push({
      name: "Exp. Curve Fit",
      type: "line",
      showSymbol: false,
      lineStyle: { color: CRIMSON, width: 2, type: "dashed" },
      itemStyle: { color: CRIMSON },
      data: fitCurve.x.map((x, i) => [x, fitCurve.y[i]]),
    });
  }

  if (recommended) {
    const rec: ScatterDatum = {
      value: [recommended.water_rate, recommended.qoil_std],
      name: pumpCode(recommended.nozzle, recommended.throat),
      rec: recommended,
    };
    series.push({
      name: "Recommended",
      type: "scatter",
      symbol: STAR_SYMBOL,
      symbolSize: 22,
      z: 10,
      itemStyle: { color: GOLD, borderColor: "#1e293b", borderWidth: 1.5 },
      data: [rec],
    });
  }

  return houseOption({
    tooltip: { ...baseTooltip, trigger: "item", formatter: tooltipFormatter },
    legend: { top: 4, right: 8, textStyle: { fontSize: 12 } },
    grid: { ...baseGrid, top: 48 },
    // Formation water sits in a narrow band (set by the IPR, not the pump);
    // anchoring at zero squashes the points, so let the axis fit the data.
    xAxis: {
      type: "value",
      ...axis(xTitle),
      ...(xMode === "formation" ? { scale: true } : { min: 0 }),
    },
    yAxis: { type: "value", ...axis("Produced Oil Rate (BOPD)", { min: 0 }) },
    series,
  });
}

interface RecommenderRow extends Record<string, unknown> {
  nozzle: string;
  throat: string;
  qoil_std: number;
  water: number;
  ratio: number | null;
  marginal_watercut: number | null;
}

export default function BatchPage() {
  const well = useParamsStore((s) => s.well);
  const params = useParamsStore((s) => s.params);
  const [snapshot, setSnapshot] = useState<SimParams | null>(null);

  const query = useBatch(well, snapshot);
  const data = query.data;

  const stale =
    snapshot !== null && stableStringify(effectiveParams(params)) !== stableStringify(snapshot);

  const xMode: WaterType = data?.x_mode ?? params.water_type;
  const waterLabel = xMode === "formation" ? "Formation Water" : "Total Water";
  const ratioLabel =
    xMode === "formation" ? "Marginal Oil/Formation Water Ratio" : "Marginal Oil/Total Water Ratio";

  /** Converged rows only - failed combos carry a non-finite oil rate. */
  const okRows = useMemo(
    () => (data ? data.rows.filter((r) => Number.isFinite(r.qoil_std)) : []),
    [data],
  );

  const chartOption = useMemo(
    () =>
      data ? performanceOption(okRows, data.fit_curve, data.recommended, data.x_mode) : null,
    [data, okRows],
  );

  const recommenderRows = useMemo<RecommenderRow[]>(() => {
    if (!data) return [];
    const ratioKey = data.x_mode === "formation" ? "mofwr" : "motwr";
    return okRows
      .filter((r) => r.semi)
      .sort((a, b) => a.qoil_std - b.qoil_std)
      .map((r) => {
        const ratio = r[ratioKey];
        const finite = typeof ratio === "number" && Number.isFinite(ratio) ? ratio : null;
        return {
          nozzle: r.nozzle,
          throat: r.throat,
          qoil_std: r.qoil_std,
          water: data.x_mode === "formation" ? r.form_wat : r.totl_wat,
          ratio: finite,
          marginal_watercut: finite !== null ? 1 / (1 + finite) : null,
        };
      });
  }, [data, okRows]);

  const resultColumns: Column<BatchRow>[] = [
    { key: "nozzle", label: "Nozzle" },
    { key: "throat", label: "Throat" },
    { key: "qoil_std", label: "Oil Rate (BOPD)", align: "right", render: (r) => fmtNum(r.qoil_std) },
    { key: "form_wat", label: "Formation Water (BWPD)", align: "right", render: (r) => fmtNum(r.form_wat) },
    { key: "lift_wat", label: "Lift Water (BWPD)", align: "right", render: (r) => fmtNum(r.lift_wat) },
    { key: "totl_wat", label: "Total Water (BWPD)", align: "right", render: (r) => fmtNum(r.totl_wat) },
    { key: "psu_solv", label: "Suction Pressure (psig)", align: "right", render: (r) => fmtNum(r.psu_solv) },
    { key: "mach_te", label: "Throat Entry Mach", align: "right", render: (r) => fmtNum(r.mach_te, 3) },
    { key: "sonic_status", label: "Sonic Flow", align: "center", render: (r) => (r.sonic_status ? "Yes" : "No") },
    { key: "semi", label: "Semi-Finalist", align: "center", render: (r) => (r.semi ? "Yes" : "No") },
  ];

  const recommenderColumns: Column<RecommenderRow>[] = [
    { key: "nozzle", label: "Nozzle" },
    { key: "throat", label: "Throat" },
    { key: "qoil_std", label: "Oil Rate (BOPD)", align: "right", render: (r) => fmtNum(r.qoil_std) },
    { key: "water", label: `${waterLabel} (BWPD)`, align: "right", render: (r) => fmtNum(r.water) },
    { key: "ratio", label: ratioLabel, align: "right", render: (r) => fmtNum(r.ratio, 3) },
    { key: "marginal_watercut", label: "Marginal Watercut", align: "right", render: (r) => fmtNum(r.marginal_watercut, 3) },
  ];

  const rec = data?.recommended ?? null;
  const combos = params.nozzle_batch_options.length * params.throat_batch_options.length;

  return (
    <div className="space-y-4">
      <Section
        title="Batch Pump Analysis"
        actions={
          <Button
            variant="primary"
            busy={query.isFetching}
            disabled={combos === 0}
            onClick={() => setSnapshot(effectiveParams(params))}
          >
            Run batch sweep
          </Button>
        }
      >
        <Card className="space-y-3">
          <div className="grid gap-3 lg:grid-cols-2">
            <MultiChipSelect
              label="Nozzle sizes to test"
              field="nozzle_batch_options"
              options={NOZZLE_OPTIONS}
            />
            <MultiChipSelect
              label="Throat ratios to test"
              field="throat_batch_options"
              options={THROAT_OPTIONS}
            />
          </div>
          <div className="flex flex-wrap items-end gap-6">
            <RadioRow
              label="Water Type"
              field="water_type"
              options={[
                { value: "total", label: "Total", hint: "Total liquid (oil + water)" },
                { value: "formation", label: "Formation", hint: "Formation water only" },
              ]}
            />
            <div className="flex items-center gap-2 pb-0.5">
              <Badge>{combos} combinations</Badge>
              <Badge title="Recommender cutoff - set under Advanced > Field in the sidebar">
                Marginal WC: {fmtNum(params.marginal_watercut, 2)}
              </Badge>
            </div>
          </div>
        </Card>
      </Section>

      {snapshot === null ? (
        <InfoNote>
          Pick the nozzle and throat grid above, then run the sweep. Nothing runs until you
          submit - this solve is expensive.
        </InfoNote>
      ) : query.isPending ? (
        <Spinner label="Sweeping nozzle x throat grid..." />
      ) : query.isError ? (
        <ErrorNote error={query.error} />
      ) : data ? (
        <>
          {stale && (
            <WarnNote>Inputs changed since this sweep - re-run to refresh</WarnNote>
          )}

          <div className="grid grid-cols-3 gap-3">
            <Metric label="Total Combinations" value={fmtNum(data.stats.total)} />
            <Metric label="Successful Runs" value={fmtNum(data.stats.successful)} />
            <Metric label="Success Rate" value={fmtPct(data.stats.success_pct / 100, 1)} />
          </div>
          {data.stats.successful < data.stats.total && (
            <WarnNote>
              {data.stats.total - data.stats.successful} of {data.stats.total} nozzle/throat
              combinations failed to converge and are excluded from the graph, table, and
              recommender below.
            </WarnNote>
          )}

          <Section title={`Jet Pump Performance (${waterLabel})`}>
            <ChartPanel option={chartOption} height={560} zoom={{ xAxisIndex: [0], yAxisIndex: [0] }} />
          </Section>

          {rec && (
            <Section title="Recommended Jet Pump">
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
                <Metric label="Nozzle Size" value={rec.nozzle} />
                <Metric label="Throat Ratio" value={rec.throat} />
                <Metric label="Oil Rate" value={`${fmtNum(rec.qoil_std, 1)} BOPD`} />
                <Metric label={waterLabel} value={`${fmtNum(rec.water_rate, 1)} BWPD`} />
                <Metric label="Marginal Watercut" value={fmtNum(rec.marginal_ratio, 3)} />
              </div>
              {rec.recommendation_type === "best_available" ? (
                <WarnNote className="mt-3">
                  No jet pump meets the specified marginal watercut threshold. This is the
                  best available option.
                </WarnNote>
              ) : (
                <p className="mt-3 text-xs text-slate-500">
                  Chosen by the marginal water-cut cutoff of{" "}
                  <span className="font-semibold">{fmtNum(snapshot.marginal_watercut, 2)}</span>
                  : the pump closest to where the curve's marginal WC reaches the cutoff
                  while staying below it. Each barrel of extra fluid from this pump is{" "}
                  {fmtNum(rec.marginal_ratio, 3)} water; larger pumps add water above the
                  cutoff.
                </p>
              )}
            </Section>
          )}

          <Section
            title="Jet Pump Performance Data"
            actions={
              <Button
                size="sm"
                onClick={() =>
                  downloadCsv(
                    "jetpump_batch_results.csv",
                    resultColumns.map((c) => ({ key: c.key, label: c.label })),
                    okRows,
                  )
                }
              >
                Download CSV
              </Button>
            }
          >
            <DataTable
              columns={resultColumns}
              rows={okRows}
              rowKey={(r) => pumpCode(r.nozzle, r.throat)}
              highlightRow={(r) => r.semi}
              maxHeight="28rem"
            />
          </Section>

          <Section
            title="Jet Pump Recommender Results"
            actions={
              <Button
                size="sm"
                disabled={recommenderRows.length === 0}
                onClick={() =>
                  downloadCsv(
                    "jetpump_recommender_results.csv",
                    recommenderColumns.map((c) => ({ key: c.key, label: c.label })),
                    recommenderRows,
                  )
                }
              >
                Download CSV
              </Button>
            }
          >
            <DataTable
              columns={recommenderColumns}
              rows={recommenderRows}
              rowKey={(r) => pumpCode(r.nozzle, r.throat)}
              highlightRow={(r) =>
                rec !== null && r.nozzle === rec.nozzle && r.throat === rec.throat
              }
              emptyLabel="No semi-finalist jet pumps found"
            />
            {rec && recommenderRows.length > 0 && (
              <p className="mt-2 text-xs text-slate-500">
                The recommended pump is highlighted: Nozzle {rec.nozzle}, Throat {rec.throat}{" "}
                with a marginal watercut of {fmtNum(rec.marginal_ratio, 3)}.
              </p>
            )}
          </Section>
        </>
      ) : null}
    </div>
  );
}
