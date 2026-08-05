/**
 * Power Fluid Range - how oil rate trades off against PF surface pressure
 * for each candidate pump. Mirrors woffl/gui/tabs/power_fluid_range.py:
 * summary metrics, per-pump performance charts, comprehensive data and
 * best-performer views. Expensive sweep: runs only on explicit submit.
 */

import { useMemo, useState } from "react";

import { stableStringify } from "../api/client";
import { usePfRange } from "../api/hooks";
import type { PfRangeRow, SimParams } from "../api/types";
import { NOZZLE_OPTIONS, THROAT_OPTIONS } from "../api/types";
import type { EChartsOption } from "../charts/echarts";
import { axis, baseGrid, baseTooltip, CATEGORY20, houseOption } from "../charts/theme";
import { useEChart } from "../charts/useEChart";
import {
  Badge,
  Button,
  type Column,
  Card,
  DataTable,
  ErrorNote,
  InfoNote,
  Metric,
  Section,
  Spinner,
  WarnNote,
} from "../components/ui";
import { MultiChipSelect, NumberField } from "../layout/ParamFields";
import { downloadCsv } from "../lib/csv";
import { fmtNum } from "../lib/format";
import { effectiveParams, useParamsStore } from "../state/params";

type SubView = "charts" | "data" | "best";

const SUB_VIEWS: Array<{ id: SubView; label: string }> = [
  { id: "charts", label: "Performance vs Pressure" },
  { id: "data", label: "Comprehensive Data" },
  { id: "best", label: "Best Performers" },
];

const DATA_COLUMNS: Column<PfRangeRow>[] = [
  { key: "pump", label: "Pump" },
  { key: "power_fluid_pressure", label: "PF Pressure (psi)", align: "right", render: (r) => fmtNum(r.power_fluid_pressure) },
  { key: "qoil_std", label: "Oil (BOPD)", align: "right", render: (r) => fmtNum(r.qoil_std) },
  { key: "form_wat", label: "Form Water (BWPD)", align: "right", render: (r) => fmtNum(r.form_wat) },
  { key: "lift_wat", label: "Lift Water (BWPD)", align: "right", render: (r) => fmtNum(r.lift_wat) },
  { key: "totl_wat", label: "Total Water (BWPD)", align: "right", render: (r) => fmtNum(r.totl_wat) },
  { key: "psu_solv", label: "Suction P (psig)", align: "right", render: (r) => fmtNum(r.psu_solv) },
  { key: "mach_te", label: "Mach", align: "right", render: (r) => fmtNum(r.mach_te, 3) },
  { key: "sonic_status", label: "Sonic", align: "center", render: (r) => (r.sonic_status ? "yes" : "no") },
];

function sweepChart(
  rows: PfRangeRow[],
  pumps: string[],
  yKey: "qoil_std" | "totl_wat",
  yName: string,
): EChartsOption {
  const series = pumps.map((pump, i) => ({
    name: pump,
    type: "line" as const,
    symbol: "circle",
    symbolSize: 7,
    lineStyle: { width: 2 },
    itemStyle: { color: CATEGORY20[i % CATEGORY20.length] },
    data: rows
      .filter((r) => r.pump === pump)
      .sort((a, b) => a.power_fluid_pressure - b.power_fluid_pressure)
      .map((r) => [r.power_fluid_pressure, r[yKey]]),
  }));
  return houseOption({
    grid: { ...baseGrid, right: 130 },
    tooltip: {
      ...baseTooltip,
      trigger: "axis",
      valueFormatter: (v: unknown) => (typeof v === "number" ? fmtNum(v) : "-"),
    },
    legend: {
      type: "scroll",
      orient: "vertical",
      right: 8,
      top: 24,
      textStyle: { fontSize: 11 },
    },
    xAxis: { type: "value", ...axis("Power Fluid Surface Pressure (psi)"), min: "dataMin", max: "dataMax" },
    yAxis: { type: "value", ...axis(yName, { min: 0 }) },
    series,
  });
}

export default function PfRangePage() {
  const well = useParamsStore((s) => s.well);
  const params = useParamsStore((s) => s.params);

  const [snapshot, setSnapshot] = useState<SimParams | null>(null);
  const [subView, setSubView] = useState<SubView>("charts");
  const query = usePfRange(well, snapshot);
  const data = query.data;

  const pumps = useMemo(() => {
    if (!data) return [];
    const seen = new Set<string>();
    for (const row of data.rows) seen.add(row.pump);
    return [...seen].sort();
  }, [data]);

  const best = useMemo(() => {
    if (!data) return [];
    const byPressure = new Map<number, PfRangeRow>();
    for (const row of data.rows) {
      const cur = byPressure.get(row.power_fluid_pressure);
      if (!cur || row.qoil_std > cur.qoil_std) byPressure.set(row.power_fluid_pressure, row);
    }
    return [...byPressure.values()].sort((a, b) => a.power_fluid_pressure - b.power_fluid_pressure);
  }, [data]);

  const overallBest = useMemo(
    () => best.reduce<PfRangeRow | null>((acc, r) => (acc === null || r.qoil_std > acc.qoil_std ? r : acc), null),
    [best],
  );

  const oilOption = useMemo(
    () => (data && subView === "charts" ? sweepChart(data.rows, pumps, "qoil_std", "Produced Oil Rate (BOPD)") : null),
    [data, pumps, subView],
  );
  const waterOption = useMemo(
    () => (data && subView === "charts" ? sweepChart(data.rows, pumps, "totl_wat", "Total Water Rate (BWPD)") : null),
    [data, pumps, subView],
  );
  const oilRef = useEChart(oilOption);
  const waterRef = useEChart(waterOption);

  const pumpCount = params.nozzle_batch_options.length * params.throat_batch_options.length;
  const stale = snapshot !== null && stableStringify(effectiveParams(params)) !== stableStringify(snapshot);

  return (
    <div className="space-y-4">
      <Card className="space-y-3">
        <div className="flex items-start justify-between gap-3">
          <div className="grid flex-1 gap-3 sm:grid-cols-3">
            <NumberField label="Min Power Fluid Pressure (psi)" field="power_fluid_min" step={100} />
            <NumberField label="Max Power Fluid Pressure (psi)" field="power_fluid_max" step={100} />
            <NumberField label="Pressure Step (psi)" field="power_fluid_step" step={50} />
          </div>
          <Button
            variant="primary"
            busy={query.isFetching}
            disabled={pumpCount === 0 || params.power_fluid_max < params.power_fluid_min}
            onClick={() => setSnapshot(effectiveParams(params))}
          >
            Run PF range analysis
          </Button>
        </div>
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
        <div className="flex flex-wrap items-center gap-2">
          <Badge tone="info">
            {fmtNum(params.power_fluid_min)}-{fmtNum(params.power_fluid_max)} psi, step{" "}
            {fmtNum(params.power_fluid_step)}
          </Badge>
          <Badge>{pumpCount} pumps</Badge>
          <Badge>{well}</Badge>
        </div>
      </Card>

      {stale && !query.isFetching && (
        <WarnNote>Inputs changed since this sweep - re-run to refresh the results.</WarnNote>
      )}

      {snapshot === null && (
        <InfoNote>
          Sweeps every selected pump across the pressure range above. Press Run PF range
          analysis to start.
        </InfoNote>
      )}

      {query.isError && <ErrorNote error={query.error} />}
      {query.isFetching && <Spinner label="Sweeping power fluid pressures..." />}

      {data && !query.isFetching && (
        <>
          <div className="flex flex-wrap gap-3">
            <Metric label="Pressure points" value={fmtNum(data.pressures.length)} />
            <Metric label="Pumps tested" value={fmtNum(pumps.length)} />
            <Metric
              label="Best oil rate"
              value={overallBest ? fmtNum(overallBest.qoil_std) : "-"}
              sub={overallBest ? `${overallBest.pump} at ${fmtNum(overallBest.power_fluid_pressure)} psi` : undefined}
              tone={overallBest ? "good" : "neutral"}
            />
            <Metric label="Successful solves" value={fmtNum(data.rows.length)} />
          </div>

          <div className="flex gap-1 rounded-lg border border-slate-200 bg-white p-1 w-fit">
            {SUB_VIEWS.map((v) => (
              <button
                key={v.id}
                type="button"
                onClick={() => setSubView(v.id)}
                className={
                  subView === v.id
                    ? "rounded-md bg-blue-600 px-3 py-1 text-sm font-medium text-white"
                    : "rounded-md px-3 py-1 text-sm text-slate-600 hover:bg-slate-100"
                }
              >
                {v.label}
              </button>
            ))}
          </div>

          {subView === "charts" && (
            <div className="space-y-4">
              <Card padded={false} className="p-2">
                <div ref={oilRef} className="h-[420px]" />
              </Card>
              <Card padded={false} className="p-2">
                <div ref={waterRef} className="h-[420px]" />
              </Card>
            </div>
          )}

          {subView === "data" && (
            <Section
              title={`All results (${data.rows.length})`}
              actions={
                <Button
                  size="sm"
                  onClick={() =>
                    downloadCsv(
                      "power_fluid_range_results.csv",
                      DATA_COLUMNS.map((c) => ({ key: c.key, label: c.label })),
                      data.rows,
                    )
                  }
                >
                  CSV
                </Button>
              }
            >
              <DataTable
                columns={DATA_COLUMNS}
                rows={data.rows}
                rowKey={(r) => `${r.pump}-${r.power_fluid_pressure}`}
                maxHeight="30rem"
              />
            </Section>
          )}

          {subView === "best" && (
            <Section title="Best pump at each pressure">
              <DataTable
                columns={DATA_COLUMNS}
                rows={best}
                rowKey={(r) => `${r.pump}-${r.power_fluid_pressure}`}
                highlightRow={(r) => overallBest !== null && r === overallBest}
                maxHeight="30rem"
              />
            </Section>
          )}
        </>
      )}
    </div>
  );
}
