/**
 * PF Scenario - oil and BHP at two power-fluid pressures.
 *
 * Port of the retired Streamlit PF Scenario tool. Sibling of Header Impact:
 * that one sweeps wellhead pressure holding PF fixed, this sweeps PF holding
 * wellhead pressure fixed.
 */

import { useMemo, useState } from "react";

import { useWells } from "../../api/hooks";
import type { ToolRow } from "../../api/types";
import { ChartPanel } from "../../charts/ChartPanel";
import type { EChartsOption } from "../../charts/echarts";
import { ACCENT, CRIMSON, axis, houseOption, ttHeader, ttRow } from "../../charts/theme";
import { Button, Card, Metric, Section } from "../../components/ui";
import { fmtNum } from "../../lib/format";
import { AutoTable, NumField, RunStatus, useToolRun } from "./ToolRun";

interface Req {
  wells: string[];
  pf_a: number;
  pf_b: number;
  months_back: number;
}

const LEAD = ["Well", "Pad", "Nozzle", "Throat", "WHP", "PfA", "OilA", "BhpA",
  "PfB", "OilB", "BhpB", "DeltaOil", "DeltaBhp", "Status"];

function deltaChart(rows: ToolRow[]): EChartsOption | null {
  const pts = rows
    .filter((r) => typeof r.DeltaOil === "number")
    .map((r) => ({ w: String(r.Well), d: r.DeltaOil as number }))
    .sort((a, b) => b.d - a.d);
  if (!pts.length) return null;
  return houseOption({
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "shadow" },
      formatter: (raw: unknown) => {
        const p = (raw as { name: string; value: number }[])[0];
        return ttHeader(p.name) + ttRow(p.value >= 0 ? ACCENT : CRIMSON, "Delta oil", `${fmtNum(p.value, 1)} BOPD`);
      },
    },
    grid: { top: 16, left: 64, right: 20, bottom: 64 },
    xAxis: { type: "category", data: pts.map((p) => p.w), ...axis(""), axisLabel: { rotate: 45, fontSize: 10 } },
    yAxis: { type: "value", ...axis("Delta oil, B - A (BOPD)") },
    series: [{
      type: "bar",
      data: pts.map((p) => ({ value: p.d, itemStyle: { color: p.d >= 0 ? ACCENT : CRIMSON } })),
    }],
  });
}

export default function PfScenarioPage() {
  const wellsQ = useWells();
  const all = useMemo(
    () => (wellsQ.data?.wells ?? []).map((w) => w.name).sort(),
    [wellsQ.data],
  );

  const [selected, setSelected] = useState<string[]>([]);
  const [pfA, setPfA] = useState(2800);
  const [pfB, setPfB] = useState(3200);
  const [months, setMonths] = useState(6);
  const run = useToolRun<Req>("/tools/pf-scenario/run");

  const result = run.result as { rows?: ToolRow[]; totals?: Record<string, number> } | null;
  const rows = result?.rows ?? [];
  const chart = useMemo(() => deltaChart(rows), [rows]);

  return (
    <div className="space-y-4">
      <Section
        title="PF Scenario"
        actions={
          <Button
            onClick={() => run.run({ wells: selected, pf_a: pfA, pf_b: pfB, months_back: months })}
            disabled={!selected.length || run.running}
          >
            {run.running ? "Solving..." : "Run"}
          </Button>
        }
      >
        <p className="mb-3 text-sm text-slate-600">
          Solves each selected well at two power-fluid surface pressures, holding wellhead
          pressure at its latest measured value. The IPR comes from the well&apos;s own Vogel
          fit where a BHP gauge exists, and from characteristics defaults otherwise.
        </p>

        <div className="flex flex-wrap items-end gap-3">
          <NumField label="Scenario A (psi)" value={pfA} onChange={setPfA} min={1000} max={5000} step={50} />
          <NumField label="Scenario B (psi)" value={pfB} onChange={setPfB} min={1000} max={5000} step={50} />
          <NumField label="Lookback (months)" value={months} onChange={setMonths} min={1} max={60} />
          <div className="flex gap-2">
            <Button size="sm" variant="ghost" onClick={() => setSelected(all)}>All wells</Button>
            <Button size="sm" variant="ghost" onClick={() => setSelected([])}>Clear</Button>
          </div>
        </div>

        <div className="mt-3">
          <span className="text-xs text-slate-500">
            Wells ({selected.length} of {all.length} selected)
          </span>
          <div className="mt-1 flex max-h-40 flex-wrap gap-1 overflow-y-auto rounded-md border border-slate-200 p-2">
            {all.map((w) => (
              <button
                key={w}
                type="button"
                onClick={() =>
                  setSelected((s) => (s.includes(w) ? s.filter((x) => x !== w) : [...s, w]))
                }
                className={
                  "rounded border px-1.5 py-0.5 text-xs transition-colors " +
                  (selected.includes(w)
                    ? "border-blue-500 bg-blue-50 font-medium text-blue-700"
                    : "border-slate-200 bg-white text-slate-500 hover:bg-slate-50")
                }
              >
                {w}
              </button>
            ))}
          </div>
        </div>
      </Section>

      <RunStatus run={run as never} idle="Select wells and press Run." />

      {result?.totals && (
        <Card>
          <div className="flex flex-wrap gap-6">
            <Metric label="Wells solved" value={fmtNum(result.totals.wells, 0)} />
            <Metric
              label={`Delta oil, ${pfB} vs ${pfA} psi`}
              value={`${fmtNum(result.totals.delta_oil, 1)} BOPD`}
            />
          </div>
        </Card>
      )}

      {chart && (
        <Section title="Oil change by well">
          <ChartPanel option={chart} height={360} zoom={{ xAxisIndex: [0], yAxisIndex: [0] }} />
        </Section>
      )}

      {rows.length > 0 && (
        <Section title="Per-well detail">
          <AutoTable rows={rows} prefer={LEAD} />
        </Section>
      )}
    </div>
  );
}
