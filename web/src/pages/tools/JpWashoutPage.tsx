/**
 * JP Wash-Out - pumps needing more PF pressure than the surface can deliver.
 *
 * Port of the retired Streamlit JP Wash-Out tool. For every JP producer whose
 * latest test measured lift water, the engine finds the PF surface pressure
 * the model needs to reproduce that measurement. Above the infrastructure cap
 * the PUMP is the problem, not the pressure.
 *
 * "Pump changed since test" matters and is shown: a well flagged on an old
 * test whose pump has since been swapped needs no changeout - that was a real
 * bug in the original tab before the flag was propagated.
 */

import { useMemo, useState } from "react";

import type { ToolRow } from "../../api/types";
import { ChartPanel } from "../../charts/ChartPanel";
import type { EChartsOption } from "../../charts/echarts";
import { ACCENT, CRIMSON, SLATE, axis, houseOption, ttHeader, ttRow } from "../../charts/theme";
import { Badge, Button, Card, Metric, Section } from "../../components/ui";
import { fmtNum } from "../../lib/format";
import { AutoTable, NumField, RunStatus, useToolRun } from "./ToolRun";

interface Req {
  months_back: number;
  ppf_limit: number;
}

const LEAD = ["Well", "Pad", "Pump", "Flagged", "PumpChangedSinceTest", "PpfRequired",
  "LiftWat", "ModeledQnz", "WtDate", "Oil", "BHP", "WHP", "Status"];

function requiredChart(rows: ToolRow[], limit: number): EChartsOption | null {
  const pts = rows
    .filter((r) => typeof r.PpfRequired === "number")
    .map((r) => ({ w: String(r.Well), p: r.PpfRequired as number, flag: Boolean(r.Flagged) }))
    .sort((a, b) => b.p - a.p);
  if (!pts.length) return null;

  return houseOption({
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "shadow" },
      formatter: (raw: unknown) => {
        const p = (raw as { name: string; value: number }[])[0];
        return ttHeader(p.name) + ttRow(p.value > limit ? CRIMSON : ACCENT, "PF required", `${fmtNum(p.value, 0)} psi`);
      },
    },
    grid: { top: 16, left: 64, right: 20, bottom: 70 },
    xAxis: { type: "category", data: pts.map((p) => p.w), ...axis(""), axisLabel: { rotate: 60, fontSize: 9 } },
    yAxis: { type: "value", ...axis("PF pressure required (psi)") },
    series: [{
      type: "bar",
      data: pts.map((p) => ({ value: p.p, itemStyle: { color: p.flag ? CRIMSON : ACCENT } })),
      markLine: {
        silent: true,
        symbol: "none",
        lineStyle: { color: SLATE, type: "dashed" },
        label: { formatter: `cap ${limit} psi`, fontSize: 10 },
        data: [{ yAxis: limit }],
      },
    }],
  });
}

export default function JpWashoutPage() {
  const [months, setMonths] = useState(6);
  const [limit, setLimit] = useState(3400);
  const run = useToolRun<Req>("/tools/washout/scan");

  const result = run.result as
    | { rows?: ToolRow[]; flagged?: number; scanned?: number; errors?: number; ppf_limit?: number }
    | null;
  const rows = result?.rows ?? [];
  const usedLimit = result?.ppf_limit ?? limit;
  const chart = useMemo(() => requiredChart(rows, usedLimit), [rows, usedLimit]);
  const [onlyFlagged, setOnlyFlagged] = useState(false);
  const shown = onlyFlagged ? rows.filter((r) => r.Flagged) : rows;

  return (
    <div className="space-y-4">
      <Section
        title="JP Wash-Out"
        actions={
          <Button onClick={() => run.run({ months_back: months, ppf_limit: limit })} disabled={run.running}>
            {run.running ? "Scanning..." : "Scan fleet"}
          </Button>
        }
      >
        <p className="mb-3 text-sm text-slate-600">
          For each JP producer, the power-fluid surface pressure the model needs to reproduce
          the latest measured lift water, at that well&apos;s own friction coefficients and the
          pump installed when the test ran. Above the cap, the pump is the constraint.
        </p>
        <div className="flex flex-wrap items-end gap-3">
          <NumField label="Lookback (months)" value={months} onChange={setMonths} min={1} max={60} />
          <NumField label="PF cap (psi)" value={limit} onChange={setLimit} min={1000} max={6000} step={100} />
        </div>
      </Section>

      <RunStatus run={run as never} idle="Press Scan fleet to calibrate every JP well." />

      {result && (
        <Card>
          <div className="flex flex-wrap items-center gap-6">
            <Metric label="Scanned" value={fmtNum(result.scanned, 0)} />
            <Metric label="Flagged" value={fmtNum(result.flagged, 0)} />
            <Metric label="Errors" value={fmtNum(result.errors, 0)} />
            <Button size="sm" variant="ghost" onClick={() => setOnlyFlagged((v) => !v)}>
              {onlyFlagged ? "Show all" : "Only flagged"}
            </Button>
            {Boolean(result.flagged) && <Badge tone="poor">{result.flagged} need review</Badge>}
          </div>
        </Card>
      )}

      {chart && (
        <Section title="PF pressure required, by well">
          <ChartPanel option={chart} height={400} zoom={{ xAxisIndex: [0], yAxisIndex: [0] }} />
        </Section>
      )}

      {rows.length > 0 && (
        <Section title="Per-well detail">
          <AutoTable rows={shown} prefer={LEAD} />
        </Section>
      )}
    </div>
  );
}
