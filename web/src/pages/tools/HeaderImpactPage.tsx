/**
 * Header Pressure Impact - what moving a pad header does to BHP and oil.
 *
 * Port of the retired Streamlit Header Impact tool. JP wells go through WOFFL
 * physics (solved at the current WHP and again at WHP + delta); ESP /
 * gas-lift / flowing wells take the empirical slope path. The verdict column
 * is the point: "sonic-decoupled" means a choked pump cannot pass a
 * downstream pressure change back to the formation, so the header lever will
 * not move that well no matter how far you pull it.
 */

import { useMemo, useState } from "react";

import { useWells } from "../../api/hooks";
import { ChartPanel } from "../../charts/ChartPanel";
import type { EChartsOption } from "../../charts/echarts";
import { ACCENT, CRIMSON, axis, houseOption, ttHeader, ttRow } from "../../charts/theme";
import { Button, Card, Metric, Section } from "../../components/ui";
import { fmtNum } from "../../lib/format";
import type { ToolRow } from "../../api/types";
import { AutoTable, NumField, RunStatus, VerdictBadge, useToolRun } from "./ToolRun";

interface Req {
  pads: string[];
  delta_p: number;
  months_back: number;
}

const LEAD = ["Well", "Pad", "Lift", "Pump", "Verdict", "WHP now", "WHP scen",
  "BHP now", "BHP scen", "DeltaBhp", "Oil now", "Oil scen", "DeltaOil"];

function waterfall(rows: ToolRow[]): EChartsOption | null {
  const pts = rows
    .filter((r) => typeof r.DeltaOil === "number")
    .map((r) => ({ well: String(r.Well), d: r.DeltaOil as number }))
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
    xAxis: { type: "category", data: pts.map((p) => p.well), ...axis(""), axisLabel: { rotate: 45, fontSize: 10 } },
    yAxis: { type: "value", ...axis("Delta oil (BOPD)") },
    series: [{
      type: "bar",
      data: pts.map((p) => ({ value: p.d, itemStyle: { color: p.d >= 0 ? ACCENT : CRIMSON } })),
    }],
  });
}

export default function HeaderImpactPage() {
  const wells = useWells();
  const pads = useMemo(
    () => [...new Set((wells.data?.wells ?? []).map((w) => w.pad).filter(Boolean))].sort(),
    [wells.data],
  );

  const [selected, setSelected] = useState<string[]>([]);
  const [deltaP, setDeltaP] = useState(-50);
  const [months, setMonths] = useState(6);
  const run = useToolRun<Req>("/tools/header-impact/run");

  const result = run.result as { rows?: ToolRow[]; totals?: Record<string, number>; no_tags?: string[] } | null;
  const rows = result?.rows ?? [];
  const chart = useMemo(() => waterfall(rows), [rows]);

  return (
    <div className="space-y-4">
      <Section
        title="Header Pressure Impact"
        actions={
          <Button
            onClick={() => run.run({ pads: selected, delta_p: deltaP, months_back: months })}
            disabled={!selected.length || run.running}
          >
            {run.running ? "Solving..." : "Run"}
          </Button>
        }
      >
        <p className="mb-3 text-sm text-slate-600">
          Models a header pressure change across every producer on the selected pads. JP wells
          use WOFFL physics; ESP, gas-lift and flowing wells use their fitted within-day slope.
          Power-fluid pressure is held fixed and seeded from the pad default.
        </p>

        <div className="flex flex-wrap items-end gap-3">
          <div>
            <span className="text-xs text-slate-500">Pads</span>
            <div className="mt-1 flex flex-wrap gap-1">
              {pads.map((p) => (
                <button
                  key={p}
                  type="button"
                  onClick={() =>
                    setSelected((s) => (s.includes(p) ? s.filter((x) => x !== p) : [...s, p]))
                  }
                  className={
                    "h-8 min-w-8 rounded-md border px-2 text-sm transition-colors " +
                    (selected.includes(p)
                      ? "border-blue-500 bg-blue-50 font-medium text-blue-700"
                      : "border-slate-300 bg-white text-slate-600 hover:bg-slate-50")
                  }
                >
                  {p}
                </button>
              ))}
            </div>
          </div>
          <NumField label="Header change (psi)" value={deltaP} onChange={setDeltaP} min={-300} max={300} step={5} />
          <NumField label="Lookback (months)" value={months} onChange={setMonths} min={1} max={60} />
        </div>
      </Section>

      <RunStatus run={run as never} idle="Pick one or more pads and press Run." />

      {result?.totals && (
        <Card>
          <div className="flex flex-wrap gap-6">
            <Metric label="Wells" value={fmtNum(result.totals.wells, 0)} />
            <Metric
              label={`Delta oil at ${deltaP > 0 ? "+" : ""}${deltaP} psi`}
              value={`${fmtNum(result.totals.delta_oil, 1)} BOPD`}
            />
            <Metric label="Responsive" value={fmtNum(result.totals.responsive, 0)} />
            <Metric label="Sonic-decoupled" value={fmtNum(result.totals.sonic, 0)} />
          </div>
        </Card>
      )}

      {chart && (
        <Section title="Oil change by well">
          <ChartPanel option={chart} height={380} zoom={{ xAxisIndex: [0], yAxisIndex: [0] }} />
        </Section>
      )}

      {rows.length > 0 && (
        <Section title="Per-well detail">
          <div className="mb-2 flex flex-wrap gap-1">
            {[...new Set(rows.map((r) => String(r.Verdict ?? "")))].filter(Boolean).map((v) => (
              <VerdictBadge key={v} value={v} />
            ))}
          </div>
          <AutoTable rows={rows} prefer={LEAD} />
        </Section>
      )}
    </div>
  );
}
