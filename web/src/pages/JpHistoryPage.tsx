/**
 * JP History - production rates and BHP over the well's jet pump install
 * eras, with a pumps-in-hole timeline strip under the main chart and the
 * install history table below. Mirrors the figure structure of
 * woffl/gui/tabs/jp_history_tab.py:build_history_with_strip_figure.
 */

import { useMemo, useState } from "react";

import { useJpHistory } from "../api/hooks";
import type { JpInstallRow } from "../api/types";
import type { EChartsOption } from "../charts/echarts";
import { ACCENT, axis, baseTooltip, CATEGORY20, CRIMSON, GOLD, houseOption, SLATE } from "../charts/theme";
import { useEChart } from "../charts/useEChart";
import { Badge, Card, type Column, DataTable, ErrorNote, InfoNote, Section, Spinner } from "../components/ui";
import { fmtDate, fmtNum, pumpCode } from "../lib/format";
import { useParamsStore } from "../state/params";

/** Finite number or null - test rows arrive as loosely typed JSON dicts. */
function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

/** Epoch milliseconds from an ISO date string, null when absent/invalid. */
function ms(v: unknown): number | null {
  if (typeof v !== "string" || v.length === 0) return null;
  const t = new Date(v).getTime();
  return Number.isNaN(t) ? null : t;
}

/** Shape of the cartesian coordinate system handed to custom renderItem. */
interface BandCoordSys {
  x: number;
  y: number;
  width: number;
  height: number;
}

interface BandRenderParams {
  coordSys: BandCoordSys;
}

interface BandRenderApi {
  value: (dim: number) => number;
  coord: (point: [number, number]) => [number, number];
}

const CHECKBOX_CLS = "h-4 w-4 rounded border-slate-300 accent-blue-600";

const INSTALL_COLUMNS: Column<JpInstallRow>[] = [
  { key: "date_set", label: "Date Set", render: (r) => fmtDate(r.date_set) },
  { key: "date_pulled", label: "Date Pulled", render: (r) => fmtDate(r.date_pulled) },
  { key: "pump", label: "Pump", render: (r) => pumpCode(r.nozzle, r.throat) },
  { key: "tubing_od", label: "Tubing OD", align: "right", render: (r) => fmtNum(r.tubing_od, 3) },
  { key: "circulating", label: "Circulation", render: (r) => r.circulating ?? "-" },
  { key: "manufacturer", label: "Manufacturer", render: (r) => r.manufacturer ?? "-" },
  {
    key: "pump_converted",
    label: "Converted",
    align: "center",
    render: (r) =>
      r.pump_converted ? (
        <Badge tone="fair" title={r.raw_pump ?? undefined}>
          Converted
        </Badge>
      ) : (
        "-"
      ),
  },
];

export default function JpHistoryPage() {
  const well = useParamsStore((s) => s.well);
  const [bhpFromZero, setBhpFromZero] = useState(true);
  const [showPf, setShowPf] = useState(false);

  const query = useJpHistory(well);
  const data = query.data;

  const option = useMemo<EChartsOption | null>(() => {
    if (!data || data.installs.length === 0) return null;
    const now = Date.now();

    const oilPts: [number, number][] = [];
    const fwatPts: [number, number][] = [];
    const pfPts: [number, number][] = [];
    const bhpTestPts: [number, number][] = [];
    for (const t of data.tests) {
      const x = ms(t.date);
      if (x === null) continue;
      const oil = num(t.oil_rate);
      if (oil !== null) oilPts.push([x, oil]);
      const fwat = num(t.fwat_rate);
      if (fwat !== null) fwatPts.push([x, fwat]);
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

    const codes: string[] = [];
    const bandData: [number, number, number][] = [];
    const markData: Record<string, unknown>[] = [];
    for (const ins of data.installs) {
      const set = ms(ins.date_set);
      if (set === null) continue;
      const idx = codes.length;
      const code = pumpCode(ins.nozzle, ins.throat);
      codes.push(code);
      bandData.push([set, ms(ins.date_pulled) ?? now, idx]);
      markData.push({
        xAxis: set,
        lineStyle: { color: GOLD, type: "dashed", width: 1 },
        label: { formatter: code, position: "insideEndTop", color: GOLD, fontSize: 10 },
      });
    }

    const allX = [
      ...oilPts,
      ...fwatPts,
      ...pfPts,
      ...bhpTestPts,
      ...bhpDailyPts,
      ...bandData.map((b): [number, number] => [b[0], 0]),
    ].map((p) => p[0]);
    if (allX.length === 0) return null;
    const minMs = Math.min(...allX);
    const maxMs = Math.max(now, ...allX);

    const renderBand = (p: BandRenderParams, api: BandRenderApi): Record<string, unknown> => {
      const x0raw = api.coord([api.value(0), 0])[0];
      const x1raw = api.coord([api.value(1), 0])[0];
      const idx = api.value(2);
      const cs = p.coordSys;
      const x0 = Math.max(x0raw, cs.x);
      const x1 = Math.min(x1raw, cs.x + cs.width);
      if (x1 <= x0) return { type: "group", children: [] };
      const bandY = cs.y + cs.height * 0.15;
      const bandH = cs.height * 0.7;
      const children: Record<string, unknown>[] = [
        {
          type: "rect",
          shape: { x: x0, y: bandY, width: x1 - x0, height: bandH },
          style: { fill: CATEGORY20[idx % CATEGORY20.length], opacity: 0.85 },
        },
      ];
      if (x1 - x0 > 28) {
        children.push({
          type: "text",
          style: {
            x: (x0 + x1) / 2,
            y: bandY + bandH / 2,
            text: codes[idx],
            align: "center",
            verticalAlign: "middle",
            fill: "#ffffff",
            fontSize: 10,
            fontWeight: 600,
          },
        });
      }
      return { type: "group", children };
    };

    const series: Record<string, unknown>[] = [
      {
        name: "Oil rate",
        type: "line",
        xAxisIndex: 0,
        yAxisIndex: 0,
        data: oilPts,
        showSymbol: false,
        lineStyle: { color: ACCENT, width: 2 },
        itemStyle: { color: ACCENT },
        markLine: { silent: true, symbol: "none", data: markData },
      },
      {
        name: "Formation water",
        type: "line",
        xAxisIndex: 0,
        yAxisIndex: 0,
        data: fwatPts,
        showSymbol: false,
        lineStyle: { color: SLATE, width: 2 },
        itemStyle: { color: SLATE },
      },
      {
        name: "BHP (daily)",
        type: "line",
        xAxisIndex: 0,
        yAxisIndex: 1,
        data: bhpDailyPts,
        showSymbol: false,
        lineStyle: { color: CRIMSON, width: 1 },
        itemStyle: { color: CRIMSON },
      },
      {
        name: "BHP (test)",
        type: "scatter",
        xAxisIndex: 0,
        yAxisIndex: 1,
        data: bhpTestPts,
        symbolSize: 4,
        itemStyle: { color: CRIMSON },
      },
      {
        name: "Pumps in hole",
        type: "custom",
        xAxisIndex: 1,
        yAxisIndex: 2,
        renderItem: renderBand,
        data: bandData,
        silent: true,
        tooltip: { show: false },
      },
    ];
    if (showPf) {
      series.splice(2, 0, {
        name: "PF pressure",
        type: "line",
        xAxisIndex: 0,
        yAxisIndex: 0,
        data: pfPts,
        showSymbol: false,
        lineStyle: { color: GOLD, width: 1.5, type: "dashed" },
        itemStyle: { color: GOLD },
      });
    }

    return houseOption({
      tooltip: { ...baseTooltip, trigger: "axis", axisPointer: { type: "cross" } },
      axisPointer: { link: [{ xAxisIndex: "all" }] },
      legend: { top: 0, right: 8, itemWidth: 16, textStyle: { fontSize: 11 } },
      grid: [
        { left: 64, right: 64, top: 30, bottom: "30%" },
        { left: 64, right: 64, top: "82%", bottom: 18 },
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
          axisLine: { lineStyle: { color: "#94a3b8" } },
        },
      ],
      yAxis: [
        {
          type: "value",
          gridIndex: 0,
          ...axis(showPf ? "Rate (BPD) / PF (psi)" : "Rate (BPD)"),
          nameGap: 44,
        },
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

  const chartRef = useEChart(option);

  if (well === "Custom") {
    return (
      <InfoNote>
        Pick a well in the sidebar to see its jet pump install history and production trend.
      </InfoNote>
    );
  }
  if (query.isError) {
    return <ErrorNote error={query.error} />;
  }
  if (!data) {
    return <Spinner label={`Loading JP history for ${well}`} />;
  }
  if (data.installs.length === 0) {
    return <InfoNote>No jet pump history recorded for {well}</InfoNote>;
  }

  return (
    <div className="space-y-4">
      <Section
        title={`JP History - ${well}`}
        actions={
          <div className="flex items-center gap-2">
            {data.source === "excel_fallback" && <Badge tone="fair">Excel fallback</Badge>}
            <Badge>{fmtNum(data.installs.length)} installs</Badge>
          </div>
        }
      >
        <div className="flex flex-wrap items-center gap-4">
          <label className="flex cursor-pointer items-center gap-2 text-xs text-slate-600">
            <input
              type="checkbox"
              checked={bhpFromZero}
              onChange={(e) => setBhpFromZero(e.target.checked)}
              className={CHECKBOX_CLS}
            />
            BHP axis from zero
          </label>
          <label className="flex cursor-pointer items-center gap-2 text-xs text-slate-600">
            <input
              type="checkbox"
              checked={showPf}
              onChange={(e) => setShowPf(e.target.checked)}
              className={CHECKBOX_CLS}
            />
            Show PF pressure
          </label>
        </div>
      </Section>

      {data.current_pump && (
        <p className="text-xs text-slate-600">
          Current pump: <span className="font-semibold">{data.current_pump}</span>
        </p>
      )}
      <Card>
        <div ref={chartRef} className="h-[470px]" />
      </Card>

      <Section title="Install history">
        <DataTable
          columns={INSTALL_COLUMNS}
          rows={data.installs}
          rowKey={(r, i) => `${r.date_set ?? "unset"}-${i}`}
        />
      </Section>
    </div>
  );
}
