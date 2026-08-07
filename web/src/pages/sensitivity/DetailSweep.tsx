/**
 * Detail sweep - the selected knob's whole swept range against the selected
 * match quantity, with the current case and the measured test as reference
 * lines. The pump knobs step through a parts catalog, so those plot on a
 * category axis of catalog codes; everything else plots on its own value
 * axis. Solves that failed are gaps in the line, never zeros - a zero here
 * would read as "the pump made nothing", which is not what happened.
 */

import { useMemo } from "react";

import type { SensitivityKnob } from "../../api/types";
import type { EChartsOption } from "../../charts/echarts";
import { ACCENT, axis, baseGrid, baseTooltip, CRIMSON, GOLD, houseOption, SLATE, ttHeader, ttRow } from "../../charts/theme";
import { ChartPanel } from "../../charts/ChartPanel";
import { Card } from "../../components/ui";
import { fmtNum } from "../../lib/format";
import { isCatalogKnob } from "./bounds";
import { type MetricSpec, pointMetric } from "./metrics";

/** The axis-pointer reading: a catalog code on the catalog knobs, the
 *  swept number everywhere else. */
function axisReading(raw: unknown): string | number | null {
  const list = Array.isArray(raw) ? raw : [raw];
  for (const p of list) {
    if (p !== null && typeof p === "object" && "axisValue" in p) {
      const v = p.axisValue;
      if (typeof v === "string" || typeof v === "number") return v;
    }
  }
  return null;
}

function sweepOption(
  knob: SensitivityKnob,
  spec: MetricSpec,
  baseline: number | null,
  target: number | null,
): EChartsOption | null {
  const catalog = isCatalogKnob(knob);
  const pts = catalog ? knob.points : [...knob.points].sort((a, b) => a.value - b.value);
  if (pts.length === 0) return null;

  const lines: Record<string, unknown>[] = [];
  if (baseline !== null) {
    lines.push({
      yAxis: baseline,
      lineStyle: { color: SLATE, width: 1, type: "dashed" },
      label: {
        show: true,
        formatter: `current ${fmtNum(baseline, spec.dp)} ${spec.unit}`,
        position: "insideStartTop",
        color: SLATE,
        fontSize: 11,
      },
    });
  }
  if (target !== null) {
    lines.push({
      yAxis: target,
      lineStyle: { color: GOLD, width: 1.5, type: "dashed" },
      label: {
        show: true,
        formatter: `test ${fmtNum(target, spec.dp)} ${spec.unit}`,
        position: "insideEndTop",
        color: GOLD,
        fontSize: 11,
      },
    });
  }

  const xName = knob.unit === "" ? knob.label : `${knob.label} (${knob.unit})`;

  return houseOption({
    grid: { ...baseGrid, right: 32, top: 34 },
    tooltip: {
      ...baseTooltip,
      trigger: "axis",
      formatter: (raw: unknown): string => {
        const reading = axisReading(raw);
        if (reading === null) return "";
        const pt =
          typeof reading === "string"
            ? pts.find((p) => p.label === reading)
            : pts.reduce((best, p) =>
                Math.abs(p.value - reading) < Math.abs(best.value - reading) ? p : best,
              );
        if (!pt) return "";
        const out = [ttHeader(`${knob.label} ${pt.label}${knob.unit === "" ? "" : ` ${knob.unit}`}`)];
        const v = pointMetric(pt, spec.id);
        if (v === null) {
          out.push(ttRow(CRIMSON, "Solve", pt.error ?? "failed"));
        } else {
          out.push(ttRow(ACCENT, spec.label, `${fmtNum(v, spec.dp)} ${spec.unit}`));
        }
        if (pt.sonic !== null) {
          out.push(ttRow(GOLD, "Throat", pt.sonic ? "sonic (choked)" : "subsonic"));
        }
        return out.join("");
      },
    },
    xAxis: catalog
      ? { type: "category", ...axis(xName), data: pts.map((p) => p.label) }
      : { type: "value", ...axis(xName), min: "dataMin", max: "dataMax" },
    yAxis: { type: "value", ...axis(spec.axisName), scale: true },
    series: [
      {
        name: spec.label,
        type: "line",
        symbol: "circle",
        symbolSize: 7,
        connectNulls: false,
        lineStyle: { width: 2, color: ACCENT },
        itemStyle: { color: ACCENT },
        data: catalog
          ? pts.map((p) => pointMetric(p, spec.id))
          : pts.map((p) => [p.value, pointMetric(p, spec.id)]),
      },
      {
        name: "",
        type: "line",
        data: [],
        silent: true,
        ...(lines.length > 0
          ? { markLine: { silent: true, symbol: "none", animation: false, data: lines } }
          : {}),
      },
    ],
  });
}

export function DetailSweep({
  knob,
  spec,
  baseline,
  target,
}: {
  knob: SensitivityKnob;
  spec: MetricSpec;
  baseline: number | null;
  target: number | null;
}) {
  const option = useMemo(
    () => sweepOption(knob, spec, baseline, target),
    [knob, spec, baseline, target],
  );
  const catalog = isCatalogKnob(knob);

  if (option === null) return null;

  return (
    <Card padded={false} className="p-2">
      <p className="px-2 pt-1 text-xs font-semibold text-slate-600" title={knob.basis}>
        {knob.label} Sweep, {spec.label}
      </p>
      <p className="px-2 text-[11px] text-slate-500">
        Current value {knob.baseline_label}
        {knob.unit === "" ? "" : ` ${knob.unit}`}
        {knob.inert ? ". Inert: no swept value moved any match quantity." : ""}
      </p>
      {/* Catalog codes have nothing to zoom along x; the value axis keeps the
          standard interaction on both. */}
      <ChartPanel
        option={option}
        height={360}
        zoom={catalog ? { xAxisIndex: "none", yAxisIndex: [0] } : { xAxisIndex: [0], yAxisIndex: [0] }}
      />
    </Card>
  );
}
