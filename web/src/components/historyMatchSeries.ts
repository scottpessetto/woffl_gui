import type { PumpMatchResult, PumpMatchRow } from "../api/types";

const DAY = 86_400_000;
export type MatchQuantity = "bhp" | "oil" | "pf";

/** Tracker timestamps without an offset follow the server's UTC convention. */
export function historyTimestamp(value: string): number {
  const normalized = value.replace(" ", "T");
  return Date.parse(normalized.includes("T") && !/(Z|[+-]\d{2}:?\d{2})$/i.test(normalized)
    ? `${normalized}Z` : normalized);
}

/** A gap or another installation must never become a continuous model line. */
export function matchLines(result: PumpMatchResult, quantity: MatchQuantity, color: string,
  xAxisIndex: number, yAxisIndex: number, selectedEra: string | null = null) {
  const series: Record<string, unknown>[] = [];
  for (const era of result.eras) {
    if (selectedEra && era.installation_id !== selectedEra) continue;
    const rows = result.rows.filter((r) => r.installation_id === era.installation_id)
      .sort((a, b) => a.date.localeCompare(b.date));
    for (const phase of ["replay", "fit", "prediction"] as const) {
      if (!rows.some((r) => r.status === phase)) continue;
      const data: { value: [number, number | null]; name: string }[] = [];
      let last: number | null = null;
      for (const row of rows) {
        const x = Date.parse(row.date);
        if (last !== null && x - last > 45 * DAY) data.push({ value: [last + 1, null], name: "" });
        const value = row.status === phase ? row[`predicted_${quantity}`] : null;
        data.push({ value: [x, value], name: row.wt_uid });
        last = x;
      }
      series.push({ name: `${quantity === "bhp" ? "BHP" : quantity === "pf" ? "PF rate" : "Oil"} ${phase === "replay" ? "model" : phase}`,
        type: "line", xAxisIndex, yAxisIndex, data, connectNulls: false,
        showSymbol: true, symbol: phase === "fit" ? "diamond" : "circle", symbolSize: 5,
        lineStyle: { color, width: 2.4, type: phase === "fit" ? "dotted" : "dashed" }, itemStyle: { color }, z: 9 });
    }
  }
  return series;
}

/** Only show a replay datum at its own test date, including a failure/gap. */
export function matchAt(rows: PumpMatchRow[], x: number): PumpMatchRow | undefined {
  return rows.find((r) => Math.abs(Date.parse(r.date) - x) < DAY / 2);
}
