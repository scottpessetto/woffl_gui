/**
 * Single-click CSV export: build the string client-side and trigger a Blob
 * download - no server round-trip, no two-step "prepare then download".
 */

export interface CsvColumn {
  key: string;
  label: string;
}

/** RFC-4180 style cell: quote when the value contains a comma/quote/newline. */
function csvCell(value: unknown): string {
  if (value === null || value === undefined) return "";
  let s: string;
  if (typeof value === "number") {
    s = Number.isFinite(value) ? String(value) : "";
  } else if (typeof value === "boolean") {
    s = value ? "True" : "False"; // matches the old pandas to_csv exports
  } else {
    s = String(value);
  }
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

export function downloadCsv(
  filename: string,
  columns: CsvColumn[],
  rows: Record<string, unknown>[],
): void {
  const header = columns.map((c) => csvCell(c.label)).join(",");
  const body = rows.map((row) => columns.map((c) => csvCell(row[c.key])).join(","));
  const csv = `${header}\r\n${body.join("\r\n")}\r\n`;
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
