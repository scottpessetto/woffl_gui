const nf = new Map<number, Intl.NumberFormat>();

function formatter(dp: number): Intl.NumberFormat {
  let f = nf.get(dp);
  if (!f) {
    f = new Intl.NumberFormat("en-US", {
      minimumFractionDigits: dp,
      maximumFractionDigits: dp,
    });
    nf.set(dp, f);
  }
  return f;
}

/** Locale number with fixed decimals; em-dash-free placeholder for nulls. */
export function fmtNum(value: number | null | undefined, dp = 0, fallback = "-"): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return fallback;
  return formatter(dp).format(value);
}

export function fmtSigned(value: number | null | undefined, dp = 0): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "-";
  const s = fmtNum(Math.abs(value), dp);
  return value >= 0 ? `+${s}` : `-${s}`;
}

export function fmtPct(fraction: number | null | undefined, dp = 1): string {
  if (fraction === null || fraction === undefined || !Number.isFinite(fraction)) return "-";
  return `${fmtNum(fraction * 100, dp)}%`;
}

/** YYYY-MM-DD from an ISO string or Date. */
export function fmtDate(value: string | Date | null | undefined): string {
  if (!value) return "-";
  const d = typeof value === "string" ? new Date(value) : value;
  if (Number.isNaN(d.getTime())) return typeof value === "string" ? value.slice(0, 10) : "-";
  return d.toISOString().slice(0, 10);
}

export function daysAgo(value: string | null | undefined): number | null {
  if (!value) return null;
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return null;
  return Math.max(0, Math.round((Date.now() - d.getTime()) / 86_400_000));
}

/** "12B" pump code. */
export const pumpCode = (nozzle: string | null | undefined, throat: string | null | undefined): string =>
  `${nozzle ?? "?"}${throat ?? "?"}`;
