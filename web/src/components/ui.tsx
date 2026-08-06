/**
 * Shared UI primitives. Dense, light, professional. All numeric cells use
 * tabular-nums so columns align. Keep these dumb: no data fetching here.
 */

import clsx from "clsx";
import { ChevronDown, ChevronUp, Loader2 } from "lucide-react";
import { useMemo, useState } from "react";
import type { ReactNode } from "react";

import { ApiError } from "../api/client";

export function Card({
  children,
  className,
  padded = true,
}: {
  children: ReactNode;
  className?: string;
  padded?: boolean;
}) {
  return (
    <div
      className={clsx(
        "rounded-lg border border-slate-200 bg-white shadow-[0_1px_2px_rgba(15,23,42,0.05)]",
        padded && "p-4",
        className,
      )}
    >
      {children}
    </div>
  );
}

export function Section({
  title,
  actions,
  children,
  className,
}: {
  title: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={className}>
      <div className="mb-2 flex items-center justify-between gap-2">
        <h3 className="text-sm font-semibold tracking-tight text-slate-700">{title}</h3>
        {actions}
      </div>
      {children}
    </section>
  );
}

type ButtonVariant = "primary" | "secondary" | "ghost" | "danger";

const BUTTON_STYLES: Record<ButtonVariant, string> = {
  primary:
    "bg-blue-600 text-white hover:bg-blue-700 disabled:bg-slate-300 disabled:text-slate-500",
  secondary:
    "border border-slate-300 bg-white text-slate-700 hover:bg-slate-50 disabled:text-slate-400",
  ghost: "text-slate-600 hover:bg-slate-100 disabled:text-slate-400",
  danger: "bg-red-600 text-white hover:bg-red-700 disabled:bg-slate-300",
};

export function Button({
  children,
  onClick,
  variant = "secondary",
  size = "md",
  disabled,
  busy,
  title,
  className,
  type = "button",
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: ButtonVariant;
  size?: "sm" | "md";
  disabled?: boolean;
  busy?: boolean;
  title?: string;
  className?: string;
  type?: "button" | "submit";
}) {
  return (
    <button
      type={type}
      title={title}
      disabled={disabled || busy}
      onClick={onClick}
      className={clsx(
        "inline-flex items-center justify-center gap-1.5 rounded-md font-medium transition-colors",
        size === "sm" ? "px-2.5 py-1 text-xs" : "px-3.5 py-1.5 text-sm",
        BUTTON_STYLES[variant],
        className,
      )}
    >
      {busy && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
      {children}
    </button>
  );
}

type Tone = "neutral" | "good" | "fair" | "poor" | "info";

const BADGE_TONES: Record<Tone, string> = {
  neutral: "bg-slate-100 text-slate-600 border-slate-200",
  good: "bg-green-50 text-green-800 border-green-200",
  fair: "bg-amber-50 text-amber-800 border-amber-200",
  poor: "bg-red-50 text-red-800 border-red-200",
  info: "bg-blue-50 text-blue-800 border-blue-200",
};

export function Badge({ children, tone = "neutral", title }: { children: ReactNode; tone?: Tone; title?: string }) {
  return (
    <span
      title={title}
      className={clsx(
        "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-medium whitespace-nowrap",
        BADGE_TONES[tone],
      )}
    >
      {children}
    </span>
  );
}

export function Metric({
  label,
  value,
  sub,
  tone,
}: {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
  tone?: Tone;
}) {
  return (
    <div className="min-w-[9rem] rounded-lg border border-slate-200 bg-white px-3.5 py-2.5">
      <div className="text-[11px] font-medium tracking-wide text-slate-500 uppercase">{label}</div>
      <div
        className={clsx(
          "mt-0.5 text-xl font-semibold tabular-nums",
          tone === "good" && "text-green-700",
          tone === "fair" && "text-amber-700",
          tone === "poor" && "text-red-700",
          (!tone || tone === "neutral" || tone === "info") && "text-slate-800",
        )}
      >
        {value}
      </div>
      {sub && <div className="mt-0.5 text-[11px] text-slate-500">{sub}</div>}
    </div>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 py-8 justify-center text-slate-500">
      <Loader2 className="h-5 w-5 animate-spin" />
      {label && <span className="text-sm">{label}</span>}
    </div>
  );
}

export function ErrorNote({ error, className }: { error: unknown; className?: string }) {
  const message =
    error instanceof ApiError
      ? error.detail.message
      : error instanceof Error
        ? error.message
        : String(error);
  return (
    <div
      className={clsx(
        "rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800",
        className,
      )}
    >
      {message}
    </div>
  );
}

export function InfoNote({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div
      className={clsx(
        "rounded-md border border-blue-200 bg-blue-50 px-3 py-2 text-sm text-blue-900",
        className,
      )}
    >
      {children}
    </div>
  );
}

export function WarnNote({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div
      className={clsx(
        "rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900",
        className,
      )}
    >
      {children}
    </div>
  );
}

// ---------------------------------------------------------------------------
// DataTable
// ---------------------------------------------------------------------------

export interface Column<Row> {
  key: string;
  label: string;
  align?: "left" | "right" | "center";
  /** Render the cell; defaults to String(row[key] ?? "-"). */
  render?: (row: Row) => ReactNode;
  width?: string;
  /** Hover tooltip on the header (ports st.column_config help). */
  help?: string;
}

/** null/undefined sort last in either direction; numbers before strings. */
function compareCells(a: unknown, b: unknown): number {
  if (a === null || a === undefined) return b === null || b === undefined ? 0 : 1;
  if (b === null || b === undefined) return -1;
  if (typeof a === "number" && typeof b === "number") return a - b;
  if (typeof a === "boolean" && typeof b === "boolean") return Number(a) - Number(b);
  return String(a).localeCompare(String(b), undefined, { numeric: true });
}

export function DataTable<Row extends Record<string, unknown>>({
  columns,
  rows,
  rowKey,
  maxHeight = "24rem",
  highlightRow,
  onRowClick,
  emptyLabel = "No data",
  sortable = false,
  pinFirst = false,
}: {
  columns: Column<Row>[];
  rows: Row[];
  rowKey: (row: Row, index: number) => string | number;
  maxHeight?: string;
  highlightRow?: (row: Row) => boolean;
  onRowClick?: (row: Row) => void;
  emptyLabel?: string;
  /** Click a header to sort by that column (raw row values, toggles dir). */
  sortable?: boolean;
  /** Keep the first column visible under horizontal scroll (wide tables). */
  pinFirst?: boolean;
}) {
  const [sort, setSort] = useState<{ key: string; dir: 1 | -1 } | null>(null);

  const sorted = useMemo(() => {
    if (!sort) return rows;
    const { key, dir } = sort;
    return [...rows].sort((a, b) => dir * compareCells(a[key], b[key]));
  }, [rows, sort]);

  if (rows.length === 0) {
    return <div className="py-6 text-center text-sm text-slate-400">{emptyLabel}</div>;
  }
  const toggleSort = (key: string) =>
    setSort((s) => (s?.key === key ? (s.dir === 1 ? { key, dir: -1 } : null) : { key, dir: 1 }));

  return (
    <div className="overflow-auto rounded-md border border-slate-200" style={{ maxHeight }}>
      <table className="w-full border-collapse text-[13px]">
        <thead className="sticky top-0 z-10">
          <tr className="bg-slate-50">
            {columns.map((col, ci) => (
              <th
                key={col.key}
                style={col.width ? { width: col.width } : undefined}
                title={col.help}
                onClick={sortable ? () => toggleSort(col.key) : undefined}
                className={clsx(
                  "border-b border-slate-200 bg-slate-50 px-2.5 py-1.5 font-semibold text-slate-600 whitespace-nowrap",
                  col.align === "right" ? "text-right" : col.align === "center" ? "text-center" : "text-left",
                  sortable && "cursor-pointer select-none hover:text-slate-900",
                  pinFirst && ci === 0 && "sticky left-0 z-20 border-r border-r-slate-200",
                )}
              >
                <span className="inline-flex items-center gap-0.5">
                  {col.label}
                  {sortable && sort?.key === col.key && (
                    sort.dir === 1
                      ? <ChevronUp className="h-3 w-3 text-blue-600" />
                      : <ChevronDown className="h-3 w-3 text-blue-600" />
                  )}
                </span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((row, i) => {
            const lit = highlightRow?.(row) ?? false;
            return (
            <tr
              key={rowKey(row, i)}
              onClick={onRowClick ? () => onRowClick(row) : undefined}
              className={clsx(
                "border-b border-slate-100 last:border-b-0",
                lit ? "bg-blue-50/60" : i % 2 === 1 ? "bg-slate-50/40" : "bg-white",
                onRowClick && "cursor-pointer hover:bg-blue-50",
              )}
            >
              {columns.map((col, ci) => (
                <td
                  key={col.key}
                  className={clsx(
                    "px-2.5 py-1.5 whitespace-nowrap tabular-nums text-slate-700",
                    col.align === "right" ? "text-right" : col.align === "center" ? "text-center" : "text-left",
                    // Pinned cells need an OPAQUE bg (the stripe tints are
                    // alpha washes; scrolled columns would ghost through).
                    pinFirst && ci === 0 &&
                      clsx(
                        "sticky left-0 z-[1] border-r border-r-slate-200",
                        lit ? "bg-blue-50" : i % 2 === 1 ? "bg-slate-50" : "bg-white",
                      ),
                  )}
                >
                  {col.render ? col.render(row) : String(row[col.key] ?? "-")}
                </td>
              ))}
            </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
