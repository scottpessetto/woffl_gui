/**
 * Well Sort shared bits: column factories (ports of st.column_config),
 * decision badges, POPs configuration controls, and small inputs reused by
 * the three views. All percent-like values arrive as FRACTIONS from the API
 * and are scaled x100 for display only - CSV exports keep raw fractions,
 * matching the old Streamlit downloads.
 */

import { RotateCcw, Search, X } from "lucide-react";
import { useMemo, useState } from "react";
import type { ReactNode } from "react";

import type { TriageOnlineCode, TriageShutCode } from "../../api/types";
import { Badge, type Column } from "../../components/ui";
import { fmtNum, fmtSigned } from "../../lib/format";

// ---------------------------------------------------------------------------
// Column factories
// ---------------------------------------------------------------------------

type AnyRow = Record<string, unknown>;

const asNum = (v: unknown): number | null =>
  typeof v === "number" && Number.isFinite(v) ? v : null;

/** Right-aligned numeric column, `dp` decimals. */
export function num<R extends AnyRow>(key: string, label: string, dp = 0, help?: string): Column<R> {
  return {
    key,
    label,
    align: "right",
    help,
    render: (r) => fmtNum(asNum(r[key]), dp),
  };
}

/** Fraction (0-1) shown as percent with `dp` decimals. */
export function pct<R extends AnyRow>(key: string, label: string, dp = 1, help?: string): Column<R> {
  return {
    key,
    label,
    align: "right",
    help,
    render: (r) => {
      const v = asNum(r[key]);
      return v === null ? "-" : fmtNum(v * 100, dp);
    },
  };
}

/** Signed fraction delta shown as percent (Oil dev, WC-vs-marginal). */
export function pctSigned<R extends AnyRow>(key: string, label: string, dp = 0, help?: string): Column<R> {
  return {
    key,
    label,
    align: "right",
    help,
    render: (r) => {
      const v = asNum(r[key]);
      return v === null ? "-" : fmtSigned(v * 100, dp);
    },
  };
}

/** Plain text column. */
export function txt<R extends AnyRow>(key: string, label: string, help?: string): Column<R> {
  return { key, label, help };
}

/** Boolean flag: dimmed "-" when false so flags pop only when set. */
export function flag<R extends AnyRow>(key: string, label: string, help?: string): Column<R> {
  return {
    key,
    label,
    align: "center",
    help,
    render: (r) =>
      r[key] ? (
        <span className="font-semibold text-amber-600">yes</span>
      ) : (
        <span className="text-slate-300">-</span>
      ),
  };
}

// ---------------------------------------------------------------------------
// Decision badges (replaces the emoji color coding of the Streamlit table)
// ---------------------------------------------------------------------------

type BadgeTone = "neutral" | "good" | "fair" | "poor" | "info";

const DECISIONS: Record<TriageOnlineCode | TriageShutCode, { label: string; tone: BadgeTone }> = {
  si: { label: "SI candidate", tone: "poor" },
  verify_si: { label: "Verify before SI", tone: "fair" },
  verify_stale: { label: "Verify - stale/no test", tone: "fair" },
  keep: { label: "Keep online", tone: "good" },
  pops: { label: "POPS (own handling)", tone: "neutral" },
  bol: { label: "BOL candidate", tone: "good" },
  bol_trial: { label: "BOL trial", tone: "info" },
  verify_form_hist: { label: "Verify - form-basis history", tone: "fair" },
  verify_no_test: { label: "Verify - no test", tone: "fair" },
  leave_shut: { label: "Leave shut", tone: "neutral" },
};

export function DecisionBadge({ code }: { code: TriageOnlineCode | TriageShutCode }) {
  const d = DECISIONS[code] ?? { label: code, tone: "neutral" as BadgeTone };
  return <Badge tone={d.tone}>{d.label}</Badge>;
}

export function decisionLabel(code: string): string {
  return DECISIONS[code as TriageOnlineCode | TriageShutCode]?.label ?? code;
}

// ---------------------------------------------------------------------------
// POPs configuration controls
// ---------------------------------------------------------------------------

/** Toggleable chip row - the pads-with-separation selector. */
export function ChipToggles({
  options,
  selected,
  onChange,
  title,
}: {
  options: string[];
  selected: string[];
  onChange: (next: string[]) => void;
  title?: string;
}) {
  const sel = new Set(selected);
  return (
    <div className="flex flex-wrap gap-1" title={title}>
      {options.map((opt) => {
        const on = sel.has(opt);
        return (
          <button
            key={opt}
            type="button"
            onClick={() => {
              const next = new Set(sel);
              if (on) next.delete(opt);
              else next.add(opt);
              onChange([...next].sort());
            }}
            className={
              on
                ? "rounded-md border border-blue-600 bg-blue-600 px-2 py-0.5 text-xs font-medium text-white"
                : "rounded-md border border-slate-300 bg-white px-2 py-0.5 text-xs text-slate-600 hover:border-slate-400"
            }
          >
            {opt}
          </button>
        );
      })}
    </div>
  );
}

/** Searchable multi-picker for the per-well PopsPad=True overrides. */
export function WellPicker({
  options,
  selected,
  onChange,
  placeholder = "Add well...",
}: {
  options: string[];
  selected: string[];
  onChange: (next: string[]) => void;
  placeholder?: string;
}) {
  const [query, setQuery] = useState("");
  const matches = useMemo(() => {
    if (query.trim().length === 0) return [];
    const q = query.trim().toUpperCase();
    const sel = new Set(selected);
    return options.filter((w) => w.toUpperCase().includes(q) && !sel.has(w)).slice(0, 8);
  }, [query, options, selected]);

  return (
    <div className="space-y-1.5">
      <div className="flex flex-wrap items-center gap-1">
        {selected.map((w) => (
          <span
            key={w}
            className="inline-flex items-center gap-1 rounded-md bg-slate-100 px-2 py-0.5 text-xs font-medium text-slate-700"
          >
            {w}
            <button
              type="button"
              aria-label={`Remove ${w}`}
              onClick={() => onChange(selected.filter((s) => s !== w))}
              className="rounded p-0.5 text-slate-400 hover:text-slate-700"
            >
              <X className="h-3 w-3" />
            </button>
          </span>
        ))}
        {selected.length === 0 && <span className="text-xs text-slate-400">none</span>}
      </div>
      <div className="relative">
        <Search className="pointer-events-none absolute left-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-slate-400" />
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={placeholder}
          className="h-7 w-52 rounded-md border border-slate-300 bg-white pl-7 pr-2 text-xs text-slate-800 outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-200"
        />
        {matches.length > 0 && (
          <div className="absolute z-20 mt-1 w-52 rounded-md border border-slate-200 bg-white py-1 shadow-lg">
            {matches.map((w) => (
              <button
                key={w}
                type="button"
                onClick={() => {
                  onChange([...selected, w].sort());
                  setQuery("");
                }}
                className="block w-full px-2.5 py-1 text-left text-xs text-slate-700 hover:bg-blue-50"
              >
                {w}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Small controls
// ---------------------------------------------------------------------------

/** Labeled range slider with live value readout. */
export function LabeledSlider({
  label,
  value,
  min,
  max,
  step,
  onChange,
  format = (v) => String(v),
  help,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
  help?: string;
}) {
  return (
    <label className="block min-w-44" title={help}>
      <span className="flex items-baseline justify-between text-xs text-slate-500">
        <span>{label}</span>
        <span className="font-medium tabular-nums text-slate-700">{format(value)}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="mt-1 w-full accent-blue-600"
      />
    </label>
  );
}

/** Compact toggle (checkbox + label). */
export function Toggle({
  label,
  checked,
  onChange,
  help,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  help?: string;
}) {
  return (
    <label className="inline-flex cursor-pointer items-center gap-1.5 text-xs text-slate-600" title={help}>
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="h-3.5 w-3.5 rounded border-slate-300 accent-blue-600"
      />
      {label}
    </label>
  );
}

/** Draft-and-commit number input (commit on blur/Enter). */
export function NumberBox({
  value,
  min,
  max,
  step = 1,
  onCommit,
  className,
}: {
  value: number;
  min: number;
  max: number;
  step?: number;
  onCommit: (v: number) => void;
  className?: string;
}) {
  const [draft, setDraft] = useState<string | null>(null);
  const commit = () => {
    if (draft === null) return;
    const v = Number(draft);
    if (Number.isFinite(v)) onCommit(Math.min(max, Math.max(min, v)));
    setDraft(null);
  };
  return (
    <input
      type="number"
      min={min}
      max={max}
      step={step}
      value={draft ?? String(value)}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === "Enter") (e.target as HTMLInputElement).blur();
      }}
      className={
        className ??
        "h-8 w-32 rounded-md border border-slate-300 bg-white px-2 text-sm tabular-nums text-slate-800 outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-200"
      }
    />
  );
}

/** Reset-to-preset button used by the pad pump-limit input. */
export function ResetButton({ onClick, title }: { onClick: () => void; title: string }) {
  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      className="inline-flex h-8 items-center gap-1 rounded-md border border-slate-300 bg-white px-2 text-xs text-slate-600 hover:border-slate-400"
    >
      <RotateCcw className="h-3 w-3" />
      preset
    </button>
  );
}

/** Section chrome for a sub-view header row. */
export function ControlRow({ children }: { children: ReactNode }) {
  return <div className="flex flex-wrap items-end gap-x-5 gap-y-3">{children}</div>;
}
