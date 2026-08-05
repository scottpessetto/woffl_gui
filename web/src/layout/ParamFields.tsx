/**
 * Typed sidebar field primitives bound to useParamsStore. Every control is
 * compact (h-8 inputs, text-xs labels) so the 300px sidebar holds the full
 * parameter tree. Numeric commits clamp through store.set -> PARAM_BOUNDS.
 */

import clsx from "clsx";
import { Lock } from "lucide-react";
import { useState } from "react";
import type { KeyboardEvent, ReactNode } from "react";

import type { SimParams } from "../api/types";
import { useParamsStore } from "../state/params";

/** SimParams keys holding number (or nullable number) values. */
export type NumericParamKey = {
  [K in keyof SimParams]-?: NonNullable<SimParams[K]> extends number ? K : never;
}[keyof SimParams];

/** SimParams keys holding string (incl. union) values. */
export type StringParamKey = {
  [K in keyof SimParams]-?: SimParams[K] extends string ? K : never;
}[keyof SimParams];

/** SimParams keys holding boolean values. */
export type BoolParamKey = {
  [K in keyof SimParams]-?: SimParams[K] extends boolean ? K : never;
}[keyof SimParams];

/** SimParams keys holding string-list values (batch sweep chips). */
export type ListParamKey = "nozzle_batch_options" | "throat_batch_options";

/** Default tooltip for as-built locked fields (mirrors sidebar.as_built_lock_help). */
export const AS_BUILT_HINT = "As-built - from prop_hist via vw_prop_mech. Read-only.";

const INPUT_CLS =
  "h-8 w-full rounded-md border border-slate-300 bg-white px-2 text-sm tabular-nums " +
  "text-slate-800 outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-200";
const LOCKED_CLS = "cursor-not-allowed border-slate-200 bg-slate-50 text-slate-400";

function FieldLabel({
  text,
  locked,
  chip,
}: {
  text: string;
  locked?: boolean;
  chip?: ReactNode;
}) {
  return (
    <span className="flex items-center gap-1 text-xs text-slate-500">
      {locked && <Lock className="h-3 w-3 shrink-0 text-slate-400" />}
      <span className="truncate">{text}</span>
      {chip}
    </span>
  );
}

export function NumberField<K extends NumericParamKey>({
  label,
  field,
  step = 1,
  dp = 0,
  unit,
  locked = false,
  lockHint = AS_BUILT_HINT,
  chip,
}: {
  label: string;
  field: K;
  step?: number;
  /** display decimals (commit keeps whatever the user typed, clamped) */
  dp?: number;
  unit?: string;
  locked?: boolean;
  lockHint?: string;
  chip?: ReactNode;
}) {
  const value = useParamsStore((s) => s.params[field]);
  const set = useParamsStore((s) => s.set);
  const [draft, setDraft] = useState<string | null>(null);

  const shown =
    draft ?? (typeof value === "number" && Number.isFinite(value) ? value.toFixed(dp) : "");

  const commit = () => {
    if (draft !== null) {
      const parsed = Number(draft);
      if (draft.trim() !== "" && Number.isFinite(parsed)) {
        set(field, parsed as SimParams[K]); // store clamps to PARAM_BOUNDS
      }
    }
    setDraft(null);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      commit();
      e.currentTarget.blur();
    } else if (e.key === "Escape") {
      setDraft(null);
      e.currentTarget.blur();
    }
  };

  return (
    <label className="block">
      <FieldLabel text={label} locked={locked} chip={chip} />
      <div className="relative mt-1">
        <input
          type="number"
          step={step}
          value={shown}
          disabled={locked}
          title={locked ? lockHint : undefined}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commit}
          onKeyDown={onKeyDown}
          className={clsx(INPUT_CLS, unit && "pr-12", locked && LOCKED_CLS)}
        />
        {unit && (
          <span className="pointer-events-none absolute inset-y-0 right-2 flex items-center text-[11px] text-slate-400">
            {unit}
          </span>
        )}
      </div>
    </label>
  );
}

export function SelectField<K extends StringParamKey>({
  label,
  field,
  options,
  locked = false,
  lockHint = AS_BUILT_HINT,
}: {
  label: string;
  field: K;
  options: readonly string[];
  locked?: boolean;
  lockHint?: string;
}) {
  const value = useParamsStore((s) => s.params[field]);
  const set = useParamsStore((s) => s.set);
  return (
    <label className="block">
      <FieldLabel text={label} locked={locked} />
      <select
        value={value}
        disabled={locked}
        title={locked ? lockHint : undefined}
        onChange={(e) => set(field, e.target.value as SimParams[K])}
        className={clsx(INPUT_CLS, "mt-1", locked && LOCKED_CLS)}
      >
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    </label>
  );
}

export function RadioRow<K extends StringParamKey>({
  label,
  field,
  options,
}: {
  label?: string;
  field: K;
  options: readonly { value: SimParams[K]; label: string; hint?: string }[];
}) {
  const value = useParamsStore((s) => s.params[field]);
  const set = useParamsStore((s) => s.set);
  return (
    <div>
      {label && <FieldLabel text={label} />}
      <div className={clsx("flex rounded-md border border-slate-300 bg-slate-50 p-0.5", label && "mt-1")}>
        {options.map((o) => (
          <button
            key={o.value}
            type="button"
            title={o.hint}
            onClick={() => set(field, o.value)}
            className={clsx(
              "h-7 flex-1 rounded text-sm transition-colors",
              value === o.value
                ? "bg-white font-medium text-slate-800 shadow-sm"
                : "text-slate-500 hover:text-slate-700",
            )}
          >
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}

export function CheckboxField({
  label,
  field,
  hint,
}: {
  label: string;
  field: BoolParamKey;
  hint?: string;
}) {
  const value = useParamsStore((s) => s.params[field]);
  const set = useParamsStore((s) => s.set);
  return (
    <label className="flex cursor-pointer items-center gap-2" title={hint}>
      <input
        type="checkbox"
        checked={value}
        onChange={(e) => set(field, e.target.checked)}
        className="h-4 w-4 rounded border-slate-300 accent-blue-600"
      />
      <span className="text-xs text-slate-600">{label}</span>
    </label>
  );
}

export function MultiChipSelect({
  label,
  field,
  options,
}: {
  label: string;
  field: ListParamKey;
  options: readonly string[];
}) {
  const value = useParamsStore((s) => s.params[field]);
  const set = useParamsStore((s) => s.set);
  const selected = new Set(value);

  const toggle = (opt: string) => {
    // Rebuild from the canonical option order so chips never scramble.
    const next = options.filter((o) => (o === opt ? !selected.has(o) : selected.has(o)));
    set(field, next);
  };

  return (
    <div>
      <FieldLabel text={label} />
      <div className="mt-1 flex flex-wrap gap-1">
        {options.map((o) => {
          const on = selected.has(o);
          return (
            <button
              key={o}
              type="button"
              onClick={() => toggle(o)}
              className={clsx(
                "rounded-full border px-2 py-0.5 text-xs tabular-nums transition-colors",
                on
                  ? "border-blue-300 bg-blue-50 font-medium text-blue-700"
                  : "border-slate-200 bg-white text-slate-400 hover:border-slate-300 hover:text-slate-600",
              )}
            >
              {o}
            </button>
          );
        })}
      </div>
    </div>
  );
}
