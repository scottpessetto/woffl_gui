/**
 * The numbers behind the tornado, and the place the ranges get set: each
 * knob's swept range as editable Low and High cells, plus its signed
 * low-case and high-case change in all four match quantities, in tornado
 * order. Clicking a row picks that input for the sweep chart - the
 * keyboard-reachable twin of clicking a bar, and the only way to reach an
 * inert knob, whose bar has no width to click.
 *
 * The default ranges are ours and defensible; they are not the engineer's.
 * They know which inputs are shaky on a given well ("GOR could be anywhere
 * from 150 to 300 on this one, ResP is solid"), so every range is typed over
 * here. Edits are clamped to the limits the model itself enforces, so the
 * sweep can never propose a value the sidebar would refuse.
 */

import clsx from "clsx";
import type { KeyboardEvent } from "react";
import { useMemo, useState } from "react";

import type { SensitivityKnob } from "../../api/types";
import { Button, type Column, DataTable, Section } from "../../components/ui";
import { type BoundsMap, effectiveRange, fmtBound, isCatalogKnob, knobStep } from "./bounds";
import { excursion, type MetricId, METRICS, signed } from "./metrics";

/** Committed by an edit; the page holds the map and debounces the refetch. */
type BoundHandler = (knob: SensitivityKnob, side: "low" | "high", value: number) => void;

interface KnobRow {
  id: string;
  knob: string;
  inert: boolean;
  overridden: boolean;
  basis: string;
  baseline: string;
  low: number;
  high: number;
  psu: string;
  qoil: string;
  qliq: string;
  qpf: string;
  /** the knob itself, so a cell can render its own editor */
  knobRef: SensitivityKnob;
  [key: string]: unknown;
}

const DELTA_HELP = "Low case / high case, measured from the current sidebar case.";
const BOUND_HELP =
  "Editable. Set the range you actually believe on this well and the sweep re-runs. " +
  "Values are clamped to the limits the model enforces.";

// The house input, at table-row height. Same tokens as RunPanel/ParamFields.
const INPUT_CLS =
  "h-7 w-24 rounded-md border border-slate-300 bg-white px-2 text-sm tabular-nums " +
  "text-slate-800 outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-200";
const SELECT_CLS =
  "h-7 w-20 rounded-md border border-slate-300 bg-white px-1.5 text-sm " +
  "text-slate-800 outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-200";
// An overridden range has to be obvious: it is the engineer's, not ours. A
// ring rather than a second border/background colour, so the marker does not
// depend on which of two same-property utilities Tailwind emits last.
const EDITED_CLS = "ring-2 ring-amber-300";

/** One end of one input's range, as a number input or a catalog picker. */
function BoundInput({
  knob,
  side,
  value,
  edited,
  onCommit,
}: {
  knob: SensitivityKnob;
  side: "low" | "high";
  /** the range end currently in effect, in the input's own units */
  value: number;
  edited: boolean;
  onCommit: BoundHandler;
}) {
  // The draft holds keystrokes verbatim while the box has focus, so typing
  // "0.4" into a three-decimal knob is not reformatted to "0.400" under the
  // cursor. Dropping it on blur brings back the canonical value, including
  // any clamp the page applied on the way in.
  const [draft, setDraft] = useState<string | null>(null);

  const limits =
    knob.clamp_low !== null && knob.clamp_high !== null
      ? ` Limits ${fmtBound(knob, knob.clamp_low)} to ${fmtBound(knob, knob.clamp_high)}.`
      : "";
  const title =
    `${side === "low" ? "Low" : "High"} end of the ${knob.label} sweep.${limits}` +
    ` Last swept: ${fmtBound(knob, knob.swept_low)} to ${fmtBound(knob, knob.swept_high)}.`;

  if (isCatalogKnob(knob)) {
    // Pump identity walks a parts catalog, so the bound is an option index.
    return (
      <select
        value={String(Math.round(value))}
        title={title}
        onChange={(e) => onCommit(knob, side, Number(e.target.value))}
        className={clsx(SELECT_CLS, edited && EDITED_CLS)}
      >
        {(knob.options ?? []).map((option, index) => (
          <option key={option} value={index}>
            {option}
          </option>
        ))}
      </select>
    );
  }

  const onKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Escape") {
      setDraft(null);
      e.currentTarget.blur();
    } else if (e.key === "Enter") {
      e.currentTarget.blur();
    }
  };

  return (
    <input
      type="number"
      value={draft ?? fmtBound(knob, value)}
      min={knob.clamp_low ?? undefined}
      max={knob.clamp_high ?? undefined}
      step={knobStep(knob)}
      title={title}
      onChange={(e) => {
        setDraft(e.target.value);
        const parsed = Number(e.target.value);
        if (e.target.value.trim() !== "" && Number.isFinite(parsed)) {
          onCommit(knob, side, parsed);
        }
      }}
      onBlur={() => setDraft(null)}
      onKeyDown={onKeyDown}
      className={clsx(INPUT_CLS, edited && EDITED_CLS)}
    />
  );
}

/** `metric` deltas as "low / high"; a side that never solved reads "-". */
function deltaCell(knob: SensitivityKnob, metric: MetricId): string {
  return `${signed(excursion(knob.low, metric))} / ${signed(excursion(knob.high, metric))}`;
}

export function KnobTable({
  knobs,
  bounds,
  selectedId,
  onSelect,
  onBound,
  onResetBounds,
}: {
  /** in tornado order, so the table and the chart read as one thing */
  knobs: SensitivityKnob[];
  /** the engineer's overrides; a missing id is that input's default range */
  bounds: BoundsMap;
  selectedId: string | null;
  onSelect: (id: string) => void;
  onBound: BoundHandler;
  onResetBounds: () => void;
}) {
  const rows = useMemo<KnobRow[]>(
    () =>
      knobs.map((k) => {
        const range = effectiveRange(k, bounds);
        return {
          id: k.id,
          knob: k.label,
          inert: k.inert,
          basis: k.basis,
          baseline: k.baseline_label,
          // Either side of the seam counts: an override we have sent, or one
          // the server says it applied.
          overridden: k.overridden || k.id in bounds,
          low: range.low,
          high: range.high,
          psu: deltaCell(k, "psu"),
          qoil: deltaCell(k, "qoil"),
          qliq: deltaCell(k, "qliq"),
          qpf: deltaCell(k, "qpf"),
          knobRef: k,
        };
      }),
    [knobs, bounds],
  );

  const columns = useMemo<Column<KnobRow>[]>(() => {
    const boundCell = (row: KnobRow, side: "low" | "high") => (
      <span className="inline-flex items-center justify-end gap-1">
        <BoundInput
          knob={row.knobRef}
          side={side}
          value={side === "low" ? row.low : row.high}
          edited={row.overridden}
          onCommit={onBound}
        />
        {row.knobRef.unit !== "" && !isCatalogKnob(row.knobRef) && (
          <span className="text-[11px] text-slate-400">{row.knobRef.unit}</span>
        )}
      </span>
    );
    return [
      {
        key: "knob",
        label: "Input",
        render: (r) => (
          <span className="inline-flex items-center gap-1.5">
            {r.knob}
            {r.inert && (
              <span className="rounded bg-slate-100 px-1 text-[10px] font-medium text-slate-500">
                inert
              </span>
            )}
            {r.overridden && (
              <span className="rounded bg-amber-100 px-1 text-[10px] font-medium text-amber-700">
                edited
              </span>
            )}
          </span>
        ),
      },
      { key: "baseline", label: "Baseline", align: "right" },
      {
        key: "low",
        label: "Low",
        align: "right",
        help: BOUND_HELP,
        render: (r) => boundCell(r, "low"),
      },
      {
        key: "high",
        label: "High",
        align: "right",
        help: BOUND_HELP,
        render: (r) => boundCell(r, "high"),
      },
      ...METRICS.map((m) => ({
        key: m.id,
        label: `${m.label} (${m.unit})`,
        align: "right" as const,
        help: DELTA_HELP,
      })),
      { key: "basis", label: "Range basis" },
    ];
  }, [onBound]);

  return (
    <Section
      title="Input ranges and deltas"
      actions={
        <Button
          size="sm"
          onClick={onResetBounds}
          disabled={Object.keys(bounds).length === 0}
          title="Clear every range you have set and go back to the default sweep."
        >
          Reset to defaults
        </Button>
      }
    >
      <DataTable
        columns={columns}
        rows={rows}
        rowKey={(r) => r.id}
        highlightRow={(r) => r.id === selectedId}
        onRowClick={(r) => onSelect(r.id)}
        maxHeight="32rem"
        pinFirst
      />
    </Section>
  );
}
