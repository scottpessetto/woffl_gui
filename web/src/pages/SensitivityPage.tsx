/**
 * Sensitivity view - what each calibration input actually does to the four
 * match quantities. The server sweeps every knob over a range around the
 * current sidebar value and reports the signed excursion; this page ranks
 * them (tornado), draws the selected knob's curve, and lists the numbers.
 * Read-only diagnostic: it changes no physics and writes nothing.
 *
 * The row that matters most is often the empty one. When the solve sits on
 * the choked-flow floor the solver returns the choked suction pressure
 * directly, so power-fluid pressure, kth, kdi and wellhead pressure come
 * back bit-identical across their whole range - inputs the engineer has been
 * turning for nothing. Inert rows stay on the chart for exactly that reason.
 *
 * The default ranges are ours; the believable ones are the engineer's. This
 * page owns the bounds map they type into the knob table, debounces it, and
 * hands the SAME map to the combined-permutations panel, so what was tuned
 * above is exactly what gets combined below. Overrides are per well and do
 * not travel: a range that is honest on one well is noise on the next. They
 * outlive the page as well - the map, the metric and the selected input sit
 * in the sensitivity store keyed by well, so a trip to the Solver and back
 * does not throw away what the engineer just typed.
 *
 * Reference lines come from the newest well test in the sidebar lookback
 * window, so the tornado can be read as "can this input close the gap".
 */

import { useCallback, useMemo, useRef, useState } from "react";

import { useSensitivity, useWellTests } from "../api/hooks";
import type { SensitivityKnob, SensitivityResponse, WellTestRow } from "../api/types";
import { ErrorNote, InfoNote, WarnNote } from "../components/ui";
import { fmtDate } from "../lib/format";
import { useDebounced } from "../lib/useDebounced";
import { effectiveParams, useParamsStore } from "../state/params";
import { DEFAULT_VIEW, NO_BOUNDS, useSensitivityStore } from "../state/sensitivity";
import { withBound } from "./sensitivity/bounds";
import { testKey } from "./solver/selection";
import { CombinePanel } from "./sensitivity/CombinePanel";
import { DetailSweep } from "./sensitivity/DetailSweep";
import { KnobTable } from "./sensitivity/KnobTable";
import { type MetricId, METRICS, pointMetric, targetFor, tornadoRows } from "./sensitivity/metrics";
import { TornadoChart } from "./sensitivity/TornadoChart";

/**
 * Notes that are about the engineer's own ranges - a clamp applied to what
 * they typed, an override for an input that does not exist - rather than about
 * the well. Those are answers to something they just did, so they get a
 * warning box instead of the grey footnote line the standing notes use.
 */
const RANGE_NOTE = /overrid|clamp|ignor/i;

export default function SensitivityPage() {
  const well = useParamsStore((s) => s.well);
  const params = useParamsStore((s) => s.params);
  const simActive = useParamsStore((s) => s.simActive);
  const months = useParamsStore((s) => s.months);
  const cap = useParamsStore((s) => s.cap);

  // Per well, and it survives the page: the store keys everything by well,
  // so a range that is a judgement about MPI-31 is never swept on the next
  // well, and a walk over to the Solver does not lose it either.
  const bounds = useSensitivityStore((s) => s.bounds[well] ?? NO_BOUNDS);
  const view = useSensitivityStore((s) => s.view[well] ?? DEFAULT_VIEW);
  const setBounds = useSensitivityStore((s) => s.setBounds);
  const resetBounds = useSensitivityStore((s) => s.resetBounds);
  const setView = useSensitivityStore((s) => s.setView);
  const selectedId = view.selectedId;
  // Every bounds edit is a new query key and the sweep takes about a second,
  // so `query.data` goes undefined while the server works. Holding the last
  // response keeps the editor mounted through the refetch - an input that
  // unmounts mid-edit takes the cursor with it.
  const held = useRef<SensitivityResponse | null>(null);
  // The well the held response belongs to. Adjusted during render, not in an
  // effect: one well's numbers may not stand in for another's while it loads.
  const [heldWell, setHeldWell] = useState(well);
  if (heldWell !== well) {
    setHeldWell(well);
    held.current = null;
  }

  const effective = useMemo(() => effectiveParams(params), [params]);
  // About 90 solves a run, so a longer leash than the Solver's 400 ms: a
  // sidebar edit should not queue a sweep per keystroke.
  const debounced = useDebounced(effective, 800);
  // Typing a bound is a deliberate act on a settled case, so it gets the
  // standard 400 ms rather than the sidebar's 800. Debounced WITH its well:
  // the switch is instant but the debounced copy lags it, and one well's
  // overrides must never ride a well change into the other's request.
  const pending = useMemo(() => ({ well, map: bounds }), [well, bounds]);
  const settled = useDebounced(pending, 400);
  const sentBounds = settled.well === well ? settled.map : NO_BOUNDS;

  const testsQ = useWellTests(well, months, cap);
  // The Solver's comparison test when it has published one for this well
  // (the IPR anchor's test, or the engineer's explicit pick), else the most
  // recent test. Scoring "Match Sensitivities" against a different test than
  // the one on the Solver page was a silent mismatch (review WEB-15).
  const compareKey = useSensitivityStore((s) => s.compareKey[well] ?? null);
  const latestTest = useMemo<WellTestRow | null>(() => {
    const rows = testsQ.data?.tests ?? [];
    if (rows.length === 0) return null;
    if (compareKey !== null) {
      const picked = rows.find((t) => testKey(t) === compareKey);
      if (picked) return picked;
    }
    return [...rows].sort((a, b) => (a.date < b.date ? 1 : a.date > b.date ? -1 : 0))[0];
  }, [testsQ.data, compareKey]);
  const targetIsSolverPick = compareKey !== null && latestTest !== null && testKey(latestTest) === compareKey;

  const targets = useMemo(
    () => ({
      target_psu: latestTest?.bhp ?? null,
      target_qoil: latestTest?.oil ?? null,
      target_qliq: latestTest?.total_fluid ?? null,
      target_qpf: latestTest?.lift_wat ?? null,
    }),
    [latestTest],
  );

  const query = useSensitivity(well, debounced, targets, sentBounds, simActive);

  if (query.data !== undefined) held.current = query.data;
  const data: SensitivityResponse | null = query.data ?? held.current;

  // The stored metric is a loose string - storage is untrusted - so it is
  // resolved against the table rather than cast, and anything unrecognised
  // reads as the first metric.
  const spec = METRICS.find((m) => m.id === view.metricId) ?? METRICS[0];
  const metricId: MetricId = spec.id;
  const rows = useMemo(() => (data ? tornadoRows(data.knobs, metricId) : []), [data, metricId]);

  // Knobs in tornado order, so the table and the chart read as one thing.
  const ordered = useMemo<SensitivityKnob[]>(() => {
    if (!data) return [];
    const byId: Record<string, SensitivityKnob | undefined> = {};
    for (const k of data.knobs) byId[k.id] = k;
    const out: SensitivityKnob[] = [];
    for (const r of rows) {
      const k = byId[r.id];
      if (k !== undefined) out.push(k);
    }
    return out;
  }, [data, rows]);

  // The current map comes off the store, not the render closure: two quick
  // edits in a row must both land, and an identity that only moves with the
  // well keeps the table's memoized columns from rebuilding per keystroke.
  const onBound = useCallback(
    (knob: SensitivityKnob, side: "low" | "high", value: number) => {
      const cur = useSensitivityStore.getState().bounds[well] ?? NO_BOUNDS;
      setBounds(well, withBound(cur, knob, side, value));
    },
    [well, setBounds],
  );
  const onResetBounds = useCallback(() => resetBounds(well), [well, resetBounds]);
  const onSelect = useCallback(
    (id: string) => {
      const cur = useSensitivityStore.getState().view[well] ?? DEFAULT_VIEW;
      setView(well, { ...cur, selectedId: id });
    },
    [well, setView],
  );

  if (!simActive) {
    return (
      <InfoNote>
        Select a well in the sidebar, or press Run with Custom inputs, to sweep the
        calibration inputs.
      </InfoNote>
    );
  }
  // A range the server refuses should not take the page down with it, so the
  // error rides above the last good sweep when there is one.
  if (query.isError && data === null) return <ErrorNote error={query.error} />;
  // Loading: render nothing. A placeholder box pretending to be a chart is
  // worse than an empty column.
  if (data === null) return null;

  // Nothing picked yet: the biggest mover is the one worth looking at.
  const selected: SensitivityKnob | null =
    ordered.find((k) => k.id === selectedId) ?? (ordered.length > 0 ? ordered[0] : null);
  const baseline = pointMetric(data.baseline, metricId);
  const target = targetFor(data, metricId);
  const editedCount = Object.keys(bounds).length;
  const rangeNotes = data.notes.filter((n) => RANGE_NOTE.test(n));
  const otherNotes = data.notes.filter((n) => !RANGE_NOTE.test(n));

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex gap-1 rounded-lg border border-slate-200 bg-white p-1 w-fit">
          {METRICS.map((m) => (
            <button
              key={m.id}
              type="button"
              onClick={() => setView(well, { metricId: m.id, selectedId })}
              className={
                metricId === m.id
                  ? "rounded-md bg-blue-600 px-3 py-1 text-sm font-medium text-white"
                  : "rounded-md px-3 py-1 text-sm text-slate-600 hover:bg-slate-100"
              }
            >
              {m.label}
            </button>
          ))}
        </div>
        {latestTest !== null && (
          <p className="text-xs text-slate-500">
            Test reference: {fmtDate(latestTest.date)}{" "}
            {targetIsSolverPick ? "(the Solver's comparison test)" : "(most recent test)"}
            {target === null ? ` (no measured ${spec.label.toLowerCase()})` : ""}
          </p>
        )}
        {editedCount > 0 && (
          <p className="text-xs font-medium text-amber-700">
            {editedCount === 1 ? "1 input swept" : `${editedCount} inputs swept`} over a range you
            set, not the default
          </p>
        )}
        {query.isFetching && <p className="text-xs text-slate-400">Sweeping</p>}
      </div>

      {query.isError && <ErrorNote error={query.error} />}

      {rangeNotes.length > 0 && (
        <WarnNote>
          <ul className="list-disc space-y-0.5 pl-4">
            {rangeNotes.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        </WarnNote>
      )}

      {otherNotes.map((note) => (
        <p key={note} className="px-1 text-xs text-slate-500">
          {note}
        </p>
      ))}

      <div className="grid items-start gap-4 lg:grid-cols-2">
        <TornadoChart
          rows={rows}
          spec={spec}
          baseline={baseline}
          target={target}
          selectedId={selected !== null ? selected.id : null}
          onSelect={onSelect}
        />
        {selected !== null && (
          <DetailSweep knob={selected} spec={spec} baseline={baseline} target={target} />
        )}
      </div>

      <KnobTable
        knobs={ordered}
        bounds={bounds}
        selectedId={selected !== null ? selected.id : null}
        onSelect={onSelect}
        onBound={onBound}
        onResetBounds={onResetBounds}
      />

      {/* combine panel mounts here - the factorial study over the SAME edited
          bounds, answering whether any combination reaches the measured test */}
      <CombinePanel
        well={well}
        params={debounced}
        knobs={data.knobs}
        bounds={bounds}
        targets={targets}
      />
    </div>
  );
}
