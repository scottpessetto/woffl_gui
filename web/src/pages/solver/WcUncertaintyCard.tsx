import { ChevronDown, Loader2 } from "lucide-react";
import { useId, useMemo, useState } from "react";

import { useWcUncertainty } from "../../api/hooks";
import type { SimParams, WcMetricRange } from "../../api/types";
import { Button, Card, ErrorNote, HelpPopover } from "../../components/ui";
import { fmtNum } from "../../lib/format";
import { useDebounced } from "../../lib/useDebounced";

function RangeMetric({ label, unit, range }: { label: string; unit: string; range: WcMetricRange }) {
  return (
    <div className="min-w-0 rounded-md border border-slate-200 bg-white px-3 py-2.5">
      <div className="mb-2 text-xs font-medium text-slate-600">
        {label} <span className="font-normal text-slate-400">{unit}</span>
      </div>
      <dl className="grid grid-cols-3 gap-2 tabular-nums">
        {([ ["Lower", range.low], ["Base", range.base], ["Upper", range.high] ] as const).map(([name, value]) => (
          <div key={name} className={name === "Base" ? "text-blue-700" : "text-slate-700"}>
            <dt className="text-[10px] font-medium uppercase tracking-wide opacity-75">{name}</dt>
            <dd className="mt-0.5 text-base font-semibold">{value === null ? "-" : fmtNum(value)}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

/** On demand: a normal solve stays cheap, and hidden ranges do no extra work. */
export function WcUncertaintyCard({ well, params, enabled }: {
  well: string;
  params: SimParams;
  enabled: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [width, setWidth] = useState("5");
  const id = useId();
  const points = Number(width);
  const valid = width.trim() !== "" && Number.isFinite(points) && points >= 0 && points <= 100;
  const supported = !params.model_as_water && params.form_wc <= .99;
  const req = useMemo(() => ({ well, params, uncertainty_points: valid ? points : 0 }), [well, params, valid, points]);
  const debounced = useDebounced(req, 400);
  const settled = req === debounced;
  const active = open && enabled && supported && valid;
  const query = useWcUncertainty(debounced, active && settled);
  // Also hide results during the debounce, before the query key can change.
  const result = active && settled && !query.isFetching && !query.isError ? query.data : undefined;
  const busy = active && (!settled || query.isFetching || query.isPending);

  return (
    <Card padded={false}>
      <button
        type="button"
        aria-expanded={open}
        aria-controls={`${id}-body`}
        onClick={() => setOpen(!open)}
        className="flex w-full items-center justify-between gap-3 rounded-lg px-4 py-3 text-left hover:bg-slate-50 focus-visible:outline-2 focus-visible:outline-blue-500"
      >
        <span>
          <span className="block text-sm font-semibold text-slate-700">WC uncertainty</span>
          {!open && <span className="mt-0.5 block text-xs text-slate-500">Explore the oil and BHP range</span>}
        </span>
        <ChevronDown aria-hidden="true" className={`h-4 w-4 shrink-0 text-slate-400 transition-transform ${open ? "rotate-180" : ""}`} />
      </button>
      {open && (
        <div id={`${id}-body`} className="space-y-3 border-t border-slate-100 px-4 pb-4 pt-3">
          {!supported ? (
            <p className="text-xs text-slate-500">Available in oil mode with formation WC from 0% to 99%.</p>
          ) : (
            <>
              <div className="flex flex-wrap items-end justify-between gap-3">
                <div>
                  <label htmlFor={`${id}-width`} className="mb-1 block text-xs font-medium text-slate-600">Watercut uncertainty</label>
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-slate-500" aria-hidden="true">+/-</span>
                    <input
                      id={`${id}-width`}
                      type="number"
                      min={0}
                      max={100}
                      step="any"
                      value={width}
                      onChange={(e) => setWidth(e.target.value)}
                      aria-invalid={!valid}
                      aria-describedby={`${id}-hint`}
                      className="w-20 rounded-md border border-slate-300 bg-white px-2 py-1 text-sm tabular-nums text-slate-800 outline-none focus:border-blue-500 focus:ring-1 focus:ring-blue-200"
                    />
                    <span className="text-xs text-slate-500">percentage points</span>
                  </div>
                </div>
                <HelpPopover label="Assumptions" align="right" width="w-72" className="ml-auto">
                  <p>Varies formation WC around the current input. Total-liquid IPR, GOR, operating pressures, pump and loss coefficients stay fixed. Returned power fluid is excluded from WC.</p>
                  <p className="mt-2">Lower and upper values are the smallest and largest predictions across up to nine WC samples, including the base. Interior samples are included because the response can be nonlinear.</p>
                  <p className="mt-2">These are WC-only scenario ranges, not confidence intervals or a guarantee that all field uncertainty is covered. The starting +/-5 points is an example; set it to reflect your test uncertainty.</p>
                </HelpPopover>
              </div>
              <p id={`${id}-hint`} className="text-[11px] text-slate-500">
                At 80% WC, +/-5 points means 75% to 85% WC.
              </p>
              {!valid && <p role="alert" className="text-xs text-amber-700">Enter a value from 0 to 100 percentage points.</p>}
              {!enabled && <p className="text-xs text-slate-500">Run the well to calculate its WC range.</p>}
              {busy && (
                <div role="status" className="flex min-h-24 items-center justify-center gap-2 text-xs text-slate-500">
                  <Loader2 aria-hidden="true" className="h-4 w-4 animate-spin" /> Updating oil and BHP ranges
                </div>
              )}
              {active && settled && query.isError && (
                <div className="space-y-2">
                  <ErrorNote error={query.error} />
                  <Button size="sm" onClick={() => { void query.refetch(); }}>Retry ranges</Button>
                </div>
              )}
              {result && (
                <div className="space-y-2" aria-live="polite">
                  <div className="flex flex-wrap items-center justify-between gap-1 text-xs text-slate-500">
                    <span>WC {fmtNum(result.wc_low * 100, 1)}% to {fmtNum(result.wc_high * 100, 1)}%</span>
                    <span>Base {fmtNum(result.wc_base * 100, 1)}%</span>
                  </div>
                  {!result.complete && (
                    <p role="alert" className="rounded-md bg-amber-50 px-2.5 py-2 text-xs text-amber-800">
                      {result.solved_count === 0
                        ? "No WC scenario solved. Oil and BHP bounds are unavailable."
                        : `Incomplete range: ${result.sample_count - result.solved_count} of ${result.sample_count} WC scenarios did not solve. Bounds cover successful scenarios only.`}
                      {!result.base_solved && result.solved_count > 0 && " The base WC did not solve."}
                    </p>
                  )}
                  {result.oil && result.bhp && (
                    <div className="grid gap-2 [grid-template-columns:repeat(auto-fit,minmax(min(100%,210px),1fr))]">
                      <RangeMetric label="Oil" unit="BOPD" range={result.oil} />
                      <RangeMetric label="BHP" unit="psig" range={result.bhp} />
                    </div>
                  )}
                  <p className="text-[11px] leading-relaxed text-slate-500">
                    WC-only scenarios; total-liquid IPR and GOR held fixed. Not confidence intervals.
                    {result.clipped && " WC range limited to 0% to 99%."}
                  </p>
                  {!result.complete && (
                    <details className="text-xs text-slate-500">
                      <summary className="cursor-pointer">Unsolved WC scenarios</summary>
                      <ul className="mt-1 space-y-1 pl-3">
                        {result.points.filter((p) => p.error !== null).map((p) => (
                          <li key={p.wc}>{fmtNum(p.wc * 100, 2)}% WC: {p.error}</li>
                        ))}
                      </ul>
                    </details>
                  )}
                </div>
              )}
            </>
          )}
        </div>
      )}
    </Card>
  );
}
