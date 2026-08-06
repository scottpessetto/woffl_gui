/**
 * Memory-gauge upload control - port of the Solver's gauge expander
 * (jetpump_solver._render_memory_gauge_section). Lives in the IPR chart
 * card header: a quiet "Add gauge data" affordance that opens a compact
 * dropdown; once a gauge is loaded the button becomes an indigo chip with
 * the coverage count so the override is always visible while reading the
 * chart.
 *
 * All parsing/combination happens server-side (POST /gauge/parse); this
 * component only holds File handles and the combined result (state/gauge).
 * Session-only, like the Streamlit original.
 */

import { Activity, Trash2, Upload, X } from "lucide-react";
import { useRef, useState } from "react";

import { upload } from "../../api/client";
import type { GaugeParseResponse, WellTestRow } from "../../api/types";
import { Spinner } from "../../components/ui";
import { fmtNum } from "../../lib/format";
import { useGaugeStore } from "../../state/gauge";

export function GaugePanel({ well, tests }: { well: string; tests: WellTestRow[] }) {
  const gauge = useGaugeStore((s) => s.byWell[well]);
  const setGauge = useGaugeStore((s) => s.setGauge);
  const clearGauge = useGaugeStore((s) => s.clearGauge);

  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement | null>(null);

  const sendFiles = async (fileObjects: File[]) => {
    if (fileObjects.length === 0) {
      clearGauge(well);
      return;
    }
    setBusy(true);
    setErr(null);
    try {
      const form = new FormData();
      for (const f of fileObjects) form.append("files", f, f.name);
      const meta = await upload<GaugeParseResponse>("/gauge/parse", form);
      setGauge(well, fileObjects, meta);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const onPick = (picked: FileList | null) => {
    if (!picked || picked.length === 0) return;
    const existing = gauge?.fileObjects ?? [];
    const names = new Set(existing.map((f) => f.name));
    const added = [...picked].filter((f) => !names.has(f.name));
    void sendFiles([...existing, ...added]);
  };

  const matched = gauge
    ? tests.filter((t) => gauge.dailyByDate[t.date] !== undefined).length
    : 0;

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        title={
          gauge
            ? `Memory gauge active: ${gauge.meta.start_date} to ${gauge.meta.end_date}, ` +
              `${matched} of ${tests.length} tests get gauge BHP. Click to manage.`
            : "Upload memory-gauge XLSX files - gauge daily medians override test BHP " +
              "inside the covered window (for wells without a trustworthy BHP feed)."
        }
        className={
          "flex items-center gap-1 rounded px-1.5 py-0.5 text-xs font-medium transition-colors " +
          (gauge
            ? "bg-indigo-50 text-indigo-700 hover:bg-indigo-100"
            : "text-slate-400 hover:bg-slate-100 hover:text-slate-600")
        }
      >
        <Activity className="h-3.5 w-3.5" />
        {gauge ? `Gauge BHP on ${matched}/${tests.length} tests` : "Add gauge data"}
      </button>

      {open && (
        <div className="absolute left-0 top-7 z-20 w-80 rounded-md border border-slate-200 bg-white p-3 shadow-lg">
          <div className="mb-2 flex items-center justify-between">
            <p className="text-xs font-semibold text-slate-700">Memory gauge - {well}</p>
            <button
              type="button"
              className="text-slate-400 hover:text-slate-600"
              onClick={() => setOpen(false)}
              aria-label="Close"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>

          {gauge && (
            <ul className="mb-2 space-y-1">
              {gauge.meta.files.map((f) => (
                <li key={f.filename} className="flex items-center gap-2 text-xs text-slate-600">
                  <span className="flex-1 truncate" title={f.filename}>
                    {f.filename}
                  </span>
                  <span className="shrink-0 tabular-nums text-slate-400">
                    {f.start_date} to {f.end_date} | {fmtNum(f.pressure_min)}-{fmtNum(f.pressure_max)} psi
                  </span>
                  <button
                    type="button"
                    disabled={busy}
                    aria-label={`Remove ${f.filename}`}
                    className="shrink-0 text-slate-400 hover:text-red-600 disabled:opacity-50"
                    onClick={() =>
                      void sendFiles(gauge.fileObjects.filter((fo) => fo.name !== f.filename))
                    }
                  >
                    <Trash2 className="h-3 w-3" />
                  </button>
                </li>
              ))}
            </ul>
          )}

          {gauge && (
            <p className="mb-2 text-[11px] text-slate-500">
              {fmtNum(gauge.meta.sample_count)} raw samples, daily medians{" "}
              {gauge.meta.start_date} to {gauge.meta.end_date}. Gauge BHP overrides{" "}
              {matched} of {tests.length} tests; the IPR fit and chart use it, and a
              saved IPR keeps the gauge-derived values. Session-only - gone on refresh.
            </p>
          )}

          <div className="flex items-center gap-2">
            <button
              type="button"
              disabled={busy}
              onClick={() => fileInput.current?.click()}
              className="flex items-center gap-1 rounded-md border border-slate-300 bg-white px-2 py-1 text-xs font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
            >
              <Upload className="h-3 w-3" />
              {gauge ? "Add another file" : "Upload XLSX"}
            </button>
            {gauge && (
              <button
                type="button"
                disabled={busy}
                className="text-xs text-slate-500 underline-offset-2 hover:text-slate-700 hover:underline disabled:opacity-50"
                onClick={() => clearGauge(well)}
              >
                Clear gauge
              </button>
            )}
            {busy && <Spinner label="Parsing" />}
          </div>
          {err && <p className="mt-2 text-xs text-amber-700">{err}</p>}

          <input
            ref={fileInput}
            type="file"
            accept=".xlsx"
            multiple
            className="hidden"
            onChange={(e) => {
              onPick(e.target.files);
              e.target.value = ""; // same file re-pickable after remove
            }}
          />
        </div>
      )}
    </div>
  );
}
