/**
 * JP Friction Calibration - fit ken/kth/kdi per well against measured BHP.
 *
 * Port of woffl/gui/scotts_tools/jp_calibration.py.
 *
 * READ-ONLY, exactly as the tab was. The SQL panel is a PREVIEW for a human
 * to run against prop_hist; nothing on this page or behind it writes, and
 * that is deliberate - the app has one write path and this is not it.
 */

import { useMemo, useState } from "react";

import { useCalibrationInputs } from "../../api/hooks";
import type { ToolRow } from "../../api/types";
import { Button, Card, InfoNote, Metric, Section, Spinner } from "../../components/ui";
import { fmtNum } from "../../lib/format";
import { AutoTable, NumField, RunStatus, useToolRun } from "./ToolRun";

interface Req {
  wells: string[] | null;
  months_back: number;
}

const LEAD = ["Well", "Pad", "Pump", "Match", "Actual BHP", "Modeled BHP", "BHP err",
  "Current ken", "Cal ken", "Current kth", "Cal kth", "Current kdi", "Cal kdi",
  "d ken", "d kth", "d kdi", "PF used", "Status"];

export default function JpCalibrationPage() {
  const [months, setMonths] = useState(6);
  const [selected, setSelected] = useState<string[]>([]);
  const inputs = useCalibrationInputs(months, true);
  const run = useToolRun<Req>("/tools/calibration/run");

  const inputRows = (inputs.data?.rows ?? []) as ToolRow[];
  const candidates = useMemo(
    () => [...new Set(inputRows.map((r) => String(r.Well)))].sort(),
    [inputRows],
  );

  const result = run.result as { rows?: ToolRow[]; sql?: string } | null;
  const rows = result?.rows ?? [];
  const converged = rows.filter((r) => r.Status === "converged").length;

  return (
    <div className="space-y-4">
      <Section
        title="JP Friction Calibration"
        actions={
          <Button
            onClick={() => run.run({ wells: selected.length ? selected : null, months_back: months })}
            disabled={run.running || !candidates.length}
          >
            {run.running ? "Calibrating..." : selected.length ? `Calibrate ${selected.length}` : "Calibrate all"}
          </Button>
        }
      >
        <p className="mb-3 text-sm text-slate-600">
          Fits the friction coefficients that reproduce each well&apos;s measured BHP, at the PF
          pressure and pump paired with that same test. Read-only: the SQL below is a preview
          for review, not a write.
        </p>
        <div className="flex flex-wrap items-end gap-3">
          <NumField label="Lookback (months)" value={months} onChange={setMonths} min={1} max={60} />
          <Button size="sm" variant="ghost" onClick={() => setSelected([])}>All wells</Button>
        </div>

        {inputs.isLoading && <Spinner label="Building the input table" />}
        {candidates.length > 0 && (
          <div className="mt-3">
            <span className="text-xs text-slate-500">
              Candidates ({selected.length ? `${selected.length} selected` : `all ${candidates.length}`})
            </span>
            <div className="mt-1 flex max-h-32 flex-wrap gap-1 overflow-y-auto rounded-md border border-slate-200 p-2">
              {candidates.map((w) => (
                <button
                  key={w}
                  type="button"
                  onClick={() =>
                    setSelected((s) => (s.includes(w) ? s.filter((x) => x !== w) : [...s, w]))
                  }
                  className={
                    "rounded border px-1.5 py-0.5 text-xs transition-colors " +
                    (selected.includes(w)
                      ? "border-blue-500 bg-blue-50 font-medium text-blue-700"
                      : "border-slate-200 bg-white text-slate-500 hover:bg-slate-50")
                  }
                >
                  {w}
                </button>
              ))}
            </div>
          </div>
        )}
      </Section>

      <RunStatus run={run as never} idle="Press Calibrate to fit the selected wells." />

      {result && (
        <Card>
          <div className="flex flex-wrap gap-6">
            <Metric label="Wells" value={fmtNum(rows.length, 0)} />
            <Metric label="Converged" value={`${converged} / ${rows.length}`} />
          </div>
        </Card>
      )}

      {rows.length > 0 && (
        <Section title="Calibration results">
          <AutoTable rows={rows} prefer={LEAD} />
        </Section>
      )}

      {result?.sql ? (
        <Section title="SQL preview (review only - nothing here writes)">
          <pre className="max-h-80 overflow-auto rounded-md border border-slate-200 bg-slate-50 p-3 text-xs leading-relaxed text-slate-700">
            {result.sql}
          </pre>
        </Section>
      ) : (
        result && <InfoNote>No SQL preview - nothing converged to a new coefficient.</InfoNote>
      )}
    </div>
  );
}
