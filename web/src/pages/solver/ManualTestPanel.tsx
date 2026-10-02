/**
 * The engineer's own test: load an LRS WELL TEST SUMMARY sheet
 * (POST /lrs/parse) or enter one by hand. The test keeps its own numbers and
 * behaves like any other test - a row in the table, a square on the IPR
 * chart, the comparison target, and a choice in the anchor picker. Loading a
 * sheet makes it the IPR anchor when it has a BHP. Session-only.
 */

import { Upload } from "lucide-react";
import { useRef, useState } from "react";

import { upload } from "../../api/client";
import type { LrsTestResponse } from "../../api/types";
import { PARAM_BOUNDS } from "../../api/types";
import { Button, Spinner, WarnNote } from "../../components/ui";
import { fmtNum } from "../../lib/format";
import { manualFitRow, manualTestFromSidebar, todayIso, useManualTestStore, type ManualTest } from "../../state/manualTest";
import { useParamsStore } from "../../state/params";

import { gaugeBhpNear, lrsManualTest } from "./lrsTest";

const INPUT_CLS =
  "mt-1 h-8 w-full rounded-md border border-slate-300 bg-white px-2 text-sm " +
  "text-slate-800 outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-200";

type NumKey = "oil" | "water" | "bhp" | "pfRate" | "pfPress" | "whp" | "gor";

const FIELDS: { key: NumKey; label: string }[] = [
  { key: "oil", label: "Oil (BOPD)" },
  { key: "water", label: "Formation water (BWPD)" },
  { key: "bhp", label: "Test BHP (psi)" },
  { key: "pfRate", label: "PF rate (BWPD)" },
  { key: "pfPress", label: "PF pressure (psi)" },
  { key: "whp", label: "WHP (psi)" },
  { key: "gor", label: "GOR (scf/stb)" },
];

/** Draft-and-commit number box: commits on blur / Enter, blank = unmeasured. */
function NumBox({ label, value, onCommit }: { label: string; value: number | null; onCommit: (v: number | null) => void }) {
  const [draft, setDraft] = useState<string | null>(null);
  const commit = () => {
    if (draft === null) return;
    const text = draft.trim();
    const parsed = text === "" ? null : Number(text);
    if (parsed === null || (Number.isFinite(parsed) && parsed >= 0)) onCommit(parsed);
    setDraft(null);
  };
  return (
    <label className="block">
      <span className="text-xs font-medium text-slate-500">{label}</span>
      <input
        type="number"
        min={0}
        value={draft ?? (value !== null ? String(value) : "")}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => { if (e.key === "Enter") e.currentTarget.blur(); }}
        className={INPUT_CLS}
      />
    </label>
  );
}

export function ManualTestPanel({
  well,
  bhpDaily,
  anchored,
  onUseAsAnchor,
  onEdited,
  onCleared,
}: {
  well: string;
  /** The well's daily gauge BHP feed - the sheet carries no BHP of its own. */
  bhpDaily: { date: string; bhp: number }[];
  /** This test is the IPR anchor right now. */
  anchored: boolean;
  /** Make this test the IPR anchor (Specific test -> your test). */
  onUseAsAnchor: (test: ManualTest) => void;
  /** A number changed: an anchored test's fit must be re-applied. */
  onEdited: () => void;
  onCleared: () => void;
}) {
  const test = useManualTestStore((s) => s.byWell[well]);
  const setManualTest = useManualTestStore((s) => s.setManualTest);
  const clearManualTest = useManualTestStore((s) => s.clearManualTest);

  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [notes, setNotes] = useState<string[]>([]);
  /** A sheet naming another well waits here for an explicit go-ahead. */
  const [foreign, setForeign] = useState<LrsTestResponse | null>(null);
  const fileInput = useRef<HTMLInputElement | null>(null);

  const apply = (sheet: LrsTestResponse) => {
    const date = sheet.test_date ?? todayIso();
    const load = lrsManualTest(sheet, date, PARAM_BOUNDS, gaugeBhpNear(bhpDaily, date));
    setManualTest(well, load.test);
    setNotes(load.notes);
    setForeign(null);
    // An uploaded test is the one the engineer wants the curve through.
    if (manualFitRow(load.test)) onUseAsAnchor(load.test);
  };

  const onPick = async (file: File | undefined) => {
    if (!file) return;
    setBusy(true);
    setErr(null);
    setNotes([]);
    setForeign(null);
    try {
      const form = new FormData();
      form.append("file", file, file.name);
      const sheet = await upload<LrsTestResponse>("/lrs/parse", form);
      if (sheet.well !== null && sheet.well !== well) setForeign(sheet);
      else apply(sheet);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const edit = (patch: Partial<ManualTest>) => {
    if (!test) return;
    setManualTest(well, { ...test, ...patch });
    onEdited();
  };

  const total = test && test.oil !== null && test.water !== null ? test.oil + test.water : null;
  const canAnchor = test ? manualFitRow(test) !== null : false;

  return (
    <div className="space-y-2 rounded-md border border-slate-200 bg-slate-50 p-2.5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-xs font-semibold text-slate-700">
          Your own test{test?.source ? <span className="font-normal text-slate-500"> - {test.source}</span> : null}
        </p>
        <div className="flex items-center gap-2">
          {busy && <Spinner label="Reading sheet" />}
          <Button size="sm" disabled={busy} onClick={() => fileInput.current?.click()}
            title="Read an LRS WELL TEST SUMMARY workbook (.xlsx) as a test of your own and anchor the IPR on it">
            <span className="flex items-center gap-1.5"><Upload className="h-3.5 w-3.5" />Load LRS test sheet</span>
          </Button>
          {!test && (
            <Button size="sm" disabled={busy}
              title="Start a test from the sidebar's rate, BHP, water cut and pressures, then correct the numbers and add the measured PF rate"
              onClick={() => { setManualTest(well, manualTestFromSidebar(useParamsStore.getState().params)); setNotes([]); }}>
              Enter by hand
            </Button>
          )}
        </div>
      </div>

      {!test && (
        <p className="text-[11px] text-slate-500">
          A test that is not in FDC yet - an LRS portable-separator test, for example. It joins the test
          list and can anchor the IPR like any other test.
        </p>
      )}

      {test && (
        <>
          <div className="grid grid-cols-2 gap-2">
            <label className="block">
              <span className="text-xs font-medium text-slate-500">Test date</span>
              <input
                type="date"
                value={test.date}
                onChange={(e) => { if (e.target.value) edit({ date: e.target.value }); }}
                className={INPUT_CLS}
              />
            </label>
            {FIELDS.map((f) => (
              <NumBox key={f.key} label={f.label} value={test[f.key]} onCommit={(v) => edit({ [f.key]: v })} />
            ))}
          </div>
          <p className="text-[11px] text-slate-600">
            {total !== null && total > 0
              ? `Total liquid ${fmtNum(total)} BLPD, water cut ${fmtNum(((test.water ?? 0) / total) * 100, 1)}%. `
              : "Enter the oil and water rates. "}
            {anchored
              ? "This test is the IPR anchor: the curve runs through it and the reservoir pressure is fitted through it. " +
                "It is not an FDC test, so Save well inputs stores the curve as a manual point with this test in the note."
              : canAnchor
                ? "It is listed with the other tests and counts in the fit."
                : "It needs a BHP before it can anchor the IPR."}
          </p>
          <div className="flex flex-wrap items-center gap-3">
            {!anchored && canAnchor && (
              <Button size="sm" variant="primary" onClick={() => onUseAsAnchor(test)}>Use as IPR anchor</Button>
            )}
            <button
              type="button"
              className="text-xs text-slate-500 underline-offset-2 hover:text-slate-700 hover:underline"
              onClick={() => { clearManualTest(well); setNotes([]); onCleared(); }}
            >
              Remove this test
            </button>
          </div>
        </>
      )}

      {foreign && (
        <WarnNote>
          <span className="flex flex-wrap items-center gap-2 text-xs">
            <span>
              This sheet is for {foreign.well_raw ?? foreign.well}, not {well}. Nothing was loaded.
            </span>
            <Button size="sm" onClick={() => apply(foreign)}>Load it into {well} anyway</Button>
          </span>
        </WarnNote>
      )}
      {notes.length > 0 && (
        <ul className="list-disc space-y-0.5 pl-4 text-[11px] text-slate-600">
          {notes.map((n) => <li key={n}>{n}</li>)}
        </ul>
      )}
      {err && <p className="text-xs text-amber-700">{err}</p>}

      <input
        ref={fileInput}
        type="file"
        accept=".xlsx,.xlsm"
        className="hidden"
        onChange={(e) => {
          void onPick(e.target.files?.[0]);
          e.target.value = ""; // the same file is re-pickable
        }}
      />
    </div>
  );
}
