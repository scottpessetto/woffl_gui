/**
 * Solver workbench - the React port of woffl/gui/tabs/jetpump_solver.py.
 * Layout: verdict bar, solve errors, IPR chart | control column (comparison,
 * anchor controls, rate calculator), full-width test table. The solve runs
 * automatically off the DEBOUNCED sidebar params; the IPR curve itself is
 * pure client math so it redraws instantly as ResP or the anchor moves.
 */

import { useEffect, useMemo, useRef, useState } from "react";

import { ApiError } from "../api/client";
import { useIprFit, useIprPin, useJpHistory, useSolve, useWellTests } from "../api/hooks";
import type { AnchorMode, WellTestRow } from "../api/types";
import { HistoryStrip } from "../components/HistoryStrip";
import { Button, Card, ErrorNote, WarnNote } from "../components/ui";
import { Welcome } from "../layout/Welcome";
import { useDebounced } from "../lib/useDebounced";
import { vogelQmax } from "../lib/vogel";
import { effectiveParams, useParamsStore } from "../state/params";

import { ComparisonCard } from "./solver/ComparisonCard";
import { IprChart } from "./solver/IprChart";
import { IprControls } from "./solver/IprControls";
import { RateCalculator } from "./solver/RateCalculator";
import { resolveAnchorTest, testKey } from "./solver/selection";
import { TestsTable } from "./solver/TestsTable";
import { VerdictBar } from "./solver/VerdictBar";

export default function SolverPage() {
  const well = useParamsStore((s) => s.well);
  const simActive = useParamsStore((s) => s.simActive);

  if (well === "Custom" && !simActive) return <Welcome />;
  // key={well}: all anchor/comparison selections are per-well UI state, so a
  // well switch remounts the workbench with fresh defaults.
  return <Workbench key={well} well={well} />;
}

function Workbench({ well }: { well: string }) {
  const params = useParamsStore((s) => s.params);
  const simActive = useParamsStore((s) => s.simActive);
  const months = useParamsStore((s) => s.months);
  const cap = useParamsStore((s) => s.cap);
  const context = useParamsStore((s) => s.context);
  const set = useParamsStore((s) => s.set);

  const effective = useMemo(() => effectiveParams(params), [params]);
  const debounced = useDebounced(effective, 400);
  const solveQ = useSolve(well, debounced, simActive);
  const testsQ = useWellTests(well, months, cap);
  const pinQ = useIprPin(well);
  const installsQ = useJpHistory(well);

  // --- local UI state: IPR anchor + comparison selection -------------------
  const [anchorMode, setAnchorMode] = useState<AnchorMode>("recent");
  const [anchorDate, setAnchorDate] = useState<string | null>(null);
  const [decouple, setDecouple] = useState(false);
  const [compareKey, setCompareKey] = useState<string | null>(null);
  const [showStrip, setShowStrip] = useState(true);

  // Seed the anchor from the saved pin ONCE per mount (= once per well):
  // an applied pin means "anchor on this specific test".
  const pinSeeded = useRef(false);
  useEffect(() => {
    const pin = pinQ.data;
    if (pinSeeded.current || !pin) return;
    pinSeeded.current = true;
    if (pin.status === "applied" && pin.date_token) {
      setAnchorMode("specific");
      setAnchorDate(pin.date_token);
    }
  }, [pinQ.data]);

  const sortedTests = useMemo<WellTestRow[]>(() => {
    const rows = testsQ.data?.tests ?? [];
    return [...rows].sort((a, b) => (a.date < b.date ? 1 : a.date > b.date ? -1 : 0));
  }, [testsQ.data]);

  const compareTest = useMemo<WellTestRow | null>(() => {
    if (sortedTests.length === 0) return null;
    if (decouple && compareKey !== null) {
      return sortedTests.find((t) => testKey(t) === compareKey) ?? sortedTests[0];
    }
    // Synced (default): the comparison test follows the IPR anchor.
    return resolveAnchorTest(sortedTests, anchorMode, anchorDate);
  }, [sortedTests, decouple, compareKey, anchorMode, anchorDate]);

  const iprFitQ = useIprFit(
    {
      well,
      anchor_mode: anchorMode,
      anchor_date: anchorMode === "specific" ? anchorDate : null,
      field_model: params.field_model,
      months,
      cap,
    },
    simActive && well !== "Custom" && !params.model_as_water && sortedTests.length >= 2,
  );

  const fit = iprFitQ.data ?? null;
  const solve = solveQ.data ?? null;
  const installs = installsQ.data?.installs ?? [];

  // Qmax for the rate calculator: same anchor precedence the chart uses
  // (fit > comparison test > sidebar inflow), at the SIDEBAR ResP.
  const qmax = useMemo<number | null>(() => {
    if (fit) return vogelQmax(fit.coeffs.qwf, fit.coeffs.pwf, params.pres);
    if (compareTest && compareTest.total_fluid !== null && compareTest.bhp !== null) {
      return vogelQmax(compareTest.total_fluid, compareTest.bhp, params.pres);
    }
    return vogelQmax(params.qwf, params.pwf, params.pres);
  }, [fit, compareTest, params.pres, params.qwf, params.pwf]);

  const solveError = solveQ.error;
  const suggestGor =
    solveError instanceof ApiError &&
    solveError.status === 422 &&
    solveError.detail.suggested_gor !== null &&
    solveError.detail.suggested_gor !== undefined;

  return (
    <div className="space-y-4">
      <VerdictBar
        well={well}
        nozzle={params.nozzle_no}
        throat={params.area_ratio}
        contextPump={context?.pump ?? null}
        solve={solve}
        compareTest={compareTest}
        fetching={solveQ.isFetching}
      />

      {solveQ.isError && (
        <div className="space-y-2">
          <ErrorNote error={solveError} />
          {suggestGor && (
            <WarnNote>
              <span className="flex flex-wrap items-center gap-3">
                <span>
                  The solver found no crossing at the current GOR - a lower gas-oil ratio usually
                  converges.
                </span>
                <Button size="sm" onClick={() => set("form_gor", 250)}>
                  Retry with GOR 250
                </Button>
              </span>
            </WarnNote>
          )}
        </div>
      )}

      {showStrip && installsQ.data && installsQ.data.installs.length > 0 && (
        <Card>
          <div className="mb-1 flex items-center justify-between">
            <h3 className="text-sm font-semibold tracking-tight text-slate-700">
              Pump history
              {installsQ.data.current_pump && (
                <span className="ml-2 font-normal text-slate-500">
                  Current: {installsQ.data.current_pump}
                </span>
              )}
            </h3>
            <button
              type="button"
              className="text-xs text-slate-500 hover:text-slate-700"
              onClick={() => setShowStrip(false)}
            >
              Hide
            </button>
          </div>
          <HistoryStrip data={installsQ.data} height={430} />
        </Card>
      )}
      {!showStrip && (
        <button
          type="button"
          className="text-xs text-slate-500 hover:text-slate-700"
          onClick={() => setShowStrip(true)}
        >
          Show pump history
        </button>
      )}

      <div className="grid gap-4 xl:grid-cols-2">
        <IprChart
          tests={sortedTests}
          fit={fit}
          params={params}
          solve={solve}
          compareTest={compareTest}
          installs={installs}
        />
        <div className="space-y-4">
          <ComparisonCard
            solve={solve}
            compareTest={compareTest}
            formWc={params.form_wc}
            ppfSurf={params.ppf_surf}
          />
          <IprControls
            anchorMode={anchorMode}
            anchorDate={anchorDate}
            onAnchorChange={(mode, date) => {
              setAnchorMode(mode);
              setAnchorDate(date);
            }}
            tests={sortedTests}
            installs={installs}
            fit={fit}
            pin={pinQ.data ?? null}
            decouple={decouple}
            onDecouple={setDecouple}
            compareKey={compareKey}
            onCompareChange={setCompareKey}
          />
          <RateCalculator
            qmax={qmax}
            pres={params.pres}
            formWc={params.form_wc}
            defaultBhp={compareTest?.bhp ?? solve?.psu ?? null}
          />
        </div>
      </div>

      <TestsTable
        tests={sortedTests}
        selectedKey={compareTest ? testKey(compareTest) : null}
        onSelect={(key) => {
          setCompareKey(key);
          setDecouple(true);
        }}
      />
    </div>
  );
}
