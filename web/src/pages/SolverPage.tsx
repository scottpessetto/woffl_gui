/**
 * Solver workbench - the React port of woffl/gui/tabs/jetpump_solver.py.
 * Layout: verdict bar, solve errors, IPR chart | control column (comparison,
 * anchor controls, rate calculator), pump-history strip, full-width test
 * table. Convention for every single-well analysis page: the pump-history
 * strip sits just above the historical-tests table, never above the page's
 * primary chart. The solve runs
 * automatically off the DEBOUNCED sidebar params; the IPR curve itself is
 * pure client math so it redraws instantly as ResP or the anchor moves.
 */

import { useEffect, useMemo, useRef, useState } from "react";

import { ApiError } from "../api/client";
import { useIprFit, useIprPin, useJpHistory, useSolve, useWellTests } from "../api/hooks";
import type { AnchorMode, WellTestRow } from "../api/types";
import { HistoryStrip } from "../components/HistoryStrip";
import { Button, Card, ErrorNote, Spinner, WarnNote } from "../components/ui";
import { Welcome } from "../layout/Welcome";
import { useDebounced } from "../lib/useDebounced";
import { vogelQmax } from "../lib/vogel";
import { gaugeMonths, useGaugeStore } from "../state/gauge";
import { effectiveParams, useParamsStore } from "../state/params";

import { ComparisonCard } from "./solver/ComparisonCard";
import { CalibrateBar } from "./solver/CalibrateBar";
import { GaugePanel } from "./solver/GaugePanel";
import { IprChart } from "./solver/IprChart";
import { IprControls } from "./solver/IprControls";
import { RateCalculator } from "./solver/RateCalculator";
import { ResponseDiagnostic } from "./solver/ResponseDiagnostic";
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
  const gauge = useGaugeStore((s) => s.byWell[well]);
  // A gauge window usually reaches further back than the sidebar lookback -
  // widen the test fetch to cover it (mirror of the Streamlit extended-tests
  // fetch), in coarse 6-month steps so the fleet-test cache stays warm.
  const effectiveMonths = gauge ? gaugeMonths(gauge.meta, months) : months;
  const testsQ = useWellTests(well, effectiveMonths, cap);
  const pinQ = useIprPin(well);
  const installsQ = useJpHistory(well);

  // --- local UI state: IPR anchor + comparison selection -------------------
  const [anchorMode, setAnchorMode] = useState<AnchorMode>("recent");
  const [anchorDate, setAnchorDate] = useState<string | null>(null);
  const [decouple, setDecouple] = useState(false);
  const [compareKey, setCompareKey] = useState<string | null>(null);
  const [showStrip, setShowStrip] = useState(true);

  // Seed the anchor ONCE per mount (= once per well): an applied pin means
  // "anchor on this specific test"; saved values with NO pin mean the anchor
  // is not a test at all, so the selector opens on Manual point and the
  // test-derived fit never runs against it.
  const pinSeeded = useRef(false);
  useEffect(() => {
    const pin = pinQ.data;
    // BOTH inputs must have landed before latching: the pin usually resolves
    // first, and latching on it alone left a manual-anchor well showing "Most
    // recent" because ipr_source had not arrived yet.
    if (pinSeeded.current || !pin || !context) return;
    pinSeeded.current = true;
    if (pin.status === "applied" && pin.date_token) {
      setAnchorMode("specific");
      setAnchorDate(pin.date_token);
    } else if (context.ipr_source === "manual") {
      setAnchorMode("manual");
    }
  }, [pinQ.data, context]);

  const sortedTests = useMemo<WellTestRow[]>(() => {
    const rows = testsQ.data?.tests ?? [];
    const sorted = [...rows].sort((a, b) => (a.date < b.date ? 1 : a.date > b.date ? -1 : 0));
    if (!gauge) return sorted;
    // Gauge daily medians override BHP wherever covered (display mirror of
    // memory_gauge.apply_to_well_tests; the fit applies the SAME overrides
    // server-side via bhp_overrides on the request).
    return sorted.map((t) => {
      const bhp = gauge.dailyByDate[t.date.slice(0, 10)];
      return bhp === undefined ? t : { ...t, bhp };
    });
  }, [testsQ.data, gauge]);

  // A manual anchor IS the sidebar's qwf/pwf, so there is nothing to fit: the
  // test-derived curve would only compete with the point the engineer chose.
  const fitEnabled =
    simActive &&
    well !== "Custom" &&
    !params.model_as_water &&
    anchorMode !== "manual" &&
    sortedTests.length >= 2;
  const iprFitQ = useIprFit(
    {
      well,
      anchor_mode: anchorMode === "manual" ? "recent" : anchorMode,
      anchor_date: anchorMode === "specific" ? anchorDate : null,
      field_model: params.field_model,
      months: effectiveMonths,
      cap,
      bhp_overrides: gauge ? gauge.meta.daily : null,
    },
    fitEnabled,
  );

  const compareTest = useMemo<WellTestRow | null>(() => {
    if (sortedTests.length === 0) return null;
    if (decouple && compareKey !== null) {
      return sortedTests.find((t) => testKey(t) === compareKey) ?? sortedTests[0];
    }
    // Synced (default): the comparison test follows the IPR anchor - the
    // FIT's own anchor_date when it has landed (the server resolves median/
    // recent, so the UI can never disagree with the drawn curve), else the
    // local mirror. A manual anchor has no test, but the engineer still
    // needs something to judge the match against, so the comparison falls
    // back to the most recent test.
    const fitAnchor = anchorMode === "manual" ? null : (iprFitQ.data?.coeffs.anchor_date ?? null);
    return resolveAnchorTest(sortedTests, anchorMode, anchorDate, fitAnchor) ?? sortedTests[0];
  }, [sortedTests, decouple, compareKey, anchorMode, anchorDate, iprFitQ.data]);

  // Auto-apply the FIRST fit's seeds once per well - the web equivalent of
  // Streamlit's open-time anchor sync (_sync_chosen_ipr_to_sidebar): the
  // chart curve and the solve then agree from the first settled paint
  // instead of waiting for a manual "Apply IPR to inputs" click. Locked
  // WC/GOR/ResP survive, and so does anything the engineer set by hand
  // (applyIprSeeds filters both). Ordering guards:
  //   - the pin must settle first, or the recent-anchor fit could land and
  //     latch before the pin switches the anchor to its specific test;
  //   - CONTEXT seeding must have applied first (seededFor === well), or
  //     the slower context response would arrive later and wholesale-
  //     overwrite the applied seeds - the exact mismatch this fixes.
  const ctxSeeded = useParamsStore((s) => s.seededFor) === well;
  const pinSettled = pinQ.isSuccess || pinQ.isError;
  // The latch is PER WELL in the params store, not per mount: a component
  // useState reset on every return to the Solver and re-applied the fit over
  // whatever the engineer had since set by hand (or applied from Match
  // Sensitivities). Cleared by selectWell and setWindow.
  const fitApplied = useParamsStore((s) => s.fitAppliedFor) === well;
  // A reviewed save OUTRANKS the fit. When the server's saved-IPR overlay won
  // (ipr_source "saved"), the seeds on screen ARE the engineer's numbers and
  // the fit must not lay itself over them - 12 of the 31 wells carrying saved
  // values have a fit that disagrees, and the worst of them differed by 2x on
  // the anchor rate (MPB-35: saved 668 BLPD, fit 322). The latch is still set
  // so the chart's settle gate closes.
  const savedWins = context?.ipr_source === "saved" || context?.ipr_source === "manual";
  useEffect(() => {
    const f = iprFitQ.data;
    if (fitApplied || !pinSettled || !ctxSeeded || !f) return;
    useParamsStore.getState().markFitApplied(well);
    if (savedWins) return;
    useParamsStore.getState().applyIprSeeds(f.seeds);
  }, [iprFitQ.data, pinSettled, ctxSeeded, fitApplied, savedWins, well]);

  // First-load settle gate for the IPR chart: hold it greyed until every
  // input series has arrived ONCE (tests, installs/pump labels, pin, the
  // fit when it will run - APPLIED, and the solve of the applied state),
  // so the plot doesn't visibly mutate while the engineer is already
  // reading it. Latches true and stays true - later param edits redraw
  // live, which is the point of the client-rendered chart. Workbench
  // remounts per well, resetting the latch.
  const settledNow =
    (testsQ.isSuccess || testsQ.isError) &&
    (installsQ.isSuccess || installsQ.isError) &&
    (pinQ.isSuccess || pinQ.isError) &&
    (!fitEnabled || ((iprFitQ.isSuccess || iprFitQ.isError) && (fitApplied || iprFitQ.isError))) &&
    // reference equality = the debounce is quiescent: the solve on screen
    // corresponds to the CURRENT params (post-auto-apply), not a stale set.
    (!simActive || (solveQ.isSuccess && debounced === effective) || solveQ.isError);
  const [iprReady, setIprReady] = useState(false);
  useEffect(() => {
    if (settledNow) setIprReady(true);
  }, [settledNow]);

  // Manual anchor: the fit is not just unused, it is absent from the UI. The
  // query key maps manual to "recent", so react-query would still hand back a
  // cached curve and re-offer "Apply IPR to inputs" - which contradicts the
  // mode the engineer just chose.
  const fit = anchorMode === "manual" ? null : (iprFitQ.data ?? null);
  const solve = solveQ.data ?? null;
  const installs = installsQ.data?.installs ?? [];

  // Pump-history strip data with gauge daily medians laid over the feed
  // (gauge wins inside its coverage - mirror of daily_bhp_from_gauge).
  const stripData = useMemo(() => {
    const data = installsQ.data;
    if (!data || !gauge) return data;
    const merged = new Map(data.bhp_daily.map((d) => [d.date.slice(0, 10), d.bhp]));
    for (const g of gauge.meta.daily) merged.set(g.date, g.bhp);
    return {
      ...data,
      bhp_daily: [...merged.entries()]
        .map(([date, bhp]) => ({ date, bhp }))
        .sort((a, b) => (a.date < b.date ? -1 : 1)),
    };
  }, [installsQ.data, gauge]);

  // Qmax for the rate calculator: the SIDEBAR inflow, the same anchor the
  // chart curve and the solve use. Preferring the fit here made the
  // calculator quote rates off a curve that was not on screen.
  const qmax = useMemo<number | null>(
    () => vogelQmax(params.qwf, params.pwf, params.pres),
    [params.pres, params.qwf, params.pwf],
  );

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

      {/* items-start: grid items stretch to the row by default, and the
          control column is usually the taller one - a stretched chart card
          would trail a block of empty white below the plot. */}
      <div className="grid items-start gap-4 xl:grid-cols-2">
        <div className="space-y-4">
          <IprChart
            tests={sortedTests}
            fit={fit}
            params={params}
            solve={solve}
            installs={installs}
            loading={!iprReady}
            gaugeSlot={<GaugePanel well={well} tests={sortedTests} />}
          />
          <RateCalculator
            qmax={qmax}
            pres={params.pres}
            formWc={params.form_wc}
            defaultBhp={compareTest?.bhp ?? solve?.psu ?? null}
          />
        </div>
        <div className="space-y-4">
          <ComparisonCard
            solve={solve}
            compareTest={compareTest}
            formWc={params.form_wc}
            ppfSurf={params.ppf_surf}
            footer={<CalibrateBar well={well} />}
          />
          <IprControls
            anchorMode={anchorMode}
            well={well}
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
        </div>
      </div>

      {/* Same-size skeleton while pump history loads: the strip appearing
          later would otherwise shove the tests table down mid-read. */}
      {showStrip && installsQ.isLoading && (
        <Card>
          <div className="mb-1 flex items-center justify-between">
            <h3 className="text-sm font-semibold tracking-tight text-slate-700">Pump history</h3>
          </div>
          <div className="flex h-[430px] animate-pulse items-center justify-center rounded-lg bg-slate-50">
            <Spinner label="Loading pump history" />
          </div>
        </Card>
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
          <HistoryStrip data={stripData ?? installsQ.data} height={430} />
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

      <TestsTable
        tests={sortedTests}
        selectedKey={compareTest ? testKey(compareTest) : null}
        onSelect={(key) => {
          setCompareKey(key);
          setDecouple(true);
        }}
      />

      {/* Advanced: daily field (Ppf, BHP) vs the current model's response
          curve. Sits below the test history - it is a fit-quality post-check,
          not part of the day-to-day solve loop. Hides itself on old servers. */}
      <ResponseDiagnostic well={well} />
    </div>
  );
}
