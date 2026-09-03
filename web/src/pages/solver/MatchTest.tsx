/**
 * "Match the test (no gauge)" - the calibration for wells without a
 * downhole gauge. The test's power-fluid rate through the nozzle IS a
 * pressure measurement: POST /match-test anchors the IPR on the test's own
 * oil at a trial BHP and fits (pwf, kth, kdi) so the installed pump
 * reproduces the test's oil AND PF. The BHP that comes back is INFERRED,
 * and the block says so, with the one thing a single test cannot separate
 * (nozzle wear vs BHP) spelled out.
 *
 * Nothing lands on the sidebar automatically: "Apply to inputs" lays the
 * anchor (qwf as TOTAL liquid, pwf, the test's water cut) and the two
 * discharge coefficients over the params store via setMany, so they become
 * engineer-owned and the open-time IPR fit stops overwriting them. ken is
 * held by the fit and is not touched. It also writes the params store's
 * matchNote - the same provenance channel the sensitivity study uses - so
 * IprControls shows WHERE the BHP came from and prefills the save comment
 * with it. An inferred BHP that reaches prop_hist has to say it is inferred.
 */

import { Crosshair } from "lucide-react";
import { useState } from "react";

import { useMatchTest } from "../../api/hooks";
import type { MatchTestResponse, WellTestRow } from "../../api/types";
import { Button } from "../../components/ui";
import { fmtDate, fmtNum } from "../../lib/format";
import { useParamsStore } from "../../state/params";

const QUALITY_TONE: Record<MatchTestResponse["match_quality"], string> = {
  good: "border-emerald-200 bg-emerald-50 text-emerald-800",
  fair: "border-amber-200 bg-amber-50 text-amber-800",
  poor: "border-amber-200 bg-amber-50 text-amber-800",
  failed: "border-red-200 bg-red-50 text-red-800",
};

function ResultBlock({ result, test }: { result: MatchTestResponse; test: WellTestRow }) {
  const setMany = useParamsStore((s) => s.setMany);
  const setMatchNote = useParamsStore((s) => s.setMatchNote);
  const failed = result.match_quality === "failed" || result.pwf === null;
  const unreachable = !failed && !result.pf_reachable;
  const tone = QUALITY_TONE[unreachable ? "poor" : result.match_quality];
  // Provenance for the save comment: an inferred BHP must never reach
  // prop_hist looking like a gauge reading.
  const note =
    `BHP ${fmtNum(result.pwf)} psi inferred from ${fmtNum(test.lift_wat)} BWPD PF on test ` +
    `${fmtDate(test.date)} (gaugeless match, ${unreachable ? "closest point - BHP not identified" : result.match_quality})`;

  const headline = failed
    ? `No match: ${result.message ?? "the pump model found no operating point"}.`
    : unreachable
      ? `BHP not identified. ${result.message ?? ""}`
      : `Inferred BHP ${fmtNum(result.pwf)} psi from ${fmtNum(test.lift_wat)} BWPD of power fluid at ` +
      `${fmtNum(result.ppf_surf_used)} psi PF pressure (${result.match_quality}). ` +
      `Model at that anchor: oil ${fmtNum(result.modeled_oil)} vs ${fmtNum(result.modeled_oil === null ? null : test.oil)} BOPD` +
      `${result.oil_error_pct === null ? "" : ` (${result.oil_error_pct >= 0 ? "+" : ""}${result.oil_error_pct.toFixed(1)}%)`}, ` +
      `PF ${fmtNum(result.modeled_pf)} vs ${fmtNum(test.lift_wat)} BWPD` +
      `${result.pf_error_pct === null ? "" : ` (${result.pf_error_pct >= 0 ? "+" : ""}${result.pf_error_pct.toFixed(1)}%)`}.` +
      (result.bhp_resolution_psi === null ? "" : ` Resolution: a 2% PF error is worth about ${fmtNum(result.bhp_resolution_psi)} psi of BHP here.`);

  return (
    <div className={`basis-full space-y-1 rounded-md border px-2.5 py-2 ${tone}`}>
      <p className="text-xs">{headline}</p>
      {!failed && !unreachable && result.message && <p className="text-xs opacity-90">{result.message}</p>}
      {unreachable && (
        <p className="text-xs opacity-90">
          Closest point: BHP {fmtNum(result.pwf)} psi, model PF {fmtNum(result.modeled_pf)} vs {fmtNum(test.lift_wat)} BWPD
          {result.pf_error_pct === null ? "" : ` (${result.pf_error_pct >= 0 ? "+" : ""}${result.pf_error_pct.toFixed(1)}%)`}, oil{" "}
          {fmtNum(result.modeled_oil)} vs {fmtNum(test.oil)} BOPD.
          {result.area_factor_needed !== null &&
            ` A nozzle area factor of about ${result.area_factor_needed.toFixed(2)} would pass this PF; the sidebar bound is 0.8 to 1.3.`}
        </p>
      )}
      {!failed && (
        <p className="text-xs opacity-80">
          {result.caveat} Throat / diffuser losses fitted to {result.kth.toFixed(3)} / {result.kdi.toFixed(3)}; entrance loss held
          at {result.ken.toFixed(3)}.
        </p>
      )}
      {!failed && (
        <Button
          variant="secondary"
          size="sm"
          title={
            "Lay the inferred anchor over the sidebar inputs: IPR anchor rate (total liquid) and BHP, " +
            "the test's water cut, and the fitted throat / diffuser coefficients. Nothing is written - " +
            "the save comment is prefilled with where this BHP came from, and Save as well default in " +
            "the IPR block is what keeps it."
          }
          onClick={() => {
            setMany({
              qwf: Math.round(result.qwf_liq),
              pwf: Math.round(result.pwf!),
              form_wc: Number(result.form_wc.toFixed(3)),
              kth: result.kth,
              kdi: result.kdi,
            });
            setMatchNote(note);
          }}
        >
          {unreachable ? "Apply the closest point anyway" : "Apply to inputs"}
        </Button>
      )}
    </div>
  );
}

export function MatchTest({ well, compareTest }: { well: string; compareTest: WellTestRow | null }) {
  const params = useParamsStore((s) => s.params);
  const mut = useMatchTest();
  const [result, setResult] = useState<{ key: string; body: MatchTestResponse; test: WellTestRow } | null>(null);

  if (well === "Custom") return null;

  const hasOil = compareTest !== null && compareTest.oil !== null && compareTest.oil > 0;
  const hasPf = compareTest !== null && compareTest.lift_wat !== null && compareTest.lift_wat > 0;
  const ready = hasOil && hasPf;
  const reason = !compareTest
    ? "Pick a test to match first."
    : !hasOil
      ? "The selected test has no oil rate to match."
      : !hasPf
        ? "The selected test has no power-fluid rate - the PF rate is what stands in for the gauge."
        : "Infer the flowing BHP from this test's power-fluid rate and fit the throat / diffuser " +
          "losses so the pump reproduces the test's oil and PF. For wells without a downhole gauge.";

  const testKey = `${well}:${compareTest?.wt_uid ?? compareTest?.date ?? ""}`;
  const shown = result && result.key === testKey ? result : null;

  return (
    <>
      <Button
        variant="secondary"
        size="sm"
        disabled={!ready || mut.isPending}
        busy={mut.isPending}
        title={reason}
        onClick={() => {
          if (!compareTest) return;
          mut.mutate(
            {
              well,
              params,
              test_oil: compareTest.oil ?? 0,
              test_water: compareTest.water ?? 0,
              test_pf: compareTest.lift_wat ?? 0,
              test_whp: compareTest.whp,
              test_pf_press: compareTest.pf_press,
              test_date: compareTest.date,
            },
            { onSuccess: (body) => setResult({ key: testKey, body, test: compareTest }) },
          );
        }}
      >
        <span className="flex items-center gap-1.5">
          <Crosshair className="h-3.5 w-3.5" />
          {mut.isPending ? "Matching..." : "Match the test (no gauge)"}
        </span>
      </Button>
      {mut.isError && (
        <span className="basis-full text-xs text-amber-700">Could not match: {mut.error.message}</span>
      )}
      {shown && <ResultBlock result={shown.body} test={shown.test} />}
    </>
  );
}
