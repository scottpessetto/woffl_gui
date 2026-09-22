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

import { useMatchTest, useMeta, useSaveIpr, useSaveMatchCalibration, useWellInputWritePending } from "../../api/hooks";
import type { MatchTestResponse, SimParams, WellTestRow } from "../../api/types";
import { Button } from "../../components/ui";
import { fmtDate, fmtNum } from "../../lib/format";
import { changedWellInputs, wellInputProblem, wellInputValues } from "../../lib/wellInputs";
import { useParamsStore } from "../../state/params";

/** The sidebar values "Apply to inputs" lays down, in sidebar units. */
function matchedValues(result: MatchTestResponse): Partial<SimParams> {
  return {
    qwf: Math.round(result.qwf_liq),
    pwf: Math.round(result.pwf ?? 0),
    form_wc: Number(result.form_wc.toFixed(3)),
    kth: result.kth,
    kdi: result.kdi,
  };
}

const CHANGE_ROWS: { key: keyof SimParams; label: string; fmt: (v: number) => string; saved: string }[] = [
  { key: "qwf", label: "IPR anchor rate", fmt: (v) => `${fmtNum(v)} BLPD`, saved: "well inputs" },
  { key: "pwf", label: "Anchor BHP (inferred)", fmt: (v) => `${fmtNum(v)} psi`, saved: "well inputs" },
  { key: "form_wc", label: "Water cut", fmt: (v) => `${fmtNum(v * 100, 1)}%`, saved: "well inputs" },
  { key: "kth", label: "Throat loss kth", fmt: (v) => v.toFixed(3), saved: "pump fit" },
  { key: "kdi", label: "Diffuser loss kdi", fmt: (v) => v.toFixed(3), saved: "pump fit" },
];

/** Before -> after for every value the match sets, and where each is saved. */
function ChangeTable({ result }: { result: MatchTestResponse }) {
  const params = useParamsStore((s) => s.params);
  const after = matchedValues(result);
  return (
    <table className="mt-1 w-full max-w-lg border-collapse text-xs">
      <thead>
        <tr className="text-left opacity-80">
          <th className="py-0.5 pr-2 font-semibold">Value</th>
          <th className="py-0.5 pr-2 font-semibold">Sidebar now</th>
          <th className="py-0.5 pr-2 font-semibold">Match</th>
          <th className="py-0.5 font-semibold">Saved as</th>
        </tr>
      </thead>
      <tbody>
        {CHANGE_ROWS.map(({ key, label, fmt, saved }) => {
          const now = params[key] as number;
          const next = after[key] as number;
          const changed = Math.abs(now - next) > 1e-9;
          return (
            <tr key={key} className={changed ? "font-medium" : "opacity-70"}>
              <td className="py-0.5 pr-2">{label}</td>
              <td className="py-0.5 pr-2 tabular-nums">{fmt(now)}</td>
              <td className="py-0.5 pr-2 tabular-nums">{fmt(next)}{changed ? "" : " (same)"}</td>
              <td className="py-0.5">{saved}</td>
            </tr>
          );
        })}
        <tr className="opacity-70">
          <td className="py-0.5 pr-2">Entrance loss ken</td>
          <td className="py-0.5 pr-2 tabular-nums" colSpan={2}>held at {result.ken.toFixed(3)}</td>
          <td className="py-0.5">pump fit</td>
        </tr>
      </tbody>
    </table>
  );
}

/**
 * Save the applied match so the well reopens with it: the IPR anchor through
 * the same well-input save as the save bar (pinned to the matched test), then
 * kth/kdi as the INSTALLED pump's calibration. Save well inputs alone never
 * kept the coefficients, so a reopened well used reference losses against the
 * matched BHP and looked reset (user report 2026-09-22).
 */
function SaveMatch({ well, result, test, note }: { well: string; result: MatchTestResponse; test: WellTestRow; note: string }) {
  const meta = useMeta();
  const saveInputs = useSaveIpr(well);
  const saveFit = useSaveMatchCalibration(well);
  const writePending = useWellInputWritePending(well);
  const params = useParamsStore((s) => s.params);
  const context = useParamsStore((s) => s.context);
  const [status, setStatus] = useState<{ ok: boolean; text: string }[]>([]);
  const token = result.save_token;
  const after = matchedValues(result);
  const applied = (Object.keys(after) as (keyof SimParams)[])
    .every((k) => Math.abs((params[k] as number) - (after[k] as number)) < 1e-9);
  const others = changedWellInputs(params, context?.seeds).filter((k) => !["qwf", "pwf", "form_wc"].includes(k));
  const busy = saveInputs.isPending || saveFit.isPending;
  const blocked = !token ? "This result is not saveable as a pump fit (the BHP was not identified)."
    : !meta.data?.writes_enabled ? "This app is read-only, so the match cannot be saved."
    : !applied ? "Apply the match to the inputs first, so what you save is what you see."
    : wellInputProblem(params);

  const onSave = () => {
    if (blocked || !token || busy || writePending) return;
    const { params: current, context: baseline } = useParamsStore.getState();
    setStatus([]);
    saveInputs.mutate({
      ...wellInputValues(current, baseline?.seeds),
      comment: note,
      pin_wt_uid: test.wt_uid ?? null,
      pin_date: test.date ?? null,
      unpin: test.wt_uid == null,
    }, {
      onSuccess: (r) => {
        if (r.n_values <= 0) { setStatus([{ ok: false, text: r.values_message }]); return; }
        const first = { ok: true, text: "Well inputs saved (IPR anchor, BHP, WC)." };
        setStatus([first]);
        saveFit.mutate(token, {
          onSuccess: (f) => setStatus([first, { ok: true, text: f.message }]),
          onError: (e) => setStatus([first, { ok: false, text: `Pump fit not saved: ${e.message}` }]),
        });
      },
      onError: (e) => setStatus([{ ok: false, text: e.message }]),
    });
  };

  return (
    <div className="space-y-1 pt-1">
      <Button size="sm" variant="primary" disabled={!!blocked || busy || writePending} busy={busy}
        title={blocked ?? "Save the matched well inputs and the fitted throat/diffuser losses so this well reopens with the match"}
        onClick={onSave}>
        Save this match
      </Button>
      <p className="text-xs opacity-90">
        {blocked ?? "Saves the IPR anchor, inferred BHP and WC as the well inputs (pinned to this test), then kth/kdi as the installed pump's fit. The well reopens with both, and new optimization runs use them."}
      </p>
      {!blocked && others.length > 0 && (
        <p className="text-xs text-amber-800">
          Your other unsaved edits ({others.join(", ")}) will be saved with it.
        </p>
      )}
      {status.map((s) => (
        <p key={s.text} role="status" className={`text-xs ${s.ok ? "text-emerald-700" : "text-red-700"}`}>{s.text}</p>
      ))}
    </div>
  );
}

const QUALITY_TONE: Record<MatchTestResponse["match_quality"], string> = {
  good: "border-emerald-200 bg-emerald-50 text-emerald-800",
  fair: "border-amber-200 bg-amber-50 text-amber-800",
  poor: "border-amber-200 bg-amber-50 text-amber-800",
  failed: "border-red-200 bg-red-50 text-red-800",
};

function ResultBlock({ well, result, test }: { well: string; result: MatchTestResponse; test: WellTestRow }) {
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
      {!failed && <ChangeTable result={result} />}
      {!failed && (
        <Button
          variant="secondary"
          size="sm"
          title={
            "Lay the inferred anchor over the sidebar inputs: IPR anchor rate (total liquid) and BHP, " +
            "the test's water cut, and the fitted throat / diffuser coefficients. Nothing is written - " +
            "the save comment is prefilled with where this BHP came from. Save this match (below) keeps " +
            "both the inputs and the fitted losses."
          }
          onClick={() => {
            setMany(matchedValues(result));
            setMatchNote(note);
          }}
        >
          {unreachable ? "Apply the closest point anyway" : "Apply to inputs"}
        </Button>
      )}
      {!failed && <SaveMatch well={well} result={result} test={test} note={note} />}
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
      {shown && <ResultBlock well={well} result={shown.body} test={shown.test} />}
    </>
  );
}
