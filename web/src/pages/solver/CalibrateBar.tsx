/**
 * "Auto-match BHP" - port of the Solver's Run BHP Calibration action bar
 * (jetpump_solver._render_fric_cal_action_bar + _execute_fric_cal). Fits
 * ken/kth/kdi (ONLY - as-built geometry is never varied, enforced
 * server-side in fric_calibration's bounds) so the modeled suction pressure
 * lands on the selected test's measured BHP, then lays the fitted coefs
 * over the sidebar - the auto-solve rerun picks them up everywhere, and
 * "Save as well default" persists them (friction rides the save).
 *
 * Gates, mirroring Streamlit:
 *   - the selected test needs a measured BHP (gauge overrides count);
 *   - the sidebar pump must match the pump installed on the test date -
 *     calibrating one pump against another's test is meaningless. Picking an
 *     older anchor usually lands on the PREVIOUS pump, so the block is the
 *     normal case, not an error: it says which pump ran that day and offers a
 *     "Model 13C" button rather than sending the engineer to the sidebar.
 */

import { Crosshair, SlidersHorizontal } from "lucide-react";
import { useState } from "react";
import { useNavigate } from "react-router-dom";

import { useCalibrate } from "../../api/hooks";
import type { CalibrateResponse, JpInstallRow, WellTestRow } from "../../api/types";
import { Button } from "../../components/ui";
import { fmtNum } from "../../lib/format";
import { useParamsStore } from "../../state/params";

import { pumpAt } from "./selection";
import { KcoefExplainer } from "./KcoefExplainer";

function resultLine(r: CalibrateResponse): { tone: "ok" | "warn"; text: string } {
  if (!r.converged) {
    return { tone: "warn", text: "Calibration failed - the solver found no valid operating point at any friction setting." };
  }
  const base =
    `Matched BHP ${fmtNum(r.modeled_bhp)} vs measured ${fmtNum(r.target_bhp)} psi ` +
    `(off by ${fmtNum(Math.abs(r.bhp_error ?? 0))}) - ` +
    `ken ${r.ken.toFixed(3)} / kth ${r.kth.toFixed(3)} / kdi ${r.kdi.toFixed(3)} applied to the sidebar. ` +
    "Save as well default to keep them.";
  const notes: string[] = [];
  if (r.sonic) notes.push("throat is sonic-choked: friction cannot pull BHP lower");
  if (r.bounded) notes.push("a coefficient sits on its search bound - treat as a limit, not a fit");
  const tone = r.match_quality === "good" ? "ok" : "warn";
  return { tone, text: notes.length ? `${base} (${notes.join("; ")})` : base };
}

export function CalibrateBar({
  well,
  compareTest,
  installs,
}: {
  well: string;
  compareTest: WellTestRow | null;
  installs: JpInstallRow[];
}) {
  const params = useParamsStore((s) => s.params);
  const setMany = useParamsStore((s) => s.setMany);
  const mut = useCalibrate();
  const [notice, setNotice] = useState<{ tone: "ok" | "warn"; text: string } | null>(null);
  const navigate = useNavigate();

  if (params.model_as_water) return null; // water mode has no oil-anchored match

  const targetBhp = compareTest?.bhp ?? null;
  const sidebarPump = `${params.nozzle_no}${params.area_ratio}`;
  const testPumpParts = compareTest ? pumpAt(installs, compareTest.date) : null;
  const testPump = testPumpParts && `${testPumpParts.nozzle}${testPumpParts.throat}`;
  const pumpMismatch = testPump !== null && testPump !== sidebarPump;

  // A pump changeout between the selected test and today is the common case
  // (pick an older anchor and you are almost always looking at the previous
  // pump). Fitting THIS pump's friction to THAT pump's test is meaningless,
  // so the gate stays - but the way out is one click, not a trip to the
  // sidebar.
  const reason =
    targetBhp === null
      ? `The ${compareTest?.date?.slice(0, 10) ?? "selected"} test has no measured BHP - pick a test with one, or add gauge data.`
      : pumpMismatch
        ? `${compareTest?.date?.slice(0, 10)} ran the ${testPump} pump; the sidebar models ${sidebarPump}.`
        : null;

  const run = () => {
    if (!compareTest || targetBhp === null) return;
    setNotice(null);
    mut.mutate(
      { well, params, target_bhp: targetBhp, test_whp: compareTest.whp ?? null },
      {
        onSuccess: (r) => {
          if (r.converged) setMany({ ken: r.ken, kth: r.kth, kdi: r.kdi });
          setNotice(resultLine(r));
        },
        onError: (e) => setNotice({ tone: "warn", text: e.message }),
      },
    );
  };

  return (
    <div className="space-y-1.5 border-t border-slate-100 pt-2.5">
      <div className="flex flex-wrap items-center gap-2">
        <Button
          variant="secondary"
          size="sm"
          disabled={reason !== null || mut.isPending}
          busy={mut.isPending}
          title={
            reason ??
            `Numerically fits the pump friction coefficients (ken/kth/kdi) so the modeled ` +
              `BHP lands on the ${compareTest?.date ?? ""} test's measured ${fmtNum(targetBhp)} psi. ` +
              "Pump depth and casing/tubing dimensions are never changed."
          }
          onClick={run}
        >
          <span className="flex items-center gap-1.5">
            <Crosshair className="h-3.5 w-3.5" />
            {mut.isPending ? "Matching BHP..." : "Auto-match BHP"}
          </span>
        </Button>
        {reason && <span className="text-xs text-amber-700">{reason}</span>}
        {pumpMismatch && testPumpParts && (
          <Button
            variant="secondary"
            size="sm"
            title={`Model the ${testPump} pump that was actually in the hole on ${compareTest?.date?.slice(0, 10)}, so the fit is against its own test.`}
            onClick={() =>
              setMany({ nozzle_no: testPumpParts.nozzle, area_ratio: testPumpParts.throat })
            }
          >
            Model {testPump}
          </Button>
        )}
        {/* Always enabled, including when the match above is blocked - a
            blocked match is exactly when you want to see which inputs are
            live on this well. */}
        <Button
          variant="secondary"
          size="sm"
          title="See what each input does to the BHP, oil, liquid and power-fluid match, and whether any combination reaches this test."
          onClick={() => navigate("/sensitivity")}
        >
          <span className="flex items-center gap-1.5">
            <SlidersHorizontal className="h-3.5 w-3.5" />
            Match Sensitivities
          </span>
        </Button>
      </div>
      {notice && (
        <p className={notice.tone === "ok" ? "text-xs text-emerald-700" : "text-xs text-amber-700"}>
          {notice.text}
        </p>
      )}
      <KcoefExplainer />
    </div>
  );
}
