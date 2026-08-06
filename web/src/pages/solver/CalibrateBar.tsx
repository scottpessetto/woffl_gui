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
 *     calibrating one pump against another's test is meaningless.
 */

import { Crosshair } from "lucide-react";
import { useState } from "react";

import { useCalibrate } from "../../api/hooks";
import type { CalibrateResponse, JpInstallRow, WellTestRow } from "../../api/types";
import { Button } from "../../components/ui";
import { fmtNum } from "../../lib/format";
import { useParamsStore } from "../../state/params";

import { pumpLabelAt } from "./selection";
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

  if (params.model_as_water) return null; // water mode has no oil-anchored match

  const targetBhp = compareTest?.bhp ?? null;
  const sidebarPump = `${params.nozzle_no}${params.area_ratio}`;
  const testPump = compareTest ? pumpLabelAt(installs, compareTest.date) : null;
  const pumpMismatch = testPump !== null && testPump !== sidebarPump;

  const reason =
    targetBhp === null
      ? "The selected test has no measured BHP - pick a test with one, or add gauge data."
      : pumpMismatch
        ? `Comparing against the test's pump ${testPump} but the sidebar models ${sidebarPump} - set the sidebar pump to ${testPump} first.`
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
      <div className="flex items-center gap-2">
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
        {reason && <span className="text-[11px] text-slate-400">{reason}</span>}
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
