/**
 * The calibration action bar - ONE calibrate action. "Calibrate to field
 * data" (EventCalibration) fits the pump model against the installed pump
 * era's daily field history server-side; when the era is too young to
 * identify anything, the SERVER falls back to matching the latest test's
 * measured BHP (the old Auto-match BHP mechanics) and the result block says
 * so in plain language. The standalone Auto-match BHP button, its
 * test-selection gates and the pump-mismatch escape hatch are gone - the
 * era fit always targets the pump actually installed today, so there is no
 * "calibrating one pump against another's test" trap to guard against.
 *
 * "Match the test (no gauge)" (MatchTest) is the second calibration, for
 * tests with no downhole BHP: the test's power-fluid rate stands in for
 * the BHP measurement and the fit infers the anchor BHP with the discharge
 * coefficients. It is HIDDEN when the selected test carries a gauge BHP:
 * the inferred BHP reads low on a worn nozzle, and applying it over a real
 * gauge reading made a worn 12 look pumped-off and "recommend" a 9B (user
 * report 2026-09-29). Gating is per TEST, not per well, so a gauge that has
 * since died still leaves its later tests matchable.
 *
 * Match Sensitivities rides along, always enabled - it is the "why doesn't
 * anything reach this test?" explorer, useful exactly when a match is poor.
 */

import { SlidersHorizontal } from "lucide-react";
import { useNavigate } from "react-router-dom";

import type { WellTestRow } from "../../api/types";
import { Button } from "../../components/ui";
import { fmtNum } from "../../lib/format";
import { useParamsStore } from "../../state/params";

import { EventCalibration } from "./EventCalibration";
import { hasGaugeBhp } from "./gaugelessMatch";
import { KcoefExplainer } from "./KcoefExplainer";
import { MatchTest } from "./MatchTest";

export function CalibrateBar({
  well,
  compareTest,
  tests,
  onSelectTest,
}: {
  well: string;
  compareTest: WellTestRow | null;
  /** The page's tests (exclusions applied), for the match's gauge cross-check. */
  tests: WellTestRow[];
  /** Select a test as the comparison test (the match runs on it). */
  onSelectTest: (t: WellTestRow) => void;
}) {
  const modelAsWater = useParamsStore((s) => s.params.model_as_water);
  const replacement = useParamsStore((s) => s.params.pump_state === "replacement");
  const navigate = useNavigate();

  if (modelAsWater) return null; // water mode has no oil-anchored match
  const gauged = hasGaugeBhp(compareTest);

  return (
    <div className="space-y-1.5 border-t border-slate-100 pt-2.5">
      <div className="flex flex-wrap items-center gap-2">
        <EventCalibration well={well} />
        {!replacement && !gauged && (
          <MatchTest well={well} compareTest={compareTest} tests={tests} onSelectTest={onSelectTest} />
        )}
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
      {!replacement && gauged && well !== "Custom" && (
        <p className="text-xs text-slate-500">
          This test has a gauge BHP ({fmtNum(compareTest?.bhp ?? null)} psi), so Calibrate to field
          data uses the measurement. Match the test is only offered for tests without a gauge BHP.
        </p>
      )}
      <KcoefExplainer />
    </div>
  );
}
