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
 * Match Sensitivities rides along, always enabled - it is the "why doesn't
 * anything reach this test?" explorer, useful exactly when a match is poor.
 */

import { SlidersHorizontal } from "lucide-react";
import { useNavigate } from "react-router-dom";

import { Button } from "../../components/ui";
import { useParamsStore } from "../../state/params";

import { EventCalibration } from "./EventCalibration";
import { KcoefExplainer } from "./KcoefExplainer";

export function CalibrateBar({ well }: { well: string }) {
  const modelAsWater = useParamsStore((s) => s.params.model_as_water);
  const navigate = useNavigate();

  if (modelAsWater) return null; // water mode has no oil-anchored match

  return (
    <div className="space-y-1.5 border-t border-slate-100 pt-2.5">
      <div className="flex flex-wrap items-center gap-2">
        <EventCalibration well={well} />
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
      <KcoefExplainer />
    </div>
  );
}
