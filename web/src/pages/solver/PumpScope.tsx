import { useEffect } from "react";
import { useWellContext } from "../../api/hooks";
import { Button, Card } from "../../components/ui";
import { CLEAN_PUMP, useParamsStore } from "../../state/params";

export function PumpScope() {
  const { well, params, context, months, cap, set, useInstalledPump } = useParamsStore();
  const live = useWellContext(well, months, cap);
  const refresh = useParamsStore((s) => s.refreshPumpContext);
  useEffect(() => { if (live.data) refresh(live.data); }, [live.data, refresh]);
  const scope = live.data?.pump_calibration ?? context?.pump_calibration;
  const pump = context?.pump;
  if (well === "Custom" || !pump) return null;
  const replacement = params.pump_state === "replacement";
  const q = scope?.quality;
  const differs = (Object.keys(CLEAN_PUMP) as Array<keyof typeof CLEAN_PUMP>).some((k) => params[k] !== (scope?.coefficients[k] ?? CLEAN_PUMP[k]));
  const provisional = q && (q.provisional || (q.bounds?.length ?? 0) > 0 ||
    (q.pf ?? 0) > 10 || (q.bhp ?? 0) > 50 ||
    (q.beta != null && q.measured_beta != null && Math.abs(q.beta - q.measured_beta) > .03));
  return (
    <Card>
      <div className="space-y-2 text-xs text-slate-600">
        <p className="font-medium text-slate-700">
          {replacement ? `Clean replacement · ${params.nozzle_no}${params.area_ratio}` :
            `Installed pump · ${pump.nozzle_no}${pump.throat_ratio} · set ${pump.date_set?.slice(0, 10) ?? "date unknown"}`}
        </p>
        <p>{replacement ? "Reference losses and catalog nozzle area. Well and fluid inputs still apply." :
          "Fitted losses and nozzle area describe this installation. They may also absorb test and model error; they do not prove wear."}</p>
        <div className="flex flex-wrap gap-2">
          <Button size="sm" variant="secondary" disabled={!replacement && !differs} onClick={useInstalledPump}>Restore installed pump</Button>
          <Button size="sm" variant="secondary" disabled={replacement} onClick={() => set("pump_state", "replacement")}>Try clean replacement</Button>
        </div>
        {!replacement && differs &&
          <p className="text-amber-700">Session coefficients differ from the saved pump fit. Optimization runs use the saved fit.</p>}
        {scope?.message && <p>{scope.message}</p>}
        {scope?.status === "active" && q && <p className={provisional ? "text-amber-700" : "text-slate-500"}>
          Saved fit: {q.n ?? "?"} points{q.bhp != null ? ` · BHP RMS ${q.bhp} psi` : ""}{q.pf != null ? ` · PF RMS ${q.pf}%` : ""}
          {provisional ? " · provisional; review response and WC uncertainty before acting." : " · review against independent tests."}
        </p>}
      </div>
    </Card>
  );
}
