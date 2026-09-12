import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api, get, post, stableStringify, retryJobPoll, jobPollDelay, isMissingJob } from "../../api/client";
import { Card, WarnNote } from "../../components/ui";
import { fmtNum } from "../../lib/format";

interface Request {
  source_job_id: string;
  wc_points: number;
  gor_percent: number;
  header_psi: number;
  joint_cases: boolean;
}
interface Score {
  plan: string;
  feasible: boolean | null;
  oil: number | null;
  machine_water: number | null;
  budget: number | null;
  regret: number | null;
  reason: string;
}
interface Result {
  request: Request;
  water_key: string;
  assumptions: string;
  min_comparable_oil_gain: number | null;
  max_comparable_oil_gain: number | null;
  cases: { case: { name: string }; header_psi: number; plans: Score[]; proposed_oil_delta: number | null; preferred: string[] }[];
  plans: { plan: string; feasible_cases: number; failed_cases: number; infeasible_cases: number; preferred_when_both_feasible: number; comparable_cases: number; max_regret: number | null }[];
}
interface Job {
  status: "running" | "done" | "error" | "cancelled";
  progress: string | null;
  error: string | null;
  result: Result | null;
}
const INPUT = "h-8 w-20 rounded border border-slate-300 px-2 text-sm";

export function PlanRobustness({ sourceJobId, available, unavailableReason }: {
  sourceJobId: string; available: boolean; unavailableReason?: string | null;
}) {
  const [wc, setWc] = useState(3);
  const [gor, setGor] = useState(20);
  const [header, setHeader] = useState(100);
  const [joint, setJoint] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const request: Request = { source_job_id: sourceJobId, wc_points: wc, gor_percent: gor, header_psi: header, joint_cases: joint };
  const start = useMutation({ mutationFn: (r: Request) => post<{ job_id: string }>("/optimize/robustness", r), onSuccess: (r) => setJobId(r.job_id) });
  const job = useQuery({ queryKey: ["plan-robustness", jobId],
    queryFn: ({ signal }) => get<Job>(`/optimize/robustness/${jobId}`, signal), enabled: !!jobId,
    refetchInterval: (q) => isMissingJob(q.state.error) ? false : !q.state.data || q.state.data.status === "running" ? 2500 : false,
    refetchIntervalInBackground: true, retry: retryJobPoll, retryDelay: jobPollDelay,
  });
  const cancel = useMutation({ mutationFn: () => api(`/optimize/robustness/${jobId}`, { method: "DELETE" }) });
  const running = start.isPending || job.data?.status === "running";
  const result = job.data?.status === "done" && stableStringify(job.data.result?.request) === stableStringify(request) ? job.data.result : null;
  const valid = Number.isFinite(wc) && wc >= 0 && wc <= 10 && Number.isFinite(gor) && gor >= 0 && gor <= 50 && Number.isFinite(header) && header >= 0 && header <= 250;
  return <Card>
    <details>
      <summary className="cursor-pointer text-sm font-semibold text-slate-700">Stress-test current and proposed plans</summary>
      <div className="mt-3 space-y-3">
        {!available ? <p className="text-xs text-slate-600">{unavailableReason ?? "Run a complete I/M/E JPCO comparison to capture the fixed plans. S-Pad and CFP require different coupled scenario controls."}</p> : <>
          <p className="text-xs text-slate-600">Compare the same two pump plans under user-selected stress cases. These starting ranges are engineering assumptions; adjust them to the well data. The oil IPR stays fixed. All wells move together in each WC/GOR case; header cases check shared plant capacity. No inputs are saved.</p>
          <div className="flex flex-wrap items-end gap-4">
            <label className="grid gap-1 text-xs">WC +/- percentage points<input aria-label="Plan stress WC range" className={INPUT} type="number" min={0} max={10} step={.5} value={wc} onChange={(e) => setWc(e.target.valueAsNumber)} /></label>
            <label className="grid gap-1 text-xs">GOR +/- percent<input aria-label="Plan stress GOR range" className={INPUT} type="number" min={0} max={50} step={5} value={gor} onChange={(e) => setGor(e.target.valueAsNumber)} /></label>
            <label className="grid gap-1 text-xs">Shared header +/- psi<input aria-label="Plan stress header range" className={INPUT} type="number" min={0} max={250} step={25} value={header} onChange={(e) => setHeader(e.target.valueAsNumber)} /></label>
            <label className="flex items-center gap-2 text-xs"><input type="checkbox" checked={joint} onChange={(e) => setJoint(e.target.checked)} />Add two joint corners</label>
          </div>
          {joint && <p className="text-xs text-slate-500">Joint assumptions: higher WC/GOR with lower header; lower WC/GOR with higher header. At most nine cases. This co-movement is a scenario assumption.</p>}
          <div className="flex gap-3">
            <button type="button" disabled={running || !valid} onClick={() => start.mutate(request)} className="rounded bg-blue-600 px-3 py-1.5 text-sm text-white disabled:opacity-50">{running ? "Checking plans..." : "Run plan stress cases"}</button>
            {job.data?.status === "running" && <button type="button" disabled={cancel.isPending} onClick={() => cancel.mutate()} className="text-sm text-slate-600">Cancel</button>}
          </div>
          {running && <p className="text-xs text-slate-500">{job.data?.progress ?? "Starting comparison..."}</p>}
          {start.isError && <WarnNote>{start.error.message}</WarnNote>}
          {cancel.isError && <WarnNote>{cancel.error.message}</WarnNote>}
          {job.isError && <WarnNote>{job.error instanceof Error ? job.error.message : String(job.error)}</WarnNote>}
          {job.data?.status === "error" && <WarnNote>{job.data.error}</WarnNote>}
          {job.data?.status === "cancelled" && <p className="text-xs text-slate-500">Comparison cancelled.</p>}
          {job.data?.status === "done" && !result && <p className="text-xs text-slate-500">Ranges changed. Run again to compare these cases.</p>}
          {result && <>
            <p className="text-xs text-slate-600">{result.assumptions}</p>
            <p className="text-sm text-slate-700">{result.min_comparable_oil_gain === null ? "No oil-gain range: both plans must solve within plant constraints in the same case." : `Proposed oil change when both plans are feasible: ${fmtNum(result.min_comparable_oil_gain)} to ${fmtNum(result.max_comparable_oil_gain)} BOPD.`}</p>
            {result.plans.map((p) => <p key={p.plan} className="text-xs text-slate-600"><strong>{p.plan}:</strong> {p.feasible_cases}/{result.cases.length} cases feasible; {p.infeasible_cases} exceed operating constraints; {p.failed_cases} unknown. Preferred or tied in {p.preferred_when_both_feasible}/{p.comparable_cases} comparable cases. Maximum regret versus the other feasible plan: {fmtNum(p.max_regret)} BOPD-equivalent.</p>)}
            <div className="overflow-x-auto"><table className="w-full min-w-[42rem] text-xs"><thead><tr className="border-b text-left text-slate-500"><th>Case / header</th><th>Plan</th><th>Oil (BOPD)</th><th>{result.water_key === "totl_wat" ? "Machine water" : "PF"} / budget (BPD)</th><th>Outcome</th><th>Regret*</th></tr></thead><tbody>
              {result.cases.flatMap((c) => c.plans.map((p) => <tr key={`${c.case.name}-${p.plan}`} className="border-b border-slate-100"><td className="py-1.5">{c.case.name}<span className="block text-slate-400">{fmtNum(c.header_psi)} psi</span></td><td>{p.plan}</td><td>{fmtNum(p.oil)}</td><td>{fmtNum(p.machine_water)} / {fmtNum(p.budget)}</td><td title={p.reason} className={p.feasible === true ? "text-slate-600" : "text-amber-700"}>{p.feasible === true ? "Feasible" : p.feasible === false ? "Outside constraints" : "Unknown"}<span className="block max-w-64 text-[10px]">{p.reason}</span></td><td>{fmtNum(p.regret, 1)}</td></tr>))}
            </tbody></table></div>
            <p className="text-[10px] text-slate-500">*Oil minus the source run's water price times the pad's machine-water stream. Regret compares these two feasible plans only. Case counts are not probabilities; failed and infeasible cases are excluded from numeric gain ranges.</p>
          </>}
        </>}
      </div>
    </details>
  </Card>;
}
