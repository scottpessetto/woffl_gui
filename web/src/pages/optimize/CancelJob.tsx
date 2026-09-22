/**
 * Cancel control for optimize-tab jobs. The server stops the job at its next
 * progress step and frees its slot (with WOFFL_MAX_JOBS=1 on the deployed
 * tier, a mistaken run otherwise blocks every other engineer's panel).
 */

import { useCancelOptimizeJob } from "../../api/hooks";
import type { OptimizeJobStatus } from "../../api/types";
import { WarnNote } from "../../components/ui";

export function CancelJobButton({ jobId, running }: { jobId: string | null; running: boolean }) {
  const cancel = useCancelOptimizeJob();
  if (!jobId || !running) return null;
  return (
    <button
      type="button"
      disabled={cancel.isPending || cancel.isSuccess}
      onClick={() => cancel.mutate(jobId)}
      title="Stop this job at its next step and free the job slot"
      className="rounded-md border border-slate-300 bg-white px-3 py-1.5 text-sm font-medium text-slate-700 hover:bg-slate-50 disabled:opacity-50"
    >
      {cancel.isPending || cancel.isSuccess ? "Cancelling..." : "Cancel"}
    </button>
  );
}

export function CancelledNote({ job }: { job: OptimizeJobStatus | undefined }) {
  if (job?.status !== "cancelled") return null;
  return <WarnNote>Cancelled after {job.seconds.toFixed(0)} s. No result was kept; run again when ready.</WarnNote>;
}
