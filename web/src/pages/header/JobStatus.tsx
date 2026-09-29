/** Progress / error lines for header-page background jobs. */

import { useCancelHeaderJob } from "../../api/hooks";
import type { HeaderJobStatus } from "../../api/types";
import { ErrorNote, WarnNote } from "../../components/ui";

export function JobLine({ job, jobId, label }: { job: HeaderJobStatus | undefined; jobId: string | null; label: string }) {
  const cancel = useCancelHeaderJob();
  if (!job || job.status !== "running") return null;
  return (
    <div className="flex items-center gap-3 text-sm text-slate-600">
      <span className="h-2 w-2 animate-pulse rounded-full bg-blue-500" />
      <span>{label}: {job.progress ?? "working"} ({job.seconds.toFixed(0)} s)</span>
      {jobId && (
        <button
          type="button"
          className="rounded-md border border-slate-300 bg-white px-2 py-0.5 text-xs hover:bg-slate-50 disabled:opacity-50"
          disabled={cancel.isPending}
          onClick={() => cancel.mutate(jobId)}
        >
          {cancel.isPending ? "Cancelling..." : "Cancel"}
        </button>
      )}
    </div>
  );
}

export function JobError({ job }: { job: HeaderJobStatus | undefined }) {
  if (!job) return null;
  if (job.status === "error") return <ErrorNote error={new Error(job.error ?? "The job failed.")} />;
  if (job.status === "cancelled") return <WarnNote>Cancelled after {job.seconds.toFixed(0)} s. No result was kept.</WarnNote>;
  return null;
}
