import { useState } from "react";

import { useMeta, useSaveIpr, useWellInputWritePending } from "../api/hooks";
import type { AnchorMode, IprPinResponse, WellTestRow } from "../api/types";
import { fmtNum } from "../lib/format";
import { changedWellInputs, wellInputProblem, wellInputValues } from "../lib/wellInputs";
import { useParamsStore } from "../state/params";
import { Button, Card } from "./ui";

/** A visible save action shared by the Solver and historical-edit workbench. */
export function SaveWellInputs({ well, anchor }: {
  well: string;
  anchor?: { mode: AnchorMode; test: WellTestRow | null; pin: IprPinResponse | null };
}) {
  const meta = useMeta();
  const save = useSaveIpr(well);
  const writePending = useWellInputWritePending(well);
  const { params, context, matchNote } = useParamsStore();
  const [comment, setComment] = useState<string | null>(null);
  const [notice, setNotice] = useState<{ ok: boolean; text: string } | null>(null);
  const ready = context?.well === well && well !== "Custom";
  const changes = changedWellInputs(params, context?.seeds);
  const problem = wellInputProblem(params);
  const blocked = !ready ? "Select a well and wait for its saved inputs to load." :
    meta.isPending ? "Checking whether saving is available..." :
    meta.isError ? "Save access could not be checked. Refresh the app to try again." :
    !meta.data?.writes_enabled ? "This app is read-only. You can edit and preview; database saving is disabled." : problem;

  const onSave = () => {
    if (blocked || writePending) return;
    const { params: current, context: baseline } = useParamsStore.getState();
    setNotice(null);
    save.mutate({
      ...wellInputValues(current, baseline?.seeds),
      comment: (comment ?? matchNote ?? "").trim() || null,
      pin_wt_uid: anchor?.test?.wt_uid ?? null,
      pin_date: anchor?.test?.date ?? null,
      unpin: anchor?.mode === "manual" && anchor.pin?.status !== "none",
    }, {
      onSuccess: (r) => {
        const warning = r.pin_message && !r.pinned && !r.pin_skipped ? `${r.pin_message} ` : "";
        setNotice({ ok: r.n_values > 0, text: warning + r.values_message + (r.n_values > 0
          ? " New optimization runs will load these saved well inputs. Start a new run to update results." : "") });
        if (r.n_values > 0) setComment("");
      },
      onError: (e) => setNotice({ ok: false, text: e.message }),
    });
  };

  return <Card className="sticky top-0 z-20 space-y-2 shadow-sm">
    <div className="flex flex-wrap items-center justify-between gap-2">
      <div>
        <p className="text-sm font-semibold text-slate-700">Well inputs · {well}</p>
        {ready && <p className={`text-xs ${changes.length ? "text-amber-700" : "text-slate-500"}`}>
          {changes.length ? `${changes.length} well ${changes.length === 1 ? "input differs" : "inputs differ"} from the database. Edits are only in this session until saved.` : "Well inputs match the loaded database values."}
        </p>}
      </div>
      <Button variant="primary" disabled={!!blocked || writePending} busy={save.isPending}
        title={blocked ?? `Save ${well}'s displayed IPR and supported fluid inputs for future optimization runs`}
        onClick={onSave}>Save well inputs</Button>
    </div>
    {blocked && <p className="text-xs text-slate-500">{blocked}</p>}
    {notice && <p role="status" className={`text-xs ${notice.ok ? "text-emerald-700" : "text-amber-700"}`}>{notice.text}</p>}
    <details className="text-xs text-slate-600">
      <summary className="cursor-pointer">Review values and add a save note</summary>
      <div className="mt-2 space-y-2">
        <p>IPR: {fmtNum(params.qwf)} BLPD at {fmtNum(params.pwf)} psi; reservoir pressure {fmtNum(params.pres)} psi.
          WC {fmtNum(100 * params.form_wc, 1)}%; GOR {fmtNum(params.form_gor)} scf/STB; WHP {fmtNum(params.surf_pres)} psi.</p>
        <p>Changed temperature ({fmtNum(params.form_temp)} °F) and bubble point ({fmtNum(params.bubble_point)} psi) are also saved.
          Pump coefficients and the hydraulic model are saved through installed-pump calibration. Other sidebar settings remain session inputs.</p>
        <input type="text" aria-label="Well save note" value={comment ?? matchNote ?? ""}
          maxLength={500} disabled={save.isPending} onChange={(e) => setComment(e.target.value)}
          placeholder="Why these values? (optional)"
          className="h-8 w-full rounded border border-slate-300 px-2 text-sm" />
      </div>
    </details>
  </Card>;
}
