/**
 * IPR anchor + comparison-test controls: mirror of the anchor selector in
 * woffl/gui/tabs/jetpump_solver.py:_render_ipr_anchor_and_seed and the
 * synced/decoupled comparison picker. "Apply IPR to inputs" lays the fit
 * seeds over the sidebar params.
 *
 * Save block (mirror of _render_ipr_pin_controls): "Save as well default"
 * pins the resolved anchor test AND pushes the sidebar's current curve +
 * rate values to mpu.wells.prop_hist in one click; "Clear saved IPR"
 * un-pins. HIDDEN entirely (not disabled) when /meta reports
 * writes_enabled=false - the Streamlit gate pre-check contract. The rules
 * live server-side in woffl.gui.ipr_anchor: as-built physical properties
 * can never be written, friction rides along only when calibrated.
 */

import { useState } from "react";

import { useClearIprPin, useMeta, useSaveIpr } from "../../api/hooks";
import type { AnchorMode, IprFitResponse, IprPinResponse, JpInstallRow, WellTestRow } from "../../api/types";
import { Badge, Button, Card, Section } from "../../components/ui";
import { useParamsStore } from "../../state/params";

import { pumpLabelAt, resolveAnchorTest, testKey, testLabel } from "./selection";

const SELECT_CLS =
  "mt-1 h-8 w-full rounded-md border border-slate-300 bg-white px-2 text-sm " +
  "text-slate-800 outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-200";

export function IprControls({
  well,
  anchorMode,
  anchorDate,
  onAnchorChange,
  tests,
  installs,
  fit,
  pin,
  decouple,
  onDecouple,
  compareKey,
  onCompareChange,
}: {
  well: string;
  anchorMode: AnchorMode;
  anchorDate: string | null;
  onAnchorChange: (mode: AnchorMode, date: string | null) => void;
  tests: WellTestRow[];
  installs: JpInstallRow[];
  fit: IprFitResponse | null;
  pin: IprPinResponse | null;
  decouple: boolean;
  onDecouple: (value: boolean) => void;
  compareKey: string | null;
  onCompareChange: (key: string) => void;
}) {
  const meta = useMeta();
  const writesOn = meta.data?.writes_enabled === true;
  const [comment, setComment] = useState("");
  const [notice, setNotice] = useState<{ tone: "ok" | "warn"; text: string } | null>(null);
  const saveMut = useSaveIpr(well);
  const clearMut = useClearIprPin(well);

  const anchorTest = resolveAnchorTest(tests, anchorMode, anchorDate);
  const busy = saveMut.isPending || clearMut.isPending;

  const onSave = () => {
    const p = useParamsStore.getState().params;
    setNotice(null);
    saveMut.mutate(
      {
        // Sidebar qwf is TOTAL LIQUID and prop_hist.ipr_qwf_liq stores
        // liquid, so it goes across verbatim (the B-28 rule).
        qwf_liq: p.qwf,
        pwf: p.pwf,
        res_pres: p.pres,
        form_wc: p.form_wc,
        form_gor: p.form_gor,
        surf_pres: p.surf_pres,
        // BHP-calibrated friction; the server skips unchanged/default values.
        ken: p.ken,
        kth: p.kth,
        kdi: p.kdi,
        comment: comment.trim() || null,
        pin_wt_uid: anchorTest?.wt_uid ?? null,
        pin_date: anchorTest?.date ?? null,
      },
      {
        onSuccess: (r) => {
          setComment("");
          const parts = [r.values_message];
          if (r.pin_message && !r.pinned && !r.pin_skipped) parts.unshift(r.pin_message);
          setNotice({
            tone: r.n_values > 0 ? "ok" : "warn",
            text: parts.join(" "),
          });
        },
        onError: (e) => setNotice({ tone: "warn", text: e.message }),
      },
    );
  };

  const onClear = () => {
    setNotice(null);
    clearMut.mutate(undefined, {
      onSuccess: (r) => setNotice({ tone: r.cleared ? "ok" : "warn", text: r.message }),
      onError: (e) => setNotice({ tone: "warn", text: e.message }),
    });
  };

  return (
    <Section title="IPR Anchor">
      <Card className="space-y-3">
        <label className="block">
          <span className="text-xs font-medium text-slate-500">IPR anchor</span>
          <select
            value={anchorMode}
            onChange={(e) => {
              const mode = e.target.value as AnchorMode;
              onAnchorChange(mode, mode === "specific" ? (anchorDate ?? tests[0]?.date ?? null) : null);
            }}
            className={SELECT_CLS}
          >
            <option value="recent">Most recent</option>
            <option value="median">Median test</option>
            <option value="specific">Specific test</option>
          </select>
        </label>

        {anchorMode === "specific" && (
          <label className="block">
            <span className="text-xs font-medium text-slate-500">Anchor test</span>
            <select
              value={anchorDate ?? tests[0]?.date ?? ""}
              onChange={(e) => onAnchorChange("specific", e.target.value)}
              className={SELECT_CLS}
            >
              {tests.map((t) => (
                <option key={testKey(t)} value={t.date}>
                  {testLabel(t, pumpLabelAt(installs, t.date))}
                </option>
              ))}
            </select>
          </label>
        )}

        {pin?.status === "applied" && (
          <Badge tone="info">
            Saved anchor: {pin.date_token ?? "?"} by {pin.entry_user ?? "unknown"}
          </Badge>
        )}
        {pin?.status === "stale" && <Badge tone="fair">Saved anchor outside current window</Badge>}

        <Button
          variant="secondary"
          disabled={!fit}
          title="Lay the fitted qwf / pwf / ResP / WC / GOR seeds over the sidebar inputs"
          onClick={() => {
            if (fit) useParamsStore.getState().setMany(fit.seeds);
          }}
        >
          Apply IPR to inputs
        </Button>

        <label className="flex cursor-pointer items-center gap-2">
          <input
            type="checkbox"
            checked={decouple}
            onChange={(e) => onDecouple(e.target.checked)}
            className="h-4 w-4 rounded border-slate-300 accent-blue-600"
          />
          <span className="text-xs text-slate-600">
            Use a different test for comparison (un-sync from the IPR anchor)
          </span>
        </label>

        {decouple && (
          <label className="block">
            <span className="text-xs font-medium text-slate-500">Compare against</span>
            <select
              value={compareKey ?? (tests[0] ? testKey(tests[0]) : "")}
              onChange={(e) => onCompareChange(e.target.value)}
              className={SELECT_CLS}
            >
              {tests.map((t) => (
                <option key={testKey(t)} value={testKey(t)}>
                  {testLabel(t, pumpLabelAt(installs, t.date))}
                </option>
              ))}
            </select>
          </label>
        )}

        {writesOn && (
          <div className="space-y-2 border-t border-slate-100 pt-3">
            <input
              type="text"
              value={comment}
              maxLength={500}
              onChange={(e) => setComment(e.target.value)}
              placeholder="Why these values? (optional)"
              title={
                "Saved with the values in mpu.wells.woffl_eng_comment and shown in the " +
                "well's save history, so the next person sees WHY these numbers were chosen."
              }
              className={SELECT_CLS}
            />
            <div className="flex items-center gap-2">
              <Button
                variant="primary"
                disabled={busy}
                busy={saveMut.isPending}
                title={
                  anchorTest?.wt_uid != null
                    ? `Saves test ${anchorTest.date} as this well's default IPR anchor AND ` +
                      "the sidebar's current curve + rate values (mpu.wells.prop_hist) so the " +
                      "well opens exactly like this in every future session."
                    : "No pinnable anchor test (manual/provisional) - saves the sidebar's " +
                      "current curve + rate values only."
                }
                onClick={onSave}
              >
                Save as well default
              </Button>
              {(pin?.status === "applied" || pin?.status === "stale") && (
                <button
                  type="button"
                  disabled={busy}
                  className="text-xs text-slate-500 underline-offset-2 hover:text-slate-700 hover:underline disabled:opacity-50"
                  title="Removes the saved default so this well falls back to the most-recent test"
                  onClick={onClear}
                >
                  Clear saved IPR
                </button>
              )}
            </div>
            {notice && (
              <p className={notice.tone === "ok" ? "text-xs text-emerald-700" : "text-xs text-amber-700"}>
                {notice.text}
              </p>
            )}
          </div>
        )}
      </Card>
    </Section>
  );
}
