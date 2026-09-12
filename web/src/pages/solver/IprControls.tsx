/**
 * IPR anchor + comparison-test controls: the anchor selector and the
 * synced/decoupled comparison picker. "Apply IPR to inputs" lays the fit
 * seeds over the sidebar params.
 *
 * Saving lives in the always-visible SaveWellInputs bar above the workbench.
 * Anchor clearing stays beside the selector and explains read-only access.
 */

import { useState } from "react";

import { useClearIprPin, useMeta, useWellInputWritePending } from "../../api/hooks";
import type { AnchorMode, IprFitResponse, IprPinResponse, JpInstallRow, SimParams, WellTestRow } from "../../api/types";
import { Badge, Button, Card, InfoNote, Section } from "../../components/ui";
import { fmtDate, fmtNum } from "../../lib/format";
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
  const [notice, setNotice] = useState<{ tone: "ok" | "warn"; text: string } | null>(null);
  const manualFields = useParamsStore((s) => s.manualFields);
  const matchNote = useParamsStore((s) => s.matchNote);
  // Only ownership that actually blocks something is worth reporting: a
  // hand-picked nozzle is "manual" too, but the fit never seeds it.
  const heldFromFit = fit
    ? (Object.keys(fit.seeds) as Array<keyof SimParams>).filter((k) => manualFields.has(k))
    : [];
  const clearMut = useClearIprPin(well);

  // Prefer the fit's own anchor resolution (server truth for median/recent);
  // the local mirror covers the gap while the fit is loading. Manual mode
  // must ignore a stale fit - there is no test behind a manual point.
  const anchorTest = resolveAnchorTest(
    tests,
    anchorMode,
    anchorDate,
    anchorMode === "manual" ? null : (fit?.coeffs.anchor_date ?? null),
  );
  const busy = useWellInputWritePending(well);

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
            <option value="median">Median - BHP</option>
            <option value="median_liq">Median - Liquid rate</option>
            <option value="specific">Specific test</option>
            <option value="manual">Manual point (no test)</option>
          </select>
        </label>
        {(anchorMode === "median" || anchorMode === "median_liq") && anchorTest && (
          <p className="text-[11px] text-slate-500">
            Anchored on the {fmtDate(anchorTest.date)} test - the one whose{" "}
            {anchorMode === "median" ? "BHP" : "liquid rate"} sits nearest the window's
            median: Liq {fmtNum(anchorTest.total_fluid)} BPD, Oil {fmtNum(anchorTest.oil)}{" "}
            BOPD, BHP {fmtNum(anchorTest.bhp)} psi.
          </p>
        )}

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

        {anchorMode === "manual" && (
          <InfoNote>
            The anchor is the sidebar's own qwf / pwf, not a well test - what a
            joint match, a backmatched BHP or an applied permutation produces.
            No Vogel fit runs against it, and saving records it as a manual
            point with no test pinned behind it.
          </InfoNote>
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
            // `release`: an explicit click hands the seeded fields back to the
            // fit, so this button still does what it says even after a
            // permutation or a hand edit claimed them.
            if (fit) useParamsStore.getState().applyIprSeeds(fit.seeds, true);
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

        {(heldFromFit.length > 0 || matchNote !== null) && (
          <p className="text-xs text-slate-500">
            {matchNote !== null && <span className="font-medium text-slate-700">{matchNote}. </span>}
            {heldFromFit.length > 0 && (
              <>
                <span title={heldFromFit.join(", ")}>
                  {heldFromFit.length === 1
                    ? "1 inflow input is set by hand, so the fit leaves it alone"
                    : `${heldFromFit.length} inflow inputs are set by hand, so the fit leaves them alone`}
                </span>
                {" - "}
                <button
                  type="button"
                  className="underline-offset-2 hover:text-slate-700 hover:underline"
                  onClick={() => {
                    if (fit) useParamsStore.getState().applyIprSeeds(fit.seeds, true);
                  }}
                >
                  take the fit instead
                </button>
              </>
            )}
          </p>
        )}

        {(pin?.status === "applied" || pin?.status === "stale") && <div className="space-y-1 border-t border-slate-100 pt-3">
          <Button size="sm" disabled={!writesOn || busy} onClick={onClear}>Clear saved IPR</Button>
          {!writesOn && <p className="text-xs text-slate-500">Clearing the saved anchor is unavailable in read-only mode.</p>}
        </div>}
        {notice && <p className={notice.tone === "ok" ? "text-xs text-emerald-700" : "text-xs text-amber-700"}>{notice.text}</p>}
      </Card>
    </Section>
  );
}
