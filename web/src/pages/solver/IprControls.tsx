/**
 * IPR anchor + comparison-test controls: mirror of the anchor selector in
 * woffl/gui/tabs/jetpump_solver.py:_render_ipr_anchor_and_seed and the
 * synced/decoupled comparison picker. Read-only with respect to the saved
 * pin (no save/pin write buttons here); "Apply IPR to inputs" lays the fit
 * seeds over the sidebar params.
 */

import type { AnchorMode, IprFitResponse, IprPinResponse, JpInstallRow, WellTestRow } from "../../api/types";
import { Badge, Button, Card, Section } from "../../components/ui";
import { useParamsStore } from "../../state/params";

import { pumpLabelAt, testKey, testLabel } from "./selection";

const SELECT_CLS =
  "mt-1 h-8 w-full rounded-md border border-slate-300 bg-white px-2 text-sm " +
  "text-slate-800 outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-200";

export function IprControls({
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
      </Card>
    </Section>
  );
}
