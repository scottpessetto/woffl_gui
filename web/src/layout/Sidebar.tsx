/**
 * The 300px parameter sidebar - the full SimParams control tree, mirroring
 * woffl/gui/sidebar.py's layout: sticky Run action, Well Selection, Field
 * Model, Pump, Pressures, then collapsible Inflow & Formation, Geometry,
 * and Advanced sections. All values live in useParamsStore.
 */

import { useState } from "react";
import type { KeyboardEvent } from "react";
import { Lock, LockOpen } from "lucide-react";

import { useMeta, usePropLock } from "../api/hooks";
import { NOZZLE_OPTIONS, THROAT_OPTIONS } from "../api/types";
import { Button } from "../components/ui";
import { useParamsStore, type PropLocks } from "../state/params";
import { CheckboxField, NumberField, RadioRow, SelectField } from "./ParamFields";
import { WellSelector } from "./WellSelector";

const WINDOW_INPUT_CLS =
  "h-8 w-full rounded-md border border-slate-300 bg-white px-2 text-sm tabular-nums " +
  "text-slate-800 outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-200";

function SectionHeader({ text }: { text: string }) {
  return (
    <h3 className="pt-3 text-[11px] font-semibold uppercase tracking-wide text-slate-400">
      {text}
    </h3>
  );
}

function SubHeader({ text }: { text: string }) {
  return <p className="text-[11px] font-semibold text-slate-500">{text}</p>;
}

/** Small amber chip for values pinned by a saved IPR (prop locks). */
function LockChip({ text }: { text: string }) {
  return (
    <span
      title="Pinned by saved IPR values for this well"
      className="shrink-0 rounded bg-amber-50 px-1 py-px text-[10px] font-medium text-amber-700"
    >
      {text}
    </span>
  );
}

// Sidebar param key + short label per lockable field (lock keys follow the
// prop_hist registry: res_pres lock rides the sidebar's `pres` field).
const LOCK_META: Record<keyof PropLocks, { label: string; paramKey: "form_wc" | "form_gor" | "pres" }> = {
  form_wc: { label: "WC", paramKey: "form_wc" },
  form_gor: { label: "GOR", paramKey: "form_gor" },
  res_pres: { label: "ResP", paramKey: "pres" },
};

/**
 * The WC/GOR/ResP lock toggle (port of the Solver's 🔒 checkboxes,
 * jetpump_solver._render_ipr_pin_controls lock block). Locking pins the
 * sidebar's CURRENT value in the same click; unlocking hands the field back
 * to the automated seed. Writes off -> falls back to the passive saved-lock
 * chip; hypothetical wells have nothing to lock.
 */
function PropLockToggle({ field }: { field: keyof PropLocks }) {
  const well = useParamsStore((s) => s.well);
  const lock = useParamsStore((s) => s.propLocks[field]);
  const setPropLock = useParamsStore((s) => s.setPropLock);
  const meta = useMeta();
  const mut = usePropLock(well);
  const [err, setErr] = useState<string | null>(null);
  const { label, paramKey } = LOCK_META[field];

  if (well === "Custom") return null;
  if (meta.data?.writes_enabled !== true) {
    return lock.locked ? <LockChip text={`${label} locked (saved)`} /> : null;
  }

  const toggle = () => {
    setErr(null);
    const next = !lock.locked;
    mut.mutate(
      { field, locked: next, value: next ? useParamsStore.getState().params[paramKey] : null },
      {
        onSuccess: (r) => {
          if (r.ok) setPropLock(field, { locked: r.locked, value: r.value });
          else setErr(r.message);
        },
        onError: (e) => setErr(e.message),
      },
    );
  };

  return (
    <button
      type="button"
      onClick={toggle}
      disabled={mut.isPending}
      title={
        err ??
        (lock.locked
          ? `${label} is locked - the saved value overrides every test-derived seed. Click to unlock.`
          : `Lock ${label} at the current sidebar value - it will override every test-derived seed (on open, on anchor change) until unlocked. For wells where the automated ${label} is systematically wrong.`)
      }
      className={
        "flex shrink-0 items-center gap-0.5 rounded px-1 py-px text-[10px] font-medium transition-colors disabled:opacity-50 " +
        (err
          ? "bg-red-50 text-red-700"
          : lock.locked
            ? "bg-amber-50 text-amber-700 hover:bg-amber-100"
            : "border border-slate-200 bg-white text-slate-400 hover:border-slate-300 hover:bg-slate-50 hover:text-slate-600")
      }
    >
      {lock.locked ? <Lock className="h-2.5 w-2.5" /> : <LockOpen className="h-2.5 w-2.5" />}
      {err ? "failed" : lock.locked ? `${label} locked` : "lock"}
    </button>
  );
}

/**
 * Draft-and-commit numeric input for the test window (months/cap). These are
 * fetch-window knobs, not SimParams fields, so they bypass ParamFields and
 * commit through store.setWindow with local clamping.
 */
function WindowInput({
  label,
  value,
  min,
  max,
  onCommit,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  onCommit: (v: number) => void;
}) {
  const [draft, setDraft] = useState<string | null>(null);

  const commit = () => {
    if (draft !== null) {
      const parsed = Number(draft);
      if (draft.trim() !== "" && Number.isFinite(parsed)) {
        onCommit(Math.min(max, Math.max(min, Math.round(parsed))));
      }
    }
    setDraft(null);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      commit();
      e.currentTarget.blur();
    } else if (e.key === "Escape") {
      setDraft(null);
      e.currentTarget.blur();
    }
  };

  return (
    <label className="block">
      <span className="text-xs text-slate-500">{label}</span>
      <input
        type="number"
        min={min}
        max={max}
        step={1}
        value={draft ?? String(value)}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={onKeyDown}
        className={`mt-1 ${WINDOW_INPUT_CLS}`}
      />
    </label>
  );
}

/** Provenance caption under the PF pressure field (mirrors _render_pf_live_caption). */
function PfProvenance() {
  const context = useParamsStore((s) => s.context);
  const ppfSurf = useParamsStore((s) => s.params.ppf_surf);
  if (!context) return null;

  const pf = context.pf;
  if (pf !== null && (pf.kind === "test day" || pf.kind === "latest daily") && pf.pf_press !== null) {
    return (
      <p className="mt-1 text-[11px] text-slate-400">
        {`Live: ${Math.round(pf.pf_press).toLocaleString("en-US")} psi from ${
          pf.pf_source ?? "unknown source"
        } on ${pf.pf_date ?? "unknown date"} (${pf.kind})`}
      </p>
    );
  }

  // Only the SERVER's seed is the pad default; the current field value may
  // be the engineer's edit and must not be captioned as the default (WEB-17).
  const seeded = pf?.ppf_surf ?? null;
  return (
    <p className="mt-1 text-[11px] text-slate-400">
      {seeded !== null
        ? `No live PF reading - seeded pad default ${Math.round(seeded).toLocaleString("en-US")} psi`
        : ppfSurf !== undefined
          ? "No live PF reading - pad default seeded; value shown is the current entry"
          : "No live PF reading"}
    </p>
  );
}

/**
 * Seeds the server's widget bounds ALTERED on the way in. A clamped seed is
 * not the well's value and must never pass as one (review 2026-09-01, SRV-9).
 */
function SeedClampNotes() {
  const context = useParamsStore((s) => s.context);
  const clamped = context?.clamped ?? [];
  if (clamped.length === 0) return null;
  return (
    <p className="mb-1 rounded border border-amber-200 bg-amber-50 px-2 py-1 text-[11px] text-amber-800">
      {`Seed${clamped.length > 1 ? "s" : ""} clamped to the input bounds: ${clamped.join("; ")}`}
    </p>
  );
}

/**
 * Where the installed-pump identity came from. The tracker is live; the
 * bundled spreadsheet is a dated snapshot, so a pump seeded from it may be
 * months out of date and must say so (review 2026-09-01, DATA-5).
 */
function PumpProvenance() {
  const context = useParamsStore((s) => s.context);
  if (!context) return null;
  const pump = context.pump;
  if (!pump || (!pump.nozzle_no && !pump.throat_ratio)) {
    return <p className="mt-1 text-[11px] text-slate-400">No install on record for this well</p>;
  }
  const code = `${pump.nozzle_no ?? "?"}${pump.throat_ratio ?? "?"}`;
  const when = pump.date_set ? ` set ${pump.date_set}` : "";
  if (pump.source === "excel_fallback") {
    return (
      <p className="mt-1 text-[11px] text-amber-700">
        {`Installed ${code}${when} per the BUNDLED SPREADSHEET SNAPSHOT (tracker unavailable) - may be stale`}
      </p>
    );
  }
  return <p className="mt-1 text-[11px] text-slate-400">{`Installed ${code}${when} (JP tracker)`}</p>;
}

export function Sidebar() {
  const run = useParamsStore((s) => s.run);
  const months = useParamsStore((s) => s.months);
  const cap = useParamsStore((s) => s.cap);
  const setWindow = useParamsStore((s) => s.setWindow);
  const asBuiltLocks = useParamsStore((s) => s.asBuiltLocks);

  return (
    <aside className="flex h-full w-[300px] flex-col overflow-y-auto border-r border-slate-200 bg-white">
      <div className="sticky top-0 z-10 border-b border-slate-200 bg-white p-3">
        <Button variant="primary" className="w-full" onClick={run}>
          Run simulation
        </Button>
        <p className="mt-1 text-center text-[11px] text-slate-400">
          re-solves automatically as inputs change
        </p>
      </div>

      <div className="divide-y divide-slate-200 px-3 pb-8">
        <section className="pb-3">
          <SectionHeader text="Well Selection" />
          <div className="mt-2 space-y-2">
            <WellSelector />
            <details>
              <summary className="cursor-pointer select-none text-xs text-slate-500 hover:text-slate-700">
                Well Test History
              </summary>
              <div className="mt-2 grid grid-cols-2 gap-2">
                <WindowInput
                  label="Lookback (months)"
                  value={months}
                  min={1}
                  max={24}
                  onCommit={(v) => setWindow(v, cap)}
                />
                <WindowInput
                  label="Max tests (0 = all)"
                  value={cap}
                  min={0}
                  max={50}
                  onCommit={(v) => setWindow(months, v)}
                />
              </div>
            </details>
          </div>
        </section>

        <section className="pb-3">
          <SectionHeader text="Field Model" />
          <div className="mt-2">
            <RadioRow
              field="field_model"
              options={[
                { value: "Schrader", label: "Schrader" },
                { value: "Kuparuk", label: "Kuparuk" },
              ]}
            />
          </div>
        </section>

        <section className="pb-3">
          <SectionHeader text="Pump" />
          <div className="mt-2 space-y-2">
            <RadioRow
              label="Circulation Direction"
              field="jpump_direction"
              options={[
                {
                  value: "reverse",
                  label: "Reverse",
                  hint: "Power fluid down annulus, production up tubing",
                },
                {
                  value: "forward",
                  label: "Forward",
                  hint: "Power fluid down tubing, production up annulus",
                },
              ]}
            />
            <div className="grid grid-cols-2 gap-2">
              <SelectField label="Nozzle" field="nozzle_no" options={NOZZLE_OPTIONS} />
              <SelectField label="Throat" field="area_ratio" options={THROAT_OPTIONS} />
            </div>
            <PumpProvenance />
          </div>
        </section>

        <section className="pb-3">
          <SectionHeader text="Pressures" />
          <div className="mt-2 space-y-2">
            <div>
              <NumberField
                label="Power Fluid Surface Pressure (psi)"
                field="ppf_surf"
                step={10}
              />
              <PfProvenance />
            </div>
            <NumberField label="Wellhead Surface Pressure (psi)" field="surf_pres" step={10} />
          </div>
        </section>

        <section className="pb-3">
          <details open>
            <summary className="cursor-pointer select-none pt-3 text-[11px] font-semibold uppercase tracking-wide text-slate-400 hover:text-slate-600">
              Inflow &amp; Formation
            </summary>
            <div className="mt-2 space-y-2">
              <SeedClampNotes />
              <NumberField
                label="Total Liquid Rate at FBHP (qwf, BLPD)"
                field="qwf"
                step={10}
              />
              <NumberField label="Flowing BHP @ qwf (pwf, psi)" field="pwf" step={10} />
              <NumberField
                label="Reservoir Pressure (psi)"
                field="pres"
                step={10}
                chip={<PropLockToggle field="res_pres" />}
              />
              <NumberField
                label="Water Cut"
                field="form_wc"
                step={0.01}
                dp={2}
                chip={<PropLockToggle field="form_wc" />}
              />
              <NumberField
                label="Gas-Oil Ratio (scf/bbl)"
                field="form_gor"
                step={25}
                chip={<PropLockToggle field="form_gor" />}
              />
              <NumberField label="Formation Temperature (deg F)" field="form_temp" />
              <CheckboxField
                label="Model as 100% water (dewatering)"
                field="model_as_water"
                hint="Backup mode for a watered-out / source well: water cut is forced to 100% and qwf is treated as the water rate"
              />
            </div>
          </details>
        </section>

        <section className="pb-3">
          <details>
            <summary className="cursor-pointer select-none pt-3 text-[11px] font-semibold uppercase tracking-wide text-slate-400 hover:text-slate-600">
              Geometry
            </summary>
            <div className="mt-2 space-y-2">
              <NumberField
                label="Jetpump TVD (feet)"
                field="jpump_tvd"
                step={10}
                locked={asBuiltLocks.jpump_tvd}
              />
              <NumberField
                label="Power Fluid Density (lbm/ft3)"
                field="rho_pf"
                step={0.1}
                dp={1}
              />
              <NumberField
                label="Tubing Outer Diameter (inches)"
                field="tubing_od"
                step={0.1}
                dp={3}
                locked={asBuiltLocks.tubing}
              />
              <NumberField
                label="Tubing Wall Thickness (inches)"
                field="tubing_thickness"
                step={0.1}
                dp={3}
                locked={asBuiltLocks.tubing}
              />
              <NumberField
                label="Casing Outer Diameter (inches)"
                field="casing_od"
                step={0.125}
                dp={3}
                locked={asBuiltLocks.casing}
              />
              <NumberField
                label="Casing Wall Thickness (inches)"
                field="casing_thickness"
                step={0.1}
                dp={3}
                locked={asBuiltLocks.casing}
              />
            </div>
          </details>
        </section>

        <section className="pb-3">
          <details>
            <summary className="cursor-pointer select-none pt-3 text-[11px] font-semibold uppercase tracking-wide text-slate-400 hover:text-slate-600">
              Advanced
            </summary>
            <div className="mt-2 space-y-3">
              <div className="space-y-2">
                <SubHeader text="Loss Coefficients" />
                <NumberField
                  label="Nozzle Loss Coefficient (ken)"
                  field="ken"
                  step={0.005}
                  dp={3}
                />
                <NumberField
                  label="Throat Loss Coefficient (kth)"
                  field="kth"
                  step={0.05}
                  dp={2}
                />
                <NumberField
                  label="Diffuser Loss Coefficient (kdi)"
                  field="kdi"
                  step={0.05}
                  dp={2}
                />
              </div>

              <div className="space-y-2">
                <SubHeader text="PVT Overrides" />
                <NumberField label="Oil API" field="oil_api" step={0.1} dp={1} />
                <NumberField label="Bubble Point (psig)" field="bubble_point" step={10} />
                <NumberField label="Gas Specific Gravity" field="gas_sg" step={0.01} dp={2} />
                <NumberField label="Water Specific Gravity" field="wat_sg" step={0.01} dp={2} />
              </div>

              {/* Batch Run and Power Fluid Range sweep selectors live on
                  their pages now - see BatchPage / PfRangePage. */}

            </div>
          </details>
        </section>
      </div>
    </aside>
  );
}
