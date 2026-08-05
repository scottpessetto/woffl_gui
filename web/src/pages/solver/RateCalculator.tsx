/**
 * What-if rate calculator: pick a flowing BHP and read the Vogel rate off
 * the current IPR curve, split into liquid / oil / water at the sidebar
 * water cut. Pure client-side math (lib/vogel), zero server round trips.
 */

import { useState } from "react";

import { Card, InfoNote, Section } from "../../components/ui";
import { fmtNum } from "../../lib/format";
import { vogelRate } from "../../lib/vogel";

const INPUT_CLS =
  "h-8 w-full rounded-md border border-slate-300 bg-white px-2 text-sm tabular-nums " +
  "text-slate-800 outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-200";

export function RateCalculator({
  qmax,
  pres,
  formWc,
  defaultBhp,
}: {
  qmax: number | null;
  pres: number;
  formWc: number;
  defaultBhp: number | null;
}) {
  // null = untouched: track the default (test BHP / solve suction) until the
  // user types, then their text wins.
  const [raw, setRaw] = useState<string | null>(null);
  const bhp = raw === null ? Math.round(defaultBhp ?? 500) : Number(raw);
  const valid = Number.isFinite(bhp);

  if (qmax === null) {
    return (
      <Section title="Rate Calculator">
        <Card>
          <InfoNote>No IPR curve yet</InfoNote>
        </Card>
      </Section>
    );
  }

  const fluid = valid ? Math.max(0, vogelRate(bhp, qmax, pres)) : null;
  const oil = fluid !== null ? fluid * (1 - formWc) : null;
  const water = fluid !== null && oil !== null ? fluid - oil : null;

  return (
    <Section title="Rate Calculator">
      <Card className="space-y-3">
        <label className="block">
          <span className="text-xs font-medium text-slate-500">Flowing BHP (psi)</span>
          <input
            type="number"
            step={25}
            min={0}
            value={raw === null ? String(bhp) : raw}
            onChange={(e) => setRaw(e.target.value)}
            className={`${INPUT_CLS} mt-1`}
          />
        </label>
        <p className="text-sm font-medium tabular-nums text-slate-700">
          {fluid !== null
            ? `${fmtNum(fluid)} BLPD / ${fmtNum(oil)} BOPD / ${fmtNum(water)} BWPD`
            : "Enter a BHP to compute a rate"}
        </p>
      </Card>
    </Section>
  );
}
