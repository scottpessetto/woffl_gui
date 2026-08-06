/**
 * "What do these coefficients represent?" - port of
 * woffl/gui/explainers.py:render_kcoef_explainer, the authoritative
 * friction-coefficient explainer that sat under the Streamlit calibrate
 * button. One divergence, deliberate: the ken range reads [0.005, 0.40]
 * here because that IS fric_calibration.KEN_BOUNDS - the Streamlit text
 * still says 0.20 from before the bound was widened.
 */

const COEFS: { code: string; name: string; body: string }[] = [
  {
    code: "knz",
    name: "nozzle - held fixed at 0.01",
    body:
      "Loss as power fluid accelerates through the nozzle. Primarily affects " +
      "PF flow rate and nozzle exit velocity. 0.01 is good when measured PF " +
      "rates match the model.",
  },
  {
    code: "ken",
    name: "entrance / suction - calibrated, range [0.005, 0.40]",
    body:
      "Loss as formation fluid enters the throat from the suction side. " +
      "Higher ken means it's harder for produced fluid to flow into the " +
      "throat - the pump can't pull suction pressure as far down - higher " +
      "modeled BHP. Affects drawdown directly.",
  },
  {
    code: "kth",
    name: "throat / mixing - calibrated, range [0.05, 1.0]",
    body:
      "Loss during mixing of high-velocity power fluid with low-velocity " +
      "formation fluid in the throat (constant-area mixing chamber) - the " +
      "biggest dissipative section in a jet pump. Higher kth means worse " +
      "momentum transfer - less pressure built downstream - the pump needs " +
      "higher suction (BHP). Affects both BHP and PF rate.",
  },
  {
    code: "kdi",
    name: "diffuser - calibrated, range [0.05, 1.0]",
    body:
      "Loss as the mixed stream decelerates in the diverging diffuser, " +
      "converting kinetic energy back into static pressure. Higher kdi means " +
      "less pressure recovery - lower discharge pressure - more suction (BHP) " +
      "needed to lift fluid out. Primarily affects BHP.",
  },
];

const WHY_CHANGE: string[] = [
  "Wear / erosion - sand or solids enlarging or roughening the throat/diffuser surfaces",
  "Scale / deposits - restricting flow areas, increasing turbulence",
  "Manufacturing tolerances - actual nozzle/throat geometry differs slightly from catalog",
  "Fluid-property assumptions - viscosity, density, or two-phase effects not captured by single-phase loss correlations",
  "Geometry simplifications - the model is a 1D approximation; real flow has 3D structure",
];

const READING_HIGH: string[] = [
  "Rule out the cheap explanations first. The coefficients absorb ALL model error, so a wrong power-fluid pressure, IPR, or GOR inflates them too - that's why calibration is gated on a good PF-rate match.",
  "A rising trend beats a single value. Erosion is progressive, so a coefficient climbing across successive tests is the real wear signal, not one high number (Streamlit: Scott's Tools - JP Fric Trend).",
  "Classic nozzle wash-out is geometric, not a friction term. An eroded, enlarged nozzle passes too much power fluid - a PF-rate mismatch, not a high coefficient (JP Wash-Out catches those).",
  'Pinned at a bound (the "sits on its search bound" note on a result) usually means friction alone cannot close the BHP gap - suspect a sonic-pinned throat or a wrong IPR rather than just wear.',
];

export function KcoefExplainer() {
  return (
    <details className="group">
      <summary className="cursor-pointer select-none text-[11px] text-slate-400 hover:text-slate-600">
        What do these coefficients represent?
      </summary>
      <div className="mt-2 space-y-3 rounded-md bg-slate-50 p-3 text-xs leading-relaxed text-slate-600">
        <p>
          The friction coefficients are <b>dimensionless energy-loss factors</b> in the four
          pressure-drop stages of the jet pump. Each captures the fraction of dynamic head lost
          to friction/turbulence in its section - a higher value means a less efficient (more
          lossy) component.
        </p>

        <dl className="space-y-2">
          {COEFS.map((c) => (
            <div key={c.code}>
              <dt className="font-semibold text-slate-700">
                <code className="rounded bg-slate-200 px-1 py-px text-[11px]">{c.code}</code>{" "}
                {c.name}
              </dt>
              <dd className="mt-0.5">{c.body}</dd>
            </div>
          ))}
        </dl>

        <div>
          <p className="font-semibold text-slate-700">Why these values change in practice</p>
          <p className="mt-0.5">
            The defaults come from idealized Cunningham-style jet pump theory. Real pumps
            deviate because of:
          </p>
          <ul className="mt-1 list-disc space-y-0.5 pl-4">
            {WHY_CHANGE.map((w) => (
              <li key={w}>{w}</li>
            ))}
          </ul>
          <p className="mt-1">
            Calibrating per-well fits a one-number-per-component "wear / efficiency factor" so
            the model matches the measured BHP. The coefficients absorb whatever the pump
            physics + simplified model couldn't predict from spec-sheet geometry alone.
          </p>
        </div>

        <div>
          <p className="font-semibold text-slate-700">Reading a high or rising value</p>
          <p className="mt-0.5">
            A kth or kdi pushed toward the top of its range means the model needs a lot of
            throat/diffuser loss to match the measured BHP - which IS consistent with
            throat/diffuser wear: sand erosion concentrates exactly there. Treat a high value
            as a flag, not proof:
          </p>
          <ul className="mt-1 list-disc space-y-0.5 pl-4">
            {READING_HIGH.map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ul>
        </div>
      </div>
    </details>
  );
}
