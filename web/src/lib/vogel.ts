/**
 * Vogel IPR math - exact mirror of the Python InFlow.vogel_qmax.
 * Mirrored client-side so the IPR curve redraws instantly as reservoir
 * pressure or the anchor changes, with zero server round trips.
 * Keep in lockstep with the Python: any change there must land here too.
 */

/** 1 - 0.2(pwf/pres) - 0.8(pwf/pres)^2 */
export function vogelFraction(pwf: number, pres: number): number {
  if (pres <= 0) return 0;
  const r = pwf / pres;
  return 1 - 0.2 * r - 0.8 * r * r;
}

/** Max rate at pwf=0 from one anchor point (qwf, pwf) on the curve. */
export function vogelQmax(qwf: number, pwf: number, pres: number): number | null {
  const frac = vogelFraction(pwf, pres);
  if (frac <= 0 || qwf <= 0) return null;
  return qwf / frac;
}

/** Rate at a given flowing BHP. */
export function vogelRate(pwf: number, qmax: number, pres: number): number {
  return qmax * vogelFraction(pwf, pres);
}

/** Inverse: flowing BHP that produces `rate` (null outside (0, qmax)). */
export function vogelPwfFromRate(rate: number, qmax: number, pres: number): number | null {
  if (!(rate > 0) || !(rate < qmax)) return null;
  const disc = 0.04 + 3.2 * (1 - rate / qmax);
  if (disc < 0) return null;
  return ((-0.2 + Math.sqrt(disc)) / 1.6) * pres;
}

export interface IprCurve {
  bhp: number[];
  fluid: number[];
}

/**
 * Curve from an anchor (qwf, pwf) at reservoir pressure `pres` -
 * mirror of ipr_analyzer.generate_ipr_curves: bhp = 0..pres step 10.
 */
export function iprCurveFromAnchor(
  qwf: number,
  pwf: number,
  pres: number,
  step = 10,
): IprCurve | null {
  const qmax = vogelQmax(qwf, pwf, pres);
  if (qmax === null || pres <= 0) return null;
  const bhp: number[] = [];
  const fluid: number[] = [];
  for (let p = 0; p < pres; p += step) {
    bhp.push(p);
    fluid.push(vogelRate(p, qmax, pres));
  }
  bhp.push(pres);
  fluid.push(0);
  return { bhp, fluid };
}
