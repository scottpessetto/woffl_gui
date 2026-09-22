/**
 * When the Cost of PF panel prices the whole pad on its own.
 *
 * The pad-wide (All wells) pricing runs without a click so the chart is
 * there when the S-/I-Pad tab opens. It must not re-run on every render or
 * tab revisit, must not start while another job for the panel is in flight
 * (the deployed tier runs one heavy job at a time, WOFFL_MAX_JOBS=1), and
 * must wait for the downtime log so shut-in wells are excluded. Single-well
 * sizing prices ~28 pumps and stays a deliberate button press.
 *
 * Keys are stableStringify(request) strings computed by the caller.
 */

export interface AutoRunState {
  /** The All wells box is ticked (pad-wide pricing, no well sized). */
  allWells: boolean;
  /** The downtime log has loaded or failed, so the offline set is final. */
  offlineSettled: boolean;
  /** The PF step (and I-Pad setpoint) are in range. */
  inputsValid: boolean;
  /** A job for this panel is starting, running, or its status is unknown. */
  busy: boolean;
  /** Key of the request the panel would send now. */
  requestKey: string;
  /** The same key after it has been stable for the debounce delay. */
  debouncedKey: string;
  /** Key stored with the panel's last job (null = none or unknown). */
  lastKey: string | null;
  /** Key of the last auto-run attempt, so a failed start is not retried in a loop. */
  attemptedKey: string | null;
}

export function shouldAutoRun(s: AutoRunState): boolean {
  return (
    s.allWells &&
    s.offlineSettled &&
    s.inputsValid &&
    !s.busy &&
    s.debouncedKey === s.requestKey &&
    s.requestKey !== s.lastKey &&
    s.requestKey !== s.attemptedKey
  );
}
