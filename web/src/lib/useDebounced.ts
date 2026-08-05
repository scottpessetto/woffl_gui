import { useEffect, useState } from "react";

/**
 * Debounce a value: emits `value` after it has been stable for `delayMs`.
 * Drives auto-solve so slider scrubbing does not spam the solver.
 */
export function useDebounced<T>(value: T, delayMs = 400): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), delayMs);
    return () => clearTimeout(timer);
  }, [value, delayMs]);
  return debounced;
}
