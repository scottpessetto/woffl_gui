/**
 * /scott - unlock Scott's Tools from the address bar.
 *
 * The second way in, alongside typing the word anywhere on the page
 * (lib/secretMenu.ts). This one exists because the keystroke listener
 * deliberately ignores anything typed while a form field has focus, so on a
 * page where the cursor is parked in the sidebar there is nothing to type
 * into - but the URL bar always works.
 *
 * This route is registered UNCONDITIONALLY, unlike /tools/*: it is the way
 * in, so it cannot be behind the thing it unlocks.
 *
 * The redirect waits for the store to actually flip rather than firing
 * alongside it. /tools only exists as a route once `toolsUnlocked` is true,
 * so navigating in the same tick would race the re-render and land on the
 * catch-all (-> /solver), which looks exactly like "the link is broken".
 */

import { useEffect } from "react";
import { Navigate } from "react-router-dom";

import { Spinner } from "../components/ui";
import { setToolsUnlocked, useToolsUnlocked } from "../lib/secretMenu";

export default function ScottUnlock() {
  const unlocked = useToolsUnlocked();

  useEffect(() => {
    // Idempotent: setToolsUnlocked early-returns when the value is unchanged,
    // so StrictMode's double-invoke in dev is a no-op.
    setToolsUnlocked(true);
  }, []);

  if (!unlocked) return <Spinner label="Unlocking" />;
  return <Navigate to="/tools" replace />;
}
