/**
 * Scott's Tools - the secret menu, ported from the Streamlit easter egg.
 *
 * The old app hid a bare `st.text_input` at the bottom of the sidebar; typing
 * "scott" into it set `session_state["_scotts_tools"] = True` and a
 * "Scott's Tools" mode appeared in the nav (woffl/gui/app.py:217-224 at
 * 826d85c). There is no equivalent of a stray unlabeled text box in this UI,
 * so the unlock is the same WORD by the same act of typing it - just without
 * a box: a rolling keystroke buffer anywhere in the app.
 *
 * Differences that matter:
 *
 * - It PERSISTS (localStorage). The Streamlit flag died with the session, so
 *   the menu had to be re-unlocked on every reload. A browser remembers.
 * - Typing is ignored while an input, textarea or contenteditable has focus,
 *   so typing "scott" into the well filter cannot unlock anything by accident.
 * - It is re-lockable, from the Tools page - the Streamlit one had no way back.
 *
 * This is an obscurity gate on a read-only internal page set, not a security
 * boundary: the /tools/* API endpoints are reachable by anyone who knows the
 * URL, exactly as the Streamlit tabs were to anyone who knew the word. Do not
 * put anything behind it that actually needs authorization.
 */

import { useEffect, useSyncExternalStore } from "react";

const WORD = "scott";
const STORAGE_KEY = "woffl.scottsTools";

let unlocked = readInitial();
const listeners = new Set<() => void>();

function readInitial(): boolean {
  try {
    return localStorage.getItem(STORAGE_KEY) === "1";
  } catch {
    // Private mode / storage disabled - the menu just will not persist.
    return false;
  }
}

function emit() {
  for (const l of listeners) l();
}

function persist(value: boolean) {
  try {
    if (value) localStorage.setItem(STORAGE_KEY, "1");
    else localStorage.removeItem(STORAGE_KEY);
  } catch {
    /* non-persistent unlock is still a working unlock for this tab */
  }
}

export function setToolsUnlocked(value: boolean) {
  if (unlocked === value) return;
  unlocked = value;
  persist(value);
  emit();
}

function subscribe(cb: () => void) {
  listeners.add(cb);
  return () => listeners.delete(cb);
}

/** Reactive read of the unlock flag. */
export function useToolsUnlocked(): boolean {
  return useSyncExternalStore(
    subscribe,
    () => unlocked,
    () => false, // server snapshot: locked
  );
}

/** True when the event target is somewhere the user is legitimately typing. */
function isTyping(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  if (!el || !el.tagName) return false;
  const tag = el.tagName.toLowerCase();
  return tag === "input" || tag === "textarea" || tag === "select" || el.isContentEditable;
}

/**
 * Mount ONCE (App). Listens for the word typed anywhere outside a field.
 *
 * The buffer keeps only the last WORD.length characters, so "sscott" or
 * "let me see scott" unlock too - the point is the word, not a clean run.
 */
export function useSecretMenuListener() {
  useEffect(() => {
    let buffer = "";

    function onKeyDown(e: KeyboardEvent) {
      if (unlocked) return; // nothing left to detect
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (isTyping(e.target)) return;
      if (e.key.length !== 1) return; // ignore Shift, Enter, arrows, ...

      buffer = (buffer + e.key.toLowerCase()).slice(-WORD.length);
      if (buffer === WORD) {
        buffer = "";
        setToolsUnlocked(true);
      }
    }

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);
}
