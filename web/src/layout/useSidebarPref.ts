/**
 * Sidebar open/closed preference. Docked mode persists to localStorage;
 * below 1100px the sidebar becomes an overlay drawer that always starts
 * closed. '[' toggles from anywhere except text inputs.
 */

import { useCallback, useEffect, useState } from "react";

const STORAGE_KEY = "woffl.sidebar";
const OVERLAY_QUERY = "(max-width: 1099.98px)";

function readStored(): boolean {
  try {
    return window.localStorage.getItem(STORAGE_KEY) !== "closed";
  } catch {
    return true;
  }
}

function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || target.isContentEditable;
}

export interface SidebarPref {
  /** sidebar visible (docked panel or overlay drawer) */
  open: boolean;
  /** true below 1100px: sidebar renders as a drawer over the content */
  overlay: boolean;
  toggle: () => void;
  close: () => void;
}

export function useSidebarPref(enabled = true): SidebarPref {
  const [overlay, setOverlay] = useState(() => window.matchMedia(OVERLAY_QUERY).matches);
  const [open, setOpen] = useState(() => (window.matchMedia(OVERLAY_QUERY).matches ? false : readStored()));

  // Track the viewport: entering overlay mode closes the drawer, returning
  // to docked mode restores the persisted preference.
  useEffect(() => {
    const mq = window.matchMedia(OVERLAY_QUERY);
    const onChange = (e: MediaQueryListEvent) => {
      setOverlay(e.matches);
      setOpen(e.matches ? false : readStored());
    };
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, []);

  // Persist only the docked preference; drawer state is transient.
  useEffect(() => {
    if (overlay) return;
    try {
      window.localStorage.setItem(STORAGE_KEY, open ? "open" : "closed");
    } catch {
      // storage unavailable (private mode) - preference just won't stick
    }
  }, [open, overlay]);

  // '[' toggles, unless the user is typing or the route has no sidebar.
  useEffect(() => {
    if (!enabled) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "[" || e.ctrlKey || e.metaKey || e.altKey) return;
      if (isTypingTarget(e.target)) return;
      e.preventDefault();
      setOpen((v) => !v);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [enabled]);

  const toggle = useCallback(() => setOpen((v) => !v), []);
  const close = useCallback(() => setOpen(false), []);

  return { open, overlay, toggle, close };
}
