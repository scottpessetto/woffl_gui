/**
 * App shell: fixed topbar, collapsible 300px sidebar (docked, or an overlay
 * drawer below 1100px), and a scrollable main region hosting the routed page.
 * Also owns the well-context wiring: when a well is selected the context
 * query resolves and seeds the params store exactly once per well.
 */

import clsx from "clsx";
import { X } from "lucide-react";
import { useEffect, useState } from "react";
import { Outlet, useLocation } from "react-router-dom";

import { useWellContext } from "../api/hooks";
import { WarnNote } from "../components/ui";
import { useParamsStore } from "../state/params";
import { Sidebar } from "./Sidebar";
import { Topbar } from "./Topbar";
import { useSidebarPref } from "./useSidebarPref";
import { ViewTabs } from "./ViewTabs";

const NO_SIDEBAR_PATHS = ["/well-database", "/well-sort", "/optimize", "/tools"];

export function Layout() {
  // Well Database and Well Sort are cross-well browsers: the single-well
  // parameter sidebar (well selector + SimParams) does not apply there.
  const { pathname } = useLocation();
  const hasSidebar = !NO_SIDEBAR_PATHS.some((p) => pathname.startsWith(p));
  const sidebar = useSidebarPref(hasSidebar);

  const well = useParamsStore((s) => s.well);
  const months = useParamsStore((s) => s.months);
  const cap = useParamsStore((s) => s.cap);
  const applyContext = useParamsStore((s) => s.applyContext);
  const seededFor = useParamsStore((s) => s.seededFor);

  const ctx = useWellContext(well, months, cap);
  const [warnDismissed, setWarnDismissed] = useState(false);

  // Seed the store once per (well, context fetch). applyContext itself
  // guards against stale responses for a superseded selection.
  useEffect(() => {
    if (ctx.data && ctx.data.well === well && seededFor !== well) {
      applyContext(ctx.data);
    }
  }, [ctx.data, well, seededFor, applyContext]);

  // A fresh failure re-surfaces the warning even if a prior one was dismissed.
  useEffect(() => {
    if (ctx.isError) setWarnDismissed(false);
  }, [ctx.isError]);

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-slate-100">
      <div className="relative z-40 shrink-0">
        <Topbar onToggleSidebar={sidebar.toggle} sidebarAvailable={hasSidebar} />
        {ctx.isFetching && (
          <div className="absolute inset-x-0 top-full h-[2px] animate-pulse bg-blue-500" />
        )}
      </div>

      <div className="flex min-h-0 flex-1">
        {hasSidebar && !sidebar.overlay && (
          <div
            className={clsx(
              // min-w-0 is load-bearing: as a flex item the container's
              // default min-width:auto would clamp it to the child's 300px
              // min-content width and width:0 would never collapse it.
              "min-w-0 shrink-0 overflow-hidden transition-[width] duration-200 ease-in-out",
              sidebar.open ? "w-[300px]" : "w-0",
            )}
          >
            <Sidebar />
          </div>
        )}

        {hasSidebar && sidebar.overlay && sidebar.open && (
          <>
            <div
              className="fixed inset-0 top-12 z-20 bg-slate-900/40"
              onClick={sidebar.close}
              aria-hidden="true"
            />
            <div className="fixed bottom-0 left-0 top-12 z-30 w-[300px] shadow-xl">
              <Sidebar />
            </div>
          </>
        )}

        <main className="min-w-0 flex-1 overflow-auto p-4 lg:p-6">
          <ViewTabs />
          <Outlet />
        </main>
      </div>

      {ctx.isError && !warnDismissed && (
        <div className="fixed right-4 top-16 z-50 max-w-sm">
          <WarnNote className="shadow-lg">
            <span className="flex items-start gap-2">
              <span>Well context unavailable - defaults in use</span>
              <button
                type="button"
                aria-label="Dismiss warning"
                onClick={() => setWarnDismissed(true)}
                className="shrink-0 rounded p-0.5 text-amber-700 hover:bg-amber-100"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            </span>
          </WarnNote>
        </div>
      )}
    </div>
  );
}
