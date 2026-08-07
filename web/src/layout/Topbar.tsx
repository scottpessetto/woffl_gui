/**
 * Fixed 48px header: sidebar toggle, wordmark + version, the two top-level
 * nav destinations, and environment badges (offline chars / read-only mode).
 */

import clsx from "clsx";
import { PanelLeft } from "lucide-react";
import { NavLink, useLocation } from "react-router-dom";

import { useMeta, useWells } from "../api/hooks";
import { Badge, Button } from "../components/ui";

/** Every route served by the Single Well workspace (shared sidebar params). */
const SINGLE_WELL_PATHS = [
  "/solver",
  "/batch",
  "/pf-range",
  "/pressure-profile",
  "/well-profile",
  "/equivalents",
  "/jp-history",
] as const;

const NAV_BASE = "rounded px-2.5 py-1 text-sm font-medium";
const NAV_ACTIVE = "bg-blue-50 text-blue-700";
const NAV_INACTIVE = "text-slate-600 hover:bg-slate-100";

export function Topbar({
  onToggleSidebar,
  sidebarAvailable = true,
}: {
  onToggleSidebar: () => void;
  /** false on routes without the parameter sidebar (Well Database). */
  sidebarAvailable?: boolean;
}) {
  const meta = useMeta();
  const wells = useWells();
  const location = useLocation();

  const singleWellActive = SINGLE_WELL_PATHS.some((p) => location.pathname.startsWith(p));

  return (
    <header className="flex h-12 items-center gap-3 border-b border-slate-200 bg-white px-3">
      {sidebarAvailable ? (
        <Button variant="ghost" size="sm" onClick={onToggleSidebar} title="Toggle sidebar ( [ )">
          <PanelLeft className="h-4 w-4" />
        </Button>
      ) : (
        <span className="w-8" aria-hidden="true" />
      )}

      <div className="flex items-baseline gap-1.5">
        <span className="font-semibold text-slate-800">WOFFL</span>
        {meta.data && <span className="text-xs text-slate-400">{meta.data.version}</span>}
      </div>

      <nav className="ml-2 flex items-center gap-1">
        <NavLink
          to="/solver"
          className={clsx(NAV_BASE, singleWellActive ? NAV_ACTIVE : NAV_INACTIVE)}
        >
          Single Well
        </NavLink>
        <NavLink
          to="/well-sort"
          className={({ isActive }) => clsx(NAV_BASE, isActive ? NAV_ACTIVE : NAV_INACTIVE)}
        >
          Well Sort
        </NavLink>
        <NavLink
          to="/well-database"
          className={({ isActive }) => clsx(NAV_BASE, isActive ? NAV_ACTIVE : NAV_INACTIVE)}
        >
          Well Database
        </NavLink>
        <NavLink
          to="/optimize"
          className={({ isActive }) => clsx(NAV_BASE, isActive ? NAV_ACTIVE : NAV_INACTIVE)}
        >
          Optimization
        </NavLink>
      </nav>

      <div className="ml-auto flex items-center gap-2">
        {wells.data?.source === "csv_fallback" && (
          <Badge tone="fair" title="Databricks unreachable - well characteristics from the bundled CSV fallback">
            OFFLINE chars
          </Badge>
        )}
        {/* Writes exist now (Solver save-as-default); the badge only shows
            when the ALLOW_DATABRICKS_WRITES gate is off, i.e. saves are
            unavailable in this environment. */}
        {meta.data?.writes_enabled !== true && (
          <Badge tone="neutral" title="ALLOW_DATABRICKS_WRITES is off - saving well defaults is disabled">
            read-only
          </Badge>
        )}
        <span className="text-xs text-slate-500">{meta.data?.user ?? "local"}</span>
      </div>
    </header>
  );
}
