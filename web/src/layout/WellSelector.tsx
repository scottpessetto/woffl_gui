/**
 * Searchable well combobox - the instant replacement for the Streamlit
 * selectbox nonce dance. Typing filters, arrows move a roving highlight,
 * Enter selects, Escape / click-outside closes. Wells group by pad letter.
 */

import clsx from "clsx";
import { ChevronDown, Search } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import type { KeyboardEvent } from "react";

import { useWells } from "../api/hooks";
import type { WellListItem } from "../api/types";
import { Badge, InfoNote } from "../components/ui";
import { useParamsStore } from "../state/params";

const CUSTOM = "Custom";

function fieldHint(w: WellListItem): string | null {
  if (w.is_sch === true) return "Schrader";
  if (w.is_sch === false) return "Kuparuk";
  return null;
}

export function WellSelector() {
  const wells = useWells();
  const well = useParamsStore((s) => s.well);
  const context = useParamsStore((s) => s.context);
  const selectWell = useParamsStore((s) => s.selectWell);

  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [highlight, setHighlight] = useState(0);
  const rootRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const sorted = useMemo(
    () => (wells.data?.wells ?? []).slice().sort((a, b) => a.name.localeCompare(b.name)),
    [wells.data],
  );

  const q = query.trim().toLowerCase();
  const filtered = q ? sorted.filter((w) => w.name.toLowerCase().includes(q)) : sorted;
  const showCustom = !q || CUSTOM.toLowerCase().includes(q);

  // Flat keyboard-navigation order: Custom first, then wells in pad groups.
  const items: string[] = useMemo(
    () => [...(showCustom ? [CUSTOM] : []), ...filtered.map((w) => w.name)],
    [showCustom, filtered],
  );
  const indexOf = useMemo(() => new Map(items.map((name, i) => [name, i])), [items]);

  const groups = useMemo(() => {
    const byPad = new Map<string, WellListItem[]>();
    for (const w of filtered) {
      const pad = w.pad || "?";
      const bucket = byPad.get(pad);
      if (bucket) bucket.push(w);
      else byPad.set(pad, [w]);
    }
    return [...byPad.entries()];
  }, [filtered]);

  // Click outside closes.
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [open]);

  const openList = () => {
    setQuery("");
    setHighlight(0);
    setOpen(true);
  };

  const pick = (name: string) => {
    selectWell(name);
    setOpen(false);
    setQuery("");
    inputRef.current?.blur();
  };

  const onKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (!open && (e.key === "ArrowDown" || e.key === "Enter")) {
      openList();
      e.preventDefault();
      return;
    }
    if (!open) return;
    if (e.key === "ArrowDown") {
      setHighlight((h) => Math.min(items.length - 1, h + 1));
      e.preventDefault();
    } else if (e.key === "ArrowUp") {
      setHighlight((h) => Math.max(0, h - 1));
      e.preventDefault();
    } else if (e.key === "Enter") {
      const name = items[highlight];
      if (name !== undefined) pick(name);
      e.preventDefault();
    } else if (e.key === "Escape") {
      setOpen(false);
      inputRef.current?.blur();
      e.preventDefault();
    }
  };

  const row = (name: string, w: WellListItem | null) => {
    const i = indexOf.get(name) ?? -1;
    const active = i === highlight;
    const hint = w ? fieldHint(w) : null;
    return (
      <button
        key={name}
        type="button"
        ref={active ? (el) => el?.scrollIntoView({ block: "nearest" }) : undefined}
        onMouseEnter={() => setHighlight(i)}
        // mousedown, not click: fires before the input's blur
        onMouseDown={(e) => {
          e.preventDefault();
          pick(name);
        }}
        className={clsx(
          "flex w-full items-center gap-2 px-2.5 py-1.5 text-left text-sm",
          active ? "bg-blue-50 text-blue-900" : "text-slate-700",
          name === well && "font-semibold",
        )}
      >
        <span className="flex-1 truncate">{name}</span>
        {hint && <span className="text-[10px] text-slate-400">{hint}</span>}
        {w && <Badge>{w.pad || "?"}</Badge>}
      </button>
    );
  };

  return (
    <div ref={rootRef} className="relative">
      <div className="relative">
        <Search className="pointer-events-none absolute left-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-slate-400" />
        <input
          ref={inputRef}
          type="text"
          role="combobox"
          aria-expanded={open}
          aria-label="Select well"
          value={open ? query : well}
          placeholder={open ? well : "Select well"}
          onFocus={openList}
          onChange={(e) => {
            setQuery(e.target.value);
            setHighlight(0);
          }}
          onKeyDown={onKeyDown}
          className={clsx(
            "h-8 w-full rounded-md border border-slate-300 bg-white pl-7 pr-7 text-sm text-slate-800",
            "outline-none focus:border-blue-400 focus:ring-1 focus:ring-blue-200",
          )}
        />
        <ChevronDown
          className={clsx(
            "pointer-events-none absolute right-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-slate-400 transition-transform",
            open && "rotate-180",
          )}
        />
      </div>

      {open && (
        <div className="absolute inset-x-0 z-20 mt-1 max-h-72 overflow-y-auto rounded-md border border-slate-200 bg-white py-1 shadow-lg">
          {wells.isPending && <p className="px-2.5 py-1.5 text-xs text-slate-400">Loading wells...</p>}
          {wells.isError && <p className="px-2.5 py-1.5 text-xs text-red-600">Well list unavailable</p>}
          {showCustom && row(CUSTOM, null)}
          {groups.map(([pad, group]) => (
            <div key={pad}>
              <p className="px-2.5 pb-0.5 pt-2 text-[10px] font-semibold uppercase tracking-wide text-slate-400">
                Pad {pad}
              </p>
              {group.map((w) => row(w.name, w))}
            </div>
          ))}
          {!showCustom && filtered.length === 0 && (
            <p className="px-2.5 py-1.5 text-xs text-slate-400">No wells match "{query}"</p>
          )}
        </div>
      )}

      {context && context.well === well && (
        <InfoNote className="mt-2">
          <div className="space-y-0.5 text-xs">
            <p className="font-medium">Loaded: {context.well}</p>
            {context.ipr_info && <p>{context.ipr_info}</p>}
            {context.saved_ipr_info && <p>{context.saved_ipr_info}</p>}
          </div>
        </InfoNote>
      )}
    </div>
  );
}
