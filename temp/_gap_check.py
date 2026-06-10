"""Find the well from Scott's screenshot (current 13B set 2025-11-25,
9 installs) and print its install records with the pulled->set gaps."""

import pandas as pd

from woffl.assembly.databricks_client import fetch_jp_history

jp = fetch_jp_history()
jp = jp.dropna(subset=["Date Set"])

for well, g in jp.groupby("Well Name"):
    g = g.sort_values("Date Set").reset_index(drop=True)
    last = g.iloc[-1]
    if (
        len(g) == 9
        and pd.notna(last["Date Set"])
        and last["Date Set"].date() == pd.Timestamp("2025-11-25").date()
        and str(last["Nozzle Number"]) in ("13", "13.0")
        and str(last["Throat Ratio"]).strip().upper() == "B"
    ):
        print(f"=== {well} — {len(g)} installs ===")
        prev_pulled = None
        for _, r in g.iterrows():
            pump = f"{int(r['Nozzle Number'])}{str(r['Throat Ratio']).strip()}"
            ds = r["Date Set"].date()
            dp = r["Date Pulled"].date() if pd.notna(r["Date Pulled"]) else "in hole"
            gap = ""
            if prev_pulled is not None:
                days = (r["Date Set"] - prev_pulled).days
                if days > 0:
                    gap = f"   <-- {days} days with NO pump on record before this set"
            print(f"  {pump:>4}  set {ds}  pulled {dp}{gap}")
            if pd.notna(r["Date Pulled"]):
                prev_pulled = r["Date Pulled"]
        break
else:
    print("no exact match found — relaxing to 13B current pumps set 2025-11-25:")
    for well, g in jp.groupby("Well Name"):
        g = g.sort_values("Date Set")
        last = g.iloc[-1]
        if pd.notna(last["Date Set"]) and last["Date Set"].date() == pd.Timestamp(
            "2025-11-25"
        ).date():
            print(f"  candidate: {well} ({len(g)} installs, last {last['Nozzle Number']}{last['Throat Ratio']})")
