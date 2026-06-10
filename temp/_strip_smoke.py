"""Smoke: combined chart+strip figure builds correctly with the shape-based
strip (real date coords) — checks axis typing, refs, and trace placement."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from woffl.gui.tabs.jp_history_tab import _create_history_chart

dates = pd.date_range("2024-01-01", "2026-06-01", freq="7D")
test_df = pd.DataFrame(
    {
        "WtDate": dates,
        "WtOilVol": 400 + 100 * np.sin(np.arange(len(dates)) / 9.0),
        "WtWaterVol": 1500 + 400 * np.cos(np.arange(len(dates)) / 7.0),
        "BHP": 900 + 150 * np.sin(np.arange(len(dates)) / 5.0),
    }
)
jp_changes = pd.DataFrame(
    {
        "Date Set": pd.to_datetime(["2024-01-05", "2024-11-01", "2026-01-15"]),
        "Date Pulled": pd.to_datetime(["2024-10-20", "2026-01-15", pd.NaT]),
        "Nozzle Number": [12, 13, 13],
        "Throat Ratio": ["B", "A", "B"],
    }
)
bhp_daily = pd.DataFrame(
    {"tag_date": dates, "bhp": 900 + 150 * np.sin(np.arange(len(dates)) / 5.0)}
)

base = _create_history_chart(
    "MPX-1", test_df, jp_changes, bhp_daily_df=bhp_daily, bhp_from_zero=True
)
n_jpco = len(base.layout.shapes)
assert n_jpco == 3

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, row_heights=[0.84, 0.16],
    vertical_spacing=0.06, specs=[[{"secondary_y": True}], [{}]],
)
for tr in base.data:
    fig.add_trace(tr, row=1, col=1, secondary_y=(getattr(tr, "yaxis", "y") == "y2"))
for shp in base.layout.shapes:
    fig.add_shape(shp)
for ann in base.layout.annotations:
    fig.add_annotation(ann)

today = pd.Timestamp("2026-06-10")
palette = px.colors.qualitative.Set2
mids, labels, custom = [], [], []
for i, (_, r) in enumerate(jp_changes.iterrows()):
    start = r["Date Set"]
    end = r["Date Pulled"] if pd.notna(r["Date Pulled"]) else today
    fig.add_shape(
        type="rect", x0=start, x1=end, y0=0.06, y1=0.94,
        xref="x2", yref="y3 domain",
        fillcolor=palette[i % len(palette)],
        line=dict(width=1, color="white"), layer="below",
    )
    mids.append(start + (end - start) / 2)
    labels.append(f"{int(r['Nozzle Number'])}{r['Throat Ratio']}")
    custom.append([labels[-1], str(start.date()), str(end.date()), (end - start).days])

fig.add_trace(
    go.Scatter(
        x=mids, y=[0.5] * len(mids), mode="markers+text",
        marker=dict(size=14, opacity=0), text=labels,
        textposition="middle center", customdata=custom, showlegend=False,
    ),
    row=2, col=1,
)
fig.update_xaxes(range=[dates[0], today], row=1, col=1, showticklabels=True)
fig.update_xaxes(range=[dates[0], today], row=2, col=1, visible=False)
fig.update_yaxes(visible=False, fixedrange=True, range=[0, 1], row=2, col=1)

js = fig.to_json()
# strip rect shapes carry REAL datetimes on x2 / y3-domain
rects = [s for s in fig.layout.shapes if s.type == "rect"]
assert len(rects) == 3
assert all(s.xref == "x2" and s.yref == "y3 domain" for s in rects)
# the label scatter sits on the strip row's axes with datetime x → date axis
strip_tr = fig.data[-1]
assert strip_tr.yaxis == "y3" and strip_tr.xaxis == "x2"
assert isinstance(strip_tr.x[0], pd.Timestamp)
# JPCO paper-lines still present alongside the rects
lines = [s for s in fig.layout.shapes if s.type == "line"]
assert len(lines) == n_jpco
print(
    f"shape-based strip OK: {len(rects)} rects (x2/y3-domain), {len(lines)} JPCO "
    f"lines, label scatter on x2 with datetime x, {len(js):,} bytes json"
)
