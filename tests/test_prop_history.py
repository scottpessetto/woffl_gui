"""Save-history view shaping (woffl/gui/prop_history.py) — pure parts.

Scott: "view all the history of saves on a well." prop_hist is append-only, so
the trail exists; these pin the shaping the Well Database section renders —
current-row flagging, NULL tombstone display, pin rendering, ordering.
"""

import pandas as pd
import pytest

from woffl.gui.prop_history import shape_history


def _df(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "prop_id", "prop_name", "units", "category",
            "prop_value", "entry_datetime", "entry_user",
        ],
    )


def _ts(s):
    return pd.Timestamp(s, tz="UTC")


ROWS = [
    # resvr_press: bulk load, then two engineer saves — newest is current
    ("resvr_press", "Reservoir Pressure", "psig", "reservoir",
     1650.0, _ts("2026-07-30 10:00"), "scott@hilcorp.com"),
    ("resvr_press", "Reservoir Pressure", "psig", "reservoir",
     1720.0, _ts("2026-07-15 09:00"), "scott@hilcorp.com"),
    ("resvr_press", "Reservoir Pressure", "psig", "reservoir",
     1800.0, _ts("2026-04-16"), "ka9612"),
    # the pin: set, later cleared (NULL) — cleared row is current
    ("ipr_wt_uid", "Inflow Performance Well Test Unique ID", "unitless",
     "reservoir", -3587790.0, _ts("2026-07-21"), "scott@hilcorp.com"),
    ("ipr_wt_uid", "Inflow Performance Well Test Unique ID", "unitless",
     "reservoir", None, _ts("2026-07-29"), "scott@hilcorp.com"),
    # a calibration
    ("jpfric_entry", "Jet Pump Throat Entry Friction", "unitless",
     "mechanical", 0.12, _ts("2026-07-30 11:00"), "scott@hilcorp.com"),
]


class TestShapeHistory:
    def test_empty_and_none_give_none(self):
        assert shape_history(None) is None
        assert shape_history(_df([])) is None

    def test_current_flag_marks_exactly_the_newest_row_per_prop(self):
        out = shape_history(_df(ROWS))
        cur = out["history"][out["history"]["is_current"]]
        assert set(cur["prop_id"]) == {"resvr_press", "ipr_wt_uid", "jpfric_entry"}
        rp = cur[cur["prop_id"] == "resvr_press"].iloc[0]
        assert rp["prop_value"] == 1650.0  # the 07-30 save, not the bulk load

    def test_history_is_newest_first(self):
        out = shape_history(_df(ROWS))
        times = list(out["history"]["entry_datetime"])
        assert times == sorted(times, reverse=True)

    def test_cleared_pin_displays_as_cleared_and_is_current(self):
        out = shape_history(_df(ROWS))
        pin_cur = out["latest"][out["latest"]["prop_id"] == "ipr_wt_uid"].iloc[0]
        assert pin_cur["display_value"] == "(cleared)"
        # …and the superseded pin row renders its uid as an int
        pin_old = out["history"][
            (out["history"]["prop_id"] == "ipr_wt_uid")
            & ~out["history"]["is_current"]
        ].iloc[0]
        assert pin_old["display_value"] == "test uid -3587790"

    def test_latest_has_one_row_per_prop_ordered_by_category(self):
        out = shape_history(_df(ROWS))
        assert len(out["latest"]) == 3
        assert not out["latest"]["prop_id"].duplicated().any()
        cats = list(out["latest"]["category"])
        assert cats == sorted(cats)

    def test_summary_numbers(self):
        out = shape_history(_df(ROWS))
        assert out["n_edits"] == 6
        assert out["n_props"] == 3
        assert out["last_edit"] == _ts("2026-07-30 11:00")
        assert out["editors"] == ["ka9612", "scott@hilcorp.com"]
