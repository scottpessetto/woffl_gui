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


class TestDerivedLiquidRate:
    """B-28, 2026-08-03: the stored IPR liquid rate (2135.29 bbl/d) read as a
    bug because nothing said what phases it covered. It is the well's TOTAL
    LIQUID; the oil and water splits it implies at the saved WC are what the
    view has to spell out."""

    def _rows(self, extra=()):
        return _df(
            [
                ("ipr_qwf_liq", "IPR Total Liquid Rate at Anchor", "bbl/d",
                 "reservoir", 2135.2941176470586, _ts("2026-08-03 21:25"),
                 "scott@hilcorp.com"),
                ("form_wc", "Formation Water Cut", "fraction", "reservoir",
                 0.83, _ts("2026-08-03 21:25"), "scott@hilcorp.com"),
                *extra,
            ]
        )

    def test_liquid_rate_shows_the_phase_split_it_implies(self):
        out = shape_history(self._rows())
        row = out["latest"][out["latest"]["prop_id"] == "ipr_qwf_liq"].iloc[0]
        assert row["derivation"] == "→ 363 BOPD oil + 1,772 BWPD water at WC 0.83"

    def test_water_cut_comes_from_the_same_save_not_the_newest_one(self):
        """A later WC-only save must not re-explain an older rate: the two
        were written together and only that pairing is meaningful."""
        out = shape_history(
            self._rows(
                extra=[
                    ("form_wc", "Formation Water Cut", "fraction", "reservoir",
                     0.50, _ts("2026-08-04 08:00"), "scott@hilcorp.com"),
                ]
            )
        )
        row = out["latest"][out["latest"]["prop_id"] == "ipr_qwf_liq"].iloc[0]
        assert "0.83" in row["derivation"] and "363" in row["derivation"]

    def test_older_rate_falls_back_to_the_wc_in_force_at_the_time(self):
        """Rows saved before WC rode along in the batch still get explained —
        by the WC that was current then, never by a later one."""
        out = shape_history(
            _df(
                [
                    ("ipr_qwf_liq", "IPR Total Liquid Rate at Anchor", "bbl/d",
                     "reservoir", 1200.0, _ts("2026-07-10"), "scott@hilcorp.com"),
                    ("form_wc", "Formation Water Cut", "fraction", "reservoir",
                     0.60, _ts("2026-07-01"), "scott@hilcorp.com"),
                    ("form_wc", "Formation Water Cut", "fraction", "reservoir",
                     0.90, _ts("2026-07-20"), "scott@hilcorp.com"),
                ]
            )
        )
        row = out["latest"][out["latest"]["prop_id"] == "ipr_qwf_liq"].iloc[0]
        assert row["derivation"] == "→ 480 BOPD oil + 720 BWPD water at WC 0.60"

    def test_no_water_cut_anywhere_makes_no_claim(self):
        out = shape_history(
            _df(
                [
                    ("ipr_qwf_liq", "IPR Total Liquid Rate at Anchor", "bbl/d",
                     "reservoir", 1200.0, _ts("2026-07-10"), "scott@hilcorp.com"),
                ]
            )
        )
        assert out["latest"].iloc[0]["derivation"] == ""

    def test_other_properties_carry_no_derivation(self):
        out = shape_history(_df(ROWS))
        assert (out["history"]["derivation"] == "").all()
