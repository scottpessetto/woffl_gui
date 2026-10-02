"""Info-only well tests: fetched, de-duplicated, listed, never auto-picked.

Until 2026-10-02 the fleet query kept allocated tests only, so a well's tests
disappeared from the app for the weeks between monthly allocation passes
(MPE-48: last allocated 2026-09-10, eight newer info-only tests hidden).
"""

from unittest.mock import patch

import pandas as pd
import pytest

from woffl.assembly import well_test_client as wtc


def _raw(rows):
    """vw_well_test-shaped query result from (uid, date, allocated, oil, water) rows."""
    return pd.DataFrame(
        {
            "well_name": ["E-048"] * len(rows),
            "wt_uid": [r[0] for r in rows],
            "wt_date": pd.to_datetime([r[1] for r in rows], utc=True),
            "allocated": [r[2] for r in rows],
            "oil_rate": [r[3] for r in rows],
            "fwat_rate": [r[4] for r in rows],
            "fgas_rate": [60.0] * len(rows),
            "form_wc": [0.4] * len(rows),
            "fgor": [60.0] * len(rows),
            "lift_wat": [5000.0] * len(rows),
            "whp": [270.0] * len(rows),
            "bhp": [575.0] * len(rows),
        }
    )


class TestFleetQuery:
    def test_query_no_longer_filters_on_allocated(self):
        assert "allocated = True" not in wtc.WELL_TEST_QUERY
        assert "vwt.allocated" in wtc.WELL_TEST_QUERY

    @patch("woffl.assembly.well_test_client.execute_query")
    def test_info_only_tests_are_returned_and_flagged(self, mock_query):
        mock_query.return_value = _raw(
            [
                (-10, "2026-09-10 00:00", True, 1021.0, 1454.0),
                (-11, "2026-09-30 12:03", False, 1043.0, 732.0),
            ]
        )
        df, _ = wtc.fetch_milne_well_tests("2026-09-01", "2026-10-02", ["E-048"])
        assert list(df["wt_uid"]) == [-10.0, -11.0]
        assert list(df["allocated"]) == [True, False]
        assert list(wtc.allocated_only(df)["wt_uid"]) == [-10.0]

    @patch("woffl.assembly.well_test_client.execute_query")
    def test_allocated_copy_absorbs_its_info_only_twin(self, mock_query):
        # FDC allocates by copying the SCADA row to a new wt_uid at midnight.
        mock_query.return_value = _raw(
            [
                (-20, "2026-09-10 06:19", False, 1021.392, 1454.408),
                (-21, "2026-09-10 00:00", True, 1021.392, 1454.408),
            ]
        )
        df, _ = wtc.fetch_milne_well_tests("2026-09-01", "2026-10-02", ["E-048"])
        assert len(df) == 1
        row = df.iloc[0]
        assert row["wt_uid"] == -21.0 and bool(row["allocated"]) is True
        assert row["dup_wt_uids"] == (-20.0,)

    @patch("woffl.assembly.well_test_client.execute_query")
    def test_repeated_info_only_rows_fold_but_distinct_same_day_tests_stay(self, mock_query):
        mock_query.return_value = _raw(
            [
                (-30, "2026-08-23 05:31", False, 997.6, 173.1),
                (-31, "2026-08-23 09:00", False, 997.6, 173.1),  # SCADA re-send
                (-32, "2026-08-23 18:52", False, 976.2, 0.0),  # a different test
            ]
        )
        df, _ = wtc.fetch_milne_well_tests("2026-08-01", "2026-10-02", ["E-048"])
        assert sorted(df["wt_uid"]) == [-32.0, -31.0]
        kept = df[df["wt_uid"] == -31.0].iloc[0]
        assert kept["dup_wt_uids"] == (-30.0,)

    def test_allocated_only_passes_through_an_unflagged_frame(self):
        frame = pd.DataFrame({"well": ["MPE-48"], "BHP": [575.0]})
        assert wtc.allocated_only(frame) is frame


def _fleet():
    """Cached-fleet-shaped frame: four allocated tests and two newer info-only."""
    dates = pd.to_datetime(
        ["2026-06-02", "2026-07-02", "2026-08-17", "2026-09-10", "2026-09-24", "2026-09-30"]
    )
    fluid = [2221.0, 2216.0, 1446.0, 2476.0, 1560.0, 1775.0]
    return pd.DataFrame(
        {
            "well": ["MPE-48"] * 6,
            "wt_uid": [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
            "WtDate": dates,
            "allocated": [True, True, True, True, False, False],
            "WtOilVol": [1000.0] * 6,
            "WtWaterVol": [f - 1000.0 for f in fluid],
            "WtGasVol": [60.0] * 6,
            "WtTotalFluid": fluid,
            "form_wc": [0.4] * 6,
            "BHP": [623.0, 552.0, 574.0, 575.0, 581.0, 583.0],
            "fgor": [60.0] * 6,
            "lift_wat": [5000.0] * 6,
            "whp": [270.0] * 6,
            "dup_wt_uids": [None, None, None, (-40.0,), None, None],
        }
    )


@pytest.fixture
def fleet(monkeypatch):
    from server.cache import clear_all_caches
    from server.services import datasources
    from server.services import tests as tests_svc

    clear_all_caches()
    frame = _fleet()
    monkeypatch.setattr(tests_svc, "fetch_all_well_tests", lambda months: frame)
    monkeypatch.setattr(
        datasources,
        "well_chars_safe",
        lambda: (pd.DataFrame({"Well": ["MPE-48"], "is_sch": [True]}), "csv_fallback"),
    )
    yield frame
    clear_all_caches()


class TestServerSlices:
    def test_default_slice_is_allocated_only(self, fleet):
        from server.services import tests as tests_svc

        assert list(tests_svc.tests_for_well("MPE-48", 6)["wt_uid"]) == [-1.0, -2.0, -3.0, -4.0]
        every = tests_svc.tests_for_well("MPE-48", 6, include_info=True)
        assert len(every) == 6

    def test_cap_counts_only_the_tests_the_slice_returns(self, fleet):
        from server.services import tests as tests_svc

        assert sorted(tests_svc.tests_for_well("MPE-48", 6, 2)["wt_uid"]) == [-4.0, -3.0]
        assert sorted(tests_svc.tests_for_well("MPE-48", 6, 2, include_info=True)["wt_uid"]) == [-6.0, -5.0]

    def test_listing_carries_the_flag_newest_first(self, fleet):
        from server.services import tests as tests_svc

        rows = tests_svc.tests_json("MPE-48", 6, include_info=True)
        assert [r["date"] for r in rows[:2]] == ["2026-09-30", "2026-09-24"]
        assert [r["allocated"] for r in rows] == [False, False, True, True, True, True]
        # The automatic callers' default never sees an info-only test.
        assert all(r["allocated"] for r in tests_svc.tests_json("MPE-48", 6))


class TestFit:
    def test_recent_mode_never_anchors_on_an_info_only_test(self, fleet):
        from server import schemas
        from server.services import ipr as ipr_svc

        out = ipr_svc.fit(schemas.IprFitRequest(well="MPE-48", anchor_mode="recent"))
        assert out["coeffs"]["anchor_wt_uid"] == -4.0
        assert out["coeffs"]["num_tests"] == 4

    @pytest.mark.parametrize("mode", ["median", "median_liq"])
    def test_median_modes_use_allocated_tests_only(self, fleet, mode):
        from server import schemas
        from server.services import ipr as ipr_svc

        out = ipr_svc.fit(schemas.IprFitRequest(well="MPE-48", anchor_mode=mode))
        assert out["coeffs"]["anchor_wt_uid"] in (-1.0, -2.0, -3.0, -4.0)
        assert out["coeffs"]["num_tests"] == 4

    def test_the_toggle_puts_info_only_tests_in_the_fit(self, fleet):
        from server import schemas
        from server.services import ipr as ipr_svc

        off = ipr_svc.fit(schemas.IprFitRequest(well="MPE-48", anchor_mode="recent"))["coeffs"]
        on = ipr_svc.fit(
            schemas.IprFitRequest(well="MPE-48", anchor_mode="recent", include_info_only=True)
        )["coeffs"]
        # every test counts: the newest (info-only) is now the recent anchor
        assert (on["num_tests"], on["anchor_wt_uid"]) == (6, -6.0)
        assert (off["num_tests"], off["anchor_wt_uid"]) == (4, -4.0)

    def test_the_toggle_lets_info_only_tests_move_reservoir_pressure(self, fleet):
        from server import schemas
        from server.services import ipr as ipr_svc

        # Two info-only tests far up the curve: only the toggled fit sees them.
        fleet.loc[fleet["wt_uid"] == -5.0, ["BHP", "WtTotalFluid", "WtWaterVol"]] = [1100.0, 400.0, 0.0]
        fleet.loc[fleet["wt_uid"] == -6.0, ["BHP", "WtTotalFluid", "WtWaterVol"]] = [1250.0, 150.0, 0.0]

        def res_p(**kw):
            req = schemas.IprFitRequest(well="MPE-48", anchor_mode="specific", anchor_wt_uid=-4.0, **kw)
            return ipr_svc.fit(req)["coeffs"]["res_p"]

        assert res_p(include_info_only=True) != res_p()
        assert res_p(include_info_only=True) > 1250.0

    def test_excluded_info_only_tests_stay_out_of_a_toggled_fit(self, fleet):
        from server import schemas
        from server.services import ipr as ipr_svc

        out = ipr_svc.fit(
            schemas.IprFitRequest(
                well="MPE-48", anchor_mode="recent", include_info_only=True, exclude_wt_uids=[-6.0]
            )
        )["coeffs"]
        assert (out["num_tests"], out["anchor_wt_uid"]) == (5, -5.0)

    def test_a_picked_info_only_test_anchors_the_fit(self, fleet):
        from server import schemas
        from server.services import ipr as ipr_svc

        out = ipr_svc.fit(
            schemas.IprFitRequest(
                well="MPE-48", anchor_mode="specific", anchor_date="2026-09-30", anchor_wt_uid=-6.0
            )
        )
        coeffs = out["coeffs"]
        assert coeffs["anchor_wt_uid"] == -6.0
        assert (coeffs["qwf"], coeffs["pwf"]) == (1775.0, 583.0)
        # the allocated tests plus the one picked: the other info-only stays out
        assert coeffs["num_tests"] == 5

    def test_a_date_only_pick_reaches_an_info_only_day(self, fleet):
        from server import schemas
        from server.services import ipr as ipr_svc

        out = ipr_svc.fit(
            schemas.IprFitRequest(well="MPE-48", anchor_mode="specific", anchor_date="2026-09-24")
        )
        assert out["coeffs"]["anchor_wt_uid"] == -5.0

    def test_wt_uid_separates_two_tests_on_one_day(self):
        from woffl.gui import ipr_anchor

        frame = _fleet()
        frame.loc[frame["wt_uid"] == -5.0, "WtDate"] = pd.Timestamp("2026-09-30 03:00")
        frame.loc[frame["wt_uid"] == -6.0, "WtDate"] = pd.Timestamp("2026-09-30 12:00")
        row = ipr_anchor.compute_anchored_vogel(
            frame, anchor_mode="specific", anchor_date="2026-09-30", anchor_wt_uid=-5.0
        )
        assert row["anchor_wt_uid"] == -5.0 and row["pwf"] == 581.0
        # date alone still resolves, to the newest test of the day
        by_date = ipr_anchor.compute_anchored_vogel(
            frame, anchor_mode="specific", anchor_date="2026-09-30"
        )
        assert by_date["anchor_wt_uid"] == -6.0


class TestManualTestAnchor:
    """The engineer's own (LRS) test rides in the fit request and can anchor."""

    LRS = dict(date="2026-10-02", total_fluid=1784.6, water=745.4, bhp=597.0, whp=363.0, pf_press=3353.8)

    def _fit(self, **kw):
        from server import schemas
        from server.services import ipr as ipr_svc

        return ipr_svc.fit(schemas.IprFitRequest(well="MPE-48", **kw))

    def test_it_anchors_the_curve_and_seeds_the_sidebar(self, fleet):
        out = self._fit(anchor_mode="specific", manual_test=self.LRS, anchor_manual=True)
        coeffs, seeds = out["coeffs"], out["seeds"]
        assert coeffs["anchor_manual"] is True and coeffs["anchor_wt_uid"] is None
        assert (coeffs["qwf"], coeffs["pwf"], coeffs["anchor_date"]) == (1784.6, 597.0, "2026-10-02")
        # the four allocated tests plus this one; info-only tests stay out
        assert coeffs["num_tests"] == 5
        assert seeds["qwf"] == 1784.6 and seeds["pwf"] == 597.0
        assert seeds["form_wc"] == pytest.approx(0.418, abs=1e-3)
        assert seeds["surf_pres"] == 363.0 and seeds["ppf_surf"] == 3353.8
        assert coeffs["res_p"] > 597.0

    def test_a_test_with_no_gor_leaves_the_wells_gor_alone(self, fleet):
        out = self._fit(anchor_mode="specific", manual_test=self.LRS, anchor_manual=True)
        assert "form_gor" not in out["seeds"]
        measured = self._fit(anchor_mode="specific", manual_test={**self.LRS, "fgor": 310.0}, anchor_manual=True)
        assert measured["seeds"]["form_gor"] == 310.0

    def test_it_is_the_most_recent_test_and_counts_in_other_anchors_fits(self, fleet):
        recent = self._fit(anchor_mode="recent", manual_test=self.LRS)["coeffs"]
        assert recent["anchor_manual"] is True and recent["num_tests"] == 5
        other = self._fit(anchor_mode="specific", anchor_wt_uid=-4.0, manual_test=self.LRS)["coeffs"]
        assert (other["anchor_manual"], other["anchor_wt_uid"], other["num_tests"]) == (False, -4.0, 5)

    def test_the_marker_uid_never_leaves_the_server(self, fleet):
        from server.services import ipr as ipr_svc

        out = self._fit(anchor_mode="median", manual_test=self.LRS)
        assert out["coeffs"]["anchor_wt_uid"] != ipr_svc._MANUAL_TEST_UID
        assert abs(ipr_svc._MANUAL_TEST_UID) > 1e9  # outside the real wt_uid range

    def test_a_request_without_it_is_unchanged(self, fleet):
        assert self._fit(anchor_mode="recent")["coeffs"]["anchor_manual"] is False


class TestPin:
    def _pin(self, monkeypatch, value):
        from server.services import ipr as ipr_svc

        monkeypatch.setattr(
            ipr_svc,
            "_saved_ipr",
            lambda well: {"pin_value": value, "pin_user": "sc9864", "pin_at": pd.Timestamp("2026-10-01")},
        )
        return ipr_svc.pin("MPE-48")

    def test_a_pinned_info_only_test_is_applied(self, fleet, monkeypatch):
        out = self._pin(monkeypatch, -6.0)
        assert (out["status"], out["wt_uid"], out["date_token"]) == ("applied", -6.0, "2026-09-30")

    def test_a_pin_survives_fdc_allocating_the_test(self, fleet, monkeypatch):
        # -40 was the info-only row pinned; its allocated copy -4 absorbed it.
        out = self._pin(monkeypatch, -40.0)
        assert (out["status"], out["wt_uid"], out["date_token"]) == ("applied", -4.0, "2026-09-10")

    def test_an_unknown_pin_is_stale(self, fleet, monkeypatch):
        out = self._pin(monkeypatch, -999.0)
        assert (out["status"], out["wt_uid"]) == ("stale", -999.0)
