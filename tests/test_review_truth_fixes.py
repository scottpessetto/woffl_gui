"""Regression guards for the 2026-09-01 review's truth-handling fixes.

Each test names the finding it pins (docs/code_review_2026-09-01.md).
"""

from __future__ import annotations

import pandas as pd
import pytest

from woffl.assembly import pump_report
from woffl.assembly.jp_history import (
    get_all_current_pumps,
    get_current_pump,
    get_pump_at_date,
    order_installs,
)


def _same_day_jpco(first_row_is_new_pump: bool) -> pd.DataFrame:
    """A same-day pull + set: 12B pulled on 2026-05-10, 14C set the same day.

    The two rows share ``Date Set``; only ``Date Pulled`` tells them apart.
    ``first_row_is_new_pump`` flips the input order so a stable sort alone
    cannot pass the test by luck.
    """
    rows = [
        # the pump that came OUT that day (set earlier, pulled 2026-05-10)
        {"Well Name": "MPB-28", "Date Set": "2026-05-10", "Date Pulled": "2026-05-10",
         "Nozzle Number": 12, "Throat Ratio": "B", "Tubing Diameter": 4.5},
        # the pump that went IN that day (still in hole)
        {"Well Name": "MPB-28", "Date Set": "2026-05-10", "Date Pulled": None,
         "Nozzle Number": 14, "Throat Ratio": "C", "Tubing Diameter": 4.5},
        # an older install for context
        {"Well Name": "MPB-28", "Date Set": "2025-11-01", "Date Pulled": "2026-05-10",
         "Nozzle Number": 12, "Throat Ratio": "B", "Tubing Diameter": 4.5},
    ]
    if first_row_is_new_pump:
        rows[0], rows[1] = rows[1], rows[0]
    df = pd.DataFrame(rows)
    df["Date Set"] = pd.to_datetime(df["Date Set"])
    df["Date Pulled"] = pd.to_datetime(df["Date Pulled"])
    return df


class TestSameDayInstallTieBreak:
    """DATA-2: same-day pull+set must resolve to the pump still in the hole."""

    @pytest.mark.parametrize("swap", [False, True])
    def test_current_pump_is_the_one_still_in_hole(self, swap):
        cur = get_current_pump(_same_day_jpco(swap), "MPB-28")
        assert cur is not None
        assert (cur["nozzle_no"], cur["throat_ratio"]) == ("14", "C")

    @pytest.mark.parametrize("swap", [False, True])
    def test_pump_at_date_on_changeout_day(self, swap):
        at = get_pump_at_date(_same_day_jpco(swap), "MPB-28", "2026-05-10")
        assert at is not None
        assert (at["nozzle_no"], at["throat_ratio"]) == ("14", "C")

    @pytest.mark.parametrize("swap", [False, True])
    def test_all_current_pumps_agrees(self, swap):
        allc = get_all_current_pumps(_same_day_jpco(swap))
        row = allc[allc["Well Name"] == "MPB-28"].iloc[0]
        assert (str(row["Nozzle Number"]), row["Throat Ratio"]) == ("14", "C")

    @pytest.mark.parametrize("swap", [False, True])
    def test_eras_put_the_new_pump_last(self, swap):
        eras = pump_report.build_pump_eras(_same_day_jpco(swap), end_date="2026-09-01")
        assert [e["pump"] for e in eras] == ["12B", "14C"]
        assert eras[-1]["active"] is True

    def test_order_is_deterministic_and_stable(self):
        df = _same_day_jpco(False)
        a = order_installs(df, ascending=False)["Nozzle Number"].tolist()
        b = order_installs(df.iloc[::-1], ascending=False)["Nozzle Number"].tolist()
        assert a == b == [14, 12, 12]
        asc = order_installs(df, ascending=True)["Nozzle Number"].tolist()
        assert asc == [12, 12, 14]


class TestOptimizerRefusesDewateringWells:
    """SRV-3: WC >= 0.99 is refused, never capped to a 1%-oil producer."""

    @pytest.mark.parametrize("wc", [0.99, 0.995, 1.0])
    def test_config_from_seeds_raises(self, wc):
        from server.services.optimizer_runs import _config_from_seeds

        with pytest.raises(ValueError, match="not modelable"):
            _config_from_seeds(
                "MPB-99", "B", {"jpump_tvd": 4000.0, "pres": 1500.0, "qwf": 800, "pwf": 600, "form_wc": wc}
            )

    def test_config_from_seeds_keeps_high_but_modelable_wc(self):
        from server.services.optimizer_runs import _config_from_seeds

        cfg = _config_from_seeds(
            "MPB-99", "B", {"jpump_tvd": 4000.0, "pres": 1500.0, "qwf": 800, "pwf": 600, "form_wc": 0.985}
        )
        assert cfg.form_wc == pytest.approx(0.985)


class TestEvidenceEventsAndEraFloor:
    """EVID-F1 / EVID-F2: a well-earned beta needs DISTINCT PF moves, and the
    measured floor belongs to the current pump era."""

    @staticmethod
    def _daily(ppf_by_day: list[float], bhp_by_day: list[float], start="2026-03-01"):
        days = pd.date_range(start, periods=len(ppf_by_day), freq="D")
        return pd.DataFrame(
            {
                "sample_date": days,
                "tubing_prs": [200.0] * len(days),
                "inn_ann_prs": ppf_by_day,  # reverse circ: annulus carries PF
                "btmhole_prs": bhp_by_day,
            }
        )

    def test_one_pf_step_is_one_event_not_a_well_earned_beta(self):
        from server.services import evidence

        # 6 flowing days at 2,900 psi PF then 6 at 3,100: ONE step, but the
        # 3-30 day pair window makes 30+ pairs across it.
        ppf = [2900.0] * 6 + [3100.0] * 6
        bhp = [700.0] * 6 + [680.0] * 6
        ev = evidence.well_evidence(self._daily(ppf, bhp))
        assert ev is not None
        assert ev["n_pairs"] >= evidence.MIN_PAIRS
        assert ev["n_events"] == 1
        assert ev["beta_source"] == "default"

    def test_two_pf_moves_earn_the_beta(self):
        from server.services import evidence

        ppf = [2900.0] * 5 + [3100.0] * 5 + [2900.0] * 5
        bhp = [700.0] * 5 + [680.0] * 5 + [700.0] * 5
        ev = evidence.well_evidence(self._daily(ppf, bhp))
        assert ev is not None
        assert ev["n_events"] == 2
        assert ev["beta_source"] == "well"
        assert ev["beta"] == pytest.approx(0.1, abs=1e-6)  # -(-20/200)
        assert ev["beta_raw"] == pytest.approx(0.1, abs=1e-6)

    def test_floor_is_current_era_only(self):
        from server.services import evidence

        # 20 days on the OLD pump reaching 500 psi, a JPCO, then 15 days on
        # the new pump never below 650.
        ppf = [3000.0] * 35
        bhp = [500.0] * 20 + [650.0] * 15
        daily = self._daily(ppf, bhp)
        jpco = daily["sample_date"].iloc[20]
        ev = evidence.well_evidence(daily, install_dates=[jpco])
        assert ev is not None
        assert ev["floor_source"] == "era"
        assert ev["floor"] == pytest.approx(650.0)

    def test_short_era_falls_back_and_is_flagged(self):
        from server.services import evidence

        ppf = [3000.0] * 25
        bhp = [500.0] * 20 + [650.0] * 5  # only 5 days on the new pump
        daily = self._daily(ppf, bhp)
        jpco = daily["sample_date"].iloc[20]
        ev = evidence.well_evidence(daily, install_dates=[jpco], min_test_bhp=480.0)
        assert ev is not None
        assert ev["floor_source"] == "prior_era"
        assert ev["floor"] == pytest.approx(480.0)

    def test_prior_era_floor_never_contradicts(self):
        from server.services.match_health import _verdict

        row = {
            "floor_violation": 200.0,
            "sonic": True,
            "floor_source": "prior_era",
            "beta_source": "default",
        }
        assert _verdict(row) == "ok"
        row["floor_source"] = "era"
        assert _verdict(row) == "contradicted"


class TestReservoirPressureProvenance:
    """SOLV-F4 / SOLV-F5: the reported R2 belongs to the returned curve, and a
    floor-fallback reservoir pressure is never presented as a fit."""

    @staticmethod
    def _tests(bhps, fluids, well="MPX-02"):
        n = len(bhps)
        return pd.DataFrame(
            {
                "well": [well] * n,
                "wt_uid": [float(i + 1) for i in range(n)],
                "WtDate": pd.date_range("2026-01-05", periods=n, freq="14D"),
                "WtTotalFluid": fluids,
                "WtWaterVol": [f * 0.8 for f in fluids],
                "BHP": bhps,
                "form_wc": [0.8] * n,
                "fgor": [300.0] * n,
                "whp": [200.0] * n,
            }
        )

    def test_floor_fallback_is_flagged_and_weak(self, monkeypatch):
        from server import schemas
        from server.services import datasources
        from server.services import ipr as ipr_svc

        # Max test BHP within 10 psi of the Schrader 1,800 cap: no RP room.
        df = self._tests([1795.0, 1700.0, 1600.0], [400.0, 900.0, 1300.0])
        monkeypatch.setattr(ipr_svc.tests, "tests_for_well", lambda well, months, cap: df)
        monkeypatch.setattr(datasources, "well_chars_safe", lambda: (pd.DataFrame({"Well": ["MPX-02"], "is_sch": [True]}), "csv_fallback"))
        out = ipr_svc.fit(schemas.IprFitRequest(well="MPX-02", anchor_mode="recent"))
        assert out["coeffs"]["rp_source"] == "floor_fallback"
        assert out["weak"] is True

    def test_kuparuk_cap_comes_from_the_wells_chars(self, monkeypatch):
        from server import schemas
        from server.services import datasources
        from server.services import ipr as ipr_svc

        # The same tests on a KUPARUK well have 1,200 psi of search room.
        df = self._tests([1795.0, 1700.0, 1600.0], [400.0, 900.0, 1300.0])
        monkeypatch.setattr(ipr_svc.tests, "tests_for_well", lambda well, months, cap: df)
        monkeypatch.setattr(datasources, "well_chars_safe", lambda: (pd.DataFrame({"Well": ["MPX-02"], "is_sch": [False]}), "csv_fallback"))
        out = ipr_svc.fit(schemas.IprFitRequest(well="MPX-02", anchor_mode="recent", field_model="Kuparuk"))
        assert out["coeffs"]["rp_source"] == "fit"
        assert out["coeffs"]["res_p"] > 1810

    def test_reported_r2_is_for_the_returned_recent_anchor(self):
        from woffl.assembly.ipr_analyzer import (
            _calculate_r_squared,
            compute_vogel_coefficients,
            estimate_reservoir_pressure,
        )

        df = self._tests([1100.0, 1000.0, 900.0, 800.0, 1050.0], [1500.0, 1700.0, 1900.0, 2050.0, 1550.0])
        merged = estimate_reservoir_pressure(df)
        coeffs = compute_vogel_coefficients(merged)
        row = coeffs.iloc[0]
        expected = _calculate_r_squared(
            df["BHP"].to_numpy(float), df["WtTotalFluid"].to_numpy(float),
            float(row["ResP"]), float(row["pwf"]), float(row["qwf"]),
        )
        assert row["R2"] == pytest.approx(round(expected, 3))
        assert "R2_median" in coeffs.columns
        assert row["RP_source"] == "fit"

    def test_anchored_fallback_is_clamped_to_the_cap(self):
        from woffl.gui.ipr_anchor import fit_rp_through_anchor, rp_is_floor_fallback

        bhp = [1795.0, 1700.0]
        fluid = [400.0, 900.0]
        rp = fit_rp_through_anchor(bhp, fluid, 1795.0, 400.0, 1800)
        assert rp == 1800  # not 1845
        assert rp_is_floor_fallback(bhp, fluid, 1795.0, 1800)
        assert not rp_is_floor_fallback(bhp, fluid, 1795.0, 3000)


class TestHeaderTrendEstimatorIsNotTruncated:
    """EVID-F13 / EVID-F14: the within-day slope ESTIMATE must not be the
    mean of only the days that happened to land in the [0.2, 1.5] band."""

    @staticmethod
    def _frame(day_slopes: list[float], span: float = 40.0) -> pd.DataFrame:
        rows = []
        for d, m in enumerate(day_slopes):
            base = pd.Timestamp("2026-05-01") + pd.Timedelta(days=d)
            for k in range(8):
                x = 200.0 + span * k / 7.0
                rows.append({"ts": base + pd.Timedelta(hours=k), "WHP": x, "BHP": 600.0 + m * (x - 200.0)})
        return pd.DataFrame(rows).set_index("ts")

    def test_estimate_is_median_over_all_fittable_days(self):
        from server.services.tools.header_trend import fit_within_day

        # true coupling ~0.1: most days read 0.05-0.15; two noisy days land in
        # the band at 0.3 and 1.0. The old estimator averaged ONLY the band.
        fit = fit_within_day(self._frame([0.05, 0.1, 0.12, 0.15, 0.3, 1.0]), y_name="BHP", x_name="WHP")
        assert fit is not None
        assert fit.n_days == 6
        assert fit.n_good_days == 2  # classification still uses the band
        assert fit.mean_slope == pytest.approx(0.135, abs=0.01)  # median of all six
        assert fit.mean_slope < 0.2  # the old floor is gone

    def test_flat_driver_days_carry_no_information(self):
        from server.services.tools.header_trend import fit_within_day

        fit = fit_within_day(self._frame([0.5, 0.5], span=5.0), y_name="BHP", x_name="WHP")
        assert fit is not None
        assert fit.n_days == 0  # 5 psi of WHP movement is below min_x_range


class TestAnchorSeedCarriesTestDayPf:
    """SRV-5: the IPR-anchor seed moves PF (and circulation direction) to the
    anchor test's own day, so /calibrate fits this test's BHP at this test's
    PF - not the most recent day's."""

    @staticmethod
    def _frame() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "well": ["MPX-01"] * 3,
                "wt_uid": [1.0, 2.0, 3.0],
                "WtDate": pd.to_datetime(["2026-06-01", "2026-06-15", "2026-07-01"]),
                "WtTotalFluid": [1500.0, 1400.0, 1300.0],
                "BHP": [1100.0, 1000.0, 900.0],
                "form_wc": [0.8, 0.8, 0.8],
                "fgor": [300.0, 300.0, 300.0],
                "whp": [200.0, 200.0, 200.0],
                # the middle test ran on a FORWARD-circ day at a different PF
                "pf_press": [3300.0, 2950.0, 3350.0],
                "pf_source": ["annulus", "tubing", "annulus"],
            }
        )

    def test_specific_anchor_seeds_its_own_pf_and_direction(self, monkeypatch):
        from server import schemas
        from server.services import ipr as ipr_svc

        monkeypatch.setattr(
            ipr_svc.tests, "tests_for_well", lambda well, months, cap: self._frame()
        )
        req = schemas.IprFitRequest(well="MPX-01", anchor_mode="specific", anchor_date="2026-06-15")
        out = ipr_svc.fit(req)
        assert out["seeds"]["ppf_surf"] == pytest.approx(2950.0)
        assert out["seeds"]["jpump_direction"] == "forward"

    def test_dead_gauge_pf_is_not_seeded(self, monkeypatch):
        from server import schemas
        from server.services import ipr as ipr_svc

        df = self._frame()
        df.loc[1, "pf_press"] = 0.0  # dead gauge on the anchor day
        monkeypatch.setattr(ipr_svc.tests, "tests_for_well", lambda well, months, cap: df)
        req = schemas.IprFitRequest(well="MPX-01", anchor_mode="specific", anchor_date="2026-06-15")
        out = ipr_svc.fit(req)
        assert "ppf_surf" not in out["seeds"]
        assert "jpump_direction" not in out["seeds"]
