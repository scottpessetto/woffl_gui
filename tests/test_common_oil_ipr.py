"""A common oil curve learns only training oil/BHP and never shifts each test."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from server import schemas
from server.main import app
from server.services import common_ipr


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    from woffl.assembly import databricks_client
    def forbidden(*args, **kwargs):
        pytest.fail("Common IPR tests cannot read or write Databricks")
    monkeypatch.setattr(databricks_client, "execute_query", forbidden)
    monkeypatch.setattr(databricks_client, "execute_write", forbidden)


def fixture():
    params = schemas.SimParams(qwf=900.25, pwf=550.75, pres=1700.5, form_wc=.713, form_gor=412.25)
    tracker = pd.DataFrame([
        {"Date Set": "2025-09-01", "Nozzle Number": "12", "Throat Ratio": "B"},
        {"Date Set": "2026-02-01", "Nozzle Number": "13", "Throat Ratio": "C"},
    ])
    rows = []
    for i, date in enumerate(pd.date_range("2026-01-10", periods=12, freq="7D")):
        bhp, wc = float(400+50*i), .5+.02*i
        oil = float(500*common_ipr.term(bhp, params.pres))
        rows.append(dict(wt_uid=str(100+i), WtDate=date.isoformat(), BHP=bhp, WtOilVol=oil,
                         WtTotalFluid=oil/(1-wc), form_wc=wc, fgor=300+i*25))
    request = schemas.CommonOilIprRequest(well="MPE-42", params=params, months=6)
    return request, tracker, pd.DataFrame(rows)


def run(req, tracker, frame):
    return common_ipr.fit_frame(req, tracker, frame, "2026-04-15T00:00:00Z")


def test_one_curve_recovers_oil_across_pumps_and_measured_compositions():
    req, tracker, frame = fixture()
    before = frame.copy(deep=True)
    result = run(req, tracker, frame)
    assert result.qmax_oil == pytest.approx(500)
    assert result.training["installations"] == 2
    assert result.training["dates"] == 9 and result.holdout["dates"] == 3
    assert result.holdout["candidate_mae"] < 1e-10
    assert set(result.seeds) == {"qwf"}
    candidate = schemas.SimParams(**{**req.params.model_dump(), **result.seeds})
    assert candidate.pwf == req.params.pwf and candidate.pres == req.params.pres
    assert candidate.form_wc == req.params.form_wc and candidate.form_gor == req.params.form_gor
    assert candidate.qwf * (1-candidate.form_wc) / common_ipr.term(candidate.pwf, candidate.pres) == pytest.approx(500)
    pd.testing.assert_frame_equal(frame, before)


def test_holdout_outcomes_and_manual_holdout_exclusions_never_change_fitted_curve():
    req, tracker, frame = fixture()
    original = run(req, tracker, frame)
    held_ids = [r["test_id"] for r in original.rows if r["phase"] == "holdout"]
    altered = frame.copy()
    held = altered.wt_uid.isin(held_ids)
    altered.loc[held, "WtOilVol"] *= .2
    req.exclude_tests = held_ids
    changed = run(req, tracker, altered)
    assert changed.split_date == original.split_date
    assert changed.qmax_oil == original.qmax_oil
    assert changed.holdout["tests"] == original.holdout["tests"]
    assert changed.holdout["candidate_mae"] > 100


def test_exclusions_follow_frozen_split_and_do_not_mutate_observations():
    req, tracker, frame = fixture()
    initial = run(req, tracker, frame)
    req.exclude_tests = ["101"]
    req.exclude_eras = [initial.eras[0]["id"]]
    excluded = run(req, tracker, frame)
    assert excluded.split_date == initial.split_date
    assert excluded.holdout == initial.holdout
    assert excluded.training["tests"] < initial.training["tests"]
    assert any(r["reason"] == "Excluded from training by user." for r in excluded.rows)


def test_bad_data_changeout_day_duplicate_ids_and_embargo_are_visible():
    req, tracker, frame = fixture()
    frame.loc[0, "WtDate"] = "2026-02-01"  # installation day
    frame.loc[1, "wt_uid"] = frame.loc[2, "wt_uid"]
    frame.loc[3, "BHP"] = req.params.pres
    frame.loc[4, "form_wc"] = np.nan
    original = run(req, tracker, frame)
    split = pd.Timestamp(original.split_date)
    frame.loc[5, "WtDate"] = (split-pd.Timedelta(days=1)).isoformat()
    changed = run(req, tracker, frame)
    reasons = " ".join(r["reason"] or "" for r in changed.rows)
    assert "Installation day" in reasons and "duplicate" in reasons
    assert "reservoir pressure" in reasons and "WC/GOR" in reasons
    assert any(r["phase"] == "embargo" and r["reason"] for r in changed.rows)


def test_invalid_and_future_dates_are_reported_and_cannot_change_split():
    req, tracker, frame = fixture()
    original = run(req, tracker, frame)
    additions = [dict(frame.iloc[0], wt_uid=str(400+i), WtDate=date)
                 for i, date in enumerate([None, "invalid", "2026-05-01"])]
    result = run(req, tracker, pd.concat([frame, pd.DataFrame(additions)], ignore_index=True))
    assert result.split_date == original.split_date
    assert result.qmax_oil == original.qmax_oil
    assert len(result.rows) == len(original.rows)
    assert any("2 tests with missing/invalid dates and 1 future-dated" in n for n in result.notes)


def test_robust_fit_limits_single_bad_training_rate_without_deleting_it():
    req, tracker, frame = fixture()
    frame.loc[2, "WtOilVol"] *= 2
    frame.loc[2, "WtTotalFluid"] *= 2
    result = run(req, tracker, frame)
    assert result.qmax_oil == pytest.approx(500, abs=5)
    bad = next(r for r in result.rows if r["test_id"] == "102")
    assert bad["reason"] is None and bad["oil"] == frame.loc[2, "WtOilVol"]
    assert result.training["tests"] == 9


def test_same_date_repetitions_have_equal_total_date_weight():
    req, tracker, frame = fixture()
    frame.loc[2, "WtOilVol"] *= 1.1
    result = run(req, tracker, frame)
    repeated = [dict(frame.iloc[2], wt_uid=str(200+i)) for i in range(12)]
    duplicated = run(req, tracker, pd.concat([frame, pd.DataFrame(repeated)], ignore_index=True))
    assert duplicated.qmax_oil == pytest.approx(result.qmax_oil, abs=1e-8)
    assert duplicated.training["dates"] == result.training["dates"]


def test_insufficient_training_returns_review_rows_without_apply_seeds():
    req, tracker, frame = fixture()
    result = run(req, tracker, frame.iloc[:3])
    assert result.qmax_oil is None and result.seeds is None
    assert len(result.rows) == 3
    assert any("three usable training dates" in note for note in result.notes)


def test_route_is_explicit_read_only_and_uses_server_data(monkeypatch):
    req, tracker, frame = fixture()
    tracker["Well Name"] = "MPE-42"
    frame["well"] = "MPE-42"
    monkeypatch.setattr(common_ipr.datasources, "jp_history", lambda: (tracker, "databricks"))
    monkeypatch.setattr(common_ipr.tests, "fetch_all_well_tests", lambda months: frame)
    # Fixed fixture date by intercepting only the pure boundary's clock argument.
    fit = common_ipr.fit_frame
    monkeypatch.setattr(common_ipr, "fit_frame", lambda r, t, f, now, source: fit(r, t, f, "2026-04-15T00:00:00Z", source))
    response = TestClient(app).post("/api/common-ipr-fit", json=req.model_dump())
    assert response.status_code == 200, response.text
    assert response.json()["seeds"].keys() == {"qwf"}
    assert response.json()["request"] == req.model_dump()
