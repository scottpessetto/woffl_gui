"""server/services/response_history.py - suction-response panel assembly.

Synthetic daily frames only (no network): build_days is pure, and the
response_history assembly runs with every fetch seam monkeypatched.
"""

from __future__ import annotations

import pandas as pd
import pytest

from server.services import response_history as rh

_START = pd.Timestamp("2026-01-01")


def _rows(bhp_by_day, ppf_by_day, tubing=250.0):
    """Daily rows in the fleet-query shape (reverse circ: annulus is PF)."""
    return [
        {
            "sample_date": _START + pd.Timedelta(days=i),
            "tubing_prs": tubing,
            "inn_ann_prs": ppf,
            "btmhole_prs": bhp,
        }
        for i, (bhp, ppf) in enumerate(zip(bhp_by_day, ppf_by_day))
    ]


# ---------------------------------------------------------------------------
# build_days: filter chain
# ---------------------------------------------------------------------------


def test_glitch_bhp_days_dropped():
    """Dead/glitching gauge days (BHP <= 50) never reach the panel."""
    rows = _rows([320.0, 29.5, 50.0, 340.0], [3000.0] * 4)
    days = rh.build_days(rows)
    assert [d["bhp"] for d in days] == [320.0, 340.0]


def test_days_without_valid_ppf_dropped():
    """No reading >= 800 psi resolves to no PF header; > 5500 is not real."""
    rows = _rows([320.0, 330.0, 340.0], [3000.0, 500.0, 6000.0], tubing=250.0)
    days = rh.build_days(rows)
    assert len(days) == 1
    assert days[0]["ppf"] == 3000.0
    assert days[0]["date"] == "2026-01-01"


def test_empty_and_missing_columns_yield_no_days():
    assert rh.build_days([]) == []
    assert rh.build_days(pd.DataFrame()) == []
    assert rh.build_days([{"sample_date": _START}]) == []


# ---------------------------------------------------------------------------
# build_days: era split + buildup flag
# ---------------------------------------------------------------------------


def test_era_split_on_current_pump_date_set():
    rows = _rows([320.0] * 6, [3000.0] * 6)
    era = _START + pd.Timedelta(days=3)  # day 3 starts the current era
    days = rh.build_days(rows, era_start=era)
    assert [d["era"] for d in days] == ["prior"] * 3 + ["current"] * 3
    # day ON the Date Set belongs to the current era
    assert days[3]["date"] == era.date().isoformat()
    assert days[3]["era"] == "current"


def test_no_era_start_marks_all_days_current():
    rows = _rows([320.0] * 4, [3000.0] * 4)
    assert all(d["era"] == "current" for d in rh.build_days(rows))
    assert all(d["era"] == "current" for d in rh.build_days(rows, era_start=None))


def test_buildup_flag_when_bhp_at_or_above_res_pres():
    rows = _rows([320.0, 1550.0, 1600.0, 400.0], [3000.0] * 4)
    days = rh.build_days(rows, res_pres=1550.0)
    assert [d["buildup"] for d in days] == [False, True, True, False]


def test_buildup_false_everywhere_when_res_pres_unknown():
    rows = _rows([320.0, 1550.0], [3000.0] * 2)
    for resp in (None, 0.0, -5.0):
        assert [d["buildup"] for d in rh.build_days(rows, res_pres=resp)] == [
            False,
            False,
        ]


# ---------------------------------------------------------------------------
# response_history assembly (fetch seams monkeypatched - no network)
# ---------------------------------------------------------------------------


def _fleet_frame(per_well: dict[str, list[dict]]) -> pd.DataFrame:
    frames = []
    for well, rows in per_well.items():
        df = pd.DataFrame(rows)
        df["well"] = well
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture()
def assembly(monkeypatch):
    """Baseline seams: one well of flowing days, no tracker, no evidence."""
    fleet = _fleet_frame({"MPM-64": _rows([320.0] * 5 + [1600.0], [3000.0] * 6)})
    monkeypatch.setattr(rh.evidence, "_fleet_pressure_daily", lambda: fleet)
    monkeypatch.setattr(rh, "_era_start_and_pump", lambda well: (None, None))
    monkeypatch.setattr(rh, "_res_pres", lambda well: None)
    monkeypatch.setattr(rh, "_well_evidence", lambda well, res_pres: None)
    return monkeypatch


def test_assembly_contract_shape(assembly):
    out = rh.response_history("MPM-64")
    assert set(out) == {"days", "era_start", "pump", "evidence", "res_pres"}
    assert len(out["days"]) == 6
    assert set(out["days"][0]) == {"date", "ppf", "bhp", "era", "buildup"}


def test_assembly_jp_history_unavailable_era_null_all_current(assembly):
    """Fail-soft: no tracker -> era_start null and every day 'current'."""
    out = rh.response_history("MPM-64")
    assert out["era_start"] is None
    assert out["pump"] is None
    assert all(d["era"] == "current" for d in out["days"])


def test_assembly_evidence_absent_is_null_not_fatal(assembly):
    assert rh.response_history("MPM-64")["evidence"] is None


def test_assembly_era_pump_res_pres_flow_through(assembly, monkeypatch):
    era = _START + pd.Timedelta(days=2)
    monkeypatch.setattr(rh, "_era_start_and_pump", lambda well: (era, "14B"))
    monkeypatch.setattr(rh, "_res_pres", lambda well: 1550.0)
    monkeypatch.setattr(
        rh,
        "_well_evidence",
        lambda well, res_pres: {
            "floor": 300.0,
            "psu_ref": 330.0,
            "beta": 0.09,
            "beta_source": "well",
            "n_pairs": 7,
        },
    )
    out = rh.response_history("MPM-64")
    assert out["era_start"] == era.date().isoformat()
    assert out["pump"] == "14B"
    assert out["res_pres"] == 1550.0
    assert out["evidence"]["beta_source"] == "well"
    assert [d["era"] for d in out["days"]] == ["prior"] * 2 + ["current"] * 4
    # the 1600 psi day sits above res_pres -> buildup, still in the scatter
    assert [d["buildup"] for d in out["days"]] == [False] * 5 + [True]


def test_assembly_unknown_well_returns_empty_days(assembly):
    out = rh.response_history("MPM-99")
    assert out["days"] == []
    assert out["evidence"] is None


def test_era_start_and_pump_fail_soft_when_tracker_down(monkeypatch):
    from server.services import datasources

    monkeypatch.setattr(datasources, "jp_history_safe", lambda: (None, None))
    assert rh._era_start_and_pump("MPM-64") == (None, None)


def test_era_start_and_pump_reads_current_pump(monkeypatch):
    from server.services import datasources

    jp_hist = pd.DataFrame(
        [
            {
                "Well Name": "MPM-64",
                "Nozzle Number": 12,
                "Throat Ratio": "A",
                "Tubing Diameter": 4.5,
                "Date Set": pd.Timestamp("2025-06-01"),
            },
            {
                "Well Name": "MPM-64",
                "Nozzle Number": 14,
                "Throat Ratio": "B",
                "Tubing Diameter": 4.5,
                "Date Set": pd.Timestamp("2026-01-03"),
            },
        ]
    )
    monkeypatch.setattr(datasources, "jp_history_safe", lambda: (jp_hist, "excel"))
    era, pump = rh._era_start_and_pump("MPM-64")
    assert era == pd.Timestamp("2026-01-03")
    assert pump == "14B"


def test_response_model_validates_payload(assembly):
    """The service payload round-trips through the pydantic response model."""
    from server.schemas import ResponseHistoryResponse

    payload = rh.response_history("MPM-64")
    model = ResponseHistoryResponse.model_validate(payload)
    assert len(model.days) == 6
    assert model.era_start is None
