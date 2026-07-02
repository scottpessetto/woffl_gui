"""Unit tests for the pad review store (woffl/gui/workflow_steps/well_review_store.py).

The store is the bridge between the reviewed sidebar state (qwf = OIL) and the
optimizer's WellConfig (qwf = TOTAL LIQUID). These tests pin the round-trip
exactness and the ≥99%-WC guard that closed the S-03 false-"optimizer shut in"
bug (P0-2 in docs/code_review_2026-07-01.md).
"""

import io

import pandas as pd
import pytest

from woffl.gui.params import SimulationParams
from woffl.gui.workflow_steps import well_review_store as wrs


def _params(**overrides) -> SimulationParams:
    base = dict(
        selected_well="MPS-99",
        qwf=400,
        pwf=900,
        pres=1800,
        form_wc=0.50,
        form_gor=250,
        form_temp=160,
        jpump_tvd=4200,
    )
    base.update(overrides)
    return SimulationParams(**base)


def _snapshot(params, **kw):
    kw.setdefault("ipr_source", "vogel")
    return wrs.snapshot_from_params(params, **kw)


class TestOilLiquidRoundTrip:
    """WellConfig.qwf (liquid) must invert back to the reviewed oil rate exactly
    via the optimizer's oil = qwf * (1 - form_wc)."""

    @pytest.mark.parametrize(
        "wc,oil", [(0.0, 750), (0.50, 400), (0.90, 120), (0.95, 60), (0.99, 25)]
    )
    def test_round_trip_exact(self, wc, oil):
        entry = _snapshot(_params(form_wc=wc, qwf=oil))
        cfg = wrs.to_well_config(entry)
        assert cfg.qwf * (1.0 - cfg.form_wc) == pytest.approx(oil, rel=1e-9)

    def test_oil_rate_field_preserved(self):
        entry = _snapshot(_params(form_wc=0.90, qwf=120))
        assert entry[wrs.OIL_RATE_FIELD] == pytest.approx(120.0)
        assert entry["qwf"] == pytest.approx(1200.0)  # 120 / (1 - 0.9)

    def test_wc_099_boundary_not_distorted(self):
        # Pre-fix, the snapshot clamp (max(1-wc, 0.01)) and the optimizer's
        # un-clamped inverse diverged ABOVE 0.99 — at exactly 0.99 both are 0.01.
        entry = _snapshot(_params(form_wc=0.99, qwf=25))
        assert entry["qwf"] == pytest.approx(2500.0)


class TestHighWaterCutGuard:
    """WC > MAX_MODELABLE_WC must refuse loudly instead of silently zeroing."""

    @pytest.mark.parametrize("wc", [0.995, 0.999, 1.0])
    def test_snapshot_raises_when_online(self, wc):
        with pytest.raises(ValueError, match="water cut"):
            _snapshot(_params(form_wc=wc, qwf=300))

    def test_snapshot_allows_offline_dewatering(self):
        # Water-pump-mode review: sidebar qwf is the WATER rate; carried as the
        # liquid rate for the (never-optimized) offline entry.
        entry = _snapshot(
            _params(form_wc=1.0, qwf=2800, model_as_water=True), offline=True
        )
        assert entry["offline"] is True
        assert entry["qwf"] == pytest.approx(2800.0)

    def test_to_well_config_raises_for_online_high_wc(self):
        # Legacy/CSV entry that predates the save guard must not silently zero.
        entry = _snapshot(_params(form_wc=1.0, qwf=300), offline=True)
        entry["offline"] = False
        with pytest.raises(ValueError, match="water cut"):
            wrs.to_well_config(entry)

    def test_offline_high_wc_excluded_from_active(self):
        entry = _snapshot(_params(form_wc=1.0, qwf=300), offline=True)
        store = {"MPS-99": entry, "MPS-1": _snapshot(_params(selected_well="MPS-1"))}
        active = wrs.active_entries(store)
        assert "MPS-99" not in active and "MPS-1" in active
        # The active set converts cleanly.
        assert len(wrs.store_to_well_configs(active)) == 1


class TestSnapshotValidation:
    def test_bad_ipr_source_rejected(self):
        with pytest.raises(ValueError, match="ipr_source"):
            _snapshot(_params(), ipr_source="guess")

    def test_bounds_clamped_into_wellconfig_range(self):
        # Current behavior: slightly-out-of-range reviewed values are clamped so
        # WellConfig.__post_init__ can't trip. (Loud validation is P0-10.)
        entry = _snapshot(_params(pres=6000))
        cfg = wrs.to_well_config(entry)
        assert cfg.res_pres == pytest.approx(5000.0)


class TestValidateStore:
    """Loud CSV validation (P0-10): holes must be flagged, not silently
    clamped into plausible defaults."""

    def test_clean_store_has_no_issues(self):
        assert wrs.validate_store({"MPS-99": _snapshot(_params())}) == {}

    def test_csv_holes_flagged(self):
        e = _snapshot(_params())
        e.update(res_pres=0.0, qwf=0.0, pwf=0.0, tubing_od=0.0)
        joined = " ".join(wrs.validate_store({"MPS-99": e})["MPS-99"])
        for token in ("res_pres", "qwf", "pwf", "tubing_od"):
            assert token in joined

    def test_degenerate_ipr_flagged(self):
        e = _snapshot(_params())
        e["pwf"] = 2000.0  # >= res_pres 1800
        msgs = wrs.validate_store({"MPS-99": e})["MPS-99"]
        assert any("degenerate" in m for m in msgs)

    def test_online_high_wc_flagged(self):
        e = _snapshot(_params(form_wc=1.0, qwf=300, model_as_water=True), offline=True)
        e["offline"] = False  # legacy/CSV-edited entry
        msgs = wrs.validate_store({"MPS-99": e})["MPS-99"]
        assert any("offline" in m for m in msgs)

    def test_wc_edited_without_rederiving_qwf_flagged(self):
        e = _snapshot(_params(form_wc=0.5, qwf=400))  # liquid = 800
        e["form_wc"] = 0.9  # hand-edited WC; qwf left at 800 (should be 4000)
        msgs = wrs.validate_store({"MPS-99": e})["MPS-99"]
        assert any("inconsistent" in m for m in msgs)


class TestStoreSignature:
    """The run-freshness signature (P0-3): sensitive to physical edits and the
    active well set, insensitive to notes/labels and dict order."""

    def _two(self):
        e1 = _snapshot(_params(selected_well="MPS-1"))
        e2 = _snapshot(_params(selected_well="MPS-2", form_wc=0.8, qwf=100))
        return e1, e2

    def test_insensitive_to_order_and_labels(self):
        e1, e2 = self._two()
        s1 = wrs.store_signature({"MPS-1": e1, "MPS-2": e2})
        e1b = dict(e1, notes="edited note", review_nozzle="14", gauge_note="x")
        s2 = wrs.store_signature({"MPS-2": e2, "MPS-1": e1b})
        assert s1 == s2

    def test_sensitive_to_physical_edit(self):
        e1, e2 = self._two()
        s1 = wrs.store_signature({"MPS-1": e1, "MPS-2": e2})
        e1b = dict(e1, res_pres=e1["res_pres"] + 50)
        assert wrs.store_signature({"MPS-1": e1b, "MPS-2": e2}) != s1

    def test_sensitive_to_well_set(self):
        e1, e2 = self._two()
        s1 = wrs.store_signature({"MPS-1": e1, "MPS-2": e2})
        assert wrs.store_signature({"MPS-1": e1}) != s1

    def test_survives_csv_round_trip(self):
        e1, e2 = self._two()
        store = {"MPS-1": e1, "MPS-2": e2}
        csv_text = wrs.store_to_dataframe(store).to_csv(index=False)
        loaded = wrs.dataframe_to_store(pd.read_csv(io.StringIO(csv_text), dtype=str))
        assert wrs.store_signature(loaded) == wrs.store_signature(store)


class TestCsvRoundTrip:
    def _store(self):
        e1 = _snapshot(_params(selected_well="MPS-1", form_wc=0.85, qwf=200))
        e1["review_nozzle"], e1["review_throat"] = "10", "B"
        e1["notes"] = "note, with comma"
        e2 = _snapshot(
            _params(selected_well="MPS-2"), offline=True, bhp_source="assumed"
        )
        return {"MPS-1": e1, "MPS-2": e2}

    def _round_trip(self, store):
        csv_text = wrs.store_to_dataframe(store).to_csv(index=False)
        # dtype=str mirrors the uploader (keeps nozzle '10' from becoming '10.0').
        return wrs.dataframe_to_store(pd.read_csv(io.StringIO(csv_text), dtype=str))

    def test_fields_survive(self):
        loaded = self._round_trip(self._store())
        assert set(loaded) == {"MPS-1", "MPS-2"}
        e1 = loaded["MPS-1"]
        assert e1["form_wc"] == pytest.approx(0.85)
        assert e1[wrs.OIL_RATE_FIELD] == pytest.approx(200.0)
        assert e1["qwf"] == pytest.approx(200.0 / 0.15)
        assert e1["review_nozzle"] == "10" and e1["review_throat"] == "B"
        assert e1["notes"] == "note, with comma"
        assert e1["offline"] is False
        assert loaded["MPS-2"]["offline"] is True
        assert loaded["MPS-2"]["bhp_source"] == "assumed"

    def test_optional_floats_stay_none(self):
        loaded = self._round_trip(self._store())
        for fld in ("oil_api", "gas_sg", "wat_sg", "bubble_point", "knz_well"):
            assert loaded["MPS-1"][fld] is None

    def test_round_trip_still_converts(self):
        loaded = self._round_trip(self._store())
        cfg = wrs.to_well_config(loaded["MPS-1"])
        assert cfg.qwf * (1.0 - cfg.form_wc) == pytest.approx(200.0, rel=1e-6)

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("schrader", "Schrader"),
            ("KUPARUK", "Kuparuk"),
            ("Schrader", "Schrader"),
            ("bogus", "Schrader"),
            ("", "Schrader"),
        ],
    )
    def test_field_model_normalized(self, raw, expected):
        df = wrs.store_to_dataframe(self._store())
        df.loc[df["well_name"] == "MPS-1", "field_model"] = raw
        loaded = wrs.dataframe_to_store(df)
        assert loaded["MPS-1"]["field_model"] == expected

    def test_rows_without_well_name_skipped(self):
        df = wrs.store_to_dataframe(self._store())
        df.loc[df["well_name"] == "MPS-2", "well_name"] = ""
        assert set(wrs.dataframe_to_store(df)) == {"MPS-1"}
