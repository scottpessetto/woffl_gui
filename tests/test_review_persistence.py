"""Review write-through into prop_hist (woffl/gui/review_persistence.py).

Kaelin's design, Scott's field selection (2026-07-30): saved reviews push
property rows — canonical characterization ids plus the five IPR ids Scott
self-added to prop_xref. No store restore here (wells re-open through the
pivots + the sidebar's saved-IPR seed); no pump ids, no workflow flags, no
tombstones. These pin the write discipline:

* changed-only against a baseline bulk-read from prop_hist;
* NULL is never pushed (a NULL resvr_press would blank the pivots);
* hypotheticals (no enthid) are skipped and counted;
* gate off → no writes, said plainly; failures → error string + retry.
"""

import pandas as pd
import pytest

from woffl.gui import review_persistence as rp
from woffl.gui.workflow_steps import well_review_store as wrs

# The live prop_xref as of 2026-07-30 — 18 originals + Scott's five.
XREF_NOW = {
    "casing_absruff", "casing_inn_dia", "casing_out_dia", "form_gas_sg",
    "form_oil_api", "form_wat_sg", "ipr_wt_uid", "jpfric_diffuser",
    "jpfric_entry", "jpfric_nozzle", "jpfric_throat", "jpump_md",
    "resvr_bubb", "resvr_press", "resvr_temp", "tubing_absruff",
    "tubing_inn_dia", "tubing_out_dia",
    "ipr_qwf_liq", "ipr_pwf", "form_wc", "form_gor", "surf_press",
}


class FakeSt:
    def __init__(self):
        self.session_state = {}


class FakeHist:
    def __init__(self, xref=XREF_NOW):
        self.xref = set(xref)
        self.rows = []  # (db_well_name, prop_id, value) — chronological
        self.pushes = []  # (gui_well, prop_id, value, user)
        self.fail_push = False

    def execute_query(self, sql):
        if "FROM mpu.wells.prop_xref" in sql:
            return pd.DataFrame({"prop_id": sorted(self.xref)})
        if "FROM mpu.wells.prop_hist" in sql:
            latest = {}
            for w, pid, v in self.rows:
                latest[(w, pid)] = v
            return pd.DataFrame(
                [
                    {"well_name": w, "prop_id": pid, "prop_value": v}
                    for (w, pid), v in latest.items()
                ],
                columns=["well_name", "prop_id", "prop_value"],
            )
        raise AssertionError(f"unexpected query: {sql[:80]}")

    def push_prop(self, well, prop_id, value, user):
        if self.fail_push:
            raise RuntimeError("warehouse unavailable")
        assert value is not None, "NULL must never be pushed from a review save"
        assert prop_id in self.xref, f"{prop_id} not whitelisted"
        self.pushes.append((well, prop_id, value, user))
        self.rows.append((well.replace("MP", "", 1), prop_id, value))
        return True


def _entry(name, **kw):
    e = wrs.hypothetical_entry(
        name=name, res_pres=1800, form_temp=160, form_wc=0.5, form_gor=250,
        oil_bopd=300, pwf=900, jpump_tvd=4200, field_model="Schrader",
        nozzle="12", throat="B",
    )
    e["is_hypothetical"] = False
    e.update(kw)
    return e


def _stored_rows_for(entry, db_well):
    """prop_hist latest rows matching this entry exactly (nothing changed)."""
    out = []
    for f in rp.FIELD_MAP:
        v = f.enc(entry.get(f.store_key))
        if v is not None:
            out.append((db_well, f.prop_id, v))
    return out


@pytest.fixture
def env(monkeypatch):
    import woffl.assembly.databricks_client as dbc
    import woffl.assembly.prop_hist_client as phc

    fake_st = FakeSt()
    hist = FakeHist()
    monkeypatch.setattr(rp, "st", fake_st)
    monkeypatch.setattr(rp, "_XREF_CACHE", {"ids": None, "at": 0.0})
    monkeypatch.setattr(dbc, "execute_query", hist.execute_query)
    monkeypatch.setattr(
        phc, "push_prop", lambda w, p, v, u: hist.push_prop(w, p, v, u)
    )
    monkeypatch.setattr(phc, "resolve_entry_user", lambda force_refresh=False: "scott")
    monkeypatch.setenv("ALLOW_DATABRICKS_WRITES", "true")
    from types import SimpleNamespace

    return SimpleNamespace(st=fake_st, hist=hist)


class TestFieldMap:
    def test_all_fields_active_against_the_live_xref(self, env):
        assert {f.prop_id for f in rp.active_fields()} == {
            f.prop_id for f in rp.FIELD_MAP
        }

    def test_the_dropped_ids_stayed_dropped(self):
        """Scott: no jp_nozzle / jp_throat_ratio / well_reviewed (nor the other
        workflow flags). Pump truth stays with the JP tracker."""
        ids = {f.prop_id for f in rp.FIELD_MAP}
        for gone in (
            "jp_nozzle", "jp_throat_ratio", "well_reviewed", "well_offline",
            "jpump_direction", "field_model_code", "ppf_surf",
        ):
            assert gone not in ids

    def test_store_qwf_is_total_liquid_passed_through(self):
        """The STORE's qwf is already total liquid — no oil conversion here
        (the sidebar's oil-based path converts in ipr_anchor.save_ipr_values)."""
        f = next(f for f in rp.FIELD_MAP if f.prop_id == "ipr_qwf_liq")
        assert f.store_key == "qwf"
        assert f.enc(600.0) == 600.0


class TestWriteThrough:
    def test_unchanged_review_writes_nothing(self, env):
        e = _entry("MPB-90")
        env.hist.rows += _stored_rows_for(e, "B-90")
        rp.sync_pad("B")  # baseline
        rp.wrs_store("B")["MPB-90"] = e
        out = rp.sync_pad("B")
        assert out["saved"] == 0 and env.hist.pushes == []

    def test_changed_ipr_values_write_exactly_those_rows(self, env):
        e = _entry("MPB-90")
        env.hist.rows += _stored_rows_for(e, "B-90")
        rp.sync_pad("B")
        e2 = _entry("MPB-90", res_pres=1650, form_wc=0.62)
        # store's qwf is derived from oil/WC at build time — rebuild changed it
        rp.wrs_store("B")["MPB-90"] = e2
        out = rp.sync_pad("B")
        pushed = {pid for (_, pid, _, _) in env.hist.pushes}
        assert "resvr_press" in pushed and "form_wc" in pushed
        assert out["saved"] == len(env.hist.pushes)
        vals = {pid: v for (_, pid, v, _) in env.hist.pushes}
        assert vals["resvr_press"] == 1650.0
        assert vals["form_wc"] == pytest.approx(0.62)

    def test_first_save_of_a_new_well_pushes_its_full_value_set(self, env):
        rp.sync_pad("B")
        rp.wrs_store("B")["MPB-90"] = _entry("MPB-90")
        out = rp.sync_pad("B")
        pushed = {pid for (_, pid, _, _) in env.hist.pushes}
        # the five IPR ids all land…
        assert {"ipr_qwf_liq", "ipr_pwf", "form_wc", "form_gor", "surf_press"} <= pushed
        # …and None-valued canon fields (oil_api etc. on a synthetic entry) don't
        assert "form_oil_api" not in pushed
        assert out["saved"] == len(env.hist.pushes)

    def test_hypotheticals_skipped_and_counted(self, env):
        rp.sync_pad("B")
        hypo = _entry("MPB-99")
        hypo["is_hypothetical"] = True
        rp.wrs_store("B")["MPB-99"] = hypo
        out = rp.sync_pad("B")
        assert out["skipped_hypotheticals"] == 1 and env.hist.pushes == []

    def test_pad_filter_scopes_the_baseline(self, env):
        e = _entry("MPB-90")
        env.hist.rows += _stored_rows_for(e, "B-90")
        rp.sync_pad("J")  # J's baseline must NOT include B's rows
        rp.wrs_store("J")["MPJ-29"] = _entry("MPJ-29")
        rp.sync_pad("J")
        assert all(w == "MPJ-29" for (w, _, _, _) in env.hist.pushes)


class TestGateAndFailure:
    def test_writes_off_reports_disabled(self, env, monkeypatch):
        rp.sync_pad("B")
        monkeypatch.delenv("ALLOW_DATABRICKS_WRITES", raising=False)
        rp.wrs_store("B")["MPB-90"] = _entry("MPB-90")
        out = rp.sync_pad("B")
        assert out["disabled"] is True and env.hist.pushes == []

    def test_push_failure_is_an_error_string_then_retries(self, env):
        rp.sync_pad("B")
        rp.wrs_store("B")["MPB-90"] = _entry("MPB-90")
        env.hist.fail_push = True
        out = rp.sync_pad("B")
        assert out["error"] and "warehouse" in out["error"]
        env.hist.fail_push = False
        out = rp.sync_pad("B")
        assert out["saved"] > 0

    def test_xref_failure_fails_soft(self, env, monkeypatch):
        import woffl.assembly.databricks_client as dbc

        rp._XREF_CACHE.update({"ids": None, "at": 0.0})
        monkeypatch.setattr(
            dbc, "execute_query",
            lambda sql: (_ for _ in ()).throw(RuntimeError("no warehouse")),
        )
        assert rp.available_prop_ids(force=True) == set()
        out = rp.sync_pad("B")
        assert out["error"] is None
