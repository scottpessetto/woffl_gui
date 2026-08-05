"""Saved IPR values — the single-well half of prop_hist Phase 2 (ipr_anchor).

Scott's goal sentence: opening a well keeps "the curve and rate the engineer
saw fit". These pin the Solver-side save (the MEASURED liquid rate stored
verbatim, the 0.99 WC cap, None-skipping) and the open-side load (requires
qwf AND pwf, memoized, fail-soft) plus the precedence rule against the anchor
pin — latest timestamp wins, ties to the VALUES (one Save writes both, and
the values were seeded from that anchor anyway).
"""

from datetime import datetime, timezone

import pandas as pd
import pytest

from woffl.gui import ipr_anchor as ia


@pytest.fixture(autouse=True)
def _clear_cache():
    ia.clear_saved_ipr_cache()
    yield
    ia.clear_saved_ipr_cache()


def _ts(s):
    return pd.Timestamp(s, tz="UTC")


# ── the precedence rule ─────────────────────────────────────────────────────


class TestSavedWins:
    def test_no_pin_means_values_win(self):
        assert ia.saved_wins(_ts("2026-07-30"), None) is True

    def test_newer_pin_wins(self):
        assert ia.saved_wins(_ts("2026-07-01"), _ts("2026-07-30")) is False

    def test_newer_values_win(self):
        assert ia.saved_wins(_ts("2026-07-30"), _ts("2026-07-01")) is True

    def test_tie_goes_to_the_values(self):
        """One Save click writes both — the values ARE the anchor's seed."""
        t = _ts("2026-07-30 12:00:00")
        assert ia.saved_wins(t, t) is True

    def test_no_values_never_win(self):
        assert ia.saved_wins(None, None) is False


class TestDefaultSigShared:
    def test_solver_aliases_the_same_object(self):
        """The sidebar seeds saved values UNDER the Solver's default anchor
        sig (the manual-edit affordance) — the two must be the same constant
        or the Solver's sync would clobber restored values on first render."""
        from woffl.gui.tabs.jetpump_solver import _IPR_SIDEBAR_DEFAULT_SIG

        assert _IPR_SIDEBAR_DEFAULT_SIG is ia.IPR_SIDEBAR_DEFAULT_SIG


# ── save: the sidebar's measured liquid rate → prop_hist ────────────────────


class TestSaveIprValues:
    @pytest.fixture
    def pushes(self, monkeypatch):
        captured = []
        monkeypatch.setattr(
            ia,
            "push_prop",
            lambda w, p, v, u, entry_datetime=None: captured.append((w, p, v, u)),
        )
        monkeypatch.setattr(ia, "resolve_entry_user", lambda: "scott")
        # save_ipr_values consults the stored latest for the friction push
        # discipline — stub it, or the test would hit the REAL Databricks
        # client (whose load_dotenv() then poisons os.environ for every
        # later test: the ALLOW_DATABRICKS_WRITES leak).
        monkeypatch.setattr(ia, "load_saved_ipr", lambda w: None)
        return captured

    def test_liquid_rate_is_stored_verbatim(self, pushes):
        n, msg = ia.save_ipr_values(
            "MPB-28", qwf_liq=600.0, pwf=900.0, res_pres=1800.0,
            form_wc=0.5, form_gor=250.0, surf_pres=210.0,
        )
        vals = {p: v for (_, p, v, _) in pushes}
        assert vals["ipr_qwf_liq"] == pytest.approx(600.0)  # not 600 / (1 - 0.5)
        assert vals["ipr_pwf"] == 900.0
        assert vals["resvr_press"] == 1800.0  # the curve travels whole
        assert vals["surf_press"] == 210.0
        assert n == 6 and msg.startswith("💾")

    @pytest.mark.parametrize("wc", [0.0, 0.5, 0.83, 0.99, 1.0])
    def test_water_cut_never_rescales_the_saved_rate(self, pushes, wc):
        """The B-28 regression (Scott, 2026-08-03). The old code pushed
        ``qwf / (1 - wc)``, so B-28's 363 BOPD at its LOCKED 0.83 WC landed as
        2135.29 BLPD — a rate no well test could corroborate, and one that
        moved every time the WC was touched. The measured liquid rate is the
        engineer's anchor: it must reach prop_hist bit-for-bit at ANY WC."""
        ia.save_ipr_values(
            "MPB-28", qwf_liq=2135.29, pwf=900.0, res_pres=1800.0,
            form_wc=wc, form_gor=250.0,
        )
        vals = {p: v for (_, p, v, _) in pushes}
        assert vals["ipr_qwf_liq"] == pytest.approx(2135.29)

    def test_wc_caps_at_099_for_its_own_row(self, pushes):
        """The rate is untouched, but the stored WC still can't be 1.0 — the
        downstream oil = liq * (1 - wc) would make the well identically dead
        (the store's MAX_MODELABLE_WC precedent)."""
        ia.save_ipr_values(
            "MPB-28", qwf_liq=1000.0, pwf=900.0, res_pres=1800.0,
            form_wc=1.0, form_gor=250.0,
        )
        vals = {p: v for (_, p, v, _) in pushes}
        assert vals["form_wc"] == pytest.approx(0.99)
        assert vals["ipr_qwf_liq"] == pytest.approx(1000.0)

    def test_missing_surface_pressure_is_skipped_not_nulled(self, pushes):
        n, _ = ia.save_ipr_values(
            "MPB-28", qwf_liq=600.0, pwf=900.0, res_pres=1800.0,
            form_wc=0.5, form_gor=250.0, surf_pres=None,
        )
        assert n == 5
        assert not any(p == "surf_press" for (_, p, _, _) in pushes)

    def test_failure_returns_the_prefixed_message(self, monkeypatch):
        def boom(w, p, v, u, entry_datetime=None):

            raise RuntimeError("gate closed")

        monkeypatch.setattr(ia, "push_prop", boom)
        monkeypatch.setattr(ia, "resolve_entry_user", lambda: "scott")
        monkeypatch.setattr(ia, "load_saved_ipr", lambda w: None)
        n, msg = ia.save_ipr_values(
            "MPB-28", qwf_liq=600.0, pwf=900.0, res_pres=1800.0,
            form_wc=0.5, form_gor=250.0,
        )
        assert n == 0 and msg.startswith(ia.VALUES_SAVE_FAILURE_PREFIX)

    def test_save_clears_the_load_memo(self, pushes):
        ia._saved_ipr_cache["MPB-28"] = {"stale": True}
        ia.save_ipr_values(
            "MPB-28", qwf_liq=600.0, pwf=900.0, res_pres=1800.0,
            form_wc=0.5, form_gor=250.0,
        )
        assert "MPB-28" not in ia._saved_ipr_cache


# ── the save batch: one stamp for every prop, and the comment bound to it ───


class TestSaveBatchStampAndComment:
    """prop_hist has no batch id, so the ONLY thing tying one save's rows
    together — and the only key an engineer comment can hang off — is a shared
    ``entry_datetime``. These pin that, and the ordering that protects it."""

    @pytest.fixture
    def rig(self, monkeypatch):
        pushes, comments = [], []

        def _push(w, p, v, u, entry_datetime=None):
            pushes.append({"prop": p, "stamp": entry_datetime})

        def _comment(w, stamp, user, text, context="ipr_save"):
            comments.append(
                {"well": w, "stamp": stamp, "user": user, "text": text,
                 "context": context}
            )

        monkeypatch.setattr(ia, "push_prop", _push)
        monkeypatch.setattr(ia, "push_eng_comment", _comment)
        monkeypatch.setattr(ia, "resolve_entry_user", lambda: "scott")
        monkeypatch.setattr(ia, "load_saved_ipr", lambda w: None)
        return pushes, comments

    def _save(self, **kw):
        return ia.save_ipr_values(
            "MPB-28", qwf_liq=600.0, pwf=900.0, res_pres=1800.0,
            form_wc=0.5, form_gor=250.0, surf_pres=210.0, **kw
        )

    def test_every_prop_in_one_save_shares_one_stamp(self, rig):
        pushes, _ = rig
        self._save()
        stamps = {p["stamp"] for p in pushes}
        assert len(pushes) == 6
        # One distinct, non-None value — not six now() calls microseconds apart.
        assert len(stamps) == 1 and None not in stamps

    def test_comment_is_bound_to_that_same_stamp(self, rig):
        pushes, comments = rig
        self._save(comment="anchored on the 7/25 test, gauge looked clean")
        assert len(comments) == 1
        assert comments[0]["stamp"] == pushes[0]["stamp"]
        assert comments[0]["user"] == "scott"
        assert comments[0]["text"].startswith("anchored on the 7/25 test")

    def test_two_saves_get_two_distinct_stamps(self, rig):
        """Decision: the grain is the TIMESTAMP, not the date — so a well
        edited twice in one day keeps both comments, each on its own rows."""
        pushes, comments = rig
        self._save(comment="first pass")
        self._save(comment="revised after the build-up")
        assert len({c["stamp"] for c in comments}) == 2
        assert len({p["stamp"] for p in pushes}) == 2

    def test_no_comment_means_no_comment_row(self, rig):
        _, comments = rig
        self._save()
        self._save(comment="   ")
        assert comments == []

    def test_comment_skipped_when_nothing_was_pushed(self, monkeypatch):
        """Decision A: a note explaining changes that never landed is worse
        than no note, so an empty batch drops the comment and says so."""
        comments = []
        monkeypatch.setattr(ia, "push_prop", lambda *a, **k: None)
        monkeypatch.setattr(
            ia, "push_eng_comment", lambda *a, **k: comments.append(a)
        )
        monkeypatch.setattr(ia, "resolve_entry_user", lambda: "scott")
        monkeypatch.setattr(ia, "load_saved_ipr", lambda w: None)
        # Every value None/NaN => the payload loop pushes nothing.
        n, msg = ia.save_ipr_values(
            "MPB-28", qwf_liq=float("nan"), pwf=float("nan"),
            res_pres=float("nan"), form_wc=float("nan"), form_gor=float("nan"),
            comment="why did I click this",
        )
        assert n == 0 and comments == []
        assert "comment skipped" in msg

    def test_comment_failure_never_undoes_the_save(self, rig, monkeypatch):
        """Properties commit first; the comment is best-effort. A comment
        error must degrade to 'saved without a note', not lose the save."""
        pushes, _ = rig
        monkeypatch.setattr(
            ia, "push_eng_comment",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("gate closed")),
        )
        n, msg = self._save(comment="this note will not land")
        assert n == 6 and len(pushes) == 6
        assert msg.startswith("💾")
        assert "comment not saved" in msg



# ── load: latest values + the pin timestamp, one query, memoized ────────────


def _hist_df(rows):
    return pd.DataFrame(
        rows, columns=["prop_id", "prop_value", "entry_datetime", "entry_user"]
    )


class TestLoadSavedIpr:
    def _wire(self, monkeypatch, df):
        import woffl.assembly.databricks_client as dbc
        import woffl.assembly.prop_hist_client as phc

        monkeypatch.setattr(phc, "_resolve_enthid", lambda w: 12345)
        monkeypatch.setattr(dbc, "execute_query", lambda sql: df)

    def test_loads_values_timestamp_user_and_pin(self, monkeypatch):
        self._wire(
            monkeypatch,
            _hist_df(
                [
                    ("ipr_qwf_liq", 600.0, _ts("2026-07-30 10:00"), "scott"),
                    ("ipr_pwf", 900.0, _ts("2026-07-30 10:00"), "scott"),
                    ("form_wc", 0.5, _ts("2026-07-30 10:00"), "scott"),
                    ("form_gor", 250.0, _ts("2026-07-30 10:00"), "scott"),
                    ("surf_press", 210.0, _ts("2026-07-30 10:00"), "scott"),
                    ("resvr_press", 1800.0, _ts("2026-04-16"), "ka9612"),
                    ("ipr_wt_uid", -3587790.0, _ts("2026-07-21"), "scott"),
                ]
            ),
        )
        info = ia.load_saved_ipr("MPB-28")
        assert info["values"]["qwf_liq"] == 600.0
        assert info["values"]["res_pres"] == 1800.0
        assert info["saved_at"] == _ts("2026-07-30 10:00")
        assert info["saved_by"] == "scott"
        assert info["pin_at"] == _ts("2026-07-21")
        assert ia.saved_wins(info["saved_at"], info["pin_at"])

    def test_canonical_resvr_press_alone_is_not_a_saved_curve(self, monkeypatch):
        """Every well has resvr_press from the bulk load — that must not read
        as 'the engineer saved an IPR here'."""
        self._wire(
            monkeypatch,
            _hist_df([("resvr_press", 1800.0, _ts("2026-04-16"), "ka9612")]),
        )
        assert ia.load_saved_ipr("MPB-28") is None

    def test_canon_timestamp_does_not_date_the_saved_set(self, monkeypatch):
        """resvr_press written later (e.g. by a pad-review canon push) must
        not make stale IPR values look newer than a fresh pin."""
        self._wire(
            monkeypatch,
            _hist_df(
                [
                    ("ipr_qwf_liq", 600.0, _ts("2026-07-01"), "scott"),
                    ("ipr_pwf", 900.0, _ts("2026-07-01"), "scott"),
                    ("resvr_press", 1650.0, _ts("2026-07-30"), "scott"),
                    ("ipr_wt_uid", -3587790.0, _ts("2026-07-15"), "scott"),
                ]
            ),
        )
        info = ia.load_saved_ipr("MPB-28")
        assert info["saved_at"] == _ts("2026-07-01")
        assert not ia.saved_wins(info["saved_at"], info["pin_at"])  # pin wins

    def test_nulled_pin_row_does_not_count_as_a_pin(self, monkeypatch):
        self._wire(
            monkeypatch,
            _hist_df(
                [
                    ("ipr_qwf_liq", 600.0, _ts("2026-07-01"), "scott"),
                    ("ipr_pwf", 900.0, _ts("2026-07-01"), "scott"),
                    ("ipr_wt_uid", float("nan"), _ts("2026-07-15"), "scott"),
                ]
            ),
        )
        info = ia.load_saved_ipr("MPB-28")
        assert info["pin_at"] is None
        assert ia.saved_wins(info["saved_at"], info["pin_at"])

    def test_memoized_per_well(self, monkeypatch):
        calls = []

        import woffl.assembly.databricks_client as dbc
        import woffl.assembly.prop_hist_client as phc

        monkeypatch.setattr(phc, "_resolve_enthid", lambda w: 12345)

        def q(sql):
            calls.append(sql)
            return _hist_df(
                [
                    ("ipr_qwf_liq", 600.0, _ts("2026-07-01"), "scott"),
                    ("ipr_pwf", 900.0, _ts("2026-07-01"), "scott"),
                ]
            )

        monkeypatch.setattr(dbc, "execute_query", q)
        ia.load_saved_ipr("MPB-28")
        ia.load_saved_ipr("MPB-28")
        assert len(calls) == 1

    def test_query_failure_fails_soft_to_none(self, monkeypatch):
        import woffl.assembly.databricks_client as dbc
        import woffl.assembly.prop_hist_client as phc

        monkeypatch.setattr(phc, "_resolve_enthid", lambda w: 12345)
        monkeypatch.setattr(
            dbc, "execute_query",
            lambda sql: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        assert ia.load_saved_ipr("MPB-28") is None


# ── BHP-calibrated friction survives the save + reload (2026-07-30) ─────────
# Scott: "i have to rerun the bhp calibration flow on single well even after
# saving and loading." The calibrate-to-BHP flow writes ken/kth/kdi into the
# sidebar; 📌 now persists them (jpfric_* canon ids) and open re-seeds them —
# INDEPENDENT of the pin-vs-values precedence (a calibration is about the
# wellbore, not the IPR).


class TestFrictionSave:
    @pytest.fixture
    def pushes(self, monkeypatch):
        captured = []
        monkeypatch.setattr(
            ia,
            "push_prop",
            lambda w, p, v, u, entry_datetime=None: captured.append((w, p, v, u)),
        )
        monkeypatch.setattr(ia, "resolve_entry_user", lambda: "scott")
        return captured

    def _save(self, **fric):
        return ia.save_ipr_values(
            "MPB-28", qwf_liq=600.0, pwf=900.0, res_pres=1800.0,
            form_wc=0.5, form_gor=250.0, surf_pres=210.0, **fric,
        )

    def test_uncalibrated_defaults_are_never_materialized(self, pushes, monkeypatch):
        """Nothing stored + sidebar at 0.03/0.3/0.4 → no jpfric rows. A 📌
        click on an uncalibrated well must not turn 'never calibrated' (NULL →
        library default) into fake explicit data."""
        monkeypatch.setattr(ia, "load_saved_ipr", lambda w: None)
        n, msg = self._save(ken=0.03, kth=0.3, kdi=0.4)
        assert n == 6  # the six IPR/curve rows only
        assert not any(p.startswith("jpfric") for (_, p, _, _) in pushes)
        assert "friction" not in msg

    def test_calibrated_friction_pushes_and_says_so(self, pushes, monkeypatch):
        monkeypatch.setattr(ia, "load_saved_ipr", lambda w: None)
        n, msg = self._save(ken=0.12, kth=0.3, kdi=0.4)
        fric = {p: v for (_, p, v, _) in pushes if p.startswith("jpfric")}
        assert fric == {"jpfric_entry": 0.12}  # only the calibrated one
        assert "BHP-calibrated friction" in msg

    def test_unchanged_stored_friction_writes_no_history_noise(
        self, pushes, monkeypatch
    ):
        stored = {
            "values": {}, "friction": {"ken": 0.12, "kth": 0.35, "kdi": 0.4},
            "saved_at": None, "saved_by": "scott", "pin_at": None,
        }
        monkeypatch.setattr(ia, "load_saved_ipr", lambda w: stored)
        self._save(ken=0.12, kth=0.35, kdi=0.4)
        assert not any(p.startswith("jpfric") for (_, p, _, _) in pushes)

    def test_recalibration_pushes_only_the_changed_coefficient(
        self, pushes, monkeypatch
    ):
        stored = {
            "values": {}, "friction": {"ken": 0.12, "kth": 0.35, "kdi": 0.4},
            "saved_at": None, "saved_by": "scott", "pin_at": None,
        }
        monkeypatch.setattr(ia, "load_saved_ipr", lambda w: stored)
        self._save(ken=0.12, kth=0.42, kdi=0.4)
        fric = {p: v for (_, p, v, _) in pushes if p.startswith("jpfric")}
        assert fric == {"jpfric_throat": 0.42}


class TestFrictionLoad:
    def _wire(self, monkeypatch, df):
        import woffl.assembly.databricks_client as dbc
        import woffl.assembly.prop_hist_client as phc

        monkeypatch.setattr(phc, "_resolve_enthid", lambda w: 12345)
        monkeypatch.setattr(dbc, "execute_query", lambda sql: df)

    def test_friction_only_record_loads_without_a_curve(self, monkeypatch):
        """A calibration can exist with no saved IPR — it must still come back
        (values empty → the sidebar seeds friction and nothing else)."""
        self._wire(
            monkeypatch,
            _hist_df(
                [
                    ("jpfric_entry", 0.12, _ts("2026-07-30"), "scott"),
                    ("jpfric_throat", 0.35, _ts("2026-07-30"), "scott"),
                ]
            ),
        )
        info = ia.load_saved_ipr("MPB-28")
        assert info is not None
        assert info["values"] == {}
        assert info["friction"] == {"ken": 0.12, "kth": 0.35}
        assert info["saved_at"] is None  # no curve → no curve timestamp

    def test_curve_and_friction_load_together(self, monkeypatch):
        self._wire(
            monkeypatch,
            _hist_df(
                [
                    ("ipr_qwf_liq", 600.0, _ts("2026-07-30"), "scott"),
                    ("ipr_pwf", 900.0, _ts("2026-07-30"), "scott"),
                    ("jpfric_diffuser", 0.55, _ts("2026-07-30"), "scott"),
                ]
            ),
        )
        info = ia.load_saved_ipr("MPB-28")
        assert info["values"]["qwf_liq"] == 600.0
        assert info["friction"] == {"kdi": 0.55}

    def test_friction_survives_even_when_the_pin_is_newer(self, monkeypatch):
        """The pin outranking the VALUES must not discard the calibration —
        friction is outside that precedence entirely."""
        self._wire(
            monkeypatch,
            _hist_df(
                [
                    ("ipr_qwf_liq", 600.0, _ts("2026-07-01"), "scott"),
                    ("ipr_pwf", 900.0, _ts("2026-07-01"), "scott"),
                    ("jpfric_entry", 0.12, _ts("2026-07-01"), "scott"),
                    ("ipr_wt_uid", -3587790.0, _ts("2026-07-20"), "scott"),
                ]
            ),
        )
        info = ia.load_saved_ipr("MPB-28")
        assert not ia.saved_wins(info["saved_at"], info["pin_at"])  # pin wins the curve
        assert info["friction"] == {"ken": 0.12}  # calibration still there


# ── the WC lock (2026-07-31): "test WC is known-bad here — always use mine" ──
# form_wc_lock = 1 ⇒ the saved WC overrides every test-derived WC (well open,
# anchor changes) until cleared with NULL. OUTSIDE the pin-vs-values
# precedence, like friction — the lock is a standing decision, not a snapshot.


class TestWcLockLoad:
    def _wire(self, monkeypatch, df):
        import woffl.assembly.databricks_client as dbc
        import woffl.assembly.prop_hist_client as phc

        monkeypatch.setattr(phc, "_resolve_enthid", lambda w: 12345)
        monkeypatch.setattr(dbc, "execute_query", lambda sql: df)

    def test_locked_well_loads_lock_and_wc_without_a_curve(self, monkeypatch):
        """A lock can exist with no saved curve — it must still come back."""
        self._wire(
            monkeypatch,
            _hist_df(
                [
                    ("form_wc", 0.45, _ts("2026-07-31"), "scott"),
                    ("form_wc_lock", 1.0, _ts("2026-07-31"), "scott"),
                ]
            ),
        )
        info = ia.load_saved_ipr("MPB-28")
        assert info is not None
        assert info["wc_locked"] is True
        assert info["wc_value"] == pytest.approx(0.45)
        assert info["values"] == {}  # no curve — precedence never applies

    def test_nulled_lock_reads_unlocked(self, monkeypatch):
        self._wire(
            monkeypatch,
            _hist_df(
                [
                    ("ipr_qwf_liq", 600.0, _ts("2026-07-01"), "scott"),
                    ("ipr_pwf", 900.0, _ts("2026-07-01"), "scott"),
                    ("form_wc_lock", 1.0, _ts("2026-07-10"), "scott"),
                    ("form_wc_lock", None, _ts("2026-07-20"), "scott"),
                ]
            ),
        )
        info = ia.load_saved_ipr("MPB-28")
        assert info["wc_locked"] is False

    def test_lock_survives_a_newer_pin(self, monkeypatch):
        """The pin winning the CURVE must not unlock the WC."""
        self._wire(
            monkeypatch,
            _hist_df(
                [
                    ("ipr_qwf_liq", 600.0, _ts("2026-07-01"), "scott"),
                    ("ipr_pwf", 900.0, _ts("2026-07-01"), "scott"),
                    ("form_wc", 0.45, _ts("2026-07-01"), "scott"),
                    ("form_wc_lock", 1.0, _ts("2026-07-01"), "scott"),
                    ("ipr_wt_uid", -3587790.0, _ts("2026-07-20"), "scott"),
                ]
            ),
        )
        info = ia.load_saved_ipr("MPB-28")
        assert not ia.saved_wins(info["saved_at"], info["pin_at"])  # pin wins curve
        assert info["wc_locked"] is True  # lock stands regardless
        assert info["wc_value"] == pytest.approx(0.45)


class TestSetWcLock:
    @pytest.fixture
    def pushes(self, monkeypatch):
        captured = []
        monkeypatch.setattr(
            ia,
            "push_prop",
            lambda w, p, v, u, entry_datetime=None: captured.append((w, p, v, u)),
        )
        monkeypatch.setattr(ia, "resolve_entry_user", lambda: "scott")
        return captured

    def test_locking_pins_the_current_wc_in_the_same_click(self, pushes):
        ok, msg = ia.set_wc_lock("MPB-28", True, form_wc=0.45)
        assert ok and "🔒" in msg
        assert pushes == [
            ("MPB-28", "form_wc", 0.45, "scott"),
            ("MPB-28", "form_wc_lock", 1.0, "scott"),
        ]

    def test_unlocking_writes_the_unlocked_sentinel_only(self, pushes):
        ok, msg = ia.set_wc_lock("MPB-28", False)
        assert ok and "🔓" in msg
        assert pushes == [("MPB-28", "form_wc_lock", 0.0, "scott")]

    def test_lock_clears_the_load_memo(self, pushes):
        ia._saved_ipr_cache["MPB-28"] = {"stale": True}
        ia.set_wc_lock("MPB-28", True, form_wc=0.45)
        assert "MPB-28" not in ia._saved_ipr_cache

    def test_failure_reports_not_raises(self, monkeypatch):
        monkeypatch.setattr(
            ia,
            "push_prop",
            lambda w, p, v, u, entry_datetime=None: (_ for _ in ()).throw(
                RuntimeError("gate")
            ),
        )
        monkeypatch.setattr(ia, "resolve_entry_user", lambda: "scott")
        ok, msg = ia.set_wc_lock("MPB-28", True, form_wc=0.45)
        assert not ok and "Could not update the WC lock" in msg


class TestPropLockRegistry:
    """The generic lock registry behind WC/GOR/ResP (2026-07-31)."""

    @pytest.fixture
    def pushes(self, monkeypatch):
        captured = []
        monkeypatch.setattr(
            ia,
            "push_prop",
            lambda w, p, v, u, entry_datetime=None: captured.append((w, p, v, u)),
        )
        monkeypatch.setattr(ia, "resolve_entry_user", lambda: "scott")
        return captured

    def test_gor_lock_pushes_value_and_flag(self, pushes):
        ok, msg = ia.set_prop_lock("MPB-28", "form_gor", True, value=450.0)
        assert ok and "GOR locked" in msg
        assert pushes == [
            ("MPB-28", "form_gor", 450.0, "scott"),
            ("MPB-28", "form_gor_lock", 1.0, "scott"),
        ]

    def test_res_pres_lock_targets_the_canonical_id(self, pushes):
        """The ResP lock pins resvr_press — the same id the pivots serve."""
        ia.set_prop_lock("MPB-28", "res_pres", True, value=1650.0)
        assert pushes == [
            ("MPB-28", "resvr_press", 1650.0, "scott"),
            ("MPB-28", "resvr_press_lock", 1.0, "scott"),
        ]

    def test_unlock_is_the_unlocked_sentinel(self, pushes):
        ia.set_prop_lock("MPB-28", "form_gor", False)
        assert pushes == [("MPB-28", "form_gor_lock", 0.0, "scott")]

    def test_unknown_field_reports_not_raises(self, pushes):
        ok, msg = ia.set_prop_lock("MPB-28", "surf_pres", True, value=210.0)
        assert not ok and "not a lockable field" in msg
        assert pushes == []

    def test_wc_wrapper_still_routes_through_the_registry(self, pushes):
        ia.set_wc_lock("MPB-28", True, form_wc=0.45)
        assert pushes == [
            ("MPB-28", "form_wc", 0.45, "scott"),
            ("MPB-28", "form_wc_lock", 1.0, "scott"),
        ]

    def _wire(self, monkeypatch, df):
        import woffl.assembly.databricks_client as dbc
        import woffl.assembly.prop_hist_client as phc

        monkeypatch.setattr(phc, "_resolve_enthid", lambda w: 12345)
        monkeypatch.setattr(dbc, "execute_query", lambda sql: df)

    def test_load_returns_the_full_lock_map(self, monkeypatch):
        self._wire(
            monkeypatch,
            _hist_df(
                [
                    ("form_gor", 450.0, _ts("2026-07-31"), "scott"),
                    ("form_gor_lock", 1.0, _ts("2026-07-31"), "scott"),
                    ("resvr_press", 1650.0, _ts("2026-07-31"), "scott"),
                    ("resvr_press_lock", 1.0, _ts("2026-07-31"), "scott"),
                ]
            ),
        )
        info = ia.load_saved_ipr("MPB-28")
        assert info["locks"] == {
            "form_wc": False, "form_gor": True, "res_pres": True,
        }
        assert info["lock_values"]["form_gor"] == 450.0
        assert info["lock_values"]["res_pres"] == 1650.0
