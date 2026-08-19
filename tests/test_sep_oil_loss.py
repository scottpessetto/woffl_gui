"""Separator Oil Loss: the three corrections that make the integral honest.

The tool charges oil leaving the first-stage water leg as
``flow x (base - wc)`` integrated in time, where ``base`` is the analyzer's
OWN trailing plateau. Three things it must keep getting right, and each has a
test here that fails loudly if someone "simplifies" it:

* the film correction (a coated Red Eye that tops out at 96% is not 2,800
  BOPD of oil forever),
* gating on FLOW and never on the water-cut value (a deep water-cut drop at
  good flow is REAL carry-under, confirmed by the analyzer sweeping
  continuously rather than railing - it is not a dropout to be filtered),
* the band: an upper bound capped at field oil rate and a lower bound capped
  at ``max_oil_frac`` of the leg, so an implausible number announces itself.

Everything runs offline. The only data seam is ``sep_oil_loss._raw``, which is
monkeypatched with synthetic historian rows - no Databricks, no network.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from server import schemas
from server.cache import clear_all_caches
from server.main import app
from server.services.tools import sep_oil_loss as svc
from woffl.assembly import historian_client

# Field-local midnight, expressed as the UTC the historian view reports.
_T0 = pd.Timestamp("2026-08-01 08:00:00", tz="UTC")

# Flow high enough to be a real separator leg, well over FLOW_MIN_BPD.
_FLOW = 70_000.0


@pytest.fixture(autouse=True)
def caches():
    """`_raw` is ttl_cache-wrapped; no payload may leak between tests."""
    clear_all_caches()
    yield
    clear_all_caches()


def _hist(
    segments: list[tuple[float, ...]],
    step_s: float = 10.0,
    wc_step_s: float = 30.0,
    level_step_s: float = 60.0,
) -> pd.DataFrame:
    """Synthetic historian rows from piecewise-constant segments.

    Mirrors the real view: one row per tag per reported value, each tag on its
    own cadence (10 s flow meter, slower analyzer, level and setpoint),
    timestamps tz-aware UTC. A ``None`` value emits no rows for that tag in
    that segment.

    Args:
        segments (list): ``(seconds, flow_bpd, wc_pct, level_pct)`` in order,
            optionally with a fifth ``level_sp_pct``. Omitting the setpoint
            parks it on the level, so the event reads as "at setpoint".
        step_s (float): Flow-meter sample interval (s).
        wc_step_s (float): Analyzer sample interval (s).
        level_step_s (float): Level and setpoint sample interval (s).

    Returns:
        df (DataFrame): ``tag`` (str), ``t`` (tz-aware UTC), ``value`` (float).
    """
    bounds: list[tuple[float, float, tuple[float | None, ...]]] = []
    total = 0.0
    for segment in segments:
        secs, flow, wc, level = segment[0], segment[1], segment[2], segment[3]
        level_sp = segment[4] if len(segment) > 4 else level
        bounds.append((total, total + secs, (flow, wc, level, level_sp)))
        total += secs

    def pick(offset: float) -> tuple[float | None, ...]:
        for lo, hi, values in bounds:
            if lo <= offset < hi:
                return values
        return bounds[-1][2]

    rows: list[tuple[str, float, float]] = []
    for tag, slot, stride in (
        (svc.FLOW_TAG, 0, step_s),
        (svc.WC_TAG, 1, wc_step_s),
        (svc.LEVEL_TAG, 2, level_step_s),
        (svc.LEVEL_SP_TAG, 3, level_step_s),
    ):
        offset = 0.0
        while offset < total:
            value = pick(offset)[slot]
            if value is not None:
                rows.append((tag, offset, float(value)))
            offset += stride

    frame = pd.DataFrame(rows, columns=["tag", "t", "value"])
    frame["t"] = _T0 + pd.to_timedelta(frame["t"], unit="s")
    return frame.sort_values(["tag", "t"]).reset_index(drop=True)


def _run(monkeypatch, frame: pd.DataFrame, days: int = 1, **kwargs) -> dict:
    """Run the tool over a synthetic frame, with the historian sealed off."""
    monkeypatch.setattr(svc, "_raw", lambda days: frame)
    return svc.sep_oil_loss(days=days, **kwargs)


def _hours(seconds: float) -> float:
    return seconds / 3600.0


# --- 1. nothing wrong, nothing charged -------------------------------------


def test_steady_clean_operation_charges_nothing(monkeypatch):
    """A separator running on spec must bill exactly zero. If this fails the
    tool has a floor of phantom loss and every number it reports is inflated."""
    payload = _run(monkeypatch, _hist([(12 * 3600, _FLOW, 100.0, 50.0)]))

    assert payload["periods"], "a 12 h window must still produce the 24 h row"
    for row in payload["periods"]:
        assert row["bbl_upper"] == 0.0
        assert row["bbl_lower"] == 0.0
    assert payload["events"] == []
    assert payload["series"]["cum_upper"][-1] == 0.0


# --- 2. the film correction (the one that matters most) --------------------


def test_a_filmed_analyzer_is_not_billed_as_oil(monkeypatch):
    """The whole reason `_film_baseline` exists. A Red Eye coated flat at 96%
    reads 4 points low forever; charging `100 - wc` invents ~2,800 BOPD that
    is not there. Referenced against its own plateau the answer is zero."""
    payload = _run(monkeypatch, _hist([(12 * 3600, _FLOW, 96.0, 50.0)]))

    row = payload["periods"][0]
    naive = _FLOW * 0.04 * row["hours"] / 24.0  # what `100 - wc` would bill
    assert naive > 1_000.0, "the naive integral has to be big enough to matter"
    assert row["bbl_upper"] < 1.0
    assert row["bbl_lower"] < 1.0
    assert row["base_avg"] == pytest.approx(96.0, abs=0.05)
    assert payload["events"] == []


# --- 3. a real carry-under is charged -------------------------------------


def _carry_under_frame() -> pd.DataFrame:
    """4 h clean at 99%, a 2 h drop to 20%, 2 h clean. Level held normal."""
    return _hist(
        [
            (4 * 3600, _FLOW, 99.0, 50.0),
            (2 * 3600, _FLOW, 20.0, 50.0),
            (2 * 3600, _FLOW, 99.0, 50.0),
        ]
    )


def test_a_real_water_cut_drop_is_charged_at_the_hand_integral(monkeypatch):
    """The headline number. A 2 h sweep to 20% water cut off a 99% plateau at
    70,000 BPD is 70000 x 0.79 x 2/24 barrels; anything else means the
    baseline, the gate or the time weighting drifted."""
    payload = _run(monkeypatch, _carry_under_frame())

    expected = _FLOW * (99.0 - 20.0) / 100.0 * 2.0 / 24.0
    row = payload["periods"][0]
    assert row["base_avg"] == pytest.approx(99.0, abs=0.05)
    assert row["bbl_upper"] == pytest.approx(expected, rel=0.01)
    assert row["upset_hours"] == pytest.approx(2.0, abs=0.02)

    assert len(payload["events"]) == 1
    event = payload["events"][0]
    assert event["wc_min"] == pytest.approx(20.0, abs=0.01)
    assert event["hours"] == pytest.approx(2.0, abs=0.02)
    assert event["bbl_upper"] == pytest.approx(expected, rel=0.01)


# --- 4. the band -----------------------------------------------------------


def test_the_band_is_ordered_and_the_fraction_cap_bites(monkeypatch):
    """`bbl_lower` is the conservative end and must stay there, and tightening
    `max_oil_frac` must actually move it - a cap that never binds is a knob
    the page cannot use."""
    frame = _carry_under_frame()
    loose = _run(monkeypatch, frame, max_oil_frac=0.25)
    tight = _run(monkeypatch, frame, max_oil_frac=0.10)

    for payload in (loose, tight):
        for row in payload["periods"]:
            assert row["bbl_lower"] <= row["bbl_upper"]
            assert row["bopd_lower"] <= row["bopd_upper"]
        for event in payload["events"]:
            assert event["bbl_lower"] <= event["bbl_upper"]

    assert tight["periods"][0]["bbl_lower"] < loose["periods"][0]["bbl_lower"]
    # The cap is on the oil FRACTION of the leg, so the bound is exactly that.
    assert loose["periods"][0]["bbl_lower"] == pytest.approx(
        _FLOW * 0.25 * 2.0 / 24.0, rel=0.01
    )


def test_the_field_oil_ceiling_caps_the_upper_bound(monkeypatch):
    """A 150,000 BPD leg at 0% water cut implies ~148,500 BOPD out the water
    leg, which is more oil than Milne produces. The upper bound is a physical
    ceiling, not a meter reading."""
    frame = _hist(
        [
            (2 * 3600, 150_000.0, 99.0, 50.0),
            (10 * 3600, 150_000.0, 0.0, 8.0),
        ]
    )
    payload = _run(monkeypatch, frame, field_oil_bopd=60_000.0)

    for row in payload["periods"]:
        assert row["bopd_upper"] <= 60_000.0 + 1e-6
    rates = [v for v in payload["series"]["oil_upper"] if v is not None]
    assert max(rates) == pytest.approx(60_000.0, rel=1e-6)
    assert max(rates) < 150_000.0 * 0.99  # uncapped this would be ~148,500


# --- 5. the gate is on flow, never on the analyzer -------------------------


def test_downtime_is_excluded_on_flow_and_a_low_water_cut_still_charges(monkeypatch):
    """Two halves of one rule. Down hours are found by the FLOW gate - both
    the negative-drift stretch and the one where the meter sits near zero
    while the analyzer still reads a clean 99% - and they create no event. But
    a LOW WATER CUT AT GOOD FLOW is real carry-under and must still be
    charged. Gating on the analyzer value fails this test at both ends: it
    keeps the near-zero-flow hours and erases the real loss."""
    frame = _hist(
        [
            (4 * 3600, _FLOW, 99.0, 50.0),
            (2 * 3600, -200.0, 0.0, 50.0),  # down: meter drifting negative
            (1 * 3600, 200.0, 99.0, 50.0),  # down: meter near zero, wc clean
            (4 * 3600, _FLOW, 99.0, 50.0),
            (1 * 3600, _FLOW, 20.0, 50.0),  # real carry-under at full flow
        ]
    )
    payload = _run(monkeypatch, frame)

    assert payload["excluded_hours"] == pytest.approx(3.0, abs=0.02)
    assert payload["valid_hours"] == pytest.approx(9.0, abs=0.02)

    # The down hours carry no barrels: the total is exactly the 1 h dip.
    expected = _FLOW * (99.0 - 20.0) / 100.0 * 1.0 / 24.0
    assert payload["periods"][0]["bbl_upper"] == pytest.approx(expected, rel=0.01)

    assert len(payload["events"]) == 1
    assert payload["events"][0]["wc_min"] == pytest.approx(20.0, abs=0.01)
    # A 0% reading while the separator is off is never an event.
    assert all(e["wc_min"] > 0.0 for e in payload["events"])


# --- 6. event stitching ---------------------------------------------------


def test_event_stitching_merges_oscillation_and_drops_blips(monkeypatch):
    """The interface oscillates on a 5-10 min cycle, so the raw mask shatters
    one upset into slivers. Dips inside EVENT_MERGE_MINUTES are one event,
    dips clearly outside it stay separate, and anything shorter than
    EVENT_MIN_MINUTES is noise."""
    frame = _hist(
        [
            (3600, _FLOW, 99.0, 50.0),
            (600, _FLOW, 20.0, 50.0),  # dip 1, 10 min
            (300, _FLOW, 99.0, 50.0),  # 5 min gap: inside the merge window
            (600, _FLOW, 30.0, 50.0),  # dip 2, merges with dip 1
            (2400, _FLOW, 99.0, 50.0),  # 40 min gap: clearly separate
            (600, _FLOW, 25.0, 50.0),  # dip 3, its own event
            (1800, _FLOW, 99.0, 50.0),
            (60, _FLOW, 25.0, 50.0),  # 1 min blip, below EVENT_MIN_MINUTES
            (3600, _FLOW, 99.0, 50.0),
        ]
    )
    payload = _run(monkeypatch, frame)
    events = payload["events"]

    assert len(events) == 2
    # The merged event carries dip 1's minimum; dip 3 stands alone.
    assert {round(e["wc_min"], 1) for e in events} == {20.0, 25.0}

    merged = [e for e in events if e["wc_min"] == pytest.approx(20.0, abs=0.01)]
    single = [e for e in events if e["wc_min"] == pytest.approx(25.0, abs=0.01)]
    assert len(merged) == 1 and len(single) == 1
    # 10 min + the 5 min gap + 10 min, charged as one upset.
    assert merged[0]["hours"] == pytest.approx(_hours(1500), abs=0.01)
    assert single[0]["hours"] == pytest.approx(_hours(600), abs=0.01)
    assert all(e["hours"] * 60.0 >= svc.EVENT_MIN_MINUTES for e in events)


# --- 7. level signature, on the CONTROLLED level ---------------------------


def _level_frame(dip_level: float, dip_sp: float | None = None) -> pd.DataFrame:
    """One 1 h excursion, with the controlled level and its setpoint set."""
    sp = dip_level if dip_sp is None else dip_sp
    return _hist(
        [
            (2 * 3600, _FLOW, 99.0, 50.0, 50.0),
            (3600, _FLOW, 20.0, dip_level, sp),
            (2 * 3600, _FLOW, 99.0, 50.0, 50.0),
        ]
    )


def test_a_collapsing_level_classifies_the_event_as_level_loss(monkeypatch):
    """Losing the vessel's water inventory outranks every other signature -
    it is the one upset that is unambiguously a level problem."""
    payload = _run(monkeypatch, _level_frame(8.0))

    assert len(payload["events"]) == 1
    event = payload["events"][0]
    assert event["level_min"] == pytest.approx(8.0, abs=0.05)
    assert event["kind"] == "level loss"


def test_holding_well_under_setpoint_is_off_setpoint(monkeypatch):
    """The loop is calling for 60 and holding 35. The vessel is not empty, so
    this is not level loss, but the loop is not in control either."""
    payload = _run(monkeypatch, _level_frame(35.0, dip_sp=60.0))

    event = payload["events"][0]
    assert event["level_sp_avg"] == pytest.approx(60.0, abs=0.05)
    assert event["level_dev_avg"] == pytest.approx(-25.0, abs=0.5)
    assert event["kind"] == "off setpoint"


def test_level_held_at_setpoint_through_the_upset_is_at_setpoint(monkeypatch):
    """The finding this tool exists to surface: level exactly where it was
    asked and the water leg ran oil anyway, so separation failed rather than
    level control. Collapsing this into a generic "normal level" bucket would
    hide the distinction that decides which crew gets called."""
    payload = _run(monkeypatch, _level_frame(50.0))

    event = payload["events"][0]
    assert event["level_min"] == pytest.approx(50.0, abs=0.05)
    assert event["level_dev_avg"] == pytest.approx(0.0, abs=0.5)
    assert event["kind"] == "at setpoint"


# --- 8. per FIELD calendar day --------------------------------------------


def _two_day_frame() -> pd.DataFrame:
    """Clean day one, a 2 h carry-under early on day two, then 12 h clean.

    _T0 is field midnight, so the segments land on known Alaska dates.
    """
    return _hist(
        [
            (24 * 3600, _FLOW, 99.0, 50.0),
            (2 * 3600, _FLOW, 49.0, 50.0),
            (10 * 3600, _FLOW, 99.0, 50.0),
        ]
    )


def test_daily_rows_split_on_field_midnight_and_partition_the_barrels(monkeypatch):
    """The bars are per FIELD day: a UTC cut would put an Alaska night-shift
    upset on the wrong bar, and the day rows must account for every barrel the
    window charged or the chart quietly disagrees with the cards."""
    payload = _run(monkeypatch, _two_day_frame(), days=7)
    daily = payload["daily"]

    assert [d["date"] for d in daily] == ["2026-08-01", "2026-08-02"]
    # The excursion is entirely on day two.
    assert daily[0]["bbl_upper"] == 0.0
    assert daily[0]["events"] == 0
    assert daily[1]["bbl_upper"] > 0.0
    assert daily[1]["events"] == 1
    assert daily[1]["upset_hours"] == pytest.approx(2.0, abs=0.05)

    # Hand integral: 2 h at 50 points below a 99 plateau.
    expected = _FLOW * 0.50 * 2.0 / 24.0
    assert daily[1]["bbl_upper"] == pytest.approx(expected, rel=0.02)

    window = next(p for p in payload["periods"] if p["days"] == 7)
    assert sum(d["bbl_upper"] for d in daily) == pytest.approx(window["bbl_upper"], rel=0.01)
    assert sum(d["bbl_lower"] for d in daily) == pytest.approx(window["bbl_lower"], rel=0.01)
    assert sum(d["events"] for d in daily) == window["events"]


def test_a_clipped_day_is_flagged_partial(monkeypatch):
    """A day the window only clips always looks quiet. Without the flag an
    engineer reads a short bar as a good day."""
    daily = _run(monkeypatch, _two_day_frame(), days=7)["daily"]

    assert daily[0]["partial"] is False
    assert daily[0]["covered_hours"] == pytest.approx(24.0, abs=0.05)
    # Day two is only 12 h of window, so it is not comparable bar for bar.
    assert daily[1]["partial"] is True
    assert daily[1]["covered_hours"] == pytest.approx(12.0, abs=0.05)


def test_a_day_the_separator_barely_ran_reports_no_share(monkeypatch):
    """Percent of field oil divides by the running hours. On a day with
    minutes of runtime that denominator turns a few barrels into a headline
    percentage, so the share is withheld and only the barrels stand."""
    frame = _hist(
        [
            # Down almost the whole field day, so what runtime day one has is
            # the 10 min upset plus a 5 min tail - well under the 1 h guard.
            (23.75 * 3600, -200.0, 0.0, 50.0),
            (600, _FLOW, 20.0, 50.0),
            (24 * 3600, _FLOW, 99.0, 50.0),
        ]
    )
    daily = _run(monkeypatch, frame, days=7)["daily"]

    short = daily[0]
    assert short["hours"] < 1.0
    assert short["bbl_upper"] > 0.0
    assert short["pct_field_upper"] is None
    assert short["pct_field_lower"] is None
    assert short["partial"] is True


# --- 9. the single-day drill-down -----------------------------------------


def test_the_day_drilldown_agrees_with_the_bar_it_opens(monkeypatch):
    """The drill-down must not restate the day. Re-running the event walk on
    a one-day slice would re-split an upset that straddles midnight, and the
    detail view would contradict the bar the engineer clicked."""
    frame = _two_day_frame()
    monkeypatch.setattr(svc, "_raw", lambda days: frame)

    window = svc.sep_oil_loss(days=7)
    bar = next(d for d in window["daily"] if d["date"] == "2026-08-02")
    day = svc.sep_oil_loss_day("2026-08-02", days=7)

    assert day["summary"] == bar
    assert len(day["events"]) == bar["events"]
    assert all(e["start"][:10] == "2026-08-02" for e in day["events"])


def test_the_day_drilldown_is_finer_than_the_window(monkeypatch):
    """The whole point: the window spends its point budget on every day, so a
    day read off it is coarse. The slice must actually buy resolution."""
    frame = _two_day_frame()
    monkeypatch.setattr(svc, "_raw", lambda days: frame)

    window_pts = svc.sep_oil_loss(days=7)["series"]["t"]
    day_pts = svc.sep_oil_loss_day("2026-08-02", days=7)["series"]["t"]

    assert all(t[:10] == "2026-08-02" for t in day_pts)
    span_window = pd.Timestamp(window_pts[-1]) - pd.Timestamp(window_pts[0])
    span_day = pd.Timestamp(day_pts[-1]) - pd.Timestamp(day_pts[0])
    per_window = span_window / max(len(window_pts) - 1, 1)
    per_day = span_day / max(len(day_pts) - 1, 1)
    assert per_day < per_window


def test_the_day_drilldown_refuses_a_day_outside_the_window(monkeypatch):
    """A stale date from a shrunk window must fail loudly, not silently
    return the nearest day's trace under the wrong heading."""
    frame = _two_day_frame()
    monkeypatch.setattr(svc, "_raw", lambda days: frame)

    with pytest.raises(ValueError, match="outside"):
        svc.sep_oil_loss_day("1999-01-01", days=7)
    with pytest.raises(ValueError, match="not a YYYY-MM-DD date"):
        svc.sep_oil_loss_day("not-a-date", days=7)
    with pytest.raises(ValueError, match="days must be"):
        svc.sep_oil_loss_day("2026-08-02", days=0)


# --- 10. what the chart is handed -----------------------------------------


def test_series_arrays_are_parallel_bounded_and_cumulative(monkeypatch):
    """The chart plots these against one shared time axis and the cumulative
    curve is a running total: ragged arrays would mis-register every trace and
    a dip in `cum_upper` would mean the integral ran backwards."""
    frame = _hist(
        [
            (2 * 3600, _FLOW, 99.0, 50.0),
            (3600, _FLOW, 20.0, 8.0),
            (3 * 3600, _FLOW, 99.0, 50.0),
        ]
    )
    series = _run(monkeypatch, frame)["series"]

    keys = (
        "t", "flow", "wc", "base", "level", "level_sp",
        "oil_upper", "cum_upper", "cum_lower",
    )
    assert set(series) == set(keys)
    lengths = {len(series[k]) for k in keys}
    assert len(lengths) == 1
    size = lengths.pop()
    assert 0 < size <= svc.MAX_SERIES_POINTS
    assert size < 6 * 360  # 6 h at 10 s was downsampled, not shipped whole

    for name in ("cum_upper", "cum_lower"):
        values = series[name]
        assert all(b >= a for a, b in zip(values, values[1:])), name
    assert series["cum_upper"][-1] > 0.0


# --- 11. arguments and an empty historian ---------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"days": 0},
        {"days": 91},
        {"max_oil_frac": 0.0},
        {"max_oil_frac": 1.5},
        {"field_oil_bopd": 100.0},
    ],
)
def test_out_of_range_arguments_raise(kwargs):
    """Every bound is refused before a query is issued. A silently clamped
    argument would return a plausible number for a window nobody asked for."""
    with pytest.raises(ValueError):
        svc.sep_oil_loss(**kwargs)


def test_an_empty_historian_returns_an_empty_payload(monkeypatch):
    """A view outage must render an empty page, not a 500."""
    empty = pd.DataFrame(columns=["tag", "t", "value"])
    payload = _run(monkeypatch, empty, days=30)

    assert payload["periods"] == []
    assert payload["daily"] == []
    assert payload["events"] == []
    assert payload["start"] is None and payload["end"] is None
    assert payload["valid_hours"] == 0.0
    assert all(v == [] for v in payload["series"].values())
    schemas.SepOilLossResponse(**payload)


# --- 12. the HTTP surface -------------------------------------------------


def test_the_endpoint_serves_the_schema_and_refuses_bad_windows(monkeypatch):
    """The React page reads this contract directly, and the query bounds are
    the only thing standing between a typo and a 90-day historian scan."""
    monkeypatch.setattr(svc, "_raw", lambda days: _carry_under_frame())
    client = TestClient(app)

    ok = client.get("/api/tools/sep-oil-loss", params={"days": 7})
    assert ok.status_code == 200
    body = ok.json()
    schemas.SepOilLossResponse(**body)
    assert body["flow_tag"] == "MPU_FI_5365"
    assert body["wc_tag"] == "MPU_AI_5317"
    assert body["level_tag"] == "MPU_LIC_5365CV1"
    assert body["level_sp_tag"] == "MPU_LC5365SP1"
    assert body["days"] == 7
    assert [p["label"] for p in body["periods"]] == ["Last 24 h", "Last 7 d"]
    assert body["events"]

    assert client.get("/api/tools/sep-oil-loss", params={"days": 0}).status_code == 422
    assert (
        client.get("/api/tools/sep-oil-loss", params={"max_oil_frac": 2}).status_code == 422
    )


def test_the_tool_is_in_the_catalog():
    """The menu renders from the catalog, so an unlisted tool is unreachable."""
    body = TestClient(app).get("/api/tools/catalog").json()
    entry = [t for t in body["tools"] if t["id"] == "sep-oil-loss"]
    assert len(entry) == 1
    assert entry[0]["path"] == "/tools/sep-oil-loss"


# --- 13. tag safety -------------------------------------------------------


def test_tag_names_are_shape_checked_before_they_reach_sql():
    """`execute_query` has no parameter binding, so tag names are spliced into
    the WHERE clause as literals. The shape check is the only thing stopping a
    tag string from carrying SQL."""
    assert historian_client.validate_tags([svc.FLOW_TAG, svc.WC_TAG]) == [
        "MPU_FI_5365",
        "MPU_AI_5317",
    ]

    for bad in ("MPU_FI_5365'; DROP TABLE x --", "MPU FI 5365", "", "a"):
        with pytest.raises(ValueError, match="unsafe historian tag"):
            historian_client.validate_tags([bad])
