"""Pump-curve report contract - shape, physics and pinned values.

Covers ``PadPlant.curve_report()`` for the three modelled pads and the
``GET /api/optimize/pump-curve`` endpoint that serves it. Nothing here reaches
Databricks or the network: curve_report is pure file-backed physics, so the
endpoint tests deliberately patch NOTHING.

Provenance anchors (loose ``abs=`` tolerances tie the model to its source):

  * S-Pad ``S_Pad_Pumps/pump_curve_for_repo.csv`` ``eff_pct`` - the committed
    vendor sheet, 79 stg at 3,500 RPM, SG 1.0
  * I-Pad ``I_Pad_Pumps/I-Pad_pump_curves_for_repo.csv`` ``stage_eff_pct`` -
    one shared stage curve at 60 Hz, SG 1.0
  * M-Pad Schlumberger set-up sheet ``design_point_60hz_SG1.05`` - 78.2 pct at
    the 32,000 BPD design point, BEP flow 27,708 BPD, DESIGN SG 1.05 (not the
    field 1.03 used for the head-to-psi conversion)
  * meta JSON marker numbers: S bep 12,000/pump and thrust window
    7,650-18,360/pump; I bep 41,250, AOR 33,000-49,500, min continuous 12,400;
    M bep 27,708/pump, recommended 8,351-34,798/pump, wear factor 0.91

Everything at ``rel=1e-6`` is a regression pin: today's computed number, which
a future refactor must reproduce. Each pin is derived by calling the same
plant method the report itself uses, so the expectation stays re-derivable.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import woffl.gui.i_pad_plant as i_pad
import woffl.gui.m_pad_plant as m_pad
import woffl.gui.s_pad_plant as s_pad
from server import schemas
from server.main import app

# contract: 61 samples per curve, inclusive of both ends, so index 0 is
# shut-off (q = 0) and index 30 is exactly the midpoint of the grid
_CURVE_POINTS = 61
_MID = 30

_S_STAGES = 79  # S-Pad meta "n_stages"
_JP_DATA = Path(__file__).resolve().parents[1] / "woffl" / "jp_data"

_PLANTS = {"S": s_pad.PLANT, "I": i_pad.PLANT, "M": m_pad.PLANT}
_PADS = ["S", "I", "M"]

_TOP_KEYS = {
    "pad",
    "coupling",
    "n_pumps",
    "sg",
    "suction_psi",
    "max_header_psi",
    "nameplate",
    "station",
    "pumps",
}
_NAMEPLATE_KEYS = {"equipment", "model", "arrangement", "speed", "source", "validated"}
_STATION_KEYS = {"curves", "frontier", "bep", "por", "aor", "min_flow", "header_cap"}
_LINE_KEYS = {"label", "n_pumps", "hz", "active", "points"}
_MACHINE_KEYS = {
    "label",
    "hz",
    "points",
    "head_derated",
    "derate_note",
    "bep",
    "por",
    "aor",
    "min_flow",
}
_JSON_SCALARS = (bool, int, float, str, type(None))

# The 5th-order Summit stage poly RISES 0.09 pct over its first 1,000 BPD
# (committed I-Pad CSV: 234.55 -> 234.76 ft/stage at 0 -> 1,000). That is a fit
# artifact, not physics, so a strictly non-increasing check is not honest for
# that pad. 0.2 pct of the shut-off value leaves the artifact alone while still
# catching a curve that actually climbs (sign error, mis-ordered poly, swapped
# axis) - and every curve still has to fall overall.
_RISE_TOL_FRAC = 0.002

_REPORT_CACHE: dict = {}


def report(pad: str, n_pumps: int | None = None) -> dict:
    """Memoized curve_report (the I-Pad frontier bisects at all 61 points)."""
    key = (pad, n_pumps)
    if key not in _REPORT_CACHE:
        _REPORT_CACHE[key] = _PLANTS[pad].curve_report(n_pumps)
    return _REPORT_CACHE[key]


def active_curve(rep: dict) -> dict:
    hits = [c for c in rep["station"]["curves"] if c["active"] is True]
    assert len(hits) == 1, "exactly one station curve must be active"
    return hits[0]


def leaves(node, path="$"):
    """Yield (json path, value) for every scalar leaf of a nested payload."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from leaves(v, path + "." + str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from leaves(v, path + "[" + str(i) + "]")
    else:
        yield path, node


def assert_falls(label: str, ys: list) -> None:
    """A centrifugal head/discharge curve must fall with flow."""
    assert ys[-1] < ys[0], label + ": curve does not fall across the grid"
    tol = _RISE_TOL_FRAC * abs(ys[0])
    for i in range(len(ys) - 1):
        assert ys[i + 1] <= ys[i] + tol, "%s: rises at point %d (%r -> %r)" % (
            label,
            i,
            ys[i],
            ys[i + 1],
        )


def read_curve_csv(rel_path: str) -> list[dict]:
    with open(_JP_DATA / rel_path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def csv_row(rows: list[dict], column: str, value: float) -> dict:
    hits = [r for r in rows if float(r[column]) == value]
    assert len(hits) == 1, "no unique %s == %r row in the committed CSV" % (
        column,
        value,
    )
    return hits[0]


# ---------------------------------------------------------------------------
# Shape - the payload the schema, the hook and the charts all agree on
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pad", _PADS)
def test_report_carries_every_contract_key(pad):
    rep = report(pad)
    assert set(rep) == _TOP_KEYS
    assert rep["pad"] == pad
    assert set(rep["nameplate"]) == _NAMEPLATE_KEYS
    assert all(isinstance(v, str) and v for v in rep["nameplate"].values())
    assert set(rep["station"]) == _STATION_KEYS

    curves = rep["station"]["curves"]
    assert len(curves) >= 1
    for curve in curves:
        assert set(curve) == _LINE_KEYS
        assert len(curve["points"]) == _CURVE_POINTS
        assert all(len(p) == 2 for p in curve["points"])
    active_curve(rep)  # exactly one active line

    frontier = rep["station"]["frontier"]
    if frontier is not None:
        assert set(frontier) == _LINE_KEYS
        assert all(len(p) == 2 for p in frontier["points"])

    assert len(rep["pumps"]) >= 1
    for machine in rep["pumps"]:
        assert set(machine) == _MACHINE_KEYS
        assert len(machine["points"]) == _CURVE_POINTS
        assert all(len(p) == 4 for p in machine["points"])
        if machine["head_derated"] is not None:
            assert all(len(p) == 2 for p in machine["head_derated"])


@pytest.mark.parametrize("pad", _PADS)
def test_report_is_json_safe(pad):
    # The payload crosses FastAPI: plain scalars only, no numpy, no pandas.
    rep = report(pad)
    for path, value in leaves(rep):
        assert type(value) in _JSON_SCALARS, "%s is %s" % (path, type(value))
    assert json.loads(json.dumps(rep)) == rep


@pytest.mark.parametrize("pad", _PADS)
def test_report_scalars_come_from_the_plant(pad):
    rep = report(pad)
    plant = _PLANTS[pad]
    assert rep["coupling"] == plant.coupling
    assert rep["n_pumps"] == plant._n(None)
    assert rep["sg"] == pytest.approx(plant.specific_gravity(), rel=1e-12)
    assert rep["suction_psi"] == pytest.approx(plant.suction_psi(), rel=1e-12)
    assert rep["max_header_psi"] == getattr(plant, "max_header_psi", None)


# ---------------------------------------------------------------------------
# Physics - falling head, bounded efficiency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pad", _PADS)
def test_station_and_machine_curves_fall_with_flow(pad):
    rep = report(pad)
    for curve in rep["station"]["curves"]:
        flows = [p[0] for p in curve["points"]]
        assert flows == sorted(flows)
        assert_falls(pad + " station " + curve["label"], [p[1] for p in curve["points"]])
    frontier = rep["station"]["frontier"]
    if frontier is not None:
        assert_falls(pad + " frontier", [p[1] for p in frontier["points"]])
    for machine in rep["pumps"]:
        assert_falls(pad + " " + machine["label"], [p[1] for p in machine["points"]])
        if machine["head_derated"] is not None:
            assert_falls(pad + " derated", [p[1] for p in machine["head_derated"]])


@pytest.mark.parametrize("pad", _PADS)
def test_machine_curve_efficiency_is_bounded_and_zero_at_shut_off(pad):
    for machine in report(pad)["pumps"]:
        q0, head0, bhp0, eff0 = machine["points"][0]
        assert q0 == 0.0
        assert head0 > 0.0 and bhp0 > 0.0
        assert eff0 == 0.0  # no flow, no hydraulic work
        for _q, _head, bhp, eff in machine["points"]:
            assert bhp > 0.0
            assert 0.0 <= eff <= 100.0


# ---------------------------------------------------------------------------
# Efficiency - the committed vendor columns, and the SG each was fit at
# ---------------------------------------------------------------------------


def test_hydraulic_efficiency_reproduces_s_pad_csv():
    # Vendor sheet columns are already whole-pump (79 stg) at SG 1.0.
    rows = read_curve_csv("S_Pad_Pumps/pump_curve_for_repo.csv")
    for flow in (12000.0, 16000.0):  # 79.4 pct and 77.4 pct in the sheet
        row = csv_row(rows, "flow_per_pump_bpd", flow)
        got = s_pad.PLANT.hydraulic_efficiency(
            flow, float(row["head_ft"]), float(row["bhp_hp"]), 1.0
        )
        assert got == pytest.approx(float(row["eff_pct"]), abs=0.05)


def test_hydraulic_efficiency_reproduces_i_pad_csv():
    # I-Pad publishes a per-STAGE curve; head and BHP per stage, SG 1.0.
    rows = read_curve_csv("I_Pad_Pumps/I-Pad_pump_curves_for_repo.csv")
    for flow in (20000.0, 30000.0):  # 61.4 pct and 73.0 pct in the sheet
        row = csv_row(rows, "flow_bpd_60hz", flow)
        got = i_pad.PLANT.hydraulic_efficiency(
            flow,
            float(row["head_ft_per_stage"]),
            float(row["bhp_per_stage_hp"]),
            1.0,
        )
        assert got == pytest.approx(float(row["stage_eff_pct"]), abs=0.05)


def test_hydraulic_efficiency_degenerate_inputs():
    eff = s_pad.PLANT.hydraulic_efficiency
    # Zero flow does zero hydraulic work no matter what the head poly says.
    assert eff(0.0, 8470.5, 434.0, 1.0) == 0.0
    # Non-positive BHP makes the ratio undefined, not infinite.
    assert eff(12000.0, 7079.6, 0.0, 1.0) == 0.0
    assert eff(12000.0, 7079.6, -5.0, 1.0) == 0.0


def test_m_pad_machine_efficiency_uses_the_design_sg():
    # The M-Pad BHP poly was fit at the Schlumberger DESIGN SG 1.05, while the
    # head-to-psi conversion uses the FIELD SG 1.03. This is the test that
    # catches someone "fixing" the efficiency SG to 1.03: the same peak drops
    # to ~77.0 pct, outside the band below, and every M-Pad efficiency number
    # in the UI silently shifts by about 1.5 points.
    points = report("M")["pumps"][0]["points"]
    q_peak, _head, _bhp, eff_peak = max(points, key=lambda p: p[3])
    assert q_peak == pytest.approx(27708.0, abs=1000.0)  # meta bep_flow_bpd_60hz
    assert 78.0 <= eff_peak <= 79.0  # datasheet design point 78.2 pct


# ---------------------------------------------------------------------------
# Markers - BEP / POR / AOR / min flow / header cap
# ---------------------------------------------------------------------------


def test_s_pad_station_markers_scale_with_pump_count():
    # Parallel bank: every per-pump meta number x 3.
    station = report("S")["station"]
    assert station["bep"] == pytest.approx(12000.0 * 3)  # bep_flow_per_pump_bpd
    assert station["aor"] == pytest.approx([7650.0 * 3, 18360.0 * 3])
    assert station["min_flow"] is None
    assert station["header_cap"] is None  # fixed speed, no operational cap


def test_i_pad_station_markers_are_train_totals():
    # Series train: the same fluid passes both pumps, so nothing is multiplied.
    station = report("I")["station"]
    assert station["bep"] == pytest.approx(41250.0)  # stage bep_flow_bpd
    assert station["aor"] == pytest.approx([33000.0, 49500.0])
    assert station["min_flow"] == pytest.approx(12400.0)  # min_continuous_flow_bpd
    assert station["header_cap"] == 3500.0


def test_m_pad_station_markers_scale_with_pump_count():
    station = report("M")["station"]
    assert station["bep"] == pytest.approx(27708.0 * 3)  # bep_flow_bpd_60hz
    assert station["aor"] == pytest.approx([8351.0 * 3, 34798.0 * 3])
    assert station["min_flow"] == pytest.approx(m_pad.PLANT.min_total_flow(3))
    assert station["header_cap"] == 3500.0


@pytest.mark.parametrize("pad", _PADS)
def test_por_is_seventy_to_one_twenty_percent_of_bep(pad):
    for markers in [report(pad)["station"]] + report(pad)["pumps"]:
        bep = markers["bep"]
        assert bep is not None and bep > 0.0
        assert markers["por"] == pytest.approx([0.70 * bep, 1.20 * bep], rel=1e-12)


@pytest.mark.parametrize("pad", ["S", "M"])
def test_por_sits_inside_aor_on_the_parallel_pads(pad):
    station = report(pad)["station"]
    lo, hi = station["por"]
    aor_lo, aor_hi = station["aor"]
    assert aor_lo <= lo and hi <= aor_hi


def test_i_pad_por_floor_sits_below_the_vendor_aor_floor():
    # NOT a nesting bug to "fix" by clamping. Summit's recommended range for
    # the I-Pad stage is 80-120 pct of BEP (33,000-49,500 on a 41,250 BEP),
    # tighter on the low side than the generic 70-120 pct POR, so the POR floor
    # lands 4,125 BPD to the left of the AOR floor. The contract says emit both
    # as-is so the chart shows the real gap.
    station = report("I")["station"]
    bep = station["bep"]
    assert station["aor"] == pytest.approx([0.80 * bep, 1.20 * bep], rel=1e-12)
    assert station["por"][0] == pytest.approx(station["aor"][0] - 4125.0)
    assert station["por"][1] == pytest.approx(station["aor"][1])


# ---------------------------------------------------------------------------
# Per-pad curve families
# ---------------------------------------------------------------------------


def test_s_pad_family_is_one_line_per_pump_count():
    rep = report("S")
    curves = rep["station"]["curves"]
    assert [c["n_pumps"] for c in curves] == [1, 2, 3]
    assert [c["label"] for c in curves] == ["1 pump", "2 pumps", "3 pumps"]
    assert all(c["hz"] == 60.0 for c in curves)  # no VFD on this pad
    assert active_curve(rep)["n_pumps"] == 3
    assert rep["station"]["frontier"] is None  # fixed speed: the curve IS it
    # 1.12 up-thrust headroom past the recommended max, per the Streamlit plot
    assert curves[2]["points"][-1][0] == pytest.approx(18360.0 * 1.12 * 3)
    assert len(rep["pumps"]) == 1
    # the head poly's stated valid range
    assert rep["pumps"][0]["points"][-1][0] == pytest.approx(21000.0)
    assert rep["pumps"][0]["head_derated"] is None
    assert rep["pumps"][0]["derate_note"] is None


def test_i_pad_family_is_iso_speed_lines_plus_an_amp_frontier():
    rep = report("I")
    curves = rep["station"]["curves"]
    assert [c["hz"] for c in curves] == [45.0, 50.0, 55.0, 60.0]
    assert all(c["n_pumps"] is None for c in curves)  # fixed 2-pump train
    assert active_curve(rep)["hz"] == 60.0
    # affinity: the stage poly is only valid to max_valid_flow at 60 Hz
    assert curves[-1]["points"][-1][0] == pytest.approx(i_pad.PLANT.max_valid_flow())
    assert curves[0]["points"][-1][0] == pytest.approx(
        i_pad.PLANT.max_valid_flow() * 45.0 / 60.0
    )
    frontier = rep["station"]["frontier"]
    assert frontier is not None
    assert "192" in frontier["label"] and "154" in frontier["label"]  # amp limits


def test_i_pad_frontier_never_exceeds_the_max_speed_line():
    # The frontier is the amp-limited capability. Below ~24,950 BPD the drive
    # is speed-limited, not amp-limited (hz_at_amp_limit caps at 60), so the
    # frontier sits exactly ON the 60 Hz line; above that the amp limit binds
    # and it falls away underneath. Both halves matter: a frontier that ran
    # above the 60 Hz line would be selling speed the drive does not have.
    plant = i_pad.PLANT

    def iso60(q: float) -> float:
        return plant.suction_psi() + sum(
            plant.pump_dP(p["n_stages"], q, 60.0) for p in plant.pumps()
        )

    points = report("I")["station"]["frontier"]["points"]
    assert points, "frontier must carry at least one deliverable point"
    for q, psi in points:
        assert psi <= iso60(q) + 1e-9
        if q <= 23000.0:
            assert psi == pytest.approx(iso60(q), rel=1e-9)
        elif q >= 26000.0:
            assert psi < iso60(q) - 1.0


def test_i_pad_machine_curves_are_lp_then_hp():
    machines = report("I")["pumps"]
    assert len(machines) == 2
    assert "LP" in machines[0]["label"] and "HP" in machines[1]["label"]
    assert all(m["hz"] == 60.0 for m in machines)
    # series order is physical: 26 stages ahead of 17, same shared stage curve
    lp_head = machines[0]["points"][_MID][1]
    hp_head = machines[1]["points"][_MID][1]
    assert lp_head == pytest.approx(hp_head * 26.0 / 17.0, rel=1e-9)
    assert machines[0]["bep"] == machines[1]["bep"]
    assert machines[0]["head_derated"] is None
    assert machines[1]["head_derated"] is None


def test_m_pad_family_is_iso_speed_lines_with_no_frontier():
    rep = report("M")
    curves = rep["station"]["curves"]
    assert [c["hz"] for c in curves] == [51.0, 55.0, 58.0, 61.0]  # freq_range_hz
    assert all(c["n_pumps"] == 3 for c in curves)
    assert active_curve(rep)["hz"] == m_pad.PLANT.hp()["hz_max"]
    # the max-speed line already IS the capability; header_cap carries the limit
    assert rep["station"]["frontier"] is None
    assert curves[-1]["points"][-1][0] == pytest.approx(
        m_pad.PLANT.hp()["rec_hi"] * (61.0 / 60.0) * 3
    )
    assert len(rep["pumps"]) == 1


def test_m_pad_machine_curve_carries_the_wear_derate():
    machine = report("M")["pumps"][0]
    wear = m_pad.PLANT.wear_factor()  # meta field_head_factor, 0.91
    assert machine["head_derated"] is not None
    assert len(machine["head_derated"]) == len(machine["points"])
    for (q, head, _bhp, _eff), (q_d, head_d) in zip(
        machine["points"], machine["head_derated"]
    ):
        assert q_d == pytest.approx(q, rel=1e-12)
        assert head_d == pytest.approx(head * wear, rel=1e-12)
    assert "0.91" in machine["derate_note"]
    assert machine["points"][-1][0] == pytest.approx(m_pad.PLANT.hp()["rec_hi"])


# ---------------------------------------------------------------------------
# n_pumps plumbing
# ---------------------------------------------------------------------------


def test_s_pad_n_pumps_argument_moves_the_whole_report():
    two = report("S", 2)
    three = report("S")
    assert two["n_pumps"] == 2
    assert active_curve(two)["n_pumps"] == 2
    assert active_curve(two)["label"] == "2 pumps"
    assert two["station"]["bep"] == pytest.approx(three["station"]["bep"] * 2.0 / 3.0)
    assert two["station"]["aor"] == pytest.approx(
        [x * 2.0 / 3.0 for x in three["station"]["aor"]]
    )
    # the machine curve is one pump's vendor sheet - pump count cannot move it
    assert two["pumps"][0]["points"] == three["pumps"][0]["points"]


# ---------------------------------------------------------------------------
# Regression pins - today's numbers, re-derived from the plant methods the
# report itself calls so the expectation stays honest under a refactor
# ---------------------------------------------------------------------------


def test_s_pad_shut_off_pin():
    point = active_curve(report("S"))["points"][0]
    assert point[0] == 0.0
    # 3,886.875 psi today
    assert point[1] == pytest.approx(s_pad.PLANT.discharge_pressure(0.0, 3), rel=1e-6)


def test_i_pad_shut_off_pin():
    plant = i_pad.PLANT
    expected = plant.suction_psi() + sum(
        plant.pump_dP(p["n_stages"], 0.0, 60.0) for p in plant.pumps()
    )
    point = active_curve(report("I"))["points"][0]
    assert point[0] == 0.0
    assert point[1] == pytest.approx(expected, rel=1e-6)  # 4,757.672 psi today


def test_m_pad_shut_off_pin():
    plant = m_pad.PLANT
    # wear-derated, at hz_max - the family the optimizer actually rides
    expected = plant.hp_suction_psi() + plant.pump_boost(0.0, 61.0)
    point = active_curve(report("M"))["points"][0]
    assert point[0] == 0.0
    assert point[1] == pytest.approx(expected, rel=1e-6)  # 4,140.772 psi today


def test_s_pad_machine_midpoint_pin():
    plant = s_pad.PLANT
    q, head, bhp, eff = report("S")["pumps"][0]["points"][_MID]
    assert q == pytest.approx(10500.0, rel=1e-12)  # 21,000 / 2
    head_exp = _S_STAGES * plant.head_per_stage(q)  # 7,418.472 ft today
    bhp_exp = _S_STAGES * plant.bhp_per_stage(q)
    assert head == pytest.approx(head_exp, rel=1e-6)
    assert bhp == pytest.approx(bhp_exp, rel=1e-6)
    assert eff == pytest.approx(
        plant.hydraulic_efficiency(q, head_exp, bhp_exp, 1.0), rel=1e-6
    )


def test_i_pad_machine_midpoint_pin():
    plant = i_pad.PLANT
    stages = plant.pumps()[0]["n_stages"]  # LP, 26 stg
    q, head, bhp, eff = report("I")["pumps"][0]["points"][_MID]
    assert q == pytest.approx(plant.max_valid_flow() / 2.0, rel=1e-12)
    head_exp = stages * plant.head_per_stage(q)  # 4,780.413 ft today
    bhp_exp = stages * plant.bhp_per_stage(q)  # 1,447.427 BHP today
    assert head == pytest.approx(head_exp, rel=1e-6)
    assert bhp == pytest.approx(bhp_exp, rel=1e-6)
    assert eff == pytest.approx(
        plant.hydraulic_efficiency(q, head_exp, bhp_exp, 1.0), rel=1e-6
    )


def test_m_pad_machine_midpoint_pin():
    plant = m_pad.PLANT
    hp = plant.hp()
    q, head, bhp, eff = report("M")["pumps"][0]["points"][_MID]
    assert q == pytest.approx(hp["rec_hi"] / 2.0, rel=1e-12)
    head_exp = plant._head_ft(hp, q)  # AS-NEW, 5,767.974 ft today
    bhp_exp = plant._bhp(hp, q)  # 1,079.958 BHP today
    assert head == pytest.approx(head_exp, rel=1e-6)
    assert bhp == pytest.approx(bhp_exp, rel=1e-6)
    # SG 1.05 (design), NOT the 1.03 field SG the head-to-psi conversion uses
    assert eff == pytest.approx(
        plant.hydraulic_efficiency(q, head_exp, bhp_exp, 1.05), rel=1e-6
    )


# ---------------------------------------------------------------------------
# GET /api/optimize/pump-curve - static physics, so nothing is monkeypatched:
# if this path ever grows a Databricks call these tests fail on the .env gate
# ---------------------------------------------------------------------------


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def test_pump_curve_endpoint_validates_against_the_schema(client):
    r = client.get("/api/optimize/pump-curve?pad=S")
    assert r.status_code == 200
    body = schemas.PumpCurveResponse.model_validate(r.json())
    assert body.pad == "S"
    assert body.n_pumps == 3  # the plant's own default
    assert body.station.bep == pytest.approx(report("S")["station"]["bep"])
    assert len(body.station.curves) == 3
    assert len(body.pumps) == 1


def test_pump_curve_endpoint_honors_n_pumps(client):
    r = client.get("/api/optimize/pump-curve?pad=M&n_pumps=2")
    assert r.status_code == 200
    body = r.json()
    assert body["n_pumps"] == 2
    assert body["station"]["bep"] == pytest.approx(27708.0 * 2)


def test_pump_curve_endpoint_carries_the_pump_count_options(client):
    # Drives the client's "pumps online" control: a pad run with a machine
    # down (e.g. one M-Pad HP pump out) is started at a reduced count. The
    # options ride OUTSIDE curve_report - its key set is contract-pinned by
    # test_report_carries_every_contract_key above.
    for pad, expected in (("M", [3, 2, 1]), ("S", [3, 2]), ("I", [])):
        r = client.get(f"/api/optimize/pump-curve?pad={pad}")
        assert r.status_code == 200
        body = schemas.PumpCurveResponse.model_validate(r.json())
        assert body.n_pump_options == expected


def test_pump_curve_endpoint_rejects_an_unmodelled_pad(client):
    assert client.get("/api/optimize/pump-curve?pad=Q").status_code == 422
