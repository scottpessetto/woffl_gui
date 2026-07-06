"""R-1 Phase C: the S/I/M pad pages are thin PadSpecs over woffl.gui.pad_page.

Guards the unification contract WITHOUT a Streamlit runtime:

* each spec carries the right pad letter / key prefix / plant / coupling /
  pump-count options (the prefix is what keeps every legacy ``sp_``/``ip_``/
  ``mp_`` session and widget key byte-identical);
* prefixes and pad letters are unique across pads;
* the module-level entry functions exist, take no required args, and are the
  exact names ``app.py`` imports;
* the pad-specific hooks are callables and produce the expected shapes from a
  plain meta dict (no Streamlit needed for the metric/caption hooks);
* a brand-new pad reduces to one ``PadSpec`` with a ``FixedHeaderPlant`` —
  construction + defaults only (rendering needs a Streamlit runtime).
"""

import inspect
from pathlib import Path

import pytest

from woffl.gui import (
    i_pad_page,
    i_pad_plant,
    m_pad_page,
    m_pad_plant,
    s_pad_page,
    s_pad_plant,
)
from woffl.gui.pad_page import PadSpec, run_pad_page
from woffl.gui.pad_plant_base import FixedHeaderPlant

SPECS = {"S": s_pad_page.SPEC, "I": i_pad_page.SPEC, "M": m_pad_page.SPEC}

_APP_PATH = Path(__file__).resolve().parent.parent / "woffl" / "gui" / "app.py"


class TestSpecIdentity:
    def test_s_spec(self):
        spec = s_pad_page.SPEC
        assert spec.pad == "S"
        assert spec.prefix == "sp"
        assert spec.plant is s_pad_plant.PLANT
        assert spec.plant.coupling == "fixed_curve"
        assert list(spec.n_pump_options) == [3, 2]
        assert spec.show_per_pump is True

    def test_i_spec(self):
        spec = i_pad_page.SPEC
        assert spec.pad == "I"
        assert spec.prefix == "ip"
        assert spec.plant is i_pad_plant.PLANT
        assert spec.plant.coupling == "free_pressure"
        assert spec.n_pump_options is None  # fixed LP+HP series train — no radio
        assert spec.n_steps == 11
        assert spec.show_per_pump is False

    def test_m_spec(self):
        spec = m_pad_page.SPEC
        assert spec.pad == "M"
        assert spec.prefix == "mp"
        assert spec.plant is m_pad_plant.PLANT
        assert spec.plant.coupling == "free_pressure"
        assert list(spec.n_pump_options) == [3, 2, 1]
        assert spec.n_steps == 9
        assert spec.show_per_pump is False

    def test_prefixes_and_pads_unique(self):
        prefixes = [s.prefix for s in SPECS.values()]
        pads = [s.pad for s in SPECS.values()]
        assert len(set(prefixes)) == len(prefixes)
        assert len(set(pads)) == len(pads)

    def test_legacy_stage_keys_preserved(self):
        # The stage key is f"{prefix}_page_stage" — these exact strings are
        # what live sessions carry; a prefix change would strand them.
        assert f"{SPECS['S'].prefix}_page_stage" == "sp_page_stage"
        assert f"{SPECS['I'].prefix}_page_stage" == "ip_page_stage"
        assert f"{SPECS['M'].prefix}_page_stage" == "mp_page_stage"


class TestEntryPoints:
    @pytest.mark.parametrize(
        "mod,fn_name",
        [
            (s_pad_page, "run_s_pad_page"),
            (i_pad_page, "run_i_pad_page"),
            (m_pad_page, "run_m_pad_page"),
        ],
    )
    def test_entry_exists_and_takes_no_required_args(self, mod, fn_name):
        fn = getattr(mod, fn_name)
        assert callable(fn)
        required = [
            p
            for p in inspect.signature(fn).parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind
            not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        assert not required

    def test_app_routes_pads_through_the_hub(self):
        # app.py routes the single "Pad Optimization" mode via a lazy import
        # of the hub, and the hub lazy-imports each pad's SPEC — pin the exact
        # lines so a rename can't silently break the routing chain.
        src = _APP_PATH.read_text(encoding="utf-8")
        assert "from woffl.gui.pad_hub import run_pad_hub" in src
        hub_src = (_APP_PATH.parent / "pad_hub.py").read_text(encoding="utf-8")
        assert "from woffl.gui.s_pad_page import SPEC" in hub_src
        assert "from woffl.gui.i_pad_page import SPEC" in hub_src
        assert "from woffl.gui.m_pad_page import SPEC" in hub_src


class TestHooks:
    """The pad-specific callables produce the legacy shapes from plain dicts."""

    def test_s_station_metric3(self):
        label, value = SPECS["S"].station_metric3(
            {"n_pumps": 3, "per_pump_bpd": 12345.6}
        )
        assert label == "Per pump (×3)"
        assert value == "12,346 BPD"

    def test_i_station_metric3(self):
        label, value = SPECS["I"].station_metric3({"frontier_cap_bpd": 50000.0})
        assert label == "Frontier budget"
        assert value == "50,000 BPD"

    def test_m_station_metric3(self):
        label, value = SPECS["M"].station_metric3({"n_pumps": 2})
        assert label == "HP pumps online"
        assert value == "2"

    def test_flow_captions(self):
        s_cap = SPECS["S"].flow_caption(3)
        assert "thrust window" in s_cap and "station capacity (3 pumps)" in s_cap
        m_cap = SPECS["M"].flow_caption(3)
        assert "min-flow (recirc) floor" in m_cap
        assert SPECS["I"].flow_caption is None  # I-Pad never had one

    def test_render_hooks_are_callables(self):
        for spec in SPECS.values():
            assert callable(spec.render_plot)
            assert callable(spec.render_station_extras)
            assert callable(spec.render_scenario_warnings)

    def test_delta_caption_templates_format(self):
        # The shared page fills {base}/{h0}/{h1}; a bad placeholder would
        # KeyError at render time on every scenario compare.
        for spec in SPECS.values():
            out = spec.delta_caption.format(base="current", h0=1234.5, h1=2345.6)
            assert "current" in out


class TestFixedHeaderSpec:
    """A new pad = one PadSpec with a FixedHeaderPlant — the R-1 goal."""

    def _spec(self):
        return PadSpec(
            pad="X",
            prefix="xp",
            plant=FixedHeaderPlant(3200.0),
            title="X-Pad Optimization",
            subtitle="Review each X-Pad well, then optimize the pad.",
            configure_caption="Delivered PF pressure is fixed at 3,200 psi.",
        )

    def test_constructs_with_defaults(self):
        spec = self._spec()
        assert spec.plant.coupling == "fixed_curve"
        assert spec.plant.header_at_flow(50000.0) == 3200.0
        # no booster model → no radio, no plots, no pad-specific hooks
        assert spec.n_pump_options is None
        assert spec.render_plot is None
        assert spec.station_metric3 is None
        assert spec.render_station_extras is None
        assert spec.render_scenario_warnings is None
        assert spec.show_per_pump is False
        # defaults the shared render path relies on
        assert spec.n_steps == 11
        assert spec.custom_expander_label
        assert spec.sweep_expander_label
        assert spec.no_active_warning
        spec.delta_caption.format(base="optimized", h0=1.0, h1=2.0)

    def test_spec_is_frozen(self):
        spec = self._spec()
        with pytest.raises(Exception):
            spec.prefix = "yp"

    def test_run_pad_page_accepts_spec(self):
        # rendering needs a Streamlit runtime — just pin the entry signature
        params = list(inspect.signature(run_pad_page).parameters)
        assert params == ["spec"]
