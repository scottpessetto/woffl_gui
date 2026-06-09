"""Wave 2a smoke checks: imports + functional checks of the new logic."""

import numpy as np
import pandas as pd

# 1. all touched GUI modules import cleanly
import woffl.gui.pdf_export  # noqa: F401
import woffl.gui.scotts_tools.jp_calibration  # noqa: F401
import woffl.gui.scotts_tools.jp_fric_trend  # noqa: F401
import woffl.gui.scotts_tools.jp_washout  # noqa: F401
import woffl.gui.scotts_tools.header_impact  # noqa: F401

print("imports: OK")

# 2. get_pump_at_date — install intervals incl. Date Pulled
from woffl.assembly.jp_history import get_current_pump, get_pump_at_date

hist = pd.DataFrame({
    "Well Name": ["MPB-28", "MPB-28", "MPB-28"],
    "Nozzle Number": [11, 12, 13],
    "Throat Ratio": ["C", "B", "C"],
    "Tubing Diameter": [4.5, 4.5, 4.5],
    "Date Set": pd.to_datetime(["2024-01-10", "2025-03-01", "2026-02-15"]),
    "Date Pulled": pd.to_datetime(["2025-03-01", "2026-02-15", pd.NaT]),
})
assert get_pump_at_date(hist, "MPB-28", "2024-06-01")["nozzle_no"] == "11"
assert get_pump_at_date(hist, "MPB-28", "2025-06-01")["nozzle_no"] == "12"
assert get_pump_at_date(hist, "MPB-28", "2026-06-01")["nozzle_no"] == "13"
assert get_pump_at_date(hist, "MPB-28", "2023-06-01") is None
assert get_current_pump(hist, "MPB-28")["nozzle_no"] == "13"
print("get_pump_at_date: OK")

# 3. build_well_config — NaN chars fall back to defaults; vogel overrides win
from woffl.gui.scotts_tools._common import build_well_config

chars = {
    "MPX-1": {
        "is_sch": True, "JP_TVD": 4100.0, "JP_MD": float("nan"),
        "res_pres": float("nan"), "form_temp": float("nan"),
        "out_dia": 4.5, "thick": 0.5, "oil_api": float("nan"),
    }
}
wc = build_well_config("MPX-1", chars)
assert wc.res_pres == 1800.0, wc.res_pres          # NaN -> default, not nan
assert wc.form_temp == 75.0
assert wc.jpump_md == 4100.0                        # NaN JP_MD -> JP_TVD
assert wc.oil_api is None                           # NaN PVT -> preset path
vog = {"ResP": 1450.0, "form_wc": 0.8, "fgor": 600.0, "qwf": 1200.0, "pwf": 900.0}
wc2 = build_well_config("MPX-1", chars, vog)
assert wc2.res_pres == 1450.0 and wc2.form_wc == 0.8
print("build_well_config NaN-coalesce + vogel: OK")

# 4. pad_updown_lever — NaN sweep point no longer creates a phantom pad loss
from woffl.gui.scotts_tools.header_engine import pad_updown_lever

curve = {
    "deltas": [-100, 0, 100],
    "wells": {
        # responsive well: loses 10 down, gains 10 up
        "W1": {"pad": "G", "oil": [90.0, 100.0, 110.0]},
        # low-WHP well: NaN at -100 (below sweep floor) — must NOT dump its
        # 500 BOPD into the down-side delta
        "W2": {"pad": "G", "oil": [np.nan, 500.0, 505.0]},
    },
}
down, up = pad_updown_lever(curve, "G")
assert down == -10.0, down   # was -510 with the old mixed-set math
assert up == 15.0, up
print("pad_updown_lever: OK")

print("all wave 2a smoke checks passed")
