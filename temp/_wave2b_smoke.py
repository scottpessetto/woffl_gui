"""Wave 2b smoke checks: imports + template/NA sentinel + viz guard."""

import pandas as pd

# 1. every touched module imports cleanly (catches syntax + circular imports)
import woffl.gui.app  # noqa: F401
import woffl.gui.sidebar  # noqa: F401
import woffl.gui.utils  # noqa: F401
import woffl.gui.well_database_page  # noqa: F401
import woffl.gui.optimization_viz  # noqa: F401
import woffl.gui.workflow_page  # noqa: F401
import woffl.gui.workflow_steps.step1_select_wells  # noqa: F401
import woffl.gui.workflow_steps.step2_review_ipr  # noqa: F401
import woffl.gui.workflow_steps.step2_5_precalibrate  # noqa: F401
import woffl.gui.workflow_steps.step3_configure_optimize  # noqa: F401
import woffl.gui.workflow_steps.step4_results  # noqa: F401
import woffl.gui.tabs.batch_run  # noqa: F401
import woffl.gui.tabs.pressure_profile  # noqa: F401
import woffl.gui.tabs.jetpump_solver  # noqa: F401
import woffl.gui.scotts_tools.header_impact  # noqa: F401
import woffl.gui.scotts_tools.pf_scenario  # noqa: F401
import woffl.gui.scotts_tools.well_sort  # noqa: F401

print("imports: OK")

# 2. template export: live chars take precedence; missing wells get pd.NA
#    (not "") so load_wells_from_dataframe's notna guard skips them
from woffl.assembly.ipr_analyzer import export_optimization_template

coeffs = pd.DataFrame([
    {"Well": "MPX-1", "ResP": 1500, "qwf": 800, "pwf": 600, "form_wc": 0.5,
     "num_tests": 4, "most_recent_date": pd.Timestamp("2026-06-01")},
    {"Well": "MPX-2", "ResP": 1600, "qwf": 700, "pwf": 650, "form_wc": 0.6,
     "num_tests": 3, "most_recent_date": pd.Timestamp("2026-06-01")},
])
chars = pd.DataFrame([
    {"Well": "MPX-1", "is_sch": True, "JP_TVD": 4321.0, "JP_MD": 5000.0,
     "out_dia": 4.5, "thick": 0.5, "form_temp": 80},
])
tmpl = export_optimization_template(coeffs, jp_chars=chars)
r1 = tmpl[tmpl["Well"] == "MPX-1"].iloc[0]
r2 = tmpl[tmpl["Well"] == "MPX-2"].iloc[0]
assert r1["JP_TVD"] == 4321.0                      # live chars used
assert pd.isna(r2["JP_TVD"])                       # missing -> NA, not ""
print("export_optimization_template (live chars + NA sentinel): OK")

# 3. run_jetpump_solver quiet param exists; ThroatEntryNoSolution re-raised
import inspect

from woffl.gui.utils import run_jetpump_solver

assert "quiet" in inspect.signature(run_jetpump_solver).parameters
print("run_jetpump_solver quiet param: OK")

# 4. load_well_characteristics exposes .clear for cache-refresh call sites
from woffl.gui.utils import load_well_characteristics

assert callable(getattr(load_well_characteristics, "clear", None))
print("load_well_characteristics.clear: OK")

print("all wave 2b smoke checks passed")
