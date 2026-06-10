"""Wave 3a smoke checks: imports + profile-cache speedup measurement."""

import time

# 1. touched modules import cleanly
import woffl.assembly.databricks_client  # noqa: F401
import woffl.gui.app  # noqa: F401
import woffl.gui.tabs.batch_run  # noqa: F401
import woffl.gui.tabs.power_fluid_range  # noqa: F401
import woffl.gui.tabs.jp_history_tab  # noqa: F401
import woffl.gui.scotts_tools.pad_watercut  # noqa: F401
import woffl.gui.scotts_tools.header_trend  # noqa: F401
import woffl.gui.scotts_tools.well_sort  # noqa: F401
import woffl.gui.memory_gauge  # noqa: F401

print("imports: OK")

# 2. token cache structure exists and connection helpers are wired
from woffl.assembly import databricks_client as dbc

assert hasattr(dbc, "_oauth_token")
assert hasattr(dbc, "_new_connection")
assert dbc._TOKEN_CACHE == {"token": None, "expires_at": 0.0}
print("databricks_client token/conn plumbing: OK")

# 3. cached WellProfile construction — second call must be near-instant
from woffl.gui.utils import create_well_profile_from_survey

t0 = time.perf_counter()
wp1 = create_well_profile_from_survey("MPB-28", 4100, "Schrader")
t1 = time.perf_counter()
wp2 = create_well_profile_from_survey("MPB-28", 4100, "Schrader")
t2 = time.perf_counter()
cold, warm = t1 - t0, t2 - t1
print(f"profile build cold={cold*1000:.0f} ms, warm={warm*1000:.1f} ms")
assert warm < cold / 5, "cache did not engage"
assert abs(wp1.jetpump_md - wp2.jetpump_md) < 1e-9
# cache_data returns copies — mutating one must not poison the other
wp2.jetpump_md = -1
wp3 = create_well_profile_from_survey("MPB-28", 4100, "Schrader")
assert wp3.jetpump_md == wp1.jetpump_md
print("create_well_profile_from_survey cache: OK")

print("all wave 3a smoke checks passed")
