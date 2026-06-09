"""Wave 1 smoke checks: imports, exception taxonomy, low-GOR recovery path."""

# 1. GUI modules import cleanly (catches circular imports from the new helpers)
import woffl.gui.sidebar  # noqa: F401
import woffl.gui.tabs.jetpump_solver  # noqa: F401
import woffl.gui.scotts_tools.pf_scenario  # noqa: F401
import woffl.gui.utils  # noqa: F401

print("GUI imports: OK")

# 2. exception taxonomy — existing handlers must still catch the typed errors
from woffl.flow.errors import ConvergenceError, JetPumpError, ThroatEntryNoSolution

assert issubclass(ConvergenceError, ValueError)
assert issubclass(JetPumpError, ValueError)
assert issubclass(ThroatEntryNoSolution, ValueError)
assert issubclass(ThroatEntryNoSolution, IndexError)
print("exception taxonomy: OK")

# 3. clamp_seed sanity
from woffl.gui.sidebar import clamp_seed

assert clamp_seed("qwf", 4) == 10
assert clamp_seed("ppf_surf", 4300) == 4300  # representable now
assert clamp_seed("pwf", 80) == 100
print("clamp_seed: OK")

# 4. low-GOR solve must fail with something the GUI recovery catches
import woffl.assembly.solopump as so
from woffl.flow.inflow import InFlow
from woffl.geometry.jetpump import JetPump
from woffl.geometry.pipe import Pipe, PipeInPipe
from woffl.geometry.wellprofile import WellProfile
from woffl.pvt.blackoil import BlackOil
from woffl.pvt.formgas import FormGas
from woffl.pvt.formwat import FormWater
from woffl.pvt.resmix import ResMix

wellbore = PipeInPipe(inn_pipe=Pipe(4.5, 0.5), out_pipe=Pipe(6.875, 0.5))
ipr = InFlow(qwf=246, pwf=1049, pres=1400)
res = ResMix(0.894, 25, BlackOil.schrader(), FormWater.schrader(), FormGas.schrader())
try:
    out = so.jetpump_solver(
        210, 80, 3168, JetPump("12", "B", 0.01, 0.03, 0.3, 0.4),
        wellbore, WellProfile.schrader(), ipr, res, FormWater.schrader(), "reverse",
    )
    print(f"low-GOR (25 scf/bbl) solved: qoil={out[2]:.1f} sonic={out[1]}")
except (IndexError, ValueError) as e:
    caught_by = []
    if isinstance(e, IndexError):
        caught_by.append("except IndexError (GUI GOR auto-recovery)")
    if isinstance(e, ValueError):
        caught_by.append("except ValueError (run_jetpump_solver)")
    print(f"low-GOR raised {type(e).__name__}: {e}")
    print("  caught by:", " AND ".join(caught_by))
