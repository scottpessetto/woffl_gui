"""Optional host integration. Standalone library behavior is unchanged.

The host supplies scheduling/caching; the library never imports the server.
Installed once at app startup, removed at shutdown; worker processes retain
the defaults and therefore cannot submit nested work to their own pool.
"""
from contextlib import nullcontext

# [LIBRARY change -> upstream PR to kwellis/woffl]
batch_runner = None
# jobs -> results: [(well, pressure, nozzles, throats), ...] -> [BatchPump, ...]
# in job order. Lets one call mix wells, headers and pump grids so they share
# the host pool in a single submit instead of one small batch each.
job_runner = None
cpu_slot = nullcontext
measure = lambda name: nullcontext()


def configure(*, batches=None, jobs=None, slot=nullcontext, timing=None):
    global batch_runner, job_runner, cpu_slot, measure
    batch_runner = batches
    job_runner = jobs
    cpu_slot = slot
    measure = timing or (lambda name: nullcontext())
