"""Exact per-well response nodes, shared by pad/CFP and scenario runs.

Only pure physics is cached. Budgets, prices and allocations are recomputed.
Pickled values provide independent mutable objects to every consumer; keys
include every WellConfig field, pressure, pump grid, model source hash and
survey contents. No measured data queries or persisted external cache.
"""
from collections import OrderedDict
from hashlib import sha256
from pathlib import Path
import pickle
from threading import RLock
from time import monotonic

from server import performance, pool

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 64 * 1024 * 1024
MAX_ENTRIES = 1024
TTL_SECONDS = 3600
_LOCK = RLock()
_BATCH_LOCK = RLock()  # single-flight batches; share the global CPU pool
_CACHE = OrderedDict()
_BYTES = 0
_HITS = _MISSES = 0
_MODEL = ""


def start():
    global _MODEL
    # Recomputed for every app lifetime, so a physics deployment invalidates
    # nodes even if the package version has not been bumped.
    digest = sha256()
    for folder in ("pvt", "flow", "geometry", "assembly"):
        for path in sorted((ROOT / "woffl" / folder).rglob("*.py")):
            digest.update(path.relative_to(ROOT).as_posix().encode())
            digest.update(path.read_bytes())
    _MODEL = digest.hexdigest()
    clear()
    from woffl.assembly import compute_runtime
    compute_runtime.configure(batches=run_batches, slot=pool.cpu_slot, timing=performance.measure)


def stop():
    from woffl.assembly import compute_runtime
    compute_runtime.configure()
    clear()


def clear():
    global _BYTES, _HITS, _MISSES
    with _BATCH_LOCK, _LOCK:
        _CACHE.clear()
        _BYTES = _HITS = _MISSES = 0


def _key(well, pressure, nozzles, throats):
    survey = ROOT / "woffl" / "jp_data" / "well_surveys" / f"{well.well_name} Deviation Survey.csv"
    try:
        survey_hash = sha256(survey.read_bytes()).digest()
    except FileNotFoundError:
        survey_hash = None
    except OSError:
        # Let the existing profile loader decide its fallback. An unreadable
        # input cannot safely identify a reusable response node.
        return None
    return sha256(pickle.dumps((_MODEL, vars(well), pressure, tuple(nozzles),
                                tuple(throats), survey_hash), protocol=5)).digest()


def _get(key):
    global _BYTES, _HITS, _MISSES
    if key is None:
        return None
    with _LOCK:
        item = _CACHE.get(key)
        if item is not None:
            expires, blob = item
            if expires > monotonic():
                _CACHE.move_to_end(key)
                _HITS += 1
                return pickle.loads(blob)
            _BYTES -= len(blob)
            del _CACHE[key]
        _MISSES += 1
    return None


def _put(key, value):
    global _BYTES
    if key is None:
        return
    # Cache only returned results, never exceptions. Failed pump rows within
    # a returned batch are deterministic infeasible nodes and may be reused.
    blob = pickle.dumps(value, protocol=5)
    if len(blob) > MAX_BYTES:
        return
    with _LOCK:
        old = _CACHE.pop(key, None)
        if old:
            _BYTES -= len(old[1])
        while _CACHE and (_BYTES + len(blob) > MAX_BYTES or len(_CACHE) >= MAX_ENTRIES):
            _BYTES -= len(_CACHE.popitem(last=False)[1][1])
        _CACHE[key] = (monotonic() + TTL_SECONDS, blob)
        _BYTES += len(blob)


def status():
    with _LOCK:
        return dict(entries=len(_CACHE), bytes=_BYTES, max_bytes=MAX_BYTES,
                    hits=_HITS, misses=_MISSES, model=_MODEL[:12])


def run_batches(wells, pressure, nozzles, throats, progress=None):
    from woffl.assembly.network_optimizer import _simulate_single_well

    with _BATCH_LOCK, performance.measure("physics.batch"):
        out, pending, keys = {}, [], []
        for well in wells:
            key = _key(well, pressure, nozzles, throats)
            cached = _get(key)
            if cached is None:
                keys.append(key)
                pending.append((well, pressure, nozzles, throats))
            else:
                out[well.well_name] = cached
        if progress:
            progress(len(out), len(wells), "cached response nodes")
        values = pool.submit_all(_simulate_single_well, pending)
        if values is None:
            with pool.cpu_slot():
                values = [_simulate_single_well(*args) for args in pending]
        for key, args, value in zip(keys, pending, values):
            _put(key, value)
            out[args[0].well_name] = value
        if progress:
            progress(len(wells), len(wells), "Complete")
        return {well.well_name: out[well.well_name] for well in wells}
