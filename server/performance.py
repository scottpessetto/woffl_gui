"""Bounded, process-local timings; no warehouse polling or external service."""
from collections import defaultdict, deque
from contextlib import contextmanager
from threading import Lock
from time import perf_counter

_LOCK = Lock()
_SAMPLES = defaultdict(lambda: deque(maxlen=256))
_COUNTS = defaultdict(int)


def record(name, seconds):
    with _LOCK:
        if name not in _SAMPLES and len(_SAMPLES) >= 100:
            name = "other"
        _SAMPLES[name].append(seconds * 1000)
        _COUNTS[name] += 1


@contextmanager
def measure(name):
    started = perf_counter()
    try:
        yield
    finally:
        record(name, perf_counter() - started)


def snapshot():
    with _LOCK:
        out = {}
        for name, samples in _SAMPLES.items():
            vals = sorted(samples)
            out[name] = {
                "count": _COUNTS[name], "samples": len(vals),
                "p50_ms": round(vals[(len(vals)-1)//2], 2),
                "p95_ms": round(vals[int((len(vals)-1)*.95)], 2),
                "max_ms": round(vals[-1], 2),
            }
        return out
