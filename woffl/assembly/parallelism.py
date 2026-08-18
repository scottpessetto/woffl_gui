"""The one parallelism budget every ProcessPool in the tree obeys.

This lived in ``woffl/gui/scotts_tools/_common.py``, which does
``import streamlit as st`` at module scope. The FastAPI server imports it
lazily from four places (sensitivity, cfp_moves, cfp_optimize, pad_optimize)
purely to read one env var, and paid 1.31 s of Streamlit import to do it -
on whichever request happened to be first. Nothing here imports Streamlit.

``_common`` re-exports ``worker_ceiling``/``_LOCAL_DEFAULT_CAP`` so the
existing call sites keep working unchanged.
"""

from __future__ import annotations

import os

# Unset-default cap for LOCAL runs. Each spawned worker re-imports the full
# app stack (pandas/plotly and, historically, streamlit) on Windows spawn;
# 14 workers OOM'd a 32 GB workstation on a real M-Pad batch, 8 ran clean
# with the same wall-clock benefit. An explicit WOFFL_MAX_WORKERS may exceed
# this (still clamped to the core count).
_LOCAL_DEFAULT_CAP = 8


def usable_cpus() -> int:
    """CPUs this process may actually run on.

    ``os.cpu_count()`` reports the HOST's cores, which inside a
    cgroup-limited container (a Databricks App) is not the quota - so it is
    not the safety net a clamp against it looks like. ``sched_getaffinity``
    is the real answer where the platform has it (Linux); Windows has no
    equivalent, so it falls back.
    """
    try:
        return max(1, len(os.sched_getaffinity(0)))  # type: ignore[attr-defined]
    except (AttributeError, OSError):  # Windows, or a restricted platform
        return max(1, os.cpu_count() or 1)


def worker_ceiling() -> int:
    """Max ProcessPool workers permitted in the current environment.

    Reads the ``WOFFL_MAX_WORKERS`` env var, parses defensively, and clamps
    by the usable CPU count. Returns at least 1.

    UNSET defaults are environment-aware: a deployed Databricks App (both
    service-principal cred vars present - the same check as
    ``databricks_client._is_deployed`` and ``server.config.is_deployed``)
    stays at 1, because the compute tier is tiny and app.yaml pins the real
    number anyway; a LOCAL run gets ``min(cpus, _LOCAL_DEFAULT_CAP)`` -
    spawn workers carry the whole import stack, so an uncapped many-core
    default exhausts memory before it wins wall-clock. Tabs that expose a
    worker slider use this as the upper bound.
    """
    raw = os.environ.get("WOFFL_MAX_WORKERS")
    cpus = usable_cpus()
    if raw is None:
        deployed = bool(
            os.environ.get("DATABRICKS_CLIENT_ID")
            and os.environ.get("DATABRICKS_CLIENT_SECRET")
        )
        return 1 if deployed else max(1, min(cpus, _LOCAL_DEFAULT_CAP))
    try:
        env_max = max(1, int(raw))
    except (TypeError, ValueError):
        env_max = 1
    return max(1, min(env_max, cpus))
