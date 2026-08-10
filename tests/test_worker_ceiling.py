"""worker_ceiling - the one parallelism budget every ProcessPool obeys.

The unset default is environment-aware: a deployed Databricks App (both
service-principal cred vars present; app.yaml pins the real number anyway)
stays at 1, while a local run gets min(cores, 8) - spawn workers carry the
whole import stack, so the uncapped default OOM'd a 32 GB box. An explicit
WOFFL_MAX_WORKERS always wins, clamped to the physical core count.
"""

import os

import pytest

from woffl.gui.scotts_tools._common import _LOCAL_DEFAULT_CAP, worker_ceiling

CPUS = os.cpu_count() or 1
LOCAL_DEFAULT = min(CPUS, _LOCAL_DEFAULT_CAP)


@pytest.fixture()
def clean_env(monkeypatch):
    """No worker override, no deployment creds - a bare local machine."""
    monkeypatch.delenv("WOFFL_MAX_WORKERS", raising=False)
    monkeypatch.delenv("DATABRICKS_CLIENT_ID", raising=False)
    monkeypatch.delenv("DATABRICKS_CLIENT_SECRET", raising=False)
    return monkeypatch


def test_unset_local_defaults_to_capped_cores(clean_env):
    assert worker_ceiling() == LOCAL_DEFAULT


def test_unset_deployed_defaults_to_one(clean_env):
    clean_env.setenv("DATABRICKS_CLIENT_ID", "svc")
    clean_env.setenv("DATABRICKS_CLIENT_SECRET", "shh")
    assert worker_ceiling() == 1


def test_one_cred_alone_is_not_deployed(clean_env):
    # Mirrors databricks_client._is_deployed: BOTH vars or it is local.
    clean_env.setenv("DATABRICKS_CLIENT_ID", "svc")
    assert worker_ceiling() == LOCAL_DEFAULT


def test_explicit_value_wins_and_clamps_to_cores(clean_env):
    clean_env.setenv("WOFFL_MAX_WORKERS", "2")
    assert worker_ceiling() == min(2, CPUS)
    clean_env.setenv("WOFFL_MAX_WORKERS", str(CPUS + 64))
    assert worker_ceiling() == CPUS


def test_explicit_value_wins_even_when_deployed(clean_env):
    clean_env.setenv("DATABRICKS_CLIENT_ID", "svc")
    clean_env.setenv("DATABRICKS_CLIENT_SECRET", "shh")
    clean_env.setenv("WOFFL_MAX_WORKERS", "2")
    assert worker_ceiling() == min(2, CPUS)


@pytest.mark.parametrize("raw", ["nope", "", "0", "-3"])
def test_garbage_and_nonpositive_pin_to_one(clean_env, raw):
    clean_env.setenv("WOFFL_MAX_WORKERS", raw)
    assert worker_ceiling() == 1
