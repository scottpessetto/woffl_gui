"""Offline workflow probes; synthetic inputs, no well simulation or database I/O."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from server import schemas
from server.services import optimizer_runs as runs


def main():
    universe = {"wells": [{"name": "EXISTING", "pad": "M"},
                          {"name": "DONOR", "pad": "B"}]}
    notes, ledger = [], {}
    req = schemas.OptimizeRunRequest(kind="pad", pad="M", future=[
        schemas.FutureWellSpec(name="EXISTING", match="DONOR", pad="M")])
    with patch.object(runs.wells_svc, "list_wells", return_value=universe), \
         patch.object(runs.wells_svc, "well_context", side_effect=lambda name, *_: {"seeds": {"source": name}}), \
         patch.object(runs, "_config_from_seeds", side_effect=lambda name, pad, seeds: SimpleNamespace(well_name=name, pad=pad, source=seeds["source"])):
        configs = runs._build_configs(["M"], set(), req.future, notes, coverage=ledger)
    duplicate_req = schemas.OptimizeRunRequest(kind="cfp", future=[
        schemas.FutureWellSpec(name="NEW", match="DONOR", pad="B"),
        schemas.FutureWellSpec(name="NEW", match="DONOR", pad="G")])
    fields = set(schemas.OptimizeRunRequest.model_fields)
    out = {
        "scope": "Synthetic request/hydration probes only; database and well geometry construction replaced with local fakes.",
        "existing_name_collision": {
            "schema_accepted": True,
            "hydrated_configs": [vars(c) for c in configs],
            "coverage": ledger,
            "notes": notes,
        },
        "cross_pad_duplicate_accepted": [f.model_dump() for f in duplicate_req.future],
        "new_well_supported_fields": list(schemas.FutureWellSpec.model_fields),
        "decision_constraints_exposed": {k: k in fields for k in (
            "must_run", "keep_online", "allow_shut_in", "max_changes", "locked_choices", "new_well_pump")},
    }
    assert len(configs) == 2 and configs[0].well_name == configs[1].well_name
    assert ledger["EXISTING"]["role"] == "future"
    path = Path(__file__).with_suffix(".json")
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
