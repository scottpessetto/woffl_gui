# Reproducing the pad and CFP capacity review

Read the [review](../pad_cfp_capacity_review_2026-09-12.md) for interpretation.
These probes reproduce the reviewed behavior; they are diagnostic artifacts,
not regression tests asserting that the behavior should be preserved.

Run from the repository directory with its existing virtual environment:

```powershell
$env:PYTHONPATH = '.'
$env:WOFFL_MAX_WORKERS = '1'
.\venv\Scripts\python.exe docs/pad_cfp_capacity_review_2026-09-12/pad_probe.py
.\venv\Scripts\python.exe docs/pad_cfp_capacity_review_2026-09-12/cfp_probes.py
.\venv\Scripts\python.exe docs/pad_cfp_capacity_review_2026-09-12/allocation_probe.py
.\venv\Scripts\python.exe docs/pad_cfp_capacity_review_2026-09-12/workflow_probe.py
```

Each script writes the corresponding JSON beside itself. Preserve the recorded
outputs before rerunning against changed engines. All calls stay local:

| Probe | Real components | Synthetic or substituted components |
|---|---|---|
| `pad_probe.py` | Plant curves, selected-pump settling, MILP, sweep and result metadata | Batch well responses; no well physics or warehouse calls |
| `cfp_probes.py` | Anchoring, interpolation, settling, frontier, move and pair ranking | Small response surfaces; 1,000 seeded three-well cases enumerate all 64 choices using the same settling model |
| `allocation_probe.py` | MILP and CP-SAT adapters, choke trimming, price derivation and reconciliation | Candidate tables; 14 named cases, 120 seeded cases checked against an independent Decimal enumeration; injected solver failures |
| `workflow_probe.py` | Request schemas and configuration-building control flow | Local well list/context and lightweight configuration factory; no database/geometry I/O |

All synthetic rates use BOPD for oil and BPD for standard-condition water.
CFP `water` is modeled machine water from the included wells, not measured
total CFP throughput. Random seeds are fixed at `20260912`. Arbitrary CFP
option names such as `small` and `large` identify table entries and do not
assert an ordering of nozzle sizes or performance. Infeasible nonfinite CFP
scores are serialized as JSON null.

The model source files and installed library versions are recorded in
`source_manifest.json`. This review added only documentation and diagnostic
artifacts. It did not alter the runtime algorithms, run a field experiment,
rerun the whole repository test suite, or deploy the application.
