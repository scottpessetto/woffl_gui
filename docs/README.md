# Documentation index

Updated September 8, 2026. Start with the [session handoff](session_learnings_2026-09-08.md)
for current decisions, evidence, test baseline and remaining work. Agents must
also read [AGENTS.md](../AGENTS.md). Dated measurements describe their recorded
inputs and revision; later implementation notes do not retroactively rerun them.

## Current operating and engineering guides

| Guide | Use it for |
|---|---|
| [App README](../README.md) / [frontend README](../web/README.md) | Setup, checks and frontend conventions |
| [Optimization user guide](optimization_user_guide.md) | Save well inputs, fit/save installed pumps, run optimization and explore WC |
| [Web architecture](web_port.md) | API, caching, writes and deployment; later sections include port history |
| [Pump calibration scope](pump_calibration_scope_2026-09-08.md) | Installation/model identity, persistence, clean replacements and tests |
| [WC uncertainty GUI](wc_uncertainty_gui_2026-09-08.md) | Sensitivity assumptions, failures, stale-result behavior and browser QA |
| [Model trust and calibration](model_trust_2026-08-10.md) | Evidence layer and fitting; current amendments supersede August Mach assumptions |
| [Pad optimization formulation](optimization_redesign_2026-09.md) | Water pricing, plant constraints, both allocators and hardware identity |
| [CFP moves methodology](cfp_moves_methodology.md) | Measured-anchor deltas and response surfaces |
| [Upstream patch register](upstream_sync.md) | Shared-library changes through patch 43 and merge regressions |

## September 7-8 implementation and validation records

| Record | Scope and interpretation |
|---|---|
| [Code review](code_review_2026-09-07.md) / [fixes](code_review_2026-09-07_fixes.md) | Findings and first fixes; later energy/fluid/scope records supersede earlier behavior |
| [Optimization review](optimization_review_2026-09-07.md) | Workflow findings and test gaps at review time |
| [Medium performance](medium_performance_2026-09-07.md) | Scheduling/cache design and original local benchmark; retain Medium |
| [Original physics qualification](physics_qualification_2026-09-07.md) | Historical critical-Mach failures, subsequently fixed |
| [Critical-Mach options](critical_mach_options_2026-09-08.md) | Pre-fix investigation; shared energy was selected and implemented |
| [Entry-energy v1 implementation](entry_energy_implementation_2026-09-08.md) / [qualification](entry_energy_qualification_2026-09-08.md) | Shared unscaled balance, first reachable limit and original v1 deltas |
| [Fluid v2 implementation and holdouts](fluid_followup_2026-09-08.md) / [qualification](fluid_followup_qualification_2026-09-08.md) | Current fluid correlations, independent checks and three-well frozen-training holdouts |
| [Fleet actuality](fleet_actuality_2026-09-08.md) | 35-well retrospective audit, frozen before scoped-fit hydration |
| [Fleet WC study](fleet_watercut_sensitivity_2026-09-08.md) | Two gas assumptions on frozen audit inputs; distinct from fixed-GOR GUI |

Adjacent JSON/CSV/HTML/PNG files are the corresponding recorded artifacts.
Preserve them on reruns by choosing a new output path. Local `build/` snapshots
and QA logs are ignored and may not exist in a fresh clone.

## Historical reviews, plans and external asks

These preserve earlier reasoning and measurements. They are not an executable
backlog or authority to restore deleted modules, issue DDL or send a message.

- Reviews: [July 1](code_review_2026-07-01.md), [July 6 status](review_status_2026-07-06.md),
  [September 1](code_review_2026-09-01.md).
- Earlier designs: [IPR review](ipr_model_review.md),
  [S-Pad joint automatch](s_pad_joint_automatch.md),
  [water-pump mode](water_pump_mode_plan.md),
  [sensitivity persistence history](sensitivity_match_persistence.md).
- External asks: [property-history/schema correspondence](prop_hist_asks.md).
  Scoped pump calibration uses the existing comment ledger; no new schema was
  required. Historical grant requests are not a current access audit.
- [Upstream PR draft](upstream_pr_draft.md) covers the early solver fallbacks,
  not all 43 patches. It has not been refreshed into a complete publication draft.
- [Retired crash/resume note](../RESUME_ENG_COMMENT.md) points here; its old
  uncommitted-file and live-save checklist is no longer actionable.
- Workspace-level `../../docs/`, `../../plans/` and `../../.claude/plans/`
  contain older Streamlit designs. The workspace agent pointers direct sessions
  into this repository. Pump vendor README/data under `woffl/jp_data/` remain
  source references, not session guidance.
