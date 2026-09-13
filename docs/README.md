# Documentation index

The [pad and CFP capacity fixes](pad_cfp_capacity_delivery_2026-09-12.md)
implement the numerical, allocation and workflow repairs from the
[capacity review](pad_cfp_capacity_review_2026-09-12.md). New runs maximize oil
within capacity by default, support required-online wells, qualify pressure
and operating limits, and show CFP bring-online/offset combinations. The
review's frozen pre-fix probes remain historical evidence. The delivery lists
the remaining full comparison study and field qualification work.

Updated September 12, 2026. The [well-fit workflow delivery](well_fit_workflow_delivery_2026-09-12.md)
implements fixed-IPR calibration, explicit common oil-IPR fitting, sensitivity
parity, saved-model identity/replay and bounded I/M/E fixed-plan stress cases.
It records remaining multi-installation and independent qualification work.
The preceding [well-fit, sensitivity and pad-decision review](well_fit_pad_workflow_review_2026-09-12.md)
preserves the original findings and offline sensitivity/Apply reproduction.
The [edit/preview/save workflow](well_input_save_workflow_2026-09-12.md)
adds visible well-save controls, historical previews of sidebar edits, saved
baseline refresh and verified hydration into new optimization configurations.
The [Pump Match Over Time implementation](pump_match_ui_2026-09-12.md)
adds BHP/oil predictions at every usable test, optional chronological validation
and optimizer history links.
The [fixed-IPR investigation](well_match_diagnostic_2026-09-12.md) records the
user's one-IPR constraint, per-test WC/GOR replay, interior-lift solver recovery,
seven-well comparison and next fitting steps.
The [optimization and multi-pump fitting status](optimization_fitting_status_2026-09-12.md)
checks the remaining work against source and specifies the requested production-plot
toggle for modeled BHP and oil. Start with the [end-of-night handoff](session_close_2026-09-11.md)
for the consolidated work, stopping state and ordered next steps. The
[resumed modeling work](jp_model_resume_2026-09-11.md),
[airport pause handoff](session_learnings_2026-09-11.md)
and [model improvement plan](jp_model_improvement_plan_2026-09-11.md) for current
priorities and resumable work. The [September 8 handoff](session_learnings_2026-09-08.md)
retains the underlying decisions and evidence. Agents must
also read [AGENTS.md](../AGENTS.md). Dated measurements describe their recorded
inputs and revision; later implementation notes do not retroactively rerun them.

## Current operating and engineering guides

| Guide | Use it for |
|---|---|
| [Well-fit workflow delivery](well_fit_workflow_delivery_2026-09-12.md) | Current workflow, integrated checks, saved-model identity and remaining qualification work |
| [Sensitivities and common oil IPR](sensitivity_common_ipr_2026-09-12.md) | Scenario/Apply consistency and explicit chronological common-curve fitting |
| [Pad accounting and stress cases](pad_decision_accounting_2026-09-12.md) | Complete modeled coverage, hardware counterfactual and bounded I/M/E plan comparisons |
| [App README](../README.md) / [frontend README](../web/README.md) | Setup, checks and frontend conventions |
| [Optimization user guide](optimization_user_guide.md) | Save well inputs, fit/save installed pumps, run optimization and explore WC |
| [Web architecture](web_port.md) | API, caching, writes and deployment; later sections include port history |
| [Pump calibration scope](pump_calibration_scope_2026-09-08.md) | Installation/model identity, persistence, clean replacements and tests |
| [WC uncertainty GUI](wc_uncertainty_gui_2026-09-08.md) | Sensitivity assumptions, failures, stale-result behavior and browser QA |
| [Model trust and calibration](model_trust_2026-08-10.md) | Evidence layer and fitting; current amendments supersede August Mach assumptions |
| [Model improvement plan](jp_model_improvement_plan_2026-09-11.md) | BHP diagnosis beyond friction, possible physics errors, multiple pump installations/wells, hydraulic alternatives and Pump Match Over Time |
| [Selectable hydraulics](hydraulics_models_2026-09-11.md) | BB, Hagedorn–Brown/Griffith and Shi/Pan; scoped model selection, primary sources, limitations and same-input historical comparison; Tulsa pending |
| [Pad optimization formulation](optimization_redesign_2026-09.md) | Water pricing, plant constraints, both allocators and hardware identity |
| [CFP moves methodology](cfp_moves_methodology.md) | Measured-anchor deltas and response surfaces |
| [Upstream patch register](upstream_sync.md) | Shared-library changes through patch 46 and merge regressions |

## September 7-8 implementation and validation records

September 11 follow-up: [recovered Fable review and fixes](recovered_review_2026-09-11.md)
records the ten confirmed defects, their local fixes and regression coverage.
The [BHP diagnostic](bhp_model_diagnostic_2026-09-11.png) and
[corrected historical preflight](pump_lifetime_preflight_v2_2026-09-11.json) support the
new plan; they are offline investigations, not independent field validation.
The [cross-pump time plot](pump_history_benchmark_2026-09-11.png) and
[benchmark](pump_history_benchmark_2026-09-11.json) forecast later installations
from earlier tests under documented historical-input assumptions.

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
  not all 44 patches. It has not been refreshed into a complete publication draft.
- [Retired crash/resume note](../RESUME_ENG_COMMENT.md) points here; its old
  uncommitted-file and live-save checklist is no longer actionable.
- Workspace-level `../../docs/`, `../../plans/` and `../../.claude/plans/`
  contain older Streamlit designs. The workspace agent pointers direct sessions
  into this repository. Pump vendor README/data under `woffl/jp_data/` remain
  source references, not session guidance.
