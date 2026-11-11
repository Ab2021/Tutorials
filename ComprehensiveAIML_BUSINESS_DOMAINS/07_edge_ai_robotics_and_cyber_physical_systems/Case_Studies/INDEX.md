# Part 07 â€” Edge AI, Robotics and Cyber-Physical Systems — Case Studies

## 1. Collection metadata

| Field | Value |
|---|---|
| Collection owner | Curriculum maintainer |
| Canonical case-ID format | CASE-07-12-01 |
| Evidence policy | Illustrative teaching inputs are not company results; unsourced claims stay OPEN. |
| Editorial state | Draft; independent publication review is not claimed. |
| Acceptance rule | A case row is `authored` only when `Tools/manuscript_lint.py` exits 0 for it. |
| Review trigger | A source body, decision policy, metric definition, or workload assumption changes. |

## 2. Case catalog

| Case ID | Case title / link | Domain | Case type | Decision | Primary pattern | Secondary patterns | Evidence | Stage | Last reviewed | Result / limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| CASE-07-12-01 | [Robot near-miss incident response: containment and resume at a blind-corner cluster](./12_Robot_Near_Miss_Incident_Response.md) | Part 07 â€” Edge AI, Robotics and Cyber-Physical Systems | incident | decision record in the linked case | containment ladder under a rare-event evidence gate | kinematic, safety, envelope, stratified, rate, attribution | illustrative | proposed | null | draft (lint open) |
| CASE-2026-0713 | [Fleet rollout readiness review for an autonomous mobile robot software release](./Fleet_rollout_readiness_review.md) | Part 07 â€” Edge AI, Robotics and Cyber-Physical Systems | analytical | decision record in the linked case | release evidence gate | freshness, gate, bounded, authority | illustrative | proposed | null | draft (lint open) |
| CASE-P07-012 | [Validating a learned navigation policy before field trial](./Validate_a_navigation_policy_before_field_trial.md) | Part 07 â€” Edge AI, Robotics and Cyber-Physical Systems | hybrid | decision record in the linked case | transfer validation ladder | scenario, coverage, controlled, recovery | illustrative | draft | null | authored |
| PLAN-P07-01 | Wearable activity classifier | Part 07 â€” Edge AI, Robotics and Cyber-Physical Systems | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-P07-02 | Industrial vibration monitor | Part 07 â€” Edge AI, Robotics and Cyber-Physical Systems | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-P07-03 | Quantise an inspection model | Part 07 â€” Edge AI, Robotics and Cyber-Physical Systems | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-P07-04 | Compare edge runtimes on a Pi-class device | Part 07 â€” Edge AI, Robotics and Cyber-Physical Systems | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-P07-05 | Warehouse asset tracking | Part 07 â€” Edge AI, Robotics and Cyber-Physical Systems | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-P07-06 | Multisensor worker-safety system | Part 07 â€” Edge AI, Robotics and Cyber-Physical Systems | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-P07-07 | Robot arm pick-and-place cell | Part 07 â€” Edge AI, Robotics and Cyber-Physical Systems | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-P07-08 | Mobile robot fleet control design | Part 07 â€” Edge AI, Robotics and Cyber-Physical Systems | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-P07-09 | Autonomous inventory robot | Part 07 â€” Edge AI, Robotics and Cyber-Physical Systems | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-P07-10 | Bin-picking system evaluation | Part 07 â€” Edge AI, Robotics and Cyber-Physical Systems | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-P07-11 | Synthetic warehouse dataset | Part 07 â€” Edge AI, Robotics and Cyber-Physical Systems | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-P07-12 | Validate a navigation policy before field trial | Part 07 â€” Edge AI, Robotics and Cyber-Physical Systems | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-P07-13 | Fleet rollout readiness review | Part 07 â€” Edge AI, Robotics and Cyber-Physical Systems | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-P07-14 | Robot near-miss incident response | Part 07 â€” Edge AI, Robotics and Cyber-Physical Systems | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |

## 3. Canonical reusable patterns

| Pattern | Diagnostic signature | Use when | Boundary |
|---|---|---|---|
| evaluation-mismatch | Offline score does not represent the decision | Ranking, forecast, threshold, agent products | The metric captures decision and guardrails |
| data-leakage | Information unavailable at decision time inflates a score | Unexpectedly strong offline result | The feature is genuinely available before the cutoff |
| distribution-shift | Live population differs from training | New market, season, channel, or policy version | The change is sampling noise only |
| human-in-the-loop | Review capacity or override changes the trade-off | High-impact, uncertain, or regulated work | Review is infeasible or demonstrably ineffective |
| model-monitoring | Live signals diverge from the evaluation | Deployed model with usable feedback | No meaningful feedback signal exists |

## 4. Similarity / contrast map

This collection holds 3 case file(s), 1 passing structural lint, and 14 planned commitment(s). Cases require independent review of point-in-time decision logging and explicit baselines; they differ in decision grain, owner, and which guardrail binds. Read the catalog's primary-pattern column to find the nearest contrast: cases sharing a primary pattern differ in constraint, and cases differing in pattern share the decision shape.

## 5. Questions / future-case backlog

Planned commitments, retained without silent deletion:

- `PLAN-P07-01` — Wearable activity classifier. Not authored; no file exists, so no link is offered.
- `PLAN-P07-02` — Industrial vibration monitor. Not authored; no file exists, so no link is offered.
- `PLAN-P07-03` — Quantise an inspection model. Not authored; no file exists, so no link is offered.
- `PLAN-P07-04` — Compare edge runtimes on a Pi-class device. Not authored; no file exists, so no link is offered.
- `PLAN-P07-05` — Warehouse asset tracking. Not authored; no file exists, so no link is offered.
- `PLAN-P07-06` — Multisensor worker-safety system. Not authored; no file exists, so no link is offered.
- `PLAN-P07-07` — Robot arm pick-and-place cell. Not authored; no file exists, so no link is offered.
- `PLAN-P07-08` — Mobile robot fleet control design. Not authored; no file exists, so no link is offered.
- `PLAN-P07-09` — Autonomous inventory robot. Not authored; no file exists, so no link is offered.
- `PLAN-P07-10` — Bin-picking system evaluation. Not authored; no file exists, so no link is offered.
- `PLAN-P07-11` — Synthetic warehouse dataset. Not authored; no file exists, so no link is offered.
- `PLAN-P07-12` — Validate a navigation policy before field trial. Not authored; no file exists, so no link is offered.
- `PLAN-P07-13` — Fleet rollout readiness review. Not authored; no file exists, so no link is offered.
- `PLAN-P07-14` — Robot near-miss incident response. Not authored; no file exists, so no link is offered.

## 6. Review / retirement log

| Date | Artifact | Event | Evidence / stage change |
|---|---|---|---|
| 2026-11-10 | [Robot near-miss incident response: containment and resume at a blind-corner cluster](./12_Robot_Near_Miss_Incident_Response.md) | index regenerated from disk | illustrative / proposed unchanged |
| 2026-11-10 | [Fleet rollout readiness review for an autonomous mobile robot software release](./Fleet_rollout_readiness_review.md) | index regenerated from disk | illustrative / proposed unchanged |
| 2026-11-10 | [Validating a learned navigation policy before field trial](./Validate_a_navigation_policy_before_field_trial.md) | index regenerated from disk | illustrative / draft unchanged |

## 7. Pre-publish checks

- [x] Every catalog row links a file that exists, or is marked planned without a link.
- [x] Evidence, implementation stage, and editorial status are separate columns.
- [x] Planned commitments are retained with no silent deletion.
- [ ] Independent source review of every authored case.
- [ ] Publication review of every authored case.
