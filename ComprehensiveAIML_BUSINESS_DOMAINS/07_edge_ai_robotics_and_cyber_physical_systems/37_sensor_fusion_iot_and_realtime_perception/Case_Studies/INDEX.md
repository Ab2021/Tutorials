# Module 37 â€” Sensor Fusion, IoT and Real-Time Perception — Case Studies

## 1. Collection metadata

| Field | Value |
|---|---|
| Collection owner | Curriculum maintainer |
| Canonical case-ID format | CASE-M37-02 |
| Evidence policy | Illustrative teaching inputs are not company results; unsourced claims stay OPEN. |
| Editorial state | Draft; independent publication review is not claimed. |
| Acceptance rule | A case row is `authored` only when `Tools/manuscript_lint.py` exits 0 for it. |
| Review trigger | A source body, decision policy, metric definition, or workload assumption changes. |

## 2. Case catalog

| Case ID | Case title / link | Domain | Case type | Decision | Primary pattern | Secondary patterns | Evidence | Stage | Last reviewed | Result / limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| CASE-M37-02 | [Multisensor worker-safety system in a mixed-traffic palletizing cell](./Multisensor_worker_safety_system.md) | Module 37 â€” Sensor Fusion, IoT and Real-Time Perception | hybrid | decision record in the linked case | safety-aware-fusion | human-review, controlled-degradation | illustrative | draft | null | draft (lint open) |
| CASE-37-001 | [Door-Attributed Pallet Exit Verification in a Distribution Warehouse](./Warehouse_asset_tracking.md) | Module 37 â€” Sensor Fusion, IoT and Real-Time Perception | streaming-systems | decision record in the linked case | freshness gate | association, uncertainty, telemetry, lineage | illustrative | draft | null | draft (lint open) |
| PLAN-M37-01 | Warehouse asset tracking | Module 37 â€” Sensor Fusion, IoT and Real-Time Perception | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-M37-02 | Multisensor worker-safety system | Module 37 â€” Sensor Fusion, IoT and Real-Time Perception | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |

## 3. Canonical reusable patterns

| Pattern | Diagnostic signature | Use when | Boundary |
|---|---|---|---|
| evaluation-mismatch | Offline score does not represent the decision | Ranking, forecast, threshold, agent products | The metric captures decision and guardrails |
| data-leakage | Information unavailable at decision time inflates a score | Unexpectedly strong offline result | The feature is genuinely available before the cutoff |
| distribution-shift | Live population differs from training | New market, season, channel, or policy version | The change is sampling noise only |
| human-in-the-loop | Review capacity or override changes the trade-off | High-impact, uncertain, or regulated work | Review is infeasible or demonstrably ineffective |
| model-monitoring | Live signals diverge from the evaluation | Deployed model with usable feedback | No meaningful feedback signal exists |

## 4. Similarity / contrast map

This collection holds 2 case file(s), 0 passing structural lint, and 2 planned commitment(s). Cases require independent review of point-in-time decision logging and explicit baselines; they differ in decision grain, owner, and which guardrail binds. Read the catalog's primary-pattern column to find the nearest contrast: cases sharing a primary pattern differ in constraint, and cases differing in pattern share the decision shape.

## 5. Questions / future-case backlog

Planned commitments, retained without silent deletion:

- `PLAN-M37-01` — Warehouse asset tracking. Not authored; no file exists, so no link is offered.
- `PLAN-M37-02` — Multisensor worker-safety system. Not authored; no file exists, so no link is offered.

## 6. Review / retirement log

| Date | Artifact | Event | Evidence / stage change |
|---|---|---|---|
| 2026-11-10 | [Multisensor worker-safety system in a mixed-traffic palletizing cell](./Multisensor_worker_safety_system.md) | index regenerated from disk | illustrative / draft unchanged |
| 2026-11-10 | [Door-Attributed Pallet Exit Verification in a Distribution Warehouse](./Warehouse_asset_tracking.md) | index regenerated from disk | illustrative / draft unchanged |

## 7. Pre-publish checks

- [x] Every catalog row links a file that exists, or is marked planned without a link.
- [x] Evidence, implementation stage, and editorial status are separate columns.
- [x] Planned commitments are retained with no silent deletion.
- [ ] Independent source review of every authored case.
- [ ] Publication review of every authored case.
