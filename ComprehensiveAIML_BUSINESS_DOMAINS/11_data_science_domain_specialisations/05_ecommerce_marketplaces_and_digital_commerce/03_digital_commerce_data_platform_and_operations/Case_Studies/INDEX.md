# Digital Commerce Data Platform and Operations — Case Studies

## 1. Collection metadata

| Field | Value |
|---|---|
| Collection owner | Curriculum maintainer |
| Canonical case-ID format | CASE-05-03-001 |
| Evidence policy | Illustrative teaching inputs are not company results; unsourced claims stay OPEN. |
| Editorial state | Draft; independent publication review is not claimed. |
| Acceptance rule | A case row is `authored` only when `Tools/manuscript_lint.py` exits 0 for it. |
| Review trigger | A source body, decision policy, metric definition, or workload assumption changes. |

## 2. Case catalog

| Case ID | Case title / link | Domain | Case type | Decision | Primary pattern | Secondary patterns | Evidence | Stage | Last reviewed | Result / limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| CASE-05-03-001 | [Inventory freshness incident response](./01_Inventory_Freshness_Incident.md) | Digital Commerce Data Platform and Operations | incident | decision record in the linked case | evaluation-mismatch | human-in-the-loop, rollout-regression | illustrative | proposed | null | draft (lint open) |
| CASE-05-03-002 | [Experiment assignment failure](./02_Experiment_Assignment_Failure.md) | Digital Commerce Data Platform and Operations | incident | decision record in the linked case | evaluation-mismatch | human-in-the-loop, rollout-regression | illustrative | proposed | null | draft (lint open) |
| CASE-05-03-003 | [Checkout latency regression](./03_Checkout_Latency_Regression.md) | Digital Commerce Data Platform and Operations | incident | decision record in the linked case | evaluation-mismatch | human-in-the-loop, rollout-regression | illustrative | proposed | null | draft (lint open) |
| CASE-05-03-004 | [Order-status agent decision boundary](./04_Order_Status_Agent.md) | Digital Commerce Data Platform and Operations | llm-agentic | decision record in the linked case | evaluation-mismatch | human-in-the-loop, rollout-regression | illustrative | proposed | null | draft (lint open) |
| CASE-05-03-005 | [Data-contract rollout](./05_Data_Contract_Rollout.md) | Digital Commerce Data Platform and Operations | streaming-systems | decision record in the linked case | evaluation-mismatch | human-in-the-loop, rollout-regression | illustrative | proposed | null | draft (lint open) |
| CASE-05-03-006 | [Cache ring migration recovery](./06_Cache_Ring_Migration.md) | Digital Commerce Data Platform and Operations | streaming-systems | decision record in the linked case | evaluation-mismatch | human-in-the-loop, rollout-regression | illustrative | proposed | null | draft (lint open) |
| PLAN-05-03-01 | Inventory freshness incident | Digital Commerce Data Platform and Operations | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-05-03-02 | Experiment-assignment failure | Digital Commerce Data Platform and Operations | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-05-03-03 | Checkout latency regression | Digital Commerce Data Platform and Operations | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-05-03-04 | Order-status agent | Digital Commerce Data Platform and Operations | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-05-03-05 | Data-contract rollout | Digital Commerce Data Platform and Operations | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-2026-IR033 | Rebalance caches without overwhelming the authoritative store | Digital Commerce Data Platform and Operations | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |

## 3. Canonical reusable patterns

| Pattern | Diagnostic signature | Use when | Boundary |
|---|---|---|---|
| evaluation-mismatch | Offline score does not represent the decision | Ranking, forecast, threshold, agent products | The metric captures decision and guardrails |
| data-leakage | Information unavailable at decision time inflates a score | Unexpectedly strong offline result | The feature is genuinely available before the cutoff |
| distribution-shift | Live population differs from training | New market, season, channel, or policy version | The change is sampling noise only |
| human-in-the-loop | Review capacity or override changes the trade-off | High-impact, uncertain, or regulated work | Review is infeasible or demonstrably ineffective |
| model-monitoring | Live signals diverge from the evaluation | Deployed model with usable feedback | No meaningful feedback signal exists |

## 4. Similarity / contrast map

This collection holds 6 authored case(s) and 6 planned commitment(s). Authored cases share point-in-time decision logging and an explicit baseline; they differ in decision grain, owner, and which guardrail binds. Read the catalog's primary-pattern column to find the nearest contrast: cases sharing a primary pattern differ in constraint, and cases differing in pattern share the decision shape.

## 5. Questions / future-case backlog

Planned commitments, retained without silent deletion:

- `PLAN-05-03-01` — Inventory freshness incident. Not authored; no file exists, so no link is offered.
- `PLAN-05-03-02` — Experiment-assignment failure. Not authored; no file exists, so no link is offered.
- `PLAN-05-03-03` — Checkout latency regression. Not authored; no file exists, so no link is offered.
- `PLAN-05-03-04` — Order-status agent. Not authored; no file exists, so no link is offered.
- `PLAN-05-03-05` — Data-contract rollout. Not authored; no file exists, so no link is offered.
- `PLAN-2026-IR033` — Rebalance caches without overwhelming the authoritative store. Not authored; no file exists, so no link is offered.

## 6. Review / retirement log

| Date | Artifact | Event | Evidence / stage change |
|---|---|---|---|
| 2026-11-10 | [Inventory freshness incident response](./01_Inventory_Freshness_Incident.md) | index regenerated from disk | illustrative / proposed unchanged |
| 2026-11-10 | [Experiment assignment failure](./02_Experiment_Assignment_Failure.md) | index regenerated from disk | illustrative / proposed unchanged |
| 2026-11-10 | [Checkout latency regression](./03_Checkout_Latency_Regression.md) | index regenerated from disk | illustrative / proposed unchanged |
| 2026-11-10 | [Order-status agent decision boundary](./04_Order_Status_Agent.md) | index regenerated from disk | illustrative / proposed unchanged |
| 2026-11-10 | [Data-contract rollout](./05_Data_Contract_Rollout.md) | index regenerated from disk | illustrative / proposed unchanged |
| 2026-11-10 | [Cache ring migration recovery](./06_Cache_Ring_Migration.md) | index regenerated from disk | illustrative / proposed unchanged |

## 7. Pre-publish checks

- [x] Every catalog row links a file that exists, or is marked planned without a link.
- [x] Evidence, implementation stage, and editorial status are separate columns.
- [x] Planned commitments are retained with no silent deletion.
- [ ] Independent source review of every authored case.
- [ ] Publication review of every authored case.
