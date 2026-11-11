# ML System Design Production — Case Studies

## 1. Collection metadata

| Field | Value |
|---|---|
| Collection owner | Curriculum maintainer |
| Canonical case-ID format | CASE-03-01-005 |
| Evidence policy | Illustrative teaching inputs are not company results; unsourced claims stay OPEN. |
| Editorial state | Draft; independent publication review is not claimed. |
| Acceptance rule | A case row is `authored` only when `Tools/manuscript_lint.py` exits 0 for it. |
| Review trigger | A source body, decision policy, metric definition, or workload assumption changes. |

## 2. Case catalog

| Case ID | Case title / link | Domain | Case type | Decision | Primary pattern | Secondary patterns | Evidence | Stage | Last reviewed | Result / limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| CASE-03-01-005 | [Triage a shifted delivery-risk model without confusing drift with harm](./01_model_monitoring_drift_incident.md) | ML System Design Production | incident | decision record in the linked case | model-monitoring | distribution-shift, label-delay | illustrative | proposed | null | draft (lint open) |
| CASE-03-01-001 | [Fund a fraud ranking platform when investigator capacity, not model accuracy, binds](./02_fraud_platform_capacity_decision.md) | ML System Design Production | predictive-ml | decision record in the linked case | evaluation-mismatch | cost-latency-tradeoff, label-delay | illustrative | proposed | null | authored |
| PLAN-03-01-01 | Fraud model platform | ML System Design Production | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-03-01-02 | Personalisation service | ML System Design Production | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-03-01-03 | Forecasting platform | ML System Design Production | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-03-01-04 | Content moderation system | ML System Design Production | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-03-01-05 | ML incident review | ML System Design Production | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-2026-IR026 | Select data structures from operation mix and failure semantics | ML System Design Production | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |

## 3. Canonical reusable patterns

| Pattern | Diagnostic signature | Use when | Boundary |
|---|---|---|---|
| evaluation-mismatch | Offline score does not represent the decision | Ranking, forecast, threshold, agent products | The metric captures decision and guardrails |
| data-leakage | Information unavailable at decision time inflates a score | Unexpectedly strong offline result | The feature is genuinely available before the cutoff |
| distribution-shift | Live population differs from training | New market, season, channel, or policy version | The change is sampling noise only |
| human-in-the-loop | Review capacity or override changes the trade-off | High-impact, uncertain, or regulated work | Review is infeasible or demonstrably ineffective |
| model-monitoring | Live signals diverge from the evaluation | Deployed model with usable feedback | No meaningful feedback signal exists |

## 4. Similarity / contrast map

This collection holds 2 case file(s), 1 passing structural lint, and 6 planned commitment(s). Cases require independent review of point-in-time decision logging and explicit baselines; they differ in decision grain, owner, and which guardrail binds. Read the catalog's primary-pattern column to find the nearest contrast: cases sharing a primary pattern differ in constraint, and cases differing in pattern share the decision shape.

## 5. Questions / future-case backlog

Planned commitments, retained without silent deletion:

- `PLAN-03-01-01` — Fraud model platform. Not authored; no file exists, so no link is offered.
- `PLAN-03-01-02` — Personalisation service. Not authored; no file exists, so no link is offered.
- `PLAN-03-01-03` — Forecasting platform. Not authored; no file exists, so no link is offered.
- `PLAN-03-01-04` — Content moderation system. Not authored; no file exists, so no link is offered.
- `PLAN-03-01-05` — ML incident review. Not authored; no file exists, so no link is offered.
- `PLAN-2026-IR026` — Select data structures from operation mix and failure semantics. Not authored; no file exists, so no link is offered.

## 6. Review / retirement log

| Date | Artifact | Event | Evidence / stage change |
|---|---|---|---|
| 2026-11-10 | [Triage a shifted delivery-risk model without confusing drift with harm](./01_model_monitoring_drift_incident.md) | index regenerated from disk | illustrative / proposed unchanged |
| 2026-11-10 | [Fund a fraud ranking platform when investigator capacity, not model accuracy, binds](./02_fraud_platform_capacity_decision.md) | index regenerated from disk | illustrative / proposed unchanged |

## 7. Pre-publish checks

- [x] Every catalog row links a file that exists, or is marked planned without a link.
- [x] Evidence, implementation stage, and editorial status are separate columns.
- [x] Planned commitments are retained with no silent deletion.
- [ ] Independent source review of every authored case.
- [ ] Publication review of every authored case.
