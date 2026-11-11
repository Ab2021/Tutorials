# Part 03 — Machine Learning, Optimisation and Intelligent Retrieval — Case Studies

## 1. Collection metadata

| Field | Value |
|---|---|
| Collection owner | Curriculum maintainer |
| Canonical case-ID format | CASE-03-16-001 |
| Evidence policy | Illustrative teaching inputs are not company results; unsourced claims stay OPEN. |
| Editorial state | Draft; independent publication review is not claimed. |
| Acceptance rule | A case row is `authored` only when `Tools/manuscript_lint.py` exits 0 for it. |
| Review trigger | A source body, decision policy, metric definition, or workload assumption changes. |

## 2. Case catalog

| Case ID | Case title / link | Domain | Case type | Decision | Primary pattern | Secondary patterns | Evidence | Stage | Last reviewed | Result / limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| CASE-03-16-001 | [Cross-hospital risk model: federated training under a fixed privacy budget](./Cross_hospital_risk_model.md) | Part 03 — Machine Learning, Optimisation and Intelligent Retrieval | hybrid | decision record in the linked case | privacy-safe collaboration | federated, learning, validation | illustrative | draft | null | draft (lint open) |
| CASE-G15-01 | [Fraud-ring investigation under a fixed investigator-hour budget](./Fraud_ring_investigation.md) | Part 03 — Machine Learning, Optimisation and Intelligent Retrieval | predictive-ml | decision record in the linked case | graph-evaluation | entity-resolution, capacity-constrained-triage | illustrative | draft | null | draft (lint open) |
| CASE-03-16-002 | [Private analytics release review: whether to publish a workforce equity and attrition benchmark](./Private_analytics_release_review.md) | Part 03 — Machine Learning, Optimisation and Intelligent Retrieval | analytical | decision record in the linked case | privacy threat model | differential, privacy, release, governance | illustrative | draft | null | draft (lint open) |
| CASE-M14-02 | [Diagnosing a part-number search ranking regression before rolling back](./Search_ranking_regression.md) | Part 03 — Machine Learning, Optimisation and Intelligent Retrieval | incident | decision record in the linked case | retrieval-ranking quality incident | observability, stratified, offline, evaluation | illustrative | draft | null | draft (lint open) |
| PLAN-M11-02 | Churn leakage remediation | Part 03 — Machine Learning, Optimisation and Intelligent Retrieval | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-M12-01 | Payment anomaly detection | Part 03 — Machine Learning, Optimisation and Intelligent Retrieval | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-M12-02 | Campaign segmentation | Part 03 — Machine Learning, Optimisation and Intelligent Retrieval | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-M13-01 | Last-mile routing | Part 03 — Machine Learning, Optimisation and Intelligent Retrieval | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-M13-02 | Workforce scheduling | Part 03 — Machine Learning, Optimisation and Intelligent Retrieval | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-M14-01 | E-commerce recommender | Part 03 — Machine Learning, Optimisation and Intelligent Retrieval | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-M14-02 | Search ranking regression | Part 03 — Machine Learning, Optimisation and Intelligent Retrieval | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-M15-01 | Fraud-ring investigation | Part 03 — Machine Learning, Optimisation and Intelligent Retrieval | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-M15-02 | Enterprise entity-resolution graph | Part 03 — Machine Learning, Optimisation and Intelligent Retrieval | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-M16-01 | Cross-hospital risk model | Part 03 — Machine Learning, Optimisation and Intelligent Retrieval | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-M16-02 | Private analytics release review | Part 03 — Machine Learning, Optimisation and Intelligent Retrieval | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |

## 3. Canonical reusable patterns

| Pattern | Diagnostic signature | Use when | Boundary |
|---|---|---|---|
| evaluation-mismatch | Offline score does not represent the decision | Ranking, forecast, threshold, agent products | The metric captures decision and guardrails |
| data-leakage | Information unavailable at decision time inflates a score | Unexpectedly strong offline result | The feature is genuinely available before the cutoff |
| distribution-shift | Live population differs from training | New market, season, channel, or policy version | The change is sampling noise only |
| human-in-the-loop | Review capacity or override changes the trade-off | High-impact, uncertain, or regulated work | Review is infeasible or demonstrably ineffective |
| model-monitoring | Live signals diverge from the evaluation | Deployed model with usable feedback | No meaningful feedback signal exists |

## 4. Similarity / contrast map

This collection holds 4 case file(s), 0 passing structural lint, and 11 planned commitment(s). Cases require independent review of point-in-time decision logging and explicit baselines; they differ in decision grain, owner, and which guardrail binds. Read the catalog's primary-pattern column to find the nearest contrast: cases sharing a primary pattern differ in constraint, and cases differing in pattern share the decision shape.

## 5. Questions / future-case backlog

Planned commitments, retained without silent deletion:

- `PLAN-M11-02` — Churn leakage remediation. Not authored; no file exists, so no link is offered.
- `PLAN-M12-01` — Payment anomaly detection. Not authored; no file exists, so no link is offered.
- `PLAN-M12-02` — Campaign segmentation. Not authored; no file exists, so no link is offered.
- `PLAN-M13-01` — Last-mile routing. Not authored; no file exists, so no link is offered.
- `PLAN-M13-02` — Workforce scheduling. Not authored; no file exists, so no link is offered.
- `PLAN-M14-01` — E-commerce recommender. Not authored; no file exists, so no link is offered.
- `PLAN-M14-02` — Search ranking regression. Not authored; no file exists, so no link is offered.
- `PLAN-M15-01` — Fraud-ring investigation. Not authored; no file exists, so no link is offered.
- `PLAN-M15-02` — Enterprise entity-resolution graph. Not authored; no file exists, so no link is offered.
- `PLAN-M16-01` — Cross-hospital risk model. Not authored; no file exists, so no link is offered.
- `PLAN-M16-02` — Private analytics release review. Not authored; no file exists, so no link is offered.

## 6. Review / retirement log

| Date | Artifact | Event | Evidence / stage change |
|---|---|---|---|
| 2026-11-10 | [Cross-hospital risk model: federated training under a fixed privacy budget](./Cross_hospital_risk_model.md) | index regenerated from disk | illustrative / draft unchanged |
| 2026-11-10 | [Fraud-ring investigation under a fixed investigator-hour budget](./Fraud_ring_investigation.md) | index regenerated from disk | illustrative / draft unchanged |
| 2026-11-10 | [Private analytics release review: whether to publish a workforce equity and attrition benchmark](./Private_analytics_release_review.md) | index regenerated from disk | illustrative / draft unchanged |
| 2026-11-10 | [Diagnosing a part-number search ranking regression before rolling back](./Search_ranking_regression.md) | index regenerated from disk | illustrative / draft unchanged |

## 7. Pre-publish checks

- [x] Every catalog row links a file that exists, or is marked planned without a link.
- [x] Evidence, implementation stage, and editorial status are separate columns.
- [x] Planned commitments are retained with no silent deletion.
- [ ] Independent source review of every authored case.
- [ ] Publication review of every authored case.
