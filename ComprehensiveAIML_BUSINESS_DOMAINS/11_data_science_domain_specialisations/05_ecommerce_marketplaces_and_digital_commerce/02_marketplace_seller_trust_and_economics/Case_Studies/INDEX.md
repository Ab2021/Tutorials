# Marketplace Seller, Trust and Economics — Case Studies

## 1. Collection metadata

| Field | Value |
|---|---|
| Collection owner | Curriculum maintainer |
| Canonical case-ID format | CASE-05-02-001 |
| Evidence policy | Illustrative teaching inputs are not company results; unsourced claims stay OPEN. |
| Editorial state | Draft; independent publication review is not claimed. |
| Acceptance rule | A case row is `authored` only when `Tools/manuscript_lint.py` exits 0 for it. |
| Review trigger | A source body, decision policy, metric definition, or workload assumption changes. |

## 2. Case catalog

| Case ID | Case title / link | Domain | Case type | Decision | Primary pattern | Secondary patterns | Evidence | Stage | Last reviewed | Result / limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| CASE-05-02-001 | [Seller-risk scoring](./01_Seller_Risk_Scoring.md) | Marketplace Seller, Trust and Economics | predictive-ml | decision record in the linked case | evaluation-mismatch | human-in-the-loop, rollout-regression | illustrative | proposed | null | draft (lint open) |
| CASE-05-02-002 | [Dispute automation with human review](./02_Dispute_Automation_Review.md) | Marketplace Seller, Trust and Economics | analytical | decision record in the linked case | evaluation-mismatch | human-in-the-loop, rollout-regression | illustrative | proposed | null | draft (lint open) |
| CASE-05-02-003 | [Marketplace fee experiment](./03_Marketplace_Fee_Experiment.md) | Marketplace Seller, Trust and Economics | analytical | decision record in the linked case | evaluation-mismatch | human-in-the-loop, rollout-regression | illustrative | proposed | null | draft (lint open) |
| CASE-05-02-004 | [Counterfeit detection triage](./04_Counterfeit_Detection.md) | Marketplace Seller, Trust and Economics | predictive-ml | decision record in the linked case | evaluation-mismatch | human-in-the-loop, rollout-regression | illustrative | proposed | null | draft (lint open) |
| CASE-05-02-005 | [Supply-demand imbalance response](./05_Supply_Demand_Imbalance.md) | Marketplace Seller, Trust and Economics | analytical | decision record in the linked case | evaluation-mismatch | human-in-the-loop, rollout-regression | illustrative | proposed | null | draft (lint open) |
| PLAN-05-02-01 | Seller-risk scoring | Marketplace Seller, Trust and Economics | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-05-02-02 | Dispute automation with human review | Marketplace Seller, Trust and Economics | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-05-02-03 | Marketplace fee experiment | Marketplace Seller, Trust and Economics | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-05-02-04 | Counterfeit detection | Marketplace Seller, Trust and Economics | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-05-02-05 | Supply-demand imbalance | Marketplace Seller, Trust and Economics | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |

## 3. Canonical reusable patterns

| Pattern | Diagnostic signature | Use when | Boundary |
|---|---|---|---|
| evaluation-mismatch | Offline score does not represent the decision | Ranking, forecast, threshold, agent products | The metric captures decision and guardrails |
| data-leakage | Information unavailable at decision time inflates a score | Unexpectedly strong offline result | The feature is genuinely available before the cutoff |
| distribution-shift | Live population differs from training | New market, season, channel, or policy version | The change is sampling noise only |
| human-in-the-loop | Review capacity or override changes the trade-off | High-impact, uncertain, or regulated work | Review is infeasible or demonstrably ineffective |
| model-monitoring | Live signals diverge from the evaluation | Deployed model with usable feedback | No meaningful feedback signal exists |

## 4. Similarity / contrast map

This collection holds 5 case file(s), 0 passing structural lint, and 5 planned commitment(s). Cases require independent review of point-in-time decision logging and explicit baselines; they differ in decision grain, owner, and which guardrail binds. Read the catalog's primary-pattern column to find the nearest contrast: cases sharing a primary pattern differ in constraint, and cases differing in pattern share the decision shape.

## 5. Questions / future-case backlog

Planned commitments, retained without silent deletion:

- `PLAN-05-02-01` — Seller-risk scoring. Not authored; no file exists, so no link is offered.
- `PLAN-05-02-02` — Dispute automation with human review. Not authored; no file exists, so no link is offered.
- `PLAN-05-02-03` — Marketplace fee experiment. Not authored; no file exists, so no link is offered.
- `PLAN-05-02-04` — Counterfeit detection. Not authored; no file exists, so no link is offered.
- `PLAN-05-02-05` — Supply-demand imbalance. Not authored; no file exists, so no link is offered.

## 6. Review / retirement log

| Date | Artifact | Event | Evidence / stage change |
|---|---|---|---|
| 2026-11-10 | [Seller-risk scoring](./01_Seller_Risk_Scoring.md) | index regenerated from disk | illustrative / proposed unchanged |
| 2026-11-10 | [Dispute automation with human review](./02_Dispute_Automation_Review.md) | index regenerated from disk | illustrative / proposed unchanged |
| 2026-11-10 | [Marketplace fee experiment](./03_Marketplace_Fee_Experiment.md) | index regenerated from disk | illustrative / proposed unchanged |
| 2026-11-10 | [Counterfeit detection triage](./04_Counterfeit_Detection.md) | index regenerated from disk | illustrative / proposed unchanged |
| 2026-11-10 | [Supply-demand imbalance response](./05_Supply_Demand_Imbalance.md) | index regenerated from disk | illustrative / proposed unchanged |

## 7. Pre-publish checks

- [x] Every catalog row links a file that exists, or is marked planned without a link.
- [x] Evidence, implementation stage, and editorial status are separate columns.
- [x] Planned commitments are retained with no silent deletion.
- [ ] Independent source review of every authored case.
- [ ] Publication review of every authored case.
