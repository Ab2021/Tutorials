# Payments, AML and Financial Crime — Case Studies

## 1. Collection metadata

| Field | Value |
|---|---|
| Collection owner | Curriculum maintainer |
| Canonical case-ID format | CASE-11-01-001 |
| Evidence policy | Illustrative teaching inputs are not company results; unsourced claims stay OPEN. |
| Editorial state | Draft; independent publication review is not claimed. |
| Acceptance rule | A case row is `authored` only when `Tools/manuscript_lint.py` exits 0 for it. |
| Review trigger | A source body, decision policy, metric definition, or workload assumption changes. |

## 2. Case catalog

| Case ID | Case title / link | Domain | Case type | Decision | Primary pattern | Secondary patterns | Evidence | Stage | Last reviewed | Result / limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| CASE-11-01-001 | [Card-fraud streaming decision](./01_Card_Fraud_Streaming_Decision.md) | Payments, AML and Financial Crime | streaming-systems | decision record in the linked case | cost-latency-tradeoff | label-delay, human-in-the-loop | illustrative | proposed | null | draft (lint open) |
| CASE-11-01-002 | [AML ring detection with conservative graph evidence](./02_AML_Ring_Detection.md) | Payments, AML and Financial Crime | predictive-ml | decision record in the linked case | - | - | illustrative | proposed | null | draft (lint open) |
| CASE-11-01-003 | [Sanctions candidate-match review design](./03_Sanctions_False_Positive_Reduction.md) | Payments, AML and Financial Crime | analytical | decision record in the linked case | - | - | illustrative | proposed | null | draft (lint open) |
| CASE-11-01-004 | [Investigation prioritisation at fixed review capacity](./04_Investigation_Prioritisation.md) | Payments, AML and Financial Crime | analytical | decision record in the linked case | - | - | illustrative | proposed | null | draft (lint open) |
| CASE-11-01-005 | [Payment outage fraud-control recovery policy](./05_Payment_Outage_Fraud_Controls.md) | Payments, AML and Financial Crime | streaming-systems | decision record in the linked case | - | - | illustrative | proposed | null | draft (lint open) |
| PLAN-11-01-01 | Card-fraud streaming | Payments, AML and Financial Crime | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-11-01-02 | AML ring detection | Payments, AML and Financial Crime | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-11-01-03 | Sanctions false-positive reduction | Payments, AML and Financial Crime | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-11-01-04 | Investigation prioritisation | Payments, AML and Financial Crime | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-11-01-05 | Payment outage fraud controls | Payments, AML and Financial Crime | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-2026-IR036 | Evaluate incremental investigator value from graph fraud signals | Payments, AML and Financial Crime | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |

## 3. Canonical reusable patterns

| Pattern | Diagnostic signature | Use when | Boundary |
|---|---|---|---|
| evaluation-mismatch | Offline score does not represent the decision | Ranking, forecast, threshold, agent products | The metric captures decision and guardrails |
| data-leakage | Information unavailable at decision time inflates a score | Unexpectedly strong offline result | The feature is genuinely available before the cutoff |
| distribution-shift | Live population differs from training | New market, season, channel, or policy version | The change is sampling noise only |
| human-in-the-loop | Review capacity or override changes the trade-off | High-impact, uncertain, or regulated work | Review is infeasible or demonstrably ineffective |
| model-monitoring | Live signals diverge from the evaluation | Deployed model with usable feedback | No meaningful feedback signal exists |

## 4. Similarity / contrast map

This collection holds 5 case file(s), 0 passing structural lint, and 6 planned commitment(s). Cases require independent review of point-in-time decision logging and explicit baselines; they differ in decision grain, owner, and which guardrail binds. Read the catalog's primary-pattern column to find the nearest contrast: cases sharing a primary pattern differ in constraint, and cases differing in pattern share the decision shape.

## 5. Questions / future-case backlog

Planned commitments, retained without silent deletion:

- `PLAN-11-01-01` — Card-fraud streaming. Not authored; no file exists, so no link is offered.
- `PLAN-11-01-02` — AML ring detection. Not authored; no file exists, so no link is offered.
- `PLAN-11-01-03` — Sanctions false-positive reduction. Not authored; no file exists, so no link is offered.
- `PLAN-11-01-04` — Investigation prioritisation. Not authored; no file exists, so no link is offered.
- `PLAN-11-01-05` — Payment outage fraud controls. Not authored; no file exists, so no link is offered.
- `PLAN-2026-IR036` — Evaluate incremental investigator value from graph fraud signals. Not authored; no file exists, so no link is offered.

## 6. Review / retirement log

| Date | Artifact | Event | Evidence / stage change |
|---|---|---|---|
| 2026-11-10 | [Card-fraud streaming decision](./01_Card_Fraud_Streaming_Decision.md) | index regenerated from disk | illustrative / proposed unchanged |
| 2026-11-10 | [AML ring detection with conservative graph evidence](./02_AML_Ring_Detection.md) | index regenerated from disk | illustrative / proposed unchanged |
| 2026-11-10 | [Sanctions candidate-match review design](./03_Sanctions_False_Positive_Reduction.md) | index regenerated from disk | illustrative / proposed unchanged |
| 2026-11-10 | [Investigation prioritisation at fixed review capacity](./04_Investigation_Prioritisation.md) | index regenerated from disk | illustrative / proposed unchanged |
| 2026-11-10 | [Payment outage fraud-control recovery policy](./05_Payment_Outage_Fraud_Controls.md) | index regenerated from disk | illustrative / proposed unchanged |

## 7. Pre-publish checks

- [x] Every catalog row links a file that exists, or is marked planned without a link.
- [x] Evidence, implementation stage, and editorial status are separate columns.
- [x] Planned commitments are retained with no silent deletion.
- [ ] Independent source review of every authored case.
- [ ] Publication review of every authored case.
