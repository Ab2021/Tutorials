# Travel, Hospitality, Revenue and Disruption — Case Studies

## 1. Collection metadata

| Field | Value |
|---|---|
| Collection owner | Curriculum maintainer |
| Canonical case-ID format | CASE-13-02-001 |
| Evidence policy | Illustrative teaching inputs are not company results; unsourced claims stay OPEN. |
| Editorial state | Draft; independent publication review is not claimed. |
| Acceptance rule | A case row is `authored` only when `Tools/manuscript_lint.py` exits 0 for it. |
| Review trigger | A source body, decision policy, metric definition, or workload assumption changes. |

## 2. Case catalog

| Case ID | Case title / link | Domain | Case type | Decision | Primary pattern | Secondary patterns | Evidence | Stage | Last reviewed | Result / limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| CASE-13-02-001 | [Illustrative Hotel revenue forecast](./01_Hotel_Revenue_Forecast.md) | Travel, Hospitality, Revenue and Disruption | analytical | decision record in the linked case | evaluation-mismatch | distribution-shift, human-in-the-loop | illustrative | proposed | null | draft (lint open) |
| CASE-13-02-002 | [Illustrative Flight disruption rebooking](./02_Flight_Disruption_Rebooking.md) | Travel, Hospitality, Revenue and Disruption | optimization | decision record in the linked case | evaluation-mismatch | distribution-shift, human-in-the-loop | illustrative | proposed | null | draft (lint open) |
| CASE-13-02-003 | [Illustrative Travel search ranking](./03_Travel_Search_Ranking.md) | Travel, Hospitality, Revenue and Disruption | predictive-ml | decision record in the linked case | evaluation-mismatch | distribution-shift, human-in-the-loop | illustrative | proposed | null | draft (lint open) |
| CASE-13-02-004 | [Illustrative No-show prediction](./04_No_Show_Prediction.md) | Travel, Hospitality, Revenue and Disruption | predictive-ml | decision record in the linked case | evaluation-mismatch | distribution-shift, human-in-the-loop | illustrative | proposed | null | draft (lint open) |
| CASE-13-02-005 | [Illustrative Service recovery prioritisation](./05_Service_Recovery.md) | Travel, Hospitality, Revenue and Disruption | analytical | decision record in the linked case | evaluation-mismatch | distribution-shift, human-in-the-loop | illustrative | proposed | null | draft (lint open) |
| PLAN-13-02-01 | Hotel revenue forecast | Travel, Hospitality, Revenue and Disruption | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-13-02-02 | Flight disruption rebooking | Travel, Hospitality, Revenue and Disruption | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-13-02-03 | Travel search ranking | Travel, Hospitality, Revenue and Disruption | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-13-02-04 | No-show prediction | Travel, Hospitality, Revenue and Disruption | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-13-02-05 | Service recovery | Travel, Hospitality, Revenue and Disruption | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |

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

- `PLAN-13-02-01` — Hotel revenue forecast. Not authored; no file exists, so no link is offered.
- `PLAN-13-02-02` — Flight disruption rebooking. Not authored; no file exists, so no link is offered.
- `PLAN-13-02-03` — Travel search ranking. Not authored; no file exists, so no link is offered.
- `PLAN-13-02-04` — No-show prediction. Not authored; no file exists, so no link is offered.
- `PLAN-13-02-05` — Service recovery. Not authored; no file exists, so no link is offered.

## 6. Review / retirement log

| Date | Artifact | Event | Evidence / stage change |
|---|---|---|---|
| 2026-11-10 | [Illustrative Hotel revenue forecast](./01_Hotel_Revenue_Forecast.md) | index regenerated from disk | illustrative / proposed unchanged |
| 2026-11-10 | [Illustrative Flight disruption rebooking](./02_Flight_Disruption_Rebooking.md) | index regenerated from disk | illustrative / proposed unchanged |
| 2026-11-10 | [Illustrative Travel search ranking](./03_Travel_Search_Ranking.md) | index regenerated from disk | illustrative / proposed unchanged |
| 2026-11-10 | [Illustrative No-show prediction](./04_No_Show_Prediction.md) | index regenerated from disk | illustrative / proposed unchanged |
| 2026-11-10 | [Illustrative Service recovery prioritisation](./05_Service_Recovery.md) | index regenerated from disk | illustrative / proposed unchanged |

## 7. Pre-publish checks

- [x] Every catalog row links a file that exists, or is marked planned without a link.
- [x] Evidence, implementation stage, and editorial status are separate columns.
- [x] Planned commitments are retained with no silent deletion.
- [ ] Independent source review of every authored case.
- [ ] Publication review of every authored case.
