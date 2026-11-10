# Mobility Dispatch, ETA and Pricing — Case Studies

## 1. Collection metadata

| Field | Value |
|---|---|
| Collection owner | Curriculum maintainer |
| Canonical case-ID format | CASE-13-01-001 |
| Evidence policy | Illustrative teaching inputs are not company results; unsourced claims stay OPEN. |
| Editorial state | Draft; independent publication review is not claimed. |
| Acceptance rule | A case row is `authored` only when `Tools/manuscript_lint.py` exits 0 for it. |
| Review trigger | A source body, decision policy, metric definition, or workload assumption changes. |

## 2. Case catalog

| Case ID | Case title / link | Domain | Case type | Decision | Primary pattern | Secondary patterns | Evidence | Stage | Last reviewed | Result / limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| CASE-13-01-001 | [ETA calibration and arrival-interval publication](./01_ETA_Calibration.md) | Mobility Dispatch, ETA and Pricing | predictive-ml | decision record in the linked case | evaluation-mismatch | distribution-shift, cost-latency-tradeoff | illustrative | proposed | null | draft (lint open) |
| CASE-13-01-002 | [Illustrative Driver-supply forecast](./02_Driver_Supply_Forecast.md) | Mobility Dispatch, ETA and Pricing | analytical | decision record in the linked case | evaluation-mismatch | distribution-shift, human-in-the-loop | illustrative | proposed | null | draft (lint open) |
| CASE-13-01-003 | [Illustrative Surge fairness review](./03_Surge_Fairness_Review.md) | Mobility Dispatch, ETA and Pricing | analytical | decision record in the linked case | evaluation-mismatch | distribution-shift, human-in-the-loop | illustrative | proposed | null | draft (lint open) |
| CASE-13-01-004 | [Illustrative Dispatch failure recovery](./04_Dispatch_Failure_Recovery.md) | Mobility Dispatch, ETA and Pricing | incident | decision record in the linked case | evaluation-mismatch | distribution-shift, human-in-the-loop | illustrative | proposed | null | draft (lint open) |
| CASE-13-01-005 | [Illustrative Safety-event triage](./05_Safety_Event_Triage.md) | Mobility Dispatch, ETA and Pricing | predictive-ml | decision record in the linked case | evaluation-mismatch | distribution-shift, human-in-the-loop | illustrative | proposed | null | draft (lint open) |
| PLAN-13-01-01 | ETA calibration | Mobility Dispatch, ETA and Pricing | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-13-01-02 | Driver-supply forecast | Mobility Dispatch, ETA and Pricing | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-13-01-03 | Surge fairness review | Mobility Dispatch, ETA and Pricing | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-13-01-04 | Dispatch failure | Mobility Dispatch, ETA and Pricing | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-13-01-05 | Safety-event triage | Mobility Dispatch, ETA and Pricing | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-2026-IR018 | Specify correctness contracts before choosing production algorithms | Mobility Dispatch, ETA and Pricing | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-2026-IR019 | Keep routing correctness separate from ETA quality | Mobility Dispatch, ETA and Pricing | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-2026-IR023 | Reject stale shortcuts without losing a shortest-path guarantee | Mobility Dispatch, ETA and Pricing | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |

## 3. Canonical reusable patterns

| Pattern | Diagnostic signature | Use when | Boundary |
|---|---|---|---|
| evaluation-mismatch | Offline score does not represent the decision | Ranking, forecast, threshold, agent products | The metric captures decision and guardrails |
| data-leakage | Information unavailable at decision time inflates a score | Unexpectedly strong offline result | The feature is genuinely available before the cutoff |
| distribution-shift | Live population differs from training | New market, season, channel, or policy version | The change is sampling noise only |
| human-in-the-loop | Review capacity or override changes the trade-off | High-impact, uncertain, or regulated work | Review is infeasible or demonstrably ineffective |
| model-monitoring | Live signals diverge from the evaluation | Deployed model with usable feedback | No meaningful feedback signal exists |

## 4. Similarity / contrast map

This collection holds 5 authored case(s) and 8 planned commitment(s). Authored cases share point-in-time decision logging and an explicit baseline; they differ in decision grain, owner, and which guardrail binds. Read the catalog's primary-pattern column to find the nearest contrast: cases sharing a primary pattern differ in constraint, and cases differing in pattern share the decision shape.

## 5. Questions / future-case backlog

Planned commitments, retained without silent deletion:

- `PLAN-13-01-01` — ETA calibration. Not authored; no file exists, so no link is offered.
- `PLAN-13-01-02` — Driver-supply forecast. Not authored; no file exists, so no link is offered.
- `PLAN-13-01-03` — Surge fairness review. Not authored; no file exists, so no link is offered.
- `PLAN-13-01-04` — Dispatch failure. Not authored; no file exists, so no link is offered.
- `PLAN-13-01-05` — Safety-event triage. Not authored; no file exists, so no link is offered.
- `PLAN-2026-IR018` — Specify correctness contracts before choosing production algorithms. Not authored; no file exists, so no link is offered.
- `PLAN-2026-IR019` — Keep routing correctness separate from ETA quality. Not authored; no file exists, so no link is offered.
- `PLAN-2026-IR023` — Reject stale shortcuts without losing a shortest-path guarantee. Not authored; no file exists, so no link is offered.

## 6. Review / retirement log

| Date | Artifact | Event | Evidence / stage change |
|---|---|---|---|
| 2026-11-10 | [ETA calibration and arrival-interval publication](./01_ETA_Calibration.md) | index regenerated from disk | illustrative / proposed unchanged |
| 2026-11-10 | [Illustrative Driver-supply forecast](./02_Driver_Supply_Forecast.md) | index regenerated from disk | illustrative / proposed unchanged |
| 2026-11-10 | [Illustrative Surge fairness review](./03_Surge_Fairness_Review.md) | index regenerated from disk | illustrative / proposed unchanged |
| 2026-11-10 | [Illustrative Dispatch failure recovery](./04_Dispatch_Failure_Recovery.md) | index regenerated from disk | illustrative / proposed unchanged |
| 2026-11-10 | [Illustrative Safety-event triage](./05_Safety_Event_Triage.md) | index regenerated from disk | illustrative / proposed unchanged |

## 7. Pre-publish checks

- [x] Every catalog row links a file that exists, or is marked planned without a link.
- [x] Evidence, implementation stage, and editorial status are separate columns.
- [x] Planned commitments are retained with no silent deletion.
- [ ] Independent source review of every authored case.
- [ ] Publication review of every authored case.
