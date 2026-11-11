# Advertising, Live Events and Playback Quality — Case Studies

## 1. Collection metadata

| Field | Value |
|---|---|
| Collection owner | Curriculum maintainer |
| Canonical case-ID format | CASE-08-03-01 |
| Evidence policy | Illustrative teaching inputs are not company results; unsourced claims stay OPEN. |
| Editorial state | Draft; independent publication review is not claimed. |
| Acceptance rule | A case row is `authored` only when `Tools/manuscript_lint.py` exits 0 for it. |
| Review trigger | A source body, decision policy, metric definition, or workload assumption changes. |

## 2. Case catalog

| Case ID | Case title / link | Domain | Case type | Decision | Primary pattern | Secondary patterns | Evidence | Stage | Last reviewed | Result / limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| CASE-08-03-01 | [Live Event Capacity Forecast](./01_Live_Event_Capacity_Forecast.md) | Advertising, Live Events and Playback Quality | analytical | decision record in the linked case | evaluation-mismatch | data-leakage, human-in-the-loop | illustrative | proposed | null | draft (lint open) |
| CASE-08-03-02 | [Playback Start Regression](./02_Playback_Start_Regression.md) | Advertising, Live Events and Playback Quality | analytical | decision record in the linked case | evaluation-mismatch | data-leakage, human-in-the-loop | illustrative | proposed | null | draft (lint open) |
| CASE-08-03-03 | [Ad Load Experiment](./03_Ad_Load_Experiment.md) | Advertising, Live Events and Playback Quality | analytical | decision record in the linked case | evaluation-mismatch | data-leakage, human-in-the-loop | illustrative | proposed | null | draft (lint open) |
| CASE-08-03-04 | [QoE Anomaly Detection](./04_QoE_Anomaly_Detection.md) | Advertising, Live Events and Playback Quality | analytical | decision record in the linked case | evaluation-mismatch | data-leakage, human-in-the-loop | illustrative | proposed | null | draft (lint open) |
| CASE-08-03-05 | [Brand Safety Incident](./05_Brand_Safety_Incident.md) | Advertising, Live Events and Playback Quality | analytical | decision record in the linked case | evaluation-mismatch | data-leakage, human-in-the-loop | illustrative | proposed | null | draft (lint open) |
| PLAN-08-03-01 | Live-event capacity forecast | Advertising, Live Events and Playback Quality | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-08-03-02 | Playback-start regression | Advertising, Live Events and Playback Quality | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-08-03-03 | Ad-load experiment | Advertising, Live Events and Playback Quality | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-08-03-04 | QoE anomaly detection | Advertising, Live Events and Playback Quality | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-08-03-05 | Brand-safety incident | Advertising, Live Events and Playback Quality | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |

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

- `PLAN-08-03-01` — Live-event capacity forecast. Not authored; no file exists, so no link is offered.
- `PLAN-08-03-02` — Playback-start regression. Not authored; no file exists, so no link is offered.
- `PLAN-08-03-03` — Ad-load experiment. Not authored; no file exists, so no link is offered.
- `PLAN-08-03-04` — QoE anomaly detection. Not authored; no file exists, so no link is offered.
- `PLAN-08-03-05` — Brand-safety incident. Not authored; no file exists, so no link is offered.

## 6. Review / retirement log

| Date | Artifact | Event | Evidence / stage change |
|---|---|---|---|
| 2026-11-10 | [Live Event Capacity Forecast](./01_Live_Event_Capacity_Forecast.md) | index regenerated from disk | illustrative / proposed unchanged |
| 2026-11-10 | [Playback Start Regression](./02_Playback_Start_Regression.md) | index regenerated from disk | illustrative / proposed unchanged |
| 2026-11-10 | [Ad Load Experiment](./03_Ad_Load_Experiment.md) | index regenerated from disk | illustrative / proposed unchanged |
| 2026-11-10 | [QoE Anomaly Detection](./04_QoE_Anomaly_Detection.md) | index regenerated from disk | illustrative / proposed unchanged |
| 2026-11-10 | [Brand Safety Incident](./05_Brand_Safety_Incident.md) | index regenerated from disk | illustrative / proposed unchanged |

## 7. Pre-publish checks

- [x] Every catalog row links a file that exists, or is marked planned without a link.
- [x] Evidence, implementation stage, and editorial status are separate columns.
- [x] Planned commitments are retained with no silent deletion.
- [ ] Independent source review of every authored case.
- [ ] Publication review of every authored case.
