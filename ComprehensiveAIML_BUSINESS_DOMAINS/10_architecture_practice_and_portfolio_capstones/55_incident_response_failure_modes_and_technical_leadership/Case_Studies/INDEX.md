# Module 55 â€” Incident Response, Failure Modes and Technical Leadership — Case Studies

## 1. Collection metadata

| Field | Value |
|---|---|
| Collection owner | Curriculum maintainer |
| Canonical case-ID format | CASE-STREAM-001 |
| Evidence policy | Illustrative teaching inputs are not company results; unsourced claims stay OPEN. |
| Editorial state | Draft; independent publication review is not claimed. |
| Acceptance rule | A case row is `authored` only when `Tools/manuscript_lint.py` exits 0 for it. |
| Review trigger | A source body, decision policy, metric definition, or workload assumption changes. |

## 2. Case catalog

| Case ID | Case title / link | Domain | Case type | Decision | Primary pattern | Secondary patterns | Evidence | Stage | Last reviewed | Result / limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| CASE-STREAM-001 | [Recovering a hot partition without losing customer-event ordering](./01_hot_partition_consumer_lag_incident.md) | Module 55 â€” Incident Response, Failure Modes and Technical Leadership | streaming-systems | decision record in the linked case | Incident diagnosis and recovery | artition-aware, capacity, planning | illustrative | proposed | null | draft (lint open) |
| CASE-STREAM-002 | [Flink checkpoint failure and memory pressure](./02_flink_checkpoint_failure_oom_incident.md) | Module 55 â€” Incident Response, Failure Modes and Technical Leadership | incident | decision record in the linked case | - | - | - | proposed | null | draft (lint open) |
| PLAN-2026-IR005 | Release and recover an agent as a compatible bundle | Module 55 â€” Incident Response, Failure Modes and Technical Leadership | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-2026-IR028 | Bound request amplification before expensive parsing and validation | Module 55 â€” Incident Response, Failure Modes and Technical Leadership | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-2026-IR029 | Return useful graph candidates under explicit traversal budgets | Module 55 â€” Incident Response, Failure Modes and Technical Leadership | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-2026-IR030 | Diagnose production failures from violated algorithm invariants | Module 55 â€” Incident Response, Failure Modes and Technical Leadership | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |

## 3. Canonical reusable patterns

| Pattern | Diagnostic signature | Use when | Boundary |
|---|---|---|---|
| evaluation-mismatch | Offline score does not represent the decision | Ranking, forecast, threshold, agent products | The metric captures decision and guardrails |
| data-leakage | Information unavailable at decision time inflates a score | Unexpectedly strong offline result | The feature is genuinely available before the cutoff |
| distribution-shift | Live population differs from training | New market, season, channel, or policy version | The change is sampling noise only |
| human-in-the-loop | Review capacity or override changes the trade-off | High-impact, uncertain, or regulated work | Review is infeasible or demonstrably ineffective |
| model-monitoring | Live signals diverge from the evaluation | Deployed model with usable feedback | No meaningful feedback signal exists |

## 4. Similarity / contrast map

This collection holds 2 authored case(s) and 4 planned commitment(s). Authored cases share point-in-time decision logging and an explicit baseline; they differ in decision grain, owner, and which guardrail binds. Read the catalog's primary-pattern column to find the nearest contrast: cases sharing a primary pattern differ in constraint, and cases differing in pattern share the decision shape.

## 5. Questions / future-case backlog

Planned commitments, retained without silent deletion:

- `PLAN-2026-IR005` — Release and recover an agent as a compatible bundle. Not authored; no file exists, so no link is offered.
- `PLAN-2026-IR028` — Bound request amplification before expensive parsing and validation. Not authored; no file exists, so no link is offered.
- `PLAN-2026-IR029` — Return useful graph candidates under explicit traversal budgets. Not authored; no file exists, so no link is offered.
- `PLAN-2026-IR030` — Diagnose production failures from violated algorithm invariants. Not authored; no file exists, so no link is offered.

## 6. Review / retirement log

| Date | Artifact | Event | Evidence / stage change |
|---|---|---|---|
| 2026-11-10 | [Recovering a hot partition without losing customer-event ordering](./01_hot_partition_consumer_lag_incident.md) | index regenerated from disk | illustrative / proposed unchanged |
| 2026-11-10 | [Flink checkpoint failure and memory pressure](./02_flink_checkpoint_failure_oom_incident.md) | index regenerated from disk | - / proposed unchanged |

## 7. Pre-publish checks

- [x] Every catalog row links a file that exists, or is marked planned without a link.
- [x] Evidence, implementation stage, and editorial status are separate columns.
- [x] Planned commitments are retained with no silent deletion.
- [ ] Independent source review of every authored case.
- [ ] Publication review of every authored case.
