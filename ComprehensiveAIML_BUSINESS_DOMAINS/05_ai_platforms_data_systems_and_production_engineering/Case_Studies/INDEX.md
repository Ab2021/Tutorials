# Part 05 — AI Platforms, Data Systems and Production Engineering — Case Studies

## 1. Collection metadata

| Field | Value |
|---|---|
| Collection owner | Curriculum maintainer |
| Canonical case-ID format | CASE-XX-NN |
| Evidence policy | Illustrative teaching inputs are not company results; unsourced claims stay OPEN. |
| Editorial state | Draft; independent publication review is not claimed. |
| Acceptance rule | A case row is `authored` only when `Tools/manuscript_lint.py` exits 0 for it. |
| Review trigger | A source body, decision policy, metric definition, or workload assumption changes. |

## 2. Case catalog

| Case ID | Case title / link | Domain | Case type | Decision | Primary pattern | Secondary patterns | Evidence | Stage | Last reviewed | Result / limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| PLAN-P05-00A | Size an inference cluster | Part 05 — AI Platforms, Data Systems and Production Engineering | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-P05-00B | Profile memory-bound training | Part 05 — AI Platforms, Data Systems and Production Engineering | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-P05-01 | Train a large model on limited cluster capacity | Part 05 — AI Platforms, Data Systems and Production Engineering | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-P05-02 | Recover from checkpoint failures | Part 05 — AI Platforms, Data Systems and Production Engineering | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-P05-03 | Multi-model inference gateway | Part 05 — AI Platforms, Data Systems and Production Engineering | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-P05-04 | Reduce p99 LLM latency | Part 05 — AI Platforms, Data Systems and Production Engineering | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-P05-05 | Deploy a credit model safely | Part 05 — AI Platforms, Data Systems and Production Engineering | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-P05-06 | Feature freshness incident response | Part 05 — AI Platforms, Data Systems and Production Engineering | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-P05-07 | Broken upstream schema incident | Part 05 — AI Platforms, Data Systems and Production Engineering | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-P05-08 | Silent label corruption investigation | Part 05 — AI Platforms, Data Systems and Production Engineering | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-P05-09 | RAG quality regression | Part 05 — AI Platforms, Data Systems and Production Engineering | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-P05-10 | Agent tool-use failure analysis | Part 05 — AI Platforms, Data Systems and Production Engineering | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-P05-11 | Internal AI platform design | Part 05 — AI Platforms, Data Systems and Production Engineering | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-P05-12 | Cloud-cost anomaly response | Part 05 — AI Platforms, Data Systems and Production Engineering | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |

## 3. Canonical reusable patterns

| Pattern | Diagnostic signature | Use when | Boundary |
|---|---|---|---|
| evaluation-mismatch | Offline score does not represent the decision | Ranking, forecast, threshold, agent products | The metric captures decision and guardrails |
| data-leakage | Information unavailable at decision time inflates a score | Unexpectedly strong offline result | The feature is genuinely available before the cutoff |
| distribution-shift | Live population differs from training | New market, season, channel, or policy version | The change is sampling noise only |
| human-in-the-loop | Review capacity or override changes the trade-off | High-impact, uncertain, or regulated work | Review is infeasible or demonstrably ineffective |
| model-monitoring | Live signals diverge from the evaluation | Deployed model with usable feedback | No meaningful feedback signal exists |

## 4. Similarity / contrast map

This collection holds 0 authored case(s) and 14 planned commitment(s). Authored cases share point-in-time decision logging and an explicit baseline; they differ in decision grain, owner, and which guardrail binds. Read the catalog's primary-pattern column to find the nearest contrast: cases sharing a primary pattern differ in constraint, and cases differing in pattern share the decision shape.

## 5. Questions / future-case backlog

Planned commitments, retained without silent deletion:

- `PLAN-P05-00A` — Size an inference cluster. Not authored; no file exists, so no link is offered.
- `PLAN-P05-00B` — Profile memory-bound training. Not authored; no file exists, so no link is offered.
- `PLAN-P05-01` — Train a large model on limited cluster capacity. Not authored; no file exists, so no link is offered.
- `PLAN-P05-02` — Recover from checkpoint failures. Not authored; no file exists, so no link is offered.
- `PLAN-P05-03` — Multi-model inference gateway. Not authored; no file exists, so no link is offered.
- `PLAN-P05-04` — Reduce p99 LLM latency. Not authored; no file exists, so no link is offered.
- `PLAN-P05-05` — Deploy a credit model safely. Not authored; no file exists, so no link is offered.
- `PLAN-P05-06` — Feature freshness incident response. Not authored; no file exists, so no link is offered.
- `PLAN-P05-07` — Broken upstream schema incident. Not authored; no file exists, so no link is offered.
- `PLAN-P05-08` — Silent label corruption investigation. Not authored; no file exists, so no link is offered.
- `PLAN-P05-09` — RAG quality regression. Not authored; no file exists, so no link is offered.
- `PLAN-P05-10` — Agent tool-use failure analysis. Not authored; no file exists, so no link is offered.
- `PLAN-P05-11` — Internal AI platform design. Not authored; no file exists, so no link is offered.
- `PLAN-P05-12` — Cloud-cost anomaly response. Not authored; no file exists, so no link is offered.

## 6. Review / retirement log

| Date | Artifact | Event | Evidence / stage change |
|---|---|---|---|
| 2026-11-10 | collection | index regenerated; no authored case yet | not-measured / proposed |

## 7. Pre-publish checks

- [x] Every catalog row links a file that exists, or is marked planned without a link.
- [x] Evidence, implementation stage, and editorial status are separate columns.
- [x] Planned commitments are retained with no silent deletion.
- [ ] Independent source review of every authored case.
- [ ] Publication review of every authored case.
