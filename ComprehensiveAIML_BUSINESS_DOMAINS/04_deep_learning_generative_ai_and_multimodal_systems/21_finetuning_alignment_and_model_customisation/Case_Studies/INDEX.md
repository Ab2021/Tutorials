# Module 21 â€” Fine-Tuning, Alignment and Model Customisation — Case Studies

## 1. Collection metadata

| Field | Value |
|---|---|
| Collection owner | Curriculum maintainer |
| Canonical case-ID format | CASE-2026-FT01 |
| Evidence policy | Illustrative teaching inputs are not company results; unsourced claims stay OPEN. |
| Editorial state | Draft; independent publication review is not claimed. |
| Acceptance rule | A case row is `authored` only when `Tools/manuscript_lint.py` exits 0 for it. |
| Review trigger | A source body, decision policy, metric definition, or workload assumption changes. |

## 2. Case catalog

| Case ID | Case title / link | Domain | Case type | Decision | Primary pattern | Secondary patterns | Evidence | Stage | Last reviewed | Result / limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| CASE-2026-FT01 | [Should support draft generation use a domain adapter?](./01_support_domain_adaptation.md) | Module 21 â€” Fine-Tuning, Alignment and Model Customisation | llm-agentic | decision record in the linked case | evaluation-mismatch | data-leakage, atomic-release-bundle | not-measured | proposed | null | draft (lint open) |
| PLAN-2026-IR004 | Customize tenant assistants without crossing isolation boundaries | Module 21 â€” Fine-Tuning, Alignment and Model Customisation | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-2026-IR009 | Allocate a domain-training budget with restart headroom | Module 21 â€” Fine-Tuning, Alignment and Model Customisation | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-2026-IR010 | Compare specialized models with authorized shared-model baselines | Module 21 â€” Fine-Tuning, Alignment and Model Customisation | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-2026-IR017 | Adapt domain ASR while preserving supported-domain quality | Module 21 â€” Fine-Tuning, Alignment and Model Customisation | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-2026-IR004 | Related adapter lifecycle | Module 21 â€” Fine-Tuning, Alignment and Model Customisation | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-2026-IR010 | Related comparison with a shared baseline | Module 21 â€” Fine-Tuning, Alignment and Model Customisation | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |

## 3. Canonical reusable patterns

| Pattern | Diagnostic signature | Use when | Boundary |
|---|---|---|---|
| evaluation-mismatch | Offline score does not represent the decision | Ranking, forecast, threshold, agent products | The metric captures decision and guardrails |
| data-leakage | Information unavailable at decision time inflates a score | Unexpectedly strong offline result | The feature is genuinely available before the cutoff |
| distribution-shift | Live population differs from training | New market, season, channel, or policy version | The change is sampling noise only |
| human-in-the-loop | Review capacity or override changes the trade-off | High-impact, uncertain, or regulated work | Review is infeasible or demonstrably ineffective |
| model-monitoring | Live signals diverge from the evaluation | Deployed model with usable feedback | No meaningful feedback signal exists |

## 4. Similarity / contrast map

This collection holds 1 authored case(s) and 6 planned commitment(s). Authored cases share point-in-time decision logging and an explicit baseline; they differ in decision grain, owner, and which guardrail binds. Read the catalog's primary-pattern column to find the nearest contrast: cases sharing a primary pattern differ in constraint, and cases differing in pattern share the decision shape.

## 5. Questions / future-case backlog

Planned commitments, retained without silent deletion:

- `PLAN-2026-IR004` — Customize tenant assistants without crossing isolation boundaries. Not authored; no file exists, so no link is offered.
- `PLAN-2026-IR009` — Allocate a domain-training budget with restart headroom. Not authored; no file exists, so no link is offered.
- `PLAN-2026-IR010` — Compare specialized models with authorized shared-model baselines. Not authored; no file exists, so no link is offered.
- `PLAN-2026-IR017` — Adapt domain ASR while preserving supported-domain quality. Not authored; no file exists, so no link is offered.
- `PLAN-2026-IR004` — Related adapter lifecycle. Not authored; no file exists, so no link is offered.
- `PLAN-2026-IR010` — Related comparison with a shared baseline. Not authored; no file exists, so no link is offered.

## 6. Review / retirement log

| Date | Artifact | Event | Evidence / stage change |
|---|---|---|---|
| 2026-11-10 | [Should support draft generation use a domain adapter?](./01_support_domain_adaptation.md) | index regenerated from disk | not-measured / proposed unchanged |

## 7. Pre-publish checks

- [x] Every catalog row links a file that exists, or is marked planned without a link.
- [x] Evidence, implementation stage, and editorial status are separate columns.
- [x] Planned commitments are retained with no silent deletion.
- [ ] Independent source review of every authored case.
- [ ] Publication review of every authored case.
