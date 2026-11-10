# Claims, Fraud, Customer and Operations — Case Studies

## 1. Collection metadata

| Field | Value |
|---|---|
| Collection owner | Curriculum maintainer |
| Canonical case-ID format | CASE-10-02-006 |
| Evidence policy | Illustrative teaching inputs are not company results; unsourced claims stay OPEN. |
| Editorial state | Draft; independent publication review is not claimed. |
| Acceptance rule | A case row is `authored` only when `Tools/manuscript_lint.py` exits 0 for it. |
| Review trigger | A source body, decision policy, metric definition, or workload assumption changes. |

## 2. Case catalog

| Case ID | Case title / link | Domain | Case type | Decision | Primary pattern | Secondary patterns | Evidence | Stage | Last reviewed | Result / limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| CASE-10-02-006 | [Medical claims document extraction: separating field accuracy from safe processing](./Medical_Claims_Document_Extraction_MediConCen.md) | Claims, Fraud, Customer and Operations | llm-agentic | decision record in the linked case | evaluation-mismatch | human-in-the-loop, cost-latency-tradeoff | published | production | null | draft (lint open) |
| PLAN-10-02-01 | FNOL triage | Claims, Fraud, Customer and Operations | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-10-02-02 | Claims-fraud detection | Claims, Fraud, Customer and Operations | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-10-02-03 | Damage-image estimate | Claims, Fraud, Customer and Operations | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-10-02-04 | Customer communications | Claims, Fraud, Customer and Operations | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-10-02-05 | Claims backlog optimisation | Claims, Fraud, Customer and Operations | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-2026-IR005 | Release and recover an agent as a compatible bundle | Claims, Fraud, Customer and Operations | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-2026-IR011 | Compare document VLMs with OCR for adjuster assistance | Claims, Fraud, Customer and Operations | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-2026-IR014 | Separate extraction correctness from routing economics | Claims, Fraud, Customer and Operations | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-2026-IR015 | Publish reviewed coaching scripts from current product evidence | Claims, Fraud, Customer and Operations | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-2026-IR016 | Distinguish speech, speaker and extraction errors in call analytics | Claims, Fraud, Customer and Operations | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-2026-IR031 | Load authorized hierarchies with bounded database work | Claims, Fraud, Customer and Operations | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-2026-IR036 | Evaluate incremental investigator value from graph fraud signals | Claims, Fraud, Customer and Operations | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |

## 3. Canonical reusable patterns

| Pattern | Diagnostic signature | Use when | Boundary |
|---|---|---|---|
| evaluation-mismatch | Offline score does not represent the decision | Ranking, forecast, threshold, agent products | The metric captures decision and guardrails |
| data-leakage | Information unavailable at decision time inflates a score | Unexpectedly strong offline result | The feature is genuinely available before the cutoff |
| distribution-shift | Live population differs from training | New market, season, channel, or policy version | The change is sampling noise only |
| human-in-the-loop | Review capacity or override changes the trade-off | High-impact, uncertain, or regulated work | Review is infeasible or demonstrably ineffective |
| model-monitoring | Live signals diverge from the evaluation | Deployed model with usable feedback | No meaningful feedback signal exists |

## 4. Similarity / contrast map

This collection holds 1 authored case(s) and 12 planned commitment(s). Authored cases share point-in-time decision logging and an explicit baseline; they differ in decision grain, owner, and which guardrail binds. Read the catalog's primary-pattern column to find the nearest contrast: cases sharing a primary pattern differ in constraint, and cases differing in pattern share the decision shape.

## 5. Questions / future-case backlog

Planned commitments, retained without silent deletion:

- `PLAN-10-02-01` — FNOL triage. Not authored; no file exists, so no link is offered.
- `PLAN-10-02-02` — Claims-fraud detection. Not authored; no file exists, so no link is offered.
- `PLAN-10-02-03` — Damage-image estimate. Not authored; no file exists, so no link is offered.
- `PLAN-10-02-04` — Customer communications. Not authored; no file exists, so no link is offered.
- `PLAN-10-02-05` — Claims backlog optimisation. Not authored; no file exists, so no link is offered.
- `PLAN-2026-IR005` — Release and recover an agent as a compatible bundle. Not authored; no file exists, so no link is offered.
- `PLAN-2026-IR011` — Compare document VLMs with OCR for adjuster assistance. Not authored; no file exists, so no link is offered.
- `PLAN-2026-IR014` — Separate extraction correctness from routing economics. Not authored; no file exists, so no link is offered.
- `PLAN-2026-IR015` — Publish reviewed coaching scripts from current product evidence. Not authored; no file exists, so no link is offered.
- `PLAN-2026-IR016` — Distinguish speech, speaker and extraction errors in call analytics. Not authored; no file exists, so no link is offered.
- `PLAN-2026-IR031` — Load authorized hierarchies with bounded database work. Not authored; no file exists, so no link is offered.
- `PLAN-2026-IR036` — Evaluate incremental investigator value from graph fraud signals. Not authored; no file exists, so no link is offered.

## 6. Review / retirement log

| Date | Artifact | Event | Evidence / stage change |
|---|---|---|---|
| 2026-11-10 | [Medical claims document extraction: separating field accuracy from safe processing](./Medical_Claims_Document_Extraction_MediConCen.md) | index regenerated from disk | published / production unchanged |

## 7. Pre-publish checks

- [x] Every catalog row links a file that exists, or is marked planned without a link.
- [x] Evidence, implementation stage, and editorial status are separate columns.
- [x] Planned commitments are retained with no silent deletion.
- [ ] Independent source review of every authored case.
- [ ] Publication review of every authored case.
