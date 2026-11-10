# Module 18 â€” NLP, Speech and Document Intelligence — Case Studies

## 1. Collection metadata

| Field | Value |
|---|---|
| Collection owner | Curriculum maintainer |
| Canonical case-ID format | CASE-18-01 |
| Evidence policy | Illustrative teaching inputs are not company results; unsourced claims stay OPEN. |
| Editorial state | Draft; independent publication review is not claimed. |
| Acceptance rule | A case row is `authored` only when `Tools/manuscript_lint.py` exits 0 for it. |
| Review trigger | A source body, decision policy, metric definition, or workload assumption changes. |

## 2. Case catalog

| Case ID | Case title / link | Domain | Case type | Decision | Primary pattern | Secondary patterns | Evidence | Stage | Last reviewed | Result / limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| CASE-18-01 | [Claims document extraction with evidence-linked abstention](./01_claims_document_extraction.md) | Module 18 â€” NLP, Speech and Document Intelligence | hybrid | decision record in the linked case | human-in-the-loop | evaluation-mismatch, authorization-scope | not-measured | proposed | null | draft (lint open) |
| CASE-18-02 | [Call-centre transcription quality for evidence-based objection reporting](./02_call_centre_transcription_quality.md) | Module 18 â€” NLP, Speech and Document Intelligence | hybrid | decision record in the linked case | evaluation-mismatch | human-in-the-loop, authorization-scope | not-measured | proposed | null | draft (lint open) |
| PLAN-2026-IR010 | Repository-aware completion / language adapter | Module 18 â€” NLP, Speech and Document Intelligence | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-2026-IR011 | VLM/OCR adjuster assistance | Module 18 â€” NLP, Speech and Document Intelligence | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-2026-IR016 | Customer-objection aggregation | Module 18 â€” NLP, Speech and Document Intelligence | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-2026-IR017 | Domain ASR adaptation | Module 18 â€” NLP, Speech and Document Intelligence | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-2026-IR017 | ASR quality can vary by language and route. | Module 18 â€” NLP, Speech and Document Intelligence | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |
| PLAN-2026-IR010 | Language-aware systems need independent splits. | Module 18 â€” NLP, Speech and Document Intelligence | planned | to be framed | - | - | not-measured | proposed | null | planned; no file yet |

## 3. Canonical reusable patterns

| Pattern | Diagnostic signature | Use when | Boundary |
|---|---|---|---|
| evaluation-mismatch | Offline score does not represent the decision | Ranking, forecast, threshold, agent products | The metric captures decision and guardrails |
| data-leakage | Information unavailable at decision time inflates a score | Unexpectedly strong offline result | The feature is genuinely available before the cutoff |
| distribution-shift | Live population differs from training | New market, season, channel, or policy version | The change is sampling noise only |
| human-in-the-loop | Review capacity or override changes the trade-off | High-impact, uncertain, or regulated work | Review is infeasible or demonstrably ineffective |
| model-monitoring | Live signals diverge from the evaluation | Deployed model with usable feedback | No meaningful feedback signal exists |

## 4. Similarity / contrast map

This collection holds 2 authored case(s) and 6 planned commitment(s). Authored cases share point-in-time decision logging and an explicit baseline; they differ in decision grain, owner, and which guardrail binds. Read the catalog's primary-pattern column to find the nearest contrast: cases sharing a primary pattern differ in constraint, and cases differing in pattern share the decision shape.

## 5. Questions / future-case backlog

Planned commitments, retained without silent deletion:

- `PLAN-2026-IR010` — Repository-aware completion / language adapter. Not authored; no file exists, so no link is offered.
- `PLAN-2026-IR011` — VLM/OCR adjuster assistance. Not authored; no file exists, so no link is offered.
- `PLAN-2026-IR016` — Customer-objection aggregation. Not authored; no file exists, so no link is offered.
- `PLAN-2026-IR017` — Domain ASR adaptation. Not authored; no file exists, so no link is offered.
- `PLAN-2026-IR017` — ASR quality can vary by language and route.. Not authored; no file exists, so no link is offered.
- `PLAN-2026-IR010` — Language-aware systems need independent splits.. Not authored; no file exists, so no link is offered.

## 6. Review / retirement log

| Date | Artifact | Event | Evidence / stage change |
|---|---|---|---|
| 2026-11-10 | [Claims document extraction with evidence-linked abstention](./01_claims_document_extraction.md) | index regenerated from disk | not-measured / proposed unchanged |
| 2026-11-10 | [Call-centre transcription quality for evidence-based objection reporting](./02_call_centre_transcription_quality.md) | index regenerated from disk | not-measured / proposed unchanged |

## 7. Pre-publish checks

- [x] Every catalog row links a file that exists, or is marked planned without a link.
- [x] Evidence, implementation stage, and editorial status are separate columns.
- [x] Planned commitments are retained with no silent deletion.
- [ ] Independent source review of every authored case.
- [ ] Publication review of every authored case.
