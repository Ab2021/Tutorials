# Module 09 Case Studies â€” Data Streaming, Event Systems and Real-Time Analytics

## 1. Collection metadata

This collection covers streaming decisions, recovery, and real-time analytics. Its teaching home is the mapped streaming module; case evidence, implementation stage, and editorial status are independent.

## 2. Authored case catalog

| Case ID | Title | Domain | Case type | Decision | Primary pattern | Secondary patterns | Evidence | Stage | Last reviewed | Result / limitation |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CASE-2026-STR001 | [PayPal observability migration](01_paypal_observability_dataflow.md) | Observability | streaming-systems | Choose streaming execution and ingestion | `recoverable-stream-processing` | `partition-aware-scaling`, `migration-reconciliation` | published | production reported | Not reviewed | Vendor coauthored narrative; qualitative outcomes not independently measured; calculations illustrative. |

Editorial status: CASE-2026-STR001 is a **draft, unreviewed**, not a publication-ready case.

## 3. Canonical reusable patterns

- `recoverable-stream-processing`: align source replay, processing state, and sink effects to a defined recovery contract.
- `partition-aware-scaling`: diagnose and size by partition and key, not only global averages.
- `migration-reconciliation`: compare equivalent source and sink identities across controlled cutover boundaries.

## 4. Similarity and contrast map

The PayPal case connects partition-aware scaling and recovery to a real-source migration; its hypothetical diagnostics are not incident reconstructions. The planned fraud design and consumer-lag incident below have distinct decisions and remain unwritten.

## 5. Questions and future-case backlog

The article does not disclose comparable before/after cost or availability measurements, exact cutover boundaries, or recovery drill artifacts. Obtain authorized evidence before making stronger claims. The planned cases and reference-informed brief below remain in the backlog.

## 6. Review and retirement log

CASE-2026-STR001: draft created 2026-11-07; substantive review not yet performed. Revisit on new primary evidence or material changes to service semantics.

## 7. Pre-publish checks

Before publication, independently review the source ledger, arithmetic, recovery and delivery-semantics boundaries, interview answers, and all links; retain the draft status until the review is completed.

## Authoring standard

Every authored case must follow the [course authoring and numerical standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked case master: all 15 sections, bounded decision and baseline, traceable evidence, checked numerical walkthrough/sizing where applicable, failure diagnostics, 12–18 answered interview questions, and Section 15 last. This is a planned catalog, not completed cases; expand to all seven collection-index sections before publication. Evidence, implementation stage and editorial status remain separate.

## Planned cases

| # | Case | Required outcome | Status |
|---:|---|---|---|
| 1 | Build fraud-alert streaming design | Design the decision, architecture, metrics, risks, and validation plan. | planned |
| 2 | debug consumer lag incident | Design the decision, architecture, metrics, risks, and validation plan. | planned |

## Suggested file naming

Use **01_case-slug.md**, then **02_case-slug.md**. Add **00_framework.md** only for shared constraints that every case needs.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Planning ID | Proposed case brief | Editorial relationship | Type | Decision / business question | Primary pattern | Secondary patterns | Evidence | Stage | Planning reviewed | Result |
|---|---|---|---|---|---|---|---|---|---|---|
| PLAN-2026-IR035 | Separate live rank approximation from exact competition settlement | Supporting curriculum; shared brief | hybrid | How should score events support fast display ranks and a reproducible final prize result? | `snapshot-consistency` | `idempotent-side-effects`, `correctness-contract` | not-measured | proposed | 2026-11-06 |  |

### Reference and transfer notes

| Planning ID | Example reference only | Scope boundary |
|---|---|---|
| PLAN-2026-IR035 | [REF-IR035](../../../../INTERVIEW_CASES/DSA/Sys_Des_DSA/Leaderboards/score_bucket_rank_consistency.md) | A sum of stale counters is not an exact settlement snapshot; tie semantics and late-event policy must be specified |

Case metadata, pattern definitions, supporting/contrasting references, structured backlog, review log and authoring checks for these additions are maintained in the linked central planning catalog. Existing planned rows elsewhere in this index retain their original status. A brief may be split or merged before canonical case IDs are assigned.

<!-- END INTERVIEW REFERENCE PLAN -->
