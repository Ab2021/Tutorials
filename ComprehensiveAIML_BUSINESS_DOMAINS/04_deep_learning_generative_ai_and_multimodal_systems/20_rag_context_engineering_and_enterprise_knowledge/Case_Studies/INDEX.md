# Module 20 Case Studies â€” RAG, Context Engineering and Enterprise Knowledge

## Authoring standard

Every authored case must follow the [course authoring and numerical standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked case master: all 15 sections, bounded decision and baseline, traceable evidence, checked numerical walkthrough/sizing where applicable, failure diagnostics, 12–18 answered interview questions, and Section 15 last. This is a planned catalog, not completed cases; expand to all seven collection-index sections before publication. Evidence, implementation stage and editorial status remain separate.

## Planned cases

| # | Case | Required outcome | Status |
|---:|---|---|---|
| 1 | Policy assistant with citations | Design the decision, architecture, metrics, risks, and validation plan. | planned |
| 2 | knowledge-base migration evaluation | Design the decision, architecture, metrics, risks, and validation plan. | planned |

## Suggested file naming

Use **01_case-slug.md**, then **02_case-slug.md**. Add **00_framework.md** only for shared constraints that every case needs.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Planning ID | Proposed case brief | Editorial relationship | Type | Decision / business question | Primary pattern | Secondary patterns | Evidence | Stage | Planning reviewed | Result |
|---|---|---|---|---|---|---|---|---|---|---|
| PLAN-2026-IR003 | Answer support questions from the applicable policy version | Proposed home | llm-agentic | When should a support assistant answer, abstain or escalate a policy question? | `evaluation-mismatch` | `authorization-scope`, `materialized-view` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR012 | Preserve answer evidence through ingestion and filtered retrieval | Proposed home | llm-agentic | Which chunking and index design retrieves every required authorized span within the service budget? | `materialized-view` | `authorization-scope`, `evaluation-mismatch` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR013 | Retrieve regulations applicable to the question date and jurisdiction | Proposed home | llm-agentic | Which sources are eligible before ranking, and when is legal-owner review required? | `eligibility-before-ranking` | `snapshot-consistency`, `human-in-the-loop` | not-measured | proposed | 2026-11-06 |  |

### Reference and transfer notes

| Planning ID | Example reference only | Scope boundary |
|---|---|---|
| PLAN-2026-IR003 | [REF-IR003](../../../../INTERVIEW_CASES/Agentic_AI_System/Applied_AI_engineering/grounded_policy_rag.md) | Citations and entailment checks require validation; never transfer the example refund window into a new business |
| PLAN-2026-IR012 | [REF-IR012](../../../../INTERVIEW_CASES/Agentic_AI_System/RAG_Infra_Research/Chunking_And_Vector_Index.md) | Reranking cannot recover a missing candidate; the source embedding limit and tenant count are not platform requirements |
| PLAN-2026-IR013 | [REF-IR013](../../../../INTERVIEW_CASES/Agentic_AI_System/RAG_Infra_Research/Regulatory_RAG_And_Reranking.md) | This is an architecture exercise, not current legal guidance; eligibility depends on maintained authoritative metadata |

Case metadata, pattern definitions, supporting/contrasting references, structured backlog, review log and authoring checks for these additions are maintained in the linked central planning catalog. Existing planned rows elsewhere in this index retain their original status. A brief may be split or merged before canonical case IDs are assigned.

<!-- END INTERVIEW REFERENCE PLAN -->
