# Module 06 Case Studies â€” Data Modelling, Databases, Lakehouse and Governance

## Authoring standard

Every authored case must follow the [course authoring and numerical standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked case master: all 15 sections, bounded decision and baseline, traceable evidence, checked numerical walkthrough/sizing where applicable, failure diagnostics, 12–18 answered interview questions, and Section 15 last. This is a planned catalog, not completed cases; expand to all seven collection-index sections before publication. Evidence, implementation stage and editorial status remain separate.

## Planned cases

| # | Case | Required outcome | Status |
|---:|---|---|---|
| 1 | Model an e-commerce lakehouse | Design the decision, architecture, metrics, risks, and validation plan. | planned |
| 2 | repair a broken customer 360 pipeline | Design the decision, architecture, metrics, risks, and validation plan. | planned |

## Suggested file naming

Use **01_case-slug.md**, then **02_case-slug.md**. Add **00_framework.md** only for shared constraints that every case needs.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Planning ID | Proposed case brief | Editorial relationship | Type | Decision / business question | Primary pattern | Secondary patterns | Evidence | Stage | Planning reviewed | Result |
|---|---|---|---|---|---|---|---|---|---|---|
| PLAN-2026-IR012 | Preserve answer evidence through ingestion and filtered retrieval | Supporting curriculum; shared brief | llm-agentic | Which chunking and index design retrieves every required authorized span within the service budget? | `materialized-view` | `authorization-scope`, `evaluation-mismatch` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR021 | Evolve an authorized near-real-time search index safely | Supporting curriculum; shared brief | hybrid | How should search remain correct across indexing, deletion, ranking and shard failures? | `materialized-view` | `authorization-scope`, `snapshot-consistency` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR031 | Load authorized hierarchies with bounded database work | Proposed home | incident | Should a subtree API use a recursive CTE or bounded level-order batching for its workload? | `bounded-work` | `authorization-scope`, `snapshot-consistency` | not-measured | proposed | 2026-11-06 |  |

### Reference and transfer notes

| Planning ID | Example reference only | Scope boundary |
|---|---|---|
| PLAN-2026-IR012 | [REF-IR012](../../../../INTERVIEW_CASES/Agentic_AI_System/RAG_Infra_Research/Chunking_And_Vector_Index.md) | Reranking cannot recover a missing candidate; the source embedding limit and tenant count are not platform requirements |
| PLAN-2026-IR021 | [REF-IR021](../../../../INTERVIEW_CASES/DSA/Algo_Design_Problems/CASE_STUDIES_04.md) | Exact global top-k needs a valid distributed protocol; figures in the source are scenario claims, not search-cluster benchmarks |
| PLAN-2026-IR031 | [REF-IR031](../../../../INTERVIEW_CASES/DSA/Production_DSA_Incidents/Tree_Queries/policy_hierarchy_n_plus_one.md) | One SQL statement can still do excessive work; an outer LIMIT is not a universal recursion work bound |

Case metadata, pattern definitions, supporting/contrasting references, structured backlog, review log and authoring checks for these additions are maintained in the linked central planning catalog. Existing planned rows elsewhere in this index retain their original status. A brief may be split or merged before canonical case IDs are assigned.

<!-- END INTERVIEW REFERENCE PLAN -->
