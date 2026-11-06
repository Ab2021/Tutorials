# Module 54 Case Studies â€” ML and AI System Design Interviews

## Authoring standard

Every authored case must follow the [course authoring and numerical standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked case master: all 15 sections, bounded decision and baseline, traceable evidence, checked numerical walkthrough/sizing where applicable, failure diagnostics, 12–18 answered interview questions, and Section 15 last. This is a planned catalog, not completed cases; expand to all seven collection-index sections before publication. Evidence, implementation stage and editorial status remain separate.

## Planned cases

| # | Case | Required outcome | Status |
|---:|---|---|---|
| 1 | Design content moderation | Design the decision, architecture, metrics, risks, and validation plan. | planned |
| 2 | design enterprise RAG assistant | Design the decision, architecture, metrics, risks, and validation plan. | planned |

## Suggested file naming

Use **01_case-slug.md**, then **02_case-slug.md**. Add **00_framework.md** only for shared constraints that every case needs.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Planning ID | Proposed case brief | Editorial relationship | Type | Decision / business question | Primary pattern | Secondary patterns | Evidence | Stage | Planning reviewed | Result |
|---|---|---|---|---|---|---|---|---|---|---|
| PLAN-2026-IR020 | Translate local algorithm guarantees into distributed service contracts | Proposed home | hybrid | Which correctness and failure guarantees survive sharding, delayed updates and partial outages? | `correctness-contract` | `resource-admission`, `materialized-view` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR026 | Select data structures from operation mix and failure semantics | Supporting curriculum; shared brief | hybrid | Which data structure fits the update/query workload while retaining the required correctness and resource bounds? | `correctness-contract` | `bounded-work`, `evaluation-mismatch` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR034 | Design stateful serving structures with explicit consistency | Supporting curriculum; shared brief | hybrid | Which cache, typeahead, leaderboard and deduplication contracts remain valid during refresh and failure? | `correctness-contract` | `materialized-view`, `snapshot-consistency` | not-measured | proposed | 2026-11-06 |  |

### Reference and transfer notes

| Planning ID | Example reference only | Scope boundary |
|---|---|---|
| PLAN-2026-IR020 | [REF-IR020](../../../../INTERVIEW_CASES/DSA/Algo_Design_Problems/CASE_STUDIES_03.md) | Three units continue earlier routing, admission and search topics; this is not a new independent evidence dataset |
| PLAN-2026-IR026 | [REF-IR026](../../../../INTERVIEW_CASES/DSA/index_basics/INTERVIEW_CASE_STUDIES.md) | Eight teaching units plus rapid prompts; approximate structures, local proofs and service guarantees must be distinguished |
| PLAN-2026-IR034 | [REF-IR034](../../../../INTERVIEW_CASES/DSA/Sys_Des_DSA/INTERVIEW_SYSTEM_DESIGN_CASES.md) | Four teaching units overlap focused examples; Bloom false positives and stale ranks need product-specific tolerance |

Case metadata, pattern definitions, supporting/contrasting references, structured backlog, review log and authoring checks for these additions are maintained in the linked central planning catalog. Existing planned rows elsewhere in this index retain their original status. A brief may be split or merged before canonical case IDs are assigned.

<!-- END INTERVIEW REFERENCE PLAN -->
