# Module 14 Case Studies â€” Recommendation, Search, Ads and Ranking

## Authoring standard

Every authored case must follow the [course authoring and numerical standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked case master: all 15 sections, bounded decision and baseline, traceable evidence, checked numerical walkthrough/sizing where applicable, failure diagnostics, 12–18 answered interview questions, and Section 15 last. This is a planned catalog, not completed cases; expand to all seven collection-index sections before publication. Evidence, implementation stage and editorial status remain separate.

## Planned cases

| # | Case | Required outcome | Status |
|---:|---|---|---|
| 1 | E-commerce recommender | Design the decision, architecture, metrics, risks, and validation plan. | planned |
| 2 | search ranking quality regression | Design the decision, architecture, metrics, risks, and validation plan. | planned |

## Suggested file naming

Use **01_case-slug.md**, then **02_case-slug.md**. Add **00_framework.md** only for shared constraints that every case needs.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Planning ID | Proposed case brief | Editorial relationship | Type | Decision / business question | Primary pattern | Secondary patterns | Evidence | Stage | Planning reviewed | Result |
|---|---|---|---|---|---|---|---|---|---|---|
| PLAN-2026-IR018 | Specify correctness contracts before choosing production algorithms | Supporting curriculum; shared brief | hybrid | Which guarantees are needed for road routing, distributed admission and phrase retrieval? | `correctness-contract` | `snapshot-consistency`, `bounded-work` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR020 | Translate local algorithm guarantees into distributed service contracts | Supporting curriculum; shared brief | hybrid | Which correctness and failure guarantees survive sharding, delayed updates and partial outages? | `correctness-contract` | `resource-admission`, `materialized-view` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR021 | Evolve an authorized near-real-time search index safely | Proposed home | hybrid | How should search remain correct across indexing, deletion, ranking and shard failures? | `materialized-view` | `authorization-scope`, `snapshot-consistency` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR024 | Bound phrase-search memory while preserving exact matches | Proposed home | hybrid | How can common-term phrase queries be evaluated without materializing huge posting lists? | `bounded-work` | `correctness-contract`, `snapshot-consistency` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR029 | Return useful graph candidates under explicit traversal budgets | Supporting curriculum; shared brief | incident | How should a recommendation API respond when graph expansion reaches a node, edge, memory or time limit? | `bounded-work` | `correctness-contract`, `incident-postmortem` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR032 | Balance autocomplete freshness, memory and abuse controls | Proposed home | hybrid | Which terms enter a bounded suggestion index, and how can safe new trends appear promptly? | `materialized-view` | `bounded-work`, `rollout-regression` | not-measured | proposed | 2026-11-06 |  |

### Reference and transfer notes

| Planning ID | Example reference only | Scope boundary |
|---|---|---|
| PLAN-2026-IR018 | [REF-IR018](../../../../INTERVIEW_CASES/DSA/Algo_Design_Problems/CASE_STUDIES_01.md) | Seven overlapping teaching units; source measurements are not independently verified and should not be added together |
| PLAN-2026-IR020 | [REF-IR020](../../../../INTERVIEW_CASES/DSA/Algo_Design_Problems/CASE_STUDIES_03.md) | Three units continue earlier routing, admission and search topics; this is not a new independent evidence dataset |
| PLAN-2026-IR021 | [REF-IR021](../../../../INTERVIEW_CASES/DSA/Algo_Design_Problems/CASE_STUDIES_04.md) | Exact global top-k needs a valid distributed protocol; figures in the source are scenario claims, not search-cluster benchmarks |
| PLAN-2026-IR024 | [REF-IR024](../../../../INTERVIEW_CASES/DSA/Algo_Design_Problems/Search/phrase_search_postings.md) | Rarity affects work, not semantic order; deleted documents and analyzer versions must share a consistent snapshot |
| PLAN-2026-IR029 | [REF-IR029](../../../../INTERVIEW_CASES/DSA/Production_DSA_Incidents/Graph_Traversal/bounded_bfs_incident.md) | Depth and output-size limits alone do not bound expansion; source product degrees and outage figures are illustrative claims |
| PLAN-2026-IR032 | [REF-IR032](../../../../INTERVIEW_CASES/DSA/Sys_Des_DSA/Autocomplete/frequency_admission_and_freshness.md) | Sketches cannot enumerate terms; approximate frequencies and novelty thresholds must be evaluated rather than copied |

Case metadata, pattern definitions, supporting/contrasting references, structured backlog, review log and authoring checks for these additions are maintained in the linked central planning catalog. Existing planned rows elsewhere in this index retain their original status. A brief may be split or merged before canonical case IDs are assigned.

<!-- END INTERVIEW REFERENCE PLAN -->
