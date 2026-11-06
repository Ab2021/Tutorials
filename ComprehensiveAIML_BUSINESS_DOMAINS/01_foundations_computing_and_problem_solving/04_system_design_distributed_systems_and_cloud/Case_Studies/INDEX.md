# Module 04 Case Studies â€” System Design, Distributed Systems and Cloud

## Authoring standard

Every authored case must follow the [course authoring and numerical standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked case master: all 15 sections, bounded decision and baseline, traceable evidence, checked numerical walkthrough/sizing where applicable, failure diagnostics, 12–18 answered interview questions, and Section 15 last. This is a planned catalog, not completed cases; expand to all seven collection-index sections before publication. Evidence, implementation stage and editorial status remain separate.

## Planned cases

| # | Case | Required outcome | Status |
|---:|---|---|---|
| 1 | Design a notification platform | Design the decision, architecture, metrics, risks, and validation plan. | planned |
| 2 | design a feature-serving API | Design the decision, architecture, metrics, risks, and validation plan. | planned |

## Suggested file naming

Use **01_case-slug.md**, then **02_case-slug.md**. Add **00_framework.md** only for shared constraints that every case needs.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Planning ID | Proposed case brief | Editorial relationship | Type | Decision / business question | Primary pattern | Secondary patterns | Evidence | Stage | Planning reviewed | Result |
|---|---|---|---|---|---|---|---|---|---|---|
| PLAN-2026-IR018 | Specify correctness contracts before choosing production algorithms | Supporting curriculum; shared brief | hybrid | Which guarantees are needed for road routing, distributed admission and phrase retrieval? | `correctness-contract` | `snapshot-consistency`, `bounded-work` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR019 | Keep routing correctness separate from ETA quality | Proposed home | hybrid | How should a navigation service publish legal optimal routes while separately evaluating predicted arrival time? | `snapshot-consistency` | `correctness-contract`, `evaluation-mismatch` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR020 | Translate local algorithm guarantees into distributed service contracts | Supporting curriculum; shared brief | hybrid | Which correctness and failure guarantees survive sharding, delayed updates and partial outages? | `correctness-contract` | `resource-admission`, `materialized-view` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR022 | Bound global API admission with token leases | Supporting curriculum; shared brief | hybrid | Which admission policy remains within its global budget during contention or coordinator failure? | `resource-admission` | `correctness-contract`, `bounded-work` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR033 | Rebalance caches without overwhelming the authoritative store | Proposed home | hybrid | How should cache fills and ring migration preserve freshness while limiting database amplification? | `materialized-view` | `resource-admission`, `snapshot-consistency` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR034 | Design stateful serving structures with explicit consistency | Proposed home | hybrid | Which cache, typeahead, leaderboard and deduplication contracts remain valid during refresh and failure? | `correctness-contract` | `materialized-view`, `snapshot-consistency` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR035 | Separate live rank approximation from exact competition settlement | Proposed home | hybrid | How should score events support fast display ranks and a reproducible final prize result? | `snapshot-consistency` | `idempotent-side-effects`, `correctness-contract` | not-measured | proposed | 2026-11-06 |  |

### Reference and transfer notes

| Planning ID | Example reference only | Scope boundary |
|---|---|---|
| PLAN-2026-IR018 | [REF-IR018](../../../../INTERVIEW_CASES/DSA/Algo_Design_Problems/CASE_STUDIES_01.md) | Seven overlapping teaching units; source measurements are not independently verified and should not be added together |
| PLAN-2026-IR019 | [REF-IR019](../../../../INTERVIEW_CASES/DSA/Algo_Design_Problems/CASE_STUDIES_02.md) | The OCR filename and ordinal ranges disagree; arbitrary traffic overlays do not preserve ordinary CH correctness |
| PLAN-2026-IR020 | [REF-IR020](../../../../INTERVIEW_CASES/DSA/Algo_Design_Problems/CASE_STUDIES_03.md) | Three units continue earlier routing, admission and search topics; this is not a new independent evidence dataset |
| PLAN-2026-IR022 | [REF-IR022](../../../../INTERVIEW_CASES/DSA/Algo_Design_Problems/Rate_Limiting/distributed_token_buckets.md) | Periodic usage reporting alone is not coordinated allocation; token buckets and exact rolling windows have different contracts |
| PLAN-2026-IR033 | [REF-IR033](../../../../INTERVIEW_CASES/DSA/Sys_Des_DSA/Cache_Resilience/cold_cache_and_ring_migration.md) | Local single-flight is not fleet-wide coordination; expected key movement is not the same as request miss share |
| PLAN-2026-IR034 | [REF-IR034](../../../../INTERVIEW_CASES/DSA/Sys_Des_DSA/INTERVIEW_SYSTEM_DESIGN_CASES.md) | Four teaching units overlap focused examples; Bloom false positives and stale ranks need product-specific tolerance |
| PLAN-2026-IR035 | [REF-IR035](../../../../INTERVIEW_CASES/DSA/Sys_Des_DSA/Leaderboards/score_bucket_rank_consistency.md) | A sum of stale counters is not an exact settlement snapshot; tie semantics and late-event policy must be specified |

Case metadata, pattern definitions, supporting/contrasting references, structured backlog, review log and authoring checks for these additions are maintained in the linked central planning catalog. Existing planned rows elsewhere in this index retain their original status. A brief may be split or merged before canonical case IDs are assigned.

<!-- END INTERVIEW REFERENCE PLAN -->
