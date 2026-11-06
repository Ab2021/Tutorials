# Module 03 Case Studies â€” Data Structures, Algorithms and Computational Thinking

## Authoring standard

Every authored case must follow the [course authoring and numerical standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked case master: all 15 sections, bounded decision and baseline, traceable evidence, checked numerical walkthrough/sizing where applicable, failure diagnostics, 12–18 answered interview questions, and Section 15 last. This is a planned catalog, not completed cases; expand to all seven collection-index sections before publication. Evidence, implementation stage and editorial status remain separate.

## Planned cases

| # | Case | Required outcome | Status |
|---:|---|---|---|
| 1 | Design a rate limiter | Design the decision, architecture, metrics, risks, and validation plan. | planned |
| 2 | implement top-K stream processing | Design the decision, architecture, metrics, risks, and validation plan. | planned |

## Suggested file naming

Use **01_case-slug.md**, then **02_case-slug.md**. Add **00_framework.md** only for shared constraints that every case needs.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Planning ID | Proposed case brief | Editorial relationship | Type | Decision / business question | Primary pattern | Secondary patterns | Evidence | Stage | Planning reviewed | Result |
|---|---|---|---|---|---|---|---|---|---|---|
| PLAN-2026-IR018 | Specify correctness contracts before choosing production algorithms | Proposed home | hybrid | Which guarantees are needed for road routing, distributed admission and phrase retrieval? | `correctness-contract` | `snapshot-consistency`, `bounded-work` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR019 | Keep routing correctness separate from ETA quality | Supporting curriculum; shared brief | hybrid | How should a navigation service publish legal optimal routes while separately evaluating predicted arrival time? | `snapshot-consistency` | `correctness-contract`, `evaluation-mismatch` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR022 | Bound global API admission with token leases | Proposed home | hybrid | Which admission policy remains within its global budget during contention or coordinator failure? | `resource-admission` | `correctness-contract`, `bounded-work` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR023 | Reject stale shortcuts without losing a shortest-path guarantee | Proposed home | hybrid | When must a route query fall back to the live base graph after a closure? | `correctness-contract` | `snapshot-consistency` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR024 | Bound phrase-search memory while preserving exact matches | Supporting curriculum; shared brief | hybrid | How can common-term phrase queries be evaluated without materializing huge posting lists? | `bounded-work` | `correctness-contract`, `snapshot-consistency` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR025 | Choose scheduling algorithms from objective and constraints | Proposed home | analytical | When is earliest-finish greedy optimal, and when must scheduling become a weighted optimization problem? | `correctness-contract` | None | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR026 | Select data structures from operation mix and failure semantics | Proposed home | hybrid | Which data structure fits the update/query workload while retaining the required correctness and resource bounds? | `correctness-contract` | `bounded-work`, `evaluation-mismatch` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR027 | Maintain exact inventory range totals under point updates | Proposed home | analytical | Which update and snapshot contract makes range totals correct and efficient? | `snapshot-consistency` | `correctness-contract` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR028 | Bound request amplification before expensive parsing and validation | Supporting curriculum; shared brief | incident | Which per-request limits and algorithm changes stop CPU exhaustion without rejecting legitimate traffic unnecessarily? | `bounded-work` | `resource-admission`, `incident-postmortem` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR030 | Diagnose production failures from violated algorithm invariants | Supporting curriculum; shared brief | incident | Which invariant failed, how can it be contained, and which replay would prevent recurrence? | `incident-postmortem` | `bounded-work`, `correctness-contract` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR031 | Load authorized hierarchies with bounded database work | Supporting curriculum; shared brief | incident | Should a subtree API use a recursive CTE or bounded level-order batching for its workload? | `bounded-work` | `authorization-scope`, `snapshot-consistency` | not-measured | proposed | 2026-11-06 |  |
| PLAN-2026-IR035 | Separate live rank approximation from exact competition settlement | Supporting curriculum; shared brief | hybrid | How should score events support fast display ranks and a reproducible final prize result? | `snapshot-consistency` | `idempotent-side-effects`, `correctness-contract` | not-measured | proposed | 2026-11-06 |  |

### Reference and transfer notes

| Planning ID | Example reference only | Scope boundary |
|---|---|---|
| PLAN-2026-IR018 | [REF-IR018](../../../../INTERVIEW_CASES/DSA/Algo_Design_Problems/CASE_STUDIES_01.md) | Seven overlapping teaching units; source measurements are not independently verified and should not be added together |
| PLAN-2026-IR019 | [REF-IR019](../../../../INTERVIEW_CASES/DSA/Algo_Design_Problems/CASE_STUDIES_02.md) | The OCR filename and ordinal ranges disagree; arbitrary traffic overlays do not preserve ordinary CH correctness |
| PLAN-2026-IR022 | [REF-IR022](../../../../INTERVIEW_CASES/DSA/Algo_Design_Problems/Rate_Limiting/distributed_token_buckets.md) | Periodic usage reporting alone is not coordinated allocation; token buckets and exact rolling windows have different contracts |
| PLAN-2026-IR023 | [REF-IR023](../../../../INTERVIEW_CASES/DSA/Algo_Design_Problems/Routing/closure_aware_shortest_paths.md) | Finding a legal route after filtering does not prove completeness or optimality; richer time/turn state needs a separate contract |
| PLAN-2026-IR024 | [REF-IR024](../../../../INTERVIEW_CASES/DSA/Algo_Design_Problems/Search/phrase_search_postings.md) | Rarity affects work, not semantic order; deleted documents and analyzer versions must share a consistent snapshot |
| PLAN-2026-IR025 | [REF-IR025](../../../../INTERVIEW_CASES/DSA/index_basics/Greedy_Scheduling/activity_selection.md) | The reference proves equal-value single-room cardinality only; warehouse jobs and staffing need their own constraints |
| PLAN-2026-IR026 | [REF-IR026](../../../../INTERVIEW_CASES/DSA/index_basics/INTERVIEW_CASE_STUDIES.md) | Eight teaching units plus rapid prompts; approximate structures, local proofs and service guarantees must be distinguished |
| PLAN-2026-IR027 | [REF-IR027](../../../../INTERVIEW_CASES/DSA/index_basics/Range_Queries/fenwick_inventory_totals.md) | Fixed ordering and one logical writer are reference assumptions; a Fenwick tree alone does not provide distributed stock correctness |
| PLAN-2026-IR028 | [REF-IR028](../../../../INTERVIEW_CASES/DSA/Production_DSA_Incidents/Algorithmic_DoS/hash_and_regex_resource_exhaustion.md) | The hash and regex scenarios are separate; legacy runtime behavior and example input limits require fresh validation |
| PLAN-2026-IR030 | [REF-IR030](../../../../INTERVIEW_CASES/DSA/Production_DSA_Incidents/INTERVIEW_CASE_STUDIES.md) | Ten reconstructed incident units include recurring mechanisms; reported timings are not measured outcomes of this library |
| PLAN-2026-IR031 | [REF-IR031](../../../../INTERVIEW_CASES/DSA/Production_DSA_Incidents/Tree_Queries/policy_hierarchy_n_plus_one.md) | One SQL statement can still do excessive work; an outer LIMIT is not a universal recursion work bound |
| PLAN-2026-IR035 | [REF-IR035](../../../../INTERVIEW_CASES/DSA/Sys_Des_DSA/Leaderboards/score_bucket_rank_consistency.md) | A sum of stale counters is not an exact settlement snapshot; tie semantics and late-event policy must be specified |

Case metadata, pattern definitions, supporting/contrasting references, structured backlog, review log and authoring checks for these additions are maintained in the linked central planning catalog. Existing planned rows elsewhere in this index retain their original status. A brief may be split or merged before canonical case IDs are assigned.

<!-- END INTERVIEW REFERENCE PLAN -->
