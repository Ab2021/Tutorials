# Module 03 Cheat Sheets â€” Data Structures, Algorithms and Computational Thinking

## Authoring standard

Every authored sheet must follow the [course authoring and numerical standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked cheat-sheet master: all 12 sections, derivations/invariants, checked numerical examples, at least four diagnostic failures, 10–15 answered interview questions, and the final Quick Reference Card. This collection has one unreviewed course-topic draft; remaining sheets and the final Common Issues artifact (at least 25 complete issues) are planned. A card alone is not a completed sheet.

## Planned cheat sheets

| # | Reference | Coverage | Status |
|---:|---|---|---|
| 1 | [DSA patterns and invariants](01_DSA_Patterns_and_Invariants.md) | Pattern selection, complexity assumptions, correctness invariants and oracle tests. | Draft; unreviewed. |
| 2 | complexity reference | Concise patterns, decision rules, assumptions, and common traps. | planned |
| 3 | interview problem selection | Concise patterns, decision rules, assumptions, and common traps. | planned |

## Suggested file naming

Use **01_reference-slug.md**, then **02_reference-slug.md**. Keep one concern per file so references remain scannable.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Related brief | Proposed cheat-sheet scope | Checks / counterexamples to include | Reference only | Status |
|---|---|---|---|---|
| PLAN-2026-IR018 | CH and A* assumptions; rolling-window semantics; clock skew; positional postings; BM25 statistics | Reference Dijkstra comparisons, admissible-heuristic checks, concurrent window simulations and brute-force phrase/ranking oracles. Boundary: Seven overlapping teaching units; source measurements are not independently verified and should not be added together. | [REF-IR018](../../../../INTERVIEW_CASES/DSA/Algo_Design_Problems/CASE_STUDIES_01.md) | planned; not written |
| PLAN-2026-IR019 | Coherent traffic snapshots; CH compatibility; fallback Dijkstra; map matching; route versus ETA objectives | Base-graph shortest-path oracle, closure and turn-restriction replays, ETA error and routing tail latency. Boundary: The OCR filename and ordinal ranges disagree; arbitrary traffic overlays do not preserve ordinary CH correctness. | [REF-IR019](../../../../INTERVIEW_CASES/DSA/Algo_Design_Problems/CASE_STUDIES_02.md) | planned; not written |
| PLAN-2026-IR022 | Token-bucket invariants; atomic refill and debit; leased budgets; expiry; failure policy | Aggregate-admission histories, refill boundaries, coordinator outages, lease races and rejected-request latency. Boundary: Periodic usage reporting alone is not coordinated allocation; token buckets and exact rolling windows have different contracts. | [REF-IR022](../../../../INTERVIEW_CASES/DSA/Algo_Design_Problems/Rate_Limiting/distributed_token_buckets.md) | planned; not written |
| PLAN-2026-IR023 | Shortcut expansion; closure snapshots; CH compatibility; directed nonnegative Dijkstra fallback | Nested-shortcut and adversarial-closure queries compared with base Dijkstra, including unreachable pairs. Boundary: Finding a legal route after filtering does not prove completeness or optimality; richer time/turn state needs a separate contract. | [REF-IR023](../../../../INTERVIEW_CASES/DSA/Algo_Design_Problems/Routing/closure_aware_shortest_paths.md) | planned; not written |
| PLAN-2026-IR024 | Rarest-term anchor; lazy postings; skips; relative positions; repeated terms; matching analyzers | Brute-force token-position oracle, peak memory and tail latency over frequent and repeated-term phrases. Boundary: Rarity affects work, not semantic order; deleted documents and analyzer versions must share a consistent snapshot. | [REF-IR024](../../../../INTERVIEW_CASES/DSA/Algo_Design_Problems/Search/phrase_search_postings.md) | planned; not written |
| PLAN-2026-IR025 | Half-open intervals; deterministic ties; exchange proof; weighted interval DP boundary | Exhaustive small-instance optimum comparisons and counterexamples for weighted or buffered variants. Boundary: The reference proves equal-value single-room cardinality only; warehouse jobs and staffing need their own constraints. | [REF-IR025](../../../../INTERVIEW_CASES/DSA/index_basics/Greedy_Scheduling/activity_selection.md) | planned; not written |
| PLAN-2026-IR026 | Cache policies; ranks/range queries; sketches; prefix/nearest-neighbor search; graphs; scheduling; online windows | Operation traces, exact small-data oracles, measured memory/latency and stated approximation or distributed guarantees. Boundary: Eight teaching units plus rapid prompts; approximate structures, local proofs and service guarantees must be distinguished. | [REF-IR026](../../../../INTERVIEW_CASES/DSA/index_basics/INTERVIEW_CASE_STUDIES.md) | planned; not written |
| PLAN-2026-IR027 | Fenwick tree lowbit traversal; inclusive ranges; point deltas; wide arithmetic; atomic visibility | Randomized update/query comparisons with a simple array, boundary/overflow checks and update-visibility tests. Boundary: Fixed ordering and one logical writer are reference assumptions; a Fenwick tree alone does not provide distributed stock correctness. | [REF-IR027](../../../../INTERVIEW_CASES/DSA/index_basics/Range_Queries/fenwick_inventory_totals.md) | planned; not written |
| PLAN-2026-IR028 | Hash collision amplification; catastrophic regex backtracking; pre-parse limits; bounded queues and downstream budgets | Adversarial near-match inputs, collision workload, CPU per request, timeout cascades and legitimate-input acceptance. Boundary: The hash and regex scenarios are separate; legacy runtime behavior and example input limits require fresh validation. | [REF-IR028](../../../../INTERVIEW_CASES/DSA/Production_DSA_Incidents/Algorithmic_DoS/hash_and_regex_resource_exhaustion.md) | planned; not written |
| PLAN-2026-IR030 | Hash flooding; BFS OOM; regex backtracking; recursive DFS depth; ORM N+1; overflow; unstable ties; floating search termination | Reproduce each mechanism separately with bounded adversarial traces; track latency, memory, database calls and deterministic outputs. Boundary: Ten reconstructed incident units include recurring mechanisms; reported timings are not measured outcomes of this library. | [REF-IR030](../../../../INTERVIEW_CASES/DSA/Production_DSA_Incidents/INTERVIEW_CASE_STUDIES.md) | planned; not written |
| PLAN-2026-IR031 | N+1 detection; adjacency-list indexing; per-step authorization; cycle checks; stable assembly; snapshot reads | Reference traversal comparisons, round trips, rows scanned, unauthorized descendants, deep/wide trees and truncation behavior. Boundary: One SQL statement can still do excessive work; an outer LIMIT is not a universal recursion work bound. | [REF-IR031](../../../../INTERVIEW_CASES/DSA/Production_DSA_Incidents/Tree_Queries/policy_hierarchy_n_plus_one.md) | planned; not written |
| PLAN-2026-IR035 | Score buckets; Fenwick count summaries; deterministic ties; deduplicated events; cross-shard moves; closing watermarks | Reference sorted snapshots, duplicate/late events, negative scores, bucket crossings and exact final-rank replay. Boundary: A sum of stale counters is not an exact settlement snapshot; tie semantics and late-event policy must be specified. | [REF-IR035](../../../../INTERVIEW_CASES/DSA/Sys_Des_DSA/Leaderboards/score_bucket_rank_consistency.md) | planned; not written |

Each future cheat sheet needs its own explanation, preconditions, formulas or algorithms where relevant, failure signals and checked examples. Do not copy the interview narrative or its scenario figures into a reference sheet.

<!-- END INTERVIEW REFERENCE PLAN -->
