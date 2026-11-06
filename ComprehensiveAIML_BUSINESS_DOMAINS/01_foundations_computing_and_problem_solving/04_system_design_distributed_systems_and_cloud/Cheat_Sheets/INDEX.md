# Module 04 Cheat Sheets â€” System Design, Distributed Systems and Cloud

## Authoring standard

Every authored sheet must follow the [course authoring and numerical standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked cheat-sheet master: all 12 sections, derivations/invariants, checked numerical examples, at least four diagnostic failures, 10–15 answered interview questions, and the final Quick Reference Card. This is a planned catalog, not completed sheets; expand to all 11 collection-index sections and a final Common Issues artifact (at least 25 complete issues) before publication. A card alone is not a completed sheet.

## Planned cheat sheets

| # | Reference | Coverage | Status |
|---:|---|---|---|
| 1 | system-design interview flow | Concise patterns, decision rules, assumptions, and common traps. | planned |
| 2 | capacity formulas | Concise patterns, decision rules, assumptions, and common traps. | planned |
| 3 | distributed-systems trade-offs | Concise patterns, decision rules, assumptions, and common traps. | planned |

## Suggested file naming

Use **01_reference-slug.md**, then **02_reference-slug.md**. Keep one concern per file so references remain scannable.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Related brief | Proposed cheat-sheet scope | Checks / counterexamples to include | Reference only | Status |
|---|---|---|---|---|
| PLAN-2026-IR018 | CH and A* assumptions; rolling-window semantics; clock skew; positional postings; BM25 statistics | Reference Dijkstra comparisons, admissible-heuristic checks, concurrent window simulations and brute-force phrase/ranking oracles. Boundary: Seven overlapping teaching units; source measurements are not independently verified and should not be added together. | [REF-IR018](../../../../INTERVIEW_CASES/DSA/Algo_Design_Problems/CASE_STUDIES_01.md) | planned; not written |
| PLAN-2026-IR019 | Coherent traffic snapshots; CH compatibility; fallback Dijkstra; map matching; route versus ETA objectives | Base-graph shortest-path oracle, closure and turn-restriction replays, ETA error and routing tail latency. Boundary: The OCR filename and ordinal ranges disagree; arbitrary traffic overlays do not preserve ordinary CH correctness. | [REF-IR019](../../../../INTERVIEW_CASES/DSA/Algo_Design_Problems/CASE_STUDIES_02.md) | planned; not written |
| PLAN-2026-IR020 | Routing production behavior; coordinated rolling counters; distributed positional inverted indexes | Coherent-snapshot queries, concurrent admission accounting, shard failure tests and search-reference comparisons. Boundary: Three units continue earlier routing, admission and search topics; this is not a new independent evidence dataset. | [REF-IR020](../../../../INTERVIEW_CASES/DSA/Algo_Design_Problems/CASE_STUDIES_03.md) | planned; not written |
| PLAN-2026-IR022 | Token-bucket invariants; atomic refill and debit; leased budgets; expiry; failure policy | Aggregate-admission histories, refill boundaries, coordinator outages, lease races and rejected-request latency. Boundary: Periodic usage reporting alone is not coordinated allocation; token buckets and exact rolling windows have different contracts. | [REF-IR022](../../../../INTERVIEW_CASES/DSA/Algo_Design_Problems/Rate_Limiting/distributed_token_buckets.md) | planned; not written |
| PLAN-2026-IR033 | Single-flight scope; jitter; fill admission; consistent hashing; old/new ring epochs; version fences and tombstones | Cold-fleet and skewed-key load replays, database concurrency, invalidation lag and stale-fill/delete races. Boundary: Local single-flight is not fleet-wide coordination; expected key movement is not the same as request miss share. | [REF-IR033](../../../../INTERVIEW_CASES/DSA/Sys_Des_DSA/Cache_Resilience/cold_cache_and_ring_migration.md) | planned; not written |
| PLAN-2026-IR034 | Profile caching; safe typeahead publication; ranked score updates; Bloom-filter crawler admission | Failure replays and reference histories; freshness, exact versus approximate results and false-positive consequences. Boundary: Four teaching units overlap focused examples; Bloom false positives and stale ranks need product-specific tolerance. | [REF-IR034](../../../../INTERVIEW_CASES/DSA/Sys_Des_DSA/INTERVIEW_SYSTEM_DESIGN_CASES.md) | planned; not written |
| PLAN-2026-IR035 | Score buckets; Fenwick count summaries; deterministic ties; deduplicated events; cross-shard moves; closing watermarks | Reference sorted snapshots, duplicate/late events, negative scores, bucket crossings and exact final-rank replay. Boundary: A sum of stale counters is not an exact settlement snapshot; tie semantics and late-event policy must be specified. | [REF-IR035](../../../../INTERVIEW_CASES/DSA/Sys_Des_DSA/Leaderboards/score_bucket_rank_consistency.md) | planned; not written |

Each future cheat sheet needs its own explanation, preconditions, formulas or algorithms where relevant, failure signals and checked examples. Do not copy the interview narrative or its scenario figures into a reference sheet.

<!-- END INTERVIEW REFERENCE PLAN -->
