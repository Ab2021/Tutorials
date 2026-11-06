# Module 09 Cheat Sheets â€” Data Streaming, Event Systems and Real-Time Analytics

## Authoring standard

Every authored sheet must follow the [course authoring and numerical standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked cheat-sheet master: all 12 sections, derivations/invariants, checked numerical examples, at least four diagnostic failures, 10–15 answered interview questions, and the final Quick Reference Card. This catalog contains one unreviewed draft; expand it to all 11 collection-index sections and a final Common Issues artifact (at least 25 complete issues) before publication. A card alone is not a completed sheet.

## Cheat sheets and planned topics

| # | Reference | Coverage | Status |
|---:|---|---|---|
| 1 | event schema checklist | Concise patterns, decision rules, assumptions, and common traps. | planned |
| 2 | streaming semantics | Concise patterns, decision rules, assumptions, and common traps. | planned |
| 3 | windowing patterns | Concise patterns, decision rules, assumptions, and common traps. | planned |
| 4 | [Kafka consumer partitions and lag](01_Kafka_Consumer_Partitions_and_Lag.md) | Partition ownership and ordering, safe offsets, skew, recovery, replay and diagnostics. | draft; unreviewed |

The Kafka sheet is an illustrative, unreviewed draft; the source inventory is not itself evidence of an incident. Its [quick reference card](01_Kafka_Consumer_Partitions_and_Lag.md#12-quick-reference-card) and [diagnostic cases](01_Kafka_Consumer_Partitions_and_Lag.md#8-failure-modes--edge-cases) are available. Other catalog entries remain planned.

## Suggested file naming

Use **01_reference-slug.md**, then **02_reference-slug.md**. Keep one concern per file so references remain scannable.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Related brief | Proposed cheat-sheet scope | Checks / counterexamples to include | Reference only | Status |
|---|---|---|---|---|
| PLAN-2026-IR035 | Score buckets; Fenwick count summaries; deterministic ties; deduplicated events; cross-shard moves; closing watermarks | Reference sorted snapshots, duplicate/late events, negative scores, bucket crossings and exact final-rank replay. Boundary: A sum of stale counters is not an exact settlement snapshot; tie semantics and late-event policy must be specified. | [REF-IR035](../../../../INTERVIEW_CASES/DSA/Sys_Des_DSA/Leaderboards/score_bucket_rank_consistency.md) | planned; not written |

Each future cheat sheet needs its own explanation, preconditions, formulas or algorithms where relevant, failure signals and checked examples. Do not copy the interview narrative or its scenario figures into a reference sheet.

<!-- END INTERVIEW REFERENCE PLAN -->
