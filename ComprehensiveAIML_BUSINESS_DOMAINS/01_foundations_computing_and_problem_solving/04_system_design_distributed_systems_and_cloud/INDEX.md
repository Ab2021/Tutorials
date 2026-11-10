# Module 04 â€” System Design, Distributed Systems and Cloud

> **Part:** [Part 01 â€” Foundations, Computing and Problem Solving](../INDEX.md)  
> **Prerequisites:** 02,03  
> **Purpose:** Build reliable services by understanding data flow, consistency, scaling, caching, queues, and failure recovery.

## Topics to develop

| # | Topic | Status |
|---:|---|---|
| 1 | requirements and capacity | [System-design flow draft](Cheat_Sheets/01_System_Design_Interview_Flow.md); remaining capacity scope planned |
| 2 | storage and caching | planned |
| 3 | queues and eventing | planned |
| 4 | consistency and idempotency | planned |
| 5 | resilience and observability | planned |

## What competence looks like

After this module, a learner can turn a real problem into an appropriate technical approach, state the relevant assumptions and trade-offs, select evaluation measures, and explain where the method can fail. The module's case studies focus on applied judgement; its cheat sheets capture reusable operating knowledge.

## Collections

- [Case Studies](./Case_Studies/INDEX.md): application briefs that deliberately include data, constraints, architecture, evaluation, risk, and operating plan.
- [Cheat Sheets](./Cheat_Sheets/INDEX.md): compact references for repeat use while designing, reviewing, or debugging.

## Authored course-topic drafts

- [System Design Interview Flow](Cheat_Sheets/01_System_Design_Interview_Flow.md) — requirements, contracts, sizing, failure recovery and validation; visible source titles are inventory only, original bodies unavailable.

## Canonical case studies

- [Design a Notification Platform](Case_Studies/01_Design_Notification_Platform.md) — illustrative idempotent-delivery decision; draft.
- [Design a Feature-Serving API](Case_Studies/02_Design_Feature_Serving_API.md) — illustrative point-in-time serving decision; draft.

## Dependency notes

Prerequisites above are module identifiers. They are the minimum conceptual background, not a requirement to complete every earlier part. For a fast path, read the prerequisite indexes, then complete one module case before advancing.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Planning ID | Detailed topics to develop | Decision to address | Validation to design |
|---|---|---|---|
| PLAN-2026-IR018 | CH and A* assumptions; rolling-window semantics; clock skew; positional postings; BM25 statistics | Which guarantees are needed for road routing, distributed admission and phrase retrieval? | Reference Dijkstra comparisons, admissible-heuristic checks, concurrent window simulations and brute-force phrase/ranking oracles |
| PLAN-2026-IR019 | Coherent traffic snapshots; CH compatibility; fallback Dijkstra; map matching; route versus ETA objectives | How should a navigation service publish legal optimal routes while separately evaluating predicted arrival time? | Base-graph shortest-path oracle, closure and turn-restriction replays, ETA error and routing tail latency |
| PLAN-2026-IR020 | Routing production behavior; coordinated rolling counters; distributed positional inverted indexes | Which correctness and failure guarantees survive sharding, delayed updates and partial outages? | Coherent-snapshot queries, concurrent admission accounting, shard failure tests and search-reference comparisons |
| PLAN-2026-IR022 | Token-bucket invariants; atomic refill and debit; leased budgets; expiry; failure policy | Which admission policy remains within its global budget during contention or coordinator failure? | Aggregate-admission histories, refill boundaries, coordinator outages, lease races and rejected-request latency |
| PLAN-2026-IR033 | Single-flight scope; jitter; fill admission; consistent hashing; old/new ring epochs; version fences and tombstones | How should cache fills and ring migration preserve freshness while limiting database amplification? | Cold-fleet and skewed-key load replays, database concurrency, invalidation lag and stale-fill/delete races |
| PLAN-2026-IR034 | Profile caching; safe typeahead publication; ranked score updates; Bloom-filter crawler admission | Which cache, typeahead, leaderboard and deduplication contracts remain valid during refresh and failure? | Failure replays and reference histories; freshness, exact versus approximate results and false-positive consequences |
| PLAN-2026-IR035 | Score buckets; Fenwick count summaries; deterministic ties; deduplicated events; cross-shard moves; closing watermarks | How should score events support fast display ranks and a reproducible final prize result? | Reference sorted snapshots, duplicate/late events, negative scores, bucket crossings and exact final-rank replay |

See the local Case_Studies and Cheat_Sheets indexes for proposed outputs and source-reference links. These scope additions do not replace the original planned topics.

<!-- END INTERVIEW REFERENCE PLAN -->
