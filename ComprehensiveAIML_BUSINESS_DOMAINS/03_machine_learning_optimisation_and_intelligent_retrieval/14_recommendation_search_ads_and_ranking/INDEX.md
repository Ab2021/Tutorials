# Module 14 â€” Recommendation, Search, Ads and Ranking

> **Part:** [Part 03 â€” Machine Learning, Optimisation and Intelligent Retrieval](../INDEX.md)  
> **Prerequisites:** 07,11,12  
> **Purpose:** Design retrieval-and-ranking systems that balance relevance, business objectives, exploration, and fairness.

## Topics to develop

| # | Topic | Status |
|---:|---|---|
| 1 | candidate generation | planned |
| 2 | learning to rank | planned |
| 3 | collaborative and content models | planned |
| 4 | bandits and exploration | planned |
| 5 | offline-online evaluation | planned |

## What competence looks like

After this module, a learner can turn a real problem into an appropriate technical approach, state the relevant assumptions and trade-offs, select evaluation measures, and explain where the method can fail. The module's case studies focus on applied judgement; its cheat sheets capture reusable operating knowledge.

## Collections

- [Case Studies](./Case_Studies/INDEX.md): application briefs that deliberately include data, constraints, architecture, evaluation, risk, and operating plan.
- [Cheat Sheets](./Cheat_Sheets/INDEX.md): compact references for repeat use while designing, reviewing, or debugging.

## Dependency notes

Prerequisites above are module identifiers. They are the minimum conceptual background, not a requirement to complete every earlier part. For a fast path, read the prerequisite indexes, then complete one module case before advancing.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Planning ID | Detailed topics to develop | Decision to address | Validation to design |
|---|---|---|---|
| PLAN-2026-IR018 | CH and A* assumptions; rolling-window semantics; clock skew; positional postings; BM25 statistics | Which guarantees are needed for road routing, distributed admission and phrase retrieval? | Reference Dijkstra comparisons, admissible-heuristic checks, concurrent window simulations and brute-force phrase/ranking oracles |
| PLAN-2026-IR020 | Routing production behavior; coordinated rolling counters; distributed positional inverted indexes | Which correctness and failure guarantees survive sharding, delayed updates and partial outages? | Coherent-snapshot queries, concurrent admission accounting, shard failure tests and search-reference comparisons |
| PLAN-2026-IR021 | WAL and immutable segments; positional phrase queries; field-weighted BM25; exact top-k; ACLs and tombstones | How should search remain correct across indexing, deletion, ranking and shard failures? | Brute-force Boolean/phrase and global top-k oracles, delete visibility, retrieval quality, index lag and merge capacity |
| PLAN-2026-IR024 | Rarest-term anchor; lazy postings; skips; relative positions; repeated terms; matching analyzers | How can common-term phrase queries be evaluated without materializing huge posting lists? | Brute-force token-position oracle, peak memory and tail latency over frequent and repeated-term phrases |
| PLAN-2026-IR029 | Enqueue-time deduplication; paged adjacency; independent work budgets; deterministic traversal; explicit partial results | How should a recommendation API respond when graph expansion reaches a node, edge, memory or time limit? | High-degree and cyclic graph replays, peak memory, examined edges, output completeness and candidate utility |
| PLAN-2026-IR032 | Trie admission; time-window sketches; bounded candidate stores; immutable snapshots; real-time overlays; CDN versioning | Which terms enter a bounded suggestion index, and how can safe new trends appear promptly? | Memory including two generations, suggestion relevance, freshness lag, abusive-query tests and per-locale tail latency |

See the local Case_Studies and Cheat_Sheets indexes for proposed outputs and source-reference links. These scope additions do not replace the original planned topics.

<!-- END INTERVIEW REFERENCE PLAN -->
