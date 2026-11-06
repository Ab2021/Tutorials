# Module 14 Cheat Sheets â€” Recommendation, Search, Ads and Ranking

## Authoring standard

Every authored sheet must follow the [course authoring and numerical standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked cheat-sheet master: all 12 sections, derivations/invariants, checked numerical examples, at least four diagnostic failures, 10–15 answered interview questions, and the final Quick Reference Card. This is a planned catalog, not completed sheets; expand to all 11 collection-index sections and a final Common Issues artifact (at least 25 complete issues) before publication. A card alone is not a completed sheet.

## Planned cheat sheets

| # | Reference | Coverage | Status |
|---:|---|---|---|
| 1 | ranking metrics | Concise patterns, decision rules, assumptions, and common traps. | planned |
| 2 | recsys architectures | Concise patterns, decision rules, assumptions, and common traps. | planned |
| 3 | online experimentation checklist | Concise patterns, decision rules, assumptions, and common traps. | planned |

## Suggested file naming

Use **01_reference-slug.md**, then **02_reference-slug.md**. Keep one concern per file so references remain scannable.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Related brief | Proposed cheat-sheet scope | Checks / counterexamples to include | Reference only | Status |
|---|---|---|---|---|
| PLAN-2026-IR018 | CH and A* assumptions; rolling-window semantics; clock skew; positional postings; BM25 statistics | Reference Dijkstra comparisons, admissible-heuristic checks, concurrent window simulations and brute-force phrase/ranking oracles. Boundary: Seven overlapping teaching units; source measurements are not independently verified and should not be added together. | [REF-IR018](../../../../INTERVIEW_CASES/DSA/Algo_Design_Problems/CASE_STUDIES_01.md) | planned; not written |
| PLAN-2026-IR020 | Routing production behavior; coordinated rolling counters; distributed positional inverted indexes | Coherent-snapshot queries, concurrent admission accounting, shard failure tests and search-reference comparisons. Boundary: Three units continue earlier routing, admission and search topics; this is not a new independent evidence dataset. | [REF-IR020](../../../../INTERVIEW_CASES/DSA/Algo_Design_Problems/CASE_STUDIES_03.md) | planned; not written |
| PLAN-2026-IR021 | WAL and immutable segments; positional phrase queries; field-weighted BM25; exact top-k; ACLs and tombstones | Brute-force Boolean/phrase and global top-k oracles, delete visibility, retrieval quality, index lag and merge capacity. Boundary: Exact global top-k needs a valid distributed protocol; figures in the source are scenario claims, not search-cluster benchmarks. | [REF-IR021](../../../../INTERVIEW_CASES/DSA/Algo_Design_Problems/CASE_STUDIES_04.md) | planned; not written |
| PLAN-2026-IR024 | Rarest-term anchor; lazy postings; skips; relative positions; repeated terms; matching analyzers | Brute-force token-position oracle, peak memory and tail latency over frequent and repeated-term phrases. Boundary: Rarity affects work, not semantic order; deleted documents and analyzer versions must share a consistent snapshot. | [REF-IR024](../../../../INTERVIEW_CASES/DSA/Algo_Design_Problems/Search/phrase_search_postings.md) | planned; not written |
| PLAN-2026-IR029 | Enqueue-time deduplication; paged adjacency; independent work budgets; deterministic traversal; explicit partial results | High-degree and cyclic graph replays, peak memory, examined edges, output completeness and candidate utility. Boundary: Depth and output-size limits alone do not bound expansion; source product degrees and outage figures are illustrative claims. | [REF-IR029](../../../../INTERVIEW_CASES/DSA/Production_DSA_Incidents/Graph_Traversal/bounded_bfs_incident.md) | planned; not written |
| PLAN-2026-IR032 | Trie admission; time-window sketches; bounded candidate stores; immutable snapshots; real-time overlays; CDN versioning | Memory including two generations, suggestion relevance, freshness lag, abusive-query tests and per-locale tail latency. Boundary: Sketches cannot enumerate terms; approximate frequencies and novelty thresholds must be evaluated rather than copied. | [REF-IR032](../../../../INTERVIEW_CASES/DSA/Sys_Des_DSA/Autocomplete/frequency_admission_and_freshness.md) | planned; not written |

Each future cheat sheet needs its own explanation, preconditions, formulas or algorithms where relevant, failure signals and checked examples. Do not copy the interview narrative or its scenario figures into a reference sheet.

<!-- END INTERVIEW REFERENCE PLAN -->
