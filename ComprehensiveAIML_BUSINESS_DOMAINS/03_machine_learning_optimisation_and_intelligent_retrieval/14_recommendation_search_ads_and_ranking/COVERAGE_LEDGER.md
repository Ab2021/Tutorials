# Module 14 exact authoring ledger

State: registered before individual authoring, 2026-11-08. Owner: module 14. Editorial state starts planned; no review, execution, deployment or measured outcome is implied. Parent owns shared registers and Part 11 source teaching homes.

| Commitment | Stable artifact | Outcome and required distinctions | Numerical oracle | Initial state |
|---|---|---|---|---|
| Original topic 1 candidate generation; original sheet recsys architectures | CS14-01 `Cheat_Sheets/01_Candidate_Generation_and_Architectures.md` | Eligible multi-source candidates; exact/approximate retrieval; stage budgets; cold start | Set union, eligibility, recall ceiling, vector storage | planned |
| Original topic 2 learning to rank | CS14-02 `Cheat_Sheets/02_Learning_to_Rank.md` | Point/pair/list losses; position bias; ad utility versus click proxy | Pairwise logistic loss, discounted gain | planned |
| Original topic 3 collaborative and content models | CS14-03 `Cheat_Sheets/03_Collaborative_and_Content_Models.md` | Explicit/implicit feedback; factors; content/hybrid; negative sampling | Dot products and regularized error | planned |
| Original topic 4 bandits and exploration | CS14-04 `Cheat_Sheets/04_Bandits_and_Exploration.md` | Exploration support, reward windows, safe eligibility, contextual allocation | Epsilon mixture and IPS | planned |
| Original topic 5 offline-online evaluation; original sheets ranking metrics and online experimentation checklist | CS14-05 `Cheat_Sheets/05_Ranking_Metrics_and_Online_Experimentation.md` | Precision/recall/NDCG; randomization/exposure/SRM; interference and delayed outcomes; full operational checklist | Precision 3/5 and recall 3/4; user-level rate difference | planned |
| PLAN-2026-IR018 units 1–2; IR020 routing unit | CS14-06 `Cheat_Sheets/06_Road_Routing_Correctness.md` | Dijkstra; admissible A*; CH witness/shortcut assumptions; coherent closures/weights; ETA separate | Directed shortest-path and admissibility oracle | planned |
| PLAN-2026-IR018 units 3–5; IR020 rolling counter unit | CS14-07 `Cheat_Sheets/07_Distributed_Window_Admission.md` | Exact rolling logs versus approximate counters; atomic admission; coordinator failure; clock skew | Half-open window history and concurrent debit oracle | planned |
| PLAN-2026-IR018 units 6–7; IR020 positional index unit; IR021 all four units | CS14-08 `Cheat_Sheets/08_Authorized_Search_Indexing_and_TopK.md`; CASE-2026-M14-03 | WAL, immutable segments, snapshots, positional fields, BM25/BM25F distinctions, exact distributed top-k, ACLs, tombstones, capacity | Brute force authorized ranking, delete visibility and replay | planned |
| PLAN-2026-IR024 | CS14-09 `Cheat_Sheets/09_Lazy_Positional_Phrase_Search.md`; CASE-2026-M14-04 | Rarest anchor is execution order not phrase order; lazy seek, skips, repeated terms, analyzers | Token-position phrase oracle | planned |
| PLAN-2026-IR029 | CS14-10 `Cheat_Sheets/10_Bounded_Graph_Candidate_Traversal.md` | Enqueue-time deduplication; page before materialization; independent node/edge/byte/time budgets; deterministic partial result contract | Cyclic and high-degree adjacency replay | planned |
| PLAN-2026-IR032 | CS14-11 `Cheat_Sheets/11_Autocomplete_Admission_and_Freshness.md`; CASE-2026-M14-05 | Sketches do not enumerate; bounded candidates; window rollover; trie admission; two generations; overlay; versioned CDN; abuse | Collision/expiry and memory/freshness oracle | planned |
| Original case E-commerce recommender | CASE-2026-M14-01 `Case_Studies/01_Ecommerce_Recommender.md` | Choose eligible hybrid retrieval/ranking under stock and evidence constraints | Candidate recall and useful conversion | planned |
| Original case search ranking quality regression | CASE-2026-M14-02 `Case_Studies/02_Search_Ranking_Quality_Regression.md` | Distinguish retrieval, scoring, analyzer, eligibility and measurement regressions | Fixed candidate versus changed candidate replay | planned |
| Final troubleshooting requirement | CS14-12 `Cheat_Sheets/12_25_Common_Issues.md` | 25 distinct complete diagnostic issues, 12 sections, answered interview pack | Multiple subsystem diagnostic traces | planned |

## Source scope and boundaries

- [Inventory](../../../Templates/directory.txt): relevant visible recommendation/search source families are reference context; exact Part 11 homes remain in the [crosswalk](../../00_SOURCE_TOPIC_CROSSWALK.md) and [data-science briefs](../../00_SOURCE_COVERAGE_DATA_SCIENCE.md#ds-cases-08).
- [Focused reference plan](../../00_REFERENCE_INFORMED_CASE_PLAN.md#ref-ir018-specify-correctness-contracts-before-choosing-production-algorithms) extends the original five topics. IR018/020 have supporting/shared cases in other homes; their distinct mechanisms receive full module-local sheets, not a survey replacement.
- IR029 incident narrative belongs its canonical graph/incident owner; module 14 authors the recommendation traversal method and explicit API guarantees, not a fabricated duplicate production incident.
- Original three cheat-sheet rows are preserved as distinct catalog commitments even when mapped to an integrated artifact with the matching reusable outcome.
- Original source availability and source measurements must be disclosed per artifact. New workload values are illustrative, not source benchmarks.

## Publication gates

Every individual artifact requires at least 1000 substantive lines, correct ordered sections, a final card, concrete source links, checked hand arithmetic, sensitivity, distinct diagnostics and complete answered interviews. Draft is not reviewed or published. Independent structural, numerical and navigation checks are delegated to parent after all authors finish; no execution is claimed here.
