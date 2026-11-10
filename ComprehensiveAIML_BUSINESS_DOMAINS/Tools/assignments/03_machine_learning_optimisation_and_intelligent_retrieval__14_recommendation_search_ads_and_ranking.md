# Unit: 03_machine_learning_optimisation_and_intelligent_retrieval/14_recommendation_search_ads_and_ranking

## Current state
- case files present (1): 01_e_commerce_recommender.md
- sheet files present (7): 01_ranking_metrics.md, 02_recsys_architectures.md, 03_online_experimentation_checklist.md, 04_retrieval_index_correctness.md, 05_phrase_and_positional_retrieval.md, 06_autocomplete_admission_and_freshness.md, 07_25_Common_Issues.md
- `Case_Studies/INDEX.md` sections: 0/7
- `Cheat_Sheets/INDEX.md` sections: 0/11

## Required new CASE manuscripts
- `Case_Studies/08_Specify_Correctness_Contracts_Before_Choosing_Production_Algorithms.md`  ← decision: Specify correctness contracts before choosing production algorithms
- `Case_Studies/09_Translate_Local_Algorithm_Guarantees_Into_Distributed_Service_Contract.md`  ← decision: Translate local algorithm guarantees into distributed service contracts
- `Case_Studies/10_Evolve_An_Authorized_Near_Real_Time_Search_Index_Safely.md`  ← decision: Evolve an authorized near-real-time search index safely
- `Case_Studies/11_Bound_Phrase_Search_Memory_While_Preserving_Exact_Matches.md`  ← decision: Bound phrase-search memory while preserving exact matches
- `Case_Studies/12_Return_Useful_Graph_Candidates_Under_Explicit_Traversal_Budgets.md`  ← decision: Return useful graph candidates under explicit traversal budgets
- `Case_Studies/13_Balance_Autocomplete_Freshness_Memory_And_Abuse_Controls.md`  ← decision: Balance autocomplete freshness, memory and abuse controls
- `Case_Studies/14_Ref_Ir018.md`  ← decision: [REF-IR018](../../../../INTERVIEW_CASES/DSA/Algo_Design_Problems/CASE_STUDIES_01.md)
- `Case_Studies/15_Ref_Ir020.md`  ← decision: [REF-IR020](../../../../INTERVIEW_CASES/DSA/Algo_Design_Problems/CASE_STUDIES_03.md)
- `Case_Studies/16_Ref_Ir021.md`  ← decision: [REF-IR021](../../../../INTERVIEW_CASES/DSA/Algo_Design_Problems/CASE_STUDIES_04.md)
- `Case_Studies/17_Ref_Ir024.md`  ← decision: [REF-IR024](../../../../INTERVIEW_CASES/DSA/Algo_Design_Problems/Search/phrase_search_postings.md)
- `Case_Studies/18_Ref_Ir029.md`  ← decision: [REF-IR029](../../../../INTERVIEW_CASES/DSA/Production_DSA_Incidents/Graph_Traversal/bounded_bfs_incident.md)
- `Case_Studies/19_Ref_Ir032.md`  ← decision: [REF-IR032](../../../../INTERVIEW_CASES/DSA/Sys_Des_DSA/Autocomplete/frequency_admission_and_freshness.md)

## Required new CHEAT-SHEET manuscripts
- `Cheat_Sheets/08_Ch_And_A_Assumptions_Rolling_Window_Semantics_Clock_Skew_Positional_Po.md`  ← scope: CH and A* assumptions; rolling-window semantics; clock skew; positional postings; BM25 statistics
- `Cheat_Sheets/09_Routing_Production_Behavior_Coordinated_Rolling_Counters_Distributed_P.md`  ← scope: Routing production behavior; coordinated rolling counters; distributed positional inverted indexes
- `Cheat_Sheets/10_Wal_And_Immutable_Segments_Positional_Phrase_Queries_Field_Weighted_Bm.md`  ← scope: WAL and immutable segments; positional phrase queries; field-weighted BM25; exact top-k; ACLs and tombstones
- `Cheat_Sheets/11_Rarest_Term_Anchor_Lazy_Postings_Skips_Relative_Positions_Repeated_Ter.md`  ← scope: Rarest-term anchor; lazy postings; skips; relative positions; repeated terms; matching analyzers
- `Cheat_Sheets/12_Enqueue_Time_Deduplication_Paged_Adjacency_Independent_Work_Budgets_De.md`  ← scope: Enqueue-time deduplication; paged adjacency; independent work budgets; deterministic traversal; explicit partial results
- `Cheat_Sheets/13_Trie_Admission_Time_Window_Sketches_Bounded_Candidate_Stores_Immutable.md`  ← scope: Trie admission; time-window sketches; bounded candidate stores; immutable snapshots; real-time overlays; CDN versioning

## Mandatory finishing artifacts
- rewrite `Case_Studies/INDEX.md` to 7 sections and `Cheat_Sheets/INDEX.md` to 11 sections, linking only files that exist
- update the unit `INDEX.md` topic-map Status column to `authored` only for topics you actually covered
- append progress rows to `03_machine_learning_optimisation_and_intelligent_retrieval/14_recommendation_search_ads_and_ranking/AUTHORING_PROGRESS.md`