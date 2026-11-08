# Part 03 — Machine Learning, Optimisation and Intelligent Retrieval

> **Folder:** 03_machine_learning_optimisation_and_intelligent_retrieval  
> **Purpose:** Core predictive, prescriptive, graph, search, and recommendation methods for production decisions.

## Position in the programme

Parts 01-02; modules build from descriptive to predictive and prescriptive methods.

## Modules

| # | Module | Focus | Indexed topics |
|---:|---|---|---:|
| 11 | [Supervised Machine Learning and Feature Engineering](./11_supervised_ml_and_feature_engineering/INDEX.md) | Build robust tabular prediction systems with appropriate validation, features, objectives, and calibration. | 5 |
| 12 | [Unsupervised ML, Anomaly Detection and Segmentation](./12_unsupervised_ml_anomaly_detection_and_segmentation/INDEX.md) | Discover structure, rare behaviour, and groups without assuming labels are complete or reliable. | 5 |
| 13 | [Optimisation, Operations Research and Simulation](./13_optimization_operations_research_and_simulation/INDEX.md) | Model constrained decisions with linear programming, integer optimisation, heuristics, and simulation. | 5 |
| 14 | [Recommendation, Search, Ads and Ranking](./14_recommendation_search_ads_and_ranking/INDEX.md) | Design retrieval-and-ranking systems that balance relevance, business objectives, exploration, and fairness. | 5 |
| 15 | [Graph ML, Knowledge Graphs and Network Analytics](./15_graph_ml_knowledge_graphs_and_network_analytics/INDEX.md) | Use relationships, topology, and semantics for fraud, discovery, reasoning, and network decisions. | 5 |
| 16 | [Privacy-Preserving ML and Federated Learning](./16_privacy_preserving_ml_and_federated_learning/INDEX.md) | Apply privacy engineering, federated learning, secure computation, and data minimisation to sensitive AI workloads. | 5 |

## Part-level collections

- [Case studies](./Case_Studies/INDEX.md) connect multiple modules into cross-cutting system and business narratives.
- [Cheat sheets](./Cheat_Sheets/INDEX.md) consolidate the part's reusable patterns, decision frameworks, and review checklists.

## Coverage ledger and counting contract

**Ledger established:** 2026-11-08, before this authoring pass. **Editorial state:** authoring in progress; no independent review or production execution is asserted. The [course standards](../00_AUTHORING_AND_NUMERICAL_STANDARDS.md), [visible inventory](../../Templates/directory.txt), [exact source crosswalk](../00_SOURCE_TOPIC_CROSSWALK.md), and [reference-informed plan](../00_REFERENCE_INFORMED_CASE_PLAN.md) jointly define scope. Original source bodies are not established as accessible merely because their filenames appear in the inventory.

Counts below describe distinct *commitments*, not independent published artifacts. A broad module outcome, an original reference family, a source title, and a reference case can share an authored file only if every distinct mechanism and decision remains explicit. Backlinks do not create additional completions. The six modules retain their original five-row topic tables in the displayed learning order.

| Ledger class | Exact initial accounting | Completion evidence required |
|---|---|---|
| Original module topic outcomes | 6 modules × 5 rows = **30** | All named outcomes taught with derivation/mechanism, examples, selection, failures and answers; local row-to-file mapping |
| Original module cheat-sheet families | 6 modules × 3 named families = **18** | Original families preserved alongside later validation/anomaly/fraud drafts |
| Original module case decisions | 6 modules × 2 named cases = **12** | Distinct consequential decisions, not replacement by one generic case |
| Reference-informed additions | **8 unique PLAN IDs; 9 module incidences** | Exact mapping in the table below; IR029 occurs in both modules 14 and 15 without double-counting its source |
| Source crosswalk assigned directly to this part | **5 nodes: 1 container and 4 technical titles** | Four graph-source technical commitments; container is not a lesson |
| Existing individual teaching drafts at entry | **3**: validation, anomaly triage and graph fraud | Expand in place; existence of an earlier short draft is not template/depth acceptance |
| Part-level original reference-family handoffs | **18**, grouped in six families | Preserve in the part cheat-sheet catalog; these are cross-module navigation, not 18 extra completions |
| Part-level original candidate-thread handoffs | **12**, grouped in six threads | Preserve in the part case catalog; do not double-count module cases as new part cases |
| Final troubleshooting collections | **7**: six module collections and the part collection | Each distinct final artifact requires at least 25 complete diagnostic issues and the full sheet structure |

### Exact original outcome inventory

| Module | Outcomes in original INDEX order | Original reference families | Original case decisions |
|---|---|---|---|
| 11 | problem formulation; linear trees and boosting; feature engineering; imbalance and calibration; validation and error analysis | model selection; feature-engineering patterns; classification metrics | Credit-risk model design; churn model with leakage remediation |
| 12 | clustering and embeddings; density and isolation methods; change-point detection; segmentation evaluation; human review loops | anomaly method matrix; clustering diagnostics; triage workflow | Payment anomaly detection; customer segmentation for campaigns |
| 13 | formulation and duality; network and scheduling models; integer programming; metaheuristics; discrete-event simulation | optimisation formulation; solver selection; simulation experiment plan | Last-mile routing optimisation; workforce scheduling under uncertainty |
| 14 | candidate generation; learning to rank; collaborative and content models; bandits and exploration; offline-online evaluation | ranking metrics; recsys architectures; online experimentation checklist | E-commerce recommender; search ranking quality regression |
| 15 | graph modelling; centrality and communities; graph embeddings and GNNs; knowledge graph construction; graph evaluation | graph data-model patterns; GNN decision guide; Cypher query primer | Fraud-ring investigation; enterprise entity resolution graph |
| 16 | privacy threat modelling; differential privacy; federated learning; secure aggregation; synthetic data limits | privacy method selection; federated-learning architecture; data minimisation list | Cross-hospital risk model; private analytics release review |

### Exact reference additions

| Reference commitment | Local home(s) | Distinct mechanisms that must not disappear |
|---|---|---|
| PLAN-2026-IR018 | Module 14 | Contraction hierarchies and A* assumptions; rolling windows and clock skew; positional postings and BM25 statistics |
| PLAN-2026-IR020 | Module 14 | Production routing correctness; coordinated rolling counters; distributed positional indexes; coherent snapshots and partial failures |
| PLAN-2026-IR021 | Module 14 | Write-ahead log; immutable segments; positional phrase queries; field-weighted BM25; exact global top-k; access controls; tombstones |
| PLAN-2026-IR024 | Module 14 | Rarest-term anchors; lazy postings; skips; relative positions; repeated terms; analyzer agreement |
| PLAN-2026-IR025 | Module 13 | Half-open intervals; deterministic ties; earliest-finish exchange argument; weighted interval dynamic programming boundary |
| PLAN-2026-IR029 | Modules 14 and 15 | Enqueue-time deduplication; paged adjacency; independent node/edge/memory/time budgets; deterministic traversal; explicit partial results |
| PLAN-2026-IR032 | Module 14 | Trie admission; time-window sketches; bounded candidate stores; immutable snapshots; live overlays; content-delivery versioning |
| PLAN-2026-IR036 | Module 15 | Conservative entity resolution; typed claim-actor graphs; motifs and supernodes; point-in-time features; reversible merges; investigator value at capacity |

### Exact directly assigned source titles

The [graph-source brief](../00_SOURCE_COVERAGE_DATA_SCIENCE.md#graph-source) assigns `Graph_ML_2025` as a container and the following four technical titles to module 15: `01_Graph_Theory_Foundations.md`, `02_Graph_Neural_Networks.md`, `03_Knowledge_Graphs_Ontologies.md`, and `04_Fraud_Detection.md`. Original source names identify provenance, not evidence of source-body access or a requirement to duplicate synonymous course filenames.

Visible production recommendation/search/graph case names elsewhere in the inventory have their separate Part 11 teaching homes in the crosswalk; linking or teaching a matching method here does not change those source assignments. Source namespaces and course module numbers are not interchangeable.

### Navigation and acceptance record

Each module INDEX is the authoritative local commitment-to-artifact ledger. Its Case_Studies INDEX distinguishes evidence, intervention stage and editorial state; its Cheat_Sheets INDEX provides reading paths, symbols, cards and the final issue locator. Part-level collections connect these methods without replacing their local decisions. Final root/source-register integration belongs to the shared course maintainer.

Authoring is not independent review. Counts of lines, files and sections establish neither mathematical correctness nor inaccessible-source completeness. Required arithmetic/oracle inputs accompany the authored examples; execution, link checks and independent publication review must be recorded separately.


## Completion rule

A learner should reconstruct the assumptions, derive or trace the method, recompute the numerical example, reject an invalid use, diagnose competing failure causes, and defend the bounded decision under changed constraints. A case-study *brief* alone does not satisfy this part. Every indexed commitment above requires substantive authored coverage under the shared standards; local catalogs must state evidence gaps and editorial state honestly.