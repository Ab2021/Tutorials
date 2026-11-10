# Part 03 Coverage Ledger

**Scope date:** 2026-11-09  
**Meaning of state:** `draft` means an artifact exists but has no independent review; `registered` means the commitment has a stable destination but no authored artifact. Source inventory names establish provenance only; they do not establish access to an original source body. No deployment, benchmark, company outcome, or regulatory claim is asserted by this ledger.

## Evidence and acceptance boundary

The governing requirements are the [course authoring and numerical standards](../00_AUTHORING_AND_NUMERICAL_STANDARDS.md), the [cheat-sheet template](../../Templates/CHEAT_SHEET_CONTENT_MASTER_TEMPLATE.md), and the [case template](../../Templates/CASE_STUDY_CONTENT_MASTER_TEMPLATE.md). A completed individual sheet needs 12 ordered sections, a final Section 12 card, substantive line-depth, numerical or state-oracle checks, failures, and answered interview questions. A completed case needs 15 ordered sections including 5a–5e and a final Section 15 card. These are authoring requirements, not satisfied by this ledger.

| Evidence label | Meaning here | Limitation |
|---|---|---|
| `VERIFIED` | Local arithmetic, enumeration, or trace was checked | Not a production measurement |
| `DERIVED` | A conclusion follows from shown assumptions | Assumptions may not match a live workflow |
| `ILLUSTRATIVE` | Constructed teaching input or scenario | Not evidence about a named organization |
| `OPEN` | Source access, review, or outcome evidence remains unavailable | No stronger claim may be made |

## Original module commitments

| Module | Commitment | Teaching home | State | Evidence / remaining gate |
|---|---|---|---|---|
| 11 | Problem formulation | `11.../Cheat_Sheets/02_Problem_Formulation_and_Losses.md` | draft | Local illustrative arithmetic checked; independent review open |
| 11 | Linear trees and boosting | `11.../Cheat_Sheets/03_Linear_Trees_and_Boosting.md` | registered | Artifact not authored |
| 11 | Feature engineering | `11.../Cheat_Sheets/04_Feature_Engineering_Patterns.md` | registered | Artifact not authored |
| 11 | Imbalance and calibration | `11.../Cheat_Sheets/05_Imbalance_and_Calibration.md` | registered | Artifact not authored |
| 11 | Validation and error analysis | `11.../Cheat_Sheets/01_Model_Validation_and_Error_Analysis.md` | draft | Existing draft requires independent review |
| 12 | Clustering and embeddings | `12.../Cheat_Sheets/02_Clustering_and_Embeddings.md` | registered | Artifact not authored |
| 12 | Density and isolation methods | `12.../Cheat_Sheets/01_Anomaly_Detection_Triage.md` | draft | Existing draft requires independent review |
| 12 | Change-point detection | `12.../Cheat_Sheets/03_Change_Point_Detection.md` | registered | Artifact not authored |
| 12 | Segmentation evaluation | `12.../Cheat_Sheets/04_Segmentation_Evaluation.md` | registered | Artifact not authored |
| 12 | Human review loops | `12.../Cheat_Sheets/05_Human_Review_Loops.md` | registered | Artifact not authored |
| 13 | Formulation and duality | `13.../Cheat_Sheets/01_optimisation_formulation.md` | registered | Artifact not authored |
| 13 | Network and scheduling | `13.../Cheat_Sheets/04_network_and_scheduling_models.md` | registered | Artifact not authored |
| 13 | Integer programming | `13.../Cheat_Sheets/05_integer_programming.md` | registered | Artifact not authored |
| 13 | Metaheuristics | `13.../Cheat_Sheets/06_metaheuristics.md` | registered | Artifact not authored |
| 13 | Discrete-event simulation | `13.../Cheat_Sheets/03_simulation_experiment_plan.md` | registered | Artifact not authored |
| 14 | Candidate generation | `14.../Cheat_Sheets/01_ranking_metrics.md` | registered | Artifact not authored |
| 14 | Learning to rank | `14.../Cheat_Sheets/01_ranking_metrics.md` | registered | Artifact not authored |
| 14 | Collaborative and content models | `14.../Cheat_Sheets/02_recsys_architectures.md` | registered | Artifact not authored |
| 14 | Bandits and exploration | `14.../Cheat_Sheets/03_online_experimentation_checklist.md` | registered | Artifact not authored |
| 14 | Offline-online evaluation | `14.../Cheat_Sheets/03_online_experimentation_checklist.md` | registered | Artifact not authored |
| 15 | Graph modelling | `15.../Cheat_Sheets/02_graph_data_model_patterns.md` | registered | Artifact not authored |
| 15 | Centrality and communities | `15.../Cheat_Sheets/03_centrality_and_communities.md` | registered | Artifact not authored |
| 15 | Graph embeddings and GNNs | `15.../Cheat_Sheets/04_gnn_decision_guide.md` | registered | Artifact not authored |
| 15 | Knowledge graph construction | `15.../Cheat_Sheets/05_knowledge_graphs_and_ontologies.md` | registered | Artifact not authored |
| 15 | Graph evaluation | `15.../Cheat_Sheets/06_graph_evaluation.md` | registered | Artifact not authored |
| 16 | Privacy threat modelling | `16.../Cheat_Sheets/01_privacy_method_selection.md` | registered | Artifact not authored |
| 16 | Differential privacy | `16.../Cheat_Sheets/01_privacy_method_selection.md` | registered | Artifact not authored |
| 16 | Federated learning | `16.../Cheat_Sheets/02_federated_learning_architecture.md` | registered | Artifact not authored |
| 16 | Secure aggregation | `16.../Cheat_Sheets/02_federated_learning_architecture.md` | registered | Artifact not authored |
| 16 | Synthetic-data limits | `16.../Cheat_Sheets/03_data_minimisation_list.md` | registered | Artifact not authored |

## Source-assigned commitments

| Planning/source ID | Required mechanism | Local home | State / boundary |
|---|---|---|---|
| IR025 | Half-open intervals, deterministic ties, exchange argument, weighted-DP boundary | `13.../Cheat_Sheets/07_interval_greedy_and_weighted_scheduling.md` | draft; locally hand-checked, source body OPEN |
| IR018 | A*/CH assumptions, rolling windows, skew, positional postings, BM25 | Module 14 collection | registered; not authored |
| IR020 | Routing/admission/index correctness under partial failure | Module 14 collection | registered; not authored |
| IR021 | WAL, immutable segments, phrases, field BM25, top-k, ACLs, tombstones | Module 14 collection | registered; not authored |
| IR024 | Rarest anchors, lazy/skipped postings, positions, repeated terms | Module 14 collection | registered; not authored |
| IR029 | Enqueue deduplication, paged adjacency, independent budgets, deterministic partial results | Modules 14 and 15 | registered; not authored |
| IR032 | Trie admission, sketches, bounded stores, snapshots/overlays/versioning | Module 14 collection | registered; not authored |
| IR036 | Conservative resolution, typed graphs, motifs, time-safe features, reversible merges | Module 15 collection | registered; not authored |
| Graph_ML_2025/01 | Graph theory foundations | Module 15 | registered; source body OPEN |
| Graph_ML_2025/02 | Graph neural networks | Module 15 | registered; source body OPEN |
| Graph_ML_2025/03 | Knowledge graphs and ontologies | Module 15 | registered; source body OPEN |
| Graph_ML_2025/04 | Fraud detection | `15.../Cheat_Sheets/01_graph_fraud_detection.md` | draft; source body OPEN |

## Case and troubleshooting requirements

| Collection | Original cases | Current state | Final issue artifact |
|---|---:|---|---|
| Module 11 | 2 | registered only | `06_25_Common_Issues.md` registered |
| Module 12 | 2 | registered only | final artifact registered |
| Module 13 | 2 plus IR025 scheduling case | registered only | `08_25_Common_Issues.md` registered |
| Module 14 | 2 | registered only | final artifact registered |
| Module 15 | 2 plus IR029/IR036 mapping | registered only | `09_25_common_issues.md` registered |
| Module 16 | 2 | registered only | final artifact registered |
| Part 03 | cross-module navigation only | no individual part case authored | part final artifact registered |

No collection currently has a completed final Common Issues artifact with 25 complete distinct entries. No case is represented as authored merely because its planning row exists.

## Local authoring checkpoint

- Module 11 formulation sheet: draft, 12 ordered sections and final card present; local illustrative arithmetic reported by its author; independent review OPEN.
- Module 13 interval scheduling sheet: draft, 12 ordered sections and final card present; recurrence trace is locally documented; it remains below the course’s substantive-depth gate and independently unreviewed.
- Module 15 graph-fraud sheet: draft, existing substantive artifact; independent source and editorial review OPEN.
- All remaining registered paths require full authorship before their module or this part can be called complete.

**Recheck triggers:** any source-body access, changed template/standards, changed prerequisites, changed interval/graph time semantics, or an independently reviewed teaching artifact.
