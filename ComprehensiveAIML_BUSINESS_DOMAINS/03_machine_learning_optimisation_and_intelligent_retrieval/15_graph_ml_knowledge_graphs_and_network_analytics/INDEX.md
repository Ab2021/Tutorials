# Module 15 â€” Graph ML, Knowledge Graphs and Network Analytics

> **Part:** [Part 03 â€” Machine Learning, Optimisation and Intelligent Retrieval](../INDEX.md)  
> **Prerequisites:** 06,11,12  
> **Purpose:** Use relationships, topology, and semantics for fraud, discovery, reasoning, and network decisions.

## Topics to develop

| # | Topic | Status |
|---:|---|---|
| 1 | graph modelling | [authored](Cheat_Sheets/01_graph_fraud_detection.md) |
| 2 | centrality and communities | [authored](Cheat_Sheets/03_centrality_and_communities.md) |
| 3 | graph embeddings and GNNs | planned |
| 4 | knowledge graph construction | planned |
| 5 | graph evaluation | [authored](Cheat_Sheets/06_graph_evaluation.md) |
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
| PLAN-2026-IR029 | Enqueue-time deduplication; paged adjacency; independent work budgets; deterministic traversal; explicit partial results | How should a recommendation API respond when graph expansion reaches a node, edge, memory or time limit? | High-degree and cyclic graph replays, peak memory, examined edges, output completeness and candidate utility |
| PLAN-2026-IR036 | Conservative entity resolution; typed claim-actor graphs; motifs and supernodes; point-in-time features; reversible merges | Which graph-enriched alerts add confirmed value at a fixed investigation capacity beyond claim-level scoring? | Temporal baseline comparison, precision at capacity, mature outcomes, sampled low-risk audits and appeal/false-positive slices |

See the local Case_Studies and Cheat_Sheets indexes for proposed outputs and source-reference links. These scope additions do not replace the original planned topics.

<!-- END INTERVIEW REFERENCE PLAN -->

## Source-tree coverage additions

The [exact source crosswalk](../../00_SOURCE_TOPIC_CROSSWALK.md) extends the original topic list with the focused briefs below. Each technical title in a linked register is a planned teaching commitment, not completed content. Follow the [master-template and numerical standards](../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) when authoring cases and sheets; hidden/clipped sources remain unresolved.

| Focused source brief | Required outputs / state |
|---|---|
| [Graph theory, GNNs, knowledge graphs/ontologies and fraud detection](../../00_SOURCE_COVERAGE_DATA_SCIENCE.md#graph-source) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |

Source module numbers and course module numbers are different namespaces. Reuse matching existing lessons when authored, but do not drop distinct variants, cases, failure themes or source limitations.

## Exact local coverage ledger

Registered before authoring on 2026-11-08. Editorial state is draft until independent review; source inventory access is not original-source access. Original rows above remain commitments.

| Commitment | Stable local artifact | Outcome / quantitative oracle | Evidence boundary |
|---|---|---|---|
| Original 1 graph modelling; original sheet Graph data-model patterns; Graph_ML_2025/01_Graph_Theory_Foundations.md | SHEET-G15-02 `Cheat_Sheets/02_graph_data_model_patterns.md` | Typed multigraph, degree identities, connectivity, A:[B,C], B:[D], C:[D], D:[] traversal | Original file unavailable; independently authored foundations |
| Original 2 centrality and communities | SHEET-G15-03 `Cheat_Sheets/03_centrality_and_communities.md` | Degree, PageRank, betweenness, modularity and stability | No measured network benchmark |
| Original 3 graph embeddings and GNNs; original sheet GNN decision guide; Graph_ML_2025/02_Graph_Neural_Networks.md | SHEET-G15-04 `Cheat_Sheets/04_gnn_decision_guide.md` | Neighborhood mean, GCN, GraphSAGE, embeddings, fanout budget | Original file unavailable; primary method references required |
| Original 4 knowledge graph construction; Graph_ML_2025/03_Knowledge_Graphs_Ontologies.md | SHEET-G15-05 `Cheat_Sheets/05_knowledge_graphs_and_ontologies.md` | RDF statements, identity, OWL versus SHACL, provenance and entailment | Open-world reasoning is not closed-world validation |
| Original 5 graph evaluation | SHEET-G15-06 `Cheat_Sheets/06_graph_evaluation.md` | Temporal/inductive splits, precision at capacity, negatives and uncertainty | Outcome validity and label maturity remain workload-specific |
| Original sheet Cypher query primer | SHEET-G15-07 `Cheat_Sheets/07_cypher_query_primer.md` | Explicit path semantics, query cardinality and bounded expansion | Syntax examples are not executed benchmarks |
| PLAN-2026-IR029 | SHEET-G15-08 `Cheat_Sheets/08_bounded_graph_traversal.md`; CASE-G15-03 `Case_Studies/03_bounded_graph_candidates.md` | Enqueue deduplication, paged adjacency, independent work budgets, deterministic partial outputs | Illustrative incident exercise, not a verified outage |
| Existing graph fraud sheet; Graph_ML_2025/04_Fraud_Detection.md | SHEET-G15-01 `Cheat_Sheets/01_graph_fraud_detection.md` | Expand existing file in place; time-safe aggregation and capacity evaluation | Original file unavailable; no fraud or AML deployment result claimed |
| Original case Fraud-ring investigation; PLAN-2026-IR036 | CASE-G15-01 `Case_Studies/01_fraud_ring_investigation.md` | Conservative resolution, typed claim-actor graph, motifs, supernodes, mature fixed-capacity comparison, audit and appeal | One local case fulfills both commitments without duplicating case count |
| Original case enterprise entity resolution graph | CASE-G15-02 `Case_Studies/02_enterprise_entity_resolution_graph.md` | Reversible merges, blocked pairs, match decisions and downstream graph sensitivity | Illustrative enterprise workflow; no real customer identity evidence |
| Final Common Issues | SHEET-G15-09 `Cheat_Sheets/09_25_common_issues.md` | 25 complete distinct diagnostics across four required categories | Hypothetical provenance; not 25 observed incidents |
| Graph_ML_2025 directory | Source container only | Four visible technical children above | Not a fifth lesson |
| Five DS_Domain/Case_Studies_Production_ML_Systems/Graph_ML production titles | Part 11 teaching-home handoff; local method coverage above preserved | GNN fraud/AML; GraphSAGE/PinSage recommendation; KG search/QA; social link prediction; marketing attribution | Owned elsewhere; not counted as local authored production cases |

Every individual artifact links the [exact inventory](../../../Templates/directory.txt) and the [focused source brief](../../00_SOURCE_COVERAGE_DATA_SCIENCE.md#graph-source). Source-specific 50/100-issue targets do not occur in the visible graph-source entries; the local final file targets 25 distinct issues. Publication gates, source access and independent numerical/navigation review remain open and must not be inferred from file length.
