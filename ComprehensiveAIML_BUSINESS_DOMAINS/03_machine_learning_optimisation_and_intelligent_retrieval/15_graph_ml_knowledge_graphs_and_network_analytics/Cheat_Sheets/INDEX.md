# Module 15 Cheat Sheets â€” Graph ML, Knowledge Graphs and Network Analytics

## Authoring standard

Every authored sheet must follow the [course authoring and numerical standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked cheat-sheet master: all 12 sections, derivations/invariants, checked numerical examples, at least four diagnostic failures, 10–15 answered interview questions, and the final Quick Reference Card. This is a planned catalog, not completed sheets; expand to all 11 collection-index sections and a final Common Issues artifact (at least 25 complete issues) before publication. A card alone is not a completed sheet.

## Planned cheat sheets

| # | Reference | Coverage | Status |
|---:|---|---|---|
| 1 | graph data-model patterns | Concise patterns, decision rules, assumptions, and common traps. | planned |
| 2 | GNN decision guide | Concise patterns, decision rules, assumptions, and common traps. | planned |
| 3 | Cypher query primer | Concise patterns, decision rules, assumptions, and common traps. | planned |
| 4 | [Graph ML for fraud and AML ring detection](01_graph_fraud_detection.md) | Time-correct typed graphs, leakage, neighbor aggregation, capacity-limited evaluation, diagnostics, and interview practice. | draft; unreviewed |

The graph-fraud sheet is a draft, not a reviewed publication; the other entries remain planned. The directory inventories `Graph_ML_2025/04_Fraud_Detection.md` and `DS_Domain/Case_Studies_Production_ML_Systems/Graph_ML/01_GNN_Fraud_AML_Ring_Detection.md`, but their contents were not available for source-specific verification. The production case is mapped to a separate case-library home and is not claimed as authored here.

## Suggested file naming

Use **01_reference-slug.md**, then **02_reference-slug.md**. Keep one concern per file so references remain scannable.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Related brief | Proposed cheat-sheet scope | Checks / counterexamples to include | Reference only | Status |
|---|---|---|---|---|
| PLAN-2026-IR029 | Enqueue-time deduplication; paged adjacency; independent work budgets; deterministic traversal; explicit partial results | High-degree and cyclic graph replays, peak memory, examined edges, output completeness and candidate utility. Boundary: Depth and output-size limits alone do not bound expansion; source product degrees and outage figures are illustrative claims. | [REF-IR029](../../../../INTERVIEW_CASES/DSA/Production_DSA_Incidents/Graph_Traversal/bounded_bfs_incident.md) | planned; not written |
| PLAN-2026-IR036 | Conservative entity resolution; typed claim-actor graphs; motifs and supernodes; point-in-time features; reversible merges | Temporal baseline comparison, precision at capacity, mature outcomes, sampled low-risk audits and appeal/false-positive slices. Boundary: Connectivity and shared providers are not proof of fraud; source reference notes are unavailable and labels are delayed/selected. | [REF-IR036](../../../../INTERVIEW_CASES/Graph_Rings/Fraud_Ring_Detection/Claims_Fraud_Ring_Case.md) | planned; not written |

Each future cheat sheet needs its own explanation, preconditions, formulas or algorithms where relevant, failure signals and checked examples. Do not copy the interview narrative or its scenario figures into a reference sheet.

<!-- END INTERVIEW REFERENCE PLAN -->
