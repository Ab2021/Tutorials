# Graph ML, Knowledge Graphs and Network Analytics — Cheat Sheets

**Status:** The graph-fraud sheet is an unreviewed draft; other titles remain planned. Apply the [course authoring and numerical standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and linked cheat-sheet master. Inventoried original graph-fraud files were unavailable, so their contents were not verified.

## 1. Overview

For analysts and engineers considering graph features for fraud and financial-crime investigation, this collection addresses time-correct data, investigation capacity, evaluation and operational safeguards. Readiness means explaining why a feature was available at decision time, measuring against a non-graph baseline, and distinguishing a connected pattern from evidence of wrongdoing.

## 2. Topic ladder

| Level | Reference | Outcome and prerequisite | Status |
|---|---|---|---|
| Practitioner | [Graph ML for fraud and AML ring detection](01_graph_fraud_detection.md) | Build and evaluate time-correct typed graphs; requires basic graph concepts and fraud evaluation | Draft; unreviewed |
| Planned | Graph data-model patterns | Define typed nodes, edges, time and provenance | Not written |
| Planned | GNN decision guide | Compare neighbor aggregation with non-graph baselines | Not written |
| Planned | Cypher query primer | Query graph data with explicit cost and semantics | Not written |
| Planned | Common Issues | Collection-wide distinct graph diagnostics | Not written |

## 3. Quick topic lookup

| Question | Reference | Condition and warning |
|---|---|---|
| Can graph signals predict fraud? | [Graph fraud sheet](01_graph_fraud_detection.md) | Compare against point-in-time baselines; shared links do not prove fraud. |
| Where can leakage enter? | [Failure modes](01_graph_fraud_detection.md#8-failure-modes--edge-cases) | Restrict nodes, edges, features and labels to information available at the decision cutoff. |
| What should investigators prioritize? | [Worked example](01_graph_fraud_detection.md#5-worked-numerical-example) | Capacity, selected labels and delayed outcomes change what reported precision means. |

## 4. Reading paths

- **Beginner:** Review nodes, typed edges, transaction timestamps, classification and fraud-label maturity, then read the draft's definitions. Foundational data-model and query sheets remain planned.
- **Practitioner:** Read the complete graph-fraud draft, reproduce the worked example and compare graph features with a time-correct non-graph baseline.
- **Advanced:** Test high-degree nodes, graph sampling, temporal leakage, selected investigations, delayed outcomes and candidate utility. Planned GNN and bounded-traversal references will extend this path.

## 5. Prerequisite skills

Understand directed and undirected graphs, entity identifiers, timestamps, classification metrics, investigation queues, transaction histories and data lineage. Consider access controls and privacy for sensitive relationships and outcomes.

## 6. What this collection does not cover

The current draft is not evidence of an actual fraud ring or deployment, a complete anti-money-laundering program, a Cypher tutorial, or a guarantee that a graph model beats a simpler model. The separate inventoried production case remains planned in its case-library home.

## 7. Master symbol glossary

| Symbol or term | Meaning | Scope / warning | Reference |
|---|---|---|---|
| Node | Typed entity in a graph | Entity-resolution errors can merge distinct entities | [Definitions](01_graph_fraud_detection.md#3-foundational-definitions) |
| Edge | Typed relationship, with provenance and time when applicable | A shared relationship alone does not prove fraud | [Definitions](01_graph_fraud_detection.md#3-foundational-definitions) |
| Decision cutoff | Time by which an input must have been available | Later events or labels can leak into historical features | [Graph fraud sheet](01_graph_fraud_detection.md) |
| Precision at capacity | Proportion of verified positives among selected cases at a stated review limit | Selected or immature labels limit inference | [Worked example](01_graph_fraud_detection.md#5-worked-numerical-example) |

## 8. Quick Reference Card directory

| Card | Core rule | Use and critical condition |
|---|---|---|
| [Graph fraud](01_graph_fraud_detection.md#12-quick-reference-card) | Restrict graph inputs to the information available at decision time. | Compare against appropriate baselines, and do not equate graph connectivity with guilt. |

## 9. Common Issues locator

A standalone Common Issues artifact is **planned, not written**; no collection-wide issue count is claimed. Consult the draft's [failure modes](01_graph_fraud_detection.md#8-failure-modes--edge-cases) for leakage, identity and evaluation diagnostics.

## 10. Sources, versions and maintenance

`Graph_ML_2025/04_Fraud_Detection.md` and `DS_Domain/Case_Studies_Production_ML_Systems/Graph_ML/01_GNN_Fraud_AML_Ring_Detection.md` were visible in the source inventory but not available for reading. The draft is not a verified summary of their contents. Review it when entity-resolution policy, graph schema, label process, source availability or decision use changes.

**Planned reference-informed briefs:** [PLAN-2026-IR029 and PLAN-2026-IR036](../../../00_REFERENCE_INFORMED_CASE_PLAN.md) cover bounded graph traversal and conservative claim-actor entity resolution. Assess high-degree and cyclic graph replays, temporal baselines, precision at capacity, label maturity and sampled audits. The [bounded-BFS](../../../../INTERVIEW_CASES/DSA/Production_DSA_Incidents/Graph_Traversal/bounded_bfs_incident.md) and [claim-ring](../../../../INTERVIEW_CASES/Graph_Rings/Fraud_Ring_Detection/Claims_Fraud_Ring_Case.md) documents are reference examples, not verified incidents; limits on output and depth alone do not bound expansion.

## 11. Collection acceptance checklist

- [ ] Planned graph topics and a distinct Common Issues artifact are authored and reviewed.
- [ ] Each sheet retains all 12 sections, checked calculations or traces, failure diagnostics, answered interviews and a final reference card.
- [ ] Entity provenance, point-in-time availability, label selection, privacy and source limitations are explicit.
- [ ] Links, glossary, paths, issue locations and editorial states match the actual files.
