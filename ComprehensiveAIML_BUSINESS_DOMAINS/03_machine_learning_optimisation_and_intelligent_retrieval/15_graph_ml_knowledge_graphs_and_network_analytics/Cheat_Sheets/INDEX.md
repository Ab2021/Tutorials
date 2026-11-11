# Module 15 â€” Graph ML, Knowledge Graphs and Network Analytics — Cheat Sheets

## 1. Overview

This collection carries 8 reference card(s) and 0 troubleshooting artifact(s) for `Module 15 â€” Graph ML, Knowledge Graphs and Network Analytics`. Cards teach a reusable method, calculation, or selection rule; the Common Issues file turns those rules into diagnostics. Status is measured, not asserted: a card is `authored` only when `Tools/manuscript_lint.py` exits 0 for it.

## 2. Topic ladder

| Level | Cards | Purpose |
|---|---|---|
| core | Graph ML for Fraud and AML Ring Detection, Graph data-model patterns, Centrality and communities, GNN decision guide, Knowledge graphs and ontologies | Build from prerequisites to production judgement |
| advanced | Graph evaluation, Bounded traversal and entity resolution, Common issues: graph ML | Build from prerequisites to production judgement |

## 3. Quick lookup

| If you need to… | Read |
|---|---|
| work on graph ml for fraud and aml ring detection | [Graph ML for Fraud and AML Ring Detection](./01_graph_fraud_detection.md) |
| work on graph data-model patterns | [Graph data-model patterns](./02_graph_data_model_patterns.md) |
| work on centrality and communities | [Centrality and communities](./03_centrality_and_communities.md) |
| work on gnn decision guide | [GNN decision guide](./04_gnn_decision_guide.md) |
| work on knowledge graphs and ontologies | [Knowledge graphs and ontologies](./05_knowledge_graphs_and_ontologies.md) |
| work on graph evaluation | [Graph evaluation](./06_graph_evaluation.md) |
| work on bounded traversal and entity resolution | [Bounded traversal and entity resolution](./07_bounded_traversal_and_entity_resolution.md) |
| work on common issues: graph ml | [Common issues: graph ML](./09_25_common_issues.md) |

## 4. Reading paths

| Path | Sequence |
|---|---|
| Beginner | foundation cards first, then the Quick Reference Card (§12) of each core card |
| Practitioner | core cards in ladder order, then one case in `../Case_Studies/` that cites them |
| Advanced | advanced cards, then the Common Issues artifact and its discriminating checks |

## 5. Prerequisite skills

| Skill | Why it is needed | Where it is available |
|---|---|---|
| Unit topic map | Defines what this collection is responsible for and what it excludes | `../INDEX.md` |
| Decision framing | Cards assume a stated decision, population, and cut-off | sibling case collection |
| Arithmetic discipline | Every card carries checked numbers with units | the cards themselves |

## 6. Exclusions and handoffs

| Excluded here | Handed to |
|---|---|
| Decision-context narrative, baselines, and outcomes | `../Case_Studies/` |
| Organisational policy and legal determination | governance modules, not this collection |
| Vendor-specific or unverified current API behaviour | treat as OPEN until sourced at authoring time |

## 7. Master symbol glossary

| Symbol / term | Meaning | Units | Shape / type | Domain | First use |
|---|---|---|---|---|---|
| Block | as defined at first use in the linking card | see card | see card | see card | `01_graph_fraud_detection.md` |
| Blocking completeness | as defined at first use in the linking card | see card | see card | see card | `01_graph_fraud_detection.md` |
| Sentinel block | as defined at first use in the linking card | see card | see card | see card | `01_graph_fraud_detection.md` |
| Sorted-neighbourhood blocking | as defined at first use in the linking card | see card | see card | see card | `01_graph_fraud_detection.md` |
| Canopy / LSH blocking | as defined at first use in the linking card | see card | see card | see card | `01_graph_fraud_detection.md` |
| Match weight | as defined at first use in the linking card | see card | see card | see card | `01_graph_fraud_detection.md` |
| Threshold | as defined at first use in the linking card | see card | see card | see card | `01_graph_fraud_detection.md` |
| Pair-level precision and recall | as defined at first use in the linking card | see card | see card | see card | `01_graph_fraud_detection.md` |
| B-cubed precision and recall | as defined at first use in the linking card | see card | see card | see card | `01_graph_fraud_detection.md` |
| Transitive closure | as defined at first use in the linking card | see card | see card | see card | `01_graph_fraud_detection.md` |
| Correlation clustering | as defined at first use in the linking card | see card | see card | see card | `01_graph_fraud_detection.md` |
| Canonical identifier | as defined at first use in the linking card | see card | see card | see card | `01_graph_fraud_detection.md` |
| Identity version | as defined at first use in the linking card | see card | see card | see card | `01_graph_fraud_detection.md` |
| Frontier | as defined at first use in the linking card | see card | see card | see card | `01_graph_fraud_detection.md` |

## 8. Card directory

| # | Card | Level | Evidence | Status |
|---|---|---|---|---|
| 1 | [Graph ML for Fraud and AML Ring Detection](./01_graph_fraud_detection.md) | core | DERIVED | authored |
| 2 | [Graph data-model patterns](./02_graph_data_model_patterns.md) | core | DERIVED | authored |
| 3 | [Centrality and communities](./03_centrality_and_communities.md) | core | DERIVED | authored |
| 4 | [GNN decision guide](./04_gnn_decision_guide.md) | core | DERIVED | draft (lint open) |
| 5 | [Knowledge graphs and ontologies](./05_knowledge_graphs_and_ontologies.md) | core | DERIVED | draft (lint open) |
| 6 | [Graph evaluation](./06_graph_evaluation.md) | advanced | DERIVED | draft (lint open) |
| 7 | [Bounded traversal and entity resolution](./07_bounded_traversal_and_entity_resolution.md) | advanced | DERIVED | authored |
| 8 | [Common issues: graph ML](./09_25_common_issues.md) | advanced | DERIVED | draft (lint open) |

## 9. Common Issues locator

- Not yet authored. The collection is incomplete until this artifact exists with at least 25 distinct issues.

## 10. Sources, versions, maintenance

| Field | Value |
|---|---|
| Evidence vocabulary | DERIVED / VERIFIED / SOURCED / ILLUSTRATIVE / HEURISTIC / OPEN |
| Version policy | Content is dated by its own frontmatter; unstable API, price, and regulatory facts are rechecked at edit time |
| Maintenance trigger | A card's validity condition, a source body, or a tool's semantics changes |
| Review cadence | Quarterly, plus on any material change to a cited source |

## 11. Acceptance checklist

- [x] Every linked card exists on disk.
- [x] Status column is measured by the lint, not asserted.
- [x] Planned commitments remain listed without links.
- [ ] Exactly one Common Issues artifact passes lint, including the distinct issue-ID check.
- [ ] Independent technical review of each card.
