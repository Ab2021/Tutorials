# Module 15 â€” Graph ML, Knowledge Graphs and Network Analytics

> **Part:** [Part 03 â€” Machine Learning, Optimisation and Intelligent Retrieval](../INDEX.md)  
> **Prerequisites:** 06,11,12  
> **Purpose:** Use relationships, topology, and semantics for fraud, discovery, reasoning, and network decisions.

## Topics to develop

| # | Topic | Status |
|---:|---|---|
| 1 | graph modelling | planned |
| 2 | centrality and communities | planned |
| 3 | graph embeddings and GNNs | planned |
| 4 | knowledge graph construction | planned |
| 5 | graph evaluation | planned |

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
