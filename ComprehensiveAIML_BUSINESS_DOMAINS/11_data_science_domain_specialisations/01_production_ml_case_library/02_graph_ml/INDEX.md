# Graph ML

> **Family:** [Production ML Case Library](../INDEX.md)  
> **Purpose:** Relationship-aware modelling for risk, recommendation, search, entity resolution, and network intelligence.  
> **Company / context examples:** Financial crime, retail, enterprise search, social products

## Detailed topic map

| # | Topic | Status |
|---:|---|---|
| 1 | graph data modelling | planned |
| 2 | entity resolution | planned |
| 3 | centrality and communities | planned |
| 4 | graph embeddings | planned |
| 5 | GNNs | planned |
| 6 | knowledge graphs | planned |
| 7 | temporal graphs | planned |
| 8 | sampling and scaling | planned |
| 9 | evaluation | planned |
| 10 | governance | planned |

## Applied collections

- [Case Studies](./Case_Studies/INDEX.md): decision-centred cases using the local v2 catalog and integrated outline standard.
- [Cheat Sheets](./Cheat_Sheets/INDEX.md): concise references for recurring design, analysis, and operating decisions.

## Scope boundary

Use public company material only where cited. Treat company-specific systems as unknown unless a primary or clearly labelled published source supports the claim. Learning cases may be simulated or illustrative, but cannot claim measured production impact.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Planning ID | Detailed topics to develop | Decision to address | Validation to design |
|---|---|---|---|
| PLAN-2026-IR036 | Conservative entity resolution; typed claim-actor graphs; motifs and supernodes; point-in-time features; reversible merges | Which graph-enriched alerts add confirmed value at a fixed investigation capacity beyond claim-level scoring? | Temporal baseline comparison, precision at capacity, mature outcomes, sampled low-risk audits and appeal/false-positive slices |

See the local Case_Studies and Cheat_Sheets indexes for proposed outputs and source-reference links. These scope additions do not replace the original planned topics.

<!-- END INTERVIEW REFERENCE PLAN -->

## Source-tree coverage additions

The [exact source crosswalk](../../../00_SOURCE_TOPIC_CROSSWALK.md) extends the original topic list with the focused briefs below. Each technical title in a linked register is a planned teaching commitment, not completed content. Follow the [master-template and numerical standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) when authoring cases and sheets; hidden/clipped sources remain unresolved.

| Focused source brief | Required outputs / state |
|---|---|
| [Graph ML production decisions](../../../00_SOURCE_COVERAGE_DATA_SCIENCE.md#ds-cases-02) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |

Source module numbers and course module numbers are different namespaces. Reuse matching existing lessons when authored, but do not drop distinct variants, cases, failure themes or source limitations.
