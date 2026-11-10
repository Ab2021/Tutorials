# Fraud Modelling

> **Family:** [Production ML Case Library](../INDEX.md)  
> **Purpose:** Payment, claims, account, application, AML, and marketplace fraud under delayed labels and adversarial adaptation.  
> **Company / context examples:** Banks, insurers, marketplaces, payments

## Detailed topic map

| # | Topic | Status |
|---:|---|---|
| 1 | fraud taxonomy | [authored](Cheat_Sheets/01_Fraud_Taxonomy_and_Decision_Grain.md) |
| 2 | decision cut-offs and losses | [authored](Cheat_Sheets/02_Decision_Cutoffs_and_Losses.md) |
| 3 | labels and investigation feedback | planned |
| 4 | real-time features | [authored](Case_Studies/01_real_time_payment_card_fraud_detection.md) |
| 5 | imbalance and calibration | planned |
| 6 | rules plus ML | planned |
| 7 | graph fraud | [authored](Case_Studies/01_real_time_payment_card_fraud_detection.md) |
| 8 | alert operations | [authored](Case_Studies/04_aml_transaction_monitoring_system.md) |
| 9 | monitoring and concept drift | planned |
| 10 | fairness and explainability | planned |
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
| [Fraud detection](../../../00_SOURCE_COVERAGE_DATA_SCIENCE.md#ds-cases-01) | Focused sheets with derivations/invariants and checked numerical examples; bounded cases with validation and failure analysis; planned |

Source module numbers and course module numbers are different namespaces. Reuse matching existing lessons when authored, but do not drop distinct variants, cases, failure themes or source limitations.
