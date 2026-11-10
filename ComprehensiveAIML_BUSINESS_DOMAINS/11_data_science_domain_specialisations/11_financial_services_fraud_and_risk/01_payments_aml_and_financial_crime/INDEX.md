# Payments, AML and Financial Crime

> **Family:** [Financial Services, Fraud and Risk](../INDEX.md)  
> **Purpose:** Event-driven and graph-aware systems for payments fraud, AML, sanctions, investigations, and controls.  
> **Company / context examples:** Banks, payment networks, fintech

## Detailed topic map

| # | Topic | Status |
|---:|---|---|
| 1 | payment flows | [authored](Case_Studies/05_Payment_Outage_Fraud_Controls.md) |
| 2 | fraud | [authored](Case_Studies/01_Card_Fraud_Streaming_Decision.md) |
| 3 | AML | [authored](Case_Studies/02_AML_Ring_Detection.md) |
| 4 | sanctions | [authored](Case_Studies/03_Sanctions_False_Positive_Reduction.md) |
| 5 | entity resolution | [authored](Cheat_Sheets/05_Entity_Resolution.md) |
| 6 | graphs | [authored](Cheat_Sheets/06_Typed_Fraud_Graphs.md) |
| 7 | streaming | [authored](Case_Studies/01_Card_Fraud_Streaming_Decision.md) |
| 8 | case management | [authored](Cheat_Sheets/08_Case_Management.md) |
| 9 | investigator UX | [authored](Cheat_Sheets/09_Investigator_UX.md) |
| 10 | model risk | [authored](Cheat_Sheets/10_Model_Risk_Controls.md) |
| 11 | privacy | [authored](Cheat_Sheets/11_Privacy_and_Access.md) |
| 12 | audits | planned |
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
