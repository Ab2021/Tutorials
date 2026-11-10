# Fraud Modelling - Cheat Sheets

> **Mandatory authoring contract:** Every authored sheet must follow the [course authoring and numerical standards](../../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and the linked cheat-sheet master: all 12 sections, derivations/invariants, checked numerical examples, at least four diagnostic failures, 10–15 answered interview questions, and the final Quick Reference Card. This is a planned catalog, not completed sheets; expand to all 11 collection-index sections and a final Common Issues artifact (at least 25 complete issues) before publication. A card alone is not a completed sheet.

## Purpose

These planned reference cards support repeatable decisions in Fraud Modelling. They complement case studies; they do not replace the assumptions, evidence, and evaluation required in a full case.

## Planned references

| # | Reference | Coverage | Status |
|---:|---|---|---|
| 1 | [Fraud taxonomy and decision grain](01_Fraud_Taxonomy_and_Decision_Grain.md) | Payment, claims, account/application, AML, and marketplace decision grains; labels, losses, timing, and graph boundaries. | draft; ILLUSTRATIVE/DERIVED |
| 2 | [Decision cut-offs and loss matrices](02_Decision_Cutoffs_and_Losses.md) | Action-conditioned losses, capacity-constrained thresholds, calibration boundaries, policy guardrails, 25 complete diagnostic entries, sensitivity drills, and release-state contracts. | draft; authored 2026-11-09; DERIVED/ILLUSTRATIVE; source evidence OPEN |
| 3 | Label-delay and feedback-loop checklist | Decision rules, inputs, metrics, trade-offs, and failure signals. | planned |
| 4 | Real-time feature and alerting patterns | Decision rules, inputs, metrics, trade-offs, and failure signals. | planned |

## Required card fields

Every card must state: intended decision and reader; preconditions and inputs; default recommendation; metrics/formulas or architecture pattern; trade-offs; failure signals; security/privacy/governance constraints; and links to deeper material. Review tool-, market-, and regulation-sensitive cards quarterly.

<!-- BEGIN INTERVIEW REFERENCE PLAN -->

## Reference-informed additions to develop

These are proposed writing briefs informed by interview examples. The examples are reference material only; no completed case, measured result or finished cheat sheet is implied. [Planning catalog, source limits and authoring gates](../../../../00_REFERENCE_INFORMED_CASE_PLAN.md).

| Related brief | Proposed cheat-sheet scope | Checks / counterexamples to include | Reference only | Status |
|---|---|---|---|---|
| PLAN-2026-IR036 | Conservative entity resolution; typed claim-actor graphs; motifs and supernodes; point-in-time features; reversible merges | Temporal baseline comparison, precision at capacity, mature outcomes, sampled low-risk audits and appeal/false-positive slices. Boundary: Connectivity and shared providers are not proof of fraud; source reference notes are unavailable and labels are delayed/selected. | [REF-IR036](../../../../../INTERVIEW_CASES/Graph_Rings/Fraud_Ring_Detection/Claims_Fraud_Ring_Case.md) | planned; not written |

Each future cheat sheet needs its own explanation, preconditions, formulas or algorithms where relevant, failure signals and checked examples. Do not copy the interview narrative or its scenario figures into a reference sheet.

<!-- END INTERVIEW REFERENCE PLAN -->
