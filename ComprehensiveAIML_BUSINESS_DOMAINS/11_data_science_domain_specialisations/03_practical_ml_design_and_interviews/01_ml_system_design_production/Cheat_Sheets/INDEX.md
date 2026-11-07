# ML System Design Production — Cheat Sheets

**Status:** The feature-store sheet is an unreviewed draft; other references remain planned. Follow the [course authoring and numerical standards](../../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and linked cheat-sheet master. A reference card does not replace evidence or evaluation in a full case.

## 1. Overview

For engineers designing operational ML systems, this collection links decision-time correctness, system boundaries and production tradeoffs. The current draft covers feature-store inputs, availability, retrieval and monitoring. Readiness means reproducing a point-in-time join and identifying when online and offline feature values are not comparable.

## 2. Topic ladder

| Level | Reference | Outcome and prerequisite | Status |
|---|---|---|---|
| Practitioner | [Feature store design](03_Feature_Store_Design.md) | Trace event-time and availability-time joins and diagnose training-serving skew; requires ML data and serving fundamentals | Draft; unreviewed |
| Planned | ML system-design framework | Define decisions, inputs, metrics and failure signals | Not written |
| Planned | Capacity estimation | Estimate throughput, state and bottlenecks under stated workload | Not written |
| Planned | Model launch gate | Specify evaluation, rollout and rollback criteria | Not written |
| Planned | Common Issues | Collection-wide distinct diagnostic reference | Not written |

## 3. Quick topic lookup

| Question | Reference | Condition and warning |
|---|---|---|
| Which historical feature value could a model have used? | [Worked example](03_Feature_Store_Design.md#5-worked-numerical-example) | Availability time and event time are distinct; a later correction must not leak backward. |
| Why do offline and online values differ? | [Failure modes](03_Feature_Store_Design.md#8-failure-modes--edge-cases) | Compare the same entity, cutoff, source version and transformation before attributing cause. |
| How is a feature selected? | [Core formulas and mechanisms](03_Feature_Store_Design.md#4-core-formulas-with-full-derivation) | A historical query depends on a defined availability and revision contract. |

## 4. Reading paths

- **Beginner:** Review supervised learning, database joins, timestamps and service requests; read the sheet's definitions. The broader system-design framework remains planned.
- **Practitioner:** Read the complete feature-store sheet; reproduce its illustrative point-in-time example and document online/offline parity and freshness for a real feature.
- **Advanced:** Analyze delayed arrivals, corrections, backfills, deletion and access control; design replay, monitoring and rollback drills. Capacity and launch-gate references remain planned.

## 5. Prerequisite skills

Understand entity keys, event versus ingestion and availability timestamps, training/validation leakage, deterministic transformations, keyed serving, freshness budgets and basic throughput/storage arithmetic.

## 6. What this collection does not cover

The current draft is not a product-specific deployment guide, measured benchmark, complete governance policy or universal feature-store requirement. System-design framework, capacity worksheet, launch gate and standalone Common Issues material remain planned.

## 7. Master symbol glossary

| Term | Meaning | Scope / warning | Reference |
|---|---|---|---|
| Entity key | Identifier used to join a feature to an example or request | Identity and consent scope must be stable and authorized | [Definitions](03_Feature_Store_Design.md#3-foundational-definitions) |
| Event time | Time a source event occurred | May precede the time the feature became queryable | [Definitions](03_Feature_Store_Design.md#3-foundational-definitions) |
| Availability time | Time a particular version could be used | A later correction cannot be treated as historically available | [Worked example](03_Feature_Store_Design.md#5-worked-numerical-example) |
| Freshness | Age of a defined value relative to a defined decision time | A recent timestamp does not by itself establish correctness | [Feature store sheet](03_Feature_Store_Design.md) |

## 8. Quick Reference Card directory

| Card | Core rule | Use and critical condition |
|---|---|---|
| [Feature store](03_Feature_Store_Design.md#12-quick-reference-card) | Historical feature values must be available by the prediction cutoff. | Define event time, availability, revision order and feature identity consistently. |

## 9. Common Issues locator

The standalone Common Issues artifact is **planned, not written**; no collection-wide issue count is claimed. The draft's [failure modes](03_Feature_Store_Design.md#8-failure-modes--edge-cases) contain hypothetical diagnostics, not observed production incidents.

## 10. Sources, versions and maintenance

The draft uses illustrative traces and sizing, not measured outcomes. Review when event or availability semantics, transformations, versions, access controls, model contracts or data retention change. The [source coverage brief](../../../../00_SOURCE_COVERAGE_DATA_SCIENCE.md#ds-ml-design) is a topic plan, not evidence that the unavailable original files were inspected.

**Planned reference-informed brief:** [PLAN-2026-IR026](../../../../00_REFERENCE_INFORMED_CASE_PLAN.md) covers cache policies, range queries, sketches, search, graphs, scheduling and online windows. Check operation traces against exact small-data oracles and distinguish approximation from distributed-service guarantees. The [interview example](../../../../../INTERVIEW_CASES/DSA/index_basics/INTERVIEW_CASE_STUDIES.md) is reference material only, not a finished sheet or measured system.

## 11. Collection acceptance checklist

- [ ] Planned references and a distinct Common Issues artifact are authored and reviewed.
- [ ] Each sheet has all 12 sections, checked calculations or traces, failure diagnostics, answered interview questions and a final card.
- [ ] Decision-time, authorization, evidence and source limitations are explicit.
- [ ] Links, glossary, paths, issue locations and editorial statuses match actual files.
