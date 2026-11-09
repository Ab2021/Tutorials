# Module 27 Case Studies — Data Quality, Observability and Reliability Engineering

**Status:** planning catalog; no authored `CASE-*` files. Future cases must meet the [course standards](../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and [case master](../../../Templates/CASE_STUDY_CONTENT_MASTER_TEMPLATE.md).

## 1. Collection metadata

| Field | Value |
|---|---|
| Domain | data contracts, lineage, SLOs, incidents, and recovery |
| Evidence boundary | illustrative/proposed unless a primary incident artifact supports a narrower claim |
| Review trigger | schema, producer, lineage, freshness objective, cache epoch, or replay semantics change |

## 2. Case catalog

| Case ID | Linked title | Domain | Case type | Decision | Primary pattern | Secondary patterns | Evidence | Stage | Last reviewed | Result/limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| PLAN-27-01 | Broken upstream schema incident | data reliability | incident | quarantine/replay/roll forward | correctness-contract | idempotent-side-effects | not-measured | proposed | — | requires input/output reconciliation |
| PLAN-27-02 | Silent label corruption investigation | data reliability | incident | gate retraining | distribution-shift | evaluation-mismatch | not-measured | proposed | — | requires adjudication evidence |
| PLAN-2026-IR001 | Gate retraining on corpus provenance and long-tail quality | data/training | hybrid | admit or quarantine documents | distribution-shift | data-leakage; evaluation-mismatch | not-measured | proposed | 2026-11-06 (planning) | authorship remains unestablished |
| PLAN-2026-IR033 | Rebalance caches without overwhelming the authoritative store | reliability | hybrid | preserve freshness under migration | materialized-view | resource-admission; snapshot-consistency | not-measured | proposed | 2026-11-06 (planning) | local single-flight is not fleet coordination |

## 3. Canonical reusable patterns

- **Correctness contract:** accept a record only when its schema, semantic checks, and lineage contract hold.
- **Materialized view:** make freshness, invalidation, version fencing, and authoritative ownership explicit.
- **Distribution shift:** evaluate changed data coverage separately from service availability.

## 4. Similarity / contrast map

| Cluster | Shared reasoning | Critical distinction |
|---|---|---|
| schema and labels | provenance, quarantine, replay | syntactic validity does not establish label correctness |
| cache migration and data incidents | version fencing and recovery | expected key movement is not the request miss distribution |

## 5. Questions and future-case backlog

1. What invariant distinguishes a stale read from a corrupted value?
2. Which replay is idempotent and how is its result reconciled?
3. What observation could falsify the claim that a data change is harmless?
4. How are SLO burn, freshness, and business impact separated?

## 6. Review / retirement log

| Date | Item | Decision | Reason |
|---|---|---|---|
| 2026-11-09 | catalog | retained as planned | no primary incident evidence or replay artifacts |

## 7. Pre-publish checks

- [ ] Link an authored 15-section `CASE-*` file with contract, lineage, recovery, and post-incident evidence.
- [ ] Check timestamps, duplicate handling, reconciliation arithmetic, fault boundaries, and interview answers.
