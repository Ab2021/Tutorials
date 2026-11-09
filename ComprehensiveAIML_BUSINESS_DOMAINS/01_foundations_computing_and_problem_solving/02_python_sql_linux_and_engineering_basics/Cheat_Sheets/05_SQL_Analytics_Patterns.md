---
collection: "python-sql-linux-engineering-basics"
file_id: "05"
topic: "SQL analytics patterns"
level: 1
status: "draft"
version: "1.0"
created: "2026-11-09"
reviewed: null
review_trigger: "schema, grain, timezone, or metric-definition change"
notation_style: "plain-text"
prerequisites: []
pattern_tags: ["joins", "windows", "metric-grain"]
source_status: "OPEN — course-topic draft; no source body asserted"
owner_role: "analytics engineer"
source_index: "INDEX.md"
---

# SQL Analytics Patterns: Grain, Joins, and Windows

> **Purpose:** Write an auditable analytical query whose row grain, join cardinality, time boundary, and metric denominator are explicit.
> **Prerequisites:** tables, primary keys, and basic percentage arithmetic.
> **Key outcome:** Given event and entity tables, select a join/window pattern and detect a duplicated denominator.
> **This file does NOT cover:** database-specific optimizer tuning, transaction isolation, or production access policy.
> **Level:** 1 Foundations
> **Version / reviewed:** 1.0 / not reviewed
> **Evidence:** DERIVED relational reasoning; ILLUSTRATIVE rows; no executed database query.
> **Retrieval cue:** If a KPI changes after a join, begin at Section 4 and Section 8.

## 2. Concept Map / Mental Model

```text
[Business question] -> [one declared row grain]
        |                    |
        v                    v
[time/eligibility rule] -> [join cardinality check]
        |                    |
        +--many-to-many?--> [aggregate/deduplicate before metric]
        v
[window partition/order/frame] -> [numerator and denominator] -> [reconciled result]
```

**Takeaway:** A valid SQL statement begins with a grain contract, not `SELECT *`.

## 3. Foundational Definitions

| Term | Meaning | Failure if omitted |
|---|---|---|
| Grain | what one output row represents | duplicated counts |
| Primary key | unique stable row identifier | accidental many-to-many join |
| Event time | when business event occurred | late data moved into wrong period |
| Processing time | when system recorded event | false temporal ordering |
| LEFT JOIN | retain every left row, add right match if present | `WHERE` filter can turn it into inner join |
| Window partition | group over which an analytic function runs | ranks cross unrelated entities |
| Frame | rows considered around current row | cumulative becomes whole-partition total |

**Takeaway:** Keys, time semantics, and denominators are part of the metric definition.

## 4. Core Formulas with Full Derivation

Validity before use: each eligible order is represented once after the declared reconciliation rule.

```text
FORMULA [SQL-01]: conversion = n_completed / n_eligible
WHERE: n_completed [orders]; n_eligible [orders] over identical cohort/window.
CONSTRAINT: duplicate joined rows must not inflate either count.
NUMERICAL EXAMPLE: 80 completed among 100 eligible orders gives 0.80 = 80%.
INTERPRETATION: 160 joined rows do not establish 160 orders.
```

Derivation: a rate is its successful eligible unit count divided by all eligible units. If joining orders to line items creates multiple rows/order, aggregate or use a distinct/order-level subquery before counting.

For a running total, with ordered values `x_i`, define `S_t = sum(i=1..t) x_i`. This requires a deterministic order and a `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` frame; default peer behavior differs by SQL engine.

```text
FORMULA [SQL-02]: S_t = S_(t-1) + x_t
WHERE: x_t [currency/event] and S_t [currency] in one partition ordered by event key.
CONSTRAINT: ties need a stable secondary key; do not order only by a nonunique timestamp.
NUMERICAL EXAMPLE: x=(5,7,-2) dollars yields S=(5,12,10) dollars.
INTERPRETATION: the final balance is $10, independently checked by 5+7-2.
```

**Takeaway:** A rate needs a shared cohort; a running total needs a deterministic order/frame.

## 5. Worked Numerical Example

**ILLUSTRATIVE tables.** Orders: O1 complete, O2 pending, O3 complete. Line items: O1 has two rows, O2 one, O3 three. A naive joined `COUNT(*)` gives six rows and four completed rows, producing 4/6=66.7%; this is not order conversion. Aggregate to one row/order or count distinct IDs: completed orders are O1 and O3, eligible orders are O1–O3, so `2/3=66.7%`. The equal percentage is coincidence; changing O3 to four line items produces 5/7=71.4% while true order conversion remains 66.7%.

```sql
WITH order_grain AS (
  SELECT order_id, MAX(status) AS status
  FROM orders
  GROUP BY order_id
)
SELECT
  COUNT(*) FILTER (WHERE status = 'complete') * 1.0 / COUNT(*) AS conversion
FROM order_grain;
```

`MAX(status)` is only illustrative and invalid for arbitrary status ordering; a real query must select the latest state by event time plus stable sequence key.

**Takeaway:** Reconcile state at the intended grain before computing a KPI.

## 6. Visual / Geometric Intuition

A join is an expansion map: one order point becomes several line-item points. Counting after expansion measures items unless the query folds points back to orders. A window is a moving view across an ordered line; changing the frame changes the view, not merely formatting.

**Takeaway:** Join expansion and window frame are semantic transformations.

## 7. Comparison / Selection Table

| Need | Pattern | Use when | Avoid when |
|---|---|---|---|
| retain entities with no match | LEFT JOIN | missing match is meaningful | right-side `WHERE` removes null rows |
| latest state per entity | `ROW_NUMBER()` over ordered events | stable event/sequence key exists | timestamp ties are unresolved |
| top N per group | `ROW_NUMBER` or `RANK` | tie policy declared | ties must all be retained but `ROW_NUMBER` used |
| cumulative metric | `SUM() OVER` explicit frame | order deterministic | duplicate/late events unhandled |
| cohort KPI | pre-aggregate to entity grain | denominator is entity | item-level join is counted as entity |

**Takeaway:** Choose a query pattern from its grain and tie policy, not its syntax familiarity.

## 8. Failure Modes & Edge Cases

| ID | Symptom | Competing causes | Diagnostic | Fix | Verify |
|---|---|---|---|---|---|
| SQL-F01 | KPI jumps after join | duplicate key; true volume change | compare row count and `COUNT(DISTINCT key)` | pre-aggregate/deduplicate | reconciles to source entity count |
| SQL-F02 | LEFT JOIN loses unmatched rows | right-side predicate in `WHERE`; no unmatched rows | count null right keys before/after filter | move predicate into `ON` or preserve null | expected unmatched IDs remain |
| SQL-F03 | latest record changes between runs | timestamp tie; late event | inspect ordering keys/event time | add sequence and late-data policy | deterministic replay matches |
| SQL-F04 | running total differs by engine | default RANGE frame; peer timestamps | specify `ROWS` frame and stable key | explicit frame/order | hand trace matches output |
| SQL-F05 | rate exceeds 100% | numerator/denominator grains differ | list IDs in each set | align cohort/exclusions | rate bounded as expected |

**Takeaway:** Reconcile IDs and row counts before trusting a polished aggregate.

## 9. Complexity & Cost Analysis

A grouped aggregation commonly needs work proportional to input rows `n`, plus sorting or hashing depending on engine/plan. A window with an ordered partition may require sorting, commonly O(n log n) absent a useful ordering/index; that is a complexity model, not a benchmark. Filter early only when it preserves the declared cohort. Materializing an entity-grain intermediate can reduce accidental fan-out and make audit cheaper.

**Takeaway:** Query cost and metric correctness both depend on cardinality and ordering.

## 10. Key Theorems / Results Reference

| Result | Use | Limit |
|---|---|---|
| Relational join cardinality | predict row multiplication | requires actual uniqueness constraints |
| Window partition/order/frame semantics | define analytic result | details vary by SQL dialect |
| Inclusion-exclusion reasoning | reconcile cohort counts | does not resolve bad identifiers |

See the SQL dialect’s official documentation for exact syntax/version; this sheet does not certify a particular engine.

**Takeaway:** SQL semantics are product/version-specific; validate on the intended engine.

## 11. Interview / Exam Questions

1. What is output grain? The real-world unit represented by one output row.
2. Why can a join corrupt a rate? It can multiply one denominator entity into several rows.
3. How do you detect fan-out? Compare total rows with distinct keys before and after each join.
4. Compute true conversion for two complete of three eligible orders. `2/3=66.7%`.
5. When does LEFT JOIN act like INNER JOIN? When a right-table predicate in `WHERE` rejects null matches.
6. Why are timestamp ties dangerous? “Latest” becomes nondeterministic without a secondary sequence.
7. What does `ROWS` specify? Physical ordered-row frame, avoiding peer grouping ambiguity.
8. What is a plausible wrong fix for duplicates? `DISTINCT *`; it may hide real duplicates and change grain unpredictably.
9. What changes with late events? Define event-time cutoff, correction, and reporting revision policy.
10. How would you validate a KPI? Hand-check sampled entity IDs and reconcile every intermediate count.

**Takeaway:** Explain the row contract before explaining the SQL syntax.

## 12. Quick Reference Card

| Cue | Rule | Avoid |
|---|---|---|
| Metric | declare grain, cohort, numerator, denominator | counting after unknown fan-out |
| Join | verify key uniqueness/cardinality | assuming keys are unique |
| LEFT JOIN | put matching predicate in `ON` when retaining nulls | null-rejecting `WHERE` filter |
| Latest | order by event time plus stable tie-breaker | timestamp-only choice |
| Window | state partition, order, and frame | dialect default frame |
| Validate | reconcile IDs/counts with a hand trace | trusting one aggregate |
