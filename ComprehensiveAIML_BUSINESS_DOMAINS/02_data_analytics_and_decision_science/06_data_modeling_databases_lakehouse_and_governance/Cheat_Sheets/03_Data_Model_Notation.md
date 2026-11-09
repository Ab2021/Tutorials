---
collection: "data-modeling-databases-lakehouse-and-governance"
file_id: "03"
topic: "Data model notation"
level: 1
status: "draft"
version: "1.0"
created: "2026-11-09"
reviewed: null
review_trigger: "A relationship, key, temporal rule, or business definition changes."
notation_style: "plain-text"
prerequisites: []
source_status: "OPEN"
owner_role: "Data architect"
source_index: "INDEX.md"
---

# Data Model Notation

> **Purpose:** Read and write a compact model that preserves entity identity, relationship cardinality, and temporal meaning.
> **Prerequisites:** None beyond basic tables and keys.
> **Key outcome:** Convert a business sentence into entities, keys, cardinalities, and an enforceable invariant.
> **This file does NOT cover:** vendor DDL, performance indexes, or a substitute for stakeholder discovery.
> **Level:** 1 — Foundations.
> **Version / reviewed:** 1.0 / not independently reviewed.
> **Evidence:** DERIVED modeling rules and VERIFIED illustrative trace. Inventory consulted: [`directory.txt`](../../../../Templates/directory.txt); no source body is asserted.
> **Retrieval cue:** If two teams disagree about “customer,” start at Section 3.

## 1. File Header

Notation is a compression device for a business agreement. A model must say what exists, how it is identified, whether a relationship is optional, how many related rows may exist, and what time rule applies. It is incomplete when it names tables but leaves “current,” “active,” or “owner” undefined.

**Takeaway:** Model notation is useful only when it resolves a decision-relevant ambiguity.

## 2. Concept Map / Mental Model

```text
Business sentence -> entity -> natural identity -> attributes
                                      |
                                      v
                              relationship/cardinality
                                      |
                                      v
                         time/version and integrity invariant
```

**Takeaway:** Identity precedes attributes; cardinality precedes joins.

## 3. Foundational Definitions

| Notation | Meaning | Constraint | Example |
|---|---|---|---|
| `Customer(customer_id)` | entity and natural identifier | key stable in declared domain | customer account |
| `Order(order_id, customer_id)` | dependent entity | each order references one customer | purchase order |
| `Customer 1 --- 0..* Order` | one-to-many, optional child count | each order has one valid parent | new customer may have no orders |
| `Order * --- * Tag` | many-to-many | needs bridge and semantic policy | order has promotion tags |
| `valid_from, valid_to` | temporal version interval | no overlap per natural key | customer tier history |

“Customer” can mean person, login, household, bill-to account, or legal organization. Pick one and state the handoff for the others.

**Takeaway:** A noun is not an entity definition until its identity boundary is declared.

## 4. Core Formulas with Full Derivation

Validity: an as-of dimension join needs non-overlapping half-open validity intervals and a fact timestamp in the same time basis.

```text
FORMULA [MN-01]: match(f,d) iff d.key=f.key AND d.from <= f.time < d.to
WHERE: f is a fact; d is one dimension version; times are UTC instants.
CONSTRAINT: exactly one matching d is required for a known historical attribute.
NUMERICAL EXAMPLE: tier A [Jan 1, Feb 1), tier B [Feb 1, infinity); fact Feb 1 00:00 matches B.
INTERPRETATION: half-open intervals avoid matching both versions at the boundary.
```

Derivation: the left-closed/right-open convention assigns each instant to the interval beginning at that instant. Boundary: a missing `to` or overlapping load can yield zero or two matches; neither should be silently resolved by arbitrary ordering.

**Takeaway:** Temporal notation must give a deterministic match at change boundaries.

## 5. Worked Numerical Example

ILLUSTRATIVE: customer C1 has tiers Silver `[2026-01-01, 2026-02-01)` and Gold `[2026-02-01, infinity)`. Orders at Jan 31 23:59 and Feb 1 00:00 map to Silver and Gold respectively. Independent check: the first time is less than Feb 1; the second is not. If both intervals were closed, Feb 1 matches twice; if both were open, it matches neither.

**Takeaway:** State interval closure in the model rather than relying on query convention.

## 6. Visual / Geometric Intuition

```text
Jan 1 ---------------- Feb 1 ------------------>
Silver [================) 
Gold                    [======================>
```

**Takeaway:** One boundary belongs to one version.

## 7. Comparison / Selection Table

| Situation | Notation choice | Benefit | Risk |
|---|---|---|---|
| mutable operational entity | normalized parent/child | integrity close to update | analytical joins proliferate |
| stable analytic measurement | fact plus dimensions | visible aggregation grain | history may be oversimplified |
| many memberships | bridge | relationship explicit | measure fan-out |
| historical attributes | interval/version | replayable context | overlap and late corrections |

**Takeaway:** Choose notation that makes the likely error visible.

## 8. Failure Modes & Edge Cases

1. **Ambiguous entity:** person versus account; inspect keys and decision owner; split definitions; verify sample linkage.
2. **Unstated optionality:** orphan child may be data loss or valid lifecycle; anti-join and inspect status; model unknown/late parent; verify lifecycle trace.
3. **Many-to-many hidden as column list:** membership edits lose history; normalize bridge; verify one row per relationship/version.
4. **Overlapping history:** two matching versions; interval-overlap test; quarantine update; verify exactly-one as-of match.

**Takeaway:** Each diagram edge needs a testable data invariant.

## 9. Complexity & Cost Analysis

Checking parent existence is proportional to child rows with indexed parent lookup; checking temporal overlap requires ordering versions per natural key. A bridge grows with relationship edges, so analytical output can exceed base facts. These are algorithmic consequences, not platform benchmarks.

**Takeaway:** Cardinality and history affect cost before SQL is written.

## 10. Key Theorems / Results Reference

A key functionally determines attributes only within its declared relation and time scope. Referential integrity proves a referenced row exists, not that the business assignment is correct. Normal forms reduce certain update anomalies but do not define a business metric.

**Takeaway:** Formal constraints narrow errors; they do not create missing semantics.

## 11. Interview / Exam Questions

1. **Why declare cardinality?** It predicts join behavior and required integrity checks.
2. **What is a natural key?** A business identity, distinct from a warehouse surrogate.
3. **Why half-open intervals?** They assign a boundary to exactly one version.
4. **When use a bridge?** For a real many-to-many relationship with explicit relationship meaning.
5. **What does an FK not prove?** Correct business attribution or temporal validity.
6. **How diagnose fan-out?** Count joined rows per fact key and compare pre/post totals.
7. **What changed constraint matters?** If history is unnecessary, current-state model may suffice; document loss of replayability.
8. **Plausible wrong method?** Calling every repeated customer row a duplicate when the fact grain is order line.
9. **How verify a model?** Exercise create/change/delete and as-of traces against stated invariants.
10. **Why distinguish event and processing time?** Late arrival otherwise rewrites business chronology.

**Takeaway:** Explain every model answer through a business invariant and a test.

## 12. Quick Reference Card

| Element | Record | Avoid |
|---|---|---|
| Entity | business boundary and natural key | noun without identity |
| Relationship | cardinality and optionality | implied join behavior |
| Fact | grain and additive measure rule | summing after uncontrolled bridge |
| History | interval closure and correction rule | overlapping versions |
| Verification | key, FK, cardinality, as-of tests | diagram-only approval |
