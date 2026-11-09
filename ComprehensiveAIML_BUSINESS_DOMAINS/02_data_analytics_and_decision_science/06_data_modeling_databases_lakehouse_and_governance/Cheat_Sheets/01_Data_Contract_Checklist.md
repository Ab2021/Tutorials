---
collection: "data-modeling-databases-lakehouse-and-governance"
file_id: "01"
topic: "Data contracts: grain, quality, delivery, and change control"
level: 1
status: "draft"
version: "1.1"
created: "2026-11-08"
reviewed: null
review_trigger: "A producer changes schema, semantics, delivery, ownership, or quality objective."
notation_style: "plain-text"
prerequisites: ["Module 02 SQL and basic relational keys"]
pattern_tags: ["data-product", "governance", "reconciliation"]
source_status: "OPEN"
owner_role: "Data product owner"
source_index: "INDEX.md"
---

# Data Contracts: Grain, Quality, Delivery, and Change Control

> **Purpose:** Specify and test an agreement that makes a dataset safe for a named downstream decision rather than merely parseable.
> **Prerequisites:** Relational keys, timestamps, SQL aggregation, and the distinction between a record and a business entity.
> **Key outcome:** Given a producer, consumer, and decision, write a contract, compute its quality signals, classify a change, and decide whether to block, quarantine, or release data.
> **This file does NOT cover:** A vendor-specific catalog, legal interpretation, or a guarantee that an upstream business process is correct.
> **Level:** 1 — Foundations.
> **Version / reviewed:** 1.1 / not independently reviewed.
> **Evidence:** DERIVED invariants and VERIFIED illustrative arithmetic; source bodies listed in `directory.txt` do not provide an accessible data-contract lesson.
> **Retrieval cue:** If a dashboard changes after an apparently compatible release, start at Section 4 and Section 8.

## 1. File Header

A data contract is an explicit, versioned agreement between a producer and identified consumers. It declares the decision grain, identity, schema, meaning, availability clock, quality objective, access boundary, and change process. A schema registry alone is not a contract: a field can remain `decimal` while changing from tax-inclusive to tax-exclusive value.

The relevant decision is not “can a pipeline run?” It is “may this consumer use this version for this decision at this time?” A contract therefore names a decision owner and a stop rule. A report can be technically fresh yet unsafe if its population silently excludes a channel.

**Takeaway:** Treat data as a product with a decision-specific acceptance rule, not as an anonymous table.

## 2. Concept Map / Mental Model

```text
[Business event] -> [producer writes versioned record]
                         | identity, grain, event/processing time
                         v
                 [contract validation]
               /          |             \
          schema       quality         semantics
            fail         fail             fail
             |            |                |
        reject/quarantine/hold decision and escalate owner
                         |
                       passes
                         v
                [consumer transformation]
                         |
                         v
               [metric/model/operational action]
                         |
                         v
              [reconciliation and change feedback]
```

The contract boundary is before a consumer silently repairs meaning. Producers own truthful production; consumers own declared use and independent reconciliation. A platform may operate checks, but it cannot infer the intended denominator or allowed lateness.

**Takeaway:** Validation must cover identity, meaning, and timeliness; passing syntax is only one branch.

## 3. Foundational Definitions

| Term | Meaning | Units / type | Nonexample |
|---|---|---|---|
| Grain | What one row or event represents | business statement | “one customer” when rows are order lines |
| Business key | Identifier of the real-world entity or event | stable tuple | a randomly regenerated load ID |
| Version | Declared mutation order for one business key | ordered integer or timestamp with tie rule | ingestion timestamp alone |
| Event time | When the represented fact occurred | UTC instant | warehouse load completion time |
| Processing time | When this system observed the fact | UTC instant | purchase time |
| Expected count `E` | Agreed number of records expected for a window | records/window | previous batch size guessed after loss |
| Received count `R` | Records delivered in the declared scope | records/window | all historical rows |
| Valid count `V` | Received records satisfying stated rules | records/window | rows that merely deserialize |
| Freshness age `a` | `now - latest complete watermark` | time | age of the newest individual row |
| Compatibility | Whether a named consumer can safely interpret a proposed change | per consumer/version | “the database accepted the DDL” |

Example grain: `order_line_snapshot` means one row per `(order_id, line_id, version)`. Summing `line_amount` across all versions double-counts revisions. A “latest snapshot” consumer must state its selector and tie rule.

**Takeaway:** Every metric starts with an entity, a population, and an as-of rule.

## 4. Core Formulas with Full Derivation

### 4.1 Base: completeness and validity

Validity before use: all records are assigned to one declared window; `E` is independently knowable; duplicate policy is applied before counting `R`; and a record either passes or fails the published rule set.

Derivation: completeness asks what fraction of the expected population arrived, so divide delivered records by expected records. Validity asks what fraction of delivered records is usable, so divide valid records by delivered records.

```text
FORMULA [DC-01]: c = R / E
WHERE: c completeness [dimensionless, 0..1 when R<=E];
       R received records [records/window]; E expected records [records/window].
CONSTRAINT: E refers to the same population/window as R; an unknown E makes c unknown.
NUMERICAL EXAMPLE: E=10,000 records/day; R=9,800 records/day; c=9,800/10,000=0.98.
INTERPRETATION: two percent of the expected population is absent; it is not evidence that the present rows are valid.
```

```text
FORMULA [DC-02]: v = V / R
WHERE: v validity [dimensionless]; V valid received records [records/window].
CONSTRAINT: validation rules and duplicate treatment are versioned; R>0.
NUMERICAL EXAMPLE: V=9,604 and R=9,800; v=9,604/9,800=0.98.
INTERPRETATION: 196 delivered records violate at least one declared rule.
```

The usable share of the expected population is `u = V / E = (V/R)(R/E) = vc`; here `u=0.9604`. Independent check: `9,604 / 10,000 = 0.9604`. Units cancel in each ratio. This is ILLUSTRATIVE and arithmetic-VERIFIED, not a production measurement.

Boundary: `R>E` can be legitimate replay or a duplicate failure; reporting `c>1` without a duplicate policy hides the problem. `c=1` does not establish semantic correctness.

### 4.2 General: freshness against a complete watermark

```text
FORMULA [DC-03]: a = t_now - w_complete
WHERE: a freshness age [minutes]; t_now current UTC time; w_complete latest watermark for which the producer asserts the declared population is complete.
CONSTRAINT: clocks and timezone are specified; “latest event” is not a completeness watermark.
NUMERICAL EXAMPLE: t_now=10:35 UTC and w_complete=10:20 UTC; a=15 minutes.
INTERPRETATION: a 10-minute objective is breached even if a row arrived at 10:34.
```

A freshness objective needs a population and consequence: “order snapshots complete within 10 minutes for US web orders” is actionable; “low latency” is not. Late facts may be accepted into history while excluded from an intraday decision until recomputation.

### 4.3 Variants: uniqueness and reconciliation

For a declared key, let `D` be distinct accepted keys and `A` accepted records after retries. Duplicate excess is `d = (A-D)/A` when `A>0`. It measures record-level repetition, not business duplication if the key definition is wrong.

For an amount reconciliation, compare independently computed totals on the same cutoff: `delta = sum_source - sum_consumer`. A zero delta can conceal offsetting row errors, so pair it with key-level anti-joins and sampled semantic checks.

### 4.4 Limits and counterexample

A contract cannot prove that a source system recorded the true real-world event. It can prove only that delivered data conforms to declared, testable properties. A null-rate rule also fails as a universal quality metric: a sudden drop to zero nulls can signal a coercion-to-default bug rather than improvement.

**Takeaway:** Ratios are decision controls only when their population, window, key policy, and consequence are explicit.

## 5. Worked Numerical Example

**Scenario — illustrative; not a company measurement.** A daily `order_line_snapshot` contract expects 10,000 unique latest-version lines. The producer delivers 9,800 rows; 24 are duplicate keys and 172 fail currency, amount, or required-status rules after deduplication.

1. Deduplicate first: `A=9,800`, `D=9,776`; duplicate excess is `(9,800-9,776)/9,800=0.002449`, or 0.2449%.
2. Let `R=9,776` unique received records and `V=9,604` valid records.
3. Completeness is `9,776/10,000=0.9776` (97.76%).
4. Validity is `9,604/9,776≈0.982406` (98.2406%).
5. Usable expected coverage is `9,604/10,000=96.04%`.
6. With a contract minimum of 99% usable coverage, hold the downstream financial metric and publish the prior certified value with its as-of date rather than imputing missing orders.

Independent arithmetic: `9,776 - 9,604 = 172`; `0.9776 × 0.982406≈0.9604`. Sensitivity: if the expected count was incorrectly inferred as 9,800, apparent completeness becomes 100%, masking 200 missing lines. The binding dependency is therefore the independent expectation ledger, not a looser threshold.

**Takeaway:** Compute quality after identity normalization, then make the decision consequence automatic and visible.

## 6. Visual / Geometric Intuition

Think of each quality measure as a nested set:

```text
Expected business population (E=10,000)
 └─ unique delivered population (R=9,776)
     └─ contract-valid population (V=9,604)
         └─ consumer-eligible population (may be smaller for authorization)
```

The gaps answer different questions: `E-R` is delivery or expectation loss; `R-V` is rule failure; valid-but-ineligible records indicate a consumer access or purpose boundary. Collapsing them into a single “quality score” loses the owner and repair path.

**Takeaway:** Preserve the layers of loss so the right team can repair the right boundary.

## 7. Comparison / Selection Table

| Situation | Choose | Why | Do not choose | Fallback |
|---|---|---|---|---|
| Stable analytical facts with revisions | versioned snapshots plus latest selector | preserves audit and correction history | overwrite without mutation semantics | rebuild latest view from immutable versions |
| Low-latency operational event | append event with idempotency key | permits replay and deduplication | assuming global arrival order | partition-local order and reorder window |
| New optional field | additive version after consumer audit | parser compatibility may be enough only for audited consumers | calling it safe because field is nullable | dual publish with expiry |
| Unit or grain change | breaking version | protects arithmetic and joins | type-compatible in-place reuse | new field/version and migration plan |
| Unknown expected population | reconciliation and explicit unknown completeness | avoids false 100% score | `R/R` completeness | create source control total |

**Takeaway:** Compatibility is consumer-specific and semantic, not a property of syntax alone.

## 8. Failure Modes & Edge Cases

### F-01 — Grain changed while schema stayed valid
- **Competing causes:** a real sales spike; a join fan-out; a new row grain.
- **Discriminating check:** compare distinct business keys, rows per key, and source control totals before/after the release.
- **Bounded fix:** freeze the affected aggregate; restore the prior transformation or select a declared latest version.
- **Prevention and verification:** require grain tests in CI; verify anti-join and aggregate reconciliation on a replayed cutoff.

### F-02 — “Complete” batch lacks a trustworthy denominator
- **Competing causes:** upstream loss; delayed source close; expected-count query itself filtered a segment.
- **Discriminating check:** obtain an independently produced control total and compare inclusion predicates.
- **Bounded fix:** mark completeness `unknown`, not 100%; prevent decision release if that is a hard requirement.
- **Prevention and verification:** version the expectation source and test it against known missing-partition fixtures.

### F-03 — Compatible schema change changes meaning
- **Competing causes:** promotion effect; currency conversion change; tax-inclusion semantic change.
- **Discriminating check:** compare field documentation, unit distribution, and a stratified sample of business records.
- **Bounded fix:** isolate the new semantic version and recompute only with an approved mapping.
- **Prevention and verification:** semantic review by producer and consumers; verify parallel totals under both definitions.

### F-04 — Late or duplicate delivery corrupts a snapshot
- **Competing causes:** producer retry; backfill; clock skew; missing version tie rule.
- **Discriminating check:** inspect business key, version, event time, ingestion time, and payload hash.
- **Bounded fix:** idempotent merge using the contract’s ordering rule; quarantine ties rather than arbitrary last-write-wins.
- **Prevention and verification:** replay a duplicate/out-of-order fixture and reconcile selected latest rows to the source ledger.

### F-05 — Ownership exists on paper but no responder acts
- **Competing causes:** alert route failure; unclear severity; quality signal not mapped to a decision.
- **Discriminating check:** run an escalation drill with a deliberately breached nonproduction contract.
- **Bounded fix:** assign a named role, response target, and hold/release authority.
- **Prevention and verification:** quarterly drill records acknowledgement, diagnosis, and closure evidence.

**Takeaway:** A useful failure entry distinguishes causes and ends with a bounded verification, not merely an alert.

## 9. Complexity & Cost Analysis

Schema and range checks are normally `O(n)` in records. Exact uniqueness requires state proportional to distinct keys, `O(D)` space; a sorted or partitioned uniqueness check may trade compute for bounded memory. Referential checks cost depends on lookup/index strategy and can become a remote join bottleneck.

Illustrative sizing: at 50,000 records/minute, retaining a 16-byte key and 8-byte version for one hour requires at least `50,000 × 60 × 24 = 72,000,000` bytes, about 68.66 MiB, before hash-table overhead, payload, replication, or checkpoints. Independent conversion uses `72,000,000 / 1,048,576≈68.66`. This is capacity arithmetic, not a platform benchmark.

For high-volume streams, sample semantic checks can detect regressions but cannot certify full coverage; reserve exact controls for invariants whose failure blocks a decision. Estimate the false-release risk before replacing a full reconciliation with sampling.

**Takeaway:** Price state, scans, and repair time; a check that cannot finish before the decision deadline is not a complete control.

## 10. Key Theorems / Results Reference

| Result | Use | Essential condition | Boundary |
|---|---|---|---|
| Primary-key uniqueness | identity control | key represents declared business identity | surrogate uniqueness does not prove semantic uniqueness |
| Referential integrity | parent/child existence | same snapshot or declared temporal rule | valid foreign key can point to obsolete entity state |
| Conservation reconciliation | compare independently derived counts/totals | equal population and cutoff | equal totals can hide offsetting row errors |
| Idempotence | replay safely | same input and key/order rule yields same state | non-idempotent side effects need a transaction or compensating action |
| Backward compatibility | old consumer can parse/use new producer version | consumer behavior audited | parseability does not establish metric equivalence |

Primary references to consult before claiming product-specific behavior: [Apache Avro specification](https://avro.apache.org/docs/), [OpenLineage specification](https://openlineage.io/docs/), and [scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) for training-only preprocessing. Access/review date: 2026-11-09. These support general mechanisms, not a measurement for this illustrative scenario.

**Takeaway:** Named results constrain a mechanism; they are not evidence that a particular pipeline satisfies it.

## 11. Interview / Exam Questions

1. **Why is a schema registry insufficient?** It cannot express grain, unit, denominator, ownership, or release consequence; a field can parse while its meaning changes.
2. **How would you define a row’s grain?** State the business statement, key tuple, mutation/version rule, and time semantics; validate with duplicate and aggregation examples.
3. **Compute usable coverage for `E=2,000`, `R=1,980`, `V=1,940`.** `V/E=1,940/2,000=0.97`, or 97%; the result assumes matching population/window.
4. **Why deduplicate before validity reporting?** Otherwise retries can inflate the denominator and conceal a missing or invalid unique business record.
5. **A nullable field was added. Is that safe?** Only after auditing each consumer’s behavior and semantic use; syntactic backward compatibility is not enough.
6. **What is the most dangerous freshness ambiguity?** Confusing newest received event with the latest complete watermark; the former can hide a missing partition.
7. **When should a consumer block?** When a stated hard control fails and the decision harm exceeds the approved stale-data fallback; document who may override.
8. **How do you diagnose a sudden revenue increase after a contract release?** Compare grain, joins, unit definition, population, and source totals before attributing it to business performance.
9. **What changed constraint would reverse your design?** If exact uniqueness state exceeds the deadline or budget, use partitioned reconciliation or a delayed certified view; do not silently weaken a hard invariant.
10. **What does idempotency not solve?** It does not decide correct event ordering, prevent semantic duplication under different keys, or roll back external side effects without a transaction boundary.
11. **How do you verify a migration?** Dual-run bounded data, compare declared controls at the same cutoff, investigate deltas, and retire the old path only after consumer acceptance.
12. **What is a plausible wrong method?** Treating a 100% schema-pass rate as data quality; it misses population loss, stale watermarks, and semantic drift.

**Takeaway:** Strong answers name the grain, cutoff, denominator, failure consequence, and evidence needed to release data.

## 12. Quick Reference Card

| Check | Rule | Use when | Critical assumption | Avoid when |
|---|---|---|---|---|
| Identity | key + mutation/order rule | replay, merge, snapshots | key is business-stable | key is only a load surrogate |
| Completeness | `R/E` | expected population known | same scope/window | E is inferred from R |
| Validity | `V/R` | rules are versioned | duplicate policy applied | R is zero or ambiguous |
| Usable coverage | `V/E` | release decision | valid means decision-eligible | authorization differs by consumer |
| Freshness | `now - complete watermark` | intraday actions | watermark certifies population | latest row is mistaken for complete |
| Change class | per-consumer semantic audit | producer release | consumer inventory current | database DDL is sole evidence |
| Reconciliation | independent totals + keys | financial/critical metrics | shared cutoff and population | offsets can hide row errors |
| Release | hold/quarantine/fallback owner | hard-control breach | authority and escalation known | alert has no decision consequence |
