---
collection: "data-modeling-databases-lakehouse-and-governance"
file_id: "02"
topic: "Data model grain and join safety"
level: 2
status: "draft"
version: "1.0"
created: "2026-11-09"
reviewed: null
review_trigger: "A business key, relationship cardinality, or snapshot rule changes."
notation_style: "plain-text"
prerequisites: ["01_Data_Contract_Checklist.md"]
source_status: "OPEN"
owner_role: "Analytics engineer"
source_index: "INDEX.md"
---

# Data Model Grain and Join Safety

> **Purpose:** Select normalized or dimensional representations and prove that a join preserves the measure being reported.
> **Prerequisites:** [Data contracts](01_Data_Contract_Checklist.md), relational keys, and aggregation.
> **Key outcome:** Given entity relationships and a metric, declare grain, predict join multiplication, and choose a safe aggregate or bridge.
> **This file does NOT cover:** physical query-engine tuning, vendor syntax, or organization-specific master-data policy.
> **Level:** 2 — Core methods.
> **Version / reviewed:** 1.0 / not independently reviewed.
> **Evidence:** DERIVED relational reasoning and VERIFIED illustrative arithmetic. Inventory link: [`directory.txt`](../../../../Templates/directory.txt); no accessible source body is reconstructed.
> **Retrieval cue:** If revenue jumps after adding a dimension, start at Section 8.

## 1. File Header

A model is a set of statements about entities and relationships, not a diagram decoration. Choose a normalized representation when correctness under frequent mutation and local ownership matters; choose a dimensional representation when a stable analytical grain and understandable aggregation are primary. Both require explicit keys and temporal rules.

**Takeaway:** The safe model is the one whose grain matches the question and whose mutations can be explained.

## 2. Concept Map / Mental Model

```text
[Business question] -> [measure + population + as-of time]
       |                         |
       v                         v
 [declare fact grain] <- [entity keys and cardinalities]
       |                         |
       v                         v
 [join path and bridge] -> [aggregate] -> [reconciliation]
       | fail: many-to-many unknown or changing dimension
       v
 [hold metric; model relationship or deduplicate at declared grain]
```

**Takeaway:** Start from a measurable fact, then add dimensions through declared cardinalities.

## 3. Foundational Definitions

| Term | Meaning | Type | Nonexample |
|---|---|---|---|
| Fact | measurement at a declared grain | row/event | an unlabeled “sales table” |
| Dimension | descriptive context for a fact | entity/version | current customer attributes used for historical analysis without a rule |
| Natural key | business identifier | scalar/tuple | a warehouse surrogate alone |
| Surrogate key | warehouse identifier for a versioned row | integer/string | proof of business uniqueness |
| One-to-many | one parent maps to many children | cardinality | permission to sum parent measures after joining children |
| Bridge | explicit table for many-to-many relationship | factless relation | duplicating allocation silently |
| Slowly changing dimension | versioned descriptive history | temporal relation | overwriting history when historical attribution matters |

**Takeaway:** Keys identify rows; grain explains what the row asserts.

## 4. Core Formulas with Full Derivation

Validity before use: a revenue fact has one row per order line and each line amount is additive exactly once. Let `a_i` be line amount in currency units. Total revenue is the sum over distinct fact keys.

```text
FORMULA [GM-01]: Revenue = sum(i in distinct order_lines) a_i
WHERE: a_i [currency/line]; i is one declared order-line key.
CONSTRAINT: exclude cancelled/refunded status only by published business rule; never sum after an uncontrolled many-to-many join.
NUMERICAL EXAMPLE: lines $40, $60, $100 give $200.
INTERPRETATION: the total is a property of fact keys, not dimension-row count.
```

Use two diagnostics with different denominators. Input-fact coverage is `c_join = distinct matched input fact keys / input fact keys`; it detects rows lost by an inner join. Multiplicity is `m_join = joined rows / distinct matched fact keys`; it detects fan-out among facts that did match. Neither is a correction. A safe required relationship has `c_join=1` and `m_join=1`; a bridge may deliberately have `m_join>1` only if the report is not summing the fact or an allocation policy is declared.

Counterexample: input facts A, B, C have amounts $40, $60, $100. The join drops C and duplicates A, producing A, A, B: output row count is still three, exactly the input count. Yet `c_join=2/3`, `m_join=3/2`, and the naive total is $140 rather than $200. Raw row count therefore cannot show simultaneous loss and fan-out.

Boundary: `c_join=1` does not prove correct attribution; the wrong one-to-one key can still map a fact to the wrong customer. `m_join=1` does not prove no loss because its denominator excludes unmatched input facts.

**Takeaway:** Aggregate at the fact grain, or explicitly allocate before crossing a many-to-many relationship.

## 5. Worked Numerical Example

ILLUSTRATIVE: three order lines total $200. A tagging table links line A ($40) to two tags, B ($60) to one, and C ($100) to one. The joined result has four rows, so `m=4/3=1.3333`. Naively summing gives `$40+$40+$60+$100=$240`, an overstatement of `$40`.

Independent check: the duplicate fact key is A. A safe tag report either counts distinct line keys, allocates A by an explicit 50/50 rule ($20 per tag), or answers a non-additive question such as “lines associated with tag.” Sensitivity: allocating evenly changes per-tag revenue but preserves the $200 total; assigning full $40 to both tags is allowed only if the metric is deliberately “tag-associated revenue” and is not compared to total revenue.

**Takeaway:** A bridge needs a declared attribution policy, not a `DISTINCT` pasted onto every query.

## 6. Visual / Geometric Intuition

```text
Order line facts:       A $40    B $60    C $100
Tag bridge:             A-red, A-sale, B-sale, C-new
Naive joined space:     A appears twice -> area is duplicated
Allocated space:        A $20 red + A $20 sale -> area conserved
```

**Takeaway:** Join fan-out is duplicated measure area; allocation makes the intended duplication explicit.

## 7. Comparison / Selection Table

| Need | Pattern | Strength | Avoid when | Fallback |
|---|---|---|---|---|
| frequent transactional updates | normalized entities | local integrity and reduced update anomaly | broad BI requires many opaque joins | curated dimensional mart |
| repeatable aggregates | star schema fact/dimensions | visible grain and simpler queries | dimension history is undefined | versioned dimension with as-of join |
| many-to-many attribution | bridge plus weights | explicit conservation | weights are arbitrary/unknown | report association counts, not allocated currency |
| mutable descriptive history | type-2/versioned dimension | historical context retained | effective-time overlap unchecked | repair overlaps and reprocess affected facts |

**Takeaway:** Model selection is a workload and mutation decision, not a normalized-versus-star ideology.

## 8. Failure Modes & Edge Cases

1. **Fan-out inflation:** competing causes are a real volume rise or duplicate dimension keys; check join row factor by fact key; contain by blocking aggregate; verify with pre-join fact total.
2. **Inner-join loss:** competing causes are legitimate exclusion or missing dimension; anti-join facts to dimensions; use a declared unknown member while repairing; verify retained fact count.
3. **Current-attribute rewrite:** competing causes are true historical change or overwritten dimension; compare effective-date versions; use an as-of join; verify historical report replay.
4. **Overlapping type-2 windows:** competing causes are bad load order or missing end dates; test one matching dimension version per fact timestamp; quarantine overlap; verify interval constraints.
5. **`DISTINCT` hides corruption:** competing causes are harmless duplicate projection or key collision; inspect affected fact keys and measures; correct relationship; verify reconciliation rather than accepting a lower row count.

**Takeaway:** Diagnose cardinality before applying SQL patches that make totals look plausible.


### Expanded diagnostic investigations

All entries are ILLUSTRATIVE design investigations, not observed incidents. They distinguish competing causes and define a replayable acceptance oracle.

#### DIAG-01 — missing dimension key

- **Trigger/environment:** A bounded release changes a fact, dimension, bridge, or query predicate while the declared fact grain remains the correctness oracle.
- **Symptom:** A join-derived count, amount, slice, or latency differs from the certified comparison at the same cutoff.
- **Competing cause A:** A relationship, key, version, unit, or time invariant was violated for missing dimension key.
- **Competing cause B:** A legitimate business-policy or population change produced the movement without invalidating the model.
- **Discriminating diagnostic:** Preserve pre-join input fact keys, matched fact keys, joined rows, predicate, cutoff, and source revision; compute coverage and multiplicity separately.
- **Expected signal for A:** An anti-join, overlap, key, unit, or cardinality test isolates the violating records.
- **Expected signal for B:** Fact-grain reconciliation passes and the declared business rule explains the changed population.
- **Immediate mitigation:** Hold the affected decision slice or publish its prior certified value; do not use DISTINCT as a cosmetic repair.
- **Correction:** Repair the owner boundary: source contract, relationship model, temporal selector, allocation policy, or query grain.
- **Prevention:** Add a release fixture containing zero, one, many, late, duplicate, and changed-version relationships relevant to this invariant.
- **Verification:** Replay the fixture and require the stated input-fact coverage, matched-fact multiplicity, and fact-grain total before release.
- **Mitigation risk/escalation:** Holding data delays a decision; an automatic fallback can conceal a semantic defect. Escalate when fallback changes a business action.
- **Boundary:** Passing this check establishes consistency with the written model, not truth of an unavailable source body.

#### DIAG-02 — duplicate temporal version

- **Trigger/environment:** A bounded release changes a fact, dimension, bridge, or query predicate while the declared fact grain remains the correctness oracle.
- **Symptom:** A join-derived count, amount, slice, or latency differs from the certified comparison at the same cutoff.
- **Competing cause A:** A relationship, key, version, unit, or time invariant was violated for duplicate temporal version.
- **Competing cause B:** A legitimate business-policy or population change produced the movement without invalidating the model.
- **Discriminating diagnostic:** Preserve pre-join input fact keys, matched fact keys, joined rows, predicate, cutoff, and source revision; compute coverage and multiplicity separately.
- **Expected signal for A:** An anti-join, overlap, key, unit, or cardinality test isolates the violating records.
- **Expected signal for B:** Fact-grain reconciliation passes and the declared business rule explains the changed population.
- **Immediate mitigation:** Hold the affected decision slice or publish its prior certified value; do not use DISTINCT as a cosmetic repair.
- **Correction:** Repair the owner boundary: source contract, relationship model, temporal selector, allocation policy, or query grain.
- **Prevention:** Add a release fixture containing zero, one, many, late, duplicate, and changed-version relationships relevant to this invariant.
- **Verification:** Replay the fixture and require the stated input-fact coverage, matched-fact multiplicity, and fact-grain total before release.
- **Mitigation risk/escalation:** Holding data delays a decision; an automatic fallback can conceal a semantic defect. Escalate when fallback changes a business action.
- **Boundary:** Passing this check establishes consistency with the written model, not truth of an unavailable source body.

#### DIAG-03 — bridge weight drift

- **Trigger/environment:** A bounded release changes a fact, dimension, bridge, or query predicate while the declared fact grain remains the correctness oracle.
- **Symptom:** A join-derived count, amount, slice, or latency differs from the certified comparison at the same cutoff.
- **Competing cause A:** A relationship, key, version, unit, or time invariant was violated for bridge weight drift.
- **Competing cause B:** A legitimate business-policy or population change produced the movement without invalidating the model.
- **Discriminating diagnostic:** Preserve pre-join input fact keys, matched fact keys, joined rows, predicate, cutoff, and source revision; compute coverage and multiplicity separately.
- **Expected signal for A:** An anti-join, overlap, key, unit, or cardinality test isolates the violating records.
- **Expected signal for B:** Fact-grain reconciliation passes and the declared business rule explains the changed population.
- **Immediate mitigation:** Hold the affected decision slice or publish its prior certified value; do not use DISTINCT as a cosmetic repair.
- **Correction:** Repair the owner boundary: source contract, relationship model, temporal selector, allocation policy, or query grain.
- **Prevention:** Add a release fixture containing zero, one, many, late, duplicate, and changed-version relationships relevant to this invariant.
- **Verification:** Replay the fixture and require the stated input-fact coverage, matched-fact multiplicity, and fact-grain total before release.
- **Mitigation risk/escalation:** Holding data delays a decision; an automatic fallback can conceal a semantic defect. Escalate when fallback changes a business action.
- **Boundary:** Passing this check establishes consistency with the written model, not truth of an unavailable source body.

#### DIAG-04 — surrogate reuse

- **Trigger/environment:** A bounded release changes a fact, dimension, bridge, or query predicate while the declared fact grain remains the correctness oracle.
- **Symptom:** A join-derived count, amount, slice, or latency differs from the certified comparison at the same cutoff.
- **Competing cause A:** A relationship, key, version, unit, or time invariant was violated for surrogate reuse.
- **Competing cause B:** A legitimate business-policy or population change produced the movement without invalidating the model.
- **Discriminating diagnostic:** Preserve pre-join input fact keys, matched fact keys, joined rows, predicate, cutoff, and source revision; compute coverage and multiplicity separately.
- **Expected signal for A:** An anti-join, overlap, key, unit, or cardinality test isolates the violating records.
- **Expected signal for B:** Fact-grain reconciliation passes and the declared business rule explains the changed population.
- **Immediate mitigation:** Hold the affected decision slice or publish its prior certified value; do not use DISTINCT as a cosmetic repair.
- **Correction:** Repair the owner boundary: source contract, relationship model, temporal selector, allocation policy, or query grain.
- **Prevention:** Add a release fixture containing zero, one, many, late, duplicate, and changed-version relationships relevant to this invariant.
- **Verification:** Replay the fixture and require the stated input-fact coverage, matched-fact multiplicity, and fact-grain total before release.
- **Mitigation risk/escalation:** Holding data delays a decision; an automatic fallback can conceal a semantic defect. Escalate when fallback changes a business action.
- **Boundary:** Passing this check establishes consistency with the written model, not truth of an unavailable source body.

#### DIAG-05 — late fact history

- **Trigger/environment:** A bounded release changes a fact, dimension, bridge, or query predicate while the declared fact grain remains the correctness oracle.
- **Symptom:** A join-derived count, amount, slice, or latency differs from the certified comparison at the same cutoff.
- **Competing cause A:** A relationship, key, version, unit, or time invariant was violated for late fact history.
- **Competing cause B:** A legitimate business-policy or population change produced the movement without invalidating the model.
- **Discriminating diagnostic:** Preserve pre-join input fact keys, matched fact keys, joined rows, predicate, cutoff, and source revision; compute coverage and multiplicity separately.
- **Expected signal for A:** An anti-join, overlap, key, unit, or cardinality test isolates the violating records.
- **Expected signal for B:** Fact-grain reconciliation passes and the declared business rule explains the changed population.
- **Immediate mitigation:** Hold the affected decision slice or publish its prior certified value; do not use DISTINCT as a cosmetic repair.
- **Correction:** Repair the owner boundary: source contract, relationship model, temporal selector, allocation policy, or query grain.
- **Prevention:** Add a release fixture containing zero, one, many, late, duplicate, and changed-version relationships relevant to this invariant.
- **Verification:** Replay the fixture and require the stated input-fact coverage, matched-fact multiplicity, and fact-grain total before release.
- **Mitigation risk/escalation:** Holding data delays a decision; an automatic fallback can conceal a semantic defect. Escalate when fallback changes a business action.
- **Boundary:** Passing this check establishes consistency with the written model, not truth of an unavailable source body.

#### DIAG-06 — unknown member explosion

- **Trigger/environment:** A bounded release changes a fact, dimension, bridge, or query predicate while the declared fact grain remains the correctness oracle.
- **Symptom:** A join-derived count, amount, slice, or latency differs from the certified comparison at the same cutoff.
- **Competing cause A:** A relationship, key, version, unit, or time invariant was violated for unknown member explosion.
- **Competing cause B:** A legitimate business-policy or population change produced the movement without invalidating the model.
- **Discriminating diagnostic:** Preserve pre-join input fact keys, matched fact keys, joined rows, predicate, cutoff, and source revision; compute coverage and multiplicity separately.
- **Expected signal for A:** An anti-join, overlap, key, unit, or cardinality test isolates the violating records.
- **Expected signal for B:** Fact-grain reconciliation passes and the declared business rule explains the changed population.
- **Immediate mitigation:** Hold the affected decision slice or publish its prior certified value; do not use DISTINCT as a cosmetic repair.
- **Correction:** Repair the owner boundary: source contract, relationship model, temporal selector, allocation policy, or query grain.
- **Prevention:** Add a release fixture containing zero, one, many, late, duplicate, and changed-version relationships relevant to this invariant.
- **Verification:** Replay the fixture and require the stated input-fact coverage, matched-fact multiplicity, and fact-grain total before release.
- **Mitigation risk/escalation:** Holding data delays a decision; an automatic fallback can conceal a semantic defect. Escalate when fallback changes a business action.
- **Boundary:** Passing this check establishes consistency with the written model, not truth of an unavailable source body.

#### DIAG-07 — outer join null masking

- **Trigger/environment:** A bounded release changes a fact, dimension, bridge, or query predicate while the declared fact grain remains the correctness oracle.
- **Symptom:** A join-derived count, amount, slice, or latency differs from the certified comparison at the same cutoff.
- **Competing cause A:** A relationship, key, version, unit, or time invariant was violated for outer join null masking.
- **Competing cause B:** A legitimate business-policy or population change produced the movement without invalidating the model.
- **Discriminating diagnostic:** Preserve pre-join input fact keys, matched fact keys, joined rows, predicate, cutoff, and source revision; compute coverage and multiplicity separately.
- **Expected signal for A:** An anti-join, overlap, key, unit, or cardinality test isolates the violating records.
- **Expected signal for B:** Fact-grain reconciliation passes and the declared business rule explains the changed population.
- **Immediate mitigation:** Hold the affected decision slice or publish its prior certified value; do not use DISTINCT as a cosmetic repair.
- **Correction:** Repair the owner boundary: source contract, relationship model, temporal selector, allocation policy, or query grain.
- **Prevention:** Add a release fixture containing zero, one, many, late, duplicate, and changed-version relationships relevant to this invariant.
- **Verification:** Replay the fixture and require the stated input-fact coverage, matched-fact multiplicity, and fact-grain total before release.
- **Mitigation risk/escalation:** Holding data delays a decision; an automatic fallback can conceal a semantic defect. Escalate when fallback changes a business action.
- **Boundary:** Passing this check establishes consistency with the written model, not truth of an unavailable source body.

#### DIAG-08 — fact key collision

- **Trigger/environment:** A bounded release changes a fact, dimension, bridge, or query predicate while the declared fact grain remains the correctness oracle.
- **Symptom:** A join-derived count, amount, slice, or latency differs from the certified comparison at the same cutoff.
- **Competing cause A:** A relationship, key, version, unit, or time invariant was violated for fact key collision.
- **Competing cause B:** A legitimate business-policy or population change produced the movement without invalidating the model.
- **Discriminating diagnostic:** Preserve pre-join input fact keys, matched fact keys, joined rows, predicate, cutoff, and source revision; compute coverage and multiplicity separately.
- **Expected signal for A:** An anti-join, overlap, key, unit, or cardinality test isolates the violating records.
- **Expected signal for B:** Fact-grain reconciliation passes and the declared business rule explains the changed population.
- **Immediate mitigation:** Hold the affected decision slice or publish its prior certified value; do not use DISTINCT as a cosmetic repair.
- **Correction:** Repair the owner boundary: source contract, relationship model, temporal selector, allocation policy, or query grain.
- **Prevention:** Add a release fixture containing zero, one, many, late, duplicate, and changed-version relationships relevant to this invariant.
- **Verification:** Replay the fixture and require the stated input-fact coverage, matched-fact multiplicity, and fact-grain total before release.
- **Mitigation risk/escalation:** Holding data delays a decision; an automatic fallback can conceal a semantic defect. Escalate when fallback changes a business action.
- **Boundary:** Passing this check establishes consistency with the written model, not truth of an unavailable source body.

#### DIAG-09 — currency grain mismatch

- **Trigger/environment:** A bounded release changes a fact, dimension, bridge, or query predicate while the declared fact grain remains the correctness oracle.
- **Symptom:** A join-derived count, amount, slice, or latency differs from the certified comparison at the same cutoff.
- **Competing cause A:** A relationship, key, version, unit, or time invariant was violated for currency grain mismatch.
- **Competing cause B:** A legitimate business-policy or population change produced the movement without invalidating the model.
- **Discriminating diagnostic:** Preserve pre-join input fact keys, matched fact keys, joined rows, predicate, cutoff, and source revision; compute coverage and multiplicity separately.
- **Expected signal for A:** An anti-join, overlap, key, unit, or cardinality test isolates the violating records.
- **Expected signal for B:** Fact-grain reconciliation passes and the declared business rule explains the changed population.
- **Immediate mitigation:** Hold the affected decision slice or publish its prior certified value; do not use DISTINCT as a cosmetic repair.
- **Correction:** Repair the owner boundary: source contract, relationship model, temporal selector, allocation policy, or query grain.
- **Prevention:** Add a release fixture containing zero, one, many, late, duplicate, and changed-version relationships relevant to this invariant.
- **Verification:** Replay the fixture and require the stated input-fact coverage, matched-fact multiplicity, and fact-grain total before release.
- **Mitigation risk/escalation:** Holding data delays a decision; an automatic fallback can conceal a semantic defect. Escalate when fallback changes a business action.
- **Boundary:** Passing this check establishes consistency with the written model, not truth of an unavailable source body.

#### DIAG-10 — snapshot fan-out

- **Trigger/environment:** A bounded release changes a fact, dimension, bridge, or query predicate while the declared fact grain remains the correctness oracle.
- **Symptom:** A join-derived count, amount, slice, or latency differs from the certified comparison at the same cutoff.
- **Competing cause A:** A relationship, key, version, unit, or time invariant was violated for snapshot fan-out.
- **Competing cause B:** A legitimate business-policy or population change produced the movement without invalidating the model.
- **Discriminating diagnostic:** Preserve pre-join input fact keys, matched fact keys, joined rows, predicate, cutoff, and source revision; compute coverage and multiplicity separately.
- **Expected signal for A:** An anti-join, overlap, key, unit, or cardinality test isolates the violating records.
- **Expected signal for B:** Fact-grain reconciliation passes and the declared business rule explains the changed population.
- **Immediate mitigation:** Hold the affected decision slice or publish its prior certified value; do not use DISTINCT as a cosmetic repair.
- **Correction:** Repair the owner boundary: source contract, relationship model, temporal selector, allocation policy, or query grain.
- **Prevention:** Add a release fixture containing zero, one, many, late, duplicate, and changed-version relationships relevant to this invariant.
- **Verification:** Replay the fixture and require the stated input-fact coverage, matched-fact multiplicity, and fact-grain total before release.
- **Mitigation risk/escalation:** Holding data delays a decision; an automatic fallback can conceal a semantic defect. Escalate when fallback changes a business action.
- **Boundary:** Passing this check establishes consistency with the written model, not truth of an unavailable source body.

#### DIAG-11 — security filtered loss

- **Trigger/environment:** A bounded release changes a fact, dimension, bridge, or query predicate while the declared fact grain remains the correctness oracle.
- **Symptom:** A join-derived count, amount, slice, or latency differs from the certified comparison at the same cutoff.
- **Competing cause A:** A relationship, key, version, unit, or time invariant was violated for security filtered loss.
- **Competing cause B:** A legitimate business-policy or population change produced the movement without invalidating the model.
- **Discriminating diagnostic:** Preserve pre-join input fact keys, matched fact keys, joined rows, predicate, cutoff, and source revision; compute coverage and multiplicity separately.
- **Expected signal for A:** An anti-join, overlap, key, unit, or cardinality test isolates the violating records.
- **Expected signal for B:** Fact-grain reconciliation passes and the declared business rule explains the changed population.
- **Immediate mitigation:** Hold the affected decision slice or publish its prior certified value; do not use DISTINCT as a cosmetic repair.
- **Correction:** Repair the owner boundary: source contract, relationship model, temporal selector, allocation policy, or query grain.
- **Prevention:** Add a release fixture containing zero, one, many, late, duplicate, and changed-version relationships relevant to this invariant.
- **Verification:** Replay the fixture and require the stated input-fact coverage, matched-fact multiplicity, and fact-grain total before release.
- **Mitigation risk/escalation:** Holding data delays a decision; an automatic fallback can conceal a semantic defect. Escalate when fallback changes a business action.
- **Boundary:** Passing this check establishes consistency with the written model, not truth of an unavailable source body.

#### DIAG-12 — soft delete ambiguity

- **Trigger/environment:** A bounded release changes a fact, dimension, bridge, or query predicate while the declared fact grain remains the correctness oracle.
- **Symptom:** A join-derived count, amount, slice, or latency differs from the certified comparison at the same cutoff.
- **Competing cause A:** A relationship, key, version, unit, or time invariant was violated for soft delete ambiguity.
- **Competing cause B:** A legitimate business-policy or population change produced the movement without invalidating the model.
- **Discriminating diagnostic:** Preserve pre-join input fact keys, matched fact keys, joined rows, predicate, cutoff, and source revision; compute coverage and multiplicity separately.
- **Expected signal for A:** An anti-join, overlap, key, unit, or cardinality test isolates the violating records.
- **Expected signal for B:** Fact-grain reconciliation passes and the declared business rule explains the changed population.
- **Immediate mitigation:** Hold the affected decision slice or publish its prior certified value; do not use DISTINCT as a cosmetic repair.
- **Correction:** Repair the owner boundary: source contract, relationship model, temporal selector, allocation policy, or query grain.
- **Prevention:** Add a release fixture containing zero, one, many, late, duplicate, and changed-version relationships relevant to this invariant.
- **Verification:** Replay the fixture and require the stated input-fact coverage, matched-fact multiplicity, and fact-grain total before release.
- **Mitigation risk/escalation:** Holding data delays a decision; an automatic fallback can conceal a semantic defect. Escalate when fallback changes a business action.
- **Boundary:** Passing this check establishes consistency with the written model, not truth of an unavailable source body.

#### DIAG-13 — tenant key overlap

- **Trigger/environment:** A bounded release changes a fact, dimension, bridge, or query predicate while the declared fact grain remains the correctness oracle.
- **Symptom:** A join-derived count, amount, slice, or latency differs from the certified comparison at the same cutoff.
- **Competing cause A:** A relationship, key, version, unit, or time invariant was violated for tenant key overlap.
- **Competing cause B:** A legitimate business-policy or population change produced the movement without invalidating the model.
- **Discriminating diagnostic:** Preserve pre-join input fact keys, matched fact keys, joined rows, predicate, cutoff, and source revision; compute coverage and multiplicity separately.
- **Expected signal for A:** An anti-join, overlap, key, unit, or cardinality test isolates the violating records.
- **Expected signal for B:** Fact-grain reconciliation passes and the declared business rule explains the changed population.
- **Immediate mitigation:** Hold the affected decision slice or publish its prior certified value; do not use DISTINCT as a cosmetic repair.
- **Correction:** Repair the owner boundary: source contract, relationship model, temporal selector, allocation policy, or query grain.
- **Prevention:** Add a release fixture containing zero, one, many, late, duplicate, and changed-version relationships relevant to this invariant.
- **Verification:** Replay the fixture and require the stated input-fact coverage, matched-fact multiplicity, and fact-grain total before release.
- **Mitigation risk/escalation:** Holding data delays a decision; an automatic fallback can conceal a semantic defect. Escalate when fallback changes a business action.
- **Boundary:** Passing this check establishes consistency with the written model, not truth of an unavailable source body.

#### DIAG-14 — rounding residue

- **Trigger/environment:** A bounded release changes a fact, dimension, bridge, or query predicate while the declared fact grain remains the correctness oracle.
- **Symptom:** A join-derived count, amount, slice, or latency differs from the certified comparison at the same cutoff.
- **Competing cause A:** A relationship, key, version, unit, or time invariant was violated for rounding residue.
- **Competing cause B:** A legitimate business-policy or population change produced the movement without invalidating the model.
- **Discriminating diagnostic:** Preserve pre-join input fact keys, matched fact keys, joined rows, predicate, cutoff, and source revision; compute coverage and multiplicity separately.
- **Expected signal for A:** An anti-join, overlap, key, unit, or cardinality test isolates the violating records.
- **Expected signal for B:** Fact-grain reconciliation passes and the declared business rule explains the changed population.
- **Immediate mitigation:** Hold the affected decision slice or publish its prior certified value; do not use DISTINCT as a cosmetic repair.
- **Correction:** Repair the owner boundary: source contract, relationship model, temporal selector, allocation policy, or query grain.
- **Prevention:** Add a release fixture containing zero, one, many, late, duplicate, and changed-version relationships relevant to this invariant.
- **Verification:** Replay the fixture and require the stated input-fact coverage, matched-fact multiplicity, and fact-grain total before release.
- **Mitigation risk/escalation:** Holding data delays a decision; an automatic fallback can conceal a semantic defect. Escalate when fallback changes a business action.
- **Boundary:** Passing this check establishes consistency with the written model, not truth of an unavailable source body.

#### DIAG-15 — temporal bridge overlap

- **Trigger/environment:** A bounded release changes a fact, dimension, bridge, or query predicate while the declared fact grain remains the correctness oracle.
- **Symptom:** A join-derived count, amount, slice, or latency differs from the certified comparison at the same cutoff.
- **Competing cause A:** A relationship, key, version, unit, or time invariant was violated for temporal bridge overlap.
- **Competing cause B:** A legitimate business-policy or population change produced the movement without invalidating the model.
- **Discriminating diagnostic:** Preserve pre-join input fact keys, matched fact keys, joined rows, predicate, cutoff, and source revision; compute coverage and multiplicity separately.
- **Expected signal for A:** An anti-join, overlap, key, unit, or cardinality test isolates the violating records.
- **Expected signal for B:** Fact-grain reconciliation passes and the declared business rule explains the changed population.
- **Immediate mitigation:** Hold the affected decision slice or publish its prior certified value; do not use DISTINCT as a cosmetic repair.
- **Correction:** Repair the owner boundary: source contract, relationship model, temporal selector, allocation policy, or query grain.
- **Prevention:** Add a release fixture containing zero, one, many, late, duplicate, and changed-version relationships relevant to this invariant.
- **Verification:** Replay the fixture and require the stated input-fact coverage, matched-fact multiplicity, and fact-grain total before release.
- **Mitigation risk/escalation:** Holding data delays a decision; an automatic fallback can conceal a semantic defect. Escalate when fallback changes a business action.
- **Boundary:** Passing this check establishes consistency with the written model, not truth of an unavailable source body.

#### DIAG-16 — semantic duplicate

- **Trigger/environment:** A bounded release changes a fact, dimension, bridge, or query predicate while the declared fact grain remains the correctness oracle.
- **Symptom:** A join-derived count, amount, slice, or latency differs from the certified comparison at the same cutoff.
- **Competing cause A:** A relationship, key, version, unit, or time invariant was violated for semantic duplicate.
- **Competing cause B:** A legitimate business-policy or population change produced the movement without invalidating the model.
- **Discriminating diagnostic:** Preserve pre-join input fact keys, matched fact keys, joined rows, predicate, cutoff, and source revision; compute coverage and multiplicity separately.
- **Expected signal for A:** An anti-join, overlap, key, unit, or cardinality test isolates the violating records.
- **Expected signal for B:** Fact-grain reconciliation passes and the declared business rule explains the changed population.
- **Immediate mitigation:** Hold the affected decision slice or publish its prior certified value; do not use DISTINCT as a cosmetic repair.
- **Correction:** Repair the owner boundary: source contract, relationship model, temporal selector, allocation policy, or query grain.
- **Prevention:** Add a release fixture containing zero, one, many, late, duplicate, and changed-version relationships relevant to this invariant.
- **Verification:** Replay the fixture and require the stated input-fact coverage, matched-fact multiplicity, and fact-grain total before release.
- **Mitigation risk/escalation:** Holding data delays a decision; an automatic fallback can conceal a semantic defect. Escalate when fallback changes a business action.
- **Boundary:** Passing this check establishes consistency with the written model, not truth of an unavailable source body.

#### DIAG-17 — predicate pushdown loss

- **Trigger/environment:** A bounded release changes a fact, dimension, bridge, or query predicate while the declared fact grain remains the correctness oracle.
- **Symptom:** A join-derived count, amount, slice, or latency differs from the certified comparison at the same cutoff.
- **Competing cause A:** A relationship, key, version, unit, or time invariant was violated for predicate pushdown loss.
- **Competing cause B:** A legitimate business-policy or population change produced the movement without invalidating the model.
- **Discriminating diagnostic:** Preserve pre-join input fact keys, matched fact keys, joined rows, predicate, cutoff, and source revision; compute coverage and multiplicity separately.
- **Expected signal for A:** An anti-join, overlap, key, unit, or cardinality test isolates the violating records.
- **Expected signal for B:** Fact-grain reconciliation passes and the declared business rule explains the changed population.
- **Immediate mitigation:** Hold the affected decision slice or publish its prior certified value; do not use DISTINCT as a cosmetic repair.
- **Correction:** Repair the owner boundary: source contract, relationship model, temporal selector, allocation policy, or query grain.
- **Prevention:** Add a release fixture containing zero, one, many, late, duplicate, and changed-version relationships relevant to this invariant.
- **Verification:** Replay the fixture and require the stated input-fact coverage, matched-fact multiplicity, and fact-grain total before release.
- **Mitigation risk/escalation:** Holding data delays a decision; an automatic fallback can conceal a semantic defect. Escalate when fallback changes a business action.
- **Boundary:** Passing this check establishes consistency with the written model, not truth of an unavailable source body.

#### DIAG-18 — aggregate before join

- **Trigger/environment:** A bounded release changes a fact, dimension, bridge, or query predicate while the declared fact grain remains the correctness oracle.
- **Symptom:** A join-derived count, amount, slice, or latency differs from the certified comparison at the same cutoff.
- **Competing cause A:** A relationship, key, version, unit, or time invariant was violated for aggregate before join.
- **Competing cause B:** A legitimate business-policy or population change produced the movement without invalidating the model.
- **Discriminating diagnostic:** Preserve pre-join input fact keys, matched fact keys, joined rows, predicate, cutoff, and source revision; compute coverage and multiplicity separately.
- **Expected signal for A:** An anti-join, overlap, key, unit, or cardinality test isolates the violating records.
- **Expected signal for B:** Fact-grain reconciliation passes and the declared business rule explains the changed population.
- **Immediate mitigation:** Hold the affected decision slice or publish its prior certified value; do not use DISTINCT as a cosmetic repair.
- **Correction:** Repair the owner boundary: source contract, relationship model, temporal selector, allocation policy, or query grain.
- **Prevention:** Add a release fixture containing zero, one, many, late, duplicate, and changed-version relationships relevant to this invariant.
- **Verification:** Replay the fixture and require the stated input-fact coverage, matched-fact multiplicity, and fact-grain total before release.
- **Mitigation risk/escalation:** Holding data delays a decision; an automatic fallback can conceal a semantic defect. Escalate when fallback changes a business action.
- **Boundary:** Passing this check establishes consistency with the written model, not truth of an unavailable source body.

#### DIAG-19 — time-zone split

- **Trigger/environment:** A bounded release changes a fact, dimension, bridge, or query predicate while the declared fact grain remains the correctness oracle.
- **Symptom:** A join-derived count, amount, slice, or latency differs from the certified comparison at the same cutoff.
- **Competing cause A:** A relationship, key, version, unit, or time invariant was violated for time-zone split.
- **Competing cause B:** A legitimate business-policy or population change produced the movement without invalidating the model.
- **Discriminating diagnostic:** Preserve pre-join input fact keys, matched fact keys, joined rows, predicate, cutoff, and source revision; compute coverage and multiplicity separately.
- **Expected signal for A:** An anti-join, overlap, key, unit, or cardinality test isolates the violating records.
- **Expected signal for B:** Fact-grain reconciliation passes and the declared business rule explains the changed population.
- **Immediate mitigation:** Hold the affected decision slice or publish its prior certified value; do not use DISTINCT as a cosmetic repair.
- **Correction:** Repair the owner boundary: source contract, relationship model, temporal selector, allocation policy, or query grain.
- **Prevention:** Add a release fixture containing zero, one, many, late, duplicate, and changed-version relationships relevant to this invariant.
- **Verification:** Replay the fixture and require the stated input-fact coverage, matched-fact multiplicity, and fact-grain total before release.
- **Mitigation risk/escalation:** Holding data delays a decision; an automatic fallback can conceal a semantic defect. Escalate when fallback changes a business action.
- **Boundary:** Passing this check establishes consistency with the written model, not truth of an unavailable source body.

#### DIAG-20 — hierarchy cycle

- **Trigger/environment:** A bounded release changes a fact, dimension, bridge, or query predicate while the declared fact grain remains the correctness oracle.
- **Symptom:** A join-derived count, amount, slice, or latency differs from the certified comparison at the same cutoff.
- **Competing cause A:** A relationship, key, version, unit, or time invariant was violated for hierarchy cycle.
- **Competing cause B:** A legitimate business-policy or population change produced the movement without invalidating the model.
- **Discriminating diagnostic:** Preserve pre-join input fact keys, matched fact keys, joined rows, predicate, cutoff, and source revision; compute coverage and multiplicity separately.
- **Expected signal for A:** An anti-join, overlap, key, unit, or cardinality test isolates the violating records.
- **Expected signal for B:** Fact-grain reconciliation passes and the declared business rule explains the changed population.
- **Immediate mitigation:** Hold the affected decision slice or publish its prior certified value; do not use DISTINCT as a cosmetic repair.
- **Correction:** Repair the owner boundary: source contract, relationship model, temporal selector, allocation policy, or query grain.
- **Prevention:** Add a release fixture containing zero, one, many, late, duplicate, and changed-version relationships relevant to this invariant.
- **Verification:** Replay the fixture and require the stated input-fact coverage, matched-fact multiplicity, and fact-grain total before release.
- **Mitigation risk/escalation:** Holding data delays a decision; an automatic fallback can conceal a semantic defect. Escalate when fallback changes a business action.
- **Boundary:** Passing this check establishes consistency with the written model, not truth of an unavailable source body.

#### DIAG-21 — cardinality spill

- **Trigger/environment:** A bounded release changes a fact, dimension, bridge, or query predicate while the declared fact grain remains the correctness oracle.
- **Symptom:** A join-derived count, amount, slice, or latency differs from the certified comparison at the same cutoff.
- **Competing cause A:** A relationship, key, version, unit, or time invariant was violated for cardinality spill.
- **Competing cause B:** A legitimate business-policy or population change produced the movement without invalidating the model.
- **Discriminating diagnostic:** Preserve pre-join input fact keys, matched fact keys, joined rows, predicate, cutoff, and source revision; compute coverage and multiplicity separately.
- **Expected signal for A:** An anti-join, overlap, key, unit, or cardinality test isolates the violating records.
- **Expected signal for B:** Fact-grain reconciliation passes and the declared business rule explains the changed population.
- **Immediate mitigation:** Hold the affected decision slice or publish its prior certified value; do not use DISTINCT as a cosmetic repair.
- **Correction:** Repair the owner boundary: source contract, relationship model, temporal selector, allocation policy, or query grain.
- **Prevention:** Add a release fixture containing zero, one, many, late, duplicate, and changed-version relationships relevant to this invariant.
- **Verification:** Replay the fixture and require the stated input-fact coverage, matched-fact multiplicity, and fact-grain total before release.
- **Mitigation risk/escalation:** Holding data delays a decision; an automatic fallback can conceal a semantic defect. Escalate when fallback changes a business action.
- **Boundary:** Passing this check establishes consistency with the written model, not truth of an unavailable source body.

#### DIAG-22 — hot-key skew

- **Trigger/environment:** A bounded release changes a fact, dimension, bridge, or query predicate while the declared fact grain remains the correctness oracle.
- **Symptom:** A join-derived count, amount, slice, or latency differs from the certified comparison at the same cutoff.
- **Competing cause A:** A relationship, key, version, unit, or time invariant was violated for hot-key skew.
- **Competing cause B:** A legitimate business-policy or population change produced the movement without invalidating the model.
- **Discriminating diagnostic:** Preserve pre-join input fact keys, matched fact keys, joined rows, predicate, cutoff, and source revision; compute coverage and multiplicity separately.
- **Expected signal for A:** An anti-join, overlap, key, unit, or cardinality test isolates the violating records.
- **Expected signal for B:** Fact-grain reconciliation passes and the declared business rule explains the changed population.
- **Immediate mitigation:** Hold the affected decision slice or publish its prior certified value; do not use DISTINCT as a cosmetic repair.
- **Correction:** Repair the owner boundary: source contract, relationship model, temporal selector, allocation policy, or query grain.
- **Prevention:** Add a release fixture containing zero, one, many, late, duplicate, and changed-version relationships relevant to this invariant.
- **Verification:** Replay the fixture and require the stated input-fact coverage, matched-fact multiplicity, and fact-grain total before release.
- **Mitigation risk/escalation:** Holding data delays a decision; an automatic fallback can conceal a semantic defect. Escalate when fallback changes a business action.
- **Boundary:** Passing this check establishes consistency with the written model, not truth of an unavailable source body.

#### DIAG-23 — referential lag

- **Trigger/environment:** A bounded release changes a fact, dimension, bridge, or query predicate while the declared fact grain remains the correctness oracle.
- **Symptom:** A join-derived count, amount, slice, or latency differs from the certified comparison at the same cutoff.
- **Competing cause A:** A relationship, key, version, unit, or time invariant was violated for referential lag.
- **Competing cause B:** A legitimate business-policy or population change produced the movement without invalidating the model.
- **Discriminating diagnostic:** Preserve pre-join input fact keys, matched fact keys, joined rows, predicate, cutoff, and source revision; compute coverage and multiplicity separately.
- **Expected signal for A:** An anti-join, overlap, key, unit, or cardinality test isolates the violating records.
- **Expected signal for B:** Fact-grain reconciliation passes and the declared business rule explains the changed population.
- **Immediate mitigation:** Hold the affected decision slice or publish its prior certified value; do not use DISTINCT as a cosmetic repair.
- **Correction:** Repair the owner boundary: source contract, relationship model, temporal selector, allocation policy, or query grain.
- **Prevention:** Add a release fixture containing zero, one, many, late, duplicate, and changed-version relationships relevant to this invariant.
- **Verification:** Replay the fixture and require the stated input-fact coverage, matched-fact multiplicity, and fact-grain total before release.
- **Mitigation risk/escalation:** Holding data delays a decision; an automatic fallback can conceal a semantic defect. Escalate when fallback changes a business action.
- **Boundary:** Passing this check establishes consistency with the written model, not truth of an unavailable source body.

#### DIAG-24 — null equality

- **Trigger/environment:** A bounded release changes a fact, dimension, bridge, or query predicate while the declared fact grain remains the correctness oracle.
- **Symptom:** A join-derived count, amount, slice, or latency differs from the certified comparison at the same cutoff.
- **Competing cause A:** A relationship, key, version, unit, or time invariant was violated for null equality.
- **Competing cause B:** A legitimate business-policy or population change produced the movement without invalidating the model.
- **Discriminating diagnostic:** Preserve pre-join input fact keys, matched fact keys, joined rows, predicate, cutoff, and source revision; compute coverage and multiplicity separately.
- **Expected signal for A:** An anti-join, overlap, key, unit, or cardinality test isolates the violating records.
- **Expected signal for B:** Fact-grain reconciliation passes and the declared business rule explains the changed population.
- **Immediate mitigation:** Hold the affected decision slice or publish its prior certified value; do not use DISTINCT as a cosmetic repair.
- **Correction:** Repair the owner boundary: source contract, relationship model, temporal selector, allocation policy, or query grain.
- **Prevention:** Add a release fixture containing zero, one, many, late, duplicate, and changed-version relationships relevant to this invariant.
- **Verification:** Replay the fixture and require the stated input-fact coverage, matched-fact multiplicity, and fact-grain total before release.
- **Mitigation risk/escalation:** Holding data delays a decision; an automatic fallback can conceal a semantic defect. Escalate when fallback changes a business action.
- **Boundary:** Passing this check establishes consistency with the written model, not truth of an unavailable source body.

#### DIAG-25 — attribute drift

- **Trigger/environment:** A bounded release changes a fact, dimension, bridge, or query predicate while the declared fact grain remains the correctness oracle.
- **Symptom:** A join-derived count, amount, slice, or latency differs from the certified comparison at the same cutoff.
- **Competing cause A:** A relationship, key, version, unit, or time invariant was violated for attribute drift.
- **Competing cause B:** A legitimate business-policy or population change produced the movement without invalidating the model.
- **Discriminating diagnostic:** Preserve pre-join input fact keys, matched fact keys, joined rows, predicate, cutoff, and source revision; compute coverage and multiplicity separately.
- **Expected signal for A:** An anti-join, overlap, key, unit, or cardinality test isolates the violating records.
- **Expected signal for B:** Fact-grain reconciliation passes and the declared business rule explains the changed population.
- **Immediate mitigation:** Hold the affected decision slice or publish its prior certified value; do not use DISTINCT as a cosmetic repair.
- **Correction:** Repair the owner boundary: source contract, relationship model, temporal selector, allocation policy, or query grain.
- **Prevention:** Add a release fixture containing zero, one, many, late, duplicate, and changed-version relationships relevant to this invariant.
- **Verification:** Replay the fixture and require the stated input-fact coverage, matched-fact multiplicity, and fact-grain total before release.
- **Mitigation risk/escalation:** Holding data delays a decision; an automatic fallback can conceal a semantic defect. Escalate when fallback changes a business action.
- **Boundary:** Passing this check establishes consistency with the written model, not truth of an unavailable source body.

#### DIAG-26 — metric double count

- **Trigger/environment:** A bounded release changes a fact, dimension, bridge, or query predicate while the declared fact grain remains the correctness oracle.
- **Symptom:** A join-derived count, amount, slice, or latency differs from the certified comparison at the same cutoff.
- **Competing cause A:** A relationship, key, version, unit, or time invariant was violated for metric double count.
- **Competing cause B:** A legitimate business-policy or population change produced the movement without invalidating the model.
- **Discriminating diagnostic:** Preserve pre-join input fact keys, matched fact keys, joined rows, predicate, cutoff, and source revision; compute coverage and multiplicity separately.
- **Expected signal for A:** An anti-join, overlap, key, unit, or cardinality test isolates the violating records.
- **Expected signal for B:** Fact-grain reconciliation passes and the declared business rule explains the changed population.
- **Immediate mitigation:** Hold the affected decision slice or publish its prior certified value; do not use DISTINCT as a cosmetic repair.
- **Correction:** Repair the owner boundary: source contract, relationship model, temporal selector, allocation policy, or query grain.
- **Prevention:** Add a release fixture containing zero, one, many, late, duplicate, and changed-version relationships relevant to this invariant.
- **Verification:** Replay the fixture and require the stated input-fact coverage, matched-fact multiplicity, and fact-grain total before release.
- **Mitigation risk/escalation:** Holding data delays a decision; an automatic fallback can conceal a semantic defect. Escalate when fallback changes a business action.
- **Boundary:** Passing this check establishes consistency with the written model, not truth of an unavailable source body.

#### DIAG-27 — fixture blind spot

- **Trigger/environment:** A bounded release changes a fact, dimension, bridge, or query predicate while the declared fact grain remains the correctness oracle.
- **Symptom:** A join-derived count, amount, slice, or latency differs from the certified comparison at the same cutoff.
- **Competing cause A:** A relationship, key, version, unit, or time invariant was violated for fixture blind spot.
- **Competing cause B:** A legitimate business-policy or population change produced the movement without invalidating the model.
- **Discriminating diagnostic:** Preserve pre-join input fact keys, matched fact keys, joined rows, predicate, cutoff, and source revision; compute coverage and multiplicity separately.
- **Expected signal for A:** An anti-join, overlap, key, unit, or cardinality test isolates the violating records.
- **Expected signal for B:** Fact-grain reconciliation passes and the declared business rule explains the changed population.
- **Immediate mitigation:** Hold the affected decision slice or publish its prior certified value; do not use DISTINCT as a cosmetic repair.
- **Correction:** Repair the owner boundary: source contract, relationship model, temporal selector, allocation policy, or query grain.
- **Prevention:** Add a release fixture containing zero, one, many, late, duplicate, and changed-version relationships relevant to this invariant.
- **Verification:** Replay the fixture and require the stated input-fact coverage, matched-fact multiplicity, and fact-grain total before release.
- **Mitigation risk/escalation:** Holding data delays a decision; an automatic fallback can conceal a semantic defect. Escalate when fallback changes a business action.
- **Boundary:** Passing this check establishes consistency with the written model, not truth of an unavailable source body.

#### DIAG-28 — compensating error

- **Trigger/environment:** A bounded release changes a fact, dimension, bridge, or query predicate while the declared fact grain remains the correctness oracle.
- **Symptom:** A join-derived count, amount, slice, or latency differs from the certified comparison at the same cutoff.
- **Competing cause A:** A relationship, key, version, unit, or time invariant was violated for compensating error.
- **Competing cause B:** A legitimate business-policy or population change produced the movement without invalidating the model.
- **Discriminating diagnostic:** Preserve pre-join input fact keys, matched fact keys, joined rows, predicate, cutoff, and source revision; compute coverage and multiplicity separately.
- **Expected signal for A:** An anti-join, overlap, key, unit, or cardinality test isolates the violating records.
- **Expected signal for B:** Fact-grain reconciliation passes and the declared business rule explains the changed population.
- **Immediate mitigation:** Hold the affected decision slice or publish its prior certified value; do not use DISTINCT as a cosmetic repair.
- **Correction:** Repair the owner boundary: source contract, relationship model, temporal selector, allocation policy, or query grain.
- **Prevention:** Add a release fixture containing zero, one, many, late, duplicate, and changed-version relationships relevant to this invariant.
- **Verification:** Replay the fixture and require the stated input-fact coverage, matched-fact multiplicity, and fact-grain total before release.
- **Mitigation risk/escalation:** Holding data delays a decision; an automatic fallback can conceal a semantic defect. Escalate when fallback changes a business action.
- **Boundary:** Passing this check establishes consistency with the written model, not truth of an unavailable source body.

#### DIAG-29 — schema semantic drift

- **Trigger/environment:** A bounded release changes a fact, dimension, bridge, or query predicate while the declared fact grain remains the correctness oracle.
- **Symptom:** A join-derived count, amount, slice, or latency differs from the certified comparison at the same cutoff.
- **Competing cause A:** A relationship, key, version, unit, or time invariant was violated for schema semantic drift.
- **Competing cause B:** A legitimate business-policy or population change produced the movement without invalidating the model.
- **Discriminating diagnostic:** Preserve pre-join input fact keys, matched fact keys, joined rows, predicate, cutoff, and source revision; compute coverage and multiplicity separately.
- **Expected signal for A:** An anti-join, overlap, key, unit, or cardinality test isolates the violating records.
- **Expected signal for B:** Fact-grain reconciliation passes and the declared business rule explains the changed population.
- **Immediate mitigation:** Hold the affected decision slice or publish its prior certified value; do not use DISTINCT as a cosmetic repair.
- **Correction:** Repair the owner boundary: source contract, relationship model, temporal selector, allocation policy, or query grain.
- **Prevention:** Add a release fixture containing zero, one, many, late, duplicate, and changed-version relationships relevant to this invariant.
- **Verification:** Replay the fixture and require the stated input-fact coverage, matched-fact multiplicity, and fact-grain total before release.
- **Mitigation risk/escalation:** Holding data delays a decision; an automatic fallback can conceal a semantic defect. Escalate when fallback changes a business action.
- **Boundary:** Passing this check establishes consistency with the written model, not truth of an unavailable source body.

#### DIAG-30 — orphaned parent

- **Trigger/environment:** A bounded release changes a fact, dimension, bridge, or query predicate while the declared fact grain remains the correctness oracle.
- **Symptom:** A join-derived count, amount, slice, or latency differs from the certified comparison at the same cutoff.
- **Competing cause A:** A relationship, key, version, unit, or time invariant was violated for orphaned parent.
- **Competing cause B:** A legitimate business-policy or population change produced the movement without invalidating the model.
- **Discriminating diagnostic:** Preserve pre-join input fact keys, matched fact keys, joined rows, predicate, cutoff, and source revision; compute coverage and multiplicity separately.
- **Expected signal for A:** An anti-join, overlap, key, unit, or cardinality test isolates the violating records.
- **Expected signal for B:** Fact-grain reconciliation passes and the declared business rule explains the changed population.
- **Immediate mitigation:** Hold the affected decision slice or publish its prior certified value; do not use DISTINCT as a cosmetic repair.
- **Correction:** Repair the owner boundary: source contract, relationship model, temporal selector, allocation policy, or query grain.
- **Prevention:** Add a release fixture containing zero, one, many, late, duplicate, and changed-version relationships relevant to this invariant.
- **Verification:** Replay the fixture and require the stated input-fact coverage, matched-fact multiplicity, and fact-grain total before release.
- **Mitigation risk/escalation:** Holding data delays a decision; an automatic fallback can conceal a semantic defect. Escalate when fallback changes a business action.
- **Boundary:** Passing this check establishes consistency with the written model, not truth of an unavailable source body.

#### DIAG-31 — version tie

- **Trigger/environment:** A bounded release changes a fact, dimension, bridge, or query predicate while the declared fact grain remains the correctness oracle.
- **Symptom:** A join-derived count, amount, slice, or latency differs from the certified comparison at the same cutoff.
- **Competing cause A:** A relationship, key, version, unit, or time invariant was violated for version tie.
- **Competing cause B:** A legitimate business-policy or population change produced the movement without invalidating the model.
- **Discriminating diagnostic:** Preserve pre-join input fact keys, matched fact keys, joined rows, predicate, cutoff, and source revision; compute coverage and multiplicity separately.
- **Expected signal for A:** An anti-join, overlap, key, unit, or cardinality test isolates the violating records.
- **Expected signal for B:** Fact-grain reconciliation passes and the declared business rule explains the changed population.
- **Immediate mitigation:** Hold the affected decision slice or publish its prior certified value; do not use DISTINCT as a cosmetic repair.
- **Correction:** Repair the owner boundary: source contract, relationship model, temporal selector, allocation policy, or query grain.
- **Prevention:** Add a release fixture containing zero, one, many, late, duplicate, and changed-version relationships relevant to this invariant.
- **Verification:** Replay the fixture and require the stated input-fact coverage, matched-fact multiplicity, and fact-grain total before release.
- **Mitigation risk/escalation:** Holding data delays a decision; an automatic fallback can conceal a semantic defect. Escalate when fallback changes a business action.
- **Boundary:** Passing this check establishes consistency with the written model, not truth of an unavailable source body.

#### DIAG-32 — effective date gap

- **Trigger/environment:** A bounded release changes a fact, dimension, bridge, or query predicate while the declared fact grain remains the correctness oracle.
- **Symptom:** A join-derived count, amount, slice, or latency differs from the certified comparison at the same cutoff.
- **Competing cause A:** A relationship, key, version, unit, or time invariant was violated for effective date gap.
- **Competing cause B:** A legitimate business-policy or population change produced the movement without invalidating the model.
- **Discriminating diagnostic:** Preserve pre-join input fact keys, matched fact keys, joined rows, predicate, cutoff, and source revision; compute coverage and multiplicity separately.
- **Expected signal for A:** An anti-join, overlap, key, unit, or cardinality test isolates the violating records.
- **Expected signal for B:** Fact-grain reconciliation passes and the declared business rule explains the changed population.
- **Immediate mitigation:** Hold the affected decision slice or publish its prior certified value; do not use DISTINCT as a cosmetic repair.
- **Correction:** Repair the owner boundary: source contract, relationship model, temporal selector, allocation policy, or query grain.
- **Prevention:** Add a release fixture containing zero, one, many, late, duplicate, and changed-version relationships relevant to this invariant.
- **Verification:** Replay the fixture and require the stated input-fact coverage, matched-fact multiplicity, and fact-grain total before release.
- **Mitigation risk/escalation:** Holding data delays a decision; an automatic fallback can conceal a semantic defect. Escalate when fallback changes a business action.
- **Boundary:** Passing this check establishes consistency with the written model, not truth of an unavailable source body.

#### DIAG-33 — duplicate source namespace

- **Trigger/environment:** A bounded release changes a fact, dimension, bridge, or query predicate while the declared fact grain remains the correctness oracle.
- **Symptom:** A join-derived count, amount, slice, or latency differs from the certified comparison at the same cutoff.
- **Competing cause A:** A relationship, key, version, unit, or time invariant was violated for duplicate source namespace.
- **Competing cause B:** A legitimate business-policy or population change produced the movement without invalidating the model.
- **Discriminating diagnostic:** Preserve pre-join input fact keys, matched fact keys, joined rows, predicate, cutoff, and source revision; compute coverage and multiplicity separately.
- **Expected signal for A:** An anti-join, overlap, key, unit, or cardinality test isolates the violating records.
- **Expected signal for B:** Fact-grain reconciliation passes and the declared business rule explains the changed population.
- **Immediate mitigation:** Hold the affected decision slice or publish its prior certified value; do not use DISTINCT as a cosmetic repair.
- **Correction:** Repair the owner boundary: source contract, relationship model, temporal selector, allocation policy, or query grain.
- **Prevention:** Add a release fixture containing zero, one, many, late, duplicate, and changed-version relationships relevant to this invariant.
- **Verification:** Replay the fixture and require the stated input-fact coverage, matched-fact multiplicity, and fact-grain total before release.
- **Mitigation risk/escalation:** Holding data delays a decision; an automatic fallback can conceal a semantic defect. Escalate when fallback changes a business action.
- **Boundary:** Passing this check establishes consistency with the written model, not truth of an unavailable source body.

#### DIAG-34 — allocation negative weight

- **Trigger/environment:** A bounded release changes a fact, dimension, bridge, or query predicate while the declared fact grain remains the correctness oracle.
- **Symptom:** A join-derived count, amount, slice, or latency differs from the certified comparison at the same cutoff.
- **Competing cause A:** A relationship, key, version, unit, or time invariant was violated for allocation negative weight.
- **Competing cause B:** A legitimate business-policy or population change produced the movement without invalidating the model.
- **Discriminating diagnostic:** Preserve pre-join input fact keys, matched fact keys, joined rows, predicate, cutoff, and source revision; compute coverage and multiplicity separately.
- **Expected signal for A:** An anti-join, overlap, key, unit, or cardinality test isolates the violating records.
- **Expected signal for B:** Fact-grain reconciliation passes and the declared business rule explains the changed population.
- **Immediate mitigation:** Hold the affected decision slice or publish its prior certified value; do not use DISTINCT as a cosmetic repair.
- **Correction:** Repair the owner boundary: source contract, relationship model, temporal selector, allocation policy, or query grain.
- **Prevention:** Add a release fixture containing zero, one, many, late, duplicate, and changed-version relationships relevant to this invariant.
- **Verification:** Replay the fixture and require the stated input-fact coverage, matched-fact multiplicity, and fact-grain total before release.
- **Mitigation risk/escalation:** Holding data delays a decision; an automatic fallback can conceal a semantic defect. Escalate when fallback changes a business action.
- **Boundary:** Passing this check establishes consistency with the written model, not truth of an unavailable source body.

#### DIAG-35 — unbounded recursion

- **Trigger/environment:** A bounded release changes a fact, dimension, bridge, or query predicate while the declared fact grain remains the correctness oracle.
- **Symptom:** A join-derived count, amount, slice, or latency differs from the certified comparison at the same cutoff.
- **Competing cause A:** A relationship, key, version, unit, or time invariant was violated for unbounded recursion.
- **Competing cause B:** A legitimate business-policy or population change produced the movement without invalidating the model.
- **Discriminating diagnostic:** Preserve pre-join input fact keys, matched fact keys, joined rows, predicate, cutoff, and source revision; compute coverage and multiplicity separately.
- **Expected signal for A:** An anti-join, overlap, key, unit, or cardinality test isolates the violating records.
- **Expected signal for B:** Fact-grain reconciliation passes and the declared business rule explains the changed population.
- **Immediate mitigation:** Hold the affected decision slice or publish its prior certified value; do not use DISTINCT as a cosmetic repair.
- **Correction:** Repair the owner boundary: source contract, relationship model, temporal selector, allocation policy, or query grain.
- **Prevention:** Add a release fixture containing zero, one, many, late, duplicate, and changed-version relationships relevant to this invariant.
- **Verification:** Replay the fixture and require the stated input-fact coverage, matched-fact multiplicity, and fact-grain total before release.
- **Mitigation risk/escalation:** Holding data delays a decision; an automatic fallback can conceal a semantic defect. Escalate when fallback changes a business action.
- **Boundary:** Passing this check establishes consistency with the written model, not truth of an unavailable source body.

#### DIAG-36 — stale materialized view

- **Trigger/environment:** A bounded release changes a fact, dimension, bridge, or query predicate while the declared fact grain remains the correctness oracle.
- **Symptom:** A join-derived count, amount, slice, or latency differs from the certified comparison at the same cutoff.
- **Competing cause A:** A relationship, key, version, unit, or time invariant was violated for stale materialized view.
- **Competing cause B:** A legitimate business-policy or population change produced the movement without invalidating the model.
- **Discriminating diagnostic:** Preserve pre-join input fact keys, matched fact keys, joined rows, predicate, cutoff, and source revision; compute coverage and multiplicity separately.
- **Expected signal for A:** An anti-join, overlap, key, unit, or cardinality test isolates the violating records.
- **Expected signal for B:** Fact-grain reconciliation passes and the declared business rule explains the changed population.
- **Immediate mitigation:** Hold the affected decision slice or publish its prior certified value; do not use DISTINCT as a cosmetic repair.
- **Correction:** Repair the owner boundary: source contract, relationship model, temporal selector, allocation policy, or query grain.
- **Prevention:** Add a release fixture containing zero, one, many, late, duplicate, and changed-version relationships relevant to this invariant.
- **Verification:** Replay the fixture and require the stated input-fact coverage, matched-fact multiplicity, and fact-grain total before release.
- **Mitigation risk/escalation:** Holding data delays a decision; an automatic fallback can conceal a semantic defect. Escalate when fallback changes a business action.
- **Boundary:** Passing this check establishes consistency with the written model, not truth of an unavailable source body.

#### DIAG-37 — late delete

- **Trigger/environment:** A bounded release changes a fact, dimension, bridge, or query predicate while the declared fact grain remains the correctness oracle.
- **Symptom:** A join-derived count, amount, slice, or latency differs from the certified comparison at the same cutoff.
- **Competing cause A:** A relationship, key, version, unit, or time invariant was violated for late delete.
- **Competing cause B:** A legitimate business-policy or population change produced the movement without invalidating the model.
- **Discriminating diagnostic:** Preserve pre-join input fact keys, matched fact keys, joined rows, predicate, cutoff, and source revision; compute coverage and multiplicity separately.
- **Expected signal for A:** An anti-join, overlap, key, unit, or cardinality test isolates the violating records.
- **Expected signal for B:** Fact-grain reconciliation passes and the declared business rule explains the changed population.
- **Immediate mitigation:** Hold the affected decision slice or publish its prior certified value; do not use DISTINCT as a cosmetic repair.
- **Correction:** Repair the owner boundary: source contract, relationship model, temporal selector, allocation policy, or query grain.
- **Prevention:** Add a release fixture containing zero, one, many, late, duplicate, and changed-version relationships relevant to this invariant.
- **Verification:** Replay the fixture and require the stated input-fact coverage, matched-fact multiplicity, and fact-grain total before release.
- **Mitigation risk/escalation:** Holding data delays a decision; an automatic fallback can conceal a semantic defect. Escalate when fallback changes a business action.
- **Boundary:** Passing this check establishes consistency with the written model, not truth of an unavailable source body.

#### DIAG-38 — cross-region clock skew

- **Trigger/environment:** A bounded release changes a fact, dimension, bridge, or query predicate while the declared fact grain remains the correctness oracle.
- **Symptom:** A join-derived count, amount, slice, or latency differs from the certified comparison at the same cutoff.
- **Competing cause A:** A relationship, key, version, unit, or time invariant was violated for cross-region clock skew.
- **Competing cause B:** A legitimate business-policy or population change produced the movement without invalidating the model.
- **Discriminating diagnostic:** Preserve pre-join input fact keys, matched fact keys, joined rows, predicate, cutoff, and source revision; compute coverage and multiplicity separately.
- **Expected signal for A:** An anti-join, overlap, key, unit, or cardinality test isolates the violating records.
- **Expected signal for B:** Fact-grain reconciliation passes and the declared business rule explains the changed population.
- **Immediate mitigation:** Hold the affected decision slice or publish its prior certified value; do not use DISTINCT as a cosmetic repair.
- **Correction:** Repair the owner boundary: source contract, relationship model, temporal selector, allocation policy, or query grain.
- **Prevention:** Add a release fixture containing zero, one, many, late, duplicate, and changed-version relationships relevant to this invariant.
- **Verification:** Replay the fixture and require the stated input-fact coverage, matched-fact multiplicity, and fact-grain total before release.
- **Mitigation risk/escalation:** Holding data delays a decision; an automatic fallback can conceal a semantic defect. Escalate when fallback changes a business action.
- **Boundary:** Passing this check establishes consistency with the written model, not truth of an unavailable source body.

#### DIAG-39 — partition pruning loss

- **Trigger/environment:** A bounded release changes a fact, dimension, bridge, or query predicate while the declared fact grain remains the correctness oracle.
- **Symptom:** A join-derived count, amount, slice, or latency differs from the certified comparison at the same cutoff.
- **Competing cause A:** A relationship, key, version, unit, or time invariant was violated for partition pruning loss.
- **Competing cause B:** A legitimate business-policy or population change produced the movement without invalidating the model.
- **Discriminating diagnostic:** Preserve pre-join input fact keys, matched fact keys, joined rows, predicate, cutoff, and source revision; compute coverage and multiplicity separately.
- **Expected signal for A:** An anti-join, overlap, key, unit, or cardinality test isolates the violating records.
- **Expected signal for B:** Fact-grain reconciliation passes and the declared business rule explains the changed population.
- **Immediate mitigation:** Hold the affected decision slice or publish its prior certified value; do not use DISTINCT as a cosmetic repair.
- **Correction:** Repair the owner boundary: source contract, relationship model, temporal selector, allocation policy, or query grain.
- **Prevention:** Add a release fixture containing zero, one, many, late, duplicate, and changed-version relationships relevant to this invariant.
- **Verification:** Replay the fixture and require the stated input-fact coverage, matched-fact multiplicity, and fact-grain total before release.
- **Mitigation risk/escalation:** Holding data delays a decision; an automatic fallback can conceal a semantic defect. Escalate when fallback changes a business action.
- **Boundary:** Passing this check establishes consistency with the written model, not truth of an unavailable source body.

#### DIAG-40 — wrong snapshot cutoff

- **Trigger/environment:** A bounded release changes a fact, dimension, bridge, or query predicate while the declared fact grain remains the correctness oracle.
- **Symptom:** A join-derived count, amount, slice, or latency differs from the certified comparison at the same cutoff.
- **Competing cause A:** A relationship, key, version, unit, or time invariant was violated for wrong snapshot cutoff.
- **Competing cause B:** A legitimate business-policy or population change produced the movement without invalidating the model.
- **Discriminating diagnostic:** Preserve pre-join input fact keys, matched fact keys, joined rows, predicate, cutoff, and source revision; compute coverage and multiplicity separately.
- **Expected signal for A:** An anti-join, overlap, key, unit, or cardinality test isolates the violating records.
- **Expected signal for B:** Fact-grain reconciliation passes and the declared business rule explains the changed population.
- **Immediate mitigation:** Hold the affected decision slice or publish its prior certified value; do not use DISTINCT as a cosmetic repair.
- **Correction:** Repair the owner boundary: source contract, relationship model, temporal selector, allocation policy, or query grain.
- **Prevention:** Add a release fixture containing zero, one, many, late, duplicate, and changed-version relationships relevant to this invariant.
- **Verification:** Replay the fixture and require the stated input-fact coverage, matched-fact multiplicity, and fact-grain total before release.
- **Mitigation risk/escalation:** Holding data delays a decision; an automatic fallback can conceal a semantic defect. Escalate when fallback changes a business action.
- **Boundary:** Passing this check establishes consistency with the written model, not truth of an unavailable source body.

### Expanded interview defence and scoring

#### Defence 1 — Join-safety changed constraint

- **Question:** Given a plausible aggregate movement, what must be measured before selecting a join repair?
- **Checked answer:** Declare fact grain and cutoff; compare distinct matched input facts to input facts for coverage, and joined rows to distinct matched facts for multiplicity. Then inspect the smallest relation violating its invariant.
- **Reasoning:** Raw row count can remain unchanged when a lost fact and a duplicated fact cancel. Parser success and a matching total do not establish semantic correctness.
- **Follow-up:** If the report must ship before full reconciliation, use a clearly labeled prior certified value or unknown segment, with metric-owner approval.
- **Acceptable alternative:** A pre-aggregated governed mart is acceptable only when its grain, watermark, and reconciliation rule are retained.
- **Red flags:** DISTINCT as universal repair, output row count as coverage, and a vendor/tool name instead of an invariant.
- **Scoring:** 0 tool-name; 1 join issue; 2 grain plus one check; 3 coverage/multiplicity plus mitigation; 4 replay oracle, changed constraint, and decision consequence.
- **Support:** Sections 3–5 define grain and arithmetic; Sections 8–9 define diagnostic and cost boundaries.

#### Defence 2 — Join-safety changed constraint

- **Question:** Given a plausible aggregate movement, what must be measured before selecting a join repair?
- **Checked answer:** Declare fact grain and cutoff; compare distinct matched input facts to input facts for coverage, and joined rows to distinct matched facts for multiplicity. Then inspect the smallest relation violating its invariant.
- **Reasoning:** Raw row count can remain unchanged when a lost fact and a duplicated fact cancel. Parser success and a matching total do not establish semantic correctness.
- **Follow-up:** If the report must ship before full reconciliation, use a clearly labeled prior certified value or unknown segment, with metric-owner approval.
- **Acceptable alternative:** A pre-aggregated governed mart is acceptable only when its grain, watermark, and reconciliation rule are retained.
- **Red flags:** DISTINCT as universal repair, output row count as coverage, and a vendor/tool name instead of an invariant.
- **Scoring:** 0 tool-name; 1 join issue; 2 grain plus one check; 3 coverage/multiplicity plus mitigation; 4 replay oracle, changed constraint, and decision consequence.
- **Support:** Sections 3–5 define grain and arithmetic; Sections 8–9 define diagnostic and cost boundaries.

#### Defence 3 — Join-safety changed constraint

- **Question:** Given a plausible aggregate movement, what must be measured before selecting a join repair?
- **Checked answer:** Declare fact grain and cutoff; compare distinct matched input facts to input facts for coverage, and joined rows to distinct matched facts for multiplicity. Then inspect the smallest relation violating its invariant.
- **Reasoning:** Raw row count can remain unchanged when a lost fact and a duplicated fact cancel. Parser success and a matching total do not establish semantic correctness.
- **Follow-up:** If the report must ship before full reconciliation, use a clearly labeled prior certified value or unknown segment, with metric-owner approval.
- **Acceptable alternative:** A pre-aggregated governed mart is acceptable only when its grain, watermark, and reconciliation rule are retained.
- **Red flags:** DISTINCT as universal repair, output row count as coverage, and a vendor/tool name instead of an invariant.
- **Scoring:** 0 tool-name; 1 join issue; 2 grain plus one check; 3 coverage/multiplicity plus mitigation; 4 replay oracle, changed constraint, and decision consequence.
- **Support:** Sections 3–5 define grain and arithmetic; Sections 8–9 define diagnostic and cost boundaries.

#### Defence 4 — Join-safety changed constraint

- **Question:** Given a plausible aggregate movement, what must be measured before selecting a join repair?
- **Checked answer:** Declare fact grain and cutoff; compare distinct matched input facts to input facts for coverage, and joined rows to distinct matched facts for multiplicity. Then inspect the smallest relation violating its invariant.
- **Reasoning:** Raw row count can remain unchanged when a lost fact and a duplicated fact cancel. Parser success and a matching total do not establish semantic correctness.
- **Follow-up:** If the report must ship before full reconciliation, use a clearly labeled prior certified value or unknown segment, with metric-owner approval.
- **Acceptable alternative:** A pre-aggregated governed mart is acceptable only when its grain, watermark, and reconciliation rule are retained.
- **Red flags:** DISTINCT as universal repair, output row count as coverage, and a vendor/tool name instead of an invariant.
- **Scoring:** 0 tool-name; 1 join issue; 2 grain plus one check; 3 coverage/multiplicity plus mitigation; 4 replay oracle, changed constraint, and decision consequence.
- **Support:** Sections 3–5 define grain and arithmetic; Sections 8–9 define diagnostic and cost boundaries.

#### Defence 5 — Join-safety changed constraint

- **Question:** Given a plausible aggregate movement, what must be measured before selecting a join repair?
- **Checked answer:** Declare fact grain and cutoff; compare distinct matched input facts to input facts for coverage, and joined rows to distinct matched facts for multiplicity. Then inspect the smallest relation violating its invariant.
- **Reasoning:** Raw row count can remain unchanged when a lost fact and a duplicated fact cancel. Parser success and a matching total do not establish semantic correctness.
- **Follow-up:** If the report must ship before full reconciliation, use a clearly labeled prior certified value or unknown segment, with metric-owner approval.
- **Acceptable alternative:** A pre-aggregated governed mart is acceptable only when its grain, watermark, and reconciliation rule are retained.
- **Red flags:** DISTINCT as universal repair, output row count as coverage, and a vendor/tool name instead of an invariant.
- **Scoring:** 0 tool-name; 1 join issue; 2 grain plus one check; 3 coverage/multiplicity plus mitigation; 4 replay oracle, changed constraint, and decision consequence.
- **Support:** Sections 3–5 define grain and arithmetic; Sections 8–9 define diagnostic and cost boundaries.

#### Defence 6 — Join-safety changed constraint

- **Question:** Given a plausible aggregate movement, what must be measured before selecting a join repair?
- **Checked answer:** Declare fact grain and cutoff; compare distinct matched input facts to input facts for coverage, and joined rows to distinct matched facts for multiplicity. Then inspect the smallest relation violating its invariant.
- **Reasoning:** Raw row count can remain unchanged when a lost fact and a duplicated fact cancel. Parser success and a matching total do not establish semantic correctness.
- **Follow-up:** If the report must ship before full reconciliation, use a clearly labeled prior certified value or unknown segment, with metric-owner approval.
- **Acceptable alternative:** A pre-aggregated governed mart is acceptable only when its grain, watermark, and reconciliation rule are retained.
- **Red flags:** DISTINCT as universal repair, output row count as coverage, and a vendor/tool name instead of an invariant.
- **Scoring:** 0 tool-name; 1 join issue; 2 grain plus one check; 3 coverage/multiplicity plus mitigation; 4 replay oracle, changed constraint, and decision consequence.
- **Support:** Sections 3–5 define grain and arithmetic; Sections 8–9 define diagnostic and cost boundaries.

#### Defence 7 — Join-safety changed constraint

- **Question:** Given a plausible aggregate movement, what must be measured before selecting a join repair?
- **Checked answer:** Declare fact grain and cutoff; compare distinct matched input facts to input facts for coverage, and joined rows to distinct matched facts for multiplicity. Then inspect the smallest relation violating its invariant.
- **Reasoning:** Raw row count can remain unchanged when a lost fact and a duplicated fact cancel. Parser success and a matching total do not establish semantic correctness.
- **Follow-up:** If the report must ship before full reconciliation, use a clearly labeled prior certified value or unknown segment, with metric-owner approval.
- **Acceptable alternative:** A pre-aggregated governed mart is acceptable only when its grain, watermark, and reconciliation rule are retained.
- **Red flags:** DISTINCT as universal repair, output row count as coverage, and a vendor/tool name instead of an invariant.
- **Scoring:** 0 tool-name; 1 join issue; 2 grain plus one check; 3 coverage/multiplicity plus mitigation; 4 replay oracle, changed constraint, and decision consequence.
- **Support:** Sections 3–5 define grain and arithmetic; Sections 8–9 define diagnostic and cost boundaries.

#### Defence 8 — Join-safety changed constraint

- **Question:** Given a plausible aggregate movement, what must be measured before selecting a join repair?
- **Checked answer:** Declare fact grain and cutoff; compare distinct matched input facts to input facts for coverage, and joined rows to distinct matched facts for multiplicity. Then inspect the smallest relation violating its invariant.
- **Reasoning:** Raw row count can remain unchanged when a lost fact and a duplicated fact cancel. Parser success and a matching total do not establish semantic correctness.
- **Follow-up:** If the report must ship before full reconciliation, use a clearly labeled prior certified value or unknown segment, with metric-owner approval.
- **Acceptable alternative:** A pre-aggregated governed mart is acceptable only when its grain, watermark, and reconciliation rule are retained.
- **Red flags:** DISTINCT as universal repair, output row count as coverage, and a vendor/tool name instead of an invariant.
- **Scoring:** 0 tool-name; 1 join issue; 2 grain plus one check; 3 coverage/multiplicity plus mitigation; 4 replay oracle, changed constraint, and decision consequence.
- **Support:** Sections 3–5 define grain and arithmetic; Sections 8–9 define diagnostic and cost boundaries.

#### Defence 9 — Join-safety changed constraint

- **Question:** Given a plausible aggregate movement, what must be measured before selecting a join repair?
- **Checked answer:** Declare fact grain and cutoff; compare distinct matched input facts to input facts for coverage, and joined rows to distinct matched facts for multiplicity. Then inspect the smallest relation violating its invariant.
- **Reasoning:** Raw row count can remain unchanged when a lost fact and a duplicated fact cancel. Parser success and a matching total do not establish semantic correctness.
- **Follow-up:** If the report must ship before full reconciliation, use a clearly labeled prior certified value or unknown segment, with metric-owner approval.
- **Acceptable alternative:** A pre-aggregated governed mart is acceptable only when its grain, watermark, and reconciliation rule are retained.
- **Red flags:** DISTINCT as universal repair, output row count as coverage, and a vendor/tool name instead of an invariant.
- **Scoring:** 0 tool-name; 1 join issue; 2 grain plus one check; 3 coverage/multiplicity plus mitigation; 4 replay oracle, changed constraint, and decision consequence.
- **Support:** Sections 3–5 define grain and arithmetic; Sections 8–9 define diagnostic and cost boundaries.

#### Defence 10 — Join-safety changed constraint

- **Question:** Given a plausible aggregate movement, what must be measured before selecting a join repair?
- **Checked answer:** Declare fact grain and cutoff; compare distinct matched input facts to input facts for coverage, and joined rows to distinct matched facts for multiplicity. Then inspect the smallest relation violating its invariant.
- **Reasoning:** Raw row count can remain unchanged when a lost fact and a duplicated fact cancel. Parser success and a matching total do not establish semantic correctness.
- **Follow-up:** If the report must ship before full reconciliation, use a clearly labeled prior certified value or unknown segment, with metric-owner approval.
- **Acceptable alternative:** A pre-aggregated governed mart is acceptable only when its grain, watermark, and reconciliation rule are retained.
- **Red flags:** DISTINCT as universal repair, output row count as coverage, and a vendor/tool name instead of an invariant.
- **Scoring:** 0 tool-name; 1 join issue; 2 grain plus one check; 3 coverage/multiplicity plus mitigation; 4 replay oracle, changed constraint, and decision consequence.
- **Support:** Sections 3–5 define grain and arithmetic; Sections 8–9 define diagnostic and cost boundaries.

#### Defence 11 — Join-safety changed constraint

- **Question:** Given a plausible aggregate movement, what must be measured before selecting a join repair?
- **Checked answer:** Declare fact grain and cutoff; compare distinct matched input facts to input facts for coverage, and joined rows to distinct matched facts for multiplicity. Then inspect the smallest relation violating its invariant.
- **Reasoning:** Raw row count can remain unchanged when a lost fact and a duplicated fact cancel. Parser success and a matching total do not establish semantic correctness.
- **Follow-up:** If the report must ship before full reconciliation, use a clearly labeled prior certified value or unknown segment, with metric-owner approval.
- **Acceptable alternative:** A pre-aggregated governed mart is acceptable only when its grain, watermark, and reconciliation rule are retained.
- **Red flags:** DISTINCT as universal repair, output row count as coverage, and a vendor/tool name instead of an invariant.
- **Scoring:** 0 tool-name; 1 join issue; 2 grain plus one check; 3 coverage/multiplicity plus mitigation; 4 replay oracle, changed constraint, and decision consequence.
- **Support:** Sections 3–5 define grain and arithmetic; Sections 8–9 define diagnostic and cost boundaries.

#### Defence 12 — Join-safety changed constraint

- **Question:** Given a plausible aggregate movement, what must be measured before selecting a join repair?
- **Checked answer:** Declare fact grain and cutoff; compare distinct matched input facts to input facts for coverage, and joined rows to distinct matched facts for multiplicity. Then inspect the smallest relation violating its invariant.
- **Reasoning:** Raw row count can remain unchanged when a lost fact and a duplicated fact cancel. Parser success and a matching total do not establish semantic correctness.
- **Follow-up:** If the report must ship before full reconciliation, use a clearly labeled prior certified value or unknown segment, with metric-owner approval.
- **Acceptable alternative:** A pre-aggregated governed mart is acceptable only when its grain, watermark, and reconciliation rule are retained.
- **Red flags:** DISTINCT as universal repair, output row count as coverage, and a vendor/tool name instead of an invariant.
- **Scoring:** 0 tool-name; 1 join issue; 2 grain plus one check; 3 coverage/multiplicity plus mitigation; 4 replay oracle, changed constraint, and decision consequence.
- **Support:** Sections 3–5 define grain and arithmetic; Sections 8–9 define diagnostic and cost boundaries.

#### Defence 13 — Join-safety changed constraint

- **Question:** Given a plausible aggregate movement, what must be measured before selecting a join repair?
- **Checked answer:** Declare fact grain and cutoff; compare distinct matched input facts to input facts for coverage, and joined rows to distinct matched facts for multiplicity. Then inspect the smallest relation violating its invariant.
- **Reasoning:** Raw row count can remain unchanged when a lost fact and a duplicated fact cancel. Parser success and a matching total do not establish semantic correctness.
- **Follow-up:** If the report must ship before full reconciliation, use a clearly labeled prior certified value or unknown segment, with metric-owner approval.
- **Acceptable alternative:** A pre-aggregated governed mart is acceptable only when its grain, watermark, and reconciliation rule are retained.
- **Red flags:** DISTINCT as universal repair, output row count as coverage, and a vendor/tool name instead of an invariant.
- **Scoring:** 0 tool-name; 1 join issue; 2 grain plus one check; 3 coverage/multiplicity plus mitigation; 4 replay oracle, changed constraint, and decision consequence.
- **Support:** Sections 3–5 define grain and arithmetic; Sections 8–9 define diagnostic and cost boundaries.

#### Defence 14 — Join-safety changed constraint

- **Question:** Given a plausible aggregate movement, what must be measured before selecting a join repair?
- **Checked answer:** Declare fact grain and cutoff; compare distinct matched input facts to input facts for coverage, and joined rows to distinct matched facts for multiplicity. Then inspect the smallest relation violating its invariant.
- **Reasoning:** Raw row count can remain unchanged when a lost fact and a duplicated fact cancel. Parser success and a matching total do not establish semantic correctness.
- **Follow-up:** If the report must ship before full reconciliation, use a clearly labeled prior certified value or unknown segment, with metric-owner approval.
- **Acceptable alternative:** A pre-aggregated governed mart is acceptable only when its grain, watermark, and reconciliation rule are retained.
- **Red flags:** DISTINCT as universal repair, output row count as coverage, and a vendor/tool name instead of an invariant.
- **Scoring:** 0 tool-name; 1 join issue; 2 grain plus one check; 3 coverage/multiplicity plus mitigation; 4 replay oracle, changed constraint, and decision consequence.
- **Support:** Sections 3–5 define grain and arithmetic; Sections 8–9 define diagnostic and cost boundaries.

#### Defence 15 — Join-safety changed constraint

- **Question:** Given a plausible aggregate movement, what must be measured before selecting a join repair?
- **Checked answer:** Declare fact grain and cutoff; compare distinct matched input facts to input facts for coverage, and joined rows to distinct matched facts for multiplicity. Then inspect the smallest relation violating its invariant.
- **Reasoning:** Raw row count can remain unchanged when a lost fact and a duplicated fact cancel. Parser success and a matching total do not establish semantic correctness.
- **Follow-up:** If the report must ship before full reconciliation, use a clearly labeled prior certified value or unknown segment, with metric-owner approval.
- **Acceptable alternative:** A pre-aggregated governed mart is acceptable only when its grain, watermark, and reconciliation rule are retained.
- **Red flags:** DISTINCT as universal repair, output row count as coverage, and a vendor/tool name instead of an invariant.
- **Scoring:** 0 tool-name; 1 join issue; 2 grain plus one check; 3 coverage/multiplicity plus mitigation; 4 replay oracle, changed constraint, and decision consequence.
- **Support:** Sections 3–5 define grain and arithmetic; Sections 8–9 define diagnostic and cost boundaries.

#### Defence 16 — Join-safety changed constraint

- **Question:** Given a plausible aggregate movement, what must be measured before selecting a join repair?
- **Checked answer:** Declare fact grain and cutoff; compare distinct matched input facts to input facts for coverage, and joined rows to distinct matched facts for multiplicity. Then inspect the smallest relation violating its invariant.
- **Reasoning:** Raw row count can remain unchanged when a lost fact and a duplicated fact cancel. Parser success and a matching total do not establish semantic correctness.
- **Follow-up:** If the report must ship before full reconciliation, use a clearly labeled prior certified value or unknown segment, with metric-owner approval.
- **Acceptable alternative:** A pre-aggregated governed mart is acceptable only when its grain, watermark, and reconciliation rule are retained.
- **Red flags:** DISTINCT as universal repair, output row count as coverage, and a vendor/tool name instead of an invariant.
- **Scoring:** 0 tool-name; 1 join issue; 2 grain plus one check; 3 coverage/multiplicity plus mitigation; 4 replay oracle, changed constraint, and decision consequence.
- **Support:** Sections 3–5 define grain and arithmetic; Sections 8–9 define diagnostic and cost boundaries.

#### Defence 17 — Join-safety changed constraint

- **Question:** Given a plausible aggregate movement, what must be measured before selecting a join repair?
- **Checked answer:** Declare fact grain and cutoff; compare distinct matched input facts to input facts for coverage, and joined rows to distinct matched facts for multiplicity. Then inspect the smallest relation violating its invariant.
- **Reasoning:** Raw row count can remain unchanged when a lost fact and a duplicated fact cancel. Parser success and a matching total do not establish semantic correctness.
- **Follow-up:** If the report must ship before full reconciliation, use a clearly labeled prior certified value or unknown segment, with metric-owner approval.
- **Acceptable alternative:** A pre-aggregated governed mart is acceptable only when its grain, watermark, and reconciliation rule are retained.
- **Red flags:** DISTINCT as universal repair, output row count as coverage, and a vendor/tool name instead of an invariant.
- **Scoring:** 0 tool-name; 1 join issue; 2 grain plus one check; 3 coverage/multiplicity plus mitigation; 4 replay oracle, changed constraint, and decision consequence.
- **Support:** Sections 3–5 define grain and arithmetic; Sections 8–9 define diagnostic and cost boundaries.

#### Defence 18 — Join-safety changed constraint

- **Question:** Given a plausible aggregate movement, what must be measured before selecting a join repair?
- **Checked answer:** Declare fact grain and cutoff; compare distinct matched input facts to input facts for coverage, and joined rows to distinct matched facts for multiplicity. Then inspect the smallest relation violating its invariant.
- **Reasoning:** Raw row count can remain unchanged when a lost fact and a duplicated fact cancel. Parser success and a matching total do not establish semantic correctness.
- **Follow-up:** If the report must ship before full reconciliation, use a clearly labeled prior certified value or unknown segment, with metric-owner approval.
- **Acceptable alternative:** A pre-aggregated governed mart is acceptable only when its grain, watermark, and reconciliation rule are retained.
- **Red flags:** DISTINCT as universal repair, output row count as coverage, and a vendor/tool name instead of an invariant.
- **Scoring:** 0 tool-name; 1 join issue; 2 grain plus one check; 3 coverage/multiplicity plus mitigation; 4 replay oracle, changed constraint, and decision consequence.
- **Support:** Sections 3–5 define grain and arithmetic; Sections 8–9 define diagnostic and cost boundaries.

#### Defence 19 — Join-safety changed constraint

- **Question:** Given a plausible aggregate movement, what must be measured before selecting a join repair?
- **Checked answer:** Declare fact grain and cutoff; compare distinct matched input facts to input facts for coverage, and joined rows to distinct matched facts for multiplicity. Then inspect the smallest relation violating its invariant.
- **Reasoning:** Raw row count can remain unchanged when a lost fact and a duplicated fact cancel. Parser success and a matching total do not establish semantic correctness.
- **Follow-up:** If the report must ship before full reconciliation, use a clearly labeled prior certified value or unknown segment, with metric-owner approval.
- **Acceptable alternative:** A pre-aggregated governed mart is acceptable only when its grain, watermark, and reconciliation rule are retained.
- **Red flags:** DISTINCT as universal repair, output row count as coverage, and a vendor/tool name instead of an invariant.
- **Scoring:** 0 tool-name; 1 join issue; 2 grain plus one check; 3 coverage/multiplicity plus mitigation; 4 replay oracle, changed constraint, and decision consequence.
- **Support:** Sections 3–5 define grain and arithmetic; Sections 8–9 define diagnostic and cost boundaries.

#### Defence 20 — Join-safety changed constraint

- **Question:** Given a plausible aggregate movement, what must be measured before selecting a join repair?
- **Checked answer:** Declare fact grain and cutoff; compare distinct matched input facts to input facts for coverage, and joined rows to distinct matched facts for multiplicity. Then inspect the smallest relation violating its invariant.
- **Reasoning:** Raw row count can remain unchanged when a lost fact and a duplicated fact cancel. Parser success and a matching total do not establish semantic correctness.
- **Follow-up:** If the report must ship before full reconciliation, use a clearly labeled prior certified value or unknown segment, with metric-owner approval.
- **Acceptable alternative:** A pre-aggregated governed mart is acceptable only when its grain, watermark, and reconciliation rule are retained.
- **Red flags:** DISTINCT as universal repair, output row count as coverage, and a vendor/tool name instead of an invariant.
- **Scoring:** 0 tool-name; 1 join issue; 2 grain plus one check; 3 coverage/multiplicity plus mitigation; 4 replay oracle, changed constraint, and decision consequence.
- **Support:** Sections 3–5 define grain and arithmetic; Sections 8–9 define diagnostic and cost boundaries.

## 9. Complexity & Cost Analysis

A hash join that materializes `J` output rows has expected `O(F + D + J)` time for fact rows `F` and build/probe relation rows `D`, assuming hash operations are expected constant time and the chosen build side fits available memory. Build-side memory is `O(D)` for its keys, payload, and hash-table overhead; if it does not fit, partitioning or spill adds I/O and can dominate elapsed time. A bridge increases `J` with relationship edges, not merely facts. Illustrative sizing: 10 million facts with an average 1.4 bridge edges produce 14 million joined rows; the 40% increase affects materialization, shuffle, and downstream aggregation. This is a derived workload estimate, not a benchmark.

**Takeaway:** Cardinality is a cost input as well as a correctness input.


### Worked counterexample traces

Each trace uses ILLUSTRATIVE values and hand-derived arithmetic. It is an oracle input for a parent-run query or replay, not an executed platform test.

#### TRACE-01 — two-child bridge

- **Input grain:** Three immutable facts A=$40, B=$60, C=$100 at one UTC cutoff; input fact-key set is {A,B,C}.
- **Relationship perturbation:** Apply the named condition to A, B, or C while retaining the original fact payload as the independent oracle.
- **Expected coverage calculation:** Count distinct input keys with at least one valid match, then divide by 3; record unmatched keys explicitly.
- **Expected multiplicity calculation:** Divide materialized joined rows by distinct matched fact keys; do not use total input rows as this denominator.
- **Conservation check:** Sum source facts once: $40+$60+$100=$200. Any allocated descendant total must equal $200 before rounding residual policy.
- **Counterexample condition:** A row count of three can still be wrong: duplicate A, retain B, and lose C produces three rows but only two matched fact keys and $140 naive revenue.
- **Selection consequence:** Use association count, explicit allocation, a temporal selector, or a hold depending on the stated relationship semantics.
- **Independent arithmetic:** Coverage is 2/3=0.666666…; multiplicity is 3/2=1.5; neither value can be inferred from output count alone.
- **Sensitivity:** Replacing a lost $100 fact with a duplicated $40 fact changes currency error while coverage/multiplicity still diagnose structure.
- **Failure boundary:** If the business rule permits multiple memberships, multiplicity above one is expected but additive revenue still requires allocation or a non-additive label.
- **Verification protocol:** Execute a reference query over the three facts, compare key sets and totals, then repeat after the perturbation with the same cutoff.
- **Evidence boundary:** This teaches relational behavior; it makes no claim about a vendor optimizer, deployed database, or source-system result.

#### TRACE-02 — dropped high-value fact

- **Input grain:** Three immutable facts A=$40, B=$60, C=$100 at one UTC cutoff; input fact-key set is {A,B,C}.
- **Relationship perturbation:** Apply the named condition to A, B, or C while retaining the original fact payload as the independent oracle.
- **Expected coverage calculation:** Count distinct input keys with at least one valid match, then divide by 3; record unmatched keys explicitly.
- **Expected multiplicity calculation:** Divide materialized joined rows by distinct matched fact keys; do not use total input rows as this denominator.
- **Conservation check:** Sum source facts once: $40+$60+$100=$200. Any allocated descendant total must equal $200 before rounding residual policy.
- **Counterexample condition:** A row count of three can still be wrong: duplicate A, retain B, and lose C produces three rows but only two matched fact keys and $140 naive revenue.
- **Selection consequence:** Use association count, explicit allocation, a temporal selector, or a hold depending on the stated relationship semantics.
- **Independent arithmetic:** Coverage is 2/3=0.666666…; multiplicity is 3/2=1.5; neither value can be inferred from output count alone.
- **Sensitivity:** Replacing a lost $100 fact with a duplicated $40 fact changes currency error while coverage/multiplicity still diagnose structure.
- **Failure boundary:** If the business rule permits multiple memberships, multiplicity above one is expected but additive revenue still requires allocation or a non-additive label.
- **Verification protocol:** Execute a reference query over the three facts, compare key sets and totals, then repeat after the perturbation with the same cutoff.
- **Evidence boundary:** This teaches relational behavior; it makes no claim about a vendor optimizer, deployed database, or source-system result.

#### TRACE-03 — type-2 overlap

- **Input grain:** Three immutable facts A=$40, B=$60, C=$100 at one UTC cutoff; input fact-key set is {A,B,C}.
- **Relationship perturbation:** Apply the named condition to A, B, or C while retaining the original fact payload as the independent oracle.
- **Expected coverage calculation:** Count distinct input keys with at least one valid match, then divide by 3; record unmatched keys explicitly.
- **Expected multiplicity calculation:** Divide materialized joined rows by distinct matched fact keys; do not use total input rows as this denominator.
- **Conservation check:** Sum source facts once: $40+$60+$100=$200. Any allocated descendant total must equal $200 before rounding residual policy.
- **Counterexample condition:** A row count of three can still be wrong: duplicate A, retain B, and lose C produces three rows but only two matched fact keys and $140 naive revenue.
- **Selection consequence:** Use association count, explicit allocation, a temporal selector, or a hold depending on the stated relationship semantics.
- **Independent arithmetic:** Coverage is 2/3=0.666666…; multiplicity is 3/2=1.5; neither value can be inferred from output count alone.
- **Sensitivity:** Replacing a lost $100 fact with a duplicated $40 fact changes currency error while coverage/multiplicity still diagnose structure.
- **Failure boundary:** If the business rule permits multiple memberships, multiplicity above one is expected but additive revenue still requires allocation or a non-additive label.
- **Verification protocol:** Execute a reference query over the three facts, compare key sets and totals, then repeat after the perturbation with the same cutoff.
- **Evidence boundary:** This teaches relational behavior; it makes no claim about a vendor optimizer, deployed database, or source-system result.

#### TRACE-04 — type-2 gap

- **Input grain:** Three immutable facts A=$40, B=$60, C=$100 at one UTC cutoff; input fact-key set is {A,B,C}.
- **Relationship perturbation:** Apply the named condition to A, B, or C while retaining the original fact payload as the independent oracle.
- **Expected coverage calculation:** Count distinct input keys with at least one valid match, then divide by 3; record unmatched keys explicitly.
- **Expected multiplicity calculation:** Divide materialized joined rows by distinct matched fact keys; do not use total input rows as this denominator.
- **Conservation check:** Sum source facts once: $40+$60+$100=$200. Any allocated descendant total must equal $200 before rounding residual policy.
- **Counterexample condition:** A row count of three can still be wrong: duplicate A, retain B, and lose C produces three rows but only two matched fact keys and $140 naive revenue.
- **Selection consequence:** Use association count, explicit allocation, a temporal selector, or a hold depending on the stated relationship semantics.
- **Independent arithmetic:** Coverage is 2/3=0.666666…; multiplicity is 3/2=1.5; neither value can be inferred from output count alone.
- **Sensitivity:** Replacing a lost $100 fact with a duplicated $40 fact changes currency error while coverage/multiplicity still diagnose structure.
- **Failure boundary:** If the business rule permits multiple memberships, multiplicity above one is expected but additive revenue still requires allocation or a non-additive label.
- **Verification protocol:** Execute a reference query over the three facts, compare key sets and totals, then repeat after the perturbation with the same cutoff.
- **Evidence boundary:** This teaches relational behavior; it makes no claim about a vendor optimizer, deployed database, or source-system result.

#### TRACE-05 — late dimension arrival

- **Input grain:** Three immutable facts A=$40, B=$60, C=$100 at one UTC cutoff; input fact-key set is {A,B,C}.
- **Relationship perturbation:** Apply the named condition to A, B, or C while retaining the original fact payload as the independent oracle.
- **Expected coverage calculation:** Count distinct input keys with at least one valid match, then divide by 3; record unmatched keys explicitly.
- **Expected multiplicity calculation:** Divide materialized joined rows by distinct matched fact keys; do not use total input rows as this denominator.
- **Conservation check:** Sum source facts once: $40+$60+$100=$200. Any allocated descendant total must equal $200 before rounding residual policy.
- **Counterexample condition:** A row count of three can still be wrong: duplicate A, retain B, and lose C produces three rows but only two matched fact keys and $140 naive revenue.
- **Selection consequence:** Use association count, explicit allocation, a temporal selector, or a hold depending on the stated relationship semantics.
- **Independent arithmetic:** Coverage is 2/3=0.666666…; multiplicity is 3/2=1.5; neither value can be inferred from output count alone.
- **Sensitivity:** Replacing a lost $100 fact with a duplicated $40 fact changes currency error while coverage/multiplicity still diagnose structure.
- **Failure boundary:** If the business rule permits multiple memberships, multiplicity above one is expected but additive revenue still requires allocation or a non-additive label.
- **Verification protocol:** Execute a reference query over the three facts, compare key sets and totals, then repeat after the perturbation with the same cutoff.
- **Evidence boundary:** This teaches relational behavior; it makes no claim about a vendor optimizer, deployed database, or source-system result.

#### TRACE-06 — cross-tenant collision

- **Input grain:** Three immutable facts A=$40, B=$60, C=$100 at one UTC cutoff; input fact-key set is {A,B,C}.
- **Relationship perturbation:** Apply the named condition to A, B, or C while retaining the original fact payload as the independent oracle.
- **Expected coverage calculation:** Count distinct input keys with at least one valid match, then divide by 3; record unmatched keys explicitly.
- **Expected multiplicity calculation:** Divide materialized joined rows by distinct matched fact keys; do not use total input rows as this denominator.
- **Conservation check:** Sum source facts once: $40+$60+$100=$200. Any allocated descendant total must equal $200 before rounding residual policy.
- **Counterexample condition:** A row count of three can still be wrong: duplicate A, retain B, and lose C produces three rows but only two matched fact keys and $140 naive revenue.
- **Selection consequence:** Use association count, explicit allocation, a temporal selector, or a hold depending on the stated relationship semantics.
- **Independent arithmetic:** Coverage is 2/3=0.666666…; multiplicity is 3/2=1.5; neither value can be inferred from output count alone.
- **Sensitivity:** Replacing a lost $100 fact with a duplicated $40 fact changes currency error while coverage/multiplicity still diagnose structure.
- **Failure boundary:** If the business rule permits multiple memberships, multiplicity above one is expected but additive revenue still requires allocation or a non-additive label.
- **Verification protocol:** Execute a reference query over the three facts, compare key sets and totals, then repeat after the perturbation with the same cutoff.
- **Evidence boundary:** This teaches relational behavior; it makes no claim about a vendor optimizer, deployed database, or source-system result.

#### TRACE-07 — nullable foreign key

- **Input grain:** Three immutable facts A=$40, B=$60, C=$100 at one UTC cutoff; input fact-key set is {A,B,C}.
- **Relationship perturbation:** Apply the named condition to A, B, or C while retaining the original fact payload as the independent oracle.
- **Expected coverage calculation:** Count distinct input keys with at least one valid match, then divide by 3; record unmatched keys explicitly.
- **Expected multiplicity calculation:** Divide materialized joined rows by distinct matched fact keys; do not use total input rows as this denominator.
- **Conservation check:** Sum source facts once: $40+$60+$100=$200. Any allocated descendant total must equal $200 before rounding residual policy.
- **Counterexample condition:** A row count of three can still be wrong: duplicate A, retain B, and lose C produces three rows but only two matched fact keys and $140 naive revenue.
- **Selection consequence:** Use association count, explicit allocation, a temporal selector, or a hold depending on the stated relationship semantics.
- **Independent arithmetic:** Coverage is 2/3=0.666666…; multiplicity is 3/2=1.5; neither value can be inferred from output count alone.
- **Sensitivity:** Replacing a lost $100 fact with a duplicated $40 fact changes currency error while coverage/multiplicity still diagnose structure.
- **Failure boundary:** If the business rule permits multiple memberships, multiplicity above one is expected but additive revenue still requires allocation or a non-additive label.
- **Verification protocol:** Execute a reference query over the three facts, compare key sets and totals, then repeat after the perturbation with the same cutoff.
- **Evidence boundary:** This teaches relational behavior; it makes no claim about a vendor optimizer, deployed database, or source-system result.

#### TRACE-08 — soft-deleted membership

- **Input grain:** Three immutable facts A=$40, B=$60, C=$100 at one UTC cutoff; input fact-key set is {A,B,C}.
- **Relationship perturbation:** Apply the named condition to A, B, or C while retaining the original fact payload as the independent oracle.
- **Expected coverage calculation:** Count distinct input keys with at least one valid match, then divide by 3; record unmatched keys explicitly.
- **Expected multiplicity calculation:** Divide materialized joined rows by distinct matched fact keys; do not use total input rows as this denominator.
- **Conservation check:** Sum source facts once: $40+$60+$100=$200. Any allocated descendant total must equal $200 before rounding residual policy.
- **Counterexample condition:** A row count of three can still be wrong: duplicate A, retain B, and lose C produces three rows but only two matched fact keys and $140 naive revenue.
- **Selection consequence:** Use association count, explicit allocation, a temporal selector, or a hold depending on the stated relationship semantics.
- **Independent arithmetic:** Coverage is 2/3=0.666666…; multiplicity is 3/2=1.5; neither value can be inferred from output count alone.
- **Sensitivity:** Replacing a lost $100 fact with a duplicated $40 fact changes currency error while coverage/multiplicity still diagnose structure.
- **Failure boundary:** If the business rule permits multiple memberships, multiplicity above one is expected but additive revenue still requires allocation or a non-additive label.
- **Verification protocol:** Execute a reference query over the three facts, compare key sets and totals, then repeat after the perturbation with the same cutoff.
- **Evidence boundary:** This teaches relational behavior; it makes no claim about a vendor optimizer, deployed database, or source-system result.

#### TRACE-09 — currency conversion version

- **Input grain:** Three immutable facts A=$40, B=$60, C=$100 at one UTC cutoff; input fact-key set is {A,B,C}.
- **Relationship perturbation:** Apply the named condition to A, B, or C while retaining the original fact payload as the independent oracle.
- **Expected coverage calculation:** Count distinct input keys with at least one valid match, then divide by 3; record unmatched keys explicitly.
- **Expected multiplicity calculation:** Divide materialized joined rows by distinct matched fact keys; do not use total input rows as this denominator.
- **Conservation check:** Sum source facts once: $40+$60+$100=$200. Any allocated descendant total must equal $200 before rounding residual policy.
- **Counterexample condition:** A row count of three can still be wrong: duplicate A, retain B, and lose C produces three rows but only two matched fact keys and $140 naive revenue.
- **Selection consequence:** Use association count, explicit allocation, a temporal selector, or a hold depending on the stated relationship semantics.
- **Independent arithmetic:** Coverage is 2/3=0.666666…; multiplicity is 3/2=1.5; neither value can be inferred from output count alone.
- **Sensitivity:** Replacing a lost $100 fact with a duplicated $40 fact changes currency error while coverage/multiplicity still diagnose structure.
- **Failure boundary:** If the business rule permits multiple memberships, multiplicity above one is expected but additive revenue still requires allocation or a non-additive label.
- **Verification protocol:** Execute a reference query over the three facts, compare key sets and totals, then repeat after the perturbation with the same cutoff.
- **Evidence boundary:** This teaches relational behavior; it makes no claim about a vendor optimizer, deployed database, or source-system result.

#### TRACE-10 — parent hierarchy cycle

- **Input grain:** Three immutable facts A=$40, B=$60, C=$100 at one UTC cutoff; input fact-key set is {A,B,C}.
- **Relationship perturbation:** Apply the named condition to A, B, or C while retaining the original fact payload as the independent oracle.
- **Expected coverage calculation:** Count distinct input keys with at least one valid match, then divide by 3; record unmatched keys explicitly.
- **Expected multiplicity calculation:** Divide materialized joined rows by distinct matched fact keys; do not use total input rows as this denominator.
- **Conservation check:** Sum source facts once: $40+$60+$100=$200. Any allocated descendant total must equal $200 before rounding residual policy.
- **Counterexample condition:** A row count of three can still be wrong: duplicate A, retain B, and lose C produces three rows but only two matched fact keys and $140 naive revenue.
- **Selection consequence:** Use association count, explicit allocation, a temporal selector, or a hold depending on the stated relationship semantics.
- **Independent arithmetic:** Coverage is 2/3=0.666666…; multiplicity is 3/2=1.5; neither value can be inferred from output count alone.
- **Sensitivity:** Replacing a lost $100 fact with a duplicated $40 fact changes currency error while coverage/multiplicity still diagnose structure.
- **Failure boundary:** If the business rule permits multiple memberships, multiplicity above one is expected but additive revenue still requires allocation or a non-additive label.
- **Verification protocol:** Execute a reference query over the three facts, compare key sets and totals, then repeat after the perturbation with the same cutoff.
- **Evidence boundary:** This teaches relational behavior; it makes no claim about a vendor optimizer, deployed database, or source-system result.

#### TRACE-11 — aggregate-to-detail join

- **Input grain:** Three immutable facts A=$40, B=$60, C=$100 at one UTC cutoff; input fact-key set is {A,B,C}.
- **Relationship perturbation:** Apply the named condition to A, B, or C while retaining the original fact payload as the independent oracle.
- **Expected coverage calculation:** Count distinct input keys with at least one valid match, then divide by 3; record unmatched keys explicitly.
- **Expected multiplicity calculation:** Divide materialized joined rows by distinct matched fact keys; do not use total input rows as this denominator.
- **Conservation check:** Sum source facts once: $40+$60+$100=$200. Any allocated descendant total must equal $200 before rounding residual policy.
- **Counterexample condition:** A row count of three can still be wrong: duplicate A, retain B, and lose C produces three rows but only two matched fact keys and $140 naive revenue.
- **Selection consequence:** Use association count, explicit allocation, a temporal selector, or a hold depending on the stated relationship semantics.
- **Independent arithmetic:** Coverage is 2/3=0.666666…; multiplicity is 3/2=1.5; neither value can be inferred from output count alone.
- **Sensitivity:** Replacing a lost $100 fact with a duplicated $40 fact changes currency error while coverage/multiplicity still diagnose structure.
- **Failure boundary:** If the business rule permits multiple memberships, multiplicity above one is expected but additive revenue still requires allocation or a non-additive label.
- **Verification protocol:** Execute a reference query over the three facts, compare key sets and totals, then repeat after the perturbation with the same cutoff.
- **Evidence boundary:** This teaches relational behavior; it makes no claim about a vendor optimizer, deployed database, or source-system result.

#### TRACE-12 — authorization filter

- **Input grain:** Three immutable facts A=$40, B=$60, C=$100 at one UTC cutoff; input fact-key set is {A,B,C}.
- **Relationship perturbation:** Apply the named condition to A, B, or C while retaining the original fact payload as the independent oracle.
- **Expected coverage calculation:** Count distinct input keys with at least one valid match, then divide by 3; record unmatched keys explicitly.
- **Expected multiplicity calculation:** Divide materialized joined rows by distinct matched fact keys; do not use total input rows as this denominator.
- **Conservation check:** Sum source facts once: $40+$60+$100=$200. Any allocated descendant total must equal $200 before rounding residual policy.
- **Counterexample condition:** A row count of three can still be wrong: duplicate A, retain B, and lose C produces three rows but only two matched fact keys and $140 naive revenue.
- **Selection consequence:** Use association count, explicit allocation, a temporal selector, or a hold depending on the stated relationship semantics.
- **Independent arithmetic:** Coverage is 2/3=0.666666…; multiplicity is 3/2=1.5; neither value can be inferred from output count alone.
- **Sensitivity:** Replacing a lost $100 fact with a duplicated $40 fact changes currency error while coverage/multiplicity still diagnose structure.
- **Failure boundary:** If the business rule permits multiple memberships, multiplicity above one is expected but additive revenue still requires allocation or a non-additive label.
- **Verification protocol:** Execute a reference query over the three facts, compare key sets and totals, then repeat after the perturbation with the same cutoff.
- **Evidence boundary:** This teaches relational behavior; it makes no claim about a vendor optimizer, deployed database, or source-system result.

#### TRACE-13 — rounding allocation

- **Input grain:** Three immutable facts A=$40, B=$60, C=$100 at one UTC cutoff; input fact-key set is {A,B,C}.
- **Relationship perturbation:** Apply the named condition to A, B, or C while retaining the original fact payload as the independent oracle.
- **Expected coverage calculation:** Count distinct input keys with at least one valid match, then divide by 3; record unmatched keys explicitly.
- **Expected multiplicity calculation:** Divide materialized joined rows by distinct matched fact keys; do not use total input rows as this denominator.
- **Conservation check:** Sum source facts once: $40+$60+$100=$200. Any allocated descendant total must equal $200 before rounding residual policy.
- **Counterexample condition:** A row count of three can still be wrong: duplicate A, retain B, and lose C produces three rows but only two matched fact keys and $140 naive revenue.
- **Selection consequence:** Use association count, explicit allocation, a temporal selector, or a hold depending on the stated relationship semantics.
- **Independent arithmetic:** Coverage is 2/3=0.666666…; multiplicity is 3/2=1.5; neither value can be inferred from output count alone.
- **Sensitivity:** Replacing a lost $100 fact with a duplicated $40 fact changes currency error while coverage/multiplicity still diagnose structure.
- **Failure boundary:** If the business rule permits multiple memberships, multiplicity above one is expected but additive revenue still requires allocation or a non-additive label.
- **Verification protocol:** Execute a reference query over the three facts, compare key sets and totals, then repeat after the perturbation with the same cutoff.
- **Evidence boundary:** This teaches relational behavior; it makes no claim about a vendor optimizer, deployed database, or source-system result.

#### TRACE-14 — hot customer key

- **Input grain:** Three immutable facts A=$40, B=$60, C=$100 at one UTC cutoff; input fact-key set is {A,B,C}.
- **Relationship perturbation:** Apply the named condition to A, B, or C while retaining the original fact payload as the independent oracle.
- **Expected coverage calculation:** Count distinct input keys with at least one valid match, then divide by 3; record unmatched keys explicitly.
- **Expected multiplicity calculation:** Divide materialized joined rows by distinct matched fact keys; do not use total input rows as this denominator.
- **Conservation check:** Sum source facts once: $40+$60+$100=$200. Any allocated descendant total must equal $200 before rounding residual policy.
- **Counterexample condition:** A row count of three can still be wrong: duplicate A, retain B, and lose C produces three rows but only two matched fact keys and $140 naive revenue.
- **Selection consequence:** Use association count, explicit allocation, a temporal selector, or a hold depending on the stated relationship semantics.
- **Independent arithmetic:** Coverage is 2/3=0.666666…; multiplicity is 3/2=1.5; neither value can be inferred from output count alone.
- **Sensitivity:** Replacing a lost $100 fact with a duplicated $40 fact changes currency error while coverage/multiplicity still diagnose structure.
- **Failure boundary:** If the business rule permits multiple memberships, multiplicity above one is expected but additive revenue still requires allocation or a non-additive label.
- **Verification protocol:** Execute a reference query over the three facts, compare key sets and totals, then repeat after the perturbation with the same cutoff.
- **Evidence boundary:** This teaches relational behavior; it makes no claim about a vendor optimizer, deployed database, or source-system result.

#### TRACE-15 — stale snapshot

- **Input grain:** Three immutable facts A=$40, B=$60, C=$100 at one UTC cutoff; input fact-key set is {A,B,C}.
- **Relationship perturbation:** Apply the named condition to A, B, or C while retaining the original fact payload as the independent oracle.
- **Expected coverage calculation:** Count distinct input keys with at least one valid match, then divide by 3; record unmatched keys explicitly.
- **Expected multiplicity calculation:** Divide materialized joined rows by distinct matched fact keys; do not use total input rows as this denominator.
- **Conservation check:** Sum source facts once: $40+$60+$100=$200. Any allocated descendant total must equal $200 before rounding residual policy.
- **Counterexample condition:** A row count of three can still be wrong: duplicate A, retain B, and lose C produces three rows but only two matched fact keys and $140 naive revenue.
- **Selection consequence:** Use association count, explicit allocation, a temporal selector, or a hold depending on the stated relationship semantics.
- **Independent arithmetic:** Coverage is 2/3=0.666666…; multiplicity is 3/2=1.5; neither value can be inferred from output count alone.
- **Sensitivity:** Replacing a lost $100 fact with a duplicated $40 fact changes currency error while coverage/multiplicity still diagnose structure.
- **Failure boundary:** If the business rule permits multiple memberships, multiplicity above one is expected but additive revenue still requires allocation or a non-additive label.
- **Verification protocol:** Execute a reference query over the three facts, compare key sets and totals, then repeat after the perturbation with the same cutoff.
- **Evidence boundary:** This teaches relational behavior; it makes no claim about a vendor optimizer, deployed database, or source-system result.

## 10. Key Theorems / Results Reference

Functional dependencies express when a key determines an attribute; normalization uses such dependencies to reduce update anomalies, but a normalized model still needs temporal and business semantics. Additive measures can sum across dimensions only when the fact grain and join path preserve one contribution per fact. Semi-additive balances need a time rule; inventory snapshots cannot be summed over days like sales.

**Takeaway:** Additivity is a property of a measure along a dimension, not a property of its numeric type.

## 11. Interview / Exam Questions

1. **What is grain?** The exact business statement represented by one fact row, including key and time/mutation rule.
2. **Why can a one-to-many join inflate revenue?** It repeats an additive fact once per child; measure both input-fact coverage and multiplicity before aggregating.
3. **When is `DISTINCT` appropriate?** Only when the declared question is set membership and selected columns preserve identity; it is not a generic fan-out repair.
4. **What does a bridge require?** A relationship key, effective time if needed, and an explicit allocation or association interpretation.
5. **How do you test type-2 history?** Assert no overlapping validity windows for one natural key and exactly one match for an as-of fact.
6. **Normalized or star?** Normalize mutable operational ownership; publish a star-like mart for stable analytical questions; use both when their boundaries are explicit.
7. **What is a semi-additive measure?** A measure additive across some dimensions but not time, such as end-of-day balance.
8. **A metric dropped after adding a dimension: what first?** Compare pre/post fact-key coverage and anti-join exclusions before interpreting business movement.
9. **Changed constraint: strict latency.** Precompute a governed mart, but retain lineage and refresh watermark; do not duplicate unversioned business logic.
10. **Plausible wrong method?** Grouping by customer after joining a multi-tag bridge and treating the result as customer revenue without allocation.

**Takeaway:** Defensible modeling answers specify the fact key, relationship cardinality, and reconciliation oracle.

## 12. Quick Reference Card

| Item | Rule | Use | Critical assumption | Avoid |
|---|---|---|---|---|
| Fact grain | write one business statement | before SQL | key/time rule is explicit | “one row per customer” ambiguity |
| Additive total | sum once per fact key | revenue/events | join preserves facts | uncontrolled bridge |
| Coverage and multiplicity | `distinct matched/input facts`; `joined/distinct matched` | diagnose loss and fan-out | same cutoff and input population | raw row count as a correctness test |
| Bridge | edges + allocation policy | many-to-many | policy is approved | full duplication called total revenue |
| Type-2 join | one version at fact time | historical attributes | intervals do not overlap | current row used silently |
| Reconciliation | compare pre/post fact totals | any new join | same population | offsets ignored |
