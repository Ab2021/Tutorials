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

Validity before use: `I` is the set of input fact keys at one declared snapshot; `M` is the subset with at least one valid relationship match; `J` is the materialized joined-row count. These controls diagnose structure only when the same predicate and cutoff define all sets.

```text
FORMULA [GM-02]: c_join = |M| / |I|
WHERE: c_join input-fact coverage [dimensionless]; I [set of fact keys];
       M [set of distinct input fact keys with one or more valid matches].
CONSTRAINT: |I|>0; facts, relationship, and predicate use the same snapshot.
NUMERICAL EXAMPLE: I={A,B,C}; M={A,B}; c_join=2/3=0.666666....
INTERPRETATION: one third of input facts has no matched relationship.
```

```text
FORMULA [GM-03]: m_join = J / |M|
WHERE: m_join matched-fact multiplicity [joined rows/matched fact]; J [rows].
CONSTRAINT: |M|>0; output rows must retain fact-key identity for diagnosis.
NUMERICAL EXAMPLE: J=3 rows for matched keys {A,B}; m_join=3/2=1.5.
INTERPRETATION: matched facts create 1.5 rows on average; this says nothing about C.
```

Derivation: coverage is the fraction of the input key set surviving a required relationship. Multiplicity is the mean number of materialized rows conditional on survival. Their denominators deliberately differ, so neither substitutes for the other. Boundary: an optional relationship can have coverage below one by contract; a valid bridge can have multiplicity above one, but neither permits summing an additive fact without an allocation policy.

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

All investigations below are ILLUSTRATIVE. The release oracle is the declared fact-key set at one cutoff; it is not a claim about a production incident.

### DIAG-01 — missing dimension versus intended exclusion
- **Symptom:** Revenue falls after adding `JOIN dim_customer`.
- **Competing causes:** an absent customer key, or a published rule excluding anonymous orders.
- **Discriminating check:** left join the fact relation, group unmatched keys by `order_status`, and compare that set with the approved exclusion predicate.
- **Expected signal:** unmatched completed orders are a referential defect; only anonymous orders disappearing under an explicit predicate is an intended population change.
- **Bounded fix:** hold the customer slice and retain the pre-join total; assign an `UNKNOWN` member only if the metric contract permits it.
- **Verify:** input fact coverage is 1 for required completed orders, and the anti-join is empty at the certified cutoff.

### DIAG-02 — type-2 overlap versus a late correction
- **Symptom:** one order line receives two customer regions.
- **Competing causes:** overlapping effective intervals, or a correction event that should supersede an earlier dimension version.
- **Discriminating check:** count dimension rows satisfying `effective_from <= event_ts < effective_to` per fact key; separately inspect source sequence/version.
- **Expected signal:** count two is overlap; a single latest sequence selected by a documented tie-breaker is a correction.
- **Bounded fix:** quarantine the affected natural key and repair intervals before reprocessing; never choose `MAX(region)` as a temporal selector.
- **Verify:** each fact has exactly one as-of dimension match and rerunning the same cutoff is stable.

### DIAG-03 — bridge attribution versus association reporting
- **Symptom:** tag-level revenue exceeds company revenue.
- **Competing causes:** full-credit association semantics being mistaken for allocated revenue, or weights no longer summing to one.
- **Discriminating check:** calculate `sum(weight)` by fact key and compare the report label with the requested business question.
- **Expected signal:** full credit is legitimate only for a non-additive “associated revenue” measure; allocated currency requires one total weight per fact.
- **Bounded fix:** relabel the association metric or allocate with a versioned, nonnegative weighting policy.
- **Verify:** allocated total equals $200 in the oracle and no weight sum differs from 1 within declared rounding tolerance.

### DIAG-04 — tenant key collision
- **Symptom:** a customer attribute appears to belong to another tenant.
- **Competing causes:** a join omitted tenant namespace, or a source genuinely consolidated tenants under a new documented identity.
- **Discriminating check:** join once on `(tenant_id, customer_id)` and once on `customer_id`; compare match counts and nonmatching tenant pairs.
- **Expected signal:** additional cross-tenant matches prove an incomplete key; a consolidation needs an explicit mapping effective date.
- **Bounded fix:** restore the composite key and revoke the affected output; do not deduplicate customer IDs across tenants.
- **Verify:** zero joined rows have unequal tenant IDs and the scoped fact total reconciles independently.

### DIAG-05 — outer join null masking
- **Symptom:** a dashboard shows a large `Unknown` region without a row-count drop.
- **Competing causes:** legitimate unmapped new geography, or a left join preserving facts while a downstream `WHERE d.region <> 'X'` turns it into an inner join.
- **Discriminating check:** compare null region counts before and after every downstream predicate and inspect predicate placement.
- **Expected signal:** a null spike at join output is a mapping issue; a drop after filtering is query semantics.
- **Bounded fix:** state whether unknown is included, excluded, or separately reported; place optional-dimension filters in the join condition when preserving facts is required.
- **Verify:** coverage stays 1 and the unknown-member count equals the left anti-join count.

### DIAG-06 — source duplicate versus semantic duplicate
- **Symptom:** `COUNT(DISTINCT order_id)` is stable while amount rises.
- **Competing causes:** retried identical events, or two revisions with the same order ID and different line/version semantics.
- **Discriminating check:** compare payload hashes, version sequence, and the contract key `(order_id,line_id,version)` rather than order ID alone.
- **Expected signal:** equal hash and version is a retry; changed version requires a latest-snapshot or append-history rule.
- **Bounded fix:** deduplicate only identical retry identity; model revisions explicitly and select at the consumer's stated as-of time.
- **Verify:** source amount at selected versions equals consumer amount, and history remains replayable.

### DIAG-07 — compensating loss and fan-out
- **Symptom:** output row count equals input row count and total is plausible.
- **Competing causes:** no defect, or one fact lost while another duplicates.
- **Discriminating check:** calculate coverage over all input fact keys and multiplicity over matched keys; inspect both key sets.
- **Expected signal:** the oracle with A duplicated, B retained, C lost has coverage `2/3` and multiplicity `3/2` despite three output rows.
- **Bounded fix:** block additive aggregation until the lost and duplicated paths are repaired independently.
- **Verify:** coverage and multiplicity are both 1 for a one-to-one required relationship; output count alone is never the oracle.

### DIAG-08 — snapshot cutoff drift
- **Symptom:** the same report changes without a source correction.
- **Competing causes:** a late fact was accepted into a later snapshot, or dimensions and facts were queried at different cutoffs.
- **Discriminating check:** record fact snapshot ID, dimension snapshot ID, event-time cutoff, and processing-time cutoff with every aggregate.
- **Expected signal:** a changed snapshot ID or mixed cutoffs explains nonreproducibility; identical immutable snapshots should replay identically.
- **Bounded fix:** pin both relations to a compatible snapshot or publish a revisioned report; do not describe “current” as a reproducible cutoff.
- **Verify:** a second execution against the recorded snapshots returns the same fact-key set and amount.

**Takeaway:** A join diagnosis is complete only when it distinguishes a competing cause, bounds the repair, and supplies a replayable verification.

## 9. Complexity & Cost Analysis

A hash join that materializes `J` output rows has expected `O(F + D + J)` time for fact rows `F` and build/probe relation rows `D`, assuming hash operations are expected constant time and the chosen build side fits available memory. Build-side memory is `O(D)` for its keys, payload, and hash-table overhead; if it does not fit, partitioning or spill adds I/O and can dominate elapsed time. A bridge increases `J` with relationship edges, not merely facts. Illustrative sizing: 10 million facts with an average 1.4 bridge edges produce 14 million joined rows; the 40% increase affects materialization, shuffle, and downstream aggregation. This is a derived workload estimate, not a benchmark.

**Takeaway:** Cardinality is a cost input as well as a correctness input.


### Executable SQLite oracle: loss and fan-out are independent

Run this ILLUSTRATIVE script as one SQLite unit. The expected results below are derived inputs/outputs, not a claimed execution.

```sql
CREATE TABLE fact_line (
  line_id TEXT PRIMARY KEY,
  amount_cents INTEGER NOT NULL,
  customer_id TEXT NOT NULL
);
INSERT INTO fact_line VALUES
  ('A',4000,'c1'), ('B',6000,'c2'), ('C',10000,'c3');
CREATE TABLE dim_customer (customer_id TEXT NOT NULL, region TEXT NOT NULL);
INSERT INTO dim_customer VALUES
  ('c1','north'), ('c1','north-duplicate'), ('c2','south');

WITH joined AS (
  SELECT f.line_id, f.amount_cents
  FROM fact_line AS f JOIN dim_customer AS d ON d.customer_id=f.customer_id
), matched AS (SELECT DISTINCT line_id FROM joined)
SELECT
  (SELECT COUNT(*) FROM fact_line) AS input_facts,
  (SELECT COUNT(*) FROM matched) AS matched_facts,
  (SELECT COUNT(*) FROM joined) AS joined_rows,
  (SELECT SUM(amount_cents) FROM fact_line) AS source_cents,
  (SELECT SUM(amount_cents) FROM joined) AS naive_cents,
  1.0*(SELECT COUNT(*) FROM matched)/(SELECT COUNT(*) FROM fact_line) AS coverage,
  1.0*(SELECT COUNT(*) FROM joined)/(SELECT COUNT(*) FROM matched) AS multiplicity;
```

Expected output is `3, 2, 3, 20000, 14000, 0.666666..., 1.5`. C is lost and A repeats, even though output row count equals input row count. $60 of naive undercount is the net of losing C ($100) and duplicating A ($40). These diagnostics have distinct denominators: coverage has every input fact in its denominator; multiplicity has only matched facts.

This query names the bad keys:

```sql
WITH joined AS (
  SELECT f.line_id
  FROM fact_line f JOIN dim_customer d ON d.customer_id=f.customer_id
)
SELECT 'unmatched' AS kind, f.line_id
FROM fact_line f LEFT JOIN joined j ON j.line_id=f.line_id
WHERE j.line_id IS NULL
UNION ALL
SELECT 'duplicated', line_id FROM joined GROUP BY line_id HAVING COUNT(*)>1;
```

Expected rows are `unmatched | C` and `duplicated | A`. `DISTINCT` cannot restore C; it is not a repair for a missing relationship.

### Executable SQLite oracle: bridge allocation conservation

```sql
CREATE TABLE line_tag_weight (
  line_id TEXT NOT NULL, tag TEXT NOT NULL, weight REAL NOT NULL CHECK(weight>=0),
  PRIMARY KEY(line_id,tag)
);
INSERT INTO line_tag_weight VALUES
  ('A','red',.5), ('A','sale',.5), ('B','sale',1), ('C','new',1);

SELECT line_id, SUM(weight) AS weight_sum
FROM line_tag_weight GROUP BY line_id
HAVING ABS(SUM(weight)-1.0)>0.0000001;

SELECT tag, SUM(f.amount_cents*t.weight) AS allocated_cents
FROM fact_line f JOIN line_tag_weight t ON t.line_id=f.line_id
GROUP BY tag ORDER BY tag;

SELECT SUM(f.amount_cents*t.weight) AS allocated_total_cents
FROM fact_line f JOIN line_tag_weight t ON t.line_id=f.line_id;
```

The first query must return no rows. Expected tag amounts are `new=10000`, `red=2000`, and `sale=8000` cents; the final result is 20,000 cents. If A's two weights are `.7`, `.7`, the weight assertion reports 1.4 and allocated total becomes 21,600 cents. That is a bridge policy failure, not a numerical formatting issue. Full-credit tag association may still be useful, but must be called association revenue and never reconciled as additive company revenue.

### Executable SQLite oracle: effective-time uniqueness

```sql
CREATE TABLE dim_region_history (
  customer_id TEXT, effective_from TEXT, effective_to TEXT, region TEXT,
  CHECK(effective_from < effective_to)
);
CREATE TABLE fact_at_time (
  line_id TEXT PRIMARY KEY, customer_id TEXT, event_ts TEXT, amount_cents INTEGER
);
INSERT INTO dim_region_history VALUES
 ('c1','2026-01-01T00:00:00Z','2026-02-01T00:00:00Z','north'),
 ('c1','2026-02-01T00:00:00Z','9999-12-31T00:00:00Z','east');
INSERT INTO fact_at_time VALUES
 ('H1','c1','2026-01-31T23:00:00Z',4000),
 ('H2','c1','2026-02-01T00:00:00Z',5000);

SELECT f.line_id, d.region
FROM fact_at_time f JOIN dim_region_history d
 ON d.customer_id=f.customer_id
AND d.effective_from<=f.event_ts AND f.event_ts<d.effective_to
ORDER BY f.line_id;

SELECT f.line_id, COUNT(*) AS matching_versions
FROM fact_at_time f LEFT JOIN dim_region_history d
 ON d.customer_id=f.customer_id
AND d.effective_from<=f.event_ts AND f.event_ts<d.effective_to
GROUP BY f.line_id HAVING COUNT(*)<>1;
```

Expected assignments are `H1 | north` and `H2 | east`; the assertion produces no rows. The half-open interval prevents both versions matching at the boundary. Changing the second start to H1's timestamp produces two matches; ending the first interval a second early produces zero. Both invalidate an as-of additive query.

### Release decision

For a required one-to-one dimension, release only when fact coverage and matched multiplicity are both one at the named snapshot. For an optional dimension, coverage may be below one only if the unknown-member policy and metric population explicitly allow it. For a bridge, multiplicity above one is normal but additive currency must be allocated and conserved. A hash join that materializes `J` output rows retains expected `O(F + D + J)` work; the bridge changes `J`, so its correctness policy is also capacity policy.

**Takeaway:** These oracle inputs separately test lost facts, repeated facts, allocation conservation, and temporal uniqueness; a matching row count tests none of them completely.

## 10. Key Theorems / Results Reference

Functional dependencies express when a key determines an attribute; normalization uses such dependencies to reduce update anomalies, but a normalized model still needs temporal and business semantics. Additive measures can sum across dimensions only when the fact grain and join path preserve one contribution per fact. Semi-additive balances need a time rule; inventory snapshots cannot be summed over days like sales.

**Takeaway:** Additivity is a property of a measure along a dimension, not a property of its numeric type.

## 11. Interview / Exam Questions

1. **What is grain?**  
   **Answer:** The complete business statement of one row: identity tuple, represented fact, time basis, and mutation rule.  
   **Reasoning:** “One row per order” is insufficient if revisions and lines coexist.  
   **Follow-up:** State whether cancelled versions remain history.  
   **Acceptable alternative:** An event grain is valid when event identity and deduplication are explicit.  
   **Red flag / scoring:** “Primary key” alone scores 1; key plus time/version and aggregation consequence scores 4.

2. **Why can a one-to-many join inflate revenue?**  
   **Answer:** It repeats a fact once for each matching child, so the additive amount contributes multiple times.  
   **Reasoning:** A line with two tags produces two joined rows but remains one sale.  
   **Follow-up:** Compute coverage and multiplicity for A, B, C in the oracle.  
   **Acceptable alternative:** Pre-aggregate only when the grouping remains at fact grain.  
   **Red flag / scoring:** Universal `DISTINCT` scores 0; bridge allocation plus reconciliation scores 4.

3. **Why are coverage and multiplicity not interchangeable?**  
   **Answer:** Coverage exposes lost input keys; multiplicity exposes repeated matched keys.  
   **Reasoning:** The oracle has coverage `2/3` and multiplicity `3/2` while three rows emerge.  
   **Follow-up:** What does `m=1` fail to prove?  
   **Acceptable alternative:** Report unmatched-key count and per-key duplication distribution alongside ratios.  
   **Red flag / scoring:** Using output/input row count scores 1; naming both denominators scores 4.

4. **When is `DISTINCT` appropriate?**  
   **Answer:** For a declared set-membership question whose selected columns preserve the intended entity identity.  
   **Reasoning:** It can remove duplicate projections but cannot infer a missing fact or allocation policy.  
   **Follow-up:** Show why it cannot restore C in the oracle.  
   **Acceptable alternative:** Deduplicate exact retries using producer identity and payload/version rules.  
   **Red flag / scoring:** Calling it a join repair scores 0; stating its grain boundary scores 4.

5. **What must a bridge contract contain?**  
   **Answer:** Both endpoint keys, relationship effective time where relevant, allocation meaning, nonnegative weights, and a conservation assertion.  
   **Reasoning:** Many memberships are not automatically many additive contributions.  
   **Follow-up:** If weights sum to 1.4, what is released?  
   **Acceptable alternative:** Publish non-additive association counts rather than invented weights.  
   **Red flag / scoring:** Full credit labeled total revenue scores 0; policy/version/verification scores 4.

6. **How do you test type-2 history?**  
   **Answer:** Assert no overlapping intervals per natural key and exactly one as-of match for every required fact.  
   **Reasoning:** Half-open intervals assign a boundary instant to one version.  
   **Follow-up:** Diagnose zero matches versus two matches.  
   **Acceptable alternative:** A source sequence selector is valid only if its correction semantics are documented.  
   **Red flag / scoring:** `MAX(region)` scores 0; temporal key-set oracle scores 4.

7. **Normalized model or star mart?**  
   **Answer:** Normalize mutable ownership and integrity rules; derive a dimensional mart for repeated, stable analytical questions.  
   **Reasoning:** They optimize different workloads and do not eliminate each other’s contracts.  
   **Follow-up:** What changes if customer history is needed?  
   **Acceptable alternative:** A governed wide table can be safe when it declares refresh, grain, and lineage.  
   **Red flag / scoring:** “Star is faster” scores 1; workload/mutation/reconciliation tradeoff scores 4.

8. **What is a semi-additive measure?**  
   **Answer:** A value additive over some dimensions but not over time, such as end-of-day inventory balance.  
   **Reasoning:** Summing daily balances invents inventory-days, not a balance.  
   **Follow-up:** Choose last snapshot versus average balance for a stated decision.  
   **Acceptable alternative:** Integrate a balance over time only when the resulting unit is explicitly balance-time.  
   **Red flag / scoring:** Treating every numeric field as additive scores 0; units and time rule score 4.

9. **A metric drops after a dimension join. What first?**  
   **Answer:** Freeze interpretation; compare fact-key coverage, anti-join keys, predicates, cutoff, and source total.  
   **Reasoning:** Business movement and accidental exclusion yield the same chart shape.  
   **Follow-up:** How does an optional dimension change the decision?  
   **Acceptable alternative:** Publish a labeled prior certified value while repairing.  
   **Red flag / scoring:** Explaining a business cause before reconciliation scores 0; bounded release decision scores 4.

10. **A strict latency limit is introduced. What changes?**  
    **Answer:** Precompute a governed mart or cached aggregate with snapshot and freshness contracts; do not copy unversioned logic.  
    **Reasoning:** Lower latency cannot waive relationship correctness.  
    **Follow-up:** What capacity term grows with a bridge?  
    **Acceptable alternative:** Defer an expensive optional slice while retaining certified core totals.  
    **Red flag / scoring:** “Add indexes” alone scores 1; cutoff/lineage/cost boundary scores 4.

11. **What is a plausible wrong method?**  
    **Answer:** Group by customer after a multi-tag bridge and call the duplicated result customer revenue.  
    **Reasoning:** Aggregating later cannot undo an undefined attribution policy.  
    **Follow-up:** Which report could legitimately use full credit?  
    **Acceptable alternative:** Count distinct associated order lines if the question is reach.  
    **Red flag / scoring:** Unlabeled “revenue by tag” scores 0; measure label and conservation check score 4.

12. **How do you defend a release?**  
    **Answer:** Name fact grain, snapshot, input population, cardinality contract, coverage, multiplicity, total reconciliation, and known limitation.  
    **Reasoning:** A query result without its input contract is not auditable.  
    **Follow-up:** What evidence would make you block the release?  
    **Acceptable alternative:** A documented exception can release an unknown slice if the decision owner accepts it.  
    **Red flag / scoring:** “It ran” scores 0; replayable oracle and authority boundary score 4.

**Takeaway:** Defensible modeling answers state the fact key, cardinality, cutoff, diagnostic denominator, and decision consequence.

## 12. Quick Reference Card

| Item | Rule | Use | Critical assumption | Avoid |
|---|---|---|---|---|
| Fact grain | write one business statement | before SQL | key/time rule is explicit | “one row per customer” ambiguity |
| Additive total | sum once per fact key | revenue/events | join preserves facts | uncontrolled bridge |
| Coverage and multiplicity | `distinct matched/input facts`; `joined/distinct matched` | diagnose loss and fan-out | same cutoff and input population | raw row count as a correctness test |
| Bridge | edges + allocation policy | many-to-many | policy is approved | full duplication called total revenue |
| Type-2 join | one version at fact time | historical attributes | intervals do not overlap | current row used silently |
| Reconciliation | compare pre/post fact totals | any new join | same population | offsets ignored |
