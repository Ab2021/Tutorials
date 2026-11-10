---
collection: "data-modeling-databases-lakehouse-and-governance"
file_id: "02"
topic: "Data model grain and join safety"
level: 2
status: "draft"
version: "2.0"
created: "2026-11-09"
reviewed: null
review_trigger: "A business key, relationship cardinality, snapshot rule, or allocation policy changes."
notation_style: "plain-text"
prerequisites: ["01_Data_Contract_Checklist.md"]
pattern_tags: ["grain", "cardinality", "fan-out", "allocation", "as-of-join"]
source_status: "OPEN"
owner_role: "Analytics engineer"
source_index: "INDEX.md"
---

# Data Model Grain and Join Safety

> **Purpose:** Decide what one row of a model asserts, then prove that every join on the path to a reported measure preserves that assertion — no fact lost, no fact multiplied, no dimension attribute silently chosen.
> **Prerequisites:** [Data contracts](01_Data_Contract_Checklist.md), relational keys, grouped aggregation, and the meaning of an inner versus outer join.
> **Key outcome:** Given an entity-relationship sketch and a metric definition, you can declare the fact grain, compute the expected join cardinality in advance, predict the sign and size of the aggregate error, and choose a safe alternative — a semi-join, a pre-aggregation, an allocated bridge, or an as-of version join.
> **This file does NOT cover:** physical index design and query-plan tuning (see the notation sheet for schema vocabulary), vendor-specific SQL dialect quirks, distributed-engine configuration, or organization-specific master-data policy.
> **Level:** 2 — Core methods.
> **Version / reviewed:** 2.0 / not independently reviewed.
> **Evidence:** DERIVED relational reasoning; VERIFIED illustrative arithmetic on a six-row constructed dataset; no accessible source body is reconstructed, so every empirical claim about production systems is OPEN. The SQLite listings are executable specifications of the invariants, not reported runs against a real warehouse.
> **Retrieval cue:** If a number changed immediately after a join was added, start at Section 8 and work backwards through Section 4.

## 1. File Header

A data model is a set of falsifiable statements about entities, their identity, their relationships, and the time over which each statement holds. It is not a diagram, and it is not a schema. Two models can produce byte-identical DDL and disagree completely about what a row means, because the meaning lives in the key definitions, the cardinality rules, and the temporal predicate. Everything in this sheet follows from that position: modelling is the act of writing down assertions precisely enough that a query can be checked against them.

The concrete decision this sheet supports appears whenever a metric is defended rather than merely printed. A revenue number moves, and someone must answer whether the movement is business or arithmetic. A customer count differs between two dashboards, and someone must say which denominator is legitimate. A tag-level report exceeds the company total, and someone must explain why that is not necessarily an error. These are not edge cases; they are the normal operating condition of an analytical warehouse, because warehouses are assembled from joins and joins are set operations whose behaviour is governed by cardinality, not by intent.

Four decisions recur, and each has a written artifact that makes it auditable.

The first decision is the **grain declaration**. For a fact table the declaration names the identity tuple, the event-time basis, the mutation rule, and the population predicate. "One row per order line, as of the latest accepted version, excluding cancelled lines" is a grain declaration. "Sales table" is not. The grain declaration is what allows a reader to compute an expected row count independently, which is the only real defence against fan-out.

The second decision is the **model family**: normalized entities, dimensional star, snowflaked dimensions, Data Vault hubs-links-satellites, or a governed wide table. This choice trades update anomaly resistance against query simplicity, and it determines where many-to-many relationships become visible. The choice is not ideological; it is a statement about which mutations are frequent, which questions are repeated, and which team owns which entity.

The third decision is the **join path**, including the cardinality contract for each hop. A join path is a chain of set operations, and each hop carries two independent risks: it can drop input rows (coverage below one) and it can replicate input rows (multiplicity above one). The two risks have different denominators, produce different symptoms, and can cancel numerically, which is why row counts alone never certify a result.

The fourth decision is the **attribution policy** for genuine many-to-many relationships. When a fact legitimately relates to several members, an additive measure must either be allocated with a conserved weighting scheme or reported under a different, explicitly non-additive label. There is no third option in which the numbers are added up unchanged and remain comparable to the total.

The working method is deliberately small. Declare the grain. Enumerate the keys. Write down the cardinality of every hop. Compute coverage and multiplicity. Reconcile to the source of record at a named snapshot. Ship the number with its contract attached. Every section that follows elaborates one step of that method, and every failure mode in Section 8 is a violation of one of those steps.

Reading order for different readers. A beginner should read Sections 2, 3, 5, and 6, then attempt the interview questions in Section 11 that are marked as recall. A practitioner should read Sections 4, 7, and 8 in full and reproduce the arithmetic in Section 5 on paper. An advanced reader should treat Sections 9 and 10 as the load-bearing material and use Section 8 as an audit checklist for a live model.

Notation conventions used throughout. Sets are written with vertical bars for cardinality, so `|F|` is the number of rows in relation `F`. Per-row functions use parentheses, so `m(f)` is the number of dimension rows that fact row `f` matches. The materialized joined relation is `J` when referred to by row count and `F ⋈ D` when referred to as a relation. Currency amounts are in integer cents so that conservation checks are exact; percentages are dimensionless ratios in `[0, 1]` unless stated otherwise. Time instants are ISO-8601 strings in UTC and intervals are half-open, `[from, to)`.

**Takeaway:** A model is only as reliable as its grain declaration, because the grain declaration is the only artifact that lets a second person predict the row count without running the query.

## 2. Concept Map / Mental Model

The map below is the whole sheet compressed to one screen. Read it top to bottom; each box is a decision that must be made before the next box is meaningful.

```text
BUSINESS QUESTION
  "revenue by customer region for completed orders in Q1"
        |
        v
[1] MEASURE + POPULATION + AS-OF TIME
    amount_cents, completed lines, event_ts in Q1
        |
        v
[2] FACT GRAIN DECLARATION
    (order_id, line_id, version) @ event_ts, latest accepted version
        |
        v
[3] KEY INVENTORY
    natural keys, surrogate keys, degenerate dimensions, tenant scope
        |
        v
[4] CARDINALITY CONTRACT PER HOP
    fact N:1 customer, fact N:M tag-bridge, fact N:1 region-as-of
        |
        +--------> contract violated? --> [8] DIAGNOSE (Section 8)
        |
        v
[5] JOIN PATH CONSTRUCTION
    inner (required), left (optional), semi (existence),
    anti (orphan detection), as-of (temporal), allocated (bridge)
        |
        v
[6] AGGREGATE AT DECLARED GRAIN
    sum once per distinct fact key, or allocate explicitly
        |
        v
[7] RECONCILIATION
    c_join = 1?  m_join = 1?  total equals certified source total?
        |
        +--- no ---> hold metric, publish prior certified value, repair
        |
       yes
        v
PUBLISHED MEASURE with grain, snapshot, coverage, multiplicity attached
```

The map encodes a strong claim: correctness is decided at steps 2 through 4, before a single line of SQL is written. Step 5 cannot repair a broken step 4. This is the most common misreading of join safety. Engineers reach for `DISTINCT`, `GROUP BY`, or a `QUALIFY` clause when a number inflates, but those tools operate on the output of a join whose cardinality was already wrong. A `DISTINCT` can collapse duplicated rows; it cannot invent a row that the join discarded, and it cannot decide how much of a shared measurement belongs to each of two tags.

The second structural feature of the map is the split at step 4 into two failure classes that are often conflated.

| Class | What it means | Denominator | Symptom on an additive measure |
|---|---|---|---|
| Loss (coverage) | An input fact found no relationship match, so an inner join removed it | all input facts | total falls, population shrinks |
| Fan-out (multiplicity) | A matched fact found more than one relationship row, so it was replicated | matched input facts only | total rises, population unchanged |

Both classes can occur in the same query, and when they do the aggregate error is the difference of two opposing quantities. That is the mechanism behind the single most misleading signal in analytical work: a plausible total produced by a query that is simultaneously dropping and duplicating facts. Section 5 constructs the six-row case, where one fact is dropped and two are duplicated for a net error of 2,500 cents on a true 35,000; the three-row case in [GM-05] is worse still, because there the output row count equals the input row count exactly while the total is wrong by 6,000 cents on a true 20,000.

The third feature is the loop back from step 7 to step 8. Reconciliation is not a formality performed after publishing; it is the release gate. When reconciliation fails, the correct action is to hold the metric and diagnose, not to explain the movement as a business effect. A business explanation offered before reconciliation is unfalsifiable, because the same chart shape is produced by a broken join and by a genuine demand change.

A complementary mental model is useful for readers who think in terms of area rather than sets. Picture each fact row as a rectangle whose height is its measure. A join that matches the row once leaves the rectangle unchanged. A join that matches it three times replaces the rectangle with three copies, and any sum over the joined relation integrates three times the area. A join that matches nothing deletes the rectangle entirely. Coverage asks "how much area was deleted"; multiplicity asks "how much area was copied". Both questions are about area, and neither can be answered by counting rectangles, because one deletion and one triplication can leave the rectangle count unchanged.

A third mental model is the contract reading. Treat every relationship as a contract with four clauses: the key tuple, the cardinality, the validity interval, and the null policy. A many-to-one relationship says "each child has at most one parent at any instant". A many-to-many relationship says "each child has any number of parents, and the edges carry their own meaning". A type-2 relationship adds "and the parent's attributes are true only within a stated interval". Reading a schema as a bag of tables loses all four clauses; reading it as a set of contracts recovers them, and the contracts are what the diagnostic queries in Section 8 test.

**Takeaway:** Decide grain and cardinality before writing the join; every downstream repair tool operates on a relationship whose contract is already either honoured or violated.

## 3. Foundational Definitions

Definitions are grouped by the job they do. Identity definitions let you say what a row is. Cardinality definitions let you predict what a join does. Temporal definitions let you say when an attribute was true. Attribution definitions let you divide a shared measurement without losing it.

### 3.1 Identity and structure

| Term | Precise meaning | Type / units | Nonexample |
|---|---|---|---|
| Relation | A set of tuples over named attributes with declared domains | schema + bag of rows | a spreadsheet tab with mixed types per column |
| Grain | The complete assertion made by one row: identity tuple, represented fact, time basis, mutation rule | declaration | "one row per customer" when versions and tenants coexist |
| Natural key | Attribute(s) meaningful in the business domain that identify an entity | scalar or tuple | an auto-increment integer that no business process can quote |
| Surrogate key | Warehouse-assigned identifier for a stored row, stable within the warehouse | integer or hash | evidence that two business records denote one entity |
| Composite key | Key formed from two or more attributes jointly | tuple | a concatenated string whose separator can appear inside a component |
| Degenerate dimension | A dimension attribute stored directly on the fact table because it has no attributes of its own | column on fact | a full dimension table for an order number |
| Foreign key | Attribute on the child relation referencing a parent key | column(s) | a free-text name matched by string equality |
| Fact table | Relation whose grain is a measurement or event and whose rows carry measures and keys | table | a list of product descriptions |
| Dimension table | Relation describing an entity, joined to facts by key | table | a table of unrelated code lookups |
| Factless fact table | Fact-grain relation with no additive measure, recording occurrence or coverage | table | a fact table whose only measure is a constant 1 added to a total |
| Bridge table | Relation whose grain is a relationship instance between two entities, possibly weighted | table | a dimension table with a comma-separated list of tags |

The distinction between a natural key and a surrogate key is not cosmetic. A surrogate key answers "which stored row", a natural key answers "which business thing". Join safety depends on the second question, because a surrogate key assigned twice to the same business entity creates fan-out that no constraint will catch, while a natural-key uniqueness assertion catches it immediately. The useful discipline is to assert uniqueness on the natural key first and let the surrogate key follow.

### 3.2 Cardinality and multiplicity

| Term | Precise meaning | Type / units | Nonexample |
|---|---|---|---|
| Cardinality | The maximum number of related rows on each side of a relationship, subject to a stated predicate | 1:1, 1:N, N:1, N:M | a row count observed once on one day |
| Functional dependency | `X -> Y` means every pair of tuples agreeing on `X` agrees on `Y` | logical constraint | a correlation observed in a sample |
| Multiplicity `m(f)` | Number of joined rows produced by one input fact row `f` | rows per fact | the average row width of the output |
| Coverage `c_join` | Fraction of input fact keys with at least one match | dimensionless in `[0,1]` | the fraction of dimension rows that were used |
| Fan-out | The event of a single input row producing more than one output row | property | duplicating the whole table deliberately |
| Chasm | Loss of input rows when an inner join traverses an optional relationship | property | a deliberate population filter |
| Many-to-many | Both sides may have several related rows | cardinality | a one-to-many relationship with duplicate dimension rows |
| Duplicate dimension row | Two physically distinct rows sharing the same natural key at the same instant | defect | a legitimate type-2 version pair |

The vocabulary of traps deserves a warning. Published sources disagree about whether "fan trap" and "chasm trap" name the multiplicity case or the coverage case, and some texts use both words for structurally different schemas. This sheet therefore avoids relying on those words for diagnosis and instead always names the measured quantity: coverage below one, or multiplicity above one. When a colleague says "fan trap", ask which of the two numbers they mean; the question resolves the ambiguity faster than a terminology debate.

### 3.3 Additivity

| Term | Precise meaning | Type / units | Nonexample |
|---|---|---|---|
| Additive measure | Sums correctly across every dimension of the model | currency, count | a balance summed over days |
| Semi-additive measure | Sums across some dimensions but not over time | balance, headcount | a daily inventory value added to yesterday's |
| Non-additive measure | Cannot be summed meaningfully on any dimension | ratio, unit price | an average of averages presented as a total |
| Allocation | Splitting a measure across relationship edges using declared weights | currency per edge | crediting the full amount to every edge |
| Conservation | The property that allocated parts sum to the original measure | assertion | weights summing to 1.4 |
| Weight | Nonnegative share assigned to one relationship edge for one fact key | dimensionless | a rank used as if it were a share |

Additivity is not a property of a column type; it is a property of a measure with respect to a dimension. Revenue is additive across customers, products, and days. A month-end account balance is additive across customers but not across days, because the sum of thirty balances is a quantity measured in balance-days, a unit nobody reports. A conversion rate is additive across nothing, because the correct combination is a ratio of sums rather than a sum of ratios. Confusing these three classes accounts for a large share of the metrics that reconcile on one axis and fail on another.

### 3.4 Time and versioning

| Term | Precise meaning | Type / units | Nonexample |
|---|---|---|---|
| Event time | When the business event occurred | instant, UTC | when the row arrived in the warehouse |
| Processing time | When the warehouse accepted the row | instant, UTC | when the invoice was printed |
| Effective interval | Half-open range `[from, to)` during which a dimension version is true | interval | an inclusive-inclusive range that assigns a boundary instant twice |
| Type-1 change | Overwrite the attribute, discarding prior value | mutation | keeping history by appending |
| Type-2 change | Add a new versioned row with a new validity interval | append | overwriting the old row |
| Snapshot fact | One row per entity per observation instant | grain | one row per transaction |
| Accumulating snapshot | One row per process instance, updated as milestones occur | grain | an immutable event log |
| Late-arriving dimension | A dimension row that appears after facts already reference it | ordering defect | a fact that predates its own event time |
| Inferred member | Placeholder dimension row created to satisfy a fact load before the real attributes arrive | row | fabricating an attribute value to avoid the placeholder |

The single most consequential choice in this cluster is load-time versus query-time key resolution. If the fact table stores the surrogate key that was current when the fact was loaded, later corrections to dimension history cannot retroactively change reported history, which is reproducible and auditable. If the fact stores only the natural key and resolves the version at query time, corrections flow through automatically, which is convenient and can silently rewrite a closed reporting period. Neither is wrong, but the choice must be recorded, because a reader who assumes the other convention will compute a different number from the same data.

### 3.5 Relationship and integrity vocabulary

| Term | Precise meaning | Type / units | Nonexample |
|---|---|---|---|
| Referential integrity | Every non-null child key has a matching parent key at the applicable instant | constraint | an orphan row tolerated because "it is rare" |
| Orphan | Child row whose parent key is absent from the parent relation | row | a fact whose dimension was legitimately deleted under a documented policy |
| Unknown member | A reserved dimension row representing "not identified", with a stable key | row | `NULL` used as the join key |
| Semi-join | Filter that keeps left rows having at least one match, without replicating them | operator | an inner join used only to test existence |
| Anti-join | Filter that keeps left rows having no match | operator | `NOT IN` over a nullable column |
| Conformed dimension | Dimension with identical keys, attributes, and grain across several fact tables | dimension | two tables both named `dim_date` with different fiscal calendars |
| Role-playing dimension | One dimension referenced several times in the same fact table under different meanings | pattern | a single date column reused for order and ship dates |

The unknown member is the design answer to a real tension. A required inner join to a dimension enforces integrity but deletes unmatched facts. A left join preserves facts but produces `NULL` attributes that poison downstream filters. A reserved member with key `-1` and attributes labelled `UNKNOWN` resolves both: the join stays inner, coverage stays at one, and the reporting layer can decide whether to include or exclude the unknown slice without changing the row population. The cost is that the unknown member accumulates facts, so it must be monitored, because its growth rate is a direct measure of dimension-loading lag.

**Takeaway:** Keys identify, cardinality predicts, intervals date, and weights divide — a model that leaves any of the four implicit will produce a number that cannot be defended.

## 4. Core Formulas with Full Derivation

The derivations below start from the definition of the join as a bag operation and build up to the general chained case. Section 4.1 establishes the base identity for a single hop; 4.2 generalises to many hops; 4.3 covers the operator variants that change the counting; 4.4 states where the whole framework stops being valid.

Throughout, let `F` be the input fact relation at its declared grain, `D` a dimension or bridge relation, and `P(f, d)` the join predicate. Let `a(f)` be the measure of fact row `f`, in integer cents. Let `F ⋈_P D` denote the materialised join under bag semantics.

### 4.1 Base case: one hop, one measure

Validity before use: the fact relation has one row per declared fact key, the measure is additive within that grain, and the join predicate is a pure function of the two rows it compares. No aggregation has occurred before the join, and the measure is expressed in a unit where addition is meaningful.

Derivation. Bag semantics defines the join as the multiset of all pairs satisfying the predicate:

```text
F ⋈_P D  =  { (f, d) : f in F, d in D, P(f, d) }
```

counted with multiplicity. Grouping those pairs by the fact row that produced them gives, for each `f`, the match count

```text
m(f)  =  | { d in D : P(f, d) } |
```

and therefore the output cardinality is the sum of the per-row counts:

```text
|F ⋈_P D|  =  sum over f in F of m(f)
```

Summing `a(f)` over the joined rows multiplies each fact's measure by its own match count, because the measure is copied into every pair that contains `f`:

```text
sum over joined rows of a  =  sum over f in F of m(f) * a(f)
```

That is the entire mechanism of fan-out inflation. It is not a database defect; it is the arithmetic definition of the operator. The correct result is the same expression with `m(f)` replaced by the indicator that `f` belongs to the reported population, which equals one for a required join with `m(f) = 1`.

```text
FORMULA [GM-01]: A_true = sum over f in F_pop of a(f)
WHERE: a(f) [currency cents per fact row]; F_pop [rows of F satisfying the
       published population predicate, e.g. completed lines].
CONSTRAINT: one row per fact key; no pre-join duplication; a is additive at
             this grain; population predicate is fixed before the join.
NUMERICAL EXAMPLE: a = 4000, 6000, 10000, 2500, 7500, 5000 cents for six
                   completed lines; A_true = 35000 cents.
INTERPRETATION: the certified total is a property of fact keys alone.
```

The distinction between `A_true` and the naive sum is the reporting error, and it decomposes into the two mechanisms with opposite signs.

```text
FORMULA [GM-02]: E = A_naive - A_true
                 = sum over matched f of (m(f) - 1) * a(f)
                   - sum over unmatched f of a(f)
WHERE: E [currency cents]; A_naive [cents], the sum over the joined relation;
       matched/unmatched defined by m(f) >= 1 versus m(f) = 0.
CONSTRAINT: F_pop is the full comparison set; unmatched rows exist only when
             the join is allowed to drop rows (inner join on a required path).
NUMERICAL EXAMPLE: duplicated rows add (2-1)*4000 + (2-1)*6000 = 10000;
                   the unmatched row subtracts 7500; E = 10000 - 7500 = 2500.
INTERPRETATION: inflation and loss are independent errors that net out, so a
                small residual error does not imply a correct query.
```

Two ratios summarise the join structure, and their denominators deliberately differ.

```text
FORMULA [GM-03]: c_join = |M| / |F|
WHERE: c_join [dimensionless, 0..1]; M = { f in F : m(f) >= 1 } the matched
       fact keys; F the input fact relation at the same snapshot.
CONSTRAINT: |F| > 0; the predicate, snapshot and population used to build M are
             identical to those used for the reported aggregate.
NUMERICAL EXAMPLE: |F| = 6, |M| = 5, so c_join = 5/6 = 0.833333...
INTERPRETATION: one sixth of the input facts has no relationship match; under an
                inner join those facts are absent from every downstream total.
```

```text
FORMULA [GM-04]: m_join = |F ⋈_P D| / |M|
WHERE: m_join [joined rows per matched fact]; the numerator is the materialised
       row count; the denominator counts only facts that matched.
CONSTRAINT: |M| > 0; joined rows must retain fact-key identity so that the
             numerator is attributable; otherwise the ratio is uninterpretable.
NUMERICAL EXAMPLE: joined rows = 1+1+2+2+1+0 = 7; m_join = 7/5 = 1.4.
INTERPRETATION: a matched fact produces 1.4 rows on average, which is
                inadmissible for an additive measure unless deliberate.
```

The two ratios are related but not redundant. By construction `|F ⋈ D| = sum_{f in M} m(f)`, so `m_join` is the arithmetic mean of the per-fact match count over the matched set. Coverage is a population-level statement about survival; multiplicity is a conditional statement about replication given survival. A query can have `c_join = 1` with `m_join = 5` (everything matched, everything quintupled), or `c_join = 0.2` with `m_join = 1` (most facts lost, survivors clean). Reporting only one of them describes half the failure.

The row-count identity is the trap that this pair of ratios exists to defeat. Suppose an analyst checks `|F ⋈ D| = |F|` and concludes the join is safe. The condition `|F ⋈ D| = |F|` is satisfied whenever the average of `m(f)` over all of `F`, including unmatched rows counted as zero, equals one:

```text
FORMULA [GM-05]: |joined| = |F|  <=>  (1/|F|) * sum over f in F of m(f) = 1
WHERE: both sides are row counts [rows]; the right side includes m(f) = 0 for
       unmatched facts.
CONSTRAINT: the equivalence assumes no other operator (filter, dedup) lies
             between the join output and the count being compared.
NUMERICAL EXAMPLE: m values 2, 1, 0 average to 1, so joined = 3 = |F|, while
                   c_join = 2/3 and m_join = 3/2.
INTERPRETATION: equality of row counts is compatible with simultaneous loss and
                duplication, so it certifies nothing.
```

**Takeaway:** The join multiplies each fact's measure by its own match count, so safety is the statement `m(f) = 1` for every reported fact — not a statement about the output row count.

### 4.2 General case: several hops and several measures

Validity before use: each hop's predicate is a pure function of its two inputs, the measures remain at one declared grain, and no operator between hops performs aggregation or deduplication. Under those conditions the per-fact match count composes multiplicatively along the path, and the total contribution of a fact is its measure times its composed match count.

Derivation. Consider the two-hop path `F ⋈_{P1} D1 ⋈_{P2} D2`. Fix a fact row `f`. Its matches into `D1` form the set `D1(f) = { d1 : P1(f, d1) }` with size `m1(f)`. Each such `d1` in turn matches `m2(d1)` rows of `D2`. The pairs `(f, d1, d2)` satisfying both predicates therefore number

```text
m_total(f)  =  sum over d1 in D1(f) of m2(d1)
```

which equals `m1(f) * m2` only when every `d1` that `f` matches has the same `m2`, in particular when `m2` is constant. The general expression is a sum of per-branch counts, not a product of averages, and this is where careless estimates go wrong: an analyst who multiplies the two average multiplicities computes `E[m1] * E[m2]` while the truth is `E[m1 * m2]`, and the two differ by the covariance between branch fan-out and branch count.

```text
FORMULA [GM-06]: m_total(f) = sum over d1 in D1(f) of m2(d1)
WHERE: m_total(f) [rows per input fact]; D1(f) [set of first-hop matches];
       m2(d1) [second-hop match count of that intermediate row].
CONSTRAINT: predicates are pure and static for the duration of the query; no
             aggregation, filtering or deduplication between hops.
NUMERICAL EXAMPLE: f1 matches d1 (m2=1) and d2 (m2=9), so m_total(f1)=10;
                   f2 matches only d2, so m_total(f2)=9; mean m_total = 9.5.
INTERPRETATION: composed fan-out must be summed over branches, never inferred
                from a product of average multiplicities.
```

The example is chosen to break the tempting shortcut. An analyst who multiplies the average first-hop degree by the average second-hop degree computes `avg(m1) * avg(m2) = 1.5 * 5 = 7.5`, while the true mean composed count is `9.5`. The gap exists because one intermediate row `d2` is shared by both facts: its fan-out of nine is paid twice, once for each fact, but it appears only once in an average over intermediate rows. Formally the identity is

```text
E_f[m_total] = E_f[m1] * ( pair-weighted mean of m2 )
```

and the second factor weights each intermediate row by the number of facts that reach it, not by one. The two factors coincide only in the special case where every intermediate row is reached by the same number of facts. That condition is rarely checkable in advance, which is why composed multiplicity must be measured on the materialised join rather than estimated hop by hop.

The second generalisation concerns several measures at different grains in one report. A line-grain fact joined to an order-grain fact is the standard construction, and it inflates the coarser measure by the number of finer rows per coarser key.

```text
FORMULA [GM-07]: A_coarse_reported = sum over o in Orders of a_order(o) * n_lines(o)
WHERE: a_order(o) [currency cents per order]; n_lines(o) [line rows per order];
       the report grain is the line.
CONSTRAINT: the coarse measure is stored once per order and the join is on
             order_id; the report does not pre-aggregate to order grain.
NUMERICAL EXAMPLE: order amount 30000 cents with 4 lines contributes
                   4 * 30000 = 120000 cents at line grain, an overstatement
                   of 90000 cents for that order.
INTERPRETATION: mixing grains in one relation multiplies every coarser measure
                by the finer row count, and the error grows with basket size.
```

The error in [GM-07] is not random. It is proportional to the average number of finer rows per coarser key, so it is largest exactly where the business cares most — large baskets, multi-item orders, high-engagement accounts. A model that mixes grains will therefore look approximately correct on small customers and dramatically wrong on large ones, which is a worse failure than a uniform bias because it distorts every comparison between segments.

Pre-aggregation is the standard correction, and it commutes with the join under a precise condition rather than in general.

```text
Validity: aggregating F up to the join-key grain before joining D yields the same
grouped totals as joining first and aggregating after, provided (i) the join is
many-to-one from the pre-aggregated grain to D, (ii) the population filter is
applied before the pre-aggregation, and (iii) the grouping attributes come from D
or from the join key itself.
```

Derivation of the condition. After pre-aggregation, the relation `G` has one row per join-key value `k` with measure `sum_{f : key(f) = k} a(f)`. Joining `G` to `D` on `k` produces, for each `k`, exactly `m(k)` copies where `m(k)` is the number of `D` rows with that key. If `D` is keyed uniquely on `k` under the predicate, `m(k) = 1` and the join is a pure attribute lookup that cannot change any sum. If `D` is not unique, the pre-aggregation has merely moved the fan-out one step later. Grouping by an attribute of `D` then partitions the `G` rows the same way the joined rows would have been partitioned, because each `G` row carries the attribute of its unique `D` partner.

The failure of the condition is instructive. Pre-aggregate to customer grain, then join to a bridge table of customer-to-segment edges where one customer belongs to three segments: the customer total is now tripled and attributed three times. Pre-aggregation did not prevent fan-out; it deferred it. The condition (i) is what forbids that pattern, and it must be checked against the actual key uniqueness at the actual snapshot, not against the documented intent.

A fourth general result is the semi-join bound. When the only purpose of a join is to test existence, the semi-join keeps each left row at most once regardless of how many right rows match.

```text
FORMULA [GM-08]: |F SEMI-JOIN_P D| = |M|  and  |F ANTI-JOIN_P D| = |F| - |M|
WHERE: M = { f : exists d in D with P(f, d) } [facts with at least one match];
       both results are row counts [rows].
CONSTRAINT: the right side is used only for existence, and no right-side column
             appears in the select list or the grouping.
NUMERICAL EXAMPLE: |F| = 6, |M| = 5, so the semi-join returns 5 rows and the
                   anti-join returns 1 row, with no measure inflation in either.
INTERPRETATION: existence tests should never be expressed as an inner join when
                the right side is non-unique; the semi-join is the safe operator.
```

The practical value of [GM-08] is that it separates two intentions that SQL syntax conflates. "Keep facts that have a matching customer" and "attach the customer's attributes" are different requirements. The first is a filter and should be a semi-join, which cannot inflate. The second requires uniqueness of the dimension key and should be guarded by an assertion. Writing both as `JOIN dim_customer` is how inflation enters reports that were never meant to be at dimension grain.

**Takeaway:** Composed fan-out is a sum over branches rather than a product of averages, mixed-grain measures are inflated by construction, and existence tests belong in semi-joins where they cannot replicate anything.

### 4.3 Variants: operator forms and their counting effects

Each variant below changes the row-count identity while leaving the underlying relationship untouched. Reading the table as a set of alternative arithmetics is the fastest way to choose an operator deliberately.

| Operator | Rows produced | Measure effect | Legitimate use |
|---|---|---|---|
| Inner join | `sum_f m(f)` | each fact copied `m(f)` times | required relationship with `m(f) = 1` |
| Left outer join | `sum_f max(m(f), 1)` | unmatched facts contribute exactly once with nulls | optional dimension, nulls explicitly handled |
| Right/full outer | as above plus unmatched right rows | right-only rows carry null measures | reconciliation and orphan discovery only |
| Semi-join | `|M|` | no replication | existence filter |
| Anti-join | `|F| - |M|` | no replication | orphan and non-participation reporting |
| Cross join | `|F| * |D|` | full cartesian inflation | deliberate calendar scaffolding, never with measures |
| As-of join | `sum_f k(f, t_f)` | one version expected; more means overlap | type-2 attribution |
| Allocated bridge | `sum_e edges` | `a(f) * w(f, e)`, conserved | many-to-many additive attribution |
| Pre-aggregate then join | `|G|` grouped rows | sum preserved if key unique in `D` | dashboard pre-computation |
| Range join | `sum_f overlap_count` | depends on interval multiplicity | period bucketing, rate cards |

The left outer join deserves a precise statement because its protection is conditional on downstream predicate placement.

```text
FORMULA [GM-09]: |F LEFT JOIN_P D| = sum over f in F of max(m(f), 1)
WHERE: max(.,1) counts an unmatched left row once with null right attributes;
       the result is a row count [rows].
CONSTRAINT: no later predicate on a right-side column, and any such predicate
             written in the WHERE clause converts the join back to an inner join.
NUMERICAL EXAMPLE: m values 2, 1, 0 give 2 + 1 + 1 = 4 rows for 3 facts, of
                   which exactly one row carries null attributes.
INTERPRETATION: the outer join preserves the fact but the null row is invisible
                to any predicate that tests the right side.
```

The predicate-placement rule in the constraint is a semantic property of SQL, not a stylistic preference. A predicate `d.region = 'north'` in the `WHERE` clause is evaluated after the join, at which point the null region from the unmatched fact fails the test and the row is removed. The same predicate in the `ON` clause is evaluated during matching, so the fact survives with null attributes. Both queries name a left join; only the second preserves the fact. Section 8's DIAG-05 turns this into a diagnostic.

The as-of join replaces equality with temporal containment, and its counting effect is governed by the number of dimension versions whose interval contains the fact's event time.

```text
FORMULA [GM-10]: k(f) = |{ d in D : d.key = f.key and d.from <= t_f < d.to }|
WHERE: k(f) [versions matching fact f at its event time t_f]; intervals are
       half-open so a boundary instant belongs to exactly one version.
CONSTRAINT: intervals for one key must be non-overlapping and must cover the
             reporting horizon; otherwise k(f) > 1 or k(f) = 0 respectively.
NUMERICAL EXAMPLE: versions [Jan01, Feb01) north and [Feb01, ...) east; a fact
                   at 2026-01-31T23:00Z has k = 1 (north); at exactly
                   2026-02-01T00:00Z it also has k = 1, now east.
INTERPRETATION: exactly-one is the contract; zero is loss and two is fan-out, and
                half-open intervals are what make the boundary unambiguous.
```

The half-open convention is doing real work in [GM-10]. With inclusive endpoints on both sides, a fact at exactly `Feb01T00:00Z` matches both versions and its measure is attributed twice. With exclusive endpoints on both sides, the boundary instant matches neither and the fact disappears from every region total. Neither error is detectable from the total across all regions if some other fact compensates, which is why the per-fact `k(f) = 1` assertion is the release gate rather than a spot check.

Allocated bridges replace the match count with a weight, and correctness becomes a conservation statement rather than an equality statement.

```text
FORMULA [GM-11]: A_alloc(e) = sum over f of a(f) * w(f, e)
                 A_alloc_total = sum over f of a(f) * W(f),  W(f) = sum over e of w(f, e)
WHERE: w(f, e) [dimensionless weight of edge e for fact f, w >= 0];
       A_alloc(e) [currency cents attributed to edge e]; W(f) [total weight].
CONSTRAINT: W(f) = 1 for every fact key in the allocated report, weights are
             nonnegative, and the weight version is recorded with the output.
NUMERICAL EXAMPLE: A = 10000 with weights 0.5 and 0.5 gives 5000 + 5000 = 10000;
                   with weights 0.7 and 0.7, W = 1.4 and the total is 14000.
INTERPRETATION: allocation conserves the measure only when the weights form a
                partition; otherwise the report is a scaled association metric.
```

The unweighted case is worth stating separately because it is the most common mislabelling in practice. If each edge receives the full measure, then `W(f) = deg(f)` and the allocated total equals the sum of `a(f) * deg(f)`, which is precisely the fan-out inflation of [GM-02]. Calling that number "revenue by tag" and reconciling it against company revenue is guaranteed to fail whenever any fact has more than one tag. The same number is perfectly valid under the label "revenue associated with each tag", where the measure is explicitly non-additive across tags. The arithmetic does not change; the contract does.

A final variant is the aggregate-before-join pattern used by pre-computed marts. Its counting effect is that the fact relation is replaced by `G`, one row per join key, so the join's multiplicity applies to groups rather than facts.

```text
FORMULA [GM-12]: |G JOIN_P D| = sum over k in G of m_D(k)
WHERE: G is F aggregated to the join key [one row per key]; m_D(k) [rows of D
       sharing that key]; the result is a row count [rows].
CONSTRAINT: G's measure is a sum over the complete fact population for key k,
             and D is keyed uniquely on k for the join to be measure-preserving.
NUMERICAL EXAMPLE: G has 4 customer keys with totals 9000, 12000, 8000, 6000;
                   D has one row per key plus one duplicate for the second key,
                   giving 5 joined rows and a naive total of 47000 versus 35000.
INTERPRETATION: aggregation does not immunise a model against duplicate
                dimension keys; it only changes the size of the damage.
```

**Takeaway:** Choose the operator from the relationship's contract — equality for unique lookups, semi-join for existence, temporal containment for history, weighted edges for many-to-many — because each operator has a different and predictable counting law.

### 4.4 Limits and counterexamples

The framework above certifies structure, and structure is necessary but not sufficient. The following limits mark where the two ratios stop being informative, and each is paired with the additional check that covers the gap.

**Limit 1 — additive measures only.** `SUM` is the measure operation that [GM-01] describes. For a ratio measure such as conversion rate, the correct aggregate is a ratio of sums, and the row-preservation argument does not transfer.

```text
FORMULA [GM-13]: rate = (sum over f in F_c of n_convert(f)) / (sum over f in F_v of n_visit(f))
WHERE: numerator and denominator are counts over possibly different fact sets
       or the same fact set with different conditions; both [events].
CONSTRAINT: numerator population must be a subset of denominator population in
             the same window; averaging per-row rates is not equivalent.
NUMERICAL EXAMPLE: 30 conversions over 1200 visits = 0.025; the mean of the
                   per-channel rates 0.10 and 0.02 need not equal 0.025.
INTERPRETATION: a ratio must be recomputed from summed components after any join,
                and the components must be inflated by the same factor.
```

The subtlety in [GM-13] is that a join can inflate numerator and denominator by different factors if the two counts come from different relations or different predicates. When both come from the same fact rows and the same join, the common factor cancels in the ratio, so the rate survives fan-out even though the counts do not. This is exactly why a broken query can produce a correct-looking rate alongside a wildly wrong volume, and why rates and volumes must be reconciled separately.

**Limit 2 — structural correctness does not establish population identity.** Two models can both satisfy `c_join = 1` and `m_join = 1` while computing different totals, because they define the population differently. One model counts a revised order once at its latest version; the other counts every version. Both joins are clean; the totals differ by the revision volume. Coverage and multiplicity are statements about a join relative to an input set, and they say nothing about whether the input set is the right one. The population predicate must therefore be reconciled against an independent source — a finance ledger, a source-system count, or a certified prior extract — and that reconciliation is a separate control from the join diagnostics.

**Limit 3 — wrong key, right cardinality.** A join can be one-to-one and still attach the wrong attribute. If `dim_customer` is unique on `customer_id` but the fact's `customer_id` was populated from a billing account that sometimes differs from the ordering account, then every fact matches exactly once and every attribute is potentially wrong. No coverage or multiplicity check detects this, because the cardinality is perfect. The discriminating evidence is a semantic comparison of key definitions in the two contracts, plus a sample audit against a second independent source. This is the failure class that motivated Section 1's insistence that a model is a set of assertions: the assertion "the fact's customer is the billing customer" is separate from "the key is unique", and only the first is true.

**Limit 4 — null semantics.** SQL's join predicate treats `NULL` as unknown rather than equal, so a fact whose key is `NULL` matches nothing even under a left join's dimension side. Its measure survives the left join with null attributes but disappears under any inner join, and a `NOT IN` subquery against a nullable column silently returns nothing at all. The check is explicit: count null join keys per relation and include them in the coverage denominator, because a fact with a null key has `m(f) = 0` and belongs in the loss class.

**Limit 5 — duplicate natural keys inside the dimension.** If a dimension table contains two rows with the same natural key and no distinguishing interval, the natural-key join produces `m(f) = 2` for every affected fact. Aggregating the dimension to remove duplicates is not safe, because the two rows may disagree on attributes and there is no basis in the data for choosing. The correct response is to quarantine the natural key, decide the tie-break rule with the owner, and reprocess; Section 8's DIAG-02 shows the mechanism.

**Limit 6 — currency and unit heterogeneity.** A join between two relations expressed in different units satisfies every cardinality check and produces a meaningless total. The 2026-era analytic stack makes this easy, because a fact may store minor units while a dimension stores major units, or two source systems may report in different currencies. The guard is a units declaration in the contract and an assertion that the measure's unit is constant across the joined population; the numeric symptom is a total that is off by a factor of exactly 100 or by a published exchange rate.

**Limit 7 — non-deterministic version selection.** A "latest version" rule implemented as `MAX(updated_at)` or `ROW_NUMBER() OVER (... ORDER BY updated_at DESC)` without a tie-breaker returns a different row on different executions when two versions share a timestamp. Coverage and multiplicity both equal one on each execution, so both checks pass, and the reported attributes change between runs. The discriminating check is to count keys with tied ordering values; the fix is a total order (timestamp, then sequence, then hash) recorded in the contract.

**Limit 8 — floating-point accumulation.** Conservation checks on currency are exact only if the arithmetic is exact. Binary floating point makes `sum` order-dependent, so a conservation assertion of `W(f) = 1` can fail by `1e-16` and a reordered `SUM` can change the last cent of a large report. The guard is integer minor units for currency and a declared tolerance for any weight assertion, with the tolerance stated in the contract rather than chosen to make the test pass.

The counterexample that binds these limits together is simple enough to state in one line: **a query can have perfect coverage, perfect multiplicity, a stable row count, and a wrong answer.** Every one of the eight limits produces that outcome. The two ratios protect against loss and replication; they do not protect against a wrong population, a wrong key, a wrong unit, or a wrong tie-break. A complete release argument therefore has four parts — grain, cardinality, population reconciliation, and semantic key review — and the interview questions in Section 11 are designed to test all four rather than the first two alone.

**Takeaway:** Coverage and multiplicity are necessary controls that certify structure only; population identity, key semantics, units, and deterministic ordering each require their own independent check.

## 5. Worked Numerical Example

The dataset below is **ILLUSTRATIVE**: six constructed order lines, integer cents, chosen so that arithmetic can be verified by hand. No production system is described. The point of the example is that a single query simultaneously loses one fact and duplicates two others, and every familiar sanity check passes.

### 5.1 Inputs

Fact relation `fact_line`, declared grain `(order_id, line_id)` at `event_ts`, population "status = completed, event_ts in Q1".

| line_id | order_id | customer_id | amount_cents | qty | event_ts |
|---|---|---|---:|---:|---|
| L1 | O1 | c1 | 4000 | 2 | 2026-01-05T10:00:00Z |
| L2 | O1 | c1 | 6000 | 1 | 2026-01-05T10:00:00Z |
| L3 | O2 | c2 | 10000 | 5 | 2026-02-11T14:30:00Z |
| L4 | O2 | c2 | 2500 | 1 | 2026-02-11T14:30:00Z |
| L5 | O3 | c3 | 7500 | 3 | 2026-03-20T09:15:00Z |
| L6 | O3 | c4 | 5000 | 2 | 2026-03-20T09:15:00Z |

Dimension relation `dim_customer`, claimed grain "one row per customer":

| customer_id | region | note |
|---|---|---|
| c1 | north | duplicate natural key, different label |
| c1 | north-legacy | duplicate natural key, different label |
| c2 | south | clean |
| c4 | west | clean |
| c5 | east | no facts reference it |

`c3` is absent from the dimension, and `c1` appears twice. Both defects are the kind that a unique constraint would have prevented at load time and that a join will convert into a numeric error at query time.

### 5.2 Step 1: certified total from fact keys alone

```text
A_true = 4000 + 6000 + 10000 + 2500 + 7500 + 5000
       = 35000 cents
       = 350.00 currency units
```

Row count check: six lines, six rows, so `|F| = 6`. Population check: all six are completed and inside Q1, so `F_pop = F`.

### 5.3 Step 2: per-fact match counts

```text
P(f, d) := (f.customer_id = d.customer_id)
m(L1) = 2   (c1 twice)
m(L2) = 2   (c1 twice)
m(L3) = 1   (c2 once)
m(L4) = 1   (c2 once)
m(L5) = 0   (c3 absent)
m(L6) = 1   (c4 once)
```

Sum of match counts: `2 + 2 + 1 + 1 + 0 + 1 = 7`. This is the materialised join cardinality, `|F ⋈ D| = 7`, and it can be predicted without executing the query — which is exactly the value of a grain declaration.

### 5.4 Step 3: the two ratios

```text
c_join = |M| / |F| = 5 / 6 = 0.833333...
m_join = |joined| / |M| = 7 / 5 = 1.4
```

Interpretation: one fact in six has no relationship match, and the five that matched produce 1.4 rows each on average. A required customer join must show `c_join = 1` and `m_join = 1`; both are violated.

### 5.5 Step 4: the naive aggregate and the error

```text
A_naive = 2*4000 + 2*6000 + 1*10000 + 1*2500 + 0*7500 + 1*5000
        = 8000 + 12000 + 10000 + 2500 + 5000
        = 37500 cents

E = A_naive - A_true = 37500 - 35000 = 2500 cents
```

Decompose the error by mechanism, using [GM-02]:

```text
inflation = (2-1)*4000 + (2-1)*6000            = 10000
loss      = 7500 (L5, unmatched)
E         = 10000 - 7500                        = 2500
```

The residual error is 2,500 cents, or 7.14% of the true total when computed as `2500/35000 = 0.071428...`. A 7% error is small enough to be attributed to seasonality by an unwary reader and large enough to distort a margin calculation.

### 5.6 Step 5: why the familiar checks pass

| Check | Value | Verdict it appears to support | What it actually establishes |
|---|---|---|---|
| Output row count equals input | `7` vs `6` | "slightly more rows, close enough" | nothing until per-key counts are inspected |
| Distinct customer count | `4` in both fact and dimension usage | "join looks fine" | dimension has 5 rows, 1 unused; says nothing about duplicates |
| Total revenue is in a plausible range | `37500` vs `35000` | "within 10%" | no range check can distinguish business movement from join error |
| No nulls in `region` | all 7 rows have a region | "join is clean" | duplicate `c1` rows both have regions; the loss of L5 produced no null |
| `COUNT(DISTINCT line_id)` | `5` | "five of six lines, close enough" | it reveals the loss of L5 but conceals the duplication of L1 and L2 |

The last row is worth stating precisely, because it is the check most often recommended and least often understood.

```text
COUNT(DISTINCT line_id) over joined  =  |M|  =  5
COUNT(line_id) over joined           =  |joined|  =  7
COUNT(*) over fact_line              =  |F|  =  6
```

So `COUNT(DISTINCT line_id) = 5` reveals the loss, and `COUNT(line_id)/COUNT(DISTINCT line_id) = 7/5 = 1.4` reveals the inflation, but neither number alone reveals both. In the earlier three-row counterexample with `m = 2, 1, 0`, `COUNT(*)` was identical before and after the join, `COUNT(DISTINCT key)` was 2 versus 3, and the total was wrong by 6000 cents. Every check is partial; only the pair of ratios plus a population reconciliation is complete.

### 5.7 Step 6: repair options and their arithmetic

There are three legitimate repairs, and the correct choice depends on what the business asserts about `c3` and about the duplicate `c1`.

**Repair A — restore referential integrity.** Load `c3` with a real region and delete or merge the duplicate `c1` row after deciding which label is true. After repair, `m(f) = 1` for all six facts, so `c_join = 1`, `m_join = 1`, and `A = 35000`. This is the only repair that also fixes the root cause.

**Repair B — reserve an unknown member.** Insert `dim_customer` row with key `-1`, region `UNKNOWN`, remap the missing `c3` fact to `-1`, and drop the duplicate `c1` row. Then `c_join = 1` and `m_join = 1` with totals `35000`, of which `7500` sits in the unknown slice. The unknown slice is measurable and can be monitored; its growth is a leading indicator of dimension-loading lag.

**Repair C — hold the metric.** Publish the prior certified total with a note that the current extract is under repair. This is the correct action when the decision cannot wait and the repair requires an owner decision about which `c1` label is true. It is not a repair; it is a controlled degradation, and it must have an owner and a deadline.

Two further options are illegitimate and should be named so they can be rejected quickly.

**Illegitimate option 1 — `SELECT DISTINCT`.** Applying `DISTINCT` to the joined rows collapses the two `c1` variants into one only if the selected columns are identical, which they are not (the regions differ). Adding `DISTINCT` to `line_id, amount` would give five rows and a total of 27500 cents, which is worse than the naive 37500 because it preserves the loss and hides the inflation. `DISTINCT` is a projection operator, not a relationship repair.

**Illegitimate option 2 — divide by the multiplicity.** Computing `37500 / 1.4 = 26785.71` cents does not recover `35000`, because the inflation factor applies only to matched facts while the loss applies to a specific unmatched fact. Dividing by an average assumes the error is uniform, which Section 5.5 has already disproved: the error is `+10000` on two facts and `-7500` on one.

### 5.8 Step 7: independent arithmetic checks

Three checks confirm the arithmetic without rerunning the logic.

```text
check 1: sum of per-fact contributions
         4000*2 + 6000*2 + 10000 + 2500 + 5000
       = 8000 + 12000 + 10000 + 2500 + 5000 = 37500        [matches A_naive]

check 2: error decomposition
         inflation 10000 - loss 7500 = 2500
         A_true + 2500 = 35000 + 2500 = 37500               [matches]

check 3: ratio consistency
         m_join * |M| = 1.4 * 5 = 7 = |joined|              [matches]
         c_join * |F| = (5/6) * 6 = 5 = |M|                 [matches]
```

A fourth check tests the dimension instead of the join: `|dim_customer| = 5` while only four distinct `customer_id` values appear in facts, so exactly one dimension row is unused and one natural key is duplicated. Counting distinct natural keys gives 4; counting rows gives 5; the difference of one is the duplicate, not the unused row. Both facts matter and both are recovered only by comparing the two counts.

### 5.9 Step 8: sensitivity to the population choice

Suppose `L6` is disputed: the order was placed on credit hold and released in Q2. The certified total becomes `35000 - 5000 = 30000`, and the naive total becomes `37500 - 5000 = 32500`. The error stays at 2500 cents, but its relative size rises:

```text
before: 2500 / 35000 = 0.071428...  (7.14%)
after:  2500 / 30000 = 0.083333...  (8.33%)
```

Relative error is therefore not stable under population changes, which is why error budgets should be stated in absolute minor units and reconciled against the source of record rather than against a percentage tolerance. A tolerance of "within 10%" would have passed both versions of this query and would also pass a query that dropped `L3` and tripled `L4`, whose error is `+5000 - 10000 = -5000` on a true 35000, or 14.3% — close enough to the boundary that someone would eventually argue about it.

**Takeaway:** Predict the joined cardinality from the key definitions, then verify both ratios and the total; the row count, the null check, and the plausibility range each fail to detect at least one of loss, duplication, and mis-population.

## 6. Visual / Geometric Intuition

### 6.1 Measure as area

Represent each fact row as a rectangle whose width is one row-unit and whose height is its measure in cents. The certified total is the sum of the areas.

```text
certified area (one column per fact row)
  ^ cents
  |  #### 10
  |  ##   7.5        #
  |  #    6      ##  5
  |  #    4      #   #
  +--#----#------#---#----> fact keys
    L1   L2     L3  L4  L5  L6   (heights: 4, 6, 10, 2.5, 7.5, 5)

total area = 4 + 6 + 10 + 2.5 + 7.5 + 5 = 35.0 (thousand cents)
```

The join transforms this picture in exactly two ways. Duplicating a row replaces one rectangle with `m` stacked copies, multiplying its area. Dropping a row deletes its rectangle, removing its area.

```text
after join
  ^ cents
  |  ##   ##  8      <- L1 copied twice
  |  ###  ### 12     <- L2 copied twice
  |            ## 10
  |            #  2.5
  |            (L5 deleted)
  |                #  5
  +--#----#----#---#---#----> joined rows
   7 rows, total area = 8 + 12 + 10 + 2.5 + 5 = 37.5
```

The geometric statement is precise: the area change is `+10.0` from the two copied rectangles and `-7.5` from the deleted one, netting `+2.5`. Counting rectangles before and after gives 6 and 7, which happens to reveal the problem here; in the three-row counterexample the counts are 3 and 3, and the area error is `+4.0 - 10.0 = -6.0`. Rectangle count and area are independent summaries, and only area answers a question about a sum.

### 6.2 Cardinality as a bipartite graph

Draw facts on the left and dimension rows on the right, with one edge per matching pair. Join cardinality is the edge count.

```text
facts                    dimension
L1 4000 ----+------------ c1 north
            +------------ c1 north-legacy
L2 6000 ----+------------ (same two rows, second edge set)
            +------------
L3 10000 ---------------- c2 south
L4  2500 ---------------- c2 south
L5  7500        (no edge)         c3 absent
L6  5000 ---------------- c4 west

             c5 east (isolated, no edges)
```

Three structural facts are visible at a glance, and all three matter.

The degree of a left node is its match count `m(f)`; the degree of a right node is the number of facts that reference it. Loss appears as a left node with degree zero. Duplication appears as a left node with degree greater than one. Unused dimension members appear as right nodes with degree zero, which is harmless for fact totals but indicates either a stale dimension or an over-broad load.

The bipartite view also makes the "wrong key, right cardinality" failure visible in a way that numbers do not. If every left node has degree exactly one, the graph is a clean matching and both ratios equal one. It still says nothing about whether each edge connects the *correct* pair, because the graph has no notion of correctness — only of matching. Correctness lives in the key definition, one level above the graph.

### 6.3 Allocation as conservation of area

For a many-to-many bridge, replace each edge with a share of the rectangle's height. Conservation is the statement that the shares reconstruct the original height.

```text
fact A = 10.0 with two edges:
   edge red   weight 0.5 -> 5.0
   edge sale  weight 0.5 -> 5.0
   sum of shares          = 10.0    CONSERVED

fact A = 10.0 with two edges, weights 0.7 and 0.7:
   edge red  0.7 -> 7.0
   edge sale 0.7 -> 7.0
   sum of shares = 14.0             NOT CONSERVED (scaled by 1.4)
```

The picture makes clear why conservation is an assertion about weights rather than about the join. The join is identical in both cases; only the weights differ. A conservation test that runs before publishing will catch the second case, and a test that runs after publication will catch it only as an unexplained excess in the tag report.

### 6.4 Time as a partition of the line

Effective intervals partition the timeline. A fact's event time falls into exactly one cell, and the join selects that cell's attributes.

```text
customer c1 timeline
 |---- north ----|---- east ----|---- (open) ---->
 Jan01          Feb01          Mar15
     ^                        ^
   H1 (Jan31)               H2 (Feb01)
   1 version                1 version
```

The partition metaphor explains the three temporal failure modes immediately. Overlapping intervals create a cell that belongs to two versions, so a fact in the overlap is counted twice. Gaps create instants belonging to no cell, so a fact there is dropped. Ambiguous boundaries — inclusive endpoints on both sides — make the single instant `Feb01T00:00Z` belong to both cells, which is the overlap case concentrated into one instant and therefore the hardest to notice in a sample.

### 6.5 The cost picture

Materialising the join is work proportional to the edge count, which is the same quantity that governs correctness. The diagram below shows why a bridge raises cost and error together.

```text
              facts 10M
                 |
        [build side: dimension]
                 |
   edges = sum of degrees = 10M * 1.4 = 14M
                 |
      [probe: 14M output rows]
             /        \
     aggregate       sort/shuffle
     35M cents      14M rows * width
```

The edge count is the single number that predicts both the correctness risk and the workload. A model whose bridge averages 1.4 edges per fact has 40% more materialised rows than its fact table, and that 40% is paid on every aggregation, every shuffle, and every spill to disk.

**Takeaway:** Loss deletes area, duplication copies area, and allocation must preserve area; every temporal and cardinality rule in this sheet is a statement about which of those three operations the join is performing.

## 7. Comparison / Selection Table

### 7.1 Model family selection

| Need | Pattern | Strength | Avoid when | Fallback |
|---|---|---|---|---|
| Frequent transactional updates with local ownership | Normalized entities in 3NF | Update anomalies minimized; each fact stored once | Analysts must traverse many joins and cannot see cardinality | Curated dimensional mart on top |
| Repeated, stable analytical questions | Star schema, fact plus conformed dimensions | Grain visible; joins shallow; aggregation obvious | Dimension history is undefined or disputed | Versioned dimensions with as-of joins |
| Complex source integration with audit needs | Data Vault hubs, links, satellites | Insert-only, auditable, parallel loadable | Consumers must write many joins and can mis-join links | Published dimensional presentation layer |
| One team, one question, high latency tolerance | Governed wide table | Fast to read, cheap to query | Refresh, grain and lineage are undocumented | Rebuild as fact plus dimensions |
| Attributes change over time and history matters | Type-2 dimension | Historical attribution correct | Effective intervals unvalidated | Repair intervals; hold historical metrics |
| Many-to-many attribution required | Bridge with weights | Conservation explicit and testable | No defensible weighting basis exists | Report association counts, not currency |
| Rapidly changing, high-cardinality attributes | Type-1 overwrite | Simple, small | Historical comparisons required later | Add a type-2 version of the changed attribute only |

The pattern choice is a statement about mutation frequency and question stability, not about fashion. A normalized core with a dimensional presentation layer is the common resolution because it puts integrity where writes happen and simplicity where reads happen; the cost is a transformation step that must itself be tested for grain preservation, which is why the presentation layer's own coverage and multiplicity assertions are not optional.

### 7.2 Operator selection by contract

| Relationship contract | Operator | Cardinality pre-condition | What it does to the measure | Disqualifier |
|---|---|---|---|---|
| Fact N:1 dimension, key unique | Inner join | `m(f) = 1` for all reported facts | Preserves each measure once | Key uniqueness unverified |
| Fact N:1 dimension, may be missing | Left join, nulls handled explicitly | `m(f) ∈ {0,1}` | Preserves facts; null attributes appear | Downstream `WHERE` on the dimension |
| Fact needs only existence | Semi-join | none required | Preserves or filters; never replicates | Dimension columns needed in output |
| Find facts with no match | Anti-join | none required | Returns only unmatched rows | `NOT IN` against a nullable key |
| Attribute true at event time | As-of join | `k(f) = 1` | Preserves measure once if intervals partition | Overlapping or gapped intervals |
| Fact relates to many members | Bridge with weights | `W(f) = 1`, `w ≥ 0` | Allocates measure across edges | Weights absent or unnormalised |
| Fact relates to many members, non-additive question | Bridge, no weights | none | Replicates; measure becomes association-level | Result labelled as additive revenue |
| Coarse and fine facts in one report | Pre-aggregate finer side, or separate facts | join key unique on coarse relation | Preserves if pre-aggregated to the join key | Coarse measure repeated at fine grain |

### 7.3 Diagnostic selection

| Symptom | First check | If it fires | If it does not fire |
|---|---|---|---|
| Total fell after adding a join | Coverage `c_join` and the anti-join key list | Missing dimension member or predicate placement | Check population predicate change and date cutoff |
| Total rose after adding a join | Multiplicity `m_join` and per-key duplicate list | Duplicate dimension rows or bridge edges | Check for a second copy of the fact relation |
| Total unchanged, distribution shifted | Per-key match counts, then key semantics | Compensating loss and duplication, or wrong key | Check whether a filter was replaced by a join |
| Row count unchanged, attributes unstable | Deterministic ordering of version selection | Tied ordering values | Check for non-deterministic sampling in the source |
| Two dashboards disagree | Grain declaration and snapshot identifiers | Different grain or cutoff | Different population predicate |
| Tag total exceeds company total | Weight sum per fact key | Unnormalised weights, or full-credit semantics | Check whether the label claims additivity |
| Null region spike | Predicate placement and unknown-member policy | Outer join demoted to inner, or mapping lag | Check the source's own geography gaps |
| Report not reproducible | Recorded snapshot and version tie-break | Mixed cutoffs, or latest-version race | Check for a mutable `current` flag |

**Takeaway:** Select the model by mutation profile, the operator by relationship contract, and the diagnostic by the direction and shape of the observed change — never by habit.

## 8. Failure Modes & Edge Cases

Each entry follows the same anatomy: symptom, competing causes, discriminating check, bounded fix, verification. All scenarios are ILLUSTRATIVE teaching constructions; the SQL and expected outputs are DERIVED from the stated inputs. None is a report of an observed production incident.

The diagnostic listings reference the six-row dataset from Section 5, extended where a failure mode requires columns the base schema does not carry. DIAG-04 adds `tenant_id` to both relations, DIAG-06 adds `hash` and `version`, and DIAG-08 adds `updated_at` and `effective_from`. Each listing therefore reads as "append these columns to the Section 5 tables" rather than as a query against an unstated schema. Where a listing's output depends on values not shown, the expected-signal paragraph states the qualitative discriminator instead of a fabricated number.

### DIAG-01 — Required dimension missing a member

**Symptom.** Total revenue falls from 35,000 to 27,500 cents after a `JOIN dim_customer` is introduced. The population count also falls, from six lines to five.

**Competing causes.** (a) The dimension genuinely lacks a member that the fact references, a referential defect. (b) A published rule excludes anonymous or unrecognized customers, in which case the population change is intended. (c) The dimension was loaded at an earlier cutoff than the fact, so the member exists but not at the join snapshot.

**Discriminating check.** Run an anti-join of facts against the dimension, grouped by order status and by load timestamp:

```sql
SELECT f.customer_id, COUNT(*) AS facts, SUM(f.amount_cents) AS cents,
       MIN(f.loaded_at) AS first_fact, MAX(f.loaded_at) AS last_fact
FROM fact_line f
LEFT JOIN dim_customer d ON d.customer_id = f.customer_id
WHERE d.customer_id IS NULL
GROUP BY f.customer_id;
```

**Expected signal.** If the missing key corresponds to completed orders with no anonymous flag, it is a referential defect. If the excluded key set matches an approved predicate exactly, the population change is intended and must be documented as such. If the missing key's `first_fact` predates the dimension's load watermark, it is a cutoff mismatch rather than a data defect.

**Bounded fix.** Do not fall back to a left join and hope. Either repair the dimension (load the member) or introduce a reserved unknown member with a stable key and an explicit label, and record which facts were attributed to it. Both preserve coverage at one; only the first restores the true attribution.

**Verification.** Coverage on completed orders equals 1.000 at the certified snapshot, the anti-join returns zero rows for that population, and the total returns to 35,000 cents with the unknown slice empty.

**Why the naive fix fails.** Converting the inner join to a left join keeps the fact but writes `NULL` into the region column. Any downstream `GROUP BY region` then produces a null bucket, and any subsequent `WHERE region IN (...)` silently restores the inner-join behaviour. The row is preserved arithmetically while remaining unusable for the question that motivated the join.

### DIAG-02 — Duplicate natural key in the dimension

**Symptom.** Revenue rises. `COUNT(DISTINCT line_id)` is five while `COUNT(*)` is seven, and two customers contribute exactly twice what their facts contain.

**Competing causes.** (a) A legitimate type-2 version pair mistakenly joined on natural key only, without a temporal predicate. (b) A load defect that inserted the same natural key twice with different attributes. (c) A deliberate multi-region membership where the grain of the dimension is genuinely one row per customer-region pair.

**Discriminating check.** Count natural-key occurrences at the join snapshot and inspect the distinguishing columns:

```sql
SELECT customer_id, COUNT(*) AS versions,
       COUNT(DISTINCT region) AS distinct_regions,
       MIN(effective_from) AS earliest, MAX(effective_from) AS latest
FROM dim_customer
GROUP BY customer_id
HAVING COUNT(*) > 1;
```

**Expected signal.** If `distinct_regions = 1` and the `effective_from` values differ, cause (a) or (b) with a benign duplicate. If `distinct_regions > 1` and the intervals are disjoint, cause (c) is a legitimate many-to-many that requires an allocation policy, not a deduplication. If the rows are identical in every column, the defect is pure duplication and the fix is a uniqueness constraint plus a delete.

**Bounded fix.** Quarantine the natural key, choose a documented tie-break with the data owner, and reprocess. Never select `MAX(region)` as a substitute for a temporal predicate: it is deterministic but arbitrary, and it will attribute January facts using a February attribute without any signal that it did so.

**Verification.** `SELECT customer_id FROM dim_customer GROUP BY customer_id HAVING COUNT(*) > 1` returns no rows, and `m_join = 1.0` exactly.

### DIAG-03 — Many-to-many bridge mislabelled as revenue

**Symptom.** Tag-level revenue sums to more than company revenue. A tag attached to every line reports the full 35,000 cents, and the sum across the four tags is 47,000 cents against a company total of 35,000.

**Competing causes.** (a) Every tag receives full credit for a shared fact, which is a legitimate association metric with a wrong label. (b) Weights exist but do not sum to one per fact. (c) The bridge contains duplicate edges for the same fact-tag pair, so a fact is credited twice to the same tag.

**Discriminating check.** Two queries: weight sums per fact, and edge uniqueness per fact-tag pair.

```sql
-- (b) non-conservation
SELECT line_id, SUM(weight) AS w_sum FROM line_tag_weight
GROUP BY line_id HAVING ABS(SUM(weight) - 1.0) > 1e-9;

-- (c) duplicate edges
SELECT line_id, tag, COUNT(*) AS edges FROM line_tag_weight
GROUP BY line_id, tag HAVING COUNT(*) > 1;
```

**Expected signal.** A non-empty first result proves cause (b) with the magnitude of the deviation printed. A non-empty second result proves cause (c). Both empty, with tags summing above the total, proves cause (a) — full credit — which is not a defect but a labelling error.

**Bounded fix.** For (a), rename the measure to "tag-associated revenue" and stop reconciling it against company totals; alternatively, publish a separate allocated measure. For (b), version the weighting policy, renormalise, and re-run; do not silently divide by the observed sum, because that hides the upstream defect and rewrites history differently on each run. For (c), deduplicate the edge and add a primary key on `(fact_key, member_key)`.

**Verification.** For the allocated measure, `SUM(allocated) = 35000` cents exactly, per-fact weight sums equal 1 within the declared tolerance, and no edge pair is duplicated. For the association measure, the label says association and the report is documented as non-additive across tags.

### DIAG-04 — Composite key collapsed to one column

**Symptom.** A customer attribute appears under the wrong tenant. Totals within each tenant are plausible, but cross-tenant attribution is wrong and a compliance review flags it.

**Competing causes.** (a) The join omits `tenant_id` from the key, so `customer_id = 'c1'` in tenant A matches the same identifier in tenant B. (b) Two tenants were genuinely consolidated under a documented identity mapping with an effective date. (c) The source system reuses identifiers across environments (test and production data in one lake).

**Discriminating check.** Compare match counts under the composite key and the single-column key:

```sql
SELECT
  (SELECT COUNT(*) FROM fact_line f JOIN dim_customer d
     ON d.customer_id = f.customer_id) AS single_col_matches,
  (SELECT COUNT(*) FROM fact_line f JOIN dim_customer d
     ON d.tenant_id = f.tenant_id AND d.customer_id = f.customer_id) AS composite_matches;
```

**Expected signal.** `single_col_matches > composite_matches` proves cause (a), and the difference is the cross-tenant collision count. A composite match count of zero for some facts proves the key direction is wrong. A documented mapping table with effective dates proves cause (b).

**Bounded fix.** Restore the composite key in every predicate, view, and mart, and revoke the affected output. Do not deduplicate identifiers across tenants, and do not "fix" the collision with `DISTINCT`, which would drop one tenant's legitimate fact rather than restoring the correct one.

**Verification.** Zero joined rows have unequal tenant identifiers, and each tenant's fact total reconciles independently against its own source extract. Add a constraint or an assertion at the modelling layer so the composite key cannot regress.

### DIAG-05 — Outer join demoted by predicate placement

**Symptom.** An `Unknown` region bucket appears, then disappears from the next dashboard refresh without any load change. Or: the unknown bucket is huge while the fact row count did not change.

**Competing causes.** (a) A legitimate unmapped geography that arrived with a new market. (b) A `WHERE d.region = ...` predicate converting the left join to an inner join, deleting the null rows. (c) An unknown-member mapping change that reclassified the slice.

**Discriminating check.** Count null-attributed rows at three points: immediately after the join, after each filter, and after the final aggregation. Also inspect whether dimension predicates sit in `ON` or `WHERE`.

**Expected signal.** Null counts constant through the join and then dropping to zero after a filter is cause (b): query semantics, not data. Null counts high at the join and stable afterwards is cause (a): a mapping lag. A step change aligned with a mapping deployment is cause (c).

**Bounded fix.** Decide explicitly whether unknown is included, excluded, or separately reported, and encode that in the contract. When preserving facts is required, put optional-dimension predicates in the `ON` clause or filter on a pre-computed inclusion flag after the join.

**Verification.** Coverage remains 1.0, the unknown row count equals the left anti-join count exactly, and the total across included plus excluded plus unknown equals the certified source total.

### DIAG-06 — Source retry versus semantic revision

**Symptom.** `COUNT(DISTINCT order_id)` is stable at 2,000 while total amount rises by 4%. Both numbers come from the same table and the same window.

**Competing causes.** (a) Retried identical events duplicated at the source, which `DISTINCT order_id` masks. (b) Genuine revisions where an order's amount was corrected, and both versions are being summed. (c) Late-arriving lines appended to existing orders with new line identifiers.

**Discriminating check.** Compare payload hashes, version sequence numbers, and the contract key:

```sql
SELECT order_id, hash, version, COUNT(*) AS copies
FROM fact_line
GROUP BY order_id, hash, version
HAVING COUNT(*) > 1;
```

and separately

```sql
SELECT order_id, COUNT(DISTINCT version) AS versions, COUNT(*) AS rows
FROM fact_line GROUP BY order_id HAVING COUNT(DISTINCT version) > 1;
```

**Expected signal.** Duplicate `(order_id, hash, version)` triples with `copies > 1` prove (a). Distinct versions with distinct hashes prove (b), and the correct total requires selecting one version per order under a stated as-of rule. New line identifiers under an existing order prove (c), which is not duplication at all.

**Bounded fix.** Deduplicate only exact retry identity, using producer identity plus hash plus version. Model revisions explicitly as an append-only history and select at the consumer's declared as-of time. For (c), leave the rows alone and fix the aggregation grain instead.

**Verification.** Amount at the selected version equals the source system's certified figure for that order, history remains replayable at an earlier as-of time, and a second run against the same snapshot returns an identical row set.

### DIAG-07 — Compensating loss and duplication

**Symptom.** Output row count equals input row count, the total is within 3% of the prior period, and everyone concludes the model is fine.

**Competing causes.** (a) Genuinely no defect and a small business change. (b) One fact lost while another is duplicated, with the amounts nearly offsetting. (c) A population shift in one direction and a join defect in the other.

**Discriminating check.** Compute both ratios and list the offending keys:

```sql
WITH joined AS (
  SELECT f.line_id FROM fact_line f JOIN dim_customer d
    ON d.customer_id = f.customer_id
), matched AS (SELECT DISTINCT line_id FROM joined)
SELECT (SELECT COUNT(*) FROM fact_line) AS input_facts,
       (SELECT COUNT(*) FROM matched)    AS matched_facts,
       (SELECT COUNT(*) FROM joined)     AS joined_rows,
       1.0*(SELECT COUNT(*) FROM matched)/(SELECT COUNT(*) FROM fact_line) AS coverage,
       1.0*(SELECT COUNT(*) FROM joined)/(SELECT COUNT(*) FROM matched)    AS multiplicity;
```

**Expected signal.** For the six-row dataset this returns `6, 5, 7, 0.8333333, 1.4`, which is neither one nor close to it. Any deviation from `(1, 1)` proves (b) or (c) regardless of the total.

**Bounded fix.** Block additive aggregation on this path until both the loss and the duplication are repaired separately. Repairing one and re-measuring is mandatory; a joint repair that happens to restore the total can conceal a second error.

**Verification.** Coverage and multiplicity are both exactly 1.0 for a required one-to-one relationship, and the anti-join and duplicate-key list are both empty at the certified snapshot.

### DIAG-08 — Snapshot and cutoff drift

**Symptom.** The same report changes between runs with no source correction and no code deployment.

**Competing causes.** (a) A late fact was accepted into a later snapshot. (b) Facts and dimensions were queried at different cutoffs. (c) A `latest` flag or a `MAX(updated_at)` selection is non-deterministic because two rows share a timestamp.

**Discriminating check.** Record and compare four values with every aggregate: fact snapshot identifier, dimension snapshot identifier, event-time cutoff, processing-time cutoff. Then count ties in the ordering column:

```sql
SELECT customer_id, updated_at, COUNT(*) AS tied
FROM dim_customer GROUP BY customer_id, updated_at HAVING COUNT(*) > 1;
```

**Expected signal.** A changed snapshot identifier proves (a). Different fact and dimension snapshots prove (b). A non-empty tie result proves (c), and the reported attributes will flip between runs.

**Bounded fix.** Pin both relations to a compatible snapshot or publish a revisioned report with an explicit as-of label. Add a total ordering (timestamp, sequence, content hash) to the version selector. Never describe "current" as a reproducible cutoff.

**Verification.** Two executions against the recorded snapshots return byte-identical fact-key sets and identical totals; the tie query returns no rows; the report carries its snapshot identifiers in the output metadata.

### DIAG-09 — Mixed-grain measure in one report

**Symptom.** Average order value is computed from a line-grain table and comes out at exactly the line-level figure rather than the order-level one, and larger baskets look disproportionately better.

**Competing causes.** (a) The order-level amount is stored on every line and summed at line grain. (b) A genuine shift in basket composition toward multi-item orders. (c) A duplicated order header due to a join on a non-unique order key.

**Discriminating check.** Compare `SUM(amount)` per order against the order header's own amount, and count lines per order:

```sql
SELECT o.order_id, o.header_cents, SUM(l.amount_cents) AS line_cents,
       COUNT(*) AS lines
FROM order_header o JOIN fact_line l ON l.order_id = o.order_id
GROUP BY o.order_id, o.header_cents
HAVING SUM(l.amount_cents) <> o.header_cents;
```

**Expected signal.** A ratio of `line_cents / header_cents` equal to the line count proves (a), because the header amount was replicated per line. Ratios scattered in both directions prove (b). A line count above the source's own count proves (c).

**Bounded fix.** Keep coarse and fine facts in separate tables with separate declared grains, and never carry a coarse measure down to a fine grain. If one table is unavoidable, store a per-line allocated share rather than the full order amount, and document the allocation.

**Verification.** At order grain, the sum of line shares equals the header amount exactly, and any "average order value" is computed as `SUM(header) / COUNT(DISTINCT order_id)` rather than as a mean of line amounts.

### DIAG-10 — Semi-additive measure summed over time

**Symptom.** Quarterly inventory value is roughly ninety times the daily figure. The daily figure is correct.

**Competing causes.** (a) A balance being summed across days, which produces balance-days. (b) A duplicate snapshot per day from an intraday refresh. (c) A genuine inventory build.

**Discriminating check.** Compare `SUM(balance)` with `LAST(balance)` per month, and count snapshots per day per item:

```sql
SELECT item_id, snapshot_date, COUNT(*) AS snapshots
FROM inventory_snapshot GROUP BY item_id, snapshot_date HAVING COUNT(*) > 1;
```

**Expected signal.** A sum approximately equal to the mean daily balance times the day count proves (a). Multiple snapshots per day proves (b). A sum consistent with the last-of-month value times a stable item count proves (c).

**Bounded fix.** Declare the measure semi-additive and use an explicit time rule: last snapshot in period, average balance, or an integrated balance-time with the unit stated. Never let `SUM` be the default aggregator for a balance column.

**Verification.** The period value equals the declared rule applied to the declared snapshot times, recomputed by hand for one item and one month, and the unit of the reported figure is stated as a balance rather than balance-days.

### DIAG-11 — Unknown-member starvation and drift

**Symptom.** The `UNKNOWN` region share rises from 0.4% to 6% over a quarter without a corresponding market expansion.

**Competing causes.** (a) Dimension loading lag, so facts arrive before their members. (b) A new source system that does not populate the natural key. (c) An overly aggressive matching rule that fails on formatting differences.

**Discriminating check.** Attribute the unknown slice by source system and by fact arrival date, and compare the member's first appearance time with the facts that referenced it.

**Expected signal.** Unknown share concentrated in the most recent two days proves (a), a pipeline ordering problem. Concentration in one source proves (b). Concentration in one key format (leading zeros, case, whitespace) proves (c).

**Bounded fix.** For (a), reorder the load so dimensions precede facts, or accept the unknown and backfill by re-pointing the surrogate key when the member arrives. For (b), reject the batch at the contract boundary. For (c), normalise keys in one place, at the contract, and never in individual queries.

**Verification.** Unknown share returns below its declared threshold, backfilled facts show the correct attributed region, and the re-pointing operation is recorded so history remains auditable.

### DIAG-12 — Non-additive ratio combined across groups

**Symptom.** The company conversion rate printed on a dashboard is 6.8% while the sum of channel rates divided by channel count is 4.1%.

**Competing causes.** (a) A mean of per-channel rates used where a ratio of sums is required. (b) Different populations between numerator and denominator. (c) A join inflating numerator and denominator by different factors.

**Discriminating check.** Recompute both ways from the same rows and compare with the raw counts:

```text
rate_weighted = sum(conversions) / sum(visits)
rate_mean     = avg(conversions_i / visits_i)
```

If `rate_weighted` differs from the published figure, the aggregation is wrong. If `sum(conversions)` exceeds `sum(visits)` anywhere, causes (b) or (c) are present.

**Expected signal.** A published figure matching `rate_mean` proves (a). Numerator events outside the denominator population prove (b). A ratio that changes when a join is removed proves (c), because the join inflated only one side.

**Bounded fix.** Store numerator and denominator as separate additive counts and always recompute the ratio after aggregation. Where a join is required, apply it identically to both sides, or pre-aggregate both before joining.

**Verification.** The recomputed ratio matches an independent extract within a declared tolerance, and both component counts reconcile to their own sources independently of the ratio.

### Edge cases that sit outside the twelve patterns

Three further cases deserve short statements because they are common and their correct handling differs.

**Empty right side.** If the dimension is empty, an inner join returns zero rows and every ratio is undefined rather than zero. Coverage `0/|F|` is well defined when `|F| > 0`; multiplicity `0/0` is not. Reports must distinguish "no data" from "no matches", and an assertion that divides by `|M|` must guard `|M| > 0`.

**Single-row fact table.** With `|F| = 1`, coverage is either 0 or 1 and multiplicity equals the single match count, so both diagnostics collapse to trivial statements. Statistical confidence in a model validated only on tiny extracts is therefore near zero; validation must use a population large enough to contain at least one multi-match and one non-match, or the diagnostics are not being exercised.

**Null join key.** A fact with `NULL` in the join column has `m(f) = 0` under every equality predicate, including a left join's matching phase. The measure survives a left join as a null-attributed row but disappears under the inner join, and `NOT IN` against a nullable column returns nothing. The check is a null count on every join key in both relations, included in the coverage denominator.

```text
FORMULA [GM-14]: c_join = |M| / (|F| - |F_null|)     [strict variant]
WHERE: F_null [facts whose join key is null] are excluded from the denominator
       because no predicate could ever match them.
CONSTRAINT: the exclusion is declared and reported; it is not a silent filter.
NUMERICAL EXAMPLE: |F| = 6, |F_null| = 0, |M| = 5, so c_join = 5/6 = 0.8333.
INTERPRETATION: the strict variant separates "cannot match" from "did not
                match", which are different defects with different owners.
```

**Takeaway:** Every failure mode here ends in a replayable assertion, because a diagnosis without a verification query is an opinion; the twelve patterns plus the three edge cases cover the loss, duplication, attribution, temporal and population errors that modelling defects actually produce.

## 9. Complexity & Cost Analysis

### 9.1 Algorithmic cost of the join itself

A hash join over fact relation `F` and build relation `D` proceeds in two phases. The build phase inserts every `D` row into a hash table keyed by the join attributes, costing `O(|D|)` expected time and `O(|D|)` space for keys, payload and table overhead. The probe phase scans `F` and emits one output row per matching pair, costing `O(|F|)` expected time plus `O(|J|)` to write the output, where `J = F ⋈ D`. The expected total is therefore

```text
FORMULA [GM-15]: T_hash = O(|F| + |D| + |J|),   S_hash = O(|D|)
WHERE: T [time units proportional to row operations]; S [space in bytes for
       the build side]; |J| = sum over f of m(f) [rows].
CONSTRAINT: hash operations are expected O(1); the build side fits in the
             memory granted to the operator; no skew collapses one bucket.
NUMERICAL EXAMPLE: |F| = 10^7, |D| = 10^5, m_join = 1.4 gives |J| = 1.4*10^7,
                   so T ~ 2.5*10^7 row operations, dominated by output.
INTERPRETATION: correctness policy is also capacity policy, because m_join
                multiplies the dominant term.
```

The formula's practical content is that the third term is the one a modeller controls. `|F|` and `|D|` are given by the business; `|J|` is given by the relationship, and a bridge that averages 1.4 edges adds 40% to the dominant term of every downstream operation. When the build side does not fit, the engine either partitions both sides by hash and joins partition-wise, adding a write and a read of both relations, or spills to disk, adding random I/O that can dominate elapsed time entirely. Either response turns an `O(|F| + |D|)` memory-bound operation into an I/O-bound one with a much larger constant.

Sort-merge join has a different cost profile: `O(|F| log |F| + |D| log |D| + |J|)`, with space proportional to the sort buffers rather than to the whole build side. It is preferable when both sides are already sorted on the join key, when the build side is too large to hash, or when the output must be ordered anyway. Nested-loop join costs `O(|F| * |D|)` without an index and `O(|F| * log |D|)` with a B-tree index on the inner side; it is competitive only when the outer side is tiny or the inner lookup is an index seek on a highly selective predicate.

### 9.2 Cost of the diagnostics

The correctness controls are not free, and their cost determines whether they can run on every build.

| Control | Work | Space | Frequency | Notes |
|---|---|---|---|---|
| Coverage `c_join` | One pass over `F` plus a semi-join | `O(|M|)` distinct keys | Every build | Can often reuse the join's own output |
| Multiplicity `m_join` | One grouped count over `J` | `O(|M|)` groups | Every build | Requires fact-key identity in `J` |
| Anti-join key list | One pass over `F` with a hash probe | `O(|D|)` | Every build | Output size equals the defect size |
| Duplicate natural key scan | One grouped count over `D` | `O(|D|)` groups | Every load | Cheapest possible check |
| Interval overlap assertion | Sort by key, compare adjacent rows | `O(|D| log |D|)` or `O(|D|)` if sorted | Every load | Adjacent comparison suffices |
| Weight conservation | One grouped sum over the bridge | `O(distinct fact keys)` | Every build | Integer minor units for exactness |
| Reconciling to source total | One independent extract plus comparison | `O(1)` | Every build | The only check that validates population |
| Per-key attribution audit | Sample of keys against a second source | `O(sample)` | Weekly or on change | Only check that validates key semantics |

Read as a budget, the table says that the cheap structural controls (duplicate keys, weight sums, coverage, multiplicity) can run on every build, while the expensive semantic controls (attribution audit against a second source) must be sampled and scheduled. A model that runs only the structural controls is protected against arithmetic errors and not against definitional ones, which is precisely the distinction Section 4.4 drew.

### 9.3 Worked sizing: a bridge in a nightly mart

ILLUSTRATIVE workload, derived arithmetic, not a benchmark of any engine.

Consider a nightly mart built over `|F| = 40,000,000` fact rows with `|D| = 250,000` customer rows, a bridge with `1.4` edges per fact, and an average payload of 180 bytes per joined row.

```text
|J| = 40,000,000 * 1.4 = 56,000,000 rows
per-run materialisation = 56,000,000 * 180 bytes
                        = 10,080,000,000 bytes
                        = 9.3879... GiB        (1 GiB = 1,073,741,824 bytes)
```

Without the bridge, the same workload materialises 40,000,000 rows:

```text
40,000,000 * 180 = 7,200,000,000 bytes = 6.7055... GiB
```

The bridge adds 2.68 GiB of materialisation per run, a 40% increase, and it does so on every downstream stage that re-reads the joined relation. Over a 30-day month that is roughly 80.5 GiB of additional intermediate data written and read once per stage — derived arithmetic on a constructed workload, not a measured figure.

The decision consequence is not "avoid bridges". Bridges are correct for many-to-many questions. The consequence is that a bridge must be introduced with its edge count measured, so capacity planning uses `|J|` rather than `|F|`, and so a bridge whose edge count drifts upward (say from 1.4 to 2.2 after a tagging initiative) is caught as a cost regression at the same time as it is caught as a fan-out risk.

### 9.4 Cost of getting it wrong

The expensive failure is not the query; it is the decision made on the query. Suppose a 7.14% overstatement in a segment with 12,000,000 cents of true quarterly revenue reaches a pricing committee, and a 3% price change is approved on the strength of a segment's apparent margin advantage. The overstatement is 857,142.86 cents on that segment; the pricing decision's error is not bounded by that number, because the decision also reallocates attention and inventory toward a segment whose true position is weaker. Nothing in this paragraph is a measured outcome — it is a consequence model, and it exists to make one point: the diagnostic budget in 9.2 is trivially small compared to the cost of a decision taken on an unreconciled number.

A useful framing for the same argument is the asymmetry of the two errors. A blocked release costs one reporting cycle and one analyst-hour of diagnosis. A released error costs an unknown multiple of that, because its consequences are distributed across decisions that are not revisited when the number is corrected. This asymmetry justifies running the structural controls unconditionally even under schedule pressure.

### 9.5 Executed oracle: the six-row dataset

The listing below was executed as one unit against an in-memory SQLite database (Python 3.12 standard-library `sqlite3`), with the schema of Section 5 plus the tag bridge. Evidence label: **VERIFIED** for the arithmetic of this constructed dataset, **ILLUSTRATIVE** for the workload it represents.

```sql
CREATE TABLE fact_line (line_id TEXT PRIMARY KEY, order_id TEXT,
                        customer_id TEXT, amount_cents INTEGER);
INSERT INTO fact_line VALUES
 ('L1','O1','c1',4000),('L2','O1','c1',6000),('L3','O2','c2',10000),
 ('L4','O2','c2',2500),('L5','O3','c3',7500),('L6','O3','c4',5000);

CREATE TABLE dim_customer (customer_id TEXT, region TEXT);
INSERT INTO dim_customer VALUES
 ('c1','north'),('c1','north-legacy'),('c2','south'),('c4','west'),('c5','east');

CREATE TABLE line_tag_weight (line_id TEXT, tag TEXT, weight REAL,
                              PRIMARY KEY(line_id,tag));
INSERT INTO line_tag_weight VALUES
 ('L1','red',0.5),('L1','sale',0.5),('L2','sale',1.0),('L3','bulk',1.0),
 ('L4','sale',1.0),('L5','new',1.0),('L6','bulk',1.0);
```

**Query 1 — loss and fan-out in one pass.**

```sql
WITH joined AS (
  SELECT f.line_id FROM fact_line f JOIN dim_customer d
    ON d.customer_id = f.customer_id
), matched AS (SELECT DISTINCT line_id FROM joined)
SELECT (SELECT COUNT(*) FROM fact_line)  AS input_facts,
       (SELECT COUNT(*) FROM matched)    AS matched_facts,
       (SELECT COUNT(*) FROM joined)     AS joined_rows,
       (SELECT SUM(amount_cents) FROM fact_line) AS true_cents,
       (SELECT SUM(f.amount_cents) FROM fact_line f JOIN dim_customer d
          ON d.customer_id = f.customer_id)      AS naive_cents;
```

Observed output: `6 | 5 | 7 | 35000 | 37500`. Coverage `5/6 = 0.8333...`, multiplicity `7/5 = 1.4`, error `+2500`. The counts confirm the hand arithmetic of Section 5 exactly.

**Query 2 — naming the offending keys.**

```sql
WITH joined AS (
  SELECT f.line_id FROM fact_line f JOIN dim_customer d
    ON d.customer_id = f.customer_id
)
SELECT 'unmatched' AS kind, f.line_id FROM fact_line f
  LEFT JOIN joined j ON j.line_id = f.line_id WHERE j.line_id IS NULL
UNION ALL
SELECT 'duplicated', line_id FROM joined GROUP BY line_id HAVING COUNT(*) > 1
ORDER BY kind, line_id;
```

Observed output: `duplicated | L1`, `duplicated | L2`, `unmatched | L5`. The defect set is exactly the two facts sharing the duplicated natural key plus the one fact whose member is absent, which is what Section 5.3 predicted.

**Query 3 — allocation conservation.**

```sql
SELECT tag, CAST(SUM(f.amount_cents * t.weight) AS INT) AS cents
FROM fact_line f JOIN line_tag_weight t ON t.line_id = f.line_id
GROUP BY tag ORDER BY tag;

SELECT SUM(f.amount_cents * t.weight) AS total
FROM fact_line f JOIN line_tag_weight t ON t.line_id = f.line_id;

SELECT line_id, SUM(weight) AS w FROM line_tag_weight
GROUP BY line_id HAVING ABS(SUM(weight) - 1.0) > 1e-9;
```

Observed: allocated cents `bulk=15000`, `new=7500`, `red=2000`, `sale=10500`; total `35000.0`; and the weight assertion returns **no rows**. The four tag amounts sum to 35,000, which is the conservation property of [GM-11] demonstrated rather than asserted. Note that `sale` spans three facts (L1 half, L2 full, L4 full) and still totals correctly, because each fact's weights form its own partition.

Changing `L1`'s two weights to `0.7` and `0.7` would make the weight assertion return `L1 | 1.4` and the total become `35,000 + 0.4 * 4,000 = 36,600`, which is the [GM-11] failure mode appearing as a concrete number rather than as a warning.

**Query 4 — uniqueness of the natural key.**

```sql
SELECT customer_id, COUNT(*) AS versions FROM dim_customer
GROUP BY customer_id HAVING COUNT(*) > 1;
```

Observed: `c1 | 2`. This single query localises the entire inflation half of the error, which is why it belongs in every load rather than in an investigation.

The oracle's value is that four short queries separate four independent properties — survival, replication, conservation, and key uniqueness — and each one has an output that either is empty or is not. A model whose build runs these four assertions is protected against the arithmetic class of defect; it is still unprotected against the definitional class described in Section 4.4, which no amount of SQL over the same two tables can detect.

## 10. Key Theorems / Results Reference

The results below are the load-bearing statements of the sheet. Each is stated with its hypotheses, because in relational modelling the hypotheses are the whole content — a theorem without them is a slogan.

### 10.1 Bag semantics multiplication

**Statement.** For relations `F` and `D` and predicate `P`, `sum_{F ⋈_P D} a = sum_{f in F} m(f) * a(f)` where `m(f) = |{d : P(f,d)}|`. This is DERIVED directly from the definition of the join as a multiset of satisfying pairs and requires no assumption beyond bag semantics and the purity of `P`.

**Why it matters.** It identifies the mechanism of inflation exactly and shows that the inflation factor is per-row rather than global. There is no average correction that recovers the truth in general, only the per-key correction `m(f) = 1`.

**Hypotheses.** `P` is a function of the two rows only. No aggregation inside the join. `a` is defined on fact rows, not on pairs. Bag semantics, not set semantics — under set semantics duplicate rows would collapse and the identity would fail.

### 10.2 Uniqueness and join preservation

**Statement.** If `D` is unique on the join attributes under `P`, then `m(f) ∈ {0, 1}` for every `f`, and `F ⋈_P D` contains each matching fact exactly once. If additionally every `f` has at least one match, then `|F ⋈_P D| = |F|` and `sum_{F ⋈ D} a = sum_F a`.

**Why it matters.** It converts the design question "is this join safe?" into the verifiable question "is this key unique?", which is a constraint that can be asserted at load time. Referential integrity plus key uniqueness gives measure preservation as a theorem rather than as a hope.

**Hypotheses.** Uniqueness holds at the snapshot used for the query, not merely in the design document. The predicate uses the full key. No nullable key columns (or nulls handled by policy).

### 10.3 Loss and inflation are independent

**Statement.** Let `E_loss = sum_{f : m(f) = 0} a(f)` and `E_infl = sum_{f : m(f) >= 2} (m(f) - 1) * a(f)`. Then `E = E_infl - E_loss` exactly, and for a given total error `E` the two components are not identified: any pair `(E_loss + c, E_infl + c)` with `c >= 0` reproduces the same `E`.

**Why it matters.** This is the formal reason a plausible total cannot certify a query. The decomposition shows that observing `E = 0` is consistent with arbitrarily large compensating errors, so a near-zero discrepancy is evidence of nothing.

**Hypotheses.** The comparison set is the full fact population, and unmatched rows are counted with their measures rather than ignored.

### 10.4 Functional dependency and grain sufficiency

**Statement.** A relation `R` with key `K` can represent a fact at grain `K` without redundancy only if every non-key attribute is functionally dependent on `K` alone. If an attribute depends on a proper subset of `K`, that attribute has a grain coarser than `K` and its value repeats across the rows of each `K`-group.

**Why it matters.** This is the normalized-model justification for splitting tables, restated in the language of this sheet: an attribute determined by a subset of the key is a coarser-grain measure living at a finer grain, and summing it at the finer grain multiplies it by the group size. [GM-07] is the arithmetic consequence.

**Hypotheses.** `K` is indeed a key (uniqueness and minimality). The functional dependency is a logical constraint of the domain, not an observed correlation.

### 10.5 Pre-aggregation commutation condition

**Statement.** Let `G` be `F` grouped by the join key with measures summed, and let `D` be unique on that key. Then `sum_{G ⋈ D, grouped by attribute of D} measure = sum_{F ⋈ D, same grouping} a`, provided the population filter is applied to `F` before aggregation and the grouping attributes come from `D` or the join key.

**Why it matters.** It licenses the pre-computed mart, which is the dominant performance pattern in analytical warehouses. It also states exactly when the licence lapses: the moment `D` is not unique, pre-aggregation defers fan-out rather than removing it.

**Hypotheses.** Uniqueness of `D` on the key at query time; filter-before-aggregate; grouping attributes not drawn from `F` in a way that would be lost by aggregation.

### 10.6 Interval partition property

**Statement.** A set of half-open intervals `{[from_i, to_i)}` over one natural key partitions time if and only if the intervals are pairwise disjoint and their union covers the reporting horizon. Under a partition, every instant has exactly one matching version, so `k(f) = 1` for every fact time inside the horizon.

**Why it matters.** It converts "is the type-2 history valid?" into two mechanical checks — no overlaps, no gaps — both of which are computable with an adjacent-row comparison after sorting by `from`. It also explains why half-open is the correct representation: closed intervals cannot partition a discrete instant without assigning boundary instants twice.

**Hypotheses.** Timestamps are comparable across sources and stored in one time zone with a declared precision. Facts inside the horizon are the population under test.

### 10.7 Allocation conservation

**Statement.** For nonnegative weights `w(f, e)` with `sum_e w(f, e) = 1` for each fact `f`, the allocated measures satisfy `sum_e A_alloc(e) = sum_f a(f) = A_true`. Conversely, if `sum_e w(f, e) = W(f) ≠ 1` for some `f`, the allocated total equals `sum_f a(f) * W(f)`, which differs from `A_true` whenever `a(f) > 0` for such an `f`.

**Why it matters.** It gives a one-line test for whether an allocated report is comparable to a total, and it shows that unnormalised weights produce a total scaled by a measure-weighted average of `W(f)` — an error that grows with the facts that have the most edges, which is again the segment the business cares most about.

**Hypotheses.** Weights are nonnegative, defined per fact-edge pair, and versioned so that a report can be reproduced. Zero weights are permitted and mean "not attributed here".

### 10.8 Semi-join cardinality bound

**Statement.** `|F ⋉_P D| = |M|` and `|F ▷_P D| = |F| - |M|` regardless of the degree distribution on either side. In particular the semi-join never replicates a left row, whatever the right relation's cardinality.

**Why it matters.** It establishes the safe formulation of existence tests, which is where most accidental inflation originates. The bound is unconditional on `D`'s uniqueness, which is why a semi-join is the right operator when the right side's quality is unknown.

**Hypotheses.** Bag semantics with duplicate elimination on the left identity; the right side contributes no columns to the projection or grouping.

### 10.9 Null-key non-match

**Statement.** For any equality predicate `P` involving a key attribute, a null value never satisfies `P`, because SQL's three-valued logic evaluates the comparison to unknown. Therefore `m(f) = 0` for every fact with a null join key, under inner, left, and semi-joins alike.

**Why it matters.** It explains why a left join with null attributes and an inner join behave identically for null-keyed facts, and why `NOT IN` over a nullable column returns no rows at all. The practical consequence is that null keys must be counted explicitly in coverage denominators rather than assumed absent.

**Hypotheses.** SQL three-valued logic, standard equality semantics, no coalescing before the predicate.

### 10.10 Reconciliation is not implied by structure

**Statement.** The pair of conditions `c_join = 1` and `m_join = 1` constrains the join relative to its input set but does not constrain the input set itself. There exist pairs of models with identical join structure and different totals, differing only in population predicate, grain convention, unit, or version rule.

**Why it matters.** It is the formal statement of Section 4.4's limits, and it is the reason the release gate has four parts rather than two. Any process that treats the two ratios as sufficient will eventually ship a structurally perfect query over the wrong population.

**Hypotheses.** The comparison models share the same source data and differ in the four listed dimensions.

**Takeaway:** Multiplication, uniqueness, independence, dependency, commutation, partition, conservation, bound, non-match and non-implication — ten statements that between them cover every failure mode the diagnostics in Section 8 are built to detect.

## 11. Interview / Exam Questions

Fifteen questions across recall, application, and judgement. Each has a checked answer, the reasoning that earns credit, a follow-up, an acceptable alternative, and a scoring note with the red flag that ends the discussion.

### Q1. What exactly does a row of a fact table assert?

**Answer.** A complete statement: an identity tuple, a measurement at a declared unit, a time basis, a mutation rule, and a population predicate. "One row per order line, latest accepted version, completed status, event time in UTC" is a grain; "one row per order line" is only its first clause.

**Reasoning.** Every downstream ambiguity — revision double counting, duplicate rows, boundary effects — traces to one of the omitted clauses. Naming all five is what makes a row count predictable by a second person.

**Follow-up.** If two versions of the same order line exist, is the grain violated?

**Acceptable alternative.** A grain that names version explicitly, such as `(order_id, line_id, version)`, is valid and sometimes preferable, provided consumers select at a declared as-of version.

**Scoring and red flag.** "The primary key" alone scores 1 of 5. Naming identity plus time or version scores 3. All five clauses plus a downstream consequence scores 5. Red flag: describing grain as a property of the table's columns rather than of its rows' meaning.

### Q2. Why does a one-to-many join inflate an additive measure, in arithmetic terms?

**Answer.** Bag semantics emits one output row per satisfying pair, so a fact with `m(f)` matches contributes its measure `m(f)` times. The joined sum equals `sum_f m(f) * a(f)`, not `sum_f a(f)`.

**Reasoning.** The inflation is per-row, not a global scaling factor, which is why dividing the total by an average multiplicity does not recover the truth.

**Follow-up.** Compute `m_join` for a fact set with match counts 3, 1, 1.

**Acceptable alternative.** Describing it as "the same sale counted three times because it has three tags" is equivalent and acceptable if the per-row nature is preserved.

**Scoring and red flag.** "Because joins duplicate rows" scores 1. Writing `sum m(f) a(f)` scores 4. Adding that per-row factors cannot be averaged away scores 5. Red flag: proposing `DISTINCT` as the general repair.

### Q3. Why are coverage and multiplicity not interchangeable? Give the counterexample.

**Answer.** Coverage counts survival over all input facts; multiplicity counts replication over matched facts only. With match counts 2, 1, 0 the output row count equals the input row count of 3, yet coverage is 2/3 and multiplicity is 3/2. Neither ratio implies the other.

**Reasoning.** The two ratios have different denominators by construction, and their product is the global average match count including zeros. Any query can trade loss against duplication and preserve the product.

**Follow-up.** What additional check distinguishes "matched once each" from "one lost, one tripled"?

**Acceptable alternative.** Reporting the unmatched-key count and the per-key duplicate distribution alongside the ratios is strictly more informative and fully acceptable.

**Scoring and red flag.** Using `|output| / |input|` scores 2. Naming both denominators scores 4. Producing the 2/3, 3/2 example and the compensating-error conclusion scores 5. Red flag: believing that equal row counts certify a join.

### Q4. When is `SELECT DISTINCT` a legitimate response to fan-out?

**Answer.** When the question is genuinely about set membership at the projected grain, and the projected columns preserve the entity identity the question requires. It is never a repair for a wrong relationship.

**Reasoning.** `DISTINCT` removes duplicate output tuples. If the duplicates differ in any projected column — as the two customer versions do — it removes nothing. If they are identical, it hides the duplication without restoring lost facts or fixing the key.

**Follow-up.** In the six-row example, what does `SELECT DISTINCT line_id, amount_cents` produce and why is it worse than the naive sum?

**Acceptable alternative.** Deduplicating exact retries by producer identity plus payload hash is a legitimate use, because it targets a real identity rule rather than a symptom.

**Scoring and red flag.** "Always use DISTINCT to be safe" scores 0. Stating the projection-grain condition scores 3. Adding that it cannot restore loss scores 5. Red flag: applying `DISTINCT` to the whole select list as a defensive habit.

### Q5. What must a bridge-table contract contain before an additive report may use it?

**Answer.** Both endpoint keys, the relationship's own effective time where relevant, the allocation meaning, nonnegative weights, a per-fact weight sum of one, the weight version, and a conservation assertion in the build.

**Reasoning.** A bridge's grain is the relationship instance, not the fact. Additive reporting across it requires an explicit partition of each fact's measure, and only the weights encode that partition.

**Follow-up.** Weights sum to 1.4 for one fact. What do you publish, and what do you say?

**Acceptable alternative.** Publishing a non-additive association measure — "revenue associated with each tag" — with the label and the non-additivity stated is fully correct and requires no weights.

**Scoring and red flag.** "Join and use DISTINCT" scores 0. Naming weights and conservation scores 4. Adding the versioned policy and the association alternative scores 5. Red flag: reconciling a full-credit tag report against company revenue.

### Q6. How do you test a type-2 dimension, and what does each failure look like?

**Answer.** Assert no overlapping intervals per natural key, no gaps inside the reporting horizon, and exactly one as-of match (`k(f) = 1`) for every required fact. Overlap gives `k(f) = 2`, which is duplication. A gap gives `k(f) = 0`, which is loss. Both are silent in the total across all regions if other facts compensate.

**Reasoning.** Half-open intervals are what make the boundary instant belong to exactly one version; closed intervals assign it twice and open intervals assign it never.

**Follow-up.** A fact sits exactly at a version boundary. Which region should it receive?

**Acceptable alternative.** A source-sequence selector is valid if the correction semantics are documented and the ordering is total, including a tie-breaker.

**Scoring and red flag.** `MAX(region)` scores 0. Naming overlap and gap checks scores 4. Adding the half-open rationale and `k(f) = 1` as the gate scores 5. Red flag: choosing a version by attribute value rather than by time.

### Q7. Normalized model or star schema? Defend the choice.

**Answer.** Normalize where writes are frequent and entity ownership is local, because update anomalies are the dominant risk. Build a dimensional mart where a small set of analytical questions repeats, because shallow joins make grain and cardinality visible. The two are complementary layers, not competitors.

**Reasoning.** The choice is a statement about mutation frequency and question stability, and about where cardinality contracts are cheapest to enforce and cheapest to verify.

**Follow-up.** What new contract does the mart need that the normalized core did not?

**Acceptable alternative.** A governed wide table is defensible when its refresh, grain, lineage and uniqueness assertions are documented and tested.

**Scoring and red flag.** "Star is faster" scores 1. Framing it as workload and mutation trade-off scores 4. Adding the presentation-layer grain assertions scores 5. Red flag: treating normalization as a correctness guarantee.

### Q8. What is a semi-additive measure, and what goes wrong if you ignore it?

**Answer.** A measure additive across some dimensions but not over time — an end-of-day inventory balance is the canonical case. Summing daily balances produces balance-days, not a balance, so a quarter's figure comes out roughly the day count times too large.

**Reasoning.** Additivity is a property of a measure with respect to a dimension, not of its numeric type. The declared aggregator must be `last`, `average`, or an integral with a stated unit, chosen by the decision the number supports.

**Follow-up.** Which time rule would you choose for a working-capital decision, and why?

**Acceptable alternative.** Integrating balance over time is valid when the resulting unit is explicitly balance-time and the consumer wants exposure rather than a level.

**Scoring and red flag.** Treating every numeric column as additive scores 0. Naming the unit problem scores 4. Choosing a rule by decision context scores 5. Red flag: a dashboard whose default aggregator is `SUM` for all measures.

### Q9. A metric drops 20% the day a dimension join ships. Walk through your first thirty minutes.

**Answer.** Freeze interpretation and stop attributing the change to the business. Compare fact-key coverage before and after, list anti-join keys, inspect predicate placement (ON versus WHERE), confirm both relations' snapshot identifiers, and reconcile the pre-join total to the certified source. Only after those five checks may the residual be treated as business movement.

**Reasoning.** Business decline and accidental exclusion produce identical chart shapes, and the checks that separate them are cheap and ordered from most to least likely.

**Follow-up.** Coverage is 0.98 and multiplicity is 1.0. Is the report safe for a required dimension?

**Acceptable alternative.** Publishing the prior certified value with an explicit "under repair" label while diagnosis runs is a legitimate controlled degradation, provided it has an owner and a deadline.

**Scoring and red flag.** Offering a business explanation first scores 0. Naming reconciliation and coverage scores 4. Ordering the checks and stating the release decision scores 5. Red flag: any answer that begins with "probably seasonality".

### Q10. How do you compute average order value, and why is the obvious query wrong?

**Answer.** `SUM(order_header_amount) / COUNT(DISTINCT order_id)` at order grain. The obvious `AVG(line_amount)` at line grain is a line average, which weights large baskets more heavily and answers a different question.

**Reasoning.** The denominator defines the unit of the ratio. An average of line amounts is an average line, not an average order, and the two diverge in proportion to the variation in basket size.

**Follow-up.** A line-grain table carries the order amount on every line. What happens to `SUM` and to `AVG` of that column?

**Acceptable alternative.** Pre-aggregating to order grain in a mart and averaging that column is equivalent and often cheaper.

**Scoring and red flag.** `AVG(line_amount)` scores 1. Correct numerator and denominator scores 4. Explaining the weighting consequence scores 5. Red flag: dividing by `COUNT(*)` on a joined relation.

### Q11. A colleague proposes `NOT IN (SELECT customer_id FROM dim_customer)` to find orphan facts. What is wrong?

**Answer.** If `dim_customer.customer_id` contains any null, `NOT IN` evaluates to unknown for every row and returns the empty set, so the orphan check silently passes. Use `NOT EXISTS` or a left anti-join, and count null keys explicitly.

**Reasoning.** Three-valued logic makes `x NOT IN (…, NULL, …)` unknown rather than true for every `x`. The failure mode is a check that reports success precisely when the data is dirtiest.

**Follow-up.** Does `NOT EXISTS` behave differently with nulls, and why?

**Acceptable alternative.** A left join with `WHERE d.key IS NULL` is correct provided the join key itself is non-nullable or nulls are counted separately.

**Scoring and red flag.** Not knowing the null behaviour scores 1. Naming `NOT EXISTS` and the reason scores 4. Adding the null-key counting discipline scores 5. Red flag: shipping a data-quality check that can only return zero rows.

### Q12. Two dashboards disagree by 4% on the same metric. Give your diagnostic order.

**Answer.** Compare grain declarations first, then population predicates, then snapshot identifiers and cutoffs, then version rules, then units, and only last the arithmetic. Four percent is small enough to be a version or cutoff difference and large enough to be a missed dimension slice.

**Reasoning.** Differences that survive a grain and population comparison are almost always temporal or definitional, and those are cheap to isolate by reading the two contracts side by side.

**Follow-up.** Both dashboards reconcile to the ledger within tolerance. What does that tell you?

**Acceptance alternative.** A reconciliation-driven approach that starts from the certified source and works outward is equally valid and often faster.

**Scoring and red flag.** "Rerun both and take the newer" scores 0. Ordered comparison of contracts scores 4. Adding the ledger reconciliation test scores 5. Red flag: assuming the newer dashboard is correct.

### Q13. What is the difference between a wrong cardinality and a wrong key, and which is harder to detect?

**Answer.** Wrong cardinality shows up in coverage or multiplicity. A wrong key — a one-to-one join on an attribute that is unique but semantically incorrect — passes both checks and produces clean-looking wrong attributions. It is harder because no structural test can see it; only a semantic review of key definitions and a sample audit against a second source can.

**Reasoning.** Structure constrains the multiplicity of matches; it says nothing about which pair should have matched. This is why the release gate has four parts, not two.

**Follow-up.** Design the audit. What sample, and compared against what?

**Acceptable alternative.** A contract-level lineage review that traces the key back to its source definition is a valid substitute where sampling is impractical.

**Scoring and red flag.** Claiming coverage catches it scores 0. Distinguishing the two failure classes scores 4. Designing the semantic audit scores 5. Red flag: equating referential integrity with correctness.

### Q14. A strict latency limit rules out the join at query time. What changes, and what does not?

**Answer.** Pre-compute a governed mart or cached aggregate with recorded snapshot, refresh contract, and lineage. What does not change is the correctness requirement: the pre-aggregation must still satisfy the grain-preservation condition, and the cached artifact must carry its snapshot identifiers so a reader can tell two vintages apart.

**Reasoning.** Latency is a delivery constraint, not a licence to relax cardinality. Pre-aggregation moves the join earlier, where it must be tested identically.

**Follow-up.** The bridge's edge count doubles after a tagging initiative. What breaks first?

**Acceptable alternative.** Serving a cached certified core with an explicitly labelled stale slice for optional dimensions is acceptable if the label and freshness bound are visible to the consumer.

**Scoring and red flag.** "Add indexes" alone scores 1. Naming a versioned pre-computed artifact scores 4. Adding grain preservation and snapshot labelling scores 5. Red flag: caching without recording the cutoff.

### Q15. Your model passes every structural check and a director asks why they should believe the number. What do you say?

**Answer.** Name the grain, the population, the snapshot, the cardinality contract per hop, the measured coverage and multiplicity, the reconciliation to the source of record and its residual, the semantic review of the join keys, and the one known limitation. Then state what would make you retract it.

**Reasoning.** A number is defensible when its inputs and controls are enumerable and a second person could reproduce it. Structural checks alone are necessary and insufficient, and saying so is stronger than claiming completeness.

**Follow-up.** What evidence would make you block the release tomorrow?

**Acceptable alternative.** Presenting the same information as a signed release record with the four gate results and their timestamps is equivalent and more auditable.

**Scoring and red flag.** "It reconciles" scores 1. Enumerating grain, ratios and reconciliation scores 4. Adding semantic review, limitation and a retraction condition scores 5. Red flag: claiming the checks prove correctness rather than constraining structure.

**Takeaway:** The questions that separate candidates are the ones about independence — of loss from inflation, of structure from population, and of cardinality from key semantics.

## 12. Quick Reference Card

| Item | Rule | When to use | Critical assumption | Failure it prevents |
|---|---|---|---|---|
| Grain declaration | Identity tuple + measure unit + time basis + mutation rule + population | Before any modelling or query | A second person can compute the expected row count from it | Every ambiguity that surfaces later as a "discrepancy" |
| Expected cardinality | Predict `|J| = sum_f m(f)` from key definitions | Before writing the join | Key uniqueness can be checked at the build snapshot | Shipping a join whose inflation is discovered in a review |
| Coverage | `c_join = |M| / |F|` | Every build on a required path | Same snapshot and predicate as the aggregate | Silent loss of facts that never reach a total |
| Multiplicity | `m_join = |J| / |M|` | Every build on a required path | Joined rows retain fact-key identity | Silent replication that inflates totals |
| Semi-join for existence | `F ⋉ D` instead of `F ⋈ D` when only testing presence | Any filter-by-membership need | Right side contributes no columns | Accidental fan-out from a non-unique filter table |
| As-of join | Half-open `[from, to)` with `k(f) = 1` asserted | Type-2 dimensions | Intervals partition the horizon | Double attribution at boundaries, gapped loss |
| Bridge allocation | `w ≥ 0`, `sum_e w(f,e) = 1`, weights versioned | Many-to-many additive reporting | A defensible weighting basis exists | Tag totals that exceed and cannot reconcile to company totals |
| Population reconciliation | Compare to the source of record at a named snapshot | Before every release | The source is authoritative for the population | Structurally perfect queries over the wrong population |
| Diagnostic order | Coverage, multiplicity, anti-join keys, predicate placement, snapshot, then arithmetic | Any unexplained change | Structural checks precede semantic reasoning | Attributing an arithmetic defect to the business |

**Takeaway:** Declare the grain, predict the cardinality, measure both ratios, allocate explicitly, reconcile to the source — and attach all five to the number, because a metric without its contract is a guess with a decimal point.
