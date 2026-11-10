---
collection: "python-sql-linux-engineering-basics"
file_id: "05"
topic: "SQL analytics patterns"
level: 1
status: "draft"
version: "1.0"
created: "2026-11-09"
reviewed: null
review_trigger: "Schema, row grain, metric denominator, time-zone convention, or engine version change"
notation_style: "plain-text"
prerequisites: ["table and key concepts", "GROUP BY aggregation", "ratio arithmetic"]
pattern_tags: ["joins", "windows", "metric-grain", "deduplication", "time-semantics"]
source_status: "OPEN — course-topic draft; no executed database, no vendor benchmark, no source body asserted"
owner_role: "analytics engineer"
source_index: "INDEX.md"
---

# SQL Analytics Patterns: Grain, Joins, Windows, and Metric Defence

## 1. File Header

> **Purpose:** Turn a business question into an analytical SQL statement whose row grain, join cardinality, time boundary, and metric denominator are declared, checkable, and reconcilable — and diagnose the query when the number moves.
> **Prerequisites:** `SELECT`/`WHERE`/`GROUP BY`, primary-key vs foreign-key vocabulary, and comfort with percentage arithmetic (numerator over denominator, same unit).
> **Key outcome:** Given two or three related tables and a metric definition, you can (a) state the output grain in one sentence, (b) predict the row count produced by each join before running it, (c) choose between conditional aggregation, `EXISTS`, ranking functions, and window frames, and (d) detect a duplicated or mis-grained denominator from arithmetic evidence alone.
> **This file does NOT cover:** physical optimizer internals and hint tuning (handoff: engine documentation and `EXPLAIN` interpretation), transaction isolation and locking (handoff: OLTP course), warehouse cost governance and role-based access policy, and orchestration/scheduling (handoff: data-engineering module).
> **Level:** 1 Foundations, with Level 2 selection reasoning and Level 3 production diagnosis.
> **Version / reviewed:** 1.0 / not independently reviewed; every number below is DERIVED arithmetic over ILLUSTRATIVE rows, not a measurement.
> **Evidence:** DERIVED relational reasoning and arithmetic (recomputed by hand and by an independent script); ILLUSTRATIVE tables and workloads; OPEN engine-specific behaviour, which must be rechecked against the target engine's version.
> **Retrieval cue:** If a KPI changed after a join or a window function was added, start at Section 4.4 and Section 8. If the KPI is stable but wrong in a way nobody can explain, start at Section 5.

### 1.1 What decision this sheet supports

An analyst rarely fails because a keyword is unknown. The failure is almost always earlier: the query answered a different question than the one asked, and the answer looked plausible. The decision this sheet supports is therefore not "how do I write a window function" but "which row does one line of my result represent, and can I prove that the numerator and denominator were counted at the same grain and over the same time window?"

Three signals tell you that the sheet is relevant to the situation in front of you:

1. A number is reported to one decimal place to a person who will spend money on it, and nobody can say how many rows the source join produced.
2. Two teams report the same metric and disagree, and both queries "look right".
3. A dashboard value moved on a day with no product change, or failed to move on a day with one.

### 1.2 Dialect stance, stated honestly

The SQL shown is written in PostgreSQL-flavoured standard SQL because it is the most widely readable dialect and it implements the window and `FILTER` clauses of the SQL standard. Where a construct is not portable, the text names the difference in place rather than hiding it. The patterns themselves — declared grain, cardinality prediction, event-time reconciliation, explicit frames — are engine-independent; only the syntax is not. Nothing in this file was executed against a live database, so the reader must verify syntax and exact null/ordering behaviour on the intended engine before relying on it.

### 1.3 Reading order

Read Sections 2 and 3 once for vocabulary. Then read Section 4 in order, because 4.2 and 4.3 reuse the base-case result and 4.4 is the counterexample set that makes the earlier results safe to use. Section 5 is the single end-to-end example; every later section refers back to its row counts. Sections 7 and 8 are lookup material. Sections 9 and 10 explain why the cost and correctness claims hold. Sections 11 and 12 are practice and recall.

**Takeaway:** the deliverable of an analytical query is an auditable claim about a counted population, and the grain sentence is the first line of that audit.

## 2. Concept Map / Mental Model

### 2.1 The five contracts of an analytical query

Every analytical query you will be asked to defend carries five contracts. Four of them are usually implicit, and the fifth — the number — is the only one anybody looks at.

```text
(1) GRAIN CONTRACT
    "one output row = one ___"
        |
        v
(2) CARDINALITY CONTRACT
    rows_out = rows_in * fan_out ; fan_out must be 1 for entity metrics
        |
        v
(3) TIME CONTRACT
    event time vs processing time; window [t0, t1); late-arrival policy
        |
        v
(4) MEMBERSHIP CONTRACT
    which entities are eligible, and are numerator entities a subset of them?
        |
        v
(5) NUMERIC CONTRACT
    numerator / denominator at the same grain; units cancel; bounded 0..1 for a rate
        |
        v
    [reconciliation: does the number reproduce from entity IDs by hand?]
```

The arrows are dependencies, not execution order. A broken cardinality contract silently corrupts the numeric contract, and the numeric contract is the only one with a visible symptom. That asymmetry is why Section 8 teaches diagnostics that look at row counts rather than at the final ratio.

### 2.2 Logical processing order, and why it bites

SQL is written in one order and evaluated in another. The difference explains a large share of "impossible" bugs.

```text
written order                     logical evaluation order
---------------                   -------------------------
SELECT                            FROM / JOIN        <- row multiplication happens here
  ...                             WHERE              <- before grouping; cannot use aggregates
FROM ... JOIN ...                 GROUP BY           <- collapses to one row per group
WHERE ...                         HAVING             <- filters groups, after aggregation
GROUP BY ...                      window functions   <- computed after grouping, before DISTINCT
HAVING ...                        SELECT             <- aliases exist only from here down
ORDER BY ...                      DISTINCT
LIMIT ...                         ORDER BY           <- may reference SELECT aliases
                                  LIMIT / OFFSET
```

Four consequences worth memorising:

- A `WHERE` predicate on the right-hand table of a `LEFT JOIN` executes before the join's null-extension is observable, so it removes the unmatched rows. The join was never the problem; the filter turned it into an inner join.
- Window functions cannot appear in `WHERE`. Filtering a window result requires a subquery or CTE, which is why "top N per group" is a two-step pattern.
- `SELECT` aliases are not visible in `WHERE` on most engines; they may be visible in `GROUP BY`, `HAVING`, and `ORDER BY` depending on dialect. Do not rely on the extension.
- `DISTINCT` runs after window functions. A window computed over duplicated rows returns duplicated window values, and `DISTINCT` will not repair them.

### 2.3 Map from question type to pattern family

| Question the stakeholder actually asked | Correct pattern family | Section |
|---|---|---|
| "What fraction of eligible X did Y?" | Grain-declared ratio with a shared cohort | 4.1, 5 |
| "How did that fraction move between two periods?" | Same ratio over two disjoint windows, with a composition check | 4.2, 7 |
| "Which is the latest state of each entity?" | Dedup by ranking, then join to the entity | 4.3, 7 |
| "What is the running total / 7-day average?" | Window aggregate with explicit frame and tie-breaker | 4.3, 6 |
| "Where do users drop out of the flow?" | Funnel by ordered step membership, not by row counting | 4.3, 5 |
| "Do users come back?" | Cohort denominator fixed at cohort entry, horizon measured forward | 4.3, 5 |
| "Is the metric different for group A versus B?" | Stratified comparison plus standardisation check | 4.2, 8 |
| "Why did the number change?" | Cardinality and membership reconciliation before re-derivation | 8 |

The mapping matters because choosing a pattern by syntax familiarity — "I'll use a window function because it's fast" — is how a funnel becomes a count of events rather than a count of users.

### 2.4 The central invariant

For any entity-level metric,

```text
INVARIANT [GRAIN-1]: 0 <= numerator <= denominator, and
                     every entity counted in the numerator is counted in the denominator.
```

This is not a stylistic preference. It follows from the numerator being defined as a subset of the eligible set. When you see a conversion rate of 112%, one of three things has happened: the numerator was counted at a finer grain than the denominator, the denominator was filtered more aggressively than the numerator, or the two were computed over different time windows. The invariant is the fastest available detector of all three, and it costs one line of arithmetic to check.

**Takeaway:** the visible number is the last contract to break and the first to be noticed; diagnose upstream at grain and cardinality.

## 3. Foundational Definitions

### 3.1 Core vocabulary

| Term | Formal meaning | Units / shape | Valid domain | Common confusion | Operational check |
|---|---|---|---|---|---|
| Grain | The real-world unit that one output row represents, stated as a noun phrase | categorical | one noun phrase, no "and" | Confused with `GROUP BY` list | Say the sentence aloud; if it contains "and", you have two grains |
| Candidate key | A column set that is unique and non-null for every row of a relation | set of columns | at least one per table | Assumed to be the primary key | `COUNT(*) = COUNT(DISTINCT key_cols)` on the whole table |
| Fan-out (join multiplier) | The number of output rows produced per input row of the preserved side of a join | rows/row, ≥0 | 1 for entity metrics | Called "duplicates" without measuring | Pre-join and post-join row counts, plus distinct-key counts |
| Event time | The instant the business event occurred, as recorded in the event payload | timestamp with zone | within retention window | Used interchangeably with load time | Compare `event_ts` to `ingest_ts` distribution |
| Processing time | The instant the system persisted or observed the event | timestamp with zone | later than event time, usually | Treated as truth for period assignment | Difference `ingest_ts - event_ts` is the lateness distribution |
| Peer | A row with an equal value of the window's `ORDER BY` expression | relation | any | Ignored when writing `RANGE` | Count rows per `(partition, order_value)`; peers > 1 exist? |
| Frame | The subset of a partition's rows visible to the current row's window computation | row set | non-empty by default clause | Left to the engine default | Write the frame explicitly, every time |
| Cohort | A set of entities defined by a shared start event inside a fixed interval | set of entity IDs | disjoint by construction date | Mixed with "segment" | Can you list the member IDs at `t0` and prove none joined later? |
| Denominator | The count of eligible units over the declared window | units | ≥ numerator, same unit | Silently switched between runs | Print numerator, denominator, and their difference |
| Dimension | An attribute used to group or filter, at or above the fact grain | categorical | attached to the entity, not the event | A fact column used as a dimension | Does the value change per event? Then it is not a stable dimension |
| SCD-2 row | A versioned dimension row valid over `[valid_from, valid_to)` | interval per entity | non-overlapping | Joined to facts without an as-of predicate | Intervals overlap or have gaps? Then the versioning is broken |
| Three-valued logic | `TRUE`, `FALSE`, `UNKNOWN` where `UNKNOWN` arises from `NULL` comparison | truth value | any expression | `NULL = NULL` assumed true | Any `NOT IN` over a nullable column, any `<>` filter |
| Semi-join | Rows of the left side that have at least one match on the right, without duplicating left rows | relation | subset of left | Implemented with `INNER JOIN` + `DISTINCT` | Row count after semi-join equals left row count at most |
| Anti-join | Rows of the left side with no match on the right | relation | subset of left | Implemented with `LEFT JOIN ... IS NULL` plus a stray filter | Same as above, plus null-behaviour of the key |
| Grain-preserving aggregate | An aggregate that does not change the row count of its input, e.g. a window `SUM` | relation | preserves cardinality | Confused with `GROUP BY` | Output rows = input rows? |

### 3.2 Grain, in the only form that is testable

A grain statement is testable when it names the entity, the time scope, and the filter that defines membership. "Daily revenue by store" is not a grain; it is a grouping. "One row per (store, calendar day in the store's local time zone), covering days with at least one settled transaction" is a grain. The second version tells you what happens on a day with zero transactions (the row is absent, not zero), which is exactly the kind of question that produces a wrong chart later.

Two tests establish that you have the grain you claim:

```text
TEST [GRAIN-A]: COUNT(*) = COUNT(DISTINCT grain_key_1, grain_key_2, ...)
```

on the final result set of the query, evaluated with no `LIMIT`. If this fails, the query produces more than one row per claimed unit and every downstream ratio is suspect. On engines without multi-column `DISTINCT` in `COUNT`, hash the concatenation with a separator that cannot appear in the components, or group and count groups.

```text
TEST [GRAIN-B]: the grain key columns are NOT NULL in the result.
```

A nullable component of a grain key is a silent cardinality bug: in a `GROUP BY`, all `NULL`s form one group, so "one row per customer" becomes "one row per customer, plus one row for all unattributed customers", and that extra row can dominate the total.

### 3.3 Keys, uniqueness, and the difference between declared and actual

A declared primary key is a promise by the schema author. An actual uniqueness property is a measurement. Analytical queries need the measurement, because ETL loads, late-arriving corrections, and unioned sources routinely violate declarations that were true last year.

```text
CHECK [KEY-1]: SELECT key_cols, COUNT(*) AS n
               FROM t GROUP BY key_cols HAVING COUNT(*) > 1
               ORDER BY n DESC LIMIT 20;
```

Running this before writing a join is cheap relative to the cost of a wrong quarterly number, and the top-20 output tells you the *shape* of the violation: one key repeated twice suggests a retried load; ten thousand keys repeated twice suggests a missing `DISTINCT` upstream; one key repeated fifty thousand times suggests an aggregate row was joined to a detail table.

### 3.4 Event time, processing time, and the reporting boundary

The same event can be assigned to two different periods depending on which timestamp you use. Consider an order placed at `2026-01-31 23:50` local time in a zone offset `+05:30`, ingested at `2026-02-01 00:04` UTC. Under event time it belongs to January; under processing time it belongs to February. Both are defensible; mixing them within one report is not.

Three rules make the choice safe:

1. Declare one timestamp per metric as authoritative for period assignment, and name the time zone. Store timestamps with zone information, never as naked local strings.
2. Define the reporting window as half-open, `[t0, t1)`. Closed intervals double-count the boundary instant, and boundary instants are where batch jobs and late events collide.
3. State the lateness policy. If events may arrive after the close, either (a) restate the closed period when late data lands, or (b) report a provisional value with the cutoff time printed next to it. Silence on this point is the single most common source of "the dashboard changed by itself" complaints.

### 3.5 NULL and three-valued logic, because it decides join answers

`NULL` is not a value; it is the absence of one. Consequently `NULL = NULL` evaluates to `UNKNOWN`, not `TRUE`, and `WHERE` keeps only rows whose predicate is `TRUE`.

| Predicate | Result when `x` is `NULL` | Row survives `WHERE`? |
|---|---|---|
| `x = 5` | `UNKNOWN` | No |
| `x <> 5` | `UNKNOWN` | No |
| `x IS NULL` | `TRUE` | Yes |
| `NOT (x = 5)` | `UNKNOWN` | No |
| `x IN (1, 2, NULL)` when `x = 3` | `UNKNOWN` | No |
| `x NOT IN (1, 2, NULL)` when `x = 3` | `UNKNOWN` | No |
| `COALESCE(x, 0) = 0` when `x` is `NULL` | `TRUE` | Yes |

Two practical consequences dominate analytics work. First, `NOT IN (subquery)` returns no rows at all if the subquery yields a single `NULL`; use `NOT EXISTS` for anti-joins. Second, a predicate like `status <> 'cancelled'` silently drops rows with unknown status, which is often the population you most wanted to see. The fix is explicit: `status IS DISTINCT FROM 'cancelled'` where the engine supports it, or `(status <> 'cancelled' OR status IS NULL)` when the null rows genuinely belong in scope.

### 3.6 Aggregate, window, and the word "distinct"

Three operations are routinely conflated because they all "collapse rows":

```text
GROUP BY        : n input rows -> g output rows (one per group)
window function : n input rows -> n output rows (value attached per row)
DISTINCT        : n input rows -> d output rows (unique tuples of the selected list)
```

`COUNT(DISTINCT x)` is a fourth thing again: it is an aggregate whose argument is deduplicated before counting, and its result cannot be added across partitions without error (see 4.4). When somebody says "I deduplicated the rows", ask which of the four they mean, because `DISTINCT *` on a table with a timestamp column removes nothing at all.

**Takeaway:** grain, key uniqueness, time semantics, null behaviour, and the collapse operation are five separate contracts, and every wrong analytical number violates at least one of them.

## 4. Core Formulas with Full Derivation

Section 4 is organised as a ladder. 4.1 derives the ratio and the cardinality condition that makes it valid. 4.2 generalises to strata, windows, and composition change, which is where most real reporting lives. 4.3 collects the variants that a working analyst actually types — deduplication, ranking, frames, cohorts, funnels, semi-joins. 4.4 is the counterexample set: each result is paired with a boundary at which naive use returns a confident wrong number.

Throughout, `e` denotes an eligible unit, `c` a converted unit, `n` a row count from a relation, and `m` a join multiplier. Units are always stated; a ratio of counts is dimensionless, a ratio of currencies is not, and mixing the two is a defect, not a rounding issue.

### 4.1 Base case: the declared ratio and its cardinality precondition

**Validity before use.** The ratio applies when (i) numerator and denominator count the same kind of unit, (ii) the numerator population is a subset of the denominator population, (iii) both are evaluated over the same half-open time window, and (iv) each unit contributes at most one to each count after the declared reconciliation rule. Conditions (i)–(iii) are definitional. Condition (iv) is the one SQL breaks silently, so it gets the derivation.

**Derivation.** Start from the definition of a proportion over a finite population `E` of eligible units with indicator `y_e ∈ {0,1}`:

```text
[1] r = (1/|E|) * sum over e in E of y_e
```

Let `c = sum y_e` be the converted count and `e_count = |E|` the eligible count. Then `r = c / e_count` exactly, with no approximation. Now suppose the query computes `c` and `e_count` by counting rows of a relation `R` obtained from `E` by a one-to-many join with multiplicity `m_e >= 1`. Row-counting gives `c_hat = sum m_e y_e` and `e_hat = sum m_e`. Then

```text
[2] r_hat = sum m_e y_e / sum m_e
```

Equation [2] is a *weighted* mean of `y_e` with weights `m_e`, whereas [1] is the unweighted mean. They coincide if and only if `m_e` is constant across `E`, or if `m_e` is independent of `y_e` in the weighted-mean sense. Neither holds in general: line-item counts correlate with order size, and order size correlates with completion. Hence row-counting after fan-out is a different estimand, not a noisier version of the same one.

```text
FORMULA [SQL-01]: r = c / n_eligible
WHERE: c converted units [units]; n_eligible eligible units [units];
       r dimensionless in [0,1]; both counts over the same half-open window [t0,t1).
CONSTRAINT: numerator population subset of denominator population; one count per unit
            after reconciliation; no row-level fan-out from joins at this grain.
NUMERICAL EXAMPLE: c = 2 orders, n_eligible = 5 orders;
                   r = 2/5 = 0.4 = 40%.
INTERPRETATION: 40% of the eligible orders converted; the other 60% are the
                population that any remediation must move.
```

```text
FORMULA [SQL-02]: bias = r_hat / r = (sum m_e y_e / sum m_e) / (sum y_e / |E|)
WHERE: m_e join multiplicity of unit e [rows/unit]; r_hat row-counted rate;
       r true unit rate; bias dimensionless, defined when r > 0.
CONSTRAINT: same population on both sides; m_e >= 1; no filter applied after the join.
NUMERICAL EXAMPLE: two converted orders with 2 and 3 line items, three unconverted
                   orders with 1, 1, 0 line items (the zero row is absent from the join):
                   r_hat = (2+3)/(2+3+1+1) = 5/7 = 0.7143; r = 2/5 = 0.4;
                   bias = 0.7143/0.4 = 1.786.
INTERPRETATION: the row-counted number overstates the unit rate by 79%, in the
                direction of whichever units have more child rows.
```

The bias formula also tells you *how* to detect the defect without knowing the truth: compute both `COUNT(*)` and `COUNT(DISTINCT unit_key)` on the joined relation. If they differ, `m_e` is not identically 1 and every ratio built on `COUNT(*)` in that query is contaminated.

```text
CHECK [CARD-1]: rows_joined / rows_preserved = average fan-out  [rows/row]
```

An average fan-out of 1.0 with a maximum of 40 is still a fan-out problem; always inspect the maximum per-key count, not only the mean, because a few heavily-weighted units can move the metric on their own.

### 4.2 General case: strata, windows, and composition change

The general case adds three things to 4.1: the population is partitioned into strata, the comparison spans two windows, and the metric must be attributed to one of them. The derivation separates the change into a within-stratum effect and a mix effect, which is the only way to answer "did we improve or did our traffic change?".

**Validity before use.** Strata must be defined on attributes measured *before* or *independently of* the outcome, must be mutually exclusive and exhaustive within each window, and must use identical definitions in both windows. A stratum key derived from the outcome (for example, "segment = completed orders") manufactures arithmetic that looks like an insight but is an identity.

**Derivation.** Let strata `k = 1..K`, with eligible counts `e_k`, converted counts `c_k`, weight `w_k = e_k / e` where `e = sum_k e_k`, and rate `r_k = c_k / e_k`. Then

```text
[3] r = c/e = sum_k c_k / sum_k e_k = sum_k (e_k/e)(c_k/e_k) = sum_k w_k r_k
```

which is exact because `sum_k w_k = 1`. Writing `Δr_k = r_k' - r_k` and `Δw_k = w_k' - w_k` for the second window and subtracting gives

```text
[4] Δr = sum_k w_k Δr_k + sum_k r_k Δw_k + sum_k Δw_k Δr_k
```

Equation [4] is an algebraic identity, not a model. The three terms are the within-stratum effect at old weights, the mix effect at old rates, and an interaction term that vanishes only when either the rates or the weights are unchanged. Reporting the first term as "our improvement" and silently dropping the other two is the most common quantitative misstatement in product analytics.

```text
FORMULA [SQL-03]: r = sum_k w_k r_k,  w_k = e_k/e
WHERE: r overall rate [dimensionless]; r_k stratum rate; w_k stratum weight
       [dimensionless, sum to 1]; K number of strata.
CONSTRAINT: strata exhaustive and disjoint within each window; identical stratum
            definitions across compared windows; e_k > 0 for every included k.
NUMERICAL EXAMPLE: K=2; e_1=200, e_2=100, c_1=180, c_2=20;
                   w=(0.6667, 0.3333); r_1=0.9, r_2=0.2;
                   r = 0.6667*0.9 + 0.3333*0.2 = 0.6 + 0.0667 = 0.6667.
                   Direct check: 200/300 = 0.6667. Agrees.
INTERPRETATION: the overall figure is a weighted mean, so it can move while every
                stratum moves the other way.
```

```text
FORMULA [SQL-04]: standardized rate = sum_k w_k' r_k
WHERE: w_k' weights from the comparison window; r_k rates from the base window;
       result dimensionless, same scale as r.
CONSTRAINT: weights must be non-negative and sum to 1; standardisation is a
            decomposition device, not a forecast of what would have happened.
NUMERICAL EXAMPLE: w' = (0.3333, 0.6667), r = (0.9, 0.2);
                   standardized = 0.3333*0.9 + 0.6667*0.2 = 0.3 + 0.1333 = 0.4333.
INTERPRETATION: at the new traffic mix, the old rates would have produced 43.3%,
                so most of the observed fall is mix, not performance.
```

The composition decomposition is worth one worked check because the sign convention is easy to invert. Using the two windows `P1: e=(200,100), c=(180,20)` and `P2: e=(100,200), c=(95,60)`:

| Quantity | P1 | P2 | Difference |
|---|---|---|---|
| Eligible `e` | 300 | 300 | 0 |
| Converted `c` | 200 | 155 | −45 |
| Rate `r_1` | 0.900 | 0.950 | +0.050 |
| Rate `r_2` | 0.200 | 0.300 | +0.100 |
| Weight `w_1` | 0.6667 | 0.3333 | −0.3333 |
| Overall `r` | 0.6667 | 0.5167 | −0.1500 |

Every stratum improved; the overall rate fell 15 points. The identity [4] explains it: mix effect `= 0.9(−0.3333) + 0.2(+0.3333) = −0.3000 + 0.0667 = −0.2333`; within-stratum effect at old weights `= 0.6667(0.05) + 0.3333(0.10) = 0.0333 + 0.0333 = 0.0667`; interaction `= (−0.3333)(0.05) + (0.3333)(0.10) = −0.0167 + 0.0333 = 0.0167`. Sum: `−0.2333 + 0.0667 + 0.0167 = −0.1500`, matching the observed change exactly. That agreement is the check: if your decomposition does not reproduce the total, you have mis-specified a weight.

**Windowed membership.** The general case also needs an explicit window. For a metric evaluated at a cutoff `T` with lateness allowance `L`:

```text
FORMULA [SQL-05]: eligible if event_ts >= t0 AND event_ts < t1 AND ingest_ts < T
WHERE: event_ts business instant; ingest_ts persistence instant; t0,t1 window bounds;
       T report cutoff; all timestamps with zone, stored in UTC.
CONSTRAINT: half-open window; both bounds on the same clock; T - t1 = L is the
            lateness allowance and must be printed with the number.
NUMERICAL EXAMPLE: t1 = 2026-03-01T00:00Z, T = 2026-03-01T06:00Z, L = 6 h.
                   An event with event_ts 2026-02-28T23:40Z and ingest_ts
                   2026-03-01T05:10Z is included; the same event ingested at
                   2026-03-01T07:00Z is excluded from this run.
INTERPRETATION: the metric is "as of 06:00Z", a statement that makes the number
                reproducible instead of mysteriously self-updating.
```

### 4.3 Key special cases and variants

Each variant below changes exactly one thing relative to its parent. The `avoid when` column is the load-bearing part: every one of these patterns has a condition under which it returns a plausible but wrong number.

| Variant | What changes | Assumptions | Use when | Avoid when | ID |
|---|---|---|---|---|---|
| Conditional aggregation | One pass, `SUM(CASE WHEN ...)` over multiple predicates | Aggregates share a grain and window | Several rates over the same population | Predicates select different populations | V-01 |
| Ratio of sums | `SUM(a)/SUM(b)` rather than `AVG(a/b)` | Denominators differ per row | Row weights differ (traffic, exposures) | Every row has equal weight by design | V-02 |
| Dedup by ranking | `ROW_NUMBER()` over `(partition, order)` then filter `=1` | A deterministic total order exists | Latest state, first touch, most recent price | Order key has unresolvable ties | V-03 |
| `DISTINCT ON` / `GROUP BY` max | Engine-specific single-pass dedup | Same as V-03 | Engine supports it and plan is acceptable | Portability required | V-04 |
| Running total | `SUM() OVER (ORDER BY k ROWS UNBOUNDED PRECEDING)` | Order is total and stable | Cumulative balance, cumulative spend | Timestamp has ties and no tie-breaker | V-05 |
| Moving average | `ROWS BETWEEN n-1 PRECEDING AND CURRENT ROW` | `n` consecutive observations exist | Smoothing a daily series | Gaps are meaningful and should break the window | V-06 |
| Previous/next row | `LAG`/`LEAD` with explicit offset and default | Same ordering guarantees as V-05 | Time between steps, period-over-period change | First row's `NULL` is treated as zero | V-07 |
| Rank vs dense rank vs row number | Tie handling differs | Tie policy declared | Top-N with or without tie retention | The three are treated as interchangeable | V-08 |
| Funnel by step | `MAX(CASE WHEN step=... )` per entity, then ordered comparison | Steps share entity key and time window | Ordered drop-off | Counting step events instead of entities | V-09 |
| Cohort retention | Denominator frozen at cohort entry | Cohort membership immutable after `t0` | Return behaviour over horizons | Denominator recomputed per period | V-10 |
| Semi-join | `EXISTS` / `IN` without duplicating left rows | Correlated key exists | Filter by existence only | Implemented as inner join + `DISTINCT` | V-11 |
| Anti-join | `NOT EXISTS` | Key semantics known | Churn, unshipped orders, gaps | `NOT IN` over a nullable column | V-12 |
| Pivot via `FILTER` | `COUNT(*) FILTER (WHERE ...)` | Dialect supports the clause | Wide report from one scan | Dialect lacks it; use `CASE` | V-13 |
| As-of join (SCD-2) | Range predicate on validity intervals | Intervals non-overlapping, half-open | Historical dimension attributes | Intervals overlap or have gaps | V-14 |
| Interval sessionisation | `LAG` + cumulative `SUM` of gap flags | Inactivity threshold declared | Sessions inferred from event streams | Events are unordered or duplicated | V-15 |

**V-01 conditional aggregation.** One scan produces several numerators over one denominator, which guarantees the shared-cohort condition by construction. The pattern fails when analysts add a new `SUM(CASE ...)` whose predicate describes a *different* population, silently widening the numerator beyond the denominator.

```text
FORMULA [SQL-06]: c_j = sum over e in E of 1[condition_j(e)]
WHERE: E the single eligible population; j = 1..J metric variants;
       c_j [units], c_j <= |E| for every j.
CONSTRAINT: all J predicates evaluated on the same rows of the same population;
            any predicate requiring rows absent from E needs its own denominator.
NUMERICAL EXAMPLE: |E| = 5 orders; conditions "completed" -> 2, "refunded" -> 1,
                   "cancelled" -> 1; the three numerators sum to 4 <= 5, so the
                   categories are disjoint and the remaining 1 order is "open".
INTERPRETATION: one denominator serves all three rates, so 40% / 20% / 20% / 20%
                is an exhaustive partition rather than four unrelated numbers.
```

**V-02 ratio of sums.** When each row carries a different exposure, the average of per-row ratios is not the population ratio. With rows `(c_i, e_i)`, `sum(c_i)/sum(e_i)` weights by `e_i`, whereas `avg(c_i/e_i)` weights every row equally. The identity is:

```text
[5] sum(c_i)/sum(e_i) = sum_i (e_i / sum_j e_j) * (c_i/e_i)
```

so the aggregate ratio is the `e`-weighted mean of the row ratios. This is why "average conversion rate across pages" is a different metric from "conversion rate across the site": the first treats a 2-visit page like a 2-million-visit page.

```text
FORMULA [SQL-07]: R = SUM(c) / SUM(e)
WHERE: c converted count per row [units]; e eligible count per row [units];
       R population-level ratio [dimensionless].
CONSTRAINT: e_i > 0 or the row is excluded with a stated rule; both sums over the
            same row set.
NUMERICAL EXAMPLE: rows (c,e) = (1,10), (9,100), (0,0 handled by exclusion);
                   R = 10/110 = 0.0909, whereas AVG(0.1, 0.09) = 0.095.
INTERPRETATION: the equal-weight average overstates the population rate because
                it gives the smallest denominator equal influence.
```

**V-03 dedup by ranking.** The latest-state pattern is a total-order requirement dressed as syntax. The formula is a rank, and correctness depends entirely on whether the ordering expression is injective on the partition.

```text
FORMULA [SQL-08]: rn = ROW_NUMBER() OVER (PARTITION BY k ORDER BY t DESC, s DESC);
                  keep rows with rn = 1
WHERE: k entity key; t event timestamp; s monotone tie-breaker (sequence number,
       ingestion id); rn ordinal [dimensionless, 1..n_k].
CONSTRAINT: (t, s) must be unique within k, otherwise the chosen row is
             arbitrary and may change between runs or between engine versions.
NUMERICAL EXAMPLE: k=O7 has three rows with (t, s) =
                   (10:00, 41), (11:00, 55), (11:00, 62);
                   ordering by t DESC, s DESC gives rn = 1 for (11:00, 62).
                   Ordering by t DESC alone leaves two rows tied at rn = 1.
INTERPRETATION: without the sequence tie-breaker the "latest" status is whichever
                row the plan happened to emit first, which is not a definition.
```

**V-05 and V-06 windows.** A cumulative aggregate is a recurrence, so its correctness is the correctness of the frame:

```text
FORMULA [SQL-09]: S_t = sum_{i=1..t} x_i with frame
                  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
WHERE: x_i additive quantity in partition order [units]; S_t same units;
       t position in the declared order, 1-indexed.
CONSTRAINT: additive measure only (never a rate or an id count); total order with a
            tie-breaker; explicit ROWS frame rather than the engine default.
NUMERICAL EXAMPLE: x = (5, 7, -2, 4) currency units -> S = (5, 12, 10, 14);
                   14 equals 5+7-2+4, the independent check.
INTERPRETATION: the final element is the partition total, so a wrong frame is
                visible by comparing the last value to a plain SUM.
```

```text
FORMULA [SQL-10]: MA_t = (1/n) * sum_{i=t-n+1..t} x_i
WHERE: n window length [observations]; x_i series value; MA_t moving mean,
       same units as x; defined for t >= n.
CONSTRAINT: exactly n observations in the frame; if the series has date gaps, ROWS
            semantics silently reaches across the gap, so join to a calendar first.
NUMERICAL EXAMPLE: x = (10, 20, 30, 40, 50), n = 3:
                   MA_3 = 20, MA_4 = 30, MA_5 = 40.
INTERPRETATION: each value is a local mean; the series has a constant +10 step,
                which the 3-point mean still shows as +10 per step after the warm-up.
```

The distinction between `ROWS` and `RANGE` is not cosmetic. `ROWS` counts physical rows; `RANGE` groups rows with equal ordering values (peers) and, in the common `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` form, includes all peers of the current row. With two sales on the same date, a daily running total under `ROWS` reports a partial figure mid-date, while `RANGE` reports the end-of-date figure for both rows. Both are correct answers to different questions, and neither is the default universally.

**V-07 lag/lead and the null boundary.** `LAG(x, 1)` returns `NULL` for the first row of a partition. Subtracting it produces `NULL`, and `NULL` propagates through further arithmetic, so a "days between steps" column becomes null for every first step, which is usually right — but if it is later `COALESCE`d to zero it silently claims zero elapsed time. Choose the default explicitly (`LAG(x, 1, 0)`) only when zero is semantically meaningful.

**V-08 the three ranks.** With values `(100, 100, 90)` ordered descending:

| Value | `ROW_NUMBER` | `RANK` | `DENSE_RANK` |
|---|---|---|---|
| 100 | 1 | 1 | 1 |
| 100 | 2 | 1 | 1 |
| 90 | 3 | 3 | 2 |

`ROW_NUMBER` invents an order among ties and is unsafe for "top N" when ties must be retained. `RANK` skips positions after a tie, so `RANK() <= 2` returns both tied leaders. `DENSE_RANK` does not skip, so it counts distinct values rather than rows. "Top 3 products" means three different things depending on this choice, and the report must state which.

**V-09 funnel by entity.** A funnel is a membership problem, not a counting problem:

```text
FORMULA [SQL-11]: step_j reached by e iff exists t with step(e)=j and
                  t_step(e,j-1) <= t (monotone path)
WHERE: step index j = 1..J; times in one zone; entity key stable across steps.
CONSTRAINT: step order enforced by time comparison; one row per entity per step
            before comparison; window bounds identical for all steps.
NUMERICAL EXAMPLE: J=3; step1 1000 entities, step2 420 with t2 >= t1, step3 310
                   with t3 >= t2; step conversions 42.0% and 73.81%;
                   end-to-end 310/1000 = 31.0%.
INTERPRETATION: 31% of entrants complete all three steps; the 73.81% second-step
                rate is computed on the 420 who reached it, not on all 1000.
```

`73.81% = 310/420 = 0.738095…`. Reporting `310/1000 = 31%` at every stage is a common slide error: it double-counts the first drop-off into every later stage.

**V-10 cohort retention.** Freeze the denominator at entry by construction date:

```text
FORMULA [SQL-12]: R(d) = |A(d) ∩ C| / |C|
WHERE: C cohort of entities whose first activity fell in [t0,t1); A(d) entities
       active at horizon d; R(d) dimensionless; d counted in whole periods.
CONSTRAINT: C immutable after t0; activity window and horizon aligned to the same
            period boundaries; entities leaving the product permanently still count
            in the denominator until the observation window ends.
NUMERICAL EXAMPLE: |C| = 1000; active at d=1: 420; d=2: 310; d=3: 250;
                   R = 42.0%, 31.0%, 25.0%.
INTERPRETATION: the curve decays and its slope, not its level, is the comparable
                quantity across cohorts of different size.
```

**V-11 and V-12 semi- and anti-joins.** These exist to *preserve* left-side cardinality. `EXISTS` is a predicate over a correlated subquery; it returns a boolean per left row, so the left row count is unchanged by construction. The anti-join is the same with `NOT EXISTS`.

```text
FORMULA [SQL-13]: keep e iff NOT EXISTS (SELECT 1 FROM R WHERE R.k = e.k)
WHERE: e left row; R right relation; k join key present on both sides;
       result is a subset of the left relation's rows.
CONSTRAINT: NULL keys never match in either direction, so an entity with k IS NULL
             is retained by the anti-join; decide explicitly whether that is right.
NUMERICAL EXAMPLE: 5 orders, 2 present in shipments table -> 3 orders retained as
                   "not yet shipped"; one of the 3 has a NULL shipment key reference
                   and is retained for the null reason, not the missing-row reason.
INTERPRETATION: the anti-join answers "no matching row exists", which includes
                "the match key is unknown" — a different claim from "shipped late".
```

**V-14 as-of joins.** Slowly changing dimensions require an interval predicate, not an equality:

```text
FORMULA [SQL-14]: join fact f to dim d on
                  f.entity = d.entity AND f.ts >= d.valid_from AND f.ts < d.valid_to
WHERE: valid_from, valid_to half-open interval bounds [timestamps];
       at most one d row should satisfy the predicate for any f.
CONSTRAINT: intervals per entity must be non-overlapping and gapless over the
            covered range; equal-to-boundary facts must be assigned to exactly one side.
NUMERICAL EXAMPLE: price v1 valid [01-01, 02-01), v2 valid [02-01, 03-01).
                   A sale at 02-01T00:00Z matches v2 only, because the upper bound
                   of v1 is exclusive and the lower bound of v2 is inclusive.
INTERPRETATION: each fact is priced at the version in force when it occurred,
                so historical revenue does not change when the price list does.
```

**V-15 interval sessionisation.** When sessions are not logged, infer them: order events, flag a new session when the gap to the previous event exceeds `θ`, and take a cumulative sum of the flags.

```text
FORMULA [SQL-15]: g_i = 1 if t_i - t_{i-1} > θ else 0 (g_1 = 1);
                  session_id_i = sum_{j<=i} g_j
WHERE: t_i ordered event timestamps within entity [timestamp]; θ inactivity
       threshold [duration]; session_id ordinal [integer].
CONSTRAINT: events deduplicated and totally ordered first; θ declared and held
            constant across the compared periods; gap computed on the same clock.
NUMERICAL EXAMPLE: θ = 30 min; event times 09:00, 09:10, 09:55, 10:05, 10:50;
                   gaps 10, 45, 10, 45 min -> flags 1,0,1,0,1 -> sessions
                   {09:00,09:10}, {09:55,10:05}, {10:50}: three sessions.
INTERPRETATION: session counts are a function of θ; changing θ from 30 to 60 minutes
                would merge sessions 1-2 and 3 into two, so θ belongs in the metric
                definition, not in a config file nobody reads.
```

### 4.4 Asymptotic / limiting behaviour and counterexamples

This subsection is the one to reread before shipping. Each row is a boundary where a formula from 4.1–4.3 stops being true, together with the smallest reproducible check.

| Result/invariant | Required assumption | Diagnostic | Violating example | Correct alternative |
|---|---|---|---|---|
| `r = c/e` in `[0,1]` | Num. subset of denom., same window | Print `c`, `e`, `c-e` | `c = 112, e = 100` from mixed grains | Recompute both at entity grain |
| `COUNT(DISTINCT x)` additivity | Never true across partitions | Compare sum of parts to whole | Days with repeat visitors | `COUNT(DISTINCT)` on unioned set, or HLL with stated error |
| `AVG` of per-group `AVG` equals overall `AVG` | Equal group sizes | Compare weighted vs unweighted | Group sizes 200 vs 100 | Weight by group count |
| `RANGE` cumulative equals `ROWS` cumulative | No peer ties | Count rows per order value | Two events at same timestamp | Add tie-breaker, or choose semantics deliberately |
| `NOT IN (subquery)` = anti-join | Subquery has no `NULL` | `SELECT COUNT(*) WHERE col IS NULL` | One null in the list | `NOT EXISTS` |
| `a/b` returns a fraction | Non-integer division | Check `2/3` result | `2/3 = 0` in integer arithmetic | Cast: `a::numeric / b` |
| Median via `AVG` | Symmetric distribution | Compare with `percentile_cont` | Values 1, 2, 100 | Use the percentile function |
| `LIMIT n` after `ORDER BY` is deterministic | Order is total | Run twice; compare | Ties at the boundary rank | Add unique tie-breaker |

**Counterexample 1: distinct counts do not add.** Visitors on Monday = 600, on Tuesday = 500. The two-day total is not 1100 if 100 people visited both days: it is 1000. This is inclusion–exclusion, `|A ∪ B| = |A| + |B| − |A ∩ B|`, and it generalises with more terms. Any report that sums daily uniques into a weekly unique is wrong by the overlap, and the overlap is usually largest exactly where the metric matters (loyal users). Approximate distinct sketches such as HyperLogLog restore additivity with a stated relative error, at the cost of not being able to list the members.

**Counterexample 2: the average of averages.** Two stores with 200 and 100 orders and conversion 0.90 and 0.20. The unweighted mean is 0.55; the true pooled rate is `(180+20)/300 = 0.6667`. The error is 11.7 points, in the direction of over-weighting the small store. The check is arithmetic, not statistical: compare `AVG(r_k)` with `SUM(c_k)/SUM(e_k)` and expect them to differ whenever `e_k` varies.

**Counterexample 3: integer division.** In engines where `/` between two integers is integer division, `2/3` is `0` and `7/2` is `3`. A conversion rate computed this way is either 0% or 100%, and it will look like a data problem rather than a type problem. Multiply by `1.0`, cast to a decimal type, or use the engine's `CAST` syntax explicitly. The check is to select the raw expression once and read the value.

**Counterexample 4: empty partitions and the missing zero.** A window function over a partition with no rows returns no rows; a `LEFT JOIN` to a calendar with no fact produces `NULL`, not `0`. Summing `NULL`s yields `NULL` (not zero), and a chart may render the gap as a straight line between distant points, implying activity that never happened. The fix is an explicit calendar spine plus `COALESCE(..., 0)`, applied *after* the outer join and only for genuinely additive measures — never `COALESCE` a rate's denominator to zero and then divide.

**Counterexample 5: ties change the answer, not just the order.** With three candidates at scores 90, 90, 80 and a top-1 cut, `ROW_NUMBER` returns one of the two leaders; the choice depends on the plan, not the data. `RANK() <= 1` returns both. If the downstream process has a fixed capacity of one, the tie must be broken by a declared business rule; if capacity is soft, retaining the tie is the honest answer. Either way, the tie policy is part of the metric.

**Counterexample 6: `COUNT(*)` after `LEFT JOIN` with an aggregate on the right.** Joining a fact to a pre-aggregated subquery is safe; joining to a detail table and aggregating at the outer level is not, unless the outer aggregate is over the correct grain. The failure is detectable by comparing the output row count to the fact table's distinct-key count, which is the same check as [CARD-1] applied one level up.

**Counterexample 7: division by a filtered denominator.** A rate whose denominator is `COUNT(*)` but whose numerator is `COUNT(*) FILTER (WHERE status='paid')` breaks the subset condition whenever the status filter's population is itself restricted, for example when `status` is null for imported orders. The arithmetic symptom is a rate below 100% that nonetheless fails to reconcile when you add up the categories; the fix is `COUNT(*) FILTER (WHERE status='paid')` over `COUNT(*) FILTER (WHERE status IS NOT NULL)` with the null category reported separately.

**Takeaway:** every formula here has a boundary where it returns a confident wrong answer, and each boundary has a one-line arithmetic check that catches it before publication.

## 5. Worked Numerical Example

### 5.1 Example record

```text
Example title:        Paid-order rate and revenue for the first week of February,
                      reconstructed from an event log and a line-item table.
Evidence label:       ILLUSTRATIVE data; DERIVED and hand-checked arithmetic.
Workflow and decision: decide whether the paid-order rate for the week clears a
                      60% operating threshold, and whether the revenue figure can
                      be published without a restatement caveat.
Input tables:         order_events (append-only status log), order_lines (one row
                      per SKU per order), calendar (date spine).
Valid assumptions:    event_ts is authoritative for period assignment; ingest_ts
                      is authoritative for availability; a status sequence number
                      resolves ties; the week is [2026-02-02T00:00Z, 2026-02-09T00:00Z).
Simplifications:      single currency, no tax, no refunds netting, no returns after
                      the cutoff, five orders only so the arithmetic is checkable.
Remaining uncertainty: the sample is far too small to estimate a rate precisely;
                      it demonstrates the mechanics, not a business conclusion.
```

**Why this example and not a clean one.** The interesting part of analytical SQL is what happens when the data is slightly hostile: a refund after shipment, a duplicated status event, an order with no line rows, and a customer with two orders. A tidy example would demonstrate syntax; this one demonstrates the reconciliation that decides whether the number can be published.

### 5.2 Input data

`order_events` — one row per status transition, with a monotone `seq` per order. Evidence: ILLUSTRATIVE.

| order_id | customer | region | event_ts (UTC) | seq | status |
|---|---|---|---|---|---|
| O1001 | C01 | north | 2026-02-02T09:00Z | 1 | placed |
| O1001 | C01 | north | 2026-02-02T09:04Z | 2 | paid |
| O1001 | C01 | north | 2026-02-03T11:00Z | 3 | shipped |
| O1002 | C02 | north | 2026-02-02T12:30Z | 1 | placed |
| O1002 | C02 | north | 2026-02-02T13:10Z | 2 | paid |
| O1003 | C01 | south | 2026-02-03T08:15Z | 1 | placed |
| O1003 | C01 | south | 2026-02-03T08:20Z | 2 | paid |
| O1003 | C01 | south | 2026-02-05T10:00Z | 3 | refunded |
| O1004 | C03 | north | 2026-02-03T19:40Z | 1 | placed |
| O1004 | C03 | north | 2026-02-04T07:05Z | 2 | cancelled |
| O1005 | C04 | south | 2026-02-04T06:00Z | 1 | placed |
| O1005 | C04 | south | 2026-02-04T06:02Z | 2 | paid |
| O1005 | C04 | south | 2026-02-04T06:02Z | 3 | paid |

`order_lines` — one row per SKU per order, with quantity and unit price. Evidence: ILLUSTRATIVE.

| order_id | sku | qty | unit_price |
|---|---|---|---|
| O1001 | sku-1 | 2 | 19.50 |
| O1001 | sku-2 | 1 | 45.00 |
| O1002 | sku-3 | 1 | 12.00 |
| O1003 | sku-1 | 1 | 19.50 |
| O1003 | sku-4 | 3 | 8.25 |
| O1004 | sku-5 | 1 | 99.00 |
| O1005 | sku-2 | 4 | 45.00 |
| O1005 | sku-6 | 2 | 7.75 |

Two facts about this input decide most of the analysis, and neither is visible without counting:

- `order_events` has **13 rows for 5 orders** (average 2.6 rows/order), so any `COUNT(*)` over that table reports events, not orders.
- `order_lines` has **8 rows for 5 orders**, so a direct join to the event log multiplies each order's events by its line count — the classic fan-out trap, here small enough to verify by hand (Section 5.3).

An order with no line rows would vanish from an inner join entirely. In this data every order has lines, which is precisely why a bug here would go unnoticed until it mattered.

### 5.3 Step 1 — declare the grain and predict cardinality

Grain sentence for the primary metric:

> One row per `order_id` whose `placed` event lies in `[2026-02-02T00:00Z, 2026-02-09T00:00Z)`, carrying the final status as of the cutoff, the customer, the region, and the order's net line revenue.

Predicted row count: 5. Predicted distinct `order_id`: 5. The test `[GRAIN-A]` must therefore return `5 = 5`.

Predicted intermediate cardinalities before running:

| Relation | Rows | Distinct `order_id` | Average fan-out | Max fan-out |
|---|---|---|---|---|
| `order_events` | 13 | 5 | 2.6 | 3 (O1001, O1005) |
| `order_lines` | 8 | 5 | 1.6 | 2 (O1001) |
| `order_events ⋈ order_lines` on `order_id` | 22 | 5 | 4.4 | 6 (O1001 and O1003) |

The per-key products are `O1001 3×2=6`, `O1002 2×1=2`, `O1003 3×2=6`, `O1004 2×1=2`, `O1005 3×2=6`, summing to **22**. The temptation is to multiply the two table totals, `13 × 8 = 104`, or to multiply row counts naively as `5 × 8 = 40`. Both are wrong. A join's size is the **sum of per-key products**, not the product of the table sizes, unless the join key is unique on at least one side or the tables are complete bipartite. Writing the per-key products out first is what makes the later reconciliation possible; an analyst who only knows the total cannot tell which order caused a discrepancy.

The complete prediction table is:

| Join | Predicted rows | Derivation |
|---|---|---|
| `orders ⋈ lines` | 22 | `sum_orders (events_o × lines_o)` |
| `orders ⋈ lines` with `status='paid'` pre-filter on events | 22 minus filtered rows | filter before or after the join changes the count |
| `orders ⋈ orders` self join for "previous order by same customer" | 1 | `sum_customers (orders_c − 1) = (2−1)+0+0+0 = 1` |

That last row illustrates another prediction: a self-join for "previous order by the same customer" produces at most `n − distinct_customers = 5 − 4 = 1` pair here (C01 has two orders), so the expected count is 1, not 5.

### 5.4 Step 2 — build the entity spine and dedup the status

The safe pattern is to reduce to entity grain first, then join outward. The reduction uses `[SQL-08]` with a two-key order.

```sql
WITH bounds AS (
  SELECT TIMESTAMPTZ '2026-02-02T00:00:00Z' AS t0,
         TIMESTAMPTZ '2026-02-09T00:00:00Z' AS t1,
         TIMESTAMPTZ '2026-02-09T06:00:00Z' AS cutoff_lateness
),
placed AS (
  SELECT e.order_id, e.customer, e.region, e.event_ts AS placed_ts
  FROM order_events e
  JOIN bounds b ON TRUE
  WHERE e.status = 'placed'
    AND e.event_ts >= b.t0 AND e.event_ts < b.t1
),
latest_status AS (
  SELECT order_id, status, event_ts, seq,
         ROW_NUMBER() OVER (
           PARTITION BY order_id
           ORDER BY event_ts DESC, seq DESC
         ) AS rn
  FROM order_events
)
SELECT p.order_id, p.customer, p.region, p.placed_ts,
       s.status AS final_status
FROM placed p
JOIN latest_status s
  ON s.order_id = p.order_id AND s.rn = 1;
```

| order_id | customer | region | placed_ts | final_status |
|---|---|---|---|---|
| O1001 | C01 | north | 2026-02-02T09:00Z | shipped |
| O1002 | C02 | north | 2026-02-02T12:30Z | paid |
| O1003 | C01 | south | 2026-02-03T08:15Z | refunded |
| O1004 | C03 | north | 2026-02-03T19:40Z | cancelled |
| O1005 | C04 | south | 2026-02-04T06:00Z | paid |

Five rows, five distinct orders. `[GRAIN-A]` passes.

Note what the `ORDER BY event_ts DESC, seq DESC` accomplishes on O1005, whose `paid` and `paid` rows share a timestamp: `seq` 3 wins deterministically. Without `seq`, either row could be chosen, and while the status happens to be identical here, a log where the second row were `refunded` would silently flip the reported state depending on plan.

### 5.5 Step 3 — the wrong path, computed honestly

Join the detail table to the spine and count rows. The naive rate:

```sql
-- WRONG: row-counted rate after fan-out
SELECT
  COUNT(*) FILTER (WHERE status IN ('paid','shipped')) * 1.0
    / COUNT(*) AS paid_rate
FROM order_events e
JOIN order_lines l ON l.order_id = e.order_id
WHERE e.status <> 'cancelled';
```

Evaluate it by hand on the 22-row join, keeping only event rows whose status is not `cancelled`:

- O1001: 3 event rows, all non-cancelled, × 2 lines = 6 rows; statuses `placed, paid, shipped` — 4 rows have `paid` or `shipped` (2 each).
- O1002: 2 × 1 = 2 rows; `placed, paid` — 1 row counts.
- O1003: 3 × 2 = 6 rows; `placed, paid, refunded` — 2 rows count (`paid`).
- O1004: 1 × 1 = 1 row (the `cancelled` row is filtered); `placed` — 0 rows count.
- O1005: 3 × 2 = 6 rows; `placed, paid, paid` — 4 rows count.

Rows: `6+2+6+1+6 = 21`. Qualifying: `4+1+2+0+4 = 11`. Naive rate `11/21 = 0.5238`.

The defensible entity-level rate, counting each order once by its final status:

| Definition of "paid" | Numerator | Denominator | Rate |
|---|---|---|---|
| Final status in `{paid, shipped}` | 3 (O1001, O1002, O1005) | 5 | 0.600 |
| Ever reached paid-or-later | 4 (add O1003) | 5 | 0.800 |
| Row-counted after fan-out, cancelled excluded | 11 | 21 | 0.5238 |

Three numbers, three different questions, and only the first two are coherent. The naive 52.4% is wrong twice over: it inflates orders with more line items (O1001 and O1005 contribute four rows each) and it drops cancelled orders from the denominator, which changes the population. Whether the threshold is met flips depending on which you pick: 60.0% meets a "60% or better" rule exactly, 52.4% fails it, 80.0% comfortably passes.

**Wrong path, stated as a lesson.** The plausible mistake is to reach for `COUNT(DISTINCT order_id)` as the repair. It changes the numerator to 3 and the denominator to 5 — the first row of the table — which happens to be right here, but only by coincidence: `DISTINCT` over the joined relation also deduplicates the *numerator's* population, and if the metric were "average line items per paid order" the same repair would silently drop legitimate rows. `DISTINCT` is a symptom treatment; the correction is to reduce to entity grain *before* the metric.

### 5.6 Step 4 — attach measures without re-fanning

Revenue must be computed at the line grain and then aggregated up, never the reverse:

```sql
WITH order_revenue AS (
  SELECT order_id, SUM(qty * unit_price) AS revenue
  FROM order_lines
  GROUP BY order_id
)
SELECT s.order_id, s.region, s.final_status,
       COALESCE(r.revenue, 0) AS revenue
FROM spine s
LEFT JOIN order_revenue r ON r.order_id = s.order_id;
```

Arithmetic per order: `O1001 = 2×19.50 + 1×45.00 = 39.00 + 45.00 = 84.00`; `O1002 = 12.00`; `O1003 = 19.50 + 3×8.25 = 19.50 + 24.75 = 44.25`; `O1004 = 99.00`; `O1005 = 4×45.00 + 2×7.75 = 180.00 + 15.50 = 195.50`. Total `84.00 + 12.00 + 44.25 + 99.00 + 195.50 = 434.75`.

The `LEFT JOIN` is deliberate: an order with no lines must still appear with zero revenue rather than disappearing. `COALESCE(..., 0)` is safe here because revenue is additive; applying the same reflex to a rate's denominator would create a division by zero.

Revenue by final status, as a reconciliation table:

| Final status | Orders | Revenue | Share of revenue |
|---|---|---|---|
| paid | 2 | 12.00 + 195.50 = 207.50 | 47.73% |
| shipped | 1 | 84.00 | 19.32% |
| refunded | 1 | 44.25 | 10.18% |
| cancelled | 1 | 99.00 | 22.77% |
| **total** | **5** | **434.75** | **100.00%** |

Check: `207.50 + 84.00 + 44.25 + 99.00 = 434.75`, and the shares `47.73 + 19.32 + 10.18 + 22.77 = 100.00` after rounding. The published GMV figure, if defined as orders that were ever paid, is `84.00 + 12.00 + 44.25 + 195.50 = 335.75`, which is `434.75 − 99.00`. Both numbers are defensible; the point is that they differ by a cancelled order worth 22.8% of the total, so "revenue" without a status rule is ambiguous by nearly a quarter.

### 5.7 Step 5 — windowed trend and the frame check

Daily paid-order rate, cumulative, with an explicit frame:

```sql
WITH daily AS (
  SELECT date_trunc('day', placed_ts) AS d,
         COUNT(*) AS eligible,
         COUNT(*) FILTER (WHERE final_status IN ('paid','shipped')) AS paid
  FROM spine
  GROUP BY 1
)
SELECT d, eligible, paid,
       SUM(paid) OVER (ORDER BY d ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
         AS cum_paid,
       SUM(eligible) OVER (ORDER BY d ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
         AS cum_eligible
FROM daily;
```

| Day | Eligible | Paid | Cumulative paid | Cumulative eligible | Cumulative rate |
|---|---|---|---|---|---|
| 2026-02-02 | 2 | 2 | 2 | 2 | 1.000 |
| 2026-02-03 | 2 | 0 | 2 | 4 | 0.500 |
| 2026-02-04 | 1 | 1 | 3 | 5 | 0.600 |

Independent check: the last cumulative rate `3/5 = 0.600` equals the entity rate from 5.5 row 1, computed by a completely different route (per-day grouping and running sum versus one-shot status filter). That agreement is the reconciliation this whole sheet is about. A second check: cumulative eligible must be non-decreasing and end at 5; it does.

Note the day-2 figure of zero. O1003 was placed on 2 February and refunded on 5 February, so its final status is `refunded` even though it was, briefly, paid. A "paid by day of payment" metric would put it in the 2 February numerator; a "paid by day of placement, as of cutoff" metric — the one declared here — does not. Both are legitimate; only one matches the declared grain sentence, and that is the one to publish.

### 5.8 Step 6 — cohort view

The same five orders, viewed as a cohort retention table over the customer key, to show the mechanical difference between an order metric and a customer metric.

| Customer | Orders | First order | Active in week 2? | Notes |
|---|---|---|---|---|
| C01 | 2 (O1001, O1003) | 2026-02-02 | yes | one order later refunded |
| C02 | 1 (O1002) | 2026-02-02 | no | |
| C03 | 1 (O1004) | 2026-02-03 | no | cancelled |
| C04 | 1 (O1005) | 2026-02-04 | no | |

Cohort size 4 customers; retention at the next week is `1/4 = 25%`. The order-level rate was 60%, the customer-level repeat rate is 25%. They measure different things and neither is "the" conversion. Presenting the 60% figure to a stakeholder who asked "do customers come back?" would be a category error, which is why Section 2.3 maps question types to pattern families explicitly.

### 5.9 Step 7 — late data and the availability caveat

Suppose the week's event volume was 1200 rows and 37 arrived more than six hours after their event time (`37/1200 = 3.083%` late). If the report cutoff is `t1 + 6h`, then 3.08% of events — not necessarily 3.08% of *orders*, and not necessarily a uniform 3.08% across regions — were excluded. The honest publication is: "paid-order rate 60.0% (3 orders of 5) as of 2026-02-09T06:00Z, with 3.08% of events still in flight at cutoff; will restate if the final load changes the numerator or denominator." The dishonest version prints 60.0% with no timestamp and then silently changes.

### 5.10 Sensitivity of the decision

The threshold is "at least 60%". The measured rate is exactly 60.0%, on five orders. Three perturbations, each trivial in size but decisive for the decision:

| Perturbation | New rate | Meets 60%? | Why |
|---|---|---|---|
| O1005's duplicate `paid` row is a distinct refund (final status `refunded`) | 2/5 = 40.0% | No | tie-break correctness decides the answer |
| O1003's refund is not counted against it (ever-paid definition) | 4/5 = 80.0% | Yes | metric definition decides the answer |
| One more eligible order that fails to pay | 3/6 = 50.0% | No | denominator growth dilutes |
| Naive row-counted with cancelled excluded | 11/21 = 52.4% | No | grain defect decides the answer |

Four different decisions from one plausible-looking dataset. The lesson is not "the threshold is fragile" — with `n = 5` everything is fragile — but that **the decision was never determined by the data alone**; it was determined by definitions chosen before the query ran. That is why the grain sentence, the status rule, and the tie-breaker are the deliverable.

### 5.11 Production-scale interpretation

At 1.2 million orders per week with 3 line rows each on average, the same query shapes scale as follows. All figures are ILLUSTRATIVE arithmetic, not measurements.

| Quantity | Small example | Production-scale analog | Scaling note |
|---|---|---|---|
| Orders per period | 5 | 1,200,000 | linear in orders |
| Event rows | 13 | 3,600,000 (3 events/order) | events per order is the multiplier |
| Line rows | 8 | 3,600,000 | same order of magnitude |
| Naive join rows | 22 | 10.8 million (3×3 per order) | **square** of the per-order multiplicity |
| Entity-grain output | 5 | 1,200,000 | linear, and 9× smaller than the naive join |
| Hand-checkable sample | all | 200 rows | sample must be drawn at entity grain |

The naive join grows with the product of the two per-order multiplicities, so a table that got three times wider in line items and three times noisier in events produces nine times the intermediate rows for the same 1.2 million orders. That is the mechanical reason the grain-first pattern is also the cheaper one: it is not merely tidier, it does an order of magnitude less work.

### 5.12 What this example does not establish

It does not establish that 60% is a good threshold, that five orders are representative, that the refund policy is correct, or that any engine will choose the plan implied above. It establishes the mechanics: grain declaration, cardinality prediction including the correction of a wrong prediction, entity-level dedup with a deterministic tie-break, a computed wrong path, an independent reconciliation between two derivations, and the sensitivity of a decision to definitions rather than to data.

**Takeaway:** when two derivations of the same metric agree to the last decimal, you have earned the right to publish it; when they disagree, the disagreement localises the defect faster than any amount of reading the SQL.

## 6. Visual / Geometric Intuition

### 6.1 Join as a set map, not a table concatenation

The useful picture of a join is a function between sets of rows with a multiplicity attached. Draw the left relation as a row of boxes and the right as a column; the join is the set of pairs `(l, r)` satisfying the predicate.

```text
LEFT (events per order)        RIGHT (lines per order)      OUTPUT pairs
  [O1001] 3 events  ─────> [2 line rows] ──> 3 x 2 = 6
  [O1002] 2 events  ─────> [1 line row ] ──> 2 x 1 = 2
  [O1003] 3 events  ─────> [2 line rows] ──> 3 x 2 = 6
  [O1004] 2 events  ─────> [1 line row ] ──> 2 x 1 = 2
  [O1005] 3 events  ─────> [2 line rows] ──> 3 x 2 = 6
                                            -------
  left total = 13          right total = 8  sum  = 22
```

**Caption.** Schematic, not to scale. Left column: per-entity event counts from the Section 5 input. Right column: per-entity line-row counts. The product per entity is the number of joined rows that entity contributes; the join total is the sum of those products. **Interpretation:** the join is one multiplication per key followed by a sum, so a single entity with a large product dominates — an order with 40 line items and 5 events contributes 200 rows, which is more than all five orders here combined. When a joined count looks inflated, the productive question is "which key contributed most rows", and the diagram says to answer it by grouping the join by key before doing anything else.

Note the two misleading summaries a reader might compute from the same picture. Multiplying the totals, `13 × 8 = 104`, overcounts badly because no event of O1002 can pair with a line of O1005. Multiplying the entity count by the line total, `5 × 8 = 40`, is also wrong. Only the per-key products sum correctly. The general rule is: a join's cardinality is `sum over keys of (left multiplicity × right multiplicity)`, which collapses to `left row count` when the right side is unique per key — the condition every fact-to-dimension join relies on.

A diagram is still only a hypothesis about the data. Reconciling the picture against a query that groups the join by key is what turns it into evidence.

### 6.2 The cardinality-preserving band

The distinction between "filter by existence" and "join for detail" is a cardinality question, and it looks like this:

```text
row count
   ^
 n |  ooooooooooooooooooooooo   LEFT relation (n rows)
   |  o  o  o  o  o  o  o  o
   |  o  o  o  o  o
   |  o     o     o
   |
 0 +------------------------------------------> join type
        INNER      LEFT      SEMI     ANTI
         |          |          |        |
         v          v          v        v
      <= n       >= n        <= n     <= n
      (drops     (keeps      (exactly (exactly
      unmatched)  all, adds   n minus  n minus
                  null cols)  matches) matches)
```

**Caption.** Schematic comparison of row-count behaviour. The `INNER` case is bounded above by `n` only when the right side is unique per key; with a one-to-many right side it can exceed `n` without bound. **Interpretation:** if your purpose is to *filter* (does a matching row exist?), use a semi- or anti-join, whose output row count is a subset of the left relation by construction. If your purpose is to *enrich* (attach attributes), use a left join and verify the right side is unique on the join key first.

The `INNER` annotation is the one people get wrong. An inner join between two many-sided tables is not "fewer rows than the left table"; it is "one row per matching pair", which is bounded by the product of the two multiplicity distributions.

### 6.3 Window frames drawn as intervals

A frame is a sub-interval of the partition, positioned relative to the current row. Drawing it clarifies why `ROWS` and `RANGE` diverge on ties.

```text
ROWS BETWEEN 1 PRECEDING AND CURRENT ROW, partition ordered by date

 date:    02-01   02-02   02-03   02-04   02-05
 value:     10     20      30      40      50
                          
 frame at 02-03:   [20  30]                     -> mean = 25
 frame at 02-04:        [30  40]                -> mean = 35
 frame at 02-05:             [40  50]           -> mean = 45

RANGE BETWEEN 1 PRECEDING AND CURRENT ROW, with a tie at 02-03

 date:    02-02   02-03   02-03   02-04
 value:     20      30      35      40
                           
 frame at the FIRST 02-03 row:  [20, 30, 35]   -> mean = 28.33
 frame at the SECOND 02-03 row: [20, 30, 35]   -> mean = 28.33
 (both peers see the same frame, because RANGE works on order *values*)
```

**Caption.** Top: physical-row semantics on a unique date key, both rows in the frame, `n = 2`. Bottom: value-based semantics with a duplicated date; the frame at a peer includes all rows whose order value falls in the value range, so the two 02-03 rows share an identical frame. **Interpretation:** with ties, `ROWS` returns different values for rows that carry the same timestamp, which is usually indefensible for a published time series; `RANGE` returns the same value, which is usually what you want for a "by date" metric. Neither is universally correct, but silently inheriting the default is always wrong.

### 6.4 The funnel as nested sets

Funnels are easiest to reason about as downward-closed membership sets, not as bar widths.

```text
   |S1| = 1000   ###############################################
   |S2| =  420   ###################                            
   |S3| =  310   ##############                                 
                                                            
   step rate 1->2 = 420/1000 = 42.00%
   step rate 2->3 = 310/420  = 73.81%
   end-to-end     = 310/1000 = 31.00%

   SET VIEW
   S3 subset of S2 subset of S1
   |S1 \ S2| = 580   lost before step 2
   |S2 \ S3| = 110   lost between 2 and 3
   |S3|      = 310   completed
   check: 580 + 110 + 310 = 1000
```

**Caption.** Schematic funnel with absolute counts. The bar widths are proportional to counts; the percentages are conditional on the previous step except the end-to-end figure. **Interpretation:** the set view makes the arithmetic check obvious — the disjoint differences plus the final set must reconstruct the entry set. The bar view invites the error of reporting 31% at every stage, which double-counts each earlier drop-off.

The check `580 + 110 + 310 = 1000` is the funnel's version of `[GRAIN-A]`: if it fails, either a step is not a subset of its predecessor (an entity can reach step 3 without step 2), or the population changed between steps (different filters, different windows).

### 6.5 Cohort decay and the denominator that must not move

```text
retention
 1.0 |*
     | *
 0.5 |  *  *
     |      *
 0.25|        *
     |
 0.0 +--+--+--+--+--> horizon (periods since entry)
       0  1  2  3
       
  R(0)=1.000 (by definition, the cohort is active at entry)
  R(1)=0.420
  R(2)=0.310
  R(3)=0.250
```

**Caption.** Schematic exponential-ish decay of a single fixed cohort. The horizontal axis is periods since entry, not calendar time; the vertical axis is the fraction of the *original* cohort active at that horizon. **Interpretation:** the curve is read horizontally — a cohort's month-3 value is comparable to another cohort's month-3 value even if the cohorts launched a year apart. The denominator is `|C|` at every point; if it changes, the curve is measuring something else (for example, active-of-active rather than active-of-cohort, which is always 100% by construction and therefore useless).

The most common mis-reading is vertical: "retention dropped from 42% to 25%" describes the decay of one cohort across horizons, not a decline over time. A genuine decline over time appears as vertically stacked curves where a newer cohort sits below an older one at the same horizon.

### 6.6 Event time versus processing time as two orderings

```text
event time axis   ---o------o---------o-------o----> t
                     e1     e2        e3      e4

processing/ingest  -------o------o-------o------o--> t
                          e1     e4      e2     e3
                                 (out of order)

window [t0, t1) on event time:  {e1, e2, e3}
window [t0, t1) on ingest time: {e1, e4}
difference:                     {e2, e3} vs {e4}
```

**Caption.** Schematic of the same four events ordered on two different clocks. `e4` occurred after `t1` but was ingested before some events that occurred inside the window. **Interpretation:** the two windows are not noisy versions of each other — they partition the events differently. Choosing event time gives you a stable answer that may change when late data arrives; choosing processing time gives you a stable answer that is permanently wrong about when things happened. Publish the choice and the cutoff.

### 6.7 The shape of a fan-out error on a trend line

```text
reported rate
 1.0 |            *
     |          *   *
 0.5 |    *   *
     |  *
 0.0 +--+---+---+---+--> period
       P1  P2  P3  P4

 true (entity) rate:   0.40  0.42  0.45  0.46
 row-counted rate:     0.71  0.55  0.52  0.50
 divergence:          +0.31 +0.13 +0.07 +0.04
```

**Caption.** Schematic. The row-counted series converges toward the entity series as line items per order fall across periods, because the multiplier `m_e` becomes less variable. **Interpretation:** a fan-out error is not a constant offset that a backfill can rescale. Its magnitude depends on the correlation between multiplicity and outcome, so it distorts the *shape* of the trend. A series that appears to be improving might be a stable metric measured with a decaying bias. This is why the fix must be a re-derivation at the correct grain, not a calibration constant.

**Takeaway:** every diagram here is a counting argument in disguise; if the picture and the arithmetic disagree, the arithmetic is the authority.

## 7. Comparison / Selection Table

### 7.1 Method comparison

The baseline in every row is the simplest thing that could work — a single-table `GROUP BY` — because most metric questions genuinely do not need a join or a window, and reaching for a window function first is how grain bugs get introduced.

| Method/variant | Preconditions | Quality/guarantee | Time/memory/cost | Best for | Avoid when | Evidence |
|---|---|---|---|---|---|---|
| Single-table `GROUP BY` with `FILTER` (baseline) | All needed attributes on one relation at the target grain | Exact; cardinality trivially preserved | One scan, hash aggregate; `O(n)` time, `O(g)` memory for `g` groups | Rate metrics where the fact table already carries the entity key and status | Attributes live on another table, or history matters | DERIVED |
| Join then aggregate (`JOIN` → `GROUP BY`) | Right side unique on the join key | Exact only if right side is unique; otherwise `[SQL-02]` bias | One scan of both, hash join, then aggregate; memory `O(build side)` | Attaching small lookup dimensions (region, category) | One-to-many child tables (line items, events, tags) | DERIVED |
| Pre-aggregate child to parent grain, then join | Child has a parent key | Exact; the child's multiplicity is collapsed before it can multiply | Two scans plus a small join; memory `O(parent)` | Any metric on parent entities with child detail present | Parent grain must be derivable from the child key | DERIVED |
| Correlation via `EXISTS`/`NOT EXISTS` (semi/anti-join) | Key present, comparison cheap | Exact; output cardinality ≤ left input | One scan of left; right side may be indexed or hashed | "Has any", "has none" filters | You also need attributes from the right side | DERIVED |
| `COUNT(DISTINCT key)` aggregate | Key comparison supported | Exact per group; not additive across groups | One scan, hash set or sort-based distinct | Per-group unique counts | Summing results across partitions | DERIVED |
| Approximate distinct sketch (e.g. HLL family) | Engine support; tunable precision | Relative error approximately `1.04/sqrt(m)` for `m` registers [SOURCED, standard HLL result] | One scan, fixed `O(m)` state per group | Large-cardinality uniques that must be merged across partitions | Exact counts needed for billing or regulatory output | SOURCED |
| `ROW_NUMBER()` dedup | Total order exists on the partition | Exact if order is injective; arbitrary otherwise | One sort or top-N per partition | Latest state, first touch, most recent price | Order key has ties and no tie-breaker available | DERIVED |
| `DISTINCT ON` / aggregate-then-join dedup | Engine support | Exact; equivalent result to `ROW_NUMBER` under the same order | Depends on plan; sometimes cheaper | Portability not required, plans verified | Cross-engine codebase | HEURISTIC |
| Window aggregate with explicit frame | Additive measure, total order | Exact per the frame definition | One sort plus a streaming pass; `O(n log n)` typical | Running totals, moving averages, deltas | Measure is non-additive (rates, ids) | DERIVED |
| Self-join for previous row | Self-join key unique per period | Exact but redundant with windows | `O(n log n)` join; quadratic without an index | Engines without window functions | Any modern engine | DERIVED |
| Materialise an entity-grain intermediate table | Storage budget, refresh schedule | Exact and cheap to re-query | Extra write `O(n)`, storage `O(n)`, faster reads | Repeated dashboards over the same metric | Metric definition still in flux | HEURISTIC |

### 7.2 Selection rules in priority order

```text
IF the metric's population cannot be defined without a join
   -> first reduce the many side to the metric's grain in a CTE, then join.
ELSE IF a validity constraint fails (right side not unique on the key)
   -> disqualify plain join-then-aggregate; use pre-aggregation or EXISTS.
ELSE IF the answer requires "which row is latest/first/highest"
   -> ROW_NUMBER with an explicit total order and a declared tie policy.
ELSE IF the answer requires a value that depends on neighbouring rows
   -> window function with an explicit ROWS/RANGE frame and a calendar spine.
ELSE IF the answer requires only existence
   -> EXISTS / NOT EXISTS; never inner join plus DISTINCT.
ELSE IF distinct counts will be combined across periods or partitions
   -> count distinct on the unioned set, or a sketch with a stated error bound.
ELSE
   -> single-table GROUP BY with FILTER; simplest, cheapest, hardest to get wrong.
```

The ordering encodes a priority: *validity first, then semantics, then cost*. When two rules conflict — say a sketch would be cheapest but the number is a billing figure — validity wins and the sketch is disqualified. Cost is the last consideration precisely because a cheap wrong number is worse than an expensive right one.

### 7.3 Worked selection decisions

Three short cases, each resolved by one row of 7.1.

**Case A: "average order value by region".** Order value is `revenue / orders`, and revenue lives at line grain while orders live at order grain. Rule 1 applies: pre-aggregate lines to order, then join to the region lookup (unique per order), then average or divide sums. Selecting `AVG(line_value)` instead would compute average *line* value, which is smaller and unrelated to the question.

**Case B: "customers who bought SKU-1 but never SKU-2".** Two existence questions over the same purchase table. Semi-join for SKU-1, anti-join for SKU-2, both on `customer_id`, and the result's row count cannot exceed the customer count. Writing it as two joins plus `DISTINCT` risks a fan-out for multi-purchase customers and produces a different population if a customer bought SKU-1 twice.

**Case C: "weekly unique visitors and their week-over-week change".** Rule 6 applies: weekly uniques cannot be summed into monthly, but they *can* be computed independently per week and differenced. If the dashboard needs a rolling 28-day unique, compute it directly over a 28-day window rather than combining four weekly values. Approximating with a sketch is acceptable for a trend line and not acceptable for a contractual figure.

### 7.4 When the window function is the wrong tool

A window function is grain-preserving, which is exactly why it cannot compute a metric that needs to change the grain. Three situations where analysts reach for it and should not:

- **Funnel stages.** A window `COUNT` over step values returns per-row values; the funnel needs one row per entity with monotone stage membership, which is a `GROUP BY` with conditional `MAX` (V-09), not a window.
- **Deduplication followed by aggregation.** `ROW_NUMBER() = 1` then `GROUP BY` is two operations; using a window aggregate *after* dedup without filtering the duplicates first double-counts them, because the window sees every row including the ones marked `rn > 1`.
- **Percentile of a rate.** `PERCENTILE_CONT` over per-row rates gives the median row rate, which is not the median entity rate unless each row is one entity. On a joined relation it is the median of a fan-out-weighted mixture.

**Takeaway:** select the pattern from the grain and the validity condition, and let cost break ties only among patterns that are already correct.

## 8. Failure Modes & Edge Cases

### 8.1 The failure index

Every entry has a stable ID, a symptom you can observe, at least two competing causes (because a symptom with one obvious cause is a bug report, not a diagnostic), the check that discriminates between them, a bounded fix, and a verification that the fix worked.

| ID | Symptom | Competing causes | Discriminating check | Correct action | Verification | Provenance |
|---|---|---|---|---|---|---|
| SQL-F01 | KPI rose after a join was added | (a) genuine volume change; (b) one-to-many fan-out; (c) join duplicated dimension rows | Compare `COUNT(*)` with `COUNT(DISTINCT entity_key)` before and after; group the join by key and inspect the top multiplicities | Pre-aggregate the many side to the metric grain | Output row count equals distinct entity count from the source | DERIVED |
| SQL-F02 | `LEFT JOIN` returns fewer rows than the left table | (a) right-side predicate in `WHERE`; (b) right side has no matches at all; (c) key types differ so nothing matches | Move the predicate into `ON` and re-count; if the count returns to the left-row count, cause (a); if it stays low, cause (b) or (c) | Move predicates to `ON`, or restructure as `WHERE (right.col IS NULL OR right.col = x)` | Left row count preserved and null-matches present | DERIVED |
| SQL-F03 | "Latest" record differs between runs or between engines | (a) timestamp ties; (b) clock skew between sources; (c) plan-dependent row order | Count rows sharing the max timestamp per key; if ties exist, cause (a) | Add a monotone tie-breaker (`seq`, ingestion id) to `ORDER BY` | Repeated runs return an identical key set | DERIVED |
| SQL-F04 | Running total differs between two engines | (a) default `RANGE` vs `ROWS`; (b) peer timestamps present; (c) different null ordering | Emit the cumulative value next to a plain `SUM` at the last row; if it differs, the frame is wrong | Write the frame explicitly and add a unique order key | Last cumulative value equals the unconditional total | DERIVED |
| SQL-F05 | Conversion rate exceeds 100% | (a) numerator counted at finer grain than denominator; (b) denominator filtered harder; (c) different windows | Print numerator, denominator, and `numerator − denominator`; positive difference means grain mismatch | Recompute both at one declared grain and one window | Rate bounded in `[0,1]`; difference non-positive for subset metrics | DERIVED |
| SQL-F06 | Rate is 0% or 100% for every row, in whole numbers | Integer division in the engine's `/` operator | `SELECT 2/3` and read the value | Cast numerator or denominator to a decimal type | Rate takes fractional values on a known 1/3 case | DERIVED |
| SQL-F07 | Weekly unique users far below the sum of daily uniques | (a) genuine overlap in visitors; (b) different time-zone boundaries between day and week aggregations | Compute the unioned distinct count in one query and compare with inclusion–exclusion estimate | Count distinct over the unioned set, or use a mergeable sketch | Estimate within the sketch's stated error of the exact value | DERIVED |
| SQL-F08 | Cohort retention exceeds 100%, or is exactly 100% at every horizon | (a) denominator recomputed per period as "active last period"; (b) cohort membership not frozen; (c) entity key reused across accounts | List cohort membership at `t0` and at the horizon; compare set sizes | Freeze `|C|` at entry, store the cohort snapshot | `R(0) = 1` exactly and `R(d) <= R(d−1)` for a single cohort | DERIVED |
| SQL-F09 | Anti-join returns zero rows although unmatched entities certainly exist | `NOT IN` over a subquery containing a `NULL` | `SELECT COUNT(*) FROM right WHERE key IS NULL`; if > 0, the predicate is unknowable | Rewrite `NOT EXISTS`, or filter nulls from the subquery | Unmatched count matches a left-join-plus-is-null count | DERIVED |
| SQL-F10 | Revenue differs from the finance system by a small percentage | (a) currency rounding at different grains; (b) refunds netted or not; (c) tax/discount convention; (d) excluded cancelled orders | Reconcile order by order for a 200-order sample and classify every difference | Adopt one netting and rounding convention, applied at line grain then summed | Sample reconciles to zero difference under the stated convention | ILLUSTRATIVE |
| SQL-F11 | Report changes by itself overnight | (a) late-arriving events restated a closed period; (b) dimension attributes mutated; (c) query has a non-deterministic `LIMIT` or tie | Diff the entity ID sets between two runs and inspect the added/removed rows | Add a cutoff timestamp to the metric, pin dimension versions, remove unorderable `LIMIT` | Two runs at the same cutoff produce identical output | DERIVED |
| SQL-F12 | Funnel shows more conversions at step 3 than step 2 | (a) steps not ordered in time; (b) counting events rather than entities; (c) different windows per step | Count entities per step with and without the time-ordering predicate | Enforce `t_j >= t_{j-1}` and one row per entity per step | `|S_j| <= |S_{j-1}|` for every `j` | DERIVED |
| SQL-F13 | Median or percentile disagrees with a spreadsheet | (a) interpolation method differs; (b) `AVG` used instead of a percentile function; (c) nulls included or excluded differently | Compute all three variants on the same 4 values and compare | Declare the percentile method; use `PERCENTILE_CONT` for continuous, `PERCENTILE_DISC` for nearest-rank | Reproduces the hand calculation for the declared method | DERIVED |
| SQL-F14 | Join takes hours after adding one filter | (a) predicate blocks pushdown; (b) implicit cast defeats index use; (c) join order changed by statistics staleness | Compare plans with and without the predicate; check for cast warnings | Rewrite to preserve sargability; align types; refresh statistics | Plan shows the same access path as the fast version | HEURISTIC |
| SQL-F15 | Two teams' identical-looking queries return different numbers | (a) different time-zone conversion; (b) different status sets; (c) different dedup rule; (d) different cutoff | Diff the queries line by line and, failing that, diff the intermediate entity counts at each CTE | Publish a metric definition with the four choices named; share one view | Both teams running the shared view agree | DERIVED |

### 8.2 Detailed failures

**SQL-F01 — the join that inflated the KPI.** This is the most consequential defect in analytical SQL, and it is usually introduced by a well-meaning improvement. Somebody needs "revenue by order status" and writes the metric against the line table so revenue is available; the numerator now counts line rows and the denominator still counts orders.

The discriminating check is not statistical. Group the joined relation by entity and look at the maximum multiplicity: if it is 1 for every entity, the join is safe for entity metrics; if it is 3 for the top vendor and 1 for the median vendor, the metric is a vendor-size-weighted average and will drift whenever the product mix changes. Verification is a row-count equality against the source entity table, which is cheap and should live in the pipeline as a test rather than in an analyst's memory.

**SQL-F02 — the left join that is not one.** `LEFT JOIN dim ON ... WHERE dim.flag = 'active'` returns the intersection, because the `WHERE` runs after the null-extension and rejects `NULL`. Three fixes, in order of preference: move the predicate into the `ON` clause (which makes non-matching rows keep their nulls), use `WHERE dim.flag = 'active' OR dim.id IS NULL` if inactive rows should still be represented but marked, or accept the inner-join semantics and rename the query so nobody believes it preserves unmatched rows. The verification for the first fix is a row count equal to the left relation's count; for the third, a count equal to the inner join's.

The competing cause that wastes the most time is (c), type mismatch. If `dim.id` is a `VARCHAR` and `fact.dim_id` is an `INTEGER`, some engines raise an error while others coerce silently, and the coercion can fail to match on values with leading zeros. A join returning exactly zero matches on a key that "obviously" exists is a type problem until proven otherwise.

**SQL-F03 — nondeterministic "latest".** The check is a single grouped count of rows per `(key, max_timestamp)`. Ties of size two are normal in event logs where two status writes share a second-resolution timestamp. The fix must be a column that is monotone within the key: an ingestion sequence, a log offset, or the tuple `(timestamp, id)` where `id` is a UUID with an ordering implementation. Timestamps alone are almost never a total order.

There is a subtler version: a *future*-dated event, caused by clock skew, wins the `ORDER BY ... DESC` permanently and suppresses every subsequent real update. The diagnostic is to look at the distribution of `event_ts - ingest_ts`; a right tail beyond a few minutes indicates skew, and the repair is to cap or quarantine future-dated rows rather than to change the ranking.

**SQL-F04 — frames and the illusion of portability.** Two engines can both implement `SUM(x) OVER (ORDER BY d)` and disagree, because the SQL standard's default frame is `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` but some engines historically treated the default differently, and `RANGE` on a date with peers produces end-of-date values for both peers. The check is to compare the last cumulative value against the unconditional `SUM`; if they differ, the frame excluded rows. The fix is to write `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` and add a tie-breaker to the order. Verification is a hand trace on three rows.

**SQL-F05 — the rate above 100%.** Print the three numbers. If `numerator − denominator > 0`, the numerator is at a finer grain; if the denominator shrank between two versions of the query, a filter moved; if the windows differ, one may span a boundary. The bounded fix is to define both from one CTE at one grain. The verification is the invariant `[GRAIN-1]`, plus a category reconciliation: the sum of exhaustive status numerators must equal the denominator exactly.

**SQL-F06 — integer division.** This one is worth a permanent habit: multiply by `1.0` or cast, and never trust a rate that is suspiciously quantised. A secondary symptom is percentages that are always multiples of a large step, which reveals a denominator as small as 1, 2, or 3 — often because the join failed and every group has one row.

**SQL-F07 — non-additive distincts.** Inclusion–exclusion for two sets gives the exact rule; for more sets the number of terms grows as `2^k − 1`, so direct unioned-set counting is preferable. When the source is too large for an exact distinct, a sketch with stated relative error is honest; summing daily uniques is not, because it is wrong by an unknown overlap and its error has no bound.

**SQL-F08 — cohorts that leak.** The giveaway is `R(d) = 100%` at every horizon, which means the denominator was redefined as "active users" each period. Freeze the cohort by writing membership to a snapshot table at `t0` and never recomputing it. The verification is monotonicity of the curve and `R(0) = 1` by construction.

**SQL-F09 — `NOT IN` and the null trap.** Three-valued logic makes `x NOT IN (1, 2, NULL)` equivalent to `x <> 1 AND x <> 2 AND x <> NULL`, and the last conjunct is `UNKNOWN` for every `x`, so nothing survives. The check is a null count on the subquery's key. `NOT EXISTS` does not have this problem because it tests row existence, not value equality.

**SQL-F10 — the reconciliation gap.** A small persistent percentage difference against a financial system is almost always convention, not error: rounding at line versus order level, tax in or out, refunds netted by original period versus refund period. Reconcile a 200-order sample and classify each difference into a bucket; if the buckets do not sum to the total difference, there is a fourth cause you have not found. This entry is ILLUSTRATIVE because the specific conventions are organisation-specific.

**SQL-F11 — the self-updating report.** Diff the entity ID sets across two runs. Added rows usually mean late data crossed the cutoff; removed rows usually mean a dimension attribute changed and the entity no longer satisfies a filter; changed values with identical ID sets mean a mutable attribute was joined without versioning. All three fixes are the same in spirit: make the query a function of a declared cutoff and immutable history.

**SQL-F12 — the non-monotone funnel.** The check is an explicit `HAVING` that verifies `|S_j| <= |S_{j-1}|` per stage, or a set-difference query. Non-monotonicity means either the steps are not ordered (a user can jump straight to step 3) or the windows differ. Both are definitional errors, not data errors, and both are caught by the subset check before any chart is drawn.

**SQL-F13 — percentiles.** Four values `10, 20, 30, 40`. `PERCENTILE_CONT(0.5)` with linear interpolation gives `25`; `PERCENTILE_DISC(0.5)` gives `20`; `AVG` gives `25` too, which is why the `AVG` error hides on symmetric data and appears on skewed data (`1, 2, 100` gives `AVG = 34.33` and median `2`). The check is to run all three on a small hand-checkable set and record which the metric definition requires.

**SQL-F14 — the join that slows down.** Not a correctness bug, but it belongs here because the "fix" often breaks correctness. A predicate on a function of a column (`WHERE date_trunc('day', ts) = '2026-02-03'`) may prevent index use; a cast (`WHERE id::text = other.id`) almost certainly does. The safe rewrite is a half-open range on the raw column, which preserves both sargability and the declared window semantics. Verification is plan equality plus an identical result set on a sample.

**SQL-F15 — two teams, two numbers.** The most efficient resolution is not to compare queries but to compare intermediate counts: the entity count after the spine, after the status reduction, and after the final join. The first step at which they diverge names the defective choice. The durable fix is a shared view with the four choices — time zone, cutoff, status set, dedup rule — encoded once.

### 8.3 Edge cases that are not failures

Some results look wrong and are not. A conversion rate of exactly 100% on a stratum with one eligible unit is arithmetically correct and statistically meaningless. A cohort curve that rises at horizon 5 is possible when the cohort is defined by first *purchase* and a repeat purchase is possible after a lull. A `LEFT JOIN` that returns fewer rows than the left table is impossible; if you observe it, you are not looking at a left join. Distinguishing "wrong" from "correct but uninformative" saves a great deal of debugging time, and the discipline is the same: state the expected value first, then compare.

**Takeaway:** each failure has a cheap check that discriminates among causes, and running the check before writing the fix is what separates diagnosis from guesswork.

## 9. Complexity & Cost Analysis

### 9a. Algorithmic complexity

Define sizes precisely, because "the query is slow" almost always means "which `n` grew?". Let `F` be fact rows, `D` dimension rows, `g` the number of output groups, `k_max` the maximum join multiplicity on the many side, `B` the build-side rows materialised in memory, and `w` the row width in bytes. Costs below are models, not measurements: actual plans choose among nested-loop, hash, and merge strategies based on statistics, and a model that ignores the choice will mispredict.

| Operation | Input sizes | Time | Space | I/O | Assumptions / dominant bottleneck |
|---|---|---|---|---|---|
| Full scan + filter | `F` | `O(F)` | `O(1)` | `O(F·w)` | Sequential read; predicate sargable |
| Hash group-by | `F`, `g` groups | `O(F)` expected | `O(g·w_g)` | `O(F·w)` | Hash table fits memory; otherwise spill |
| Sort-based group-by | `F` | `O(F log F)` | `O(F)` or `O(1)` with external merge | `O(F·w)` plus temp writes | Memory insufficient for hash table |
| Hash join | `F` probe, `D` build | `O(F + D)` expected | `O(D·w)` | `O(F·w + D·w)` | Build side fits memory; uniform hash |
| Nested-loop join with index | `F` probes | `O(F · log D)` | `O(1)` | `O(F · random reads)` | Index exists and is selective |
| Nested-loop join without index | `F × D` | `O(F·D)` | `O(1)` | `O(F·D·w)` | No index; small inputs only |
| Window with sort | `F` | `O(F log F)` | `O(F)` for the sort buffer | `O(F·w)` | Partition order differs from scan order |
| Window without sort | `F` | `O(F)` | `O(frame)` | `O(F·w)` | Input already sorted by partition/order |
| `COUNT(DISTINCT x)` exact | `F` | `O(F)` expected | `O(distinct)` | `O(F·w)` | Distinct set fits memory |
| Approximate distinct (HLL, `m` registers) | `F` | `O(F)` | `O(m)` per group | `O(F·w)` | Fixed relative error, mergeable |
| Anti/semi-join via hash | `F`, `D` | `O(F + D)` | `O(D)` | `O(F·w + D·w)` | Right side hashed once |
| Self-join for previous row | `F` | `O(F log F)` join | `O(F)` | `O(F·w)` | Equivalent to `LAG`, more expensive |

The single most useful consequence of the table: a fact-to-detail join converts an `O(F)` aggregate into an `O(F · k_max)` intermediate, and in the Section 5 data `k_max = 3` per order but the aggregate output is nine times smaller than the naive intermediate. Where `k_max` is large — tags per item, events per session — the difference is the difference between a query that runs and a query that spills to disk.

### 9b. Worked sizing

**Workload (ILLUSTRATIVE).** One week of retail orders: 1,200,000 orders; 3 events per order on average (3.6 million event rows); 3 line rows per order (3.6 million line rows). Row widths assumed at 90 bytes (orders), 120 bytes (events), 70 bytes (lines) including alignment overhead.

**Environment.** Single-node analytical engine, sequential scan throughput assumed at 500 MiB/s for a wide table, hash-aggregate memory budget assumed at 2 GiB. These are assumed planning numbers, not measurements.

**Step 1 — data volume.**

```text
orders  = 1,200,000 × 90 B  = 108,000,000 B = 102.997 MiB
events  = 3,600,000 × 120 B = 432,000,000 B = 411.987 MiB
lines   = 3,600,000 × 70 B  = 252,000,000 B = 240.326 MiB
total   = 792,000,000 B     = 755.310 MiB = 0.7376 GiB
```

Check: `108,000,000 + 432,000,000 + 252,000,000 = 792,000,000`. MiB conversion uses `2^20 = 1,048,576`, so `792,000,000 / 1,048,576 = 755.310` MiB. Using decimal MB instead gives `792.0 MB`; the two differ by 4.9%, which is why units must be stated.

**Step 2 — scan time.**

```text
t_scan = 755.310 MiB / 500 MiB/s = 1.511 s
```

This is a floor, not a latency estimate. It assumes one sequential pass, no decompression, no network transfer, and no queueing. A columnar format that reads only three of twenty columns would reduce it; a row-store with cold cache would increase it by an order of magnitude.

**Step 3 — the fan-out cost.**

```text
naive intermediate rows = 1,200,000 × 3 × 3 = 10,800,000
naive intermediate bytes = 10,800,000 × 210 B = 2,268,000,000 B = 2163.0 MiB
```

That exceeds a 2 GiB hash budget, which means the 10.8-million-row intermediate either spills to disk or is never materialised at all because the plan is smart. The grain-first version produces 1.2 million rows in roughly `1,200,000 × 200 B = 240,000,000 B = 228.9 MiB` of grouping state, comfortably in memory.

```text
FORMULA [SQL-16]: I_naive = N × k_e × k_l ; I_grain = N
WHERE: N entities [rows]; k_e events per entity [rows/entity]; k_l child rows per
       entity [rows/entity]; I intermediate rows [rows].
CONSTRAINT: k_e and k_l constant or bounded per entity; no filter changes the
            multiplicity between the two sides.
NUMERICAL EXAMPLE: N = 1,200,000; k_e = k_l = 3; I_naive = 10,800,000; I_grain = 1,200,000;
                   ratio = 9.
INTERPRETATION: the grain-first pattern is a 9× reduction in intermediate volume
                for this workload, which is the difference between fitting a budget
                and spilling.
```

**Step 4 — the cost of the wrong answer.** Suppose the naive path yields 52.4% instead of 60.0% for a threshold decision that gates a campaign worth 100,000 currency units. The expected value of deciding on the wrong number is not computable from the data alone, but the *cost of the check* is: `COUNT(*)` versus `COUNT(DISTINCT key)` on the joined relation is one extra aggregate over 10.8 million rows, roughly 0.5–2 seconds of the same scan. A check costing two seconds is worth running before a decision worth six figures, and that comparison — cheap check versus expensive error — is the whole economic argument for this sheet.

**Step 5 — annualised storage and scan volume.**

```text
weekly bytes    = 792,000,000 B
annual bytes    = 52 × 792,000,000 = 41,184,000,000 B = 38.356 GiB = 0.03746 TiB
annual scan at 500 MiB/s = 41,184,000,000 / (500 × 1,048,576) s ≈ 78.55 s
```

The annual scan is under two minutes of pure I/O at this throughput, which is why the dominant cost in analytical SQL is almost never the data volume — it is the intermediate cardinality, the sort, and the number of times the same expensive computation is repeated across a dashboard.

**Step 6 — approximate distinct sizing.** For 500,000 distinct keys per week, an exact `COUNT(DISTINCT)` needs a hash set of roughly `500,000 × 44 B = 22,000,000 B = 20.98 MiB` per group, which is fine for one group and impossible for 500 groups. A HyperLogLog sketch with `m = 4096` registers (about 12 KiB) has a standard relative error of approximately `1.04/√m = 1.04/64 = 1.625%`; with `m = 16384`, `1.04/128 = 0.813%`. Whether 1.6% is acceptable is a business question: it is acceptable for a traffic trend and not acceptable for a billing count.

**Bottleneck and production difference.** The model above says the scan is trivial and the intermediate is decisive. In production, the plan may sort 10.8 million rows, spill a 2 GiB hash table, transfer data across a network shuffle in a distributed engine, and run concurrently with a dozen peer queries competing for the same memory. Every one of those can multiply the wall-clock time by an order of magnitude without changing the row counts. The measurement that confirms the bottleneck is the plan's actual row counts versus estimated, plus spill counters — not the `EXPLAIN` text alone.

**Economics.** At an illustrative `3.00` currency units per TiB scanned, a weekly run costs `3.00 × 0.03746 = 0.112` units in scan charges; the grain-first reduction in intermediate volume saves nothing on scan charges but saves memory, spill, and the engineering time of a query that fails at month end. Pricing is illustrative and must be replaced with the actual contract before any budget decision.

### 9c. Where the model misleads

- **Big-O is not latency.** A 10.8-million-row hash join in memory is fast; the same join with a 40-byte key hashing to one hot partition is slow. Skew, not complexity, dominates distributed joins.
- **Estimated rows are a guess.** Statistics go stale after a bulk load, and a plan built on a stale estimate of `k_max` chooses the wrong join strategy. Refreshing statistics is usually cheaper than rewriting the query.
- **Sorting is not always needed.** A window whose partition and order match the input's clustering may avoid a sort entirely; materialising `ORDER BY (partition_key, order_key)` in a table or indexed view can turn `O(n log n)` into `O(n)`.
- **Repeated scans are the real cost.** A dashboard issuing twelve queries over the same fact table scans it twelve times. The fix is a materialised entity-grain intermediate refreshed once, not micro-optimising each query.

**Takeaway:** size the intermediate, not the input, because the intermediate is where grain bugs and cost blow-ups both live.

## 10. Key Theorems / Results Reference

The results below are the formal backing for the practical rules. Each is stated precisely enough to use, with its hypotheses, because every one of them is routinely quoted without them.

| Theorem/result | Correct statement | Essential hypotheses | Practical implication | What it does not imply | Source |
|---|---|---|---|---|---|
| Natural-join cardinality | For relations `R` and `S` joined on key `k`, `|R ⋈ S| = Σ_v n_R(v) · n_S(v)` over values `v` in the key domain | Key domains comparable; equality semantics; nulls do not match | Predict join size before running; `|R ⋈ S| = |R|` iff `S` is key-unique | Nothing about output row *order* or plan | DERIVED from bag semantics |
| Bag (multiset) semantics | SQL relations are bags: `SELECT` without `DISTINCT` preserves duplicates and their multiplicities | Standard SQL semantics | `COUNT(*)` counts bags, not sets; every dedup must be explicit | Not true in relational algebra's set model | DERIVED |
| Three-valued logic | With `UNKNOWN` for null comparisons, `WHERE` keeps rows evaluating to `TRUE` only; `NOT UNKNOWN = UNKNOWN` | Nullable columns; SQL standard semantics | `NOT IN` over nullable subqueries yields nothing; `NULL` propagates through arithmetic | Does not apply to `IS NULL` / `IS DISTINCT FROM` | SQL standard |
| `GROUP BY` functional dependency | `GROUP BY` on a key `K` of a relation makes every other attribute functionally determined by `K`, so bare column references are unambiguous | `K` is a key (unique, non-null) | Grouping by a true entity key is safe and often allows bare columns | Grouping by a non-key makes bare columns arbitrary; some engines reject, some silently pick | DERIVED |
| Inclusion–exclusion | `|A ∪ B| = |A| + |B| − |A ∩ B|`; for `k` sets, `2^k − 1` terms | Finite sets | Explains why daily uniques do not sum to weekly uniques | Does not give the overlap without computing it | DERIVED (combinatorics) |
| Weighted-mean identity | `Σc_k / Σe_k = Σ (e_k/Σe_j)(c_k/e_k)` | `e_k > 0` for all included `k` | Pooled rate equals the `e`-weighted mean of stratum rates | Does not say which weighting is correct for the decision | DERIVED |
| Change decomposition | `Δr = Σw_kΔr_k + Σr_kΔw_k + ΣΔw_kΔr_k` | Disjoint, exhaustive, identically defined strata | Separates performance change from mix change; the three terms sum exactly to the total | Not a causal decomposition; it is arithmetic | DERIVED (algebraic identity) |
| HyperLogLog error | Relative standard error approximately `1.04/√m` for `m` registers | Uniform hashing; cardinality well below hash-range limits | Sizing a sketch to a required accuracy | Not a worst-case bound; no membership listing | Standard result (Flajolet et al.); cite primary source before quoting numerically |
| Amortised hash-join cost | Expected `O(|R| + |S|)` with a uniform hash and a build side that fits memory | Uniform hashing; no skew; memory sufficient | Hash join is the default choice for equi-joins | Says nothing about skew, spill, or distributed shuffle | DERIVED under stated assumptions |
| Window frame semantics | A frame is a contiguous subrange of the partition defined by `ROWS` (physical) or `RANGE` (value-based, peers included) | Partition and order defined; explicit frame recommended | `ROWS` and `RANGE` give different answers in the presence of ties | Does not specify engine defaults; those are dialect-specific | SQL standard |
| Sort lower bound | Comparison sorting requires `Ω(n log n)` comparisons in the worst case | Comparison-based model | Explains the cost of `ORDER BY` and sorted windows | Radix/hash approaches can beat it under restricted key domains | Classic result (comparison model) |

### 10.1 How to use a theorem without overclaiming

A theorem tells you what follows from its hypotheses. When you quote one in a review, name the hypothesis you are relying on and the one you cannot verify. For example: "the hash join is `O(n + m)` *assuming the build side fits in memory*; our `EXPLAIN` shows a 2.2 GiB build against a 2 GiB budget, so the assumption fails and the cost model does not apply." That sentence is more useful than the complexity class alone, because it identifies the exact premise to test.

The same discipline applies to the two results most often overclaimed in analytical work:

- **Inclusion–exclusion is exact for sets, not for samples.** If you estimate the overlap from a sample, the union estimate inherits sampling error, and the three terms' errors do not cancel.
- **The change decomposition is arithmetic, not causal.** A positive within-stratum term does not prove the intervention worked; traffic composition could change within strata. It localises *where* to look, which is a smaller and more honest claim.

### 10.2 Results that are engine-specific, and must be rechecked

| Behaviour | Why it varies | How to verify on your engine |
|---|---|---|
| Default window frame | Standard specifies `RANGE ... CURRENT ROW`; historical implementations differ | Emit a window with peers and compare with explicit `ROWS` |
| `NULL` ordering in `ORDER BY` | `NULLS FIRST`/`LAST` default differs | Sort a column containing `NULL` and read the first/last row |
| Integer division of `/` | Some engines return integer for integer operands; others return numeric | `SELECT 2/3` and read the type and value |
| `GROUP BY` bare-column rule | Strict (`ONLY_FULL_GROUP_BY`) versus permissive | Group by a non-key and select a non-aggregated column |
| String comparison collation and case | Collation-dependent equality | Compare `'A' = 'a'` and a trailing-space pair |
| `DISTINCT ON` availability | PostgreSQL extension | Attempt the syntax; check the engine's documentation |
| `FILTER (WHERE ...)` on aggregates | Standard but not universal at all versions | Attempt `COUNT(*) FILTER (WHERE TRUE)` |

Each row is a five-minute check and a potential silent wrong answer. The list is deliberately short and honest: it names the behaviours where a reasonable analyst has been burned, not every dialect difference.

**Takeaway:** a theorem is only as strong as the hypothesis you verified, and the fastest review question is "which assumption did you check?"

## 11. Interview / Exam Questions

Fifteen questions across recall, application, and judgment. Each has a checked answer, the reasoning that earns credit, a follow-up, an acceptable alternative answer, and a red flag. "Checked" means the arithmetic was recomputed independently; nothing here was asked by a named employer.

### Q1 — Definition / recall

**Prompt.** In one sentence, what is the grain of a query?

**Answer.** The real-world unit that one output row represents, stated together with the time scope and membership rule that make it testable.

**Reasoning.** Any of the three parts can be absent and the others still sound complete. "One row per order" does not say whether cancelled orders are included; "orders placed in February" does not say what one row holds. Credit requires all three.

**Follow-up.** What single test proves the claim? `COUNT(*) = COUNT(DISTINCT grain_key)` on the full result set, with no `LIMIT`.

**Acceptable alternative.** "The finest level of detail the result can be disaggregated to" — defensible if the candidate adds that it must be enforced by a uniqueness test.

**Red flag.** Answering with the `GROUP BY` list. Grouping is a mechanism; the grain is the claim about the world that the mechanism is supposed to enforce.

**Scoring.** 0: defines grain as "the columns you group by". 1: mentions the unit but not scope. 2: unit plus scope plus a test. 3: adds that fan-out and null keys are the two ways the claim silently fails.

**Support.** Sections 3.1, 3.2, `[GRAIN-A]`.

### Q2 — Definition / recall

**Prompt.** Why does `NOT IN (SELECT key FROM other)` sometimes return zero rows, and what should be used instead?

**Answer.** If the subquery yields any `NULL`, every comparison evaluates to `UNKNOWN`, so nothing survives `WHERE`. Use `NOT EXISTS`, or filter nulls from the subquery.

**Reasoning.** `x NOT IN (1, 2, NULL)` expands to `x <> 1 AND x <> 2 AND x <> NULL`. The last conjunct is `UNKNOWN` for every `x`, and `TRUE AND UNKNOWN` is `UNKNOWN`. `NOT EXISTS` tests row existence and never compares against a null value, so it is immune.

**Follow-up.** Does `NOT IN` over a guaranteed-non-null primary key have the same problem? No — but "guaranteed" must be enforced by a `NOT NULL` constraint, not by belief.

**Acceptable alternative.** `LEFT JOIN ... WHERE right.key IS NULL` is correct and equivalent for a single non-null key, but fails to preserve left rows if the right table has duplicate keys on a different column.

**Red flag.** "It's a bug in the engine." This is standard three-valued logic, not an implementation defect.

**Scoring.** 0: no explanation. 1: says "nulls". 2: explains `UNKNOWN` propagation and offers `NOT EXISTS`. 3: adds the `NOT NULL` constraint as the durable prevention.

**Support.** Sections 3.5, SQL-F09.

### Q3 — Definition / recall

**Prompt.** State the difference between `ROW_NUMBER`, `RANK`, and `DENSE_RANK` on the values `100, 100, 90` ordered descending.

**Answer.** `ROW_NUMBER` gives `1, 2, 3`; `RANK` gives `1, 1, 3`; `DENSE_RANK` gives `1, 1, 2`.

**Reasoning.** `ROW_NUMBER` is positional and invents an arbitrary order among peers. `RANK` assigns the same rank to peers and then skips the positions they consumed. `DENSE_RANK` assigns the same rank without skipping, so it counts distinct values.

**Follow-up.** For "top 2 products must include all ties", which do you use? `RANK() <= 2` returns both tied leaders; `ROW_NUMBER() <= 2` returns one leader and the runner-up, which is wrong.

**Acceptable alternative.** None — the three outputs are fixed by definition, and answering with the wrong one is an arithmetic error.

**Red flag.** Treating them as interchangeable "row number functions", which produces a report whose population changes when the data gains a tie.

**Scoring.** 0: wrong outputs. 1: correct outputs without the tie explanation. 2: correct plus tie explanation. 3: adds the downstream consequence for top-N selection under fixed capacity.

**Support.** Section 4.3 (V-08).

### Q4 — Derivation / apply

**Prompt.** Two stores: store A has 1,000 eligible sessions and a 10% conversion rate; store B has 2,000 eligible sessions and a 20% rate. What is the pooled rate, and what is the unweighted average of the two rates? Which should be reported, and when is the other defensible?

**Answer.** Pooled `= (100 + 400)/3,000 = 500/3,000 = 16.667%`. Unweighted average `= (0.10 + 0.20)/2 = 15.0%`. Report the pooled rate for a site-wide metric; the unweighted average is defensible only when the question is "what does a typical store look like", which is a different estimand.

**Reasoning.** The pooled rate is the `e`-weighted mean, `Σc/Σe`, which follows from the definition of a proportion over the union of both populations. The unweighted mean gives a 1,000-session store the same influence as a 2,000-session store, so it answers a question about the distribution of stores, not about traffic.

**Follow-up.** Add store C with 500 sessions and a 40% rate. Pooled `= (100 + 400 + 200)/3,500 = 700/3,500 = 20.0%`; unweighted `= (0.10 + 0.20 + 0.40)/3 = 23.333%`. The unweighted figure now *overstates* the pooled rate, because the highest rate sits on the smallest store — the sign of the error depends on the correlation between size and rate, which is why the error cannot be corrected by a constant.

**Acceptable alternative.** Reporting both, labelled, is often the best answer in practice, provided the decision uses the pooled figure.

**Red flag.** Averaging percentages without checking the denominators. The resulting number is internally consistent and externally wrong.

**Scoring.** 0: averages the rates and stops. 1: computes the pooled rate. 2: computes both and says which to report. 3: extends to the third store and explains that the direction of the bias depends on the size–rate correlation.

**Support.** Sections 4.2, `[SQL-07]`, counterexample 2 in 4.4.

### Q5 — Derivation / apply

**Prompt.** Monday had 600 unique visitors, Tuesday had 500. What is the two-day unique count? State what you need to know and what you can say without it.

**Answer.** The union is `600 + 500 − overlap`. Without the overlap you can bound it: it lies in `[600, 1100]`. If 100 visitors appear on both days, it is exactly 1,000.

**Reasoning.** Inclusion–exclusion, `|A ∪ B| = |A| + |B| − |A ∩ B|`, is exact for sets. Since the overlap cannot be negative, the sum of daily uniques is an upper bound (at most 1,100, achieved only when no visitor appears twice). Since the union contains each daily set, the union is at least the larger day, so the lower bound is `max(600, 500) = 600` (achieved when Tuesday's visitors are a subset of Monday's). The exact union therefore lies in `[600, 1100]`, and the overlap pins it down.

**Follow-up.** Extend to a week with seven daily values. The naive sum needs `2^7 − 1 = 127` terms for exact inclusion–exclusion, so compute the distinct count directly on the unioned set or use a mergeable sketch.

**Acceptable alternative.** "I would query `COUNT(DISTINCT visitor_id)` over the two-day range" is the correct engineering answer and should be offered before the arithmetic.

**Red flag.** Adding 600 and 500 and reporting 1,100. This is wrong whenever any visitor returns, and returning visitors are usually the population the business cares about.

**Scoring.** 0: 1,100. 1: says "not simply additive". 2: states inclusion–exclusion and the bound. 3: adds the sketch/unioned-set approach and the `2^k − 1` term count.

**Support.** Section 4.4, counterexample 1; SQL-F07.

### Q6 — Derivation / apply

**Prompt.** You have `SUM(x) OVER (ORDER BY d)` and the result differs from a colleague's on a table where two rows share the same date. Give two mechanisms and the fix.

**Answer.** The engines may be using different default frames (`RANGE` versus `ROWS`), and with peer dates `RANGE ... CURRENT ROW` includes all peers, so both same-date rows show the end-of-date cumulative value while `ROWS` shows an intermediate value. Fix: write the frame explicitly as `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` and add a unique tie-breaker to `ORDER BY`.

**Reasoning.** Frames are defined over the ordered partition; `ROWS` counts physical positions while `RANGE` operates on ordering *values* and therefore cannot split a peer group. With unique dates the two coincide, which is why the discrepancy only appears when a tie exists.

**Follow-up.** Which is correct for a "daily cumulative" report? `RANGE` gives the end-of-day value on every row of that day, which is what a daily report usually wants; `ROWS` gives a within-day partial that is meaningless to a reader who sees one row per date.

**Acceptable alternative.** Pre-aggregating to one row per date before the window makes the question moot and is often the better engineering answer.

**Red flag.** "The engine is broken" or "just add `DISTINCT`". `DISTINCT` runs after window evaluation, so it cannot repair a frame that saw the wrong rows.

**Scoring.** 0: no mechanism. 1: names one mechanism. 2: names both and the fix. 3: adds that pre-aggregation removes the ambiguity and that `DISTINCT` is not a repair.

**Support.** Sections 4.3 (V-05), 6.3, SQL-F04.

### Q7 — Application

**Prompt.** Write the shape of a query for "latest status per order as of a cutoff" without using a window function, and say when you would prefer the window version.

**Answer.** Aggregate-then-join: take the maximum `(event_ts, seq)` per order in a subquery, then join back on the full ordering tuple to retrieve the row. Prefer the window version when you need several attributes from the chosen row and want a single pass, or when the engine's sort is cheaper than the extra join.

**Reasoning.** The aggregate gives the winning key tuple; the join on `(order_id, event_ts, seq)` retrieves the corresponding row. The danger is ties on `event_ts` alone, which produce multiple rows after the join — this is exactly why the tuple includes `seq`. The window version computes the same rank in one pass but requires a sort unless the input is already clustered.

**Follow-up.** What if the status table has 40 columns and you only need three? Both patterns read the whole row in a row-store; a columnar store with late materialisation favours the window version.

**Acceptable alternative.** `DISTINCT ON (order_id) ... ORDER BY order_id, event_ts DESC, seq DESC` on PostgreSQL, which is equivalent and usually the cheapest.

**Red flag.** Joining on `MAX(event_ts)` alone. Any order with two events in the same second returns two rows, and the "latest" status becomes a coin flip.

**Scoring.** 0: `MAX(status)` as a string, which is lexicographic nonsense. 1: max timestamp join without the tie-breaker. 2: full tuple join. 3: adds the window alternative and the plan tradeoff.

**Support.** Section 4.3 (V-03, V-04), SQL-F03.

### Q8 — Application

**Prompt.** Design the query for "customers who bought SKU-1 but never SKU-2" and state the maximum possible output row count.

**Answer.** `SELECT c.customer_id FROM customers c WHERE EXISTS (buy SKU-1) AND NOT EXISTS (buy SKU-2)`. Output ≤ the customer count.

**Reasoning.** Both predicates are existence tests over the purchase relation, so neither multiplies rows. Writing it as two joins and a `DISTINCT` risks fan-out for customers with multiple purchases of SKU-1 and produces a different population if the join's null behaviour differs.

**Follow-up.** What if a customer bought and then returned SKU-2? The anti-join must be over a purchases relation filtered by the return status, or over a net-purchases view; "never bought" and "does not currently own" are different populations.

**Acceptable alternative.** `NOT IN` is acceptable only after proving the purchase key is non-null; otherwise it silently returns nothing.

**Red flag.** `LEFT JOIN purchases ON sku = 'SKU-2' WHERE purchases.id IS NULL` combined with an inner join for SKU-1 without checking multiplicities.

**Scoring.** 0: inner join plus `DISTINCT` with no cardinality reasoning. 1: correct pattern without the row-count argument. 2: correct plus the bound. 3: adds the return-status distinction.

**Support.** Sections 4.3 (V-11, V-12), 7.3 Case B.

### Q9 — Application

**Prompt.** A funnel reports 1,000 at step 1, 420 at step 2, and 310 at step 3. Give the two step-conversion rates and the end-to-end rate, then state the check that proves the funnel is well formed.

**Answer.** `420/1,000 = 42.00%`, `310/420 = 73.81%`, end-to-end `310/1,000 = 31.00%`. The check is `|S3| ⊆ |S2| ⊆ |S1|`, which holds iff `580 + 110 + 310 = 1,000`.

**Reasoning.** Step rates are conditional on reaching the previous step; the end-to-end rate is conditional on entering. Reporting 31% at every stage double-counts the first drop-off.

**Follow-up.** Suppose step 3 counts 320. The subset check fails. Two causes: the time-ordering predicate is missing (users reach step 3 without step 2), or the windows differ per step. The discriminating check is to add `t3 >= t2` and see whether the count falls to 310 or below.

**Acceptable alternative.** Reporting only the end-to-end rate is defensible for an executive summary, but the step rates are what identify where to intervene.

**Red flag.** "310 out of 1,000 at step 3, so step 3 has a 31% conversion rate" — that is the end-to-end rate mislabelled as a step rate.

**Scoring.** 0: 31% everywhere. 1: correct rates, no check. 2: correct rates plus subset check. 3: adds the two causes of non-monotonicity and the discriminating test.

**Support.** Section 4.3 (V-09), 6.4, SQL-F12.

### Q10 — Judgment

**Prompt.** A stakeholder says the weekly report "changed on its own". Give your diagnostic sequence in order, and say what each outcome implies.

**Answer.** (1) Diff the entity ID sets between the two runs. (2) Count late-arriving events at the cutoff. (3) Check dimension attributes for mutation. (4) Check for `LIMIT` without a total order or a non-deterministic tie. Added IDs imply late data; removed IDs imply a filter or dimension change; same IDs with different values imply an unversioned join; identical output implies the stakeholder is comparing different filters.

**Reasoning.** Each check discriminates a different mechanism, and the order is chosen by cost and frequency: late data is common and cheap to measure, dimension mutation is rarer, and plan instability is rarer still.

**Follow-up.** If the report must be stable *and* correct, what do you publish? A value with an explicit cutoff timestamp and a stated restatement policy, so stability is a property of the artefact rather than of luck.

**Acceptable alternative.** Reconciling aggregate totals first is cheaper than ID-level diffing and will localise the period even if it cannot localise the row.

**Red flag.** Re-running the query and declaring it fixed because the number did not change this time. Non-determinism often needs several runs to manifest.

**Scoring.** 0: "check the data". 1: one plausible cause. 2: ordered checks with implications. 3: adds the cutoff-plus-restatement-policy publication rule.

**Support.** Section 8 (SQL-F11), 4.2 `[SQL-05]`.

### Q11 — Judgment

**Prompt.** Your conversion metric is 112%. Walk through the diagnosis and give the bounded fix.

**Answer.** Print numerator, denominator, and their difference. A positive difference means the numerator is at a finer grain (fan-out or a distinct-count at a different level) or the denominator is filtered harder; equal-but-different windows is the third possibility. Recompute both from one CTE at one grain and one window.

**Reasoning.** The metric is defined as a subset ratio, so `numerator ≤ denominator` is a theorem, not a hope. A violation is a proof that the implementation does not match the definition, which is why the fix is definitional, not statistical.

**Follow-up.** After the fix the rate is 40% and the stakeholder preferred 112%. What do you say? Show the two row-counted and entity-counted derivations side by side and explain that 112% cannot be a proportion of a population; offer the entity-level figures with the population sizes.

**Acceptable alternative.** If the metric was intended as "orders per customer" rather than a proportion, then 112% is a category error in the *label*, and the fix is to rename the metric and change its display.

**Red flag.** Clamping to 100% or applying `LEAST(numerator, denominator)`. Both hide a defect that will resurface with a different sign.

**Scoring.** 0: clamps the value. 1: says "grain mismatch" without the check. 2: gives the check and the fix. 3: adds the label-versus-definition distinction and refuses to clamp.

**Support.** Sections 4.1, 4.4, SQL-F05.

### Q12 — Judgment

**Prompt.** A dashboard issues twelve queries over the same 3.6-million-row fact table and takes 40 seconds. Where do you optimise, and what would you refuse to do?

**Answer.** Materialise an entity-grain intermediate once per refresh, and point the twelve queries at it. Refuse to add indexes blindly or to add caching at the dashboard layer, because both hide the cardinality problem rather than fixing it.

**Reasoning.** The modelled scan is about 1.5 seconds per pass at 500 MiB/s, so twelve passes cost roughly 18 seconds of pure I/O before any join or sort. The dominant term is repetition, not per-query inefficiency. A materialised intermediate reduces both the scan count and the intermediate cardinality, which is where the fan-out cost lives.

**Follow-up.** The metric definition is still changing weekly. Then materialisation is premature; instead consolidate the twelve queries into one grouped query that returns all the needed slices, so the fact table is scanned once.

**Acceptable alternative.** A materialised *view* with the same effect, if the engine refreshes it automatically and the definition is stable.

**Red flag.** Adding an index to every filtered column and declaring victory. On an analytical scan, indexes help selective predicates and do nothing for a full-table aggregate.

**Scoring.** 0: "add indexes". 1: identifies repetition. 2: proposes materialisation with the reasoning. 3: adds the "definition still moving" counter-case and the single-pass consolidation alternative.

**Support.** Section 9b, steps 2 and 3; Section 9c.

### Q13 — Judgment

**Prompt.** Your retention curve is flat at 100% for every horizon. What went wrong and how do you prove the fix?

**Answer.** The denominator is being recomputed per period as "users active in the previous period", so every surviving user counts as retained by construction. Freeze `|C|` at cohort entry, store the membership snapshot, and prove the fix by checking `R(0) = 1` exactly and `R(d) ≤ R(d−1)` for each `d`.

**Reasoning.** A retention curve is `|A(d) ∩ C| / |C|` with `C` immutable. Recomputing `C` at each horizon changes the estimand to active-of-active, which is 100% whenever the same definition is used for numerator and denominator.

**Follow-up.** A cohort curve rises at horizon 5. Is that necessarily a bug? No — if the cohort is defined by first purchase, a later repeat purchase is legitimate re-engagement. The check is that the *set* at horizon 5 is a subset of `C`, not of `A(4)`.

**Acceptable alternative.** Reporting active-of-active alongside retention is fine if it is labelled, because it measures stickiness among survivors; it must not be called retention.

**Red flag.** "100% retention is great news." It is a definitional artefact and the most reliable sign that the denominator moved.

**Scoring.** 0: celebrates the number. 1: suspects the denominator. 2: names the frozen-cohort fix. 3: adds monotonicity checks and the legitimate-rising-curve case.

**Support.** Section 4.3 (V-10), 6.5, SQL-F08.

### Q14 — Judgment

**Prompt.** Two analysts produce 42% and 47% for "February conversion". Both queries look correct. How do you resolve it without arbitrating by seniority?

**Answer.** Compare intermediate entity counts at each stage: after the spine, after the status reduction, after each join. The first divergent stage names the offending choice. Then reconcile to a shared definition of the four choices — time zone, cutoff, status set, dedup rule — and encode them in one view.

**Reasoning.** Two end-to-end numbers cannot localise a difference; staged counts can. The four choices are the only degrees of freedom in a definition this simple, so the divergence must be one of them, and the stage at which it appears identifies which.

**Follow-up.** Both are defensible (one uses event time, the other processing time). Then the resolution is a definitional decision by the metric owner, documented with the cutoff timestamp; the analysts' job is to make the trade-off visible, not to pick silently.

**Acceptable alternative.** Diffing the SQL line by line works but scales badly and misses differences hidden in shared views.

**Red flag.** Averaging the two numbers, or choosing the one that matches the previous month's trend.

**Scoring.** 0: picks one by seniority or recency. 1: diffs the SQL. 2: staged counts. 3: adds the four-choice definition of record and the owner's decision.

**Support.** Section 8 (SQL-F15), 5.7 reconciliation.

### Q15 — Judgment / verification

**Prompt.** You are asked to sign off a KPI that will determine a budget allocation. What do you actually run before signing, and what would make you refuse?

**Answer.** Run four things: the grain test `[GRAIN-A]`; a row-count comparison across every join against the preserved-side count; a hand reconciliation of 200 sampled entities from source IDs to the aggregate; and a two-method derivation of the same number (for example, a running total to the final period versus a one-shot filter). Refuse if any of the four fails and cannot be explained, or if the numerator and denominator cannot be shown to share a population and a window.

**Reasoning.** Each check targets a different contract from Section 2.1: grain, cardinality, membership, and numeric. Passing three and failing one is a fail, because the contracts are independent — a correct grain with a duplicated denominator is still a wrong number.

**Follow-up.** You have one hour. Which single check? The 200-entity hand reconciliation, because it exercises all four contracts at once on a tractable sample and produces evidence a reviewer can re-run.

**Acceptable alternative.** A pipeline test suite that enforces the same checks on every refresh is strictly better than a manual sign-off, and saying so is a strong answer.

**Red flag.** "I'd check it against last month." A plausible trend is not evidence of correctness; two wrong numbers can trend consistently.

**Scoring.** 0: "eyeball the chart". 1: one check. 2: several checks with rationale. 3: adds the refusal criteria and the pipeline-test alternative.

**Support.** Sections 2.1, 5.7, 5.10, 9b step 4.

**Takeaway:** the strongest interview answers in this area describe a check that can fail, not a definition that sounds right.

## 12. Quick Reference Card

| Item | Value |
|---|---|
| Topic | Declare, check, and defend the grain, join cardinality, time window, and denominator of an analytical SQL query; diagnose the number when it moves. |
| Core formula | `r = c / n_eligible` with `c ⊆ n_eligible` and both over one half-open window. **Critical validity cue:** valid only after each unit is counted once — check `COUNT(*) = COUNT(DISTINCT unit_key)`; if a join multiplies rows, the row-counted rate is `Σm_e y_e / Σm_e`, a different estimand. |
| Key assumption | Every join on the preserved side has multiplicity exactly 1 at the metric's grain, and the "latest" row is determined by a total order (event time plus a monotone tie-breaker), not by timestamp alone. |
| Complexity | `O(F)` scan and hash aggregate for `F` fact rows; `O(F + D)` hash join with a build side that fits memory; `O(F log F)` when a window or `ORDER BY` needs a sort. Sizing example: 1.2 M orders + 3.6 M events + 3.6 M lines = 755.3 MiB, about 1.51 s of pure scan at an assumed 500 MiB/s; the naive 3×3 join intermediate is 10.8 M rows (2163 MiB), 9× the entity-grain output. |
| Use when | You are producing a ratio, a trend, a cohort curve, a funnel, or a "latest state" table that someone will act on, and you can name the population and cutoff. |
| Avoid when | The right-hand side of a join is not unique on the key and you intend to aggregate at entity grain; the "latest" ordering has unresolved ties; or the measurement is a distinct count that will be summed across periods — all three require a different pattern, not a different query. |
| Numeric invariant | `0 ≤ numerator ≤ denominator`; a rate outside this range proves a grain, filter, or window mismatch before any statistical test is needed. Exhaustive category numerators must sum exactly to the denominator. |
| Failure diagnostic | When a KPI moves, first compare `COUNT(*)` with `COUNT(DISTINCT key)` across each join and compare entity ID sets between runs. Added IDs mean late data; removed IDs mean a filter or dimension mutation; identical IDs with changed values mean an unversioned join. |
| Operational guardrail | Print the cutoff timestamp and the lateness allowance beside every published number, use `NOT EXISTS` rather than `NOT IN`, write window frames explicitly, and cast before dividing. |
| Go deeper | Physical plan and optimizer interpretation, transaction isolation, and orchestration are out of scope here; within the collection, continue to the API/contract sheet for interface semantics and to the data-engineering material for ingestion and cutoff enforcement. |
| Evidence limit | All tables and workloads are ILLUSTRATIVE; every arithmetic result was recomputed independently and is DERIVED, but no query was executed against a live engine, and engine-specific behaviours in Section 10.2 must be rechecked on the target version. |
