---
title: "SQL Window Frames and Aggregate Semantics"
topic: "window frames, partition semantics, ranking ties, aggregate evaluation"
collection: "python-sql-linux-engineering-basics"
level: "core"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
notation_style: "plain-text fenced blocks"
prerequisites: ["GROUP BY aggregation", "join cardinality", "date arithmetic", "NULL three-valued logic"]
pattern_tags: ["window-frame", "running-total", "ranking-ties", "lag-lead", "frame-exclusion"]
engine_stance: "PostgreSQL 14+ syntax; portability differences named in place"
owner_role: "analytics engineer"
source_index: "INDEX.md"
review_trigger: "Engine major-version change, warehouse dialect migration, or metric frame definition change"
---

# SQL Window Frames and Aggregate Semantics

## 1. File Header

> **Purpose:** Make the *frame* — the exact set of rows an aggregate sees for each output row — an explicit, reviewable part of every analytical query, so that rolling totals, moving averages, deltas, and ranks can be recomputed by hand and defended in review.
> **Intended reader decision:** *Given this rolling metric, this rank, or this period-over-period delta, is the frame the one the business question needs — and if the number disagrees with the spreadsheet, which frame bound explains the gap?*
> **Prerequisites:** `GROUP BY` and `HAVING`, primary-key versus foreign-key vocabulary, date arithmetic with intervals, and the rule that `NULL` is neither equal nor unequal to anything.
> **Key outcome:** After this sheet you can (a) state in one sentence which rows each output row aggregates, (b) predict the difference between `ROWS`, `RANGE`, and `GROUPS` frames *before* running the query, (c) choose `RANK`, `DENSE_RANK`, or `ROW_NUMBER` from the tie policy the business actually wants, and (d) diagnose an off-by-one-day or off-by-one-row discrepancy from arithmetic alone.
> **This file does NOT cover:** `GROUP BY` grain reasoning and join fan-out (handoff: `05_SQL_Analytics_Patterns.md`), recursive CTEs and graph traversal, query-optimizer cost modelling beyond the window-specific part in Section 9, and procedural scheduling of incremental refreshes (handoff: data-engineering module).
> **Level:** core, with advanced selection reasoning in Sections 7 and 9 and production diagnosis in Section 8.
> **Evidence:** `DERIVED`. Every semantic statement is derived from the SQL standard's window model and from arithmetic recomputed by hand over ILLUSTRATIVE rows; no query in this file was executed against a live server, so engine-specific details carry an explicit "verify on target version" marker instead of a claimed observation.
> **Retrieval cue:** If a running total jumps on a day with duplicate timestamps, start at 4.1 and 4.4. If a rolling window is short by a day at the edges, start at 4.2. If a rank skips numbers, start at 4.3.

### 1.1 Why the evidence level is DERIVED and not VERIFIED

The distinction matters to a reviewer, so it is stated rather than implied. `VERIFIED` would mean the statements were checked by executing the queries on a named engine version and recording the output. That did not happen here. What did happen is stronger than a summary from memory but weaker than execution: the frame membership of every row in Section 5 was enumerated by hand, and the resulting sums were cross-checked against an independent prefix-sum computation, so the arithmetic is reproducible from the tables printed in this file. Any reader can re-derive each number with a calculator and the eight-row table, which is the property `DERIVED` promises.

The semantics themselves are standard-model statements: the frame is a multiset of rows of the current partition, bounded by a lower and an upper bound, and the aggregate is evaluated once per row over that multiset. That model is defined in SQL:2011 and implemented with cosmetic differences across PostgreSQL, SQL Server, Oracle, Snowflake, BigQuery, and DuckDB. Where an implementation detail is known to vary — default null ordering, availability of `EXCLUDE`, availability of `FILTER`, whether `GROUPS` exists — the text names the variation and asks the reader to confirm on the target version rather than asserting a behaviour that may have changed.

### 1.2 The one-line frame discipline

If you retain one habit from this sheet, retain this: **write the frame clause, even when the default would have been correct.** A frame left implicit is a frame nobody reviewed, and the default is not "the obvious rows" — it is `RANGE UNBOUNDED PRECEDING TO CURRENT ROW`, which silently merges rows that share an ordering key. Section 5.3 shows a four-row partition where the implicit frame overstates one running total by 10 currency units, and Section 8 shows why the reviewer could not see it. Explicit frames turn a semantic argument into a diff.

### 1.3 Reading order

Read Sections 2 and 3 for the vocabulary of partition, peer, and frame bound. Read Section 4 in order: 4.1 establishes the default, 4.2 generalises the bounds across the three frame modes, 4.3 covers the clause modifiers, and 4.4 is the set of counterexamples that makes the earlier formulas safe to apply. Section 5 is one retail-order dataset carried through every calculation in the file; later sections cite its row identifiers. Sections 7 and 9 are selection and cost lookup; Sections 8 and 10 are diagnosis; Sections 11 and 12 are practice and recall.

### 1.4 What "aggregate semantics" adds beyond frame syntax

Frame syntax is memorised in an afternoon. Aggregate semantics is the harder half, and it is the half that decides
whether a number is right. Three semantic questions sit underneath every frame clause, and each has an answer that
is independent of the engine.

The first is *membership*: which rows are in the frame for this output row. That is the frame clause, and it is the
only part most readers check. The second is *identity*: does the aggregate skip nulls or count rows. `SUM` skips,
`COUNT(*)` counts, `COUNT(expr)` skips, `AVG` skips and divides by the count of non-nulls rather than by the frame
size. That second answer explains why an average over a frame containing nulls is not the sum divided by the frame
cardinality, and why adding nulls to a frame can change a sum's membership without changing its value.

The third is *ordering sensitivity*: does the result depend on the order rows arrive. Cumulative aggregates do not,
once the frame is fixed, because addition is commutative. Ordered-set aggregates do, and so does anything involving
`FIRST_VALUE` or `LAST_VALUE`, which read a position rather than a set. A query mixing an ordered-set aggregate with
a frame has two independent ordering dependencies, and a review that checks only one of them is incomplete.

The three questions also sequence a review well. Ask membership first, because a wrong frame makes every later
question moot. Ask identity second, because it is a one-word answer per function and it explains most surprises in
averages. Ask ordering sensitivity last, because it only matters for a handful of functions and only when the
ordering is non-total.

## 2. Concept Map / Mental Model

### 2.1 Three nested sets: partition, order, frame

A window function does not see "the table". It sees a partition of the rows that survived `WHERE`, `GROUP BY`, and `HAVING`, and inside that partition it sees a frame determined by one row's position. The nesting is the whole mental model, and every bug in Section 8 is a confusion between two of the three levels.

```text
TABLE (after FROM / JOIN / WHERE / GROUP BY / HAVING)
  |
  +-- PARTITION BY cust            <- splits rows into independent groups
        |
        +-- ORDER BY order_date    <- sequences rows inside each partition
              |
              +-- FRAME          <- selects a contiguous slice around CURRENT ROW
                    |
                    +--> aggregate value for this output row
```

Three consequences follow immediately and are worth stating because they remove most confusion at the source.

1. `PARTITION BY` is *not* `GROUP BY`. It does not collapse rows; every input row produces exactly one output row, and the partition key is merely an address used to find neighbours.
2. `ORDER BY` inside `OVER` is *not* the statement's `ORDER BY`. One controls frame position and peer detection; the other controls presentation and is applied last.
3. The frame is a *contiguous run* in the window ordering. It can never be "all rows with amount above 100" — that is a `WHERE` or a `FILTER`, not a frame.

### 2.2 Evaluation order and why it is a design constraint

The logical evaluation order for a `SELECT` that uses windows is: `FROM` and joins, then `WHERE`, then `GROUP BY`, then `HAVING`, then window evaluation, then `SELECT` list projection and `DISTINCT`, then statement-level `ORDER BY`, then `LIMIT`. Two practical rules fall out of that sequence.

A window result cannot be referenced in `WHERE` or `HAVING` of the same query block, because those clauses run before windows exist. Filtering on a rank or on a running total requires a subquery or CTE that computes the window, with the predicate applied one level out. A query that tries `WHERE rn = 1` with `rn` defined in the same `SELECT` list fails to parse for precisely this reason, and the error message is about the column not existing rather than about evaluation order, which is why the rule is memorised rather than re-derived each time.

Window evaluation happens before `DISTINCT`. If a query computes `ROW_NUMBER()` and then applies `DISTINCT`, duplicate removal cannot rescue a nondeterministic row choice: the row was already chosen, the duplicates were already labelled with different numbers, and `DISTINCT` now sees them as distinct. Deduplication must be `ROW_NUMBER()` inside a subquery plus `WHERE rn = 1` outside, never `DISTINCT` plus a window.

### 2.3 Peers: the concept that separates RANGE from ROWS

Two rows are *peers* when they are tied on every column of the window `ORDER BY` — that is, the comparator cannot distinguish them. Peers are the reason the default frame includes more than one row's worth of data on a tie, and the reason `RANK` skips numbers. A frame expressed in `RANGE` or `GROUPS` units is expressed in peer-group units, whereas a frame in `ROWS` units cuts rows arbitrarily through a tie.

```text
order_date     amount    peers?         ROWS frame cuts   RANGE/GROUPS cut
2026-01-01     100       alone          after row 1       after group 1
2026-01-03      50   <-- tied -->       after row 2       after group 2
2026-01-03      25   <-- tied -->       after row 3       (both rows together)
2026-01-07     200       alone          after row 4       after group 3
```

The diagram is the entire distinction: `ROWS 1 PRECEDING` can separate the two 3 January orders, while `RANGE 1 DAY PRECEDING` and `GROUPS 1 PRECEDING` cannot. Whether separation is desirable depends on the question: a per-order sequential ledger wants `ROWS`; a per-day cumulative revenue wants `RANGE` or `GROUPS`, because a day is an indivisible business unit and splitting it makes the intraday numbers meaningless.

### 2.4 Two families of window function

Aggregate window functions (`SUM`, `COUNT`, `AVG`, `MIN`, `MAX`, plus statistical variants) read the frame. Ranking and offset functions (`ROW_NUMBER`, `RANK`, `DENSE_RANK`, `NTILE`, `LAG`, `LEAD`, `FIRST_VALUE`, `LAST_VALUE`, `NTH_VALUE`) read the partition ordering. That split predicts which of them can be repaired by editing the frame clause and which cannot — a distinction that saves time in code review, because adding `ROWS BETWEEN ...` to a `RANK` call is a no-op that looks like a fix.

## 3. Foundational Definitions

**Window function.** A function evaluated over a set of rows related to the current row, returning one value per input row without collapsing them. Contrast with an aggregate function under `GROUP BY`, which returns one value per group.

**Partition.** The subset of rows sharing identical values of the `PARTITION BY` expressions. `NULL` forms its own partition: all rows whose partition key is `NULL` are grouped together, because window partitioning uses "not distinct from" semantics rather than `=`. This differs from `GROUP BY` only in that `GROUP BY` also puts nulls in one group, so the two agree here; the difference people trip over is in joins, not windows.

**Window ordering.** The `ORDER BY` inside `OVER`, defining sequence and peers. Its direction changes what "preceding" means: with `ORDER BY ts DESC`, `1 PRECEDING` is the row with the next-larger timestamp, because preceding is relative to traversal order, not to value magnitude.

**Peer group.** A maximal run of rows that the window ordering cannot distinguish. A partition with keys 1, 2, 2, 2, 5 has four peer groups and five rows.

**Frame.** For the current row, the contiguous sub-multiset of its partition from which the aggregate reads. It is defined by a frame mode and two bounds. "Contiguous" is in window order over the extended sort key that includes the ordering expressions and the peer tie-break.

**Frame mode.** How bound offsets are counted: `ROWS` counts physical rows, `RANGE` counts value distance on the single ordering column, `GROUPS` counts peer groups.

**Bound.** One of `UNBOUNDED PRECEDING`, `<n> PRECEDING`, `CURRENT ROW`, `<n> FOLLOWING`, `UNBOUNDED FOLLOWING`. `CURRENT ROW` means "this row" in `ROWS` mode and "this row's peer group boundary" in `RANGE` and `GROUPS` mode — the single most consequential definition in the sheet.

**Default frame.** What the engine uses when `OVER` has an `ORDER BY` but no frame clause: `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`. Without `ORDER BY`, the default is the entire partition. Both defaults are specified by the standard, so this is portable behaviour rather than a vendor quirk.

**Exclusion.** A modifier subtracting rows from an otherwise computed frame: `EXCLUDE CURRENT ROW`, `EXCLUDE GROUP`, `EXCLUDE TIES`, `EXCLUDE NO OTHERS` (the default, which subtracts nothing).

**Running aggregate.** An aggregate whose frame starts at the partition start and ends at the current row's position, so each output row reports a cumulative value.

**Moving (sliding) aggregate.** An aggregate whose frame is bounded on both sides with a fixed width, so values enter and leave as the current row advances.

**Deterministic ordering.** An ordering that distinguishes every row in the partition, achieved by appending a unique key as a final `ORDER BY` term. Without it, `ROWS` frames and `ROW_NUMBER` are permitted to return different answers on different executions, because ties may be broken arbitrarily.

**Frame-independent function.** A function whose result does not change when the frame clause changes: all ranking functions and `LAG`/`LEAD`. Recognising this class prevents wasted effort in debugging.

**Null ordering.** Placement of `NULL` values in the window ordering. PostgreSQL and Oracle default to `NULLS LAST` for ascending order and `NULLS FIRST` for descending; SQL Server sorts nulls lowest, which coincides with `NULLS FIRST` ascending. Because the default is not portable, state `NULLS FIRST` or `NULLS LAST` explicitly whenever the ordering column is nullable.

### 3.1 Symbols used in this sheet, with units

The formulas in Section 4 use a small symbol set, and each symbol has a type that constrains which frames are legal.
`P` is a set of rows and carries no unit. `i(q)` is a 1-based row ordinal and is dimensionless. `g(q)` is a peer-group
ordinal, also dimensionless, and satisfies `g(q) <= i(q)` always, with equality only when every prior group is a
singleton. `k(q)` is the ordering value and inherits the column's type: a date, a timestamp, a numeric, or a string.

`amount(q)` is the aggregated expression, in currency units in every example here. `V(r)` shares that unit. The offsets
`a` and `b` are integers in `ROWS` and `GROUPS` mode, and carry the type of the ordering expression in `RANGE` mode: an
interval for a date, a number for a numeric. Treating those two interpretations as interchangeable in review is a real
source of confusion, even though mixing them in a query is a syntax error in every engine.

`W` is a window width in days, used only in 5.7, and `C(d)` is a cumulative total in currency units at date `d`.
`share_excl(r)` is dimensionless and lies in `[0, 1]` whenever all amounts are non-negative.

### 3.2 The vocabulary that separates a frame bug from a data bug

Two adjectives do most of the diagnostic work in Section 8, and they are worth fixing precisely. A *scope* defect is
one where the wrong rows reached the query at all: a missing filter, an extra join, a partition key taken from the
wrong table. A *frame* defect is one where the right rows reached the query but the aggregate read the wrong subset
of them. The distinction matters because the two have different owners and different tests.

Scope defects change row counts. Frame defects do not. That single observable separates them in almost every case,
and it costs one `COUNT(*)` comparison to obtain. If the base population of the window query matches the population
of an independent grouped query, the rows are right and the defect is in the frame or the ordering. If the counts
differ, no amount of frame editing will fix the number, because the arithmetic is being done over the wrong set.

A third category, *cardinality* defects, sits between the two: the rows are the right entities but appear more than
once. They inflate both the count and the aggregate, and they are the only category where the correct repair is
upstream of the query. Section 8.8 treats them because they present as window bugs, and misdiagnosing them wastes
the most time of the three.

### 3.3 What "aggregate semantics" means for a metric definition

A metric definition and a window frame are two descriptions of the same thing, and they must agree or the metric is
undefined. "Trailing seven-day revenue per order" names a population (orders), a denominator (seven days, one per
order), and an aggregation (`SUM` of amounts, refunds included). The frame clause is the executable form of the first
two. If the definition says "per calendar day" then the population changes to days and the query must pre-aggregate,
because a frame cannot change grain.

Reading a metric definition as a frame has a diagnostic payoff: disagreements between two teams usually reduce to a
difference in one of the three, and each can be tested. Population differences show up as row-count differences at a
chosen date. Denominator differences show up as a ratio that is constant across rows when it should vary. Aggregation
differences show up as an offset that equals the excluded class, such as the refund total.

The habit worth building is to write the frame clause before writing the metric's prose. The clause forces every
choice into the open, and the prose then has nothing to hide behind.

### 3.4 Terminology that differs between engines

Four names have local variants that cost time in a migration. `ROWS BETWEEN ... AND ...` is universal. Windowing
itself is called `OVER` everywhere, but Oracle documentation discusses "analytic functions" where PostgreSQL says
"window functions", and job postings echo both. `FILTER` is called a "conditional aggregate" in some MySQL material,
which describes the concept rather than the clause. `GROUPS` mode is occasionally described as "peer-group frame",
which is clearer than the standard's name.

The vocabulary matters less than the mapping, and the mapping is one line: an analytic function is a window function,
a conditional aggregate is `FILTER` or its `CASE` rewrite, and a peer-group frame is `GROUPS`. Anyone who can state
that mapping can read documentation from the other three engines without translating the conceptual model.

### 3.5 Event time versus processing time inside a frame

A frame's ordering key is a value in the data, so it describes event time: when the order happened, not when the row
arrived. A late-arriving row with an old `order_date` therefore changes history when it lands, because the frames
whose key ranges include that date must be recomputed. This is not a window-function defect; it is the ordinary
consequence of computing aggregates on event time over a table that is still receiving events.

Two designs avoid the surprise. The first is to snapshot: compute the metric once per day over a fixed cut-off and
store it, so late arrivals affect only future snapshots and the historical series is stable. The second is to accept
mutation and re-run affected partitions, which requires the pipeline to know which frames contain the new row — a
range determined by the frame width, hence one more reason the width belongs in the metric specification rather
than only in the query.

A frame over processing time (ingestion timestamp) has the opposite property: it never changes retroactively, and it
misattributes events to the day they arrived. Choosing between the two is a business decision about which error is
cheaper, and it cannot be read off the query text.

## 4. Core Formulas with Full Derivation

### 4.1 Base case

The base case is the one every other case generalises: an ordered window with no frame clause, aggregating a `SUM` from the start of the partition to the current row. Getting its membership rule exactly right is what makes the counterexamples in 4.4 legible, because every one of them is a single violated assumption of this case.

```text
FORMULA
  For output row r in partition P, the default frame F_default(r) is

      F_default(r) = { q in P : key(q) <=_ord key(r) }

  where <=_ord is the window ordering comparator, extended so that peers
  compare equal. The reported value is

      V(r) = SUM_{q in F_default(r)} amount(q)

WHERE
  P        : rows sharing the PARTITION BY key of r; type set<row>
  key(q)   : the ORDER BY expression value of row q, in window order
  <=_ord   : total preorder over keys; equal keys are peers, not ordered
  amount   : the aggregated expression; numeric, NULL allowed but skipped
  V(r)     : cumulative sum at r; same unit as amount; monotone only if
             amount >= 0 for every row

CONSTRAINT
  The rule "key(q) <= key(r)" is a value test, not a positional test.
  Therefore every peer of r satisfies it, including peers that appear
  after r in an arbitrary physical scan. This is the assumption that
  Section 4.4 breaks deliberately.

NUMERICAL EXAMPLE
  Partition B (customer B), ordered by order_date only:

      row   order_date    amount
      B1    2026-01-02      30
      B2    2026-01-05      90
      B3    2026-01-05      10
      B4    2026-01-12      40

  B1: peers = {B1}          -> V = 30
  B2: peers = {B2, B3}      -> V = 30 + 90 + 10 = 130
  B3: peers = {B2, B3}      -> V = 30 + 90 + 10 = 130
  B4: peers = {B4}          -> V = 30 + 90 + 10 + 40 = 170

INTERPRETATION
  On 5 January the cumulative figure is 130 for both of that day's orders.
  The sheet is internally consistent — a daily cumulative revenue should
  not depend on which of the day's two orders is printed first — but the
  number 130 at B2 already contains B3's 10 units, which a reader who
  assumes "current row" means "this row only" will not expect.
```

Computing the same frame from two different rows of one peer group and requiring identical answers is the cheapest arithmetic check available, because the first instinct when reading row B2 is to stop the sum at 90 and add only the current row, producing 120. Any hand computation that yields 120 and 130 for B2 and B3 has mistaken the default frame for a row frame. A reviewer who computes the frame twice and gets different answers has found either an arithmetic slip or a genuine non-determinism, and distinguishing those two is the subject of Section 8.

The monotonicity clause in the `WHERE` block deserves its own sentence because analysts routinely assume it. A cumulative sum of non-negative amounts is non-decreasing, so a running total that dips is proof that (a) some amount is negative — a refund or a credit — or (b) the frame is not what the author believed. That deduction is free and should be used before any query is rewritten.

### 4.2 General case

The general frame is a pair of bounds evaluated against the current row. Deriving it once, in a form that covers all three frame modes, removes the need to memorise six syntax variants and makes the mode differences a matter of one function definition.

```text
FORMULA
  A frame is a pair (lower, upper). Each bound is a triple
  (mode, direction, offset) where direction is PRECEDING or FOLLOWING and
  offset is a non-negative integer (or an interval in RANGE mode).

  Define a positional index i(q) for each row q in P, in window order,
  with peers sharing a value of the ordering key. Then

      ROWS  mode:  frame at r = { q : i(r) - a <= i(q) <= i(r) + b }
      GROUPS mode: frame at r = { q : g(r) - a <= g(q) <= g(r) + b }
      RANGE mode:  frame at r = { q : k(r) - a <= k(q) <= k(r) + b }

  where
      i(q)  : row ordinal in the partition, 1-based, tie-broken arbitrarily
      g(q)  : peer-group ordinal, 1-based, ties share a value
      k(q)  : the single ordering value, numeric or temporal
      a, b  : resolved offsets; UNBOUNDED maps to +/- infinity

WHERE
  a is the distance from the current row back to the lower bound and
  b is the distance forward to the upper bound; both are counts in ROWS
  and GROUPS mode and value deltas in RANGE mode.

CONSTRAINT
  RANGE mode requires exactly one ORDER BY expression, of a type that
  supports addition and subtraction with the offset's type. Two ordering
  columns plus a RANGE offset is a syntax error, not a semantic subtlety.
  An absent bound defaults to CURRENT ROW for the upper side and
  UNBOUNDED PRECEDING for the lower side when only one is written.

NUMERICAL EXAMPLE
  Partition A, ordered by order_date, amounts 100, 50, 25, 200, 60 on
  2026-01-01, 01-03, 01-03, 01-07, 01-10. Take the frame
  RANGE BETWEEN INTERVAL '6 days' PRECEDING AND CURRENT ROW.

      A1 01-01  lower 2025-12-26  members {A1}            -> 100
      A2 01-03  lower 2025-12-28  members {A1,A2,A3}     -> 175
      A3 01-03  lower 2025-12-28  members {A1,A2,A3}     -> 175
      A4 01-07  lower 2026-01-01  members {A1,A2,A3,A4}  -> 375
      A5 01-10  lower 2026-01-04  members {A4,A5}        -> 260

  The same five rows under ROWS BETWEEN 6 PRECEDING AND CURRENT ROW,
  tie-broken by order_id, give 100, 150, 175, 375, 435.

INTERPRETATION
  The two frames agree at A3 and A4 and disagree at A2 by 25 and at A5 by
  175. Agreement at a tie's last row and disagreement at its first is the
  signature of ROWS-versus-RANGE confusion: ROWS splits the tie, RANGE
  cannot. The A5 gap is not a tie effect at all — it is window width, and
  Section 5 separates the two causes.
```

The formula exposes a subtlety that syntax hides: `CURRENT ROW` resolves differently per mode. In `ROWS` mode it is `i(r)`, one row. In `GROUPS` and `RANGE` mode it is the *far edge* of the current row's peer group, which for an upper bound means the last peer and for a lower bound means the first peer. That asymmetry is why `RANGE BETWEEN CURRENT ROW AND CURRENT ROW` covers an entire tie group, while `ROWS BETWEEN CURRENT ROW AND CURRENT ROW` covers exactly one row. The two clauses have identical text and different cardinality, so any review that compares queries by text rather than by resolved frame will miss the difference.

A second consequence worth stating: bounds must not cross. `ROWS BETWEEN 1 FOLLOWING AND 1 PRECEDING` is rejected by the standard and by every mainstream engine, because the frame would be empty for every row and the query is almost certainly a typo rather than an intentional no-op. Engines differ on whether the rejection happens at parse time or at plan time, which affects how early the mistake is caught in development but not whether it is caught.

### 4.3 Variants

Four families of variant change what the base case returns without changing the partition or the ordering: exclusion modifiers, the `FILTER` clause, offset functions with boundary defaults, and the tie-sensitive ranking trio.

```text
FORMULA
  (a) EXCLUDE variants subtract from an already-computed frame F:

        EXCLUDE NO OTHERS  ->  F
        EXCLUDE CURRENT ROW->  F \ {r}
        EXCLUDE GROUP      ->  F \ peers(r)
        EXCLUDE TIES       ->  F \ (peers(r) \ {r})

  (b) FILTER restricts the aggregated rows before the aggregate runs:

        SUM(amount) FILTER (WHERE amount >= t) OVER w
            == SUM(CASE WHEN amount >= t THEN amount END) OVER w

  (c) LAG(e, n, d) reads the row n positions before r in window order,
      returning d when no such row exists; the frame is irrelevant.

  (d) Ranking on an ordering with peer groups of sizes s_1..s_m:

        ROW_NUMBER:  1, 2, ..., |P|                     (unique, unstable)
        RANK:        cumulative boundary  1 + SUM_{j<g} s_j
        DENSE_RANK:  g, the peer-group ordinal

WHERE
  peers(r)  : rows tied with r on every window ORDER BY term
  t         : threshold constant or correlated expression, same type as amount
  n         : integer offset, >= 0; n = 0 returns the current row
  d         : default value; type must be coercible to the expression type
  s_j       : cardinality of peer group j; SUM over all j equals |P|

CONSTRAINT
  EXCLUDE requires an explicit frame clause in the standard; writing
  EXCLUDE with no frame is allowed by some engines and rejected by others.
  FILTER is standard SQL and supported by PostgreSQL and SQLite, while
  SQL Server and MySQL require the CASE rewrite; both forms are
  semantically identical when the aggregate ignores NULLs.

NUMERICAL EXAMPLE
  Partition A, partition total 435. For row A2 (amount 50, date 01-03):
      EXCLUDE CURRENT ROW over UNBOUNDED PRECEDING..UNBOUNDED FOLLOWING
          -> 435 - 50 = 385
      EXCLUDE GROUP over the same frame
          -> 435 - (50 + 25) = 360
      EXCLUDE TIES over the same frame
          -> 435 - 25 = 410
  Ranking by order_date within A: dates 01-01, 01-03, 01-03, 01-07, 01-10
      ROW_NUMBER -> 1, 2, 3, 4, 5
      RANK       -> 1, 2, 2, 4, 5
      DENSE_RANK -> 1, 2, 2, 3, 4
  LAG(amount, 2) over A:  NULL, NULL, 100, 50, 25

INTERPRETATION
  EXCLUDE GROUP and EXCLUDE TIES differ only when the tie is not a
  singleton, and the 360-versus-410 gap is exactly the peer's amount.
  Choosing between them is a statement about the business question: a
  share-of-customer calculation usually wants EXCLUDE CURRENT ROW (leave
  peers in), while a leave-one-day-out baseline wants EXCLUDE GROUP.
```

The `FILTER` equivalence in (b) is worth internalising because it makes the two dialects interchangeable during review. It holds because `SUM` ignores `NULL` inputs: the `CASE` expression yields `NULL` for excluded rows, and `SUM` skips them. The equivalence breaks for `COUNT(*)` — which counts rows, not non-null values — so a `COUNT(*) FILTER (WHERE ...)` must be rewritten as `SUM(CASE WHEN ... THEN 1 ELSE 0 END)` and not as `COUNT(CASE WHEN ... THEN 1 END)`, which happens to work, or as `COUNT(*)` on a `CASE` that yields `NULL`, which does not.

The ranking table in (d) is the most useful three-line result in the sheet, and it is worth deriving rather than memorising. `RANK` reports the position of the *first* row of the peer group, which is one plus the number of rows strictly before it, hence the skip. `DENSE_RANK` reports the peer-group ordinal, hence no skip. `ROW_NUMBER` reports an arbitrary total order refined through the tie, hence unique values whose assignment is unspecified unless the ordering is deterministic. A query asking for "the top three products" and returning four rows has used `RANK` on a tie at the boundary, and that is correct behaviour for `RANK`; the defect is in the requirement, not the function.

### 4.4 Limits and counterexamples

Four limits bound the safe use of everything above. Each is stated as a failure of one assumption from 4.1, with the arithmetic that exposes it.

```text
COUNTEREXAMPLE 1 — the default frame is not "rows so far"
  Query: SUM(amount) OVER (PARTITION BY cust ORDER BY order_date)
  Partition B, order_date 01-02, 01-05, 01-05, 01-12; amounts 30, 90, 10, 40

      RANGE default (what runs):  30, 130, 130, 170
      ROWS  1-rows-so-far (what the author believed): 30, 120, 130, 170

  The gap at B2 is 10 units, equal to the amount of the *later* peer B3.
  That is 8.33 percent of the correct 120 at that row, with no error,
  no warning, and an unchanged row count.

COUNTEREXAMPLE 2 — RANGE does not accept two ordering columns
  SUM(x) OVER (ORDER BY d, id RANGE BETWEEN 6 PRECEDING AND CURRENT ROW)
  is rejected: RANGE with an offset needs exactly one ordering term.
  The repair is not to drop id from the ORDER BY (that reintroduces the
  peer merge above) but to switch to ROWS or to pre-aggregate to one row
  per day and keep RANGE on that.

COUNTEREXAMPLE 3 — LAST_VALUE with the default frame
  LAST_VALUE(amount) OVER (ORDER BY order_date) returns the current row's
  peer-group-relative last value, not the partition's last value, because
  the default upper bound is CURRENT ROW. On partition A it yields 100 at
  A1, then 50 or 25 at the two 3 January rows (whichever peer the engine
  places last), then 200 at A4 and 60 at A5 — a sequence that reads like a
  permuted copy of the input. Adding ROWS BETWEEN UNBOUNDED PRECEDING AND
  UNBOUNDED FOLLOWING makes it 60 for every row, which is what "the last
  value" means in ordinary English.

COUNTEREXAMPLE 4 — NULL ordering is not portable
  ORDER BY ship_date with two NULL ship dates: PostgreSQL places NULLs
  last in ascending order, SQL Server places them first. A RANGE frame
  over that column then contains different rows per engine even though
  the query text is identical.
```

Counterexample 1 is the one the assignment asks for, and its mechanism is worth spelling out in the language of 4.2 rather than as a trivia item. The author wrote no frame clause, so the engine applied `RANGE UNBOUNDED PRECEDING TO CURRENT ROW`. The author's mental model was "everything up to and including this row", which is the `ROWS` reading. Those two readings coincide whenever the ordering key is unique — which is true for every test dataset built with one row per key, and false for every production dataset built on a date. The bug therefore passes review, passes testing, and fails only on data with same-day duplicates, which is most transactional data. The size of the error equals the sum of the amounts of the *later* peers in the current group, so it is largest on the busiest days and vanishes on quiet ones — the diagnostic signature of a frame bug rather than a data bug.

Counterexample 3 is a different species: the function is not doing anything wrong, but its name suggests a frame that the default does not provide. `LAST_VALUE` is frame-relative, `FIRST_VALUE` is frame-relative, and both are commonly used with the default frame that makes `FIRST_VALUE` behave intuitively and `LAST_VALUE` behave surprisingly. The asymmetry is not arbitrary: the default lower bound is unbounded, so the first value is stable, while the default upper bound is current, so the last value moves.

The validity condition for this whole section, stated once: **every formula in 4.1 to 4.3 is safe when the window ordering is deterministic and the frame is written explicitly; when either condition fails, the result is engine-permitted but not reproducible.** Both conditions are cheap to satisfy, and both are invisible in a query diff that only shows the `SELECT` list.

## 5. Worked Numerical Example

### 5.1 The dataset

One table, `orders`, eight rows, two customers, chosen so that ties, gaps, and partition changes all appear in a dataset small enough to hold in working memory. Amounts are in whole currency units; `order_date` is `date`, not `timestamp`.

```text
order_id  cust  order_date    amount  channel
A1        A     2026-01-01      100   web
A2        A     2026-01-03       50   web
A3        A     2026-01-03       25   store
A4        A     2026-01-07      200   web
A5        A     2026-01-10       60   store
B1        B     2026-01-02       30   web
B2        B     2026-01-05       90   store
B3        B     2026-01-05       10   web
B4        B     2026-01-12       40   store
```

Partition totals: A = 100 + 50 + 25 + 200 + 60 = 435. B = 30 + 90 + 10 + 40 = 170. Grand total = 605. Row count = 8. These three numbers are the invariants every later check reconciles against, and any query in this section that violates one of them has a partition-scope bug rather than a frame bug.

### 5.2 Check 1 — seven-day RANGE frame versus seven-row ROWS frame

The question: "revenue in the trailing seven calendar days, per order". The frame is `RANGE BETWEEN INTERVAL '6 days' PRECEDING AND CURRENT ROW`, so the window spans `d - 6` through `d` inclusive, which is seven distinct dates.

```text
FORMULA
  R7(r) = SUM amount(q) for q in same partition as r with
          order_date(r) - 6 days <= order_date(q) <= order_date(r)

WHERE
  the bound is inclusive on both ends; the offset is 6, not 7, because
  day 0 (the current day) counts as one of the seven.

CONSTRAINT
  order_date must be a temporal type; INTERVAL arithmetic is exact for
  date types and has no daylight-saving hazard. With timestamp types the
  same clause still works but the window edges become instants, so a
  23:59 order and an 00:01 order are 6 days 23h58m apart, not 7 days.

NUMERICAL EXAMPLE (partition A)
  A1 2026-01-01: lower 2025-12-26, members {A1}                = 100
  A2 2026-01-03: lower 2025-12-28, members {A1,A2,A3}          = 175
  A3 2026-01-03: lower 2025-12-28, members {A1,A2,A3}          = 175
  A4 2026-01-07: lower 2026-01-01, members {A1,A2,A3,A4}       = 375
  A5 2026-01-10: lower 2026-01-04, members {A4,A5}             = 260

INTERPRETATION
  A3's 2026-01-01 member is exactly 6 days before 2026-01-03, so the
  bound is inclusive at the edge. Shrinking the offset to 1 day would
  drop A1 from A2's frame and give A2 = 75, a 100-unit swing from a
  single character in the interval literal.
```

Recomputing by prefix sums gives an independent check on the same numbers. The cumulative-by-date series for A is 100 at 01-01, 175 at 01-03, 375 at 01-07, 435 at 01-10. A window ending at 01-10 that excludes everything before 01-04 is the total minus the cumulative through 01-03, that is 435 - 175 = 260. A window ending at 01-03 that starts at 01-01 is the cumulative through 01-03 minus the cumulative before 01-01, that is 175 - 0 = 175. Two derivations, one by frame enumeration and one by subtraction of cumulative totals, agree at every row; this is the check to run whenever a sliding total is disputed.

The same five rows under `ROWS BETWEEN 6 PRECEDING AND CURRENT ROW` with tie-break `order_id` give 100, 150, 175, 375, 435. The differences are confined to A2 (150 against 175) and A5 (435 against 260), and they have different causes. At A2 the `ROWS` frame is short by exactly A3's 25 because the row count ran out before the calendar did — a tie-splitting artefact. At A5 the `ROWS` frame is long by 175 because six rows reach back to A1, seven calendar days do not — a width artefact. A single discrepancy therefore has two possible mechanisms, and Section 8 gives the discriminating test.

### 5.3 Check 2 — the default frame silently inflates a running total

Partition B ordered by `order_date` alone, so 5 January is a peer group of two rows.

```text
FORMULA
  implicit:  SUM(amount) OVER (PARTITION BY cust ORDER BY order_date)
  explicit:  SUM(amount) OVER (PARTITION BY cust ORDER BY order_date
                               ROWS BETWEEN UNBOUNDED PRECEDING
                                        AND CURRENT ROW)

WHERE
  the two queries differ only by the frame clause; partition and ordering
  are byte-identical, so any difference is attributable to the frame.

CONSTRAINT
  the implicit form is only reproducible when order_date is unique inside
  the partition; on unique dates the two forms agree exactly.

NUMERICAL EXAMPLE (partition B)
  B1 2026-01-02: implicit 30,  explicit 30
  B2 2026-01-05: implicit 130, explicit 120
  B3 2026-01-05: implicit 130, explicit 130
  B4 2026-01-12: implicit 170, explicit 170

  Relative error at B2 = (130 - 120) / 120 = 0.08333 = 8.33 percent
  Absolute error at B2 = 10 = amount(B3)

INTERPRETATION
  The implicit frame reports a running total that already includes a sale
  booked later the same day. Nothing in the result set signals this; the
  row count is unchanged at 4 and the final total is unchanged at 170, so
  both cheap reconciliation checks pass.
```

The final-total invariance is what makes this dangerous. A reviewer who reconciles the last row against the partition total sees 170 on both sides and concludes the query is correct, because the last row's frame is the whole partition in both cases. The error is transient — present in the middle rows, absent at the end — so it survives any test that only inspects the terminal value. The correct reconciliation checks the *increments*: the difference between consecutive running totals must equal the current row's amount under a `ROWS` frame, and must equal the current row's *peer group* total under the default frame. Applying that test to the implicit series gives 130 - 30 = 100 against a current-row amount of 90, an immediate mismatch of 10.

An 8.33 percent error on a mid-month cumulative revenue is not cosmetic. It is the kind of figure that appears in a weekly business review and gets attributed to a demand spike, because the same-day duplicates that cause it cluster on exactly the high-volume days. The correlation between error magnitude and business activity is the reason frame bugs are mistaken for real signal.

### 5.4 Check 3 — GROUPS frame and the cost of peer-group counting

The question: "revenue across this order and the previous order day". `GROUPS BETWEEN 1 PRECEDING AND CURRENT ROW` counts peer groups, so a two-order day is one unit.

```text
FORMULA
  G(r) = SUM amount(q) for q in peer groups g(r)-1 .. g(r)

WHERE
  g(r) is the 1-based peer-group ordinal within the partition; group
  boundaries are induced by equality on the full window ORDER BY list.

CONSTRAINT
  GROUPS is absent from SQL Server and from MySQL before 8.0 window
  support matured; the portable substitute is a pre-aggregated daily
  table with a ROWS frame over distinct dates.

NUMERICAL EXAMPLE (partition A, groups {A1}, {A2,A3}, {A4}, {A5})
  group 1 (01-01): groups 1..1     -> 100
  group 2 (01-03): groups 1..2     -> 100 + 75  = 175
  group 3 (01-07): groups 2..3     ->  75 + 200 = 275
  group 4 (01-10): groups 3..4     -> 200 + 60  = 260

INTERPRETATION
  Unlike a 2-row ROWS frame, both 3 January rows report 175, and the
  series never dips below a single day's revenue. That stability is the
  property a business reader expects from "last two order days".
```

The series is not monotone, and that is correct rather than a defect: 175 then 275 then 260 reflects a genuinely quieter second day, because the frame moves. A reader who expects a running total to be non-decreasing and sees 260 after 275 has confused a moving aggregate with a cumulative one, which is the second most common misreading of window output after the default frame itself.

A cross-check ties this back to Check 1. At A5, `GROUPS 1 PRECEDING` gives 260 and the seven-day `RANGE` also gives 260, because 7 January and 10 January are three days apart and the group frame's one-group reach coincides with the calendar frame's seven-day reach on this data. That coincidence is accidental, which is precisely why it is worth noting: two different frames agreeing on one row is not evidence that they are the same frame, and a test suite built only from rows where they agree would pass while the production numbers drifted.

### 5.5 Check 4 — EXCLUDE, FILTER, and ranking on the same rows

Five derived columns are computed over the same partition, so the arithmetic is visible side by side and each column's frame is an explicit claim.

```text
FORMULA
  share_excl(r) = SUM(amount) OVER w_all EXCLUDE CURRENT ROW
                  / SUM(amount) OVER w_all
  where w_all = (PARTITION BY cust ORDER BY order_date
                 ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)

  big_orders(r) = COUNT(*) FILTER (WHERE amount >= 50) OVER w_cum
  where w_cum = the default ordered frame

CONSTRAINT
  the ratio's denominator is the partition total 435, constant for A, so
  the ratio is defined for every row; a customer with a zero or NULL total
  would need a guard, because the division is undefined at 0.

NUMERICAL EXAMPLE (partition A, total 435, cumulative by date 100/175/375/435)
  row  amount  excl_sum  excl_share  big_orders_cum  rank  dense  rn
  A1     100       335     0.7701           1            1     1    1
  A2      50       385     0.8851           2            2     2    2
  A3      25       410     0.9425           2            2     2    3
  A4     200       235     0.5402           3            4     3    4
  A5      60       375     0.8621           4            5     4    5

  checks: 335 = 435 - 100 ; 385 = 435 - 50 ; 410 = 435 - 25
          0.7701 x 435 = 335.0 ; 0.5402 x 435 = 235.0 (rounded to 4 dp)

INTERPRETATION
  A4's share excluding itself is the smallest at 0.5402, matching its
  largest amount: the share is a leave-one-out ratio, so it falls as the
  row's own contribution rises. Ranks use order_date, hence the 2,2 tie
  and the RANK skip to 4.
```

The `big_orders_cum` column is the clearest demonstration of `FILTER` semantics: at A3 the cumulative count is 2, not 3, because A3's amount of 25 fails the threshold, and `FILTER` removed it before the count. Under the `CASE` rewrite, `COUNT(CASE WHEN amount >= 50 THEN 1 END)` also gives 2, because the `ELSE` branch is implicit `NULL` and `COUNT` skips it. Under `COUNT(CASE WHEN amount >= 50 THEN 1 ELSE 0 END)` the answer is 3 — the zero is counted as a row — and that difference between 2 and 3 is the single most common `FILTER`-to-`CASE` translation error.

### 5.6 Sensitivity analysis

Each parameter of Check 1 is perturbed by one step, holding everything else fixed, so the reader can see which choices are load-bearing.

```text
perturbation                          A5 result   delta
7 days inclusive (offset 6)               260      baseline
8 days inclusive (offset 7)               335      +75  = A2 + A3
6 days inclusive (offset 5)               260       0    (A2,A3 fall outside)
ROWS 6 PRECEDING, tie-break order_id      435      +175
GROUPS 1 PRECEDING                        260       0
GROUPS 2 PRECEDING                        335      +75
```

Two rows of that table carry the lesson. The offset from 6 to 5 changes nothing at A5 — the frame already excluded 3 January, so a slightly narrower bound removes nothing — while the offset from 6 to 7 adds A2 and A3, worth 75. Sensitivity is therefore not monotone in the parameter, and a single-value unit test on A5 would fail to distinguish offset 6 from offset 5. The second lesson is subtler. `ROWS 6 PRECEDING` yields 100, 150, 175, 375, 435 while `GROUPS 1 PRECEDING` yields 100, 175, 175, 275, 260; the two agree at A1, A3 and A4 and disagree at A2 and A5. An equality assertion placed on any one of the agreeing rows would pass for both frames and prove nothing, which is why the membership-predicate test below is written over every row.

The robust test is structural rather than numeric: assert that the frame's member count matches an independently computed membership predicate over every row, not just the last one. For the seven-day frame that predicate is one line of Python or SQL over `order_date`, and it catches offset errors at the row where they first bite rather than at the end of the series.


### 5.7 Recheck via an independent implementation

The three checks above were enumerated by frame membership. A second, mechanically different derivation catches enumeration slips: compute the cumulative series once, then obtain every sliding value by subtraction.

```text
FORMULA
  Let C(d) = SUM amount for all partition rows with order_date <= d,
  computed by a plain GROUP BY (no window at all). Then for a trailing
  window of W days ending at d,

      R_W(d) = C(d) - C(d - W days)

  where C(x) = 0 for x before the partition's first date.

WHERE
  C is defined only at dates that exist in the data, so C(d - W days)
  must be taken as the cumulative value at the greatest existing date
  strictly less than d - W days + 1, i.e. at d - (W-1) days minus one day.

CONSTRAINT
  The identity holds only when the window is calendar-based (RANGE).
  It does not hold for ROWS frames, because a row frame's boundary is not
  expressible as a function of the date.

NUMERICAL EXAMPLE (partition A, W = 7)
  C: 01-01 -> 100 ; 01-03 -> 175 ; 01-07 -> 375 ; 01-10 -> 435
  R_7(01-01) = 100 - 0            = 100
  R_7(01-03) = 175 - 0            = 175
  R_7(01-07) = 375 - 0            = 375
  R_7(01-10) = 435 - 175          = 260

INTERPRETATION
  The subtraction path reproduces 100, 175, 375, 260 exactly, matching the
  enumeration in 5.2 row for row, with no shared code and no shared error.
```

The subtraction form is worth keeping because it is the version a finance partner can check in a spreadsheet: cumulative by day, then subtract the cumulative from a week earlier. It also makes the boundary convention visible in a way the frame syntax hides. An off-by-one in `W` shows up as subtracting a cumulative value from the wrong row, which is immediately apparent in a two-column spreadsheet and invisible in a `RANGE` clause.

### 5.8 What the checks do not cover

Three properties of this dataset make it a friendlier test than production data, and naming them prevents over-confidence. Every amount is positive, so monotonicity cannot distinguish a correct cumulative from a wrong one. Every partition has at least four rows, so the short-window head-of-series case never arises. No `order_date` is `NULL`, so the null-ordering hazard in 4.4 counterexample 4 never fires.

A test suite built only from this table would pass while all three production hazards remained. The repair is a second fixture of four rows: amounts `(-50, 100, 100, 100)` on dates with one `NULL`, one singleton, and one duplicate. That fixture makes monotonicity fail once, makes the head-of-series window contain a single row, and moves the null row to a position that differs between PostgreSQL and SQL Server — three assertions, three hazards covered.


### 5.9 A second partition shape: the same metric with a channel dimension

Adding `channel` to the partition changes which comparisons are legal, and the arithmetic shows it immediately.
Partitioning by `(cust, channel)` splits A into web rows of 100, 50, 200 and store rows of 25, 60, turning two
partitions into four. The seven-day trailing revenue for the store series at 10 January is 60, because
the store rows are 3 January's 25 and 10 January's 60 and the earlier of the two is outside seven days.

The grand total is unchanged at 435, so a reconciliation check on the partition sum passes for both queries while the
per-row values differ on every web row. This is the cleanest illustration in the sheet of why the partition key
belongs in a review: it determines which rows are neighbours, and neighbours are what a frame reads.

Whether the channel belongs in the partition is a business question about whether "trailing revenue" means
all-channel or per-channel. Both are defensible; only one matches a given dashboard's label. The failure mode is
silent, because both queries run, both return one row per order, and only the numbers differ.

### 5.10 Interpreting the three checks together

The three checks measure different properties and should not be collapsed. Check 1 tests frame *width* on a calendar
axis and produces 260 at the last row of A. Check 2 tests frame *mode* on a dataset with a tie and produces an 8.33
percent error at the first tied row of B. Check 3 tests peer-group counting and produces a stable, non-monotone
series for A.

A query that passes all three is not proven correct, but a query that fails any one is proven wrong, and the failure
tells you which clause to edit. That asymmetry is the useful part: the checks are cheap falsifiers rather than
expensive verifiers, and a falsifier that reliably names the defective clause is worth more than a test that
confirms an output the author already expected.

Used together, the three also bound the error. A width error scales with the data density inside the window; a mode
error scales with the amount of the current peer group; a counting error scales with the partition size. Since those
three quantities are measurable on the source table, the likely magnitude of a suspected bug can be estimated
before the query is fixed, which is how a reviewer decides whether an incident is urgent.

## 6. Visual / Geometric Intuition

### 6.1 The frame as a window sliding over a rail

Picture the partition laid out on a rail in window order, current row under a fixed cursor, and the frame as a bracket painted on the rail. Every clause in `OVER` moves either the rail (partition), the cursor (order), or the bracket (frame).

```text
partition A, ordered by order_date

 rail:   [A1 100] [A2 50] [A3 25] [A4 200] [A5 60]
         01-01    01-03   01-03   01-07     01-10

 RANGE 6 DAYS PRECEDING, cursor at A5 (01-10):
                    |<------ 7 calendar days ------>|
 rail:   [A1 100] [A2 50] [A3 25] [A4 200] [A5 60]
                     excluded    |---- frame ----|
                                  200    +   60   = 260

 ROWS 6 PRECEDING, cursor at A5:
         |<----------- 6 rows ------------------->|
 rail:   [A1 100] [A2 50] [A3 25] [A4 200] [A5 60]
          included (only 5 exist)               = 435
```

The bracket on the `RANGE` row is anchored to *dates*; the bracket on the `ROWS` row is anchored to *row positions*. When the data has one row per date the two brackets coincide, and that is why the difference is invisible in development and visible in production.

### 6.2 Why a tie makes the ROWS bracket ambiguous

A tie means several rows occupy the same coordinate on the rail. A `ROWS` bracket counts cars, so it can stop between two cars at the same station; a `RANGE` bracket counts distance, so it stops at the platform.

```text
station 01-03 holds two rows, drawn stacked:

        A1        A2        A3        A4        A5
  ------+---------+---------+---------+---------+
       100       50        25       200        60
                 ^         ^
                 |         |
        ROWS 1 PRECEDING..CURRENT ROW at A2 -> 100+50 = 150
        ROWS 1 PRECEDING..CURRENT ROW at A3 ->  50+25 =  75
        RANGE (peer-aware) at A2 and A3    -> 100+50+25 = 175
```

The `ROWS` bracket produces two different values within one day; the `RANGE` bracket produces one. Neither is wrong. The question is whether "cumulative revenue for 3 January" is a per-order quantity (then 150 and 75 describe different positions in a ledger) or a per-day quantity (then 175 twice is the only defensible answer).

### 6.3 Peer groups as floors in a building

`GROUPS` mode is easiest to see as floors rather than metres. Rows are people; the ordering key is the floor they are on; a `GROUPS 1 PRECEDING` frame is "this floor and the one below".

```text
floor  people (rows)      floor total   GROUPS 1 PRECEDING running
  4    A5                          60        200 + 60 = 260
  3    A4                         200         75 + 200 = 275
  2    A2, A3                      75        100 + 75  = 175
  1    A1                         100              100
```

Reading upward, the frame always contains exactly two floors and between two and four rows. That is the invariant `GROUPS` guarantees and `ROWS` does not: a fixed number of *groups*, a variable number of rows.

### 6.4 The cumulative/moving distinction in one picture

```text
cumulative (lower bound pinned)      moving (both bounds move)
  |=======>                            |<--->|
  |=========>                            |<--->|
  |============>                          |<--->|
  |================>                        |<--->|
  monotone if amounts >= 0               not monotone in general
  frame grows every row                  frame width constant
```

Confusing these two produces the "why did my total go down" question. A cumulative frame's value can only fall if an amount is negative; a moving frame's value falls whenever a large row exits the window.

## 7. Comparison / Selection Table

### 7.1 Frame modes

| Mode | Offset unit | Tie behaviour | Ordering columns allowed with offset | Use when |
|---|---|---|---|---|
| `ROWS` | physical rows | splits ties arbitrarily | any number | ledger position matters; one row per event to be counted |
| `RANGE` | value distance | never splits ties | exactly one | calendar or numeric distance matters (`7 days`, `100 units`) |
| `GROUPS` | peer groups | never splits ties | any number | "previous N distinct days/values" without arithmetic on the key |

### 7.2 Default frames

| `OVER` content | Implied frame | Consequence to remember |
|---|---|---|
| `PARTITION BY` only | whole partition | every row sees the full total; safe for shares |
| `ORDER BY` only | `RANGE UNBOUNDED PRECEDING TO CURRENT ROW` | peers are merged into one value |
| `ORDER BY` plus explicit frame | as written | reproducible only if ordering is deterministic |

### 7.3 Ranking and offset functions

| Function | Tie output | Frame-sensitive | Deterministic input required | Typical question |
|---|---|---|---|---|
| `ROW_NUMBER` | 1,2,3,4,5 | no | yes, for stable assignment | pick exactly one row per key |
| `RANK` | 1,2,2,4,5 | no | no | "position, with gaps for ties" |
| `DENSE_RANK` | 1,2,2,3,4 | no | no | "how many distinct levels at or above" |
| `NTILE(n)` | near-equal buckets | no | yes, for stable buckets | decile reporting |
| `LAG`/`LEAD` | reads neighbours | no | yes, else neighbour is arbitrary | period-over-period delta |
| `FIRST_VALUE` | frame-relative | yes | yes | opening balance in window |
| `LAST_VALUE` | frame-relative | yes | yes | closing balance (needs full frame) |
| `SUM`/`AVG`/`COUNT` | aggregates frame | yes | yes for `ROWS` | running and moving metrics |

### 7.4 Choosing between a window and a self-join

| Criterion | Window function | Correlated subquery / self-join |
|---|---|---|
| Scans the partition | one ordered pass | one pass per row in naive form |
| Expresses "previous N rows" | trivially | needs row numbering and a join |
| Expresses "all rows with key < k" | `RANGE` only with a single column | natural |
| Excludes the current row | `EXCLUDE CURRENT ROW` | natural |
| Portable across engines pre-2012 | no | yes |

The last row of 7.4 is not academic: SQL Server gained window frames in 2012, MySQL in 8.0, SQLite in 3.25, so a "modern" frame query in a codebase older than those versions has to be written as a self-join. The rewrite is mechanical for running totals (join on `q.key <= r.key` and group) and awkward for moving windows (join on a band and group), and the result is usually slower by a factor that grows with partition size.

### 7.5 Engine portability matrix for the clauses in this sheet

| Clause | PostgreSQL | SQL Server | MySQL | SQLite | BigQuery |
|---|---|---|---|---|---|
| `ROWS` frame | yes | 2012+ | 8.0+ | 3.25+ | yes |
| `RANGE` value offset | yes | `UNBOUNDED` only | 8.0+ | yes | numeric only |
| `GROUPS` frame | 11+ | no | no | 3.28+ | no |
| `EXCLUDE` | 11+ | no | no | 3.28+ | no |
| `FILTER` on aggregate | yes | no | no | yes | no |
| `NULLS FIRST/LAST` | yes | no | no | 3.30+ | yes |

The matrix is a planning aid, not a specification: versions move, and a dialect can gain a clause in a minor
release. Its use is to answer one question quickly — will this sheet's syntax survive the migration — and the
answer for the most advanced clauses is no on three of five engines. When a clause is unavailable, the fallbacks
are mechanical: `GROUPS` becomes a pre-aggregated table with a `ROWS` frame, `EXCLUDE` becomes a self-join or a
subtraction from a partition total, and `FILTER` becomes a `CASE` inside the aggregate.

The `RANGE` interval row deserves a note because it is the most commonly assumed-portable clause that is not.
BigQuery accepts only numeric offsets in `RANGE`; the idiom there is to convert the date to a day number and use a
numeric offset, which changes the key's type and therefore the units in which the bound is expressed. SQL Server permits
`RANGE` only with `UNBOUNDED` bounds, so every calendar window on that engine is a self-join or a pre-aggregation.
MySQL 8.0 supports `ROWS` and `RANGE` but neither `GROUPS` nor `EXCLUDE`, so the peer-group idioms in Section 5.4 must
be rewritten as pre-aggregated `ROWS` frames there.
A team migrating a reporting layer learns this on the first rolling metric and not before.

### 7.6 Choosing the null policy explicitly

Nulls enter the ordering in three ways and each wants a decision rather than a default. A row with a null ordering
key cannot participate in a `RANGE` frame meaningfully, because interval arithmetic on `NULL` yields `NULL` and the
row is excluded from every frame including its own. A null in the partition key puts the row in the null bucket,
which is often a large phantom partition. A null in the aggregated expression is skipped by `SUM` and `COUNT(expr)`
but counted by `COUNT(*)`.

The policy that avoids most surprises is to filter null ordering keys before the window, or to coalesce them to a
sentinel date that is documented. Filtering is cleaner when the rows are genuinely undatable; a sentinel is cleaner
when the null means "not yet scheduled" and the row must appear somewhere in the report. Either way the choice must
be visible in the query, because the default placement differs between engines and a report whose ordering depends
on the engine is not a report.

## 8. Failure Modes & Edge Cases

Each entry states the symptom a user reports, the competing causes that produce that same symptom, the check that discriminates between them, and the repair.

### 8.1 Running total inflated on high-volume days

**Symptom.** Cumulative revenue is plausible early in the month and drifts above the finance figure in the last week; the daily increments do not match the daily sales report.

**Competing causes.** (a) Implicit `RANGE` frame merging same-day peers; (b) duplicated source rows from a join fan-out upstream; (c) refunds recorded as positive amounts elsewhere in the pipeline.

**Discriminating check.** Compute `running(r) - running(prev)` and compare to the current row's own amount. Under `ROWS UNBOUNDED PRECEDING`, the difference must equal `amount(r)` exactly. Under the default frame it equals the sum of the current row's whole peer group. If the difference exceeds the row amount *on exactly the rows that share a date with other rows*, the cause is (a); if it exceeds it uniformly and the row count of the base table exceeds the distinct key count, the cause is (b).

**Repair.** Add the explicit frame, and add a uniqueness assertion on `(partition key, order key)` in the model tests so a future schema change cannot silently reintroduce peers.

### 8.2 Rolling window short by one day at the start of the series

**Symptom.** The seven-day average is correct from roughly day 10 onward but too low, or absent, on days 1 to 7.

**Competing causes.** (a) A `ROWS 7 PRECEDING` frame on sparse data, where fewer than eight rows exist; (b) an off-by-one in the offset (`6` is seven days inclusive, `7` is eight); (c) timezone conversion shifting boundary events into the previous day.

**Discriminating check.** Print the frame cardinality per row. If it grows from 1 to 7 and then stays at 7, the cause is (a) and the behaviour is correct for a row frame — the metric is a row-average, not a calendar average. If the cardinality is 7 everywhere but the first computed value differs from a hand sum by one day's worth, the cause is (b). If cardinality is right but membership names a different date than expected, the cause is (c).

**Repair.** For calendar semantics use `RANGE BETWEEN INTERVAL '6 days' PRECEDING AND CURRENT ROW` and accept a smaller denominator at the series head; for row semantics keep `ROWS` and document that the denominator varies.

### 8.3 Rank skips a number and a "top 3" report returns four rows

**Symptom.** A leaderboard labelled top three contains four products; or a pagination page of 10 rows returns 12.

**Competing causes.** (a) `RANK` used where `DENSE_RANK` was intended, so the boundary tie admits both rows; (b) a genuine tie at the cut-off that the requirement never resolved; (c) `ROW_NUMBER` without a deterministic order producing inconsistent pages between runs.

**Discriminating check.** Inspect the ordered key values at the boundary. Equal keys with ranks 3 and 3 means (a)/(b) — the function behaved as specified. Unequal keys with ranks 3 and 4 means the problem is upstream. Identical keys but different `ROW_NUMBER` values across two runs means (c).

**Repair.** Decide the tie policy explicitly and encode it in the `ORDER BY`: append a business-meaningful tie-break (revenue, then product id) so `ROW_NUMBER` becomes deterministic, or filter with `WHERE rank <= 3` knowingly and label the report "top 3 incl. ties".

### 8.4 `LAG` returns `NULL` on the first row of every partition, not just the first row

**Symptom.** Month-over-month deltas are null for the first month of every customer rather than only for the first month of the dataset.

**Competing causes.** (a) `LAG` is partition-relative, so the first row of each partition legitimately has no predecessor; (b) the partition key is `NULL` for some rows and window partitioning puts all nulls in one partition, producing a large spurious partition; (c) the ordering key has duplicates, so "the previous row" is one of several arbitrary choices.

**Discriminating check.** Count rows per partition. If the count equals the expected per-customer month count, the cause is (a) and the query is right. If one partition is far larger than any customer, the cause is (b) — join the partition key back to the dimension table and check `IS NULL`. If deltas are stable within a run but change between runs, the cause is (c).

**Repair.** For (a) supply a default: `LAG(x, 1, 0)`, which converts the null into zero and makes the first delta equal the first value. That is a *definitional* choice and must be stated, because a zero delta and a null delta mean different things in a chart.

### 8.5 `LAST_VALUE` returns a permuted copy of the input

**Symptom.** The column that was supposed to hold the closing balance of the window instead repeats values that look like the raw rows in a different order.

**Competing causes.** (a) The default frame ends at `CURRENT ROW`, so `LAST_VALUE` is the last peer of the current row; (b) the intended semantics was "last row in the partition" and needs `UNBOUNDED FOLLOWING`; (c) the ordering is nondeterministic, so "last" is arbitrary within a tie.

**Discriminating check.** Compare `LAST_VALUE` against a `MAX` over the full partition. If they agree only on the final row, the cause is (a). If they disagree everywhere including the final row on a tied dataset, the cause is (c).

**Repair.** Write the frame in full: `ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING`. Consider `MAX`/`MIN` instead when the intent is an extremum rather than a positional value, since extrema are frame-robust and cheaper to reason about.

### 8.6 The same query returns different rows on different days

**Symptom.** A nightly job's top-10 list differs between two runs on unchanged data.

**Competing causes.** (a) `ROW_NUMBER` over a non-unique ordering, where the engine may break ties differently after a plan change or a parallel scan; (b) `LIMIT` without a full `ORDER BY`, so the cut is arbitrary; (c) genuinely changed data upstream, which is the null hypothesis that must be excluded first.

**Discriminating check.** Re-run the query twice within one transaction on a snapshot. Difference under a repeatable-read snapshot excludes (c) and implies (a) or (b). Then add a unique tie-break and re-run: if the output becomes stable, the cause is confirmed.

**Repair.** Always terminate the window `ORDER BY` with a unique key; always terminate the statement `ORDER BY` with a unique key when a `LIMIT` is present.

### 8.7 A `ROWS` offset larger than the partition silently returns the partition total

**Symptom.** A "last 30 orders" metric returns the same value as "all orders" for small accounts, and the smallest accounts dominate a per-account average.

**Competing causes.** (a) The frame offset exceeds the partition size, so the frame is clamped to the partition start and the metric degenerates to a cumulative total; (b) the account genuinely has fewer than 30 orders, so the metric is correct but undefined for comparison; (c) an upstream filter removed recent orders, shrinking every partition.

**Discriminating check.** Compute `COUNT(*) OVER (frame)` alongside the metric. A count equal to the partition size on rows where the account has fewer than 30 orders confirms (a). A count that matches a `GROUP BY` count of the source confirms (b). A count lower than the source count for every account points at (c).

**Repair.** Report the frame cardinality with the metric, or restrict the population to accounts with at least the required order count. A moving metric over a variable denominator must publish the denominator, otherwise averages across accounts mix window widths.

### 8.8 A window over already-aggregated rows double-counts after a join

**Symptom.** A daily running total computed from a joined fact table exceeds the sum of the daily figures by a factor that changes per day.

**Competing causes.** (a) The join to a dimension with multiple matching rows fanned out the fact rows before the window saw them, so the window summed duplicates; (b) the window was applied to a table that already had a running total column, so totals were summed rather than amounts; (c) the partition key came from the dimension and split one logical entity into several.

**Discriminating check.** Compare the window input's row count to the fact table's row count. Inflation by a factor equal to the average dimension fan-out confirms (a), and it is visible without touching the window at all — which is the point: window bugs are often join bugs wearing window clothes. A per-key count of 1 against an expected many confirms (b) if the column was pre-aggregated.

**Repair.** Aggregate to the reporting grain before applying the window, and assert uniqueness of the grain key in the model tests. This is the same defence as 5.7's pre-aggregation advice, applied for a different reason: there it removed peers, here it removes fan-out.


### 8.9 The frame that is right in testing and wrong in production

**Symptom.** Every unit test passes, the sample dashboard looks correct, and the production figure is off by a
percentage that grows during the month.

**Competing causes.** (a) Test fixtures have one row per ordering key, so `ROWS` and `RANGE` agree and the bug is
invisible; (b) production data includes a business unit whose rows arrive through a second pipeline with a different
timestamp convention; (c) the frame is correct but the metric's denominator was changed in a specification revision
that never reached the query.

**Discriminating check.** Count duplicate ordering keys in the fixture and in production. A fixture with zero
duplicates against a production table with a duplicate ratio of, say, 12 percent is a confirmed (a): the test set
cannot distinguish the frames it is meant to distinguish. For (b), compare the distribution of the ordering key's
time-of-day component between pipelines; a spike at exactly midnight indicates a date-only timestamp convention.

**Repair.** Build fixtures from the hazard, not from convenience: at least one tie, one null key, one singleton
partition, and one negative amount. Four rows are enough, and they convert the whole class of frame bugs from
invisible to caught.


### 8.10 Duplicate rows produced by the window query's own join

**Symptom.** The per-partition total from the window exceeds the same total computed by a plain `GROUP BY` on the
fact table, and the excess is proportional to the number of dimension attributes joined.

**Competing causes.** (a) A many-to-many join expanded rows before the window; (b) the partition key is drawn from
the dimension, so one fact row lands in several partitions; (c) the fact table genuinely has multiple rows per
business event and the "total" being compared is a deduplicated figure.

**Discriminating check.** Join the dimension alone and count rows per fact key before any window exists. A maximum
count above one is (a), and its value equals the fan-out. Compare distinct partition keys against distinct fact
entity ids for (b). A count of fact rows above the business event count with no join at all is (c), which is a data
model question and not a SQL one.

**Repair.** Aggregate to the reporting grain first, join only attributes that are unique on that grain, and add a
uniqueness assertion on the join key. The window then sees one row per report row and every frame in this sheet
behaves as derived.

### 8.11 A refund row reverses the running total and looks like corruption

**Symptom.** A cumulative revenue series drops by an amount that does not correspond to any sale, and a monotonicity
assertion in the data tests fails on a specific date.

**Competing causes.** (a) The frame includes a negative amount, which is a refund or a credit note, so the sum
legitimately falls; (b) the amount column stores refunds as negative values in the same table as sales and the
metric definition says "gross", not "net"; (c) the partition boundary changed and rows migrated between partitions.

**Discriminating check.** Inspect the sign of the amounts in the frame at the drop. A single negative row whose
magnitude equals the drop confirms (a). A cluster of negatives equal to the drop, with the same `order_id` as sales
in the same partition, confirms (b) and is a definition question. A drop with no negative amounts anywhere points at
(c), and the partition key values at the boundary will show the migration.

**Repair.** State gross versus net in the metric definition, keep refunds in a separate column when both are needed,
and replace the monotonicity assertion with one that permits a drop when a negative amount is present. A test that
forbids all decreases will fail on correct data eventually, and the cost of the false alarm is that the test gets
disabled.

## 9. Complexity & Cost Analysis

### 9.1 Sorting dominates

A window function requires its partition to be ordered, so the cost model starts with a sort of the filtered rows: `O(n log n)` comparisons on the window ordering key, where `n` is the number of rows reaching the window stage. The frame evaluation itself is usually linear per partition, because the engine maintains a running accumulator and adds or removes one row as the frame edge advances.

The practical consequence for a 10-million-row fact table is that the sort is the whole bill: a per-row frame computation is negligible next to it, so optimising the aggregate expression (`SUM` versus a hand-rolled `CASE`) buys nothing, while removing a redundant `ORDER BY` column that forces a second sort can halve the runtime.

### 9.2 Multiple windows with the same specification share a sort

Two window functions with identical `PARTITION BY` and `ORDER BY` are computed in one pass over one sorted stream. Two with different orderings require two sorts, and the cost is additive.

```text
cost(one window)      = sort(n, key1) + scan(n)
cost(two same-spec)   = sort(n, key1) + 2 * scan(n)
cost(two diff-spec)   = sort(n, key1) + sort(n, key2) + 2 * scan(n)
```

That is why a query mixing `ORDER BY order_date` and `ORDER BY amount` is materially slower than one that reuses a single ordering, and why rewriting "rank by revenue, then running total by date" as two CTEs is often *faster* than one wide `SELECT`, because each CTE sorts a smaller projection.

### 9.3 Frame width changes the constant, not the asymptote

A naive mental model says a wider frame costs more. In a running-accumulator implementation it does not: `ROWS BETWEEN 1000 PRECEDING AND CURRENT ROW` costs the same as `ROWS BETWEEN 1 PRECEDING AND CURRENT ROW`, because one row enters and one leaves per step. What does cost more is a frame that cannot be maintained incrementally — `EXCLUDE GROUP`, `DISTINCT` inside a window aggregate, and `RANGE` with a non-arithmetic ordering all force per-row recomputation in some engines.

```text
incremental frames: ROWS n PRECEDING/FOLLOWING, RANGE with numeric/interval,
                    UNBOUNDED bounds  -> O(n) per partition
recomputed frames : EXCLUDE GROUP/TIES, DISTINCT aggregates,
                    percentile_cont over a moving frame -> O(n * w) per
                    partition in engines without a specialised path
```

### 9.4 Partitioning and memory

Each partition must be materialised independently. A query partitioning by `customer_id` over 10 million rows with 5 million customers creates 5 million tiny partitions and pays per-partition overhead for each — frequently slower than no partition at all, because the unpartitioned sort is one merge rather than millions of micro-sorts. Partition by the column you would `GROUP BY`; a high-cardinality key is a warning sign that the metric is per-entity and the query should be scoped to those entities first.

### 9.5 Cost of the alternatives, for comparison

| Approach | Sort cost | Passes | Notes |
|---|---|---|---|
| Window function | one sort per distinct spec | 1 | frame maintained incrementally |
| Correlated subquery | index seeks per row | n | degrades to n scans without an index |
| Self-join running total | one sort plus one join | 2 | output size grows quadratically before aggregation |
| Pre-aggregate then window | sort on a much smaller table | 2 | usually the cheapest for daily reporting |

The fourth row is the standard answer for a dashboard: aggregate to one row per day and customer, then apply a `ROWS` window over that dense table. It removes the peer problem entirely, because after daily aggregation the ordering key is unique by construction, and it shrinks the sort input by the average number of orders per day.

### 9.6 Spill behaviour and the memory cliff

A window operator buffers each partition until its last row arrives. For a query partitioned by a low-cardinality key the buffer is the whole table, and engines spill to disk beyond a configured work-memory limit. The visible symptom is a query that runs in two seconds on a 1-million-row sample and in twenty minutes on the full table, with the plan unchanged.

Two design responses follow. First, prefer pre-aggregation so the sorted input is small — the window then operates on thousands of rows rather than millions. Second, if a full-table window is unavoidable, partition by a high-cardinality key so partitions fit in memory individually; paradoxically the per-partition overhead rises while the spill risk falls, and the crossover point is worth measuring rather than guessing.


### 9.7 Estimating the sort's cost before running the query

The dominant term is comparisons times rows. A sort of `n` rows needs roughly `n log2 n` comparisons, and each
comparison on a wide key costs more bytes. Two numbers make the estimate usable: for a million rows, `log2(10^6)` is
about 20, so the sort performs on the order of 20 million comparisons; for a ten-million-row table the exponent is
about 23, giving 230 million. The growth from one million to ten million rows is therefore about 11.5 times, not
ten, and the discrepancy is the logarithm.

Frame evaluation adds a linear pass per partition, so a million-row single-partition window adds a million
accumulator steps — negligible next to the sort but not next to nothing if the aggregate is expensive. An aggregate
whose per-row cost is high, such as a percentile over a moving frame, inverts the balance: the linear term acquires
a large constant and the sort stops being the bottleneck.

Partition count changes the constant rather than the exponent. A thousand partitions of a thousand rows each still
sorts a thousand rows a thousand times, which is `1000 * 1000 * log2(1000)` or about ten million comparisons, close
to the unpartitioned single sort of a million. Millions of tiny partitions are where the model breaks, because the
per-partition fixed cost dominates and the measured time can exceed the unpartitioned plan by an order of magnitude.

### 9.8 When to replace a window with a running table

A dashboard that recomputes a seven-day window over a year of orders on every page load is doing work that changes
only at the edges. The running-table alternative stores, per key and per day, a cumulative total, and obtains the
window by subtracting two stored cumulatives — the identity derived in 5.7. The query becomes a join to the
cumulative table, the sort disappears, and the cost per page drops to an index lookup.

The trade is a maintenance obligation: the cumulative table must be updated when a late-arriving row lands in the
past, and that update touches every cumulative row from the affected date forward. For a metric with a wide frame
and infrequent late data, that is a good trade. For a narrow frame, the subtraction identity still works but the
savings are small, and the extra table is not worth its schema and backfill burden.

The decision rule is arithmetic: if the average number of orders per day per key is large, the window query reads
many rows to produce one number and pre-aggregation wins. If it is near one, the window reads almost the same rows
as the cumulative table stores, and the two are equivalent in cost with the window far simpler to maintain.

## 10. Key Theorems / Results Reference

**R1 — Default frame.** An `OVER` clause with `ORDER BY` and no frame implies `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`; with no `ORDER BY` it implies the entire partition. Standard-specified, therefore portable.

**R2 — Peer invariance.** Under `RANGE` and `GROUPS` frames built only on the ordering key, all rows in a peer group receive identical aggregate values. Under `ROWS` they may differ. This single statement explains most cross-frame discrepancies.

**R3 — Rank identities.** For a peer group of size `s` starting at position `p`: `RANK = p = 1 + s_1 + ... + s_{g-1}`, `DENSE_RANK = g`, and `ROW_NUMBER` takes the values `p .. p+s-1` in some order. Consequence: `RANK` and `ROW_NUMBER` agree exactly when no ties exist before the row.

**R4 — `DENSE_RANK` counts distinct levels.** `DENSE_RANK(r) = |{distinct key values <= key(r)}|` within the partition. Therefore `MAX(DENSE_RANK)` equals the number of distinct ordering values, a cheap and exact distinct-count over an ordered key.

**R5 — Frame determinism condition.** A `ROWS`-framed result is reproducible across runs if and only if the window ordering is total on the partition (no peers). A `RANGE`/`GROUPS` result is reproducible whenever the ordering is deterministic on key values, because peers are handled as a unit.

**R6 — `LAG`/`LEAD` are frame-invariant.** Their output depends only on partition and ordering, never on the frame clause. Corollary: a bug in `LAG` output cannot be fixed by editing the frame; look at the ordering and the partition key.

**R7 — Cumulative equals partition total at the last row.** For any cumulative frame `UNBOUNDED PRECEDING TO CURRENT ROW`, the final row of the partition reports the partition total. Corollary: a cumulative column whose last value differs from a `GROUP BY` total indicates a partition or filter mismatch, not a frame error.

**R8 — `EXCLUDE` preserves cardinality.** Exclusions change values, never row counts; every input row still produces one output row. A row count that changes after adding `EXCLUDE` means the clause was placed in a subquery with a filter, not that `EXCLUDE` behaved unexpectedly.

**R9 — `NULL` partition grouping.** All rows with `NULL` in the partition key form one partition, because window partitioning uses not-distinct semantics. A large partition with an unexpected key is usually the null bucket.

**R10 — Aggregate window over an empty frame.** If a frame contains no rows, `SUM`, `AVG`, `MIN`, `MAX` return `NULL` and `COUNT` returns `0`. This asymmetry is why `COALESCE` is needed around `SUM` but not around `COUNT`.

### 10.1 Using the results as an audit checklist

The ten results compress into a five-line audit that can be run against any window query in review, and it takes less time than reading the query.

1. Is the frame written? If not, apply R1 and say out loud which rows that admits.
2. Is the ordering total? If not, apply R5 and decide whether reproducibility matters.
3. Which function family? Ranking and offset functions are frame-invariant (R6), so a value problem there is an ordering or partition problem.
4. Does the last row's cumulative value equal the grouped total (R7)? If not, the scope is wrong.
5. Does any aggregate return null where a zero is expected (R10)? If so, the frame may be empty rather than the data missing.

Each line corresponds to a failure family in Section 8, so the audit doubles as a triage index: a failed line 1 points at 8.1 and 8.5, a failed line 2 at 8.3, and a failed line 4 at 8.2.

### 10.2 Window aggregate versus `GROUP BY`: when they are equivalent

For a frame that covers the whole partition, a window aggregate and a `GROUP BY` aggregate produce the same per-group
value; the difference is row multiplication, not arithmetic. That equivalence is useful in review: a share-of-total
calculation can be validated by comparing the window's denominator against a grouped total, and any mismatch is a
scope or fan-out defect rather than a frame defect.

The equivalence breaks at the moment the frame is bounded. A `GROUP BY` has no notion of "the previous six rows", so
no grouping rewrite can produce a moving metric; conversely a window cannot reduce row count, so no window can
replace `GROUP BY` for a summary table. Choosing between them is decided by the output grain rather than by the
metric: one row per group means `GROUP BY`, one row per input row means a window.

### 10.3 Results that are commonly misquoted

Three claims circulate about window functions and each is false in a specific way that this sheet's derivations can
correct. The first is that `RANK` "skips numbers on duplicates", which is true but incomplete: it skips numbers
because it reports the count of rows strictly before the peer group plus one, so the size of the skip equals the
size of the group minus one. A group of ten at position one is followed by rank eleven, not rank two, and a
dashboard that expected a maximum rank equal to the group count will be surprised.

The second is that `RANGE` is "slower than `ROWS`". The frame mode itself does not determine cost; the ordering
column's arithmetic does. A `RANGE` on an indexed numeric column maintained incrementally is as cheap as a `ROWS`
frame. What costs more is a `RANGE` whose bound cannot be maintained by moving an edge, which happens when the
ordering is a string or when the offset requires recomputation per row.

The third is that a window function "runs after `ORDER BY`". It runs before. The statement-level `ORDER BY` is the
last clause evaluated, which is why a window result can be sorted but cannot be filtered in `WHERE`, and why a
`LIMIT` applied to a window result takes the sorted order of the outer query rather than the window order. Getting
this backwards produces queries that sort by the window key and then take an arbitrary subset, and the arbitrariness
is invisible because the output happens to look sorted.

### 10.4 The identity that ties the whole sheet together

Every result in Sections 4 and 10 can be read as a special case of one statement: a window aggregate is a function of
the *membership set*, and every clause in `OVER` exists to determine that set. `PARTITION BY` filters it to one key,
`ORDER BY` induces the order that makes "preceding" meaningful, the frame bounds cut a contiguous slice, and
`EXCLUDE` subtracts a named subset. `FILTER` is different in kind: it does not change the set, it changes which
members contribute values.

That one sentence resolves the confusion between `WHERE`, `FILTER`, and a frame bound, which look interchangeable
and are not. `WHERE` changes the population before any window exists, so it changes row counts. `FILTER` keeps the
row in the frame but substitutes a null for its value, so `COUNT(*)` still counts it. A frame bound removes the row
from the frame entirely, so both the aggregate and the numerator of an average change. Three different levers, three
different observable effects, and only one of them affects the output row count.

When a reviewer asks whether a query is right, that hierarchy is the answer's skeleton: population first, then
membership, then contribution.

## 11. Interview / Exam Questions

### Q1. What rows does `SUM(x) OVER (ORDER BY d)` aggregate for a row whose date is shared with two other rows?

**Answer.** It aggregates every row in the partition whose date is less than or equal to the current date, which is the current row's entire peer group plus everything before it. The value is therefore identical for all three rows sharing that date, and it already contains amounts from rows that may appear after the current one in an arbitrary scan order. That is not a bug; it is the defined behaviour of the implied `RANGE UNBOUNDED PRECEDING TO CURRENT ROW` frame, where `CURRENT ROW` in `RANGE` mode means the far edge of the peer group.

### Q2. Given amounts 30, 90, 10, 40 on dates 01-02, 01-05, 01-05, 01-12, compute the running total under the default frame and under `ROWS UNBOUNDED PRECEDING`. State the error.

**Answer.** Default frame: 30, 130, 130, 170. Rows frame: 30, 120, 130, 170. The two differ only at the first 5 January row, where the default reports 130 against 120, an absolute error of 10 and a relative error of 10/120 = 8.33 percent. The 10 equals the amount of the second 5 January row, which the default frame pulls in because it is a peer. The final totals agree at 170, so a reconciliation that only checks the last row misses the discrepancy entirely.

### Q3. Why does `RANGE BETWEEN 6 PRECEDING AND CURRENT ROW` fail when the window has two ordering columns?

**Answer.** `RANGE` offsets are defined as arithmetic on the ordering value, so the standard permits exactly one ordering expression when an offset is used; with two columns there is no single value to subtract from. The failure is a compile-time rejection, not a silent wrong answer, which makes it benign. The repair is not to delete the second column — that reintroduces peer merging — but to switch to `ROWS` if positional semantics are acceptable, or to pre-aggregate to one row per key and keep `RANGE` over that reduced table.

### Q4. A top-three report returns four rows. Diagnose.

**Answer.** The query almost certainly used `RANK` and there is a tie at position three: two products with identical ordering-key values both receive rank 3, and both satisfy `rank <= 3`. That is `RANK` behaving as specified. The defect is a requirement that says "three" without a tie policy. Two repairs are legitimate and must be chosen by the business owner, not the analyst: accept four rows and label the report "top 3 including ties", or impose a tie-break in the window ordering (revenue descending, then product id ascending) and switch to `ROW_NUMBER` so exactly three rows are returned deterministically.

### Q5. Compute `RANK`, `DENSE_RANK`, and `ROW_NUMBER` for five rows ordered by a key with values 10, 20, 20, 20, 50.

**Answer.** `ROW_NUMBER` gives 1, 2, 3, 4, 5 in some order within the tied block. `RANK` gives 1, 2, 2, 2, 5 — the tie occupies positions 2 through 4 and the next value starts at 5. `DENSE_RANK` gives 1, 2, 2, 2, 3. The identities are checkable: `MAX(DENSE_RANK) = 3` equals the number of distinct key values; `RANK` at the last row equals 5, the row count; `DENSE_RANK` plus the size of prior peer groups minus one reconstructs `RANK`.

### Q6. When is `LAG(x, 1, 0)` wrong even though it eliminates nulls?

**Answer.** It is wrong whenever "no previous row" and "previous row was zero" are different facts. A month-over-month delta of zero on a customer's first month asserts that the customer had the same revenue as a nonexistent previous month, which is false; a null asserts that no comparison exists. Charts and downstream `AVG` calls both treat those differently: the zero drags an average down, the null is skipped. The default value is a semantic choice about the metric's definition, so it belongs in the metric specification rather than in the query.

### Q7. Why does `LAST_VALUE(x) OVER (ORDER BY d)` not return the last value in the partition?

**Answer.** The implied frame ends at `CURRENT ROW`, so `LAST_VALUE` reports the last row of the current row's peer group — the frame's last, not the partition's last. On a five-row partition it returns a sequence that resembles a rearranged copy of the input, which is the reported symptom. To get the partition's last value, write `ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING`; to get an extremum rather than a positional value, use `MAX` or `MIN`, which are frame-robust and easier to audit.

### Q8. A seven-day average is wrong on days 1 to 6. Which frame was used, and is it wrong?

**Answer.** The query almost certainly used a row-based frame such as `ROWS 6 PRECEDING`, which produces a growing window (1 row, 2 rows, ... 7 rows) at the head of the series. That behaviour is correct for a row frame and the metric is genuinely a row average over the available rows. It is "wrong" only against a specification that asks for a calendar average over seven days, and the discriminating test is to print the frame cardinality: growth from 1 to 7 identifies the row frame, a constant 7 with mismatched membership identifies an offset or timezone error.

### Q9. Rewrite `COUNT(*) FILTER (WHERE amount >= 50) OVER w` for a dialect without `FILTER`, and explain the classic mistake.

**Answer.** The correct rewrite is `SUM(CASE WHEN amount >= 50 THEN 1 ELSE 0 END) OVER w`. The classic mistake is `COUNT(CASE WHEN amount >= 50 THEN 1 ELSE 0 END) OVER w`, which counts every row because the `ELSE 0` branch produces a non-null value that `COUNT` includes. Omitting the `ELSE` entirely, `COUNT(CASE WHEN amount >= 50 THEN 1 END)`, happens to be correct because the implicit null is skipped, which is why the mistake is inconsistent: the same author gets it right in one query and wrong in another.

### Q10. Two queries over identical data and identical `OVER` clauses return different top-10 lists on consecutive runs. What is the cause and what is the check?

**Answer.** The window ordering is not total: `ROW_NUMBER` assigns distinct numbers to tied rows, but which tied row gets which number is unspecified, and a plan change, a parallel scan, or a different physical row order can permute them. The check is to run both queries inside one repeatable-read transaction and diff the outputs: a difference on unchanged data excludes changed source data and isolates the nondeterminism. The repair is a unique final tie-break in the window ordering, and the same discipline applied to the statement-level `ORDER BY` whenever a `LIMIT` is present.

### Q11. Compute the result of `EXCLUDE GROUP` versus `EXCLUDE TIES` for a row whose peer has an amount of 25, in a partition totalling 435 that includes the current row's 50.

**Answer.** `EXCLUDE GROUP` removes the current row and all its peers, so the frame sums 435 - (50 + 25) = 360. `EXCLUDE TIES` removes the peers but keeps the current row, so the frame sums 435 - 25 = 410. The 50-unit gap between the two answers is exactly the current row's own amount, which is the general rule: the two clauses differ by the current row's contribution whenever the peer group is non-singleton, and are identical when it is a singleton.

### Q12. Why can a window-function result not be filtered in the same query's `WHERE` clause?

**Answer.** Window evaluation occurs after `WHERE`, `GROUP BY`, and `HAVING` in the logical order, so the column does not exist when `WHERE` is evaluated. Engines differ on whether the error is a parse error or a resolution error, but all reject it. The standard remedy is one level of nesting: compute the window in a subquery or CTE, then apply the predicate outside. This is also the correct pattern for top-N-per-group, where the inner query computes `ROW_NUMBER` and the outer query keeps `rn = 1`.

### Q13. What is the difference in cost between `ROWS 1000 PRECEDING` and `ROWS 1 PRECEDING`?

**Answer.** In a running-accumulator implementation, essentially none: the engine adds one row at the leading edge and drops one at the trailing edge per output row, so per-partition cost is linear in partition size regardless of frame width. The frame width changes the size of the multiset being reported, not the number of operations. The frames that do cost more are those that break incrementality — `EXCLUDE GROUP`, `DISTINCT` inside a window aggregate, and some percentile computations over moving frames — because the engine must recompute per row.

### Q14. A cumulative column's final value disagrees with a `GROUP BY` total for the same partition. What does that prove?

**Answer.** It proves the window query's scope differs from the grouped query's scope, not that the frame is wrong, because a cumulative frame ending at the current row necessarily equals the partition total at the last row. The divergence therefore comes from a filter applied to one query and not the other, a different join that changed cardinality, or a partition key that splits what the grouped query merges — including the case where `NULL` keys form their own partition. The check is to compare row counts and distinct key counts between the two inputs before comparing sums.

### Q15. Under what condition is a `RANGE`-framed result reproducible while a `ROWS`-framed result is not, on the same query?

**Answer.** When the ordering key is non-unique. `RANGE` and `GROUPS` treat each peer group as an indivisible unit, so the frame membership and therefore the aggregate value are determined by key values alone. `ROWS` counts physical positions through the tie, so two runs may assign different rows to the same frame edges and produce different per-row values, even though the partition total is unchanged. Adding a unique tie-break to the ordering makes `ROWS` deterministic as well, which is the standard repair.

## 12. Quick Reference Card

| Clause / function | Resolved frame or rule | Value it produces | Trap to state aloud |
|---|---|---|---|
| `OVER (PARTITION BY k)` | whole partition | partition total per row | no ordering means no cumulative behaviour |
| `OVER (ORDER BY d)` | `RANGE UNBOUNDED PRECEDING TO CURRENT ROW` | cumulative to end of peer group | peers inflate the total silently |
| `ROWS BETWEEN 6 PRECEDING AND CURRENT ROW` | 7 physical rows | row-count sliding sum | splits ties; head of series is short |
| `RANGE BETWEEN INTERVAL '6 days' PRECEDING AND CURRENT ROW` | 7 calendar days | date sliding sum | needs exactly one ordering column |
| `GROUPS BETWEEN 1 PRECEDING AND CURRENT ROW` | 2 peer groups | group sliding sum | unavailable on older SQL Server |
| `EXCLUDE GROUP` / `EXCLUDE TIES` | subtract peers / subtract peers but keep self | leave-one-day-out baseline | differ by the current row's amount |
| `RANK` / `DENSE_RANK` / `ROW_NUMBER` | first position / group ordinal / arbitrary total order | 1,2,2,4,5 / 1,2,2,3,4 / 1,2,3,4,5 | `RANK` admits both boundary ties in top-N |
| `LAG(e, n, d)` | frame-independent | value n rows earlier, else `d` | null at every partition start |
| `LAST_VALUE` with the default frame | ends at peer-group end | last peer's value | needs `UNBOUNDED FOLLOWING` for partition end |
| `FILTER (WHERE p)` | rows failing `p` removed pre-aggregate | conditional aggregate | `CASE ... ELSE 0` inside `COUNT` counts all rows |

Five habits, in the order a reviewer applies them: write the frame explicitly; make the ordering total; choose the rank function from the tie policy; check the increment, not the final total; state `NULLS FIRST` or `NULLS LAST` whenever the ordering column is nullable.
