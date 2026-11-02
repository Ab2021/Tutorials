> Module 03 — Data Engineering and Interoperability
> Topic 04 of 11
> Estimated study time: 3.5 hours
> Prerequisites: 03.01, 03.03

# SQL Patterns That Carry Supply Chain Semantics

---

## 1. Why this topic exists

A demand model at Calder Valley Supplies, a wholesaler of industrial components, backtested at
94% forecast accuracy at item–location–week grain. The number was reported to the commercial
director and used to justify a planning-system upgrade. Eighteen months later the warehouse was
migrated, the feature pipeline was rebuilt, and the same model, on the same history, scored 71%.

Nobody had changed the model. The first pipeline joined order history to the *current* item
master. The item master had been reorganised twice, and the reorganisation was a Type 1
overwrite: every historical order was retrospectively attributed to today's product family. The
model had been trained on a version of the past that contained the answer. The second pipeline
used as-of joins and produced the honest number. Both queries ran without error. Both returned
rows. Only one of them described a world that ever existed.

This is the subject of this topic. Structured query language (SQL) has no native concept of
"the item as it was in March". It has dates, joins, and window functions, and the supply chain
semantics — validity, knowledge, grain, additivity — have to be carried by the query author.
Four failures account for most of the damage:

1. **The wrong clock.** A query joins on the date the row was written rather than the date the
   fact was true, or the reverse: a restatement of history disguised as analysis, or a backtest
   that knows the future.
2. **The wrong grain.** A daily snapshot is summed across days, producing unit-days reported as
   units; a movement ledger is queried for "how much is there".
3. **The wrong version.** A fact joins to a dimension on the natural key with no date predicate,
   the join fans out across every version, and the revenue total doubles.
4. **The wrong environment.** The same SQL returns different rows under a different engine, time
   zone, `NULL` ordering, or integer-division default.

None of these produces an error message. All four produce numbers that look plausible, which is
why they survive review. A supply chain organisation will argue about a 2% forecast improvement
for a quarter and never notice that its inventory history is overstated by 200%.

**Why this matters to a data scientist specifically.** Every model in this program consumes a
feature table built by SQL. Leakage is not only a machine-learning problem; it is usually a join
problem. An as-of join written with `<=` instead of `<` on a half-open interval leaks one day. A
snapshot averaged instead of differenced hides a stockout. An allocation computed with floating
point loses units that reappear as an unexplained reconciliation gap three months later. The
patterns in this topic are the ones that decide whether a model's training set is a record of
the past or a reconstruction of it.

The previous topics built the layers (03.01) and moved data into them (03.03). This topic is
about what the queries inside those layers must say.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** valid time from transaction time in a named table, and **state** which one a
   given report is using.
2. **Write** an as-of join that resolves the dimension version in force on a business date, and
   **explain** why the temporal join replaces a correlated subquery without changing its result.
3. **Declare** the grain of a snapshot fact and a transaction fact, and **state** which
   aggregations are legal across each dimension.
4. **Reconcile** a balance derived from movements against a snapshot, and **diagnose** a
   double-counted opening anchor.
5. **Allocate** an integer constrained supply across competing orders so that the parts sum
   exactly to the total, using a deterministic remainder rule.
6. **Detect** a stockout spell with a gap-and-island pattern, and **explain** how a missing
   snapshot day changes the answer.
7. **Design** a slowly changing dimension (SCD) at attribute level, choosing type 1 or type 2
   per attribute, and **identify** the join that would fan out.
8. **Reproduce** a query's output across engines by pinning dialect, session settings, and
   tie-break columns, and **name** three defaults that silently change results.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold one picture: every row has two dates, and they are not the same date. The **valid time** is
when the fact was true in the world. The **transaction time** — also called record time or
system time — is when the warehouse learned it. A plan made on a Tuesday can only use what had
been recorded by Tuesday. A report run today can use everything, including corrections that
arrived last week and describe last year.

Most supply chain disagreements about "the right number" are disagreements about which clock
the question is on.

```text
   SKU-4417 case pack at Calder Valley Supplies

   VALID TIME  (when the packing was true in the world)
   2025-12-01            2026-01-19              2026-09-01
       |---------------------|-----------------------|---------->
       |     24 eaches       |      12 eaches        |  6 eaches
                             ^                       ^
                             |                       |
   TRANSACTION TIME  (when the warehouse recorded it)
       |                     |                       |
   recorded              recorded                recorded
   2025-11-14            2026-02-02              2026-08-20
   (initial load)        (14 days late)          (12 days early)

   ORDER 2026-03-02, quantity 240 eaches
   ------------------------------------------------
   join on item_id only (current row)   ->  6  ->  40 cases
   as-of valid time, 2026-03-02         -> 12  ->  20 cases
   as-of valid + tx time, 2026-01-25    -> 24  ->  10 cases
   as-of valid + tx time, 2026-02-16    -> 12  ->  20 cases
```

Read the four answers carefully, because each one is correct for a different question and three
of them are wrong for the others.

- The **current-row join** answers "what does this item look like today". It is the default
  behaviour of a query with no date predicate, and it is the source of the backtest failure in
  section 1.
- The **valid-time as-of** answers "what was true on the order date". This is the right join for
  operational reporting: it describes the world the order lived in.
- The **bitemporal as-of at 25 January** answers "what could a planner have known on 25 January".
  The January change had not yet been recorded, so the planner saw 24 eaches and ordered 10
  cases. That plan was wrong about the world and right about the information available. This is
  the only honest basis for a backtest.
- The **bitemporal as-of at 16 February** answers the same question a week after the late entry
  landed, and returns 20 cases.

Five properties follow.

**Property 1 — an as-of join is version selection, not date filtering.** Filtering
`where order_date between start and end` restricts *which facts* you read. An as-of join decides
*which version of the attribute* each fact is paired with. Confusing the two produces a query
that looks temporal and is not.

**Property 2 — grain is an arithmetic permission.** A grain states what one row means, and that
meaning licenses some aggregations and forbids others. Summing a balance across dates is not a
bigger number; it is a different unit.

**Property 3 — every dimension version needs a unique key, and every fact needs the right one.**
The surrogate key lets a fact point at the version in force when it happened. Join on the natural
key instead and the fact fans out across every version, multiplying the measure.

**Property 4 — allocation is integer arithmetic with an explicit remainder rule.** Percentage
splits are not allocation. The moment the unit is indivisible — a case, a pallet, a patient —
the parts must sum exactly to the whole, and the rule for the leftover units must be stated and
deterministic.

**Property 5 — "absent" is not one thing.** A stockout day, a day with no snapshot, and a day
before the item was listed are three different states. Collapsing them to `on_hand = 0` or to a
missing row changes the length of every spell you measure.

### 3.1 Balance and movement are different objects

The single most useful distinction in inventory SQL is between a state and a flow. The snapshot
is the integral of the ledger; the ledger is the derivative of the snapshot. Adding snapshots
together does not give a bigger snapshot, it gives unit-days, which is an exposure measure and
not a stock.

| Question | Correct object | Legal operation | Illegal operation |
|---|---|---|---|
| How much is on hand at month end? | Snapshot | Last value in the period | Sum of daily balances |
| How much did we receive in January? | Ledger | Sum of receipts | Count of snapshot rows |
| What was average inventory? | Snapshot | Mean of daily balances | Last minus first |
| What was the stockout exposure? | Snapshot or ledger | Count of zero days | Mean of zero flags |
| How long did the stockout last? | Snapshot | Gap-and-island over the date spine | Count of zero rows |
| What is the inventory turn? | Both | Issues from ledger / mean snapshot | Issues / closing snapshot |

The last row is a genuine choice with a defensible convention behind it: dividing annual issues
by the *average* balance measures the average holding period, while dividing by the *closing*
balance measures the current holding period. Both are called "turn". They differ whenever
inventory is trending, which is most of the time.

> **Note:** The word "snapshot" is overloaded in this field. A *snapshot fact table* is a table
> at periodic grain. A *database snapshot* is a read-consistent view used for isolation. A
> *snapshot date* is the business date the row describes. The three share a word and nothing
> else. Section 8.4 returns to the isolation meaning.

### 3.2 Why the same text returns different answers

Reproducibility is a property of a query *and* its environment. Six defaults change results
without changing the SQL:

| Default | What varies | Example consequence |
|---|---|---|
| Integer division | Truncation versus error versus float promotion | `240 / 24` is `10` in one engine and `10.0` in another |
| `NULL` ordering | Nulls first or last in `ORDER BY` | "Latest row" becomes the null row in one engine |
| Ranking ties | Arbitrary order among equal sort keys | Deduplication keeps a different duplicate |
| Empty-set aggregates | `SUM` returns `NULL` or `0` | A missing day becomes a zero or a gap |
| Week start and ISO week | Monday versus Sunday; week 53 handling | A weekly rollup shifts by a day or a week |
| Session time zone | `TIMESTAMP` versus `TIMESTAMPTZ` rendering | A 23:30 event lands on the next day |

None of these is a bug. Each is a documented behaviour of a particular dialect, and each is
invisible in a query that does not state its assumptions. Section 5 step 12 makes pinning them
an explicit part of the work.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Valid time | The interval during which a fact was true in the world | Treated as the date the row was created | Half-open date interval |
| Transaction time | When the warehouse recorded the fact | Assumed equal to valid time | Timestamp, UTC |
| Record time | Synonym for transaction time in event standards | Used loosely as "load time" | Timestamp, UTC |
| Bitemporal | Carrying both valid and transaction time on one row | Called "auditing" | Two interval pairs |
| As-of join | Selecting the dimension version in force at a stated date | Implemented as equality on the natural key | Join predicate on date |
| Temporal join | Joining two tables on overlapping validity periods | Assumed to be a date filter | Interval overlap predicate |
| Point-in-time correctness | A query returns what was knowable at a stated instant | Claimed without naming the instant | Boolean plus a date |
| SCD type 1 | Overwrite an attribute in place, keeping no history | Used for attributes that price history | None |
| SCD type 2 | Insert a new row with a new surrogate key per version | Applied to every attribute by default | Row version |
| SCD type 3 | Keep one previous-value column beside the current one | Used for more than one prior value | Column pair |
| Natural key | The business identifier stable across versions | Used as the dimension primary key | Source code |
| Durable key | Kimball's term for the natural key that ties versions together | Confused with the surrogate key | Source code |
| Surrogate key | Warehouse-assigned key unique to one version | Assumed stable across reloads | Integer |
| Current-row flag | Boolean marking the open version | Used as the only join predicate | Boolean |
| Effective date | First date a version applies | Confused with the entry date | Date |
| Expiry date | First date a version no longer applies, exclusive | Treated as inclusive | Date |
| Half-open interval | `[from, to)`: includes the start, excludes the end | Mixed with closed intervals in one model | Interval |
| Snapshot fact | One row per entity per period, recording state | Summed across periods | Units at a date |
| Transaction fact | One row per event, recording flow | Used to answer "how much is there" | Units per event |
| Accumulating snapshot | One row per process instance, updated at milestones | Confused with a periodic snapshot | Duration in days |
| Semi-additive measure | Additive across all dimensions except time | Summed over dates | Units |
| Non-additive measure | Cannot be summed across any dimension | Averaged without weights | Ratio or currency per unit |
| Grain | The exact meaning of one row | Stated as the table name | Declaration |
| Date spine | A complete calendar used to make missing days visible | Replaced by an ad hoc series per query | One row per date |
| Movement ledger | Append-only record of receipts, issues, adjustments | Reconciled only at year end | Signed units |
| Opening balance anchor | Balance at the close of the day before the window | Added again as a movement | Units |
| Allocation | Assigning constrained supply to competing demand | Done in a spreadsheet after the fact | Units per order |
| Apportionment | Dividing an integer total so the parts sum exactly | Rounded independently | Integer units |
| Largest remainder | Floor each share, award leftovers by fractional part | "Put the remainder in the last row" | Integer units |
| Deduplication | Reducing duplicate source rows to one per business key | Ordered by an arbitrary column | Row count |
| Tiebreaker | A unique column appended to `ORDER BY` | Omitted when the sort key looks unique | Column |
| Gap-and-island | Grouping consecutive rows sharing a state into spells | Applied without defining "missing" | Spell identifier |
| Stockout spell | A contiguous run of days with zero available stock | Counted as the number of zero rows | Days |
| Isolation level | The visibility rules a transaction sees | Assumed identical across engines | ANSI level name |
| Snapshot isolation | Readers see a consistent point-in-time view | Confused with a snapshot fact table | Engine setting |
| Fan-out | Row multiplication from a join matching several rows | Discovered after totals double | Row-count ratio |

Two of these deserve expansion, because they carry the most weight in the rest of the topic.

**Valid time versus transaction time.** These are independent axes, and a row can be right on
one and wrong on the other. A late-recorded price change is *valid* from 1 January and *known*
from 15 February. Before 15 February the warehouse's best statement about January was the old
price; after it, the statement changed. A query that ignores transaction time silently replaces
the first statement with the second, which is exactly what a restatement does — and exactly what
a backtest must not do.

**Semi-additive.** On-hand units add across items and across locations: 100 units in Leeds plus
50 in Wakefield is 150 units. They do not add across dates: Monday's 100 plus Tuesday's 100 is
not 200 units of anything. A quantity in unit-days is a legitimate exposure measure and an
illegitimate stock measure, and the two are routinely confused in inventory dashboards.

## 5. The mechanics, step by step

The sequence below is the order in which the decisions constrain each other. You cannot choose a
join before you have chosen a key; you cannot choose a key before you have decided which
attributes need history; you cannot decide that before you have named the decision and the date
it is made on.

### Phase A — Fix the two clocks

**Step 1. Name the decision date and the knowledge date.**

For every report, dashboard, or feature table, write two dates at the top of the model: `D`, the
business date the decision is made for, and `K`, the instant by which the decision-maker could
have known what the model shows. If `K` is "now", the artifact is not reproducible. The header
is `-- D = 2026-03-02, K = 2026-03-02 06:00 UTC`. The test is to run the model twice, a month
apart, against the same history: if the output changes and no correction was declared, `K` was
implicitly "now". The common failure is using `D` for both, which assumes the planner knew on the
planning date everything the warehouse knows today.

**Step 2. Classify every input table by temporal kind.**

Four kinds cover almost everything in a supply chain warehouse. Write the kind and the governing
column for each input before writing any join.

| Kind | Example | Governing column | What a row means | Typical defect |
|---|---|---|---|---|
| Event or movement | Goods receipt, shipment scan | `event_ts` | Something happened at an instant | Duplicate delivery of the same event |
| State or balance | Daily stock snapshot | `snapshot_date` | The state at a cut-off | Missing day treated as zero |
| Versioned attribute | Item master, supplier terms | `valid_from`, `valid_to` | A value in force over an interval | Overlapping intervals |
| Derived or allocated | Order allocation, plan | `plan_version`, `run_ts` | A rule applied to inputs | Inputs not stored with the output |

The classification matters because it decides what a duplicate row means. In an event table, a
duplicate is a delivery defect to be deduplicated. In a snapshot table, a duplicate is a second
row for the same day and must never be summed. In a versioned table, two rows for one date are an
overlap defect, and the join will fan out.

### Phase B — Build the dimension with the right history

**Step 3. Choose the SCD type per attribute, not per table.**

A dimension table is not "a Type 2 dimension". Individual attributes are Type 1 or Type 2, and
the choice is a statement about whether history needs to be priced correctly.

| Attribute | Type | Why | Test that proves it |
|---|---|---|---|
| Item description | 1 | A correction, not a change in the world | No closed-period report depends on the old spelling |
| Case pack | 2 | Changes what every historical quantity in cases meant | A past order recomputes to the packing then in force |
| Product hierarchy | 2 | Reorganisations must not rewrite past category performance | A closed month's category total is stable across reloads |
| Standard cost | 2 | Valuation and margin history | A closed period's margin does not move |
| Supplier lead time | 2 | Planning inputs must be auditable | A past plan reproduces with the lead time then in force |
| Unit of measure | 2, never 1 | A unit change makes old and new quantities incomparable | A quantity never changes meaning silently |
| Country of origin | 2 | Duty and trade compliance | A past declaration reproduces |
| Preferred-supplier flag | 1 or 2 | Depends on whether sourcing decisions are audited | State the choice and the audit requirement |

The rule of thumb: if a historical number would change meaning when the attribute changes, the
attribute is Type 2. If only the label changes, Type 1 is defensible and cheaper.

**Step 4. Assign surrogate keys and half-open intervals.**

- One surrogate key per version, assigned by the warehouse, never reused.
- `valid_from` inclusive, `valid_to` exclusive, with a real sentinel such as `9999-12-31` for the
  open version — never `NULL`, because `NULL` in a comparison is unknown and drops rows.
- Exactly one current row per natural key, enforced by a partial unique index or a test.
- No gaps: for each natural key, the next version's `valid_from` equals the previous `valid_to`.
  A one-day gap means an order placed in that gap joins to nothing.
- No overlaps: no two versions of a key cover the same date.

Half-open intervals tile without either defect, which is why `d >= valid_from and d < valid_to`
is the portable predicate.

**Step 5. Handle late-arriving and restated versions.**

A version recorded after its effective date is the normal case in supply chain data, not an
exception: a price change agreed in January is keyed in February. You have three options.

| Option | What it preserves | What it costs | When to choose it |
|---|---|---|---|
| Overwrite the open row | Nothing about the old belief | Cheapest; history is "as currently believed" | Attributes with no audit requirement |
| Add a transaction-time range | Both beliefs, queryable | Doubles the date columns and the join predicates | Anything used in backtests or regulatory reporting |
| Append to a restatement log | The fact of restatement, outside the main table | A second object to maintain | When the warehouse cannot be bitemporal |

If you choose the first, say so in the table comment. "History is as currently believed" is a
defensible contract. An unstated one is not.

### Phase C — Resolve the version

**Step 6. Decide where version resolution happens, and write it down.**

There are two architectures, and they fail differently.

| Approach | Where the work happens | Speed | Failure mode | Detection |
|---|---|---|---|---|
| Surrogate key on the fact | ETL, at load time | Fast queries | Wrong key assigned at load; invisible forever | Recompute the key from the fact date and compare |
| As-of join at query time | Every query | Slower; needs an index on `(natural_key, valid_from)` | Missing date predicate; fan-out | Assert one row per fact row after the join |
| Pre-joined wide table | Build time | Fastest | Both of the above, frozen | Rebuild and diff |

Kimball's guidance is the first: resolve the correct version in the ETL pipeline and store the
surrogate key on the fact. That is efficient and it makes the query simple. It also means that a
bug in the key lookup is baked into the fact table and no downstream query can detect it. The
query-time as-of join is slower and self-documenting: the version rule is visible in the SQL.

Whichever you choose, the other one is your test. If facts carry a surrogate key, the as-of join
recomputes it and any mismatch is a defect. If queries resolve as-of, the row-count assertion is
the test.

**Step 7. Guard the join against fan-out.**

The assertion is one line of reasoning and one number: the row count after the join must equal
the row count before it, for a dimension join that is expected to be many-to-one. If it is
greater, either the interval set overlaps or the join lacks a date predicate.

```text
rows_before = 1,000,000 order lines
rows_after  = 1,000,000  -> one version per line: correct
rows_after  = 3,214,000  -> 3.2 versions per line: overlap or missing predicate
rows_after  =   998,412  -> 1,588 lines matched nothing: gap in the interval set
```

Both directions matter. A fan-out inflates totals; a miss silently drops rows and deflates them.
A join that returns fewer rows than it started with is not "cleaner data"; it is lost revenue.

### Phase D — Fix the fact grain

**Step 8. Choose snapshot or transaction grain, and never mix them in one table.**

| Measure | Additive across items | Across locations | Across dates | Legal aggregate |
|---|---|---|---|---|
| On-hand units | Yes | Yes | No | `SUM` over items and locations at one date; `LAST` or `AVG` over dates |
| Receipt units | Yes | Yes | Yes | `SUM` anywhere |
| Unit cost | No | No | No | Weighted average by quantity |
| Fill rate | No | No | No | Recompute from numerator and denominator |
| Days of supply | No | No | No | Recompute from stock and run rate |
| In-transit units | Yes | Yes | No | `LAST` at a date; the same shipment must not be counted twice |

A table that mixes a balance and a flow in one row cannot be aggregated correctly in any
direction. If both are needed at the same grain, they belong in two tables joined on the grain
keys, which is what Kimball calls complementary fact table types.

**Step 9. Build a date spine and anchor the balance.**

A dense calendar is the only way to distinguish "zero" from "no row". Generate it once, as a
governed table, and left join everything to it.

The anchor is the subtle part. To compute a running balance over a window you need the balance at
the *close of the day before* the window starts. If you take the first snapshot inside the window
as the anchor and also add the first day's movements, you count that day twice.

```text
correct:    opening(close of D-1) + SUM(movements from D to D+n)
wrong:      first snapshot in window + SUM(movements from D to D+n)
wrong:      opening included again as a receipt on D
```

### Phase E — Derive the supply chain semantics

**Step 10. Allocate with integer arithmetic and a stated remainder rule.**

The procedure that always sums correctly is the largest remainder method, also called Hamilton's
method:

1. Compute the exact share `qty_i * supply / total_demand` as a rational number.
2. Take the integer floor of each share.
3. Compute the residual `supply - SUM(floors)`.
4. Rank rows by fractional part descending, with a deterministic tiebreaker such as order ID.
5. Add one unit to each of the first `residual` rows.

The tiebreaker is not cosmetic. Without it, two runs of the same allocation can award the last
unit to different customers, and the allocation is not reproducible or auditable.

**Step 11. Detect spells on a dense calendar, with three states.**

Gap-and-island assigns a constant to each run of consecutive rows in the same state, usually by
subtracting a row number from a date, and it is correct only if the dates are dense. The workable
pattern is: left join the state table to the date spine, classify each date as `in_stock`,
`zero`, or `unknown`, and only then form islands over `zero` dates adjacent on the calendar.
Whether an `unknown` day extends a spell is a business decision that must be stated, because it
changes reported stockout days.

**Step 12. Pin the environment.**

Record, alongside the query, what changes its result without changing its text: engine and
version; session time zone and the treatment of `TIMESTAMP` versus `TIMESTAMP WITH TIME ZONE`;
`NULL` ordering in every `ORDER BY` feeding a `row_number`; integer division behaviour; week-start
convention and ISO week handling; the tiebreak column in every ranking. A query without this
header is a result, not a reproducible artifact.

## 6. Worked example

Calder Valley Supplies distributes industrial components from six distribution centres (DCs) to
about 4,000 trade customers. The example follows one item, `SKU-4417` (a hydraulic seal kit), at
one location, `DC-LEEDS`, through the planning cycle for the week beginning 2 March 2026. All
figures are illustrative and internally consistent.

### 6.1 The decision and the four tables

The decision is the replenishment order for `SKU-4417` at `DC-LEEDS`, placed on 2 March 2026 for
delivery four weeks later. The order quantity is computed in cases, so the case pack is a
load-bearing attribute. Four tables feed the decision:

| Table | Temporal kind | Grain | Key columns |
|---|---|---|---|
| `staging.order_header` | Event | One row per order line | `order_id`, `item_id`, `order_date`, `qty_eaches` |
| `conformed.dim_item` | Versioned attribute (SCD2) | One row per item per version | `item_sk`, `item_id`, `valid_from`, `valid_to`, `case_pack` |
| `mart.inventory_snapshot` | State | One row per item per location per day | `snapshot_date`, `item_id`, `location_id`, `on_hand` |
| `mart.stock_movement` | Event | One row per movement | `movement_id`, `movement_ts`, `item_id`, `location_id`, `qty_delta` |

The item dimension has three versions for this item:

| Version | `item_sk` | `valid_from` | `valid_to` | `case_pack` | Recorded on |
|---|---:|---|---|---:|---|
| v1 | 88410 | 2025-12-01 | 2026-01-19 | 24 | 2025-11-14 |
| v2 | 88411 | 2026-01-19 | 2026-09-01 | 12 | 2026-02-02 |
| v3 | 88412 | 2026-09-01 | 9999-12-31 | 6 | 2026-08-20 |

Two things are already wrong with this table if you read it naively. Version 2 was recorded on
2 February but is valid from 19 January: it is a late entry, and for two weeks the warehouse's
best statement about that period was the old one. Version 3 is future-dated: it is recorded before
it is valid, which is legitimate for a planned change, and it is the row that makes the
current-row join fail.

### 6.2 The same 240 eaches, four answers

The order is 240 eaches. Converted to cases, it is 40, 20, or 10 depending on which version the
join picks.

| Join | Version chosen | `case_pack` | Full cases | Question it answers |
|---|---|---:|---:|---|
| Equality on `item_id`, no date | v3, the current row | 6 | 40 | What does the item look like today? |
| As-of valid time at 2026-03-02 | v2 | 12 | 20 | What was true on the order date? |
| As-of valid and transaction time at 2026-01-25 | v1 | 24 | 10 | What could a planner have known on 25 January? |
| As-of valid and transaction time at 2026-02-16 | v2 | 12 | 20 | What could a planner have known on 16 February? |

The first row overstates the cases by 100% relative to the second. A plan built on 40 cases orders
twice the physical volume the planner intended: not a rounding difference, a different order.

The third and fourth rows differ only in the knowledge date, and they bracket the late entry.
This is the pair a backtest must use. If you backtest the January plan with today's table, you
give the model the 12-case pack — a fact the January planner did not have — and you cannot tell
afterwards whether the resulting accuracy came from the model or the leak.

### 6.4 Balance versus movement: the January reconciliation

January 2026 at `DC-LEEDS`, `SKU-4417`:

```text
opening balance, close of 2025-12-31          1,200 units
receipts in January                           3,400 units   (900 + 1,200 + 800 + 500)
issues in January                             3,950 units
cycle-count adjustments                         -40 units
closing balance, 2026-01-31                   1,200 + 3,400 - 3,950 - 40 = 610 units
```

Now the snapshot table. It has 31 daily rows for January, and the balances sum to 24,180
unit-days.

```text
mean on-hand = 24,180 / 31 = 780.0 units
mean daily issues = 3,950 / 31 = 127.42 units per day
days of supply = 780.0 / 127.42 = 6.12 days
```

Both numbers are correct measurements of different things. The 610 is a stock. The 780 is an
average stock over the month. The 24,180 is neither; it is an exposure in unit-days, and the only
legal use of it is to divide by the number of days.

Now introduce the defect that appears in almost every inventory pipeline built from a feed that
posts an opening balance. Suppose the source system emits the 1 January opening as a *receipt*
as well as an opening balance, because the extract is a movement feed that includes the opening
line.

```text
receipts become 3,400 + 1,200 = 4,600
closing becomes 1,200 + 4,600 - 3,950 - 40 = 1,810 units
error = 1,810 - 610 = 1,200 units, or 197% of the true closing balance
```

The number is not implausible on inspection. It is larger than the true value but smaller than
the opening, it moves in the right direction, and it reconciles against a movement feed that
also contains the opening line. This is why the reconciliation must be run against the snapshot,
not against a second copy of the ledger: the ledger and the ledger agree by construction.

### 6.5 Allocation: 4,800 units against 7,300 units of demand

A supplier allocation for the week gives Calder Valley 4,800 units of `SKU-4417`. Five customer
orders are outstanding.

| Order | Demand (units) | Exact share | Floor | Fractional part | Award | Final |
|---|---:|---:|---:|---:|---:|---:|
| C-101 | 2,400 | 1,578.08 | 1,578 | 0.08 | 0 | 1,578 |
| C-102 | 1,800 | 1,183.56 | 1,183 | 0.56 | 1 | 1,184 |
| C-103 | 1,500 | 986.30 | 986 | 0.30 | 0 | 986 |
| C-104 | 1,100 | 723.29 | 723 | 0.29 | 0 | 723 |
| C-105 | 500 | 328.77 | 328 | 0.77 | 1 | 329 |
| **Total** | **7,300** | **4,800.00** | **4,798** | **2.00** | **2** | **4,800** |

The exact share is `demand * 4,800 / 7,300 = demand * 0.657534`. The floors sum to 4,798, two
units short. The two largest fractional parts belong to C-105 (0.77) and C-102 (0.56), so those
orders receive one extra unit each. The parts sum to 4,800 exactly.

Note what happens with the naive alternative. Rounding each share independently gives
1,578 + 1,184 + 986 + 723 + 329 = 4,800 — correct in this instance, by luck. On a three-way
split of 100 units it gives 33 + 33 + 33 = 99, one unit short, and the shortfall is discovered by
a customer rather than by a test. Independent rounding is not an apportionment rule; it is a
rule that usually works.

### 6.6 The stockout spell and the missing day

In February the item runs out. Daily snapshots for 9–17 February:

| Date | On hand | State |
|---|---:|---|
| 2026-02-09 | 120 | In stock |
| 2026-02-10 | 40 | In stock |
| 2026-02-11 | 0 | Zero |
| 2026-02-12 | 0 | Zero |
| 2026-02-13 | — | No row |
| 2026-02-14 | 0 | Zero |
| 2026-02-15 | 0 | Zero |
| 2026-02-16 | 0 | Zero |
| 2026-02-17 | 210 | In stock |

Three candidate answers, all defensible and all different:

1. **Count of zero rows: 5.** Says nothing about duration, and treats 13 February as in stock
   because it has no row.
2. **Span from first to last zero: 11 February to 16 February, 6 days.** Assumes 13 February was
   a zero day.
3. **Gap-and-island on the filtered rows: two spells, 2 days and 3 days.** This is the one that
   should worry you, because it is what the standard pattern actually returns.

Trace the island key. Filtering to zero rows and numbering them gives 11 February as row 1, 12
February as row 2, 14 February as row 3, 15 February as row 4, 16 February as row 5. Subtracting
the row number from the date gives island 10, 10, 11, 11, 11. The missing day splits one
continuous stockout into two spells of two and three days.

The honest report is "at least 5 confirmed zero days, one unknown day, spell length 5 to 6 days",
with the unknown day flagged. Whether the business counts that as one spell or two is a decision
about what a stockout means when the data is incomplete, and it belongs in the metric definition,
not in the query.

## 7. Code, query, or calculation

One query, in two forms. The first is the correlated subquery that most analysts write; the
second is the window form that replaces it. They return the same rows when the dimension is
gapless and non-overlapping, and the second makes the version choice explicit rather than
implicit.

```sql
-- Before: a correlated lookup, evaluated once per order line.
select o.order_id, o.order_date,
       (select d.case_pack
          from conformed.dim_item d
         where d.item_id = o.item_id
           and o.order_date >= d.valid_from
           and o.order_date <  d.valid_to
         order by d.valid_from desc
         limit 1) as case_pack
from staging.order_header o;

-- After: rank every candidate version once, keep the first.
select order_id, order_date, case_pack
from (
    select o.order_id, o.order_date, d.case_pack,
           row_number() over (partition by o.order_id, o.order_date
                              order by d.valid_from desc, d.item_sk desc) as rn
    from staging.order_header o
    join conformed.dim_item d
      on d.item_id = o.item_id
     and o.order_date >= d.valid_from
     and o.order_date <  d.valid_to
) ranked
where rn = 1;
```

Four things to check before trusting the second form.

1. **The `valid_to` predicate is not redundant.** With gapless intervals, the greatest
   `valid_from` at or before the date is sufficient. With a gap it is not: the query would attach
   the previous version across a period when no version was in force. The predicate turns a gap
   into a dropped row, which the row-count assertion catches.
2. **The tiebreak is a real column.** `d.item_sk desc` breaks ties when two versions share a
   `valid_from`, which should not happen and does. Without it, the survivor is engine-dependent.
3. **The partition includes every key that identifies the fact.** Partitioning on `order_id`
   alone is wrong when an order has several lines for the same item at different dates.
4. **The rank is computed before the filter.** `where rn = 1` inside the subquery is illegal in
   most engines, and `QUALIFY` exists only in some dialects. The outer filter is portable.

Dialect notes. DuckDB has a dedicated `ASOF LEFT JOIN` for the same intent, usually faster.
Snowflake and BigQuery accept `QUALIFY`. SQL Server and Oracle require `ORDER BY` inside the
window and reject constant expressions. None of this changes the semantics; all of it changes
whether the query runs.

## 8. Data, standards, and regulatory context

### 8.1 What the SQL standard actually specifies

SQL:2011 (ISO/IEC 9075:2011) introduced the temporal features that most of this topic
approximates by hand. Kulkarni and Michels survey them in *SIGMOD Record*; the summary below
follows that survey and the standard's structure rather than any vendor's implementation.

| Feature | What it gives you | What it does not give you |
|---|---|---|
| `PERIOD FOR` over two columns | Declares a validity interval as one object | Enforcement unless the engine implements it |
| Application-time period | Valid time, the supply chain's "when was it true" | Any notion of when it was recorded |
| `SYSTEM VERSIONING` with `PERIOD FOR SYSTEM_TIME` | Transaction time maintained by the engine | Valid time |
| `FOR SYSTEM_TIME AS OF t` | The table as the system recorded it at `t` | The table as the world was at `t` |
| `FOR SYSTEM_TIME FROM a TO b` | Closed-open history window | Valid-time history |
| Temporal primary and foreign keys `WITHOUT OVERLAPS` | Overlap prevention at the constraint level | Gaplessness |
| `FOR PORTION OF` update and delete | Automatic row splitting on a change | Support in most engines |
| Period predicates `CONTAINS`, `OVERLAPS`, `PRECEDES` | Readable interval logic | Portability |

The critical gap for this topic: **SQL:2011's `FOR SYSTEM_TIME AS OF` is transaction time, not
valid time.** It answers "what did the database believe at time `t`", which is the backtest
question. It does not answer "what was the case pack on the order date", which is the
operational question, unless the case pack change happened to be recorded on the day it took
effect. Application-time periods cover the second question, and far fewer engines implement
them.

### 8.2 Engine support, as consulted

Engine support changes with every release. The table below reflects the sources consulted for
this topic — a 2019 practitioner survey and vendor documentation pages — and should be re-checked
against current documentation before it is depended on. Grades follow Module 28.

| Engine | System-time `AS OF` | Application-time periods | Notes | Grade |
|---|---|---|---|---|
| SQL Server 2016+ | Yes | No | Separate history table; `SYSTEM_VERSIONING = ON` | B |
| MariaDB 10.3.4+ | Yes | Since 10.4.3 | Closest open-source conformance; no temporal keys | B |
| IBM Db2 10.1+ | Yes | Yes | Most complete; some syntax deviations | B |
| Oracle | Flashback `AS OF TIMESTAMP` / `AS OF SCN` | Partial | Not the SQL:2011 syntax; no `FOR PORTION OF` | B |
| PostgreSQL | No in core | No | `temporal_tables` extension is trigger-based, not conforming | B |
| MySQL | No | No | Triggers and audit tables only | B |
| Snowflake | `AT` / `BEFORE` time travel | No | Time travel, not SQL:2011 temporal tables | D |
| BigQuery | `FOR SYSTEM_TIME AS OF` | No | System time only | D |
| DuckDB | No | No | `ASOF JOIN` covers valid-time joins, not history | D |

Two readings of this table are both correct. The standard exists and is coherent; the installed
base does not implement it uniformly. A team that writes `FOR SYSTEM_TIME AS OF` and assumes
portability will discover on the second engine that the query does not parse. A team that writes
the window-function as-of join will discover that it runs everywhere and needs its own tests.
For supply chain work, the second is usually the right trade, because the dominant need is valid
time, which the standard's most widely implemented feature does not provide.

> **Note:** Time travel is not bitemporality. A warehouse that can restore a table to its state
> at 03:00 last Tuesday answers an operational recovery question. It does not tell you what a
> planner knew on 25 January unless you also recorded which version of the attribute was current
> then. Vendors use "time travel" for both, and the conflation is expensive.

### 8.3 Time in event standards

Event standards carry two timestamps for the same reason this topic carries two clocks. GS1's
EPCIS distinguishes the time an event occurred from the time it was recorded (Topic 09 treats the
event model). The SQL consequence is narrow and important: an EPCIS-derived table has both
columns, and using record time where you mean event time attributes a February receipt to a March
load. ISO 8601 supplies the interval notation that makes half-open intervals readable as text
(`2026-01-19/2026-09-01`), but it does not say whether the end is inclusive; state that in the
column comment and enforce it with a test.

### 8.4 Isolation, snapshots, and the read-consistent view

The isolation level of a transaction decides which committed rows it can see. The ANSI levels —
read uncommitted, read committed, repeatable read, serializable — are the vocabulary;
implementations differ substantially, and several engines default to a snapshot-isolation variant
that is not one of the four.

The supply chain consequence is concrete. A replenishment query that reads orders and stock in
two statements can see an order allocated but the stock not yet decremented, or the reverse; the
total looks wrong by one order. Snapshot isolation makes the total coherent, but it may be
coherent at a point before the allocation committed. So: read from a single snapshot or a mart
built at one cut-off, and record the cut-off, because "consistent" is not "current". Topic 02
spends most of its design effort on that distinction.

### 8.5 Retention, erasure, and the immutability conflict

SCD type 2 history and erasure obligations pull in opposite directions: a row that records "this
customer was in segment B until March" is both the personal data an erasure request targets and
the audit trail behind a restatement. Warehouses resolve this with pseudonymisation at the
conformed layer, per-column retention schedules, and a documented exception process — never by
silently overwriting history. This is regulatory context, not legal advice; the engineering point
is that retention belongs in the same design review as grain.

### 8.6 What is convention rather than requirement

Half-open intervals, the `9999-12-31` sentinel, "one current row per key", and the SCD type
vocabulary are conventions. None is enforced by most engines, and Kimball's type numbers are a
shared language rather than a standard — type 6 and type 7 exist in practice and are defined
differently by different authors. State the convention in the column comment and test it.

## 9. Industry practice and case evidence

### 9.1 The academic lineage

Temporal database theory is older than the warehouse. Snodgrass's *Managing Temporal Data: A
Five-Part Series* (TimeCenter TR-28) sets out the valid-time and transaction-time state tables
that this topic's two clocks describe, and gives the sequenced join — intersecting periods of
validity using closed-open intervals — as the formal operation behind the as-of join. His
*Developing Time-Oriented Database Applications in SQL* adds the bitemporal case and the
audit-log triggers that approximate system time. Grade A.

The SQL:2011 temporal features standardise part of that programme, and Kulkarni and Michels's
survey in *SIGMOD Record* is the clearest public account of what was included and what was left
out. The commercial gap — valid time standard but thinly implemented, transaction time
implemented but answering a different question — is visible there and in the matrix in section
8.2. Grade A.

### 9.2 Kimball: the practitioner canon

The dimensional-modelling vocabulary used here — SCD types, periodic snapshot facts,
semi-additive measures — comes from Kimball and Ross and the Kimball Group's technique notes. The
relevant positions, as stated in those notes:

- A Type 2 change inserts a new dimension row with a newly assigned surrogate key, which becomes
  the fact's foreign key from that moment forward; prior facts keep the old key, which correctly
  points at history.
- Constraining on a non-changing attribute picks up all versions automatically; grouping on the
  changed attribute splits history into regimes, which is the intended behaviour.
- Joining facts to dimensions on the natural key plus a date range is explicitly rejected: join
  performance, a date predicate in every query, gap and overlap problems at boundaries,
  portability, and breakage when sources have inconsistent natural keys.
- Balance measures are semi-additive; periodic snapshot facts are dense by construction.

Grade C. This is practitioner canon, widely adopted and internally consistent, but it is
consultant guidance rather than peer-reviewed evidence, and it describes a warehouse architecture
that many modern lakehouses only partly implement.

### 9.3 The "join key trap" is shorthand, not a term of art

The phrase "SCD type 2 join key trap" is widely used and does not appear in the Kimball technique
pages as a named concept. What those pages document is the mechanism: the fact must carry the
*version* surrogate key, and a join on the natural key alone multiplies facts across versions.
Do not cite the phrase as a defined term; state the mechanism.

The phrase also hides a second half. A join that fans out inflates totals; a join that misses
deflates them. Design Tip #171's "unknown" member row, for keys that arrive before their
dimension, addresses the deflation case. Both halves need a row-count assertion.

### 9.4 Snapshots and spells in practice

The Kimball position on periodic snapshots and the practitioner material on inventory snapshots
agree on the mechanics: the movement ledger is ground truth, the snapshot is its integral, and a
backdated correction must be merged rather than inserted, or the snapshot double-counts at its
own grain. The practitioner material adds two operational warnings: the opening anchor belongs at
the close of the day *before* the window, and a correction must be recomputed forward from the
earliest corrected date. Grade C for the modelling literature, D for the practitioner blogs.

Gap-and-island is likewise a standard pattern, visible in public question banks as filter →
date minus row number → group → count, with a stockout variant that counts spells and finds the
longest. Grade E for the question banks. The gap in that literature is the one demonstrated in
section 6.6: the pattern assumes a dense calendar, and on snapshots from a system that emits no
row for a quiet day it fragments one spell into several. A candidate who writes the textbook
answer without asking about density has produced a query that runs and is wrong.

The honest summary for both patterns is that they are well understood and poorly implemented. The
defect is almost never in the concept; it is in an anchor that overlaps the first day, an
`INSERT` where a `MERGE` was needed, or a sparse calendar passed to an island join.

### 9.6 Allocation: the oldest algorithm in the topic

The largest remainder method is Hamilton's method from the American apportionment debates of the
1790s, and its properties are well documented: it satisfies the quota rule, it is not
house-monotone, and it is subject to the Alabama paradox, in which increasing the total can
reduce a party's allocation. Those pathologies matter for legislative seats and almost never for
pallet allocation, where the totals are small and the tiebreak is arbitrary but stated.

What matters in supply chain practice is narrower: the parts must sum to the whole, the rule must
be deterministic, and the tiebreak must be documented. The public SQL recipes for this are
uniformly community-sourced — Stack Overflow answers, a SQLAuthority worked example, a GitHub
gist — and they converge on the same three-step shape. Grade E, and adequate: the algorithm is
simple enough that community consensus is reliable, which is not true of the temporal material.

### 9.7 What the evidence does not establish

Three things this topic cannot support with the sources consulted.

1. **No measured defect rate.** There is no public study of how often as-of joins are written
   without date predicates, or how much revenue is double-counted by fan-out. Claims of the form
   "80% of warehouses have this bug" are not available and should not be invented.
2. **No comparative engine benchmark.** The engine matrix in section 8.2 is a feature table, not
   a performance comparison. Whether a window-function as-of join is faster than an `ASOF JOIN`
   on a given warehouse depends on the data and the optimizer.
3. **No causal claim about the opening case.** The 94%-to-71% backtest collapse is illustrative.
   It is a realistic mechanism, not a documented incident, and should be labelled as such
   wherever it is reused.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Current-row join on a versioned dimension | A past period's numbers change when the dimension is reloaded | No date predicate, so the current version is used | Backtest leakage; historical reports restate | As-of join; store the knowledge date with the report |
| Join on the natural key with no version key | Totals roughly equal to the number of versions times the old total | One fact matches every version | Revenue and quantity multiplied | Surrogate key on the fact; row-count assertion |
| Inclusive `valid_to` | An order on the boundary date matches two versions or none | `<=` on the end date | Double count or dropped row on one day per change | Half-open intervals, tested at the boundary |
| Gap in the interval set | A handful of facts join to nothing | Next `valid_from` does not equal prior `valid_to` | Silent revenue loss | Test that the intervals tile |
| Snapshot summed across dates | Inventory "value" in the millions for one SKU | Unit-days reported as units | Absurd totals that get normalised away | Declare additivity; use `LAST` or `AVG` |
| Opening anchor double-counted | Closing balance exceeds opening plus receipts | Opening posted as a movement and as a balance | Inventory overstated by the opening amount | Reconcile to the snapshot, not to the ledger |
| Snapshot treated as dense when it is sparse | Mean inventory shifts when a feed fails | Missing day read as zero or skipped | Distorted averages and spell lengths | Date spine; classify states explicitly |
| Movement ledger queried for a balance | "On hand" grows monotonically over years | Cumulative sum with no anchor | Nonsense stock levels | Anchor at a known balance and sum forward |
| Allocation by floating point | Allocated units sum to supply ± 1 or ± 2 | Independent rounding | Units lost or invented; customer disputes | Integer floor plus largest remainder |
| Non-deterministic tiebreak | Two runs of the same allocation differ | `ROW_NUMBER` over equal keys | Unauditable allocation | Append a unique column to `ORDER BY` |
| Sparse-calendar gap-and-island | Spell counts higher than the operations team recognises | Missing day splits an island | Overstated stockout frequency | Dense calendar before islanding |
| Engine-default drift | Numbers differ between the warehouse and a laptop | Integer division, `NULL` ordering, week start | Irreproducible analysis | Pin dialect and session settings; test on both |
| Isolation read across two statements | Total is off by exactly one order | Non-atomic read of related tables | Planner acts on an impossible state | Single snapshot or mart cut-off |
| `SUM` of an empty set read as zero | A no-data day appears as a stockout | `SUM` returns `NULL` on no rows, `0` after `COALESCE` | Phantom stockouts | Distinguish "no rows" from "zero value" |

Three deserve a note. The **current-row join** is the most consequential because it is the
default: correct for exactly one question, "what does the item look like today", and wrong for
every historical one. **Sparse-calendar gap-and-island** is the most deceptive because the query
is textbook-correct and it fails whenever the source omits rows for quiet days; test the row count
per item-location-day against the calendar. **Engine-default drift** is the hardest to trace, and
the fix is the header in section 5 step 12.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Effective dates in the source match business reality | The as-of join trusts `valid_from` | Keyed-in changes backdated to a policy date | Sample changes and compare to the contract or approval date |
| Validity intervals do not overlap | The join must return one version | Late entry inserted without shortening the prior row | Count versions per natural key per date; expect 1 |
| Intervals are gapless | "Greatest `valid_from`" is sufficient | A change was deleted or never loaded | Compare `valid_to` to the next `valid_from` |
| Transaction timestamps are monotonic | Bitemporal queries order by them | Multiple producers with skewed clocks | Check for record times before the event time |
| Snapshot days are complete | Mean and spell calculations depend on density | Feed failures, new items, closed locations | Row count per item-location-day against the spine |
| A missing snapshot means no change | Carrying forward is valid | The source omits zero rows deliberately | Ask the source owner; compare to the ledger |
| Units are stable within a column | Arithmetic presumes one unit | Unit of measure changed and was overwritten | SCD type 2 on the unit column; check for jumps |
| Engine defaults are known and pinned | Reproducibility | Version upgrade changes a default | Run the test suite on the target engine |
| The calendar is Gregorian and single-time-zone per query | Date arithmetic is naive about zones | Cross-border events near midnight | Compare UTC and local renderings of a boundary event |
| Allocation weights sum to a positive number | Division by total demand | Zero-demand or excluded orders in the set | Assert `SUM(weight) > 0` before dividing |

Two limits deserve stating. **The patterns reconstruct belief, not truth.** A bitemporal table
tells you what the warehouse recorded, not what the planner actually looked at, which may have
been a spreadsheet exported the week before or a system that was never ingested.
**SQL dialects diverge faster than standards converge.** The portable subset used here is
deliberately conservative — joins with date predicates, window functions, common table
expressions, `CASE`, integer arithmetic. Anything beyond that buys readability at the cost of
portability, and the trade belongs in the model header.

## 12. Exercises

### Exercise 1 — Four answers to one question (application, 45 minutes)

Using section 6.1, compute full cases for a 240-each order dated 2026-03-02 under four joins:
current row, as-of valid time, as-of valid and transaction time at 2026-01-25, and at 2026-02-16.

**Deliverable:** a four-row table of version, case pack, full cases, and the question answered.
**Acceptance criteria:** all four values correct; each row names its question; one sentence on
which row is the correct backtest basis.
**Time:** 45 minutes.

### Exercise 2 — Reconcile balance and movement (calculation, 40 minutes)

From section 6.4, compute closing balance, mean on-hand, mean daily issues, and days of supply;
then recompute the closing balance under the double-counted anchor.

**Deliverable:** the arithmetic with units, plus a paragraph on why the wrong number survives.
**Acceptance criteria:** 610 units; 780.0 mean on-hand; 6.12 days; 1,810 double-counted; error
stated as 1,200 units and 197%.
**Time:** 40 minutes.

### Exercise 3 — Allocate so the parts sum (calculation, 50 minutes)

Allocate 4,800 units across the five orders in section 6.5 by floor plus largest remainder, then
by independent rounding and by "remainder in the last row".

**Deliverable:** three allocation tables with row and column totals, plus a defensible rule.
**Acceptance criteria:** largest remainder sums to exactly 4,800; floors sum to 4,798 with
residual 2; award order justified by fractional part and a stated tiebreak; the rounding result
reported honestly, including that it happened to sum correctly here.
**Time:** 50 minutes.

### Exercise 4 — Stockout spells with a missing day (analysis, 45 minutes)

Using section 6.6, compute spell count and lengths under three definitions: zero-row count,
first-to-last span, and gap-and-island on the filtered rows.

**Deliverable:** the three answers, the island key traced for each zero row, and a recommended
metric definition.
**Acceptance criteria:** 5 zero rows; 6-day span; two islands of 2 and 3 days with key values
shown; a recommendation stating how an unknown day is treated.
**Time:** 45 minutes.

### Exercise 5 — SCD design at attribute level (design, 60 minutes)

Assign type 1 or 2 to each attribute of a supplier master: legal name, tax identifier, payment
terms, standard lead time, approved status, country of registration, bank account.

**Deliverable:** a table of attribute, type, justification, and a test that would catch a wrong
choice.
**Acceptance criteria:** at least two type-1 attributes with a defensible reason; one genuinely
arguable case with the argument stated; every test executable rather than aspirational.
**Time:** 60 minutes.

### Exercise 6 — Deduplication audit (analysis, 35 minutes)

A staging table has 1,000,000 rows and 12,400 duplicate groups on `(order_id, line_number)`;
3,100 groups tie on the single sort column `source_updated_at`.

**Deliverable:** the post-deduplication row count, the number of arbitrary survivors, and a
corrected `ORDER BY`.
**Acceptance criteria:** 987,600 rows; 3,100 arbitrary survivors; a tiebreak unique within the
group and stable across reloads.
**Time:** 35 minutes.

### Exercise 7 — Engine-default differential (experiment, 50 minutes)

Run one query that converts eaches to cases and returns the latest row per order on two engines,
or on one engine under two session configurations differing on integer division, `NULL`
ordering, or week start.

**Deliverable:** both result sets, a diff, and a header comment that pins the behaviour.
**Acceptance criteria:** at least one genuine difference produced or explained; each traced to a
named default; the header sufficient for a third party to reproduce the result.
**Time:** 50 minutes.

### Exercise 8 — Bitemporal backtest design (design, 75 minutes)

Design the feature table for a weekly demand model backtested honestly over 18 months.

**Deliverable:** a feature specification with source, temporal kind, join type, knowledge date
rule, and leakage test per feature.
**Acceptance criteria:** every feature has an explicit knowledge date; at least one feature is
dropped as unavailable at that date; a leakage test would fail if the current-row join were
substituted.
**Time:** 75 minutes.

### Exercise 9 — Red-team a mart (critique, 60 minutes)

Take an inventory mart you know, or the one in this topic, and find every place where grain, join
key, or temporal predicate is unstated.

**Deliverable:** a defect list ordered by consequence, each with the query that demonstrates it.
**Acceptance criteria:** at least six defects; each has a demonstration query, not a description;
the top two quantified in units or currency.
**Time:** 60 minutes.

## 13. Self-check answers

**Exercise 1 arithmetic.**

```text
order quantity = 240 eaches

current-row join, v3, case_pack 6:
    240 / 6  = 40 cases

as-of valid time 2026-03-02, v2, case_pack 12:
    2026-03-02 is within [2026-01-19, 2026-09-01)
    240 / 12 = 20 cases

as-of valid and transaction time 2026-01-25, v1, case_pack 24:
    at that instant v2 had not been recorded (recorded 2026-02-02)
    240 / 24 = 10 cases

as-of valid and transaction time 2026-02-16, v2, case_pack 12:
    v2 recorded 2026-02-02, so it is visible
    240 / 12 = 20 cases
```

The backtest basis is the third row. A model trained to predict the January plan must see 24
eaches per case, because that is what the planner saw; 20 cases gives the model a correction it
could not have known. A common wrong answer uses `BETWEEN valid_from AND valid_to`, which returns
20 for the valid-time case and two rows on 2026-09-01.

**Exercise 2 arithmetic.**

```text
opening                       1,200
receipts                      3,400
issues                       -3,950
adjustments                     -40
closing                         610 units

snapshot sum                 24,180 unit-days over 31 days
mean on-hand                24,180 / 31 = 780.0 units
mean daily issues            3,950 / 31 = 127.42 units/day
days of supply               780.0 / 127.42 = 6.12 days

double-counted anchor:
receipts become 3,400 + 1,200 = 4,600
closing becomes 1,200 + 4,600 - 3,950 - 40 = 1,810 units
error = 1,200 units = 1,200 / 610 = 196.7% of the true closing balance
```

The erroneous number survives because it is internally consistent with the movement feed. Only a
reconciliation against a complete snapshot catches it; if the snapshots derive from the same
feed, the defect is invisible until a physical count.

**Exercise 3 allocation.**

```text
total demand 7,300; supply 4,800; ratio 4,800 / 7,300 = 0.6575342

C-101  2,400 x 0.6575342 = 1,578.08  floor 1,578  frac 0.08
C-102  1,800 x 0.6575342 = 1,183.56  floor 1,183  frac 0.56
C-103  1,500 x 0.6575342 =   986.30  floor   986  frac 0.30
C-104  1,100 x 0.6575342 =   723.29  floor   723  frac 0.29
C-105    500 x 0.6575342 =   328.77  floor   328  frac 0.77

sum of floors = 1,578 + 1,183 + 986 + 723 + 328 = 4,798
residual = 4,800 - 4,798 = 2

rank by fractional part: C-105 (0.77), C-102 (0.56)
award one unit to each

final: 1,578, 1,184, 986, 723, 329
check: 1,578 + 1,184 + 986 + 723 + 329 = 4,800
```

Independent rounding gives 1,578 + 1,184 + 986 + 723 + 329 = 4,800, which is correct here and is
not a property of the method. On a three-way split of 100 it gives 99. "Remainder in the last
row" gives C-105 330 and a visibly unfair outcome that a customer will notice. The defensible
rule is floor plus largest remainder with a stated tiebreak.

**Exercise 4 spell answers.**

```text
zero rows: 2026-02-11, 12, 14, 15, 16  -> 5 rows

first-to-last span: 2026-02-11 to 2026-02-16 -> 6 calendar days

island key on filtered zero rows:
    date         rn   date - rn   island
    2026-02-11    1        10        A
    2026-02-12    2        10        A
    2026-02-14    3        11        B
    2026-02-15    4        11        B
    2026-02-16    5        11        B

spells: A = 2 days, B = 3 days
```

The correct report is "5 confirmed zero days; 13 February unknown; one continuous stockout of 5
to 6 days". Counting the unknown day as a stockout reports 6 and requiring confirmation reports
5; both are defensible. Reporting two spells is not, because the state never changed.

**Exercise 5 rubric.**

A strong answer assigns type 1 to legal name, tax identifier, and bank account, none of which
prices a historical number; change history for the bank account belongs in the payment system.
It assigns type 2 to payment terms, standard lead time, approved status, and country of
registration, each of which changes what a past transaction meant. The arguable one is
approved-supplier status: type 1 if approval is operational, type 2 if a past purchase must be
defensible to an auditor. A weak answer marks every attribute type 2 "for safety", which doubles
the row count and forces a date predicate into every join for no analytical gain.

**Exercise 6 arithmetic.**

```text
rows before                    1,000,000
duplicate groups                  12,400
rows removed = 12,400 x (2 - 1) = 12,400   (assuming pairs)
rows after = 1,000,000 - 12,400 = 987,600

groups with a tie on source_updated_at = 3,100
survivors chosen arbitrarily = 3,100
```

The corrected order is `source_updated_at desc, ingest_ts desc, source_file_row_id desc`, where
the last column is unique within the group. If none exists, hash the full row and record the
choice.

**Exercise 7 model answer.**

Expected differences, in order of likelihood:

1. **Integer division.** `240 / 24` returns `10` in engines that perform integer division on
   integer operands and `10.0` elsewhere. The fix is an explicit `CAST` or a `FLOOR`, stated once.
2. **`NULL` ordering.** `order by source_updated_at desc` places `NULL` first in some engines and
   last in others. A row with a null timestamp becomes the "latest" row in one of them.
3. **Ranking ties.** Without a unique tiebreak, `row_number` over equal keys is arbitrary.
4. **Empty-set aggregates.** `SUM` over no rows returns `NULL`; `COALESCE(SUM(...), 0)` returns
   zero, which turns a missing day into a stockout.
5. **Week start.** `date_trunc('week', ...)` starts on Monday in some engines and Sunday in
   others; ISO week numbering can differ from both.
6. **Session time zone.** A `TIMESTAMP` rendered in local time can land on a different date.

A complete header comment names the engine and version, the session time zone, the week-start
convention, the integer division behaviour, and the tiebreak column.

**Exercise 8 rubric.**

A strong submission gives every feature a knowledge date expressed relative to the forecast
origin, uses as-of joins for every versioned attribute, and drops at least one feature that
cannot be made point-in-time rather than approximating it. The leakage test is executable: rebuild
with the current-row join substituted and assert that a feature changes for a historical
origination date. A weak submission lists features and sources but leaves the knowledge date
implicit, which is invisible until the backtest is challenged.

**Exercise 9 rubric.**

A strong defect list is ordered by consequence, not by file order, and quantifies the top two in
units or currency. Each defect has a demonstration query that returns two different numbers.
Typical findings: a snapshot summed across dates in a dashboard tile; a fact joined to the
dimension on the natural key; an `ORDER BY` with no tiebreak in a deduplication; an allocation
rounded in the BI tool rather than in SQL. A weak submission lists "data quality issues" without
a reproducing query, which no engineer can act on.

## 14. Interview questions

**1. "What is the difference between valid time and transaction time?"**
Strong: defines both, gives a divergence — a price agreed in January and keyed in February — and
notes that `FOR SYSTEM_TIME AS OF` covers transaction time, not valid time. Weak: "one is when it
happened, one is when we loaded it", with no view on which a backtest needs.

**2. "How would you join an order to the item master as it was on the order date?"**
Strong: `order_date >= valid_from and order_date < valid_to`, greatest `valid_from`, plus a
tiebreak; notes that the window form and the correlated subquery agree when intervals are clean.
Weak: filters the dimension to `is_current = true`.

**3. "Why is summing a daily inventory snapshot wrong?"**
Strong: the result is unit-days, not units, because on-hand is semi-additive; the legal uses are
last value for a period-end stock and mean for an average. Weak: "it double counts", which is
true and does not say what the number is.

**4. "What is the SCD type 2 join key trap?"**
Strong: describes the mechanism — joining on the natural key matches every version and multiplies
the measure — and adds the opposite failure, where an interval gap drops facts. Weak: recites
"use a surrogate key" without being able to say what breaks without one.

**5. "When would you choose type 1 over type 2?"**
Strong: the choice is per attribute and turns on whether a historical number changes meaning; a
misspelling is type 1, a case pack is type 2; type 2 costs rows and date predicates. Weak: "type 2
is best practice".

**6. "How do you allocate 4,800 units across orders demanding 7,300?"**
Strong: floor plus largest remainder, parts sum exactly, deterministic tiebreak; notes that
independent rounding usually works and the failure is found by a customer. Weak: "allocate
proportionally", ignoring the integer unit.

**7. "How would you find the longest stockout in a year of daily snapshots?"**
Strong: gap-and-island, then immediately asks whether the calendar is dense, and states how an
unknown day is treated. Weak: writes the textbook query and stops.

**8. "A report's numbers changed after a pipeline migration. What do you check first?"**
Strong: the date predicate on versioned dimensions, then the opening anchor, then engine defaults,
and asks for the knowledge date of each report. Weak: blames the data.

**9. "What is the difference between a snapshot fact and a transaction fact?"**
Strong: grain, density, additivity — dense and semi-additive versus event-level and fully
additive — and the snapshot is the integral of the ledger. Weak: "snapshots are point in time".

**10. "Why might the same query return different rows on two engines?"**
Strong: names integer division, `NULL` ordering, and ranking ties, plus the session time zone,
and fixes them in a model header. Weak: "SQL is standard", which is true of syntax and false of
results.

**11. "What does point-in-time correctness mean for a feature table?"**
Strong: every feature computable from data recorded before the forecast origin, tested by
rebuilding with the knowledge date pinned. Weak: "no future data", which misses restatements.

**12. "How would you prove that a historical report is reproducible?"**
Strong: states the knowledge date, engine version, session settings, and tiebreak columns, then
re-runs and diffs; notes that an implicit "now" cannot be reproduced. Weak: "we version the SQL".

## 15. Key terms

- **Valid time** — when a fact was true in the world.
- **Transaction time** — when the warehouse recorded the fact.
- **Bitemporal** — carrying both valid and transaction time on one row.
- **As-of join** — selecting the dimension version in force at a stated date.
- **Temporal join** — joining two tables on overlapping validity periods.
- **Point-in-time correctness** — a result contains only what was knowable at a stated instant.
- **SCD type 1** — overwrite the attribute, keeping no history.
- **SCD type 2** — insert a new row per version, with a new surrogate key.
- **SCD type 3** — keep one previous value beside the current one.
- **Natural key** — the business identifier, constant across versions.
- **Surrogate key** — a warehouse-assigned key unique to one version.
- **Half-open interval** — `[from, to)`, includes the start and excludes the end.
- **Current-row flag** — a boolean marking the open version.
- **Snapshot fact** — one row per entity per period, recording state.
- **Transaction fact** — one row per event, recording flow.
- **Accumulating snapshot** — one row per process instance, updated at milestones.
- **Semi-additive** — additive across every dimension except time.
- **Grain** — the exact meaning of one row, stated as a sentence.
- **Date spine** — a complete calendar that makes missing days visible.
- **Movement ledger** — append-only record of receipts, issues, and adjustments.
- **Opening anchor** — the balance at the close of the day before the window.
- **Allocation** — assigning constrained supply to competing demand.
- **Apportionment** — dividing an integer total so the parts sum exactly.
- **Largest remainder** — floor each share, award leftovers by fractional part.
- **Deduplication** — reducing duplicates to one row per business key.
- **Tiebreaker** — a unique column appended to `ORDER BY` for determinism.
- **Gap-and-island** — grouping consecutive same-state rows into spells.
- **Stockout spell** — a contiguous run of days with zero available stock.
- **Isolation level** — the visibility rules a transaction sees.
- **Fan-out** — row multiplication from a join matching several rows.

## 16. Sources

**Peer-reviewed and academic**

- Snodgrass, R. T. *Managing Temporal Data: A Five-Part Series*. TimeCenter Technical Report
  TR-28. The valid-time and transaction-time state tables and the sequenced join. Grade A.
- Snodgrass, R. T. (1999). *Developing Time-Oriented Database Applications in SQL*. Morgan
  Kaufmann. ISBN 978-0-08-050422-3. Bitemporal tables and audit-log triggers. Grade A.
- Kulkarni, K., and Michels, J.-E. (2012). "Temporal Features in SQL:2011." *SIGMOD Record*,
  41(3). The standard's temporal model and what it does and does not cover. Grade A.

**Standards and official documentation**

- ISO/IEC 9075:2011, *Information technology — Database languages — SQL*, Part 2: Foundation.
  Temporal features: `PERIOD FOR`, system versioning, `FOR SYSTEM_TIME`, temporal keys. Grade A;
  consult the current edition before citing clause numbers.
- Microsoft Learn, "ROW_NUMBER (Transact-SQL)." Results are deterministic only when the partition
  and sort columns are unique. Grade B.

**Practitioner canon**

- Kimball Group, "Type 2: Add New Row"; Kimball, R. (2008), "Slowly Changing Dimensions, Part 2."
  The mechanics of versioning and the surrogate key. Grade C.
- Kimball Group, "Design Tip #8: Perfectly Partitioning History with the Type 2 Slowly Changing
  Dimension." History partitioning and the fan-out mechanism. Grade C.
- Kimball Group, "Design Tip #100: Keep Your Keys Simple." Rejection of natural key plus date
  range as a fact-to-dimension join. Grade C.
- Mundy, J. (2015), "Design Tip #171: Unclogging the Fact Table Surrogate Key Pipeline." Unknown
  member rows and late-arriving dimensions. Grade C.
- Kimball, R. (2008), "Fact Tables"; Kimball Group, "Additive, Semi-Additive, and Non-Additive
  Facts"; Becker, B. (2014), "Design Tip #167: Complementary Fact Table Types." Grade C.

**Vendor and survey documentation**

- "Survey of SQL:2011 Temporal Features," illuminatedcomputing.com, 2019; canmydb.com,
  "Temporal / System-Versioned Tables." Cross-engine comparison used for section 8.2. Grade D;
  dated, re-verify.
- Snowflake community article, "How the ROW_NUMBER Function Can Return Non-Deterministic
  Results." Grade D.
- Summit Analytics, "Inventory Snapshot Patterns: Modeling Stock Correctly." Anchors, density,
  and restatement. Grade D; accessed 2026.

**Community and question banks**

- Stack Overflow, "Lossless / Fair Share Division in SQL"; SQLAuthority, "Whole Unit Allocation:
  Keep a Proportional Split Exact." Largest-remainder implementations. Grade E.
- Wikipedia, "Largest remainder method." Hamilton's method, the quota rule, and the Alabama
  paradox. Grade E for the article; the result is textbook.
- Data Career School, "Stockout Spells" and "Longest Stockout Run" SQL practice questions. Grade
  E; evidence of what is asked, not of correctness.
- Ben-Gan, I. (2019), "Row Numbers with Nondeterministic Order," SQLPerformance.com. Grade C.

**Note on citations.** No public study was found that measures how often as-of joins are written
without date predicates, or that quantifies revenue lost to fan-out. No claim of that kind is
made here, and the 94%-to-71% backtest in section 1 is illustrative rather than documented.
Engine support changes with every release; the matrix in section 8.2 reflects the sources above.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: `03_ingestion-patterns.md` — not yet written.
- Foundation: [`01_layered-architecture.md`](01_layered-architecture.md)
- Foundation: [`02_batch-microbatch-streaming.md`](02_batch-microbatch-streaming.md)
- Next: `05_python-structure-tests.md`, then `06_dbt-models-tests.md` — neither yet written.
- Related: [`../01A_foundations_units_time_assumptions/07_hierarchy-design.md`](../01A_foundations_units_time_assumptions/07_hierarchy-design.md)
- Survey file: [`../../../course/03_data_engineering_interoperability.md`](../../../course/03_data_engineering_interoperability.md)
- Return to the part map: [`../_index.md`](../_index.md)


