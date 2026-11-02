> Module 03 — Data Engineering and Interoperability
> Topic 03 of 11
> Estimated study time: 3.5 hours
> Prerequisites: 03.01, 03.02

# Ingestion Patterns

---

## 1. Why this topic exists

In March 2026 a distributor of sterile surgical consumables closed its quarter with a book
inventory of EUR 18.6 million across nine distribution centres (DCs). The physical count came to
EUR 18.2 million. The difference, EUR 412,000 or 2.2%, was recorded as "inventory variance" and
assigned to the operations director, who spent three weeks explaining shrinkage that had not
happened.

None of the four causes was shrinkage. One was a nine-and-a-half-hour gap in a change stream that
nobody noticed because the job had restarted successfully. One was a supplier file that had been
read while it was still being written. One was a receipt date that arrived three days after the
record that carried it. One, a genuine EUR 13,700, was shrinkage. The other EUR 398,300 was
ingestion, and it had been accumulating since December.

This is the defining property of the topic. **An ingestion defect does not throw an error. It
produces a number that is plausible, arrives on time, and passes every downstream test that was
written against the shape of the data rather than its completeness.** By the time it is visible,
it has been through a forecast, a safety-stock calculation, a purchase order, and a physical
count. It surfaces as an operational failure in a different department, in a different quarter,
and it is diagnosed by people who have no reason to suspect the pipeline.

The previous topic, 03.02, chose a latency class — batch, micro-batch, streaming — and therefore
how often the pipeline runs. This topic chooses what each run is allowed to read, what it is
allowed to miss, and what it must do when it runs twice. Those three questions are independent of
latency. A nightly batch job and a five-second streaming job can both lose the same deletes, both
double-count the same retry, and both mis-handle the same backdated row.

Three claims organise what follows.

1. **Ingestion is a claim about what cannot happen.** A full extract asserts the source does not
   change during the read; a timestamp extract asserts every change touches the column; a log
   reader asserts the log outlives the longest outage. Each assertion is false somewhere, and
   the failure is silent exactly where it was made.
2. **Idempotency is a property of the key, not the tool.** "Exactly once" describes a failure
   model. It does not help if the source delivers two rows for one key.
3. **Deletes and late rows are the two things that incremental strategies structurally cannot
   see.** Everything else is engineering. These two are the reason a pipeline can be correct
   every day and wrong for a quarter.

**Why this matters to a data scientist specifically.** Every model in this program inherits the
ingestion layer's blind spots as if they were business facts. A demand history that is missing
the two days after a long weekend teaches a forecast model that demand is low after holidays. A
stock ledger that never received a delete teaches an inventory model that discontinued items are
available. A receipt feed that drops 2.3% of lines teaches a safety-stock calculation a demand
level that is too low. None of these produces a modelling error that a validation split can
detect, because the defect is in the label, not in the fit. The most expensive model failures in
supply chain analytics are ingestion failures that were never named as such.

## 2. Learning outcomes

By the end of this topic you can:

1. **Compare** full, incremental, change-data-capture (CDC), and file-drop ingestion on what each
   can see, what each structurally cannot see, and what each costs per row and per day.
2. **State** the three clocks in any feed — event time, commit time, and ingest time — and
   **identify** which one a watermark is built on and which one it should be.
3. **Design** a high-water mark with an overlap window, a tie-breaker, and an advance-after-commit
   rule, and **justify** the window length from measured lateness rather than from habit.
4. **Distinguish** log-based, trigger-based, and timestamp-based CDC by their failure signatures,
   and **choose** among them for a named source system with stated constraints.
5. **Detect** deletes where the source does not report them, and **design** a tombstone that
   preserves history without corrupting current-state queries.
6. **Diagnose** a schema drift event on arrival and **decide** between quarantine, coercion, and
   batch failure, with a stated consequence for each.
7. **Write** an idempotent load using deduplication plus merge semantics, and **explain** why the
   merge is not idempotent without the deduplication.
8. **Attribute** an inventory variance to an ingestion defect using counts, dates, and affected
   keys rather than narrative.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold one picture: a source system on the left, a raw layer on the right, and four different
machines in between, each of which can drop something on the floor without telling anyone.
The letters mark where each defect enters.

```text
  SOURCE SYSTEMS              EXTRACTION                 LANDING            RAW
  ──────────────              ──────────                 ───────            ───

  ERP (Oracle)
   inventory, receipts  ──►  [ redo-log reader ]  ──┐
        │                    offset / SCN          │
        │                    (A) log recycled      │
        │                        before read       │
        │                                          │
  3PL portal                                      │
   9 CSV files          ──►  [ file watcher ]   ──┤
        │                    40 s to write         │
        │                    (B) reads on create,   │
        │                        not on close        │
        │                                          │
  TMS (SaaS API)                                 ▼
   shipment events      ──►  [ paginator ]  ──► [ STAGE ]  ──► [ RAW ]
        │                    cursor                 │           append-only
        │                    (C) cursor advanced    │           _ingest_ts
        │                        before response    │           _batch_id
        │                        is validated       │           _source_file
        │                                           │           _record_hash
  WMS nightly                                   (D) source not
   snapshot file        ──►  [ bulk loader ]        deduplicated
        │                    full extract,          before merge
        │                    no snapshot isolation        │
        │                                                 ▼
        │                                        [ MERGE / UPSERT ]
        │                                         (E) watermark
        │                                             advanced before
        │                                             target write verified
        ▼                                                 │
   (F) hard deletes never                                 ▼
       propagated                              [ CONFORMED current state ]
```

Six properties follow, and each explains a class of failure that no amount of testing at the
mart can repair.

**Property 1 — There are three clocks, and only one is yours.** *Event time* is when the thing
happened: the pick, the receipt, the shipment. *Commit time* is when the source database made the
change durable and visible. *Ingest time* is when your platform received it. A receipt can have
an event time of Monday, a commit time of Thursday, and an ingest time of Thursday plus forty
seconds. A watermark built on event time will declare Thursday's rows late and never look for
them again. A watermark built on commit time will find them, but cannot answer a business
question phrased in event time without a second mapping. Almost every late-arrival defect in
practice is a watermark built on one clock and a business question asked on another.

**Property 2 — Extraction reads changes or state, and confusing the two is the root error.** Log
readers and trigger tables deliver *changes*: one row per insert, update, or delete. Full extracts
and nightly snapshots deliver *state*: the current value of every row, with the history
implicitly discarded. A change feed can be replayed and summed. A state feed cannot, because
summing two snapshots double-counts. A pipeline that mixes them without declaring which is which
will produce inventory that is either multiplied by the number of days or frozen at whatever the
last snapshot said.

**Property 3 — Every strategy is a claim about what cannot happen.** Full extract: "nothing
changes during the read, or I do not care." Timestamp incremental: "every change updates the
timestamp column." Trigger CDC: "the trigger is installed, enabled, and covers this operation."
Log CDC: "the log is retained longer than the longest outage." File drop: "the file is complete
when it appears." Write the claim down at design time and the failure mode writes itself.

**Property 4 — Idempotency is a key, not a checkbox.** A load is idempotent when running it twice
with the same input produces the same target state as running it once. That requires a key that
identifies the logical row, a deterministic rule for choosing among versions of that row, and a
write that replaces rather than appends. A tool can supply the third. Only you can supply the
first two.

**Property 5 — Deletes are the hardest thing to see, because absence has no row.** A hard delete
removes the evidence of its own occurrence. Incremental extraction by timestamp cannot see it,
file snapshots can only infer it by comparing key sets, and only a log or a trigger records it
explicitly. A pipeline that never propagates deletes does not lose data; it retains data that
should have gone, which is worse, because the retained row looks exactly like a valid one.

**Property 6 — The failure is delayed, plausible, and attributed to operations.** An ingestion
defect shifts a level. A level shift in inventory looks like shrinkage, in demand looks like a
market change, and in service looks like a warehouse problem. The delay between cause and
symptom is typically one planning cycle plus one physical count, which in most companies is one
to four months. This is why the topic belongs in a data engineering module and why its evidence
is usually found in an operations meeting.

### 3.1 The extraction matrix

The single most useful table in this topic. Read down the last column before choosing a strategy.

| Strategy | What it reads | Sees inserts | Sees updates | Sees hard deletes | Boundary it needs | Characteristic silent failure |
|---|---|---|---|---|---|---|
| Full extract | Current state of every row | Yes | Yes, as current value | Only by absence between runs | None, but needs a consistent read | Read is smeared over minutes; cost grows with history, not with change |
| Timestamp incremental | Rows where `updated_at` moved | Yes | Only if the column is maintained | No | `max(updated_at)` | Backdated rows fall below the mark; bulk operations bypass the column |
| Trigger CDC | Shadow table written inside the transaction | Yes | Yes | Yes, if the delete trigger exists | Trigger sequence or commit order | `TRUNCATE` and data-definition changes are missed; triggers disabled in maintenance |
| Log-based CDC | Database transaction log | Yes | Yes, with before and after images | Yes, explicitly | Log sequence number (LSN) or system change number (SCN) | Log retention expires during an outage; snapshot and stream overlap or gap |
| File drop | Whatever the producer wrote | Yes | As full state | Only by absence | File name and arrival time | File read before it is complete; duplicate delivery; positional columns reordered |

Two things are worth stating plainly. First, only trigger and log CDC can see a hard delete as
an event; everything else infers deletion from the disappearance of a key, which requires a
complete key set and therefore either a full extract or a maintained manifest. Second, the cost
drivers differ in kind: full extract scales with table size, incremental with change volume, CDC
with retention and operational attention. A table changing 0.1% per day is a poor candidate for
full extract; one changing 80% is the reverse.

### 3.2 Where the variance comes from

Four ingestion defects produce four distinct arithmetic signatures at the physical count. Being
able to recognise the signature narrows the investigation before any query is written.

| Ingestion defect | Ledger signature | Direction | Typical magnitude |
|---|---|---|---|
| Missed updates (CDC gap, watermark too aggressive) | A small set of item-location positions frozen at an old value | Either sign, concentrated | Large per position, few positions |
| Missing rows (partial file, failed page) | Many positions carried forward from the prior period | Depends on the carry-forward rule | Small per position, many positions |
| Missed deletes | Discontinued or transferred stock still on the book | Book too high | Grows monotonically with time |
| Duplicate loads (non-idempotent retry) | Receipts or issues counted more than once | Book too high or demand too high | Proportional to retry count |

The first two are the ones that get misdiagnosed, because both look like "the warehouse is
wrong" and neither produces an error message. Section 6 works through all four in one estate.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Ingestion | The process of moving records from a source system into a raw or staging store, with metadata describing the circumstances of arrival | Used to mean the whole pipeline, including transformation | Batch identifier plus arrival timestamp |
| Full extract | A read of every row of a source object at one point in time | Assumed to be consistent because it is one query | Rows per run; run duration in seconds |
| Incremental extract | A read of only the rows that changed since a stored boundary | Assumed to capture all changes | Rows per run; boundary value |
| High-water mark | The stored boundary value above which the next extraction will read | Confused with the maximum value currently in the source | Timestamp, LSN, SCN, or integer sequence |
| Committed watermark | A high-water mark advanced only after the target write has been verified | Used interchangeably with high-water mark | Same type as the boundary |
| Lookback window | A trailing slice re-read on every run to catch rows that arrived below the mark | Confused with a retention period | Duration, usually hours or days |
| Late-arriving row | A record whose commit or ingest time is later than the boundary appropriate to its event time | Used for any delayed record, including one that merely arrived late in the day | Event time minus ingest time, in hours |
| Event time | When the business event occurred | Treated as the only time that matters | ISO 8601 timestamp with offset |
| Commit time | When the source database made the change durable and visible to readers | Ignored in business-facing extracts | Database timestamp, source clock |
| Ingest time | When the receiving platform recorded the arrival | Used as a proxy for event time | Platform timestamp, Coordinated Universal Time (UTC) |
| Change data capture (CDC) | A set of techniques for reading inserts, updates, and deletes from a source as discrete change events | Used to mean any incremental load | Event count per table per day |
| Log-based CDC | CDC that reads the database transaction log rather than the tables | Assumed to require no source configuration | Log bytes read per day |
| Trigger-based CDC | CDC that installs database triggers to write changes to a shadow table | Assumed to capture every operation | Trigger invocations per day |
| Timestamp-based CDC | CDC approximated by polling a last-modified column | Called CDC when it is only incremental extraction | Rows per poll interval |
| Redo log | Oracle's transaction log, read by LogMiner for CDC | Confused with an audit trail | Archived log sequence |
| Write-ahead log (WAL) | PostgreSQL's transaction log, read by logical decoding | Assumed to be readable without `wal_level=logical` | Log sequence number (LSN) |
| Log sequence number (LSN) | A byte offset identifying a position in a PostgreSQL WAL | Treated as a timestamp | Monotonically increasing integer |
| System change number (SCN) | Oracle's monotonic change identifier | Treated as a wall-clock time | Integer |
| Snapshot | A complete copy of a table's state at a point in time | Confused with a backup | Snapshot identifier plus start and end SCN |
| Consistent read | A read that sees the database as of one instant, despite concurrent writes | Assumed for any `SELECT` | Isolation level plus read timestamp |
| Change envelope | The record emitted by CDC containing operation type, before image, after image, and source coordinates | Ignored in favour of the after image alone | Operation code: I, U, D, T |
| Natural key | The business identifier of a row, meaningful outside the database | Assumed unique across source systems | Source-specific, often composite |
| Idempotency key | The set of columns that determines whether two records represent the same logical row version | Confused with the primary key | Column list plus comparison rule |
| Upsert | A write that inserts a row when absent and updates it when present | Assumed idempotent without deduplication | Rows merged per run |
| Deduplication | Reducing multiple records for one key to one, by a deterministic rule | Done after the merge instead of before | Survivor rows; duplicates removed |
| Tombstone | A record that marks a deletion without removing the row or its history | Treated as a physical delete | Key plus deletion timestamp plus reason code |
| Soft delete | A source-side flag indicating the row is logically deleted but physically present | Assumed to be visible to every extract | Boolean plus timestamp |
| Hard delete | Physical removal of a row from the source | Assumed detectable by incremental extraction | Row count delta; not a timestamp |
| Delete propagation | Carrying source deletions into the warehouse as tombstones | Omitted because the source "never deletes" | Deleted keys per period |
| Schema drift | Any change to the structure or typing of source data between runs | Used only for added columns | Schema version plus change type |
| Schema registry | A service that stores versioned schemas and enforces compatibility rules | Treated as a validator of business meaning | Subject, version, compatibility mode |
| Compatibility mode | The rule set governing which schema changes are permitted | Assumed to be the same everywhere | BACKWARD, FORWARD, FULL, and transitive variants |
| File drop | Delivery of data as a file to a shared location, with no protocol beyond the file itself | Treated as a reliable transport | File name pattern plus arrival time |
| Landing zone | The location where files arrive before any parsing | Confused with the raw layer | Object path plus retention |
| Manifest | A producer-supplied statement of what a delivery contains, with counts and checksums | Omitted because it is "extra work" | File count, row count, hash |
| Checksum | A fixed-length digest used to detect truncation or corruption | Assumed to detect semantic errors | SHA-256 hex digest |
| Atomic rename | Writing to a temporary name and renaming on completion, so readers never see a partial file | Assumed when the producer writes directly to the final name | Temporary path plus final path |
| Replay | Reprocessing a declared range of historical inputs with a pinned code version to reproduce a past output | Used to mean "rerun the job on today's data" | Input range plus code version |
| Backfill | Filling a gap in history that was never loaded | Confused with replay | Missing range plus reason |
| Reconciliation | Comparing counts or totals at two points in the pipeline to prove nothing was lost | Reported as a row count without a comparator | Extracted, landed, staged, and target counts |

Six of these deserve expansion, because the distinctions are routinely lost and each loss
produces a specific defect.

**High-water mark versus committed watermark.** The high-water mark is a number. The committed
watermark is a number plus a guarantee. The difference matters because the common bug is not
computing the mark wrongly; it is advancing it too early. If the process writes the new mark,
then writes the data, and then crashes, the next run starts after data that was never written.
The rule is: write the data, verify it, then advance the mark, in that order, in one transaction
where the store allows it.

**Late-arriving row versus late-delivered row.** A row is late-arriving when its *event time* is
earlier than a boundary that has already been passed. A row is late-delivered when it simply
took a while to travel. The first is a data-model problem: the period it belongs to is closed.
The second is a latency problem, solved by a faster pipe. Conflating them leads teams to buy
streaming when they need an overlap window.

**Soft delete versus tombstone.** A soft delete is a source-system behaviour: the row stays, with
a flag. A tombstone is a warehouse behaviour: the warehouse records that the entity is gone,
usually by setting a flag and a timestamp on the existing row rather than removing it. A source
with soft deletes is easy to ingest. A source with hard deletes requires you to manufacture
tombstones yourself. A warehouse that has neither cannot answer "what did we have in March"
after an item is discontinued.

**Schema drift versus schema change.** A schema change is a deliberate, announced modification.
Schema drift is the same thing observed by a pipeline that was not told. The data is identical;
the difference is whether a human was informed. Every ingestion design should assume drift and
treat announcement as a bonus.

**Idempotency versus exactly-once.** Idempotency is a property of your write: same input, same
result. Exactly-once is a property of a delivery mechanism, defined against a failure model, and
usually means "effectively once" end to end. A pipeline can have exactly-once delivery from the
broker and still double-load, because the sink is not idempotent. The reverse is also true and is
usually cheaper.

**Replay versus backfill.** Replay reproduces a result from inputs that are still held. Backfill
creates a result for inputs that were never loaded. Replay is a test of the architecture.
Backfill is a repair of the data. A team that says "we backfilled" when it means "we replayed"
has not tested whether its inputs are retained.

## 5. The mechanics, step by step

The procedure has nine steps and one precondition. The precondition is a declared row identity,
without which none of the later steps can be made idempotent. The order matters: the boundary
chosen in step 2 determines which defects are possible, and the reconciliation in step 8 is the
only thing that detects them.

### 5.1 Step 0 — Declare the identity of a row and of a change

Two keys, not one.

- The **entity key** identifies the thing: the item, the location, the order line. It is stable
  across versions and is what the business means by "the row."
- The **change key** identifies a version of the thing: the commit timestamp plus a monotonic
  tie-breaker, or the log position, or a source sequence number.

A pipeline that uses only the entity key cannot tell an update from a duplicate. A pipeline that
uses only the change key cannot tell a restatement from a correction. Both are needed, and they
belong in the staging table as explicit columns, not as an implicit `distinct`.

> **Note:** The most common single defect in warehouse ingestion is a primary key on the entity
> alone in a table that receives change events. The second most common is a primary key on the
> whole row, which makes every update look like an insert and every duplicate invisible.

### 5.2 Step 1 — Choose the strategy against what the source can prove

The choice is a constraint problem, not a preference. Answer five questions about the source in
order, and the strategy narrows to one or two options.

| Question | Why it decides the strategy |
|---|---|
| Can you obtain log access? | If not, log-based CDC is unavailable regardless of its advantages |
| Does the source maintain a reliable last-modified column? | If not, timestamp incremental is unavailable |
| Does the source delete rows physically? | If yes, only CDC or a full key comparison can see it |
| What is the largest acceptable loss window? | Sets the minimum log retention or the overlap length |
| What share of the table changes per day? | Sets whether full extract is affordable |

The costs differ in kind, and the comparison is usually made badly. A full extract of a
40-million-row table at 200 bytes per row is 8 GB per run; daily, that is 2.9 TB per year before
compression. The same table changing 0.4% per day produces 160,000 changed rows, or 32 MB per run,
or 11.7 GB per year. Roughly 250:1 in storage before read time is counted. The counter-argument is
that full extract needs no source cooperation and no boundary bookkeeping: a real saving for a
small reference table, a poor trade for a transactional ledger.

### 5.3 Step 2 — Pin the boundary and write down which clock it uses

The boundary is a value plus a comparison rule plus a clock. All three must be written down,
because each has a distinct failure.

| Decision | Safe choice | Failure if chosen badly |
|---|---|---|
| Comparison | `>=` rather than `>` | `>` loses every row that shares the boundary timestamp |
| Tie-breaker | Composite cursor `(timestamp, id)` | Ties are re-read or skipped arbitrarily |
| Clock | Commit time for capture; event time for reporting | Backdated rows fall below an event-time mark and are never read |
| Advance | Only after the target write is verified | A crash between mark and write loses the batch |
| Monotonicity | Never allow the mark to move backwards | A source clock correction causes a permanent re-read or a permanent gap |

The last row is subtle and expensive. If the source clock is corrected backwards by two hours, an
observed-maximum mark moves backwards and the next run re-reads two hours: safe if the load is
idempotent, wasteful if not. Teams often "fix" this by clamping the mark to its maximum, which
converts a clock correction into a permanent two-hour gap. The correct handling is a committed
watermark that never regresses for processing, a separate observed maximum for reporting, and an
alert when the two diverge.

### 5.4 Step 3 — Land the delivery whole

A file that is visible is not a file that is complete. The standard failure is a producer that
writes over File Transfer Protocol (FTP) or Secure FTP (SFTP) directly to the final name: the
directory entry appears within milliseconds, and the content arrives over the following seconds
or minutes. A watcher that reacts to the directory entry reads whatever exists at that moment.

The remedies are old and reliable.

- **Atomic rename.** The producer writes `name.csv.tmp` and renames to `name.csv` on close. The
  rename is atomic on a POSIX filesystem and on object stores that support copy-then-delete.
- **A completion marker.** The producer writes `name.csv.done` after the data file. The watcher
  looks only for the marker.
- **A stability check.** The watcher records the file size, waits, and records it again. Two
  identical sizes at least one polling interval apart is weak evidence of completion, not proof.
- **A manifest.** The producer states the file count, row count, and checksum. This is the only
  mechanism that detects a *complete but wrong* file, not merely a truncated one.

> **Note:** The stability check is the most common remedy and the weakest. A slow producer can
> pause for longer than the polling interval in the middle of a write. It reduces the probability
> of a partial read; it does not eliminate it. Use it only where the producer cannot be changed
> and the volume is low enough that a periodic full reconciliation catches the residue.

### 5.5 Step 4 — Detect and carry deletes

Three cases, three mechanisms, and only one of them is a data problem.

| Source behaviour | Detection mechanism | Warehouse treatment | Cost |
|---|---|---|---|
| Soft delete with flag and timestamp | Read the flag; it is an ordinary column | Set tombstone; exclude from current views | Low, if the flag is indexed |
| Hard delete with CDC available | Read the delete event from the log or trigger | Set tombstone; retain last known attributes | Low |
| Hard delete with no CDC | Compare the full key set against the previous complete snapshot | Set tombstone for absent keys | High: requires a full key scan or a reliable delta |

The third row is where teams get into trouble. Comparing key sets requires a complete list of
keys, which means either a full extract of the key column (cheap if the key is narrow and the
table is not enormous) or a producer-maintained manifest of active keys (reliable only if the
producer is contractually bound to it). A partial comparison produces false tombstones: an item
is marked deleted because its page of the extract failed, not because it was deleted. False
tombstones are worse than missed deletes, because they remove valid stock from the current view
and then produce a stock-out that no one can explain.

For the tombstone itself, four columns are the minimum: a boolean or timestamp marking deletion,
the source commit time of the deletion, a reason code where the source supplies one, and the
batch identifier that first observed it. Keep the attribute values. A discontinued item's last
known cost is still needed to value the historical stock that was held before it was
discontinued.

### 5.6 Step 5 — Absorb late arrivals with an overlap window

The overlap window re-reads a trailing slice of the source on every run. It is the standard
remedy for the event-time problem and it is not a substitute for a correct clock.

The sizing procedure has four steps.

1. **Measure the lateness distribution.** For each feed, compute commit time minus event time for
   a month of records. Report p50, p95, p99, p99.9, and the maximum, in hours.
2. **Choose the window at the percentile that matches the decision.** A daily replenishment
   decision tolerates a p99 window. A financial close tolerates p99.9 or a hard cut-off with a
   stated restatement policy. A regulatory report may tolerate neither.
3. **Compute the cost.** Extra rows read per run equals the change rate multiplied by the window
   length. Compare that with the cost of the rows that would be missed.
4. **Reconcile against the source once per period.** An overlap window cannot detect a row that
   never arrived at all. Only a periodic count or checksum comparison can.

The choice of what to do with a late row is a business decision and must be recorded. Two
policies are defensible: **restate the period** (the March ledger changes in April) or **book in
the arrival period** (March stays as reported, April absorbs the correction). What is not
defensible is doing both in different reports without saying so. The same defect that produces a
variance also produces two reconciliations that disagree, and the disagreement is usually
discovered by an auditor.

> **Note:** Late-arriving facts and late-arriving dimensions are different problems. A late fact
> needs the dimension version current at event time. A fact that arrives before its dimension
> needs an inferred member: a placeholder row with the natural key and default attributes, later
> overwritten. Applying one technique to the other produces a wrong historical attribute or a
> duplicate dimension row. (Grade C.)

### 5.7 Step 6 — Absorb schema drift at arrival

Four kinds of drift, in ascending order of danger.

| Drift | Example | Detectable by | Consequence if unhandled |
|---|---|---|---|
| Additive | A new `promised_date` column appears | Column count or registry diff | Ignored; the field is simply unavailable |
| Rename | `qty` becomes `quantity` | Positional parse breaks or column is null | Silent nulls in a non-null column |
| Retype | `qty` changes from integer to decimal string | Cast failure, or silent truncation | Quantities off by a factor or a rounding |
| Reorder | Columns move position in a headerless CSV | Nothing, unless the values are profiled | Values land in the wrong columns, plausibly |

The fourth survives every schema check, because the schema is unchanged: same column count, same
types, wrong assignment. It happens with positional flat files, with delimited files whose header
is ignored, and with hand-edited spreadsheets. Only value-level profiling defends against it: a
column that was a unit count yesterday and a price today shows a distribution change, and a
plausibility rule per column catches the swap.

Schema registries address the first three for event streams. The compatibility modes are worth
knowing precisely, because "we use Avro" is not a policy.

| Mode | Rule | Producer and consumer upgrade order |
|---|---|---|
| BACKWARD (default) | New schema can read data written with the previous version | Upgrade consumers first |
| BACKWARD_TRANSITIVE | New schema can read data written with every previous version | Upgrade consumers first |
| FORWARD | Previous schema can read data written with the new version | Upgrade producers first |
| FORWARD_TRANSITIVE | Every previous schema can read data written with the new version | Upgrade producers first |
| FULL | Both directions against the immediately previous version | Independent |
| FULL_TRANSITIVE | Both directions against all previous versions | Independent |
| NONE | No check | Undefined; a change can break any consumer |

The default is BACKWARD, which permits removing a required field. That is correct for a
consumer-driven contract and wrong for an append-only raw layer that intends to replay three
years of history, because the old records can no longer be read by the new schema. A raw layer
that must replay should generally require FULL_TRANSITIVE for its retained types, or store the
raw payload unparsed alongside a parsed view. These are vendor definitions, not a standard; a
team using a different registry should check that its mode names mean the same thing. (Grade D.)

### 5.8 Step 7 — Load idempotently

Three patterns, with different guarantees and different costs.

| Pattern | How it works | Idempotent when | Failure it permits |
|---|---|---|---|
| Merge or upsert | Match on key, update or insert | Source is deduplicated to one row per key, and the match is deterministic | Duplicate source rows cause an error or an arbitrary pick |
| Partition replace | Delete the target partition, insert the batch | The batch is exactly the partition | A partial batch replaces a complete one |
| Insert-only with dedupe on read | Append every batch; readers select the latest version | The reader's ordering is deterministic | Storage grows without bound; every consumer must dedupe correctly |

The merge pattern is the one that gets written and the one that gets written wrongly. A merge
whose source has two rows for one key is either an error or a silent arbitrary choice, depending
on the engine. Snowflake raises "Duplicate row detected during DML action" when more than one
source row matches; BigQuery raises an error for update or delete when multiple source rows
match. Both behaviours are correct, and both are avoided by the same fix: deduplicate the source
in a common table expression before the merge, with a deterministic ordering. (Grade D for the
vendor behaviours, which are documented product semantics.)

The second failure is the replay regression. A merge that updates unconditionally will overwrite
a newer row with an older one when a late batch is reprocessed. The guard is a comparison on the
change key: `when matched and source.commit_ts >= target.commit_ts then update`. Without it,
replaying January in March silently reverts every July change to the January value, and the
damage is invisible because the row count is correct.

### 5.9 Step 8 — Reconcile, then advance the watermark

This is the step that detects everything the previous steps could not prevent. Four counts, in
order, each with a defined comparator.

| Check | Numerator | Denominator | What a mismatch means |
|---|---|---|---|
| Extracted to landed | Rows written by the extractor | Rows present in the landing zone for the batch | Transport truncation or parse failure |
| Landed to staged | Rows in the landing zone | Rows accepted plus rows quarantined | A filter or cast is dropping rows silently |
| Staged to target | Distinct entity keys in staging | New or changed entity keys in the target | The merge is not applying every key |
| Target to source | Distinct entity keys in the target | Distinct entity keys reported by the source | Missed deletes, or missed inserts |

The fourth is the only one that catches deletes, and it is the one most often omitted because it
requires the source to report a count. Ask for it in the contract. A source team that cannot
report a key count for a table it owns has not finished operating that table.

> **Note:** Advance the watermark after the reconciliation, not before. A pipeline that advances
> first and reconciles second has already committed to having read the data; if the check fails,
> the only remedy is a manual backfill, and the failure window is however long it takes someone
> to notice.

### 5.10 Step 9 — Replay and backfill as declared operations

Replay and backfill are different operations with different preconditions, and both need the
same five controls.

| Control | Replay | Backfill |
|---|---|---|
| Declared range | Start and end boundary values | Missing range and the reason it is missing |
| Pinned code | Commit or model version | Commit or model version |
| Idempotent target write | Required | Required |
| Reconciliation | Against the original run's totals | Against the source's totals for the range |
| Record of the operation | Queryable log with operator and reason | Queryable log with operator and reason |

A backfill into a table that already has data for part of the range is a replay with a gap, and
it will double-count unless the write replaces the partition or merges on the change key. This
is where most "we fixed the history" incidents produce a second, larger defect.

## 6. Worked example

Halden Medical Logistics distributes sterile consumables and implant kits to 240 hospitals from
nine DCs. It runs an enterprise resource planning (ERP) system for inventory and receipts, a
third-party logistics (3PL) portal that drops nightly inventory files, a transportation
management system (TMS) with a REST API, and a warehouse management system (WMS) at three sites.
All figures are illustrative and internally consistent.

### 6.1 The estate

| Feed | Transport | Volume per day | Delivery lag (p95) | Deletes visible |
|---|---|---:|---|---|
| ERP inventory and receipts | Log-based CDC from Oracle redo | 1.9 million change events | 40 seconds | Yes |
| 3PL inventory snapshot | Nine comma-separated value (CSV) files over SFTP | 774,000 rows | 04:10 arrival | No, absence only |
| TMS shipment events | REST API, cursor pagination | 61,000 events | 5 minutes | No |
| WMS transactions | Nightly full extract | 2.4 million rows | 23:00 snapshot | No |

The ERP feed is the one that matters, because it is the only one that carries quantity changes as
events. The others carry state or movement, and all are reconciled against the ERP eventually —
which is what made the investigation possible, and why it took three weeks.

### 6.2 Defect A: the redo gap

The Oracle archived redo log retention was set to six hours, a default inherited from a storage
budget review in 2023. The CDC connector was paused for a scheduled database migration on a
Saturday, from 21:00 to 06:30, a gap of nine and a half hours. When it resumed, the archived logs
it needed had been deleted by the retention policy. The connector failed with a missing-log
error. The on-call engineer did what the runbook said: restart from the current system change
number.

The arithmetic is simple and was never performed at the time.

| Quantity | Value | Derivation |
|---|---:|---|
| Outage duration | 9.5 hours | 21:00 to 06:30 |
| Mean change events per hour | 79,167 | 1,900,000 / 24 |
| Change events never applied | 752,083 | 79,167 x 9.5 |
| Share that changed a quantity | 5.5% | Measured on a sample week |
| Quantity changes lost | 41,364 | 752,083 x 0.055 |
| Distinct item-location positions affected | 1,847 | Counted from the change stream |
| Mean changes per affected position | 22.4 | 41,364 / 1,847 |
| Retention required for the observed outage plus one repeat | 19 hours | 2 x 9.5 |

A re-snapshot of the 1,847 affected positions, at 12 milliseconds per lookup, would have taken
22 seconds. Instead, the stream resumed at the current SCN, and every position that received no
further change before the quarter-end count stayed frozen at its pre-migration value. That was
61 positions. Their total value was EUR 268,400, an average of EUR 4,400 per position, which is
consistent with implant kits and high-value sterile packs.

The failure was not the retention setting, although 6 hours against a 9.5-hour maintenance
window is a defect in its own right. The failure was that "connector restarted successfully" was
treated as "no data was lost." Those are different claims, and only the second one matters.

### 6.3 Defect B: the partial file

The 3PL wrote each DC's file directly to its final name. Each file held about 86,000 rows and
took roughly 40 seconds to appear in full. The ingestion watcher polled every 60 seconds and
read any new file immediately. On 14 October it ran three seconds after the Leeds file appeared
and read 35,400 rows before reaching end of file.

| Quantity | Value |
|---|---:|
| Rows in the complete file | 86,000 |
| Rows read | 35,400 |
| Share read | 41.2% |
| Rows absent | 50,600 |
| Completeness test applied | `row_count > 0` |
| Job status reported | Success |

The absence was invisible for four days because the mart's current-state view carried forward the
previous day's value for any item-location with no row in the latest snapshot. That carry-forward
rule is common, defensible, and dangerous: it converts a missing row into a plausible stale row,
which no freshness or null test can detect. Two hundred and twelve pick failures followed, at an
expedite cost of EUR 18,400.

The fix was a manifest: the 3PL now writes a `.done` file containing the file name, row count, and
a SHA-256 digest, and the loader refuses any file without one. The secondary fix was a
reconciliation of row count per DC against a 28-day median, which would have flagged 41.2% on the
day.

### 6.4 Defect C: the backdated receipt

The ERP sets `posted_at` to the goods-receipt date, which can be up to 72 hours before the record
is committed, because receipt posting waits for a quality approval. The ingestion job filtered on
`posted_at` and advanced a mark on `posted_at`, with a two-hour lookback. Every receipt backdated
by more than two hours fell below the mark and was never read.

| Quantity | Value |
|---|---:|
| Receipt lines per day | 4,100 |
| Share arriving with `posted_at` more than 2 hours before commit | 2.3% |
| Lines below the mark per day | 94 |
| Over 90 days | 8,487 |
| Mean line value | EUR 38.10 |
| Gross receipts value outside their period | EUR 323,355 |

The team had chosen `posted_at` deliberately, because it made the daily receipt report align with
the DC's operating day. That is a legitimate reporting requirement and an illegitimate watermark.
The defect is Property 1 in its purest form: a business-time question answered with a
business-time filter, over a process that runs on commit time.

The repair was a re-extract by commit SCN, which recovered all 8,487 lines in about nine minutes.
The pipeline healed. The forecast model did not: it had been trained for three months on a demand
history that understated receipts, and its safety-stock parameters carried the bias for another
quarter.

### 6.5 The variance that appeared in month four

The quarter-end count produced a book-to-physical difference of EUR 412,000 on a book value of
EUR 18.6 million. The investigation allocated it as follows.

| Cause | Positions affected | Variance (EUR) | Share | Detectable by |
|---|---:|---:|---:|---|
| CDC gap, stale on-hand | 61 | -268,400 | 65.1% | Key-count reconciliation against source |
| Partial file, carried forward | 1,240 | -98,200 | 23.8% | Row count per DC against median |
| Late receipts below the mark | 318 | -31,700 | 7.7% | Lateness distribution, overlap sizing |
| Genuine count and shrinkage | 44 | -13,700 | 3.3% | Physical count only |
| **Total** | **1,663** | **-412,000** | **100%** | |

The signs are all negative because the book exceeded the physical count in every category, which
is not the usual direction for a stale-update defect. The frozen positions were mostly high-value
kits already issued to hospitals, so the book retained stock that had left. A missed issue and a
missed receipt have opposite signs, and the aggregate sign is a clue to which dominates.

### 6.6 What the reconciliation could and could not attribute

Three of the four rows are arithmetic once the mechanism is known. The fourth is judgement.

- The CDC gap is unambiguous: the stream has a hole with a start SCN and an end SCN, and the 61
  positions are exactly those with no later change. The EUR 268,400 is their book value at the
  freeze date. This is attribution by construction.
- The partial file is nearly unambiguous: the file's row count, the loaded count, and the 1,240
  carried-forward positions are all recorded. The EUR 98,200 depends on a counterfactual — what
  the position would have held had the row been read — and is an estimate with a stated method.
- The late receipts are weakest. The 8,487 missing lines are known, but deciding which were
  material to a quarter-end position required repricing each against the position it would have
  affected. The EUR 31,700 is an allocation, not a measurement.
- The shrinkage is the residual after the other three, and therefore carries their error. A
  residual category is not an independent finding; presenting it as one is how a data problem
  becomes a personnel problem.

That last point is the methodological lesson of the case. **Variance attribution is only as good
as its weakest category, and the weakest category is usually the one nobody can query.**

## 7. Code, query, or calculation

One query, because it encodes the concept most often got wrong: an idempotent merge with a
deterministic deduplication and a replay guard.

```sql
-- Idempotent upsert of one day's CDC batch into current item-location state.
with src as (
    select *,
           row_number() over (
               partition by source_key
               order by source_commit_ts desc, source_lsn desc, ingest_ts desc
           ) as rn
    from staging.erp_inventory_change
    where ingest_date = date '2026-10-27'
)
merge into conformed.item_location_current as t
using (select * from src where rn = 1) as s
   on t.source_key = s.source_key
when matched and s.source_commit_ts >= t.source_commit_ts then update set
     t.on_hand_units     = s.on_hand_units,
     t.source_commit_ts  = s.source_commit_ts,
     t.is_deleted        = (s.op = 'D'),
     t.deleted_at        = case when s.op = 'D' then s.source_commit_ts end,
     t.loaded_at         = current_timestamp()
when not matched then insert (source_key, on_hand_units, source_commit_ts,
                               is_deleted, deleted_at, loaded_at)
values (s.source_key, s.on_hand_units, s.source_commit_ts, (s.op = 'D'),
        case when s.op = 'D' then s.source_commit_ts end, current_timestamp());
```

Four things in that block are load-bearing, and removing any one changes the guarantee.

- **The `row_number()` filter is not optional.** Without it, a batch containing two versions of
  one key either errors or picks arbitrarily, depending on the engine. The ordering is
  deterministic because it ends in `ingest_ts`, which is unique per arrival.
- **`>=` rather than `>` in the match condition.** A replay of an identical batch should be a
  no-op by value, and `>=` makes it idempotent in the strict sense: the same input produces the
  same rows. `>` would skip a restatement that shares a commit timestamp.
- **The tombstone is a flag plus a timestamp, not a delete.** The row keeps its attributes, so
  historical valuation and as-of queries still work.
- **`current_timestamp()` breaks byte-identical replay.** Two replays produce different
  `loaded_at` values. That is acceptable for an operational column and unacceptable in a table
  used as evidence; in the latter case, use the batch's recorded start time instead.

The query does not do four things a reader might assume: it does not detect rows missing from the
batch (section 5.9), resolve a renamed key, handle a delete arriving before its insert, or advance
a watermark. Those are separate operations with separate tests, and collapsing them into one
statement is how a pipeline acquires a failure mode nobody can name.

## 8. Data, standards, and regulatory context

Ingestion has almost no governing standard. That is the first thing to know about it, because
teams arriving from messaging or from financial reporting expect one and invent one badly.

### 8.1 What is standardised

| Area | Specification | What it constrains | Grade |
|---|---|---|---|
| Merge semantics | SQL:2003 `MERGE`, retained in later editions | The statement exists and has a defined match behaviour; duplicate-source behaviour is engine-defined | B |
| Timestamps | ISO 8601-1:2019 | Representation of instants and offsets; not which instant a field means | B |
| Delimited files | RFC 4180 | Quoting, escaping, and line endings for comma-separated values | B |
| Schema evolution | Apache Avro specification, schema resolution rules | How a reader schema is matched to a writer schema | B |
| Compatibility policy | Confluent Schema Registry documentation | BACKWARD, FORWARD, FULL, and transitive variants; vendor-defined | D |
| Change capture | Debezium documentation, stable release | Log-based capture, initial and incremental snapshots, delete capture | D |
| Source connector delivery | Apache Kafka Improvement Proposal KIP-618 | Exactly-once for source connectors, available since Kafka 3.3 | B |
| Provenance | W3C PROV-DM (2013) | Entities, activities, and agents in a derivation chain | B |

Three of these are more limited than they look. SQL's `MERGE` defines what happens when a target
row matches a source row; it does not define what happens when several source rows match one
target row, which is the case that matters. Avro defines schema resolution precisely, but only
for data written with a schema identifier; a CSV has none. KIP-618 defines exactly-once for a
source connector's writes to Kafka and the offsets topic in one transaction; it says nothing
about what happens when that Kafka topic is consumed into a warehouse, which is where most
pipelines actually double-load.

### 8.2 What is not standardised

There is no standard for any of the following, and every vendor has invented its own answer.

- The meaning of a watermark, and whether it is inclusive or exclusive.
- Whether a snapshot is taken inside or outside the change stream, and how the overlap is
  deduplicated.
- Whether a delete is represented as a record, a flag, a tombstone message, or nothing.
- How a partial delivery is signalled.
- What "completeness" means for a batch.

The practical consequence is that **ingestion semantics live in the contract, not in the tool.**
Two teams using the same connector can have different guarantees, because they configured
different snapshot modes, offset handling, and sink behaviour. A tool name is not a specification.

### 8.3 Regulatory interfaces that constrain ingestion

| Requirement | Jurisdiction | Effect on ingestion | Grade |
|---|---|---|---|
| General Data Protection Regulation (GDPR) Article 17, erasure | European Union | A raw layer that retains personal data must support deletion, which conflicts with append-only | B |
| GDPR Article 5, data minimisation | European Union | Ingest only the fields needed; do not copy an entire source table "just in case" | B |
| Financial record retention, typically 5 to 10 years | Varies by jurisdiction | Transaction and receipt records must be retained and be retrievable | B |
| Product traceability retention | Sector-specific, for example medicinal products | Lot and event records retained for defined periods; implies ingestion completeness is auditable | B |
| Electronic invoicing and real-time reporting mandates | Italy, Spain, France, others | Invoice data may arrive from a tax authority channel as well as from the ERP; the two must reconcile | B |

The erasure conflict is the one that bites an ingestion team. The workable pattern is 03.01's:
keep the event with a pseudonymous key, keep identifiable attributes in a separate restricted
table with its own deletion process, and let a deletion write a tombstone against the key.
Copying the whole source row into an immutable raw layer makes erasure unlawful or destructive.

### 8.4 What is convention rather than requirement

- A nightly batch is a habit. Most decisions do not need sub-hour data.
- "Exactly once" is a marketing phrase until qualified by a failure model.
- A six-hour log retention is an infrastructure default, not a data-protection choice.
- A 24-hour overlap is right for daily cycles and wrong for a weekly approval step.
- Storing raw payloads forever is sound until the payload holds personal data.

## 9. Industry practice and case evidence

The evidence base for ingestion is unusual: the mechanisms are well documented by the projects
that build them, the failure modes are extensively described by practitioners, and there is
almost no published measurement connecting ingestion defects to business outcomes. That last gap
is why section 6 is a constructed case rather than a citation.

### 9.1 What the CDC vendors claim

Debezium's feature documentation describes log-based capture as reading each database's native
change mechanism, and claims four benefits: that "all data changes are captured," that delay is
very low (milliseconds for MySQL and PostgreSQL), that it requires "no changes to your data
model," and that it "can capture deletes." It also documents optional initial snapshots, taken
when a connector starts and the relevant logs no longer exist, and incremental snapshots that can
be triggered at connector runtime.

Those claims are accurate about the mechanism and incomplete about the envelope. The page does
not say what happens when the log is deleted before the connector reads it, which is the failure
in section 6.2. "No changes to your data model" is true of the schema and false of the
privileges: the connector needs replication access many managed services restrict. (Grade D,
consulted November 2026.)

The design document behind incremental snapshots is more useful than the marketing: the snapshot
is chunked, and low and high watermarks are written into the change stream so events captured
during the snapshot can be deduplicated against it. That is section 5.6's overlap problem solved
inside the connector, and worth reading for the pattern even if you never use the product.
Grade C.

### 9.2 What is actually specified for delivery

KIP-618, accepted for Apache Kafka, defines exactly-once support for source connectors. Before
it, a connector wrote records to Kafka and periodically flushed source offsets, so a crash
between the two produced duplicates on restart: at-least-once with a duplicate window. With it
enabled, each source task uses a transactional producer and writes the records and the offsets
in the same transaction, so offsets commit if and only if the records do.

The proposal is precise about its limits, and they are the interesting part. It requires the
connector to assign each source partition to at most one task at a time, and to use the
source-offsets API accurately. It is not supported in standalone mode, cannot be enabled for one
connector in isolation, and a single non-participating worker breaks the cluster-level
guarantee. Aborted transactions still commit offsets unless the task throws, so records are not
reprocessed after an abort.

This is a good model for reading any exactly-once claim. Ask three questions: what is the
transaction boundary, what is inside it, and what happens outside it. For KIP-618 the answers are
"a poll batch," "the records and the offsets," and "everything downstream of the Kafka topic,"
which is where most supply chain pipelines live. Grade B for the specification, with the caveat
that it is a specification for one project, not a general law.

### 9.3 Trigger versus log: the overhead numbers

The comparative literature is dominated by vendor and practitioner sources, and the numbers vary
by an order of magnitude. The consistent claims are directional, not precise.

| Claim | Reported range | Source type | Grade |
|---|---|---|---|
| Trigger write-latency overhead | 10–30% | Practitioner comparison articles | E |
| Per-row trigger cost | 2–15 microseconds (PostgreSQL) | Practitioner benchmark | E |
| Trigger throughput ceiling per node | 2,000–5,000 writes per second | Practitioner comparison | E |
| Log-based CPU overhead on source | Rarely above 10% | Vendor troubleshooting documentation | D |
| Log-based throughput | 50,000 or more writes per second | Practitioner comparison | E |
| Supplemental logging increases log generation | 20% or more | Vendor documentation | D |

Read the table as a shape rather than a measurement. Triggers cost write latency because they run
inside the transaction; logs cost storage and attention because they are read asynchronously. The
structural claims are more reliable than the numbers: triggers miss `TRUNCATE` and break on
data-definition changes; logs depend on retention and slot management, and a stalled slot can fill
a disk.

### 9.4 Late-arriving facts in dimensional practice

Kimball's dimensional modelling guidance is the clearest published treatment of the two arrival
problems, and it predates the tooling by two decades.

A late-arriving fact is one whose dimensional context has changed since the event; the correct key
is the dimension version current at event time, not now. A late-arriving dimension is the inverse:
a fact references a member that does not yet exist, and the remedy is an inferred member, a
placeholder row carrying the natural key and default attributes, later overwritten. Kimball
recommends inferred members over destructively rewriting fact keys, and a separate default member
per condition rather than one generic unknown.

The ingestion consequence is specific. A pipeline that assigns dimension keys at load time, using
whatever is current, cannot represent a late fact correctly however complete its extraction is.
The defect is in the join, not the feed, and it produces a variance no reconciliation will find,
because every row is present. Grade C: canonical practice, not measured evidence.

### 9.5 Deletes in practice

Vendor documentation converges on the same design, a useful signal in a field with few standards.
Stitch documents soft deletes as a `deleted_on` flag and log-based replication as a tombstone
column, and states that hard deletes are unsupported with key-based replication. Sling's
`delete_missing` option offers hard and soft modes, at the cost of a full primary-key scan.

The pattern to take is the direction of the default: **soft deletion in the warehouse, physical
deletion only after a retention period, and never a physical delete in a layer used for replay.**
A separate practitioner guide on warehouse deletes makes the same recommendation and adds the
warning that comparing a restored or archived source against the live one produces false
deletions, which is the false-tombstone problem of section 5.5 in a different costume. Grade D.

### 9.6 What is not measured

Three things are missing from the public evidence.

- **No published rate of ingestion-caused inventory variance.** The EUR 412,000 in section 6 is
  constructed. Companies measure shrinkage and uptime; almost none measure the variance
  attributable to ingestion, because attribution needs the counterfactual in section 6.6.
- **No controlled comparison of CDC strategies on total cost.** The numbers in section 9.3
  measure source overhead, not the cost of the failures each strategy permits.
- **No accepted benchmark for idempotency.** There is no standard test suite a pipeline can pass
  to be called idempotent, which is why the property is asserted more often than demonstrated.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Watermark advanced before the write | Reruns show fewer rows than the first run | Mark committed, then crash | Permanent gap, invisible after restart | Verify target write, then advance, in one transaction |
| Event-time watermark on a commit-time process | Backdated records vanish without a count change | Backdated rows fall below the mark | Missing receipts or issues, understated demand | Watermark on commit time; overlap sized to event-time lateness |
| `>` instead of `>=` on the boundary | Rows sharing a timestamp appear intermittently | Boundary row excluded | Lost updates at batch edges | Inclusive comparison plus idempotent merge |
| Partial file read | Row count far below the median; job says success | File visible before complete | Positions carried forward, silent | Atomic rename, manifest, per-file median check |
| Completeness test is `row_count > 0` | Every file passes, including a truncated one | Test has no comparator | Truncation is undetectable | Compare to manifest or to a rolling median |
| Deletes never propagated | Discontinued items remain orderable | No delete event and no key comparison | Book inventory too high, phantom stock | Tombstone policy plus periodic key-set comparison |
| False tombstones from a partial key extract | Items disappear, then return | Incomplete comparison set | Stock-outs for stock that exists | Never tombstone from a partial extract; require a complete set |
| Non-idempotent retry | Row counts rise after every incident | Append without merge or partition replace | Duplicated receipts, inflated demand | Merge on the change key or replace the partition |
| MERGE with duplicate source rows | Engine error, or arbitrary values after a rerun | No dedupe before the match | Non-reproducible current state | Dedupe to one row per key with a deterministic order |
| Replay overwrites newer data | July values revert to January after a backfill | Unconditional update in the merge | Silent regression, correct row counts | Guard on the change key |
| Schema drift absorbed silently | A column is null for a week | Positional read or renamed field | Missing or misassigned values | Registry, value profiling, quarantine on unknown column |
| Column reorder in a headerless file | Values are plausible but wrong | Same shape, different meaning | Quantities swapped with prices | Header validation, per-column plausibility rules |
| Log retention shorter than the outage | Connector fails after maintenance | Archived logs deleted | Whole-window loss | Retention at least twice the longest planned outage |
| Restart treated as recovery | Incident closed when the job turns green | No reconciliation after restart | Gap persists for months | Post-restart key-count reconciliation, mandatory |
| Time zone assumed rather than read | Daily totals shift by an hour twice a year | Local timestamps without offset | Duplicated or missing hour | Store UTC plus the source offset |
| Carry-forward hides missing rows | A warehouse position never changes | Mart fills absent rows from prior period | Stale value looks valid | Flag carried-forward rows; alert on their count |

Three of these deserve a note.

**Restart treated as recovery.** This is the most expensive failure in the table because it is a
process failure, not a technical one. Every monitoring system reports the job as healthy. The
only defence is a reconciliation that compares the target to the source after any restart, and a
runbook that says so in the first paragraph rather than the appendix.

**Carry-forward hides missing rows.** Carrying the previous value forward is often the right
business rule for a sparse snapshot. It becomes a defect when the mart cannot distinguish "the
source said this value" from "we reused yesterday's value." Add a `carried_forward` flag, publish
the count daily, and alert when it exceeds a threshold. A stale value that is visibly stale is
an operational fact; a stale value that looks fresh is a data defect.

**Column reorder in a headerless file.** No schema check catches this, because the schema is
identical. Value profiling is the only defence: a column whose distribution changes shape
overnight, or whose values violate a plausibility rule, is a drifted column regardless of its
name.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The source has a definable row identity | Idempotency requires a key | Tables without a natural key, or with reuse of keys | Attempt a merge on the key; count collisions |
| Commit order is observable | Watermarks and deduplication depend on it | Distrusted clocks, multi-node writes, batch jobs | Compare commit time to log position on a sample |
| Change volume is small relative to table size | Makes incremental cheaper than full extract | Bulk updates that touch most rows | Measure changed rows per day against total rows |
| Log retention exceeds the longest outage | Log CDC depends on it | Maintenance windows, disasters, cost cuts | Compare retention to the maximum planned pause |
| The producer can be changed to signal completion | Atomic rename or manifest requires it | Third parties under no operational control | Ask what the contract permits; if nothing, use stability checks |
| Deletes are rare enough to reconcile periodically | Key-set comparison is expensive | High-delete tables, for example session or cart data | Measure deletes per day against rows per day |
| Lateness is stationary enough to size a window | Overlap windows assume a stable distribution | Seasonal approvals, month-end batching, new partners | Plot lateness by week; look for level shifts |
| Schema changes are infrequent and announced | Registry and contracts assume this | Vendors that ship changes without notice | Count unannounced changes per quarter |
| The warehouse engine's merge is deterministic | Idempotency depends on it | Duplicate source keys; engine-specific behaviour | Run the same batch twice; compare row hashes |
| Reconciliation counts are available from the source | The only defence against missed deletes | Sources that cannot report a key count | Ask for it; treat absence as a known risk |

Two limits are worth stating. First, the topic assumes a source system that can be observed at
all. For a partner that delivers a spreadsheet by email, none of the CDC material applies, and the
only viable controls are a manifest, a schema check, and a reconciliation. Second, the arithmetic
in section 6 assumes that the cost of an ingestion defect is bounded by the value of the affected
positions. In practice the cost includes the investigation, the loss of trust in the inventory
number, and the operational workarounds that persist after the fix. Those are larger and are not
modelled here.

A third limit is specific to the evidence: the mechanisms here are well established and the
magnitudes are not. Any team quoting a percentage for "ingestion-caused variance" is quoting a
number it constructed.

## 12. Exercises

**Exercise 1 — Extraction strategy audit (analysis, 45 minutes).** Take five source tables from a
system you know, or from the Halden estate in section 6. For each, record the change rate, whether
deletes occur, whether a reliable last-modified column exists, and whether log access is possible.

**Deliverable:** a five-row table with the four attributes and a recommended strategy per table.
**Acceptance criteria:** each recommendation cites the attribute that decided it; at least one
table is recommended for full extract with a stated reason; the cost driver for each is named.
**Time:** 45 minutes.

**Exercise 2 — Design the idempotency key (design, 40 minutes).** For an order-line feed that
receives inserts, updates, and cancellations, define the entity key, the change key, and the
deduplication ordering.

**Deliverable:** a one-page key specification with column names, types, and the deduplication
window function.
**Acceptance criteria:** the two keys are distinct and each is justified; the ordering is total
and deterministic; the specification states what happens when a cancellation arrives before the
insert it cancels.
**Time:** 40 minutes.

**Exercise 3 — Size a lookback window (calculation, 50 minutes).** A receipts feed carries 4,100
lines per day. Measured lateness of `posted_at` behind commit time: p50 four minutes, p95 two
hours, p99 nine hours, p99.9 thirty-one hours, maximum fifty-two hours. The current window is two
hours and processing runs at 1,200 rows per second.

**Deliverable:** the window you would set, the rows missed per day at the current setting, and
the extra processing cost of your recommended setting.
**Acceptance criteria:** the percentile choice is justified against a named decision; the missed
rows are computed, not estimated; the cost is expressed in seconds and rows, not adjectives.
**Time:** 50 minutes.

**Exercise 4 — Write the tombstone policy (design, 45 minutes).** A product master has 14,200
active items. About 40 are discontinued per month. The source hard-deletes them after 90 days in
an archive process. Design the delete handling.

**Deliverable:** a policy covering detection, tombstone columns, retention, and the effect on
current-state and as-of queries.
**Acceptance criteria:** detection mechanism is specified with its cost; the tombstone preserves
the last known attributes; a false-tombstone guard is stated; the 90-day archive is used rather
than ignored.
**Time:** 45 minutes.

**Exercise 5 — Schema drift drill (analysis, 50 minutes).** A supplier's daily CSV gains a column,
loses a column, and swaps the positions of two others across three consecutive days. Design the
detection that catches each.

**Deliverable:** a detection table mapping drift type to the check that catches it, with the false
positive rate you would accept.
**Acceptance criteria:** the positional swap is caught by a value-level rule, not a schema check;
each check names what it would have missed; quarantine is distinguished from batch failure.
**Time:** 50 minutes.

**Exercise 6 — Partial-delivery control (design, 40 minutes).** A partner cannot change its SFTP
process and writes directly to the final file name. Design the strongest available control.

**Deliverable:** a control specification with the mechanism, the threshold, and the action on
breach.
**Acceptance criteria:** the control does not rely solely on a stability check; a per-file
comparator is defined; the breach action is executable without a meeting; the residual risk is
stated.
**Time:** 40 minutes.

**Exercise 7 — Replay and reconcile (calculation, 70 minutes).** The Halden ERP feed missed 752,083
change events over a 9.5-hour outage. Of these, 5.5% changed a quantity, affecting 1,847 distinct
item-location positions. A re-snapshot costs 12 milliseconds per position.

**Deliverable:** the missed-event arithmetic, the re-snapshot cost, and the retention setting you
would require.
**Acceptance criteria:** every figure has a derivation; the retention recommendation is tied to a
stated outage assumption; the difference between restart and recovery is stated in one sentence.
**Time:** 70 minutes.

**Exercise 8 — Variance attribution (analysis, 60 minutes).** Using the section 6.5 table, explain
which rows are measurements and which are allocations, and restate the variance under the
assumption that the CDC gap had been detected within 24 hours.

**Deliverable:** a two-column comparison of reported and counterfactual variance, with the method
stated for each line.
**Acceptance criteria:** the counterfactual is computed; at least one category is labelled as an
allocation; the residual category is not presented as an independent finding.
**Time:** 60 minutes.

**Exercise 9 — CDC decision memo (design, 55 minutes).** A team proposes replacing a nightly full
extract of a 40-million-row order table with log-based CDC. Write the memo.

**Deliverable:** the conditions under which the proposal is correct, the conditions under which it
is not, and the migration sequence.
**Acceptance criteria:** the memo names the retention requirement and the source privilege
requirement; it states what happens to the existing full extract during migration; it includes a
reconciliation plan for the first 30 days.
**Time:** 55 minutes.

## 13. Self-check answers

**Exercise 3 arithmetic.** The current two-hour window covers up to p95, so it misses at least 5%
of the lateness distribution. At 4,100 lines per day, 5% is 205 lines per day. Over 30 days that
is 6,150 lines. At a mean line value of EUR 38.10, that is EUR 234,315 of receipts landing outside
the period in which they were earned — on the order of the EUR 323,355 gross figure in section
6.4, which used a slightly different lateness share.

The cost of the larger window: 36 hours captures p99.9, leaving only the top 0.1% and the
52-hour maximum. Rows read per run rise from `4,100 x 2/24 = 342` to `4,100 x 36/24 = 6,150`, an
increase of 5,808 rows. At 1,200 rows per second that is 4.84 seconds of additional processing per
run. Four point eight seconds of compute against 205 lost rows per day is not a trade-off; it is
a correction. A p99 window of nine hours would read 1,538 rows, an increase of 1,196 rows or about
one second, and would still miss 41 lines per day.

The percentile should be argued from the decision: a daily replenishment order tolerates p99
because a missing line is corrected the next day, while a monthly financial close needs p99.9 or
a stated restatement policy, because the period is closed.

**Exercise 7 arithmetic.** `1,900,000 / 24 = 79,167` events per hour. `79,167 x 9.5 = 752,083`
events during the outage. `752,083 x 0.055 = 41,364` quantity changes. `41,364 / 1,847 = 22.4`
changes per affected position.

Re-snapshot cost: `1,847 x 0.012 seconds = 22.2 seconds`. The retention requirement should be at
least twice the longest planned outage to allow one failed recovery attempt: `2 x 9.5 = 19`
hours, rounded up to 24 hours. The distinction to state: a restart resumes the stream from now; a
recovery restores the state that the stream would have produced. Only the second one closes the
incident.

**Exercise 8 arithmetic.** Reported variance is EUR 412,000. The CDC gap contributed EUR 268,400.
If it had been detected within 24 hours and healed by re-snapshot, the counterfactual variance
would be `412,000 - 268,400 = EUR 143,600`, or 0.77% of book value instead of 2.22%. The 61 stale
positions would have been corrected before the quarter-end count, so their contribution goes to
zero. The other three categories are unchanged, because the partial file and the late receipts
were independent defects.

Of the four rows, only the CDC gap is a measurement: the change stream has a hole, the positions
are identifiable, and their book value is recorded. The partial-file row is an estimate against a
counterfactual position. The late-receipts row is an allocation. The shrinkage row is a residual
and therefore contains the error of the other three; presenting it as a finding is the error the
case exists to warn against.

**Rubric for Exercise 1.** A strong table names the decisive attribute for each recommendation and
distinguishes "cannot see deletes" from "does not need to see deletes." It recommends full extract
for at least one small reference table and says why the cost is acceptable. A weak table
recommends CDC for everything and justifies it with freshness.

**Rubric for Exercise 2.** A strong specification defines an entity key that survives a
cancellation and a change key that totally orders versions, and it states the cancellation-before-
insert case explicitly. A weak specification uses the order number as both keys.

**Rubric for Exercise 4.** A strong policy detects deletion by comparing against the 90-day
archive before it is purged, tombstones rather than deletes, keeps the last known cost and
description, and guards against a partial comparison producing false tombstones. A weak policy
says "the source does not delete" without checking.

**Rubric for Exercise 5.** A strong answer catches the positional swap with a plausibility rule on
each column and accepts a small false-positive rate in exchange. It distinguishes quarantine
(the row is held) from batch failure (nothing loads). A weak answer relies on the header row,
which the producer may not send.

**Rubric for Exercise 6.** A strong answer combines a per-file row-count comparator against a
28-day median with a digest supplied by the partner's scheduling tool, and states the residual
risk that a complete but wrong file remains undetectable. A weak answer proposes a longer sleep.

**Rubric for Exercise 9.** A strong memo states the retention requirement, the replication
privilege, and the migration sequence: run both paths in parallel, reconcile daily for 30 days,
then retire the extract. It notes that the full extract is the fallback during migration and must
not be removed first. A weak memo argues only about cost or latency.

## 14. Interview questions

**1. "What is the difference between a high-water mark and a committed watermark?"** A strong
answer separates the value from the guarantee, and says the common bug is advancing before the
write is verified. It names the crash window between the two. A weak answer treats the terms as
synonyms.

**2. "How would you capture deletes from a source that does not report them?"** A strong answer
offers the key-set comparison, states its cost and its false-tombstone risk, and distinguishes a
complete comparison from a partial one. A weak answer says "use CDC" without checking whether CDC
is available.

**3. "Why is a timestamp column not sufficient for incremental extraction?"** A strong answer
gives three mechanisms: bulk operations that bypass the column, backdated rows that fall below the
mark, and hard deletes that leave no trace. A weak answer says timestamps can be inaccurate.

**4. "You are told the pipeline is idempotent. What do you check?"** A strong answer asks for the
key, the deduplication rule, and a demonstration of two identical runs, then checks whether the
source can contain two rows for one key. A weak answer asks which tool is used.

**5. "What does exactly-once mean for a Kafka source connector, and where does it stop?"** A strong
answer describes a transactional producer committing records and offsets together, and states
that the guarantee ends at the topic boundary. A weak answer says duplicates are impossible.

**6. "A file arrives at 04:10 every day. What could still go wrong?"** A strong answer lists
partial visibility before close, duplicate delivery, encoding or delimiter drift, a complete but
truncated content set, and a late file that is silently skipped. A weak answer says the schedule is
reliable.

**7. "How do you size an overlap window?"** A strong answer measures the lateness distribution,
chooses a percentile against a named decision, computes the extra rows and time, and adds a
periodic full reconciliation because no window catches a row that never arrived. A weak answer
picks 24 hours because it is a day.

**8. "What is the difference between a late-arriving fact and a late-arriving dimension?"** A
strong answer separates the as-of key lookup from the inferred member, and notes that the first is
a join problem and the second is a dimension problem. A weak answer calls both "late data."

**9. "A backfill reverts three months of changes. What happened?"** A strong answer identifies an
unconditional update in the merge, and prescribes a change-key guard on the match condition. A
weak answer blames the scheduler.

**10. "How would you prove that an ingestion defect caused an inventory variance?"** A strong
answer describes a change-stream gap with start and end coordinates, a set of affected keys with
no subsequent change, and a counterfactual for the unaffected categories. It labels the residual
as a residual. A weak answer says the numbers do not match.

**11. "What does a schema registry's BACKWARD mode permit that a replayable raw layer might not
want?"** A strong answer says removing a required field is backward compatible for new readers
and unreadable for old records under a newer schema, so a replay layer needs a stricter mode or an
unparsed payload. A weak answer says backward compatible means safe.

**12. "Why do ingestion defects surface as operational problems rather than as data errors?"** A
strong answer explains that a shift in a level is indistinguishable from a business change,
that no schema or null test detects a missing row, and that the delay is one planning cycle plus
one physical count. A weak answer says monitoring was insufficient.

## 15. Key terms

- **Ingestion** — moving source records into a raw store with metadata about arrival.
- **Full extract** — reading every row at one point in time; costs scale with table size.
- **Incremental extract** — reading only rows beyond a stored boundary.
- **High-water mark** — the stored boundary value for the next extraction.
- **Committed watermark** — a mark advanced only after the target write is verified.
- **Lookback window** — a trailing slice re-read each run to catch late rows.
- **Event time / commit time / ingest time** — when it happened, when the database made it
  durable, when you received it.
- **Late-arriving row** — a record whose event time precedes an already-passed boundary.
- **CDC** — change data capture; reading inserts, updates, and deletes as events.
- **Log-based CDC** — CDC reading the database transaction log.
- **Trigger-based CDC** — CDC writing changes to a shadow table from inside the transaction.
- **Timestamp-based CDC** — polling a last-modified column; incremental extraction, not CDC.
- **Redo log / WAL / binlog** — Oracle, PostgreSQL, and MySQL transaction logs.
- **LSN / SCN** — monotonic log positions in PostgreSQL and Oracle.
- **Snapshot** — a complete copy of state at one instant; **consistent read** — a read that sees
  that instant despite concurrent writes.
- **Change envelope** — operation type plus before and after images plus source coordinates.
- **Natural key** — the business identifier; **idempotency key** — what makes two records the same.
- **Upsert / MERGE** — insert when absent, update when present.
- **Deduplication** — reducing many records per key to one by a deterministic rule.
- **Tombstone** — a record marking deletion without removing history.
- **Soft delete / hard delete** — flagged in the source / physically removed.
- **Delete propagation** — carrying source deletions into the warehouse.
- **Schema drift** — unannounced change to structure or typing.
- **Schema registry** — versioned schemas with an enforced compatibility mode.
- **Compatibility mode** — BACKWARD, FORWARD, FULL, and transitive variants.
- **File drop** — delivery as a file with no protocol beyond the file itself.
- **Manifest** — a producer's statement of counts and checksums for a delivery.
- **Atomic rename** — write to a temporary name, rename on completion.
- **Replay / backfill** — reproduce a past result / fill a gap that was never loaded.
- **Reconciliation** — comparing counts at two points to prove nothing was lost.

## 16. Sources

**Specifications and standards bodies**

- SQL:2003, `MERGE` statement; retained in SQL:2008, SQL:2011, and later editions. Defines match
  semantics; leaves duplicate-source behaviour to the implementation. (B)
- ISO 8601-1:2019, *Date and time — Representations for information interchange — Part 1: Basic
  rules*. (B)
- RFC 4180, *Common Format and MIME Type for Comma-Separated Values (CSV) Files*. (B)
- Apache Avro specification, schema resolution and schema evolution rules. (B)
- Apache Kafka Improvement Proposal KIP-618, "Exactly-Once Support for Source Connectors."
  https://cwiki.apache.org/confluence/spaces/KAFKA/pages/153816406/KIP-618+Exactly-Once+Support+for+Source+Connectors (B)
- W3C (2013). *PROV-DM: The PROV Data Model*, Recommendation. (B)
- Debezium design document DDD-3, "Incremental snapshotting." Watermark-based chunked snapshot
  with deduplication against the change stream. (C)

**Vendor and product documentation**

- Debezium, "Debezium Features," stable release documentation.
  https://debezium.io/documentation/reference/stable/features.html Consulted November 2026. (D)
- Debezium, "Sending signals to a Debezium connector," stable release documentation. (D)
- Debezium, PostgreSQL and MySQL connector reference, snapshot and incremental snapshot
  properties. (D)
- Confluent, "Schema Evolution and Compatibility for Schema Registry on Confluent Platform."
  https://docs.confluent.io/platform/current/schema-registry/fundamentals/schema-evolution.html
  BACKWARD default, transitive variants, Avro change table. (D)
- Stitch (Qlik), "Deleted Record Handling." Soft delete flags, tombstone column, hard-delete
  limitation with key-based replication. (D)
- Sling, "Capture Deletes." `delete_missing` hard and soft modes, primary-key scan cost. (D)
- Fivetran, "What Is the Impact of Running Log-Based CDC on My Database Server?" Source CPU
  overhead claims. (D)
- Oracle Analytics, "Deletions," 26r3 documentation. `TRACKDELETES`, soft delete flagging,
  stability period. (D)

**Practitioner and industry**

- Modern Data Engineering Guide, "Extraction patterns: full, incremental and idempotent," and
  "Idempotent SQL." Boundary comparison, overlap, dedupe-before-merge. (E)
- Praveen T N, "Incremental Overlap Window." Sizing by lateness percentile, failure modes of too
  short and too long a window. (E)
- Unstructured, "Incremental and Continuous Data Ingestion Strategies." Overlap windows sized
  from write-to-visibility delay. (E)
- LeapFrogBI, "Dealing with Deletes in the Data Warehouse." Detection, consolidation, and model
  impact; false deletions from archive comparison. (E)
- Qlik, "Managing Data Warehouse Deletes," white paper. Operation indicators and delete
  propagation patterns. (E)
- System Overflow, "Trigger Based vs Log Based CDC: Decision Framework," and "Trade-Offs:
  Log-Based CDC vs Alternative Approaches." Overhead ranges cited in section 9.3. (E)
- pg_trickle, "Triggers vs Replication." Per-row trigger cost measurements on PostgreSQL. (E)
- Coalesce, "Duplicate row detected during DML action," error reference, and Analistable, "How to
  use MERGE (upsert) in BigQuery and Snowflake." Engine behaviour with duplicate source rows. (D)

**Canonical modelling**

- Kimball, R. (2004). "Design Tip #57: Early Arriving Facts." Inferred members and the case
  against destructively rewriting fact keys. (C)
- Becker, B. (2010). "Design Tip #128: Selecting Default Values for Nulls." Distinct default
  members per condition. (C)
- Kimball Group, *Kimball Dimensional Modeling Techniques*. Late-arriving facts and as-of
  dimension lookups. (C)

**Note on grading and gaps.** Grades follow Module 28: A peer-reviewed or audited, B regulatory,
specification, or official statistic, C framework or documented design, D vendor documentation,
E practitioner essay or vendor comparison. Vendor pages are dated at the version consulted;
figures are as the sources state them and are not independently verified. The overhead ranges in
section 9.3 are practitioner measurements on unspecified hardware and should be read as
directional. No published measurement of ingestion-caused inventory variance was found, and
section 6 is constructed to be internally consistent rather than empirical.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`02_batch-microbatch-streaming.md`](02_batch-microbatch-streaming.md)
- Next topic (planned): [`04_sql-supply-chain-patterns.md`](04_sql-supply-chain-patterns.md)
- Later topic (planned): [`11_quality-quarantine-lineage.md`](11_quality-quarantine-lineage.md)
- Prerequisite: [`01_layered-architecture.md`](01_layered-architecture.md)
- Transport and arrival envelopes: [`../02_supply_chain_data_landscape/06_data-movement.md`](../02_supply_chain_data_landscape/06_data-movement.md)
- Survey file: [`../../../course/03_data_engineering_interoperability.md`](../../../course/03_data_engineering_interoperability.md)
- Return to the part map: [`../_index.md`](../_index.md)
