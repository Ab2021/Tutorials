> Module 03 — Data Engineering and Interoperability
> Topic 11 of 11
> Estimated study time: 3.5 hours
> Prerequisites: 03.01, 03.06, 03.07

# Quality Tests, Quarantine, Versioning, and Lineage

---

## 1. Why this topic exists

Kestrel Health Distribution runs a daily replenishment plan for 6,240 item-locations across
three distribution centres (DCs). The plan is published at 05:30 and consumed by buyers from
06:00. In the first quarter of 2026 the pipeline was green on 61 of 63 trading nights. On the
other two it was red, and both failures were fixed within the hour.

During the same quarter, the plan understated projected supply for a high-volume molecule by an
average of 4.7% of units. Nine emergency purchases were raised at a premium of $96,820 in total.
The cause was a data test that had been set to `warn` fourteen months earlier, during a supplier
master migration, and never reset. It fired on 63 consecutive nights. Each firing wrote a line
to a log that no one read. The build exited zero. The plan published.

Three artifacts would have prevented this, and this topic is about all three.

1. **A severity policy that distinguishes a warning from a gate.** A warning records a fact. A
   gate withholds a decision. The two must be different objects with different owners, not two
   values of one configuration flag.
2. **A quarantine table that keeps the excluded rows and prices their exclusion.** A pipeline
   that drops bad rows and continues is not resilient; it is silently partial. The question is
   not whether to proceed with a partial population but whether the partial population has been
   measured, declared, and accepted by the person who owns the decision.
3. **A version tuple and a lineage index that make any past plan re-derivable.** A plan is a
   claim about the world made from named inputs. If those inputs cannot be named, the claim
   cannot be defended, audited, or reproduced.

The fourth thing this topic supplies is a test of the first three: **can you regenerate last week's
plan exactly, from the recorded inputs, and explain every difference?** That replay test is the
only check that exercises architecture, ingestion, transformation, orchestration, and
documentation at once. Most pipelines have never run it.

**Why this is the closing topic.** Each earlier topic removes one source of ambient state: layers
and contracts (03.01), cadence (03.02), arrival and redo (03.03), temporal joins (03.04), code
testability (03.05), executable assertions (03.06), the cutoff and run identity (03.07), the
document (03.08), the identifier (03.09), and the entity and schema promise (03.10). This topic
names what remains: the unread warning, the dropped row, the unpinned rate, the unknown consumer.

The module's argument, stated once: **a planning number is reproducible when it is a function of
recorded inputs, recorded code, and recorded reference data, and when every excluded row and
every downstream consumer can be enumerated.** Everything else in the module is machinery for
making one of those clauses true.

**Scope.** This is not a tour of observability vendors. It assumes 03.01 (layers, contracts,
replay), 03.06 (grain, tests, severity), and 03.07 (cutoff, run identity, backfill), and
concentrates on whether a failed check changes what the organisation does.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** an advisory warning from a blocking gate in terms of exit status, downstream
   execution, owner, and evidence required, and **convert** a named warning into a gate with a
   stated population contract.
2. **Design** a quarantine table with a deterministic key, a reason envelope, a status lifecycle,
   and a reconciliation that proves input equals accepted plus quarantined plus rejected.
3. **Quantify** the difference between a row-weighted and a unit-weighted quarantine rate, and
   **decide** whether a run may proceed on a partial population.
4. **Name** at least six mechanisms by which a pipeline fails open by default, and **locate** each
   in a pipeline you can inspect.
5. **Define** a version tuple for a planning run — inputs, code, reference data, parameters, and
   run identity — and **write** the run record that carries it.
6. **Specify** a replay class and a tolerance, run a replay test, and **classify** every
   difference as explained or unexplained.
7. **Use** column-level lineage to answer "which model used the field that changed?", and
   **perform** an impact analysis before a schema change with a stated blast radius and
   deprecation window.
8. **State** the module's reproducibility argument and **map** each of its eleven topics to the
   source of ambient state it removes.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold one picture: a run is an assertion with a receipt, and every failed check is routed
somewhere. The routing is the design. Most pipelines route every failure to the same place — a
log line — which is equivalent to routing none of them anywhere.

```text
  RUN 2026-11-02 02:15   interval [11-01, 11-02)   code 7f3c1a9   refdata fx=2026-10-31
        |
        v
  sources ---> conform ---> tests ---+---> GATE ---pass---> publish plan v2026-11-02.1
                                     |       |                        |
                                     |       | fail                   +--> run record
                                     |       v                             (inputs, code,
                                     |   QUARANTINE                         refdata, params)
                                     |   rows + reason + key
                                     |       |
                                     |       +--> population contract --> proceed (declared)
                                     |                              \---> halt (prior plan,
                                     |                                     flagged stale)
                                     v
                              warning log  (no effect on the run)

  REPLAY of plan v2026-11-02.1:
     same manifest + same code + same reference data --> compare --> differences
     every difference either explained by a recorded change or the replay fails
```

Read the picture with six properties in mind.

**Property 1 — a run is an assertion with a receipt.** The published plan is the assertion; the
run record is the receipt. A plan without a receipt is a number whose provenance is a memory.

**Property 2 — severity is a routing decision, not a label.** `warn` routes a failure to a log.
`error` routes it to the gate. Quarantine routes individual rows to a table and the decision to
the population contract. These are three different destinations, and the choice between them is
a statement about what the organisation is willing to do when the check fires.

**Property 3 — quarantine is a ledger, not a bin.** A bin is where rows go to be forgotten. A
ledger has a key, a reason, a status, an owner, a reconciliation, and a path back. If quarantined
rows cannot be counted against the input, the pipeline does not know what it excluded and cannot
say how partial its output is.

**Property 4 — a version is a tuple, not a timestamp.** "The 2 November run" is not a version.
The version is the tuple (input manifest, code commit, reference-data vintage, parameters, run
identifier). Two runs with the same timestamp and different tuples are different plans.

**Property 5 — lineage is the index into the version tuple.** Versioning tells you what a run
consumed. Lineage tells you which other artifacts consumed the same thing, and therefore what
else must be replayed when it changes. Without lineage, a version is a library with no catalogue.

**Property 6 — reproducibility is a test, not a claim.** A pipeline is reproducible when a replay
produces the same result or an explained difference. Until that has been executed and recorded,
"reproducible" is a design intention.

### 3.1 The four responses to a failed check

| Response | Effect on the run | Who decides | Evidence required before it is allowed | Typical check |
|---|---|---|---|---|
| Advisory (`warn`) | Continues; writes to a warning log and a ticket queue | Data owner | A triage owner and a review cadence | Distribution drift, a soft threshold |
| Quarantine (row) | Continues with a declared partial population | Decision owner | Unit-weighted impact below the contract limit; reason recorded per row | Unit of measure, domain, referential |
| Gate (`error`) | Halts downstream; no plan published | Decision owner | The population contract itself | Key uniqueness, coverage, reconciliation |
| Halt and withdraw | Stops the run and withdraws the published plan | Incident commander | Evidence that the published plan is wrong | Post-publication reconciliation |

The first row is where most controls live and where almost none of them act. A warning with no
named triage owner is a message to nobody. The second row is the one most organisations lack
entirely, which forces every row-level defect into either a warning or a halt.

### 3.2 The fail-open default

Pipelines fail open because their components were designed for availability, not for planning.
Six mechanisms do most of the work, and each is a default rather than a decision.

1. **Parsing.** The loader reads what it can, writes the rest to an error log, and reports
   success on the rows it read. The count of unread rows is a log line, not a partition.
2. **Inner joins.** An orphan fact row disappears. The query is correct; the population is
   smaller. Nothing in the result says a row was expected.
3. **`left join` plus `coalesce(..., 0)`.** Absence becomes a zero, and zero is a valid quantity.
   A missing receipt and a receipt of nothing are indistinguishable downstream.
4. **Aggregation over an empty set.** `sum()` over no rows returns zero or null, both of which
   render as a plan. A pipeline with no input produces a complete-looking empty plan.
5. **Severity.** A test set to `warn` reports and continues; the run's exit status is unchanged.
   A migration is the usual occasion, and the setting is rarely reverted.
6. **Orchestration success.** A task that writes zero rows exits successfully unless something
   asserts otherwise. "The task completed" and "the data arrived" are different claims.

Two more appear once quarantine exists: an **unreconciled quarantine**, where accepted plus
quarantined does not equal input and nobody notices the difference, and a **wall-clock filter**,
where a replay silently returns a different population because the query read `current_date`.

> **Note:** Failing open is not always wrong. For a telemetry feed used in an exploratory
> dashboard, proceeding with 97% of rows is usually better than stopping. The error is
> inheriting that posture for a feed that sets a purchase order. The response level must be
> chosen from the consequence of the decision, not from the convenience of the pipeline.

### 3.3 Three clocks and one tuple

Every row in a planning system carries three times, and conflating them is the root of most
replay failures.

| Clock | Meaning | Example | Consequence if confused |
|---|---|---|---|
| Event time | When the event happened in the world | Goods received at 14:20 on 1 November | Population shifts when the run date changes |
| Ingestion time | When the row arrived in the platform | File loaded at 02:10 on 2 November | Late data looks like new data |
| Knowledge time | When the organisation knew the fact | Correction posted on 6 November | A replay uses knowledge that did not exist |

A plan is a function of what was known at the cutoff. That is a statement about **knowledge
time**, not event time and not ingestion time. A bitemporal table records event time and
knowledge time; a versioned snapshot records the whole table as of a knowledge instant. Either
can support replay. A table with only a `loaded_at` column supports neither, because a
correction overwrites the row and the prior belief is gone.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Data quality test | A query that returns rows violating a predicate; zero rows means pass | Called a unit test | Rows returned |
| Severity | The routing level assigned to a failed test | Treated as a priority label | Enum: advisory, gate, halt |
| Warning (`warn`) | A failure that is recorded and does not stop the run | Assumed to stop publication | Enum value |
| Gate (`error`) | A failure that halts downstream execution and withholds the plan | Set on a test nobody can act on | Enum value |
| Quarantine | Isolation of individual rows that fail validation, with the run continuing on the accepted set | Used to mean "delete bad rows" | Rows |
| Dead-letter queue (DLQ) | A durable store for messages a consumer could not process, with a replay path | Used as a dumping ground | Messages |
| Fail-open | Continuing when a check fails, producing a partial or degraded output | Described as resilience | Behaviour |
| Fail-closed | Stopping when a check fails, producing no output | Described as brittleness | Behaviour |
| Partial population | The subset of expected entities present in a run's input | Reported as a percentage without a denominator | Set of keys |
| Population contract | A pre-agreed statement of minimum coverage, maximum unit impact, decision owner, and fallback | Replaced by a threshold in a config file | Document |
| Completeness | The fraction of expected rows or partitions that arrived | Confused with freshness | Proportion |
| Reconciliation | Proof that input equals accepted plus quarantined plus rejected | Replaced by a row-count print | Identity |
| Tolerance | The stated admissible difference before a check fails | Left implicit | Units or proportion |
| Quarantine rate | Quarantined rows divided by input rows | Used as a proxy for business impact | Proportion |
| Unit-weighted impact | Quarantined units divided by total units | Omitted because it costs a join | Proportion |
| Write-audit-publish (WAP) | Write to an isolated location, audit, then atomically publish | Implemented without the atomic step | Pattern |
| Snapshot | An immutable record of a dataset at a point in time | Used to mean a current-state copy | Dataset version |
| Data version | A named, immutable state of a dataset | Equated with a file timestamp | Commit, snapshot, or tag |
| Code version | The exact transformation logic used in a run | Recorded as "the current branch" | Commit hash or model version |
| Reference data vintage | The as-of version of calendars, rates, unit factors, and masters | Read as current state | Version identifier |
| Event time | When a fact occurred | Confused with load time | ISO 8601 instant |
| Ingestion time | When a fact arrived in the platform | Confused with event time | ISO 8601 instant |
| Knowledge time | When the organisation first knew a fact | Absent from most schemas | ISO 8601 instant |
| Replay class | The required degree of agreement between a replay and the original | Unstated | Enum: R0 to R4 |
| Manifest | The recorded list of input artifacts and their hashes or versions | Replaced by a folder path | Document |
| Run record | The stored tuple of interval, code, inputs, reference data, and outcome | Replaced by a log line | Row |
| Plan version | The immutable identity of a published plan | Overwritten in a current-state table | Identifier |
| Lineage | The graph of datasets and jobs, and the fields that flow between them | Assumed to be the orchestration DAG | Graph |
| Column-level lineage | Mapping from each output field to the input fields that determine it | Approximated by table-level edges | Graph |
| Impact analysis | Enumerating the artifacts affected by a proposed change | Done after the change | List |
| Breaking change | A change that alters meaning, type, or population for a consumer | Detected only when a report breaks | Change class |
| Retention period | The minimum time records must be kept | Confused with backup retention | Duration |
| Point-in-time reconstruction | Rebuilding the exact state and output of a system as of a past instant | Attempted from current-state tables | Procedure |

Three of these repay expansion.

**Quarantine versus dead-letter queue.** A DLQ originates in messaging: a consumer cannot process
a message, so it parks it and moves on. Quarantine originates in data quality: a row is valid
except for a stated defect, so it is set aside while the batch continues. The mechanics are
similar and the governance is not. A DLQ is usually owned by the consumer's operator; a
quarantine table is owned by the decision owner, because the decision to proceed without those
rows is theirs.

**Snapshot versus immutable log.** A snapshot stores state; a log stores changes. Snapshots are
cheap to read and expensive to store; logs are cheap to store and require computation to read.
Most table Formats provide both: a snapshot list plus the files that compose each snapshot. The
practical difference for replay is that a snapshot can reconstruct the table at an instant
without replaying every change, while a log can reconstruct intermediate states that no snapshot
captured.

**Determinism versus idempotency.** Determinism is about output given input. Idempotency is about
state given repetition. A nondeterministic job that writes to a fresh partition is idempotent in
effect; a deterministic job that appends without a key is not. Replay requires both.

## 5. The mechanics, step by step

The sequence below runs from the meaning of a failure to the proof that the run can be repeated.
It is ordered by dependency: a severity policy cannot be written before the consequence is known,
a quarantine cannot be reconciled before the key is defined, and a replay cannot be tested before
the version tuple exists.

### Phase A — Decide what a failure means

**Step 1. Write the invariant as a predicate with a grain and a tolerance.**

"The share of open purchase order lines with a null supplier must be zero" is testable. "Supplier
data must be good" is not. State the grain (line), the predicate (`supplier_id is not null`), and
the tolerance (zero rows). Where the quantity is a measure rather than a key, state a tolerance
in units and a reason for it.

*If skipped:* the test is written against whatever column is convenient, usually a surrogate key
that is non-null by construction.

**Step 2. Classify the consequence, not the data.**

Ask what decision consumes the field and what happens if it is wrong. A null supplier on a
purchase order line removes projected supply; a null `preferred_vendor` flag changes a
tie-break; a null marketing description changes nothing. The same null has three different
severities because it has three different consequences.

| Consequence | Example | Response |
|---|---|---|
| Changes a quantity in a published plan | Null supplier on an open PO | Gate |
| Changes a ranking or a tie-break | Missing vendor preference | Quarantine plus contract |
| Changes presentation only | Missing long description | Advisory |
| Changes nothing measurable | Deprecated field | Delete the test |

*If skipped:* every nullable column gets the same severity, which produces either a brittle
pipeline or a noisy one.

**Step 3. Choose the response level from the consequence.**

Use the four-level ladder in section 3.1. Record the choice in the model's documentation with an
owner and a review date. A severity without an owner is a setting; a severity with an owner is a
policy.

*If skipped:* `warn` becomes the default because it is the only setting that does not cause an
argument, and the pipeline fails open one column at a time.

**Step 4. Write the population contract before the first failure.**

The contract is a short document, not a threshold. It states:

- the expected key set and where it is defined;
- the minimum coverage by segment, where a segment is a DC, a business unit, or a molecule
  family — the sets that must all be present for the plan to mean anything;
- the maximum unit-weighted impact of quarantine, not the maximum row share;
- the decision owner and the fallback: proceed with a declared defect flag, or halt and use the
  prior plan with a stale flag;
- the review date.

*If skipped:* the decision is made at 05:20 by whoever is on call, using the only number on the
screen, which is a row percentage.

### Phase B — Quarantine without lying

**Step 5. Route rows, not batches, when the failure is row-local.**

A unit-of-measure mismatch on one line does not invalidate the other 3,488 lines in the file.
Batch-level rejection is appropriate when the defect is in the file itself — a truncated
transmission, a wrong header count, a schema that does not match the contract. The distinction is
whether the defect can be attributed to a key.

**Step 6. Give every quarantined row a reason envelope and a deterministic key.**

The envelope carries the source, the rule that failed, the observed value, the expected domain,
the load identifier, and the event time. The key must be deterministic — a hash of the natural
key and the rule, not a sequence — so that reprocessing the same bad row twice updates one
quarantine record rather than creating two.

| Field | Purpose | Example |
|---|---|---|
| `quarantine_key` | Deterministic identity for deduplication | `sha256(po_line_id + rule_id)` |
| `source_dataset` | Where it came from | `raw.edi_856_lines` |
| `rule_id` | Which invariant failed | `uom_in_domain` |
| `failed_value` | What was observed | `CS` |
| `expected_domain` | What was allowed | `EA, CA, PK` |
| `load_id` | Which run saw it | `2026-11-02T02:15Z` |
| `event_time` | When the underlying event occurred | `2026-11-01T18:40Z` |
| `status` | Lifecycle position | `quarantined`, `fixed`, `rejected` |

**Step 7. Reconcile input to output, every run.**

The identity is `input rows = accepted rows + quarantined rows + rejected rows`, evaluated per
source and per load. Without it, quarantine becomes a place where rows disappear and the
disappearance is indistinguishable from a parse failure. Store the three counts in the run
record; a difference is itself a gate failure.

*If skipped:* the quarantine table grows, the accepted table grows, and no one can say whether
the missing rows were bad or never arrived.

**Step 8. Measure impact in units, not rows.**

Join the quarantined rows to their quantities and compute the share of units, then the share of
the decision variable. Section 6.3 shows a case where 10.6% of rows carried 78.0% of units. A row
threshold of 15% passed and the plan lost four-fifths of its inbound supply for one item.

> **Note:** The unit-weighted rate is not a refinement. Row counts are valid only when rows are
> homogeneous, which in supply chain data they almost never are. A single line for a
> high-volume molecule can outweigh a thousand lines for slow movers.

**Step 9. Declare proceed or halt against the contract, and record which.**

The run record carries one of three outcomes: `published_complete`, `published_partial` with the
excluded key set and the impact, or `withheld`. "Published" alone is not an outcome, because it
does not say what population was published.

**Step 10. Close the loop: repair, recycle, expire.**

A quarantine row has three exits. **Repair** corrects the source and re-ingests; the quarantine
record moves to `fixed` and the row enters the accepted set on the next run. **Recycle** revalidates
parked rows against a corrected rule and promotes the survivors. **Expire** removes rows older
than the retention rule requires, with the count recorded. A quarantine table with no exits is a
second copy of the raw data with worse documentation.

### Phase C — Version and replay

**Step 11. Define the version tuple.**

| Component | What it names | Where it is recorded | Failure if missing |
|---|---|---|---|
| Input manifest | Every source partition read, with hash or snapshot identifier | Run record | Replay reads a different population |
| Code version | Commit hash, model version, or image digest | Run record | Replay runs different logic |
| Reference data vintage | Calendar, foreign exchange (FX), unit factors, item master as of the cutoff | Run record | Replay applies today's rates to last week's plan |
| Parameters | Cutoff, horizon, service targets, allocation weights | Run record | Plans are compared that answered different questions |
| Run identifier | Unique, immutable identity of the execution | Run record and plan record | Two runs cannot be distinguished in an incident |

**Step 12. Pin reference data and forbid ambient time.**

Reference data is input. Calendars, FX rates, unit factors, and the item master are as-of objects,
and reading their current state bets they have not changed since the cutoff. The rule is
mechanical: no `current_date`, `now()`, or unseeded random in logic; every such value is a run
parameter.

**Step 13. Write the run record and the plan record.**

The run record is the receipt for the execution. The plan record is the receipt for the
decision: plan identifier, data interval, cutoff, run identifier, input manifest identifier,
supersession pointer, and the population status. Write both once and never update them; a
correction is a new version with a pointer to the old one.

**Step 14. Choose a replay class and test it on a schedule.**

| Class | Requirement | Use |
|---|---|---|
| R0 | Byte-identical artifact | Regulatory reproduction of a submitted file |
| R1 | Same rows and keys, order-insensitive | Integer quantities, plan recommendations |
| R2 | Same values within a stated tolerance | Currency, rates, computed ratios |
| R3 | Every difference attributed to a recorded change | Incident review, post-change validation |
| R4 | Differences unexplained | A defect, not a class |

For planning data the default should be R1 on keys and quantities and R2 on currency. R3 is
acceptable only for a replay whose purpose is to explain a known change, and only when every
differing row carries an attribution. R4 is never acceptable and should be treated as a failed
gate.

### Phase D — Lineage and change

**Step 15. Capture lineage at the level your question needs.**

Table-level lineage answers "what breaks if this table is late". Column-level lineage answers
"which model used the field that changed". The second is the one that schema changes require, and
it is the one most pipelines lack.

| Mechanism | How it works | What it misses | Grade |
|---|---|---|---|
| Static SQL parsing | Parse each statement into input and output fields | Dynamic SQL, wildcard selects, Python transforms | B |
| Runtime instrumentation | Capture the query plan or the read set from the engine | Logic outside the engine; cost | B |
| Framework manifest | Read the transformation tool's compiled graph and tests | Anything outside the framework | B |
| Manual documentation | Maintain a mapping by hand | Decay, within weeks | E |
| None | — | Everything | — |

**Step 16. Run impact analysis before the change, not after.**

Enumerate the downstream fields to transitive closure, join them to published artifacts and their
consumers, classify each as a breaking or non-breaking change, and state the blast radius in
artifacts and in decisions. A rename is breaking even when the values are identical, because
every consumer's SQL references the old name.

**Step 17. Deprecate with a window and a consumer list.**

Publish the change, the replacement, the removal date, and the list of known consumers. Add the
old field as a view or a computed alias for the window. Remove it only when the consumer list is
empty and the window has elapsed. The lineage graph supplies the list; without it, the window is
announced to whoever happens to read the release note.

### 5.1 The module, restated as one argument

The eleven topics are not eleven skills. They are eleven removals of ambient state, and the
argument only closes when all of them hold.

| Topic | What it fixes | Ambient state it removes |
|---|---|---|
| 03.01 Layers | Boundaries and contracts | "The data is somewhere in the warehouse" |
| 03.02 Cadence | Batch, micro-batch, or stream | "Current" meaning whatever the reader assumes |
| 03.03 Ingestion | Arrival, deletes, late data | "The extract is the source of truth" |
| 03.04 SQL patterns | As-of joins, grain, allocation | "The join returns one row" |
| 03.05 Python structure | Types, tests, packaging | "It works on my machine" |
| 03.06 dbt models and tests | Executable assertions | "Someone checked it" |
| 03.07 Orchestration | Cutoff, interval, backfill | "The nightly run" |
| 03.08 EDI (electronic data interchange) | Documents and acknowledgements | "The order says what happened" |
| 03.09 GS1 and EPCIS | Identity and events | "The same product has one identifier" |
| 03.10 APIs, contracts, MDM | Schema and entity ownership | "That column means what it meant last year" |
| 03.11 This topic | Severity, quarantine, versions, lineage | "We can reproduce it if we need to" |

Read the last column downwards: every topic converts a belief into a recorded input. A plan built
on eleven recorded inputs can be re-derived; one built on ten and a belief cannot.

## 6. Worked example

Kestrel Health Distribution is introduced in section 1. It is a pharmaceutical wholesaler with
three DCs, 6,240 active item-locations, a daily plan cutoff at 05:30, and a published plan version
consumed by buyers at 06:00. All figures are illustrative but internally consistent.

### 6.1 The estate

| Artifact | Grain | Rows | Role |
|---|---|---:|---|
| `raw.edi_856_lines` | One row per advance ship notice line | 3,900 per file | Inbound supply evidence |
| `raw.erp_open_po` | One row per open purchase order line | 18,400 | Projected supply |
| `conformed.dim_item` | One row per item version | 6,240 active | Item attributes |
| `fct_inbound_supply` | One row per expected receipt | 214,000 per month | Supply fact |
| `fct_stock_position` | One row per item-location-day | 6,240 per day | Inventory fact |
| `plan_purchase_recommendation` | One row per item-DC recommendation | 1,180 per plan | Decision artifact |
| `meta.column_lineage` | One row per field-to-field edge | 18,700 edges | Impact index |

The pipeline has 96 models, 410 data tests, one gate, and two quarantine tables. Of the 410
tests, 314 are at gate severity and 96 at advisory. That ratio is itself a finding: 96 advisory
tests produce more messages than any team can triage, and the one that mattered was among them.

### 6.2 Incident A — the warning that was a gate

A supplier master migration in December 2024 changed the map that populates `supplier_id` on
purchase order lines. A `not_null` test on the field was set to `warn` for the migration and
never reset. From January 2026 the upstream map change caused 2.1% of open lines to arrive with
a null supplier.

| Quantity | Value | Derivation |
|---|---:|---|
| Open purchase order lines | 18,400 | Estate |
| Lines with null supplier | 386 | 2.1% of 18,400 |
| Open units | 2,410,000 | Estate |
| Units on null-supplier lines | 112,800 | 4.68% of open units |
| Nights the test warned | 63 | Consecutive trading nights |
| Nights an expedite was raised | 9 | Plan output |
| Units expedited | 41,200 | Plan output |
| Premium per unit | $2.35 | Contract versus spot |
| Premium paid | $96,820 | 41,200 x $2.35 |

The arithmetic that matters is the second and fourth rows. Two percent of lines carried nearly
five percent of units, because the nulls were concentrated on the high-volume molecule. A row
count made the defect look small; the unit count made it expensive.

The test did exactly what it was configured to do. It returned 386 failing rows every night. A
`warn` severity test records the failure and does not fail the run, and in a build that
interleaves tests with models it does not skip downstream models either. The plan was built,
published, and consumed. Sixty-three warnings is not a signal; it is a background level.

**The control that would have stopped it** is a gate on the plan's supply population: every open
purchase order line in the horizon has a supplier, or the plan is withheld. That gate belongs at
the boundary between the fact and the plan, not on the staging model, because it is the plan that
must not be wrong.

### 6.3 Incident B — quarantine that looked small

An inbound advance ship notice (ASN, the EDI 856 document) arrived with 3,900 lines. Of these,
412 failed a unit-of-measure (UOM) check: the line said `CS` where the item domain allowed
`EA`. The pipeline routed the 412 to a quarantine table and continued with 3,488 lines.

| Quantity | Accepted | Quarantined | Total |
|---|---:|---:|---:|
| Lines | 3,488 | 412 | 3,900 |
| Average units per line | 8 | 240 | — |
| Units | 27,904 | 98,880 | 126,784 |
| Share of lines | 89.4% | 10.6% | 100% |
| Share of units | 22.0% | 78.0% | 100% |

The quarantine gate was configured at 15% of rows. The run proceeded. It published a plan in
which 78% of the units in that file were absent, for an item whose twenty-one-day requirement was
98,700 units — almost exactly the quantity quarantined. The plan saw essentially no inbound
supply for the molecule and released substitute purchases against a three-week lead time.

Two design errors compounded. The first was denominating the threshold in rows. The second was
treating the quarantine as a technical event: the run record said `published`, with no statement
of population, so no downstream consumer could tell that the plan had been built on one-fifth of
its inbound units.

**The control that would have caught it** is the unit-weighted impact in the population contract,
computed before publication: 98,880 units against a 21-day requirement of 98,700 units is 100% of
the horizon, and no threshold above zero would have permitted publication. The fallback is the
prior plan with a stale flag and an exception list naming the affected item.

### 6.4 The partial-population decision

The decision is not "is 10.6% acceptable". It is four questions, answered in order.

1. **Which keys are missing, and are they in a critical segment?** One molecule is a critical
   segment even at 0.1% of rows.
2. **What is the unit-weighted impact on the decision variable?** Here, 100% of the horizon for
   that molecule.
3. **What does the plan do with the missing population?** It substitutes, which converts a data
   defect into a purchase commitment.
4. **What is the fallback, and who owns it?** Prior plan, stale flag, exception list, buyer
   notified before 06:00.

A contract that answers these four permits a partial publication for a non-critical segment and
forbids it here. A threshold answers none of them.

### 6.5 Incident C — the replay test

On 9 November the team replayed the plan of 2 November: same code commit, same input manifest,
same parameters. The comparison key was (item, DC), and the measure was recommended units.

| Quantity | Original | Replay | Difference |
|---|---:|---:|---:|
| Recommendations | 1,180 | 1,180 | 0 |
| Differing recommendations | — | 41 | 3.47% |
| Recommended units | 1,842,000 | 1,815,600 | 26,400 (1.43%) |
| Attributed to current-state FX | — | 38 | 24,900 units |
| Attributed to `current_date` filter | — | 3 | 1,500 units |
| Unexplained | — | 0 | 0 |

Both causes are ordinary. The FX table was read as current state rather than as of the cutoff,
so a rate published on 5 November was applied to a 2 November plan. One model filtered history
with `where order_date >= current_date - 365`, so the replay silently dropped a day at the far
edge of the window. Neither is a modelling error in the ordinary sense; both are ambient state.

The result is R3, not R1: every difference is explained, and the plan is still not reproducible
without manual intervention. The fix is to pass the FX vintage and the window start as run
parameters, after which the replay returns zero differences and the class becomes R1.

> **Note:** "96.5% reproducible" is not a meaningful claim. Reproducibility is evaluated per
> declared comparison key, and a plan either reproduces or it does not. Reporting an aggregate
> match rate hides the fact that the differing rows are usually concentrated in the decisions
> that matter.

### 6.6 Impact analysis before a schema change

The item master proposes to rename `shelf_life_days` to `shelf_life_days_at_receipt` and to
change its unit from days to months for one legacy source. The lineage query in section 7
returns the following downstream fields.

| Hop | Dataset | Field | Artifact type |
|---:|---|---|---|
| 1 | `int_item_location_enriched` | `shelf_life_days` | Intermediate model |
| 2 | `fct_stock_weekly` | `days_of_cover` | Mart |
| 2 | `fct_stock_weekly` | `expiry_risk_flag` | Mart |
| 3 | `mart_replenishment` | `recommended_order_qty` | Mart |
| 3 | `report_expiry_exposure` | `units_at_risk` | Report |
| 4 | `plan_purchase_recommendation` | `qty` | Published decision |

Six fields, two marts, one report, one published plan. The blast radius is not the row count; it
is the fact that a unit change multiplies days of cover by roughly thirty for one source. A
rename alone would break six models at parse time, which is the easy failure. The unit change
would parse, run, and produce cover thirty times too large, which is the expensive one.

The impact analysis therefore classifies the change as breaking on two axes — name and unit —
and requires a deprecation window, an alias, and a replay comparison of the plan before and after
the change on the same inputs. The replay is the acceptance test: if the plan changes, the
difference must be attributable to the unit conversion and to nothing else.

### 6.7 What these controls still did not catch

After the three incidents the pipeline had a gate, a reconciled quarantine, and a passing R1
replay. It still could not answer four questions.

- **Whether the demand signal was consumption.** The plan measured DC shipments, not dispensed
  demand; every key was unique and the population was wrong.
- **Whether the allocation rule matched contract.** Conservation was tested, fairness was not.
- **Whether a consumer outside the warehouse used the field.** One report read the plan through a
  direct connection that no lineage edge recorded.
- **Whether the quarantine was complete.** Rows rejected by the parser never reached the
  reconciliation; they were counted in a log.

These are the boundaries of the machinery, and section 11 states them as limits rather than
leaving them to be discovered.

## 7. Code, query, or calculation

The single block below is the lineage query that answers the question a schema change actually
asks: which fields downstream of this one are determined by it. It is chosen over a quarantine
insert because the insert is mechanical and the traversal is where the judgement lives.

### 7.1 Downstream impact of one field

```sql
-- Downstream fields determined by conformed.dim_item.shelf_life_days, to any depth.
with recursive impact as (
    select source_dataset, source_column,
           target_dataset, target_column,
           1 as depth
    from meta.column_lineage
    where source_dataset = 'conformed.dim_item'
      and source_column  = 'shelf_life_days'

    union all

    select i.source_dataset, i.source_column,
           l.target_dataset, l.target_column,
           i.depth + 1
    from impact i
    join meta.column_lineage l
      on l.source_dataset = i.target_dataset
     and l.source_column  = i.target_column
    where i.depth < 12
)
select target_dataset, target_column, min(depth) as hops
from impact
group by 1, 2
order by hops, 1, 2;
```

### 7.2 The decisions the query does not state

| Decision | Choice made here | Why | What would change it |
|---|---|---|---|
| Direction | Downstream from the changed field | The question is impact, not origin | A root-cause investigation traverses upstream |
| Depth bound | 12 | Prevents a cycle from running forever | A deeper graph needs a visited-set, not a bound |
| Deduplication | `min(depth)` per target field | The shortest path is the most legible explanation | Path enumeration is a different query |
| Edge semantics | A field is an input if it can affect the output value | Direct and indirect determination are not distinguished | Transformation type must be recorded on the edge |
| Renames | Not followed | The graph stores names, not identity | A field-identity registry exists |
| Deleted fields | Not followed | Deletion removes the edge | The graph is bitemporal |

The depth bound is a pragmatic guard, not a semantic statement. A recursive query over a cyclic
graph returns rows forever, and lineage graphs acquire cycles when a model is repointed during a
migration. A visited set is correct; the bound is cheap.

> **Note:** SQL dialect matters. PostgreSQL and DuckDB require `with recursive`; SQL Server uses
> `with` and infers recursion; BigQuery requires `with recursive` and has its own iteration limit.
> The query is portable in intent and not in syntax.

### 7.3 What column-level lineage cannot see

- **Dynamic SQL.** A query assembled at runtime has no static parse. The graph records the
  template, not the fields.
- **Wildcard selects.** `select *` produces edges only if the parser expands the schema at parse
  time, and the expansion is a snapshot of a schema that may change.
- **Python and notebook transforms.** A feature computed in a dataframe has no SQL to parse. It
  needs instrumentation or a declared mapping.
- **Semantic-layer calculations.** A measure defined in a business intelligence (BI) tool is
  downstream of the warehouse field but absent from the warehouse graph.
- **Direct connections.** A report that queries the warehouse with its own credentials creates no
  job and no edge. Access logs are the only evidence.
- **Meaning.** The graph says the field flows. It does not say the consumer uses it correctly.

A lineage graph is a lower bound on impact, and the analysis should say so. The supplement is a
consumer registry from access logs, the only mechanism that sees the direct connection.

## 8. Data, standards, and regulatory context

Nothing in this topic is mandated by a standard. What is mandated is retention and production of
records, and those obligations turn reproducibility from good practice into an operational
requirement with a deadline.

### 8.1 Standards and rules that bear on reproducibility

| Standard or rule | What it constrains | How it touches this topic | Grade |
|---|---|---|---|
| OpenLineage specification | Lineage event schema, including column-level facets | The interoperable form of the graph in section 7 | B |
| ISO 8601 | Date and time representation | Event, ingestion, and knowledge timestamps | B |
| W3C PROV | Provenance model for entities, activities, and agents | Vocabulary for run records and manifests | B |
| 21 U.S.C. § 360eee-1 (DSCSA) | Retention of transaction information, history, and statements for at least six years | Sets the reconstruction horizon for pharmaceutical planning data | B |
| FDA Food Traceability Rule, 21 CFR Part 1 Subpart S (FSMA 204) | Key data elements at critical tracking events; records kept two years; produced within 24 hours | Sets both the horizon and the response time for food planning data | B |
| Sector audit-trail rules | Tamper-evident records of who changed what | Constrains whether a plan version may be edited | B |
| Records retention for financial reporting | Commonly seven years, jurisdiction-dependent | Longer than most pipeline retention defaults | C |

### 8.2 The retention arithmetic

Six years of daily plans at 6,240 item-locations is:

```text
plan rows        = 6,240 x 365 x 6 = 13,665,600
input partitions = 3 sources x 365 x 6 = 6,570
```

That is small. The expensive part is not the output; it is the inputs and the reference data. If
the plan depends on an item master that is overwritten nightly, six years of plan rows are six
years of numbers whose derivation cannot be checked. Retention without reconstruction is storage,
not compliance.

The FSMA 204 response time makes the point sharper. A record must be produced within 24 hours of
a request. If reconstruction requires a six-hour engineering effort, an unavailable source
credential, or a code version that no longer builds, the organisation cannot meet the deadline
even though it holds every row. Reproducibility is therefore an operational service level, not an
archival property.

> **Note:** The compliance dates for FSMA 204 have moved. The FDA proposed an extension of the
> compliance date to 20 July 2028, and congressional direction in the FY2026 appropriations act
> barred enforcement before that date. Consult the current rule text before relying on any date;
> the retention and production obligations themselves are unchanged.

### 8.3 What regulation does not require

- It does not require column-level lineage. It requires the ability to answer a question, and
  lineage is one efficient way to answer it.
- It does not require a quarantine table. It requires records that are complete and traceable;
  quarantine is a mechanism for knowing what is missing.
- It does not require that every test be a gate. It requires that decisions be supportable, which
  implies that the tests protecting a decision are gates.

The distinction matters because compliance projects tend to produce archives, and archives do not
answer questions at 24 hours' notice.

## 9. Industry practice and case evidence

### 9.1 Where the patterns came from

The write-audit-publish pattern is widely attributed to a 2017 Netflix DataWorks talk by Michelle
Winters, which described writing to an isolated location, validating, and promoting atomically.
The analogy to blue-green deployment is exact: production is never the place where an unvalidated
artifact first appears.

Quarantine has a different lineage, from messaging. A dead-letter queue isolates a message the
consumer cannot process so that the partition does not stall. Data platforms adapted it to
row-level validation, and the adaptation is where most implementations go wrong: a DLQ is designed
to preserve a message for a retry, while a quarantine must also price the exclusion for the
decision that proceeds without it.

### 9.2 What vendor documentation specifies

| Claim | Source | Grade | Comment |
|---|---|---|---|
| A test at `warn` severity never fails the run unless warnings are promoted | dbt documentation, severity reference | B | Documented behaviour, verifiable by exit status |
| `error_if` and `warn_if` accept count thresholds independent of severity | dbt documentation, severity reference | B | Allows a tolerance without changing the route |
| Stored test failures replace the prior result on every run, including zero-failure runs | dbt documentation, `store_failures` | B | A test audit table is not a quarantine ledger |
| Iceberg snapshots support branches and tags with independent retention | Apache Iceberg documentation | B | The mechanism behind write-audit-publish on a table |
| Delta Lake time travel addresses versions and timestamps; log retention defaults to 30 days and file retention to 7 days | Databricks documentation | B | Defaults are short relative to a six-year obligation |
| Expectations can warn, drop, or fail a record or a pipeline | Databricks documentation | B | The three responses, expressed as an action |
| A SQL parser can extract column-level lineage across named dialects | OpenLineage SQL parser | B | Dialect coverage is enumerated, not universal |
| Marquez exposes a column-lineage endpoint over the aggregated graph | Marquez API documentation | B | The graph is only as complete as the emitters |

Two qualifications. Retention defaults are configurable; a platform configured for a six-year
obligation is not the platform as shipped. And the parser extracts what the SQL says, not what the
job does; dynamic SQL and Python transforms are outside it.

### 9.3 Quarantine as implemented

Three implementation shapes recur in vendor documentation.

- **A boolean split.** Databricks' pattern adds an `is_quarantined` column, partitions on it, and
  derives valid and invalid views. Simple, and it keeps the reason only as rule outcomes.
- **A universal dead-letter table.** Lakehouse Plumber writes a deterministic key, a status, the
  failed rules, and the row payload, then recycles fixed rows through an outbox. This is the
  ledger shape, closest to section 5 step 6.
- **A message envelope.** Streaming tools write payload, reason, error kind, and record index,
  with a policy for whether a batch error sends all records or only the failing one. Sending the
  whole batch on one bad record converts a row defect into a batch loss.

All three share the same failure: an unmonitored quarantine. A table that fills and is never read
is a slower way of losing data.

### 9.4 What the evidence does not establish

There is no controlled study showing that a severity policy, a quarantine table, or a replay test
reduces planning error. The available evidence is mechanical (what a tool does), regulatory (what
a record must contain and how long it must be kept), and anecdotal (what practitioners report).
Vendor claims about efficiency gains from write-audit-publish should be read as marketing unless
a method is published; one such article reports a pipeline time reduction in a single customer
case, which is not evidence of general effect.

The claims that survive are narrow: a `warn` test does not stop a run; a stored failure table is
replaced each run; default time-travel retention is shorter than a regulatory horizon; a parser
extracts edges from the SQL it can parse.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Warning treated as a gate | The same test has warned for weeks | `warn` does not fail the run | Silent partial data in a published plan | Promote decision-protecting tests to gates; review advisory tests quarterly |
| Quarantine threshold in rows | Row rate under limit; output obviously wrong | Rows are heterogeneous | Large unit loss passes the check | Compute unit-weighted impact |
| Unreconciled quarantine | Accepted plus quarantined never checked against input | No identity | Rows vanish undetected | Reconciliation as a gate |
| Fail-open by join | Orphan count in a log | Inner join drops rows | Population shrinks silently | Anti-join count as a gate |
| Missing value becomes zero | A zero where no receipt exists | `coalesce(..., 0)` | Phantom supply, under-ordering | Preserve null; separate absence from zero |
| Empty input, green run | Zero rows written, task succeeded | Aggregation over empty set | A complete-looking empty plan | Non-emptiness gate relative to history |
| Wall-clock in logic | Replay differs by a day | `current_date` filter | Unreproducible output | Parameterise the window |
| Current-state reference data | Replay differs after a rate change | FX or factor table read as current | Plans compared across vintages | As-of reference data in the tuple |
| Version recorded as a timestamp | Two runs, same timestamp, different inputs | No manifest | Cannot distinguish plans in review | Version tuple and run record |
| Test failure store used as quarantine | Failures overwritten nightly | `store_failures` replaces prior results | No history of defects | Separate ledger table |
| Lineage only table-level | Impact analysis names a table, not a field | No column edges | Underestimated blast radius | Column-level capture plus consumer registry |
| Direct warehouse connections | A report breaks with no lineage edge | Credentials outside the platform | Invisible consumer | Access-log registry; review grants |
| Deprecation announced after removal | Consumers discover at failure | No consumer list | Broken reports and plans | Window, alias, and lineage-derived list |

Two rows deserve emphasis because they are the most common and the least discussed.

**Warning treated as a gate** is not a tooling failure. The test worked. The severity was a
decision made during a migration and never revisited, and no review process owned it. The
mitigation is a review with a date, not a better test.

**Missing value becomes zero** is the quietest fail-open mechanism in supply chain SQL. A
`left join` to receipts followed by `coalesce(qty, 0)` turns "we do not know" into "we know it is
none", and every downstream calculation is arithmetically correct on a false premise. The fix is
to keep null and to make the plan handle unknown explicitly, which is more work than a threshold
change and is the actual work.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Every input can be enumerated | A manifest requires a closed set | Direct queries, undocumented feeds | Compare manifest against access logs |
| The code version is recoverable | Replay needs the same logic | Deleted branches, unpinned images | Attempt to check out and build each recorded commit |
| Reference data has an as-of version | Replay needs the vintage | Overwritten rate or master tables | Change a rate and replay |
| The comparison key is stable | Replay compares artifacts | Key renames, grain changes | Freeze the key; version it with the plan |
| Tests run on representative data | Passes mean something | Skewed or empty partitions | Track row counts with results |
| Quarantine reasons are accurate | The ledger explains the exclusion | Default or free-text codes | Sample and adjudicate monthly |
| Lineage captures the consumers that matter | Impact analysis is complete | BI tools and direct connections | Reconcile graph against access logs |
| The population contract is current | Proceed or halt is defensible | Business change, new segments | Review at a stated cadence |
| Retention exceeds the reconstruction horizon | Audit can be answered | Short source retention | Compare source and platform retention |
| The replay environment exists | A replay can be executed | Tool version drift, missing credentials | Schedule a replay, not a plan for one |

Four limits are worth stating plainly.

**Reproducibility does not imply correctness.** A pipeline can reproduce a wrong plan perfectly.
Every input recorded, every row identical, and the demand signal is shipments rather than
consumption. Reproduction defends the derivation, not the premise.

**Lineage does not capture meaning.** A field edge says the value flows. It does not say the
consumer uses it as a daily quantity rather than a weekly one. Impact analysis finds the
artifacts; a person must judge the semantic change.

**Quarantine does not make partial data safe.** It makes partial data visible. Whether the
partial population is usable is a decision about the decision, and no table can make it.

**A replay is only as good as its comparison.** An aggregate match rate can hide compensating
errors in opposite directions. Compare per key, on the measures the decision uses, and state the
tolerance before looking at the result.

## 12. Exercises

Nine exercises, ordered by the size of the decision they support, each with a deliverable, an
acceptance test, and a time estimate.

### Exercise 1 — Severity audit (critique, 50 minutes)

Take a pipeline you can inspect, or the Kestrel estate in section 6. List every test and its
severity. For each, name the decision that consumes the field and the consequence of a wrong
value.

**Deliverable:** a table of test, field, consumer decision, consequence, current severity, and
proposed severity, with a one-line justification for every change.
**Acceptance criteria:** at least three tests are promoted from advisory to gate with a stated
consumer; at least two advisory tests are deleted as having no consequence; every gate names an
owner and a review date; the submission distinguishes a warning from a gate in terms of run
behaviour, not importance.

### Exercise 2 — Quarantine impact (calculation, 45 minutes)

A file has 2,400 lines. Of these, 180 fail a unit-of-measure check. Accepted lines average 6
units; quarantined lines average 150 units. The quarantine gate is set at 10% of rows.

**Deliverable:** row rate, unit-weighted impact, the gate decision under the stated rule, and the
decision under a contract that denominates impact in units.
**Acceptance criteria:** both percentages computed with their denominators; the proceed or halt
recommendation justified against the decision, not the threshold; one sentence naming what the
plan will do with the missing units.

### Exercise 3 — Replay classification (analysis, 60 minutes)

A replay of last week's plan returns 2,340 recommendations against 2,340 originally, with 63
recommendations differing. Original recommended units were 2,910,000; the replay differs by
18,700 units. Of the 63, 55 are traced to a current-state FX table and 8 to a `current_date`
filter. No difference is unexplained.

**Deliverable:** the row and unit difference rates, a replay class with justification, and the
three changes that would move the result to R1.
**Acceptance criteria:** the class is chosen from the R0–R4 scale with a stated tolerance; the
difference between "explained" and "reproducible" is stated explicitly; each proposed change
names the ambient state it removes.

### Exercise 4 — Design the quarantine ledger (design, 60 minutes)

Design a quarantine table for the ASN lines in section 6.3. It must support deduplication on
reprocessing, a repair and recycle path, and a monthly impact report.

**Deliverable:** a table definition with columns and types, a status lifecycle diagram, and the
reconciliation query in prose.
**Acceptance criteria:** a deterministic key; a reason envelope with rule and observed value; a
status of quarantined, fixed, or rejected; an expiry rule tied to retention; a stated owner; one
sentence on why a test failure store is not a substitute.

### Exercise 5 — Severity coverage arithmetic (calculation, 35 minutes)

A suite has 410 tests: 314 at gate severity and 96 at advisory. Of the advisory tests, 61 have no
named owner. Each advisory test has a 2% chance of firing on a given run, and triage takes six
minutes. There are 260 runs per year.

**Deliverable:** the share of advisory tests without an owner, the expected warnings per year,
the triage hours per year, and a recommendation.
**Acceptance criteria:** the expected value is computed as tests x runs x firing probability; the
triage cost is computed in hours and in currency at a stated rate; the recommendation addresses
ownership before volume.

### Exercise 6 — Impact analysis memo (design, 55 minutes)

A proposed change renames `shelf_life_days` and changes its unit from days to months for one
source. Using the lineage result in section 6.6, write the impact analysis.

**Deliverable:** a one-page memo with the affected fields, the change classification, the blast
radius, the deprecation window, and the acceptance test.
**Acceptance criteria:** at least four downstream fields named with hops; name and unit changes
classified separately; the blast radius stated in published artifacts as well as models; the
acceptance test is a replay on identical inputs.

### Exercise 7 — Replay test plan (design, 45 minutes)

Write the plan for running a replay test on a monthly cadence for a daily plan.

**Deliverable:** the comparison key, the measures, the tolerance, the environment, the pass and
fail actions, and the owner.
**Acceptance criteria:** the key is at the decision grain; the tolerance is stated before the
test; the environment can build a recorded commit; a failed replay has a named action, not
"investigate"; the plan says what is compared when the code version no longer builds.

### Exercise 8 — Fail-open hunt (critique, 55 minutes)

Inspect a pipeline and find every mechanism by which it fails open. Use the six mechanisms in
section 3.2 as a checklist.

**Deliverable:** a table of mechanism, location, evidence, consequence, and cheapest control.
**Acceptance criteria:** at least six mechanisms located in real code or configuration; each
consequence stated in units or decisions, not "data quality"; at least one control is cheaper
than the failure it prevents; one mechanism is deliberately kept open with a written reason.

### Exercise 9 — Module synthesis (design, 70 minutes)

Write a one-page argument that a named planning number is reproducible. Use the eleven topics as
headings, and for each state what would have to be true.

**Deliverable:** eleven short paragraphs and a final paragraph naming the weakest link.
**Acceptance criteria:** every topic is addressed with a specific artifact, not a generality; at
least three topics are marked as partially satisfied; the weakest link is named with the evidence
that would close it; the argument does not claim reproducibility without a replay result.

## 13. Self-check answers

### 13.1 Exercise 2 arithmetic

```text
accepted lines      = 2,400 - 180 = 2,220
accepted units      = 2,220 x 6   = 13,320
quarantined units   = 180 x 150   = 27,000
total units         = 13,320 + 27,000 = 40,320

row rate            = 180 / 2,400        = 7.50%
unit impact         = 27,000 / 40,320    = 66.96%
```

Under the stated rule the run proceeds, because 7.5% is below the 10% row threshold. Under a
contract that denominates impact in units, the run halts: two-thirds of the units in the file are
missing, and the plan will substitute against a lead time. The correct answer states both and
recommends halt with the prior plan and an exception list.

The row rate understates impact by a factor of 8.9 here (66.96 / 7.50), the ratio of average
quarantined to accepted units diluted by line shares. It is not a constant; compute it from the
data on each run.

### 13.2 Exercise 3 arithmetic

```text
row difference   = 63 / 2,340        = 2.69%
unit difference  = 18,700 / 2,910,000 = 0.64%
attributed       = 55 + 8 = 63
unexplained      = 0
```

The result is R3: every difference is explained, and the plan is not reproducible without
manual attribution. R1 requires zero differences on the declared key and measures. The three
changes that would reach R1 are: pass the FX vintage as a run parameter and read the rate as of
the cutoff; pass the window start as a parameter and remove `current_date`; and record both in
the run record so the next replay can assert equality rather than explain a difference.

A weak answer reports 99.36% agreement and calls it reproducible. Agreement on units is not
reproduction; the differing rows are the decisions.

### 13.3 Exercise 5 arithmetic

```text
advisory tests without owner = 61 / 96 = 63.5%
expected warnings per year   = 96 x 260 x 0.02 = 499.2
triage hours per year        = 499.2 x 6 / 60 = 49.9 hours
cost at $85/hour             = 49.9 x 85 = $4,243
```

Fifty hours a year is a working week spent reading messages that stop nothing. Assign an owner to
each advisory test or delete it, and promote the subset that protects a decision. The 61 unowned
tests are the background level against which a real warning is invisible.

### 13.4 Exercise 6 model answer

A strong memo separates the two changes. The rename is breaking at parse time and affects six
fields across four hops; it is caught by a build failure and is therefore cheap. The unit change
is semantic: it parses, it runs, and it multiplies days of cover by approximately thirty for one
source. The blast radius is six fields, two marts, one report, and one published plan, and the
plan is the artifact that matters. The deprecation window carries an alias for the old name and a
conversion test for the unit. The acceptance test is a replay of the same input manifest before
and after the change, with every difference attributed to the conversion and nothing else.

A weak memo lists the models and says "low risk because only six fields". It has counted the
graph and not the decision.

### 13.5 Rubrics for the open exercises

- **Exercise 1.** Strong: severity derived from consequence, gates with owners and review dates,
  at least two deletions justified. Weak: every test promoted to error, or severity discussed as
  a matter of importance rather than routing.
- **Exercise 4.** Strong: deterministic key, reason envelope, three exits, reconciliation
  identity, and an explicit contrast with `store_failures`. Weak: a table with a payload and a
  timestamp and no lifecycle.
- **Exercise 7.** Strong: comparison key at the decision grain, tolerance fixed in advance,
  environment that can build an old commit, and a fail action. Weak: "compare row counts" or a
  plan that assumes the current branch still represents the old code.
- **Exercise 8.** Strong: six mechanisms with file-level evidence and unit-denominated
  consequences; one deliberate exception documented. Weak: a list of tools or a general claim
  that "we should add tests".
- **Exercise 9.** Strong: eleven specific artifacts, three partials, a named weakest link with
  closing evidence, and no claim of reproducibility without a replay. Weak: a summary of the
  module with no artifact names.

## 14. Interview questions

**1. "What is the difference between a warning and a gate?"**

A strong answer describes routing, not importance: a warning records a failure and the run
continues with an unchanged exit status; a gate halts downstream execution and withholds the
plan. It adds that the choice is made from the consequence of the decision the field feeds, and
that a warning without a named owner is a message to nobody. A weak answer says a warning is less
serious, which describes a label rather than a behaviour.

**2. "When would you let a pipeline proceed with a partial population?"**

A strong answer requires a population contract: the missing keys, the critical segments, the
unit-weighted impact, the decision owner, and the fallback. It notes that a non-critical segment
at 2% of units can proceed and a critical molecule at 0.1% of rows may not. A weak answer gives a
percentage threshold without saying what the percentage is of.

**3. "Why is a row-based quarantine threshold dangerous?"**

A strong answer explains that rows are heterogeneous: one line for a high-volume molecule can
carry more units than a thousand lines for slow movers, and cites a case where 10.6% of rows
carried 78% of units. It says the threshold must be denominated in the decision variable. A weak
answer says row thresholds are fine if set low enough.

**4. "What does a quarantine table need that a test failure table does not?"**

A strong answer names a deterministic key, a reason envelope, a status lifecycle with repair and
recycle, a reconciliation to input, and an owner. It notes that dbt's `store_failures` replaces
the prior result on every run, including a zero-failure run, so it cannot serve as a defect
ledger. A weak answer says both store bad rows.

**5. "Name three ways a pipeline fails open."**

A strong answer gives mechanisms with consequences: an inner join drops orphans and shrinks the
population; `coalesce(qty, 0)` turns unknown into zero; a `warn` test leaves the exit status
unchanged; a task that writes zero rows reports success. It notes that each is a default, not a
decision. A weak answer says "bad data gets through".

**6. "What is a version tuple for a planning run?"**

A strong answer lists inputs with hashes or snapshot identifiers, code commit, reference-data
vintage, parameters, and a run identifier. It explains that a timestamp is not a version because
two runs can share a timestamp and differ in inputs. A weak answer says "the run date and the git
commit".

**7. "How would you test whether last week's plan is reproducible?"**

A strong answer specifies the comparison key at the decision grain, the measures, a tolerance
fixed in advance, and the classification of differences as explained or unexplained. It notes
that agreement on aggregate units can hide compensating errors and that an R3 result is not an R1
result. A weak answer says "rerun it and compare the totals".

**8. "A model filters on `current_date`. Why does that break replay?"**

A strong answer explains that the window moves with the replay date, so the population differs
even when the code and inputs are identical. The fix is to pass the window start as a parameter
recorded in the run. It adds that the same defect class includes unseeded randomness and
sequence-generated keys. A weak answer says it makes the query slow.

**9. "What can column-level lineage not tell you?"**

A strong answer lists dynamic SQL, wildcard expansion, Python transforms, semantic-layer
measures, direct warehouse connections, and meaning. It concludes that a lineage graph is a lower
bound on impact and needs a consumer registry from access logs. A weak answer treats the graph as
complete.

**10. "How do you run an impact analysis before a schema change?"**

A strong answer traverses downstream to closure, joins to published artifacts and consumers,
separates name changes from semantic changes, states the blast radius in decisions, and sets a
deprecation window with an alias and a replay acceptance test. A weak answer searches the
codebase for the column name, which misses dynamic SQL and every consumer outside the repository.

**11. "What is the difference between retention and reproducibility?"**

A strong answer says retention keeps the rows; reproducibility can re-derive the output from
them. It gives the regulatory version: a 24-hour production obligation is unmet by an archive
that requires a six-hour rebuild. A weak answer treats a backup as reproducibility.

**12. "Why is this the last topic in the module?"**

A strong answer says the earlier topics each remove a source of ambient state and this one tests
whether all of them hold at once. It names the replay test as the only check that exercises
layers, ingestion, transformation, orchestration, and documentation together. A weak answer says
because it is about quality, which could have been first.

## 15. Key terms

- **Data quality test** — a query returning failing rows; zero rows means pass.
- **Severity** — the routing level of a failed test: advisory, gate, or halt.
- **Gate** — a check whose failure halts downstream execution and withholds the plan.
- **Quarantine** — isolation of individual bad rows while the run continues on the accepted set.
- **Dead-letter queue** — a durable store for unprocessable messages with a replay path.
- **Fail-open** — continuing after a failed check and producing a partial output.
- **Partial population** — the subset of expected entities present in a run's input.
- **Population contract** — pre-agreed coverage, impact limit, owner, and fallback.
- **Reconciliation** — proof that input equals accepted plus quarantined plus rejected.
- **Tolerance** — the stated admissible difference before a check fails.
- **Write-audit-publish** — write to an isolated location, audit, then publish atomically.
- **Snapshot** — an immutable record of a dataset at a point in time.
- **Immutable log** — an append-only record of changes from which state is derived.
- **Data version** — a named, immutable state of a dataset.
- **Code version** — the exact transformation logic used in a run.
- **Reference data vintage** — the as-of version of calendars, rates, factors, and masters.
- **Bitemporal** — recording both event time and knowledge time.
- **Event time** — when the event happened in the world.
- **Ingestion time** — when the row arrived in the platform.
- **Knowledge time** — when the organisation first knew the fact; the clock a plan depends on.
- **Replay** — reprocessing a declared range with a declared code and reference vintage.
- **Replay class** — R0 byte-identical through R4 unexplained; the required agreement.
- **Manifest** — the recorded list of input artifacts with hashes or versions.
- **Run record** — the stored tuple of interval, code, inputs, reference data, and outcome.
- **Plan version** — the immutable identity of a published plan.
- **Lineage** — the graph of datasets, jobs, and the fields that flow between them.
- **Column-level lineage** — mapping each output field to the input fields that determine it.
- **Impact analysis** — enumerating artifacts and consumers affected by a proposed change.
- **Deprecation window** — the announced period before a field or model is removed.
- **Point-in-time reconstruction** — rebuilding the exact state and output as of a past instant.

## 16. Sources

**Peer-reviewed and preprint research**

- Bauplan and Nessie, "Reproducible data science over data lakes: replayable data pipelines with
  Bauplan and Nessie," arXiv:2404.13682, 2024. Preprint; proposes a run identifier over versioned
  data, code, runtime, and hardware. Grade B as a preprint, not peer-reviewed.
- Tala, "Building a Reproducible Machine Learning Pipeline," arXiv:1810.04570, 2018. Preprint;
  covers data, feature, and model provenance and content hashes. Grade B as a preprint.
- Kimball, R., and Ross, M. (2013). *The Data Warehouse Toolkit*, 3rd edition. Wiley. Source for
  grain, as-of semantics, and slowly changing dimensions; not specific to quarantine or replay.
  Grade A for the modelling concepts.

**Standards, statutes, and official guidance**

- OpenLineage specification, OpenLineage project, including the column-lineage dataset facet.
- ISO 8601, *Date and time — Representations for information interchange*.
- W3C, *PROV-DM: The PROV Data Model*, W3C Recommendation.
- 21 U.S.C. § 360eee-1, Drug Supply Chain Security Act, requirements including six-year record
  retention.
- U.S. Food and Drug Administration, "FSMA Final Rule on Requirements for Additional Traceability
  Records for Certain Foods," 21 CFR Part 1 Subpart S.
- U.S. Food and Drug Administration, "Frequently Asked Questions: FSMA Food Traceability Rule,"
  including the two-year retention period and 24-hour record production.
- U.S. Food and Drug Administration, "Small Entity Compliance Guide: Requirements for Additional
  Traceability Records for Certain Foods."

**Vendor and project documentation (Grade B for documented behaviour, D for advice)**

- dbt Labs, "severity, error_if, and warn_if," dbt Developer Hub, accessed November 2026.
  https://docs.getdbt.com/reference/resource-configs/severity
- dbt Labs, "store_failures," dbt Developer Hub, accessed November 2026.
  https://docs.getdbt.com/reference/resource-configs/store_failures
- dbt Labs, "Data test configurations," dbt Developer Hub, accessed November 2026.
  https://docs.getdbt.com/reference/data-test-configs
- Apache Iceberg, "Branching and Tagging," accessed November 2026.
  https://iceberg.apache.org/docs/latest/branching/
- Databricks, "Work with table history," accessed November 2026.
  https://docs.databricks.com/aws/en/tables/history
- Databricks, "Manage data quality with pipeline expectations" and "Expectation recommendations
  and advanced patterns," accessed November 2026.
  https://docs.databricks.com/aws/en/ldp/expectations
- OpenLineage, "SQL integration" and pull request 1432, "sql: column-level lineage in SQL parser,"
  accessed November 2026. https://github.com/OpenLineage/OpenLineage/tree/main/integration/sql
- Marquez, "Get column lineage" API reference, accessed November 2026.
  https://github.com/marquezproject/marquez/blob/main/docs/docs/api/get-column-lineage.api.mdx
- DVC, "Get Started: Data Pipelines," accessed November 2026.
  https://doc.dvc.org/start/data-pipelines/data-pipelines
- lakeFS, "Open Table Formats vs. Data Version Control Systems," accessed November 2026.
  https://lakefs.io/blog/open-table-formats-vs-data-version-control-systems/
- Dremio, "Data lake versioning: Nessie vs Iceberg vs LakeFS," accessed November 2026.
  https://www.dremio.com/blog/data-lakehouse-versioning-comparison-nessie-apache-iceberg-lakefs/

**Practitioner and vendor commentary (Grade C to E)**

- dbt Labs, "Testing is not enough: Transforming data quality with Write, Audit, Publish,"
  accessed November 2026. Vendor argument with a single-customer efficiency claim; grade D.
- lakeFS, "Data Engineering Patterns: Write-Audit-Publish (WAP)," accessed November 2026.
  https://lakefs.io/blog/data-engineering-patterns-write-audit-publish/ Grade D.
- AWS Big Data Blog, "Build Write-Audit-Publish pattern with Apache Iceberg branching and AWS Glue
  Data Quality," accessed November 2026. Grade D.
- Lakehouse Plumber, "Quarantine (Dead Letter Queue)," accessed November 2026.
  https://lakehouse-plumber.readthedocs.io/quarantine.html. Implementation documentation; grade D.
- faucet-stream, "Dead-letter queues," accessed November 2026.
  https://faucet-hq.github.io/faucet-stream/cookbook/dlq.html. Implementation documentation; grade D.
- Coginiti, "Data Quality Design Patterns: WAP and AWAP," accessed November 2026. Grade D.
- Michelle Winters, Netflix DataWorks 2017 talk on write-audit-publish, as reported in secondary
  sources; the recording was not consulted. Grade C for the account, Grade E for summaries.

**Note on citations.** The Netflix origin of write-audit-publish is attributed as reported in
practitioner sources; the primary talk was not consulted. The FSMA 204 compliance date is in
flux. No controlled study was found on the effect of severity policy, quarantine, or replay on
planning outcomes; outcome claims in section 9 are attributed and graded, and no numeric vendor
claim is repeated as fact.

## 17. Links

### 17.1 The module in one argument

A plan is a claim about the world, and a claim is defensible only when its inputs are named. The
module's eleven topics are eleven ways of naming an input that would otherwise be ambient: a
layer boundary instead of a vague location, a cadence instead of "current", an ingestion
manifest instead of a folder, an as-of join instead of a coincidence, a test instead of an
assumption, a cutoff instead of a habit, a document instead of a rumour, an identifier instead of
a description, a contract instead of a convention, and — in this topic — a severity, a ledger, a
version tuple, and a graph instead of the belief that it can all be reconstructed later.

The argument closes only when the replay test passes. Each topic removes one excuse; the replay
is the check that none of them was reacquired.

### 17.2 Links

- Module map: [`_index.md`](_index.md)
- Prerequisite — layers, contracts, and replay: [`01_layered-architecture.md`](01_layered-architecture.md)
- Prerequisite — tests, grain, and severity: [`06_dbt-models-tests.md`](06_dbt-models-tests.md)
- Prerequisite — cutoff, run identity, and backfill: [`07_orchestration.md`](07_orchestration.md)
- Sibling — temporal SQL semantics used by the replay comparison: [`04_sql-supply-chain-patterns.md`](04_sql-supply-chain-patterns.md)
- Sibling — cadence and what "current" means: [`02_batch-microbatch-streaming.md`](02_batch-microbatch-streaming.md)
- Sibling — arrival, deletes, and redo gaps: [`03_ingestion-patterns.md`](03_ingestion-patterns.md)
- Sibling — testability of transformation code: [`05_python-structure-tests.md`](05_python-structure-tests.md)
- Sibling — documents as assertions: [`08_edi-x12-edifact.md`](08_edi-x12-edifact.md)
- Sibling — identity and event data: [`09_gs1-epcis.md`](09_gs1-epcis.md)
- Planned sibling — APIs, data contracts, and master data management: `10_apis-contracts-mdm.md`
  (not yet written)
- Time, units, and assumption conventions used in the worked example: [`../01A_foundations_units_time_assumptions/_index.md`](../01A_foundations_units_time_assumptions/_index.md)
- Quality dimensions referenced throughout: [`../02_supply_chain_data_landscape/08_data-quality-dimensions.md`](../02_supply_chain_data_landscape/08_data-quality-dimensions.md)
- Survey file for this module: [`../../../course/03_data_engineering_interoperability.md`](../../../course/03_data_engineering_interoperability.md)
