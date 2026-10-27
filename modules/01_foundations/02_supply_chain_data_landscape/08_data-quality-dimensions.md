# Data-Quality Dimensions

> Module 02 — The Supply Chain Data Landscape
> Topic 08 of 10
> Estimated study time: 4 hours
> Prerequisites: 02.01–02.07
> Version 2026.5

---

## 1. Why this topic exists

The previous seven topics mapped where supply chain data lives: the systems of record, the
master data that joins them, the transactions, the external signals, the sensors, the transport
mechanisms, and the contractual limits on access. Every one of them ended with the same
unanswered question. The data exists, you can reach it, you are allowed to use it. Is it good
enough for the decision in front of you?

That question has no binary answer, and the habit of pretending otherwise produces most of the
bad analytics in this field. Three examples show the shape of the problem.

**The carrier that was not the worst.** A logistics team ranks eight carriers by on-time
delivery from a shipment event table. The carrier ranked last has 3,100 events, of which 214 are
duplicate loads and 290 have a delivered status with no delivery timestamp. Both defects are
concentrated in that carrier because it sends batch files rather than using an application
programming interface (API). Remove the duplicates and count undated deliveries as unknown
rather than on-time, and the carrier moves from eighth to fourth. The ranking was never a
measurement of the carrier. It was a measurement of the carrier's integration quality.

**The 99.5% that hid a 4% hole.** A planning team reports 99.5% completeness on its
item-location-week demand history. The number is computed at cell level across a wide table. At
record level, 4.1% of item-location combinations have no rows at all for the last thirteen weeks,
because the extract filters on a warehouse attribute that was deprecated for slow-moving items.
The model trains on a history in which the intermittent items it most needs to handle are
absent. Cell-level completeness was correct and irrelevant.

**The forecast that trained on its own corrections.** A model is trained on confirmed order
quantities. Planners routinely amend orders downward when supply is short, and the amendment
overwrites the original quantity. At training time the model sees the amended number, which the
planner could not have known when the decision was made. Offline accuracy is excellent and
production performance is poor. No completeness or validity check would have caught it.

Each example is a quality failure that survived a quality dashboard. The dashboards were not
lying. They were answering a different question from the one being asked.

This topic exists because "the data is bad" is an unusable statement. It cannot be tested,
prioritised, budgeted, or fixed. What can be tested is a named dimension, at a stated grain,
against a stated requirement, with a stated consequence when it fails. The discipline is the
conversion of a vague complaint into that sentence.

Three claims organise what follows.

1. **A dimension without a test is an adjective.** Completeness, validity, consistency,
   timeliness, uniqueness, and accuracy become useful only when each is written as a rule, a
   grain, a denominator, and a threshold. The same word at a different grain is a different
   number, often by several percentage points.
2. **The six classical dimensions are necessary and not sufficient.** Two more carry most of the
   operational weight in supply chain work: plausibility, which is what you can measure when no
   reference truth exists, and fitness for purpose, which is the only dimension that answers the
   question the business actually asked.
3. **Data quality is a process, not a score.** Detection without triage produces alert fatigue.
   Triage without root-cause analysis produces repeated fixes to symptoms. Remediation without
   prevention guarantees the defect returns with the next release.

For a data scientist the stakes are asymmetric. A quality failure upstream of a dashboard produces
a wrong number someone may notice; the same failure upstream of a model produces a plausible
number nobody notices for a quarter. Monitoring catches drift in inputs, not a label that was wrong
at source, a denominator that excludes failures, or a training window containing the future.

## 2. Learning outcomes

By the end of this topic you can:

1. **Define** each of the six classical dimensions plus plausibility and fitness for purpose as a
   computable test, stating entity, grain, rule, denominator, and threshold.
2. **Compute** completeness at cell, column, row, and record grain for one dataset, and
   **explain** why the four numbers differ and which one the decision requires.
3. **Distinguish** validity from accuracy, and syntactic from semantic accuracy, using an example
   in which a value passes every rule and is still wrong.
4. **Design** an accuracy measurement when no reference truth exists, choosing among external
   reference, cross-source agreement, physical audit sample, and outcome proxy, and stating what
   the chosen substitute cannot prove.
5. **Identify** at least three ways a metric computed from the data it audits inherits that data's
   defects, and propose an independent instrumentation for each.
6. **Quantify** the business impact of a defect set, rank remediation by impact and effort, and
   present the estimate as a range with its assumptions exposed.
7. **Design** a six-gate pipeline check (schema, freshness, volume, distribution, referential
   integrity, business rules) with block-versus-warn decisions and a quarantine path.
8. **Critique** a published cost-of-poor-quality figure for sample selection, construct
   ambiguity, and currency, and explain why it cannot substitute for a local estimate.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold one picture: a pipeline of gates, one dimension each, running from source systems to consumer.
Every data scientist who has been surprised by a model in production was surprised at one of these
gates.

```text
   SOURCE SYSTEMS                    QUALITY GATES (in pipeline order)                 CONSUMER
   ----------------                  ----------------------------------                 --------
   ERP  ─┐
   WMS  ─┼──▶ [1 SCHEMA] ─▶ [2 FRESHNESS] ─▶ [3 VOLUME] ─▶ [4 DISTRIBUTION] ─▶ [5 REFERENTIAL] ─▶ [6 BUSINESS RULES] ─▶ model
   TMS  ─┤        │               │                │                 │                    │                    │            report
   EDI  ─┤     validity        timeliness       completeness      consistency         integrity            plausibility     planner
   API  ─┘        │            + currency        (record level)    (over time)        + validity           + cross-field
                  v               v                v                 v                    v                    v
              quarantine      warn / block     warn / block      warn only            block                block
                  │
                  └──────────▶ defect register ──▶ triage ──▶ root cause ──▶ fix at source
```

```text
   MEASURABLE FROM THE DATA ALONE                    REQUIRES SOMETHING OUTSIDE THE DATA
   completeness, validity, consistency,             accuracy      -> reference truth
   uniqueness, timeliness, plausibility             fitness for purpose -> a decision owner
```

Five properties follow, and they explain most of what comes later.

**Property 1 — A dimension is a test, not an adjective.** "The carrier data is incomplete" cannot
be true or false. "94.2% of shipment rows have a non-null carrier identifier, against a
requirement of 99.0% for cost allocation" can be. Conversations that stay at the adjective level
end in a disagreement about feelings.

**Property 2 — Grain determines the number.** Cell, column, row, and record completeness are four
measurements of one table, and they can differ by ten percentage points with none of them wrong.
A model that consumes rows cares about row completeness. A reconciliation that needs one field
cares about column completeness. A planning process that assumes every item-location exists
cares about record completeness.

**Property 3 — Dimensions are coupled.** Enforcing uniqueness deletes records, which reduces
completeness. Enforcing validity by rejecting out-of-range values nulls them, which reduces
completeness and destroys the evidence that an exception existed. Publishing early to improve
timeliness reduces completeness because late records are missing. There is no setting at which
all dimensions improve at once. There is a chosen operating point and a documented reason for it.

**Property 4 — Only accuracy needs the world.** The other six are comparisons against a
specification or against the data itself, so they can be computed for any dataset with no
external input. Accuracy compares a value to what happened, which the data cannot supply about
itself. This is why accuracy is expensive, why it is sampled rather than exhaustive, and why
claims of high accuracy without a named reference are marketing.

**Property 5 — Quality is relative to a decision.** A dataset unfit for carrier scorecards may be
perfectly fit for five-year network design, which aggregates away the defects that destroy the
scorecard. Fitness for purpose is not a dimension alongside the others. It is the judgement that
selects which of the others matter, at what grain and against what threshold.

### 3.1 Two axes that place a quality programme

Quality work is chosen, not exhaustive. Effort belongs where measurement is cheap and decision
impact is high: those checks automate and block. Cheap checks with low impact are logged and
ignored; expensive checks with low impact are documented as gaps; expensive checks with high impact
are sampled and funded with a reference source. Most teams invert this, automating the easy checks
and skipping accuracy work because it requires leaving the building.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Data quality | Degree to which data meets stated requirements for a stated use | Treated as an intrinsic property | Dimension name plus score |
| Data-quality dimension | A named, measurable aspect of that degree | Used as a synonym for "metric" | Dimension name |
| Data-quality rule | A deterministic test a record passes or fails | Confused with a model or prediction | Rule identifier |
| Completeness (cell) | Share of expected cells that are populated | Reported as "completeness" with no grain stated | Proportion (0–1) or percent |
| Completeness (column) | Share of rows with a usable value in one named column | Computed including placeholders | Proportion |
| Completeness (row) | Share of rows with every required field usable | Confused with cell completeness | Proportion |
| Completeness (record) | Share of expected entities that are present at all | Omitted from most dashboards | Proportion |
| Validity | Conformance to a declared domain, type, format, range, or code list | Treated as proof of correctness | Proportion of values passing |
| Code list | A controlled vocabulary, usually versioned and externally maintained | Assumed static | Version identifier plus issue date |
| Referential integrity | Every foreign key resolves to an existing parent record | Checked only at initial load | Violation count |
| Consistency | Agreement between two representations of the same fact | Used to mean "no contradictions anywhere" | Conflict rate |
| Cross-field consistency | Agreement between fields within one record | Confused with cross-system consistency | Violation count |
| Temporal consistency | Agreement of a state with its own history | Ignored in snapshot-only tables | Reconciliation residual |
| Timeliness | Data available when the decision that uses it is made | Used to mean "recent" | Hours |
| Currency | Age of the newest record in a dataset | Used interchangeably with latency | Hours or days |
| Latency | Elapsed time from the physical event to query availability | Measured at load, not at event | Hours |
| Freshness | Pipeline-level proxy for currency, usually load time | Asserted in a dashboard instead of measured | Timestamp delta |
| Uniqueness | One record per entity per declared grain | Enforced by deleting rows | Duplicate rate |
| Duplicate | A record representing an entity already represented at that grain | Applied to near-duplicates without a matching rule | Count |
| Entity resolution | Deciding which records refer to the same real-world entity | Treated as exact key matching | Match rate, false-merge rate |
| Accuracy | Agreement with a reference accepted as true | Claimed without naming the reference | Proportion correct |
| Syntactic accuracy | Agreement with a dictionary, format, or type constraint | Reported as accuracy | Proportion |
| Semantic accuracy | Agreement with the real-world fact the value describes | Assumed once syntax passes | Proportion |
| Reference truth | An independent source accepted as correct for the comparison | The system's own "verified" flag | Source identifier plus date |
| Plausibility | The value is possible under known physical or business constraints | Treated as accuracy | Violation rate |
| Fitness for purpose | Documented judgement that a dataset meets a decision's requirements | Treated as a score out of 100 | Signed requirement set |
| Quality gate | An automated check that blocks or warns at a pipeline boundary | Placed only at the end of the pipeline | Gate identifier |
| Quarantine | Holding area for records that failed a gate, with a reason code | Used as a delete | Table plus reason code |
| Data contract | An agreed, versioned specification between producer and consumers | Treated as a dashboard | Contract version |
| Service-level objective (SLO) | A target for a measurable service property, such as freshness | Confused with a contractual SLA | Target plus window |
| Tolerance threshold | The failure rate above which a gate blocks or a dimension fails | Set to zero, or to whatever passes today | Proportion |
| Critical data element (CDE) | An attribute whose error materially changes a decision | Every column declared critical | Element identifier |
| Sentinel value | A placeholder encoding a meaning other than its literal value | Counted as a real observation | Value plus mapping |
| Survivorship bias | Records absent from the dataset are excluded from its denominator | Invisible in every computed metric | Share of expected records |
| Measurement paradox | A quality metric computed from the data it audits | Unrecognised, so never corrected | None |
| Root cause | The earliest point at which the defect could have been prevented | The last system in which it surfaced | Category code |

Five deserve expansion, because they cause the most damage when confused.

**Validity versus accuracy.** Validity asks whether a value belongs to the permitted set.
Accuracy asks whether it is true. A carrier identifier can be a perfectly formed, currently
active code belonging to a real carrier that did not move the shipment. Validity 100%, accuracy
0% for that record. Every organisation reporting "data quality" from validation rules alone is
reporting validity and calling it accuracy.

**Completeness at four grains.** One 10,000-row table can be 98.9% complete at cell level, 95.8%
at the grain of its most important column, 89.7% at row level, and 96.7% at record level against
the population it should represent. None is wrong. A requirement that says "completeness must
exceed 99%" without stating the grain cannot be tested.

**Timeliness, currency, latency.** Currency is how old the newest record is. Latency is how long
one event takes to become queryable. Timeliness is whether the data arrived in time for the
decision. A dataset can be current, low-latency, and untimely, if the decision is made at 06:00
and the load completes at 06:30.

**Plausibility versus accuracy.** Plausibility checks whether a value is possible: a 40-tonne
pallet, a delivery before dispatch, a negative transit time, a temperature excursion on a frozen
lane. It catches gross errors cheaply and passes plausible falsehoods silently. It is a screen,
not a measurement of truth.

**Fitness for purpose versus a score.** A score aggregates incommensurable dimensions with
weights that are usually arbitrary. Fitness for purpose is a requirement set written by the
decision owner: which dimensions, at which grain, against which threshold, with which
consequence. Longer to write, far more useful.

## 5. The mechanics, step by step

### 5.1 The five-part specification

A dimension becomes usable when it is written in five parts. Anything less is a wish.

1. **Entity and grain.** What is one row? One shipment, one shipment event, one order line, one
   item-location-week? The grain determines the denominator and therefore the number.
2. **Rule.** The deterministic test, written so a record either passes or fails. "Carrier
   identifier resolves to a carrier active on the ship date" is a rule. "Carrier data is
   reliable" is not.
3. **Denominator.** The population over which the proportion is computed, including records
   absent from the table. This is where survivorship bias enters.
4. **Threshold.** The value at which the dimension passes or fails, derived from the decision's
   tolerance for error rather than from last month's performance.
5. **Consequence.** What happens when the threshold is crossed: block the load, warn the owner,
   annotate the output, escalate. A rule with no consequence is documentation, not control.

Write all five before writing any code. Teams that start with the rule and add the denominator
later almost always compute a number over the wrong population.

### 5.2 The eight dimensions as computable tests

| Dimension | Computable test | Numerator / denominator | Typical threshold | What a pass does not prove |
|---|---|---|---|---|
| Completeness | Required field populated with a non-placeholder value | Populated cells ÷ expected cells (grain stated) | 95–99.5% by use case | That the expected records exist |
| Validity | Value belongs to domain, type, range, or active code list | Passing values ÷ values present | 99–100% per rule | That the value is true |
| Consistency | Two representations of one fact agree | Agreeing pairs ÷ comparable pairs | 98–99.9% | That either representation is right |
| Timeliness | Record available before the decision cut-off | Records on time ÷ records | 95–99% within SLO | That the record is complete or correct |
| Uniqueness | One record per entity per declared grain | Distinct entities ÷ rows | 99.5–100% | That near-duplicates were caught |
| Accuracy | Value agrees with an accepted reference | Correct units ÷ sampled units | Use-case specific | Anything beyond the sampled frame |
| Plausibility | Value satisfies physical and business constraints | Passing records ÷ records | 99%+ | That a plausible value is true |
| Fitness for purpose | All of the above meet the decision's requirement set | Requirements met ÷ requirements | All critical requirements | That the decision itself is right |

The thresholds are starting points from practice, not standards. Section 8 explains why no
standard supplies them.

**Completeness.** Two tests, usually conflated. The cell test counts populated cells against
expected cells. The record test counts present entities against expected entities, where the
expected count comes from an independent source: the order table, the carrier manifest, the
ledger. The record test catches extract filters, partition drops, and late partitions, which are
the failures that damage models; the cell test appears on dashboards because it is easy.

Placeholder handling is the second trap. A null is visible. A zero in a nullable quantity column,
a `-1`, a `1900-01-01` date, `N/A`, `UNKNOWN`, or a whitespace character are all missing values
that pass a null check. Normalise sentinels before computing completeness, and version the
mapping, because the set grows with every new source.

**Validity.** Validity is per rule, and rules do not compose. A record can pass forty rules and
still be nonsense, because each check tests one field in isolation. Report it per rule with
identifier and version; an aggregate is meaningless once rules have different denominators.

The most common supply chain validity failure is not a malformed value but a stale code list.
Carrier, plant, and reason codes live in a reference system that changes on its own schedule, so
a shipment coded to a carrier merged into another entity six months ago is syntactically valid
and referentially dead. Version the list, join on validity dates, and test the join, not the
format.

**Consistency.** Three tests. Cross-field: a delivered status needs a delivery timestamp, and
delivered quantity must not exceed shipped quantity beyond tolerance. Cross-system: WMS and ERP
on-hand should agree within the timing difference between cut-offs. Temporal: closing inventory
equals opening plus receipts minus issues plus adjustments, and the residual is a direct measure
of consistency failure. The trap in all three is comparing representations that are not
comparable; a 14:00 WMS snapshot and a 23:59 ERP snapshot differ by a day of transactions, and
that difference is not a defect.

**Timeliness.** Measure the share of records available before the cut-off and the latency
distribution at median and p95. The mean is useless because latency is heavy-tailed; four hours
on average can hide a 5% tail of two days. Compute latency from the source event timestamp, not
the ingest timestamp: the pipeline's clock is the thing under test and cannot be the instrument.

**Uniqueness.** Three phenomena hide under the word duplicate. Exact key duplicates are the same
primary key twice, usually from a retried load without an idempotency key. Business-key
duplicates are different surrogate keys for the same real shipment, detectable only by a matching
rule. Entity duplicates are different records for the same real entity, such as one supplier
under two legal names; that is entity resolution, not a constraint. A database constraint handles
the first, deletes the second keeping whichever record arrived first, and cannot see the third.

**Accuracy, plausibility, fitness.** Accuracy is the only dimension requiring something outside
the data, treated in 5.3. Plausibility encodes what is physically or commercially possible —
transit times by lane and mode, quantities by vehicle capacity, ordered dates, temperatures
within tolerance — and catches gross errors cheaply while passing plausible falsehoods silently.
Fitness for purpose is written by the decision owner as a requirement set; a dataset meets it or
does not, and a score of 94 out of 100 says nothing about that.

### 5.3 Measuring accuracy when there is no reference truth

Accuracy compares a value to a reference accepted as true. When no reference exists there are
four substitutes. Each answers a different question and has a characteristic blind spot.

| Substitute | How it works | What it can establish | What it cannot establish |
|---|---|---|---|
| External reference | Compare to an independent system of record, counterparty document, or registry | Agreement with that source | That the source is right; both may share a cause |
| Cross-source agreement | Compare independent sources; treat agreement as evidence | Convergence; disagreement as a flag | Truth when both derive from one upstream entry |
| Physical audit sample | Count, weigh, or inspect a random sample | Accuracy on the sampled population, with an interval | Accuracy outside the frame; rare errors |
| Outcome proxy | Compare a data-derived prediction to a later realised outcome | Predictive usefulness | Correctness of individual values |

Two design rules follow. **State the reference and its independence**: "accurate" without a named
reference is a claim about nothing, and if the reference is the same system's verification flag,
the measurement is circular. **Sample randomly and report the interval**: an estimate from 200
randomly drawn records carries roughly plus or minus 3 percentage points at the 95% level when
the estimate is near 95%, whereas an estimate from 200 records a clerk already reviewed carries no
interpretable error, because the sample was selected on the outcome.

### 5.4 The measurement paradox

Every quality metric is computed from data, and if that data is produced by the pipeline it
audits, the metric inherits the pipeline's defects. Five mechanisms recur.

1. **Survivorship in the denominator.** Completeness is computed over rows that arrived. Rows
   that failed ingestion cannot be counted as missing, so the metric describes the population
   that did not need checking.
2. **The auditing clock.** Latency is computed from the pipeline's ingest timestamp. If the job
   assigning the timestamp is delayed, the timestamp is late by the same amount and measured
   latency shrinks. The instrument is calibrated by the fault.
3. **The self-referential verification flag.** Accuracy is computed against a `verified` flag set
   by the process that entered the data. Records entered wrongly and flagged verified count as
   correct.
4. **Rule-derived dimensions hiding rule failure.** Uniqueness is computed on the key the
   deduplication step already applied. That step cannot find the duplicates it missed, so the
   reported rate is zero by construction.
5. **Aggregation over a defective grain.** A defect that removes whole entities leaves no trace
   in any row-level metric, because the rows are gone. Only a comparison to an independent
   expected count detects it.

Four countermeasures are worth building. **Instrument outside the audited path**: event times
from the source system, counts from the source extract, expected records from a system
independent of the pipeline. **Reconcile to an independent count**: every load compares its row
count to an upstream expectation, and a variance outside tolerance blocks regardless of how
complete the survivors look. **Seed known-bad records** into a test environment and confirm the
gates catch them; a gate that has never failed has never been tested. **Audit the auditor**:
recompute one dimension by hand from source documents for a sample, and treat disagreement as a
defect in the metric, which is more dangerous than a defect in the data because it is invisible.

The paradox cannot be eliminated, only acknowledged. The practical rule: **never report a quality
metric without naming the population it was computed over and the system that produced that
population.**

### 5.5 Dimensional trade-offs

Every control moves the operating point. The table names the trade and who bears the cost.

| Trade-off | Mechanism | Who bears the cost |
|---|---|---|
| Uniqueness versus completeness | Deduplication deletes records; the survivor may be wrong | The entity that disappears |
| Validity versus completeness | Out-of-range values nulled or rejected; real exceptions lost | The planner who needed the exception visible |
| Timeliness versus completeness | Publishing before late records arrive | The analyst who reconciles a number that later changes |
| Consistency versus local autonomy | One code list forced across countries or units | The local team whose legal code disappears |
| Accuracy versus cost | Each further point of verified accuracy costs more than the last | The budget owner |
| Precision versus recall | Stricter tolerance cuts false positives and raises false negatives | The team that stops seeing small defects |

The last trade is why gates get disabled. A 0% tolerance in a process generating 0.5% exceptions
blocks every load, and within two weeks someone adds an override. The override is permanent,
undocumented, and the gate is gone. Set the threshold where the gate catches what matters and the
organisation can still operate, and treat every override as a defect report against the rule
rather than against the operator.

The optimum is not 100% quality. Haug, Zachariassen, and van Liempd (2011) formalise this: the
cost of poor quality falls as quality rises, the cost of assurance rises, and total cost is
minimised short of perfection. A quality programme therefore needs a stopping rule, and the
stopping rule is economic.

### 5.6 The quality process

Quality is a five-stage loop: detect, triage, root-cause analysis, remediate, prevent. Skipping a
stage converts the loop into a recurring incident.

**Detect.** Gates, profiling, and reconciliation produce a defect register: what failed, where,
when, how many records, which rule version. The register is a dataset with its own grain (one row
per rule evaluation per batch), retention, and owner.

**Triage.** Not every defect deserves a fix. Triage classifies by decision impact: does this
change a number someone acts on, and by how much? A 2% null rate in a display field is not a 2%
null rate in the field that allocates cost. Triage is where programmes fail, because it requires
knowing what the data is used for.

**Root cause.** Assign each defect to the earliest layer where it could have been prevented.
Fixing at the detection layer is expensive and temporary.

| Layer | Typical causes | Fix location |
|---|---|---|
| Capture | Free-text entry, no mandatory fields, defaults, workarounds | Form design, validation at entry, lookup |
| Interface | Schema drift, truncation, encoding, units, time zone | Mapping spec, contract test, schema registry |
| Transformation | Join fan-out, wrong grain, deprecated filter, coercion | Code review, grain tests, reconciliation counts |
| Reference data | Stale code list, unversioned hierarchy, merged entities | Versioned reference data with validity dates |
| Process and timing | Batch cut-offs, late approvals, retroactive amendments | Process redesign, event capture, cut-off policy |
| Design | No idempotency key, mutable history, no quarantine | Schema and pipeline redesign |

**Remediate.** Fix at the earliest feasible layer, backfill what has already propagated, and
record what was backfilled and when. Backfills change history and therefore every model trained
on it. Tag the affected period so a later analyst can distinguish a level shift from a repair.

**Prevent.** Add the rule to the gate set, add a regression test, assign an owner. A fix without
a regression test has a scheduled recurrence.

### 5.7 Pipeline quality gates

Six gate types, in the order they should run. Each is cheap relative to the next, which is why
the order is not arbitrary.

| Gate | What it checks | Typical test | Default action |
|---|---|---|---|
| Schema | Column presence, type, nullability, order | Comparison against a contract registry | Block |
| Freshness | Currency and latency against SLO | Max event timestamp, p95 load lag | Warn, then block after two misses |
| Volume | Row count against expected range | Count within ±X% of trailing median, reconciled upstream | Block on large deviation |
| Distribution | Shape stability of key measures | Quantiles, KL divergence, Kolmogorov–Smirnov test | Warn |
| Referential integrity | Foreign keys resolve | Join to dimension, expecting zero orphans | Block on critical keys |
| Business rules | Cross-field and cross-table logic | Delivered implies dated; receipts within shipment | Block or quarantine by rule class |

Three design rules make gates survive operations. **Place the gate at the boundary, not the end**:
a schema change caught at ingestion costs an hour; caught after scoring, a day and a retraining
cycle. **Quarantine, do not delete**: failed records go to a table with rule identifier,
evaluation timestamp, and raw payload; deleting them destroys root-cause evidence and flatters the
completeness metric. **Separate blocking from warning by reversibility**: block when the
downstream decision is expensive to reverse and the defect is unambiguous; warn when the defect
is a distributional shift that might be legitimate. Blocking on a warn-class check is the fastest
route to an override culture.

### 5.8 Setting thresholds from the decision backwards

A threshold is not a target. It is the error rate at which the decision changes.

1. State the decision and the smallest change in it that matters. A replenishment decision that
   moves in pallets of 60 does not care about a 1% quantity error.
2. Propagate the tolerance through the calculation. If the decision moves at a 5% input change and
   the calculation multiplies four independent inputs, each can tolerate roughly 1.2% before the
   product moves 5%.
3. Respect the process's natural variation. A threshold tighter than normal variation fires
   constantly without improving anything.
4. Set the threshold at the tighter of the decision tolerance and the achievable level, and
   record which of the two bound it.

A threshold derived this way can be defended in review. A threshold copied from a vendor template
cannot.

## 6. Worked example

A third-party logistics provider maintains `fact_shipment_event`, used for carrier scorecards,
on-time-in-full (OTIF) reporting, and a lead-time model. All figures are illustrative and
internally consistent.

### 6.1 The dataset

| Property | Value |
|---|---|
| Rows | 10,000 |
| Columns | 9: `shipment_id`, `order_id`, `carrier_id`, `dc_id`, `ship_date`, `delivery_date`, `qty_shipped`, `qty_delivered`, `status` |
| Cells | 90,000 |
| Distinct `shipment_id` | 9,905 |
| Snapshot date | 2026-09-30 |
| Expected shipment headers from the order system, same period | 10,240 |
| Latency SLO | 4 hours from event to queryable |

Five defects were seeded, plus two dimensions measured without a row-level defect. Each defect
corresponds to a failure mode from section 5 rather than to random noise.

### 6.2 The five seeded defects

| ID | Defect | Rows | Dimension | Plausible cause |
|---|---|---:|---|---|
| D1 | `carrier_id` is null | 420 | Completeness | Spot shipments entered without the carrier lookup |
| D2 | `carrier_id` not in the carrier master for the ship date | 240 | Validity (referential) | Carrier merged into another entity; list not versioned |
| D3 | `status = 'DELIVERED'` and `delivery_date` is null | 610 | Consistency (cross-field) | Proof-of-delivery scan not mandatory in the driver app |
| D4 | `delivery_date < ship_date` | 180 | Plausibility | Time-zone conversion applied to one timestamp only |
| D5 | Duplicate `shipment_id` rows from a retried load | 95 extra | Uniqueness | Non-idempotent load |

| Dimension | Measurement | Result |
|---|---|---|
| Timeliness | 7,800 of 10,000 rows queryable within 4 hours; p95 latency 26 hours; max 71 hours | 78.0% within SLO |
| Accuracy | Random sample of 200 shipments matched to proof-of-delivery documents; 11 disagree on delivered quantity by more than ±2% | 189 / 200 = 94.5% |

D4 is both a consistency failure (two timestamps disagree about one timeline) and a plausibility
failure (the event is impossible). The label is a convention. The test is what matters, and it
must be written once and owned by one team.

### 6.3 Scoring each dimension

**Completeness, four grains.**

```text
cell level:    missing cells = 420 (carrier_id) + 610 (delivery_date) = 1,030
               (90,000 - 1,030) / 90,000 = 88,970 / 90,000 = 98.86%
column level:  (10,000 - 420) / 10,000 = 9,580 / 10,000 = 95.80%   (carrier_id)
row level:     (10,000 - 1,030) / 10,000 = 8,970 / 10,000 = 89.70%
record level:  9,905 present / 10,240 expected = 96.73%
```

Four numbers, one table, no arithmetic error, a 9.2-point spread. A dashboard reporting 98.9%
and a planner reporting that a tenth of the records are unusable are both describing this table.

**Validity.** The rule is "`carrier_id` resolves to a carrier active on `ship_date`". Null
carrier identifiers are excluded from the denominator because completeness owns them; the
exclusion is documented, not silent.

```text
values present = 10,000 - 420 = 9,580
failures = 240
validity = (9,580 - 240) / 9,580 = 9,340 / 9,580 = 97.49%
```

Against the full table the same defect reads as 2.40% invalid, or 97.60% valid. Both are correct
and answer different questions. The per-value denominator is defensible for a validity claim;
the full-table one is defensible for a business impact estimate.

**Consistency, plausibility, uniqueness, timeliness.**

```text
consistency (delivered implies dated): 610 / 10,000 -> 93.90%
plausibility (delivery >= ship):       180 / 10,000 -> 98.20%
uniqueness:   9,905 distinct / 10,000 rows -> 99.05%; 1.0096 records per shipment
timeliness:   7,800 / 10,000 within 4 h -> 78.0%; median 1.8 h, p95 26 h, max 71 h
```

The median says the pipeline is fast. The p95 says one shipment in twenty is more than a day
late. For a decision made at a fixed cut-off, the p95 is the number that matters, because the
tail is the population that crosses the cut-off.

**Accuracy.** The estimate and its interval, using the normal approximation for a proportion:

```text
p = 189 / 200 = 0.945
standard error = sqrt( p (1 - p) / n ) = sqrt( 0.945 x 0.055 / 200 )
               = sqrt( 0.000260 ) = 0.0161
95% interval  = 0.945 +/- 1.96 x 0.0161 = 0.945 +/- 0.0316
              = 94.5% +/- 3.2 pp, i.e. 91.3% to 97.7%
```

A finite-population correction for 10,000 records narrows this to about ±3.1 points and does
not change the reading. The honest statement is "between 91% and 98%, on a random sample of 200,
against proof-of-delivery documents". The dishonest one is "94.5%".

### 6.4 The composite score, and why not to compute it

The seven scores are 98.86, 97.49, 93.90, 98.20, 99.05, 78.00, and 94.50.

```text
sum  = 98.86 + 97.49 + 93.90 + 98.20 + 99.05 + 78.00 + 94.50 = 660.00
mean = 660.00 / 7 = 94.29
```

A composite of 94.3 out of 100 looks like a pass. It is not a measurement, for three reasons.
The dimensions are not commensurable: a point of timeliness and a point of validity have nothing
in common. The mean hides concentration: if 78% of undated deliveries belong to the two
batch-file carriers, the table is adequate for six carriers and unusable for two. And the grain
choice moves the answer:

```text
660.00 - 98.86 + 89.70 = 650.84 ; 650.84 / 7 = 92.98
```

A 1.3-point swing from a definitional choice most reports do not state. The defensible
alternative is a requirement set per decision with pass or fail on each critical item. The
carrier scorecard needs record completeness ≥ 99%, uniqueness = 100%, and carrier attribution
accuracy ≥ 99%. This table fails all three, whatever its composite.

### 6.5 Business impact and remediation priority

Impact is estimated per defect per month from the decision it disturbs. Every assumption is
stated so a reader can replace it.

| Defect | Mechanism of harm | Assumption | Monthly impact |
|---|---|---|---:|
| D5 duplicates | Double-counted volume in scorecards; duplicate freight invoices | 35% reach invoicing; average freight €180 | 33 × 180 = €5,940 |
| D3 consistency | 610 undated deliveries counted as on-time; audit of 100 shows 41% actually late | Service recovery €120 per misclassified late shipment | 610 × 0.41 × 120 = €30,012 |
| D1 completeness | 420 shipments cannot be allocated to a cost centre | €22 per manual resolution | 420 × 22 = €9,240 |
| D2 validity | 240 shipments attributed to a merged carrier; chargebacks misallocated | €38 per manual correction | 240 × 38 = €9,120 |
| D4 plausibility | 180 impossible transits contaminate the lead-time model | €15 review per record; model error unquantified | 180 × 15 = €2,700 |
| Timeliness | 2,200 late events; planners act on a stale exception list | 8% cross a dispatch cut-off and need premium freight at €310 | 176 × 310 = €54,560 |

Total quantified: `5,940 + 30,012 + 9,240 + 9,120 + 2,700 + 54,560 = €111,572 per month`, about
€1.34 million annualised.

> **Note:** This total is an upper bound, not an estimate. Duplicates inflate the denominators
> used for consistency and timeliness, so the per-defect figures overlap, and the assumptions are
> invented for teaching. Present each defect separately with a range and the total only as "not
> more than."

The OTIF distortion is worth isolating because it is the number management sees. Of 10,000 rows,
9,390 have a delivery date. Of those, 704 are late, giving a reported OTIF of
`(9,390 - 704) / 9,390 = 92.50%`. Adding the estimated 250 late shipments hidden among the
undated rows:

```text
true late = 704 + 250 = 954
true shipments (deduplicated) = 9,905
true OTIF = (9,905 - 954) / 9,905 = 8,951 / 9,905 = 90.37%
overstatement = 92.50 - 90.37 = 2.13 percentage points
```

Two points of OTIF is larger than most annual service programmes deliver, and it was produced
entirely by a missing timestamp.

### 6.6 Priority is impact divided by effort

| Rank | Defect | Impact/month | Effort | Lead time | Rationale |
|---:|---|---:|---|---|---|
| 1 | D3 consistency | €30,012 | Low: mandatory date on the delivered transition; backfill 610 | 2 weeks | Largest impact per unit of effort; also removes the OTIF distortion |
| 2 | D1 completeness | €9,240 | Low: default carrier at order creation; block entry without lookup | 1 week | Cheap; removes recurring manual cost |
| 3 | D2 validity | €9,120 | Low: version the carrier list; add a referential gate | 2 weeks | Same root cause as D1; fix together |
| 4 | Timeliness | €54,560 | High: replace batch file with event push; streaming ingest | 8–10 weeks | Largest impact, largest effort; start now, deliver later |
| 5 | D5 uniqueness | €5,940 | Medium: idempotency key; dedupe rule with a documented survivor | 4 weeks | Also repairs every other metric's denominator |
| 6 | D4 plausibility | €2,700 | Low: cross-field check with quarantine | 1 week | Do it, but it changes no decision this quarter |

Ranking by impact alone puts timeliness first and delays the two fixes that pay for themselves
within a month. Ranking by effort alone produces a fortnight of easy fixes and no structural
change. The sequence above starts the expensive item immediately and ships the cheap ones
alongside it, which is the only order that survives a budget review.

### 6.7 What the analyst should have asked

1. Is the 10,240 expected count complete, or does it exclude cancellations? If cancellations are
   excluded there, record completeness is overstated, because some of the 335 missing shipments
   were cancelled.
2. Are the 95 duplicates exact copies or do they differ in one field? If they differ, one is a
   correction and deduplicating by arrival order may keep the wrong one.
3. Is the 41% late rate among undated deliveries from a random subsample or the hundred easiest
   to audit? The second overstates the effect.
4. Does the 4-hour SLO apply to all carriers? If two are structurally incapable of meeting it,
   78% describes integration architecture rather than data quality.
5. Was the proof-of-delivery sample drawn from the same period as the table? Accuracy measured on
   last quarter's documents says nothing about this month.

Each question tests whether a number measures what it claims. None requires a model.

## 7. Code, query, or calculation

This structured query language (SQL) fragment computes six of the seven scores from the table in
section 6. It assumes PostgreSQL syntax and a Type 2 carrier dimension with validity dates. The
point is that each score is one aggregate over one denominator, and that the denominators differ.

```sql
with base as (
    select e.*, (c.carrier_id is not null) as carrier_valid
    from wh.fact_shipment_event e
    left join wh.dim_carrier c
           on c.carrier_id = e.carrier_id
          and e.ship_date between c.valid_from and c.valid_to
), s as (
    select count(*) as n_rows,
           count(distinct shipment_id) as n_shipments,
           count(*) filter (where carrier_id is null) as null_carrier,
           count(*) filter (where carrier_id is not null and not carrier_valid) as bad_carrier,
           count(*) filter (where status = 'DELIVERED' and delivery_date is null) as no_pod,
           count(*) filter (where delivery_date < ship_date) as impossible,
           count(*) filter (where load_lag_hours <= 4) as within_sla,
           percentile_cont(0.95) within group (order by load_lag_hours) as p95_lag
    from base
)
select round(100.0 * (1 - null_carrier::numeric / n_rows), 2) as completeness_pct,
       round(100.0 * (1 - bad_carrier::numeric  / n_rows), 2) as validity_pct,
       round(100.0 * (1 - no_pod::numeric       / n_rows), 2) as consistency_pct,
       round(100.0 * (1 - impossible::numeric   / n_rows), 2) as plausibility_pct,
       round(100.0 * n_shipments / n_rows, 2)                 as uniqueness_pct,
       round(100.0 * within_sla / n_rows, 1)                  as timeliness_pct,
       round(p95_lag, 1)                                      as p95_lag_hours
from s;
```

Four decisions in that query deserve challenge. The **validity denominator excludes nulls**, so
completeness and validity do not double-count a record; drop that condition and validity absorbs
the missing values, making both scores wrong in opposite directions. The **join is date-bounded**,
which makes a merged carrier visible as invalid rather than valid forever. **Uniqueness is
`n_shipments / n_rows`**, not the inverse; inverting it yields a number above 100%, the most common
error in published duplicate metrics. And **`load_lag_hours` must come from source event time**;
from ingest time, the query reproduces the paradox in 5.4.

Accuracy cannot be computed here. It has no denominator inside the table, which is the point of
section 5.3.

## 8. Data, standards, and regulatory context

Several standards define what a dimension is. None defines the threshold you should use, which
is why every implementation still makes a local judgement.

### 8.1 The two ISO quality models

| Standard | What it defines | Structure | What it does not give | Grade |
|---|---|---|---|---|
| ISO/IEC 25012:2008, *Data quality model* | A general model for structured data | 15 characteristics, inherent and system-dependent | Measures, thresholds, or a method for choosing among them | B |
| ISO/IEC 25024:2015, *Measurement of data quality* | Quantitative measures for the 25012 characteristics | Measure elements, target entities, application guidance | Rating levels or target values; explicitly context-dependent | B |

The 15 characteristics are broader than the six-dimension list most teams use. Accuracy,
completeness, consistency, credibility, and currentness are treated as inherent. Accessibility,
compliance, confidentiality, efficiency, precision, traceability, and understandability are both
inherent and system-dependent. Availability, portability, and recoverability are
system-dependent. In supply chain work, traceability and credibility carry more weight than their
position suggests, because a lot-traceability requirement is a completeness requirement at event
grain.

> **Note:** ISO/IEC 25012 was reviewed and confirmed as current in 2025. Cite the edition and the
> confirmation date together; a claim that the standard is "being replaced" is not supported by
> the public record as of 2026.

### 8.2 The ISO 8000 series

| Standard | Scope | Useful idea | Grade |
|---|---|---|---|
| ISO 8000-1:2022, *Data quality — Overview* | Scope, principles, structure of the series | Quality as fitness for purpose, with a "path" rather than a checklist | B |
| ISO 8000-8:2015, *Concepts and measuring* | How to plan and perform measurement | Three levels: syntactic, semantic, pragmatic | B |

The syntactic, semantic, and pragmatic distinction is the most portable idea in the series.
Syntactic quality is conformance to a grammar or schema. Semantic quality is that the value means
what it should. Pragmatic quality is that the data is useful for the intended purpose. Most
tooling stops at syntactic. Most supply chain failures occur at semantic. Most business
complaints are about pragmatic. ISO 8000 and ISO/IEC 25012 come from different technical
committees and are complementary; ISO 8000-1:2022 maps the relationship explicitly.

### 8.3 Data quality for analytics and machine learning

| Standard | Scope | Relevance | Grade |
|---|---|---|---|
| ISO/IEC 5259-2:2024, *Data quality for analytics and ML — Part 2: Measures* | Measures for data used in analytics and machine learning | Builds on ISO 8000, 25012, and 25024; adds representativeness, balance, and diversity | B |

Cite this one when a model rather than a report is the consumer. It adds the dimensions that
matter when a dataset is used to generalise rather than describe: representativeness of the
population, balance across classes, diversity of sources. A model trained on a non-representative
history is defective in a way no completeness check detects.

### 8.4 The practitioner frameworks

| Framework | Dimensions | Status | Grade |
|---|---|---|---|
| DAMA-DMBOK 2 (2017), Chapter 13 | 8: completeness, uniqueness, validity, timeliness (currency), consistency, integrity, accuracy, reasonableness | Canonical reference text | C |
| DAMA-DMBOK revised maintenance release | 9: adds currency, renames reasonableness | The set with general agreement | C |
| DAMA UK working group whitepaper | 6: completeness, uniqueness, timeliness, validity, accuracy, consistency | Source of the widely repeated "DAMA six dimensions" | C |

The whitepaper is often more useful for teaching because it gives a formal definition, measures,
units, and a worked scenario per dimension, and because it states plainly that accuracy requires
primary research or trusted reference data and cannot be assessed from the data alone. That
sentence is the most important line in the practitioner literature on this topic.

Its definitions, paraphrased: completeness is the proportion of stored data against a "100%
complete" requirement; uniqueness is the inverse of duplication; timeliness is the degree to
which data represent reality at the required point in time, measured as a time difference;
validity is conformance to the syntax of the definition; accuracy is the degree to which data
correctly describe the real-world object or event; consistency is the absence of difference when
comparing representations against a definition. Note what is absent: plausibility, and fitness for
purpose. The first is sometimes subsumed under validity and sometimes separated. The second is
not a dimension in any framework, because it is the frame that selects the dimensions.

### 8.5 Coding systems that define validity

| Coding system | Governs | Why versioning matters | Grade |
|---|---|---|---|
| ISO 8601 | Date and time representation, including offsets | A date without an offset is ambiguous across a network | B |
| ISO 3166-1 | Country codes | Codes are reassigned and retired; joins need validity dates | B |
| ISO 4217 | Currency codes | Minor unit changes and redenominations break stored decimals | B |
| UN/LOCODE | Port, airport, and inland point codes | Subdivisions and status changes alter the code set | B |
| GS1 GTIN | Trade item identification | A pack-size change creates a new GTIN; reuse corrupts history | B |
| GS1 GLN | Party and location identification | Merged legal entities leave stale identifiers | B |

The practical implication: a validity test is a join to a versioned reference table, not a
regular expression. Regular expressions test syntax. Joins test membership. Only the second is
validity in the sense that matters.

### 8.6 What standards do not supply

No standard here supplies a threshold. ISO/IEC 25024 is explicit that rating levels are
context-dependent. ISO 8000-1 frames quality as fitness for purpose rather than conformance to a
fixed scale. So the claim "our data quality is 98%, which meets the ISO standard" is not a
meaningful sentence. There is no ISO threshold to meet.

Regulation constrains the data rather than the dimension. Article 10 of the EU Artificial
Intelligence Act ties data governance obligations for high-risk systems to training, validation,
and test sets, framed around relevance, representativeness, and completeness, with an explicit
examination for biases. That is a requirement on the data and on its documentation, not a numeric
threshold. (Grade B; consult the consolidated text and applicable timeline before relying on any
date.)

## 9. Industry practice and case evidence

### 9.1 The economics: an optimum short of perfection

Haug, Zachariassen, and van Liempd (2011), "The costs of poor data quality," *Journal of
Industrial Engineering and Management* 4(2), 168–193, argues that perfect data quality should not
be the goal. It models two curves: the cost of poor quality, falling as quality rises, and the
cost of assurance, rising. Total cost is minimised at an interior point. It then classifies the
costs of poor quality on two axes, direct versus hidden and operational versus strategic, and
illustrates the classification with a single case: an automotive spare-parts manufacturer.

The case reports hidden strategic costs estimated at 5–7% of total fixed costs and duplicate
pricing work equal to two full-time marketing salaries. Grade A for the method and framework; the
magnitudes are a single self-estimated case, so grade C for the numbers. The durable contribution
is the shape of the argument: a quality programme without a stopping rule spends more than the
errors cost.

### 9.2 The number everyone quotes

The most cited figure in the field is that poor data quality costs organisations an average of
$12.9 million per year.

| Property | Detail |
|---|---|
| Publisher and year | Gartner, 2020 |
| Source document | Magic Quadrant for Data Quality Solutions, 27 July 2020 |
| Sample | Approximately 154–157 reference customers of 16 data quality vendors |
| Method | Self-reported estimate by respondents |
| Grade | D |

Three problems follow. **Selection bias**: respondents are customers of data quality vendors,
so by construction they are organisations that already believe the problem is large enough to buy
software for it. **Self-report**: the figure is a respondent's estimate, not a measured loss, and
no counterfactual is constructed. **Construct drift**: Gartner's own published figures across
years include roughly $8 million (2009), $14.2 million (2013), $9.7 million (2017), $15 million
(2017, a different survey), and $12.9 million (2020). Different samples, questions, and years. The
sequence is not a trend and should never be plotted as one.

The figure is usable for one purpose: establishing that the problem is large enough to warrant
attention. It is not usable as an input to a business case, because it is not about your
organisation, your data, or your decisions.

### 9.3 Statistics that should not be quoted

| Claim | Actual provenance | Grade | Use |
|---|---|---|---|
| "Bad data costs the US economy $3.1 trillion per year" | A 2011 trade-magazine extrapolation, described by its author as possibly far off | E | Do not use |
| "Poor data quality costs 15–25% of revenue" | An individual expert's estimate, not a study | E | Do not use |
| "Analysts spend 80% of time cleaning data" | A survey about task time, simplified and misattributed | E | Do not use |
| "Only 3% of companies' data meets basic quality standards" | Vendor survey with an undisclosed sampling frame | D | Do not use |

The pattern is consistent: a number with a plausible order of magnitude, detached from its
sample and its question, repeated until it acquires authority. The defence is procedural. Ask for
the sample, the question, the year, and the construct. If any is missing, the number is an
anecdote.

### 9.4 What teams actually run

The dominant open-source pattern for pipeline checks is an expectation suite: declarative
assertions about a dataset, evaluated as a batch, with results written to a report. Great
Expectations is the most widely deployed implementation. Its documentation describes expectations
for table shape and row counts, column existence and type, nullability and uniqueness, value
ranges and sets, and distributional comparisons using KL divergence, bootstrapped
Kolmogorov–Smirnov tests, and chi-square tests. Suites run at pipeline stages as checkpoints.
(Grade D: vendor documentation for a commercial open-source product. The pattern is real; the
outcome claims are marketing.)

Three practitioner observations, offered as judgement rather than evidence. The `mostly`
parameter — which lets a check pass when a stated proportion of records satisfy it — is the most
important feature in the tool, because it is the only way to run a rule in a process with known
exceptions; a tool without a tolerance produces a rule set that gets overridden. Most teams stop
at schema and volume, which catch the loud failures; the expensive defects here — missing proof of
delivery, stale carrier codes, impossible transit times — are business-rule and referential
checks, the ones most often absent. And a gate that has never failed has never been tested;
seeding a known-bad record into staging is a five-minute exercise that separates a working control
from a decorative one.

### 9.5 Building a local estimate instead

The defensible alternative to a quoted aggregate is a bottom-up estimate: count the incidents per
period from the defect register, attach a unit cost taken from the ledger or the contract rather
than an industry average, and multiply by the share that actually reaches a decision. That third
factor is the one people forget, and it is usually the largest source of overstatement. Then
present a range and list what is excluded rather than omitting it.

The worked example follows this method and produces a number roughly two orders of magnitude
smaller than the Gartner figure, for one table in one organisation. That is not a contradiction.
It is what happens when the estimate is about a specific decision rather than an average
organisation.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Completeness reported at cell grain only | Dashboard green; analysts say the data is unusable | A cell-level average dilutes a defect concentrated in required fields | Models trained on absent entities | Report all four grains; threshold at the decision grain |
| Sentinel values counted as data | Null rate suspiciously low; spikes at 0, −1, 1900-01-01 | Placeholders pass null checks | Wrong means, trends, models | Normalise sentinels; version the mapping |
| Survivorship in the denominator | Completeness improves after a pipeline change that dropped records | Absent records are not counted as missing | Metric improves as data worsens | Reconcile counts to an independent source |
| Validity reported as accuracy | 100% valid; users still find wrong values | Rules test membership, not truth | False confidence; no accuracy budget | Name the reference; sample for accuracy |
| Duplicate rule keeps the wrong record | Row count falls; totals shift by more than the duplicate count | Survivor chosen by arrival order | Lost corrections; understated quantities | Document the survivor rule; inspect before enforcing |
| Freshness measured at load time | "Updated 5 minutes ago" while planners say it is yesterday's data | Pipeline clock used as the instrument | Late decisions presented as current | Measure from event time; report p95 |
| Consistency compared across cut-offs | A stable, explained gap treated as a defect | Snapshots at different times | Noise treated as error; real errors missed | Align cut-offs; compare like with like |
| Gate threshold set to zero | Rule disabled within a month | Any exception blocks the load | Control disappears silently | Set tolerance from impact; log overrides as defects |
| Composite score averaged across dimensions | One number on the cover; no requirement met | Incommensurable dimensions equally weighted | False pass; wrong investment | Replace with a requirement set per decision |
| Metric computed inside the audited path | Latency improves when the pipeline is late | The instrument shares the fault | Systematic understatement | Instrument outside; audit the auditor |
| Accuracy from a convenience sample | Accuracy suspiciously high; no frame documented | Reviewed records are easier to check | Overstated accuracy; bad labels | Random sampling; report the interval |

Three deserve a note.

**Sentinel values** are the most under-reported defect in supply chain data because they make the
metric look better. A quantity column filled with zeros for "not recorded" produces a spike that
models learn as real demand. The fix is mundane and must be maintained: a mapping from sentinel to
meaning, applied before profiling, with the mapping under version control.

**Survivorship in the denominator** is the mechanism behind most "our data quality is improving"
claims that coincide with a decline in usefulness. A filter that removes 4% of records improves
every row-level metric and removes the population the model needed.

**Composite scores** are the most politically durable failure on the list, because they are
convenient. A single number can be reported to a steering committee and compared across quarters.
That convenience is exactly why it survives, and exactly why it should be replaced with a short
requirement set a decision owner has signed.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| A reference exists or can be constructed | Accuracy is defined against truth | No independent source; only the system's own flag | Name the reference; if internal, call the measure consistency |
| The declared grain matches the decision's grain | Every score depends on it | Grain changes between source and use | Recompute one dimension at two grains |
| The expected-record count is itself complete | Record completeness needs a denominator | Upstream count excludes cancellations or intercompany | Reconcile the denominator to a third source |
| Code lists are versioned with validity dates | Validity joins depend on it | Reference data overwritten in place | Attempt a historical join; check orphans |
| Event timestamps come from the source system | Latency from ingest is circular | The pipeline assigns the only timestamp | Inspect the lineage of the timestamp column |
| The sample is representative | Accuracy intervals require random sampling | Convenience sample or anomalous period | Document the frame; compare sample to population |
| Defects are independent enough to cost separately | Impact estimates sum per defect | One defect inflates another's denominator | Present per-defect figures; cap the total |
| Quality is stable over the measurement window | Trends need a stable definition | Rule version changed mid-period | Version rules; annotate the series at every change |
| The decision owner can state a tolerance | Thresholds derive from it | Nobody owns the decision, or it is contested | Ask for the smallest change that alters an action |
| Fixing at source is feasible | Prevention is the only durable remedy | Vendor-controlled system; contractual constraint | Map which layers you control before promising a fix |

Two limits of the framing are worth stating plainly.

**The eight dimensions are a vocabulary, not a checklist.** No dataset needs all eight measured
exhaustively. A programme that measures everything produces a dashboard nobody reads. The useful
discipline is to name the critical data elements, measure the two or three dimensions that matter
for each, and document the rest as known and accepted.

**Quality measurement is always a sample of a moving process.** Even an exhaustive scan measures
one instant, and the process that produced it changes continuously. A score is a reading, not a
property. The useful question is not "what is the quality" but "what changed, and does it matter
for the decision this week."

## 12. Exercises

### Exercise 1 — Write the five-part specification (design, 45 minutes)

Choose a table you have worked with and write a five-part specification for each of the six
classical dimensions.

**Deliverable:** six specifications, plus one sentence per dimension naming the decision it
protects.
**Acceptance criteria:** every rule is deterministic and could be coded as written; every
denominator names a population including absent records where relevant; every threshold is
justified by a decision tolerance rather than a round number; at least one consequence is "block"
and one is "warn."

### Exercise 2 — Completeness at four grains (calculation, 30 minutes)

A table has 50,000 rows and 12 columns, 8 of them required. Missing cells across required columns:
3,400. Rows with at least one missing required field: 2,900. The worst required column is missing
1,900 values. The source system reports 52,000 records for the period; the table holds 49,300
distinct business keys.

**Deliverable:** cell, column, row, and record completeness with the arithmetic shown.
**Acceptance criteria:** four correct percentages; one paragraph on which number the decision
should use; one sentence naming a decision for which each of the other three would be right.

### Exercise 3 — Accuracy without a reference (design, 40 minutes)

For a dataset of supplier lead times with no independent source, design an accuracy measurement
using at least two of the four substitutes in section 5.3.

**Deliverable:** a one-page design stating the reference, sampling frame, sample size, metric, and
interval.
**Acceptance criteria:** the reference is named and its independence argued; the frame is random
and its construction described; the design states what the result cannot prove; the sample size is
justified, not asserted.

### Exercise 4 — Audit the measurement paradox (critique, 45 minutes)

Take a quality dashboard you have seen, or construct one from a public data-quality report, and
identify every metric whose inputs are produced by the process it audits.

**Deliverable:** a table of metric, audited population, instrument, and failure mode from the five
in section 5.4.
**Acceptance criteria:** at least three metrics; one countermeasure per metric that moves the
instrument outside the audited path; each specific enough to implement.

### Exercise 5 — The deduplication decision (design, 40 minutes)

A shipment table has 0.9% duplicate business keys. The duplicates differ in one field: the later
record has a corrected delivered quantity. Enforcing uniqueness on the business key keeps the
earlier record.

**Deliverable:** a rule for which record survives, the predicate that implements it, and the
effect on completeness and totals.
**Acceptance criteria:** the rule names the fields used to choose the survivor; the effect on row
count and summed quantity is quantified; the choice is defensible to an auditor who asks why the
other record was discarded.

### Exercise 6 — Design the gate set (design, 60 minutes)

For a daily shipment feed, design the six gates from section 5.7, stating the test, threshold,
action, and owner for each.

**Deliverable:** a gate specification table plus a paragraph on the quarantine path and its
retention.
**Acceptance criteria:** schema and volume block; distribution warns; at least one gate uses a
tolerance rather than zero; every gate names an owner role; the quarantine table has a reason code
and a review cadence.

### Exercise 7 — Local cost of poor quality (calculation, 45 minutes)

Using section 6.5, recalculate the monthly impact if the share of late-arriving events requiring
an expedite rises from 8% to 12% and average freight rises from €180 to €240. Then present the
total as a range with those two assumptions varied independently by ±25%.

**Deliverable:** the recalculated figures, the range, and a paragraph on what is excluded.
**Acceptance criteria:** correct arithmetic; the range computed from the stated variation rather
than guessed; the exclusion list includes at least two unquantified costs.

### Exercise 8 — Threshold from tolerance (application, 40 minutes)

A replenishment decision orders in pallets of 60 units. The order quantity is a forecast
multiplied by a service factor and divided by a lead-time term with four inputs.

**Deliverable:** a validity and completeness threshold for each input, derived from the pallet
granularity.
**Acceptance criteria:** the derivation is shown, not asserted; it acknowledges that independent
errors combine sub-additively; at least one threshold is looser than current practice, with a
justification.

### Exercise 9 — Root cause and prevention (analysis, 50 minutes)

Take the five defects in section 6.2. For each, assign the earliest prevention layer from the
taxonomy in section 5.6, name the specific change at that layer, and name the regression test.

**Deliverable:** a five-row table with defect, layer, change, regression test, and owner role.
**Acceptance criteria:** no two defects share a fix; each fix is at the earliest feasible layer
rather than the detection layer; each regression test would fail if the defect recurred.

## 13. Self-check answers

**Exercise 2 arithmetic.**

```text
cells = 600,000 ; required cells = 400,000
cell level:   (600,000 - 3,400) / 600,000 = 596,600 / 600,000 = 99.43%
column level: (50,000 - 1,900) / 50,000 = 48,100 / 50,000 = 96.20%
row level:    (50,000 - 2,900) / 50,000 = 47,100 / 50,000 = 94.20%
record level: 49,300 / 52,000 = 94.81%
```

The spread is 5.2 points, from 99.43% to 94.20%. The decision should use the row level if it
consumes whole records, which most models do. The cell level is right for a rule needing one
field, such as a freight allocation keyed on weight. The column level is right for a single-field
join. The record level is right for a process that assumes every expected entity exists, such as
a demand history that must cover every item-location. A submission reporting only 99.43% and
calling the dataset complete has made the error the exercise exists to expose.

**Exercise 7 arithmetic.**

```text
timeliness, base:    2,200 x 8%  x €310 = 176 x 310 = €54,560
timeliness, revised: 2,200 x 12% x €310 = 264 x 310 = €81,840   (+€27,280)
duplicates, base:    95 x 35% x €180 = 33 x 180 = €5,940
duplicates, revised: 95 x 35% x €240 = 33 x 240 = €7,920        (+€1,980)
new total = 111,572 + 27,280 + 1,980 = €140,832 per month
```

For the range, vary the expedite share between 6% and 10% and freight between €135 and €225,
independently:

```text
timeliness low  = 2,200 x 0.06 x 310 = 132 x 310 = €40,920
timeliness high = 2,200 x 0.10 x 310 = 220 x 310 = €68,200
duplicates low  = 95 x 0.35 x 135 = 33 x 135 = €4,455
duplicates high = 95 x 0.35 x 225 = 33 x 225 = €7,425
other defects   = 30,012 + 9,240 + 9,120 + 2,700 = €51,072
range low  = 40,920 + 4,455 + 51,072 = €96,447
range high = 68,200 + 7,425 + 51,072 = €126,697
```

So roughly €96,000 to €127,000 per month, with a point estimate near €112,000. Excluded: model
error from contaminated training data, managerial reconciliation time, and decisions deferred
because the number was not trusted. List those; do not estimate them.

**Exercise 3 rubric.** A strong answer names a reference independent of the system measured —
supplier-confirmed dispatch dates from a transport management system, or a sample of delivery
receipts — and argues why independence is plausible. It defines a random frame, justifies sample
size from the desired interval width, and reports the interval with the point estimate. It states
the blind spot: agreement cannot detect an error common to both sources, and a sample cannot bound
rare errors. A weak answer compares to historical averages (that is consistency) or samples
records operations already flagged.

**Exercise 5 rubric.** A strong answer chooses the survivor by a documented rule — latest
correction timestamp, or the record with a proof-of-delivery reference — and quantifies the effect:
row count falls by 95, summed quantity changes by the difference between original and corrected
values, and row-level completeness may change because the discarded record may have had a field
the survivor lacks. It keeps the earlier record in a history table with a superseded flag rather
than deleting it. A weak answer deduplicates on `shipment_id` with `row_number() = 1` and no
ordering rationale.

**Exercise 8 rubric.** A strong answer works backwards from the pallet of 60. If the current order
is around 600 units, one pallet is 10% of it, so an input error below roughly 3% cannot move the
rounded order at all when four inputs contribute. That gives thresholds in the 1–3% range, looser
than the 0.5% often asserted. A weak answer asserts 99.9% because it sounds rigorous, without
connecting the threshold to the rounding step.

**Exercise 9 rubric.** A strong answer assigns D1 to capture and D2 to reference data, then notices
they share a root cause — no carrier lookup at entry and an unversioned list — so one change fixes
both. It assigns D3 to process (the driver app permits a delivered status without a scan), D4 to
interface (time-zone conversion on one field), and D5 to design (no idempotency key). Each
regression test fails on recurrence: a null-carrier count, an orphan join count, a status-date
rule, a negative-transit rule, a duplicate-key count on reload. A weak answer fixes everything in
the reporting layer.

## 14. Interview questions

**1. "What does data quality mean?"**

A strong answer refuses the intrinsic framing: quality is the degree to which data meets stated
requirements for a stated use, so one dataset can be fit for one decision and unfit for another.
A weak answer recites a dimension list with no mention of use or grain.

**2. "How would you measure completeness?"**

A strong answer asks "at what grain?", gives cell, column, row, and record with an example of how
far apart they can be, and notes that record completeness needs an independent expected count. A
weak answer says "percentage of non-null values" and stops.

**3. "What is the difference between validity and accuracy?"**

Validity is membership in a declared domain or code list; accuracy is agreement with a reference
truth. A syntactically perfect, currently active carrier code on the wrong shipment is valid and
inaccurate. A weak answer treats them as synonyms.

**4. "You have no ground truth. How do you measure accuracy?"**

A strong answer offers the four substitutes — external reference, cross-source agreement,
physical audit sample, outcome proxy — states each blind spot, and insists on a random sample
with an interval. A weak answer says "you can't," which is unhelpful in practice.

**5. "What is the measurement paradox in data quality?"**

A strong answer: the metric is computed from the data it audits, so it inherits the defects, as
when latency is measured from the pipeline's own clock or completeness over rows that survived
the filter. Countermeasures: instrument outside, reconcile to an independent count, seed
known-bad records, audit the auditor. A weak answer reaches for Goodhart's law without a mechanism.

**6. "Why not enforce uniqueness with a database constraint and be done?"**

A strong answer separates exact key, business-key, and entity duplicates: a constraint handles the
first, deletes the second keeping an arbitrary survivor, and cannot see the third. It proposes a
documented survivor rule and a history table. A weak answer treats deduplication as solved.

**7. "A vendor says poor data quality costs $12.9 million a year. How do you use that?"**

A strong answer identifies it as a Gartner 2020 survey of data quality vendors' reference
customers, self-reported, so it establishes salience but not a business case. It proposes a local
bottom-up estimate. A weak answer quotes it or dismisses it without knowing what it is.

**8. "Your completeness dashboard is green but analysts say the data is unusable. What happened?"**

A strong answer offers survivorship (the bad records never arrived), grain mismatch (a cell-level
average hiding a required-field defect), and sentinel values counted as data. A weak answer says
the analysts are wrong.

**9. "How do you set a quality threshold?"**

Derive it from the decision: the smallest change that alters an action, propagated through the
calculation, bounded below by the process's natural variation. A zero-tolerance gate gets
disabled. A weak answer picks 99% because it is conventional.

**10. "What is the difference between timeliness, currency, and latency?"**

Currency is the age of the newest record, latency is event-to-queryable time, and timeliness is
whether the data arrived before the decision cut-off. A dataset can be current, low-latency, and
untimely. A weak answer uses the three interchangeably.

**11. "When is a composite quality score acceptable?"**

Almost never: the dimensions are not commensurable, the weights are arbitrary, and the average
hides concentration. Propose a requirement set per decision with pass or fail on each critical
item. A weak answer defends a weighted average without examining the weights.

**12. "How would you convince a team to fix data at the source rather than in the report?"**

Quantify the recurring cost of the report-layer fix, show that the defect returns each period,
and propose a regression test plus an owner. If the source is outside the team's control, document
the workaround and escalate the constraint. A weak answer appeals to engineering virtue.

## 15. Key terms

- **Data quality** — fitness of data for a stated use, not an intrinsic property.
- **Dimension** — a named, measurable aspect of quality; useless without a grain and a rule.
- **Grain** — the business definition of one row; determines every denominator.
- **Completeness** — share of expected cells, rows, or records present and usable.
- **Sentinel value** — a placeholder such as 0, −1, or 1900-01-01 that encodes missingness.
- **Validity** — conformance to a declared domain, type, range, or versioned code list.
- **Code list** — a controlled vocabulary, usually externally maintained and versioned.
- **Referential integrity** — every foreign key resolves to an existing parent record.
- **Consistency** — agreement between two representations of the same fact.
- **Temporal consistency** — agreement of a state with its own history, as in reconciliation.
- **Timeliness** — data available before the decision cut-off.
- **Currency** — age of the newest record in a dataset.
- **Latency** — elapsed time from the source event to query availability.
- **Freshness** — a pipeline-level proxy for currency, usually load time.
- **Uniqueness** — one record per entity per declared grain.
- **Entity resolution** — deciding which records refer to the same real-world entity.
- **Accuracy** — agreement with a reference accepted as true.
- **Reference truth** — the independent source against which accuracy is measured.
- **Gold record** — the survivor of entity resolution, not necessarily the correct record.
- **Plausibility** — the value is possible under physical or business constraints.
- **Fitness for purpose** — documented judgement that a dataset meets a decision's requirements.
- **Measurement paradox** — a quality metric computed from the data it audits.
- **Survivorship bias** — absent records excluded from the denominator, improving the score.
- **Quality gate** — an automated check that blocks or warns at a pipeline boundary.
- **Quarantine** — a holding area for failed records with a reason code, not a delete.
- **Data contract** — a versioned specification agreed between a producer and its consumers.
- **Tolerance threshold** — the failure rate at which a gate blocks or a dimension fails.
- **Critical data element (CDE)** — an attribute whose error materially changes a decision.
- **Root cause layer** — the earliest point at which a defect could have been prevented.

## 16. Sources

**Standards and official**

- ISO/IEC 25012:2008, *Software engineering — SQuaRE — Data quality model*. 15 characteristics,
  split into inherent and system-dependent. https://www.iso.org/standard/35736.html (Grade B)
- ISO/IEC 25024:2015, *Measurement of data quality*. Measures for the 25012 characteristics;
  explicitly does not set rating levels. https://www.iso.org/standard/35749.html (Grade B)
- ISO 8000-1:2022, *Data quality — Part 1: Overview*. Scope, principles, structure.
  https://www.iso.org/standard/81745.html (Grade B)
- ISO 8000-8:2015, *Concepts and measuring*. Syntactic, semantic, and pragmatic levels.
  https://www.iso.org/obp/ui (Grade B)
- ISO/IEC 5259-2:2024, *Data quality for analytics and ML — Part 2: Measures*. Preview via ANSI.
  https://webstore.ansi.org (Grade B)
- European Union, Artificial Intelligence Act, Article 10, data governance for high-risk systems.
  Consult the consolidated text and applicable timeline. (Grade B)

**Peer-reviewed and technical**

- Haug, A., Zachariassen, F., and van Liempd, D. (2011). "The costs of poor data quality."
  *Journal of Industrial Engineering and Management*, 4(2), 168–193.
  DOI: 10.3926/jiem.2011.v4n2.p168-193 (Grade A for method; case magnitudes Grade C)
- Talha, M., Elmarzouqi, N., and Abou El Kalam, A. (2020). "Towards a Powerful Solution for Data
  Accuracy Assessment in the Big Data Context." *IJACSA*, 11(2).
  DOI: 10.14569/ijacsa.2020.0110254 (Grade A)
- *Data Quality Assessment: Challenges and Opportunities*. arXiv:2403.00526 (2024). Five-facet
  framework; accuracy requires reference data. (Grade A−, preprint)
- NFDI4ING, "General quality metrics." Cell- and row-level completeness with the
  placeholder-normalisation caveat. https://quality.nfdi4ing.de (Grade C)

**Practitioner and vendor documentation**

- DAMA International, *DAMA-DMBOK*, 2nd edition (2017), Chapter 13, Data Quality. Eight dimensions.
  https://technicspub.com/wp-content/uploads/2023/07/DAMA-DMBOK2.pdf (Grade C)
- DAMA-DMBOK maintenance release, adding currency and renaming reasonableness for nine
  dimensions. https://www.damadmbok.org/dmbok2-revisions (Grade C)
- DAMA UK Working Group, *Defining Data Quality Dimensions*. Six dimensions with definitions,
  measures, and a worked scenario.
  https://www.sbctc.edu/resources/documents/colleges-staff/commissions-councils/dgc/dama-data-quality.pdf
  (Grade C)
- Great Expectations documentation, "Distributional Expectations," and the dimension series.
  Expectation suites, checkpoints, KL divergence, bootstrapped KS, chi-square.
  https://docs.greatexpectations.io (Grade D)

**Analyst and press (use with the caveats in section 9)**

- Gartner, *Magic Quadrant for Data Quality Solutions*, 27 July 2020. Origin of the $12.9 million
  figure; approximately 154–157 reference customers of 16 vendors, self-reported.
  https://www.gartner.com/en/data-analytics/topics/data-quality (Grade D)
- Liliendahl, H. (2017). "I am afraid that Gartner does not help." Critique of the $9.7 million
  figure. https://liliendahl.com/2017/03/21/i-am-afraid-that-gartner-does-not-help/ (Grade E)
- Docsumo, "Cost of bad data statistics." Sources the Gartner series, the $3.1 trillion
  extrapolation, and the 80%-cleaning claim.
  https://www.docsumo.com/blog/cost-of-bad-data-statistics (Grade D)
- Fry & Laurie Consulting, "Data Provenance of Statistics." Selection-bias critique of the Gartner
  sample. https://www.frylaurie.com/blog/data-provenance-of-data-stats (Grade D)

**Note on grading.** Grades follow Module 28: A peer-reviewed or audited, B regulatory or official,
C company case or framework documentation, D vendor or analyst marketing, E press, blog, or
anecdote. Preprints are marked A− and treated as peer-review-pending. Survey figures are reported
as the sources state them and have not been independently verified.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`07_access-and-licensing.md`](07_access-and-licensing.md)
- Next topic: [`09_master-data-drift.md`](09_master-data-drift.md)
- Related: [`03_transaction-data.md`](03_transaction-data.md)
- Related: [`05_inventory-and-service.md`](../01_supply_chain_foundations/05_inventory-and-service.md)
- Survey file: [`../../../course/02_supply_chain_data_landscape.md`](../../../course/02_supply_chain_data_landscape.md)
- Return to the part map: [`../_index.md`](../_index.md)
