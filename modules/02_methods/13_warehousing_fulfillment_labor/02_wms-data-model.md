# Process Flows and the WMS Data Model

> Module 13 — Warehousing, Fulfillment, and Labor Analytics
> Topic 02 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 13, topic 01; basic SQL
> Version 2026.5

*Tasks, locations, inventory states, and timestamps that reveal bottlenecks.*

---

## 1. Why this topic exists

A warehouse management system (WMS) records instructions and confirmations for moving, storing,
counting, and changing the status of goods.
A manager sees that orders miss dispatch cut-off and asks where the time went.
A report says that the average pick took six minutes.
That number may omit the hour between the order's release and the first pick assignment.
It may also include travel, waiting for equipment, and a paused task without distinguishing them.
The report is not necessarily wrong; it answers a narrower question than the manager asked.

A useful warehouse data model connects physical flow to digital evidence.
It identifies what moved, where it was, what state it was in, which work was requested, and when each
observable transition occurred.
It must also distinguish a physical event from the time an integration recorded that event.
Without that distinction, a delayed scanner upload can look like a late worker or an old event can
make a current queue appear to grow.

Different people need different views of the same flow.
The receiving supervisor needs to know which receipts await inspection or put-away.
The inventory controller needs to reconcile the quantities held, available, damaged, and allocated.
The outbound lead needs to see work ready to pick, in progress, and complete but waiting for packing.
The operations analyst needs a defensible denominator and a complete cohort, not just closed tasks.
The engineer needs stable keys and event semantics before building a dashboard.
The site leader needs an intervention that can be checked against service and safety, not a ranking
of people based on an ambiguous timestamp.

The cost of a poor model is not limited to reporting effort.
A location master that treats a staging lane as a storage bin can hide congestion at the dock.
An overwritten task status destroys the queue history needed to explain a missed cut-off.
An inventory snapshot joined to every task can multiply quantities and suggest stock that does not
exist.
A timestamp without a zone can move an event into the wrong shift.
A task completion scan is not necessarily proof that a pallet physically reached its destination.
A queue that looks long because of missing closure messages needs a data repair, not more labor.

This topic builds an event-centered model of warehouse operations.
It covers the interfaces among receiving, quality control, put-away, storage, replenishment, picking,
packing, staging, loading, and shipping.
It develops a reproducible method for measuring waiting and processing intervals while preserving
open work and uncertainty.
It does not prescribe a particular WMS product, labor standard, slotting policy, or legal conclusion.

> **Note:** All operational records and business figures in the worked example are synthetic.
> They illustrate a method, not a measured result from a named facility.

## 2. Learning outcomes

After completing this topic you can:

1. **Draw** a warehouse flow with work queues, physical locations, and inventory state changes;
   distinguish a task from the goods and order lines it acts on.
2. **Specify** entity grains, keys, relationships, and effective dates for tasks, events, locations,
   inventory, and order lines.
3. **Reconstruct** a task lifecycle from immutable, deduplicated events without losing retries,
   cancellations, or open work.
4. **Calculate** queue time, execution time, elapsed time, inventory balances, and coverage using
   explicit units, boundaries, cohorts, and denominators.
5. **Diagnose** a candidate bottleneck using queue accumulation, capacity evidence, and downstream
   service, while separating observation from causal inference.
6. **Write** a runnable query that handles an open task at an as-of time and exposes invalid sequences.
7. **Evaluate** an event feed for clock, identifier, location-history, quantity, and privacy defects.
8. **Design** a practical validation and governance plan for a warehouse-facing metric.

Exercise 1 checks outcome 1.
Exercise 2 checks outcome 2.
Exercises 3 and 4 check outcomes 3 and 4.
Exercise 5 checks outcome 5.
Exercise 6 checks outcome 6.
Exercises 7 and 8 check outcomes 7 and 8.
Section 13 provides calculations or a rubric for each exercise.

## 3. The core mental model

### 3.1 Goods, work, and evidence are different things

A physical unit can exist before any task is created.
A task can be cancelled without moving the unit.
A scan can arrive after the physical movement, and an operator may perform an action that is never
scanned.
The model should not equate any pair of these objects automatically.

```text
Inbound appointment / purchase order
              |
              v
    receipt event -> inspection / hold -> put-away task -> storage location
                          |                    |                 |
                          v                    v                 v
                  inventory state      task event stream    balance snapshot
                                                                |
Order line -> allocation -> replenishment -> pick -> pack -> stage -> load -> ship
                                |             |       |       |       |
                              queue         queue   queue   queue   queue
                                \____________ event timestamps ____________/
                                               |
                                               v
                               cohort, interval, coverage, decision
```

The arrows show a common path, not a mandatory path in every facility.
Cross-docking can bypass storage.
A return may enter through a returns dock rather than receiving.
An order may be picked directly to a shipping carton, bypassing a separate packing task.
A quality hold may interrupt any eligible stage.
Map the actual WMS configuration and physical process rather than assuming the diagram is universal.

### 3.2 Three related state machines

The **work state** describes the instruction: created, ready, assigned, started, paused, completed,
cancelled, or otherwise configured by the site.
The **inventory state** describes the stock's eligibility or custody: for example available,
quality hold, damaged, or in transit.
The **order state** describes the customer commitment and fulfillment progress.
These state machines interact, but none is a substitute for another.
A completed pick task may move stock to a packing location while the order line remains open.
A quality release changes stock eligibility without necessarily moving it.
An order cancellation may cancel some tasks and leave others already physically executed.

Events are evidence of transitions between states.
The event type, object key, occurrence time, source, and payload are needed to interpret that evidence.
The current status is useful for executing work now, but a status overwritten in place cannot by
itself reconstruct how long the task waited last Tuesday.
An append-only event history preserves the individual observations and their corrections.

### 3.3 A queue is an interval, not a label

Define a queue by an entry event and an exit event at a specified process boundary.
For example, ready-to-start pick waiting time is the interval from `ready` to `started`.
Assignment can happen during that interval, so adding ready-to-assignment time and
assignment-to-start time is valid only when the intervals do not overlap.
For a task that pauses and resumes, active processing time is the sum of execution intervals, not
necessarily the difference between first start and final completion.
Elapsed execution includes pauses unless they are separately recorded.

A location can hold several kinds of work simultaneously.
A staging lane is a physical place; a pack queue is a process state.
The two align only when the local workflow guarantees that association.
Conversely, a virtual queue can exist in the WMS before goods have been moved to a dedicated area.

### 3.4 Evidence and inference

A large queue in front of packing is an observation.
A sustained queue, high packing utilization, and a lower completion rate than arrivals are consistent
with packing constraining that flow during the observed period.
They do not prove that packing caused the entire delay.
Order mix, a downstream trailer shortage, blocked labels, or incomplete scans may explain the pattern.
Test alternatives against independent evidence before changing staffing or promising a service gain.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| WMS | System coordinating warehouse inventory and work | Assumed to record every physical action | Product-specific configuration |
| Facility | Warehouse or operational site | Combined with another site sharing local identifiers | Facility ID and time zone |
| Process flow | Ordered or branching activities through which goods or work pass | Assumed identical for all order types | Versioned process map |
| Receipt | Record of goods accepted or observed at inbound | Equated with purchase order creation | Receipt ID and line |
| Put-away | Movement from inbound or intermediate area to a destination | Assumed complete when a task is merely created | Task ID and source/destination |
| Replenishment | Movement of stock to make a forward pick location usable | Treated as customer picking | Task type and quantity |
| Pick | Retrieval of stock for an order or batch | Assumed equivalent to shipping | Pick task and order-line link |
| Pack | Preparation and confirmation of outbound handling unit | Assumed present for every flow | Pack event or task |
| Staging | Holding goods for a subsequent movement or load | Treated as permanent storage | Location and entry/exit events |
| Ship | Confirmed handoff or dispatch as locally defined | Equated with label creation | Shipment ID and event |
| Order line | A requested product and quantity within an order | Assumed to map one-to-one to a task | Order ID plus line ID |
| Task | A work instruction with a defined lifecycle | Treated as the physical inventory itself | Facility ID plus task ID |
| Task attempt | One execution attempt or locally defined retry | Overwritten when reassigned or retried | Task ID plus attempt ID |
| Event | Recorded occurrence or transition | Assumed to be the entire task | Source event ID and event type |
| Event grain | What one event record represents | Changed mid-table without documentation | One source event version |
| Location | Identifiable physical or logical place | Used without site or history | Facility ID plus location ID |
| Location hierarchy | Relationships among zones, aisles, bins, doors, and other places | Assumed static over time | Effective-dated parent links |
| Inventory balance | Quantity associated with a defined stock key at a time | Summed across snapshots | Quantity in a stated unit |
| Inventory state | Eligibility, condition, or custody classification | Confused with physical location | Controlled status code |
| Lot | Traceability grouping assigned under a defined rule | Assumed unique without product context | Product ID plus lot ID |
| Serial number | Identifier of an individual serialized item | Treated as a quantity of an arbitrary SKU | Serial identifier and scheme |
| SKU | Stock keeping unit, a product identifier at a business-defined granularity | Assumed universally unique across owners | Owner and SKU ID |
| LPN | License plate number, an identifier for a handling unit in many WMSs | Assumed to be a GS1 identifier automatically | Site-specific or standardized identifier |
| Handling unit | Pallet, tote, carton, or other grouped physical unit | Assumed to contain a single SKU | Handling-unit ID |
| Allocation | Reservation or assignment of inventory to demand | Assumed a physical movement | Order line, inventory key, quantity |
| On hand | Quantity owned or controlled under a stated accounting rule | Assumed all available to promise | Units, explicit status scope |
| Available | Quantity eligible for a specified use after constraints | Defined identically in all WMS products | Units, formula and rule version |
| Hold | Status restricting an inventory use | Assumed to imply damaged stock | Hold reason and effective time |
| Event time | Time a source says an event occurred | Substituted with warehouse load time | Timestamp with offset or UTC |
| Ingest time | Time an analytical system received a record | Treated as physical action time | UTC timestamp |
| Processing time | Time an analytical transform processed a record | Used as a source event time | UTC timestamp |
| Local time | Clock reading in the facility's time zone | Used without daylight-saving rules | Zone identifier and offset |
| Ready time | Time work became eligible to start under a defined rule | Assumed equal to creation time | Timestamp and event definition |
| Start time | Observed start or first execution evidence | Assumed to be first physical touch | Timestamp and scanner semantics |
| Completion time | Observed task-complete transition | Assumed to be shipment time | Timestamp and completion semantics |
| Queue time | Time between specified queue entry and exit | Treated as travel or labor time | Minutes per task or unit |
| Elapsed execution | Time between start and completion, including unobserved pauses | Assumed active labor | Minutes per task |
| Active time | Sum of observed running intervals | Imputed from elapsed time without pause events | Minutes per task |
| Cycle time | Elapsed time across explicitly named boundaries | Used without boundaries | Minutes per named cohort |
| As-of time | Cutoff at which a state or interval is evaluated | Confused with report execution time | UTC timestamp |
| Right censoring | End of observation before the outcome occurs | Dropped so completed work looks representative | Open age at as-of time |
| Late event | Event received after its occurrence or reporting window | Always interpreted as a bad event | Ingest minus occurrence time |
| Duplicate | Repeated delivery of the same logical event | Equated with a legitimate repeated scan | Source-defined idempotency key |
| Correction | Source-authorized revision or reversal of an event | Silently overwritten with no history | Reference to original event |
| Bottleneck | Constraint limiting throughput for a defined flow and period | Assigned to the slowest individual task | Capacity, queue, and throughput evidence |
| Throughput | Completed flow across a boundary per period | Confused with work in process | Units, lines, or tasks per hour |
| Work in process | Items or work between defined entry and exit boundaries | Added across different grains | Units or tasks at a time |
| Cohort | Population selected by a common rule and time boundary | Changed between numerator and denominator | Entry, completion, or snapshot cohort |
| Coverage | Share of expected records with usable evidence | Measured only among records already captured | Percent with named denominator |
| Grain | Meaning of one row in a table | Assumed from its name | Written key and row definition |
| Slowly changing dimension | Historical dimension with effective versions | Joined to today's location attributes | Valid-from and valid-to timestamps |
| Idempotency | Reprocessing an event has no unintended additional effect | Implemented by dropping all matching timestamps | Stable source event key |
| Reconciliation | Comparison of independently derived quantities | Limited to row-count matching | Quantity and boundary-specific residual |

## 5. The mechanics, step by step

### Step 1. Start with a decision and a physical boundary

**Input:** A service problem, operating period, and people who know the actual workflow.
**Output:** A decision statement and a map with observable entry and exit points.

Ask what action the analysis could change.
An example is whether the site should move one trained packer from a low-volume activity to packing
for the final two hours before carrier cut-off.
The relevant flow is outbound standard orders in one facility, excluding returns and cross-dock orders.
The relevant outcome is shipment confirmation before the applicable cut-off, not merely a faster
individual scan.

Walk the path with the operations lead.
Mark where product physically waits, where the system creates work, where a scanner is used, and
where paper or manual exceptions occur.
Mark shared resources such as printers, dock doors, and material-handling equipment.
Ask what happens when the normal sequence is interrupted.
A queue definition must identify the population, entry event, exit event, unit, and time window.
For a pack queue, `pick_complete` to `pack_start` is plausible only if completed picks are eligible
for packing immediately and both events refer to the same handling unit or order line.

**Branch:** If no exit event exists, report entry volume and open age, not a fabricated completed
waiting-time distribution.
If the exit is observed only for some workflows, stratify and report coverage.

### Step 2. Define the entities and their grains

**Input:** Source-system tables, process map, and identifier documentation.
**Output:** A grain and key contract for every table.

Use separate facts for different kinds of evidence.
A practical logical model contains:

| Dataset | One row represents | Candidate key | Essential links |
|---|---|---|---|
| `task` | One logical work instruction | `facility_id, task_id` | Type, creation source, order or movement references |
| `task_event` | One source event or event version | `source_system, event_id, version` | Task, attempt, type, occurrence and ingestion times |
| `task_allocation` | One task-to-demand or task-to-stock association | Association ID or documented composite | Task, order line, stock key, quantity |
| `inventory_movement` | One inventory ledger posting leg | Movement ID plus leg ID | From/to keys, signed quantity, event |
| `inventory_snapshot` | One balance at one as-of time | Snapshot time plus complete stock key | Product, owner, location, state, lot, unit |
| `location_version` | One version of a location's attributes | Facility, location, valid-from | Parent, type, validity range |
| `order_line` | One requested product line | Facility, order, line | Product, requested quantity, service |
| `shipment_line` | One quantity on a shipment line | Shipment and line | Order line, handling unit, shipped quantity |

These are logical grains, not mandatory source table names.
A source may store multiple entities in one record or split one event across tables.
A transformation should preserve enough raw information to reverse its interpretation.
Use the real source key, including tenant or owner when local identifiers overlap.

A task may serve multiple order lines, and an order line may require multiple tasks.
Represent that many-to-many relationship in a bridge with quantities where the source supports them.
Do not join a task directly to all order lines and then sum task duration as if each resulting row
represented distinct work.
Likewise, a pallet can contain multiple products or lots and a single product can occupy many pallets.
A handling-unit identifier alone is not necessarily a complete inventory key.

**Branch:** If an association lacks quantities, use it for lineage only; do not allocate time or units
across order lines without an explicit allocation rule.
Keep unmatched records visible and quantify them.

### Step 3. Version the location master

**Input:** Location master, layout history, and task source and destination fields.
**Output:** An effective-dated location dimension and an unmapped-location report.

A location key should be unique within its facility and stable across renames where the source permits.
Keep the source code alongside a canonical identifier.
Record the type: receiving door, inspection lane, reserve rack, forward pick face, packing station,
staging lane, trailer, virtual location, or another locally defined class.
Record parent relationships such as building, zone, aisle, and bay only where known.
A location's physical dimensions and permitted handling types may matter for a capacity study, but
are not implied by its name.

Use half-open validity intervals `[valid_from, valid_to)` so that adjacent versions do not overlap.
Join an event to the version valid at the event's occurrence time.
If a location moved from reserve to forward pick on 1 October, a September event should retain the
September classification.
A current-only join silently rewrites historical flow.
Check for overlapping versions, gaps, recycled codes, inactive locations with later activity, and
unknown locations.
Maintain a distinct unknown member rather than deleting the affected events.

**Branch:** If historical layouts are unavailable, label the historical mapping as current-layout
approximation and do not claim exact historical zone attribution.

### Step 4. Define the inventory key and allowed states

**Input:** Inventory ledger and site-specific status definitions.
**Output:** A stock grain and a state-transition dictionary.

A typical stock key includes facility, owner, SKU, lot, serial if applicable, location, handling unit,
inventory status, and unit of measure.
Not every site uses every attribute, but omitted attributes must not distinguish balances in the source.
Two quantities with different owners or states are not interchangeable merely because they share a SKU.
Record the conversion between cases, inner packs, and eaches with effective dates.
Avoid converting quantities using a present-day pack size when the historic pack size differed.

Define whether each inventory status is eligible for allocation, picking, shipment, or inspection.
These rules are local; `available` in one system may mean something different elsewhere.
A quality hold can be physically located in an ordinary rack while unavailable for outbound work.
A transfer from available to hold may change state without changing total on-hand quantity.
A movement from one bin to another changes location without changing facility total, assuming both
locations are inside the same accounting boundary.
A shipment reduces the facility balance only at the business-defined posting point.

For a single stock key and period:

`closing = opening + receipts + transfers_in - transfers_out + adjustments`.

For the facility total, internal transfers cancel only when both legs are included and measured in the
same unit.
State changes can be represented as a negative posting to one state and a positive posting to another.
A missing leg creates a residual and must not be hidden by adding an invented adjustment.

**Branch:** If negative balances are allowed operationally, flag them and investigate timing or
posting rules rather than automatically deleting them.

### Step 5. Map the task lifecycle and attempts

**Input:** WMS event codes, workflow configuration, and operator interviews.
**Output:** Versioned mapping from source codes to analytical states and transitions.

First distinguish task creation from readiness.
Created work may be blocked by inventory, an order hold, a wave schedule, or an upstream dependency.
Ready work is eligible to be claimed under the agreed definition.
Assignment identifies a resource or queue, but may happen automatically before an operator can act.
A start scan may be a device interaction rather than the first physical movement.
Completion may require confirmation at a destination, or only a confirmation on a handheld.
Document the local semantics.

A sample lifecycle is:

```text
created -> blocked -> ready -> assigned -> started -> completed
                \       |          |          |
                 \      |          |          +-> paused -> resumed -> completed
                  \     |          +-> unassigned -> ready
                   \    +-> cancelled
                    +-> cancelled
```

Allow for a task to be reissued, split, merged, cancelled, or reopened.
A task ID may be reused by the source, so confirm whether an attempt identifier or lifecycle version
is necessary.
Preserve all state transitions and assignment changes.
A reassignment is not automatically a new task, and a retry is not automatically a duplicate.

Define terminal states separately for operational completion, cancellation, and error closure.
Do not count cancelled tasks as completed throughput.
If a cancelled task is replaced, link the replacement when evidence supports that relationship.
Measure repeated work separately if the business question concerns rework.

**Branch:** If a task only has current status and last-update time, a historical queue cannot be
reconstructed reliably; start capturing changes prospectively or obtain a system audit log.

### Step 6. Preserve event time, ingestion time, and sequence

**Input:** Raw event records with source identifiers, clocks, offsets, and integration metadata.
**Output:** An ordered, deduplicated event history with data-quality flags.

Persist source event time, source time zone or offset, ingestion time, source sequence if available,
and the original payload or a traceable reference to it.
Convert unambiguous timestamps to coordinated universal time (UTC) for arithmetic.
Retain the facility's named time zone to derive local shift and operating date.
A local wall-clock value during a daylight-saving fall-back hour can occur twice; an offset or other
reliable disambiguation is required.
A nonexistent spring-forward time requires a documented source-specific treatment.
Never infer an exact instant from an ambiguous local time without flagging the assumption.

Network delays can produce out-of-order arrival.
Order by occurrence time and a source sequence or other documented deterministic tie-breaker, not
by load order alone.
A tie-breaker makes a result reproducible; it does not prove which action happened first.
When two events cannot be ordered reliably, mark the sequence uncertain rather than asserting causality.

Use a stable source event identifier for idempotency.
Identical timestamps and types do not necessarily imply a duplicate: two real scans can happen within
one clock resolution.
Preserve source corrections and reversals with their lineage.
A late correction can revise prior metrics, so retain report version, data freshness, and as-of time.

**Branch:** If no trustworthy event key exists, develop a source-specific duplicate rule, measure its
false-match risk, and preserve raw records for review.

### Step 7. Reconstruct intervals without inventing work

**Input:** Ordered task events and the lifecycle mapping.
**Output:** One or more valid intervals per task attempt, plus exceptions.

For a simple attempt with one ready, one start, and one completion event:

- Ready-to-start queue minutes = `(start - ready)` in minutes.
- Start-to-complete elapsed minutes = `(complete - start)` in minutes.
- Ready-to-complete elapsed minutes = `(complete - ready)` in minutes.
- When all three timestamps are valid, queue plus elapsed execution equals total elapsed time.

The definitions are not universal.
A task may cycle between ready and blocked or be paused after starting.
Build each state interval from a transition into the state to the next valid transition out.
Sum nonoverlapping ready intervals for cumulative eligible waiting.
Sum running intervals for observed active time only if pause and resume events are complete.
If they are not, label start-to-complete as elapsed execution, including unknown pauses.
Do not call this worker hands-on time.

An open interval ends at the report's as-of time for an age calculation, not as a claim that the
work completed then.
Keep an `is_open` indicator and the original null end.
A cancelled interval ends at cancellation but is not a completed task.
If completion precedes start, retain and flag the invalid sequence; do not take an absolute value.

**Branch:** If the data only records assignment and completion, measure assignment-to-completion and
state what it contains; do not label it pure execution.

### Step 8. Select a cohort and handle incomplete work

**Input:** Valid intervals, reporting boundary, and decision question.
**Output:** Named population and coverage metrics.

A completion cohort contains work completed within the window.
It is useful for measuring what left a process, but excludes long-running tasks still open at the end.
An entry cohort contains work that became ready during the window and can be followed to a later
observation cutoff.
It is useful for understanding the experience of newly released work, but recent entries may be open.
A point-in-time snapshot contains work present at one instant, including tasks that entered earlier.
It is useful for queue management but overrepresents longer-lived work relative to arrivals.

Do not pool these cohorts or call their averages interchangeable.
For an entry cohort, report count completed, count cancelled, count open, and invalid or missing
interval count at the cutoff.
For open work, report age separately; a completed-only average is conditional on completion.
For a high-stakes distribution of time to completion, use a survival method that handles censoring
and competing cancellation when its assumptions are appropriate.
A task with no recorded completion is not proof it is physically incomplete.

State whether a day is a local operating day, a calendar day, or a shift.
If the report covers a period shorter than some task lifetimes, carry in existing work and carry out
open work explicitly.
A queue accounting identity is:

`ending open = starting open + entries - completions - cancellations - other exits`.

Every term must refer to the same boundary and unit.
Investigate residuals, including late arrivals and reclassifications.

### Step 9. Measure flow, stock, and service together

**Input:** Cohorts, inventory balances, shipment events, and effective-dated location context.
**Output:** A metric table with numerators, denominators, units, and coverage.

Useful task measures include ready-to-start minutes, start-to-complete elapsed minutes, open task age,
arrivals per hour, completions per hour, and counts by terminal state.
Useful stock measures include quantity by location and status, stock blocked by holds, and the age of
unresolved holds.
Useful service measures include order lines shipped on time and the share of orders with all required
lines shipped by the applicable commitment.
These quantities have different grains and must not be added together.

Report the distribution as well as the mean.
A median can describe a typical completed task while a high percentile reveals a small set of very
old tasks.
Always show the count and quantile method when reporting a percentile.
A mean can change because a small number of exceptions entered the cohort, because the workflow
changed, or because data coverage improved.
Compare like task types, zones, product characteristics, shifts, and time windows before attributing a
change to performance.

Calculate coverage against an independent expected population when possible.
For example, compare task events to the task master and shipped lines to order lines expected to ship.
A 100% completeness rate among events that arrived says nothing about events never produced.

### Step 10. Identify a candidate constraint, then test alternatives

**Input:** Arrival and departure rates, queue trajectory, resource availability, and service outcomes.
**Output:** A prioritized hypothesis and evidence needed to confirm or reject it.

A growing queue indicates arrivals exceed exits during the measured interval after accounting for
other entries and exits.
That is a flow imbalance, not by itself proof of a root cause.
Examine the queue immediately upstream and downstream of the suspected activity.
Look for sustained accumulation upstream, effective capacity near its limit, downstream starvation,
and a service effect in the same window.
Determine whether the supposed constraint is blocked by a downstream process: packing may stop
because finished cartons have nowhere safe to stage.

Compare a short, defined operational period with its own arrival pattern and order mix.
A daily average can hide a two-hour surge before carrier collection.
Cross-check scanner-derived states against a floor observation, equipment logs, staffing rosters, and
exception tickets when available.
A single observation can establish a plausible mechanism but not an average rate over a month.

**Branch:** If the queue is an integration artifact, repair the feed and recompute before changing
staffing.
If the evidence points to a real constraint, specify a reversible intervention, a guardrail, and a
comparison period.
Examples of guardrails are mis-pick rate, safety, overtime, and on-time shipment.

### Step 11. Reconcile and publish a contract

**Input:** Metric definitions, source-to-target mappings, and independent controls.
**Output:** A reproducible metric contract and an exception-handling process.

Reconcile inventory opening and closing balances with postings at both the stock-key and site levels.
Reconcile task lifecycle counts with the source task master.
Compare shipment quantities with shipment confirmations, after accounting for known timing differences.
A row-count match is insufficient when a duplicated quantity offsets a missing one.
Report residual quantities, missing IDs, invalid sequence counts, late-event rates, and the time
window affected.

For each published measure, name the owner, definition, grain, source, refresh cadence, time zone,
cutoff, known exclusions, and revision policy.
A dashboard should show its as-of timestamp and whether a number is provisional.
A late event may change a past report; retain the original publication and the revised version when
an audit requires both.
Give an operations owner a way to challenge a record and give a data owner responsibility for fixing
its cause.

## 6. Worked example

### 6.1 Decision and setting

A synthetic site, North DC, is investigating missed outbound cut-offs on 14 April 2026.
The decision is whether to test added packing capacity during the last hour before the 17:00 local
carrier cut-off.
This is an operating decision, not a forecast or a commitment to the customer.
All times below are local site times on the same date, with a consistent offset.
The example uses order lines as the outbound flow unit and tasks as a separate work unit.
No labor cost or monetary benefit is assumed.

The process map is pick completion, pack readiness, pack start, pack completion, and shipment.
The local WMS creates one packing task per line for this example.
That one-to-one relationship is an example assumption, not a general warehouse property.
A pack task becomes ready immediately after its associated pick completes.
All four example tasks are in the same order class and have compatible handling requirements.

### 6.2 Task event data

At the 17:00 as-of time the event reconstruction yields:

| Task | Ready | Start | Complete | Outcome at 17:00 |
|---|---|---|---|---|
| P101 | 15:00 | 15:12 | 15:20 | Completed |
| P102 | 15:05 | 15:35 | 15:47 | Completed |
| P103 | 15:10 | 16:00 | 16:10 | Completed |
| P104 | 15:15 | 16:20 | — | Open |

The task records have event IDs and a reliable ordering; no pauses are recorded.
No independent evidence establishes whether the fourth task is physically still in progress.
Its open status means that a completion event has not been observed by the cutoff.

### 6.3 Queue and execution arithmetic

For P101, ready-to-start is 12 minutes and start-to-complete is 8 minutes.
The ready-to-complete elapsed time is 20 minutes, equal to 12 + 8 minutes.
For P102, the corresponding values are 30, 12, and 42 minutes.
For P103, the values are 50, 10, and 60 minutes.
For P104, ready-to-start is 65 minutes.
At the as-of time its observed start-to-as-of age is 40 minutes.
That 40-minute age is censored, not a completed execution duration.
Its ready-to-as-of age is 105 minutes.

| Task | Ready-to-start, min | Completed elapsed execution, min | Ready-to-end or as-of, min | Censored? |
|---|---:|---:|---:|---|
| P101 | 12 | 8 | 20 | No |
| P102 | 30 | 12 | 42 | No |
| P103 | 50 | 10 | 60 | No |
| P104 | 65 | — | 105 | Yes |

All four started, so their mean ready-to-start wait is `(12 + 30 + 50 + 65) / 4 = 39.25` minutes
per started task.
The median ready-to-start wait is `(30 + 50) / 2 = 40` minutes.
The three completed tasks have mean elapsed execution `(8 + 12 + 10) / 3 = 10` minutes per completed
task.
Their mean ready-to-completion time is `(20 + 42 + 60) / 3 = 40.67` minutes per completed task,
rounded to two decimal places.
Do not compare 39.25 + 10 with 40.67 as an identity: the first average includes four tasks and the
other two include three.
Within the completed cohort, mean wait is `(12 + 30 + 50) / 3 = 30.67` minutes; adding 10 gives
40.67 minutes after rounding.

At 17:00, three of four tasks are completed and one is open.
The completed share is `3 / 4 = 75%` of this ready-entry cohort at the as-of time.
This is not a facility-wide on-time shipment rate: packing completion is not shipment confirmation.
If the open task is simply dropped, the reported average excludes the longest observed case and
cannot describe the full cohort.

### 6.4 Queue accounting

For the illustrative hour 16:00–17:00, suppose the site-wide pack queue has 48 lines waiting at 16:00,
65 eligible lines enter, 80 lines start packing, and 3 are cancelled before starting.
Then the number waiting at 17:00 is `48 + 65 - 80 - 3 = 30` lines.
The interval's arrival rate is 65 lines per hour and its start rate is 80 lines per hour.
The queue shrank by 18 lines over the hour, including three cancellations.
A rising wait in the four-task example does not establish a growing site-wide queue in that hour.
The four records describe a small earlier cohort, while the accounting includes all site-wide lines.
This distinction prevents a small sample from overturning a complete boundary count.

Suppose instead the preceding hour had 85 eligible arrivals, 60 starts, and no cancellations.
Its queue grew by `85 - 60 = 25` lines.
The two hours together show a time-dependent imbalance and recovery, not one constant operating rate.
The analyst should investigate whether the earlier surge, staffing, downstream blockage, or another
factor produced the buildup.

### 6.5 Inventory state reconciliation

Consider SKU A in units of each, with no other stock states in this small example.
At 15:00 the site has 120 available eaches and 20 quality-hold eaches, total 140 eaches.
During the period it receives 36 available eaches, places 8 available eaches on hold, releases 5
held eaches, and ships 24 available eaches.
There are no adjustments or other boundary movements.

| State | Opening eaches | Receipt | Placed on hold | Released | Shipped | Closing eaches |
|---|---:|---:|---:|---:|---:|---:|
| Available | 120 | +36 | -8 | +5 | -24 | 129 |
| Quality hold | 20 | 0 | +8 | -5 | 0 | 23 |
| Total | 140 | +36 | 0 | 0 | -24 | 152 |

The available balance is `120 + 36 - 8 + 5 - 24 = 129` eaches.
The hold balance is `20 + 8 - 5 = 23` eaches.
The total is `129 + 23 = 152` eaches, which also equals `140 + 36 - 24 = 152` eaches.
The internal state changes cancel at the site boundary.
If an observed snapshot instead contains 151 eaches, the residual is `151 - 152 = -1` each.
Investigate timing, missing postings, conversion, or a count adjustment; do not silently force the
snapshot to balance.

### 6.6 Interpretation and next action

Observed facts are increasing waits across four example tasks, a 75% completed share at the cutoff,
and a preceding hour in which arrivals exceeded starts by 25 lines.
The later full-hour queue shrank by 18 lines.
These facts justify examining packing around the surge; they do not establish that adding a packer
will improve shipment service.

A useful next comparison would observe several comparable surge periods, distinguish label-printer
outages and downstream staging blockage, and compare actual pack capacity with arrival timing.
If operational leaders approve a controlled trial, document the staffing change, training and safety
constraints, order mix, cut-off compliance, and quality guardrails.
The outcome of that trial is not present in this synthetic example.

## 7. Code, query, or calculation

The following example is runnable using Python 3 and its standard-library SQLite module.
It builds the four synthetic task records above, validates ordering, and calculates queue and elapsed
execution intervals at a fixed as-of time.
It intentionally represents one simple lifecycle per task: exactly one ready event, at most one start,
and at most one completion.
A production state-machine implementation must handle multiple ready intervals, pause/resume,
cancellation, attempts, correction lineage, and the other cases described in section 5.
The program flags extra or invalid events instead of silently treating them as the simple lifecycle.

All sample timestamps are UTC ISO-8601 strings with a `Z` suffix.
SQLite's `julianday` supports these inputs; arithmetic is rounded to two decimal places.
The fixed as-of time is 17:00 UTC on 14 April 2026.
For this illustration UTC is also the displayed site time.

```python
import sqlite3

AS_OF = "2026-04-14T17:00:00Z"

connection = sqlite3.connect(":memory:")
connection.row_factory = sqlite3.Row
connection.executescript("""
CREATE TABLE task (
    facility_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    PRIMARY KEY (facility_id, task_id)
);
CREATE TABLE task_event (
    event_id TEXT PRIMARY KEY,
    facility_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    FOREIGN KEY (facility_id, task_id)
        REFERENCES task (facility_id, task_id)
);
""")
connection.executemany(
    "INSERT INTO task VALUES (?, ?)",
    [("NORTH", task_id) for task_id in ("P101", "P102", "P103", "P104")],
)
connection.executemany(
    "INSERT INTO task_event VALUES (?, ?, ?, ?, ?)",
    [
        ("e01", "NORTH", "P101", "ready", "2026-04-14T15:00:00Z"),
        ("e02", "NORTH", "P101", "start", "2026-04-14T15:12:00Z"),
        ("e03", "NORTH", "P101", "complete", "2026-04-14T15:20:00Z"),
        ("e04", "NORTH", "P102", "ready", "2026-04-14T15:05:00Z"),
        ("e05", "NORTH", "P102", "start", "2026-04-14T15:35:00Z"),
        ("e06", "NORTH", "P102", "complete", "2026-04-14T15:47:00Z"),
        ("e07", "NORTH", "P103", "ready", "2026-04-14T15:10:00Z"),
        ("e08", "NORTH", "P103", "start", "2026-04-14T16:00:00Z"),
        ("e09", "NORTH", "P103", "complete", "2026-04-14T16:10:00Z"),
        ("e10", "NORTH", "P104", "ready", "2026-04-14T15:15:00Z"),
        ("e11", "NORTH", "P104", "start", "2026-04-14T16:20:00Z"),
    ],
)

query = """
WITH event_summary AS (
    SELECT
        t.facility_id,
        t.task_id,
        SUM(CASE WHEN e.event_type = 'ready' THEN 1 ELSE 0 END) AS ready_count,
        SUM(CASE WHEN e.event_type = 'start' THEN 1 ELSE 0 END) AS start_count,
        SUM(CASE WHEN e.event_type = 'complete' THEN 1 ELSE 0 END) AS complete_count,
        SUM(CASE WHEN e.event_type NOT IN ('ready', 'start', 'complete')
                 THEN 1 ELSE 0 END) AS other_count,
        MIN(CASE WHEN e.event_type = 'ready' THEN e.occurred_at END) AS ready_at,
        MIN(CASE WHEN e.event_type = 'start' THEN e.occurred_at END) AS start_at,
        MIN(CASE WHEN e.event_type = 'complete' THEN e.occurred_at END) AS complete_at
    FROM task AS t
    LEFT JOIN task_event AS e
      ON e.facility_id = t.facility_id
     AND e.task_id = t.task_id
     AND julianday(e.occurred_at) <= julianday(:as_of)
    GROUP BY t.facility_id, t.task_id
), checked AS (
    SELECT *,
        CASE
          WHEN ready_count <> 1 OR start_count > 1 OR complete_count > 1
            OR other_count > 0 THEN 'unsupported_lifecycle'
          WHEN ready_at IS NULL THEN 'missing_ready'
          WHEN complete_at IS NOT NULL AND start_at IS NULL THEN 'missing_start'
          WHEN start_at IS NOT NULL AND julianday(start_at) < julianday(ready_at)
            THEN 'invalid_order'
          WHEN complete_at IS NOT NULL AND julianday(complete_at) < julianday(start_at)
            THEN 'invalid_order'
          ELSE 'valid'
        END AS quality
    FROM event_summary
)
SELECT task_id, quality,
    CASE WHEN quality = 'valid' AND start_at IS NOT NULL
      THEN ROUND((julianday(start_at) - julianday(ready_at)) * 1440, 2)
    END AS queue_minutes,
    CASE WHEN quality = 'valid' AND complete_at IS NOT NULL
      THEN ROUND((julianday(complete_at) - julianday(start_at)) * 1440, 2)
    END AS completed_execution_minutes,
    CASE WHEN quality = 'valid' AND complete_at IS NULL
      THEN ROUND((julianday(:as_of) - julianday(ready_at)) * 1440, 2)
    END AS open_ready_age_minutes,
    CASE WHEN quality = 'valid' AND complete_at IS NULL THEN 1 ELSE 0 END AS is_open
FROM checked
ORDER BY task_id;
"""

for row in connection.execute(query, {"as_of": AS_OF}):
    print(dict(row))
connection.close()
```

Expected output:

```text
{'task_id': 'P101', 'quality': 'valid', 'queue_minutes': 12.0, 'completed_execution_minutes': 8.0, 'open_ready_age_minutes': None, 'is_open': 0}
{'task_id': 'P102', 'quality': 'valid', 'queue_minutes': 30.0, 'completed_execution_minutes': 12.0, 'open_ready_age_minutes': None, 'is_open': 0}
{'task_id': 'P103', 'quality': 'valid', 'queue_minutes': 50.0, 'completed_execution_minutes': 10.0, 'open_ready_age_minutes': None, 'is_open': 0}
{'task_id': 'P104', 'quality': 'valid', 'queue_minutes': 65.0, 'completed_execution_minutes': None, 'open_ready_age_minutes': 105.0, 'is_open': 1}
```

The output's queue is only available once a start has been observed.
For a valid ready-but-not-started task, use its open ready age, not a completed queue duration.
The query includes tasks with no events and marks them unsupported rather than removing them with an
inner join.
The `event_id` primary key demonstrates idempotency only for this small source; real feeds may require
source-system and version components as well.
Events after the as-of time are deliberately excluded so that the calculation reflects the historical
view, not today's eventual completion.
In a production pipeline also filter by knowledge or ingestion time when reproducing what a prior report
actually knew; event-time cutoff alone produces a restated historical view.

## 8. Data, standards, and regulatory context

### 8.1 Data contract

A WMS may export operational tables, event streams, database change records, flat files, or APIs.
None is automatically a complete physical event log.
For each feed document the producer, schema version, delivery frequency, event semantics, identity,
clock source, expected delay, retention, corrections, and outage recovery.
Agree whether the producer emits an event when a transition is reversed or merely overwrites a row.
Agree what a missing location, quantity, or actor means: unknown, not applicable, or not collected.

Separate raw, normalized, and analytical layers.
The raw layer retains source payload and receipt metadata under the organization's retention rules.
The normalized layer applies mappings while preserving source keys and mapping versions.
The analytical layer derives task intervals, location-at-event attributes, and business metrics.
Keep exception records alongside accepted records so that coverage can be measured.
Do not let an inner join silently turn unknown location into nonexistent work.

Event time and ingestion time answer different questions.
For physical flow, use occurrence time when reliable.
For an audit of what a dashboard showed at 17:00, also limit records to those received by 17:00.
For a later corrected view of the 17:00 physical state, include later-arriving evidence but label the
result as restated.
The two views may legitimately differ.

### 8.2 GS1 identifiers and events

GS1 publishes standards for identifying trade items, parties and locations, and logistics units.
A Global Trade Item Number (GTIN) identifies a trade item under GS1 rules.
A Global Location Number (GLN) identifies a party or location according to its allocation rules.
A Serial Shipping Container Code (SSCC) identifies a logistics unit.
A site's internal SKU, bin ID, or license plate is not automatically a GTIN, GLN, or SSCC.
Maintain explicit mappings and validate identifiers under the applicable GS1 specification.
Whether a facility must use a particular identifier depends on its trading agreements and applicable
law; the mere existence of a standard does not impose a universal legal obligation.

GS1 Electronic Product Code Information Services (EPCIS) provides a standard for sharing visibility
events about the what, when, where, why, and how of supply-chain objects.
A WMS task lifecycle is not automatically an EPCIS event stream.
Mapping requires agreement about the business step, location, identifier, event time, and object or
quantity represented.
EPCIS does not prove that a scan corresponds to physical reality or supply a task's unrecorded start.
For this topic, the standard is a useful interoperability reference rather than a replacement for the
site's event dictionary.

### 8.3 Location, time, and historical meaning

Use an Internet Assigned Numbers Authority time-zone identifier when the facility's civil time rules
are known, and retain an actual offset for each event where possible.
An offset identifies an instant when paired with a local timestamp, but does not alone identify all
historic and future rules of a named time zone.
UTC is suitable for interval arithmetic; local time is suitable for shift and cut-off reporting when
the governing zone is explicit.
A fixed-offset assumption can be wrong across daylight-saving changes.

The validity of a location mapping is a separate temporal problem from event occurrence.
Use an effective date for a business change and, when necessary, a knowledge date for when the
analytics team learned of that change.
A bitemporal history can answer both what the layout was believed to be and what it is now understood
to have been.
It adds complexity, so use it when corrections and audit requirements justify it.

### 8.4 Inventory and units

Units must be consistent within a reconciliation.
Record original unit, converted quantity, conversion factor, and factor validity.
Weight-variable goods and partial cases require rules beyond a fixed cases-to-eaches factor.
Rounding individual movement legs can create a residual even when unrounded quantities reconcile.
Decide the rounding precision and identify whether the unit is measured or estimated.

A WMS operational balance is not necessarily the financial inventory ledger.
Ownership, consignment, returns, in-transit goods, and cutoff timing can differ.
Reconcile to the appropriate accounting definition rather than treating either system as wrong by
construction.
An operational hold is not itself evidence of a regulatory recall or a financial impairment.

### 8.5 Privacy and workforce data

Task and scanner events can identify workers or reveal individual work patterns.
Limit access, retain data only for an approved purpose and period, and aggregate or pseudonymize when
individual identity is unnecessary.
Pseudonymization does not necessarily make data anonymous.
Consult the organization's privacy, labor, and legal specialists before using such data for
individual monitoring or employment decisions.

For example, the EU General Data Protection Regulation, Regulation (EU) 2016/679, applies under its
specified territorial and material scope and sets principles including purpose limitation and data
minimization.
Its application depends on the facts, and local labor rules may impose additional requirements.
This educational treatment is not legal advice and does not state a universal retention period.
Publish operational metrics with access controls and a clear explanation of their limitations.
A scanner timestamp cannot independently measure a person's effort, breaks, or compliance.

### 8.6 Safety and record integrity

The physical intervention must respect the site's safety procedures and applicable law.
A faster route through a pedestrian exclusion zone or an overloaded staging lane is not an acceptable
optimization simply because the WMS records fewer minutes.
The United States Occupational Safety and Health Administration (OSHA) publishes warehouse safety
resources and applicable workplace requirements; applicability depends on jurisdiction and activity.
A standards document, a guidance page, and a binding regulation are different kinds of authority.
Consult the applicable text and qualified specialists for a specific site.

## 9. Industry practice and case evidence

### 9.1 DHL Supply Chain and vision picking

DHL publicly described trials and subsequent use of augmented-reality vision picking in warehouse
operations, using smart glasses to present picking information to workers.
The company's 2017 announcement reported an average productivity improvement in its pilots.
That figure is a company-reported result, not independent proof that every warehouse will achieve the
same improvement.
The relevant source is linked in section 16.

The case illustrates that a work instruction, an operator interaction, and a physical pick are
related but distinct records.
**Inference:** If a site changes from a handheld workflow to a vision-assisted workflow, the timing
and meaning of start and confirmation events may change even if the physical process is similar.
Before comparing before and after, document the event definitions and look for a measurement break.
The public case does not establish the exact schema or queue durations used by the synthetic example.

### 9.2 Amazon Robotics and fulfillment operations

Amazon describes using robotic drive units and other technologies to move inventory and assist work
in its fulfillment network.
Amazon's own descriptions are company sources and should be read as descriptions of its operations,
not as a neutral estimate of benefits in another network.
The company describes interactions among inventory pods, workstations, and people; its actual
internal data model is not disclosed by these public descriptions.

**Inference:** In a goods-to-person configuration, the physical waiting boundary may be a workstation
waiting for a pod, an arriving pod waiting for a workstation, or an order waiting for both.
A task-only average cannot distinguish these mechanisms without resource and location events.
A useful local model would link the relevant work instruction to the equipment and workstation
observations, subject to identifier and timestamp coverage.
Do not treat that proposed model as a claim about Amazon's proprietary system.

### 9.3 Walmart and traceability

Walmart has publicly described work with IBM Food Trust on food traceability, including an exercise
tracing mangoes and a reported reduction in the time needed to trace their origin.
The reported comparison concerns traceability lookup, not warehouse picking speed or a general
estimate of operational savings.
It is a company/partner-reported case rather than an independent controlled estimate.

The modeling lesson is narrower than a claim that a particular technology solves inventory accuracy.
A trace requires identifiers, linked events, and enough participating parties to connect the chain.
**Inference:** Within a warehouse, a lot-aware movement ledger can make the location and state of
potentially affected stock easier to investigate, provided the underlying observations are accurate.
A complete database cannot compensate for a missing scan or incorrectly identified lot.

### 9.4 What can and cannot be transferred

A public case gives a real implementation context and a reported observation.
It does not reveal the counterfactual for your facility, the complete sample, or all local process
constraints.
A case about identification does not prove a labor productivity effect.
A case about picking technology does not prove an inventory ledger is accurate.
Use cases to formulate hypotheses, then measure your own baseline and intervention with documented
boundaries and quality checks.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Task and line join fanout | Task count grows after join | Many-to-many associations duplicate facts | Inflated time and quantities | Aggregate at grain; use bridge and reconciliation |
| Missing open work | Completion rate looks perfect | Only terminal events enter report | Long queues disappear | Start from task master and retain open status |
| Creation treated as readiness | Large waits on blocked orders | Upstream hold included as eligible queue | Wrong capacity decision | Separate created, blocked, and ready states |
| Completion treated as shipment | Packing KPI improves but shipments miss cut-off | Different physical and business boundaries | False service claim | Join to actual shipment confirmation |
| Current location used historically | Old activity appears in a newly created zone | Current master overwrites prior classifications | Misattributed congestion | Effective-date location attributes |
| Duplicate delivery | Repeated event IDs after retry | At-least-once integration delivery | Double-counted throughput | Idempotent ingest and raw audit trail |
| Legitimate repeat dropped | Repeated scans share a timestamp | Heuristic deduplication removes real events | Missing attempts and understated work | Use source IDs; quantify uncertain matches |
| Out-of-order ingestion | Completion arrives before start in the feed | Network delay changes arrival order | Negative or missing intervals | Order by reliable occurrence and sequence |
| Clock or time-zone error | Negative durations near shift changes | Unsynchronized clocks or ambiguous local time | Incorrect shift and queue attribution | Preserve offset, clock metadata, and flags |
| Overwritten state | Only final status exists | Source captures no transition history | Historical waits unrecoverable | Obtain audit log or capture changes prospectively |
| Cancellation counted as completion | Throughput rises during a wave cancellation | All terminal statuses are pooled | False productivity gain | Separate completion, cancellation, error |
| Pauses ignored | Long tasks cluster around breaks or equipment failures | Elapsed time includes unobserved pause | Mislabelled active labor | Capture pause/resume or label elapsed explicitly |
| Missing inventory transfer leg | Site total changes on bin move | One side of ledger is absent or late | Phantom gain or loss | Reconcile both legs and timing |
| Status conflation | Held stock appears available to pick | Eligibility is inferred from location | Invalid allocation and service promises | Model state separately and version rules |
| Unit mismatch | Case counts summed with eaches | Conversion factor missing or stale | Inflated inventory | Effective-dated conversions and unit controls |
| Late correction hidden | Prior dashboards change without explanation | History is restated silently | Loss of trust and auditability | Record as-of time, revisions, and lineage |
| Bottleneck guessed from mean | Slowest average task is blamed | Queue and resource context omitted | Misallocated staffing | Inspect flow rates, queues, blocking, and mix |
| Scanner event equated with physical truth | Floor count disagrees with dashboard | Offline or missed confirmation | Data issue mistaken for process issue | Validate against floor and independent records |
| Individual ranking from task duration | Workers with difficult assignments rank poorly | Mix and unobserved waiting differ | Unfair management decisions | Use contextual aggregates and governance review |
| Mixed cohorts | Queue mean plus execution mean differs from cycle mean | Denominators and observation windows differ | Misleading arithmetic | Publish cohort and count for each measure |

A red flag is a prompt for investigation, not a diagnosis by itself.
A negative interval can arise from a clock defect, a legitimate correction, or a wrong lifecycle map.
A missing completion may be an open task or a missing integration message.
The appropriate repair depends on the source evidence and the operational process.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| Source event IDs are stable | Idempotent ingestion | IDs reused or changed on retry | Compare payloads and keys across replay windows |
| Event clocks are sufficiently accurate | Meaningful duration arithmetic | Device drift or wrong zone | Compare device and server clocks on sampled events |
| Ready means eligible to work | Queue measures available work | Blocked tasks coded ready | Compare hold history and supervisor observations |
| Start is comparable across tasks | Fair within-cohort comparison | Some flows auto-start and others scan | Inspect workflow configuration by task type |
| Completion means the stated boundary | Valid throughput measure | Auto-close before physical movement | Trace sample tasks to independent confirmation |
| Task relationships are complete | Correct line-level attribution | Batch tasks or manual work omitted | Reconcile bridge to orders and task master |
| Location history is available | Historical zone attribution | Reused bin codes or missing layout changes | Check effective-date overlaps, gaps, and samples |
| Stock postings balance | Reliable inventory reconciliation | Missing legs, timing differences, corrections | Compare opening, movements, and closing balances |
| Unit conversion is valid at event time | Comparable quantity arithmetic | Changed pack size or variable weight | Compare effective conversion to source documents |
| Open status is observable | Censoring and backlog metrics | Missing completion messages | Compare open list with physical floor and downstream events |
| Comparable periods have similar mix | Meaningful before/after comparison | Different order profile or service windows | Stratify by SKU, order type, and workload |
| Capacity evidence reflects actual availability | Constraint hypothesis is interpretable | Equipment downtime or breaks unrecorded | Compare roster, maintenance, and work logs |
| A physical scan is representative | Digital events approximate flow | Manual bypass or offline operation | Sample independent observations and exception logs |
| Intervention does not shift delay elsewhere | End-to-end service can improve | Downstream pack, stage, or dock saturates | Track adjacent queues and shipment outcomes |

The synthetic calculations assume all stated events belong to the same time zone and date.
They do not estimate a causal effect or a labor standard.
A four-task sample is deliberately too small to support a site-wide staffing conclusion.
The code demonstrates interval construction for a restricted lifecycle, not a complete production
WMS integration.

## 12. Exercises

### Exercise 1 — Map a flow

**Time:** 15 minutes.
A pallet arrives, is inspected, placed on quality hold, released, put away, replenished to a pick
face, picked for an order, packed, staged, and shipped.
A hold release occurs without moving the pallet.

**Deliverable:** Draw a process diagram and identify at least three task events, three locations,
and two inventory states.
**Acceptance criteria:** Separate the physical movement from the hold-state change; mark the point at
which the order is shipped; show at least one potential queue.

### Exercise 2 — Specify grains and joins

**Time:** 20 minutes.
One batch pick task serves three order lines; one of those lines is filled by two pick tasks.
Each task has many events, and each line has a requested quantity.

**Deliverable:** Write keys and row meanings for task, event, order line, and a bridge table.
Explain how to calculate unique task count without multiplying it by the number of order lines.
**Acceptance criteria:** Represent the many-to-many relationship, preserve event grain, and identify
why summing line quantity after joining all task events can overcount.

### Exercise 3 — Calculate a lifecycle

**Time:** 15 minutes.
Task T1 is ready at 09:10, starts at 09:34, and completes at 09:47 on the same day.
Task T2 is ready at 09:20 and has not started by the 10:00 as-of time.
Times are unambiguous and in the same zone.

**Deliverable:** Calculate T1 queue, elapsed execution, and total elapsed minutes; report T2's open
age and status.
**Acceptance criteria:** Verify the interval identity for T1 and do not classify T2's age as a
completed wait.

### Exercise 4 — Reconcile stock states

**Time:** 20 minutes.
At opening there are 80 available eaches and 12 held eaches.
The site receives 25 available eaches, places 7 available eaches on hold, releases 4 held eaches,
and ships 18 available eaches.
There are no other movements.

**Deliverable:** Calculate closing balances by state and site total.
**Acceptance criteria:** Show signed state transfers, demonstrate that internal transitions cancel,
and state all results in eaches.

### Exercise 5 — Diagnose a queue

**Time:** 20 minutes.
A pack queue has 40 lines at 14:00.
During the next hour 90 lines become ready, 65 lines start, and 5 are cancelled before starting.
At the same time a staging lane is full and packers report waiting for space.

**Deliverable:** Calculate the 15:00 queue and propose two competing explanations and two independent
checks.
**Acceptance criteria:** Use the queue identity; distinguish a flow imbalance from proof of root
cause; include downstream blocking as a possible explanation.

### Exercise 6 — Change the query

**Time:** 25 minutes.
Use the section 7 program as the starting point.
Add a fifth task ready at 16:40 UTC with no start or completion by 17:00 UTC.

**Deliverable:** Provide the new event row and the expected output fields for that task.
**Acceptance criteria:** The queue duration is null, the open ready age is 20 minutes, and `is_open`
is 1; no completion is invented.

### Exercise 7 — Critique an executive report

**Time:** 20 minutes.
A report says: “Average pick time fell from 25 to 18 minutes, so the new process increased worker
productivity by 28%.”
The earlier figure covers tasks completed in the morning; the later figure covers tasks completed
throughout the day.
The new workflow also changed the definition of the start event.
Open tasks and task types are not reported.

**Deliverable:** Identify at least four defects and specify a defensible comparison.
**Acceptance criteria:** Address cohort, event semantics, censoring, task mix, and the difference
between elapsed time and labor productivity.

### Exercise 8 — Design a production contract

**Time:** 35 minutes.
A facility wants an hourly dashboard showing aged replenishment tasks by zone and stock on quality
hold.
Some scanners operate offline; location codes can be reused after a layout change.
The dashboard will be viewed by supervisors and a central analytics team.

**Deliverable:** Write a one-page contract covering grain, identifiers, events, timestamps, as-of
policy, location history, inventory state, quality controls, privacy, and ownership.
**Acceptance criteria:** Preserve open tasks and late arrivals; give at least three measurable quality
checks; distinguish occurrence from ingestion time; explain how corrections change published results.

## 13. Self-check answers

### Exercise 1

A valid diagram has receiving and inspection before a hold, then release and put-away, followed by
replenishment, pick, pack, staging, and shipment.
Receiving dock, inspection location, reserve bin, forward pick face, pack station, and staging lane
are possible locations; the exact site may use fewer or different locations.
Available and quality hold are example states.
The transition from hold to available changes eligibility and need not move the pallet.
A put-away task creation, start, and completion can record the work instruction's lifecycle.
The buffer between release and put-away start is a possible queue.
Shipment is a separately defined outbound confirmation, not automatically the pack completion event.

### Exercise 2

Use one task row per facility and task ID, one event row per source event version, and one line row
per order and line ID.
Use a bridge with one row per task-line association and an association key or a documented composite
key; include allocated quantity where the source supports it.
The bridge allows both one task to serve several lines and one line to use several tasks.
Count distinct task keys, or aggregate tasks before joining to the bridge.
Joining three line associations to ten task events can produce thirty rows for that task.
Summing a line quantity on those rows repeats the quantity; aggregate at the correct grain first.

### Exercise 3

T1's queue is `09:34 - 09:10 = 24` minutes.
Its elapsed execution is `09:47 - 09:34 = 13` minutes.
Its total ready-to-completion is `09:47 - 09:10 = 37` minutes, and `24 + 13 = 37` minutes.
T2 is open and not started at 10:00; its open ready age is `10:00 - 09:20 = 40` minutes.
The 40 minutes are a lower bound on its eventual wait if the task later starts without interruption,
not a completed observation.

### Exercise 4

Available closing is `80 + 25 - 7 + 4 - 18 = 84` eaches.
Held closing is `12 + 7 - 4 = 15` eaches.
Total closing is `84 + 15 = 99` eaches.
At the site boundary the same answer is `80 + 12 + 25 - 18 = 99` eaches.
The 7-each hold and 4-each release transfer stock between states and do not change total stock.

### Exercise 5

The ending queue is `40 + 90 - 65 - 5 = 60` lines at 15:00.
It grew by 20 lines over the hour.
One explanation is insufficient packing capacity relative to arrivals.
Another is that the full staging lane blocked packing, making downstream space the immediate
constraint; an event-feed defect is a further possibility.
Check physical staging occupancy and packing interruptions independently of the WMS queue.
Compare release timestamps and start confirmations to source logs, and inspect staffing and equipment
availability over the same hour.
The observed increase alone does not select the correct explanation.

### Exercise 6

Add task `P105` to the task insert and add an event such as:

```python
("e12", "NORTH", "P105", "ready", "2026-04-14T16:40:00Z")
```

Its expected fields are `quality='valid'`, `queue_minutes=None`,
`completed_execution_minutes=None`, `open_ready_age_minutes=20.0`, and `is_open=1`.
The task is waiting at the as-of time; no start or completion has been observed.
The query's open age is not a completed queue duration.

### Exercise 7

The arithmetic reduction in the reported mean is `(25 - 18) / 25 = 28%`, but this is a reduction in
the reported duration, not proof of a 28% productivity increase.
The time windows differ, so demand and work mix may differ.
The start event definition changed, so the two durations may cover different physical intervals.
Completed-only selection can hide long open tasks.
Task types, difficulty, staffing, and pauses may differ.
Compare a consistent cohort and event boundary over comparable shifts and task strata, report open
work and coverage, and measure output and an appropriate labor denominator separately.
Check service and quality as well as duration before attributing a change to the workflow.

### Exercise 8

A strong contract identifies one task attempt and one event version as distinct grains; keys include
facility and source identity.
It defines ready, start, completion, cancellation, and corrections, and retains open work at the
hourly cutoff.
It records event occurrence and ingestion times, source offset, and a late-arrival policy.
It joins location versions valid at event time and reports unmapped or overlapping versions.
It defines quality-hold quantity at the stock-key and snapshot-time grain, without joining balances
blindly to task events.

At least three measurable checks could be the share of tasks missing ready events, the count of
negative intervals, and the share of events arriving more than a specified number of minutes late.
The threshold must be chosen for the facility's reporting requirement rather than assumed universal.
Other checks include inventory ledger residual and location mapping coverage.
The contract names data and business owners, restricts worker identifiers, shows dashboard freshness,
and distinguishes the originally published view from a later restatement.

## 14. Interview questions

1. **How would you distinguish inventory from a task?**
   A strong answer separates stock identity and state from a work instruction and its events.
   A weak answer assumes one task always equals one pallet or one order line.
2. **What is the grain of a task event table?**
   A strong answer specifies one source event or version, its key, and the task relationship.
   A weak answer names columns without stating what a row represents.
3. **Why can a task-to-order join inflate quantities?**
   A strong answer explains many-to-many relationships, event fanout, and preaggregation.
   A weak answer applies `DISTINCT` to an arbitrary sum without understanding the grain.
4. **How do you measure ready-to-start wait?**
   A strong answer defines eligibility and both event boundaries and treats open work separately.
   A weak answer subtracts creation from completion and calls it queue time.
5. **What does a missing completion mean?**
   A strong answer distinguishes genuinely open work, late events, missing integration, and cancellation.
   A weak answer assumes the worker failed to complete the task.
6. **How do you handle a location that changed zones?**
   A strong answer uses effective-dated attributes and flags unavailable history.
   A weak answer joins all old tasks to the current zone.
7. **How do you reconcile a stock status change?**
   A strong answer moves equal quantities between states and checks the total at the chosen boundary.
   A weak answer treats the hold as a shipment or a new receipt.
8. **How do you identify a bottleneck?**
   A strong answer examines queue growth, rates, resource availability, downstream blocking, and mix.
   A weak answer chooses the task with the largest average duration.
9. **Why retain both event and ingest timestamps?**
   A strong answer distinguishes physical sequence from data availability and report restatement.
   A weak answer assumes the last loaded event was the last physical action.
10. **What is wrong with ranking workers by elapsed task duration?**
    A strong answer notes task mix, waiting, missing pauses, privacy, and governance.
    A weak answer treats a scanner interval as a complete measure of individual effort.

## 15. Key terms

- [Allocation](#4-definitions-and-vocabulary) — reservation or assignment of inventory to demand.
- [As-of time](#4-definitions-and-vocabulary) — cutoff for evaluating a state.
- [Bottleneck](#4-definitions-and-vocabulary) — capacity constraint for a defined flow and period.
- [Cohort](#4-definitions-and-vocabulary) — population selected by a shared rule.
- [Completion time](#4-definitions-and-vocabulary) — observed task-complete transition.
- [Correction](#4-definitions-and-vocabulary) — traceable source revision or reversal.
- [Coverage](#4-definitions-and-vocabulary) — share of expected evidence that is usable.
- [Cycle time](#4-definitions-and-vocabulary) — elapsed time across stated boundaries.
- [Duplicate](#4-definitions-and-vocabulary) — repeated delivery of the same logical event.
- [Elapsed execution](#4-definitions-and-vocabulary) — start-to-completion time including unknown pauses.
- [Event](#4-definitions-and-vocabulary) — recorded observation or state transition.
- [Event time](#4-definitions-and-vocabulary) — source-reported occurrence time.
- [Facility](#4-definitions-and-vocabulary) — operational site with its own identity and zone.
- [Grain](#4-definitions-and-vocabulary) — precise meaning of one data row.
- [Handling unit](#4-definitions-and-vocabulary) — pallet, tote, carton, or other grouped unit.
- [Hold](#4-definitions-and-vocabulary) — restriction on an inventory use.
- [Idempotency](#4-definitions-and-vocabulary) — safe reprocessing without unintended duplicate effect.
- [Ingest time](#4-definitions-and-vocabulary) — time a record reaches the analytics system.
- [Inventory balance](#4-definitions-and-vocabulary) — quantity at a specified stock key and time.
- [Inventory state](#4-definitions-and-vocabulary) — classification of eligibility, condition, or custody.
- [Location hierarchy](#4-definitions-and-vocabulary) — effective-dated structure of warehouse places.
- [Lot](#4-definitions-and-vocabulary) — traceability grouping under a defined rule.
- [Order line](#4-definitions-and-vocabulary) — requested product and quantity within an order.
- [Queue time](#4-definitions-and-vocabulary) — elapsed time between defined queue boundaries.
- [Ready time](#4-definitions-and-vocabulary) — time work becomes eligible to start.
- [Reconciliation](#4-definitions-and-vocabulary) — comparison of independently derived balances.
- [Replenishment](#4-definitions-and-vocabulary) — movement supporting a forward pick location.
- [Right censoring](#4-definitions-and-vocabulary) — observation ends before the outcome.
- [SKU](#4-definitions-and-vocabulary) — business-defined stock keeping unit.
- [Task](#4-definitions-and-vocabulary) — work instruction with a lifecycle.
- [Throughput](#4-definitions-and-vocabulary) — completed flow per period at a named boundary.
- [WMS](#4-definitions-and-vocabulary) — warehouse management system.
- [Work in process](#4-definitions-and-vocabulary) — work between named boundaries at a time.

## 16. Sources

1. GS1, [GS1 General Specifications](https://www.gs1.org/standards/barcodes-epcrfid-id-keys/gs1-general-specifications).
   Primary standards reference for GS1 identification keys and related rules; consult the applicable
   edition and local implementation guidance.
2. GS1, [EPCIS and CBV](https://www.gs1.org/standards/epcis).
   Primary standards reference for interoperable supply-chain visibility events.
3. Internet Assigned Numbers Authority, [Time Zone Database](https://www.iana.org/time-zones).
   Reference for named civil-time zones and historical rule data.
4. SQLite, [Date And Time Functions](https://www.sqlite.org/lang_datefunc.html).
   Official documentation for `julianday` and supported time-value formats used in section 7.
5. European Union, [Regulation (EU) 2016/679](https://eur-lex.europa.eu/eli/reg/2016/679/oj).
   Primary legal text for the GDPR; applicability and current amendments require legal assessment.
6. United States Occupational Safety and Health Administration,
   [Warehousing](https://www.osha.gov/warehousing).
   Government safety resource; consult applicable regulations for specific requirements.
7. DHL, [DHL Supply Chain makes smart glasses new standard in logistics](https://www.dhl.com/global-en/home/press/press-archive/2017/dhl-supply-chain-makes-smart-glasses-new-standard-in-logistics.html), 2017.
   Company-reported vision-picking case; claims are not independent causal estimates.
8. Amazon, [Amazon Robotics](https://www.aboutamazon.com/what-we-do/robotics).
   Company description of fulfillment robotics; not a disclosure of the internal event schema.
9. Linux Foundation Decentralized Trust, [How Walmart brought unprecedented transparency to the food supply chain with Hyperledger Fabric](https://www.lfdecentralizedtrust.org/case-studies/walmart-case-study).
   Industry case study reporting Walmart and IBM's traceability implementation and mango exercise.

Sources support the attributed standards and cases, not the synthetic figures or a universal WMS
schema.
Consult the current source edition and local law before implementing a production or compliance rule.

## 17. Links

- [Module 13 index](_index.md) — module scope and topic sequence.
- [Warehouse flow system](01_warehouse-flow-system.md) — physical process and buffers.
- [Productivity metrics](03_productivity-metrics.md) — output and labor denominators.
- [Module 13 survey and lab context](../../../course/13_warehousing_fulfillment_labor.md) —
  broader warehouse, fulfillment, labor, and hands-on context.
