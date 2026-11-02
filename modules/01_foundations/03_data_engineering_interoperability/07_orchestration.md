> Module 03 — Data Engineering and Interoperability
> Topic 07 of 11
> Estimated study time: 3.5 hours
> Prerequisites: 03.01, 03.02, 01A.04

# Orchestration and Scheduling

---

## 1. Why this topic exists

At 05:12 on 18 February the planning dashboard turns green. Every task in the nightly
pipeline has succeeded. The plan is published forty-eight minutes before the 06:00 cutoff,
which the operations team reads as a good night. At 09:00 the first pick list comes back
empty. The extract from the enterprise resource planning (ERP) system returned zero order
lines, because a source view had been rebuilt overnight and the extract queried the old name.
The query did not fail. It returned an empty set, the job wrote zero rows, every downstream
model computed correctly on zero rows, and the supply plan concluded that no demand existed.
The pipeline was on time and wrong.

That incident is not a data-quality story. A test would have caught it, and Topic 11 covers
the tests. It is an orchestration story, because the orchestrator was the only component that
knew what had to be true before a plan was allowed to exist, and it was told only what order
to run things in.

Three claims organise this topic.

1. **A directed acyclic graph (DAG) is a specification of preconditions, not a to-do list.**
   Each node should assert a property that must hold before the next stage may run: this
   extract is complete for this interval, this dimension is current as of this instant, this
   fact reconciles to a control total. A DAG read as a sequence hides missing assertions. A
   DAG read as a specification makes them visible, because every node is a claim someone can
   challenge.
2. **The planning cutoff is the only deadline that matters, and it is a wall-clock instant in
   a named time zone.** Everything else in the schedule is negotiable. Every runtime, retry,
   backfill, and queue delay is a claim on the interval before that instant, and slack is the
   currency in which those claims are paid.
3. **Success is not the absence of red.** There are four states, not two: on time and right,
   late and right, on time and wrong, late and wrong. Most orchestration tooling can express
   the first and the last. The second and third require design, and the third is the one that
   damages a company.

**Why this matters to a data scientist specifically.** Every model in this program consumes
data that arrives through this machinery, and most model failures that look like modelling
failures are scheduling failures wearing a costume. A feature build that calls `now()` is not
reproducible when you backfill it, so your training set silently changes between the
experiment you reported and the model you shipped. A retry that appends instead of replaces
doubles a demand series, and the forecast learns a spike that never happened. A daylight
saving transition that fires the job twice gives you two plans for one day, and the comparison
between them measures the scheduler, not the policy. A plan that runs on an empty input
produces a model evaluation with no denominator at all, and the metric returns a number
anyway.

Topic 06 derived freshness thresholds from the schedule. This topic makes the schedule itself
the object of study: what the orchestrator is asserting, when it must finish, what it may
retry, what it may replay, and how it should tell the difference between late and wrong.
Topic 01 established the layers and their contracts; Topic 02 established the cadence and the
latency budget. This topic is the machinery that holds both to a clock.

## 2. Learning outcomes

By the end of this topic you can:

1. **Read** a DAG as a set of preconditions rather than a sequence, and **identify** the
   assertion each node is making and the assertion that is missing.
2. **Perform** a backward schedule calculation from a stated cutoff, producing earliest start,
   latest start, and slack for every task, and **identify** the critical path.
3. **Compute** the latest permissible start of a pipeline given a cutoff, a critical-path
   runtime, and a safety buffer, and **state** the result in local time and in Coordinated
   Universal Time (UTC).
4. **Distinguish** transient from deterministic failures and **set** a retry policy whose worst
   case fits inside the task's slack.
5. **Diagnose** daylight saving time (DST) hazards in a cron schedule, and **state** the
   behaviour of at least four scheduler implementations at both transitions.
6. **Design** an idempotent task, and **test** idempotency by re-running on unchanged input and
   comparing row counts and a checksum.
7. **Plan** a backfill that repairs a mart partition and every downstream artifact, without
   overwriting a plan version that has already been acted upon.
8. **Write** a pipeline service-level objective (SLO) measured from the cutoff, and **separate**
   alerting for lateness from alerting for incorrectness.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold one picture: a chain of assertions running left to right against a vertical deadline.
The horizontal axis is the night. The vertical line at 06:00 is not a target; it is the
instant at which the planning meeting starts, the carriers are already routed, and the
production sequence is already frozen. After it, a better plan has no buyer.

```text
 LOCAL TIME    00:00    01:00    02:00    03:00    04:00    05:00   05:30   06:00
                 |        |        |        |        |        |       |       |
 SOURCES         [ERP orders 35]                                        |       |
                 [WMS stock  20]-.                                      |       |
                 [Open POs   25]-+-->[conform]-->[fact]-->[demand]-->[supply]-->[exceptions]-->[publish]
                 [Forecast   15]-'      20          25       40         55          12          8
                 |        |        |        |        |        |       |       |
 CRITICAL PATH   |<--------------------- 3 h 15 min ----------------->|       |       |
                 |        |        |        |        |        |  30 m |       |       |
                 |        |        |        |        |        |<-buf->|       |       |
                 |        |        |        |        |        |       |       |
 LATEST START    .        .       X        .        .        .       .       X  CUTOFF
                              02:15                                      06:00
                 |        |        |        |        |        |       |       |
                 |        |        |        |        |        |       |       |
              scheduled    the four extracts may run in parallel; the chain that
              launch       determines the finish time is ERP -> conform -> fact ->
                           demand -> supply -> exceptions -> publish
```

Read the picture with four properties in mind, because they explain most of what follows.

**Property 1 — The deadline is exogenous.** Nobody in the data team chose 06:00. It is set by
a meeting, a carrier collection, a production freeze, or a regulatory filing. A service-level
target can be renegotiated in a planning cycle; a cutoff cannot be renegotiated at 05:50. The
orchestrator's job is to meet an external clock with an internal process.

**Property 2 — Time-based scheduling is a proxy for data readiness.** A job that starts at
02:15 is asserting, without evidence, that its inputs are ready at 02:15. That assertion is
true most nights and false on the nights that matter. Data-aware scheduling — triggering a
run when an upstream asset has been updated rather than when the clock says so — replaces the
proxy with the thing itself, at the cost of losing the predictable start time that makes
backward scheduling possible. Most real designs need both: a time trigger for the cadence and
a data assertion for the gate.

**Property 3 — Schedule time and validity time are different clocks.** A run that starts at
02:15 on 19 February is not "the 19 February run". It is the run for a data interval, usually
`[18 February 00:00, 19 February 00:00)`. The interval, not the start time, is the identity of
the run. Every task that reads "the latest data" instead of its declared interval is
unbackfillable, because it cannot answer which day it was supposed to be processing.

**Property 4 — Slack is the only currency.** The schedule has a fixed budget measured in
minutes before the cutoff. Every design choice spends from it: an extra source, a heavier
model, a retry, a wider backfill window, a serialized step that could have been parallel. A
design that does not state its slack cannot say which of those choices it can afford.

### 3.1 The four-state grid

The most useful discipline in this topic is refusing to describe a run as "successful" or
"failed". A run has two independent properties: whether it met the cutoff, and whether its
output was correct.

| | **Correct output** | **Incorrect output** |
|---|---|---|
| **Met the cutoff** | The design goal. The plan is usable and the numbers are sound. | The dangerous state. Green dashboards, an authoritative-looking plan, no complaint until someone acts on it. |
| **Missed the cutoff** | A degraded but recoverable state, if the organisation has a stated fallback: prior plan plus a flagged exception list. | The visible failure. Expensive, noisy, and usually fixed quickly because it is obvious. |

Two observations follow.

First, the top-right cell is the only one that gets worse with time. A late-and-wrong run is
discovered at 06:05 and rerun. An on-time-and-wrong run is discovered when a truck arrives at
the wrong DC, which may be days later, and by then the erroneous plan has been acted on by
people who had no reason to doubt it. Every control in this topic is aimed primarily at that
cell.

Second, the bottom-left cell is a design choice, not an accident. If there is no stated
fallback, a missed cutoff becomes a crisis and the team will be pressured to remove the
guards that caused the miss. Organisations that write down "if the plan is not published by
06:00, planners use the previous version and the exception list is marked stale" are the ones
that can afford to fail a run on an empty input. Organisations without that sentence will
always choose a wrong plan over a late one.

> **Note:** Airflow's documentation uses "logical date" for the start of the data interval,
> while other tools use "partition key", "business date", or "run date". These are the same
> concept and the differences in naming cause real confusion in incident reviews. Pick one
> term per organisation and state its time zone in the definition.

### 3.2 Task-centric and asset-centric orchestration

Two families of tool express the same graph differently, and the difference changes what the
tool can tell you.

| Question | Task-centric (for example Airflow) | Asset-centric (for example Dagster) |
|---|---|---|
| What is the unit? | A task that runs at a time | A data asset that has a version |
| What does a run mean? | This task executed for this interval | This partition of this asset was materialised |
| What does a backfill do? | Re-runs tasks for past intervals | Re-materialises partitions and their dependents |
| What can it tell you about freshness? | When the task last succeeded | When the asset was last materialised, per partition |
| Default failure mode | A task succeeds but writes nothing | An asset has no new partition and nobody notices |
| Where the cutoff lives | In the scheduler's timetable | In a schedule condition or a sensor |

Neither family removes the need to think. A task-centric tool expresses "run this at 02:15"
easily and "this partition is complete" awkwardly; an asset-centric tool makes the partition
first-class but will still materialise it from an empty input and call it a success.

### 3.3 The cutoff is not the same as the service level

A cutoff is a business instant: the plan must exist by 06:00 or the day runs on yesterday's
plan. A service-level objective is a statistical statement about how often you meet it. The
two are related but not interchangeable, and conflating them produces two common errors.

- **An SLO stated as an average hides the tail.** "Plans are published by 06:00 on average"
  is compatible with missing every Monday, which is the day that matters most.
- **A cutoff stated as a target invites negotiation.** "We aim for 06:00" invites a request to
  accept 06:20. "The plan is consumed at 06:00" states a fact about the consumer, and the
  conversation becomes about what to drop when the pipeline cannot make it.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Directed acyclic graph (DAG) | A set of nodes and directed edges with no cycles, used to express task dependencies | Treated as a synonym for "pipeline" | Graph structure |
| Task | One unit of work with inputs, outputs, and a state | Called a "job" when it is one step of many | Task identifier |
| Scheduler | The component that decides when a run starts and which run is due | Confused with the executor | Service |
| DAG run | One execution of the whole graph for one data interval | Identified by its start time | Run identifier plus interval |
| Logical date | The start of the data interval a run is responsible for | Read as the time the run started | ISO 8601 instant in a stated zone |
| Data interval | The half-open period `[start, end)` a run processes | Assumed to be "yesterday" | Two instants |
| Timetable | The rule that generates run intervals and their start times | Treated as an immutable cron string | Schedule object |
| Catchup | Automatic creation of runs for intervals missed since the start date | Confused with a backfill | Boolean plus interval range |
| Backfill | Deliberate reprocessing of a stated set of past intervals | Run without a stated range | Interval set |
| Idempotency | The property that applying an operation twice leaves the same state as applying it once | Assumed because the target has a primary key | Boolean property |
| Deterministic task | A task whose output depends only on its declared inputs | Assumed because the code has no random seed | Boolean property |
| Partition | The slice of a dataset that one run owns and may replace | Treated as an index, not an ownership boundary | Key, usually a date |
| Upsert (MERGE) | Insert or update rows on a declared key | Applied without a monotonic guard, so an older row overwrites a newer one | Merge statement |
| Delete-insert | Removing all rows for a partition and inserting the recomputed set | Used without a transaction, leaving a visible gap | Two statements in one transaction |
| Retry | A further attempt of the same task after a failure | Applied to failures that cannot succeed | Count |
| Backoff | The delay before a retry | Fixed at zero, producing a retry storm | Seconds or minutes |
| Jitter | Random variation added to a backoff to de-synchronise parallel retries | Omitted, so all tasks retry at once | Seconds |
| Maximum retries | The cap on attempts before the task is marked failed | Set without checking the schedule budget | Integer |
| Poison task | A task that fails deterministically on every attempt | Retried indefinitely at full cost | Task identifier |
| Poison message | A record that always causes the consumer to fail | Retried in place, blocking the queue behind it | Message identifier |
| Dead-letter queue (DLQ) | A holding queue for messages that exhausted their retries | Used as an archive | Queue plus retention |
| Execution timeout | The maximum wall-clock time a task may run before being killed | Set to the default, which is often unlimited | Minutes |
| Deadline alert | A time-based alert that fires when a run exceeds a stated budget, independent of task state | Confused with a failure alert | Duration from a reference instant |
| Service-level objective (SLO) | A stated percentile bound on a measurable outcome over a window | Written without a measurement method | Percentile plus window |
| Data dependency | A task may start only when a named input is complete | Replaced by a time trigger and a hope | Asset or partition |
| Execution dependency | A task may start only when another task has finished | Confused with a data dependency | Edge in the graph |
| Trigger rule | The condition under which a task runs given its upstream states | Left at `all_done`, which publishes after failures | Rule name |
| Asset-aware scheduling | Triggering a run when an upstream data asset updates | Assumed to guarantee the data is correct | Asset key |
| Critical path | The longest chain of dependent tasks, which sets the earliest finish | Computed from the scheduled start rather than from the cutoff | Task sequence plus duration |
| Slack | The time a task can slip without delaying the cutoff | Reported as a percentage rather than minutes | Minutes |
| Latest start time | The latest instant a task may begin and still meet the cutoff | Computed once and never revisited | Local time plus UTC |
| Cutoff | The business instant by which the plan must exist | Expressed as a date, a target, or a server-local time | Local instant plus zone |
| Planning cycle | The interval between two consecutive cutoffs, which defines one plan generation | Assumed to be 24 hours | Duration |
| Plan freeze | The moment a plan version becomes immutable and is handed to execution | Confused with publication to a dashboard | Instant plus version |
| Plan version | An immutable identifier for one published plan | Overwritten in place | Version identifier plus timestamp |

Five of these deserve expansion, because the distinctions are routinely lost.

**Idempotency versus determinism.** A deterministic task produces the same output from the
same input. An idempotent task leaves the same state whether it runs once or five times.
Determinism is about the computation; idempotency is about the write. A task can be perfectly
deterministic and still corrupt a table by appending its correct output five times. When
people say "the job is idempotent" they usually mean "it uses MERGE", which is necessary and
not sufficient: a MERGE keyed on an ingestion timestamp is an append with extra steps.

**Retry versus backfill.** A retry is an attempt to complete the same interval within the same
run, under time pressure. A backfill is a deliberate reprocessing of an interval after the
fact, under no time pressure and with different obligations to downstream consumers. Mixing
them is how a failed night becomes a corrupted history: the scheduler retries an old interval
at 05:50 and overwrites a partition that a published plan already consumed.

**Cutoff versus freeze.** The cutoff is when the plan must exist. The freeze is when it stops
changing. In a well-designed system they are minutes apart, and the plan identifier is written
at the freeze. In a poorly designed system there is no freeze, the plan table is mutable, and
"the plan we ran on Tuesday" cannot be reconstructed on Thursday.

**Late versus wrong.** Lateness is a property of the schedule. Wrongness is a property of the
data. They are measured by different instruments — a clock and a test — and they must page
different people with different runbooks. A system that reports only "pipeline failed" reports
neither.

**Poison task versus poison message.** A poison task fails on every attempt because of its
code, its configuration, or a schema change; retrying it burns the cutoff and nothing else. A
poison message is one record that always breaks the consumer; retrying it in place blocks
every record behind it. The first is handled by failing fast, the second by quarantining the
record and continuing. Both look like "the job keeps failing" in a dashboard.

## 5. The mechanics, step by step

The procedure has ten steps. The order matters because each step consumes a resource the
previous one defined: the cutoff fixes the budget, the budget fixes the slack, the slack
constrains the retries, and the retries constrain the backfill.

### 5.1 Step 1 — Fix the cutoff, the calendar, and the time zone

Write down four things before drawing any graph.

- The cutoff instant, in a named time zone from the IANA time zone database, not in "server
  time".
- Which days are planning days. A six-day operation has roughly 313 runs a year, not 365, and
  an SLO computed on 365 is measured against days that do not exist.
- The holiday calendar the business uses, and whether a holiday shifts the cutoff or cancels
  the run. Both are common; only one is usually documented.
- The consumer of the output, and what they do in the first ten minutes after 06:00. This
  determines whether a late plan is usable at all.

*If skipped:* the schedule is anchored to whatever the server's clock happens to be, the
first DST transition or infrastructure migration moves the effective cutoff by an hour, and
nobody can say which runs were supposed to exist.

### 5.2 Step 2 — Schedule backward from the cutoff, not forward from midnight

Forward scheduling answers "when will this finish if it starts at 02:00". Backward scheduling
answers the only question that matters: "when must this start to finish by 05:30". The
difference is slack, and slack is what you spend on retries and late sources.

The procedure is mechanical:

1. List every task with its observed 95th-percentile (p95) duration, not its average. A
   schedule built on averages misses the cutoff on roughly half of the nights by construction.
2. Draw the precedence edges.
3. Set the finish of the last task to the cutoff minus the safety buffer.
4. Walk backward, subtracting durations, taking the minimum latest start across successors.
5. Walk forward from the planned launch time, taking the maximum earliest finish across
   predecessors.
6. Slack is latest start minus earliest start. Zero-slack tasks form the critical path.

The output is a table, and section 6 shows one in full.

*If skipped:* the schedule is a guess that happens to work when everything is fast, which is
most nights, which is why the failure is discovered in a peak week rather than in testing.

### 5.3 Step 3 — Declare the data interval and forbid wall-clock reads

Every task receives its interval as a parameter and reads only data belonging to it. Two
rules make this enforceable.

- **No `now()` in transformation logic.** The only legitimate use of the current time is
  logging and timeout calculation. A filter such as `where created_at >= current_date - 1`
  makes the task unbackfillable, because on a backfill it selects the wrong day.
- **No "latest partition" reads.** A task that reads the newest available partition of an
  upstream table will silently read a partial partition at 02:15 and a complete one at 09:00.
  Read the partition named by the interval, and fail if it does not exist.

*If skipped:* the same code produces different outputs on different days from the same input,
and the difference is attributed to the data.

### 5.4 Step 4 — Make every write idempotent, and prove it

Idempotency is a property of the pair (task, target), not of the task alone. It holds when
three conditions are met.

| Condition | What it means | How it fails |
|---|---|---|
| Same input, same output | The task is deterministic: no random seeds, no `now()`, no unordered `LIMIT` | A `row_number()` without a full ordering tie-break |
| Replace, not append | The write owns a partition and deletes it before inserting, or merges on a business key | `INSERT INTO` on a table with no unique constraint |
| Transactional visibility | Readers never see the half-written state | Delete and insert in separate transactions |

The test is simple and should be in continuous integration: run the task twice on unchanged
input, then compare the row count and a checksum over the key and measure columns. If either
changes, the task is not idempotent, and the retry policy is unsafe by construction.

*If skipped:* retries and backfills accumulate rows at a rate proportional to the number of
attempts, and no uniqueness test catches it if the key includes an ingestion timestamp.

### 5.5 Step 5 — Classify failures before choosing a retry count

A retry policy is a statement about which failures are worth repeating. Get the
classification right and the count follows; get it wrong and the count is decoration.

| Failure class | Examples | Retry? | Why |
|---|---|---|---|
| Transient infrastructure | Connection reset, lock timeout, warehouse queue rejection | Yes, with backoff and jitter | The same request will probably succeed |
| Transient dependency | Upstream application programming interface (API) returned 503, object store throttled | Yes, bounded | The dependency recovers on its own timescale |
| Data not yet arrived | Partition missing at the scheduled time | Yes, but as a sensor with a timeout, not a blind retry | The condition is expected to change |
| Deterministic data defect | Constraint violation, unparseable row, null in a required field | No | Every attempt fails identically |
| Schema or contract change | Column renamed, type widened incompatibly | No | Requires a human decision |
| Resource exhaustion caused by the job | Out of memory, disk full | No, or once after a size reduction | Repeating the same request repeats the exhaustion |
| Timeout too tight | Task killed at a limit below its p99 | No | Fix the limit, not the retry count |

Two quantitative rules follow, and they are the part practitioners skip.

**Rule 1 — the retry must fit the slack.** For a task with slack `s`, duration `d`, backoff
`b`, and `n` retries, the worst case from the first attempt is `(n + 1) * d + n * b`. This
must be less than or equal to the task's slack plus the end buffer. If it is not, the retry
policy is a promise the schedule cannot keep, and the second attempt will run past the
cutoff and be killed mid-write.

**Rule 2 — critical-path tasks usually get zero or one retry.** A task on the critical path
has no slack by definition. Its only budget is the end buffer, which is shared with every
other overrun of the night. Retrying a fifty-five-minute critical-path task after a ten-minute
backoff spends sixty-five minutes to recover a fifty-five-minute step, which is more than most
buffers contain.

*If skipped:* retries convert a recoverable night into a missed cutoff, and they do it
precisely on the nights when the system is already under stress.

### 5.6 Step 6 — Put a gate between the facts and the plan

The plan is the first artifact a human acts on. Nothing downstream of it should run unless
the facts it consumes are complete. A gate task asserts at least four things.

1. **Non-emptiness, relative to history.** The fact table has at least half the trailing
   twenty-eight-day median row count for this weekday. An absolute threshold of zero catches
   nothing.
2. **Coverage.** Every expected distribution centre, warehouse, or business unit is present.
   A plan built from four of six DCs is not a small plan; it is a wrong plan with a plausible
   size.
3. **Freshness.** The newest source record is no older than the schedule allows, derived as in
   Topic 06, not chosen as a round number.
4. **Reconciliation.** The fact total matches a control total from the source — order count,
   inventory value, or unit count — within a stated tolerance.

A gate that fails must fail the run, not warn. A warning on a gate is a comment, and comments
do not stop a plan.

*If skipped:* the pipeline is green precisely when the input is empty, because every
transformation is correct on zero rows.

### 5.7 Step 7 — Define the plan version and the freeze

A published plan needs an identity that survives the night. At minimum, record the plan
identifier, the data interval, the cutoff it was produced for, the run identifier that
produced it, and the input snapshot identifiers. Write it once and never update it.

When the plan must change — a late order, a corrected inventory position — publish a new
version with a reason code and a supersession pointer. The question "what did we plan on
Tuesday, and what did we know when we planned it" must be answerable without reconstructing
the warehouse.

*If skipped:* the plan table is a mutable current-state table, incident reviews cannot
distinguish a modelling error from a data error, and audit cannot reconstruct a decision.

### 5.8 Step 8 — Treat backfill as a first-class operation

A backfill is not "running the DAG for an old date". It is a transaction with four
obligations.

| Obligation | Question to answer before launching | Default if unanswered |
|---|---|---|
| Scope | Which intervals, and which tasks within them? | The whole graph, for every date, which is rarely what is wanted |
| Order | Oldest first or newest first? | Newest first, which is wrong when a downstream model depends on history |
| Downstream rebuild | Which marts and plan versions consumed the repaired partition? | None, so the mart stays inconsistent |
| Isolation | Can the backfill write to production while the nightly runs? | No, and the two collide at 02:15 |

The practical rule is to backfill into a separate schema or with a distinct run identifier,
verify, and then promote. The alternative — writing directly to production at 22:00 while
the nightly is queued — produces a mart whose partitions were built from different code
versions, which is worse than the error being repaired.

*If skipped:* the repair is invisible to everything downstream, and the mart contains
partitions that disagree with the facts they were built from.

### 5.9 Step 9 — Instrument for lateness and correctness separately

Emit four timestamps per run, at minimum, and store them where a query can reach them.

| Timestamp | Meaning | Used for |
|---|---|---|
| Interval start | The first instant of the data interval | Identity and reproducibility |
| Scheduled start | When the run was due to start | Detecting scheduler delay |
| Actual start | When the first task began | Queue and slot contention |
| Publish instant | When the plan version became readable | The only lateness measure that matters |

Lateness is `publish instant - cutoff`, signed. Correctness is a separate boolean produced by
the gate and by post-publication tests. Storing them in one status column destroys the ability
to say "we were late but right", which is the state to aim for when something goes wrong.

*If skipped:* every incident review argues about whether the run was late, using three
different definitions, and never reaches the question of whether it was wrong.

### 5.10 Step 10 — Alert on lateness with a timer and on correctness with a test

Two mechanisms, two channels, two runbooks.

- **Lateness** is detected by a deadline: a timer that fires at `cutoff - buffer` if the plan
  version does not yet exist. It must be independent of task state, because a hung task never
  fails and never triggers a failure callback. Airflow's migration notes make this point about
  its older service-level agreement: it was evaluated at run completion, so a run that never
  finished was never checked.
- **Correctness** is detected by a test: the gate, plus post-publication checks on the
  published artifact rather than on the tables it came from.

Route them differently. A late-but-correct plan usually needs a person to decide whether to
publish late or use the fallback, which is an operational decision. A correct-but-wrong plan
needs the pipeline stopped and the plan withdrawn, which is an incident.

*If skipped:* the only alert is "task failed", which fires for neither of the two states that
matter, and the organisation learns about both from the planning meeting.

## 6. Worked example

Arden Cold Chain is a chilled-foods distributor. It runs six distribution centres (DCs),
14,200 item-locations, and a daily replenishment plan for a seven-day operation, drawing
stock positions from a warehouse management system (WMS). The plan meeting is at 06:00 local
time in New York. Everything below is illustrative but internally consistent.

### 6.1 The nightly chain

Eleven tasks, four source extracts in parallel, one gate, and a publish step. Durations are
p95 observed over the preceding quarter, rounded to the minute. All times are local wall
clock on a normal February night.

| ID | Task | Duration (min) | Predecessors | Earliest start | Earliest finish | Latest start | Latest finish | Slack (min) |
|---|---|---:|---|---:|---:|---:|---:|---:|
| T1 | Extract ERP orders | 35 | — | 02:15 | 02:50 | 02:15 | 02:50 | 0 |
| T2 | Extract WMS stock | 20 | — | 02:15 | 02:35 | 02:30 | 02:50 | 15 |
| T3 | Extract open purchase orders | 25 | — | 02:15 | 02:40 | 02:25 | 02:50 | 10 |
| T4 | Extract forecast | 15 | — | 02:15 | 02:30 | 02:35 | 02:50 | 20 |
| T5 | Conform and deduplicate | 20 | T1–T4 | 02:50 | 03:10 | 02:50 | 03:10 | 0 |
| T6 | Build inventory position fact | 25 | T5 | 03:10 | 03:35 | 03:10 | 03:35 | 0 |
| T7 | Run demand plan | 40 | T6 | 03:35 | 04:15 | 03:35 | 04:15 | 0 |
| T8 | Run supply plan | 55 | T7 | 04:15 | 05:10 | 04:15 | 05:10 | 0 |
| T9 | Build exception report | 12 | T8 | 05:10 | 05:22 | 05:10 | 05:22 | 0 |
| T10 | Publish plan version | 8 | T9, T11 | 05:22 | 05:30 | 05:22 | 05:30 | 0 |
| T11 | Reconcile to general ledger | 15 | T6 | 03:35 | 03:50 | 05:07 | 05:22 | 92 |

The critical path is T1 → T5 → T6 → T7 → T8 → T9 → T10: 35 + 20 + 25 + 40 + 55 + 12 + 8 =
195 minutes, or 3 hours 15 minutes. The last task finishes at 05:30, leaving thirty minutes
before the cutoff. Working backward from 05:30 gives the latest start of 02:15.

Two features matter. T11 has 92 minutes of slack, exactly the kind of task that should absorb
a retry. And the extracts' 10, 15, and 20 minutes of slack is real only because T5 waits for
all four; serialized, their durations would add and the schedule would move two hours later.

### 6.2 What lateness the schedule tolerates

The operations team asks "how late can the WMS be and still get a plan". The answer is the
source task's slack plus the end buffer, because the buffer is available to any overrun once.

| Source | Slack (min) | End buffer (min) | Lateness tolerated before the 06:00 cutoff is missed |
|---|---:|---:|---:|
| ERP orders (T1) | 0 | 30 | 30 |
| WMS stock (T2) | 15 | 30 | 45 |
| Open purchase orders (T3) | 10 | 30 | 40 |
| Forecast (T4) | 20 | 30 | 50 |

This table is more useful than a service-level percentage, because it is stated in the units
the source-system owner controls. "The WMS extract may be up to forty-five minutes late" is
an operational agreement. "The plan is available 99% of the time" is not.

> **Note:** the buffer is counted once, not once per source. If two sources are late in the
> same night, the plan misses when the *sum* of their excess lateness exceeds thirty minutes
> plus their slack, not when either alone is. The table is a single-source statement.

### 6.3 The retry that does not fit

The supply plan (T8) is configured with two retries and a ten-minute backoff, which is the
platform default. Test it against the schedule.

The work downstream of T8 is T9 (12 minutes) and T10 (8 minutes), so 20 minutes must remain
after a successful T8. A retry starting at instant `f` finishes at `f + 10 + 55`, and the plan
publishes at `f + 10 + 55 + 20`.

```text
admissible while   f + 10 + 55 + 20 <= 06:00
                   f <= 04:35
```

If the first attempt fails at 04:30 — the optimistic case, when everything ran on time and
T8 started at 04:15 — the retry publishes at 05:55 and the plan is five minutes early. If the
first attempt fails at 05:10, which is the *scheduled* finish, the retry publishes at 06:35
and misses by thirty-five minutes. The same policy is safe on a good night and useless on a
normal one, because the scheduled finish already consumes the entire buffer.

The correct policy is not a number. It is a rule evaluated at failure time: retry only if
`now + backoff + duration + downstream remaining <= cutoff - safety`. Encode it, or set
`retries=0` on T8 and define the fallback. A static `retries=2` on this task is a promise the
schedule cannot keep.

### 6.4 The DST night

The run is scheduled at 02:15 local, which is Eastern Standard Time (EST) or Eastern Daylight
Time (EDT) depending on the date. In 2026, US daylight saving time begins on Sunday 8 March
and ends on Sunday 1 November. The 02:15 slot sits inside the 02:00–03:00 gap on the March
date, and after the repeated 01:00–02:00 hour on the November date.

| Date | Transition | Wall clock at 02:15 | Absolute offset | Consequence |
|---|---|---|---|---|
| 8 March 2026 | 02:00 → 03:00 | The time does not exist | 02:15 EST would be 07:15 UTC; the jump occurs at 07:00 UTC | Vixie cron fires the job at 03:00 EDT, fifteen minutes early in absolute time; some libraries skip it entirely, producing no plan |
| 1 November 2026 | 02:00 → 01:00 | Exists once | 02:15 EST = 07:15 UTC | Runs once in every implementation; a job moved earlier into 01:00–02:00 would run twice in several libraries |

The deeper arithmetic is about the cutoff, not the job. The cutoff is 06:00 local, which is
11:00 UTC under EST and 10:00 UTC under EDT. A job fixed in UTC cannot be three hours and
forty-five minutes before a cutoff that moves by an hour: whichever UTC time you pick is
correct on one side of the transition and off by sixty minutes on the other. Scheduling in
the same zone as the deadline preserves the relationship, at the price of a discontinuous
wall clock twice a year.

The design rule is therefore: schedule in the zone of the deadline, choose a start outside
01:00–03:00 where the business allows it, and test both transition dates. The test is not
"does it run"; it is "does exactly one run exist per interval, and does the published plan
carry the right interval".

### 6.5 The green run on empty input

On 18 February the ERP extract selected from a view that had been rebuilt and renamed. The
query returned no rows and no error. The run published at 05:12, forty-eight minutes early.

The gate would have had four independent opportunities to stop it, each cheap to compute.

| Assertion | Expected | Observed | Threshold | Result |
|---|---|---:|---:|---|
| Order lines in the fact | 41,200 (trailing 28-day weekday median) | 0 | ≥ 20,600 (50%) | Fail |
| DCs represented | 6 | 0 | 6 of 6 | Fail |
| Newest source record age at 03:10 | 1 h 40 min | 26 h 10 min | ≤ 4 h | Fail |
| Fact value versus ERP control total | equal | 0 versus €4.1m | within 0.5% | Fail |

Any one of these stops the plan. None of them existed, because the DAG was written as a
sequence of transformations and every transformation was correct.

### 6.6 What the backfill owes downstream

On 12 February a purchase order arrived at 14:00 with an order date of 11 February. The
correct repair is not "rerun the 11 February DAG". It is four operations in order.

1. Rebuild the 11 February fact partition from the corrected extract.
2. Rebuild every mart partition derived from it, including the weekly aggregate that spans
   11 February.
3. Re-run the plans for 11 and 12 February, into a staging schema, and compare the exception
   lists.
4. Publish new plan versions with a reason code, superseding the old ones, only if the
   difference is material. Do not overwrite the originals.

Step 4 is the one teams skip. Overwriting an acted-upon plan destroys the only record of what
the business was told, and it makes the next incident review unable to separate a data
correction from a modelling change.

## 7. Code, query, or calculation

One code block, because the declarations carry more meaning than the body.

```python
# Airflow 3.x. A run is a claim about a data interval, never about "now".
with DAG(
    dag_id="plan_daily",
    schedule="15 2 * * *",                # 02:15 America/New_York: inside the 8 Mar spring gap
    start_date=datetime(2026, 1, 5, tzinfo=ZoneInfo("America/New_York")),
    catchup=False,                        # backfills are deliberate, never automatic
    max_active_runs=1,                    # one writer per interval
    default_args={
        "retries": 2,                     # extracts: 15-20 min slack each
        "retry_delay": timedelta(minutes=10),
        "execution_timeout": timedelta(minutes=40),   # p99 35 min, not the default
    },
) as dag:
    orders = extract("erp_orders", partition="{{ data_interval_start | ds }}")  # delete+insert
    stock = extract("wms_stock", partition="{{ data_interval_start | ds }}")
    pos = extract("open_pos", partition="{{ data_interval_start | ds }}")
    fcst = extract("forecast", partition="{{ data_interval_start | ds }}")
    fact = build_fact(conform([orders, stock, pos, fcst]))
    gate = assert_complete(fact)          # row count, DC coverage, freshness, control total
    demand = demand_plan(fact)
    supply = supply_plan(demand, retries=0)   # no retry fits at the scheduled 04:15 start
    publish = publish_plan(supply, as_of="{{ data_interval_start | ds }}")
```

Four declarations do most of the work, and each is a design decision rather than syntax.

| Declaration | What it asserts | What breaks without it |
|---|---|---|
| `schedule="15 2 * * *"` | The cadence, in a named zone | A UTC string drifts an hour against a local cutoff |
| `catchup=False` | Missed intervals are not silently replayed | Re-enabling a paused DAG creates a queue of old runs |
| `max_active_runs=1` | One writer per interval | A slow night overlaps the next and two runs write the same partition |
| `retries=0` on supply | The schedule, not the platform, decides retryability | A default retry runs past the cutoff and is killed mid-write |

The gate is the node that makes the graph a specification. Without it the graph asserts only
that each step ran, which is compatible with every step having run on nothing.

## 8. Data, standards, and regulatory context

Orchestration is governed less by regulation than by standards for the two things it
manipulates: instants and records. The regulatory content enters when the output of a
pipeline becomes a record that an auditor or a regulator may examine.

### 8.1 Time and schedule standards

| Standard | Governs | What it constrains | Grade |
|---|---|---|---|
| International Organization for Standardization (ISO) 8601 | Date and time representation | Instants, offsets, intervals, week dates; the format of every logical date | B |
| Request for Comments (RFC) 3339 | Internet timestamps | The subset of ISO 8601 used in APIs and logs, including the `Z` suffix | B |
| Internet Assigned Numbers Authority (IANA) time zone database | Civil time rules | Zone identifiers and the historical and future offset transitions | B |
| Portable Operating System Interface (POSIX) crontab specification | Cron syntax | Five time fields and the command; says nothing about DST or time zones | B |
| RFC 5545, section 3.3.5 | Calendar recurrence | An ambiguous local time refers to its first occurrence; a nonexistent one is handled by the implementation | B |
| Vendor timetable semantics | Scheduler behaviour | Whether a schedule is a trigger or a data-interval generator, and how catchup behaves | D |

The POSIX standard is the important entry. It defines the syntax that every cron
implementation accepts and leaves the behaviour at a DST transition undefined. That is not an
oversight in the standard; it is a boundary the standard drew, and every implementation filled
it differently. Any claim that "cron handles DST" is a claim about one implementation.

### 8.2 Records, audit, and control

When a plan drives purchasing commitments, inventory valuation, or a regulated decision, the
pipeline that produced it is inside the control environment.

| Requirement | Where it applies | Effect on the orchestrator | Grade |
|---|---|---|---|
| IT general controls over financial reporting | Entities subject to Sarbanes-Oxley or equivalent | Change management over DAG code, segregation between developer and operator, retained run logs | B |
| Electronic records and audit trails | U.S. Food and Drug Administration (FDA) regulated manufacturing and distribution, 21 CFR Part 11 | Attributable, time-stamped records; retained versions; controlled access | B |
| Computerised system validation | European Union good manufacturing practice (GMP) Annex 11 and equivalent | Documented, reproducible execution; change control over schedules | B |
| Data integrity principles (ALCOA+) | GxP environments | Attributable, legible, contemporaneous, original, accurate, complete, consistent, enduring, available | B |
| Retention obligations | Tax, customs, pharmacovigilance | Plan versions and input snapshots retained for the stated period | B |
| Personal data minimisation | General Data Protection Regulation (GDPR) and equivalent | Run logs must not retain personal data beyond need; parameters may contain identifiers | B |

The practical consequence is narrow and important: **if a plan version cannot be reproduced
from retained inputs and code, the pipeline cannot be audited, however good the plan was.**
Reproducibility is an orchestration property: pinned code, recorded interval identifiers, and
immutable outputs. A DAG that reads "latest" and overwrites its output fails every row above.

### 8.3 What is convention rather than requirement

Several practices in this area feel like rules and are not.

- **"The nightly batch runs at 02:00."** A habit from a mainframe window when interactive users
  were offline; it has no operational meaning for a cloud warehouse, and it places the job
  inside the DST transition window in most temperate zones.
- **"The planning cutoff is 06:00."** The instant is a business fact; the specific hour usually
  traces to when a meeting was scheduled decades ago.
- **"Catchup should be on."** Whether missed intervals should run automatically depends
  entirely on whether the DAG is written against its interval. There is no general answer.
- **"Retries make the pipeline reliable."** Retries make transient failures survivable and
  deterministic failures more expensive. The count is a schedule quantity, not a virtue.
- **"A green dashboard means the data is good."** Green means the process exited zero. It is a
  statement about the process, and the process can be correct about nothing.

## 9. Industry practice and case evidence

### 9.1 What the orchestration documentation codifies

Apache Airflow's best-practices page states the idempotency rule directly: tasks should
produce the same outcome on every re-run, an `INSERT` on a re-run creates duplicate rows and
should be an upsert, and tasks should read and write specific partitions using
`data_interval_start` rather than reading the latest available data. It also advises against
`datetime.now()` inside a task for critical computation, because the value changes between
attempts and between the original run and a backfill. (Grade D: vendor documentation, version
3.3.2, accessed 2 November 2026.)

The same documentation describes catchup as disabled by default in Airflow 3, with the
scheduler creating a run only for the latest interval. When catchup is enabled, the scheduler
creates a run for every interval that has not run since the start date, executes them
sequentially, and also does so when a DAG is re-enabled after being paused. That last clause
is the operational trap: pausing a DAG for a month and re-enabling it does not resume; it
queues a month of runs. (Grade D.)

Backfill in Airflow 3 is explicit, with a reprocess policy (never, failed only, or completed
and failed), independent concurrency control, reverse ordering, and a dry run that prints
candidate intervals. The dry run is the strongest statement in the documentation: the vendor
expects you to look before you leap. (Grade D.)

Dagster's documentation makes the complementary argument from the asset side: partitions are
a property of data rather than of tasks, and backfills re-materialise partitions and their
dependents, with a choice between one run per partition, batched runs, and a single run over a
range. Its tutorial demonstrates idempotent partitioning by deleting a partition before
inserting it. (Grade D: vendor documentation, accessed 2 November 2026.)

> **Note:** vendor documentation describes intended semantics, not observed reliability. It
> tells you what the tool will do if configured as documented. It does not tell you how often
> teams configure it that way, and there is no public dataset on the incidence of
> green-but-empty runs.

### 9.2 Deadline alerting: a documented design failure

Airflow's service-level agreement feature was removed in version 3.0 and replaced by deadline
alerts in 3.1. The design proposal behind the change (AIP-86) lists the reasons, and they read
as a catalogue of the problem this topic is about.

- The check ran when the DAG run finished, so a run that never finished was never checked.
- It only evaluated tasks that had at some point succeeded or been skipped.
- New or renamed tasks never had an agreement evaluated.
- Manually triggered and asset-triggered runs had no fixed interval to measure from.

The replacement evaluates deadlines from the moment the run starts, in the scheduler loop,
and fires the callback even if the run is still running or never completes. It also supports
a fixed-instant reference with a negative interval, which allows an alert *before* a deadline
rather than only after a budget has been exceeded. (Grade C: design proposal and vendor
documentation, 2025–2026.)

The lesson generalises beyond one tool. **Lateness must be checked by a clock that runs
independently of the thing being measured.** Any design in which lateness is derived from a
completion event cannot detect the case that matters most: the pipeline that is still running
at the cutoff and will never finish.

### 9.3 Cron and DST across implementations

A cross-implementation survey published as an architecture decision record in an open-source
cron library reports the following behaviour at the autumn transition. Read it as a survey of
library documentation, not as an audit of each project.

| Implementation | Spring forward (job in the gap) | Fall back (job in the repeated hour) | Grade |
|---|---|---|---|
| Vixie/ISC cron | Runs immediately after the jump | Fixed-time jobs run once; wildcard-hour jobs run twice | B |
| FreeBSD cron | Runs after the jump | Once by default; the `-s` flag forces exactly once for infrequent jobs | B |
| Quartz (Java) | Implementation-defined | Once | D |
| robfig/cron (Go) | Not specified in the survey | Twice; reported as unfixed since 2014 | D |
| APScheduler (Python) | Not specified | Twice; documentation recommends UTC | D |
| Celery Beat | Not specified | Raises an ambiguous-time error | D |
| systemd timers | Not specified | Skips the entire repeated hour | D |

Two conclusions follow. "Once" is not the default: a non-idempotent daily job inside the
repeated hour double-writes once a year in several popular stacks. The safe designs are
boring: schedule outside 01:00–03:00 local, or in UTC and accept that the job's distance from
a local cutoff shifts by an hour twice a year. Test both dates.

### 9.4 How planning systems actually schedule

Enterprise planning systems treat the planning run as a batch job with a window. Oracle
recommends running batch plans after work hours, not while users are working on the plan, and
allows only one instance of a given plan type and name at a time. Infor describes regeneration
as deleting the existing plan and replanning all items, advises running it at night, and notes
that changing certain parameters forces a regeneration rather than a net change; its M3
documentation says regenerative recalculation should not be used routinely because of its
capacity cost. A practitioner blog on Dynamics 365 Business Central reports that full
regenerative material requirements planning (MRP) commonly takes more than an hour and
recommends an overnight job queue. (All Grade D: vendor and practitioner documentation,
2025–2026.)

This is the origin of the cutoff convention. The planning run is expensive, it cannot safely
run while planners edit the plan, and its output is needed at the start of the working day.
The cutoff is the intersection of those three facts, not a data-engineering invention. A team
that moves it without consulting the planning run's duration and locking behaviour will
produce a plan that is late for a different reason.

### 9.5 The academic treatment of the cutoff

The cutoff has a small but real research literature in fulfilment operations. A 2013 Auburn
University doctoral dissertation on wave release strategies defines the planning cycle
explicitly as the interval between consecutive cutoff times, so that orders arriving between
two cutoffs are due by the next deadline, and uses a next-scheduled-deadline metric to measure
the fraction of orders shipped before a specific departure. It derives an optimal cutoff as a
function of utilisation and the number of waves, and observes that the cutoff should be no
later than the last wave release time. It also tabulates published retailer cutoffs from the
period, which vary by several hours across comparable businesses. (Grade B: scholarly
dissertation, not a peer-reviewed journal article.)

The transferable idea is that **the cutoff is a decision variable with an optimum, not a
constant inherited from history.** Too early and orders that could have shipped today roll to
tomorrow; too late and the plan misses its departure. The optimum depends on utilisation and
on the number of release waves — information a data team rarely has and a planning team rarely
writes down.

### 9.6 What the evidence does not establish

There is no public dataset on how often production pipelines succeed on empty input, how often
plans are published from stale data, or what fraction of retry policies exceed their schedule
slack. The mechanisms here are documented; the incidence is not. Treat any vendor claim of the
form "customers reduce failed runs by X%" as marketing until it names its measurement.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Green run on empty input | Row counts absent from monitoring | Every transformation is correct on zero rows | A plausible, authoritative, entirely wrong plan | Gate on row count versus trailing median, coverage, freshness, control total |
| Non-idempotent append on retry | Row counts drift upward after incidents | `INSERT` with no key, or a key containing ingestion time | Demand and inventory double-counted | Delete-insert per partition or merge on a business key; test by re-running |
| Wall-clock reads inside tasks | Backfills produce different results from the original run | `now()` or "latest partition" in transformation logic | Unreproducible history; training-serving skew in time | Pass the interval; fail when the named partition is absent |
| Backfill overwrites a published plan | Plan identifiers change retroactively | Mutable plan table updated in place | Audit cannot reconstruct what was told to the business | Immutable versions with supersession pointers |
| Catchup storm after a pause | Hundreds of queued runs after re-enabling a DAG | Catchup replays every missed interval sequentially | Hours of stale plans published in order, each overwriting the last | `catchup=False`; explicit backfill with a stated range |
| Retry on a deterministic failure | Same error text on every attempt | Policy ignores failure class | Cutoff consumed with no chance of success | Classify failures; fail fast on schema and constraint errors |
| Retry budget exceeds slack | Retry starts after the last admissible instant | Static retry count, dynamic start time | Task killed mid-write; partially replaced partition | Evaluate admissibility at failure time; zero retries on the critical path |
| DST double fire or skip | A missing or duplicated plan on two dates a year | Local-time schedule inside the transition window | No plan, or two plans, for one interval | Schedule outside 01:00–03:00; test both transition dates |
| Cutoff in server-local time | Effective deadline moves after an infrastructure change | The zone is inherited, not declared | Silent one-hour shift in the schedule | Declare the zone per metric and per DAG |
| Failure-only alerting | A hung run produces no page | Lateness derived from completion events | The worst outages are the quietest | Independent deadline timer at cutoff minus buffer |
| Late and wrong share a channel | Operators cannot triage the page | One status enum for two properties | Wrong plans treated as delays, or delays treated as incidents | Separate alerts, separate runbooks, separate owners |
| `all_done` trigger on publish | Plan publishes after an upstream failure | Trigger rule ignores upstream state | Green plan built from partial data | `all_success` on the publish path, with an explicit degraded-mode task if needed |
| Backfill not propagated downstream | Mart partitions disagree with their facts | Only the upstream model is rebuilt | Reports and plans built on inconsistent inputs | Rebuild the dependency closure; verify before promotion |

Two of these deserve a note.

**Green run on empty input** is the defining failure because it looks best. A failed run
generates a page; an empty-but-green run generates a dashboard that loads quickly. Monitoring
is rewarding the worse outcome, and the only defence is an assertion that the output is
non-trivial relative to history.

**`all_done` on publish** is a small choice with a large consequence. `all_done` runs a task
once all upstream tasks reach a terminal state, failure included. It is occasionally right for
cleanup, never for publishing a plan.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Task durations are stationary enough to schedule against | Backward scheduling needs a duration per task | Model change, data growth, warehouse contention, a new source | Plot p95 duration by week; alert on a 20% shift |
| The cutoff is fixed and known | Every slack calculation depends on it | A meeting moves, a carrier changes, a peak-season exception | Confirm the cutoff with the consumer each quarter, not each year |
| A plan is consumed once, at a known instant | Lateness has meaning only against a consumer | Analysts query the plan all day; some teams rerun it ad hoc | Ask the consumer what they do at 06:05 |
| Inputs arrive in a predictable window | Extracts start at fixed times | Source-system maintenance, upstream batch drift | Measure source-arrival distribution weekly |
| The scheduler's clock and the business's clock agree | Local scheduling assumes one zone | Distributed teams, daylight saving, infrastructure in another region | Assert the zone in configuration; test both transitions |
| Retries are safe because tasks are idempotent | Retry policy assumes repeatability | Append-only writes, side effects such as sending email or calling an API | Re-run on unchanged input and compare checksums |
| The DAG's dependencies are complete | Missing edges are invisible | A source added without an edge; an implicit dependency through a shared table | Review the graph against the data lineage, not against the code |
| Backfills are rare and supervised | Backfill design assumes human judgement | A month of restatements after a source correction | Measure backfill frequency; automate the common case |
| The warehouse can absorb concurrent runs | Concurrency limits are set for throughput | Month-end, quarter-end, a large backfill alongside the nightly | Load-test at peak, not at median |
| One plan version is authoritative at a time | Publish semantics assume a single writer | Multiple planners editing, regional overrides | Enforce with a uniqueness constraint, not a convention |

Three limits are worth stating plainly.

**The schedule is a model, and it goes stale.** Every duration is an estimate from a past
period. A schedule not re-derived quarterly describes how the pipeline ran last year, and the
gap is where missed cutoffs live.

**Orchestration cannot make data correct.** It can refuse to publish when assertions fail, but
it cannot distinguish a plausible wrong number from a right one. It catches empty, incomplete,
stale, and unreconciled inputs, not a correctly shaped dataset in the wrong units (Topic 11).

**The grid assumes one consumer with one deadline.** A plan may have a 06:00 deadline for
operations and a 09:00 deadline for finance. That is two SLOs against one artifact; state which
consumer each bound protects.

## 12. Exercises

### Exercise 1 — Backward schedule (calculation, 50 minutes)

A pipeline has seven tasks: extract A (40 min), extract B (25 min), extract C (18 min),
conform (22 min, after A, B, C), fact (30 min), model (45 min), publish (10 min). The cutoff is
07:00 and the buffer is 20 minutes.

**Deliverable:** a table with earliest start, earliest finish, latest start, latest finish, and
slack for every task; the critical path; the latest start of extract A.
**Acceptance criteria:** all times on the clock, not as durations; the backward pass takes the
minimum latest start across successors; the critical path is stated as a task sequence; the
latest start is given in both local time and UTC for a stated zone.
**Time estimate:** 50 minutes.

### Exercise 2 — Cutoff arithmetic under three policies (calculation, 30 minutes)

Using the same pipeline, compute the latest start for cutoffs of 06:00, 07:00, and 08:00.

**Deliverable:** three numbers, each with the arithmetic shown, plus one sentence on which
cutoff you would recommend and what it costs.
**Acceptance criteria:** the buffer is subtracted, not added; the answer is not rounded to a
convenient half hour without saying so; the recommendation names the trade-off between
planner working hours and pipeline slack.
**Time estimate:** 30 minutes.

### Exercise 3 — Idempotency audit (critique, 45 minutes)

Take one task you have written or maintained that writes to a table. List every way a second
run on unchanged input could change the target.

**Deliverable:** a table of failure mechanisms with the observed or expected row-count effect,
and the smallest change that would make the task idempotent.
**Acceptance criteria:** at least four mechanisms, including one involving an ingestion
timestamp and one involving a non-deterministic ordering; the fix is stated as a write
pattern, not as a wish; a test is specified that would fail before the fix.
**Time estimate:** 45 minutes.

### Exercise 4 — DST test plan (design, 40 minutes)

Write a test plan for a daily job scheduled at 02:30 local in `America/New_York` and another
in `Europe/Berlin`.

**Deliverable:** a table of the four transition dates in 2026, the expected local and UTC
instants, the expected number of runs under two different scheduler behaviours, and the
assertion that proves correctness.
**Acceptance criteria:** both zones covered; the spring and autumn cases distinguished; the
assertion is about the number of runs per interval, not about exit codes; one mitigation is
proposed that removes the hazard entirely.
**Time estimate:** 40 minutes.

### Exercise 5 — Retry policy (design, 40 minutes)

Given ten described failures — a lock timeout, a renamed column, a missing partition, a 503
from a carrier API, a null in a required field, an out-of-memory kill, a duplicate primary key,
a stale dimension, a timeout at the configured limit, and a permission error — classify each
and set a retry policy.

**Deliverable:** a table with failure, class, retry count, backoff, and the schedule budget it
consumes.
**Acceptance criteria:** at least two failures are classified as non-retryable with a reason;
one policy is justified by arithmetic against a stated slack; no policy retries an operation
with an external side effect without an idempotency key.
**Time estimate:** 40 minutes.

### Exercise 6 — Empty-input gate (design, 45 minutes)

Design the assertions that would have stopped the incident in section 6.5, for a fact table
you know or can describe.

**Deliverable:** four assertions, each with a threshold, a derivation of the threshold, and the
action on failure.
**Acceptance criteria:** no threshold is a round number without a derivation; at least one
assertion is about coverage rather than volume; the action is "fail the run", not "warn";
each assertion is computable in under two minutes.
**Time estimate:** 45 minutes.

### Exercise 7 — Backfill plan for a restatement (design, 55 minutes)

A supplier corrects three months of delivery confirmations. Design the repair.

**Deliverable:** an ordered plan covering scope, staging, downstream rebuild, verification, and
plan-version handling.
**Acceptance criteria:** the dependency closure is stated, not assumed; the plan does not
overwrite acted-upon versions; a rollback or abort condition is named; the estimated runtime is
compared against the nightly window.
**Time estimate:** 55 minutes.

### Exercise 8 — Two runbooks (design, 35 minutes)

Write the on-call runbooks for a late plan and for a wrong plan.

**Deliverable:** two one-page runbooks, each with detection, first five actions, decision
authority, and communication.
**Acceptance criteria:** the two do not share a first action; the late runbook names the
fallback explicitly; the wrong runbook names who may withdraw a published plan; each states
what evidence is retained.
**Time estimate:** 35 minutes.

### Exercise 9 — SLO definition and measurement (application, 40 minutes)

Write an SLO for a daily plan and the query that measures it.

**Deliverable:** the SLO statement, the allowed number of misses over a 90-day window, and a
query over a run-history table.
**Acceptance criteria:** the percentile, the window, the population of scheduled runs, and the
exclusion policy are all stated; the allowed-miss arithmetic is shown; the query counts from
the publish instant, not the run start.
**Time estimate:** 40 minutes.

### Exercise 10 — Postmortem (critique, 50 minutes)

Given the timeline in section 6.5 — green at 05:12, empty pick list at 09:00 — write the
postmortem.

**Deliverable:** a timeline with timestamps, three contributing controls that were absent, and
one change that would have converted the incident into a late-but-correct run.
**Acceptance criteria:** the analysis distinguishes the trigger from the contributing causes;
the recommended change is a control, not a reminder; the postmortem states what would still
not be caught afterwards.
**Time estimate:** 50 minutes.

## 13. Self-check answers

**Calculation 1 — the brief's cutoff arithmetic.** A final task takes 3 hours, the cutoff is
06:00, and the buffer is 30 minutes.

```text
latest finish of the final task = 06:00 - 00:30 = 05:30
latest start  of the final task = 05:30 - 03:00 = 02:30
```

The final task must start no later than 02:30 local. If the zone is `America/New_York` in
February, that instant is 07:30 UTC. State both, because the person running the job and the
person reading the logs may be in different zones.

**Calculation 2 — the section 6 schedule.**

```text
critical path = T1 + T5 + T6 + T7 + T8 + T9 + T10
              = 35 + 20 + 25 + 40 + 55 + 12 + 8 = 195 min = 3 h 15 min
publish latest finish = 06:00 - 00:30 = 05:30
latest start of T1     = 05:30 - 3:15  = 02:15
```

Slack for a parallel root is `latest start - earliest start`. For T2, `02:30 - 02:15 = 15`
minutes; for T3, `02:25 - 02:15 = 10`; for T4, `02:35 - 02:15 = 20`. The sum of the four
extract durations is 95 minutes, but the conform step waits for the slowest, so the extracts
contribute 35 minutes to the critical path and 60 minutes of slack is distributed among the
rest.

**Calculation 3 — Exercise 1.**

```text
publish latest finish 07:00 - 00:20 = 06:40
publish (10)  LS 06:30
model   (45)  LS 05:45
fact    (30)  LS 05:15
conform (22)  LS 04:53
A       (40)  LS 04:13   <- critical path start
B       (25)  LS 04:28   slack 15
C       (18)  LS 04:35   slack 22
critical path = 40 + 22 + 30 + 45 + 10 = 147 min = 2 h 27 min
04:13 + 2:27 = 06:40
```

A common wrong answer forgets the buffer and subtracts the 147-minute path from 07:00, giving
04:33 and a plan that is twenty minutes late every night.

**Calculation 4 — lateness tolerance.** For a source task with slack `s` and a shared end
buffer `b`, the tolerated lateness is `s + b`. T2: `15 + 30 = 45` minutes. This is a
single-source figure; with two late sources, the plan misses when the excess lateness beyond
each source's slack sums to more than `b`.

**Calculation 5 — retry admissibility.** With a 55-minute task, a 10-minute backoff, and 20
minutes of downstream work, a retry beginning at `f` publishes at `f + 85`. The cutoff is
06:00, so `f <= 04:35`.

```text
worst-case overrun of one retry = 55 + 10 = 65 min
available end buffer            = 30 min
65 > 30  ->  the policy cannot be satisfied from the scheduled start time
```

Retry is admissible only if the first failure occurs before 04:35, which is 20 minutes after
the scheduled start of 04:15. That is a narrow window, and it closes earlier on any night when
an upstream task ran long.

**Calculation 6 — DST offsets.** The cutoff is 06:00 local in `America/New_York`.

```text
EST (UTC-05:00): 06:00 local = 11:00 UTC
EDT (UTC-04:00): 06:00 local = 10:00 UTC
difference: 60 min
```

A job that must start 3 h 45 min before the cutoff therefore starts at 07:15 UTC under EST
and 06:15 UTC under EDT. No single UTC cron expression satisfies both. On 8 March 2026 the
02:15 local slot does not exist; the jump occurs at 02:00 EST, which is 07:00 UTC, so a Vixie
cron job that fires immediately after the jump starts fifteen minutes before its nominal
absolute time, while implementations that skip the slot produce no plan at all. On 1 November
2026 the 02:15 slot exists once and runs once everywhere; the double-fire hazard belongs to a
job inside the repeated 01:00–02:00 hour, which is exactly where a start moved earlier for
more slack would land.

**Calculation 7 — the SLO percentile at small n.** A six-day operation has 313 scheduled runs
in a year, or about 77 in a 90-day window.

```text
99.0% of 77 runs = 76.23 successes required  -> 77 of 77, i.e. zero misses
98.7% of 77 runs = 76.0  successes required  -> one miss allowed
```

A 99% target over a quarter is therefore a zero-miss target, intended or not. To tolerate one
miss per quarter, the honest number is 98.7%. Over 250 business days, 99% allows 2.5 misses:
two pass at 99.2%, three fail at 98.8%. Percentiles are coarse at small n, and an SLO without a
stated denominator cannot be interpreted.

**Calculation 8 — the empty-input threshold.** With a trailing 28-day weekday median of 41,200
order lines, a 50% floor is 20,600, and zero fails by 20,600 rows. The floor is deliberately
loose so a quiet day does not stop the plan; pair it with coverage, which catches normal
volume with one DC missing.

### 13.1 Design rubrics

| Artifact | Strong | Adequate | Weak |
|---|---|---|---|
| DAG | Every node names an assertion; the gate precedes the plan; the interval is the identity | Correct order and dependencies; gate present but only checks non-emptiness | A sequence of tasks with no assertions; success means "exited zero" |
| Retry policy | Failure classes stated; count derived from slack; admissibility evaluated at failure time | Counts differ by task and are justified informally | Platform default applied everywhere |
| Backfill plan | Scope, order, downstream closure, staging, verification, versioning | Scope and order stated; downstream handled by "rerun everything" | "Rerun the DAG for that date" |
| Alerting | Lateness on a timer, correctness on a test, separate channels and owners | Both exist but share a channel | Only task-failure alerts |
| SLO | Percentile, window, population, exclusions, measurement query | Percentile and window stated | "99.9% uptime" with no definition |

## 14. Interview questions

**1. "What is a DAG, in the context of a data pipeline?"**

A strong answer describes nodes as assertions about data state and edges as ordering
constraints, and notes that the graph's real job is to decide when the output may exist. A
weak answer says "a graph of tasks with no cycles" and stops at the definition.

**2. "How would you schedule a pipeline that must produce a plan by 06:00?"**

A strong answer schedules backward from the cutoff minus a buffer, uses p95 durations, computes
slack per task, and identifies the critical path before choosing a start time. A weak answer
picks a start time that "usually works" and adds retries.

**3. "What is idempotency, and how would you test it?"**

A strong answer separates determinism from the write pattern, names delete-insert or merge on
a business key, and proposes running twice on unchanged input and comparing row counts and a
checksum. A weak answer says "use upsert" without saying what the key is or how to verify it.

**4. "Why is `datetime.now()` a problem inside a task?"**

A strong answer explains that it breaks reproducibility and backfillability, and that two
attempts of the same task can see different data. A weak answer says it is a lint warning.

**5. "A pipeline runs green but the plan is wrong. How is that possible?"**

A strong answer gives the empty-input mechanism, or a partial-coverage mechanism, and notes
that every transformation is correct on zero rows. It then names the assertion that would have
failed. A weak answer blames the data team or says "bad data".

**6. "How should retries be configured?"**

A strong answer classifies failures, computes the worst-case duration against slack, and notes
that a static count is wrong when the start time varies. A weak answer says "three retries with
exponential backoff" as though it were a universal constant.

**7. "What happens to your cron job at a daylight saving transition?"**

A strong answer states that POSIX leaves it undefined, gives at least two implementations that
differ, and proposes scheduling outside the window or in UTC with the offset consequence
stated. A weak answer says "cron handles it" or "use UTC" without acknowledging the cutoff
drift.

**8. "What is the difference between a backfill and a retry?"**

A strong answer distinguishes time pressure, scope, and downstream obligations, and says a
backfill must rebuild the dependency closure and not overwrite acted-upon plan versions. A weak
answer treats them as the same operation with a different date.

**9. "How would you alert on a pipeline that is late?"**

A strong answer uses a timer referenced to the cutoff that fires independently of task state,
and explains why completion-based alerting misses a hung run. A weak answer attaches an alert
to the failure callback.

**10. "Late but right, or on time but wrong?"**

A strong answer refuses the framing slightly: it prefers late and right, but only if a stated
fallback exists, and it argues that on-time-and-wrong is worse because it is acted upon
silently. A weak answer picks one without discussing the consumer.

**11. "How would you design a backfill for a month of corrected source data?"**

A strong answer states scope, order, staging, downstream rebuild, verification, and version
handling, and compares the runtime against the nightly window. A weak answer says "rerun the
month" without mentioning downstream artifacts or the risk of colliding with the nightly.

**12. "What does an SLO of 99% completion by 06:00 actually commit you to?"**

A strong answer converts it into an allowed number of misses for the actual population of
scheduled runs, and notes that 99% over a quarter of a six-day operation means zero misses. A
weak answer repeats the number.

## 15. Key terms

- **DAG** — a directed graph with no cycles, expressing task or asset dependencies.
- **Task** — one unit of work with declared inputs, outputs, and state.
- **DAG run** — one execution of the graph for one data interval.
- **Logical date** — the start instant of the data interval a run is responsible for.
- **Data interval** — the half-open period a run processes.
- **Catchup** — automatic creation of runs for intervals missed since the start date.
- **Backfill** — deliberate reprocessing of a stated set of past intervals.
- **Idempotency** — the same state after one application or many.
- **Partition** — the slice of a dataset one run owns and may replace.
- **Upsert** — insert or update on a declared key.
- **Delete-insert** — remove a partition and insert its recomputation, in one transaction.
- **Retry** — a further attempt after failure; **backoff** — the delay before it.
- **Poison task** — a task that fails deterministically on every attempt.
- **Poison message** — a record that always breaks the consumer and blocks the queue.
- **Dead-letter queue** — a holding queue for messages that exhausted their retries.
- **Execution timeout** — the wall-clock limit after which a task is killed.
- **Deadline alert** — a timer-based alert independent of task state.
- **SLO** — a stated percentile bound on an outcome over a window.
- **Data dependency** — a task may start only when a named input is complete.
- **Trigger rule** — the upstream-state condition under which a task runs.
- **Sensor** — a task that waits for an external condition.
- **Asset-aware scheduling** — triggering on an upstream data update rather than a clock.
- **Critical path** — the longest dependent chain, which sets the earliest finish.
- **Slack** — minutes a task may slip without delaying the cutoff.
- **Latest start time** — the last instant a task may begin and still meet the cutoff.
- **Cutoff** — the business instant by which the plan must exist.
- **Planning cycle** — the interval between consecutive cutoffs.
- **Plan freeze** — the instant a plan version becomes immutable.
- **Plan version** — an immutable identifier for one published plan.

## 16. Sources

**Standards and official specifications**

- ISO 8601-1:2019, *Date and time — Representations for information interchange*. (B)
- RFC 3339, *Date and Time on the Internet: Timestamps*. (B)
- RFC 5545, *Internet Calendars: iCalendar*, section 3.3.5 on recurrence and local time. (B)
- IANA Time Zone Database, current release; zone identifiers and transition rules. (B)
- POSIX crontab specification, IEEE Std 1003.1, Shell and Utilities, `crontab`. (B)
- Vixie cron manual page, `cron(8)`, including the daylight saving time paragraph. (B)
- FreeBSD `cron(8)` manual page, including the `-s` flag. (B)

**Scholarly**

- Čeven, E. (2013). *Wave Release Strategies for Order Fulfillment Systems with Deadlines*.
  Doctoral dissertation, Auburn University. Defines the planning cycle between consecutive
  cutoffs and derives an optimal cutoff from utilisation and wave count. (B)

**Vendor and product documentation**

- Apache Airflow, "Best Practices," version 3.3.2, accessed 2 November 2026. Idempotency,
  partition-scoped reads, and the warning against `datetime.now()`. (D)
- Apache Airflow, "DAG Runs," version 3.3.2, accessed 2 November 2026. Catchup default,
  data intervals, and sequential execution of missed runs. (D)
- Apache Airflow, "Backfill," version 3.3.2, accessed 2 November 2026. Reprocess policies,
  concurrency, ordering, and dry run. (D)
- Apache Airflow, "Data-aware Scheduling" and "Asset-Aware Scheduling," version 3.3.2,
  accessed 2 November 2026. (D)
- Apache Airflow, "Deadline Alerts" and "Migrating from SLA to Deadline Alerts," version
  3.3.2, accessed 2 November 2026. (D)
- Apache Software Foundation, AIP-86, "Deadline Alerts (Formerly SLA)," design proposal. (C)
- Dagster, "Backfilling data," "Partition backfill strategies," and "What are
  software-defined assets?", accessed 2 November 2026. (D)
- Oracle, "Batch Run Plan," Fusion Cloud Supply Chain and Manufacturing, release 25d. One
  instance per plan type and name; after-hours scheduling recommended. (D)
- Infor, "Running MRP Planning," CloudSuite Industrial 2026.10. Regeneration semantics and
  overnight scheduling. (D)
- Infor, "Planning Frequency," M3 16.x. Batch net change versus regenerative recalculation. (D)
- IBM, "Shipping cutoff times" and "Pickup times and schedules," Sterling Intelligent
  Promising, accessed 2 November 2026. Node and shipping-group cutoffs. (D)

**Practitioner and engineering**

- Simren, O. "Schedule MRP," Dynamics 365 Business Central blog, accessed 2 November 2026.
  Reports full regenerative MRP commonly exceeding one hour and recommends an overnight job
  queue. (E)
- netresearch/go-cron, "ADR-016: DST Normalization" and "DST_HANDLING.md," accessed
  2 November 2026. Cross-implementation survey of cron DST behaviour; a library's own
  architecture record, not an audit of each project. (D)
- Microsoft Azure Architecture Center, "Idempotent Consumer Pattern," accessed 2 November
  2026. Deduplication keys, transactional inbox, and unique constraints. (D)
- Stack Overflow, "Daylight Savings and Cron," question 13195999, accessed 2 November 2026.
  Practitioner reports; used for orientation only. (E)

**Note on grading.** Grades follow Module 28: A peer-reviewed or audited, B regulatory,
official, or scholarly, C framework or design proposal, D vendor documentation, E press, blog,
or anecdote. Vendor pages are dated at the version consulted. The cross-implementation DST
table reports one library's survey of other projects, not each project's own documentation. No
public dataset exists on green-but-empty run incidence, and no such figure is asserted here.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Prerequisite topic: [`01_layered-architecture.md`](01_layered-architecture.md)
- Prerequisite topic: [`02_batch-microbatch-streaming.md`](02_batch-microbatch-streaming.md)
- Related sibling: [`06_dbt-models-tests.md`](06_dbt-models-tests.md)
- Related sibling (planned, not yet written): `11_quality-quarantine-lineage.md`
- Time zone and calendar foundations:
  [`../01A_foundations_units_time_assumptions/04_calendars-weeks-time-zones.md`](../01A_foundations_units_time_assumptions/04_calendars-weeks-time-zones.md)
- Survey file: [`../../../course/03_data_engineering_interoperability.md`](../../../course/03_data_engineering_interoperability.md)
- Return to the part map: [`../_index.md`](../_index.md)
