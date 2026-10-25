# Orchestration

> Module 03 — Data Engineering and Interoperability
> Topic 04 of 10
> Estimated study time: 3.5 hours
> Prerequisites: 02, 03
> Version 2026.3

## 1. Why this topic exists

A pipeline that runs once is a script. A pipeline that runs every night for three years is a system with
a scheduler, a state machine, a failure policy, and an audit trail. Most biomedical data incidents happen
in that system, not in the transformation logic.

The pattern is recognizable.

- A job is scheduled at 02:00 local time. On the spring daylight-saving transition, 02:00 does not exist.
  The scheduler skips the run or fires at 03:00, and one day of data is never processed.
- A task fails after writing half its output. The retry appends the full output again. The partition now
  contains 1.5 times the correct rows, and every downstream rate is inflated.
- An analyst triggers a backfill for the last two years. The backfill writes to the same table as the
  nightly job. The two runs interleave, and the table contains a mixture of old and new logic.
- A sensor waits for a file that never arrives. The task retries for 24 hours, occupying a worker slot,
  and the rest of the DAG never starts.
- A pipeline succeeds every night but processes the previous day's data because the logical date was
  interpreted as the processing date. Every feature is shifted by one day, and the model's performance
  degrades slowly enough that nobody notices for a quarter.
- A retry succeeds, but the alert already fired and the on-call engineer restarted the whole DAG,
  producing two concurrent runs that overwrite each other.

These are orchestration failures. The SQL was correct, the Python was correct, and the data were
available. The scheduler's semantics were not understood.

This topic teaches the semantics that matter: what a logical date means, how to make a task idempotent,
how a backfill differs from a re-run, how retries interact with partial writes, and how to schedule
clinical pipelines so that a daylight-saving transition does not delete a day.

> **Note:** The examples target Apache Airflow 3.x, which is the version line current in 2026. The
> concepts transfer to Dagster, Prefect, Argo Workflows, and cloud schedulers, but the vocabulary and the
> default behaviors differ. Always state the orchestrator and version when describing a pipeline.

## 2. Learning outcomes

By the end of this topic you can:

1. **Explain** the difference between logical date, data interval, and wall-clock execution time, and
   show how each affects a partition. Assessed in Exercise 1 and self-check item 2.
2. **Design** an idempotent task that can be retried and backfilled without duplicating output. Assessed
   in Exercise 2.
3. **Choose** a schedule that is correct across daylight-saving transitions and time zones. Assessed in
   Exercise 3 and self-check item 4.
4. **Compute** backfill duration and concurrency from task runtime and partition count. Assessed in
   Exercise 4.
5. **Specify** retry, timeout, and alerting policies that distinguish transient from permanent failures.
   Assessed in Exercise 5.
6. **Handle** late-arriving data with a reprocessing window and a documented completeness rule. Assessed
   in Exercise 6.
7. **Write** an Airflow DAG that avoids top-level work, passes references rather than data, and uses
   `dag.test()` for local validation. Assessed in Exercise 7.
8. **Diagnose** a duplicate-output incident by tracing run IDs, partitions, and write semantics. Assessed
   in Exercise 8.

## 3. The core mental model

An orchestrator is a state machine over partitions. The unit of work is not "the pipeline"; it is "the
pipeline for one data interval."

```text
                     +-------------------+
                     |  DATA INTERVAL    |
                     |  [2026-10-24 00:00|
                     |   , 2026-10-25)   |
                     +---------+---------+
                               |
        +----------------------+----------------------+
        |                      |                      |
+-------v--------+    +--------v--------+    +--------v--------+
| scheduled      |--->| running         |--->| success         |
+-------+--------+    +--------+--------+    +-----------------+
        |                      |
        |                      +---- failure ----+
        |                                       |
        |                              +--------v--------+
        |                              | retry (n left)  |
        |                              +--------+--------+
        |                                       |
        |                              exhausted  |
        |                                       |
        |                              +--------v--------+
        +------------------------------> failed          |
                                       +-----------------+
```

Every run has an identity: DAG ID, run ID, logical date, and data interval. Two runs with the same
identity must not execute concurrently. Two runs with different identities may write the same partition
if the task is not partition-scoped, which is the root cause of most duplicate-output incidents.

### 3.1 Logical date is not wall-clock time

The logical date (called `execution_date` in Airflow 2 and `logical_date` in Airflow 3) is the start of
the data interval that a run processes. A run triggered at 02:00 on 25 October with a daily schedule
processes the interval `[24 October 00:00, 25 October 00:00)`. The logical date is 24 October, not 25
October.

This is the single most misunderstood concept in orchestration. A task that uses the wall-clock date to
filter data processes the wrong day. A task that uses the logical date processes the intended interval,
even when it runs late, early, or during a backfill.

### 3.2 Idempotency is a property of the write, not the task

A task is idempotent if running it twice for the same partition produces the same final state. There are
three reliable patterns:

1. **Partition replace.** Delete the partition and insert the new rows in one transaction. Re-running
   replaces rather than appends.
2. **Upsert on a deterministic key.** Merge on the partition key and a record key, with a rule for which
   version wins.
3. **Write to a new immutable file and swap the pointer.** The old file remains for rollback; the table
   metadata points to the new one.

The unreliable patterns are append-only inserts, `INSERT` without a key, and writes to a shared staging
table that multiple runs use.

### 3.3 Retries repeat the whole task

A retry re-executes the task from the beginning. It does not resume from the point of failure. If the
task wrote 40% of its output before failing, the retry starts again and writes another 40%, unless the
write is idempotent. This is why retries and idempotency are the same design decision.

A retry also re-runs side effects: sending a notification, calling a rate-limited API, or charging a
credit card. A task with external side effects needs either an idempotency key for the side effect or a
separate task that is not retried automatically.

### 3.4 Event time, processing time, and completeness

```text
event time        t0 ------ t1 ------ t2 ------ t3 ------ t4
arrival                  t1        t3        t3        t6
processing run                     [run at t4]        [run at t6]

At t4, the run sees events t0..t3. The event at t2 arrived late at t6.
A pipeline that closes the partition at t4 is incomplete by one event.
A pipeline that reprocesses [t0, t4) at t6 is complete for that window.
```

Orchestration must decide when a partition is complete. Options: close at a fixed latency after the
interval ends, wait for a watermark, or reprocess a trailing window on every run. Each has a cost: fixed
latency misses late data, watermarks require source cooperation, and reprocessing costs compute.

### 3.5 The DAG is a contract with the operations team

A DAG declares dependencies, retries, timeouts, and ownership. It is also a promise: if this task fails,
this alert fires, this runbook applies, and this person is responsible. A DAG without an owner, a
timeout, or an alert is a script with a calendar entry.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or standard |
|---|---|---|---|
| DAG | Directed acyclic graph of tasks | Called a pipeline | — |
| Task | A unit of work in a DAG | Confused with a task instance | — |
| Task instance | One execution of a task for one run | Called a task | — |
| Operator | The implementation class of a task | Used for any function | — |
| Logical date | Start of the data interval a run processes | Confused with run time | ISO 8601 with time zone |
| Data interval | `[start, end)` of the data a run processes | Confused with the run's duration | — |
| Run ID | Unique identifier of a DAG run | Assumed to equal the logical date | String |
| Schedule | Cron expression, timedelta, or timetable | Assumed to be time-zone safe | Cron, ISO 8601 |
| Catchup | Whether missed intervals are scheduled automatically | Confused with backfill | Boolean |
| Backfill | Explicit execution of past intervals | Confused with a retry | — |
| Retry | Re-execution of a failed task | Confused with a backfill | Count |
| Retry delay | Wait between attempts | Set to zero by default | Duration |
| Exponential backoff | Increasing delay between retries | Confused with a fixed delay | Duration |
| Timeout | Maximum runtime before the task is killed | Absent by default in many operators | Duration |
| SLA | Service-level agreement on completion time | Confused with a timeout | Duration or clock time |
| Sensor | Task that waits for a condition | Used without a timeout | — |
| Deferrable operator | Operator that releases its worker while waiting | Confused with async | — |
| Trigger rule | Condition under which a task runs | Left at `all_success` unintentionally | Enum |
| XCom | Small message passed between tasks | Used for large data | Bytes |
| Pool | Limit on concurrent slots for a class of tasks | Confused with parallelism | Slots |
| Concurrency | Maximum parallel tasks or runs | Confused with throughput | Count |
| Idempotency key | Deterministic identifier for a write | Absent | — |
| Partition | Independently writable subset of data | Confused with a file | Date, key range |
| Watermark | Lateness threshold for completeness | Treated as exact | Duration |
| Late-arriving data | Events that arrive after their interval closed | Ignored | — |
| Reprocessing window | Trailing interval re-run on each execution | Confused with backfill | Days |
| Exactly-once | A write takes effect once | Confused with at-least-once delivery | — |
| At-least-once | A write may occur more than once | Assumed harmless | — |
| Dead letter | Storage for records that repeatedly fail | Absent | — |
| Asset | A data object a DAG produces or consumes | Confused with a task | URI |
| Lineage | Graph of datasets and jobs | Confused with a DAG diagram | OpenLineage |
| Runbook | Procedure for an operational event | Confused with documentation | — |

Two terms deserve expansion.

**Catchup versus backfill.** Catchup is a schedule property: when a DAG is unpaused or its start date
moves earlier, the scheduler creates runs for every missed interval. Backfill is an explicit command to
execute a range. A DAG with `catchup=True` and a start date two years ago will attempt 730 runs on first
deployment. That is rarely what a new pipeline owner wants, and it is a common cause of a first-day overload.

**Exactly-once.** True exactly-once delivery is rare. Most systems provide at-least-once delivery plus
idempotent writes, which produces exactly-once effects. The distinction matters because the guarantee
lives in the write, not in the message broker or the scheduler.

## 5. The mechanics, step by step

The procedure designs a scheduled clinical pipeline from the decision it supports. It assumes a daily
cadence; the same reasoning applies to hourly or streaming micro-batches.

### Step 1 — Identify the decision and its deadline

Input: the consumer and the decision.
Output: a required completion time and a maximum acceptable data age.

Example: the quality dashboard must be current by 06:00 each day, and the underlying data may be up to
26 hours old. That permits a nightly run with a two-hour buffer.

Reason: the schedule is derived from the decision, not from the source's convenience. A pipeline that
finishes at 09:00 for a 06:00 decision is late every day, even if it succeeds.

### Step 2 — Define the data interval

Input: the decision deadline and the source's event time.
Output: a half-open interval per run, with a stated time zone.

Example: daily interval `[00:00, 00:00)` in UTC, processing the previous calendar day. If the clinical
day is defined in local time, the interval must be expressed with the correct offset and the DST policy
stated.

Reason: the interval is the contract between the scheduler and every task. A task that filters on a
different interval produces a partition that does not match its label.

### Step 3 — Choose the schedule in a fixed time zone

Input: the data interval.
Output: a cron expression or timetable with an explicit time zone.

Prefer UTC for storage and scheduling, and convert for display. If a local-time schedule is required
because a clinical process starts at 02:00 local, use a timezone-aware timetable rather than a naive cron
expression.

Reason: cron in local time behaves differently across DST transitions. A 02:30 schedule has no valid
instant on the spring-forward day and two on the fall-back day in many zones.

### Step 4 — Set the start date and catchup policy

Input: the earliest data the pipeline should process.
Output: a start date and `catchup` true or false.

For a new pipeline, set `catchup=False` and backfill explicitly after validation. For a pipeline that
must fill history on deployment, set a bounded start date and a concurrency limit.

Reason: an unbounded catchup can create hundreds of runs at once, saturate the scheduler, and write
partitions that have never been tested.

### Step 5 — Declare the partition key

Input: the interval and the target table.
Output: a partition column or path, such as `partition_date`.

Every write must be scoped to the partition. A task that writes "the latest data" without a partition
predicate is not idempotent and cannot be backfilled safely.

Reason: the partition key is the join between the scheduler's identity and the storage layout. If they
do not match, a backfill writes over the wrong data.

### Step 6 — Design the extract to be replayable

Input: the source interface.
Output: an extraction that accepts an interval and returns the records for that interval, with a stable
order and a pagination strategy.

Use the source's event time or last-updated filter, not "all records." Record the extraction window and
the source's response metadata. If the source cannot filter by time, extract a snapshot and let the
transform select the interval.

Reason: replayability is what makes a backfill possible. An extract that only returns "today" cannot
reprocess yesterday.

### Step 7 — Land raw data immutably

Input: the extracted records.
Output: files under a path that includes source, interval, run ID, and content hash.

Never overwrite a raw extract. If a re-run produces a different extract for the same interval, land it as
a new object and let the transform choose. The raw zone is the evidence that the pipeline ran and what it
saw.

Reason: when the transform is wrong, the raw extract is the only way to rebuild without re-querying a
source that may have changed.

### Step 8 — Separate the stages

Input: the transformation logic.
Output: distinct extract, validate, transform, load, and test tasks.

Separate tasks give independent retries, clear failure attribution, and observable intermediate counts.
A single monolithic task that does everything is cheaper to write and impossible to diagnose.

Reason: when a pipeline fails, the first question is which stage failed. A monolith answers "somewhere."

### Step 9 — Make the load idempotent

Input: the transformed rows and the partition.
Output: a write that replaces or merges the partition deterministically.

Pattern: `DELETE FROM target WHERE partition_date = :d; INSERT ...` inside one transaction, or a `MERGE`
on the deterministic key. For file-based tables, write to a staging location and atomically replace the
partition.

Reason: retries and backfills are guaranteed to happen. The only question is whether they duplicate data.

### Step 10 — Choose the retry policy per task

Input: the failure modes of each task.
Output: retry count, delay, backoff, and which exceptions are retryable.

Transient failures (network timeouts, throttling, temporary database unavailability) deserve retries with
exponential backoff. Permanent failures (schema mismatch, invalid configuration, authorization denied)
should fail immediately; retrying them wastes time and delays the alert.

Reason: a blanket retry count treats a broken schema as a temporary glitch and hides it for an hour.

### Step 11 — Set timeouts

Input: the expected runtime distribution.
Output: a timeout per task, typically three to five times the p95 runtime.

A task without a timeout can hang for days, holding a slot and blocking downstream work. The timeout
should be long enough to survive a slow but valid run and short enough to free the slot before the
decision deadline.

Reason: orchestration failures are often stalls, not errors. A timeout converts a stall into a visible
failure.

### Step 12 — Define SLAs and alerts

Input: the decision deadline and the critical path.
Output: an SLA per DAG or task, an alert destination, and a runbook link.

Alert on the SLA, not only on task failure. A task that succeeds after the deadline is a failure from the
consumer's perspective. Route alerts to an owned channel, and include the run ID, the partition, and the
first failing task.

Reason: an alert without a runbook creates a second incident: the on-call engineer has to investigate
from scratch at 03:00.

### Step 13 — Handle late-arriving data explicitly

Input: the source's lateness distribution.
Output: a reprocessing window and a completeness flag.

Options: reprocess the last N days on every run; wait for a watermark; or accept the partition as
complete and publish a correction later. Whichever is chosen, the partition should carry a
`is_complete` or `data_through` field so consumers know what they are reading.

Reason: a dashboard that silently mixes complete and incomplete partitions produces trends that are
artifacts of arrival timing.

### Step 14 — Define the backfill procedure

Input: the reprocessing window and the partition key.
Output: a documented command, a concurrency limit, and a validation step.

Backfill in reverse chronological order if consumers need recent data first. Limit active runs to protect
the source. After the backfill, compare row counts and key metrics to the pre-backfill values, and record
the comparison.

Reason: a backfill is a controlled change to historical data. Without a before-and-after comparison, it is
indistinguishable from a data corruption event.

### Step 15 — Configure concurrency and pools

Input: source rate limits, warehouse capacity, and task runtimes.
Output: `max_active_runs`, pool sizes, and task-level concurrency.

A pool prevents a backfill from consuming every worker. A `max_active_runs` of one for a DAG that writes
a shared table prevents two runs from interleaving. For independent partitions, a higher limit improves
throughput.

Reason: concurrency is a safety control, not only a performance setting. The wrong limit causes either a
stalled pipeline or a saturated source.

### Step 16 — Set trigger rules deliberately

Input: the dependency structure.
Output: a trigger rule per task.

The default `all_success` is correct for most tasks. Use `all_done` for cleanup, `one_success` for
fallback paths, and `none_failed` when a skipped upstream task should not block. Do not use
`all_done` for a task that consumes data; it will run after a failure and process incomplete input.

Reason: trigger rules encode failure semantics. A cleanup task that runs on failure is correct; a
transformation that runs on failure is dangerous.

### Step 17 — Keep top-level code out of the DAG file

Input: the DAG definition.
Output: a file that only declares the DAG and its tasks.

Database connections, API calls, and heavy imports at module level run on every scheduler parse. They
slow the scheduler, can fail the whole DAG file, and may leak credentials into logs.

Reason: the scheduler parses DAG files frequently. Top-level work turns a parse into an execution.

### Step 18 — Pass references, not data, through XCom

Input: inter-task communication.
Output: a path, a URI, or a small identifier passed via XCom; the data itself stored externally.

XCom is stored in the metadata database and is not designed for large payloads. Passing a data frame
through XCom can exceed limits, slow the scheduler, and duplicate sensitive data into a system that may
not be in scope for PHI.

Reason: the orchestrator should coordinate work, not carry clinical records.

### Step 19 — Test the DAG locally

Input: the DAG file and a small fixture.
Output: a `dag.test()` run or an equivalent local execution that exercises dependencies and writes.

Test the DAG structure (no cycles, expected task count), the interval logic (the correct partition is
selected), and the idempotent write (two runs produce one partition). Do not test only that the file
imports.

Reason: a DAG that imports is not a DAG that works. Most orchestration defects are in the interval and
write logic, which an import test never exercises.

### Step 20 — Version, deploy, and monitor

Input: the tested DAG.
Output: a deployed DAG with a version identifier, lineage emission, and dashboards for duration, failures,
and freshness.

Record the code version and the image digest in the run metadata. Emit OpenLineage events or the
orchestrator's native lineage. Monitor task duration distributions, queue time, and SLA misses, not only
success and failure.

Reason: success rates hide degradation. A pipeline that succeeds in 20 minutes instead of 6 is heading
for an SLA breach, and the trend is visible before the breach.

### Step 21 — Write the incident procedure

Input: the failure modes identified in steps 10 through 14.
Output: a runbook with detection, triage, containment, recovery, and prevention.

Include the exact commands to inspect a run, clear a task, pause a DAG, and roll back a partition. Include
the escalation path and the criteria for declaring a data incident rather than a pipeline incident.

Reason: orchestration incidents happen outside business hours. A runbook is the difference between a
15-minute recovery and a three-hour investigation.

### Step 22 — Handle late partitions and superseded runs

Input: a partition whose interval has closed but whose data are still arriving.
Output: a reprocessing rule and a version marker.

Reprocess the trailing window, write a new partition version, and record `data_through`. Do not mutate
the prior version in place; keep it until the next successful reprocess. If two runs target the same
partition, the later run wins only if it has a higher run sequence and the partition lock is held.

Reason: late data are normal, but silent mutation of a closed partition makes a published number
unreproducible. The version marker tells the consumer which state they are reading.

### Step 23 — Handle partial commits

Input: a task that writes multiple objects or tables.
Output: a single commit point and a recovery rule.

Write to a staging location, validate, then commit atomically. If the commit fails, leave the prior state
intact and mark the run failed. Never leave a partition with some files from run A and some from run B.

| Failure point | State | Recovery |
|---|---|---|
| Extract fails | No landing object | Retry extract |
| Validate fails | Landing object exists, no write | Quarantine; retry after fix |
| Transform fails | Staging partial, target intact | Truncate staging; retry |
| Load fails mid-commit | Transaction rolled back | Retry load |
| Test fails after load | Target replaced, tests red | Restore prior snapshot; fix; reload |
| Publish fails | Consumer on prior version | Republish after validation |

Reason: a partial commit is a data corruption that looks like a successful run. The commit point is the
only defense.

### Step 24 — Handle clock and timezone boundaries

Input: the schedule and the interval predicates.
Output: a rule for each boundary.

Use UTC for storage and scheduling. If a local-time schedule is required, use a timezone-aware timetable
and test both DST transition dates. For interval predicates, use half-open intervals and an explicit
offset; never compare a naive timestamp to an aware one.

| Boundary | Example | Rule |
|---|---|---|
| Spring forward | 31 March 2024, 01:00 → 02:00 London | No 02:30 local run; use UTC |
| Fall back | 27 October 2024, 02:00 → 01:00 London | 02:30 occurs twice; run once in UTC |
| Month end | 31 January 23:59 | Half-open interval to 1 February 00:00 |
| Leap day | 29 February 2024 | Interval arithmetic, not day counts |
| Clock skew | Node clock 4 minutes fast | NTP; record event time, not node time |

Reason: timezone bugs are seasonal, so they survive testing for months and then fail once. The rule must
be encoded in the schedule, not in a comment.

### Branch point A — The pipeline is streaming

Replace the daily interval with a window and a watermark. Idempotency becomes a function of the window
and the event key, and the orchestrator may only manage deployment and monitoring rather than individual
events. The state machine is the stream processor's, not the scheduler's.

### Branch point B — The pipeline is a dbt project

Let dbt manage model dependencies and tests; let the orchestrator manage when dbt runs and what happens on
failure. Do not duplicate the model graph in the orchestrator. Use `dbt build --select` with selectors
rather than listing models manually.

### Branch point C — The pipeline spans sites

Run the same DAG at each site with a site parameter, and aggregate results in a separate DAG. Do not
orchestrate remote sites from one scheduler unless the network and governance model support it; a
scheduler outage at the center should not stop local clinical reporting.

## 6. Worked example

A hospital research data platform runs a nightly pipeline that ingests a FHIR server, transforms
observations into an OMOP-shaped measurement table, and refreshes a cohort. The quality dashboard reads
the cohort at 06:00.

### 6.1 Pipeline shape

| Task | Runtime p50 | Runtime p95 | Retries | Timeout |
|---|---:|---:|---:|---:|
| extract_bulk | 22 min | 48 min | 3 | 90 min |
| validate_resources | 8 min | 14 min | 1 | 30 min |
| transform_measurements | 35 min | 62 min | 2 | 120 min |
| load_partition | 12 min | 25 min | 2 | 60 min |
| run_tests | 9 min | 15 min | 0 | 30 min |
| publish_cohort | 4 min | 9 min | 1 | 20 min |

The critical path at p50 is 22 + 8 + 35 + 12 + 9 + 4 = 90 minutes. Starting at 02:00, the pipeline
finishes at 03:30, leaving 150 minutes of buffer before 06:00.

At p95, the path is 48 + 14 + 62 + 25 + 15 + 9 = 173 minutes, finishing at 04:53, leaving 67 minutes.
A single retry of the slowest task at p95 adds 62 minutes and breaches the deadline at 05:55 if the retry
delay is five minutes. That is the reason the transform task has a 120-minute timeout and an alert at
05:00, not only on failure.

### 6.2 Late-arriving data

The FHIR server delivers 2.1 million resources per day. Monitoring shows that 0.7% of resources arrive
more than 24 hours after their `meta.lastUpdated` value.

- Late resources per day: 2,100,000 × 0.007 = 14,700.
- Late resources per 30 days: 14,700 × 30 = 441,000.
- If the pipeline closes each partition at T+24 hours, those 441,000 resources are missed every month.

The pipeline reprocesses the trailing 7 days on every run. The additional cost is six extra partition
runs per night: 6 × 35 minutes of transform = 210 minutes of transform work, which is parallelized across
partitions with `max_active_runs=4`. With four concurrent partitions, the added wall-clock time is
210 / 4 = 52.5 minutes, which fits within the 150-minute p50 buffer but not within the 67-minute p95
buffer if the day's own run is also slow.

The alternative is to close partitions at T+7 days and publish with a `data_through` field. That reduces
nightly compute but means the dashboard shows a seven-day lag for late corrections. The team chooses the
trailing reprocess for the cohort, because the cohort is small, and accepts the p95 risk with an alert.

### 6.3 Backfill arithmetic

A logic change requires reprocessing two years of daily partitions: 730 partitions.

- Sequential at 35 minutes per transform plus 22 minutes extract plus 12 load = 69 minutes per
  partition: 730 × 69 = 50,370 minutes = 839.5 hours = 35.0 days.
- With `max_active_runs=8`: 839.5 / 8 = 104.9 hours = 4.4 days.
- With `max_active_runs=8` and extract parallelized separately, the critical resource is the source's rate
  limit, not the scheduler. Eight concurrent extracts occupy 8 × 22 = 176 worker-minutes per wave; whether
  that wave completes in 22 minutes depends on whether the source sustains eight simultaneous clients.

This is why a backfill plan must name the bottleneck. The scheduler's concurrency is often not the
constraint; the source's rate limit or the warehouse's queue is.

### 6.4 Daylight saving

The schedule is `0 2 * * *` in `Europe/London`. In 2024:

- 31 March: clocks move from 01:00 GMT to 02:00 BST. The local time 02:00 does not exist. A naive cron
  may skip the run or fire at 03:00.
- 27 October: clocks move from 02:00 BST to 01:00 GMT. The local time 02:00 occurs twice. A naive cron
  may fire twice.

A UTC schedule `0 2 * * *` has exactly 365 fires in 2024 and no ambiguity. If the clinical process
requires 02:00 local, use a timezone-aware timetable and test the two transition dates explicitly.

### 6.5 The incident

On 4 November, the cohort count drops by 6,100 patients (8.2% of 74,390). The pipeline succeeded. The
investigation finds that a backfill for October ran concurrently with the nightly job, both wrote the
same partitions, and the nightly job's `DELETE` and `INSERT` interleaved with the backfill's. The final
partition contains rows from both runs, and the cohort's deduplication selected the backfill's older
records.

The fix is not a retry. It is a lock: `max_active_runs=1` for the DAG, a partition-level advisory lock in
the load task, and a separate backfill DAG that shares the lock. The runbook adds a check for concurrent
runs before any manual trigger.

### 6.6 Task-level run ledger

The pipeline's own logs record the following for the 24 October partition:

| Task | Rows in | Rejected | Rows out | Duration |
|---|---:|---:|---:|---:|
| `extract_bulk` | 2,100,000 | 0 | 2,100,000 | 22 min |
| `validate_resources` | 2,100,000 | 13,600 | 2,086,400 | 8 min |
| `transform_measurements` | 2,086,400 | 0 | 2,086,400 | 35 min |
| `load_partition` | 2,086,400 | 0 | 2,086,400 | 12 min |
| `run_tests` | 2,086,400 | 0 | 2,086,400 | 9 min |
| `publish_cohort` | 7,412 persons | 0 | 7,412 | 4 min |

The rejected 13,600 resources are not errors in the pipeline; they are resources that fail profile
validation, such as an `Observation` with no subject or an `Encounter` with no period. They are stored in
a quarantine table with the validation message.

### 6.7 Reconciliation from source to load

| Stage | Count | Calculation |
|---|---:|---|
| Source reports | 2,100,000 | Server count for the day |
| Received | 2,100,000 | Manifest rows |
| Duplicates | 0 | Same resource ID and version |
| Profile-invalid | 13,600 | 0.65% of received |
| Loaded | 2,086,400 | 2,100,000 − 13,600 |

The loaded share is 2,086,400 / 2,100,000 = 99.35%. The reconciliation is published with the run; a
consumer who sees a count different from the source can see where the difference went.

### 6.8 Second run and idempotency check

The partition is loaded twice with the same input.

| Metric | Run 1 | Run 2 | Expected |
|---|---:|---:|---|
| Rows | 2,086,400 | 2,086,400 | Equal |
| Distinct `resource_id` | 2,086,400 | 2,086,400 | Equal |
| Partition digest | `7b3e...41c9` | `7b3e...41c9` | Equal |
| Duplicate rows | 0 | 0 | Equal |

A controlled fault replaces the `DELETE` with an append. The second run then produces 4,172,800 rows, an
increase of 100%, and the count check fails. The experiment demonstrates that the idempotency comes from
the delete-and-insert transaction, not from the scheduler's retry policy.

### 6.9 Late-arrival replay check

The trailing seven days are reprocessed on every run. The 24 October partition receives 102,900 corrected
resources from the 0.7% late-arrival rate over seven days:

- Late per day: 2,100,000 × 0.007 = 14,700.
- Late per seven days: 14,700 × 7 = 102,900.
- After replay: 2,086,400 + 102,900 = 2,189,300 rows.

A second replay with the same input produces 2,189,300 again, not 2,292,200. If the count grows on the
second replay, the load is appending rather than replacing.

### 6.10 Diagnostic checks

**Check A: freshness.** The newest `meta.lastUpdated` is 02:14; the check runs at 06:00. Age = 226
minutes. Warning threshold 180, error 360. Result: warn. A value above 360 means the source or the
extract is late, and the dashboard should show a stale marker.

**Check B: duplicate rate.** Expected 0.0%; observed 0.0%. Pass. A non-zero rate means the source
retransmitted or the load appended.

**Check C: empty partition.** If the extract returns 0 rows, the count check must fail even though the
task succeeds. Expected: fail with `empty_partition`. A pipeline that treats zero rows as success
publishes a day with no patients.

### 6.11 Decision table for a failed run

| Symptom | First check | Likely cause | Action |
|---|---|---|---|
| Task failed, no output | Task log and retry count | Transient source error | Retry; escalate after 3 |
| Task succeeded, zero rows | Extract manifest | Empty source or wrong interval | Fail the run; do not publish |
| Count doubled | Load method | Append instead of replace | Restore prior; fix load |
| Count 0.65% low | Validation report | Profile-invalid resources | Review quarantine; accept or fix |
| Freshness warn | Source `lastUpdated` | Source lag | Notify source owner |
| SLA missed, run green | Task durations | Slow query or queue | Profile; add capacity |

### 6.12 Reprocessing impact of a late replay

The late replay changes the cohort for 24 October. Lineage identifies three downstream outputs that must
be rebuilt, and the model does not need retraining because it is trained weekly on a snapshot.

| Output | Rows affected | Rebuild time |
|---|---:|---:|
| `gold.hf_cohort` | 7,412 | 12 min |
| `gold.hf_features` | 7,412 | 18 min |
| `report.quality_daily` | 1 row | 4 min |
| **Targeted total** | | **34 min** |
| Full pipeline rebuild | 2,086,400 resources | 90 min |

Targeted reprocessing costs 34 minutes against 90 minutes for a full rebuild, a saving of 56 minutes or
62.2%. The saving depends on lineage being complete; if the dashboard's dependency on the cohort is not
recorded, it is rebuilt late or not at all, and the consumer sees a mix of old and new numbers.

### 6.13 What a green run still does not prove

A successful run with all tasks green does not prove:

1. That every source resource was received. The source count and the manifest must reconcile.
2. That the rows are clinically valid. Profile validation catches structure, not plausibility.
3. That the partition is complete. Late data may still arrive after the run.
4. That downstream consumers read the new partition. Publish and cache invalidation are separate steps.

Each is a check with an expected value, not an assumption. The runbook lists them because a green
orchestrator status is the most misleading signal in the platform.

### 6.14 Superseded resource versions

A source corrects 14 observations and republishes them with the same `resource_id` and a higher
`meta.versionId`. The loader must keep the highest version, not the first or the last arrival.

| Scenario | Rows | Outcome |
|---|---:|---|
| Correct handling (max version) | 2,086,400 | 14 rows updated, count unchanged |
| Append both versions | 2,086,414 | 14 duplicate clinical facts |
| Keep first arrival | 2,086,400 | 14 stale values |

The count is identical in the correct and stale cases, which is why a version check is necessary: only a
comparison of `meta.versionId` or a value audit reveals that the correction was dropped. The check is
`COUNT(*)` grouped by `(resource_id)` with `COUNT(DISTINCT version)` greater than 1; expected 0 after the
merge.

A second-order failure appears when the corrected resource belongs to a partition that has already been
published. The correction must trigger a replay of that partition and a republication of the cohort, not
only a row update. Otherwise the warehouse is correct and the dashboard is stale, which is the most
confusing state for a consumer.

## 7. Code, query, or calculation

### 7.1 Airflow 3.3 DAG (not executable here)

The following targets Apache Airflow 3.3.x, using the Task SDK decorators. Airflow is not installed in
this environment, so the block is syntax-checked with `compile()` rather than executed; it runs unchanged
in an Airflow 3.3 environment with the referenced helper functions available.

```python
# dags/fhir_daily.py — Apache Airflow 3.3.x
from datetime import datetime, timedelta

from airflow.sdk import dag, task

DEFAULT_ARGS = {
    "owner": "data-platform",
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=30),
    "execution_timeout": timedelta(minutes=90),
}


@dag(
    dag_id="fhir_daily",
    schedule="0 2 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["fhir", "clinical"],
    doc_md=__doc__,
)
def fhir_daily():
    @task
    def extract(**context):
        # data_interval_start/end are timezone-aware UTC datetimes
        start = context["data_interval_start"]
        end = context["data_interval_end"]
        run_id = context["run_id"]
        path = extract_bulk(start=start, end=end, run_id=run_id)
        return path  # a URI, not the data

    @task(retries=1)
    def validate(path: str):
        return validate_resources(path)

    @task(retries=2, execution_timeout=timedelta(minutes=120))
    def transform(path: str, **context):
        partition = context["data_interval_start"].date().isoformat()
        return transform_measurements(path, partition)

    @task(retries=2)
    def load(path: str, **context):
        partition = context["data_interval_start"].date().isoformat()
        load_partition(path, partition)  # DELETE + INSERT in one transaction

    @task(retries=0)
    def test(partition: str):
        run_tests(partition)

    @task
    def publish(partition: str):
        publish_cohort(partition)

    raw = extract()
    checked = validate(raw)
    transformed = transform(checked)
    loaded = load(transformed)
    test(loaded)
    publish(loaded)


fhir_daily()
```

The essential properties: no top-level I/O, the interval comes from the scheduler context, the partition
is derived from the interval start, `max_active_runs=1` prevents concurrent writes, and XCom carries a
URI rather than a data frame.

### 7.2 Idempotent partition writer (SQLite, executable)

The script below simulates the load task. It writes one partition inside a transaction, and it is run
three times: twice with the same input and once after a simulated failure. Expected output is stated
after the code.

```python
import hashlib
import sqlite3

con = sqlite3.connect(":memory:")
con.execute("""
CREATE TABLE measurement (
    partition_date TEXT NOT NULL,
    person_id      INTEGER NOT NULL,
    value          REAL NOT NULL,
    run_id         TEXT NOT NULL,
    PRIMARY KEY (partition_date, person_id)
)
""")


def rows_for(partition: str, version: int) -> list[tuple[int, float]]:
    base = [(101, 2.4), (102, 1.9), (103, 3.1)]
    if version == 2:
        base = [(101, 2.4), (102, 2.0), (103, 3.1), (104, 1.7)]
    return base


def load_partition(partition: str, rows: list[tuple[int, float]], run_id: str,
                   fail_after: int | None = None) -> None:
    """Replace the partition atomically. A failure leaves the prior state intact."""
    try:
        con.execute("BEGIN")
        con.execute("DELETE FROM measurement WHERE partition_date = ?", (partition,))
        for i, (person_id, value) in enumerate(rows, start=1):
            if fail_after is not None and i > fail_after:
                raise RuntimeError("simulated write failure")
            con.execute(
                "INSERT INTO measurement VALUES (?,?,?,?)",
                (partition, person_id, value, run_id),
            )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise


def state(partition: str) -> tuple[int, str]:
    rows = con.execute(
        "SELECT person_id, value FROM measurement WHERE partition_date = ? "
        "ORDER BY person_id", (partition,)
    ).fetchall()
    digest = hashlib.sha256(
        "|".join(f"{p}:{v}" for p, v in rows).encode()
    ).hexdigest()[:10]
    return len(rows), digest


P = "2026-10-24"
load_partition(P, rows_for(P, 1), "run-1")
print("after_run_1", state(P))

load_partition(P, rows_for(P, 1), "run-2")
print("after_run_2", state(P))

try:
    load_partition(P, rows_for(P, 2), "run-3-failed", fail_after=2)
except RuntimeError as exc:
    print("failed_as_expected", str(exc))

print("after_failed_run", state(P))

load_partition(P, rows_for(P, 2), "run-4")
print("after_corrected_run", state(P))
```

Expected output:

```text
after_run_1 (3, '5739e3ec6e')
after_run_2 (3, '5739e3ec6e')
failed_as_expected simulated write failure
after_failed_run (3, '5739e3ec6e')
after_corrected_run (4, '553001c88b')
```

The first two runs produce the same row count and the same digest, which is the definition of idempotent
for this task. The failed run leaves the prior state untouched because the transaction rolled back. The
corrected run replaces the partition with four rows. The digest is computed over ordered
`person_id:value` pairs, so it does not depend on insertion order.

### 7.3 Backfill cost calculation (standard library)

```python
partitions = 730
minutes_per_partition = 69
max_active_runs = 8

sequential_hours = partitions * minutes_per_partition / 60
parallel_hours = sequential_hours / max_active_runs

print(f"sequential_hours {sequential_hours:.1f}")
print(f"parallel_hours {parallel_hours:.1f}")
print(f"calendar_days_at_8h_per_day {parallel_hours / 8:.1f}")
```

Expected output:

```text
sequential_hours 839.5
parallel_hours 104.9
calendar_days_at_8h_per_day 13.1
```

The calculation assumes perfect parallelism and no source throttling. In practice, the source rate limit
and the warehouse queue determine the duration, which is why Section 6.3 names the bottleneck rather than
trusting the arithmetic.

## 8. Data, standards, and regulatory context

### 8.1 Orchestrator versions and behavior

| Orchestrator | Current line in 2026 | Scheduling model | Idempotency support | Notes |
|---|---|---|---|---|
| Apache Airflow | 3.3.x (3.3.2, 17 September 2026) | Cron, timedelta, timetable, assets | Not automatic; task responsibility | Mature; large operator ecosystem |
| Dagster | 1.x | Asset-centric, partitions | Asset materializations are partition-scoped | Strong lineage model |
| Prefect | 3.x | Flows and deployments | Task-run retries; idempotency by key | Pythonic, dynamic |
| Argo Workflows | 3.x | Kubernetes-native DAGs | Pod retries; workflow templates | Container-first |
| dbt Cloud scheduler | Managed | Model DAG | Incremental models need keys | SQL-centric |

Airflow 3.0 was released on 22 April 2025 with a task execution API, a React UI, and DAG versioning.
The 3.3 line added a task and asset state store, Java and Go task SDKs, and expanded asset partitioning.
The version matters because the decorator import path (`airflow.sdk`), the context variable names
(`logical_date` versus `execution_date`), and the API for backfills differ from Airflow 2.

### 8.2 Scheduling standards

Cron is defined by POSIX and implemented with variations. The five-field form is
`minute hour day-of-month month day-of-week`; the day-of-month and day-of-week fields are ORed when both
are restricted, which surprises many users. Airflow supports cron plus presets such as `@daily`, and
`timedelta` schedules for fixed intervals.

ISO 8601 defines date, time, duration, and interval representations. A data interval should be written as
`2026-10-24T00:00:00Z/2026-10-25T00:00:00Z`, with the time zone explicit. A timestamp without an offset
is ambiguous for scheduling.

### 8.3 Lineage and audit

OpenLineage is an open specification for run, job, and dataset events. Its current release line in 2026
is 1.x (1.54.0 in October 2025), with a versioned facet model. Emitting OpenLineage events from the
orchestrator and the transformation layer gives a lineage graph that is independent of the vendor.

Under 21 CFR Part 11, records that support a regulated submission need audit trails that record who did
what and when, and changes must not obscure prior values. An orchestrator's run history is part of that
evidence: it records the run ID, the logical date, the operator, and the outcome. It does not record the
data values, so it must be paired with a data snapshot and a code version.

Under HIPAA, the Security Rule requires audit controls that record and examine activity in systems
containing electronic PHI. Orchestrator logs that include patient identifiers would themselves be a
disclosure risk; logs should contain run metadata, counts, and error codes, not record contents.

### 8.4 What orchestration standards do not specify

- When a partition is complete.
- How late data are handled.
- Whether a failed task should block downstream consumers.
- How to roll back a bad partition.
- Who owns an alert.

Those are design decisions. A DAG that is syntactically valid and has no owner is an operational liability.

## 9. Industry practice and case evidence

### 9.1 Airflow's evolution

Airflow began at Airbnb in 2014 and became an Apache top-level project in 2019. Its 3.0 release in April
2025 was the largest architectural change in its history: a task execution API that separates the
scheduler from task execution, a React-based UI, DAG versioning, and assets as first-class scheduling
dependencies. The 3.3 line in 2026 added a task and asset state store and language SDKs. The practical
lesson is that orchestrator APIs change across major versions, and a pipeline's DAG code is a dependency
that must be upgraded deliberately.

### 9.2 dbt's split of responsibilities

dbt manages the SQL model graph, tests, and documentation; an orchestrator manages when dbt runs and what
happens on failure. dbt's documentation recommends `dbt build` to run models and tests together, and
source freshness checks to detect stale inputs. The split avoids duplicating the model graph in the
scheduler, but it also means the orchestrator's DAG is coarse: one task may run dozens of models, and a
failure inside dbt requires reading dbt's logs rather than the orchestrator's task log.

### 9.3 OpenLineage adoption

OpenLineage's model of run, job, and dataset events has been adopted by Airflow, Spark, dbt, and other
tools. The value is a vendor-neutral lineage graph; the limitation is that instrumentation must be enabled
and maintained, and a partial graph can be more misleading than no graph because it looks complete.

### 9.4 What is inference

- That a nightly cadence is sufficient for a quality dashboard. It depends on the decision's latency, not
  on convention.
- That `max_active_runs=1` prevents all concurrency incidents. It prevents concurrent DAG runs, not
  concurrent writes from a separate backfill DAG or a manual notebook.
- That retries improve reliability. They improve recovery from transient failures and can amplify
  permanent ones.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Duplicate output after retry | Row counts exceed the source count | Append without a key | Inflated rates | Partition replace or upsert |
| Wrong interval | Data are one day stale | Wall-clock date used for filtering | Systematic lag | Use logical date and interval |
| DST skip or double run | Missing or duplicated day at transition | Local-time cron | Data gap or duplicate | UTC schedule or timezone timetable |
| Unbounded catchup | Hundreds of runs queued on deploy | `catchup=True` with old start date | Scheduler overload | Bounded start; explicit backfill |
| Retry storm | Source throttling after a failure | Fixed short delay, many retries | Outage amplification | Exponential backoff; retryable errors only |
| No timeout | Task running for hours | Hung network call | Blocked slots | Per-task timeout |
| Shared table race | Counts fluctuate between runs | Two runs writing one partition | Corruption | `max_active_runs=1`; partition lock |
| Sensor without timeout | DAG stuck in running | Waiting for a file that never arrives | Downstream never runs | Timeout and deferrable sensor |
| Top-level DB call | Scheduler slow; parse errors | Connection at import time | DAG file fails to load | Move I/O into tasks |
| XCom with a data frame | Scheduler database growth | Large payload | Slow UI; PHI in metadata DB | Pass URIs |
| Silent success | Task succeeds with zero rows | Empty extract treated as valid | Missing day unnoticed | Row-count and freshness checks |
| Alert with no runbook | On-call investigates from scratch | Alert only names the task | Longer outage | Runbook link and owner |

Two red flags deserve emphasis.

**The DAG has no owner.** Airflow's `owner` field is often set to a team name that no longer exists. An
alert routed to an unowned channel is not an alert.

**The backfill procedure is "run the DAG again."** A backfill that does not limit concurrency, verify
output, and record a before-and-after comparison is a data change without a control.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| The source can filter by event time | Replayability | API only supports "latest" | Inspect the API; otherwise snapshot |
| The partition key is stable | Idempotent writes | Source changes the date semantics | Compare partition counts to source dates |
| Task runtime is predictable | Timeouts and SLAs | A new data volume or a slow query | Track p50/p95 duration |
| The orchestrator clock is reliable | Interval correctness | Clock skew or DST bugs | Test transition dates |
| Late data are bounded | Completeness rule | A source outage delivers a week at once | Monitor lateness distribution |
| The warehouse can absorb a backfill | Throughput | Shared cluster saturation | Load test with production-like volume |
| Alerts reach an owner | Incident response | Rotated team, stale channel | Quarterly alert test |
| The DAG version is recorded | Audit and rollback | Scheduler does not expose it | Log code version and image digest |
| Downstream consumers tolerate reprocessing | Correction policy | A dashboard reads partitions mid-backfill | Publish `data_through`; use snapshots |

The largest limit is that orchestration does not create correctness. A perfectly scheduled pipeline that
computes the wrong cohort produces the wrong answer on time, every day, with a green status. The tests and
definitions in Topics 02 and 10 are what make the schedule meaningful.

A second limit is that the orchestrator is a single point of failure for the schedule, not for the data.
If the scheduler is down, the data remain in the source and can be reprocessed; if the warehouse is down,
no amount of scheduling helps. Architecture should separate those failure domains.

## 12. Exercises

### Exercise 1 — Logical date (recall, 20 minutes)

A daily DAG runs at 02:00 on 25 October and processes `[24 October, 25 October)`. Deliverable: the logical
date, the data interval, and the partition key, written in ISO 8601 with UTC.

Acceptance criteria: the logical date is 24 October, not 25 October; the interval is half-open; the
partition key matches the interval start.

### Exercise 2 — Idempotent load (application, 45 minutes)

Deliverable: a load function that replaces one partition in a transaction, plus a test that runs it twice
and asserts identical row counts and content hashes.

Acceptance criteria: a simulated failure leaves the prior partition intact; a corrected run replaces it;
the test does not depend on row order.

### Exercise 3 — DST schedule (application, 30 minutes)

Deliverable: a schedule for 02:00 local time in `Europe/London` that produces exactly one run per day,
and a test plan for 31 March and 27 October 2024.

Acceptance criteria: the schedule is timezone-aware; the test names the two transition dates; the
behavior on each is stated.

### Exercise 4 — Backfill plan (calculation, 40 minutes)

Given 730 partitions, 69 minutes per partition, and a source limit of 40 requests per second, deliverable:
a backfill plan with concurrency, expected duration, and the bottleneck.

Acceptance criteria: the arithmetic is shown; the bottleneck is named; the plan includes a validation
step and a rollback step.

### Exercise 5 — Retry policy (design, 40 minutes)

Deliverable: a table of tasks with retry count, delay, backoff, timeout, and retryable exception classes.

Acceptance criteria: transient and permanent failures are distinguished; no task has an infinite retry;
the policy names the alert that fires after exhaustion.

### Exercise 6 — Late data (design, 45 minutes)

Deliverable: a completeness rule for a FHIR ingestion pipeline with a 0.7% lateness rate, including a
`data_through` field and a consumer contract.

Acceptance criteria: the rule states the reprocessing window; the consumer knows how to interpret an
incomplete partition; the cost of reprocessing is estimated.

### Exercise 7 — DAG review (critique, 40 minutes)

Review the DAG in Section 7.1. Deliverable: a list of five improvements with rationale.

Acceptance criteria: at least one improvement concerns failure semantics; at least one concerns
observability; no improvement is purely stylistic.

### Exercise 8 — Incident diagnosis (critique, 60 minutes)

A cohort drops 8.2% after a backfill. Deliverable: a 500-word incident analysis with a timeline, a root
cause, a blast-radius estimate, and three preventive controls.

Acceptance criteria: distinguishes orchestration from data causes; quantifies the affected partitions;
names an enforceable control, not a reminder.

## 13. Self-check answers

**1. What is a DAG?**

A directed acyclic graph of tasks with declared dependencies. The acyclicity guarantees termination; the
edges define execution order and failure propagation. A DAG is not a pipeline; it is the schedule and
dependency structure for one.

**2. What is the logical date?**

The start of the data interval a run processes. For a daily run on 25 October processing the previous
day, the logical date is 24 October. It is the identifier that makes a run reproducible and
backfillable.

**3. Why does a retry duplicate data?**

Because a retry re-executes the whole task. If the task appends rows and fails after writing some of them,
the retry appends again. Idempotent writes (replace or upsert) are the fix; retry counts are not.

**4. How many runs does a UTC daily schedule produce in 2024?**

365. A UTC schedule has no DST transition, so every calendar day has exactly one 02:00 UTC instant. A
local-time schedule may skip or duplicate a run at the transition.

**5. What is the difference between catchup and backfill?**

Catchup is automatic scheduling of missed intervals when a DAG is unpaused or its start date moves.
Backfill is an explicit command for a date range. Catchup can create hundreds of runs unintentionally;
backfill is a controlled operation.

**6. Compute the backfill duration.**

730 partitions × 69 minutes = 50,370 minutes = 839.5 hours. At 8 concurrent runs, 839.5 / 8 = 104.9
hours, or 4.4 days of continuous operation. At 8 working hours per day, 13.1 calendar days.

**7. Why pass a URI through XCom instead of a data frame?**

Because XCom is stored in the orchestrator's metadata database, which is not designed for large payloads
and may not be in scope for PHI. A URI keeps the data in governed storage and the orchestrator small.

**8. What is the right trigger rule for a cleanup task?**

`all_done`, so it runs whether upstream tasks succeed or fail. A transformation task should use
`all_success`, because running it on incomplete input produces a plausible wrong result.

**9. Why set a timeout when there is already an SLA?**

The SLA detects lateness; the timeout frees the worker. Without a timeout, a hung task can hold a slot
indefinitely and block other work even after the SLA has fired.

**10. What does `max_active_runs=1` not prevent?**

Concurrent writes from another DAG, a manual backfill, or a notebook. It limits runs of the same DAG
only. A partition-level lock is needed for cross-DAG concurrency.

## 14. Interview questions

**1. "How do you schedule a nightly clinical pipeline?"**

Strong: derives the cadence from the decision deadline, uses UTC or a timezone-aware timetable, defines
the data interval, sets catchup and backfill policy, and makes the load idempotent. Weak: "a cron job at
2 a.m."

**2. "What happens on a daylight-saving transition?"**

Strong: explains the missing and duplicated local times, prefers UTC, and tests the two dates. Weak:
"the scheduler handles it," without knowing which scheduler or how.

**3. "How do you make a task idempotent?"**

Strong: partition replace or upsert on a deterministic key, inside a transaction, with a test that runs
twice. Weak: "we delete the table first," which is not atomic and loses data on failure.

**4. "A retry wrote duplicate rows. What do you do?"**

Strong: contain the affected partitions, quantify the duplication, repair with an idempotent reload, and
add a uniqueness test. Weak: "delete duplicates," without identifying which copy is correct.

**5. "How do you backfill two years of data?"**

Strong: estimates duration from partition count and runtime, names the bottleneck, limits concurrency,
validates before and after, and has a rollback. Weak: "trigger the DAG with an earlier start date."

**6. "How do you handle late-arriving data?"**

Strong: measures lateness, chooses a reprocessing window or watermark, marks partitions with
`data_through`, and documents the consumer contract. Weak: "we rerun yesterday."

**7. "What would you monitor?"**

Strong: duration distribution, queue time, SLA misses, row counts, freshness, and failure reasons, with
an owner per alert. Weak: "success and failure."

**8. "Why keep top-level code out of the DAG file?"**

Strong: the scheduler parses files frequently; top-level I/O slows parsing, can fail the file, and may
leak credentials. Weak: "it is a best practice," without a mechanism.

**9. "How do you test a DAG?"**

Strong: structure tests, interval tests, idempotency tests, and a local run with `dag.test()`. Weak:
"it imports without error."

**10. "When would you not use an orchestrator?"**

Strong: a single daily query with no dependencies can be a scheduled warehouse task; adding an
orchestrator adds operational overhead without benefit. Weak: "always use Airflow."

## 15. Key terms

- **Airflow** — Apache workflow orchestrator.
- **Asset** — data object produced or consumed by a DAG.
- **At-least-once** — delivery may duplicate.
- **Backfill** — explicit execution of past intervals.
- **Catchup** — automatic scheduling of missed intervals.
- **Concurrency** — maximum parallel tasks or runs.
- **DAG** — directed acyclic graph of tasks.
- **Data interval** — half-open span a run processes.
- **Dead letter** — storage for repeatedly failing records.
- **Deferrable operator** — releases its worker while waiting.
- **Exactly-once** — effect occurs once.
- **Exponential backoff** — increasing retry delay.
- **Idempotency key** — deterministic write identifier.
- **Late-arriving data** — events after their interval closed.
- **Logical date** — start of the data interval.
- **OpenLineage** — lineage event specification.
- **Partition** — independently writable data subset.
- **Pool** — concurrency limit for a task class.
- **Reprocessing window** — trailing interval re-run each execution.
- **Retry** — re-execution of a failed task.
- **Runbook** — operational procedure for an incident.
- **Sensor** — task that waits for a condition.
- **SLA** — service-level agreement on completion.
- **Task instance** — one execution of a task for one run.
- **Timeout** — maximum runtime before termination.
- **Trigger rule** — condition under which a task runs.
- **Watermark** — lateness threshold for completeness.
- **XCom** — small inter-task message.

## 16. Sources

All URLs accessed 25 October 2026.

1. Apache Airflow. "Backfill." https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/backfill.html
2. Apache Airflow. "Best Practices." https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html
3. Apache Airflow. "Dag Runs." https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/dag-run.html
4. Apache Airflow. "Dags." https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/dags.html
5. Apache Airflow. "Releases." https://github.com/apache/airflow/releases/
6. Astronomer. "DAG writing best practices in Apache Airflow."
   https://www.astronomer.io/docs/airflow/dag-best-practices
7. dbt Labs. "About dbt build." https://docs.getdbt.com/reference/commands/build
8. dbt Labs. "Add sources to your DAG." https://docs.getdbt.com/docs/build/sources
9. OpenLineage. "Specification." https://github.com/OpenLineage/OpenLineage/blob/main/spec/OpenLineage.md
10. OpenLineage. "Releases." https://github.com/OpenLineage/OpenLineage/releases
11. Dagster. "Asset-centric orchestration." https://docs.dagster.io/
12. Prefect. "Documentation." https://docs.prefect.io/
13. eCFR. "21 CFR Part 11." https://www.ecfr.gov/current/title-21/chapter-I/subchapter-A/part-11
14. eCFR. "45 CFR 164.312." https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-C/part-164/subpart-C/section-164.312
15. ISO. "ISO 8601 Date and time format." https://www.iso.org/iso-8601-date-and-time-format.html

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`03_python-engineering.md`](03_python-engineering.md)
- Next topic: [`05_fhir-ingestion.md`](05_fhir-ingestion.md)
- Related sibling: [`10_testing-and-observability.md`](10_testing-and-observability.md)
- Root survey: [`../../03_data_engineering_interoperability.md`](../../03_data_engineering_interoperability.md)
- Labs: [`../../labs/`](../../labs/) — Lab 3.6 simulates a broken pipeline and requires an incident report.
