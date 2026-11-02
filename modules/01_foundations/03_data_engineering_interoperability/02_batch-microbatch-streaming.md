> Module 03 — Data Engineering and Interoperability
> Topic 02 of 11
> Estimated study time: 3.5 hours
> Prerequisites: 03.01

# Batch, Micro-Batch, and Streaming

---

## 1. Why this topic exists

At 09:14 a cold-chain monitor raises an alert for a pallet of vaccine that has drifted to 9.4 °C.
The alert reaches the quality manager at 09:41, twenty-seven minutes after the excursion began.
The decision window — the time in which quarantining the pallet still prevents it being picked
for the 11:00 dispatch — closes at 09:35. The pallet ships. Six days later the customer reports a
spoiled batch, and the investigation finds that the technology was not the problem.

Trace the twenty-seven minutes. The sensor samples every 60 seconds. The gateway uploads within
20 seconds. The broker accepts the message in 2 seconds, the stream processor evaluates the rule
in 5, and the alerting service commits in 3. That is 90 seconds of system latency. The remaining
twenty-five and a half minutes are the on-call rota, the paging policy, and the fact that the alert
went to a shared mailbox. Meanwhile the design review had spent four months debating a 5-second stream against a
30-second stream.

The lesson is not that latency does not matter. It is that latency is routinely argued about in
the one stage where it is cheap, and left unexamined in the stages where it is expensive. This
topic is about choosing latency deliberately, and about the consequences of each choice for a
planning or execution decision.

Three claims organise what follows.

1. **Latency is not one number.** Processing latency, end-to-end data latency, data age at
   decision, and decision lag are four different quantities measured between four different pairs
   of timestamps. Only the last one determines whether a faster pipeline changes an outcome.
2. **Freshness is a budget, not an aspiration.** A decision tolerates a stated maximum data age.
   That tolerance is allocated across source commit, transport, processing, waiting, and human
   action. Spending the budget in a stage that is not the constraint buys nothing.
3. **Every rung on the latency ladder costs a different kind of complexity.** Moving from nightly
   batch to micro-batch costs engineering time. Moving from micro-batch to stream costs state
   management, checkpointing, on-call, and the ability to replay a month of history cheaply. The
   compute bill is the smallest part of the bill.

**Why this matters to a data scientist specifically.** Every model in this program assumes a data
cadence, usually without saying so. A demand model trained on nightly snapshots and served from a
five-minute stream has a training-serving skew in *time* rather than in features: it sees
intraday noise it was never fitted to model. A safety-stock policy derived for daily review, fed
by a five-minute feed, will react to the same signal many times and over-order. A backtest that
replays an event stream without reconstructing event time will let events that arrived late leak
information backwards. None of these failures appears in a feature-importance plot.

The previous topic established where data lands and what each layer may promise. This topic
decides how often it moves and how late it may be. Topic 03 treats the ingestion mechanisms that
deliver each cadence, and Topic 07 treats the scheduling and backfill machinery that keeps it
honest.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** processing latency, end-to-end data latency, data age at decision, and decision
   lag, and **compute** each from a timestamp ledger.
2. **Derive** a freshness requirement from a decision window, an action lag, and a safety margin,
   and **state** the cheapest processing mode that meets it.
3. **Draw** a latency ladder placing at least eight named supply chain decisions on the rungs,
   with the mode and cost shape at each rung.
4. **Decompose** end-to-end latency into five stages and **identify** which stage dominates for a
   given system, including the cases where the source commit or the human dominates.
5. **Explain** why a watermark is a heuristic lower bound rather than a completeness proof, and
   **design** a late-data policy with a bounded out-of-orderness setting, an allowed lateness, and
   a side output.
6. **Distinguish** exactly-once within a system boundary from effectively-once at a sink, and
   **choose** an idempotency key that makes at-least-once delivery safe.
7. **Diagnose** ordering and duplication defects from a described event sequence, and **state**
   the partition key and revision rule that would prevent each.
8. **Price** a freshness choice as cost per consumed update and cost per decision changed, rather
   than cost per event, and **justify** or **reject** a streaming proposal on that basis.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold one picture: four clocks, not one. An event happens, a system records it, a pipeline makes it
available, and a person or service acts. Latency is the distance between two of those points, and
which two you mean changes the answer by an order of magnitude.

```text
   event happens        record available        decision taken
        t0                    t1                     t2
        |---------------------|----------------------|
        |<-- processing lag -->|                      |
        |<-------- end-to-end data latency ---------->|
        |<------------- data age at decision -------->|

   Data age at t2 is t2 - t0 only when t0 is the OLDEST input the decision uses.
   A decision that reads nine DCs is as stale as its stalest DC.
```

The four quantities behave differently under improvement, which is why conflating them produces
confident nonsense in design reviews.

| Quantity | Measured between | Who controls it | Typical magnitude | What improving it buys |
|---|---|---|---|---|
| Processing latency | Record entering a processor and its result leaving | Data engineering | Seconds to minutes | Very little when it is not the dominant term |
| End-to-end data latency | Business event and record availability | Source owner plus data engineering | Minutes to hours | A fresher input at the moment of availability |
| Data age at decision | Decision time and oldest input event time | Whoever owns the decision clock | Hours to days | A decision made on more current reality |
| Decision lag | Data availability and action | Operations, on-call, authority | Minutes to hours | Whether the freshness is used at all |

Three consequences follow immediately, and they are the ones practitioners get wrong.

**Consequence 1 — a faster pipeline does not make a stale decision fresh.** If the warehouse
management system (WMS) commits its final transactions at 23:00 and the replenishment decision is
taken at 06:00, the data is seven hours old. Replace the 65-minute pipeline with a 5-minute stream
and the data is still six hours and twenty minutes old, a 14% improvement on a number that already
has hours of slack. The binding constraint is the source commit and the decision time, neither of
which is a data-platform artefact.

**Consequence 2 — freshness is a per-partition property.** A board that shows "last updated 4
minutes ago" is typically reporting the newest partition. The decision may depend on nine
distribution centres (DCs), eight of which are current and one of which has been silent since
22:00. The decision is as stale as the silent one, and the headline number conceals it. This is
why freshness is reported per source and per partition, as a percentile over records, not as a
single timestamp in a header.

**Consequence 3 — decision lag sets a ceiling on useful freshness.** If the on-call manager takes
25 minutes to acknowledge an alert, a 5-second stream and a 60-second stream produce the same
outcome. Buying sub-minute latency without buying sub-minute authority is the most common way to
spend a streaming budget and change nothing.

### 3.1 The latency ladder

The ladder is a planning instrument, not a product taxonomy. Each rung states the freshness a
decision needs at the moment it is taken, the mode that can supply it, and the decision that sits
there. Redraw it from memory with your own decisions.

```text
   FRESHNESS NEEDED AT DECISION        DECISION                   MODE

       1 min |  live order promise      quote a delivery date      on demand
             |
      10 min |  cold-chain quarantine    quarantine a pallet       stream
             |  dock re-slot             move a trailer             event push
             |
       1 h   |  carrier exception        divert a delayed load     stream
             |
       4 h   |  intraday reallocation    move stock between DCs     micro-batch
             |
      12 h   |  DC replenishment         place tomorrow's order     batch, nightly
             |
       1 wk  |  S&OP supply review       commit supply              batch, weekly
             |
       1 mo  |  network design           open or close a DC         batch, quarterly
             v
```

Read the ladder with three rules.

- **The rungs are separated by roughly an order of magnitude, and so are the costs.** A step down
  the ladder is not a tuning change; it is a different operating model with different failure
  modes and a different on-call rota.
- **Execution decisions cluster on the upper rungs, planning decisions on the lower ones.** On
  this ladder, four of eight are execution and four are planning; the planning rungs take their
  cadence from a meeting calendar, not from a technology.
- **A decision can be on a rung it does not need.** Placing a weekly supply review on a streaming
  rung does not make the review faster; it makes the pipeline more expensive and the review
  unchanged.

### 3.2 The five-stage ledger

Every latency claim decomposes into five stages. Ask for all five, in minutes, at p95.

| Stage | What it measures | Who owns it | Typical p95 | Why it is missed |
|---|---|---|---|---|
| Source commit | Event occurrence to the source system exposing it | Operations and the application owner | Minutes to 8 hours | It is inside a system the data team does not control |
| Capture and transport | Exposure to arrival at the platform | Integration engineering | Seconds to 15 minutes | Vendors quote this one because it is small |
| Processing | Arrival to a queryable, conformed record | Data engineering | Seconds to 2 hours | It is the only stage most teams measure |
| Publish and serve | Conformed record to the consumer's read | Analytics engineering | Seconds to 30 minutes | Cache invalidation is invisible in job metrics |
| Decide and act | Availability to an action | The decision owner | Minutes to hours | It is not a data metric at all |

Two entries in that table deserve emphasis. Source commit is frequently the largest term and is
almost never in the data team's remit: a WMS that confirms picks at end of shift contributes eight
hours before a single byte moves. Decide and act is frequently the second largest and is almost
never measured, because it lives in a rota rather than a pipeline.

> **Note:** The five-stage ledger is the same decomposition used in Module 02, Topic 06, which
> calls the stages application, network, pipeline, and decision lag. The naming differs because the
> purpose differs: that topic asks which transport to choose, and this one asks which stage to buy
> down. Use one vocabulary within one document, and state which you are using.

### 3.3 Why freshness is a budget

A budget has a total, an allocation, and a breach condition. For a decision, the total is the
decision window minus the action lag minus a safety margin:

`required p95 data age <= decision window - action lag - safety margin`

The decision window is the interval after the triggering event during which action still changes
the outcome. The action lag is how long the actor takes once informed. The margin absorbs the
difference between p95 and the tail you actually experience. If the result is negative, no data
system can help; the process, not the pipeline, is the constraint.

| Decision | Window | Action lag | Margin | Required p95 data age | Cheapest mode that meets it |
|---|---:|---:|---:|---:|---|
| Cold-chain quarantine | 15 min | 6 min | 2 min | 7 min | Stream |
| Live order promise | 120 s | 5 s | 15 s | 100 s | On-demand query or stream |
| Dock re-slot | 25 min | 4 min | 3 min | 18 min | Event push or 5-minute micro-batch |
| Intraday reallocation | 5 h | 40 min | 20 min | 4 h | 15-minute micro-batch |
| DC replenishment | 16 h | 0 (system) | 90 min | 14.5 h | Nightly batch |
| Weekly supply review | 5 d | 1 d | 4 h | 3.8 d | Weekly batch |

The table is the whole argument of this topic in six rows. The cold chain needs a stream because
the arithmetic leaves seven minutes. Replenishment needs a nightly batch because the arithmetic
leaves fourteen and a half hours, and a stream would consume budget it does not need. The mode
follows the number, not the fashion.

### 3.4 The shape of the cost curve

Cost rises roughly linearly with refresh frequency for a fixed-cadence design. Value does not. A
decision has a deadline; freshness that arrives before the deadline has value, and freshness that
arrives after it has none. The result is a step, not a slope.

```text
   value of
   reducing
   latency
      ^
      |                                    +--------  beyond this point,
      |                                   /           faster changes nothing
      |                                  /
      |                        +--------+
      |                       /
      |             +--------+
      |     +------+
      +-----+----------------------------------------------> freshness
          deadline for the decision        (log scale)
```

A team that computes only the cost curve will always conclude that batch is efficient. A team that
computes only the value curve will always conclude that streaming is transformative. The design
question is where the deadline sits relative to the current rung, and that is a business fact, not
a technical one.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Latency | Elapsed time between two defined points in a process | Used without naming the two points | Minutes or seconds |
| Processing latency | Time from a record entering a processor to that processor emitting a result | Reported as end-to-end latency | Seconds to minutes |
| End-to-end data latency | Time from a business event to the record being readable by its consumer | Measured only to the landing zone | Minutes to hours |
| Data age at decision | Decision time minus the event time of the oldest input the decision depends on | Confused with pipeline run time | Hours |
| Decision lag | Time from data availability to a human or system acting on it | Omitted from latency budgets | Minutes to hours |
| Freshness | How up to date a dataset is at a stated time: the age of the newest event it reflects | Reported as job completion time | Minutes, as a percentile |
| Freshness budget | The allocation of tolerable data age across stages for one decision | Treated as one service level for all data | Minutes per stage |
| Decision window | The interval after a triggering event in which action still changes the outcome | Assumed infinite | Minutes to days |
| Batch processing | Processing a bounded set of records at a scheduled time | Used to mean "slow" or "legacy" | Runs per day |
| Micro-batch | Processing small bounded sets at short fixed intervals, typically seconds to minutes | Assumed to be streaming | Interval in seconds |
| Stream processing | Continuous processing of unbounded records as they arrive | Assumed to be complete or globally ordered | Events per second |
| Event-driven push | Producer-initiated delivery of a message to a consumer endpoint | Treated as guaranteed delivery | Messages per second |
| Event time | The timestamp at which the business event occurred | Taken from the source's commit clock | ISO 8601 timestamp with offset |
| Processing time | The wall-clock time at which a system handles a record | Used for windows that must be reproducible | Timestamp |
| Watermark | A heuristic assertion that no records with event time earlier than `t` will arrive | Treated as a completeness proof | Event-time timestamp |
| Bounded out-of-orderness | The lateness bound used to generate a watermark | Set to zero, which drops late data | Seconds or minutes |
| Allowed lateness | The grace period after the watermark in which late records still update results | Confused with retention | Seconds or minutes |
| Late event | An event whose event time precedes the current watermark | Dropped without being counted | Lateness in seconds |
| Side output | A secondary stream carrying records diverted from the main path, such as too-late events | Deleted after inspection | Stream name plus reason code |
| Window | A bounded slice of event time or processing time over which an aggregate is computed | Assumed to align with calendar periods | Tumbling, sliding, session |
| Trigger | The condition that causes a window result to be emitted | Assumed to fire exactly once | Trigger policy |
| Accumulation mode | How successive results for one window relate: discarding, accumulating, or accumulating and retracting | Left unspecified | Mode name |
| At-most-once | Delivery that may lose messages but never duplicates | Chosen accidentally by auto-commit | Delivery semantic |
| At-least-once | Delivery that never loses messages but may duplicate | Assumed to be exactly-once | Delivery semantic |
| Exactly-once | Processing effect applied once, normally within one system boundary | Claimed end-to-end across systems | Delivery semantic |
| Effectively-once | At-least-once delivery plus an idempotent write keyed on the business event | Used as a synonym for exactly-once | Delivery semantic |
| Idempotent write | A write that leaves the same state whether applied once or many times | Assumed because the target has a primary key | Boolean property |
| Deduplication key | The field combination identifying one business event | Set to arrival time or offset | Composite key |
| Upsert | Insert or update on a declared key | Applied without a monotonic guard, so an older version overwrites a newer one | Merge statement |
| Ordering guarantee | The scope within which message order is preserved, usually one partition | Assumed global | Partition key |
| Partition key | The field determining which partition, and therefore which ordering scope, a record belongs to | Chosen for load balance alone | Field name |
| Offset | The position of a record within a partition | Used as a business identifier | Integer per partition |
| Log retention | How long a stream retains records for replay | Confused with backup | Hours or days |
| Change data capture (CDC) | Reading a database's change log to emit row-level changes | Confused with a polling extract | Log sequence number |
| Log-based CDC | CDC that reads the write-ahead or binary log rather than polling tables | Assumed to capture every change including schema changes | Log position |
| Incremental snapshot | A chunked, resumable snapshot run alongside log capture | Assumed to be point-in-time consistent | Chunk boundary plus watermark |
| Lambda architecture | Parallel batch and stream paths over one input, merged at query time | Adopted without costing the duplicate logic | Architecture pattern |
| Kappa architecture | A single stream path, reprocessed by replaying the retained log | Assumed to eliminate the batch layer's problems | Architecture pattern |
| Replay | Reprocessing retained input to reproduce or correct an output | Used to mean "rerun tonight's job" | Offset or date range |
| Freshness SLO | A stated percentile bound on data age, with a measurement window and a breach action | Written as a target with no measurement | Percentile plus window |

Five of these deserve expansion, because the distinctions are routinely lost.

**Processing latency, end-to-end latency, and data age at decision.** Processing latency is a
property of a job. End-to-end latency is a property of a chain. Data age at decision is a property
of a decision. A pipeline can improve the first, hold the second roughly constant, and leave the
third unchanged, because the decision is taken at a fixed hour regardless of when data arrives.
Report all three, and state which one the service level covers.

**Watermark versus completeness.** A watermark says "I believe nothing earlier than `t` will
arrive." It is a heuristic, and the literature is explicit that it can be too fast, letting late
data slip behind it, or too slow, holding every downstream result hostage to one straggler. It is
a latency-control device, not a correctness guarantee. Completeness is established by a
reconciliation against a manifest, a count, or a source-side total, and by nothing else.

**Exactly-once versus effectively-once.** Exactly-once delivery over a network is not achievable;
what is achievable is an exactly-once *effect* at a sink that can recognise a repeated write. A
broker can offer transactional produce and offset commit inside its own boundary. The moment the
sink is a different system, the guarantee becomes at-least-once plus whatever the sink does about
it. Say "effectively-once" when you mean the second thing, and name the key.

**Freshness versus refresh frequency.** Refresh frequency is how often a job runs. Freshness is
how old the data is when someone reads it. A job that runs every five minutes over a source that
commits once a day produces five-minute job intervals and twenty-four-hour freshness. The two
numbers are related only through the source.

**Batch versus micro-batch.** Micro-batch is not "small streaming." It is a sequence of small
batch jobs, each with a bounded input, a checkpoint, and often a transaction. It inherits batch's
recovery model and batch's latency floor, which is the scheduling and commit overhead per batch.
That is why it sits between the two on the ladder and why its failure mode is a batch that
overruns its interval rather than a watermark that stalls.

## 5. The mechanics, step by step

The procedure has eight steps. The order matters: each step narrows the next. The decision window
fixes the budget, the budget fixes the stage allocation, the allocation rules out modes, and the
mode determines which correctness machinery you must operate.

### 5.1 Step 1 — Fix the decision window and derive the requirement

Write the decision as an action with an actor, a clock, and a deadline. The deadline is not the
meeting time; it is the last moment at which the action still changes the outcome.

For a cold-chain quarantine, the action is "hold this pallet." The deadline is the moment the
pallet is picked for dispatch, not the moment the shift ends. For a replenishment order, the
deadline is the carrier cut-off, not the planning meeting. Getting this wrong inflates the budget
by hours and produces a streaming business case that dissolves on contact with the first
incident review.

Apply the formula from section 3.3, then round *down* to the next rung. If the requirement is 7
minutes, a stream that achieves 5 minutes at p95 is correct and a 5-minute micro-batch that
achieves 6.5 minutes at p95 is not, because the p95 is the number the decision experiences on its
worst day.

> **Note:** Use p95, not the mean, and not the maximum. The mean hides the days that matter. The
> maximum is dominated by one incident and cannot be designed against. A decision that fails on
> one day in twenty is a decision with a known, priced failure rate; a decision designed to the
> mean fails on one day in two.

### 5.2 Step 2 — Build the five-stage ledger before choosing a mode

Instrument the chain you already have. Do not estimate; measure, because the estimates are
usually wrong in the same direction, which is to over-attribute latency to the stage the data
team owns.

| Stage | Instrument | Timestamp pair | What the number tells you |
|---|---|---|---|
| Source commit | Source audit table or change-log timestamp | Business event to row commit | Whether the source is the constraint |
| Capture and transport | Ingestion metadata | Commit to `_ingest_ts` | Whether transport is material |
| Processing | Job or operator metrics | `_ingest_ts` to mart write | What engineering can improve |
| Publish and serve | Cache and semantic-layer logs | Mart write to consumer read | Whether serving adds a hidden hour |
| Decide and act | Ticket, page, or action log | Consumer read to action | Whether freshness is used at all |

A worked reading: if source commit contributes 20 minutes, transport 2, processing 65, publish 10,
and action 265, the total is 362 minutes. Processing and publishing together are 21% of it; the
wait for the decision is 73%. Halving processing latency saves about 9% of data age. Moving the
decision two hours earlier saves 33%. One of those is an engineering project and one is a meeting.

### 5.3 Step 3 — Separate event time from processing time

Event time is when the thing happened. Processing time is when the system saw it. Every window,
join, and state machine must declare which one it uses, because the two produce different answers
under replay.

| Choice | Deterministic on replay | Latency cost | Correct for | Failure when misapplied |
|---|---|---|---|---|
| Event time | Yes | Waits for out-of-order data | Financial periods, service levels, anything audited | Windows never fire when a source is idle |
| Processing time | No | Lowest | Approximate monitoring, cache warming | Backtests cannot be reproduced; restatements differ |
| Ingestion time | Yes, given the same arrival order | Low | Raw landing and arrival-cohort analysis | Treated as business time, so late events land in the wrong period |

The supply chain consequence is concrete. A carrier sends a "delivered" event with an event time
of 14:05 that arrives at 19:40. A processing-time window assigns it to the 19:00–20:00 bucket and
reports the delivery as four hours late. An event-time window assigns it to 14:00–15:00 and reports
it correctly, at the cost of waiting until the watermark passes 15:00 before the number is final.
Choose which error you can live with, and write it down.

### 5.4 Step 4 — Choose the mode against the requirement, not the fashion

Four modes, four contracts. The table states what each actually guarantees.

| Mode | Input | Latency floor | State and recovery | Exactly-once mechanism | Operational burden | Correct use | Characteristic failure |
|---|---|---|---|---|---|---|---|
| Batch | A bounded partition or extract | Schedule-bound: 15 min to 24 h | Stateless per run; rerun replaces a partition | Partition replace or merge on key | Low: scheduler plus idempotent SQL | History, reconciliation, planning cycles | A late source makes the whole run late |
| Micro-batch | Unbounded, consumed in small bounded batches | About 100 ms engine floor; 1–15 min practical | Per-batch checkpoint; state store | Per-batch transaction; sink must still be idempotent | Medium: checkpoint tuning, backpressure | Near-real-time aggregates on a 1–15 minute SLO | A batch overruns its interval and latency degrades silently |
| Stream | Unbounded, record at a time | Sub-second to seconds | Long-lived state, checkpoints, savepoints | Checkpoint plus transactional or idempotent sink | High: on-call, state size, rescaling | Sub-minute decisions with measured value | Watermark stall, unbounded state, skewed keys |
| Event push | Unbounded, producer-initiated | One network round trip | None at the broker; consumer must persist | Consumer-side idempotency only | Low to build, high to make reliable | Low-volume, high-urgency notifications | Lost during consumer downtime, with no replay |

Three selection rules follow.

- **Choose the slowest mode that meets the p95 requirement.** Slower modes are cheaper to build,
  cheaper to operate, and easier to reconcile. There is no prize for a lower rung than the
  decision needs.
- **Do not choose micro-batch to avoid a stateful service.** Micro-batch still has state and
  checkpoints; it simply hides them in a batch abstraction. The operational difference between
  micro-batch and stream is smaller than the difference between either and batch.
- **Do not choose stream for a decision with a human action lag of hours.** Fix the rota first.
  A stream that pages a mailbox is a batch with a larger bill.

### 5.5 Step 5 — Set the watermark and the late-data policy

A watermark advances event time. When it passes the end of a window, the window fires. The
setting that matters is how far behind real time the watermark is allowed to run, and what
happens to records that arrive after it has passed.

| Parameter | Controls | How to set it | If set wrong |
|---|---|---|---|
| Bounded out-of-orderness | How long windows wait for stragglers | Observed p99 lateness per source, reviewed monthly | Too small: late data is dropped or restated; too large: unnecessary latency |
| Allowed lateness | Grace period after the watermark | Business tolerance for restating a published number | Zero: corrections are lost; large: state grows without bound |
| Idleness timeout | Whether a silent partition holds the watermark | Slightly above the source's normal inter-event gap | Unset: one silent DC stalls every window globally |
| Side output | Where too-late records go | Always configured, with a count and an alarm | Unset: loss is invisible |
| Emit interval | How often the watermark advances | 1–5 seconds for supply chain; engine defaults are often 200 ms | Too slow: added latency; too fast: needless overhead |
| State time-to-live | How long per-key state is retained | Window size plus allowed lateness plus a margin | Too short: wrong aggregates; too long: state growth and recovery time |

The idleness row is the one that produces the most spectacular incidents. In a parallel operator
with several inputs, the current event time is the minimum across inputs. One partition that
stops sending holds the watermark for all of them. Nine DCs feed a stream; DC-7's gateway fails at
22:00; the global watermark freezes at 21:59; no window fires for any DC for the next seven hours.
The job is healthy, the CPU is idle, and no alert triggers, because nothing failed. Detection
requires a watermark-lag metric — current event time against wall clock — not a job-status check.

> **Note:** A watermark is not a promise from the producer. It is an assumption the consumer makes
> about the producer. If the producer's lateness distribution changes — a new carrier, a gateway
> firmware update, a public holiday — the assumption silently becomes false. Re-estimate the bound
> from data, and alarm when observed lateness exceeds it.

### 5.6 Step 6 — Choose the delivery semantic and the idempotency key

Delivery semantics describe what happens between a producer and a broker, or a broker and a
consumer. They do not describe what happens at your sink. That is your design.

| Guarantee | Broker promise | Consumer obligation | Where it fails | Supply chain example |
|---|---|---|---|---|
| At-most-once | May lose, never duplicates | Commit the offset before processing | Crash after commit, before the write | A dock event never reaches the yard system |
| At-least-once | Never loses, may duplicate | Commit the offset after processing | Crash after the write, before the commit | A duplicate receipt inflates on-hand |
| Exactly-once in-system | Transactional produce and offset commit | Read committed isolation; transactional sink | Any sink outside the transaction | Kafka-to-Kafka only |
| Effectively-once | Nothing on its own | Idempotent write on a business key | The key is not unique per event | Duplicate advance ship notices converge to one row |

The practical design is always the last row: accept at-least-once delivery, and make the write
idempotent on a key that identifies the *business event*, not the transport. `(shipment_id,
event_type, event_ts)` is such a key. The partition offset is not, because a replay produces the
same event at a different offset. The arrival timestamp is not, because a redelivery produces a
different one.

Two refinements make the key safe in practice. First, add a monotonic guard so that an older
version of an event cannot overwrite a newer one that arrived first. Second, decide explicitly
whether a later delivery of the same key with different content is a correction to be applied or
a duplicate to be ignored. In supply chains it is usually a correction: a carrier re transmits an
estimated arrival with a revised time. Overwrite only if the incoming record is newer by a
declared version field, and keep the previous value if the domain requires an audit trail.

### 5.7 Step 7 — Make ordering explicit

Ordering is guaranteed within a partition, not across a stream. The partition key therefore
decides which events can be reordered relative to each other, and it is a business decision
disguised as a configuration value.

| Symptom | Mechanism | Design response |
|---|---|---|
| "Delivered" arrives before "out for delivery" | Out-of-order arrival across partitions or producers | Partition by `shipment_id`; make the state machine accept a transition on event time, not arrival order |
| Two receipts for one pallet | At-least-once redelivery | Deduplicate on the business key before applying |
| A backdated correction rewrites a closed period | Late event after the window fired | Apply as a revision with a reason code; never silently overwrite a published number |
| Timestamps disagree by minutes across sites | Unsynchronised clocks | Synchronise with NTP or PTP; record the source clock offset in the raw record |
| One key receives all traffic | A poor partition key, such as a constant or a low-cardinality status | Partition by the entity whose order matters, and accept the skew |

Partitioning by shipment identifier gives per-shipment order, which is what a state machine needs.
Partitioning by warehouse gives per-warehouse order, which is usually wrong for shipment state and
right for warehouse aggregates. If both are needed, build two streams; do not compromise on one
key and hope.

### 5.8 Step 8 — Price the freshness and set the service level

Three cost metrics are useful, and they answer different questions.

| Metric | Formula | What it answers | How it misleads |
|---|---|---|---|
| Cost per event | Monthly cost divided by events processed | Infrastructure efficiency | Events are not decisions; most are never read |
| Cost per refresh | Monthly cost divided by dataset refreshes | Platform efficiency | Rewards refreshing data nobody consumes |
| Cost per consumed update | Monthly cost divided by updates actually read by a consumer | Whether the cadence matches demand | Still ignores whether the decision changed |
| Cost per decision changed | Monthly cost divided by decisions whose outcome differed | Whether the latency earns its keep | Requires a counterfactual and is hard to measure |

The service level is written at the decision, not at the job. "p95 data age at 06:00 is at most 8
hours, measured per DC, with a breach action of publishing the previous day's plan flagged as
stale" is a service level. "Data refreshed nightly by 04:30" is a schedule.

## 6. Worked example

Meridian Health Distribution supplies medical consumables from nine DCs. It handles 4,800 active
items, 1,400 temperature-controlled shipments a day, 3,200 order promises a day, and about 18,000
dock events a month. All figures are illustrative and internally consistent.

### 6.1 Four decisions, four requirements

| Decision | Window | Action lag | Margin | Required p95 | Mode in use | Verdict |
|---|---:|---:|---:|---:|---|---|
| Cold-chain quarantine | 15 min | 6 min | 2 min | 7 min | Stream | Meets it at 5.5 min |
| Order promise | 120 s | 5 s | 15 s | 100 s | 5-minute micro-batch | Fails |
| Dock re-slot | 25 min | 4 min | 3 min | 18 min | Event push | Meets it at 4 s |
| DC replenishment | 16 h | 0 | 90 min | 14.5 h | Nightly batch | Meets it with 5.4 h to spare |

The order-promise row is the expensive mistake. A customer on the telephone waits about 120
seconds before abandoning a quote, so the system has 100 seconds to answer. A five-minute
micro-batch refreshes inventory every 300 seconds, which means the quote is based on data between
0 and 300 seconds old, and the worst case exceeds the budget by a factor of three. The team had
built the micro-batch because "order promising is real time," without computing the window.

### 6.2 The cold-chain ledger

| Stage | p95 | Owner | Note |
|---|---:|---|---|
| Sensor sample interval | 60 s | Facilities | Fixed by the device; the largest single term |
| Gateway upload | 20 s | Facilities | Cellular, retries on loss |
| Broker ingress | 2 s | Platform | Not material |
| Stream rule evaluation | 5 s | Data engineering | Not material |
| Alert commit | 3 s | Data engineering | Not material |
| **System action** | **90 s** | | |
| On-call acknowledgement | 240 s | Operations | The largest controllable term |
| **Human action** | **330 s (5.5 min)** | | Budget is 420 s; slack 90 s (21%) |

Now replace the stream with the 5-minute micro-batch the team already operated for order
promising. The pipeline term becomes 300 seconds at worst case, so system action is 388 seconds
and human action is 628 seconds, which is 208 seconds over a 420-second budget, or 49% over.
Replace both with the nightly batch and the term becomes hours. The arithmetic, not the
architecture diagram, decides the mode.

The cost side: the streaming platform costs about €14,700 a month including compute, storage, and
0.4 of a full-time equivalent at a loaded €9,000 a month. It produces 240 alerts a month, of which
62% are true, so 149 true alerts and 91 false ones. Each false alert costs about 20 minutes of a
technician's time at €55 an hour, or €18.33, which is €1,668 a month of avoidable work. The cost
per true alert is €98.66. A true alert prevents a mean of 40 units of spoilage at €85 a unit, or
€3,400, so the ratio of prevented loss to cost is about 34 to 1. That is a defensible case, and it
rests on the false-positive rate and the prevented-loss estimate, both of which should be stated
rather than assumed.

### 6.3 The replenishment freshness arithmetic

Meridian's nightly replenishment is often proposed for streaming, on the grounds that planners
"need current inventory." The timestamps tell a different story.

| Timestamp | Time | Interval | Share of data age |
|---|---|---|---:|
| Last warehouse transaction of the day | 22:40 | | |
| WMS batch commit | 23:00 | 20 min | 4.5% |
| Extract scheduled start | 00:30 | 90 min | 20.5% |
| Mart available | 01:35 | 65 min | 14.8% |
| Replenishment decision | 06:00 | 265 min | 60.2% |
| **Data age at decision** | | **440 min (7 h 20 min)** | **100%** |

The pipeline is 14.8% of the data age. The wait between availability and the decision is 60.2%.
Making the pipeline instantaneous would leave data age at 375 minutes, a 14.8% improvement, and
would not change a single order, because the requirement is 870 minutes and the current state
already has 430 minutes of slack.

The two interventions that would matter are moving the extract earlier, which saves up to 90
minutes, and moving the decision earlier, which saves up to 265. Neither is a streaming project.
This is the most common shape of a misplaced latency investment: engineering optimises the stage
it owns, while the decision clock sits untouched.

> **Note:** There is a legitimate streaming case at Meridian, but it is not the 06:00
> replenishment. It is exception management during the day: a DC that loses a receipt at 11:00
> should not wait until the next morning to see its position change. That is a different decision
> with a different window, and it should be argued on its own numbers.

### 6.4 The micro-batch cost trap at order promising

The order-promise service refreshes 43,200 item-location rows every five minutes.

| Quantity | Value | Arithmetic |
|---|---:|---|
| Item-locations | 43,200 | 4,800 items x 9 DCs |
| Refreshes per day | 288 | 24 x 60 / 5 |
| Row refreshes per day | 12,441,600 | 43,200 x 288 |
| Row refreshes per month | 373,248,000 | x 30 |
| Promises consumed per month | 96,000 | 3,200 x 30 |
| Platform cost per month | €8,200 | €1,500 fixed plus €6,700 variable |
| Cost per row refresh | €0.000022 | 8,200 / 373,248,000 |
| Cost per consumed refresh | €0.085 | 8,200 / 96,000 |
| Ratio | 3,880x | 0.085 / 0.000022 |

Only 0.026% of the refreshes are ever read. The platform is paying to keep 43,200 rows current
for a consumer that touches 3,200 of them a day.

The redesign is to refresh an item-location when it changes, not on a timer. About 24,000
distinct item-locations change on a typical day through receipts, picks, and adjustments, which
is 720,000 changes a month. The new variable cost is 720,000 x (€6,700 / 373,248,000) = €12.93.
With the €1,500 fixed floor, the new monthly cost is €1,512.93, a saving of €6,687 a month or about
€80,200 a year, before any engineering cost.
The fixed floor is now 99.1% of the bill, which is the correct signal that the platform is
right-sized: further volume reduction buys almost nothing.

### 6.5 The watermark stall

On 14 March, DC-7's telemetry gateway failed at 22:00 and was restored at 05:40. The exception
stream consumed nine DC feeds, and its global watermark was the minimum across them, so it froze
at 21:59. The thirty-minute "shipments at risk" windows stopped firing. No error was raised: the
job was running, checkpoints were succeeding, and CPU was idle.

| What happened | Mechanism | Consequence |
|---|---|---|
| No exception alerts for 7 h 40 min | Global watermark held by one idle input | All nine DCs unmonitored, not only DC-7 |
| Job reported healthy | Liveness was monitored; progress was not | Detection depended on a planner noticing an empty board at 06:10 |
| 14 shipments dispatched unchecked | Windows never emitted | 3 spoilage claims totalling €10,200, plus one customer audit |
| Backfill was possible | Seven days of log retention | Recovered the next day, with a 24-hour delay in the audit trail |

The fix has three parts. Enable idleness detection on each source so a silent partition stops
holding the watermark after a configured gap. Add a watermark-lag alert: current event time
against wall clock, alarm above 15 minutes, which is a progress metric rather than a failure
metric. And route too-late records to a side output with a count, so that a late correction is
visible rather than dropped.

## 7. Code, query, or calculation

One query, because the concept it demonstrates — idempotent convergence — is otherwise easy to
describe and hard to believe. The statement merges a batch of raw events into a conformed table on
a business key, and updates only when the incoming copy is newer. Run it twice and the second run
changes nothing. Replay a month and the result matches the month's end state.

```sql
-- Effectively-once: replays and duplicate deliveries converge on one state.
merge into conformed.shipment_event as t
using (
    select
        shipment_id,
        event_type,
        event_ts,
        arg_max(payload, ingest_ts) as payload,      -- latest delivery wins
        max(ingest_ts)              as last_seen_ts
    from staging.shipment_event_raw
    where ingest_date between :start_date and :end_date
    group by shipment_id, event_type, event_ts
) as s
on  t.shipment_id = s.shipment_id
and t.event_type  = s.event_type
and t.event_ts    = s.event_ts
when matched and s.last_seen_ts > t.last_seen_ts then update set
    payload      = s.payload,
    last_seen_ts = s.last_seen_ts
when not matched then insert
    (shipment_id, event_type, event_ts, payload, last_seen_ts)
values
    (s.shipment_id, s.event_type, s.event_ts, s.payload, s.last_seen_ts);
```

Three things to check before trusting it. The key must be the business event, not the offset or
the arrival time, or a replay will insert rather than merge. The `ingest_ts` comparison must be
monotonic across producers; if two producers write the same event with independently generated
timestamps, the guard is arbitrary and needs a tie-breaker. And the statement does not fix a
record that arrived after the retained window closed, so the late-data path still needs a
reconciliation against the source's own count.

Dialect notes: `arg_max` is available in DuckDB and Spark SQL; Snowflake spells it `max_by`, and
PostgreSQL uses `distinct on` with an ordered subquery. The merge pattern is the same in all three.

## 8. Data, standards, and regulatory context

Latency is mostly an engineering choice. Three areas are constrained by standards: the
representation of time, the separation of event time from record time, and the retention and
exchange obligations that make replay a legal requirement rather than a convenience.

### 8.1 Representing time

| Standard | What it fixes | Consequence for latency work | Grade |
|---|---|---|---|
| ISO 8601 | Date and time representation, including offsets | Every event timestamp must carry an offset or be declared UTC | B |
| RFC 3339 | A profile of ISO 8601 for internet timestamps | The format most APIs actually emit | B |
| IANA time zone database | Offset rules, including daylight saving transitions | Local planning cut-offs must resolve to an offset per date | B |
| Network Time Protocol (NTP) | Clock synchronisation to milliseconds on a network | Sets the floor on measurable latency across sites | B |
| Precision Time Protocol (PTP) | Sub-microsecond synchronisation on a local network | Relevant only for equipment-level control | B |

The practical rule: store event timestamps in UTC with the source offset preserved, and store the
planning cut-off as a local time plus a time zone identifier, resolved per date. A cut-off written
as "06:00" across a daylight-saving transition is a cut-off that moves by an hour once a year, and
the incident it causes will be attributed to the pipeline.

### 8.2 Event time and record time in EPCIS

GS1's EPCIS 2.0 standard makes the distinction explicit, which is why it is useful even for teams
that never implement it. Each event carries two timestamps.

| EPCIS field | Meaning | Maps to | Why both are needed |
|---|---|---|---|
| `eventTime` | When the event occurred in the physical world | Event time | Ordering, windows, service-level calculation |
| `recordTime` | When the event was captured in the information system | Ingestion time | Audit of capture delay, late-arrival analysis |
| `eventTimeZoneOffset` | The local offset at the event location | Offset preservation | Reconstructing local operating hours |

The existence of both fields is the standard's acknowledgement that capture is not instantaneous
and that a system which keeps only one timestamp cannot distinguish a late event from a late
capture. A pipeline that stores only `recordTime` has thrown away the ability to compute lateness,
and therefore the ability to set a watermark.

### 8.3 Regulatory clocks

Regulation rarely imposes a numeric latency bound on analytics. It imposes existence, retention,
and exchange obligations, and those obligations determine what must be replayable.

| Requirement | Jurisdiction | What it obliges | Numeric latency bound | Grade |
|---|---|---|---|---|
| DSCSA interoperable exchange | United States, pharmaceuticals | Transaction information exchanged at or before the transaction | None stated; exchange is tied to transfer of ownership | B |
| EU Falsified Medicines Directive | European Union | Verification of a medicine pack at dispensing | None; verification is at the point of use | B |
| FSMA 204 Food Traceability Rule | United States | Key data elements retained and produced on request | None stated; production on request within a reasonable time | B |
| EU e-invoicing and VAT reporting mandates | European Union and member states | Invoice data reported to the tax authority, in some states near real time | Member-state specific; often minutes to hours | B |

The design consequence is retention, not speed. A regulator asking "where was this lot on this
date" is asking a replay question. A stream with seven days of log retention cannot answer it; a
raw layer with two years of immutable partitions can. This is the strongest argument for keeping a
batch replay path even when the operational path is streaming, and it is an argument about
evidence rather than about latency.

> **Note:** Retention periods and exchange obligations vary by jurisdiction and product category,
> and the summaries above are educational. Confirm the applicable requirements with regulatory
> counsel before designing a retention policy around them.

### 8.4 What is convention rather than requirement

- "Real-time" is a marketing term with no agreed threshold. It has meant sub-second, sub-minute,
  and same-day in different vendor documents.
- Five-minute refresh is a habit inherited from polling intervals, not a requirement of any
  supply chain decision.
- A streaming architecture is not a maturity level. A nightly batch that meets its decision
  windows is a better system than a stream that does not.
- Exactly-once is a property of a boundary, not a badge a product carries.
- "Lambda" and "Kappa" name architectures from a specific decade and a specific vendor context.
  They are useful for arguing about duplicate logic and replay, not for classifying systems.

## 9. Industry practice and case evidence

The evidence on latency is abundant at the mechanism level and thin at the outcome level. Broker
and processor documentation describes precisely what is guaranteed. Academic work describes the
event-time model and its trade-offs. Practitioner writing describes overbuild. Almost nothing
measures whether a faster pipeline improved a supply chain decision.

### 9.1 What the broker and processor documentation actually says

Kafka's consumer documentation is unusually precise, and it repays reading. Its default is
at-least-once. Manual offset control, committing only after processing, gives at-least-once.
Exactly-once requires storing offsets in the same external system as the results, or using
transactions, in which case consumers must set `isolation.level=read_committed` to see only
committed data up to the last stable offset. The design proposal behind this,
KIP-98, introduces an idempotent producer with a producer identifier and sequence numbers, plus
transactional writes across partitions. None of this extends past the Kafka boundary. Grade C for
the mechanism, D for any claim that it delivers exactly-once to your application.

Apache Flink's documentation is equally direct. Processing time is the operator's system clock: it
is simple and low-latency and non-deterministic. Event time is the timestamp in the record:
results are consistent and replayable, at the cost of waiting for out-of-order events. A
`Watermark(t)` declares that no more elements with a timestamp at or before `t` should arrive. In
parallel operators the current event time is the **minimum** across input streams. The same page
documents allowed lateness and side outputs for late data. Grade D, but the mechanism description
is accurate and matches the academic model.

### 9.2 The academic foundation: correctness, latency, and cost

The clearest statement of the trade-off is Akidau et al., "The Dataflow Model" (PVLDB 8(12), 2015,
pp. 1792–1803). Its thesis is that a system should not assume unbounded data will ever become
complete, and should instead offer tunable trade-offs across correctness, latency, and cost.

Three of its claims matter here. First, watermarks are heuristic lower bounds, not correctness
signals: they can be too fast, letting late data arrive behind them, or too slow, letting one
straggler hold back every result. Second, triggers decouple when a result is emitted from when the
window is complete, so a system can publish provisional results and refine them. Third, late data
requires an explicit accumulation mode: discarding, accumulating, or accumulating and retracting.
The retraction mode exists because a pipeline with more than one grouping stage cannot correct a
downstream aggregate without being told to remove the old contribution. Grade A.

The supply chain translation is direct. A "shipments at risk" count published at 09:00 and revised
at 09:20 is an accumulating result with a watermark trigger. If a downstream report sums those
counts, it needs the retraction, or it double-counts the revision.

### 9.3 Lambda, Kappa, and the cost of two code paths

The Lambda architecture runs batch and stream paths over the same immutable input and merges the
results at query time. Jay Kreps' 2014 essay "Questioning the Lambda Architecture" is the
canonical critique, and it is worth noting that it is by Kreps, not by Kleppmann, a confusion that
is common in secondary sources. Kreps argues that maintaining equivalent logic in two distributed
systems is "exactly as painful as it seems," that the abstraction layers intended to hide the
duplication leak, and that the real benefit of a single path is developing, testing, debugging,
and operating one framework rather than two.

His alternative, which he named Kappa, keeps a log with long retention and reprocesses by starting
a second instance of the same streaming job from offset zero, writing to a new output table, and
cutting over when it catches up. The trade-off is temporary double storage during reprocessing
against the permanent cost of two code paths. Jimmy Lin's 2017 IEEE Internet Computing column
"The Lambda and the Kappa" is a useful secondary check: it treats Kappa as a trade-off rather than
an improvement, and notes that a streaming engine will not match batch throughput. Grade C for
Kreps as a practitioner essay, B for Lin as a peer-reviewed column.

For this topic the lesson is narrower than the architecture debate. The reason to keep a batch
path is not that batch is faster. It is that reprocessing and reconciliation need a mode with
cheap, repeatable, whole-population reads, and streaming engines are built for the opposite
access pattern.

### 9.4 Change data capture and the latency floor it removes

Log-based change data capture reads a database's write-ahead or binary log and emits row-level
changes, rather than polling tables for modified timestamps. Debezium's documentation describes
this as capturing all changes with low delay — milliseconds for MySQL and PostgreSQL — without
requiring schema changes, and it captures deletes and the before-image that a polling extract
misses.

The same documentation is candid about the limits. Incremental snapshots are chunked by primary
key and coordinated with a watermark so that a snapshot can run alongside log capture without
blocking; they require primary keys, and the documentation notes that snapshot `read` events do
not represent point-in-time state. For a supply chain team, the practical reading is that CDC
removes the source-extract latency for databases you own, which makes the source commit stage
small again, and does nothing for the WMS or the carrier gateway that batches at end of shift.
Grade D.

### 9.5 The overbuild anecdote

A widely shared practitioner post describes inheriting 47 Kafka topics at a company where three
needed sub-second latency; the other 44 were batch jobs "disguised as streaming." The author
reports monthly cost falling from about $22,000 to about $3,000 and incidents from about ten a
month to one, after consolidation. Grade E: a single, unverified, anonymous account. It is
useful as an illustration of a pattern that many engineers recognise, and it is not a measurement.
Do not quote the figures as evidence; quote the mechanism, which is that a fixed-cadence stream
pays for freshness that no decision consumes.

### 9.6 How freshness is actually measured

Three practitioner sources converge on the same method, which is worth stating because it
contradicts the single-timestamp dashboard.

Google Cloud's observability documentation defines data-processing freshness service-level
indicators over a distribution of lag, with an example threshold in which the oldest data element
processed is under 100 seconds, 99% of the time, over a rolling one-hour window. MerceMay's article on
measuring freshness argues that lag is a distribution over records, should be reported at p50,
p95, and p99 with a count of negative-lag records, and that a maximum-age check only proves the
pipeline is alive. Coveo's engineering blog implements a per-project threshold with a "95% of
minutes within threshold" ratio, computed from batch completion tables.

The consensus: measure event time to availability, per source and per partition, as a percentile,
with the measurement running outside the pipeline it monitors. Grade D for the vendor
documentation, E for the practitioner posts, but the method is consistent across all three and
matches the academic treatment of watermarks as a lag distribution.

### 9.7 What the evidence does not establish

No source found in this research measures whether a move from batch to streaming improved a
supply chain outcome — service level, spoilage, expedite cost, or planner time. Vendor case
studies assert such improvements, and they are Grade D by construction. The honest position is
that the mechanism is well understood and the outcome is not measured in public. Design the
measurement into the project, because nobody will supply it afterwards.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| "Real-time" measured at the dashboard | Refresh interval is quoted; source cadence is not | The dashboard poll is the only latency that is visible | A five-minute dashboard over a nightly feed | Measure event time to availability, per source |
| Processing latency reported as end-to-end | The job duration is the only number in the review | The largest stages are outside the job | A streaming project that changes nothing | Publish the five-stage ledger |
| Average freshness hides the stale partition | One headline timestamp; no per-DC view | Averages across partitions | A decision as stale as its worst input | Report p95 per source and partition |
| Watermark treated as completeness | "The window fired, so the data is complete" | Watermarks are heuristic lower bounds | Silently missing late records | Reconcile against a manifest or source count |
| One idle input stalls the watermark | No output for hours; job healthy | Global event time is the minimum across inputs | Every partition unmonitored | Idleness timeout; watermark-lag alarm |
| Late data dropped without a count | No side output; no late-event metric | Records behind the watermark are discarded | Silent undercount that grows with source drift | Side output plus a counted quarantine |
| Deduplication on arrival time or offset | Duplicate counts change after a replay | The key is transport-scoped, not business-scoped | Inflated inventory after every replay | Key on `(entity, event_type, event_ts)` |
| Exactly-once claimed end-to-end | The vendor deck says exactly-once | The guarantee stops at the system boundary | Duplicate postings at the sink | Say effectively-once; prove the idempotent write |
| Global ordering assumed | State transitions fail under load | Ordering is per partition | Out-of-order transitions, stuck state machines | Partition by entity; monotonic state machine |
| Micro-batch chosen without a consumer | Refresh every five minutes; reads are ad hoc | Cadence set by fashion, not by demand | Paying for 3,880 refreshes per read | Cost per consumed update |
| Streaming chosen for a human-lag decision | Alert goes to a shared mailbox | Freshness exceeds the action capability | Expensive alerts nobody acts on | Fix the rota before the pipeline |
| Unbounded state growth | Checkpoint size climbs weekly; recovery slows | Windows retained without TTL | Job restart takes hours, then fails | State TTL at window plus lateness plus margin |
| Backfill with no replay path | "We would have to rebuild the cluster" | Streaming output is not reproducible from raw | Cannot correct history or answer an audit | Retain raw; rehearse a replay quarterly |
| Freshness SLO with no breach action | The SLO is a number in a wiki | No owner, no consequence | Breaches discovered by users | Name a breach action and an owner |

Three deserve a note.

**Watermark treated as completeness.** This is the most consequential conceptual error in
streaming supply chain data, because it produces a number that looks finished. A window that
fires is a window that has waited as long as it was configured to wait. If a carrier's lateness
distribution shifted when it changed telematics vendors, the configured wait is now too short,
and the aggregate is missing records that will never be counted. The only defence is a
reconciliation against the source's own total, which is why the batch path remains necessary even
in a streaming architecture.

**One idle input stalls the watermark.** The failure is invisible to every conventional monitor.
The job is up, the checkpoints succeed, the CPU is idle, and the output is empty. It is found by a
human noticing a blank screen. Any streaming design with more than one input needs a progress
metric — watermark lag against wall clock — and an alarm on it. This is the single highest-value
monitoring addition in a multi-source pipeline.

**Micro-batch chosen without a consumer.** Fixed-cadence refresh multiplies cost by the number of
intervals in the day, while consumption does not multiply with it. The ratio in section 6.4 was
3,880 to 1. That ratio is not unusual; it is the default outcome of setting a cadence by
convention. Compute the ratio before the next cadence review.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The decision window is known and stable | The freshness budget derives from it | Reviews move, cut-offs shift with season | Ask the decision owner to state the deadline; compare to the calendar |
| The source records event time | Lateness and event-time windows depend on it | Sources stamp at commit, not at occurrence | Compare source timestamp to a known physical event |
| Clocks are synchronised within the tolerance | Latency measurement and ordering need it | Unsynchronised site clocks, virtual machines | Record source clock offset; alarm above tolerance |
| The actor can respond inside the window | Freshness is only useful if acted on | Rota gaps, shared mailboxes, approval chains | Measure availability-to-action, not availability |
| Lateness is stationary enough to set a bound | Watermarks are configured from observed lateness | New carrier, gateway change, holiday peak | Re-estimate monthly; alarm when exceeded |
| The idempotency key is unique per business event | Effectively-once depends on it | Reused identifiers, corrections with the same timestamp | Count duplicate keys in raw; expect a stable rate |
| Volume is predictable enough to size capacity | Streaming capacity is provisioned continuously | Promotions, seasonal peaks, source bursts | Compare p99 throughput to provisioned capacity |
| Retention is affordable for the replay window | Replay and audit need retained input | Volume growth, privacy constraints, cost pressure | Cost per month against the value of a replay |
| The team can operate a stateful service | Streaming shifts the burden to operations | No on-call, no runbooks, no savepoint discipline | Run a restore drill; measure recovery time |
| Freshness can be expressed as one number per decision | Budgets need a target | Decisions with several inputs of different ages | Check whether the oldest input changes the action |

Two limits are worth stating plainly.

**This topic assumes the decision is worth making faster.** Some are not. A weekly supply review
that concludes the same thing every week does not become more valuable at five-minute latency; it
becomes more expensive. The framework here helps you compute whether a decision justifies its
latency, and it cannot tell you whether the decision is worth making at all.

**The cost figures are structural, not market prices.** Cloud and platform pricing changes, and
per-unit rates vary by contract, region, and commitment. The worked example uses internally
consistent illustrative costs to demonstrate the arithmetic of cost per consumed update. Substitute
your own rates before making a decision, and keep the ratio, which is far more stable than the
absolute number.

## 12. Exercises

**Exercise 1 — Draw your latency ladder (design, 45 minutes).** Choose a supply chain you know and
place at least eight decisions on the ladder from section 3.1.
**Deliverable:** one hand-drawn ladder with freshness needed, decision, actor, and mode at each
rung.
**Acceptance criteria:** eight decisions; freshness stated in a unit; at least one decision placed
on a rung slower than the current implementation, with a reason; one rung marked as unjustified.

**Exercise 2 — Derive five freshness requirements (calculation, 40 minutes).** For each case,
compute the required p95 data age and name the cheapest mode that meets it.

| Case | Window | Action lag | Margin |
|---|---:|---:|---:|
| Cold-chain quarantine | 15 min | 6 min | 2 min |
| Live order promise | 120 s | 5 s | 15 s |
| Dock re-slot | 25 min | 4 min | 3 min |
| Intraday reallocation | 5 h | 40 min | 20 min |
| DC replenishment | 16 h | 0 | 90 min |

**Deliverable:** a table of requirements in a single unit, with the chosen mode and one sentence of
justification each.
**Acceptance criteria:** arithmetic shown; all values converted to the same unit; a mode rejected
for at least one case with the reason stated in latency terms.

**Exercise 3 — Attribute data age (calculation, 50 minutes).** A replenishment decision is taken at
06:00. The last warehouse transaction is at 22:40, the WMS commits at 23:00, the extract starts at
00:30, and the mart is available at 01:35. Compute the data age at decision, decompose it into
source commit, schedule wait, pipeline, and decision wait, and rank the interventions.
**Deliverable:** a waterfall table with minutes and percentage shares, plus a ranked list of three
interventions with the minutes each would save.
**Acceptance criteria:** shares sum to 100%; the intervention ranking is by minutes saved, not by
ease; the recommendation states what it does *not* fix.

**Exercise 4 — Design a late-data policy (design, 55 minutes).** A carrier event stream arrives
with a p99 lateness of 40 minutes and an occasional tail of six hours. The consuming decision is a
30-minute "shipments at risk" window.
**Deliverable:** a policy table with bounded out-of-orderness, allowed lateness, idleness timeout,
side-output handling, state TTL, and the alarm thresholds.
**Acceptance criteria:** every parameter has a value and a justification from the stated
distribution; the six-hour tail has a defined destination; the policy states what a consumer sees
when a result is provisional.

**Exercise 5 — Choose the deduplication key (design, 45 minutes).** A shipment event feed contains
duplicates, redeliveries after a consumer restart, and genuine corrections to estimated arrival
times.
**Deliverable:** a key definition, a merge rule, a monotonic guard, and a table of three cases
showing the resulting state after each is applied twice.
**Acceptance criteria:** the key excludes offset and arrival time; a correction is distinguished
from a duplicate; the merge is shown to be idempotent by a two-run trace.

**Exercise 6 — Price the cadence (calculation, 60 minutes).** An order-promise service refreshes
43,200 item-location rows every five minutes. It serves 3,000 promises a day. The platform costs
€9,000 a month, of which €7,200 is variable. Compute the cost per refresh, the cost per consumed
update, and the ratio. Then estimate the effect of refreshing only when an item-location changes,
at 24,000 changes a day.
**Deliverable:** a table of the six quantities plus the redesigned monthly cost and annual saving.
**Acceptance criteria:** all arithmetic shown; the fixed-versus-variable split used explicitly;
the answer states why the fixed floor dominates after redesign.

**Exercise 7 — Diagnose the stall (analysis, 50 minutes).** A nine-DC exception stream produces no
output between 22:00 and 05:40. The job is healthy, checkpoints succeed, and CPU is idle. One DC's
gateway was down.
**Deliverable:** a causal chain from the gateway to the empty output, the metric that would have
detected it within 15 minutes, and the configuration change that limits the blast radius.
**Acceptance criteria:** the mechanism names the minimum-across-inputs rule; the detection metric is
a progress metric, not a failure metric; the fix is per-source idleness, not a longer window.

**Exercise 8 — The streaming justification memo (design, 60 minutes).** A planner asks for
"real-time inventory" for the 06:00 replenishment decision. Write the memo.
**Deliverable:** a one-page memo with the current five-stage ledger, the required data age, the
data age achieved, the proposed change, its cost, and its effect on the decision.
**Acceptance criteria:** the memo separates processing latency from data age at decision; it
quantifies the improvement in minutes and as a share of the budget; it recommends an alternative
use of the same budget, or explains why none exists.

**Exercise 9 — Write the freshness SLO (design, 45 minutes).** For one decision, write a service
level that could be monitored.
**Deliverable:** an SLO with the event definition, the timestamp pair, the percentile, the
measurement window, the per-partition breakdown, the breach threshold, the breach action, and the
owner.
**Acceptance criteria:** measured from event time; p95 rather than mean; per-partition reporting;
a breach action that a named role can execute without a meeting; the measurement runs outside the
pipeline it monitors.

## 13. Self-check answers

**Exercise 2 arithmetic.** Convert every case to minutes at the point of calculation, then state
the result in the unit the decision uses.

| Case | Window | Action | Margin | Required | Mode |
|---|---:|---:|---:|---:|---|
| Cold-chain quarantine | 15.00 min | 6.00 | 2.00 | **7.00 min (420 s)** | Stream |
| Live order promise | 2.00 min | 0.083 | 0.25 | **1.667 min (100 s)** | On-demand query or stream |
| Dock re-slot | 25.00 min | 4.00 | 3.00 | **18.00 min** | Event push or 5-minute micro-batch |
| Intraday reallocation | 300.00 min | 40.00 | 20.00 | **240.00 min (4 h)** | 15-minute micro-batch |
| DC replenishment | 960.00 min | 0 | 90.00 | **870.00 min (14.5 h)** | Nightly batch |

The order promise is the case that catches people. 120 seconds minus 5 seconds minus 15 seconds
is 100 seconds, and a five-minute micro-batch has a worst-case data age of 300 seconds, three
times the budget. A one-minute micro-batch would still fail at the margin. The cheapest mode that
meets 100 seconds is a direct query against the inventory service at the moment of the quote, or
a stream; it is not a cadence change.

**Exercise 3 arithmetic.**

| Segment | From | To | Minutes | Share |
|---|---|---|---|---:|
| Source commit | 22:40 | 23:00 | 20 | 4.5% |
| Schedule wait | 23:00 | 00:30 | 90 | 20.5% |
| Pipeline | 00:30 | 01:35 | 65 | 14.8% |
| Decision wait | 01:35 | 06:00 | 265 | 60.2% |
| **Total** | **22:40** | **06:00** | **440** | **100%** |

Check: 20 + 90 + 65 + 265 = 440 minutes, which is 7 hours 20 minutes, and the shares sum to
100.0% after rounding.

Ranked interventions. First, move the decision earlier: a decision at 03:00 instead of 06:00
saves 180 minutes, which is 40.9% of 440. Second, move the extract earlier: starting at 23:00
instead of 00:30 saves 90 minutes, 20.5%. Third, make the pipeline five minutes instead of 65,
which saves 60 minutes, 13.6%. Note that the first and second overlap: moving the extract to
23:00 also makes an earlier decision possible, so the combined saving is 270 minutes, not 270
plus.

What the recommendation does not fix: none of these improves the *source* commit, and none
improves correctness. If the decision moves to 03:00, the planners must be at work at 03:00, which
is a rota change, not a data change. Say so in the memo.

**Exercise 6 arithmetic.**

| Quantity | Value | Arithmetic |
|---|---:|---|
| Item-locations | 43,200 | 4,800 x 9 |
| Refreshes per day | 288 | 1,440 / 5 |
| Row refreshes per month | 373,248,000 | 43,200 x 288 x 30 |
| Promises consumed per month | 90,000 | 3,000 x 30 |
| Cost per row refresh | €0.0000241 | 9,000 / 373,248,000 |
| Cost per consumed update | €0.100 | 9,000 / 90,000 |
| Ratio | 4,147x | 0.100 / 0.0000241 |

Redesign: 24,000 changed item-locations a day is 720,000 a month. The new variable cost is
720,000 x (€7,200 / 373,248,000) = €13.89. With the €1,500 fixed floor, the new monthly cost is
€1,513.89, a saving of €7,486.11 a month, or €89,833 a year.

The fixed floor is 99.1% of the redesigned cost, so further volume reduction is pointless. That
is the correct stopping signal: the design has moved from volume-dominated to fixed-dominated,
and the next question is whether the fixed platform should exist at all, not whether the cadence
can be reduced again.

**Rubric for Exercise 1.** A strong ladder states freshness in a unit, names the actor as well as
the decision, and places at least one decision on a slower rung than its current implementation
with a justification based on the decision window. It marks one rung as unjustified and says what
evidence would justify it. A weak ladder lists technologies rather than decisions, or places every
decision on the same rung.

**Rubric for Exercise 4.** A strong policy sets bounded out-of-orderness from the stated p99 of 40
minutes — commonly 45 to 60 minutes — sets allowed lateness to cover the six-hour tail or routes
that tail to a side output with an alarm, and sets the idleness timeout slightly above the normal
inter-event gap. It states that results are provisional until the watermark passes plus allowed
lateness, and defines what the consumer sees. A weak policy sets a round number such as 10 minutes
with no reference to the distribution, and drops late data silently.

**Rubric for Exercise 5.** A strong answer uses `(shipment_id, event_type, event_ts)` as the key,
distinguishes a correction from a duplicate by a version or ingest timestamp, and shows that
applying the merge twice yields the same state. It names the case that the key cannot resolve —
two genuine events of the same type with the same timestamp — and states the tie-breaker. A weak
answer keys on the message identifier or the arrival time, or treats every later delivery as a
correction without a guard.

**Rubric for Exercise 7.** A strong causal chain names the minimum-across-inputs rule, identifies
the missing metric as watermark lag against wall clock, and fixes the blast radius with per-source
idleness. It also notes that a longer window would not have helped, because the watermark was not
advancing at all. A weak answer proposes restarting the job or increasing the watermark delay,
neither of which addresses a stalled input.

**Rubric for Exercise 8.** A strong memo shows that 440 minutes of data age contains 65 minutes of
pipeline, and that the required age is 870 minutes, so the proposed change improves a number that
has 430 minutes of slack and changes no decision. It quantifies the improvement as a share of the
budget and recommends either moving the decision or addressing the exception-management case,
which has a different window. A weak memo argues about dashboard refresh rates.

**Rubric for Exercise 9.** A strong SLO measures event time to consumer read, at p95, per DC, over
a rolling seven days, with a breach action such as publishing the prior plan flagged as stale, an
owner role, and a monitor that runs outside the pipeline. A weak SLO says "data available by
04:30" and calls the job's completion time the measurement.

## 14. Interview questions

**1. "What is the difference between processing latency and data age at decision?"** A strong
answer defines the first as job duration and the second as decision time minus the oldest input's
event time, then gives an example where the pipeline is 15% of the data age and the decision wait
is 60%. A weak answer treats them as the same number.

**2. "How would you decide between batch and streaming for a replenishment decision?"** A strong
answer derives the decision window, subtracts the action lag and a margin, compares the required
data age with what each mode achieves at p95, and chooses the slowest mode that meets it. It notes
that the answer is usually batch because the window is hours. A weak answer says streaming is more
modern or batch is cheaper without the arithmetic.

**3. "What is a watermark, and what does it guarantee?"** A strong answer calls it a heuristic
lower bound on event time, notes that it can be too fast or too slow, and says it guarantees
nothing about completeness. It mentions the reconciliation that does. A weak answer says it
guarantees all data has arrived.

**4. "Nine sources feed a stream and one goes silent. What happens?"** A strong answer explains
that the global watermark is the minimum across inputs, so the watermark stalls, no windows fire
for any source, and the job stays healthy. It proposes idleness detection and a watermark-lag
alarm. A weak answer says only the silent source is affected.

**5. "Is exactly-once possible?"** A strong answer distinguishes exactly-once effect within a
system boundary from end-to-end delivery, notes that at-least-once plus an idempotent write on a
business key gives effectively-once, and names a key such as `(shipment_id, event_type,
event_ts)`. A weak answer says yes because Kafka supports it.

**6. "Why not deduplicate on the message offset?"** A strong answer says the offset is a transport
position that changes on replay, so the same business event appears at a new offset and the
duplicate survives. It proposes a business key. A weak answer says offsets are unique.

**7. "A vendor claims sub-second latency. What do you ask?"** A strong answer asks for the
five-stage ledger: source commit, transport, processing, publish, and action. It notes that the
quoted number is usually transport, and that source commit or human action often dominates. A weak
answer asks for a benchmark.

**8. "How do you set the allowed lateness for a window?"** A strong answer uses the observed
lateness distribution, sets bounded out-of-orderness near the p99, sets allowed lateness to the
business tolerance for restatement, routes the tail to a side output with a count, and re-estimates
monthly. A weak answer picks a round number.

**9. "How would you measure whether a streaming investment paid off?"** A strong answer measures
cost per consumed update and cost per decision changed, and defines a counterfactual for the
decision — orders changed, spoilage avoided, expedites prevented. It admits the counterfactual is
hard and proposes a before-and-after with a stated comparison period. A weak answer cites
dashboard latency.

**10. "When is micro-batch the right choice?"** A strong answer places it between batch and stream,
notes that it is a sequence of small batch jobs with checkpoints and a per-batch transaction, and
chooses it for a one-to-fifteen-minute requirement where the operational simplicity of batch
recovery is worth the latency floor. A weak answer calls it "streaming with a small interval."

**11. "What breaks when you replay a month of events through a streaming pipeline?"** A strong
answer names state size and recovery time, watermark behaviour on historical data, sink
idempotency, and the difference between event-time and processing-time windows. It notes that a
batch path is usually needed for the replay itself. A weak answer says replay is just rerunning
the job.

## 15. Key terms

- **Latency** — elapsed time between two named points; never meaningful without both.
- **Processing latency** — time from a record entering a processor to its result leaving.
- **End-to-end data latency** — time from a business event to a consumer being able to read it.
- **Data age at decision** — decision time minus the event time of the oldest input used.
- **Decision lag** — time from data availability to action.
- **Decision window** — the interval in which action still changes the outcome.
- **Freshness** — the age of the newest event a dataset reflects, reported as a percentile.
- **Freshness budget** — the allocation of tolerable data age across stages for one decision.
- **Batch** — scheduled processing of a bounded input.
- **Micro-batch** — repeated small batch jobs at short intervals, with per-batch checkpoints.
- **Stream** — continuous processing of an unbounded input as records arrive.
- **Event push** — producer-initiated delivery to a consumer endpoint, with no broker retention.
- **Event time** — when the business event occurred.
- **Processing time** — when the system handled the record.
- **Watermark** — a heuristic event-time progress marker; not a completeness proof.
- **Bounded out-of-orderness** — the lateness allowance used to generate a watermark.
- **Allowed lateness** — the grace period after the watermark in which late records still count.
- **Late event** — a record whose event time precedes the watermark.
- **Side output** — a diverted stream for records excluded from the main path.
- **At-least-once** — no loss, possible duplicates.
- **Exactly-once** — a single processing effect, normally only within one system boundary.
- **Effectively-once** — at-least-once delivery plus an idempotent write on a business key.
- **Idempotency key** — the fields that make a repeated write converge on one state.
- **Partition key** — the field that defines the ordering scope.
- **Log retention** — how long a stream can be replayed.
- **Change data capture (CDC)** — emitting row-level database changes from the change log.
- **Lambda architecture** — parallel batch and stream paths merged at query time.
- **Kappa architecture** — one stream path, reprocessed by replaying the log.
- **Replay** — reprocessing retained input to reproduce or correct an output.
- **Freshness SLO** — a percentile bound on data age with a window and a breach action.

## 16. Sources

**Peer-reviewed and academic**

- Akidau, T., Bradshaw, R., Chambers, C., Chernyak, S., Fernández-Moctezuma, R. J., Lax, R.,
  McVeety, S., Mills, D., Perry, F., Schmidt, E., and Whittle, S. (2015). "The Dataflow Model: A
  Practical Approach to Balancing Correctness, Latency, and Cost in Massive-Scale, Unbounded,
  Out-of-Order Data Processing." *PVLDB*, 8(12), 1792–1803. DOI: 10.14778/2824032.2824076. (A)
- Lin, J. (2017). "The Lambda and the Kappa." *IEEE Internet Computing*. Trade-offs between
  Lambda and Kappa; argues Kappa is not a free lunch. (B)

**Standards and specifications**

- ISO 8601, date and time representation. (B)
- RFC 3339, *Date and Time on the Internet: Timestamps*. (B)
- IANA Time Zone Database. (B)
- GS1, *EPCIS 2.0* and the EPCIS event model, including `eventTime`, `recordTime`, and
  `eventTimeZoneOffset`. (B)
- GS1 General Specifications, current edition. (B)
- U.S. Food and Drug Administration, DSCSA interoperable exchange requirements. (B)
- European Commission, VAT in the Digital Age (ViDA) proposal, and member-state e-invoicing
  mandates; consult the current consolidated texts and the relevant implementing rules. (B)

**Vendor and product documentation**

- Apache Software Foundation, "KafkaConsumer" Javadoc, Kafka 3.9.2, delivery semantics and
  `isolation.level`. (D)
- Apache Software Foundation, KIP-98, "Exactly Once Delivery and Transactional Messaging." (C)
- Confluent, "Kafka Message Delivery Guarantees," documentation, accessed 2 November 2026. (D)
- Apache Flink, "Timely Stream Processing," concepts, accessed 2 November 2026. (D)
- Apache Flink, "Generating Watermarks," DataStream API guide, accessed 2 November 2026. (D)
- Apache Spark, "Structured Streaming Programming Guide," 4.3.0, triggers and micro-batch
  latency, accessed 2 November 2026. (D)
- Apache Spark, "Real-time Mode," in-progress documentation, accessed 2 November 2026. (D)
- Debezium, "Debezium Features," and "Incremental Snapshots in Debezium," 7 October 2021. (D)
- Google Cloud, "Data processing services" service-level indicators, Cloud Observability
  documentation, accessed 2 November 2026. (D)

**Practitioner essays, press, and engineering blogs**

- Kreps, J. (2014). "Questioning the Lambda Architecture." O'Reilly Radar, 2 July 2014.
  https://www.oreilly.com/radar/questioning-the-lambda-architecture/ (C)
- MerceMay, "Measuring Freshness From Event Time, Not Arrival Time," data engineering article,
  accessed 2 November 2026. (E)
- Coveo, "Keeping our data pipelines under watch and on good behavior," engineering blog,
  7 May 2024. (E)
- "Real-Time is an SLA, Not an Architecture: When You Actually Need Kafka (And When You Don't),"
  DEV Community, accessed 2 November 2026. Single unverified anecdote. (E)

**Note on grading.** Grades follow Module 28: A peer-reviewed or audited, B regulatory or official,
C framework, design proposal, or sustained practitioner essay, D vendor documentation, E press,
blog, or anecdote. Vendor pages are dated at the version consulted. The 47-topic cost anecdote is
reported as the author states it and is not independently verified; it is used as an illustration
of a mechanism, not as evidence of a typical outcome.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Prerequisite topic: [`01_layered-architecture.md`](01_layered-architecture.md)
- Related sibling: [`../02_supply_chain_data_landscape/06_data-movement.md`](../02_supply_chain_data_landscape/06_data-movement.md)
- Module 02 map: [`../02_supply_chain_data_landscape/_index.md`](../02_supply_chain_data_landscape/_index.md)
- Survey file: [`../../../course/03_data_engineering_interoperability.md`](../../../course/03_data_engineering_interoperability.md)
- Next topic (not yet written): `03_ingestion-patterns.md`
- Later topic (not yet written): `07_orchestration.md`
- Return to the part map: [`../_index.md`](../_index.md)
