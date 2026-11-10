---
collection: streaming-event-systems
file_id: "01"
topic: Kafka consumer partitions and lag
level: 3
status: draft
version: "2.0"
created: "2026-11-07"
reviewed: null
review_trigger: Kafka client, broker, group protocol, workload or sink contract changes
notation_style: plain-text
prerequisites: ["../INDEX.md"]
pattern_tags: [partitioning, consumer-groups, recovery]
source_status: OPEN
owner_role: streaming-platform maintainer
source_index: INDEX.md
---
# Kafka Consumer Partitions and Lag

## 1. File Header

> **Purpose:** Diagnose a lagging consumer group at partition granularity, decide which of four limiting resources is actually binding, and advance offsets without ever skipping an unfinished durable effect.
> **Prerequisites:** Log-structured storage, at-least-once delivery, rate-and-unit arithmetic, and relational transaction commit; the collection map is at [module index](../INDEX.md).
> **Key outcome:** Given per-partition arrival rates, per-partition durable completion rates, committed offsets, assignment and a sink contract, you can (a) compute whether the backlog can drain at all, (b) compute how long a drain takes under stated assumptions, (c) choose between adding members, raising per-record efficiency, repartitioning, or relieving a downstream sink, and (d) state the largest safe commit position after out-of-order completions.
> **This file does NOT cover:** Change-data-capture snapshot handoff (see the CDC sheet), watermark and session-window semantics (see the Flink sheet), and broker-side storage tiering or quota administration beyond the sizing arithmetic in Section 9.
> **Level:** 3 — advanced production; assumes the reader has already operated a consumer group.
> **Version / reviewed:** 2.0 / not reviewed. Review status is OPEN, not "passed".
> **Evidence:** DERIVED and ILLUSTRATIVE. Every numeric workload in this file is a constructed teaching example. No production measurement, benchmark, incident or vendor performance claim is asserted anywhere below.
> **Retrieval cue:** "Lag is growing" → Sections 4.4 and 5.2. "Which member is slow" → Section 8.3. "Can I commit past offset X" → Sections 4.2 and 10.2. "Should I add consumers" → Section 7 and Section 4.4.

The decision this sheet supports is narrower than "make the dashboard green". A consumer group
holds two independent kinds of state: *where it may read next* (offsets) and *what the outside
world has durably observed* (effects). Optimising the first without the second produces a system
that reports zero lag while silently dropping business events; optimising the second without the
first produces correct-but-duplicating replay. Almost every real incident in this area is a
disagreement between those two clocks, amplified by an assumption that a topic behaves like a
single queue.

A second decision is resource attribution. A rising aggregate lag number is consistent with at
least five distinct worlds: producers genuinely increased volume; one key concentrates traffic on
one partition; the group has fewer members than partitions; the members are healthy but blocked on
a database; or the measurement itself is wrong because two dashboards subtract different end
offsets. These worlds require mutually exclusive fixes. Adding consumers helps exactly one of
them, and actively hurts one other by multiplying pressure on an already saturated sink.

**Provenance honesty.** The source tree referenced by the collection brief was not readable in the
workspace where this draft was written, so nothing here is presented as originating from it. Kafka
semantics quoted below are the stable, documented behaviours of the partitioned log and the classic
consumer-group protocol; anything that varies by release — default timeouts, assignment strategy
defaults, whether the group coordinator assigns on the client or the broker, whether the reported
end offset is transaction-aware — is flagged as version-sensitive and must be re-checked against
the broker and client versions actually deployed. The [Kafka documentation](https://kafka.apache.org/documentation/)
is the primary reference for that check.

**Takeaway:** Establish a correct read boundary and a correct effect boundary first; only then does the word "lag" identify a problem you can fix.

## 2. Concept Map / Mental Model

The first map is the write-and-read path. Read it left to right: a producer's key decides a
partition, the partition decides an order, the group's assignment decides who may read that order,
and the sink decides when the order has actually been honoured.

```text
 producer record {key, value, headers}
        |
        v
 partitioner: murmur2(key) mod P   (null key -> sticky/round-robin, version-specific)
        |
        v
 +-------------------+   +-------------------+        +-------------------+
 | partition 0 log   |   | partition 1 log   |  ...   | partition P-1 log |
 | off 0,1,2,3,...   |   | off 0,1,2,3,...   |        | off 0,1,2,3,...   |
 +---------+---------+   +---------+---------+        +---------+---------+
           |                       |                            |
           +-----------+-----------+----------------------------+
                       |
                       v
        group coordinator: membership, generation/epoch, assignment
                       |
        +--------------+---------------+
        |              |               |
     member A       member B        member C        (partitions owned, not records split)
        |              |               |
        v              v               v
   fetch -> poll -> decode -> business logic -> durable sink effect
        |              |               |
        v              v               v
   position P_i    in-flight set   effect ledger (idempotency keys)
        |                              |
        +--------> safe commit frontier <--------+
                         |
                         v
              __consumer_offsets: committed next offset C_i
```

Note what is *not* in that picture: there is no cross-partition ordering edge, and there is no
edge from "record was read" to "record was honoured". Both omissions are the source of most
misdiagnosis.

The second map is the diagnostic decision tree. It is ordered deliberately: capacity questions
come before tuning questions, because no amount of tuning recovers a system whose arrival rate
exceeds its service rate on a partition that cannot be split.

```text
 lag L_max rising?
   |
   +-- no, but business freshness is bad --> lag metric disagrees with objective
   |        check: end-offset definition, isolation level, scrape skew (Section 8.9)
   |
   +-- yes
        |
        +-- is one partition's lag far above the others?
        |      |
        |      +-- yes -> skew. Is its ingress concentrated on few keys? (8.3)
        |      |        yes -> hot key: salt / re-key / redesign (4.4, 7)
        |      |        no   -> per-record cost differs, or owner is degraded (8.4)
        |      |
        |      +-- no -> uniform growth
        |               |
        |               +-- members < partitions? -> add members (4.4)
        |               +-- members >= partitions? -> member-side or sink-side limit
        |                        profile: CPU, GC, poll cadence, sink latency, retries (8.5)
        |
        +-- does drain even exist?  compare a_i to s_i per partition (4.1)
                 a_i >= s_i on any partition -> no finite recovery without change
```

Three state variables must be kept separate in the reader's head, because they move
independently and each has its own failure mode:

| Variable | Owner | Meaning | Typical failure when confused |
|---|---|---|---|
| Group membership / assignment | coordinator + members | which member may read which partition, under which epoch | stale owner keeps writing effects after revocation |
| Position / committed offset | consumer + `__consumer_offsets` | next offset to read; checkpoint for restart | premature commit destroys the only record of unfinished work |
| Durable effect | the sink | what the outside world has actually observed | duplicates on replay, or silent loss with a green dashboard |

The mental model to carry into an incident is therefore a *two-clock* model. Clock one is the log:
monotonic within a partition, cheap to query, and always available. Clock two is the sink: not
monotonic, not queryable in Kafka, and authoritative for correctness. Lag measures the distance
between clock one's write head and clock one's read head. It says nothing at all about clock two.

**Takeaway:** Partitions supply order, groups supply ownership, sinks supply truth — lag measures only the first two.

## 3. Foundational Definitions

Every definition below is scoped to one topic, one group and one observation instant unless stated
otherwise. Mixing scopes is the single most common cause of a number that cannot be reproduced.

### 3.1 Log and addressing terms

A **topic** is a named collection of partitions. It is not an ordered sequence, and "the topic's
offset" is meaningless: offsets exist per partition and are only comparable within one partition.
The number of partitions is fixed at creation in the sense that it may be increased but never
decreased, and increasing it remaps keys.

A **partition** is an append-only, immutable, ordered log. Records are assigned strictly
increasing offsets by the broker as they are appended. The offset is an address, not an identity:
compaction, aborted transactions and control records can leave holes, so subtracting two offsets
gives an *offset distance*, which is not necessarily a count of business records.

A **record key** is producer-supplied bytes. With the default partitioner the key's hash selects
the partition deterministically for a fixed partition count, which is what makes per-key ordering
possible. A null key has no such contract; behaviour depends on client version (older clients
rotate, newer clients use a sticky partitioner that changes target on batch completion).

The **log end offset** (LEO) is the next offset that would be assigned to a new append. The **high
watermark** (HW) is the highest offset that has been replicated to the in-sync replica set;
consumers can only read below the HW, so a healthy cluster can legitimately show non-zero lag for a
record that was written microseconds ago. The **last stable offset** (LSO) is the transaction-aware
boundary: with `isolation.level=read_committed`, a consumer cannot read past the first open
transaction, so a producer that opens a transaction and stalls will pin the readable boundary while
LEO keeps advancing.

The **log start offset** is the earliest addressable position. It advances when retention deletes
segments or compaction rewrites them. Consequently a committed offset that was perfectly valid
yesterday can become *out of range* today.

| Term | Formal meaning | Units / domain | Not the same as |
|---|---|---|---|
| Topic | Named set of partitions | — | A queue with a global order |
| Partition | Ordered append log | Index `0..P-1` | A shard of records by count |
| Offset | Position within one partition | Integer, monotonic, gapped | Event time; global ID |
| LEO | Next offset to be appended | Integer per partition | Number of live records |
| HW | Replicated-safe read boundary | Integer per partition | LEO |
| LSO | Transaction-visible boundary | Integer per partition | HW, when transactions are in use |
| Log start offset | Earliest addressable position | Integer per partition | Offset 0 |
| Committed offset `C` | Group checkpoint: next offset to read | Integer per partition | Last processed record's offset |

### 3.2 Group coordination terms

A **consumer group** is a named set of members that collectively consume a subscription, with
per-partition offsets stored under the group id. The group id is the identity of the *state*, not
of the deployment: two applications that share a group id will silently steal each other's
partitions and interleave their offset commits.

The **group coordinator** is the broker responsible for membership and offset storage for that
group. **Membership** is maintained by heartbeats; a member that stops heartbeating for the
session timeout is removed. **Rebalance** is the protocol transition that redistributes partitions
when membership, subscription or partition count changes. Each successful rebalance produces a new
**generation** (classic protocol) or **assignment epoch** (newer server-side protocols); writes and
commits carrying a stale generation are rejected. **Fencing** means ensuring that work begun under
an old epoch cannot produce effects after the epoch changed.

An **assignment strategy** decides which member gets which partition. Range and round-robin
strategies are eager: every rebalance revokes all partitions from all members before reassigning.
Cooperative-sticky strategies revoke only what must move, which reduces the stop-the-world window.
The choice is a client-side configuration in the classic protocol and a broker-side policy in the
newer protocol; verify which applies to the deployed version.

| Term | Formal meaning | Failure it prevents | Failure it causes when wrong |
|---|---|---|---|
| Group id | Namespace for offsets and membership | Two apps sharing state | Split-brain consumption, mixed commits |
| Generation / epoch | Monotonic assignment version | Stale owner writing effects | Fencing rejects legitimate late commits |
| Session timeout | Heartbeat liveness window | Ghost members holding partitions | Churn under GC pauses |
| `max.poll.interval.ms` | Max time between `poll()` calls | Silent-but-alive stuck member | Rebalance storms from slow batches |
| Assignment strategy | Partition-to-member mapping rule | Manual partition pinning | Full revoke on every restart |

### 3.3 Position, lag and freshness terms

The **position** `P_i` is the offset the consumer will fetch next from partition `i`. It exists
only in the consumer's memory and moves forward as records are returned by `poll()`. It is not a
promise about processing. The **committed offset** `C_i` is the durable checkpoint: on restart the
group resumes at `C_i`, which is why `C_i` points at the *next* record to read rather than the last
one completed.

**Offset lag** is `E_i − C_i` for a declared end boundary `E_i`. Common boundary choices are LEO,
the high watermark, and the transaction-aware readable boundary; they produce different numbers for
the same instant. **Time lag** is the age of the oldest unprocessed record, measured from the
record's timestamp to now. **Oldest-record age** is the same idea measured on the oldest *record*,
not the oldest offset. **Consumer time lag** (used by some tooling) is derived from the commit
timestamp instead, and answers a different question.

These four quantities diverge exactly when it matters. A partition with 5,000,000 records of lag
at 12,000 records/s ingress is about 417 seconds behind; a partition with 5,000 records of lag
where one record is eight hours old is a severe freshness breach hidden inside a small number.
Offset lag answers "how much work"; time lag answers "how stale is a customer". A service-level
objective is almost always expressed in the second and alerted on the first, which is a design bug.

### 3.4 Rate, capacity and skew terms

The **arrival rate** `a_i` is the effective number of records appended to partition `i` per second
over a declared interval. The **service rate** `s_i` is the sustained rate at which required
effects for partition `i` become durable, measured under the same conditions, excluding retried
work that never succeeds. Both are averages over the stated window and both must be recorded in
the same unit as the backlog; comparing a record backlog to a byte rate is the arithmetic
equivalent of comparing metres to seconds.

**Skew** is the concentration of work across partitions. It is measured on ingress records, on
bytes, or on *cost-weighted* work, and the three can disagree sharply: a partition carrying 8% of
records but 40% of bytes is skewed in the dimension that matters for I/O. **Hot partition** means a
partition whose offered load exceeds the achievable single-partition service rate, so no
redistribution of the *group* can help it. **Hot key** is the usual cause: a key whose hash lands
in one partition and whose rate alone saturates that partition.

**Backpressure** is the propagation of a downstream capacity limit upstream through bounded buffers,
blocking calls or paused partitions. Its distinguishing signature is that upstream lag rises while
downstream utilisation is at its ceiling, and consumer CPU is *not* saturated. **Replay** is
re-reading records from an earlier addressable offset for recovery or recomputation, and it is the
only sanctioned response to lost or corrupted downstream state.

### 3.5 Terms that are routinely conflated

Correct diagnosis depends on keeping these pairs apart. Each pair below looks interchangeable in a
dashboard and is not interchangeable in a decision.

- **At-least-once vs exactly-once effects.** Kafka's transactional producer can make
  read-process-write atomic *within Kafka*. It cannot make an arbitrary external HTTP call or a
  non-transactional database write exactly-once. That requires an idempotent sink keyed by a stable
  logical event id, with deduplication retention at least as long as the supported replay horizon.
- **Offset reset vs replay.** Resetting to `earliest` is an administrative operation on a group's
  checkpoint with immediate side effects on live traffic; a replay group is an isolated consumer
  with its own offsets and a quarantined sink. One is a config change, the other is a data
  operation.
- **Partition count increase vs repartitioning.** Increasing the count changes `key mod P` for
  almost every key, so the ordering contract that existed before the change does not survive it.
  Any migration plan must state how per-key order is preserved across the cutover.
- **Zero lag vs correctness.** Lag is computed from Kafka metadata only. A consumer that commits
  before writing to its sink shows zero lag and loses records; a consumer that writes before
  committing shows zero lag and may duplicate.
- **Broker timestamp vs event timestamp.** `LogAppendTime` reflects when the broker accepted the
  record; the producer's event time reflects when the business fact happened. A backlog metric
  built on one and an SLA defined on the other will disagree during producer outages and clock skew.

**Takeaway:** A number called "lag" is only interpretable together with its end boundary, its group, its unit and its timestamp.

## 4. Core Formulas with Full Derivation

Notation used throughout: partition index `i ∈ {0..P−1}`; arrival rate `a_i` [records/s]; durable
service rate `s_i` [records/s]; backlog `B_i` [records]; offset lag `L_i` [offsets]; end boundary
`E_i` [offset]; committed offset `C_i` [offset]; position `P_i` [offset]; time `t` [s]. A rate and a
backlog are never mixed in one expression without converting through a declared per-record cost or
a declared record-per-offset ratio.

### 4.1 Simplest / Base Case

**Validity before use.** One partition; constant arrival and service rates; no loss, no retries that
never complete, no rebalances; each offset corresponds to one readable record; the consumer is
keeping up in the sense that service capacity is actually available whenever work is waiting.

**Derivation.** The partition is a single-server queue observed in counts rather than seconds.
Conservation of records over an interval `Δt` gives the change in backlog as arrivals minus
completions:

```text
B(t + dt) - B(t) = a * dt - s * dt  =  (a - s) * dt
```

Integrating from `t = 0` to `t` with initial backlog `B(0)` and clamping at zero, because a queue
cannot hold negative work and an idle consumer cannot bank service credit for future arrivals:

```text
[Q-01] FORMULA: B(t) = max(0, B(0) + (a - s) * t)
[2] WHERE: B [records]; a, s [records/s]; t [s]; both rates measured over the same window.
[3] CONSTRAINT: single partition, constant rates, no loss, no unfinishable retries, B(0) >= 0.
[4] NUMERICAL EXAMPLE: B(0)=0, a=120, s=100, t=60 s
                   => B = max(0, 0 + (120 - 100) * 60) = 20 * 60 = 1,200 records.
[5] INTERPRETATION: at a 20 records/s deficit the partition accumulates 1,200 records in one
                   minute; the number doubles every further minute the deficit persists.
```

The `max(0, ·)` clamp is not cosmetic. Without it, a consumer with `s = 400` and `a = 100` would
appear to accumulate −300 records per second of "credit", and a later burst would appear free.
Real systems cannot bank idle capacity beyond the in-flight buffer.

**Lag identity.** The observable quantity is not `B` but the offset distance. Relating them requires
one more premise: that one business record occupies one offset over the measured range.

```text
[Q-02] FORMULA: L_i(t) = E_i(t) - C_i(t)
[2] WHERE: L_i [offsets]; E_i [offset] the declared end boundary (LEO, HW or LSO);
           C_i [offset] the group's committed next offset for partition i.
[3] CONSTRAINT: same observation instant for both terms; same partition; one declared boundary type.
[4] NUMERICAL EXAMPLE: E_i = 9,400,000, C_i = 7,150,000 => L_i = 2,250,000 offsets.
[5] INTERPRETATION: if the boundary is LEO at a single instant, this is the offset work outstanding;
                   it is not yet a record count and not yet a time.
```

**Converting lag to time.** Given the arrival rate, an offset backlog converts to a time-behind
estimate. This is the quantity that actually matches a freshness objective, and it is the one most
dashboards omit.

```text
[Q-03] FORMULA: T_behind = L_i / a_i
[2] WHERE: T_behind [s]; L_i [offsets]; a_i [records/s] measured over the same recent window.
[3] CONSTRAINT: a_i > 0; offsets and records are locally 1:1; arrival rate stable over T_behind;
                the partition's processing cost is roughly stationary.
[4] NUMERICAL EXAMPLE: L_i = 2,250,000, a_i = 900 records/s => T_behind = 2,500 s ≈ 41.7 min.
[5] INTERPRETATION: records entering now will be visible roughly 42 minutes from now if nothing
                   changes; this is a freshness statement, unlike the raw offset count.
```

**Boundary and counterexample.** If compaction removed 30% of the offsets in that range, the record
count is smaller than 2,250,000 and `T_behind` computed from offsets overstates the record backlog
while still correctly describing the *offset* distance. The two answers serve different questions:
`L/a` bounds how long the log's tail lags the head, and a record-count reconciliation from an effect
ledger bounds how much work remains. Report which one you computed.

### 4.2 General Case

**Validity before use.** `P` independent partitions; rates constant over a declared interval; the
group's assignment is stable; each member's service rate is measured while it owned the partitions
it is being credited for. Shared resources (CPU, network, connection pools, sink capacity) mean the
per-partition rates are *not* additive in reality, so the aggregate below is an accounting identity
for the backlog, not a model of contention.

**Derivation.** Backlog is a sum over partitions, and the clamp must be applied *inside* the sum
because an idle partition cannot lend its spare capacity to a busy one. Summing the clamps first and
clamping the sum is the single most common modelling error in this topic:

```text
[Q-04] FORMULA: B_total(t) = SUM_{i=0}^{P-1} max(0, B_i(0) + (a_i - s_i) * t)
[2] WHERE: B_total [records]; a_i, s_i [records/s]; P [count]; t [s].
[3] CONSTRAINT: no cross-partition work stealing; per-partition rates measured under concurrent load.
[4] NUMERICAL EXAMPLE: two empty partitions, (a,s) = (120,100) and (20,100), t = 60 s
                   => max(0, 20*60) + max(0, (-80)*60) = 1,200 + 0 = 1,200 records.
                   The wrong form max(0, (120+20-200)*60) = max(0, -3,600) = 0.
[5] INTERPRETATION: total nominal capacity of 200/s exceeds total arrival of 140/s while one
                   partition still grows; aggregate headroom is not transferable.
```

**The two aggregate metrics that matter.** Because the sum can hide a failing partition, operational
alerting must use the worst partition, and the *distribution* rather than the total:

```text
[Q-05] FORMULA: L_max(t) = max_i (E_i(t) - C_i(t))
[2] WHERE: L_max [offsets]; the maximum over partitions actually assigned to the group.
[3] CONSTRAINT: every partition's offsets observed within the same scrape window;
                partitions with no committed offset must be handled explicitly, not as zero.
[4] NUMERICAL EXAMPLE: four partitions with lag 900,000 / 40,000 / 35,000 / 25,000
                   => SUM = 1,000,000 but L_max = 900,000: one partition is 90% of the total.
[5] INTERPRETATION: an alert on SUM would fire late and name the wrong cause; L_max identifies
                   that the incident is a single-partition problem.
```

**Skew quantification.** To distinguish "one partition is hot" from "all partitions are rising",
normalise the per-partition rates. The coefficient of variation is scale-free and comparable across
topics, which makes it usable as a standing health metric rather than a per-incident judgement call.

```text
[Q-06] FORMULA: CV_a = sd(a_0..a_{P-1}) / mean(a_0..a_{P-1});
                skew_ratio = a_max / mean(a)
[2] WHERE: CV_a [dimensionless]; a_i [records/s]; P >= 2.
[3] CONSTRAINT: rates measured over the same window; a null-key topic may legitimately be skewed
                by the sticky partitioner and that skew is not a defect.
[4] NUMERICAL EXAMPLE: a = [900, 300, 300, 300], mean = 450,
                   sd = sqrt((450^2 + 3*150^2)/4) = sqrt((202,500 + 67,500)/4) = sqrt(67,500) ≈ 259.8,
                   CV_a ≈ 0.577; skew_ratio = 900/450 = 2.0.
[5] INTERPRETATION: perfectly balanced traffic gives CV_a = 0; the value 0.577 with a 2× ratio
                   says the imbalance is structural in the key distribution, not random noise.
```

**Parallelism and Little's Law.** Memory, connection pools and replay exposure are governed by the
number of records concurrently in flight, not by the fetch size alone. Little's Law ties the three
observable quantities together and gives the only defensible way to bound in-flight work.

```text
[Q-07] FORMULA: K = a * W  =>  W = K / a
[2] WHERE: K [records] average in-flight; a [records/s] completion rate; W [s] mean time in system.
[3] CONSTRAINT: steady state over the measurement window; a single queue in front of the sink;
                K is the average, so tail behaviour needs a percentile, not this identity.
[4] NUMERICAL EXAMPLE: a = 600 records/s and W = 40 ms => K = 600 * 0.040 = 24 records.
                   At W = 900 ms (a degrading sink) => K = 600 * 0.900 = 540 records.
[5] INTERPRETATION: a sink slowdown from 40 ms to 900 ms multiplies in-flight work by 22.5 at the
                   same arrival rate; an unbounded async pool converts this into memory growth.
```

**Service-rate decomposition.** To decide *which* resource to fix, decompose the per-record service
cost. If one term dominates, the intervention is decided by arithmetic rather than by preference.

```text
[Q-08] FORMULA: s = 1 / (c_cpu + R * c_io)      (per member, c_io serialised into 1 of R slots)
[2] WHERE: s [records/s]; c_cpu [s/record] CPU-bound cost; c_io [s/record] blocking I/O cost;
           R [count] concurrent I/O slots per member.
[3] CONSTRAINT: costs measured on the deployed hardware and payload mix; CPU and I/O overlap as
                modelled; no lock contention inside the member.
[4] NUMERICAL EXAMPLE: c_cpu = 0.6 ms, c_io = 24 ms, R = 1
                   => s = 1/(0.0006 + 0.024) = 40.65 records/s per member.
                   With R = 16 => s = 1/(0.0006 + 0.0015) = 476.2 records/s per member.
[5] INTERPRETATION: the first configuration saturates at about 41 records/s per member regardless
                   of CPU, so adding members is a legitimate fix; the second is now CPU-bound and
                   further members help only until CPU per host is exhausted.
```

### 4.3 Key Special Cases / Variants

| Variant | What changes | Precondition | Consequence for lag | Misuse |
|---|---|---|---|---|
| Sequential handler | `R = 1`, order by construction | Per-key order required | Throughput bounded by `1/(c_cpu + c_io)` | Raising fetch size to "go faster" |
| Async handler + frontier | `R > 1` with completion tracking | Effects independent per key, or per-key serialisation | Throughput scales until CPU or sink saturates | Committing `position` instead of the safe frontier |
| Batch commit | Checkpoint every `n` records or `τ` seconds | Replay tolerated up to the batch window | Lag metric lags reality by one commit | Committing on a timer without checking completion |
| Transactional read-process-write | Offsets written inside a Kafka transaction | Sink is Kafka; consumers read `read_committed` | LSO may pin the readable boundary | Assuming it covers a non-Kafka database |
| Cooperative-sticky assignment | Only moved partitions revoke | Client and broker versions support it | Smaller catch-up spikes on rebalance | Assuming zero interruption |
| Isolated replay group | Separate group id and sink namespace | Retained history and idempotent effects | Live lag unaffected; replay competes for broker I/O | Sharing the production group id |

**Variant 1 — sequential handler.** With `R = 1` the member processes one record at a time, so the
service rate is exactly the reciprocal of total per-record cost. The safe commit position after each
successful effect is `offset + 1`, which is why sequential consumers rarely lose data and frequently
cannot keep up.

```text
[Q-09] FORMULA: s_seq = 1 / c_total,  c_total = c_cpu + c_io + c_sink_wait
[2] WHERE: s_seq [records/s]; c_* [s/record].
[3] CONSTRAINT: one record in flight; no batching of the sink call; per-record cost stationary.
[4] NUMERICAL EXAMPLE: c_cpu=0.8 ms, c_io=0.4 ms, c_sink_wait=28.8 ms
                   => c_total = 30.0 ms => s_seq = 33.33 records/s.
[5] INTERPRETATION: 96% of the cycle is waiting, so concurrency is the only lever with real headroom;
                   optimizing the CPU path would change the throughput by under 5%.
```

**Variant 2 — async with a completion frontier.** Here order is preserved for *effects* only where
required, and the commit position is the longest contiguous completed prefix. The throughput gain is
the standard concurrency gain, and the cost is the memory and bookkeeping of tracking each in-flight
offset's completion state.

```text
[Q-10] FORMULA: s_async = min(K_slots / c_sink_wait, 1 / c_cpu, s_sink_capacity)
[2] WHERE: s_async [records/s]; K_slots [count] bounded concurrent calls per member;
           s_sink_capacity [records/s] the sink's own sustainable ceiling.
[3] CONSTRAINT: effects are reorderable or per-key serialised; K_slots is a hard bound, not a target.
[4] NUMERICAL EXAMPLE: K_slots=16, c_sink_wait=28.8 ms => 16/0.0288 = 555.6 records/s;
                   c_cpu=0.8 ms => 1,250 records/s; suppose s_sink_capacity = 400 records/s
                   => s_async = min(555.6, 1,250, 400) = 400 records/s.
[5] INTERPRETATION: past 400 records/s the binding constraint is the sink, so raising K_slots
                   converts throughput gains into queueing delay and errors at the database.
```

**Variant 3 — batch commit and the replay window.** Committing every record costs coordinator
round-trips; committing every `τ` seconds or every `n` records trades throughput for a bounded
amount of reprocessing after a crash. The exposure is bounded and computable, which is why it should
be a written budget rather than an accident of configuration.

```text
[Q-11] FORMULA: E[replay] = a * τ / 2      (uniform crash time within the commit interval)
                max replay = a * τ
[2] WHERE: E[replay] [records]; a [records/s]; τ [s] commit interval.
[3] CONSTRAINT: steady arrival, crash uniformly distributed in the interval, effects idempotent.
[4] NUMERICAL EXAMPLE: a = 400 records/s, τ = 5 s
                   => E[replay] = 400 * 5 / 2 = 1,000 records; worst case 2,000 records.
[5] INTERPRETATION: the sink must tolerate 1,000 duplicate attempts on average per crash, so the
                   deduplication horizon and the idempotency contract are sized from τ, not guessed.
```

**Variant 4 — rebalance interruption.** An eager rebalance revokes every partition, so the group's
throughput drops to zero for the rebalance duration, plus a catch-up period while new owners warm
caches and state. The cost per rebalance is small and the cost of a rebalance *loop* is unbounded,
which is why the count matters more than the individual duration.

```text
[Q-12] FORMULA: availability = 1 - (f * t_rb) / T_window
[2] WHERE: f [count] rebalances per window; t_rb [s] each including catch-up; T_window [s].
[3] CONSTRAINT: uniform rebalance cost; catch-up included in t_rb; no overlapping rebalances.
[4] NUMERICAL EXAMPLE: f = 12, t_rb = 25 s, T_window = 3,600 s
                   => lost fraction = 300/3,600 = 0.0833 => availability 91.7%.
[5] INTERPRETATION: a dozen ordinary-looking restarts consume 8% of the hour's processing
                   capacity, which alone can turn a marginal surplus into a growing backlog.
```

**Variant 5 — transactional read-process-write.** When both the record effects and the offset commit
live in Kafka, a transaction makes them atomic and the consumer reads only committed data. This is
genuinely strong, and genuinely bounded: it does nothing for an external database, an HTTP call, or
a cache, all of which remain at-least-once with respect to the consumer.

```text
[Q-13] FORMULA: visible_end = min(LSO, LEO)   for isolation.level = read_committed
[2] WHERE: visible_end, LSO, LEO [offsets]; LSO is the first offset of the earliest open transaction.
[3] CONSTRAINT: transactional producer in use; consumer configured read_committed;
                transaction timeouts configured at both producer and broker.
[4] NUMERICAL EXAMPLE: LEO = 5,000,000 and an open transaction began at 4,998,000
                   => visible_end = 4,998,000; lag computed against LEO overstates *readable* lag
                   by 2,000 offsets.
[5] INTERPRETATION: an idle or stuck producer transaction suppresses the readable boundary with no
                   consumer misbehaviour at all; the group looks lagged while it is merely blocked.
```

**Variant 6 — isolated replay group.** Replay must never share the live group id, and must never
share the live sink namespace without a mode flag. The sizing question is whether replay plus live
work fits inside the broker's read bandwidth and the sink's write capacity simultaneously.

```text
[Q-14] FORMULA: T_replay = B_replay / (s_replay - a_replay_in_window)
[2] WHERE: T_replay [s]; B_replay [records] from the earliest required offset;
           s_replay [records/s] allocated to the replay group; a_replay_in_window [records/s]
           of new records the replay group must also absorb if it tails to the end.
[3] CONSTRAINT: replay group capacity reserved rather than borrowed from live members;
                retained history covers the whole window.
[4] NUMERICAL EXAMPLE: B_replay = 40,000,000, s_replay = 1,500, a_replay_in_window = 900
                   => T_replay = 40,000,000 / 600 = 66,667 s ≈ 18.5 h.
[5] INTERPRETATION: a naive estimate of 40,000,000/1,500 = 26,667 s (7.4 h) is wrong by 2.5×
                   because it forgets that new records keep arriving during the rebuild.
```

### 4.4 Asymptotic / Limiting Behaviour and Counterexample

**Limit 1 — the no-recovery boundary.** From `Q-01`, the sign of `a − s` decides everything. If
`a = s` the backlog is frozen at its current value and never clears; if `a > s` it grows without
bound; only `a < s` yields a finite drain. This is a statement about the *binding* partition, not the
topic average, and it is the first thing to compute in any incident.

```text
[Q-15] FORMULA: T_drain = B_0 / (s - a)   for s > a
[2] WHERE: T_drain [s]; B_0 [records] at intervention time; a, s [records/s] during recovery.
[3] CONSTRAINT: s > a strictly; both rates constant over the recovery interval; source retained
                for the whole interval; sink sustains s without secondary failure.
[4] NUMERICAL EXAMPLE: B_0 = 360,000, a = 900, s = 1,200
                   => T_drain = 360,000 / 300 = 1,200 s = 20 min.
                   With s = 1,000 => 3,600 s = 1 h. With s = 900 => undefined (no finite drain).
[5] INTERPRETATION: the estimate is acutely sensitive to the surplus; halving the surplus doubles
                   the time, and a 25% surplus shortfall removes the recovery entirely.
```

Independent arithmetic check for the example: over 1,200 s the consumer completes
`1,200 × 1,200 = 1,440,000` records and receives `1,200 × 900 = 1,080,000`, a difference of
360,000, matching `B_0`. The common wrong computation divides by gross service
(`360,000 / 1,200 = 300 s`) and omits the arrivals that continue during recovery, understating the
answer by 4× in this case.

**Limit 2 — the parallelism ceiling.** Group parallelism is capped by partition count, and the
per-member benefit stops at one partition each:

```text
[Q-16] FORMULA: active_members_effective = min(N, P);  utilisation = min(N, P) / N
[2] WHERE: N [count] group members; P [count] partitions.
[3] CONSTRAINT: each partition is owned by exactly one member in the group; no standby-only members;
                P is not changing.
[4] NUMERICAL EXAMPLE: P = 12, N = 20 => effective capacity = 12 members, utilisation = 60%;
                   8 members sit idle. P = 12, N = 6 => effective = 6 (each owns 2 partitions).
[5] INTERPRETATION: if a single partition is the bottleneck, the ceiling is 1 — no value of N helps.
```

**Limit 3 — the hot-key wall.** A single key cannot be split without changing the ordering contract,
so the achievable service rate for that key's records is bounded by one partition's rate. The only
legitimate escapes are (a) make per-record work cheaper, (b) salt the key and restore order
downstream, or (c) change the business contract.

```text
[Q-17] FORMULA: feasible  <=>  a_key <= s_partition / (1 + headroom)
[2] WHERE: a_key [records/s] for the dominant key; s_partition [records/s] achievable on one
           partition; headroom [-] fraction reserved for bursts.
[3] CONSTRAINT: strict per-key order required; key cannot be split.
[4] NUMERICAL EXAMPLE: s_partition = 600, headroom = 0.25 => feasible ceiling = 480 records/s.
                   a_key = 900 => infeasible by 420 records/s; the backlog grows at ≥420/s.
[5] INTERPRETATION: with 900 records/s on one key the design is infeasible as stated; adding
                   members changes nothing and only a key or contract change can close the gap.
```

**Counterexample — the average that hides the incident.** Four partitions at
`a = [100, 100, 100, 800]` records/s and `s = [600, 600, 600, 600]`. Mean arrival is 275/s against
600/s service, so an average-based alarm reads "comfortable". Partition 3 grows at 200 records/s,
and after one hour (3,600 s) holds 720,000 records while the other three hold zero. Any capacity
decision made from the mean is wrong in both directions: it neither adds the capacity that would
help nor identifies that capacity cannot help.

**Counterexample — the lag that never clears because of a control record.** A partition can show
non-zero lag that no business record explains: an open transaction pins the LSO (`Q-13`), or a
compacted-away segment leaves an offset gap. In both cases the consumer is doing nothing wrong, and
the correct action is to examine the transaction or the offset range, not to raise concurrency.

**Asymptotic behaviour of the commit interval.** As `τ → 0` the replay exposure (`Q-11`) approaches
zero and coordinator traffic per record approaches one commit per record, which will saturate the
coordinator long before it improves durability. As `τ → ∞` commit traffic vanishes and replay
exposure grows linearly with `a`. The usable range is set by the sink's duplicate tolerance and the
coordinator's throughput, both measurable.

**Takeaway:** There is no recovery without a strict per-partition surplus, and no amount of group members bypasses the single-partition ceiling.
## 5. Worked Numerical Example

The example is one continuous incident, carried from detection through intervention, because the
value of this topic is in the sequencing rather than in any single calculation. All numbers are
**ILLUSTRATIVE**; the arithmetic is checked, the workload is invented.

### 5.1 The observation

A payments-event pipeline. Topic `payments.events`, `P = 16` partitions, `N = 6` members, three
availability zones. The group `payments-writer` performs an idempotent UPSERT into a relational
ledger keyed by `event_id`, in batches of up to 200 records, committing the safe frontier after each
batch. The sink sustains a measured 2,400 records/s in this configuration, and the ledger records
which partition each applied row came from, so per-partition effect rates are observable rather
than inferred.

Observed at `T = 14:05:00` on a single aligned scrape:

| Partition | LEO | Committed `C` | Lag | Ingress `a` | Applied `s` |
|---|---:|---:|---:|---:|---:|
| 0 | 9,412,000 | 9,190,000 | 222,000 | 2,404 | 88.8 |
| 1 | 9,388,000 | 9,201,000 | 187,000 | 2,398 | 88.7 |
| 2 | 9,405,000 | 9,160,000 | 245,000 | 2,402 | 88.9 |
| … (12 more, similar to 0–2) | | | | | |
| 15 | 9,420,000 | 6,510,000 | **2,910,000** | **12,600** | **1,068** |

Totals: `ΣL = 6,540,000` offsets; `L_max = 2,910,000` on partition 15; `Σa = 48,650` records/s;
`Σs = 2,400` records/s. Reconcile both totals explicitly before using either:

```text
Σa: 15 partitions at mean 2,403.3 records/s = 36,050, plus 12,600 = 48,650 ✓
Σs: 15 partitions at mean 88.8 records/s    = 1,332,  plus 1,068  = 2,400  ✓
```

Both sums land, so the two columns describe the same population over the same one-minute window. If
they did not, the correct next step would be to re-scrape both on one window rather than to compute
with either: two numbers labelled "throughput" over different windows produce a conclusion that
neither supports.

Note the second sum's distribution: partition 15 consumes `1,068 / 2,400 = 44.5%` of the sink's
delivered capacity while carrying `12,600 / 48,650 = 25.9%` of arrivals. That asymmetry is the owner
giving the hot lane more of its share, which is what any queue-aware implementation naturally does —
and it is still nowhere near enough.

### 5.2 Step 1 — is there any drain at all?

Before any percentage or estimate, compute the sign of the deficit. Partition 15 receives `12,600`
records/s and its owner applies about `1,068` records/s of ledger rows attributable to it. The
deficit is `12,600 − 1,068 = 11,532` records/s. The answer to "is there any drain" is **no**: this
partition grows by more than 11,500 records every second, and no member count fixes it while the key
distribution stays as it is.

Cross-check the aggregate. `Σa = 48,650` against `Σs = 2,400` records/s is roughly a 20×
oversubscription. Even the healthy partitions show it: partition 0 receives `2,404` records/s and
applies `88.8`, a deficit of `2,315.2` records/s. So the group is not "one bad partition plus fifteen
fine ones"; it is one badly skewed partition *on top of* a system whose total capacity is far below
total arrival. Both defects must be named, because fixing only the skew still leaves a capacity gap
and fixing only the capacity still leaves the skew.

The 20× figure is worth decomposing because it is not a rounding artefact. Total group capacity is
`2,400` records/s delivered by six members, so `400` records/s per member on average. Total arrivals
are `48,650`, so sustaining them would need `48,650 / 400 = 121.6` members — or, more practically, a
fundamental change in either per-member efficiency or the sink's ceiling. A system operating at 5% of
required capacity is not experiencing a temporary anomaly; whatever produced the arrival rate is a
change of regime that the consumer tier never absorbed.

This is the point where an incident review usually goes wrong. A reviewer who computes only
`Σs > Σa` on a stale window concludes the system is healthy; one who computes only `a_15 > s_15`
concludes the problem is a hot key and re-keys it; neither sees that the two defects are additive.
Compute both, state both, and rank them by which one binds first.

### 5.3 Step 2 — how long until the retention boundary is crossed?

The topic retains 24 h. Partition 15's oldest outstanding offset is at `C = 6,510,000`. Retention
deletes by segment age, so the question is how much history remains between `C` and the log start
offset. Suppose the observed log start offset for partition 15 is `7,880,000`, which is the first
offset of the oldest retained segment.

That is a problem already: `C = 6,510,000 < log start = 7,880,000`. The committed offset has fallen
*outside* the retained range. The group cannot resume from its checkpoint without an explicit reset
decision, and any records between the two numbers are gone from this topic. This is the
out-of-range condition, and the honest framing is a data-loss assessment for the `1,370,000` offsets
between `C` and the log start — not a lag calculation. Note the arithmetic: a partition can be
catastrophically behind while the group-level lag metric looks dominated by a *different* partition.

For a partition still inside retention, compute the headroom. Take partition 0 with `C = 9,190,000`,
log start `8,200,000`, and a record density of one offset per record:

```text
consumable offsets ahead of checkpoint = 9,190,000 - 8,200,000 = 990,000 offsets
retention at current ingress           = 990,000 / 2,404 records/s = 411.8 s ≈ 6.9 min
```

So even the partitions that look merely "a bit behind" lose their checkpoint in under seven minutes
unless the group advances or deletion is slowed. Two clocks are now in play — the drain clock and
the retention clock — and here the retention clock is roughly an order of magnitude shorter.

### 5.4 Step 3 — bounded intervention and its arithmetic

The intervention chosen must be reversible and must not sacrifice correctness. Three candidates are
evaluated with the same arithmetic, because "we did something" is not a diagnosis.

**Candidate A — add members from 6 to 16.** Effective parallelism `min(N, P) = min(16, 16) = 16`,
a `2.67×` increase in partition-owner slots. Each member would then own one partition instead of
`2.667`, so the per-partition service share rises from `1,608 / 2.667 = 602.9` to `1,608 / 1 = 1,608`
records/s — a genuine `2.67×` improvement for every partition including 15. That is worth stating
precisely, because it shows that member count *does* help here, just not nearly enough: after the
change partition 15 has `12,600 − 1,608 = 10,992` records/s of deficit and every other partition has
`2,404 − 1,608 = 796` records/s of deficit. Group capacity is `16 × 1,608 = 25,728` against
`48,650` arrivals, so the group deficit falls from `39,002` to `22,922` records/s. **Insufficient
alone, and worth doing only alongside B**, because at six members the sink is also being asked to
serve 20× its delivered rate and adding members while leaving that unchanged merely spreads the
shortfall more evenly.

**Candidate B — reduce per-record cost on the hot path.** Suppose profiling shows the UPSERT cost is
dominated by an unbatched per-record round trip of 24 ms, and batching 200 records per statement
reduces the per-record sink cost to about 0.35 ms. Recompute the member service rate with
`Q-08` using `c_cpu = 0.6 ms`, `c_io = 0.35 ms`, `R = 16`:

```text
s = 1 / (0.0006 + 0.35/1000 / 16)
  = 1 / (0.0006 + 0.000021875)
  = 1 / 0.000621875
  ≈ 1,608 records/s per member
```

With 6 members: `6 × 1,608 ≈ 9,648` records/s of group capacity, up from 2,400/s — a 4× improvement
with no new hardware. Against `Σa = 48,650` it is still not enough on its own, and partition 15's
share of a member serving 2.667 partitions is `602.9`/s against `12,600`/s. **Necessary but not
sufficient**, and it is the cheapest of the three, so it ships first.

**Candidate C — split the dominant key.** Identify which key produces partition 15's traffic. Suppose
one merchant account, `merchant:7741`, produces 11,800 of the 12,600 records/s because a downstream
batch job emits per-line-item events under the merchant key. Salting that key into 16 sub-keys
`merchant:7741#s` for `s ∈ 0..15` spreads it across 16 partitions at `11,800/16 = 737.5` records/s
each. Ordering is preserved per line item only if the downstream ledger keys on `event_id` and
applies versioned conditionals; the merchant-level *total* order is sacrificed and must be
reconciled by a separate aggregation. That is a business-contract change and must be agreed by the
owner of the merchant reporting path, not decided by the streaming team alone.

Apply the feasibility bound `Q-17` honestly rather than hopefully. With six members the per-partition
service share is `602.9` records/s, so the budgeted rate at 25% headroom is `602.9 / 1.25 = 482.3`
records/s per partition. Under that ceiling, supporting `11,800 / 482.3 = 24.5` sub-keys would be
required before the salted traffic fits — more sub-keys than the topic has partitions. Salting
therefore *cannot* fix this on its own: it distributes a workload that the group cannot carry at all.
That is exactly the finding the arithmetic should produce, and it is why the analysis must compute
the capacity requirement rather than stop at "add sub-keys until it fits".

### 5.5 Step 4 — the drain forecast, with its assumptions

Now recompute with Candidates B and C applied, and follow the arithmetic rather than the narrative.
Candidate C salts `merchant:7741` into 16 sub-keys, so its `11,800` records/s spread over 16
partitions at `11,800 / 16 = 737.5` records/s each. Everything else stays: the remaining
`48,650 − 11,800 = 36,850` records/s spread over the same 16 partitions at
`36,850 / 16 = 2,303.125` records/s each.

```text
per-partition arrival after salting = 2,303.125 + 737.5   = 3,040.625 records/s
group arrival (16 partitions)       = 16 * 3,040.625     = 48,650 ✓ (salting preserves total)
group service after B (6 members)   = 6 * 1,608          = 9,648 records/s
group deficit                       = 48,650 - 9,648     = 39,002 records/s
per-member owned partitions         = 16 / 6             = 2.6667
per-partition service share         = 1,608 / 2.6667     = 602.99 records/s
per-partition deficit               = 3,040.625 - 602.99 = 2,437.6 records/s
group deficit (cross-check)         = 16 * 2,437.6       = 39,002 records/s ✓
```

The `16/6` divisor is itself worth noting: because it is not an integer, a six-member group cannot
give every partition an equal share of a member's capacity. Sixteen partitions across six members
means four members own three partitions and two own two. A partition owned by a three-partition
member receives `1,608 / 3 = 536` records/s of that member's attention; one owned by a two-partition
member receives `1,608 / 2 = 804`. The `602.99` above is the average of that uneven distribution.

The worst case is therefore worse than the average suggests: a partition on a three-partition member
has deficit `3,040.625 − 536 = 2,504.6` records/s, which is `(2,504.6 − 2,437.6) / 2,437.6 = 2.7%`
larger than the average deficit. Report the worst case, because a capacity decision made from the
average leaves the three-partition members structurally short before any burst arrives. The general
point: `P mod N ≠ 0` produces a per-member imbalance of up to `ceil(P/N) / (P/N)`, and only matching
`N` to a divisor of `P` removes it.

This is the step where the diagnosis changes, and it changes by arithmetic rather than by opinion.
Before salting, the problem presented as one partition at `12,600` against a share of about `603` —
a deficit of roughly `12,000` on one lane, which invites a re-keying fix and nothing else. After
salting, `CV_a` collapses toward zero and every partition sits at `3,040.5` against `602.9`. Salting
removed the *skew* and revealed a *capacity gap* of `39,002` records/s that had been there the whole
time, masked because a skewed metric made the hot partition look like the only problem.

The distinction matters because the two defects have different owners and different costs. The skew
was a producer-side keying decision, cheap to change and fully within the streaming team's control.
The capacity gap is a sink-throughput problem: closing it requires either a large increase in
per-member efficiency, a large increase in partition count with a migration, or a reduction in
upstream volume — all cross-team decisions with their own review and their own timeline.

The required remediation is now computable rather than debatable. To hold `48,650` records/s with 25%
headroom, the group needs `48,650 × 1.25 = 60,812.5` records/s of durable service:

```text
members needed at 1,608 records/s = ceil(60,812.5 / 1,608) = ceil(37.8) = 38 members
partitions available              = 16  -> min(38, 16) = 16 effective
```

So 38 members would still cap at 16 effective members and `16 × 1,608 = 25,728` records/s, which is
still only 53% of the `48,650` arrival rate and 42% of the headroomed requirement of `60,812.5`. The
`min(N, P)` ceiling binds before the sink does. The conclusion: either `P` must grow past 38 (with a
key-remap migration and a per-entity ordering plan), or per-member service must rise to
`60,812.5 / 16 = 3,801` records/s, or the arrival rate must fall. Nothing about consumer
configuration alone closes this gap, and stating it that way is the deliverable of the analysis.

Model the drain for a *future* state, after remediation lands, so the numbers are internally
consistent. Suppose the upstream team re-batches the per-line-item emission so that the hot key's
volume falls and total arrivals become `Σa = 3,040` records/s (`190` per partition across 16), and
sink capacity is provisioned so each member delivers `3,000` records/s while owning `16 / 6 = 2.667`
partitions — a per-partition share of `3,000 / 2.667 = 1,125` records/s. For a partition holding
`B_0 = 222,000` records:

```text
net drain per partition = 1,125 - 190   = 935 records/s
T_drain                 = 222,000 / 935 = 237.4 s ≈ 4.0 min
group-level check       = 16 * 935      = 14,960 records/s net drain
```

Independent check: in `237.4` s the group completes `16 × 1,125 × 237.4 = 4,273,200` records and
receives `16 × 190 × 237.4 = 721,696`; the difference is `3,551,504`. But the actual backlog is
`16 × 222,000 = 3,552,000`, and the two agree to within `496` records — the residue of rounding
`237.4` from `237.43…`. Use the unrounded value `222,000/935 = 237.433…` s and the identity is exact
by construction, since `T` was defined by that equation.

State the assumptions of that forecast plainly, because each one is a way for it to be wrong: rates
constant over the whole interval; the sink does not degrade as it sustains `3,000`/s per member; no
rebalance during the window; retention covers the entire drain; and no retries inflate the effective
arrival rate. Sensitivity: at `3,000` records/s per member the per-partition share is
`3,000 × 6 / 16 = 1,125` and `T_drain = 237.4` s (the base case). At `2,000` records/s per member the
share is `750`, net drain `750 − 190 = 560`, and `T_drain` becomes `222,000 / 560 = 396.4` s — a 67%
increase. At `1,125` records/s per member the share is `421.9`, net drain `231.9`, and `T_drain`
becomes `222,000 / 231.875 = 957.6` s. The estimate is not robust to the service figure, so the
service figure must be measured under load rather than taken from a vendor ceiling.

### 5.6 Step 5 — committing safely through out-of-order completions

The batch handler submits 200 UPSERTs concurrently. Completion order is not offset order. At one
instant the in-flight set for partition 0 is:

```text
completed:  9,190,000 .. 9,190,140    and    9,190,161 .. 9,190,190
pending:    9,190,141 .. 9,190,160    (20 records, one shard of the batch is slow)
```

The safe committed next offset is **9,190,141**, not 9,190,191. Committing 9,190,191 would skip
offsets 141–160, which have no durable effect, and a crash would lose them permanently because the
group would resume past them. Once the slow 20 complete, the contiguous prefix is 9,190,000–9,190,190
and the next commit is **9,190,191**.

Now the replay exposure. From `Q-11`, `E[replay] = a·τ/2`, and for a batch-triggered commit the
interval is the time to fill one batch, `τ = n/a`. Substituting cancels the rate:

```text
τ         = n / a = 200 / 2,404 = 0.0832 s     (partition 0 at its observed rate)
E[replay] = a * τ / 2 = 2,404 * 0.0832 / 2 = 100.0 records
max       = a * τ = 2,404 * 0.0832 = 200 records
```

The cancellation is the useful result: with batch-triggered commits, expected replay is `n/2` and
worst case is `n` records **regardless of the arrival rate**. Doubling throughput does not double the
duplicate exposure; doubling the batch size does. That is a far more actionable statement than "replay
exposure depends on rates", and it is why the batch size is the parameter to govern here.

The ledger's `ON CONFLICT (event_id) DO NOTHING` absorbs up to 200 re-applications per restart. The
deduplication retention must exceed the supported replay horizon of the *replay group*, not merely
the live group's — if replay can reach back 30 days, the `event_id` record must live at least that
long, or the 200-record exposure repeats as a duplicate effect.

### 5.7 Step 6 — what the sink's cost model implies for group sizing

Size from the sink and the partition count together, because `min(N, P)` is a conjunction and either
term can bind. Using the example's actual arrival rate `Σa = 48,650` records/s and a per-member
service of `3,000` records/s (a plausible post-remediation figure, not a measured one):

```text
required service at 25% headroom = 48,650 * 1.25 = 60,812.5 records/s
members required                 = ceil(60,812.5 / 3,000) = ceil(20.3) = 21 members
partitions available             = 16
partition ceiling on members     = min(21, 16) = 16 effective members
deliverable at that ceiling      = 16 * 3,000 = 48,000 records/s
still short by                   = 60,812.5 - 48,000 = 12,812.5 records/s
```

So even a 25% improvement in per-member service leaves the design infeasible at `P = 16`: the group
can deliver `48,000` records/s against a requirement of `60,812.5`. Two levers remain, and the
arithmetic quantifies each. Either raise `P`:

```text
P required = ceil(60,812.5 / 3,000) = ceil(20.3) = 21 partitions
```

or raise per-member service to `60,812.5 / 16 = 3,801` records/s, a further 27% improvement on top of
the 25% already assumed. The first is a key-remap migration; the second is more engineering on the
sink path. Both are decisions, not configurations, and the number says which one is cheaper to
evaluate first.

Note also the assignment granularity problem even when `P` is sufficient. With `P = 21` and `N = 21`
each member owns exactly one partition, which is the ideal. With `P = 16` and `N = 6`, each member owns
`16/6 = 2.667`, meaning four members own 3 partitions and two own 2 — a 50% difference in assigned
count between members. The assignment strategy will produce that split, and no tuning removes it;
only matching `N` to a divisor of `P` (or accepting the imbalance) does.

**Takeaway:** Compute the deficit sign before anything else, then check the retention clock, then choose an intervention whose arithmetic you can defend — and notice when the numbers say "capacity", not "tuning".

## 6. Visual / Geometric Intuition

### 6.1 The two clocks on one axis

The picture that prevents most errors is a timeline carrying both the write head and the commit
head, with the sink's progress shown beneath it as a third, independent track.

```text
offsets →  9,190,000        9,190,141       9,190,191                9,412,000
log:       |====================|===============|========================|
                               ^               ^                          ^
                        pending 141..160   completed to 190        LEO (write head)
                               |
committed:  C = 9,190,000  ─────┘ (safe frontier stalls here while 141..160 are outstanding)

sink track: [applied 9,190,000..9,190,140] [gap] [applied ..9,190,190] [not yet read]
```

Caption: the commit head may only advance across a *contiguous* completed run, so a single slow
record creates a visible stall even though later offsets are durable. The gap in the sink track is
the physical reason the commit head cannot move: it is not a policy choice.

### 6.2 Backlog geometry: slope, not level

```text
lag (records)
  ^
  |                        /                 <- a > s: no recovery, unbounded growth
  |                      /
  |                    /          ___________  <- a = s: frozen at the intervention level
  |                  /          /
  |                /          /               <- a < s: finite drain, slope (a - s)
  |              /          /
  |            /          /
  |  _________/ ______________________________ > time (s)
      T0      intervention
```

Caption: what matters is the slope after the intervention, and the correct slope is `a − s`, not
`−s`. A drain forecast that reads the vertical distance and divides by gross service implicitly
draws the steepest line and is optimistic by the factor `s/(s−a)`.

### 6.3 Skew as a shape, not a statistic

Plotting per-partition lag as a set of bars makes the diagnosis visual. Three canonical shapes:

```text
shape A: one spike            shape B: staircase           shape C: uniform plateau
|          █                  |            █               |  ███ ███ ███ ███
|  ▁ ▁ ▁ ▁ █                  |        ███ █               |  ███ ███ ███ ███
+--0-1-2-3-4-->               +-------0--1--2--3-->        +---0--1--2--3------>
  hot key / per-record cost     one member owns a hot        global shortfall:
  -> members cannot help        set of partitions           -> capacity or sink
```

Caption: shape A is the only case where re-keying or per-record optimisation is the primary lever;
shape B is an assignment or member-capacity problem; shape C is the only shape where adding members
is the first move.

### 6.4 Why the partition is the unit of everything

```text
        topic (no order)                      16 independent ordered lanes
   ┌───────────────────────┐          ┌───┬───┬───┬───┬───┬─ ... ─┬────┐
   │ r r r r r r r r r r r │   ==>    │ 0 │ 1 │ 2 │ 3 │ 4 │       │ 15 │
   └───────────────────────┘          └───┴───┴───┴───┴───┴─ ... ─┴────┘
                                        │   │   │   │   │           │
                                        each: own offsets, own LEO, own lag,
                                        at most one owner in the group
```

Caption: capacity, ordering, offset arithmetic and ownership all live in the lanes. Any aggregate
figure is a lossy compression of these lanes, and the loss is exactly the information needed to
diagnose.

### 6.5 The rebalance timeline as a gap function

```text
throughput
  ^  ████████████                        ██████████████████████
  |  ████████████                        ██████████████████████
  |  ████████████        ░░ rebalance ░░ ██████████████████████
  |  ████████████        ░░  + catch-up ██████████████████████
  +----------------------------------------------------------> time
       steady            revoked          new owners warm
                        (zero service)    caches / state
```

Caption: the lost work is the area of the gap plus the lost surplus during catch-up. Twelve such
gaps per hour, each 25 s, is 5 minutes of zero throughput — often enough on its own to turn a 10%
capacity surplus into growth (see `Q-12`).

**Takeaway:** Draw the commit head, the write head and the sink track separately; most "lag" questions are answered by seeing which of the three is actually the one that stopped.

## 7. Comparison / Selection Table

### 7.1 The intervention catalogue

Each row is a distinct lever with its own precondition, its own failure signature and its own
verification. The rows are ordered by reversibility, because during an incident the reversible options
must be tried and measured before the irreversible ones.

| Lever | Precondition to be applicable | Expected effect | Cost / risk | Disqualifier | Verify by |
|---|---|---|---|---|---|
| Reduce per-record work | Profile shows one dominant term in `c_total` | Raises `s_i` without touching order | Engineering, regression risk | The dominant term is external and uncontrollable | Same workload, lower CPU/statement time, identical effects |
| Batch sink calls | Sink supports multi-row atomic writes | Lowers `c_io` by ~`1/batch` | Larger blast radius per failure | Per-record conditional logic required | Effect counts and reconciliation unchanged |
| Raise bounded concurrency | Effects reorderable or per-key serialised | Raises `s_i` up to `K/c_io` | Memory, duplicate exposure on crash | Sink already at its own ceiling | In-flight bounded, error rate flat |
| Add members | `N < P` and shared resources have headroom | Raises group capacity by `min(ΔN, P−N)` | Rebalance cost, doubled sink pressure | One hot partition is the bottleneck | `min(N,P)` grows, per-partition `s` improves |
| Re-key / salt | Business order per key not strict at top level | Spreads a hot key across partitions | Ordering contract change, migration | Strict global order for that key required | Per-partition `a` converges, `CV_a` falls |
| Increase partitions | Future traffic needs more lanes | Raises the `min(N,P)` ceiling | Key remap breaks old ordering contract | Records must stay in their current lane | Post-cutover per-entity sequence reconciles |
| Pause/partition producers | Upstream can buffer or backpressure safely | Stops backlog growth | Availability, upstream overflow | Upstream would drop data | Ingress falls, lag flattens, then drains |
| Increase retention | Broker storage available | Protects the replay window | Storage cost, longer recovery after disk faults | Disk already near capacity | Log start offset stops advancing |
| Isolated replay group | History retained and effects idempotent | Rebuilds downstream state without touching live lag | Doubles read bandwidth and sink writes | Sink cannot take the extra load | Replay completes inside `Q-14` forecast |
| Reset offsets | Explicit, approved data decision | Restores a resumable checkpoint | Permanent record loss if wrong | Records still needed downstream | Read-back position equals the approved target |
| Roll back consumer version | Change correlates with the regression | Restores prior cost profile | Loses the fix contained in the new version | The regression predates the deploy | Cost per record returns to baseline |
| Reduce fetch/batch size | Poll interval violations observed | Restores heartbeat compliance | Lower amortisation | Real cause is an unbounded external call | `max.poll.interval.ms` no longer breached |

### 7.2 The selection procedure

The procedure below is deliberately ordered so that each step can *disqualify* everything after it.
Skipping a step is how teams end up tuning a system whose arrival rate is 20× its service rate.

**Step 1 — confirm the deficit sign per partition.** Compute `a_i − s_i` for every partition, not the
topic. If any partition has `a_i ≥ s_i`, that partition cannot recover and no downstream step can
change it. Record the binding partition by name.

**Step 2 — confirm the measurement.** Recompute the lag from raw offsets and one declared end
boundary. If two tools disagree, the disagreement itself is the finding (Section 8.9) and the next
action is to fix the metric, not the system. A single aligned snapshot is worth more than a month of
dashboard history.

**Step 3 — classify the shape.** Using Section 6.3: one spike means skew (go to Step 4); a staircase
means assignment (go to Step 5); a uniform plateau means global capacity or sink (go to Step 6).

**Step 4 — skew.** Is the concentration caused by a small set of keys (measurable: top-`k` key share
of the partition's records) or by per-record cost (measurable: mean sink duration for records on the
hot partition vs peers)? Key concentration is fixed by re-keying or by splitting the producer's
event grain; cost non-uniformity is fixed by optimizing the expensive path or routing it to a
separate topic.

**Step 5 — assignment.** Is `min(N, P) < N`? Then idle members exist; the fix is either fewer members
or more partitions, and the second requires a migration plan. Is `min(N, P) = N` but some member owns
a disproportionate share of *hot* partitions? Then the assignment is technically correct and
practically bad; cooperative-sticky assignment with a warm-up or explicit partition grouping helps.

**Step 6 — global capacity or sink.** Decompose `s` with `Q-08`. If CPU dominates, the fix is
per-record work or more members. If I/O dominates and the sink is below its own ceiling, the fix is
concurrency. If the sink is at its ceiling, the fix is sink capacity, and adding consumers converts
a backlog into an error storm — the lag may even *fall* while data is being dropped into a
dead-letter path (Section 8.21).

### 7.3 Decision guide under conflicting constraints

Real incidents present conflicting constraints, and the guide below resolves them by priority rather
than by preference. The priority order is: irreversible-effect correctness, then retention
sufficiency, then throughput, then latency, then cost.

| Conflict | Resolution | Reason |
|---|---|---|
| Throughput vs strict order | Keep order for the constrained key; parallelise everything else | Order is a correctness property; throughput is not |
| Add members vs protect the sink | Protect the sink first | Extra consumers at a saturated sink increase duplicates and errors |
| Commit more often vs coordinator load | Batch commits up to the duplicate budget `Q-11` | The budget is explicit; coordinator saturation is not |
| Longer retention vs disk capacity | Shorten retention only with a written replay-window decision | Silent loss of the replay window is unrecoverable |
| Rebalance often (fast failover) vs stability | Lengthen timeouts up to the `max.poll.interval.ms` budget | Churn costs more capacity than slow detection |
| Bigger fetch vs poll deadline | Shrink fetch; bound per-record calls | A missed poll deadline revokes all partitions |
| Reset offsets vs investigate | Investigate; reset only with an approved target | A reset is a permanent data decision |

**Disqualifiers that override every optimisation.** Any of the following means stop and fix
correctness before any throughput work: offsets committed ahead of durable effects; a replay path
writing to a live sink; a group id shared by two deployments; a reset already executed whose target
was never verified; retention shorter than the outage-plus-catch-up envelope; a dead-letter path with
no consumer and no reconciliation.

### 7.4 Comparing measurement instruments

The tools matter because they disagree, and knowing *why* they disagree is itself diagnostic.

| Instrument | Measures | Strength | Weakness | Use when |
|---|---|---|---|---|
| Broker group-offset API | `E_i − C_i` for a named group | No client instrumentation, works on any group | Blocked by per-partition enumeration; boundary is LEO | First look; fleet-wide inventory |
| Client-side metrics | position, commit, assigned partitions, per-poll timing | Per-member detail and internal queue depth | Only for that member; resets on restart | Diagnosing one member |
| `kafka-consumer-groups --describe` | Per-partition C, LEO, lag, owner | Human-readable snapshot with owners | One instant; changes while you read | Incident documentation |
| Sink-side counters | durable effects/s, errors, retries | The only authoritative effect measure | No offset context | Correctness reconciliation |
| Business reconciliation | source ids vs sink rows for a cohort | Detects loss and duplication | Sampled; needs a cohort definition | Post-incident proof |

**Takeaway:** Choose the lever whose precondition you actually verified; every other lever is a guess dressed as a plan.

## 8. Failure Modes & Edge Cases

Every entry below is a **HYPOTHETICAL** teaching scenario, constructed to isolate a mechanism. None
is an observed incident, and none should be cited as evidence that a specific system behaved this
way. Each entry names competing causes, a discriminating check that separates them, a bounded fix,
the verification that closes the loop, and the reason a naive fix fails.

### 8.1 One partition grows while fifteen are flat

**Competing causes.** (a) A dominant key or key family lands on one partition. (b) Per-record cost is
much higher for that partition's records (large payloads, expensive downstream calls, a different
event type). (c) The member owning it is degraded — GC, a noisy neighbour, a throttled network path.

**Discriminating check.** Compare three ratios for the hot partition and a peer: records/s,
bytes/s, and mean sink duration per record. If records/s is 10× and bytes/records is 1×, it is (a).
If records/s is 1× and bytes/s is 10×, it is (b) in its payload form. If records/s and bytes/s are
1× but mean sink duration is 5×, it is (b) in its cost form. If the hot partition's metrics match
its peers but its *member's* CPU or poll latency is anomalous, it is (c).

**Bounded fix.** For (a): salt or split the key after confirming the ordering contract permits it;
never re-key during a peak without a migration plan. For (b): optimise the measured term, or route
that event type to its own topic with its own consumer. For (c): replace or restart the member only
after capturing its state, and confirm the effect ledger reconciles.

**Verification.** Per-partition arrival rates converge (`CV_a` falls materially), and the hot
partition's net drain is positive for at least a full commit interval. Confirm the effect ledger has
one row per `event_id` for the affected interval.

**Why the naive fix fails.** Adding members redistributes partitions that are not the problem and
raises concurrency against a sink that did nothing wrong.

### 8.2 Extra members sit idle while lag persists

**Competing causes.** (a) `N > P`, so there is nothing to assign. (b) Members are subscribed to a
topic that does not include the lagging partition (subscription mismatch, a regex that does not
match). (c) Authorization: some members cannot access the topic and repeatedly fail assignment
without appearing "down". (d) A static-membership or custom assignor pins partitions to a subset.

**Discriminating check.** Enumerate the group's members with their assigned partition lists. If
`assigned = 0` for some members, distinguish by error log: subscription mismatch logs no error but
shows a different topic set; authorization failure logs a specific `TOPIC_AUTHORIZATION_FAILED`;
static membership shows the same partition count after a restart with no rebalance.

**Bounded fix.** Remove surplus members (cheapest correct action), or fix the subscription, or grant
the missing ACL through the approved access process. Do not repartition an existing topic merely to
use idle replicas that were provisioned by mistake.

**Verification.** `Σ assigned partitions = P`, and every member's assigned count is within one of
`P/N`. Then confirm throughput rose; if it did not, the binding constraint was elsewhere.

**Why the naive fix fails.** Repartitioning to consume idle members remaps keys and can break
per-entity ordering for every producer, which is a far larger change than the one being solved.

### 8.3 The staircase: one member owns every hot partition

**Competing causes.** (a) Range assignor with co-partitioned topics puts partition `i` of each topic on
the same member, so every topic's hot partition lands together. (b) Custom assignor or manual
assignment pinned hot partitions to one member. (c) Uneven partition counts between co-subscribed
topics force an unbalanced range split.

**Discriminating check.** List partitions with their owners and their `a_i`. If the highest-rate
partitions share an owner, the assignment is the cause, not the traffic. Recompute what
round-robin would produce; if round-robin would separate them, the assignor is the defect.

**Bounded fix.** Switch to a cooperative-sticky strategy, which balances on prior assignment and
moves only what must move, or explicitly group partitions so that hot lanes are not co-located.

**Verification.** The maximum per-member assigned *arrival rate* (not partition count) is within a
small factor of the mean, and no single member's CPU exceeds its budget.

**Why the naive fix fails.** Scaling out the group does not help when the skew is inside the
assignment: the new members get cold partitions and the hot ones stay together.

### 8.4 The sink is the bottleneck and everyone blames the consumer

**Competing causes.** (a) The database is at its connection or IOPS ceiling. (b) The consumer's own
connection pool is exhausted by leaks or by unbounded concurrency. (c) Lock contention on a hot
business key inside the sink turns parallel writes into serial waits.

**Discriminating check.** Plot consumer CPU, pool wait time, and sink-side latency on the same axis.
Low consumer CPU with high pool wait points at (b). High pool wait with healthy consumer but rising
sink query time points at (a). Parallel writes that get slower as concurrency rises, with sink CPU
low, points at (c).

**Bounded fix.** Cap concurrency at the level where sink latency is flat, add a request deadline so
slow calls cannot occupy slots indefinitely, and fix the pool leak if present. For (c), reduce
per-key write parallelism rather than adding capacity.

**Verification.** Sustained completion rate rises while sink error rate and p99 latency stay flat.
If error rate rises with throughput, the new concurrency level is above the sink's knee.

**Why the naive fix fails.** Raising consumer concurrency moves the queue from Kafka into the
database's connection pool, which has a much smaller buffer and fails louder.

### 8.5 Retry amplification converts a blip into an outage

**Competing causes.** (a) Transient sink errors are retried without jitter or cap, so all members
retry in phase. (b) A permanent error (bad payload, constraint violation) is retried forever. (c)
Producer-side retries are duplicating records because idempotence is not enabled on the producer, so
the consumer sees more records than the business emitted.

**Discriminating check.** Count retries per `event_id`. One id with thousands of attempts is (b).
Many ids with a synchronised retry cadence is (a). Ingress record count exceeding producer emit
count, measured at the producer, is (c).

**Bounded fix.** Split retryable from permanent errors with an explicit classifier; bound retries;
add exponential backoff with jitter; route permanent failures to a durable, monitored dead-letter
topic under an approved no-loss policy. Enable producer idempotence so broker-level retries do not
duplicate.

**Verification.** Retry volume per id has a hard ceiling, and `Σ durable effects = Σ unique source
ids` for the interval.

**Why the naive fix fails.** Removing retries entirely converts a transient blip into permanent loss;
the fix is bounded retry, not no retry.

### 8.6 One record blocks a partition forever

**Competing causes.** (a) A malformed payload that the deserializer rejects. (b) A schema that the
consumer cannot read because it was deployed before a producer change. (c) A downstream dependency
that is permanently gone (a deleted customer row, a closed account).

**Discriminating check.** Fetch the raw bytes at the stuck offset and run them through the consumer's
deserializer in isolation. A local failure is (a) or (b); inspect the schema id in the record header
to separate them. If the record deserialises cleanly but the sink rejects it deterministically, it is
(c).

**Bounded fix.** Quarantine to a dead-letter topic with the original bytes and the error class, then
advance. Only under a written no-loss policy, and only with a replay route back. Never skip the offset
silently.

**Verification.** The partition advances past the offset, the quarantined record is readable and
replayable, and a reconciliation lists every quarantined id with an owner.

**Why the naive fix fails.** Restarting the consumer or resetting offsets makes the same record block
again, possibly at a worse time.

### 8.7 Lag is zero and records are missing

**Competing causes.** (a) Offsets were committed before the effect was durable (auto-commit, or a
commit after enqueue rather than after acknowledgement). (b) The dead-letter path accepted the failure
and nobody reconciled it. (c) The sink accepted the write but the read path queries the wrong cohort —
a time filter, a partition of the ledger table, an environment.

**Discriminating check.** Pick twenty source ids from the interval and query the sink directly by id,
bypassing the application's read path. If they are absent, the commit ordering is wrong (a) or they
were quarantined (b) — check the dead-letter topic for the same ids. If they are present but invisible
to the application, it is (c).

**Bounded fix.** Stop advancing the unsafe frontier; recover the missing interval from the retained
source with an idempotent consumer; fix the commit ordering so that commit follows durable effect.

**Verification.** `Σ source ids in ledger = Σ source ids produced`, with the difference explained id
by id, and the fix exercised by a deliberate crash between effect and commit.

**Why the naive fix fails.** Increasing retention or replaying the whole topic without isolating the
sink can duplicate effects and make reconciliation harder, not easier.

### 8.8 Duplicates appear after every restart

**Competing causes.** (a) The effect commits before the offset commit — the classic
at-least-once window. (b) The producer emits two distinct records for one logical operation (retry
without idempotence, or a double-submitting UI). (c) Deduplication exists but its key differs between
attempts — a random UUID generated per attempt.

**Discriminating check.** For a duplicated effect, compare the *source coordinates* (partition,
offset). Same coordinates means (a): one record processed twice. Different coordinates with the same
business fields means (b). Same coordinates and different dedup keys means (c) — inspect the key
derivation.

**Bounded fix.** For (a): idempotent sink keyed on a stable logical id, or a transaction that covers
both effect and offset when the sink is Kafka. For (b): producer idempotence plus a business-level
unique constraint. For (c): derive the key deterministically from stable event fields.

**Verification.** Deliberate crash injection at the exact point between sink commit and offset commit
produces exactly one effect. Run it repeatedly, not once.

**Why the naive fix fails.** Committing offsets earlier removes duplicates by introducing loss; that
trade is only acceptable under an explicitly accepted data-loss policy.

### 8.9 Two dashboards disagree about lag

**Competing causes.** (a) Different end boundaries: LEO versus HW versus LSO. (b) Different isolation
levels on the consumer side (`read_committed` versus `read_uncommitted`). (c) Different scrape instants
on a system whose lag is changing quickly. (d) One tool sums lag across all partitions while another
reports the maximum.

**Discriminating check.** Take one partition and compute all candidate lags from a single snapshot of
raw numbers: `LEO − C`, `HW − C`, `LSO − C`. Match each dashboard's value to one of these. If
all three differ from both dashboards, check the scrape timestamps.

**Bounded fix.** Publish one operational definition — which boundary, which isolation level, which
aggregation — and version it with the dashboard. Keep the raw offsets visible so any disputed number
can be reconstructed.

**Verification.** Two independent tools fed the same snapshot produce the same number, and a
documented test case (including an open transaction) reproduces the expected difference.

**Why the naive fix fails.** Averaging the two numbers produces a figure that corresponds to no
defensible boundary and cannot be reconciled with a reset or replay decision.

### 8.10 Rebalance loop: partitions move every few minutes

**Competing causes.** (a) Processing between `poll()` calls exceeds `max.poll.interval.ms`, so the
member is evicted while still working. (b) Heartbeat thread starved by a long GC pause or CPU
starvation, so the session times out. (c) Coordinator or network instability causing repeated
`REBALANCE_IN_PROGRESS` responses.

**Discriminating check.** Log the wall-clock duration of the poll loop and compare with the configured
interval. If durations approach the limit, it is (a). Check GC pause logs and CPU steal for (b). If
polls are fast and the member is healthy but the group still churns, inspect coordinator logs and
network errors for (c).

**Bounded fix.** For (a): reduce per-poll work (smaller batch, bounded batch duration), or raise the
interval *and* the corresponding broker-side maximum if the deployed version has one. For (b):
reduce heap pressure and allocation rate. For (c): repair connectivity; do not "fix" churn by
lengthening timeouts indefinitely.

**Verification.** Rebalance count over a one-hour window drops to the planned number, and the
`Q-12` availability fraction returns above target.

**Why the naive fix fails.** Disabling the poll timeout hides a stuck member that is holding
partitions and processing nothing, converting churn into silent stall.

### 8.11 Stale owner keeps writing after losing its partition

**Competing causes.** (a) The application ignores the revocation callback and continues in-flight
work. (b) In-flight work is correctly abandoned but a retry queue flushes afterwards. (c) Two
different group ids consume the same topic and both write to the same sink — there is no "stale
owner", there are two owners by design.

**Discriminating check.** Compare assignment epochs recorded at effect time with the epoch at commit
time. Effects stamped with an older epoch are (a). Effects from a *different group id* are (c) — check
the group id in the effect's provenance metadata.

**Bounded fix.** Fence writes by epoch: include the generation/epoch in the sink's conditional write,
or use a transactional producer whose commits are fenced by the broker. On revocation, stop taking
new records and let in-flight work either complete under a valid epoch or be abandoned.

**Verification.** A test that revokes ownership during a deliberately slow sink call produces exactly
one effect per logical id, and the abandoned work leaves no partial rows.

**Why the naive fix fails.** Reducing `max.poll.records` shrinks the window but does not eliminate it;
the epoch check is what makes it impossible.

### 8.12 The commit frontier stalls behind one slow record

**Competing causes.** (a) A single record's sink call has no deadline and is hanging. (b) One key is
serialised behind a slow predecessor. (c) The completion tracker has a bug: a completion callback is
lost, so the frontier never advances even though the effect is durable.

**Discriminating check.** Inspect the in-flight set: if it contains one entry with an age far above
the p99 of sink latency, it is (a). If the stalled entry belongs to a key with other in-flight work,
it is (b). If the in-flight set is *empty* while the committed offset does not advance, the tracker is
the defect (c) — the offsets were completed but never retired.

**Bounded fix.** Set a request deadline on every external call so a hang becomes a retryable failure.
For (b), accept that per-key serialisation is the contract and size concurrency per key rather than
globally. For (c), add a frontier-age metric and a unit test that simulates out-of-order completion.

**Verification.** Frontier age never exceeds the declared maximum, and a test that completes offsets
out of order advances the frontier exactly to the first gap.

**Why the naive fix fails.** Committing the position instead of the frontier "unsticks" progress by
skipping the gap, which is precisely the loss the tracker exists to prevent.

### 8.13 Out-of-range: the commit is older than the retained log

**Competing causes.** (a) The group was stopped or stalled longer than retention. (b) Retention was
shortened by a configuration change or by disk pressure policies. (c) The commit itself was reset
backwards (or the group id was reused after a reset).

**Discriminating check.** Compare `C` with the log start offset and the segment timestamps. If
`C < log_start` and the group was healthy until recently, compute how long ago `C`'s record would have
been written; if that exceeds configured retention, it is (a). Check configuration history for (b).
Check group-id reuse and reset records for (c).

**Bounded fix.** Treat it as a data decision: identify exactly which ids are unavailable, determine
whether an archive, a downstream copy, or the producer's own outbox can supply them, and only then
choose `earliest` (accepting a gap) or a documented reconstruction. Never reset to `latest` to make
the error disappear.

**Verification.** A written gap report enumerates the missing id range, its cause, and the
reconstruction or the accepted-loss approver. After the reset, the consumer's read-back position
matches the approved target.

**Why the naive fix fails.** Resetting to `latest` restores green dashboards and silently abandons
every record in the gap; nobody notices until reconciliation, often weeks later.

### 8.14 Retention headroom is shorter than the recovery time

**Competing causes.** (a) The backlog's oldest record is within hours of deletion because recovery is
slow. (b) Retention is shorter than the team believed (defaults applied, or per-topic overrides).

**Discriminating check.** Compute `time_to_deletion = (C − log_start) / a_i` and compare with the
`Q-15` drain estimate. If the deletion time is smaller, retention is the binding clock regardless of
throughput.

**Bounded fix.** Increase retention if storage allows; prioritise draining the oldest partitions;
consider an archival sink so that the replay window does not depend on broker storage.

**Verification.** `time_to_deletion` exceeds `T_drain` with a stated margin, and the margin is
displayed on the operational dashboard rather than recomputed during the next incident.

**Why the naive fix fails.** Adding consumers does not move the log start offset; only time and
retention policy do.

### 8.15 Replay silently re-triggers live side effects

**Competing causes.** (a) The replay group writes to the production sink with no mode flag. (b) A mode
flag exists but the replay path bypasses it for one code branch (a notification, a webhook). (c) Event
ids are regenerated during replay so idempotency does not recognise the originals.

**Discriminating check.** Compare the sink destination, the group id, and the event id derivation
between live and replay paths. Grep for direct calls to the external-effect client that do not consult
the mode flag.

**Bounded fix.** Make the sink destination a required configuration for replay with no default,
and deny the replay principal write access to live external systems. Regenerate no ids: replay must
carry the original id.

**Verification.** A replay over an interval containing a notification-producing event creates zero
notifications, and the effect ledger still shows the business mutation.

**Why the naive fix fails.** Relying on a boolean read at one call site fails as soon as a new effect
type is added without the check; permission boundaries fail closed.

### 8.16 Deduplication state expires inside the replay horizon

**Competing causes.** (a) The dedup table has a TTL shorter than the oldest replayable record. (b)
Dedup exists only in memory, so any restart forgets it. (c) The dedup key is not stable across
producer restarts.

**Discriminating check.** Compare the dedup retention setting with the oldest offset the replay plan
can reach, converted to time. If dedup TTL < replay horizon, it is (a). Inspect whether the store is
durable for (b).

**Bounded fix.** Align the dedup horizon with the maximum replay window, or make the sink's unique
constraint the deduplication mechanism so retention is the table's own policy.

**Verification.** Replaying the oldest supported record produces no second effect — tested, not
assumed.

**Why the naive fix fails.** Making dedup permanent grows without bound and eventually becomes its own
capacity problem; alignment with the declared replay window is the only stable answer.

### 8.17 Poll deadline breached by batch size under load

**Competing causes.** (a) A batch of 500 records takes longer than `max.poll.interval.ms` at peak
load. (b) One record inside the batch triggers a retry with long backoff. (c) A single external call
has no timeout and blocks for minutes.

**Discriminating check.** Plot the distribution of batch durations, not the mean. If the p99 exceeds
the interval while the median is fine, it is (a) or (b). If a single call's duration dominates that
p99, it is (c).

**Bounded fix.** Bound the batch *by duration* as well as by count: stop taking new records within a
poll once the elapsed time reaches a fraction (commonly a third to a half) of the interval. Add
deadlines to every external call.

**Verification.** p99 batch duration stays under the budget at peak synthetic load, and rebalance
count during a peak test remains zero.

**Why the naive fix fails.** Raising `max.poll.interval.ms` moves the boundary but the tail grows with
load, so the breach returns at a higher traffic level — and detection is now slower.

### 8.18 Commit frequency at both extremes

**Competing causes.** Excessive: commit per record. Insufficient: commit every 60 s or only at
shutdown. Silent: commits failing with retriable errors that are logged and swallowed.

**Discriminating check.** Compare successful commit count with the configured expectation. A gap
between attempted and successful commits identifies the silent case; a rate near one per record
identifies the excessive case.

**Bounded fix.** Commit on a bounded interval or record count, whichever comes first, always behind a
verified safe frontier. Alert on commit errors and on frontier age.

**Verification.** `Q-11` replay exposure stays within its budget, and coordinator request rate stays
inside its quota. Both metrics on one panel, because the fix trades one for the other.

**Why the naive fix fails.** Setting commit interval to zero (every record) will saturate the
coordinator at high throughput; the failure then looks like group instability rather than commit
volume.

### 8.19 Lag flat while business freshness degrades

**Competing causes.** (a) The consumer processes quickly but a downstream stage is slow, so end-to-end
age grows while Kafka lag is constant. (b) Lag is measured in offsets, and payloads grew so the same
offset count represents far more bytes and processing time. (c) Records are being written to a
dead-letter path at a rate that keeps the frontier moving while real effects are rejected.

**Discriminating check.** Measure age at each stage boundary, not only at the source. Compare bytes/s
and per-record duration with the previous week. Compare the ratio of durable effects to consumed
records — a ratio below one with a healthy frontier is (c).

**Bounded fix.** Define the freshness SLO at the *last* stage and alert there. Track per-record cost
as a metric so that (b) is visible before it becomes an incident. Reconcile the effect ratio
continuously.

**Verification.** The end-to-end age percentile matches a hand-traced record from produce to sink, and
the effect ratio is 1.0 for a sampled cohort.

**Why the naive fix fails.** Tuning the consumer's throughput does nothing when the delay lives two
stages later; the metric being watched is simply the wrong one.

### 8.20 Producer surge versus producer retry storm

**Competing causes.** (a) Genuine business volume increase. (b) Producer retries duplicating records
because acknowledgements time out while the write actually succeeded. (c) An upstream job re-emitting
a historical interval.

**Discriminating check.** Compare *unique* ids/s with *total* records/s at the topic. Equality means
(a). A gap means duplication (b) or re-emission (c); check whether duplicate ids are clustered around
a broker or network event (b) or around a job run (c).

**Bounded fix.** For (b): enable producer idempotence, fix acknowledgement timeout settings, and
reduce retry pressure on the broker. For (c): add an emission-id or interval marker so downstream can
deduplicate a re-emitted range. For (a): admission control coordinated with the producer owners.

**Verification.** Unique-id rate equals business volume, and the consumer's backlog slope matches the
genuine arrival rate.

**Why the naive fix fails.** Scaling consumers to absorb 2× traffic caused by duplication wastes
capacity and doubles the duplicate effects downstream.

### 8.21 Dead-letter path as a silent loss channel

**Competing causes.** (a) Failures are published to a dead-letter topic with no consumer, so nothing
downstream knows. (b) The dead-letter publish itself fails and the offset is committed anyway. (c)
Records are dead-lettered correctly but their business owner never sees them.

**Discriminating check.** Count consumed records minus (durable effects + dead-lettered records). A
non-zero residual means (b). A zero residual with a backlog-free dead-letter topic and no readers
means (a). Check whether the dead-letter volume appears in any business report (c).

**Bounded fix.** Make the dead-letter publish part of the atomic unit: commit the offset only after
the dead-letter record is durable. Give every dead-letter topic an owner, an alert on ingress rate,
and a documented replay procedure.

**Verification.** For a sampled interval, `consumed = effects + quarantined`, every quarantined id has
an owner and a disposition, and a test that fails the dead-letter publish leaves the offset
uncommitted.

**Why the naive fix fails.** Deleting the dead-letter topic to "stop losing data" loses it faster and
more visibly; the fix is accounting, not deletion.

### 8.22 Partition count change breaks per-entity order

**Competing causes.** (a) `key mod P` changed, so one entity's records now map to two partitions and
two consumers process them concurrently. (b) The producer changed key serialisation (a string to a
struct, a case change), which changes the hash input. (c) A migration replayed old records into the
new layout without a barrier.

**Discriminating check.** For a specific entity, list its records with partition and offset before
and after the cutover. Interleaved processing of two partitions for one entity is (a) or (b);
determine which by checking whether the serialised key bytes changed at the same time.

**Bounded fix.** Stop the producer, drain, increase partition count, and resume — accepting a bounded
gap — or use a two-topic migration with a versioned key and a downstream merge that restores order by
version. Never claim per-entity order across an unannounced repartition.

**Verification.** For a sample of entities, the version sequence at the sink is monotonic. Run an
adversarial test that produces two records for one entity in quick succession around the cutover.

**Why the naive fix fails.** Relying on the sink's UPSERT to "fix" order only works if the sink applies
version conditionals; without them, the later arrival wins regardless of which is newer.

### 8.23 Compaction removed the history a rebuild needed

**Competing causes.** (a) The topic is `cleanup.policy=compact`, retaining only the latest value per
key. (b) `delete,compact` deleted old segments by time before compaction could preserve them. (c) The
rebuild expected to see every event but the upstream was keyed such that intermediate states were
overwritten.

**Discriminating check.** Inspect the topic's cleanup policy and, for one key, list its retained
records. One record per key is (a). Missing old segments for keys that exist is (b).

**Bounded fix.** Rebuild from an uncompacted audit log or the source system's change history. If
neither exists, reconstruct only the final state per key and document that intermediate transitions
are unrecoverable from this source.

**Verification.** The rebuilt output matches a known reference snapshot for a sampled key set, and
the documented contract states "latest state" rather than "full history".

**Why the naive fix fails.** Reading the compacted topic from `earliest` and expecting a full event
history misunderstands the log: compaction preserves state, not history.

### 8.24 Record density: bytes and records diverge

**Competing causes.** (a) Payload size distribution shifted (images, enriched documents). (b)
Compression changed (a codec change or a batch of incompressible data). (c) The consumer's fetch
settings now return fewer records per byte, so the same record rate costs more round trips.

**Discriminating check.** Compare bytes/s and records/s trends and the p50/p99 record size. Byte
growth with flat record count is (a). Bytes/s flat with record count flat but broker network up is
(b). Fetch latency up with unchanged sizes is (c).

**Bounded fix.** Provision for bytes as well as records (Section 9.3), bound payload size at the
producer, and set fetch bounds in bytes rather than only in record counts.

**Verification.** End-to-end latency and broker network utilisation return to baseline at unchanged
record throughput.

**Why the naive fix fails.** Sizing purely on records/s under-provisions network and disk, and the
failure appears as mysterious fetch latency rather than as capacity exhaustion.

**Takeaway:** Each failure mode has a competing cause that produces the same symptom and a different fix; the discriminating check is the whole skill.
## 9. Complexity & Cost Analysis

### 9.1 Algorithmic cost of the consumer's own bookkeeping

Let `R` be records processed on one partition, `K` the bound on concurrently in-flight records per
member, and `P`, `N` as before. The completion tracker is the only non-trivial data structure, and
its cost determines whether a high-throughput consumer can afford out-of-order processing.

| Operation | Data structure | Time | Space | Notes |
|---|---|---|---|---|
| Insert completion mark | hash set keyed by offset | `O(1)` average | `O(1)` per in-flight record | must be bounded by `K` |
| Advance frontier | scan of contiguous marks | `O(1)` amortised per advance | — | each mark examined at most once |
| Worst-case advance | scan after a large gap fills | `O(K)` at that instant | — | bounded by `K`, not by `R` |
| Commit | network round trip to coordinator | `O(1)` messages | — | rate-limited by coordinator quota |
| Total over `R` records | — | `O(R)` amortised | `O(K)` | not `O(R log R)` unless a heap is used |

The key claim: with a hash set plus a cursor, tracking is `O(R)` total rather than `O(R log R)`,
because each offset is inserted once and retired once. Using a priority queue instead is also correct
and adds `log K` per operation with no benefit unless completion marks can arrive arbitrarily far in
the future, which they cannot when `K` is bounded.

```text
[Q-18] FORMULA: total_tracking_ops = R (inserts) + R (retirements) = 2R;  peak_space = K
[2] WHERE: R [count] records on one partition; K [count] in-flight bound.
[3] CONSTRAINT: K is enforced; completions strictly correspond to previously inserted offsets.
[4] NUMERICAL EXAMPLE: R = 10,000,000 records, K = 200
                   => 20,000,000 hash operations and at most 200 marks resident.
[5] INTERPRETATION: the tracker is never the bottleneck; the sink round trip is, by four to six
                   orders of magnitude per record.
```

**Boundary.** If `K` is not enforced, memory is `O(R)` in the worst case and the process fails by
heap exhaustion rather than by throughput. An unbounded async consumer is therefore a memory bug
that presents as a lag incident.

### 9.2 Per-member capacity arithmetic

The sizing question is: given a target steady-state arrival rate, how many members and partitions are
needed, and which of the two ceilings binds?

```text
[Q-19] FORMULA: N_required = ceil( (SUM_i a_i) * (1 + h) / s_member )
                N_effective = min(N_required, P)
[2] WHERE: N_required [count]; SUM_i a_i [records/s]; h [-] headroom fraction; s_member
           [records/s] measured per-member durable service rate.
[3] CONSTRAINT: s_member measured under concurrent load with the real payload mix; headroom covers
                member loss and bursts; partitions can be distributed evenly.
[4] NUMERICAL EXAMPLE: SUM a = 9,000, h = 0.25, s_member = 3,000
                   => N_required = ceil(9,000 * 1.25 / 3,000) = ceil(3.75) = 4.
                   With P = 3 => N_effective = 3, and the group's ceiling is 3 * 3,000 = 9,000/s,
                   exactly at arrival: no headroom, so the design is infeasible as stated.
[5] INTERPRETATION: the headroom requirement is not satisfied by "it fits", because at exactly
                   9,000/s of capacity against 9,000/s of arrivals the backlog never drains (4.4).
```

### 9.3 Storage and network sizing, with units

Broker storage is a replicated byte quantity, not a record quantity. The conversion chain is
records → bytes → replicated bytes → retained bytes, and each step has its own multiplier.

```text
[Q-20] FORMULA: S = a * b * (1 - c) * r * T_ret
[2] WHERE: S [bytes]; a [records/s]; b [bytes/record] uncompressed payload; c [-]
           compression ratio achieved; r [count] replication factor; T_ret [s] retention.
[3] CONSTRAINT: b measured as the real serialised size including headers; c measured on this
                payload mix; framing, indexes and segment overhead excluded and added separately.
[4] NUMERICAL EXAMPLE: a=20,000, b=500, c=0.40, r=3, T_ret=86,400
                   => S = 20,000 * 500 * 0.60 * 3 * 86,400
                        = 20,000*500 = 10,000,000 B/s raw
                        => *0.60     = 6,000,000 B/s compressed
                        => *3        = 18,000,000 B/s replicated
                        => *86,400   = 1,555,200,000,000 B = 1.5552 TB (decimal).
[5] INTERPRETATION: without compression the same retention needs
                   10,000,000*3*86,400 = 2,592,000,000,000 B = 2.592 TB; compression saves 40% of
                   the most expensive line item in this design.
```

Independent check: `6,000,000 B/s × 86,400 s = 518,400,000,000 B` per replica, times 3 equals
`1,555,200,000,000 B`, matching the chain above. In binary units
`1,555,200,000,000 / 2^40 ≈ 1.4146 TiB` — a 9.9% difference from the decimal figure, which matters
when comparing against a disk vendor's capacity claims.

Add framing explicitly rather than inside `b`: record headers, offsets, batch headers and index
entries commonly add a few percent, and a provisioned figure should state whether they are included.

### 9.4 Operational cost of the levers

| Lever | Recurring cost | One-off cost | Failure cost if wrong |
|---|---|---|---|
| Extra members | Instances, plus rebalance churn | Deployment work | More sink pressure: duplicates, errors |
| Extra partitions | Broker metadata and fewer keys per lane | Migration and key remap | Broken per-entity order across the cutover |
| Higher replication factor | `r ×` storage and write amplification | Rebalancing data movement | Slower writes, longer leader election |
| Longer retention | Linear in `T_ret` (`Q-20`) | Disk provisioning | Disk pressure during a fault |
| Frequent commits | Coordinator request rate | — | Group instability at high record rates |
| Bounded retries + DLQ | DLQ storage and an owner | Tooling and runbook | Silent loss if the DLQ has no consumer |
| Isolated replay group | Doubled read bandwidth, sink writes | Environment and access design | Live contamination if not isolated |
| Deduplication store | Rows proportional to replay horizon | Schema and index design | Duplicate effects after TTL expiry |

### 9.5 The cost of being wrong, quantified

The interesting comparison is not "throughput versus latency" but "cost of an hour of lag" versus
"cost of the fix", and the two are on different scales.

```text
[Q-21] FORMULA: exposure = a_stale * w_business * h_duration
[2] WHERE: exposure [business units]; a_stale [records/s] arriving behind the staleness threshold;
           w_business [business units/record]; h_duration [s] the incident duration.
[3] CONSTRAINT: w_business is a declared value per event type, not an estimate invented at incident
                time; the staleness threshold is defined on the business objective.
[4] NUMERICAL EXAMPLE: a_stale = 190 records/s, w_business = 1 order affected per record,
                   h_duration = 2,400 s (40 min) => exposure = 190 * 1 * 2,400 = 456,000 records
                   delayed. If 0.4% breach an SLA with a fixed penalty of 2 currency units:
                   456,000 * 0.004 * 2 = 3,648 currency units.
[5] INTERPRETATION: this converts a lag figure into a decision quantity; the same 456,000 records
                   at 40% SLA breach would be 364,800 currency units, which changes the acceptable
                   cost of prevention by two orders of magnitude.
```

**Boundary.** `w_business` is an organisational input, and the linearity assumption (each delayed
record has independent cost) fails when the cost is dominated by a threshold — a settlement cutoff, a
regulatory window — where the loss jumps discontinuously.

**Takeaway:** Size for bytes and replication as well as records, bound `K` or you have a memory bug, and price the delay in business units before arguing about capacity.

## 10. Key Theorems / Results Reference

Each result below is stated with its hypotheses, because the common misuse is to apply the conclusion
outside them.

### 10.1 Single-owner theorem (group assignment)

**Statement.** Under the classic consumer-group protocol, at any instant within a stable generation, each
partition of a subscribed topic is assigned to at most one member of the group. Consequently the number
of partitions concurrently processed by the group is at most `min(N, P)`.

**Hypotheses.** All members in the same group id; no manual assignment duplicating a partition across
groups; the generation is stable (not mid-rebalance).

**Consequence.** Adding members beyond `P` adds no processing capacity. A single-partition bottleneck
has a group-parallelism ceiling of one.

**What it does not say.** It does not prevent a member from processing several records from its partition
concurrently, and it does not prevent a *different group id* from reading the same partition at the same
time. Both are common sources of "but I added members and it scaled" and "but two consumers are reading
the same partition".

### 10.2 Safe-frontier invariant (commit correctness)

**Statement.** If the committed next offset `C` is always at most the smallest offset in the in-flight or
uncompleted set, then no crash-and-restart can skip a record that has not been durably effected.

**Proof sketch.** On restart the group resumes at `C`. By hypothesis every offset `< C` is either durably
effected or deliberately excluded by policy. Therefore no offset in `[C, E)` that lacked an effect is
retrospectively declared complete. The invariant is maintained inductively: a commit is issued only when
the prefix `[old_C, new_C)` is complete, so `old_C → new_C` preserves it.

**Hypotheses.** Completion means every required external effect is durable; offsets are committed only by
the current owner under the current epoch; the sink does not lose acknowledged effects.

**Consequence.** The frontier can stall behind a single record, which is the *point*. Stalling is a
correctness property, not a performance defect.

**What it does not say.** It does not prevent duplicates: a crash after the effect but before the commit
replays the interval `[C, effect_frontier)`. Preventing duplicates requires idempotency or a transaction,
which is a separate result.

### 10.3 Duplicate bound under periodic commit

**Statement.** With commits every `τ` seconds and steady arrival `a`, a crash causes at most `a·τ`
replayed records and, under a uniformly distributed crash instant, an expected `a·τ/2`.

**Hypotheses.** Arrival at rate `a`; the commit interval is respected and successful; the crash is
independent of the commit phase.

**Consequence.** Deduplication capacity and idempotency can be sized from `τ` and `a` rather than guessed.

**Boundary.** Adversarial crash timing (a crash loop at a fixed phase) violates the uniformity assumption
and makes the worst case recur, so the worst case — not the expectation — governs the storage requirement.

### 10.4 Drain feasibility condition

**Statement.** In the fluid model with constant `a` and `s`, the backlog converges to zero in finite time
if and only if `s > a`; it is constant if `s = a`; it diverges if `s < a`.

**Hypotheses.** Single server, no loss, capacity available whenever work is queued, rates constant over
the horizon considered.

**Consequence.** `T_drain = B/(s − a)` and its sensitivity to the surplus is `dT/ds = −B/(s−a)²`, so
halving the surplus doubles the time and a surplus near zero makes the estimate explode.

**Boundary.** Rates are never truly constant: bursts, retries, competing tenants and rebalances make `s`
time-varying, and a time-varying `s(t)` below `a(t)` for any sustained sub-interval restarts the growth
phase. Any forecast must be revalidated against measured net drain within one commit interval.

### 10.5 Little's Law applied to in-flight work

**Statement.** In a stable system, `K = a · W` exactly, where `K` is the time-average number in the
system, `a` the throughput and `W` the mean time in system.

**Hypotheses.** Stability (arrival ≈ departure over the window); the definitions of "in system" are the
same for all three quantities.

**Consequence.** A sinking service rate does not reduce `K`; it raises `W` and, if the consumer keeps
admitting work, raises `K` too. Bounding `K` is the only way to bound memory independently of `W`.

**Boundary.** It is an identity about averages: it says nothing about the tail. A p99 latency requirement
needs queueing theory with a service-time distribution, not Little's Law.

### 10.6 Offset distance is not a record count

**Statement.** For a partition whose log has had segments deleted, been compacted, or contains aborted
transaction markers, the offset distance `E − C` is an upper bound on the number of readable business
records in `[C, E)`, with equality only when the range contains no gaps.

**Hypotheses.** Offsets are assigned densely at append time and can become sparse afterwards.

**Consequence.** Lag denominated in offsets overstates record work whenever gaps exist; every conversion
from offsets to records must state its density assumption.

**Boundary.** A density of 1.0 is exact for a short, recently written, uncompacted range — which is why
converting to records is usually safe for a live tail and unsafe for a multi-day backlog.

### 10.7 Retention race

**Statement.** A group can recover from a backlog only while the earliest required offset remains
addressable. Formally, recovery requires `t_now + T_drain < t_delete(C)`, where `t_delete(C)` is the time
the segment containing `C` is removed.

**Hypotheses.** Retention deletes by segment age; no archival copy exists; the drain rate estimate is
valid.

**Consequence.** Retention is a *clock* that competes with the drain, and it can bind first even when
throughput is adequate. Long outages require either longer retention or an archival sink.

**Takeaway:** Four results do the heavy lifting — single ownership, the safe-frontier invariant, the strict-surplus feasibility condition, and the retention race; everything else in this file is an application of one of them.

## 11. Interview / Exam Questions

Fifteen questions, ordered from mechanism to judgement. Each carries a checked answer, the reasoning
behind it, and the red flag that would make an interviewer doubt the candidate.

### 11.1 What exactly is ordered in Kafka?

**Answer.** Order exists within a partition, in offset order. A topic has no order across partitions.
Consumers see each partition's records in offset order, but interleaving across partitions is arbitrary
and, once multiple members process partitions concurrently, even arrival order at a sink is not
guaranteed.

**Reasoning.** The log is a per-partition append structure with a single writer per partition. There is
no global sequencer, and adding one would destroy the scalability that makes the design useful.

**Red flag.** Saying "Kafka guarantees ordering" without the qualifier, or claiming `max.in.flight` tuning
creates topic-level order. It creates per-partition producer-side order, not cross-partition order.

### 11.2 Given this table, is the group able to drain? Show your arithmetic.

Partition A: `a = 12,600`, `s = 1,600`. Partition B: `a = 190`, `s = 3,000`. Others similar to B.

**Answer.** Partition A has `a − s = +11,000` records/s, so no drain is possible for that partition at
any member count. Partition B drains at `2,810` records/s. The group's aggregate does not matter: the
system diverges at the rate of the worst partition.

**Reasoning.** From `Q-01` and `Q-04`, the clamp is inside the sum. Partition B's spare capacity cannot
be transferred across the partition boundary, so aggregate headroom of `~3,000/s` is unusable by A.

**Red flag.** Computing `Σa = 12,790` against `Σs = 4,600` and concluding anything about recovery, or
proposing more consumers without naming the single-partition ceiling.

### 11.3 The committed offset is 41 and the last completed record was 41. What do you commit?

**Answer.** The question is malformed on purpose: offsets identify the *next* record to read, so if the
last completed record's offset is 41 then the correct commit is 42. If instead the committed value is
41 and records through 40 are durable, then 41 is already correct.

**Reasoning.** The commit is a resume position, not a receipt. Confusing the two yields off-by-one
replay of one record per commit, which is harmless with idempotency and catastrophic without it — and
which produces a permanent, mysterious duplicate on every restart.

**Red flag.** Answering "41, because that's the last one done" without asking which convention the
store uses; the candidate has memorised a number rather than a contract.

### 11.4 Offsets 10 and 12 completed, 11 is in flight. What is the safe commit, and when does it move?

**Answer.** 11. It moves to 13 only when 11 completes, because the tracker retires marks in offset order
and stops at the first hole. If 11 never completes, the frontier stays at 11 indefinitely, and that stall
is the correct behaviour.

**Reasoning.** From `10.2`, the invariant is that no commit exceeds the smallest unfinished offset.
Committing 13 would declare 11 done and lose it on restart.

**Red flag.** Saying "commit 13 and rely on the retry", or "commit the minimum of all in-flight plus one"
without noticing that the minimum of `{11, 12}` is 11 but 12 being complete does not license 13.

### 11.5 Why is lag measured in offsets a misleading alert?

**Answer.** Because it is not the objective. A partition with 5,000,000 offsets of lag at 12,000
records/s is ~7 minutes behind; a partition with 5,000 offsets where one record is eight hours old is a
severe freshness breach. Alert on the age of the oldest record at the *last* stage, and keep offset lag
as a diagnostic secondary signal.

**Reasoning.** `Q-03` converts lag to time using the arrival rate, and the conversion factor varies by
orders of magnitude across partitions and traffic regimes. The business cares about staleness, which is a
time quantity.

**Red flag.** "Lag is lag." The candidate has not distinguished work-in-queue from staleness, and will
mis-prioritise during an incident.

### 11.6 Compute the drain time: 360,000 records, 900/s arriving, 1,200/s served.

**Answer.** `360,000 / (1,200 − 900) = 1,200 s = 20 minutes`. Check: completions `1,200 × 1,200 =
1,440,000`; arrivals `900 × 1,200 = 1,080,000`; difference `360,000`. ✓

**Reasoning.** Net drain is the surplus, not gross service. Dividing by 1,200 gives 300 s and is wrong
by 4×, because it assumes no arrivals during recovery.

**Red flag.** Dividing by gross service; or not asking whether 1,200/s is measured or nominal. The
nominal figure is often the sum of per-thread ceilings that the sink cannot actually sustain.

### 11.7 A candidate says "add consumers, lag is growing". What do you ask?

**Answer.** Three things in order: (1) how many partitions versus members, (2) is one partition
disproportionately loaded, (3) is the sink at its own ceiling. If `N ≥ P`, no members are idle-able; if
one partition dominates, members cannot help it; if the sink is saturated, more members make it worse.

**Reasoning.** `min(N,P)` bounds group parallelism, and the clamp inside the sum means the binding
partition governs. Sink saturation converts added concurrency into retries and errors.

**Red flag.** Agreeing immediately, or asking only about CPU. The candidate is pattern-matching a
familiar remedy rather than locating the constraint.

### 11.8 How do you achieve exactly-once effects in a relational database?

**Answer.** You don't get it from the offset commit. You get it by making the effect idempotent: a
unique constraint or deduplication row keyed on a stable logical event id, written in the same
transaction as the business mutation. Replay then re-attempts and is absorbed. The dedup retention must
cover the maximum replay horizon.

**Reasoning.** A crash between the database commit and the offset commit is unavoidable in a two-system
write without distributed transactions. Idempotency converts the at-least-once delivery into
effectively-once application.

**Red flag.** "Enable `enable.idempotence` and you're done", or "use `read_committed`". The first is
producer-side broker deduplication only; the second affects Kafka-internal transactions.

### 11.9 What breaks when you increase the partition count from 8 to 16?

**Answer.** `key mod 8` becomes `key mod 16`, so roughly half of all keys change partition. Any entity
whose records previously shared one partition may now be split across two, and two consumers can process
its records concurrently — per-entity order is lost at the cutover unless the migration defines a
barrier or the sink applies version conditionals.

**Reasoning.** The partitioner's target depends on the partition count, which is part of the key-to-lane
function. Changing `P` changes the function's output for most inputs.

**Red flag.** "Ordering is preserved because the key is the same." The key is the same; the mapping is
not. This candidate will cause a production ordering bug.

### 11.10 The consumer is at 20% CPU and lag is rising. Diagnose.

**Answer.** Low CPU with rising lag means the consumer is waiting, not computing. Check in order: pool
wait and sink latency (blocked on I/O); in-flight limit reached with a slow sink; poll cadence versus
`max.poll.interval.ms` with rebalance churn consuming the gaps; paused partitions never resumed; and
fetch-side throttling or broker-side quota.

**Reasoning.** CPU is a lower bound on work, not a measure of it. A blocked thread consumes no CPU, so
the profile of "idle consumer with growing backlog" always points downstream or at a stop-the-world
event, never at the deserialiser.

**Red flag.** "Scale out" — with 20% CPU the member has capacity; the constraint is elsewhere, and more
members multiply the blocking pressure.

### 11.11 Why can lag be zero while records are missing? Give two mechanisms.

**Answer.** (1) The application commits offsets before its effect is durable, so the resume point has
advanced past work that never happened. (2) Failures are routed to a dead-letter path and the offset is
committed after the routing attempt rather than after a durable dead-letter write, so nothing retains
the record. A third: the group id was changed, so the "zero lag" belongs to a fresh group that started
at `latest` and never consumed the interval.

**Reasoning.** Lag is metadata-only (`Q-02`), computed from Kafka's own numbers. It cannot observe an
external effect, a DLQ failure, or a group id change.

**Red flag.** Treating zero lag as a correctness certificate; such a candidate would miss data loss
entirely and report success.

### 11.12 Two dashboards disagree by 2,000 offsets on the same partition. Why?

**Answer.** Likely different end boundaries — LEO versus LSO under an open transaction (`Q-13`) — or
different isolation levels. Next most likely is a scrape-time difference on a moving target. Least
likely but possible: one reports a sum and the other a maximum across partitions.

**Reasoning.** All three boundaries are legitimate; they answer different questions. Only a single
snapshot with all three computed from raw numbers resolves which one a dashboard used.

**Red flag.** "One of them is broken." Naming a bug before naming the definition is guessing; the two
numbers may both be right.

### 11.13 How would you design a replay that cannot contaminate live output?

**Answer.** Separate group id, separate sink principal with no write access to live external systems, a
required explicit destination configuration with no default, event ids carried through unchanged, and a
mode flag that is enforced at the *permission* layer rather than at one call site. Replay into a staging
schema, reconcile against production, then promote through a reviewed operation.

**Reasoning.** A flag at a call site fails open when a new effect type is added; permissions fail closed.
Carrying original ids is what makes idempotency recognise replayed records as the same logical events.

**Red flag.** "Set `dryRun=true`." One boolean in application code is not a boundary, and the candidate
has not thought about which effects bypass it.

### 11.14 You have 24 hours of retention and a 30-hour backlog. What are your options?

**Answer.** Four, in order of preference. (1) Raise retention now, before the oldest segment deletes, if
storage permits. (2) Raise effective service so the drain fits inside the remaining window, checking the
`Q-15` sensitivity. (3) Prioritise the oldest partitions to move the log start boundary out of the
danger zone for the required keys. (4) Accept a documented gap and reconstruct from an archive, the
producer's outbox, or a downstream replica — with an approver and a written id list.

**Reasoning.** `10.7`: retention is a clock that competes with the drain, and it can bind first even
when throughput is fine. Option (4) is not a fix; it is an accepted, recorded loss.

**Red flag.** "Reset to latest and move on", which converts the problem into silent permanent loss, or
"delete the consumer group", which abandons offsets without a target decision.

### 11.15 The same event causes two payments after a restart. Walk the diagnosis.

**Answer.** (1) Compare the two effects' source coordinates. Same partition+offset means one record was
processed twice → the commit followed the effect, or the effect was not idempotent. (2) Different
coordinates with identical business fields means the producer emitted twice → check producer retries and
idempotence, or a double-submitting UI. (3) Same coordinates and different idempotency keys means the key
is derived non-deterministically → inspect key generation. Then fix the specific layer, not all three.

**Reasoning.** This is `8.8` applied. The discriminator is the source coordinate because it separates
"consumer duplicated" from "producer duplicated" with one query.

**Red flag.** "Add a retry limit." Retry limits do not cause duplication in this scenario, and the
candidate has not located the layer that repeated the operation.

**Takeaway:** Strong answers always name the mechanism, the discriminating measurement, and the boundary at which the mechanism stops applying.

## 12. Quick Reference Card

| Field | Content |
|---|---|
| **Core rule** | Per-partition surplus decides recovery (`s_i > a_i`), and the commit frontier never passes the first unfinished offset. |
| **Critical validity condition** | Rates and backlogs must share one partition, one unit and one interval; offset distance equals record count only over a gap-free range. |
| **Key formulas** | `T_drain = B/(s − a)` for `s > a`; `L_i = E_i − C_i`; `T_behind = L_i/a_i`; `N_req = ceil(Σa·(1+h)/s_member)`; `K = a·W`; `E[replay] = a·τ/2`. |
| **Complexity** | Tracker: `O(R)` time and `O(K)` space over `R` records with `K` in-flight; group parallelism capped at `min(N, P)`. |
| **Use when** | Diagnosing which of four resources binds — skew, assignment, per-record cost, or sink — and choosing a reversible lever. |
| **Avoid when** | Aggregate lag is the only signal, or the sink's effect count is unverified; both mislead in opposite directions. |
| **Diagnostic order** | Deficit sign per partition → measurement reconciliation → shape (spike / staircase / plateau) → binding resource → reversible lever → verification. |
| **Primary failure risk** | Committing `position` instead of the safe frontier, and adding consumers against a saturated sink. |
| **Evidence status** | All workloads ILLUSTRATIVE; all mechanisms DERIVED from the partitioned-log model. Review status OPEN; version-sensitive behaviour must be re-checked against the deployed broker and client. |
| **Go deeper** | [Collection index](INDEX.md) → CDC ordering and snapshot handoff, then Flink event-time state; verify against the [Kafka documentation](https://kafka.apache.org/documentation/) for the deployed release. |
