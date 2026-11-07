---
collection: "data-streaming-event-systems"
file_id: "03"
topic: "Event-time session windows and stateful processing"
level: 3
status: "draft"
version: "1.0"
created: "2026-11-07"
reviewed: null
review_trigger: "Flink, connector, state serializer, event-time policy, or sink correction contract changes"
notation_style: "plain-text"
prerequisites: ["01_Kafka_Consumer_Partitions_and_Lag.md", "02_CDC_Ordering_Snapshot_and_Materialization.md"]
source_status: "OPEN"
owner_role: "Streaming platform and analytics owners"
source_index: "INDEX.md"
---
# Flink Event-Time Session Windows and Stateful Processing

## 1. File Header

> **Purpose:** Design and diagnose event-time session aggregates that remain interpretable under disorder, late arrivals, recovery, and correction.
> **Prerequisites:** [Kafka partitions and lag](01_Kafka_Consumer_Partitions_and_Lag.md), [CDC materialization](02_CDC_Ordering_Snapshot_and_Materialization.md), timestamp arithmetic, keyed state, and basic database transactions.
> **Outcome:** Given timestamped events and an explicit window contract, trace membership, watermark-triggered output, a late merge, retention, and downstream correction.
> **Exclusions:** This is not a connector configuration recipe, a CEP fraud implementation, or proof of exactly-once effects in an arbitrary sink.
> **Level:** Advanced/production; **version / review:** 1.0 / not reviewed.
> **Evidence:** Versioned Apache documentation is SOURCED; the operational design and numbers are ILLUSTRATIVE or DERIVED, not measured.
> **Retrieval cue:** Missing or contradictory sessions: Sections 5 and 8; restore and correction: Sections 4, 8 and 9.

**OPEN — source provenance:** The inventory names `Case_Studies_Streaming_Kafka_Flink_Redpanda/Flink_Stateful_Processing/02_IoT_Telemetry_Analytics_Session_Windows.md`.
The original source case directory is unavailable in this workspace.
Its title is only a topic lead, not evidence of a device fleet, architecture, incident, or outcome.
The [source crosswalk](../../../00_SOURCE_COVERAGE_EDGE_INDUSTRY.md#flink) requests a timestamp trace; this sheet constructs one independently.
No implementation has been run, no throughput measured, and no production outcome established.
The arithmetic is checked by a separate interval-connectivity method in Section 5, not by an executed Flink job.

**Decision context:** A hypothetical equipment-monitoring team groups a device's telemetry into activity sessions.
Operators use session summaries to investigate abnormal operating periods, not as an automatic safety interlock.
A session can appear complete and subsequently be connected to another session by a delayed event.
The decision is whether the consumer can accept revisions and how long state and correction rights are retained.
If it cannot accept corrections, a later finalization or batch reconciliation is needed.
This tradeoff cannot be resolved by claiming a watermark proves all events have arrived.

**Takeaway:** The application contract, not a framework name alone, defines what an emitted session means.

## 2. Concept Map / Mental Model

```text
Physical event -> immutable ID + entity key + occurrence timestamp + payload
                           |
                 validate timestamp and identity
                           |
              source split watermark / idleness
                           |
                key partition -> deduplicate
                           |
            assign interval -> merge active sessions
                           |
      watermark -> fire / retain for lateness -> cleanup
                           |
        versioned replacement + explicit obsolete-ID removals
                           |
            recoverable materialization / reconciliation
```

Caption: The boxes describe a proposed logical pipeline; specific operator placement and sink features are implementation choices.
Event timestamp controls event-time grouping, whereas arrival and processing times describe observation and execution.
The minimum active upstream watermark can hold back output even while records continue arriving.
Deduplication must happen before a duplicate contributes to the aggregate unless the aggregate is intrinsically duplicate-insensitive.
A late event can alter both a session's aggregate and its identity.
The sink must learn which previously published identities are no longer valid.
A checkpoint restores internal state and input positions; it does not automatically undo arbitrary external effects.

**Takeaway:** Follow identity, event time, retained state, and external visibility through the whole graph.

## 3. Foundational Definitions

### 3.1 Event, key, and timestamp

An **event** is a logical observation with an immutable source identity and a payload.
An **event ID** identifies a logical observation across transport retries, assuming the producer preserves it.
A **delivery** is one observed transport instance and can be repeated for the same event ID.
An **entity key** identifies the device or other subject whose observations may share a session.
Keys must use a documented tenant namespace to avoid joining unrelated devices with the same local ID.
The key is a logical partitioning attribute, not necessarily a physical source partition number.
The **event timestamp** `t` is the occurrence time assigned to the observation, in epoch milliseconds.
An observed device clock can be wrong even when the timestamp field is present and parseable.
**Processing time** is the worker's wall-clock time when it executes an operation.
**Ingestion time** is a separate time marking entry into some defined ingestion boundary.
An event with `t=4,000 ms` processed at wall-clock time 20 seconds is not an event at 20 seconds.
The timestamp extractor, timezone interpretation, units, and malformed-value policy are part of correctness.

### 3.2 Progress and disorder

A **watermark** `w` is a progress assertion in event-time units, not a wall-clock timer.
It expresses an expectation about event-time completeness, not proof that older records cannot arrive.
Events whose timestamps decrease in arrival order are **out of order**.
An event can be out of order without arriving after the relevant window has fired.
A **late event**, in the window context, arrives after the window's event-time firing threshold has been reached.
Some late events can still be accepted while the relevant state is retained.
An event timestamp older than the current watermark is not alone a complete test of session acceptance.
A new session interval can merge with existing retained windows and change the effective end.
A **watermark strategy** defines timestamp assignment and watermark generation in Flink's DataStream API [S1].
An upstream input marked **idle** is excluded from the active-input minimum until it resumes [S1].
Idleness is a processing-time observation about missing traffic, not evidence that the input has no older event-time records.
A resumed input can therefore produce records that are late relative to downstream progress.

### 3.3 Session and lateness

A **session gap** `g` is a positive duration used to turn an event at `t` into a candidate interval.
The illustrative contract in this sheet uses the half-open interval `[t, t+g)`.
A **session** is a connected component of candidate intervals for the same key under the stated merge rule.
Our generic trace merges candidate intervals only when they overlap with positive duration.
It does not claim that every Flink session assigner uses that equality rule.
The start of a component is the smallest candidate start and its end is the largest candidate end.
The end includes an inactivity allowance; it is not the timestamp of the last observed event.
A **trigger** decides when a retained window invokes its calculation.
A **fire** is an invocation, not a guarantee that the window is permanently complete.
A **purge** removes window contents according to trigger semantics; it is distinct from all metadata cleanup [S2].
**Allowed lateness** `L` is a retention and acceptance policy relative to a window boundary.
Its exact cleanup and too-late comparisons must be specified for the implementation and timestamp resolution.
The trace's cleanup rule is a deliberately explicit generic contract, not a universal Flink claim.
A **side output** is a separate stream for records handled outside the main output path; configure it explicitly where supported [S2].

### 3.4 State, checkpoints, and effects

**Keyed state** stores per-key information associated with a keyed stream [S3].
Window state can include aggregate values, candidate intervals, timer metadata, and merge bookkeeping.
A **checkpoint** records recoverable source positions and operator state under the configured fault-tolerance mode [S3].
A **savepoint** is an operationally managed snapshot; portability and restore compatibility still need validation [S3].
A **replay** rereads input after a recovered source position.
A **duplicate** is a repeated delivery of the same logical event, not necessarily a record with identical payload.
An **upsert** replaces the value for a key in a materialized view.
A **retraction** invalidates an earlier output; it is not implied by emitting a replacement under a different key.
A **tombstone** is one possible representation of removal, whose meaning depends on the sink contract.
A **revision** orders successive descriptions of a logical output within a defined namespace.
A **finality policy** states when ordinary online corrections stop and how exceptions are reconciled.
Finality under a finite lateness policy is operational finality, not proof that no older physical event exists.

### 3.5 Notation

| Symbol | Meaning | Units/type | Scope and warning |
|---|---|---|---|
| `t` | Event occurrence timestamp | integer epoch ms | Must be comparable across candidate events |
| `p` | Processing-time observation | wall-clock ms | Not a substitute for occurrence time |
| `g` | Session gap | positive ms | Merge-at-equality policy must be explicit |
| `w` | Current input watermark | integer epoch ms | Monotone progress estimate, not completeness proof |
| `L` | Allowed lateness in the stated contract | nonnegative ms | Different from out-of-orderness bound |
| `[a,b)` | Session interval | integer-ms boundaries | Inclusive start and exclusive end |
| `m` | Maximum timestamp in `[a,b)` | `b-1 ms` | Integer millisecond representation |
| `c` | Cleanup threshold in the generic trace | `m+L` ms | The trace removes state at `w >= c` |
| `r` | Input rate | events/s | A scenario assumption, not a benchmark |
| `mu` | Sustainable processing rate | events/s | Must be measured under replay conditions |
| `B` | Replay backlog | events | Includes only the specified source positions |
| `K` | Number of active keys | count | Key count does not imply session count |
| `N` | Retained session count | count | Skew can dominate per-task load |

`W` in the CDC sheet denotes a database snapshot boundary; it is not the watermark `w` here.
A millisecond is a unit of timestamp resolution, not a claim about device clock accuracy.

**Takeaway:** Watermark, event timestamp, allowed lateness, and source offset answer different questions.

## 4. Core Formulas with Full Derivation

### 4.1 Simplest / Base Case

**Validity:** One key, integer-millisecond timestamps, positive fixed gap, and the generic strict-overlap contract below.
**Mechanism:** Assign each event to `[t,t+g)`, then replace each connected set of positively overlapping intervals with its union.
A connected set can contain intervals that do not overlap each other directly if intermediate intervals bridge them.
The aggregate must support merging if the implementation incrementally merges session state.
For a count, the merge operation sums counts after deduplication.
A mean must retain both sum and count; averaging already averaged session means is generally incorrect.

```text
[1] FORMULA: candidate(t) = [t, t + g)
[2] WHERE: t is the event timestamp in ms; g is the positive gap in ms.
[3] CONSTRAINT: fixed gap, one key, and the stated half-open interval convention.
[4] NUMERICAL EXAMPLE: t=4,000 and g=5,000 give [4,000,9,000) ms.
[5] INTERPRETATION: the candidate reaches 5 seconds beyond the event, not 5 seconds beyond processing time.
```

Derivation: the start is the event time by definition and the candidate duration is the configured gap.
Boundary: `g=0` yields an empty half-open candidate and is outside this contract.
Cross-key overlap is irrelevant because sessions are computed independently for each key.
An event exactly at the prior candidate's end is a policy-sensitive equality case.
The generic contract chooses a split at equality; a production Flink implementation must be tested against its version's actual assigner behavior.

### 4.2 General Case

**Validity:** All candidates for one key use the same fixed positive gap and strict-overlap merge rule.
Sort event timestamps as `t1 <= t2 <= ... <= tn`.
Start a new component when the next candidate does not overlap the current component.
Because each candidate has the same length, this is equivalent to checking successive timestamp gaps against `g`.

```text
[1] FORMULA: same component across neighbors iff t(i+1) - t(i) < g.
[2] WHERE: both timestamps and g are in integer ms, for the same entity key.
[3] CONSTRAINT: fixed gap, complete candidate set, strict positive-overlap rule.
[4] NUMERICAL EXAMPLE: 0, 4, 8, 12 seconds with g=5 seconds have gaps 4, 4, 4 seconds.
[5] INTERPRETATION: all four observations belong to one [0,17) second component.
```

Derivation: `[ti,ti+g)` overlaps the next interval exactly when `t(i+1) < ti+g`.
A chain of these overlaps connects all its observations even if its first and last candidates do not overlap.
The formula reduces to the base candidate when `n=1`.
For per-event dynamic gaps, compare interval endpoints directly; adjacent timestamp differences alone are insufficient.
For a correction or deletion, an existing component can split, so a merge-only accumulator may not retain enough information.

### 4.3 Key Special Cases / Variants

| Variant | Change and assumptions | Use when | Avoid or qualify when |
|---|---|---|---|
| Strict overlap | Equality splits; trace convention | Domain defines a gap of exactly `g` as inactivity | Deployed assigner merges touching intervals |
| Touching merge | Equality merges by explicit convention | Business contract defines `<= g` connectivity | Never silently substitute for strict overlap |
| Dynamic gap | Each event supplies a gap | Event class legitimately determines inactivity | User-controlled or invalid gaps can inflate state |
| Processing-time session | Worker-clock arrival groups events | Fast operational monitoring tolerates replay variation | Historical occurrence-time analysis is required |
| Event-time session | Timestamp and watermark control progress | Delayed arrival must be interpreted by occurrence time | Timestamps are untrustworthy and cannot be repaired |
| Incremental aggregate | Merge compact sufficient statistics | Aggregate is associative and its merge is implemented | Exact arbitrary deletion or split requires missing details |
| Full event buffer | Retain event-level data | Recompute and correction rights require the raw members | Per-key volume and retention can exhaust state |
| Early/custom trigger | Fire before event-time completion | Consumer accepts repeated provisional outputs | Trigger does not support merging or sink treats every fire as a new fact |

[S2] documents Flink event-time and processing-time session assigners, default triggers, merging requirements, and allowed lateness.
It does not establish that a particular sink retracts previously emitted sessions.
A custom trigger or evictor changes the trace assumptions and must be assessed separately.

### 4.4 Asymptotic / Limiting Behaviour and Counterexample

If every successive gap is strictly below `g`, a session can span an arbitrarily long time.
A gap defines inactivity, not a maximum session duration.
A permanently stalled watermark can retain event-time windows much longer than expected in wall-clock time.
If `L=0`, the generic cleanup threshold equals the firing threshold and there is no post-fire correction interval.
If `L` grows, more late data can be accepted, but state and correction exposure can also grow.
If a key's observations are missing, no choice of gap reconstructs them without an explicit imputation model.
If input contains duplicate IDs and deduplication is absent, a count is biased upward even with perfect window boundaries.
If one key receives all traffic, adding more key-partitioned workers does not automatically distribute that key's state.

| Invariant | Assumption | Violating example | Consequence / alternative |
|---|---|---|---|
| Merge preserves count | Disjoint deduplicated event sets | Same ID in both retained aggregates | Deduplicate before summing or retain identity-aware state |
| Sorted gap test is sufficient | Fixed gap and strict overlap | Dynamic gap varies by event | Compare complete candidate intervals |
| Output is current | Obsolete IDs are removed | Late bridge creates a new session ID | Publish explicit invalidations and replacement |
| State restores consistently | Recoverable checkpoint plus replayable source | Required source positions expired | Restore an earlier valid data source or reconcile from history |
| Cleanup bounds state | Watermark progresses | Active input watermark remains stalled | Diagnose input progress and idleness without fabricating completeness |

**Takeaway:** Session correctness depends on explicit interval, state, and correction invariants, including the equality boundary.

## 5. Worked Numerical Example

### 5.1 Input, provenance, and contract

**ILLUSTRATIVE and DERIVED:** A hypothetical sensor emits observations for key `D1`.
All timestamps are milliseconds relative to a convenient origin; the table also shows seconds for readability.
The device and observations are invented teaching inputs, not a measured workload.
The gap is `g=5,000 ms` and allowed lateness is `L=10,000 ms`.
Each event has a distinct ID and contributes one to a count; no duplicate is present in the main trace.
An event's candidate is `[t,t+g)` and strictly overlapping candidates merge transitively.
Touching candidates do not merge in this generic model.
The stored window is represented by `[start,end)`, count, and original event IDs for this small example.
The maximum included integer timestamp is `m=end-1 ms`.
A watermark reaching `m` fires the window once unless a later accepted change causes another eligible firing.
State is removed at watermark `w >= m+L` in this generic model.
A received candidate is merged with retained windows before its resulting cleanup deadline is evaluated.
If the merged result has already reached its cleanup deadline, the record is routed to the late-data path instead of modifying that result.
If an accepted merged window has `m <= w`, this generic contract fires it immediately; otherwise it waits for `w >= m`.
Existing published IDs remain visible until the correction bundle on the merged window's next firing.
This design intentionally specifies behavior that the Flink API does not supply as an automatic downstream retraction contract.
The trace has one active input watermark and processes each listed action in order.
No automatic watermark generation, early trigger, purge, asynchronous reorder, or checkpoint failure is assumed.

| Action | Event ID | Event timestamp | Candidate | Purpose |
|---:|---|---:|---|---|
| 1 | A | 0 ms | `[0,5000)` | First observation |
| 2 | B | 4000 ms | `[4000,9000)` | Extends first session |
| 3 | C | 12000 ms | `[12000,17000)` | Separate session before bridge |
| 4 | — | watermark 9000 ms | — | Fires first session only |
| 5 | D | 8000 ms | `[8000,13000)` | Late bridge between retained sessions |
| 6 | — | watermark 16999 ms | — | Fires merged session and correction |
| 7 | — | watermark 26999 ms | — | Cleans merged state |

### 5.2 State transition trace

**Before action 1:** No session state and no published output exist.
The initial watermark is below zero and never moves backward.

**Action 1 — A at 0 ms:**
The candidate is `[0,5000)`.
There is no retained same-key interval to merge.
State becomes `S1=[0,5000), count=1, IDs={A}`.
Its `m=4999 ms`; its generic cleanup threshold is `14999 ms`.
No output is emitted because the watermark is below the firing threshold.

**Action 2 — B at 4000 ms:**
B's candidate `[4000,9000)` overlaps S1 over `[4000,5000)`.
The positive-width intersection is 1,000 ms.
The merged session is `S1=[0,9000), count=2, IDs={A,B}`.
Its new firing timestamp is `8999 ms` and its cleanup threshold is `18999 ms`.
The old timer and merge metadata must be reconciled according to the chosen implementation.
No output is emitted in this action.

**Action 3 — C at 12000 ms:**
C's candidate `[12000,17000)` does not overlap `[0,9000)`.
The gap between the intervals is `[9000,12000)`, with duration 3,000 ms.
State now contains `S1=[0,9000), count=2` and `S2=[12000,17000), count=1`.
S2 has firing timestamp `16999 ms` and cleanup threshold `26999 ms`.
The event-time gap between B and C is 8,000 ms, exceeding the chosen 5,000 ms gap.
No output is emitted in this action.

**Action 4 — watermark advances to 9000 ms:**
For S1, `9000 >= 8999`, so the generic trigger fires.
For S1, `9000 < 18999`, so its state remains available for correction.
The application publishes provisional row `P1=(D1, [0,9000), count=2, revision=1)`.
For S2, `9000 < 16999`, so S2 does not fire.
Both windows remain in state; only P1 is visible downstream.
The watermark is not the event timestamp of a newly received observation.

**Action 5 — D at 8000 ms after watermark 9000 ms:**
D is out of order relative to C and its timestamp is behind the watermark.
Its candidate `[8000,13000)` overlaps S1 over `[8000,9000)`.
It also overlaps S2 over `[12000,13000)`.
Each intersection has positive duration 1,000 ms.
The transitive union is `[0,17000)` with `count=4` and `IDs={A,B,C,D}`.
The merged window's `m=16999 ms`, so `9000 < 16999`: the generic trigger waits.
The merged cleanup threshold is `26999 ms`, and `9000 < 26999`, so the event is accepted.
The previous P1 is still visible until a correction bundle is emitted under this example's contract.
At this point P1 is known to be incomplete and must be labeled provisional by the consuming application.
No second row is emitted merely because the merge occurred under this chosen contract.

**Action 6 — watermark advances to 16999 ms:**
The merged window reaches its firing threshold exactly: `16999 >= 16999`.
It remains retained because `16999 < 26999`.
The application emits an atomic correction bundle: invalidate P1 and upsert `P2=(D1,[0,17000),count=4,revision=2)`.
The distinct P1 and P2 boundaries make a plain upsert of P2 insufficient to remove P1.
After the bundle is visible, the materialized view contains P2 and not P1.
The atomic bundle is an application/sink design requirement, not a built-in promise of all Flink sinks.
If the sink cannot atomically apply the bundle, consumers need a documented intermediate-state and reconciliation policy.

**Action 7 — watermark advances to 26999 ms:**
The cleanup comparison is equality: `26999 >= 26999`.
The generic contract removes the merged window state and timer metadata.
P2 remains in the external materialized view; cleanup of internal state does not delete the output.
A later event at 8000 ms has candidate `[8000,13000)` and cleanup threshold `22999 ms` if no retained merge applies.
At watermark 26999 ms that isolated candidate is too late under this contract and is routed to the late path.
The late-path record retains ID, key, event time, observed watermark, and routing reason for reconciliation.
This statement is about the stated generic cleanup contract, not all Flink versions or all possible new candidate intervals.

### 5.3 Independent arithmetic and membership check

Sort the four event times: `0, 4000, 8000, 12000 ms`.
Subtract adjacent times: `4000, 4000, 4000 ms`.
Each difference is strictly less than `g=5000 ms`, so all four form one connected component.
The earliest candidate begins at `0 ms`.
The latest candidate ends at `12000+5000=17000 ms`.
The alternate sorted-gap argument independently gives `[0,17000)` and count 4.
The interval-intersection argument in action 5 reaches the same boundaries by a different route.
The maximum timestamp is `17000-1=16999 ms` and adding lateness gives `16999+10000=26999 ms`.
The checks are arithmetic and logical, not a Flink execution or a measured integration test.

### 5.4 Counterfactuals

Without D, sorted times are `0,4000,12000 ms` and the successive differences are `4000,8000 ms`.
The 8,000 ms separation exceeds 5,000 ms, so the original two sessions remain separate.
If D had arrived before the watermark, the same final membership would result without needing to retract P1.
If D had timestamp 9000 ms, its candidate touches S1 at 9000 ms under the generic strict rule.
That equality does not connect S1 in this model, although it overlaps S2; a different deployed equality contract can change the result.
If D were a duplicate delivery of B with the same immutable ID, deduplication would suppress its extra contribution.
If D had a new ID but an identical timestamp and payload, it would be a separate logical event unless a domain rule says otherwise.
If the timestamp were a future clock error, accepting it might advance a generated watermark and prematurely age other windows.
If downstream merely appends P2, the view incorrectly contains both P1 and P2.
If P1 was already used in an irreversible business action, the correction does not automatically undo that action.
If the input includes deletions, removing D can split the merged component and requires retained membership or recomputation.

### 5.5 Production interpretation

The example has one key, one input watermark, four records, a fixed gap, and no failure.
Real deployments have partition-specific disorder, clock drift, multiple watermarks, traffic bursts, and uneven key distributions.
A sample trace cannot establish how frequently late bridges occur.
A selected lateness allowance must come from measured delay distributions and a business correction policy.
The downstream owner must agree whether provisional rows, atomic replacements, and post-finality backfills are acceptable.
The safety consequence of displaying an incomplete session must be assessed outside this example.

**Takeaway:** A single late bridge can invalidate a previously published session identity even though every event is counted once.

## 6. Visual / Geometric Intuition

### 6.1 Candidate interval geometry

```text
Event-time axis, seconds (not to scale):
0       4       8       12      17
A [----------)
        B [----------)
                         C [----------)
                D [----------)
Before D: [0,9)       [12,17)
After D:  [0,----------------17)
```

Caption: Each 5-second candidate is half-open; the schematic emphasizes overlaps rather than precise character positions.
A and B overlap; D bridges B's component to C's component.
The session end is 17 seconds although the latest event occurs at 12 seconds.
A visual overlap at exactly an endpoint must be resolved by the chosen merge contract, not by the drawing.

### 6.2 Processing versus event time

```text
Processing order:   A(0s) -> B(4s) -> C(12s) -> watermark(9s) -> D(8s)
Event-time order:   A(0s) -> B(4s) -> D(8s) -> C(12s)
Output visibility:             P1=[0,9) ---- correction ----> P2=[0,17)
```

Caption: Parentheses contain event timestamps except for the explicitly labeled watermark.
The late event does not move the watermark backward.
The correction changes the materialized interpretation of earlier observations.
A wall-clock timeline is not implied by the spacing between characters.

### 6.3 State lifecycle

```text
candidate -> active -> fired but retained -> cleaned
                ^             |
                +-- merge ----+
                         |
                  replacement output
```

Caption: The merge arrow represents a logical transition under the chosen application contract.
A fire need not purge state, and a cleanup need not retract a previously published result.
A late merge can extend the event-time end, changing when a subsequent trigger becomes eligible.
A processing-time timeout is not interchangeable with an event-time cleanup threshold.

### 6.4 Active-input minimum

```text
Input A watermark:  30 s ---- 40 s ---- 50 s
Input B watermark:  10 s ---- 10 s ---- idle
Operator minimum:   10 s ---- 10 s ---- 50 s (A only)
```

Caption: This is a simplified two-input example with B excluded once declared idle.
If B resumes with an event at 8 seconds, the operator watermark cannot be rolled back to 8 seconds.
Whether that event is accepted depends on the target window and retained state.
Incorrectly declaring B idle can make output look timely while increasing late corrections.

**Takeaway:** Arrival order, event-time geometry, and output visibility are three different timelines.

## 7. Comparison / Selection Table

### 7.1 Window and timestamp choices

| Method | Preconditions | Guarantee or quality boundary | Cost / use | Disqualifier | Evidence |
|---|---|---|---|---|---|
| Batch sort and sessionize | History retained and sufficiently complete | Exact relative to the finite selected input and stated boundary | Sorting and delayed results; useful audit baseline | Decision cannot wait for batch | DERIVED |
| Processing-time sessions | Processing clock and arrival order acceptable | Groups actual processing activity, not historical occurrence | Simple operational signal; replay may change groups | Occurrence-time membership is required | SOURCED [S2] |
| Event-time sessions | Trustworthy timestamp and watermark policy | Repeatable membership only under matching input, ordering-sensitive policy, and correction contract | Retained state and delayed finality | Event times cannot be compared reliably | SOURCED [S1,S2] |
| Fixed tumbling windows | Fixed aligned period is meaningful | Period boundaries are fixed rather than activity-defined | Simpler grouping and stable IDs | Sessions crossing periods must remain whole | SOURCED [S2] |
| Dynamic-gap sessions | Gap policy is validated and version-supported | Per-event gap changes interval geometry | Flexible behavior at extra policy and test cost | Untrusted gap creates unbounded or misleading sessions | SOURCED [S2] |

### 7.2 Correction and recovery choices

| Method | Preconditions | What it can ensure | Cost / use | Disqualifier | Evidence |
|---|---|---|---|---|---|
| Append every firing | Consumers understand revisions and can reconcile | Preserves an audit log of emissions | More storage and query logic | Consumers sum all rows as independent sessions | DERIVED |
| Upsert only | Stable output key and idempotent revision handling | Latest value for the same key | Compact view | Merge changes the key and old keys cannot be deleted | DERIVED |
| Retraction plus replacement | Consumer supports invalidation and ordering | Removes obsolete session identities | Requires effect protocol and durable lineage | Sink has no deletion or no reliable reconciliation | DERIVED |
| Final-only after horizon | Business accepts waiting and bounded late policy | Avoids ordinary provisional updates | Increases decision delay; beyond-horizon data remains possible | Immediate decision needs a preliminary signal | DERIVED |
| Checkpoint-aware transaction | Connector and destination document compatible commit protocol | Can coordinate visible effects with recovery within their contract | Transaction and timeout overhead | Unsupported connector or nontransactional side effect | SOURCED [S4] |
| Idempotent conditional write | Stable identity and monotone revision, atomic compare/update | Replayed identical effect does not change final view | Sink-specific conflict handling | External action has no idempotency or compensation mechanism | DERIVED |

**Priority guide:**
1. Reject a design that cannot meet safety, privacy, source retention, or effect-correction requirements.
2. If immediate results are unnecessary and full historical inputs exist, use a finite batch baseline for auditability.
3. If occurrence-time sessions are required, use event time only after validating clock and key contracts.
4. Choose final-only output when downstream cannot safely interpret provisional corrections and delay is acceptable.
5. Choose provisional output only with explicit invalidation and durable correction handling.
6. Choose watermark and lateness policies from observed delay and business tolerances, not a universal default.
7. If constraints conflict, escalate the freshness-versus-correctness decision with representative measurements.

**Benchmark status:** NOT MEASURED.
A valid comparison would pin Flink and connector builds, hardware, topology, event-time disorder, key skew, state backend, sink contract, workload, and repetitions.
Report session correctness, late-event handling, checkpoint duration, recovery time, resource use, and end-to-end freshness together.
Do not infer performance superiority from these conceptual rows.

**Takeaway:** Select by clock validity and correction capability before optimizing throughput or freshness.

## 8. Failure Modes & Edge Cases

All entries are **HYPOTHETICAL** diagnostic patterns, not observed incidents.
The standalone collection Common Issues artifact is planned and has no assigned completed issue IDs.
A diagnostic result narrows a hypothesis; it does not establish causation without correlated evidence.

### F01 — Watermark stalls on an idle split

**Symptom:** Window output stops while some partitions continue receiving records.
**Competing causes:** A silent input holds the minimum, an upstream watermark generator is broken, or processing is backpressured.
**Discriminator:** Compare per-input event arrivals and watermarks with downstream watermark and task throughput.
An idle partition with a frozen watermark and healthy active peers supports the idleness hypothesis.
If all input watermarks advance but output does not, inspect the trigger and operator execution instead.
**Action:** Confirm that silence is normal, then configure a version-supported idleness policy and measure resumed-input lateness.
**Risk:** An aggressively short idle timeout can advance the downstream watermark past legitimate delayed records.
**Verify:** Replay a silent-then-resuming split and inspect both output and late-data handling.

### F02 — Watermark jumps because a device clock is in the future

**Symptom:** Many ordinary events suddenly arrive behind the watermark.
**Competing causes:** A future timestamp, a changed extractor, or a newly aggressive watermark bound.
**Discriminator:** Inspect the maximum assigned timestamp per source split and compare it with independently trusted receipt time.
A single extreme timestamp immediately before the jump supports a timestamp-quality problem.
A correct future timestamp can still be expected for a legitimately scheduled event, so apply a domain-specific rule.
**Action:** Quarantine implausible clock values using a documented policy and preserve original payloads for investigation.
**Risk:** A tight filter can discard valid delayed uploads or devices with known clock offsets.
**Verify:** Inject an extreme timestamp and confirm it cannot silently contaminate the progress signal.

### F03 — Late data disappears without visibility

**Symptom:** Batch reconciliation finds events absent from online session counts.
**Competing causes:** Post-cleanup arrival, filtering, identity collision, source loss, or sink failure.
**Discriminator:** Compare source IDs, window late-data side output, filter counters, and sink acknowledgments.
An ID present in the late path but absent from the main result supports the post-cleanup explanation.
An ID missing from both requires tracing earlier than the window operator.
**Action:** Enable or repair the documented late path and choose a correction or backfill policy with the owner.
**Risk:** Increasing lateness blindly increases retained state and does not restore source data already lost.
**Verify:** Deliver an event on each side of the configured cleanup boundary and reconcile all IDs.

### F04 — Two sessions remain after a late bridge

**Symptom:** The sink displays overlapping or obsolete session rows after a delayed event arrives.
**Competing causes:** Merge state was lost, correction was not emitted, the sink ignored a deletion, or revisions arrived out of order.
**Discriminator:** Compare operator output and sink change log for the obsolete ID and the replacement ID.
A correct correction bundle at the operator but missing delete at the sink isolates the external effect path.
A missing bundle before the sink points to application lineage or trigger behavior.
**Action:** Add explicit invalidation and idempotent version-aware materialization, or rebuild from a canonical event log.
**Risk:** Deleting old rows before safely publishing the replacement can temporarily undercount.
**Verify:** Replay the Section 5 bridge with failure injected between removal and replacement.

### F05 — Duplicates increase counts after restart

**Symptom:** Event count rises after recovery without matching new logical events.
**Competing causes:** Source duplicate delivery, non-idempotent sink replay, identity collisions, or an incorrectly restored dedup state.
**Discriminator:** Trace logical event IDs and effect IDs across the checkpoint boundary and sink commit log.
Duplicate source IDs with two aggregate contributions implicate deduplication; one internal contribution with two sink effects implicates the sink.
**Action:** Restore durable deduplication or sink idempotency according to the diagnosed boundary.
**Risk:** A payload hash can suppress legitimate equal-valued observations; use producer-defined identity where possible.
**Verify:** Fail after an output effect and before checkpoint completion, then compare logical counts after replay.

### F06 — Session never closes

**Symptom:** A key has an unexpectedly long session or retained state grows continuously.
**Competing causes:** Frequent legitimate activity, a stalled watermark, wrong time units, or a key combining unrelated entities.
**Discriminator:** Plot sorted event gaps for the key and compare event-time progress with operator watermark.
If all successive gaps are below `g`, an indefinitely extended session is consistent with the model.
If event time advances while the watermark does not, inspect active input progress instead.
**Action:** Correct the key, timestamp unit, or watermark issue; add a maximum duration only if the business contract requires it.
**Risk:** A forced maximum duration changes the definition of a session.
**Verify:** Exercise long continuous activity and a real inactivity interval separately.

### F07 — A corrected or deleted observation should split a session

**Symptom:** Removing a bridging event does not split a formerly merged session.
**Competing causes:** Append-only window operator, missing raw membership, or unsupported retraction semantics.
**Discriminator:** Recompute the affected key from event-level history after removing the bridge.
If the recomputed connectivity has two components but the incremental state has one, a split-capable representation is absent.
**Action:** Rebuild the affected key from retained history or implement explicit membership-aware dynamic connectivity.
**Risk:** Aggregate-only state cannot generally infer which previous member connected two regions.
**Verify:** Add and then remove D from the Section 5 set; expect two components under the strict rule.

### F08 — Recovery fails despite a completed checkpoint

**Symptom:** Restore fails or cannot replay all needed inputs.
**Competing causes:** Missing checkpoint files, serializer incompatibility, source retention expiry, incompatible operator IDs, or credentials.
**Discriminator:** Restore the exact artifact into isolation and inspect the first concrete failure at state load or source seek.
A failed source seek after successful state load differs from a serializer incompatibility before source processing.
**Action:** Preserve evidence and the last usable snapshot; restore a compatible build with retained inputs or initiate a documented rebuild.
**Risk:** Repeated restarts can overwrite or obscure useful evidence and may repeat nontransactional effects.
**Verify:** Perform a clean restore drill with real retention and access-control policies.

### F09 — Hot key or session state exhausts resources

**Symptom:** One task accumulates disproportionate state or becomes the bottleneck.
**Competing causes:** One high-volume key, extreme gap/lateness, stalled watermark, or large buffered payloads.
**Discriminator:** Compare state, record rate, watermark, and key distribution by subtask and selected key.
A single key accounting for most work cannot generally be spread by adding workers while preserving simple keyed ordering.
**Action:** Fix the data contract or redesign the aggregation with a proven merge strategy and correction model.
**Risk:** Arbitrary key salting can split sessions and make late bridges incorrect.
**Verify:** Load-test the observed skew and compare reconstructed sessions with a finite reference implementation.

### F10 — Equality boundary disagrees with the business rule

**Symptom:** Events exactly one gap apart sometimes appear as one session when two were expected, or vice versa.
**Competing causes:** Inclusive versus strict interval merging, unit conversion, or rounding.
**Discriminator:** Test timestamps at `g-1 ms`, `g`, and `g+1 ms` using the deployed assigner and exact output.
A mismatch only at `g` isolates the equality convention rather than general disorder.
**Action:** Document the required boundary and use a compatible assigner or explicit application logic.
**Risk:** Shifting timestamps by one millisecond as an undocumented patch creates new edge cases.
**Verify:** Keep the three boundary examples in the version-upgrade acceptance suite.

### F11 — Deduplication TTL expires too soon

**Symptom:** A very late replay is counted again despite deduplication.
**Competing causes:** Expired identity state, IDs regenerated by producers, or an uncheckpointed dedup store.
**Discriminator:** Compare event identity retention with the oldest duplicate delivery and permitted replay horizon.
If the identity expired before replay, a finite TTL cannot support the desired horizon.
**Action:** Set a measured and contractually defensible identity horizon, or use a durable ledger and reconciliation.
**Risk:** Extending identity retention costs state and still cannot protect against changed IDs.
**Verify:** Replay the same ID just before and just after the selected expiry boundary.

### F12 — Checkpoint durations grow under backpressure

**Symptom:** Checkpoints time out while state or queueing increases.
**Competing causes:** Barrier alignment delay, snapshot storage throttling, large state, or a slow sink.
**Discriminator:** Separate alignment duration, snapshot upload duration, per-task backpressure, and checkpoint bytes.
High alignment with a slow downstream task differs from slow upload with little alignment delay.
**Action:** Address the measured bottleneck; evaluate unaligned checkpoints only where the documented constraints fit [S3].
**Risk:** Capturing in-flight buffers may increase snapshot I/O and does not cure an overloaded storage backend.
**Verify:** Compare checkpoint and restore behavior under a representative controlled backpressure test.

### F13 — A non-mergeable trigger breaks session behavior

**Symptom:** Session window configuration fails or trigger state is wrong after a merge.
**Competing causes:** A custom trigger does not support merging, timer state is not merged, or purge semantics were misunderstood.
**Discriminator:** Inspect the trigger's documented merge contract and test two windows connected by a bridge.
A fixed-window trigger with no merge implementation is not automatically suitable for session windows [S2].
**Action:** Use a merge-capable trigger or implement and test merged trigger state under the pinned release.
**Risk:** Purging on an early fire may remove data needed for subsequent corrections.
**Verify:** Exercise merge before fire, after fire, and at a timer boundary.

### F14 — Rescaling changes state ownership but not the business key

**Symptom:** Counts or output revisions conflict after a topology change.
**Competing causes:** State restore incompatibility, a changed key selector, nondeterministic IDs, or independent sink writers.
**Discriminator:** Compare operator identifiers, serializer snapshots, key derivation, and revision ownership before and after restore.
A changed key selector is a semantic migration, not merely a change in parallelism.
**Action:** Rehearse a supported state migration and enforce a sink conflict-resolution contract.
**Risk:** Starting from empty state can produce plausible but incomplete current sessions.
**Verify:** Restore a known checkpoint at the intended parallelism and reconcile a bounded event trace.

### F15 — Duplicate event ID carries different payloads

**Symptom:** A duplicate is suppressed but source and materialized data disagree.
**Competing causes:** Producer ID reuse, legitimate correction represented as a duplicate, or corruption.
**Discriminator:** Store or audit a digest and source version for repeated IDs, subject to privacy requirements.
Same ID with a changed payload is a conflict requiring a policy, not automatically a harmless retry.
**Action:** Quarantine conflicts or handle a versioned correction explicitly.
**Risk:** Last-arrival-wins can be wrong under reordering and can hide producer defects.
**Verify:** Send identical and conflicting duplicate IDs and check the two distinct paths.

### F16 — All inputs become idle

**Symptom:** Operators expect wall-clock progress but event-time windows do not close.
**Competing causes:** No active watermark, a source shutdown, or an incorrectly configured terminal progress signal.
**Discriminator:** Inspect input activity states and actual emitted watermarks, not just elapsed wall time.
No events for an hour does not itself establish an event-time watermark advancing by an hour.
**Action:** Choose a documented bounded-input completion or domain heartbeat policy if justified.
**Risk:** Artificial progress can classify later legitimate telemetry as late.
**Verify:** Simulate a long outage followed by historical device uploads.

### F17 — Wrong timestamp extractor

**Symptom:** Sessions follow broker arrival time rather than device occurrence time.
**Competing causes:** The source record timestamp was used by default, the wrong field was parsed, or a unit conversion failed.
**Discriminator:** Select a record with intentionally different device, broker, and processing timestamps.
Compare the timestamp seen by the window operator with the intended occurrence timestamp.
**Action:** Specify the extractor, timezone, precision, and invalid-value policy in the source contract.
**Risk:** Reprocessing with a corrected extractor can change historical window IDs and require a materialization rebuild.
**Verify:** Test a delayed event whose three candidate timestamps differ materially.

### F18 — Corrections arrive out of order

**Symptom:** An older provisional revision overwrites a newer corrected session.
**Competing causes:** Asynchronous sink writes, retry reorder, multiple writers, or revision reset after restore.
**Discriminator:** Compare the sink's stored revision with the source effect sequence and completion timestamps.
A later-arriving lower revision identifies a missing conditional-write or ordering guarantee.
**Action:** Apply version-aware conditional updates and tombstone obsolete IDs with durable effect identity.
**Risk:** Revision numbers need a stable generation and recovery rule; per-process counters alone can repeat.
**Verify:** Deliberately deliver revision 2 before revision 1 and replay both.

### F19 — Wrong assumption about checkpointed side effects

**Symptom:** An alert or API call is sent twice after a successful restore.
**Competing causes:** External side effect happened before checkpoint completion or sink retry is not idempotent.
**Discriminator:** Correlate checkpoint completion, call IDs, and the external service's receipt log.
A repeated call after rollback with the same logical effect demonstrates the effect boundary.
**Action:** Use a supported transactional connector, an idempotency key accepted by the external service, or a durable outbox/compensation design.
**Risk:** A local flag alone can roll back with the checkpoint while the external action remains committed.
**Verify:** Inject failure immediately after the remote acknowledgment and before checkpoint completion.

### F20 — Cross-key contamination

**Symptom:** One session contains observations from unrelated devices or tenants.
**Competing causes:** Missing tenant namespace, mutable key derivation, or a faulty join.
**Discriminator:** Trace original tenant and device fields alongside the computed keyed-stream key.
A collision in the composite identity explains merging without requiring a windowing bug.
**Action:** Correct identity derivation and rebuild affected materializations.
**Risk:** Changing the key changes state ownership and may require a migration, not just a live configuration flip.
**Verify:** Test identical local device IDs in two tenants through the full pipeline.

**Takeaway:** Separate input completeness, watermark progress, window state, and sink effects before choosing a fix.

## 9. Complexity & Cost Analysis

### 9a. Algorithmic complexity

Define `n` as the number of retained events for one key, `N` as its retained candidate/session intervals, and `k` as intervals actually merged by one arrival.
The following bounds describe possible reference algorithms, not Flink implementation guarantees.
A batch oracle can sort `n` timestamps in `O(n log n)` comparison time and scan them in `O(n)` time.
It needs `O(n)` stored timestamps if all are held in memory; external sorting has different I/O costs.
A simple online reference can scan all `N` intervals in `O(N)` time per arrival and use `O(N)` interval storage.
An ordered interval index may reduce search cost, but its exact bounds depend on the data structure and the number of overlaps traversed.
Merging `k` compact count aggregates requires at least inspecting or incorporating the `k` selected states in the stated algorithm.
A full-membership representation additionally stores event identities and may need to process them for exact splits.
An aggregate-only representation uses less information but cannot generally reconstruct event-level membership after a deletion.
Checkpoint cost includes state serialization, metadata, storage transfer, possible in-flight buffers, and coordination.
Restore cost includes reading state, reinitializing operators, fetching replay data, and reapplying sink effects safely.
Network shuffle and skew may dominate CPU complexity; no asymptotic expression predicts production latency by itself.

| Operation | Input sizes | Reference cost | Space / I/O boundary |
|---|---|---|---|
| Batch session oracle | `n` timestamps per key | `O(n log n)` sort plus `O(n)` scan | `O(n)` in-memory representation; external sort differs |
| Naive online candidate search | `N` current intervals | `O(N)` scan | Stores `N` intervals plus aggregate state |
| Merge selected count states | `k` overlapping intervals | `O(k)` for the selected reference operation | Does not include lookup or output invalidations |
| Exact dedup lookup | `d` retained identities | Implementation-dependent | Durable identities and indexes consume storage |
| Rebuild after a split | `n` retained events | Sort and scan as above | Requires complete retained history and deletion semantics |
| Restore and replay | checkpoint bytes and `B` events | Environment-dependent | Storage throughput, source rate, and sink behavior matter |

### 9b. Back-of-envelope recovery sizing

**ILLUSTRATIVE inputs:** Assume a steady input rate `r=1000 events/s` and 60 seconds of outage with no processing.
Assume all outage events remain available in the replayable source.
Then the backlog generated during the outage is 60,000 events.
This is a count, not a memory allocation or a measured recovery time.

```text
[1] FORMULA: B = r * T
[2] WHERE: B is events, r is events/s, T is outage duration in seconds.
[3] CONSTRAINT: constant rate, no processing during outage, no missing or pre-existing backlog.
[4] NUMERICAL EXAMPLE: 1000 events/s * 60 s = 60000 events.
[5] INTERPRETATION: the source must retain at least this modeled volume plus all other required history and margin.
```

Derivation: integrate the assumed constant rate over the outage interval.
Independent check: 1,000 events in each second for 60 seconds gives sixty groups of 1,000.
A changing arrival rate requires integration or discrete summation instead.

Assume additionally that after restart the system sustains `mu=1500 events/s` while new events continue at `r=1000 events/s`.
Assume both rates remain constant and no other backlog exists.
The surplus is `mu-r=500 events/s`.
The time to drain the 60,000-event backlog is `60000/500=120 seconds` after service resumes.

```text
[1] FORMULA: drain time = B / (mu - r)
[2] WHERE: B is events; mu and r are events/s; result is seconds.
[3] CONSTRAINT: mu > r, fixed rates, retained input, and sufficient sink capacity.
[4] NUMERICAL EXAMPLE: 60000 / (1500 - 1000) = 120 s.
[5] INTERPRETATION: nominal catch-up takes two minutes after recovery, excluding startup and restore.
```

Derivation: in each second, processing exceeds new arrivals by `mu-r` events.
Independent check: 500 surplus events/s times 120 s equals 60,000 events.
At `mu=1000 events/s`, the denominator is zero and this model never drains the backlog.
At `mu<r`, the backlog grows rather than drains.
If restored replay speed is slower than the assumed sustainable rate, the estimate is optimistic.
If one partition is hot, fleet-average surplus can hide a partition that never catches up.
The outage duration is not the checkpoint age; records since the last completed checkpoint can add replay work.

### 9c. Illustrative state budget

Assume 200,000 concurrently retained session records and 300 bytes of logical metadata and aggregate per record.
The logical subtotal is `200000 * 300 = 60000000 bytes`.
Using `1 MiB = 1048576 bytes`, this is approximately `57.22 MiB`.
Independently, 200,000 times 3 bytes is 600,000 bytes, then multiplying by 100 gives 60,000,000 bytes.
This excludes keys, serializers, indexes, timers, deduplication, allocator overhead, backend amplification, replication, checkpoints, and buffered events.
It is not a sizing recommendation and cannot be translated into a container memory limit without measurement.
If each retained session instead averages 600 logical bytes, the subtotal doubles to 120,000,000 bytes, about 114.44 MiB.
If watermark progress stalls, the number of retained sessions can grow beyond the assumption even when bytes per record stay fixed.
If the state backend stores data outside the managed heap, heap-only monitoring misses relevant resource pressure.
If a late bridge invalidates many already-published outputs, effect lineage can be a material separate state component.

### 9d. Operational measurement plan

Record the deployed Flink build, connector artifacts, source partition count, maximum and actual parallelism, and state backend.
Characterize rates and sizes by partition and by high-volume key rather than only using fleet averages.
Measure occurrence-to-arrival lag and clock quality separately from processing delay.
Measure accepted-late, rejected-late, duplicate, conflict, correction, and post-finality reconciliation counts.
Track watermark progression per relevant input and the operator minimum.
Track state size and checkpoint completion, alignment, storage transfer, and restore duration using metrics available in the pinned release.
Test a cold restore and replay while the normal input stream continues.
Include the sink's transactional overhead, retry behavior, and correction writes in the load profile.
Size source retention for the oldest required restart point plus detection, repair, restore, and replay windows with a justified safety margin.
Include storage, network, compute, operational labor, retention, and reconciliation in cost per useful corrected session.
Currency, prices, and cost per session are **NOT MEASURED** in this sheet.
Escalate if measured recovery capacity cannot drain the busiest partition before retention expires.

### 9e. Executable specification for an acceptance harness

The following is a proposed test design, not a claim that tests were run.
It uses the generic contract of Section 5 unless a row explicitly says otherwise.
A production harness must replace generic expectations with expectations for its actual API and application contract.
Record the source revision and container digest used for every run.
Record the Flink and connector versions independently because they need not share a version number.
Record timestamp resolution and timezone conversion before constructing input.
Record all gap, lateness, watermark, and idle-time parameters with units.
Record source parallelism, partition assignments, and the exact arrival order.
Record whether the job is bounded, because terminal watermark behavior can differ from an indefinitely running source.
Record checkpoint mode, storage, restart policy, and state backend.
Record the sink schema, conflict resolution rule, transaction boundary, and consumer isolation.
Record the ID namespace and the logical meaning of a revision.
Save the input, operator output, late output, sink changelog, final view, and checkpoint identifiers.
Assert membership and effect correctness separately from processing latency.
Separate a reproducible failure from an inference about its cause.
Report tests that could not be run rather than marking them passed by inspection.

#### Boundary fixtures

**Fixture B1 — one isolated event.**
Input is key `D1`, ID `A`, timestamp 0 ms, gap 5000 ms.
Before firing, the generic state contains `[0,5000)` with count 1.
At watermark 4998 ms, the firing threshold has not yet been reached.
At watermark 4999 ms, the specified trigger fires the window.
This distinguishes end-exclusive maximum timestamp from the end boundary.
The corresponding production assertion must use the deployed trigger's documented behavior.

**Fixture B2 — just inside the gap.**
Input event timestamps are 0 and 4999 ms with distinct IDs and the same key.
Candidates `[0,5000)` and `[4999,9999)` overlap for one millisecond.
The strict generic result is one session `[0,9999)` and count 2.
An implementation that rounds both timestamps to seconds needs a separate stated precision contract.
The fixture detects an off-by-one error without relying on a large workload.

**Fixture B3 — exactly at the gap.**
Input timestamps are 0 and 5000 ms.
Candidates `[0,5000)` and `[5000,10000)` touch at 5000 ms.
The strict generic result contains two sessions of count 1.
The expected result for an actual Flink assigner must be observed in a version-pinned test.
This fixture is a specification discriminator, not a claim about Flink's touching-interval implementation.

**Fixture B4 — just beyond the gap.**
Input timestamps are 0 and 5001 ms.
Their candidate intervals are separated by one millisecond.
The strict generic result contains two sessions.
This differs from the exact-gap fixture only by a single timestamp unit.
Checking both fixtures catches accidental conversion from `<` to `<=`.

**Fixture B5 — same timestamp, distinct IDs.**
Input has IDs `A` and `B`, both at 0 ms and both for `D1`.
Their candidate intervals coincide completely.
The session count is 2 because the logical identities are distinct.
Deduplication by timestamp alone would incorrectly remove one observation.
If the business domain declares one observation per timestamp, that is a different explicit identity contract.

**Fixture B6 — identical ID delivered twice.**
Input repeats ID `A`, key `D1`, timestamp 0 ms, and the same payload.
Deduplication before aggregation produces count 1.
The source delivery count remains 2 and should not be confused with the logical-event count.
The test should retain both deliveries in the input audit record.
This behavior requires the duplicate identity to survive source retry and recovery.

**Fixture B7 — same ID, conflicting payload.**
Input repeats ID `A` with different payloads or event timestamps.
The expected result is determined by the application's conflict policy, not by ordinary duplicate suppression.
A quarantine policy should produce one conflict record with enough context for investigation.
A correction policy must define a source revision and how the old contribution is removed.
Silently counting both records or silently choosing the latest arrival is not a neutral policy.

**Fixture B8 — identical local device ID in separate tenants.**
Input includes `(tenant-1, device-7)` and `(tenant-2, device-7)`.
Both events may have the same occurrence timestamp without sharing state.
The expected grouping uses the composite entity key.
An unqualified `device-7` key fails the fixture even if interval arithmetic is correct.
This check also protects sink identifiers from cross-tenant collisions.

#### Watermark and lateness fixtures

**Fixture W1 — out of order before firing.**
Deliver 12000 ms before 8000 ms while keeping the watermark below the affected firing thresholds.
The record order differs from sorted event time.
Under the generic candidate rule, the final session membership is determined by interval connectivity.
No already-published row requires retraction if nothing has fired.
This fixture separates disorder from post-fire lateness.

**Fixture W2 — accepted late bridge.**
Use actions 1 through 6 of Section 5 in the listed order.
The first emitted output is P1 with count 2.
After the bridge and the second watermark, the replacement has count 4.
The sink must not retain P1 as an additional independent current session.
Capture both intermediate and final views so a passing final count does not conceal an invalid transient state.

**Fixture W3 — event after cleanup.**
Advance the generic watermark to 26999 ms after the merged window has fired.
Then deliver the old event at 8000 ms with a new logical ID.
Under the specified isolated-candidate cleanup rule, route it to the late path.
Assert that the current online count is not silently incremented.
Also assert that the late record retains enough information for reconciliation.

**Fixture W4 — cleanup boundary equality.**
For the merged window, advance the generic watermark first to 26998 ms.
Its state remains because the threshold is 26999 ms.
Then advance to 26999 ms and assert cleanup under the stated generic comparison.
The test should observe state or an equivalent externally supported signal, not infer cleanup only from a clock.
For a Flink-specific harness, replace this comparison with the release's verified semantics.

**Fixture W5 — active minimum.**
Use two active inputs with watermarks 30000 and 10000 ms.
The simplified downstream minimum is 10000 ms.
Advance the first input to 50000 ms while the second remains at 10000 ms.
The downstream minimum remains at 10000 ms under the active-minimum model.
The fixture detects an erroneous use of the maximum input watermark.

**Fixture W6 — idle and resume.**
Mark the second input idle using the configured source strategy.
Under the simplified model, the active input can now advance the downstream watermark.
Resume the second input with a historical observation.
Assert that the watermark does not regress and record the resulting late-data decision.
Repeat with idleness disabled to demonstrate the freshness-versus-lateness tradeoff.

**Fixture W7 — timestamp clock outlier.**
Inject an event far in the future relative to a trusted receipt clock.
Record whether the validation policy quarantines it before watermark generation.
If it is accepted, observe its effect on the generated watermark and other records.
The expected future-skew tolerance must come from a documented domain contract.
The test must not rewrite the original event timestamp without retaining audit provenance.

**Fixture W8 — all inputs quiet.**
Stop delivery to all source inputs without sending an artificial final watermark.
Wait in processing time longer than an ordinary session gap.
Observe whether any source policy actually advances event time.
Do not assume elapsed wall time alone closes an event-time window.
If a heartbeat is used, test that it reflects an authorized completeness assertion.

#### Merge and correction fixtures

**Fixture M1 — bridge before any firing.**
Deliver A, B, C, and D before the firing watermark.
The merged state is `[0,17000)` with four identities.
Only one current session needs to be published when the trigger fires.
No obsolete earlier publication exists in this fixture.
Compare with W2 to isolate the cost of provisional publication.

**Fixture M2 — bridge after both component publications.**
Choose a generic watermark and lateness horizon that leave both component states retained after both fire.
Deliver a bridging candidate that is still admissible under that explicitly selected contract.
Expect the correction to identify both obsolete published IDs.
The replacement must represent the union without double-counting the bridge.
The exact timestamps should be constructed to satisfy the selected cleanup thresholds rather than assumed from Section 5.

**Fixture M3 — bridge after one component has been cleaned.**
Clean one component before a delayed connector event arrives.
The online operator no longer has the full history needed to recover the original union from retained window state alone.
Assert that any online result is labeled according to the application's incomplete-history policy.
Use the late path or historical rebuild to reconcile the missing component.
Do not silently claim that increasing a future lateness setting resurrects cleaned state.

**Fixture M4 — merge with unequal aggregation sizes.**
Let the first component have sum 20 and count 2; let the second have sum 90 and count 3.
Their combined sum is 110 and combined count is 5.
The combined mean is 22, not the unweighted mean of the component means 10 and 30.
The latter would give 20 and incorrectly weight the two components equally.
An incremental mean implementation must merge sufficient statistics or an equivalent correct representation.

**Fixture M5 — duplicate bridge.**
Deliver an identical logical bridge twice with the same ID.
The second delivery must not add another count when the deduplication contract applies.
Connectivity and count are separate checks: a duplicate may leave the boundaries unchanged while corrupting the aggregate.
Include the source delivery count and the logical event count in the audit.
Repeat after checkpoint recovery to exercise persisted dedup state.

**Fixture M6 — delete a bridge.**
Begin with event times 0, 4, 8, and 12 seconds and the strict 5-second gap.
Remove the event at 8 seconds under an explicit deletion contract.
The remaining sorted gaps are 4 and 8 seconds.
The resulting sessions are `[0,9)` and `[12,17)` with counts 2 and 1.
Verify both the retraction of the old merged session and the publication of both replacements.

**Fixture M7 — replay correction bundle.**
Apply the same invalidate-and-replace bundle twice.
An idempotent materialization should have the same final view after either one or two applications.
Inspect side effects separately: an audit log may intentionally record both delivery attempts.
The application must define the durable effect identifier used for deduplication.
This property is not guaranteed merely by naming a write an upsert.

**Fixture M8 — reorder revisions.**
Deliver revision 2 before revision 1 to a sink that promises monotone conditional updates.
The final current row should retain revision 2.
The older update should be rejected, ignored, or recorded as stale according to the contract.
Test tombstone ordering as well as replacement ordering.
An implementation without atomic revision checks cannot claim this property without another ordering mechanism.

#### Recovery and migration fixtures

**Fixture R1 — failure before a completed checkpoint.**
Begin from a known completed checkpoint with recoverable source positions.
Process additional observations and fail before the next checkpoint completes.
Restore the previous state and replay the corresponding inputs.
Compare the resulting logical state with a no-failure reference run.
Check external effects independently because they may have escaped before the failure.

**Fixture R2 — failure after a sink acknowledgment.**
Cause the sink to acknowledge an effect and fail before its relevant checkpoint completes.
On replay, observe whether the same logical effect is attempted again.
The final visible result must satisfy the chosen transaction or idempotency contract.
Record both the external acknowledgment and checkpoint completion sequence.
An end-state match alone cannot prove that an irreversible alert was not sent twice.

**Fixture R3 — missing replay position.**
In an isolated environment, make the required source history unavailable.
Attempt restore from the selected checkpoint and record the actual behavior.
The expected safe response is an explicit failure or a separately approved rebuild policy.
Starting at the newest position without an explicit data-loss decision is not equivalent recovery.
Track the earliest retained source position against the oldest usable checkpoint.

**Fixture R4 — serializer change.**
Create state with one serializer and attempt a restore with the intended new artifact.
Check the supported compatibility or migration procedure for that release and state descriptor.
Compare restored aggregate, timer, and deduplication semantics with the original.
A successful process startup is not enough if the decoded state is semantically wrong.
Keep the old recoverable artifact until the migration's rollback decision is complete.

**Fixture R5 — rescaling.**
Take a compatible state snapshot and restore at the proposed new parallelism.
Verify that every test key is present exactly once in the restored logical state.
Verify that source positions and sink writers do not create missing or duplicate effects.
Recalculate hot-key distribution because more workers do not automatically split one key.
Record any maximum-parallelism constraints and the relevant release documentation.

**Fixture R6 — sink transaction expiry.**
Pause the system for longer than a hypothetical sink transaction timeout.
Exercise the connector's documented recovery behavior in a disposable environment.
Observe whether transactions commit, abort, expire, or require manual reconciliation.
Align configured timeouts with measured outage and restore behavior where the connector permits it.
Do not assume all sinks expose the same transaction protocol.

**Fixture R7 — state and output version disagreement.**
Restore an earlier internal checkpoint while retaining a sink view containing later revisions.
Confirm that replay cannot replace a newer valid view with an older revision.
If revisions can be reset on restore, use a durable epoch or another documented conflict rule.
The design must also identify obsolete keys created after the restored checkpoint.
Where a clean replay cannot reconcile those keys, rebuild into a fresh destination and switch after validation.

**Fixture R8 — backfill isolation.**
Replay historical input into a separate output namespace.
Compare final sessions against the production namespace before any promotion.
Include corrections and removals, not just aggregate counts.
Document how concurrent live events are incorporated at the cutover boundary.
Require an explicit owner decision before replacing an authoritative materialized view.

### 9f. Observability and reconciliation design

The following fields are a proposed instrumentation contract, not metrics claimed to exist in a particular Flink release.
Choose actual metric names and availability from the deployed version.
Do not attach unbounded event IDs as monitoring labels.
Use restricted audit records rather than high-cardinality metric dimensions for individual IDs.
Aggregate telemetry by source, partition, operator, and bounded time bucket where privacy policy permits.
Record the timestamp extractor version alongside data-quality measurements.
Record the session algorithm version alongside emitted output.
Record the configured gap in the emitted result or in an immutable referenced policy version.
Record the lateness policy version so later investigators can interpret finality.
Record the logical key namespace independently of its serialized representation.
Record the source event's identity scheme and producer version.
Record the checkpoint identifier associated with a transactional output only if the connector exposes a meaningful association.
Record the correction reason when a previously published session is invalidated.
Record a deterministic effect ID when the sink's idempotency design requires one.
Record whether a session is provisional, operationally final, or later amended by backfill.
Record the source observation time without using it as a substitute for event time.
Record wall-clock processing time with a clear clock and timezone.
Avoid logging sensitive payload fields merely to make debugging convenient.
Apply retention limits and access control to event-level audit records.

#### Input quality indicators

Count records whose event timestamp cannot be parsed.
Count records rejected because a timestamp is outside the domain-approved range.
Count records whose event timestamp is implausibly far ahead of receipt time.
Count records whose device-to-receipt delay exceeds each chosen operational threshold.
Partition delay measurements by source type when the sources have different upload behavior.
Distinguish a large delay caused by offline buffering from a device clock error.
Compare device clock offsets against a trusted clock where that comparison is justified.
Measure the share of records whose declared identity was previously observed.
Separate identical duplicate deliveries from same-ID conflicting payloads.
Count invalid or missing tenant-aware keys before the keyed partition.
Track producer schema versions to identify a timestamp-field migration.
Record the fraction of traffic lacking a trusted occurrence timestamp.
Measure missing partitions or source splits using the source's own inventory where available.
Do not interpret an absence of received events as proof of an absence of real-world activity.
Alert thresholds need a baseline and an accountable response owner.

#### Event-time progress indicators

Observe each relevant input's most recent watermark where the implementation exposes it.
Observe the downstream operator watermark at the same sampling time.
Record input activity or idle status alongside watermark values.
Calculate progress differences only between timestamps in the same epoch and units.
Compare watermark movement with incoming record volume.
Compare watermark movement with checkpoint progress without assuming one causes the other.
Track the duration of a stationary watermark while relevant inputs are active.
Track the duration for which a source is considered idle.
Count records arriving after an idle input resumes.
Measure how many resumed records are routed to the late-data path.
Investigate a sudden watermark jump together with future-dated source events.
Avoid equating a watermark age with end-to-end latency without defining both clocks.
Use a controlled trace to establish the expected watermark sequence for each source strategy.
Interpret source-level watermarks carefully when multiple splits share a source task.
Escalate sustained disagreement between data arrival and expected event-time progress.

#### Window and correction indicators

Count newly created session candidates.
Count window merges separately from new event arrivals.
Measure how many previously emitted identities each merge invalidates.
Count firing events separately from distinct logical sessions.
Count provisional outputs and later revisions.
Measure the interval between initial publication and a subsequent correction.
Count accepted late observations separately from rejected late observations.
Count late records that could not be routed to the configured side channel.
Track output revisions rejected by the sink as stale.
Track correction bundles with incomplete removal or replacement steps.
Compare the count of invalidation requests with acknowledged removals.
Count sessions amended by an offline rebuild after online cleanup.
Measure session duration distributions and the number of continuously active keys.
Identify whether unexpectedly long sessions arise from dense activity or stalled progress.
Do not infer the number of distinct users or devices by counting provisional output rows.

#### Recovery indicators

Record the most recent completed checkpoint and its age.
Record failed checkpoint reasons with sufficient detail to distinguish storage and coordination failures.
Track checkpoint bytes and duration separately from checkpoint interval.
Measure barrier alignment time where the deployed metrics support it.
Measure snapshot upload time independently of alignment time.
Observe whether checkpoint storage and source retention remain accessible during an incident.
Track estimated replay backlog per input partition.
Measure restored processing rate under continuing live traffic.
Compare restore duration against the business recovery objective.
Track whether the required replay position is approaching source-retention expiry.
Count sink transaction aborts or retries using connector-specific signals.
Observe deduplication state availability after a restore.
Track the oldest checkpoint known to be restorable, not merely the oldest file present.
Rehearse recovery in an isolated destination before treating a recovery objective as achievable.
Flag repeated restart attempts that make no measurable progress.

#### Reconciliation procedure

Select a bounded event-time interval and specify any necessary guard band around its boundaries.
Select the source partitions and source positions that define the input snapshot.
Include events delayed into the selected interval after its ordinary online finality.
Apply the same identity and conflict-resolution policy as the intended business definition.
Recompute strict or touching-gap sessions according to the approved policy.
Compare event-ID membership where privacy and retention allow it.
Compare boundaries, counts, and sufficient aggregate statistics for every key.
Compare obsolete IDs as well as current session rows.
Separate expected provisional differences from incorrect final materialization.
Classify differences as missing source data, late policy, timestamp correction, deduplication, or sink effect failure.
Preserve the input snapshot and policy version used for the comparison.
Reconcile cross-boundary sessions before treating each time bucket as independent.
Issue a correction bundle or rebuild only after determining the authoritative input set.
Record the decision owner and the approved correction cutoff.
Re-run the comparison after applying corrections, using a new immutable run identifier.
Document residual discrepancies and the reason they cannot yet be resolved.
An empty discrepancy report is meaningful only for the selected source snapshot and comparison rules.
When original events have expired, state explicitly that exact event-level reconciliation may be impossible.
Retention requirements should follow from these audit and correction needs.
No reconciliation report or operational measurements exist for the illustrative scenario in this sheet.

### 9g. Release and operational sign-off

The event producer owner approves identity stability, timestamp meaning, and clock-error handling.
The platform owner approves source retention and the restore procedure.
The application owner approves session gap, equality behavior, and aggregation rules.
The consumer owner approves provisional visibility, finality, and correction semantics.
The sink owner confirms transactions, conditional updates, or reconciliation capabilities with version-specific evidence.
The data governance owner approves retention and access to raw event and identity state.
The incident owner defines what happens when the late path or reconciliation service is unavailable.
The deployment owner records rollback criteria and a compatible restore point.
The business owner accepts the measured freshness and correction tradeoff rather than a theoretical guarantee.
No approval substitutes for a restore drill against representative input and sink behavior.

**Takeaway:** Size retained state and recovery against measured skew and progress, not only average input rate.

## 10. Key Theorems / Results Reference

This systems sheet does not claim a new mathematical theorem.
The following are scoped logical results and version-specific documented behaviors.

| Result | Statement and conditions | Practical implication | Does not imply | Source |
|---|---|---|---|---|
| Fixed-gap connectivity | Under strict-overlap fixed-gap assumptions, adjacent sorted gaps below `g` connect a component | A finite sorted scan is a useful oracle | Equality behavior for every framework | DERIVED, Section 4 |
| Watermark minimum | Flink 1.20 documentation describes stalled minimum progress from idle inputs | Diagnose slow or idle inputs individually | Idleness proves all old data has arrived | [S1] |
| Time-window bounds | Flink 1.20 describes inclusive start and exclusive end and a `maxTimestamp` method | Write boundary tests at millisecond precision | Every custom window or trigger behaves like the example | [S2] |
| Session merging | Flink 1.20 documents session assigners and merge-capable functions/triggers | Verify custom merge behavior | Automatic retraction in an arbitrary sink | [S2] |
| Checkpoint and replay | With recoverable state and rewindable sources, Flink restores a consistent processing state | Preserve checkpoint and source history | Exactly-once arbitrary external side effects | [S3,S4] |
| Window lateness | Flink 1.20 describes retention and late firings for configured allowed lateness | Specify and test late-update policy | A watermark proves no late records exist | [S2] |

### Source registry and version boundaries

**[S1]** Apache Flink, [Generating Watermarks, release 1.20](https://nightlies.apache.org/flink/flink-docs-release-1.20/docs/dev/datastream/event-time/generating_watermarks/), consulted 2026-11-07.
Supports timestamp assignment, watermark strategies, idle inputs, and documented alignment limitations.
The release page itself identifies 1.20 as an older release; it is pinned for reproducibility, not recommended as a current deployment target.
Connector-specific source capabilities and behavior still require matching connector documentation.

**[S2]** Apache Flink, [Windows, release 1.20](https://nightlies.apache.org/flink/flink-docs-release-1.20/docs/dev/datastream/operators/windows/), consulted 2026-11-07.
Supports window bounds, session assigners, triggers, merging, allowed lateness, and late-data handling at the documented API scope.
The generic trace is a separately declared teaching contract and is not asserted to reproduce every internal edge case of this release.
In particular, equality at touching endpoints, exact cleanup, merged trigger behavior, and post-cleanup events must be tested against the chosen implementation.

**[S3]** Apache Flink, [Stateful Stream Processing, release 1.20](https://nightlies.apache.org/flink/flink-docs-release-1.20/docs/concepts/stateful-stream-processing/), consulted 2026-11-07.
Supports keyed state, checkpoint and replay concepts, barrier alignment, and the documented unaligned-checkpoint tradeoff.
A completed checkpoint is not an audit of an external database or irreversible action.

**[S4]** Apache Flink, [Fault Tolerance Guarantees of Data Sources and Sinks, release 1.20](https://nightlies.apache.org/flink/flink-docs-release-1.20/docs/connectors/datastream/guarantees/), versioned primary reference.
Connector guarantees are configuration- and implementation-specific; check the actual source, sink, and release rather than extrapolating from a general concept page.
The sink's own documentation and transaction configuration remain necessary evidence.

**Review trigger:** Reassess these claims when upgrading Flink, changing connectors or serializers, revising timestamp or lateness policy, or altering downstream correction semantics.
**Evidence limit:** None of these sources establish the hypothetical sensor workload, its event delays, or any production benefit.

**Takeaway:** Documented framework behavior is version-scoped and cannot substitute for an end-to-end effect contract.

## 11. Interview / Exam Questions

### Level 1 — Definition / Recall

**Q1. What distinguishes event time from processing time?**
**Answer:** Event time is the assigned occurrence timestamp; processing time is when a worker executes an operation.
A delayed sensor reading can occur at 08:00 and be processed at 08:10.
A processing-time session can therefore differ when the same records are replayed at another speed.
The source timestamp must still be validated before treating it as an accurate occurrence time.

**Q2. What does a watermark say?**
**Answer:** It communicates event-time progress according to a generator's assumptions.
It is not a guarantee that older records are physically impossible.
A resumed idle partition can produce an event behind the current downstream watermark.
Handling that event depends on the relevant window's retained state and configured policy.

**Q3. What is the difference between a trigger and allowed lateness?**
**Answer:** A trigger decides when a window calculation fires; allowed lateness controls a window's late acceptance/retention under the applicable implementation contract.
A fired window can remain retained and later fire again.
A custom purge changes retained contents and must not be confused with ordinary firing.
Neither setting automatically makes downstream updates idempotent.

**Q4. Why does a session end later than its last event?**
**Answer:** Each observation contributes a candidate interval extending by the inactivity gap.
With an event at 12 seconds and a 5-second gap, its candidate ends at 17 seconds.
That end is a window boundary rather than a new observation timestamp.
The exact boundary and equality policy must be part of the contract.

### Level 2 — Derivation / Apply

**Q5. Under the generic strict contract, what happens to events at 0, 4, and 12 seconds with a 5-second gap?**
**Answer:** The first two candidates overlap and form `[0,9)`; the third forms `[12,17)`.
The first adjacent gap is 4 seconds and the second is 8 seconds.
Only the first is strictly below 5 seconds.
All three events have the same assumed key and distinct IDs.

**Q6. What changes when the event at 8 seconds is accepted later?**
**Answer:** Its candidate `[8,13)` overlaps both retained sessions, connecting them into `[0,17)` with count 4.
If `[0,9)` has already been published, it must be invalidated or superseded under the downstream contract.
The result depends on retaining the relevant old state when the bridge is processed.
The sorted gaps 4, 4, and 4 seconds independently confirm one component.

**Q7. Why can an event older than the watermark still be accepted?**
**Answer:** Lateness is assessed in relation to the candidate/merged window and its cleanup policy, not just the event timestamp.
In Section 5, the 8-second event arrives at watermark 9 seconds but connects to a retained window ending at 17 seconds.
The generic contract accepts the merged window because its cleanup threshold is 26.999 seconds.
A different engine or cleanup rule must be evaluated using its own version-specific behavior.

**Q8. What is the generic trace's fire and cleanup time for `[0,17000)` with `L=10000 ms`?**
**Answer:** The maximum included timestamp is 16,999 ms.
The stated firing threshold is 16,999 ms and the cleanup threshold is 26,999 ms.
These are event-time watermark thresholds, not wall-clock deadlines.
The equality comparisons are explicitly defined by the teaching contract.

**Q9. Under strict overlap, do events exactly 5 seconds apart merge with a 5-second gap?**
**Answer:** No: their candidate intervals touch but have no positive-width overlap.
For events at 0 and 5 seconds, the intervals are `[0,5)` and `[5,10)`.
A touching-merge policy would give a different answer and must be tested in the deployed assigner.
Do not transfer this generic equality decision to every Flink release without verification.

**Q10. Calculate backlog and catch-up for a 60-second outage at 1,000 events/s, followed by 1,500 events/s service.**
**Answer:** The outage adds 60,000 events if none are processed and all are retained.
New input continues at 1,000 events/s, so the surplus is 500 events/s.
The modeled drain time is 60,000 divided by 500, or 120 seconds after service resumes.
Startup, restore, skew, and pre-existing backlog are excluded.

### Level 3 — Judgment / Design / Debug

**Q11. Why is a checkpoint not enough to guarantee exactly-once customer-visible effects?**
**Answer:** It protects a recoverable processing state only under the configured checkpoint and source assumptions.
A nontransactional remote call might succeed before failure and be repeated during replay.
Use a connector's documented transactional protocol, a durable idempotency key, or an explicit compensation design.
Validate the exact destination and connector rather than generalizing from Flink's internal state guarantee.

**Q12. How would you distinguish an idle input from a slow sink when windows stop firing?**
**Answer:** Compare input-level watermarks and activity with downstream throughput and backpressure.
A frozen active-input minimum with healthy other inputs points toward a stalled or idle input.
Advancing watermarks with backed-up downstream tasks point toward an execution or sink bottleneck.
Inspect both before changing lateness or declaring a partition idle.

**Q13. Why is an upsert keyed by session start and end insufficient after a merge?**
**Answer:** A late bridge changes the boundaries, so the merged session has a different output key.
Upserting the new key leaves the old key visible unless it is explicitly invalidated.
Publish removals and replacement in a well-defined durable effect protocol.
If the sink cannot apply them atomically, define intermediate visibility and reconciliation.

**Q14. How long should duplicate IDs be retained?**
**Answer:** At least for the business-required duplicate and replay horizon, subject to the chosen state and recovery design.
An arbitrary short TTL can forget an ID before an old delivery is replayed.
A finite TTL cannot guarantee deduplication against unbounded replay.
Measure delay and source retention, then choose a defensible horizon or durable ledger.

**Q15. Can removing a bridge be handled by merging counts in reverse?**
**Answer:** Not generally: a merged count does not reveal which member connected separate time regions.
Removing the event at 8 seconds from the four-event example splits the session into `[0,9)` and `[12,17)`.
Retain sufficient membership information or recompute from a complete event log.
A compact aggregate is appropriate only if the allowed correction operations do not require lost detail.

**Q16. How would you choose a watermark and lateness policy?**
**Answer:** Measure source-specific event-time disorder and clock errors, and agree on tolerated provisional duration and correction rates.
Distinguish the watermark generator's disorder bound from the window's allowed-lateness retention.
Test silent and resumed partitions, post-horizon events, backfill, and sink updates.
If freshness conflicts with correctness, make the tradeoff explicit to the decision owner rather than assuming a universal setting.

**Takeaway:** A strong answer traces the effect all the way from occurrence time through recovery and downstream correction.

## 12. Quick Reference Card

| Field | Quick reference |
|---|---|
| **Topic** | Event-time session grouping, state recovery, and downstream corrections. |
| **Core formula** | Under the generic strict fixed-gap contract, adjacent sorted events connect iff `t(i+1) - t(i) < g`; a late bridge can merge previously published sessions. |
| **Key assumption** | Integer-ms precision, half-open intervals, strict overlap, retained merge state, explicit trigger/cleanup and a correction-capable materialization. Verify the deployed assigner separately. |
| **Complexity** | For `n` retained events, sort and scan costs `O(n log n)` time and up to `O(n)` space; incremental state and recovery depend on the implementation. |
| **Use when** | Activity gaps define sessions and consumers can handle the chosen lateness and correction contract. |
| **Avoid when** | Timestamps are untrustworthy or irreversible consumers cannot tolerate provisional sessions and have no reconciliation path. |
| **Go deeper** | [CDC ordering and materialization](02_CDC_Ordering_Snapshot_and_Materialization.md); broader windowing patterns remain planned. |
| **Worked result** | At `0,4,12 s`, gap `5 s` yields `[0,9)` and `[12,17)`; adding `8 s` yields `[0,17)` with four distinct events. |
| **Diagnostic** | Compare input watermarks and output IDs; stalled progress and missing retractions require different fixes. |
| **Guardrail** | Pin Flink and connector versions; test equality, recovery, deduplication, retention and sink effects end to end. |
