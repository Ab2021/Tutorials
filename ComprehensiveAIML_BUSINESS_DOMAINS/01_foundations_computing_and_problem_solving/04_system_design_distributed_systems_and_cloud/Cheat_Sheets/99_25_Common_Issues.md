---
title: "25 Common Issues — 04_system_design_distributed_systems_and_cloud"
topic: "Distributed-systems and cloud defect diagnosis: violated assumptions, numerical edges, scale failures, concept confusions"
collection: "04_system_design_distributed_systems_and_cloud"
level: "advanced"
evidence: "ILLUSTRATIVE"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# 25 Common Issues — 04_system_design_distributed_systems_and_cloud

This file is the unit's defect catalog. Every workload figure used to demonstrate a diagnostic is a
constructed teaching input and is labelled **ILLUSTRATIVE** where it appears; the arithmetic over
those inputs is recomputed by hand and shown so a reader can audit it. Nothing here reports a
measured production incident, a company architecture, or an observed outage. Issue identifiers
`CI-001` through `CI-030` are stable within this file: they are referenced by the interview section
and by the cross-reference table, and they must not be renumbered on revision.

## 1. File Header

**Scope.** This catalog covers the recurring defects that appear when a design, a report, or a
dashboard built on distributed infrastructure is reasoned about incorrectly: contracts between
producer and consumer, unit and precision handling, capacity and queue behaviour under load, and
vocabulary that hides a missing guarantee. It does not cover application logic bugs that are
independent of distribution, compiler defects, or security exploit classes beyond the authorization
ordering issue in `CI-021`.

**Reader task.** Given one observed symptom, the reader must be able to (a) assign it to exactly one
of the four categories below, (b) name the two or three mechanisms that could produce the same
symptom, (c) select a check that separates those mechanisms rather than confirming the first guess,
(d) apply a containment action that is bounded in time and blast radius, and (e) state a
replay-based verification that would fail if the fix were cosmetic.

**Categories.** The four categories are not decorative; they determine which evidence is worth
collecting first.

| Category | Definition | First evidence to collect | Typical wasted effort |
|---|---|---|---|
| VIOLATED_ASSUMPTION | The system is behaving as built, but a precondition the reasoning relied on does not hold | Timestamps, versions, actor identity, and ordering at the boundary | Tuning capacity when the problem is a clock or contract |
| NUMERICAL_EDGE | Arithmetic is correct in the ordinary range and wrong at a boundary or unit change | Numerator and denominator separately, plus the unit of each field | Rewriting business logic when a conversion is missing |
| SCALE_FAILURE | A relationship holds at small load and inverts as concurrency, cardinality, or skew grows | Rate, concurrency, and per-key distribution, not averages | Adding replicas when the bottleneck is one key or one lock |
| CONCEPT_CONFUSION | A term is used to assert a guarantee the mechanism never provided | The mechanism's contract text and the actual failure behaviour | Debating definitions instead of testing an invariant |

**Severity scale.** Severity here expresses consequence inside the declared teaching scope, not an
incident claim. Use it only to order the diagnosis queue.

| Level | Meaning in this catalog | Response expectation |
|---|---|---|
| S1 | Silent wrong answer that a downstream consumer treats as verified | Stop the consumer, then diagnose; never "observe for a week" |
| S2 | Correct answer with unavailable or unbounded latency for some population | Contain with a bounded fallback while diagnosing |
| S3 | Degraded but visible signal; operators can already see it is wrong | Diagnose in the normal queue |
| S4 | Cost, storage, or cardinality growth with no correctness effect yet | Schedule; the trend is the finding |

**Evidence policy.** Three labels appear throughout: `DERIVED` for statements that follow from a
named formula and its stated conditions, `ILLUSTRATIVE` for constructed inputs, and `OPEN` for a
cause that the available evidence does not settle. A cause is never upgraded from `OPEN` because a
plausible story fits; it is upgraded only when a discriminating check returns a result that rules
out the competing mechanisms.

**ID scheme and record shape.** Every issue carries eight fields: identifier, category, symptom,
competing causes, discriminating check, correction, prevention, and verification. The table in
section 8 is the compact index; the blocks below it carry the mechanism and the numbers. A field is
never left as "investigate further" — an unresolved field states which measurement would resolve it
and what value would falsify the leading hypothesis.

**Triage order.** Diagnose in this order, because later steps depend on earlier ones being sound:
contract and clocks first, units and precision second, capacity and skew third, vocabulary last.
Reversing the order produces the classic failure of scaling a system that is mislabelling its
timestamps, or of renaming a metric whose denominator was zero.

**What "fixed" means here.** A correction changes the mechanism: a precondition check, a versioned
contract, a widened accumulator, a bounded retry budget. A mitigation contains harm without changing
the mechanism: deferring an item, serving a cached answer, refusing a write. Reporting a mitigation
as a fix is itself a catalogued defect (`CI-012` adjacent confusion, `CI-025` configuration drift).

**Out of scope.** Physical network faults, hardware failure rates, and vendor SLA disputes are
outside this catalog except where they enter through a composition error (`CI-004`) or a lease
margin error (`CI-029`).

## 2. Concept Map / Mental Model

The diagnosis loop has five states. The value of drawing it is that containment and correction are
separate states with separate exit conditions; collapsing them is how a hotfix becomes permanent.

```text
  observe symptom
        |
        v
  classify category  ---> (ambiguous) ---> collect boundary evidence, reclassify
        |
        v
  enumerate competing causes (>=2)
        |
        v
  run discriminating check  ---> result contradicts all causes? -> widen population/window
        |
        +--> confirmed cause
                |
                v
        contain (bounded, reversible, logged) --> correct mechanism --> replay verification
```

A second mental model is the **evidence ladder**. Each rung is cheaper than the one above it and
answers a narrower question. Climb only as far as the ambiguity requires.

```text
  rung 1  aggregate dashboard value          (tells you a symptom exists)
  rung 2  numerator and denominator split    (tells you which side moved)
  rung 3  per-key / per-shard distribution   (tells you whether it is skew)
  rung 4  raw record with timestamps         (tells you event time vs processing time)
  rung 5  replay of one record + boundary    (tells you whether the fix is real)
```

The third model is the **category router**. When a symptom could belong to two categories, the
router picks the category whose first evidence is cheaper to obtain and whose correction is
reversible. A clock problem misdiagnosed as a capacity problem leads to adding replicas, which is
expensive and hides the clock problem for another week.

```text
  symptom mentions "sometimes", "one customer", "only after deploy"
        -> VIOLATED_ASSUMPTION (ordering, version, identity)

  symptom mentions "zero", "infinity", "negative", "off by a constant factor"
        -> NUMERICAL_EDGE (unit, precision, overflow, empty window)

  symptom mentions "fine in staging", "grew over weeks", "one key is slow"
        -> SCALE_FAILURE (skew, cardinality, concurrency, retention)

  symptom mentions "we assumed", "it says eventually", "the doc says exactly"
        -> CONCEPT_CONFUSION (guarantee that the mechanism never provided)
```

The fourth model is the **four-way trade at containment time**. Every containment action spends one
of: freshness, completeness, latency, or money. Naming which one you are spending prevents claiming
a free fix, and it also tells the reviewer what the user-visible cost will be during the incident.

```text
  freshness spent      -> serve cached or last-known value, label its age
  completeness spent   -> defer a subset, record the deferred key set
  latency spent        -> queue the work, publish a completion estimate
  money spent          -> add capacity or storage, bound the spend with a quota
```

The fifth model is **symptom versus mechanism**. Two mechanisms frequently share a symptom, so the
mental checklist is always "what else produces exactly this number". A p99 of 900 ms can come from a
hot shard, from a garbage-collection pause, from a connection-pool wait, or from a retry that
finally succeeded. Only the distribution shape across keys and the presence of a wait-time metric
separate them.

Takeaway: classification precedes correction, containment is not correction, and every containment
action has a named cost.

## 3. Foundational Definitions

Definitions are given with units and validity ranges because most of this catalog's defects are
definitional before they are computational.

| Term | Meaning in this file | Units / type | Valid range | Failure when confused |
|---|---|---|---|---|
| Event time | When the fact happened at its source | timestamp, UTC | any instant | Compared against processing time, producing negative latencies |
| Processing time | When the pipeline handled the record | timestamp, UTC | >= event time for the same record in a sane pipeline | Used as event time in windows, double-counting late data |
| Cut-off | The instant after which inputs no longer influence a decision | timestamp, UTC | must be published with the decision | Decision re-derived later with different inputs |
| Arrival rate λ | Items offered to a service per unit time | items/s | >= 0 | Measured over a window that includes a stall |
| Service rate μ | Items a single worker completes per unit time | items/s per worker | > 0 | Summed across workers without multiplying by worker count |
| Capacity c·μ | Sustainable completion rate of a pool | items/s | > 0 | Compared against λ without the worker factor |
| Utilization ρ | λ divided by capacity | dimensionless | [0,1) for stability | Reported as a percentage of a wrong denominator |
| Service time S | Time one worker spends per item | seconds | > 0 | Confused with response time, which includes waiting |
| Backlog B | Items accepted but not completed | items | >= 0 | Allowed to go negative in an unsigned gauge |
| Concurrency L | Items in flight at an instant | items | >= 0 | Estimated by request rate alone, ignoring duration |
| Idempotency key | A value that makes a repeated request map to one effect | string, stable per logical request | non-empty, stable across retries | Regenerated per attempt, so retries duplicate |
| Deduplication window | How long a completed key is remembered | seconds | > maximum retry horizon | Shorter than retries, so late retries duplicate |
| Quorum | The subsets whose intersection the protocol requires | set of nodes | R + W > N for strong intersection | Arithmetic done without counting the subsets |
| Replication lag | Delay between a write being durable on primary and visible on a replica | seconds | >= 0 | Assumed zero by a read-after-write path |
| Lease | A time-bounded grant of authority | seconds | must exceed worst-case pause | Shorter than a pause, producing two leaders |
| Retry budget | The fraction of extra attempts a caller may spend | dimensionless | typically 0.1 to 0.2 of base traffic | Unbounded, so retries multiply load |
| Tail quantile | The value below which a stated fraction of observations fall | same unit as the measurement | 0 < q < 1 | Averaged across shards instead of merged |
| Cardinality | Number of distinct keys in a set | count | >= 0 | Reported as exact when it is an estimate |
| Error budget | Allowed failure within a window implied by an objective | minutes per window | >= 0 | Converted to a per-request probability and misapplied |
| Skew | Concentration of load on a key or partition | fraction of total | [1/N, 1] for N partitions | Hidden by averaging partition metrics |
| Fan-out | Downstream calls issued per inbound request | calls/request | >= 1 | Multiplied by retries only after saturation |
| Headroom | Capacity held above expected peak | fraction of peak | commonly 0.15 to 0.30 | Set to zero "because autoscaling exists" |

**Symptom.** An observable that the system emits without interpretation: a queue depth, an error
rate, a latency quantile, a reconciliation difference. A symptom does not name a cause even when its
name suggests one.

**Cause.** The mechanism that, if changed, would remove the symptom. Causes are stated at the level
of a contract or an arithmetic operation, not at the level of a component, because "the database is
slow" is not a cause that a fix can target.

**Discriminating check.** A measurement whose outcome differs between the competing causes. A check
that returns the same result under both hypotheses is confirmation, not diagnosis, and it is the
most common wasted step in incident review.

**Mitigation.** A bounded action that reduces harm while the cause is unknown: defer, cache, refuse,
route around. Mitigations carry an expiry; without one they become the operating design.

**Correction.** A change to the mechanism: a contract version, a precondition, a type width, a
budget, a reordering of authorization before access.

**Verification.** A replay of the triggering record plus one adjacent boundary record, with the
output, reason code, side-effect count, and actor recorded. Verification exists to make an
overstated fix visible.

Takeaway: a term without units and a stated valid range cannot be checked, and a check that cannot
fail is not evidence.

## 4. Core Formulas with Full Derivation

Each result below uses the same anatomy: formula, where each symbol lives and in what unit, the
constraint under which it holds, a recomputed numerical example, and an interpretation that names
what the number licenses you to conclude. Fenced blocks use plain text so the arithmetic can be
copied into a review document without rendering changes.

### 4.1 Base case

**Result 4.1a — Little's law as a pool-sizing identity.**

```text
FORMULA: L = λ · W
WHERE:   λ = arrival rate [requests/s], scalar, >= 0
         W = mean time in system [s], scalar, > 0
         L = mean number in system [requests], scalar, >= 0
CONSTRAINT: open system in steady state over the measurement window; every arrival eventually
            departs; no long-lived population change inside the window.
NUMERICAL EXAMPLE: λ = 2,000 requests/s, W = 0.040 s
         L = 2,000 × 0.040 = 80 requests in flight
INTERPRETATION: A pool must hold at least 80 simultaneous workers to sustain this rate at this
            duration; any pool smaller than 80 forces waiting rather than service.
```

Derivation: in a window of length T, arrivals number λT and each spends W in the system. The total
occupancy-time is λT·W; dividing by T gives the time-average occupancy L. The identity is
dimensional: [1/s]·[s] = [1], so a unit error shows up immediately as a non-dimensionless L.

**Result 4.1b — Backlog accumulation under sustained imbalance.**

```text
FORMULA: B(t) = max(0, (λ - μ_sustain) · t),  drain time T_d = B / (μ_drain - λ)
WHERE:   λ = offered rate [items/s]; μ_sustain = sustainable completion rate [items/s]
         μ_drain = completion rate after capacity is restored [items/s]
         B = backlog [items]; t = elapsed imbalance time [s]; T_d = time to empty [s]
CONSTRAINT: constant rates, no loss, no retry traffic, service not yet degraded by queue depth.
NUMERICAL EXAMPLE: λ = 2,000, μ_sustain = 1,800, t = 300
         B = (2,000 − 1,800) × 300 = 200 × 300 = 60,000 items
         restated in arrival-time terms: 60,000 / 2,000 = 30 s of traffic
INTERPRETATION: A 10% rate deficit for five minutes costs thirty seconds of traffic to clear; the
            recovery rate, not the deficit, sets how long the tail lasts.
```

The restatement matters because operators routinely compare backlog in items against rates in
items per second and conclude the backlog is small. Sixty thousand items sounds large and thirty
seconds of traffic sounds small; both describe the same state.

### 4.2 General case

**Result 4.2a — Utilization and the pool-wait approximation.**

```text
FORMULA: ρ = λ / (c · μ) = λ · S / c
         W_q ≈ ρ · S / (2 · (1 − ρ))      (single-queue, near-deterministic service)
         W = S + W_q
WHERE:   c = worker count [count]; S = service time [s]; ρ = utilization [dimensionless]
         W_q = mean queue wait [s]; W = mean response time [s]
CONSTRAINT: single queue feeding c interchangeable workers, service-time variability low
            (coefficient of variation well below 1), arrival process not bursty, ρ < 1.
NUMERICAL EXAMPLE: λ = 2,000/s, S = 0.040 s, c = 100
         capacity = 100 / 0.040 = 2,500/s
         ρ = 2,000 / 2,500 = 0.80
         W_q ≈ 0.80 × 0.040 / (2 × 0.20) = 0.032 / 0.40 = 0.080 s = 80 ms
         W = 40 + 80 = 120 ms
INTERPRETATION: At 80% utilization each request spends twice its service time waiting; the pool is
            not "sized at peak", it is sized at 25% spare and that spare is consumed by queueing.
```

The same arithmetic at c = 90 gives capacity 2,250/s, ρ = 0.889, W_q ≈ 160 ms and W ≈ 200 ms. Two
fewer workers is a 10% capacity cut and a doubling of wait, which is the practical reason pool
sizing does not scale linearly with intuition.

**Result 4.2b — Composite availability, serial and redundant.**

```text
FORMULA: A_series = Π A_i                     (all stages required)
         A_parallel = 1 − Π (1 − A_i)         (any one of the redundant paths suffices)
WHERE:   A_i = availability of stage or path i over the same window [fraction]
CONSTRAINT: failures independent; a failed stage cannot be bypassed; the window is shared.
NUMERICAL EXAMPLE: five serial stages at 0.9995 each
         A_series = 0.9995^5 = 0.997502
         downtime = (1 − 0.997502) × 8,760 h = 0.002498 × 8,760 = 21.88 h/yr
INTERPRETATION: Five "three-nines-and-a-half" stages compose to less than three nines; the extra
            nine must be bought at the stage with the largest downtime contribution.
```

**Result 4.2c — Deadline allocation across a call chain.**

```text
FORMULA: T_slack = T_total − Σ T_i − T_queue
WHERE:   T_total = caller deadline [ms]; T_i = worst-case allowance for hop i [ms]
         T_queue = expected scheduling and retry queueing [ms]
CONSTRAINT: sequential hops; each hop's allowance is a stated worst case, not a mean.
NUMERICAL EXAMPLE: T_total = 800 ms, hops 250 + 250 + 120 = 620 ms
         T_slack = 800 − 620 = 180 ms
INTERPRETATION: The 180 ms is the only room for a retry; a second retry with a 150 ms attempt does
            not fit, so the retry policy must be one attempt, not two.
```

### 4.3 Variants

**Result 4.3a — Retry amplification with a geometric backoff policy.**

```text
FORMULA: attempts = (r^n − 1) / (r − 1)      for r > 1, n retries
         downstream_load = λ · attempts · f
WHERE:   r = retry fan factor [dimensionless]; n = maximum retries [count]
         f = downstream calls per attempt [calls]; λ = logical request rate [1/s]
CONSTRAINT: every attempt reaches the dependency; retries are not deduplicated upstream;
            the dependency is still accepting connections.
NUMERICAL EXAMPLE: r = 3, n = 3  ->  attempts = (81 − 1)/2 = 40
         λ = 2,000/s, f = 5  ->  downstream_load = 2,000 × 40 × 5 = 400,000 calls/s
INTERPRETATION: A policy chosen to "try a few times" multiplies offered load by forty; against a
            dependency already saturated at 2,000/s this policy is the outage.
```

**Result 4.3b — Tail exposure under fan-out.**

```text
FORMULA: P(at least one call exceeds q) = 1 − (1 − p_q)^m
WHERE:   p_q = probability a single call exceeds the q quantile [fraction]; m = parallel calls
CONSTRAINT: calls independent; the request completes only when all m return; no hedged request.
NUMERICAL EXAMPLE: p_99 = 0.01, m = 100
         P = 1 − 0.99^100 = 1 − 0.3660 = 0.634
INTERPRETATION: A request that fans out to one hundred leaves has a 63% chance of inheriting a
            p99 leaf; the composite distribution is not the leaf distribution.
```

**Result 4.3c — Quorum disjointness arithmetic.**

```text
FORMULA: P(read set disjoint from write set) = C(N − W, R) / C(N, R)
WHERE:   N = replica count; R = read quorum; W = write quorum [counts]
         C(a,b) = binomial coefficient, 0 when a < b
CONSTRAINT: replica sets chosen uniformly; no sloppy quorum, no handoff hints.
NUMERICAL EXAMPLE: N = 4, W = 2, R = 2
         P = C(2,2) / C(4,2) = 1 / 6 = 0.1667
INTERPRETATION: With R + W = N, one read in six can miss the newest write; R + W > N is the
            condition that removes the term entirely, not a rule of thumb.
```

**Result 4.3d — Sustainable throughput under key skew.**

```text
FORMULA: total_capacity = per_partition_capacity / hotspot_share
WHERE:   hotspot_share = fraction of total traffic landing on the hottest partition [fraction]
         per_partition_capacity = sustainable rate of one partition [items/s]
CONSTRAINT: routing is by key and the key distribution does not change during the window.
NUMERICAL EXAMPLE: per_partition_capacity = 25,000/s, hotspot_share = 0.40
         total_capacity = 25,000 / 0.40 = 62,500/s
         uniform case for contrast: 12 partitions × 25,000 = 300,000/s
INTERPRETATION: A single key taking 40% of traffic removes 79% of the nominal cluster capacity
            before any partition is saturated on average.
```

### 4.4 Limits and counterexamples

**4.4a — Little's law in a closed loop.** When the number of clients is fixed (a fixed thread pool
or a fixed set of devices), arrivals depend on completions, so λ is not an independent input. Using
L = λW to derive a new λ after a latency change produces a circular answer. Counterexample: ten
clients, each issuing a request, waiting for the response, then issuing the next. If response time
doubles, λ halves and L stays exactly ten. The formula is still an identity, but every term is
determined by the other two; nothing about capacity can be inferred from it.

**4.4b — Queue-wait approximation near saturation.** The W_q ≈ ρS/(2(1−ρ)) approximation diverges as
ρ → 1 and assumes low service-time variability. With heavy-tailed service times the realized wait at
ρ = 0.8 can exceed the approximation by a factor of several. Counterexample: a service with a 1%
population taking 2 s and the rest taking 20 ms has the same mean service time as a uniform 40 ms
service, and a far worse p99. The discriminating signal is the shape of the service-time histogram,
not its mean.

**4.4c — Retry amplification after the circuit opens.** The geometric formula assumes every attempt
reaches the dependency. Once a circuit breaker or admission cap rejects attempts, realized
downstream load is bounded by the cap, and the formula overstates load while understating client
latency. Counterexample: 40 logical attempts per request, of which 39 are rejected locally in
microseconds — downstream load is one attempt per request, and the client still waits for the full
backoff schedule.

**4.4d — Availability products under common-mode failure.** Multiplication assumes independence. If
all stages share a control plane, a DNS zone, a certificate authority, or a single region, one event
removes them together. Counterexample: five stages in one region, each 0.9995 available
independently, but with a shared regional control plane that is unavailable 0.1% of the year. The
product predicts 21.9 h/yr; the observed availability is bounded above by the control plane at
8.76 h/yr regardless of stage quality.

**4.4e — The unit of the window.** Every rate above is defined over a window. A rate over a window
that contains a stall is not the rate during normal operation, and a rate over a window shorter than
the reporting interval produces zeros that are read as outages (`CI-002`). Validity for all results
in this section requires the window to be at least several times the service time and free of
maintenance pauses; otherwise measure backlog slope directly.

Takeaway: each result carries a condition that a real system can violate, and the violation is where
the catalogued issues live.

## 5. Worked Numerical Example

**Scenario (ILLUSTRATIVE).** An order-intake service receives 2,000 requests/s at peak. Median
service time is 40 ms. The worker pool has 90 workers. Each request makes 5 downstream calls. The
retry policy allows 3 retries with a fan factor of 3. Logging emits 1.2 KB per event at 5,000
events/s. A cache in front of the catalog sets `ttl=86400` intending one day.

**Step 1 — Required concurrency.** By Result 4.1a, L = 2,000 × 0.040 = 80 in flight. The pool of 90
sits 10 above the requirement, which sounds safe and is not: wait grows non-linearly as the margin
shrinks.

**Step 2 — Realized utilization and wait.** Capacity = 90 / 0.040 = 2,250/s; ρ = 2,000/2,250 = 0.889.
W_q ≈ 0.889 × 0.040 / (2 × 0.111) = 0.03556 / 0.2222 = 0.160 s. Response time ≈ 200 ms. Doubling the
pool to 120 gives capacity 3,000/s, ρ = 0.667, W_q ≈ 0.040 s, response ≈ 80 ms. The 30 extra workers
halve wait; the first 90 workers bought almost nothing above the requirement.

**Step 3 — Incident backlog and drain.** Suppose capacity degraded to 1,800/s for 300 s while
arrivals held at 2,000/s. B = 200 × 300 = 60,000 items, which is 30 s of arrivals. Restoring
2,400/s drains at 400/s: T_d = 60,000/400 = 150 s. Restoring only 2,200/s drains at 200/s:
T_d = 300 s. The recovery rate determines the tail an operator experiences; the deficit determined
only the peak.

**Step 4 — Retry load.** attempts = (3^3 − 1)/(3 − 1) = 40. If every attempt reached the dependency
with 5 calls each, offered load would be 2,000 × 40 × 5 = 400,000 calls/s against a dependency that
was sized for 10,000 calls/s. This is why the retry budget, not the retry count, is the control:
capping extra attempts at 10% of base traffic bounds the amplification to 0.1 × 10,000 = 1,000
calls/s.

**Step 5 — Log volume and retention.** 5,000 events/s × 1,200 B = 6.0 MB/s = 518.4 GB/day (decimal
GB). Over 30 days, 518.4 × 30 = 15,552 GB ≈ 14.14 TiB (binary). At an illustrative 0.023 currency
units per GB-month, storing the full stream costs about 15,552 × 0.023 = 357.7 units per month.

**Step 6 — TTL unit check.** `ttl=86400` with a millisecond client means 86.4 s; the intended value
in milliseconds is 86,400,000. The factor between the two interpretations is 86,400,000/86,400 =
1,000. The observable consequence is a hit rate near zero, not an error, which is why the defect
survives dashboards that only track errors.

**Independent re-check.** Two routes to the concurrency figure: (i) L = λW = 80; (ii) a sampled
in-flight gauge averaging 78 over the window. The 2.5% difference is consistent with sampling
noise, so the derivation stands. Two routes to the backlog: (i) 200 × 300 = 60,000; (ii) 60,000/2,000
= 30 s of arrivals, which matches the observed queue depth in arrival-seconds. Two routes to the log
volume: (i) 5,000 × 1,200 = 6,000,000 B/s; (ii) 6,000,000 × 86,400 = 518,400,000,000 B/day =
518.4 GB/day. Both agree exactly.

**Sensitivity.** If service time rises from 40 ms to 60 ms at 2,000/s, required concurrency becomes
2,000 × 0.060 = 120, above the pool of 90: ρ = 2,000/(90/0.060) = 2,000/1,500 = 1.33, so the queue is
unstable and no steady-state wait exists. A pool of 150 gives capacity 2,500/s and ρ = 0.80 with
W ≈ 120 ms. The lesson is that a 50% increase in service time requires a 67% increase in workers to
hold the same utilization — pool sizing is sensitive to S, not to λ alone.

Takeaway: five distinct quantities in one scenario, each derived twice; the sensitivity step is what
converts the example into a sizing rule.
