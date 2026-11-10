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
`CI-001` through `CI-030` are stable within this file: they are referenced from the interview section
and from the cross-reference table, and they are not renumbered on revision.

## 1. File Header

**Scope.** This catalog covers the recurring defects that appear when a design, a report, or a
dashboard built on distributed infrastructure is reasoned about incorrectly: contracts between
producer and consumer, unit and precision handling, capacity and queue behaviour under load, and
vocabulary that hides a missing guarantee. Application bugs that are independent of distribution,
compiler defects, and security exploit classes beyond the authorization-ordering issue in `CI-021`
are outside scope.

**Reader task.** Given one observed symptom, the reader must be able to (a) assign it to exactly one
of four categories, (b) name the two or three mechanisms that could produce the same symptom, (c)
select a check that separates those mechanisms rather than confirming the first guess, (d) apply a
containment action bounded in time and blast radius, and (e) state a replay-based verification that
would fail if the fix were cosmetic.

**Categories.** The categories are operative, not decorative: they determine which evidence is worth
collecting first and which correction is even available.

| Category | Definition | First evidence to collect | Typical wasted effort |
|---|---|---|---|
| VIOLATED_ASSUMPTION | The system behaves as built; a precondition the reasoning relied on does not hold | Timestamps, versions, actor identity, ordering at the boundary | Tuning capacity when the problem is a clock or a contract |
| NUMERICAL_EDGE | Arithmetic is right in the ordinary range and wrong at a boundary or unit change | Numerator and denominator separately, plus the unit of each field | Rewriting business logic when a conversion is missing |
| SCALE_FAILURE | A relationship holds at small load and inverts as concurrency, cardinality, or skew grows | Rate, concurrency, and per-key distribution rather than averages | Adding replicas when the bottleneck is one key or one lock |
| CONCEPT_CONFUSION | A term asserts a guarantee the mechanism never provided | The mechanism's contract text and its actual failure behaviour | Debating definitions instead of testing an invariant |

**Severity scale.** Severity expresses consequence inside the declared teaching scope, not an
incident claim. It is used only to order the diagnosis queue.

| Level | Meaning in this catalog | Response expectation |
|---|---|---|
| S1 | Silent wrong answer that a downstream consumer treats as verified | Stop the consumer, then diagnose; never "observe for a week" |
| S2 | Correct answer with unavailable or unbounded latency for some population | Contain with a bounded fallback while diagnosing |
| S3 | Degraded but visible signal; operators can already see it is wrong | Diagnose in the normal queue |
| S4 | Cost, storage, or cardinality growth with no correctness effect yet | Schedule; the trend is the finding |

**Evidence policy.** Three labels appear: `DERIVED` for statements that follow from a named formula
and its stated conditions, `ILLUSTRATIVE` for constructed inputs, and `OPEN` for a cause the
available evidence does not settle. A cause is not upgraded from `OPEN` because a plausible story
fits; it is upgraded when a discriminating check returns a result that rules out the competing
mechanisms.

**Record shape.** Every issue carries eight fields: identifier, category, symptom, competing causes,
discriminating check, correction, prevention, and verification. A field is never "investigate
further"; an unresolved field states which measurement would resolve it and what value would falsify
the leading hypothesis.

**Title and count.** The file keeps the unit's catalogue title and its slot number; the catalog
currently holds thirty issues rather than twenty-five. The five additional entries were kept separate
because merging them would have combined distinct mechanisms under one identifier — wrap arithmetic
with empty-window arithmetic, for instance — and a merged entry cannot carry a discriminating check
that separates both.

**Triage order.** Diagnose contracts and clocks first, units and precision second, capacity and skew
third, vocabulary last. Reversing that order produces the classic failure of scaling a system that
is mislabelling its own timestamps, or of renaming a metric whose denominator was zero.

**What "fixed" means.** A correction changes the mechanism: a precondition check, a versioned
contract, a widened accumulator, a bounded retry budget. A mitigation contains harm without changing
the mechanism: deferring an item, serving a cached answer, refusing a write. Reporting a mitigation
as a fix is itself catalogued, under `CI-013` (drift without trace) and beside `CI-025` (override
without record).

## 2. Concept Map / Mental Model

The diagnosis loop has five states. Its value is that containment and correction are separate states
with separate exit conditions; collapsing them is how a hotfix becomes permanent architecture.

```text
  observe symptom
        |
        v
  classify category  ---> (ambiguous) ---> collect boundary evidence, reclassify
        |
        v
  enumerate competing causes (>= 2)
        |
        v
  run discriminating check ---> contradicts all causes? -> widen population or window
        |
        +--> confirmed cause
                |
                v
        contain (bounded, reversible, logged) --> correct mechanism --> replay verification
```

A second model is the **evidence ladder**. Each rung is cheaper than the one above it and answers a
narrower question. Climb only as far as the ambiguity requires, and never present a rung-1 number as
if it were rung-5 evidence.

```text
  rung 1  aggregate dashboard value          (a symptom exists)
  rung 2  numerator and denominator split    (which side moved)
  rung 3  per-key / per-shard distribution   (skew or uniform saturation)
  rung 4  raw record with both timestamps    (event time versus processing time)
  rung 5  replay of one record and boundary  (fix is real, not cosmetic)
```

A third model is the **category router**. When a symptom could belong to two categories, route by
which first evidence is cheaper to obtain and which correction is reversible. A clock problem
misdiagnosed as a capacity problem leads to adding replicas, which is expensive and conceals the
clock problem for another week.

```text
  symptom says "sometimes", "one customer", "only after deploy"
        -> VIOLATED_ASSUMPTION (ordering, version, identity)

  symptom says "zero", "infinity", "negative", "off by a constant factor"
        -> NUMERICAL_EDGE (unit, precision, overflow, empty window)

  symptom says "fine in staging", "grew over weeks", "one key is slow"
        -> SCALE_FAILURE (skew, cardinality, concurrency, retention)

  symptom says "we assumed", "it says eventually", "the doc says exactly"
        -> CONCEPT_CONFUSION (guarantee the mechanism never provided)
```

A fourth model is the **containment trade**. Every containment action spends one of four things:
freshness, completeness, latency, or money. Naming what is spent prevents claiming a free fix and
tells the reviewer what the user-visible cost is during the incident.

```text
  freshness spent     -> serve last-known value, label its age
  completeness spent  -> defer a subset, record the deferred key set
  latency spent       -> queue the work, publish a completion estimate
  money spent         -> add capacity or storage, bound the spend with a quota
```

A fifth model is **symptom versus mechanism**. Two mechanisms frequently share one symptom, so the
standing question is "what else produces exactly this number". A p99 of 900 ms can come from a hot
shard, a garbage-collection pause, a connection-pool wait, or a retry that finally succeeded. Only
the per-key distribution and the presence of a distinct wait-time metric separate them.

Takeaway: classification precedes correction, containment is not correction, and every containment
action has a named cost.

## 3. Foundational Definitions

Definitions carry units and validity ranges because most defects in this catalog are definitional
before they are computational.

| Term | Meaning in this file | Units / type | Valid range | Failure when confused |
|---|---|---|---|---|
| Event time | When the fact happened at its source | timestamp, UTC | any instant | Compared against processing time, producing negative durations |
| Processing time | When the pipeline handled the record | timestamp, UTC | at or after event time in a sane pipeline | Used as event time in windows, double-counting late data |
| Cut-off | The instant after which inputs no longer influence a decision | timestamp, UTC | published with the decision | Decision re-derived later from different inputs |
| Arrival rate λ | Items offered to a service per unit time | items/s | >= 0 | Measured over a window containing a stall |
| Service rate μ | Items one worker completes per unit time | items/s per worker | > 0 | Summed across workers without the worker factor |
| Capacity c·μ | Sustainable completion rate of a pool | items/s | > 0 | Compared against λ while omitting worker count |
| Utilization ρ | λ divided by capacity | dimensionless | [0,1) for stability | Reported against a wrong denominator |
| Service time S | Time one worker spends per item | seconds | > 0 | Confused with response time, which includes waiting |
| Backlog B | Items accepted but not completed | items | >= 0 | Allowed to go negative in an unsigned gauge |
| Concurrency L | Items in flight at an instant | items | >= 0 | Estimated from request rate alone, ignoring duration |
| Idempotency key | A value making a repeated request map to one effect | string, stable per logical request | non-empty, stable across attempts | Regenerated per attempt, so retries duplicate |
| Deduplication window | How long a completed key is remembered | seconds | longer than the retry horizon | Shorter than retries, so late retries duplicate |
| Quorum | The subsets whose intersection the protocol requires | set of nodes | R + W > N for intersection | Arithmetic done without counting subsets |
| Replication lag | Delay from durable write to replica visibility | seconds | >= 0 | Assumed zero by a read-after-write path |
| Lease | A time-bounded grant of exclusive authority | seconds | exceeds worst-case pause | Shorter than a pause, producing two holders |
| Retry budget | Fraction of extra attempts a caller may spend | dimensionless | commonly 0.1 to 0.2 of base traffic | Unbounded, so retries multiply load |
| Tail quantile | Value below which a stated fraction of observations fall | same unit as the measurement | 0 < q < 1 | Averaged across shards instead of merged |
| Cardinality | Number of distinct keys in a set | count | >= 0 | Reported as exact when it is an estimate |
| Error budget | Allowed failure in a window implied by an objective | minutes per window | >= 0 | Converted to a per-request probability and misapplied |
| Skew | Concentration of load on a key or partition | fraction of total | [1/N, 1] for N partitions | Hidden by averaging partition metrics |
| Fan-out | Downstream calls issued per inbound request | calls/request | >= 1 | Multiplied by retry attempts only after saturation |
| Headroom | Capacity held above expected peak | fraction of peak | commonly 0.15 to 0.30 | Set to zero "because autoscaling exists" |

**Symptom.** An observable the system emits without interpretation: queue depth, error rate, latency
quantile, reconciliation difference. A symptom does not name a cause even when its name suggests one.

**Cause.** The mechanism that, if changed, would remove the symptom. Causes are stated at the level
of a contract or an arithmetic operation, not at the level of a component, because "the database is
slow" is not something a fix can target.

**Discriminating check.** A measurement whose outcome differs between the competing causes. A check
returning the same result under both hypotheses is confirmation, not diagnosis, and it is the most
common wasted step in incident review.

**Mitigation.** A bounded action reducing harm while the cause is unknown: defer, cache, refuse,
route around. Mitigations carry an expiry; without one they become the operating design.

**Correction.** A change to the mechanism: a contract version, a precondition, a type width, a
budget, an ordering change that authorizes before access.

**Verification.** A replay of the triggering record plus one adjacent boundary record, with output,
reason code, side-effect count, and actor recorded. Verification exists to make an overstated fix
visible.

Takeaway: a term without units and a stated valid range cannot be checked, and a check that cannot
fail is not evidence.

## 4. Core Formulas with Full Derivation

Each result uses one anatomy: formula, where each symbol lives and in what unit, the constraint under
which it holds, a recomputed numerical example, and an interpretation naming what the number licenses
you to conclude. Plain-text fences are used so arithmetic can be pasted into a review document
without rendering change.

### 4.1 Base case

**Result 4.1a — Little's law as a pool-sizing identity.**

```text
FORMULA: L = λ · W
WHERE:   λ = arrival rate [requests/s], scalar, >= 0
         W = mean time in system [s], scalar, > 0
         L = mean number in system [requests], scalar, >= 0
CONSTRAINT: open system in steady state over the window; every arrival eventually departs;
            no long-lived population change inside the window.
NUMERICAL EXAMPLE: λ = 2,000 requests/s, W = 0.040 s
         L = 2,000 × 0.040 = 80 requests in flight
INTERPRETATION: A pool must hold at least 80 simultaneous workers to sustain this rate at this
            duration; any pool smaller than 80 forces waiting instead of service.
```

Derivation: in a window of length T, arrivals number λT and each spends W in system. Occupancy-time
totals λT·W; dividing by T gives the time-average occupancy. The identity is dimensional —
[1/s]·[s] = [1] — so a unit error surfaces immediately as a non-dimensionless L.

**Result 4.1b — Backlog accumulation under sustained imbalance.**

```text
FORMULA: B(t) = max(0, (λ − μ_sustain) · t);   T_d = B / (μ_drain − λ)
WHERE:   λ = offered rate [items/s]; μ_sustain = sustainable completion rate [items/s]
         μ_drain = completion rate after capacity is restored [items/s]
         B = backlog [items]; t = elapsed imbalance [s]; T_d = time to empty [s]
CONSTRAINT: constant rates, no loss, no retry traffic, service not yet degraded by queue depth.
NUMERICAL EXAMPLE: λ = 2,000; μ_sustain = 1,800; t = 300
         B = (2,000 − 1,800) × 300 = 200 × 300 = 60,000 items
         restated in arrival time: 60,000 / 2,000 = 30 s of traffic
INTERPRETATION: A 10% deficit held five minutes costs thirty seconds of arrivals to clear; the
            recovery rate, not the deficit, sets how long the tail lasts.
```

The restatement matters because operators compare backlog in items against rates in items per second
and conclude the backlog is small. Sixty thousand items sounds large; thirty seconds of traffic
sounds small; both describe one state.

### 4.2 General case

**Result 4.2a — Utilization and pool wait.**

```text
FORMULA: ρ = λ / (c · μ) = λ · S / c
         W_q ≈ ρ · S / (2 · (1 − ρ))
         W = S + W_q
WHERE:   c = worker count [count]; S = service time [s]; ρ = utilization [dimensionless]
         W_q = mean queue wait [s]; W = mean response time [s]
CONSTRAINT: single queue feeding c interchangeable workers, service-time variability low,
            arrivals not bursty, ρ < 1.
NUMERICAL EXAMPLE: λ = 2,000/s, S = 0.040 s, c = 100
         capacity = 100 / 0.040 = 2,500/s;  ρ = 2,000 / 2,500 = 0.80
         W_q ≈ 0.80 × 0.040 / (2 × 0.20) = 0.032 / 0.40 = 0.080 s = 80 ms
         W = 40 + 80 = 120 ms
INTERPRETATION: At 80% utilization each request spends twice its service time waiting; the pool is
            not "sized at peak" but sized with 25% spare, and the spare is consumed by queueing.
```

At c = 90 the same arithmetic gives capacity 2,250/s, ρ = 0.889, W_q ≈ 160 ms, W ≈ 200 ms. Ten
percent less capacity doubles the wait, which is why pool sizing does not scale with intuition.

**Result 4.2b — Composite availability.**

```text
FORMULA: A_series = Π A_i            (every stage required)
         A_parallel = 1 − Π (1 − A_i) (any one redundant path suffices)
WHERE:   A_i = availability of stage or path i over a shared window [fraction]
CONSTRAINT: failures independent; a failed stage cannot be bypassed; windows coincide.
NUMERICAL EXAMPLE: five serial stages at 0.9995 each
         A_series = 0.9995^5 = 0.997502
         downtime = (1 − 0.997502) × 8,760 h = 0.002498 × 8,760 = 21.88 h/yr
INTERPRETATION: Five stages at three-nines-and-a-half compose to below three nines; the next nine
            must be bought at the stage contributing the most downtime.
```

**Result 4.2c — Deadline allocation across a chain.**

```text
FORMULA: T_slack = T_total − Σ T_i − T_queue
WHERE:   T_total = caller deadline [ms]; T_i = worst-case allowance for hop i [ms]
         T_queue = expected scheduling and retry queueing [ms]
CONSTRAINT: sequential hops; each allowance is a stated worst case, not a mean.
NUMERICAL EXAMPLE: T_total = 800 ms; hops 250 + 250 + 120 = 620 ms
         T_slack = 800 − 620 = 180 ms
INTERPRETATION: Only 180 ms remains for a retry; a second attempt costing 150 ms does not fit, so
            the policy is one attempt, not two.
```

### 4.3 Variants

**Result 4.3a — Retry amplification under geometric backoff.**

```text
FORMULA: attempts = (r^n − 1) / (r − 1)   for r > 1
         downstream_load = λ · attempts · f
WHERE:   r = retry fan factor [dimensionless]; n = maximum retries [count]
         f = downstream calls per attempt [calls]; λ = logical request rate [1/s]
CONSTRAINT: every attempt reaches the dependency; retries are not deduplicated upstream;
            the dependency still accepts connections.
NUMERICAL EXAMPLE: r = 3, n = 3  ->  attempts = (81 − 1)/2 = 40
         λ = 2,000/s, f = 5  ->  downstream_load = 2,000 × 40 × 5 = 400,000 calls/s
INTERPRETATION: A policy chosen to "try a few times" multiplies offered load by forty; against a
            dependency sized for 10,000 calls/s the policy is the outage.
```

**Result 4.3b — Tail exposure under fan-out.**

```text
FORMULA: P(at least one leaf exceeds quantile q) = 1 − (1 − p_q)^m
WHERE:   p_q = per-leaf probability of exceeding q [fraction]; m = parallel leaves [count]
CONSTRAINT: leaves independent; the request completes only when all m return; no hedging.
NUMERICAL EXAMPLE: p_99 = 0.01, m = 100
         P = 1 − 0.99^100 = 1 − 0.3660 = 0.634
INTERPRETATION: A request fanning out to one hundred leaves inherits a p99 leaf with 63% chance;
            the composite distribution is not the leaf distribution.
```

**Result 4.3c — Quorum disjointness.**

```text
FORMULA: P(read set disjoint from write set) = C(N − W, R) / C(N, R)
WHERE:   N = replicas; R = read quorum; W = write quorum [counts]
         C(a,b) = binomial coefficient, zero when a < b
CONSTRAINT: replica sets chosen uniformly; no sloppy quorum, no handoff hints.
NUMERICAL EXAMPLE: N = 4, W = 2, R = 2
         P = C(2,2) / C(4,2) = 1 / 6 = 0.1667
INTERPRETATION: With R + W = N, one read in six can miss the newest write; R + W > N is the
            condition that removes the term, not a rule of thumb.
```

**Result 4.3d — Sustainable throughput under key skew.**

```text
FORMULA: total_capacity = per_partition_capacity / hotspot_share
WHERE:   hotspot_share = fraction of traffic on the hottest partition [fraction]
         per_partition_capacity = sustainable rate of one partition [items/s]
CONSTRAINT: key-based routing; the key distribution does not change inside the window.
NUMERICAL EXAMPLE: per_partition_capacity = 25,000/s, hotspot_share = 0.40
         total_capacity = 25,000 / 0.40 = 62,500/s
         uniform contrast: 12 partitions × 25,000 = 300,000/s
INTERPRETATION: A key taking 40% of traffic removes 79% of nominal cluster capacity before any
            partition is saturated on average.
```

### 4.4 Limits and counterexamples

**4.4a — Little's law in a closed loop.** With a fixed number of clients that issue, wait, and issue
again, arrivals depend on completions, so λ is not an independent input. Counterexample: ten clients
cycling. If response time doubles, λ halves and L remains exactly ten; every term is determined by
the other two and no capacity conclusion follows. The identity still holds and licenses nothing.

**4.4b — Pool-wait approximation near saturation.** W_q ≈ ρS/(2(1−ρ)) diverges as ρ → 1 and assumes
low variance. Counterexample: 1% of requests take 2 s and the rest 20 ms; mean service time is about
40 ms, identical to a uniform 40 ms service, while the p99 is two orders of magnitude worse. The
discriminating signal is the service-time histogram's tail, not its mean.

**4.4c — Retry amplification after admission control.** The geometric formula assumes every attempt
reaches the dependency. Once a circuit breaker or admission cap rejects attempts locally, realized
downstream load is bounded by the cap while the formula overstates it and understates client latency.
Counterexample: 40 logical attempts of which 39 are rejected in microseconds; downstream sees one
call per request and the client still waits the full backoff schedule.

**4.4d — Availability products under common-mode failure.** Multiplication assumes independence.
Counterexample: five stages in one region each at 0.9995, sharing a control plane unavailable 0.1% of
the year. The product predicts 21.88 h/yr; realised availability is bounded above by the control
plane at 0.001 × 8,760 = 8.76 h/yr regardless of stage quality.

**4.4e — The window unit.** Every rate here is defined over a window. A rate over a window that
contains a maintenance pause is not the operating rate, and a window shorter than the reporting
interval produces zeros read as outages (`CI-002`). Validity for all results in this section requires
the window to be several times the service time and free of pauses; otherwise measure backlog slope
directly.

Takeaway: each result carries a condition a real system can violate, and the violation is where the
catalogued issues live.

## 5. Worked Numerical Example

**Scenario (ILLUSTRATIVE).** An order-intake service receives 2,000 requests/s at peak. Median
service time is 40 ms. The worker pool has 90 workers. Each request makes 5 downstream calls. The
retry policy allows 3 retries with fan factor 3. Logging emits 1.2 KB per event at 5,000 events/s. A
cache in front of the catalog sets `ttl=86400` intending one day.

**Step 1 — Required concurrency.** From Result 4.1a, L = 2,000 × 0.040 = 80 in flight. A pool of 90
sits ten above the requirement, which sounds safe and is not: wait grows non-linearly as the margin
shrinks.

**Step 2 — Realized utilization and wait.** Capacity = 90/0.040 = 2,250/s; ρ = 2,000/2,250 = 0.889.
W_q ≈ 0.889 × 0.040 / (2 × 0.111) = 0.03556/0.2222 = 0.160 s, so response ≈ 200 ms. Raising the pool
to 120 gives capacity 3,000/s, ρ = 0.667, W_q ≈ 0.040 s, response ≈ 80 ms. The thirty added workers
halve wait; the first ninety barely improved on the bare requirement.

**Step 3 — Incident backlog and drain.** Suppose capacity degraded to 1,800/s for 300 s while
arrivals held at 2,000/s. B = 200 × 300 = 60,000 items, which is 30 s of arrivals. Restoring to
2,400/s drains at 400/s: T_d = 60,000/400 = 150 s. Restoring to only 2,200/s drains at 200/s:
T_d = 300 s. The recovery rate sets the tail an operator experiences; the deficit set only the peak.

**Step 4 — Retry load.** attempts = (3³ − 1)/(3 − 1) = 40. If every attempt reached the dependency
with five calls each, offered load would be 2,000 × 40 × 5 = 400,000 calls/s against a dependency
sized for 10,000 calls/s. This is why the retry budget, not the retry count, is the control: capping
extra attempts at 10% of base traffic bounds amplification to 0.10 × 10,000 = 1,000 calls/s.

**Step 5 — Log volume and retention.** 5,000 events/s × 1,200 B = 6.0 MB/s = 518.4 GB/day in decimal
units. Over 30 days, 518.4 × 30 = 15,552 GB = 14.14 TiB in binary units. At an illustrative 0.023
currency units per GB-month the full stream costs about 15,552 × 0.023 = 357.7 units per month.

**Step 6 — TTL unit check.** `ttl=86400` read by a millisecond client means 86.4 s; the intended
value in milliseconds is 86,400,000. The ratio between interpretations is 86,400,000/86,400 = 1,000.
The consequence is a hit rate near zero, not an error, which is why the defect survives dashboards
that track only errors.

**Independent re-check.** Two routes to occupancy: (i) L = λW = 80; (ii) a sampled in-flight gauge
averaging 78 over the window. The 2.5% gap is consistent with sampling noise, so the derivation
stands. Two routes to backlog: (i) 200 × 300 = 60,000; (ii) 60,000/2,000 = 30 s of arrivals, matching
the queue depth expressed in arrival-seconds. Two routes to log volume: (i) 5,000 × 1,200 =
6,000,000 B/s; (ii) 6,000,000 × 86,400 = 518,400,000,000 B/day = 518.4 GB/day. Exact agreement.

**Sensitivity.** If service time rises from 40 ms to 60 ms at 2,000/s, required concurrency becomes
2,000 × 0.060 = 120, above the pool of 90; ρ = 2,000/(90/0.060) = 2,000/1,500 = 1.333, so no
steady-state wait exists. A pool of 150 gives capacity 2,500/s, ρ = 0.80, W_q ≈ 0.80 × 0.060/0.40 =
120 ms, so W = 60 + 120 = 180 ms. A 50%
rise in service time therefore needs a 67% larger pool to hold the same utilization: sizing is
sensitive to S, not to λ alone.

Takeaway: five distinct quantities in one scenario, each derived twice, and the sensitivity step is
what converts the example into a sizing rule.

## 6. Visual / Geometric Intuition

**The utilization curve.** Response time against utilization is nearly flat below 0.5, bends upward
between 0.7 and 0.9, and is vertical at 1.0. Three points from Result 4.2a with S = 40 ms: ρ = 0.5
gives W ≈ 60 ms, ρ = 0.8 gives W ≈ 120 ms, ρ = 0.889 gives W ≈ 200 ms. Geometrically, a pool at 0.89
has no room for a 10% traffic spike: the spike does not add 10% latency, it moves the operating point
onto the vertical part of the curve.

```text
  W (ms)
  400 |                                              *
  300 |                                        *
  200 |                                  *
  120 |                        *
   80 |             *
   60 |      *
      +----+----+----+----+----+----+----+----+----+--> rho
         0.4  0.5  0.6  0.7  0.8  0.9  0.95  1.0
```

**The backlog staircase.** Backlog over time is a piecewise linear path whose slope is λ − μ; every
regime change is a corner, and the area under the path is the queued work. The drain segment's slope
depends only on the restored surplus, so two outages with identical peak backlog differ entirely in
tail: 400/s drains 60,000 items in 150 s while 200/s takes 300 s. A reviewer looking only at peak
backlog cannot distinguish them.

```text
  B
  60k |        /|                  outage A: drain 400/s -> 150 s
      |      /  |\
      |    /    | \__
      |  /      |    \__
    0 +---------+-------\-------> t
        0    300  450
```

**The fan-out triangle.** Each added leaf multiplies the chance of inheriting a tail. With p99 = 1%
per leaf, ten leaves give 1 − 0.99^10 = 9.56%, fifty give 39.50%, one hundred give 63.40%. The shape
is a saturating exponential: the first ten leaves cost about a percentage point each, the next ninety
cost roughly half a point each, so fan-out control matters most in the first order of magnitude.

```text
  P(tail) 0.63 |                                   *
          0.40 |                          *
          0.10 |        *
               +----+----+----+----+----+----+----> m
                   1   10   25   50   75  100
```

**The quorum grid.** For N replicas with quorums R and W the disjoint fraction is C(N−W,R)/C(N,R).
Cells with R + W > N are zero; cells with R + W = N are positive and shrink as N grows, which is the
geometric statement of "R + W = N is eventual, not strong".

```text
        W=1    W=2    W=3
  R=1  0.667  0.333  0.000     (N=3)
  R=2  0.333  0.000   n/a       (N=3)
  R=2  0.500  0.167  0.000      (N=4)
  R=2  0.600  0.300  0.100      (N=5)
```

**The error-budget rectangle.** A 99.9% objective over 30 days allows 0.001 × 43,200 = 43.2 minutes
of failure. Each incident spends a slice whose width is detection plus mitigation plus drain tail, so
a two-minute defect can cost twelve minutes of budget once the tail is counted. Drawing the three
segments separately is what makes the tail visible to a reviewer.

```text
  allowed 43.2 min
  |####|                                  detection 1 min
      |##########|                        mitigation 5 min
                 |######|                 drain tail 6 min
  used 12 min of 43.2 -> 27.8% of the budget for the month
```

Takeaway: these five pictures are the shared vocabulary between the arithmetic of section 4 and the
catalog of section 8; each catalogued defect corresponds to misreading one of them.

## 7. Comparison / Selection Table

The table selects a first action from a *symptom shape*, not from a component name. Read down the
symptom column, then across to the discriminator that separates the mechanisms.

| Symptom shape | Candidate mechanism A | Candidate mechanism B | Discriminating check | First bounded action |
|---|---|---|---|---|
| Values cluster at a suspiciously round bound | Unit mismatch (s versus ms) | Clamp or cap in code | Compare two consumers reading the same field | Label the unit in the schema, then recompute |
| Latency p50 fine, p99 collapsed | Hot key or hot shard | Client connection-pool wait | Per-key latency histogram versus pool wait gauge | Bound per-key work; raise the pool only if wait dominates |
| Throughput flat while replicas added | Single-writer bottleneck | Downstream dependency at its cap | Per-replica completion counts | Confirm the bottleneck, then batch or route around |
| Errors spike immediately after deploy | Contract or schema change | Capacity removed by the deploy | Diff config and pool size against the previous version | Roll back the contract change first |
| Backlog drains slower than predicted | Drain surplus smaller than assumed | Arrivals rose during recovery | Compare measured drain slope with (μ_drain − λ) | Hold intake flat until slope matches prediction |
| Duplicate side effects after an incident | Unstable idempotency key | Dedup window shorter than retries | Replay one key and count final effects | Extend the window before replaying anything |
| Read returns an older version than a prior write | Read served by a lagging replica | Cache holding a stale entry | Read with replica identity attached | Pin read-after-write to the primary for the session |
| Cost rises with no traffic change | Retention or cardinality growth | A new per-request log field | Bytes per event, then events per day | Cap retention on the high-cardinality stream |
| Two writers accept conflicting updates | Lease shorter than a pause | Clock skew between nodes | Node monotonic versus wall clock at grant time | Fence writes with a lease token |
| Totals drop with no deployment | Counter wrap | Series reset on process restart | Inspect the raw counter, not the derived rate | Widen the accumulator; handle wrap explicitly |

**Selecting a containment action.** The right containment spends the cheapest resource that keeps
the promise that matters most. When the promise is correctness, spend latency. When it is
availability, spend completeness and record what was deferred. When it is freshness, spend money.
When nothing is promised yet, publish a degraded status rather than inventing policy mid-incident.

| Containment | Spends | Reversible in | Requires recorded artifact | Wrong when |
|---|---|---|---|---|
| Defer the item with a reason code | Completeness | Minutes, by replay | Deferred key set and reason | Deferral volume exceeds the drain window |
| Serve a cached value with its age | Freshness | Minutes | Cache age and origin version | The consumer cannot tolerate stale reads |
| Reject the write with a retry hint | Availability for that caller | Immediate | Rejection count by caller | The caller has no retry path |
| Add capacity | Money | Hours for cost, minutes for effect | Pool size before and after | The bottleneck is a single key or lock |
| Disable a feature | Functionality | Minutes | Feature-flag state and owner | The feature is load-bearing for safety |
| Reduce retention | History | Not reversible | Retention before and after | The history is under legal hold |

**Selecting a diagnostic.** Prefer checks whose outcome differs between hypotheses. A check that
"looks at the dashboard" is worthless once the dashboard is the thing reporting the symptom.

| Check | Cost | Separates | Cannot separate |
|---|---|---|---|
| Split numerator and denominator | One query | Empty window from real movement | Two causes moving both terms |
| Per-key distribution | One grouped query | Skew from uniform saturation | Skew from one slow dependency |
| Event-time minus processing-time delta | Join on record id | Clock and backfill defects | Two pipelines sharing one clock error |
| Replay one record with its boundary twin | Minutes of engineering | Real fix from cosmetic fix | Causes that appear only under load |
| Config and schema diff against prior run | One diff | Drift from capacity | Capacity from dependency saturation |
| Attempts per logical request | One counter | Retry amplification from traffic growth | Amplification from a client bug |

Takeaway: choose the response from observed evidence and the promise being protected, and refuse to
let a familiar slogan pick the action.
## 8. Failure Modes & Edge Cases

The compact index gives one row per issue; the blocks below each row carry the mechanism, a
recomputed number where one applies, and what distinguishes the leading cause from its competitor.
Thirty issues are catalogued: `CI-001` to `CI-030`. Each block names its category so the router in
section 2 can be checked against the classification.

| ID | Category | Symptom |
|---|---|---|
| CI-001 | VIOLATED_ASSUMPTION | Records arrive after the decision cut-off |
| CI-002 | NUMERICAL_EDGE | A rate reads as zero because the denominator window is empty |
| CI-003 | SCALE_FAILURE | Backlog grows linearly with no error signal |
| CI-004 | CONCEPT_CONFUSION | A score threshold is treated as approval authority |
| CI-005 | VIOLATED_ASSUMPTION | A retry produces a second side effect |
| CI-006 | NUMERICAL_EDGE | Bytes are reported as item counts |
| CI-007 | SCALE_FAILURE | One key dominates latency and capacity |
| CI-008 | CONCEPT_CONFUSION | A planned test is reported as a result |
| CI-009 | VIOLATED_ASSUMPTION | A feature encodes information unavailable at decision time |
| CI-010 | NUMERICAL_EDGE | Rounding reverses a boundary decision |
| CI-011 | SCALE_FAILURE | Logging exhausts storage during an incident |
| CI-012 | CONCEPT_CONFUSION | No baseline exists for a claimed improvement |
| CI-013 | VIOLATED_ASSUMPTION | Configuration changes with no version trace |
| CI-014 | NUMERICAL_EDGE | Duration is negative after a timezone conversion |
| CI-015 | SCALE_FAILURE | Fan-out amplifies one request into an incident |
| CI-016 | CONCEPT_CONFUSION | Correlation is presented as causation |
| CI-017 | VIOLATED_ASSUMPTION | Schema evolution changes a field's meaning silently |
| CI-018 | NUMERICAL_EDGE | A counter wraps and totals collapse |
| CI-019 | SCALE_FAILURE | A cache cold start stampedes the origin |
| CI-020 | CONCEPT_CONFUSION | Exact and approximate answers are mixed in one field |
| CI-021 | VIOLATED_ASSUMPTION | Authorization is evaluated after the data is read |
| CI-022 | NUMERICAL_EDGE | NaN silently satisfies no comparison and changes routing |
| CI-023 | SCALE_FAILURE | A retry storm hides the original failure |
| CI-024 | CONCEPT_CONFUSION | A formula is applied outside its validity domain |
| CI-025 | VIOLATED_ASSUMPTION | A human override leaves no audit record |
| CI-026 | NUMERICAL_EDGE | A TTL is set in the wrong unit |
| CI-027 | SCALE_FAILURE | A connection pool is sized by concurrency, not by rate and duration |
| CI-028 | CONCEPT_CONFUSION | Averages of quantiles are reported as a quantile |
| CI-029 | VIOLATED_ASSUMPTION | A lease is shorter than a process pause |
| CI-030 | SCALE_FAILURE | Storage replication cost grows quadratically with retention copies |

### CI-001 — Records arrive after the decision cut-off
**Category:** VIOLATED_ASSUMPTION.
**Symptom:** A decision for period P changes when reprocessed the next day, or an item is scored
using inputs that were not yet published at the time of the decision.
**Mechanism:** the pipeline joins on the record's event time but the source publishes with a delay.
The decision uses whatever exists at processing time, so the effective cut-off is the processing
instant, not the published decision time.
**Competing causes:** (a) producer publishes late relative to its own event time; (b) consumer
applies a different timezone offset than the producer; (c) an upstream backfill rewrote history
after the decision. `ILLUSTRATIVE` arithmetic: with 50,000 records and 0.8% arriving after the
cut-off, 400 records are affected; if those records carry 4 decisions each, 1,600 decision fields
are re-derivable.
**Discriminating check:** for a fixed record id, compare `event_time`, `ingested_at`, and the
published cut-off in one table. Cause (a) shows a stable event time with a late ingestion; cause (b)
shows a constant offset equal to a timezone difference; cause (c) shows an ingestion time before the
cut-off with an event time rewritten later.
**Correction:** pin decisions to a cut-off and make post-cut-off arrivals enter a correction stream
rather than mutating the original decision. Record which inputs the decision saw.
**Prevention:** publish the cut-off as part of the dataset contract, and have the consumer refuse
records whose event time exceeds it.
**Verification:** replay the 400 late records and the boundary record exactly at the cut-off;
confirm the late ones route to the correction stream and the boundary one is accepted.

### CI-002 — A rate reads as zero because the denominator window is empty
**Category:** NUMERICAL_EDGE.
**Symptom:** A request rate or error rate drops to exactly 0 for a period when the service is in fact
down, so a dashboard shows calm during an outage.
**Mechanism:** the rate is computed as events divided by a window count. If no events are published
at all — because the exporter itself is down — the denominator is zero and the implementation emits
0, or the numerator is zero and the ratio is 0 for a different reason. Both render as the same value.
**Competing causes:** (a) exporter stopped publishing; (b) traffic genuinely ceased; (c) the metric
series was dropped by the store for cardinality reasons. Arithmetic: 0 events / 300 s window is
undefined, not 0.000/s; reporting 0.000/s asserts a measurement that was never taken.
**Discriminating check:** query a companion series that must be nonzero while the service runs — a
heartbeat counter, a scrape success count, or process uptime. A zero rate with zero scrapes is (a); a
zero rate with scrapes succeeding is (b); a missing series is (c).
**Correction:** emit `undefined` for an empty denominator; keep a staleness marker with the last
sample time so a consumer can distinguish "no events" from "no data".
**Prevention:** define an explicit empty-window policy per metric and make the dashboard render
staleness, not zero.
**Verification:** stop the exporter in a test environment and confirm the series shows undefined
with an age, then restore it and confirm the age resets.

### CI-003 — Backlog grows linearly with no error signal
**Category:** SCALE_FAILURE.
**Symptom:** Queue depth climbs steadily; every error counter stays near zero; latency at p50 looks
acceptable because the queue absorbs the imbalance.
**Mechanism:** offered rate exceeds sustainable completion rate. The system is not failing; it is
accumulating. From Result 4.1b, a 10% deficit of 200 items/s over 300 s is 60,000 items, which is
30 s of arrivals at 2,000/s.
**Competing causes:** (a) genuine arrival growth; (b) completion rate degraded by a slow dependency;
(c) a consumer group lost a member and never rebalanced. The three produce identical backlog slopes
and differ in the per-worker completion counter.
**Discriminating check:** plot backlog slope, arrivals, and completions per worker on one time axis.
Slope equal to λ − μ confirms the imbalance; a fall in per-worker completions with flat arrivals
points to (b); a fall in active worker count points to (c).
**Correction:** restore the surplus — either reduce intake to a stated cap or raise completion rate —
and hold the change until the measured slope matches the predicted surplus.
**Prevention:** alert on backlog slope and projected drain time, not on depth; depth without slope
cannot distinguish a large but draining queue from a compounding one.
**Verification:** measure drain time after the change and compare it with 60,000/(μ_drain − λ). A
drain of 150 s requires a 400/s surplus; an observed 300 s means the surplus was 200/s and the
restoration was half of what was assumed.

### CI-004 — A score threshold is treated as approval authority
**Category:** CONCEPT_CONFUSION.
**Symptom:** A model score above a threshold triggers an irreversible action with no named approver,
and a later review cannot say who authorized it.
**Mechanism:** a threshold is a sorting device; authority is a policy object with an owner, a scope,
and a version. Collapsing them means the numeric value becomes the only evidence of authorization.
**Competing causes:** (a) the policy was never written and the threshold filled the gap; (b) a policy
exists but the code path bypasses it; (c) a policy exists and grants automatic approval for a defined
score band, but the band was widened without review.
**Discriminating check:** for the triggering decision, retrieve the actor, policy identifier, policy
version, and the resulting action. (a) yields no policy identifier at all; (b) yields a policy
identifier that does not match the code path taken; (c) yields a policy version whose diff shows the
band change.
**Correction:** require a policy object with an owner and version for any consequential action, and
record the approver separately from the score.
**Prevention:** treat thresholds as configuration under change control, and make the action log
reference the policy version rather than the numeric value alone.
**Verification:** replay the decision with the policy version pinned and confirm the recorded
approver, policy version, and action match; then replay with a widened band and confirm the diff is
visible in the record.

### CI-005 — A retry produces a second side effect
**Category:** VIOLATED_ASSUMPTION.
**Symptom:** A payment, message, or file appears twice after a timeout, though the client issued one
logical request.
**Mechanism:** the transport retried; the server had already committed the effect but the
acknowledgement was lost. Without a stable idempotency key, the second attempt is a second request.
`ILLUSTRATIVE` arithmetic: 50,000 requests with a 0.04% retry rate gives 20 duplicated effects; if
each duplicate costs a manual reconciliation of 15 minutes, the operational load is 300 minutes.
**Competing causes:** (a) client generated a fresh key per attempt; (b) the deduplication window is
shorter than the retry horizon; (c) two independent producers emit the same logical event with
different keys.
**Discriminating check:** replay the same key twice and count final effects; then replay with a new
key and count again. One effect then two effects isolates key stability. If keys are stable but
duplicates persist, compare the dedup record's expiry against the observed retry interval.
**Correction:** persist the deduplication record before performing the effect, keyed on a value the
client derives from the logical request, with a window longer than the maximum retry horizon.
**Prevention:** make the key part of the API contract and reject requests without one for
non-idempotent operations.
**Verification:** replay the duplicate-triggering key plus a neighbouring key after the dedup window;
confirm exactly one effect per key and that the second response returns the original result.

### CI-006 — Bytes are reported as item counts
**Category:** NUMERICAL_EDGE.
**Symptom:** A capacity plan says "we process 20 GB per day" and a dashboard says "20,000 items",
and the two numbers are used interchangeably in a sizing discussion.
**Mechanism:** the field name carries no unit, so two consumers attach different units to the same
integer. A 1,000-byte average record makes 20,000,000,000 B equal 20,000,000 records — a factor of
1,000 apart from the item reading.
**Competing causes:** (a) decimal versus binary prefix (GB versus GiB, ratio 1.048576); (b) payload
bytes versus compressed bytes; (c) per-record bytes versus per-request bytes with multiple records.
**Discriminating check:** compute one day's volume from the record count and the average payload
independently, then compare with the reported figure. A ratio of exactly 1.048576 isolates (a); a
ratio near 5 to 10 isolates (b); a ratio equal to records per request isolates (c).
**Correction:** put the unit in the field name (`payload_bytes`, `record_count`) and reject arithmetic
across differently suffixed fields.
**Prevention:** dimensional analysis on every multiplication in a capacity model: bytes/s × s =
bytes, and bytes ÷ bytes/record = records.
**Verification:** recompute the capacity plan from raw counts and confirm the resulting plan changes
by the identified factor; a plan that is unchanged after the unit fix was never unit-sensitive.

### CI-007 — One key dominates latency and capacity
**Category:** SCALE_FAILURE.
**Symptom:** Cluster averages look healthy while one customer's requests take ten times longer, and
adding partitions does not raise total throughput.
**Mechanism:** routing is by key, and one key takes a large share. From Result 4.3d with a 40%
hotspot and 25,000 items/s per partition, total capacity is 62,500/s against a uniform 300,000/s
across twelve partitions — a 79% loss of nominal capacity.
**Competing causes:** (a) genuine key skew in the data; (b) a partitioner that maps many keys to one
partition (hash collision or a deliberate range split); (c) one key's work is intrinsically heavier
per item.
**Discriminating check:** per-key throughput and per-key latency histograms. (a) shows a heavy key
with normal work per item; (b) shows many distinct keys on the hot partition; (c) shows a normal key
count with high CPU per item on that key.
**Correction:** bound per-key work (batch or rate-limit that key), or split the key space using a
composite key with a bounded salt.
**Prevention:** monitor the maximum partition share, not the mean; alert when one partition exceeds
roughly twice the uniform share.
**Verification:** confirm the hot partition's share falls and that total throughput rises when
partitions are added; if throughput is unchanged by adding partitions, the bottleneck was never the
partition count.

### CI-008 — A planned test is reported as a result
**Category:** CONCEPT_CONFUSION.
**Symptom:** A status document says a change "improved latency by 30%" while no measurement with a
comparator exists.
**Mechanism:** a proposal, an estimate, or a modelled expectation is written in the past tense. The
artifact has no window, no population, and no comparator, so no reader can falsify it.
**Competing causes:** (a) the measurement exists but the comparator was omitted; (b) the measurement
exists with a comparator but a different population; (c) no measurement exists and the number came
from a model. Distinguishing them requires the raw artifact, not the sentence.
**Discriminating check:** ask for window, population, numerator, denominator, and comparator. A
missing comparator without a missing window is (a); a different population is (b); no artifact at all
is (c).
**Correction:** relabel the claim as `not-measured` with the protocol that would measure it, and
publish the comparator alongside any later number.
**Prevention:** require an evidence label on every quantitative claim and reject past-tense verbs
attached to unexecuted work.
**Verification:** re-read the claim after correction and confirm a reader can reproduce it from the
named artifact; if the artifact does not exist, the claim must say so.

### CI-009 — A feature encodes information unavailable at decision time
**Category:** VIOLATED_ASSUMPTION.
**Symptom:** Offline evaluation is excellent and live performance collapses.
**Mechanism:** the feature is joined from a table that is populated later — a settlement record, a
review, a churn label — so it carries future information. The model learns a relationship that the
serving path cannot supply.
**Competing causes:** (a) point-in-time leakage through an unrestricted join; (b) legitimate feature
whose source is unavailable at serving time due to pipeline lag; (c) distribution shift between the
evaluation window and live traffic.
**Discriminating check:** for each feature, state the earliest timestamp at which its value is
knowable and compare with the decision time. (a) shows a knowable time after the decision; (b) shows
a knowable time before the decision but a pipeline delay; (c) shows correct availability with a
changed input distribution.
**Correction:** remove the field or rebuild it with a point-in-time join that returns the value as of
the decision instant, including nulls where it did not exist.
**Prevention:** make the point-in-time availability part of the feature definition and test it by
recomputing a sample of features as of an earlier instant.
**Verification:** recompute the feature for a sample at the decision instant and confirm the values
match serving; a mismatch means the join is still reading the present.

### CI-010 — Rounding reverses a boundary decision
**Category:** NUMERICAL_EDGE.
**Symptom:** Two records with values 0.496 and 0.504 land on the same side of a 0.50 threshold, or a
record displayed as 0.50 is rejected.
**Mechanism:** the comparison is performed on a rounded value, or the threshold sits at a precision
the display cannot represent. 0.496 rounded to two decimals is 0.50, which then compares as equal to
the threshold; the unrounded comparison rejects it, a difference of 0.004.
**Competing causes:** (a) comparison on a rounded display value; (b) a threshold chosen at a float
boundary where representation error matters; (c) a tie policy that is not stated, so equal values go
either way.
**Discriminating check:** recompute the decision from unrounded inputs for the boundary record and
compare with the stored decision. A reversal only at the boundary isolates (a) or (c); a reversal for
values several units away from the boundary isolates representation error (b).
**Correction:** compare unrounded values, specify the tie policy (inclusive or exclusive), and keep
the unrounded value in the audit record.
**Prevention:** choose thresholds with a margin larger than the display precision, and test both
sides of the boundary in the decision test suite.
**Verification:** replay the pair 0.496 and 0.504 and confirm different outcomes; a shared outcome
proves the comparison is being made on rounded values.

### CI-011 — Logging exhausts storage during an incident
**Category:** SCALE_FAILURE.
**Symptom:** During the incident, log ingestion multiplies and the log store rejects writes or the
retention job deletes the window under investigation.
**Mechanism:** each failed request logs multiple events, each retry logs again, and the payload grows
with stack detail. `ILLUSTRATIVE`: 40,000 events/s at 500 B each is 20 MB/s = 1,728 GB/day = 1.57
TiB/day; a 30-day retention of that stream is 47.1 TiB.
**Competing causes:** (a) per-retry logging without deduplication; (b) expanded payload on error
paths; (c) a second consumer replaying the stream and re-emitting logs after failures.
**Discriminating check:** bytes per event during normal and incident windows, and events per logical
request in both. Stable bytes with multiplied events isolates (a); rising bytes per event isolates
(b); a duplicated stream identifier isolates (c).
**Correction:** sample error-path logs at a stated rate, cap payload size, and set a quota per
service so one producer cannot consume the shared budget.
**Prevention:** budget log bytes as a first-class capacity number and alert on bytes per request
rather than total volume.
**Verification:** replay an incident-sized burst into a staging store and confirm the quota holds and
the sampled records still identify the failing request id.

### CI-012 — No baseline exists for a claimed improvement
**Category:** CONCEPT_CONFUSION.
**Symptom:** A method is reported as better than its predecessor, but the predecessor was never
measured in the same window with the same population.
**Mechanism:** the comparison is between a new measurement and a remembered impression. Without a
no-change comparator, seasonality, traffic mix, and instrumentation changes are indistinguishable
from the effect being claimed.
**Competing causes:** (a) baseline never measured; (b) baseline measured but on a different
population; (c) baseline measured on the same population but in a different season.
**Discriminating check:** request the baseline artifact with its window and population. Absence is
(a); a population difference is (b); a window that does not overlap in season is (c). Only (b) and
(c) can be repaired by reweighting; (a) requires a new run.
**Correction:** restore a comparator — either a no-change path, a holdout, or a documented manual
process — and measure both in the same window.
**Prevention:** require the comparator to be named in the experiment plan before the change ships.
**Verification:** compute the effect twice, once against the restored baseline and once against the
manual comparator; if the two disagree in sign, the claim is unresolved rather than positive.

### CI-013 — Configuration changes with no version trace
**Category:** VIOLATED_ASSUMPTION.
**Symptom:** A metric moves and nobody can establish which configuration produced the affected
records; a rerun does not reproduce the original output.
**Mechanism:** configuration is read at run time from a mutable source, so the record of what was
used exists only in the operator's memory.
**Competing causes:** (a) a mutable default changed in place; (b) a manual override applied for one
run and not reverted; (c) a dependency's default changed on upgrade.
**Discriminating check:** compare the configuration hash recorded with the run against the current
hash, and diff the two bodies. A diff in the service's own config isolates (a) or (b); an unchanged
service config with a changed dependency version isolates (c).
**Correction:** pin configuration per run, record its hash with the output, and require an approval
record for changes to values that affect decisions.
**Prevention:** treat decision-relevant configuration as code with review, and fail the run when the
hash is absent.
**Verification:** rerun one affected period with the pinned hash and confirm byte-identical outputs
for that period; a diff means another input is also unpinned.

### CI-014 — Duration is negative after a timezone conversion
**Category:** NUMERICAL_EDGE.
**Symptom:** A duration field is negative, or a session length is absurd by exactly one hour.
**Mechanism:** one timestamp is local time and the other is UTC, or a daylight-saving transition
shifts one of them. A one-hour offset on a 24-hour span produces a duration that is either 23/24 or
25/24 of the truth — a 4.17% error — and on a short span it can invert the sign.
**Competing causes:** (a) mixed timezone representation; (b) daylight-saving transition inside the
window; (c) clock skew between two hosts.
**Discriminating check:** convert both timestamps to UTC and recompute. A duration that becomes
correct isolates (a); a residual one-hour error on the transition date isolates (b); a residual
sub-second error that varies by host isolates (c).
**Correction:** store UTC with an explicit offset, convert at the boundary, and reject records whose
computed duration is negative.
**Prevention:** make the type carry the zone and disallow arithmetic between differently zoned
values.
**Verification:** replay a record spanning a transition date plus a normal record; confirm the
transition record yields 23 or 25 hours as appropriate and the normal record yields exactly 24.

### CI-015 — Fan-out amplifies one request into an incident
**Category:** SCALE_FAILURE.
**Symptom:** A single inbound request produces dozens of downstream calls, and a modest traffic rise
saturates a dependency that looked over-provisioned.
**Mechanism:** per-request fan-out multiplies with retries. From Result 4.3a, 3 retries at fan factor
3 gives 40 attempts; at 5 downstream calls per attempt, 2,000 requests/s becomes 400,000 calls/s.
**Competing causes:** (a) unbounded per-request fan-out (one call per item in a list); (b) retries
multiplying an already-large fan-out; (c) a client bug issuing duplicate requests.
**Discriminating check:** calls per inbound request, measured at the dependency with the request id
propagated. A constant high count isolates (a); a count that rises with dependency error rate
isolates (b); a count that is high only for one client isolates (c).
**Correction:** cap calls per request, batch the list into one call, and put an attempt budget at the
caller.
**Prevention:** publish a fan-out budget per endpoint and test it with a dependency that returns
errors on a schedule.
**Verification:** replay one inbound request with the dependency returning a 5% error rate and count
downstream calls; the count must stay within the budget rather than approaching the attempt product.

### CI-016 — Correlation is presented as causation
**Category:** CONCEPT_CONFUSION.
**Symptom:** A dashboard shows two series moving together and a change is proposed on that basis; the
change does not move the outcome.
**Mechanism:** a third variable drives both, or the population was selected on the outcome. The
observed association is real and the proposed intervention is not implied by it.
**Competing causes:** (a) common cause driving both series; (b) selection on the outcome; (c) reverse
ordering, where the outcome influences the candidate cause.
**Discriminating check:** draw the causal graph and name the comparator. A common cause is testable
by conditioning on it; selection is testable by checking how the population was enrolled; reverse
ordering is testable by lag structure.
**Correction:** narrow the claim to association, or design an intervention with a comparator and a
stated estimand.
**Prevention:** require a stated estimand — the population, the intervention, and the comparison —
before any effect claim is written.
**Verification:** state what observation would falsify the causal claim; if no such observation is
specified, the claim is association only and must be labelled so.

### CI-017 — Schema evolution changes a field's meaning silently
**Category:** VIOLATED_ASSUMPTION.
**Symptom:** Downstream aggregates change level by a factor of 1,000 after a producer release, with
no schema error raised.
**Mechanism:** a field keeps its name and type but changes unit or semantics — seconds to
milliseconds, one row per order to one row per line item. Consumers compiled against the name keep
working and produce wrong numbers.
**Competing causes:** (a) unit change inside the same numeric type; (b) grain change (row meaning);
(c) null semantics change, where a missing value becomes an explicit zero.
**Discriminating check:** compare the distribution of the field before and after the release. A
factor-of-1,000 shift isolates (a); a change in row count per key isolates (b); a rise in zero
frequency isolates (c).
**Correction:** version the field or the dataset, migrate consumers explicitly, and keep the old
field until the migration completes.
**Prevention:** make unit part of the field name and part of the schema contract test.
**Verification:** run the contract test against the new producer version; a unit change must fail the
test rather than pass silently.

### CI-018 — A counter wraps and totals collapse
**Category:** NUMERICAL_EDGE.
**Symptom:** A cumulative total drops sharply, or a rate goes negative, with no deployment.
**Mechanism:** a 32-bit unsigned counter wraps at 4,294,967,295. At 10,000 items/s that is 429,496 s
= 119.3 h ≈ 4.97 days; a 32-bit millisecond timestamp wraps after 2,147,483,647 ms = 24.86 days.
**Competing causes:** (a) unsigned wrap; (b) signed overflow at 2,147,483,647 producing a negative;
(c) a store that resets the series on restart, which looks like a wrap in the derived rate.
**Discriminating check:** inspect the raw counter value rather than the derived rate. A drop of
exactly 2^32 isolates (a); a sign flip isolates (b); a drop to a small value coinciding with a
process start time isolates (c).
**Correction:** use a 64-bit accumulator, and compute rates as deltas with an explicit wrap handler
rather than as differences of cumulative values.
**Prevention:** test the maximum value in a unit test and reject narrowing conversions.
**Verification:** replay a sequence that crosses the wrap boundary and confirm the derived rate
remains positive and continuous; a negative spike means the handler is absent.

### CI-019 — A cache cold start stampedes the origin
**Category:** SCALE_FAILURE.
**Symptom:** After a deploy or an expiry event, origin load spikes far above steady state and the
origin saturates while caches are being filled.
**Mechanism:** all clients miss at once. With 10,000 requests/s and a 300 ms fill time, the origin
sees 3,000 concurrent fills — the full concurrency of the missed keys, not a warmed fraction.
**Competing causes:** (a) simultaneous expiry from a shared TTL with no jitter; (b) a cold cache
after a deploy; (c) a hot key evicted repeatedly by a competing key.
**Discriminating check:** origin concurrency and request distribution by key. A single expiry instant
for many keys isolates (a); all keys missing at process start isolates (b); one key dominating misses
isolates (c).
**Correction:** single-flight the fill so one request populates the value and the rest wait, add
jitter to TTLs, and cap concurrent fills at the origin.
**Prevention:** warm the cache before routing traffic, and set TTLs with a random offset of a few
percent.
**Verification:** expire a key set in staging and confirm origin concurrency stays near one per key;
a concurrency close to the request rate means single-flight is not engaged.

### CI-020 — Exact and approximate answers are mixed in one field
**Category:** CONCEPT_CONFUSION.
**Symptom:** A distinct-count field is exact for small populations and approximate for large ones,
and a downstream reconciliation treats both as exact.
**Mechanism:** a sketch is used above a threshold. A 4,096-register sketch has relative error about
1.04/√4,096 = 1.625%, so a reported count of 100,000 carries roughly ±1,625 — enough to fail an
exact-match reconciliation.
**Competing causes:** (a) sketch approximation; (b) exact count over a partially loaded window;
(c) sampling applied to the source.
**Discriminating check:** recompute the count exactly for the same window. A difference within the
sketch's stated error bound isolates (a); a difference proportional to ingestion lag isolates (b); a
difference matching the sample fraction isolates (c).
**Correction:** label the field as approximate with its error bound, or compute exactly for the
populations that require exactness and keep both fields distinct.
**Prevention:** put the precision guarantee in the field name and in the consumer contract.
**Verification:** compare the sketch, an exact count, and the reconciliation consumer on the same
window; the sketch must be within its bound and the consumer must not treat it as exact.

### CI-021 — Authorization is evaluated after the data is read
**Category:** VIOLATED_ASSUMPTION.
**Symptom:** A caller receives a record outside its permitted scope, or an audit finds a read that
should have been refused.
**Mechanism:** the query fetches a broad set and filters afterwards in application code. Any path
that returns early, logs the fetched rows, or throws before filtering exposes data. `ILLUSTRATIVE`:
a scope check placed after a query that returns 250 rows for 10 permitted keys reads 25 times the
permitted data per request.
**Competing causes:** (a) filter applied after fetch; (b) filter applied on a field the caller can
influence; (c) a cache key that omits the caller's scope, serving another tenant's entry.
**Discriminating check:** trace the requested scope, the query predicate, and the returned rows. A
predicate without the scope term isolates (a); a scope term derived from request input isolates (b);
a cache hit rate that crosses tenants isolates (c).
**Correction:** put the scope into the query predicate so the store never returns out-of-scope rows,
and refuse at the boundary rather than filtering later.
**Prevention:** test with a forbidden descendant — a key that exists but is outside the caller's
scope — and assert a refusal, not an empty result indistinguishable from "not found".
**Verification:** issue the request as a caller whose scope excludes one existing key; confirm the
store's read count for that key is zero, not one.

### CI-022 — NaN silently satisfies no comparison and changes routing
**Category:** NUMERICAL_EDGE.
**Symptom:** Records with missing arithmetic inputs disappear from both the accepted and rejected
branches, so totals do not reconcile.
**Mechanism:** every comparison with NaN is false, including equality. A filter that routes
`score >= threshold` to accept and `score < threshold` to reject drops NaN from both.
**Competing causes:** (a) NaN from a division by zero or an empty mean; (b) NaN from a parsing
failure that produced a missing numeric; (c) infinity from an overflow, which compares but
propagates.
**Discriminating check:** test `is_finite` before ordering and count non-finite values by source.
A count concentrated at one computation isolates (a); a count concentrated at one input format
isolates (b); an infinite magnitude isolates (c).
**Correction:** quarantine non-finite values into a named branch with a reason code, and make
acceptance require an explicit finite check.
**Prevention:** assert finiteness at the boundary where the numeric enters, not at the point of
comparison.
**Verification:** replay a NaN record and a finite boundary record; both must land in a named branch,
and the counts must sum to the input count.

### CI-023 — A retry storm hides the original failure
**Category:** SCALE_FAILURE.
**Symptom:** Attempt rate rises, success rate falls, and the dependency's logs fill with rejections
while the original defect is buried under retry noise.
**Mechanism:** synchronized retries multiply load. Forty attempts per request at 2,000 requests/s is
80,000 attempts/s, of which the dependency can serve none; the failures are retries, not new demand.
**Competing causes:** (a) synchronized backoff with no jitter; (b) unbounded retry depth; (c) an
upstream timeout set shorter than the dependency's normal service time, converting slow successes
into retries.
**Discriminating check:** plot attempt rate against distinct logical request rate. A ratio far above
one with clustered attempt timestamps isolates (a) or (b); a ratio near one with elevated timeouts
isolates (c).
**Correction:** apply jitter, cap the attempt budget at a fraction of base traffic, and trip a circuit
so attempts stop while the dependency recovers.
**Prevention:** set client timeouts above the dependency's p99 and below the caller's own deadline;
make the retry budget a configured quota rather than a per-call constant.
**Verification:** replay the failing dependency condition with the budget in place and confirm the
attempt rate stays bounded while the error is still visible in the logical-request series.

### CI-024 — A formula is applied outside its validity domain
**Category:** CONCEPT_CONFUSION.
**Symptom:** A queueing or availability estimate is arithmetically correct and operationally wrong
by a wide margin.
**Mechanism:** the formula's assumptions — independence, stationarity, low variability, steady state
— do not hold. The output is still a number, which is why the error is invisible.
**Competing causes:** (a) non-stationary arrivals (a burst inside the window); (b) dependent
failures sharing a control plane; (c) heavy-tailed service times invalidating the wait approximation.
**Discriminating check:** test each assumption directly — arrival rate over sub-windows, failure
correlation across stages, and the service-time histogram's tail. The assumption that fails is the
one whose removal changes the estimate most.
**Correction:** switch method or state the estimate's condition and defer the conclusion; never
present the number without its condition.
**Prevention:** attach the validity condition to the formula in the design document and review the
condition when the workload changes.
**Verification:** construct the boundary case that violates the assumption and confirm the estimate
diverges; if it does not diverge, the assumption was not load-bearing for that estimate.

### CI-025 — A human override leaves no audit record
**Category:** VIOLATED_ASSUMPTION.
**Symptom:** The final action differs from the recommendation, and the reason cannot be recovered;
a later reconciliation shows unexplained divergence.
**Mechanism:** the override happens through a side channel — a manual database update, an
undocumented admin tool — so the actor and reason never reach the decision ledger.
**Competing causes:** (a) override through a side channel; (b) an actor field that exists but is
populated with a service account; (c) the ledger records the recommendation and never the final
action.
**Discriminating check:** compare the recommendation ledger with the system of record for the final
action. A difference with no ledger row isolates (a); a row with a generic actor isolates (b); a
ledger that stops at recommendation isolates (c). `ILLUSTRATIVE`: 37 unexplained divergences out of
1,200 decisions is 3.08% of the population.
**Correction:** require overrides to go through a recorded path with actor, reason code, and
timestamp; reconcile the two ledgers daily.
**Prevention:** remove write access that bypasses the ledger, and alert on divergence between
recommendation and final action.
**Verification:** perform one override through the recorded path and one attempt through the blocked
path; the first must produce a ledger row and the second must fail.

### CI-026 — A TTL is set in the wrong unit
**Category:** NUMERICAL_EDGE.
**Symptom:** A cache has a near-zero hit rate or serves data hours stale, with no errors logged.
**Mechanism:** `ttl=86400` means 86.4 s to a millisecond client and 24 h to a second client — a
factor of 1,000. Neither interpretation raises an error, so the defect appears only as a hit-rate
or freshness anomaly.
**Competing causes:** (a) wrong unit for the client; (b) TTL applied to the wrong key namespace, so
entries never match; (c) a write path that bypasses the cache and invalidates entries immediately.
**Discriminating check:** inspect the configured integer, the client's unit, and the observed entry
age at eviction. An entry age near 86 s for an intended day isolates (a); a hit rate near zero with
normal entry age isolates (b); an entry age of zero isolates (c).
**Correction:** name the unit in the configuration key and assert the value falls in an expected
range at startup.
**Prevention:** reject configurations whose TTL is outside a stated range for the namespace, and
test the configured value against the intended duration.
**Verification:** read an entry twice separated by more than the intended eviction age and confirm
the second read misses; then repeat with the corrected value and confirm a hit.

### CI-027 — A connection pool is sized by concurrency, not by rate and duration
**Category:** SCALE_FAILURE.
**Symptom:** Requests wait for a connection while the database reports low CPU; raising pool size
helps briefly and then worsens latency.
**Mechanism:** required concurrency is rate × hold time, not the number of application threads. At
200 requests/s and a 40 ms hold, 8 connections are needed; a pool of 200 threads each holding a
connection multiplies the database's concurrency by 25 and produces contention that looks like
database slowness.
**Competing causes:** (a) pool larger than the database can serve concurrently; (b) pool smaller
than required concurrency, so requests queue on acquisition; (c) connections held across a
non-database wait, inflating hold time.
**Discriminating check:** measure hold time and acquisition wait separately. High acquisition wait
with low hold time isolates (b); high hold time with normal acquisition isolates (c); both normal
with database contention isolates (a).
**Correction:** size the pool from rate × hold time plus headroom, and cap it at a value the
database can serve; release connections before non-database waits.
**Prevention:** publish the required-concurrency calculation next to the pool configuration and
review it when service time changes.
**Verification:** raise the hold time by a factor and confirm acquisition wait changes as the
formula predicts; a wait that does not move means the pool is not the constraint.

### CI-028 — Averages of quantiles are reported as a quantile
**Category:** CONCEPT_CONFUSION.
**Symptom:** A cluster-wide p99 looks acceptable while individual shards exceed the objective, or the
reverse.
**Mechanism:** quantiles are not additive. Averaging twelve shard p99 values where eleven are 120 ms
and one is 900 ms gives 185 ms, which is not the p99 of the merged population and understates the
slow shard by a factor of 4.86.
**Competing causes:** (a) arithmetic mean of shard quantiles; (b) quantile computed from averaged
inputs rather than merged observations; (c) shard weights ignored, so a tiny shard contributes
equally.
**Discriminating check:** recompute the quantile from merged raw observations or from a mergeable
sketch, and compare. A large gap isolates (a) or (b); a gap that disappears when shards are weighted
isolates (c).
**Correction:** merge distributions rather than averaging summaries, or keep per-shard quantiles and
report the worst shard explicitly.
**Prevention:** forbid arithmetic on quantile-typed fields in the metrics layer and expose a
mergeable representation.
**Verification:** compute the merged p99 and the shard-averaged p99 for the same window; they must
differ, and the reported value must be the merged one.

### CI-029 — A lease is shorter than a process pause
**Category:** VIOLATED_ASSUMPTION.
**Symptom:** Two nodes perform the same exclusive work; both believe they hold the lease.
**Mechanism:** the holder pauses — a long garbage-collection stop, a suspended virtual machine, a
blocked system call — past lease expiry. The new holder begins, the old one resumes and continues
writing. A 7 s lease with a 5 s pause leaves 2 s of margin, so overlap requires a pause longer than
7 s — or a pause starting immediately after a renewal that runs past the next renewal deadline,
which is the same 7 s bound measured from the last successful renewal.
**Competing causes:** (a) pause longer than the lease; (b) clock skew making expiry evaluation
disagree; (c) renewal succeeding late but writes continuing regardless.
**Discriminating check:** compare the grant time on the local monotonic clock with the wall clock at
the moment of a write, and inspect GC or suspend logs around the overlap. A pause overlapping expiry
isolates (a); a monotonic-versus-wall divergence isolates (b); a renewal after the write isolates (c).
**Correction:** fence writes with a lease token that the store validates, so a resumed holder's write
is rejected, and set the lease above the observed maximum pause.
**Prevention:** measure pause durations and derive the lease from the maximum, not from the median.
**Verification:** simulate a pause exceeding the lease and confirm the resumed holder's write is
rejected; a successful write proves fencing is absent.

### CI-030 — Storage replication cost grows with retention copies
**Category:** SCALE_FAILURE.
**Symptom:** Storage cost rises faster than data volume; each additional replica or snapshot
multiplies the bill.
**Mechanism:** retention multiplies copies. `ILLUSTRATIVE`: 12 TB of live data with 8 retained copies
is 96 TB; at an illustrative 0.023 currency units per GB-month that is 96,000 GB × 0.023 = 2,208
units per month, before egress or request charges.
**Competing causes:** (a) copy count growth from retention policy; (b) snapshot chain where each
snapshot stores a full copy; (c) egress from a cross-region replica.
**Discriminating check:** bytes stored per retained copy, and the ratio of bill growth to live-data
growth. Equal growth isolates (a); superlinear growth isolated by full-copy snapshots isolates (b); a
line item concentrated in transfer isolates (c).
**Correction:** switch to incremental snapshots, set retention per class rather than uniformly, and
tier older copies to cheaper storage.
**Prevention:** model cost as bytes × copies × price and review the copy count whenever retention
changes.
**Verification:** reduce copies by one and confirm the bill falls by 12,000 GB × price; a smaller
fall means the copies were not independent.

Takeaway: each issue pairs a mechanism with a check whose outcome differs between the competing
causes, and each correction changes a mechanism rather than the wording of a report.
## 9. Complexity & Cost Analysis

Diagnosis is itself a workload, and its cost has to be bounded or the investigation becomes the
incident. Three costs matter: the cost of the checks, the cost of the corrections, and the cost of
carrying the wrong fix.

**Cost of a check.** Each rung of the evidence ladder has an order-of-growth price.

```text
  aggregate value            O(1) query over a pre-aggregated series
  numerator/denominator      O(1) two series reads
  per-key distribution       O(K) grouped scan, K = distinct keys in the window
  record-level join          O(N) over records in the window
  replay of one record       O(1) records plus the dependency calls it makes
```

A per-key grouped scan over K = 10,000,000 keys is not a free diagnostic; it is a full scan whose
cost grows with cardinality, exactly the quantity `CI-020` warns about. Bound it by sampling keys,
not by scanning all of them, and state the sample fraction with the result.

**Cost of validation work in the hot path.** A per-item check costing 1 µs at 5,000 items/s consumes
5,000 × 10⁻⁶ = 0.005 core-seconds per second, or 0.5% of one core. The same check at 500,000 items/s
consumes 0.5 cores. The check is cheap and its growth is linear, which is why the failure mode is not
CPU but the temptation to check the same invariant twice: two checks at 500,000 items/s cost a full
core, and that core is invisible in a per-request latency number.

**Cost of storage.** Storage cost is bytes × copies × price × time. The illustrative stream from
section 5 is 518.4 GB/day; at 1% sampling it is 5.184 GB/day, a ninety-nine-fold reduction that still
identifies a failing request when the identifier is retained. Retaining copies is the multiplier that
`CI-030` catalogues: 12 TB live with 8 copies is 96 TB, and each removed copy is worth 12,000 GB of
the bill.

**Cost of snapshot and restore.** A 14 GiB snapshot at 200 MiB/s restores in 14,336/200 = 71.68 s.
To restore inside a four-hour recovery objective, the minimum sustained rate is 14,336 MiB / 14,400 s
= 0.996 MiB/s, so the 200 MiB/s path has two hundred times the required throughput and the restore is
not the constraint. The constraint is usually the time to obtain a consistent snapshot, which grows
with write rate rather than with size.

**Cost of an index or a sketch.** An in-memory index over 12,000,000 keys at 8 B per key is 96 MB,
which fits one node and not a fleet's per-node budget if replicated. A 4,096-register sketch is
O(1) memory with 1.625% relative error; a 65,536-register sketch reduces the error to 0.406% at
sixteen times the memory. That trade is the correct place to spend, because the alternative — exact
distinct counts — costs O(K) memory where K is the cardinality.

**Cost of carrying a wrong fix.** A mitigation left in place spends one of the four containment
resources indefinitely. A cache serving stale values changes the correctness contract of every
downstream consumer; a retry budget that hides a saturated dependency converts an outage into a
permanent latency tax. Quantify the carry cost in the correction ticket: the resource spent, the rate
at which it is spent, and the condition under which it must be removed.

**Cost of capacity versus cost of queueing.** From Result 4.2a, going from 90 to 120 workers raises
compute cost by 33% and halves wait time. Going from 120 to 150 raises cost another 25% and improves
wait only from 40 ms to 33 ms. The marginal return falls sharply, which is the economic statement of
the utilization curve and the reason headroom is set at a stated fraction rather than "as much as
fits the budget".

**Scaling limits.** Amdahl's law bounds the speedup from adding workers when a serial fraction
exists: with 5% serial work, 16 workers give 1/(0.05 + 0.95/16) = 9.14×, 64 workers give 15.42×, and
the limit is 1/0.05 = 20×. A universal-scalability model with contention σ = 0.1 and coherency
κ = 0.001 gives 16/(1 + 0.1×15 + 0.001×16×15) = 16/2.74 = 5.84× at N = 16, and it has a maximum
before it declines. Both bounds are the reason "add replicas" fails as a first action when the
measured symptom is flat throughput (`CI-007`, `CI-027`).

```text
FORMULA: speedup(N) = 1 / (s + (1 − s)/N);   limit = 1/s
WHERE:   s = serial fraction [dimensionless, 0 <= s < 1]; N = workers [count]
CONSTRAINT: fixed problem size, no coordination cost; the serial fraction does not shrink with N.
NUMERICAL EXAMPLE: s = 0.05, N = 16 -> 1/(0.05 + 0.059375) = 1/0.109375 = 9.14
                  N = 64 -> 1/(0.05 + 0.014844) = 1/0.064844 = 15.42
INTERPRETATION: Quadrupling workers from 16 to 64 raises throughput 1.69×, not 4×; the serial
                  fraction, not the worker count, sets the ceiling.
```

Takeaway: bound the diagnostic, price the correction, and record what a leftover mitigation spends
each day it survives.

## 10. Key Theorems / Results Reference

**Little's law (L = λW).** An identity for any stable system, not an approximation. It requires a
measurement window in steady state and a population that does not grow inside the window. It licenses
pool sizing and forbids inferring capacity in a closed loop (`4.4a`).

**Conservation of work in a queue.** In a single-server queue with constant rates and no loss, the
backlog rate is λ − μ and the integrated backlog is the area between the arrival and departure
curves. It licenses the drain-time calculation and fails once service degrades with depth, because
then μ is itself a function of B.

**The utilization–wait relationship.** For a single queue with low service-time variability, mean
wait grows as ρ/(1 − ρ). It licenses the statement that 80% utilization costs roughly double the
service time in wait, and it fails under heavy tails (`4.4b`).

**Quorum intersection.** Read and write quorums intersect with certainty exactly when R + W > N. With
R + W = N the disjoint probability is C(N−W,R)/C(N,R), which is positive for every finite N. It
licenses strong-read claims only under the strict inequality.

**Availability composition.** Series availability is a product and parallel availability is one minus
the product of complements, both conditional on independence. It licenses an availability budget per
stage and fails under common-mode failure (`4.4d`).

**The CAP-family reading.** Under a network partition a system must choose between refusing some
requests and returning some stale answers; the theorem says nothing about the partition-free case. The
practical consequence is that the choice belongs to the caller's contract, which is why "the database
is highly available" is not a statement about read-after-write.

**Idempotency as an application invariant.** No transport retry policy grants one-effect-per-request.
The invariant holds when a stable key is persisted before the effect and the deduplication window
exceeds the retry horizon. It licenses a claim about duplicates and nothing about ordering.

**Amdahl's law and the universal-scalability model.** Both bound the benefit of added parallelism; the
first from a serial fraction, the second from contention and coherency. They license a prediction of
diminishing returns and forbid extrapolating a small-N speedup to a large N.

**Quantile non-additivity.** A quantile of a union is not any function of the constituents' quantiles
alone. It licenses the rule that quantiles must be merged from distributions or sketches, and it
forbids averaging `CI-028`'s shard summaries.

**Counter arithmetic.** An n-bit unsigned counter wraps at 2ⁿ − 1 and an n-bit signed counter at
2ⁿ⁻¹ − 1. At 10,000 increments/s a 32-bit unsigned counter wraps in 429,496.7 s = 119.3 h = 4.97
days; a 32-bit signed millisecond timestamp wraps in 2,147,483.647 s = 24.86 days. Both license an
explicit wrap handler and forbid deriving rates by differencing cumulative values without one.

Takeaway: every result above is a conditional statement; the condition is part of the result, and the
catalog's `CONCEPT_CONFUSION` entries are all cases of dropping it.

## 11. Interview / Exam Questions

**Q1.** A pool of 90 workers serves 2,000 requests/s with a 40 ms service time. What utilization is it
running at, and what mean wait does that imply?
**Answer:** Capacity is 90/0.040 = 2,250/s, so ρ = 2,000/2,250 = 0.889. With low service-time
variability, W_q ≈ ρS/(2(1−ρ)) = 0.889 × 0.040/(2 × 0.111) = 0.03556/0.2222 = 0.160 s, so mean
response is about 200 ms. The number to say out loud is that the pool holds 11% spare capacity and
each request spends four times its service time waiting.
**Diagnostic follow-up:** if the measured mean response is 130 ms rather than 200 ms, which
assumption failed? The most likely one is service-time variability: the approximation is conservative
for near-deterministic service and badly optimistic for heavy tails, so a lower measured mean points
to a distribution narrower than the model assumed, and the p99 is the next thing to inspect.
**Scoring:** full credit for computing capacity before utilization and for naming the variability
assumption; partial credit for the ratio alone.

**Q2.** Sixty thousand items are queued when capacity is restored to 2,400/s against arrivals of
2,000/s. How long until the queue empties, and what changes if arrivals fall to 1,900/s during
recovery?
**Answer:** The surplus is 400/s, so 60,000/400 = 150 s. If arrivals fall to 1,900/s the surplus
becomes 500/s and the drain takes 60,000/500 = 120 s. The deficit determined the peak; the surplus
determines the tail, which is why an operator who reduces intake during recovery shortens the visible
incident without adding any capacity.
**Trap:** answering 60,000/2,400 = 25 s treats the service rate as the drain rate and ignores
arrivals still arriving during the drain; it understates the tail by a factor of six.

**Q3.** A request-rate panel shows a clean 0.000/s through a forty-minute window while customers
report timeouts. What are the competing explanations and what single check separates them?
**Answer:** Three explanations: the exporter stopped publishing, traffic genuinely ceased because
callers gave up before connecting, or the series was dropped by the metrics store. The separating
check is a companion series that must be nonzero while the process lives — scrape successes, process
uptime, or a heartbeat counter. Zero rate with zero scrapes is an exporter failure; zero rate with
scrapes succeeding and a collapse in connection attempts is genuine cessation; a missing series with
scrapes still recorded is a storage or cardinality drop.
**Why it matters:** reporting 0.000/s asserts a measurement that was never taken; the empty
denominator is undefined and must render as undefined with an age, which is `CI-002`.

**Q4.** With five replicas, a read quorum of two and a write quorum of three, what is the probability
that a read set is disjoint from the most recent write set?
**Answer:** R + W = 5 = N, so the sets can be disjoint. The probability is C(N−W,R)/C(N,R) =
C(2,2)/C(5,2) = 1/10 = 0.10. One read in ten can miss the newest write. Setting W = 4 makes R + W = 6
> 5 and the probability zero, at the cost of requiring four of five replicas to acknowledge each
write.
**Boundary:** the same formula at N = 4, R = 2, W = 2 gives 1/6 = 0.167, so the disjoint fraction
shrinks as N grows even while R + W = N stays fixed. "Eventual" is not a constant quality.

**Q5.** Cluster-wide p99 is 185 ms and the objective is 200 ms, so nobody is on call. What would you
check before accepting that number?
**Answer:** Whether it is a merged quantile or an average of shard quantiles. Twelve shards where
eleven report 120 ms and one reports 900 ms average to (11 × 120 + 900)/12 = 2,220/12 = 185 ms, and
that 185 is not the p99 of the merged population. The slow shard is 4.86× the reported figure. The
check is to recompute the quantile from merged observations or a mergeable sketch, and to report the
worst shard alongside the cluster number.
**Selection:** if the merged p99 is also 185 ms, the average was harmless in this instance; the
defect is still present because the next distribution will not be harmless.

**Q6.** A service logs 1.2 KB per event at 5,000 events/s with thirty-day retention. What is the
monthly stored volume, and what does 1% sampling with the request identifier retained change?
**Answer:** 5,000 × 1,200 B = 6,000,000 B/s = 6.0 MB/s. Per day: 6,000,000 × 86,400 =
518,400,000,000 B = 518.4 GB. Over thirty days: 15,552 GB = 14.14 TiB in binary units. At 1% sampling
the volume is 5.184 GB/day and 155.5 GB over thirty days, a hundredfold reduction. Sampling is
acceptable for the search use case because the request identifier is retained, so a specific failure
is still findable; it is not acceptable for an audit requirement that needs every event.
**Trap:** 15,552 GB is 15,187.5 GiB, and only that GiB figure equals 14.14 TiB. Relabelling the
decimal 15,552 GB as 15.5 TiB overstates the stored volume by about 9.9%, because the GB-to-GiB
factor is 1.048576 and the error compounds once a tera prefix is applied — small in this scenario,
decisive in a capacity plan with thin headroom.

**Q7.** A payment appears twice on one customer's statement after a timeout. What two mechanisms
produce this, and what replay distinguishes them?
**Answer:** Either the client generated a fresh idempotency key per attempt, or the keys were stable
but the deduplication window was shorter than the retry horizon. Replay the same key twice: one final
effect means the key path is sound and the window is the suspect, so compare the dedup record's
expiry with the observed retry interval — a window of 90 s against retries at 120 s explains the
duplicate. Replay with two different keys: two effects confirms that the key is not derived from the
logical request.
**Correction boundary:** persisting dedup before the effect is the invariant; persisting it after the
effect leaves a crash window in which the effect exists with no record of it.

**Q8.** A 32-bit unsigned counter increments at 10,000/s. When does it wrap, and what does an
operator see?
**Answer:** 2³² − 1 = 4,294,967,295 increments; 4,294,967,295/10,000 = 429,496.7 s = 119.3 h = 4.97
days. What the operator sees depends on the arithmetic reading the counter: a signed difference of
two cumulative values across the wrap is a large negative number, while an unsigned modular
difference is a large positive spike that inflates the rate for one interval. Either way the total
restarts from near zero with no deployment and no error. A 32-bit signed millisecond timestamp wraps
in 2,147,483,647/1,000 = 2,147,483.6 s = 24.86 days, which produces a date jump rather than a rate
anomaly and is far more confusing.
**Correction:** widen to 64-bit and compute rates from deltas with an explicit wrap handler; widening
alone does not fix a pipeline that differences cumulative values without one.

**Q9.** An engineer proposes that any score above 0.82 should release the hold on an account
automatically. What evidence would you require before agreeing?
**Answer:** A policy object with an owner, a scope, and a version — not a threshold. Concretely: the
named approver for the action class, the population the threshold applies to, the measured rate of
false releases at that threshold, and the reversal path with its cost. Without those, the score is a
sorting device and the release has no authorizing actor. `CI-004` is the catalogued failure: the
record shows a number, no actor, and no policy version, so a later review cannot establish authority.
**Judgment:** an automatic band is acceptable if the band, the population, and the reversal path are
each written down and reviewed; widening the band later is a policy change and needs its own review.

**Q10.** With a per-leaf p99 of 1%, how many parallel leaves can a request tolerate while keeping the
probability of at least one slow leaf under 10%?
**Answer:** Solve 1 − 0.99^m < 0.10. At m = 10 the probability is 1 − 0.99^10 = 1 − 0.9044 = 0.0956,
just under 10%. At m = 11 it is 1 − 0.99^11 = 0.1047, over the limit. So ten leaves is the maximum.
At m = 50 the figure reaches 39.50%, and at m = 100 it reaches 63.40%.
**Implication:** the budget is consumed in the first order of magnitude, so batching fifty small
calls into one is worth more than shaving a millisecond off each leaf; and any retry that adds a
second parallel attempt is an addition to m, not a replacement.

**Q11.** A cache configured with `ttl=86400` has a hit rate near zero. How do you confirm the unit
defect rather than a key-namespace problem?
**Answer:** Read one entry twice and measure its age at eviction. If the entry disappears after about
86 s, the client is reading milliseconds and the configured 86,400 means 86.4 s; the intended
millisecond value is 86,400,000 and the ratio between interpretations is 1,000. If the eviction age
is near 24 hours but the hit rate is still zero, the key namespace is wrong: the write path populates
one prefix and the read path looks in another, which `CI-026` lists as the competing cause.
**Prevention:** assert the configured value falls in an expected range at startup, and name the unit
in the configuration key so the defect cannot be written down ambiguously.

**Q12.** A holder of a seven-second lease is paused for five seconds by a garbage-collection stop.
Is it safe, and what measurement would change the answer?
**Answer:** The margin is 7 − 5 = 2 s, which is safe only if five seconds bounds every pause. If the
pause distribution's p99.9 is eleven seconds, a lease of seven seconds allows two holders for four
seconds, and both will write. The measurement that changes the answer is the maximum pause observed
in production, not the median; the lease must exceed it with margin, and writes must be fenced so a
resumed holder's write is rejected by the store rather than by its own clock check.
**Why fencing is not optional:** extending the lease reduces the probability and never removes it;
only a token validated at the store turns the overlap into a rejected write.

**Q13.** Five serial stages each report 99.95% availability, all in one region behind one control
plane that is unavailable 0.1% of the year. What availability can the chain actually deliver?
**Answer:** The product predicts 0.9995^5 = 0.997502, or 21.88 h/yr of downtime. The control plane
bounds availability at 0.999, or 8.76 h/yr, whenever it is a prerequisite for all five stages, so the
achievable figure is the worse of the two and the stage product is a fiction. If the control plane
were required only at start-up, its event would not remove steady-state traffic and the product would
apply after the initial hour.
**Judgment:** the correct answer names the shared dependency before multiplying; `4.4d` is the
counterexample that makes this concrete.

**Q14.** With a 5% serial fraction, what speedup do sixteen workers give, and what do sixty-four give?
**Answer:** 1/(0.05 + 0.95/16) = 1/(0.05 + 0.059375) = 1/0.109375 = 9.14×. At sixty-four workers,
1/(0.05 + 0.014844) = 1/0.064844 = 15.42×. Quadrupling the workers improves throughput 1.69×, and
the ceiling is 1/0.05 = 20× no matter how many workers are added.
**Implication:** an observation that "throughput rose 1.7× when we quadrupled capacity" is not
evidence of a linear system with a small constant; it is the signature of a serial section, and the
next action is to find it rather than to add more workers.

**Q15.** A rate limit of 500 connections is configured against 200 application threads each holding a
connection for 40 ms at 200 requests/s. What is the required concurrency and what does the mismatch
cost?
**Answer:** Required concurrency is rate × hold time = 200 × 0.040 = 8 connections. The configured pool
of 200 threads × 40 ms is 8,000 potential concurrent connections, sixteen times the 500 limit, so
threads contend for permits and acquisition wait appears while the database reports low CPU. The
required number is 8; a pool of, say, 12 leaves 50% headroom without multiplying database
concurrency.
**Discriminating detail:** high acquisition wait with low hold time identifies the pool as too small;
high hold time with normal acquisition identifies connections held across non-database waits, which
is a different correction.

## 12. Quick Reference Card

| Step | Rule | Number to recompute | Fails when |
|---|---|---|---|
| 1 | Classify the symptom into one of four categories before proposing a fix | category, not a value | The symptom fits two categories and both are asserted |
| 2 | State at least two competing mechanisms for the same symptom | count of hypotheses | Only one mechanism is named, so no check can discriminate |
| 3 | Run the check whose outcome differs between hypotheses | one comparison, not a dashboard glance | The check returns the same result under both |
| 4 | Recompute the governing arithmetic with units attached | L = λW; B = (λ−μ)t; T_d = B/surplus | A unit or worker factor is omitted |
| 5 | Contain with a bounded action and record what it spends | freshness, completeness, latency, or money | The mitigation has no expiry or no recorded artifact |
| 6 | Correct the mechanism, not the wording | precondition, version, width, budget, ordering | The change only relabels a field or a report |
| 7 | Verify by replaying the triggering record and its boundary twin | effect count per key, reason code | Only the happy path is replayed |
| 8 | Record the evidence label and the owner | `DERIVED`, `ILLUSTRATIVE`, or `OPEN` | A constructed number is reported as measured |
