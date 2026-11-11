## 5. Worked Numerical Example

The scenario is ILLUSTRATIVE: a support-assistant agent that issues refunds and labels on a B2C
order platform. Every input is a constructed value chosen to make the arithmetic legible; no
deployment is being reported.

**Setup.**

| Input | Value | Unit | Note |
|---|---|---|---|
| Distinct refund intents offered | 18 | intents/s | during a 120 s campaign burst |
| Executor completion rate | 15 | intents/s | one worker pool, sink-limited |
| Retry probability per attempt | 0.08 | dimensionless | transient sink timeouts |
| Duplicate suppression effectiveness | 0.94 | dimensionless | measured on the ledger path |
| Mean refund value | 42.50 | USD | 4250 minor units |
| Recovery lane capacity | 21 | intents/s | dedicated drain workers |
| Effect class | W4 | ladder rung | externally visible money movement |
| Key entropy | 64 | bits | UUIDv4-grade |

**Step 1 - attempts and duplicate pressure.**
`A = 1/(1-rho) = 1/0.92 = 1.086957`.
`attempts = 2160 x 1.086957 = 2347.83`, so 187.83 attempts are surplus.
Check by the direct form: `2160 x 0.08/0.92 = 2160 x 0.086957 = 187.83`. Agreement.
Undetected duplicates: `187.83 x (1 - 0.94) = 187.83 x 0.06 = 11.27`.
Money at risk: `11.27 x 42.50 = 478.97` USD.

**Step 2 - the retry feedback loop.**
The naive drain calculation uses the *intent* arrival rate: `B = (18-15) x 120 = 360`, and with a
recovery lane at 21 intents/s against 15 live intents/s the surplus is 6, hence `T_drain = 60` s.
But the executor sees attempts, not intents. The attempt arrival rate is `2347.83/120 = 19.565` per
second, so the backlog the executor actually holds is `(19.565 - 15) x 120 = 547.83`, a factor of
1.522 over the naive figure.
Worse, the recovery lane's surplus is now `21 - 19.565 = 1.435` intents/s, so
`T_drain = 547.83/1.435 = 381.8` s - 6.36 times the naive estimate.

```text
 naive drain     360 / (21 - 15)     =  60.0 s
 attempt-aware   547.83/(21-19.565)  = 381.8 s      ratio 6.36
 if rho rises to 0.12:  attempts 2454.55, rate 20.455/s, surplus 0.5455/s
                        backlog (20.455-15)*120 = 654.55, drain 654.55/0.5455 = 1200 s
```

INTERPRETATION.
Retries consume the same capacity as first attempts and can drive the drain surplus toward zero.
At `rho = 0.125` the attempt rate reaches `18/0.875 = 20.571` per second, the surplus falls to
0.429/s, and drain stretches to `668.57/0.429 = 1559` s; a further increase puts the system in
positive feedback where retries sustain the overload they are meant to cure.
The controlling decision is therefore a *retry budget* - cap attempts per intent and cap retry rate as
a fraction of first attempts - not a larger recovery pool.

**Step 3 - ledger sizing and retention.**
Burst-only accounting is misleading, so size against the steady mean of 4 intents/s for 90 days:
rows `= 4 x 86400 x 90 = 31,104,000`; bytes `= 31,104,000 x 512 = 1.5930e10`; size
`= 1.5930e10 / 2^30 = 14.83` GiB.
At an ILLUSTRATIVE 0.023 USD per GiB-month the ledger costs 0.34 USD/month.
Collision expectation at 64 bits: `n^2/2^65 = (3.1104e7)^2 / 3.6893e19 = 2.62e-5`.
At 48 bits the same ledger yields `9.674e14 / 5.629e14 = 1.72` expected collisions, which is why the
minimum is 64 bits.

**Step 4 - approval freshness.**
The refund tool is W4, so a human approves payloads above 4000 minor units.
Approvals are granted in a queue and executed later; the approved payload includes the destination
instrument, whose validity changes at a rate of about `gamma = 2e-4` per second (one change per 83
minutes).
With a 900 s queueing delay, `P(stale) = 1 - exp(-0.18) = 0.1647`; across 2160 intents that is
355.97 stale-approval opportunities per burst window, against 11.27 duplicate effects.
At a 60 s delay, `P = 1 - exp(-0.012) = 0.0119`, i.e. 25.7 opportunities.

INTERPRETATION.
Stale approval is the larger defect by an order of magnitude at realistic queue delays.
The remedy is digest re-check at dispatch plus a TTL near the dispatch p99 (measured, not assumed),
and escalation on mismatch rather than silent re-approval.
The counter-intuitive consequence: shortening the approval window matters more than shortening the
retry window.

**Step 5 - batch versus per-item decisions.**
A nightly label batch carries 20 order labels; the sink fails per item with `q = 0.05`.
All-or-nothing retry succeeds with `0.95^20 = 0.3585`, delivering `20 x 0.3585 = 7.17` labels per
attempt and wasting 12.83.
Per-item keys deliver `20 x 0.95 = 19` per attempt.
But if all 20 items sit on one shard that fails as a unit, per-item retries deliver 0 and the
computation above is invalid - the shard failure correlation is the discriminator.

**Step 6 - the decision.**
Hold refunds in `UNKNOWN` for at most 4 attempts of 3 s (12 s per intent) and then reconcile by
querying the sink with the operation key.
Reconciliation at 0.15 USD per query against 187.83 surplus attempts would cost 28.17 USD per burst
if every attempt were reconciled, versus 478.97 USD of exposure at `d = 0.94`; but reconciling only
terminal `UNKNOWN` states - which is `187.83 x 0.06 = 11.27` at this effectiveness - costs 1.69 USD.
The economics favour "suppress first, reconcile the residue", which is exactly what a correct ledger
plus a bounded retry budget produces.

**Independent re-check.**
`2160 x 1/0.92`: `2160/0.92 = 2347.826`; `2347.826 x 0.92 = 2159.999` rounded to 2160.
`547.826 - 360 = 187.826`, matching the surplus attempts - the identity holds because both quantities
derive from the same `N x rho/(1-rho)` term.
`360/6 = 60` and `547.826/1.435 = 381.76`, so the ratio is 6.363.
`31,104,000 x 512 = 1.5925e10`, and `1.5925e10/1.0737e9 = 14.83`.
Four independent checks agree.

**Sensitivity note.**
The three inputs that move the answer are `rho`, `d`, and the recovery surplus.
Halving `rho` to 0.04 cuts surplus attempts to `2160 x 0.041667 = 90` and exposure to
`90 x 0.06 x 42.50 = 229.50` USD (a 2.09x gain), while drain falls to
`450/2.25 = 200` s - a 1.91x improvement, not 2x, because backlog and surplus improve at different
rates: backlog scales with `rho/(1-rho)` while surplus is
`mu_r - lambda/(1-rho)`, which changes less near the operating point.
Dropping `d` from 0.94 to 0.85 raises exposure to `187.83 x 0.15 x 42.50 = 1197.4` USD while changing
nothing about latency, which is why duplicate detection and throughput must be tested separately.
Raising recovery capacity from 21 to 24 collapses drain to `547.83/4.435 = 123.5` s, a 3.09x gain for
14.3 percent more capacity: surplus is superlinearly valuable near saturation.

**Takeaway.**
The two numbers that decide the design are the surplus in the recovery lane and the product
`(1-d) x unit_value`; everything else in the example is arithmetic that follows from them.

## 6. Visual / Geometric Intuition

**The gate and the conveyor.**
Picture a warehouse gate. The model writes a shipping label and hands the box to an attendant. The
attendant checks four things before the box proceeds: does the label parse (shape), is this box
supposed to leave through this gate (authority), is the destination still valid today (meaning), and
has this exact box already gone through (claim).
Past the gate is a conveyor into a truck. The truck departing is the effect; the attendant's receipt
stub is the ledger row; the driver's radio is the response channel.
A broken radio does not un-ship the box, and this single picture disposes of most retry bugs.

**State graph.**
The effect lifecycle is a directed graph with one absorbing branch that is not terminal.

```text
                 +-------------+
   PROPOSED ---> | VALIDATED   | ---> REJECTED   (typed reason; safe to abandon)
                 +-------------+
                        |
                        v
                 +-------------+     ledger claim: insert-if-absent
                 | RESERVED    |     holds key + digest + deadline
                 +-------------+
                        |
            +-----------+------------+
            v                        v
     +-------------+          +-------------+
     | SUBMITTED   |          | EXPIRED     |  (never dispatched; safe to re-plan)
     +-------------+          +-------------+
            |
      +-----+------+----------------+
      v            v                v
 COMMITTED     REJECTED          UNKNOWN
      |            |                |
      |            |                +--> RECONCILING --> COMMITTED | NOT_FOUND
      |            |                                        |
      |            +--> (no compensation needed)            +--> re-dispatch with SAME key
      |
      +--> COMPENSATING --> COMPENSATED | COMPENSATION_FAILED --> escalate
```

Three edges are the ones people get wrong.
`SUBMITTED -> COMMITTED` is written by the executor after the sink acknowledges, never by the caller.
`UNKNOWN -> NOT_FOUND` is only legal after an authoritative sink query that supports the key; a
timeout is not evidence of absence.
`COMMITTED -> COMPENSATING` creates a *new* keyed operation; there is no edge that erases an effect.

**Geometry of the duplicate window.**
Draw the key's life as a horizontal bar from mint to ledger expiry, and each attempt as a tick on it.

```text
key rf_8841_a  |=================================================|  t
attempt 1         ^ committed, response lost
attempt 2                ^ same key -> ledger returns receipt (correct)
attempt 3 (bug)          *new key rf_8841_b* -> second effect (defect)
expiry                                          !  retention must exceed max inflight + clock skew
```

The defect is visible as a second bar starting while the first is still open.
Retention shorter than the maximum in-flight window leaves the same gap at the right edge: a retry
arriving after expiry mints nothing and hits an empty ledger.

**Backlog as a trapezoid.**
Plot queue depth against time: a rising line of slope `lambda - mu` for the overload window, then a
falling line of slope `-(mu_r - lambda_r)`.

```text
 depth
  547 |            /\
      |           /  \
      |          /    \
  360 |         /      \        naive peak
      |        /        \
      |       /          \
     0+------/------------\----------- time
         0      120       502 s
```

The naive model draws the peak at 360 and the descent reaching zero at 180 s.
The attempt-aware model draws the peak at 547.8 and zero at 501.8 s.
The area under both curves equals total queued work; the difference in *shape* is entirely the retry
inflow, which is why an operator watching depth sees a plateau rather than the expected decline.

**Fanout as a tree.**
One intent expands into `f` records, each of which may have its own children (a refund creating a
ledger entry, a notification, and a reconciliation task).
Blast radius is the number of leaves touched; `f` is the branching factor and the number of levels is
usually 1-2.
The visual point: bounding the branching factor at the schema is the only place where the whole tree
can be bounded, because each child's tool call is made by the executor, not the model.

**Approval as a signed envelope.**
An approval wraps a canonical payload in a signature; the payload is the contents, the digest is the
seal, and the TTL is the expiry stamp.
If the seal covers only the amount, then changing the destination leaves the seal intact - the
envelope is intact and the contents are different, which is precisely the substitution attack.
Digest coverage is therefore the thickness of the seal, and `c^k` is the fraction of substitutions it
resists.

**Where intuition misleads.**
Latency averaging hides the retry tail: a mean of 1.09 attempts implies almost nothing about p99,
because the retry tail is bounded by the attempt cap (`4 x 3 s = 12 s` here) rather than by the mean.
Operators who size timeouts from the mean attempt count build systems that time out on the exact
requests that needed retrying.

**Takeaway.**
The state graph's `UNKNOWN` node and the duplicate-window bar are the two drawings to keep; every
failure mode in §8 is a mis-drawn edge on one of them.

## 7. Comparison / Selection Table

**Choosing the effect-control pattern.**
The column that decides most cases is "what does the sink support?", because client-side cleverness
cannot substitute for sink-side deduplication.

| Pattern | Use when | Benefit | Avoid when | Fallback |
|---|---|---|---|---|
| Read-through cache | effect class R0, idempotent by nature | cheapest, no ledger | fresh state is required for a decision | scoped fresh read |
| Keyed write with replay | sink accepts an idempotency key | true duplicate suppression and receipt replay | sink has no key support | hold queue + reconcile |
| Write + reconciliation | sink supports query-by-key but not dedup | tolerates lost responses | sink has no query by key | manual review queue |
| Saga with compensation | multi-step effect across systems | explicit recovery per step | compensation is user-visible or fee-bearing | reserve-then-commit |
| Reserve/commit two-phase | inventory, quota, credit | no oversell under concurrency | reservation expiry is unmanaged | conditional write at sink |
| Deterministic workflow | route and actions are known statically | auditable, replayable, no model in the loop | discovery or adaptation is required | bounded planner loop |
| Human approval gate | effect class W4/W5 or novel payload | bounds worst-case blast radius | volume exceeds reviewer capacity | sampled review + limits |

**Choosing the concurrency control.**
This table is about the `meaning` conjunct: how the executor prevents two admissible calls from
jointly breaking an invariant.

| Control | Guarantees | Cost | Fails when |
|---|---|---|---|
| Read-check-write in app | nothing under concurrency | low | two workers read the same snapshot |
| Optimistic version check | serialisable per row | one retry per conflict | conflict rate high -> livelock |
| Conditional write at sink | invariant holds at the data store | needs sink support | sink API lacks conditions |
| Pessimistic lock | strict serialisation | latency and deadlock risk | lock held across a network call |
| Reservation with TTL | bounded oversell with expiry | expiry sweeper required | sweeper lags or crashes |
| Serialised single writer | simplicity | throughput ceiling | volume exceeds one worker |

The worked example in §5 fails the first row: two concurrent 700-unit debits against a 1000-unit
balance both pass an application-level read-check-write and both succeed, producing a negative
balance. Only the third or fifth row prevents that.

**Choosing key scope.**
| Key scope | Duplicate suppression | Correct for | Wrong for |
|---|---|---|---|
| Per transport attempt | none | never | anything with an effect |
| Per tool call as emitted by the model | partial | single-turn, single-effect tools | retries, replans |
| Per intent (planner-owned) | complete for that intent | all writes | nothing |
| Per batch | all-or-nothing | atomic multi-line journal entries | independent items |
| Per (server, tool, key) | complete in federation | multi-server deployments | single-server (harmless overhead) |

**Choosing retention.**
| Retention | Adequate when | Risk |
|---|---|---|
| 1 hour | retries are in-process only | replays from queues and DLQs hit an empty ledger |
| 24 hours | synchronous retry plus same-day replay | overnight batch replays duplicate |
| 7 days | includes weekend incident response | month-end reconciliation duplicating |
| 90 days | covers the longest business reversal cycle | storage cost (0.34 USD/month here) |
| Permanent (archived) | regulated effects | audit surface grows without a policy |

The arithmetic in §5 says 90 days costs 0.34 USD/month against 478.97 USD per uncorrected burst, so
the decision is not close; the only real question is whether the archive is queryable when needed.

**Choosing the failure response.**
| Observed | Response | Never |
|---|---|---|
| Schema violation | typed client error, no ledger write | coerce or drop fields |
| Policy denial | typed error with the rule identifier | escalate to a broader role automatically |
| Ledger conflict (same key, different digest) | reject as conflict, require new approval | overwrite the existing row |
| Ledger hit (same key, same digest) | return stored receipt | dispatch again |
| Timeout, no response | mark `UNKNOWN`, schedule reconcile | mark failed and retry with a new key |
| Sink 4xx permanent | mark rejected, no retry | retry until the budget exhausts |
| Sink 5xx repeated | bounded retry with same key, then `UNKNOWN` | unbounded retry, backoff without jitter |

**Takeaway.**
Select the smallest control whose guarantee matches the effect class, and let the sink's capabilities
- not the client's preferences - decide whether deduplication or reconciliation is the primary
mechanism.
