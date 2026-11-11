## 5. Worked Numerical Example

The example is a single procurement connector that turns an approved purchase request into a posted
supplier invoice. Every parameter is ILLUSTRATIVE and chosen so the arithmetic is checkable by hand;
the structure is what transfers to your system, not the values.

**Setup.** One tool, `create_supplier_invoice`, effect class E2 (keyed write against a sink that honours
the key). Population: 500,000 invoice-creation operations per business day, concentrated in an
eight-hour window of 28,800 seconds. Within that window there is a one-hour close-out peak. The sink is
an ERP that is idempotent on our key for seven days and then forgets it. Manual review exists for
payments above a threshold and costs 45 EUR in blended reviewer time when a duplicate must be unwound.

**Step 1 — mean arrival rate.** Conservation of counts: 500,000 operations spread over 28,800 seconds.

```text
λ_mean = 500,000 / 28,800 = 17.361 operations/s
Check: 17.361 * 28,800 = 499,997 ≈ 500,000 (rounding of λ). Consistent.
```

**Step 2 — peak and off-peak rates.** The peak hour carries 45,000 operations; the remaining 455,000
spread over the other 27,000 seconds.

```text
λ_peak = 45,000 / 1,800 = 25.000 operations/s
λ_off  = 455,000 / 27,000 = 16.852 operations/s
Check: 25.000 * 1,800 = 45,000; 16.852 * 27,000 = 455,004; total 500,004 ≈ 500,000. Consistent.
```

**Step 3 — backlog during the peak.** The executor completes 22 operations/s in steady state, so the
peak runs a deficit and a backlog accumulates. This is T-03 applied to one interval.

```text
B = (λ_peak - μ) * t_peak = (25.000 - 22.000) * 1,800 = 3.000 * 1,800 = 5,400 operations
Check: 25 * 1,800 = 45,000 arrived; 22 * 1,800 = 39,600 completed; difference 5,400. Consistent.
```

**Step 4 — drain during the off-peak.** A recovery lane runs at 28 operations/s while live arrivals
continue at 16.852 operations/s. Surplus is what drains the backlog.

```text
surplus = μ_r - λ_off = 28.000 - 16.852 = 11.148 operations/s
T_drain = 5,400 / 11.148 = 484.4 s ≈ 8.1 minutes
Check: in 484.4 s the lane completes 484.4 * 28 = 13,563 operations and live arrivals add
484.4 * 16.852 = 8,163, so the net reduction is 13,563 - 8,163 = 5,400. Consistent.
```

**Step 5 — the boundary of that drain plan.** The same peak lasting 45 minutes instead of 30 minutes
gives a different answer, and the off-peak surplus then has a fixed budget.

```text
B(45 min) = 3.000 * 2,700 = 8,100 operations
T_drain = 8,100 / 11.148 = 726.6 s ≈ 12.1 minutes
Remaining off-peak time = 27,000 - 2,700 = 24,300 s; drain consumes 726.6 s, leaving 23,573 s of
uninterrupted off-peak operation. Safe.
But if the peak rate were 28.000 operations/s (equal to μ_r), surplus during the peak is zero and
B = (28.000 - 22.000) * 1,800 = 10,800 with no lane fast enough to catch up during the same hour.
```

The second number is the one to remember: a recovery lane slower than the peak arrival rate cannot
recover *during* the peak no matter how long it runs, because the backlog is not merely conserved, it is
growing. Capacity decisions made against the daily mean hide this entirely.

**Step 6 — expected duplicates.** Retried fraction q = 0.004 (4 in 1,000 operations experience at least
one timeout), ledger coverage c = 0.985, sink self-dedup p_sink = 0 for the first 60 seconds because the
sink's dedup is asynchronous. Applying T-02 for the full day:

```text
D = N * q * (1 - c) * (1 - p_sink) = 500,000 * 0.004 * 0.015 * 1
  = 2,000 * 0.015 = 30 duplicate effects/day
Check: 500,000 * 0.004 = 2,000 retried; 2,000 * 0.015 = 30. Consistent both ways.
```

**Step 7 — price the fix.** Improving coverage from 0.985 to 0.9999 reduces the uncovered fraction from
0.015 to 0.0001, a factor of 150.

```text
D(0.9999) = 500,000 * 0.004 * 0.0001 = 0.2 duplicates/day
Annual cost at 45 EUR per rework and 250 working days:
  D(0.985)  -> 30.0 * 45 * 250 = 337,500 EUR/year
  D(0.9999) ->  0.2 * 45 * 250 =   2,250 EUR/year
  difference = 337,500 - 2,250 = 335,250 EUR/year
```

That saving is an upper bound, because it assumes every duplicate is detected and every detection costs
exactly 45 EUR. In practice an undetected duplicate invoice costs far more per event (a payment run, an
apology, a credit note) but occurs at an unknown rate. The bound is still useful: it shows the coverage
work is worth more than the engineering it costs at almost any plausible rate.

**Step 8 — authority staleness.** Approvals carry a 45-second execution token; authority-relevant
changes (budget revisions, delegation revocations, period locks) arrive at ν = 0.0025 per second, about
one every 6.7 minutes. Applying T-06:

```text
ν * g = 0.0025 * 45 = 0.1125
P(stale) = 1 - exp(-0.1125) = 1 - 0.8936 = 0.1064, about 10.6% of approved operations
```

Ten percent is not a defect rate you can wave through: it means one in ten dispatches uses an authority
decision that the world has already invalidated. The two corrections both cost less than the exposure:
re-check at the executor (g becomes the duration of the executor's own check, effectively milliseconds)
or shorten the token to 5 seconds, which gives ν*g = 0.0125 and P = 0.0124.

**Step 9 — dedup ledger sizing.** Steady 25 operations/s is the sizing rate for the seven-day window
(the peak hour dominates storage), at 768 bytes per record including index overhead.

```text
Records = 25 * 86,400 * 7 = 2,160,000 * 7 = 15,120,000
Bytes   = 15,120,000 * 768 = 11,612,160,000
S = 11.61 GB for seven days
Check: 25 * 86,400 = 2,160,000/day; * 7 = 15,120,000; * 768 = 1.1612e10 B. Consistent.
```

**Step 10 — in-flight and queueing.** Mean time in system W = 0.42 s (admission 45 ms, dispatch 90 ms,
sink 285 ms). Utilisation at the daily mean against the normal lane is 17.361/22.000 = 0.789.

```text
L = λ_mean * W = 17.361 * 0.42 = 7.29 in-flight operations
P(W_queue > 1 s) at ρ = 0.789, μ = 22: 0.789 * exp(-22 * 0.211 * 1)
  = 0.789 * exp(-4.642) = 0.789 * 0.00964 = 0.0076, about 0.8%
At ρ = 0.90: 0.90 * exp(-2.2) = 0.90 * 0.1108 = 0.0997, about 10.0%
```

The pair is the argument for a utilisation target. At the daily mean the tail is under 1%; push the same
system to 90% by adding traffic and the one-second tail multiplies by thirteen.

**Step 11 — retry amplification.** Per-attempt success s = 0.75, cap R = 4, with jitter.

```text
E[attempts] = (1 - 0.75^4) / (1 - 0.75) = (1 - 0.3164) / 0.25 = 0.6836 / 0.25 = 2.734
Attempt load = 17.361 * 2.734 = 47.5 operations/s at the sink
Check: 0.75^4 = 0.316406; 1 - 0.316406 = 0.683594; / 0.25 = 2.734. Consistent.
```

The sink must be provisioned for 47.5 ops/s, not 17.4, whenever the first-attempt success rate is 0.75.
This is the number that turns "we have headroom" into an outage: the headroom was computed against
first attempts only.

**Step 12 — key space.** 2.5 million distinct keys per year, 64-bit space.

```text
P ≈ n^2 / (2M) = (2.5e6)^2 / (2 * 1.8447e19) = 6.25e12 / 3.689e19 = 1.694e-7, about 1 in 5.9 million
```

Acceptable, and 128-bit keys would remove the question. The point of computing it is that the number is
tiny *only because the key is random*; a tenant-local counter restarts at zero and collides immediately.

**Sensitivity.** Four parameters dominate the answers. (i) Doubling q from 0.004 to 0.008 doubles
duplicates to 60/day and doubles the rework cost. (ii) Raising c from 0.985 to 0.9999 reduces duplicates
by 150x, which is why coverage beats every other lever. (iii) Raising the peak from 25 to 28 ops/s makes
the backlog 10,800 and removes any same-hour recovery. (iv) Raising W from 0.42 s to 0.60 s at the same
arrival rate pushes L from 7.29 to 10.4 and ρ from 0.789 to a level where the pool saturates. Only
(ii) is a design decision under your control in a single sprint; the other three are properties of the
traffic and the sink you must measure.

## 6. Visual / Geometric Intuition

**Picture 1 — the conveyor with four gates.** A conveyor belt carries a labelled box. The label is the
model's proposal: a description of what it thinks should happen. Gate 1 checks the box's shape against a
template; a box that is too wide is rejected without judgement. Gate 2 checks whether this box is
allowed to exist at all today: is the period open, is the budget unspent, does the business key already
exist in a posted document. Gate 3 checks the bearer's badge against the destination zone: this principal
may move boxes into warehouse 7 but not warehouse 9. Gate 4 is the loading dock: a clerk stamps a
manifest with the box's identity and only then lets it leave, and the manifest is what the receiving
warehouse confirms.

The picture makes one thing obvious that prose obscures: the gates are physically ordered and each has
its own failure mode. A box rejected at Gate 3 does not need Gate 4 to be consulted; a box admitted at
Gate 4 has a manifest regardless of whether the radio to the receiving warehouse works.

**Picture 2 — the evidence ladder as a ladder, not a staircase.** Draw four rungs bottom to top:
proposed, admitted, dispatched, committed. The model lives on rung 1. Your admission service climbs to
rung 2 and stops. The network climbs to rung 3 and stops. Only the sink reaches rung 4. Every recovery
question is really the question "which rung am I standing on, and which rung did I assume I was on?" A
timeout looks like rung 1 from the client's memory and rung 4 from the sink's memory; that discrepancy
*is* the UNKNOWN state, and it exists because rungs are climbed by different parties.

**Picture 3 — the state graph is a partial order, not a cycle.** Model the lifecycle as directed edges
and forbid the ones that do not exist.

```text
PROPOSED ──schema ok──> VALIDATED ──policy ok──> ADMITTED ──dispatch──> SUBMITTED
    │                       │                        │                    │
    └─reject──> REJECTED    └─reject──> REJECTED     └─conflict──> CONFLICT
                                                                          │
                                                              ┌───────────┼───────────┐
                                                              v           v           v
                                                         COMMITTED    REJECTED     UNKNOWN
                                                                          ^           │
                                                                          │           │
                                                                    reconcile ──────┘
```

Three edges are forbidden and each one is a real defect if drawn. `UNKNOWN -> NOT_SENT` is forbidden
without new authoritative evidence, because the only thing that changed is your patience. `COMMITTED ->
PROPOSED` is forbidden, because a committed effect cannot become an intention again; a correction is a
new operation with its own key. `REJECTED -> SUBMITTED` is forbidden for permanent reasons, because
retrying a permanent denial is the mechanism that converts a policy error into a load spike.

**Picture 4 — blast radius as concentric circles.** Centre: one resource (one invoice, one record).
Ring 1: the document family (all invoices for one vendor). Ring 2: the tenant. Ring 3: the region. Ring
4: everything the service credential can reach. Schema breadth and credential scope decide which ring
one call can touch. A wildcard target field or a `*` in a resource path moves a single call outward by a
ring, which is why wildcard-shaped parameters deserve review even when the schema is perfectly typed.

**Picture 5 — the two clocks as a slanted plane.** Draw event time on the horizontal axis and processing
time on the vertical. A healthy system is a narrow diagonal band. A reconciliation gap is a widening of
the band; a period close is a vertical cliff in processing time where many events process at one instant.
An operation whose receipt records only processing time sits at the cliff and cannot be compared against
a period boundary. This is why the audit record needs both stamps: the diagram shows exactly where the
missing one hurts.

**Picture 6 — the digest as a shadow of the payload.** The canonical payload is a solid object; the
digest is its shadow. Two objects with the same shadow are the same for approval purposes. The failure of
digest coverage is a light source that casts no shadow for certain parts of the object: those parts are
invisible to the approval, not absent from the effect. When you draw it this way, "we include the amount"
immediately invites the question "what else casts no shadow", and the answer is usually currency, date,
or cost centre.

## 7. Comparison / Selection Table

**Effect classes and their mandatory controls.** Classify first, then choose controls. The class is a
property of the sink, not of your intentions.

| Class | Example | Reversible? | Key required | Receipt source | Human gate |
|---|---|---|---|---|---|
| E0 read | price lookup, record fetch | n/a (no effect) | no | response body | none |
| E1 natural upsert | write a config value, set a field | yes, last write wins | no, but record digest | read-back | none |
| E2 keyed write | create invoice, create ticket | usually compensate | yes, durable | sink receipt on key | above threshold |
| E3 irreversible | payment release, delete record, send email | no | yes, plus intent record | sink transaction id | always |
| E4 fanout | broadcast to N systems | partially | yes, plus per-item keys | per-item receipts | above N items |

The table's most consequential row is E2, because it is where most business tools actually live and where
teams wrongly apply E1 thinking ("we can just retry, it will overwrite"). Creating a second invoice does
not overwrite the first; it duplicates it. If the sink cannot hold a uniqueness constraint on your key,
the tool is E3 for practical purposes and needs an intent record plus reconciliation.

**Recovery strategies compared.** Four options are usually on the table after a lost response. They
differ in what evidence they require and what they cost.

| Strategy | Requires | Duplicate risk | Recovery latency | Use when | Refuse when |
|---|---|---|---|---|---|
| Blind retry, new key | nothing | unbounded, grows with outage length | seconds | E0 reads only | any write |
| Blind retry, same key | key persisted pre-dispatch | low, bounded by coverage | seconds | E2 with reliable ledger | key was not persisted |
| Reconcile then retry | queryable sink or outbox in sink transaction | near zero | seconds to hours | E3, or opaque sink | no authoritative query exists |
| Manual review queue | reviewer capacity, redacted context | zero by construction | hours to days | high value, low volume | volume exceeds reviewer capacity |

The last row has a hard arithmetic limit that teams routinely ignore. If an outage strands 3,000
operations and a reviewer can process 40 per hour with two reviewers, the queue takes 3,000 / 80 = 37.5
hours to clear, during which new operations keep arriving. A manual fallback is a recovery plan only when
its throughput exceeds the arrival rate of the thing it is recovering.

**Hold, shed, or replay.** Under overload, three levers exist and they are not interchangeable.

| Lever | Mechanism | Effect on correctness | Effect on user | Prefer when |
|---|---|---|---|---|
| Hold (queue) | bounded queue with backpressure | preserves every intent; risks stale intents | latency grows, eventually timeouts | overload is short and capacity returns |
| Shed | reject with retry-after and a typed reason | loses no committed work; drops intents | explicit, actionable failure | overload is sustained or the tool is low value |
| Replay | reconcile and re-drive from a durable record | duplicates only where coverage is incomplete | delayed completion | after recovery, from recorded intents only |

Hold is the instinct and the worst choice for long overloads, because a queue of stale intents is a queue
of operations whose authority has expired by the time they run. Shed is honest. Replay is only legitimate
from a durable intent record with an explicit outcome, never by scanning logs.

**Authorization placement.** Where the check happens determines what the check is worth.

| Placement | Latency added | Freshness | Weakness | Mitigation |
|---|---|---|---|---|
| Prompt time (model context) | none | none | attacker-influenced text, no principal | never count as authorization |
| Admission service | ~5–20 ms | good | TOCTOU gap to effect | short scope, recheck at executor |
| Executor, at effect time | ~1–5 ms (local policy) | best | needs fresh state or a policy cache | cache with version stamp |
| Downstream sink | 0 for you | authoritative | error surfaces as sink failure | map sink denials to typed reasons |

The second row is where most systems stop and where T-06 then bites. The fourth row is often forgotten:
the sink has its own authorization and will deny operations your admission service allowed. That denial
must be mapped back to a typed error, or callers will treat a permanent denial as a transient failure and
retry it.

**Schema strictness compared.** The schema is the first gate and its strictness is a trade with real
consequences on both sides.

| Choice | Rejects | Accepts | Failure mode | When it wins |
|---|---|---|---|---|
| Strict, no defaults, all fields required | malformed and under-specified calls | exactly one interpretation | brittle to optional business fields | E3 tools, regulated flows |
| Strict with typed optionals | malformed | several legal shapes | ambiguity about which shape ran | E2 with optional metadata |
| Permissive, extra fields allowed | almost nothing | future fields silently | forward-compatible but effect-opaque | E0 reads only |
| Open free-form object | nothing | anything | schema becomes documentation | never for a write |

The permissive row deserves a warning: allowing unknown fields means a new field added by a client
deployment can enter the canonical payload and change the digest without any reviewer seeing a schema
change. If the tool must tolerate extra fields, exclude them from the digest explicitly and record them
for inspection, so that "unknown field present" is a visible audit fact rather than an invisible one.

**Receipt queries compared.** How you learn the outcome determines how fast and how safely you recover.

| Source | Authority | Latency | Failure when unavailable | Cost |
|---|---|---|---|---|
| Your own outbox row | weak: written before or after the effect | ms | outage loses the trail | low |
| Outbox written in sink transaction | strong | ms to s | only if the sink's DB is down | medium |
| Sink read API on your key | strong | 10 ms to s | outages block all recovery | low |
| Sink business-key search | medium: may match unrelated records | s to minutes | fuzzy matches mislead | medium |
| Reviewer inspecting the sink UI | strong but slow | minutes | human bottleneck | high |

The first row is the trap. An outbox you write yourself before dispatch tells you what you *intended*
and nothing about the effect. It is a legitimate record of intent and an illegitimate receipt. Name it
accordingly in code, because a field called `status: COMMITTED` in a pre-dispatch table will eventually
be read as proof by someone under pressure at 03:00.
