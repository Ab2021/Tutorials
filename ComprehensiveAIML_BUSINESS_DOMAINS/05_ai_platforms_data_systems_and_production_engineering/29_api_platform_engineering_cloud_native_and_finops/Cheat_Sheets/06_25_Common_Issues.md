---
title: "25 Common Issues — API Platforms, Cloud-Native Systems and FinOps"
topic: "Diagnostic catalogue: API contracts, admission and quota, retries, caching, Kubernetes rollout, tenancy, SLOs and unit economics"
collection: "api-platform-engineering-cloud-native-and-finops"
level: "advanced"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
---

# 25 Common Issues — API Platforms, Cloud-Native Systems and FinOps

## 1. File Header

**Purpose.** This is the unit's final troubleshooting artifact. It converts the recurring failure shapes of an
API platform — contract drift, admission overspend, retry amplification, cache staleness, rollout churn,
tenancy leakage, misleading SLOs and unattributable cost — into thirty stable records, each with a
discriminating check that separates two plausible causes *before* anyone changes a setting.

**Why the file is called 25 and carries 30.** The `25` is the series slot every unit uses for its final
diagnostic artifact. The catalogue holds thirty records because the four record families that this module
owns (contract, capacity, platform, cost) each needed more than six entries to cover the failures that a
practitioner can actually settle today. Nothing here is a renamed duplicate of another entry; the
cross-reference in Section 8 says which records share a mechanism and must be distinguished.

**Prerequisites.** You should be able to read a Kubernetes manifest, an HTTP gateway route, a token-bucket
configuration, an autoscaling policy and a cloud bill export. You should also be comfortable with
utilisation, queueing delay, percentiles, geometric series and per-unit cost normalisation, because every
quantitative example here is arithmetic on one of those.

**Key outcome.** Given a symptom — "the invoice doubled but traffic did not", "the retry made a second
charge", "the canary is green and the users are not", "team B's chargeback is 40% above their own estimate"
— you can name the two or three causes with the highest base rate, run one check that separates them, and
choose a bounded correction with a stated cost.

**Exclusions.** This file does not restate the derivations of the topic sheets in this collection; it
references them by formula number. It does not cover model quality, prompt design, or the legal
determination of a data-residency obligation. It does not assert any vendor's current pricing, quota
behaviour or API semantics: those are OPEN facts that must be re-read at edit time.

**Level.** Advanced. Every record assumes you can read a manifest, a configuration diff, and a bill.

**Version and review.** 1.0, created 2026-11-10, not independently reviewed. Each record carries its own
validity condition, so a record may stay valid while a neighbouring one is retired.

**Evidence.** All arithmetic is DERIVED on a constructed platform described below (`ILLUSTRATIVE` wherever a
number depends on that construction). No incident, customer result, provider benchmark or production
measurement is claimed. Where a claim would depend on a vendor's behaviour, the record says so and treats it
as OPEN rather than asserting it.

**The reference platform.** To keep numbers comparable across thirty records, every quantitative example
uses one fictional system. It serves a request/response API in front of a hosted model. Eight inference
replicas each hold one accelerator, each sustaining six requests per second inside the latency target, so
provisioned capacity is 48 req/s. Mean demand is 12 req/s and the daily peak is 40 req/s. A month is 730
hours. One accelerator costs $3.20 per hour. Four gateway instances cost $0.052/hour each; two admission
instances cost $0.078/hour each. A month is 31,536,000 requests at mean demand and 220 output tokens per
request. Those five numbers — 8, 6, 12, 40, 730 — appear in most records below, and a record that changes
one of them says so.

**How to use the catalogue.** Section 8 holds the table. Scan the symptom column until a row matches, then
read the expansion below the table, which gives the mechanism and the arithmetic that localises the cause.
Section 11 rehearses the same material as questions, which is the fastest way to find out whether you have
absorbed it rather than merely recognised it.

**Retrieval cue.** Formulas are Section 4. The end-to-end bill trace is Section 5. The records are Section 8.
The cost model for a proposed fix is Section 9. Rehearsal is Section 11. The card is Section 12.

**Takeaway.** A catalogue earns its length only if every entry can be *settled* by a check somebody can run
today, so each record names its check before it names its fix.

## 2. Concept Map / Mental Model

The organising idea is that a platform failure always has an **owner of state**. Numbers that disagree mean
two owners; behaviour that disagrees with numbers means the owner changed; and a symptom that cannot be
reproduced means the owning state is not in the reproduction.

```text
                        symptom
                           |
        +------------------+-------------------+
        |                  |                   |
   a count is wrong   a rate is wrong    behaviour is wrong
        |                  |             but counts agree
        v                  v                   v
   accounting      capacity or          contract, auth or
   defect          concurrency          data boundary
        |                  |                   |
        +------------------+-------------------+
                           |
              name the owner of the state
                           |
              build a bounded reproduction
                           |
        +------------------+-------------------+
        |                                      |
   separates the causes                does not separate
        |                                      |
   correction + cost                   the finding is the
        |                              instrumentation gap
   verification signal
        |
   prevention that survives the next change
```

**Three questions that route almost every symptom.** First: is the quantity you are comparing defined over
the same population and window on both sides? Most "the bill is wrong" and "the metric is wrong" reports die
here. Second: which component is allowed to mutate this value, and is it the only one? Most duplicate
charges, over-admitted quotas and stale permissions die here. Third: at what rate does the discrepancy
accumulate, and does that rate match the symptom?

**Why a symptom is not evidence.** A 429 response at the gateway is consistent with a correct quota, a
per-replica quota that is 8× too small, a retry storm, and a dependency outage. The four have entirely
different corrections, and choosing the wrong one converts a capacity problem into a correctness problem.
The catalogue's whole discipline is to make the cheapest observation that leaves one cause standing.

**The cost axis is not separate from the reliability axis.** Every reliability decision in this module is
priced. Adding two replicas to halve tail latency costs $4,672 per month on the reference platform, and the
second two replicas buy 25% of the tail improvement the first two bought. A platform team that cannot state
that price cannot defend either the addition or the refusal.

**Takeaway.** Route by owner of state, not by the loudest metric, and price the correction before proposing
it.

## 3. Foundational Definitions

| Term | Meaning in this file | Boundary that matters |
|---|---|---|
| tenant | The billing and authorisation subject that owns data | A header value is not a tenant boundary unless every read enforces it |
| admission credit | An integer unit of concurrency or rate that a request must hold to proceed | Credits without an expiry policy are leaks |
| global budget | The total credits the platform may have outstanding in one scope | Scope must be one region, one pool, one interval |
| lease | A time-bounded grant of credits to one actor | A lease is not ownership; expiry returns the credits |
| fence token | A monotonic number that makes a late message harmless | Without it, a late revoke deletes a younger grant |
| idempotency key | A client-supplied value that collapses repeats of one intent | Per-request keys make retries duplicates |
| service rate | Sustainable requests per second per replica inside the latency target | Measured, not configured |
| utilisation | Arrival rate divided by provisioned capacity | Average utilisation prices capacity wrongly |
| error budget | The failure fraction an SLO permits in its window | 0.1% of 30 days is 43.2 minutes |
| burn rate | Observed error rate divided by the budgeted error rate | Window length changes the alert's meaning |
| unit cost | Total cost divided by a denominator such as 1k requests | The denominator decides the story |
| amortised commitment | A prepaid or committed discount spread across its term | Differs from the invoice line by design |
| tag coverage | Fraction of billable resources carrying an owner label | Coverage bounds attribution accuracy |
| series cardinality | Distinct label combinations a metric produces | Cost is multiplicative in label domains |
| readiness | The signal that a replica may receive traffic | Distinct from liveness, which restarts |
| drain | The ordered removal of a replica from the traffic set | Requests in flight at drain start still matter |
| cache epoch | A version that invalidates cached authorisation decisions | TTL alone is a time bound, not a truth bound |
| retry budget | The permitted retry-to-original ratio per caller | Distinct from the per-attempt timeout |
| at-least-once | Delivery guarantee that permits duplicates | Requires an idempotent sink to be safe |
| effectiveness | Successful tasks divided by attempted tasks | A ratio, not a latency, and not a cost |

**Takeaway.** The three definitions that cause the most arguments are "utilisation" (which capacity?),
"unit cost" (which denominator?) and "tenant" (enforced where?). Say which one you mean before comparing.

## 4. Core Formulas with Full Derivation

### 4.1 Base case

**Unit compute cost, and why the replica count cancels.**

```text
FORMULA [C1]: u = p_h / (3600 * s * rho)
WHERE: u is accelerator cost per request in currency; p_h is price per accelerator-hour (3.20);
       s is sustainable requests per second per replica (6); 3600 converts hours to seconds;
       rho is utilisation of provisioned capacity, dimensionless in (0, 1).
CONSTRAINT: rho must be measured against the same capacity that the numerator pays for.
NUMERICAL EXAMPLE: rho = 12/48 = 0.25 at mean demand, so u = 3.20 / (3600 * 6 * 0.25)
       = 3.20 / 5400 = $0.000593 per request, which is $0.5926 per 1,000 requests.
       Checked against the bill: 8 * 3.20 * 730 = $18,688 per month over 31,536,000 requests
       gives 18,688 / 31,536 = $0.5926 per 1,000. The two agree exactly.
INTERPRETATION: the number of replicas cancels because both cost and capacity scale with it, so
       unit cost is governed by utilisation and service rate alone. Every dollar of idle capacity
       appears as a higher u even though nobody changed a price.
```

**Utilisation has two values, and mixing them costs a factor of 3.33.**

```text
FORMULA [C2]: rho_peak = lambda_peak / (R * s);  rho_avg = lambda_avg / (R * s)
WHERE: lambda_peak = 40 req/s, lambda_avg = 12 req/s, R = 8 replicas, s = 6 req/s.
CONSTRAINT: One volume must pair with one capacity. A peak utilisation paired with average volume
       is meaningless, and so is the reverse.
NUMERICAL EXAMPLE: rho_avg = 12/48 = 0.2500; rho_peak = 40/48 = 0.8333. Applying C1 at the
       peak gives 3.20/(3600*6*0.8333) = $0.000178 per request; at the average it gives
       $0.000593. The ratio is 0.8333/0.2500 = 3.333.
INTERPRETATION: neither number is "the" cost. The average-based figure is the realised cost of the
       fleet; the peak-based figure is the marginal cost of one more request served at the peak.
       Quoting the peak-based figure as the product's unit cost understates spend by 3.33x.
```

**Tail latency under provisioning, from the queue rather than the mean.**

```text
FORMULA [C3]: for M/M/1, P(Wq > t) = rho * exp(-(mu - lambda) * t),
              so t_99 = ln(rho / 0.01) / (mu - lambda)
WHERE: mu is capacity in req/s (R * s), lambda is arrival rate in req/s, t is waiting time in
       seconds before service starts; Wq excludes service time Ts = 1/s.
CONSTRAINT: Poisson arrivals, exponential service, one queue, one pool. Correlated arrivals at
       a burst make the real tail worse than this bound, never better.
NUMERICAL EXAMPLE: R = 8, mu = 48, lambda = 40, mu - lambda = 8, rho = 0.8333.
       t_99 = ln(83.333) / 8 = 4.42285 / 8 = 0.5529 s. Adding Ts = 0.1667 s gives 0.7195 s.
       At R = 10: mu = 60, mu - lambda = 20, rho = 0.6667, ln(66.667) = 4.19971,
       t_99 = 0.2100 s, plus Ts 0.1667 s gives 0.3767 s. At R = 12: rho = 0.5556,
       ln(55.556) = 4.01738, t_99 = 0.1255 s, total 0.2922 s.
INTERPRETATION: the mean says both fleets are fine; the p99 says 8 replicas sits at 720 ms
       against a 1,000 ms deadline, which is why "the average is 180 ms" never settles an argument.
```

**Validity condition for 4.1.** C1 requires a constant service rate and a single price per accelerator-hour.
C2 requires that both volumes and capacity refer to the same interval. C3 requires Poisson arrivals,
independent service times, and a single queue: any per-tenant queue, sticky routing or batch co-tenancy
invalidates the closed form and requires measurement instead.

### 4.2 General case

**Conservation of admission credits, which is the only invariant a quota has.**

```text
FORMULA [C4]: sum over actors a of held(a) + free = B, and held(a) <= cap(a)
WHERE: B is the global budget in credits, held(a) is the credits currently leased to actor a,
       free is the unallocated remainder, cap(a) is that actor's ceiling.
CONSTRAINT: grant, renew and revoke must be serialised against one authority, or fenced so that a
       stale message cannot act after a newer one.
NUMERICAL EXAMPLE: B = 100, three replicas each configured with a local ceiling of 40 and no
       shared ledger. Each admits on its own: total held = 120, free = -20. The invariant fails
       by 20 credits, i.e. 20% of the budget, and no single replica ever sees a violation.
INTERPRETATION: summing local views is the check. Any admission design that cannot answer "what is
       held in total right now" has no invariant; it has a hope.
```

**Retry amplification, including the finite-attempt version people actually implement.**

```text
FORMULA [C5]: q_inf = 1/(1 - a) for unlimited attempts;  q_n = sum from i = 0 to n-1 of a^i
WHERE: a is the per-attempt failure probability in (0, 1); n is the maximum attempt count;
       q is offered load divided by original load.
CONSTRAINT: a must be the failure probability of the *attempt*, not of the request. If the caller
       retries on timeout, a is the timeout rate, which rises with load - that is the feedback.
NUMERICAL EXAMPLE: a = 0.02, n = 4: q_4 = 1 + 0.02 + 0.0004 + 0.000008 = 1.0204. At 40 req/s of
       originals, extra load is 40 * 0.0204 = 0.82 req/s. With a = 0.35: q_3 = 1 + 0.35 + 0.1225
       = 1.4725 and q_4 = 1.515375, so four attempts captures 1.515375/1.538462 = 98.5% of the
       infinite limit; a full outage with n = 3 turns 40 req/s into 120 req/s.
INTERPRETATION: below 5% failure the geometric tail is negligible; between 20% and 50% the
       amplifier is the outage. A retry policy without a budget is a load multiplier wired to the
       thing that is already failing.
```

**Cache staleness exposure, which is a count of responses and not a duration.**

```text
FORMULA [C6]: E = Q * TTL * f_revoked
WHERE: E is the expected number of responses served from a cache entry whose underlying
       permission changed; Q is queries per second hitting that entry class; TTL in seconds;
       f_revoked is the fraction of entries whose authorisation changed within the TTL.
CONSTRAINT: Uniform arrival and independent revocations. A burst after a bulk permission change
       makes f_revoked far higher than its average, so E is a floor, not a ceiling.
NUMERICAL EXAMPLE: Q = 40, TTL = 300 s, f_revoked = 0.18 gives E = 40 * 300 * 0.18 = 2,160
       responses. Reducing TTL to 30 s cuts E to 216 but multiplies origin load by ten.
INTERPRETATION: the safe fix is an epoch in the key, which makes E = 0 at the cost of one
       invalidation event, instead of a TTL that trades correctness for hit rate.
```

**Attribution, which must state what happens to the unattributable remainder.**

```text
FORMULA [C7]: A_t = (C - U) * w_t,  with sum over t of w_t = 1;  A'_t = C * w_t if the
              unattributed U is spread pro rata instead of held centrally
WHERE: C is the total bill, U is the portion on resources with no owner tag, w_t is team t's
       share of attributed spend.
CONSTRAINT: If U is spread pro rata, a team's chargeback grows when *another* team loses a tag,
       which destroys the incentive to tag.
NUMERICAL EXAMPLE: C = $20,719.83, U = 6.5% = $1,346.79, so attributed spend is $19,373.04.
       Team A at w = 0.45 pays $8,717.87 under central holding, and $9,323.93 under pro rata
       (8,717.87 + 0.45 * 1,346.79 = 8,717.87 + 606.06).
INTERPRETATION: the $606.06 difference is the price of the policy choice, not of the infrastructure.
```

**Validity condition for 4.2.** C4 requires one serialisation point per scope. C5 requires independent
attempt failures; a shared dependency failure makes every retry fail together and the series diverges to n.
C6 requires a known f_revoked, which most systems do not measure, so treat E as illustrative until it is.
C7 requires tag coverage to be measured rather than assumed.

### 4.3 Variants

**Token bucket: burst is a policy number, not a configuration accident.**

```text
FORMULA [C8]: bucket capacity b, refill rate r per second; a client may emit up to
              b + r*t credits in any interval t. Emission interval = 1/r.
WHERE: b is in requests (credits); r = rate/60 when the rate is configured per minute;
       t is the observation interval in seconds.
CONSTRAINT: The bucket is per key. If the key is per replica or per connection, the effective
       burst is b times the number of keys.
NUMERICAL EXAMPLE: rate 600/min gives r = 10/s and an emission interval of 100 ms. With b = 2,400
       a tenant may emit 2,400 immediately, which is four minutes of average traffic. Sending
       3,000 in the first second admits 2,400 and rejects 600 - a 20% rejection. Sustained
       900/min against 600/min rejects (900-600)/900 = 33.3%.
INTERPRETATION: b = 2,400 is a decision to tolerate a 4-minute burst; nobody chose it, it is the
       default. Express b as seconds of rate so the decision is visible.
```

**Burn-rate alerting, where the window is the whole design.**

```text
FORMULA [C9]: burn = observed_error_rate / (1 - SLO);  budget consumed in window T =
              burn * T / W, where W is the SLO window.
WHERE: SLO = 0.999 means budgeted error rate 0.001; T and W in the same unit (hours).
NUMERICAL EXAMPLE: W = 720 h (30 days), so 43.2 minutes of budget. Burn 14.4 sustained for
       T = 1 h consumes 14.4 * 1 / 720 = 2.0% of budget = 0.864 min. Burn 2 for 1 h consumes
       0.278% = 0.12 min. Burn 1 for the whole window consumes 100%.
INTERPRETATION: a 5-minute window with a burn threshold of 14.4 is noisy; a 1-hour window with
       the same threshold is slow. The pair (long window, low threshold) plus (short window, high
       threshold) is what separates a page from a ticket.
```

**Metric cardinality is a product, and that is why it surprises people.**

```text
FORMULA [C10]: N_series = product over label i of |values(i)|, minus combinations that never occur
WHERE: |values(i)| is the distinct value count of label i on one metric.
CONSTRAINT: Labels that are unique per request (request id, user id, trace id) make N_series
       equal to traffic, not to configuration.
NUMERICAL EXAMPLE: tenant 200, route 35, status 6 gives 200 * 35 * 6 = 42,000 series for that
       metric alone, replacing its 210. At $0.192 per series-month the metric goes from
       210 * 0.192 = $40.32 to 42,000 * 0.192 = $8,064.00, a 200x increase, and the monitoring
       line goes from 1,250 to 43,040 series, i.e. $240.00 to $8,263.68, a 34.4x increase.
INTERPRETATION: the expensive operation is not adding a metric, it is adding a label with high
       domain. Cardinality review belongs on the same gate as a schema change.
```

**Little's law applied to draining, which decides how many requests a rollout can lose.**

```text
FORMULA [C11]: L = lambda * W
WHERE: L is requests in flight, lambda is arrival rate in req/s, W is time in system in seconds.
CONSTRAINT: Steady state, and the same population on both sides. During a drain the arrival rate
       into a single replica is not the aggregate rate; divide by the replica count.
NUMERICAL EXAMPLE: aggregate 1,200 req/s with W = 0.05 s gives L = 60 in flight at any instant.
       If 4% of those arrive after the listener closes and are not retried, one drain drops
       60 * 0.04 = 2.4 requests; a rollout touching all 8 replicas drops about 19.
INTERPRETATION: the drain timeout must exceed W by a margin, and the client must retry what it
       loses. A drain that only closes the listener is a coin flip.
```

**Validity condition for 4.3.** C8 requires one bucket per key and a monotone clock. C9 requires the SLO
window to match the budget being consumed. C10 requires the label domains to be measured, not estimated.
C11 requires steady state; a canary at 5% of traffic is not steady state for that replica.

### 4.4 Limits and counterexamples

**Where C3 fails, and how it fails.** M/M/1 assumes a single queue and memoryless arrivals. A platform that
batches inference requests has a service time that is *lower* at higher load (batching amortises the fixed
cost), so measured p99 can improve while utilisation rises. That is not a contradiction; it is C3 being
inapplicable. The discriminating observation is whether service time varies with batch size: if it does,
quote measured quantiles and never the closed form.

**Where C4 fails.** A lease scheme with a fencing token is safe under partitions only if the resource itself
checks the token on use. If the token is checked at grant time but not at use time, a partitioned replica
can still spend credits granted before the partition; conservation holds on paper and fails in the sink.

**Where C1 fails.** C1 assumes one price and one service rate. On spot capacity the price moves and
interruptions remove capacity, so `u` becomes a mixture: you cannot average the price without averaging the
interruption rate, and the correct statement is a range with the interruption assumption attached.

**A counterexample to "lower unit cost is better".** Dropping from 10 to 8 replicas cuts the monthly bill by
$4,672 and *improves* cost per request from $0.80517/1k to $0.65702/1k — 18.4% cheaper. Now attach a 300 ms
deadline and compute the queue-time breach rate from C3's own tail formula, `P(Wq > t) = rho*exp(-(mu-lambda)t)`
with `t = 0.300 - 0.1667 = 0.1333` s:

```text
8 replicas:  rho = 0.8333, mu - lambda = 8.  P = 0.8333 * exp(-8 * 0.1333)
             = 0.8333 * exp(-1.0667) = 0.8333 * 0.34415 = 0.28678
10 replicas: rho = 0.6667, mu - lambda = 20. P = 0.6667 * exp(-20 * 0.1333)
             = 0.6667 * exp(-2.6667) = 0.6667 * 0.06948 = 0.04631
```

Cost per resolved task: 0.65702 / (1 - 0.28678) = $0.92031 per 1,000 resolved, against
0.80517 / (1 - 0.04631) = $0.84426. The configuration that is 18.4% cheaper per request is **9.0% dearer per
resolved task**. No price changed; only the denominator did. Any unit-cost claim that does not name its
denominator and its deadline can be reversed without touching the infrastructure.

**Validity condition for 4.4.** Each formula above is a model with named preconditions. When a precondition
is unmet, the honest output is a measurement with its sampling window, not a corrected formula.

## 5. Worked Numerical Example

This trace prices one month on the reference platform and then stress-tests the answer. Every figure is
`ILLUSTRATIVE` construction; the arithmetic is exact.

**Step 1 — component bill.**

| Component | Driver | Quantity | Rate | Cost |
|---|---|---|---|---|
| Inference accelerators | 8 replicas × 730 h | 5,840 accelerator-hours | $3.20/h | $18,688.00 |
| Gateway | 4 instances × 730 h | 2,920 instance-hours | $0.052/h | $151.84 |
| Admission service | 2 instances × 730 h | 1,460 instance-hours | $0.078/h | $113.88 |
| Log ingestion | 31,536,000 req × 2.5 KiB | 75.19 GiB | $0.50/GiB | $37.59 |
| Audit archive | retained volume | 1,240 GiB | $0.023/GiB | $28.52 |
| Metrics | 1,250 series | 1,250 series | $0.192/series | $240.00 |
| Non-production | staging and preview namespaces | — | — | $1,460.00 |
| **Total** | | | | **$20,719.83** |

**Step 2 — normalisation.** Volume is 12 req/s × 3,600 s × 730 h = 31,536,000 requests. Output tokens are
31,536,000 × 220 = 6,937,920,000. Therefore cost per 1,000 requests is 20,719.83 / 31,536 = $0.65702, and
cost per 1 million output tokens is 20,719.83 / 6,937.92 = $2.98646.

**Step 3 — independent re-check.** Multiply the rounded rate back: 0.65702 × 31,536 = $20,719.78, which is
$0.05 below the component sum — the rounding loss of quoting five decimals is 31,536 × $0.0000015 = $0.05.
The token rate behaves the same way: 2.98646 × 6,937.92 = $20,719.83, agreeing to the cent. Two
normalisations that reconstruct the same bill to within a rounding error are consistent; one that does not
has a denominator problem.

**Step 4 — how much is compute.** Accelerators are 18,688 of 20,719.83, i.e. 90.2% of the bill. Yet
18,688 / 31,536 = $0.5926 per 1,000 requests is *entirely* a utilisation statement: at the same price and
service rate, doubling utilisation would halve it. The remaining $0.0644 per 1,000 is fixed overhead that
does not move with traffic at all.

**Step 5 — what one more replica costs and buys.** A tenth replica adds 3.20 × 730 = $2,336 per month,
taking the bill to $25,391.83 (+22.6%). Cost per 1,000 requests becomes 25,391.83 / 31,536 = $0.80517 and
cost per 1M output tokens becomes $3.6599. The marginal replica costs $2,336 / 6,937.92 = $0.3367 per 1M
tokens, and two replicas added $0.6734, so the decomposition is exact.

**Step 6 — the latency side.** From C3, p99 total goes 719.5 ms at 8 replicas, 376.7 ms at 10 and 292.2 ms at
12. The first increment buys 342.8 ms for $4,672; the second buys 84.5 ms for the same $4,672. Marginal
price of tail latency rises from $13.63/ms to $55.29/ms — a 4.06× deterioration with no change in unit price.

**Step 7 — sensitivity, stated as intervals not points.**

| Perturbation | Bill | Per 1k requests | Why |
|---|---|---|---|
| Baseline, 8 replicas, 12 req/s | $20,719.83 | $0.65702 | reference |
| Demand +25% to 15 req/s, no capacity change | $20,719.83 | $0.52562 | denominator grows, numerator does not |
| Baseline costed at the *peak* pairing rho = 0.8333 | $5,606.40 equiv. | $0.17778 | wrong pair: peak capacity, average volume |
| 10 replicas, demand unchanged | $25,391.83 | $0.80517 | provisioned for tail, not mean |
| Log level to debug, 3× ingest volume | $20,795.03 | $0.65941 | 37.59 → 112.79 on 225.57 GiB at $0.50 |
| Metrics label `tenant` added (total 43,040 series) | $28,743.51 | $0.91145 | +$8,023.68 on series cost |

**Step 8 — arithmetic that ties the sensitivity table together.** The debug-log row adds
225.57 − 75.19 = 150.38 GiB at $0.50 = $75.19, so the log line goes 37.59 → 112.79 and the bill
20,719.83 + 75.19 = $20,795.02, giving 20,795.02 / 31,536 = $0.65941 (the cent difference in the table is
where the two rounding paths meet). The cardinality row replaces the metric's 210 series with 42,000, so total
series become 43,040 and the line becomes 43,040 × 0.192 = $8,263.68: a delta of $8,023.68, giving
$20,719.83 + 8,023.68 = $28,743.51 and 28,743.51 / 31,536 = $0.91145 per 1,000. The peak-pairing row is not a
bill at all: it is C1 evaluated with rho = 0.8333, and the $5,606.40 "equivalent" (0.177778 × 31,536) is the
fiction that the fleet runs at peak utilisation around the clock, which is exactly the error C2 catches.

**Step 9 — the sensitivity note that matters.** The demand row is the dangerous one. A 25% traffic increase
with no capacity change *reduces* cost per 1,000 requests from $0.65702 to $0.52562, a 20% improvement, while
by C2 the peak utilisation goes from 40/48 = 0.8333 to 50/48 = 1.0417 — which is not a degradation, it is
infeasibility: demand exceeds provisioned capacity and the queue has no stationary distribution. Even at the
average rate the picture inverts: rho_avg rises 0.2500 → 0.3125 and C3 gives t_99 =
ln(0.3125/0.01)/(48 − 15) + 1/6 = ln(31.25)/33 + 0.1667 = 0.1043 + 0.1667 = 271 ms, but that figure is
meaningless on a system whose peak already saturates. Unit cost
improves precisely when the system is about to fall over, so a dashboard that reports only unit cost rewards
the behaviour that burns the error budget.

**Takeaway.** On this platform, cost is a utilisation statement and latency is a provisioning statement, and
the two move in opposite directions for every traffic change. Any single number that claims to summarise
both is hiding one of them.

## 6. Visual / Geometric Intuition

**The queue is the picture of the whole capacity problem.** Waiting time is not linear in load; it is the
area between the arrival curve and the service curve, and that area grows as a hyperbola.

```text
rho      Wq (s)   Wq + Ts (ms)   reading
0.50     0.1667   333            comfortable, half the fleet idle
0.667    0.3333   500            two spare replicas over the 8-replica case
0.833    0.8333   1000           one bad minute from the deadline
0.9375   2.5000   2667           cost per request looks great, users do not
```

Each row is C3 with s = 6 req/s: Wq = rho/(1-rho) × Ts. At rho = 0.833 the mean wait is five service
times; at 0.9375 it is fifteen, and the fourth row is 2.7x the third for a load change of 12.5%. The curve
has no knee to warn you — it is smooth everywhere and steep everywhere near 1.

**Admission is a ledger, and the ledger has one page.**

```text
client -> gateway (auth, route) -> admission (credits) -> workload -> idempotent sink
                                        |
                                   one ledger, one scope
                                        |
             +--------------------------+--------------------------+
             |                          |                          |
        grant + fence              renew + fence             revoke + fence
             |                          |                          |
        held(a) += n              held(a) = n                held(a) -= n
             |                          |                          |
        reject if free < n        reject if fence stale       ignore if fence stale
```

The four boxes on the right are the entire safety argument. Any arrow that skips the ledger — a replica
trusting a cached allowance, a client trusting a response header — is a place where the sum in C4 can exceed
the budget.

**Cost attribution is a partition, and a partition has a remainder.**

```text
total bill  $20,719.83  ==============================
  attributed $19,373.04  ========================  93.5%
  unattributed $1,346.79 =                        6.5%
                                  ^
                                  +-- 275 of 1,250 resources carry no owner label
```

The remainder is not noise; it is a measurable object with an owner (nobody) and a trend (usually growing).
Watching the remainder is cheaper and more honest than watching the sum of the parts.

**The rollout window is where contracts break.**

```text
t=0     new replica starts, model loads (9.4 s), readiness probe passes at t=1 s  <- too early
t=1     traffic arrives at a cold replica: p50 2.1 s -> 11.5 s
t=9.4   warm; the canary's own 30-sample check now looks fine
t=20    drain begins: 60 requests in flight, 4% not retried -> 2.4 lost
```

Three separate failures share this one picture, which is why "the rollout broke it" is a symptom and not a
diagnosis: warm-up, drain, and the canary's sample size are three different owners.

**Takeaway.** Draw the ledger or the queue before drawing a fix; the picture tells you which formula applies.

## 7. Comparison / Selection Table

**Admission and quota mechanisms.**

| Mechanism | Invariant it can hold | Failure mode | Cost per decision | Choose when |
|---|---|---|---|---|
| Per-replica token bucket | Local rate only | Aggregate is n × configured rate | ~50 ns, in process | A single tenant, no global claim needed |
| Shared atomic counter | Global rate exactly | Coordinator outage stops all traffic | One round trip, 0.4 ms p50 | Correctness of spend matters more than availability |
| Leased budget with fencing | Global budget under partition | Lease leaks inflate spend until expiry | Background renewal, amortised | High throughput, tolerant of bounded overshoot |
| Hierarchical quota (tenant → key) | Per-level ceilings | Parent/child skew; child starves | O(depth) lookups | Multi-tenant billing with contractual ceilings |
| Queue with admission by deadline | Served work meets SLO | Queue depth hides load; timeouts masquerade as rejections | Queue wait | Batch-tolerant work, not interactive traffic |

**Alerting mechanisms.**

| Mechanism | Detects | Misses | Operational cost |
|---|---|---|---|
| Static threshold on error rate | Gross breakage | Slow burns below threshold | Low, but noisy at low traffic |
| Burn rate, short window | Fast, large burns | Slow sustained burns | Pages often; needs a long-window partner |
| Burn rate, long window | Slow burns | Fast events after they are over | Slow to act |
| SLO attainment at end of window | Policy compliance | Everything, too late to act | Reporting only |
| Synthetic probe | Availability of one path | Everything with state | Cheap, shallow |

**Cost allocation mechanisms.**

| Mechanism | Accuracy | Incentive effect | Effort |
|---|---|---|---|
| Tag-based showback, remainder held centrally | Bounded by tag coverage (93.5% here) | Rewards tagging | Low |
| Tag-based chargeback, remainder pro rata | Same accuracy | Punishes teams for others' gaps | Low, but resented |
| Namespace/account-per-team | High for compute, poor for shared services | Strong ownership | Medium |
| Metered usage records | High, auditable | Strong, defensible | High |
| Committed-spend amortisation | Matches the contract | Encourages forecasting | High, needs finance |

**Takeaway.** Pick the weakest mechanism that can hold the invariant you actually claim. Every step up the
table buys accuracy with availability, latency or effort, and saying which one you traded is the design.

## 8. Failure Modes & Edge Cases

**Catalogue.** Columns follow the contract: identifier, category, symptom, competing causes, discriminating
check, correction, prevention, verification. All thirty records are `ILLUSTRATIVE` constructions of commonly
reported failure shapes; no incident is claimed. Category vocabulary is VIOLATED_ASSUMPTION, NUMERICAL_EDGE,
SCALE_FAILURE, CONCEPT_CONFUSION.

| ID | Category | Symptom | Competing causes | Discriminating check | Correction | Prevention | Verification |
|---|---|---|---|---|---|---|---|
| CI-01 | VIOLATED_ASSUMPTION | A retried payment creates a second charge | Idempotency key per request; sink not idempotent; retry after a commit | Replay one intent with the same key twice and count sink rows | Key on intent, enforce a unique constraint in the sink | Sink owns uniqueness, not the client | Two identical submits produce one row |
| CI-02 | NUMERICAL_EDGE | Duplicates appear only 26 hours after the event | Idempotency window 24 h; retry horizon 26 h; clock skew | Compare the dedupe window with the maximum retry horizon | Window at least the retry horizon plus skew | Record both values in one config block | A replay at horizon+1 is rejected |
| CI-03 | CONCEPT_CONFUSION | Rejected requests are counted as served | 429 mapped to success; retry wrapper swallows status; metric counts attempts | Join the served count to the sink's row count | Count outcomes, not attempts | Effectiveness metric is first-class | Served count equals sink rows within one interval |
| CI-04 | VIOLATED_ASSUMPTION | Tenant A's quota is consumed by tenant B | Quota keyed by API key; tenant in header; shared key across tenants | Send two tenants through one key and read the bucket key | Key on tenant id resolved server-side | Quota key derived from authenticated identity | Two tenants with one key hold separate balances |
| CI-05 | NUMERICAL_EDGE | A tenant exceeds its minute rate by ~3% | Refill on a jittered wall clock; burst carried over; rounding up | Compare admitted credits over a window with the analytic bound | Refill on a monotonic clock, compute from the last refill instant | Unit test the refill at 250 ms jitter | Admitted total never exceeds b + r*t |
| CI-06 | SCALE_FAILURE | Aggregate traffic is 3× the configured platform limit | Per-replica buckets; HPA scaling the bucket count; no global view | Sum configured ceilings across replicas | One ledger per scope, or divide the ceiling by replica count | Ceiling expressed as a global number | Sum of replicas' held credits never exceeds the budget |
| CI-07 | NUMERICAL_EDGE | Renewal produces 140 credits against a 100 budget | Renew before revoke; no fence; duplicate renewal messages | Log every grant with fence and sum per scope | Fence tokens; renew replaces rather than adds | Ledger invariant asserted per write | Total held is constant across a renewal |
| CI-08 | VIOLATED_ASSUMPTION | A burst at exactly the lease expiry | Expiry treated as release; client keeps spending into expiry | Correlate admitted rate with expiry timestamps | Expire by returning credits, not by forgetting them | Expiry and release share one code path | Admitted rate at expiry stays inside the budget |
| CI-09 | SCALE_FAILURE | An outage becomes a self-sustaining storm | Unlimited retries; no backoff; synchronised jitter | Compute C5 at the observed a and compare with capacity | Retry budget as a ratio; jitter; circuit breaker | Budgets enforced by the client library | Offered load stays under capacity during an injected fault |
| CI-10 | CONCEPT_CONFUSION | A "retry budget" is set to 3 and latency still explodes | Retry count versus retry ratio; timeout budget confused | Read both the attempt cap and the ratio from config | Hold the ratio; derive attempts per call site | Timeout and retry documented together | Worst-case latency equals the sum of stated timeouts |
| CI-11 | VIOLATED_ASSUMPTION | A revoked user reads a cached response | Cache key lacks authorisation epoch; TTL-only invalidation | Revoke a principal, then request the same URL | Include epoch in the key; purge on change | Authorisation state in the cache key | Post-revocation request misses the cache |
| CI-12 | NUMERICAL_EDGE | Stale exposure is far larger than the TTL suggests | Hot key concentration; bulk role change; negative caching | Compute C6 with the measured f_revoked, not an assumed one | Epoch keys; cap TTL on authorisation-bearing responses | Separate cache policy for protected responses | Exposure count measured per incident window |
| CI-13 | SCALE_FAILURE | First requests to a new replica are slow | Readiness before warm; probe on a cheap endpoint; no startup gate | Compare probe success time with warm completion time | Gate readiness on a warm-up request | Readiness includes a dependency check | Cold-replica p50 within the warm p50 band |
| CI-14 | CONCEPT_CONFUSION | A replica restarts in a loop under load | Liveness includes the dependency; probe timeout too short | Read the probe definitions side by side | Liveness local only; readiness may include dependencies | Probe contract written down | Injected dependency fault does not restart pods |
| CI-15 | NUMERICAL_EDGE | Autoscaler holds 3 replicas while p99 breaches | Utilisation measured against request not limit; target too high | Read the target, the denominator and the measured value | Target utilisation with the correct denominator | Scaling policy unit-tested against a load profile | Replica count at peak matches C2 arithmetic |
| CI-16 | SCALE_FAILURE | Rollout drops a handful of requests per pod | Drain closes the listener; no preStop delay; client does not retry | Compare in-flight count at drain with retried count | Drain delay exceeding W, plus client retry | Drain budget derived from C11 | Zero unretried drops across a full rollout |
| CI-17 | SCALE_FAILURE | Rollout takes six minutes per node | Image size; serial pulls; registry egress limit | Time one pull and multiply by nodes and concurrency | Smaller base image; pre-pull daemon | Image size budget in CI | Pull completes inside the rollout budget |
| CI-18 | VIOLATED_ASSUMPTION | A node drain takes the whole service down | PDB minAvailable 100%; single replica; no anti-affinity | Read PDB and replica count together | PDB below replica count; spread across zones | PDB reviewed against replica count | Drain completes with capacity above the floor |
| CI-19 | NUMERICAL_EDGE | Mean latency passes while 10% of users time out | Mean over a bimodal distribution; retries hidden; client deadline | Plot the distribution, not the mean | SLO on a percentile with a deadline | Percentile-based SLO at the client | p99 at the client under the deadline |
| CI-20 | CONCEPT_CONFUSION | Gateway SLO is green, support tickets rise | SLO measured at the gateway; client-side retries; DNS and TLS excluded | Compare gateway and client-side timings for one trace | Measure at the boundary the user sees | SLI instrumentation at the client | Gateway and client p99 gaps explained |
| CI-21 | NUMERICAL_EDGE | The page fires after the budget is half gone | Burn window too long; threshold too low; window shorter than scrape interval | Recompute C9 at the observed burn and window | Pair a long window with a short one | Alert rules reviewed against the budget arithmetic | Injected fault consumes under 2% before paging |
| CI-22 | VIOLATED_ASSUMPTION | Chargeback exceeds the invoice for a team | List price versus amortised commitment; credits unallocated | Reconcile one team's sum against the invoice line | Amortise commitments; allocate credits explicitly | One pricing basis across all reports | Sum of team charges equals the invoice |
| CI-23 | SCALE_FAILURE | Monitoring cost grows 30× after one label | Cardinality product; high-domain label; unique ids | Compute C10 from the label schema | Drop the label or aggregate; use exemplars | Cardinality budget per metric | Series count within the budget after deploy |
| CI-24 | VIOLATED_ASSUMPTION | A budget alert fires on day two of the month | Alert on committed spend; forecast versus actual; amortisation | Read which quantity the alert evaluates | Alert on run-rate forecast and on actuals separately | Alert definitions name their quantity | Alert fires at the intended spend condition |
| CI-25 | CONCEPT_CONFUSION | Cost per request falls while cost per task rises | Denominator changed; retries counted as requests | Recompute both ratios over the same window | Report cost per resolved task | One canonical denominator | Both ratios move consistently with the story |
| CI-26 | NUMERICAL_EDGE | A saving of 30% disappears at month end | Unit cost computed at peak utilisation; discount expiry; egress excluded | Recompute C1/C2 with the realised utilisation | Quote cost per month at realised load | Cost model reviewed with the capacity plan | Modelled and actual bills agree within 2% |
| CI-27 | SCALE_FAILURE | Log volume triples after a config change | Debug level; per-request payload logging; retry duplication | Histogram bytes per request by log level | Sampling and structured levels | Log budget per service in CI | Ingested volume within the monthly budget |
| CI-28 | VIOLATED_ASSUMPTION | A debug endpoint exposes a credential | Verbose error; env echo; long-lived secret | Request the endpoint as an unauthorised caller | Remove the endpoint; rotate the secret | Secret scanning plus endpoint inventory | Unauthorised request returns no secret material |
| CI-29 | CONCEPT_CONFUSION | One tenant's content appears in another's answer | Prompt-level instruction as the boundary; shared retrieval index | Retrieve the same query as two tenants and diff document ids | Enforce at the data layer, per-tenant partitions | Isolation test in the release gate | Cross-tenant retrieval returns zero foreign documents |
| CI-30 | NUMERICAL_EDGE | Autoscaler reacts after the burst is over | Stabilisation window longer than the burst; scale-up delay | Compare burst duration with the window and the cold-start time | Shorter scale-up window; pre-provisioned floor | Policy tuned to the measured burst profile | Replica count tracks a synthetic burst |

**Co-occurrence and masking.** Records cancel each other, and the cancellation is usually what makes a
diagnosis slow. CI-25 (wrong denominator) hides CI-26 (wrong utilisation), because both inflate the same
number and the corrected figure looks like a regression. CI-12 (stale exposure) hides CI-11 (missing epoch)
because the TTL makes the leak look like a timing artefact. CI-03 (attempts counted as served) hides CI-09
(retry storm) because the storm inflates the served count that is being used as the denominator. When two
records are plausible, run the *cheaper* check first even if it belongs to the less likely cause: a defect
ruled out in five minutes is worth more than a hypothesis argued for a day.

**Severity, assigned only after the check.** CI-01, CI-11, CI-28 and CI-29 are Sev-1 because they are
correctness, security or isolation defects that no latency or cost metric in this file would surface. CI-02,
CI-05, CI-07, CI-15, CI-19 and CI-22 are Sev-2 because they invalidate a decision input — a number somebody
will act on. The remainder are usually Sev-3 cost or operability issues, with two exceptions: CI-24 becomes
Sev-1 the moment a team is paged on it, and CI-21 becomes Sev-1 when a release gate depends on the SLO.

### CI-01 — the retry that charges twice

**Mechanism.** The client generates the idempotency key inside the request loop rather than once per user
intent, so a retry carries a fresh key. The gateway forwards both, the sink inserts two rows, and the
reconciliation job sees two legitimate-looking payments.

**Arithmetic that localises it.** Take 40 req/s with a 2% timeout rate and one retry: the extra call rate is
40 × 0.02 = 0.8 req/s, so 0.8 duplicate intents per second exist, and every one of them is a potential double
charge. Over a month at 31,536,000 requests and 2% timeouts, the exposure is 630,720 retried requests.

**Check that settles it.** Submit one intent twice with the identical key and count sink rows. If the count
is two, the sink is not enforcing uniqueness and no client-side change can save you.

**Correction and its cost.** Move uniqueness into the sink as a constraint on (scope, key). The cost is one
index and a rejected-write path; the rejected write must return the original result, or the client will treat
a successful retry as a failure.

**Prevention.** The client library generates the key from the intent, and a contract test asserts that a
retried call produces one row.

### CI-02 — the duplicate that appears a day later

**Mechanism.** Deduplication keeps keys for 24 hours. A downstream batch retries stragglers for up to 26
hours. Intents in the two-hour gap lose dedupe protection and become duplicates, which is why the symptom
appears on a schedule rather than at the moment of failure.

**Arithmetic.** With 31,536,000 requests a month and a 0.05% straggler rate, 15,768 requests land in the gap;
at a 3% duplicate-producing failure rate inside that gap the expected duplicates are near 473 per month.

**Check.** Compare the dedupe window and the maximum retry horizon as two numbers in one place. If the window
is smaller, the defect is proved without any traffic experiment.

**Correction.** Set the window to the retry horizon plus clock skew plus the longest queue delay. Cost: key
storage grows linearly in window length, so the window is a storage decision as well as a correctness one.

**Prevention.** Both numbers live in one configuration object and a test fails when the window drops below the
horizon.

### CI-03 — rejected requests that count as served

**Mechanism.** The client wrapper treats any HTTP response as success, including 429 and 503. The served
metric counts completions rather than sink writes, so a quota that is rejecting 20% of traffic shows as a
healthy request count and a modest latency improvement.

**Arithmetic.** With a token bucket of 600/min and a burst of 2,400, a tenant sending 3,000 in the first
second is rejected 600 times — a 20% rejection that the wrapper reports as 3,000 successes.

**Check.** Join the served count against the sink's row count for the same interval. A gap of 20% is the
rejection rate; a gap of 2% is retries plus in-flight requests.

**Correction.** Count outcomes at the sink, and make the wrapper distinguish 2xx from everything else. The
cost is one extra metric and a slightly wider cardinality budget.

**Prevention.** The effectiveness metric is derived from the sink, so no instrumentation change in the gateway
can inflate it.

### CI-04 — the tenant whose quota somebody else spent

**Mechanism.** Quota is keyed on the API key, but one key is shared by a partner that proxies several
tenants. The bucket is correct in isolation and wrong in aggregation: one tenant's burst consumes another's
allowance, and the tenant that complains is not the tenant that caused it.

**Arithmetic.** A 600/min bucket shared by three tenants gives each an effective 200/min, while each
contract promises 400/min. The 200/min shortfall appears only under contention, which is exactly when it
matters.

**Check.** Send two tenants through the same key and read the bucket key from the admission log. If the key
is the API key, the defect is proved.

**Correction.** Derive the quota key from the authenticated identity server-side, never from a header the
caller supplies. Cost: one identity resolution that the gateway already performs.

**Prevention.** A header-based tenant is rejected at review; the quota key function is owned by the admission
service and tested with two tenants on one key.

### CI-05 — the refill that runs early

**Mechanism.** Refill is computed from wall-clock deltas, which drift under NTP correction and jitter. A
negative delta is treated as zero and a large positive delta refills a full step, so an over-admitted window
appears at every clock adjustment.

**Arithmetic.** With a 600/min bucket (r = 10/s) and a 250 ms jitter, one early refill of 2.5 credits adds
2.5 per adjustment; at 20 adjustments a minute that is 50 extra credits per minute, so the admitted rate is
650/min against a nominal 600 — an 8.3% overspend. The analytic bound is b + r·t, so with b = 2,400 and
r = 10/s the tolerated total in 60 s is 3,000, and the jitter case stays inside the burst allowance and
therefore hides: the configuration is not violated, only the intent.

**Check.** Compare admitted credits over a long window with b + r·t computed independently. A sustained
excess that scales with clock adjustments localises the cause.

**Correction.** Refill from a monotonic clock and compute from the stored last-refill instant rather than
from now minus a rounded interval.

**Prevention.** A property test drives the refill with ±250 ms jitter and asserts the bound.

### CI-06 — eight replicas, eight budgets

**Mechanism.** The rate limiter is an in-process token bucket with a per-replica limit of 600/min. At eight
replicas the aggregate ceiling is 4,800/min, and the platform's contractual limit is 1,600/min. Nobody
configured 3×; the configuration is simply not the quantity anybody meant.

**Arithmetic.** 600 × 8 = 4,800 against 1,600 is exactly 3.0×. Adding replicas during an incident raises the
ceiling, which is the worst possible coupling: the autoscaler becomes a quota amplifier.

**Check.** Sum the configured ceilings across replicas and compare with the contractual number. One
multiplication settles it.

**Correction.** Either divide the ceiling by the replica count at render time, or move the ledger to one
authority. Dividing is cheaper and fails under uneven routing; a single ledger costs a round trip.

**Prevention.** The ceiling is expressed once, as a global number, and rendered per replica by the deployment
template.

### CI-07 — renewal that charges the budget twice

**Mechanism.** Renewal is implemented as grant-then-revoke. Between the two messages the ledger holds both
the old and the new grant, so a concurrent admission decision sees a budget that is 40% larger than it is.

**Arithmetic.** Budget 100, three leases of 40. A renewal that adds before it removes gives 40 + 40 + 40 + 40
= 160 transiently, i.e. 60% over budget for the duration of one round trip. If that window is 20 ms and
admissions arrive at 40/s, roughly 0.8 admissions per renewal are decided against phantom credits.

**Check.** Sum held credits before, during and after one renewal with the fence value logged on each write.
A transient overshoot proves the ordering defect.

**Correction.** Make renewal a replace keyed on the fence token, so the new grant supersedes the old one
atomically.

**Prevention.** The ledger asserts the invariant on every write and rejects a write that would break it.

### CI-08 — the burst at the expiry boundary

**Mechanism.** Expiry is implemented as "forget the lease". The credits vanish from the ledger but the client
keeps spending for the remainder of its in-flight window, so the ledger reports free capacity that is already
committed.

**Arithmetic.** If a client's in-flight window is one second and 40 leases expire in the same second, up to
one second of spend at the freed rate is invisible. At 6 req/s per lease released with 40 credits each, that
is up to 240 decisions made against credit that was already spent.

**Check.** Correlate admitted rate with expiry timestamps: a spike within one second of a batch of expiries
is the signature.

**Correction.** Expiry returns credits *and* increments a generation counter that the client must present on
the next use.

**Prevention.** Expiry and release share one code path, so there is only one place to get it wrong.

### CI-09 — the outage that retries itself

**Mechanism.** Three clients retry three times with no budget and no jitter. During a partial outage the
per-attempt failure rate rises to 0.35, and the retries are what keeps it there: the offered load exceeds the
capacity that the residual healthy capacity must absorb.

**Arithmetic.** C5 at a = 0.35 gives q_inf = 1.5385; with n = 4 attempts, q_4 = 1.5154. Base load 40 req/s
becomes 60.6 req/s. In a full outage with n = 3, offered load is 120 req/s against 48 req/s of capacity — the
recovery never gets a quiet moment.

**Check.** Compute q at the observed failure rate and compare offered load with provisioned capacity. If
offered exceeds capacity during the incident, the retry policy is a participant, not a victim.

**Correction.** A retry budget expressed as a ratio (retries ≤ 10% of originals) plus full jitter. Cost:
some legitimately transient failures are no longer retried, so the correction trades a small success loss for
recoverability.

**Prevention.** The ratio is enforced in the client library, not in each service's configuration.

### CI-10 — the retry budget that was a retry count

**Mechanism.** One team documents a "retry budget of 3" meaning three attempts; another reads it as a 3×
ratio. The second team's client retries every failure three times, and the difference only appears at the
worst moment.

**Arithmetic.** Three attempts at a = 0.35 give q_3 = 1.4725; a "3× ratio" permits q = 3. The factor between
the two readings is 3 / 1.4725 = 2.04, which is the difference between a degraded service and a dead one at
48 req/s of capacity.

**Check.** Read the attempt cap and the ratio from configuration and put them side by side. If a document
says "budget" without a unit, the confusion is already present.

**Correction.** Rename both: `max_attempts` and `max_retry_ratio`, with units in the key.

**Prevention.** Configuration keys carry units, and the worst-case latency is derived as the sum of the stated
timeouts, which makes an inconsistent pair visible in review.

### CI-11 — the revoked user with a warm cache

**Mechanism.** Cached responses are keyed on (path, query, tenant) and invalidated by TTL. A permission change
does not touch the key, so a revoked principal receives a response that was authorised before the change.

**Arithmetic.** At 300 s TTL and 40 req/s on one entry class, the exposure window is 12,000 responses; at an
18% revocation rate inside the window, 2,160 responses are served to principals whose authorisation changed.

**Check.** Revoke a principal, then request the same URL immediately. A 200 with cached-body headers proves
the key is missing the epoch.

**Correction.** Put an authorisation epoch in the key and purge on change. Cost: one cache version bump per
permission change, which lowers the hit rate exactly once per change instead of continuously.

**Prevention.** A release-gate test revokes a principal and asserts a cache miss.

### CI-12 — stale exposure larger than the TTL implies

**Mechanism.** The TTL is chosen from an average revocation rate, but revocations arrive in bulk: a role
definition changes and a thousand principals change at once. The average f_revoked understates the incident
value by an order of magnitude.

**Arithmetic.** C6 with Q = 40 and TTL = 300 s gives a 12,000-response exposure per entry class regardless of
f_revoked; the incident fraction then decides how many are wrong. During a bulk change, the affected
population is the whole class, not 18% of it.

**Check.** Measure f_revoked for the incident window rather than for the month, and recompute E.

**Correction.** Separate cache policy for responses that carry authorisation decisions: shorter TTL, epoch
keying, or no caching at the shared layer.

**Prevention.** Exposure is a measured quantity per incident, and the cache policy names the response classes
it is not allowed to hold.

### CI-13 — traffic arrives before the model is warm

**Mechanism.** Readiness probes a health endpoint that returns 200 as soon as the process listens. Model load
takes 9.4 s, so for 9.4 s a replica is in the service and cannot serve within the latency target.

**Arithmetic.** Warm p50 is 2.1 s and cold p50 is 11.5 s — 5.5× worse. A rolling update that replaces one
replica at 6 req/s sends 6 × 9.4 = 56 requests into the cold window, and against a 2 s deadline essentially all
of them breach it, because the cold p50 alone is 11.5 s. The canary that samples 30 requests therefore sees a
mean of about 11.5 s if all 30 land in the window, or 2.1 s if none does — which is why the canary sometimes
passes and the same commit fails on the next rollout.

**Check.** Compare probe success time with warm completion time from the same replica's logs. The gap is the
defect.

**Correction.** Gate readiness on a dependency-aware warm-up request. Cost: rollout takes 9.4 s longer per
replica, which for eight replicas with one surge slot is 75 s of extra rollout time.

**Prevention.** The readiness handler performs one warm request against the model and fails closed.

### CI-14 — liveness that restarts a healthy pod

**Mechanism.** The liveness probe queries the admission service. When admission is slow, every replica fails
liveness and restarts, which removes warm capacity at the moment the platform is struggling.

**Arithmetic.** Eight replicas restarting with a 9.4 s warm-up means provisioned capacity is zero for 9.4 s at
peak demand of 40 req/s, which is 376 requests with nowhere to go; requests arriving at the mean rate of
12 req/s would be 113. The restarts are also correlated, because they are triggered by the same slow dependency,
so this is not a rolling replacement — it is an outage with a health-check-shaped cause.

**Check.** Read the liveness and readiness definitions side by side. Liveness that names a dependency is the
defect, and it is visible in configuration without a load test.

**Correction.** Liveness tests only the process's own health; readiness may include dependencies.

**Prevention.** The probe contract is written down, and an injected dependency fault is part of the test suite.

### CI-15 — the autoscaler that never catches the tail

**Mechanism.** The scaling target is 70% utilisation, but the measured utilisation divides by the *requested*
CPU while the scheduler gives the pod a fraction of a core, so the reported figure is 30% lower than reality.
The policy believes it has headroom it does not have.

**Arithmetic.** At peak 40 req/s and a target of 70% of a 6 req/s service rate, the required replica count is
40 / (6 × 0.70) = 9.52, i.e. 10 replicas. Holding 8 means running at 40/48 = 0.8333 utilisation against a 0.70
target — a deficit of 1.52 replicas, or 9.14 req/s of capacity that the policy believes it has but does not.
That is why p99 sits at 720 ms rather than 377 ms.

**Check.** Read the target, the denominator and the measured value together for one scaling decision.

**Correction.** Measure utilisation against the resource limit and the actual request, and set the target from
C2 arithmetic rather than from a default.

**Prevention.** A load profile test asserts the replica count at peak against the computed value.

### CI-16 — the drain that closes the door on requests

**Mechanism.** The preStop hook closes the listener immediately, then sleeps. Requests that arrived between
the endpoint removal and the listener close are reset, and a client that classifies a reset as permanent does
not retry them.

**Arithmetic.** C11 at 1,200 req/s aggregate and W = 0.05 s gives 60 requests in flight. With 4% arriving
after the close, one drain loses 2.4 requests; a rollout of eight replicas loses about 19, which is invisible
in a 30-sample canary and visible to users.

**Check.** Compare the in-flight count at drain start with the number of retried requests recorded by the
client. A gap proves unretried loss.

**Correction.** Sleep before closing, longer than W, and make the client retry connection resets.

**Prevention.** The drain delay is derived from the measured W rather than from a default of 5 s.

### CI-17 — six minutes per node for one image

**Mechanism.** The image is 1.4 GB because the build stage was not separated. Twenty-four nodes pull it
through one registry egress path, so the rollout is serialised on bandwidth rather than on scheduling.

**Arithmetic.** 24 × 1.4 GB = 33.6 GB. Through a 100 MB/s path that is 336 s of pure transfer, before
scheduling, which is why a two-minute rollout budget becomes a six-minute one.

**Check.** Time one pull on an idle node and multiply by the node count divided by the concurrency of pulls.
If the product exceeds the rollout budget, bandwidth is the constraint.

**Correction.** Multi-stage build, smaller base, and a pre-pull daemon. Cost: build complexity and a second
artifact to keep current.

**Prevention.** Image size is a CI check with a stated budget, and the rollout budget is derived from it.

### CI-18 — a drain that takes the service down

**Mechanism.** The PDB declares `minAvailable: 100%` to "protect availability". With two replicas and one
surge slot, the eviction API refuses every drain, and when an operator forces it the service goes to zero.

**Arithmetic.** With two replicas and `minAvailable: 1`, a drain can proceed in 50% steps and capacity never
drops below 6 req/s. With 100%, capacity drops to zero whenever the drain is forced, which is the opposite of
the intent.

**Check.** Read the PDB and the replica count together. A PDB equal to the replica count is a deadlock, not a
guarantee.

**Correction.** Set the PDB from the capacity floor: with a 40 req/s peak and 6 req/s per replica, the floor is
7 replicas, so 8 replicas with `minAvailable: 7` allows exactly one drain at a time.

**Prevention.** The PDB is derived from the capacity floor whenever the replica count changes.

### CI-19 — the mean that hides a tenth of the users

**Mechanism.** Latency is bimodal: 90% of requests complete at 80 ms and 10% at 2,000 ms. The arithmetic mean
is 272 ms, comfortably inside a 300 ms target, while one user in ten waits two seconds.

**Arithmetic.** 0.90 × 80 + 0.10 × 2,000 = 72 + 200 = 272 ms. Move 1% of traffic from the slow mode to the
fast mode and the mean falls to 252.8 ms while the p90 is unchanged — the mean responds to the tail's mass,
not to the tail's length.

**Check.** Plot the distribution. If two modes are visible, the mean is not a statistic about any user.

**Correction.** Put the SLO on a percentile with a deadline attached ("99% under 1,000 ms"), and measure it at
the client.

**Prevention.** Percentile-based SLOs at the boundary the user sees, with the deadline stated in the same
sentence.

### CI-20 — the green SLO and the support queue

**Mechanism.** The SLI is measured at the gateway, so time spent in DNS, TLS negotiation and client-side
retry backoff is excluded. A client that retries after 800 ms sees 1,600 ms while the gateway reports 700 ms.

**Arithmetic.** With one retry at an 800 ms timeout and a 50% retry rate on the slow path, mean observed
client latency exceeds gateway latency by 0.5 × 800 = 400 ms — larger than the entire gateway p99 for the fast
path.

**Check.** Compare gateway and client timings for one trace identifier. A constant gap points at retries; a
variable gap points at DNS or TLS.

**Correction.** Instrument the client and define the SLI there, accepting that client telemetry is sampled.

**Prevention.** The SLO document names its measurement point explicitly.

### CI-21 — the page that arrives late

**Mechanism.** A single burn-rate rule with a 6-hour window and a threshold of 2. A fast incident that burns
the whole budget in forty minutes never reaches the threshold long enough to page.

**Arithmetic.** C9: burn 14.4 for one hour consumes 2.0% of a 30-day budget; burn 2 for one hour consumes
0.278%. An incident at burn 14.4 lasting 40 minutes consumes 1.33% — under a 2% threshold evaluated over six
hours, so the page never fires while the incident is live.

**Check.** Recompute the consumed fraction at the observed burn and duration. If the incident consumes less
than the threshold within the window, the rule cannot see it.

**Correction.** Pair a long window with a low threshold (ticket) and a short window with a high threshold
(page), so both slow and fast burns are visible.

**Prevention.** Alert rules are reviewed against the budget arithmetic, not against intuition.

### CI-22 — the chargeback above the invoice

**Mechanism.** The report multiplies resource-hours by list price. The account is on a committed-use discount
and holds credits, so the sum of team charges exceeds the invoice by the discount plus the credit.

**Arithmetic.** If the commitment is 20% off list and credits cover 3% of the bill, the sum of list-priced
charges is 20,719.83 / (0.80 × 0.97) = $26,700.81, i.e. 28.87% above the actual $20,719.83. Teams cannot
reconcile a number that is larger than the bill.

**Check.** Sum one team's charges and compare with the invoice line that team can see. A stable ratio across
teams points at a pricing-basis mismatch, not at a metering error.

**Correction.** Amortise the commitment across the term and allocate credits explicitly, then assert that the
sum of team charges equals the invoice to within rounding.

**Prevention.** One pricing basis for every report, and a reconciliation test in the reporting job.

### CI-23 — one label, thirty times the monitoring bill

**Mechanism.** A `tenant` label is added to a request metric to make per-tenant dashboards possible. The label
has 200 values, the metric has 35 routes and 6 statuses, and the product is 42,000 series instead of 1,250.

**Arithmetic.** C10: 200 × 35 × 6 = 42,000 series for that one metric. At $0.192 per series-month that is
$8,064 against $40.32 for the same metric's 210 series before — a 200× increase on the metric. Across the whole
monitoring bill, total series go from 1,250 to 1,250 − 210 + 42,000 = 43,040, so the line grows from $240.00 to
43,040 × 0.192 = $8,263.68, i.e. 34.4×. The bill grows 34× from a change that passed review because it was one
line, and the growth is invisible in the diff that caused it.

**Check.** Compute the product from the label schema before deploying. If the product exceeds the cardinality
budget, the change is a cost decision.

**Correction.** Drop the high-domain label from the metric and use exemplars or logs for per-tenant analysis.

**Prevention.** A cardinality budget per metric, checked in CI against the label schema.

### CI-24 — the alert that fires on day two

**Mechanism.** The budget alert evaluates committed spend, which is recognised at the start of the month, so
it crosses 80% of the monthly budget while actual usage is still tiny.

**Arithmetic.** With a $20,000 monthly budget and a $18,000 annual commitment amortised at $1,500 per month,
committed spend is 7.5% of budget on day one. If the alert also counts the annualised remainder, it reads 90%
immediately and the team learns to ignore it.

**Check.** Read which quantity the alert expression evaluates. Committed, actual and forecast are three
different numbers and must be three different alerts.

**Correction.** Separate alerts: forecast-based for run-rate, actual-based for hard budget, commitment-based
for cash planning.

**Prevention.** Every alert names its quantity in the rule description, and the runbook says what to do for
each.

### CI-25 — cheaper per request, dearer per task

**Mechanism.** The denominator changes from resolved tasks to served requests. Retries inflate requests, so a
system that retries more looks cheaper per request while costing more per outcome.

**Arithmetic.** Take a 300 ms deadline as the definition of "resolved". At 8 replicas cost per 1k requests is
$0.65702 and the queue-time breach rate from C3 is 0.28678, so cost per 1k *resolved* is
0.65702/(1 − 0.28678) = $0.92031. At 10 replicas, $0.80517 with a breach rate of 0.04631 gives
0.80517/(1 − 0.04631) = $0.84426. The cheaper-per-request configuration is 9.0% dearer per resolved task, and
the ranking has flipped without a single price change.

**Check.** Recompute both ratios over the same window with the same failure count. If the two stories
disagree, one denominator is wrong.

**Correction.** Adopt cost per resolved task as the canonical figure and publish cost per request only as a
diagnostic.

**Prevention.** One canonical denominator, defined in the metric catalogue, and a dashboard that cannot be
built without it.

### CI-26 — the 30% saving that evaporates

**Mechanism.** The saving was computed with C1 at peak utilisation, applied to average volume, at list price,
excluding egress. At month end the discount expired and egress grew because the cheaper region is farther from
users.

**Arithmetic.** C1 at rho = 0.8333 gives $0.000178 per request; at rho = 0.25 it gives $0.000593. The
"30% saving" was measured in the first units and realised in the second, so a 3.33× discrepancy in the base
swamps a 30% claim.

**Check.** Recompute the model with realised utilisation, the discount end date, and egress in the numerator.
Each of the three can alone reverse the sign of the claim.

**Correction.** Quote cost per month at realised load, with the discount term explicit and egress included.

**Prevention.** The cost model is reviewed alongside the capacity plan, and the modelled bill is compared with
the actual bill within 2%.

### CI-27 — the log volume that tripled

**Mechanism.** A request-payload log line was added at info level for a debugging session and never removed.
At 2.5 KiB per request and 31.5 million requests the base volume is 75.19 GiB; payload logging at three times
that is 225.57 GiB.

**Arithmetic.** 225.57 − 75.19 = 150.38 GiB extra at $0.50/GiB = $75.18 per month, plus the retention cost of
whatever the pipeline indexes. The line itself is cheap; the retention is not.

**Check.** Histogram bytes per request by log level and by message. One message contributing 60% of bytes is
the finding.

**Correction.** Remove the line, or sample it at 1% behind a flag that expires.

**Prevention.** A log budget per service in CI, and a flag expiry date enforced by the config linter.

### CI-28 — the debug endpoint that leaks a credential

**Mechanism.** A diagnostic endpoint echoes the environment for troubleshooting, including the upstream API
key. It is unauthenticated because it was reachable only from a jump host when it was written.

**Arithmetic.** With 90-day rotation the exposure window is 90 days; with 365-day rotation it is a year. The
window, not the endpoint, decides the blast radius, and the window is usually a default nobody chose.

**Check.** Request the endpoint as an unauthenticated caller from outside the expected network. A 200
containing key-shaped strings is the finding.

**Correction.** Remove the endpoint, rotate the key, and check whether the key was used from an unexpected
address during the exposure window.

**Prevention.** Secret scanning in CI plus an inventory of endpoints that return configuration.

### CI-29 — tenancy enforced by a sentence in the prompt

**Mechanism.** The retrieval index is shared, and isolation is requested by instructing the model not to use
other tenants' documents. The instruction is a preference, not a boundary; a query that retrieves a foreign
document can surface it.

**Arithmetic.** With 200 tenants and a shared index of 2 million chunks, one tenant owns 10,000 chunks — 0.5% of
the corpus. If a query's top 8 results are drawn without a tenant filter, the expected number of foreign chunks
in that result set is 8 × (1 − 0.005) = 7.96, i.e. essentially all eight. The leak does not need to be
probable to be fatal: it needs to happen once, on one query, in one tenant's session, and the corpus share
that makes it unlikely is exactly what makes it invisible in a staging test with two tenants.

**Check.** Retrieve the same query as two tenants and diff document ids. Any intersection of ids is a breach,
regardless of whether the text differs.

**Correction.** Enforce the tenant filter in the data layer, before ranking, and assert that the returned id
set is a subset of the tenant's partition.

**Prevention.** A cross-tenant retrieval test in the release gate, with a failure that blocks the release.

### CI-30 — the autoscaler that reacts after the burst

**Mechanism.** The scale-up stabilisation window is 300 s and the burst is 90 s. The policy averages the burst
away, decides nothing, and by the time it reacts the burst is over. The next burst repeats the pattern.

**Arithmetic.** A 90 s burst at 4.5× average demand is 54 req/s for 90 s against 48 req/s of provisioned
capacity. Averaged over the 300 s stabilisation window the policy sees (54 × 90 + 12 × 210) / 300 = 24.6 req/s,
i.e. 24.6/48 = 51.25% utilisation — below the 70% target, so nothing scales. Meanwhile 54/48 = 1.125 means the
queue is unstable and the backlog grows at 6 req/s for 90 s, reaching 540 requests. After the burst it drains
at 48 − 12 = 36 req/s, taking 15 s, so about 54 × 90 + 12 × 15 = 5,040 requests observe queue waits that peak
near 540/48 = 11.25 s.

**Check.** Compare the burst duration with the stabilisation window and the cold-start time. If the burst is
shorter than the window, the policy cannot see it.

**Correction.** A shorter scale-up window plus a pre-provisioned floor sized to the observed burst profile.

**Prevention.** The policy is tuned against a recorded burst profile, and the floor is derived from the
capacity arithmetic rather than from cost alone.

## 9. Complexity & Cost Analysis

**Decision cost on the request path.** An in-process token bucket is a few dozen nanoseconds and no
allocation. A shared ledger adds a network round trip: at 0.4 ms p50 and roughly 3 ms p99, that round trip is
added to *every* request, which on a 180 ms mean response is a 0.2% mean cost and a 1.7% p99 cost. On a
40 req/s peak it is also 40 additional round trips per second against the ledger, which is why the ledger must
be horizontally scaled and why leases exist.

**Complexity of the ledger itself.** A single atomic counter per scope is O(1) per decision but serialises at
the counter. Sharding by scope keeps O(1) and removes the hot spot; hierarchical quotas add O(depth) lookups
and a new failure mode (parent/child skew). Leases move cost off the request path entirely and replace it with
a background renewal rate of `replicas × lease_ttl⁻¹` operations per second — eight replicas renewing every
60 s is 0.13 decisions per second, which is free.

**Cardinality, the only cost that multiplies.**

| Label set | Series | Monthly cost at $0.192 | Comment |
|---|---|---|---|
| route, status | 35 × 6 = 210 | $40.32 | the pre-change baseline |
| + tenant (200) | 42,000 | $8,064.00 | one label, 200× on the metric |
| + tenant, aggregated to 20 buckets | 20 × 35 × 6 = 4,200 | $806.40 | bucketing recovers 90% of the analysis |
| + region (4) on the bucketed set | 16,800 | $3,225.60 | the product punishes every further label |

**Log volume, costed by the byte.** Base ingestion is 2.5 KiB per request; at 31,536,000 requests that is
75.19 GiB and $37.59 at $0.50/GiB. Payload logging triples it to 225.57 GiB and $112.79. The ratio is exact
because the price is linear; the surprise is always retention, where indexed storage costs an order of
magnitude more than ingestion and is not in the request-path budget at all.

**The cost of a latency target, in closed form.** From C3, the marginal price of tail latency is
`Δcost / Δt_99`, and it *rises* as rho approaches 1, because t_99 grows without bound while each replica's
price is fixed. On the reference platform the first 343 ms cost $4,672 (i.e. $13.63/ms) and the next 85 ms cost
the same money ($55.29/ms). A team that asks for "another 100 ms" without stating which increment it is asking
for is asking for either a cheap change or a 4.06× more expensive one. The series is not a law: it is a
consequence of the convexity of rho/(1−rho), and a system whose service time falls under batching can invert it.

**The cost of isolation.** Moving from a shared index to per-tenant partitions multiplies index count by the
tenant count but does not multiply total document volume. The cost is operational: 200 small indexes,
200 rebuild jobs, and a routing layer. At 200 tenants the fixed per-index overhead dominates, which is why
the usual answer is a filter enforced in the data layer plus a partition for the largest tenants, not 200
partitions for everyone.

**Cost of verification, which is the cheapest line in the budget.** Every check in Section 8 is a query, a
replay or a config read. The most expensive is a fault injection that costs a few replica-hours; the cheapest
is summing configured ceilings, which costs nothing and settles CI-06 outright. A team that runs the cheap
checks monthly will not need the expensive ones often.

## 10. Key Theorems / Results Reference

| Result | Statement | Use in this file | Where it fails |
|---|---|---|---|
| Little's law | L = λW in steady state | C11 drain sizing, in-flight counts | Non-stationary bursts, per-replica subsets |
| M/M/1 waiting tail | P(Wq > t) = ρe^{−(μ−λ)t} | C3 latency provisioning | Batch service, per-tenant queues, correlated arrivals |
| Pollaczek–Khinchine | Wq = ρ/(1−ρ) · (1+Cs²)/2 · Ts | generalising C3 to variable service | No closed form for networks of queues |
| Conservation of credits | Σ held + free = B per scope | C4 admission invariant | When the resource ignores the fence token |
| Geometric retry series | q = 1/(1−a) | C5 amplification | Correlated failures make every attempt fail together |
| Token-bucket bound | admitted(t) ≤ b + r·t | C8 burst policy | Multiple keys per logical client |
| Burn-rate identity | burn × T / W = budget fraction | C9 alerting | Windows shorter than the scrape interval |
| Unit-cost identity | u = p_h/(3600·s·ρ) | C1, C2, CI-26 | Mixed pricing, spot interruption, varying service time |
| Cardinality product | N = Π |values(label i)| | C10, CI-23 | Sparse combinations, histogram buckets |
| Amdahl-style marginal cost | Δcost/Δt → ∞ as ρ → 1 | Section 9 | Not a theorem: an empirical shape in this system |
| PASTA | Poisson arrivals see time averages | justifies measuring at arrival | Non-Poisson traffic, closed-loop clients |
| At-least-once + idempotent sink | effectively-once within the dedupe window | CI-01, CI-02 | Window shorter than the retry horizon |

**Note on the marginal-cost row.** It is included because practitioners cite it as a theorem. It is not one.
The divergence of marginal cost as utilisation approaches one is a property of the queueing model, and it
holds for any system whose wait is convex in load; the *rate* at which it diverges is system-specific.

## 11. Interview / Exam Questions

### Q1. A platform bills $20,719.83 for 31,536,000 requests with 8 replicas at $3.20/h. What is the accelerator cost per 1,000 requests, and what does changing the replica count do to it?

**Answer.** Accelerators cost 8 × 3.20 × 730 = $18,688, so 18,688 / 31,536 = $0.5926 per 1,000. Changing the
replica count alone does not change it: both the numerator and the capacity scale with replicas, so C1 reduces
to 3.20/(3600 × 6 × ρ). What changes it is utilisation — which changes when you add replicas without adding
demand.

**Reasoning.** The cancellation is the whole insight. A candidate who says "more replicas means more cost per
request" has missed that utilisation absorbs it; a candidate who says "replicas never matter" has missed that
they change ρ.

### Q2. Compute p99 for 8 and 10 replicas at 40 req/s, s = 6. Which increment would you buy, and what does the second one cost per millisecond?

**Answer.** μ = 48 gives ρ = 0.8333 and t_99 = ln(83.333)/8 + 1/6 = 0.5529 + 0.1667 = 0.7195 s. μ = 60 gives
ρ = 0.6667, t_99 = ln(66.667)/20 + 0.1667 = 0.2100 + 0.1667 = 0.3767 s. The first increment buys 343 ms for
$4,672, i.e. $13.63/ms. The second, to 12 replicas, gives ρ = 0.5556, t_99 = ln(55.556)/32 + 0.1667 = 0.2922 s,
so 84.5 ms for the same $4,672, i.e. $55.29/ms.

**Reasoning.** Both increments are priced identically in dollars and differ 4.1× in what they buy. The
discriminator is the convexity of ρ/(1−ρ), not any vendor's pricing.

### Q3. Three replicas each admit 40 credits against a global budget of 100. What invariant fails, by how much, and what is the one-line check?

**Answer.** Conservation fails: Σ held = 120 against B = 100, an overshoot of 20 credits or 20%. The check is
to sum the configured ceilings across replicas and compare with the budget; no single replica can see the
violation, which is why the check must be aggregate.

**Reasoning.** The failure is architectural, not numerical: there is no serialisation point. A candidate who
proposes "lower each ceiling to 33" has replaced an invariant with a rounding hope, because 3 × 33 = 99 works
only while every replica is alive and evenly loaded.

### Q4. A token bucket is 600/min with a 2,400 burst. A tenant sends 3,000 in the first second. How many are rejected, and what happens at a sustained 900/min?

**Answer.** r = 10/s, so the bucket admits its full 2,400 immediately and rejects 600 — 20% rejection. At a
sustained 900/min against 600/min, rejection is (900 − 600)/900 = 33.3%, and the burst allowance delays but
cannot prevent it.

**Reasoning.** The burst is a policy number: 2,400 credits at 10/s is four minutes of average traffic. The
candidate should notice that the rejection rate depends on the observation window, and that the first-second
figure (20%) understates the sustained figure (33.3%).

### Q5. Retries: a = 0.35, four attempts. What is the amplification, and what happens in a total outage with three attempts at 40 req/s?

**Answer.** q_4 = 1 + 0.35 + 0.1225 + 0.042875 = 1.515375. The infinite limit is 1/(1−0.35) = 1.538462, so four
attempts captures 1.515375/1.538462 = 98.5% of it. A total outage with three attempts sends 40 × 3 = 120 req/s
at 48 req/s of capacity.

**Reasoning.** The important number is the last one: the retry policy must be evaluated against capacity, not
against the success rate. 120/48 = 2.5× oversubscription is the mechanism by which a recovery is prevented.

### Q6. A 99.9% SLO over 30 days: how much budget is that, and what does a burn rate of 14.4 for one hour consume?

**Answer.** 30 days is 43,200 minutes; the budget is 0.1% = 43.2 minutes. Burn 14.4 for one hour consumes
14.4 × 1/720 = 2.0% of the budget, i.e. 0.864 minutes.

**Reasoning.** The 720-hour denominator is the SLO window, and mixing it with a 24-hour window is the standard
error. A candidate who computes 14.4 hours of budget has used a 30-hour window.

### Q7. Cache TTL 300 s, 40 req/s, 18% of entries change authorisation inside the TTL. How many responses are exposed, and how do you fix it without destroying the hit rate?

**Answer.** E = 40 × 300 × 0.18 = 2,160 responses. Fix with an authorisation epoch in the key: exposure goes to
zero on invalidation, and the hit rate drops once per permission change rather than continuously, which is a
far smaller cost than reducing TTL to 30 s (which multiplies origin load tenfold).

**Reasoning.** TTL is a time bound and correctness needs a truth bound. The trade is between a continuous hit-
rate penalty and a discrete invalidation event.

### Q8. Adding a `tenant` label with 200 values to a metric that already has 35 routes and 6 statuses: what happens to cost at $0.192/series-month?

**Answer.** 200 × 35 × 6 = 42,000 series, or $8,064 per month for that metric, against 210 series and $40.32
before — a 200× increase on the metric and a 34.4× increase on the whole line (1,250 → 43,040 series, $240.00 →
$8,263.68). Bucketing tenants to 20 values gives 20 × 35 × 6 = 4,200 series and $806.40, recovering 90% of the
per-tenant analysis for 10% of the cost.

**Reasoning.** Cardinality is multiplicative in label domains, so the review question is not "is this label
useful" but "what is the product".

### Q9. Mean latency is 272 ms with 90% at 80 ms and 10% at 2,000 ms. A 300 ms mean SLO is green. What is wrong?

**Answer.** The mean is inside the target and one user in ten waits two seconds. The distribution is bimodal,
so the mean is not a statistic about any user; the correct SLI is a percentile with a deadline, e.g. "99% under
1,000 ms".

**Reasoning.** Compute the mean: 0.9 × 80 + 0.1 × 2,000 = 272. Then note that shifting 1% of traffic from the
slow mode to the fast mode moves the mean by 19.2 ms while the p90 does not move at all — the mean tracks mass,
not length.

### Q10. A rollout drains 8 replicas at 1,200 req/s aggregate with 50 ms in flight, and 4% of in-flight requests are not retried. How many requests does the rollout lose?

**Answer.** L = 1,200 × 0.05 = 60 in flight per instant; 4% of 60 = 2.4 per drain; across 8 replicas about 19.

**Reasoning.** Little's law gives the in-flight count, and the drain policy decides how many of those are
retried. The number is small enough to be invisible in a canary and large enough to generate tickets.

### Q11. `minAvailable: 100%` on a two-replica deployment. What happens on a node drain, and what should the value be?

**Answer.** Every eviction is refused, so the drain either stalls or is forced and takes the service to zero.
With a 40 req/s peak and 6 req/s per replica the capacity floor is 7 replicas, so an 8-replica deployment
should use `minAvailable: 7` — allowing exactly one drain at a time.

**Reasoning.** The PDB is a capacity statement expressed as a count. Writing 100% encodes "never", not
"always available".

### Q12. A bill is $20,719.83 with 6.5% unattributed and a 45/30/25 split. What does team A pay, and what changes if the remainder is spread pro rata?

**Answer.** Attributed spend is 20,719.83 × 0.935 = $19,373.04, so A pays 0.45 × 19,373.04 = $8,717.87. Spread
pro rata, A pays 8,717.87 + 0.45 × 1,346.79 = $9,323.93 — $606.06 more, for a gap somebody else created.

**Reasoning.** The policy choice has a price and an incentive effect: pro-rata spreading punishes teams for
others' missing tags, which reduces the incentive to tag.

### Q13. Configuration A costs $0.6570 per 1k requests with 2.1% failures; configuration B costs $0.80517 with 0.4% failures. Which is cheaper, and by what measure?

**Answer.** Per request A is cheaper by 18.4%. Per resolved task, A is 0.65702/0.979 = $0.67112 and B is
0.80517/0.996 = $0.80840, so A is cheaper there too — by 17.0%. The decision is not cost: B costs $4,672 per
month more and converts 1.7 pp of 31,536 thousand requests, i.e. 536.1 thousand additional successes, so the
marginal price is 4,672/536.1 = $8.71 per 1,000 additional resolved tasks.

**Reasoning.** The two ratios move in the same direction here, which is what makes this a judgement question
rather than a trap: the honest framing is cost per *additional* success, not cost per unit, and $8.71 per
thousand is the price of converting a failure into a success by over-provisioning. Below some failure rate that
price stops being worth paying, and the question does not tell you where.

### Q14. A team claims moving to spot saves 30%. What three numbers do you demand before believing it?

**Answer.** The realised utilisation that the saving is computed at (a peak-based figure overstates savings by
up to 3.33× per C1/C2), the interruption rate and the recovery time (which convert a price saving into lost
capacity), and the list-price-versus-amortised-commitment basis (which decides whether 30% is measured against
the bill or against a number the bill never contained).

**Reasoning.** Each of the three can reverse the sign of the claim on its own. On the reference platform 60% off
eight accelerators is 0.60 × 18,688 = $11,212.80 per month, i.e. 54.1% of the whole bill — so the headline
number is plausible and still wrong three ways. First, interruption: at 2.1% per hour across 8 replicas that is
8 × 0.021 × 730 = 122.6 events per month, each removing 6 req/s for the 9.4 s warm-up, i.e. 56.4 requests and
6,917 requests per month (0.022% of volume) — small, but only because a replacement is already warm. Second,
the spare: absorbing 122.6 events means running a ninth replica at on-demand price, $2,336, so the real saving
is 0.40 × 8 × 2,336 = $7,475.20 plus $2,336 spare = $9,811.20 of spend against $18,688, a 47.5% saving, not
60%. Third, the basis: if the 30% was computed at peak utilisation per C1, it is applied to a base 3.33×
smaller than the realised one. A 47.5% saving measured on the wrong denominator can present as 30% or as 90%.

## 12. Quick Reference Card

| If you see | Suspect first | Run this check | Then price the fix |
|---|---|---|---|
| Duplicate side effect after a retry | Idempotency key scope | Replay one intent, count sink rows | One unique index, one replay path |
| Duplicate arriving hours later | Dedupe window < retry horizon | Compare the two numbers | Storage linear in window length |
| Rejections counted as served | Outcome measured at the caller | Served count vs sink rows | One metric moved to the sink |
| Aggregate rate above the contract | Per-replica buckets | Sum configured ceilings | One ledger, or divide by replicas |
| Budget overshoot at renewal | Grant before revoke, no fence | Sum held across one renewal | Fenced replace instead of add |
| Spend spike at lease expiry | Expiry forgets instead of returning | Correlate admissions with expiries | Generation counter on reuse |
| Outage that will not recover | Retry amplification | Compute q = 1/(1−a) vs capacity | Retry ratio plus jitter |
| p99 breach at low mean | Provisioning, not code | C3 at the realised ρ | $13.63/ms first, $55.29/ms after |
| Cold replica latency | Readiness before warm | Probe time vs warm time | 9.4 s per replica of rollout time |
| A handful of lost requests per deploy | Drain window shorter than W | In-flight count vs retries | Drain delay derived from W |
| Monitoring bill ×30 | Cardinality product | Π label domains | Bucket the high-domain label |
| Chargeback above the invoice | List price vs amortised commitment | Reconcile one team's sum | One pricing basis, explicit credits |
| Cheaper per request, dearer per outcome | Denominator changed | Both ratios, same window | Adopt cost per resolved task |
