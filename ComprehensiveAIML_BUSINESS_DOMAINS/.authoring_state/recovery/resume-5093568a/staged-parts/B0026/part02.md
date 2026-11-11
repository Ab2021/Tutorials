## 4. Core Formulas with Full Derivation

Every result below is stated in the five-part anatomy: the formula, what each symbol ranges over, the
precondition that makes the algebra valid, a numeric substitution, and the decision the number
supports.
Workload constants are ILLUSTRATIVE; the derivations are DERIVED and the arithmetic is VERIFIED
against the substitutions shown.

### 4.1 Base case

The base case is one intent, one attempt, no concurrency, and a single sink.
It establishes the admission predicate and the meaning of a duplicate.

**Formula 1 - the admission predicate.**

```text
FORMULA [T-01]: accept = auth(p, r, t) AND shape(x) AND meaning(x, s) AND claim(k, h(x)) = NEW

WHERE  p      principal identity, one tenant
       r      resource set resolved by the HOST, never copied from arguments
       t      authority evaluation time [s epoch]
       x      typed argument object validated against pinned schema version
       s      business state snapshot read at admission (balance, order status, quota)
       k      operation key: one intended business effect
       h(x)   digest over the canonical payload
       claim  atomic compare-and-insert into the operation ledger
```

CONSTRAINT - the four conjuncts must be independent and the fourth must be atomic.
If `claim` is implemented as an upsert, an existing row is overwritten and the predicate always
returns NEW, which silently converts every retry into a second effect.
If `r` is taken from the argument object, a caller can name a resource it was never granted.
If `s` is read outside the same transaction as the effect, the check can pass on a balance that the
executor has already spent.

NUMERICAL EXAMPLE.
Principal `p=agent_support_07`, tool `payments.refund`, key `rf_8841_a`.
`auth=1` (role support_agent holds `refund` on merchant M-77), `shape=1`, `meaning=1` (order 8841 is
`SETTLED`, refundable amount 4500 minor units available), and `claim` returns `EXISTING` in state
`COMMITTED` with receipt `RCP-55120`.
Then `accept = 1 AND 1 AND 1 AND 0 = 0`.
The correct action is to return receipt `RCP-55120` to the caller, not to dispatch again and not to
mint key `rf_8841_b`.

INTERPRETATION.
The predicate separates four questions that are often collapsed into "is the call valid?".
Shape failures are client bugs, meaning failures are stale-state bugs, authority failures are access
bugs, and claim failures are duplicate-suppression working as designed.
Only the last should return the original receipt; the first three must return typed errors.

**Formula 2 - expected attempts per intent under probabilistic retry.**

```text
FORMULA [T-02]: A = 1 / (1 - rho)

WHERE  A    expected number of transport attempts for one intent, dimensionless
       rho  probability that any given attempt is retried, [0, 1)
       precondition: retries are independent and bounded in number by policy
```

Derivation: the count of attempts is geometric with success probability `1 - rho`, and the mean of a
geometric variable on {1,2,...} is `1/(1-rho)`.
The series is `1 + rho + rho^2 + ...`, which converges only for `rho < 1`; at `rho = 1` the intent
never terminates and the formula is undefined rather than infinite by accident.

CONSTRAINT - `rho` must be the *retry* probability, not the error rate.
If 8 percent of attempts fail and half of those failures are non-retryable (validation errors,
permanent rejections), the effective `rho` is 0.04, not 0.08, and using 0.08 double-counts.

NUMERICAL EXAMPLE.
`rho = 0.08` gives `A = 1/0.92 = 1.08696` attempts per intent.
For `N = 2160` intents (18 intents/s for 120 s), total attempts are `2160 x 1.08696 = 2347.8`.
Attempts above one are `2347.8 - 2160 = 187.8`, which is exactly `N x rho/(1-rho) = 2160 x 0.08696`.
Each surplus attempt is a candidate duplicate effect if the ledger is absent.

INTERPRETATION.
Duplicate pressure scales with `rho/(1-rho)`, so it grows faster than linearly as reliability
degrades: at `rho = 0.30` the surplus fraction is 0.4286, five times the value at `rho = 0.08`.
This is why a ledger is mandatory long before a system looks unreliable.

**Formula 3 - residual duplicate effects given partial duplicate detection.**

```text
FORMULA [T-03]: U = N * (rho / (1 - rho)) * (1 - d)

WHERE  U  expected undetected duplicate effects, count
       d  probability that a duplicate attempt is suppressed when it occurs, [0,1]
       N  distinct intents in the window
```

CONSTRAINT - `d` is a measured property of the ledger path, not a design intention.
It degrades when the ledger write and the sink commit are not ordered, when keys are minted inside
the retry loop, and when a fallback path bypasses the ledger under load.

NUMERICAL EXAMPLE.
With `N = 2160`, `rho = 0.08`, `d = 0.94`: surplus attempts are 187.8 and undetected duplicates are
`187.8 x 0.06 = 11.27`.
At an average refund of 4250 minor units (42.50 USD), the exposure is `11.27 x 42.50 = 478.96 USD`
for a two-minute burst.
At `d = 0.75` the same burst yields `187.8 x 0.25 = 46.96` duplicates and `1995.65 USD`.

INTERPRETATION.
A "94 percent effective" ledger sounds adequate and costs roughly 500 USD per burst at this volume.
The decision-relevant form of the statement is the product `N x rho/(1-rho) x (1-d) x unit_value`,
because that is the number a finance owner can price.

### 4.2 General case

The general case adds finite service rate, concurrent clients, fanout, and multiple sinks.
Two of the results below are queueing identities; two are probabilistic bounds on identifier design.

**Formula 4 - backlog and drain time.**

```text
FORMULA [T-04]: B(t) = max(0, lambda - mu) * t ;  T_drain = B / (mu_r - lambda_r)

WHERE  lambda   arrival rate of intents [intents/s]
       mu       completion rate of the executor [intents/s]
       t        overload duration [s]
       B        backlog at the end of overload [intents]
       mu_r     completion rate while draining [intents/s]
       lambda_r arrival rate while draining [intents/s]
       T_drain  time to return to zero backlog [s]
```

Derivation: arrivals minus completions integrated over the interval, floored at zero because a queue
cannot hold negative work.
Drain time follows from the same conservation law applied to the recovery lane.

CONSTRAINT - constant rates, single queue, no priority classes, and `mu_r > lambda_r`.
The formula gives the time to *empty* the queue, not the time for the oldest item to be served;
with FIFO it is also the latency of the last-queued item, which is why tail latency explodes during
drain even though throughput looks healthy.

NUMERICAL EXAMPLE.
`lambda = 18`, `mu = 15`, `t = 120` gives `B = (18-15) x 120 = 360` intents.
Recovery runs at `mu_r = 21` while live arrivals remain `lambda_r = 15`, so surplus is 6 intents/s and
`T_drain = 360/6 = 60` s.
Independent check: during those 60 s the lane completes `21 x 60 = 1260`, of which `15 x 60 = 900`
serve live arrivals and 360 serve the backlog - the arithmetic closes exactly.

INTERPRETATION.
Drain time is governed by the *surplus*, not the capacity: doubling `mu_r` to 30 cuts drain to 24 s,
whereas raising it to 22 only reaches 51.4 s.
Capacity purchased without surplus is invisible to recovery.

**Formula 5 - identifier collision expectation.**

```text
FORMULA [T-05]: E[collisions] ~= n^2 / 2^(b+1)

WHERE  n  number of distinct keys issued over the retention window
       b  effective random bits in the key
       E  expected number of colliding key pairs (Poisson approximation)
```

Derivation: the birthday approximation counts pairs `n(n-1)/2 ~= n^2/2` and multiplies by the
per-pair collision probability `2^-b`, giving `n^2 / 2^(b+1)`.

CONSTRAINT - keys must be drawn uniformly and independently, and the approximation holds for
`n << 2^(b/2)`.
Sequential integers have `b = 0` in this model: they never collide, but they are guessable, so
collision bounds are the wrong tool for them and predictability analysis is the right one.
Truncated UUIDs, base-36 timestamps, and hash-prefix keys are where this formula earns its keep.

NUMERICAL EXAMPLE.
`n = 31,104,000` keys (4 intents/s for 90 days).
At `b = 48`: `E ~= 9.674e14 / 5.63e14 = 1.72` expected collisions - unusable, since a colliding key
rejects a legitimate new intent as a duplicate.
At `b = 40`: `E ~= 439.9`.
At `b = 64`: `E ~= 2.62e-5`, i.e. one collision per 38,000 such retention windows.

INTERPRETATION.
Forty-eight bits of a truncated digest is the classic mistake: it looks long, and it fails at a
30-million-row ledger.
Use at least 64 random bits, and treat a collision as a *rejection* rather than silently proceeding,
so the failure is loud instead of a missed payment.

**Formula 6 - ledger storage sizing.**

```text
FORMULA [T-06]: S = lambda_avg * 86400 * D * b_row / 2^30

WHERE  lambda_avg  mean admitted intents per second [intents/s]
       D          retention in days [day]
       b_row      stored bytes per ledger row [byte]
       S          steady-state ledger size [GiB]
```

CONSTRAINT - retention must be at least the maximum retry horizon plus clock skew, otherwise the
ledger forgets a key while a retry is still in flight.
`b_row` must include indexes; a 512-byte row with a primary key, a digest, a state enum, and two
timestamps is realistic for a B-tree with a secondary index on state.

NUMERICAL EXAMPLE.
`lambda_avg = 4`, `D = 90`, `b_row = 512`: rows are `4 x 86400 x 90 = 31,104,000`, bytes are
`31,104,000 x 512 = 1.593e10`, and `S = 14.83` GiB.
At an ILLUSTRATIVE 0.023 USD per GiB-month the ledger costs 0.34 USD per month - negligible against
the 478.96 USD exposure of a single uncorrected burst.

INTERPRETATION.
Retention arguments are usually settled by this ratio: the storage line is three orders of magnitude
smaller than the duplicate-effect exposure, so the correct default is to keep keys longer than
anyone asks for.

### 4.3 Variants

The variants below change one structural assumption at a time: batch atomicity, fanout, digest
coverage, compensation, approval lifetime, and federated keyspaces.

**Variant A - batch with per-item failure.**

```text
FORMULA [T-07]: E[committed] = m * (1 - q)^m  ;  E[partial] = m - E[committed]

WHERE  m  items in the batch
       q  per-item failure probability, independent
       E[committed]  expected items committed by an all-or-nothing retry policy
```

CONSTRAINT - independent item failures and a retry policy that discards the whole batch on any
failure. Both are wrong in production and the formula shows the cost of the second.

NUMERICAL EXAMPLE.
`m = 20`, `q = 0.05`: success probability is `0.95^20 = 0.3585`, so the batch commits intact only
about 36 percent of the time and 12.83 items are wasted per attempt on average.
Splitting into 20 single-item calls with individual keys raises delivered items to
`20 x 0.95 = 19` per attempt.

INTERPRETATION.
Atomic batches are efficient when items share one semantic intent (a single journal entry with
several lines) and wasteful when they are independent (twenty unrelated emails).
The identifier that decides this is the key's scope: one key per batch means all-or-nothing.

**Variant B - fanout blast radius.**

```text
FORMULA [T-08]: R = lambda * t * f

WHERE  R  external records altered by one undetected defect
       f  records per intent (fanout), records/intent
       lambda, t as in T-04
```

CONSTRAINT - fanout is uniform; if a small fraction of intents fan out enormously (a bulk update
tool), `R` is dominated by the maximum, not the mean, and the mean is the wrong statistic.

NUMERICAL EXAMPLE.
`lambda = 18`, `t = 120`, `f = 2`: `R = 4320` records in two minutes.
With one intent in a thousand fanning out to 10,000 labels, the expected extra is
`2160 x 0.001 x 10000 = 21,600` records from a single call.

INTERPRETATION.
Bound `f` in the schema (maximum array length, maximum resolved resource count) rather than in the
prompt; a schema bound is enforced by the executor, a prompt bound is a suggestion.

**Variant C - digest coverage.**

```text
FORMULA [T-09]: P(detect one changed field) = c  ;  P(detect all) = c^k

WHERE  c  fraction of effect-relevant fields included in the digest, [0,1]
       k  number of effect-relevant fields
```

CONSTRAINT - each field independently included, and exactly one field changed per retry.
Multiple simultaneous changes are not additive: the probability that *some* change is undetected is
`1 - c^n` for `n` changed fields, which is worse than the single-field case.

NUMERICAL EXAMPLE.
A refund has `k = 5` effect-relevant fields (order, amount, currency, destination, instrument) and
the digest covers 4 of them, so `c = 0.8`.
`c^5 = 0.3277`: only a third of payload substitutions are caught.
If the omitted field is the destination account, every retry that redirects funds is invisible.

INTERPRETATION.
Digest coverage is a test you can write: mutate each effect-relevant field in turn and assert the
ledger rejects the call with `CONFLICT`.
Coverage below 1.0 is a defect, not a configuration choice.

**Variant D - compensation cost.**

```text
FORMULA [T-10]: E[cost] = c_e + q_c * c_c

WHERE  c_e  cost of the original effect [currency]
       q_c  probability the effect must be compensated, [0,1]
       c_c  cost of compensation, including fees and human time [currency]
```

CONSTRAINT - compensation is *not* a rollback: it must be admitted, authorized, and keyed like any
other write, and it may itself fail.

NUMERICAL EXAMPLE.
An original effect costs 2.00 USD in fees. With `q_c = 0.04` and `c_c = 25.00` USD (a 2.00
non-refundable fee plus 6 minutes of agent time at 230 USD/hour = 23.00), expected cost is
`2.00 + 0.04 x 25.00 = 3.00` USD.
Reconciliation instead of compensation, at 0.15 USD per sink query and the same 0.04 rate, is
`2.00 + 0.006 = 2.006` USD.

INTERPRETATION.
Design the reconciliation path first; compensation is the fallback for effects the sink refuses to
reverse, and it costs an order of magnitude more per event.

**Variant E - approval freshness.**

```text
FORMULA [T-11]: P(stale approval used) = 1 - exp(-gamma * T_tail)

WHERE  gamma   rate at which the approved payload would change if re-derived [1/s]
       T_tail  delay between approval and dispatch [s]
```

CONSTRAINT - `gamma` constant over the window and the payload change independent of the approval.
In practice `gamma` spikes at period boundaries (price updates, quota resets), so a single `gamma`
understates risk on those boundaries.

NUMERICAL EXAMPLE.
`gamma = 2e-4` per second (a price or stock level that changes about once per 83 minutes) and
`T_tail = 900` s: `P = 1 - exp(-0.18) = 0.1648`, so roughly one in six executions runs on an
approval whose payload no longer matches the current state.
At `T_tail = 60` s the same `gamma` gives `1 - exp(-0.012) = 0.0119`.

INTERPRETATION.
Approval TTL should be tied to the *dispatch* delay distribution (p99, not median) and the payload
must be re-checked against the digest at dispatch; a mismatched digest escalates rather than
re-approving silently.

**Variant F - federated keyspace.**

```text
FORMULA [T-12]: uniqueness domain = (server_id, tool_name, operation_key)

WHERE  server_id  identity of the tool server (with certificate or pinned key)
       tool_name  declared name, not the model-visible alias
       operation_key  key minted by the intent owner
```

CONSTRAINT - at least one component is trusted and stable; if the client mints keys without a server
identity, two servers offering the same tool name share a namespace.

NUMERICAL EXAMPLE.
Ten servers each expose `create_ticket`; with a two-part key `(tool, key)` and 500 intents/s across
them, the probability that a random key from server A collides with one from server B within a
31-million-row ledger is the T-05 value at 64 bits, `2.62e-5` - small, but the consequence is a
rejected legitimate ticket, and the fix is free.

INTERPRETATION.
Include the server identity in the uniqueness constraint from day one; retrofitting it means a
migration that must preserve in-flight keys, which is exactly when mistakes are expensive.

### 4.4 Limits and counterexamples

**Limit 1 - exactly-once across an opaque sink.**
No client-side ledger can guarantee exactly-once delivery when the sink does not support a
deduplication key: the client cannot distinguish "committed, response lost" from "never arrived".
The strongest achievable property is exactly-once *effect* where the sink honours a key, and
at-least-once *attempt* with reconciliation everywhere else.
Stated as a rule: if the downstream API has no idempotency key and no query-by-your-key, then every
retry is a bet, and the correct design is a hold queue with a human or scheduled reconciler.

**Limit 2 - the queue formula at saturation.**
T-04 assumes `mu_r > lambda_r`.
At `lambda_r = 21` and `mu_r = 21`, the surplus is zero and drain time diverges; the backlog is
absorbed only by shedding, prioritising, or adding capacity.
A reviewer who reports "drain in 60 s" while live arrivals equal recovery capacity has used the
formula outside its domain.

**Limit 3 - independence in T-07.**
Per-item failures correlate strongly in practice: one shard outage fails every item targeting that
shard, so `(1-q)^m` understates batch failure probability.
The counterexample is a batch of 20 items all on the same shard with `q = 0.05` at the shard level:
success probability is 0.95, not 0.3585.

**Limit 4 - digest coverage and semantic equivalence.**
Two payloads that differ textually but mean the same thing (amount `45` versus amount `45.00`,
`USD` versus `usd`, ISO timestamp versus epoch) produce different digests and are rejected as
conflicts.
Canonicalisation must normalise numeric scale, currency case, and timestamp format before hashing;
otherwise the ledger is correct and the operator experience is a stream of false conflicts.

**Limit 5 - the meaning check reads a stale snapshot.**
`meaning(x, s)` is only as fresh as `s`.
Under a burst, the snapshot may be 400 ms old while another executor spends the same budget in that
window; the predicate then admits two effects that individually looked affordable.
The counterexample is a 1000-unit balance with two concurrent 700-unit debits: both pass against the
snapshot, and the invariant is broken by concurrency rather than by logic.
The fix is a conditional write at the sink (`debit where balance >= amount`), not a better snapshot.

**Limit 6 - approval freshness with correlated change.**
Variant E treats payload drift as a Poisson process.
For scheduled events - a price list reload at 00:00, a quota reset on the first of the month - the
rate is a spike, and any TTL that looks safe on average will fire on the boundary.
The discriminating check is to plot stale-approval incidents against time of day; clustering at
boundaries falsifies the Poisson assumption.

**Takeaway.**
Each formula carries a domain; three of the six limits are about concurrency and sink semantics
rather than algebra, which is where real systems diverge from their models.
