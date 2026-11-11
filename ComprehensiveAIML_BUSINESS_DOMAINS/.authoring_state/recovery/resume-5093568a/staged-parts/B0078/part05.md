## 9. Complexity & Cost Analysis

### 9.1 Cost model with all inputs stated

Take an ILLUSTRATIVE platform with these parameters, all chosen to be round and plausible rather
than measured:

```text
N  = 25,000,000 entities          (cards)
V  = 6 feature views              (spend, count, velocity, merchant-diversity, tenor, risk-band)
B  = 120 bytes per entity per view   (8 features x 12 bytes + key + version + timestamp)
E  = 1,400,000 events/day
W  = 168 h window
R  = 2 online replicas
P_mem = 0.05 USD per GB-hour of provisioned memory   (ILLUSTRATIVE)
```

### 9.2 Online footprint

```text
raw values      = N * B * V = 25e6 * 120 * 6 = 18.0e9 bytes = 18.0 GB
with 2 replicas = 36.0 GB
add 30% for per-key overhead, allocator slack, and index:
                36.0 * 1.30 = 46.8 GB
monthly memory  = 46.8 GB * 0.05 USD/GB-h * 730 h = 1,708 USD   (ILLUSTRATIVE)

halving V to 3:  25e6 * 120 * 3 = 9.0 GB raw, 23.4 GB provisioned, 854 USD/month
```

The lesson is not the absolute number but the 2x: view count is the dominant lever, and it is a
design decision made once and rarely revisited. Every "just one more feature view" is a recurring
monthly cost plus a permanent increment to read fan-out.

### 9.3 Write throughput

```text
average = E * V / 86,400 = 1.4e6 * 6 / 86,400 = 97.2 writes/s
peak    = 3x average (ILLUSTRATIVE diurnal factor) = 291.7 writes/s
```

Under 300 writes/s is a single-node workload; the interesting part is not the rate but the
*ordering*: writes for one entity must be applied in a way that a stale update cannot overwrite a
newer one. That is a versioning problem (compare-and-set on `t_i`, or last-write-wins with a
monotonic stamp), and it is where the real engineering is at even trivial rates.

### 9.4 History and rebuild cost

```text
event history, 1 year, V=6 materialized views:
  rows  = 1.4e6 * 365 * 6 = 3.07e9 rows
  bytes = 3.07e9 * 90 B   = 276 GB     (before compression; columnar typically 3-5x better)

snapshot table for hourly cutoffs over 90 days:
  cells = 25e6 entities * 2,160 cutoffs = 5.4e10 cells
  bytes = 5.4e10 * 8 B  = 432 GB
  scan for one full rebuild at 1e6 rows/s: 1.26e8 rows / 1e6 = 126 s
```

The snapshot table looks affordable at 432 GB and becomes a problem the moment cutoffs are
per-request rather than hourly — the cell count is `entities × cutoffs`, so ad hoc scoring at
arbitrary `t_c` cannot be served from a snapshot table. That is the precise boundary between
"precompute" and "join on demand," and it is set by the cutoff cardinality, not the row count.

### 9.5 Latency budget arithmetic

End-to-end p99 targets compose multiplicatively in the tail, not additively. With six views each at
per-view p99 = 5 ms, fetched in parallel and assumed independent:

```text
P(at least one view exceeds 5 ms) = 1 - 0.99^6 = 1 - 0.941480 = 0.05852
so the fan-out's own p99 is worse than 5 ms; it sits where  1-(1-q)^6 = 0.01
per-view requirement for a fan-out p99 of 5 ms:  q = 0.99^(1/6) = 0.998331
i.e. each view must be at p99.83 for six independent views to meet p99 at 5 ms

sequential 6 views at 5 ms each            = 30 ms
parallel, independent                      ~ 5 ms + 1-2 ms coordination
parallel, shared slow dependency            ~ the dependency's latency, all views wait together
budget available for features at 120 ms end-to-end = 65 ms
```

Independence is the assumption that fails: a network event or a busy store delays all six views at
once, so a real fan-out's p99 is bounded below by the shared dependency's p99 and no per-view target
fixes that. Measure the fan-out; do not infer it from per-view numbers.

The budget arithmetic is the reason to cap `V` and to fetch in parallel. It is also the reason a
percentile feature must be a cache read: a 90-day percentile over 1.26e8 rows cannot be computed in
5 ms, so either it is precomputed (and the budget applies to the read) or it is unavailable at that
latency.

### 9.6 Cost of the parity check

```text
sample of n = 3,012 keys, one historical query + one online read each
historical query: partition-pruned scan of 168 h for 3,012 keys
  rows touched ~ 3,012 * 280 events = 843,360 rows -> sub-second on a columnar store
online reads: 3,012 point lookups at 3,000 req/s capacity = 1.0 s of capacity
total: minutes of engineer-attention per run, negligible compute
```

That the check is this cheap and this diagnostic is the whole argument for running it on every
definition change and nightly. The expensive part is retaining history long enough to query it,
which is a retention decision worth making explicitly.

### 9.7 Complexity classes

| Operation | Time | Space | Dominant term |
|---|---|---|---|
| Single online read | O(1) | O(1) | one round trip + one deserialization |
| Online read across `V` views | O(V) latency (parallel) or O(V·l) serial | O(V·B) | fan-out coordination |
| Materialize latest for one entity | O(window rows) or O(1) with incremental state | O(1) incremental | decomposability of the aggregate |
| Point-in-time join for one cutoff | O(rows in window) | O(rows in window) | window cardinality |
| Build training set for `C` cutoffs | O(C · rows per window) naive; O(rows) with a sorted sweep | O(C · entities) | cutoff count |
| Full history rebuild | O(total history rows) | O(output cells) | retention volume |
| Non-decomposable recompute | O(window rows) per cutoff | O(window distinct values) | cannot be incrementalized exactly |

The "sorted sweep" row is the standard optimization: sort events and cutoffs together by time and
maintain a rolling aggregate, so a training set over `C` cutoffs costs one pass rather than `C`
passes. It works for decomposable aggregates and fails for percentiles over a window that expires
rows — which is the algorithmic reason percentile features are usually bucketed approximations.

## 10. Key Theorems / Results Reference

The results below are the ones worth being able to state and justify in a design review. They are
elementary, but each one is a decision in disguise.

### R-01 Monotonicity of the legal row set

**Statement.** For fixed `e` and `W`, if `t_c1 <= t_c2` then every row legal at `t_c1` is legal at
`t_c2` unless it has left the window (`t_e < t_c2 - W`).

**Justification.** The filter is a conjunction of two half-open upper bounds and one lower bound:
`t_e < t_c` and `t_i <= t_c` both relax as `t_c` grows; `t_e >= t_c - W` is the only term that can
fail later. So legality is monotone up to window expiry.

**Consequence.** Historical feature values can be computed once and carried forward; a row never
becomes illegal for a reason other than expiry. This is what permits incremental materialization
and what makes a snapshot table meaningful.

**Where it breaks.** With an entity key that is rewritten (merge), or a definition that references
"current" attributes, monotonicity fails and precomputation must be invalidated rather than
extended.

### R-02 The leak bound for a fixed lag

**Statement.** With constant lag `L` and window `W > L`, the fraction of nominal window mass that
is invisible at `t_c` is `L/W`; with `L >= W` the feature is always empty.

**Justification.** The visible band is `[L, W)` in age, of width `W - L`; the nominal band is
`[0, W)`. Uniform rate gives the ratio `(W - L)/W`.

**Consequence.** Under a constant-lag source, leakage from a `t_e`-only join is exactly `L/W` of
the aggregate, which is computable before writing any code. For the illustrative `W = 168`,
`L = 26`: `26/168 = 15.48%`. (The worked example shows 2.97% because its lag is not constant —
one row of 31.5 h that is nonetheless visible. The two numbers are not in conflict; the formula
bounds the mass that *could* be late, not the mass that is.)

**Where it breaks.** Non-constant lag: the bound is the expected value of `(L_i)/W` over the rows,
not the ratio of means.

### R-03 Grace period optimum under exponential lag

**Statement.** Minimizing `alpha*E*exp(-G/mu) + beta*G` gives `G* = mu*ln(alpha*E/(beta*mu))`, and
`G* = 0` when `alpha*E <= beta*mu`.

**Justification.** First-order condition; the second derivative is
`-(alpha*E/mu^2)*exp(-G/mu) < 0`, so the stationary point is a minimum. Convexity holds because the
exponential is convex and the linear term is convex.

**Consequence.** The optimum scales with `mu` (better pipelines want shorter grace) and
logarithmically with the cost ratio. A 10x cost ratio change moves `G*` by `mu*ln 10`, not by 10x.

**Where it breaks.** Heavy-tailed lag (Section 4.4, Pareto): the exponential tail underestimates the
mass beyond `G`, so the model under-protects exactly the rows the grace exists to exclude.

### R-04 TTL null rate

**Statement.** Under Poisson refresh at rate `r` and reads independent of refresh, the share of reads
with no live value at TTL `T` is `exp(-r*T)`.

**Justification.** Inter-arrival times are exponential; the gap from an arbitrary read to the
previous refresh exceeds `T` with probability `exp(-r*T)`.

**Consequence.** Null rate is exponential in the product `r*T`, so doubling `T` squares the
"survival" factor. Setting `T` from an availability target means solving
`T = -ln(target_null)/r`.

**Where it breaks.** Correlated refreshes after outages, and entities whose refresh depends on
activity (dormant customers), both of which make the Poisson assumption wrong in the direction that
increases nulls.

### R-05 Fan-out tail inflation

**Statement.** For `V` independent views each with per-view tail probability `q` of exceeding
latency `x`, the fan-out exceeds `x` with probability `1 - (1 - q)^V`.

**Justification.** Union of independent exceedance events.

**Consequence.** Tail targets must be set per view: for a fan-out p99 with `V = 6`, per-view
`p = 0.99^(1/6) = 0.99833`.

**Where it breaks.** Correlated latency (shared network, shared store) makes the union bound
pessimistic — the true fan-out tail is often dominated by the shared dependency, so the per-view
target is necessary but not sufficient.

### R-06 Non-decomposability of distinct counts and quantiles

**Statement.** Distinct counts and exact quantiles over a sliding window cannot be maintained in
`O(1)` space per entity; exact maintenance requires `O(distinct)` or `O(window rows)` state.

**Justification.** The exact distinct count of a set requires distinguishing `2^d` possible subsets;
exact quantiles require the full order statistic. Both need space proportional to the input in the
worst case.

**Consequence.** A "real-time distinct merchants" or "rolling p95" feature is approximate by
construction, and the approximation must be documented with its error bound. Snapshotted exact
values will diverge from any recomputation, so the versioning rule from FM-08 applies.

**Where it breaks.** For small bounded domains (a merchant-category count over 20 categories), an
exact count is a bitmap of fixed size and is genuinely `O(1)`; the claim is about unbounded
domains.

### R-07 The sampling floor for parity testing

**Statement.** Detecting a mismatch rate `p` to absolute precision `eps` at confidence `1-a`
requires `n >= p(1-p)(z/eps)^2` samples, which for small `p` is approximately
`n >= p(z/eps)^2`.

**Justification.** Normal approximation to the binomial standard error.

**Consequence.** Verifying a 2% defect to `+/-0.5%` needs about 3,012 samples; verifying 0.1% to
`+/-0.05%` needs about 15,366. Small-defect verification is cheap enough to run nightly, which is
why "we cannot afford parity testing" is not a defensible position.

**Where it breaks.** Non-independent samples (repeated requests for one entity) inflate the
effective variance and require clustering by entity in the interval, or a larger `n`.

### R-08 Little's law for the online tier

**Statement.** `L = lambda * W_q` where `L` is concurrent requests, `lambda` arrival rate,
`W_q` time in system.

**Justification.** Conservation of requests in a stationary system.

**Consequence.** At `18,000` lookups/s and `5` ms service time, concurrency is
`18,000 * 0.005 = 90` in flight. That number sizes connection pools and thread counts, and it
explains why a pool of 20 serializes under load even though each call is fast.

**Where it breaks.** Non-stationary traffic (bursts), where instantaneous concurrency exceeds the
average by the burst factor and pools sized on the mean queue.

### R-09 Cost of a definition change

**Statement.** Changing a definition invalidates every historical value produced by it; the rebuild
cost is proportional to the number of `(entity, cutoff)` cells the training set spans, not to the
number of features.

**Justification.** Each cell is a function of the definition, so a change to the function changes
all cells.

**Consequence.** Definition changes should be batched, versioned, and scheduled against the retrain
cadence; a per-feature rebuild triggered by a whitespace fix is a recurring cost that never appears
in the design document.

**Where it breaks.** If the training set is small (few cutoffs, sampled entities), rebuilds are
cheap and versioning discipline matters less — which is also why small teams get away with practices
that break at scale.

### R-10 Train/serve parity as a consistency requirement

**Statement.** If the historical and online paths are implementations of different functions, the
difference is not bounded by any test; if they are implementations of the same function with
independent errors, the difference is bounded by the sum of the per-path error rates.

**Justification.** Two distinct functions can differ arbitrarily on inputs outside the tested set;
one function with two implementations differs only where an implementation is wrong.

**Consequence.** Generate both paths from one definition and test the implementations against each
other. The requirement is architectural, not statistical: no sampling plan detects a divergence
that only occurs on an input slice you never sample.

**Where it breaks.** Where generation is impossible — a streaming engine and a SQL warehouse cannot
share an implementation for every operator — the substitute is a shared test suite of golden
vectors covering the operator set, which bounds the divergence to operators not covered.
