## 4. Core Formulas with Full Derivation

The section derives four results that a design review will ask about: what a legal historical
aggregate is, how to measure agreement between the two paths, how large a grace period or TTL
should be, and where each derivation stops being true. Each result uses the same anatomy —
FORMULA, WHERE, CONSTRAINT, NUMERICAL EXAMPLE, INTERPRETATION — and ends with its validity
condition, because the validity condition is what an interviewer probes.

### 4.1 Base case

**FORMULA.**

```text
f(e, t_c) = AGG { x_r  :  r in H,
                        r.entity   = e,
                        t_c - W   <= t_e(r) <  t_c,
                        t_i(r)    <= t_c }

with the deterministic-lag special case  L(r) = t_i(r) - t_e(r)  constant:

        t_i(r) <= t_c   <=>   t_e(r) <= t_c - L
   so   f(e, t_c) = AGG { x_r : t_c - W <= t_e(r) <= t_c - L }
```

**WHERE.**

- `e` — entity key, a normalized string or integer with an explicit namespace.
- `t_c` — prediction cutoff, a UTC instant. Window is half-open: `[t_c - W, t_c)`.
- `W` — window length in hours; `W > 0`. For the examples `W = 168` h (7 days).
- `t_e(r)` — event time of row `r`, UTC instant. `t_i(r)` — ingestion time, UTC instant.
- `L(r)` — lag in hours; `L(r) = t_i(r) - t_e(r) >= 0` by construction of the ingest boundary.
- `x_r` — the aggregated quantity; for a spend sum, USD with 2 decimals in the source.
- `H` — the event history table, append-only, deduplicated on `(source_id, t_e, amount_minor)`.
- `AGG` — `SUM`, `COUNT`, `COUNT DISTINCT`, `MAX`, or a windowed statistic. Section 4.4 shows
  why the choice changes everything about backfill.

**CONSTRAINT.** The aggregate must be a function of the *visible* set, and the visible set must be
monotone in `t_c`: for `t_c1 < t_c2`, every row legal at `t_c1` is legal at `t_c2` unless it has
aged out of the window. Monotonicity is what makes the historical table reproducible; if a row can
become legal *and then illegal* for a later cutoff, the definition is not point-in-time.

**NUMERICAL EXAMPLE (ILLUSTRATIVE).** Take `W = 168` h and a fixed lag `L = 26` h. The usable span
of event times is `168 - 26 = 142` h, which is `142/168 = 84.52%` of the nominal window. With a
uniform event rate of `lambda = 40/24 = 1.6667` events/hour, the nominal window contains
`1.6667 * 168 = 280.0` events, and the legal window contains `1.6667 * 142 = 236.7` events. So a
naive join that ignores `t_i` over-counts by `43.3` events, i.e. `280.0/236.7 = 1.183` times the
correct count, or `+18.3%` on every entity with a uniform rate.

Now let the lag be exponential with mean `mu = 8.656` h — chosen so the median lag is 6 h, since
`mu = 6 / ln 2`. The expected visible fraction is

```text
E[visible] = lambda * INTEGRAL_0^W (1 - exp(-u/mu)) du
           = lambda * [ W - mu * (1 - exp(-W/mu)) ]
           = 1.6667 * [168 - 8.656 * (1 - exp(-19.41))]
           = 1.6667 * [168 - 8.656 * (1 - 0.0000)]
           = 1.6667 * 159.344
           = 265.6 events
```

Against 280 nominal, the loss is `(280 - 265.6)/280 = 5.15%`, not the 15.5% a fixed-lag reading
would suggest. Both numbers are correct for their own lag model, and the gap between them is the
first quantitative lesson: *the lag distribution, not its maximum or its mean, sets the loss.*

**INTERPRETATION.** The availability filter is not an optimization; it is part of the definition of
the feature. Its cost is measured in information discarded, and that cost is governed by how much
probability mass of the lag distribution exceeds zero inside the window. A team that skips the
filter does not get 18% more signal; it gets 18% of its aggregate computed from the future, and the
excess concentrates on exactly the entities whose late-arriving rows are interesting.

**VALIDITY CONDITION.** The base case holds when (a) `t_i` is recorded at the durable ingest
boundary, not copied from the source payload; (b) the event history is deduplicated before
aggregation; (c) `AGG` is decomposable so that the value is a function of the row set alone; and
(d) `t_e` and `t_c` share one timezone and one calendar. Fail any of the four and the arithmetic
above describes a different quantity than the one you computed.

### 4.2 General case

**FORMULA.** Define the parity residual for entity `e` and cutoff `t_c` as the relative difference
between the historical value and the value actually served, and test it against a tolerance on a
sample of production keys:

```text
delta(e, t_c) = ( v_hist(e, t_c) - v_serve(e, t_c) ) / max(|v_hist(e, t_c)|, eps)

PASS  <=>  share{ |delta| <= tau } >= 1 - alpha

sample size for a mismatch-rate estimate p with absolute precision eps_p:
        n = p (1 - p) (z_{1-a/2} / eps_p)^2
```

**WHERE.**

- `v_hist` — value from the historical feature table at the same `(e, t_c)`.
- `v_serve` — value returned by the online read, captured in the serving log, same request.
- `eps` — a floor on the denominator, e.g. `eps = 1.0` USD for a spend feature, so that a
  difference from 0.00 to 0.01 does not produce a residual of infinity.
- `tau` — tolerance, a dimensionless relative bound; `alpha` — acceptable share of violations.
- `p` — true mismatch rate; `z_{1-a/2}` — normal quantile, 1.959964 at 95% two-sided.
- `n` — number of sampled keys, integer; `eps_p` — half-width of the confidence interval.

**CONSTRAINT.** The residuals must be computed on the *same* cutoff, not merely the same calendar
day, and the sampling frame must be production traffic rather than a curated entity list. If you
sample entities that all traded yesterday, you test the materialization path and never the
staleness path.

**NUMERICAL EXAMPLE (ILLUSTRATIVE).** Suppose a smoke test on 20,000 production keys finds 400
mismatches, so `p_hat = 400/20,000 = 0.02`. The 95% confidence half-width is
`1.959964 * sqrt(0.02*0.98/20000) = 1.959964 * sqrt(0.00000098) = 1.959964 * 0.000990 = 0.00194`,
so the interval is `[1.81%, 2.19%]`. To resolve that rate to `+/- 0.5%` absolute you need

```text
n = 0.02 * 0.98 * (1.959964 / 0.005)^2
  = 0.0196 * 153,664.0
  = 3,011.8  ->  3,012 keys
```

and to resolve it to `+/- 0.2%` you need `0.0196 * (979.98)^2 = 18,824` keys. The factor of 6.25
in sample size for a 2.5x tighter interval is just `(0.005/0.002)^2 = 6.25`; the point is that
"we sampled a thousand requests and they all matched" cannot distinguish a 2% defect from a
0.5% defect.

Now translate a residual into a modelling consequence. Take a perfectly separating binary
indicator: present with probability 0.95 for positives and 0.05 for negatives. Its single-feature
AUC is `0.5 * (TPR + TNR) = 0.5 * (0.95 + 0.95) = 0.95`. A parity defect that flips 5% of the
indicator in one direction moves this to somewhere between 0.90 and 0.95 depending on which class
is corrupted — a change that is visible but not catastrophic. The reason parity errors are treated
as severe is not that they destroy AUC; it is that they destroy *calibration* on the subpopulation
where the online value is stale, and the decision threshold was tuned on the historical marginal.

**INTERPRETATION.** Parity is a measurable invariant with a tolerance, a population, and a sample
size — not a slogan. State `tau`, `alpha`, `eps`, the sampling frame, and the frequency, and the
check becomes a regression test that a platform team can own. Leave any of them unspecified and
the check degenerates into a dashboard that everyone has learned to ignore.

**VALIDITY CONDITION.** Valid when the historical table and the online table are produced from the
same definition version for the compared keys; when the residual is computed per entity rather than
on aggregates (aggregate comparison cancels errors of opposite sign and will pass while every
individual value is wrong); and when `eps` is chosen per feature unit so that near-zero values do
not dominate the violation count.

### 4.3 Variants

Three variants cover almost all design questions: choosing a grace period when lag is uncertain,
choosing a TTL for the online table, and choosing an aggregation granularity for cost.

**FORMULA (grace period).** With leak cost `alpha` per event exposed after the cutoff, information
cost `beta` per hour of excluded recency, and an exponential lag of mean `mu`:

```text
C(G) = alpha * E * exp(-G/mu) + beta * G

dC/dG = -(alpha*E/mu) * exp(-G/mu) + beta = 0
  =>   G* = mu * ln( alpha * E / ( beta * mu ) )
```

**WHERE.** `E` — expected number of events per entity-window (dimensionless count, 280 above);
`alpha` — cost per leaked event, in the same notional unit as `beta`; `beta` — cost per hour of
recency excluded, so `beta*G` and the first term share units; `G` — grace in hours, `G >= 0`;
`mu` — lag mean in hours. Use `G* = 0` when the logarithm is negative, i.e. when
`alpha*E <= beta*mu`.

**NUMERICAL EXAMPLE (ILLUSTRATIVE).** `E = 280`, `mu = 8.656`, `alpha = 0.05`, `beta = 0.02`:

```text
G* = 8.656 * ln( 0.05 * 280 / (0.02 * 8.656) )
   = 8.656 * ln( 14.0 / 0.17312 )
   = 8.656 * ln( 80.87 )
   = 8.656 * 4.3928
   = 38.02 h

check:  C(38) = 0.05*280*exp(-38/8.656) + 0.02*38
              = 14.0*0.01245 + 0.76
              = 0.1743 + 0.76 = 0.934
        C(24) = 14.0*0.0625 + 0.48 = 0.875 + 0.48 = 1.355
        C(48) = 14.0*0.00391 + 0.96 = 0.0547 + 0.96 = 1.015
```

The curve is flat near the optimum — 38 h costs 0.934, 48 h costs 1.015, a 8.7% penalty for a
26% longer grace — which is why arguing about the exact value is less useful than agreeing on
`alpha` and `beta`. Disagreement about `G*` is almost always disagreement about the two costs.

**FORMULA (TTL and null rate).** If online values for an entity are refreshed by a Poisson process
with rate `r` per hour and reads occur at arbitrary times independent of refresh, the probability
that a read finds no live entry under TTL `T` hours is `exp(-r*T)`, and the null share is

```text
null_share(T) = exp(-r * T)
```

`r` in refreshes/hour, `T` in hours, both positive; result dimensionless. At `r = 0.5/day = 0.02083/h`:
`T = 7 d` gives `exp(-0.5*7) = exp(-3.5) = 0.0302`, so a **3.02%** null rate; `T = 14 d` gives
`exp(-7) = 0.00091`, so **0.09%**, for double the memory. If instead entities refresh only when
they transact and 25% of entities are dormant, `r = 0.25/day` and `T = 7 d` gives
`exp(-1.75) = 17.4%` nulls — the same TTL with a different arrival process is not the same
availability policy.

**FORMULA (fan-out cost).** With `N` entities, `B` bytes per entity per view, and `V` views, the
online footprint is `N*B*V` and the read amplification at `Q` requests per second is `Q*V`
point lookups per second. For `N = 25e6`, `B = 120`, `V = 6`: `25e6 * 120 * 6 = 18.0 GB` of values
before replication and index overhead; at `Q = 3,000` req/s the tier must sustain
`18,000` lookups/s, which is roughly one moderately sized node at `20,000` lookups/s, i.e. no
headroom. Cutting `V` from 6 to 3 halves both memory and lookup rate and is a design change worth
more than any cache tuning.

**INTERPRETATION.** Each variant converts a qualitative design debate into one number to agree on:
`alpha/beta` for grace, `r` for TTL, `V` for footprint. Publish those three parameters next to the
feature definitions and the debates stop recurring.

**VALIDITY CONDITION.** The grace derivation assumes exponential lag and a linear recency cost; a
heavy-tailed lag invalidates it (Section 4.4). The TTL derivation assumes independent Poisson
refresh, which fails after a partial outage — post-incident refreshes are correlated, so the null
rate spikes far above the formula for the duration of the catch-up. The fan-out formula counts
values only; add replicas, per-key overhead, and allocator slack before quoting a memory bill.

### 4.4 Limits and counterexamples

**Counterexample 1 — non-decomposable aggregates break immutability.** Suppose the feature is
"distinct merchants transacted with in 30 days." A row with `t_e = day 0` and `t_i = day 3` is
legal at every cutoff from day 3 onward, and it can change the value at each of those cutoffs if
its merchant is otherwise absent. With hourly cutoffs, one late row can revise
`30 * 24 = 720` historical cells. The formula in 4.1 is still correct — the aggregate is a function
of the visible set — but the operational assumption that historical feature rows are immutable
after write is false. Any pipeline that appends feature snapshots and never revisits them will
serve stale distinct-counts forever, and any pipeline that rewrites them needs a defined
supersede rule.

```text
affected cutoffs for a late row r  =  { t_c : t_c >= t_i(r) and t_c - W < t_e(r) }
                                   =  t_c in [ t_i(r), t_e(r) + W )
count with hourly cutoffs         =  W - L   (hours),  = 142 above
```

**Counterexample 2 — a copied ingestion timestamp destroys the filter.** If `t_i` is set from the
source record's own "received" field, and the source backfills three days of records after an
outage, those records carry `t_i` values in the past. The filter `t_i <= t_c` then admits rows that
were physically absent at `t_c`, and the leak is proportional to the backfill size, not to the lag
distribution. The discriminating check is structural rather than statistical: compare the
distribution of `t_i - ingest_wallclock`; a healthy pipeline has a tight, non-negative
distribution. Any mass at values below the deployment date of the ingest schema indicates copied
timestamps.

**Counterexample 3 — Poisson thresholds on a mixture.** A monitoring rule that alerts when a
count exceeds `mean + 4*sqrt(mean)` assumes Poisson variance. Take a population that is 90% light
users at `0.2` events/day and 10% heavy users at `20` events/day, over `W = 7` days:

```text
mean count  = 0.9*(0.2*7) + 0.1*(20*7) = 0.9*1.4 + 0.1*140.0 = 1.26 + 14.00 = 15.26
E[lambda*W] = 15.26
Var(lambda*W) = 0.9*(1.4 - 15.26)^2 + 0.1*(140.0 - 15.26)^2
              = 0.9*192.10 + 0.1*15,560.1
              = 172.89 + 1,556.01 = 1,728.90
Var(count)  = E[Var|lambda] + Var(lambda*W) = 15.26 + 1,728.90 = 1,744.16
sd(count)   = 41.76          vs Poisson sd = sqrt(15.26) = 3.91
variance/mean = 114.3
```

A Poisson rule would flag any entity above `15.26 + 4*3.91 = 30.9` events; that is roughly the
median of the heavy segment, so the rule alerts on most of the 10% heavy tail and nothing else.
The dispersion index of 114 is the diagnostic: compute `Var/mean` per segment before writing a
threshold. Alert on a per-segment quantile or on a robust z-score against the segment's own
distribution, never on a global Poisson band.

**Counterexample 4 — heavy tails invalidate the exponential grace.** With lag modelled as Pareto
with shape `k = 1.5` and scale `x_m = 0.5` h, the mean is
`x_m*k/(k-1) = 0.5*1.5/0.5 = 1.5` h, yet `P(L > 168) = (0.5/168)^1.5 = 0.0001624`, i.e. `1.62`
rows per ten thousand. With `1.4e6` events/day that is still about `227` rows per day whose
lag exceeds the entire feature window. Those rows are not merely lost to the window; they arrive
after the cutoffs at which they would have mattered, so a grace period tuned on the mean
under-protects them by construction, and the entities affected are the ones whose upstream feed
was degraded — usually a specific merchant or processor, which makes the leak cluster in a
subpopulation instead of spreading evenly.

**Counterexample 5 — window boundary off-by-one changes the count.** With a half-open window
`[t_c - W, t_c)` and events exactly hourly, the count per entity is `W - 1`, not `W`. For
`W = 168` that is a 0.6% discrepancy, invisible per entity and systematic across the table.
Worse, mixing a closed convention in one feature and a half-open convention in another makes the
ratio of two features biased by a factor that grows as `W` shrinks; at `W = 24` the same
off-by-one is 4.2%.

**INTERPRETATION.** The derivations in 4.1 through 4.3 are correct only under their stated
assumptions, and production violates them in four recurring ways: aggregates that are not
decomposable, timestamps that were copied rather than recorded, distributions that are mixtures
disguised as single populations, and conventions that differ between features. Each violation is
detectable by a specific, cheap check — supersede rules, ingestion-clock skew, per-segment
dispersion, and a boundary unit test.

**VALIDITY CONDITION FOR THE WHOLE SECTION.** Every formula here presupposes that a legal row set
is well defined for `(e, t_c)`. That presupposition fails when entity keys are recycled (a card
number reissued to a different account), when entity history is merged or split without a
mapping table, or when the cutoff is expressed in a local calendar whose offset changed inside the
window. In those three cases the correct repair is not a better formula but an entity-resolution
table and a stored UTC cutoff, and no amount of arithmetic on the aggregate will recover the
missing distinction.

