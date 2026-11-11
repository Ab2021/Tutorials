## 11. Interview / Exam Questions

Thirteen questions, ordered by difficulty. At least half require arithmetic, selection, or diagnosis;
the answers are written to be given aloud in under two minutes.

**Q1 (recall). Define a point-in-time-correct join in one sentence, then name the two timestamps it
must filter on.**

A point-in-time join returns, for each entity and cutoff, an aggregate over only those history rows
whose event occurred before the cutoff and whose ingestion was visible at the cutoff. It filters on
`t_e < t_c` and `t_i <= t_c`. The first prevents future facts from entering; the second prevents
facts that had occurred but were not yet observable. Either filter alone leaves leakage, and the
second is the one that is silently omitted.

**Q2 (calculation). A feature uses a 7-day window and the source has a constant 26-hour lag. What
fraction of the nominal window is unusable, and what does that do to a count feature?**

The usable band is `168 - 26 = 142` hours, so `26/168 = 15.48%` of the window is unusable. A count
feature over a uniform event rate of `40/24 = 1.6667` events/hour drops from `280.0` to `236.7`,
a shortfall of `43.3` events. A sum feature drops by the same 15.48% if amounts are independent of
time, but not otherwise: if recent amounts are larger, the sum shortfall exceeds 15.48% while the
count shortfall stays at 15.48%.

**Q3 (diagnosis). Offline AUC 0.91, online precision at the operating threshold is barely above the
base rate, and the serving log shows the feature distributions are similar. Where do you look?**

Similar distribution marginals with a performance gap points at the *joint* rather than the marginal:
feature correlations or the calibration of the score, not the feature values. Check three things in
order: whether the parity residual per entity passes (if marginals match, per-entity residuals
should too, and a failure there means offsetting errors within the population); whether the score's
calibration curve on the serving population differs from training, which indicates prior shift
rather than skew; and whether the threshold was carried over from a population with a different base
rate. A base rate of 2% in training and 0.6% in production moves precision at a fixed threshold by a
factor of roughly three even with a perfect model.

**Q4 (calculation). You need to verify a suspected 2% parity defect to plus or minus 0.5% absolute
at 95% confidence. How many keys, and how does that change for 0.2%?**

`n = p(1-p)(z/eps)^2 = 0.02 * 0.98 * (1.959964/0.005)^2 = 0.0196 * 153,664 = 3,012`. For `eps =
0.002`: `0.0196 * (979.98)^2 = 18,824`. The ratio is `(0.005/0.002)^2 = 6.25`, so a 2.5x tighter
interval costs 6.25x the sample. Both are affordable nightly, which is the argument for making the
check a scheduled test rather than a one-off investigation.

**Q5 (selection). You need "number of distinct merchants in the last 30 days" served at p99 under
5 ms. Which of these do you choose: exact snapshot, exact recompute on read, or a sketch, and what
do you tell the model owner?**

A sketch (HyperLogLog-class) maintained incrementally, or a precomputed bucketed approximation. Exact
recompute on read cannot meet 5 ms at window cardinalities in the millions, and exact snapshots
cannot be revised when late rows arrive. Tell the model owner the error bound explicitly — typically
a relative standard error near `1.04/sqrt(m)` for `m` registers — and version the sketch parameters,
because changing `m` changes the value for the same inputs and invalidates any training set built
with the old parameter.

**Q6 (calculation). Six feature views, each independently at p99 = 5 ms, fetched in parallel. What is
the fan-out p99, and what per-view target meets a 5 ms fan-out p99?**

`1 - 0.99^6 = 5.85%` of fan-outs exceed 5 ms, so the fan-out p99 is above 5 ms. For the fan-out at
p99 = 5 ms with independence, each view needs `0.99^(1/6) = 0.99833`, i.e. p99.83. State the
independence caveat: shared dependencies correlate the tails, so the real answer is dominated by the
slowest shared hop and no per-view target compensates.

**Q7 (judgment). Your team wants a streaming feature store for a monthly-retrained churn model
scored daily. Argue the other side.**

Sub-second freshness buys nothing when the decision cadence is daily and the label matures over 30
days; the model consumes whatever the vector was at scoring time, and a one-hour-old vector is
indistinguishable in effect from a one-second-old one. The costs are real: streaming state is not
replayable at arbitrary cutoffs, so the training set cannot be built from the same code path; the
cluster adds an on-call surface; and late rows require retraction handling that a daily batch job
gets for free by recomputing. The defensible design is a daily snapshot table with an
availability-filtered join, and streaming only for the one real-time decision in the portfolio.

**Q8 (diagnosis). Counts doubled after a weekend backfill but averages are unchanged. Give the most
likely cause, one alternative, and the check that separates them.**

Most likely a non-idempotent write: the backfill appended rather than upserting, so every reprocessed
row exists twice. Count scales by 2, sum by 2, mean unchanged — exactly the observed pattern. The
alternative is genuine source duplicates, but those do not align to the backfill's time range. The
check: does the affected range's boundary coincide with the job's declared interval, and does a
second re-run double again? A second doubling is decisive for non-idempotency.

**Q9 (calculation). An entity refreshes every 12 hours like clockwork and the TTL is 7 days. What is
the null rate, and why does the Poisson formula not apply?**

Deterministic refresh at `r = 2/day` with `T = 7` days: the Poisson formula gives `exp(-14)`, which
is meaningless here because the gap between refresh and read is bounded well below 7 days. The true
null rate is 0 for any read that occurs after the first refresh cycle, so the TTL is inert. The
Poisson model applies when refresh times are unpredictable relative to read times; the diagnostic is
the coefficient of variation of the refresh interval, near 0 for clockwork and near 1 for Poisson.

**Q10 (judgment). Two teams compute `spend_7d` with different definitions. How do you resolve it
without a redesign?**

Pick the definition that matches the decision's information set at `t_c` — usually the
availability-filtered one — and publish it as a versioned artifact with a hash. Rename the other
rather than deleting it, because a live model may depend on it: `spend_7d_eventtime` is an honest
name for the version without the availability filter. Then reconcile: compute both on the same
cutoffs, report the per-entity differences and the population share where they differ by more than
the tolerance. The reconciliation table is what convinces the owning team, not the argument about
correctness.

**Q11 (calculation). A 25M-entity base with 6 views at 120 bytes each, two replicas, 30% overhead.
What is the provisioned memory, and what does the third view cost monthly at 0.05 USD per GB-hour?**

`25e6 * 120 * 6 = 18.0 GB` raw, `36.0 GB` with two replicas, `46.8 GB` with 30% overhead, and
`46.8 * 0.05 * 730 = 1,708 USD/month`. The third view is `25e6*120*1 = 3.0 GB` raw, `7.8 GB`
provisioned, `285 USD/month`. Per-view cost is linear and permanent, which is why view count is a
design budget item rather than an afterthought.

**Q12 (diagnosis). A feature's value changes for a fixed `(entity, t_c)` when recomputed two months
later, and no version changed. What are the candidate causes and how do you test them?**

Three candidates: the aggregate is non-decomposable and the recompute now sees later-arriving rows
(distinct count, percentile); the entity key mapping changed so a different row set is selected
under the same key; or the definition's dependencies changed silently — a referenced dimension
table was updated in place without versioning. Tests in order: compute the row set's checksum before
and after (a change means key mapping or filtering, candidates two and three); if the row set is
identical, the aggregate must be the cause, which confirms a non-decomposable feature being
recomputed over a superset. Then check whether the affected entities are exactly those with late
rows.

**Q13 (judgment). When would you deliberately *not* build a feature store?**

When there is one model, a batch decision cadence, and no second consumer in sight. A scheduled job
that writes a wide table and a model that reads it is reproducible, auditable, and has no cluster to
operate; the store's value is reuse across models and a latency tier, and neither exists yet.
Building it early buys a schema you will change, an on-call rotation you do not need, and a
definition-ownership argument before there is any dispute to settle. Revisit when the second model
needs the same aggregate with the same time semantics, which is the moment the reuse argument
becomes concrete.

## 12. Quick Reference Card

| Item | Rule | Number / form |
|---|---|---|
| Legal row set | filter event time *and* ingestion time | `t_c - W <= t_e < t_c` and `t_i <= t_c` |
| Constant-lag leak bound | fraction of window unusable | `L/W`; for `W=168, L=26`: 15.48% |
| Exponential-lag loss | expected visible events | `lambda*[W - mu*(1 - e^{-W/mu})]`; 5.15% at `mu=8.656` |
| Grace optimum | minimize leak + recency cost | `G* = mu*ln(alpha*E/(beta*mu))`; 38.0 h at the worked costs |
| TTL null rate | Poisson refresh | `exp(-r*T)`; 3.02% at `r=0.5/day, T=7d` |
| Parity sample size | binomial precision | `p(1-p)(1.96/eps)^2`; 3,012 at `p=2%, eps=0.5%` |
| Online footprint | entities x bytes x views | `N*B*V`; 18.0 GB raw for 25M x 120 B x 6 |
| Fan-out tail | per-view target for a fan-out tail | `p_fanout^(1/V)`; p99.83 per view for p99 at `V=6` |
| Diagnosis order | cheapest observable first | L4 units/defaults -> L3 parity -> L2 row set -> L1 lag |

Validity conditions that travel with the numbers: the leak bound assumes constant lag and a uniform
event rate; the grace optimum assumes exponential lag and a linear recency cost; the TTL formula
assumes independent Poisson refresh and fails after outages; the footprint formula counts values
only, excluding replicas beyond those stated, per-key overhead, and allocator slack; the sample-size
formula assumes independent samples, so cluster by entity when requests repeat. Every figure in this
card is ILLUSTRATIVE arithmetic from Section 5 and Section 9, not a measurement of any system.
