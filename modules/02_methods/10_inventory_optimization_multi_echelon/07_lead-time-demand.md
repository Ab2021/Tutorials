# Lead-Time Demand Distributions — compounding demand and lead-time variability correctly

> Module 10 — Inventory Optimization and Multi-Echelon Planning
> Topic 07 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 08 topics 05 and 06; Module 10 topic 04
> Version 2026.5

---

## 1. Why this topic exists

Every buffer and every trigger in inventory planning is a quantile of one object: the distribution
of demand over the protection period. The reorder point is its 95th percentile. The fill target is
an expectation over its upper tail. The safety stock is the distance between its mean and a chosen
quantile. In most planning systems that object does not exist. What exists is a mean, a standard
deviation typed into a material master record, and a family assumed by whoever wrote the formula
module twenty years ago.

The consequence is that two items with the same mean demand, the same standard deviation, and the
same target can require stock levels that differ by a third, and nothing on the item card
distinguishes them. In the worked example of this topic, a slow-moving seal kit with a mean of 12
units over five weeks needs 30.2 units to reach a 95 per cent cycle service level. A normal model
fitted to the same mean and standard deviation recommends 27.6 and delivers 92.9 per cent. Nobody
sees the 2.1-point gap, because both numbers are computed correctly from the same two inputs. The
difference is the shape.

Three costs follow, in ascending order of visibility.

1. **The family is never stated, so the number cannot be audited.** A planner who asks why the
   buffer is 194 receives the formula. The formula contains a `z` value and a sigma, and neither
   carries the assumption that produced it. When the item changes behaviour there is no record of
   what the number was a quantile of, and no way to know whether it still is.
2. **The compounding is dropped, so the variance is short.** Demand over a lead time is a random
   sum: a random number of periods, each of random size. The variance of a random sum contains a
   term that scales with the square of the mean, and a formula written for a fixed horizon does
   not have it. Module 10 topic 04 works the operational consequence; this topic supplies the
   object that makes the omission visible.
3. **The estimate is computed on censored data, so sigma is biased down exactly where service is
   already failing.** Observed sales are the minimum of demand and available stock, so the periods a
   stock-out removes are those with the highest demand: the sample standard deviation understates the
   population one, and the buffer reproduces the stock-out it was meant to prevent.

> **Note:** This file does not re-derive the compound Poisson or the hurdle model. Module 08 topic
> 06 establishes both, including the fraction-of-zeros estimator and the Panjer recursion; those
> results are cited here and extended to the lead-time dimension. Module 08 topic 05 establishes
> the probabilistic forecasting vocabulary. Module 10 topic 04 derives the buffer formula, which
> this topic feeds but does not repeat.

## 2. Learning outcomes

By the end of this topic you can:

1. **Draw** lead-time demand as a two-level random sum — random number of periods, random size in
   each — and **state** which parameter governs which feature of the resulting distribution.
2. **Derive** the variance identity `Var(S) = E[N]·Var(Z) + Var(N)·(E[Z])²` from the law of total
   variance, and **identify** the term that appears as lead-time variability in the buffer formula.
3. **Decide** whether a normal approximation is defensible for a named item, using the coefficient
   of variation (CV, the standard deviation divided by the mean), the skewness, and the probability
   mass at zero, and **quantify** the service consequence when it is not.
4. **Test** equidispersion against overdispersion with a dispersion statistic, and **choose**
   between the Poisson and negative binomial (NB) counting distributions on the evidence.
5. **Distinguish** a compound Poisson from a hurdle model in what each produces at a horizon longer
   than the fitted bucket, and **choose** between them for a named dataset.
6. **Select** a magnitude distribution from what its shape implies about the upper tail, and
   **name** the diagnostic it will most likely fail.
7. **Quantify** the Monte Carlo standard error of a simulated quantile, and **state** how many
   replications a service decision actually needs.
8. **Adjust** a variance for serially correlated demand and for correlated lead times, and
   **explain** why the square-root-of-time rule is a special case rather than a law.
9. **Detect** censoring, **choose** a reconstruction method, and **state** the regime in which
   reconstruction becomes less reliable than the raw estimate.
10. **Write** a distribution specification that states its family, parameters, window, censoring
    treatment, and validation result.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold one picture: lead-time demand is not a quantity per period multiplied by a horizon. It is a
sum of a random number of random quantities, and the randomness at each level contributes a
separate term to the variance.

### 3.1 The tree

```text
                       LEAD-TIME DEMAND  D(L)  =  what is demanded in L periods
                                            |
              +-----------------------------+-----------------------------+
              |                                                           |
      HOW MANY PERIODS?                                          HOW BIG IN EACH?
      the horizon is itself random                               a period is itself a sum
              |                                                           |
      L ~ F_L,  mean 4.0 wk, sd 1.0 wk                           Y_t = Z_1 + ... + Z_{K_t}
      (order-to-availability, measured)                          K_t ~ Poisson(0.48 per wk)
              |                                                  Z ~ Gamma(k=2, scale 2.5)
              v                                                           v
      +-------------------+                                     +---------------------+
      | Var(L) * (E[Y])^2 |  <- the lead-time term             | Var(Y) > Var of a   |
      | often omitted     |     in the buffer formula           | single unit         |
      +-------------------+                                     +---------------------+
              |                                                           |
              +-----------------------------+-----------------------------+
                                            |
                                            v
                             D(L) = sum over t = 1..L of Y_t
                             E[D]    = E[L] * E[Y]              = 4.0 * 3.0   = 12.0
                             Var(D)  = E[L] * Var(Y) + Var(L) * (E[Y])^2
                                            |
                 +--------------------------+--------------------------+
                 |                          |                          |
          reorder point              fill rate                  cost of a
          at quantile tau        E[(D - S)^+] / E[D]           service target
```

The left branch is uncertainty about *when*; the right branch is uncertainty about *how much*. They
are different mechanisms with different parameters, estimated from different data, entering the
variance through different terms. A buffer formula containing only one branch is not approximately
right. It answers a different question.

### 3.2 The two compounding problems

**Problem one — demand arrives in lots.** A period's demand is not a single unit but a customer order,
a case, or a pallet, so the period total is itself a sum of random quantities whose variance depends on
the *second moment* of the lot size: the compound Poisson structure of Module 08 topic 06.

**Problem two — the number of periods is itself random.** "Four weeks" is a mean; the realised lead
time is a draw, and demand accumulates over whatever horizon occurs. This produces the term
`Var(L)·(E[Y])²`, which grows with the square of the demand rate. When both structures nest, the
variance is

```text
   Var(D) = E[L] * Var(Y)  +  Var(L) * (E[Y])^2
            where Var(Y) = lambda * E[Z^2]      (compound Poisson periods)
```

The first term carries problem one. The second carries problem two. Neither is visible in a
one-period standard deviation, which is what most systems store.

### 3.3 The variance identity, and where it comes from

Everything in this topic rests on one result, and it is two lines of algebra. For
`S = Z_1 + ... + Z_N` with `N` independent of the sizes:

```text
   E[S]    = E[N] * E[Z]
   Var(S)  = E[ Var(S | N) ]  +  Var( E[S | N] )
           = E[ N * Var(Z) ]  +  Var( N * E[Z] )
           = E[N] * Var(Z)    +  Var(N) * (E[Z])^2
```

Three consequences worth carrying into every later section.

**The second term is the one that gets dropped.** It contains `(E[Z])²`, so it grows with the
square of the mean. For a high-volume item with a moderately unreliable supplier it dominates. For
a low-volume item it is negligible. That is why the omission is invisible on a slow mover and
expensive on a fast one, which is the opposite of where most quality effort goes.

**The lead-time term is this term, not a separate formula.** In the buffer formula of Module 10
topic 04, `mu_d² · sigma_L²` is exactly `Var(N)·(E[Z])²` with `N` set to the lead time and `Z` set
to one period's demand. The formula has not added a new idea. It has specialised the identity.

**For a Poisson count, the two terms merge.** When `N ~ Poisson(λL)`, `Var(N) = E[N]`, so
`Var(S) = E[N]·Var(Z) + E[N]·(E[Z])² = E[N]·E[Z²]`. The compound Poisson variance is a single
product, and its second term is hidden inside `E[Z²]`. This is why analysts who know that formula
often fail to recognise the lead-time term when they meet it in a buffer formula: they are the same
term, written in two notations.

### 3.4 What the two distributions look like

```text
   density
      |                     normal fit                true lead-time demand
      |                     (symmetric)               (right-skewed, atom at 0)
      |                         .                              |
      |                       .   .                            |##
      |                      .     .                           |####
      |                     .       .                          |######
      |                    .         .                     .   |########
      |                   .           .                 .      |###########
      |   .................             ...........  .          |##############...
      |   :  impossible                      .                  |####################...
      +---+----------+----------+----------+---+----------+----+-----------------------> units
        -6           0         12        24  27.6       30.2  36
         ^           ^                                   ^     ^
         |           |                                   |     |
   10.3% of the   true atom at zero:                  normal  true
   normal's mass  9.1% of cycles                    95th     95th
   lands here                                         27.6     30.2
```

The normal curve is not wrong everywhere. It is wrong at two places that matter: below zero, where
it assigns probability to a physical impossibility, and above its 95th percentile, where it is 2.6
units short of the truth. The first is a diagnostic. The second is the cost.

### 3.5 Why the square-root-of-time rule is a special case

The rule "the standard deviation over `P` periods is `sigma` times the square root of `P`" is the
independence case of a more general expression:

```text
   Var( sum of P periods ) = P * sigma^2 + 2 * sigma^2 * sum_{k=1..P-1} (P - k) * rho(k)
```

where `rho(k)` is the lag-`k` autocorrelation of demand. Set every `rho(k)` to zero and you have
the rule. Set them positive — as they are for a promoted item, a trending item, or any item whose
demand follows a customer ordering cycle — and the true variance is larger, sometimes several times
larger, and the rule understates. The rule is not a property of time. It is a property of
independence, used as though independence were the default, and in the buffer formula it produces a
systematic understatement for exactly the items with the most structure in their demand.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Lead-time demand (LTD) | Total demand realised over the protection period | Computed as a rate times a rounded horizon | Units per cycle |
| Protection period | Review period plus lead time; the span a buffer covers | Assumed equal to lead time | Weeks or days |
| Random sum | A sum of a random number of random quantities | Treated as a product of two means | Units |
| Random horizon | A horizon whose length is itself a random variable | Replaced by its mean and treated as fixed | Periods |
| Compound distribution | The law of a random sum; a mixture over the count | Used loosely for any heavy-tailed law | Model class |
| Count distribution | The law of the number of periods or events in the horizon | Confused with the demand distribution | Count per period |
| Magnitude distribution | The law of the quantity carried by one event; the size law | Fitted to period totals rather than to events | Units per event |
| Compound Poisson | Poisson arrivals, each carrying an independent random size | Used for any zero-inflated series | Events per period, plus a size law |
| Hurdle model | Bernoulli occurrence plus a zero-truncated size law, at most one event per period | Treated as identical to a zero-inflated model | Model class |
| Intermittent demand | Demand with periods of zero and irregular positive values | Defined by a single CV threshold | Classification |
| Arrival rate (λ) | Expected number of demand events per unit time | Confused with the occurrence probability | Events per week |
| Occurrence probability (p) | Probability that a period contains at least one event | Plugged where an arrival rate is expected | Proportion per period |
| Poisson distribution | Count law with variance equal to the mean | Applied to any count data without a dispersion check | Count per period |
| Negative binomial (NB) | Count law with variance greater than the mean; a gamma-mixed Poisson | Used for continuous magnitudes | Count per period |
| Variance-to-mean ratio (VMR) | Sample variance divided by the sample mean | Reported as "CV", which is dimensionless and different | Dimensionless |
| Index of dispersion | `(n − 1)s²/x̄`; chi-square with `n − 1` degrees of freedom under Poisson | Reported without its degrees of freedom | Dimensionless statistic |
| Equidispersion | Variance equal to the mean; the Poisson property | Assumed rather than tested | Model property |
| Overdispersion | Variance exceeds the mean | Treated as a nuisance rather than as evidence of compounding | Dimensionless |
| Dispersion test | A test of the null that variance equals the mean | Run on autocorrelated data, where it over-rejects | Test |
| Gamma distribution | Continuous right-skewed family with shape `k` and scale `θ`; mode above zero when `k > 1` | Fitted to counts without acknowledging continuous support | Units per event |
| Lognormal distribution | A family whose logarithm is normal; heavier right tail than the gamma at equal CV | Assumed to fit small sizes well, which it does not | Units per event |
| Skewness | Standardised third central moment | Reported without saying whether it is sample or model | Dimensionless |
| Zero mass | Probability that the horizon total is exactly zero, `P(D = 0)` | Ignored because "demand is never negative" | Proportion |
| Convolution | The operation that gives the law of a sum of independent variables | Performed on rates rather than distributions | Distribution |
| Panjer recursion | A recursion computing the compound sum for discrete magnitudes | Applied to continuous magnitudes without discretisation | Algorithm |
| Fast Fourier transform (FFT) convolution | A numerical route to the compound sum via characteristic functions | Used without checking discretisation error in the tail | Algorithm |
| Monte Carlo standard error | The standard deviation of a simulated statistic across replications | Confused with the standard deviation of the data | Units |
| Censoring | Observed demand truncated at available stock | Treated as a data-cleaning problem | Observation property |
| Reconstruction | Estimating the unconstrained demand distribution from censored observations | Applied without checking when it is identifiable | Estimator |
| Temporal aggregation | Combining periods into coarser buckets | Assumed to leave the process unchanged | Operation |
| Autocorrelation | Correlation between observations at different lags | Assumed zero because the series "looks random" | Dimensionless |
| Effective variance | Variance adjusted for serial correlation over the horizon | Omitted from every textbook buffer formula | Units squared |
| Square-root-of-time rule | Scaling a one-period standard deviation by the square root of the horizon | Applied under correlation, where it understates | Scaling rule |
| Coefficient of variation (CV) | Standard deviation divided by the mean | Compared across different buckets | Dimensionless |
| Probability integral transform (PIT) | `F(y)` evaluated at the realised value; uniform when the forecast is calibrated | Reported as a histogram without a binning rule | Dimensionless, in [0, 1] |
| Akaike information criterion (AIC) | A penalised likelihood for comparing non-nested families | Used to compare families fitted to different data | Dimensionless |

Two pairs deserve expansion, because each is where a competent analyst goes wrong.

**VMR and CV are not the same diagnostic.** The VMR is variance over mean, used to choose between
count families; 1 means Poisson, above 1 means NB, below 1 means neither. The CV is standard
deviation over mean, used to judge whether a normal approximation is safe. For a count with mean 3.7
and variance 15.6 the VMR is 4.2 and the CV is 1.07. Both are large, for different reasons, and
quoting one where the other is meant changes which family you fit.

**Censoring is an observation property, not a data quality defect.** Nothing is wrong with the
recording. The world produced less observed demand than it would have produced with infinite stock,
and the recording is faithful to the world. Treating censored periods as missing values deletes
exactly the information about how large demand can get.

## 5. The mechanics, step by step

Each step states its input, its output, and the branch where the analysis usually goes wrong.

### Phase A — Fix the object

**Step 1. Name the item, the location, the unit, the bucket, and the construct.**

- Input: an SKU (stock-keeping unit) identifier, a stocking location, a unit of measure, a bucket, and
  a definition of the demand series (consumption, customer orders, shipments, or point-of-sale scans).
- Branch: **selling unit or stocking unit?** A case-packed item held in eaches but planned in cases
  produces a factor-of-twelve error that survives every downstream check.
- Branch: **does the construct contain the variability you are buffering against?** Shipments absorb
  batching and are smoother than demand; orders contain gaming; scans are closest to demand.

**Step 2. State the horizon and whether it is fixed.**

- Input: the review period `R` and the realised lead times for the lane.
- Output: either a fixed `P = R + L` or a fitted distribution for `L`.
- Branch: **is the lead-time standard deviation more than about a fifth of its mean?** Below that, a
  fixed horizon is defensible; above it, the horizon is a random variable. Never round 3.7 weeks to
  4: scale the rate instead.

### Phase B — Establish the compounding structure

**Step 3. Test whether periods are themselves sums.** Do the period totals cluster on multiples of a
pack quantity — 12, 24, 36, 48 with nothing between? Then the period is a sum of cases and a
continuous model smooths away the structure service depends on. Is the variance materially above what
a single draw implies? A CV above 1 with strong right skew in period totals usually means lots.

**Step 4. Test whether the horizon is random, and place the item in one of four structures.**

- Input: order release dates and goods receipt dates, per lane, per order.
- Diagnostic: **is the lead time stationary?** A drifting mean, a variance that grows in peak
  season, or a bimodal histogram (fast and slow modes) all mean a single fitted distribution is a
  summary of two regimes.

| Structure | What is random | Variance of the horizon total | Where you meet it |
|---|---|---|---|
| Deterministic horizon, single draw | Only the period demand | `P · σ_d²` | Domestic supplier, smooth item, daily bucket |
| Deterministic horizon, compound periods | Demand lots within each period | `P · E[K] · E[Z²]` | Spare parts, wholesale with pack sizes |
| Random horizon, single draw | Lead time only | `E[L]·σ_d² + Var(L)·μ_d²` | Imported goods, forecast-driven demand |
| Random horizon, compound periods | Both | `E[L]·E[K]·E[Z²] + Var(L)·(E[K]·μ_Z)²` | Most import lanes for slow movers |

The fourth row is the general case and is where most of a distributor's catalogue sits. The first
three are special cases, each defensible only when its restriction is tested. The practical point is
to know which restriction you have taken and to state it.

### Phase C — Compute the exact first two moments

**Step 5. Write the identity, then substitute.** A computation that produces only the first term is
not an approximation to the second. It is a different quantity.

```text
   S = Z_1 + ... + Z_N,   N independent of the sizes
   E[S]   = E[N] * E[Z]
   Var(S) = E[N] * Var(Z)  +  Var(N) * (E[Z])^2
```

**Step 6. Compute the three terms separately and label them.**

| Term | Expression (random horizon, compound periods) | What it represents | Typical share |
|---|---|---|---|
| Lot term | `E[L] · E[K] · Var(Z)` | Variability of lot sizes within a period | 10–40% |
| Period-compounding term | `E[L] · Var(K) · μ_Z²` | Variability in how many lots a period holds | 10–40% |
| Horizon term | `Var(L) · (E[K]·μ_Z)²` | Variability in how long the exposure lasts | 0–70% |

For a Poisson count the first two collapse into `E[L]·E[K]·E[Z²]`, the compound Poisson variance of
Module 08 topic 06. It is the same identity with `Var(K) = E[K]` substituted.

**Step 7. Check the mean against an independent estimate.** The mean is identified by the data even
when the decomposition into count and magnitude is not. If `E[N]·E[Z]` does not reproduce observed
mean demand, one of the two parameters is wrong, and the variance built from it will be wrong in a
direction nobody can see.

**Step 8. Propagate the sigma estimation error into the buffer.** The buffer is linear in sigma, so
the elasticity of the buffer to sigma equals the share of variance the error touches. For an item
with `P = 5`, `μ_d = 12`, `σ_d = 6`, `σ_L = 1.0`, and `z = 1.6449`, the protection-period standard
deviation is 18.00 units and the safety stock is 29.61 units, with demand contributing 55.6 per cent
of the variance.

| Perturbation | Resulting sigma | Resulting buffer | Change |
|---|---:|---:|---:|
| None | 18.00 | 29.61 | — |
| `σ_d` +20% | 20.08 | 33.03 | +11.6% |
| `σ_d` −20% | 16.10 | 26.49 | −10.6% |
| `σ_L` +20% | 19.68 | 32.37 | +9.3% |
| `σ_L` −20% | 16.50 | 27.14 | −8.3% |
| `σ_d` ±20%, deterministic lead time | ±20.0% | ±20.0% | ±20.0% |

A 20 per cent error in sigma is a 20 per cent error in the buffer when sigma is the only source of
variability, which is the single-term case most planning systems assume. When a second source
exists the response is damped to that source's share — here to about 11.6 per cent — but it is
damped, not removed, and in neither case is the error visible in the output. That is the argument
for estimating sigma on a defensible window and stating the window, not for adding a fudge factor.

### Phase D — Decide whether the normal approximation is defensible

**Step 9. Screen on the coefficient of variation of the horizon total.** For a normal with mean `μ`
and standard deviation `σ`, the probability of a negative value is `Φ(−1/CV)`. That is a physical
impossibility expressed as a probability, and its size is the cheapest available diagnostic.

| CV of lead-time demand | `P(X < 0)` under a normal fit | Verdict |
|---:|---:|---|
| 0.2 | 0.0000 | Normal is safe in the body and usually in the tail |
| 0.4 | 0.0062 | Normal is acceptable for most decisions |
| 0.6 | 0.0478 | Normal is questionable; the tail is mis-shaped |
| 1.0 | 0.1587 | Normal is wrong; 16% of the mass is on impossible values |
| 1.5 | 0.2525 | Normal is indefensible; a quarter of the mass is negative |

The CV of the horizon total is not the CV of weekly demand. It falls as the horizon lengthens,
`CV(D over P) ≈ CV_d / sqrt(P)` under independence: a weekly CV of 1.77 becomes 0.79 monthly and 0.49
quarterly. That is why a monthly planning process can get away with a normal approximation that a
weekly process cannot.

**Step 10. Check the zero mass and the skewness.** A normal assigns zero probability to zero demand,
so any item whose horizon total is zero in a material share of cycles cannot be normal. In the worked
example the zero mass is 9.1 per cent and the skewness 1.05, and the normal's implied `P(X < 0)` of
10.3 per cent coincides with the true zero mass. That is the trap: the two agree at the bottom, so
nothing looks wrong, while the upper tail is 2.6 units short.

**Step 11. Check the mean in units, not the CV alone.** Below about ten expected units over the
protection period, use the exact distribution. Above about thirty, the normal is usually adequate
for targets up to 95 per cent and increasingly poor above it. Between the two, decide on skewness
and on the cost of being wrong.

> **Note:** The evidence is not unanimous in the direction practitioners expect. Lau and Lau (2003)
> show that cost penalties from the normal approximation can be serious even at modest CV, while
> Tyworth and O'Neill (1997) find it robust across seven groups of fast-moving finished goods at a
> demand CV near 0.40. "Defensible at high mean demand" is a summary of a contested literature, not
> a theorem. Section 9 sets out the disagreement.

### Phase E — Choose the counting distribution

**Step 12. Run the dispersion test before fitting anything.**

```text
   VMR        = s^2 / xbar
   dispersion = (n - 1) * s^2 / xbar      ~ chi-square with n - 1 df, under Poisson
```

A statistic near its degrees of freedom is consistent with equidispersion. A statistic far above it
means the variance exceeds the mean, which rules out the Poisson and does not by itself select the
NB.

**Step 13. Choose the count family on the direction of the departure.**

| Evidence | Reading | Family | What to watch |
|---|---|---|---|
| VMR ≈ 1, dispersion near `n − 1` | Equidispersion | Poisson | Overdispersion hidden by pooling heterogeneous items |
| VMR > 1.5, dispersion far above `n − 1` | Overdispersion | Negative binomial | The dispersion parameter is poorly identified on short series |
| VMR < 1 | Underdispersion | Neither Poisson nor NB | Suggests a capped process, a fixed schedule, or a mis-defined event |
| VMR > 1 with a level shift | Overdispersion from heterogeneity | Segment first, then test | Pooling two rates produces spurious overdispersion |

A practical threshold is that a Poisson is adequate when the Pearson dispersion statistic is at or
below about 1.2, an NB is indicated between about 1.2 and 1.5, and the choice matters above that. The
NB is not free: on 60 periods the dispersion parameter is estimated with wide uncertainty, and its
quantiles can be less stable than the Poisson's even when the Poisson is wrong.

**Step 14. Test the assumption both families share.** Both assume independent increments. Seasonal
items, promoted items, and items with a single large customer violate that in a way no dispersion
test detects, because the marginal VMR can look correct while the serial structure is wrong.

### Phase F — Choose the magnitude distribution

**Step 15. Choose from what the shape implies about the tail.**

| Family | Parameters | Shape it imposes | Tail behaviour | Diagnostic it usually fails |
|---|---|---|---|---|
| Gamma | shape `k`, scale `θ`; `k = (μ/σ)²` | Right skew, mode above zero when `k > 1`, exponential when `k = 1` | Lighter than lognormal at equal CV | Upper-decile fit; a P-P (probability-probability) plot departing above the 90th percentile |
| Lognormal | `μ_log`, `σ_log` | Multiplicative, very heavy right tail, mode far below the mean | Heavier than gamma; can produce implausible extremes | Lower-tail fit; cannot represent a minimum order quantity |
| Weibull | shape, scale | Flexible hazard; shape below 1 gives a decreasing hazard | Between the two | Modal fit when shape is below 1 |
| Empirical | none | Nothing beyond the sample | Truncated at the sample maximum | The tail, always |

The gamma has a property the lognormal lacks: it is closed under the operations that generate the
process. A Poisson number of gamma-distributed lots is a compound-Poisson-gamma, a member of the
Tweedie family, with a closed-form density. That is why the gamma is the default in parts of the
spare-parts literature: a computational advantage, not a claim that demand is gamma in nature.

**Step 16. Fit by maximum likelihood and record the diagnostic you will be judged on.** Report the
AIC only between families fitted to the same observations. An AIC computed on period totals and one
computed on event sizes are not comparable, and the comparison is silently made in more vendor
evaluations than anyone admits.

### Phase G — Compute the distribution

**Step 17. Pick the route and state what it adds.**

| Route | What it gives | Assumption it adds | When it wins |
|---|---|---|---|
| Closed form | Exact quantiles for special pairings, such as compound-Poisson-gamma | The pairing is right | Long histories, stable process, extrapolation |
| Panjer recursion | Exact discrete compound sum in linear time per grid point | A discretisation fine enough for the tail | Discrete magnitudes, or a fine grid |
| FFT convolution | Fast numerical compound sum via characteristic functions | Discretisation error checked in the tail | Large grids, many items, batch processing |
| Simulation | Any structure, including correlation and random horizons | The generative model, plus enough replications | Random lead times, correlated arrivals, policy testing |

**Step 18. Size the simulation, because the quantile is noisier than the mean.** The Monte Carlo
standard error of a quantile is approximately `sqrt(q(1 − q)/M) / f(q)`, with `M` the replication
count and `f(q)` the density at the quantile. With `f(30.15) ≈ 0.0070` per unit, 1,000 replications
give a standard error of 0.98 units on the 95th percentile and 100,000 give 0.10, so a thousand
replications and two decimals is false precision by a factor of ten. The mean converges at `σ/√M`.

### Phase H — Correlation, censoring, aggregation

**Step 19. Adjust the variance for serial correlation.**

```text
   Var( sum_{t=1..P} d_t ) = P * sigma^2 + 2 * sigma^2 * sum_{k=1..P-1} (P - k) * rho(k)
```

An AR(1) (first-order autoregressive) process with `rho = 0.3` and `P = 5` gives a ratio of
`1 + 2(0.4·0.3 + 0.3·0.09 + 0.2·0.027 + 0.1·0.0081) = 1.30` in the variance, or a 14 per cent larger
standard deviation. The buffer rises by 14 per cent and no line in the formula shows it.

**Step 20. Test whether lead times are correlated across successive orders.** Consecutive orders
share congestion, modes, capacity constraints, and one supplier's queue, so they are almost never
independent. Positive correlation makes the shortfall distribution more variable than the
independent model implies and can make it multimodal, where normal and NB approximations both fail.
The statement worth carrying into a meeting: **correlation in lead times does not change the
distribution of lead-time demand; it changes the distribution of the shortfall, which is what the
buffer covers.** An analyst who checks only lead-time demand concludes correlation is irrelevant.

**Step 21. Detect and repair censoring.** Observed sales are `min(demand, available stock)`. The
truncation removes the upper tail, which is where the variance lives, so the sample standard
deviation is biased downward, and the bias grows with the stock-out frequency.

| Method | What it does | Assumption | When it becomes unreliable |
|---|---|---|---|
| Exclusion | Drops censored periods | Censoring is unrelated to demand level | Availability correlates with season or demand |
| Availability-conditioned mean | Scales the available-period mean by the availability ratio | Demand in unavailable periods resembles demand in available ones | Demand is seasonal and stock-outs cluster in peaks |
| Projection | Replaces each censored observation by `E[D \| D > stock]` under a fitted law | The fitted law is right in the tail | Short histories, where the tail is least well estimated |
| Censored likelihood | Maximises the censored likelihood, each stock-out a lower bound | Only the censoring mechanism, not the tail shape | When the availability record is itself unreliable |
| Expectation-maximisation (EM) | Treats latent demand as missing and iterates | A full model for demand and availability | High censoring with a misspecified model |

The standard stock-out adjustment degrades once the censored share passes roughly half the periods.
Above that, the honest output is the raw estimate with the censored share stated, not a corrected
number pretending to be an estimate.

**Step 22. Decide the reporting bucket, and recompute rather than rescale.** For a compound Poisson
the family is closed under temporal aggregation, so the rate scales exactly and the magnitude law is
unchanged. That is the only clean case. For an autocorrelated series, one with a day-of-week pattern,
a censored series, or one whose bucket boundary interacts with the review cycle, aggregation changes
the shape and the parameters must be re-estimated rather than multiplied.

### Phase I — Report

**Step 23. Write the distribution card.** A distribution is not specified until five things are
stated with it: family, parameters, window, censoring treatment, and validation. Line 1 is what most
reports omit, and it determines every quantile downstream; line 4 changes the answer most and is
almost never present. A filled card appears in section 6.9.

## 6. Worked example

One item at one distribution centre, from the raw statistics to the service decision. All figures are
constructed to be internally consistent and are reproducible from the inputs shown.

### 6.1 The item card

| Quantity | Value | Unit |
|---|---:|---|
| Item | 4417-C | replacement seal kit, three pump models |
| Mean demand (`mu_d`) | 2.4 | kits per week |
| Weeks with zero demand | 64 of 104 (61.5%) | proportion |
| Sample standard deviation, weekly | 4.24 | kits per week |
| Review period (`R`) | 1 | week |
| Mean lead time (`L`) | 4.0 | weeks |
| Standard deviation of lead time (`sigma_L`) | 1.0 | week |
| Protection period (`P = R + L`) | 5.0 | weeks |
| Target service | 95% cycle service level (CSL) | proportion |

The weekly coefficient of variation is 1.77, ordinary for a slow-moving replacement part and the
first indication that a normal model will be uncomfortable: 61.5 per cent of weeks have no demand.

### 6.2 Estimating the generative model

The zero fraction identifies the arrival rate under Poisson arrivals, by the estimator of Module 08
topic 06:

```text
   n0 / n        = 0.615
   lambda_hat    = -ln(0.615) = 0.486 per week    (used below as 0.48)
   mean demand   = 2.40 kits per week
   mean size     = 2.40 / 0.48 = 5.00 kits per event
```

The magnitude is fitted as a gamma with shape `k = 2` and scale `θ = 2.5` kits: mean 5.00 kits and
second moment `E[Z²] = k(k+1)θ² = 2 × 3 × 6.25 = 37.5` kits², which is what enters the lead-time
variance. The dispersion check on weekly counts gives `VMR = 7.5` and a statistic of 772 on 103
degrees of freedom, so equidispersion is decisively rejected.

### 6.3 The moments, by hand

```text
   horizon            P = 5.0 weeks
   expected arrivals  lambda * P = 0.48 * 5 = 2.40
   E[D]               = 2.40 * 5.00                = 12.00 kits
   Var(D)             = 2.40 * 37.50               = 90.00 kits^2
   sd(D)              = sqrt(90.00)                = 9.487 kits
   P(D = 0)           = exp(-2.40)                 = 0.0907
   CV                 = 9.487 / 12.00              = 0.791
   skewness           = lambda*P*E[Z^3] / Var^1.5
                      = 2.40 * 375 / 90^1.5        = 1.054
```

The third moment of the gamma is `θ³·k(k+1)(k+2) = 15.625 × 24 = 375` kits³. The skewness of 1.05 is
the model's statement that the distribution is asymmetric — the quantity a normal sets to zero.

### 6.4 The normal route, and its two failures

```text
   stock for 95% CSL   = 12.00 + 1.6449 * 9.487  = 27.60 kits
   P(X < 0)            = Phi(-12.00 / 9.487)    = Phi(-1.265) = 0.1030
   90th percentile     = 12.00 + 1.2816 * 9.487 = 24.16 kits
   99th percentile     = 12.00 + 2.3263 * 9.487 = 34.07 kits
```

**Failure one — impossible mass.** 10.3 per cent of the fitted probability lies at negative demand.
This is not a rounding artifact; it is the model stating that in one cycle in ten the warehouse owes
the customer minus one and a half kits.

**Failure two — the level is short.** Evaluating the true compound distribution at the normal's 95
per cent level of 27.6 kits gives a cycle service level of 0.929, not 0.95. The planner who set 95
per cent gets 92.9, and the two-point gap is permanent, distributed across items, and never large
enough in any single event to investigate. At the 99 per cent target the normal's 34.07 kits delivers
0.972 — a 1.8-point miss on a level that is itself 6.8 kits below the true 99th percentile.

### 6.5 The compound route

Computing the exact compound distribution by Panjer recursion on a grid of 0.05 kits:

| Quantity | Exact (analytic) | Computed (grid) | Difference |
|---|---:|---:|---:|
| Mean | 12.00 kits | 12.06 kits | 0.06 (0.5%) |
| Standard deviation | 9.487 kits | 9.518 kits | 0.031 (0.3%) |
| `P(D = 0)` | 0.0907 | 0.0907 | 0 |
| 50th percentile | — | 10.35 kits | — |
| 90th percentile | — | 25.05 kits | — |
| 95th percentile | — | 30.15 kits | — |
| 99th percentile | — | 40.90 kits | — |

The grid error is visible in the mean and the standard deviation and small enough to ignore at this
resolution; the check is the point, because an implementation that reproduces the analytic mean and
zero mass will produce a plausible percentile whether or not it is right. The cost of the correct
family is 2.55 kits, 9.3 per cent more stock, for the service level the normal claimed and missed.

### 6.6 Both service measures, from the same distribution

| Stock level (kits) | CSL | Expected units short per cycle | Fill rate |
|---:|---:|---:|---:|
| 24 | 0.886 | 0.813 kits | 0.933 |
| 27 | 0.923 | 0.529 kits | 0.956 |
| 30 | 0.949 | 0.339 kits | 0.972 |
| 34 | 0.971 | 0.183 kits | 0.985 |

Fill rate exceeds CSL at the same stock level, because a stock-out is weighted by how much demand it
costs; the gap is about seven points at a stock of 24 kits. A contract written on fill rate and a
policy managed on cycle service level will disagree about whether 95 per cent was achieved.

### 6.7 What aggregation does to the same item

| Bucket | Mean demand | Variance | Standard deviation | CV | `P(zero demand)` |
|---|---:|---:|---:|---:|---:|
| Week | 2.40 kits | 18.0 kits² | 4.24 kits | 1.77 | 0.619 |
| Two weeks | 4.80 kits | 36.0 kits² | 6.00 kits | 1.25 | 0.383 |
| Month (4.33 weeks) | 10.39 kits | 77.9 kits² | 8.83 kits | 0.85 | 0.125 |
| Quarter (13 weeks) | 31.20 kits | 234.0 kits² | 15.30 kits | 0.49 | 0.002 |

The mean and the variance both scale exactly, because the compound Poisson is closed under aggregation
and the rate simply multiplies. What does not scale is the shape: the CV falls with the square root of
the horizon and the zero mass collapses by orders of magnitude. An item that looks intermittent
weekly looks almost smooth quarterly, so a system re-estimating quarterly classifies it differently
from one planning it weekly, with no change in the physical world. Scaling stops altogether with
autocorrelated demand, with a bucket boundary cutting across a day-of-week pattern, or with censoring
present at one bucket and not another.

### 6.8 The simulation route and its error

A 1,000-path simulation of the same model, with seed 20261103:

| Statistic | Simulated value | Monte Carlo standard error | True value |
|---|---:|---:|---:|
| Mean | 11.84 kits | 0.30 kits | 12.00 kits |
| Standard deviation | 9.55 kits | 0.21 kits | 9.487 kits |
| 90th percentile | 24.73 kits | 0.74 kits | 25.05 kits |
| 95th percentile | 29.22 kits | 0.98 kits | 30.15 kits |
| Share of paths at zero | 0.101 | 0.009 | 0.0907 |

The mean is within one standard error of the truth. The 95th percentile is 0.93 kits low, also within
one standard error — and that error is almost a full kit, so a simulation at `M = 1,000` cannot
distinguish a correct implementation from a modestly wrong one, and "the simulated 95th percentile is
29.22" is noise quoted to two decimals. At `M = 100,000` the error falls to 0.10 kits.

### 6.9 The error budget, and the card

| Source of error | Size on the buffer | Direction | Visible in the output |
|---|---:|---|---|
| Normal family for a right-skewed item | 2.6 kits (8.6%) | Understates | No |
| Censoring in 8.7% of weeks | 3–8% on sigma | Understates | Only with availability flags |
| Autocorrelation omitted | 5–15% on the variance | Understates | Only with a correlogram |
| Simulation at `M = 1,000` | ±1.0 kit on the 95th percentile | Symmetric | Only with a replication count |
| Magnitude fitted to period totals | 20–35% on mean size | Overstates | No |

The ordering is the lesson: the family error is the largest and the easiest to fix, the censoring
error is second and needs a data decision, and the simulation error is the most reported and least
consequential.

```text
   1. Family:      compound Poisson with gamma magnitudes
   2. Parameters:  lambda = 0.48/wk [0.34, 0.62]; k = 2.0 [1.4, 2.9]; scale = 2.5 kits [1.8, 3.4]
   3. Window:      weekly, 104 weeks to 2026-09-30
   4. Censoring:   9 of 104 weeks unavailable (8.7%); availability-conditioned; flags non-seasonal
   5. Validation:  PIT uniform on 26 out-of-sample windows; 95th percentile within 1.4 kits
```

## 7. Code, query, or calculation

One item, built three ways, by hand. The arithmetic is the whole of section 6 and is reproducible on
paper.

```text
Item 4417-C, weekly bucket, P = R + L = 1 + 4 = 5 weeks, target CSL 95%
inputs: mu_d = 2.4 kits/wk   lambda = 0.48/wk   Gamma k = 2, scale = 2.5 kits
        mu_Z = 5.00 kits    E[Z^2] = 37.5 kits^2      z(95%) = 1.6449

(1) NORMAL      mean = mu_d * P = 12.00 kits    sd = sqrt(P * sigma_d^2) = 9.487 kits
    level      = 12.00 + 1.6449 * 9.487 = 27.60 kits
    P(X < 0)   = Phi(-12.00 / 9.487) = Phi(-1.265) = 0.1030   <- 10.3% on negatives
    true CSL at 27.60, evaluated on the compound law            = 0.929

(2) COMPOUND POISSON     lambda * P = 2.40 arrivals per cycle
    mean      = 2.40 * 5.00     = 12.00 kits
    variance  = 2.40 * 37.50    = 90.00 kits^2     sd = 9.487 kits
    P(D = 0)  = exp(-2.40)      = 0.0907
    skewness  = 2.40 * 375 / 90^1.5 = 1.054
    95th pct  = 30.15 kits (Panjer, grid 0.05)  ->  CSL 0.950
    mass below zero = 0   (support is 0, 0.05, 0.10, ...)

(3) SIMULATION, M = 1,000 paths, seed 20261103
    draw N ~ Poisson(2.40); draw N gamma sizes; sum; sort
    mean   = 11.84  +/- 0.30   (se = sd / sqrt(M))
    sd     =  9.55  +/- 0.21   (se ~ sd / sqrt(2(M-1)))
    95th   = 29.22  +/- 0.98   (se = sqrt(.95*.05/M) / f(q), f(30.15) = 0.0070)
    -> 1,000 paths cannot resolve the quantile to better than about one kit
```

Expected output: the normal assigns 10.3 per cent of its mass below zero and delivers 0.929 where it
promises 0.950; the compound model puts 9.1 per cent of its mass exactly at zero and delivers 0.950
at 30.15 kits; the simulation reproduces both to within its own standard error. Four checks before
trusting any implementation, including your own:

1. **Check the mean against `λLμ`.** It is identified even when the decomposition is not.
2. **Check the zero mass against `e^(−λL)`.** It tests the arrival rate in isolation.
3. **Halve the grid step.** If the 95th percentile moves by more than a tenth of a unit, the grid is
   too coarse and the tail is the part that suffers.
4. **Rerun the simulation with a second seed.** If the 95th percentile moves by more than two
   standard errors, `M` is too small.

## 8. Data, standards, and regulatory context

No standard defines the lead-time demand distribution, the choice of family, or the parameters'
estimator. There is no ISO, ANSI, or IEC document that fixes the compounding structure or the
censoring treatment. What the standards constrain is the **input**: the identity of the item, the
identity of the location, the meaning of the timestamps that define a lead time, and the definition
of the service commitment the distribution is quantiled against. Those constraints are what make a
distribution auditable, and they are the part usually missing from the method note.

### 8.1 Identity and event data

| Standard or instrument | What it governs | What it constrains here | Grade |
|---|---|---|---|
| GS1 GTIN (Global Trade Item Number) | Product identity | A pack-size change is a new GTIN, splitting one history into two short ones and biasing both fits | B |
| GS1 GLN (Global Location Number) | Location and party identity | A warehouse re-code creates a false new series of zeros | B |
| EPCIS 2.0 (GS1) | Event data: what, where, when, why | Timestamped goods issue and receipt events, the only sound basis for a realised lead-time distribution | B |
| ISO 8000 series | Data quality | Completeness of the item, location, and period keys a dense zero grid depends on | B |
| ISO 3534-1 | Statistical vocabulary | The meanings of estimator, variance, and confidence used throughout | B |
| ISO 3534-2 | Applied statistics vocabulary | The meaning of a quantile and of a tolerance interval, both used loosely in planning | B |

Lead-time measurement deserves a specific warning. An arrival rate and a lead-time distribution both
require **event** timestamps. An order line with a creation date is an intention with a date. A goods
issue, a gate-in, a receipt confirmation, or a scan is an event. If the only available data are order
dates and invoice dates, the lead time you measure is the order-to-invoice cycle, which includes the
supplier's payment terms and excludes the transit, and the resulting distribution describes neither.

### 8.2 Service commitments and metrology

| Instrument | Where | What it constrains here | Grade |
|---|---|---|---|
| Service level agreement (SLA) | Contract | Names the measure the quantile is computed for; a 95% fill target is not a 95% CSL | B |
| Incoterms 2020 | Global contract convention | Defines when risk and cost transfer, hence when the lead time starts and ends | B |
| ISO 9001:2015 clause 8.4 | Quality management, externally provided processes | Requires that criteria for selection and monitoring of external providers be defined, where a lead-time distribution belongs | B |
| ISO 16269-6 | Statistical interpretation of data — tolerance intervals | The closest formal object to a service quantile; almost never cited in planning documentation | B |
| ISO 5725-1:2023 | Accuracy of measurement methods | Defines accuracy for an instrument; does not define forecast or distribution accuracy | B |
| JCGM 200:2008 (VIM) | Metrology vocabulary | Distinguishes accuracy, precision, trueness, and bias | B |
| SCOR delivery performance metrics | Reference framework | Standard vocabulary for perfect order fulfilment; not a regulation | C |

The Incoterm point is load-bearing. Under free carrier (FCA) the buyer's lead time begins when the
supplier hands over; under delivered duty paid (DDP) it ends when the goods arrive at the buyer's
dock. The same physical lane produces two different lead-time distributions depending on the term.

### 8.3 What is convention rather than requirement

- The choice of gamma over lognormal is a convention in parts of the spare-parts literature,
  supported by closure properties and comparative studies, not by a standard.
- "Fit the magnitude to the non-zero periods" is a convention inherited from software and is wrong
  under a compound model whenever the arrival rate is not small.
- The `z` values 1.28, 1.65, and 2.33 are arithmetic. The target is policy.
- "The normal approximation is fine above a mean of ten" is a rule of thumb with a contested
  literature behind it, not a threshold anyone derived.

## 9. Industry practice and case evidence

### 9.1 How the normal became the default

The normal approximation is not a recent simplification. Babai, Syntetos and Teunter, reviewing fifty
years of inventory research, note that the normal has been the most-used lead-time demand model since
the early 1960s, through Hadley and Whitin (1963), Naddor (1975) and Wagner (1975). The reason was
computational rather than empirical: before cheap numerical integration, a normal table was the only
quantile function a planner could consult, and mean and variance were the only parameters a smoothed
forecast could supply. R. G. Brown's *Statistical Forecasting for Inventory Control* (1959) tied the
smoothed forecast error to the reorder point, with the standard deviation estimated from the
exponentially smoothed mean absolute deviation by a factor of approximately 1.25. That combination
became the architecture of the first commercial systems and remains the architecture of most of their
descendants. The early practitioners were not wrong: given a mainframe and a table, they were right.
The constraint disappeared; the architecture did not.

### 9.2 The dispute about how much the family matters

| Study | Design | Finding | Grade |
|---|---|---|---|
| Fortuin (1980), *JORS* 31(10) | (s,Q) model with Gaussian, logistic, gamma, lognormal, and Weibull lead-time demand | Numerical differences in service level were negligible across families | A |
| Tadikamalla (1984), *Omega* 12(6) | Comparison of several approximations to lead-time demand | Normal and logistic are inadequate when the CV of lead-time demand is large; lognormal, gamma, and Weibull are all adequate | A |
| Tyworth and O'Neill (1997), *NRL* 44(2) | Seven industry groups of fast-moving finished goods, demand CV near 0.40 | The normal approximation is robust for cost and service | A |
| Lau and Lau (2003), *NRL* 50(2) | Analytical and numerical, (Q,R) system | Cost penalties from the normal can be serious even at quite low CV, and the worst cases are identifiable | A |
| Rossetti and Ünlü (2011), *IJPE* 134(1) | Robustness study across lead-time demand models and demand classes | Gamma and negative binomial perform well; lognormal is comparatively weak; expected backorders are the most sensitive measure | A |
| Strijbosch and Heuts (1994), working paper | Monte Carlo in a continuous-review model | Gamma is a good approximation in many cases, but fails when the maximum lead time is short or the lead-time distribution is multimodal | C |

The apparent contradiction is not one: Fortuin measured service levels, Lau and Lau measured cost. A
family 2 per cent wrong on service can be 15 per cent wrong on cost, because cost is a derivative and
service is a level. The family matters most in a cost trade-off, above a 95 per cent target, and above
a CV of about 0.5.

### 9.3 What the correlation literature establishes

| Study | Design | Finding | Grade |
|---|---|---|---|
| Hellemans, Boute and Van Houdt (2019), *EJOR* 276(2) | Base-stock policy with Markov-modulated lead times and order crossover | Lead-time correlation leaves lead-time demand unchanged but changes the shortfall distribution; inventory variance is non-decreasing in lag-1 lead-time autocorrelation | A |
| Boute, Disney, Lambrecht and Van Houdt (2014), *EJOR* 232(1) | Analytical model with endogenous lead times | Treating lead times as independent and exogenous systematically understates safety stock | A |
| Disney, Maltz, Wang and Warburton (2016), *EJOR* 248(2) | Open-order distribution under crossover | Traditional lead-time-demand safety stock is wrong once crossover is present; minimum inventory variance does not imply minimum cost | A |
| Babai, Dai, Li, Syntetos and Wang (2021), *EJOR* | Three variance strategies under ARMA demand | Classical square-root scaling is the least accurate except under strong negative autocorrelation; under positive autocorrelation it raises cost and lowers service | A |
| Saoud, Kourentzes and Boylan (2022), *Omega* 110 | Comparison of lead-time variance approximations | One-step error variance scaled by lead time ignores correlation between forecast errors at different horizons; monitoring the empirical variance of lead-time errors performs better | A |

### 9.4 Censoring in practice

| Study | Design | Finding | Grade |
|---|---|---|---|
| Wecker (1978), *Management Science* 24(10) | Sales data with stock-outs | Ignoring censoring produces low-biased demand and prediction-error variance estimates | A |
| Nahmias (1994), *NRL* 41(6) | Lost-sales system, IID (independent and identically distributed) normal demand | Compares maximum likelihood, best linear unbiased, and a new estimator; the naive estimator is biased | A |
| Conlon and Mortimer (2013), *AEJ Micro* | Vending-machine data with controlled availability | Two biases: censoring of demand at stock-outs, and forced substitution inflating sales of available products | A |
| Mersereau (2015), *M&SOM* 17(3) | Censored observations with inventory record inaccuracy | If true inventory is unknown, stock-out timing is unknown, and ignoring that adds a systematic downward bias | A |
| Tong, Feiler and Larrick (2017), *POM* | Behavioural experiment with managers | Managers exhibit censorship bias in their beliefs, not only in their data; a recording intervention reduces it | A |

The Tong, Feiler and Larrick result adds a layer most treatments miss: the bias is not only in the
dataset but in the belief it produces. A planner who has spent a year seeing sales capped at 30 units
believes demand is about 30, and a correction must overcome that intuition as well as the arithmetic.

### 9.5 What practitioners actually do

| Practice | Observed | Consequence | Grade |
|---|---|---|---|
| Store a sigma and a `z`, not a family | Standard in material master records | The distribution is unstated and unauditable | C |
| Fit the normal by default | Dominant in ERP and spreadsheet implementations | Systematic understatement on skewed items | C |
| Scale a one-step sigma by `sqrt(L)` | Common in older planning engines | Ignores forecast-error correlation; understates by 7–30% | C |
| Ignore censoring | Common where availability flags are not joined to demand | Sigma biased down exactly where service is worst | C |
| Use one bucket across the business | Common for system simplicity | Different items need different buckets for the same statistical reason | C |

These are inferences from the systems the literature describes and from the structure of commercial
products, not an audit: no study this file could locate quantifies how many installed systems store a
sigma without a family. Treat the table as hypotheses to test in your own organisation.

## 10. Failure modes and red flags

| Failure | Red flag visible in a meeting or dataset | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Family never stated | The item card has a sigma and no distribution | The number is a quantile of an unnamed law | Cannot be audited or revised | Add family and parameters to the card |
| Normal used on a skewed item | A negative percentile in a working | Symmetric support on the whole line | 2–10 point service shortfall at 95% | CV screen; exact or simulated quantile |
| Zero mass ignored | "Demand can't be zero, it's a rate" | The model assigns no probability to a real event | Tail understated on intermittent items | Report the zero share; use a compound or hurdle law |
| Lead-time variance term dropped | Buffer equals `z · sigma_d · sqrt(P)` exactly | The horizon term is absent | Up to a factor of two on high-volume items | Compute the screening ratio; see topic 04 |
| Lead time rounded to a whole period | `L = 3.7` becomes 4 | The rate-times-horizon product moves | 5–13% of expected demand on slow movers | Scale the rate, not the horizon |
| Square-root scaling under autocorrelation | Sigma from one-step errors, buffer misses target by points | Correlation inflates the true variance | 7–30% understatement | Estimate effective variance; correlogram |
| Lead times treated as IID | No lag-1 autocorrelation ever computed | Congestion and shared modes correlate successive orders | Shortfall variance understated | Test; model the correlation or add margin |
| Sigma computed on shipments | Sigma falls after a service improvement | Shipments are smoother than demand | Buffer shrinks as service improves | Use consumption or demand; state the construct |
| Censored periods left in the window | Unusually low sigma on the worst-service items | Sales truncated at available stock | Under-buffer where service already fails | Report the censored share; reconstruct |
| Censored periods simply deleted | The exclusion is described as "cleaning" | Deletes the upper tail | Understates variance more than leaving them in | Use a reconstruction method; state it |
| Magnitude fitted to period totals | The fit reports `n` equal to the period count | Period totals are compound sums | Mean size overstated by 20–35% | Fit to events, or fit the compound likelihood |
| Simulation reported without `M` | A two-decimal percentile from a laptop run | Quantile standard error is large | False precision; model comparison invalid | Report `M` and the standard error |
| Poisson applied without a dispersion check | VMR of 7 and a Poisson fit | Variance equals mean is false | Tail far too thin | Dispersion test; NB or compound |

Two deserve expansion. **Censored periods simply deleted** looks like good practice, because
exclusion is a recognisable data-quality action; it removes exactly the observations that carry the
upper tail and leaves a series too clean to be true. **Lead times treated as IID** survives longest,
because the item card has one lead-time field and no place to put a correlation, so the question is
never asked.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks if it fails | How to test it |
|---|---|---|---|
| The horizon total is a random sum of independent increments | Gives the variance identity | Positive correlation inflates the variance; the buffer is short | Autocorrelation at lags 1 to `P − 1`; effective variance ratio |
| The lead time is independent of demand intensity | Allows the two variance terms to be added without a covariance | A supplier that slows when demand rises leaves a missing positive term | Correlation between demand level and realised lead time |
| Lead times are independent across orders | Justifies one fitted distribution per lane | Persistent lateness makes the shortfall more variable than the model says | Lag-1 autocorrelation of successive lead times |
| The count process has independent increments | Poisson and NB increments are independent | Clustering, seasonality, and single-customer effects bias the rate | Runs test on zero/non-zero; interarrival histogram |
| The magnitude law is right in the upper tail | Every target above 80% is a tail statement | The quantile is wrong by several units despite a good overall fit | Tail-weighted goodness of fit; P-P plot above the 90th percentile |
| Magnitudes are IID over time | One size law describes all events | A quarterly batch buyer makes the size law multimodal | Autocorrelation of sizes; Ljung–Box |
| The parameters are stationary over the window | The fitted law describes the future | A listing change, competitor exit, or redesign shifts them | Split-window comparison; structural break test |
| Demand is observed, not censored | The parameters must describe the market, not the stock policy | Sigma biased down; the buffer reproduces the stock-out | Availability flags; censored share |
| The bucket is the decision period | Parameters are properties of the bucket | Shape changes with aggregation | Compare fit statistics across buckets |
| The discretisation is fine relative to the tail | Recursion and FFT accuracy | The 99th percentile carries the grid error | Halve the step and compare the target quantile |

Three limits deserve statement. **The distribution describes a process, not an item.** It says the
history is consistent with a family and a set of parameters; it does not say the future will be, and
the parameters most likely to move first are the ones governing the tail. **The family choice is not
identified by the data.** Two families can fit a hundred periods equally well and disagree by several
units at the 99th percentile, because the tail is where the data are thinnest and the decision lives.
That is why validation must be out of sample and at the decision horizon. **A distribution is not a
policy.** It is one input to a decision that also needs a review period, a replenishment rule, a cost
structure, and a service definition, and errors in any of those can dominate the distributional error.

## 12. Exercises

### Exercise 1 — Moments by hand (calculation, 35 minutes)

An item has weekly demand with `λ = 0.9` events per week and gamma event sizes with mean 6.0 units
and second moment 62.0 units². The review period is one week, the lead time is four weeks and
deterministic, and the target is a 95 per cent cycle service level.

**Deliverable:** horizon mean, variance, standard deviation, zero mass, and stock level, each with
units. **Acceptance criteria:** the identity written before substitution; the stock level rounded up
with the direction stated. **Time:** 35 minutes.

### Exercise 2 — Is the normal admissible (critique, 30 minutes)

Four items, all planned weekly with a five-week horizon.

| Item | Mean over horizon (units) | Standard deviation (units) | Zero share of cycles |
|---|---:|---:|---:|
| A | 96 | 19 | 0.00 |
| B | 42 | 25 | 0.00 |
| C | 14 | 11 | 0.06 |
| D | 3 | 3.4 | 0.42 |

**Deliverable:** the CV, the implied `P(X < 0)` under a normal fit, and a one-line verdict for each.
**Acceptance criteria:** the CV to two decimals; the verdict names the diagnostic deciding it. **Time:**
30 minutes.

### Exercise 3 — The dispersion decision (application, 40 minutes)

A count series has 78 periods, a mean of 3.6 events per week, a variance of 11.2, and a lag-1
autocorrelation of 0.31 in the counts.

**Deliverable:** the VMR, the dispersion statistic with its degrees of freedom, a family
recommendation, and what the autocorrelation does to the test. **Acceptance criteria:** the statistic
computed; the recommendation justified by the direction of the departure. **Time:** 40 minutes.

### Exercise 4 — Gamma or lognormal (analysis, 40 minutes)

Two magnitude laws are fitted to 240 event sizes: gamma with `k = 1.8`, `θ = 4.2`; and lognormal with
`μ_log = 1.75`, `σ_log = 0.72`. Both reproduce the sample mean.

**Deliverable:** the mean, standard deviation, CV, and skewness of each, and a recommendation with
the diagnostic you would run first. **Acceptance criteria:** gamma skewness `2/√k`; lognormal skewness
`(e^{σ²} + 2)·√(e^{σ²} − 1)`; the recommendation tied to the target level. **Time:** 40 minutes.

### Exercise 5 — Correct for correlation (calculation, 40 minutes)

Weekly demand has `σ = 12` units and autocorrelations `ρ(1) = 0.35`, `ρ(2) = 0.20`, `ρ(3) = 0.10`,
`ρ(4) = 0.05`. The protection period is five weeks.

**Deliverable:** the effective variance, the ratio to the independent case, and the corrected buffer
at a 95 per cent CSL. **Acceptance criteria:** each lag term shown with its weight `(P − k)`; the ratio
stated as a percentage. **Time:** 40 minutes.

### Exercise 6 — Detect and treat censoring (critique, 45 minutes)

A distributor's history shows 21 of 104 weeks with at least one day out of stock. The all-week
estimate gives mean 41.0 units and `σ = 12.5`; the available-week estimate gives mean 45.8 and
`σ = 15.1`. The planning system uses the all-week figures.

**Deliverable:** the buffers implied by both, the percentage difference, the censored share, and a
recommendation that names its risk. **Acceptance criteria:** the direction of the bias explained
mechanically; the share compared with the regime where corrections fail. **Time:** 45 minutes.

### Exercise 7 — Aggregation audit (design, 45 minutes)

A European distributor plans 4,000 slow-moving items monthly, but the items are replenished weekly
and the supplier lead time is three weeks. The parameters in the system were estimated on monthly
demand.

**Deliverable:** a one-page audit identifying the horizon and bucket mismatches, the direction of each
bias, the parameters to re-estimate, and the order. **Acceptance criteria:** each mismatch expressed in
weeks or units, not as a label; one bias quantified from section 6.7. **Time:** 45 minutes.

### Exercise 8 — Write the specification (design, 40 minutes)

Write the five-line distribution card from section 6.9 for a real or constructed item, every line
filled and every number carrying a unit.

**Deliverable:** the completed card plus a note on which line was hardest to fill and what that
reveals. **Acceptance criteria:** all five lines present; line 2 carries intervals; line 4 states the
share and the treatment; line 5 an out-of-sample result. **Time:** 40 minutes.

## 13. Self-check answers

### Exercise 1

```text
   lambda * P       = 0.9 * 5              = 4.50 arrivals
   E[D]             = 4.50 * 6.00          = 27.00 units
   Var(D)           = 4.50 * 62.00         = 279.00 units^2
   sd(D)            = sqrt(279.00)         = 16.70 units
   P(D = 0)         = exp(-4.50)           = 0.0111
   stock at 95%     = 27.00 + 1.6449*16.70 = 27.00 + 27.47 = 54.47 -> 55 units
```

The CV is 0.62, so a normal fit places about 5 per cent of its mass below zero against a true zero
mass of 1.1 per cent: borderline rather than indefensible, the honest reading of a CV near 0.6.
Rounding up is correct because the service constraint is a lower bound on the quantile.

### Exercise 2

```text
A: CV = 19/96 = 0.20    P(X<0) = Phi(-5.05) = 0.0000    normal admissible
B: CV = 25/42 = 0.60    P(X<0) = Phi(-1.68) = 0.046    borderline; check skewness
C: CV = 11/14 = 0.79    P(X<0) = Phi(-1.27) = 0.102    not admissible; use exact law
D: CV = 3.4/3 = 1.13    P(X<0) = Phi(-0.88) = 0.189    indefensible; 42% of cycles are zero
```

Item D is the one to report, because 42 per cent of its cycles have zero demand and a normal assigns
zero probability to that event. Item A is the only one where the normal is clearly safe. Item B sits
where the literature disagrees: Tyworth and O'Neill would accept it, Lau and Lau would look for cost.

### Exercise 3

```text
   VMR        = 11.2 / 3.6 = 3.11
   dispersion = 77 * 11.2 / 3.6 = 239.6  on 77 df
   => overdispersion; Poisson rejected
```

A VMR of 3.1 is well above the practical threshold near 1.5 and the statistic is more than three times
its degrees of freedom, so a negative binomial is the natural count family. The caveat is decisive: a
lag-1 autocorrelation of 0.31 violates the independence both Poisson and NB increments assume and
inflates the dispersion statistic, so test for a level shift first, segment if one exists, and only
then fit. Absent segmentation, use the NB and report the autocorrelation as unmodelled dependence that
makes the fitted variance too small.

### Exercise 4

```text
Gamma k=1.8, theta=4.2
   mean = 7.56      sd = 4.2*sqrt(1.8) = 5.635      CV = 0.745
   skew = 2/sqrt(1.8) = 1.491

Lognormal mu_log=1.75, sigma_log=0.72
   mean = exp(1.75 + 0.72^2/2) = exp(2.0092) = 7.457
   sd   = mean * sqrt(exp(0.5184) - 1) = 7.457 * 0.8096 = 6.037
   CV   = 0.810
   skew = (exp(0.5184) + 2) * sqrt(exp(0.5184) - 1) = 3.679 * 0.8096 = 2.978
```

The means are close, but the lognormal has twice the skewness and a heavier tail, so at a 99 per cent
target it produces a larger quantile. The first diagnostic is the upper decile: if the largest
observed sizes sit on the gamma's tail, the lognormal is producing implausible extremes.

### Exercise 5

```text
   Var = sigma^2 * [ P + 2 * sum (P-k) * rho(k) ]
       = 144 * [ 5 + 2*(4*0.35 + 3*0.20 + 2*0.10 + 1*0.05) ]
       = 144 * [ 5 + 2*(1.40 + 0.60 + 0.20 + 0.05) ]
       = 144 * [ 5 + 4.50 ] = 144 * 9.50 = 1,368
   sd  = 36.99 units        (independent case: 12 * sqrt(5) = 26.83)
   ratio = 1.90 in the variance, 1.38 in the standard deviation
   buffer at 95% = 1.6449 * 36.99 = 60.9 units
```

The correction nearly doubles the variance. At a 30-week horizon the weights `(30 − k)` are much
larger, so even a small `ρ` produces a large multiplier. Because the weights decline linearly, only
the first few lags matter much, which is why estimating `ρ(1)` to `ρ(3)` well is usually enough.

### Exercise 6

```text
   all weeks:        buffer = 1.6449 * 12.5 * sqrt(5) = 45.97 -> 46 units
   available weeks:  buffer = 1.6449 * 15.1 * sqrt(5) = 55.53 -> 56 units
   difference:       10 units, 21.7 per cent
   censored share:   21 / 104 = 20.2 per cent
```

The bias is downward because observed sales are `min(demand, stock)`, which truncates the upper tail
and removes exactly the weeks that carry the variance. At 20.2 per cent censoring the correction is
inside the regime where the standard adjustments hold — degradation sets in near half the periods.
The risk is that the available weeks may be unrepresentative: if stock-outs cluster in peaks, the
corrected sigma overstates. Test the flags for seasonality first.

### Exercise 7

The horizon is not the issue: a three-week lead time plus a one-week review gives a four-week horizon,
which is a weekly decision problem wearing a monthly coat. The bucket is the issue, in two places.
**Parameter mismatch:** parameters fitted to monthly demand describe monthly variability, and the
monthly CV of a slow mover is smaller than its weekly CV by roughly the square root of the bucket
ratio — 0.49 against 1.77 in section 6.7 — so applying the monthly CV to a weekly decision understates
the weekly tail. **Zero-mass mismatch:** the share of zero-demand periods falls from 62 per cent
weekly to 12 per cent monthly, so a classification rule using the zero share places the item in a
different class at each bucket, and the class determines the policy family. **Order of work:**
re-estimate at the weekly bucket first, then re-run the classification, then re-derive the buffers.

### Exercise 8

A strong card has five lines and no unspecified numbers. The hardest is almost always line 4, the
censoring treatment, because it requires joining an availability history to a demand history never
designed to be joined, and because the answer depends on a judgement — exclude, condition, project, or
accept — most organisations have never recorded. "Censoring: handled" has not filled the line; "9 of
104 weeks unavailable (8.7%); availability-conditioned mean; flags non-seasonal" has. The second
hardest is line 5, because out-of-sample validation needs historical forecasts that many systems
overwrite.

## 14. Interview questions

**1. "What is lead-time demand, and why is it not a rate times a horizon?"**

A strong answer describes it as a random sum of a random number of random quantities, draws the two
levels, and gives the variance identity with both terms. A weak answer gives a mean and calls the rest
variability.

**2. "Derive the variance of a random sum."**

A strong answer writes `Var(S) = E[Var(S|N)] + Var(E[S|N])`, substitutes, arrives at
`E[N]Var(Z) + Var(N)(E[Z])²`, and says which term dominates for which kind of item. A weak answer
recites the compound Poisson formula without knowing it is a special case.

**3. "When is a normal approximation to lead-time demand acceptable?"**

A strong answer gives the CV screen with the `P(X<0)` table, adds the zero mass and skewness as
diagnostics, notes that the CV of the horizon total falls as the horizon lengthens, and mentions that
the literature disagrees at moderate CV. A weak answer says "at high volume" with no diagnostic.

**4. "What does a zero in the demand history tell you that a mean does not?"**

A strong answer says the zero share identifies the arrival rate under a Poisson assumption, that a
normal assigns zero probability to the event, and that the zero mass is the cheapest diagnostic for
choosing between a normal and a compound law. A weak answer calls zeros missing data.

**5. "Poisson or negative binomial — how do you decide?"**

A strong answer computes the VMR and the dispersion statistic with its degrees of freedom, interprets
the direction of the departure, notes that underdispersion rules out both, and adds that a level shift
produces spurious overdispersion. A weak answer says the negative binomial is "more flexible".

**6. "What is the difference between a compound Poisson and a hurdle model?"**

A strong answer says the compound Poisson has a continuous clock and an arrival rate, the hurdle a
discrete clock and an occurrence probability, they coincide at the fitted bucket when `p = 1 − e^(−λ)`,
and they diverge at a longer horizon. A weak answer says one is a special case without saying which.

**7. "How many simulation replications do you need?"**

A strong answer gives `sqrt(q(1−q)/M)/f(q)` for the quantile, notes the density is the term that
matters, gives a concrete number, and contrasts it with the faster convergence of the mean. A weak
answer says ten thousand because the tutorial did.

**8. "What is wrong with scaling a one-period sigma by the square root of the lead time?"**

A strong answer says the rule assumes zero autocorrelation, gives the general expression with the
lag-weighted correlation terms, and says the rule understates for positively correlated demand. A
weak answer says it is approximately right.

**9. "Do correlated lead times change lead-time demand?"**

A strong answer says no, they change the shortfall distribution, which is what the buffer covers, and
that inventory variance is non-decreasing in the lag-1 lead-time autocorrelation. A weak answer says
correlation always increases variability.

**10. "A stock-out removes the demand you were trying to measure. What do you do?"**

A strong answer describes the truncation, the downward bias in sigma, the direction of the consequent
error, the reconstruction options, and the regime in which reconstruction fails. A weak answer says
exclude the period, which deletes the upper tail.

## 15. Key terms

- **Lead-time demand** — demand accumulated over the protection period; the object every buffer is a
  quantile of.
- **Protection period** — review period plus lead time.
- **Random sum** — a sum of a random number of random quantities.
- **Random horizon** — a horizon whose length is itself random.
- **Compound distribution** — the law of a random sum.
- **Count distribution** — the law of how many periods or events fall in the horizon.
- **Magnitude distribution** — the law of the quantity in one event; the size law.
- **Compound Poisson** — Poisson arrivals carrying independent random sizes.
- **Hurdle model** — Bernoulli occurrence plus a zero-truncated size law, one event per period.
- **Arrival rate (λ)** — expected events per period.
- **Occurrence probability (p)** — probability a period contains at least one event.
- **VMR** — variance-to-mean ratio; 1 for Poisson.
- **Index of dispersion** — `(n − 1)s²/x̄`, chi-square under Poisson.
- **Overdispersion** — variance exceeds the mean; the signature of compounding.
- **Dispersion test** — a test of variance equals mean.
- **Gamma distribution** — right-skewed magnitude family with shape `k` and scale `θ`.
- **Lognormal distribution** — heavier-tailed magnitude family.
- **Skewness** — standardised third moment; the quantity the normal sets to zero.
- **Zero mass** — `P(D = 0)`; a probability the normal cannot represent.
- **Convolution** — the law of a sum of independent variables.
- **Panjer recursion** — a linear-time recursion for the compound sum.
- **FFT convolution** — a fast numerical compound sum via characteristic functions.
- **Monte Carlo standard error** — the standard deviation of a simulated statistic across paths.
- **Censoring** — observed demand truncated at available stock.
- **Reconstruction** — estimating unconstrained demand from censored observations.
- **Temporal aggregation** — combining periods into coarser buckets.
- **Effective variance** — variance adjusted for serial correlation.
- **Square-root-of-time rule** — the independence case of the effective variance.
- **CV** — coefficient of variation, the standard deviation over the mean.
- **PIT** — probability integral transform; uniform when the distribution is calibrated.

## 16. Sources

**Peer-reviewed and academic (A)**

- Babai, M. Z., Dai, Y., Li, H., Syntetos, A., and Wang, X. (2021). "Forecasting of lead-time demand
  variance: Implications for safety stock calculations." *European Journal of Operational Research*.
  DOI: 10.1016/j.ejor.2021.04.017. (A)
- Babai, M. Z., Syntetos, A. A., and Teunter, R. H. (2026). "Fifty years of inventory research from
  a forecasting perspective." *European Journal of Operational Research*, 331(1), 1–20. (A)
- Boute, R. N., Disney, S. M., Lambrecht, M. R., and Van Houdt, B. (2014). "Coordinating lead times
  and safety stocks under autocorrelated demand." *European Journal of Operational Research*,
  232(1), 52–63. (A)
- Conlon, C. T., and Mortimer, J. H. (2013). "Demand estimation under incomplete product
  availability." *American Economic Journal: Microeconomics*. NBER working paper 14315. (A)
- Disney, S. M., Maltz, A., Wang, X., and Warburton, R. D. H. (2016). "Inventory management for
  stochastic lead times with order crossovers." *European Journal of Operational Research*, 248(2),
  473–486. (A)
- Eppen, G. D., and Martin, R. K. (1988). "Determining Safety Stock in the Presence of Stochastic
  Lead Time and Demand." *Management Science*, 34(11), 1380–1390. (A)
- Fortuin, L. (1980). "Five Popular Probability Density Functions: A Comparison in the Field of
  Stock-Control Models." *Journal of the Operational Research Society*, 31(10), 937–942.
  DOI: 10.1057/jors.1980.169. (A)
- Hellemans, T., Boute, R. N., and Van Houdt, B. (2019). "Analysis of lead time correlation under a
  base-stock policy." *European Journal of Operational Research*, 276(2), 519–535. (A)
- Lau, A. H.-L., and Lau, H.-S. (2003). "Nonrobustness of the normal approximation of lead-time
  demand in a (Q,R) system." *Naval Research Logistics*, 50(2), 149–166. DOI: 10.1002/nav.10053. (A)
- McFadden, F. R. (1972). "On Lead Time Demand Distributions." *Decision Sciences*.
  DOI: 10.1111/j.1540-5915.1972.tb00539.x. (A)
- Mersereau, A. J. (2015). "Demand Estimation from Censored Observations with Inventory Record
  Inaccuracy." *Manufacturing & Service Operations Management*, 17(3). (A)
- Nahmias, S. (1994). "Demand estimation in lost sales inventory systems." *Naval Research
  Logistics*, 41(6), 739–757. (A)
- Prak, D., Teunter, R., and Syntetos, A. (2017). "On the calculation of safety stocks when demand
  is forecasted." *European Journal of Operational Research*, 256(2), 454–461. (A)
- Rossetti, M. D., and Ünlü, Y. (2011). "Evaluating the robustness of lead time demand models."
  *International Journal of Production Economics*, 134(1), 159–176. (A)
- Saoud, P., Kourentzes, N., and Boylan, J. E. (2022). "Approximations for the Lead Time Variance."
  *Omega*, 110, 102614. (A)
- Tadikamalla, P. R. (1984). "A comparison of several approximations to the lead time demand
  distribution." *Omega*, 12(6), 575–581. DOI: 10.1016/0305-0483(84)90060-4. (A)
- Tong, J., Feiler, D., and Larrick, R. P. (2017). "A Behavioral Remedy for the Censorship Bias."
  *Production and Operations Management*. DOI: 10.1111/poms.12823. (A)
- Tyworth, J. E. (2025). "The Gamma Distribution and Inventory Control: Disruptive Lead Times Under
  Conventional and Nonclassical Conditions." *Logistics*, 9(2), 67. (A)
- Tyworth, J. E., and O'Neill, L. (1997). "Robustness of the normal approximation of lead-time
  demand in a distribution setting." *Naval Research Logistics*, 44(2), 165–186. (A)
- Wecker, W. E. (1978). "Predicting Demand from Sales Data in the Presence of Stockouts."
  *Management Science*, 24(10), 1043–1054. (A)

**Books and reference works (A−/B)**

- Rossetti, M. D. *Analysis of Inventory Systems*, Appendix C, "Common Probability Distributions".
  Open textbook, https://rossetti.github.io/RossettiInventoryBook/. The `P(X < 0)` table by CV and
  the VMR selection rule. (B)
- Hadley, G., and Whitin, T. M. (1963). *Analysis of Inventory Systems*. Prentice-Hall. (A−)
- Brown, R. G. (1959). *Statistical Forecasting for Inventory Control*. McGraw-Hill. (A−)

**Standards and official (B)**

- GS1 General Specifications, current edition, including GTIN, GLN, and EPCIS 2.0. (B)
- ISO 8000 series, *Data quality*. (B)
- ISO 3534-1, *Statistics — Vocabulary and symbols — Probability and general statistical terms*. (B)
- ISO 3534-2, *Statistics — Vocabulary and symbols — Applied statistics*. (B)
- ISO 5725-1:2023, *Accuracy (trueness and precision) of measurement methods and results*. (B)
- ISO 16269-6, *Statistical interpretation of data — Determination of statistical tolerance
  intervals*. (B)
- ISO 9001:2015, clause 8.4, control of externally provided processes, products and services. (B)
- JCGM 200:2008, *International Vocabulary of Metrology*, 3rd edition. (B)
- Incoterms 2020, International Chamber of Commerce. (B)

**Professional and teaching (C)**

- Strijbosch, L. W. G., and Heuts, R. M. J. (1994). "Investigating several alternatives for
  estimating the lead time demand distribution in a continuous review inventory model." Working
  paper. (C)

**Note on grading and unverified items.** Grades follow the programme scheme: A peer-reviewed or
audited, B regulatory or official, C conference paper, teaching material, or company case, D vendor
documentation or marketing, E press or anecdote. Three flags are needed. First, Babai, Syntetos and
Teunter (2026) is cited from a search summary; its claim that the normal has dominated lead-time
demand modelling since the early 1960s rests on its reading of Hadley and Whitin (1963), Naddor
(1975) and Wagner (1975), and should be checked against the published text. Second, the account of
Brown (1959) and the 1.25 × MAD convention was assembled from secondary descriptions. Third, no
source consulted quantifies how many commercial systems store a sigma without a family; the table in
section 9.5 is labelled inference and graded C.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Immediate predecessor: [`04_safety-stock.md`](04_safety-stock.md) — the buffer formula consuming
  this distribution.
- Service measures: [`05_service-metrics.md`](05_service-metrics.md) — CSL and fill rate as
  functionals of it.
- Policy families: [`06_policy-families.md`](06_policy-families.md) — the rules that turn a quantile
  into an order.
- Cost framing: [`01_cost-service-frontier.md`](01_cost-service-frontier.md) — the marginal cost of
  the next point of service.
- Single-period case: [`03_newsvendor.md`](03_newsvendor.md) — the quantile with no replenishment.
- Distribution prerequisite: [`../08_hierarchical_intermittent_probabilistic_forecasting/06_compound-poisson-hurdle.md`](../08_hierarchical_intermittent_probabilistic_forecasting/06_compound-poisson-hurdle.md)
  — compound Poisson and hurdle models, cited here and extended to the lead-time dimension.
- Probabilistic forecasting: [`../08_hierarchical_intermittent_probabilistic_forecasting/07_probabilistic-forecasts.md`](../08_hierarchical_intermittent_probabilistic_forecasting/07_probabilistic-forecasts.md)
  — what a predictive distribution is for.
- Aggregation: [`../08_hierarchical_intermittent_probabilistic_forecasting/02_aggregation-consistency.md`](../08_hierarchical_intermittent_probabilistic_forecasting/02_aggregation-consistency.md)
  — temporal and cross-sectional aggregation, and the closure property.
- Demand signal: [`../09_demand_sensing_promotions_npi/01_demand-sensing-signals.md`](../09_demand_sensing_promotions_npi/01_demand-sensing-signals.md)
  — the construct you are estimating from.
- Later in this module: [`08_multi-echelon.md`](08_multi-echelon.md),
  [`09_pooling-postponement.md`](09_pooling-postponement.md),
  [`10_segmentation-shelf-life.md`](10_segmentation-shelf-life.md) — lead-time demand across
  echelons, pooling, and segmentation.
- Part II map: [`../_index.md`](../_index.md)
- Part I map: [`../../01_foundations/_index.md`](../../01_foundations/_index.md)
- Survey file: [`../../../course/10_inventory_optimization_multi_echelon.md`](../../../course/10_inventory_optimization_multi_echelon.md)
