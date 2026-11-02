> Module 01A — Foundations: Units, Time, and Assumptions
> Topic 11 of 11
> Estimated study time: 3.5 hours
> Prerequisites: 01A.05, 01A.09, 01A.10

# Demand and Lead-Time Distributions

---

## 1. Why this topic exists

Every inventory policy is a statement about a distribution, whether or not anyone wrote it down.
The reorder point is a quantile. The safety stock is the distance between that quantile and the
mean. The fill rate is an expectation over a tail. When a planner says "we hold four weeks of
cover", they have made a bet about the shape of lead-time demand, and the bet is usually
implicit, untested, and normal by default.

**One.** A spare-parts distributor holds a part whose daily demand is zero on 57% of days and
occasionally arrives in batches because a maintenance team replaces a set at once. The planner
applies `safety stock = z x sigma x sqrt(L)` with a daily standard deviation of 2.5 units and a
ten-day lead time. At a 95% target the buffer is `1.645 x 2.5 x sqrt(10) = 13.0` units. Realised
service is near 71%. No number was mistyped. The formula assumed a symmetric distribution on the
whole real line; the item has a point mass at zero and a right tail made of campaigns.

**Two.** An analyst fits a gamma to daily demand containing 55% zeros. The maximum likelihood
fit either fails, because the log-likelihood contains `ln(0)`, or it is dominated by the zero
mass and returns a shape with no operational meaning. A gamma has no mass at zero. This is a
specification error, not a numerical one, and no amount of optimisation repairs it.

**Three.** A buyer notices that lead times lengthen in the quarter when demand peaks, because the
supplier's line is loaded and the carrier is congested. The safety-stock model assumes the two
are independent. At a correlation of 0.5 between the demand rate and the lead time, the variance
of lead-time demand rises by about 39% and the 95% buffer by about 18%. The formula is used
correctly; the world it describes is wrong.

Each is a distribution failure, invisible in the arithmetic: mean, standard deviation, and service
factor are all right, and the answer is wrong.

**Why this topic closes the module.** Topic 05 gave lead time a distribution. Topic 10 gave
intermittent demand a two-process structure. This topic puts the two together. Lead-time demand
is a random sum: a random number of demand events, each of a random size, accumulated over a
random duration. That object is where units, time buckets, assumptions, and error metrics meet.
Get it wrong and every downstream quantity inherits the error: the reorder point in Module 10,
the prediction interval in Modules 07 and 08, the supplier buffer in Module 14, and the stress
scenario in Module 20.

**The claim.** A distribution is a model of a mechanism, not a curve that looks acceptable on a
histogram. The choice must be justified by the support of the quantity, the mechanism that
generates it, the horizon over which it accumulates, and the part of the distribution the
decision uses. The last is almost always the upper tail, and the upper tail is where every
convenient assumption is weakest.

## 2. Learning outcomes

By the end of this topic you can:

1. **State**, for the normal, gamma, lognormal, Poisson, negative binomial, and compound Poisson
   families, the mechanism each models, its support, and the units of every parameter.
2. **Derive** the mean and variance of lead-time demand for fixed and random lead time, showing
   the units of each term.
3. **Explain** what the central limit theorem (CLT) licenses in planning and **name** three
   failures: a short horizon, a low event count, and dependence or a random number of terms.
4. **Quantify** normal-approximation error at the 95th and 99th percentiles for skewed lead-time
   demand, and explain why the shortage error exceeds the quantile error.
5. **Distinguish** the quantile (risk function) view from the expected-shortfall view, and
   **compute** cycle service level, fill rate, and expected units short from one fitted
   distribution.
6. **Fit** by the method of moments (MM) and by maximum likelihood estimation (MLE), state which
   suits the sample, and name the failure of each.
7. **Detect** overdispersion, zero inflation, and censoring before fitting, and choose a model
   that respects the support.
8. **Test** demand-lead-time independence and **compute** the change in safety stock when the
   correlation is non-zero.
9. **Write** a distribution specification: quantity, unit, horizon, family, parameters with units,
   fitting method, window, tail validation, and service measure.
10. **Critique** a claimed service improvement that does not state the distribution, the service
    measure, or the estimation window.


## 3. The core mental model

Hold three pictures. The first is a family tree of mechanisms, not of curves. The second is a
timeline in which demand events arrive at random and the lead time decides how many count. The
third is a pair of questions — how often, and how much — that the same distribution answers
differently.

### 3.1 A family is a mechanism, not a curve

A histogram can be fitted by many families. Only one is a plausible description of how the number
was generated. The table below lets you reject a family before you fit it.

| Family | Mechanism it models | Support | Parameters and units | What it is not |
|---|---|---|---|---|
| Normal | A sum of many small independent finite-variance effects | All real numbers | Mean `mu` in units; SD `sigma` in units | A model of counts, or of a positive quantity near zero |
| Gamma | A sum of exponential waits; a positive right-skewed quantity with a floor at zero | `[0, infinity)` | Shape `k` dimensionless; scale `theta` in units, or rate in 1/units | A model of a count with mass at zero |
| Lognormal | A product of many small multiplicative effects | `(0, infinity)` | `mu_ln`, `sigma_ln` in log units, dimensionless | A model of a sum, or of an exact zero |
| Poisson | Counts of rare events in a fixed interval at a constant rate | `{0, 1, 2, ...}` | Rate `lambda` in events per interval | A model of event size |
| Negative binomial | A Poisson count whose rate is gamma-distributed | `{0, 1, 2, ...}` | Mean in events; dispersion `k` dimensionless | A model of a continuous quantity |
| Compound Poisson | A Poisson number of events, each with a random size | `{0, 1, 2, ...}` in size units | `lambda` per interval; size moments in units and units squared | A model of a smooth rate, or of a single large order |

**Support is a hard test.** If the data contain zeros, gamma and lognormal cannot have generated
them, whatever a goodness-of-fit statistic says. If the data are integer counts, a continuous
distribution is an approximation at best, and it matters at the low end. Rejecting on support
costs nothing and prevents the most common specification error in the field.

### 3.2 Lead-time demand is a random sum

The object at the centre of this topic is not a product. It is a sum with a random number of
terms.

```text
LEAD-TIME DEMAND IS A RANDOM SUM, NOT A RATE TIMES A DURATION

  day        1     2     3     4     5     6     7     8
             |     |     |     |     |     |     |     |
  demand     0     3     0     1     0     0     4     0    units
             |     |     |     |     |     |     |     |
             +-----+-----+-----+-----+-----+-----+-----+
             <------------ lead time L = 7 days --------->
  Y = 0 + 3 + 0 + 1 + 0 + 0 + 4 = 8 units
  but L is random, so  Y = X_1 + X_2 + ... + X_N(L)
  where N(L) is the number of demand events in L days.
```

The moments follow from the two-stage structure, with event sizes independent and identically
distributed (iid): condition on the lead time, take the moments of the conditional sum, then
average over the lead-time distribution.

```text
  N(L) ~ Poisson(lambda x L)          N is a count, dimensionless
  X_i  ~ size distribution, iid       X is in units
  lambda                              events per day
  L                                   days

  E[Y]   = E[L] x lambda x E[X]

  Var[Y] = E[L] x lambda x E[X^2]  +  Var[L] x (lambda x E[X])^2
           \_____________________/     \________________________/
            expected conditional        variance of the
            variance                    conditional mean
```

Both terms are in units squared, which is the check that they may be added. The first is the
demand-variability term; the second is the lead-time term. In Topic 05 the second dominated for
an imported item. In the worked example the first dominates for a regional item with lumpy
demand. Neither is universal; the ratio is a property of the item and must be computed.

> **Note:** This assumes demand events are independent of lead time and the event rate is
> constant within the lead time. Section 6.8 relaxes the first. Section 11 explains why the second
> fails when orders cross.

### 3.3 Four shapes at the same mean

The panels are schematic, not to scale. They share a mean of 100 units; the marker is the 95th
percentile, the quantity a safety-stock calculation actually uses.

```text
FOUR FAMILIES AT THE SAME MEAN (100 UNITS), 95TH PERCENTILE MARKED

   NORMAL, CV 0.50            GAMMA k=4, CV 0.50
   |       .-''''-.           |        _.-'
   |     .'        '.         |      .'    '.
   |    /            \        |     /        '.
   |   /              \       |    /           '.
   |  /                \      |   /              '.
   +--|----|----|----|--+     +--|----|----|----|--+
      0   100  182   250        0   100  194   250
              ^ p95                     ^ p95

   LOGNORMAL, CV 0.50         EXPONENTIAL, CV 1.00
   |       .-'''-.            | *
   |     .'       '.          |  *
   |    /           \         |   *
   |   /             '.       |    *
   |  /                '.     |     *  *
   +--|----|----|----|--+     +--|----|----|----|--+
      0   100  194   250        0   100  300   450
              ^ p95                     ^ p95
```

| Family | Parameters | Median | 90th | 95th | 99th | Skewness |
|---|---|---:|---:|---:|---:|---:|
| Normal | `mu = 100`, `sigma = 50` units | 100.0 | 164.1 | 182.2 | 216.3 | 0.00 |
| Gamma | `k = 4`, `theta = 25` units | 91.8 | 167.0 | 193.8 | 251.1 | 1.00 |
| Lognormal | `mu_ln = 4.494`, `sigma_ln = 0.472` | 89.4 | 163.9 | 194.5 | 268.4 | 1.63 |
| Exponential | `theta = 100` units | 69.3 | 230.3 | 299.6 | 460.5 | 2.00 |

**The tail diverges.** At the 99th percentile the spread is 216 to 461 units, a factor of 2.1.
The exponential is shown at a coefficient of variation (CV) of 1.0 because at CV 0.5 it collapses
toward the gamma; the label is part of the point. Two distributions that look alike where you
have data can differ by more than 100 units where you make the decision.

### 3.4 What the central limit theorem does and does not license

The CLT is why the normal approximation is defensible at all. It is also the most over-extended
theorem in planning.

**The horizon is short.** The CLT's `n` is the number of terms in the sum. For daily demand over a
6-day lead time, `n = 6`. Six is not infinity. At 30 days, `n = 30`, better but not decisive when
the daily distribution is itself skewed.

**The count is low.** With intermittent demand the relevant `n` is demand events, not days.
Thirty days at 0.2 events per day contains about six events. A sum of six lumpy draws is not
normal.

**The number of terms is random.** When lead time varies, `Y` is a mixture of sums of different
lengths. A mixture of normals with different means and variances is not normal. If the lead-time
CV is small the mixture is close to normal; if it is large the mixture can be multimodal and its
tail heavier than any component's.

**The terms are dependent.** Promotions, paydays, weather, and campaigns cluster demand, reducing
the effective sample size; the normal then understates variance, and the error compounds.

### 3.5 Two questions, two functions

The same distribution answers two different questions, and the answers diverge exactly where the
distribution is uncertain.

The quantile reads one point of the cumulative distribution function (CDF); expected shortfall integrates the tail.

Two identities every plannerTwo identities every planner should be able to write from memory:

```text
  cycle service level   CSL(Q) = F(Q)
  fill rate             beta(Q) = 1 - E[(Y - Q)^+] / E[Y]
```

The first reads the CDF at a point. The second divides a tail expectation by the mean. A
distribution can be accurate at the point and wrong about the expectation, and that is the usual
pattern: the normal is closest in the body and at moderate quantiles, and furthest in the
expected shortage. Section 6.6 shows the 95th percentile off by one unit while the shortage is
off by a factor of 2.2.

> **Note:** "Risk function" here means the inverse CDF that maps a service target to a stock
> level. In statistics the same phrase means the expected loss of an estimator. Ask which one a
> document means before using the number.

## 4. Definitions and vocabulary

The fourth column names the unit or coding system. A definition without one is the defect this
module exists to prevent.

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Quantile | The value below which a stated fraction lies | Reported without the fraction or tail direction | Units of the variable |
| Service quantile (risk function) | `Q = F^-1(p)`, mapping a service probability to a stock level | Confused with the statistical risk function, or with expected shortage | Units of the stocked quantity |
| Expected shortfall (ES) | `E[Y - Q \| Y > Q]`, the mean exceedance over a threshold | Confused with the quantile, or with `E[Y \| Y > Q]` | Units |
| First-order loss function | `n(Q) = E[(Y - Q)^+]`, expected units above a stock level | Called "expected backorders" without adjusting for partial fulfilment | Units |
| Mean | The probability-weighted average | Used as the only summary of a skewed distribution | Units |
| Variance | The probability-weighted average of squared deviations | Reported in the wrong units, or as a standard deviation | Units squared |
| Standard deviation | The square root of the variance | Added to a mean without checking units | Units |
| Coefficient of variation (CV) | Standard deviation divided by the mean | Compared across different time buckets | Dimensionless |
| Skewness | The third standardised moment, measuring asymmetry | Assumed zero because the mean is large | Dimensionless |
| Support | The set of values a variable can take | Ignored when choosing a family | A set of values or an interval |
| Normal distribution | Symmetric distribution on the whole real line, defined by mean and variance | Used for positive quantities and counts | `mu` in units, `sigma` in units |
| Gamma distribution | Right-skewed distribution on `[0, infinity)`, defined by shape and scale | Used for data containing exact zeros | `k` dimensionless, `theta` in units |
| Lognormal distribution | The distribution of a variable whose logarithm is normal | Used for data containing zeros, or for a sum | `mu_ln`, `sigma_ln` in log units, dimensionless |
| Poisson distribution | The distribution of a count of rare events at a constant rate | Used for counts whose variance exceeds the mean | `lambda` in events per interval |
| Poisson rate | The expected number of events per interval | Quoted per day and applied per week | Events per interval |
| Compound Poisson | A Poisson number of events, each with a random size | Called "Poisson demand" when sizes vary | `lambda` per interval; size moments in units |
| Random sum | A sum whose number of terms is itself random | Treated as a product of rate and duration | Units of the terms |
| Negative binomial | A Poisson count with a gamma-distributed rate; overdispersed | Used without checking the dispersion index | Mean in events; dispersion dimensionless |
| Overdispersion | Variance greater than the mean for count data | Ignored, so a Poisson fit understates the tail | Variance-to-mean ratio, dimensionless |
| Dispersion index | Variance divided by the mean | Reported without the time bucket | Dimensionless |
| Zero inflation | More exact zeros than a candidate count model predicts | Treated as missing data rather than a mixture | Proportion of zeros |
| Hurdle model | One process for occurrence, one for size | Confused with a zero-inflated mixture | Probability, and units |
| Method of moments (MM) | Equating sample moments to theoretical moments | Used without checking that the estimate is feasible | Parameter units follow the family |
| Maximum likelihood estimation (MLE) | Choosing parameters that maximise the probability of the sample | Used on data that violate the support | Parameter units follow the family |
| Plug-in estimator | Using a point estimate as if it were the true value | Ignoring estimation error in the service calculation | Units of the parameter |
| Central limit theorem (CLT) | Standardised sums of many independent finite-variance terms tend to normal | Invoked without checking term count or dependence | None |
| Berry-Esseen bound | An upper bound on normal approximation error, of order `1 / sqrt(n)` | Treated as an exact error for a given sample | Dimensionless |
| Edgeworth expansion | A series correction to the normal CDF in skewness and kurtosis | Used as a production method at small samples | Dimensionless |
| Continuity correction | Adjusting a continuous approximation for a discrete variable by half a unit | Applied inconsistently between CDF and quantile | Units |
| Newsvendor critical ratio | `C_u / (C_u + C_o)`, the optimal service quantile under linear costs | Set to 0.95 by habit rather than derived | Dimensionless |
| Cycle service level (CSL) | Probability that a replenishment cycle ends without a stock-out | Reported as fill rate | Proportion |
| Fill rate | Fraction of demand units served from stock | Reported as CSL | Proportion |
| Lead-time demand (LTD) | Demand accumulated over a replenishment lead time | Computed on the review period instead | Units |
| Inventory shortfall | Gap between inventory position and demand over the effective lead time | Confused with lead-time demand under order crossover | Units |
| Order crossover | Arrival of orders out of the sequence in which they were placed | Ignored when lead times are stochastic | Event ordering |
| Independence assumption | The assumption that demand and lead time are uncorrelated | Treated as a property of the data rather than a choice | None |
| Estimation window | The period over which parameters are estimated | Reported as a date range without the inclusion rule | Calendar dates |

Six issues carry most of the damage. **Mean versus variance units:** the mean is in units, the
variance in units squared, the standard deviation back in units; `variance = 26.99 units` is an
error that propagates into every interval and buffer. **Shape versus scale:** shape is
dimensionless, scale carries units; shape 4 with scale 25 units has mean 100 units, while shape 4
with rate 25 per unit has mean 0.16 units. **Poisson rate versus count:** `lambda` is events per
interval, and a daily rate applied to a weekly lead time is a factor-of-seven error. **Zero
inflation versus a hurdle:** a zero-inflated model mixes a point mass with a count and allows
zeros from both parts; a hurdle uses separate occurrence and size processes. **Quantile versus
expected shortfall:** one point against an integral; a model can pass a quantile check and fail
the shortage calculation. **Independence versus uncorrelatedness:** zero correlation is weaker
than independence, and neither equals "we did not check"; a scatter plot is a five-minute
diagnostic most planning models have never had.

## 5. The mechanics, step by step

The procedure has four phases. Each step names what you do, what you check, and what goes wrong
if you skip the check.

### Phase A — Specify the object before fitting anything

**Step 1. Name the quantity and the horizon.** Write one sentence: "Y is the demand in units for
item I at location L over one replenishment lead time." Check that it contains a unit and a
horizon, and convert every rate to that horizon before comparing anything. Failure: fitting daily
demand and applying it to a lead time, or a daily Poisson rate with a weekly lead time, which
misstates mean and variance by a factor of seven.

**Step 3. Name the support.** Classify the quantity as continuous positive, non-negative integer,
all real, or a mixture with a point mass at zero. Count the zeros, the minimum, and whether
fractional values are possible. Failure: a gamma or lognormal fit to exact zeros, or a normal fit
to a positive quantity whose mean is within two standard deviations of zero.

**Step 4. Name the mechanism.** Write down why the quantity varies: a sum of small effects, a
product of multiplicative effects, a count of events, a random number of events with random
sizes, or a positive duration with a hard floor. Check that the family's mechanism matches the
sentence. Failure: choosing a family because the software offers it.

### Phase B — Diagnose the data before estimating

**Step 5. Compute the moments with units.** Compute mean, variance, standard deviation, CV, and
skewness at the decision horizon. Mean in units, variance in units squared, CV and skewness
dimensionless. Failure: reporting a variance as a standard deviation, which makes a buffer look
four times larger than it is.

**Step 6. Test the count structure.** For count data compute the dispersion index,
`variance / mean`. Poisson requires approximately 1; a compound Poisson requires
`E[X^2] / E[X]`. Above 1 means overdispersion and rules out a simple Poisson; below 1 means
underdispersion, which is rarer and usually means constrained events or a coarse bucket.
Failure: a Poisson fit to overdispersed demand, where the mean is right and the tail far too
thin.

**Step 7. Test the zero structure.** Compare the observed zero fraction with the fraction the
candidate model predicts. For a Poisson with mean `m` the prediction is `exp(-m)`. If observed
zeros exceed it, the rate is not constant, which points to a negative binomial, a hurdle, or a
compound process with a low arrival rate. Failure: dropping excess zeros as data quality
problems, which inflates the mean of what remains and hides the occurrence process.

### Phase C — Fit with a stated method and validate in the tail

**Step 8. Choose the estimator.** Use MM when you need a closed form and the sample is large; use
MLE when the sample supports it and the support is respected. Check that the estimate lies
inside the parameter space; a gamma shape below zero, or a lognormal fit to zeros, is a
specification failure rather than a numerical one. Failure: MM on a small sample, which can
produce a shape below 1 and a density that diverges at zero, or MLE on data with zeros, which
does not run at all.

**Step 9. Estimate and record the units.** Record each parameter with its unit and the formula
used. For a gamma, record whether the second parameter is a scale or a rate. Plug the parameters
back in and recompute the mean and variance; under MM they should match the sample moments to
rounding. Failure: a parameter table a second analyst cannot reproduce.

**Step 10. Validate in the tail, not the body.** Compare fitted and empirical CDFs at the 90th,
95th, and 99th percentiles; compare expected shortfall at the planned stock level; plot the upper
decile on a probability scale. The body will agree for almost any family; the tail discriminates.
Failure: a Kolmogorov-Smirnov test dominated by the body, reported as evidence that the tail is
adequate.

### Phase D — Propagate to the decision

**Step 11. Form lead-time demand.** Use the fixed-lead-time formula if lead time is genuinely
stable, and the random-sum formula if it is not, and state which. Check the units of each term
and whether the lead-time variance term is material at this item's CVs. Failure: adding a
lead-time standard deviation to a demand standard deviation, a units error; combine variances,
then take a square root.

**Step 12. Compute the service measure you will be judged on.** Compute CSL, fill rate, and
expected units short at the candidate stock level. If the business is judged on fill rate, size
on fill rate. Check that the three move together; if a model improves CSL while worsening
expected shortage, it has changed the shape of the tail rather than the level. Failure:
optimising CSL and reporting fill rate, or the reverse.

**Step 13. Run the sensitivity that matters.** Refit with a different family, a different
estimation window, and a non-zero correlation, and report the range of required stock. If that
range is wider than the improvement claimed by a new model or system, the comparison is not
identifiable at the current data quality. Failure: a single number presented without family
sensitivity, which is usually larger than the year-on-year change being celebrated.

### 5.1 The CLT's scope, quantified

The table holds the demand process fixed — the compound Poisson process of section 6 — and varies
only the horizon. It isolates how the normal approximation improves as the number of terms
grows. Skewness falls as `1 / sqrt(L)`, the CLT's rate. The quantile error falls quickly; the
expected-shortage error falls much more slowly.

| Horizon `L` (days) | Mean (units) | `sigma` (units) | Skewness | Exact 95th | Normal 95th | Exact 99th | Normal 99th | Shortage ratio at exact 95th |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.6 | 2.00 | 1.46 | 6 | 4.89 | 8 | 6.25 | 5.86 |
| 2 | 3.2 | 2.83 | 1.03 | 9 | 7.85 | 12 | 9.78 | 3.58 |
| 6 | 9.6 | 4.90 | 0.60 | 18 | 17.66 | 23 | 21.00 | 1.87 |
| 12 | 19.2 | 6.93 | 0.42 | 31 | 30.60 | 37 | 35.32 | 1.61 |
| 30 | 48.0 | 10.95 | 0.27 | 67 | 66.02 | 76 | 73.48 | 1.40 |
| 90 | 144.0 | 18.97 | 0.15 | 176 | 175.21 | 190 | 188.14 | 1.22 |

**The quantile converges before the loss does.** At a 6-day horizon the normal 95th percentile is
within 0.34 units of the exact value, a 1.9% error; at the same horizon the expected shortage is
46% too low. The model looks calibrated where a planner checks it and is badly calibrated at the
quantity that drives fill rate.

**The 99th converges more slowly than the 95th.** At `L = 6` the normal 99th is 2 units low, an
8.7% error, and at `L = 30` it is still 2.5 units low. The further into the tail the decision
sits, the more horizon the approximation needs.

**Skewness alone does not settle it.** At `L = 90` skewness is 0.15 and the shortage ratio is
still 1.22. A nearly symmetric third moment does not guarantee a tail the normal captures.

### 5.2 Fitting: method of moments against maximum likelihood

For the gamma family the two estimators are easy to write and behave differently.

```text
  METHOD OF MOMENTS                       MAXIMUM LIKELIHOOD
  k_hat = x_bar^2 / s^2                  solve  ln(k) - psi(k) = ln(x_bar) - mean(ln x)
  theta_hat = s^2 / x_bar                then   theta_hat = x_bar / k_hat

  closed form                            iterative; needs strictly positive data
  can return k < 1 or k = infinity       returns a finite estimate if the data allow
  sensitive to outliers in s^2           sensitive to the smallest observations
```

**Match the estimator to the sample size.** With fewer than about 30 observations the sample
standard deviation has a relative standard error of roughly `1 / sqrt(2n)`, which is 13% at
`n = 30`. A shape computed from a noisy variance is noisy in a way the point estimate does not
show. Report the standard error, or a range.

**Never fit a continuous family through a zero.** The log-likelihood is undefined, and software
that returns a number has silently dropped the zeros or added a constant. Both change the object
being modelled. Use a hurdle or compound specification instead.

Check the parameterisation in the software: a gamma with mean 100 units is shape 4, scale 25
units, or shape 4, rate 0.04 per unit. If it reports a "beta" of 25, check whether the mean is
100 or 0.16.

### 5.3 Parameter units when the bucket changes

| Family | Parameter | Symbol | Unit | Effect of changing the bucket from day to week |
|---|---|---|---|---|
| Normal | Mean | `mu` | Units | `mu_week = 7 mu_day` |
| Normal | Standard deviation | `sigma` | Units | `sigma_week = sqrt(7) sigma_day` |
| Poisson | Rate | `lambda` | Events per day | `lambda_week = 7 lambda_day`; variance also multiplies by 7 |
| Gamma (duration) | Scale | `theta` | Days | `theta_week = theta_day / 7` if the time unit is the week |
| Gamma (quantity) | Scale | `theta` | Units | Unchanged by the time bucket; changes if the quantity unit changes |
| Lognormal | Log-mean, log-SD | `mu_ln`, `sigma_ln` | Log units, dimensionless | Change if the variable is rescaled; adding a constant to `ln y` multiplies `y` |
| Compound Poisson | Arrival rate | `lambda` | Events per day | `lambda_week = 7 lambda_day` |
| Compound Poisson | Size moments | `E[X]`, `E[X^2]` | Units, units squared | Unchanged by the time bucket |


## 6. Worked example

Ravensworth Industrial Supply is a fictional regional distributor of maintenance parts. It stocks
the RK-14 seal kit at one warehouse. All figures are constructed to be internally consistent and
checkable by hand.

### 6.1 The two sources of randomness

Orders arrive at a Poisson rate of `lambda = 0.8` per day. Each order is for 1 to 4 kits with the
probabilities below. The supplier's lead time has taken five values with the frequencies below.

**Table 6.1 — Order size distribution and its moments**

| `x` (units) | `P(X = x)` | `x P` | `x^2 P` | `x^3 P` |
|---:|---:|---:|---:|---:|
| 1 | 0.40 | 0.40 | 0.40 | 0.40 |
| 2 | 0.30 | 0.60 | 1.20 | 2.40 |
| 3 | 0.20 | 0.60 | 1.80 | 5.40 |
| 4 | 0.10 | 0.40 | 1.60 | 6.40 |
| **Total** | **1.00** | **2.00** | **5.00** | **14.60** |

`E[X] = 2.00` units, `E[X^2] = 5.00` units squared, `Var(X) = 5.00 - 2.00^2 = 1.00` units squared,
`E[X^3] = 14.60` units cubed.

**Table 6.2 — Lead-time distribution and its moments**

| `l` (days) | `P(L = l)` | `l P` | `l^2 P` |
|---:|---:|---:|---:|
| 4 | 0.10 | 0.40 | 1.60 |
| 5 | 0.25 | 1.25 | 6.25 |
| 6 | 0.35 | 2.10 | 12.60 |
| 7 | 0.20 | 1.40 | 9.80 |
| 8 | 0.10 | 0.80 | 6.40 |
| **Total** | **1.00** | **5.95** | **36.65** |

`E[L] = 5.95` days, `E[L^2] = 36.65` days squared, `Var(L) = 36.65 - 5.95^2 = 1.2475` days
squared, `sigma_L = sqrt(1.2475) = 1.117` days.

### 6.2 Daily demand moments

Daily demand is compound Poisson: a Poisson number of orders, each of random size.

```text
  mu_D    = lambda x E[X]      = 0.8 x 2.00          = 1.60 units per day
  Var(D)  = lambda x E[X^2]    = 0.8 x 5.00          = 4.00 units squared per day
  sigma_D = sqrt(4.00)                              = 2.00 units per day
  CV_D    = 2.00 / 1.60                             = 1.25
```

The variance is `lambda E[X^2]`, not `lambda Var(X)`; the difference is the size's second moment,
which carries the lumpiness. Using `lambda Var(X) = 0.8` understates daily variance fivefold.

### 6.3 Lead-time demand moments with both quantities random

```text
  E[Y]    = E[L] x lambda x E[X]
          = 5.95 x 0.8 x 2.00
          = 5.95 x 1.60
          = 9.52 units

  Var(Y)  = E[L] x lambda x E[X^2]  +  Var(L) x (lambda x E[X])^2
          = 5.95 x 4.00             +  1.2475 x 1.60^2
          = 23.80                   +  1.2475 x 2.56
          = 23.80                   +  3.1936
          = 26.9936 units squared

  sigma_Y = sqrt(26.9936) = 5.1955 units
```

**Table 6.3 — Variance decomposition**

| Source | Formula | Substitution | Value (units squared) | Share of variance |
|---|---|---|---:|---:|
| Demand variability | `E[L] lambda E[X^2]` | `5.95 x 4.00` | 23.80 | 88.2% |
| Lead-time variability | `Var(L) (lambda E[X])^2` | `1.2475 x 2.56` | 3.19 | 11.8% |
| **Total** | | | **26.99** | **100%** |

The demand term dominates at 88%, the opposite of the imported item in Topic 05 where the
lead-time term carried 98%. The difference is the relative CVs: `CV_D = 1.25` and
`CV_L = 1.117 / 5.95 = 0.188`.

The crossover formula from Topic 05 makes this explicit. The lead-time term equals the demand
term when

```text
  sigma_L* = CV_D x sqrt(E[L])
           = 1.25 x sqrt(5.95)
           = 1.25 x 2.439
           = 3.05 days
```

The observed lead-time standard deviation is 1.12 days, well below the 3.05-day crossover. The
lead-time variance is real but secondary. A planner who applied the Topic 05 lesson mechanically
— "lead-time variance always dominates" — would over-invest in this supplier and under-invest in
the demand process.

### 6.4 The exact distribution: mixing over lead time

For each possible lead time, demand is compound Poisson with rate `0.8 l`. The distribution of
`Y` is the weighted mixture of those five distributions:

```text
  P(Y = y) = 0.10 P(Y = y | L = 4) + 0.25 P(Y = y | L = 5)
           + 0.35 P(Y = y | L = 6) + 0.20 P(Y = y | L = 7)
           + 0.10 P(Y = y | L = 8)
```

**Table 6.4 — Components of the mixture and their contributions at Q = 19**

| `L` (days) | Weight `P(L = l)` | `lambda l` | Mean (units) | `sigma` (units) | 95th (units) | `F_l(19)` | `E[(Y - 19)^+ \| L = l]` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.10 | 3.2 | 6.4 | 4.000 | 14 | 0.9957 | 0.0110 |
| 5 | 0.25 | 4.0 | 8.0 | 4.472 | 16 | 0.9859 | 0.0408 |
| 6 | 0.35 | 4.8 | 9.6 | 4.899 | 18 | 0.9651 | 0.1129 |
| 7 | 0.20 | 5.6 | 11.2 | 5.292 | 21 | 0.9294 | 0.2551 |
| 8 | 0.10 | 6.4 | 12.8 | 5.657 | 23 | 0.8769 | 0.4969 |
| **Mixture** | **1.00** | | **9.52** | **5.196** | **19** | **0.9574** | **0.1515** |

The weighted sums reproduce `F(19) = 0.9574` and `E[(Y - 19)^+] = 0.1515` units per cycle.

The mixture mean recomputes from the component means as
`0.10(6.4) + 0.25(8.0) + 0.35(9.6) + 0.20(11.2) + 0.10(12.8) = 9.52` units, and the mixture
variance from the component second moments as `117.624 - 9.52^2 = 26.9936` units squared.

The exact CDF gives `F(17) = 0.9239`, `F(18) = 0.9427`, `F(19) = 0.9574`, `F(20) = 0.9686`.
Since `F(18) < 0.95 <= F(19)`, the 95th percentile is 19 units. The 99th is 24, the 90th is 16,
and the median is 9.

### 6.5 Four fits at the same moments

Fit four families to `mu = 9.52` units and `sigma = 5.1955` units. The conversions are shown
because they are where most errors enter.

```text
  Normal:      mu = 9.52 units, sigma = 5.1955 units

  Gamma:       k = mu^2 / sigma^2 = 90.6304 / 26.9936 = 3.3575
               theta = sigma^2 / mu = 26.9936 / 9.52 = 2.8355 units

  Lognormal:   sigma_ln^2 = ln(1 + sigma^2 / mu^2)
                          = ln(1 + 26.9936 / 90.6304) = ln(1.2979) = 0.2607
               sigma_ln  = 0.5106
               mu_ln     = ln(9.52) - 0.2607 / 2 = 2.2533 - 0.1304 = 2.1229
```

**Table 6.5 — Quantiles from the exact distribution and three fits**

| Model | Parameters with units | Median | 90th | 95th | 99th | Fitted skewness |
|---|---|---:|---:|---:|---:|---:|
| Exact mixture | None | 9 | 16 | 19 | 24 | 0.670 |
| Normal | `mu = 9.52` units, `sigma = 5.1955` units | 9.52 | 16.18 | 18.07 | 21.61 | 0.000 |
| Gamma | `k = 3.3575` dimensionless, `theta = 2.8355` units | 8.59 | 16.49 | 19.35 | 25.53 | 1.091 |
| Lognormal | `mu_ln = 2.1229`, `sigma_ln = 0.5106` log units | 8.36 | 16.08 | 19.35 | 27.41 | 1.800 |

**The normal is low at both upper quantiles.** It is 0.93 units low at the 95th (4.9%) and 2.39
low at the 99th (10.0%). It is also the only fit that puts probability below zero:
`Phi((0 - 9.52) / 5.1955) = Phi(-1.832) = 3.3%` of its mass is assigned to negative demand.

**Gamma and lognormal agree at the 95th and diverge at the 99th.** Both give 19.35, which rounds
to 19 and matches the exact answer; at the 99th they give 25.53 and 27.41 against 24. A
heavier-tailed fit is not automatically safer: it is conservative at one service level and
wasteful at another. **Fitted skewness is a warning, not a validation.** The exact skewness is
0.670; the gamma, constrained by two moments, implies `2 / sqrt(k) = 1.091`, and the lognormal
1.800. Matching two moments does not determine the tail.

> **Note:** The exact 99th percentile rests on the upper tail of a five-point lead-time
> distribution and a four-point size distribution, each estimated from a few years of data. A fit
> that agrees with it to one unit is not evidence that the fit is right; both may be wrong in the
> same direction.

### 6.6 Quantile error against expected-shortage error

The 95th percentile is off by less than one unit; the expected shortage at the same point is off
by more than a factor of two. This is the section that changes how you validate a distribution.

**Table 6.6 — Service measures at four stock levels**

| `Q` (units) | CSL exact | CSL normal | Shortage exact (units/cycle) | Shortage normal (units/cycle) | Ratio | Fill rate exact | Fill rate normal |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 0.9239 | 0.9250 | 0.2848 | 0.1745 | 1.63 | 0.9701 | 0.9817 |
| 18 | 0.9427 | 0.9487 | 0.2088 | 0.1119 | 1.87 | 0.9781 | 0.9882 |
| 19 | 0.9574 | 0.9660 | 0.1515 | 0.0697 | 2.17 | 0.9841 | 0.9927 |
| 20 | 0.9686 | 0.9782 | 0.1089 | 0.0421 | 2.59 | 0.9886 | 0.9956 |

At `Q = 19` the normal says the cycle service level is 96.6% when it is 95.7%, a difference of 0.9
percentage points. The same approximation says expected shortage is 0.070 units per cycle when
it is 0.152, a difference of 54%, and reports fill rate 99.27% when it is 98.41%.

The mechanism is visible in the loss function. For a normal distribution,

```text
  E[(Y - Q)^+] = sigma x phi(z) - (Q - mu) x (1 - Phi(z)),   z = (Q - mu) / sigma
```

The first term is tail mass times average distance into the tail; the second subtracts the part
of the threshold below the mean. Both are large and nearly cancel, so the difference is sensitive
to tail shape even when `z` is accurate to two decimals.

**Table 6.7 — The normal loss function, term by term, at `mu = 9.52`, `sigma = 5.1955`**

| `Q` | `z` | `1 - Phi(z)` | `phi(z)` | `sigma phi(z)` | `(Q - mu)(1 - Phi)` | Loss (units) |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 1.4397 | 0.0750 | 0.1415 | 0.7352 | 0.5610 | 0.1745 |
| 18 | 1.6322 | 0.0513 | 0.1053 | 0.5471 | 0.4350 | 0.1119 |
| 19 | 1.8246 | 0.0340 | 0.0755 | 0.3923 | 0.3223 | 0.0697 |
| 20 | 2.0171 | 0.0218 | 0.0522 | 0.2712 | 0.2285 | 0.0421 |

At `Q = 19`: `z = 9.48 / 5.1955 = 1.8246`, `1 - Phi(z) = 0.0340`, `phi(z) = 0.0755`, so
`sigma phi(z) = 5.1955 x 0.0755 = 0.3923` and `(Q - mu)(1 - Phi) = 9.48 x 0.0340 = 0.3223`,
giving `loss = 0.3923 - 0.3223 = 0.0700` units per cycle. Against the exact 0.1515 the normal
understates shortage by 0.0815 units per cycle, or 54%. The conditional expected shortfall tells
the same story: exact `0.1515 / (1 - 0.9574) = 0.1515 / 0.0426 = 3.56` units, normal
`0.0700 / (1 - 0.9660) = 0.0700 / 0.0340 = 2.06` units. Conditioned on a stock-out, the exact
distribution is short by 3.56 units on average; the normal says 2.06. The normal is wrong about
the severity of the event even though it is approximately right about its frequency.

### 6.7 Why the skew produces exactly this error

A first-order Edgeworth correction gives the direction: for standardised `x`, the CDF is
`F(x) = Phi(x) + (gamma / 6)(1 - x^2) phi(x)`, so the quantile correction is
`delta_z = (gamma / 6)(z^2 - 1)`. With `gamma = 0.670`, this predicts 19.06 at the 95th and
24.17 at the 99th, against exact values of 19 and 24. It is a sanity check, not a production
method: it needs the true skewness, which is itself estimated with error, and it can overshoot
for heavy tails. Its use is diagnostic. If a fitted family's implied skewness differs from the
sample, you can predict the direction of its quantile error before computing it.

### 6.8 Correlation between demand and lead time

The random-sum formula assumes independence. Relax it in the simplest model where the assumption
can be seen: a demand rate `D` applying over a lead time `L`, both random and correlated.

```text
  Var(D L) = mu_D^2 sigma_L^2 + mu_L^2 sigma_D^2
           + 2 mu_D mu_L Cov(D, L)
           + sigma_D^2 sigma_L^2 + Cov(D, L)^2
```

Take `mu_D = 2.0` units per day, `sigma_D = 1.0`, `mu_L = 6.0` days, `sigma_L = 1.5` days. The
first two terms are `4.0 x 2.25 = 9.00` and `36.0 x 1.00 = 36.00`. The covariance is
`rho x 1.0 x 1.5 = 1.5 rho`.

**Table 6.9 — Variance and safety stock under four correlations**

| `rho` | `mu_D^2 sigma_L^2` | `mu_L^2 sigma_D^2` | `2 mu_D mu_L Cov` | `sigma_D^2 sigma_L^2` | `Cov^2` | Variance | `sigma` | Buffer 95% | Buffer 99% |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 9.00 | 36.00 | 0.00 | 2.25 | 0.00 | 47.25 | 6.87 | 11.31 | 15.99 |
| 0.5 | 9.00 | 36.00 | 18.00 | 2.25 | 0.56 | 65.81 | 8.11 | 13.34 | 18.87 |
| 0.8 | 9.00 | 36.00 | 28.80 | 2.25 | 1.44 | 77.49 | 8.80 | 14.48 | 20.48 |
| -0.5 | 9.00 | 36.00 | -18.00 | 2.25 | 0.56 | 29.81 | 5.46 | 8.98 | 12.70 |

At `rho = 0.5` variance rises from 47.25 to 65.81, 39%, and the 95% safety stock from 11.31 to
13.34 units, 18%. At `rho = 0.8` the buffer rises 28%; at `rho = -0.5` it falls 21%.

> **Note:** This is a rate-times-duration model, not the day-by-day random sum of section 6.1.
> The two have different variances even at `rho = 0`, because the rate model lets a high demand
> day persist across the whole lead time while the random-sum model draws each day independently.
> For the general correlated random-sum result, see Wang, Zinn and Croxton (2009). The qualitative
> direction — positive correlation widens the buffer — is the same in both.

Positive correlation means the shortfall arrives when demand is already high, the worst possible
time: the independent model treats the two bad events as separate, the correlated model as one.

### 6.9 What the analyst should recommend

1. **Do not use the normal for this item.** The 95th percentile is close enough for a rough cut,
   but expected shortage is 2.2 times the normal's and the business is judged on fill rate.
2. **Fit the compound process directly.** Both input distributions are small enough to estimate
   and state, so the mixture is exact up to their sampling error.
3. **Report the window and test the correlation.** The 8-day lead time is one observation in ten;
   plot lead time against demand rate and add the covariance term if the sign is positive.

## 7. Code, query, or calculation

One code block plus its output. It computes the exact mixture distribution of section 6.4 and the expected shortage
at two stock levels, so the tables can be checked rather than trusted. It uses only the standard
library.

```python
import math
lam, sizes = 0.8, [0.4, 0.3, 0.2, 0.1]
Ls, ps = [4, 5, 6, 7, 8], [0.10, 0.25, 0.35, 0.20, 0.10]

def poisson(m, k):
    return math.exp(-m) * m**k / math.factorial(k)

def compound(m, ymax=80):
    d, conv = [0.0]*(ymax+1), [0.0]*(ymax+1)
    conv[0] = 1.0
    for n in range(50):
        w = poisson(m, n)
        for y in range(ymax+1):
            d[y] += w * conv[y]
        conv = [sum(conv[y-s] * p for s, p in enumerate(sizes, 1) if y >= s)
                for y in range(ymax+1)]
    return d

parts = [compound(lam * l) for l in Ls]
mix = [sum(pl * d[y] for pl, d in zip(ps, parts)) for y in range(81)]
mu = sum(y * p for y, p in enumerate(mix))
var = sum((y - mu)**2 * p for y, p in enumerate(mix))
short = lambda q: sum((y - q) * p for y, p in enumerate(mix) if y > q)
print(round(mu, 3), round(var, 3), [round(short(q), 4) for q in (18, 19)])
```

Expected output:

```text
9.52 26.994 [0.2088, 0.1515]
```

Three checks before trusting any such block:

- The mean and variance must match the closed-form moments of section 6.3. If not, the
  convolution is truncated or the Poisson probabilities are not normalised.
- The probabilities must sum to 1.000 within rounding. A sum below 1 means `ymax` is too small
  for the tail, which biases every quantile downward.
- The shortage at a very high `Q` must approach zero, and the shortage at `Q = 0` must equal the
  mean. Both are identities, and both catch an off-by-one in the tail sum.

## 8. Data, standards, and regulatory context

### 8.1 What the data must contain

A distribution fit is only as good as the definition of the observation. Before estimation, the
dataset must answer five questions.

| Question | Why it matters | What to record |
|---|---|---|
| Is the observation demand, shipment, or order? | Shipments are censored by stock-outs; orders contain batching | Source system and field name |
| What is the time bucket? | Every rate and variance scales with it | Bucket length and boundary convention |
| What is the unit? | Cases, eaches, and kilograms cannot be pooled | Unit of measure and conversion factor |
| Is lead time measured from order or confirmation? | The two differ by approval and transmission time | Start and end timestamps |
| Are zero-demand periods recorded? | Missing zeros inflate the mean and hide the occurrence process | Row count per period and the zero rule |

The first is the most consequential. A stock-out truncates sales at available inventory, so the
fit describes *fulfilled* demand, not demand, and cuts the tail exactly where the buffer decision
is made. If service is low, the fit is self-confirming and wrong.

### 8.2 Standards that define the vocabulary and the time axis

No standard mandates a family. The standards that matter fix the words and the clock, because a
distribution is defined relative to both.

| Standard | Governs | What it constrains | Grade |
|---|---|---|---|
| ISO 3534-1:2016 | Statistics vocabulary and symbols | Meanings of distribution function, quantile, expected value, variance | B |
| ISO 8601-1:2019 and ISO 8601-2:2019 | Date and time representation | The definition of an interval and a duration, hence what a lead time is measured on | B |
| ISO 5725 series | Accuracy, trueness, and precision of measurement | Separates measurement error from process variation | B |
| JCGM 100:2008 (GUM) | Expression of uncertainty in measurement | Defines the coverage factor and one-sided interval, the same object as a service factor | B |
| GS1 EPC Information Services (EPCIS) 2.0 | Event data for objects in a supply chain | Occurrence timestamps from which demand and lead time are reconstructed | B |

**ISO 3534-1** settles vocabulary collisions: when a document says "confidence interval" but
means "prediction interval", the standard supplies the definition. It does not supply a
distribution choice, and no standard does. **JCGM 100:2008** gives a coverage factor `k` for a
one-sided 95% interval of 1.645 under a normal model, numerically identical to the inventory
service factor `z`; the assumptions that make each valid, and the ways they fail, are the same.

### 8.3 What regulation does not supply

No regulation requires a gamma rather than a lognormal, and none specifies a service factor.
Financial reporting standards constrain the valuation of inventory, not the stochastic model
behind its replenishment. Traceability regulations constrain the existence and retention of event
records, not the distribution fitted to them.

Distribution choice is therefore a model-risk decision. The documentation that survives an audit
answers four questions: what family, fitted to what window, by what method, validated where in
the tail. A model that cannot answer them is not reproducible or defensible.

### 8.4 Convention rather than requirement

Several things that feel like rules are habits. The 95% target is a quality-control convention,
not an optimum; the optimum is the newsvendor critical ratio, set by stock-out cost relative to
holding. The `1.645` factor is the normal 95th percentile, valid only under normality and a
one-sided target. "Two weeks of cover" and a twelve-month window are conventions, not rules.

## 9. Industry practice and case evidence

### 9.1 The normal-approximation debate, in two results

**Tyworth and O'Neill (1997)** compared optimal order quantities and reorder points from the
normal approximation with a theoretically correct procedure for fast-moving finished goods across
seven industry groups, measuring safety stock, logistics cost, total cost, and fill rate; they
found the normal robust in both cost and service. Grade A.

**Lau and Lau (2003)** presented evidence that the system-cost penalty from the normal can be
serious even when the lead-time demand coefficient of variation is low, contrary to the
prevailing view. With modern computing, they argued, the shape should be estimated rather than
assumed. Grade A.

Both can be true. "Robust" is a property of the family combined with a cost structure, a service
measure, and an item class, not of the family in the abstract. Tyworth and O'Neill studied
fast-moving finished goods; Lau and Lau studied a `(Q, R)` system on system cost. Test robustness
in your own cost structure and at your own service measure.

**Tyworth (2025)** revisited the gamma approximation with multimodal maritime lead times, reporting
annual total cost errors of a few percent or less for fast-moving goods, with accuracy improving
as demand variability rises. Grade A, open access.

### 9.2 What the simulation studies find

| Study | Design | Reported finding | Grade |
|---|---|---|---|
| Tadikamalla (1984) | Normal, logistic, lognormal, gamma, Weibull fits to lead-time demand | Normal and logistic inadequate at large CV; lognormal, gamma, Weibull more versatile | A |
| Bagchi (1987) | Lumpy demand as compound Poisson, variable lead time | Derives geometric Poisson normal and gamma; ignoring lead-time variability causes material error | A |
| Rossetti and Ünlü (2011) | Normal, lognormal, gamma, Poisson, negative binomial under lumpy demand | Normal worst; gamma and negative binomial better under high variability; Poisson only near variance-to-mean 1 | A |
| Tyworth (2025) | Gamma fit to multimodal maritime lead times | Annual cost errors of a few percent or less; higher moments unreliable | A |

The pattern is consistent even where conclusions differ: the normal is acceptable in the body and
at moderate service for fast-moving items, and degrades as the item becomes lumpier, the target
rises, or the measure shifts from cycle service level to expected backorders.

### 9.3 The practitioner default

Planning modules in several major enterprise resource planning (ERP) and advanced planning suites
expose a single service factor per item and apply a normal quantile to a mean and standard
deviation. Documentation reviewed for this course in 2026 does not expose an item-level
distribution choice in the standard configuration. Grade D, dated 2026: product documentation,
not evidence of accuracy.

The spreadsheet pattern is equally common: `safety stock = z x CV x mean demand x sqrt(L)`. The
formula is dimensionally consistent because the CV is dimensionless, and distributionally valid
only under the normal. Substituting the CV of *forecast error* for the CV of *demand* changes the
meaning without changing the appearance; Topic 09 covers the distinction.

"Add 10% for uncertainty" is a convention with no distributional content and no stated target.
Grade E as evidence.

### 9.4 What is missing from the public record

Company-level evidence that isolates distribution choice is thin: disclosures rarely state the
family, window, or service measure, and vendor cases lack a baseline distribution. The strongest
evidence remains the analytical and simulation literature, Grade A but setting-specific.

The direction is clear (skewed demand needs a skewed family), the magnitude is setting-dependent,
and the decision should be validated on your own tail, not settled by a citation.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Normal fitted to lumpy demand | "The mean is large, so the normal is fine" | Symmetric support assigns mass below zero and thins the upper tail | Service below target; fill rate worse than CSL suggests | Check zero fraction and skewness first |
| Gamma or lognormal fitted through zeros | Fitting code drops zeros or adds a constant | Continuous support has no point mass at zero | Mean and variance biased; occurrence process lost | Use a hurdle or compound specification |
| Poisson assumed for all counts | Variance-to-mean ratio never computed | Real counts are overdispersed | Tail too thin; frequent stock-outs | Compute the dispersion index at the decision horizon |
| Variance reported in units | Table says "variance = 27 units" | Units squared confused with units | Intervals and buffers wrong by a square root | Write the exponent beside every variance |
| Lognormal parameters read as units | `mu_ln = 3.1 units` | Log-scale parameters are dimensionless | Mean and quantiles wrong by an exponent | Record the parameterisation and log base |
| Poisson rate used with the wrong bucket | Daily `lambda` with a weekly lead time | Rate scales linearly with the interval | Mean off by a factor of seven | Convert every rate to the decision horizon |
| Quantile validated, shortage ignored | Model looks calibrated at the 95th percentile | The loss function integrates the tail | Fill rate materially worse than predicted | Validate on expected shortage as well |
| 99th percentile from a handful of points | Tail cell has three observations | Extreme quantiles are sample-starved | Buffer is an artefact of two or three events | Report effective tail count; cap the target |
| CSL and fill rate conflated | Both reported as "service level" | Different functionals of the same distribution | Wrong stocking target | State the measure and compute all three |
| Independence assumed without a test | No scatter plot of lead time against demand rate | Positive correlation widens the true distribution | Buffer understated in peak periods | Estimate correlation; add the covariance term |
| Order crossover ignored | Receipts arrive out of sequence under stress | Lead-time demand is not the right object | Reorder point wrong even with a correct family | Model the inventory shortfall |
| Window spans a regime change | Parameters unstable across subperiods | Mixture of two processes | Fitted distribution describes neither | Test stability; shorten the window or model the break |

**The 99th percentile is the least supported number in most inventory models.** In the worked
example it depends on the 8-day lead time, probability 0.10, and the 4-unit order, probability
0.10. With 40 observations each cell holds about four. The number is computable and not reliable.
A target above the level the data support should be met with a policy buffer whose size is stated
as a policy, not dressed as an estimated quantile.

**The quantile-versus-shortage gap is the most consequential validation error.** A model can be
within one unit at the 95th percentile and wrong by a factor of two in expected units short. The
quantile reads one point of the CDF; the shortage integrates the whole tail. If the business
measures fill rate, a quantile check does not validate the model that sets the fill rate.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Demand events are independent | Variance of a sum is the sum of variances | Promotions, paydays, weather, campaigns | Lag-1 autocorrelation at the decision bucket |
| Event sizes are iid | Compound moments use `E[X]` and `E[X^2]` | A few customers dominate; sizes trend | Plot sizes; check trend or mixture |
| Lead time is independent of demand | The variance formula has no covariance term | Capacity congestion, peak-season carriers | Scatter plot and correlation of lead time against demand rate |
| Lead time is stationary in the window | One distribution describes the whole window | Route changes, supplier changes, port disruption | Split the window and compare quantiles |
| Orders do not cross | Lead-time demand is the right object | Stochastic lead times with independent orders | Check whether receipts follow dispatch order |
| Demand is uncensored | The fitted tail is the demand tail | Stock-outs truncate sales at available stock | Compare sales with unconstrained orders or lost-sales records |
| The bucket is fixed and parameters scale | Rates and variances are bucket-dependent | Mixed daily and weekly data | Recompute at one bucket and compare |
| Parameters are known, not estimated | The service calculation treats estimates as truth | Small samples, short windows | Bootstrap parameters and service level |
| The item is homogeneous | One unit is exchangeable with another | Substitution, grade mix, kit configuration | Check the hierarchy in Topic 01 |
| Costs are linear for the critical ratio | The critical ratio is the optimal quantile | Fixed order costs, capacity, penalties | Compare the cost function with the decision rule |

**Order crossover invalidates the object, not just the parameters.** When lead times are
stochastic and orders can arrive out of sequence, demand over a lead time is no longer the
quantity that sets the reorder point; the relevant object is the inventory shortfall
distribution. Bradley and Robinson (2005) show a negative-binomial shortfall approximation
substantially outperforms the usual normal lead-time demand approximation under crossover, and
Disney, Maltz, Wang and Warburton (2016) document the violation in global logistics data and
derive exact safety stocks for a policy that accounts for it. If receipts routinely arrive out of
order, fix the object before refining the distribution.

**Parameter uncertainty is not process uncertainty.** Every formula here treats estimates as
true. In a 40-observation sample the standard error of the mean is `s / sqrt(40)`, 16% of the
standard deviation, and the standard error of skewness is larger. Eppen and Martin (1988) show
the textbook procedure can miss the target materially when demand and lead time are both random
and parameters are estimated. The buffer that ignores estimation error is optimistically sized,
and the optimism grows as the sample shrinks.

**Non-stationarity is the limit no family fixes.** A gamma fitted across a supplier change, a port
closure, or a demand regime shift describes a mixture of two processes and neither of them. Its
tail can be heavier than either component, and its parameters have no operational meaning. Test
with a split-window comparison of the 90th and 95th percentiles; if they differ by more than
sampling error, the window is too long or the process needs a structural model.

## 12. Exercises

### Exercise 1 — Compound moments by hand (calculation, 20 minutes)

Daily arrivals are Poisson with `lambda = 0.6`; sizes are 1, 2, or 3 units with probabilities 0.5,
0.3, and 0.2; lead time is 10 days.
**Deliverable:** `E[X]`, `E[X^2]`, `Var(X)`, daily mean and variance, and lead-time demand mean and
variance, each with units. **Acceptance:** every moment carries a unit; variance uses
`lambda E[X^2]`; repeated for `Var(L) = 4` days squared. **Time:** 20 minutes.

### Exercise 2 — Four families at matched moments (calculation, 45 minutes)

A fitted distribution has mean 64 units and standard deviation 32 units. Compute parameters and
the 95th and 99th percentiles for the normal, gamma, and lognormal families.
**Deliverable:** a table of parameters with units and four quantiles. **Acceptance:** gamma scale
in units; lognormal parameters labelled log-scale and dimensionless; values computed, not quoted.
**Time:** 45 minutes.

### Exercise 3 — The horizon effect (calculation, 40 minutes)

Using the process in section 5.1, compute skewness and normal 95th-percentile error at horizons of
1, 6, 30, and 90 days. State where the quantile error falls below 2% and where the expected-shortage
error falls below 20%.
**Deliverable:** a table of skewness, exact and normal quantiles, percentage error, and shortage
ratio. **Acceptance:** the `1 / sqrt(L)` scaling is shown; the two convergence horizons are
distinguished; the conclusion is one sentence. **Time:** 40 minutes.

### Exercise 4 — Expected shortfall by hand (calculation, 30 minutes)

For `mu = 9.52`, `sigma = 5.1955`, and `Q = 19`, compute the normal expected shortage using
`sigma phi(z) - (Q - mu)(1 - Phi(z))`, compare it with 0.1515, and compute conditional expected
shortfall and fill rate for both.
**Deliverable:** each substitution, the shortage values, the ratio, and the two fill rates.
**Acceptance:** `z`, `phi(z)`, and `1 - Phi(z)` to four decimals; conditional shortfall divides by
the correct tail probability; fill rate uses the correct denominator. **Time:** 30 minutes.

### Exercise 5 — Diagnose before fitting (analysis, 35 minutes)

A 60-day series has 34 zero days, mean 1.3 units, variance 6.4 units squared. The 26 demand days
total 78 units, with counts 2, 6, 10, 6, and 2 for sizes 1 through 5.
**Deliverable:** observed zero fraction, Poisson-predicted zero fraction, dispersion index,
compound variance from the size distribution, and a recommendation. **Acceptance:** Poisson
rejected on two separate grounds; residual overdispersion quantified; recommendation names a
family. **Time:** 35 minutes.

### Exercise 6 — Unit audit (analysis, 30 minutes)

Find and correct the unit error. (a) A gamma with shape 3.2 and scale 0.8 days is fitted to demand
in units. (b) A lognormal has `mu = 3.1 units` and `sigma = 0.4 units`. (c) A Poisson rate of 4.2
per week is applied to a 6-day lead time. (d) A normal has mean 120 cases and standard deviation
12 eaches, with 24 eaches per case. (e) A compound Poisson with `lambda = 0.5` per day is applied
to 3 weeks using `lambda L = 1.5`.
**Deliverable:** for each, the error, the correction, and the numerical consequence.
**Acceptance:** each correction carries a unit; consequence quantified where the data allow.
**Time:** 30 minutes.

### Exercise 7 — Correlation sensitivity (calculation, 40 minutes)

With `mu_D = 2.0`, `sigma_D = 1.0`, `mu_L = 6.0`, `sigma_L = 1.5`, compute variance and 95%
safety stock at `rho = 0`, `0.5`, `0.8`, and `-0.5`.
**Deliverable:** variance decomposition table and safety stock at each correlation.
**Acceptance:** covariance term shown explicitly; percentage change computed against `rho = 0`;
direction stated in words. **Time:** 40 minutes.

### Exercise 8 — Tail audit (design, 45 minutes)

For the worked example, list every parameter and tail cell the 99th percentile depends on, with
estimate, supporting observation count, and approximate standard error.
**Deliverable:** a table of parameters, estimates, counts, standard errors, and a judgement on
whether the 99th percentile is supported. **Acceptance:** at least six parameters audited; counts
are relevant observations, not calendar days; judgement explicit. **Time:** 45 minutes.

### Exercise 9 — Write the specification (design, 50 minutes)

Write a one-page distribution specification for an item you know, or for the RK-14 kit.
**Deliverable:** quantity, unit, horizon, family, parameters with units, fitting method, window,
tail validation, service measure, and sensitivity plan. **Acceptance:** a second analyst could
reproduce the fit from the document; every parameter has a unit; validation is in the tail; the
service measure is named. **Time:** 50 minutes.

### Exercise 10 — Critique a vendor claim (critique, 30 minutes)

A vendor states: "Our AI selects the optimal demand distribution automatically and improves
service by 30%." Write a one-page critique.
**Deliverable:** the claim decomposed into testable parts, the missing baseline, the data
required, and three questions for the vendor. **Acceptance:** at least one part identified as
unfalsifiable as stated; baseline distribution and service measure named; questions specific.
**Time:** 30 minutes.

## 13. Self-check answers

### Exercise 1 arithmetic

```text
E[X]    = 0.5(1) + 0.3(2) + 0.2(3) = 0.5 + 0.6 + 0.6 = 1.70 units
E[X^2]  = 0.5(1) + 0.3(4) + 0.2(9) = 0.5 + 1.2 + 1.8 = 3.50 units squared
Var(X)  = 3.50 - 1.70^2 = 3.50 - 2.89 = 0.61 units squared
mu_D    = 0.6 x 1.70 = 1.02 units per day
Var(D)  = 0.6 x 3.50 = 2.10 units squared per day
sigma_D = sqrt(2.10) = 1.449 units per day
E[Y]    = 10 x 1.02 = 10.20 units
Var(Y)  = 10 x 2.10 = 21.00 units squared
sigma_Y = sqrt(21.00) = 4.583 units
with Var(L) = 4 days squared:
Var(Y)  = 21.00 + 4 x 1.02^2 = 21.00 + 4.1616 = 25.16 units squared
sigma_Y = sqrt(25.16) = 5.016 units
```

Using `lambda Var(X) = 0.366` for the daily variance understates it by a factor of 5.7.
### Exercise 2 arithmetic

```text
Normal:    mu = 64 units, sigma = 32 units
           q95 = 64 + 1.6449 x 32 = 64 + 52.64 = 116.64 units
           q99 = 64 + 2.3263 x 32 = 64 + 74.44 = 138.44 units
Gamma:     k = 64^2 / 32^2 = 4096 / 1024 = 4.000 (dimensionless)
           theta = 32^2 / 64 = 1024 / 64 = 16.00 units
           q95 = 16 x 7.7536 = 124.06 units
           q99 = 16 x 10.0452 = 160.72 units
Lognormal: sigma_ln^2 = ln(1 + 32^2 / 64^2) = ln(1.25) = 0.22314
           sigma_ln = 0.47238
           mu_ln = ln(64) - 0.22314 / 2 = 4.15888 - 0.11157 = 4.04731
           q95 = exp(4.04731 + 1.6449 x 0.47238)
               = exp(4.04731 + 0.77684) = exp(4.82415) = 124.47 units
           q99 = exp(4.04731 + 2.3263 x 0.47238)
               = exp(4.04731 + 1.09892) = exp(5.14623) = 171.77 units
```

The three agree within 7% at the 95th and diverge by up to 24% at the 99th, with the normal
lowest at both. Values use the incomplete gamma quantile function and the normal quantile.

### Exercise 3 arithmetic

```text
skewness(L) = 1.46 / sqrt(L)
L = 1: 1.46 / 1.000 = 1.460
L = 6: 1.46 / 2.449 = 0.596
L = 30: 1.46 / 5.477 = 0.267
q95 error as a percentage of the exact value:
L = 1: (6.00 - 4.89) / 6.00 = 18.5%
L = 6: (18.00 - 17.66) / 18.00 = 1.9%
L = 30: (67.00 - 66.02) / 67.00 = 1.5%
shortage ratio at the exact q95:
L = 1: 5.86; L = 6: 1.87; L = 30: 1.40
```

The quantile error falls below 2% by a 6-day horizon. The expected-shortage error falls below
20% only between 30 and 90 days: at 30 days the ratio is 1.40, an error of 29%, and at 90 days it
is 1.22, an error of 18%. The two convergence horizons differ by more than an order of magnitude.

### Exercise 4 arithmetic

```text
z          = (19 - 9.52) / 5.1955 = 9.48 / 5.1955 = 1.8246
1 - Phi(z) = 0.0340
phi(z)     = 0.0755
sigma phi(z)      = 5.1955 x 0.0755 = 0.3923
(Q - mu)(1 - Phi) = 9.48 x 0.0340 = 0.3223
normal shortage   = 0.3923 - 0.3223 = 0.0700 units per cycle
exact shortage    = 0.1515 units per cycle
ratio             = 0.1515 / 0.0700 = 2.16
conditional expected shortfall:
exact = 0.1515 / 0.0426 = 3.556 units
normal = 0.0700 / 0.0340 = 2.059 units
fill rate:
exact = 1 - 0.1515 / 9.52 = 0.9841
normal = 1 - 0.0700 / 9.52 = 0.9926
```

The normal halves the expected shortage and the conditional severity while placing the 95th
percentile within one unit. A quantile backtest is not a validation of a fill-rate model.

### Exercise 5 arithmetic

```text
observed zero fraction  = 34 / 60 = 0.567
mean                    = 78 / 60 = 1.30 units per day
Poisson zero fraction   = exp(-1.30) = 0.273
dispersion index        = 6.40 / 1.30 = 4.92
demand-day arrival rate = 26 / 60 = 0.4333 per day
mean size               = 78 / 26 = 3.000 units
E[X^2] = (2(1) + 6(4) + 10(9) + 6(16) + 2(25)) / 26
       = (2 + 24 + 90 + 96 + 50) / 26 = 262 / 26 = 10.077 units squared
compound variance = 0.4333 x 10.077 = 4.367 units squared
```

The Poisson is rejected twice: zero fraction 0.567 against 0.273, and dispersion 4.92 against
1.0. The compound fit gives 4.367, still below 6.40; the residual ratio 1.47 points to a rate
that varies across days, so use a mixed Poisson or negative binomial, or split a non-stationary
series.

### Exercise 6 answers

(a) The scale is in days, so the family models a duration; for demand in units the scale must be
in units. (b) `mu_ln` and `sigma_ln` are log-scale and dimensionless: `mu_ln = 3.1`,
`sigma_ln = 0.4`, with the variable expressed as `ln(Y / 1 unit)`. (c) `lambda = 4.2 x 6 / 7 =
3.60` events over the 6-day lead time; using 4.2 overstates the mean by 16.7%.

(d) Convert the standard deviation to cases: `12 / 24 = 0.5` case, or convert the mean to eaches:
`120 x 24 = 2,880` eaches. Mixing them gives a CV of 0.1 instead of 0.0042.

(e) `lambda L = 0.5 x 21 = 10.5` events, not 1.5. The mean is understated by a factor of seven.

### Exercise 7 arithmetic

```text
rho = 0.0: Cov = 0; cross term = 0
Var = 4(2.25) + 36(1.00) + 0 + 2.25 + 0 = 47.25
sigma = 6.874; SS95 = 1.6449 x 6.874 = 11.31 units
rho = 0.5: Cov = 0.5 x 1.0 x 1.5 = 0.75
2 mu_D mu_L Cov = 2 x 2 x 6 x 0.75 = 18.00
Var = 9.00 + 36.00 + 18.00 + 2.25 + 0.5625 = 65.81
sigma = 8.112; SS95 = 13.34 units (+17.9%)
rho = 0.8: Cov = 1.20; cross term = 28.80; Cov^2 = 1.44
Var = 9.00 + 36.00 + 28.80 + 2.25 + 1.44 = 77.49
sigma = 8.803; SS95 = 14.48 units (+28.0%)
rho = -0.5: Cov = -0.75; cross term = -18.00
Var = 9.00 + 36.00 - 18.00 + 2.25 + 0.5625 = 29.81
sigma = 5.460; SS95 = 8.98 units (-20.6%)
```

Positive correlation under-protects if ignored; negative correlation over-protects. The costs are
asymmetric: only under-protection is visible as a service failure.

### Exercises 8 to 10 rubrics

**Exercise 8.** Strong: audits the arrival rate, four size probabilities, five lead-time
probabilities, and upper tail cells with the count behind each; notes that the 99th depends on the
4-unit size (about 3 observations in 32 events) and the 8-day lead time (about 4 in 40), and
concludes it is unsupported at the stated precision. Weak: lists parameters without counts and
calls the fit adequate because mean and variance match.

**Exercise 9.** Strong: names quantity, unit, horizon, family, and every parameter with a unit;
states MM or MLE and why; gives the window with an inclusion rule; validates at the 90th, 95th,
and 99th percentiles and on expected shortage; names the service measure and a sensitivity plan.
Weak: names a family and a mean, omits the window, and validates with one fit statistic.

**Exercise 10.** Strong: separates automatic selection, optimality, the 30%, and the service
measure; notes that "optimal" requires a cost function, that 30% has no baseline, and that
selection on the body can worsen the tail; asks for baseline family, window, measure, and tail
validation. Weak: says the vendor is probably exaggerating without naming what would falsify it.

## 14. Interview questions

**1. "Which distribution would you use for lead-time demand, and why?"**
Strong: starts from mechanism and support — compound Poisson for intermittent demand with random sizes, gamma or lognormal for a continuous positive quantity, normal only for many small independent terms away from the tail — and names each parameter's unit.
Weak: names a common family, or says "normal, because of the central limit theorem", without counting terms.

**2. "What does the central limit theorem actually license here?"**
Strong: states the iid finite-variance condition, notes that `n` is demand events rather than days, and names short horizon, low count, and dependence or random `N` as failures; adds that the theorem promises no rate and nothing about the tail.
Weak: says a sum of many random variables is normal and stops.

**3. "When is a Poisson model wrong for demand counts?"**
Strong: computes the dispersion index and zero fraction. Poisson requires variance equal to the mean and predicts `exp(-mean)` zeros; lumpy demand is overdispersed with excess zeros, so use a negative binomial or compound model.
Weak: says Poisson is for counts, which is true and insufficient.

**4. "Why can a model be right at the 95th percentile and wrong about fill rate?"**
Strong: distinguishes the quantile, one point of the CDF, from expected shortage, an integral of the tail; gives `CSL = F(Q)` and `beta = 1 - E[(Y-Q)^+]/E[Y]`; notes the normal understates the loss even when `z` is accurate.
Weak: treats CSL and fill rate as two names for one thing.

**5. "How would you validate a fitted demand distribution?"**
Strong: validates in the tail — empirical against fitted quantiles at 90, 95, and 99, expected shortage at the planned stock, and stability across subperiods — and notes that a goodness-of-fit test is dominated by the body.
Weak: reports an R-squared or chi-square statistic and stops.

**6. "Method of moments or maximum likelihood?"**
Strong: gives the trade-off — MM is closed-form and transparent but inefficient and can return infeasible parameters in small samples; MLE is efficient but needs a correct support, fails with zeros, and is sensitive to the smallest observations — and reports standard errors either way.
Weak: says MLE is always better.

**7. "What if demand and lead time are correlated?"**
Strong: says the independent formula omits a covariance term, that positive correlation widens the distribution and understates the buffer, and that a scatter plot checks the sign; cites the general random-sum result rather than assuming the product model.
Weak: says correlation does not matter for the mean.

**8. "You have two years of data but only 30 demand events. What can you estimate?"**
Strong: the arrival rate and mean size with wide standard errors, the size distribution if classes are coarse, but not the 99th percentile; proposes pooling, wider classes, or a policy buffer with a stated rationale.
Weak: says two years is a large sample.

**9. "Why not just use the empirical distribution?"**
Strong: it is the best description of the observed range and the worst beyond it; it cannot produce a 99th percentile past the maximum observation, is lumpy at small samples, and imposes no mechanism, so it is a validation benchmark and a poor tail policy.
Weak: says empirical is always better because it makes no assumptions.

**10. "When would you not use lead-time demand at all?"**
Strong: names order crossover, where the inventory shortfall distribution is the right object, and highly seasonal or promotional demand, where the lead-time window is not the decision window; may add multi-echelon settings where the relevant lead time is net of the echelon's own stock.
Weak: says never.

## 15. Key terms

- **Lead-time demand (LTD)** — demand accumulated over a replenishment lead time, in units.
- **Random sum** — a sum whose number of terms is random, as in compound demand.
- **Compound Poisson** — a Poisson number of events, each with a random size.
- **Poisson rate** — expected events per interval; scales linearly with the interval.
- **Dispersion index** — variance divided by the mean; 1 for Poisson, above 1 for overdispersed.
- **Overdispersion** — variance greater than the mean in count data.
- **Gamma distribution** — right-skewed positive family with dimensionless shape and unit scale.
- **Lognormal distribution** — a positive family whose logarithm is normal; parameters are log-scale and dimensionless.
- **Normal distribution** — symmetric family on the real line, defined by mean and variance.
- **Support** — the set of values a variable can take; a hard test for family choice.
- **Coefficient of variation (CV)** — standard deviation divided by the mean; dimensionless.
- **Skewness** — third standardised moment; drives the normal's tail error.
- **Central limit theorem (CLT)** — convergence of standardised sums to normality as `n` grows.
- **Quantile (risk function)** — the inverse CDF mapping a service target to stock.
- **Expected shortfall** — mean exceedance over a threshold, `E[Y-Q | Y > Q]`.
- **First-order loss function** — `E[(Y-Q)^+]`, expected units above a stock level.
- **Cycle service level (CSL)** — probability of no stock-out in a replenishment cycle.
- **Fill rate** — fraction of demand units served from stock.
- **Method of moments (MM)** — matching sample moments to theoretical moments.
- **Maximum likelihood estimation (MLE)** — maximising the sample likelihood.
- **Order crossover** — receipt of orders out of the sequence in which they were placed.

## 16. Sources

**Peer-reviewed and academic**

- Tadikamalla, P. R. (1984). "A comparison of several approximations to the lead time demand distribution." *Omega*, 12(6), 575-581. Grade A.
- Bagchi, U. (1987). "Modeling lead-time demand for lumpy demand and variable lead time." *Naval Research Logistics*, 34(5), 687-704. Grade A.
- Eppen, G. D. and Martin, R. K. (1988). "Determining Safety Stock in the Presence of Stochastic Lead Time and Demand." *Management Science*, 34(11), 1380-1390. Grade A.
- Tyworth, J. E. and O'Neill, L. (1997). "Robustness of the normal approximation of lead-time demand in a distribution setting." *Naval Research Logistics*, 44(2), 165-186. Grade A.
- Lau, H.-S. and Lau, A. H.-L. (2003). "Nonrobustness of the normal approximation of lead-time demand in a (Q, R) system." *Naval Research Logistics*, 50(2), 149-166. Grade A.
- Bradley, J. R. and Robinson, L. W. (2005). "Improved Base-Stock Approximations for Independent Stochastic Lead Times with Order Crossover." *Manufacturing & Service Operations Management*. Grade A.
- Wang, X., Zinn, W. and Croxton, K. L. (2009). "Sizing Inventory When Lead Time and Demand are Correlated." *Production and Operations Management*. Grade A.
- Rossetti, M. D. and Ünlü, Y. (2011). "Evaluating the robustness of lead time demand models." *International Journal of Production Economics*, 134(1), 159-176. Grade A.
- Disney, S. M., Maltz, A., Wang, X. and Warburton, R. D. H. (2016). "Inventory management for stochastic lead times with order crossovers." *European Journal of Operational Research*, 248(2), 473-486. Grade A.
- Tyworth, J. E. (2025). "The Gamma Distribution and Inventory Control: Disruptive Lead Times Under Conventional and Nonclassical Conditions." *Logistics*, 9(2), 67. Grade A.

**Standards and official**

- ISO 3534-1:2016, *Statistics — Vocabulary and symbols — Part 1*. Grade B.
- ISO 8601-1:2019 and ISO 8601-2:2019, *Date and time — Representations for information interchange*. Grade B.
- JCGM 100:2008, *Evaluation of measurement data — Guide to the expression of uncertainty in measurement*. Grade B.
- GS1 EPCIS 2.0, *EPC Information Services*. Grade B.

**Industry and analyst**

- ERP and advanced planning documentation reviewed in 2026: one service factor per item and a normal quantile are the standard configuration. Grade D, dated 2026.
- Vendor case studies claiming service improvement from automated distribution selection: Grade D, undated and without a stated baseline.

**Note on citations.** Findings are reported as the cited papers state them and were not replicated. No citation is invented; where no company-level case exists, the text says so. Unconfirmed DOIs are omitted.

## 17. Links

- Module map and topic sequence: [`_index.md`](_index.md)
- Unit discipline every parameter depends on: [`01_unit-of-measure-hierarchy.md`](01_unit-of-measure-hierarchy.md)
- The lead-time distribution this topic builds on: [`05_lead-time-definitions.md`](05_lead-time-definitions.md)
- Error metrics used before fitting a distribution: [`09_forecast-error-metrics.md`](09_forecast-error-metrics.md)
- The occurrence-and-size decomposition behind compound models: [`10_intermittent-metrics.md`](10_intermittent-metrics.md)
- Next module, on where the data come from: [`../02_supply_chain_data_landscape/_index.md`](../02_supply_chain_data_landscape/_index.md)
- Module survey file: [`../../../course/01A_foundations_units_time_assumptions.md`](../../../course/01A_foundations_units_time_assumptions.md)
- Forward, time-series forecasting: [`../../../course/07_time_series_forecasting.md`](../../../course/07_time_series_forecasting.md)
- Forward, hierarchical and probabilistic forecasting: [`../../../course/08_hierarchical_intermittent_probabilistic_forecasting.md`](../../../course/08_hierarchical_intermittent_probabilistic_forecasting.md)
- Forward, inventory optimisation: [`../../../course/10_inventory_optimization_multi_echelon.md`](../../../course/10_inventory_optimization_multi_echelon.md)
- Forward, supplier risk: [`../../../course/14_procurement_supplier_risk_spend.md`](../../../course/14_procurement_supplier_risk_spend.md)
- Forward, simulation under uncertainty: [`../../../course/20_digital_twins_simulation_optimization_uncertainty.md`](../../../course/20_digital_twins_simulation_optimization_uncertainty.md)
- Return to the part map: [`../_index.md`](../_index.md)
