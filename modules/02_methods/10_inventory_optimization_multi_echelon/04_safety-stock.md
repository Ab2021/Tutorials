# Safety Stock under Demand and Lead-Time Uncertainty — combining variability, lead time, and review period

> Module 10 — Inventory Optimization and Multi-Echelon Planning
> Topic 04 of 10
> Estimated study time: 3.5 hours
> Prerequisites: Module 08 topics 06 and 10; Module 10 topics 01–03
> Version 2026.5

---

## 1. Why this topic exists

A planner opens a material master record. There is a field called safety stock. It contains the
number 194. Nobody in the room can say what protection period that number assumes, what service
measure it was set for, or which standard deviation produced it. The number is correct in the
narrow sense that a formula produced it and the formula has been in use since before anyone
present can remember. It is unverifiable in every sense that matters.

This is the normal state of affairs, and it is expensive in a way that does not appear on any
report. A buffer that is 40 per cent too small does not cause a visible failure. It causes a
service level that sits three to five points below target, permanently, with the shortfall
distributed across items and weeks so that no single incident is large enough to investigate.
The organisation concludes that the target is aspirational, then stops believing service metrics
altogether. That is a cultural cost that started as an arithmetic omission.

Three specific failures recur, in ascending order of how hard they are to see.

1. **The review period is left out of the protection period.** A system computes the buffer over
   the lead time only. If the company reviews weekly and the lead time is three weeks, the buffer
   covers three weeks of a four-week exposure. The understatement is 13 per cent in the standard
   deviation; with monthly review and a two-week lead time it is 42 per cent. Nothing in the
   output looks wrong, because the output is a number with no visible derivation.
2. **The lead-time variance term is dropped.** The classical formula has two terms: demand
   variance over the protection period, and mean demand squared times lead-time variance. Many
   implementations, and many spreadsheets, contain only the first. For a high-volume item with a
   moderately unreliable supplier, the second term can be two thirds of the variance. Dropping it
   halves the buffer. Section 6 works a case where it costs 42 per cent of the buffer.
3. **The standard deviation is measured on the wrong object.** It is computed from shipments
   rather than demand, from a window that contains a promotion, or from sales that were censored
   by stock-outs. Each of these biases the estimate, and none of them leaves a trace in the
   number that reaches the planner.

The third failure dominates the first two, and this is the uncomfortable point of the topic. A
badly chosen formula with a well-estimated standard deviation produces a buffer that is wrong by
tens of per cent in a known direction. A good formula with a standard deviation estimated from
the wrong series produces a buffer that is wrong by more, in an unknown direction, and looks
rigorous while doing it. Section 5 ranks the error sources; the ranking is not the one most
improvement programmes assume.

> **Note:** Lead-time demand distributions and the compounding of demand and supply variability
> are treated in Module 08 topic 06 and Module 10 topic 07. Censored demand estimation is treated
> in Module 09 topic 09. None of those is re-derived here. This file is about the operational
> buffer formula, the assumptions it encodes, and the reporting discipline that keeps it honest.

## 2. Learning outcomes

By the end of this topic you can:

1. **Define** the protection period as review period plus lead time, **draw** it on a timeline,
   and **explain** why a periodic-review system that buffers over lead time alone is
   systematically short.
2. **Derive** the standard formula from the variance of demand over the protection period, and
   **state** each independence assumption at the point in the derivation where it is used.
3. **Extend** the formula to a random lead time, **identify** the term most implementations omit,
   and **compute** the error when it is dropped for a given item.
4. **Distinguish** safety stock, safety lead time, and the order-up-to level, and **determine**
   which object a named planning system actually stores.
5. **Choose** between cycle service level (CSL) and fill rate, **convert** each to its multiplier,
   and **explain** why the fill-rate multiplier depends on the order quantity.
6. **Estimate** a demand standard deviation defensibly — window, bucket, promotions, censoring —
   and **quantify** the sampling error that remains.
7. **Explain** how forecast-error autocorrelation violates independence, **apply** a correction
   factor, and **state** when the correction is material.
8. **Diagnose** a short-lifecycle item where the formula does not apply, and **name** the decision
   object that replaces it.
9. **Write** a buffer specification that states its protection period, service measure, standard
   deviation source, and the error the formula assumes away.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold one picture: a timeline on which a commitment is made, material arrives later, and the next
chance to correct arrives later still. The buffer exists to cover the whole span between the
commitments, not the transport time.

```text
  units
    ^
    |            .   .   .   .   .   .   .   .   .   .
    |        .                                       .
    |      .        demand over the protection period    .
    |    .       (mean mu_d per week, sd sigma_d)      .
    |   .   .   .   .   .   .   .   .   .   .   .   .   .   .
    |  /                                                   \
    +------------------------------------------------------------> time (weeks)
       0     1     2     3     4     5     6
       |     |     |     |     |     |
       R0    R1    R2    R3    R4    R5          review points, R = 1 week
       |           |
       O0          A0                            order placed at R0, arrives at week 3
       |<- L = 3 ->|
       |<-------- P = R + L = 4 weeks -------->|

  The order placed at R0 must cover demand until the order placed at R1 arrives.
  That is week 4, not week 3. The last week is the review period, and it is
  inside the exposure whether or not the formula knows it.
```

Read the diagram as an argument about when you can act. At week 0 you place an order that arrives
at week 3. If demand in weeks 1 to 4 is higher than expected, you cannot correct before week 4,
because the next order was placed at week 1 and arrives at week 4. The buffer therefore has to
cover weeks 0 to 4. The exposure is the **protection period** `P = R + L`, and the buffer is a
quantile of the demand distribution over that whole span.

Three consequences follow immediately, and they organise everything else in the topic.

**Consequence 1 — the buffer is a quantile, not a quantity.** Safety stock is not "some extra
units". It is the difference between a chosen quantile of protection-period demand and the mean of
that demand. Change the quantile and you have changed the buffer, with no change in the physical
world. That is why a service target is a policy input and not a modelling output.

**Consequence 2 — the variance grows with the span, but not proportionally to the span.** Under
independent demand the variance of demand over `P` periods is `P` times the one-period variance,
so the standard deviation grows with the square root. Doubling the review period from one week to
two raises the demand-driven standard deviation by 41 per cent, not by 100. This is the square-root
law, and it is the reason a monthly review does not require four times the buffer of a weekly one.

**Consequence 3 — a random lead time adds a term that scales with volume squared.** When the lead
time is itself random, the demand accumulated over it is a random sum, and its variance has two
pieces: the average span times the demand variance, and the mean demand rate squared times the
lead-time variance. The second piece contains `mu_d^2`, so it grows with the square of volume.
A high-volume item with a slightly unreliable supplier is far more exposed than a low-volume item
with the same supplier, and the classical one-term formula treats them as equally exposed.

### 3.2 What "the formula" is, stated once

For a periodic-review system with protection period `P = R + L`, independent demand with mean
`mu_d` and standard deviation `sigma_d` per period, independent lead time with mean `L` and
standard deviation `sigma_L`, and a service factor `z`:

```text
   safety stock  SS = z * sqrt( (R + L) * sigma_d^2  +  mu_d^2 * sigma_L^2 )
   order-up-to   S  = mu_d * (R + L) + SS
```

The first term inside the square root is demand variability over the protection period. The
second is lead-time variability, expressed in units by multiplying the lead-time standard
deviation by the demand rate. When the lead time is deterministic the second term vanishes and
the formula reduces to `z * sigma_d * sqrt(P)`. When the review period is ignored, `P` becomes `L`
and the first term shrinks.

> **Note:** The multiplication `mu_d * sigma_L` is dimensionally exact only if `mu_d` is in units
> per period and `sigma_L` is in periods. If lead time is recorded in days and demand in units per
> week, convert one of them first. A unit mismatch here produces a buffer wrong by a factor of
> seven and is invisible in the output.

### 3.3 The three objects a planning system can store

The same physical intent can be encoded three ways, and the encoding determines what an analyst
can and cannot see.

| Object stored | What the number means | Where you meet it | What it hides |
|---|---|---|---|
| Safety stock quantity | Units held above expected demand | Material master, buffer field | The protection period and service measure behind it |
| Safety lead time | Days added to the planned lead time | Planning item record | The quantity consequence, which is demand-rate dependent |
| Order-up-to level | Target position after ordering | Periodic-review policy | The split between cycle stock and buffer |
| Reorder point | Position at which to order | Continuous-review policy | The order quantity, which affects fill rate |
| Days of supply | Coverage expressed in time | Retail and distribution systems | Volume, so two items look equal at very different risk |

An analyst asked to "check the safety stock" must first establish which of these the system
stores. A quantity of 194 units and a policy of "four weeks of cover" can describe the same
position, and a change to one is a change to the other only through the demand rate. Section 6.4
shows why this matters when a forecast is revised.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Safety stock | Buffer held above expected demand over the protection period to absorb variability | Treated as a policy constant independent of service target | Units |
| Buffer | Any stock held to decouple a stage from variability, including safety and cycle stock | Used as a synonym for safety stock only | Units |
| Protection period | The span a buffer must cover: review period plus lead time | Assumed equal to the lead time | Weeks or days |
| Review period (R) | Interval between replenishment decisions | Confused with the forecast refresh interval | Weeks |
| Lead time (L) | Elapsed time from order release to availability at the stocking point | Used to mean transit time only | Weeks |
| Lead-time demand | Demand accumulated over the protection period | Computed as rate times a rounded horizon | Units per cycle |
| Order-up-to level (S) | Target inventory position after an order is placed | Confused with the buffer itself | Units |
| Reorder point (ROP) | Inventory position at which a replenishment order is triggered | Reported without the order quantity | Units |
| Cycle service level (CSL) | Probability that demand over a replenishment cycle is met from stock | Reported as fill rate | Proportion |
| Fill rate | Expected fraction of demand units served from stock in a cycle | Reported as CSL | Proportion |
| Loss function | Expected units short per cycle at a given standardised buffer, `L(z)` | Omitted when a fill target is converted to a multiplier | Units per standard deviation |
| Service factor (z) | Standardised quantile of the unit normal distribution implied by the service measure | Chosen from a remembered table without the measure being named | Dimensionless |
| Sigma of demand | Standard deviation of demand per period, in the bucket of the model | Computed on shipments rather than demand | Units per period |
| Sigma of lead time | Standard deviation of realised lead time | Reported as a percentage of the mean | Weeks |
| Coefficient of variation | Standard deviation divided by the mean | Compared across different buckets | Dimensionless |
| Demand construct | The definition of the quantity being modelled: consumption, orders, or shipments | Left unstated | Enumeration |
| Time bucket | The period over which demand is aggregated | Changed without recomputing sigma | Day, week, month |
| Forecast error | Realised demand minus forecast for the same period | Equated with demand variability | Units per period |
| Forecast bias | Mean of the forecast error over a window | Confused with accuracy | Units per period |
| Error autocorrelation | Correlation between forecast errors at different lags | Assumed zero because the model is "statistical" | Dimensionless |
| Independence assumption | Demand in one period carries no information about the next | Never tested | Model property |
| Stationarity | The demand process parameters do not change over the estimation window | Assumed from a long history | Model property |
| Safety lead time | Planned time added to the lead time so material arrives earlier than needed | Used interchangeably with safety stock | Days |
| Cycle stock | Stock that exists because orders are placed in batches | Included in the safety stock figure | Units |
| Pipeline stock | Goods owned but not yet available for sale | Omitted from the position | Units |
| Decoupling point | The point where a buffer separates a downstream stage from upstream variability | Placed by convention | Node and stage |
| DDMRP | Demand driven material requirements planning; buffers sized from average daily usage and decoupling lead time | Described as safety stock with a different name | Method |
| Days of supply | Coverage available or targeted, expressed in time | Converted to units without stating the demand rate | Days |
| Periodic review (R,S) | Policy that orders up to `S` every `R` periods | Modelled as continuous review | Policy |
| Continuous review (s,Q) | Policy that orders `Q` when the position falls to `s` | Modelled with a review period it does not have | Policy |
| Censoring | Observed demand truncated at available stock | Treated as a data-cleaning problem | Observation property |
| Lost sale | Demand that disappears when stock is unavailable | Recorded as zero demand | Units |
| Backorder | Demand deferred rather than lost | Assumed equivalent to a lost sale | Units |
| Standard error | Standard deviation of an estimator across hypothetical samples | Reported as the standard deviation of the data | Units |
| Estimation window | The historical span used to estimate parameters | Chosen by habit | Weeks |
| Intermittent demand | Demand with periods of zero and irregular positive values | Buffered with a normal formula | Classification |
| Promotional period | A period whose demand is generated by a temporary intervention | Left inside the sigma window | Calendar flag |
| Substitution | Demand for one item met by another | Ignored when computing item-level sigma | Units |

Two rows deserve expansion, because each one is where a real argument happens.

**Safety stock versus buffer.** A buffer is any stock that decouples; safety stock is the part
computed from a service target and a variability estimate. Cycle stock also decouples, and in some
policies the two are not separable, because the order-up-to level is a single number and the
"buffer" is whatever is left after expected demand. When a planner says "we hold two weeks of
safety stock", ask whether that is a quantity above the mean or a total position, because the two
differ by the cycle stock and can differ by a factor of two.

**Protection period versus lead time.** These are equal only under continuous review. Under
periodic review the protection period exceeds the lead time by exactly one review interval, and
the buffer must cover it. The distinction is not academic: it is the difference between a 95 per
cent target and an 89 per cent outcome for a weekly review with a three-week lead time, and worse
for a monthly review.

## 5. The mechanics, step by step

The procedure below is written as a sequence because the errors compound: a wrong bucket makes the
window wrong, a wrong window makes sigma wrong, and a wrong sigma makes every later step
meaningless. Each step states its input, its output, and the branch where analysts diverge.

### Phase A — Fix the object before computing anything

**Step 1. Name the item, the stocking location, and the unit.**

- Input: a material or item identifier and a location.
- Output: a single series key with a stated unit of measure.
- Branch: **is the unit the selling unit or the stocking unit?** A case-packed item can be held in
  eaches, cases, or pallets, and a conversion error is a factor-of-twelve error that survives every
  downstream check because the arithmetic is internally consistent.

**Step 2. Fix the bucket and keep it fixed.**

- Input: the replenishment or planning frequency.
- Output: a bucket (day, week, month) used for sigma, for the protection period, and for the
  service measurement.
- Branch: **is the bucket finer or coarser than the decision?** A daily bucket with a weekly
  decision adds noise to sigma without adding information; a monthly bucket with a weekly decision
  destroys the within-month pattern that generates the buffer requirement.

**Step 3. Name the demand construct.**

- Input: the available series — consumption, customer orders, shipments, or point-of-sale scans.
- Output: a one-line definition of what the series represents.
- Branch: **does the series contain the variability you are buffering against?** Shipments absorb
  batching and are smoother than demand. Orders contain gaming. Scans are closest to demand and
  are rarely available upstream. The construct determines which variability the buffer covers,
  and therefore what it does not cover.

**Step 4. Identify the review discipline.**

- Input: the planning calendar and the ordering rule.
- Output: a statement of whether the system reviews continuously or periodically, and at what
  interval.
- Branch: **does the system's review interval match the operational one?** A system nominally
  reviewing daily but constrained by a weekly order cut-off behaves as weekly review. The
  protection period follows behaviour, not configuration.

### Phase B — Build the protection period

**Step 5. Add the review period to the lead time.**

- Input: `R` and `L`, both in the same unit.
- Output: `P = R + L`.
- Reasoning: the order placed at a review must cover demand until the *next* order arrives. The
  next order is placed one review period later and arrives one lead time after that.

**Step 6. Check whether the system already includes it.**

- Input: the system's buffer configuration or documentation.
- Output: a statement of the protection period the system assumes.
- Branch: **if the system's lead time field includes a review allowance, do not add it twice.**
  Double-counting is less common than omitting, but it happens when a legacy field was
  repurposed. The only safe check is to recompute the buffer by hand for one item and compare.

The size of the omission depends entirely on the ratio. In variance terms the omitted share is
`R / (R + L)`; in standard-deviation terms it is `1 - sqrt(L / (R + L))`.

| Review period (R) | Lead time (L) | P = R + L | Share of variance omitted | Buffer understatement |
|---:|---:|---:|---:|---:|
| 1 day | 3 weeks | 3.2 weeks | 0.6% | 0.3% |
| 1 week | 3 weeks | 4 weeks | 25.0% | 13.4% |
| 2 weeks | 3 weeks | 5 weeks | 40.0% | 22.5% |
| 4 weeks | 2 weeks | 6 weeks | 66.7% | 42.3% |
| 4 weeks | 1 week | 5 weeks | 80.0% | 55.3% |

The last two rows are the ones that matter, and they describe a company that plans monthly and
buys on short lead times — a common configuration in domestic distribution, where lead times are
short enough that nobody thinks about them and the review period dominates the exposure.

### Phase C — Derive the variance, and state the assumptions where they enter

**Step 7. Write demand over the protection period as a sum.**

```text
   D_P = d_1 + d_2 + ... + d_P
```

where `d_t` is demand in period `t`. Nothing is assumed yet beyond the definition.

**Step 8. Take the variance under independence.**

```text
   Var(D_P) = sum_t Var(d_t) + 2 * sum_{s<t} Cov(d_s, d_t)
            = P * sigma_d^2                    if Cov = 0 for all s != t
```

This is where the first assumption enters: **demand is serially independent**. It is not a
technicality. If consecutive periods are positively correlated, the covariance terms are positive
and the variance is larger than `P * sigma_d^2`. Step 16 quantifies how much larger.

**Step 9. Add the random lead time via the law of total variance.**

```text
   Var(D_P) = E[ Var(D_P | L) ] + Var( E[D_P | L] )
            = E[ (R + L) * sigma_d^2 ] + Var( (R + L) * mu_d )
            = (R + L_bar) * sigma_d^2 + mu_d^2 * sigma_L^2
```

The second assumption enters here: **lead time is independent of demand**. Independence is what
allows the conditional variance to be averaged and the conditional mean to be varied separately.
If a supplier is slower when demand is high — a real pattern during promotions and capacity
crunches — there is a covariance term and the formula understates.

**Step 10. Compute what the omitted term costs before deciding to include it.**

- Input: `mu_d`, `sigma_L`, `sigma_d`, `P`.
- Output: the share of variance contributed by the lead-time term, and the buffer ratio.
- Decision rule: if `mu_d * sigma_L` exceeds roughly `sigma_d * sqrt(P)`, the lead-time term
  dominates and no defensible buffer can omit it.

```text
   share of variance from lead time = mu_d^2 * sigma_L^2 / Var(D_P)
   buffer ratio if term dropped      = sqrt( Var_with / Var_without )
```

| `mu_d * sigma_L / (sigma_d * sqrt(P))` | Share of variance from lead time | Buffer understated if dropped |
|---:|---:|---:|
| 0.25 | 5.9% | 3.0% |
| 0.50 | 20.0% | 11.8% |
| 1.00 | 50.0% | 41.4% |
| 1.50 | 69.2% | 80.3% |
| 2.00 | 80.0% | 123.6% |

Read the table as a screening rule. A ratio below about 0.3 means the omission is a rounding
error and can be documented as one. Above 1.0 it is the difference between a buffer and a gesture.

> **Note:** The ratio `mu_d * sigma_L / (sigma_d * sqrt(P))` is the coefficient of variation of the
> lead time multiplied by `sqrt(P) / CV_d`. It can therefore be computed from two coefficients of
> variation and the protection period alone, which is useful when the units are inconvenient.

### Phase D — Choose the service measure and the multiplier

**Step 11. Choose the service measure the contract or the policy names.**

- Input: the service level agreement (SLA), the internal service definition, or the policy.
- Output: one of cycle service level (CSL), fill rate, or a time-based measure such as on-time
  in-full (OTIF).
- Branch: **is the measure a probability of a cycle without stock-out, or a fraction of units
  served?** They imply different multipliers and can differ by tens of per cent of buffer.

**Step 12. Convert the measure to a multiplier.**

For CSL the conversion is direct: `z = Phi^-1(CSL)`. For fill rate the conversion runs through the
standard normal loss function `L(z) = phi(z) - z * (1 - Phi(z))`:

```text
   fill rate beta = 1 - sigma_P * L(z) / Q
   therefore      L(z) = (1 - beta) * Q / sigma_P
```

- Input: target fill rate, `sigma_P`, and the order quantity `Q`.
- Output: `z`, obtained by inverting `L(z)` numerically or from a table.
- Critical consequence: **the fill-rate multiplier depends on `Q`.** The CSL multiplier does not.
  Two items with identical variability and identical fill targets but different order quantities
  need different buffers, and a system that applies one multiplier to all items is applying a CSL
  logic to a fill-rate target.

### Phase E — Estimate sigma honestly

**Step 13. Choose the estimation window, and state why.**

- Input: available history and knowledge of structural breaks.
- Output: a window length in periods, with a justification.
- Branch: **long enough to estimate, short enough to be stationary.** The sampling error of a
  standard deviation from `n` observations is approximately `sigma / sqrt(2(n - 1))`, so a year of
  weekly data gives a standard error near 10 per cent and a 95 per cent interval near plus or
  minus 20 per cent. Thirteen weeks gives roughly double that.

| Window | `n` (weekly) | Standard error of `s` | Approximate 95% interval |
|---|---:|---:|---:|
| 8 weeks | 8 | 26.7% | ±52% |
| 13 weeks | 13 | 20.4% | ±40% |
| 26 weeks | 26 | 14.1% | ±28% |
| 52 weeks | 52 | 9.9% | ±19% |
| 104 weeks | 104 | 7.0% | ±14% |

**Step 14. Decide what to do with promotional and outlier periods.**

- Input: a promotion calendar, a price-change log, and a plot of the series.
- Output: either an exclusion rule, a segmentation, or an explicit decision to retain them.
- Branch: **is the promotion part of the world you are buffering against, or an artifact of the
  window?** If promotions recur on a known calendar, the buffer must cover them and excluding them
  understates. If the window contains one exceptional event, keeping it overstates for years.

**Step 15. Diagnose censoring before computing sigma.**

- Input: inventory availability flags, stock-out dates, or service logs.
- Output: the share of periods in which demand was truncated, and an adjustment decision.
- Branch: **was the item unavailable in enough periods to matter?** Observed sales are
  `min(demand, available stock)`, so the sample standard deviation of sales is biased downward,
  and the bias grows with the stock-out frequency. Module 09 topic 09 treats the estimators; here
  the requirement is only that the share be reported.

### Phase F — Test what the formula assumes away

**Step 16. Test forecast-error autocorrelation.**

For a forecast produced by a simple moving average over `M` periods, the errors over the lead time
are positively correlated even when demand is independent, because the estimation error in the
mean is common to every period. The exact variance is `L * sigma^2 + L^2 * Var(forecast)`, and the
correction factor on the buffer is `sqrt(1 + L / M)` against the naive substitution.

| `M` (window) | `L` (periods) | Correction factor | Implied error correlation |
|---:|---:|---:|---:|
| 52 | 4 | 1.038 | 0.019 |
| 26 | 4 | 1.074 | 0.037 |
| 12 | 4 | 1.155 | 0.077 |
| 8 | 4 | 1.225 | 0.111 |
| 12 | 3 | 1.118 | 0.077 |

An error correlation of four per cent sounds negligible and produces a 7 per cent buffer
understatement. The correlation is small because the variance contribution is `L^2` while the
correlation is diluted by the idiosyncratic term; the `L` multiplier is what makes it matter.

**Step 17. Test whether demand and lead time move together.**

- Input: paired observations of demand intensity and realised lead time.
- Output: a correlation, and a statement of whether the independence assumption is defensible.
- Branch: **is the correlation driven by a shared cause?** Promotions raise demand and strain
  supply simultaneously; capacity crunches raise lead times and suppress availability. A positive
  correlation means the formula understates; a negative one means it overstates.

**Step 18. Check the normality of the quantile you are using.**

- Input: the empirical distribution of protection-period demand, or its skewness.
- Output: a decision to keep the normal multiplier, use an empirical quantile, or switch to a
  compound distribution.
- Branch: **how far into the tail is the target, and how large is the mean relative to the
  standard deviation?** The normal approximation is adequate near the centre and deteriorates as
  the target approaches 99 per cent, as the coefficient of variation rises, and as the item becomes
  intermittent. Below roughly ten expected units over the protection period, use the exact
  distribution.

### Phase G — Decide what is stored, and report it

**Step 19. Decide which object the system should hold.**

- Input: the review discipline, the planning engine, and the ownership of the number.
- Output: a decision to store a quantity, a days-of-supply equivalent, or a computed level.
- Reasoning: a stored quantity goes stale as the demand rate changes; a computed level stays
  current but changes without warning; a days-of-supply target survives forecast revisions but
  hides volume. Each has a failure mode, and the specification must name the one accepted.

**Step 20. Write the reporting card.**

A buffer is not specified until four things are stated with it: the protection period, the service
measure, the standard deviation source, and the error the formula assumes away.

```text
   1. Item, location, unit:       4417-B, DC North, kits
   2. Bucket and window:          weekly, 52 weeks to 2026-09-30
   3. Protection period:          R = 1 wk, L = 3.0 wk, P = 4.0 wk
   4. Service measure and target: CSL 95%, z = 1.645
   5. Sigma source:               demand (not shipments), promotions excluded,
                                  stock-out weeks excluded (9 of 52, 17%)
   6. Buffer:                     194 kits (sigma_P = 117.6 kits)
   7. Assumed away:               lead-time independence of demand;
                                  autocorrelation corrected by 1.07 for M = 26;
                                  normal tail; parameters treated as known
```

Line 7 is the one that is always missing and the one that makes the number auditable. It is not a
disclaimer. It is the list of conditions under which the number is the right number.

## 6. Worked example

One item at one distribution centre, carried end to end. All figures are constructed to be
internally consistent; the item is synthetic and the arithmetic is reproducible by hand.

###  The item card

| Quantity | Value | Unit |
|---|---:|---|
| Item | 4417-B | identifier |
| Description | Replacement seal kit, three pump models | — |
| Location | DC North | identifier |
| Unit of measure | kit | eaches |
| Mean demand (`mu_d`) | 120 | kits per week |
| Standard deviation of demand (`sigma_d`) | 34 | kits per week |
| Coefficient of variation | 0.283 | dimensionless |
| Review period (`R`) | 1 | week |
| Mean lead time (`L`) | 3.0 | weeks |
| Standard deviation of lead time (`sigma_L`) | 0.8 | weeks |
| Target service | 95% cycle service level | proportion |
| Service factor (`z`) | 1.645 | dimensionless |
| Holding cost | 22% per year of unit cost | proportion per year |
| Unit cost | EUR 18.40 | currency per kit |

The coefficient of variation of demand is 0.283, which is ordinary for a replacement part sold
through a distribution centre. The lead time has a coefficient of variation of 0.267, which is
also ordinary for a supplier shipping from a regional warehouse with occasional stock-outs of its
own. Neither number is alarming. Together they produce the result in section 6.3.

###  Case A — deterministic lead time

Take the lead time as fixed at three weeks, which is what most spreadsheet implementations assume
implicitly.

```text
   P        = R + L      = 1 + 3         = 4.0 weeks
   Var(D_P) = P * sigma_d^2 = 4 * 1156   = 4,624 kits^2
   sigma_P  = sqrt(4,624)                = 68.00 kits
   SS       = 1.645 * 68.00              = 111.9 -> 112 kits
   S        = mu_d * P + SS = 480 + 112  = 592 kits
```

The buffer is 112 kits, or 0.93 weeks of mean demand. The order-up-to level is 592 kits, which is
four weeks of mean demand plus the buffer. This is the number that a system using the one-term
formula will store, and it is the number that most planners would defend if asked.

###  Case B — the same item with lead-time variance

Now restore the lead-time standard deviation of 0.8 weeks. Nothing else changes: same demand, same
review period, same target.

```text
   Var_demand = P * sigma_d^2        = 4 * 1156        = 4,624
   Var_lead   = mu_d^2 * sigma_L^2   = 14,400 * 0.64   = 9,216
   Var_total  = 4,624 + 9,216                          = 13,840
   sigma_P    = sqrt(13,840)                          = 117.64 kits
   SS         = 1.645 * 117.64                        = 193.5 -> 194 kits
   S          = 480 + 194                             = 674 kits
```

Two numbers describe the change. The lead-time term contributes
`9,216 / 13,840 = 66.6%` of the variance and `sqrt(9,216) / 117.64 = 81.6%` of the standard
deviation. The buffer rises from 112 to 194 kits, a factor of 1.73. Dropping the term understates
the buffer by `1 - 112/194 = 42.3%`.

That is the cost of one omitted multiplication. Expressed in money at EUR 18.40 per kit and a 22
per cent annual holding cost, the difference is `82 kits * 18.40 * 0.22 = EUR 332` per year for
this item. Across a catalogue of eight hundred items with similar volumes, the same omission is
not a rounding error; it is a service programme that never reaches its target.

The screening ratio from step 10 is `mu_d * sigma_L / (sigma_d * sqrt(P)) = 96 / 68 = 1.41`, which
the table places between the 1.0 and 1.5 rows: the lead-time term should dominate and it does.
An analyst who computes this ratio before choosing a formula knows in one line whether the
distinction matters for the item in front of them.

### 6.4 The order-up-to level, and what the system actually stores

The buffer of 194 kits is not the number a planner sees. In a periodic-review policy the system
stores the order-up-to level, 674 kits, and derives the buffer as the difference between that
level and expected demand over the protection period.

| Object | Value | Definition | What changes it |
|---|---:|---|---|
| Expected demand over `P` | 480 kits | `mu_d * (R + L)` | A forecast revision |
| Safety stock | 194 kits | `z * sigma_P` | A service target or sigma change |
| Order-up-to level | 674 kits | Sum of the two | Both |
| Reorder point equivalent | 480 + 194 | Same as `S` under review at every period | Policy family |

### 6.5 The service measure changes the multiplier

Everything above uses a 95 per cent cycle service level, `z = 1.645`. Replace it with a 95 per cent
fill rate, which is what many commercial contracts actually specify.

```text
   L(z)  = (1 - beta) * Q / sigma_P
   Q     = mu_d * R = 120 kits               (one week of demand per order)
   L(z)  = 0.05 * 120 / 117.64 = 0.0510
   z     ~ 1.25                              (from the loss-function table)
   SS    = 1.25 * 117.64 = 147 kits
```

The buffer falls from 194 to 147 kits, a reduction of 24 per cent, with no change to the physical
world and no change to the words "95 per cent". The mechanism is that fill rate counts units, and
a stock-out in a week with small demand costs fewer units than one in a week with large demand,
so the same probability of a stock-out produces a lower expected shortfall.

If the order quantity is larger, the effect is much bigger. At `Q = 480` kits, four weeks of
demand per order, `L(z) = 0.05 * 480 / 117.64 = 0.204`, which gives `z` near 0.50 and a buffer of
59 kits. Fill rate is a ratio of expected values and therefore depends on the replenishment
quantity; CSL does not. A planning system that computes one multiplier per item class and applies
it to both measures is wrong for at least one of them.

### 6.6 The autocorrelation correction

Suppose the standard deviation of 34 kits per week was estimated from a 26-week simple moving
average, which is a common configuration. The correction factor from step 16 is
`sqrt(1 + L/M) = sqrt(1 + 4/26) = 1.074`.

```text
   sigma_d corrected = 34 * 1.074 = 36.5 kits/week
   Var_demand        = 4 * 1,333  = 5,332
   Var_total         = 5,332 + 9,216 = 14,548
   sigma_P           = 120.6 kits
   SS                = 1.645 * 120.6 = 198.4 -> 198 kits
```

The buffer rises by 2 per cent, from 194 to 198, because the lead-time term dominates and the
correction applies only to the demand term. On an item with a short, reliable lead time the same
correction would raise the buffer by 7.4 per cent. The general rule: **the autocorrelation
correction matters in proportion to the demand term's share of the variance**, so it is largest
for domestic, short-lead-time, high-variability items — again, the ones nobody audits.

### 6.7 Safety stock or safety lead time

The supplier offers to accept orders one week earlier, which is a safety lead time of one week. The
cost is one week of pipeline stock, `120` kits, held continuously. The alternative is to keep the
safety stock that covers lead-time variability, `1.645 * 120 * 0.8 = 157.9` kits.

```text
   safety lead time of 1 week      costs 120 kits, covers 1.0 / 0.8 = 1.25 sigma_L
   lead-time safety stock term     costs 158 kits, covers 1.645 sigma_L
   cost per sigma_L covered:       96 kits in both cases
```

Within the normal model the two are priced identically per unit of lead-time coverage. They differ
in what they cover. The safety lead time protects against lateness only, and only up to one week,
which is the 89th percentile of this lead-time distribution. The safety stock protects against
demand and lateness jointly and covers the full 95 per cent target. One week of safety lead time
also lengthens the protection period to five weeks, which raises the demand term by 12 per cent and
the buffer by about 13 kits, so the total cost is 133 kits against 158. The comparison is close
enough that the decision should turn on which risk is real — a supplier that is late but reliable
in quantity, or one that is punctual but erratic in what it ships.

### 6.8 The error budget for this item

| Source of error | Size on the buffer | Direction | Detectable from the number alone? |
|---|---:|---|---|
| Lead-time variance term omitted | 42% | Understates | No |
| Review period omitted | 5% | Understates | No |
| CSL used for a fill-rate target | 24% | Overstates | No |
| Sigma from 26 weeks | ±14% | Symmetric | No |
| Autocorrelation uncorrected | 2% | Understates | No |
| Censoring in 9 of 52 weeks | 5–15% | Understates | Only with availability flags |
| Promotions left in the window | 10–25% | Usually overstates | Only with a calendar |

The table is the point of the worked example. The largest single error is a formula error that can
be fixed in a spreadsheet. The next largest is an estimation error that requires a data decision.
The smallest is the one that most quality programmes measure first. An analyst who presents the
buffer with this table attached has done something more useful than an analyst who presents a more
accurate buffer with no table.

## 7. Code, query, or calculation

The calculation below is the whole of section 6 in twenty lines, written so that it can be
reproduced on paper or in any language. It is deliberately not library code: a safety stock
function that returns 194 without showing the two variance terms is a function nobody can audit.

```text
Item 4417-B, weekly bucket, units = kits
inputs
  mu_d    = 120 kits/week       sigma_d = 34 kits/week
  R       = 1 week              L_bar   = 3.0 weeks
  sigma_L = 0.8 weeks           z(95% CSL) = 1.645

CASE A - deterministic lead time (sigma_L = 0)
  P        = R + L_bar            = 1 + 3.0           = 4.00 weeks
  Var      = P * sigma_d^2        = 4 * 1156          = 4,624
  sigma_P  = sqrt(4,624)                              = 68.00 kits
  SS       = 1.645 * 68.00                            = 111.9 -> 112 kits
  S        = mu_d * P + SS        = 480 + 112         = 592 kits

CASE B - same item, lead time random (sigma_L = 0.8)
  Var_dem  = P * sigma_d^2        = 4 * 1156          = 4,624
  Var_lead = mu_d^2 * sigma_L^2   = 14400 * 0.64      = 9,216
  Var_tot  = 4,624 + 9,216                            = 13,840
  sigma_P  = sqrt(13,840)                             = 117.64 kits
  SS       = 1.645 * 117.64                           = 193.5 -> 194 kits
  S        = 480 + 194                                = 674 kits

  lead-time share of variance         = 9,216 / 13,840  = 66.6%
  buffer understatement if dropped    = 1 - 112 / 194   = 42.3%
```

Expected output: case A gives 112 kits and a level of 592; case B gives 194 kits and a level of
674. Four checks before you trust any implementation, including your own:

1. **Set `sigma_L` to zero.** The answer must collapse to `z * sigma_d * sqrt(P)`. Most
   implementation errors are in the second term, and this check isolates them.
2. **Halve `sigma_d`.** The buffer must fall by exactly half in case A, and by less than half in
   case B, because the lead-time term does not contain `sigma_d`.
3. **Double `mu_d`.** The demand term quadruples and the lead-time term quadruples, so the
   variance quadruples and the buffer doubles — but only if `sigma_L` is held fixed. If `sigma_L`
   was expressed as a percentage of the mean, doubling `mu_d` changes both terms and the check
   fails for a legitimate reason.
4. **Check the units of `sigma_L`.** It must be in the same unit as `L`. A lead time in days with a
   standard deviation in weeks gives a buffer that is too small by a factor of about seven, and
   nothing in the output flags it.

## 8. Data, standards, and regulatory context

No standard defines the safety stock formula. There is no ISO, ANSI, or IEC document that fixes the
protection period, the service measure, or the estimator for sigma. What the standards constrain is
the **input**: the identity of the item, the identity of the location, the meaning of the
timestamp, and the definition of the service commitment the buffer is meant to satisfy. Those
constraints are what make a buffer auditable, and they are the part that is usually missing.

### 8.1 Identity and event data

| Standard or instrument | What it governs | What it constrains here | Grade |
|---|---|---|---|
| GS1 GTIN (Global Trade Item Number) | Product identity | The item dimension; a pack-size change creates a new GTIN and splits one sigma history into two shorter ones | B |
| GS1 GLN (Global Location Number) | Location identity | The location dimension; a warehouse re-code creates a false new series | B |
| EPCIS 2.0 (GS1) | Event data: what, where, when, why | Timestamped goods issue and receipt events, the basis for realised lead times | B |
| ISO 8000 series | Data quality | Completeness and consistency of the item, location, and period keys | B |
| ISO 3534-1 | Statistical vocabulary | The meanings of estimator, variance, and confidence used in step 13 | B |
| ISO 22400-2 | Manufacturing operations key performance indicators (KPIs) | Constrains how a KPI is defined and reported; does not choose between CSL and fill rate for your contract | B |
| IAS 2 / ASC 330 | Inventory measurement | Inventories at the lower of cost and net realisable value (NRV); safety stock is a policy quantity, not an accounting category | B |
| Service level agreement (SLA) | Contract | Names the service measure the buffer must satisfy | B |

### 8.2 What is convention rather than requirement

- The `z` values 1.28, 1.65, and 2.33 for 90, 95, and 99 per cent are arithmetic, not policy. The
  policy is the target.
- A 95 per cent service target is a management choice with no regulatory standing in any
  jurisdiction this programme covers. Section 9 discusses what the evidence says about how such
  targets are actually set.
- "Safety stock equals two weeks of demand" is a habit inherited from a period when demand data
  were monthly and sigma was expensive to compute. It is not derived from anything.

## 9. Industry practice and case evidence

### 9.1 Where the formula came from, and what is not known

The practitioner literature most often cited for the protection-period form is Peter L. King's
"Crack the Code: Understanding Safety Stock and Mastering Its Equations" in *APICS Magazine*,
July/August 2011, which is available as a PDF from the MIT 2.810 course readings. King gives four
variants: demand variability alone, lead-time variability alone, the independent combination with
the square root, and a dependent form that adds the two. He also gives the `z` table and a worked
example with a plastic film at a 95 per cent target.

What the article does not give is a history. It presents the equations as established practice and
cites no originator. A search of the secondary literature returned one provenance claim — that a
"McKinsey" variant traces to Fetter and Dalleck, *Decision Models for Inventory Management* (1961)
— but that claim comes from a vendor blog and was not verified. **Do not attribute the formula to
a named inventor.** The honest statement is that the square-root-of-protection-period form is
standard in inventory texts by the 1960s and its first publication was not established by the
sources consulted here.

The absence of a canonical origin has a practical consequence. Because no single source is
authoritative, implementations differ, and each difference is defended by reference to a different
textbook. Section 9.2 catalogues the differences that matter.

### 9.2 Why implementations disagree about which variance they use

Three choices are made independently in every implementation, and the combinations are not
equivalent.

| Choice | Option A | Option B | Consequence of the difference |
|---|---|---|---|
| Protection period | `L` only | `R + L` | Up to 55% buffer difference at monthly review with short lead time |
| Sigma source | Demand history | Forecast error | Over-buffers when the forecast carries information; under-buffers when it does not |
| Lead-time treatment | Deterministic | Random, second term included | Up to a factor of two on high-volume items |
| Service measure | CSL | Fill rate | Up to 60% buffer difference at the same nominal target |
| Object stored | Quantity | Days of supply | Buffer silently changes when the forecast changes |

A vendor system that uses `R + L`, demand history, a random lead time, and CSL is not "more
correct" than one that uses `L`, forecast error, a deterministic lead time, and fill rate. The two
answer different questions. They are only comparable if both are specified, and the specification
is exactly the reporting card in step 20.

The one implementation difference that can be documented from public sources is the separation of
safety stock and safety lead time as distinct fields. Microsoft's Dynamics 365 Business Central
documentation describes a safety lead time field on the item and stock-keeping unit (SKU) planning
tabs and advises using safety stock when demand quantity is uncertain and safety lead time when
supplier delivery time is uncertain. That is a statement about the product's data model, which is
checkable, and not a statement about outcomes. Grade D: vendor documentation, factual about the
product.

### 9.3 What the research literature establishes

| Study or source | Design | Finding | Grade |
|---|---|---|---|
| Eppen and Martin (1988), *Management Science* 34(11), 1380 | Analytical and numerical | The standard procedure for stochastic lead time and demand can produce badly wrong safety stocks; a correct procedure is derived | A |
| Chopra, Reinhardt and Dada (2004) | Analytical | Lead-time uncertainty affects safety stock non-linearly; the effect depends on whether the uncertainty is in the mean or the variance | A |
| Humair, Ruark, Tomlin and Willems (2013), *Interfaces* | Multi-echelon guaranteed-service model | Extends safety stock placement to random lead times in a network, with closed-form expected safety stock | A |
| Babai, Dai, Li, Syntetos and Wang (2021), *EJOR* | ARMA(1,1) demand with discrete stochastic lead times | The textbook variance formula is often the least accurate of three strategies under positive autocorrelation | A |
| Prak, Teunter and Syntetos (2017), *EJOR* 256(2), 454–461 | Analytical, with simulation | Forecast-error correlation understates safety stock by up to 30%, with service up to 10 points below target | A |
| Saoud, Kourentzes and Boylan (2022), *Omega* 110, 102614 | Empirical comparison of approximations | Recommends monitoring the empirical variance of lead-time errors directly rather than scaling a one-period sigma | A |
| Van Donselaar and Broekmeulen (2011), *IJPE* | Periodic review, lost sales, lot sizing | Treating a lost-sales system as a backorder system overestimates required safety stock; average error 3.8% above a 95% fill target | A |
| Bell (2000), *JORS* 51(3), 358–363 | Estimation under stock-outs | The standard stock-out adjustment for demand variance fails once the stock-out share exceeds about 50% | A |
| Wecker (1978), *Management Science* 24(10), 1043–1054 | Analytical | Quantifies the bias in demand and prediction-error variance caused by stock-outs | A |
| van Kampen, van Donk and van der Zee (2010), *IJPR* | Simulation of an industrial case | Safety lead time performs better under supply variability; safety stock performs better under demand-information unreliability | A |
| Buzacott and Shanthikumar (1994), *Management Science* 40(12), 1678 | Analytical, MRP setting | Safety time is preferable only when future requirements over the lead time can be forecast accurately | A |
| King (2011), *APICS Magazine* | Practitioner exposition | Presents the four formula variants and the `z` table; no historical attribution | C |

Three of these rows deserve a sentence each, because they change what a practitioner should do.

**Prak, Teunter and Syntetos (2017) is the most actionable.** The result is not that forecasts are
bad. It is that even with independent demand, estimating the mean from a finite window makes the
errors over a lead time positively correlated, because the same estimation error is present in
every period. The correction is a factor on the buffer of `sqrt(1 + L/M)` for a moving average over
`M` periods, which is small enough to be dismissed and large enough to cost several points of
service. The paper's simulation reports service levels up to ten percentage points below target
when the correction is omitted.

**Van Donselaar and Broekmeulen (2011) matters because most retail systems are lost-sales
systems.** When a stock-out occurs, the unmet demand is not deferred; it goes to a competitor.
A formula derived for backorders assumes that demand is eventually served, which makes the
effective protection period longer than it is. The error is modest at high fill targets and grows
as the target falls, which is the opposite of the usual intuition that approximations are worst in
the tail.

**Bell (2000) is the practical limit on correction.** Analysts who know about censoring often apply
a stock-out adjustment, and the adjustment has a domain. Above roughly half the periods censored,
the correction itself becomes unreliable and the honest response is to say so rather than to
report a corrected sigma.

## 10. Failure modes and red flags

| Failure | Red flag visible in a meeting or dataset | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Protection period set to lead time | Nobody can say what review period the buffer assumes | Periodic review buffered as if continuous | 13–55% understatement depending on the ratio | Publish `R`, `L`, and `P` on the item card |
| Lead-time variance term omitted | Buffer equals `z * sigma_d * sqrt(L)` exactly | Only the demand term implemented | Up to a factor of two on high-volume items | Recompute one item by hand; compare the two terms |
| Sigma computed from shipments | Sigma falls after a service improvement, which is impossible | Shipments are smoother than demand | Buffer shrinks as service improves | Use consumption or demand; state the construct |
| Stock-out periods left in the sigma window | Low sigma on the worst-performing items | Sales are censored at available stock | Under-buffer exactly where service is already failing | Report the censored share; exclude or correct |
| Promotion weeks inside the window | Sigma jumps after every promotion quarter | A temporary intervention enters a permanent parameter | Over-buffer for a year, then under-buffer | Segment or calendar-adjust; state the rule |
| CSL multiplier applied to a fill target | One `z` table across the catalogue | The two measures need different multipliers | 20–60% overstatement | Convert through the loss function; state `Q` |
| Buffer quoted without a service measure | "Our safety stock is 194" with no target named | The number is a quantile of an unnamed distribution | Unfalsifiable | Reporting card, line 4 |
| Order-up-to level reported as safety stock | Inventory looks 40% higher than the buffer | Cycle and safety stock summed | Wrong comparison across items | Separate expected demand from buffer |
| Stored quantity never revised | The buffer is the same as three years ago | Quantity is static while demand moved | Silent drift in either direction | Recomputed level or scheduled review |
| Sigma from 13 weeks reported to the unit | A buffer of 194 from a quarter of data | Sampling error dominates | ±40% interval presented as a point | Report the interval; lengthen the window |
| Autocorrelation ignored | Sigma from a short moving average; buffer misses target by 3–6 points | Estimation error is common across the lead time | 7–30% understatement | Apply `sqrt(1 + L/M)`; report `M` |
| Formula applied when lead time exceeds the season | A buffer on a one-shot seasonal buy | No replenishment can arrive within the selling window | The buffer protects a period that does not exist | Use the newsvendor quantile; see topic 03 |
| Normal quantile used on an intermittent item | A negative lower bound in the working | The normal has support on the whole line | Materially wrong tail probability | Use the exact distribution; see Module 08 topic 06 |
| Lead-time sigma expressed as a percentage | Sigma changes when the mean lead time changes | Relative and absolute variance conflated | Buffer moves for the wrong reason | Store both; state which the formula uses |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks if it fails | How to test it |
|---|---|---|---|
| Demand is serially independent | Makes `Var(D_P) = P * sigma_d^2` | Positive correlation inflates the true variance; the buffer is short | Autocorrelation of residuals at lags 1–3; Ljung–Box |
| Lead time is independent of demand | Allows the law of total variance with no covariance term | A supplier that slows when demand rises leaves a missing positive covariance term | Correlation between demand intensity and realised lead time |
| Lead time is independent across cycles | Justifies treating each cycle with the same distribution | Persistent lateness makes the buffer a one-off hedge rather than a recurring one | Sequential correlation of successive lead times |
| The distribution is approximately normal at the target quantile | Justifies the `z` multiplier | The tail is wrong; buffers miss at 99% and on skewed items | Compare the empirical quantile with the normal one at the target |
| Parameters are known rather than estimated | Justifies using `z` instead of a Student-t quantile | Estimation error is ignored; buffer is short by `sqrt(1 + L/M)` and more | Test the correction; compare with a t-based quantile |
| The demand process is stationary over the window | The historical sigma is the future sigma | A listing change, competitor exit, or redesign moves the parameter | Split the window; test for a break |
| The service measure is the one the contract names | The multiplier depends on it | A CSL buffer against a fill target over-buffers; the reverse under-buffers | Read the SLA; compute both multipliers |
| The review discipline matches the system configuration | Determines the protection period | Continuous-review formula applied to periodic review | Observe actual order release dates for a month |
| Demand is observed, not censored | Sigma must reflect the market, not the stock policy | Censoring biases sigma down; the buffer reproduces the stock-out | Availability flags; censored share |
| At least one replenishment can arrive within the demand horizon | Gives the protection period meaning | Seasonal or one-shot items have no recurring cycle to protect | Compare `R + L` with the selling season length |
| The item is buffered individually, not as part of a pool | Justifies computing per item | Pooling or substitution changes the effective variance | Check for substitutes and shared components |

Three limits deserve statement. **The formula is a quantile of a model,** and it says only that if
demand is independent, if lead time is independent, and if the parameters are stable, then this
quantity achieves this service level; the reporting card exists to keep those clauses visible.
**The error is dominated by the input,** not by the formula: section 6.8 ranks the sources, and a
sigma estimated from a quarter of data costs more than any formula variant. **Short-lifecycle items
are outside the model:** when `R + L` exceeds the selling season there is no recurring cycle to
protect, and the correct object is the newsvendor quantile of total season demand.

## 12. Exercises

### Exercise 1 — Two terms or one (calculation, 35 minutes)

Three items, all weekly buckets, all with a one-week review period and a 95 per cent CSL target.

| Item | `mu_d` (units/wk) | `sigma_d` (units/wk) | `L` (wk) | `sigma_L` (wk) |
|---|---:|---:|---:|---:|
| A | 80 | 12 | 2.0 | 0.0 |
| B | 80 | 12 | 2.0 | 0.6 |
| C | 400 | 25 | 2.0 | 0.3 |

**Deliverable:** for each item, the protection period, both variance terms, `sigma_P`, the buffer,
and the share of variance from lead time.
**Acceptance criteria:** units stated at every step; the share computed as a percentage; one
sentence naming which item the one-term formula damages most and why.
**Time estimate:** 35 minutes.

### Exercise 2 — Audit a protection period (application, 40 minutes)

For a system you can inspect, or from a described configuration: find the safety stock field, the
lead time field, the review interval, and any safety lead time field. Determine what protection
period the stored buffer implies.

**Deliverable:** a one-page audit with the four field values, the implied protection period, the
protection period implied by the operating calendar, and the gap.
**Acceptance criteria:** the gap is expressed in weeks and as a buffer percentage; at least one
recommendation is specific to a field, not to a process; a note states whether the buffer is
stored as a quantity or derived.
**Time estimate:** 40 minutes.

### Exercise 3 — Convert the service measure (calculation, 40 minutes)

Using `sigma_P = 117.64` units and `mu_d = 120` units per week, compute the buffer for: 95 per cent
CSL; 95 per cent fill rate with `Q = 120`; 95 per cent fill rate with `Q = 480`; 99 per cent fill
rate with `Q = 120`.

**Deliverable:** a four-row table with `z`, the buffer, and the ratio to the CSL buffer.
**Acceptance criteria:** `L(z)` shown for each fill case; the inversion method stated (table or
iteration); one sentence explaining why the `Q = 480` case differs from the `Q = 120` case.
**Time estimate:** 40 minutes.

### Exercise 4 — The sigma window (application, 50 minutes)

Take a real or constructed weekly series with at least 104 weeks. Compute the standard deviation
on windows of 8, 13, 26, 52, and 104 weeks ending at the same date.

**Deliverable:** the five estimates, the approximate standard error of each, and the implied
range of buffers.
**Acceptance criteria:** the standard error formula stated; the estimates plotted or tabulated
against window length; a recommendation with a stated criterion (sampling error against
stationarity), not a preference.
**Time estimate:** 50 minutes.

### Exercise 5 — Diagnose censoring (critique, 40 minutes)

A catalogue shows a mean weekly demand of 118 units and a standard deviation of 26 units, computed
across 52 weeks. Service logs show 9 weeks in which the item was unavailable for at least part of
the week. Recomputing on the 43 available weeks gives a mean of 126 and a standard deviation of 31.

**Deliverable:** the buffers implied by both estimates at a 95 per cent CSL, the percentage
difference, and a recommendation.
**Acceptance criteria:** the direction of the bias explained mechanically; the censored share
computed; the recommendation states the risk of the correction as well as the benefit, with
reference to the regime in which corrections become unreliable.
**Time estimate:** 40 minutes.

### Exercise 6 — Correct for autocorrelation (calculation, 45 minutes)

A buffer of 194 units was computed from a standard deviation estimated by a simple moving average
over 26 weeks, with a four-week lead time and a one-week review period.

**Deliverable:** the correction factor for `M = 8, 12, 26, 52`; the corrected buffer for each;
the implied error correlation `1/(M+1)` for each.
**Acceptance criteria:** the factor applied to the demand term only, and the reason stated; the
corrected buffers recomputed through the variance, not multiplied onto the final answer; one
sentence on when the correction is immaterial.
**Time estimate:** 45 minutes.

### Exercise 7 — Safety stock or safety lead time (design, 45 minutes)

A supplier offers to accept orders one week earlier. The buyer currently holds safety stock sized
for a lead time with mean 3.0 weeks and standard deviation 0.8 weeks. Demand is 120 units per week.
Write a one-page recommendation.

**Deliverable:** the inventory cost of each option, the coverage each provides in standard
deviations of lead time, and a recommendation with a condition under which it reverses.
**Acceptance criteria:** both costs in units, not just percentages; the protection-period effect of
the safety lead time stated; the service measure caveat stated; the reversal condition specific.
**Time estimate:** 45 minutes.

### Exercise 8 — Diagnose the item outside the model (critique, 35 minutes)

A fashion accessory has a 10-week selling season, a 14-week lead time, a one-week review period,
and a safety stock of 240 units computed with the standard formula at a 95 per cent CSL.

**Deliverable:** a diagnosis in no more than 300 words, naming the object that should replace the
buffer and the two inputs it needs.
**Acceptance criteria:** the arithmetic shown for why the protection period exceeds the horizon;
the replacement object named; no claim that the existing number is arithmetically wrong.
**Time estimate:** 35 minutes.

### Exercise 9 — Write the reporting card (design, 40 minutes)

Write the seven-line card from step 20 for an item you know, with every line filled.

**Deliverable:** the completed card plus a note on which line was hardest to fill and what that
difficulty reveals about the organisation.
**Acceptance criteria:** all seven lines present; every number carries a unit; line 5 states the
window, the construct, and at least one exclusion; line 7 names at least two assumptions the
formula makes and the direction of their error.
**Time estimate:** 40 minutes.

## 13. Self-check answers

### Exercise 1

```text
A: P = 3.0   Var = 3 * 144 = 432 + 0        = 432     sigma_P = 20.78   SS = 34.2  -> 34
B: P = 3.0   Var = 432 + 6400 * 0.36 = 432 + 2,304  = 2,736   sigma_P = 52.31   SS = 86.0
C: P = 3.0   Var = 1,875 + 160000 * 0.09 = 1,875 + 14,400 = 16,275  sigma_P = 127.57  SS = 209.9
```

Lead-time shares: A 0 per cent, B `2,304/2,736 = 84.2` per cent, C `14,400/16,275 = 88.5` per
cent. C is damaged most in absolute terms despite a smaller lead-time coefficient of variation
than B, because the lead-time term scales with `mu_d^2` and C carries five times the volume. The
screening ratio makes the point in one line: C is `120/43.3 = 2.77`, B is `48/20.8 = 2.31`, A is
zero. Volume, not supplier unreliability, drives the omission.

### Exercise 2

A strong audit ends with three numbers: the implied protection period, the operating protection
period, and the buffer difference. A typical finding is that the stored buffer implies a 3-week
protection period while the operating calendar implies 4 weeks, which is a 13 per cent
understatement. A weak audit lists field names and stops, which does not establish what the system
computes.

### Exercise 3

```text
CSL 95%           z = 1.645   SS = 193.5  -> 194   ratio 1.00
Fill 95%, Q = 120 L(z) = 0.0510   z = 1.25   SS = 147.1 -> 147   ratio 0.76
Fill 95%, Q = 480 L(z) = 0.2040   z = 0.50   SS = 58.8  -> 59    ratio 0.30
Fill 99%, Q = 120 L(z) = 0.0102   z = 1.82   SS = 214.1 -> 214   ratio 1.11
```

The `Q = 480` case differs because the fill rate is a ratio of expected units short to expected
demand per cycle, and a larger cycle has more demand over which to spread the same expected
shortfall. The buffer falls as the order quantity rises, which is why a fill-rate target is
incompatible with a fixed multiplier across a catalogue.

### Exercise 4

For a normal sample, `se(s) ~ s / sqrt(2(n-1))`: 8 weeks gives 26.7 per cent, 13 gives 20.4, 26
gives 14.1, 52 gives 9.9, 104 gives 7.0. The recommendation should be a window that keeps the
sampling error acceptable while spanning no structural break. A defensible default for weekly
data is 52 weeks with a documented break test; a defensible answer for a fast-moving item is
26 weeks with a monthly re-estimation. An answer that names a window without the standard error
has not engaged with the question.

### Exercise 5

```text
all 52 weeks:      sigma = 26   SS = 1.645 * 2 * 26  = 85.5  -> 86
available weeks:   sigma = 31   SS = 1.645 * 2 * 31  = 102.0 -> 102
difference:        19.3 per cent
```

The bias is downward because observed sales are `min(demand, stock)`, which truncates the upper
tail and removes exactly the weeks that carry the variance. The censored share is 9 of 52, or 17
per cent, which is well below the regime in which correction procedures become unreliable — so
here the correction is worth making. The risk of correcting is that the available-week series may
also be unrepresentative in the other direction, if availability correlates with season.

### Exercise 6

```text
M = 52:  factor sqrt(1 + 4/52) = 1.038   rho = 1/53  = 0.019
M = 26:  factor sqrt(1 + 4/26) = 1.074   rho = 1/27  = 0.037
M = 12:  factor sqrt(1 + 4/12) = 1.155   rho = 1/13  = 0.077
M =  8:  factor sqrt(1 + 4/8)  = 1.225   rho = 1/9   = 0.111

M = 26: sigma_d = 36.52; Var = 4 * 1,334 = 5,336; total = 14,552; sigma_P = 120.6; SS = 198
```

The factor applies to the demand term because it corrects the variance of forecast errors over
the protection period, and the lead-time term is a separate source. Multiplying 194 by 1.074 would
give 208, which overstates. The correction is immaterial when the lead-time term dominates, which
is the case here at 2 per cent, and material when the demand term dominates, where it approaches
the full 7.4 per cent.

### Exercise 7

The safety lead time costs 120 units of pipeline stock. The lead-time term costs
`1.645 * 120 * 0.8 = 158` units. Per standard deviation of lead time covered, both cost 96 units:
`120 / 1.25` and `158 / 1.645`. They are the same insurance at the same price. The safety lead time
covers lateness to the 89th percentile of the lead-time distribution and nothing else; the safety
stock covers the joint distribution to the 95th percentile. The safety lead time also extends the
protection period to five weeks, adding roughly 13 units to the buffer, so the total is 133 against
158. The recommendation should reverse if the supplier is punctual but erratic in quantity, in
which case only safety stock helps.

### Exercise 8

The protection period is `1 + 14 = 15` weeks against a 10-week season. No order placed after the
season opens can arrive before it closes, so there is no recurring replenishment cycle and no
period for a buffer to protect. The number is arithmetically valid and decision-theoretically
meaningless. The replacement is a single-period order quantity from the critical fractile, which
needs the underage and overage costs (or the critical ratio) and a distribution of total season
demand. That object is treated in Module 10 topic 03.

### Exercise 9

A strong card has seven filled lines and no unspecified numbers. The hardest line is usually 5,
the sigma source, because it requires three decisions that most organisations have never recorded:
the window, the construct, and the exclusions. A submission that writes "sigma from history" has
not filled the line. One that writes "weekly demand, 52 weeks to 2026-09-30, promotion weeks
excluded by calendar, 9 stock-out weeks excluded (17%), s = 34 kits/week" has. The second hardest
is line 7, because naming the assumptions requires knowing which ones the implementation actually
made.

## 14. Interview questions

**1. "What is the protection period, and why is it not the lead time?"**

A strong answer defines `P = R + L`, draws the timeline, and says the order must cover demand
until the next order arrives, which is one review period after the current one. A weak answer
says "lead time plus some buffer" without being able to say what the buffer is.

**2. "Derive the safety stock formula."**

A strong answer writes demand over the protection period as a sum, takes the variance under
independence, applies the law of total variance for a random lead time, and states where each
assumption entered. A weak answer recites `z * sigma * sqrt(L)` with no derivation and no
statement of what `L` means.

**3. "What changes when the lead time is random?"**

A strong answer gives the second term `mu_d^2 * sigma_L^2`, explains that it scales with volume
squared, and says the term dominates for high-volume items with moderately unreliable suppliers.
A weak answer says "you add lead-time variability" without a formula or a magnitude.

**4. "Which term do implementations omit, and how large is the error?"**

A strong answer names the lead-time variance term and gives a range: a few per cent when the
lead-time coefficient of variation is small relative to demand variability, more than 40 per cent
when the screening ratio exceeds one. A weak answer says implementations vary, which is true and
useless.

**5. "CSL or fill rate — does it matter?"**

A strong answer gives both definitions, converts the fill target through the loss function, notes
that the fill multiplier depends on the order quantity, and gives a numerical gap. A weak answer
says they are "different service metrics" and picks one.

**6. "How long a history do you use for sigma?"**

A strong answer gives the sampling-error arithmetic `s/sqrt(2(n-1))`, states the trade-off against
stationarity, and names a window with a criterion rather than a preference. A weak answer says
"a year" because that is what the system does.

**7. "What does a stock-out do to your sigma estimate?"**

A strong answer says sales are censored at available stock, so the observed standard deviation is
biased down, and the bias is worst exactly where service is already poor. It mentions the regime
in which corrections become unreliable. A weak answer treats stock-outs as missing data to be
cleaned.

**8. "Why does forecast-error autocorrelation matter if demand is independent?"**

A strong answer explains that the estimation error in the mean is common to every period of the
lead time, so the errors are correlated even under independent demand, and gives the
`sqrt(1 + L/M)` correction and a magnitude. A weak answer says autocorrelation is a property of
time series and moves on.

**9. "Safety stock or safety lead time?"**

A strong answer says safety lead time protects timing and safety stock protects quantity, notes
that under a normal model they cost the same per unit of lead-time coverage, and chooses based on
which risk is real. A weak answer says safety lead time is cheaper, which is true only for the
same coverage and is not usually the comparison being made.

**10. "When is the formula meaningless?"**

A strong answer names the case where the lead time exceeds the selling season, explains that there
is no recurring cycle to protect, and names the newsvendor quantile as the replacement.
A weak answer says the formula is always approximately right.

## 15. Key terms

- **Safety stock** — buffer above expected demand over the protection period.
- **Protection period** — review period plus lead time; the span the buffer covers.
- **Review period** — interval between replenishment decisions.
- **Lead time** — elapsed time from order release to availability.
- **Lead-time demand** — demand accumulated over the protection period.
- **Order-up-to level** — target inventory position after ordering; buffer plus expected demand.
- **Reorder point** — position at which a replenishment order is triggered.
- **Cycle service level (CSL)** — probability of no stock-out in a replenishment cycle.
- **Fill rate** — expected fraction of demand units served from stock.
- **Loss function** — expected units short per cycle at a standardised buffer, `L(z)`.
- **Service factor** — standardised quantile implied by the service measure and target.
- **Sigma of demand** — standard deviation of demand per period in the model's bucket.
- **Sigma of lead time** — standard deviation of realised lead time, in periods.
- **Demand construct** — the definition of the series: consumption, orders, or shipments.
- **Censoring** — observed demand truncated at available stock.
- **Safety lead time** — planned time added so material arrives before it is needed.
- **Pipeline stock** — goods owned but not yet available.
- **Error autocorrelation** — correlation of forecast errors across periods within a lead time.
- **Estimation window** — the historical span used to estimate parameters.
- **Standard error** — standard deviation of an estimator across hypothetical samples.
- **Law of total variance** — `Var(X) = E[Var(X|Y)] + Var(E[X|Y])`, the step that adds lead-time variability.
- **Newsvendor quantile** — the single-period order quantile from the critical fractile.

## 16. Sources

**Peer-reviewed and academic (A)**

- Eppen, G. D., and Martin, R. K. (1988). "Determining Safety Stock in the Presence of Stochastic
  Lead Time and Demand." *Management Science*, 34(11), 1380. (A)
- Wecker, W. E. (1978). "Predicting Demand from Sales Data in the Presence of Stockouts."
  *Management Science*, 24(10), 1043–1054. (A)
- Nahmias, S. (1994). "Demand estimation in lost sales inventory systems." *Naval Research
  Logistics*, 41(6), 739–757. (A)
- Buzacott, J. A., and Shanthikumar, J. G. (1994). "Safety Stock versus Safety Time in MRP
  Controlled Production Systems." *Management Science*, 40(12), 1678. (A)
- Bell, P. C. (2000). "Forecasting demand variation when there are stockouts." *Journal of the
  Operational Research Society*, 51(3), 358–363. DOI: 10.1057/palgrave.jors.2600877. (A)
- Chopra, S., Reinhardt, G., and Dada, M. (2004). "The Effect of Lead Time Uncertainty on Safety
  Stocks." Journal and volume not returned by the searches; cited by title only. (A)
- van Kampen, T. J., van Donk, D. P., and van der Zee, D. J. (2010). "Safety stock or safety lead
  time: coping with unreliability in demand and supply." *International Journal of Production
  Research*. DOI: 10.1080/00207540903348346. (A)
- Van Donselaar, K. H., and Broekmeulen, R. A. C. M. (2011). "Determination of safety stocks in a
  lost sales inventory system with periodic review, positive lead-time, lot-sizing and a target
  fill rate." *International Journal of Production Economics*.
  DOI: 10.1016/j.ijpe.2011.05.020. (A)
- Humair, S., Ruark, J., Tomlin, B., and Willems, S. P. (2013). "Incorporating Stochastic Lead
  Times Into the Guaranteed Service Model of Safety Stock Optimization." *Interfaces*.
  DOI: 10.1287/inte.2013.0699. (A)
- Prak, D., Teunter, R., and Syntetos, A. (2017). "On the calculation of safety stocks when demand
  is forecasted." *European Journal of Operational Research*, 256(2), 454–461.
  DOI: 10.1016/j.ejor.2016.06.035. (A)
- Babai, M. Z., Dai, Y., Li, H., Syntetos, A., and Wang, X. (2021). "Forecasting of lead-time
  demand variance: Implications for safety stock calculations." *European Journal of Operational
  Research*. DOI: 10.1016/j.ejor.2021.04.017. (A)
- Saoud, P., Kourentzes, N., and Boylan, J. E. (2022). "Approximations for the Lead Time
  Variance." *Omega*, 110, 102614. (A)
- Mersereau, A. J. (2015). "Demand Estimation from Censored Observations with Inventory Record
  Inaccuracy." *Manufacturing & Service Operations Management*, 17(3).
  DOI: 10.1287/msom.2015.0520. (A)
- Ban, G.-Y. (2020). "Confidence Intervals for Data-Driven Inventory Policies with Demand
  Censoring." *Operations Research*, 68(2). DOI: 10.1287/opre.2019.1883. (A)

**Standards and official (B)**

- GS1 General Specifications, current edition, including GTIN, GLN, and EPCIS 2.0. (B)
- ISO 8000 series, *Data quality*. (B)
- ISO 3534-1, *Statistics — Vocabulary and symbols — Probability and general statistical terms*. (B)
- ISO 22400-2, *Automation systems and integration — Key performance indicators for manufacturing
  operations*. (B)
- IAS 2, *Inventories*; ASC 330, *Inventory*. (B)

**Professional, teaching and practitioner (C)**

- King, P. L. (2011). "Crack the Code: Understanding Safety Stock and Mastering Its Equations."
  *APICS Magazine*, July/August 2011. PDF at the MIT 2.810 course readings. (C)
- King, P. L. (2011). "Cycle Service Level and Fill Rate." APICS companion paper. (C)
- Bricker, D. *Reorder Point* lecture notes, University of Iowa, Department of Mechanical and
  Industrial Engineering. Derivation of `E[T]sigma_d^2 + mu_d^2 sigma_t^2`. (C)
- Loucka, L. (2007). "Safety Stock Calculation." Resource Systems Consulting blog; a roundup of
  competing formulas. Useful as a catalogue of variants, not as evidence. (E)

**Vendor documentation and marketing (D)**

- Microsoft, Dynamics 365 Business Central documentation: safety lead time on the item card and
  stock-keeping unit planning tabs. (D)
- Anaplan, "Standard Safety Stock Simulation (DC-Product)" application documentation. Defines the
  protection period as lead time plus review period. (D)
- MetricGate, "Periodic Review (R, S) Inventory Policy" and "Service Level" calculator
  documentation. (D)
- Umbrex, "Safety Stock Policies and Calculation Methods." Includes a fill-rate relation that does
  not match the loss-function formulation used here; treat as a lead, not a source. (D)
- Netstock, "Answers to the Most Common Safety Stock Questions." (D)

**Note on grading and on unverified items.** Grades follow the programme scheme: A peer-reviewed or
audited, B regulatory or official, C professional article, conference paper, or teaching material,
D vendor documentation or marketing, E blog or anecdote. Three items need flags. The Chopra,
Reinhardt and Dada reference was returned by the searches as a title with a year and no journal or
volume; it is cited by title only and should be completed from a library record before quotation.
The Fetter and Dalleck (1961) provenance for a "McKinsey" formula variant came from a vendor blog
and is not claimed above. No source consulted quantifies omission rates in commercial systems;
section 9 marks those claims as inference.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topics in this module: [`01_cost-service-frontier.md`](01_cost-service-frontier.md) —
  the trade-off the service target sits on; [`02_eoq-limits.md`](02_eoq-limits.md) — the order
  quantity that enters the fill-rate multiplier.
- Later in this module, not yet on disk: `03_newsvendor.md` — the single-period quantile that
  replaces the buffer when the lead time exceeds the season.
- Later in this module, not yet on disk: `05_service-metrics.md`, `06_policy-families.md`, and
  `07_lead-time-demand.md` — measurement definitions, policy families, and the distribution this
  buffer is a quantile of.
- Later in this module, not yet on disk: `08_multi-echelon.md`, `09_pooling-postponement.md`, and
  `10_segmentation-shelf-life.md`.
- Distribution prerequisite: [`../08_hierarchical_intermittent_probabilistic_forecasting/06_compound-poisson-hurdle.md`](../08_hierarchical_intermittent_probabilistic_forecasting/06_compound-poisson-hurdle.md)
  — lead-time demand for intermittent items, when the normal multiplier fails.
- Quantile decision: [`../08_hierarchical_intermittent_probabilistic_forecasting/10_quantiles-decisions.md`](../08_hierarchical_intermittent_probabilistic_forecasting/10_quantiles-decisions.md)
  — choosing the quantile from costs rather than from a target.
- Censored demand: [`../09_demand_sensing_promotions_npi/09_distribution-assortment.md`](../09_demand_sensing_promotions_npi/09_distribution-assortment.md)
  — distribution and censoring in the demand signal.
- Part II map: [`../_index.md`](../_index.md)
- Part I map: [`../../01_foundations/_index.md`](../../01_foundations/_index.md)
- Survey file: [`../../../course/10_inventory_optimization_multi_echelon.md`](../../../course/10_inventory_optimization_multi_echelon.md)

