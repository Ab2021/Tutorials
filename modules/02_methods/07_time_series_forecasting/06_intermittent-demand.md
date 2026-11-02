# Intermittent Demand — Croston, SBA, TSB, and classification before method choice

> Module 07 — Time Series Forecasting Foundations
> Topic 06 of 10
> Estimated study time: 3.5 hours
> Prerequisites: 01A.10, 01A.11, 07.01–07.05
> Version 2026.5

---

## 1. Why this topic exists

Open the item master of a distributor, a hospital supply chain, or a service-parts catalogue and sort by annual units.
The median item moves a handful of times a year; a large share of item-location combinations moved nothing at all last
quarter. The methods taught in the first five topics of this module were built for the other end of that list.

Single exponential smoothing (SES), Holt's method, Holt–Winters, and ARIMA all carry a **level**: a running estimate of
"what demand is currently like per period". On a series that is zero four weeks in five, the level sits near zero, and
the variance of the series is not driven by the level at all. It is driven by whether an order arrived. A method that
tracks the level is tracking the wrong object.

The operational consequences run in both directions at once.

1. **Dead stock accumulates.** Croston's 1972 paper makes the point quantitatively: for demand of
   constant size at constant intervals, conventional exponential smoothing with typical parameters
   produces stock levels up to double what the demand requires. The excess is invisible in a stock
   report, because the item looks low-volume and nobody audits it.
2. **The items that matter go short.** The same smoothing behaviour that over-stocks a slow mover
   under-stocks the week a real order lands, because the forecast is still catching up from the
   previous zero run.
3. **Method choice becomes a political argument.** Two analysts backtest two methods on the same
   portfolio, get opposite rankings, and neither is wrong. The ranking depends on the demand-class
   mix, the error metric, and the horizon — and those are usually unstated.

Three methods dominate practice, and they form a sequence of increasingly correct diagnoses of one problem:

| Method | Year | Core move | What it fixes | What it still gets wrong |
|---|---|---|---|---|
| Croston | 1972 | Smooth intervals and sizes separately; forecast the ratio | Stops zeros dragging the level down | The ratio is positively biased |
| SBA | 2001 | Multiply Croston by `(1 − α/2)` | Removes most of the bias | Frozen during zero runs; blind to obsolescence |
| TSB | 2011 | Smooth the *probability* of demand every period | Decays when an item stops being ordered | A second constant to tune; slower on stationary demand |

The chain is the valuable part: each method exists because someone measured a specific failure in the previous one.

> **Note:** The measurement side — average inter-demand interval (ADI), squared coefficient of
> variation (CV²), the four-class grid, and why percentage errors fail on zero-heavy series — is in
> [Module 01A topic 10](../../01_foundations/01A_foundations_units_time_assumptions/10_intermittent-metrics.md).
> This file assumes those definitions and spends its length on the methods and their failure modes.

## 2. Learning outcomes

By the end of this topic you can:

1. **Explain**, without a formula, why a level-based method fails on a series whose zeros outnumber
   its spikes, and **demonstrate** the failure on a ten-period series you construct.
2. **Decompose** a history into an occurrence component and a size component, **state** the interval
   convention you used, and **explain** why the two need separate smoothing.
3. **Hand-run** Croston's recursion for a short series with a stated initialisation, and **state**
   what the result is and is not.
4. **Derive** the direction of Croston's bias from the ratio structure, and **explain** why the
   correction is a function of the smoothing constant rather than a fixed number.
5. **Apply** the Syntetos–Boylan approximation (SBA) and **state** when it over-corrects.
6. **Contrast** SBA and the Teunter–Syntetos–Babai (TSB) method on an item that stops being
   ordered, and **quantify** the inventory consequence of the stale rate.
7. **Classify** a portfolio with ADI and CV², **route** each class to a shortlist, and **critique**
   the two-threshold rule as a heuristic rather than a law.
8. **Design** a method bake-off evaluated on the decision rather than on a percentage error.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold three pictures: the shape of the series, its decomposition, and the order of the decisions — which is not the order
in which methods are usually compared.

### 3.1 The shape: zeros with spikes

A twenty-four week history for one spare part, in eaches per week:

```text
 y_t   0  0  6  0  0  0  0  4 10  0  0  0  0  5  0  0  0  0  0  7  0  0  0  0
       .  .  #  .  .  .  .  #  #  .  .  .  .  #  .  .  .  .  .  #  .  .  .  .
 week  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
```

Five of twenty-four weeks contain demand; total demand is 32 eaches. The mean is 1.33 units per week and the standard
deviation is 2.53 — nearly twice the mean. Almost all of that dispersion comes from five observations.

```text
 units
  10 |                          *
   8 |                          |
   6 |  *                       |                   *
   4 |  |              *        |                   |
   2 |  |              |        |                   |
   0 +--+----+----+----+----+---+----+----+----+----+----> week
      3    8    9   10   14  15   20   21   22   23  24
```

A level-based method has one number to describe this series. Whatever it picks, it is wrong in most weeks and wrong in
the other direction in the weeks that matter.

### 3.2 The decomposition: two processes wearing one series

```text
                        observed series  y_t  (units per week)
                                   |
                 +-----------------+------------------+
                 |                                    |
        OCCURRENCE PROCESS                       SIZE PROCESS
        did demand happen in t?                   how much, given that it did?
        o_t in {0, 1}                            z_t > 0  (units per occurrence)
                 |                                    |
        summarise as interval q:                 mean and spread of nonzero sizes
        weeks since the last demand              z = 6, 4, 10, 5, 7
        q = 5, 1, 5, 6                                  |
                 |                                    |
        smooth p  (interval, periods)            smooth z  (units)
        or smooth pi (probability)                     |
                 |                                    |
                 +-----------------+------------------+
                                   |
                     rate  =  z / p        or   pi * z      (units per week)
                                   |
                     x horizon  (lead time + review, weeks)
                                   |
                     expected lead-time demand  (units)
                                   |
                     quantile at the critical ratio  ->  order quantity
```

Two facts follow, and they explain every method in this topic.

**The zeros carry information about occurrence and none about size.** Smoothing the raw series lets
a zero drag the size estimate down even though no size observation occurred. Croston's insight is that a zero is an
observation of the occurrence process — but not of the size process, and it should not move the size estimate at all.

**The ratio is where the bias lives.** You divide one smoothed quantity by another. The expectation
of a ratio is not the ratio of expectations, and the gap depends on how noisy the denominator is. That single fact
explains Croston's bias, the shape of the SBA correction, and why the correction is not a universal constant.

### 3.3 The decision order, which is not the method order

Most comparisons start at the last box. They should start at the first.

```text
   1.  What is the decision?        order, stock, quote, scrap, expedite
   2.  What horizon does it need?   lead time + review period, in weeks
   3.  What service target?         critical ratio from underage / overage cost
   4.  What class is the item?      ADI and CV^2, at the decision bucket
   5.  Which method?                SES / Croston / SBA / TSB / nothing
```

Read this as a claim about leverage. Steps 2 and 3 change the answer by more than step 5 in most portfolios, and section
6 makes it concrete: there, the choice among four methods moves expected lead- time demand by about 2.8 units, while
moving from the mean to the 90th percentile of the same quantity moves it by 8. That ordering is not a law — section 9
cites studies where method choice dominates — but it is the right default, and it is the opposite of how most bake-offs
are run.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Intermittent demand | Demand that occurs in some periods and not others, so the history contains genuine zeros | Applied to any low-volume item, including one with small but continuous demand | Occurrence pattern, not a magnitude |
| Lumpy demand | Intermittent demand whose nonzero sizes are also highly variable | Used as a synonym for intermittent | Class label from ADI and CV² |
| Erratic demand | Demand in most periods with highly variable size | Confused with lumpy, because both show high CV² | Class label from ADI and CV² |
| Smooth demand | Demand in most periods with a stable size | Assumed to be the normal case; it is the minority of most catalogues | Class label from ADI and CV² |
| Zero share | Fraction of periods in the evaluation window with no demand | Never reported, although it decides whether the series is in scope | Proportion of periods |
| Demand occurrence | The binary event that demand exceeds zero in a period | Treated as a modelling detail rather than half the process | 0 or 1 per item-period |
| Demand size | The quantity demanded in a period where demand occurred | Confused with period demand including zeros | Units per occurrence (each, case, kg) |
| Inter-demand interval | Periods between one demand occurrence and the next | Counted inclusively in some tools and exclusively in others | Periods |
| ADI | Average inter-demand interval: periods in the window divided by demand occurrences | Computed over nonzero periods only, which always returns 1.0 | Periods per demand |
| CV² | Squared coefficient of variation of the nonzero demand sizes | Computed over the full series including zeros, which conflates the two processes | Dimensionless |
| Occurrence probability (π) | Probability that demand occurs in a period; roughly the reciprocal of ADI | Confused with the demand rate; it is dimensionless and carries no size | Probability per period |
| Demand rate | Expected demand per period: occurrence probability times mean size | Used interchangeably with the forecast, which it is not | Units per period |
| Croston's method | Separate exponential smoothing of intervals and sizes, combined as a ratio | Applied to series with no zeros, where it degenerates to a worse SES | Units per period |
| Smoothing constant (α) | Weight given to the newest observation in an exponential recursion | Quoted without stating whether it multiplies the new or the old value | Dimensionless, 0 to 1 |
| Interval smoothing constant | Weight on the newest interval in Croston's interval recursion | Confused with the TSB occurrence constant, also often called β | Dimensionless, 0 to 1 |
| Initialisation | Starting values of `z` and `p` before any update | Left to software defaults and never inspected, though it dominates short histories | Units for `z`, periods for `p` |
| SBA | Syntetos–Boylan approximation: Croston multiplied by `(1 − α/2)` | Described as unbiased; it is approximately unbiased and can overshoot negative | Units per period |
| Bias correction factor | The multiplier applied to Croston to remove the leading bias term | Treated as a fixed 0.95; it depends on the smoothing constant actually used | Dimensionless, typically 0.85 to 0.95 |
| TSB | Teunter–Syntetos–Babai method: smooths occurrence probability every period, size only on demand | Described as a Croston variant; the state it carries is different | Units per period |
| Obsolescence | The state in which an item is no longer demanded, so its true rate has fallen to zero | Detected late, usually by a planner rather than by a model | Event, then units per period |
| Dead stock | Inventory of an item with no reasonable prospect of demand | Confused with slow-moving stock, which still has a positive rate | Units and currency |
| Excess stock | Inventory above a stated policy level for an item that still has demand | Reported without the policy that defines "excess" | Units and currency |
| Stale rate | A forecast that keeps asserting demand after the demand has stopped | Invisible in an accuracy report until the item is written off | Units per period |
| Forecast horizon | Periods ahead for which the forecast is required | Assumed to be one period because that is what was backtested | Periods |
| Review period (R) | Interval between replenishment decisions | Confused with lead time | Weeks or days |
| Lead time (L) | Time from placing an order to having it available | Treated as a constant rather than a distribution | Days or weeks |
| Lead-time demand | Demand accumulated over L + R; what an order-up-to level must cover | Replaced by the one-period forecast times L, which ignores the zero structure | Units per replenishment cycle |
| Critical ratio | Underage cost divided by the sum of underage and overage cost; the quantile to target | Set by policy without reference to costs | Proportion, 0 to 1 |
| Cycle service level (CSL) | Probability that a replenishment cycle ends without a stock-out | Reported as fill rate | Proportion |
| Fill rate | Fraction of demand units served from stock | Reported as CSL | Proportion |
| Temporal aggregation | Forecasting at a coarser bucket than the decision bucket, then disaggregating | Assumed to lose nothing; it changes the class of the item | Bucket size in periods |
| ADIDA | Aggregate–disaggregate intermittent demand approach: aggregate to the lead-time-plus-review bucket, forecast, disaggregate | Confused with simple monthly aggregation | Bucket = L + R |
| Censored demand | True demand that is unobservable because stock was insufficient | Ignored, so the model learns from the stock-out it caused | Units |
| Bootstrap forecast | Resampling nonzero sizes and intervals to build an empirical lead-time demand distribution | Assumed more accurate than parametric methods; the evidence is mixed | Units, as a distribution |
| Installed base | Population of machines or devices in service that generates spare-parts demand | Omitted, so demand is modelled without its driver | Count of units in service |

Two rows deserve expansion. **Rate versus quantity:** every method here returns a rate, and the moment a planner reads
it as a quantity, two errors follow — an order of 1.48 units gets rounded by someone who configured a minimum order
quantity years ago, and the mean gets treated as a typical week when the mean of a lumpy process is rarely demanded.
**Probability versus interval:** SBA carries an interval in periods, TSB a probability in [0, 1]. They are reciprocals
in expectation and that is where the similarity ends, because a probability can be updated on a zero period and an
interval cannot. That asymmetry is why TSB exists.

## 5. The mechanics, step by step

The sequence below exists because the failure mode is not misunderstanding a formula. It is skipping the step that
decides whether the formula applies.

### Phase A — Decide whether the series is in scope

**Step 1. Fix the grid, the bucket, and the zero convention.** Write down the item, the location,
and the period that constitute one observation, and confirm the table is dense: every item-location in every period,
with explicit zeros. A sales-line extract has no zeros in it and cannot be used. The bucket must match the replenishment
decision — weekly if orders are placed weekly, even if the forecast runs nightly. Class is a property of the bucket: the
same item can be intermittent weekly and smooth monthly.

**Step 2. Compute ADI and CV², and write the class down.**

```text
ADI = periods in window / periods with demand
CV2 = ( standard deviation of nonzero sizes / mean of nonzero sizes ) ^ 2
```

For the section 3 series: ADI = 24 / 5 = 4.80 periods per demand, and the sizes 6, 4, 10, 5, 7 give a mean of 6.40, a
standard deviation of 2.06, and CV² = 0.104. With the conventional cut-offs of
1.32 and 0.49, the item is **intermittent**.

**Step 3. Ask whether the zeros are demand or absence.** Four facts render as a zero, and only one
belongs in the history:

| What the zero means | Belongs? | What to do |
|---|---|---|
| Ranged, period open, nobody ordered | Yes | Keep it |
| Item not ranged at this location in this period | No | Exclude, and record what you excluded |
| Stock was zero, so demand was lost and never recorded | Yes, flagged | Keep but mark censored; it is an outcome of policy, not of the market |
| Item launched or discontinued mid-window | No | Truncate to the live window |

Censoring does the most damage and is noticed least. If the item was out of stock when a customer wanted it, recorded
demand for that week is zero, and a model trained on that history learns that demand falls when the item is unavailable
— then reproduces the stock-out. There is no statistical fix inside the recursion; you flag availability and model
demand rather than shipments.

### Phase B — Split the series into its two components

**Step 4. Extract the sizes and fix the interval convention.** Walk the series in order; when demand
occurs, record the size and the interval since the previous occurrence. Two conventions circulate:
**exclusive**, counting periods strictly between (week 3 to week 8 gives 4), and **inclusive**,
counting the current period (the same pair gives 5). Both are used in the literature and in software; neither is wrong,
but mixing them is, and the error is invisible in the output because it shifts every rate by a few per cent. State the
convention.

**Step 5. Check whether the components are plausibly separable.** The decomposition assumes the
size of a demand does not depend on how long it has been since the last one. That is often false: emergency orders after
a stock-out are small, and orders after a long gap are large. Plot nonzero size against the preceding interval. A clear
slope means the multiplicative decomposition is mis-specified, and the methods will be systematically wrong in a
direction you can at least name.

**Step 6. Initialise, and state the initialisation.** Croston-type recursions need starting values
for `z` and `p`: commonly the first observed size and the periods up to and including the first demand, or portfolio
priors such as the median size and median ADI of comparable items. The choice barely matters after thirty demand events
and dominates the first ten. If a planner says the slow movers are badly forecast, check the initialisation before the
method.

### Phase C — Run Croston's recursion

**Step 7. Update only when demand occurs. Do nothing on a zero.**

```text
on a period with demand (y_t > 0):
    q   = t - last_demand_period            (interval, periods)
    z   = alpha * y_t + (1 - alpha) * z     (size, units)
    p   = alpha * q   + (1 - alpha) * p     (interval, periods)
    last_demand_period = t

on a period with no demand (y_t = 0):
    z and p are unchanged

forecast for every period until the next demand:
    rate = z / p                            (units per period)
```

Note the convention: α multiplies the newest observation. Some texts reverse it, putting `(1 − α)` on the new
observation. The same numeric α means a very different memory length under the two, so establish the convention before
copying a published parameter into a system.

**Step 8. Read the result as a rate, and stop.** The output is one number per period, constant until
the next demand. It is an estimate of the conditional mean rate, not a quantity to order, and it carries no
distribution.

**Step 9. Notice what the recursion does across a zero run.** Nothing. That is the design, and it is
also the defect. An item demanded every four weeks for two years that has now gone twenty weeks without an order still
shows the same rate. The method cannot express "this item is dying", because the only state it carries is the interval
at the last demand, and a run of zeros has not yet produced an interval to average.

### Phase D — Understand and correct the bias

**Step 10. Diagnose the bias from the ratio structure.** The estimator divides two smoothed random
quantities, and a second-order expansion gives:

```text
E[X / Y]  ~=  ( E[X] / E[Y] ) * ( 1 + Var(Y) / E[Y]^2 )
```

The forecast is inflated by roughly the squared coefficient of variation of the interval estimate. Two predictions
follow, both visible in data. The inflation grows as the interval estimate becomes noisier — that is, as α rises, since
a larger α gives a more responsive and more erratic `p`. And it is largest where intervals are short and few have been
observed, shrinking as the item becomes slower and the interval estimate stabilises.

That is the substance of the Syntetos–Boylan diagnosis, and the reason it was the right one is not the size of any
constant. The bias is a function of the item's demand frequency and of α, so no fixed percentage correction could work
across a portfolio. The correction had to be expressed in terms of the only tunable in the recursion.

> **Note:** You will see this described in shorthand as a bias of order `1/(2p)` — the mean interval
> in the denominator, doubled. Read that as a statement about the shape of the error, not a formula
> to apply to a specific item. Published approximations differ in their constants and in their
> parameterisation.

**Step 11. Apply the SBA correction.**

```text
rate_SBA = ( 1 - alpha / 2 ) * ( z / p )
```

With α = 0.2 the multiplier is 0.9; with α = 0.1 it is 0.95. That is where the frequently quoted "0.95 correction" comes
from: it is not a universal constant, it is `(1 − α/2)` at one system's default smoothing constant. Change α and the
correction changes with it.

The correction removes the leading term and slightly overshoots. Teunter and Sani showed SBA can carry a negative bias
that grows with the mean interval, and proposed a refinement reducing average absolute bias to around one per cent
against roughly five to six per cent for SBA and Croston in their simulation design. A later "exact" correction (Shale,
Boylan and Johnston) was found empirically to differ from SBA only in the fourth or fifth significant digit — a useful
calibration of how much precision this argument is worth.

The practical reading: SBA is a better default than Croston, the improvement is real but modest, and the gap between SBA
and a more elaborate correction is far smaller than the gap between either of them and getting the horizon or the
service target wrong.

### Phase E — Replace the interval with a probability: TSB

**Step 12. Carry a probability, and update it every period.**

```text
every period:
    pi = beta * o_t + (1 - beta) * pi        where o_t = 1 if y_t > 0, else 0
    if y_t > 0:
        z = alpha * y_t + (1 - alpha) * z    (size updates only on demand)

forecast:
    rate = pi * z                            (units per period)
```

Two structural differences follow, and they are the whole argument. **A zero run now moves the forecast:** each zero
multiplies `π` by `(1 − β)`, so the rate decays geometrically toward zero and an item that has stopped being ordered
fades out instead of holding a stale rate. **The estimator is unbiased at an arbitrary point in time** under stationary
demand, because it is a product rather than a ratio, so there is no inversion to correct.

**Step 13. Choose β deliberately, and know what it buys.** β controls how fast the method concludes
an item is dead. A small β (0.05 to 0.1) is stable and slow; a large β (0.3 and above) writes an item off quickly and
will also write it off during a normal gap. Guidance in the original paper suggests keeping β below α, and in the region
of `β ≤ π·α`, to control variance on stationary demand. There is no universal value.

This parameter encodes a business judgement — how long a silence before the system stops believing in the item — and it
is almost always left at a software default. Section 10 lists that as a failure mode for a reason.

### Phase F — Turn the rate into a decision

**Step 14. Multiply by the horizon.**

```text
expected lead-time demand = rate * ( L + R )
```

With SBA at 1.48 units per week and a four-week lead time plus one-week review, expected lead-time demand is 7.4 units.
This multiplication is where model choice stops mattering and horizon choice starts.

**Step 15. Choose a quantile, then round.** Expected lead-time demand is a mean; a replenishment
policy needs a quantile set by the critical ratio, `tau = C_u / ( C_u + C_o )`. For a critical ratio of
0.9 you need the 90th percentile, which for the worked example is about 14 units against a mean of 5.8 —
a factor of 2.4. Then round to an orderable multiple and record the rule, because round-to-nearest and round-up give
different inventory positions when the forecast is below one unit. Methods in this topic produce the mean, not the
quantile; bridging that gap is Module 08.

### Phase G — Choose and validate

**Step 16. Route by class, backtest on a rolling origin, score on the decision.**

1. **Classify** every series at the decision bucket and store the class with the series.
2. **Route**: smooth → SES or SBA; erratic → SES with a short memory, or SBA; intermittent → SBA or
   TSB; lumpy → SBA or TSB, and expect all of them to be poor.
3. **Backtest** every shortlisted method on a rolling origin, retaining the forecast that would have
   been made at each origin. Tune on an earlier window; evaluate on a later one.
4. **Score on the decision**, not on MAPE. A fill-rate or cost simulation at the real lead time and
   service target is the criterion; accuracy measures are diagnostics.
5. **Report by class**, because a portfolio average hides the fact that the method winning on smooth
   items usually loses on lumpy ones.

### 5.10 Parameter practice

Smoothing constants are chosen by one of four routes, in descending order of defensibility:

| Route | How it works | Failure mode |
|---|---|---|
| Grid search on a validation window, minimising decision loss | Defensible | Expensive; overfits if the window is short |
| Grid search minimising a scaled accuracy measure | Common | Optimises the wrong objective; see 01A.10 |
| Fixed by software default | Almost universal | Nobody knows what it is or when it was reviewed |
| Chosen by the planner | Sometimes | Not reproducible, not documented, not auditable |

Two constraints apply regardless of route. Report α and β next to every result, because a comparison between two
parameterisations is not a comparison between two methods. And constrain the search range: α above roughly 0.5 makes the
size estimate a near-copy of the last demand, converting the method into "last observation divided by a smoothed
interval" and losing the stability that was the point.

### 5.11 The classification caution

The four-class grid earns its place because it is cheap and better than treating every item alike. It is not a law, and
three cautions apply.

**The cut-offs are derived, not natural.** The values 1.32 and 0.49 come from a theoretical
mean-squared-error comparison of competing estimators, validated on a large sample of automotive series. Change the
loss, the horizon, or the portfolio and the boundary moves. Kostenko and Hyndman subsequently derived a better boundary
and showed the constant is not constant at all — it is a function of ADI.

**The classes are not ordered by difficulty.** An intermittent item with a stable size is easy: the
only question is when. An erratic item with demand every period and a wild size is a variance problem, and a method that
ignores occurrence may handle it better. Section 9 cites a study where the ranking reversed between the intermittent and
erratic quadrants.

**The class is a property of the history, not of the future.** Items move between quadrants because
of a listing change, a competitor exit, a redesign that removes a failure mode, or a promotion. Classify on a rolling
window and watch migration; a class stored in master data for three years describes a different item.

## 6. Worked example

One warehouse, one spare part, twenty-four weeks, eaches per week. Parameters: α = 0.2 for both components in Croston
and SBA; α = 0.2 and β = 0.1 for TSB; TSB probability initialised at 0.25; Croston interval initialised at the week of
the first demand.

### 6.1 The history and its classification

```text
week  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24
y     0  0  6  0  0  0  0  4 10  0  0  0  0  5  0  0  0  0  0  7  0  0  0  0
```

| Quantity | Value | Unit |
|---|---:|---|
| Periods in window | 24 | weeks |
| Periods with demand | 5 | weeks |
| Total demand | 32 | eaches |
| ADI | 4.80 | periods per demand |
| Mean nonzero size | 6.40 | eaches per occurrence |
| Standard deviation of sizes | 2.06 | eaches |
| CV² | 0.104 | dimensionless |
| Class | intermittent | ADI ≥ 1.32, CV² < 0.49 |

The realised rate over the window is 32 / 24 = 1.333 eaches per week. Keep that number; it is the reference against
which the methods are read at the end of the window.

### 6.2 Croston, step by step

Initialisation at week 3: `z = 6`, `p = 3` (the week of the first demand), so the rate is 2.0000 from weeks 3 to 7.

**Week 8.** Interval q = 8 − 3 = 5, size y = 4.

```text
z = 0.2 * 4  + 0.8 * 6.000 = 0.800 + 4.800 = 5.600
p = 0.2 * 5  + 0.8 * 3.000 = 1.000 + 2.400 = 3.400
rate = 5.600 / 3.400 = 1.6471          SBA = 0.9 * 1.6471 = 1.4824
```

**Week 9.** Interval q = 1, size y = 10.

```text
z = 0.2 * 10 + 0.8 * 5.600 = 2.000 + 4.480 = 6.480
p = 0.2 * 1  + 0.8 * 3.400 = 0.200 + 2.720 = 2.920
rate = 6.480 / 2.920 = 2.2192          SBA = 0.9 * 2.2192 = 1.9973
```

**Week 20.** Interval q = 20 − 14 = 6, size y = 7. (The week-14 update gives `z = 6.184`,
`p = 3.336`, rate 1.8537, SBA 1.6683.)

```text
z = 0.2 * 7  + 0.8 * 6.184 = 1.400 + 4.947 = 6.347
p = 0.2 * 6  + 0.8 * 3.336 = 1.200 + 2.669 = 3.869
rate = 6.347 / 3.869 = 1.6406          SBA = 0.9 * 1.6406 = 1.4766
```

### 6.3 The full recursion

| Week | Demand | Croston | SBA | TSB |
|---:|---:|---:|---:|---:|
| 1 | 0 | — | — | — |
| 2 | 0 | — | — | — |
| 3 | 6 | 2.0000 | 1.8000 | 1.6935 |
| 4 | 0 | 2.0000 | 1.8000 | 1.5242 |
| 5 | 0 | 2.0000 | 1.8000 | 1.3717 |
| 6 | 0 | 2.0000 | 1.8000 | 1.2346 |
| 7 | 0 | 2.0000 | 1.8000 | 1.1111 |
| 8 | 4 | 1.6471 | 1.4824 | 1.4933 |
| 9 | 10 | 2.2192 | 1.9973 | 2.2032 |
| 10 | 0 | 2.2192 | 1.9973 | 1.9829 |
| 11 | 0 | 2.2192 | 1.9973 | 1.7846 |
| 12 | 0 | 2.2192 | 1.9973 | 1.6061 |
| 13 | 0 | 2.2192 | 1.9973 | 1.4455 |
| 14 | 5 | 1.8537 | 1.6683 | 1.8599 |
| 15 | 0 | 1.8537 | 1.6683 | 1.6739 |
| 16 | 0 | 1.8537 | 1.6683 | 1.5065 |
| 17 | 0 | 1.8537 | 1.6683 | 1.3559 |
| 18 | 0 | 1.8537 | 1.6683 | 1.2203 |
| 19 | 0 | 1.8537 | 1.6683 | 1.0983 |
| 20 | 7 | 1.6406 | 1.4766 | 1.6493 |
| 21 | 0 | 1.6406 | 1.4766 | 1.4843 |
| 22 | 0 | 1.6406 | 1.4766 | 1.3359 |
| 23 | 0 | 1.6406 | 1.4766 | 1.2023 |
| 24 | 0 | 1.6406 | 1.4766 | 1.0821 |

Read the table by column. **Croston and SBA are flat across every zero run:** weeks 10 to 13 hold
2.2192 for four consecutive weeks while nothing is demanded. **TSB decays through every zero run:**
the same four weeks fall 1.98, 1.78, 1.61, 1.45. **The ending values bracket the realised rate:** Croston ends at 1.641
(23 per cent above the realised 1.333), SBA at 1.477 (11 per cent above), TSB at
1.082 (19 per cent below) — the ordering the bias literature predicts on a stationary series. Do not
over-read it: one series of twenty-four weeks is an illustration, and five demand events cannot measure a bias of a few
per cent.

### 6.4 What the rate means for the decision

With a four-week lead time and a one-week review, the horizon is five weeks, and the SBA rate gives an expected lead-
time demand of 1.4766 × 5 = 7.38 eaches. The planner does not order 7.38 eaches, so compute the empirical distribution
of four-week demand over the twenty-one overlapping windows:

| Statistic | Value (eaches per 4-week window) |
|---|---:|
| Windows | 21 |
| Windows containing no demand | 5 (23.8%) |
| Mean | 5.81 |
| Median | 6 |
| 90th percentile | 14 |
| Maximum | 14 |

The distribution is not a bell curve around the mean. It has mass at zero, a cluster at 5 to 7, and a tail at 10 to 14;
a mean of 5.8 describes none of those clusters. Now compare the two levers:

| Lever | Change | Effect on the order (eaches) |
|---|---|---:|
| Method, at the end of the window | Croston 1.641 → TSB 1.082, times 5 weeks | 8.20 → 5.41, a spread of 2.8 |
| Service target, same method | mean 5.81 → 90th percentile 14.00 | a spread of 8.2 |

The service target moves the number roughly three times as much as the method choice. That is the quantitative version
of section 3.3, and it is the most useful thing to say in a meeting that has become Croston versus SBA.

### 6.5 The obsolescence case

A second item from the same warehouse: demand in weeks 2, 5, 9 and 12 at sizes 8, 6, 9, 4, then nothing for twelve
weeks. Classification: ADI = 24 / 4 = 6.00, sizes with mean 6.75 and CV² =
0.081, so intermittent. Same parameters, window weeks 13 to 24.

| Week | Demand | SBA | TSB |
|---:|---:|---:|---:|
| 12 | 4 | 2.4145 | 2.3174 |
| 13 | 0 | 2.4145 | 2.0856 |
| 14 | 0 | 2.4145 | 1.8771 |
| 15 | 0 | 2.4145 | 1.6894 |
| 16 | 0 | 2.4145 | 1.5204 |
| 17 | 0 | 2.4145 | 1.3684 |
| 18 | 0 | 2.4145 | 1.2315 |
| 19 | 0 | 2.4145 | 1.1084 |
| 20 | 0 | 2.4145 | 0.9975 |
| 21 | 0 | 2.4145 | 0.8978 |
| 22 | 0 | 2.4145 | 0.8080 |
| 23 | 0 | 2.4145 | 0.7272 |
| 24 | 0 | 2.4145 | 0.6545 |
| **Sum, weeks 13–24** | **0** | **28.97** | **14.97** |

Over twelve weeks with no demand, SBA asserts 29 units of expected demand and TSB asserts 15. The true rate is zero. The
29 units are not a forecast error in an accuracy metric's sense, since no demand occurred; the difference shows up in
inventory, where 29 units justify a reorder point that will never be consumed.

Note what TSB costs. On the first item, which was very much alive, TSB ended 19 per cent below the realised rate. A β
chosen to detect death quickly will also conclude, briefly, that a live item is dead every time it has a normal gap.
That trade is a business decision about how much dead stock you will tolerate against how much service risk you will
take on a slow mover.

### 6.6 A one-step bake-off, and why it settles nothing

Forecasts for weeks 5 to 24 (20 evaluation points), from data available before each week. Mean absolute error (MAE) in
eaches and mean absolute scaled error (MASE), scaled by the in-sample naive error — 2.435 for item 1, 2.348 for item 2.

| Method | Item 1 MAE | Item 1 MASE | Item 2 MAE | Item 2 MASE |
|---|---:|---:|---:|---:|
| Naive (last value) | 2.2000 | 0.904 | 1.9000 | 0.809 |
| SES, α = 0.2 | 2.2898 | 0.941 | 1.7452 | 0.743 |
| Croston | 2.4494 | 1.006 | 2.8590 | 1.218 |
| SBA | 2.3345 | 0.959 | 2.6681 | 1.136 |
| TSB | 2.2924 | 0.942 | 2.0580 | 0.877 |
| Zero forecast | 1.3000 | 0.534 | 0.9500 | 0.405 |

The zero forecast wins both items on both measures. On item 1 the naive forecast beats Croston, SBA and TSB. Three
observations follow.

**The metric is rewarding absence.** With five demand events in twenty weeks, a forecast of zero is
right in most periods and wrong by the size of a demand in the rest. MAE and MASE are minimised by the conditional
median, and the median of a series that is zero three-quarters of the time is zero.

**The sample cannot settle a bias question.** Twenty evaluation points with five events cannot
distinguish a five per cent bias from noise. The table demonstrates metric behaviour, not a method.

**The metric rank contradicts the inventory rank.** The zero forecast has a fill rate of zero by
construction. It is first on the scoreboard and last on the decision, and both are computed correctly. A bake-off
reporting only the left-hand columns is not a bake-off.

## 7. Code, query, or calculation

Two recursions, no library, same convention as section 5. State the initialisation and the interval convention wherever
you deploy this; they are not cosmetic.

```python
def croston(y, a=0.2, sba=False):
    z = p = last = None; out = []
    for t, v in enumerate(y, 1):
        if v > 0:
            if z is None:
                z, p = float(v), float(t)          # init: first size, week of first demand
            else:
                q = t - last                       # inclusive interval, periods
                z = a * v + (1 - a) * z
                p = a * q + (1 - a) * p
            last = t
        out.append(None if z is None else (1 - a / 2 if sba else 1.0) * z / p)
    return out

def tsb(y, a=0.2, b=0.1, p0=0.25):
    z = None; p = p0; out = []
    for v in y:
        p = b * (1.0 if v > 0 else 0.0) + (1 - b) * p
        if v > 0:
            z = float(v) if z is None else a * v + (1 - a) * z
        out.append(None if z is None else p * z)
    return out
```

Expected output for the section 6 history: `croston(y)[-1] == 1.6406`, `croston(y, sba=True)[-1] ==
1.4766`, `tsb(y)[-1] == 1.0821`. For the obsolescence history the last values are 2.6828, 2.4145 and
0.6545 respectively.

Three checks before trusting an implementation, yours or a vendor's. Run it on a series with demand in every period:
Croston should reduce to a ratio of two smoothed series with the SBA multiplier still applied, and if the output equals
plain SES the implementation has discarded the decomposition. Run it on a series whose first demand is in week 1: `p = t
= 1` gives a rate equal to the size, which is correct but arbitrary, so confirm there is no division by zero or silent
default interval. Run it on an all-zero series: every implementation should return "no forecast", not zero, because a
system returning zero has made a claim the data does not support.

## 8. Data, standards, and regulatory context

No standard defines Croston's method, the SBA correction, or TSB. There is no ISO, ANSI, or IEC document fixing the
recursion, the smoothing constants, or the classification cut-offs. What the standards constrain is the **input**: the
identity of the series, the completeness of the zero grid, and the meaning of the contract the forecast feeds. Those
constraints are what make an implementation defensible, and they are usually the part that is missing.

| Standard | Governs | What it constrains here | Grade |
|---|---|---|---|
| GS1 GTIN | Product identity | The item dimension; a pack-size change is a new GTIN, splitting one intermittent history into two shorter, sparser ones | B |
| GS1 GLN | Party and location identity | The location dimension; a warehouse re-code creates a false zero history and a false new item | B |
| GS1 SSCC | Logistics unit identity | Reconstructing receipts and shipments when demand must be derived from movement rather than orders | B |
| EPCIS 2.0 (GS1) | Event data: what, where, when, why | Timestamped consumption events, the cleanest basis for a series not censored by shipment batching | B |
| ISO 8000 series | Data quality | Completeness and consistency of the item, location, and period keys the dense grid depends on | B |

### 8.2 The zero grid is a data-engineering requirement

A history suitable for these methods has four properties, none optional:

| Property | Why it is needed | How it is usually lost |
|---|---|---|
| Dense | Zeros must exist as rows; a sparse sales-line table has none | Extract built from transactions rather than a calendar-cross-item grid |
| Keyed to a stable item and location | The series must be the same object over time | Re-used codes, pack changes, warehouse re-codes |
| Bucketed at the decision period | Class and rate are bucket-dependent | Daily data aggregated to monthly for convenience, then planned weekly |
| Flagged for availability | Censored demand must be distinguishable from absence | No stock-out flag captured at item-day level |

This is the same requirement that [Module 01A topic 10](../../01_foundations/01A_foundations_units_time_assumptions/10_intermittent-metrics.md)
sets for evaluation, and it applies with more force here: a method estimating an occurrence probability from a grid with
missing rows produces a probability that is too low, and under-orders every item affected.

### 8.3 Installed base and asset identity

For spare parts the demand rate is not a free parameter. It is approximately the product of the installed base and a
failure rate: `expected demand per period ≈ installed base × failure rate per unit per period`. Two consequences follow.
An asset register — which units are in service, of which model, since when — is a better predictor of obsolescence than
any run of zeros, because it tells you when the last unit of a model leaves service. And identifiers matter: a part
fitting three device models has three installed bases, and a single series conflates them.

| Requirement | Jurisdiction | Status as of 2026 | Grade |
|---|---|---|---|
| Unique Device Identification (UDI), 21 CFR Part 830, with the Global Unique Device Identification Database (GUDID) | United States, medical devices | In force; identifiers published in GUDID | B |
| UDI under Regulation (EU) 2017/745 (MDR), Article 27 | European Union | In force; phased by device class, with the Eudamed database | B |

These do not regulate forecasting. They supply a stable device identity that makes an installed-base join possible,
which is more than most spare-parts catalogues have.

### 8.4 Contractual service definitions

| Instrument | What it fixes | Consequence for method choice | Grade |
|---|---|---|---|
| Service level agreement (SLA) | Fill rate, on-time-in-full, measurement window, exclusions | The target quantile comes from the contract, not a default; a 95 per cent fill target is not a 95 per cent CSL | B |
| Availability-based service contracts | Payment tied to asset uptime rather than parts delivered | Pushes the critical ratio up sharply, because a stock-out stops a revenue-bearing asset | C |
| Internal planning targets by item class | Target CSL by ABC class | A target is not a measurement, and class-based targets usually ignore the critical ratio | C |

### 8.5 Convention rather than requirement

The cut-offs 1.32 and 0.49 are conventions from one study's loss function and sample, not a law. "α =
0.1" and "β = 0.1" are software defaults, not recommendations. "SBA is the standard benchmark" is a
statement about the literature's habits. "Croston is what the ERP does" is a statement about a configuration decision
made by someone who has left the company. An analyst who can separate these from requirements can re-derive them; one
who cannot will defend a default as a physical constant.

## 9. Industry practice and case evidence

### 9.1 The original motivation

Croston's 1972 paper, "Forecasting and Stock Control for Intermittent Demands", was written by an author at P. & O.
Steam Navigation Co. and published in *Operational Research Quarterly*. The abstract states the finding plainly:
exponential smoothing is frequently used for stock-control forecasts, but intermittent demands almost always produce
inappropriate stock levels, and demand for constant quantities at fixed intervals can generate stock levels **up to
double** the quantity really needed. The remedy was to estimate the size of demand and the demand frequency separately,
and to adjust the safety- stock rules accordingly. (Grade A)

That is worth pausing on. The method that became the standard for slow movers was not invented to improve accuracy. It
was invented to stop a stock-control system from holding twice the inventory it needed. The framing of the problem —
inventory, not error — is the one this topic keeps returning to.

### 9.2 The bias story, in order

| Step | Contribution | Source | Grade |
|---|---|---|---|
| 1 | Croston's method is widely treated as unbiased for two decades | Practice, reflected in the textbooks | — |
| 2 | The derivation contains an error: the expectation of a ratio is not the ratio of expectations, so the estimator is positively biased | Syntetos and Boylan (2001), *IJPE* 71, 457–466 | A |
| 3 | A correction of `(1 − α/2)` is proposed and outperforms Croston at conventional smoothing constants | Syntetos and Boylan (2005), *IJF* 21(2), 303–314 | A |
| 4 | The correction over-compensates; SBA can carry a negative bias of up to roughly 12 per cent at long intervals | Teunter and Sani (2009), *EJOR* 194(1), 177–183 | A |
| 5 | A refinement reports mean absolute bias near 1 per cent against roughly 5–6 per cent for SBA and Croston in the same simulation design | Teunter and Sani (2009) | A |
| 6 | An "exact" correction exists and empirically improves on SBA only in the fourth or fifth significant digit | Shale, Boylan and Johnston (2006), *JORS* 57(5), 588–592 | A |
| 7 | Croston and SBA do not update during zero periods, creating an obsolescence problem and a positive bias at any point in time | Teunter, Syntetos and Babai (2011), *EJOR* 214(3), 606–615 | A |

The shape of this sequence is more instructive than any entry in it. A method was believed unbiased, was shown biased,
was corrected, and the correction was shown to overshoot — and then the whole framing was replaced by a method that does
not use a ratio at all. Each step corrects the previous one; the last step changes the state variable.

### 9.3 Which method wins: the honest answer

| Study | Data | Finding | Grade |
|---|---|---|---|
| Petropoulos et al. (2013), *IMDS* | 3,000 automotive SKUs | SBA most unbiased and most accurate on MASE and MdASE; Croston positively biased, worsening as α rose | A |
| Teunter and Duncan (2008), *JORS* | UK Royal Air Force data | On service-level and stock-holding criteria rather than per-period error, Croston and bootstrapping clearly beat moving average and SES | A |
| Syntetos, Babai and Gardner (2015), *JBR* | Over 7,000 series, jewellery retail and electronics spares | Simple parametric methods perform well on inventory-service trade-off curves; bootstrapping's complexity may not be justified | A |
| Pennings, van Dalen and van der Laan (2016), *EJOR* | Five empirical datasets including RAF, automotive, navy | SBA best overall on forecast accuracy; a method conditioning on elapsed time since last demand best on inventory, with up to 14 per cent inventory investment reduction | A |
| Doszyń (2019), *Journal of Forecasting* | About 9,000 monthly series | TSB plus a seasonal method recommended; Croston and SBA performed worse than much simpler SES and moving average | A |
| Doszyń (2020), *ERSJ* | Over 16,000 warehouse items, by demand class | TSB best overall and best on intermittent and lumpy items; Croston and SBA worst. On erratic and smooth items the ranking reversed: SBA best, TSB worst | A− |
| Thaiudomsap and Sribundit (2022), *SEHS* | 41 hospital drug series, 19 intermittent, 22 lumpy | TSB best accuracy (RMSE 12.28, MASE 1.04) but highest bias, deliberately under-forecasting to limit obsolescence; no method won on both accuracy and bias | B |

Read the table as a set of disagreements, because that is what it is. The automotive and RAF studies favour SBA. The
large corporate and warehouse studies favour TSB and find Croston and SBA worse than SES. The hospital study finds TSB
most accurate and most biased at once.

Three defensible conclusions follow. **SBA is a reasonable default** for intermittent and lumpy items, with the most
consistent support across independent accuracy studies. **TSB is preferable when obsolescence is a live risk**, and the
studies favouring it are disproportionately the large heterogeneous corporate portfolios closest to most readers'
situation. **A simple method is a legitimate control**: in two of the largest studies SES or a moving average beat
Croston and SBA, and if your method cannot beat SES on your data and your metric, it has not earned its complexity.

### 9.4 What practitioners actually do

| Observed practice | Why it persists | Consequence |
|---|---|---|
| One method for the whole catalogue | Simpler to configure, audit, and explain | Wrong for most items by construction |
| Croston chosen because it is the only intermittent option in the ERP | Procurement decision made years ago | SBA's correction unavailable even where it would help |
| Smoothing constants left at defaults | Nobody owns them; changing them needs a change request | The method's behaviour is unknown to its defenders |
| Classification stored in master data and never refreshed | It was a project deliverable | Items routed by a label describing a series that ended |
| Method chosen by MAPE bake-off | MAPE is the metric in the template | The quietest method wins |
| Obsolescence handled by a planner's manual review | The system cannot express "this item is dead" | Dead stock found at the write-off |

### 9.5 The evidence that horizon and target dominate

| Finding | Source | Grade |
|---|---|---|
| Aggregating into lower-frequency buckets yields higher achieved service levels and better service–cost efficiency, especially at high target service levels | Babai, Ali and Nikolopoulos (2012), *Omega* 40(6), 713–721 | A |
| Aggregating to lead time plus review period (L + R) is recommended, tested on about 5,000 RAF SKUs | Nikolopoulos et al. (2011), *JORS* 62(3), 544–554 | A |
| Temporal aggregation using lead-time information gives higher inventory efficiency and realised cycle service level when high service levels matter | Chau (2020), MIT thesis, 5,368 SKUs | C |
| Aggregation-to-horizon is a practical heuristic; gains from aggregation increase with the forecast horizon | Kourentzes, Rostami-Tabar and Barrow (2017), *JBR* 78 | A |
| No overall accuracy gain from aggregation alone, but benefits appear with weighted combination, especially at longer lead times | Rostami-Tabar, Goltsos and Wang (2023), *Computers in Industry* 145 | A |

The consistent finding is that the bucket and the horizon change the service outcome more reliably than the estimator
does. That is the empirical version of section 6.4's arithmetic, and the reason section
3.3 puts the method last.

### 9.6 The 2026 tooling context

The methods are widely implemented in R's `forecast` (Croston with a type argument for SBA and the Shale–Boylan–Johnston
variant), `fable`, `tsintermittent`, and in Python via `statsforecast` (`CrostonSBA`) and AutoGluon. (Grade C:
implementations, not findings.) Defaults for initialisation and the α convention differ between libraries, so two tools
can produce different numbers from the same series with the same stated parameters, and none of them chooses the method
for you.

## 10. Failure modes and red flags

| Failure | Red flag visible in a meeting or a dataset | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| A level-based method on a zero-heavy series | Forecast decays to zero through every gap, then spikes after each order | The level tracks a series that is mostly zero | Systematic under-order, then over-order | Classify first; route intermittent and lumpy series away from SES and ARIMA |
| The rate is read as an order quantity | A purchase order for 1.4 units, or a system minimum silently overriding it | The forecast is per period; the order is per cycle and must be a quantile | Ordering below any real demand size, or rounding nobody has examined | Multiply by L + R, apply the critical ratio, then round and record the rule |
| Croston applied to a series with no zeros | Every item in the class has demand in every period | Croston degenerates to a ratio with extra parameters | Worse than SES with more configuration to defend | Route by class; keep SES for smooth items |
| The forecast freezes on a dead item | A discontinued part still shows a healthy rate twelve months after the last order | Croston and SBA do not update during zero runs | Phantom demand justifies inventory that will be written off | Use TSB where obsolescence is material, or add an explicit death test |
| SBA described as unbiased | A method note says "SBA removes the bias" | It removes the leading term and can overshoot negatively | Small systematic under-forecast, worst at long intervals | Say "approximately unbiased"; measure bias and report its sign |
| Constants tuned on the evaluation window | Reported accuracy better than any later replication | Selection on the outcome | The improvement is the tuning, not the method | Tune on an earlier window; report α and β with every result |
| Bake-off scored on MAPE or MAE | The zero forecast places mid-table or better | The metric's optimum is the median, which is zero | The quietest method wins as service deteriorates | Score on a service or cost simulation at the real lead time and target |
| Classification computed on the wrong bucket | An item is "intermittent" in master data and smooth weekly | ADI and CV² are bucket-dependent | Wrong method routed, under an authoritative-looking label | Classify at the decision bucket; re-derive after any bucket change |
| Class assigned once and never revisited | The field has not changed in three years | Lifecycle events move items between quadrants | Items routed by a series that no longer exists | Reclassify on a rolling window; monitor migration counts |
| Cut-offs treated as law | A design document cites 1.32 and 0.49 as requirements | They are one study's error boundaries | Boundary items misrouted; the rule defended instead of tested | Re-derive the boundary on your portfolio and loss, or label it a convention |
| Censored demand read as zero demand | Recorded demand equals recorded shipments in stock-out weeks | Unmet demand is unobservable | The model learns the stock-out and reproduces it | Flag availability; model demand, not shipments |
| Method chosen at one horizon, used at another | A one-week bake-off sets a twelve-week safety stock | Error grows with horizon and rankings can change | The winner at h = 1 is not the winner at h = 12 | Backtest at the decision horizon, or aggregate to L + R |
| Forecasts not retained | The backtest rebuilds history from today's item master | Series identity changes retroactively | The evaluation is a fit to history dressed as a forecast | Version the forecast and the item master; evaluate on the vintage |

Two deserve expansion. **The forecast freezes on a dead item** produces no error signal — an item with no demand
produces no forecast error — so it never appears in an accuracy report, only years later as a write-off. **Forecasts not
retained** invalidates every other row, because a bake-off that rebuilds its own inputs can prove anything.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks if it fails | How to test it |
|---|---|---|---|
| The occurrence process is stationary over the window | The historical rate is used as the future rate | A trend, listing change, or competitor exit shifts the true rate and the forecast lags by its memory | Plot occurrence rate by quarter; test for a break; compare with installed base |
| Demand size is independent of the preceding interval | The decomposition multiplies two separately estimated quantities | Emergency orders after stock-outs are small; accumulation orders after long gaps are large | Plot nonzero size against preceding interval |
| Zeros mean no demand, not missing data | The occurrence estimate counts periods | The rate is understated in proportion to the missing share | Compare the grid against the assortment master |
| Demand is not censored by availability | The occurrence estimate must reflect the market, not the policy | The model reproduces the stock-out it learned from | Flag stock-out days; compare demand in available and unavailable periods |
| The bucket matches the decision period | Class and rate are properties of the bucket | A monthly rate applied to a weekly cycle under-orders by roughly the bucket ratio | Compare the forecasting bucket with the replenishment cycle |
| The series is long enough to estimate a rate | A rate from three events has a wide interval around it | The estimate is dominated by the initialisation and the last event | Count demand events, not periods; require a minimum and report it |
| Smoothing constants are stable and documented | The recursions are entirely determined by them | An undocumented change silently changes every forecast | Read α and β out of the system; diff against policy |
| The item remains ranged at the location | A de-listed item's zero is not the same process | The occurrence probability is dragged down by periods when the item could not be ordered | Join to ranging history; truncate at de-listing |
| There is no substitution between items | Each series is modelled as its own process | A stock-out of A produces demand for B, contaminating both | Look for negative correlation between substitutes at a location |
| The evaluation horizon equals the decision horizon | Method rankings are horizon-dependent | The winner at one week is not the winner at twelve | Backtest at the decision horizon |
| The service target is a quantile, not a preference | The order quantity comes from the quantile | The mean is used and service sits below target, worst on lumpy items | Derive the critical ratio from cost; compare with policy |
| The installed base is stable or known | For spares, demand is base times failure rate | A model change removes a failure mode and demand stops without a trend | Track the asset register; plot demand against base |

Three limits deserve statement. **No method here produces a distribution:** each returns a conditional mean, the
decision needs a quantile, and bridging that gap requires the compound Poisson or hurdle models of Module 08, or an
empirical bootstrap. **The methods are descriptive of a rate, not causal:** a rate estimated from history does not tell
you what happens when the installed base changes, when a customer consolidates suppliers, or when a redesign removes a
failure mode. **The evidence base is fragmented:** section 9's studies disagree in ways that track their datasets rather
than their methods, and anyone claiming one method is best is quoting one of them without the others.

## 12. Exercises

### Exercise 1 — Classify and route (application, 40 minutes)

Classify each ten-week series with ADI and CV², and name the method you would route it to and why. Quantities are eaches
per week.

| Series | Weeks 1 to 10 |
|---|---|
| a | 5, 5, 4, 6, 5, 5, 4, 6, 5, 5 |
| b | 1, 9, 1, 9, 1, 9, 1, 9, 1, 9 |
| c | 0, 0, 6, 0, 0, 7, 0, 0, 6, 0 |
| d | 0, 0, 9, 0, 0, 1, 0, 0, 18, 0 |
| e | 0, 7, 0, 0, 0, 8, 0, 0, 0, 7 |
| f | 0, 0, 0, 0, 0, 0, 0, 0, 0, 4 |

**Deliverable:** a six-row table with nonzero count, ADI, CV², class, the route, and one sentence of
justification. **Acceptance criteria:** formulas shown; CV² on nonzero sizes only; series f handled explicitly rather
than dropped; a method named for each class. **Time:** 40 minutes.

### Exercise 2 — Hand-run Croston (calculation, 45 minutes)

Run Croston by hand on `0, 0, 5, 0, 0, 0, 8, 0, 0, 6` with α = 0.3, initialising `z` at the first size and `p` at the
week of first demand, inclusive interval convention. Then apply the SBA correction.

**Deliverable:** a table of the three demand events with intermediate values, and a paragraph on how
many periods the forecast stays constant. **Acceptance criteria:** every arithmetic step shown; the correction applied
to each rate; constant periods identified by week number. **Time:** 45 minutes.

### Exercise 3 — TSB on the same series (calculation, 40 minutes)

Run TSB on the same series with α = 0.2, β = 0.2 and initial probability 0.3. Compare the final forecast with exercise
2's results and explain the difference in sign and magnitude.

**Deliverable:** a ten-row table of the TSB forecast and a paragraph of comparison. **Acceptance
criteria:** the probability recursion shown for every period including zeros; the final rate computed as
`π × z`; the explanation identifies which method reacted to the week-10 demand. **Time:** 40 minutes.

### Exercise 4 — The obsolescence arithmetic (analysis, 45 minutes)

Using section 6.5, compute the total expected demand SBA and TSB assert over weeks 13 to 24. Assume a four-week lead
time, a one-week review, and a policy of ordering expected lead-time demand at each cycle start. How many units would
each method cause to be ordered across the twelve weeks, and what happens to them?

**Deliverable:** the two totals, the order schedules, and a one-paragraph recommendation.
**Acceptance criteria:** arithmetic shown; units converted into inventory and then into a cost at a
stated holding rate; the recommendation names the parameter controlling the trade-off. **Time:** 45 minutes.

### Exercise 5 — Design the bake-off (design, 60 minutes)

Write the evaluation protocol for choosing between SES, Croston, SBA and TSB across 4,000 item-location series,
implementable by another analyst without asking you a question.

**Deliverable:** a one-page protocol covering the demand construct, grid, bucketing,
train/validate/test split, parameter search, rolling-origin scheme, metrics, aggregation, and decision rule.
**Acceptance criteria:** the horizon is stated and matched to the replenishment cycle; the decision metric is a service
or cost measure; parameters are tuned outside the evaluation window; results are reported by demand class; a zero
forecast is included as a control. **Time:** 60 minutes.

### Exercise 6 — The accuracy trap (critique, 40 minutes)

A colleague presents section 6.6 and concludes that the zero forecast is best and forecasting should be switched off for
slow movers. Write a one-page response.

**Deliverable:** a critique with the mechanism, the correct interpretation, and the measurement you
would add. **Acceptance criteria:** the mechanism is identified as the metric's optimum rather than a property of the
data; the response does not merely assert that service matters; a specific additional measurement is proposed with its
grain and window. **Time:** 40 minutes.

### Exercise 7 — Parameter sensitivity (analysis, 45 minutes)

Recompute Croston and SBA at α = 0.05, 0.1, 0.2, 0.4, and TSB at β = 0.05, 0.1, 0.3 with α = 0.2, on the section 6
history.

**Deliverable:** a table of final rates by method and parameter, plus a paragraph on which parameter
moves the answer most and in which direction. **Acceptance criteria:** at least twelve combinations; the direction of
each effect stated; one sentence on why a comparison at software defaults is not a method comparison. **Time:** 45
minutes.

### Exercise 8 — Horizon and target leverage (calculation, 50 minutes)

Using the section 6.4 lead-time distribution, compute the order quantity at critical ratios of 0.50,
0.80, 0.90 and 0.95, and compare the spread with the spread across methods at a fixed target.

**Deliverable:** a table of quantiles by critical ratio, the interpolated quantities, and the
comparison. **Acceptance criteria:** the interpolation rule stated; the method spread taken from the same series; the
conclusion expressed as a ratio, not an adjective. **Time:** 50 minutes.

### Exercise 9 — The classification audit (critique, 40 minutes)

A master-data field assigns every item to one of four classes. It was populated once, three years ago, from monthly
data, and the company has since moved to weekly planning. Write an audit note.

**Deliverable:** a note listing the three defects, the direction of each effect, and a remediation
plan with an owner and cadence. **Acceptance criteria:** each defect signed as over- or under-stating intermittency; the
bucket change identified as a classification change, not a data-quality issue; remediation a recurring process rather
than a one-off rebuild. **Time:** 40 minutes.

### Exercise 10 — The method policy memo (design, 55 minutes)

Write a one-page memo to a planning director recommending a method policy for 12,000 spare-parts items, 70 per cent of
which have fewer than six demand events a year.

**Deliverable:** the policy — which classes get which methods, parameters and their owner, evaluation
cadence, obsolescence rule, and the one thing you would fix first. **Acceptance criteria:** stated as rules a system can
implement; parameter ownership is a named role; evaluation is at the decision horizon; the first fix is justified by
expected effect, not novelty. **Time:** 55 minutes.

## 13. Self-check answers

**Exercise 1.** `a: n=10, positive=10, ADI=1.00, mean=5.00, sd=0.63, CV2=0.0160 → smooth → SES.`
`b: ADI=1.00, mean=5.00, sd=4.00, CV2=0.6400 → erratic → SES with a short memory.`
`c: positive=3, ADI=3.33, mean=6.33, sd=0.47, CV2=0.0055 → intermittent → SBA.`
`d: positive=3, ADI=3.33, mean=9.33, sd=6.94, CV2=0.5536 → lumpy → SBA or TSB.`
`e: positive=3, ADI=3.33, mean=7.33, sd=0.47, CV2=0.0041 → intermittent → SBA.`
`f: positive=1, ADI=10.0, CV² undefined → unclassifiable → no method.`

Series c and e differ only in demand sizes and gap placement; both are intermittent because the sizes are stable, and
for both the hard question is "when", not "how much". Series d has the same ADI and a CV² two orders of magnitude
larger, which makes it lumpy and makes every method here provisional. Series f is the case that breaks naive code: with
one nonzero size the standard deviation is zero and CV² is undefined, not small. An implementation writing `"lumpy" if
cv2 >= 0.49 else "intermittent"` classifies it as lumpy because a comparison against `NaN` is false — wrong. The correct
answer is that the class cannot be assigned, and the item needs a prior, more history, or judgement. One observation is
not a rate.

**Exercise 2.** `init at week 3: z = 5.000, p = 3.000, rate = 1.6667.` Week 7: q = 4, y = 8, so
`z = 0.3*8 + 0.7*5.000 = 5.900`, `p = 0.3*4 + 0.7*3.000 = 3.300`, rate 1.7879, SBA × 0.85 = 1.5197.
Week 10: q = 3, y = 6, so `z = 0.3*6 + 0.7*5.900 = 5.930`, `p = 0.3*3 + 0.7*3.300 = 3.210`, rate
1.8474, SBA 1.5702. The forecast holds at 1.6667 for weeks 3 to 6, at 1.7879 for weeks 7 to 9, and
at 1.8474 in week 10. Five of ten periods contain no update. That is the freezing behaviour, and it is why the method is
stable and why it is blind.

**Exercise 3.** `w1: pi = 0.2400, no forecast. w2: pi = 0.1920, no forecast.` w3: `pi = 0.2 +
0.8*0.192 = 0.3536`, z = 5.000, rate 1.7680. w4: pi 0.2829, rate 1.4144. w5: pi 0.2263, rate 1.1315.
w6: pi 0.1810, rate 0.9052. w7: `pi = 0.2 + 0.8*0.1810 = 0.3448`, `z = 0.2*8 + 0.8*5.000 = 5.600`, rate
1.9311. w8: pi 0.2759, rate 1.5449. w9: pi 0.2207, rate 1.2359. w10: `pi = 0.2 + 0.8*0.2207 =
0.3766`, `z = 0.2*6 + 0.8*5.600 = 5.680`, rate 2.1388.

TSB ends higher than both Croston (1.8474) and SBA (1.5702), the opposite of section 6. The reason is mechanical: β =
0.2 gives the occurrence probability a short memory, so a demand in the last period lifts it hard, and the size estimate
also rises because the week-10 demand of 6 exceeded the smoothed size. TSB is not systematically higher or lower than
SBA; it responds to the most recent occurrence, and where that occurrence sits at the end of the window, it wins. On a
ten-week window with three events, the ranking is determined by where the last event fell — a property of the sample,
not of the methods.

**Exercise 4.** SBA over weeks 13–24 is `12 × 2.4145 = 28.97` units; TSB is 14.97; the difference is
14.00 units. Under a policy of ordering expected lead-time demand each cycle with L + R = 5 weeks,
SBA orders 12.07 units per cycle and TSB falls from about 10.4 units in week 13 to about 2.8 in week
24. Across roughly two and a half cycles, SBA places about 29 units into stock and TSB about 15. At a
20 per cent annual holding rate on a 40-currency-unit part, the 14-unit difference is 560 currency units of inventory,
or about 112 a year of holding cost. At 2,000 dying items the same arithmetic gives roughly 1.1 million of inventory and
224,000 a year of holding cost. The numbers are illustrative; the mechanism is not. The recommendation should name β as
the control: a larger β writes items off faster and reduces dead stock at the cost of under-ordering live items during
normal gaps, which is a business decision for whoever owns the write-off budget.

**Exercise 5 rubric.** A strong protocol states the demand construct (consumption, not shipments),
the dense grid, and the decision bucket; splits the history into an early window for parameter search and a later window
for evaluation with a gap against leakage; uses a rolling origin at the decision horizon; tunes α and β per class rather
than per item, to avoid overfitting sparse series; scores on a fill-rate or cost simulation at the real lead time and
target; reports by class with counts; and includes SES, naive and the zero forecast as controls. A weak protocol
evaluates one step ahead on a single holdout, tunes on that same window, reports MASE only, and compares a tuned method
against one at default — the most common defect and the hardest to see in a results table.

**Exercise 6 rubric.** A strong response shows that MAE and MASE are minimised by the conditional
median, that the median of a series which is zero in three-quarters of periods is zero, and that the table therefore
describes the metric rather than the data. It proposes the missing measurement — a fill- rate simulation at the real
lead time, by class — and predicts that the zero forecast has a fill rate near zero. A weak response asserts that
service matters without the mechanism, or attacks the arithmetic, which is correct.

**Exercise 7 rubric.** A strong table shows the final Croston rate rising monotonically with α,
because a larger α weights the most recent demand more heavily and shortens the memory of the interval, and the SBA
multiplier falling with α, so the two effects partly offset. It shows TSB's final rate falling as β falls, because a
small β gives the probability a long memory and a long zero run leaves it low. It concludes that a comparison at
software defaults compares two arbitrary points.

**Exercise 8.** The sorted four-week windows are
`0, 0, 0, 0, 0, 4, 5, 5, 5, 5, 6, 6, 6, 7, 7, 7, 7, 10, 14, 14, 14` (twenty-one windows). With linear
interpolation between order statistics, the quantiles are 6.0 at a critical ratio of 0.50, 10.0 at
0.80, and 14.0 at both 0.90 and 0.95. The last two coincide because the distribution is discrete and
lumpy: nothing lies between 10 and 14, and the top three order statistics are all 14. That is itself a finding — the
service target stops being a continuous dial and becomes a choice between covering one demand event and covering two.
The spread from median (6) to the 90th percentile (14) is 8 units; the spread across methods at a fixed target is 2.8
units over the five-week horizon. The ratio is roughly
2.9 to 1 in favour of the target, so a team spending a quarter on Croston versus SBA and no time on the
critical ratio has spent its effort in the wrong proportion.

**Exercise 9 rubric.** A strong note identifies three defects. First, monthly bucketing understates
intermittency: a weekly series with demand every third week has ADI 3 and is intermittent, while the same demand
aggregated monthly may have demand every month and classify as smooth, so the stored classes overstate smoothness.
Second, the field is static, so items that died or launched in the last three years carry a class from a different
series. Third, the population is stale: items added since have no class and deleted items still do. Remediation is
scheduled reclassification on a rolling window at the decision bucket, with migration counts reported, owned by the
demand-planning lead, on a quarterly cadence.

**Exercise 10 rubric.** A strong memo routes smooth items to SES, intermittent and lumpy items to SBA
with TSB as the obsolescence control, and holds a small evaluation set at the decision horizon. It names the owner of α
and β as a role, sets a review cadence, and states an obsolescence rule in terms of elapsed time and installed base
rather than a forecast threshold. Its first fix is the largest expected effect — usually the critical ratio or the lead-
time bucket, not the method — justified with exercise 8's arithmetic. A weak memo recommends "the best method" for the
whole catalogue with no owner, no cadence, and no quantile.

## 14. Interview questions

**1. "Why does exponential smoothing fail on intermittent demand?"**
Strong: SES carries a level, the level of a series that is zero in most periods sits near zero, and the variance is
driven by occurrence rather than by the level; the forecast decays through each zero run and jumps after each demand, so
it is both late and volatile. Weak: "because there are zeros in the data" — the symptom without the mechanism.

**2. "What are the two components, and why smooth them separately?"**
Strong: the occurrence process and the size process; a zero is an observation of occurrence and not of size, so
smoothing the raw series lets a zero drag the size estimate down when no size observation occurred; the components can
have different dynamics and warrant different constants. Weak: "size and frequency", which is vocabulary without the
reason.

**3. "What is the bias in Croston's method, and where does it come from?"**
Strong: the estimator is a ratio of two smoothed random quantities, so the expectation of the ratio exceeds the ratio of
expectations by roughly the squared coefficient of variation of the denominator; the inflation grows with the smoothing
constant and with short intervals, so the correction is a function of α. Weak: "Croston over-forecasts" with no
mechanism, or a claim that the bias is a fixed percentage.

**4. "Why is the SBA correction (1 − α/2) rather than a fixed number?"**
Strong: the bias scales with the noise in the interval estimate, which α governs, so a fixed multiplier could only be
right at one parameter value; the quoted 0.95 is `(1 − α/2)` at α = 0.1, one system's default. Weak: reciting "multiply
by 0.95" as a rule.

**5. "What does TSB do that SBA cannot?"**
Strong: TSB carries an occurrence probability updated every period rather than an interval updated only on demand, so a
zero run drives the forecast down and a dead item decays instead of holding a stale rate; it is unbiased at an arbitrary
point in time, at the cost of a second constant encoding how quickly the business will declare an item dead. Weak:
calling TSB "Croston with a different smoothing constant".

**6. "Which method is most accurate?"**
Strong: refuses the premise — it depends on class mix, error measure, horizon, and whether the criterion is accuracy or
inventory; cites the disagreement, with automotive and RAF studies favouring SBA, large corporate portfolios favouring
TSB, and two large studies finding SES beat both; then says what it would measure on the questioner's data. Weak: names
a method confidently.

**7. "How would you evaluate a method choice for 10,000 slow-moving SKUs?"**
Strong: classify at the decision bucket, route by class, backtest on a rolling origin at the decision horizon, tune on
an earlier window, score on a service or cost simulation at the real lead time and critical ratio, report by class with
counts, and include SES, naive and the zero forecast as controls. Weak: compute MAPE by SKU and average.

**8. "A vendor claims AI improves intermittent-demand accuracy by 30 per cent. What do you ask?"**
Strong: the demand construct, grid and bucket, class mix, metric and its zero rule, horizon, whether the comparison
method was tuned equally, whether forecasts were retained, and whether the improvement is on service or on error; grade
the claim D until those arrive. Weak: asks which model they use.

**9. "Why might a better forecast make service worse on a slow mover?"**
Strong: the forecast enters a policy that needs a quantile, and a method improving mean accuracy can shift the
distribution's shape or the bias direction in a way that lowers the achieved quantile; on lumpy demand the mean is a
quantity rarely demanded, so improving it can leave the order quantity unchanged where it matters. Weak: says it cannot
happen.

**10. "What is the difference between ADI and the occurrence probability?"**
Strong: ADI is periods per demand and the probability is its reciprocal in expectation; ADI is estimated from completed
intervals and cannot be updated during a zero run, while the probability can be updated every period — which is exactly
why TSB exists. Weak: says they are the same thing in different units.

**11. "Where would you spend effort first: the method, the horizon, or the service target?"**
Strong: the target and the horizon, because the order quantity is a quantile of lead-time demand, so the target moves it
multiplicatively while the method moves the mean by a few per cent; offers the arithmetic — a factor of two or more
between median and 90th percentile against a ten to twenty per cent spread across methods — and notes the ordering
should be checked on the portfolio. Weak: says the method, because that is what the question sounds like it is about.

**12. "How would you detect that an item has become obsolete?"**
Strong: combines three signals — a run of zeros long relative to the item's own ADI, a decline in installed base or
active customers, and a lifecycle event in the product master such as a last-time buy or successor part; notes that
forecast decay alone is slow and the installed-base signal is usually earlier, and that the write-down is a financial
decision, not a forecasting one. Weak: says "when the forecast goes to zero", which is what the method is supposed to
do, not how you know.

## 15. Key terms

- **Intermittent demand** — demand in some periods and not others, so the history contains zeros.
- **Lumpy demand** — intermittent demand whose nonzero sizes are also highly variable.
- **Erratic demand** — demand in most periods with a highly variable size; a variance problem.
- **Smooth demand** — demand in most periods with a stable size.
- **ADI** — average inter-demand interval, in periods per demand occurrence.
- **CV²** — squared coefficient of variation of nonzero demand sizes; dimensionless.
- **Occurrence probability** — the per-period probability that demand occurs; the TSB state.
- **Demand size** — the quantity demanded in a period where demand occurred.
- **Demand rate** — expected demand per period; units per period.
- **Croston's method** — separate exponential smoothing of intervals and sizes, combined as a ratio.
- **Smoothing constant** — the weight given to the newest observation in an exponential recursion.
- **Initialisation** — starting values of the size and interval estimates; dominates short histories.
- **SBA** — the Syntetos–Boylan approximation: Croston multiplied by `(1 − α/2)`.
- **Bias correction factor** — the multiplier removing the leading bias term; a function of α.
- **TSB** — the Teunter–Syntetos–Babai method, smoothing occurrence probability every period.
- **Stale rate** — a forecast that still asserts demand after demand has stopped.
- **Obsolescence** — the state in which an item is no longer demanded.
- **Dead stock** — inventory of an item with no reasonable prospect of demand.
- **Excess stock** — inventory above a stated policy level for an item that still has demand.
- **Lead-time demand** — demand accumulated over lead time plus review period.
- **Critical ratio** — underage cost divided by underage plus overage cost; the target quantile.
- **CSL** — cycle service level, the probability of no stock-out in a replenishment cycle.
- **Fill rate** — the fraction of demand units served from stock.
- **Temporal aggregation** — forecasting at a coarser bucket than the decision bucket.
- **ADIDA** — aggregate–disaggregate intermittent demand approach; aggregates to L + R.
- **Censored demand** — true demand unobservable because stock was insufficient.
- **Installed base** — the population of assets in service that generates spare-parts demand.
- **Bootstrap forecast** — resampling sizes and intervals to build an empirical lead-time demand
  distribution.

## 16. Sources

**Peer-reviewed**

- Croston, J. D. (1972). "Forecasting and Stock Control for Intermittent Demands." *Operational
  Research Quarterly*, 23(3), 289–303. DOI: 10.1057/jors.1972.50. (A)
- Syntetos, A. A., and Boylan, J. E. (2001). "On the bias of intermittent demand estimates."
  *International Journal of Production Economics*, 71(1–3), 457–466.
  DOI: 10.1016/S0925-5273(00)00143-2. (A)
- Syntetos, A. A., and Boylan, J. E. (2005). "The accuracy of intermittent demand estimates."
  *International Journal of Forecasting*, 21(2), 303–314.
  DOI: 10.1016/j.ijforecast.2004.10.001. (A)
- Syntetos, A. A., Boylan, J. E., and Croston, J. D. (2005). "On the categorization of demand
  patterns." *Journal of the Operational Research Society*, 56(5), 495–503.
  DOI: 10.1057/palgrave.jors.2601841. (A)
- Teunter, R. H., and Sani, B. (2009). "On the bias of Croston's forecasting method." *European
  Journal of Operational Research*, 194(1), 177–183. DOI: 10.1016/j.ejor.2007.12.001. (A)
- Teunter, R. H., Syntetos, A. A., and Babai, M. Z. (2011). "Intermittent demand: Linking
  forecasting to inventory obsolescence." *European Journal of Operational Research*, 214(3),
  606–615. DOI: 10.1016/j.ejor.2011.05.018. (A)
- Shale, E. A., Boylan, J. E., and Johnston, F. R. (2006). "Forecasting for intermittent demand: A
  comparative study of the accuracy of alternative approaches." *JORS*, 57(5), 588–592. (A; DOI
  not confirmed during preparation)
- Kostenko, A. V., and Hyndman, R. J. (2006). "A note on the categorization of demand patterns."
  *JORS*, 57(10), 1256–1257. DOI: 10.1057/palgrave.jors.2602211. (A)
- Johnston, F. R., and Boylan, J. E. (1996). "Forecasting for items with intermittent demand."
  *JORS*, 47(1), 113–121. (A)
- Shenstone, L., and Hyndman, R. J. (2005). "Stochastic models underlying Croston's method for
  intermittent demand forecasting." *Journal of Forecasting*, 24(4), 285–300.
  DOI: 10.1002/for.963. (A)
- Teunter, R. H., and Duncan, L. (2008). "Forecasting intermittent demand: A comparative study."
  *JORS*, 60(3), 321–329. DOI: 10.1057/palgrave.jors.2602569. (A)
- Petropoulos, F., Nikolopoulos, K., Spiliotis, E., and Assimakopoulos, V. (2013). "Empirical
  heuristics for improving intermittent demand forecasting." *Industrial Management & Data
  Systems*, 113(5). (A; DOI not confirmed during preparation)
- Syntetos, A. A., Babai, M. Z., and Gardner, E. S. (2015). "Forecasting intermittent inventory
  demands: Simple parametric methods vs. bootstrapping." *Journal of Business Research*, 68(8).
  (A; DOI not confirmed during preparation)
- Doszyń, M. (2019). "Intermittent demand forecasting in the Enterprise: Empirical verification."
  *Journal of Forecasting*, 38(5). DOI: 10.1002/for.2575. (A)
- Doszyń, M. (2020). "Accuracy of Intermittent Demand Forecasting Systems in the Enterprise."
  *European Research Studies Journal*, 23(2). DOI: 10.35808/ersj/1723. (A−)
- Babai, M. Z., Ali, M. M., and Nikolopoulos, K. (2012). "Impact of temporal aggregation on stock
  control performance of intermittent demand estimators." *Omega*, 40(6), 713–721. (A)
- Nikolopoulos, K., Syntetos, A. A., Boylan, J. E., Petropoulos, F., and Assimakopoulos, V. (2011).
  "An aggregate–disaggregate intermittent demand approach (ADIDA) to forecasting." *JORS*, 62(3),
  544–554. DOI: 10.1057/jors.2010.32. (A)
- Kourentzes, N., Rostami-Tabar, B., and Barrow, D. K. (2017). "Demand forecasting by temporal
  aggregation." *Journal of Business Research*, 78. DOI: 10.1016/j.jbusres.2017.04.016. (A)
- Rostami-Tabar, B., Goltsos, T. E., and Wang, S. (2023). "Forecasting for lead-time period by
  temporal aggregation." *Computers in Industry*, 145. DOI: 10.1016/j.compind.2022.103803. (A)
- Pennings, C. L. P., van Dalen, J., and van der Laan, E. A. (2016). "Exploiting elapsed time for
  managing intermittent demand for spare parts." *European Journal of Operational Research*.
  DOI: 10.1016/j.ejor.2016.09.017. (A)
- Syntetos, A. A., Babai, M. Z., and Altay, N. (2012). "On the demand distributions of spare
  parts." *International Journal of Production Research*, 50(8).
  DOI: 10.1080/00207543.2011.562561. (A)
- Thaiudomsap, T., and Sribundit, N. (2022). "Study on demand forecasting techniques for drugs with
  sporadic demand in a community hospital." *Science, Engineering and Health Studies*. (B)

**Books and teaching material**

- Hyndman, R. J., and Athanasopoulos, G. *Forecasting: Principles and Practice*, 3rd edition. Open
  access; the chapters on exponential smoothing and on count and intermittent data. (A−)
- Silver, E. A., Pyke, D. F., and Thomas, D. J. *Inventory and Production Management in Supply
  Chains*, 4th edition. (A−)
- *Forecasting: Theory and Practice*, open encyclopedia, the sections on intermittent demand and on
  forecast evaluation. https://forecasting-encyclopedia.com (B)

**Standards and official**

- GS1 General Specifications, current edition, including GTIN, GLN, SSCC, EPCIS 2.0, and Digital
  Link. (B)
- ISO 8000 series, *Data quality*. (B)
- U.S. Food and Drug Administration, 21 CFR Part 830, *Unique Device Identification System*, and
  GUDID. (B)
- Regulation (EU) 2017/745 on medical devices (MDR), Article 27, unique device identification. (B)

**Software and practitioner material**

- Hyndman, R. J., et al., `forecast` package for R, `croston` documentation and its `type`
  argument. (C)
- `tsintermittent` package for R, documentation of the Croston, SBA and TSB family. (C)
- Nixtla, `statsforecast` documentation, `CrostonSBA` model. (C)
- Chau, C. (2020). "Intermittent demand forecasting for inventory control: The impact of temporal and
  cross-sectional aggregation." MIT thesis. (C)

**Note on grading and two unverified items.** Grades follow the programme scheme: A peer-reviewed or
audited, B regulatory or official, C company case, thesis, or implementation documentation, D vendor marketing, E press
or anecdote; working papers and teaching texts are marked A−. Two items were not fully verifiable. The ADI and CV² cut-
offs of 1.32 and 0.49 are the values most commonly attributed to Syntetos, Boylan and Croston (2005); the framework and
automotive validation sample are confirmed, but the full text was unavailable to confirm the constants, which this file
therefore treats as conventions. The bias percentages attributed to Teunter and Sani (2009) are as reported in the
accessible summary and should be checked against the published article before being quoted in a policy. Neither affects
the qualitative argument.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`04_regression-lags-fourier.md`](04_regression-lags-fourier.md) — regression with
  calendar and causal features, and the leakage discipline assumed in section 5.
- Planned sibling: `05_hierarchy-aggregation.md` (planned) — where to forecast, and how aggregation
  changes error.
- Related: [`02_exponential-smoothing.md`](02_exponential-smoothing.md) — the level, trend, and
  seasonality recursions that fail on zero-heavy series.
- Related: [`01_forecastability-benchmarks.md`](01_forecastability-benchmarks.md) — the naive and
  seasonal-naive floors used as controls in section 6.6.
- Later in this module: `08_rolling-origin-backtesting.md` (planned) and
  `09_error-metrics-cost.md` (planned).
- Measurement prerequisite:
  [`../../01_foundations/01A_foundations_units_time_assumptions/10_intermittent-metrics.md`](../../01_foundations/01A_foundations_units_time_assumptions/10_intermittent-metrics.md)
  — ADI, CV², the four-class grid, and why percentage errors fail here.
- Distribution prerequisite:
  [`../../01_foundations/01A_foundations_units_time_assumptions/11_demand-lead-time-distributions.md`](../../01_foundations/01A_foundations_units_time_assumptions/11_demand-lead-time-distributions.md)
  — lead-time demand distributions and the critical ratio used in step 15.
- Next module: [`../08_hierarchical_intermittent_probabilistic_forecasting/_index.md`](../08_hierarchical_intermittent_probabilistic_forecasting/_index.md)
  — compound Poisson and hurdle models, predictive distributions, and the quantiles these methods
  cannot produce.
- Module 01A map: [`_index.md`](../../01_foundations/01A_foundations_units_time_assumptions/_index.md)
- Part I map: [`../../01_foundations/_index.md`](../../01_foundations/_index.md)
- Survey file: [`../../../course/07_time_series_forecasting.md`](../../../course/07_time_series_forecasting.md)
- Part map: [`../_index.md`](../_index.md)
