# Hierarchy and Aggregation — Where to Forecast and How Aggregation Changes Error

> Module 07 — Time Series Forecasting Foundations
> Topic 05 of 10
> Estimated study time: 3.5 hours
> Prerequisites: Module 01A topic 07 (hierarchy design); Module 04 topic 04 (regression with time-series structure); Module 07 topics 01–04
> Version 2026.5

---

## 1. Why this topic exists

A demand review opens with two numbers describing the same demand. The category forecast for
wound-care dressings is 94% accurate at the monthly, national level. Service level on the same
dressings at distribution-centre (DC) level is 71%. Both are computed correctly. Both are reported
to the same committee. Neither is lying, and the gap between them is this topic.

The gap exists because a forecast is made at a level and a decision is made at a level, and the two
are almost never the same. The category manager sees a stable, seasonal series of about 24,000 units
a month. The replenishment planner has to decide how many of SKU D4 to order for a week in which the
item sold 11 units, 0 units, 0 units, 4 units, and 19 units. Those are not two views of one problem.

Three facts make this structural.

**The decision level is statistically hopeless and operationally necessary.** A distributor with 11
stock-keeping units (SKUs), 9 DCs, and 104 weeks has 99 SKU–DC series, 46% of whose weekly
observations are zero. Fitting a model per series is not a hard problem a better algorithm solves;
it is a problem with too little information for the parameter count, and the honest baseline is a
naive or mean forecast. Yet the order is placed at SKU–DC–week, because that is where the shelf and
the truck are.

**Aggregation improves forecastability, but less than the textbook says.** Summing independent
series reduces relative variation roughly with the square root of their count. Demand series are
not independent: they share a season, a price list, a promotion calendar. The gain is bounded by
that correlation, and in most product hierarchies the correlation is high. Section 3.2 supplies the
arithmetic, which is the most useful formula in the topic.

**Disaggregation is not the inverse of aggregation.** Going up loses information about composition;
coming down requires inventing composition. The invention is a model whether or not anyone writes
it down. A historical-share allocation asserts the mix will repeat; a planner's manual spread
asserts the planner knows the mix. Neither assertion is free, and neither is usually tested.

The commercial consequences are specific. A team improves national weighted absolute percentage
error (WAPE) from 0.13 to 0.11 and reports success while SKU–DC service is unchanged, because the
improvement came from the high-volume items that dominate the aggregate denominator. A month forecast exactly still produces
a weekly plan with 15.4% error (section 6.7), because the monthly number knows nothing about the
intra-month profile. Finance plans at category × region × month and operations at SKU × DC × week;
the bottom-up sum is 6.7% below the top-down total, and the gap is closed by proration, which
preserves every relative error it was meant to correct. A new item with 3% of the family's history
and 12% of its current demand is given the history.

This topic sits after the benchmark, smoothing, ARIMA, and regression topics because you cannot ask
"where should I forecast?" until you can forecast at all. It precedes intermittent demand (Topic 06)
because intermittency is largely a consequence of the level you chose: the same product is smooth at
family level and lumpy at SKU–DC level. It precedes combination, backtesting, and error metrics
because each has to be evaluated at a level, and a metric computed at the wrong level flatters
whatever it measures. It forward-references Module 08, which turns the coherence requirement stated
here into estimators this topic deliberately does not teach.

> **Note:** Nothing here says "forecast at the aggregate and allocate down". The record in section 9
> supports no universal rule in either direction. It supports a procedure: measure at each level,
> test share stability, and choose on the evidence.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** decision grain from forecasting grain, and **state** both for a named planning
   process with their units and time buckets.
2. **Compute** aggregation gain from variances and pairwise correlations, and **explain** why the
   square-root heuristic overstates it.
3. **Compare** top-down, bottom-up, middle-out, and reconciled approaches on five stated
   determinants, and **recommend** one with a justification a backtest could falsify.
4. **Allocate** an aggregate forecast three ways, **show** the three differ, and **name** the
   assumption each encodes.
5. **Diagnose** a temporal aggregation mismatch, **compute** the error from dividing a monthly
   forecast into weeks, and **propose** a temporal hierarchy as the repair.
6. **Measure** error at three levels of one hierarchy and **decompose** the difference into
   cancellation, denominator weighting, and genuine differences in forecastability.
7. **Identify** a coherence gap between bottom-up and top-down totals, and **state** what proration
   does and does not fix.
8. **Assign** forecast ownership by level and **describe** how an override propagates, including
   when it does not.
9. **Write** a forecast-level register naming level, method, owner, bucket, and reconciliation rule
   for each planning stream.

Outcomes 2, 4, 5, and 6 are exercised with arithmetic in section 12 and answered in section 13.

## 3. The core mental model

Hold three pictures: a three-dimensional grid of product, location, and time; a variance argument
telling you what the grid's coarse cells are worth; and a set of arrows showing where the forecast
is made and where it is used.

### 3.1 The product–location–time grid

```text
                        PRODUCT                    LOCATION                  TIME
  level 0            Total company               All sites                 Year
                      /          \               /    |    \                 |
  level 1        Category A   Category B    Region N  Region C  Region S   Quarter
                  /      \        |            /   \      |        /  \        |
  level 2      Family A1  A2   Family B1     DC1  DC2   DC3      DC4  DC5    Month
                  |        |        |          |    |     |        |    |      |
  level 3       SKU       SKU      SKU       ship-to points             ISO week
                  +--------+--------+----------+----+-----+--------+----+------+
                                    Day

  One observation = one cell: (SKU, DC, week).
  A forecast is a claim about a set of cells at one level, for one future bucket.
  A decision consumes cells at one level. They are frequently different levels.
```

Three properties matter.

**The axes are not interchangeable in structure.** The product axis is a strict tree: one parent per
SKU. The location axis is a tree. But their cross is not: "Dressings in the North" is a child of
"Dressings" and of "North" simultaneously, which makes the structure **grouped** rather than
hierarchical, with no unique aggregation path. This is the `hts` versus `gts` distinction, and it is
why a two-axis planning grid cannot be reconciled by a single bottom-up pass without choosing a
path.

**The time axis does not nest cleanly.** Days nest into weeks; weeks do not nest into months. A month
contains four or five ISO weeks, and the fifth is usually shared with the neighbouring month.

**The level with the best signal is rarely the level with the decision.** Signal-to-noise improves
upward; decision specificity improves downward. The forecasting level is the compromise, made
explicitly rather than inherited from whichever table the extract landed in.

### 3.2 How much aggregation actually buys

Take `n` series with equal mean `mu`, equal standard deviation `sigma`, and constant pairwise
correlation `rho`. The sum has mean `n·mu` and variance:

```text
   Var(sum) = n * sigma^2 * (1 + (n - 1) * rho)

   CV(sum) / CV(item) = sqrt( (1 + (n - 1) * rho) / n )
```

Read the last line as the ratio of the sum's coefficient of variation (CV) to the item's, not as a
percentage. For `n = 10`:

| Correlation `rho` | `CV(sum) / CV(item)` | Interpretation |
|---:|---:|---|
| 0.0 | 0.316 | Removes 68% of the relative variation |
| 0.2 | 0.529 | Removes 47% |
| 0.5 | 0.742 | Removes 26% |
| 0.8 | 0.906 | Removes 9% |
| 1.0 | 1.000 | Removes nothing; the series are one series |

The `sqrt(n)` heuristic is the `rho = 0` row — the best case, which almost never occurs, because
items in a family share a season, a price change, a promotion calendar, and a network. At `rho =
0.5`, `n = 10`, the gain is a quarter of the relative variation, not two thirds.

Two consequences follow. **The aggregation gain is empirical, not a design assumption.** You cannot
compute it from the SKU count; you compute it from residual correlation after common seasonality is
removed, or, more practically, by forecasting at both levels on a backtest (section 5, step 4).
**And the gain is partly an artefact of cancellation.** Summed series offset positive and negative
errors, so the aggregate looks accurate because the errors that remain are common to all members,
and a single item's replenishment experiences its own error in full.

> **Note:** The formula describes series variation, and forecast error follows it only when errors
> are uncorrelated. A missed promotion or a competitor entry produces correlated errors at every
> level, which is why aggregate WAPE falls more slowly than the CV suggests, and sometimes not at
> all.

### 3.3 Four strategies on one pyramid

```text
        TOTAL  ----------------------------------------------
          |                                                 |
       FAMILY        (3) middle-out starts here            |
        /    \            and moves both ways               |
     SKU     SKU                                            |
      |       |                                             |
   DC-week  DC-week                                         |

  (1) TOP-DOWN     TOTAL -> FAMILY -> SKU -> DC-week   (forecast top, allocate by share)
  (2) BOTTOM-UP    DC-week -> SKU -> FAMILY -> TOTAL   (forecast leaves, sum upward)
  (3) MIDDLE-OUT   FAMILY -> {TOTAL, SKU, DC-week}     (both directions from the middle)
  (4) RECONCILE    forecast every level, adjust all simultaneously (Module 08)
```

The four are not a ranking. They are four answers to "where is the information?" Top-down trusts a
smooth aggregate and a stable mix. Bottom-up trusts each leaf's own history and dynamics.
Middle-out trusts an intermediate level that has both volume and specificity. Reconciliation trusts
all levels together. Each fails where its trust is misplaced: drifting shares, sparse leaves, an
unowned middle level, or an unestimable covariance.

### 3.4 Coherence, and where the identity breaks

```text
   coherent                          incoherent
   TOTAL 24,000                      TOTAL 24,000   <- top-down plan
     /      \                          /      \
  North    rest                     North    rest
  9,126   14,874                    8,900   15,100  <- bottom-up sum
     |                                 |
  SKUs sum to 9,126                 SKUs sum to 8,900

   every sum holds                   the two totals differ by 1,600 units
```

A set of forecasts is **coherent** when every parent equals the sum of its children at every level
and bucket. Coherence is arithmetic, not accuracy. A coherent set can be badly wrong and can be
worse than the incoherent forecasts it replaced. The incoherent case above is the normal state of a
company that plans top-down for finance and bottom-up for replenishment.

### 3.5 Why a monthly number cannot be divided into weeks

```text
   Mon 29 Dec 2025 .......................................
   |  W1  |  W2  |  W3  |  W4  |  W5  |  W6  |  W7  |  W8  |
   |<------------------ January, 31 days ------------->|
                              |<---- February, 28 days ---->|
                                        ^  W5 straddles the boundary

   52 weeks / 12 months = 4.333 weeks per month, on average.
   A given month contains 4 or 5 ISO weeks, and 28 to 31 days.
   Dividing by 4.333 misstates a month by -7.7% (4 weeks) to +15.4% (5 weeks),
   before any consideration of the intra-month profile.
```

A monthly forecast carries no information about *when* inside the month demand falls. Dividing it
assumes a flat profile, and the profile is never flat: weekends, delivery days, paydays, and
promotions all have within-month structure. The repair is not a better divisor; it is to forecast
at the finer bucket and reconcile upward, or to model the profile with its own uncertainty.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Hierarchy | A containment relation in which each node has one parent, used to aggregate measures | Applied to any grouping, including crossed ones | Level list plus parent relation |
| Grouped structure | Two or more crossed hierarchies with no unique parent path | Modelled as a tree | Summing matrix with several factors |
| Level | A named depth in a hierarchy, such as family or region | Assumed identical across systems | Ordinal integer, root = 0 |
| Bottom level | The finest level at which forecasts are produced or decisions taken | Confused with the finest level stored | Node key, maximum depth |
| Disaggregation | Allocating a forecast from a coarser to a finer level | Treated as the inverse of aggregation | Share vector, sums to 1 |
| Top-down | Forecast the total, allocate down by shares | Assumed unbiased | Share method plus forecast |
| Bottom-up | Forecast each leaf, sum upward | Assumed to preserve aggregate accuracy | Sum of leaf forecasts |
| Middle-out | Forecast an intermediate level, allocate down and sum up | Described without naming the middle level | Named intermediate level |
| Base forecast | An independent forecast at one level, before reconciliation | A "raw" forecast, at an unnamed level | Units per bucket per node |
| Coherent forecast | A set in which every parent equals the sum of its children | Confused with an accurate forecast | Boolean per node per bucket |
| Reconciliation | Adjusting base forecasts so coherence holds, usually by weighted combination | Used as a synonym for proration | Estimator name plus weights |
| Minimum trace (MinT) | A reconciliation estimator minimising the trace of the forecast-error covariance | Treated as a default that always wins | Estimator variant; Module 08 |
| Summing matrix | The matrix mapping bottom series to every level of the structure | Treated as a data table rather than a constraint | Rows = all series, columns = bottom series |
| Proportion | A member's share of its parent's demand | Averaged without stating the weighting | Ratio, 0 to 1 |
| Average of proportions | The mean, over periods, of each period's share | Confused with proportion of averages | Ratio, 0 to 1 |
| Proportion of averages | The member's total divided by the parent's total | Confused with average of proportions | Ratio, 0 to 1 |
| Forecast proportion | A share computed from base forecasts rather than from history | Assumed to require a reconciled forecast | Ratio, 0 to 1 |
| Share drift | The tendency of a member's share to change over time | Assumed absent | Change in share per period |
| Allocation key | The driver used to distribute a pool or a forecast | Chosen for convenience | Quantity with unit |
| Proration | Scaling every member by one factor to close a gap | Presented as reconciliation | Multiplicative factor |
| Aggregation gain | The reduction in relative forecast error from forecasting a sum | Quoted as a fixed percentage | Ratio of aggregate error to item error |
| Disaggregation cost | The error added by allocating a forecast to a finer level | Ignored in top-down business cases | Ratio of allocated error to direct error |
| Error cancellation | Offsetting of positive and negative errors when series are summed | Mistaken for accuracy | Signed versus absolute error |
| Specification error | Error from modelling heterogeneous members as one series | Confused with sampling error | Variance component |
| Sampling error | Error from estimating parameters on too little data | Confused with specification error | Variance component |
| Temporal aggregation | Summing a series into coarser time buckets | Assumed lossless | Bucket length in days |
| Temporal hierarchy | One series held at several frequencies simultaneously | Confused with a cross-sectional hierarchy | Frequency list |
| Cross-temporal reconciliation | Reconciliation across structure and time frequencies at once | Handled as two separate problems | Estimator plus structure |
| Bucket nesting | The property that finer buckets sum exactly into coarser ones | Assumed for weeks and months | Boolean per calendar |
| Planning grain | The level and bucket at which a plan is produced | Used interchangeably with decision grain | Level plus bucket |
| Decision grain | The level and bucket at which a commitment is made | Left unstated | Level plus bucket |
| Weighted absolute percentage error (WAPE) | Sum of absolute errors divided by sum of actuals | Compared across levels without stating the denominator | Ratio, unit-free |
| Mean absolute scaled error (MASE) | Mean absolute error over the in-sample naive mean absolute error | Denominator taken from the evaluation window | Ratio, unit-free |
| Root mean squared scaled error (RMSSE) | The squared-error analogue of MASE | Confused with MASE | Ratio, unit-free |
| Weighted RMSSE (WRMSSE) | RMSSE weighted by each series' share of value, as in the M5 competition | Reported without the weighting scheme | Ratio, unit-free |
| Forecast ownership | The role accountable for the forecast at a stated level | Assigned to a department rather than a role | Role plus level |
| Override cascade | The rule governing how an adjustment at one level propagates | Assumed to propagate automatically | Rule plus direction |
| Single source of truth | One authoritative value for a measure at a stated grain | Used to mean one system | Measure plus grain plus owner |
| New-item cold start | Forecasting a member with no usable history | Handled by giving the item a zero share | Observation count |
| Substitution | Demand moving between members in response to price, promotion, or availability | Ignored in share-based allocation | Cross-elasticity or transfer ratio |

Four rows repay expansion.

**Average of proportions versus proportion of averages.** Two different operations that both produce
a vector summing to one, and they diverge whenever the mix drifts. The first gives every period
equal weight; the second gives every unit equal weight. In a growing region the second is higher,
because the high-volume months are the later ones. Neither is correct in the abstract; the choice
states which weighting you believe.

**Base versus coherent forecast.** A base forecast is what a method produces at a level from that
level's history. A coherent forecast is what survives after the levels are forced to agree. Most
reported "accuracy" is base-forecast accuracy; most systems consume coherent forecasts. Comparing
the two is the commonest way a reconciliation project claims a benefit it did not deliver.

**Proration versus reconciliation.** Proration multiplies every member by the same factor, changing
no relative position. Reconciliation distributes the gap by estimated error covariance, so members
with more reliable base forecasts move less. The first is arithmetic; the second is a model.

**Sampling versus specification error.** Zotteri and Kalchschmidt frame the choice between them.
Aggregating too far produces specification error, modelling heterogeneous members as one series;
aggregating too little produces sampling error, too few observations to estimate anything. The best
level minimises their sum, and it moves with variability, member count, heterogeneity, and history
length.

## 5. The mechanics, step by step

Each step names its input, output, and the branch where the analysis can legitimately stop.

### Phase A — Locate the decision and test the data

**Step 1. Name the decision, its owner, and its grain.** Write one sentence fixing the decision
grain: "The DC replenishment planner orders SKU D4 for DC North every Monday, covering a three-week
lead time, in eaches." Write a second fixing the planning grain: "The category demand planner
produces a monthly family × region forecast for sales and operations planning (S&OP), in eaches."
*Input:* process description and executing system. *Output:* two grains in writing, with units.
*Branch:* if they are identical, skip to Phase D. The distinction is purpose, not level: the
planning grain supports capacity, budget, and mix decisions, the decision grain supports a
commitment, and forcing them into one grain damages both.

**Step 2. Measure the data at the decision grain.**

| Measure | Definition | Threshold that should worry you | Unit |
|---|---|---|---|
| Observations per series | Non-missing periods in the estimation window | Below 52 weekly observations | Count |
| Zero share | Fraction of periods with zero demand | Above 0.40 | Ratio |
| Average inter-demand interval (ADI) | Mean periods between non-zero demands | Above 1.32 | Periods |
| CV² of non-zero sizes | Variance of non-zero demand over the square of its mean | Above 0.49 | Ratio |

The ADI and CV² thresholds come from the Syntetos–Boylan–Croston classification and are
conventions, not laws, but a portfolio above both on most series is one where per-series models
lose to simple benchmarks; Topic 06 treats the classification in depth. *Output:* four
distributions, not four averages, because the average hides the tail. *Branch:* if more than half
the series fail two or more thresholds, proceed to step 3 rather than fitting models.

**Step 3. Measure the data at candidate planning levels.**

| Level | Series | Median observations | Zero share | Median ADI | Typical model |
|---|---:|---:|---:|---:|---|
| SKU × DC × week | 99 | 31 non-zero | 0.46 | 2.1 | Benchmark or intermittent method |
| SKU × region × month | 44 | 12 | 0.18 | 1.3 | Exponential smoothing |
| Family × region × month | 12 | 12 | 0.00 | 1.0 | Seasonal model with regression |
| Category × month | 1 | 24 | 0.00 | 1.0 | Seasonal model |

The table is the argument in miniature: at the bottom the median series has 31 usable observations
and an ADI of 2.1, intermittent by any definition; at family × region there are no zeros.

> **Note:** Observations per series is not history length. Twelve monthly observations is one year,
> which cannot separate seasonality from trend. Judge a level on complete seasonal cycles.

**Step 4. Measure the aggregation gain empirically.** Take a backtest window, forecast at both
levels with the best method available at each, aggregate the bottom forecasts to the candidate
level, and compute `1 - (error of the summed bottom forecasts / error of the direct aggregate
forecast)`. A positive value means the parts carry information the aggregate loses; a negative value
means the aggregate is better, the usual case when the parts are noisy. Report the gain per level
and horizon, because it shrinks with horizon. *Branch:* if the gain is inside the backtest's noise,
the cheaper level wins; do not pay for a hierarchy you cannot demonstrate.

**Step 5. Test share stability.** For each member compute its share of its parent in every period.

- **Range**: maximum minus minimum share. A range wider than 0.10 on a member averaging 0.15 is a
  warning.
- **Trend**: the slope of share against time. A significant slope means the historical average is
  stale by construction.
- **Persistence**: correlation between this period's share and the same member's share a year
  earlier. Low persistence means the share is mostly noise.

*Branch:* if fewer than 70% of members are stable on all three, top-down with historical shares is
not defensible.

**Step 6. Choose and record the forecasting level.** Record level, bucket, method family, share
method if any, and the backtest evidence, because the choice will be challenged in a meeting where
nobody remembers the analysis.

### Phase B — Choose the strategy

**Step 7. Enumerate the candidates honestly.** Four, not two. Most organisations compare top-down
with bottom-up and never test the two middle options.

| Strategy | What must be estimated | Strong | Weak |
|---|---|---|---|
| Top-down | A share vector per parent | Aggregate is smooth; few parameters | Shares drift; new items starved; biased |
| Bottom-up | One model per leaf | No share model; preserves item dynamics | Leaf noise accumulates; weak at aggregates |
| Middle-out | Both, around a chosen level | Balances specificity and volume | Needs a defensible middle level and owner |
| Reconciliation | Base forecasts at every level plus weights | Uses all information; coherent by construction | Needs covariance or a proxy; can be unstable |

**Step 8. Score the candidates on five determinants.**

| Determinant | Measurement | Favours |
|---|---|---|
| Bottom-level noise and intermittency | Zero share, ADI, CV² | Top-down or middle-out when high |
| Cross-member correlation after deseasonalising | Mean pairwise residual correlation | Bottom-up when low; top-down adds little when high |
| Share stability | Range and persistence of shares | Top-down when stable; forecast shares when not |
| History length per series | Complete seasonal cycles at the leaf | Bottom-up when long; aggregation when short |
| Evaluation level and metric | Where error is scored, and how weighted | Whatever the decision consumes |

The last determinant produces most of the apparent contradictions in the literature: a study scoring
at the bottom level often finds bottom-up winning, and a study weighting all levels equally often
finds reconciliation winning. Both can be correct measurements of different questions.

**Step 9. Pick a primary strategy and a stated fallback.** A fallback is not indecision: it is the
answer to "what do we do when the primary fails its backtest", written before the backtest runs.

### Phase C — Disaggregate

**Step 10. Choose the share method.**

| Method | Formula (member `j`, periods `t`) | Assumption encoded | Behaviour under drift |
|---|---|---|---|
| Average of proportions | `p_j = (1/T) * SUM_t ( y_jt / y_t )` | Every period is equally informative about the mix | Trails a growing member; leads a declining one |
| Proportion of averages | `p_j = (SUM_t y_jt) / (SUM_t y_t)` | Every unit is equally informative | Follows volume, so tracks a growing member faster |
| Forecast proportions | `p_j` computed recursively from base forecasts at each level | The mix will follow the forecast, not the history | Responds to promotions and launches |

The third matters when the future mix is known to differ from the past, and it requires base
forecasts at every level, which makes it half a reconciliation.

> **Note:** No top-down method is unbiased in general. The share vector is estimated from data that
> already contains the parent's history, so shares and parent forecast are not independent, and the
> result cannot satisfy the unbiasedness condition bottom-up satisfies by construction.

**Step 11. Handle new, discontinued, and promoted members explicitly.**

- **New member.** Near-zero historical share, so top-down starves it. Assign a launch share from
  the category plan, or forecast it bottom-up and reconcile the rest around it.
- **Discontinued member.** Remove it from the share base at the discontinuation date, not at the end
  of the history, or it inflates its share after demand stops.
- **Promoted member.** Use forecast proportions where a promotion calendar exists, and test the
  uplift rather than assuming the baseline share.

**Step 12. Round so the parts sum to the whole.** Floor every allocation, compute the residual
against the parent, then distribute it one unit at a time to the largest fractional parts. The
result sums exactly and distorts no member by more than one unit.

**Step 13. Publish the shares, not just the allocation.** A reader who sees only the allocated
number cannot tell whether it came from history, a forecast, or a planner's judgement. Publish the
parent forecast, the share vector, the method, and the rounding residual.

### Phase D — Handle time

**Step 14. Fix the temporal hierarchy before the forecast.** List the buckets the business uses and
note which nest: days into weeks, months into quarters and years, but not weeks into months. A 4-4-5
retail calendar makes weeks nest into periods by construction, at the cost of a 53-week year every
five or six years.

**Step 15. Test the nesting assumption on real dates.** Count, for each month, the whole ISO weeks
it contains and the days outside a whole week. January holds four weeks and three stray days,
February four and none, March four and three. That variation is the error term a flat divisor
ignores.

**Step 16. Decide whether to forecast fine or reconcile fine.** **Forecast fine, aggregate up**
preserves intra-month information and makes the monthly number a consequence. **Forecast coarse,
reconcile down** is cheaper and right when the weekly decision tolerates profile error. What is not
defensible is dividing the monthly number by a constant.

### Phase E — Reconcile across the structure

**Step 17. Measure the coherence gap.** Sum the bottom forecasts, sum the top forecast, and report
the difference in units, as a ratio to the top-down total, and by node, because the gap is usually
concentrated in one family or region.

**Step 18. Choose a reconciliation approach, and say which one.** The options run from proration (a
scalar), through ordinary and weighted least squares, to minimum-trace estimators using the
estimated forecast-error covariance. This topic stops here deliberately: the estimators, their
assumptions, and their behaviour with zeros are Module 08. What you need now is the vocabulary to
specify the requirement — coherent, unbiased where possible, with a stated loss.

> **Note:** Reconciliation guarantees coherence, not accuracy. A reconciled set can be worse than
> the base forecasts at every level if the base forecasts are biased, because the adjustment spreads
> the bias rather than removing it. Test reconciled against base at every level.

### Phase F — Govern

**Step 19. Assign ownership by level.** Own the level where your action changes the number. A
central demand planner owns family × region × month because they influence promotion timing and
pricing; a DC planner owns SKU–DC–week replenishment because they influence allocation and
expediting. Nobody owns a level they cannot affect, and a level with no owner degrades silently.

**Step 20. Define the override cascade.**

| Cascade rule | Effect | Consequence |
|---|---|---|
| Local only | The node changes and nothing else | The set becomes incoherent |
| Proportional | The change is spread across siblings so the parent is unchanged | The planner's intent is diluted |
| Parent-propagating | The change flows up; siblings are unchanged | Finance sees a different total |
| Reconciled | The change is applied and the set is re-reconciled | Coherent, but the number is adjusted again |

Most systems implement "local only" by default and most planners assume "parent-propagating". The
gap between those assumptions produces the meeting where two people report different totals for the
same week.

**Step 21. Publish the forecast-level register.** One page, one row per planning stream.

| Field | Content | Example |
|---|---|---|
| Stream | The decision served | DC replenishment |
| Level and bucket | Where the forecast is produced | SKU × DC × week |
| Method family | What produces the base forecast | Intermittent method with seasonal profile |
| Share method | How it is disaggregated | Not applicable |
| Owner | The accountable role | DC replenishment planner, North |
| Horizon | Steps produced | 1 to 8 weeks |
| Reconciliation | How coherence is enforced | Bottom-up to region, reconciled monthly |
| Accuracy target | Metric, level, threshold | WAPE at SKU–DC–week below 0.45 |

The register ends the argument about "which forecast is right". There is no single right forecast;
there is a register saying which forecast serves which decision, and owned by whom.

## 6. Worked example

Aldergrove Medical distributes wound-care consumables through four regions and nine DCs. It plans
at family × region × month and replenishes at SKU × DC × week. The figures are constructed for
teaching and are internally consistent; every total reconciles unless the text says otherwise.

### 6.1 The structure

| Axis | Levels | Members |
|---|---|---|
| Product | Total, category, family, SKU | 3 families, 11 SKUs |
| Location | Total, region, DC, ship-to | 4 regions, 9 DCs |
| Time | Year, quarter, month, ISO week, day | 104 weeks of history |

The planning grid is 3 families × 4 regions = 12 cells per month. The decision grid is 11 SKUs × 9
DCs = 99 series per week. Twelve planning cells against 99 decision series is the whole problem in
one line.

### 6.2 Six months of region demand for the dressings family

| Month | North | Central | South | East | Total |
|---|---:|---:|---:|---:|---:|
| Jul | 900 | 600 | 420 | 180 | 2,100 |
| Aug | 880 | 610 | 430 | 210 | 2,130 |
| Sep | 850 | 620 | 440 | 250 | 2,160 |
| Oct | 820 | 630 | 450 | 300 | 2,200 |
| Nov | 790 | 640 | 460 | 360 | 2,250 |
| Dec | 760 | 650 | 470 | 430 | 2,310 |
| **Total** | **5,000** | **3,750** | **2,670** | **1,730** | **13,150** |

East grows about 19% a month over the six months; North declines about 3% a month. The total rises,
so the mix shift is masked in the total and visible in the shares.

### 6.3 Two historical-share methods, computed by hand

| Region | Total (units) | Proportion of averages | Average of monthly proportions |
|---|---:|---:|---:|
| North | 5,000 | 0.38023 | 0.38135 |
| Central | 3,750 | 0.28517 | 0.28522 |
| South | 2,670 | 0.20304 | 0.20301 |
| East | 1,730 | 0.13156 | 0.13043 |
| **Sum** | **13,150** | **1.00000** | **1.00001** |

The proportion of averages is the column total over the grand total. The average of monthly
proportions is the mean of six monthly shares; for East:

```text
   180/2,100 = 0.08571     210/2,130 = 0.09859     250/2,160 = 0.11574
   300/2,200 = 0.13636     360/2,250 = 0.16000     430/2,310 = 0.18615
   mean = 0.78255 / 6 = 0.13043
```

The second method gives East a share 0.86% lower (0.13043 against 0.13156), because it gives
low-volume July the same weight as high-volume December. The direction is what drift predicts. The
magnitude is small here because six months of a steady ramp is a mild case; with a launch or a step
change the two methods diverge by far more.

### 6.4 Allocating the family forecast

The national dressings forecast for month M is 24,000 units.

| Region | Share (proportion of averages) | Allocation | Share (average of proportions) | Allocation | Difference (units) |
|---|---:|---:|---:|---:|---:|
| North | 0.38023 | 9,126 | 0.38135 | 9,153 | +27 |
| Central | 0.28517 | 6,844 | 0.28522 | 6,845 | +1 |
| South | 0.20304 | 4,873 | 0.20301 | 4,872 | −1 |
| East | 0.13156 | 3,157 | 0.13043 | 3,130 | −27 |
| **Total** | **1.00000** | **24,000** | **1.00001** | **24,000** | **0** |

The second allocation needed one rounding unit: the floors summed to 23,999 and the largest
fractional remainder belonged to North (0.40). Both columns sum exactly to 24,000. The 27-unit
difference on East is 0.86% of its allocation, small here and material when the drifted region is
large.

### 6.5 SKU shares inside a region, and the promoted item

North's 9,126 units split across four dressing SKUs. Twelve months of history gives these shares,
and that history does not contain the promotion scheduled for month M.

| SKU | Historical share | Allocation at 9,126 | Planned share in month M | Allocation at 9,126 |
|---|---:|---:|---:|---:|
| D1 gauze | 0.42 | 3,833 | 0.36 | 3,285 |
| D2 foam | 0.28 | 2,555 | 0.26 | 2,373 |
| D3 post-op | 0.18 | 1,643 | 0.16 | 1,460 |
| D4 island | 0.12 | 1,095 | 0.22 | 2,008 |
| **Total** | **1.00** | **9,126** | **1.00** | **9,126** |

D4 is promoted in month M and its share roughly doubles. A top-down allocation on history gives it
1,095 units against a requirement of 2,008, a shortfall of 913 units, or 45% of its requirement. The
other three SKUs are collectively over-allocated by the same amount. The aggregate forecast is
untouched and the family total is exactly right. Every unit of error in this table is created by the
allocation, not by the forecast.

### 6.6 Error at three levels for the same demand

| Level | Series | Median observations | Zero share | WAPE | Signed bias | WAPE ratio to top level |
|---|---:|---:|---:|---:|---:|---:|
| Family × region × month | 12 | 12 | 0.00 | 0.11 | +0.008 | 1.0 |
| SKU × region × month | 44 | 12 | 0.18 | 0.29 | +0.004 | 2.6 |
| SKU × DC × week | 99 | 31 non-zero | 0.46 | 0.52 | −0.019 | 4.7 |

Signed bias is the sum of signed errors over the sum of actuals. Four readings matter.

**The ratio is 4.7.** SKU–DC–week WAPE is 4.7 times family–region–month WAPE on the same demand,
and that ratio is the price of the decision grain: the decision is made where information is
thinnest.

**Bias nearly vanishes at the top and is uneven at the bottom.** The family-level bias of +0.008
looks unbiased, but 31% of SKU–DC series carry a bias larger than 0.20 in one direction. The
aggregate bias is small because item biases cancel, and cancellation is not correction. That is
also the aggregation gain: the absolute item errors are much larger than their net, and a single
item's replenishment experiences its own error in full.

**The zero share explains most of the ratio.** At 0.46 zero weeks, nearly half of all SKU–DC
observations carry no level information, and the forecast is effectively deciding whether the next
non-zero event is this week or next. No method resolves that from 31 non-zero observations.

> **Note:** A common and wrong inference is "therefore forecast at family level". The table says
> the family level is more accurate *for the family total*. It says nothing about whether a family
> forecast, allocated down, beats a direct SKU–DC forecast at SKU–DC level. That is a separate
> measurement, and section 9 shows it goes both ways.

### 6.7 The temporal split

North's allocation for month M is 9,126 units over five ISO weeks.

| Week | Actual | Plan at 1,825.2 (÷5) | Error | Plan at 2,106.2 (÷4.333) | Error |
|---|---:|---:|---:|---:|---:|
| 1 | 2,010 | 1,825.2 | +184.8 | 2,106.2 | −96.2 |
| 2 | 2,240 | 1,825.2 | +414.8 | 2,106.2 | +133.8 |
| 3 | 1,930 | 1,825.2 | +104.8 | 2,106.2 | −176.2 |
| 4 | 1,760 | 1,825.2 | −65.2 | 2,106.2 | −346.2 |
| 5 | 1,186 | 1,825.2 | −639.2 | 2,106.2 | −920.2 |
| **Sum** | **9,126** | **9,126** | **1,408.8** | **10,531** | **1,672.6** |

**Dividing by 4.333 is not even internally consistent.** It produces a weekly plan summing to
10,531 units for a month forecast at 9,126, an overstatement of 1,405 units or 15.4%. The factor is
exactly `5 / 4.333 = 1.154`. A planner who divides by 4.333 in a five-week month orders 15% more
than the month needs and then spends the last week cancelling.

**Even the correct divisor leaves 15.4% weekly WAPE.** With a perfect monthly total and the right
number of weeks, the flat profile still misses by 1,408.8 units, because week 5 contains a public
holiday and runs 35% below the average. The monthly number cannot know that. Only a weekly forecast,
or a weekly profile with its own calendar terms, can.

### 6.8 The coherence gap

| Quantity | Value | Unit |
|---|---:|---|
| Top-down family forecast | 24,000 | units |
| Sum of bottom-up SKU–DC forecasts | 22,400 | units |
| Gap | 1,600 | units |
| Gap as a ratio of the top-down total | 0.0667 | ratio |
| Proration factor required to close it | 1.0714 | ratio |

Proration multiplies every SKU–DC forecast by 1.0714. The item that was 40% low is still 40% low
relative to its siblings, and the item that was 30% high is still 30% high. The gap closes and the
error structure is untouched. That is why proration is popular (one line of arithmetic, and the
meeting ends) and why it is not reconciliation.

## 7. Code, query, or calculation

One block, hand arithmetic, because the point is the identity rather than the tooling. Run it on a
calculator and check every line against section 6.

```text
Inputs: family forecast F = 24,000 units
        region totals over six months: N 5,000  C 3,750  S 2,670  E 1,730   (sum 13,150)

1. share_prop = region_total / 13,150
     N 5,000/13,150 = 0.380228    C 3,750/13,150 = 0.285171
     S 2,670/13,150 = 0.203042    E 1,730/13,150 = 0.131559     sum 1.000000

2. allocation = F * share_prop, then largest-remainder rounding to whole units
     N 24,000*0.380228 =  9,125.5  -> 9,126
     C 24,000*0.285171 =  6,844.1  -> 6,844
     S 24,000*0.203042 =  4,873.0  -> 4,873
     E 24,000*0.131559 =  3,157.4  -> 3,157        sum 24,000, no adjustment

3. coherence check on the rounded allocation
     9,126/24,000 = 0.38025 against a target of 0.38023   (ratio 1.00005)
     sum of allocations - F = 0

4. weekly split of the North allocation in a five-week month
     divide by 5      : 9,126/5     = 1,825.2 per week
     divide by 4.333  : 9,126/4.333 = 2,106.2 per week   (2,106.2/1,825.2 = 1.154)
     actual weeks     : 2,010  2,240  1,930  1,760  1,186  (sum 9,126)
     WAPE vs 1,825.2  : 1,408.8/9,126 = 0.154
     WAPE vs 2,106.2  : 1,672.6/9,126 = 0.183   (on an overstated plan)
```

Expected output: shares summing to 1.000000, allocations summing to 24,000, and two weekly WAPEs of
0.154 and 0.183. The divisor `4.333` is `52/12`, an average that applies to no individual month; the
WAPE denominator is the sum of actuals, so an over-forecasting process is penalised once in the
numerator and not at all in the denominator; and the rounding must be largest remainder.

## 8. Data, standards, and regulatory context

No international standard defines a planning hierarchy, a forecasting level, a reconciliation
method, or a forecast ownership model. Analysts sometimes cite a product classification standard as
if it governed planning. It does not. What exists is a set of identifier standards, a calendar
standard, and a body of practice applied by analogy.

### 8.1 What the standards actually fix

| Standard or framework | Governs | What it fixes | What it does not fix | Grade |
|---|---|---|---|---|
| GS1 Global Product Classification (GPC) | Product classification | Four levels: segment, family, class, brick | The commercial or planning tree | B |
| GS1 Global Trade Item Number (GTIN) | Trade item identity | One identifier per trade item per packaging level | Which GTIN is the planning unit | B |
| GS1 Global Location Number (GLN) | Party and place identity | One identifier per physical or legal location | Which GLN is the planning location | B |
| ISO 8601 (International Organization for Standardization) | Date and time representation | Week numbering, ordering, durations | Which week an observation belongs to | B |
| National Retail Federation (NRF) 4-4-5 retail calendar | Retail period structure | Weeks nest into periods by construction | Month-based financial reporting | C |
| ASCM/APICS dictionary (Association for Supply Chain Management) | Planning terminology | Definitions of aggregate planning, S&OP | Any level, method, or ownership rule | C |
| SCOR framework (Supply Chain Operations Reference, ASCM) | Process reference model | Plan, source, make, deliver, return | Data model or forecast granularity | C |

**GPC is a classification, not a planning hierarchy.** It is mandatory in the Global Data
Synchronisation Network, which makes it tempting to reuse. But a manufacturer's planning families
are built around substitutability and changeover, and a retailer's around shelf adjacency. Neither
matches GPC's brick structure, and neither should.

**GTIN granularity changes the hierarchy.** A case of 24 and the individual unit are different
GTINs. If the planning hierarchy mixes them, the rollup double counts by a factor of 24. Module 01A
topic 01 treats the unit-of-measure machinery; the narrow point here is that the hierarchy is
additive only if every level is expressed in the same unit.

**ISO 8601 weeks do not nest into months.** That is not a defect in the standard but a property of
the calendar. Any temporal hierarchy spanning weeks and months needs an explicit rule for the
straddling week, and the rule is a convention the business chooses.

### 8.2 The calendars that determine the temporal hierarchy

| Calendar | Structure | Where used | Consequence for forecasting |
|---|---|---|---|
| Gregorian month | 28 to 31 days | Finance, most enterprise resource planning (ERP) systems | Month length varies by 10.7% between February and March |
| ISO week | 7 days, Monday start, 52 or 53 per year | Logistics, manufacturing | Weeks do not align to months |
| 4-4-5 retail | 4, 4, 5 weeks per quarter | Retail, consumer goods | Weeks nest into periods; 53-week year periodically |
| 4-5-4 retail | Variant of the above | Some retailers | Same nesting, different boundaries |
| Fiscal year | Arbitrary start date | Corporate reporting | May align to neither calendar |

A seasonal model estimated on one calendar and applied on another carries a definitional error no
method removes. A monthly model on Gregorian months sees a February 10% shorter than January and
must either learn that or receive a trading-day regressor. A weekly model on ISO weeks sees a
53-week year once every five or six years and will mishandle the extra week unless the calendar is
a feature.

### 8.3 Where regulation does reach the hierarchy

- **Financial reporting.** If the forecast feeds revenue guidance or inventory valuation, the
  process is in scope for internal control over financial reporting in listed companies. The
  control governs who may change the number and how it is documented, not the statistical method.
  It does constrain the hierarchy, because the plan must roll up to the reporting segments.
- **Transfer pricing.** Where an allocation moves cost or margin between legal entities, the
  allocation key is a tax-relevant choice. A key chosen for planning convenience can become an
  intercompany pricing position, and the two are governed differently.
- **Sector traceability.** Serialisation regimes in pharmaceuticals create item-level data finer
  than any planning hierarchy, and in some cases the serial number is the only reliable aggregation
  key. That does not change the forecasting level, but it changes what is joinable. Module 01 topic
  08 covers traceability standards.

> **Note:** "One number" plans are sometimes justified by audit. The audit requirement is that the
> number in the financial statement is controlled and traceable, not that the operational forecast
> and the financial plan are identical. They serve different decisions and can legitimately differ,
> provided the bridge is documented.

## 9. Industry practice and case evidence

### 9.1 The history of the argument

The debate is older than supply chain analytics and has changed its mind more than once.

| Period | Contribution | What it established | Grade |
|---|---|---|---|
| 1980s | Schwarzkopf, Tersine and Morris compare top-down and bottom-up strategies | Each has conditions where it is preferred; neither dominates | A |
| 1990 | Gross and Sohl compare disaggregation methods for product-line forecasting | Historical-proportion methods perform respectably; method choice matters | A |
| 1992 | Fliedner and Mabert test group size and grouping criterion at Cummins Engine | Grouping decisions change accuracy; implementation guidelines were missing | A |
| 2001 | Fliedner reviews hierarchical forecasting and issues use guidelines | Much remains unknown about how level and grouping affect performance | A |
| 2005 | Zotteri, Kalchschmidt and Caniato test aggregation on grocery store–SKU–day data | Both aggregating and disaggregating can improve accuracy; the best level depends on the process | A |
| 2007 | Zotteri and Kalchschmidt model the choice of aggregation level | The trade is specification versus sampling error | A |
| 2009 | Athanasopoulos, Ahmed and Hyndman introduce forecast proportions | Forecast proportions beat historical proportions; historical proportions worst below the top level | A |
| 2011 | Hyndman, Ahmed, Athanasopoulos and Shang introduce optimal combination | Forecasting every level and reconciling gives unbiased, minimum-variance forecasts under stated assumptions | A |
| 2017 | Athanasopoulos, Hyndman, Kourentzes and Petropoulos introduce temporal hierarchies | Reconciled forecasts beat base forecasts at every frequency; gains largest at the coarsest level | A |
| 2019 | Oliveira and Ramos test a Portuguese retailer with 1,751 series | Minimum-trace reconciliation beat base forecasts at all seven levels by 1.7% to 3.7% | A |
| 2021 | Mirčetić and colleagues test a European brewery | Combinations more accurate and less biased than any single approach | A |
| 2022 | Babai, Boylan and Rostami-Tabar review aggregation and hierarchy | The most accurate forecasts are not always at the natural level | A |
| 2022 | The M5 competition reports results across 12 aggregation levels | Gains shrink from roughly half at the total to about 3% at product–store level | A |
| 2024 | Silveira Netto, Brei and Hyndman compare structures on B2B component sales | Bottom-up won in most nodes of a grouped structure; minimum trace did not improve most cases | A |

The arc is not "simple methods lost". It is that the question moved. The 1980s and 1990s asked
which strategy is better, the 2000s asked under what conditions, and the 2010s and 2020s asked
whether all of them can be had at once. The answer to the third is "sometimes, at a cost in
estimation", which is why the older literature has not been superseded.

### 9.2 The apparent contradiction

The timeline contains one: a study finds bottom-up winning, another finds top-down with forecast
proportions winning, a third finds reconciliation winning. All are peer-reviewed work on real data,
and resolving the contradiction is the most important thing in this topic. Two further data points
sharpen it. Oliveira and Ramos tested 1,751 series across seven levels and found minimum-trace
reconciliation best at every level, with bottom-up poor at aggregates. Silveira Netto, Brei and
Hyndman tested B2B electrical-component sales with 18 base models and seven coherent methods, and
found bottom-up winning in most nodes of a grouped structure, with minimum trace failing to improve
most cases. Same question, opposite answers, both credible.

### 9.3 Why the answer varies

Six quantities explain most of the between-study variation, and each is measurable on your own data
before you choose.

**Bottom-level noise and intermittency.** Sparse leaves make bottom-up a sum of noise, and the
aggregate inherits it; Silveira Netto and colleagues report bottom-up degrading as disaggregation
and noise increase. Dense leaves preserve dynamics the aggregate cannot see.

**Correlation across members.** If members move together, aggregation buys little (section 3.2). If
they are negatively correlated, as substitutes are, it buys a great deal.

**Share stability.** Oliveira and Ramos report that top-down never produces unbiased coherent
forecasts, and the bias is largest where the mix moves. Where the mix is stable, the bias is small
and the variance advantage dominates.

**History length.** Zotteri and Kalchschmidt show the optimal level moving with member count,
heterogeneity, and history length: short history at the leaf favours aggregation.

**Evaluation level and weighting.** Scoring at the bottom level usually finds bottom-up winning;
weighting all levels equally usually finds reconciliation winning. M5 weighted all 12 levels equally
and its overall winner ranked 11th at the bottom, the cleanest demonstration that "best" is
level-dependent.

**Bias in the base forecasts.** Reconciliation distributes bias rather than removing it: if the
promotion calendar was not loaded, every level becomes slightly wrong instead of one level badly
wrong.

### 9.4 What companies do, versus what the literature recommends

| Recommendation | Observed practice | Gap and why it persists |
|---|---|---|
| Test more than two strategies | Top-down and bottom-up only | Reconciliation needs software and a covariance estimate |
| Measure aggregation gain before choosing | Level chosen by system convention | The ERP module is configured at one level |
| Use forecast proportions where events are known | Historical shares, refreshed annually | Forecast proportions need base forecasts at every level |
| Report error at the decision level | Reported at the level that looks best | The reporting hierarchy is owned by finance |
| Reconcile rather than prorate | Proration in a spreadsheet | Proration is one line and closes the meeting |
| Publish the share vector | Only the allocated number is published | The allocation is treated as an output, not a model |

No public statistic establishes how often companies prorate rather than reconcile. The claim that
proration is the default is an inference from practice, not a measured rate, and any specific
prevalence figure should be treated as unsourced.

### 9.5 The M5 lesson, stated carefully

M5 is the largest public evidence on aggregation levels: 42,840 daily series in 12 levels, scored
with a value-weighted hierarchical metric (WRMSSE), with all 12 levels weighted equally. Three
findings transfer.

**Gains shrink with depth.** The winning entry improved on the exponential smoothing benchmark by
roughly half at the total level and about 3% at product–store level. The bottom level is where
replenishment happens, and it is where method choice matters least in relative terms.

**The overall winner is not the winner at every level.** The top-ranked entry overall ranked 11th at
the bottom level, while the sixth-ranked entry had the best bottom-level score. Different methods
win at different levels, which argues for level-specific method choice rather than one
portfolio-wide model.

**Rankings are sensitive to the evaluation window.** A re-analysis over five rolling windows found
the global bottom-up approach robust at the bottom level but erratic at aggregate levels, and
characterised the final ranking as unstable. Any claim of the form "method X wins" should be read
as "method X won on this window at this level".

> **Note:** M5's bottom level is product–store, not product–DC–week, and its series are retail
> scanner data rather than shipments. The direction of the finding transfers; the magnitudes do not.

### 9.6 What practitioners argue about

**"Forecast where you decide" versus "forecast where the signal is".** The first is operational, the
second statistical, and both are true; the job is the bridge, which is what disaggregation and
reconciliation are.

**"Reconciliation killed top-down."** It did not. Minimum-trace estimators need a forecast-error
covariance, unstable with many series, many zeros, or short histories, and in those conditions
simple methods stay competitive. Reconciliation is a strong default, not a universal replacement.

**"We need one number."** A single number at a single grain is achievable; a single number across
grains is not, because the grains answer different questions. What is achievable is a documented
bridge and a register naming which number governs which decision.

**"The bottom level is the truth."** It is the most granular, not the most accurate: it is where the
most decisions are made and where the data is thinnest. Confusing the two is the commonest
conceptual error in the topic.

## 10. Failure modes and red flags

| Failure | Red flag visible in a meeting or dataset | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Forecasting at the decision grain because the decision is there | The accuracy slide and the service slide disagree by 20 points | Signal-to-noise is worst where the decision is made | Expensive models fitted to noise | Test three levels; measure aggregation gain |
| Aggregate accuracy reported as item accuracy | "94% accuracy" with no level stated | Aggregate denominators are dominated by high-volume items | Improvement programmes that do not move service | State level, weighting, and population |
| Historical shares used through a known mix change | A promoted SKU is short while its siblings are long | Share vector estimated before the event | Stock-out on the promoted item, excess on the rest | Forecast proportions where events are known |
| New item given a historical share | A launch has a share of 0.01 and a stock-out in week 3 | No history means no share | Launch failure attributed to the market | Launch share from the plan, or bottom-up |
| Monthly forecast divided into weeks | Weekly plans that sum to more than the month | `52/12 = 4.333` applied to every month | 15% over-order in five-week months | Weekly forecast, or reconciled temporal hierarchy |
| Proration presented as reconciliation | Every SKU moves by the same percentage | Scalar correction preserves relative error | The gap closes and the error survives | Measure error before and after; test against base |
| Bottom-up and top-down totals both in use | Two reports, two totals, no bridge | Two processes, two owners, no rule | Decisions on incompatible numbers | One register, one bridge, stated precedence |
| Shares that do not sum to one after rounding | The residual is dumped on the largest member | Round-half-up applied per member | Systematic bias against one member | Largest-remainder rounding; publish the residual |
| Grouped structure treated as a tree | Category × region rollups disagree with both parents | Crossed dimensions have no unique path | Double counting or silent loss | Model as grouped; use a summing matrix |
| Override that does not propagate | The planner's number and the region total diverge within a day | System default is local-only, assumption is upward | Plan and forecast drift apart | Define the cascade; audit overrides |
| Error measured only where it flatters | Accuracy always reported at family level | Reporting hierarchy chosen by the reporting team | The decision level is never measured | Report at the decision grain as primary metric |
| Temporal hierarchy assumed to nest | February and March plans do not reconcile to the quarter | Weeks do not sum into months | The quarterly plan is arithmetically false | State nesting; use 4-4-5 or reconcile |

Three deserve a note. **The promoted SKU is the most expensive**: the aggregate is perfect, the
family total exactly right, and the promoted item 45% short. **Proration is the most common**: it is
the only reconciliation most tools offer by default, and the test is whether it improved error
rather than appearance. **The local-only override is the most invisible**: it happens one node at a
time, and the incoherence surfaces weeks later.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks if it fails | How to test it |
|---|---|---|---|
| Shares are stable over the horizon | Top-down assumes the mix repeats or is known | Promotions, launches, competitor entry, price changes | Share range, trend, and year-on-year persistence |
| The aggregate forecast is unbiased | Reconciliation and allocation both inherit the parent's bias | A missed event at the top is spread to every member | Signed bias by level over a rolling window |
| Members are homogeneous enough for a share to be meaningful | A share is a statement about substitutable demand | Grouping items with different dynamics | Within-group versus between-group variance |
| The structure is a partition at the level used | Additive rollups require one parent per node | Dual listing, or a grouped structure treated as a tree | Count nodes with more than one parent |
| Leaf history is long enough to estimate anything | Per-series models need observations | Fewer than two seasonal cycles; many zeros | Observations per series; complete cycles |
| Demand is measured consistently across levels | A rollup is valid only in one unit | Case versus each, returns included at one level only | Test the roll-up identity on raw data |
| The calendar nests as assumed | Temporal reconciliation requires exact nesting | ISO weeks against months | Count weeks wholly inside each month |
| Bottom-level errors are not perfectly correlated | Aggregation gain depends on imperfect correlation | Common promotional or macro shocks | Residual correlation after deseasonalising |
| The decision tolerates the chosen level's error | Every level has a different error | Service targets set at a level the forecast cannot support | Compare WAPE with the safety-stock assumption |
| Overrides propagate as the organisation believes | Coherence depends on it | Local-only system default | Audit one override end to end |

Two limits are worth stating plainly. **The aggregation gain cannot be assumed**: it is computable
from the covariance structure and measurable from a backtest, but it does not transfer between
companies, and every figure in section 9 is a property of a dataset rather than of a method.
**Reconciliation guarantees coherence, not accuracy**: whether a coherent set beats the incoherent
forecasts it replaced must be answered at every level on a backtest with the same origins. Module 08
treats the estimators; this topic only establishes the requirement.

## 12. Exercises

### Exercise 1 — Compute both historical-share methods (calculation, 40 minutes)

Compute both share vectors for the four regions in section 6.2 and verify each sums to 1.
**Deliverable:** both vectors to five decimals, with one region's monthly shares shown as working.
**Acceptance criteria:** each vector sums to 1.00000 within rounding, and one sentence explains which
method gives East the higher share. **Time:** 40 minutes.

### Exercise 2 — Allocate with rounding (calculation, 45 minutes)

Allocate 30,000 units using the proportion-of-averages shares and largest-remainder rounding.
**Deliverable:** the allocation table with floors, remainders, residual, and final integers.
**Acceptance criteria:** allocations sum exactly to 30,000; the residual goes to the largest
remainders and the choice is stated; per-member error is below one unit. **Time:** 45 minutes.

### Exercise 3 — Diagnose the weekly split (calculation, 35 minutes)

A four-week month forecasts 8,000 units; weekly actuals are 2,300, 2,150, 1,980, and 1,570. Compute
the WAPE of a flat plan under `8,000/4` and `8,000/4.333`, and state which is worse.
**Deliverable:** both plans, both error sums, both WAPEs, and the ratio between them.
**Acceptance criteria:** divisor error and profile error are separated; the denominator is stated;
one sentence says what a weekly forecast would add. **Time:** 35 minutes.

### Exercise 4 — Choose a forecasting level (design, 50 minutes)

A distributor has 4,000 SKU–DC series with a median of 18 non-zero weeks in two years, a zero share
of 0.55, and stable family shares. It plans at family × region × month and replenishes weekly.
**Deliverable:** a recommendation naming the level, strategy, share method, and fallback.
**Acceptance criteria:** cites at least three determinants from section 5, step 8; the fallback has
a stated trigger; the answer says what evidence would change it. **Time:** 50 minutes.

### Exercise 5 — Critique the memo (critique, 40 minutes)

A memo argues: "We forecast at SKU–DC–week because that is where the decision is made. Accuracy at
that level is 61%, which is acceptable, so the family-level number is not needed."
**Deliverable:** a numbered critique with at least five distinct defects and a corrected policy.
**Acceptance criteria:** at least two defects concern measurement; at least one concerns the missing
link between levels; the policy names an owner. **Time:** 40 minutes.

### Exercise 6 — Build the error-by-level table (application, 60 minutes)

Take a dataset with at least two hierarchy levels and 12 or more periods. Compute WAPE and signed
bias at two levels, and the ratio between them.
**Deliverable:** a table with level, series count, zero share, WAPE, bias, and ratio.
**Acceptance criteria:** same origins and buckets at both levels; the denominator is stated; one
paragraph separates cancellation from genuine forecastability. **Time:** 60 minutes.

### Exercise 7 — Design the governance map (design, 45 minutes)

For a consumer goods manufacturer with central demand planning, four regional planners, and a
national accounts team, assign forecast ownership at four levels.
**Deliverable:** a one-page map with level, owner role, cadence, and override rule.
**Acceptance criteria:** one owner per level; the cascade is stated for each level; at least one
level is left unowned with a reason. **Time:** 45 minutes.

### Exercise 8 — Middle-out design (design, 45 minutes)

A retailer has 12,000 SKU–store series but reliable data at department × region × week.
**Deliverable:** the middle level, the disaggregation method, the aggregation method, and the
reconciliation step in prose. **Acceptance criteria:** the middle level is justified on volume and
specificity; the share method is named with its assumption; the design says how a new store is
handled. **Time:** 45 minutes.

### Exercise 9 — Measure and attribute the coherence gap (calculation, 40 minutes)

Bottom-up forecasts sum to 46,800 units; the top-down plan is 51,000.
**Deliverable:** the gap, the ratio, the proration factor, the effect on a 900-unit SKU, and a
statement of what proration did not fix. **Acceptance criteria:** the factor is applied correctly;
the gap appears in units and as a ratio; the statement names a specific surviving error.
**Time:** 40 minutes.

### Exercise 10 — Temporal hierarchy design (design, 50 minutes)

A business plans monthly, orders weekly, and delivers daily.
**Deliverable:** a frequency table, a nesting statement, a reconciliation direction, and a rule for
the straddling week. **Acceptance criteria:** nesting is stated for each pair of frequencies; the
straddling-week rule is implementable; the design names the level at which the forecast is produced.
**Time:** 50 minutes.

## 13. Self-check answers

**Exercise 1.**

| Region | Proportion of averages | Average of monthly proportions |
|---|---:|---:|
| North | 0.38023 | 0.38135 |
| Central | 0.28517 | 0.28522 |
| South | 0.20304 | 0.20301 |
| East | 0.13156 | 0.13043 |

East's average of monthly proportions is lower because July, when East was smallest, gets the same
weight as December; the proportion of averages weights each month by volume, so later months
dominate. The ratio is 0.13043/0.13156 = 0.9914, or 0.86% lower. A common wrong answer reverses the
two methods and over-orders the growing region.

**Exercise 2.** Using the proportion-of-averages shares:

| Region | Share | Exact | Floor | Remainder | Final |
|---|---:|---:|---:|---:|---:|
| North | 0.38023 | 11,406.9 | 11,406 | 0.90 | 11,407 |
| Central | 0.28517 | 8,555.1 | 8,555 | 0.10 | 8,555 |
| South | 0.20304 | 6,091.2 | 6,091 | 0.20 | 6,091 |
| East | 0.13156 | 3,946.8 | 3,946 | 0.80 | 3,947 |
| **Total** | **1.00000** | **30,000.0** | **29,998** | | **30,000** |

The floors sum to 29,998, so the two largest remainders — North (0.90) and East (0.80) — each take
one. The residual is 0.0067% of the total and is distributed by a stated rule, not dumped on the
largest member.

**Exercise 3.** The flat plan under `8,000/4 = 2,000` per week:

| Week | Actual | Plan | Absolute error |
|---|---:|---:|---:|
| 1 | 2,300 | 2,000 | 300 |
| 2 | 2,150 | 2,000 | 150 |
| 3 | 1,980 | 2,000 | 20 |
| 4 | 1,570 | 2,000 | 430 |
| **Sum** | **8,000** | **8,000** | **900** |

WAPE = 900/8,000 = 0.1125. Under `8,000/4.333 = 1,846.3` per week the plan sums to 7,385.2, a
shortfall of 7.7%, and the absolute errors sum to 1,167.4 for a WAPE of 0.1459, worse by a ratio of
1.30. Two errors compound: the divisor is wrong for a four-week month, and the plan no longer sums
to the month. The profile error of 0.1125 remains in both.

**Exercise 4.** A strong answer forecasts at family × region × month with top-down allocation by
forecast proportions, on the evidence of a 0.55 zero share and 18 non-zero weeks at the leaf. It
names reconciled middle-out at SKU × region × month as the fallback, triggers it on a share range
above 0.10 on more than 30% of members, and says what evidence would change the recommendation. A
weak answer recommends SKU–DC because that is where the decision is, without engaging with the zero
share.

**Exercise 5.** Defects: accuracy is stated without a level or denominator; the two levels are not
alternatives, since the decision needs SKU–DC and the plan needs family; "acceptable" is asserted
rather than derived from the service target and the cost of error; there is no benchmark, so 61% may
be worse than seasonal naive; the aggregate's own accuracy is unmeasured; and no owner is assigned
for the bridge between levels. The corrected policy states decision level, planning level,
allocation method, reconciliation rule, and owner.

**Exercise 6.** A strong table uses the same origins and buckets at both levels, reports WAPE with
its denominator, and separates the ratio into cancellation (the gap between the sum of absolute
errors and the absolute sum of errors) and forecastability (each level's error against its own
seasonal naive benchmark). A weak table compares a weekly bottom-level WAPE with a monthly aggregate
WAPE and attributes the whole difference to aggregation.

**Exercise 7.** A strong map gives central demand planning category × region × month, regional
planners family × region × week, the national accounts team customer × month, and leaves SKU × DC ×
week to the replenishment system with a named exception owner. A regional override propagates upward
and is reconciled; a national accounts override is local-only because it does not affect supply. A
weak map assigns "demand planning" to all levels and leaves the cascade unstated.

**Exercise 8.** A strong design uses department × region × week as the middle level, allocates to
SKU × store by forecast proportions within department, aggregates by summing, and reconciles the
bottom level to the middle with a stated estimator. A new store gets the department average per
store for eight weeks, flagged as an estimate. A weak design picks the store level and has no rule
for new stores.

**Exercise 9.** Gap = 51,000 − 46,800 = 4,200 units, a ratio of 4,200/51,000 = 0.0824, so the
bottom-up total is 8.24% low. Proration factor = 51,000/46,800 = 1.0897, so the 900-unit SKU becomes
981 units. Proration did not fix the relative error: a SKU already 30% high remains 30% high, and
the 81 extra units scale the error rather than correcting it. A common wrong answer uses 0.9176 and
deflates every forecast.

**Exercise 10.** A strong design states that days nest into weeks and months do not nest into weeks,
forecasts at ISO week, aggregates to month for the financial plan, and reconciles month to week with
a stated straddling-week rule (majority of days, or the 4-4-5 calendar). The daily schedule is a
profile within the week, not a reconciliation level. A weak design divides the month by 4.333.

## 14. Interview questions

**1. "At what level would you forecast a product hierarchy?"**

A strong answer refuses the question as posed: it depends on the decision grain, the data at each
level, the stability of the shares, and the evaluation level, and it names the measurement that
would settle it. A weak answer says "at the SKU level, because that is the most granular".

**2. "Why does aggregating improve forecast accuracy?"**

A strong answer gives the variance identity, shows the gain is bounded by the correlation between
members, and notes that part of the apparent gain is error cancellation rather than better
prediction. A weak answer says "because there is less noise" without quantifying.

**3. "When is bottom-up better than top-down?"**

A strong answer names the conditions: dense, long, low-variability leaf series; low correlation
across members; stable shares; evaluation at the leaf. It cites one study where bottom-up won and
one where it lost. A weak answer says "bottom-up is more accurate because it uses more data".

**4. "What is the difference between average of proportions and proportion of averages?"**

A strong answer gives both formulas, explains that one weights periods equally and the other weights
units equally, and says which direction the difference goes under growth. A weak answer says they
are two ways of saying the same thing.

**5. "Why can you not divide a monthly forecast by 4.333 to get weeks?"**

A strong answer gives the two reasons: months contain four or five weeks, and the intra-month
profile is not flat. It quantifies the error (about 15% in a five-week month before any profile
effect) and proposes a temporal hierarchy. A weak answer says "because months have different
lengths" and stops.

**6. "What is a coherent forecast?"**

A strong answer defines it as one where every parent equals the sum of its children at every level
and bucket, and immediately distinguishes coherence from accuracy. A weak answer says "one that
adds up" without the level-and-bucket qualification.

**7. "We reconcile our forecasts. Does that make them more accurate?"**

A strong answer says reconciliation guarantees coherence and often improves accuracy but not
always: if the base forecasts are biased, reconciliation spreads the bias. It requires a backtest of
reconciled against base at every level. A weak answer says yes, because reconciliation uses all the
information.

**8. "A planner overrides a SKU forecast. What should happen to the region total?"**

A strong answer describes the four cascade rules and says the organisation must choose one in
advance, because the system default and the planner's assumption usually differ. A weak answer says
"the total should update" without recognising that this is a policy choice.

**9. "How would you decide between forecasting at family level and at SKU level?"**

A strong answer describes the procedure: measure data sufficiency at both, measure aggregation gain
on a backtest, test share stability, and score on the five determinants. A weak answer picks one
from intuition about noise.

**10. "What did the M5 competition show about aggregation levels?"**

A strong answer gives the direction and the caveat: gains shrink from roughly half at the total to
about 3% at product–store level, the overall winner was not the best at the bottom level, and the
rankings are sensitive to the evaluation window. A weak answer says "the best methods used
bottom-up" without the level-dependence.

**11. "Is a grouped structure the same as a hierarchy?"**

A strong answer says no: crossed dimensions have no unique parent path, so the structure needs a
summing matrix encoding several aggregation routes, and reconciliation is not unique without a
constraint. A weak answer treats the two as synonyms.

## 15. Key terms

- **Hierarchy** — a containment relation with one parent per node.
- **Grouped structure** — crossed hierarchies with no unique parent path.
- **Level** — a named depth; **bottom level** — the finest level forecast or decided.
- **Decision grain** — the level and bucket at which a commitment is made.
- **Planning grain** — the level and bucket at which a plan is produced.
- **Top-down** — forecast the total, allocate down by shares.
- **Bottom-up** — forecast the leaves, sum upward.
- **Middle-out** — forecast an intermediate level, move both ways.
- **Base forecast** — an independent forecast at one level before reconciliation.
- **Coherent forecast** — a set in which every parent equals the sum of its children.
- **Reconciliation** — adjusting base forecasts so coherence holds.
- **MinT** — minimum-trace reconciliation; Module 08.
- **Average of proportions** — the mean of each period's share.
- **Proportion of averages** — the member total over the parent total.
- **Forecast proportion** — a share computed from base forecasts.
- **Proration** — scaling all members by one factor to close a gap.
- **Aggregation gain** — the error reduction from forecasting a sum.
- **Disaggregation cost** — the error added by allocating to a finer level.
- **Error cancellation** — offsetting of signed errors when series are summed.
- **Specification error** — error from modelling heterogeneous members as one.
- **Sampling error** — error from too little data per series.
- **Temporal aggregation** — summing a series into coarser buckets.
- **Temporal hierarchy** — one series at several frequencies at once.
- **WAPE** — sum of absolute errors over sum of actuals.
- **WRMSSE** — value-weighted RMSSE, the M5 metric.
- **Override cascade** — the rule for how an adjustment propagates across levels.

## 16. Sources

**Peer-reviewed and academic — Grade A**

- Schwarzkopf, A. B., Tersine, R. J., and Morris, J. S. (1988). "Top-down versus bottom-up
  forecasting strategies." *International Journal of Production Research*.
- Gross, C. W., and Sohl, J. E. (1990). "Disaggregation methods to expedite product line
  forecasting." *Journal of Forecasting*, 9(3), 233–254. DOI: 10.1002/for.3980090304.
- Fliedner, E. B., and Mabert, V. A. (1992). "Constrained Forecasting: Some Implementation
  Guidelines." *Decision Sciences*, 23(5), 1143–1161. DOI: 10.1111/j.1540-5915.1992.tb00440.x.
- Fliedner, G. (2001). "Hierarchical forecasting: issues and use guidelines." *Industrial
  Management & Data Systems*, 101(1). DOI: 10.1108/02635570110365952.
- Zotteri, G., Kalchschmidt, M., and Caniato, F. (2005). "The impact of aggregation level on
  forecasting performance." *International Journal of Production Economics*, 93, 479–491.
- Zotteri, G., and Kalchschmidt, M. (2007). "A model for selecting the appropriate level of
  aggregation in forecasting processes." *International Journal of Production Economics*,
  108(1–2), 74–83.
- Athanasopoulos, G., Ahmed, R. A., and Hyndman, R. J. (2009). "Hierarchical forecasts for
  Australian domestic tourism." *International Journal of Forecasting*, 25(1), 146–166.
- Hyndman, R. J., Ahmed, R. A., Athanasopoulos, G., and Shang, H. L. (2011). "Optimal combination
  forecasts for hierarchical time series." *Computational Statistics and Data Analysis*, 55(9),
  2579–2589. DOI: 10.1016/j.csda.2011.03.006.
- Athanasopoulos, G., Hyndman, R. J., Kourentzes, N., and Petropoulos, F. (2017). "Forecasting with
  temporal hierarchies." *European Journal of Operational Research*, 262(1), 60–74.
  DOI: 10.1016/j.ejor.2017.02.046.
- Kourentzes, N., Rostami-Tabar, B., and Barrow, D. K. (2017). "Demand forecasting by temporal
  aggregation." *Journal of Business Research*. DOI: 10.1016/j.jbusres.2017.04.016.
- Oliveira, J. M., and Ramos, P. (2019). "Assessing the Performance of Hierarchical Forecasting
  Methods on the Retail Sector." *Entropy*, 21(4), 436.
- Mirčetić, D., Rostami-Tabar, B., and colleagues (2021). "Forecasting hierarchical time series in
  supply chains: an empirical investigation." *International Journal of Production Research*.
  DOI: 10.1080/00207543.2021.1896817.
- Babai, M. Z., Boylan, J. E., and Rostami-Tabar, B. (2022). "Demand forecasting in supply chains: a
  review of aggregation and hierarchical approaches." *International Journal of Production
  Research*, 60(1), 324–348. DOI: 10.1080/00207543.2021.2005268.
- Makridakis, S., Spiliotis, E., and Assimakopoulos, V. (2022). "M5 accuracy competition: Results,
  findings, and conclusions." *International Journal of Forecasting*, 38(4), 1346–1364.
  DOI: 10.1016/j.ijforecast.2021.11.013.
- Athanasopoulos, G., Hyndman, R. J., Kourentzes, N., and Panagiotelis, A. (2024). "Forecast
  reconciliation: A review." *International Journal of Forecasting*, 40(2), 430–456.
- Silveira Netto, L. F., Brei, V. A., and Hyndman, R. J. (2024). "Forecasting hierarchical time
  series in supply chains: an empirical investigation." *Applied Stochastic Models in Business and
  Industry*. DOI: 10.1002/asmb.2823.
- Wickramasuriya, S. L., Athanasopoulos, G., and Hyndman, R. J. (2019). "Optimal forecast
  reconciliation for hierarchical and grouped time series through trace minimization." *Journal of
  the American Statistical Association*. Source of the MinT estimator; treated in Module 08.

**Books and teaching texts — Grade B**

- Hyndman, R. J., and Athanasopoulos, G. *Forecasting: Principles and Practice*, current edition.
  Chapter on hierarchical and grouped time series; source of the top-down formula and the statement
  that no top-down method is unbiased.
- ASCM/APICS Dictionary, current edition, for aggregate planning and S&OP definitions.

**Standards and official — Grade B**

- GS1, *GPC Development and Implementation Guide*, and the GPC level definitions (segment, family,
  class, brick).
- GS1 General Specifications, current edition, for GTIN and GLN.
- ISO 8601, *Date and time — Representations for information interchange*, for week numbering.
- National Retail Federation, 4-4-5 retail calendar documentation.

**Practitioner and vendor — Grade C, D, or E**

- Kiuru, J. (2021). "On hierarchical retail forecasting: Assessing forecast aggregation and
  disaggregation methods." Aalto University thesis. Grade A for method, C for generality.
- Ma, S., and Fildes, R. (2021). "The performance of the global bottom-up approach in the M5
  accuracy competition." Lancaster University eprint. Grade A for the re-analysis, C for the
  "lottery" characterisation.

**Note on citations.** Every empirical figure in section 9 is reported as the cited study states it.
The M5 level-1 and level-12 figures are quoted from the competition's reported results, and exact
percentages differ between the competition paper and later re-analyses, so the text gives direction
and approximate magnitude rather than a precision the sources do not agree on. The Aldergrove
figures in section 6 are constructed for teaching, and the construction is shown in full. No
statistic here is invented.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic, regression with lags and Fourier terms: [`04_regression-lags-fourier.md`](04_regression-lags-fourier.md)
- The benchmark discipline this topic assumes: [`01_forecastability-benchmarks.md`](01_forecastability-benchmarks.md)
- Exponential smoothing as the base forecaster at the planning level: [`02_exponential-smoothing.md`](02_exponential-smoothing.md)
- ARIMA and SARIMAX as an alternative base forecaster: [`03_arima-sarimax.md`](03_arima-sarimax.md)
- Next topic, intermittent demand where the bottom level lives: [`06_intermittent-demand.md`](06_intermittent-demand.md)
- Forecast combination, which interacts with reconciliation: [`07_forecast-combination.md`](07_forecast-combination.md)
- Backtesting at multiple levels: [`08_rolling-origin-backtesting.md`](08_rolling-origin-backtesting.md)
- Error metrics and their cost interpretation: [`09_error-metrics-cost.md`](09_error-metrics-cost.md)
- Overrides and forecast value added: [`10_overrides-fva.md`](10_overrides-fva.md)
- Reconciliation estimators and the MinT family: [`../08_hierarchical_intermittent_probabilistic_forecasting/_index.md`](../08_hierarchical_intermittent_probabilistic_forecasting/_index.md)
- Hierarchy design, grain, and additivity: [`../../01_foundations/01A_foundations_units_time_assumptions/07_hierarchy-design.md`](../../01_foundations/01A_foundations_units_time_assumptions/07_hierarchy-design.md)
- Forecast error metrics at the foundations level: [`../../01_foundations/01A_foundations_units_time_assumptions/09_forecast-error-metrics.md`](../../01_foundations/01A_foundations_units_time_assumptions/09_forecast-error-metrics.md)
- Regression with time-series structure: [`../../01_foundations/04_statistics_causal_inference/04_regression-time-series.md`](../../01_foundations/04_statistics_causal_inference/04_regression-time-series.md)
- Demand sensing and promotions, which move shares: [`../09_demand_sensing_promotions_npi/_index.md`](../09_demand_sensing_promotions_npi/_index.md)
- Inventory policy downstream of the forecast: [`../10_inventory_optimization_multi_echelon/_index.md`](../10_inventory_optimization_multi_echelon/_index.md)
- Part map for Methods: [`../_index.md`](../_index.md)
- Foundations part map: [`../../01_foundations/_index.md`](../../01_foundations/_index.md)
- Survey file: [`../../../course/07_time_series_forecasting.md`](../../../course/07_time_series_forecasting.md)
- Hierarchical and probabilistic forecasting survey: [`../../../course/08_hierarchical_intermittent_probabilistic_forecasting.md`](../../../course/08_hierarchical_intermittent_probabilistic_forecasting.md)
