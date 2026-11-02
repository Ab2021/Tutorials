> Module 01A — Foundations: Units, Time, and Assumptions
> Topic 10 of 11
> Estimated study time: 3.5 hours
> Prerequisites: 01A.09

# Why MAPE Fails on Intermittent Demand

---

## 1. Why this topic exists

A demand-planning scorecard shows two items. The first is a fast-moving consumable with a mean absolute percentage error (MAPE)
of 18%. The second is a spare part with a MAPE of 340%. The planning director concludes that the second item is badly forecast
and asks for the model to be fixed. The model is not the problem. The denominator is.

Percentage error divides the absolute error by the actual value. When the actual is 40, an error of 4 costs 10%. When the actual
is 4, the same error costs 100%. When the actual is 0, the ratio is undefined, and every spreadsheet, planning system, and
dashboard resolves that undefined value with a silent local decision: drop the period, add one, replace the zero, or return a
blank. Those four choices produce four different numbers from identical forecasts and identical outcomes. None of them is
usually documented.

Intermittent demand is the regime where the denominator is frequently zero or small. It is not an exotic corner of forecasting.
It covers spare parts, pharmaceutical wholesalers, industrial distributors, grocery long-tail assortments, and most of the
stock-keeping unit (SKU) count in any large catalogue. In these portfolios the majority of item-location-weeks contain no demand
at all. A metric whose denominator vanishes on the majority of observations is not measuring forecast quality. It is measuring
the size of the denominator.

The consequences are concrete and expensive:

1. **The wrong method wins a bake-off.** A method that forecasts near zero is rewarded, because near-zero forecasts are cheap
   when the actual is usually zero, and percentage error penalises over-forecasting more than under-forecasting. The method that
   is best for service can lose to the method that is best at being absent.
2. **Item prioritisation inverts.** Slow items look worse by construction, so effort flows to the items where percentage error
   is smallest in absolute terms and away from the items where a stock-out stops a line or a treatment.
3. **The service number and the accuracy number describe different worlds.** A team can improve reported accuracy for a year
   while fill rate is flat or falling, and both numbers can be computed correctly.

This topic does four things: it classifies demand so that the regime is explicit, shows where percentage errors break and why
the break is structural, presents the metrics that survive zeros and small denominators, and shows how to compare two methods
across a portfolio in which most item-weeks are zero.

## 2. Learning outcomes

By the end of this topic you can:

1. **Classify** a demand series as smooth, erratic, intermittent, or lumpy using the average inter-demand interval (ADI) and the
   squared coefficient of variation (CV²), and **state** the bucket and the observation window used.
2. **Demonstrate**, with arithmetic, how a fixed absolute error produces percentage errors that differ by two or three orders of
   magnitude as the denominator falls.
3. **Explain** why MAPE is undefined at zero demand, and **audit** the four common zero-handling rules for their effect on a
   reported number.
4. **Identify** the forecast that each common loss function prefers, and **explain** why the mean absolute error (MAE), the mean
   absolute scaled error (MASE), and MAPE can all favour a zero forecast on intermittent demand.
5. **Compute** MASE, the root mean squared scaled error (RMSSE), weighted MAPE (wMAPE), and pinball loss for a small series,
   with the scaling denominator and the zero rule stated.
6. **Choose** an evaluation grain and an aggregation rule that match a named decision, and **justify** the choice against at
   least two alternatives.
7. **Compare** two forecasting methods on a portfolio where most observations are zero, using pooled and equal-weight measures,
   win rates, and a service measure, and **explain** any disagreement between them.
8. **Design** a scorecard that cannot be improved by forecasting zero, dropping zero periods, or aggregating to a level where
   the denominator is large.

Each outcome maps to an exercise in section 12 and to a self-check item in section 13.

## 3. The core mental model

Hold two pictures at once. The first is that a demand series is not one process but two: whether demand occurs at all, and how
large it is when it does. The second is that every error metric is a loss function in disguise, and every loss function names
the forecast it prefers. Most of the confusion in intermittent-demand evaluation comes from applying the first picture without
the second.

### 3.1 Demand as two processes

Write each observation as a product:

```text
y_t = o_t * z_t

  o_t  in {0, 1}      occurrence: did demand happen in period t?
  z_t  > 0            size: how much, given that it happened
```

This decomposition is not a modelling convenience. It is the structure that the classification in Section 5 measures and that
the metrics in Section 6 stress. The occurrence process determines how often the denominator of a percentage error is zero. The
size process determines how large the numerator can be when it is not.

Two summaries describe the two processes:

```text
ADI = n / (number of periods with y_t > 0)          units: periods per demand
CV2 = ( standard deviation of nonzero z_t / mean of nonzero z_t ) ^ 2    units: dimensionless
```

The ADI is the average number of periods you wait between demands. An ADI of 1.0 means demand in every period. An ADI of 4.0
means demand, on average, one week in four. The CV² measures the relative dispersion of the demand sizes, using only the periods
where demand occurred. It is dimensionless, which is why it can be compared across items measured in eaches, cases, kilograms,
or litres.

> **Note:** Some software computes CV² over the full series including zeros. That is a different
> quantity, and it conflates occurrence with size: a series with steady demand every fourth week
> and a series with erratic demand every week can produce similar full-series values. The
> classification literature uses the CV² of demand sizes. Check which one your tool computes
> before comparing an item to a published cut-off.

### 3.2 The classification grid

Plotting ADI against CV² gives four regions with names that have become standard in inventory research:

```text
                          CV2 of demand size
                       low  (< 0.49)        high (>= 0.49)
                    +---------------------+---------------------+
                    |                     |                     |
     ADI < 1.32     |       SMOOTH        |       ERRATIC       |
     (frequent)     |  demand most weeks  |  demand most weeks  |
                    |  size fairly stable |  size jumps around  |
                    |                     |                     |
                    +---------------------+---------------------+
                    |                     |                     |
     ADI >= 1.32    |    INTERMITTENT     |        LUMPY        |
     (sporadic)     |  regular gaps,      |  gaps and jumps;    |
                    |  size fairly stable |  hardest to plan    |
                    |                     |                     |
                    +---------------------+---------------------+
```

Read the grid as a map of which problem you have. Smooth demand is a level-estimation problem. Erratic demand is a variance
problem: the level may be stable but individual periods are not. Intermittent demand is an occurrence problem: the question is
when, not how much. Lumpy demand is both at once, and it is where point forecasts and percentage errors fail most completely.

The cut-offs of 1.32 and 0.49 are the values most often quoted from Syntetos, Boylan and Croston (2005). They were derived from
the forecast-error performance of competing estimators on a large sample of automotive series, not from a physical law. The
paper's durable contribution is the two-dimensional framing and the practice of deriving cut-offs from error performance on your
own data, not the constants themselves. Treat the numbers as conventions and re-derive them if your portfolio differs
materially.

### 3.3 The denominator cliff

For a fixed absolute error, percentage error is a hyperbola in the actual value. An error of five units produces the following:

```text
 MAPE for a fixed absolute error of 5 units
   %
 500 |*
     |
 400 |
     | *
 300 |
     |   *
 200 |
     |      *
 100 |            *
     |                    *
   0 +----------------------------------> actual demand y (units)
       1    5    10   20    50   100  200
```

The same five-unit miss is 500% at an actual of 1, 100% at 5, 25% at 20, and 2.5% at 200. Nothing about the forecast changed.
Only the denominator did. In a portfolio, this means the reported error is dominated by the smallest observations, which are
usually the least consequential ones for inventory.

The cliff has a second face. When the forecast is exactly zero and the actual is positive, the percentage error is exactly 100%,
regardless of whether the miss was 5 units or 20. Percentage error saturates from below, so it cannot distinguish a near miss
from a large one in precisely the regime where misses are large.

### 3.4 A metric is a loss function

Every accuracy measure has an implicit answer to the question "which point forecast would you like me to produce?" Minimising
the measure selects that forecast. This is the single most useful idea in forecast evaluation, and it is usually left implicit.

| Loss or measure | Point forecast it is minimised by | What that means under intermittent demand |
|---|---|---|
| Squared error (MSE, RMSE) | The conditional mean | A small positive number, pulled up by the size of large demands |
| Absolute error (MAE, MASE, wMAPE) | The conditional median | Exactly zero whenever demand occurs in fewer than half of periods |
| Percentage error (MAPE) | A low-biased functional, below the median for right-skewed demand | Zero or near zero; undefined where the actual is zero |
| Pinball loss at level τ | The conditional τ-quantile | Zero whenever τ is below the probability of no demand |
| Fill rate or newsvendor cost | Depends on the critical ratio and the policy | Often a high quantile, which is the opposite of what MAE wants |

Three consequences follow, and they are the reason this topic exists.

First, **a zero forecast is not a stupid forecast under MAE or MASE**. If demand occurs in 20% of weeks, the median weekly
demand is zero, so zero is the MAE-optimal constant forecast. A method that forecasts zero every week is doing exactly what the
metric asked. It will also produce a fill rate of zero.

Second, **MAPE's preferred forecast is biased low**. Over-forecasting is penalised more heavily than under-forecasting. Given an
actual of 100, a forecast of 150 scores 50%; given an actual of 150, a forecast of 100 scores 33.3%. A forecaster or a model
selection procedure that minimises MAPE learns to under-forecast, which is the opposite of the direction that protects service.

Third, **choosing a metric is choosing an objective**, and an objective that does not match the decision will be optimised into
a worse decision. This is not a hypothetical. It is the normal result of evaluating a replenishment model on MAE and then using
its output to set safety stock.

### 3.5 Aggregation is a second model

A portfolio number is not the average of item numbers. It is a new statistic with its own behaviour. Four common rules answer
four different questions:

- **Pooled or micro average.** Sum absolute errors across items, divide by summed demand. Answers "what share of total units did
  we misallocate?" Dominated by the largest items.
- **Equal-weight or macro average.** Compute the metric per item, then take the arithmetic mean. Answers "how does the typical
  item fare, counting each item once?" Dominated by whichever items have extreme ratios.
- **Median across items.** Answers "what does the middle item look like?" Insensitive to both volume and extremes, which is a
  virtue for robustness and a defect for cost.
- **Win rate.** Count the items on which method A beats method B. Answers "how often does A win?" and ignores the size of each
  win.

On intermittent portfolios these four rules routinely rank two methods in opposite orders. That is not a paradox to be resolved.
It is four questions, and the analyst's job is to say which one the business is asking.

### 3.6 Accuracy is not service

The forecast does not serve the customer. The inventory does. A forecast enters a replenishment rule, which becomes an order
quantity, a stock position, and a fill rate. Each step adds assumptions, and the mapping from accuracy to service is neither
monotonic nor stable.

Two service measures answer different questions. **Fill rate** is the fraction of demand units served from stock: how much
demand did we fail to meet? **Stockout opportunity rate** is the fraction of demand periods in which some demand went unmet: how
often did we fail? A forecast can improve one while worsening the other, depending on whether its errors fall on large or small
demand events. Neither can be inferred from MAE, MAPE, or MASE.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Intermittent demand | Demand that occurs in some periods and not others, with gaps between occurrences | Used for any low-volume item, including one with demand every period but small size | Occurrence pattern, not a magnitude |
| Lumpy demand | Intermittent demand whose nonzero sizes are also highly variable | Used as a synonym for intermittent | ADI ≥ 1.32 and CV² ≥ 0.49 |
| Erratic demand | Demand in most periods with highly variable size | Confused with lumpy because both have high CV² | ADI < 1.32 and CV² ≥ 0.49 |
| Smooth demand | Frequent demand with stable size | Assumed to be the normal case; it is the minority in most catalogues | ADI < 1.32 and CV² < 0.49 |
| Demand occurrence | The binary event that demand is greater than zero in a period | Treated as a modelling detail rather than half the process | 0 or 1 per item-period |
| Demand size | The quantity demanded in a period where demand occurred | Confused with the period demand including zeros | Units (each, case, kg) |
| Average inter-demand interval (ADI) | Mean number of periods between demand occurrences, estimated as periods divided by occurrences | Computed on the nonzero periods only, which always gives 1.0 | Periods per demand |
| Squared coefficient of variation (CV²) | Variance of demand sizes divided by the square of their mean | Computed over the full series including zeros | Dimensionless |
| Croston's method | Exponential smoothing applied separately to occurrence intervals and demand sizes, with the rate as their ratio | Applied to series with no zeros, where it reduces to a worse exponential smoothing | Units per period |
| Syntetos–Boylan approximation (SBA) | Croston's method with a bias correction of (1 − α/2) on the rate | Assumed unbiased; it over-corrects and can be negatively biased | Units per period |
| Teunter–Syntetos–Babai (TSB) method | A method that updates the occurrence probability every period, including zero-demand periods | Confused with Croston, which does not update during zeros | Units per period |
| Forecast bias | The mean signed error, forecast minus actual, over an evaluation window | Reported as an absolute value, which hides direction | Units per period |
| Mean absolute percentage error (MAPE) | Mean of absolute error divided by actual, over periods with nonzero actuals | Reported without stating the zero rule or the coverage | Percent |
| Symmetric MAPE (sMAPE) | Mean of absolute error divided by the average of actual and forecast | Assumed bounded by 100%; it is not | Percent |
| Weighted MAPE (wMAPE) | Sum of absolute errors divided by sum of actuals | Assumed to be an average of per-item MAPEs; it is a ratio of sums | Percent |
| Mean absolute scaled error (MASE) | Mean absolute error divided by the in-sample MAE of the naive one-step forecast | Compared across series with different seasonal periods without adjustment | Dimensionless ratio |
| Root mean squared scaled error (RMSSE) | Root mean squared error divided by the in-sample RMSE of the naive forecast | Treated as interchangeable with MASE | Dimensionless ratio |
| Naive benchmark | The one-step forecast that repeats the last observation | Chosen casually; a seasonal naive is the correct floor for seasonal data | Units per period |
| Average relative MAE (AvgRelMAE) | Geometric mean of per-series ratios of MAE to benchmark MAE | Confused with GMRAE, which uses a different denominator | Ratio |
| Forecast value added (FVA) | Change in a chosen error measure attributable to a process step, relative to a naive baseline | Computed against a baseline that is itself a forecast | Percent or units, depending on the measure |
| Fill rate | Fraction of demand units served from stock | Confused with cycle service level | Proportion |
| Cycle service level (CSL) | Probability that a replenishment cycle ends without a stock-out | Reported as fill rate | Proportion |
| Stockout opportunity | A period in which demand occurred and some of it went unmet; the rate is such periods divided by demand periods | Defined differently by different teams; the definition must be stated | Proportion of demand periods |
| Lost sales | Demand that was not served and was not backordered | Replaced by recorded sales, which are censored | Units |
| Censored demand | True demand that is unobservable because stock was insufficient | Ignored, so the forecast learns from the stock-out it caused | Units |
| Pinball loss | An asymmetric absolute loss whose minimiser is a named quantile | Confused with MAE, which is the special case at the median | Units, or units times cost |
| Quantile forecast | A forecast of the value below which demand falls with a stated probability | Confused with a prediction interval, which has two endpoints | Units at a stated probability |
| Critical ratio | Underage cost divided by the sum of underage and overage costs; the service level a newsvendor should hold | Set by policy without reference to costs | Proportion |
| Evaluation grain | The item, location, and period level at which the metric is computed | Set by data availability rather than by the decision | Item × location × period |
| Metric coverage | Share of evaluation periods that contribute to a metric after zero handling | Never reported, which is how zero-dropping hides | Percent of periods |
| Cost-based metric | A loss expressed in currency, derived from the consequences of error | Approximated by an accuracy measure and assumed proportional | Currency |
| Newsvendor cost | Expected overage plus underage cost at a chosen order quantity | Omitted when the cost parameters are unknown, which is usual | Currency per period |
## 5. The mechanics, step by step

The procedure below is the one to follow whenever a forecast is about to be judged. It is deliberately mechanical, because the
failure mode is not ignorance of the formulas. It is skipping a step and not noticing.

### Phase A — Define the observation

**Step 1. Fix the grid.**

Decide the item, location, and period that constitute one observation. In most operational systems this is the stock-keeping
unit at a stocking location in a week or a day. Write it down before looking at the data, because the grid determines which
zeros are real and which are artifacts of the query.

- If the grid is item × warehouse × week, an item that is not stocked in a warehouse produces a structural zero, not a demand
  observation.
- If the grid is item × region × month, an item with no activity in a region for a month produces a zero that may mean "no
  demand" or "not ranged here". Those are different facts.
- The grid must be dense: every item-location combination present for every period in scope, with explicit zeros. A sparse table
  of sales lines is not a demand history.

**Step 2. Define the demand construct.**

Demand is what customers wanted, not what the warehouse shipped. Recorded shipments are censored by availability: when stock
runs out, the unmet demand disappears from the data. A forecast trained on shipments learns that demand falls whenever the item
is unavailable, and then reproduces the stock-out.

Three constructs are common, and they are not interchangeable:

| Construct | What it measures | Bias when used as demand |
|---|---|---|
| Point-of-sale (POS) or consumption | Units scanned or consumed at the point of use | Close to demand where the item is always available; censored where it is not |
| Shipments from a distribution centre | Units leaving a node | Includes channel stuffing and forward buying; excludes unmet demand |
| Customer orders | Units requested | Includes batching, minimum order quantities, and shortage gaming |

State which one is used. If the only available series is shipments, say so, and say which direction the bias is likely to run.

**Step 3. Fix the time bucket.**

The bucket determines both the ADI and the apparent zero rate. Daily data aggregated to weekly turns five demand days into one
demand week, which converts an intermittent series into a smoother one and changes its classification. Monthly aggregation can
turn a lumpy item into a smooth one and hide the stock-outs entirely.

The bucket must match the replenishment decision. If orders are placed weekly, weekly is the evaluation grain even if the
forecast is generated daily. Topic 04 of this module treats the calendar mechanics that make buckets comparable across systems
and time zones.

### Phase B — Describe the series

**Step 4. Separate occurrence from size.**

For each item, compute the count of periods with demand, the share of periods with demand, and the mean and standard deviation
of the nonzero sizes. These four numbers are the sufficient description for classification, and they are worth keeping alongside
the error metrics, because they explain the metrics.

**Step 5. Compute ADI and CV².**

```text
ADI  = n / n_positive
CV2  = ( sd(z) / mean(z) ) ^ 2      over z = { y_t : y_t > 0 }
```

Worked on a twelve-week series with four demand weeks and sizes 5, 6, 5, 5:

```text
ADI  = 12 / 4 = 3.00 periods per demand
mean(z) = 5.25, sd(z) = 0.50
CV2  = (0.50 / 5.25)^2 = 0.0091
```

That item is intermittent: ADI above 1.32, CV² below 0.49. It is also easy to forecast in the only sense that matters
operationally, because when demand comes it is almost always five units. Its MAPE will nonetheless look terrible, because three
weeks in four have a zero denominator.

**Step 6. Record the zero share and the median.**

Two more numbers explain most metric behaviour:

- **Zero share** is the fraction of periods with no demand. It is the fraction of periods that MAPE must discard, replace, or
  fail on.
- **Median demand** is zero whenever the zero share exceeds one half. That single fact explains why MAE-optimal and MASE-optimal
  forecasts are zero for the majority of items in a typical catalogue.

### Phase C — Choose the loss

**Step 7. Derive the decision quantile.**

Ask what the forecast is used for. If it sets a reorder point or an order-up-to level under a newsvendor-like trade-off, the
correct target is a quantile, not a mean:

```text
tau = C_u / (C_u + C_o)

  C_u = underage cost per unit (lost margin, expedite premium, downtime)
  C_o = overage cost per unit (holding, markdown, obsolescence, disposal)
```

A critical ratio of 0.9 means the order-up-to level should be the 90th percentile of lead-time demand. A critical ratio of 0.5
means the median, and under intermittent demand the median is often zero. The ratio is a business input. It cannot be inferred
from the forecast.

**Step 8. Choose the metric family.**

| Family | Examples | Scale-free | Handles zeros | Question it answers | Where it breaks |
|---|---|---|---|---|---|
| Scale-dependent | MAE, RMSE, mean error | No | Yes | How large is the miss, in units | Not comparable across items or units of measure |
| Percentage | MAPE, sMAPE, wMAPE | Yes | MAPE no; wMAPE yes | How large is the miss relative to demand | Denominator near zero; asymmetry; saturation at 100% |
| Scaled | MASE, RMSSE, MdASE | Yes | Yes | Is the method better than naive? | Optimum is the median; denominator can be zero |
| Relative | MRAE, GMRAE, AvgRelMAE, percentage best | Yes | Yes | Is the method better than a named benchmark? | Heavy-tailed ratios; zero ratios break the geometric mean |
| Service | Fill rate, CSL, stockout opportunity rate | Proportion | Yes | Did we serve the demand? | Depends on policy and on demand being observed |
| Cost | Newsvendor cost, pinball loss, interval score | Yes if normalised | Yes | What did the error cost? | Requires cost parameters that are often unavailable |
| Distributional | CRPS, PIT, log score | Yes | Yes | Is the whole predictive distribution right? | Harder to explain; needs a distributional forecast |

The practical rule is to use one metric from the accuracy families as a diagnostic and one from the service or cost families as
the decision criterion. Reporting only the first is how a team can be accurate and unhelpful at the same time.

**Step 9. Fix the zero rule and report coverage.**

Whichever percentage measure is used, state exactly what happens when the actual is zero. The four common rules produce
materially different numbers:

| Rule | Operation | Effect on the metric |
|---|---|---|
| Drop the period | Exclude periods with zero actual | Conditions on demand occurring; coverage falls; the number describes demand weeks only |
| Replace the actual | Substitute 1, or the item mean, for zero | Arbitrary; can dominate the average |
| Add one to both | Compute abs(e)/(y+1) | Bounded, but no longer a percentage error of anything in particular |
| Aggregate first | Compute wMAPE over summed demand | Defined whenever the portfolio total is positive; hides item-level failure |

Report the coverage: the share of evaluation periods that actually entered the calculation. A MAPE of 100% computed on 37.5% of
periods is a different claim from a MAPE of 100% computed on all of them, and the difference is invisible unless coverage is
printed next to the number.

### Phase D — Compare and report

**Step 10. Choose the aggregation and state it.**

Compute at the decision grain, then aggregate deliberately. If the decision is per item, compute per item and report the
distribution, not only the mean. If the decision is a budget across the portfolio, a volume-weighted total is defensible, but it
must be labelled as such.

**Step 11. Compare methods pairwise, per series.**

Comparing two methods on a single portfolio average discards the pairing. The correct comparison uses the per-series loss
difference and asks whether the differences are consistently one-sided. Two simple tools:

- **Sign test or Wilcoxon signed-rank test** on the per-series losses. This asks whether one method wins more often than chance,
  without assuming normality.
- **Paired bootstrap** over series, resampling items rather than periods, to get an interval for the mean difference in loss.

Both treat the item, not the item-week, as the unit of evidence. Item-weeks within an item are strongly dependent, and a test
that treats them as independent will report significance for trivial differences.

**Step 12. Translate into cost or service.**

Multiply the loss by the consequence. A unit of unmet demand on a cheap consumable and a unit of unmet demand on a critical
spare part are not the same unit, and no scale-free metric knows the difference. This step is where the evaluation becomes a
business case rather than a scoreboard.

**Step 13. Publish a metric contract.**

Five lines, printed with the number:

1. Grain: item × location × period, with the period defined.
2. Demand construct: consumption, shipments, or orders, and the censoring risk.
3. Zero rule: what happens at zero, and the resulting coverage.
4. Loss: the formula, the quantile if applicable, and the scale denominator.
5. Aggregation: micro, macro, median, or win rate, and the weights.

A number without those five lines is not a measurement. It is a rumour with a decimal point.

### 5.1 Choosing a denominator is a modelling decision

The denominator of MAPE is actual demand; of MASE, the naive error; of wMAPE, total demand. Each encodes a claim about what a
big error means: proportional to what was demanded, better than repetition, or a small share of volume. No denominator is
correct for all three claims. The mistake is not choosing one; it is choosing one without saying which claim it encodes.

## 6. Worked example

An industrial distributor stocks seven items at one warehouse. Demand is recorded weekly in eaches over twelve weeks. Item Z1 is
a new item with no demand yet; it is included because the all-zero series is the case that breaks most scorecards, and
pretending it does not exist is the usual response.

### 6.1 The demand history

| Week | S1 | E1 | I1 | I2 | L1 | L2 | Z1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10 | 2 | 0 | 0 | 0 | 0 | 0 |
| 2 | 12 | 20 | 0 | 3 | 0 | 1 | 0 |
| 3 | 9 | 3 | 5 | 0 | 12 | 0 | 0 |
| 4 | 11 | 22 | 0 | 0 | 0 | 0 | 0 |
| 5 | 10 | 4 | 0 | 4 | 0 | 2 | 0 |
| 6 | 13 | 18 | 6 | 0 | 1 | 0 | 0 |
| 7 | 9 | 2 | 0 | 0 | 0 | 15 | 0 |
| 8 | 12 | 24 | 0 | 3 | 0 | 0 | 0 |
| 9 | 11 | 5 | 5 | 0 | 20 | 0 | 0 |
| 10 | 10 | 16 | 0 | 0 | 0 | 1 | 0 |
| 11 | 12 | 3 | 0 | 3 | 0 | 0 | 0 |
| 12 | 9 | 21 | 5 | 0 | 2 | 0 | 0 |

All quantities are eaches per item per week. The six active items were chosen to populate the four quadrants of the
classification rather than to represent a realistic mix; a real catalogue is dominated by the intermittent and lumpy quadrants.

### 6.2 Classification

| Item | Demand weeks | ADI (periods) | CV² of sizes | Class | Total demand (eaches) |
|---|---:|---:|---:|---|---:|
| S1 | 12 | 1.00 | 0.015 | smooth | 128 |
| E1 | 12 | 1.00 | 0.560 | erratic | 140 |
| I1 | 4 | 3.00 | 0.007 | intermittent | 21 |
| I2 | 4 | 3.00 | 0.018 | intermittent | 13 |
| L1 | 4 | 3.00 | 0.793 | lumpy | 35 |
| L2 | 4 | 3.00 | 1.560 | lumpy | 19 |
| Z1 | 0 | infinite | undefined | no demand | 0 |

Two things to notice. First, the ADI of 3.00 for four items means demand occurs once every three weeks on average, which is mild
intermittency; many real spare-parts series run at ADI 10 or higher. Second, I1 and L1 have identical ADI and very different
CV², which is exactly the distinction the grid is for: I1's demand is regular in size, L1's is not.

### 6.3 Three methods compared

Three one-week-ahead forecasts are produced for each item, using only data available before the forecast week. Evaluation covers
weeks 5 to 12, giving eight evaluation points per item.

- **Method A, naive carry-forward.** The forecast for week t is the observed demand in week t−1. This is the standard naive
  benchmark.
- **Method B, rate estimate.** The forecast for week t is the share of nonzero weeks in the history multiplied by the mean of
  the nonzero sizes in the history. This is a simple occurrence-times-size estimator in the spirit of Croston's method; it omits
  the SBA bias correction, which is stated as a simplification rather than a recommendation.
- **Method Z, do nothing.** The forecast is zero every week. It corresponds to holding no stock and ordering nothing. It is
  included because it is the forecast that accuracy metrics on intermittent demand are most likely to reward, and because no
  planning director would accept it.

For I1, Method B's forecasts develop as follows:

| Week | History nonzero sizes | Occurrence share | Mean size | Forecast B | Actual |
|---:|---|---:|---:|---:|---:|
| 5 | 5 | 1/4 = 0.250 | 5.00 | 1.25 | 0 |
| 6 | 5 | 1/5 = 0.200 | 5.00 | 1.00 | 6 |
| 7 | 5, 6 | 2/6 = 0.333 | 5.50 | 1.83 | 0 |
| 8 | 5, 6 | 2/7 = 0.286 | 5.50 | 1.57 | 0 |
| 9 | 5, 6 | 2/8 = 0.250 | 5.50 | 1.38 | 5 |
| 10 | 5, 6, 5 | 3/9 = 0.333 | 5.33 | 1.78 | 0 |
| 11 | 5, 6, 5 | 3/10 = 0.300 | 5.33 | 1.60 | 0 |
| 12 | 5, 6, 5 | 3/11 = 0.273 | 5.33 | 1.45 | 5 |

Method B never forecasts more than 1.83 units on an item whose demand, when it comes, is five or six. That is the correct
expected value per week and a hopeless order quantity for any individual week. This gap between the conditional mean and the
quantity a planner needs is the whole argument for quantile and cost-based evaluation.

### 6.4 Accuracy results

| Item | MAE A | MAE B | MAE Z | MAPE A | MAPE B | MAPE Z | MASE A | MASE B | MASE Z |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S1 | 2.25 | 1.31 | 10.75 | 21.5 | 12.3 | 100.0 | 0.990 | 0.578 | 4.73 |
| E1 | 16.38 | 8.68 | 11.63 | 298.4 | 159.1 | 100.0 | 0.974 | 0.516 | 0.691 |
| I1 | 3.38 | 2.53 | 2.00 | 100.0 | 75.6 | 100.0 | 1.003 | 0.751 | 0.595 |
| I2 | 2.50 | 1.67 | 1.25 | 100.0 | 71.5 | 100.0 | 1.058 | 0.707 | 0.529 |
| L1 | 5.50 | 4.35 | 2.88 | 100.0 | 94.0 | 100.0 | 0.890 | 0.703 | 0.465 |
| L2 | 4.50 | 3.02 | 2.25 | 100.0 | 88.2 | 100.0 | 1.303 | 0.876 | 0.651 |

MAPE is computed only on weeks where the actual is nonzero. Coverage is 100% for S1 and E1, and 37.5% (three weeks of eight) for
I1, I2, L1 and L2. MAE and MASE use all eight weeks.

Read the MAPE column first, because it is the column a dashboard would show.

- E1, the item with demand every week and no zeros, has the worst MAPE in the table at 298.4% under Method A. Its small actuals
  of 2 and 3 produce errors of 600% to 900%; the denominator is doing most of the work, not the model.
- Four items report exactly 100.0%, because a zero forecast with any positive actual always scores 1. A five-unit miss and a
  twenty-unit miss score identically.
- The do-nothing forecast scores 100.0% on every item. On E1 it beats the naive method by 198 points; on S1 it is five times
  worse. A metric that ranks a zero forecast above a real method on any item is describing its denominator.

### 6.5 Service results

To connect forecast to service, treat each forecast as an order-up-to level for a zero-lead-time, weekly-review policy. Stock on
hand at the start of the week equals the rounded forecast, demand is served from it, and unmet demand is lost. This is a
teaching device, not a realistic inventory model: real lead times are longer than zero and real policies are not set to the
forecast itself. It isolates the direction of the effect.

| Item | Fill rate A | Fill rate B | Fill rate Z | Stockout weeks A | Stockout weeks B | Stockout weeks Z |
|---|---:|---:|---:|---:|---:|---:|
| S1 | 90.7% | 94.2% | 0.0% | 3/8 | 3/8 | 8/8 |
| E1 | 30.1% | 60.2% | 0.0% | 4/8 | 4/8 | 8/8 |
| I1 | 0.0% | 18.8% | 0.0% | 3/8 | 3/8 | 3/8 |
| I2 | 0.0% | 30.0% | 0.0% | 3/8 | 3/8 | 3/8 |
| L1 | 0.0% | 21.7% | 0.0% | 3/8 | 1/8 | 3/8 |
| L2 | 0.0% | 5.6% | 0.0% | 3/8 | 2/8 | 3/8 |

Fill rate is the fraction of demand units served. Stockout weeks count weeks in which some demand went unmet.

Method Z has a fill rate of zero on every item and a stockout opportunity rate of 100% on every demand week. It is also the
second-best method on pooled MAPE and the best method on median MASE. That combination is the point of the example.

Method A produces a zero forecast on most weeks for the slow items, because the previous week was usually zero. It therefore
orders nothing in precisely the weeks when stock is needed. Its fill rate on the four slow items is zero. A planner looking at
MAPE would see 100% and conclude that the forecast is poor but not catastrophic. A planner looking at fill rate would see that
the item never ships.

### 6.6 The zero rule changes the number

Take item I1 under Method A. The eight evaluation weeks have actuals 0, 6, 0, 0, 5, 0, 0, 5 and forecasts 0, 0, 6, 0, 0, 5, 0,
0. The absolute errors are 0, 6, 6, 0, 5, 5, 0, 5, summing to 27 against total demand of 16.

| Zero rule | Computation | Result |
|---|---|---:|
| Exclude zero actuals | Three demand weeks, each with 100% error | 100.0% |
| Replace actual zero with 1 | Errors of 0, 6, 6, 0, 5, 5, 0, 5 over denominators 1, 6, 1, 1, 5, 1, 1, 5 | 175.0% |
| Add one to both | Sum of abs(e)/(y+1), divided by 8 | 169.0% |
| Aggregate first (wMAPE) | 27 / 16 | 168.8% |

The same forecasts, the same outcomes, four published numbers between 100% and 175%, and a coverage figure of 37.5% that none of
them displays. If the zero rule is chosen after seeing the results, the choice is worth 75 percentage points on a single item.

### 6.7 Portfolio aggregation

| Method | Pooled wMAPE | Mean of item MAPE | Mean MASE | Median MASE | Volume-weighted fill |
|---|---:|---:|---:|---:|---:|
| A, naive | 112.2% | 120.0% | 1.036 | 0.997 | 43.1% |
| B, rate | 70.1% | 83.5% | 0.689 | 0.705 | 60.6% |
| Z, do nothing | 100.0% | 100.0% | 1.277 | 0.623 | 0.0% |

Five aggregation rules, three methods, and no stable ordering except at the bottom: Method Z is last on service by construction.

- **Pooled wMAPE** ranks B first, Z second, A third. It is dominated by E1, which contributes 93 of the 246 units of demand.
- **Mean of item MAPE** gives the same order, but for a different reason: each item counts once, and the four slow items all sit
  at exactly 100%.
- **Mean MASE** ranks A second and Z last. Z is dragged down by S1, where a zero forecast against a stable series produces a
  MASE of 4.73.
- **Median MASE** reverses that: Z is first, because on five of six items the zero forecast beats the naive one, and the median
  ignores the single fast mover.
- **Volume-weighted fill** ranks B first and puts Z last with nothing between.

The mean-versus-median MASE reversal is not a statistical curiosity. It is the difference between asking "how does the average
item fare?" and "how does the typical item fare?" In a catalogue where most items are slow, those are different questions, and
the second one is usually the one the planner means.

### 6.8 What the example establishes

Four findings survive the arithmetic. Percentage error saturates at 100% when the forecast is zero and explodes into the
hundreds when the actual is small; both are properties of the ratio. The do-nothing forecast is second on pooled wMAPE, tied
first on mean MAPE, and first on median MASE, while last on service at zero. The ranking depends on the aggregation rule, and
mean and median MASE disagree about the naive and zero methods while both are computed correctly. Service separates the methods
that accuracy cannot separate, and it does so on every item.

## 7. Code, query, or calculation

The snippet below computes the classification and two zero-safe metrics without any third-party library. It is deliberately
short: the arithmetic, not the tooling, is the subject. The function `mase` uses the in-sample naive mean absolute error as its
scale, which is the Hyndman–Koehler definition with a one-step horizon.

```python
from statistics import mean, pstdev

def classify(y, adi_cut=1.32, cv2_cut=0.49):
    nz = [v for v in y if v > 0]
    adi = len(y) / len(nz) if nz else float("inf")
    cv2 = (pstdev(nz) / mean(nz)) ** 2 if len(nz) > 1 else float("nan")
    if adi < adi_cut:
        return ("smooth" if cv2 < cv2_cut else "erratic"), adi, cv2
    return ("intermittent" if cv2 < cv2_cut else "lumpy"), adi, cv2

def mase(y, f):
    scale = mean(abs(b - a) for a, b in zip(y, y[1:]))   # in-sample naive MAE
    if scale == 0:
        return float("inf")                               # constant series
    return mean(abs(a - b) for a, b in zip(y, f)) / scale

def wmape(y, f):
    total = sum(y)
    return sum(abs(a - b) for a, b in zip(y, f)) / total if total else float("nan")
```

Called on the I1 series from Section 6, `classify` returns `('intermittent', 3.0, 0.0068)`. Note that `mase` returns infinity
for a constant series, because the naive forecast has zero error and the ratio is undefined. That is a real limit of the
measure, not a bug to be patched with a small constant. Report the case separately.
## 8. Data, standards, and regulatory context

Forecast metrics are not regulated in the way that, say, revenue recognition is. That absence matters, because it means the
definitions are conventions chosen by teams rather than requirements imposed from outside. The standards that do apply constrain
the inputs and the contractual outputs, not the error measure itself.

### 8.1 Identity and event standards

A metric computed at item × location × period requires that items and locations have stable identifiers, and that demand events
carry a time and a place.

| Standard | Governs | What it constrains for evaluation | Grade |
|---|---|---|---|
| GS1 GTIN (Global Trade Item Number) | Product identity | The item dimension of the series; a change of pack size is a new GTIN, which breaks comparability across a re-packing | B |
| GS1 GLN (Global Location Number) | Party and location identity | The location dimension; a warehouse re-code creates a false zero history | B |
| GS1 SSCC (Serial Shipping Container Code) | Logistics unit identity | Reconstructing receipts and shipments when demand must be derived | B |
| EPCIS 2.0 | Event data: what, where, when, why | Timestamped movement and transformation events; the cleanest available basis for consumption-based demand | B |
| GS1 Digital Link | Product identification at item level | Traceability to the unit, not usually the demand grain used in planning | B |

The practical consequence is that a re-packing, a warehouse re-code, or a change of unit of measure silently creates a new
series. The metric computed across the break is a mixture of two items, and its history is not a history. Topic 01 of this
module treats the unit hierarchy; Topic 07 treats the product and location hierarchies that make the join possible.

### 8.2 Metrology standards and the gap

| Standard | Scope | Relationship to forecast accuracy | Grade |
|---|---|---|---|
| ISO 5725-1:2023 | Accuracy (trueness and precision) of measurement methods and results | Defines accuracy for a measurement instrument; does not define forecast accuracy | B |
| ISO 8000 series | Data quality | Constrains completeness and consistency of the inputs, not the error measure | B |
| JCGM 200:2008 (VIM) | International vocabulary of metrology | Supplies precise meanings of accuracy, precision, trueness, and bias that forecast practice uses loosely | B |

There is no ISO, ANSI, or IEC standard that defines MAPE, MASE, or fill rate. The de facto references are the forecasting
literature: Hyndman and Koehler (2006) for scaled errors, and the inventory-forecasting literature for the intermittent-demand
measures. A vendor claim that a metric is "standard" should be read as "common", and dated.

> **Note:** Because no external standard fixes the definitions, two teams in the same company can
> both report "forecast accuracy" correctly and disagree by tens of percentage points. The fix is
> a published metric dictionary, not a better model. This is the same lesson as Topic 08 on
> measurement error, applied to a derived quantity rather than a physical one.

### 8.3 Contractual service definitions

Where service is contractual, the definition moves into the contract, and the contract usually does not use the forecasting
literature's vocabulary.

| Instrument | What it typically fixes | Consequence for evaluation | Grade |
|---|---|---|---|
| Service level agreement (SLA) | Fill rate, on-time in-full, order completeness, measurement window and exclusions | The metric must be computed exactly as the contract states, including exclusions | B |
| Supplier scorecards | Delivery reliability and quality thresholds | A supplier's measured service is not your item's demand, and the two must not be substituted | B |
| Internal planning targets | Target fill rate or CSL by item class | A target is not a measurement; a 95% target says nothing about achieved service | C |

The evaluation trap is to compute an internal fill rate, compare it to a contractual target, and discover only at audit that the
contract excludes cancelled lines, intercompany shipments, or substitutions. The metric is arithmetically correct and
contractually wrong.

### 8.4 The fields a metric needs

Most scorecards fail because a required field was never captured. The minimum set for the evaluation described in this topic:

| Field | Why it is needed | Common absence |
|---|---|---|
| Item identifier | Series identity across pack changes | Re-used codes for retired and new items |
| Location identifier | Series identity across re-codes | Warehouse codes recycled after closure |
| Period key | Bucket alignment and dense zeros | Local dates without a time zone or calendar |
| Demand quantity | Numerator and denominator | Only shipment quantity captured |
| Unit of measure | Comparability and conversion | Each and case mixed in one column |
| Demand type or status | Separating firm demand, forecast, and promotional volume | Everything recorded as "sales" |
| Stock-out or availability flag | Detecting censored demand | Not captured, or captured at order level only |
| Forecast value and version | The thing being evaluated | Only the latest forecast retained |
| Forecast timestamp and horizon | Preventing leakage and mixing horizons | Overwritten each planning cycle |

The last two rows are the most consequential and the least often fixed. If the forecast is overwritten every week, a backtest
cannot reconstruct what was known at the time, and any evaluation is a fit to history dressed as a forecast.

### 8.5 Convention rather than requirement

Reporting accuracy at 95%, the 80/15/5 ABC thresholds, the 1.32 and 0.49 cut-offs, and a MAPE gate of 10% are conventions, not
requirements. An analyst who can tell them apart can challenge them; one who cannot will pursue numbers the data cannot produce.

## 9. Industry practice and case evidence

### 9.1 The classification evidence

Syntetos, Boylan and Croston (2005), "On the categorization of demand patterns", *Journal of the Operational Research Society*
56(5), 495–503, is the source of the ADI × CV² framework used in Section 3. The paper argues that the arbitrary categories built
into inventory software are inferior to rules derived from forecast-error performance, compares exponential smoothing, Croston's
method, and the Syntetos–Boylan approximation by theoretical mean squared error, and tests the resulting rules on 3,000 real
intermittent series from the automotive industry. Grade A for the framework. The specific cut-offs are commonly quoted as 1.32
and 0.49; the search material available during preparation did not include the full text, so the constants should be verified
against the paper before being written into a policy.

### 9.2 The bias of Croston-type methods

The bias literature matters here because it shows how a metric and a method interact.

| Finding | Source | Grade | Consequence for evaluation |
|---|---|---|---|
| Croston's method has a positive bias, traced to treating the expectation of a reciprocal as the reciprocal of an expectation | Syntetos and Boylan (2001), *International Journal of Production Economics* 71, 457–466 | A | A positively biased rate over-orders; MAPE may still reward it |
| The SBA correction over-compensates and can produce a negative bias larger than Croston's positive bias | Teunter and Sani (2009), *European Journal of Operational Research* 194(1), 177–183 | A | Unbiasedness is not guaranteed by a published correction |
| A modification retaining the neglected term reports about 1% average absolute bias against about 5% for SBA and 6% for Croston in the study's simulation design | Teunter and Sani (2009) | A | Bias must be measured on your data, not assumed from the method name |
| Croston's method does not update during zero-demand periods, which creates an obsolescence problem, and is positively biased at all points in time | Teunter, Syntetos and Babai (2011), *EJOR* 214(3), 606–615 | A | A long run of zeros should reduce the forecast; Croston's does not |
| SBA remains preferable on mean squared error while the bias-corrected variant wins on average absolute bias | Teunter and Sani (2009) | A | The "best" method depends on the loss, which is the theme of this topic |

The lesson is not that one method is best. It is that a method's bias is a property that shows up in the mean error and in the
service outcome, and may be invisible in a percentage accuracy measure because the percentage measure is itself biased in the
same direction.

### 9.3 Accuracy and service

The evidence that forecast accuracy and inventory service are weakly related is now substantial, and it is strongest precisely
for intermittent and lumpy demand.

| Study | Design | Finding | Grade |
|---|---|---|---|
| Syntetos, Nikolopoulos and Boylan (2010), *International Journal of Forecasting* 26(1), 134–143 | Conceptual and empirical review of accuracy versus inventory implications | Inventory system efficiency does not relate directly to standard forecast accuracy; accuracy-implication metrics are needed | A |
| Solis (2015), MAS proceedings | Lumpy demand, (T, S) policy, neural network versus exponential smoothing and SBA | The statistically more accurate forecasts required significantly higher on-hand inventory to reach 90% and 95% fill rates | C |
| Theodorou, Spiliotis and Assimakopoulos (2025), *EJOR* 322(2), 414–426 | M5 competition data, accuracy versus inventory performance | The relationship is often weak and depends on demand pattern, policy, and cost structure; accuracy matters more when holding cost is at least as large as lost-sales cost | A |
| Chin, Cheng and Gunawan (2026), preprint, arXiv:2609.13840 | 38 methods on intermittent industrial demand | Reports that accuracy rank and order-service rank can be negatively correlated under heavy intermittency, and that service tracks bias direction more than point accuracy | C |

The last row is a preprint and should be treated as a lead rather than a settled result. The first three are enough to establish
the practical point: a method ranked first on accuracy is not thereby ranked first on service, and on lumpy demand it can be
ranked last.

### 9.4 Portfolio practice and forecast value added

**Forecast value added (FVA)** attributes the change in an error measure to a process step, relative to a naive baseline.
Gilliland's (2013) stairstep presentation shows naive, statistical model, and override, with the change at each step in the
chosen metric. Fildes, Goodwin and De Baets (2024) pool about 147,000 forecasts from six studies and report that judgemental
adjustments improved accuracy or bias for only just over half of SKUs, with large negative adjustments more likely to help than
positive ones. Goodwin, Petropoulos and Hyndman (2017) derive upper bounds on achievable FVA and cite Morlidge's finding that
more than half of over 300,000 forecasts across eight companies failed to beat a naive benchmark. If the naive benchmark is a
zero forecast, MASE and FVA are measured against a baseline that is already near-optimal for the metric, and the apparent value
added of a real method collapses.

### 9.5 What practitioners report, and what it means

| Reported number | Typical computation | What it actually says |
|---|---|---|
| "Forecast accuracy 85%" | 100% minus MAPE, or 100% minus wMAPE, at some aggregation | Depends entirely on the zero rule and the aggregation; not comparable across companies |
| "Fill rate 96%" | Units served divided by units demanded, exclusions unstated | May exclude cancelled lines, substitutes, or intercompany demand |
| "Forecast bias 2%" | Mean error divided by mean demand | Sign and denominator rarely stated; a 2% bias can be 2% under or over |
| "95% of SKUs forecast within tolerance" | Share of items below an error threshold | Tolerance chosen post hoc; zero-demand items may be excluded |
| "AI improved accuracy by 12 points" | Vendor case study, baseline unstated | Grade D until the baseline, the metric, and the evaluation window are published |

The last row is the one to date and grade. Vendor accuracy claims are marketing until the metric contract in Section 5, Step 13,
is supplied. A claim of a twelve-point improvement is not a finding; it is a hypothesis with a number attached.

### 9.6 The 2026 technology context

Planning suites now advertise quantile forecasting as standard, and some correctly describe the reorder point as a quantile
forecast. Lokad's practitioner material, accessed in 2026, states that native quantile forecasts are preferable when service
exceeds 90%, demand is intermittent, or orders are bulky, and reports that a 1% reduction in pinball loss yields roughly a 0.5%
to 1% reduction in safety stock. Grade D: vendor material, dated 2026, with no published methodology. The underlying claim that
pinball loss is the right loss for a quantile forecast is Grade A and comes from the forecasting literature. The academic
evidence says the accuracy-to-inventory mapping is weak; the vendor says a one-point loss improvement buys half a point of
safety stock. Both can be true, because they concern different policies.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| MAPE computed with zero actuals dropped | Coverage is never printed; item count in the metric is lower than item count in the portfolio | The undefined ratio is removed rather than resolved | The number describes demand weeks only, and slow items are scored on a handful of observations | State the zero rule; print coverage next to every percentage |
| Zero actual replaced by 1 or by the mean | Two teams report different MAPE for the same item | The replacement is a free parameter chosen locally | Numbers move by tens of points with no change in forecasts | Ban silent substitution; use wMAPE or a scaled measure instead |
| Small denominators dominate | A handful of items account for most of the portfolio error | The ratio is convex in the denominator | Effort flows to trivial items; the metric is noisy | Stratify by volume class; report per-class |
| Under-forecast bias rewarded | Mean error is persistently negative while MAPE improves | Percentage error penalises over-forecasting more than under-forecasting | Service silently deteriorates as accuracy improves | Report mean error alongside the accuracy measure |
| Zero forecast exits the scorecard | An item has no accuracy value for several periods | Division by zero returns blank, and blanks are excluded from averages | The worst possible policy improves the reported average | Treat a blank as a failure, not a missing value; report the count |
| Zero forecast competes on accuracy | A trivial baseline ranks mid-table or better | MAE and MASE are minimised by the median, which is zero when demand is sparse | A do-nothing policy looks defensible | Always include the zero forecast as a benchmark and a service column beside it |
| Mean MASE dominated by one fast mover | One item's MASE is an order of magnitude above the rest | Unweighted mean is sensitive to extremes | The ranking describes one item, not the portfolio | Report median, win rate, and volume-weighted measures together |
| Median MASE flatters the zero forecast | The zero method wins the median while losing service | The median ignores the item where the method fails catastrophically | A policy that never stocks is selected | Pair every accuracy statistic with a cost or service measure |
| Pooled metric hides item-level failure | Portfolio wMAPE falls while a key item's fill rate falls | Large items dominate the numerator and denominator | The aggregate improves while the commercially important item degrades | Report at the decision grain and aggregate afterwards |
| Evaluation grain differs from decision grain | Accuracy is reported monthly, orders are placed weekly | Aggregation smooths the errors the policy actually faces | The metric is optimistic relative to the decision | Evaluate at the replenishment period |
| Naive denominator is zero or tiny | MASE is infinite or in the hundreds | The series is constant, or nearly so | The metric is undefined or dominated by numerical noise | Detect constant series; use RMSSE or report separately |
| Demand is censored by stock-outs | Recorded demand is exactly equal to recorded shipments in stock-out weeks | Unmet demand is unobservable | The model learns to reproduce the stock-out | Flag availability; reconstruct or bound demand |
| Units of measure differ across items | Some items in eaches, others in cases | Scale-free metrics hide it, scale-dependent metrics do not | A case-versus-each error is silently a factor of twelve | Normalise to a base unit before evaluation |
| Metric chosen after seeing results | The reported measure changes between review cycles | Selection on the outcome | The apparent improvement is the choice, not the method | Pre-register the metric contract |
| Accuracy rank used to set safety stock | The item with the best MASE has the worst fill rate | Accuracy and service are not monotonically related | Inventory is allocated to the wrong items | Set policy from the cost critical ratio and the predictive distribution |

Two deserve expansion. **A zero forecast exits the scorecard** because a blank MAPE is excluded from portfolio averages, so the
worst possible policy improves the reported number; a blank in an accuracy report should be a data-quality incident, and its
count a reported KPI. **Under-forecast bias is rewarded** because MAPE's asymmetry is small per observation and large in
aggregate, so a search over many models finds the ones that sit low; mean error, reported with its sign, is the cheapest
defence.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Recorded demand equals true demand | Every accuracy metric compares forecast to an observed outcome | Stock-outs censor demand; promotions pull it forward; substitutes absorb it | Compare demand with availability flags; look for demand capped at stock |
| A zero is an observation, not a missing value | The classification and every zero-counting metric depend on it | Item not ranged at a location; data feed down; item discontinued | Check the dense grid against the assortment master |
| The occurrence process is stationary over the window | ADI and CV² are summary statistics of a stable process | New listing, end of life, seasonality, competitor exit | Plot occurrence rate by quarter; test for a break |
| ADI and CV² are computed on one bucket | Classification is bucket-dependent | Daily series classified and then managed weekly | State the bucket; reclassify after any change |
| Demand sizes are independent of the intervals | The occurrence-times-size decomposition assumes separability | Large orders arrive less frequently; emergency orders are small | Plot size against interval |
| The naive benchmark is a fair floor | MASE, RMSSE, and FVA are ratios to it | Seasonal data, trending data, or a constant series | Use seasonal naive where seasonality is present; detect constant series |
| The loss matches the decision | The metric selects the forecast that the decision will use | Replenishment needs a quantile; the metric is a mean or median | Derive the critical ratio and compare it with the metric's optimum |
| Cost parameters are known and stable | Cost-based and newsvendor metrics require them | Lost margin varies by customer; obsolescence varies by item | Segment the critical ratio; test sensitivity |
| The horizon is one period | MASE and most scaled measures are defined for one-step errors | Multi-step planning; lead-time demand | Use the multi-step scaled error or aggregate to lead-time buckets |
| Series are long enough to estimate scale | MASE divides by an in-sample naive error | Short histories; new items | Require a minimum number of nonzero observations; report the count |
| Units of measure are comparable after conversion | Scale-free metrics still require a common meaning | Case pack changes; unit hierarchy errors | Audit conversions against the master data |
| Forecasts use only information available at the time | Backtests must reproduce the decision | Forecasts overwritten; parameters tuned on the evaluation window | Retain forecast versions and timestamps; use a rolling origin |
| The portfolio definition is stable | Comparing two periods requires the same item set | Items added or retired between periods | Report on a fixed panel, and separately on the full set |

Three limits deserve statement. **No metric recovers unobserved demand:** if stock ran out, the demand that was not served is
absent, and the bias is largest where the stakes are highest. **The classification is descriptive, not causal:** ADI and CV²
describe a history, not why it is intermittent or whether the pattern will persist. **Accuracy comparisons are only as good as
the evaluation design:** a method tuned on the evaluation window will beat one that was not, so rolling-origin evaluation and
retained forecast versions are not optional refinements.

## 12. Exercises

### Exercise 1 — Classify the series (application, 45 minutes)

Classify each of the following ten-week series using ADI and CV². State the bucket, the numbers, and the observation window. All
quantities are eaches per week.

| Series | Weekly demand, weeks 1 to 10 |
|---|---|
| a | 4, 5, 4, 6, 5, 4, 5, 6, 4, 5 |
| b | 1, 9, 2, 11, 1, 8, 2, 10, 1, 9 |
| c | 0, 0, 6, 0, 0, 7, 0, 0, 6, 0 |
| d | 0, 0, 9, 0, 0, 1, 0, 0, 18, 0 |
| e | 0, 7, 0, 0, 0, 8, 0, 0, 0, 7 |
| f | 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 |

**Deliverable:** a six-row table with nonzero count, ADI, CV², class, and one sentence per series on how you would expect MAPE
to behave.

**Acceptance criteria:** formulas shown; CV² computed on nonzero sizes only; series f handled explicitly rather than dropped;
units stated for ADI.

**Time estimate:** 45 minutes.

### Exercise 2 — Percentage-error arithmetic (calculation, 30 minutes)

1. Compute the percentage error for an absolute error of one unit when the actual is 1, 10, 100, and 1,000.
2. Compute the absolute error that corresponds to a 20% percentage error when the actual is 2, 20, and 200.
3. Compare the percentage error of over-forecasting by 50 units against an actual of 100 with under-forecasting by 50 units
   against an actual of 150.
4. Explain what happens to each of the four values in part 3 if the actual is zero.

**Deliverable:** a table of values and a short paragraph on the direction of the bias.

**Acceptance criteria:** all arithmetic shown; the asymmetry stated in the correct direction; the zero case described as
undefined rather than as a large number.

**Time estimate:** 30 minutes.

### Exercise 3 — Zero-policy audit (analysis, 40 minutes)

A colleague reports a MAPE of 96% for a spare-parts portfolio. You discover that periods with zero demand were excluded, that
four items with no demand in the quarter were dropped entirely, and that the portfolio contains 1,200 item-location series.

**Deliverable:** a one-page audit listing the four zero-handling rules, the direction of the effect of each on the reported 96%,
and the three questions you would ask before using the number.

**Acceptance criteria:** each rule's effect is signed, not merely described; the dropped items are identified as a selection
effect rather than a data-quality problem; the recommendation is specific and does not require rebuilding the forecasting
system.

**Time estimate:** 40 minutes.

### Exercise 4 — Write a metric contract (design, 50 minutes)

Choose a forecast accuracy report you have seen or can construct from public material. Write the five-line metric contract from
Section 5, Step 13, for it, and identify which lines were previously unstated.

**Deliverable:** the contract, plus a short note on what would change in the reported number if each unstated line were varied.

**Acceptance criteria:** all five lines present; the grain and the bucket are named explicitly; the zero rule produces a stated
coverage; at least one line is identified as materially changing the result.

**Time estimate:** 50 minutes.

### Exercise 5 — The portfolio ranking that flips (analysis, 60 minutes)

Two methods have been evaluated on five items. The table gives total demand over the evaluation window, the MAE of each method
in eaches, and the MAE of the naive benchmark.

| Item | Total demand | MAE A | MAE B | Naive MAE |
|---|---:|---:|---:|---:|
| P1 | 1,000 | 90 | 100 | 120 |
| P2 | 800 | 70 | 75 | 110 |
| P3 | 40 | 6 | 4 | 8 |
| P4 | 20 | 4 | 3 | 6 |
| P5 | 5 | 2 | 1.2 | 3 |

Compute the pooled wMAPE for each method, the mean MASE, the median MASE, and the item win rate. State which method you would
deploy and why.

**Deliverable:** a table of the four statistics, the arithmetic, and a recommendation with one sentence of justification.

**Acceptance criteria:** pooled wMAPE computed as a ratio of sums, not as a mean of ratios; MASE computed per item before
averaging; the recommendation names the decision and the weights implied by each statistic.

**Time estimate:** 60 minutes.

### Exercise 6 — Pinball loss by hand (calculation, 40 minutes)

Demand for a period realises at 7 units. Three candidate forecasts are 5, 7, and 9. Compute the pinball loss at τ = 0.9 and at τ
= 0.5 for each. Explain the ratio between the penalties for under- and over-forecasting at each level.

**Deliverable:** a six-cell table plus the two ratios.

**Acceptance criteria:** the correct branch of the loss used for under- and over-forecasting; the ratio at τ = 0.5 stated as 1;
one sentence connecting τ = 0.9 to a high service target.

**Time estimate:** 40 minutes.

### Exercise 7 — Fill rate against accuracy (simulation, 75 minutes)

Using any demand history with at least 20 intermittent series, compute MAE and MASE for two methods of your choice. Then apply
each method as an order-up-to level in a periodic-review simulation with a lead time of one period, and compute fill rate and
stockout opportunity rate.

**Deliverable:** a scatter of accuracy against fill rate, one point per item and method, and a rank correlation with its sign.

**Acceptance criteria:** the simulation policy is stated in full; demand is not censored by the simulated stock-outs when it is
being measured; the result is reported as a correlation, not as a causal claim.

**Time estimate:** 75 minutes.

### Exercise 8 — Scorecard design (design, 50 minutes)

Design a monthly forecast scorecard for a portfolio in which 65% of item-weeks are zero. It must resist improvement by
forecasting zero, by dropping zero periods, and by aggregating to a level where the denominator is large.

**Deliverable:** the scorecard specification: metrics, grain, aggregation, coverage reporting, and the benchmark included.

**Acceptance criteria:** at least one service or cost measure; coverage printed; a zero-forecast benchmark present; the
aggregation rule stated with its weights; no metric that is undefined at zero without a stated rule.

**Time estimate:** 50 minutes.

## 13. Self-check answers

### Exercise 1 arithmetic

```text
a: n = 10, positive = 10, ADI = 1.00, mean = 4.80, sd = 0.75, CV2 = 0.024  -> smooth
b: n = 10, positive = 10, ADI = 1.00, mean = 5.40, sd = 4.08, CV2 = 0.571  -> erratic
c: n = 10, positive = 3,  ADI = 3.33, mean = 6.33, sd = 0.47, CV2 = 0.006  -> intermittent
d: n = 10, positive = 3,  ADI = 3.33, mean = 9.33, sd = 6.94, CV2 = 0.554  -> lumpy
e: n = 10, positive = 3,  ADI = 3.33, mean = 7.33, sd = 0.47, CV2 = 0.004  -> intermittent
f: n = 10, positive = 0,  ADI = infinite, CV2 undefined                    -> no demand
```

Series c and e differ only in the size of the nonzero demands and the placement of the gaps, and both classify as intermittent
because the sizes are stable. Series d has the same ADI as c and e and a CV² two orders of magnitude larger; it is the lumpy
case, and it is the one where a single period can carry most of the quarter's volume.

On MAPE expectations: series a will report a modest percentage, in the low tens, because every denominator is around five.
Series b will report a very large percentage despite having no zeros, because the small actuals of 1 and 2 turn ordinary
absolute errors into hundreds of percent. Series c, d, and e will report values dominated by the three demand weeks, with
anything from about 50% to 100% depending on the forecast. Series f has no defined percentage error at all, and any tool that
returns a number for it is returning a convention, not a measurement.

### Exercise 2 arithmetic

Part 1, an absolute error of one unit:

| Actual | Percentage error |
|---:|---:|
| 1 | 100.0% |
| 10 | 10.0% |
| 100 | 1.0% |
| 1,000 | 0.1% |

Part 2, a 20% percentage error:

| Actual | Absolute error |
|---:|---:|
| 2 | 0.4 |
| 20 | 4.0 |
| 200 | 40.0 |

Part 3, an error of 50 units:

```text
over-forecast:  actual 100, forecast 150  ->  50 / 100 = 50.0%
under-forecast: actual 150, forecast 100  ->  50 / 150 = 33.3%
```

The same absolute error costs 16.7 percentage points less when the forecast is low. A procedure that minimises MAPE therefore
prefers to sit below the outcome. Under a symmetric error distribution, the MAPE-optimal point forecast is below the median, not
at it.

Part 4: with an actual of zero, every value is undefined. The four common conventions produce zero (if the forecast is also zero
and the ratio is treated as zero), infinity (if the forecast is positive), a blank (which silently removes the period), or a
finite number after substituting a denominator. None of these is a percentage error, and the choice must be stated.

### Exercise 5 arithmetic

```text
Total demand            = 1000 + 800 + 40 + 20 + 5 = 1,865

Pooled wMAPE A = (90 + 70 + 6 + 4 + 2) / 1,865 = 172 / 1,865 = 9.22%
Pooled wMAPE B = (100 + 75 + 4 + 3 + 1.2) / 1,865 = 183.2 / 1,865 = 9.82%

MASE per item, A:
  P1 90/120 = 0.750    P2 70/110 = 0.636    P3 6/8 = 0.750
  P4 4/6 = 0.667       P5 2/3 = 0.667
  mean = 0.694, median = 0.667

MASE per item, B:
  P1 100/120 = 0.833   P2 75/110 = 0.682    P3 4/8 = 0.500
  P4 3/6 = 0.500       P5 1.2/3 = 0.400
  mean = 0.583, median = 0.500
```

| Statistic | Method A | Method B | Winner |
|---|---:|---:|---|
| Pooled wMAPE | 9.22% | 9.82% | A |
| Mean MASE | 0.694 | 0.583 | B |
| Median MASE | 0.667 | 0.500 | B |
| Item win rate (MAE) | 2 of 5 | 3 of 5 | B |

The flip is real and it is not an arithmetic error. Method A is better by 0.6 percentage points on the volume-weighted total
because it is better on P1 and P2, which together are 96% of the demand. Method B is better on every small item, and the
equal-weight and median statistics count those items as heavily as the large ones.

The recommendation depends on the decision. If the scorecard governs a total inventory budget across the catalogue, the pooled
number is the relevant one and A wins narrowly. If the scorecard governs whether a given item is in stock when a customer asks,
B is better on three of five items and better on the typical item, and the pooled number is the wrong question. A defensible
answer states the decision, reports both statistics, and does not claim that one number settles it.

### Exercise 6 arithmetic

```text
Pinball loss at tau:
  L(y, F) = tau * (y - F)        if y >= F
          = (1 - tau) * (F - y)  if y < F

tau = 0.9, y = 7:
  F = 5:  0.9 * (7 - 5) = 1.80
  F = 7:  0.0
  F = 9:  0.1 * (9 - 7) = 0.20

tau = 0.5, y = 7:
  F = 5:  0.5 * 2 = 1.00
  F = 7:  0.0
  F = 9:  0.5 * 2 = 1.00
```

At τ = 0.9, under-forecasting by two units costs 1.80 and over-forecasting by two units costs 0.20, a ratio of 9 to 1. At τ =
0.5 the ratio is 1 to 1, which is why MAE is the median case of pinball loss. A service target of 90% is a statement that the
asymmetry should be 9 to 1; a model evaluated on MAE is being told that the asymmetry is 1 to 1, which is a different business.

### Exercise 3 rubric

A strong audit signs each rule's effect: excluding zeros raises the average when the excluded periods would have scored well,
and the direction cannot be known without computing both; dropping four no-demand items removes exactly the cases where the
metric is undefined; substituting one creates a large error on no-demand periods; aggregating first turns the metric into a
ratio of sums. The three questions concern coverage, the dropped items, and what forecast the metric would have preferred. A
weak audit lists the rules without signing their effects.

### Exercise 4 rubric

A strong contract names the grain, states that demand is consumption, states that zeros are observations with 100% coverage,
names pinball loss at the critical ratio with its cost inputs, and reports both volume-weighted and equal-weight aggregation. A
weak contract restates the formula and stops; a formula is not a contract, because it does not say what entered it.

### Exercise 7 rubric

A strong submission states the review period, lead time, order-up-to rule, treatment of unmet demand, and initial stock; reports
the correlation without claiming causation; and expects a weak or negative correlation for intermittent items and a positive one
for smooth items. A weak submission treats simulated sales as demand after stock-outs, censoring the errors it measures.

### Exercise 8 rubric

A strong scorecard has a service measure at the decision grain, signed bias, a scaled accuracy measure reported per item and as
a distribution, coverage, a zero-forecast benchmark, and an explicit rule that the zero forecast is excluded from deployment
regardless of rank. A weak scorecard has MAPE, an average, and a target, and can be improved by forecasting zero.

## 14. Interview questions

**1. "Why does MAPE fail on intermittent demand?"**

A strong answer gives three mechanisms: the denominator is zero on most periods, so the metric is undefined and the handling
rule silently changes the number; the denominator is small on many of the remaining periods, so ordinary absolute errors become
hundreds of percent; and the ratio is asymmetric, penalising over-forecasting more than under-forecasting. It adds that the
failure is structural, not a software defect. A weak answer says only that MAPE divides by zero.

**2. "What are ADI and CV², and what do the four quadrants mean?"**

A strong answer defines ADI as periods per demand and CV² as the squared coefficient of variation of nonzero demand sizes, notes
that CV² must be computed on sizes rather than the full series, and explains that the axes separate the occurrence problem from
the size problem. It names smooth, erratic, intermittent, and lumpy, and says that lumpy is where point forecasts fail most. A
weak answer recites the four names without saying what the axes measure.

**3. "If MAPE is broken, why not just switch to MASE?"**

A strong answer says MASE fixes scale and zero denominators but not the decision mismatch: it is a scaled MAE, so its optimum is
the median, and the median of an intermittent series is usually zero. It notes that MASE also has an undefined denominator for a
constant series, and that it should be reported as a diagnostic alongside a service or cost measure. A weak answer treats MASE
as the correct metric and stops there.

**4. "A vendor claims 95% forecast accuracy. What do you ask?"**

A strong answer asks for the metric, the zero rule, the aggregation and weights, the evaluation window and whether it was out of
sample, the item population and its demand regime, and the baseline. It grades the claim D until those are supplied and notes
that 95% of a portfolio that is mostly zeros may mean very little. A weak answer asks which algorithm they use.

**5. "How would you compare two methods across 10,000 SKUs where most weeks are zero?"**

A strong answer computes the loss per series, compares paired differences with a sign test or a paired bootstrap over items, and
reports pooled and equal-weight statistics plus a win rate. It includes a zero-forecast benchmark and a service measure, and it
checks that both methods were evaluated on the same rolling origin with the same information. A weak answer averages a portfolio
MAPE and declares a winner.

**6. "What is the difference between fill rate and cycle service level?"**

A strong answer says fill rate is the fraction of demand units served and cycle service level is the probability of no stock-out
in a replenishment cycle. It notes that a company can have 95% CSL and 80% fill if the stock-outs fall on its highest-volume
item, and that the two respond differently to forecast error. A weak answer treats them as two names for the same service
number.

**7. "Why can a more accurate forecast produce worse service?"**

A strong answer explains that accuracy rank does not determine service rank: the policy converts a forecast into an order, and a
method that is better on average can be worse in the tail that drives stock-outs. It cites the lumpy-demand evidence where
statistically more accurate methods needed more inventory to hit the same fill rate, and notes that bias direction can matter
more than dispersion. A weak answer says it cannot happen, or that the service measure must be wrong.

**8. "What is pinball loss and why would an inventory planner care?"**

A strong answer defines it as the asymmetric absolute loss minimised by the τ-quantile, connects τ to the critical ratio of
underage to overage cost, and notes that MAE is the special case at τ = 0.5. It explains that a reorder point is a quantile
forecast, so pinball is the loss that matches the decision. A weak answer describes it as "another error metric".

**9. "How do you handle zero actuals in a percentage metric?"**

A strong answer states that the ratio is undefined, lists the four common conventions, gives the direction of each, and
recommends either avoiding percentage metrics for intermittent data or using wMAPE with a stated coverage. It insists that the
rule be printed with the number. A weak answer says "add one" without acknowledging that the choice is arbitrary.

**10. "What is forecast value added, and what is its main failure mode?"**

A strong answer defines FVA as the change in an error measure attributable to a process step relative to a naive baseline,
describes the stairstep, and names the failure: the baseline is itself a forecast, and on intermittent demand a zero or
near-zero baseline can make a good method look worthless. It adds that judgemental overrides often fail to beat the baseline. A
weak answer defines FVA as "accuracy improvement".

## 15. Key terms

- **Intermittent demand** — demand that occurs in some periods and not others, so the series contains genuine zeros.
- **Lumpy demand** — intermittent demand whose nonzero sizes are also highly variable; the hardest quadrant to plan.
- **Erratic demand** — demand in most periods with a highly variable size; a variance problem rather than an occurrence problem.
- **ADI** — average inter-demand interval, in periods per demand occurrence.
- **CV²** — squared coefficient of variation of nonzero demand sizes; dimensionless.
- **Demand occurrence** — the binary event that demand is greater than zero in a period.
- **Demand size** — the quantity demanded in a period where demand occurred, excluding zero periods.
- **Croston's method** — separate exponential smoothing of occurrence intervals and sizes, combined as a rate.
- **SBA** — the Syntetos–Boylan approximation, a bias-corrected variant of Croston's method.
- **TSB** — the Teunter–Syntetos–Babai method, which updates occurrence probability in every period including zeros.
- **Forecast bias** — the mean signed error, forecast minus actual; direction matters and is often hidden.
- **MAPE** — mean absolute percentage error; undefined at zero actuals and asymmetric.
- **sMAPE** — symmetric MAPE; not bounded by 100% and not a solution to the zero problem.
- **wMAPE** — weighted MAPE, the ratio of summed absolute error to summed actual demand.
- **MASE** — mean absolute scaled error, scaled by the in-sample naive one-step MAE.
- **RMSSE** — root mean squared scaled error, the squared-error analogue of MASE.
- **Naive benchmark** — the forecast that repeats the last observation; the floor against which scaled measures are read.
- **FVA** — forecast value added, the change in a metric attributable to a process step over a naive baseline.
- **Fill rate** — the fraction of demand units served from stock.
- **Cycle service level** — the probability that a replenishment cycle ends without a stock-out.
- **Stockout opportunity rate** — the share of demand periods in which some demand went unmet.
- **Censored demand** — true demand that cannot be observed because stock was insufficient.
- **Pinball loss** — an asymmetric absolute loss minimised by a named quantile; MAE at τ = 0.5.
- **Critical ratio** — underage cost divided by underage plus overage cost; the quantile a newsvendor should target.
- **Quantile forecast** — a forecast of the value below which demand falls with a stated probability.
- **Proper scoring rule** — a loss minimised in expectation only by the true predictive distribution.
- **Evaluation grain** — the item, location, and period level at which the metric is computed.
- **Micro average** — aggregate numerators and denominators first, then form the ratio.
- **Macro average** — compute the ratio per series, then average the ratios.
- **Metric coverage** — the share of evaluation periods that enter the calculation after zero handling.

## 16. Sources

**Peer-reviewed and academic**

- Syntetos, A. A., Boylan, J. E., and Croston, J. D. (2005). "On the categorization of demand patterns." *Journal of the
  Operational Research Society*, 56(5), 495–503. DOI: 10.1057/palgrave.jors.2601841 (Grade A)
- Hyndman, R. J., and Koehler, A. B. (2006). "Another look at measures of forecast accuracy." *International Journal of
  Forecasting*, 22(4), 679–688. DOI: 10.1016/j.ijforecast.2006.03.001 (Grade A)
- Syntetos, A. A., and Boylan, J. E. (2001). "On the bias of intermittent demand estimates." *International Journal of
  Production Economics*, 71, 457–466. (Grade A)
- Teunter, R. H., and Sani, B. (2009). "On the bias of Croston's forecasting method." *European Journal of Operational
  Research*, 194(1), 177–183. DOI: 10.1016/j.ejor.2007.12.001 (Grade A)
- Teunter, R. H., Syntetos, A. A., and Babai, M. Z. (2011). "Intermittent demand: Linking forecasting to inventory
  obsolescence." *European Journal of Operational Research*, 214(3), 606–615. (Grade A)
- Syntetos, A. A., Nikolopoulos, K., and Boylan, J. E. (2010). "Judging the judges through accuracy-implication metrics: The
  case of inventory forecasting." *International Journal of Forecasting*, 26(1), 134–143. (Grade A)
- Theodorou, E., Spiliotis, E., and Assimakopoulos, V. (2025). "Forecast accuracy and inventory performance: Insights on their
  relationship from the M5 competition data." *European Journal of Operational Research*, 322(2), 414–426. (Grade A)
- Fildes, R., Goodwin, P., and De Baets, S. (2024). "Forecast value added in demand planning." *International Journal of
  Forecasting*. DOI: 10.1016/j.ijforecast.2024.07.006 (Grade A)
- Goodwin, P., Petropoulos, F., and Hyndman, R. J. (2017). "A note on upper bounds for forecast-value-added relative to naive
  forecasts." *Journal of the Operational Research Society*. DOI: 10.1057/s41274-017-0218-3 (Grade A)
- Kolassa, S. (2016). "Evaluating predictive count data distributions in retail sales forecasting." *International Journal of
  Forecasting*, 32(3), 788–803. DOI: 10.1016/j.ijforecast.2015.12.004 (Grade A)
- Kolassa, S. (2019). "Why the 'best' point forecast depends on the error or accuracy measure." *International Journal of
  Forecasting*. DOI: 10.1016/j.ijforecast.2019.02.017 (Grade A)
- Kolassa, S. (2023). "How we deal with zero actuals has a huge impact on the MAPE and optimal forecasts." *Foresight*, issue
  69, 13–16. (Grade B)
- Kolassa, S., and Martin, R. (2011). "Percentage errors can ruin your day (and rolling the dice shows how)." *Foresight*, issue
  23, 21–27. (Grade B)
- Davydenko, A., and Fildes, R. (2016). "Forecast error measures: Critical review and practical recommendations." Working paper
  and teaching text. (Grade A−)
- Bruzda, J. (2016). "Quantile forecasting in operational planning and inventory management: An initial empirical verification."
  *Dynamic Econometric Models*. DOI: 10.12775/dem.2016.001 (Grade A−)
- Kourentzes, N., Barrow, D., and Trapero, J. R. (2019). "Optimising forecasting models for inventory planning." Working paper.
  (Grade A−)

**Books, teaching material, and reference works**

- Hyndman, R. J., and Athanasopoulos, G. *Forecasting: Principles and Practice*, 3rd edition. Open access text; the chapters on
  forecast accuracy and intermittent demand. (Grade A−)
- *Forecasting: Theory and Practice*, open encyclopedia, Section 3.1 "Practice", including the forecast value added stairstep.
  https://forecasting-encyclopedia.com/practice.html (Grade B)
- Gilliland, M. (2013). "FVA: A reality check on forecasting practices." *Foresight*, issue 29. (Grade B)

**Standards and official**

- GS1 General Specifications, current edition, including GTIN, GLN, SSCC, and EPCIS 2.0. (Grade B)
- ISO 5725-1:2023, *Accuracy (trueness and precision) of measurement methods and results — Part 1: General principles and
  definitions*. (Grade B)
- ISO 8000 series, *Data quality*. (Grade B)
- JCGM 200:2008, *International Vocabulary of Metrology (VIM)*, 3rd edition. (Grade B)

**Industry, practitioner, and press**

- Svetunkov, I. (2024). "Avoid using MAPE!" OpenForecast. https://openforecast.org/2024/04/17/avoid-using-mape/ (Grade C)
- Hyndman, R. J. (2014). "Errors on percentage errors." Hyndsight. https://fpp.robjhyndman.com/hyndsight/smape/ (Grade C)
- Solis, A. (2015). "Better statistical forecast accuracy does not always lead to better inventory control efficiency: The case
  of lumpy demand." *MAS Proceedings*. (Grade C)
- Chin, C., Cheng, T., and Gunawan, A. (2026). "Accuracy is not service: A decision-aware benchmark for intermittent-demand
  forecasting." Preprint, arXiv:2609.13840. (Grade C)
- Vandeput, N. (2021). "Forecast KPI: Accuracy of a product portfolio." Towards Data Science. (Grade D)

**Vendor material**

- Lokad. "Reorder point (supply chain)." Vendor knowledge base, accessed 2026. Claims that native quantile forecasts reduce
  reorder-point error and that a 1% pinball-loss reduction yields roughly a 0.5–1% safety-stock reduction. (Grade D, dated 2026)

**Note on grading and on two uncertain constants.** Grades follow Module 28: A peer-reviewed or audited, B regulatory or
official, C company case, industry study, or preprint, D vendor marketing, E press or anecdote. Working papers and teaching
texts are A−. Vendor figures are reported as the vendor states them and are not independently verified.

Two items were not fully verifiable from the sources available during preparation. First, the ADI and CV² cut-offs of 1.32 and
0.49 are the values most commonly attributed to Syntetos, Boylan and Croston (2005); the framework and the 3,000-series
automotive sample are confirmed, but the full text was not available to confirm the constants. Second, the bias percentages
attributed to Teunter and Sani (2009) are as reported in the accessible working-paper version and should be checked against the
published article before quotation. Both are presented as conventions or reported findings, not as settled constants.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic, not yet on disk: `09_forecast-error-metrics.md` — the general definitions of MAE, RMSE, MAPE, sMAPE, and MASE
  assumed throughout this file.
- Related: [`04_calendars-weeks-time-zones.md`](04_calendars-weeks-time-zones.md) — buckets, fiscal calendars, and why the
  evaluation grain depends on the calendar.
- Related: [`08_measurement-error-bias.md`](08_measurement-error-bias.md) — accuracy, precision, trueness, and bias as
  measurement concepts, and how they differ from forecast error.
- Next topic, not yet on disk: `11_demand-lead-time-distributions.md` — the distributions behind the quantiles and the critical
  ratio used in Sections 5 and 6.
- Survey file:
  [`../../../course/01A_foundations_units_time_assumptions.md`](../../../course/01A_foundations_units_time_assumptions.md)
- Return to the part map: [`../_index.md`](../_index.md)
