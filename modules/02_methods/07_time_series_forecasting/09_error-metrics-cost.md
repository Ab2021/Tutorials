# Error Metrics and Their Cost Interpretation

> Module 07 — Time Series Forecasting Foundations
> Topic 09 of 10
> Estimated study time: 3.5 hours
> Prerequisites: Module 01A topics 09–11; Module 07 topics 01–08
> Version 2026.5

---

## 1. Why this topic exists

Slide 14 of the monthly sales and operations planning (S&OP) pack says "Forecast accuracy: 82%".
The cell is shaded green. The room nods and moves to the service slide. Nobody asks what the 18%
cost, and nobody could answer, because the pack has never contained a currency figure and the
metric's definition sits in a footnote on slide 31 that three of the eleven attendees have read.

The number is not wrong. It is answering a question nobody asked: how far the forecast sat from the
actual, on average, weighted somehow, over some population, at some grain. It says nothing about
what the error did to the business, and only that second question can justify a decision.

The gap has three parts:

1. **A loss function the metric does not encode.** Mean absolute error (MAE) charges one euro for
   a unit short and one for a unit over. A fresh-food distributor charges perhaps fourteen for the
   first and twelve for the second. No symmetric statistic can represent that, and every symmetric
   statistic silently asserts that it does not exist.
2. **A decision the metric does not name.** The forecast is an input. The decision is an order
   quantity, a capacity reservation, a markdown date, or a headcount. The transformation between
   them has its own parameters: safety stock, case pack, minimum order quantity (MOQ), review period.
3. **A report the metric does not survive.** A single accuracy percentage aggregated over thousands
   of series is a statement about net error after cancellation. The decisions that failed are not
   in it.

**The metrics themselves are defined elsewhere in this program.** Module 01A topic 09,
[`09_forecast-error-metrics.md`](../../01_foundations/01A_foundations_units_time_assumptions/09_forecast-error-metrics.md),
gives the formulas, denominators, behaviour at zero, ranking flips, and the specification template.
This topic re-derives none of that. It spends its length on three harder questions: which member of
the family maps to which decision, what the wrong choice costs in money, and how to say the answer
out loud to people who will otherwise read "82%" as "fine".

**A short history, because the inheritance explains the problem.** Forecast error measurement began
as process control, not model scoring. Brown's *Statistical Forecasting for Inventory Control*
(1959) and Holt's 1957 exponential-smoothing memorandum served inventory systems monitored
continuously, and Trigg's 1964 paper "Monitoring a Forecasting System" introduced the tracking
signal as a control-chart statistic: is this replenishment system drifting?

The forecasting competitions changed the purpose. From M1 in 1982 onward, organisers needed one
number that could rank hundreds of methods across thousands of series of different scales. That
produced scale-free measures and composite scores, and rewarded comparability over commercial
relevance. Mean absolute percentage error (MAPE) survived that transition and then the spreadsheet
era because it is dimensionless, trivially computable, and because "100 minus MAPE" reads like a
school grade. None of those are reasons to believe it measures anything a business cares about.

The inheritance is visible in every planning suite: the default measure is a percentage, the
default grain is the category, and the default comparison is against the plan rather than a naive
benchmark. An analyst who accepts those defaults has accepted a loss function chosen by a software
vendor in a previous decade.

**Who is blocked by this.** The demand planning lead who cannot defend a model change because the
old model scores better on a metric that no longer matches the cost structure. The finance partner
who must accrue for forecast risk with no currency figure to accrue against. The S&OP chair who
needs a number that survives a challenge from the commercial director. The data scientist who has
built a better model and cannot show it, because "better" was never defined in the units the
business spends.

> **Note:** Nothing here claims that cost parameters are knowable to the cent. They are not. The
> discipline being taught is not precision. It is stating the ratio, showing which side of 0.5 the
> critical fractile falls on, and testing whether the decision changes across the plausible range.
> A cost model with wide error bars that changes the decision is more useful than a precise metric
> that does not.

## 2. Learning outcomes

By the end of this topic you can:

1. **Map** a named supply chain decision to a loss function: underage cost, overage cost, critical fractile, and target quantile.
2. **Derive** the critical fractile from a one-unit marginal argument, and **explain** why the cost-minimising point forecast is a quantile.
3. **Compute** the asymmetric cost of an error series by hand, and **show** a case where the better MAE is the more expensive forecast.
4. **Audit** a reported accuracy percentage and reconstruct its numerator, denominator, grain, horizon, weighting, and exclusions.
5. **Detect** the three ways a metric can flatter a model, and **state** the governance control that prevents each.
6. **Aggregate** error across a hierarchy with declared weights, and **explain** what the aggregate conceals.
7. **Interpret** a tracking signal against a stated window, and **explain** how bias accumulates behind a stable accuracy report.
8. **Present** a cost-weighted accuracy figure in currency per period, with a counterfactual and a stated range.

Each outcome maps to an exercise in section 12 and to an answer or rubric in section 13.

## 3. The core mental model

Hold one picture: a cost curve with a kink, and a forecast that has to be placed at the bottom of
it. The shape of the curve is a commercial fact. The position of the bottom is a statistical
choice. Most organisations compute the statistic without ever drawing the curve.

```text
   EXPECTED COST PER PERIOD, ONE ITEM, ONE COMMITMENT

   cost
    ^
    | \                                          /
    |  \   slope = C_o  (cost per unit over)    /   slope = C_u (cost per unit short)
    |   \                                      /
    |    \                                    /
    |     \                                  /
    |      \                                /
    |       \                              /
    |        \______________._____________/
    |                       ^
    |                       |
    +-----------------------|--------------------------------> Q committed
                        median            Q*
                                     (here q = 0.70)

   three shapes, three decisions
   C_u  =  C_o   equal slopes      bottom at the median      MAE is the right loss
   C_u  >  C_o   right steeper     bottom right of median    target a high quantile
   C_o  >  C_u   left steeper      bottom left of median     target a low quantile

   the horizontal distance from the median to Q* is the subject of this topic
```

Move one unit right of the bottom and you buy a unit that probably will not sell: you pay `C_o`
with probability `q`. Move one left and you forgo a unit that probably would have sold: you lose
`C_u` with probability `1 − q`. The bottom is where those expected marginal costs are equal, and
that equality is the whole derivation. Section 5.4 does the algebra; the picture is enough to carry.

The forecast is not the bottom of the curve. It is an estimate of a location in the demand
distribution, and the bottom is a quantile of that distribution. When the slopes are equal the
quantile is 0.5 and the bottom is the median, so MAE and the decision agree. When the slopes differ,
the bottom moves, and a forecast trained to minimise MAE stays at the median. The gap is the cost
of using a symmetric metric.

The chain runs error vector to loss function to decision to money: the error distribution sets the
scale, the slope ratio sets the quantile, the buyer executes the quantity, and finance reads the
P&L consequence. Each link is owned by a different function, which is why the chain usually breaks
at the joint between the analyst's metric and the buyer's quantity.

Six properties follow, and they explain the rest of the topic.

**Property 1 — A metric is a sample estimate of a loss function, whether or not anyone says so.**
MAE estimates expected absolute loss, root mean squared error (RMSE) expected squared loss, and
MAPE expected percentage loss weighted by the reciprocal of the actual. Each is a claim about what
a unit of error costs, wrong unless the business has that cost structure. Choose the loss
deliberately and write down why.

**Property 2 — The loss function has a slope ratio, and the slope ratio is the critical fractile.**
`C_u / (C_u + C_o)` is not a statistical parameter but the ratio of two commercial quantities, and
it determines which quantile of demand to aim at. A business that cannot state it does not know
what its forecast is for.

**Property 3 — The minimising functional is a quantile, not a fudge factor.** Practitioners often
treat a "bias adjustment" as an admission that the forecast is wrong. It is not: adding a quantile
shift to a median forecast is the correct response to asymmetric cost, and the size of the shift is
computable. Under squared error the optimum is the mean, under absolute error the median, under the
newsvendor loss the `q`-quantile. Gneiting (2011) generalises this: a score is consistent for a
quantile if and only if it is generalised piecewise linear, and pinball is the canonical member.

**Property 4 — A forecast is not a commitment.** The order is the forecast transformed by safety
stock, case pack rounding, MOQ, review period, and capacity. Treating the forecast as the committed
quantity overstates small errors and understates rounding, most at low volumes where a case pack of
twelve swamps an error of three.

**Property 5 — Aggregation cancels, and cancellation is not improvement.** `|Σe| ≤ Σ|e|`, so an
aggregate accuracy number is a statement about net error. A category can be 96% accurate in
aggregate while a third of its items are more than 50% wrong, and both numbers can be correct.

**Property 6 — The published metric is an incentive.** Planners optimise what they are measured on:
switch from MAPE to cost and the overrides change, switch back and they change again. The choice of
metric is a management decision disguised as an analytics decision.

> **Name the decision. Price the two errors. Target the quantile. Score with a consistent loss.
> Aggregate with declared weights. Report in currency against a counterfactual.**

Every failure mode in section 10 violates one of those six clauses, and every exercise in section
12 practises one of them.

## 4. Definitions and vocabulary

Formulas are stated in Module 01A topic 09. What is given here is the unit, the common misuse, and
the coding case that decides whether the number you compute is the number you meant.

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Forecast error | Observed value minus forecast, `e = y − F` | Called "accuracy"; sign left unstated | Same unit as the series |
| Signed error | An error that retains direction; positive means actual exceeded forecast | Reported as an absolute value, destroying direction | Units |
| Mean absolute error (MAE) | The mean of absolute errors over a stated window | Compared across products of different scale | Units |
| Mean squared error (MSE) | The mean of squared errors | Quoted alongside unit-scale metrics | Units squared |
| Root mean squared error (RMSE) | The square root of MSE | Described as "MAE but stricter" | Units |
| Mean absolute percentage error (MAPE) | The mean of `|e|/y`, times 100 | Used where actuals can be zero or small | Percent, denominator = actual |
| Symmetric MAPE (sMAPE) | The mean of `2|e|/(|y|+|F|)`, times 100 | Assumed symmetric because of the name | Percent, bounded 0 to 200 |
| Weighted absolute percentage error (WAPE) | Total absolute error divided by total actual | Called MAPE in reports | Proportion or percent |
| Mean absolute scaled error (MASE) | MAE divided by the in-sample naive one-step MAE | Denominator computed on the test window | Dimensionless ratio |
| Bias | Mean signed error over a stated window | Reported without a denominator or window | Units, or percent of mean actual |
| Mean absolute deviation (MAD) | Mean absolute error over the tracking window | Confused with the standard deviation | Units |
| Running sum of forecast errors (RSFE) | Cumulative signed error over the tracking window | Reset silently after a model change | Units |
| Tracking signal (TS) | RSFE divided by MAD | Read as a significance test | Dimensionless ratio |
| Trip value | The threshold at which a tracking signal flags bias | Quoted without the window length | Dimensionless |
| Pinball loss | `q·e` for `e ≥ 0` and `(q−1)·e` otherwise, at level `q` | Unknown outside specialist teams | Units |
| Quantile forecast | A forecast of the value demand will not exceed with probability `q` | Treated as a point forecast plus a margin | Units, indexed by `q` |
| Critical ratio | `C_u / (C_u + C_o)`; the cost-minimising quantile | Confused with a service-level target | Dimensionless, 0 to 1 |
| Underage cost (`C_u`) | The cost of being one unit short of demand | Stated as gross margin only | Currency per unit, year stated |
| Overage cost (`C_o`) | The cost of one unit left when demand ends | Stated as full unit cost, ignoring salvage | Currency per unit, year stated |
| Salvage value | Recovery obtained for an unsold unit, including markdown | Assumed zero, overstating overage cost | Currency per unit |
| Newsvendor problem | Single-period quantity choice before demand is known | Applied to multi-period inventory with carryover | Problem class |
| Cost of forecast error (CFE) | Currency cost of forecast error against a named counterfactual | Computed with no counterfactual stated | Currency per period |
| Asymmetric loss | A loss with different unit costs in each direction | Handled by ad hoc weighting of errors | Function plus parameters |
| Proper scoring rule | A score whose expectation is optimised by reporting the true predictive distribution | Applied to point forecasts without qualification | Depends on the score |
| Continuous ranked probability score (CRPS) | The integral of quantile loss over all levels | Used when only a point forecast exists | Units |
| Forecast value added (FVA) | Improvement attributable to one process step, against a named benchmark | Reported without the benchmark | Metric difference or percent |
| Aggregation weight | Weight applied to a series when errors are pooled | Left implicit in an unweighted mean | Volume, revenue, margin, or cost |
| Cost-weighted accuracy | One minus the ratio of realised cost to a counterfactual's cost | Reported without the counterfactual | Proportion, definition-dependent |
| Cycle service level (CSL) | Probability of no stock-out within a replenishment cycle | Reported interchangeably with fill rate | Proportion |
| Fill rate | Fraction of demand units served from stock | Confused with CSL | Proportion |
| Holding cost rate | Annual cost of carrying one unit of value, as a fraction of value | Applied uniformly across item types | Percent per year, year stated |
| Stockout cost | Cost incurred per unit of unmet demand | Equated with lost margin | Currency per unit |
| Forecast skill score | A score expressed relative to a benchmark's score | Reported without the benchmark's identity | Ratio or proportion |

Three pairs are routinely confused in a way that changes the answer.

**Cost-weighted accuracy is not accuracy.** "Accuracy" is `1 −` some error measure. Cost-weighted
accuracy is `1 −` the ratio of realised cost to the cost of a counterfactual policy. The two can
move in opposite directions, and section 6 shows them doing so: the forecast with the higher
accuracy has the higher cost. If a report says "accuracy" without a definition, ask which of the
two it means.

**The critical ratio is not a service-level target.** They are numerically related but conceptually
different. The critical ratio derives from costs and determines the cost-minimising quantile. A
service-level target is a commitment, sometimes contractual, sometimes aspirational, sometimes a
rule of thumb ("95% for A items"). When they differ, the business has chosen to spend money on
service beyond the cost-minimising point, which may be entirely rational. What is not rational is
reporting the target as though it were the ratio.

**Pinball loss is not a penalty invented to punish under-forecasting.** It is the scoring function
a forecaster minimising expected cost would choose when the target is a quantile. Using MAE to
select a model that will be used at the 0.70 quantile is not conservative; it selects a model
optimised for a different decision.

## 5. The mechanics, step by step

This section builds the mapping from decision to metric in fifteen steps across five phases. The
order matters — cost model, then metric, then model selection — and most organisations run it
backwards, which is how they improve a metric without improving a decision.

### 5.1 The family in one table

The eight measures below are defined and derived in Module 01A topic 09; what is repeated is the
reward structure, because that is what the decision mapping depends on.

| Measure | What it rewards | What it ignores | Fails when |
|---|---|---|---|
| MAE | Closeness in units; the conditional median | Direction; the largest miss | Costs are asymmetric |
| RMSE | Avoiding large deviations; the conditional mean | Many small errors; direction | One outlier dominates |
| MAPE | Accuracy on small-actual periods | Volume; direction after transformation | Actuals are zero or near zero |
| sMAPE | A bounded percentage error | True symmetry; volume | Series are intermittent; saturation at 200% |
| MASE | Beating the naive benchmark on each series | Commercial consequence entirely | The naive benchmark is not the alternative |
| Bias | Direction and persistence | Magnitude relative to typical error | Reported alone, without MAD |
| Tracking signal | Persistence of direction over a window | The size of the cost | The window is not fixed or stated |
| Pinball at `q` | The correct quantile for a stated cost ratio | Anything outside level `q` | The forecast is a point, not a distribution |

Read the last row as the destination and the first seven as diagnostic instruments: MAE, RMSE,
MAPE, sMAPE, and MASE describe how the model behaves, bias and the tracking signal whether it
behaves consistently in one direction, and pinball at the critical fractile whether it behaves
correctly for the decision. You need all three kinds of information.

### 5.2 Phase A — Fix the evaluation contract

**Step 1. Name the decision, the owner, and the cadence.** Write one sentence: "This metric governs
the weekly replenishment order for item–location combinations in category X, owned by the
replenishment manager, reviewed every Tuesday." If you cannot write it, stop: without a decision
there is no loss function, and without a loss function every metric choice is arbitrary. The
cadence sets the horizon — a weekly decision with a three-week lead time is a three-week-ahead
decision evaluated weekly, not a one-week-ahead task nobody performs.

**Step 2. Name the quantity actually committed.** The forecast is not the order. The order is the
forecast plus or minus a safety-stock adjustment, rounded to a case pack, raised to an MOQ, and
constrained by capacity and contract. Each transformation changes the effective critical ratio:

- **Case pack and MOQ** create a step cost: an error inside a case is free, one that crosses a boundary is expensive, which pushes the optimum toward the middle of the step.
- **Safety stock** is often set from the error distribution, so forecast and buffer are not independent; improving the forecast without re-setting the buffer leaves inventory unchanged.
- **Capacity constraints** truncate the achievable quantile, and a target above the supply cap is not achievable.

State which quantity the cost calculation uses. Using the forecast as a proxy for the order is
acceptable when the transformations are small and disclosed, and misleading when they are not.

**Step 3. Fix the evaluation contract.** Six fields, each of which changes the number: origin,
vintage, horizon, actual, grain, sign. A cost figure computed on a different vintage from the one
the buyer acted on is not the cost of the decision; it is the cost of a forecast nobody used.

### 5.3 Phase B — Price the two errors

**Step 4. Build the underage cost from components.** `C_u` is the cost of being one unit short at
the moment demand occurs. It is almost never the gross margin, and almost always larger.

| Component | Question | Typical treatment |
|---|---|---|
| Lost contribution margin | What margin is forgone on the unsold unit? | Always included |
| Expedite or substitute premium | What does recovery cost? | Included where recovery is attempted |
| Contractual service penalty | Is there a charge per unit short or per late order? | Included where contracted |
| Customer lifetime effect | Does a stock-out cause measurable switching? | Included only with evidence |
| Production disruption | Does a shortage stop a line or break a batch? | Included where the line stops |
| Goodwill and escalation cost | What management time does it consume? | Usually excluded; state the exclusion |

**Step 5. Build the overage cost from components.** `C_o` is the cost of one unit left when its
demand window closes. It is almost never the unit cost, and usually smaller than people assume,
because salvage and markdown recover part of the value.

| Component | Question | Typical treatment |
|---|---|---|
| Unit cost or transfer price | What was paid for the unit? | Starting point |
| Salvage or markdown recovery | What fraction of value is recovered? | Subtracted; often 30–70% |
| Disposal or recycling fee | What does disposal cost? | Added |
| Handling and reverse logistics | What does processing the remainder cost? | Added |
| Storage and obsolescence | Carrying cost over the residual life? | Added |
| Working capital | Financing cost of the cash tied up? | Added at the stated rate |

The most common error in the overage build is ignoring salvage, which inflates `C_o`, lowers the
critical ratio, and pushes the business to under-order. The second most common is ignoring carrying
cost, which does the opposite. Both are directional and both are invisible unless the build-up is
shown.

**Step 6. Test whether the cost is linear in units.** The fractile is constant only if both unit
costs are constant, and both usually rise with quantity: markdown depth with overage, expedite cost
with shortage, plus step costs at overtime and truckload thresholds. Where non-linearity is
material, the correct object is a cost function, evaluated across candidate quantities (section
6.7) rather than read off a single fractile; the fractile remains the right first approximation.

### 5.4 Phase C — From cost to quantile

**Step 7. Derive the critical fractile by marginal analysis.** Order one more unit than a candidate
`Q`. With probability `1 − q`, where `q = P(D ≤ Q)`, demand exceeds `Q` and the unit sells, gaining
`C_u`. With probability `q` it is left over, losing `C_o`.

```text
   expected marginal gain = (1 - q) * C_u  -  q * C_o
   set to zero :  (1 - q) * C_u = q * C_o
                  C_u           = q * (C_u + C_o)
                  q*            = C_u / (C_u + C_o)
```

That is the whole derivation, and it is worth being able to reproduce on a whiteboard, because the
moment a commercial stakeholder sees it they can challenge the two cost numbers rather than the
statistics. That is the right argument to have.

**Step 8. Convert the fractile into a target for the forecast.** If the forecast is an unbiased
estimate of the median and the error distribution is approximately normal with standard deviation
`σ`, the `q`-quantile of demand is the forecast plus `z_q · σ`. Three consequences:

- The shift is proportional to `σ`, so a model with smaller `σ` needs a smaller shift; that is how forecast improvement converts into inventory benefit.
- The shift is not a bias correction: an unbiased forecast may still need one, because the decision is not at the median.
- If errors are skewed, the normal approximation is wrong in the tail that matters most; use the empirical error quantile where history permits.

> **Note:** Applying a quantile shift and then reporting accuracy against the shifted forecast
> produces a worse accuracy number by construction. That is expected. The shifted forecast is a
> better decision input and a worse median estimate, and the two facts do not conflict. Report
> both, labelled.

**Step 9. Choose the scoring rule that is consistent for that quantile.** Pinball loss at level `q`
is the scoring function whose expectation is minimised by the true `q`-quantile. Gneiting and
Raftery (2007) characterise the class of consistent quantile scores, and Gneiting (2011) shows that
consistency for a quantile holds if and only if the score is generalised piecewise linear. Pinball
is the canonical member:

```text
   L_q(e) = q * e            if e >= 0   (under-forecast)
   L_q(e) = (q - 1) * e      if e <  0   (over-forecast)
```

At `q = 0.5` this is half the absolute error, so MAE and pinball rank models identically at the
median. At `q = 0.7` an under-forecast costs 0.7 per unit and an over-forecast 0.3, a ratio of 7:3,
which is exactly the cost ratio implied by `C_u = 14` and `C_o = 6`. The loss function and the
decision are the same statement in two notations.

**Step 10. Score candidates on pinball, and report MAE alongside.** Model selection uses pinball at
the decision's fractile. Reporting keeps MAE, WAPE, and bias, because those are the numbers planners
and finance already understand, and because a model that wins on pinball but is wildly biased has a
calibration problem the pinball average can hide. Two numbers, two purposes, one slide.

### 5.5 Phase D — Aggregate without lying

**Step 11. Choose weights that match the decision.** An unweighted mean asserts that a slow-moving C
item matters as much as a fast-moving A item. That is almost never true for replenishment and
almost always true for service, because a stock-out is a stock-out to the customer who experienced
it. The weight is a statement about which decision the number serves:

| Weight | Asserts | Appropriate for |
|---|---|---|
| Equal per series | Every item matters equally | Service and availability reporting |
| Volume | Every unit matters equally | Production and transport planning |
| Revenue | Every euro of sales matters equally | Commercial and financial reporting |
| Margin | Every euro of contribution matters equally | Assortment and pricing decisions |
| Cost of error | Every euro of consequence matters equally | Model selection and investment cases |

M5 made this explicit by publishing a revenue-weighted hierarchical metric across twelve levels, so
that error on high-revenue series counted more (Makridakis et al., 2022). That is the pattern to
copy: the weights are a design choice, and they are published.

**Step 12. Report the distribution, not only the mean.** A mean pinball loss of 3.2 euros per period
across 4,000 series could come from 4,000 series at 3.2, or from 200 at 60 and 3,800 at 0.2, and the
two portfolios need different interventions. Report the weighted mean with its weight, the
unweighted median and 90th percentile, the share of series above a cost threshold, and the
item–location combinations that generated most of the cost. That last line is the one a commercial
audience acts on.

**Step 13. Reconcile the aggregate to the hierarchy.** If item forecasts are reconciled to a
category forecast, the item-level errors change by construction, and a metric computed before
reconciliation is not comparable with one computed after. State which version is scored. The
hierarchy machinery is treated in [`05_hierarchy-aggregation.md`](05_hierarchy-aggregation.md) and
in Module 08; here the point is that the aggregate and the item-level numbers answer different
questions and must both be reported when both decisions exist.

### 5.6 Phase E — Monitor and present

**Step 14. Monitor bias separately, on a fixed window.** Accuracy is a magnitude statistic; bias is
a direction statistic. A report that shows only accuracy cannot show bias, because a forecast that
is 8% low every week has a respectable mean absolute error and a cumulative error that grows without
limit.

```text
   SAME ACCURACY, DIFFERENT BIAS

   week            1    2    3    4    5    6    7    8
   actual        100  100  100  100  100  100  100  100
   forecast A    108   92  108   92  108   92  108   92    errors +-8, mean 0
   forecast B    108  108  108  108  108  108  108  108    errors -8 every week

   MAE            A 8.0          B 8.0            identical
   bias           A 0.0          B -8.0           not identical
   tracking sig   A 0.0          B -8.0           B is out of control
```

Both forecasts have an MAE of 8 units on a mean actual of 100, which is an 8% error and, in most
reporting conventions, 92% accuracy. Only one is systematically wrong, and only the bias and
tracking-signal columns reveal which. The window must be fixed, because the statistic's variance
grows with the window length, and a threshold quoted without a window is not a threshold.

**Step 15. Pre-register the metric and publish the change log.** The metric, weights, grain,
horizon, benchmark, and cost parameters are declared before the evaluation window is scored. Any
change is versioned and disclosed with a restatement note. This is the control that prevents the
most common institutional failure in section 10: a metric that drifts until nobody can tell whether
performance improved.

### 5.7 The decision mapping

The table is the payload of the topic. The ranges in the last column are illustrative, derived from
the cost structures named, and must be replaced with the reader's own numbers.

| Decision | What a unit short costs | What a unit over costs | Score with | Illustrative `q` |
|---|---|---|---|---|
| Weekly replenishment, ambient grocery, 3-week lead time | Lost margin plus expedite | Holding plus eventual markdown | Pinball at `q` | 0.60–0.75 |
| Daily order, fresh produce, 2-day life | Lost margin, some substitution | Full cost plus disposal, little salvage | Pinball at `q` | 0.25–0.45 |
| Fashion first buy, 12-week season | Lost full-price sale, hard to recover | Deep markdown, 60–80% off | Pinball at `q` | 0.65–0.80 |
| Critical spare part, 48-hour downtime cost | Downtime, contractual penalty | Holding, obsolescence | Pinball at high `q` | 0.90–0.98 |
| Contract manufacturing capacity reservation | Overtime or premium freight | Idle capacity charge | Pinball at `q`, capacity-truncated | 0.50–0.70 |
| Promotion forward buy | Lost promotional margin | Post-promotion markdown and pantry loading | Cost-weighted, event level | Event-specific |
| Labour scheduling for a warehouse | Overtime and service failure | Paid idle hours | Pinball at `q` | 0.55–0.70 |

Two rows deserve comment. Fresh produce has a critical ratio below 0.5, which surprises people who
assume service always dominates: when a leftover costs the full cost and a shortage only the
margin, the cost-minimising order sits below the median, so the business intentionally under-orders
it. Spare parts sit near the top of the range for the opposite reason, because downtime dominates
holding cost by orders of magnitude. Both are commercial judgments the metric should reflect.

### 5.8 Branch points that change the method

Four branches decide whether the newsvendor framing applies at all.

**Single period or multi-period?** The newsvendor is single-period. With carryover the correct
policy is dynamic, and the fractile is a heuristic that works when carryover value is close to
salvage; applying it to a multi-period problem systematically under-orders.

**Is supply capacitated?** If the maximum obtainable quantity is `K`, the achievable quantile is
`min(q*, F(K))`, and ignoring the cap recommends a quantity the business cannot buy — a supply
failure recorded as a demand failure.

**Is demand censored?** Sales are demand truncated at available stock, so the error series
computed against sales is biased toward zero exactly where the forecast was most wrong. See
[`06_intermittent-demand.md`](06_intermittent-demand.md) and Module 01A topic 11.

**Is the item substitutable?** A customer who cannot get item A may buy item B, so the cost of A's
stock-out is the margin difference, not the full margin; substitution lowers `C_u` and therefore
the critical ratio, and ignoring it over-orders substitutable items.

## 6. Worked example

### 6.1 Setting

A regional distributor of fresh-cut fruit supplies 180 stock-keeping units (SKUs) to convenience
stores. One SKU is a 200-gram mango chunk pack, sold in cases of twelve. The pack has a four-day
shelf life and is delivered twice weekly. The distributor plans weekly and commits to a quantity
per store cluster per delivery.

All money is euro at 2026 price levels, per case. The example is constructed to be internally
consistent and to isolate the metric choice; it is not drawn from a published case.

### 6.2 Two forecasts and ten weeks of actuals

| Week | Actual | Forecast A | Error A | Forecast B | Error B |
|---:|---:|---:|---:|---:|---:|
| 1 | 120 | 114 | +6 | 130 | −10 |
| 2 | 90 | 84 | +6 | 76 | +14 |
| 3 | 140 | 134 | +6 | 150 | −10 |
| 4 | 70 | 64 | +6 | 56 | +14 |
| 5 | 150 | 144 | +6 | 160 | −10 |
| 6 | 60 | 54 | +6 | 46 | +14 |
| 7 | 160 | 154 | +6 | 170 | −10 |
| 8 | 80 | 74 | +6 | 66 | +14 |
| 9 | 130 | 124 | +6 | 140 | −10 |
| 10 | 100 | 60 | +40 | 86 | +14 |
| **Total** | **1,100** | | **+94** | | **+20** |

Forecast A is a lean model: six cases below actual in nine weeks, and a forty-case miss on the peak.
Forecast B is noisy and level: alternately ten below and fourteen above. Neither is a straw man;
both are the kind of output a real planning system produces. The construction is deliberate — A is
directionally biased and low on average, B is unbiased but noisier — because that is the combination
that separates symmetric from asymmetric evaluation.

### 6.3 Symmetric metrics, computed by hand

| Metric | Method A | Method B | Winner |
|---|---:|---:|---|
| Mean error (bias) | +9.40 cases/week | +2.00 cases/week | A is more biased |
| Mean absolute error (MAE) | 9.40 cases | 12.00 cases | A |
| Root mean squared error (RMSE) | 13.87 cases | 12.17 cases | B |
| Mean absolute percentage error (MAPE) | 9.44% | 12.65% | A |
| Weighted absolute percentage error (WAPE) | 8.55% | 10.91% | A |
| Implied "accuracy" (`1 − WAPE`) | 91.45% | 89.09% | A |
| Tracking signal | +10.00 | +1.67 | A is out of control |

By hand: A sums to 94 absolute units and 1,924 squared units over ten weeks, giving MAE 9.40 and
RMSE 13.87; B sums to 120 and 1,480, giving MAE 12.00 and RMSE 12.17. WAPE divides each by the
1,100-unit total, and the tracking signal divides the cumulative error by the MAE.

Three readings matter. **A wins on MAE, MAPE, WAPE, and "accuracy"**, which by ordinary reporting
conventions makes it the better model. **B wins on RMSE because A's week 10 contributes 1,600 of
A's 1,924 squared units**, 83% of its MSE; that is standard RMSE behaviour and an argument for
looking at the error vector rather than the score. **A's tracking signal of +10 is a red flag no
accuracy number shows**: a cumulative error of 94 cases against a MAD of 9.4 would have fired a
control chart in week 2, while the monthly pack reports 91.5%.

### 6.4 The cost model

| Underage component | EUR/case | Overage component | EUR/case |
|---|---:|---|---:|
| Lost contribution margin | 9.00 | Unit cost | 22.00 |
| Expedite premium on recovery | 3.00 | Markdown recovery (subtracted) | −14.00 |
| Contract service penalty | 2.00 | Disposal fee | 1.00 |
| **Total `C_u`** | **14.00** | Handling and reverse logistics | 2.00 |
| | | Storage and capital over residual life | 1.00 |
| | | **Total `C_o`** | **12.00** |

The overage components sum to `22.00 − 14.00 + 1.00 + 2.00 + 1.00 = 12.00`, so:

```text
   q* = C_u / (C_u + C_o) = 14 / (14 + 12) = 14 / 26 = 0.538
```

That is a mild asymmetry, and a useful corrective to the assumption that fresh produce always has a
low critical ratio. The markdown recovery is 14 of 22 euros, 64%, which pulls `C_o` down and the
ratio up. If the pack were unsaleable after day four and recovery were 4 euros, `C_o` would be 22
and the ratio would fall to `14/36 = 0.389`, well below the median. The same item, in the same week,
has a critical ratio of 0.54 or 0.39 depending on a salvage assumption nobody in the S&OP meeting
has been asked about.

### 6.5 Asymmetric cost per method

Using `e = actual − forecast`, so positive errors are shortfalls:

| Method | Sum of positive errors | Sum of negative errors | Cost per 10 weeks | Cost per week |
|---|---:|---:|---:|---:|
| A | 94 | 0 | `14 × 94 + 12 × 0 = 1,316` | **EUR 131.60** |
| B | 70 | 50 | `14 × 70 + 12 × 50 = 1,580` | **EUR 158.00** |

A is cheaper by 26.40 euros per week despite having no negative errors, because its shortfalls are
smaller in total than B's. On this cost structure the MAE ranking and the cost ranking agree. That
is a legitimate outcome worth stating: an asymmetric model does not always reverse the ranking. It
reverses when the directional difference is large relative to the magnitude difference.

### 6.6 The reversal

Now raise the service penalty so `C_u = 40` while `C_o` stays at 12. The critical ratio becomes
`40/52 = 0.769`, a high quantile.

| Method | Cost at `C_u = 40`, `C_o = 12` | Cost per week | Rank |
|---|---:|---:|---|
| A | `40 × 94 + 12 × 0 = 3,760` | EUR 376.00 | 2 |
| B | `40 × 70 + 12 × 50 = 3,400` | EUR 340.00 | 1 |

The ranking flips. Method B, with the worse MAE, MAPE, and WAPE, is 36 euros per week cheaper,
because its errors are balanced and A's are one-directional. That is the argument of this topic in
one table. **No symmetric metric can see it, because the information that produces the flip is the
sign of the errors, and symmetric metrics discard the sign.**

Reverse the economics instead — destroy leftovers, so `C_o = 23` with no recovery, while `C_u` stays
14 — and the ratio falls to `14/37 = 0.378`. A then costs `14 × 94 = 1,316` (EUR 131.60) against B's
`14 × 70 + 23 × 50 = 2,130` (EUR 213.00). A wins by more, because destroying leftovers penalises
B's over-forecasts and A has none. The perishable case and the fast-mover case pull in opposite
directions, which is why the two unit costs must be built for the item rather than for the category.

Scaled to the family, the 36-euro weekly gap becomes `36 × 52 × 180 = EUR 336,960` per year if it
were representative. It is not; the figure is an order-of-magnitude illustration of why the per-SKU
number must be aggregated before it is presented.

> **Note:** A real business case computes the gap per SKU and sums, rather than multiplying an
> average. It also nets out the cost of changing models, which is not zero.

### 6.7 The quantile target, on a discrete demand distribution

| Demand (cases) | Probability | Cumulative probability |
|---:|---:|---:|
| 60 | 0.10 | 0.10 |
| 90 | 0.20 | 0.30 |
| 120 | 0.30 | 0.60 |
| 150 | 0.25 | 0.85 |
| 180 | 0.15 | 1.00 |

The mean is `0.10×60 + 0.20×90 + 0.30×120 + 0.25×150 + 0.15×180 = 124.5`. The median is 120. A
critical ratio of 0.769 falls between the cumulative probabilities of 0.60 and 0.85, so the
cost-minimising quantity is 150.

| Order `Q` | Expected shortfall (cases) | Expected overage (cases) | Expected cost (EUR) |
|---:|---:|---:|---:|
| 60 | 64.5 | 0.0 | 2,580 |
| 90 | 37.5 | 3.0 | 1,536 |
| 120 | 16.5 | 12.0 | 804 |
| 150 | 4.5 | 30.0 | 540 |
| 180 | 0.0 | 55.5 | 666 |

The minimum is at 150, twenty-five cases above the mean and thirty above the median. Ordering the
median instead costs `804 − 540 = 264` euros per delivery, a 49% increase in expected cost. This
table is the most useful artefact in the topic for a commercial audience: it does not argue about
whether the forecast is good, it shows the cost at each quantity.

### 6.8 What the commercial audience sees

| Framing | Number | What the audience hears |
|---|---|---|
| Implied accuracy | 91.5% | "Fine" |
| Mean absolute error | 9.4 cases per week on a mean of 110 | "About one case per cluster per week" |
| Cost of error at the decision's cost structure | EUR 376 per week for this SKU | "That is a real number" |
| Cost against a counterfactual | 36 EUR/week worse than a balanced forecast; EUR 337k/year across the family | "Fix it" |

The first line stops the conversation. The fourth starts it. The transition requires two things the
pack does not contain: the cost build-up and the counterfactual. A counterfactual is any named
alternative whose cost is computed on the same weeks — the naive forecast, the incumbent model, a
balanced model, or a perfect-foresight lower bound. "Our forecast cost 376 euros per week" is a
reading without a scale. "376, against 340 for the model we rejected, and 210 for perfect
foresight" tells the room where the money is.

### 6.9 What the analyst should have asked

Five questions decide whether the figure is usable: was the 91.5% computed on the vintage the buyer
acted on; does the error series use sales or demand, since stock-out weeks censor the latter; is
`C_o` built on realised markdown recovery or list markdown; is `C_u` the margin alone or the margin
plus penalty; and are ten weeks representative of a seasonal cycle. Each is a commercial question
wearing a statistical costume, which is the normal shape of this work.

## 7. Code, query, or calculation

The calculation that matters is the critical fractile and the shift it implies. It is short enough
to do on paper, and doing it on paper is the point: the numbers are arguable, and the argument is
easier when the arithmetic is visible.

```text
NEWSVENDOR CRITICAL FRACTILE -- HAND CALCULATION

given   C_u = 14 EUR per case short  (margin 9 + expedite 3 + penalty 2)
        C_o = 12 EUR per case over  (cost 22 - recovery 14 + disposal 1 + handling 2 + carry 1)

step 1  one more case: sells with probability (1 - q), gain C_u; spoils with probability q, lose C_o
step 2  order while (1 - q) * C_u > q * C_o, so q < C_u / (C_u + C_o)
step 3  q* = 14 / (14 + 12) = 14 / 26 = 0.538

step 4  normal errors, sigma = 25 cases: z(0.538) = 0.095, shift = 0.095 * 25 = +2.4 cases
step 5  if recovery falls to 4 EUR, C_o = 22, q* = 14/36 = 0.389, z = -0.282, shift = -7.1
step 6  if penalty lifts C_u to 40, C_o = 12, q* = 40/52 = 0.769, z = +0.735, shift = +18.4

read    the decision moves by 25 cases across the plausible cost range.
        the forecast has not changed. only the assumption has.
```

Three checks before trusting any such calculation. Confirm that `C_o` subtracts recovery rather
than adding it; a sign error here is common and produces a ratio on the wrong side of 0.5. Confirm
that both costs are in the same currency and price year. Confirm that `σ` is the error standard
deviation at the decision horizon, not the standard deviation of demand over a different window.

## 8. Data, standards, and regulatory context

No international standard defines a cost-weighted forecast accuracy metric, and no regulator
specifies a critical fractile. That absence is informative: the choice is commercial, which is why
it must be documented rather than defaulted. What exists is a set of adjacent standards that
constrain the inputs and the reporting.

### 8.1 Measurement vocabulary

| Standard | Governs | What it fixes | What it does not fix | Grade |
|---|---|---|---|---|
| ISO 3534-1 | Statistics vocabulary and symbols | Definitions of error, bias, variance, estimator | Any forecast error formula | B |
| ISO 5725-1 | Trueness and precision of measurement methods | Accuracy as a combination of bias and precision | Forecast evaluation, which has no repeated reference | B |
| JCGM 200:2008 (VIM) | Metrology vocabulary | Accuracy is not a quantity and has no numerical value | Anything specific to forecasts | B |
| ISO 80000-1 | Quantities and units | Unit conventions for money, time, ratios | Which currency or price year to use | B |
| ISO 8601 | Date and time representation | Week numbering and ordering | Which week a demand observation belongs to | B |

The VIM point transfers directly. "Forecast accuracy of 91.5%" is a conformance rate against an
implicit tolerance, exactly as "inventory accuracy of 98%" was in Module 01A topic 08: a statistic
about a definition and a sample, not a property of the forecast. The standard does not forbid the
phrase; it forbids pretending it has a unique meaning.

### 8.2 Inventory valuation, where the overage cost becomes an accounting number

`C_o` is not only an operational parameter. It is the shadow of an accounting rule, because the
loss on unsold inventory is recognised under inventory valuation standards.

| Requirement | Where it applies | Effect on `C_o` | Grade |
|---|---|---|---|
| IAS 2 *Inventories* | IFRS-reporting entities | Lower of cost and net realisable value (NRV); write-downs when NRV falls | B |
| ASC 330 *Inventory* | US GAAP entities | Similar lower-of-cost-or-market logic, with industry practice | B |
| IFRS 15 / ASC 606 | Revenue | Unsold units generate no revenue, so the full margin is forgone | B |
| Local waste and disposal regulation | Jurisdiction-specific | Disposal fees and restrictions raise `C_o` directly | B |

When the finance partner says the overage cost is the unit cost, they describe the balance-sheet
loss before recovery; when the commercial manager says it is the markdown loss, they describe the
realised recovery. Both are correct, they describe different stages of the same event, and the
critical ratio depends on which is used. Reconcile them before the S&OP meeting, not during it.

### 8.3 Internal control and model risk

Forecast accuracy feeds inventory valuation and revenue forecasts, and in listed companies those
figures fall within internal control over financial reporting under Section 404 of the
Sarbanes-Oxley Act of 2002 (SOX 404). The control governs the process — who may change a forecast,
how the change is documented, whether the model is validated — not the choice of metric. A company
can be fully compliant and still report a metric that mismeasures its cost.

Banking model-risk guidance such as the Federal Reserve's SR 11-7 is a useful template by analogy,
requiring model inventory, validation, and ongoing monitoring. It does not govern demand planning
models, and citing it as though it did is a category error: use it for the shape of the control,
not as a legal obligation.

### 8.4 Perishables and waste

Where the product is perishable, `C_o` is partly a regulatory and reputational quantity.

| Instrument | Jurisdiction | Relevance to `C_o` | Grade |
|---|---|---|---|
| EU Waste Framework Directive 2008/98/EC, as amended | European Union | Waste hierarchy duties; handling and reporting obligations | B |
| French food-waste legislation of 2016 | France | Donation and destruction obligations for unsold food | B |
| National food-waste measurement duties | Several EU member states | Reporting obligations that create a data trail | B |
| Retailer donation agreements | Contractual | Alternative recovery route, usually below markdown value | C |

The effect is that `C_o` for a perishable is not stable across jurisdictions or customers: a
donation agreement can raise recovery relative to destruction, lowering `C_o`, while a reporting
obligation raises administrative cost. A single `C_o` for a European portfolio averages across
different regimes and should say so.

### 8.5 What is convention rather than requirement

- The ±4 tracking-signal trip value is a practitioner convention, not a significance level; the APICS/ASCM literature uses 3.0 and 4.0, and its meaning depends on the window.
- A 95% cycle service level is a policy choice in most companies, not a regulatory requirement.
- ABC thresholds of 80/15/5 are a historical convention, not an optimum.
- Markdown recovery rates are negotiated commercial outcomes, not accounting constants, and the choice of WAPE over MAPE is an engineering decision, not a compliance one.

An analyst who can distinguish a requirement from a convention can challenge a convention. One who
cannot will treat every inherited parameter as a physical constant, including the ones that decide
which model the company deploys.

## 9. Industry practice and case evidence

### 9.1 The MAPE default and its documented costs

MAPE remains the most widely reported measure in commercial demand planning for historical rather
than statistical reasons: it is scale-free, it was easy to compute in spreadsheets, and "100 minus
MAPE" reads as a grade. The costs are documented.

| Evidence | What it shows | Grade |
|---|---|---|
| Hyndman and Koehler (2006) | MAPE, sMAPE, and several relatives are degenerate when actuals are zero or near zero; MASE is proposed and works across all their series | A |
| Goodwin and Lawton (1999) | sMAPE is not symmetric; large under-forecasts are penalised differently from equivalent over-forecasts | A |
| Kolassa (2026) | For sufficiently intermittent series, sMAPE is minimised by a flat zero forecast, which makes it gameable for replenishment | A |
| Syntetos and Boylan (2005) | On 3,000 automotive spare-parts series, percentage measures were inapplicable because actual demand can be zero | A |
| Davydenko and Fildes (2016) | Recommends scaled and relative measures, with caution about averaging ratios across series | B |

The pattern is consistent: the percentage family is convenient, widespread, and known to fail in
precisely the segment — low-volume, intermittent items — where service failures are most visible to
customers.

### 9.2 The cost-of-error literature

A separate practitioner literature values forecast error in currency. It is smaller and less
rigorous than the metric literature, and it is the one that answers the question a commercial
audience asks.

| Source | Method | Reported result | Grade |
|---|---|---|---|
| Catt (2007), *Foresight* | Cost of forecast error as safety-stock holding cost plus lost-sales cost | Frames CFE as the basis for safety stock and service level | C |
| Jain (2003), *Journal of Business Forecasting* | Decomposes error cost into over-forecast inventory and under-forecast lost sales | Qualitative framework, widely taught | C |
| Nguyen, Ni and Rossetti (2010), IERC | CFE as the cost gap between a policy run on forecasts and one run on known demand | 50,259.70 known demand versus 54,513.03 moving average, CFE about 4,253; exponential smoothing 53,817.07, CFE about 3,557 | B |
| Sosa and Wellens (2016), EMSS | Service level derived from a CFE function | Optimum near 98% service; about 6.9% saving against 95% in their case | B |
| Gogolev and Ozhegov (2023), *Applied Econometrics* | Newsvendor-derived asymmetric loss on product-level sales | Asymmetric-loss models achieve lower economic cost than symmetric ones | A |
| Saltzman (2003), working paper | Least squares with adjustment versus weighted least absolute value under asymmetric costs | At a 3:1 cost ratio, opportunity loss cut 19–23%; at 2:1, 6–7% | B |

Two things are worth taking. The CFE framing — cost of a policy run on forecasts minus cost of one
run on known demand — is the cleanest definition available, because it makes the counterfactual
explicit. And the empirical results are modest but consistent in direction: asymmetric evaluation
helps, and helps more as the cost ratio moves away from 1:1.

### 9.3 Quantile forecasting in practice

The link between the critical fractile and the quantile forecast is theoretically exact and
empirically uneven.

| Evidence | What it shows | Grade |
|---|---|---|
| Gneiting and Raftery (2007); Gneiting (2011) | Pinball loss is consistent for the quantile; consistency holds iff the score is generalised piecewise linear | A |
| Bruzda (2016) | Compares 26 quantile-forecasting methods across 30 series; regression quantiles and bootstrap perform well, the normal formula is a safe choice at high service levels | A |
| Liu, Letchford and Svetunkov (working paper) | The data-driven big-data-newsvendor approach is equivalent to quantile regression in the linear case | B |
| *Journal of Risk and Financial Management* (2026) | Targeting the 90th percentile instead of the 50th cut conditional value at risk at the 95% level (CVaR95) by 26.7% and raised cycle service level from 44.4% to 89.5%, with 48.7% lower mean cost — but only at high cost ratios; at ratios at or below 2:1 the higher quantile raised mean cost while lowering tail risk | A |

The last row is the most honest evidence in the topic and the one most likely to be omitted from a
vendor presentation: quantile targeting is not a free lunch, because it reduces tail risk and
raises mean cost when the cost ratio is mild. That is what theory predicts, since at a critical
ratio near 0.5 the quantile is near the median and pushing to 0.9 is a deliberate purchase of
insurance. Buying that insurance is a commercial decision and should be made explicitly.

### 9.4 What the evidence supports


**The metric selects the model**, because a ranking flip is the normal behaviour of a loss function
applied to directional error: change the metric without changing the model and you have changed
which model wins. **Symmetric metrics assert `C_u = C_o`** through their transformations, and the
cost-of-error literature consistently finds ratios far from 1:1. **Cost parameters are the weak
link**, so the response is a stated range and a check that the decision survives it.

## 10. Failure modes and red flags

| Failure | Red flag visible in a meeting or dataset | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Symmetric metric on an asymmetric decision | Nobody in the room can state the cost of a unit short versus a unit over | The loss function asserts equal costs by construction | Model selected for the wrong direction | Derive the critical ratio; score with pinball at that level |
| MAPE on low-volume items | A footnote says zero-demand weeks were excluded, with no count | Division by a small actual explodes the percentage | Slow movers systematically under-forecast | Use MASE or WAPE; report the excluded count |
| Metric chosen after the leaderboard | The evaluation slide has a footnote reading "metric revised in Q3" | Selection on the evaluation set | Inflated apparent performance; unreproducible gains | Pre-register the metric and its version |
| Only aggregate accuracy reported | Category accuracy stable while regional fill rate falls | Errors cancel across items and locations | Item-level failure invisible | Report at the decision grain, plus the aggregate with a caveat |
| Bias never reported | The monthly pack contains no cumulative-error series | Accuracy is a magnitude statistic | Persistent under- or over-forecast accumulates unnoticed | Report bias and tracking signal on a fixed rolling window |
| Tracking signal on an expanding window | The alarm fires every month, or has never fired | The statistic's variance grows with the window | Alarm fatigue or no alarm at all | Fix the window length and state it beside the threshold |
| One cost pair applied to a whole portfolio | The same `C_u` and `C_o` for fresh and ambient items | Costs differ by shelf life, margin, and contract | Wrong quantile for most items | Segment the cost model; publish the segments |
| Overage cost omitting salvage | `C_o` equals the full unit cost for every item | Recovery exists but is not modelled | Critical ratio too low; systematic under-ordering | Build `C_o` from components, subtracting realised recovery |
| Underage cost equal to gross margin | `C_u` is stated as margin only, with no penalty or expedite | Recovery costs and contractual penalties are ignored | Critical ratio too low for high-service items | Add penalty and expedite components, or state the exclusion |
| Forecast treated as the committed quantity | The cost calculation ignores case pack, MOQ, and safety stock | The decision variable is the order, not the forecast | Small errors overstated; rounding effects missed | State which quantity is used; model the transformations |
| No counterfactual | The cost figure stands alone | A reading without a scale | Audience cannot judge whether to act | Report against naive, incumbent, or perfect foresight |

Three rows deserve expansion. **Symmetric metric on an asymmetric decision** is the failure this
topic exists to fix and the hardest to see from inside, because the metric is inherited rather than
chosen; ask in any meeting what a case short costs and what a case over costs, since "the same"
makes the metric defensible and "I would have to ask finance" makes it unvalidated. **Metric chosen
after the leaderboard** usually arrives by drift: a dashboard starts on MAPE, WAPE is added, a
"blended accuracy" reconciles them, and within two years nobody can say whether performance improved
because the definition moved three times. **No counterfactual** makes every other number unusable
commercially, because "our forecast cost 376 euros per week" invites "compared with what?" and the
room reverts to accuracy, where the scale is at least familiar.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks if it fails | How to test it |
|---|---|---|---|
| Unit underage and overage costs are known and constant per item | The fractile is a ratio of two numbers | Markdown depth varies with quantity; penalties vary by customer | Compute the ratio at low, mid, and high volumes; check the sign of the change |
| Cost is linear in units | The marginal argument needs a constant slope | Case packs, MOQ, overtime thresholds, truckload steps | Plot cost against quantity; look for steps |
| The demand distribution is stationary over the window | The fractile is a property of one distribution | Trend, seasonality, promotions, new listings | Compare error distributions across sub-periods |
| Errors are independent across periods | Tracking-signal thresholds assume it | Promotions, smoothing, planner overrides | Autocorrelation of residuals; runs test |
| The forecast is the decision variable, or the transformation is disclosed | Cost is incurred on the order, not the forecast | Safety stock, rounding, capacity | State the transformation; recompute on the order series |
| The decision is single-period, or carryover approximates salvage | The newsvendor framing is single-period | Multi-period inventory with carryover | Compare against a dynamic programme on simulated data |
| Demand is uncensored by stock-outs | The error series must reflect true demand | Sales truncated at available stock | Flag stock-out periods; estimate censored demand |
| Aggregation weights reflect decision value | The weighted mean is what the business acts on | Revenue concentration, margin heterogeneity | Recompute under an alternative weighting; check the conclusion |

Two limits of the whole approach are worth stating plainly.

**The cost model is the weak part, not the statistics.** Every calculation here depends on two
numbers a business estimates approximately, sometimes politically. That is not a reason to use a
symmetric metric, which also embeds two numbers — one and one — and hides them. It is a reason to
show the build-up, state the range, and report whether the decision survives it: "the answer is the
same for any cost ratio between 1.5 and 4" is the most useful sentence in a cost-based deck.

**A cost figure is not a forecast of savings.** The gap between two models' historical costs
estimates the value of switching assuming the future resembles the evaluation window; it does not
account for the cost of change, out-of-sample risk, or transition friction. Present it as an
estimate with a stated window, not a business case with a guaranteed return.

## 12. Exercises

### Exercise 1 — Compute the family by hand (calculation, 40 minutes)

Compute MAE, RMSE, MAPE, WAPE, bias, and the tracking signal for each method below. The naive one-step MAE from the training history is 18.0 units.

| Week | Actual | Forecast P | Forecast Q |
|---:|---:|---:|---:|
| 1 | 80 | 76 | 92 |
| 2 | 50 | 58 | 38 |
| 3 | 110 | 104 | 122 |
| 4 | 40 | 46 | 30 |
| 5 | 90 | 84 | 102 |
| 6 | 30 | 42 | 24 |
| 7 | 120 | 112 | 132 |
| 8 | 60 | 48 | 52 |

**Deliverable:** a table with every metric, its unit, and the numerator sum shown.
**Acceptance criteria:** all six metrics per method; correct units; MAPE terms shown individually; one sentence naming which method you would deploy and under what cost assumption.
**Time:** 40 minutes.

### Exercise 2 — Derive the fractile from scratch (derivation, 25 minutes)

Without consulting the text, derive `q* = C_u / (C_u + C_o)` from a one-unit marginal argument, then state what happens to `q*` when `C_u` doubles, when `C_o` halves, and when both double.

**Deliverable:** half a page of algebra plus the three comparative statics.
**Acceptance criteria:** the argument appears in words as well as symbols; the comparative statics are correct; the answer states why the ratio, not either cost alone, determines the decision.
**Time:** 25 minutes.

### Exercise 3 — Build a cost table (calculation, 45 minutes)

Compute expected cost at order quantities of 80, 110, 140, and 170, with `C_u = 30` and `C_o = 10`. Identify the optimum and its distance from the mean and the median.

| Demand | 60 | 90 | 120 | 150 | 180 |
|---|---:|---:|---:|---:|---:|
| Probability | 0.10 | 0.25 | 0.30 | 0.20 | 0.15 |

**Deliverable:** a cost table with expected shortfall, expected overage, and expected cost for each quantity, plus the optimum.
**Acceptance criteria:** arithmetic reproducible by hand; the optimum compared with the mean and median; one sentence on why the optimum sits where it does.
**Time:** 45 minutes.

### Exercise 4 — Reconcile one accuracy number three ways (analysis, 40 minutes)

Take a reported forecast accuracy of 82% from any real or constructed dashboard. Compute the number under three definitions: `1 − WAPE`, `100 − MAPE` on nonzero actuals, and `1 −` the ratio of absolute error to mean actual. State the inputs you had to invent.

**Deliverable:** three numbers, the definitional differences that produced them, and a specification entry resolving which is intended.
**Acceptance criteria:** each definition's numerator and denominator stated; the direction of bias in each identified; at least one exclusion that would flatter the number named.
**Time:** 40 minutes.

### Exercise 5 — Design the commercial slide (design, 60 minutes)

Design a single slide reporting forecast performance to a commercial audience for one product family. It must carry a currency figure, a counterfactual, a range, and the decision it supports.

**Deliverable:** a slide layout with a headline sentence, a chart specification, and speaker notes.
**Acceptance criteria:** the currency figure has a stated price year and period; the counterfactual is named and computed on the same weeks; the range comes from a stated cost sensitivity; the decision and its owner appear on the slide.
**Time:** 60 minutes.

### Exercise 6 — Audit a metric for governance (critique, 45 minutes)

Find a real or constructed forecast accuracy definition in a dashboard, vendor document, or internal report, and assess it against the fourteen-field specification in Module 01A topic 09.

**Deliverable:** a completed specification with unknowns marked, plus the three fields whose absence most affects the number.
**Acceptance criteria:** at least eight fields completed or marked unknown; the three critical gaps justified by their effect on the number; one recommendation implementable this quarter.
**Time:** 45 minutes.

### Exercise 7 — Tracking signal on a monthly report (analysis, 35 minutes)

Construct twelve months of errors for two forecasts with the same MAE but different bias. Compute the tracking signal on a rolling six-month window for both, and identify the months in which the conventional ±4 threshold is crossed.

**Deliverable:** a table of errors, MAE, bias, and rolling tracking signal by month, plus a one-page note on what the monthly accuracy report would have shown.
**Acceptance criteria:** the two series have identical MAE to one decimal; the rolling window is stated; the crossing months are identified; the note explains why accuracy alone concealed the difference.
**Time:** 35 minutes.

### Exercise 8 — Aggregate without letting big items dominate silently (design, 50 minutes)

For a three-level hierarchy (SKU, category, total), design the weighting scheme for a model selection metric. Justify the weight and state what the metric will conceal.

**Deliverable:** a weighting table by level, the formula, and a statement of the decision each level serves.
**Acceptance criteria:** weights sum to one at each level; the choice among equal, volume, revenue, and cost weighting is justified by a named decision; at least one concealment is stated explicitly; a sensitivity check under an alternative weighting is specified.
**Time:** 50 minutes.

### Exercise 9 — Sensitivity of the decision to the cost ratio (analysis, 35 minutes)

Using the data from Exercise 1, compute asymmetric cost for both methods at underage-to-overage ratios of 1:1, 2:1, 5:1, and 10:1. Identify the ratio at which the ranking changes, if it changes.

**Deliverable:** a table of costs and a break-even ratio, or a statement that no break-even exists in the range.
**Acceptance criteria:** positive and negative error sums shown separately; cost per period, not total; the break-even computed rather than estimated; a sentence on whether the decision is robust.
**Time:** 35 minutes.

### Exercise 10 — Write the specification (design, 55 minutes)

Write the full metric specification for a replenishment decision with a four-week lead time, including the cost build-up and the scoring rule.

**Deliverable:** the fourteen-field specification from Module 01A topic 09, extended with a cost build-up table and a stated critical ratio.
**Acceptance criteria:** the horizon matches the lead time; `C_u` and `C_o` are built from named components; the scoring rule is consistent with the critical ratio; the benchmark is defined precisely enough for a stranger to compute; the owner is a role.
**Time:** 55 minutes.

## 13. Self-check answers

### 13.1 Exercise 1 arithmetic

Actuals: 80, 50, 110, 40, 90, 30, 120, 60. Total 580, mean 72.5.

**Method P.** Forecasts 76, 58, 104, 46, 84, 42, 112, 48. Errors `e = actual − forecast`:
+4, −8, +6, −6, +6, −12, +8, +12.

```text
sum |e|  = 4 + 8 + 6 + 6 + 6 + 12 + 8 + 12 = 62
MAE      = 62 / 8                          = 7.75 units
sum e^2  = 16 + 64 + 36 + 36 + 36 + 144 + 64 + 144 = 540
MSE      = 540 / 8 = 67.5                  RMSE = 8.216 units
bias     = 10 / 8                          = +1.25 units
TS       = 10 / 7.75                       = +1.29
WAPE     = 62 / 580                        = 10.69%
MASE     = 7.75 / 18.0                     = 0.431
```

MAPE terms for P: `5.00, 16.00, 5.45, 15.00, 6.67, 40.00, 6.67, 20.00`. Sum 114.79, mean
**14.35%**.

**Method Q.** Forecasts 92, 38, 122, 30, 102, 24, 132, 52. Errors: −12, +12, −12, +10, −12, +6,
−12, +8.

For Q: `sum |e| = 84`, so MAE `10.50`; `sum e^2 = 920`, so MSE `115.0` and RMSE `10.724`; bias
`−12/8 = −1.50`; tracking signal `−1.14`; WAPE `84/580 = 14.48%`; MASE `10.50/18.0 = 0.583`. MAPE
terms: `15.00, 24.00, 10.91, 25.00, 13.33, 20.00, 10.00, 13.33`, sum 131.57, mean **16.45%**.

P wins every metric, and its MASE of 0.431 means it beats the naive benchmark by 57%. Q is also
better than naive at 0.583, which is worth stating: both models add value, and the question is
which adds more. P also has the lower absolute bias, so this dataset does not separate magnitude
from direction; Exercise 9 tests that separation.

A common error roots the sum rather than the mean, producing 23.24 for P instead of 8.22. Another
drops the week with actual 30 before computing MAPE, which lowers both means by about two points.
Both errors flatter the models without changing the ranking here, which is exactly why they
survive undetected.

### 13.2 Exercise 2 model answer

Order one more unit beyond a candidate `Q`. With probability `1 − q` demand exceeds `Q` and the unit
sells, gaining `C_u`. With probability `q` demand is at or below `Q` and the unit is left over,
losing `C_o`. The expected marginal gain is `(1 − q)C_u − qC_o`; ordering more is worthwhile while it
is positive, so the optimum satisfies `(1 − q)C_u = qC_o`, giving `q* = C_u/(C_u+C_o)`.

Comparative statics: if `C_u` doubles, `q*` rises; if `C_o` halves, `q*` rises; if both double, `q*`
is unchanged. The last case matters most, because it shows the decision depends on the ratio, not
the absolute magnitude. That is why a business can act on approximate costs: it needs the ratio
roughly right, not the levels exactly right.

### 13.3 Exercise 3 arithmetic

Cumulative probabilities: 0.10, 0.35, 0.65, 0.85, 1.00. The critical ratio `30/40 = 0.75` falls
between 0.65 and 0.85, so the optimum should be 150. The mean is
`6 + 22.5 + 36 + 30 + 27 = 121.5`; the median is 120.

| `Q` | Expected shortfall | Expected overage | Cost at `C_u=30`, `C_o=10` |
|---:|---:|---:|---:|
| 80 | `0.30×40 + 0.20×70 + 0.15×100 = 41` | `0.10×20 = 2` | `30×41 + 10×2 = 1,250` |
| 110 | `0.20×40 + 0.15×70 = 18.5` | `0.10×50 + 0.25×20 = 10` | `30×18.5 + 10×10 = 655` |
| 140 | `0.15×40 = 6` | `0.10×80 + 0.25×50 + 0.30×20 = 26.5` | `30×6 + 10×26.5 = 445` |
| 170 | `0` | `0.10×110 + 0.25×80 + 0.30×50 + 0.20×20 = 50` | `30×0 + 10×50 = 500` |

The optimum among the candidates is 140, at 445, which is 18.5 above the mean and 20 above the
median. The true optimum is 150, costing `30×0 + 10×(0.10×90 + 0.25×60 + 0.30×30) = 300`. The gap
between 445 and 300 is the cost of a coarse quantity grid, not of a bad forecast; separate the two
in any presentation.

### 13.4 Exercise 4 rubric

A strong answer produces three genuinely different numbers and explains each difference: `1 − WAPE`
is usually highest because it is volume-weighted; `100 − MAPE` is lower because it weights small
actuals equally; `1 −` mean absolute error over mean actual sits between them and is often close to
WAPE. It names at least one exclusion, usually zero-demand weeks, and states the direction rather
than the mere presence of bias. A weak answer produces three numbers without saying why they differ.

### 13.5 Exercise 5 rubric

A strong slide has a headline in currency with the counterfactual in the same sentence: "Forecast
error cost EUR 376k in the last twelve months, EUR 36k more than the balanced model, and the gap is
stable for any cost ratio above 2:1." It states price year and period, a range from a named
sensitivity, and a decision with a role attached. A weak slide leads with a percentage, or gives a
currency figure with no comparator.

### 13.6 Exercise 6 rubric

A strong audit completes at least eight of the fourteen fields and marks the rest unknown rather
than guessing, then prioritises gaps by effect: typically the denominator, the population, and the
weighting. It notes the direction of each bias and proposes a change implementable this quarter,
such as publishing the exclusion count. A weak audit lists fields without testing sensitivity.

### 13.7 Exercise 7 model answer

Take two twelve-month error series with identical MAE of 10.0 on a mean actual of 100. Series A
alternates +10 and −10, giving a cumulative error near zero and a tracking signal near zero; series
B has −10 every month, giving a cumulative error of −120 and a tracking signal of −12 on a
twelve-month window, or −6 on a six-month window. The monthly accuracy report shows 90% for both.
The acceptable answer states the window, shows where the rolling six-month signal crosses −4, and
concludes that an accuracy report cannot distinguish a well-calibrated forecast from one that is
persistently low by a fixed amount.

### 13.8 Exercise 8 rubric

A strong design names the decision at each level and matches the weight to it: equal weights at SKU
level if service is the concern, revenue weights at category level for the commercial report, cost
weights for model selection. It states the formula with normalisation, shows the weights sum to one,
names what the chosen weight conceals — usually that a revenue-weighted category metric is
insensitive to C-item failure — and specifies a sensitivity check. A weak design picks revenue
weighting because M5 did.

### 13.9 Exercise 9 arithmetic

Using P and Q from Exercise 1, with positive errors as shortfalls:

```text
P: positive 4 + 6 + 6 + 8 + 12 = 36 ;  negative 8 + 6 + 12 = 26
Q: positive 12 + 10 + 6 + 8 = 36 ;     negative 12 + 12 + 12 + 12 = 48

ratio 1:1   P = (36 + 26)/8 = 7.75    Q = (36 + 48)/8 = 10.50    P wins
ratio 2:1   P = (72 + 26)/8 = 12.25   Q = (72 + 48)/8 = 15.00    P wins
ratio 5:1   P = (180 + 26)/8 = 25.75  Q = (180 + 48)/8 = 28.50   P wins
ratio 10:1  P = (360 + 26)/8 = 48.25  Q = (360 + 48)/8 = 51.00   P wins
```

No break-even exists in the range, because P has both the lower total absolute error and the lower
directional imbalance. The correct answer says so. A break-even requires two methods with similar
total absolute error and opposite directional bias, which is the shape constructed in section 6.
Manufacturing one by changing the numbers has not understood the exercise.

### 13.10 Exercise 10 rubric

A strong specification names a four-week horizon matching the lead time, builds `C_u` and `C_o` from
named components with a price year, states the critical ratio and the scoring rule, and defines the
benchmark precisely enough to compute. It states grain, population, zero handling, weighting, and
owner as a role, plus the range of cost ratios over which the model remains preferred. A weak
specification gives formulas without a cost build-up or omits the benchmark.

## 14. Interview questions

**1. "Our forecast accuracy is 82%. Is that good?"**

A strong answer says the question needs four things first: the metric definition, the grain, the benchmark, and the cost of a unit of error, then asks what decision the forecast drives and what a unit short versus a unit over costs. A weak answer compares 82% to an industry average that may not measure the same thing.

**2. "Why is a point forecast the wrong object for an inventory decision?"**

A strong answer explains that the decision is a quantile of the demand distribution set by the critical ratio, and that a point forecast may be a mean, a median, or neither depending on how it was trained; a median is optimal only when `C_u = C_o`. A weak answer says a point forecast "lacks uncertainty", which misses that the problem is the functional, not the variance.

**3. "Derive the newsvendor critical fractile."**

A strong answer gives the marginal argument in words and symbols, arrives at `q* = C_u/(C_u+C_o)`, and notes that only the ratio matters and that the result assumes one period and linear costs. A weak answer recites the formula without being able to say what the costs mean.

**4. "What is the difference between accuracy and cost-weighted accuracy?"**

A strong answer explains that accuracy is `1 −` an error measure while cost-weighted accuracy is `1 −` realised cost over counterfactual cost, that the two can rank models differently, and constructs the case where the more accurate forecast is more expensive. A weak answer treats them as two ways of saying the same thing.

**5. "Why does MAPE survive if everyone knows it is flawed?"**

A strong answer gives the historical reasons — spreadsheet convenience, scale-free arithmetic, "100 minus MAPE" reading as a grade — then the institutional ones: it is the system default, it is in the template, and replacing it needs a governance decision nobody owns. A weak answer says people do not know better, which is condescending and usually wrong.

**6. "A model has a better MAE but a worse pinball loss at the 0.8 quantile. Which do you deploy?"**

A strong answer asks what decision the forecast serves: at the 0.8 quantile pinball is the selection criterion, and the better-pinball model may be biased as a median estimate, which is acceptable when the decision is the quantile. A weak answer picks the better MAE because it is familiar.

**7. "How do you set a tracking-signal threshold?"**

A strong answer describes the statistic, notes that ±4 is a practitioner convention rather than a significance level, and explains that its variance grows with the window length, so the threshold is meaningless without the window; it recommends a fixed rolling window and a cumulative-error plot. A weak answer quotes ±4 as a probability statement.

**8. "Your category accuracy is 95% but regional service is falling. How is that possible?"**

A strong answer explains cancellation across items and locations and the difference between a volume-weighted aggregate and an unweighted item-level view: a category can be accurate in total while a third of its items are badly wrong. A weak answer blames the regional team's measurement, which is possible but not the first hypothesis.

**9. "How would you present forecast performance to a commercial director?"**

A strong answer leads with currency per period, names a counterfactual, gives a range from a cost sensitivity, and connects the number to a decision with an owner, keeping accuracy as a secondary line for continuity. A weak answer leads with a percentage and a trend arrow.

**10. "What is the cost of a stock-out?"**

A strong answer decomposes it into lost contribution margin, expedite or substitution cost, contractual penalty, and customer lifetime effect, the last included only with evidence, and notes that the answer is item-specific and that gross margin alone understates it for high-service items. A weak answer gives a single number, usually the margin.

**11. "When would you not use a newsvendor framing?"**

A strong answer gives the conditions: multi-period inventory with carryover, capacitated supply that truncates the achievable quantile, substitutable products, and decisions dominated by case pack or contract, while noting that the fractile remains a useful first approximation. A weak answer says the newsvendor is too theoretical, which is an assertion rather than a condition.

**12. "How do you stop a metric from being changed to make a model look good?"**

A strong answer describes pre-registration, versioning, a published specification with numerator, denominator, grain, weighting, and benchmark, and a restatement note when the definition changes, and notes that the control is organisational rather than statistical. A weak answer says "use a good metric", which does not address the incentive.

## 15. Key terms

- **Forecast error** — actual minus forecast, in the units of the series, sign stated.
- **MAE** — mean absolute error; linear penalty; targets the median; units.
- **RMSE** — root mean squared error; quadratic penalty; targets the mean; units.
- **MAPE** — mean absolute percentage error; undefined at zero; asymmetric toward small actuals.
- **sMAPE** — symmetric MAPE; bounded at 200%; not actually symmetric.
- **WAPE** — weighted absolute percentage error; total absolute error over total actual.
- **MASE** — mean absolute scaled error; MAE over in-sample naive MAE; dimensionless.
- **Bias** — mean signed error; direction and magnitude; units.
- **Tracking signal** — running sum of forecast errors divided by mean absolute deviation.
- **Trip value** — the threshold at which a tracking signal flags bias; convention, not a test.
- **Pinball loss** — the loss consistent for a quantile; asymmetric by construction.
- **Quantile forecast** — a forecast of the value demand will not exceed with probability `q`.
- **Critical ratio** — `C_u/(C_u+C_o)`; the cost-minimising quantile.
- **Underage cost (`C_u`)** — cost of one unit short, including margin, expedite, and penalty.
- **Overage cost (`C_o`)** — cost of one unit left, net of salvage and markdown recovery.
- **Salvage value** — recovery obtained for an unsold unit.
- **Newsvendor problem** — single-period quantity choice before demand is known.
- **Cost of forecast error** — currency cost of error against a stated counterfactual.
- **Asymmetric loss** — a loss with different unit costs in each direction.
- **Proper scoring rule** — a score optimised by reporting the true predictive distribution.
- **CRPS** — continuous ranked probability score; the integral of quantile loss over all levels.
- **Forecast value added** — improvement over a named benchmark attributable to one process step.
- **Coverage** — fraction of outcomes inside a stated prediction interval.
- **Calibration** — agreement between stated probabilities and observed frequencies.
- **Cost-weighted accuracy** — one minus realised cost over counterfactual cost.
- **Aggregation weight** — the weight applied to a series when errors are pooled; a decision.
- **Censored demand** — demand truncated by available stock, understating true demand.

## 16. Sources

**Peer-reviewed and academic — Grade A**

- Arrow, K. J., Harris, T., and Marschak, J. (1951). "Optimal Inventory Policy." *Econometrica*, 19(3), 250–272. Original newsvendor formulation.
- Trigg, D. W. (1964). "Monitoring a Forecasting System." *Operational Research Quarterly*, 15(3), 271–274. The tracking signal as a control statistic.
- Goodwin, P. and Lawton, R. (1999). "On the Asymmetry of the Symmetric MAPE." *International Journal of Forecasting*, 15(4), 405–408. DOI: 10.1016/S0169-2070(99)00007-2.
- Hyndman, R. J. and Koehler, A. B. (2006). "Another Look at Measures of Forecast Accuracy." *International Journal of Forecasting*, 22(4), 679–688. DOI: 10.1016/j.ijforecast.2006.03.001.
- Gneiting, T. and Raftery, A. E. (2007). "Strictly Proper Scoring Rules, Prediction, and Estimation." *Journal of the American Statistical Association*, 102(477), 359–378. DOI: 10.1198/016214506000001437.
- Gneiting, T. (2011). "Making and Evaluating Point Forecasts." *JASA*, 106(494), 746–762. DOI: 10.1198/jasa.2011.r10138.
- Gneiting, T., Wolffram, D., Resin, J., et al. (2023). "Model Diagnostics and Forecast Evaluation for Quantiles." *Annual Review of Statistics and Its Application*, 10, 597–621.
- Gneiting, T. and Resin, J. (2023). "Regression Diagnostics Meets Forecast Evaluation." *Electronic Journal of Statistics*, 17, 3226–3286. DOI: 10.1214/23-ejs2180.
- Makridakis, S., Spiliotis, E., and Assimakopoulos, V. (2020). "The M4 Competition." *International Journal of Forecasting*, 36(1).
- Makridakis, S., Spiliotis, E., and Assimakopoulos, V. (2022). "The M5 Accuracy Competition." *International Journal of Forecasting*, 38(4).
- Bruzda, J. (2016). "Quantile Forecasting in Operational Planning and Inventory Management." *Dynamic Econometric Models*, 16. DOI: 10.12775/dem.2016.001.
- Kolassa, S. (2026). "The Symmetric Mean Absolute Percentage Error: Unnecessary or Dangerous." *Forecasting*, 8(2), 24. MDPI.
- Syntetos, A. A. and Boylan, J. E. (2005). "The Accuracy of Intermittent Demand Estimates." *International Journal of Forecasting*, 21(2), 303–314.
- Gogolev, D. and Ozhegov, E. (2023). "Asymmetric Loss Function in Product-Level Sales Forecasting." *Applied Econometrics*, 70, 109–121. DOI: 10.22394/1993-7601-2023-70-109-121.
- Petropoulos, F., et al. (2022). "Forecasting: Theory and Practice." *International Journal of Forecasting*, 38(3).

**Working papers and conference proceedings — Grade B**

- Liu, S., Letchford, A., and Svetunkov, I. "Newsvendor Problems: An Integrated Method for Estimation and Optimisation." Working paper, Lancaster University; reported as forthcoming in *European Journal of Operational Research*. Consult the published version before citing.
- Saltzman, R. M. (2003). "Coordinating Demand Forecasting and Operational Decision-Making with Asymmetric Costs: The Trend Case." Working paper.
- Nguyen, C., Ni, Y., and Rossetti, M. (2010). "Exploring the Cost of Forecast Error in Inventory Systems." *Proceedings of the Industrial Engineering Research Conference*.
- Sosa, A. and Wellens, A. (2016). "Service Level Definition from Forecast Error Cost Evaluation." *Proceedings of the European Modeling and Simulation Symposium*.
- Athanasopoulos, G. and Kourentzes, N. (2020). "On the Evaluation of Hierarchical Forecasts." Monash Business School working paper 02/20.
- Hollyman, R., Petropoulos, F., and Tipping, M. E. (2022). "Hierarchies Everywhere — Managing and Measuring Uncertainty in Hierarchical Time Series." arXiv:2209.15583.
- Krishnamurthy, V. (2006). "A Comparison of the Relative Efficiency of Tracking Signals in Forecast Control." Doctoral thesis, West Virginia University.
- "Stochastic Inventory Optimization with Coherent Risk Measures." *Journal of Risk and Financial Management*, 2026. DOI: 10.3390/jrfm19030173. Numerical results in section 9.3 are reported as the published abstract states them and were not independently replicated.

**Standards, official, and regulatory — Grade B**

- ISO 3534-1, *Statistics — Vocabulary and Symbols — Part 1*. Current edition.
- ISO 5725-1, *Accuracy (Trueness and Precision) of Measurement Methods and Results — Part 1*.
- ISO 80000-1, *Quantities and Units — Part 1: General*.
- ISO 8601, *Date and Time — Representations for Information Interchange*.
- JCGM 200:2008, *International Vocabulary of Metrology* (VIM).
- IAS 2, *Inventories*. IFRS Foundation.
- ASC 330, *Inventory*. Financial Accounting Standards Board.
- IFRS 15, *Revenue from Contracts with Customers*; ASC 606.
- Sarbanes-Oxley Act of 2002, Section 404, internal control over financial reporting.
- Directive 2008/98/EC on waste (Waste Framework Directive), as amended.
- French food-waste legislation of 2016. Consult the current consolidated text before relying on
  any obligation.

**Books and professional references — Grade B or C**

- Brown, R. G. (1959). *Statistical Forecasting for Inventory Control*. McGraw-Hill. Grade B.
- Holt, C. C. (1957). "Forecasting Seasonals and Trends by Exponentially Weighted Moving Averages." Office of Naval Research memorandum 52; reprinted in the *International Journal of Forecasting*, 2004. Grade B.
- Silver, E. A., Pyke, D. F., and Thomas, D. J. *Inventory and Production Management in Supply Chains*. Current edition. Grade B.
- Hyndman, R. J. and Athanasopoulos, G. *Forecasting: Principles and Practice*. Current edition. Grade B.
- Davydenko, A. and Fildes, R. (2016). "Forecast Error Measures: Critical Review and Practical Recommendations." In *Business Forecasting: Practical Problems and Solutions*. Wiley. Grade B.
- APICS/ASCM Dictionary, current edition. Tracking signal and trip values. Grade C.

**Practitioner and vendor — Grade C or D**

- Catt, P. (2007). "Assessing the Cost of Forecast Error: A Practical Example." *Foresight*. Grade C.
- Jain, C. L. (2003). "How to Measure the Cost of a Forecast Error." *Journal of Business Forecasting*. Grade C.
- "FVA: A Reality Check on Forecasting Practices." *Foresight*, Spring 2013. Grade C.
- SAP Supply Chain Management documentation, "Trigg's Tracking Signal." Grade B for the behaviour of that specific implementation; not evidence of general best practice.
- Brightwork Research and Analysis, "Lost Sales Costs Calculator." Grade D. Cites asymmetry ratios far higher than any peer-reviewed source here; treat as an illustration of range, not a measured benchmark.
- Kvantis Atlas, "The Newsvendor Model." Grade D. A vendor explainer; its formula agrees with the peer-reviewed sources above, which is the only reason to cite it.

**Note on citations.** Bibliographic details were checked against the sources returned by the searches. Volume and page numbers are given where confirmed and omitted where not. The numeric results attributed to Catt, Nguyen et al., Sosa and Wellens, Saltzman, and the 2026 quantile-newsvendor study are reported as those sources state them and were not independently replicated. The cost example in section 6 is constructed and internally consistent, not drawn from a published case, and every derived number shows its derivation.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`08_rolling-origin-backtesting.md`](08_rolling-origin-backtesting.md)
- Next topic: topic 10, *Overrides and Forecast Value Added* (`10_overrides-fva.md`, planned)
- Forecastability and the benchmark habit: [`01_forecastability-benchmarks.md`](01_forecastability-benchmarks.md)
- Hierarchy and aggregation: [`05_hierarchy-aggregation.md`](05_hierarchy-aggregation.md)
- Intermittent demand and censored sales: [`06_intermittent-demand.md`](06_intermittent-demand.md)
- Combination and benchmark comparison: [`07_forecast-combination.md`](07_forecast-combination.md)
- The definitions this topic assumes: [`../../01_foundations/01A_foundations_units_time_assumptions/09_forecast-error-metrics.md`](../../01_foundations/01A_foundations_units_time_assumptions/09_forecast-error-metrics.md)
- Intermittent measures: [`../../01_foundations/01A_foundations_units_time_assumptions/10_intermittent-metrics.md`](../../01_foundations/01A_foundations_units_time_assumptions/10_intermittent-metrics.md)
- Demand and lead-time distributions: [`../../01_foundations/01A_foundations_units_time_assumptions/11_demand-lead-time-distributions.md`](../../01_foundations/01A_foundations_units_time_assumptions/11_demand-lead-time-distributions.md)
- Cost and working capital: [`../../01_foundations/01_supply_chain_foundations/06_cost-and-working-capital.md`](../../01_foundations/01_supply_chain_foundations/06_cost-and-working-capital.md)
- Inventory policy that consumes the quantile: [`../10_inventory_optimization_multi_echelon/_index.md`](../10_inventory_optimization_multi_echelon/_index.md)
- Probabilistic and hierarchical scoring in depth: [`../08_hierarchical_intermittent_probabilistic_forecasting/_index.md`](../08_hierarchical_intermittent_probabilistic_forecasting/_index.md)
- Survey file: [`../../../course/07_time_series_forecasting.md`](../../../course/07_time_series_forecasting.md)
- 01A module map: [`../../01_foundations/01A_foundations_units_time_assumptions/_index.md`](../../01_foundations/01A_foundations_units_time_assumptions/_index.md)
- Foundations part map: [`../../01_foundations/_index.md`](../../01_foundations/_index.md)
- Return to the part map: [`../_index.md`](../_index.md)
