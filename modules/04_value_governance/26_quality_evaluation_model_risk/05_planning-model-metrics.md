# Metrics That Matter for Planning Models

> Module 26 — Quality, Evaluation, and Model Risk  
> Topic 05 of 10  
> Estimated study time: 4 hours  
> Prerequisites: Modules 06, 07, and 25; see the bridge below  
> Version 2026.5

## 1. Why this topic exists

A forecasting team may report that its model improved accuracy by five percentage points.
The planner still needs to know whether the model reduces shortages, excess stock, costly schedule changes, or some combination of them.
An average percentage error alone cannot answer that question.
A model can look accurate across a network while systematically underforecasting the products that constrain customer service.
It can also have unbiased total volume while making large errors that cancel across products or weeks.
A narrow interval may look useful but be unreliable precisely when a supplier disruption makes risk visible.

The decision is not merely which score is smallest.
A planning organization must select metrics whose data, timing, aggregation, and consequences correspond to the decision the model actually supports.
It must agree on a small scorecard, document how each measure is calculated, and define what action follows a breach.
The same model can be suitable for a monthly purchasing review and unsuitable for daily order automation.
A lower average error does not establish that the forecasts are stable enough to avoid nervous replenishment.
A well-calibrated probability does not establish that the resulting action is economical when costs or constraints differ.

The stakeholders see different parts of the problem.
A demand planner wants interpretable exceptions and lead-time visibility.
A supply planner wants the effect on feasible orders and capacity.
A finance partner wants transparent currency assumptions and a defensible comparison.
A validator wants an out-of-sample population, sound denominators, and uncertainty.
A business owner must decide whether the evidence is strong enough for the approved use.

### Prerequisite bridge

Module 06 is deferred in this edition.
For this topic, a dataset is a set of observations with a defined source, time, population, unit, and quality limitations.
Training means choosing a model or its parameters using data; evaluation means measuring it on observations not used to make that choice.
A baseline is a simpler comparator evaluated on the same observations.
A forecast estimates an uncertain future outcome, a plan is an intended action, a target is desired performance, and a commitment is an authorized obligation.
Module 07 develops forecasting and temporal evaluation; Module 25 develops business value and counterfactual reasoning.
No machine-learning implementation is required here.

This unit uses demand and replenishment because they make the consequences visible.
The same logic applies to lead-time forecasts, capacity estimates, late-delivery probabilities, and production-yield predictions if the outcome, decision, and units are changed explicitly.
It does not prescribe a universal threshold or a legal compliance test.

## 2. Learning outcomes

After completing this topic, you can:

1. Define forecast error and calculate mean absolute error, root mean squared error, weighted absolute percentage error, and signed bias; see Exercises 1 and 2.
2. Explain zero denominators, aggregation, cancellation, and why alternative weighting changes the business question; see Exercise 2.
3. Evaluate probabilistic forecasts with coverage, interval width, quantile loss, and probability calibration; see Exercise 3.
4. Measure forecast revisions at a fixed horizon and separate instability from justified response to information; see Exercise 4.
5. Translate predictions into constrained actions and compare asymmetric shortage and excess costs with a stated baseline; see Exercise 5.
6. Design a temporal, segmented scorecard with clear denominators, ownership, and action thresholds; see Exercise 6.
7. Challenge an apparent aggregate improvement using segment performance and uncertainty; see Exercise 7.
8. Specify governance for revisions, overrides, missing outcomes, and monitoring; see Exercise 8.

You should be able to reproduce every calculation without assuming the outcome is known at forecast issue time.
A strong answer states whether lower or higher is better, the unit, and what the measure does not establish.

## 3. The core mental model

```text
Decision and service constraints
              |
              v
Forecast issued at known origin and horizon
              |
              +---- point value, quantiles or event probability
              |
              v
Planning rule + constraints + authorized override
              |
              v
Order / capacity action -----> realized demand and outcome
              |                         |
              +-----------+-------------+
                          v
       Matched observations and comparable baseline
                          |
          +---------------+----------------+
          |               |                |
       Accuracy       Calibration       Stability
       and bias       and sharpness     of revisions
          |               |                |
          +---------------+----------------+
                          |
                          v
      Decision consequences, segments, uncertainty
                          |
                          v
       Owner reviews evidence -> action -> record
```

The first arrow matters as much as the final number.
Forecasts must be those available when the decision was made, not reconstructed using later corrections.
The comparison target must represent the quantity the forecast intended to predict.
Sales can be lower than demand when stock was unavailable; a model can appear to overforecast simply because the outcome is censored.

A point forecast is one number per product, location, target period, issue time, and horizon.
It is not a complete representation of uncertainty.
A probability forecast concerns an event, such as demand exceeding a capacity threshold.
A quantile describes a conditional distribution: a calibrated 80th percentile should be exceeded approximately 20% of the time in a suitable evaluation population.

Accuracy describes the size of point errors.
Bias describes their signed direction.
Calibration compares predicted uncertainty or probabilities with observed frequencies.
Sharpness describes concentration of the predictive distribution, independent of what happened.
Stability describes changes between forecast vintages and must be interpreted against newly available information.
Decision cost measures consequences after specifying an action rule and cost model.
None of these dimensions replaces all the others.

The aggregation level is part of the question.
An aggregate forecast can be accurate while errors across items offset one another.
Such offsetting is useful only if the operation can pool inventory or capacity across those items in time.
Calculate error at the decision grain before rolling it up, and also report relevant higher-level totals.

The scorecard is a decision instrument rather than a leaderboard.
The review must identify who responds to a breach, how quickly, what evidence is retained, and what fallback remains available.
A review may retain a model with higher average error when it better protects a critical service segment, provided the trade-off and approval are explicit.

## 4. Definitions and vocabulary

All calculations in this unit use error `e = forecast - actual` unless otherwise stated.
Positive signed bias therefore means overforecasting.
Other organizations use the opposite convention; never compare signed values without checking it.

| Term | Definition | Common misuse / unit |
|---|---|---|
| Forecast origin | Time at which a forecast was available | Using a later revised forecast as the original |
| Target period | Future bucket to which a forecast refers | Mixing daily and weekly totals |
| Horizon | Distance from origin to target period | Pooling short and long horizons without labels |
| Decision grain | Item, location, period, and other keys at which action is taken | Measuring only a network total |
| Actual | Measured outcome matched to the forecast target | Treating constrained sales as unconstrained demand |
| Forecast error | Forecast minus actual | Reversing sign without stating it; units of outcome |
| Absolute error | Absolute value of error | Confusing it with signed error; units of outcome |
| Squared error | Error multiplied by itself | Reporting squared units as if they were units |
| Mean absolute error (MAE) | Mean absolute error across matched rows | Comparing products with incomparable unit scales |
| Root mean squared error (RMSE) | Square root of mean squared error | Calling it average absolute error |
| Mean error (ME) | Mean signed error | Assuming zero ME means small errors |
| Weighted absolute percentage error (WAPE) | Sum of absolute errors divided by sum of actuals | Calling it a mean of row percentage errors |
| Mean absolute percentage error (MAPE) | Mean of row absolute errors divided by row actuals | Ignoring zero actuals or near-zero instability |
| Aggregate bias | Sum of signed errors divided by sum of actuals | Treating cancellation as proof of local unbiasedness |
| Bias | Systematic tendency of signed error | Calling every individual error bias |
| Baseline | Simple comparator available at each forecast origin | Comparing against a baseline with later information |
| Calibration | Agreement between stated probabilities and observed frequencies | Equating narrow intervals with calibrated intervals |
| Prediction interval | Range intended to cover a future outcome at a stated probability | Calling a confidence interval for a mean a prediction interval |
| Coverage | Fraction of realized outcomes inside stated intervals | Ignoring interval width or sampling uncertainty |
| Sharpness | Concentration of the predictive distribution | Treating concentration as proof of calibration |
| Quantile | Value with specified cumulative probability | Treating every quantile as the mean |
| Pinball loss | Asymmetric loss for a predicted quantile | Comparing different quantile levels as identical objectives |
| Brier score | Mean squared error of binary-event probabilities | Using it as a measure of point-demand accuracy |
| Revision | Difference between forecasts for the same target at two origins | Comparing different targets as revisions |
| Forecast value added (FVA) | Performance difference after a process step against before it | Calling a correlation causal proof of an override's value |
| Shortage cost | Marginal consequence of under-ordering, as modeled | Assuming every unfilled unit has the same penalty |
| Excess cost | Marginal consequence of over-ordering, as modeled | Charging entire unit purchase price when stock retains value |
| Counterfactual | Outcome under an alternative action or process | Treating simulated outcomes as directly observed |
| Lost sales | Demand not fulfilled and not subsequently backordered | Counting observed sales as total demand during stockouts |
| Segment | Defined subset such as lead-time or product class | Changing segment definitions after observing results |
| Exposure | Count or volume at risk in the evaluated population | Reporting a percentage without its denominator |
| Override | Authorized human change to a forecast or action | Combining forecast and order overrides without distinction |
| Monitoring threshold | Pre-agreed condition that initiates review | Treating an alert as automatic proof of failure |
| Confidence interval | Range from an inferential procedure for an unknown quantity | Treating it as a range for a future demand observation |

## 5. The mechanics, step by step

### Step 1 — Specify the decision before selecting metrics

Input: a named business decision and its owner.
Record the action, the decision date, product-location population, time bucket, lead time, constraints, and service obligations.
For example, a planner might choose weekly order quantities for an item at a distribution center, with a two-week lead time and a minimum order quantity.
A forecast for tomorrow cannot be substituted for a forecast covering the two-week exposure without explaining the transformation.

Write down the intended use of each output.
A mean forecast might support an aggregate financial projection.
A higher quantile might support a stock buffer when shortages cost more than excess.
A probability of capacity breach might trigger an expedited reservation.
The same decision can require more than one output and more than one metric.

Output: a decision statement and a list of actions that can be taken on evidence.
Branch: if the organization cannot identify the action, report descriptive forecast performance but do not claim demonstrated business value.
If safety or contractual requirements are binding, treat them as constraints rather than merely adding a dollar penalty.

### Step 2 — Define the outcome and freeze the information set

Input: a forecast archive, an outcome source, and the decision calendar.
Create a key containing model version, origin timestamp, target period, horizon, item, location, unit, and forecast type.
Freeze the input information available at each origin.
Retain both raw forecasts and any planner-adjusted forecasts as separate versions.
Record any transformation from a forecast into an order.

Decide whether the target is orders, shipments, sales, consumption, or unconstrained demand.
These are different quantities.
For example, observed sales are not full demand when shelves are empty.
Backorders and lost sales require distinct treatment because later fulfillment can otherwise be counted twice.
Document the actuals revision policy: initial operational actuals and final corrected actuals may answer different questions.

Report missing or unmatched rows by model and segment.
Do not quietly compare models on different populations.
If an outcome is unavailable because the action affected observation, the missingness is part of the evaluation problem, not a row-cleaning inconvenience.
Output: a reproducible matched panel and a data-quality exception ledger.

### Step 3 — Establish a comparable baseline

Input: the matched panel and the information available at each historical origin.
Choose a baseline appropriate to the decision, such as last observed week, seasonal naive, or the existing approved process.
A seasonal naive forecast uses the observation from the corresponding previous season, provided it was available at the origin.
If actuals were reported with delay, the last calendar week may not have been available.
Apply the same grain, horizon, target definition, exclusions, and period to the candidate and baseline.

A model can improve on a weak comparator yet fail to improve on the current process.
A no-change benchmark is also useful when evaluating forecast revisions.
Do not retune the baseline on the test period while keeping the candidate fixed, or vice versa.
Output: paired predictions and actions on the same eligible decisions.

### Step 4 — Choose evaluation windows and segments

Use rolling origins or a chronological holdout that respects data availability and the operational retraining schedule.
Each prediction must be generated without observing its future target.
Separate development, model selection, and final evaluation periods.
If a model is changed after seeing the test results, that period is no longer an untouched final test for the changed model.

Specify each segment before reviewing results: horizon, item class, location, volume, intermittency, promotion status, and criticality are common examples.
Allow exploratory segments but label them as discovered after inspection.
Show segment sample counts and denominators alongside scores.
If sample counts are small, accumulate more evidence or use uncertainty intervals rather than treating a dramatic percentage as conclusive.
Output: evaluation schedule and segment register.

### Step 5 — Calculate point accuracy

For `n` matched rows with forecast `f_i` and actual `y_i`, define `e_i = f_i - y_i`.
Assume actual demand is nonnegative for the percentage formulas below.

- `MAE = sum(abs(e_i)) / n`, in units per row.
- `RMSE = sqrt(sum(e_i^2) / n)`, in units per row.
- `WAPE = sum(abs(e_i)) / sum(y_i)`, a dimensionless ratio, often multiplied by 100%.
- `MAPE = mean(abs(e_i) / y_i)` only when every included `y_i` is nonzero.

MAE gives each evaluated row equal weight in absolute units.
RMSE emphasizes large errors because squaring increases their contribution.
WAPE weights row-level percentage errors by actual volume when actuals are positive, and still counts the absolute error for zero-actual rows in its numerator.
A zero total actual makes WAPE undefined; do not replace that denominator with an arbitrary small constant.
MAPE is undefined on zero actuals and highly sensitive to near-zero actuals.
Removing zero-actual rows silently changes the population and can reward a model that produces large forecasts when no demand occurs.

Report numerator, denominator, count, and the share of zero-actual observations.
If the total actual is zero, show absolute error in units and flag WAPE as undefined.
Compare absolute metrics only where units and aggregation support the comparison.
Multiplying errors by price or margin can be useful, but it changes the weighting and does not automatically estimate loss.

### Step 6 — Calculate bias without hiding cancellation

`ME = sum(e_i) / n` measures average signed error in units per row.
`Aggregate bias = sum(e_i) / sum(y_i)` expresses signed error relative to total actual volume, when the denominator is positive.
Positive values indicate overforecast under this unit's convention.
Report the absolute-error measure next to bias.
A zero aggregate bias can result from large positive and negative errors offsetting across products or time.

Calculate signed errors by segment and inspect persistent runs.
A segment that underforecasts for several periods can generate shortages even when another segment overforecasts by the same total.
Consider whether signs reflect structural demand change, stale promotion inputs, censoring, or a business adjustment.
Do not automatically subtract observed bias from future forecasts without evaluating the correction on later data.

### Step 7 — Evaluate predictive distributions

For a central 80% interval, compare actuals with the lower and upper bounds.
Observed coverage is the count inside the interval divided by the count of eligible intervals.
Use a consistent rule for endpoints; this unit includes both endpoints.
Report average width in demand units as well as coverage.
An extremely wide interval can have high coverage and be practically useless.
A narrow interval with low coverage gives false reassurance.

For quantile `q` and forecast `z`, define pinball loss as `q*(y-z)` when `y >= z`, and `(1-q)*(z-y)` otherwise.
It has units of the outcome and penalizes underprediction more heavily for a high quantile.
Evaluate each quantile with its intended `q` and check that lower quantiles do not exceed higher quantiles.
A quantile's empirical coverage alone is not enough: a proper quantile score also rewards informative magnitudes.

For a binary event with predicted probability `p` and outcome `o` equal to zero or one, the Brier score is the mean of `(p-o)^2`.
Lower is better, but prevalence affects the baseline and interpretation.
A reliability table groups similar probabilities and compares average prediction with event frequency and sample count.
Do not conclude that a probability is calibrated for every subgroup because the overall table looks good.

### Step 8 — Measure revision stability

Match consecutive forecast vintages for the same item, location, and target period.
Compute `revision = later forecast - earlier forecast` and summarize absolute revisions in units.
State the horizons being compared; comparing a four-week-ahead forecast with a three-week-ahead forecast is different from comparing two runs at the same horizon for different targets.
Optionally scale the sum of absolute revisions by a stated positive volume denominator.
A small revision is not automatically desirable if new information legitimately changes expected demand.

Compare revisions with the no-change forecast and with information arrival, such as a confirmed promotion or supplier disruption.
Look for alternating large revisions that cause order churn.
Measure the actual downstream action change separately: forecasts can move while order quantities remain unchanged due to constraints, and small forecast changes can cross a minimum-order threshold.

### Step 9 — Connect the forecast to the actual decision

Specify the decision rule before observing the evaluation outcome.
For a one-period, unconstrained order quantity `Q` and demand `D`, an illustrative cost is `C_u*max(D-Q,0) + C_o*max(Q-D,0)`.
`C_u` is marginal underage cost in currency per unit and `C_o` is marginal overage cost in currency per unit.
This simplified model excludes fixed ordering costs, capacity, multi-period carryover, substitutions, and service constraints unless separately added.

For continuous demand under the standard single-period assumptions, the cost-minimizing quantity is the `C_u/(C_u+C_o)` quantile of demand.
The result is not generally the mean forecast.
When demand is discrete, select a quantity whose cumulative distribution crosses that critical ratio, with ties potentially yielding multiple optima.
If the business uses a different rule, evaluate that actual rule rather than silently replacing it with the theoretical optimum.

Translate both candidate and baseline forecasts into actions under the same lead-time and operating constraints.
Use an observed or validated demand estimate and apply the same cost assumptions to both.
If historical actions affected demand observations, or if alternative actions would change prices, stock availability, customer behavior, or future inventory, a replay is a model-based counterfactual rather than an observed saving.
Output: paired cost results in currency for the evaluated horizon, plus service and constraint results.

### Step 10 — Quantify uncertainty and heterogeneity

Show the number of paired decisions and the distribution of paired differences, not just the difference between two rounded percentages.
Dependence across weeks, items, locations, or overlapping horizons makes a naive independent-row standard error unreliable.
Where data permit, resample suitable time blocks or clusters, preserving relevant dependence, and state the procedure and its limitations.
For a very small sample, report the individual observations and avoid pretending to have a precise population estimate.

Investigate whether volume mix or a few large events drive an aggregate result.
Give decision-critical low-volume segments their own service measure rather than letting high-volume items dominate a weighted score.
Do not optimize dozens of segments and then report only the best one.
Output: an uncertainty statement and a list of segment-level trade-offs.

### Step 11 — Set thresholds and escalation

A threshold is an organizational choice, not a mathematical constant.
Specify metric, population, window, denominator, direction, owner, data lag, and response.
For example, a two-week-horizon critical-item WAPE above an agreed level for two consecutive completed months may trigger a review, while a verified safety incident triggers immediate escalation regardless of WAPE.
Use minimum sample sizes or a separate sparse-data process to avoid reacting to unstable percentages.

An alert initiates diagnosis; it does not by itself prove model failure.
The owner checks data freshness, actuals revisions, mix, stockouts, promotions, and changes to the model or planning rule.
Document the decision to continue, restrict, recalibrate, fall back, or retire, along with authority and review date.
Retain both the alert and the rationale for overriding it.

### Step 12 — Reconcile the operational scorecard

A practical scorecard reports accuracy, bias, uncertainty, revision stability, decision cost, and service or safety constraints side by side.
It identifies the candidate, baseline, version, decision grain, horizon, evaluation dates, and currency assumptions.
It reports coverage of the intended population, missingness, stockout exposure, and material exclusions.
It distinguishes realized operational outcomes from simulated counterfactuals.
The reviewer signs off on use and limitations, not just a favorable metric.

### Additional implementation: the metric specification

Before a dashboard is built, write one specification for each displayed number.
Give the metric an owner and a stable identifier.
State whether a lower or higher value is desirable.
Define the evaluated forecast type, such as unadjusted mean or planner-adjusted median.
Identify the unit, such as cases per item-location-week.
Write the exact numerator and denominator.
State whether the result is a ratio, percentage, currency amount, or physical unit.
Define how null forecasts, null actuals, and zero actuals are handled.
Specify how returns or negative corrections enter the actual.
Specify when the metric is considered final and whether prior results can be restated.
Record the baseline and eligible population alongside the candidate.
Give the metric version a change owner and effective date.

These details avoid common but subtle disagreements.
For example, two teams may both say “monthly WAPE” while one sums weekly absolute errors and the other takes the absolute error of monthly totals.
The former evaluates weekly decisions; the latter allows weekly over- and underforecasts to offset.
If a monthly decision is genuinely pooled across weeks, the latter may be relevant.
It must not be substituted silently for the weekly operating metric.

### Worked aggregation check

Suppose actual weekly demand is 40 and 60 cases.
The corresponding forecasts are 60 and 40 cases.
At weekly grain, the absolute errors are 20 and 20 cases.
Weekly-grain WAPE is `(20 + 20) / (40 + 60) = 40%`.
The two-week actual total and forecast total are both 100 cases.
The absolute error of the two-week totals is therefore zero cases.
The two-week aggregate error ratio is 0%, not 40%.
Both computations are correct for their respective grains.
They answer different questions and should have different labels.
If the order must be committed separately each week, the 0% result hides exposure.
If all 100 cases can be shared without time or item constraints, aggregation may be meaningful.
Physical transfer time, perishability, substitution, and capacity decide whether that pooling assumption is credible.

### Intermittent demand and percentage measures

Intermittent demand contains many periods with no observed demand and occasional nonzero periods.
It is common in spare parts and low-volume items, but the degree of intermittency varies.
An item can also show zeros because it was not offered, because it was out of stock, or because data failed to arrive.
Those causes are not interchangeable.
First distinguish genuine zero demand from missing or censored observations.
Then choose a score that includes all eligible periods.

Suppose actual demand is `[0, 0, 5]` units and the forecast is `[1, 1, 1]` units.
The absolute errors are 1, 1, and 4 units.
MAE is `6/3 = 2` units per period.
WAPE is `6/5 = 120%` over the three periods.
That value is mathematically possible because forecast error can exceed observed demand.
Defining accuracy as `100% - WAPE` would produce negative 20%.
This is an arithmetic transformation, not a probability or a physical accuracy limit.
The MAPE terms for the two zero-actual periods are undefined.
Dropping them leaves only the 80% error in the nonzero period and ignores two unnecessary predicted units.
An all-zero actual window has undefined WAPE even when forecasts are nonzero.
Report the absolute error and count of such windows instead of forcing a percentage.

An intermittent-demand scorecard may combine absolute error, occurrence probability, size conditional on occurrence, service, and inventory consequences.
It should make clear whether a model predicts demand occurrence, positive demand size, or total units.
A strong occurrence forecast can coexist with a poor estimate of the size of a rare order.
A good average total can coexist with unreliable replenishment for the rare high-impact demand.
Evaluate the actual operational action, not just the apparent smoothness of the forecast.

### Scale and weighting choices

MAE is interpretable in the original unit but can be dominated by high-scale items when pooled.
It is not meaningful to add cases of one product to kilograms of another without a justified conversion.
RMSE gives especially large misses more influence; this is useful if large errors are especially costly.
Squaring is a mathematical weighting choice, not a direct proof that consequences grow quadratically.
WAPE divides a sum of absolute errors by total actual volume.
It gives large-volume observations more influence than an unweighted mean of percentage errors.
This can make a model look good overall even while low-volume critical items deteriorate.

Consider a high-volume segment with 900 actual units and 45 units of absolute error.
Its WAPE is `45/900 = 5%`.
A critical low-volume segment has 100 actual units and 30 units of absolute error.
Its WAPE is `30/100 = 30%`.
Combined WAPE is `(45 + 30)/(900 + 100) = 7.5%`.
The combined number is not the simple average of 5% and 30%.
It is a volume-weighted combination because the denominators differ.
Reporting only 7.5% conceals the critical segment's 30% error.
If shortage consequences are severe in that segment, use a separate guardrail.
Do not infer actual shortage cost from WAPE alone because the sign and order rule remain unknown.

Price weighting can make errors in high-value products more visible.
For example, multiplying case errors by a standard unit price produces a currency-weighted deviation.
The resulting amount is not necessarily lost revenue, profit, or incremental inventory expense.
A misforecast can be corrected before execution, and excess stock can retain salvage value.
To estimate consequence, model the action and its marginal costs instead.
If using price weights, specify price date, currency, conversion, and treatment of returns.

### Bias diagnostics beyond one number

With `forecast - actual`, positive mean error is overforecast and negative mean error is underforecast.
Record the sign convention directly on each dashboard.
An indicator named only “bias” is unsafe when different teams use opposite signs.
Signed error in cases is useful for operational magnitude.
Signed error divided by total actual provides scale context when the denominator is positive.
Neither statistic tells you how many individual rows were positive or negative.

Count underforecast, exact forecast, and overforecast rows as separate descriptive measures.
Inspect sequences of negative errors where shortages compound over time.
Inspect sequences of positive errors where inventory accumulates.
Compare the mean signed error with median signed error when a few extreme misses dominate.
Show both row count and volume in each segment.
Check whether an apparent trend is caused by the mix of products, locations, or horizons changing.
Recalculate on a stable cohort where feasible, while separately reporting new and retired products.
Do not discard new items merely because they make the metric less favorable.

A tracking signal is sometimes defined as cumulative signed error divided by an error-scale estimate.
Its sign and denominator vary across organizations.
If used, publish the exact formula, initialization, reset rule, and escalation threshold.
An accumulating numerator can indicate persistence but can also reflect changes in population or scale.
It should prompt investigation rather than automatic manipulation of model parameters.
An illustrative local threshold is a policy choice, not a universally valid statistical test.

### Service metrics and censoring

Service can be expressed at different grains.
An order fill rate might divide units shipped on time by units requested.
A line fill rate might count order lines fully fulfilled.
An on-time delivery measure may count shipments arriving by an agreed date.
These are distinct measures with distinct denominators.
A scorecard must define partial fulfillment, substitutions, cancellations, and timing.

Forecast quality is one contributor to service, not the only one.
Supplier failure, transport disruption, allocation policy, and capacity constraints also matter.
A stockout can happen despite a good forecast, and a poor forecast can be masked by excess inventory.
For this reason, report forecast metrics next to service outcomes rather than equating them.
If a service failure is unacceptable regardless of average cost, impose a guardrail.
Record which team has authority to waive or alter it.

The outcome itself can be distorted by the inventory decision.
If only 80 units were available and 80 were sold, true demand may have been 80 or more.
Calling 80 units total demand without further evidence assumes away any lost sales.
For backorders, distinguish the original requested date from the later shipment date.
Double counting the original order and later fulfillment inflates demand.
For lost sales, use a documented estimation method or flag the observation as censored.
Do not replace unavailable demand with an undocumented guessed number.
Compare sensitivity under several plausible demand assumptions where direct observation is impossible.

### Intervals and probability forecasts in practice

A central prediction interval has a lower and upper quantile.
For an 80% central interval, these are ordinarily the 10th and 90th percentiles.
Other intervals can be asymmetric or selected for a particular operational purpose.
Always state both bounds and their nominal levels.
If the bounds are ordered incorrectly, the forecast distribution is internally inconsistent.
An endpoint rule matters when demand is discrete and observations often equal the bound.
This topic counts an endpoint as covered.
Use the same convention for candidate and baseline.

Coverage can be checked overall, by horizon, and by material segment.
Coverage above nominal does not by itself prove the model is superior.
An extremely wide interval can cover nearly every outcome without supporting useful action.
Coverage below nominal indicates a potential problem but can be noisy in small samples.
Show the count of intervals and the count covered, not just a percentage.
Do not pool 50%, 80%, and 95% intervals into one coverage rate.
Likewise, avoid pooling horizons if uncertainty naturally increases with lead time.

For a probability bin, report predicted probability, number of cases, and observed event frequency.
Suppose ten stockout risks each equal 0.30 and three stockouts occur.
The observed event frequency is `3/10 = 30%` in that bin.
That single bin agrees numerically with the average prediction.
Ten observations are not enough to establish precise calibration or subgroup validity.
If risks are mostly 0.01, a model predicting zero for every event can appear superficially accurate.
Compare with an appropriate prevalence baseline and use proper probability scores.
Separate event definition changes, such as any stockout versus a shortage exceeding ten cases.

### Quantile loss worked in both directions

Take a target quantile of 0.8 and a forecast of 100 cases.
If actual demand is 110 cases, the underprediction is 10 cases.
Pinball loss is `0.8 * 10 = 8` cases.
If actual demand is 90 cases, the overprediction is 10 cases.
Pinball loss is `(1 - 0.8) * 10 = 2` cases.
The same absolute error receives different loss because the target quantile is asymmetric.
That asymmetry corresponds to the intended quantile objective, not to an automatically verified financial cost.
For the median quantile of 0.5, either 10-case error produces 5 cases of pinball loss.
Multiplying every median pinball loss by two yields absolute error.
Compare scores for the same quantile level, population, horizon, and unit.
If a high quantile is used as an order quantity, separately evaluate the actual ordering constraints and economics.

### Revision analysis with horizon discipline

Select one fixed target period and collect all authorized forecast vintages for it.
Each vintage has a timestamp and a remaining horizon.
Join on target, item, location, forecast type, and compatible units.
For successive versions, subtract the older value from the newer value.
Store the reason for a manual override separately from an automatic model update.
Describe whether the two versions include the same information sources.

Suppose a four-week-ahead forecast is 200 cases and a three-week-ahead forecast is 230 cases.
The signed revision is +30 cases and the absolute revision is 30 cases.
If a confirmed promotion was announced between those origins, the increase may be appropriate.
If the increase reverses the following day with no new information, investigate instability.
The fact of a revision alone does not establish either conclusion.
If the order quantity changes by 100 cases because of a minimum order increment, record that action change too.
Conversely, a 30-case forecast change may have no effect when existing inventory covers both values.
The economic effect is mediated by the actual action rule.

For an ex-post analysis, compare each vintage against the same realized outcome.
Do not use that outcome as if it were known when the revision was made.
Compare revision benefit over many paired targets, not just the most favorable example.
Separate routine updates from exceptional updates triggered by newly confirmed events.
If a proposed freeze window reduces revisions, measure missed opportunities to respond as well as avoided churn.

### From scores to decision economics

Start with the current process and its feasible action.
Then specify how the candidate changes the action, including minimum order quantities and capacity.
If an item can be substituted or inventory can carry forward, include those features explicitly.
Underage and overage costs should reflect the incremental consequence of the action.
The purchase cost of unsold but recoverable stock is not automatically a total loss.
Lost contribution, expediting, holding, disposal, and contractual penalties must not be counted twice.
Document currency, price date, time horizon, and owner approval.

Suppose shortage costs USD 8 per unit and excess costs USD 2 per unit.
The critical ratio is 0.8 under the stated single-period model.
If the cost ratio changes to USD 4 shortage and USD 2 excess, the ratio becomes `4/6 = 0.667`.
The preferred quantile changes with the cost assumptions.
That does not mean an 80th-percentile forecast became statistically wrong.
It means the decision maker's economic objective changed.
When costs are uncertain, evaluate a plausible range and identify where the recommended action changes.
If safety or service rules mandate a minimum action, test feasibility before optimizing cost.

One-period cost can underestimate inventory interactions.
An excess unit this week may satisfy demand next week instead of being discarded.
A shortage may create a backorder, a permanently lost sale, or a safety incident.
Orders can have fixed costs and lead-time uncertainty.
Expediting may change the timing and cost of later stock.
Customer substitution can shift demand to a different product.
A practical multi-period evaluation needs an explicit state transition and appropriate validation.
Its outputs remain conditional on the modeled relationships and historical data quality.

### Paired comparison and uncertainty

Compute each model's error on the same target rows.
Then compute the per-row or per-block difference between candidate and baseline.
A paired design controls for much of the shared difficulty of the target period.
It does not remove bias caused by both models using leaked information.
If the candidate is missing for difficult items, report that missingness separately.
Do not fill missing candidate forecasts with the baseline unless that fallback was the actual approved process.
If it was, evaluate the whole process and identify which rows used fallback.

For a pooled WAPE difference, retain each model's total absolute error and common actual denominator.
The difference of ratios with a common denominator is interpretable in percentage points.
A relative reduction divides that difference by the baseline WAPE and requires a nonzero baseline.
If baseline WAPE is zero, a relative reduction is undefined.
Do not divide by a tiny baseline without emphasizing instability.
For operational interpretation, also show the absolute error difference in cases.
If product units are not comparable, analyze compatible units or use a justified normalized grouping.

Uncertainty depends on more than the number of rows.
A regional promotion may affect thousands of item-location records at once.
Those records are not independent evidence of thousands of unrelated outcomes.
Overlapping forecast horizons can share the same demand shock.
Repeated measurements of one item can be serially dependent.
An uncertainty method must reflect the dependence that matters for the intended use.
When resampling time blocks, preserve all relevant items within each selected block when that is the assumed dependence structure.
Document the block choice and avoid treating a bootstrap interval as proof that the assumed sampling design is correct.

### Decision record and release gate

The validation record should identify the candidate and baseline versions.
It should name the authorized use, action rule, forecast horizon, and evaluation dates.
It should record actuals source, revision vintage, and excluded populations.
It should reproduce metric definitions, numerators, denominators, and segment counts.
It should include accuracy, signed bias, calibration, revision stability, and decision consequences.
It should identify any service or safety constraint that overrides the aggregate score.
It should state uncertainty and unresolved data limitations.
It should distinguish observed outcomes from modeled alternatives.
It should name the reviewer, business decision maker, and approval date.

The release decision may be approval, restricted approval, remediation, or rejection.
Restricted approval can limit product classes, geography, horizon, automation, or financial exposure.
It must include monitoring, fallback, a responsible owner, and an expiry or review date.
Remediation should identify a measurable condition for reconsideration.
An unresolved critical-data limitation cannot be converted into evidence by writing a favorable summary.
Retain the evidence for the decision, including unfavorable results and approved exceptions.

### Metric drift versus data drift

Data drift describes a change in input or population distributions.
Metric drift describes a change in measured performance or related operational outcomes.
The first can occur without a material loss of performance.
The second can occur even when obvious input distributions seem stable.
For example, a supplier may change behavior while recorded demand features stay similar.
An aggregate accuracy decline can also be caused by a shift toward longer horizons.
Investigate population mix before concluding the model itself changed.

Use leading indicators while outcomes are unavailable.
These can include missing input rates, late upstream feeds, forecast revisions, and output-distribution shifts.
They are warning signals, not replacements for realized backtesting.
Use lagging indicators when actual demand and service outcomes have matured.
Align the review window with lead time and reporting delay.
An alert must state the data completeness needed for a valid decision.
Repeated alerts without remediation should escalate through a named governance path.

### Operational interpretation checklist

Ask whether the metric addresses the same decision grain and horizon as the intended use.
Ask whether candidate and baseline had the same information.
Ask whether actuals represent demand or only observed sales.
Ask whether missing forecasts, missing actuals, and zero actuals are visible.
Ask whether the sign convention is explicit.
Ask whether absolute errors and signed bias are both shown.
Ask whether volume weighting obscures critical segments.
Ask whether probability calibration is shown with counts and sharpness.
Ask whether forecast revisions are separated from order changes.
Ask whether the cost model includes the actual constraints and marginal consequences.
Ask whether financial results are observed or simulated.
Ask who can approve, restrict, or stop the process.
Ask when the next matured outcomes will be available.
The answers determine whether the number can support a decision.

## 6. Worked example

### Decision and data

A distribution center decides a weekly order quantity for one product family.
The forecast target is weekly demand in cases at the decision grain.
The example is synthetic; all data and unit costs are illustrative, not evidence of an actual company's results.
Assume demand is observed without censoring, no starting inventory remains, lead time is already reflected in the target, and a one-period action can be evaluated independently.
The forecast is a prediction, the quantity ordered is the plan, the service target is a desired outcome, and an approved purchase order is a commitment.

The candidate and current baseline were issued at the same origin for each target week.
The business rule orders the point forecast rounded to cases; no minimum order, capacity, or override applies in this simplified replay.
Actual demand and forecasts are:

| Week | Actual, cases | Candidate, cases | Baseline, cases | Candidate error, cases | Baseline error, cases |
|---:|---:|---:|---:|---:|---:|
| 1 | 100 | 110 | 90 | 10 | -10 |
| 2 | 120 | 110 | 100 | -10 | -20 |
| 3 | 80 | 90 | 100 | 10 | 20 |
| 4 | 100 | 90 | 110 | -10 | 10 |
| Total | 400 | 400 | 400 | 0 | 0 |

The total actual is 400 cases over four weeks.
Both models forecast 400 cases in total and both have zero aggregate bias.
Neither fact means that their week-level forecasts were perfect.
There are four separately evaluated decisions, and the time of each order matters.

### Accuracy arithmetic

The candidate's absolute errors are 10, 10, 10, and 10 cases.
Their sum is 40 cases and candidate MAE is `40 / 4 = 10` cases per week.
Its squared errors sum to `100 + 100 + 100 + 100 = 400` cases squared.
Candidate RMSE is `sqrt(400 / 4) = 10` cases per week.
Candidate WAPE is `40 / 400 = 0.10`, or 10%.

The baseline's absolute errors are 10, 20, 20, and 10 cases.
Their sum is 60 cases and baseline MAE is `60 / 4 = 15` cases per week.
Its squared errors sum to `100 + 400 + 400 + 100 = 1,000` cases squared.
Baseline RMSE is `sqrt(1,000 / 4) = sqrt(250)`, approximately 15.81 cases per week.
Baseline WAPE is `60 / 400 = 0.15`, or 15%.
The candidate improves WAPE by 5 percentage points, a one-third relative reduction from the baseline's 15%.
The same four rows and 400-case denominator are used for both models.

Candidate MAPE is `(10/100 + 10/120 + 10/80 + 10/100) / 4`, approximately 10.21%.
Baseline MAPE is `(10/100 + 20/120 + 20/80 + 10/100) / 4`, approximately 15.42%.
These are not WAPE because each week receives equal weight in the average of row percentages.
Every actual in this example is positive; the calculation does not establish a policy for zero-demand weeks.

### Bias and segmentation

Candidate signed errors sum to `10 - 10 + 10 - 10 = 0` cases.
Baseline signed errors sum to `-10 - 20 + 20 + 10 = 0` cases.
Thus each model's mean error and aggregate bias are zero.
Candidate weeks 2 and 4 together underforecast by 20 cases; weeks 1 and 3 overforecast by 20 cases.
An operation unable to move stock across weeks still experiences both shortages and excess.

If weeks 2 and 4 represent promotion weeks, their signed bias is `-20 / (120 + 100) = -9.09%`.
The other two weeks have signed bias `20 / (100 + 80) = +11.11%`.
This segment labeling is illustrative and was not used to select the candidate.
Two rows per segment are insufficient to establish durable systematic bias.

### Interval calibration and sharpness

Suppose the candidate also issued 80% prediction intervals of [90, 120], [95, 125], [85, 105], and [80, 110] cases for weeks 1–4.
Actuals of 100, 120, and 100 cases are inside the respective intervals; the actual of 80 cases in week 3 is below its interval.
Observed coverage is `3 / 4 = 75%`.
The interval widths are 30, 30, 20, and 30 cases; average width is `110 / 4 = 27.5` cases.
This four-week sample is too small to declare the nominal 80% interval calibrated or miscalibrated.
A wider interval could raise coverage, but width must be considered with a proper score and business usefulness.

### Stability arithmetic

Suppose the previous-vintage forecasts for the same four target weeks were 105, 115, 85, and 95 cases.
The later candidate forecasts are 110, 110, 90, and 90 cases.
The revisions are +5, -5, +5, and -5 cases.
Mean absolute revision is `20 / 4 = 5` cases per target week.
Relative to 400 actual cases, the sum of absolute revisions is 5%; that denominator is an ex-post normalization, not information available at issue time.
These figures describe magnitude only; whether the revisions improved decisions requires the earlier forecasts, later information, and subsequent outcomes.

### Decision-cost arithmetic

Assume an underage costs USD 8 per case and an overage costs USD 2 per case for this one-period replay.
These costs are marginal consequences, not item revenue or purchase price by default.
Underage includes the stipulated lost contribution and service penalty; overage includes the stipulated residual handling or disposal cost.
No separate safety or contractual penalty is modeled.

For the candidate, weeks 1 and 3 each have 10 excess cases: `20 * USD 2 = USD 40`.
Weeks 2 and 4 each have 10 shortage cases: `20 * USD 8 = USD 160`.
Total candidate cost is USD 200 across four weeks.

For the baseline, weeks 1 and 2 have 10 and 20 shortage cases: `30 * USD 8 = USD 240`.
Weeks 3 and 4 have 20 and 10 excess cases: `30 * USD 2 = USD 60`.
Total baseline cost is USD 300 across four weeks.
The simulated difference is USD 100 in favor of the candidate over these four weeks, under these exact assumptions.
It is not an annual saving, an audited financial result, or a causal estimate of a live deployment.

The critical ratio from the illustrative costs is `8 / (8 + 2) = 0.8`.
Under the single-period assumptions, an 80th percentile rather than a mean is the theoretical quantity to examine.
The provided point forecasts are not identified as 80th percentiles, so the replay does not prove either action rule is optimal.
A real system would incorporate starting inventory, pipeline orders, lead-time demand, constraints, and carryover before a cost comparison could inform an ordering decision.

### Management interpretation

The candidate beats the baseline on the four-week point metrics and this illustrative cost replay.
Both are unbiased in aggregate but have material week-level misses.
The interval evidence is weak because only four outcomes are available.
The stability statistic needs information on whether revisions were justified.
A responsible decision is to continue evaluation on a longer, representative rolling-origin panel rather than declare a universal production benefit.

## 7. Code, query, or calculation

The following standard-library Python script reproduces the worked example.
Save it as a local `.py` file and run it with Python 3.9 or later.
It uses synthetic, fully observed data and the same `forecast - actual` sign convention.
It is educational calculation code, not a production validation pipeline.
The zero-denominator behavior is explicit; an undefined WAPE returns `None`.

```python
from math import sqrt

actual = [100, 120, 80, 100]
candidate = [110, 110, 90, 90]
baseline = [90, 100, 100, 110]
previous = [105, 115, 85, 95]
intervals = [(90, 120), (95, 125), (85, 105), (80, 110)]


def score(actuals, forecasts):
    if len(actuals) != len(forecasts) or not actuals:
        raise ValueError("Expected equally sized, nonempty arrays")
    if any(y < 0 for y in actuals):
        raise ValueError("This WAPE definition assumes nonnegative actuals")
    errors = [f - y for y, f in zip(actuals, forecasts)]
    n = len(errors)
    actual_total = sum(actuals)
    absolute_total = sum(abs(e) for e in errors)
    return {
        "mae": absolute_total / n,
        "rmse": sqrt(sum(e * e for e in errors) / n),
        "wape": absolute_total / actual_total if actual_total else None,
        "mean_error": sum(errors) / n,
        "bias": sum(errors) / actual_total if actual_total else None,
    }


def replay_cost(actuals, quantities, underage, overage):
    if len(actuals) != len(quantities) or not actuals:
        raise ValueError("Expected equally sized, nonempty arrays")
    return sum(
        underage * max(y - q, 0) + overage * max(q - y, 0)
        for y, q in zip(actuals, quantities)
    )


def interval_summary(actuals, bounds):
    if len(actuals) != len(bounds) or not actuals:
        raise ValueError("Expected equally sized, nonempty arrays")
    if any(lo > hi for lo, hi in bounds):
        raise ValueError("Lower interval endpoint exceeds upper")
    coverage = sum(lo <= y <= hi for y, (lo, hi) in zip(actuals, bounds))
    width = sum(hi - lo for lo, hi in bounds) / len(bounds)
    return coverage / len(actuals), width


def pinball(actual_value, quantile_value, level):
    if not 0 < level < 1:
        raise ValueError("Quantile level must be strictly between 0 and 1")
    residual = actual_value - quantile_value
    return level * residual if residual >= 0 else (1 - level) * -residual


for name, forecasts in (("candidate", candidate), ("baseline", baseline)):
    s = score(actual, forecasts)
    print(
        f"{name}: MAE={s['mae']:.2f}, RMSE={s['rmse']:.2f}, "
        f"WAPE={100*s['wape']:.2f}%, bias={100*s['bias']:.2f}%"
    )
    print(f"{name}: simulated cost USD {replay_cost(actual, forecasts, 8, 2):.2f}")

coverage, width = interval_summary(actual, intervals)
revision = sum(abs(new - old) for old, new in zip(previous, candidate)) / len(actual)
print(f"Interval coverage={100*coverage:.2f}%, mean width={width:.2f} cases")
print(f"Mean absolute revision={revision:.2f} cases")
print(f"Example 0.8-quantile loss={pinball(120, 110, 0.8):.2f} cases")
```

Expected output:

```text
candidate: MAE=10.00, RMSE=10.00, WAPE=10.00%, bias=0.00%
candidate: simulated cost USD 200.00
baseline: MAE=15.00, RMSE=15.81, WAPE=15.00%, bias=0.00%
baseline: simulated cost USD 300.00
Interval coverage=75.00%, mean width=27.50 cases
Mean absolute revision=5.00 cases
Example 0.8-quantile loss=8.00 cases
```

The script uses lists with a common ordering; production systems should join explicit keys instead.
The printed ratio is defined in this example because total actual demand is positive.
A production report must handle undefined percentages instead of formatting `None` as a number.
It must also retain the data and code versions, missing rows, actuals vintage, and action constraints.
The replay is intentionally not an inventory simulator; no causal conclusion follows from its output alone.

## 8. Data, standards, and regulatory context

### Data contract

A usable record identifies the decision key, model and process version, issue timestamp, target period, unit of measure, forecast value, actual value, and source.
Probabilistic outputs additionally need quantile levels or event definitions, interval endpoints, and the probability scale.
Store the time zone and calendar because a local business week may not match a UTC calendar week.
Record whether quantities are in eaches, cases, pallets, weight, or currency and preserve the effective conversion factor.

Late-arriving actuals, returns, cancellations, stockouts, and product substitutions need documented treatment.
If case pack sizes change, preserve the conversion valid at the transaction date.
If products are introduced or retired, define the eligible population and distinguish absence of a forecast from a forecast of zero.
For promotions, retain the information actually available at the origin rather than the final promotion calendar.

Actuals may be revised after an initial close.
Freeze the evaluation extract or version it, and publish the actuals vintage with the score.
Revisions to the result should be traceable to changes in actuals, mapping, code, exclusions, or forecasts.
Store sufficient row-level lineage to reproduce the numerator and denominator, subject to applicable retention and access controls.

### Identifiers and interoperability

GS1 identifies Global Trade Item Numbers (GTINs) and Global Location Numbers (GLNs) as standardized identifiers for trade items and locations.
They can help join datasets across partners when assigned and maintained correctly.
They do not guarantee that product hierarchies, unit conversions, historical mappings, or demand definitions are correct.
Internal item and location identifiers remain valid where the organization has defined and governed them.
A standardized identifier is not a substitute for an effective-dated master-data mapping.

### Model risk guidance

The U.S. Federal Reserve and Office of the Comptroller of the Currency published SR 11-7 / OCC 2011-12 guidance on model risk management for supervised banking organizations in 2011.
Its discussion of conceptual soundness, ongoing monitoring, and outcomes analysis illustrates a governance approach, not a rule that every supply-chain company or forecast is automatically subject to banking supervision.
Applicability depends on the institution, activity, jurisdiction, and current supervisory requirements; obtain qualified advice for a real compliance question.
Do not represent a metric score as a regulatory approval.

The NIST AI Risk Management Framework 1.0, published in 2023, is voluntary guidance for managing AI risk.
Its Govern, Map, Measure, and Manage functions provide useful structure for documenting context, measurement, and response.
It does not prescribe a universal WAPE threshold or determine whether a specific model is suitable for a particular decision.
Where a planning system does not involve AI, the measurement concepts in this topic can still be relevant without labeling the system as AI.

ISO 31000:2018 provides risk-management guidelines rather than a model-specific certification rule.
Use it as a vocabulary for context, risk treatment, and monitoring where appropriate.
The organization must still choose metrics, controls, decision rights, and tolerances suited to its obligations.

### Privacy and sensitive outcomes

Item-location records may reveal customer, facility, supplier, or commercially sensitive information.
Some planning data can contain personal data, for example when highly granular demand is linked to individuals.
Use data minimization, access controls, retention rules, and applicable legal review.
A published aggregate score can still disclose sensitive information when a segment contains very few observations.
Privacy requirements do not remove the need for a reproducible controlled validation record.

## 9. Industry practice and case evidence

### What the public evidence supports

The M5 forecasting competition used Walmart unit sales data made available through the competition and evaluated hierarchical retail forecasts.
The competition's published paper describes the data, evaluation setup, and lessons from submissions.
The data and competition are evidence that forecast accuracy can be evaluated across a large retail hierarchy; they do not establish that a particular participant's model created a specific live inventory saving.
The evaluation is a research benchmark, not a complete reproduction of every retailer's ordering constraints.

[INFERENCE] A retail planning team using hierarchical forecasts should inspect item-location performance and relevant aggregate levels because its operational decisions and pooling opportunities differ by level.
Whether such segmentation improves a specific company's results is an empirical question.
Do not transfer a competition's ranking directly to a new population, horizon, cost structure, or deployment period.

Makridakis and colleagues' M4 competition evaluated many forecasting methods across series and horizons.
Its published findings support the practical value of comparing methods against benchmarks across varied series.
They do not show that any single model is best for every supply-chain decision or that a point-accuracy ranking determines inventory cost.
Forecast competitions also generally cannot reveal all production changes, planner overrides, and business constraints.

### Operating patterns

Many organizations report a volume-weighted error, signed bias, and service metric together because they answer different questions.
The particular formula, population, and horizon must appear in the scorecard; the label "forecast accuracy" alone is ambiguous.
A report that defines accuracy as `100% - WAPE` can be negative when WAPE exceeds 100% and should not be interpreted as a probability.
Do not compare such a score to another company's accuracy without reconciling definitions and selection rules.

A common process compares the statistical forecast with the planner-adjusted forecast and the final consensus forecast.
Paired evaluation can show whether an adjustment stage improved measured error in the observed sample.
The result is not automatically causal because planners choose which forecasts to override based on information or difficulty.
To assess an intervention more credibly, predefine comparison cohorts or use a feasible randomized or otherwise justified design.

[INFERENCE] A company with constrained supply may prefer a forecast and action combination with slightly worse aggregate WAPE if it reduces high-consequence shortages without violating other constraints.
That preference depends on independently agreed costs, service requirements, and out-of-sample evidence.
The metric itself cannot authorize the trade-off.

### Review cadence

Operational reviews often run more frequently than formal model revalidation.
The monitoring window must respect outcome latency: a twelve-week-ahead forecast cannot be fully assessed one week after issue.
A model owner can review input quality and forecast revisions earlier while waiting for target outcomes.
A validator should distinguish this leading evidence from realized accuracy.

The best practice is not to react to every noisy score.
The owner reviews context, exposure, uncertainty, and the persistence of the signal.
Where a segment is critical, one severe realized event can justify investigation even if aggregate metrics remain within limits.
The review record should identify the evidence, decision, accountable approver, and next date.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Actuals leakage | Near-perfect backtest but weak live performance | Later information enters historical forecasts | Inflated approval evidence | Freeze as-of data and reproduce issue-time features |
| Wrong target | Sales fall during stockouts | Sales are treated as uncensored demand | Understated demand and incorrect error | Flag constrained periods and evaluate demand measurement |
| Cancellation | Zero total bias with many large misses | Positive and negative errors offset | False reassurance | Pair bias with absolute error and segments |
| Wrong grain | Excellent network score but local shortages | Errors offset across unpooled locations | Poor allocation | Score at action grain and separately at pooled levels |
| Zero denominators | Missing or extreme percentage rows | MAPE divides by zero or tiny actuals | Selection bias or volatile scores | Publish zero count and use suitable absolute metrics |
| Undefined WAPE | No actual demand in a segment | Total actual denominator is zero | Invented or misleading percentage | Return undefined and report absolute errors |
| Horizon mixing | Overall score shifts after horizon mix changes | Easier short horizons dominate | Apparent improvement without within-horizon gain | Report fixed horizons and mix |
| Baseline leakage | Baseline uses final corrected actuals | Comparator has unavailable information | Unfair comparison | Use the same origin information and eligibility |
| Population mismatch | Candidate and baseline row counts differ | Different exclusions or missing forecasts | Biased relative performance | Evaluate paired rows and publish coverage |
| Cost omission | Savings estimate equals error reduction times price | Revenue or inventory value is mistaken for marginal cost | Inflated business case | Model shortage and excess consequences explicitly |
| Censored counterfactual | Large improvement claimed from historical sales | Alternative stock levels would alter observed sales | Unsupported causal savings | Model censoring and label simulated results |
| Narrow intervals | Low width with frequent misses | Distribution is overconfident | Unprotected disruption risk | Report coverage, width, and proper score |
| Empty calibration bins | Some probability bins have one observation | Sparse evidence mistaken for stability | Overconfident calibration claims | Show counts and intervals; collect more data |
| Forecast nervousness | Frequent large revisions and order changes | Model reacts to noise or unstable inputs | Expedites and schedule churn | Compare revisions with information and action impact |
| Frozen forecast | Very few revisions despite major news | Stability is rewarded regardless of responsiveness | Failure to adapt | Test whether new information improves decisions |
| Improper quantile | Lower bound exceeds upper bound | Quantiles fitted inconsistently | Invalid distribution and confusing actions | Check ordering and remediate upstream |
| Overrides hidden | Final forecast lacks version lineage | Human changes replace original values | No accountability or FVA analysis | Retain raw and adjusted versions with reasons |
| Unrepresentative test | Only ordinary weeks are evaluated | Promotions and disruptions excluded | Poor performance in critical conditions | Define population, report exclusions and stress cases |
| Sparse segment | Extreme percentage based on a few cases | Small denominator or sample count | Unstable escalation | Show exposure and use sparse-data review |
| Stale cost | Unit penalties have not changed in years | Business economics or service obligations changed | Wrong action comparison | Review costs, sources, and scenarios periodically |
| Metric gaming | High-volume safe items dominate dashboard | Target is optimized instead of decision value | Neglect of critical small segments | Maintain service guardrails and segment review |
| No response | Repeated alert without owner | Threshold has no decision rights | Persistent unmanaged model risk | Assign escalation, fallback, and due date |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or respond |
|---|---|---|---|
| Matched origin and outcome | Ensures the forecast addresses the evaluated decision | Revised forecast is substituted | Inspect immutable timestamps and lineage |
| Outcome represents demand | Errors must refer to the intended target | Stockouts censor sales | Compare availability, orders, and lost-sales evidence |
| Consistent unit | Differences require comparable quantities | Case packs change | Reconcile effective-dated unit conversions |
| Comparable population | Relative scores need the same decisions | One model lacks hard items | Publish paired coverage and missingness |
| Stable metric definition | Trend needs consistent calculation | Denominator or exclusions change | Version the metric and backfill transparently |
| Relevant horizon | Decision needs forecasts at an actionable lead time | Pooled horizons conceal deterioration | Report each horizon and decision calendar |
| Reasonable cost inputs | Cost replay depends on marginal consequences | Margin, salvage, or service penalties change | Obtain owner sign-off and run sensitivity scenarios |
| Single-period replay | Simplifies the worked cost calculation | Inventory carries forward or orders interact | Use a validated multi-period inventory model |
| No action effect on outcome | Historical demand can be used under either action | Stock availability changes observed sales | Model censoring and validate counterfactual assumptions |
| Representative evaluation | Past results inform future use | Mix, promotion, or supply regime changes | Compare distributions and evaluate later periods |
| Interval observations are comparable | Coverage has a coherent interpretation | Different nominal levels are pooled | Separate coverage by level and horizon |
| Dependence handled | Uncertainty reflects shared shocks | Overlapping horizons or correlated products | Use appropriate blocks or clusters and disclose limits |
| Overrides are observable | Process stages can be compared | Changes are overwritten or made offline | Retain versioned raw and adjusted outputs |
| Governance is enforceable | Alerts can change decisions | No owner, fallback, or authority exists | Test escalation and record decisions |

A violated assumption does not always require discarding all evidence.
It changes what can be inferred and may require a narrower scope, a different measure, or additional data.
Do not label an unobserved counterfactual as a realized financial benefit.

## 12. Exercises

### Exercise 1 — Accuracy and bias

Time: 15 minutes.
For actuals `[40, 60, 50]` cases and forecasts `[50, 50, 55]` cases, use forecast minus actual.
Deliverable: calculate each signed error, MAE, RMSE, WAPE, and aggregate bias.
Acceptance: show numerators, denominators, units, and direction of bias.
Explain why bias is smaller than WAPE.

### Exercise 2 — Zero demand and aggregation

Time: 15 minutes.
Two items have actuals `[0, 100]` cases and forecasts `[10, 90]` cases in one week.
Deliverable: calculate item-grain WAPE and aggregate-grain absolute error divided by total actual.
Acceptance: explain why MAPE is undefined on one row and why the aggregate calculation hides item-level misses.
State one circumstance where aggregation could nevertheless be operationally relevant.

### Exercise 3 — Probability and quantile performance

Time: 20 minutes.
Four binary stockout forecasts are `[0.2, 0.2, 0.8, 0.8]`, with outcomes `[0, 1, 1, 1]`.
For one separate observation, actual demand is 120 cases and the 0.8 quantile is 110 cases.
Deliverable: compute the Brier score, two probability-bin frequencies, and the pinball loss for the quantile.
Acceptance: give bin counts and explain why four events do not establish calibration.

### Exercise 4 — Revision and information

Time: 15 minutes.
A prior forecast for one target week is 100 cases; the latest is 125 cases; actual demand is 120 cases.
Deliverable: compute signed and absolute revision and both absolute forecast errors.
Acceptance: state whether the revision helped ex post and identify one fact needed to judge whether it was justified at the time.

### Exercise 5 — Decision cost

Time: 20 minutes.
A decision maker orders either 90 or 110 cases; demand realizes as 100 cases.
Underage costs USD 6 per case and overage costs USD 2 per case.
Deliverable: calculate the one-period cost of each order and the critical ratio.
Acceptance: distinguish the realized comparison from a proof that 110 cases is always optimal.

### Exercise 6 — Scorecard design

Time: 30 minutes.
A company places weekly orders at item-location level with lead times of two to six weeks.
Deliverable: design a six-dimension scorecard with metrics, grains, denominators, horizons, owners, and escalation actions.
Acceptance: include accuracy, bias, calibration, stability, cost, and a nonfinancial service guardrail; address zero demand and actuals latency.

### Exercise 7 — Aggregate reversal

Time: 25 minutes.
A candidate improves total WAPE but has worse WAPE for critical low-volume items.
Deliverable: write a recommendation to an approval committee.
Acceptance: identify volume weighting, segment counts, uncertainty, service constraints, and at least one conditional approval or mitigation.
Do not claim a numeric saving without cost inputs.

### Exercise 8 — Monitoring and lineage

Time: 30 minutes.
Planners overwrite forecasts in spreadsheets, actuals are revised after month-end, and no one owns the alert.
Deliverable: propose an evidence flow and an escalation procedure.
Acceptance: retain forecast vintages and adjustments, version actuals and formulas, assign owner and fallback, and explain how the first review handles missing historical evidence.

## 13. Self-check answers

### Exercise 1

Errors are `[10, -10, 5]` cases.
Absolute errors sum to 25 cases; `MAE = 25/3 = 8.33` cases per row.
Squared errors sum to `100 + 100 + 25 = 225` cases squared.
`RMSE = sqrt(225/3) = sqrt(75) = 8.66` cases per row, rounded.
Total actual is 150 cases, so `WAPE = 25/150 = 16.67%`.
Signed errors sum to 5 cases, so aggregate bias is `5/150 = +3.33%`, an overforecast under the stated convention.
The signed errors partially cancel; absolute errors do not.

### Exercise 2

The item-level absolute errors are 10 and 10 cases.
Item-grain WAPE is `20/100 = 20%`.
The aggregate forecast and actual are both 100 cases, so aggregate absolute error is zero and its ratio to actual is 0%.
The aggregate score hides one overforecast and one underforecast.
The first row's MAPE term divides by zero and is undefined.
Aggregation may be operationally relevant if the two items are truly interchangeable and stock can be pooled in time, which must be established rather than assumed.

### Exercise 3

Squared probability errors are `0.04`, `0.64`, `0.04`, and `0.04`.
The Brier score is `0.80/4 = 0.20`.
The 0.2 bin has one event in two observations, or 50%; the 0.8 bin has two events in two observations, or 100%.
Those are noisy observed frequencies, not conclusive calibration estimates.
The 0.8 quantile underpredicts by 10 cases, so pinball loss is `0.8 * 10 = 8` cases.

### Exercise 4

The signed revision is `125 - 100 = +25` cases and absolute revision is 25 cases.
The prior absolute error is `abs(100 - 120) = 20` cases; the latest is `abs(125 - 120) = 5` cases.
The revision helped on this one realized target, but a reviewer needs to know what information was available when it was made and whether comparable revisions help over many targets.

### Exercise 5

Ordering 90 cases leaves 10 cases short, costing `10 * USD 6 = USD 60`.
Ordering 110 cases leaves 10 cases excess, costing `10 * USD 2 = USD 20`.
The critical ratio is `6/(6+2) = 0.75`.
The realized result favors 110 cases for this outcome; expected optimality depends on the demand distribution and model assumptions.

### Exercise 6 rubric

A strong answer defines a point metric and its denominator at item-location and lead-time horizon.
It pairs signed bias with absolute errors and separates nominal interval coverage from width.
It matches forecast vintages for the same target, links forecast outputs to constrained order rules, and includes service or safety outcomes.
It specifies missing and zero actuals, the delayed availability of outcomes, review ownership, and a documented response.
A weak answer lists six metric names with no population or action.

### Exercise 7 rubric

A strong answer refuses to let a volume-weighted aggregate overrule a critical-segment guardrail without an explicit decision.
It asks for paired counts, exposure, uncertainty, and evidence about whether products can be pooled.
It proposes a restricted rollout, fallback, or remedial evaluation with accountable approval if appropriate.
It labels any modeled financial difference as a counterfactual and does not invent costs.
A weak answer chooses the lower aggregate WAPE without examining the segment or claims the model is universally worse from an unstable small sample.

### Exercise 8 rubric

A strong answer records immutable model and planner forecasts, timestamps, reasons, actuals vintages, transformations, and the metric version.
It assigns an alert owner, response time, escalation authority, fallback, and closure evidence.
It treats missing history as a limitation, starts prospective capture, and avoids reconstructing unavailable forecasts as if observed.
A weak answer adds a dashboard but leaves data lineage and decision rights unresolved.

## 14. Interview questions

1. **Why can a zero-bias model still cause shortages?** Strong: signed errors cancel across time or products, so inspect absolute errors and decision grain. Weak: zero bias means every forecast was correct.
2. **When is WAPE undefined?** Strong: total actual volume is zero; report absolute errors and the denominator. Weak: substitute an arbitrary constant without disclosure.
3. **Why can a better WAPE fail to reduce cost?** Strong: asymmetric penalties, action rules, constraints, and segment mix matter. Weak: every unit of absolute error has the same financial consequence.
4. **What does 80% interval coverage mean?** Strong: long-run frequency in a specified evaluation population, with width and uncertainty also assessed. Weak: the realized fixed value has an 80% probability after it is observed.
5. **How do you detect leakage?** Strong: reconstruct origin-time features and compare immutable forecast vintages. Weak: randomly split rows and assume all future information is excluded.
6. **How would you evaluate a planner override?** Strong: retain before and after forecasts and examine paired results while addressing selection. Weak: compare overridden hard cases against unadjusted easy cases as causal proof.
7. **Is a stable forecast always better?** Strong: stability is useful only alongside responsiveness and downstream action. Weak: any revision is failure.
8. **What does the critical ratio represent?** Strong: the target quantile under explicit single-period underage and overage assumptions. Weak: a universal service target regardless of costs and constraints.
9. **Why report counts in calibration bins?** Strong: observed frequencies from sparse bins are uncertain and subgroup results may differ. Weak: a single matching bin establishes calibration everywhere.
10. **What is the difference between a replay and realized savings?** Strong: replay models an unobserved alternative and needs counterfactual assumptions. Weak: a historical simulated difference is automatically an audited saving.

## 15. Key terms

- **Actual:** measured outcome corresponding to a forecast target; see [definitions](#4-definitions-and-vocabulary).
- **Aggregate bias:** signed error divided by total actual volume; see [mechanics](#step-6--calculate-bias-without-hiding-cancellation).
- **Baseline:** comparable simpler or existing alternative; see [mechanics](#step-3--establish-a-comparable-baseline).
- **Brier score:** mean squared error for binary probabilities; see [Exercise 3](#exercise-3--probability-and-quantile-performance).
- **Calibration:** agreement between predicted probabilities and observed frequencies; see [mechanics](#step-7--evaluate-predictive-distributions).
- **Censoring:** observation limited by conditions such as stock availability; see [data contract](#data-contract).
- **Counterfactual:** outcome under an alternative unobserved action; see [decision cost](#step-9--connect-the-forecast-to-the-actual-decision).
- **Coverage:** fraction of outcomes within a prediction interval; see [worked example](#interval-calibration-and-sharpness).
- **Critical ratio:** underage cost divided by total underage and overage costs; see [decision cost](#step-9--connect-the-forecast-to-the-actual-decision).
- **Decision grain:** level at which action is chosen; see [mental model](#3-the-core-mental-model).
- **Error:** forecast minus actual in this unit; see [definitions](#4-definitions-and-vocabulary).
- **Forecast origin:** time the forecast was available; see [data contract](#data-contract).
- **Horizon:** distance from forecast origin to target; see [mechanics](#step-4--choose-evaluation-windows-and-segments).
- **MAE:** mean absolute error in the outcome's units; see [point accuracy](#step-5--calculate-point-accuracy).
- **MAPE:** mean of row-level absolute percentage errors; see [point accuracy](#step-5--calculate-point-accuracy).
- **Mean error:** average signed error; see [bias](#step-6--calculate-bias-without-hiding-cancellation).
- **Overage cost:** marginal cost of ordering above demand in the simplified model; see [worked example](#decision-cost-arithmetic).
- **Pinball loss:** asymmetric quantile score; see [probabilistic evaluation](#step-7--evaluate-predictive-distributions).
- **Prediction interval:** range intended to cover a future outcome at a specified probability; see [worked example](#interval-calibration-and-sharpness).
- **Quantile:** value associated with a stated cumulative probability; see [probabilistic evaluation](#step-7--evaluate-predictive-distributions).
- **Revision:** change between forecast vintages for the same target; see [stability](#step-8--measure-revision-stability).
- **RMSE:** square root of mean squared error; see [point accuracy](#step-5--calculate-point-accuracy).
- **Sharpness:** concentration of a predictive distribution; see [probabilistic evaluation](#step-7--evaluate-predictive-distributions).
- **Underage cost:** marginal cost of ordering below demand in the simplified model; see [worked example](#decision-cost-arithmetic).
- **WAPE:** total absolute error divided by total actual; see [point accuracy](#step-5--calculate-point-accuracy).

## 16. Sources

- Board of Governors of the Federal Reserve System, [SR 11-7, Guidance on Model Risk Management](https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm), 2011. Banking supervisory guidance; assess applicability separately.
- National Institute of Standards and Technology, [AI Risk Management Framework 1.0](https://doi.org/10.6028/NIST.AI.100-1), 2023. Voluntary AI risk-management guidance.
- International Organization for Standardization, [ISO 31000:2018, Risk management — Guidelines](https://www.iso.org/standard/65694.html), 2018. General risk-management guidance.
- GS1, [Global Trade Item Number](https://www.gs1.org/standards/id-keys/gtin) and [Global Location Number](https://www.gs1.org/standards/id-keys/gln). Identifier definitions, not metric validation rules.
- Hyndman, Rob J., and George Athanasopoulos, [Forecasting: Principles and Practice, 3rd edition](https://otexts.com/fpp3/accuracy.html). Forecast accuracy, scaling, and forecast evaluation.
- Gneiting, Tilmann, and Adrian E. Raftery, [Strictly Proper Scoring Rules, Prediction, and Estimation](https://doi.org/10.1198/016214506000001437), *Journal of the American Statistical Association*, 2007. Proper scoring rules and probabilistic predictions.
- Koenker, Roger, and Gilbert Bassett Jr., [Regression Quantiles](https://doi.org/10.2307/1913643), *Econometrica*, 1978. Foundational quantile-regression loss.
- Makridakis, Spyros, Evangelos Spiliotis, and Vassilios Assimakopoulos, [The M4 Competition: Results, findings, conclusion and way forward](https://doi.org/10.1016/j.ijforecast.2018.06.001), *International Journal of Forecasting*, 2018.
- Makridakis, Spyros, Evangelos Spiliotis, and Vassilios Assimakopoulos, [The M5 accuracy competition: Results, findings, and conclusions](https://doi.org/10.1016/j.ijforecast.2021.11.013), *International Journal of Forecasting*, 2022.

## 17. Links

- [Module 26 index](./_index.md).
- [Model inventory and tiering](./01_model-inventory-tiering.md).
- [Conceptual soundness](./02_conceptual-soundness.md).
- [Module 26 survey](../../../course/26_quality_evaluation_model_risk.md).
- [Forecasting survey](../../../course/07_time_series_forecasting.md).
- [Product economics and leadership survey](../../../course/25_product_economics_leadership.md).
- [Simulation and optimization production labs](../../../course/30_simulation_optimization_production_labs.md).
