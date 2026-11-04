# Global versus Local Forecasting Models

> Module 18 — Deep Learning, Foundation Models, and LLM Agents  
> Topic 03 of 10  
> Estimated study time: 4 hours  
> Prerequisites: basic arithmetic and spreadsheet skills; forecasting and machine-learning concepts are defined here.  
> Version 2026.5

## 1. Why this topic exists

A replenishment manager may own forecasts for tens of thousands of item-location pairs.
Each pair is a **series**: observations of a quantity indexed in time.
The manager must decide how much to order and where to place inventory before future demand is known.
A local forecasting approach fits a separate model to each series.
A global approach estimates some shared model parameters from multiple series.
Neither choice determines the final order automatically.

The distinction matters most when individual histories are short or noisy.
A new store may have only eight weeks of sales, while similar stores have years of records.
A global model can learn a pattern from established stores and apply it to the new store.
That transfer is useful only when the pattern and input definitions are sufficiently comparable.
A comparable product at a different price or in a different climate can have a different response.
The wrong shared pattern can give a confident but systematically wrong forecast.

The opposite problem also matters.
A mature, high-volume series may have a stable local pattern that a pooled model smooths away.
A national model can be accurate in total while underforecasting small, remote locations.
Aggregate improvement does not guarantee that a particular planner, region, or customer benefits.

Local does not mean simple and global does not mean neural.
A separate seasonal-naive rule per series is local.
A pooled linear regression is global.
A neural network trained on windows from many products is also global.
A single-series neural network is local.
The business choice is the scope of information sharing, not a brand name or architecture.

The decision owner should ask four questions before a model project starts.
What action will the forecast inform?
Which series can reasonably share information?
What evidence would justify changing the current workflow?
Who pays for errors, infrastructure, and planner review?

The forecasting team needs to distinguish four business artifacts.
A **forecast** estimates an uncertain future outcome under stated information and assumptions.
A **plan** is a selected set of intended actions, such as a replenishment order.
A **target** is a desired result, such as a service objective.
A **commitment** is a quantity or date accepted by a party, subject to contract and policy.
An accurate forecast is not itself an approved purchase order or supplier commitment.

This topic develops a practical comparison using a common origin, horizon, and information set.
It also develops a decision-oriented business case, including the cost of errors and the cost of operating the model.
A global neural model is a candidate only after simple local and pooled baselines earn their place in the comparison.

## 2. Learning outcomes

After completing this topic, you can:

1. **Classify** local, pooled, and partially pooled approaches by the parameters they share.
   Exercise 1 asks you to distinguish shared fitting from a shared codebase.
2. **Construct** an origin-safe panel and identify which series have usable history.
   Exercise 2 tests availability dates, zero sales, and stockout flags.
3. **Calculate** per-series and volume-weighted forecast errors without confusing their denominators.
   Exercise 3 uses a small holdout panel.
4. **Design** rolling-origin comparisons that separate future-period performance from performance on unseen series.
   Exercise 4 asks for a split and leakage controls.
5. **Diagnose** negative transfer and propose a local or segment-level fallback.
   Exercise 5 examines a high-volume and a small-location trade-off.
6. **Compute** an incremental operating case using comparable cost categories and uncertainty.
   Exercise 6 requires a net-value and break-even calculation.
7. **Critique** a pooling proposal against data rights, drift, service, and governance constraints.
   Exercise 7 asks for a documented go/no-go decision.
8. **Design** a staged deployment and monitoring plan with explicit owners and triggers.
   Exercise 8 requests a rollout and rollback proposal.

Successful completion requires more than naming a model.
You should be able to tell a planner which information was shared, what held-out evidence supports it, and what happens when it fails.

## 3. The core mental model

```text
Observed histories + known-at-origin context + valid identifiers
                |                         |
       eligibility and controls     business decision definition
                |                         |
          +-----+-------------------------+
          |
   choose sharing boundary and training windows
          |
   +------+---------------------+-------------------+
   |                            |                   |
local per-series fit      segment/global fit   partial pooling
   |                            |                   |
   +----------------------------+-------------------+
                |
     identical rolling-origin evaluation
                |
     per-series, segment, and total evidence
                |
     forecast -> policy/simulation -> proposed order
                |
     human/automated authorization -> commitment
                |
     outcomes, costs, monitoring, rollback
```

Think of pooling as a controlled exchange of evidence.
The unit of exchange may be a coefficient, an embedding, a seasonal shape, a distribution, or all network weights.
An **embedding** is a learned vector representing an identifier or category; it is not an observed business attribute.
The exchange occurs during fitting, when examples from one series influence parameters used for another.

A local model restricts fitting information to one series, apart from predefined constants or external features.
The same local algorithm can be deployed on every series without making it global.
The resulting fitted coefficients are still estimated independently.
A pooled model fits at least some common coefficients using multiple series.
A partially pooled model combines common and series- or group-specific parameters.
A hierarchical statistical model is a common partial-pooling design, although a business product hierarchy is a different concept.

A simple mental model is a family of slopes.
One extreme fits a different price coefficient for every item-location.
The other fits one price coefficient to all items and locations.
Partial pooling allows a shared central coefficient with controlled deviations by group or series.
Short, noisy histories tend to rely more on the shared estimate; well-supported histories can retain more local information.
This is a conceptual tendency, not a guarantee for every implementation.

Sharing can reduce estimation variance because more observations inform common parameters.
Variance is sensitivity of fitted results to the particular sample.
Sharing can also increase bias when a shared relation is wrong for an individual series.
Bias here means systematic deviation induced by a modeling assumption, not a protected-class fairness definition.
The appropriate balance depends on data quality, sample size, heterogeneity, and the decision cost of each error.

A neural network can represent nonlinear and interacting patterns using layers of parameterized transformations.
Training adjusts its weights to reduce a loss function on training examples.
A global neural network is global because windows from multiple series update shared weights.
The word neural does not imply that it learns causal relationships or that a forecast is calibrated.
It can memorize identifiers, learn correlations driven by availability, or favor large-scale series.

The forecast horizon is how far beyond an origin a prediction is made.
At an origin, the forecaster must use only information actually available then.
An evaluation that uses a future revised price or a later stockout flag as an input is not a valid simulation of that origin.
A series-held-out test asks a different question from a time-held-out test.
Both can be useful, but neither substitutes for the other.

## 4. Definitions and vocabulary

| Term | Definition and unit | Common misuse |
|---|---|---|
| Series | Time-ordered observations for a defined key, such as item-store-week in units/week. | Treating a renamed item as a new independent demand process without checking continuity. |
| Panel | Collection of series with time and identifying keys. | Assuming a rectangular panel contains observed zeros in every empty cell. |
| Grain | Meaning of one row, including key and time bucket. | Joining daily inventory to weekly sales without defining aggregation. |
| Local model | Parameters are fitted separately per series. | Calling it global because all fits use the same library. |
| Global model | At least some fitted parameters are shared across series. | Assuming every item has identical forecasts or that global always means neural. |
| Partial pooling | Shared parameters and group- or series-specific deviations estimated together or in a coherent two-stage design. | Claiming separate models with a common template are partially pooled. |
| Segment model | A model fitted within an explicitly defined subset. | Choosing segments after inspecting the final test errors. |
| Cross-learning | One series' examples affect fitted parameters used to forecast another. | Using the term for simple aggregation without tracing the fit. |
| Transfer learning | Reuse of knowledge learned in one task or population for another. | Assuming transferred knowledge is appropriate merely because the architecture supports it. |
| Negative transfer | Sharing produces worse outcomes for a defined recipient than a relevant nonsharing comparator. | Declaring it from one noisy forecast without a valid comparison. |
| Training window | A historical input sequence and its target used to fit a model. | Allowing target values after a backtest origin into that origin's training set. |
| Origin | Timestamp when a forecast would have been issued. | Using a reporting date rather than the actual information cutoff. |
| Horizon | Distance from origin to target, often in weeks. | Averaging one-week and 13-week errors without reporting the mix. |
| Lead time | Time between order and receipt, in days or weeks. | Equating forecast horizon with lead time when review periods also matter. |
| As-of snapshot | Values that were available at a specific historical time. | Replacing a historical snapshot with a later corrected value. |
| Target | Quantity to predict, such as observed sales units/week. | Calling censored sales unconstrained demand without adjustment. |
| Feature | Input available to the forecasting procedure at prediction time. | Treating realized future promotions as known future plans. |
| Static feature | Attribute assumed fixed over a forecast window, such as product family at origin. | Ignoring that attributes can be revised or change over a product life. |
| Dynamic feature | Attribute indexed in time, such as planned price by week. | Assuming every future value is known with certainty. |
| Scale | Characteristic magnitude of a series, in its target units. | Applying an unprotected divide-by-zero normalization to all-zero histories. |
| Normalization | Transforming inputs or targets to comparable scale using a stated rule. | Fitting normalization statistics on the entire panel including the test period. |
| Intermittency | Pattern of many zero periods interspersed with positive observations. | Treating every zero as evidence of zero underlying demand. |
| Stockout | Period in which inventory is unavailable for some demand. | Assuming zero sales during a stockout mean zero customer interest. |
| Cold start | Forecasting with little or no target-series history. | Assuming an identifier embedding exists for a genuinely unseen identifier. |
| Embedding | Learned numeric vector for a category or identifier, typically dimensionless. | Assuming a learned identifier vector provides an explanation of causality. |
| Loss | Objective minimized during fitting, such as mean absolute error in units. | Assuming a loss optimized on training windows is the business cost. |
| Weight | Importance assigned to an example or series during fitting or evaluation. | Using the same word for network parameters and economic weights without distinction. |
| MAE | Mean absolute error: average absolute forecast error, in target units. | Averaging MAEs of unequal-length groups without declaring the weights. |
| WAPE | Weighted absolute percentage error: sum absolute errors divided by sum absolute actuals, expressed as a percentage. | Averaging individual percentage errors and calling the result WAPE. |
| Bias metric | Signed forecast-minus-actual total divided by an explicit denominator. | Omitting sign convention or hiding offsetting segment biases. |
| Quantile | Value exceeded with a stated probability under a predictive distribution. | Treating a single 90th percentile as a calibrated interval. |
| Coverage | Fraction of actuals inside a forecast interval, dimensionless. | Interpreting nominal coverage as guaranteed for each subgroup. |
| Rolling origin | Multiple historical forecast issuance times used to evaluate future outcomes. | Refitting once using all observations and pretending each point is an origin. |
| Leakage | Use of information unavailable at the evaluated forecast origin or reserved for model selection. | Assuming a chronological row sort alone eliminates all leakage. |
| Baseline | A reproducible comparator representing a practical alternative. | Comparing a global candidate only against an unrealistic weak rule. |
| Drift | Change in data or predictive relationship over time. | Attributing every one-week error to structural change. |
| Reconciliation | Adjustment to make forecasts consistent across aggregation levels. | Assuming reconciliation repairs biased bottom-level demand estimates. |
| Counterfactual | Outcome under an action not actually taken. | Treating an observed service improvement as caused by a new forecast without a credible comparison. |
| Champion/challenger | Current authorized approach compared with a candidate under common conditions. | Letting a challenger use data that the champion could not have obtained. |
| Decision loss | Cost assigned to an action and realized outcome, in currency. | Claiming generic forecast accuracy is already measured in money. |

## 5. The mechanics, step by step

### Step 1 — Specify the action and decision calendar

**Input:** planning policy, item-location scope, and operating calendar.
**Output:** an agreed forecast use case and a record of decision rights.

1. Name the decision: for example, a weekly store replenishment proposal in units.
2. Record who can approve or reject the proposal and who owns supplier commitments.
3. Define the order cadence, review period, lead time, minimum order quantity, and service objective.
4. Determine the relevant demand horizon from those constraints rather than from a vendor benchmark.
5. Record the cost of ordering too much and too little where defensible estimates exist.
6. Separate an operational target from an estimate of likely demand.
7. Document any capacity limits, shelf-life limits, or allocation priorities that constrain action.
8. Define which decisions will remain manual during the pilot.

For a weekly review and a two-week lead time, a policy might require a distribution over demand during a three-week protection period.
A one-week point forecast alone may not be sufficient for that policy.
If no credible cost or policy can be specified, compare forecasting performance first and avoid assigning invented financial benefits.

### Step 2 — Define series and map identities through time

**Input:** sales transactions, product and location master data, calendars, and inventory records.
**Output:** an auditable series key and lifecycle table.

1. Define the row key as, for example, `(item_id, store_id, week_start)`.
2. Define weeks using a documented time zone and business calendar.
3. Decide whether quantities are sold units, shipped units, orders, or a demand estimate.
4. Convert package sizes and units before comparing histories.
5. Preserve effective dates for store openings, closures, item introductions, and substitutions.
6. Track item replacements and pack changes explicitly instead of silently stitching dissimilar histories.
7. Record which stores were eligible to sell each item in each week.
8. Check that keys are unique at the chosen grain after aggregation.
9. Identify returns, cancellations, and corrections and define their target treatment.
10. Assign a data steward to resolve ambiguous master-data mappings.

An absent transaction is not automatically an observed zero.
If a store was closed, an item was not ranged, or the source feed was missing, the appropriate representation depends on the intended target.
A zero is defensible when the item was eligible, the observation system worked, and zero observed sales is meaningful.

### Step 3 — Separate observed sales from latent demand

**Input:** observed sales, inventory availability, and the decision target.
**Output:** a target definition and treatment of unavailable or censored periods.

Observed sales may be below customer demand when stock is unavailable.
That is censoring: the observation is limited by the selling process.
The unobserved lost quantity cannot be recovered just by relabeling the sales column.
A lost-sales estimate needs additional assumptions, data, or a separate model.

If the decision concerns shipments, stockout-adjusted end-customer demand may not be the right label either.
Discuss the target with the planning owner.
Possible treatments include excluding severely unavailable periods from fitting, providing an availability feature, or estimating lost demand separately.
Each changes the estimand and may introduce selection effects.
Use the same target definition for all compared approaches.

Availability itself may be known only after the week ends.
It can be used to label or filter historical training observations if it is available by the training cutoff.
It must not be supplied as a known future input unless an actually available forecast or scenario is used.
For evaluation, report both measured sales error and any supported demand-adjusted analysis separately.

### Step 4 — Set eligibility and history thresholds

**Input:** lifecycle table, usable observations, and current production scope.
**Output:** a cohort inventory and fallback policy.

For each series, count eligible observed periods, nonzero periods, gaps, and periods with availability problems.
Report the age distribution rather than only its mean.
A model trained on 100,000 rows may still have only two useful observations for a particular new product.
Correlated sliding windows are not 100,000 independent series or independent demand shocks.

Separate at least these cohorts:

- Mature, regularly selling series with enough local history.
- Short-history series with a few eligible observations.
- Truly new series with no own-target observations.
- Intermittent series with many observed zeros.
- Discontinued or temporarily inactive series.
- Series with unresolved source or unit problems.

Choose thresholds before final evaluation.
A threshold might require a minimum number of eligible weeks for a seasonal baseline, but the right number depends on the seasonal cycle and operating environment.
Short history does not mean global will work: the shared population must also be relevant.
A series with invalid units should be repaired or excluded, not pooled harder.

### Step 5 — Build the candidate population and sharing boundary

**Input:** eligible panel, business taxonomy, and data-use permissions.
**Output:** one or more explicit pooling hypotheses.

A pooling hypothesis says what is shared, among whom, and why.
For example: weekly demand ratios for comparable ambient grocery items may share a seasonal response within a climate region.
The hypothesis should describe expected exceptions, not just a large row count.

Compare plausible boundaries:

1. One independent local model per series.
2. A global model across all permitted series, with common time and item features.
3. Separate global models by a stable segment such as temperature-sensitive versus ambient goods.
4. A partial-pooling model with shared effects and local deviations.
5. A rules-based fallback for unsupported or exceptional cases.

A highly granular segment can lose the sample-size benefit of pooling.
An overly broad segment can mix incompatible price, lifecycle, or calendar effects.
Segment membership itself must be known at the origin.
Avoid segmenting with future sales outcomes, such as labeling a product a future bestseller.

Data-sharing boundaries may follow contracts or geography rather than statistical similarity.
A single model trained on multiple customers' data may reveal sensitive patterns or violate agreed use restrictions even if raw rows are not exposed.
Security and legal owners must approve the intended use; a modeling team should not infer rights from technical access.

### Step 6 — Construct examples without peeking

**Input:** timestamped source snapshots, candidate features, training cutoffs, and horizon.
**Output:** reproducible training examples indexed by historical origin.

For a four-week forecast, a training example might use the previous 12 eligible weeks to predict the next four.
A sliding window generates many examples from a series.
Each example must preserve its own origin and information set.
If labels extend beyond a backtest origin, the example cannot be used to fit that origin's model.

A compact leakage checklist:

- Use the price plan version available at the origin, not realized future price.
- Use promotion plans as issued; archive cancellations and revisions separately.
- Fit imputers, encoders, scalers, and feature selection only on permitted training data.
- Keep target-derived rolling averages strictly behind the origin.
- Calculate group aggregates without including a target period that would not yet exist.
- Prevent late-arriving corrections from rewriting historical feature snapshots silently.
- Avoid encoding a category using target means calculated from the validation fold.
- Record publication timestamps for weather forecasts, calendars, and economic releases.
- Ensure the newest training label is fully observed at the fitting cutoff.
- If labels overlap in time, assess whether a gap is needed to prevent boundary contamination.

Known-future covariates are known values or plans at origin, not values observed later.
A holiday calendar can usually be known ahead, but the business effect of a holiday is not known with certainty.
A promotion schedule may be known as a plan and later change.
For operational deployment, forecasts should use the plan actually available at issuance and document subsequent changes.

### Step 7 — Choose scaling and loss deliberately

**Input:** training targets, decision objectives, and candidate model.
**Output:** fitted training transformations, weighting rules, and a stated loss.

Without scaling, a high-volume item can dominate a pooled squared-error objective because large absolute errors contribute quadratically.
This may be appropriate if volume is the business priority, but it should not be accidental.
Scaling each series by a training-only mean can make low- and high-volume patterns comparable.
However, small or zero means need a protected denominator and a documented fallback.

Possible training objectives include absolute error, squared error, or a probabilistic likelihood.
Absolute error targets a conditional median under suitable conditions.
Squared error targets a conditional mean under suitable conditions.
Quantile loss targets a selected conditional quantile under suitable conditions.
The best training loss depends on the decision and the assumptions; none automatically produces the best inventory outcome.

A useful example weight can represent a deliberate mixture of equal series, volume, or economic importance.
Write the weight formula before fitting.
A volume weight computed with future test volume leaks information if used at historical training time.
Report unweighted and weighted performance so that the trade-off is visible.
A weighting scheme can improve total error while harming the tail of small stores.

If a neural model uses a learned series identifier, define what happens to an unseen identifier.
Options include an unknown-category representation, observable item attributes, or a non-neural fallback.
An identifier embedding trained on established items cannot magically represent a new identifier without a defined mechanism.
Also check whether a new product's attributes are genuinely available before launch.

### Step 8 — Fit baselines and candidates on equal information

**Input:** origin-safe examples and approved model configurations.
**Output:** forecasts for every eligible candidate-origin-series-horizon combination.

At minimum, consider a last-observation or seasonal-naive baseline, the incumbent local method, and a simple pooled baseline.
A seasonal-naive forecast uses the corresponding observation from the previous seasonal cycle when that observation exists.
A last-observation forecast repeats the most recent eligible value.
Neither is guaranteed to be adequate for a promotion or launch.

Include a local model that can actually be run with available history.
If a local seasonal model cannot forecast a new series, define its fallback before scoring.
Compare the operational system, including its fallback, not only the subset where the model succeeds.
If a global model fails on an unseen category, its operational fallback belongs in the same comparison.

Tune model hyperparameters using training and validation origins only.
Hyperparameters are configuration choices such as model capacity, regularization, and learning rate.
Refit transformations and model weights at each simulated retraining cutoff according to the deployment schedule.
Do not select the best model by repeatedly examining the final test period.
Record random seeds and configuration, while recognizing that some hardware and software operations can still be nondeterministic.

### Step 9 — Evaluate across time and across series

**Input:** held-out forecasts, actuals, decision weights, and cohort labels.
**Output:** a comparison table with uncertainty and operational coverage.

Rolling-origin validation chooses a series of historical issue dates.
At each origin, fit using information available then and score subsequent observations.
The same horizon and eligible population should be used for each candidate.
A training window ending before the origin is not sufficient if its labels or preprocessing contain future information.

For future-time performance on existing series, allow only pre-origin history for those series.
For generalization to new series, hold out entire item or location identifiers during training.
Then supply only the features and history that would be available for the intended cold-start situation.
A new store with several weeks of sales differs from a launch with no sales at all.
A random row split can place adjacent weeks or the same item in both sets and exaggerate performance.

Use several origins that span relevant seasonal and disruption conditions when the data permit.
Keep a final untouched time period for the decision after development.
For dependent errors, use origin-level or appropriately clustered resampling rather than treating every window as independent.
A small number of origins limits uncertainty estimates; say so explicitly.

Report:

- Count of origins, series, forecast rows, and failed or fallback forecasts.
- Error by horizon and by origin.
- Aggregate WAPE with its denominator and absolute-error numerator.
- Per-series MAE distribution and number of winners and losers.
- Signed bias overall and within business-critical cohorts.
- Results for mature, short-history, intermittent, high-volume, and new series.
- Interval coverage and sharpness if probabilistic forecasts are provided.
- Inventory or service simulation results only with a documented policy and assumptions.

For WAPE, sum absolute errors over the specified scope and divide by the sum of absolute actuals over that same scope.
If the actual denominator is zero, WAPE is undefined for that scope.
Report absolute error and volume rather than silently replacing the denominator.
Do not calculate a percentage change relative to zero without a stated convention.

### Evaluation detail — make the comparison auditable

A single overall score is not a sufficient model-selection record.
Create a row for every intended forecast, whether a model produces its primary forecast or a fallback.
The row should retain the origin, target period, series identifier, model version, forecast, actual, and eligibility state.
Use the same series-origin-horizon keys for paired comparisons.
If one approach cannot produce a primary forecast, its fallback remains part of its result.

The following example evaluation ledger makes each decision explicit.

| Question | Decision to record | Why it changes interpretation |
|---|---|---|
| What is the population? | All eligible ranged item-store pairs at origin. | Restricting to mature products excludes the cold-start use case. |
| What is the target? | Observed weekly sales in cases, with stockout flags. | The result does not directly measure unconstrained demand. |
| What is the origin? | Monday 08:00 local planning time. | Information arriving later cannot be a feature. |
| What is the horizon? | Weeks one through four, scored separately. | Short-horizon gains may not extend to longer lead times. |
| How is training refreshed? | Refit once per month using data available at the cutoff. | Daily refitting in the test would give the candidate an operational advantage. |
| How are parameters selected? | Choose on earlier validation origins and freeze for the final period. | Final-test selection overstates expected performance. |
| How are late corrections handled? | Preserve origin snapshots and retain a separate final outcome version. | Rewritten inputs can leak future corrections. |
| What is scored after a failure? | The published fallback forecast, with a failure flag. | Dropping failures selects easy cases. |
| How are groups compared? | Predefine store size, lifecycle, and intermittent-demand cohorts. | Post-hoc group selection can exaggerate exceptional results. |
| How is uncertainty reported? | Compare paired origin-level differences and disclose the number of origins. | Millions of correlated rows do not imply millions of independent experiments. |

There are two separate timing questions.
First, was each input available when the forecast was issued?
Second, was the corresponding target fully observed when the model was fitted?
A weekly target can straddle a cutoff even if the input window ends before it.
For instance, a four-week label beginning two weeks before the origin includes two future weeks.
That training example must be excluded at the origin.

Forecast issue times and retraining times may differ.
If the production model is refitted on the first Monday of each month, a backtest should not refit its weights every Monday unless the proposed operating model and cost case include that change.
Features may still update weekly when those features would be available in production.
Distinguish refitting parameters from refreshing model inputs.
The local comparator should receive the same reasonable data access and retraining opportunity.

For each origin, calculate a paired difference between the candidate and comparator.
For absolute error, one possible sign convention is candidate minus comparator.
A negative difference then favors the candidate.
Keep this convention consistent in tables and explain it next to the results.
Pairing reduces confusion from comparing different populations, but it does not eliminate serial correlation or selection bias.

Suppose four origins produce candidate-minus-local total absolute errors of -80, -40, +30, and -10 cases.
The sum is -100 cases across those four origins.
Three origins favor the candidate and one favors the local comparator.
The average difference is -25 cases per origin, but four origins are too few for a precise general claim.
The case counts at each origin and the corresponding actual volume are also needed to interpret materiality.
If one origin contains a unique holiday, describe that fact rather than assuming the same gain every month.

Intervals around a difference should reflect the dependence structure.
Adjacent item-store windows share both time and product shocks.
Resampling individual rows as though they were independent may produce an unrealistically narrow interval.
Depending on the question and available data, resampling whole origins or suitable time blocks can be more defensible.
If locations share weather or supply disruptions, a design may also need to account for shared geographic shocks.
The number and diversity of independent blocks limit how precise any interval can be.

Cold-start validation requires extra care.
Holding out a product identifier prevents that product's observed targets from updating global weights.
However, another identifier may represent the same product after a packaging change.
If the intended question is genuinely new products, group linked replacements together when feasible.
Likewise, a store-held-out test measures new-location generalization, not necessarily new-product generalization.
An item-store pair absent from training may still involve a familiar item and a familiar store.
State which of these populations the experiment represents.

Attributes can also leak future knowledge into a cold-start test.
A final product category or success-tier label might be assigned only after launch.
A launch-time model cannot use that future classification.
Use the category version and product description available when the forecast would have been issued.
If those attributes are incomplete for some launches, preserve that missingness in the evaluation.
The apparent cold-start benefit may disappear once real launch-time information is enforced.

### Evaluation detail — distributions and operational consequences

A point forecast is one summary of uncertain demand.
Inventory policies often need a distribution, an interval, or a selected quantile over a protection period.
For example, a 90th-percentile forecast is a value that should be at or above realized demand about 90% of the time if the predictive distribution is calibrated for the relevant population.
That statement concerns repeated comparable cases; it is not a promise for an individual week.

To evaluate a central 80% interval, count how often actual demand falls between its lower and upper bounds.
Report the count of eligible observations, empirical coverage, and mean interval width in cases.
Coverage alone can be made high by issuing an extremely wide interval.
Width alone can be made low by issuing an interval that misses too often.
Inspect both quantities by horizon and business cohort.
Where there are few observations, disclose that a cohort coverage estimate is unstable.

A model can produce apparently good average coverage while under-covering during promotions.
The operational harm may concentrate exactly when the replenishment decision is most consequential.
Use promotion status as known at origin for a prospective cohort and record realized promotion execution separately.
Do not retrospectively redefine the model's inputs using the realized event.
Consider separately reporting missed demand above the upper bound and excess below the lower bound.
Those counts reveal which side of the distribution is poorly represented.

Multi-week decisions require more than adding independent one-week quantiles.
Demand errors across weeks can be correlated.
In general, the sum of weekly 90th percentiles is not the 90th percentile of cumulative demand.
Use a model or simulation that represents the joint uncertainty relevant to the protection period.
If only marginal forecasts are available, state the approximation and test its sensitivity.
An aggregate demand distribution also does not capture every capacity and supplier constraint.

To translate a predictive distribution into an order, specify the policy's cost and feasibility assumptions.
A simple newsvendor-style calculation uses underage and overage costs to determine a critical fractile.
For underage cost USD 8 per case and overage cost USD 2 per case, the ratio is `8 / (8 + 2) = 0.80`.
Under the simple one-period model's assumptions, the corresponding decision uses the 80th demand percentile.
This is not a universal service target and does not address fixed order costs, lead-time uncertainty, or perishability.
In practice, include on-hand and pipeline inventory, pack constraints, review frequency, and supplier restrictions.

An operational replay should run each candidate through the same policy wherever possible.
Freeze the policy and cost assumptions before comparing models.
Record the actual stock, inbound orders, minimum quantities, and stockout assumptions at each decision.
If the historical order influenced later sales availability, replacing that order in a replay changes a counterfactual.
A replay based only on observed sales cannot directly reveal the lost demand that would have occurred under another stock position.
Bound that uncertainty with scenarios or stronger evidence rather than asserting an exact saving.

Finally, distinguish three levels of evidence.
A forecast backtest estimates predictive performance under its data and split assumptions.
A policy replay estimates modeled consequences under its policy and counterfactual assumptions.
A controlled operational pilot measures outcomes under a real intervention but still needs an appropriate comparison and consideration of spillovers.
Each level answers a different question.
Document which level supports every financial and service claim.

### Step 10 — Convert evidence to an operating decision

**Input:** held-out metrics, planner workflow, costs, and operational constraints.
**Output:** a go/no-go decision and a bounded pilot.

Forecast error is not money by itself.
Estimate incremental value through a credible policy replay, a controlled rollout, or an explicitly labeled scenario.
A stockout cost estimate might include contribution margin and contractual penalties, but should avoid counting the same lost margin twice.
An inventory reduction has a carrying-cost benefit, not an automatic benefit equal to the entire inventory value every year.

A fair comparison includes development amortization where relevant, ongoing compute, data operations, model monitoring, vendor charges, integration, and planner review.
Include the cost of failed forecasts and local fallbacks.
Use a common period and currency for benefits and costs.
Estimate volume and service impacts separately before combining them.
Avoid claiming causality from a before/after comparison that also contains promotion, supply, or policy changes.

A pilot should have an owner, duration, cohorts, guardrails, success criteria, and a rollback path.
When feasible, randomize eligible locations or time blocks while preventing spillovers.
If randomization is infeasible, state the comparison design and residual confounding.
The sponsor should approve a decision threshold before inspecting pilot outcomes.

### Step 11 — Deploy, monitor, and retire

**Input:** approved pilot decision, serving contract, and support roster.
**Output:** an auditable production process and rollback mechanism.

Publish forecasts with issue timestamp, target period, series key, units, model version, input snapshot version, and fallback status.
Make missing and stale inputs visible.
A plan or order generated from a forecast must be separately versioned and authorized.

Monitor data freshness, identity changes, missing values, zero frequency, and out-of-range features.
Monitor forecast error and signed bias after outcomes arrive, by horizon and cohort.
Measure distributional calibration for probabilistic outputs and track service outcomes separately.
Record retraining events and verify that training data remained within authorized boundaries.

Define thresholds for review, not just dashboards.
For example, a persistent regional bias exceeding a pre-agreed limit may trigger investigation and a fallback.
A data outage may require immediate reversion without waiting for realized forecast error.
The fallback must be executable with currently available data, and its owner must be reachable.
Retire obsolete models and access credentials under the organization's retention and security policy.

## 6. Worked example

### Business setting and assumptions

A fictional grocery distributor replenishes three store-item series every week.
The decision is a proposed order, not an automatic supplier commitment.
The comparison concerns a one-week forecast made at the same origin for each series.
All quantities below are cases per week; each case has the same pack size.
The numbers are illustrative and are not reported results from a named company.

The stores have the following latest observed eligible sales and next-week actuals:

| Series | Latest observed sales, cases | Next-week actual sales, cases | Local forecast, cases | Global forecast, cases |
|---|---:|---:|---:|---:|
| A | 100 | 130 | 100 | 120 |
| B | 20 | 18 | 20 | 24 |
| C | 10 | 11 | 10 | 12 |
| Total | 130 | 159 | 130 | 156 |

The local rule repeats the latest observed sales.
The global rule multiplies the latest observed sales by a shared factor of 1.20.
Suppose that factor was fixed using other eligible training histories available before the origin.
It was not estimated from the three next-week actuals shown above.
The global rule is deliberately simple: it makes cross-learning visible without requiring a neural network.
It is not a claim that a constant multiplier is generally a good forecasting model.

For each series, error means forecast minus actual.
Absolute error removes the sign but not the unit.

| Series | Local signed error, cases | Local absolute error, cases | Global signed error, cases | Global absolute error, cases |
|---|---:|---:|---:|---:|
| A | -30 | 30 | -10 | 10 |
| B | +2 | 2 | +6 | 6 |
| C | -1 | 1 | +1 | 1 |
| Total signed / sum absolute | -29 | 33 | -3 | 17 |

Local WAPE is `33 / 159 = 20.75%`, rounded to two decimal places.
Global WAPE is `17 / 159 = 10.69%`.
The absolute improvement is `20.75% - 10.69% = 10.06 percentage points` after rounding.
The relative reduction in absolute error is `(33 - 17) / 33 = 48.48%`.
The two percentages answer different questions and must not be interchanged.

The equal-series mean absolute error for this one origin is `33 / 3 = 11.00 cases` locally and `17 / 3 = 5.67 cases` globally.
That mean is in cases, not percent.
Series A improves by 20 cases, series B worsens by 4 cases, and series C is unchanged at 1 case of absolute error.
The global result improves the aggregate while harming one of the three series.
In particular, the common 20% uplift is inappropriate for B in this realized week.
One week cannot establish that B systematically suffers negative transfer.
Repeated paired origins and a relevant local comparator would be needed to support that conclusion.

The signed aggregate bias is `-29 / 159 = -18.24%` for the local rule and `-3 / 159 = -1.89%` for the global rule, using forecast minus actual.
These totals conceal B's overforecast.
If A's underforecast and B's overforecast occur in different stores, inventory at B may not satisfy demand at A.
A low network-wide bias does not imply locally balanced inventory.

### Forecast versus order

Assume, only for this illustration, that the inventory policy orders the rounded one-week forecast with no stock on hand, no pipeline inventory, and no constraints.
Then the proposed global orders are 120, 24, and 12 cases.
The realized shortages are 10, 0, and 0 cases.
The realized excesses are 0, 6, and 1 cases.
The local proposed orders imply shortages of 30, 0, and 1 cases and excess of 0, 2, and 0 cases.

These are simplified one-period outcomes, not a complete replenishment simulation.
In a real operation, on-hand stock, lead time, review period, pack sizes, spoilage, substitution, and capacity change the order.
The forecast is an estimate, the order is a plan, and acceptance by a supplier can create a commitment.
A 98% fill-rate target is neither the forecast nor the observed fill rate.

### Illustrative decision loss

Suppose each shortage case costs USD 8 in contribution and service impact, and each excess case costs USD 2 over the evaluation period.
Assume those costs are incremental and do not double-count a separate inventory charge.
Local loss is `(31 × USD 8) + (2 × USD 2) = USD 252`.
Global loss is `(10 × USD 8) + (7 × USD 2) = USD 94`.
The illustrative difference is `USD 158` for this one origin and these three series.
If shortage and excess costs differ, or if inventory can be transferred, the ranking and value can change.
The example does not prove annual savings or causality.

### Illustrative annual business case

Suppose a separate, representative policy replay estimates annual gross operating benefit of USD 420,000.
Treat that number as an assumption for this exercise, not as an extrapolation from the three-series example.
Suppose annual recurring global-model costs are USD 90,000 compute and licensing, USD 70,000 data operations and monitoring, and USD 40,000 planner review.
The incumbent costs USD 80,000 per year to operate and remains the comparator.
Incremental recurring cost is `(90,000 + 70,000 + 40,000) - 80,000 = USD 120,000 per year`.
Annual net recurring benefit is `420,000 - 120,000 = USD 300,000`.

Assume one-time integration and validation cost of USD 360,000 at launch.
Ignoring discounting, taxes, and benefit ramp, simple payback is `360,000 / 300,000 = 1.2 years`.
The annual gross benefit required to cover recurring incremental cost alone is USD 120,000.
To recover the one-time cost within two years on a simple undiscounted basis, required annual gross benefit is `120,000 + 360,000 / 2 = USD 300,000`.
These break-even thresholds are not a net present value calculation.

If annual gross benefit falls to USD 180,000, annual net recurring benefit becomes USD 60,000 and simple payback becomes six years.
If the global system must run alongside the full incumbent rather than replace it, the USD 80,000 incumbent cost is not avoided.
Then incremental recurring cost is USD 200,000 and the base-case annual net recurring benefit is USD 220,000.
The accounting boundary and coexistence period therefore materially affect the decision.

A prudent sponsor would request confidence intervals or scenario ranges around policy benefit, segment guardrails, and a pilot before authorizing the full rollout.
The example supports a methodology, not a forecast of any actual company's return.

## 7. Code, query, or calculation

The following Python 3 example uses only the standard library.
It constructs a small panel, estimates a shared multiplicative factor from training series, and compares it with local persistence on held-out future observations.
It prints overall and per-series errors, a signed-bias metric, and a simple decision-cost calculation.
The global factor is fitted using only the training pairs; test actuals are not used to estimate it.

This example demonstrates information sharing, not a neural architecture or a production backtest.
The small deterministic data set is deliberately transparent.
The three holdout series and outcomes match the worked example; the shared factor equals 1.20 by construction.
Run the code with Python 3; no network access or package installation is required.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Pair:
    series: str
    previous: float
    actual: float


# Historical pairs from other series, available before the test origin.
# Fit a single through-origin least-squares coefficient across the panel.
train = [
    Pair("T1", 10.0, 12.0),
    Pair("T2", 20.0, 24.0),
    Pair("T3", 30.0, 36.0),
]

# Next-week observations are used for scoring, never for fitting.
test = [
    Pair("A", 100.0, 130.0),
    Pair("B", 20.0, 18.0),
    Pair("C", 10.0, 11.0),
]


def fit_shared_factor(rows):
    denominator = sum(row.previous ** 2 for row in rows)
    if denominator == 0:
        raise ValueError("Cannot fit a factor from all-zero previous values")
    numerator = sum(row.previous * row.actual for row in rows)
    return numerator / denominator


def score(rows, factor, shortage_cost=8.0, excess_cost=2.0):
    if not rows:
        raise ValueError("At least one scored row is required")
    forecasts = [factor * row.previous for row in rows]
    errors = [forecast - row.actual for row, forecast in zip(rows, forecasts)]
    actual_total = sum(abs(row.actual) for row in rows)
    abs_error = sum(abs(error) for error in errors)
    bias = sum(errors) / actual_total if actual_total else None
    wape = abs_error / actual_total if actual_total else None
    loss = sum(
        shortage_cost * max(-error, 0.0) + excess_cost * max(error, 0.0)
        for error in errors
    )
    return forecasts, errors, abs_error, wape, bias, loss


shared_factor = fit_shared_factor(train)
print(f"shared factor: {shared_factor:.2f}")
for name, factor in (("local", 1.0), ("global", shared_factor)):
    predictions, errors, absolute, wape, bias, loss = score(test, factor)
    print(
        f"{name}: absolute error={absolute:.2f} cases, "
        f"WAPE={100 * wape:.2f}%, bias={100 * bias:.2f}%, "
        f"loss=USD {loss:.2f}"
    )
    for row, prediction, error in zip(test, predictions, errors):
        print(
            f"  {row.series}: forecast={prediction:.2f} cases, "
            f"actual={row.actual:.2f} cases, error={error:+.2f} cases"
        )
```

Expected output:

```text
shared factor: 1.20
local: absolute error=33.00 cases, WAPE=20.75%, bias=-18.24%, loss=USD 252.00
  A: forecast=100.00 cases, actual=130.00 cases, error=-30.00 cases
  B: forecast=20.00 cases, actual=18.00 cases, error=+2.00 cases
  C: forecast=10.00 cases, actual=11.00 cases, error=-1.00 cases
global: absolute error=17.00 cases, WAPE=10.69%, bias=-1.89%, loss=USD 94.00
  A: forecast=120.00 cases, actual=130.00 cases, error=-10.00 cases
  B: forecast=24.00 cases, actual=18.00 cases, error=+6.00 cases
  C: forecast=12.00 cases, actual=11.00 cases, error=+1.00 cases
```

The factor is the through-origin least-squares estimate `sum(previous × actual) / sum(previous²)`.
It shares one slope across training series and applies that slope to the test series.
The local rule is persistence, expressed by factor 1.00; it does not fit a slope from other series.
This code assumes nonnegative case quantities and complete, comparable observations.
The scoring function explicitly treats WAPE and bias as undefined when total actual volume is zero.
The printed example has a nonzero denominator.

For a realistic evaluation, replace the fixed test list with multiple archived origins and fit separately at each simulated retraining cutoff.
Add availability and lifecycle eligibility, preserve as-of features, compare operational fallbacks, and report uncertainty.
Do not use this example's invented shortage and excess costs as estimates for a real business.
A production implementation also needs monitoring, versioned inputs, access controls, and documented exception handling.

## 8. Data, standards, and regulatory context

### Data contract

A panel should have a documented schema, ownership, update schedule, and quality controls.
A practical minimum includes the following fields.

| Field | Meaning | Control |
|---|---|---|
| Item identifier | Stable product key at the intended grain. | Check effective dates and pack changes. |
| Location identifier | Store, warehouse, or customer key. | Check closures, relocations, and authorized access. |
| Bucket start and end | Time interval and time zone. | Validate calendar alignment and daylight-saving treatment where relevant. |
| Quantity and unit | Observed sales, demand estimate, or shipment. | Reject mixed units and document conversions. |
| Eligibility | Whether the item could be sold or supplied. | Distinguish not ranged from observed zero. |
| Availability | Inventory or on-shelf state and observation quality. | Track missingness and censoring. |
| Source and publication timestamp | When a value was available to the forecaster. | Preserve historical versions for backtesting. |
| Price and promotion plan | Versioned intended commercial context. | Prevent realized future values entering historical inputs. |
| Product and store attributes | Categories, geography, and lifecycle data. | Preserve effective dates and unknown values. |
| Forecast version | Model, issue time, horizon, and fallback flag. | Make outputs reproducible and auditable. |

Cross-learning depends on comparability across these fields.
A shared model cannot repair a row that combines unit sales from one source and case shipments from another without a valid conversion.
It may instead learn a persistent source-specific distortion.
Coverage should be measured both by number of series and by business volume.

### Product identification and messaging

GS1 maintains standards for identifying trade items and locations, including Global Trade Item Numbers (GTINs) and Global Location Numbers (GLNs).
These identifiers can support joins across trading partners when the partners implement and maintain them correctly.
A GTIN does not by itself define the forecasting grain, the historical equivalence of package sizes, or permission to pool partner data.
An internal product key may be necessary alongside the external identifier and its effective dates.

Electronic Data Interchange (EDI) messages can communicate transactions such as orders or invoices using agreed message standards.
A received order is not necessarily end-consumer demand, and a standardized message is not automatically accurate or timely.
Map each source document to a business event, timestamp, and unit before using it in a panel.
The applicable message standard and contractual obligations depend on trading partners and jurisdictions.

### Privacy, confidentiality, and access

A product-store aggregate may still be commercially sensitive.
Customer, employee, or supplier records may contain personal data depending on the source and aggregation.
Restrict access to the minimum required and review contracts before sharing data across customers, geographies, or legal entities.
Model artifacts can encode information about training data; withholding raw data alone does not establish that a model is safe to share.

Where the EU General Data Protection Regulation (GDPR) applies, its principles include purpose limitation, data minimization, and appropriate security; its applicability depends on the processing and parties.
Consult the regulation and qualified counsel rather than treating this educational example as legal advice.
The GDPR does not prescribe a particular local or global forecasting architecture.
Forecasting teams should document the lawful basis and retention and access decisions with the responsible privacy function where applicable.

### Model and record governance

The NIST AI Risk Management Framework is voluntary guidance, not a universal legal mandate.
Its govern, map, measure, and manage functions are useful for allocating accountability and documenting risks.
They do not certify that a model is accurate, fair, or legally compliant.
Preserve model versions, training cutoffs, relevant data lineage, evaluations, approvals, and incident decisions according to organizational policy.

Forecast records should distinguish a published forecast from a planner override, a replenishment plan, and a supplier commitment.
If overrides are used for evaluation, record when and why they occurred.
A model can be technically accurate while a downstream override or constraint changes the eventual order.
That distinction is required to diagnose where value or harm arose.

### Practical data readiness checks

Before pooling, inspect at least the following questions.

1. Are item and location identities stable through the evaluation horizon?
2. Are product replacements and pack-size changes mapped with effective dates?
3. Is each row unique at the declared grain?
4. Are all volumes expressed in the same unit within a comparison?
5. Are missing observations distinguishable from eligible zero sales?
6. Can availability-limited sales be identified with an understood quality level?
7. Are promotions and prices stored as historical as-of plans?
8. Are data revisions timestamped rather than overwriting history?
9. Are store opening, closure, and ranging dates available?
10. Is there enough historical time to evaluate seasonal and disruption conditions?
11. Are source-specific breaks and extraction delays visible?
12. Are cross-customer training and model-sharing rights documented?
13. Is the planned future feature actually available at forecast issuance?
14. Are sensitive attributes and access restrictions handled appropriately?
15. Can the forecast and the order be traced to distinct versions and approvals?

A failed check does not always rule out a model.
It does require a documented mitigation and a bounded claim about what the evaluation measures.

## 9. Industry practice and case evidence

### Published multi-series evidence

Salinas and colleagues' DeepAR paper describes an autoregressive recurrent neural-network approach trained across related time series to produce probabilistic forecasts.
The paper evaluates benchmark data sets and illustrates the idea of learning shared patterns across series.
It is evidence that such a modeling approach can be implemented and evaluated, not evidence that it will improve a particular retailer's inventory economics.
The paper's data, horizons, training choices, and metrics must be checked before transferring its conclusions.

The M5 forecasting competition used Walmart unit-sales data at multiple aggregation levels and challenged participants to forecast retail demand.
The competition literature reports that leading approaches used many related series and explanatory variables.
The public competition setting is not the same as an enterprise's live replenishment workflow.
Public leaderboard performance does not establish causal inventory savings, the availability of the same features at a live origin, or performance under later structural changes.
A competition score also reflects its specified metric, hierarchy, and evaluation periods.

Montero-Manso and Hyndman's work discusses local and global forecasting models and theoretical considerations around their expressive capability.
The important operational lesson is that a shared model is not automatically more restrictive in every possible hypothesis class.
In a finite-data implementation, however, representation, optimization, and regularization still determine what can actually be learned.
Theory does not waive the need for an origin-safe business comparison.

### Named commercial context

Amazon's published DeepAR work includes authors affiliated with Amazon and evaluates the method on several data sets.
That is a research publication, not a verified report of a specific retailer's savings from replacing local models.
The method's use of shared parameters motivates a candidate architecture; it does not settle the business case for another organization.

Walmart's M5 data provide a concrete retail forecasting context with items, stores, calendar variables, prices, and grouped evaluation.
The source competition documents the forecasting task and data set.
It does not establish that a particular competition submission was deployed into Walmart's replenishment system.
[INFERENCE] A practitioner using similar data would need to reproduce the relevant price and promotion information as it was available at each historical origin before relying on a backtest.

### Operational patterns

In practice a portfolio can contain several approaches rather than a single winner.
Mature high-volume products can retain a strong local baseline.
A global model can serve short-history products when appropriate comparison series and features exist.
An intermittent-demand rule can remain the fallback where long zero stretches make standard scaling unstable.
A planner may override a forecast for a known event, but the override needs a reason and later evaluation.

Segment and fallback decisions must be evaluated as part of the complete system.
Selecting the best method for each series after seeing its test error is an oracle comparison, not a deployable policy.
Choose routing rules using prior data or validation evidence, then freeze them for the final test.
A routing policy also creates monitoring obligations when product categories or data quality change.

Evidence should be matched to the claim:

| Claim | Suitable evidence | Evidence that is insufficient alone |
|---|---|---|
| Shared training is technically feasible. | Reproducible implementation and documented data contract. | A vendor slide without implementation detail. |
| Accuracy improves in this portfolio. | Paired, origin-safe out-of-time test with cohort results. | A single aggregate score on a public data set. |
| New products benefit. | Held-out identifiers with launch-time information only. | Random splits of rows from already seen products. |
| Inventory cost falls. | Controlled rollout or credible policy replay with costs and constraints. | Lower WAPE alone. |
| Service is protected. | Service outcomes and cohort-level guardrails. | Low average signed bias across the network. |
| Data use is permitted. | Contract and privacy review for the actual data flow. | Technical ability to access the data. |

A named case is useful because it anchors a method in a real setting.
It is not a substitute for describing the decision, evidence quality, and transfer limits in the reader's own setting.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Future-price leakage | Historical validation improves sharply when realized prices are included. | Post-origin information enters the feature set. | Inflated performance and poor live forecasts. | Use archived as-of plans and rerun the comparison. |
| Mixed units | One source has a discontinuous level after integration. | Cases and individual units are combined. | Misleading shared scale and biased forecasts. | Validate conversions and master-data effective dates. |
| Censored sales treated as demand | Low sales coincide with low availability. | Stockouts suppress observed sales. | Replenishment can reinforce shortages. | Flag availability and evaluate a defensible demand treatment. |
| Dominance by large series | Aggregate metric improves while small stores deteriorate. | Training loss and WAPE emphasize volume. | Local service harms are hidden. | Report per-series and cohort results and set guardrails. |
| Negative transfer | A subgroup repeatedly loses to its local baseline. | Shared relation does not fit that subgroup. | Systematic subgroup error. | Test segment models, partial pooling, or a local fallback. |
| Unseen identifier | A launch maps to an unknown category at serving. | Its learned embedding was never trained. | Failed or generic predictions. | Use observed attributes and an explicit unknown/fallback policy. |
| All-zero scaling | Infinite or extremely large normalized values appear. | The series scale is zero or nearly zero. | Unstable training and forecasts. | Use a documented protected scale and intermittent-demand fallback. |
| Random row split | Validation looks much stronger than a later time test. | Adjacent windows and identities appear on both sides. | Overstated generalization. | Split by time and, for cold start, by series. |
| Label overlap | A training target occurs after a simulated issue date. | Window construction crosses the cutoff. | Future outcomes influence the fit. | Enforce label-end cutoffs and consider an embargo. |
| Stale shared model | Bias grows after a range or promotion change. | Shared parameters no longer describe current behavior. | Repeated operational error. | Monitor drift and use scheduled or triggered retraining. |
| Hidden fallback | Accuracy reports omit failed forecasts. | Only successful candidate rows are scored. | Unrealistic system-level results. | Score the complete routing and fallback policy. |
| Aggregate cancellation | Total signed error is near zero but locations disagree. | Positive and negative errors offset. | Inventory is misplaced. | Report local signed bias and transfer constraints. |
| Oracle routing | Reported hybrid wins every test series. | Best method was selected after seeing test outcomes. | Unattainable performance. | Freeze routing on training/validation data. |
| Short evaluation | One seasonal period drives the result. | Limited origins do not represent future conditions. | Fragile rollout decision. | Add origins and uncertainty; limit claims when history is short. |
| Source revision | Backtest cannot be reproduced from archived data. | Corrected values overwrite historical snapshots. | Leakage and weak auditability. | Version source data and publication timestamps. |
| Policy mismatch | Forecast improves but inventory cost worsens. | Loss and replenishment decision are misaligned. | No realized business benefit. | Test the actual policy and cost asymmetry. |
| Unauthorized pooling | A data owner cannot identify permitted cross-use. | Training scope exceeds contractual rights. | Legal, security, and trust consequences. | Approve boundaries and enforce access controls before fitting. |
| Unsupported causal claim | Savings are attributed to a model after simultaneous policy changes. | Confounding affects observed outcomes. | Overstated return on investment. | Use controlled comparison or label scenario assumptions. |
| Missing operational owner | Alerts persist without action. | No escalation or rollback responsibility. | A known failure continues to affect orders. | Assign an owner, trigger, and executable fallback. |
| Forecast-plan confusion | Planner treats the point forecast as an approved order. | Decision and authorization layers are merged. | Excess or unauthorized commitment. | Separate artifacts, permissions, and approval records. |

A warning is a prompt to investigate, not proof of a particular cause.
For example, a subgroup error increase can result from missing promotions, inventory censoring, or genuine negative transfer.
Compare like-for-like inputs and origins before attributing the error to pooling.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or bound it |
|---|---|---|---|
| Shared patterns exist among selected series. | Cross-learning requires transportable information. | Different customer behavior, lifecycle, or fulfillment process. | Compare paired outcomes by segment and inspect local residuals. |
| Observed history is relevant to the future. | Training uses past relationships to forecast. | Regime change, promotions, closures, or supply shocks. | Evaluate across several origins and monitor drift. |
| Target quantities have consistent meaning. | A shared loss compares observations. | Mixing sales, shipments, cases, and eaches. | Audit units, lineage, and sample source records. |
| Availability is understood. | Sales can be censored by stockouts. | Missing or inaccurate inventory flags. | Compare sales against availability audits and sensitivity scenarios. |
| Inputs exist at forecast issuance. | A backtest should reproduce a feasible decision. | Future realized prices or late revisions enter features. | Reconstruct as-of snapshots and publication timestamps. |
| Evaluation cohorts represent deployment. | Held-out metrics support a rollout decision. | Excluding failed, new, sparse, or difficult series. | Reconcile evaluated series and volume to the intended scope. |
| Model selection is separate from final testing. | The final estimate should not be optimized by repeated selection. | Tuning and routing use test outcomes. | Audit configuration and freeze dates. |
| Economic costs are incremental and comparable. | Net benefit requires a common baseline and period. | Double-counted lost margin or omitted coexistence cost. | Reconcile cost ledger with finance and scenario-test. |
| Decision policy can use the forecast. | Accuracy has value through an actionable decision. | Capacity, contracts, or lead time prevent action. | Replay the actual policy with constraints and approvals. |
| Training rights cover pooled use. | Cross-learning may combine data from multiple owners. | Contractual restrictions or privacy requirements. | Obtain documented approvals and enforce boundaries. |
| Subgroups are sufficiently observed. | Segment performance needs enough evidence. | Few origins or highly correlated series. | Report counts and uncertainty; avoid precise claims from tiny cohorts. |
| The fallback remains available. | Operational failures require a safe route. | Retired incumbent, missing source, or unsupported ownership. | Exercise the fallback under realistic access and timing constraints. |

The worked example assumes observed sales equal the quantity the policy seeks to forecast.
That is often false during availability constraints.
Its one-period loss also ignores multi-period inventory dynamics and transfer between stores.
Its purpose is to show arithmetic and competing objectives, not to recommend a specific forecast or order.

A global model need not win every series to be economically attractive.
Conversely, aggregate improvement is not enough if an agreed service floor or contractual obligation is violated.
The decision owner must specify the acceptable trade-off before observing the final results.

## 12. Exercises

### Exercise 1 — Identify the sharing boundary

A retailer uses the same exponential-smoothing software on 500 item-store series, fitting each one separately.
A second team fits one regression coefficient vector using all 500 series and includes item-category indicators.
A third team estimates a common category effect plus series-specific deviations.

- Deliverable: classify each approach and identify which fitted quantities are shared.
- Acceptance criteria: distinguish reuse of software from reuse of fitted information; explain partial pooling.
- Time: 10 minutes.

### Exercise 2 — Build an origin-safe data contract

At Monday 08:00 a planner forecasts sales for the next week.
The sales feed covers the preceding week but can be corrected on Wednesday.
A planned promotion was entered Friday, then canceled Tuesday.
One store was closed for refurbishment, and another was open but out of stock.

- Deliverable: a six-row table identifying the usable snapshot, target treatment, and data-quality flags.
- Acceptance criteria: exclude the future cancellation and later sales correction from Monday inputs; distinguish closed, stockout, observed zero, and missing data.
- Time: 20 minutes.

### Exercise 3 — Calculate forecast metrics

Use the worked-example actuals of 130, 18, and 11 cases.
Local forecasts are 100, 20, and 10 cases; global forecasts are 120, 24, and 12 cases.

- Deliverable: absolute error by series, total WAPE for each model, and global signed bias.
- Acceptance criteria: show numerator and denominator, specify forecast-minus-actual sign, and identify the series harmed by the global forecast.
- Time: 15 minutes.

### Exercise 4 — Design two evaluations

The business wants both next-quarter forecasts for existing products and launch forecasts for products not yet seen in model training.
Available data cover three years of weekly sales and product attributes.

- Deliverable: a rolling-origin design and an unseen-product design, with training, tuning, final test, and feature availability rules.
- Acceptance criteria: respect time and label cutoffs; hold out identifiers for the launch question; define whether launch history is zero or partial.
- Time: 25 minutes.

### Exercise 5 — Diagnose pooling harm

A global model lowers total WAPE from 24% to 20%.
Large stores improve, while 70% of small rural stores have higher per-series absolute error.
Stockouts are more common in the rural stores, and the model uses observed sales as its target.

- Deliverable: a diagnostic plan and a proposed deployment guardrail.
- Acceptance criteria: consider both target censoring and negative transfer; avoid treating aggregate WAPE as proof that every store benefits.
- Time: 20 minutes.

### Exercise 6 — Evaluate annual economics

A replay estimates gross benefit of USD 420,000 per year.
Global recurring costs are USD 200,000 per year; the incumbent's avoidable recurring cost is USD 80,000 per year.
One-time implementation costs are USD 360,000.

- Deliverable: incremental recurring cost, annual net recurring benefit, simple payback, and two-year break-even annual gross benefit.
- Acceptance criteria: subtract only avoidable incumbent costs and label the undiscounted assumptions.
- Time: 15 minutes.

### Exercise 7 — Decide whether pooling is authorized and useful

A vendor proposes one model across two customers' store-level histories.
One customer's contract restricts use to its own planning, while the other has an unresolved missing-data problem.
The vendor reports a strong public benchmark score but no customer-specific backtest.

- Deliverable: a short decision memo stating whether to proceed, what evidence is missing, and who must approve the data flow.
- Acceptance criteria: address contract rights, data quality, benchmark transfer, and viable isolated alternatives; do not present legal advice as a conclusion.
- Time: 20 minutes.

### Exercise 8 — Design a controlled pilot

The sponsor wants to release a global forecast to 200 stores immediately.
The incumbent runs weekly, and planners can override proposed orders.

- Deliverable: a one-page pilot plan covering comparison, service guardrails, ownership, monitoring, and rollback.
- Acceptance criteria: distinguish forecast from order, define a pre-agreed decision rule, include failed forecasts and overrides, and describe a feasible control or acknowledge its limitations.
- Time: 30 minutes.

## 13. Self-check answers

### Exercise 1

The first approach is local because each series gets separately fitted parameters.
A common software package does not make its fitted values shared.
The second is global because examples from multiple series estimate a common regression vector.
Category indicators allow different predictions but do not make the shared vector local.
The third is partially pooled if the common effects and series deviations are estimated in a way that shares information.
The details of estimation matter; merely placing independent series models in the same file does not establish pooling.

### Exercise 2

A strong answer includes the following distinctions.

| Item | Monday-origin treatment |
|---|---|
| Prior-week sales | Use the version available by Monday 08:00, not Wednesday's later correction. |
| Planned promotion | Use the plan available by Monday 08:00; record uncertainty. |
| Tuesday cancellation | Exclude from Monday inputs because it was not yet known. |
| Closed store | Mark closure and eligibility; do not automatically interpret the absent sale as unconstrained demand of zero. |
| Stockout store | Record observed sales and availability; do not claim unobserved demand is zero. |
| Missing feed versus eligible zero | Mark a missing feed as unknown; record zero only when an eligible observed period genuinely had no sales. |

The corrected data and realized cancellation can be used later for appropriate retrospective analysis, but not as Monday-known features.
The target and scoring treatment must be specified consistently for every candidate.

### Exercise 3

Local absolute errors are 30, 2, and 1 cases, totaling 33 cases.
Global absolute errors are 10, 6, and 1 cases, totaling 17 cases.
Actuals sum to 159 cases.
Local WAPE is `33 / 159 = 20.75%`; global WAPE is `17 / 159 = 10.69%`.
Global signed errors are -10, +6, and +1 cases, totaling -3 cases.
Global bias is `-3 / 159 = -1.89%` using forecast minus actual.
Series B worsens from 2 to 6 cases of absolute error; C is unchanged.

### Exercise 4

A strong design uses multiple historical origins and ensures every training label ends by the relevant fitting cutoff.
For the existing-product question, a product's pre-origin history can be used, but post-origin observations cannot.
Tune on earlier origins and reserve later origins for the final comparison.
For the new-product question, hold out whole product identifiers from shared fitting and model selection as appropriate.
Provide only attributes and history available at the launch decision.
Zero-history launch and a four-week-history launch are distinct experiments.
Report both forecast success coverage and fallback behavior.
The final test should reflect the intended deployment population and calendar.

### Exercise 5

First verify consistent units, eligible weeks, and matched origins.
Separate periods with and without reliable availability; test whether censored sales distort the target and local comparator.
Then compare paired per-store errors and signed bias across multiple origins.
A subgroup may have a genuinely different shared relationship, but the observed result alone does not establish the mechanism.
A defensible guardrail could require no unacceptable degradation in rural service or error relative to the incumbent, with a specified fallback.
Set the threshold and the sample-size requirement before a final decision.

### Exercise 6

Incremental recurring cost is `USD 200,000 - USD 80,000 = USD 120,000 per year`.
Net recurring benefit is `USD 420,000 - USD 120,000 = USD 300,000 per year`.
Simple payback is `USD 360,000 / USD 300,000 = 1.2 years`.
Two-year break-even gross benefit is `USD 120,000 + USD 360,000 / 2 = USD 300,000 per year`.
This is undiscounted and assumes immediate full benefit and that the incumbent cost is actually avoidable.
If the incumbent remains necessary, subtracting its entire cost is inappropriate.

### Exercise 7

A strong memo does not approve cross-customer pooling on the information given.
It requests contract and privacy review by the accountable functions, along with documented permitted use and security controls.
It proposes resolving or bounding missingness and running an origin-safe customer-specific comparison.
A separate model per customer or a local baseline may be an option if permitted and technically feasible.
A public benchmark is evidence about that benchmark, not proof of local value or contractual authorization.
The memo must not claim that a particular contract or law permits the proposed use without review.

### Exercise 8

A strong pilot defines eligible stores and a common issue schedule, with a randomized or otherwise defensible comparison where feasible.
It prespecifies forecast metrics, operational outcomes, service floors, and the evaluation period.
It scores the complete candidate system including fallbacks and records planner overrides separately.
A named owner investigates data and cohort alerts, and a named operator can revert to the incumbent.
The plan specifies who authorizes orders and how the forecast, proposed order, and accepted commitment are recorded.
If a valid control is not feasible, the plan explains the resulting causal limitation.

## 14. Interview questions

1. **What makes a model global?**
   A strong answer identifies parameters learned from multiple series and reused across them.
   A weak answer says that a common library or a large number of models is sufficient.
2. **When can cross-learning help a new item?**
   A strong answer names relevant comparable histories, available attributes, and a realistic cold-start evaluation.
   A weak answer assumes that any global model can infer an unseen item's demand without useful inputs.
3. **Why might a global model hurt a small store?**
   A strong answer discusses heterogeneous patterns, weighting, censoring, and segment-level evidence.
   A weak answer relies exclusively on an improved network-wide WAPE.
4. **How do you prevent future-feature leakage?**
   A strong answer uses archived as-of plans, publication times, and fold-specific preprocessing.
   A weak answer merely sorts the final data table by date.
5. **How would you validate new-series generalization?**
   A strong answer holds out identifiers and gives the model only launch-time information.
   A weak answer randomly splits rows from the same products across train and test.
6. **What does WAPE conceal?**
   A strong answer describes volume weighting, zero-denominator cases, and the need for cohort and bias metrics.
   A weak answer treats it as an equal-weight average of per-series percentage errors.
7. **What is negative transfer?**
   A strong answer defines a recipient, a relevant comparator, and repeatable held-out degradation.
   A weak answer labels any single forecast error negative transfer.
8. **Why can lower error fail to reduce inventory cost?**
   A strong answer names the inventory policy, lead time, asymmetric costs, constraints, and forecast distribution.
   A weak answer assumes a fixed currency saving for each WAPE point.
9. **How do you handle an unseen embedding identifier?**
   A strong answer describes an unknown representation, observable attributes, and an evaluated fallback.
   A weak answer assumes an untrained identifier has a meaningful learned vector.
10. **What is required to deploy a pooled model across customers?**
    A strong answer includes permissions, data segregation, security, and customer-specific evaluation.
    A weak answer treats access to the files as authorization to reuse them.
11. **How do you detect and respond to drift?**
    A strong answer proposes cohort-level monitoring, thresholds, investigation, and owned rollback.
    A weak answer proposes a dashboard with no action or assumes all drift requires immediate retraining.
12. **How do you distinguish forecast, plan, target, and commitment?**
    A strong answer gives an uncertain estimate, chosen action, desired outcome, and authorized obligation respectively.
    A weak answer calls the point forecast an approved order.

## 15. Key terms

- [Series](#4-definitions-and-vocabulary): one identified time-indexed observation sequence.
- [Panel](#4-definitions-and-vocabulary): a collection of related or unrelated series.
- [Grain](#4-definitions-and-vocabulary): the exact business meaning of one row.
- [Local model](#3-the-core-mental-model): separately fitted series-specific parameters.
- [Global model](#3-the-core-mental-model): parameters learned across multiple series.
- [Partial pooling](#3-the-core-mental-model): shared effects combined with local deviations.
- [Cross-learning](#3-the-core-mental-model): information from one series influences another's forecast.
- [Negative transfer](#10-failure-modes-and-red-flags): a defined recipient loses from sharing versus a relevant comparator.
- [Cold start](#5-the-mechanics-step-by-step): little or no history for the series being forecast.
- [Embedding](#4-definitions-and-vocabulary): learned vector representing a category or identifier.
- [Origin](#5-the-mechanics-step-by-step): the forecast issuance time and information cutoff.
- [Horizon](#5-the-mechanics-step-by-step): time between origin and forecast target.
- [As-of snapshot](#8-data-standards-and-regulatory-context): data values available at a specified time.
- [Leakage](#10-failure-modes-and-red-flags): use of unavailable or reserved information.
- [Rolling origin](#5-the-mechanics-step-by-step): evaluation across multiple simulated issue dates.
- [Stockout](#5-the-mechanics-step-by-step): unavailability that can censor sales.
- [Censoring](#5-the-mechanics-step-by-step): observations limited by the process that produces them.
- [Normalization](#5-the-mechanics-step-by-step): scaling values using a declared transformation.
- [Loss](#5-the-mechanics-step-by-step): objective used to fit a model.
- [WAPE](#6-worked-example): total absolute error divided by total absolute actual volume.
- [Bias](#6-worked-example): signed error summary under a stated convention.
- [Fallback](#5-the-mechanics-step-by-step): executable alternative when the candidate cannot safely serve.
- [Drift](#10-failure-modes-and-red-flags): change in data or relationships over time.
- [Decision loss](#6-worked-example): monetary consequence of an action and outcome.
- [Forecast](#1-why-this-topic-exists): estimate of an uncertain future outcome.
- [Plan](#1-why-this-topic-exists): intended action selected from available information.
- [Target](#1-why-this-topic-exists): desired result, not necessarily the expected result.
- [Commitment](#1-why-this-topic-exists): accepted obligation governed by the applicable arrangement.

## 16. Sources

1. Salinas, D., Flunkert, V., Gasthaus, J., and Januschowski, T. “DeepAR: Probabilistic forecasting with autoregressive recurrent networks.” *International Journal of Forecasting*, 36(3), 2020, 1181–1191. [https://doi.org/10.1016/j.ijforecast.2019.07.001](https://doi.org/10.1016/j.ijforecast.2019.07.001). Shared neural probabilistic forecasting method and evaluation.
2. Makridakis, S., Spiliotis, E., and Assimakopoulos, V. “M5 accuracy competition: Results, findings, and conclusions.” *International Journal of Forecasting*, 38(4), 2022, 1346–1364. [https://doi.org/10.1016/j.ijforecast.2021.11.013](https://doi.org/10.1016/j.ijforecast.2021.11.013). Competition context and limitations of transferring a benchmark.
3. Montero-Manso, P., and Hyndman, R. J. “Principles and algorithms for forecasting groups of time series: Locality and globality.” *International Journal of Forecasting*, 37(4), 2021, 1632–1653. [https://doi.org/10.1016/j.ijforecast.2021.03.004](https://doi.org/10.1016/j.ijforecast.2021.03.004). Theoretical and methodological treatment of global and local forecasts.
4. Hyndman, R. J., and Athanasopoulos, G. *Forecasting: Principles and Practice*, third edition. [https://otexts.com/fpp3/](https://otexts.com/fpp3/). Forecast accuracy, time-series cross-validation, and practical baselines.
5. GS1. [GS1 identification keys](https://www.gs1.org/standards/id-keys). Official descriptions of identification keys including GTIN and GLN; identifiers do not determine data-use rights.
6. European Union. [Regulation (EU) 2016/679, General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj). Binding requirements apply where the regulation applies; see the actual text and qualified advice.
7. National Institute of Standards and Technology. [AI Risk Management Framework 1.0](https://www.nist.gov/itl/ai-risk-management-framework). Voluntary risk-management framework; not a model certification.
8. Makridakis Open Forecasting Center. [M5 competition](https://mofc.unic.ac.cy/m5-competition/). Competition information and retail forecasting context.

Sources support the methods and contextual claims indicated above.
The numerical worked case, cost figures, and training data in this topic are fictional teaching assumptions.

## 17. Links

- [Module 18 topic index](_index.md).
- [Neural network fundamentals](01_neural-network-fundamentals.md).
- [Sequence architectures](02_sequence-architectures.md).
- [Module 18 survey and its lab sections](../../../course/18_deep_learning_foundation_models_llm_agents.md).
- [Production labs module index](../../05_labs_electives/30_simulation_optimization_production_labs/_index.md).
