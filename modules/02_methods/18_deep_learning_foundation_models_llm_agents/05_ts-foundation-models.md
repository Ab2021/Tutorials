# Time-Series Foundation Models

> Module 18 — Deep Learning, Foundation Models, and LLM Agents  
> Topic 05 of 10  
> Estimated study time: 4 hours  
> Prerequisites: basic arithmetic and a spreadsheet; neural-model background is introduced below  
> Version 2026.5

## 1. Why this topic exists

A replenishment planner needs forecasts for thousands of product-location pairs.
Some pairs have years of clean weekly history; others are newly listed or intermittently sold.
A forecasting team may be able to fit separate models to the long histories but not maintain one model per short series.
A supplier offers a pretrained time-series model and claims it forecasts unfamiliar series without fitting on them.
The decision is whether that capability improves stock and service enough to pay for data, inference, integration, and operating risk.

A foundation model is a shared model trained in advance across many series and reused on another task or dataset.
Its prior exposure can let it recognize patterns with little local history.
That is a hypothesis about transfer, not a guarantee that the local demand process resembles the training distribution.
Pretraining is not the same thing as observing the future of the local series.
A model can fail on an important promotion or stockout even when its aggregate public benchmark score is excellent.

The business forecast is an estimate of a possible future, not an order quantity.
A plan is a chosen set of actions given constraints; a target is a desired outcome.
A commitment is an approved promise to a supplier or customer.
A forecast does not become a commitment just because an application writes a number into an order screen.
Inventory and service choices also depend on lead time, order multiples, capacity, cost, and risk tolerance.

This unit shows how to evaluate the whole decision, not just invoke an API.
You will establish a fair baseline, protect the evaluation timeline, interpret uncertainty, and price deployment.
The examples are fictional; no supplier claim or company outcome is implied.
Model selection remains conditional on the data and on the decision that consumes the forecast.

## 2. Learning outcomes

After this unit you can:

1. Explain pretraining and distinguish zero-shot inference from local fitting and fine-tuning.
   Exercise 1 checks the distinction.
2. Construct a leakage-safe rolling-origin evaluation with information available at each origin.
   Exercise 2 asks you to specify the information boundary.
3. Compute a weighted absolute error and compare it with a seasonal-naive baseline.
   Exercise 3 asks for the arithmetic and interpretation.
4. Interpret a forecast quantile for a supply decision without treating it as a guarantee.
   Exercise 4 tests the inventory interpretation.
5. Reconcile license, training-data, covariate, latency, and benchmark claims with evidence.
   Exercise 5 asks for a vendor-evidence table.
6. Calculate incremental annual economics and test the result against a downside case.
   Exercise 6 asks for a decision, not just a score.
7. Design a controlled fine-tuning experiment and decide whether deployment is justified.
   Exercises 7 and 8 test controls and rollout design.

## 3. The core mental model

A time-series foundation model is a reusable statistical machine, not a universal demand oracle.
A neural network maps input numbers into an output using learned numerical parameters called weights.
Training adjusts those weights to reduce an objective on examples; inference applies fixed weights to inputs.
A loss is a numerical penalty for poor predictions, not necessarily the same as business cost.
The pretraining corpus contains many sequences; the target business provides its own histories and decisions.

```text
historical public/licensed series          local series available at origin
            |                                          |
  pretraining objective                                | context, calendar,
            |                                          | known future inputs
    learned weights -------------------------> inference at each origin
            |                                          |
            | optional local training                  v
            +------------------------------> forecast distribution
                                                       |
                                   inventory / staffing / capacity policy
                                                       |
                                   cost, service, feasibility, monitoring
```

The arrows are temporal and contractual boundaries as well as data flow.
Pretraining uses information from a corpus that is available before the chosen model release.
At a historical evaluation origin, the local input must contain only records actually available then.
If the pretrained model itself was trained on future records from the evaluated dataset, a backtest can be contaminated.
A release date after the test period does not prove contamination, but makes a historical counterfactual less clear.
The responsible evaluator records model version, corpus disclosure, and uncertainty about exposure.

Zero-shot means no parameter updates on the target data for the stated task before prediction.
It still normally supplies a target history as context; zero-shot does not mean no target observations.
A frozen model can transform its context in memory without changing the stored parameters.
Fine-tuning updates some or all weights using target-related training examples.
Prompting with past observations, training a separate calibration layer, and updating weights are different interventions.
Label the calibration layer as local fitting even if the underlying model weights stay frozen.

Model outputs can be point forecasts or distributions.
A point forecast compresses uncertainty into one number and can be optimized for different losses.
A probability distribution describes a range of plausible futures conditional on the information and model.
Neither can reveal the precise effect of an unrecorded future competitor launch.
Economic value arises only when the forecast changes a feasible decision in a favorable way.
The same forecast error reduction can be valuable for an expensive, long-lead-time item and immaterial for a cheap item with abundant capacity.

## 4. Definitions and vocabulary

The vocabulary below is deliberately operational: the unit and the common misinterpretation matter.

| Term | Meaning and unit | Common misuse |
|---|---|---|
| Time series | values indexed by ordered time buckets, such as units/week | assuming every absent record is zero demand |
| Target | quantity to predict, such as sales units per SKU-location-week | confusing a forecast target with a management goal |
| Series identifier | stable key for item and location | treating a changed product code as a new demand process without mapping |
| Frequency | duration of one bucket, such as one week | assuming a monthly model has the same context length in calendar time as a daily one |
| Context | historical values and allowed features supplied at inference | calling a context-free request zero-shot by definition |
| Horizon | number of future buckets predicted from an origin | comparing models with different horizons |
| Origin | timestamp at which a forecast is issued | using revised data published after that timestamp |
| Pretraining | parameter learning on a broad corpus before target-task deployment | assuming it necessarily includes the buyer’s series |
| Checkpoint | versioned saved set of model weights and configuration | treating a model family name as a reproducible version |
| Zero-shot | prediction without target-task parameter updates | claiming that no local history or preprocessing is supplied |
| Fine-tuning | additional parameter updates on local or related training data | calling prompt construction fine-tuning |
| Frozen model | model whose pretrained parameters are not updated | assuming all downstream processing is also untrained |
| Global model | one set of parameters shared across multiple series | assuming it is necessarily a pretrained foundation model |
| Local model | model estimated separately for a single series | assuming local means simple or necessarily worse |
| Transfer | use of knowledge learned on a source distribution for a target | assuming all transfers are beneficial |
| Negative transfer | degradation caused by transferred patterns | mistaking it for evidence that all shared learning fails |
| Domain shift | difference between training and operating data-generating conditions | assuming every shift is visible from aggregate accuracy |
| Covariate | a feature other than the target, with an explicit availability time | using realized future price as if it were known |
| Known-future feature | input genuinely available for each forecasted bucket at the origin | treating an unapproved promotion plan as a committed fact |
| Static feature | attribute treated as constant for an example, such as product category | assuming categories never change |
| Missing observation | value not observed or not delivered by the cutoff | silently replacing missing data with zero |
| Censoring | observed sales limited by availability rather than demand | training as though a stockout reveals zero customer demand |
| Intermittent demand | many periods without observed demand and occasional positive demand | assuming every zero reflects a stockout |
| Scaling | transformation such as dividing a series by a historical magnitude | using future test values in scale calculation |
| Rolling origin | repeated forecasts at historical decision timestamps | mixing training and evaluation dates |
| Holdout | data not used for model or policy selection | retuning after looking at its results and still calling it untouched |
| Leakage | use of information unavailable or impermissible at a forecast origin | limiting the concept to a single train/test split |
| Contamination | evaluation examples or close derivatives exposed during pretraining | assuming exposure is proven solely from a strong score |
| MAE | mean absolute error, in target units per bucket | comparing unweighted MAE across incompatible portfolios without interpretation |
| WAPE | sum of absolute error divided by sum of actual demand, dimensionless | assuming it is defined when the total actual is zero |
| Bias | signed forecast error summary with an explicit sign convention | assuming low absolute error rules out systematic underforecasting |
| Quantile | value qτ with modeled probability τ of demand at or below it | treating a 90th percentile as a 90% guarantee for every item |
| Coverage | fraction of observed values inside a stated prediction interval | assuming nominal coverage alone proves useful interval width |
| Lead time | time between an order decision and receipt, measured in buckets | evaluating one-week accuracy for a twelve-week order decision |
| Incremental value | difference in business outcome versus a stated alternative | counting all portfolio margin as model benefit |
| Total cost of ownership | recurring and one-off costs of an operating solution | comparing API price alone with internal labor |

A forecast request should name the target, units, frequency, origin, horizon, model version, and output type.
Where a model reports quantiles, specify whether they are marginal by bucket or samples of joint paths.
Marginal quantiles cannot in general be summed to produce the same quantile of cumulative lead-time demand.
Dependencies between weeks matter when the order is exposed to aggregate demand.

## 5. The mechanics, step by step

The sequence below describes a defensible buying or building process.
Each step names its input, work, output, and decision branch.
A small, reliable baseline is a requirement for comparison, not a consolation prize.

### Step 1: Specify the decision

**Input:** A replenishment or capacity decision with an owner.

**Work:** Document the cadence, lead time, constraints, and cost of shortage and surplus.

**Output:** A decision card with a frozen comparison policy.

**Branch:** If no decision changes when the forecast changes, do not attribute financial benefit to the model.

### Step 2: Define the target

**Input:** Transaction, availability, and master-data records.

**Work:** Choose sales, unconstrained demand, shipments, or another measurable quantity; state the unit and time bucket.

**Output:** A target definition with a mapping from source records.

**Branch:** If availability censors sales, flag the target and avoid claiming measured sales are true demand.

### Step 3: Create an as-of dataset

**Input:** Timestamped historical values and ingestion timestamps.

**Work:** Reconstruct the data actually visible at each origin, including revisions and delayed reports.

**Output:** A versioned dataset and availability-time rules.

**Branch:** If historical snapshots do not exist, label the backtest approximate and quantify likely revision exposure.

### Step 4: Inspect data quality

**Input:** Per-series histories, lifecycle dates, and availability flags.

**Work:** Measure missingness, zeros, returns, aggregation changes, promotion coverage, and item substitutions.

**Output:** A segment-level quality report with exclusions and reasons.

**Branch:** If the latest context is mostly missing, request remediation or a fallback rather than inventing zeros.

### Step 5: Choose decision-relevant segments

**Input:** The product-location population and business priorities.

**Work:** Group by demand volume, intermittency, lifecycle, seasonality, service criticality, and lead time.

**Output:** A frozen segment map and weights.

**Branch:** If a high-cost segment is small, report it separately so that a high-volume group cannot conceal its failure.

### Step 6: Define historical origins

**Input:** The business forecast cadence and at least one protected test period.

**Work:** Select several origins across seasonal and promotional regimes; state context and horizon.

**Output:** A dated evaluation calendar with development and untouched test periods.

**Branch:** If only one origin exists, call the result a single holdout, not robust evidence across regimes.

### Step 7: Build local baselines

**Input:** The same as-of contexts and forecast horizons.

**Work:** Calculate last-value and seasonal-naive forecasts where their histories exist; include an established production model.

**Output:** Baseline forecasts and documented fallback behavior.

**Branch:** If a seasonal value is unavailable, use a predefined fallback and report its frequency.

### Step 8: Inspect model eligibility

**Input:** A versioned model card, license, hardware or API contract, and data policy.

**Work:** Check target frequency, context and horizon limits, missing-value treatment, output format, and permitted use.

**Output:** A compatibility matrix and implementation estimate.

**Branch:** If an essential covariate or frequency is unsupported, change the experiment or reject the candidate.

### Step 9: Audit pretraining claims

**Input:** Vendor model documentation and corpus descriptions.

**Work:** Record source domains, cutoff if disclosed, potential overlap with test datasets, and rights to use outputs.

**Output:** A provenance register distinguishing known, unknown, and asserted facts.

**Branch:** If overlap cannot be excluded, qualify the historical test and prefer an additional prospective evaluation.

### Step 10: Prepare model inputs

**Input:** Origin-safe histories and compatible feature schemas.

**Work:** Apply documented context truncation, scaling, missing-value handling, and any allowed frequency conversion.

**Output:** Reproducible inputs with preprocessing parameters stored per origin.

**Branch:** If scaling divides by zero or uses future values, repair the transform before scoring.

### Step 11: Run a frozen zero-shot trial

**Input:** The fixed checkpoint and prepared historical contexts.

**Work:** Generate forecasts without updating target-task model weights and record all non-model local transformations.

**Output:** Versioned zero-shot forecasts, runtime, failures, and request costs.

**Branch:** If the vendor silently adapts to local data, classify and evaluate it as adaptation, not a pure frozen trial.

### Step 12: Develop fine-tuning only on training data

**Input:** A separate earlier training window and frozen validation origins.

**Work:** Create input-output windows with label ends strictly before validation origins; tune hyperparameters without using the final holdout.

**Output:** A fine-tuned checkpoint and full training recipe.

**Branch:** If local data are sparse or validation worsens, retain zero-shot or the baseline.

### Step 13: Compare point forecasts

**Input:** All candidates scored on identical valid series-origin-horizon pairs.

**Work:** Calculate WAPE, signed bias, and metrics by horizon and segment; report denominators and failures.

**Output:** A paired comparison with excluded cases and sample counts.

**Branch:** If a model fails on requests, count the fallback outcome and operating cost rather than dropping hard cases.

### Step 14: Assess uncertainty

**Input:** Distributional output or forecast paths and realized demand.

**Work:** Evaluate quantile loss, interval coverage, interval width, and lead-time aggregate distributions.

**Output:** Calibration and sharpness results by segment and horizon.

**Branch:** If the model emits only point forecasts, do not invent a calibrated probability distribution.

### Step 15: Simulate the operating policy

**Input:** Forecasts, actuals, lead times, starting stock, receipts, capacity, and a fixed policy.

**Work:** Replay replenishment with the same rules for every candidate, modeling stock and shortages over time.

**Output:** Comparable cost and service outcomes with assumptions.

**Branch:** If demand is censored by stockouts, bound the impact and avoid pretending observed sales equal unmet demand.

### Step 16: Calculate financial impact

**Input:** The policy replay, current baseline costs, implementation estimates, and vendor contract.

**Work:** Annualize incremental benefits and subtract recurring charges plus an explicitly treated share of setup cost.

**Output:** Base, downside, and upside scenarios with sensitivity to realization.

**Branch:** If net value is uncertain or negative, run a limited prospective trial or do not deploy.

### Step 17: Pilot and release

**Input:** A selected candidate, data-sharing approval, and operational runbook.

**Work:** Shadow the incumbent, compare paired forecasts, test fallbacks, and obtain business sign-off before changing orders.

**Output:** A versioned release decision with rollback and named ownership.

**Branch:** If a service-level or data-control gate fails, keep the incumbent while correcting the issue.

### Step 18: Monitor and retire

**Input:** Production forecasts, actuals as they mature, costs, model versions, and intervention logs.

**Work:** Track bias, errors, calibration, latency, failures, segment drift, and realized operating outcomes.

**Output:** A periodic review and threshold-based escalation plan.

**Branch:** If the model version or business process changes, reopen the evaluation and preserve the previous checkpoint.

### Evaluation design in calendar time

Consider a buyer who issues a four-week forecast every Monday.
Suppose the forecast is made at 09:00 in the buyer's local time zone.
A sales file for the prior week normally arrives at 06:00 on Monday.
The file is sometimes revised at 15:00 on Tuesday.
For that Monday's backtest, the 06:00 file is eligible and Tuesday's revision is not.
The target actual used for eventual scoring can be the later finalized value.
That difference is legitimate because the input and the realized outcome serve different purposes.
Keep both versions and document the time at which each became available.

For each historical origin, make a separate context ending at the latest eligible observation.
Generate four distinct horizon values, labeled h=1 through h=4.
The h=1 value belongs to the first future week and the h=4 value to the fourth.
Different origins can forecast the same future week at different horizons.
These are distinct decision records, not duplicate forecasts to be discarded automatically.
Decide whether the business consumes every weekly refresh or only the original order forecast.
Score the records corresponding to the actual decision cadence.

Adjacent origin errors are correlated because their contexts and forecasted weeks overlap.
Do not treat every horizon row as an independent experiment when estimating uncertainty.
Where practical, resample or compare at the origin or other appropriate business cluster.
Report the number of origins, series, and forecast rows separately.
A small number of seasonal cycles limits the evidence even when the table has millions of rows.
Consider separate peak and non-peak reports rather than relying on one annual average.

An illustrative split might reserve the earliest available years for any local training.
Use later rolling origins for feature choices and fine-tuning decisions.
Reserve the latest complete seasonal cycle as an untouched final test when data permit.
The exact dates must precede the operational deployment date.
Windows that begin in training but have labels reaching into validation are not training-safe.
Exclude or truncate those windows so that the training labels end before validation begins.
Also account for delayed publication, not only the calendar date of the label.

If a checkpoint was released after a historical origin, it did not exist for a real decision at that origin.
The exercise may still measure retrospective predictive association under a present-day model.
It cannot be described without qualification as a feasible historical deployment simulation.
If its pretraining cutoff is later than the historical test values, exposure is an additional concern.
Where the corpus is opaque, preserve that uncertainty in the result.
A forward test issued after the checkpoint release addresses the feasibility issue.
It does not by itself eliminate other data leakage or operational confounding.

### Compare methods on the same population

A model evaluation should start with an eligibility table, not with whichever predictions returned successfully.
Each row should identify the series, origin, horizon, target unit, and actual-availability status.
Freeze eligibility rules before comparing candidate outcomes.
For example, require an active item at the origin and a measurable subsequent target.
Do not require a full year of history if the business also needs short-history items.
Instead, identify which baseline and model can serve each history-length segment.

Define a fallback for each method before looking at test errors.
If a seasonal-naive forecast lacks a comparable week, a documented last-value rule may be used.
If a hosted model times out, the approved incumbent may be the fallback.
Measure the overall outcome of the model-plus-fallback policy.
Also report the primary model's failure rate and the standalone fallback's contribution.
Otherwise a candidate can appear accurate by omitting its hardest requests.

Compare at least three operationally meaningful alternatives where feasible.
The current production process establishes the decision-relevant incumbent.
A simple seasonal-naive rule shows how much benefit comes from recurring patterns.
The frozen foundation model tests transfer without local weight adaptation.
A fine-tuned version tests whether extra local training earns its added cost.
Their inputs should have comparable information rights and the same scoring population.

If one candidate cannot use a known future promotion indicator, do not secretly supply it to another.
One fair comparison uses the common feature set accepted by all candidates.
A separate decision scenario can compare the best feasible configurations of each product.
Label these two questions distinctly: model transfer versus total solution capability.
The difference matters because business value may come from the covariate pipeline rather than the checkpoint.
Record the extra data, approval, and maintenance cost in the solution comparison.

### Interpret error measurements

WAPE is attractive because the numerator and denominator aggregate naturally across rows.
Its denominator is the sum of actual volume across the scored rows.
It is undefined when that sum is zero, not automatically zero.
It can also change with the scale and mix of demand even if the underlying errors are unchanged.
Report both numerator and denominator when comparing periods or portfolios.
For intermittent low-volume items, add measures that do not hide critical zero-demand cases.

Mean absolute error divides absolute error by the number of scored observations.
Its unit remains the same as the target, such as units per item-week.
The metric is meaningful only when the aggregation level and row population are clear.
A portfolio average can be dominated by many easy low-volume observations.
Conversely, WAPE can be dominated by a few high-volume items.
Use both alongside business segments, rather than declaring one universally correct.

Signed bias requires an explicit convention.
This unit uses forecast minus actual, so positive bias means overforecasting.
The opposite convention is also used in practice and reverses the sign.
Bias can cancel across weeks or items even when all individual errors are large.
Inspect the distribution of signed errors and systematic directional errors by segment.
Compare bias with service and excess outcomes rather than treating it as a complete score.

If prediction intervals are reported, calculate empirical coverage after outcomes mature.
For an 80% interval, coverage is the share of eligible actuals between its lower and upper bounds.
Also report average interval width in target units and the number of eligible observations.
An interval that always spans every plausible number can achieve high coverage but be operationally useless.
Break out coverage by forecast horizon and important business segment.
Small segment counts require caution; a single miss can materially change the observed percentage.

### Reconcile different forms of uncertainty

Uncertainty in future customer demand is not the same as uncertainty about the model's parameters.
Measurement error, missing values, and future promotions add further uncertainty.
An output interval may cover some of these sources but not all of them.
Ask the provider exactly what its forecast distribution is conditional on.
If a promotion is supplied as certain, the prediction may not include the risk of cancellation.
Run separate scenarios or use an explicit model for uncertain plans when the decision requires it.

For a four-week lead time, the relevant demand is often the sum of four future weekly values.
Each weekly marginal distribution says something about a single week.
It does not completely specify whether the weeks rise and fall together.
Positive dependence can create a high aggregate tail even if weekly intervals look reasonable.
Joint forecast paths preserve the model's stated dependence across the horizon.
Sum each path first, then calculate the desired quantile across path totals.

If a model supplies only independent weekly samples, that independence is an assumption.
It may underestimate or overestimate cumulative uncertainty when real demand is correlated.
If only marginal quantiles are supplied, their sums are not generally aggregate quantiles.
Measure aggregate calibration against historical lead-time totals.
If appropriate joint predictions are unavailable, treat the resulting safety-stock estimate as conditional.
The policy owner may prefer an independently validated aggregate-demand model.

### Investigate a failed transfer

Suppose the zero-shot model is accurate on stable high-volume products but weak on new launches.
First confirm that launch dates and product mappings were available at each origin.
Then examine context length, missingness, stockout flags, and whether initial sales were distribution-constrained.
Compare the same launch cohort with the current rule and a relevant analog-item method.
Do not attribute the failure to model architecture before checking the data and population.
Record whether the target is a truly new item or a recoded continuation.

Suppose fine-tuning improves training loss but worsens the protected validation period.
Possible explanations include insufficient independent series, regime change, and inappropriate learning rate.
A local dataset with many overlapping windows can still contain few independent demand episodes.
Inspect per-segment and per-horizon results before interpreting the aggregate.
You may select a frozen checkpoint if fine-tuning fails the predeclared validation gate.
Do not use the protected final test repeatedly to rescue a failing training recipe.

Suppose the candidate lowers WAPE while shortages rise in the policy replay.
Check whether underforecasting is concentrated on expensive or long-lead-time items.
Check whether the policy uses a mean forecast when its shortage costs call for a higher quantile.
Check lead-time assumptions, order constraints, and inventory starting conditions.
An error metric and a policy outcome answer different questions.
The policy comparison is decisive for a replenishment use case only if its assumptions are credible.

### Move from replay to a controlled pilot

In shadow mode, publish candidate forecasts alongside the incumbent without changing orders.
Store the forecast before the corresponding actual is known.
Record any human overrides separately from the model output.
When actuals mature, compare paired forecasts and identify difficult segments.
Shadow mode checks information flow and prediction quality without exposing live inventory to a new policy.
It cannot alone prove that a different ordering policy will deliver the simulated savings.

In a bounded pilot, choose eligible items and locations before observing the outcome.
Where practical, allocate comparable units to candidate and control policies.
Document operational constraints, planner interventions, and the chosen comparison method.
Avoid switching the treatment based on observed early performance without recording the rule.
Track stock availability, excess, order stability, and realized costs as well as forecast metrics.
Explain residual confounding when random allocation is infeasible.

Specify stop conditions in advance.
Examples include a sustained service deterioration, an invoice above the approved limit, or repeated inference failures.
The exact thresholds belong to the business owner and must match the risk of the pilot.
A fallback must be executable within the actual decision window, not merely named in a document.
Keep the previous policy and required data available until rollback is no longer needed.
Record who can stop the pilot and who authorizes a later rollout.

### Account for recurring change

Demand patterns change when an item matures, a competitor enters, or distribution expands.
Supplier lead times and service priorities can change independently of the demand pattern.
A model trained on old behavior may retain useful patterns but respond too slowly to a new regime.
Compare recent errors with appropriate historical segments rather than using a single universal threshold.
Investigate missing feeds and hierarchy changes before interpreting a score change as model drift.
Retain the historical inputs needed to reproduce material decisions.

If the provider releases a new checkpoint, treat it as a new candidate.
Re-run the agreed evaluation on data available at the relevant prospective date.
Compare output formats, feature support, cost, latency, privacy terms, and fallback behavior as well as accuracy.
Do not overwrite the previous forecast history or its associated version metadata.
If a business rule or inventory target changes, document that separately from the model change.
Otherwise observed financial differences cannot be attributed cleanly to either intervention.

Review model ownership at least as often as the business review cadence requires.
The owner should know who manages the source data, checkpoint, API contract, and policy.
Retirement may be appropriate when an incumbent becomes cheaper, more reliable, or equally effective.
Export permitted artifacts and follow contractual deletion procedures when leaving a provider.
Maintain access to approved historical forecasts for audit and finance review as required by policy.
A foundation model is a replaceable component in the planning process, not the process owner.

### Calculate infrastructure volume

Suppose 20,000 series are forecast each week with one request per series.
With 52 weekly runs, successful demand is 1,040,000 requests per year.
If 2% of those requests each require one retry, total traffic becomes 1,060,800 requests per year.
At an illustrative USD 0.01 per request, request charges alone are USD 10,608 per year.
This example assumes retries are billed at the same rate and every retry succeeds.
It excludes storage, networking, security, support, and minimum contractual charges.

If the provider charges by input length instead, request count alone is insufficient.
Multiply each request's actual billed input and output quantities by the applicable rates.
Include variable context length, padding, output samples, and the frequency of batch runs.
If there are service tiers, apply each tier rather than one average unit price.
Review whether minimums and reserved capacity are incremental or already included elsewhere.
Record currency, tax treatment, contract period, and exchange-rate assumptions.

For self-hosting, include the useful life and utilization of shared hardware.
Account for engineering time, monitoring, security patches, and resilience capacity.
Do not assign all shared platform cost to one pilot if the same expense is already funded for other workloads.
Conversely, do not call hardware free just because a different team owns the budget.
Use the same accounting boundary for the incumbent and the proposed solution.
Finance should reconcile the decision model to expected invoices and staffing commitments.

### Pretraining objectives and architecture

Different model families build generality in different ways.
An autoregressive objective predicts later values or tokens from earlier context.
A masked-reconstruction objective hides part of a sequence and asks the network to reconstruct it.
A direct multi-horizon objective predicts a future block from an observed context.
Each objective defines what the model practices during training; none proves that its forecasts minimize your inventory cost.

Some models scale continuous values and quantize them into discrete tokens before sequence modeling.
Quantization trades numeric resolution for a finite vocabulary and requires careful decoding back to original units.
Other models process numerical patches or generate output values directly.
An architecture diagram alone does not establish whether future price, promotion, or static attributes are usable.
Ask the provider which features the exact checkpoint and inference interface support.

Training may mix real and synthetic series with different frequencies, lengths, and domains.
Synthetic data can expand coverage but may omit practical causal mechanisms and observation biases.
The number of series, observations, tokens, and domains are different denominators.
A larger reported corpus does not prove more independent examples, cleaner labels, or better local fit.
The useful question is whether performance transfers to the decisions you must make.

### Zero-shot is a protocol, not an endpoint label

At each origin choose a context ending before the forecast horizon.
Apply only transformations permitted under the experiment protocol.
Load one immutable checkpoint and make the prediction without target-task weight updates.
If an API adapts the weights behind the scenes or stores customer-specific state, disclose that separately.
If you fit a bias correction using historical target errors, report it as local post-processing.
Run the same correction opportunity for a baseline when it is an allowed method.

Context is not a training label in the ordinary online inference sense, but can include local history.
A model may exploit the recent level, weekly seasonal cycle, or an anomalous last observation in that context.
Longer context is not automatically better: truncation, drift, and training limits affect results.
Select the length on development origins, not the final test.
Describe new-series cases separately when there is genuinely no local history.

### Fine-tuning and validation

Construct a supervised window from context ending at time t and labels in the next H buckets.
During training, update weights using only windows whose labels precede the validation boundary.
Overlapping windows are dependent observations; their count is not the number of independent series.
Select checkpoint, learning rate, stopping rule, and any frozen layers on development validation only.
Compare against the frozen checkpoint on the same origins and under the same feature permissions.
Do not quietly give fine-tuning a covariate unavailable to its competitors.

Some adaptation updates a small subset of weights; full fine-tuning updates the entire checkpoint.
Both require a record of the base version, training data, seed where applicable, and resulting artifact.
A small update may be cheaper to store, but is not guaranteed to avoid overfitting or privacy leakage.
When local history is short, training may memorize a promotion or a temporary capacity constraint.
Reserve a genuinely later holdout and inspect errors by horizon, not only the mean score.

### Distributions and replenishment

For a critical-fractile decision under a simplified single-period model,
choose a quantile with probability equal to underage cost divided by the sum of underage and overage costs.
For example, underage cost USD 9/unit and overage cost USD 3/unit imply a 0.75 quantile.
This result assumes known costs, one decision period, and a demand distribution for the relevant exposure.
It does not by itself handle multiple locations, capacity, minimum orders, or substitutions.
For a multiweek lead time, evaluate samples or an appropriate model of aggregate demand rather than summing weekly 0.75 quantiles.

For quantile level τ and outcome y, pinball loss is `(τ - I[y < q]) * (y - q)`.
Lower average pinball loss is better for the specified quantile and target units.
For an interval, also report coverage and average width; a vacuously wide interval may cover almost everything.
Compute metrics after the relevant actuals arrive, and retain the origin-specific forecast version.

## 6. Worked example

### Business setting and fixed rules

A fictional distributor replenishes a regional product family weekly.
The lead time is four weeks and the model is refreshed every week.
The decision owner wants to reduce shortage and excess stock compared with a seasonal-naive policy.
The target is recorded sales units per week for the family and region, not a direct measure of unconstrained demand.
Stockout flags are available; their presence limits interpretation of observed sales.
The case is illustrative and does not claim a real deployment outcome.

At one historical origin, the four actual sales totals are 112, 97, 134, and 107 units/week.
A seasonal-naive baseline predicts 105, 110, 121, and 115 units/week.
The frozen pretrained candidate predicts 109, 101, 129, and 111 units/week.
Both see the same available history and neither uses actual future prices or sales.
Assume the model version was available for the trial and potential corpus overlap has been investigated.
This is one origin and one aggregate series, so the example illustrates arithmetic rather than proving generalization.

| Week | Actual units | Baseline units | Candidate units | Baseline absolute error units | Candidate absolute error units |
|---:|---:|---:|---:|---:|---:|
| 1 | 112 | 105 | 109 | 7 | 3 |
| 2 | 97 | 110 | 101 | 13 | 4 |
| 3 | 134 | 121 | 129 | 13 | 5 |
| 4 | 107 | 115 | 111 | 8 | 4 |
| Total | 450 | 451 | 450 | 41 | 16 |

Baseline WAPE is `41 / 450 = 0.09111`, or 9.11% of actual units.
Candidate WAPE is `16 / 450 = 0.03556`, or 3.56% of actual units.
The absolute reduction is 5.56 percentage points, not 5.56% relative improvement.
The relative reduction is `(41 - 16) / 41 = 60.98%` in WAPE for this example.
Baseline mean absolute error is `41 / 4 = 10.25 units/week`.
Candidate mean absolute error is `16 / 4 = 4.00 units/week`.
The denominator for WAPE is positive; if it were zero, WAPE would be undefined.

Using forecast minus actual, baseline signed errors are -7, +13, -13, and +8 units.
Their sum is +1 unit, so signed bias is `1 / 450 = +0.22%` of actual volume.
Candidate signed errors are -3, +4, -5, and +4 units.
Their sum is 0 units, so signed bias is 0% over these four weeks.
Zero aggregate bias masks both underforecast and overforecast in individual weeks.
It also says nothing about lead-time quantile calibration or inventory cost.

### Translate the forecast into a decision

Suppose a separate simulation applies the same replenishment rule to all candidates.
The simulation uses actual lead times, opening inventory, scheduled receipts, order multiples, and an agreed shortage-cost assumption.
Its result is an estimated annual shortage-cost reduction of USD 82,000 and excess-stock carrying-cost reduction of USD 28,000 versus the baseline.
These two savings are assumed incremental and non-overlapping; the four-week WAPE arithmetic does not generate them.
Assume the annual figures come from a broader representative replay, not the single four-week example.
A controller reviews the assumptions because counterfactual shortage costs are not directly observable from sales alone.

The annual recurring API and infrastructure cost is USD 26,000.
Data operations and support cost USD 18,000/year.
One-time implementation costs USD 60,000, evaluated as USD 20,000/year over three years without discounting.
The simplified annual net benefit is `82,000 + 28,000 - 26,000 - 18,000 - 20,000 = USD 46,000/year`.
The undiscounted three-year total is `3 * (110,000 - 44,000) - 60,000 = USD 138,000`.
Do not subtract the annualized setup allowance a second time in that three-year calculation.
This is an accounting illustration, not a net present value or a claim that value can be realized immediately.

If only half the simulated gross benefit is realized, gross benefit is USD 55,000/year.
The same annual cost and setup allowance imply `55,000 - 44,000 - 20,000 = -USD 9,000/year`.
Break-even realized gross benefit is USD 64,000/year, or `64,000 / 110,000 = 58.18%` of the estimated gross benefit.
The result is sensitive to realization and to the shortage-cost assumption.
A decision maker should request a prospective shadow test and controlled policy trial before committing the full rollout.
The forecast itself is not the purchase order; the approved plan and supplier commitment remain separate records.

### What the example cannot establish

One origin cannot measure stability across seasonal peaks, promotions, and supply disruptions.
A strong aggregate result may conceal poor performance on low-volume critical items.
The annual economic estimates need traceable simulation inputs and a comparison to the existing policy.
If model requests fail, the economic result must include fallback performance and incident labor.
A model with lower error but higher supply cost should not win merely because its WAPE is better.

## 7. Code, query, or calculation

The following Python 3 example uses only the standard library.
It reproduces the four-week comparison and economics above, including signed bias and the downside case.
It is an evaluation harness for supplied predictions, not a foundation-model implementation or a demonstration of pretrained performance.
Substitute forecasts generated by a real versioned checkpoint only after respecting the origin and data controls.
The example treats all weekly actuals as observed, positive-volume sales and does not simulate inventory.

```python
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Evaluation:
    mae_units_per_week: float
    wape: float
    bias: float


def evaluate(actual: Sequence[float], forecast: Sequence[float]) -> Evaluation:
    if not actual or len(actual) != len(forecast):
        raise ValueError("actual and forecast must have equal, nonzero length")
    total_actual = sum(actual)
    if total_actual <= 0:
        raise ValueError("WAPE requires a positive total actual")
    errors = [f - a for a, f in zip(actual, forecast)]
    absolute_error = sum(abs(error) for error in errors)
    return Evaluation(
        mae_units_per_week=absolute_error / len(errors),
        wape=absolute_error / total_actual,
        bias=sum(errors) / total_actual,
    )


def annual_net(gross_benefit: float, recurring_cost: float,
               setup_cost: float, years: int) -> float:
    if years <= 0:
        raise ValueError("years must be positive")
    return gross_benefit - recurring_cost - setup_cost / years


def main() -> None:
    actual = [112, 97, 134, 107]
    baseline = [105, 110, 121, 115]
    candidate = [109, 101, 129, 111]
    base = evaluate(actual, baseline)
    model = evaluate(actual, candidate)
    for name, result in (("baseline", base), ("candidate", model)):
        print(f"{name}: MAE={result.mae_units_per_week:.2f} units/week, "
              f"WAPE={result.wape:.2%}, bias={result.bias:+.2%}")
    relative = (base.wape - model.wape) / base.wape
    print(f"relative WAPE reduction={relative:.2%}")
    gross = 82_000 + 28_000
    recurring = 26_000 + 18_000
    setup = 60_000
    print(f"base annual net=USD {annual_net(gross, recurring, setup, 3):,.0f}")
    print(f"half-benefit annual net=USD {annual_net(gross / 2, recurring, setup, 3):,.0f}")
    print(f"break-even realization={(recurring + setup / 3) / gross:.2%}")


if __name__ == "__main__":
    main()
```

Expected output, with standard Python rounding:

```text
baseline: MAE=10.25 units/week, WAPE=9.11%, bias=+0.22%
candidate: MAE=4.00 units/week, WAPE=3.56%, bias=+0.00%
relative WAPE reduction=60.98%
base annual net=USD 46,000
half-benefit annual net=USD -9,000
break-even realization=58.18%
```

For a real experiment, store one row per model, series, origin, and horizon.
Save actuals separately with their publication and revision timestamps.
Join on those keys, assert identical eligibility, and report missing forecasts rather than silently discarding them.
Use total absolute error divided by total actual for portfolio WAPE, not an unweighted mean of per-series WAPEs.
Do not annualize a single four-week result as if it were a representative full-year policy experiment.

## 8. Data, standards, and regulatory context

### The information contract

A series needs a stable item and location identity, a bucket definition, and a time-zone and calendar convention.
A week can end on different days across trading partners; reconcile the convention before comparing forecasts.
Record the period covered, the date a value became available, and any later revision.
A data extract made today is not automatically a faithful reconstruction of what was known last year.
Master-data changes and product replacements also need effective dates.
Without them, a backtest may use today's product hierarchy to select a historical population.

GS1 identifiers can support consistent identification and exchange across organizations.
The GS1 General Specifications define identification and barcode-related rules; they do not define your statistical target or guarantee data quality.
A Global Trade Item Number (GTIN) identifies a trade item, but forecasting still needs a declared location, time bucket, and lifecycle mapping.
A shared identifier does not prove that two partners measure the same demand quantity.
Consult the applicable GS1 specification and trading-partner agreement rather than assuming all item codes are GTINs.

### Data rights and privacy

Before uploading data to a hosted model, identify the controller or owner, allowed purpose, retention, subprocessors, and permitted geography.
Supplier prices, customer orders, inventory positions, and locations may be commercially sensitive even if not personal data.
Personal data can appear in order histories or free-text fields; remove fields not required for forecasting.
Ask whether provider training, abuse monitoring, caching, support access, or logging uses customer inputs or outputs.
Record the contract, actual configuration, and deletion procedure; a marketing statement is not a substitute for terms.

The EU General Data Protection Regulation (GDPR), where applicable, governs personal-data processing, not all time-series data as such.
Its requirements depend on facts including role, purpose, lawful basis, and transfers; consult qualified counsel.
The regulation's data-minimization and security provisions are not a blanket prohibition on using forecasting models.
The EU AI Act has a phased application schedule and role- and use-specific requirements.
A generic demand-forecasting model is not automatically high-risk merely because it is called a foundation model.
Check current legal text and applicable dates with counsel rather than relying on a vendor label.
These statements describe the framework as of November 2026 and are educational, not legal advice.

### Model rights and reproducibility

A downloadable checkpoint may have different terms from the code repository or training data.
Verify commercial-use rights, redistribution conditions, attribution, restrictions, and third-party dependencies for the exact version.
A public model card can describe intended use and limitations but is not necessarily an audited warranty.
An API can change implementation without changing an endpoint name; obtain a pinned version or a change-notice and requalification process.
Track the checkpoint hash or provider version, runtime, preprocessing, context length, prediction settings, and response schema.

Security controls should cover credentials, access logging, encryption, network boundaries, and incident response.
Retain only the data and predictions needed under the applicable policy.
If a model is self-hosted, include dependencies, patching, and hardware access in the risk assessment.
For hosted inference, ask for documented service-level commitments and a tested fallback.
Standards and contracts govern controls and responsibilities; they do not validate forecast accuracy for your portfolio.

## 9. Industry practice and case evidence

### Public evidence is not a local acceptance test

Amazon researchers introduced Chronos as a family of pretrained probabilistic forecasting models.
Their paper describes scaling and quantizing time-series values into tokens and training language-model architectures for forecasting.
It reports zero-shot benchmark results across multiple datasets.
Those are authors' research results, not evidence that a particular distributor will reduce its inventory cost.
The exact model version, benchmark composition, and preprocessing must be checked before comparison.

Google Research's TimesFM work describes a pretrained decoder-style foundation model for time-series forecasting.
The paper evaluates zero-shot forecasts on benchmark datasets and discusses scaling pretraining.
The work supports the feasibility of transferring forecasting models across datasets under reported experimental conditions.
It does not establish that every future promotion, demand shock, or item lifecycle can be inferred from history.
Check the current implementation and model card for supported context, horizon, covariates, license, and hardware.

IBM and collaborators describe Tiny Time Mixers (TTMs) as compact pretrained models for multivariate time-series forecasting.
Their paper examines transfer and fine-tuning across datasets.
The public research evidence addresses benchmark performance under the authors' setup, not a guaranteed customer return.
Model size alone is not total cost: integration, availability, compute, and retraining also matter.

The M5 Forecasting competition used Walmart retail sales data and has a documented hierarchical retail setting.
It is useful for evaluating demand forecasts in a recognizable retail structure.
It is also a public dataset, so inclusion in pretraining or tuning corpora is a meaningful contamination concern.
Strong results on a public competition dataset alone do not establish performance on an unseen retailer's future decisions.
Check provenance and use private or prospective evidence where overlap cannot be ruled out.

### A procurement evidence register

| Claim | Evidence to request | Decision implication |
|---|---|---|
| Zero-shot on new series | Exact checkpoint, protocol, and whether weights or calibration are updated | Separate frozen transfer from local fitting. |
| Better than incumbents | Paired results on the same origins, features, horizons, and cases | Require your own baselines and segments. |
| Trained on a vast corpus | Data sources, dates, sampling, permissions, and exclusions | Volume is not independence, relevance, or rights. |
| Probabilistic output | Distribution format, calibration, samples, and evaluation method | Verify lead-time coverage and decision use. |
| Fast inference | Batch size, hardware, context, horizon, queueing, and percentile latency | Measure your operational request mix and failures. |
| Low cost | Unit price, minimums, retries, egress, support, and annual volume | Compute end-to-end cost per successful forecast. |
| Secure and private | Contract, retention settings, subprocessors, controls, and incident process | Obtain security and legal review before data transfer. |
| Fine-tunes efficiently | Training recipe, validation, resource use, and comparison with frozen model | Count training and ongoing maintenance in incremental value. |
| Handles promotions | Exact accepted features and when those values must be known | Test only information available at the forecast origin. |
| Generalizes across domains | Per-domain results and the distribution of failures | Protect high-cost local segments with separate gates. |

A model card, paper, and vendor demonstration are different evidence types.
A paper can disclose methods and benchmarks but may still reflect author-selected datasets.
A company case study can document that a company used a product without independently verifying causal savings.
A vendor's promotional estimate should be labeled as a vendor claim until its assumptions and comparison are reproduced.
A prospective local trial can reduce uncertainty about your deployment but remains conditional on its trial period and policy.

### Cost-per-forecast definition

Define one forecast as one series-origin request covering a specified horizon and output type.
If the provider bills by token, time step, or compute second, translate the invoice using the actual request distribution.
Include unsuccessful requests, retries, data preparation, storage, monitoring, and fallback execution.
Distinguish marginal serving cost from fixed annual platform cost and one-time integration expense.
Compare alternatives over the same volume, service level, and time horizon.
A cheaper request can be more expensive overall if long contexts trigger large bills or frequent manual intervention.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Future feature leakage | Backtest uses final promotion calendar | Revisions entered after origin are treated as known | Inflated test accuracy | Use as-of snapshots and availability timestamps |
| Pretraining contamination | Public test series appear in corpus list | Model may have encountered evaluation values | Misleading transfer claim | Record provenance and add private prospective tests |
| Silent adaptation | Customer-specific quality improves after repeated calls | Provider changes weights or state | Protocol no longer zero-shot | Obtain written behavior and separate adaptation arms |
| Sales censoring | Forecast errors spike during stockouts | Observed sales truncate demand | Understated shortage risk | Use availability flags and bound latent demand |
| Missing versus zero | Many identical zero runs after feed outage | Missing records are filled as zero | Artificial demand collapse | Keep missingness distinct and test fallback |
| Frequency mismatch | Seasonality appears at the wrong period | Weekly and daily buckets are misinterpreted | Bad forecasts and distorted lead times | Enforce explicit frequency and calendar mapping |
| Scale leakage | Backtest scaler uses full series maximum | Future observations affect context transform | Optimistic scores | Fit scaling only from data available at origin |
| Unstable item mapping | Large jumps coincide with SKU remaps | Changed keys create false histories | Invalid transfer and segment metrics | Use effective-dated mappings and audit joins |
| Inadequate horizon | Only next-week scores are reported | Decision exposure is four weeks | Good metric does not improve orders | Evaluate the full lead-time horizon |
| Aggregate masking | Portfolio WAPE falls but service complaints rise | High-volume items dominate the score | Critical small segments fail | Set segment and service guardrails |
| Calibration failure | Nominal intervals miss far more than expected | Predictive uncertainty is too narrow | Insufficient safety stock | Check coverage and width by horizon and segment |
| Quantile addition | Sum of weekly p90s used as lead-time p90 | Weekly demand dependence is ignored | Wrong order quantity | Use joint paths or a validated aggregate-demand model |
| Overfit fine-tuning | Training loss falls while later origins worsen | Local windows memorize noise | Negative transfer and unnecessary compute | Use time-separated validation and retain frozen baseline |
| Different feature rights | Candidate gets future price unavailable to baseline | Unequal information creates an unfair comparison | Wrong model selection | Use common origin-safe inputs or report distinct scenarios |
| Survivorship bias | Only currently active products are scored | Discontinued or failed products are excluded | Overstated historical performance | Reconstruct the historical eligible population |
| Dropping API failures | Only successful requests enter scoring | Difficult cases disappear from evaluation | Optimistic accuracy and cost | Score fallback forecasts on all eligible cases |
| Vendor version drift | Same endpoint produces different outputs | Unannounced checkpoint or preprocessing changes | Loss of reproducibility | Pin version or require notice and requalification |
| Privacy breach | Sensitive order fields sent in requests | Data-sharing controls were not reviewed | Contractual or regulatory exposure | Minimize fields and approve provider terms |
| Price shock | Invoice exceeds pilot estimate | Retries or longer contexts are billed | Negative economics | Forecast usage distribution and cap spend |
| Unrealized savings | Error falls but inventory policy is unchanged | Forecast has no causal path to action | Business case does not materialize | Simulate and pilot a feasible policy change |

Treat these warnings as triggers for investigation, not automatic proof of a specific cause.
Keep evidence at the level of the individual series and origin so a portfolio summary can be decomposed.
For each incident, preserve the model version, input snapshot, output, fallback, and business impact.
Do not silently replace difficult cases with a better-looking subset after seeing results.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Target approximates the quantity needed for planning | The forecast must relate to the decision | Stockouts censor observed sales | Compare sales with availability and lost-sales estimates |
| Historical origins represent future operating regimes | A backtest estimates future performance | New channel, competitor, or disruption changes demand | Evaluate across regimes and run a prospective pilot |
| Information is reconstructed as of each origin | Comparisons require a feasible historical decision | Revised price or sales data are used prematurely | Audit availability timestamps and source snapshots |
| Provider behavior is stable and documented | A checkpoint comparison must be reproducible | Endpoint changes or hidden adaptation | Pin versions and replay a fixed control set |
| Test data were not used in model selection | Holdout measures generalization after development | Repeated test-driven tuning | Lock a later holdout and record access |
| Pretraining overlap is known or bounded | A public benchmark may have been seen | Training sources are opaque | Review corpus disclosures and add private or forward evaluation |
| Metrics use a common population | A paired comparison isolates candidate differences | Requests fail and hard cases are dropped | Count eligible, predicted, failed, and fallback rows |
| Demand and supply inputs support the policy simulation | Value depends on operational decisions | Lead times, costs, and receipts are inaccurate | Backtest simulated inventory against observed movements |
| Savings do not overlap | Cost categories must be additive | Shortage margin and lost-sales benefit count the same loss | Map each benefit to a distinct accounting mechanism |
| Implementation costs and volume are representative | Economic evaluation needs full ownership costs | Pilot usage excludes peak requests or support | Price production-like loads and sensitivity ranges |
| Quantiles are calibrated for the exposure | Inventory decision relies on tail probabilities | Marginal outputs do not capture lead-time dependence | Evaluate aggregate-demand samples and empirical coverage |
| The organization can change decisions | Prediction alone cannot create operating benefit | Orders remain fixed or constraints bind | Trace forecast to policy, approval, and executed order |

The fictional numerical case assumes no discounting, constant annual benefit, and a three-year evaluation period.
Real cash flows can be delayed, taxed, discounted, or vary by season; use the organization's finance method.
Neither point accuracy nor interval coverage establishes causality for realized savings.
Treat estimated value as conditional on data quality, policy implementation, and the counterfactual.

## 12. Exercises

Complete each exercise with a short, checkable artifact.
Use the definitions and rules in this unit; do not assume a particular vendor supports a feature.

### Exercise 1 — Classify adaptation

**Time:** 10 minutes.  
**Deliverable:** A four-row classification table.  
**Acceptance criteria:** State whether model weights change and whether local fitting occurs.

A. A fixed checkpoint receives the most recent 52 weeks of sales.  
B. The checkpoint is updated on earlier local training windows.  
C. A fixed checkpoint is followed by a bias correction fitted to local errors.  
D. A local seasonal-naive rule repeats the same week last year.

### Exercise 2 — Set the information boundary

**Time:** 15 minutes.  
**Deliverable:** An as-of feature table for a Monday forecast.  
**Acceptance criteria:** Identify availability time, allowed use, and a reason for each feature.

Consider last week's finalized sales received Sunday, next week's proposed promotion approved Wednesday,
a calendar holiday known before Monday, and next week's realized competitor price obtained afterward.
Describe a safe alternative for each prohibited feature.

### Exercise 3 — Compute the score

**Time:** 15 minutes.  
**Deliverable:** Baseline and candidate WAPE, relative error reduction, and signed bias.  
**Acceptance criteria:** Show denominators, the sign convention, and a limitation of the sample.

Use the four-week figures in Section 6.
Explain why the candidate's zero aggregate bias does not prove it predicts every week accurately.

### Exercise 4 — Choose an inventory quantile

**Time:** 10 minutes.  
**Deliverable:** The critical fractile and two limiting assumptions.  
**Acceptance criteria:** Use underage cost USD 9/unit and overage cost USD 3/unit;
distinguish the quantile from a guarantee and say which demand exposure it should represent.

### Exercise 5 — Audit a vendor statement

**Time:** 20 minutes.  
**Deliverable:** An evidence request with at least five rows.  
**Acceptance criteria:** Cover pretraining exposure, paired baselines, covariate availability,
uncertainty, and full serving cost; distinguish a claim from a local finding.

The supplier says: “Our zero-shot model beats classical forecasting on all supply chains.”
Do not accept or reject the statement solely because a public benchmark score is high.

### Exercise 6 — Evaluate the business case

**Time:** 15 minutes.  
**Deliverable:** Base net benefit, downside net benefit, break-even realization, and a recommendation.  
**Acceptance criteria:** Use the Section 6 annual and setup assumptions;
do not double count setup or infer simulation savings from WAPE alone.

### Exercise 7 — Design fine-tuning

**Time:** 25 minutes.  
**Deliverable:** A one-page protocol with training, development, and test boundaries.  
**Acceptance criteria:** Define windows, checkpoint selection, same-feature baselines,
contamination risk, segment metrics, and a rule to reject a fine-tuned model.

Include how the protocol changes for a new item with no local training labels.
Make the prospective date and allowed model release explicit.

### Exercise 8 — Design a controlled release

**Time:** 25 minutes.  
**Deliverable:** A shadow-to-pilot release plan and a rollback checklist.  
**Acceptance criteria:** Name a decision owner, fallback, service and cost gates,
forecast versioning, human approval, and a method for measuring realized impact.

State how the team handles an unavailable API and an unexpected vendor model update.
Explain how you avoid confusing a forecast, plan, target, and supplier commitment.

## 13. Self-check answers

### 1. Classification

A is zero-shot under the stated protocol: local history is context and the weights stay frozen.
B is fine-tuning because the checkpoint's parameters are updated using local windows.
C is frozen base-model inference plus locally fitted calibration; report both components.
D is a local baseline, not a pretrained foundation model.
A vendor-managed update that is not visible to the user needs disclosure before it can be classified confidently.

### 2. Information boundary

The finalized sales received Sunday are allowed if the Monday origin comes after their receipt.
The holiday calendar is allowed if its value was known at that origin.
A promotion approved Wednesday is not a known-future fact on Monday.
Use the Monday proposal with a clear scenario label if it existed then, or omit it.
The realized future competitor price is forbidden; omit it or use an origin-safe price forecast with its own uncertainty.
Keep the availability timestamp as well as the event date for each field.

### 3. Forecast arithmetic

The actual denominator is `112 + 97 + 134 + 107 = 450 units`.
The baseline absolute error is `7 + 13 + 13 + 8 = 41 units`, so WAPE is 9.11%.
The candidate absolute error is `3 + 4 + 5 + 4 = 16 units`, so WAPE is 3.56%.
Relative reduction is `25 / 41 = 60.98%`.
Using forecast minus actual, the baseline bias is `1 / 450 = +0.22%` and candidate bias is `0 / 450 = 0%`.
The positive and negative candidate errors cancel; four weeks and one origin cannot demonstrate generalization.

### 4. Quantile

The fraction is `9 / (9 + 3) = 0.75`, so select the 75th percentile under the simple model.
The distribution must describe demand during the decision's exposure, not merely a convenient one-week marginal.
The costs must represent the relevant incremental shortage and excess consequences.
The percentile is a probabilistic statement conditional on the model; it is not a service-level guarantee.
Capacity, order multiples, multiple periods, and dependent demand can require a different optimization.

### 5. Vendor audit rubric

A strong answer asks for the exact checkpoint and a description of pretraining sources and overlap.
It specifies identical held-out origins, horizons, inputs, population, and baselines for a paired comparison.
It asks what the model can actually ingest and whether future features were known at the decision time.
It asks for calibrated output evidence, failure handling, and all cost components.
It explicitly identifies the public benchmark as external evidence rather than a verified local result.

### 6. Economics

Gross benefit is `82,000 + 28,000 = USD 110,000/year`.
Recurring expense is `26,000 + 18,000 = USD 44,000/year`.
Annualized setup is `60,000 / 3 = USD 20,000/year`.
Base net is USD 46,000/year; at half benefit, net is negative USD 9,000/year.
Break-even is USD 64,000/year or 58.18% of the assumed gross benefit.
Recommend a controlled trial with finance review because estimated realization determines the sign of value.

### 7. Fine-tuning rubric

A strong protocol separates earlier training windows, later development origins, and a final untouched test.
All label ends must precede their validation or test origins, and all features must be available then.
It compares fine-tuned, frozen, and local baseline candidates on identical cases and feature rights.
It checks segment and horizon errors, failure rates, training cost, and possible pretraining overlap.
Reject fine-tuning if its protected test or decision economics fail agreed gates.
For an item with no local labels, use the frozen model or a documented baseline and do not claim item-specific fine-tuning.

### 8. Release rubric

A strong plan starts in shadow mode, stores immutable forecasts, and compares them after actuals mature.
A business owner approves a bounded pilot only if service, accuracy, cost, and data-control gates pass.
The existing model supplies the fallback when the API fails or the new version has not been qualified.
A changed checkpoint triggers a fresh comparison and approval; rollback restores the previous approved path.
A controlled comparison of decisions and outcomes, with confounders recorded, supports impact estimation.
The forecast informs a plan; targets and external commitments require their own authorized decisions.

## 14. Interview questions

### 1. What is zero-shot forecasting?

**Strong answer:** A frozen pretrained checkpoint predicts a target series without target-task weight updates; local context can still be present.

**Weak answer:** It claims that zero-shot requires no historical observations or that all preprocessing is automatically training.

### 2. When is fine-tuning justified?

**Strong answer:** Local training and time-separated evaluation beat the frozen alternative after compute, maintenance, and decision costs.

**Weak answer:** It assumes local updates always improve results.

### 3. How would you compare a vendor model fairly?

**Strong answer:** Use paired origins, horizons, data availability, fallbacks, segments, and a credible incumbent baseline.

**Weak answer:** It compares a vendor benchmark with a different portfolio or ignores failed requests.

### 4. How can pretraining contaminate a backtest?

**Strong answer:** The public test series or future values may occur in the source corpus; document exposure and use private or prospective evidence.

**Weak answer:** It asserts contamination from a good score alone or assumes unknown provenance proves independence.

### 5. Why can sales understate demand?

**Strong answer:** Stockouts and constrained availability censor observed sales; use availability signals and acknowledge uncertainty.

**Weak answer:** It treats every low-sales week as a demand collapse.

### 6. Why report horizon-specific metrics?

**Strong answer:** Errors at replenishment lead time affect decisions differently from errors next week.

**Weak answer:** It relies only on a one-step aggregate score.

### 7. What does a forecast percentile mean?

**Strong answer:** It is a conditional modeled quantile and requires calibration on the relevant exposure.

**Weak answer:** It promises that each actual will fall below the percentile.

### 8. Can weekly p90 forecasts be summed into a lead-time p90?

**Strong answer:** Not generally; dependence matters, so evaluate joint paths or an aggregate distribution.

**Weak answer:** It adds marginal quantiles without qualification.

### 9. What belongs in total cost?

**Strong answer:** Include inference, retries, operations, integration, training, security, support, and fallback over a common horizon.

**Weak answer:** It compares only an advertised price per call.

### 10. How do you establish business value?

**Strong answer:** Replay or test a feasible policy against a stated counterfactual and measure service and incremental net costs.

**Weak answer:** It multiplies a percentage-point accuracy gain by total revenue.

## 15. Key terms

The following index points to the fuller definitions and procedure above.

- **Time series:** values indexed by ordered time buckets, such as units/week; see [definitions](#4-definitions-and-vocabulary).
- **Target:** quantity to predict, such as sales units per SKU-location-week; see [definitions](#4-definitions-and-vocabulary).
- **Series identifier:** stable key for item and location; see [definitions](#4-definitions-and-vocabulary).
- **Frequency:** duration of one bucket, such as one week; see [definitions](#4-definitions-and-vocabulary).
- **Context:** historical values and allowed features supplied at inference; see [definitions](#4-definitions-and-vocabulary).
- **Horizon:** number of future buckets predicted from an origin; see [definitions](#4-definitions-and-vocabulary).
- **Origin:** timestamp at which a forecast is issued; see [definitions](#4-definitions-and-vocabulary).
- **Pretraining:** parameter learning on a broad corpus before target-task deployment; see [definitions](#4-definitions-and-vocabulary).
- **Checkpoint:** versioned saved set of model weights and configuration; see [definitions](#4-definitions-and-vocabulary).
- **Zero-shot:** prediction without target-task parameter updates; see [definitions](#4-definitions-and-vocabulary).
- **Fine-tuning:** additional parameter updates on local or related training data; see [definitions](#4-definitions-and-vocabulary).
- **Frozen model:** model whose pretrained parameters are not updated; see [definitions](#4-definitions-and-vocabulary).
- **Global model:** one set of parameters shared across multiple series; see [definitions](#4-definitions-and-vocabulary).
- **Local model:** model estimated separately for a single series; see [definitions](#4-definitions-and-vocabulary).
- **Transfer:** use of knowledge learned on a source distribution for a target; see [definitions](#4-definitions-and-vocabulary).
- **Negative transfer:** degradation caused by transferred patterns; see [definitions](#4-definitions-and-vocabulary).
- **Domain shift:** difference between training and operating data-generating conditions; see [definitions](#4-definitions-and-vocabulary).
- **Covariate:** a feature other than the target, with an explicit availability time; see [definitions](#4-definitions-and-vocabulary).
- **Known-future feature:** input genuinely available for each forecasted bucket at the origin; see [definitions](#4-definitions-and-vocabulary).
- **Static feature:** attribute treated as constant for an example, such as product category; see [definitions](#4-definitions-and-vocabulary).
- **Missing observation:** value not observed or not delivered by the cutoff; see [definitions](#4-definitions-and-vocabulary).
- **Censoring:** observed sales limited by availability rather than demand; see [definitions](#4-definitions-and-vocabulary).
- **Intermittent demand:** many periods without observed demand and occasional positive demand; see [definitions](#4-definitions-and-vocabulary).
- **Scaling:** transformation such as dividing a series by a historical magnitude; see [definitions](#4-definitions-and-vocabulary).
- **Rolling origin:** repeated forecasts at historical decision timestamps; see [definitions](#4-definitions-and-vocabulary).
- **Holdout:** data not used for model or policy selection; see [definitions](#4-definitions-and-vocabulary).
- **Leakage:** use of information unavailable or impermissible at a forecast origin; see [definitions](#4-definitions-and-vocabulary).
- **Contamination:** evaluation examples or close derivatives exposed during pretraining; see [definitions](#4-definitions-and-vocabulary).
- **MAE:** mean absolute error, in target units per bucket; see [definitions](#4-definitions-and-vocabulary).
- **WAPE:** sum of absolute error divided by sum of actual demand, dimensionless; see [definitions](#4-definitions-and-vocabulary).
- **Bias:** signed forecast error summary with an explicit sign convention; see [definitions](#4-definitions-and-vocabulary).
- **Quantile:** value qτ with modeled probability τ of demand at or below it; see [definitions](#4-definitions-and-vocabulary).
- **Coverage:** fraction of observed values inside a stated prediction interval; see [definitions](#4-definitions-and-vocabulary).
- **Lead time:** time between an order decision and receipt, measured in buckets; see [definitions](#4-definitions-and-vocabulary).
- **Incremental value:** difference in business outcome versus a stated alternative; see [definitions](#4-definitions-and-vocabulary).
- **Total cost of ownership:** recurring and one-off costs of an operating solution; see [definitions](#4-definitions-and-vocabulary).

## 16. Sources

Sources below identify the original evidence, not an endorsement of a model or vendor.
Product capabilities and terms change; consult the exact release and agreement used in an experiment.

1. Ansari, A. F., et al. “Chronos: Learning the Language of Time Series.”
   *Transactions on Machine Learning Research* (2024).
   [Paper and open review](https://openreview.net/forum?id=gerNCVqqtR).
   Describes tokenization, pretraining, probabilistic forecasting, and reported benchmark evaluations.
2. Das, A., et al. “A Decoder-Only Foundation Model for Time-Series Forecasting.”
   *International Conference on Machine Learning* (2024).
   [Proceedings of Machine Learning Research](https://proceedings.mlr.press/v235/das24c.html).
   Describes TimesFM and the authors' zero-shot evaluation.
3. Ekambaram, V., et al. “Tiny Time Mixers (TTMs): Fast Pre-trained Models for Enhanced Zero/Few-Shot Forecasting of Multivariate Time Series.”
   [arXiv preprint, 2024](https://arxiv.org/abs/2401.03955).
   A research report on compact pretrained multivariate forecasting and adaptation; not a customer case study.
4. Makridakis, S., et al. “The M5 Accuracy Competition: Results, Findings, and Conclusions.”
   *International Journal of Forecasting* 38(4), 2022.
   [DOI: 10.1016/j.ijforecast.2021.11.013](https://doi.org/10.1016/j.ijforecast.2021.11.013).
   Describes the retail forecasting competition and its findings.
5. Gneiting, T., and Raftery, A. E. “Strictly Proper Scoring Rules, Prediction, and Estimation.”
   *Journal of the American Statistical Association* 102(477), 2007.
   [DOI: 10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437).
   Gives the statistical basis for scoring probabilistic forecasts.
6. GS1. [GS1 General Specifications](https://www.gs1.org/standards/barcodes-epcrfid-id-keys/gs1-general-specifications).
   Official identification and barcode specifications; use the version applicable to the trading relationship.
7. European Parliament and Council. [Regulation (EU) 2016/679, GDPR](https://eur-lex.europa.eu/eli/reg/2016/679/oj).
   Official legal text; applicability and obligations depend on the processing context.
8. European Parliament and Council. [Regulation (EU) 2024/1689, Artificial Intelligence Act](https://eur-lex.europa.eu/eli/reg/2024/1689/oj).
   Official legal text; consult applicable amendments, phase-in dates, and legal advice.
9. NIST. [AI Risk Management Framework 1.0](https://www.nist.gov/itl/ai-risk-management-framework).
   Voluntary risk-management guidance, not a certification of a model or a legal safe harbor.

## 17. Links

- [Module 18 index](_index.md): topic order, scope, and assessment context.
- [Neural network fundamentals](01_neural-network-fundamentals.md): weights, objectives, and regularization.
- [Global versus local models](03_global-local-models.md): cross-learning and local trade-offs.
- [Neural forecasting zoo](04_neural-forecasting-zoo.md): candidate architecture families.
- [Module 18 survey](../../../course/18_deep_learning_foundation_models_llm_agents.md): broader narrative and lab context.
- [Simulation and optimization labs](../../05_labs_electives/30_simulation_optimization_production_labs/_index.md): operational evaluation context.
