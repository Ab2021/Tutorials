# The Neural Forecasting Zoo

> Module 18 — Deep Learning, Foundation Models, and LLM Agents  
> Topic 04 of 10  
> Estimated study time: 4 hours  
> Prerequisites: basic arithmetic and familiarity with a demand-planning decision; necessary machine-learning and forecasting concepts are explained here.  
> Version 2026.5

## 1. Why this topic exists

A replenishment manager needs a forecast before placing an order.
The order covers demand during supplier lead time and the interval until the next review.
If the forecast is too low, the business may lose sales or pay for an emergency shipment.
If it is too high, stock may occupy cash and warehouse space or expire.
A model can improve a forecast score and still make the order worse.
For example, a better average forecast is not necessarily a better estimate of the upper tail needed for a service-level decision.

The model names DeepAR, N-BEATS, N-HiTS, Temporal Fusion Transformer, and PatchTST appear in forecasting proposals.
They describe different ways to learn patterns from historical observations.
They do not describe a universal ranking of models.
The right shortlist depends on the forecast horizon, available histories, covariates, probability requirements, latency, and the cost of errors.

A grocery planner might require a daily distribution over demand for the next two weeks.
A manufacturer might need a monthly point forecast for components over a year.
A distribution network might need coherent aggregate scenarios for transport capacity.
Each decision imposes a different requirement on the forecast product.
A forecast is an estimate of what will happen under stated information and assumptions.
A plan is a chosen response to that estimate.
A target is an aspiration or performance objective.
A commitment is an agreed obligation, such as a purchase order accepted by a supplier.
Do not substitute one for another in training data or evaluation.

Neural networks are useful candidates when they can learn repeated patterns across many related histories.
They can also be expensive to tune, difficult to operate, and misleading when evaluated with future information.
The first choice is not which neural architecture is fashionable.
It is whether a more complex forecast can improve a consequential decision relative to a simple, reproducible alternative.
This topic gives you a decision process, not a promise that any one architecture wins.

## 2. Learning outcomes

After studying this topic, you can:

1. **Define** a forecast origin, horizon, target, loss, and information set; exercise 1 checks these definitions.
2. **Compare** the five architectures by information flow, native output, and operational trade-offs; exercise 2 checks the comparison.
3. **Construct** a rolling-origin validation design with as-of covariates and an untouched test period; exercise 3 checks the design.
4. **Calculate** point error, quantile loss, empirical coverage, and a decision cost; exercises 4 and 5 check the arithmetic.
5. **Select** candidates for a stated demand segment without treating a paper's benchmark result as a guarantee; exercise 6 checks the choice.
6. **Estimate** annual net value, including forecast, implementation, and operating costs; exercise 7 checks the economics.
7. **Critique** a deployment proposal for leakage, uncertainty, and lifecycle risks; exercise 8 checks the critique.
8. **Design** a controlled operational pilot with stop rules and monitoring; exercise 9 checks the design.

## 3. The core mental model

```text
Decision and cost of errors
        |
        v
Demand target + forecast origin + horizon + information available then
        |
        v
Simple baselines ----> candidate families ----> locked rolling-origin comparison
        |                    |                          |
        |                    v                          v
        |             model fit on past          point, distribution, segment,
        |                    |                   latency, and decision metrics
        +--------------------+--------------------------+
                                                        |
                                                        v
                                        simulate the same decision policy
                                                        |
                                                        v
                                          controlled pilot or keep baseline
                                                        |
                                                        v
                                      monitor, retrain, rollback, and retire
```

A time series is an ordered sequence of values, such as weekly units sold for one item at one location.
A forecasting example contains a lookback window of observations before an origin and a future horizon of values to predict.
A model learns a mapping from the lookback and permitted explanatory variables to future forecasts.
A parameter is a number learned during training; weights in a neural network are parameters.
A hyperparameter, such as lookback length or network width, is selected by the practitioner rather than directly learned by the usual training step.

A neural network composes simple numerical transformations.
Training changes its parameters to reduce a loss function on examples made from historical data.
A loss is a numerical penalty for a forecast and its realized outcome.
Gradient-based optimization computes how a small change in each parameter affects loss and updates the parameters repeatedly.
Training loss is not proof of future performance: a flexible model can memorize historical peculiarities.
Validation estimates performance on observations not used to fit those parameters.
A final test estimates performance on a later period not used for model or hyperparameter selection.

A local model is fitted separately to each series.
A global model shares parameters across multiple series, optionally with item or location features.
Global learning can help sparse series borrow information from related series.
It can also transfer the wrong pattern across dissimilar products.
The five architectures can be used in different training arrangements; an architecture name alone does not establish the production implementation's pooling strategy.

A point forecast is one number per item, location, and horizon.
A probabilistic forecast describes uncertainty, for example through quantiles, a parametric distribution, or sample paths.
A quantile is a threshold such that a specified fraction of outcomes should fall at or below it under the predictive distribution.
A 90th percentile is not a 90% prediction interval.
An interval needs two bounds, such as the 5th and 95th percentiles for a nominal 90% interval.
Calibration asks whether stated probabilities match observed frequencies over comparable cases.
Sharpness asks whether calibrated forecasts are concentrated enough to be useful.

The diagram deliberately places the business decision first and the model family in the middle.
It places validation before operational adoption.
A neural model can be a candidate, not an exemption from a baseline or an information cutoff.

## 4. Definitions and vocabulary

| Term | Meaning and unit | Common misuse |
|---|---|---|
| Series | Ordered measurements for one entity; for example units per SKU-location-week | Treating different product definitions as one stable history |
| SKU | Stock keeping unit, an operational product identifier | Assuming an identifier is always stable across pack changes |
| Target | Quantity to predict, such as unconstrained unit demand per week | Treating a sales target as a realized demand observation |
| Forecast origin | Time at which a forecast is issued | Using data recorded only after that instant |
| Horizon | Number of future periods forecast from an origin | Mixing one-week and 13-week errors without reporting the mix |
| Lead time | Time from an order to its receipt, in days or weeks | Evaluating a horizon unrelated to replenishment |
| Review period | Time between ordering decisions | Assuming lead time alone defines protection exposure |
| Lookback | Past observations supplied to a model, in periods | Assuming more history always helps |
| Context window | Input span available to a forecasting model | Confusing a long window with usable signal |
| Covariate | A variable supplied alongside the target, such as a calendar indicator | Treating all recorded future values as known at the origin |
| Static feature | Attribute held constant within the modeled series or version, such as a product family | Using a current attribute retroactively without an as-of version |
| Observed-past feature | Variable observed through the origin, such as recent store traffic | Filling future steps with the subsequently realized values |
| Known-future feature | Variable genuinely available for future periods at the origin, such as the published calendar | Treating a later revised promotion plan as originally known |
| Data vintage | Snapshot of values as available at a specified time | Evaluating with the latest corrected version of every input |
| Censoring | Observation restricted by availability; sales may be capped by stock | Assuming zero sales prove zero demand |
| Leakage | Use of information unavailable when a forecast would have been issued | Assuming time-ordered splits alone eliminate every leak |
| Baseline | A simple, reproducible comparator, such as seasonal naive | Comparing only sophisticated models with each other |
| Seasonal naive | Forecast repeating the value from a prior corresponding season | Applying a seasonal lag longer than the available history without a rule |
| Global model | Model sharing parameters across series | Assuming all products are identical |
| Local model | Separate fit for each series | Assuming a local model cannot use external variables |
| Training | Parameter fitting on historical examples | Using the test period to fit weights or scalers |
| Validation | Data used for selection and tuning | Reporting it as an untouched final test |
| Test | Held-out data used for final assessment after choices are frozen | Repeatedly trying variants against it |
| Rolling origin | Sequence of historical forecast issuance dates | Training past each origin before making its forecast |
| Refit | Re-estimation of learned parameters at a scheduled time | Silently refitting one candidate more often than another |
| Retraining cadence | Interval between model fits, in days or weeks | Confusing it with the frequency of forecast issuance |
| Loss function | Numerical training penalty in units determined by the loss | Assuming the training loss is identical to business cost |
| MAE | Mean absolute error, in target units | Averaging across incomparable scales without segmentation |
| WAPE | Sum of absolute errors divided by sum of absolute actuals, dimensionless | Using it when the denominator is zero or calling it an unbiased measure |
| Bias | Signed mean forecast minus actual, in target units here | Assuming small aggregate bias implies small item-level errors |
| Quantile | Value below which a specified probability of outcomes lies | Calling a single upper quantile an interval |
| Pinball loss | Asymmetric quantile error penalty, in target units | Evaluating a quantile with MAE alone |
| Coverage | Fraction of outcomes inside an interval, dimensionless | Treating nominal coverage as sufficient without interval width |
| Sample path | One coherent sequence of sampled future values | Summing independently sampled marginal quantiles as if they were a joint path |
| Autoregression | Predicting using previous target values, including generated values at future steps | Ignoring propagation of early simulation errors |
| Likelihood | Probability model or density assigned to observations under parameters | Assuming a distributional form fits all demand types |
| Residual | Observed value minus fitted value under a specified model | Treating fitted in-sample residuals as a leakage-free test |
| Backcast | Model output representing a fitted component of an input history | Calling it a forecast for future periods |
| Attention | Data-dependent weighting of representations inside a model | Interpreting every attention weight as a causal effect |
| Patch | Contiguous block of time points represented together as a token | Assuming patching preserves every short-lived signal |
| Embedding | Learned numerical representation of a category or input | Assuming nearby embeddings prove a business relationship |
| Reconciliation | Adjustment to make forecasts add consistently across a hierarchy | Assuming individual quantiles can simply be summed |
| Service level | Explicit operational target, for example fraction of cycles without stockout | Equating it automatically with fill rate |
| Fill rate | Fraction of demand supplied from available inventory, under a stated definition | Calling it the probability of no stockout |
| Decision cost | Monetary consequence under an explicit policy and scenario | Counting gross savings as net value |

## 5. The mechanics, step by step

### 5.1 Start with a decision contract

1. Name the decision owner and the action: for example, a planner places a weekly order for each SKU-location.
2. Specify the decision date, order cutoff, replenishment lead time, and review interval.
3. Define the target in units and buckets: for example, unconstrained demand in units per SKU-location-week.
4. Define the required horizon: for a four-week lead time and weekly review, examine at least the five-week protection period if that is the policy's exposure.
5. State whether the decision needs a point forecast, a percentile, or joint scenarios.
6. Identify how forecast outputs enter the order rule, including minimum order quantities and capacity restrictions.
7. Record the incumbent forecast and current decision process before changing anything.
8. Define failure consequences, data access, and a minimum decision-value improvement worth the implementation effort.

Input: a decision process, not a preferred model name.
Output: a written contract that defines target, timing, horizon, action, and success criteria.
Branch: if no one can say how a forecast will change an action, first establish a decision use case.

An order-up-to policy may compare inventory position with a protection-period demand estimate.
Inventory position usually means on-hand plus on-order minus backorders, with organization-specific definitions documented.
A policy using an upper quantile is sensitive to the distribution of cumulative demand, not merely average demand in each period.
A model that supplies only a mean needs a separately validated uncertainty method before that policy can use it.

### 5.2 Build an as-of dataset

1. Identify source systems and the time each field became available, not just the time it describes.
2. Join transactions to item and location identifiers valid at each historical origin.
3. Preserve the event date, publication date, ingestion date, and correction date where available.
4. Record unit-of-measure conversions and pack changes as time-varying mappings.
5. Construct a complete calendar of active SKU-location-period combinations.
6. Separate a true zero observation from a missing feed, inactive item, and unavailable inventory.
7. Flag store closures, stockouts, extraordinary events, product launches, and delistings.
8. Use only observations that would have been accessible at each origin for model inputs.
9. Save the data extract and feature versions used for every validation run.

Input: transaction, availability, item, location, and calendar histories.
Output: a reproducible panel with target definition, availability flags, and origin-specific features.
Branch: if unconstrained demand cannot be estimated credibly, forecast observed sales and disclose the resulting limitation rather than call sales demand.

Sales during stockouts are censored because customers cannot buy inventory that is not present.
A model trained on such sales may learn that low availability predicts low future demand.
An availability flag can help diagnose this problem but does not by itself reconstruct lost demand.
Causal correction requires additional assumptions or evidence about customers who could not buy.
For intermittent demand, report the fraction of zero periods and assess occurrence and size separately where useful.

### 5.3 Separate feature availability classes

Static attributes can include product family or region, provided their historical versions are recoverable.
Observed-past features can include sales through the last completed week or measured weather through the origin.
Known-future features can include day of week or a promotion plan that was approved and archived before the origin.
Future realized weather, future actual price, and future stockout outcomes are not known-future features.
A published promotion plan is known as a plan, not as a guarantee that the promotion will execute.

For each feature, create an availability contract:

- **Name and units:** promotion depth as percentage of reference price.
- **Owner:** commercial planning team.
- **Earliest availability:** approved plan published before the weekly forecast cutoff.
- **Revision rule:** keep the snapshot available at each origin.
- **Missing rule:** use an explicit missing category or the documented baseline plan.
- **Scenario rule:** alternative future promotions are inputs to separate conditional forecasts.
- **Leakage test:** verify publication timestamp is not later than the forecast cutoff.

Input: candidate features and their vintages.
Output: an allowlist for every origin and horizon.
Branch: exclude a feature when its historical as-of value cannot be recovered, unless the evaluation is explicitly labeled as an optimistic upper bound.

### 5.4 Establish a fair baseline

A last-value forecast repeats the most recent observed value.
A seasonal-naive forecast repeats the value from the same position in a prior season, if that season exists.
A moving average uses the mean of a specified number of past periods.
For irregular and intermittent series, include an appropriate intermittent-demand benchmark where applicable.
A baseline can also be the production forecast, reconstructed from its historical issue-time snapshots.

Choose the baseline before inspecting the final test period.
Give all methods the same target, origins, horizon, and data-availability rules.
Do not make the incumbent deliberately stale while allowing a neural model frequent refits.
If the neural model uses promotion plans, add a reasonable feature-aware non-neural competitor where feasible.
When a model uses more information than another, report that difference as part of the result.

Input: data panel and incumbent process.
Output: a baseline forecast table indexed by item, location, origin, horizon, and forecast type.
Branch: if a baseline cannot meet runtime or data constraints, document the reason rather than silently dropping it.

### 5.5 Understand DeepAR

DeepAR is a probabilistic autoregressive recurrent neural network introduced for forecasting related time series.
A recurrent network updates a hidden state as it reads a sequence one period at a time.
The state summarizes past inputs for the next step; it is learned, not a human-readable inventory state.
DeepAR shares parameters across series in the original global-model setup.
The model can use time-dependent covariates and series-specific scale information.

At a future step, the network predicts parameters of a chosen likelihood distribution.
A likelihood might be appropriate for nonnegative counts or continuous values, depending on the implementation and data.
For a multi-step sample path, draw a value from the predicted distribution and feed that generated value into the next step.
Repeat this process to obtain many paths and estimate percentiles or cumulative-demand distributions.
Forecast errors can propagate through the recursive path; uncertainty depends on the likelihood and learned dynamics.

Consider DeepAR when many related series exist, probabilistic paths matter, and available covariates are useful.
Check whether the implementation supports the desired distribution, static identifiers, missingness, and feature availability.
Sampling can increase inference time and cost, especially for many series and long horizons.
An inappropriate likelihood can make apparently precise distributions badly calibrated.
DeepAR is not guaranteed to outperform direct multi-horizon alternatives.

Input: histories, permissible covariates, training configuration, and likelihood choice.
Output: distribution parameters and sampled future paths, subject to implementation details.
Branch: if cumulative lead-time uncertainty is essential, evaluate simulated sums and their coverage, not just individual-week quantiles.

### 5.6 Understand N-BEATS

N-BEATS is a deep neural architecture built from stacks of fully connected blocks for univariate point forecasting.
A fully connected layer applies learned weights to all inputs of the layer, followed by a nonlinear transformation.
Each block produces a backcast for the history and a forecast contribution for the future.
The backcast is subtracted from the input residual passed to the next block.
The block forecasts are added to form the final forecast.

The original work includes a generic configuration and an interpretable configuration with trend and seasonal basis functions.
A basis is a family of predefined shapes that combines with learned coefficients.
A trend or seasonal component is a model decomposition, not proof of a causal mechanism.
In its original basic form, N-BEATS is a point forecaster based on target history.
Implementations and extensions may add covariates or probabilistic outputs; verify the exact package rather than assuming those capabilities are intrinsic to every version.

Consider N-BEATS for a direct multi-horizon point forecast with sufficiently informative histories.
It is a useful benchmark when future covariates are weak or unavailable.
Evaluate whether a global implementation shares parameters appropriately across series.
A historical-only model cannot react directly to a newly approved future promotion it has never been given.
If a decision requires quantiles, add and validate a suitable probabilistic method or select an implementation that provides one.

Input: historical target window and configuration.
Output: a direct vector of point forecasts for the requested horizon in the original formulation.
Branch: when feature-rich conditional scenarios are central, compare an implementation that explicitly supports the necessary features rather than inferring support from the family name.

### 5.7 Understand N-HiTS

N-HiTS extends the residual-block forecasting idea with multi-rate input sampling and hierarchical interpolation of forecast components.
The intent is to represent long-horizon patterns at different temporal scales efficiently.
Different blocks can summarize the input at different resolutions.
Their outputs are interpolated to the full horizon and combined.
Interpolation imposes a structure on how a coarse output changes between points; it is not a guarantee that every spike is preserved.

Consider N-HiTS when the horizon is long relative to the input frequency and slow-moving structure is important.
Compare its fit time, memory, and forecast quality on the actual horizon and hardware.
A configuration that smooths heavily may miss short promotions or sudden disruptions.
Like N-BEATS, the original formulation emphasizes univariate point forecasting; feature and distribution support varies by implementation and extension.
Do not assume speed or accuracy improvements reported on a benchmark automatically transfer to a specific supply chain.

Input: historical windows, horizon, and multi-resolution settings.
Output: a direct multi-step forecast built from interpolated components.
Branch: retain a higher-resolution candidate if critical decisions depend on brief peaks that coarse representations fail to capture.

### 5.8 Understand Temporal Fusion Transformer

The Temporal Fusion Transformer, or TFT, is a multi-horizon architecture designed to combine static attributes, observed historical inputs, and known-future inputs.
Its design includes variable selection, gating, recurrent processing for local patterns, and attention for longer temporal relationships.
Gating can control how information passes through parts of the network.
The original formulation commonly produces multiple forecast quantiles using quantile losses.
Exact available features and outputs depend on the software implementation.

TFT is a candidate when a reliable future calendar, published promotion plan, or other genuinely known input changes the conditional forecast.
Its rich input structure makes timestamp discipline especially important.
A model trained with the final promotion schedule but used with an earlier draft receives an unfair advantage in backtesting.
Feature importance and attention visualizations are diagnostics of the fitted model, not estimates of causal promotion effects.
Quantile outputs can cross, meaning a predicted lower percentile exceeds a higher percentile; inspect and address this under a documented rule.
Marginal quantiles do not automatically define coherent joint paths or aggregate correctly across products and weeks.

Input: target history, static features, observed-past features, and origin-valid known-future features.
Output: typically horizon-specific conditional quantiles in the original forecasting setup.
Branch: if future plans are unreliable, compare plan-vintage forecasts and a no-plan baseline; consider scenario uncertainty separately.

### 5.9 Understand PatchTST

PatchTST is a transformer-based time-series architecture that represents consecutive observations as patches.
A transformer uses attention to combine information across representations of the input.
Patching reduces the number of tokens relative to treating each time point as a separate token for the same context.
The original PatchTST design emphasizes channel independence: shared model weights process each series channel separately.
Channel independence is not the same as proving that real demand series are independent.

Patch length and stride determine which local patterns are grouped and how patches overlap.
Longer patches can compress the input and capture local structure, while potentially blurring a brief event.
A long lookback can expose annual seasonality in high-frequency data, but memory and useful sample count remain practical constraints.
The original forecasting benchmark emphasizes point forecasts; probabilistic outputs, covariates, and cross-channel interactions depend on the specific extension or implementation.
A channel-independent setup does not automatically model substitution or correlated supply disruptions between products.

Consider PatchTST for long input windows and repeated temporal patterns when hardware and data support the candidate.
Compare a shorter-window baseline and assess promotional peaks separately.
Check whether the implementation consumes known-future covariates before claiming it uses them.
If inventory aggregation requires joint uncertainty, use an explicit, validated joint scenario method rather than assuming transformer attention supplies one.

Input: historical windows, patch and stride settings, and model configuration.
Output: direct horizon forecasts, with output type determined by the implementation.
Branch: if cross-series relationships drive the decision, evaluate explicit multivariate alternatives or a separately validated dependency model.

### 5.10 Compare candidates on the same contract

| Family | Characteristic mechanism | Original typical output | Candidate circumstance | Principal qualification |
|---|---|---|---|---|
| DeepAR | Global autoregressive recurrent likelihood | Probabilistic sample paths | Related series and cumulative uncertainty | Recursive sampling cost and likelihood fit |
| N-BEATS | Residual backcast and forecast blocks | Direct point vector | History-driven multi-horizon baseline | Native formulation does not ensure covariate or distribution support |
| N-HiTS | Multi-rate blocks and hierarchical interpolation | Direct point vector | Long horizon with multi-scale signal | Coarse structure may miss short events |
| TFT | Gating, variable selection, recurrent processing, attention | Horizon-specific quantiles | Reliable origin-known future features | Many inputs create leakage and operational burden |
| PatchTST | Attention over patches with channel-independent sharing | Direct point vector | Long lookback and repeated patterns | Channel independence and patching limit what is modeled by default |

These are starting hypotheses, not a tournament result.
Packages can modify architectures, output heads, covariate support, and training strategies.
Record package version, parameters, loss, features, compute environment, and seed policy for every candidate.
A model family is not a complete experimental specification.

### 5.11 Choose a rolling-origin evaluation

At each origin, create a training dataset ending at or before the forecast cutoff.
Fit any scaler, imputer, encoder, and feature selection using only the training data available at that origin.
Produce forecasts for the same future horizons for every candidate.
Advance the origin and repeat, using a cadence representative of production.
The training window may expand over time or roll forward with a fixed length; state which and why.
A fixed window may adapt faster after a regime change but discard useful old seasons.
An expanding window retains more observations but can overweight an obsolete regime.

Use several origins spanning ordinary periods, seasonal peaks, and promotion regimes where history permits.
If labels for the longest horizon overlap the final test period, purge or truncate the affected training or validation examples as appropriate.
Overlapping forecast horizons create correlated errors: a single actual can be scored multiple times from different origins.
That may correctly reflect repeated operational decisions, but the sample count is not the number of independent outcomes.
Use origin-level or appropriate block resampling for uncertainty rather than treating all overlapping item-horizon errors as independent.

Input: a timeline, historical data vintages, and operational issuance schedule.
Output: comparable forecast records with origin, horizon, actual, model version, and feature snapshot.
Branch: if only one seasonal cycle exists, report limited evidence rather than claiming stable seasonal generalization.

### 5.12 Separate selection from final testing

Reserve a final contiguous later period before experimenting.
Use earlier rolling origins for feature selection, architecture choices, and hyperparameter tuning.
Freeze the selection rule and chosen configurations before evaluating the held-out test.
A final test can include multiple rolling origins, provided the refit protocol was fixed in advance.
At each test origin, use only data actually available then, including realized outcomes from earlier origins if production would have them.
Do not use later test outcomes to revise the architecture and continue reporting the same period as untouched.
If the final test informs another round of selection, it becomes development data and a new holdout is needed.

Input: candidate set and time boundaries.
Output: a locked protocol and a final comparison.
Branch: where insufficient data prevents a meaningful holdout, state that uncertainty and use a prospective pilot rather than claiming conclusive offline evidence.

### 5.13 Score forecasts at the decision horizon

For actual values y and point forecasts f, define error as f minus y.
MAE is the mean of absolute errors, expressed in the same units as y.
Bias is the mean signed error under that convention.
WAPE is sum of absolute errors divided by sum of absolute actuals, when the denominator is positive.
WAPE can be dominated by high-volume items and changes with the evaluation mix.
Report item-level or segment-level summaries alongside an aggregate.

For quantile level q, pinball loss is q times (y minus f) when y is at least f, and (1 minus q) times (f minus y) otherwise.
A lower pinball loss is better for the same q and evaluation population.
For interval coverage, count outcomes inside the stated lower and upper limits, divide by the number of outcomes, and report average interval width.
Report coverage by horizon, volume, season, and promotion state when sample sizes allow.
A nominal 90% interval with 90% coverage overall may still fail badly for promoted items.

For cumulative demand, sum each sampled path over the protection period and calculate its quantiles.
Do not add weekly 90th percentiles and call the result the 90th percentile of the sum.
Quantiles of a sum depend on the dependence between periods.
If only marginal quantiles exist, a separate dependence assumption or scenario model is needed.
Validate the resulting cumulative distribution against realized cumulative demand.

Input: forecast and actual records joined without changing the evaluated population.
Output: point and probabilistic metrics by operationally relevant segment and horizon.
Branch: if an error denominator is zero, mark WAPE undefined and use a suitable alternative rather than substituting an arbitrary number.

### 5.14 Translate forecast performance into decisions

Hold the replenishment policy constant when comparing forecast inputs, unless the experiment is explicitly about a policy change.
Use the same initial inventory, inbound orders, lead times, capacity, and service assumptions for each candidate.
Simulate stock carried, lost sales or backorders, waste, expedite orders, and constraint violations.
Apply marginal cost assumptions once, consistently, in a common currency and time period.
A unit forecast error is not itself a dollar saving.

Avoid double counting the same recovered margin as both revenue and profit.
For working capital, distinguish the inventory asset balance from the financing or carrying cost of holding it.
Account for training, inference, engineering, monitoring, vendor fees, and planner time.
Report gross operational value, incremental recurring cost, one-time implementation cost, and net value separately.
Compare forecast latency with the order cutoff and include fallback behavior.

Input: frozen forecasts, a transparent policy simulator, cost rates, and operational constraints.
Output: incremental annualized economics and a sensitivity range.
Branch: if simulation assumptions cannot be validated, use results as screening evidence and test the decision prospectively.

### 5.15 Pilot, operate, and revisit

Choose comparable operational units and define a control or a defensible phased rollout.
Where practical, randomize eligible units and protect against spillovers such as shared inventory pools.
Freeze eligibility, intervention, primary outcome, observation window, and analysis method before launch.
If randomization is impossible, record the comparison limitations and baseline differences.
Track actual planner actions, overrides, orders, inventory, service, and forecast availability.
A shadow forecast can establish technical readiness but does not prove the value of changed decisions.

Set stop rules for unacceptable service degradation, stale data, missing forecasts, or excessive latency.
Define a fallback to the incumbent process and assign operational ownership.
Monitor forecast error and distribution calibration after outcomes arrive; monitor data completeness and timeliness immediately.
Review changes in product mix, promotions, fulfillment constraints, and model costs.
Retrain only according to an approved trigger or schedule and compare each new version against the incumbent.
Retire a model when its incremental value no longer covers its cost or its assumptions fail.

Input: validated candidate, intervention protocol, operating owner, and fallback.
Output: a controlled decision and an auditable production service, or an explicit decision to retain the baseline.

### 5.16 Apply the shortlist to different decisions

The following cases illustrate how the contract changes the shortlist.
They are decision patterns, not claims about measured model performance.
Each assumes the simple baselines remain in the comparison.

**Case A: stable replenishment with limited covariates.**
The buyer orders established products weekly.
The histories contain repeated seasonal patterns, but there is no reliable future promotion feed.
The required output is a point estimate for the next eight weeks.
N-BEATS is a natural direct-forecast candidate.
N-HiTS is also worth considering because the horizon spans several temporal scales.
DeepAR may still be useful if the eventual policy requires a distribution.
Compare candidates with seasonal naive on the same weekly origins.
Reject an architecture if its aggregate gain depends only on a few large items and its critical segments deteriorate.

**Case B: promotion-sensitive retail.**
The retailer has archived promotion plans at each order cutoff.
The plan contains promotion dates and the intended discount, but execution sometimes differs.
TFT is a candidate because it can use a known-future plan as a conditional input.
The same plan should be supplied to a reasonable non-neural feature-aware baseline.
Score separately for executed promotions, canceled promotions, and ordinary weeks.
The operational forecast must use the latest plan actually available at the cutoff.
If plan revisions are frequent, consider separate scenarios rather than calling the initial plan certain.
Do not interpret the forecast change under a promotion scenario as its causal effect without further assumptions.

**Case C: uncertain cumulative consumption.**
A spare-parts distributor needs protection against total demand during a variable lead time.
DeepAR can produce paths for a fixed horizon, subject to likelihood and calibration checks.
The team must additionally represent lead-time uncertainty and any dependence between lead time and demand.
Simply sampling demand paths while fixing lead time at its average can understate exposure.
If lead time and demand worsen together during disruptions, independent sampling is not automatically appropriate.
The policy simulator should evaluate joint demand and supply scenarios.
Compare empirical stockout frequency and carrying costs under the same order rules.

**Case D: long daily histories and periodic behavior.**
A manufacturer records daily component demand over several years.
The planning horizon is a quarter and long seasonal context may be useful.
PatchTST is a candidate if patch length, stride, context, and hardware fit the workload.
N-HiTS is another candidate for multi-resolution direct forecasting.
Compare daily peaks and aggregate weekly totals, not only overall average error.
If the data show intermittent, low-volume demand, add a suitable intermittent-demand baseline.
Long history alone does not establish that old observations remain relevant.

**Case E: scarce new-product history.**
A newly introduced product has only six weeks of sales.
Most of the five architectures cannot learn its behavior from that series alone reliably.
Global pooling can transfer information from related products if their relationships are defensible.
Attributes, launch timing, and substitutions should be available at the historical forecast origin.
Compare against an explicit analogue-product baseline.
Run a launch-specific evaluation that excludes the target product's later history.
If meaningful analogues do not exist, communicate wide uncertainty and use planner judgment under a documented policy.

### 5.17 Make the experimental comparison attributable

The observed difference between two forecasting systems can arise from several sources.
Architecture is only one source.
Others include additional features, different training windows, tuning budgets, and different refit schedules.
A fair comparison does not require identical implementations, but it does require disclosure of material differences.

Use two complementary comparisons when the data permit.
First compare deployable end-to-end systems under the information each can actually obtain.
That comparison answers which system best supports the decision.
Second compare candidates using a matched feature set and evaluation protocol.
That comparison helps isolate whether the architecture adds value beyond the information.
Neither comparison is a randomized causal experiment on the architecture unless the experimental design supports that interpretation.

For each candidate, record:

1. Target and unit of measure.
2. Training window and eligible series.
3. Historical and future feature lists.
4. Forecast origin and horizon.
5. Training loss and forecast output.
6. Tuning budget and selection metric.
7. Refit frequency and cutoff.
8. Hardware, model version, and inference batch size.
9. Treatment of missing observations.
10. Failure behavior and fallback.

An equal tuning budget may be expressed in compute time or a prespecified number of trials.
Some architectures have more hyperparameters and may need more tuning to be viable.
Report the actual tuning resources rather than implying a perfectly controlled comparison.
Repeat candidate fits across a small prespecified set of random seeds when training variability matters.
Do not select the luckiest seed on the final test.
If a candidate fails to fit or deliver a forecast, record the failure instead of silently excluding difficult items.

### 5.18 Interpret uncertainty and economic risk together

Quantile forecasts can serve several different purposes.
A median can support a decision with symmetric absolute error costs.
An upper quantile can support an asymmetric shortage-versus-overage decision under a specified cost model.
For a single-period unconstrained newsvendor problem, the ideal quantile under standard assumptions is underage cost divided by the sum of underage and overage costs.
This formula depends on the assumptions of that decision model.
It is not automatically the correct target for a multi-period constrained replenishment network.

For example, suppose underage cost is $9 per unit and overage cost is $3 per unit.
The newsvendor critical fractile is $9 / ($9 + $3) = 0.75.
The corresponding decision uses the 75th percentile of demand under that simplified model.
If leftover stock can be sold later, its effective overage cost may be lower than purchase cost.
If an unmet order damages a contractual relationship, its underage cost may exceed the immediate margin.
Estimate costs with the business owner and test a range of plausible values.

Keep uncertainty about the forecast separate from uncertainty about the business case.
Forecast uncertainty concerns future demand conditional on the available information and model.
Business-case uncertainty also includes adoption, implementation time, inventory constraints, and cost assumptions.
A calibrated distribution can be costly to operate and therefore uneconomic.
A low-cost point forecast can be inadequate when the action depends on tail risk.
Both predictive and economic evidence are needed for the selection decision.

### 5.19 Set an operating service contract

Specify the batch deadline in the ordering team's time zone.
Specify the expected number of series, horizons, scenarios, and quantiles per run.
Specify how missing input data, failed model fits, and delayed upstream feeds are surfaced.
Specify whether the incumbent forecast or a seasonal-naive fallback is used after a failure.
Specify who approves a manual override and how its reason is recorded.
Specify how forecasts and recommendations are stored separately from approved orders.
Specify when realized outcomes become available for retrospective scoring.

Monitor input completeness and job timeliness at every run.
Monitor forecast accuracy and calibration only when the relevant outcomes have arrived.
Compare current error with a contemporaneous baseline, not solely with a fixed historical threshold.
A general demand shock may worsen both models even if their relative performance remains unchanged.
Conversely, stable aggregate error can conceal a deterioration for high-risk items.
Monitor cost per run and planner workload alongside model quality.
Define an owner for investigating every alert and an escalation path for missed service levels.

Record a versioned model artifact, code version, data snapshot, configuration, and timestamp for each published forecast.
This lineage makes it possible to reconstruct which information informed a decision.
Protect the final test and the operational pilot from informal retuning.
If a new model version is proposed, compare it prospectively or on a fresh eligible evaluation period.
Retain the prior version and fallback until the replacement has met the approved criteria.

## 6. Worked example

### 6.1 Setting and decision

A fictional regional distributor replenishes ambient grocery products weekly.
The example is illustrative and is not a reported company result.
The pilot covers 120 SKU-location combinations with comparable weekly ordering.
The supplier lead time is four weeks and the review interval is one week.
The policy therefore uses a five-week protection period, assuming orders arrive as scheduled.
Units below are product units, not cases; money is in US dollars.

The business wants to compare its incumbent forecast with a neural candidate.
The incumbent is a seasonal-naive forecast with a documented fallback for new items.
A DeepAR candidate produces simulated weekly paths and five-week demand quantiles.
A TFT candidate uses archived promotion-plan vintages and produces marginal weekly quantiles.
N-BEATS, N-HiTS, and PatchTST are also evaluated as point candidates.
The example numbers below are hypothetical evaluation results; they are not architecture performance claims.

The forecast estimates future customer demand.
The replenishment plan translates forecast information and inventory position into proposed orders.
The target is a 95% cycle-service objective, not a guaranteed forecast outcome.
The commitment is the purchase order accepted by the supplier, subject to contractual terms.

### 6.2 Dataset and chronology

The historical panel contains 156 completed weeks for eligible mature items.
Weekly sales are adjusted only where availability evidence supports a correction; other stockout weeks are flagged.
Item reclassifications are joined using effective dates.
The training examples are built from historical windows using only information available at their origins.
The business archives each promotion schedule as published at the weekly cutoff.
Calendar features are known at all future weeks; realized weather and future stockouts are excluded.

The development period contains eight rolling origins, each four weeks apart.
A later test period contains four rolling origins, also four weeks apart.
Each origin scores five future weeks; adjacent origins therefore share one realized week.
That overlap reflects repeated decisions but makes origin scores dependent.
The hyperparameter search is completed on development origins before the test is opened.
The models are refitted at each origin using a fixed, predeclared schedule.

A data quality review finds 7% of eligible item-weeks have an availability warning.
Results are reported both for the full eligible panel and the availability-warning subset.
A promotion revision after a historical origin is not allowed into that origin's forecast.
If the original promotion snapshot is missing, the observation is flagged and the declared missing-plan rule is used.
The 120 combinations are selected independently of final test errors.

### 6.3 Candidate results

Suppose the held-out test produces the following aggregate point results.
The population, origins, and target observations are identical for each row.
The denominators contain 48,000 units of actual demand across all scored records.
The figures are illustrative and do not encode a general model ranking.

| Forecast | Sum of absolute errors, units | WAPE | Mean signed error, units per scored record | Interpretation |
|---|---:|---:|---:|---|
| Seasonal naive | 9,600 | 20.0% | -0.8 | Incumbent benchmark |
| N-BEATS | 8,880 | 18.5% | -0.5 | History-driven direct forecast |
| N-HiTS | 8,640 | 18.0% | -0.4 | Multi-resolution candidate |
| PatchTST | 8,736 | 18.2% | -0.3 | Patch-based candidate |
| TFT | 8,160 | 17.0% | -0.2 | Uses archived plans |
| DeepAR median | 8,400 | 17.5% | -0.3 | Median of simulated paths |

For seasonal naive, WAPE is 9,600 / 48,000 = 0.20, or 20.0%.
For TFT, WAPE is 8,160 / 48,000 = 0.17, or 17.0%.
The absolute improvement is 3.0 percentage points.
The relative reduction in WAPE is (20.0% - 17.0%) / 20.0% = 15.0%.
Neither number says that a 15% monetary saving has been achieved.
A median is an appropriate point forecast for absolute error, but it is not the mean of an asymmetric distribution.
The reported bias uses forecast minus actual; negative values indicate underforecasting on average.

The point table is not sufficient for a service policy.
The TFT row uses a useful plan feature while the history-only rows in this illustration do not.
A separate feature-aware baseline is needed to distinguish the value of the promotion information from the value of the architecture.
A comparison of TFT against DeepAR is also affected by their respective information sets and training choices.
The four test origins are few, and the error differences may be unstable.

### 6.4 Probabilistic assessment

Suppose DeepAR generates 500 sample paths per forecast origin and item-location.
Each path is summed across the five-week protection period.
For 480 scored five-week item-origin cases, 432 realized sums fall within the predicted 5th-to-95th percentile interval.
Empirical coverage is 432 / 480 = 0.90, or 90%.
The nominal interval coverage is also 90%; average interval width is 74 units in this illustration.
The agreement is descriptive, not proof of calibration in every segment.
Overlapping origins and common shocks reduce effective independent information.

Suppose the 90th percentile of five-week demand for one item-origin is 176 units.
Inventory position at the order cutoff is 143 units.
Ignoring case packs and capacity, a simple order-up-to rule recommends 176 - 143 = 33 units.
If the case pack is 12 units and orders must be whole cases, rounding up gives 36 units.
The 90th percentile is an example policy input; it is not automatically the right percentile for the stated service target under all lead-time and review assumptions.
Lost sales, backorders, uncertain lead time, and order constraints change the relationship between quantiles and service.
The planner must apply the approved policy and check the resulting proposed order.

Consider one test outcome with a predicted 90th percentile of 176 units and realized five-week demand of 190 units.
The underprediction is 190 - 176 = 14 units.
Its pinball loss is 0.90 × 14 = 12.6 units.
For a different case with the same prediction and an actual of 160 units, overprediction is 16 units.
Its pinball loss is 0.10 × 16 = 1.6 units.
The asymmetric penalty reflects the chosen quantile, not a direct accounting cost.

### 6.5 Decision simulation and economics

A frozen policy simulator uses the same starting inventory, delivery assumptions, constraints, and demand paths for both forecasts.
Suppose the annualized simulation estimates $84,000 less shortage and expedite cost and $26,000 less carrying and waste cost for the candidate relative to the incumbent.
The gross annual benefit is $84,000 + $26,000 = $110,000 per year.
This result depends on assumed marginal costs and no double counting.
It is a simulation estimate, not a realized company saving.

Incremental recurring costs are $18,000 per year for compute and storage, $22,000 per year for engineering and monitoring, and $10,000 per year for planner and support effort.
The recurring total is $50,000 per year.
The one-time implementation cost is $45,000.
Net first-year value is $110,000 - $50,000 - $45,000 = $15,000.
Net later-year value, assuming no change in benefits or recurring costs, is $110,000 - $50,000 = $60,000 per year.
The first-year break-even gross benefit is $50,000 + $45,000 = $95,000.
The benefit estimate exceeds this threshold by $15,000, a relatively small margin for an uncertain simulation.

If gross benefit is 30% lower, it is $110,000 × 0.70 = $77,000 per year.
First-year net value then becomes $77,000 - $50,000 - $45,000 = -$18,000.
If recurring cost rises by $20,000 per year instead, first-year net value is $110,000 - $70,000 - $45,000 = -$5,000.
These sensitivities argue for a measured pilot rather than an immediate full rollout.
The simulation should also test lead-time variability, promotion-plan errors, demand censoring, and planner overrides.

### 6.6 Decision and next evidence

The hypothetical team selects DeepAR for a prospective pilot because the policy needs cumulative-demand scenarios and the implementation can produce paths within the cutoff.
This is a case-specific choice, not a claim that DeepAR has the best point score.
TFT remains a comparison candidate because its point score is lower, but weekly marginal quantiles alone do not provide a joint five-week distribution.
A separately validated dependence model could make TFT useful for that decision.
N-HiTS and the other point candidates remain valuable controls for whether neural complexity improves point accuracy.

The pilot compares comparable item-location groups under a predeclared assignment and policy.
It tracks realized service, shortage cost, carrying cost, override rate, computation, and missed cutoffs.
The baseline remains available as a rollback.
The team commits to a stop rule if data arrive late or the agreed service threshold is breached.
It does not claim savings until operational results and their uncertainty are assessed.

## 7. Code, query, or calculation

The following Python 3 example uses only the standard library.
It compares two precomputed sets of hypothetical forecasts against the same four outcomes.
It is a scoring and economics harness, not an implementation or training run of any neural architecture.
Replace the illustrative rows with out-of-sample forecast records produced by the frozen rolling-origin protocol.
Each row represents one five-week cumulative-demand case, measured in units.
The quantiles are for cumulative demand, not sums of weekly marginal quantiles.
The economics values are hypothetical annual US dollars.

```python
from statistics import mean

# Every row is a five-week demand outcome and forecasts available at its origin.
# Point forecasts are evaluated with absolute error. Bounds are a nominal
# 80% interval, and q90 is a 90th-percentile cumulative-demand forecast.
rows = [
    {"actual": 100, "base": 90, "candidate": 95,
     "lower": 80, "upper": 120, "q90": 120},
    {"actual": 120, "base": 110, "candidate": 125,
     "lower": 100, "upper": 140, "q90": 140},
    {"actual": 80, "base": 95, "candidate": 85,
     "lower": 70, "upper": 105, "q90": 105},
    {"actual": 150, "base": 130, "candidate": 140,
     "lower": 110, "upper": 145, "q90": 145},
]


def point_metrics(records, forecast_key):
    actual_total = sum(abs(r["actual"]) for r in records)
    if not records or actual_total == 0:
        raise ValueError("Need records with a nonzero actual denominator")
    errors = [r[forecast_key] - r["actual"] for r in records]
    mae = mean(abs(e) for e in errors)
    bias = mean(errors)
    wape = sum(abs(e) for e in errors) / actual_total
    return mae, bias, wape


def pinball(actual, forecast, q):
    if not 0 < q < 1:
        raise ValueError("Quantile level must be strictly between 0 and 1")
    error = actual - forecast
    return max(q * error, (q - 1) * error)


def interval_metrics(records):
    if not records:
        raise ValueError("Need at least one record")
    if any(r["lower"] > r["upper"] for r in records):
        raise ValueError("Interval bounds are reversed")
    covered = [r["lower"] <= r["actual"] <= r["upper"] for r in records]
    widths = [r["upper"] - r["lower"] for r in records]
    return mean(covered), mean(widths)


def first_year_value(gross_benefit, recurring_cost, implementation_cost):
    return gross_benefit - recurring_cost - implementation_cost


for name, key in [("baseline", "base"), ("candidate", "candidate")]:
    mae, bias, wape = point_metrics(rows, key)
    print(f"{name}: MAE={mae:.2f} units, bias={bias:.2f} units, "
          f"WAPE={wape:.2%}")

coverage, width = interval_metrics(rows)
q90_loss = mean(pinball(r["actual"], r["q90"], 0.90) for r in rows)
print(f"candidate: interval coverage={coverage:.2%}, "
      f"mean width={width:.2f} units")
print(f"candidate: mean q90 pinball loss={q90_loss:.2f} units")
print(f"first-year net value=${first_year_value(110000, 50000, 45000):,.0f}")
```

Expected output:

```text
baseline: MAE=13.75 units, bias=-6.25 units, WAPE=12.22%
candidate: MAE=6.25 units, bias=-1.25 units, WAPE=5.56%
candidate: interval coverage=75.00%, mean width=37.50 units
candidate: mean q90 pinball loss=2.75 units
first-year net value=$15,000
```

The actual denominator is 450 units.
Baseline absolute errors total 55 units; candidate absolute errors total 25 units.
Three of four outcomes are within the nominal 80% interval.
One interval is missed on the upper side because 150 units exceeds 145 units.
The four q90 pinball losses are 2.0, 2.0, 2.5, and 4.5 units; their mean is 2.75 units.
Four observations cannot establish 80% coverage, nor can this scoring script establish production value.
In production, join on series, origin, and horizon; check missing forecasts and evaluate all eligible cases.
Store the actual data vintage and the version of the forecast used in the decision.

## 8. Data, standards, and regulatory context

A reliable forecast depends on a reliable definition of the entity and event being forecast.
Record product identifiers, location identifiers, units of measure, time zones, and effective dates.
A consumer unit, an inner pack, and a shipping case are not interchangeable.
A model that sees a pack-size change as a demand jump can learn a false pattern.
A location closure, item substitution, or product rebranding can change the meaning of a series.

GS1 publishes identification standards, including the Global Trade Item Number (GTIN) and Global Location Number (GLN).
These are conventions for identifying trade items and locations, not guarantees that an organization's historical master data are accurate.
Use a documented mapping from operational SKU identifiers to GTINs where applicable.
A GTIN may identify a trade-item configuration, while an internal SKU may encode additional local distinctions.
Do not merge histories merely because names look similar.
Preserve effective dates and ownership of identifier corrections.

GS1 also publishes Electronic Product Code Information Services (EPCIS) for sharing supply-chain event information.
An event can describe what happened, when, where, and under what business context.
An event standard can improve traceability, but it does not automatically establish customer demand or prove when a model could access an event.
Maintain publication and ingestion timestamps alongside event time.
Use those timestamps to reconstruct a valid information set for historical forecasting.

Electronic data interchange (EDI) is a family of structured business-message conventions.
A purchase order or advance shipping notice can describe a commitment or expected delivery, not necessarily realized demand.
Message interpretation depends on the trading-partner agreement and implementation guide.
Treat duplicate messages, revisions, and cancellations as part of the data contract.
An order placed because of a promotion is not automatically an independent measure of end-customer consumption.

Distinguish the following clocks:

- **Event time:** when a sale, receipt, or promotion activity occurred.
- **Source recording time:** when the source system recorded the event.
- **Availability time:** when the forecasting process could access it.
- **Correction time:** when the record was changed later.
- **Forecast issue time:** when a specific forecast version was published.
- **Decision time:** when the planner or system made the order.

A historical test should not give a model a correction that was unavailable at the relevant issue time.
If archived vintages do not exist, quantify the uncertainty introduced by reconstructing them.
A retrospective perfect-data analysis may be useful as a bound, but must be labeled accordingly.

Where customer-level or employee-level information enters a forecast, privacy law may apply.
For example, the EU General Data Protection Regulation (GDPR), Regulation (EU) 2016/679, sets requirements for processing personal data when it applies.
Applicability depends on the processing and jurisdiction; a product-level forecast is not automatically personal data.
Data minimization, access control, retention, and a lawful basis require assessment in context.
The GDPR does not prescribe one of these forecasting architectures.
Consult qualified privacy counsel and the organization's data governance function for a real deployment.

The EU AI Act, Regulation (EU) 2024/1689, has phased provisions and a scope determined by the use and legal classification of an AI system.
A supply-chain forecast is not automatically a high-risk system solely because it uses a neural network.
Assess applicable obligations against the actual use case and effective dates rather than inferring compliance from a model name.
This educational discussion is not legal advice.

Document data quality checks for timeliness, completeness, duplication, allowable values, and unit consistency.
Track the frequency and impact of missing promotion snapshots and stockout flags.
Set retention and access controls for raw records and model outputs according to company policy and applicable law.
Log a reproducible lineage from source snapshot to feature set, model version, forecast, and order recommendation.
An audit trail supports investigation; it does not substitute for calibrated forecasts or responsible decisions.

## 9. Industry practice and case evidence

### 9.1 What the original DeepAR study establishes

Salinas and coauthors describe DeepAR as an autoregressive recurrent probabilistic forecasting approach trained across related series.
Their paper evaluates the method on several datasets and discusses likelihood-based forecasting and sample paths.
The authors include Amazon affiliations, but the paper's empirical results do not prove that any particular reader will achieve the same savings.
The documented claim is about the method and reported experiments, not an audited commercial return on investment.
The paper supports DeepAR as a candidate for probabilistic forecasting, not as an unconditional production winner.

### 9.2 What the N-BEATS and N-HiTS papers establish

Oreshkin and coauthors present N-BEATS as a deep architecture using backcast and forecast residual connections.
They report results on public forecasting benchmarks and describe both generic and interpretable variants.
A benchmark result establishes performance on the studied datasets, splits, metrics, and configurations.
It does not establish performance under a retailer's stockouts, promotion revisions, or order constraints.
The interpretation of a learned trend or seasonality component is a model description, not causal identification.

Challu and coauthors present N-HiTS with multi-rate sampling and hierarchical interpolation for long-horizon forecasting.
Their results motivate testing the method on long horizons and comparing efficiency on the actual workload.
The appropriate input length, hierarchy, and computation depend on frequency and horizon.
A company should independently test whether brief demand peaks survive the chosen settings.

### 9.3 What the TFT and PatchTST papers establish

Lim and coauthors introduce TFT for interpretable multi-horizon forecasting with heterogeneous inputs.
The architecture is a natural candidate when static, observed-past, and known-future inputs are all available at issue time.
Its variable selection and attention can help inspect the model's behavior.
Neither visualization is a controlled experiment on the causal effect of a promotion.

Nie and coauthors introduce PatchTST using patching and channel independence for time-series transformers.
Their published experiments motivate a candidate for long-context forecasting.
Performance and efficiency depend on the dataset, tokenization, model configuration, and hardware.
Channel independence should be checked against a supply-chain use case with shared shocks or substitution.

### 9.4 A public competition as a reality check

The M5 forecasting competition used Walmart unit sales data and included uncertainty as well as point-forecasting tracks.
Makridakis and coauthors describe its setup and results in the International Journal of Forecasting.
This is documented evidence that a realistic retail benchmark can require hierarchical and probabilistic evaluation.
It is not evidence that Walmart deployed a particular architecture or obtained a particular saving.
Public competition data can also become familiar through repeated research use; success on them is not a substitute for an untouched local test.

[Inference] A retailer evaluating these models should expect item mix, availability, promotions, and the decision policy to affect transfer from a public benchmark.
That inference follows from the differences between benchmark evaluation and the local information and cost contract.
A prospective operational pilot is needed to establish local value.

### 9.5 Practice implications

A mature team compares models using the same origin-valid data and an operationally meaningful horizon.
It treats model choice, feature availability, and decision policy as separate experimental factors where feasible.
It measures data preparation, forecast latency, and human override effort alongside accuracy.
It keeps a simple fallback when a neural service fails or misses the ordering cutoff.
It publishes uncertainty about results, including thin segments and correlated rolling-origin outcomes.
These are recommended practices, not claims that every named organization follows them.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Future promotion leakage | Backtest uses a schedule revised after origin | Revised plan contains later information | Inflated expected accuracy | Archive and use issue-time plan vintages |
| Stockout-as-zero | Largest errors occur during low availability | Sales are censored by stock | Demand and order quantities are understated | Flag availability and validate any demand adjustment |
| Missing baseline | Proposal compares only neural models | No counterfactual for complexity | Spending without demonstrated incremental value | Include incumbent and simple baselines |
| Wrong horizon | Accuracy improves at one week but not five | Metric does not match protection period | Ordering decision sees no benefit | Score cumulative and horizon-specific forecasts |
| Invalid quantile aggregation | Weekly upper quantiles are added | Dependence between weeks is ignored | Misstated safety stock | Use validated joint paths or dependence model |
| Uncalibrated likelihood | Actual counts frequently outside intervals | Distributional family or parameters are wrong | Service targets are missed | Check calibration and compare likelihood choices |
| Recursive drift | Long-horizon samples wander implausibly | Generated errors feed later steps | Excessive or understated tails | Inspect path behavior and horizon-level errors |
| Quantile crossing | Predicted lower bound exceeds upper bound | Independently fitted quantiles are inconsistent | Invalid interval and downstream rules | Detect and apply a validated monotonicity treatment |
| Hidden feature asymmetry | One candidate receives more timely inputs | Information advantage mistaken for architecture advantage | Wrong investment attribution | Compare matched-information and practical-system results |
| Aggregate masking | Total WAPE improves but critical items deteriorate | High-volume series dominate | Local shortages and service loss | Report segment and item distributions |
| Unstable identifiers | Sudden break at a packaging change | Series definition changed | False trend and poor transfer | Version identifiers and units of measure |
| Incomplete vintages | Historical features look implausibly clean | Current master data overwrite history | Optimistic validation | Preserve snapshots and disclose reconstruction limits |
| Overlapping test labels | Many scores share the same realized week | Dependence inflates apparent sample size | Overconfident selection | Use origin or block uncertainty estimates |
| Excessive tuning | Test score improves after repeated model changes | Test became part of development | Biased generalization estimate | Freeze choices and reserve a new holdout |
| Patch smoothing | Brief promotion peaks are consistently missed | Patch representation obscures short events | Under-ordering at peaks | Tune patching and compare event-level performance |
| Misread attention | A weight is called promotion impact | Predictive association is treated as causation | Bad promotion or pricing decision | Use causal design for causal questions |
| Forecast service misses cutoff | Batch latency rises with sampling volume | Inference or data pipeline exceeds time budget | Orders use stale or absent inputs | Set latency budget and automatic fallback |
| Cost double count | Benefits include both recovered revenue and full margin | Same economic gain counted twice | False positive business case | Use a marginal-cost ledger and review assumptions |
| Model decay | Error and calibration worsen after a mix shift | Relationships change after training | Gradual loss of service or value | Monitor by segment, retrain or roll back |
| False service interpretation | A 90th percentile is described as 90% fill rate | Different service definitions are conflated | Misleading commitments | Define service metric and validate policy outcomes |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or respond |
|---|---|---|---|
| Target measures relevant demand | Forecast must match the decision | Stockouts censor sales | Compare with availability and lost-sales evidence |
| Historical vintages are recoverable | Backtest must reproduce available information | Plans and corrections overwrite past values | Audit timestamps and snapshot completeness |
| Related histories share useful structure | Global learning pools information | Dissimilar products or structural changes | Compare segment and local baselines |
| Future feature is known at origin | Conditional forecast must be feasible | Promotion is revised after cutoff | Compare archived plan and realized execution |
| Chosen likelihood fits demand | DeepAR paths depend on distributional assumptions | Excess zeros, heavy tails, or censoring | Assess calibration and distributional residuals |
| Sample paths represent dependence | Protection-period quantiles depend on joint behavior | Independent or misspecified temporal shocks | Validate cumulative sums and coverage |
| Costs are incremental | Economic comparison needs a common counterfactual | Shared costs or benefits counted twice | Review cost ledger with finance |
| Simulator captures policy constraints | Offline value depends on realistic actions | Capacity, case packs, or lead times omitted | Replay historical decisions and stress constraints |
| Backtest period is representative enough | Past outcomes inform prospective risk | New regime, launch, or supplier disruption | Segment by regime and run a pilot |
| Model fits operational cutoff | Forecast must arrive before orders | Data or sampling latency increases | Measure production-scale latency and fallback |
| Observations support uncertainty estimates | Coverage and segment metrics need enough cases | Few origins or correlated overlaps | Report counts and block-level uncertainty |
| Metric matches decision | Model selection should value relevant errors | Point MAE used for a service quantile | Score quantiles and simulate the policy |

No assumption is made true by naming it in a model card.
Where evidence is weak, report sensitivity and avoid overconfident economic claims.
No architecture can infer unavailable future shocks with certainty.
A probabilistic forecast is conditional on its inputs and modeling assumptions, not a guarantee of coverage during a regime never represented in its evaluation.

## 12. Exercises

### Exercise 1 — Define the contract

A buyer issues orders every Monday for goods arriving three weeks later.
A planning file contains weekly sales, a promotion calendar, and final realized prices.
Define the target, origin, minimum protection period under a simple periodic-review assumption, and one feature that may leak.
**Deliverable:** four labeled statements and one caveat about sales.
**Acceptance:** the answer distinguishes forecast time from outcome time and identifies the review period.
**Time:** 8 minutes.

### Exercise 2 — Compare mechanisms

Match each of the five families to its characteristic mechanism and typical original output.
Explain why marginal weekly quantiles are not automatically a cumulative-demand distribution.
**Deliverable:** five-row table and a two-sentence explanation.
**Acceptance:** the table distinguishes autoregressive paths, residual blocks, interpolation, heterogeneous inputs, and patching.
**Time:** 15 minutes.

### Exercise 3 — Design a backtest

A distributor has 160 weeks of history and forecasts four weeks ahead every week.
Design a development period and a later final test with at least four forecast origins.
Explain feature vintages, refitting, overlap, and how the final test remains untouched.
**Deliverable:** a timeline and five methodological bullets.
**Acceptance:** no feature or label unavailable at an origin is used to fit that origin's forecast; selection precedes test inspection.
**Time:** 20 minutes.

### Exercise 4 — Calculate point metrics

Actual demand is 40, 60, and 100 units.
A model forecasts 50, 55, and 90 units.
Calculate MAE, signed bias using forecast minus actual, and WAPE.
**Deliverable:** formulas and numerical answers with units where applicable.
**Acceptance:** the denominator of WAPE is total actual demand, and the sign convention is explicit.
**Time:** 8 minutes.

### Exercise 5 — Evaluate a quantile and an interval

A 90th-percentile forecast is 80 units and realized demand is 90 units.
A separate nominal 80% interval covers 8 of 10 cases and has average width 30 units.
Compute the pinball loss and empirical coverage; explain one reason the coverage result is inconclusive.
**Deliverable:** two calculations and one limitation.
**Acceptance:** correct asymmetric loss and recognition of the small sample.
**Time:** 8 minutes.

### Exercise 6 — Select a shortlist

A manufacturer has 5,000 related monthly component series, 48 months of history, and a 12-month planning horizon.
Promotion data do not apply; the decision initially requires point forecasts, but cumulative uncertainty may matter later.
Select at least two neural candidates and two simple or established baselines.
**Deliverable:** candidate table and a plan to assess the later probability requirement.
**Acceptance:** each selection follows from horizon, data, and output needs rather than a claimed universal ranking.
**Time:** 15 minutes.

### Exercise 7 — Calculate economic value

A simulation estimates $210,000 per year in gross benefit.
Incremental recurring costs are $95,000 per year and implementation costs are $130,000 once.
Calculate first-year net value, later annual net value assuming unchanged economics, and first-year break-even benefit.
**Deliverable:** three dollar amounts and one sensitivity worth testing.
**Acceptance:** one-time and recurring costs are separated; no claim of realized benefit is made.
**Time:** 8 minutes.

### Exercise 8 — Critique a proposal

A vendor reports a 12% WAPE reduction using final promotion schedules, scores one origin, and says attention proves promotions cause demand.
It estimates savings by multiplying unit error reduction by selling price.
**Deliverable:** four specific defects and a corrective action for each.
**Acceptance:** the critique covers leakage, sampling, causality, and decision economics.
**Time:** 15 minutes.

### Exercise 9 — Design an operational pilot

A candidate provides calibrated five-week paths in offline testing and passes a technical latency check.
Design a pilot that compares its replenishment recommendations with the incumbent.
**Deliverable:** a one-page protocol with allocation, policy, primary outcome, stop rule, and fallback.
**Acceptance:** the design distinguishes shadow scoring from an intervention and accounts for shared-inventory spillovers.
**Time:** 25 minutes.

## 13. Self-check answers

### Answer 1

The origin is each Monday's issue cutoff.
The target can be weekly unconstrained demand in product units for each eligible SKU-location.
If sales are used as a proxy, disclose stockout censoring and missing feeds.
Three weeks of lead time plus one week between reviews gives a four-week protection period under the stated simple assumption.
Final realized prices can leak if they were not available at the Monday cutoff.
A promotion calendar is usable only in the archived version available then.

### Answer 2

DeepAR uses an autoregressive recurrent likelihood and can generate sample paths.
N-BEATS uses stacked residual backcast and forecast blocks and originally emphasizes point forecasts.
N-HiTS uses multi-rate representations and hierarchical interpolation for a direct forecast.
TFT combines heterogeneous inputs with gating and attention and commonly predicts marginal quantiles.
PatchTST turns time observations into patches and uses a channel-independent transformer design for point forecasting.
Weekly quantiles lack the temporal dependence needed to derive a quantile of their sum.
A joint scenario method must be evaluated on cumulative outcomes.

### Answer 3 rubric

A strong design uses earlier origins for development and four or more later origins for the locked test.
For example, use origins through week 128 for development and later origins 140, 144, 148, and 152 for a four-week horizon, if the training and evaluation boundaries are specified consistently.
Any training example with a future label beyond an origin's permitted training cutoff must be excluded at that origin.
The final model configuration and refit schedule are fixed before test outcomes are inspected.
The protocol records as-of feature versions and accounts for correlated overlapping outcomes where origins overlap.
A design that revises model settings using the same final test is not an untouched test.

### Answer 4

Errors are +10, -5, and -10 units.
Absolute errors are 10, 5, and 10 units.
MAE is 25 / 3 = 8.33 units, rounded to two decimal places.
Signed bias is -5 / 3 = -1.67 units.
WAPE is 25 / (40 + 60 + 100) = 0.125, or 12.5%.

### Answer 5

The forecast is below the outcome by 10 units.
The 90th-percentile pinball loss is 0.90 × 10 = 9 units.
Empirical coverage is 8 / 10 = 80%.
Ten cases provide little evidence about long-run coverage, especially if cases are dependent or not representative.
Interval width also matters: arbitrarily wide intervals can obtain high coverage without being useful.

### Answer 6 rubric

A strong answer might shortlist N-HiTS for a long direct horizon and N-BEATS as a history-driven comparator.
DeepAR is a reasonable additional candidate if future cumulative uncertainty and related-series pooling matter.
PatchTST is also defensible if the available 48-month histories and patch choices support meaningful long-context evaluation.
Seasonal naive and a moving average or a suitable statistical forecast are defensible baselines.
An answer must not claim that any candidate is certain to win.
For uncertainty, evaluate joint cumulative paths and calibration rather than treating a point forecast as a distribution.

### Answer 7

First-year net value is $210,000 - $95,000 - $130,000 = -$15,000.
Later annual net value under unchanged assumptions is $210,000 - $95,000 = $115,000 per year.
First-year break-even gross benefit is $95,000 + $130,000 = $225,000.
A useful sensitivity varies the gross benefit or the cost of ongoing support.
The simulation estimate is not a realized saving.

### Answer 8

The final schedule can leak changes made after the forecast origin; use archived issue-time plans.
One origin gives fragile evidence; evaluate multiple origins and report uncertainty.
Attention is a fitted-model diagnostic, not causal identification; use an appropriate causal design for promotion effects.
Multiplying unit error by selling price ignores the decision policy, marginal margin, inventory, and costs; simulate decisions and validate incremental economic outcomes.
A strong critique also requests a common baseline and a locked test.

### Answer 9 rubric

A strong protocol defines eligible units, comparable assignment, and an intervention in which the forecast actually informs orders.
Randomization is desirable where feasible, with shared inventory and supplier-capacity spillovers addressed in the design.
The primary outcome is an agreed operational or monetary measure, with service guardrails and a fixed observation window.
The protocol specifies data and service stop rules, human authority, fallback forecasts, and how overrides are logged.
Shadow scoring is a useful technical control but cannot establish the effect of changing the order decision.
A nonrandomized design states its confounding limitations and avoids causal certainty.

## 14. Interview questions

1. **Why can a global model outperform local models?**
   A strong answer explains shared parameters, related-series data, and negative transfer.
   A weak answer says that pooling always improves accuracy.
2. **When would you shortlist DeepAR?**
   A strong answer connects related histories, probabilistic paths, likelihood choice, and inference cost.
   A weak answer assumes all sample paths are calibrated by construction.
3. **What is distinctive about N-BEATS?**
   A strong answer describes residual backcast and forecast blocks and distinguishes the original point output from extensions.
   A weak answer calls the backcast a future prediction.
4. **Why might N-HiTS help on a long horizon?**
   A strong answer discusses multi-rate representations and interpolation, then requires local evidence.
   A weak answer promises that coarse representations preserve every short spike.
5. **What makes TFT attractive and risky?**
   A strong answer discusses heterogeneous inputs, known-future vintages, and noncausal interpretations.
   A weak answer feeds future realized prices into a backtest.
6. **What is a PatchTST patch?**
   A strong answer explains a contiguous group of observations, token reduction, and patch-size trade-offs.
   A weak answer assumes patching establishes cross-product dependence.
7. **How do you test five-week inventory uncertainty?**
   A strong answer evaluates joint cumulative-demand distributions and realized coverage.
   A weak answer sums weekly 90th percentiles without a dependence model.
8. **How do you prevent time-series leakage?**
   A strong answer reconstructs origin-valid vintages, fits preprocessing within each origin, and separates test from selection.
   A weak answer relies only on a chronological split.
9. **Can a lower WAPE justify deployment?**
   A strong answer connects the change to the decision policy, marginal economics, runtime, and uncertainty.
   A weak answer converts percentage error reduction directly into dollar savings.
10. **What does an 80% interval mean?**
    A strong answer defines nominal coverage, checks empirical coverage and width, and notes segment limitations.
    A weak answer treats an upper 80th percentile as an 80% interval.
11. **How do you compare two candidates using different covariates?**
    A strong answer reports both practical-system performance and matched-information comparisons when feasible.
    A weak answer attributes all improvement to architecture.
12. **What makes a pilot credible?**
    A strong answer specifies assignment, intervention, outcomes, guardrails, spillovers, and fallback.
    A weak answer treats a shadow forecast as proof of operational savings.

## 15. Key terms

- [Attention](#4-definitions-and-vocabulary): data-dependent weighting within a model.
- [Autoregression](#55-understand-deepar): use of prior target values, including generated values, for subsequent steps.
- [Backcast](#56-understand-n-beats): fitted representation of part of the input history.
- [Baseline](#54-establish-a-fair-baseline): simple or incumbent comparator chosen before final testing.
- [Bias](#513-score-forecasts-at-the-decision-horizon): mean signed forecast error under a specified convention.
- [Calibration](#3-the-core-mental-model): agreement of predictive probabilities with observed frequencies.
- [Censoring](#52-build-an-as-of-dataset): observation limited by availability or another constraint.
- [Context window](#4-definitions-and-vocabulary): history supplied to a model.
- [Coverage](#64-probabilistic-assessment): observed proportion within prediction bounds.
- [Covariate](#53-separate-feature-availability-classes): explanatory input beyond the target series.
- [Data vintage](#53-separate-feature-availability-classes): snapshot actually available at an origin.
- [DeepAR](#55-understand-deepar): probabilistic autoregressive recurrent forecasting family.
- [Forecast horizon](#51-start-with-a-decision-contract): future periods predicted from an origin.
- [Forecast origin](#511-choose-a-rolling-origin-evaluation): time at which a forecast is issued.
- [Global model](#3-the-core-mental-model): model sharing parameters across series.
- [Known-future feature](#53-separate-feature-availability-classes): input genuinely available for a future period at issuance.
- [Likelihood](#55-understand-deepar): assumed distribution or density for observations.
- [MAE](#513-score-forecasts-at-the-decision-horizon): mean absolute error in target units.
- [N-BEATS](#56-understand-n-beats): residual-block direct forecasting architecture.
- [N-HiTS](#57-understand-n-hits): multi-rate hierarchical interpolation forecasting architecture.
- [PatchTST](#59-understand-patchtst): patch-based time-series transformer architecture.
- [Pinball loss](#513-score-forecasts-at-the-decision-horizon): asymmetric scoring loss for a quantile.
- [Protection period](#61-setting-and-decision): exposure covered by a particular replenishment policy.
- [Quantile](#3-the-core-mental-model): probability-indexed threshold of a forecast distribution.
- [Rolling origin](#511-choose-a-rolling-origin-evaluation): repeated issue-time evaluation through history.
- [Sample path](#55-understand-deepar): one simulated sequence of future values.
- [Seasonal naive](#54-establish-a-fair-baseline): forecast based on a corresponding past season.
- [TFT](#58-understand-temporal-fusion-transformer): Temporal Fusion Transformer for heterogeneous multi-horizon inputs.
- [WAPE](#513-score-forecasts-at-the-decision-horizon): absolute errors divided by absolute actuals.

## 16. Sources

Primary architecture papers and public documentation:

1. Salinas, D., Flunkert, V., Gasthaus, J., and Januschowski, T. (2020). “DeepAR: Probabilistic forecasting with autoregressive recurrent networks.” *International Journal of Forecasting*, 36(3), 1181–1191. [DOI: 10.1016/j.ijforecast.2019.07.001](https://doi.org/10.1016/j.ijforecast.2019.07.001).
2. Oreshkin, B. N., Carpov, D., Chapados, N., and Bengio, Y. (2020). “N-BEATS: Neural basis expansion analysis for interpretable time series forecasting.” *International Conference on Learning Representations*. [OpenReview](https://openreview.net/forum?id=r1ecqn4YwB).
3. Challu, C., Olivares, K. G., Oreshkin, B. N., Ramirez, F. G., Canseco, M. M., and Dubrawski, A. (2023). “N-HiTS: Neural hierarchical interpolation for time series forecasting.” *Proceedings of the AAAI Conference on Artificial Intelligence*. [AAAI article](https://ojs.aaai.org/index.php/AAAI/article/view/25854).
4. Lim, B., Arik, S. O., Loeff, N., and Pfister, T. (2021). “Temporal Fusion Transformers for interpretable multi-horizon time series forecasting.” *International Journal of Forecasting*, 37(4), 1748–1764. [DOI: 10.1016/j.ijforecast.2021.03.012](https://doi.org/10.1016/j.ijforecast.2021.03.012).
5. Nie, Y., Nguyen, N. H., Sinthong, P., and Kalagnanam, J. (2023). “A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.” *International Conference on Learning Representations*. [OpenReview](https://openreview.net/forum?id=Jbdc0vTOcol).
6. Makridakis, S., Spiliotis, E., and Assimakopoulos, V. (2022). “The M5 accuracy competition: Results, findings, and conclusions.” *International Journal of Forecasting*, 38(4), 1346–1364. [DOI: 10.1016/j.ijforecast.2021.11.013](https://doi.org/10.1016/j.ijforecast.2021.11.013).
7. Makridakis, S., Spiliotis, E., and Assimakopoulos, V. (2022). “The M5 uncertainty competition: Results, findings and conclusions.” *International Journal of Forecasting*. [DOI: 10.1016/j.ijforecast.2021.10.009](https://doi.org/10.1016/j.ijforecast.2021.10.009).
8. Hyndman, R. J., and Athanasopoulos, G. *Forecasting: Principles and Practice*, 3rd ed. [Time-series cross-validation](https://otexts.com/fpp3/tscv.html) and [forecast accuracy](https://otexts.com/fpp3/accuracy.html).
9. GS1. [GTIN identification](https://www.gs1.org/standards/id-keys/gtin), [GLN identification](https://www.gs1.org/standards/id-keys/gln), and [EPCIS standard](https://www.gs1.org/standards/epcis).
10. European Union. [Regulation (EU) 2016/679, General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj), and [Regulation (EU) 2024/1689, Artificial Intelligence Act](https://eur-lex.europa.eu/eli/reg/2024/1689/oj). Consult current applicable provisions and professional advice for a specific use.

The paper results are evidence about the authors' reported settings, not independent proof of a local operational benefit.
Regulatory applicability and dates depend on the particular system and jurisdiction.

## 17. Links

- [Module 18 index](_index.md).
- [Module 18 survey and lab sections](../../../course/18_deep_learning_foundation_models_llm_agents.md).
- [Sequence architectures](02_sequence-architectures.md), a sibling topic in the module index.
- [Global versus local models](03_global-local-models.md), a sibling topic in the module index.
- [Time-series foundation models](05_ts-foundation-models.md), a sibling topic in the module index.
- [Forecasting survey](../../../course/07_time_series_forecasting.md), for baseline and evaluation context.
- [Simulation, optimization, and production labs](../../../course/30_simulation_optimization_production_labs.md), for operational experimentation context.
- [Datasets, tools, and benchmarks](../../../course/27_datasets_tools_benchmarks.md), for public data and benchmark context.
