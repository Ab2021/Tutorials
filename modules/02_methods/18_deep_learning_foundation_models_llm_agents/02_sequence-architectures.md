# Sequence Architectures for Supply-Chain Time Series

> Module 18 — Deep Learning, Foundation Models, and LLM Agents  
> Topic 02 of 10  
> Estimated study time: 4 hours  
> Prerequisites: basic arithmetic and Python; neural-network essentials are introduced below  
> Version 2026.5

## 1. Why this topic exists

A replenishment planner must place an order before future demand is known.
The question is not whether a neural network is fashionable.
It is whether a particular forecasting system improves an executable order compared with a cheaper, credible alternative.

Suppose a grocery distributor replenishes 800 store–product pairs each Monday.
Each forecast covers the next four weekly buckets.
The planner needs projected demand, uncertainty, and an explanation of data quality before approving orders.
A forecast is an estimate of future demand, not the order itself.
A plan combines that estimate with inventory, lead time, constraints, and policy.
A target is a desired outcome, such as a service level.
A commitment is an approved quantity or delivery promise.
Confusing these four objects can turn a forecast improvement into an infeasible purchase order.

Seasonal patterns, promotions, availability, and holidays appear at different distances in the past.
A sequence architecture determines how historical observations can interact before a prediction is made.
Recurrent neural networks, temporal convolutional neural networks, and transformer encoders represent those interactions differently.
They do not repair incorrect data or make the future observable.

A retailer with three years of weekly data may have only about 156 observations for each item–store pair.
That is often too little to fit a flexible neural network independently to every pair.
A shared model can learn across related pairs, but only if their patterns and input definitions are sufficiently comparable.
A seasonal-naive forecast may still be the sensible production choice.

This topic gives a buyer of analytics, a practitioner, and a reviewer a common way to compare options.
The unit of comparison is the whole decision pipeline: data availability, model, forecast, replenishment rule, operating cost, and control.
The worked case and code use synthetic data; neither is a claim about a retailer's actual performance.

### Decision at stake

The demand planner needs a forecast by 07:00 each Monday.
The purchasing manager needs a proposed order before the supplier cut-off.
The data engineer needs a reproducible as-of snapshot.
The finance partner needs a defensible cost comparison.
The risk owner needs a fallback if the model or upstream feed fails.
The model owner needs a measured reason to choose one architecture over another.

### Where sequence models earn consideration

They may be useful when many related series share patterns that are too complex for the available simple models.
They may incorporate multiple covariates consistently at a common forecast origin.
A global model can share information between products without training a separate model per product.
None of these possibilities proves that neural models will beat a local benchmark.

### Where to stop early

Do not train an encoder if the decision owner cannot say what the forecast will change.
Do not trust a demand label when stockouts have hidden a material share of demand.
Do not compare models on different forecast origins, horizons, or data vintages.
Do not deploy a candidate when its benefit is smaller than uncertainty and operating cost.

## 2. Learning outcomes

After this topic, you can perform the following observable tasks.
Each outcome is assessed by a named exercise in section 12.

1. **Define** a forecast origin, lookback, horizon, and as-of feature rule.
   Exercise 1 asks you to identify permissible information.
2. **Compute** window counts, a convolutional receptive field, and lead-time demand.
   Exercises 2 and 4 require explicit arithmetic.
3. **Explain** recurrent state, convolutional filters, and attention with positional information.
   Exercise 3 asks you to compare the information paths.
4. **Design** a leakage-resistant rolling-origin comparison and an appropriate baseline.
   Exercise 5 requires a split, a data-vintage policy, and a comparison unit.
5. **Evaluate** forecast quality, inventory consequences, uncertainty, and operating cost jointly.
   Exercise 6 requires a decision calculation and a sensitivity check.
6. **Diagnose** stockout censoring, missingness, and a misleading attention explanation.
   Exercise 7 requires evidence and a corrective action.
7. **Choose** a model and a fallback using explicit constraints and a review trigger.
   Exercise 8 requires a business recommendation rather than an architecture ranking.

A strong outcome is reproducible by another practitioner from the same snapshot and assumptions.
A merely plausible explanation is not enough if it cannot survive an as-of data audit.

## 3. The core mental model

A sequence encoder is a mechanism for turning a permitted history into a representation.
A prediction head maps that representation, and any legitimately known future inputs, to a forecast.
The head can output one number, several horizon-specific numbers, quantiles, or parameters of a predictive distribution.
A replenishment policy consumes those predictions; the encoder itself does not place an order.

```text
Historical facts available at origin t
  sales, availability, prices, calendar, approved events
                         |
           as-of join + regular time grid
                         |
      window [t-L+1, ..., t] and validity masks
                         |
         +---------------+---------------+
         |               |               |
      RNN state       CNN filters    attention + position
         |               |               |
         +---------------+---------------+
                         |
             encoded historical context
                         |
       prediction head + known future covariates
                         |
       demand forecasts for t+1 ... t+H
                         |
      inventory, lead time, costs, constraints
                         |
          proposed order -> human approval
                         |
   observed outcomes -> monitoring and review
```

The forecast origin is the time at which a real decision could have been made.
The **lookback** is the number of historical buckets supplied to the model.
The **horizon** is the number of future buckets predicted.
The **lead time** is the time between ordering and receipt.
These quantities are related but not interchangeable.
A four-week horizon does not imply a four-week lead time.

### Neural-network essentials

A neural network is a parameterized function.
A layer multiplies inputs by learned weights, adds biases, and may apply a nonlinear activation.
Training adjusts the parameters to reduce a loss measured on historical examples.
Backpropagation computes how the loss changes with each parameter.
An optimizer updates parameters using those gradients.
A model can fit training data well and generalize poorly to unseen dates.
Validation data support design choices; a held-out test period estimates final prospective performance.
A loss is a training objective, while an operational metric evaluates the resulting decision.

### Three different information paths

A recurrent model reads the sequence in order and updates a state at each time step.
A temporal convolution combines values in local neighborhoods with shared filters.
A transformer encoder combines token representations using content-dependent attention weights.
Each architecture can be used as the history encoder for a direct multi-horizon forecast.
Each can also be misused by feeding it future information.

The simplest difference is not whether an architecture can represent a particular pattern in principle.
The practical differences are what patterns it represents easily, how much history it can exploit, and what it costs to train and operate.
The data and evaluation design determine whether any apparent difference is meaningful.

### Information is not causation

A learned association between a promotion flag and sales need not equal the causal effect of a promotion.
Promotions may be scheduled for products expected to sell well, or stock may be allocated in anticipation.
An attention weight is a model-internal mixing weight, not a causal attribution.
An architecture comparison cannot on its own justify a pricing or promotion intervention.

### One forecast, several decisions

The same forecast can be adequate for a weekly labor plan but inadequate for a long-lead purchase.
A point forecast can support a mean-demand budget but not fully specify a stockout-risk calculation.
A low overall error can conceal poor performance for high-margin or capacity-constrained products.
A good choice therefore begins with the decision and horizon, not a model leaderboard.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse or relevant unit |
|---|---|---|
| Series | Measurements for one entity over ordered time buckets. | A store–SKU series is not a single transaction. |
| SKU | Stock keeping unit, a product identifier within an organization. | A SKU code alone need not be stable after a product replacement. |
| Bucket | Interval to which observations are aggregated. | State the timezone and whether a week starts Monday. |
| Target | Quantity forecast, such as weekly units sold. | A forecast target is not necessarily a business target. |
| Demand | Quantity customers would request under the stated availability conditions. | Sales can be lower than demand during stockouts. |
| Origin | Last information cutoff used to issue a forecast. | A calendar date alone omits the cutoff time and timezone. |
| Lookback `L` | Number of past buckets in a model input window. | A 52-week lookback is not 52 independent examples. |
| Horizon `H` | Number of future buckets predicted from an origin. | A four-week horizon does not specify lead time. |
| Lead time | Duration from order placement to receipt. | It can be uncertain and can differ across suppliers. |
| Feature | Model input derived from information available at the origin. | A realized future promotion price is not automatically known. |
| Covariate | An explanatory input other than the target. | A correlated covariate is not necessarily causal. |
| Static feature | Attribute treated as constant within a forecast example. | Product category can change and must be versioned. |
| Known-future feature | Value available for a future bucket at forecast origin. | A promotion plan revised after the origin is unavailable then. |
| Observed feature | A measurement available only after the relevant event. | Realized future weather is not a known-future feature. |
| As-of join | Join using the latest version actually available by a cutoff. | Joining to the latest database value leaks revisions. |
| Vintage | Version of a plan or measurement as known at a specific time. | A current plan is not proof of a historical plan. |
| Window | A contiguous input segment and its target horizon. | Overlapping windows are not independent observations. |
| Encoder | Function mapping historical inputs to representations. | It need not be a transformer. |
| Decoder | Component generating outputs from a representation. | A direct forecast head need not be an autoregressive decoder. |
| Hidden state | Learned recurrent summary carried from one step to another. | It is not guaranteed to retain every earlier fact. |
| RNN | Recurrent neural network that updates state in time order. | A vanilla RNN is not synonymous with an LSTM. |
| LSTM | Long short-term memory recurrent architecture with gated cell state. | Gating mitigates but does not eliminate forgetting. |
| GRU | Gated recurrent unit with a simplified gated state update. | Fewer gates do not guarantee better accuracy. |
| CNN | Convolutional neural network applying shared filters over a grid. | A temporal CNN operates along time, not necessarily on images. |
| Kernel | Learned set of weights applied across nearby positions. | Its width is measured in time steps. |
| Dilation | Spacing between positions sampled by a kernel. | Dilation expands reach but can leave uncovered positions. |
| Receptive field | Set or span of input positions affecting an output. | Nominal span does not ensure every lag has influence. |
| Causal convolution | Convolution at time `t` that uses no input after `t`. | A full historical window can also be encoded noncausally at its final origin. |
| Padding | Extra values used to define outputs at boundaries. | Padding must not be mistaken for real zero demand. |
| Transformer | Architecture using attention and feed-forward layers. | A transformer encoder is not automatically an LLM. |
| Token | A position or patch represented as a vector for attention. | A token may represent one week or a multi-week patch. |
| Positional encoding | Signal conveying ordering or location to attention. | Calendar features do not always uniquely encode sequence order. |
| Query, key, value | Learned projections used to score and mix token representations. | Attention weights are not causal effects. |
| Attention mask | Rule excluding selected token pairs from attention. | A padding mask and a future mask solve different problems. |
| Autoregressive | Generating a future value conditional on previously generated values. | Direct multi-horizon prediction is not autoregressive decoding. |
| Global model | One model shared across multiple series. | Global does not mean geographically global. |
| Local model | Model fitted separately to each series. | It is not necessarily a simple model. |
| Loss | Numerical objective minimized during training. | Training loss is not automatically the business cost. |
| MAE | Mean absolute error, average absolute forecast error. | Its unit is the unit of the forecast target. |
| WAPE | Weighted absolute percentage error, sum absolute errors divided by sum actuals. | Undefined when the total actual is zero. |
| Bias | Mean forecast minus actual under the convention used here. | Always state sign convention. |
| Quantile | Value below which a specified fraction of a predictive distribution lies. | A 90th percentile is not a 90% prediction interval. |
| Coverage | Fraction of actuals falling within a stated interval. | Coverage without interval width rewards uninformative intervals. |
| Stockout censoring | Sales observation constrained by product availability. | A zero sale during a stockout is not proven zero demand. |
| Backtest | Simulated historical forecast evaluation using information available at each origin. | Randomly splitting rows is not a valid temporal backtest. |
| Leakage | Use of information unavailable when the forecast would have been issued. | Leakage includes preprocessing and revised plans, not just labels. |
| Drift | Change over time in data, relationship, or error behavior. | A one-week outlier alone does not establish drift. |

## 5. The mechanics, step by step

### Step 1 — Define a decision, not just a target

**Input:** the decision owner's order workflow, supplier terms, and a sample of historical orders.
**Action:** write down who acts, the origin, the horizon, the target unit, and the cost of being early or late.
**Output:** a one-page forecast contract.

For a Monday 07:00 weekly run, say whether Monday's sales are already complete.
If the most recent complete week ends Sunday, the input window ends on that Sunday.
A supplier with a two-week lead time may require forecasts for the first two weeks after the order.
A periodic review policy may additionally protect demand between receipt and the next review.
Document the exact protection period rather than equating it automatically to lead time.

**Branch:** if no decision changes when the forecast changes, stop model selection.
If a forecast feeds multiple decisions, define each decision's timing separately.
A daily labor decision and a monthly procurement decision can need different horizons and metrics.

### Step 2 — Establish the target and the grid

**Input:** transaction records, product and location identifiers, availability records, and calendar definitions.
**Action:** aggregate to a consistent series and time bucket.
**Output:** one row per series and bucket, with explicit units and availability status.

A weekly unit-sales target is the sum of valid transactions in the week.
Returns, cancellations, substitutions, and late-arriving transactions require a documented policy.
A missing row may mean no sale, a closed store, a missing feed, or an unlaunched product.
Those meanings must not be collapsed into one numeric zero.

Sales under an available shelf can be a useful proxy for demand.
Sales during a stockout are censored by supply and may understate desired demand.
The counterfactual demand is not directly observed in that interval.
Options include flagging or excluding censored target periods, estimating lost sales with a separately validated method, or explicitly modeling availability.
Each option changes the population and assumptions; report the choice.

**Branch:** if availability is absent and outages are frequent, limit the claim to sales forecasting.
If the decision requires unconstrained demand, quantify the risk before deployment.

### Step 3 — Build a historical information set

**Input:** event-time data plus the time each record or revision became available.
**Action:** reconstruct the latest permitted version of each feature at every historical origin.
**Output:** an as-of feature table and a lineage record.

Separate event time from availability time.
A Monday transaction posted on Wednesday cannot be used in Tuesday's run.
A promotion planned on January 5 and revised January 12 has two vintages.
A forecast issued January 8 may use the January 5 version, not the January 12 revision.
An as-of join must enforce both the entity key and the availability-time cutoff.

Keep a versioned calendar for holidays, planned closures, and approved campaigns.
Future calendar dates are normally known; future price and promotion plans are known only to the extent they were approved and accessible at the origin.
Realized competitor prices, actual weather, and future stockouts are not known in advance.
If using weather forecasts, archive the weather forecast vintage, not the weather that later occurred.

**Branch:** if historical vintages are unavailable, do not claim an as-of backtest for that feature.
Exclude it, reconstruct it from independent logs, or label the evaluation as optimistic and unsuitable for a final deployment decision.

### Step 4 — Establish simple benchmarks

**Input:** the same historical target and the same permissible information as the proposed system.
**Action:** implement persistence, seasonal naive where history permits, and a relevant existing production method.
**Output:** benchmark forecasts at every evaluation origin and horizon.

Persistence repeats the latest observed value.
Seasonal naive repeats the value from the comparable season, such as the same week a year earlier when calendar alignment is justified.
For intermittent demand, an intermittent-demand benchmark may be more informative than a dense-series benchmark.
An existing planner-adjusted forecast is a separate benchmark; retain its actual historical vintage.

Compare on identical series, dates, horizons, units, and exclusions.
A baseline that is not allowed to use the same known future calendar or promotion information can be an unfair comparison.
A baseline cannot always consume every feature in the same way; explain that design difference rather than hiding it.

**Branch:** if a simple method meets the required service and cost limits, retain it unless a more complex candidate earns its incremental cost.

### Step 5 — Form training examples

**Input:** a regularized grid, permissible historical features, and chosen lookback `L` and horizon `H`.
**Action:** create windows only where the required history and labels are valid.
**Output:** examples of shape `[batch, L, features]` and targets of shape `[batch, H]` for a direct forecast.

For a series of `N` complete buckets, the maximum number of stride-one windows is `N - L - H + 1` when this quantity is positive.
For `N = 104`, `L = 12`, and `H = 4`, the maximum is `89` windows.
This count assumes no invalid labels and no additional split restrictions.
Adjacent windows share most input values and target periods.
They are therefore highly correlated and must not be treated as 89 independent business events.

Historical missing values can be imputed using only past information available at the origin.
Provide a missingness indicator when the difference between observed and imputed values matters.
Do not interpolate a missing input using a later value that was unavailable at the origin.
A target loss mask can exclude invalid labels, but report how much data was excluded and why.

**Branch:** for new products with insufficient lookback, use a separately specified cold-start policy.
Do not silently fill the entire history with zeros and call it observed demand.

### Step 6 — Split by forecast origin and label availability

**Input:** the sequence of actual decision dates and the horizon.
**Action:** choose chronological train, validation, and final test periods before tuning.
**Output:** origin lists plus an auditable rule for which labels were available at each model fit.

At a model-fit time `T`, every training target used in the fit must have been observed by `T`.
For an `H`-bucket direct target, a training origin immediately before `T` can have labels that extend beyond `T`.
Exclude such examples until all required labels have arrived, or train with a carefully defined label mask.
Validation and test origins may use earlier actual observations as historical inputs when those observations would have been available operationally.
That is not leakage; using outcomes beyond the origin is leakage.

Use several rolling origins that represent seasonal and operational conditions.
For each origin, fit or update according to the actual planned retraining schedule.
Freeze hyperparameter selection before examining the final test results.
If repeated experimentation has consumed the test period, reserve a later untouched period or run a prospective shadow test.

**Branch:** if only one seasonal cycle exists, seasonal claims are weak.
Report this limitation rather than manufacturing independent samples from overlapping windows.

### Step 7 — Fit preprocessing on training data only

**Input:** the training portion of each fit and its feature definitions.
**Action:** estimate scaling and categorical mappings without validation or test observations.
**Output:** versioned preprocessing parameters fitted at the correct origin.

Standardization subtracts a training mean and divides by a training scale.
A scale of zero needs a defined fallback to avoid division by zero.
Per-series scaling can stop high-volume items from dominating optimization, but it may obscure absolute volume.
Global scaling preserves common units but can let large series dominate.
The choice affects the loss and must be compared on the original unit scale.

For an unseen product identifier, define an unknown-category representation.
If the product's future existence was unknown at the fit date, do not let its later category mapping leak into earlier experiments.
A time-varying product hierarchy requires historical mappings, not only today's hierarchy.

**Branch:** when a feature is intrinsically known in advance, such as a fixed holiday calendar, future values may be supplied only under an explicit origin-time availability rule.

### Step 8 — Understand the RNN encoder

**Input:** a chronological sequence of `L` feature vectors.
**Action:** update a hidden state one bucket at a time.
**Output:** a state, or sequence of states, summarizing the history for a forecast head.

A simple recurrent update is `h_t = tanh(W_x x_t + W_h h_{t-1} + b)`.
Here `x_t` is the current input vector, `h_t` is the hidden state, and the `W` matrices and `b` are learned parameters.
The same update weights are used at each time step.
A prediction head can map the final state `h_t` to all `H` future values at once.

A vanilla RNN may struggle to learn long dependencies because gradients can shrink or grow as they pass through many recurrent updates.
A long short-term memory network (LSTM) introduces a cell state and input, forget, and output gates.
A gated recurrent unit (GRU) uses a different gated update with fewer gate types.
Gates control information flow; they do not prove the model remembers all relevant events.
Gradient clipping can limit exploding updates, while suitable lookbacks and validation help expose forgetting.

Sequential recurrence limits parallelism across the time dimension during training.
It can be attractive for streaming updates if state management is correct.
At an entity boundary, state must be reset or explicitly associated with that entity.
A bidirectional RNN over a wholly historical input window can be valid at the final origin.
It is invalid for an intermediate-time prediction if its backward pass sees information after that intermediate origin.

**Branch:** start with a small recurrent model if histories are moderate, streaming state is valuable, and a global dataset is sufficiently rich.
Retain it only if the same-origin comparison justifies its costs.

### Step 9 — Understand the temporal CNN encoder

**Input:** a sequence arranged as channels by time.
**Action:** apply learned kernels across time and stack layers.
**Output:** a representation whose receptive field depends on kernel widths, dilation, and stride.

A width-three causal kernel at time `t` can combine inputs at `t`, `t-1`, and `t-2`.
Weights are shared across positions, so a learned local pattern can be recognized at many dates.
A second width-three layer with dilation two looks at layer-one positions separated by two buckets.
With stride one, two layers of widths three and dilations one and two have nominal receptive-field span `1 + 2*1 + 2*2 = 7` buckets.
The network can therefore connect the final output to inputs from `t-6` through `t` in this configuration.

More generally, with stride one, receptive-field span is `1 + sum((k_i - 1) * d_i)` for kernel width `k_i` and dilation `d_i` at each layer.
With strides greater than one, multiply each contribution by the product of preceding strides.
The exact set of reachable lags can have gaps for some dilation choices.
Inspect the actual connectivity rather than relying only on the span.

A causal convolution is necessary when a prediction is made at each time step using only information available then.
A convolution over a complete historical window may use both directions within that window for a single prediction issued at its end.
In either case, no value after the forecast origin is permitted.
Padding at a boundary is a computational convenience, not an observed zero-sale week.
A validity mask or carefully chosen boundary treatment prevents confusion.

CNN layers typically parallelize across positions more readily than recurrence.
Their local bias can be useful for recurring short-term shapes and event responses.
A seasonal lag outside the receptive field cannot affect a final-position representation through that convolutional path.
A global pooling layer changes the paths by aggregating information from more positions; describe the actual architecture before making a reach claim.

**Branch:** a temporal CNN is a useful candidate when local patterns and throughput matter, provided its receptive field covers the lags the decision requires.

### Step 10 — Understand the transformer encoder

**Input:** token representations for a permitted historical window, plus positional information and masks.
**Action:** calculate attention-based mixtures and feed-forward transformations in stacked layers.
**Output:** context-dependent representations used by a forecast head.

For a token matrix `X`, learned projections produce queries `Q`, keys `K`, and values `V`.
A standard attention head computes `softmax(QK^T / sqrt(d_k)) V`, with appropriate masks.
The score for a query–key pair influences how much of that value contributes to the output.
Multiple heads can learn different projections and interaction patterns.
Residual connections, normalization, and feed-forward layers are common parts of the block.

Without a positional signal or another ordering mechanism, standard self-attention alone is permutation-equivariant.
It does not inherently know which observed week came first.
Positional embeddings, relative position methods, calendar features, or explicit structure supply ordering information in different ways.
Calendar features can help represent seasonality but are not automatically a complete positional encoding.

An encoder can attend bidirectionally across the historical window because all those tokens are known at the origin.
Use a padding mask for absent tokens.
A causal mask is required when a representation at an earlier position must not see later positions, such as autoregressive training or intermediate forecasts.
For direct prediction from the final history, a future mask is not necessary if future observations are not present in the input.
Known future covariates may be added through a properly designed forecast head or decoder with careful masking.

Full attention creates an `L` by `L` score matrix per head, so its attention-memory and score-computation cost scale quadratically with `L` in the basic implementation.
Actual runtime also depends on embedding width, batch size, implementation, and hardware.
Patching multiple adjacent time steps into one token can reduce token count, at the cost of a different representation and potential loss of fine timing detail.
A longer lookback is not automatically useful or affordable.

**Branch:** consider an encoder when cross-time interactions and sufficient shared data justify the additional computation.
A modest CNN or RNN can still outperform it on a given dataset and operating budget.

### Step 11 — Choose the forecast head and objective

**Input:** encoded history, the decision horizon, and the costs of forecast errors.
**Action:** choose direct or recursive prediction and a loss aligned with the intended output.
**Output:** a forecast with documented semantics and units.

A direct `H`-output head predicts all horizons in one pass.
It avoids feeding predicted values back into later predictions, but each horizon must have enough training signal.
A recursive model predicts one step and reuses its own predictions for later steps.
Recursive generation can accumulate error and creates a training–inference mismatch if training always uses actual prior values.
A multi-output head is not automatically a coherent joint probabilistic distribution.

Mean squared error favors a conditional mean under idealized assumptions.
Mean absolute error favors a conditional median under idealized assumptions.
Quantile loss for quantile `q` is `max(q*(y - yhat), (q-1)*(y - yhat))`.
A high quantile may be useful for a service-sensitive order, but it is not automatically the right order quantity.
A constrained order requires inventory position, lead time, cost, capacity, and business policy.

A nonnegative output transformation can enforce nonnegative forecasts if the target is intrinsically nonnegative.
That constraint does not solve intermittent demand or missing sales.
If using intervals, evaluate coverage and width by horizon, segment, and origin.
A set of marginal per-week quantiles cannot generally be summed to obtain the same quantile of cumulative lead-time demand.
Dependence across weeks matters; use a validated joint distribution or calibrated lead-time-demand estimate where required.

**Branch:** when the operational decision uses a lead-time total, directly score that total as well as individual weekly buckets.

### Step 12 — Train and control capacity

**Input:** training examples, a candidate architecture, loss, preprocessing, and budget.
**Action:** optimize on the training set and select settings on validation origins only.
**Output:** model artifact, training configuration, and validation record.

Set a reproducible random seed but do not assume identical results across all hardware and library versions.
Use mini-batches with masks for variable-length sequences if needed.
Early stopping can use validation loss, with the stopping rule fixed before the final test.
Regularization, model width, lookback, and learning rate affect both fit and cost.
Small data can favor a simpler architecture even when a larger model has more expressive capacity.

Training across related series can create many examples, but common shocks and overlapping windows reduce effective independence.
Ensure the training population matches the deployment population or measure the resulting domain shift.
Record library versions, training snapshot, feature definitions, random seed, and model checkpoint.
Compare a reasonable range of settings for each family within a stated search budget.
A single badly tuned baseline is not a fair reference.

**Branch:** if results vary materially across seeds, report the variation and use a predeclared selection rule.
Do not choose the best final-test seed after seeing its result.

### Step 13 — Backtest in operational time

**Input:** frozen candidates, the rolling origin schedule, and as-of datasets.
**Action:** issue predictions as the historical process would have done and preserve each output.
**Output:** one forecast record per series, origin, horizon, model version, and target bucket.

A fixed model can be evaluated across multiple origins if that is the planned update policy.
If production retrains monthly, the backtest should simulate monthly retraining and intervening inference.
Keep validation choices and the final test separate.
For each origin, use only labels observed by the model-fit cutoff.
Store forecast creation time as well as the target date.

Compare paired forecast errors on the same cases.
Summaries by product, store, volume, promotion, availability, and horizon reveal cancellation hidden by an overall average.
When estimating uncertainty for a difference, account for clustering and serial dependence; for example, resample meaningful time blocks or independent business units when justified.
A confidence interval is conditional on the sampling design and historical regime.

**Branch:** if a launch or disruption was never represented in historical data, add a stress scenario and restrict generalization claims.

### Step 14 — Translate accuracy into business value

**Input:** paired predictions, actuals, a candidate replenishment rule, and cost assumptions.
**Action:** apply the same decision policy and inventory constraints to each forecast.
**Output:** accuracy, service, cost, and operating burden on a comparable basis.

MAE is the average absolute error in units per forecasted bucket.
WAPE is `sum(abs(forecast - actual)) / sum(actual)` when total actual demand is positive.
Bias is `mean(forecast - actual)` under this topic's convention.
High-volume series contribute more to WAPE; low-volume segments still need separate review.
A zero denominator makes WAPE undefined, not zero.

A historical decision simulation needs starting inventory, open orders, lead-time assumptions, and a consistent treatment of stockouts.
If a model would have changed availability, observed sales alone may not reveal the counterfactual demand.
Use scenarios or a validated lost-sales model and disclose their assumptions.
Avoid presenting a simulation as measured incremental profit.

Add annual training, inference, data engineering, monitoring, support, and exception-handling costs.
A model that saves more units on paper but misses the order cutoff is not a viable candidate.
Include the opportunity cost of planner intervention and the fallback's operational burden.

**Branch:** adopt only when the measured and stress-tested benefit exceeds incremental cost and satisfies service, latency, and governance constraints.

### Step 15 — Deploy with a bounded control loop

**Input:** selected candidate, fallback, data contracts, and owners.
**Action:** run in shadow mode, compare outcomes, then authorize a limited decision scope.
**Output:** versioned predictions, approval records, monitoring, and rollback criteria.

At each run, validate the input freshness, entity coverage, units, and feature availability.
Record the model version, preprocessing version, data snapshot, origin, and horizon.
If a critical feed is late or out of range, use the predefined fallback and alert an owner.
Do not substitute a future value or an unreviewed imputation to keep a dashboard green.

Monitor errors after labels mature, broken down by horizon and decision-relevant segments.
Monitor input drift separately from outcome degradation.
Inspect intervals for calibration and width, not merely the point forecast.
A planner override should record who changed what, when, and why, and should be evaluated against the unadjusted forecast when outcomes arrive.
Retraining is a controlled change, not a cure for every data-quality problem.

**Branch:** if the candidate fails an agreed service, cost, data-quality, or latency threshold, revert to the validated fallback and investigate before re-entry.

## 6. Worked example

### 6.1 Decision and assumptions

A fictional regional distributor orders a packaged food item every Monday.
The supplier's assumed lead time is two weeks.
For this illustration, the order protects exactly the next two weekly buckets.
A real periodic-review policy may need a longer protection period; this simplified policy does not.
All quantities below are units of one item at one distribution center.
All currency amounts are US dollars (USD).

The point forecast is an estimate of demand, not the order.
The order is a plan until the buyer approves it.
The service target is 95% fill rate; it is not a forecast or a supplier commitment.
The supplier commitment is the agreed receipt date and accepted order quantity.

Assume the inventory position at the origin is 184 units.
Inventory position here means on-hand plus confirmed inbound minus backorders.
For simplicity, assume no minimum order quantity, pack rounding, capacity limit, or shelf-life restriction.
Use a fixed safety allowance of 25 units for all candidate forecasts.
The candidate forecasts and outcomes are illustrative, not empirical results.

### 6.2 The two forecast candidates

A seasonal-naive benchmark forecasts 92 units in week 1 and 104 units in week 2.
A shared temporal CNN forecasts 99 units in week 1 and 110 units in week 2.
The actual observed demand proxy is 101 units in week 1 and 108 units in week 2.
Assume availability was complete so observed sales equal demand for this illustration.
The comparison is paired: both forecasts were issued at the same origin for the same item and weeks.

| Quantity | Week 1 | Week 2 | Two-week sum |
|---|---:|---:|---:|
| Seasonal-naive forecast | 92 units | 104 units | 196 units |
| CNN forecast | 99 units | 110 units | 209 units |
| Actual demand | 101 units | 108 units | 209 units |

For the benchmark, absolute errors are `|92 - 101| = 9` units and `|104 - 108| = 4` units.
Its MAE is `(9 + 4) / 2 = 6.5` units per week.
Its WAPE is `13 / 209 = 6.22%`, rounded to two decimals.
Its signed bias is `(92 - 101 + 104 - 108) / 2 = -6.5` units per week.
Under the declared sign convention, negative bias indicates underforecasting.

For the CNN, absolute errors are `|99 - 101| = 2` units and `|110 - 108| = 2` units.
Its MAE is `(2 + 2) / 2 = 2.0` units per week.
Its WAPE is `4 / 209 = 1.91%`, rounded to two decimals.
Its signed bias is `(99 - 101 + 110 - 108) / 2 = 0` units per week.
Zero aggregate bias does not mean both weekly forecasts were exact.

### 6.3 Calculate the illustrative order

Use the same policy for both candidates:

`proposed order = max(0, two-week forecast + 25 units - 184 units)`.

The benchmark proposes `max(0, 196 + 25 - 184) = 37` units.
The CNN proposes `max(0, 209 + 25 - 184) = 50` units.
The CNN therefore recommends 13 more units under this policy.
Neither forecast alone authorizes the purchase.

The actual two-week demand was 209 units.
Under the simplified coverage accounting, benchmark position after demand is `184 + 37 - 209 = 12` units.
The CNN position after demand is `184 + 50 - 209 = 25` units.
These numbers ignore the timing of receipts and intra-period demand.
They do not prove that the 95% fill-rate target was met.
A real simulation must track receipt and demand events in sequence.

### 6.4 Evaluate a wider hypothetical portfolio

Suppose a paired backtest across 40 origins and 800 series yields the following predeclared results.
These values are assumed for illustration, not generated by the code in section 7.
They represent outcomes from the same forecast origins, data vintages, and eligible series.

| Measure | Seasonal naive | CNN | Interpretation |
|---|---:|---:|---|
| WAPE | 24.0% | 21.6% | CNN is lower by 2.4 percentage points. |
| Relative WAPE reduction | reference | 10.0% | `2.4 / 24.0`; not 10 percentage points. |
| Two-week demand MAE | 31.0 units | 28.5 units | Aggregate across the defined evaluation cases. |
| Estimated annual decision savings | USD 0 | USD 74,000 | Scenario estimate against benchmark, not audited realized profit. |
| Incremental annual operating cost | USD 0 | USD 29,000 | Training, inference, data, and support. |
| Incremental net value | USD 0 | USD 45,000 | `74,000 - 29,000`, before unmodeled effects. |
| Forecast delivery | 06:20 | 06:42 | Both satisfy the 07:00 deadline in this scenario. |

The savings estimate depends on the demand reconstruction and replenishment simulation.
Assume a block-based uncertainty analysis yields a wide plausible savings range of USD 20,000–110,000 per year.
This interval is hypothetical and depends on the chosen resampling method.
At the lower bound, savings minus the USD 29,000 incremental cost is a USD 9,000 loss per year.
At the upper bound, net value is USD 81,000 per year.
A positive central estimate is therefore not enough to ignore downside risk.

Assume the service simulation also shows a deterioration for a small, high-criticality segment.
A portfolio-level WAPE improvement does not cancel that segment's risk.
The buyer could retain the benchmark for that segment and run a controlled shadow trial elsewhere.
This recommendation is conditional on predeclared segmentation, not selected after observing the final test without further validation.
The decision owner should request a later prospective period if this split was discovered during final evaluation.

### 6.5 Compare the other architecture families

An RNN with a 26-week input may provide a low-latency shared baseline for this portfolio.
A CNN with a seven-week receptive field cannot use a 13-week lag through its final convolutional output alone.
The design can increase dilation or depth, or add an explicit permitted lag feature.
A transformer encoder with 52 historical weekly tokens can compare distant positions directly.
It must still carry positional information, respect the as-of boundary, and meet the same run deadline.
None of these architectural facts establishes that one method will win the distributor's backtest.

### 6.6 Final business decision

On the hypothetical evidence, do not roll out the CNN universally.
Run a shadow trial with a fixed policy and an approved fallback.
Protect the critical segment until a separate, prospective evaluation supports a change.
Measure lead-time demand error, fill rate, stockouts, inventory, and total operating cost.
Revisit the decision when outcomes mature and the savings uncertainty narrows.

## 7. Code, query, or calculation

The following standalone Python example trains three small global encoders on synthetic weekly series.
It uses a direct two-week forecast, a training-only scaler, a chronological cutoff, and a seasonal-naive comparator.
It demonstrates data and evaluation mechanics; it is not evidence that any architecture wins in production.
The forecast is in units per weekly bucket.
The synthetic dataset has no missing values, stockouts, future covariates, or product launches.
The held-out origins are a final demonstration set; a real project needs separate validation origins for architecture and hyperparameter selection.

Install a compatible current Python 3, NumPy, and PyTorch environment before running.
Save this block as a local Python file and execute it.
No network access, files, GPU, or external data are required at runtime.
The numerical metrics vary by platform and PyTorch version.
Expected output is four lines, one for `seasonal_naive` and one for each encoder, each containing a finite WAPE percentage and MAE in units.
There is no promised ranking or exact numerical value.

```python
import math
import numpy as np
import torch
from torch import nn

# Synthetic series: 24 related products, 120 weekly observations each.
np.random.seed(7)
torch.manual_seed(7)
torch.set_num_threads(1)
n_series, n_weeks = 24, 120
lookback, horizon, season = 16, 2, 12
train_end = 88  # First unavailable training-label index.
t = np.arange(n_weeks, dtype=np.float32)
series = []
for i in range(n_series):
    level = 25.0 + 1.3 * i
    wave = (4.0 + i / 12.0) * np.sin(2 * np.pi * (t + i % 4) / season)
    noise = np.random.normal(0.0, 1.5, n_weeks)
    series.append(np.maximum(0.0, level + wave + noise))
y = np.asarray(series, dtype=np.float32)

# Fit preprocessing only on values available by the training cutoff.
mean = float(y[:, :train_end].mean())
scale = float(y[:, :train_end].std())
scale = scale if scale > 1e-8 else 1.0
z = (y - mean) / scale

def make_windows(origins):
    xs, ys = [], []
    for s in range(n_series):
        for origin in origins:
            # origin is the first target index; all input indices are earlier.
            xs.append(z[s, origin - lookback:origin, None])
            ys.append(z[s, origin:origin + horizon])
    return torch.tensor(np.asarray(xs)), torch.tensor(np.asarray(ys))

# Last training target index is at most train_end - 1.
train_origins = range(lookback, train_end - horizon + 1)
# Evaluation begins after the cutoff; later origins use then-observed history.
test_origins = range(train_end, n_weeks - horizon + 1, horizon)
x_train, y_train = make_windows(train_origins)
x_test, y_test = make_windows(test_origins)

class RNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.GRU(input_size=1, hidden_size=16, batch_first=True)
        self.head = nn.Linear(16, horizon)

    def forward(self, x):
        _, h = self.encoder(x)
        return self.head(h[-1])

class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # No padding: all positions are inside the historical input window.
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3), nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=3, dilation=2), nn.ReLU(),
        )
        self.head = nn.Linear(16, horizon)

    def forward(self, x):
        h = self.encoder(x.transpose(1, 2))
        return self.head(h[:, :, -1])

class TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        width = 16
        self.project = nn.Linear(1, width)
        self.position = nn.Parameter(torch.zeros(1, lookback, width))
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=4, dim_feedforward=32,
            dropout=0.0, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(width, horizon)

    def forward(self, x):
        # Bidirectional attention is safe: x contains only past observations.
        h = self.encoder(self.project(x) + self.position)
        return self.head(h[:, -1, :])

def score(name, prediction, actual):
    error = np.abs(prediction - actual)
    denom = np.abs(actual).sum()
    wape = 100.0 * error.sum() / denom if denom > 0 else math.nan
    print(f"{name:16s} WAPE={wape:6.2f}% MAE={error.mean():6.2f} units")

actual = y_test.numpy() * scale + mean
# Seasonal-naive values are available before each test origin because season >= horizon.
benchmark = []
for s in range(n_series):
    for origin in test_origins:
        benchmark.append(y[s, origin - season:origin - season + horizon])
score("seasonal_naive", np.asarray(benchmark), actual)

for name, cls in [("rnn", RNNEncoder), ("cnn", CNNEncoder),
                  ("transformer", TransformerEncoder)]:
    model = cls()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    model.train()
    for _ in range(35):
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(x_train), y_train)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        pred = model(x_test).numpy() * scale + mean
    score(name, pred, actual)
```

### How to read the artifact

The first test origin is index 88, and its inputs end at index 87.
The final training origin has its last target at index 87.
No training label at index 88 or later enters the fit.
The models are fixed during the held-out period, while later test windows use observations that would already be known at those origins.
The baseline uses a 12-week lag for both predicted weeks, which is available before the origin.
The CNN's final output has a seven-week receptive field, despite receiving a 16-week input tensor.
The transformer uses learned positional vectors and attends only within the historical window.
The code intentionally omits a validation loop and is therefore a mechanics demonstration, not an architecture-selection experiment.

### What must change for an operational evaluation

Use actual as-of vintages and a calendar matching the order workflow.
Add separate chronological validation origins before the final test period.
Tune each candidate under a stated comparable budget.
Handle missingness, stockouts, new items, future covariates, and product hierarchy changes explicitly.
Report paired errors by horizon and segment, uncertainty, runtime, and total operating cost.
Do not infer an inventory service level from the printed MAE or WAPE.

## 8. Data, standards, and regulatory context

### 8.1 Data contract

A sequence model needs more than a rectangular numeric table.
The data contract should specify the entity key, bucket boundary, unit, and time zone.
It should state when each source is expected and how late arrivals are revised.
It should define whether zero denotes an observed zero or a placeholder.
It should identify the source of inventory availability and whether returns are netted.
It should specify which future features are published before the origin.
It should record how product substitutions and store openings are represented.

A product and location hierarchy is often revised after organizational changes.
Use the hierarchy as it was known at the origin for historical decisions.
Separately report restated hierarchy results if the business needs today's reporting structure.
Mixing the two silently can create false apparent gains or losses.

### 8.2 Identifiers and interoperability

GS1's Global Trade Item Number (GTIN) identifies trade items within the GS1 system.
The GS1 General Specifications describe rules for identifiers and barcodes.
A GTIN can help join records across organizations, but it does not by itself define the internal SKU, forecasting bucket, demand signal, or product substitution history.
The same trade item can appear in several locations, pack levels, and channels.
A reliable forecasting key must include the relevant location and other business dimensions.

Electronic Data Interchange (EDI) messages can carry orders, forecasts, and shipment events between partners.
The exact message, implementation guide, trading-partner agreement, and revision determine field semantics.
An EDI transmission timestamp is not necessarily the physical event time.
Standards improve interoperability; they do not guarantee accurate or timely source data.

### 8.3 Privacy and access

Aggregated item–store demand often contains no direct personal identifier, but a dataset can still become personal data when linked to identifiable individuals.
Delivery addresses, driver identifiers, employee schedules, and customer-level transactions require separate review.
Apply data minimization, role-based access, retention controls, and documented purpose to the actual dataset.
Hashing an identifier does not automatically make linked data anonymous.
Do not export supplier-confidential or personal data to a hosted training service without approved contractual and security controls.

The EU General Data Protection Regulation (GDPR), Regulation (EU) 2016/679, applies when its territorial and material conditions are met.
Its principles include purpose limitation, data minimization, accuracy, storage limitation, and integrity and confidentiality.
It does not mandate a particular forecasting architecture.
Applicability depends on the data, processing, and jurisdiction; seek qualified advice.

### 8.4 AI governance and law

The EU AI Act, Regulation (EU) 2024/1689, uses a risk-based framework and has staged application dates.
As of the regulation's published text, obligations depend on the system, actor, use case, and applicable date; a generic demand forecaster is not automatically a high-risk system merely because it uses a transformer.
Check the current legal text, amendments, and authoritative guidance for the actual deployment.
This educational discussion is not a legal determination.

NIST's AI Risk Management Framework 1.0 is voluntary US guidance, not a law.
Its govern, map, measure, and manage functions help organize risk ownership and evidence.
It does not certify a model or prescribe one architecture.
A forecasting team can use it to record intended use, failure modes, measurement, and ongoing controls.

### 8.5 Reproducibility and retention

Record source identifiers, extraction time, cutoff, feature version, and model artifact for each forecast run.
Retain historical plan vintages if the organization needs auditable backtests.
Restrict access according to contractual and legal requirements.
Define retention and deletion schedules before collecting sensitive data.
A reproducible run should reconstruct the information available at the actual decision time, not today's corrected database.

## 9. Industry practice and case evidence

### 9.1 Public evidence is not local proof

A public research result demonstrates performance under its stated data and evaluation protocol.
A production case may describe a system without publishing a controlled architecture comparison.
Neither automatically establishes savings for a different product mix or supplier network.
Distinguish peer-reviewed research, an official dataset, and a company-authored report.

### 9.2 Walmart and the M5 forecasting competition

The M5 Accuracy competition used Walmart unit sales data, with hierarchical product and store series and explanatory information including calendar and price data.
The competition's published description and the M5 paper document the dataset and evaluation design.
The benchmark is useful for studying retail demand, aggregation, and model comparison.
It is not a randomized study of one enterprise's end-to-end inventory savings from a particular encoder.
Public familiarity with the dataset also creates a contamination risk for subsequent pretrained systems.

**Inference:** a distributor should use M5 as a reproducible research starting point, then evaluate on its own previously unseen data vintages and decision rules.
A high public leaderboard position alone does not establish local value.

### 9.3 Amazon and DeepAR

The DeepAR paper by Salinas and colleagues describes an autoregressive recurrent probabilistic forecasting method trained across related time series.
It discusses retail demand forecasting as one application and evaluates on multiple datasets.
Its recurrent architecture and global training are useful reference points for understanding shared sequence learning.
The paper's reported results are conditional on its datasets and experimental choices.
They are not evidence that every recurrent model is superior to a tuned local model.

**Inference:** a retailer with many related sparse series can test shared learning, but should include a suitable intermittent-demand baseline and examine stockout censoring.
DeepAR is a specific probabilistic autoregressive model; a generic GRU point forecaster is not automatically DeepAR.

### 9.4 Temporal convolution research

Bai, Kolter, and Koltun's empirical study compares generic convolutional and recurrent sequence models across a range of sequence tasks.
It argues that temporal convolution should be considered as a natural starting point for sequence modeling.
The paper does not establish that a particular seven-week convolution is optimal for retail replenishment.
Receptive field, data, tuning, and forecast objective remain case-specific.

**Inference:** compare a well-designed temporal CNN with a recurrent model rather than treating architecture popularity as evidence.
Include the computation and operational costs of the full pipeline.

### 9.5 Transformer forecasting evidence

The original Transformer paper introduces attention-based sequence modeling, not a universal proof of time-series superiority.
The PatchTST paper studies patch-based transformer methods for long-term time-series forecasting.
Its reported benchmark results depend on its datasets and protocol.
A patch-based model is not identical to every transformer encoder.
A public benchmark result should not be interpreted as a guarantee of procurement savings.

**Inference:** attention-based encoders are candidates when long historical interactions and shared data make their additional complexity plausible.
They should face the same origin-time controls and cost accounting as simpler alternatives.

### 9.6 How an organization should use external evidence

Start with a cited claim and identify its population, target, horizon, baseline, and metric.
Ask whether the result is independent research, a company report, or a vendor assertion.
Record whether source data could have appeared in model pretraining.
Replicate the relevant protocol only where the required data and permissions exist.
Then run a prospective or genuinely held-out local comparison.
An architecture paper is a hypothesis generator, not an approval to automate purchasing.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Future plan leakage | Backtest uses today's promotion table. | Later revisions enter earlier origins. | Inflated accuracy. | Archive and as-of join plan vintages. |
| Label overlap at fit | Training targets extend into test dates. | Horizon crosses model-fit cutoff. | Future actuals influence weights. | Restrict labels to those observed by fit time. |
| Full-data scaling | Scaler statistics change when test is added. | Future distribution affects preprocessing. | Optimistic evaluation. | Fit scalers separately on each training fit. |
| Random window split | Near-identical windows appear in both sets. | Overlapping inputs and labels cross the split. | False confidence. | Split by origin and respect label availability. |
| Hidden stockouts | Low sales coincide with zero stock. | Sales are capped below demand. | Under-ordering and misleading labels. | Flag censoring and validate a demand treatment. |
| Missing equals zero | Many zeros appear when feeds are late. | Missing records become observed zeros. | Biased forecasts. | Distinguish missingness and use masks. |
| Short CNN reach | Seasonal lag exceeds receptive field. | No information path from lag to output. | Missed seasonal signal. | Adjust architecture or add permitted lag features. |
| Transformer without order | Token permutation barely changes output. | Attention lacks a temporal position signal. | Misread temporal pattern. | Add and audit positional representation. |
| Wrong mask | Earlier token representation sees future actuals. | Causal restriction omitted where needed. | Leakage in intermediate forecasts. | Test masks against each output's information set. |
| Cross-series RNN state | Forecasts change with batch entity order. | Hidden state carries across products. | Contaminated predictions. | Reset or key state by entity. |
| Future weather used | Weather feature matches realized weather perfectly. | Actual future measurements used in history. | Unrealistic benefit. | Archive forecast-vintage weather. |
| Unfair benchmark | Candidate has promotion inputs but baseline cannot use any. | Different information or tuning budgets. | False architectural advantage. | Equalize information where feasible and disclose differences. |
| Censoring after deployment | Service drops while sales error looks low. | Unavailable stock suppresses actual sales. | Failure hidden by target choice. | Monitor availability and demand proxies jointly. |
| Attention as explanation | Planner is told a weight proves causality. | Mixing weights are interpreted as intervention effects. | Unsupported business action. | Use causal design for intervention claims. |
| Aggregate-only reporting | Portfolio improves while critical segment worsens. | Large series dominate pooled metrics. | Local stockouts or inequity. | Report predeclared segments and guardrails. |
| Interval misuse | Weekly 90th percentiles are simply summed. | Dependence between weeks is ignored. | Misstated protection level. | Model or calibrate lead-time total directly. |
| Cold-start collapse | New products receive zero-history forecasts. | Unseen entities and absent lags are mishandled. | Lost sales at launch. | Use an explicit cold-start policy and monitor it. |
| Runtime overrun | Model finishes after supplier cutoff. | Inference or data pipeline exceeds budget. | No actionable order. | Measure end-to-end latency and maintain fallback. |
| Cost omission | Accuracy improves but infrastructure charges rise. | Only model error is valued. | Negative net benefit. | Include support, training, and inference costs. |
| Final-test tuning | Repeated changes follow test results. | Test information becomes part of model selection. | Biased final estimate. | Reserve a later test or prospective shadow period. |
| Hierarchy revision leakage | Historical items are mapped using today's category. | Future reclassification enters old examples. | Misleading segment performance. | Version hierarchy mappings by availability time. |
| Extrapolation failure | Forecasts fail during a regime change. | Training relationships no longer hold. | Large decision errors. | Stress test, monitor, and trigger controlled fallback. |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or respond |
|---|---|---|---|
| Historical sales proxy demand. | Defines the forecast label. | Stockouts or rationing censor sales. | Join availability; report censored share and sensitivity. |
| Inputs are available at origin. | Makes backtest operationally feasible. | Late feeds or revised plans. | Audit availability timestamps and vintages. |
| Buckets are consistently defined. | Makes lags and horizons meaningful. | Timezone changes or calendar restatement. | Reconcile bucket boundaries against source events. |
| Related series share useful patterns. | Justifies a global model. | Product mix or channel behavior differs materially. | Compare segments and local baselines. |
| Training history represents deployment. | Supports generalization. | Launches, policy changes, or disruptions. | Run time-sliced tests and stress scenarios. |
| Labels are mature at model fit. | Prevents future information entering weights. | Delayed transactions or multi-step labels. | Enforce fit-time label cutoffs. |
| Feature definitions remain stable. | Preserves learned relationships. | Unit, source, or business-rule changes. | Version contracts and monitor schema and distributions. |
| Evaluation cases reflect decisions. | Connects metrics to business value. | Missing critical segments or wrong horizon. | Check coverage against the decision population. |
| Costs in simulation are plausible. | Supports an economic recommendation. | Unknown shortage penalties or omitted labor. | Vary cost assumptions and report break-even values. |
| Inventory policy is held comparable. | Isolates forecast contribution. | Different constraints or overrides by model. | Replay the same rule and record exceptions. |
| Lead-time demand can be reconstructed. | Enables service and inventory analysis. | Unobserved lost sales and variable lead time. | Use validated estimates and report uncertainty. |
| Sample dependence is addressed. | Makes uncertainty estimates interpretable. | Overlapping windows and common shocks. | Use suitable blocks or clusters and disclose limitations. |
| The architecture can meet the cutoff. | Makes forecasts actionable. | Hardware contention or slow joins. | Measure end-to-end runtime in representative operations. |
| Access and use are authorized. | Protects personal and commercial data. | Undocumented sharing or changing legal scope. | Review contracts, jurisdiction, retention, and permissions. |

A strong historical result does not eliminate these assumptions.
The correct response to an untestable assumption is to narrow the claim and design a way to collect better evidence.

## 12. Exercises

### Exercise 1 — Reconstruct the origin-time information set

**Time:** 15 minutes.  
**Situation:** A forecast is issued Monday at 07:00 for the next four weeks.
Sunday sales arrive at 06:30; a promotion plan is approved Sunday at 18:00 and revised Monday at 08:00.
Actual weather for the next week is observed only after the week ends.

**Deliverable:** a four-row table identifying whether each item is permitted at the origin and why.  
**Acceptance:** distinguish the approved plan from the later revision, and distinguish a weather forecast vintage from actual future weather.

### Exercise 2 — Windows and convolution reach

**Time:** 15 minutes.  
A series has 104 complete weekly buckets, lookback 12 weeks, horizon 4 weeks, and stride one.
A two-layer stride-one causal CNN has kernel widths three and three, with dilations one and two.

**Deliverable:** the maximum window count and the nominal receptive-field span, with arithmetic.  
**Acceptance:** state assumptions about completeness and identify the oldest lag reachable from the last output.

### Exercise 3 — Compare information paths

**Time:** 20 minutes.  
A buyer asks whether a transformer always remembers 52 weeks better than an RNN or CNN.
Each candidate receives a 52-week historical input.

**Deliverable:** a comparison of recurrent state, convolutional reach, and attention with position and masks.  
**Acceptance:** distinguish theoretical access from useful learned influence and explain why benchmark results decide the choice.

### Exercise 4 — Calculate forecast and order quantities

**Time:** 20 minutes.  
A two-week demand forecast is 87 and 103 units; actual demand is 91 and 98 units.
Inventory position is 160 units and the policy adds 20 units of safety allowance.

**Deliverable:** MAE, WAPE, signed bias using forecast minus actual, and the proposed order.  
**Acceptance:** show arithmetic, state units, and distinguish the proposed order from an approved commitment.

### Exercise 5 — Design a rolling backtest

**Time:** 25 minutes.  
A weekly model forecasts four weeks ahead and retrains monthly.
Promotions are frequently revised after their first approval.

**Deliverable:** an origin schedule, fit-label cutoff, feature-vintage rule, baseline, validation/test separation, and two segment reports.  
**Acceptance:** no training label or feature may use information unavailable at the corresponding fit or forecast origin.

### Exercise 6 — Value a candidate

**Time:** 20 minutes.  
A model is estimated to save USD 80,000 per year before operating costs.
Incremental annual costs are USD 35,000, while the plausible annual savings range is USD 25,000–110,000.
The model misses the deadline on 4% of runs; the agreed maximum is 1%.

**Deliverable:** central and range-bound net values plus a deployment recommendation.  
**Acceptance:** calculate the net values and treat the deadline as a binding operational guardrail rather than averaging it away.

### Exercise 7 — Diagnose a misleading result

**Time:** 25 minutes.  
A retailer reports an apparent 15% reduction in WAPE, but its test table uses revised promotions and contains weeks when inventory was zero.
A dashboard claims the highest attention weight proves a promotion caused demand.

**Deliverable:** three distinct issues, each with mechanism, consequence, and corrective evidence.  
**Acceptance:** cover vintage leakage, censored labels, and the difference between association and causation.

### Exercise 8 — Recommend an architecture and fallback

**Time:** 30 minutes.  
A distributor has 2,000 related weekly item–location series, two years of history, and a strict 20-minute inference window.
One critical group represents 5% of series but 40% of shortage cost.
The current seasonal-naive method meets the runtime requirement.
No candidate has yet been evaluated on a held-out local period.

**Deliverable:** a short decision memo naming candidate families, a fair test, segment guardrails, cost measures, shadow procedure, and fallback.  
**Acceptance:** do not assert an unmeasured winner; make the 20-minute limit and the critical group's outcomes explicit.

## 13. Self-check answers

### Answer 1

Sunday sales arriving at 06:30 are available by the 07:00 origin, assuming the source passed quality checks.
The promotion plan approved Sunday at 18:00 is available.
The revision at 08:00 is not available at 07:00.
Actual future weather is unavailable; an archived weather forecast issued before 07:00 could be a permitted but uncertain input.
The relevant distinction is availability time, not simply the date represented by a value.

### Answer 2

There are `104 - 12 - 4 + 1 = 89` possible windows before split and validity restrictions.
Their overlap means they are not 89 independent observations.
The nominal receptive-field span is `1 + (3 - 1)*1 + (3 - 1)*2 = 7` weeks.
The last output can reach back to lag six, including the current input position.
This configuration has contiguous reach from lag zero through lag six.
A longer input alone does not increase this final output's convolutional reach.

### Answer 3

The RNN processes the window in order and carries a learned state; long-range retention is not guaranteed.
The CNN uses shared local filters and can reach a lag only through its configured receptive field or another input path.
The transformer can directly mix historical positions through attention, provided position and masks are correctly specified.
Access to 52 weeks is not proof that a model learns a useful 52-week dependency.
A fair local rolling-origin test, including runtime and cost, is needed before selecting a winner.

### Answer 4

The absolute errors are `|87 - 91| = 4` units and `|103 - 98| = 5` units.
MAE is `(4 + 5) / 2 = 4.5` units per week.
WAPE is `(4 + 5) / (91 + 98) = 9 / 189 = 4.76%` approximately.
Signed bias is `((87 - 91) + (103 - 98)) / 2 = 0.5` units per week.
The forecast total is `87 + 103 = 190` units.
The proposed order is `max(0, 190 + 20 - 160) = 50` units.
It becomes a commitment only after the appropriate approval and supplier acceptance.

### Answer 5 rubric

A strong answer lists monthly fit dates and weekly forecast origins.
It restricts all training labels to those available by each fit date, including all four target weeks.
It uses the latest promotion version available at each forecast origin, not the version currently stored.
It compares a seasonal or operational baseline on identical origin–series–horizon cases.
It uses validation origins for design and a later untouched test period for final assessment.
It reports at least horizon and critical-product segments, and records the model update schedule.

### Answer 6

The central net estimate is `USD 80,000 - USD 35,000 = USD 45,000` per year.
The lower and upper net estimates are `USD 25,000 - USD 35,000 = -USD 10,000` and `USD 110,000 - USD 35,000 = USD 75,000` per year.
The 4% late-run rate exceeds the 1% guardrail.
Do not deploy as the primary production process until the deadline issue is resolved and re-evaluated.
A validated fallback may support a shadow trial, but it does not erase the measured operational failure.

### Answer 7 rubric

A strong answer identifies revised promotions as leakage and requires archived origin-time plan versions.
It identifies zero-inventory sales as censored, requiring availability-aware labels and sensitivity analysis.
It states that attention weights describe internal mixing, not the causal effect of a promotion.
It proposes a causal identification strategy or controlled experiment before making intervention claims.
It reruns the evaluation on a clean holdout after fixing the information set.

### Answer 8 rubric

A strong memo retains seasonal naive as a real benchmark and fallback.
It proposes small recurrent, temporal CNN, and transformer candidates without asserting a winner.
It specifies training-only preprocessing, rolling origins, and origin-time features.
It evaluates the critical group separately, including shortage cost and service.
It measures end-to-end runtime, infrastructure, support, and decision value.
It starts in shadow mode and defines an owner and threshold for fallback.
It reserves an untouched local test or prospective period for the final selection.

## 14. Interview questions

1. **What is the forecast origin?**
   A strong answer specifies the timestamp and information available then.
   A weak answer treats the target week as the origin without a cutoff.
2. **How does an RNN use a sequence?**
   A strong answer explains shared recurrent updates, state, and long-dependency limits.
   A weak answer says that it stores all prior observations perfectly.
3. **How do you compute a CNN receptive field?**
   A strong answer accounts for kernel widths, dilation, stride, and other aggregation paths.
   A weak answer assumes the input window and receptive field are always equal.
4. **Why does a transformer need position information?**
   A strong answer explains that self-attention alone does not encode temporal order.
   A weak answer assumes the attention matrix automatically knows the calendar.
5. **When is bidirectional historical attention valid?**
   A strong answer ties validity to what was known at the specific forecast origin.
   A weak answer treats every bidirectional encoder as either always safe or always leaking.
6. **What is the difference between a padding mask and a causal mask?**
   A strong answer explains invalid tokens versus prohibited future positions.
   A weak answer treats them as interchangeable.
7. **How can a promotion feature leak?**
   A strong answer describes plan revisions and an as-of join.
   A weak answer checks only whether the promotion date is in the future.
8. **Why can a better WAPE fail to improve service?**
   A strong answer mentions segmentation, lead time, policy, and censored demand.
   A weak answer assumes percentage accuracy directly equals fill rate.
9. **What is the correct comparison for a global encoder?**
   A strong answer includes relevant local and global benchmarks with the same origins and information.
   A weak answer compares against an untuned straw man on different cases.
10. **Why can weekly quantiles not simply be added?**
    A strong answer identifies dependence and the distribution of cumulative demand.
    A weak answer assumes marginal quantiles always add to the same cumulative quantile.
11. **What should trigger a fallback?**
    A strong answer names monitored data, latency, service, or quality thresholds and an owner.
    A weak answer waits for a planner to notice a bad order.
12. **Does attention establish why a promotion worked?**
    A strong answer distinguishes internal weights from causal effects and proposes an appropriate design.
    A weak answer interprets a large weight as proof of causality.

## 15. Key terms

The links below return to the vocabulary or mechanics where each term is defined and applied.

- [As-of join](#4-definitions-and-vocabulary) — reconstructs information available at a cutoff.
- [Attention](#step-10--understand-the-transformer-encoder) — mixes token values using learned scores.
- [Backtest](#step-13--backtest-in-operational-time) — replays historical forecasting decisions.
- [Bias](#step-14--translate-accuracy-into-business-value) — signed average error under a stated convention.
- [Causal convolution](#step-9--understand-the-temporal-cnn-encoder) — excludes later positions from each output.
- [CNN](#step-9--understand-the-temporal-cnn-encoder) — uses shared temporal filters.
- [Covariate](#4-definitions-and-vocabulary) — explanatory input other than the target.
- [Decoder](#4-definitions-and-vocabulary) — component that generates outputs from a representation.
- [Dilation](#step-9--understand-the-temporal-cnn-encoder) — spacing between kernel positions.
- [Encoder](#3-the-core-mental-model) — maps history to a representation.
- [Forecast origin](#step-1--define-a-decision-not-just-a-target) — decision-time information cutoff.
- [Global model](#4-definitions-and-vocabulary) — shares parameters across series.
- [GRU](#step-8--understand-the-rnn-encoder) — gated recurrent unit.
- [Hidden state](#step-8--understand-the-rnn-encoder) — recurrent representation passed between steps.
- [Horizon](#3-the-core-mental-model) — future buckets predicted at an origin.
- [Lead time](#step-1--define-a-decision-not-just-a-target) — elapsed time from order to receipt.
- [Leakage](#step-3--build-a-historical-information-set) — use of unavailable information.
- [Lookback](#3-the-core-mental-model) — number of historical input buckets.
- [LSTM](#step-8--understand-the-rnn-encoder) — gated recurrent architecture with cell state.
- [MAE](#step-14--translate-accuracy-into-business-value) — average absolute error in target units.
- [Padding mask](#step-10--understand-the-transformer-encoder) — excludes absent tokens from attention.
- [Positional encoding](#step-10--understand-the-transformer-encoder) — communicates token order or position.
- [Quantile](#step-11--choose-the-forecast-head-and-objective) — threshold for a specified predictive probability.
- [Receptive field](#step-9--understand-the-temporal-cnn-encoder) — input positions able to affect an output.
- [RNN](#step-8--understand-the-rnn-encoder) — recurrent neural network.
- [Stockout censoring](#step-2--establish-the-target-and-the-grid) — sales constrained by availability.
- [Transformer encoder](#step-10--understand-the-transformer-encoder) — attention-based history representation.
- [Vintage](#step-3--build-a-historical-information-set) — version known at a particular time.
- [WAPE](#step-14--translate-accuracy-into-business-value) — total absolute error divided by total actuals.
- [Window](#step-5--form-training-examples) — history and associated forecast targets.

## 16. Sources

The following sources are directly checkable.
Research papers describe their own datasets and experimental conditions; their reported performance is not a guarantee for this worked example.
Regulatory sources are cited for their published provisions, not as legal advice.

1. Vaswani, A., et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). *Advances in Neural Information Processing Systems 30*. Original transformer architecture and scaled dot-product attention.
2. Hochreiter, S., and Schmidhuber, J. (1997). [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735). *Neural Computation*, 9(8), 1735–1780. LSTM and long-range gradient issues.
3. Cho, K., et al. (2014). [Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation](https://aclanthology.org/D14-1179/). *Proceedings of EMNLP*. Introduces a gated recurrent unit in the encoder–decoder setting.
4. Bai, S., Kolter, J. Z., and Koltun, V. (2018). [An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271). Research preprint on temporal convolution and recurrent comparisons.
5. Salinas, D., Flunkert, V., Gasthaus, J., and Januschowski, T. (2020). [DeepAR: Probabilistic forecasting with autoregressive recurrent networks](https://doi.org/10.1016/j.ijforecast.2019.07.001). *International Journal of Forecasting*, 36(3), 1181–1191.
6. Nie, Y., Nguyen, N. H., Sinthong, P., and Kalagnanam, J. (2023). [A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://arxiv.org/abs/2211.14730). *International Conference on Learning Representations*. PatchTST.
7. Makridakis, S., Spiliotis, E., and Assimakopoulos, V. (2022). [M5 accuracy competition: Results, findings, and conclusions](https://doi.org/10.1016/j.ijforecast.2021.11.013). *International Journal of Forecasting*. Retail benchmark context.
8. Hyndman, R. J., and Athanasopoulos, G. [Forecasting: Principles and Practice, third edition](https://otexts.com/fpp3/). Open textbook; forecast evaluation and time-series cross-validation.
9. GS1. [GS1 General Specifications](https://www.gs1.org/standards/barcodes-epcrfid-id-keys/gs1-general-specifications). Identifier and barcode rules; consult the applicable release.
10. European Union. [Regulation (EU) 2016/679, General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj). Official legal text; application depends on facts and jurisdiction.
11. European Union. [Regulation (EU) 2024/1689, Artificial Intelligence Act](https://eur-lex.europa.eu/eli/reg/2024/1689/oj). Official legal text with staged applicability; check current amendments and guidance.
12. National Institute of Standards and Technology. [Artificial Intelligence Risk Management Framework (AI RMF 1.0)](https://doi.org/10.6028/NIST.AI.100-1) (2023). Voluntary risk-management guidance.
13. PyTorch. [TransformerEncoder documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html). Implementation reference for the code pattern.

## 17. Links

- [Module 18 index](./_index.md) — topic sequence and status.
- [Neural-network fundamentals](./01_neural-network-fundamentals.md) — layers, optimization, and regularization.
- [Module 18 survey](../../../course/18_deep_learning_foundation_models_llm_agents.md) — module narrative and lab sections.
- [Rolling-origin backtesting](../07_time_series_forecasting/08_rolling-origin-backtesting.md) — detailed temporal evaluation.
- [Forecastability and benchmarks](../07_time_series_forecasting/01_forecastability-benchmarks.md) — baseline selection.
- [Forecast error metrics and cost](../07_time_series_forecasting/09_error-metrics-cost.md) — decision-aligned evaluation.
- [Simulation and optimization labs index](../../05_labs_electives/30_simulation_optimization_production_labs/_index.md) — related applied exercises.
- [M5 competition data](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) — external retail dataset; access and terms are controlled by the host.

The module index lists additional sibling topics on global models, neural forecasting, and benchmark contamination; consult their status before following planned links.
