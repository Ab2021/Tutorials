# Neural Network Fundamentals for Supply-Chain Decisions

> Module 18 — Deep Learning, Foundation Models, and LLM Agents  
> Topic 01 of 10  
> Estimated study time: 4 hours  
> Prerequisites: Basic arithmetic and familiarity with supply-chain planning; statistical and machine-learning ideas are introduced here.  
> Version 2026.5

## 1. Why this topic exists

A replenishment planner must decide how many units to order before next week's demand is known.
A supplier needs lead time to prepare the order, and shortages can cost sales or disrupt production.
Excess stock ties up cash and may expire, so forecasting is useful only when it improves the resulting decision.
A neural network is one candidate for estimating future demand from information available at the decision time.
It is not an inventory policy, a promise from a supplier, or an explanation of why customers buy.

Consider a distributor with thousands of product-location combinations.
Some combinations have long histories; others have little data or irregular promotions.
A network fitted across related combinations may learn shared nonlinear relationships.
For example, the effect of a promotion might depend on recent sales, product category, and season.
That potential sharing is a hypothesis to test, not an automatic benefit.
A seasonal-naive forecast or a small regression can outperform a poorly evaluated neural network.

The operations manager needs three distinct objects.
A **forecast** is an estimate of a future outcome under stated information and assumptions.
A **plan** combines forecasts with constraints, costs, and operational choices.
A **target** is a desired result, such as a service level, not a statistical prediction.
A **commitment** is an accepted obligation, such as a confirmed purchase order.
Treating one as another can turn a model error into an unauthorized order.

A data scientist needs to know what the network is optimizing.
A mean-oriented loss, a median-oriented loss, and a service-oriented quantile loss reward different forecasts.
The business owner needs to know whether an improvement survives an honest comparison with a baseline.
The risk owner needs to know which input data may be wrong, private, unavailable, or stale.
The planner needs a fallback for missing inputs and a human escalation path for unusual orders.

This topic develops the machinery from first principles.
No knowledge of the deferred machine-learning module is required.
It connects layers, activations, loss, training, regularization, and validation to a concrete weekly replenishment decision.
The numerical case is illustrative, not a claim about a real company.

### Decisions before algorithms

- Decide the prediction target and the decision horizon.
- Decide who uses the prediction and who can approve an order.
- Decide the cost of overage and underage in the same currency and unit.
- Decide which data are actually available when a prediction is made.
- Decide the baseline and the evaluation period before selecting a model.
- Decide what level of evidence would justify a pilot.
- Decide when the process must fall back to a simpler method.

## 2. Learning outcomes

After this topic you can:

1. **Describe** the information flow from known inputs to a proposed replenishment decision; exercise 1 checks this distinction.
2. **Compute** a forward pass through an affine layer and a nonlinear activation; exercise 2 checks the arithmetic.
3. **Compare** squared, absolute, and quantile losses against a stated decision objective; exercises 3 and 4 check this choice.
4. **Explain** how gradients, learning rate, and training iterations change model parameters; exercise 5 checks the explanation.
5. **Design** a chronological validation and test procedure without future information; exercise 6 checks the design.
6. **Diagnose** overfitting and choose an appropriate regularization response; exercise 7 checks the diagnosis.
7. **Evaluate** a model against a baseline using error, operational costs, and segment performance; exercise 8 checks the business decision.
8. **Specify** ownership, monitoring, and fallback controls for a limited pilot; exercise 9 checks the operational design.

The outcomes are deliberately measurable.
Being able to name an architecture without explaining its inputs, loss, and evaluation is not sufficient.
Being able to compute a loss without stating the decision it supports is also not sufficient.

## 3. The core mental model

```text
Historical information available at forecast origin
  sales, prices, known promotion, calendar, stock flags
                         |
                         v
       feature preparation fitted on training data
                         |
                         v
       input x --> weighted layers --> activation --> output q
                         ^                              |
                         |                              v
                backpropagated gradient <-- training loss
                         |
                         v
        trained parameters fixed for a validation origin
                         |
                         v
       forecast q --> constrained replenishment proposal
                         |
                         v
       planner review --> approved order --> outcome and monitoring
```

The downward path on the left represents prediction.
The training path compares a prediction with an observed outcome and adjusts parameters.
The decision path uses a prediction together with inventory, lead time, constraints, and authority.
An observed outcome returns only after the forecast horizon closes.
It cannot be used to construct a feature for the original prediction.

A network is a parameterized function.
Its parameters are numerical weights and biases learned from examples.
The function converts an input vector into an output using a sequence of operations.
Training selects parameters that reduce a chosen loss on the training examples.
Validation estimates how well a selected procedure may perform on later, unseen examples.
Neither step proves that the model has learned a causal mechanism.

### A single neuron

For two inputs, a neuron first computes `z = w1*x1 + w2*x2 + b`.
The quantities `w1` and `w2` are weights and `b` is a bias.
The activation transforms `z` into an output, for example `max(0, z)`.
The weighted sum combines inputs; the activation allows a network to represent nonlinear relationships.
A layer contains several such neurons, each with its own weights and bias.

### A stack of layers

The input layer represents the measured or encoded features.
A hidden layer forms intermediate features that the model learns during training.
An output layer maps the last hidden representation to the desired prediction.
The word **deep** generally means that several learned layers are composed; it is not a guarantee of better forecasts.
Without a nonlinearity between affine layers, their composition is another affine transformation.
Adding such linear layers alone does not create extra nonlinear expressive power.

### Pattern versus explanation

The network may assign a large output when a certain combination of recent sales and promotion flags appears.
It has learned a predictive association in the training data.
It has not necessarily learned that promotion caused those sales.
A promotion could coincide with a holiday, a stockout recovery, or a price change.
Changing a feature artificially and observing the prediction is a model sensitivity test, not proof of a real-world intervention effect.

### Forecast versus decision

A forecast is measured in future units per stated period.
An order is measured in units to buy or transfer now.
The translation may use on-hand stock, inbound stock, reservations, pack sizes, and supplier limits.
A point forecast alone does not encode all of these operational facts.
Even a well-calibrated quantile is not a complete order quantity when the policy has omitted costs or constraints.

## 4. Definitions and vocabulary

| Term | Definition and unit | Common misuse |
|---|---|---|
| Observation | One recorded example, such as a product-location-week | Assuming every record is independent |
| Target | Quantity to predict, such as units in the next week | Confusing the statistical target with a business service target |
| Feature | Input available at the forecast origin | Including a value first known after the origin |
| Label | Observed target paired with an input | Treating unobserved demand as observed sales during a stockout |
| Forecast origin | Time at which the prediction is issued | Using revised data not yet available then |
| Horizon | Time between origin and target, in weeks here | Mixing one-week and four-week accuracy |
| Parameter | Learned numerical weight or bias | Calling a learning rate a learned parameter |
| Hyperparameter | Procedure choice such as learning rate or layer width | Tuning it on the final test period |
| Weight | Coefficient multiplying a layer input | Treating it as a stable causal effect |
| Bias | Additive parameter in a layer | Confusing it with mean forecast bias |
| Neuron | Weighted combination followed by an activation | Imagining it has an independently readable business meaning |
| Layer | Group of transformations at a network stage | Assuming more layers always help |
| Hidden representation | Intermediate activation vector | Assuming each coordinate maps to one human concept |
| Activation | Nonlinear function applied to a layer output | Using no nonlinearity and claiming nonlinear capacity |
| ReLU | Rectified linear unit, `max(0,z)` | Assuming its derivative is unique at zero |
| Sigmoid | Function mapping real values to `(0,1)` | Treating its output as calibrated probability without testing |
| Softmax | Transformation of scores to values summing to one | Applying it to unconstrained unit demand |
| Affine transformation | Matrix multiplication plus bias | Calling an affine map nonlinear |
| Forward pass | Computation from input to prediction | Confusing it with parameter updates |
| Loss | Numerical penalty for a prediction and label | Assuming any lower loss means lower business cost |
| MSE | Mean squared error, measured in squared target units | Reporting it as though it were units |
| MAE | Mean absolute error, measured in target units | Ignoring that it weights overage and underage equally |
| Pinball loss | Asymmetric quantile loss, in target units | Choosing its quantile without relating it to a decision |
| Gradient | Derivative of loss with respect to parameters | Treating a gradient as the parameter itself |
| Backpropagation | Efficient chain-rule calculation of gradients | Calling it a separate prediction architecture |
| Learning rate | Step size used by an optimizer | Assuming larger steps always converge faster |
| Batch | Set of examples used to calculate one update | Assuming a batch is a time period |
| Epoch | One pass through the training examples | Treating an epoch count as evidence of generalization |
| Optimizer | Rule that updates parameters using gradients | Confusing optimizer choice with loss choice |
| Regularization | Constraint or procedure discouraging excessive fitting | Assuming it cures leakage or data drift |
| Weight decay | Update penalty that tends to shrink weights | Assuming all optimizers implement it identically |
| Dropout | Randomly masking activations during training | Applying random masks at ordinary deterministic inference |
| Early stopping | Selecting a training checkpoint using validation performance | Selecting against the final test repeatedly |
| Overfitting | Learning details that do not transfer to new examples | Diagnosing it from training loss alone |
| Underfitting | Failure to capture useful relationships | Assuming it can only be fixed with a larger network |
| Leakage | Using information unavailable at prediction time, directly or indirectly | Restricting the term to target columns only |
| Training set | Examples used to fit parameters | Describing it as independent evidence of performance |
| Validation set | Held-out examples used for model selection | Calling it a final untouched test after repeated tuning |
| Test set | Held-out examples reserved for final evaluation | Reusing it to choose hyperparameters |
| Scaling | Transforming feature magnitude using fitted statistics | Fitting the scaler on future test data |
| Baseline | Simpler comparison procedure available at the same origin | Comparing against an intentionally weak method |
| Calibration | Agreement between predicted quantities and observed frequencies or levels | Equating calibration with low point error |
| Drift | Change in data or relationships over time | Assuming every change requires an automatic retrain |
| Stockout censoring | Sales limited by unavailable inventory rather than demand | Labeling low sales as low customer demand |
| Quantile | Value with a specified fraction of a distribution at or below it | Treating an 80th percentile as an 80% increase |
| Forecast bias | Average signed forecast error under a declared sign convention | Confusing it with the bias parameter |
| Lead time | Time between order release and availability | Assuming a one-week forecast covers a multiweek lead time |

## 5. The mechanics, step by step

### Step 1 — Frame the operational decision

**Input:** decision owner, item-location scope, decision cadence, lead time, and cost assumptions.
**Output:** a forecast target, horizon, and decision-use statement.

Suppose a planner places an order every Monday for a product at a regional distribution center.
The supplier delivers before the next selling week, subject to a confirmed one-week lead time.
The forecast target might be unit demand during that next selling week.
The observed label might instead be sales, which is only a valid demand proxy when stock was available.
Document the distinction before choosing the model.

The order decision may use a lead-time demand distribution rather than a single weekly point forecast.
If lead time varies or the review period spans multiple weeks, forecast the correct cumulative demand exposure.
A one-week point estimate is not a substitute for this calculation.
For a new product without history, identify a cold-start policy instead of filling missing history with invented zeros.

Record what the model may recommend and what it may not do.
In this example, it may propose a quantity but cannot submit a purchase order automatically.
An authorized planner decides whether the proposal is consistent with supplier capacity and inventory records.

### Step 2 — Define the unit of analysis

**Input:** source-system grain and planned forecast grain.
**Output:** one unique training row per product, location, origin, and horizon.

A product-location-week row must have a stable product identifier and location identifier.
Define week boundaries, the time zone, units of measure, returns treatment, and the cut-off time.
A duplicate created by a many-to-many promotion join doubles the apparent weight of an observation.
A changed pack size can make two numerical quantities incomparable unless converted to a common base unit.
A product replacement or hierarchy change needs effective dates, not just the current description.

Check whether the demand history represents all selling channels or only one.
Do not silently mix retail sales, wholesale shipments, and customer orders.
They are different processes with different delays and constraints.
Preserve a data lineage key that links every row to source snapshots.

### Step 3 — Establish what was knowable

**Input:** an as-of timestamp for every feature and label.
**Output:** a feature availability contract.

At Monday's origin, last week's finalized sales may be available.
A promotion planned before Monday may be available, but its eventual execution is not.
The realized selling price next week is unavailable unless fixed contractually and recorded before origin.
Future inventory, later returns, and revised demand estimates are unavailable.
A feature can carry a timestamp earlier than the origin and still leak if it was revised later.
Use an as-of snapshot or an equivalent reconstruction of the actual information set.

Separate an event's effective date from the date the system first learned it.
A late-entered supplier delay cannot be used as if it were known before the entry.
For missing values, use a policy that could have run at the same origin.
A missing promotion flag should not be automatically treated as no promotion unless the source contract says so.

### Step 4 — Choose informative inputs

**Input:** the availability contract and a documented baseline.
**Output:** candidate features with units, missing-value rules, and provenance.

Useful candidates can include recent unit sales, calendar position, product group, and a preapproved promotion flag.
Recent sales can summarize current demand level, subject to stockout censoring.
Calendar features can represent known seasonal timing but need enough history to estimate recurring effects.
Product groups may allow shared learning while hiding within-group differences.
A price feature needs currency, unit of measure, and the known effective interval.
A promotion flag should state whether it means planned, approved, or executed promotion.

Keep the input set small enough to audit initially.
Do not add a feature merely because it is available in a warehouse.
A feature may increase apparent accuracy while weakening robustness or exposing protected information.
For each feature ask whether it will still be available, on time and in the same definition, during deployment.

### Step 5 — Split by time before fitting transformations

**Input:** dated rows and decision horizons.
**Output:** training, validation, and final test windows with explicit boundaries.

Use earlier dates for training, later dates for validation, and the latest untouched dates for final testing.
For each validation origin, fit or update the procedure using only data available before that origin.
Prevent target windows from overlapping the training cut-off in a way that uses future labels.
A gap or embargo may be required when labels mature late or horizons overlap.
For example, a four-week cumulative label issued on the last training origin is not fully known one week later.
Only train on labels that have matured by the validation origin.

Randomly splitting rows from the same series can leak future regimes and correlated neighbors into training.
It also measures the wrong deployment problem when the intended use is forecasting later dates.
A random split may be reasonable for a different independent-sampling problem, but must be justified separately.

Reserve the test period before trying architectures, losses, and features.
Choose model configuration and checkpoint using validation results only.
After freezing the procedure, evaluate the final test once for the intended decision.
If the test drives another round of design, it has become validation; reserve new evidence for a final test.

### Step 6 — Prepare inputs safely

**Input:** training rows and a feature schema.
**Output:** fitted transformations and transformed training, validation, and test inputs.

Numeric features can be standardized by subtracting a training mean and dividing by a training standard deviation.
Apply the same fitted values to later data; never recompute them using the full time span.
If a training feature is constant, define a safe scale rather than dividing by zero.
The fitted mean and scale are part of the model artifact and must travel with it.
Scaling often improves numerical conditioning, but it does not create information or fix leakage.

Categories can be encoded as indicators or learned embedding vectors.
An embedding assigns each known category a trainable vector rather than a hand-coded numeric rank.
Its coordinates have no guaranteed business interpretation.
Provide an unknown-category behavior for new products or suppliers.
Avoid encoding product IDs as sequential numbers when their numerical distances have no meaning.

Missingness can be informative, such as a supplier not sending a feed.
A missing-value indicator may help distinguish an imputed value from an observed one.
Fit imputation statistics on training data only and assess the effect of missingness by segment.
Do not impute a stockout-affected label as true demand without a separately justified estimation method.

### Step 7 — Define a simple baseline

**Input:** the same information set and evaluation dates as the candidate network.
**Output:** baseline predictions and baseline decision costs.

A last-observation baseline predicts next week's sales as the latest observed sales.
A seasonal-naive baseline predicts a week using the comparable week in a prior season.
A moving average or a regularized linear regression can be a stronger baseline when appropriate.
All baselines must receive the same legitimate feature availability and be evaluated at the same horizons.
Compare full operating costs, not just a model's training time.

A neural model must earn its additional complexity.
If it cannot beat a reasonable baseline on later dates at an acceptable cost, retain the baseline.
If it helps only some segments, assess whether a segment-specific policy is reliable and governable.
Do not select segments by inspecting the final test and then report their performance as independently confirmed.

### Step 8 — Specify the layer computation

**Input:** a numeric feature vector `x` with `d` components.
**Output:** a prediction in the target's units or in a documented transformed scale.

For a hidden layer with `m` neurons, `z = W x + b`.
The weight matrix `W` has `m` rows and `d` columns.
The bias `b` has `m` components.
Each hidden activation is `h_j = activation(z_j)`.
For one scalar output, a final affine layer can compute `q = v·h + c`.
The output may be negative if unconstrained; decide explicitly whether negative forecasts are admissible.

For nonnegative demand, a nonnegative output transformation is one possible design.
Clipping a negative forecast to zero is another post-processing choice that must be included in validation.
Either choice changes the function being evaluated and may affect calibration near zero.
An output transformation should match the target support; it does not solve censored demand.

A larger hidden width gives the model more adjustable coefficients.
For `d` inputs and `m` hidden neurons, the first layer has `m*d + m` parameters.
A scalar output adds `m + 1` parameters.
With two inputs and three hidden neurons, this totals `3*2 + 3 + 3 + 1 = 13` parameters.
The number of observations alone does not determine whether 13 parameters generalize, because rows may be correlated.

### Step 9 — Select an activation

**Input:** hidden-layer design and expected input range.
**Output:** an activation function and an initialization/training plan.

ReLU returns zero for negative inputs and the input for positive values.
It is simple and piecewise linear; several ReLU units can form flexible piecewise-linear functions.
A unit that remains on the negative side may receive zero gradient through that path and stop learning.
At exactly zero its ordinary derivative is undefined; implementations choose a subgradient convention.

Sigmoid produces a number between zero and one.
It can represent a binary probability after suitable training and calibration, not a unit-demand forecast by itself.
For large positive or negative inputs, its slope becomes small, which can slow gradient-based learning.
The hyperbolic tangent is also saturating but maps inputs to a symmetric interval around zero.
Other activations are valid candidates; evaluate them as part of the complete procedure.

If all hidden activations are identities, any stack of affine layers reduces to one affine map.
A nonlinear activation introduces interaction capacity, but does not automatically find useful interactions.
Model capacity must be supported by data and checked on future periods.

### Step 10 — Choose loss in relation to the decision

**Input:** label, prediction, operating costs, and business objective.
**Output:** a mathematically specified training loss and separate decision metrics.

For observation `i`, let `y_i` be actual demand and `q_i` the prediction.
Squared error is `(y_i - q_i)^2` in squared units.
Mean squared error is the average of those terms across the selected observations.
For an unrestricted prediction under suitable conditions, expected squared error is minimized by the conditional mean.
Large misses receive disproportionately large penalties.

Absolute error is `|y_i - q_i|` in units.
Its average is mean absolute error (MAE).
Expected absolute error is minimized by a conditional median, which need not be unique.
It treats one unit of underprediction and one unit of overprediction equally.

For quantile level `tau` between zero and one, define residual `u = y - q`.
Pinball loss is `tau*u` when `u >= 0`, and `(1-tau)*(-u)` when `u < 0`.
For `tau = 0.8`, a one-unit underprediction costs 0.8 loss units and a one-unit overprediction costs 0.2.
Expected pinball loss is minimized by a conditional `tau` quantile, subject to the usual quantile qualifications.

In a one-period newsvendor setting, let underage cost be `Cu` and overage cost be `Co` per unit.
With linear marginal costs, no other constraints, and an appropriate demand distribution, the critical fractile is `Cu/(Cu+Co)`.
A corresponding demand quantile minimizes the expected stylized cost.
That does not mean every supply chain should train at that quantile.
Lead-time uncertainty, lost-customer effects, fixed order costs, capacity, perishability, and service constraints can alter the policy.

Loss is a training objective, not automatically a full business profit model.
Measure decision cost using a separately specified operational simulation or observed decision outcomes.
Report the quantities in meaningful units, including the currency and time period.

### Step 11 — Run the forward pass

**Input:** a batch of prepared feature vectors and current parameters.
**Output:** predictions and a batch loss.

Initialize weights using a reproducible method appropriate for the activation.
For each example, multiply inputs by weights, add biases, apply activations, and compute the output.
Compare each output with its observed label using the selected loss.
Average the example losses or use explicit business weights whose interpretation is documented.
A batch average treats each row equally unless weighting changes that treatment.

The prediction before training is generally not useful.
A reproducible seed can reproduce an experiment in a fixed environment, but does not make the result robust.
Repeat selected experiments across seeds and origins when variability matters to the decision.
Store feature schema, scaling parameters, weights, and code version together.

### Step 12 — Update with backpropagation

**Input:** batch loss and current parameters.
**Output:** updated parameters.

A derivative measures how a small change in a quantity affects another quantity locally.
Backpropagation applies the chain rule from loss to output, hidden activations, and each parameter.
The gradient is the vector of these partial derivatives.
For simple gradient descent, each parameter becomes `old value - learning_rate * gradient`.
The minus sign moves locally downhill for a sufficiently small step under suitable conditions.

A learning rate that is too high can overshoot or make loss unstable.
A rate that is too small can make training impractically slow.
An optimizer such as Adam modifies the update using running gradient statistics.
Its hyperparameters are choices to validate, not a guarantee of a global optimum.
Nonconvex neural-network training can reach different solutions from different initializations.

For mini-batch training, each update uses a subset of training examples.
A batch gradient is an estimate of the full training gradient, so updates vary by batch composition.
An epoch is one pass through the training examples, not a fixed amount of elapsed time.
When constructing batches, do not let a held-out label enter the training loss.

### Step 13 — Control overfitting

**Input:** training and validation learning curves, model capacity, and data quality checks.
**Output:** a selected regularization strategy and checkpoint.

Overfitting occurs when training performance improves while performance on relevant later data deteriorates.
It is not established merely because training loss is lower than validation loss.
The gap can also reflect regime change, a different segment mix, or a data-quality problem.
Inspect both the direction and the size of the gap across multiple chronological origins.

Weight penalties discourage large coefficients under the chosen parameterization.
For an L2 penalty, a typical objective adds `lambda * sum(w^2)` to the data loss.
The exact scaling and whether biases are penalized depend on implementation.
Decoupled weight decay is not generally identical to adding an L2 penalty under every adaptive optimizer.
Compare penalty strength using validation rather than assuming the largest penalty is safest.

Dropout randomly masks some hidden activations during training.
Ordinary inference uses the full network with the framework's corresponding scaling convention.
Dropout may reduce reliance on individual activations, but can hurt small or noisy datasets.
A smaller network or fewer features may be more interpretable and operationally cheaper.

Early stopping tracks a validation loss during training.
Retain the checkpoint with the best validation score under a declared selection rule.
A patience parameter can avoid stopping on a single noisy epoch.
Repeatedly choosing checkpoints makes the validation set part of the selection process.
The final test remains untouched until the whole procedure is frozen.

### Step 14 — Evaluate future performance

**Input:** frozen predictions on chronological holdouts and a predeclared comparison.
**Output:** error, bias, quantile coverage, business cost, and uncertainty by segment.

Compute error as `prediction - actual` if using that sign convention for bias.
State the convention: positive average error then means overforecasting.
MAE gives the average magnitude of a miss but hides direction.
A model may have low overall MAE while consistently underforecasting important promoted products.
Report errors by horizon, volume, product family, location, stock availability, and promotion status where sample sizes permit.

For a predicted 80th percentile, empirical coverage is the fraction of actuals at or below the forecast.
A coverage near 80% overall does not establish 80% coverage in every segment.
Report the count and uncertainty, especially when observations are temporally correlated.
Coverage alone can be gamed with excessively high forecasts, so also report pinball loss and resulting overage.
For discrete unit demand, exact nominal coverage may not be attainable; examine the quantile definition and ties.

For a business comparison, apply identical ordering rules and constraints to model and baseline predictions.
Use realized costs only where their assumptions are defensible and observations are not confounded by policy changes.
When data are serially correlated, uncertainty estimates should respect time blocks or other dependence.
A tiny average improvement that reverses across origins may not justify a production migration.

### Step 15 — Decide, deploy, and monitor

**Input:** frozen test evidence, costs, owner, and operational controls.
**Output:** a go, limited-pilot, or no-go decision with measurable exit criteria.

A limited pilot can run in shadow mode before predictions influence orders.
Compare proposed orders with actual planner choices and investigate material discrepancies.
If the pilot affects orders, predefine the authority, scope, and review frequency.
A human reviewer needs an explanation of inputs, uncertainties, and alternatives, not merely a confidence-colored badge.

Monitor feature freshness, missingness, unit changes, prediction distributions, and downstream forecast errors.
Monitor realized outcomes only after labels mature, and distinguish late labels from true zeros.
Track error and decisions by segment to find concentrated harm.
A statistical change in inputs is a signal for investigation, not proof of performance degradation.
Set concrete thresholds, ownership, and fallback behavior before deployment.

A rollback must restore a tested baseline, its preprocessing, and the associated decision controls.
Retain model and data versions so an affected forecast can be reconstructed.
Retrain only after validating that the new data and evaluation process justify a new model.
Record overrides and their reasons, while recognizing that planner overrides themselves can affect observed outcomes.

## 6. Worked example

### Case and business decision

A regional distributor orders a temperature-sensitive food ingredient weekly.
The illustrative decision is for one product at one distribution center.
At the Monday forecast origin, a one-week delivery lead time is assumed to be reliable.
The target is customer demand, in units during the next selling week.
Stockout-free sales are used as the label in this simplified example.
No supplier capacity limit, fixed order cost, pack multiple, spoilage beyond the overage cost, or demand substitution is modeled.

The planner has a target to avoid lost demand but that target is not itself a forecast.
A forecast estimates next week's demand.
The inventory plan combines that estimate with on-hand and inbound units.
The purchase order becomes a commitment only after planner approval and supplier acceptance.
The numerical costs below are illustrative assumptions, not observed company results.

Assume a missing unit creates an underage cost `Cu = USD 9/unit`.
Assume an excess unit creates an overage cost `Co = USD 3/unit` for the same one-week decision.
The implied critical fractile is `9/(9+3) = 0.75`.
Under the stated single-period assumptions, a conditional 75th-percentile demand forecast is a candidate order-up-to level.
If the model instead predicts the mean, that point is not necessarily the 75th percentile.

### Two inputs known at origin

Use `x1` for recent sales expressed in hundreds of units per week.
Use `x2` for an approved promotion flag, with 1 meaning yes and 0 meaning no.
Suppose `x1 = 1.20` and `x2 = 1` at this origin.
These numbers are illustrative prepared inputs; production scaling would be fitted using training data only.

The demonstration network has two hidden neurons with ReLU activation.
The final output is in units per week.
Assume the following already-fitted parameters.

| Quantity | Value | Interpretation |
|---|---:|---|
| Hidden 1, sales weight | 0.50 | Effect on preactivation per hundred recent units |
| Hidden 1, promotion weight | 0.40 | Effect on preactivation when flag changes from 0 to 1 |
| Hidden 1, bias | -0.20 | Additive preactivation |
| Hidden 2, sales weight | -0.30 | Effect on preactivation per hundred recent units |
| Hidden 2, promotion weight | 0.20 | Effect on preactivation when flag changes from 0 to 1 |
| Hidden 2, bias | 0.10 | Additive preactivation |
| Output weight for hidden 1 | 80 units | Contribution per activation unit |
| Output weight for hidden 2 | 50 units | Contribution per activation unit |
| Output bias | 75 units | Additive weekly forecast |

Compute hidden preactivations.

```text
z1 = 0.50 * 1.20 + 0.40 * 1 - 0.20 = 0.80
z2 = -0.30 * 1.20 + 0.20 * 1 + 0.10 = -0.06
```

Apply the ReLU activation.

```text
h1 = max(0, 0.80) = 0.80
h2 = max(0, -0.06) = 0.00
```

Compute the model's output.

```text
q = 80 * 0.80 + 50 * 0.00 + 75 = 139 units/week
```

Suppose this network was trained with 75th-percentile pinball loss and validated for the intended population.
Its output is a proposed 75th-percentile forecast, not a guarantee that demand will be at most 139 units.
The numerical parameters are illustrative and are not fitted by the separate code example below.
The output layer is unconstrained, so negative values would need a documented post-processing policy.

### Translate forecast to a proposal

Assume on-hand stock available for this demand is 28 units.
Assume confirmed inbound stock arriving before the selling week is 7 units.
Assume no reservations or backorders and no other supply.
The simple order-up-to proposal is `max(0, 139 - 28 - 7) = 104 units`.
The planner checks that these stock numbers and timing are correct before approving 104 units.
If the supplier requires cases of 12 units, rounding to 108 units may be necessary, but that constraint is outside the initial calculation.

The 139-unit forecast is a statistical estimate.
The 104-unit order is a derived plan under the assumptions.
The supplier's confirmed quantity is a commitment, not the model output.
The model is not authorized to place the order.

### Observe and score a single outcome

Suppose true next-week demand is 151 units and stock was available to observe it.
The forecast error using `forecast - actual` is `139 - 151 = -12 units`.
The absolute error is 12 units and the squared error is 144 units squared.
The 75th-percentile pinball loss is `0.75 * 12 = 9 units`.
The stylized underage cost is `12 * USD 9 = USD 108` for this one week.
This is not a full profit calculation and excludes downstream effects.

Compare a baseline that predicts 145 units for the same origin.
Its error is `145 - 151 = -6 units`.
Its absolute error is 6 units and squared error is 36 units squared.
Its pinball loss is `0.75 * 6 = 4.5 units`.
Its assumed underage cost is `6 * USD 9 = USD 54`.
At this single origin, the baseline is better on each listed measure.
One week is not enough evidence to choose either model for future deployment.

### Compare four held-out weeks

Suppose the following four outcomes were reserved before model selection.
For teaching clarity, assume the numbers are independent of the separate illustrative forward pass.
All numbers are units per week for the same product and location.

| Week | Actual demand | Neural 75th-percentile forecast | Baseline forecast |
|---|---:|---:|---:|
| 1 | 151 | 139 | 145 |
| 2 | 112 | 125 | 120 |
| 3 | 134 | 140 | 125 |
| 4 | 97 | 105 | 110 |

For the neural model, absolute errors are 12, 13, 6, and 8 units.
Their sum is 39 units, so MAE is `39/4 = 9.75 units/week`.
Signed errors are -12, +13, +6, and +8 units.
Their mean is `15/4 = +3.75 units/week`, an overforecast under this sign convention.
Demand is at or below the neural forecast in 3 of 4 weeks, giving empirical coverage of 75%.
Four observations are much too few to establish reliable 75% coverage.

For the baseline, absolute errors are 6, 8, 9, and 13 units.
Their sum is 36 units, so MAE is `36/4 = 9 units/week`.
Signed errors are -6, +8, -9, and +13 units.
Their mean is `6/4 = +1.5 units/week`.
Demand is at or below the baseline forecast in 2 of 4 weeks, giving empirical coverage of 50%.
Coverage is descriptive here; neither procedure is proven calibrated by four observations.

Calculate the neural pinball losses at `tau = 0.75`.
The four losses are 9, 3.25, 1.5, and 2 units.
The mean pinball loss is `15.75/4 = 3.9375 units/week`.
Calculate the baseline pinball losses.
The four losses are 4.5, 2, 6.75, and 3.25 units.
The mean is `16.5/4 = 4.125 units/week`.
The neural model has worse MAE but lower pinball loss in this small example.
That is possible because the metrics reward different error patterns.

Calculate the stylized neural decision cost with USD 9/unit underage and USD 3/unit overage.
The four costs are USD 108, USD 39, USD 18, and USD 24.
The total is USD 189, or USD 47.25 per week.
The baseline costs are USD 54, USD 24, USD 81, and USD 39.
The total is USD 198, or USD 49.50 per week.
These totals equal the corresponding pinball sums multiplied by USD 12/unit because the cost ratio matches `tau = 0.75`.
That equality relies on the stated linear cost model, not on a general property of all supply-chain decisions.

### Decision and limits

The neural model saves an illustrative USD 9 across the four held-out weeks under the simplified cost rule.
Its lower pinball loss supports the direction of the business-cost result under those assumptions.
The evidence does not establish a reliable operational saving or justify deployment by itself.
The sample is tiny, the forecast series are hypothetical, and uncertainty is large.
A proper decision requires many later origins, more product-location segments, and realistic stock and capacity constraints.
Add model maintenance, compute, planner review, and potential failure costs to the benefit calculation.
Retain the baseline until a prespecified evaluation supports a limited pilot.

## 7. Code, query, or calculation

The following standalone Python 3 example trains a small neural network from scratch using only the standard library.
It predicts next-week units from a known recent-sales signal and a planned promotion flag.
It demonstrates a forward pass, ReLU, mean squared loss, analytic backpropagation, and chronological validation.
It is an educational deterministic synthetic dataset, not the preceding 75th-percentile business model.
Its squared loss targets a conditional mean under appropriate assumptions, not a 75th percentile.
The baseline is last week's sales and is evaluated over the same later dates.
No external files, downloads, or machine-learning libraries are required.

The synthetic generator creates one series with a predictable trend and scheduled promotion pattern.
Its rows overlap in time, so the displayed validation results are not an independent uncertainty estimate.
The final four weeks are a test set and are not used to select the checkpoint.
The model-selection rule uses the lowest validation MSE across the training epochs.
The scaler is fitted only on the training portion.
The output is unconstrained; the synthetic example does not establish safe handling of negative predictions.

```python
import math
import random

# One observation represents one week; sales values are units per week.
# Promotions are known at the forecast origin for this synthetic example.
N_WEEKS = 60
sales = []
promotions = []
for week in range(N_WEEKS):
    promo = 1.0 if week % 6 == 0 else 0.0
    # Deterministic synthetic demand; no stockouts are simulated.
    units = 80.0 + 0.6 * week + 12.0 * promo + 3.0 * math.sin(week / 4.0)
    sales.append(units)
    promotions.append(promo)

# At the start of week t, last week's units and this week's approved
# promotion are known. The label is realized sales in week t.
rows = []
for t in range(1, N_WEEKS):
    rows.append(([sales[t - 1], promotions[t]], sales[t]))

# Chronological partitions by target week, not by random row.
train = rows[:39]       # target weeks 1..39
validation = rows[39:55]  # target weeks 40..55
test = rows[55:]        # target weeks 56..59

# Fit feature scaling only on training rows. Promotion is left as 0/1.
mean_x = sum(x[0] for x, _ in train) / len(train)
variance_x = sum((x[0] - mean_x) ** 2 for x, _ in train) / len(train)
scale_x = math.sqrt(variance_x) or 1.0
# Center/scale the target to make optimization numerically convenient.
mean_y = sum(y for _, y in train) / len(train)
variance_y = sum((y - mean_y) ** 2 for _, y in train) / len(train)
scale_y = math.sqrt(variance_y) or 1.0

def prepare(data):
    return [
        ([(x[0] - mean_x) / scale_x, x[1]], (y - mean_y) / scale_y)
        for x, y in data
    ]

tr = prepare(train)
va = prepare(validation)
te = prepare(test)

# A 2-input, 4-hidden-unit ReLU network with one linear output.
rng = random.Random(42)
width = 4
W = [[rng.uniform(-0.4, 0.4) for _ in range(2)] for _ in range(width)]
b = [0.0] * width
v = [rng.uniform(-0.4, 0.4) for _ in range(width)]
c = 0.0

def forward(x):
    z = [sum(W[j][k] * x[k] for k in range(2)) + b[j] for j in range(width)]
    h = [max(0.0, value) for value in z]
    prediction = sum(v[j] * h[j] for j in range(width)) + c
    return prediction, z, h

def mse(data):
    return sum((forward(x)[0] - y) ** 2 for x, y in data) / len(data)

def snapshot():
    return ([row[:] for row in W], b[:], v[:], c)

def restore(state):
    global W, b, v, c
    W, b, v, c = [row[:] for row in state[0]], state[1][:], state[2][:], state[3]

best = snapshot()
best_val = mse(va)
best_epoch = 0
learning_rate = 0.03

# Full-batch gradient descent. Validation chooses the checkpoint only.
for epoch in range(1, 1201):
    grad_W = [[0.0, 0.0] for _ in range(width)]
    grad_b = [0.0] * width
    grad_v = [0.0] * width
    grad_c = 0.0
    for x, y in tr:
        prediction, z, h = forward(x)
        d_output = 2.0 * (prediction - y) / len(tr)
        grad_c += d_output
        for j in range(width):
            grad_v[j] += d_output * h[j]
            # At zero use the conventional ReLU subgradient 0.
            d_hidden = d_output * v[j] * (1.0 if z[j] > 0.0 else 0.0)
            grad_b[j] += d_hidden
            for k in range(2):
                grad_W[j][k] += d_hidden * x[k]
    for j in range(width):
        for k in range(2):
            W[j][k] -= learning_rate * grad_W[j][k]
        b[j] -= learning_rate * grad_b[j]
        v[j] -= learning_rate * grad_v[j]
    c -= learning_rate * grad_c
    score = mse(va)
    if score < best_val:
        best_val = score
        best_epoch = epoch
        best = snapshot()

restore(best)

def predictions(data):
    return [forward(x)[0] * scale_y + mean_y for x, _ in data]

def mae(pred, actual):
    return sum(abs(p - y) for p, y in zip(pred, actual)) / len(actual)

actual = [y for _, y in test]
model = predictions(te)
baseline = [x[0] for x, _ in test]
print(f"Training rows: {len(train)}")
print(f"Validation rows: {len(validation)}")
print(f"Test rows: {len(test)}")
print(f"Selected epoch: {best_epoch}")
print(f"Test neural MAE: {mae(model, actual):.2f} units/week")
print(f"Test last-week MAE: {mae(baseline, actual):.2f} units/week")
```

Expected output has 39 training rows, 16 validation rows, and 4 test rows.
It also reports an integer selected epoch from 0 through 1200 and two finite MAE values in units per week.
Exact model scores depend on Python's floating-point execution; the example makes no claim that the network will beat the baseline.
The test metric is printed for instruction; do not repeatedly modify the model after seeing it and still call that set an untouched test.

To adapt this to operational data, replace the synthetic generator with an as-of data extract.
Define product-location-origin keys, a mature label, and a lead-time-aligned horizon.
Use rolling chronological evaluation rather than four final weeks alone.
A production framework can implement automatic differentiation and more efficient batching.
Document its version, random seed, scaling artifact, model artifact, and inference environment.
A real 75th-percentile model requires a pinball objective and validation of quantile behavior.
Do not infer a 75th percentile by labeling this MSE model's output as one.

## 8. Data, standards, and regulatory context

### Data contracts and traceability

A neural model is only as reproducible as its information set.
Store each feature's source system, owner, refresh cadence, unit, and as-of semantics.
For every training label, store the target period and when it became available for training.
A feature record updated after a forecast origin needs a historical snapshot or equivalent audit trail.
Current-state tables alone may be insufficient to reproduce a historical prediction.

Demand, orders, shipments, receipts, and sales have distinct meanings.
A retailer's recorded sales can be constrained by shelf availability.
A distributor's shipments can be constrained by allocation or transport capacity.
A purchase order reflects a previous planning decision and can amplify upstream variability.
Choose the target based on the actual decision and explicitly document its limitations.

Use a common unit of measure before combining cases and individual units.
A product identifier may be reused or changed during a packaging transition.
An effective-dated product mapping is needed to avoid joining today's product hierarchy to historical data incorrectly.
Record currency and exchange-rate timing for financial costs.
A model should not infer operational units from a free-text description alone.

### GS1 identification

[GS1](https://www.gs1.org/standards/id-keys/gtin) defines the Global Trade Item Number (GTIN) identification key for trade items.
A GTIN can support consistent product identity across trading partners when assigned and maintained correctly.
It does not by itself define the target, validate sales, establish demand, or guarantee a correct product hierarchy.
Internal product-location keys and effective dates may still be required.
Whether a GTIN is mandated for a specific transaction depends on applicable contracts and rules; the standard itself is not a universal legal mandate.

A data-quality check can compare the GTIN and unit-of-measure mapping against a trusted master record.
Investigate duplicate identifiers, invalid check digits, and changes in pack configuration.
An identifier that joins successfully can still refer to the wrong historical package.
Keep original source keys for reconciliation and retain approved corrections with effective dates.

### Privacy and access

Demand data may contain customer, employee, or supplier information.
Use the minimum data needed for the forecast and restrict access to authorized purposes.
Aggregate where this preserves the decision value and reduces unnecessary exposure.
Pseudonymization reduces direct identification but does not automatically make data anonymous.
A trained model can also disclose information in some circumstances, so treat artifacts and logs as governed assets.

The EU General Data Protection Regulation, Regulation (EU) 2016/679, applies to personal-data processing within its scope.
Its provisions include purpose limitation, data minimization, security, and other obligations; applicability depends on facts and jurisdiction.
The official regulation is linked in the sources, and this discussion is educational rather than legal advice.
A demand forecast using aggregate product data is not automatically exempt if underlying records remain personal data.
Consult the relevant privacy owner before using customer-level histories or sharing models with a vendor.

### AI governance and standards

The [NIST AI Risk Management Framework 1.0](https://www.nist.gov/itl/ai-risk-management-framework) is voluntary guidance, not a general legal certification.
Its Govern, Map, Measure, and Manage functions provide a useful structure for model ownership and risk assessment.
They do not prescribe a particular neural architecture or establish that a forecast is accurate.
A documented model inventory, intended use, evaluation record, and incident path make the guidance actionable.

The EU Artificial Intelligence Act, Regulation (EU) 2024/1689, has phased application dates and a risk-based scope.
A supply-chain forecast is not automatically a high-risk use merely because it uses a neural network.
Specific obligations depend on the system's intended purpose, roles, jurisdiction, and applicable dates.
Read the official text and obtain qualified legal advice for an actual deployment.
Do not confuse a voluntary framework with a binding regulation, or a product feature with a regulatory classification.

### Retention and external vendors

A hosted model provider may receive inputs, logs, metadata, and prediction requests.
Review the contract for permitted use, retention, training on customer data, subprocessors, and deletion.
Verify data residency and incident notification requirements where applicable.
A vendor's model card or benchmark is evidence about its stated test, not proof of performance on your own operations.
Confirm that the production data contract permits the actual use before sending records outside the organization.

### Quality controls at the interface

Validate schema, units, key uniqueness, expected ranges, and data freshness before generating predictions.
Flag missing features separately from true zeros.
Quarantine late or contradictory inputs according to a documented rule.
Do not silently substitute future revisions or use an average fitted on the evaluation period.
If critical input quality checks fail, invoke a tested fallback and notify the owner.
Keep the original input snapshot and prediction for audit within the applicable retention policy.

## 9. Industry practice and case evidence

### Published evidence is conditional

A network's business value depends on the data, benchmark, horizon, and operating decision.
Peer-reviewed forecasting competitions show that methods can perform differently across datasets and evaluation protocols.
They do not prove that one architecture is best for every product or that benchmark accuracy translates directly into profit.
Review the forecast origin, input availability, training population, and metrics before transferring a claim.

The M4 competition, reported by Makridakis and colleagues in the *International Journal of Forecasting*, compared methods across many time series.
Its winning method combined statistical and machine-learning components.
The published result supports evaluating combinations and strong baselines; it does not demonstrate that a simple feed-forward network will win on a particular distributor's data.
The competition's series and evaluation setup are not necessarily representative of a local supply chain.

The M5 forecasting competition used Walmart retail sales data and emphasized hierarchical, intermittent, and cross-sectional forecasting challenges.
Its published analysis provides a concrete retail forecasting benchmark and shows the importance of evaluation design.
The data and competition results should not be treated as a randomized measurement of savings at a specific store.
A high public leaderboard rank also does not prove that all production information was available at a commercial forecast origin.

### Company case: Amazon forecasting research

Salinas and coauthors' DeepAR paper describes a probabilistic forecasting method trained across related time series.
The paper reports experiments on public and proprietary datasets, including retail-relevant series.
This is documented research evidence for a shared neural forecasting approach, not proof that every local series benefits from cross-learning.
DeepAR is a recurrent probabilistic model, not the small feed-forward network in this lesson.
The transferable lesson is to compare probabilistic predictions at the relevant horizon against appropriate alternatives.

A global model can share information across items with similar history.
It may also transfer the wrong pattern when products have different lifecycles, substitutions, or supply constraints.
Evaluate new items, sparse items, high-volume items, and discontinued items separately.
If a category has few examples, aggregate results do not provide strong segment-specific evidence.

### Company case: Walmart competition data

The M5 dataset is based on Walmart unit sales for products and stores, as described in the competition publication.
Its availability makes it useful for reproducing a shared retail forecasting task.
Its calendar, product hierarchy, and prices require careful as-of treatment when reproducing a real operating process.
Competition participants may have information that differs from a particular retailer's production data feed.
A leaderboard score is therefore not a substitute for an internal rolling-origin trial.

[INFERENCE] A distributor considering a global model can use such a dataset to rehearse feature pipelines and error decomposition.
It cannot infer its own financial benefit without its own lead times, costs, constraints, and deployment trial.
The case illustrates an evaluation method, not an observed adoption or saving by the distributor in section 6.

### What an operational team should require

An analyst should provide baseline and model forecasts for identical origins.
The business owner should approve the costs and decision rule before seeing comparative results.
The data owner should sign off on feature availability and unit definitions.
The risk owner should assess privacy, access, and operational impact.
The planner should review whether recommendations remain feasible under supplier constraints.

A model card should identify intended use, target population, out-of-scope cases, and known limitations.
A runbook should define the data checks, service owner, retraining gate, and fallback.
A pilot report should separate statistical forecast improvement from modeled cost improvement and realized operational outcomes.
When a result is simulated, label it simulated; when it is a vendor claim, identify the vendor and the evaluation conditions.
No public case in this section is presented as evidence of the illustrative USD 9 saving.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Future feature leakage | Validation suddenly becomes implausibly accurate | Data revised after the forecast origin enters inputs | Production accuracy collapses | Reconstruct as-of snapshots and audit availability |
| Label leakage | A feature closely matches the next-week target | Target or later aggregate is joined into inputs | False confidence and poor orders | Trace each column to source and time of availability |
| Random temporal split | Excellent random split, weak later performance | Correlated future rows enter training | Inflated model-selection score | Use rolling chronological origins |
| Unmatured labels | Training labels overlap validation origin | Outcomes were not known when model would be trained | Hidden lookahead | Train only on labels mature at each origin |
| Stockout censoring | Low sales coincide with zero inventory | Sales understates latent demand | Chronic underordering | Flag availability and evaluate a defensible demand proxy |
| Wrong target | Forecast follows shipments but business needs consumption | Shipments reflect allocation and earlier orders | Misaligned replenishment | Define target and map it to the decision |
| Scale leakage | Validation scaler uses entire history | Future distribution affects training transform | Optimistic evaluation | Fit preprocessing inside each training window |
| Unit conversion error | Forecast jumps by a case-pack factor | Units and cases are mixed | Large overorders or shortages | Validate effective-dated units and pack sizes |
| Overfitting | Training loss falls while later-origin loss worsens | Capacity fits noise or obsolete relationships | Weak future performance | Reduce capacity, regularize, and validate across origins |
| Underfitting | Training and validation errors are both high | Model or inputs cannot represent useful patterns | Missed benefit | Review features, target, and model capacity |
| Wrong loss | MAE improves while shortage cost rises | Symmetric loss ignores asymmetric costs | Lower service or higher cost | Align training and evaluation with decision costs |
| Miscalibrated quantile | Coverage varies sharply by segment | Distribution shifts or model is biased | Service failures or excess stock | Check coverage and pinball loss by segment |
| Promotion leakage | Accuracy relies on executed promotion data | Future execution substituted for known plan | Production inputs unavailable | Use the approved plan as known at origin |
| Category drift | Many unknown product IDs at inference | Product launches or identifiers change | Unreliable outputs | Define unknown-category handling and cold-start policy |
| Negative output | Demand predictions fall below zero | Unconstrained regression output | Invalid order proposal | Validate nonnegative transformation or documented clipping |
| Gradient instability | Loss becomes nonfinite or oscillates | Learning rate or feature scales are unsuitable | Training fails or gives arbitrary weights | Scale inputs, reduce step size, inspect gradients |
| Dead ReLU units | Many hidden activations remain zero | Persistent negative preactivation blocks gradients | Lost effective capacity | Inspect activations and consider initialization or alternatives |
| Segment masking | Overall MAE improves while small locations worsen | High-volume groups dominate aggregate metrics | Unequal operational harm | Report segment counts and separate costs |
| Repeated test tuning | Test score improves after many redesigns | Final test becomes a selection set | Overstated generalization | Reserve a new untouched period |
| Data latency | Required feed misses its forecast cut-off | Pipeline timing differs from training assumptions | Stale or missing predictions | Monitor freshness and invoke fallback |
| Unreviewed automation | Orders are released from raw predictions | Forecast confused with authorized commitment | Financial and service exposure | Enforce approval and operational constraints |
| Uncontrolled retraining | Output changes without a release record | New weights or scalers replace validated artifacts | Untraceable performance regression | Version artifacts and require validation before release |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or respond |
|---|---|---|---|
| Labels represent demand | Training needs a meaningful outcome | Stockouts hide unmet demand | Compare sales with availability and flag censored periods |
| Features exist at origin | Future deployment must reproduce inputs | Late revision or future execution data | Audit as-of timestamps and historical snapshots |
| Comparable units | Loss and costs require consistent quantities | Case-pack or product changes | Reconcile against effective-dated master data |
| Training covers relevant conditions | Learned patterns need examples | New products, promotions, or disruptions | Segment by novelty and define cold-start fallback |
| Future relationships are sufficiently stable | Historical fit must transfer | Price changes, competitor actions, or channel shifts | Evaluate multiple rolling origins and monitor drift |
| Costs are defensible | Quantile choice depends on cost ratio | Lost sales costs or salvage values are wrong | Run sensitivity analysis with business owners |
| Single-period assumptions hold | Critical fractile derives from a simple decision | Variable lead times, fixed costs, or capacity | Model the actual lead-time and constrained policy |
| Validation is representative | Model selection uses validation results | One anomalous season or unrepresentative location | Compare across multiple periods and segments |
| Test remains untouched | Final estimate must be independent of tuning | Repeated redesign using test outcomes | Freeze procedure and reserve a new later test |
| Network output matches target support | Invalid values are not actionable | Unconstrained output produces negative units | Include output treatment in evaluation |
| Scale and schema persist | Inference must reproduce training transformations | Changed feature order, missing scaler, or feed redesign | Version artifacts and test schema at serving |
| Planner can act on proposal | Statistical forecasts need a feasible decision path | Supplier cannot deliver, inventory records are wrong | Integrate constraints and require operational review |
| Model benefit exceeds cost | Adoption consumes people and infrastructure | Marginal accuracy gain is too small | Include maintenance, inference, and review costs |
| Reported segment results have enough evidence | Small samples are noisy | Sparse item locations or rare promotions | Report counts, uncertainty, and pooling assumptions |

These assumptions are not guarantees.
Failing one does not always invalidate a model, but it changes what the reported result means.
For example, a model may predict constrained sales accurately while being unsuitable for estimating lost demand.
A pilot should name which assumptions are most consequential and how their failure will be detected.

## 12. Exercises

### Exercise 1 — Forecast, plan, and commitment

**Time:** 10 minutes.
A model estimates 180 units next week; available stock is 50 units and confirmed inbound is 20 units.
The planner proposes an order of 110 units, then a supplier confirms 96 units.
**Deliverable:** classify each number as forecast, inventory input, plan, or commitment.
**Acceptance:** explain why the 180-unit forecast is neither the submitted order nor a guaranteed outcome.
**Outcome:** 1.

### Exercise 2 — Compute a forward pass

**Time:** 15 minutes.
Let `x1 = 2`, `x2 = 1`, `z1 = 0.5*x1 - x2 + 0.2`, and `z2 = -0.5*x1 + 0.4*x2`.
Apply ReLU to each hidden preactivation and compute `q = 10*h1 + 5*h2 + 20` units.
**Deliverable:** show `z1`, `z2`, both activations, and the final forecast.
**Acceptance:** retain the sign before activation and state the output unit.
**Outcome:** 2.

### Exercise 3 — Compare error losses

**Time:** 15 minutes.
For actual demand of 120 units, compare forecasts of 110 units and 125 units.
Compute absolute error, squared error, and pinball loss at `tau = 0.8`.
**Deliverable:** a two-row table with each calculation.
**Acceptance:** explain why the preferred forecast can differ under absolute and pinball losses.
**Outcome:** 3.

### Exercise 4 — Derive a cost-aligned quantile

**Time:** 15 minutes.
Assume USD 8 per unit of underage and USD 2 per unit of overage in a one-period decision.
There are no fixed costs, capacity constraints, or lead-time complications.
**Deliverable:** compute the critical fractile and name two assumptions that make the calculation incomplete in practice.
**Acceptance:** connect the quantile to the cost ratio rather than a generic service target.
**Outcome:** 3.

### Exercise 5 — Explain an update

**Time:** 10 minutes.
A scalar parameter is 2.0, its loss gradient is 0.4, and the learning rate is 0.1.
Use simple gradient descent.
**Deliverable:** calculate the new parameter and explain how backpropagation supplies the gradient.
**Acceptance:** apply the minus sign and distinguish parameter from learning rate.
**Outcome:** 4.

### Exercise 6 — Design a temporal evaluation

**Time:** 25 minutes.
Weekly data cover 2022 through 2025; the deployment decision is made each Monday for the next four weeks.
Labels arrive one week after their target period ends, and promotion execution is known only after the promotion.
**Deliverable:** propose training, validation, and final test windows, plus a label-maturity and feature-availability rule.
**Acceptance:** no test tuning, no unmatured labels, and no realized promotion execution at the forecast origin.
**Outcome:** 5.

### Exercise 7 — Diagnose model behavior

**Time:** 20 minutes.
Across training epochs, training MAE falls from 22 to 8 units while chronological validation MAE falls from 24 to 16 and then rises to 21.
The validation period also includes a known supplier disruption absent from training.
**Deliverable:** state two plausible explanations and a procedure to distinguish them.
**Acceptance:** consider both overfitting and regime difference; propose at least one regularization or checkpoint choice without asserting it is proven.
**Outcome:** 6.

### Exercise 8 — Evaluate a business comparison

**Time:** 20 minutes.
A model has errors `-4, +10, -2, +6` units, using forecast minus actual.
A baseline has errors `-6, +4, -4, +4` units.
Underage costs USD 9/unit and overage costs USD 3/unit.
**Deliverable:** calculate MAE and total stylized cost for each method and decide whether four weeks justify deployment.
**Acceptance:** show the cost for each sign and distinguish the illustrative result from sufficient evidence.
**Outcome:** 7.

### Exercise 9 — Specify a controlled pilot

**Time:** 25 minutes.
A proposed system forecasts weekly sales for 500 product-location pairs and sends order suggestions to planners.
Data arrive from a vendor feed that can be late; some products are frequently out of stock.
**Deliverable:** draft a one-page pilot protocol with owner, baseline, data checks, metrics, approval, fallback, and stop criteria.
**Acceptance:** address as-of data, censoring, segment reporting, maturity of labels, and a human approval boundary.
**Outcome:** 8.

## 13. Self-check answers

### Answer 1

The 180-unit estimate is the forecast for next week.
The 50 units on hand and 20 confirmed inbound units are inventory inputs.
The 110-unit proposed order is a plan derived from `180 - 50 - 20` under the stated simple policy.
The supplier-confirmed 96 units are a commitment, subject to whatever terms define confirmation.
A forecast is uncertain and does not authorize procurement.
A confirmed order can differ from the proposed plan because of supplier capacity or other constraints.

### Answer 2

`z1 = 0.5*2 - 1 + 0.2 = 0.2`.
`z2 = -0.5*2 + 0.4*1 = -0.6`.
After ReLU, `h1 = 0.2` and `h2 = 0`.
The forecast is `10*0.2 + 5*0 + 20 = 22 units`.
The negative preactivation is not itself a negative forecast because the activation sets it to zero.

### Answer 3

| Forecast | Absolute error | Squared error | Pinball loss at 0.8 |
|---:|---:|---:|---:|
| 110 units | 10 units | 100 units squared | `0.8*10 = 8` units |
| 125 units | 5 units | 25 units squared | `0.2*5 = 1` unit |

Both metrics prefer 125 units in this particular pair.
The reason for the asymmetric calculation is that 110 units underpredicts while 125 units overpredicts.
To see a ranking reversal, compare 110 units with 130 units: both have 10-unit absolute error, but pinball losses are 8 and 2 units.
An even stronger reversal compares 110 with 135: absolute error prefers 110, while pinball loss prefers 135 because `3 < 8` units.

### Answer 4

The critical fractile is `8/(8+2) = 0.8`, or the 80th percentile.
The derivation assumes linear overage and underage costs for a single period.
It also assumes the relevant demand distribution and inventory position match the decision horizon.
Variable lead time, order minimums, fixed costs, perishability, or substitution can change the optimal policy.
An 80th percentile is not automatically an 80% fill-rate guarantee.

### Answer 5

The update is `2.0 - 0.1*0.4 = 1.96`.
The parameter decreases because the local gradient is positive.
Backpropagation applies the chain rule to compute how the loss changes with each parameter.
The learning rate controls the update size; it is not the gradient or a learned weight in this calculation.

### Answer 6 rubric

A defensible plan can train on earlier years, validate on later origins, and reserve 2025 as a final test.
The exact boundaries must reflect the amount of available data and business seasonality.
At each origin, include only four-week target labels whose target window ended and whose one-week reporting delay has passed.
Fit preprocessing only on the eligible training rows at that origin.
Use approved promotion plans known at the origin, not later execution.
Choose model and checkpoint using validation only, then freeze the full procedure for test.
Explain how rolling origins, horizon overlap, and product-location correlations affect uncertainty.

### Answer 7 rubric

Overfitting is plausible because training continues improving after validation deteriorates.
A regime shift is also plausible because the disruption is present only in validation.
Inspect errors by origin and disruption status and compare with a baseline under the same conditions.
Use more chronological origins if available and check whether the validation deterioration repeats.
Early stopping at the minimum validation MAE is a candidate, not a proven remedy for a future disruption.
Reducing width or adding regularization can be tested; neither repairs missing disruption features automatically.

### Answer 8

The model's absolute errors sum to `4+10+2+6 = 22` units, giving MAE `5.5 units/week`.
The baseline's absolute errors sum to `6+4+4+4 = 18` units, giving MAE `4.5 units/week`.
The model underage is 6 units and overage is 16 units.
Its stylized cost is `6*USD 9 + 16*USD 3 = USD 102`.
The baseline underage is 10 units and overage is 8 units.
Its stylized cost is `10*USD 9 + 8*USD 3 = USD 114`.
The model has worse MAE but USD 12 lower modeled cost across these four weeks.
Four observations cannot establish durable savings, calibration, or safe deployment.

### Answer 9 rubric

A strong protocol names a decision owner and a data owner.
It defines scope, forecast horizon, target, and decision cadence.
It compares the network with a prespecified baseline at identical chronological origins.
It checks freshness and as-of provenance, flags stockout-censored labels, and waits for label maturity.
It measures MAE, bias, decision cost, and segment-level performance with counts.
It requires planner approval, logs overrides, and prevents direct purchase-order submission.
It specifies late-feed fallback, stop thresholds, incident ownership, and a review date.
It includes compute, maintenance, and review costs in the go/no-go assessment.

## 14. Interview questions

### 1. Why do neural networks need nonlinear activations?

A strong answer explains that affine layers without intervening nonlinearities collapse to one affine map.
It gives an example such as ReLU and explains that capacity is not the same as generalization.
A weak answer says more layers alone necessarily learn nonlinear relationships.

### 2. What does a hidden neuron learn?

A strong answer describes a weighted combination and activation learned to reduce the training objective.
It notes that a coordinate may not have a stable business interpretation and is not automatically causal.
A weak answer claims each neuron must correspond to a single product attribute.

### 3. How do you choose between MSE and pinball loss?

A strong answer relates the former to conditional means and the latter to conditional quantiles under appropriate assumptions.
It connects the cost asymmetry and horizon to the decision and evaluates full policy cost separately.
A weak answer chooses the metric solely because a library uses it by default.

### 4. What does backpropagation do?

A strong answer explains chain-rule derivatives of loss with respect to parameters, followed by an optimizer update.
It distinguishes a forward pass, gradient, learning rate, and optimizer.
A weak answer says backpropagation sends actual future values to the prediction service.

### 5. Why is a random split risky for a weekly forecast?

A strong answer identifies temporal leakage, correlated rows, label maturity, and the need for later origins.
It recognizes that the split must match the intended deployment problem.
A weak answer asserts random splitting is always invalid for every possible task.

### 6. How do you detect overfitting?

A strong answer compares training and later-origin validation curves and considers regime shift and data quality.
It proposes controlled experiments with early stopping, capacity, or regularization.
A weak answer treats any training-validation gap as conclusive proof of overfitting.

### 7. What does 75% quantile coverage mean?

A strong answer says approximately 75% of relevant outcomes should be at or below the forecast under suitable calibration and sampling conditions.
It discusses discrete outcomes, sample size, segments, and sharpness or pinball loss.
A weak answer promises that every individual week has a 75% fill rate.

### 8. How can a network improve forecast MAE but worsen replenishment?

A strong answer identifies asymmetric shortage costs, lead-time mismatch, constraints, or segment-specific errors.
It compares decision costs under the same policy and includes model operating expense.
A weak answer assumes a lower symmetric error must always reduce cost.

### 9. What does a stockout do to a demand label?

A strong answer explains that observed sales can be below latent demand when availability is constrained.
It flags the periods and describes a defensible treatment or uncertainty rather than inventing demand.
A weak answer treats zero sales as proof of zero demand.

### 10. What evidence justifies production use?

A strong answer asks for frozen chronological comparisons against a credible baseline, segment checks, and cost evidence.
It includes ownership, approval, monitoring, fallback, and data governance.
A weak answer relies on a vendor leaderboard or a single strong training score.

## 15. Key terms

- [Activation](#4-definitions-and-vocabulary): nonlinear transform applied to a preactivation.
- [Affine transformation](#4-definitions-and-vocabulary): weighted sum plus bias.
- [Backpropagation](#step-12--update-with-backpropagation): chain-rule computation of parameter gradients.
- [Baseline](#step-7--define-a-simple-baseline): simpler comparison available at the same origin.
- [Batch](#4-definitions-and-vocabulary): examples contributing to one parameter update.
- [Bias parameter](#4-definitions-and-vocabulary): additive learned layer value.
- [Calibration](#step-14--evaluate-future-performance): agreement between forecasts and observed frequencies or levels.
- [Commitment](#1-why-this-topic-exists): accepted operational obligation.
- [Critical fractile](#step-10--choose-loss-in-relation-to-the-decision): underage cost divided by total marginal cost in the stated setting.
- [Epoch](#4-definitions-and-vocabulary): one pass over training examples.
- [Feature](#step-4--choose-informative-inputs): input known at the prediction origin.
- [Forecast](#1-why-this-topic-exists): estimate of a future outcome under stated information.
- [Forecast origin](#step-3--establish-what-was-knowable): time at which a prediction is issued.
- [Gradient](#step-12--update-with-backpropagation): local loss derivatives with respect to parameters.
- [Hidden representation](#3-the-core-mental-model): intermediate learned activations.
- [Horizon](#4-definitions-and-vocabulary): time between forecast origin and target.
- [Label](#4-definitions-and-vocabulary): realized target used for fitting or scoring.
- [Leakage](#10-failure-modes-and-red-flags): use of information unavailable at prediction time.
- [Learning rate](#step-12--update-with-backpropagation): update step-size choice.
- [Loss](#step-10--choose-loss-in-relation-to-the-decision): numerical penalty optimized during training.
- [MAE](#step-10--choose-loss-in-relation-to-the-decision): mean absolute error, in target units.
- [MSE](#step-10--choose-loss-in-relation-to-the-decision): mean squared error, in squared target units.
- [Overfitting](#step-13--control-overfitting): fitting details that fail to transfer.
- [Parameter](#4-definitions-and-vocabulary): a learned weight or bias.
- [Pinball loss](#step-10--choose-loss-in-relation-to-the-decision): asymmetric loss for quantile prediction.
- [Plan](#1-why-this-topic-exists): operational choice combining a forecast and constraints.
- [Quantile](#step-10--choose-loss-in-relation-to-the-decision): distribution value at a specified cumulative level.
- [ReLU](#step-9--select-an-activation): activation returning the positive part of its input.
- [Regularization](#step-13--control-overfitting): constraint or procedure limiting excessive fitting.
- [Stockout censoring](#8-data-standards-and-regulatory-context): observed sales constrained by unavailable inventory.
- [Test set](#step-5--split-by-time-before-fitting-transformations): held-out final evaluation period.
- [Validation set](#step-5--split-by-time-before-fitting-transformations): held-out model-selection period.
- [Weight](#4-definitions-and-vocabulary): learned coefficient on a layer input.

## 16. Sources

1. Goodfellow, Bengio, and Courville, *Deep Learning*, MIT Press, 2016, chapters 6–8 and 11. [Publisher's open text](https://www.deeplearningbook.org/). Feed-forward networks, optimization, and regularization.
2. Rumelhart, Hinton, and Williams, “Learning representations by back-propagating errors,” *Nature* 323, 533–536, 1986. [DOI: 10.1038/323533a0](https://doi.org/10.1038/323533a0). Foundational backpropagation reference.
3. Gneiting, “Making and Evaluating Point Forecasts,” *Journal of the American Statistical Association* 106, 746–762, 2011. [DOI: 10.1198/jasa.2011.r10138](https://doi.org/10.1198/jasa.2011.r10138). Forecast functionals and consistent scoring.
4. Koenker and Bassett, “Regression Quantiles,” *Econometrica* 46, 33–50, 1978. [DOI: 10.2307/1913643](https://doi.org/10.2307/1913643). Quantile regression and asymmetric loss.
5. Salinas, Flunkert, Gasthaus, and Januschowski, “DeepAR: Probabilistic forecasting with autoregressive recurrent networks,” *International Journal of Forecasting* 36, 1181–1191, 2020. [DOI: 10.1016/j.ijforecast.2019.07.001](https://doi.org/10.1016/j.ijforecast.2019.07.001). Shared neural probabilistic forecasting.
6. Makridakis, Spiliotis, and Assimakopoulos, “The M4 Competition: 100,000 time series and 61 forecasting methods,” *International Journal of Forecasting* 36, 54–74, 2020. [DOI: 10.1016/j.ijforecast.2019.04.014](https://doi.org/10.1016/j.ijforecast.2019.04.014). Competition design and results.
7. Makridakis, Spiliotis, and Assimakopoulos, “M5 accuracy competition: Results, findings, and conclusions,” *International Journal of Forecasting* 38, 1346–1364, 2022. [DOI: 10.1016/j.ijforecast.2021.11.013](https://doi.org/10.1016/j.ijforecast.2021.11.013). Walmart retail benchmark and evaluation.
8. GS1, [Global Trade Item Number](https://www.gs1.org/standards/id-keys/gtin). Official identifier description; not a substitute for an organization's effective-dated master data.
9. NIST, [AI Risk Management Framework 1.0](https://www.nist.gov/itl/ai-risk-management-framework), 2023. Voluntary AI risk guidance.
10. European Union, [Regulation (EU) 2016/679, General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj), 2016. Official legal text; applicability depends on the processing.
11. European Union, [Regulation (EU) 2024/1689, Artificial Intelligence Act](https://eur-lex.europa.eu/eli/reg/2024/1689/oj), 2024. Official legal text, including phased application provisions.
12. Hyndman and Athanasopoulos, *Forecasting: Principles and Practice*, third edition, [time-series cross-validation](https://otexts.com/fpp3/tscv.html). Rolling-origin evaluation principles.

The academic papers establish methods or report results under their own experimental conditions.
The official standards and regulations establish their stated definitions or requirements, not the commercial value of this particular model.
The numerical cases and synthetic code in this topic are educational illustrations.

## 17. Links

- [Module 18 topic index](./_index.md).
- [Module 18 survey and hands-on labs](../../../course/18_deep_learning_foundation_models_llm_agents.md).
- [Sibling topic 02: Sequence Architectures](./_index.md#3-topics) — see the indexed topic; its detailed file is planned.
- [Sibling topic 03: Global versus Local Models](./_index.md#3-topics) — see the indexed topic; its detailed file is planned.
- [Module 30 simulation and production labs index](../../05_labs_electives/30_simulation_optimization_production_labs/_index.md).
- [Course expansion specification](../../../course/MODULE_EXPANSION_SPEC.md).
