# Outcomes Analysis and Backtesting

> Module 26 — Quality, Evaluation, and Model Risk  
> Topic 04 of 10  
> Estimated study time: 4 hours  
> Prerequisites: basic arithmetic, forecast errors, decision framing, and value measurement; Modules 07 and 25 provide useful context.  
> Version 2026.5

## 1. Why this topic exists

A supply-chain model can be theoretically sensible and still produce poor outcomes in use.
A demand forecast can miss a new product launch, an allocation model can favor the wrong customers, and an estimated delivery date can be accurate on average while systematically late for critical materials.
The organization needs evidence about those outcomes before it relies on the model and after conditions change.

Backtesting reconstructs a sequence of past decisions using information that would have been available at each decision time.
It compares the resulting predictions or decisions with subsequently observed outcomes.
The exercise helps a model owner answer whether a proposed method would have performed better than a credible alternative under a specified historical operating regime.
It does not automatically establish that deploying the model caused a business benefit.

A planner may need to choose next week's order quantities before demand is known.
The model developer wants a fair estimate of prediction performance.
The finance partner wants a defensible estimate of financial exposure and realized benefit.
The data owner must recover historical inputs as they were known at the time.
An independent validator needs to challenge both the model and the evaluation.
The accountable business owner must decide whether the remaining uncertainty is acceptable for a bounded use.

A random train-test split can give an attractive answer while allowing future promotions or revised history into earlier predictions.
An unrealistically weak benchmark can make a marginal model look valuable.
A backtest that ignores capacity, override behavior, and lead time can measure a forecast without measuring the decision.
A benefits report that compares post-deployment savings with an unrelated prior period may mistake seasonality or price changes for model impact.

The aim is not to produce one universal score.
The aim is to build a chain of evidence from an operational decision to a reproducible historical comparison, then to prospective outcome measurement.
Each link has a different uncertainty and owner.

### Prerequisite bridge

Module 06 is deferred in the expanded edition, so this topic does not rely on its expanded material.
A **feature** is a model input available at prediction time.
A **target** is the quantity the model is intended to predict.
A **training set** is the historical information used to estimate model parameters.
A **holdout** is data withheld from development and selection until the evaluation is locked.
An **error** is an observed value minus its prediction; this sign convention is used throughout.
A **loss** converts a prediction or decision error into a penalty.
A **counterfactual** is the outcome that would have occurred under an alternative action for the same decision unit; it is usually not directly observable.

A forecast is an estimate of an uncertain future quantity.
A plan is a proposed set of actions.
A target is a desired outcome, which need not equal the forecast.
A commitment is an authorized promise to a customer or supplier.
The worked example keeps these four quantities separate.

## 2. Learning outcomes

After working through this topic, you can produce the following artifacts.

1. **Define** an outcome evaluation contract with decision unit, horizon, cutoff, and ownership.
   Exercise 1 checks the contract.
2. **Construct** a rolling-origin design with explicit training, selection, and final evaluation periods.
   Exercise 2 checks the chronology.
3. **Identify** leakage and rebuild a point-in-time input.
   Exercise 3 checks information availability.
4. **Compare** a champion, a simple benchmark, and a challenger on common eligible decisions.
   Exercise 4 checks comparator fairness.
5. **Calculate** forecast errors and an asymmetric decision-loss proxy.
   Exercise 5 checks the arithmetic.
6. **Critique** a claimed realized business benefit and choose stronger evidence.
   Exercise 6 checks causal reasoning.
7. **Design** an evidence package, decision rule, and monitoring response.
   Exercises 7 and 8 check operational governance.

A learning outcome is not met merely by naming a metric.
You should be able to explain the decision it informs and the information that the metric cannot provide.

## 3. The core mental model

```text
Business decision and approved use
        |
        v
Decision time, horizon, action, constraints, fallback
        |
        v
Historical information as known at each origin
        |
        +-----> benchmark and champion under the same rules
        |                    |
        +-----> challenger  |
                    \       /
                     v     v
               locked predictions
                     |
                     v
         later outcomes and decision replay
                     |
          +----------+----------+
          v                     v
  predictive evidence     economic proxy and harms
          \                     /
           +---------+---------+
                     v
       independent challenge and decision
                     |
                     v
       prospective monitoring and value study
```

Read the diagram from top to bottom.
A forecast is useful only relative to a decision, its available actions, and its consequences.
Each historical origin is a simulated moment of ignorance about the future.
The evaluation must preserve that ignorance, including data publication delays and model-selection choices.

The historical replay produces a prediction for each eligible origin, item, location, and horizon.
The same keys and outcome definition must be used for all methods.
The benchmark provides a minimum standard of practical usefulness.
The champion represents the current authorized process, including relevant human overrides if those are part of the process.
The challenger is a proposed alternative that must earn its place through comparable evidence.

The first evidence layer measures predictive performance.
The second layer estimates decision consequences under an explicit action and cost model.
The third layer asks whether actual deployment changed operational or financial outcomes relative to a credible counterfactual.
Improvement in an earlier layer is evidence for, but not proof of, improvement in a later one.

A historical replay may show lower simulated shortage penalties.
That result depends on the modeled cost, inventory state, supplier response, and action policy.
It does not show that the deployed organization actually saved that amount.
A prospective randomized or otherwise credible comparative study can provide stronger evidence of impact, subject to its own assumptions.

### Three clocks

The **event clock** records when an underlying event occurred.
The **availability clock** records when a decision-maker or system could first know it.
The **evaluation clock** records when the outcome became sufficiently mature to score.
These clocks are often different.
An order entered on Monday, corrected on Thursday, and finalized next month must not be represented by its final value in a Monday decision replay unless that final value was actually known on Monday.

### The unit of comparison

Compare methods on the same eligible decisions, not merely on the same number of rows.
A decision key can be `(item, location, forecast origin, horizon)`.
If one method cannot produce a prediction for a key, record that failure and apply a predeclared fallback or report coverage separately.
Silently removing difficult keys makes the comparison selective.

### The evidence boundary

A backtest asks what would have happened to a metric under reconstructed information and a stated replay rule.
It does not observe all actions that would have followed a different recommendation.
If the action changes demand, supplier capacity, or customer behavior, fixed historical outcomes may no longer be valid for that alternative.
Document that boundary before using the result in an investment or approval decision.

## 4. Definitions and vocabulary

| Term | Definition and unit | Common misuse |
|---|---|---|
| Decision unit | Entity and moment at which an action is chosen, such as item-location-week | Treating correlated rows as independent decisions |
| Forecast origin | Timestamp at which the forecast is issued | Using the outcome timestamp as the origin |
| Horizon | Time from origin to predicted period, measured in days or weeks | Combining horizons without reporting their mix |
| Target | Future quantity being estimated, such as units of demand per week | Calling an order plan the target |
| Outcome | Later observed measurement used for scoring, in target units | Assuming shipments always equal unconstrained demand |
| As-of snapshot | Data values available no later than the decision cutoff | Using the latest corrected value for every past origin |
| Event time | Time an event occurred | Assuming it equals its availability time |
| Availability time | Earliest time the decision process could access a value | Using a date of business occurrence as proof of availability |
| Label maturity | Time when the outcome is sufficiently complete for the defined evaluation | Scoring incomplete returns as final outcomes |
| Leakage | Information unavailable at the origin influencing training, selection, features, or scoring | Limiting the concept to an obviously future-dated column |
| Rolling origin | Repeated chronological evaluations at successive forecast origins | Randomly shuffling rows and calling it temporal validation |
| Expanding window | Training window that grows as later data become available | Assuming all old data remain relevant |
| Sliding window | Training window with a fixed or bounded lookback | Ignoring the loss of useful seasonal history |
| Training period | Historical records used to fit a model | Fitting once with data from after an evaluated origin |
| Selection period | Period used to choose features, hyperparameters, or policy | Calling repeatedly inspected selection data a final holdout |
| Final holdout | Later data reserved for one locked evaluation | Retuning after seeing its result without declaring reuse |
| Gap or embargo | Excluded time interval between samples when needed to prevent overlap or delayed-label leakage | Applying an arbitrary gap without relating it to information flow |
| Benchmark | Simple, reproducible comparator with the same information set | Choosing an intentionally poor straw man |
| Champion | Current authorized model or operational process | Assuming a model alone represents an override-heavy process |
| Challenger | Alternative method evaluated against the champion | Promoting a method based on a noncomparable sample |
| Naive forecast | Simple reference such as last observed value | Using a value that was not yet published at the origin |
| Seasonal naive | Forecast equal to the corresponding prior seasonal period | Assuming a prior season is available for every new item |
| MAE | Mean absolute error, in target units | Comparing MAE across differently scaled populations without context |
| WAPE | Sum of absolute errors divided by sum of absolute actuals, dimensionless | Interpreting it when the denominator is zero or demand is censored |
| Bias | Average signed error under a stated sign convention, in target units | Treating small aggregate bias as proof that every segment is unbiased |
| Coverage | Fraction of eligible decisions with usable forecasts, dimensionless | Excluding missing predictions from the comparison without disclosure |
| Quantile coverage | Fraction of outcomes below a stated predicted quantile | Confusing quantile coverage with service level |
| Decision loss | Stated penalty associated with an action and an outcome, often currency per decision | Treating a simplified proxy as realized accounting savings |
| Counterfactual | Unobserved outcome under an alternative action | Assuming the observed before period is automatically the counterfactual |
| Censoring | Observation is constrained by the process, for example sales capped by stock | Calling a stockout-constrained sale true demand |
| Selection bias | Difference caused by who or what enters a sample or receives treatment | Attributing all post-deployment differences to the model |
| Interference | One unit's treatment changes another unit's outcome | Randomizing items independently when they share a capacity constraint |
| Guardrail | Metric or rule that limits acceptable harm during a trial | Reporting only average benefit while hiding service degradation |
| Evidence dossier | Versioned record of data, code, decisions, results, limitations, and approvals | Saving only a chart or slide deck |
| Revalidation trigger | Predefined event requiring renewed review | Waiting for an annual review after a material change |

## 5. The mechanics, step by step

### Step 1 — Contract the decision

**Input:** a proposed use, business owner, and operating process.
**Output:** a written evaluation contract.

1. Name the decision-maker and the action they can actually take.
2. Identify the item, location, customer, or route to which the action applies.
3. Record the decision time, time zone, and operational cutoff.
4. Define the prediction horizon and the action's effective period.
5. Identify the target quantity and its measurement unit.
6. State what happens when the model is unavailable or its output is rejected.
7. Identify decision constraints such as minimum order quantity, supplier capacity, and budget.
8. Specify the business harm of both underaction and overaction.

A demand forecast can support a weekly replenishment decision, but the forecast itself is not an order.
The action policy transforms a forecast, available inventory, outstanding orders, and constraints into an order.
A forecast evaluation alone cannot establish that the action policy is beneficial.

Branch: if no authorized user can describe the action, stop and clarify the use before choosing a metric.
Branch: if the model is advisory, include the human decision and override process in the intended-use description.
Branch: if the model is autonomous, specify hard bounds, rollback authority, and incident handling.

### Step 2 — Define the target and outcome

**Input:** decision contract and source-system definitions.
**Output:** target specification and outcome-quality rules.

Define whether the target is requested demand, accepted orders, shipments, consumption, or another quantity.
Specify aggregation, unit conversion, returns, cancellations, and treatment of substitutions.
Record when the outcome becomes mature enough for scoring.

Observed shipments may be capped by available inventory.
If stockouts occur, comparing a demand forecast to shipments can reward underforecasting.
Collect lost-sales, backorder, or availability indicators where available.
If true unconstrained demand cannot be reconstructed, state that the observed outcome is censored and qualify conclusions.
Do not manufacture an imputed truth without documenting the estimation method and its uncertainty.

Branch: if the target is revised, preserve both its as-known and final versions.
Use as-known values for training at each origin and a predeclared mature version for outcome scoring.
Explain why the chosen mature version is appropriate for the business question.

### Step 3 — Inventory data and information availability

**Input:** source tables, data contracts, and historical system records.
**Output:** point-in-time feature inventory.

For every field record:

- Business meaning, unit, and entity key.
- Event time and first availability time.
- Publication schedule and ordinary delay.
- Revision behavior and availability of old versions.
- Missingness, quality checks, and permitted fallback.
- Owner and access permissions.
- Earliest reliable historical coverage.

A promotion calendar may have an event date two weeks after the origin but still be a valid feature if the promotion was approved and visible before the origin.
Conversely, an invoice correction describing last month's event can leak if the correction was published only after the historical forecast.
The validity test is availability to the actual decision process, not whether the field's name contains a date.

Use a point-in-time join that selects the latest record available at or before the cutoff for each origin and entity.
Resolve ties deterministically, and retain both the source timestamp and the version identifier.
Do not substitute a modern table snapshot when historical versions were not retained without marking that as a limitation.

### Step 4 — Design chronology before fitting

**Input:** decision cadence, history, label delays, and intended deployment period.
**Output:** locked chronological evaluation plan.

Choose a development period for initial exploration and fitting.
Choose a later selection period for tuning candidate methods and policies.
Reserve a still later final holdout for a locked evaluation.
The separation should reflect the intended deployment cadence and available history rather than a universal percentage.

Within development and selection, create rolling origins.
At each origin, fit or update only on data and labels available by that origin.
Predict the required future horizon and wait for its outcome to mature.
Repeat over enough periods to include relevant seasonality and operating regimes where history permits.

An expanding window uses all eligible prior history.
A sliding window uses a defined recent lookback.
Both are legitimate candidates if selected using only the selection period.
Compare them when structural change makes old data potentially less relevant, and record the trade-off in seasonal coverage.

A gap may be required when labels overlap in time or mature with a delay.
For example, an eight-week target ending after the forecast origin cannot be used as a training label until its outcome is observable.
The required gap depends on target construction, publication delay, and model update timing.
A gap alone does not fix revised features, global preprocessing, or tuning on the holdout.

Branch: if the history covers only one season, say that a repeat-season conclusion is unsupported.
Branch: if the process is event-driven rather than weekly, define origins at actual decision events.
Branch: if the business has materially changed, test older and newer regimes separately and explain the scope of transportability.

### Step 5 — Prevent leakage throughout the pipeline

**Input:** full transformation and model workflow.
**Output:** evidence that each learned operation respects the origin.

Leakage can enter before model fitting.
A dataset assembled from today's item catalog may exclude discontinued products that were active historically.
A global median imputer may use later observations to fill earlier missing inputs.
A scaler fitted on the entire dataset may reveal the distribution of the holdout.
Feature selection on all dates can use holdout outcomes indirectly.
A forecast adjusted retrospectively using the actual promotion result is not a historical forecast.

For every origin:

1. Restrict source records by availability time.
2. Restrict training labels by maturity time.
3. Fit imputers, scalers, encoders, and feature selectors on eligible training data only.
4. Apply those fitted transformations to the origin's input.
5. Select model settings only from earlier development or selection evidence.
6. Preserve the prediction before the outcome is known.
7. Score only after the outcome meets the declared maturity rule.

Review derived variables as well as raw columns.
A rolling mean that includes the target week leaks even when its inputs have innocent names.
A supplier lead-time feature calculated from the completed shipment being predicted leaks its own eventual duration.
A stockout indicator recorded after the selling week is an outcome or diagnostic, not a valid pre-week input unless a genuinely available prior indicator is used.

Negative controls and availability audits can reveal suspicious behavior, but they cannot prove absence of leakage.
Inspect lineage and timestamps, not just performance.
A sudden improvement after adding a field whose availability cannot be demonstrated is a reason to suspend that feature.

### Step 6 — Specify comparators

**Input:** current process, permitted information, and alternative methods.
**Output:** comparator registry and fallback rules.

Use at least one simple benchmark that can be computed from data actually available at each origin.
For a stable weekly series, the latest available observation may be useful.
For a strongly seasonal series, a corresponding prior-season observation may be more meaningful.
A simple moving average may be appropriate when noise makes the last observation unstable.
No single benchmark is best in every series.

The champion is the current authorized process, not necessarily the previous algorithm alone.
If planners adjust the model, retain the pre-override forecast, final operational forecast, timestamp, and override reason.
A comparison with an unadjusted model should not be called a comparison with the operational champion when overrides are material.

A challenger can be a new model, a simpler method, or a revised action policy.
Give it the same permissible information, decision timing, eligible population, and outcome definition.
If a challenger needs new data or a longer processing window, show the operational cost and feasibility rather than quietly giving it an unfair advantage.

For missing predictions, predeclare a fallback such as the benchmark where feasible.
Report both coverage and the score of the deployed fallback policy.
If the fallback cannot be computed, mark the decision unserved and evaluate its operational consequences separately.

### Step 7 — Freeze selection and final evaluation

**Input:** candidate methods and a selection-period scorecard.
**Output:** frozen model, policy, and final-holdout protocol.

Use development data for method construction.
Use the selection period to choose among candidates, tune settings, and set thresholds.
Document the candidate set, selection criterion, and material experiments.
Freeze features, transformations, policy rules, and evaluation code before accessing final outcomes.

The final holdout evaluates the selected process under the locked rules.
It may include retraining at each origin if the retraining schedule and algorithm were locked in advance and use only then-available data.
It must not include discretionary tuning based on holdout performance.
If the holdout is inspected and the method is changed in response, that period has become selection evidence.
Use a later untouched period or prospective evaluation for a fresh final assessment, and disclose the prior inspection.

Repeatedly testing many challengers on the same holdout creates selection optimism.
Record the number and nature of comparisons and avoid presenting the best observed score as though it were an independent first test.
A validator should have access to unsuccessful as well as successful experiments where feasible.

### Step 8 — Produce a paired scorecard

**Input:** locked forecasts and mature outcomes for common decision keys.
**Output:** aggregate and segmented evaluation results.

For each eligible key calculate signed error as actual minus forecast.
Calculate absolute error and any decision-specific loss under the predeclared policy.
Aggregate results using explicitly stated weights and denominators.

MAE is the sum of absolute errors divided by the number of scored decisions.
WAPE is the sum of absolute errors divided by the sum of absolute actuals; it is undefined when the denominator is zero.
Signed bias is the mean actual-minus-forecast error.
Positive bias under this convention means the method underforecast on average.
Opposite errors can cancel, so bias must be interpreted alongside absolute error and segment results.

Report item-location coverage, missing predictions, excluded records, and mature outcome coverage.
Slice results by horizon, volume, product lifecycle, location, season, and operational importance where sample size permits.
Identify low-volume segments where percentage metrics become unstable.
If comparing quantile forecasts, check coverage against the nominal quantile and review sharpness as well as calibration.

For a paired comparison, use the same keys and compute each key's challenger loss minus champion loss.
This preserves the common shocks affecting both methods.
The average difference estimates the historical improvement for the evaluated population, not a universal future effect.

### Step 9 — Characterize uncertainty and dependence

**Input:** paired losses, decision clusters, and the evaluation period.
**Output:** uncertainty statement and sensitivity analysis.

Weekly observations from the same item are usually dependent.
Items in the same location may share promotions, capacity, or weather shocks.
Multiweek forecast horizons can overlap and share actual outcomes.
Treating all rows as independent can make uncertainty intervals misleadingly narrow.

Choose an uncertainty method appropriate to the design.
For example, a block bootstrap can resample contiguous time blocks long enough to preserve relevant temporal dependence.
Cluster resampling can preserve within-location dependence when there are enough independent locations.
When there are few periods or clusters, report that uncertainty is poorly identified rather than claiming precision from a mechanically computed interval.

State the resampling unit, number of blocks or clusters, interval method, and limitations.
Report the estimated difference and interval, not only whether a threshold was crossed.
Examine sensitivity to exceptional weeks, alternative weights, and plausible cost assumptions.
Predeclare exclusions and distinguish planned analyses from post hoc investigations.

### Step 10 — Translate forecasts into decisions

**Input:** forecasts, initial operational state, action policy, constraints, and cost parameters.
**Output:** simulated decision outcomes and a traceable cost bridge.

A one-period proxy may charge different amounts for underforecast and overforecast units.
It is useful for showing the consequence of asymmetric errors.
It is not an inventory simulator and does not establish actual stockouts, inventory balances, or accounting savings.

For a more realistic replay, initialize inventory and open orders at each relevant decision time.
Apply the same feasible policy to each method, including ordering lead time, capacity, minimum order quantities, and authorized overrides.
Update simulated inventory as receipts, demand, and returns occur.
Specify how unmet demand is treated: lost, backordered, substituted, or delayed.
Keep scenarios internally consistent across time; do not reset inventory after each week unless the business process truly resets it.

Price, margin, holding cost, and expedite cost must be tied to sources, units, and effective dates.
Avoid double counting a lost margin that is already included in a shortage penalty.
Separate accounting costs from opportunity costs and distinguish one-time implementation costs from recurring operating costs.

Branch: if the action can change customer demand or supplier behavior, replaying the observed outcome unchanged is a structural assumption.
Use sensitivity analysis and consider a prospective experiment to measure behavioral response.

### Step 11 — Design prospective outcome evidence

**Input:** deployment proposal, eligible population, interference risks, and guardrails.
**Output:** evaluation design and benefit measurement plan.

A prospective randomized trial can assign eligible decision units to the champion process or challenger process before outcomes are known.
Randomize at a level that reduces contamination, such as location or product family when inventory capacity is shared.
Specify eligibility, assignment, rollout duration, primary outcome, and stopping rules before observing results.
Track actual exposure, overrides, exclusions, and missing outcomes.

Analyze by assigned group when estimating the effect of offering the new process, including units that did not fully comply.
A per-use or as-treated analysis can be informative but is vulnerable to selection because users choose when to follow recommendations.
Explain the estimand and any assumptions before interpreting it causally.

If randomization is infeasible, consider a matched comparison, staggered rollout, or interrupted time-series design.
Each requires assumptions about comparability, concurrent changes, and trends.
A before-after comparison alone is vulnerable to seasonality, promotions, price changes, and regression to the mean.
A difference-in-differences design requires a defensible parallel-trends assumption and attention to other concurrent interventions.
Pre-period checks can reveal some violations but cannot prove that the assumption holds after rollout.

Predefine safety guardrails such as service level, expedite volume, supplier burden, and prohibited segment deterioration.
An average saving cannot justify an unacceptable safety or service harm merely because its financial score is positive.
Identify who can pause the trial and restore the champion process.

### Step 12 — Make and preserve the decision

**Input:** scorecard, independent challenge, uncertainty, operational controls, and value evidence.
**Output:** approval, conditional approval, or rejection with rationale.

The validator checks reproducibility, information availability, benchmark fairness, outcome quality, and the reasonableness of the decision proxy.
The business owner assesses whether the evidence supports the intended decision and operational risk.
Finance reviews the benefit bridge and cost assumptions.
The data owner acknowledges unresolved quality and lineage limitations.
The accountable approver records the permitted scope, restrictions, and review date.

Keep versioned source references, code, model settings, historical snapshots, run logs, predictions, outcomes, and scorecards.
Record exclusions, failed runs, selection history, and deviations from the original protocol.
Protect commercially sensitive records with access controls and retention rules.
Reproduction should recreate the result from retained artifacts, not rely on a developer's memory.

Specify triggers for revalidation: material changes to model logic, data availability, target definition, supplier lead time, product mix, decision policy, or operational use.
Monitoring should compare predictions and actions with mature outcomes, including coverage and guardrails.
A breach should have a named owner, escalation path, investigation window, and fallback.
A dashboard without a response authority is not a control.

### Practical chronology: a four-week demand target

The following miniature calendar makes the maturity rule concrete.
It is an independent illustration, not an extension of the six-week example below.
Assume forecasts are issued at 09:00 each Monday.
Each target covers the four complete weeks beginning on that Monday.
Assume the target is finalized seven calendar days after the target window ends.
The exact timestamp of finalization must be checked against the cutoff.

| Forecast origin | Target starts | Target ends | Assumed label available | Available for the next Monday origin? |
|---|---|---|---|---|
| 1 January, 09:00 | 1 January | 28 January, end of day | 4 February, end of day | Yes, on 5 February at 09:00 |
| 8 January, 09:00 | 8 January | 4 February, end of day | 11 February, end of day | Yes, on 12 February at 09:00 |
| 15 January, 09:00 | 15 January | 11 February, end of day | 18 February, end of day | Yes, on 19 February at 09:00 |

At 5 February, the 1 January label is eligible under these assumptions.
The 8 January label is not yet eligible, even though its target period has just ended.
The 15 January label is also unavailable.
A developer who filters training data only by forecast origin could accidentally use both future labels.
Filtering on target-end time is not enough when finalization has an additional delay.
The correct filter uses the actual or conservatively assumed label-availability timestamp.

The target windows overlap.
The 1 January and 8 January forecasts both contain demand from 8 through 28 January.
Their forecast errors consequently share realized demand shocks.
A confidence interval that treats those errors as independent can substantially overstate the effective amount of evidence.
The overlap is a reason to consider time blocks and to report how many distinct calendar weeks and cycles the evaluation covers.
It is not a reason to discard all rolling-origin evaluation.

The same calendar applies to transformations involving target history.
If a lagged four-week demand feature requires a mature label, it cannot be refreshed until that label is available.
An independently available daily demand feed could support a different feature, but its own availability and revision rules must be established.
The evaluator should not infer daily availability from a weekly finalized table.

If the target is revised after finalization, document whether training uses the first mature version or later revisions.
Either approach may be operationally defensible if it matches production.
For an origin after a revision, the revised value may be eligible.
For an origin before the revision, it is not.
Maintain an auditable version history rather than retroactively overwriting the earlier value.

### Practical benchmark design for difficult populations

A single benchmark may have different failure modes across the portfolio.
Use the decision contract to decide which comparator or fallback can actually run for each key.
Record benchmark eligibility before evaluating the challenger.

| Population | Candidate reference | Availability question | Interpretation limit |
|---|---|---|---|
| Stable established product | Last available weekly demand | Was last week's value published before cutoff? | Sensitive to one unusual week |
| Repeating seasonal product | Corresponding prior-season demand | Was the prior season observed for this item? | Breaks when calendar or promotion timing shifts |
| Noisy stable product | Mean of recent available weeks | Were all included weeks known at origin? | Can react slowly to a regime change |
| New product | Analog item or approved launch plan | Was the analog choice made before launch? | Analog choice can itself introduce hindsight |
| Intermittent product | Simple intermittent-demand rule | Is sufficient nonzero history available? | Average accuracy can conceal rare expensive shortages |
| Discontinued product | Explicit end-of-life rule | Was discontinuation known by the cutoff? | Today's catalog may conceal historical active status |

For a new product, selecting the best historical analog after seeing the new product's outcome is leakage.
Define an analog selection rule based only on attributes available before launch.
Retain candidate analogs, the selected analog, and the reason for selection.
If no suitable analog existed, report that limitation rather than silently excluding the product.

For an intermittent product, a forecast of zero can appear accurate for many weeks.
Evaluate the operational consequence when nonzero demand arrives.
Inspect error aggregation over the actual replenishment lead time as well as per-week scores.
Do not infer that a sparse weekly score automatically reflects cumulative inventory risk.

For a changing assortment, define the eligible universe separately at each origin.
An item should not be considered known to be discontinued merely because it is absent from today's active catalog.
Conversely, an item introduced later should not appear as an eligible past decision.
Retain lifecycle status with its own availability timestamp.

When a benchmark cannot run, the scorecard needs a specific disposition.
One option is a prior-approved fallback based on an available analog or planner plan.
Another is an unserved flag with a separate operational review.
Do not replace a missing benchmark with the eventual actual.
Do not let each model select its own scoring population.

### Practical value bridge

A decision evaluation and a finance evaluation answer related but distinct questions.
The first asks whether the decision policy is expected to improve an operational outcome.
The second asks which changes appear in financial accounts, cash flow, or a documented opportunity-cost estimate.
Trace the bridge one step at a time.

1. Start with an observed or simulated operational change, such as fewer expedited units.
2. Identify the period, decision units, and comparison process.
3. Multiply units by an appropriate incremental cost per unit only when that cost is actually avoided.
4. Separate one-time and recurring costs.
5. Subtract additional data, software, labor, and operating expenses in the same defined boundary.
6. Identify whether the result is cash, accounting expense, working capital, or an opportunity-cost estimate.
7. State the evidence design that attributes the change to the intervention.
8. Document uncertainty, exclusions, and any benefits shared with another initiative.

For example, assume a valid trial estimates 40 fewer expedited units in one month.
If each unit would otherwise incur an incremental USD 3 expedite charge, the gross avoided charge is USD 120 for that month.
If the new process incurs USD 35 in additional monthly operating expense, the net difference for those two included items is USD 85.
This illustrative calculation excludes implementation cost, tax, other service effects, and uncertainty.
It is not a claim about the worked example or any company.
If the USD 3 charge was not actually avoided because a fixed freight minimum still applied, the USD 120 calculation would overstate financial benefit.

An inventory reduction is a change in a balance, not automatically an equal recurring income benefit.
The value of released working capital depends on the financial question, timing, funding assumptions, and whether the reduction persists.
Do not add the full inventory balance reduction to an annual carrying-cost saving as though both were independent annual income benefits.
Coordinate the accounting treatment with finance.
Maintain the operational metric even when the financial conversion is uncertain.

A trial estimate also needs an explicit counterfactual.
If a location's expedite units fall after deployment, identify what would have happened without deployment.
Random assignment can support that comparison when assignment and interference are controlled.
A matched location can be informative if its trend and concurrent changes are credible.
Neither design makes the cost assumptions self-validating.
Likewise, accurate cost accounting does not remove confounding from a weak impact design.

### Practical approval record

The approval decision should be short enough to be used and detailed enough to be challenged.
Attach the full technical evidence rather than burying essential restrictions in an appendix.
The record should answer each of the following questions.

- **Scope:** Which item-location populations, horizons, users, and decisions are authorized?
- **Evidence:** Which independent period, origins, benchmarks, and outcomes support the decision?
- **Coverage:** What fraction of eligible decisions were scored and what happens for missing predictions?
- **Quality:** Which known data defects or censored outcomes affect interpretation?
- **Performance:** What were paired results, meaningful segments, and material uncertainty?
- **Economics:** Which effects are simulated, observed, attributed, or still unmeasured?
- **Controls:** What action bounds, overrides, and fallback are required?
- **Ownership:** Who operates, validates, approves, monitors, and can pause use?
- **Conditions:** Which unresolved findings must be addressed, by whom, and by when?
- **Triggers:** What data, model, policy, or operating changes require revalidation?
- **Retention:** Where are the exact data versions, code, forecasts, and approval stored?
- **Review:** When will the decision be revisited and what evidence will be required?

A conditional approval is not a blank check.
Its conditions should be measurable and relevant to the remaining risk.
If historical snapshots are unavailable for a critical segment, the approver can restrict that segment or require new prospective evidence.
If a service guardrail cannot be monitored in time to intervene, the approver may reject autonomous use even when the average forecast score improves.
The final decision is a risk judgment supported by evidence, not a mechanical consequence of one metric.

## 6. Worked example

### Decision and scope

A distributor reviews one item at one location each Friday for the following week.
The decision is how many units to make available for next week's customer demand.
The model produces a weekly demand forecast.
An illustrative policy sets the one-week availability plan equal to that forecast.
The example ignores lead time, beginning inventory, case packs, capacity, returns, and substitution.
That simplification makes the calculation a decision-loss proxy, not a realized inventory result.

The planner owns the decision.
The authorized commitment to customers remains a separate allocation decision.
A service target of 98% is a management aspiration, not a forecast or a guaranteed commitment.
The example does not calculate service level and cannot establish whether that target was met.

All values below are invented teaching data, not company observations.
The six weeks are the locked final evaluation period.
Assume both methods were fixed using earlier data and each forecast was issued before its corresponding outcome.
The example does not demonstrate actual model training or recover real historical snapshots.

### Methods and inputs

The benchmark uses the preceding week's demand, assumed available by the Friday cutoff.
The proposed challenger supplies a precomputed forecast under the same cutoff.
Actual demand is assumed fully observed and mature in units per week.
The first benchmark forecast uses a prior-week actual of 100 units.

| Week | Actual demand, units | Benchmark forecast, units | Challenger forecast, units |
|---:|---:|---:|---:|
| 1 | 102 | 100 | 105 |
| 2 | 118 | 102 | 112 |
| 3 | 95 | 118 | 101 |
| 4 | 131 | 95 | 120 |
| 5 | 109 | 131 | 115 |
| 6 | 125 | 109 | 120 |

The benchmark is explicitly the immediately preceding actual, so week 2 uses 102 units.
This differs from a constant 100-unit baseline and must be preserved in both arithmetic and code.

### Error calculations

For each method, signed error equals actual minus forecast.
Absolute error is the magnitude of signed error.

| Week | Benchmark error, units | Benchmark absolute error, units | Challenger error, units | Challenger absolute error, units |
|---:|---:|---:|---:|---:|
| 1 | 2 | 2 | -3 | 3 |
| 2 | 16 | 16 | 6 | 6 |
| 3 | -23 | 23 | -6 | 6 |
| 4 | 36 | 36 | 11 | 11 |
| 5 | -22 | 22 | -6 | 6 |
| 6 | 16 | 16 | 5 | 5 |
| **Total** | **25** | **115** | **7** | **37** |

Benchmark MAE = 115 / 6 = 19.17 units per week, rounded to two decimals.
Challenger MAE = 37 / 6 = 6.17 units per week.
The paired reduction in MAE is (115 - 37) / 6 = 13.00 units per week.
The relative reduction is 78 / 115 = 67.83% of benchmark absolute error.

Benchmark bias = 25 / 6 = 4.17 units per week.
Challenger bias = 7 / 6 = 1.17 units per week.
Both are net underforecasts under the stated sign convention.
That aggregate does not mean every week was underforecast.

Total actual demand is 102 + 118 + 95 + 131 + 109 + 125 = 680 units.
Benchmark WAPE = 115 / 680 = 16.91%.
Challenger WAPE = 37 / 680 = 5.44%.
These percentages describe this six-week sample, not a guaranteed future accuracy rate.

### Decision-loss proxy

Assume an underforecast incurs a proxy penalty of USD 8 per unit and an overforecast incurs USD 2 per unit.
The penalty represents an illustrative asymmetric decision consequence, not a measured accounting cost.
For signed error `e`, loss is `8 * max(e, 0) + 2 * max(-e, 0)` in USD.

| Week | Benchmark loss, USD | Challenger loss, USD | Calculation for challenger |
|---:|---:|---:|---|
| 1 | 16 | 6 | 3 excess units × USD 2 |
| 2 | 128 | 48 | 6 shortage-proxy units × USD 8 |
| 3 | 46 | 12 | 6 excess units × USD 2 |
| 4 | 288 | 88 | 11 shortage-proxy units × USD 8 |
| 5 | 44 | 12 | 6 excess units × USD 2 |
| 6 | 128 | 40 | 5 shortage-proxy units × USD 8 |
| **Total** | **650** | **206** | Six weekly decisions |

Benchmark positive errors total 70 units and negative-error magnitudes total 45 units.
Its proxy loss is 70 × USD 8 + 45 × USD 2 = USD 650.
Challenger positive errors total 22 units and negative-error magnitudes total 15 units.
Its proxy loss is 22 × USD 8 + 15 × USD 2 = USD 206.
The historical proxy difference is USD 650 - USD 206 = USD 444 over six weeks.
The mean difference is USD 74 per week for this single item-location.

Do not multiply USD 74 by a year or all items and report it as realized benefit without evidence of representativeness, persistent performance, feasible actions, and cost validity.
The six observations are too few to support a reliable distributional or seasonal conclusion.

### Sensitivity and interpretation

If shortage and excess penalties were both USD 2 per unit, the benchmark proxy would be 115 × USD 2 = USD 230.
The challenger proxy would be 37 × USD 2 = USD 74.
The difference would be USD 156 over six weeks.
The challenger still wins in this particular sample because it also has lower total absolute error.
This does not establish that the same ranking holds for every plausible asymmetric cost ratio or another population.

Suppose that during week 4 only 120 units could be shipped because of a stockout, while true demand was 131 units.
Scoring the benchmark and challenger against 120 shipments instead of demand would use a censored outcome.
The challenger would appear exact in that week, but the remaining 11 units of demand would be invisible.
A validator must investigate the target and availability history rather than treating the shipment score as proof of good demand forecasting.

A reasonable governance conclusion is conditional: the challenger merits a broader, predeclared evaluation and perhaps a guarded prospective trial.
Six synthetic observations cannot justify automatic rollout, a causal savings claim, or a 98% service commitment.
A real approval would need coverage, point-in-time lineage, more representative origins, uncertainty, feasible decision replay, and operational sign-off.

## 7. Code, query, or calculation

The following Python 3 program reproduces the worked example using only the standard library.
Save it as a local `.py` file and run it with `python filename.py`.
The arrays represent already locked forecasts and mature outcomes; the program does not fit a model or demonstrate leakage-free feature construction.
Its benchmark is the previous actual, including the 100-unit prior observation.
All six decisions are assumed eligible and scored, with no missing data.

```python
from statistics import mean

# One item-location, six weekly decisions. All quantities are units/week.
actual = [102, 118, 95, 131, 109, 125]
previous_actual = [100] + actual[:-1]
challenger = [105, 112, 101, 120, 115, 120]

UNDER_COST = 8.0  # illustrative USD per underforecast unit
OVER_COST = 2.0   # illustrative USD per overforecast unit


def score(actuals, forecasts):
    """Score identical, complete decision keys using actual - forecast."""
    if len(actuals) != len(forecasts) or not actuals:
        raise ValueError("actuals and forecasts must be equally sized and nonempty")
    errors = [a - f for a, f in zip(actuals, forecasts)]
    absolute = [abs(error) for error in errors]
    loss = [
        UNDER_COST * max(error, 0) + OVER_COST * max(-error, 0)
        for error in errors
    ]
    denominator = sum(abs(a) for a in actuals)
    return {
        "errors": errors,
        "mae": mean(absolute),
        "bias": mean(errors),
        "wape": sum(absolute) / denominator if denominator else None,
        "loss": sum(loss),
    }


benchmark_result = score(actual, previous_actual)
challenger_result = score(actual, challenger)

for name, result in (
    ("benchmark", benchmark_result),
    ("challenger", challenger_result),
):
    wape = result["wape"]
    wape_text = "undefined" if wape is None else f"{100 * wape:.2f}%"
    print(
        f"{name}: MAE={result['mae']:.2f} units/week, "
        f"bias={result['bias']:.2f} units/week, "
        f"WAPE={wape_text}, proxy loss=USD {result['loss']:.2f}"
    )

print(
    "proxy loss reduction=USD "
    f"{benchmark_result['loss'] - challenger_result['loss']:.2f}"
)
```

Expected output:

```text
benchmark: MAE=19.17 units/week, bias=4.17 units/week, WAPE=16.91%, proxy loss=USD 650.00
challenger: MAE=6.17 units/week, bias=1.17 units/week, WAPE=5.44%, proxy loss=USD 206.00
proxy loss reduction=USD 444.00
```

The benchmark array is constructed from actuals for concision.
In a real replay, the preceding actual must be checked for availability at each forecast origin.
A delayed observation would require the latest available value, not the latest event value.
The code checks equal lengths but has no entity keys; production scoring should join and validate unique decision keys explicitly.
It also assumes all actuals are observed and comparable.
The WAPE zero-denominator branch returns an undefined value rather than inventing a percentage.

To extend this calculation responsibly, build point-in-time snapshots and a locked origin schedule first.
Fit each candidate and every learned transformation only on the eligible history for that origin.
Preserve predictions before joining them to mature outcomes.
Then score the same keys and report forecast coverage, uncertainty, and exclusions.
For an inventory application, replace the proxy with a separately validated, stateful decision simulation where appropriate.

## 8. Data, standards, and regulatory context

### Minimum data contract

The evaluation needs more than a table of actuals and predictions.
At minimum retain the following fields with definitions and units.

| Field group | Required examples | Why it matters |
|---|---|---|
| Decision identity | Item, location, origin, horizon, time zone | Supports paired comparison and prevents accidental duplication |
| Source version | Extract ID, system, record version, availability timestamp | Reconstructs information available at the cutoff |
| Forecast | Model version, forecast value, issuance time, unit | Establishes what was actually predicted and when |
| Operational action | Final plan, override, decision-maker, reason | Distinguishes a model from the process used in practice |
| Outcome | Definition, value, event time, maturity time, revision | Defines what was scored and whether it was complete |
| Inventory context | Stock availability, stockout or backorder indicator | Reveals potential demand censoring |
| Eligibility | Lifecycle state, inclusion and exclusion reason | Makes the evaluation population auditable |
| Economics | Unit cost, currency, effective date, source | Makes simulated loss and value assumptions reviewable |
| Governance | Owner, approvals, retention and access rules | Supports accountable use and reproduction |

A point-in-time warehouse or versioned event log can support reconstruction, but the name of the technology does not establish correctness.
Check timestamps against source-system behavior, including late arrivals and corrections.
A daily snapshot may be too coarse for an intraday decision.
A field that was visible to an analyst but unavailable to the production planning system may not be a fair production input.

### Model-risk guidance

The U.S. Federal Reserve and Office of the Comptroller of the Currency issued interagency model-risk guidance in 2011, commonly referred to as SR 11-7 and OCC 2011-12.
For Federal Reserve supervision, [SR 26-2](https://www.federalreserve.gov/supervisionreg/srletters/sr2602.htm) superseded SR 11-7 in April 2026; consult the relevant agency's current scope.
The guidance discusses effective challenge, ongoing monitoring, and outcomes analysis, including backtesting where appropriate.
Its direct applicability depends on the organization and jurisdiction; it is not automatically a binding rule for every supply-chain company.
Using its principles voluntarily does not certify compliance or eliminate the need for legal and supervisory interpretation.

The National Institute of Standards and Technology (NIST) AI Risk Management Framework 1.0 describes voluntary risk-management functions including measurement and management.
It is a general framework, not a prescribed forecast metric or a guarantee that a model is fit for a particular decision.
Whether an AI-specific framework applies to a given forecasting method depends on the organization's definitions and governance policy.

### Privacy, access, and retention

Planning data can contain customer, employee, supplier, or commercially sensitive information.
Minimize personal data, restrict access, and retain evidence according to applicable law and organizational policy.
Do not publish row-level trial results when aggregation or controlled access is required.
Privacy obligations depend on jurisdiction, purpose, and data subject; this teaching unit is not legal advice.

### Units and comparability

Normalize units and currencies before comparing results.
Record whether a weekly bucket starts on Monday or Sunday and how time zones and daylight-saving transitions are handled.
A product pack-size change can make an apparent demand shift a unit-conversion error.
A financial comparison across periods should state currency, price basis, and whether inflation or exchange-rate effects are included.

Version outcome definitions as well as model definitions.
If a return changes a sale after a month, distinguish the initial shipment, net sale, and mature demand measure.
Do not silently mix these definitions within a scorecard.

## 9. Industry practice and case evidence

### Public evidence and what it supports

The M5 forecasting competition used Walmart unit sales data at multiple aggregation levels and evaluated probabilistic and point forecasts under defined competition rules.
The competition provides a public example of real retail forecasting data, hierarchical structure, and comparative evaluation.
Its results do not establish that a particular competition model caused Walmart to achieve a specific operational saving.
The published M5 papers document the data and competition design; the simulation in section 6 is unrelated to their results.

The M4 competition compared forecasting methods across a large collection of time series and used specified accuracy measures and benchmarks.
It illustrates why evaluating multiple series and credible alternatives matters.
A competition ranking does not by itself validate a model for a particular replenishment decision, lead time, or business cost.
Differences in data availability, service requirements, and decision policies limit transportability.

The U.S. Federal Reserve's SR 11-7 guidance describes outcomes analysis as a comparison of model outputs with corresponding actual outcomes and discusses backtesting.
It is evidence of a formal model-risk practice in banking, not evidence that a specific supply-chain model is compliant or effective.
Adapting such governance ideas to supply-chain planning is an organizational design choice, not a claim that the guidance directly regulates all manufacturers and retailers.

### Interpreting operational reports

A supplier might report that forecast accuracy improved after a planning-system rollout.
Ask whether the pre- and post-periods cover the same products, horizons, target definitions, and demand regimes.
Ask whether returns, stockouts, and promotions were treated consistently.
A change in the set of products can improve an aggregate metric even when no individual product improves.

A distributor might report lower inventory following deployment.
Lower inventory can reflect a model, a change in supplier lead time, a deliberate service reduction, or a temporary demand shock.
Review service, stockouts, expedite costs, purchase prices, and capacity alongside inventory.
A credible impact estimate needs a counterfactual and a bridge from operational changes to financial consequences.

A retailer might report that planners override fewer forecasts after a model release.
Fewer overrides could indicate greater trust, lack of time, or changes in process requirements.
It is not by itself proof that forecasts or decisions improved.
Retain both override rates and outcome results, and investigate why overrides occurred.

### Evidence grading

Peer-reviewed competition papers and official guidance are useful public references for methodology.
They do not validate the fabricated example in this lesson.
Vendor case studies may describe implementation details but often lack an independently observed counterfactual.
Treat a company's own savings claim as a company claim unless the underlying design and audit are available.
Label an inference as an inference, identify alternative explanations, and do not promote a simulated proxy into an observed financial result.

### Evidence that is unavailable

This lesson makes no claim that a named enterprise deployed the section 6 challenger.
It does not report a documented business result for that synthetic dataset.
A real case dossier should identify the actual intervention, comparison group, time period, outcome definition, and independent evidence before making such a claim.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Random temporal split | Later records appear in training for earlier forecasts | Future regimes inform the fitted model | Inflated historical performance | Use rolling origins and enforce as-of cutoffs |
| Revised-history leakage | Old values match today's corrected table exactly | Later corrections replace the version known at origin | Optimistic feature quality | Retain revision timestamps and historical snapshots |
| Delayed-label leakage | Training includes targets not yet mature at an origin | Future outcomes influence fitting | Unrealistic retraining performance | Restrict training by label availability |
| Global preprocessing | One scaler or imputer was fit before splitting | Holdout distribution influences training | Biased comparison | Fit transformations inside each training window |
| Reused holdout | Many experiments cite the same final period | Selection adapts to holdout noise | Optimistic final result | Record reuse and reserve later untouched evidence |
| Straw-man benchmark | Challenger is compared only with a constant zero | Comparator is operationally irrelevant | False impression of value | Include a credible simple baseline and current process |
| Unequal keys | Challenger scores fewer difficult products | Missing predictions disappear from averages | Selective improvement | Report coverage and predeclared fallback on all eligible keys |
| Censored demand | Accuracy looks best during stockouts | Shipments are mistaken for demand | Underforecasting is rewarded | Use availability indicators and qualify the outcome |
| Horizon mixing | Aggregate score improves after short-horizon share rises | Different horizon mix changes the denominator | Misleading trend | Report horizon-specific results and fixed weights |
| Aggregate bias cancellation | Overall bias is near zero but segments have opposite errors | Positive and negative errors offset | Persistent local shortages or excess | Report segment bias and absolute error |
| Invalid inventory replay | Each week restarts with ideal inventory | State and lead times are ignored | Invented cost reduction | Use a stateful policy and realistic initial conditions |
| Double-counted benefits | Both avoided lost margin and a shortage penalty enter the sum | Same consequence is counted twice | Inflated value claim | Map each cost to a unique financial mechanism |
| Confounded rollout | Benefits follow a price change or promotion | Concurrent events affect outcomes | Incorrect causal attribution | Use a comparison design and document concurrent changes |
| Interference | Treated and control products share constrained capacity | One group's actions affect another's outcome | Biased trial comparison | Randomize at a suitable cluster and monitor spillovers |
| Immature labels | Recent outcomes show fewer returns | Outcomes have different observation windows | Artificial improvement | Enforce maturity windows and report missingness |
| Unsupported precision | Tiny intervals from thousands of correlated rows | Rows are treated as independent | Overconfidence | Use appropriate blocks or clusters and disclose small samples |
| Ignored overrides | Offline model wins while planners consistently reverse it | Evaluated process differs from deployment | Unreliable business conclusion | Compare the full operational champion and track overrides |
| Silent fallback | Missing challenger forecasts are simply omitted | Model failures vanish from the score | Operational risk hidden | Predefine fallback, report coverage, and score the combined policy |
| No response owner | Monitoring breach remains on a dashboard | No one has authority to act | Harm persists | Assign thresholds, escalation, and rollback authority |
| Unreproducible result | Only a slide with a final score remains | Data and selection history are lost | Validation cannot be challenged | Retain versioned data, code, predictions, and decisions |

A red flag is a reason to investigate, not automatic proof of misconduct.
The significance depends on the decision's materiality and the direction and magnitude of possible bias.
Record both the defect and its effect on the decision, including uncertainty where the effect cannot be quantified.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or challenge |
|---|---|---|---|
| Historical inputs are recoverable as of each origin | Prevents hindsight | Overwritten records or missing timestamps | Compare retained snapshots with source audit logs |
| Outcome approximates the target | Makes error meaningful | Stockouts censor sales or returns are incomplete | Compare availability, orders, backorders, and maturity |
| Benchmark is executable | Establishes a practical alternative | Prior observation arrives after the cutoff | Verify publication times and fallback behavior |
| Eligible keys are comparable | Supports paired differences | Candidate fails disproportionately on difficult cases | Report coverage and score a fallback policy |
| Period covers relevant regimes | Supports intended-use inference | Only calm weeks or one season are present | Slice by season, promotion, disruption, and lifecycle |
| Future resembles relevant history | Allows cautious transport | New supplier, demand regime, or product portfolio | Monitor drift and evaluate new regimes prospectively |
| Cost parameters reflect the decision | Makes economic proxy interpretable | Stale margins, nonlinear costs, or double counting | Validate with finance and run sensitivity scenarios |
| Replay policy is feasible | Connects forecast to action | Capacity, lead times, or ordering rules omitted | Review with operators and simulate constraints |
| Fixed historical outcomes are plausible under alternatives | Enables offline policy comparison | Actions change demand, supplier behavior, or substitution | Examine feedback pathways and run prospective trials |
| Uncertainty method reflects dependence | Avoids false precision | Overlapping horizons or shared shocks | Inspect correlations and use suitable blocks or clusters |
| Human behavior is represented | Matches evaluated and deployed processes | Overrides or adoption differ by group | Log pre- and post-override values and reasons |
| Trial groups are comparable | Supports causal interpretation | Selection, interference, or concurrent interventions | Randomize where feasible and review balance and spillovers |
| Guardrails can be observed promptly | Enables safe rollout | Delayed outcomes or missing service data | Test alert latency and escalation before launch |
| Evidence remains reproducible | Supports independent challenge | Data retention expires or code versions are lost | Recreate a run from retained artifacts |

Some assumptions are not testable from a single historical sample.
A pre-period trend check, for example, cannot prove future parallel trends.
Present the assumption, plausible failure direction, and operational consequence rather than declaring it verified.
A validation decision may be conditional when residual uncertainty is material.

## 12. Exercises

### Exercise 1 — Define the evaluation contract

A planner orders a component every Monday for a four-week supplier lead time.
The system produces a four-week cumulative demand forecast, but receipts can be delayed.
The company currently uses a planner-adjusted moving average.

**Deliverable:** a one-page decision and evaluation contract.
**Acceptance criteria:** identify the decision unit, cutoff, horizon, target, champion, action, feasible constraints, owner, fallback, and one reason forecast accuracy alone is insufficient.
**Time:** 15 minutes.

### Exercise 2 — Build a temporal design

You have 36 months of weekly history and a four-week target that matures one week after the target period ends.
A new forecast is issued every week.
The team wants to tune a model and then report an independent final result.

**Deliverable:** a timeline with development, selection, final holdout, and two representative rolling origins.
**Acceptance criteria:** each training label is mature at its origin; settings are locked before final evaluation; overlapping target windows and limited seasonal coverage are acknowledged.
**Time:** 20 minutes.

### Exercise 3 — Find leakage

A Monday forecast uses a table containing event date, last-modified date, final returns, and a rolling four-week mean that includes the forecast week.
A promotion plan is approved the previous Friday but its actual lift is calculated after the week ends.

**Deliverable:** an availability table and corrected feature rules.
**Acceptance criteria:** distinguish event time from availability, exclude the future-inclusive mean and actual lift, and permit the approved plan only if it was accessible at the cutoff.
**Time:** 15 minutes.

### Exercise 4 — Compare methods fairly

The challenger produces forecasts for 90 of 100 eligible item-weeks.
The current process covers all 100.
The 10 missing challenger forecasts are concentrated among new items.
The team reports the challenger's score only on the 90 available forecasts.

**Deliverable:** a revised comparison and coverage table design.
**Acceptance criteria:** include the common-key comparison, coverage by lifecycle segment, an executable fallback for missing predictions, and an all-eligible policy score where outcomes are available.
**Time:** 15 minutes.

### Exercise 5 — Calculate a loss comparison

Actual weekly demand is 80, 110, and 100 units.
Forecast A is 75, 120, and 95 units; forecast B is 85, 100, and 100 units.
An underforecast unit costs USD 6 in a decision-loss proxy; an overforecast unit costs USD 2.

**Deliverable:** signed errors, MAE, total proxy loss, and the better method under that proxy.
**Acceptance criteria:** use actual minus forecast, show arithmetic, and distinguish the proxy from realized savings.
**Time:** 15 minutes.

### Exercise 6 — Critique a benefits claim

A company says that its new model reduced inventory by USD 2 million because average inventory fell from USD 12 million to USD 10 million after rollout.
During the same period, supplier lead time fell and the company reduced its service target.
No comparison group was retained.

**Deliverable:** a short challenge memo and an improved evidence design.
**Acceptance criteria:** identify at least two alternative explanations, distinguish inventory level from realized savings, include service and cost guardrails, and propose a credible comparator.
**Time:** 20 minutes.

### Exercise 7 — Design an outcome trial

A proposed allocation model will share a warehouse capacity constraint across products.
Managers want a six-week rollout and want to stop if service deteriorates.

**Deliverable:** a trial protocol outline.
**Acceptance criteria:** specify eligibility, assignment unit, primary outcome, interference risk, duration rationale, guardrails, stopping authority, analysis population, and limitations.
**Time:** 25 minutes.

### Exercise 8 — Assemble a validation dossier

A model has lower selection-period error but is missing historical feature versions for one important supplier.
The business requests immediate unrestricted approval.

**Deliverable:** an approval recommendation and evidence checklist.
**Acceptance criteria:** distinguish observed evidence from assumption, identify the missing point-in-time record and affected scope, propose a bounded decision or rejection with fallback, and name revalidation triggers and owners.
**Time:** 20 minutes.

## 13. Self-check answers

### Exercise 1

A strong contract names a component-location-order date as the decision unit and the Monday ordering cutoff in a stated time zone.
The target is four-week cumulative demand in units; the action is a feasible purchase order, not the forecast itself.
The champion is the planner-adjusted moving-average process, including overrides.
Supplier minimum quantities, receipt delays, inventory, open orders, and capacity constrain the action.
The planner owns the decision and the current approved process is a plausible fallback.
Accuracy alone does not measure late receipts, inventory state, service, or the cost of the resulting order.

### Exercise 2

One valid design uses an early development period, a later selection period, and a final later holdout with dates fixed before tuning.
At each weekly origin, only four-week labels whose target period has ended and whose additional one-week maturity delay has passed may enter training.
The exact last eligible label depends on the cutoff and bucket boundaries.
Rolling origin predictions can overlap in their target periods, so their errors are dependent.
Thirty-six months provides at most three annual cycles, and less for some training and evaluation windows; conclusions about unusual seasonal regimes are limited.

### Exercise 3

The event date alone does not show whether a record was available on Monday.
Use the version whose availability timestamp is no later than the cutoff.
Final returns known later are not a valid Monday input unless a separately available earlier estimate is used.
Recompute the rolling mean from past, available observations only.
An approved promotion plan may be used if the decision system could access that version by Monday; realized lift is a later outcome and must not be used.

### Exercise 4

Report 90% overall challenger coverage and the coverage of new versus established items.
Compare challenger and champion on the same 90 item-weeks for a conditional performance view.
Separately evaluate the challenger-plus-fallback policy on all 100 eligible item-weeks, using a fallback available at each origin.
Show missing outcomes and any remaining unserved decisions.
The 90-key score alone cannot support an all-population conclusion.

### Exercise 5

Forecast A signed errors are 5, -10, and 5 units.
Its absolute errors sum to 20 units and MAE is 20 / 3 = 6.67 units per week.
Its positive errors total 10 units and its negative-error magnitude totals 10 units.
Its proxy loss is 10 × USD 6 + 10 × USD 2 = USD 80.

Forecast B signed errors are -5, 10, and 0 units.
Its absolute errors sum to 15 units and MAE is 15 / 3 = 5.00 units per week.
Its proxy loss is 10 × USD 6 + 5 × USD 2 = USD 70.
Forecast B has the lower proxy loss by USD 10 over the three decisions.
The calculation does not show USD 10 in realized savings because the cost and action mechanism are hypothetical.

### Exercise 6

The lower inventory level is a stock measure, not automatically a USD 2 million recurring saving.
Shorter lead times and the reduced service target are plausible alternative explanations.
A proper bridge would consider carrying cost, financing, obsolescence, stockouts, lost margin, and implementation costs without double counting.
A randomized or defensible matched comparison across suitable locations could improve attribution if spillovers and concurrent changes are addressed.
Report service, expedites, and customer harms as guardrails.

### Exercise 7 rubric

A strong protocol randomizes at a cluster level that accounts for shared warehouse capacity, or explains why interference remains unavoidable.
It defines the eligible population and the effect of assignment to the model-enabled process.
It states a primary outcome, measurement window, power or precision limitations, and how six weeks relates to operational cycles.
It names service and safety guardrails, a person authorized to stop, and the fallback process.
It records overrides and analyzes assigned groups while explaining any additional as-treated analysis.

### Exercise 8 rubric

A strong recommendation does not call the selection period an independent final holdout.
It identifies the supplier and decisions affected by missing historical feature versions.
It either restricts use to a supportable scope with a documented fallback or rejects approval pending adequate evidence.
It identifies an owner for reconstructing data, an independent reviewer, and triggers for reconsideration.
The dossier contains versioned data, code, selection history, predictions, outcomes, assumptions, and decision rationale.

## 14. Interview questions

1. **How would you evaluate a weekly forecast for a four-week decision?**
   A strong answer defines origin, horizon, label maturity, rolling evaluation, and the action.
   A weak answer proposes a random split without addressing chronology.
2. **What is the difference between event time and availability time?**
   A strong answer gives a delayed or revised record example.
   A weak answer assumes an old event date proves the value was known.
3. **Why compare against both a benchmark and a champion?**
   A strong answer distinguishes a simple minimum standard from the current operational process.
   A weak answer chooses a deliberately poor baseline.
4. **What changes when planners override the forecast?**
   A strong answer treats the human-plus-model workflow as the process and retains both versions.
   A weak answer assumes an offline model score is the deployed outcome.
5. **How can a holdout stop being a holdout?**
   A strong answer explains adaptive tuning and the need for later untouched evidence.
   A weak answer believes a file name alone preserves independence.
6. **Why can low bias coexist with harmful performance?**
   A strong answer explains cancellation and examines absolute and segment errors.
   A weak answer equates zero average error with perfect decisions.
7. **What is wrong with scoring demand forecasts against shipments during stockouts?**
   A strong answer identifies censoring and seeks availability or lost-demand evidence.
   A weak answer assumes all unobserved demand is zero.
8. **When does a backtest support a financial claim?**
   A strong answer separates simulated loss from realized benefit and states policy and cost assumptions.
   A weak answer annualizes an offline accuracy improvement without a decision bridge.
9. **How would you handle missing challenger predictions?**
   A strong answer reports coverage, common keys, and a predeclared fallback policy.
   A weak answer removes missing rows and presents the result as population-wide.
10. **What makes a trial comparison credible?**
    A strong answer addresses assignment, interference, outcome maturity, guardrails, and the estimand.
    A weak answer attributes any before-after change to the model.
11. **How do you quantify uncertainty with overlapping horizons?**
    A strong answer considers temporal and cluster dependence and the number of independent units.
    A weak answer treats every overlapping row as independent.
12. **What would trigger revalidation?**
    A strong answer names material changes to data, model, action policy, population, or operating regime and assigns response authority.
    A weak answer relies exclusively on a calendar date.

## 15. Key terms

- **Action policy:** rule converting information and forecasts into feasible decisions; see [mechanics](#step-10--translate-forecasts-into-decisions).
- **As-of snapshot:** source values available at a particular cutoff; see [data inventory](#step-3--inventory-data-and-information-availability).
- **Availability time:** earliest time information could be used; see [three clocks](#three-clocks).
- **Backtest:** historical replay under an information and action protocol; see [core model](#3-the-core-mental-model).
- **Benchmark:** simple, reproducible comparator; see [comparators](#step-6--specify-comparators).
- **Bias:** average signed error under a declared convention; see [scorecard](#step-8--produce-a-paired-scorecard).
- **Censoring:** observation limited by an underlying constraint; see [target definition](#step-2--define-the-target-and-outcome).
- **Challenger:** candidate alternative to the current process; see [comparators](#step-6--specify-comparators).
- **Champion:** current authorized process; see [comparators](#step-6--specify-comparators).
- **Counterfactual:** outcome under an unobserved alternative; see [prospective evidence](#step-11--design-prospective-outcome-evidence).
- **Coverage:** share of eligible decisions with usable predictions; see [scorecard](#step-8--produce-a-paired-scorecard).
- **Decision loss:** penalty assigned to an action and outcome; see [worked example](#decision-loss-proxy).
- **Embargo:** excluded interval used to prevent a specified information overlap; see [chronology](#step-4--design-chronology-before-fitting).
- **Event time:** time an event occurred; see [three clocks](#three-clocks).
- **Expanding window:** training window that grows over time; see [chronology](#step-4--design-chronology-before-fitting).
- **Final holdout:** later data withheld until the process is locked; see [selection](#step-7--freeze-selection-and-final-evaluation).
- **Forecast origin:** time a prediction is issued; see [contract](#step-1--contract-the-decision).
- **Guardrail:** metric or rule limiting unacceptable harm; see [prospective evidence](#step-11--design-prospective-outcome-evidence).
- **Horizon:** elapsed time from origin to predicted period; see [contract](#step-1--contract-the-decision).
- **Interference:** treatment of one unit affects another unit's outcome; see [prospective evidence](#step-11--design-prospective-outcome-evidence).
- **Label maturity:** point when an outcome is sufficiently complete to use; see [target definition](#step-2--define-the-target-and-outcome).
- **Leakage:** use of unavailable information in the evaluation pipeline; see [leakage controls](#step-5--prevent-leakage-throughout-the-pipeline).
- **MAE:** mean absolute error in target units; see [scorecard](#step-8--produce-a-paired-scorecard).
- **Paired difference:** loss difference between methods on the same decision key; see [uncertainty](#step-9--characterize-uncertainty-and-dependence).
- **Rolling origin:** repeated chronological evaluation at successive decision times; see [chronology](#step-4--design-chronology-before-fitting).
- **Selection period:** earlier data used to choose a candidate; see [selection](#step-7--freeze-selection-and-final-evaluation).
- **Sliding window:** bounded recent training history; see [chronology](#step-4--design-chronology-before-fitting).
- **WAPE:** absolute error sum divided by absolute actual sum; see [scorecard](#step-8--produce-a-paired-scorecard).

## 16. Sources

1. Board of Governors of the Federal Reserve System, [SR 11-7, Guidance on Model Risk Management](https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm), 2011. Official supervisory guidance; applicability depends on institution and jurisdiction.
2. Office of the Comptroller of the Currency, [OCC 2011-12, Supervisory Guidance on Model Risk Management](https://www.occ.gov/news-issuances/bulletins/2011/bulletin-2011-12.html), 2011. Companion interagency guidance.
3. National Institute of Standards and Technology, [Artificial Intelligence Risk Management Framework (AI RMF 1.0)](https://www.nist.gov/itl/ai-risk-management-framework), 2023. Voluntary general risk-management framework.
4. Hyndman, Rob J., and George Athanasopoulos, [Forecasting: Principles and Practice, third edition, time series cross-validation](https://otexts.com/fpp3/tscv.html). Open forecasting text describing rolling-origin evaluation.
5. Makridakis, Spyros, Evangelos Spiliotis, and Vassilios Assimakopoulos, [The M4 Competition: Results, findings, conclusion and way forward](https://doi.org/10.1016/j.ijforecast.2018.06.001), *International Journal of Forecasting*, 2018. Public comparison of forecasting methods.
6. Makridakis, Spyros, Evangelos Spiliotis, and Vassilios Assimakopoulos, [The M5 accuracy competition: Results, findings, and conclusions](https://doi.org/10.1016/j.ijforecast.2021.11.013), *International Journal of Forecasting*, 2022. Public retail forecasting competition; not a causal deployment study.
7. Hernán, Miguel A., and James M. Robins, [Causal Inference: What If](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/), 2020. Open text on causal questions, comparisons, and assumptions.
8. Bergmeir, Christoph, Rob J. Hyndman, and Bonsoo Koo, [A note on the validity of cross-validation for evaluating autoregressive time series prediction](https://doi.org/10.1016/j.csda.2017.11.003), *Computational Statistics & Data Analysis*, 2018. Discusses conditions for cross-validation in time-series settings; do not treat it as a blanket justification for shuffled operational backtests.

Sources support the methods and public examples attributed to them.
The arithmetic, dataset, cost assumptions, and governance conclusion in section 6 are invented for instruction.
Regulatory and financial discussion is educational and does not replace professional advice.

## 17. Links

- [Module 26 topic index](_index.md): scope and sequence of the model-risk topics.
- [Model inventory and tiering](01_model-inventory-tiering.md): connect evidence burden to use and materiality.
- [Conceptual soundness](02_conceptual-soundness.md): challenge the target, method, and decision assumptions.
- [Module 26 survey](../../../course/26_quality_evaluation_model_risk.md): broader model-risk overview.
- [Rolling-origin backtesting](../../02_methods/07_time_series_forecasting/08_rolling-origin-backtesting.md): forecasting-specific temporal evaluation.
- [Baseline and counterfactual](../25_product_economics_leadership/02_baseline-counterfactual.md): frame a value comparison and its causal limitations.
- [Simulation and optimization production labs](../../../course/30_simulation_optimization_production_labs.md): practice decision and operational simulation concepts.
