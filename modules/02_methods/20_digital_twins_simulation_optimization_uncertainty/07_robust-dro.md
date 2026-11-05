# Robust and Distributionally Robust Optimization

> Module 20 — Digital Twins, Simulation, and Optimization under Uncertainty  
> Topic 7 of 10  
> Estimated study time: 4 hours  
> Prerequisites: input modelling, validation, probability, and basic optimization  
> Version 2026.5

## 1. Why this topic exists

A capacity planner must reserve warehouse shifts before the demand signal is complete.
The financial commitment is real even though the demand forecast will change.
A conventional stochastic plan assigns probabilities to demand outcomes and chooses the lowest expected cost.
That plan is only as credible as the probability distribution used to calculate the expectation.
A distribution estimated from a quiet period may understate demand spikes, supplier failures, or simultaneous disruptions.

Robust optimization asks a different question: which decision performs acceptably against every realization in a specified uncertainty set?
Distributionally robust optimization (DRO) asks which decision has the best worst-case expected performance across a specified set of probability distributions.
Neither approach predicts the future.
Each makes uncertainty assumptions explicit and translates them into a decision.

The business question is not whether caution is good.
It is how much capacity, inventory, or flexibility to buy, against which failure modes, and with what measurable tradeoff in ordinary periods.
A model that protects against impossible combinations can waste money.
A model that excludes plausible disruptions can create a false sense of security.

The planning owner needs an answer before the booking deadline.
Finance needs the premium over the current plan.
Operations needs to know whether the proposed action can actually be executed.
Risk owners need to know what the model protects, what it does not protect, and which events trigger replanning.

This topic gives you a way to make those conversations concrete.
The running example is a stylized weekly warehouse-capacity decision in US dollars (USD).
It is not a claim about a particular company or a substitute for local costing.

### The decision in one sentence

Choose capacity before observing demand, knowing that the historical probability of high demand may be too low.
The action must be booked once for the coming week.
The eventual demand is observed later.
The cost of unused capacity is borne when demand is low.
The cost of unserved demand is borne when demand exceeds booked capacity.

### Why distributional misspecification matters

A sample can be accurate for the period in which it was collected and still be poor for the period being planned.
A promotion, competitor exit, labor disruption, supplier change, or new customer can alter the distribution.
Even without a regime change, a finite sample estimates rare outcomes poorly.
If high-demand observations are censored by stockouts, the observed sales distribution can miss the very events that matter most.

A robust model does not repair bad data automatically.
A DRO model does not identify the correct future distribution automatically.
Both require a defensible set, a decision timeline, a cost model, and independent validation.

## 2. Learning outcomes

After completing this topic, you should be able to:

1. Distinguish uncertain realizations from uncertain probability distributions, and formulate a decision for each.
2. Identify first-stage actions, observations, and any feasible recourse on a decision timeline.
3. Construct and interpret a box, budgeted, scenario, and probability-ambiguity set without confusing their units.
4. Calculate expected, worst-case, and worst-case expected cost for a small capacity problem.
5. Explain how a distance radius, support assumption, and dependence assumption change the implied protection.
6. Compare the cost of protection with out-of-sample service and downside performance.
7. Design a validation and governance plan that separates model selection from final evaluation.
8. Recommend a business action with an explicit risk appetite, limits, and trigger for revision.

Exercises 1–8 respectively check these outcomes.
The numerical answers use the stated example, not an unstated solver convention.
A business recommendation can differ from the minimum-cost recommendation when the decision maker states a different risk constraint or valuation.

## 3. The core mental model

```text
                 DATA AND BUSINESS EVIDENCE
         transactions, forecasts, disruptions, constraints
                           |
            diagnose missingness and regime change
                           |
          +----------------+----------------+
          |                                 |
     realizations u                  distributions P
     in uncertainty set U            in ambiguity set Pset
          |                                 |
          v                                 v
 robust: min_x max_u cost(x,u)    DRO: min_x max_P E_P[cost(x,u)]
          |                                 |
          +---------------+-----------------+
                          |
           compare candidate decisions and tradeoffs
                          |
       independent historical and stress evaluation
                          |
          approval, commitment, monitoring, revision
```

A realization is a concrete future, such as 120 pallets of demand next week.
A probability distribution assigns weights to possible futures.
An uncertainty set contains realizations.
An ambiguity set contains distributions.
A robust model defends against the worst permitted realization.
A DRO model defends against the worst permitted distribution, evaluated through an expectation or another specified risk functional.

Both outer minimizations choose a decision before the adversarial comparison in the expressions above.
The adversary is a mathematical device, not a person with the power to choose demand.
It identifies a challenging realization or distribution within the set you have allowed.
If the set is wrong, the adversary tests the wrong possibilities.

The nominal stochastic model is a useful benchmark.
It uses one estimated distribution and minimizes its expected cost.
DRO generalizes that comparison to a family of distributions.
A singleton ambiguity set containing only the nominal distribution reduces the DRO objective to the nominal expected-cost objective.
If an ambiguity set contains every distribution on a finite support, its worst expected cost equals the maximum cost over that support.
That equivalence is a useful check, not a statement that all robust and DRO models are interchangeable.

### The timeline matters

For the example, the planner books capacity on Friday.
Demand for the following week is observed during the week.
No late capacity adjustment is assumed in the base model.
Thus capacity is a first-stage decision and shortage is an outcome.

In another business, an agency contract may offer overtime after early orders arrive.
Then the feasible overtime action is recourse, not a free correction to the first-stage decision.
Its availability, lead time, cap, and cost must be specified.
A plan that lets capacity respond to demand before demand is observable has used forbidden information.

### Different kinds of protection

A robust objective may sacrifice average performance to protect the worst realization in its set.
A DRO objective may sacrifice nominal expected performance to protect against changes in distributional weights or shapes within its ambiguity set.
A robust constraint requires a constraint to hold for all allowed realizations.
A distributionally robust chance constraint requires a probability statement to hold for all allowed distributions.
These are distinct choices; an objective alone does not guarantee a service-level constraint.

### Model is not governance

Optimization returns a recommendation conditional on inputs.
Approval establishes whether the organization accepts the modeled tradeoff.
Execution creates commitments and records exceptions.
Monitoring checks whether the assumptions remain credible.
A digital twin or simulation can test the candidate policy, but only if its behavior and inputs are credible for the decision.

## 4. Definitions and vocabulary

| Term | Working definition | Common misuse | Unit or interpretation |
|---|---|---|---|
| Decision variable | Quantity selected by the planner | Treating an observed outcome as a choice | Pallets of weekly capacity in the example |
| Uncertain parameter | Quantity not known at commitment | Calling a controllable policy an uncertainty | Pallets of weekly demand |
| Realization | One possible value of uncertainty | Calling a probability distribution a realization | One demand value |
| Scenario | A specified possible realization or joint path | Assuming each scenario must be equally likely | Demand, disruption, and time path as defined |
| Support | Values a random variable can take under a model | Confusing observed sample range with known physical limits | Pallets per week |
| Nominal distribution | Single reference distribution used for comparison | Treating the estimate as certain truth | Probabilities summing to one |
| Empirical distribution | Equal mass on each retained observation, unless weighted | Assuming raw observations are independent and unbiased | Probability per retained observation |
| Uncertainty set | Permitted realizations used by a robust model | Describing it as a probability distribution | Same units as the uncertain parameters |
| Ambiguity set | Permitted probability distributions used by DRO | Treating a realization as a distribution | Set of distributions |
| Robust objective | Optimize the worst allowed realized cost | Equating it to a service guarantee | USD per planning week |
| Robust constraint | Constraint enforced for every allowed realization | Assuming a worst-case objective enforces every constraint | Units of the constrained quantity |
| Recourse | Feasible action after specified information arrives | Allowing it before information arrives | Overtime pallets, USD, or other action units |
| Adjustable robust optimization | Robust planning with recourse rules dependent on observed information | Letting rules depend on future unobserved information | Policy, not a single scalar |
| Nonanticipativity | Decisions use only information available when taken | Assuming all scenario paths can be distinguished immediately | Information constraint |
| Distributional misspecification | Reference distribution differs from the relevant future distribution | Assuming it means only a wrong mean | May affect tails, dependence, support, or dynamics |
| Ambiguity radius | Size of a distance-based set around a reference distribution | Comparing radii from incompatible metrics as if equivalent | Metric-specific; total variation is dimensionless |
| Total variation distance | Maximum probability difference over events; half the L1 distance on finite support | Using the full L1 distance while calling it half | Probability mass, from zero to one |
| Wasserstein distance | Minimum transportation cost between probability measures for a specified ground metric | Omitting the ground metric and its units | Ground-distance units for order one |
| Moment set | Distributions satisfying bounds on moments such as means | Assuming moments determine the entire distribution | Units depend on the moment |
| Divergence ball | Distributions constrained by a chosen statistical divergence from a reference | Assuming all divergences can add mass to unseen support | Depends on divergence convention |
| Budgeted uncertainty | A set limiting the joint amount of adverse deviation | Calling its budget a monetary amount by default | Often dimensionless |
| Correlation | A measure of linear dependence | Treating zero correlation as independence | Dimensionless |
| Tail | High- or low-value portion of a distribution | Assuming a finite sample identifies an extreme tail | Quantile or loss units |
| Expected cost | Probability-weighted average cost | Calling the expected outcome the most likely outcome | USD per planning week |
| Worst-case cost | Largest cost over an uncertainty set | Presenting it as a probable outcome without probabilities | USD per planning week |
| Worst-case expected cost | Largest expected cost over an ambiguity set | Calling it the largest single-scenario cost in all cases | USD per planning week |
| Robustness premium | Difference in cost relative to a stated benchmark | Assuming it is always positive in every realized period | USD per week, with evaluation distribution named |
| Out-of-sample evaluation | Evaluation on data not used to fit or select the policy | Reusing calibration data as an independent test | Evaluation period and metric |
| Stress test | Evaluation under a specified adverse condition | Treating an unweighted stress test as a probability forecast | Scenario-specific result |
| Regret | Cost of a decision minus a stated comparator's cost | Comparing to a clairvoyant comparator without saying so | USD per period |
| Service level | A defined measure of service performance | Treating fill rate and cycle service as identical | Fraction or percent with denominator specified |
| Distribution shift | Change between historical and deployment distributions | Assuming every shift can be seen in a short holdout | Difference in inputs, outcomes, or both |
| Calibration | Choosing model parameters or a radius using evidence | Calling a radius validated solely because it was optimized on the test set | Metric-specific |
| Decision horizon | Period over which the action and outcomes are evaluated | Combining weekly capacity with annual loss without conversion | Week in the example |

## 5. The mechanics, step by step

### Step 1 — Frame the decision and its owner

**Input:** a business problem, a commitment deadline, and a responsible decision maker.

**Action:** write a decision statement with a measurable action, horizon, and objective.

**Output:** a decision charter that can be challenged before model building.

For example: select next week's booked capacity, in pallets per week, by Friday noon.
Minimize total weekly capacity and shortage cost, subject to documented operational limits.
The planner owns the recommendation; the warehouse manager confirms feasibility; finance approves the spend.

Separate a forecast from a plan.
The forecast is an estimate of future demand.
The plan is the chosen capacity.
A service target is an aspirational or constrained performance measure, depending on how it is used.
A commitment is an action authorized with a counterparty.
The model must not silently turn a forecast into a contractual commitment.

Branch when the organization has a hard service requirement.
Represent the requirement explicitly if it is feasible and properly defined.
Do not assume that a cost penalty in the objective automatically enforces the requirement.
If the requirement is legally binding, obtain the relevant legal interpretation rather than inferring it from a model.

### Step 2 — Draw the information timeline

**Input:** decision dates, demand visibility, supplier lead times, and adjustment options.

**Action:** mark each action before or after the observation that enables it.

**Output:** first-stage variables, admissible recourse, and information constraints.

A capacity booking made before demand arrives is common across scenarios.
A later overtime decision can differ across scenarios only after the observations distinguish those scenarios.
If early and late demand look identical until Wednesday, a Monday recourse decision must be the same across those paths.
This is nonanticipativity.

Do not give a plan unrestricted access to a future scenario label.
A simulation that supplies the final weekly total to an early policy creates a clairvoyant advantage.
If decisions recur, define the state and observation schedule rather than treating a week as an indivisible draw.

### Step 3 — Inventory uncertainty and the mechanisms behind it

**Input:** demand history, forecast errors, lead times, disruptions, order policies, and stakeholder knowledge.

**Action:** identify uncertain quantities and how they jointly affect cost and feasibility.

**Output:** a list of uncertain parameters, dependencies, and evidence gaps.

Demand may be uncertain in volume and timing.
Available capacity may be uncertain because of attendance and equipment reliability.
A supplier disruption can increase inbound delay and reduce available material at the same time.
A promotion can raise volume while changing the product mix and handling time.

Distinguish aleatory variability from uncertainty about the model used to describe that variability.
The distinction is useful for communication even when it is not clean in practice.
Repeat observations can reduce some estimation uncertainty.
They do not remove a future structural change or guarantee that the tail has been observed.

Ask whether the historical data are censored.
Sales are not identical to demand when stockouts prevent purchases.
Recorded service times may omit abandoned jobs.
An estimate built from those records may be biased before any optimization begins.

### Step 4 — Build a nominal benchmark

**Input:** a specified reference distribution and a defensible cost function.

**Action:** calculate the expected cost of each feasible decision or solve the corresponding stochastic model.

**Output:** a nominal decision, expected cost, and scenario-by-scenario outcomes.

For capacity `x` and demand `d`, the teaching example uses

```text
cost(x, d) = 2 x + 10 max(d - x, 0) USD per week.
```

Here `x` and `d` are measured in pallets per week.
The booking cost is USD 2 per pallet of weekly capacity.
The shortage penalty is USD 10 per unserved pallet.
No salvage, backlog, revenue, overtime, or fixed charge is included.

This cost function values a shortage, but it does not specify which customers are shorted.
If allocation fairness or contractual priority matters, add the necessary constraints and data rather than assuming the aggregate model answers that question.
The penalty should include only costs relevant to the decision and should avoid double counting.
A contribution margin, expedited freight, and contractual penalty may overlap if defined carelessly.

### Step 5 — Choose an uncertainty set for robust optimization

**Input:** physical bounds, historical evidence, plausible business stressors, and a risk mandate.

**Action:** choose which joint realizations count as possible for the decision.

**Output:** an explicit set `U`, including units and inclusions or exclusions.

A finite scenario set lists specific allowed outcomes.
A simple example is demand in `{80, 100, 120}` pallets per week.
The robust cost of a decision is its largest cost across those three outcomes.
The model makes no claim about the probability of any of them.

A box set bounds each uncertain quantity separately.
For two demands, one might write `80 <= d1 <= 120` and `40 <= d2 <= 70` pallets per week.
The box includes the simultaneous upper bounds.
That inclusion may be appropriate when simultaneous peaks are physically plausible.
It may be wasteful when those peaks cannot occur together because of a shared constraint.

A budgeted set limits the sum of normalized adverse deviations.
For baseline `d_i^0`, deviation `Delta_i`, and `z_i` between zero and one, one form is

```text
d_i = d_i^0 + Delta_i z_i,  sum_i z_i <= Gamma.
```

`Gamma` is dimensionless in this formulation.
When `Gamma` is zero, no positive deviation is permitted.
When `Gamma` equals the number of uncertain components, every component can reach its specified upper deviation together.
Intermediate values limit joint deviation; fractional values can permit partial deviations.
This construction is not a probability statement by itself.

An ellipsoidal set can encode joint directions and scaling.
It may be useful when a covariance-like geometry is defensible.
It can still be wrong about tails or structural regimes.
Check whether the selected optimization method can solve the resulting problem at the required scale.

A data-driven set can be calibrated from errors, but the calibration procedure must match the future deployment setting.
A collection of independently estimated one-dimensional intervals does not automatically describe joint coverage.
Similarly, an interval that covered old forecasts does not prove coverage after a distribution shift.

Branch on physical plausibility.
If a combination violates a known conservation law or shared resource limit, remove it using an explicit constraint.
If it is merely rare but possible, decide with the risk owner whether exclusion is acceptable.
Record the rationale either way.

### Step 6 — Choose an ambiguity set for DRO

**Input:** a reference distribution, plausible deviations from it, and a decision about allowed support.

**Action:** define a family of candidate probability distributions.

**Output:** an ambiguity set with a metric, radius, support, and any other restrictions.

For the example's finite support, write

```text
Pset(rho) = {p >= 0 : sum_i p_i = 1,
             0.5 * sum_i |p_i - p0_i| <= rho}.
```

`p0` is the nominal probability vector.
The quantity on the left is total variation distance for distributions on the same finite support.
The radius `rho` is probability mass, not a demand quantity and not USD.
A radius of 0.05 permits moving at most five percentage points of probability mass from one or more outcomes to others.
It does not mean every individual probability changes by exactly five percentage points.

The half factor matters.
Moving 0.05 from one outcome to another changes two vector entries by 0.05.
The L1 difference is therefore 0.10, while the total variation distance is 0.05.
Mixing these conventions doubles or halves the intended radius.

A moment ambiguity set might constrain the mean and variance to ranges.
Many different distributions can share those moments and have different tail behavior.
Always specify support if physical limits are known and essential to the result.
In an unbounded support, a moment condition may lead to very conservative or even ill-posed formulations depending on the loss and restrictions.

A Wasserstein ambiguity set limits the cost of transporting probability mass away from a reference distribution.
The ground metric says how costly it is to move from one outcome to another.
For multivariate demand, scale and units matter: one pallet of one product need not be comparable to one hour of delay.
A radius meaningful for an unscaled vector may change completely after a unit conversion.
Some Wasserstein formulations permit mass at values not seen in the sample, subject to the support and distance definition.

A divergence-based set can be computationally convenient on a discrete support.
Its behavior depends on the chosen divergence and direction.
For example, a forward Kullback–Leibler divergence `D(P || P0)` is infinite if `P` assigns positive mass to an outcome where `P0` assigns zero mass.
Thus a finite-radius set of that form cannot add mass to an outcome excluded by the reference distribution.
Do not generalize that property to every divergence or every direction.

Branch on the actual misspecification concern.
If the concern is incorrect weights on known scenarios, a finite-support probability ball may be useful.
If the concern includes unseen magnitudes, a fixed-support reweighting set cannot address it.
Add justified support or use another method and report the remaining limitations.

### Step 7 — Select and challenge the radius

**Input:** historical estimation uncertainty, deployment horizon, stress scenarios, and decision-maker risk appetite.

**Action:** evaluate a range of defensible radii rather than choosing one because its result looks comfortable.

**Output:** a sensitivity curve and an approved calibration rationale.

One route uses a statistical confidence construction under its stated sampling assumptions.
Another uses time-respecting validation across historical forecast origins.
A third chooses a policy based on a budget for nominal cost and a minimum acceptable stress performance.
These approaches answer different questions.
A manager-selected risk appetite is not a statistical coverage guarantee.
A radius tuned for cost on old periods is not automatically a confidence region.

A finite sample does not reliably reveal an unprecedented disruption.
A large radius on a misspecified support still cannot protect against excluded outcomes.
A small radius may be appropriate for routine weekly variation but inadequate for a known upcoming event.
Show results at several radii, including zero and one or more larger plausible values.
Record any threshold at which the recommended action changes.

### Step 8 — Formulate the optimization

**Input:** feasible decisions `X`, cost `c(x,u)`, and chosen set or reference distribution.

**Action:** write the decision rule and its evaluation measure explicitly.

**Output:** a mathematical problem that matches the decision timeline.

The nominal stochastic objective is

```text
min over x in X of E_(P0)[c(x, u)].
```

The static robust objective is

```text
min over x in X of max over u in U of c(x, u).
```

The DRO objective is

```text
min over x in X of sup over P in Pset of E_P[c(x, u)].
```

A maximum is attained in the finite example.
A supremum is written in the general expression because an arbitrary infinite set need not attain its bound.
The three objectives answer different questions; none is universally correct.

If the chosen action must satisfy capacity, budget, safety, or customer constraints, include them in `X` or formulate their uncertain versions explicitly.
For a robust constraint, require it for every `u` in `U`.
For an expected constraint, specify the probability distribution.
For a distributionally robust chance constraint, specify the allowed violation probability and require it for every distribution in the ambiguity set.
These choices can produce materially different plans.

### Step 9 — Solve the inner problem and the outer decision

**Input:** a finite decision set, scenario costs, and an ambiguity set.

**Action:** find the worst permitted cost or distribution for each decision, then choose the best decision.

**Output:** a candidate plan plus evidence of its adverse scenario or distribution.

For a finite total-variation ball, shifting probability from low-loss outcomes toward high-loss outcomes increases expected loss until the mass budget or relevant probability limits bind.
For a larger model, formulate the inner adversarial problem as a linear program when the objective and constraints permit it.
Do not assume a greedy routine remains valid after adding coupling restrictions, minimum probabilities, or other ambiguity constraints.

The outer problem can be enumerated when the decision set is small.
For continuous decisions, integer investments, or network constraints, use a suitable optimization solver and inspect solver status and optimality gaps.
A feasible solution with an unknown gap is not an established optimum.
An infeasible model needs diagnosis; do not silently relax business constraints.

Report ties.
When two choices have the same objective, secondary criteria such as lower spend, implementation flexibility, or service may distinguish them.
Those criteria must be stated rather than smuggled into the mathematical result.

### Step 10 — Quantify the tradeoff

**Input:** nominal, robust, and DRO candidate decisions.

**Action:** evaluate each under the same nominal distribution and the same stress scenarios.

**Output:** a comparable table of cost, service, and sensitivity.

Report the nominal expected-cost premium as

```text
nominal expected cost of protected policy
minus nominal expected cost of nominal policy.
```

This is not a universal insurance price.
Its value depends on the stated nominal model and the comparator.
A policy with a positive nominal premium can be cheaper under a different realized distribution.
It can also be worse than expected if the protection targets the wrong uncertainty.

Include more than one metric.
A cost objective may hide severe customer shortages.
A fill rate can hide the number of weeks with any shortage.
A worst-week cost can hide what happens in most weeks.
A decision maker should see the denominator and time horizon of each metric.

### Step 11 — Validate independently

**Input:** candidate policies, historical decision origins, and a deployment-like evaluation process.

**Action:** replay each decision with only the information available at the historical decision date.

**Output:** out-of-sample cost and service comparisons, uncertainty intervals where feasible, and a stress-test report.

Use a training period for distribution estimation.
Use separate validation periods for model and radius selection.
Reserve later test periods for a final evaluation that is not repeatedly tuned against.
For rolling deployment, reproduce the update cadence and lead times at each historical origin.
When observations are serially dependent, simple independent-observation confidence intervals can be misleading.
Choose an evaluation method that preserves relevant time dependence and report its assumptions.

Use paired comparisons when each policy can be evaluated against the same realized demand path.
Pairing removes some background variation from policy differences.
It does not fix an unrealistic simulator, missing counterfactual capacity constraints, or unobserved lost demand.

Check realized coverage separately from decision performance.
A set can cover many historical outcomes and still miss the damaging ones.
A policy can perform well in a small holdout by chance.
Distributional shift may make the next period different from all recorded validation periods.

Stress testing complements, rather than replaces, holdout evaluation.
A stress scenario is a stated what-if unless supported by a probability model.
Do not calculate an expected annual loss from an unweighted scenario list without an explicit frequency assumption.

### Step 12 — Approve, execute, and monitor

**Input:** the comparison, constraints, validation results, and remaining limitations.

**Action:** obtain an authorized decision and translate it into an operational commitment.

**Output:** a decision record, execution instruction, and monitoring thresholds.

The record should identify the data cutoff, cost version, set geometry, radius, support, solver version if applicable, and approver.
It should also identify any manual override and its rationale.
The commitment must be feasible under actual staffing, space, and supplier terms.

Monitor demand, shortages, booking utilization, model error, and distribution shift at the chosen cadence.
Monitor whether key assumptions have failed, not just whether the objective was low.
An early promotion announcement or sustained demand outside the modeled support should trigger review.
A breach is evidence for investigation, not automatic proof that a particular radius was mathematically wrong.

## 6. Worked example

### Business setting and decision

A regional fulfillment center must reserve next week's labor-equivalent processing capacity.
Management can book 80, 100, or 120 pallets per week.
The contract charges USD 2 for each booked pallet of weekly capacity, whether used or not.
Demand above capacity incurs an estimated shortage cost of USD 10 per unserved pallet.
For this teaching example, the shortage penalty aggregates the business consequences and is assumed linear.
No overtime, carryover, minimum service constraint, or demand response to the decision is available.

The demand forecast is represented by three demand levels.
The nominal probabilities are planning assumptions, not a guarantee or a contractual commitment.
The team suspects that the high-demand probability may be underestimated.

| Demand scenario | Demand, pallets/week | Nominal probability |
|---|---:|---:|
| Low | 80 | 0.60 |
| Middle | 100 | 0.30 |
| High | 120 | 0.10 |

The nominal expected demand is

```text
0.60 * 80 + 0.30 * 100 + 0.10 * 120 = 90 pallets/week.
```

Expected demand is not the optimization answer.
Booking 90 pallets is not even one of the three permitted contract choices.
The shortage penalty makes the shape of demand important in addition to its mean.

### Scenario costs

Use `cost = 2 * capacity + 10 * max(demand - capacity, 0)`.
All entries below are USD per week.

| Booked capacity, pallets/week | Demand 80 | Demand 100 | Demand 120 |
|---:|---:|---:|---:|
| 80 | 160 | 360 | 560 |
| 100 | 200 | 200 | 400 |
| 120 | 240 | 240 | 240 |

For capacity 80 and demand 100, capacity costs USD 160 and 20 unserved pallets cost USD 200.
The total is USD 360 for that week.
For capacity 100 and demand 120, capacity costs USD 200 and 20 unserved pallets cost USD 200.
The total is USD 400 for that week.
At capacity 120, all modeled demand is served and the capacity charge is USD 240 regardless of the scenario.

### Nominal stochastic decision

The nominal expected costs are

```text
80:  0.60 * 160 + 0.30 * 360 + 0.10 * 560 = USD 260/week
100: 0.60 * 200 + 0.30 * 200 + 0.10 * 400 = USD 220/week
120: 0.60 * 240 + 0.30 * 240 + 0.10 * 240 = USD 240/week
```

The nominal decision is therefore 100 pallets per week at USD 220 expected weekly cost.
The expected weekly shortage at that choice is `0.10 * 20 = 2` pallets.
The expected demand is 90 pallets, so the ratio of expected served pallets to expected demand is `88 / 90`, approximately 97.78%.
That ratio is a volume-weighted fill-rate measure under the nominal distribution.
It is not the probability of a week with no shortage.
The latter is 90% under the nominal distribution.

### Static robust decision

Let the uncertainty set contain the three demand levels.
The worst scenario costs are USD 560, USD 400, and USD 240 per week for capacities 80, 100, and 120 respectively.
The robust decision is 120 pallets per week.
Its worst-case cost is USD 240 per week.
Its nominal expected-cost premium relative to the nominal decision is USD 20 per week.

This premium buys protection against shortage within the three-value support.
It does not protect against demand above 120 pallets per week.
It does not address an equipment failure that reduces actual capacity below the booking.
Those possibilities were excluded, not solved.

### DRO with a small probability ambiguity radius

Use the same support and let nominal probabilities be `p0 = (0.60, 0.30, 0.10)`.
Let total variation radius `rho = 0.05`.
A feasible adversary can move 0.05 probability from the low-cost scenario to the high-cost scenario.
For capacity 80, the loss range is USD 560 minus USD 160, or USD 400.
The worst expected cost increases by `0.05 * 400 = USD 20` per week, from USD 260 to USD 280.
One maximizing distribution is `(0.55, 0.30, 0.15)`.

For capacity 100, both the low and middle demand scenarios cost USD 200.
Moving 0.05 from either to high demand increases expected cost by `0.05 * (400 - 200) = USD 10` per week.
The worst expected cost is USD 230 per week.
For capacity 120, every scenario costs USD 240; changing probabilities cannot change the expectation.

| Capacity, pallets/week | Nominal expected cost, USD/week | DRO worst expected cost at radius 0.05, USD/week |
|---:|---:|---:|
| 80 | 260 | 280 |
| 100 | 220 | 230 |
| 120 | 240 | 240 |

At radius 0.05, the DRO decision remains 100 pallets per week.
That does not mean ambiguity has no effect: its evaluated worst expected cost rose from USD 220 to USD 230 per week.
It means the rise was not enough to change the selected capacity.

### Sensitivity to a larger radius

At radius 0.10, the worst expected costs are USD 300, USD 240, and USD 240 per week.
The 100- and 120-pallet decisions tie on the DRO objective.
The organization can choose 100 for lower committed capacity spend or 120 for no shortage on modeled support, but that tie-break is a separate preference.

At radius 0.15, the worst expected costs are USD 320, USD 250, and USD 240 per week.
The DRO choice becomes 120 pallets per week.
The threshold is specific to the support, cost function, nominal probabilities, and feasible decisions in this example.
It is not a general recommendation to use a radius of 0.10 or 0.15.

### A hypothetical shifted distribution

For an illustrative evaluation only, suppose a future period has probabilities `(0.30, 0.30, 0.40)` on the same support.
This is a hypothetical distribution, not a measured result or a forecast.
Its total variation distance from the nominal distribution is 0.30.
It lies outside the 0.05 and 0.15 ambiguity sets.

The expected costs under this hypothetical distribution are

```text
80:  0.30 * 160 + 0.30 * 360 + 0.40 * 560 = USD 380/week
100: 0.30 * 200 + 0.30 * 200 + 0.40 * 400 = USD 280/week
120: 0.30 * 240 + 0.30 * 240 + 0.40 * 240 = USD 240/week
```

The 120-pallet plan costs USD 40 less per week than the 100-pallet plan under that hypothetical distribution.
That comparison does not establish the future distribution or justify paying the nominal premium without further evidence.
It shows why the decision can reverse when high-demand probability increases.

Under the shifted distribution, expected shortage is 8 pallets per week for the 100-pallet plan.
Expected demand is 102 pallets per week.
The ratio of expected served pallets to expected demand is `94 / 102`, approximately 92.16%.
For the 120-pallet plan it is 100% on the modeled support.
Demand exceeding 120 remains outside this calculation.

### Decision memo

**Recommendation conditional on the illustrative business preference:** if management accepts a USD 20 per week nominal expected-cost premium to avoid shortage across the three modeled demand levels, reserve 120 pallets per week.
If the cost premium is not authorized and radius 0.05 captures the approved ambiguity, reserve 100 pallets per week and acknowledge the high-demand shortage exposure.
Neither recommendation is a binding booking until the contract and operational feasibility are confirmed.

**Evidence needed before real deployment:** validate the shortage penalty, investigate censored demand, estimate how often demand exceeds the modeled support, and test capacity availability.
Use a later holdout to compare policies and document the conditions under which the capacity booking can be revised.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
It enumerates the three feasible capacity choices and computes the exact finite-support total-variation worst expectation by moving probability from low-loss to high-loss outcomes.
The method assumes no additional constraints on the probability vector beyond nonnegativity, unit sum, and the stated total-variation radius.
It is not a general-purpose DRO solver for coupled, multistage, or continuous-support problems.

```python
from math import isclose

DEMAND = (80, 100, 120)  # pallets per week
CAPACITY = (80, 100, 120)  # pallets per week
NOMINAL = (0.60, 0.30, 0.10)
SHIFTED = (0.30, 0.30, 0.40)  # hypothetical evaluation distribution


def cost(capacity, demand):
    """Return USD per week for one decision and one demand outcome."""
    return 2.0 * capacity + 10.0 * max(demand - capacity, 0)


def expectation(losses, probabilities):
    return sum(loss * p for loss, p in zip(losses, probabilities))


def worst_tv_expectation(losses, nominal, radius):
    """Maximize expected loss on fixed finite support in a TV ball.

    TV(p, nominal) = 0.5 * sum(abs(p_i - nominal_i)).
    A probability transfer of t uses t units of the radius.
    No extra probability constraints are imposed.
    """
    if len(losses) != len(nominal) or not losses:
        raise ValueError("losses and nominal must have equal nonzero length")
    if not 0.0 <= radius <= 1.0:
        raise ValueError("radius must lie in [0, 1]")
    if any(p < 0.0 for p in nominal) or not isclose(sum(nominal), 1.0):
        raise ValueError("nominal probabilities must be nonnegative and sum to 1")

    p = list(nominal)
    order = sorted(range(len(losses)), key=lambda i: losses[i])
    low, high = 0, len(order) - 1
    remaining = radius
    eps = 1e-12
    while low < high and remaining > eps:
        i, j = order[low], order[high]
        if losses[j] <= losses[i]:
            break
        moved = min(p[i], 1.0 - p[j], remaining)
        p[i] -= moved
        p[j] += moved
        remaining -= moved
        if p[i] <= eps:
            low += 1
        if 1.0 - p[j] <= eps:
            high -= 1
    return expectation(losses, p), tuple(p)


def main():
    print("capacity nominal robust dro_005 dro_015 shifted")
    for capacity in CAPACITY:
        losses = tuple(cost(capacity, d) for d in DEMAND)
        nominal = expectation(losses, NOMINAL)
        robust = max(losses)
        dro_005, _ = worst_tv_expectation(losses, NOMINAL, 0.05)
        dro_015, _ = worst_tv_expectation(losses, NOMINAL, 0.15)
        shifted = expectation(losses, SHIFTED)
        print(f"{capacity:8d} {nominal:7.0f} {robust:6.0f} "
              f"{dro_005:7.0f} {dro_015:7.0f} {shifted:7.0f}")

    for radius in (0.0, 0.05, 0.10, 0.15):
        values = []
        for capacity in CAPACITY:
            losses = tuple(cost(capacity, d) for d in DEMAND)
            worst, _ = worst_tv_expectation(losses, NOMINAL, radius)
            values.append(worst)
        best = min(values)
        winners = [capacity for capacity, value in zip(CAPACITY, values)
                   if isclose(value, best, abs_tol=1e-9)]
        print(f"radius={radius:.2f} winners={winners} cost={best:.0f}")


if __name__ == "__main__":
    main()
```

Expected output; all costs are USD per week and capacities are pallets per week:

```text
capacity nominal robust dro_005 dro_015 shifted
      80     260    560     280     320     380
     100     220    400     230     250     280
     120     240    240     240     240     240
radius=0.00 winners=[100] cost=220
radius=0.05 winners=[100] cost=230
radius=0.10 winners=[100, 120] cost=240
radius=0.15 winners=[120] cost=240
```

At radius zero the ambiguity set contains only the nominal distribution.
At radius 0.10, the equal objective values are reported as a tie rather than silently broken.
The `SHIFTED` distribution is deliberately outside both displayed positive-radius balls.
The program does not estimate a radius, test statistical coverage, or authorize a booking.

## 8. Data, standards, and regulatory context

### Data needed for a decision

A robust model needs more than a list of demand observations.
Record the forecast available at each historical decision origin, the realized demand, and the information available before commitment.
Record order date, requested delivery date, fulfilled quantity, cancellations, and evidence of lost sales where available.
Retain the capacity booked, actual available capacity, and any emergency actions taken.

Use a stable definition of a pallet.
A physical pallet, a pallet-equivalent workload, and a pallet position are different measures.
If handling effort varies by product, convert to workload units with an explicit and versioned conversion.
Do not add units of delay to units of volume without a meaningful scaling rule.

Join records using reliable product, location, and time identifiers.
Preserve the raw source and document transformations.
Duplicate transactions, missing cancellations, or time-zone mismatches can create artificial peaks.
A source-system migration can create a structural break that looks like distribution shift.

### Observation is not demand

Orders can be constrained by allocation rules before they enter the dataset.
Shipments can be lower than demand because inventory was unavailable.
A canceled order can be a genuine disappearance of demand or a response to an unfulfilled promise.
The correct treatment depends on the decision and the evidence.
If true lost demand cannot be recovered, include that uncertainty in the limitations and stress design.

### Privacy and proportionality

For staffing or customer-level decisions, records may contain personal information.
Use only the fields needed for the planning purpose and apply the organization's access and retention controls.
The EU General Data Protection Regulation, Regulation (EU) 2016/679, establishes obligations for processing personal data where applicable; its scope depends on the facts and jurisdiction.
This educational material is not legal advice and does not decide whether a particular dataset or model is compliant.
See the regulation's official text in Sources and consult qualified counsel where needed.

### Standards and identification

GS1 identifiers can support consistent identification of products and locations across trading partners when implemented correctly.
A standard identifier does not itself establish demand accuracy, causality, probability calibration, or a legal service obligation.
Trading-partner agreements and applicable law determine which identifiers and exchange formats are required in a particular transaction.

ISO 31000 provides risk-management guidance; it is not a recipe for a DRO ambiguity set and does not certify that a model's radius is correct.
Use the guidance to structure risk ownership, communication, monitoring, and review, not as a substitute for quantitative validation.
A model can align with a risk-management process and still contain incorrect data or assumptions.

### Versioning for reproducibility

Preserve the data cutoff and the exact forecast vintage.
Store the scenario definitions, support, distance metric, radius, and probability constraints.
Keep the cost coefficients and any contractual limits separately versioned.
Log solver status, numerical tolerance, and any manual overrides.
Record the outcome after deployment without overwriting what was known at decision time.

When the metric changes, recalibrate the radius.
A total-variation radius is not numerically comparable to a Wasserstein radius.
When units or scaling change, revisit ground distances and any cost coefficients.
A newly introduced product or warehouse can invalidate a historical distance even if the mathematical code still runs.

### Regulatory claims and business constraints

A service agreement may create an obligation, but the model does not determine its legal meaning.
If there are safety, labor, competition, or product-specific rules, identify their source and jurisdiction with the relevant owner.
A policy that appears economical but violates an applicable constraint is not a feasible business option.
As of the version date of this lesson, the cited GDPR and standards sources are references for context, not a representation that all provisions apply to every supply chain.

## 9. Industry practice and case evidence

### What public evidence can and cannot establish

Public cases often describe a planning method without disclosing its full data, parameter choices, baseline, or counterfactual.
A published improvement attributed to optimization does not establish that a specific ambiguity radius would produce the same result elsewhere.
Treat peer-reviewed methodology, reported implementations, and hypothetical teaching examples as different forms of evidence.

### Supply-chain applications of robust optimization

Bertsimas and Sim describe the budget of uncertainty as a way to control the price of robustness in linear optimization.
Their published framework is methodological evidence for a tunable uncertainty budget, not a claim that a particular warehouse should use any specific budget.
The budget limits simultaneous adverse deviations in the mathematical set; it is not automatically a confidence level.

Ben-Tal and coauthors study a two-echelon, multiperiod retailer-supplier commitment problem under uncertain demand in *Manufacturing & Service Operations Management*.
They evaluate an adjustable robust formulation through simulation.
This is a published research case, not a claim that a named company achieved measured savings.
The lesson's capacity example is invented for teaching.

### Named operational case: IBM

An IBM-authored *Interfaces* paper reports use of a mixed-integer programming model and heuristics for semiconductor supply-chain planning.
The authors report that IBM used the method daily for operational and strategic planning and achieved substantial cost savings.
The public abstract does not establish that this deployment used the DRO formulation or ambiguity set in this lesson.
It is evidence of an operational optimization implementation, not evidence of a DRO-specific benefit.
**Inference:** its emphasis on computational methods and discrete decisions illustrates why operational feasibility and solution time matter alongside the uncertainty model.
The public abstract does not give enough detail to reproduce the savings calculation here.

### Distributionally robust evidence

Delage and Ye study distributionally robust optimization under moment uncertainty.
Their work illustrates why imperfect knowledge of a distribution can be modeled directly rather than represented by one fitted distribution.
The guarantees depend on their formulation and assumptions; citing a paper does not give a guarantee to a different model.

Esfahani and Kuhn study data-driven DRO using Wasserstein ambiguity sets.
Their results connect a specific set construction with tractable reformulations and statistical properties under stated conditions.
The ground metric, sampling assumptions, support, and loss function matter to the applicability of those results.
A total-variation ball over three scenarios is a different construction.

### How to interpret an implementation claim

Ask which business decision was changed and how often.
Ask what information was available at commitment and which actions were actually feasible afterward.
Ask whether the reported comparator was a practical incumbent policy or an unrealistically weak baseline.
Ask whether savings are realized, simulated, or model-estimated.
Ask whether the evaluation period includes the disruption the model claims to protect against.

A public implementation can demonstrate feasibility without proving general superiority.
A theoretical bound can demonstrate a conditional mathematical property without proving operational adoption.
A backtest can show historical performance without establishing future performance after structural change.
These forms of evidence complement each other; they are not substitutes.

### Business interpretation of protection

A manufacturer may use alternate suppliers to reduce exposure to one facility.
A retailer may reserve flexible capacity before a promotion.
A distributor may pre-position inventory against uncertain lead time.
These are generic decision patterns, not assertions that a named company used the exact model in this lesson.

In each case, quantify the reservation or inventory premium against the stated baseline.
Check whether the proposed flexibility is physically and contractually available during the adverse event.
Correlated failures can make nominally independent backup suppliers unavailable together.
The model must represent that shared exposure if it is material.

### Evidence ledger for the example

| Statement | Evidence status | Implication |
|---|---|---|
| Total-variation probability reweighting is the example's ambiguity construction | Mathematical definition in this lesson | Reproducible from stated probabilities |
| Three demand levels and USD costs | Hypothetical teaching assumptions | Not company observations |
| Capacity 100 minimizes nominal expected cost | Arithmetic under stated assumptions | Not a future outcome guarantee |
| Capacity 120 protects against modeled demand shortages | Consequence of finite support and cost model | Does not cover demand above 120 |
| Shifted distribution makes capacity 120 cheaper in expectation | Hypothetical comparison | Not evidence the shift will occur |
| Robust optimization has published supply-chain applications | Published modelling and simulation research cited below | Transferability requires local validation |

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Wrong support | Real demand exceeds every modeled scenario | Unseen magnitudes are excluded | Apparent protection fails | Revisit physical bounds and stress outside support |
| Censored demand | Sales track available stock exactly | Lost demand is unobserved | Tail is understated | Estimate lost demand where defensible; report uncertainty |
| Arbitrary radius | Radius chosen after seeing desired decision | Selection is driven by outcome | Misleading risk claim | Predefine calibration and show sensitivity |
| Unit confusion | TV radius described as pallets | Probability and physical distances are mixed | Incorrect interpretation | Document metric and units |
| Factor-of-two error | L1 and TV used interchangeably | TV is half L1 on finite support | Wrong allowed mass transfer | State formula and verify simple transfer arithmetic |
| Impossible joint stress | Every supplier fails independently at its maximum in one scenario | Box ignores structural constraints | Excessive cost | Add justified joint constraints |
| Missing common cause | Backup suppliers share a site or utility | Dependencies omitted | Protection is illusory | Map shared exposures and stress them jointly |
| Fixed support blind spot | No mass can reach an unobserved extreme | Probability reweighting only changes existing weights | Tail misspecification persists | Expand justified support or change ambiguity construction |
| Clairvoyant recourse | Overtime is selected using final weekly demand | Future information leaks into early action | Inflated policy value | Enforce observation-based nonanticipativity |
| Service confused with cost | Low expected cost presented as a fill-rate guarantee | Objective does not constrain service | Customer commitments missed | Define and enforce service measure if required |
| Overstated statistical coverage | Manager-selected radius called a 95% confidence set | Preference and inference are conflated | False assurance | Label calibration method and assumptions |
| Holdout leakage | Radius repeatedly tuned on final test period | Test data become training data | Optimistic evaluation | Separate selection and final evaluation |
| Wrong baseline | Robust plan compared only with no planning | Comparator is weak | Inflated benefit | Compare with current and nominal feasible policies |
| Cost double counting | Shortage penalty includes the same margin twice | Inconsistent cost definition | Excess capacity | Reconcile components with finance |
| Infeasible backup | Contract assumes unlimited overtime | Operational or legal cap omitted | Plan cannot execute | Validate capacity, lead time, and contract terms |
| False precision | Recommendations change with tiny coefficient changes | Inputs are imprecise | Fragile approval | Show ranges, thresholds, and decision stability |
| Dependence ignored | Weekly errors cluster during promotions | Independent sampling destroys persistence | Underestimated risk | Validate using time-respecting blocks or paths |
| Tail guarantee overstated | Worst expected cost described as worst realized loss | Expectation is confused with an extreme | Wrong risk communication | Report both measures explicitly |
| Missing solver status | Output table has no gap or feasibility information | Numerical or time-limit failure hidden | Unreliable recommendation | Record status and gap; diagnose infeasibility |
| Stale model | Repeated outcomes outside support | Regime has changed | Coverage and cost estimates degrade | Set monitoring triggers and accountable review |

### Reading a red flag

A warning does not uniquely identify its cause.
For example, repeated support breaches may reflect an actual shift, a unit conversion error, or an incorrect calendar join.
Investigate the data pipeline and operational context before changing the radius.
A larger radius on the same fixed support cannot solve an omitted-support problem.

A highly conservative solution is not automatically wrong.
It may reflect a genuine asymmetric loss or a hard service obligation.
It is a problem when the decision maker has not accepted the cost or when the adverse combination is unsupported.

## 11. Assumptions and limits

| Assumption | Why needed here | What breaks it | How to test or challenge it |
|---|---|---|---|
| Demand support is 80, 100, or 120 pallets/week | Makes the example finite and transparent | Intermediate or higher demand occurs | Compare with uncensored demand history and explicit stresses |
| Capacity choices are 80, 100, or 120 pallets/week | Represents the assumed contract menu | Other booking quantities are available | Confirm current contract and scheduling rules |
| Cost is USD 2 per booked pallet | Makes capacity premium measurable | Tiered rates, fixed fees, or overtime apply | Reconcile with actual contract and finance |
| Shortage costs USD 10 per pallet | Allows a common monetary comparison | Different customers have different harm or hard priorities | Estimate range and add relevant service constraints |
| No late recourse | Defines a static decision | Overtime or transfers are feasible | Review timing, costs, and caps with operations |
| Booked capacity is delivered | Separates demand uncertainty from supply uncertainty | Absenteeism or equipment failure reduces throughput | Compare booking with observed availability |
| Demand is unaffected by capacity | Treats demand as external | Availability changes orders or customers switch channels | Examine substitution and customer behavior |
| Weekly outcomes are independent for the simple expectation | Permits one-period calculations without carryover | Backlogs and demand persist across weeks | Inspect autocorrelation and simulate paths |
| Nominal probabilities are credible reference values | Establishes a benchmark | Data are biased or regime has changed | Check vintage, representativeness, and rolling errors |
| Total variation measures relevant ambiguity | Allows probability reweighting on fixed support | Magnitude and dependence errors dominate | Compare with expanded-support and alternative-set stresses |
| Capacity and shortage costs are linear | Enables simple arithmetic | Congestion and nonlinear penalties occur | Fit or elicit piecewise costs and test sensitivity |
| One objective captures the decision | Lets the model rank alternatives | Service, equity, or safety constraints bind | Add explicit constraints and consult owners |
| Evaluation distribution is hypothetical | Illustrates shifted outcomes | It is mistaken for observed future evidence | Label it clearly and use actual held-out data for validation |
| Cost estimates are comparable across policies | Supports fair comparisons | One policy has omitted implementation costs | Include transition and operating costs consistently |

The expected-value calculations do not require future weeks to be independent when interpreted as one-week marginal expectations.
Independence becomes relevant for many multiweek probability calculations and simple confidence intervals.
Do not infer a multiweek guarantee from a one-week example.

A worst-case bound is conditional on its set.
It does not say the worst case will occur, and it does not protect against a case the set excludes.
A statistical guarantee, if available for a particular DRO construction, is also conditional on its sampling and modelling assumptions.

## 12. Exercises

### Exercise 1 — Realization versus distribution

**Time:** 10 minutes.

A planner writes: “Demand will be either 80 or 120 pallets next week, and high demand has probability 0.20.”
Identify the realizations, the support, and the distribution.
Then describe an uncertainty set and an ambiguity set that use those two outcomes.

**Deliverable:** four labeled definitions and one sentence distinguishing their roles.

**Acceptance criteria:** realizations have pallet units; probabilities sum to one; the ambiguity set contains distributions rather than individual demand values.

### Exercise 2 — Decision timing

**Time:** 15 minutes.

Capacity is booked Friday, early orders arrive Monday, and a capped overtime decision is made Tuesday.
Final demand is known Friday of the following week.
Classify the booking and overtime actions and state what information overtime may use.

**Deliverable:** a four-event timeline and one nonanticipativity rule.

**Acceptance criteria:** the booking does not depend on next week's demand; overtime depends only on observations available by Tuesday; the cap is acknowledged.

### Exercise 3 — Cost table

**Time:** 20 minutes.

Using the worked example, calculate all three scenario costs for capacity 100 pallets per week.
Calculate its nominal expected cost and worst scenario cost.

**Deliverable:** three costs and two summaries, each with units.

**Acceptance criteria:** show capacity and shortage components and use nominal probabilities 0.60, 0.30, and 0.10.

### Exercise 4 — Total variation arithmetic

**Time:** 20 minutes.

For `p0 = (0.60, 0.30, 0.10)` and `p = (0.55, 0.30, 0.15)`, calculate L1 distance and total variation distance.
Use the capacity-100 loss vector to calculate expected cost under `p`.
Explain why this distribution is feasible for radius 0.05.

**Deliverable:** distance arithmetic and expected cost in USD per week.

**Acceptance criteria:** include the factor of one half and distinguish probability mass from demand units.

### Exercise 5 — Radius and business tradeoff

**Time:** 25 minutes.

Compare capacity choices 100 and 120 at total-variation radius 0.15.
Calculate the nominal expected-cost premium of the 120 choice.
Explain a reason a decision maker might reject that premium even though the DRO objective favors 120.

**Deliverable:** a two-row comparison and a three-sentence decision recommendation.

**Acceptance criteria:** use consistent USD per week; distinguish the nominal evaluation from the DRO objective; state a decision preference rather than inventing a probability guarantee.

### Exercise 6 — Support critique

**Time:** 20 minutes.

A manager says: “We used a large total-variation ball on the three historical demand levels, so demand of 150 pallets per week is covered.”
Assess the claim and propose two ways to investigate the missing risk.

**Deliverable:** a short critique and two concrete analysis actions.

**Acceptance criteria:** explain fixed support, avoid treating an unweighted stress as a probability, and identify the evidence needed for any support expansion.

### Exercise 7 — Validation design

**Time:** 30 minutes.

Design a rolling-origin comparison of the nominal, robust, and two DRO policies for a weekly warehouse decision.
Include policy selection, a later holdout, metrics, and a plan for clustered demand shocks.

**Deliverable:** a one-page evaluation plan and a table of data available at each decision origin.

**Acceptance criteria:** no future information at the booking date; radius selection is separate from final test; paired policy comparisons use common demand paths; limits of historical evaluation are stated.

### Exercise 8 — Executive decision memo

**Time:** 30 minutes.

Write a recommendation for the example under a stated willingness to pay for protection.
Include the forecast, plan, target if any, and commitment as separate items.
Add two conditions for approval and two monitoring triggers.

**Deliverable:** a memo of at most 300 words and a compact risk ledger.

**Acceptance criteria:** cite the USD 20 per week nominal premium if choosing 120 over 100; do not claim protection above the modeled support; assign an owner and review cadence.

## 13. Self-check answers

### Exercise 1

The realizations are 80 and 120 pallets per week.
The support is `{80, 120}` pallets per week.
The stated distribution assigns probability 0.80 to 80 and 0.20 to 120.
An uncertainty set can contain the two realizations.
An ambiguity set can contain all two-entry probability vectors within a stated total-variation radius of `(0.80, 0.20)`.
The first set contains demand values; the second contains probability distributions.

### Exercise 2

Friday's booking is a first-stage action.
Tuesday's capped overtime is recourse after Monday's early-order information arrives.
Two future paths with identical information through Tuesday must receive the same Tuesday overtime decision.
Final weekly demand cannot be used to choose Tuesday overtime unless it is genuinely known by then.
The overtime cap and cost belong in the feasible recourse model.

### Exercise 3

At 80 pallets of demand, capacity 100 costs USD 200 with no shortage.
At 100 pallets, it also costs USD 200.
At 120 pallets, it costs USD 200 for capacity plus USD 200 for 20 unserved pallets, or USD 400.
Nominal expected cost is `0.60 * 200 + 0.30 * 200 + 0.10 * 400 = USD 220/week`.
Worst scenario cost is USD 400 per week.
The expected cost and the worst scenario cost answer different questions.

### Exercise 4

The L1 distance is `|0.55-0.60| + |0.30-0.30| + |0.15-0.10| = 0.10`.
Total variation is half of that, or 0.05.
The distribution is feasible at radius 0.05 because the distance equals the radius and the entries are nonnegative and sum to one.
Expected capacity-100 cost is `0.55 * 200 + 0.30 * 200 + 0.15 * 400 = USD 230/week`.
The five percentage points transferred are probability mass, not five pallets of demand.

### Exercise 5

At radius 0.15, the DRO worst expected costs are USD 250 per week for capacity 100 and USD 240 per week for capacity 120.
The 120 choice has nominal expected cost USD 240 per week compared with USD 220 for the 100 choice.
Its nominal premium is therefore USD 20 per week.
A decision maker might prefer the 100 choice if the radius is not supported by evidence or if the premium exceeds the approved risk budget.
That preference does not change the mathematical ranking at the specified radius; it changes the business criterion or set selection.

### Exercise 6

The claim is false for a fixed-support set.
The adversary can change probabilities only among 80, 100, and 120 pallets in the stated construction.
A demand outcome of 150 is not included even at the largest radius.
Investigate uncensored demand and operational drivers to assess whether 150 is plausible.
Separately run an explicitly labeled 150-pallet stress and consider expanding the support or using a suitable alternative ambiguity set.
An unweighted stress does not itself establish the event's probability.

### Exercise 7 rubric

A strong plan freezes information at each Friday booking date.
It fits a nominal model on earlier data and selects the radius using a separate validation period.
It evaluates the selected policies once on a later holdout or a pre-specified sequence of rolling test origins.
It reports paired cost differences, volume-weighted fill rate, shortage frequency, and operational feasibility.
It accounts for clustered shocks with a time-aware method and explains limited sample size and regime-shift limitations.
A weak plan randomly shuffles time periods, tunes on the final test set, or assumes a stress has a known probability without evidence.

### Exercise 8 rubric

A strong memo states the chosen capacity, its evaluation basis, and the USD 20 per week nominal premium where applicable.
It distinguishes the nominal distribution from the committed capacity and names any service target separately.
It names an approver, data and cost checks, and a feasible execution deadline.
It includes support breach and capacity-availability deterioration as possible triggers.
It does not claim that the three-point example covers every real-world disruption.

## 14. Interview questions

1. **What distinguishes robust optimization from DRO?**
   A strong answer distinguishes a worst realization from a worst distribution and explains the chosen objective.
   A weak answer treats both terms as synonyms for adding a safety factor.
2. **What is the business meaning of a total-variation radius?**
   A strong answer explains permitted probability mass transfer and the half-L1 convention.
   A weak answer calls the radius a number of pallets or a guaranteed service percentage.
3. **Can DRO protect against a value absent from the nominal sample?**
   A strong answer says it depends on the ambiguity set and support; fixed-support reweighting cannot.
   A weak answer says every DRO method automatically creates realistic unseen events.
4. **Why might a box uncertainty set be too conservative?**
   A strong answer discusses simultaneous extremes and physical or dependence constraints.
   A weak answer removes inconvenient stress scenarios without evidence.
5. **What is the cost of robustness?**
   A strong answer specifies the comparator and evaluation distribution and notes that realized differences can reverse.
   A weak answer presents a universal percentage independent of the model.
6. **How do you prevent a recourse model from cheating?**
   A strong answer maps observations and actions over time and enforces nonanticipativity.
   A weak answer lets every scenario choose a complete plan after observing its final outcome.
7. **How would you calibrate an ambiguity radius?**
   A strong answer distinguishes statistical coverage, historical validation, and management risk preference.
   A weak answer tunes against the final test set and calls the result a confidence guarantee.
8. **What would you report to an operations leader?**
   A strong answer reports the action, premium, service effects, support exclusions, and decision trigger.
   A weak answer reports only the solver objective without units or an owner.
9. **Why is a worst expected cost not necessarily a worst realized cost?**
   A strong answer explains that a distribution averages over its outcomes and may not place all mass on the worst one.
   A weak answer assumes the two always coincide.
10. **How would you test a policy after deployment?**
    A strong answer proposes time-respecting comparisons, monitoring for shift, and operational feasibility checks.
    A weak answer treats a single good week as proof that the model is calibrated.

## 15. Key terms

- [Adjustable robust optimization](#4-definitions-and-vocabulary): robust decisions with observation-dependent recourse.
- [Ambiguity radius](#step-6--choose-an-ambiguity-set-for-dro): permitted deviation under a specified distributional metric.
- [Ambiguity set](#3-the-core-mental-model): family of candidate distributions.
- [Backtest](#step-11--validate-independently): historical replay using only information available at each origin.
- [Box set](#step-5--choose-an-uncertainty-set-for-robust-optimization): componentwise bounds on uncertain values.
- [Budgeted uncertainty](#step-5--choose-an-uncertainty-set-for-robust-optimization): controlled joint deviations from baseline.
- [Calibration](#step-7--select-and-challenge-the-radius): evidence-based choice of a model parameter or set size.
- [Chance constraint](#step-8--formulate-the-optimization): constraint involving a specified violation probability.
- [Decision horizon](#step-1--frame-the-decision-and-its-owner): period over which action and outcomes are assessed.
- [Distribution shift](#why-distributional-misspecification-matters): difference between reference and deployment distributions.
- [DRO](#3-the-core-mental-model): distributionally robust optimization over an ambiguity set.
- [Empirical distribution](#4-definitions-and-vocabulary): distribution constructed from observed samples.
- [Expected cost](#nominal-stochastic-decision): probability-weighted cost.
- [Fill rate](#nominal-stochastic-decision): served volume divided by demand volume under a stated aggregation.
- [Ground metric](#step-6--choose-an-ambiguity-set-for-dro): distance used to define transport cost in a Wasserstein set.
- [Holdout](#step-11--validate-independently): evaluation data reserved from fitting and selection.
- [L1 distance](#step-6--choose-an-ambiguity-set-for-dro): sum of absolute differences between probability vectors.
- [Moment set](#step-6--choose-an-ambiguity-set-for-dro): distributions satisfying stated moment restrictions.
- [Nominal distribution](#step-4--build-a-nominal-benchmark): reference probability model.
- [Nonanticipativity](#step-2--draw-the-information-timeline): information-consistent decisions across scenarios.
- [Out-of-sample evaluation](#step-11--validate-independently): evaluation using data not used for policy selection.
- [Recourse](#the-timeline-matters): action taken after specified information arrives.
- [Robust constraint](#different-kinds-of-protection): constraint satisfied for every allowed realization.
- [Robustness premium](#step-10--quantify-the-tradeoff): cost difference from a stated benchmark.
- [Scenario](#step-5--choose-an-uncertainty-set-for-robust-optimization): specified future realization or path.
- [Service level](#step-10--quantify-the-tradeoff): explicitly defined service performance measure.
- [Stress test](#step-11--validate-independently): evaluation under a stated adverse condition.
- [Support](#step-6--choose-an-ambiguity-set-for-dro): values the model allows a random variable to take.
- [Total variation](#step-6--choose-an-ambiguity-set-for-dro): probability distance equal to half L1 on finite support.
- [Uncertainty set](#step-5--choose-an-uncertainty-set-for-robust-optimization): set of allowed uncertain realizations.
- [Wasserstein distance](#step-6--choose-an-ambiguity-set-for-dro): minimum transport cost under a specified ground metric.
- [Worst-case expected cost](#dro-with-a-small-probability-ambiguity-radius): largest expectation over the ambiguity set.

## 16. Sources

1. Bertsimas, D., and Sim, M. (2004). “The Price of Robustness.” *Operations Research* 52(1), 35–53. [https://doi.org/10.1287/opre.1030.0065](https://doi.org/10.1287/opre.1030.0065). Budgeted uncertainty and its tradeoffs.
2. Ben-Tal, A., Golany, B., Nemirovski, A., and Vial, J.-P. (2005). “Retailer-Supplier Flexible Commitments Contracts: A Robust Optimization Approach.” *Manufacturing & Service Operations Management* 7(3), 248–271. [https://doi.org/10.1287/msom.1050.0081](https://doi.org/10.1287/msom.1050.0081). Robust supply-chain commitments.
3. Denton, B. T., and coauthors (2006). “IBM Solves a Mixed-Integer Program to Optimize Its Semiconductor Supply Chain.” *Interfaces*. [https://doi.org/10.1287/inte.1060.0238](https://doi.org/10.1287/inte.1060.0238); [IBM Research abstract](https://research.ibm.com/publications/ibm-solves-a-mixed-integer-program-to-optimize-its-semiconductor-supply-chain). Operational optimization case, not a claim of DRO use.
4. Delage, E., and Ye, Y. (2010). “Distributionally Robust Optimization Under Moment Uncertainty with Application to Data-Driven Problems.” *Operations Research* 58(3), 595–612. [https://doi.org/10.1287/opre.1090.0741](https://doi.org/10.1287/opre.1090.0741). Moment-based ambiguity.
5. Mohajerin Esfahani, P., and Kuhn, D. (2018). “Data-Driven Distributionally Robust Optimization Using the Wasserstein Metric: Performance Guarantees and Tractable Reformulations.” *Mathematical Programming* 171, 115–166. [https://doi.org/10.1007/s10107-017-1172-1](https://doi.org/10.1007/s10107-017-1172-1). Wasserstein DRO and conditional guarantees.
6. Rahimian, H., and Mehrotra, S. (2019). “Distributionally Robust Optimization: A Review.” [https://arxiv.org/abs/1908.05659](https://arxiv.org/abs/1908.05659). Survey of ambiguity sets and formulations.
7. European Union. Regulation (EU) 2016/679, General Data Protection Regulation. [Official EUR-Lex text](https://eur-lex.europa.eu/eli/reg/2016/679/oj). Scope and obligations require case-specific interpretation.
8. ISO. [ISO 31000 risk management](https://www.iso.org/iso-31000-risk-management.html). General risk-management guidance, not a model validation certificate.
9. GS1. [GS1 standards](https://www.gs1.org/standards). Identification and data-sharing standards; implementation and contractual requirements vary.

## 17. Links

- [Module 20 topic index](_index.md).
- [Input modelling](03_input-modeling.md): distribution fitting, missingness, and input uncertainty.
- [Validation](04_validation.md): face, structural, and output validation.
- [Discrete-event simulation](02_discrete-event-simulation.md): simulate queues and resource constraints that a static model omits.
- [Twin, shadow, and simulation](01_twin-shadow-simulation.md): select the appropriate model family and operating purpose.
- [Module 20 survey and its lab material](../../../course/20_digital_twins_simulation_optimization_uncertainty.md): broader context and simulation-study exercises.
