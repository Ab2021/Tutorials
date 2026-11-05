# Stochastic Optimization for Supply-Chain Decisions

> Module 20 — Digital Twins, Simulation, and Optimization under Uncertainty  
> Topic 06 of 10  
> Estimated study time: 4 hours  
> Prerequisites: scenario inputs, basic probability, linear optimization, and simulation validation  
> Version 2026.5

## 1. Why this topic exists

A procurement manager must reserve stock before the next selling period.
Demand is not known when the order is placed.
Ordering too little can cause lost sales, emergency freight, or production stoppages.
Ordering too much ties up cash and can leave inventory that must be discounted or discarded.
A single forecast does not reveal that trade-off.

The same timing problem appears in many decisions:

- A manufacturer books transport capacity before customers finalize their orders.
- A retailer buys seasonal goods before weather and demand are known.
- A distribution network reserves warehouse labor before volumes are observed.
- A buyer signs a supplier capacity agreement before a disruption is known.
- A planner assigns inventory to markets before local demand is realized.

In each case, some decisions must be made before uncertainty resolves.
Other decisions remain available later, but they may be more expensive or constrained.
A stochastic optimization model represents both the timing and the consequences.
It chooses an implementable decision by comparing outcomes across possible futures.

The manager's question is not whether the model predicts the future perfectly.
It is whether committing to a particular action now is economically preferable to the alternatives, given defensible assumptions about future outcomes and available responses.
A high expected saving alone is not enough if the business cannot fund the action, execute the response, or tolerate the downside.

The analysis in this topic addresses a single-period purchasing decision.
It uses a small scenario distribution so that the calculation can be checked by hand.
Real studies may use many products, locations, periods, and correlated sources of uncertainty.
The discipline is the same: make the decision timing explicit, price consequences consistently, and test the chosen decision on data not used to select it.

### Decision ownership

The procurement owner decides whether to place an order.
Finance checks cost definitions, time value, and budget impact.
Commercial owners help estimate demand and the economics of a missed sale.
Operations confirms whether any emergency or disposal response is physically feasible.
An analyst builds and evaluates the model; the analyst does not own the risk appetite.

### What is different from simulation alone

A simulation estimates what happens under a specified policy.
Optimization searches over decisions according to a stated objective and constraints.
A simulation can evaluate an optimized policy, but a favorable simulation result does not by itself prove that the policy is optimal.
Likewise, an optimizer's objective value is not evidence that the input distribution or operating assumptions are correct.

### Forecast, plan, target, and commitment

A **forecast** is an estimate of future demand and its uncertainty.
A **plan** specifies what the organization intends to do, conditional on the stated information.
A **target** is a desired performance level, such as a 95% fill rate.
A **commitment** is an action or obligation that is difficult or costly to reverse, such as a purchase order already accepted by a supplier.
The expected-demand quantity is a forecast statistic; it is not automatically the economically best commitment.
A service target is a business requirement, not a probability distribution.

## 2. Learning outcomes

After working through this topic, you should be able to:

1. Draw the information timeline and distinguish a first-stage commitment from scenario-dependent recourse.
2. Construct a finite scenario model with probabilities, units, costs, and nonanticipativity.
3. Explain sample average approximation (SAA) and why its optimization sample is not an independent test sample.
4. Compute the expected-value decision, stochastic solution, value of the stochastic solution (VSS), and expected value of perfect information (EVPI) for a cost-minimization problem.
5. Evaluate two fixed policies out of sample using paired costs and an uncertainty interval.
6. Convert a modeled improvement into a decision on investment using a consistent counterfactual, time horizon, and implementation cost.
7. Diagnose correlation, rare-event, feasibility, and information-leakage problems in a proposed scenario study.
8. Recommend a decision with limitations, owners, monitoring triggers, and an escalation path.

Exercises 1–8 map respectively to these outcomes.
Use the self-check section to inspect the arithmetic and the design rubric.

## 3. The core mental model

```text
   Before demand is known                     After demand is known
   ----------------------                     ---------------------
   Forecast + scenarios + costs               Scenario s is observed
              |                                        |
              v                                        v
   Choose one order q ---------------------->  Sell, short, or salvage
   Same q in every scenario                    Pay cost C(q, D_s)
              |                                        |
              +---------- expected cost <--------------+
                             |
                             v
                    Select the lowest-cost q
                             |
                 independent future scenarios
                             |
                 compare fixed selected policies
                             |
                 business decision and monitoring
```

The crucial arrow crosses the boundary between what is known now and what will be known later.
The first-stage order cannot depend on tomorrow's demand when tomorrow's demand is unavailable today.
If a model gives each scenario a different pre-demand order, it has granted the planner information the planner does not possess.
That solution can be a useful perfect-information benchmark, but it is not an implementable ordering policy.

A scenario is a coherent possible realization of uncertain inputs.
A scenario may include demand, supplier availability, lead time, and freight cost together.
These quantities should be sampled jointly when they are dependent.
For example, a port disruption can lengthen lead time and raise spot freight prices in the same scenario.
Sampling each variable independently would break that relationship and can understate the cost of disruption.

For a cost objective, the model minimizes the probability-weighted average cost.
Weights must represent the decision maker's assumed distribution, not simply a list of interesting stories.
If there are two equally likely scenarios, their weights are 0.5 and 0.5.
If a rare disruption receives an intentionally inflated stress-test weight, label the resulting objective as a stress-weighted criterion rather than an unbiased expected cost.

The model is a decision aid, not a crystal ball.
An optimal answer means optimal within the modeled action set, constraints, cost coefficients, and probability distribution, subject to solver tolerances.
Independent evaluation measures how a selected decision performs on another set of outcomes.
Neither exercise guarantees performance under a future distribution that differs from both samples.

### Why the average forecast can mislead

Costs are often asymmetric or nonlinear.
A shortage may cost far more than the carrying cost of a surplus.
Applying an optimizer to the mean input ignores that asymmetry until after a commitment has already been chosen.
The best quantity under the mean input can therefore differ from the quantity minimizing expected cost over the distribution.
The difference in their expected costs is a measure of the value of taking the distribution into account, not a guarantee of realized savings.

### What the model does not decide

Expected-cost minimization does not automatically satisfy a service commitment, a cash limit, or a board risk limit.
These can be constraints or separate criteria if specified before optimization.
A target chosen after seeing an unattractive result is not the same as an independently governed service requirement.
If the organization cannot quantify its tail exposure credibly, compare stress tests and alternative distributions as well as the expected-cost result.

## 4. Definitions and vocabulary

| Term | Definition and units | Common misuse |
|---|---|---|
| Decision | Controllable choice, such as order quantity in units per period | Treating uncertain demand as controllable |
| Uncertain parameter | Input not known at decision time, such as future demand in units | Treating an estimate as a guaranteed value |
| Scenario | Joint realization of relevant uncertain parameters | Calling a policy alternative a scenario |
| Scenario probability | Assumed probability mass, dimensionless, nonnegative and summing to one | Assigning story weights without explaining the implied distribution |
| Scenario tree | Branching representation of information arriving through time | Allowing early actions to depend on later branches |
| First stage | Decisions made before the represented uncertainty is observed | Making a separate initial commitment for each scenario |
| Recourse | Feasible action after some uncertainty is observed | Assuming a response is available without lead-time or capacity evidence |
| Two-stage model | Model with a first decision and then scenario-dependent recourse | Applying it unchanged when information arrives in several rounds |
| Nonanticipativity | Constraint that actions with the same information history must agree | Equating only actions that share the same final outcome |
| Objective | Measurable quantity to minimize or maximize, such as USD per period | Mixing cost, revenue, and risk without a stated interpretation |
| Expected cost | Probability-weighted average of costs, in USD per period | Calling it the most likely or guaranteed cost |
| Feasible solution | Decision satisfying all modeled constraints | Assuming feasibility outside the model's scope |
| Complete recourse | Every feasible first-stage decision has feasible recourse in every possible scenario | Assuming penalty variables make physical impossibility disappear |
| Sample average approximation | Replacing an expectation by the average over a finite sampled set | Treating the optimized sample average as an unbiased estimate of its own policy's future performance |
| Training sample | Scenarios used to select or tune a policy | Reporting its best objective as a holdout result |
| Test sample | Independent scenarios or observations reserved for evaluation | Repeatedly selecting policies against it and still calling it untouched |
| Common random numbers | Same simulated outcomes used to evaluate competing policies | Using separate draws and calling the resulting difference paired |
| Optimality gap | Difference between incumbent and bound reported by a solver, expressed in its stated units or relative convention | Confusing the solver gap with error from an incorrect probability model |
| Expected-value problem | Deterministic problem formed by replacing uncertain inputs with their means | Calling its objective value the expected cost of its decision |
| EEV | Expected cost of the expected-value decision evaluated over the uncertainty distribution | Using the deterministic objective itself as EEV |
| RP | Optimal expected cost of the stochastic recourse problem | Calling an in-sample objective a proven real-world cost |
| VSS | EEV minus RP for minimization, in USD per period | Adding it to EVPI or calling it a guaranteed cash saving |
| Wait-and-see value | Expected cost if each scenario were known before choosing its first-stage action | Treating it as implementable without perfect advance information |
| EVPI | RP minus wait-and-see expected cost for minimization | Treating it as the value of any particular imperfect forecast |
| Stress test | Evaluation under selected adverse assumptions | Calling a deliberately selected case a probability-weighted forecast |
| Fill rate | Fraction of demand satisfied, defined over a stated aggregation and horizon | Confusing order fill, unit fill, and probability of no stockout |
| Opportunity cost | Value of the best forgone alternative, in the same currency and horizon | Charging lost revenue and contribution margin twice |
| Net present value | Discounted benefit less discounted cost, in a stated base-date currency | Adding savings from different years without discounting |
| Distribution shift | Future data-generating behavior differs from the modeled distribution | Treating a narrow simulation interval as protection from structural change |

## 5. The mechanics, step by step

### Step 1: Frame the business choice

**Input:** a pending decision, its deadline, and the owner who can authorize it.

Write the decision as a choice among feasible actions.
State the planning period, currency, geographic scope, and product scope.
Record what can still change after the decision deadline and what becomes a contractual commitment.
For example, an order placed on Monday may be fixed for the week, while inventory disposal can be chosen after the selling period.

Specify the baseline before calculating model benefits.
The baseline might be the current replenishment rule, an approved forecast-based order, or the actual feasible manual plan.
The baseline must be evaluated under the same demand scenarios and cost accounting as the proposed policy.
A comparison with an unrealistic baseline can make a weak policy look valuable.

**Output:** a decision charter with an owner, deadline, baseline, scope, units, and success criteria.

**Branch:** if the decision can be postponed cheaply until uncertainty resolves, compare postponement explicitly.
If the response has to be committed now, a first-stage decision is appropriate.
If there are multiple observation and decision dates, use a multistage representation rather than silently giving the planner future information.

### Step 2: Draw the information timeline

**Input:** contractual dates, data availability, operating calendars, and decision rights.

List each action in chronological order.
For each action, list the information actually observed at that time.
Distinguish a forecast update from realized demand.
Record decision lead times, cancellation rights, and any minimum order or capacity restrictions.

A two-stage model assumes that all represented uncertainty is observed before second-stage recourse.
That can be an acceptable approximation when the modeled response happens after the selling period.
It can be misleading when inventory must be allocated during the period while demand continues to arrive.
In the latter case, a recourse decision that sees final demand may overstate flexibility.

**Output:** a timeline connecting decisions to the information available at each decision point.

**Branch:** use a scenario tree with nonanticipativity by shared history when decisions recur as information arrives.
For example, two branches sharing the same first-week demand must have the same decision at the end of that week, even if their second-week demand differs.

### Step 3: Define uncertain quantities and scenarios

**Input:** history, forecasts, operating knowledge, structural assumptions, and data lineage.

For each uncertain variable, state its unit, horizon, possible range, and dependence on other variables.
Demand should refer to unconstrained customer demand where that is the modeled concept.
Observed sales may be censored by stockouts; they should not automatically be treated as true demand.
Supplier disruption, yield, and lead time may be related to demand or each other.

Create scenarios that preserve relevant joint relationships.
Alternatives include historical blocks, a fitted multivariate distribution, a forecasting model's joint predictive samples, and expert-elicited discrete states.
Each method has trade-offs.
A historical block preserves observed joint patterns but cannot represent an event never observed.
A fitted distribution can generate unseen outcomes but depends on functional-form and parameter assumptions.
Expert scenarios can express structural change but require explicit elicitation and review of probabilities.

Check that scenario probabilities are nonnegative and sum to one.
For empirical equal-weight samples of size N, each sampled scenario has weight 1/N; repeated observations remain repeated unless weights are aggregated correctly.
Do not average demand, cost, and lead time independently if their correlation changes the economics.

**Output:** a versioned scenario set with weights, joint variables, generating method, and provenance.

**Branch:** when probabilities cannot be defended, report sensitivity to alternative plausible distributions and use stress scenarios separately.
Robust or distributionally robust formulations address a different objective; they should not be labeled ordinary expected-cost minimization.

### Step 4: Define decisions and recourse

**Input:** the timeline and operational action set.

In the example, `q` is an order in units and must be chosen before demand `D` is realized.
After demand is observed, unsold units may be salvaged, and unsatisfied demand is treated as a shortage.
These quantities are consequences of `q` and `D`, not separately chosen sources of supply.

A general two-stage model is:

```text
minimize over x in X:  c(x) + E[Q(x, ξ)]
Q(x, ξ) = minimum feasible recourse cost after ξ is observed
```

Here `x` is the vector of first-stage decisions, `X` is its feasible set, and `ξ` is the uncertain state.
The function `Q` must include the feasible response to that state and the cost of that response.
For a finite scenario set, replace the expectation by `sum_s p_s Q(x, ξ_s)`.

If each scenario has a separate copy `x_s` of the initial decision, add equality constraints requiring all `x_s` to match.
In a multistage tree, apply the equalities only among branches sharing the same information history at the action date.
The model may otherwise appear cheaper precisely because it has been allowed to know the future.

**Output:** decision variables, feasible sets, timing, and recourse constraints.

**Branch:** if some scenario makes recourse infeasible, determine whether that scenario represents a physically impossible plan, a missing action, or an allowable violation.
Add an expensive shortage or violation variable only when it represents a real business outcome; an artificial penalty is not a physical emergency supplier.
Report any remaining infeasibility to the decision owner.

### Step 5: Build a consistent economic objective

**Input:** incremental costs and benefits for every action and outcome.

Choose either total profit maximization or total cost minimization with equivalent terms.
In this lesson, purchase expenditure plus a shortage penalty minus salvage proceeds defines cost.
The shortage penalty represents an incremental opportunity cost per unsatisfied unit; it must be calibrated to the specific counterfactual.
If it includes lost contribution, do not also subtract the same lost contribution in a separate revenue term.

Use a common currency and price basis.
Separate sunk costs from costs changed by the decision.
Include disposal, expedited freight, production disruption, contractual penalties, or goodwill effects only if supported and not already included elsewhere.
Place cash constraints in constraints when exceeding the cash limit is not an acceptable trade-off.
Distinguish a cash outlay from an accrual expense and from a model opportunity cost.

If timing matters, express future costs at a common valuation date using an agreed discount rate.
When analyzing a recurring policy, multiply per-period benefits only after accounting for seasonality, changing demand, reoptimization, and operational capacity.

**Output:** a documented cost dictionary with unit, owner, source, timing, range, and treatment of overlap for each coefficient.

**Branch:** if a material penalty cannot be estimated, show a threshold analysis over defensible values instead of presenting an unsupported point estimate as fact.

### Step 6: State constraints and risk preferences

**Input:** operating limits, service agreements, management risk tolerance, and the cost dictionary.

Typical constraints include order bounds, supplier minimums, budget, storage space, and labor availability.
A service target may constrain an expected fill rate, the probability of no stockout, or a worst-case shortfall.
These are different measures and lead to different decisions.
Define the numerator, denominator, horizon, and aggregation level before implementation.

Expected cost alone permits rare high losses if compensated by sufficiently good average outcomes.
If management wants a tail risk measure, choose one explicitly and explain its estimation uncertainty.
Conditional value at risk, for example, averages outcomes in a specified adverse tail under a chosen convention; it is not a substitute for choosing and validating a scenario distribution.
A binding service constraint can also change the meaning of VSS and EVPI, so use the same feasible set for every comparison.

**Output:** a model specification with constraints, objective, risk criterion, and infeasibility handling.

**Branch:** if the desired service level makes the model infeasible, present the conflicting constraints and feasible trade-offs rather than quietly removing the constraint.

### Step 7: Solve the finite scenario problem

**Input:** scenario set, weights, actions, constraints, and objective.

For a small decision space, enumerate feasible quantities and compute weighted cost for each.
For a large linear or mixed-integer model, use an appropriate solver and record termination status, time limit, incumbent objective, bound, gap, and feasibility tolerances.
A feasible incumbent found before a time limit is not necessarily optimal.
The solver gap describes the remaining optimization bound under the given mathematical model, not the error in demand forecasts or economic assumptions.

Inspect selected decisions and scenario outcomes.
Check arithmetic, units, constraint activity, and extreme cases.
Compare against a simple baseline and a hand-calculated miniature instance.
A result that depends on an implausible recourse action is not ready for operational approval.

**Output:** a candidate policy, scenario-level consequences, solver status, and an auditable model version.

**Branch:** if several actions have similar expected costs, compare their service, cash, tail loss, and operational complexity rather than overinterpreting a small model difference.

### Step 8: Use sample average approximation when the expectation is not tractable

**Input:** a sampler for joint uncertain inputs and an actionable feasible set.

Draw N training scenarios independently from the assumed joint distribution when independent sampling is justified.
For each candidate first-stage decision, average its scenario costs over the same N training draws.
Optimize that sample average to obtain a candidate `x_N`.
The procedure is called sample average approximation because a sample average substitutes for the expectation.

For minimization, selecting the best action on the same sample tends to make the optimized in-sample objective optimistic as an estimate of future performance.
That effect is a selection effect: the optimizer can exploit favorable noise in the training scenarios.
It does not mean every individual realization is too optimistic.
Increase N, repeat with independently generated training sets, and compare the stability of both decisions and independent evaluations.
A large N reduces Monte Carlo noise under the sampler; it does not correct a misspecified distribution or data leakage.

For a finite scenario model specified as the business distribution itself, its weighted optimum is exact for that finite model if solved exactly.
For SAA, a sampled finite model approximates a larger assumed distribution.
Do not confuse the two claims.

**Output:** candidate solutions with random seeds, training sample definitions, solver settings, and training objective values labeled as in-sample.

**Branch:** if independent samples produce materially different decisions, investigate sample size, flat objective regions, tail frequency, and model assumptions before committing.
If samples are generated by dependent methods, quantify uncertainty using a method appropriate to the dependence rather than an independent-observation formula.

### Step 9: Compute information benchmarks

**Input:** the same scenario distribution, objective, and operational constraints for all benchmark calculations.

For minimization, solve the recourse problem to obtain `RP`, the optimal expected cost when the first-stage decision is made before uncertainty resolves.
Next, replace uncertain inputs by their means and solve that deterministic expected-value problem.
Fix its first-stage decision and evaluate it over the original distribution; call that expected cost `EEV`.
Then calculate `VSS = EEV - RP`.

For the wait-and-see benchmark, give the decision maker perfect advance knowledge of each realized scenario.
Optimize the first-stage decision separately in each scenario and average those scenario-specific minimum costs.
Call that expected cost `WS`.
Then `EVPI = RP - WS`.
For exact solutions with consistent feasible sets and a feasible expected-value decision, `WS <= RP <= EEV`.
A negative result should trigger a check of sign convention, feasibility, units, and whether the problems actually match.
Approximate solves and sampling noise can create small apparent reversals and should be reported rather than silently clipped.

VSS is a comparison between two decision procedures under the assumed distribution.
EVPI is an upper bound on the value of perfect advance information about the modeled uncertainty under those assumptions.
A practical signal is noisy, may arrive too late, and may cost money; its value is generally lower and must be modeled using the information actually available.
VSS and EVPI answer different counterfactual questions and should not be added as if they were independent savings.

**Output:** `RP`, `EEV`, `WS`, `VSS`, and `EVPI` with explicit units and definitions.

**Branch:** if the mean-value solution is infeasible in some scenario, state how the baseline handles that outcome.
Do not compute a comforting EEV by dropping difficult scenarios.

### Step 10: Evaluate selected policies out of sample

**Input:** fixed candidate policies and an independent evaluation set.

Keep the evaluation set separate from scenario generation used to select and tune policies.
Use the same independent realized scenario for each fixed policy so the difference in cost is paired.
For each test scenario `i`, compute `d_i = cost_baseline_i - cost_candidate_i`.
Positive `d_i` means the candidate is cheaper on that scenario.
The mean of `d_i` estimates expected improvement under the test distribution.

When test scenarios are independent and identically distributed and the usual finite-variance approximation is reasonable, estimate the standard error as the sample standard deviation of the paired differences divided by the square root of the test count.
A normal approximation interval is mean difference plus or minus 1.96 standard errors for a nominal 95% interval.
For a small test sample, severe skew, serial dependence, or rare events, use an appropriate alternative such as a justified bootstrap or time-block method and state the limitations.
No confidence interval corrects a biased scenario generator or a distribution shift.

Record other decision-relevant outputs: probability of a shortage, expected shortage units, tail costs, service level, and cash requirement.
A positive average improvement can coincide with worse performance on a particular disruption.
If the holdout set is repeatedly used to select among policies, it becomes part of model development; reserve another untouched set or use a pre-specified nested evaluation protocol.

**Output:** paired performance estimates, uncertainty, scenario-level diagnostics, and a record of any test-set reuse.

**Branch:** if historical data are scarce, avoid pretending synthetic holdout samples are independent real-world evidence.
They are independent draws conditional on the generator, not independent validation of the generator.

### Step 11: Turn model value into business economics

**Input:** the chosen baseline, estimated outcome differences, deployment costs, and decision horizon.

Start with incremental economic benefit, not the total inventory cost in the model.
Translate cost components to cash only where the mapping is justified.
An opportunity-cost estimate of lost contribution is not necessarily an invoiced expense; a reduction in shortage does not become realized revenue if customers would not have purchased.
Subtract incremental implementation, integration, analyst time, license, maintenance, change-management, and working-capital financing costs as appropriate.
Avoid counting purchase cost twice when comparing operating savings and cash investment.

For benefits `B_t` and incremental costs `K_t` at the end of period `t`, use:

```text
NPV = -K_0 + sum over t=1..T of (B_t - K_t) / (1 + r)^t
```

Here `r` is the discount rate per period and `K_0` is the initial investment.
The rate and period must agree: do not use an annual rate with monthly cash flows without conversion.
Use a conservative deployment ramp and explicit probability of adoption if supported.
Present a break-even benefit and a range of outcomes, not only a point NPV.

**Output:** a cash-flow bridge, break-even calculation, sensitivity analysis, and recommendation.

**Branch:** if the business requires an investment larger than plausible incremental benefit, recommend a simpler policy or a limited decision-support use rather than an expensive platform.

### Step 12: Govern implementation and learning

**Input:** approved decision, owner, operational controls, and evaluation evidence.

Document when the policy runs, what data it reads, who may override it, and how overrides are logged.
Set triggers for review, such as forecast bias, supplier capacity changes, persistent service misses, or cost coefficients outside the approved range.
Keep policy versions and data snapshots so decisions can be reconstructed.
Compare realized outcomes to the committed baseline where a credible comparison is possible, noting that unobserved counterfactual demand and concurrent business changes complicate attribution.

**Output:** a controlled decision workflow with monitoring and an accountable owner.

**Branch:** if input data are late or invalid, use an approved fallback decision process and escalate; do not silently run with stale inputs.

### Sensitivity analysis: shortage penalty

The optimal order depends on the relative costs of shortage and surplus.
Use sensitivity analysis to distinguish a stable recommendation from one driven by a contested coefficient.
Let `p` denote the shortage penalty in USD per unit.
Keep purchase cost at USD 4 per unit, salvage value at USD 1 per unit, and the worked example's demand distribution.

The 100-unit order has expected purchase cost USD 400 and expected salvage credit USD 5.
Its expected shortage is 10 units, so its expected cost is `395 + 10p` USD per week.
The 105-unit order has expected purchase cost USD 420 and expected salvage credit USD 8.75.
Its expected shortage is 8.75 units, so its expected cost is `411.25 + 8.75p` USD per week.
The 105-unit order is cheaper than the 100-unit order when `411.25 + 8.75p < 395 + 10p`.
The break-even penalty is `p = 13` USD per unit.
At that penalty, the two quantities have equal expected cost.

This pairwise comparison does not prove that either quantity is globally optimal.
For a discrete demand distribution and linear costs, optimal quantities often occur at demand breakpoints.
The expected-cost slope between 100 and 140 units is `4 - 0.75 × 1 - 0.25p`.
At `p = 13` USD per unit, this slope is zero, so every quantity from 100 to 140 units has the same expected cost.
For `p > 13`, increasing the order toward 140 units becomes attractive under the remaining assumptions.
For `p < 13`, the slope on this interval is positive.
Check the slopes in the other intervals before naming a global optimum.

| Shortage penalty, USD/unit | Cost at q=100, USD/week | Cost at q=105, USD/week | Pairwise interpretation |
|---:|---:|---:|---|
| 6 | 455.00 | 463.75 | q=100 is cheaper |
| 8 | 475.00 | 481.25 | q=100 is cheaper |
| 10 | 495.00 | 498.75 | Base case |
| 12 | 515.00 | 516.25 | Difference is small |
| 13 | 525.00 | 525.00 | The orders tie |
| 14 | 535.00 | 533.75 | q=105 is cheaper |
| 16 | 555.00 | 551.25 | q=105 is cheaper |

Do not infer that the actual penalty equals USD 13 because it is a mathematical threshold.
Ask finance and commercial owners whether the plausible range crosses that threshold.
If it does, the ranking is sensitive to a contested input.
That is a reason to improve the estimate, present both choices, or apply a separately stated service constraint.
If both quantities are near-optimal, operational simplicity or budget availability may matter more than the calculated difference.

### Sensitivity analysis: demand probabilities

Probabilities also determine the preferred quantity.
Suppose the demand values remain 80, 100, and 140 units, but the probability of the high state is `h`.
Suppose the low-state probability remains 0.25 and the middle-state probability is `0.75 - h`.
The admissible range is `0 <= h <= 0.75`.
With the base costs, the slope of expected cost between 100 and 140 units is `4 - (1-h) × 1 - 10h = 3 - 9h`.
That slope is positive when `h < 1/3` and negative when `h > 1/3`.
At `h = 1/3`, the model is indifferent among quantities on that interval, subject to the other slopes.

| High-demand probability | Middle-demand probability | Slope from 100 to 140, USD/unit | Local implication |
|---:|---:|---:|---|
| 0.10 | 0.65 | 2.10 | Additional units increase expected cost |
| 0.20 | 0.55 | 1.20 | Additional units increase expected cost |
| 0.25 | 0.50 | 0.75 | Base case favors the 100-unit breakpoint |
| 0.30 | 0.45 | 0.30 | The local difference is modest |
| 1/3 | 5/12 | 0.00 | Quantities in this interval tie locally |
| 0.40 | 0.35 | -0.60 | Additional units reduce expected cost |
| 0.50 | 0.25 | -1.50 | Additional units reduce expected cost |

The low-demand probability is held at 0.25 in every row.
The changing middle probability keeps total probability equal to one.
The slope is local to the interval from 100 to 140 units; it is not a substitute for examining other intervals and constraints.
A forecast owner should assess whether the plausible high-state probability straddles one third.
If it does, a point recommendation at either end of the interval may be unstable even if a solver returns an exact answer.

### Sensitivity analysis: service constraints

Service constraints can change the preferred action and the value benchmarks.
Suppose management requires at least 95% of expected demand units to be supplied.
Define the metric as `E[min(q, D)] / E[D]` under the same scenario distribution.
This is a ratio of expectations, not the expected value of scenario-level fill rates.
For this example, expected demand is 105 units per week.
The requirement therefore needs expected fulfilled demand of at least `0.95 × 105 = 99.75` units per week.

For any order between 100 and 140 units, expected fulfilled demand is `0.25 × 80 + 0.50 × 100 + 0.25q = 70 + 0.25q` units.
Solving `70 + 0.25q >= 99.75` gives `q >= 119` units.
The expected cost slope in this interval is positive at the base cost coefficients.
Therefore, if the 95% requirement is the only new constraint and integer orders are allowed, the cheapest feasible order is 119 units.
Its expected fulfilled demand is `70 + 0.25 × 119 = 99.75` units.
Its expected shortage is `105 - 99.75 = 5.25` units.

At `q = 119`, low-demand surplus is 39 units and typical-demand surplus is 19 units.
The expected salvage credit is `0.25 × 39 + 0.50 × 19 = USD 19.25`.
Purchase cost is `4 × 119 = USD 476`.
Expected shortage cost is `10 × 5.25 = USD 52.50`.
Expected total cost is `476 + 52.50 - 19.25 = USD 509.25` per week.
Relative to the unconstrained optimum, the requirement adds `509.25 - 495 = USD 14.25` per week in modeled expected cost.
This difference is the modeled cost of this constraint, not automatically the value customers place on the improvement.

A 95% probability of a stockout-free week is a different requirement.
In the three-state distribution, ordering fewer than 140 units yields a stockout in the high state, which has probability 25%.
Ordering 140 units yields no shortage in any represented state.
Thus the minimum feasible order for this second requirement is 140 units, not 119 units.
The distinction matters for procurement cost, service reporting, and contractual interpretation.

### Selecting a test protocol

Different kinds of test data answer different questions.
Keep the following evidence streams distinct:

| Evaluation data | Main question | Strength | Limitation |
|---|---|---|---|
| Fresh draws from the assumed generator | How does sampling variation affect comparison under the model? | Cheap paired evaluation | Cannot validate the generator |
| Historical holdout periods | How would fixed rules have behaved on observed history? | Uses real recorded outcomes | Demand censoring and structural shifts can bias conclusions |
| Rolling-origin forecasts | How stable is the full prediction-and-decision process through time? | Respects temporal information availability | Requires archived features and careful retraining rules |
| Prospective controlled pilot | Does the operational intervention improve observed outcomes? | Tests execution as well as analytics | Randomization, contamination, and sample size may be difficult |
| Predefined stress cases | Where can the policy fail under adverse assumptions? | Exposes vulnerabilities | Not a frequency estimate without defensible probabilities |

In a rolling-origin design, select a historical cutoff and use only information available at that date.
Fit or update the forecast using data available before the cutoff.
Generate scenarios, select the order, and evaluate against the following period.
Advance the cutoff and repeat according to a pre-specified schedule.
If the actual historical purchase constrained observed sales, the historical outcome may not reveal demand under a larger counterfactual order.
Document the resulting uncertainty instead of claiming a fully observed counterfactual.

In a prospective pilot, define assignment units and a comparable baseline before launch.
For example, allocate comparable items or locations to a decision-support process and a control process.
Check whether shared inventory, common suppliers, or planner behavior causes the groups to affect one another.
Record overrides and simultaneous promotions so that interpretation does not depend on hidden operational changes.
A prospective pilot can still be inconclusive if the number of independent decision opportunities is small.

For paired synthetic evaluation, inspect the distribution of scenario-level differences, not only their mean.
Report the fraction of test scenarios in which the candidate is more expensive than the baseline.
Report shortage units and cash commitment separately from cost.
If few tail scenarios drive most of the difference, explain that dependence and show the number of tail observations.
If the test distribution gives a disruption probability of 1%, a sample of 100 scenarios has only one expected disruption observation.
The actual number may be zero, one, or more; one expected observation is not enough to establish precise tail performance.

### Valuing imperfect information

Perfect advance information is useful as a ceiling, but a real forecast update is neither perfect nor necessarily timely.
Suppose a signal arrives before the order deadline and can be either `high-alert` or `ordinary`.
To value it, specify the conditional probability of each signal under each demand scenario.
Combine those conditional probabilities with the original scenario probabilities to obtain the joint distribution.
Then calculate posterior demand probabilities for each signal using Bayes' rule.
Choose the best feasible order separately for each observed signal, using only information available when ordering.
Average the resulting costs over signal probabilities and compare with RP.

The gross expected value of the signal is `RP - expected cost using the signal`.
Subtract the incremental cost of acquiring and using the signal to calculate net value.
If the signal arrives after the order is irrevocable, its value for that order is zero unless it changes some later recourse action.
If the signal is available only for a subset of products, evaluate the affected decisions rather than applying the benefit portfolio-wide.
If users ignore the signal or cannot execute the recommended change, its realized value can be lower than the modeled gross value.
Under the same model and action set, a signal cannot be worth more than perfect information.
An estimate above EVPI suggests inconsistent scenarios, constraints, cost accounting, or numerical estimation error.

### Decision review before release

A defensible decision package answers the following questions without requiring the reviewer to inspect code:

| Review question | Evidence to present | Responsible role |
|---|---|---|
| What is the deadline? | Timestamp, time zone, and supplier acceptance cutoff | Buyer |
| What is fixed before uncertainty resolves? | Timeline and signed order constraints | Procurement |
| What can be changed later? | Lead times, capacity, and approval rights | Operations |
| Which distribution is used? | Scenario weights, origin, and calibration period | Forecast owner |
| What makes scenarios dependent? | Joint generation method and disruption assumptions | Model owner |
| Which objective is optimized? | Cost dictionary, currency, horizon, and exclusions | Finance |
| What must never be violated? | Contractual and physical constraints | Operations and legal |
| What is the baseline? | Executable current decision rule and its data | Decision owner |
| How strong is the solution? | Solver termination, bound, and scenario feasibility | Analyst |
| How was performance tested? | Fixed-policy paired comparison and holdout provenance | Validation owner |
| What remains uncertain? | Sensitivity ranges and tail-data limitations | Model owner |
| What will be implemented? | Approved quantity, authorization, and fallback | Buyer |
| What triggers review? | Threshold, cadence, accountable owner, and escalation | Decision owner |

Keep the supporting scenario-level records available for audit.
If a reviewer changes the objective or rejects a constraint, solve the revised decision problem rather than presenting the old recommendation with a new label.
If the resulting economics are close, state the materiality limit and avoid presenting the model's final decimal as a business commitment.

## 6. Worked example

### Setting and assumptions

A buyer must place one weekly order for a perishable component before observing demand.
Quantities are measured in units per week and costs in USD per week.
The component costs USD 4 per unit ordered.
Every unsold unit can be salvaged at USD 1 per unit at the end of the week.
Every unit of unmet demand incurs USD 10 of incremental shortage opportunity cost.
There is no fixed order cost, capacity limit, lead-time uncertainty, or partial-week replenishment in this example.
The shortage cost does not also include a separate revenue line.

The forecast distribution is:

| Scenario | Weekly demand, units | Probability | Business interpretation |
|---|---:|---:|---|
| Low | 80 | 0.25 | Weak customer pull |
| Typical | 100 | 0.50 | Normal customer pull |
| High | 140 | 0.25 | Elevated customer pull |

These probabilities are illustrative, not a claim about any company's demand.
Expected demand is `(0.25 × 80) + (0.50 × 100) + (0.25 × 140) = 105` units per week.
The forecast is the stated distribution and its 105-unit mean.
A plan will name an order quantity.
There is no separately imposed service target in the calculation.
The commitment is the order actually placed before demand is known.

For an order `q` and realized demand `D`, define:

```text
cost(q, D) = 4q + 10 max(D - q, 0) - max(q - D, 0)
```

Each term is in USD per week.
The first term is purchase expenditure; the second prices shortage; the last credits salvage proceeds.
This is an illustrative economic objective, not a complete profit-and-loss statement.

### Evaluate candidate orders

| Order q, units | Cost if demand 80, USD | Cost if demand 100, USD | Cost if demand 140, USD | Expected cost, USD/week |
|---:|---:|---:|---:|---:|
| 80 | 320 | 520 | 920 | 570.00 |
| 100 | 380 | 400 | 800 | 495.00 |
| 105 | 395 | 415 | 770 | 498.75 |
| 140 | 500 | 520 | 560 | 525.00 |

For `q = 100`, the low-demand scenario leaves 20 units to salvage.
Its cost is `4 × 100 - 1 × 20 = USD 380`.
The typical scenario costs `4 × 100 = USD 400`.
The high scenario has 40 shortage units and costs `4 × 100 + 10 × 40 = USD 800`.
The weighted cost is `0.25 × 380 + 0.50 × 400 + 0.25 × 800 = USD 495` per week.

For `q = 105`, the low-demand scenario has 25 units salvaged, so cost is `420 - 25 = USD 395`.
The typical scenario has 5 units salvaged, so cost is `420 - 5 = USD 415`.
The high-demand scenario has 35 shortage units, so cost is `420 + 350 = USD 770`.
The weighted cost is `0.25 × 395 + 0.50 × 415 + 0.25 × 770 = USD 498.75` per week.

The stochastic solution is `q = 100` units.
To see why this is globally optimal for continuous nonnegative orders, examine the expected cost slope between demand points.
Below 80 units, increasing the order by one unit reduces shortage in all scenarios; the slope is `4 - 10 = -6` USD per unit.
Between 80 and 100 units, 25% of scenarios add one surplus unit while 75% avoid one shortage unit: the slope is `4 - 0.25 × 1 - 0.75 × 10 = -3.75` USD per unit.
Between 100 and 140 units, the slope is `4 - 0.75 × 1 - 0.25 × 10 = 0.75` USD per unit.
Above 140 units, the slope is `4 - 1 = 3` USD per unit.
The slope changes from negative to positive at 100 units.
Thus `RP = USD 495` per week under this model.

### Expected-value decision and VSS

Replacing demand by its mean creates a deterministic problem with demand 105 units.
Because ordering up to that demand avoids a shortage at a lower incremental cost and ordering beyond it creates a net surplus cost, the deterministic optimum is `q = 105` units.
The deterministic objective at that mean is USD 420, but this is not the expected cost of ordering 105 units.
The correct evaluation over the three original scenarios gives `EEV = USD 498.75` per week.

Therefore:

```text
VSS = EEV - RP = 498.75 - 495.00 = USD 3.75 per week
```

The stochastic solution saves USD 3.75 per week in modeled expected cost relative to ordering the mean demand.
This small difference is not the total cost of uncertainty, and it is not a promise that the stochastic order will cost less in every week.
At demand 140 units, the 105-unit order costs USD 770 while the 100-unit order costs USD 800.
At demand 80 or 100 units, the 100-unit order costs USD 15 less than the 105-unit order.

### Perfect information and EVPI

With perfect advance knowledge, order exactly the demand in each scenario.
The three scenario costs are USD 320, USD 400, and USD 560.
Their weighted average is `0.25 × 320 + 0.50 × 400 + 0.25 × 560 = USD 420` per week.
This is the wait-and-see value, `WS`.

```text
EVPI = RP - WS = 495 - 420 = USD 75 per week
```

The perfect-information benchmark varies the initial order with future demand and therefore is not implementable without perfect advance information.
USD 75 per week is the maximum gross modeled benefit of perfect information for this particular decision and distribution.
It is not the expected benefit of a particular demand-sensing tool.
The tool's signal quality, timing, adoption, and cost must be modeled separately.

### Compare service and liquidity

The 100-unit policy experiences shortage only in the high scenario: probability 0.25 and expected shortage `0.25 × 40 = 10` units per week.
The 105-unit policy also has shortage probability 0.25, with expected shortage `0.25 × 35 = 8.75` units per week.
The ratios of expected units sold to expected demand are `95 / 105 ≈ 90.48%` and `96.25 / 105 ≈ 91.67%`, respectively.
These ratios are not the average of each scenario's fill-rate percentage.
They also are not probabilities of a stockout-free week.
A service requirement could therefore favor a different order even when expected cost is higher.

The initial purchase cash outlay is USD 400 for `q = 100` and USD 420 for `q = 105`.
Salvage arrives later in this example; it should not be netted against an immediate cash limit without approval.
A supplier minimum or shelf-space limit could also change the feasible optimum.

### Illustrative investment decision

Suppose the baseline is the expected-value order and the model's USD 3.75 weekly benefit applies for 50 comparable weeks per year.
Annual gross expected benefit is `3.75 × 50 = USD 187.50` before deployment costs and financing.
If a workflow costs USD 1,000 initially and USD 50 at each year end for three years, and the annual discount rate is 10%, then:

```text
NPV = -1000 + 137.50/1.10 + 137.50/1.10^2 + 137.50/1.10^3
    ≈ -USD 658.06
```

This illustrative standalone investment is unattractive on these assumptions.
The break-even gross annual benefit is approximately USD 452.11 for three equal year-end benefits at 10%, including USD 50 annual operating cost.
A shared planning capability serving many independent decisions might have different economics; it cannot be justified by simply multiplying this example without evidence.
The model may still be useful as an inexpensive decision aid, but a costly platform is not supported by this single-item estimate.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
It enumerates integer order quantities, computes the finite-distribution benchmarks, builds SAA decisions from independent training samples, and compares two fixed policies on an independent synthetic test sample.
The finite scenario distribution and costs are exactly those in the worked example.
Synthetic test draws assess performance conditional on that assumed distribution; they do not validate the distribution against real operations.

The random generator is seeded for reproducibility.
The exact printed training candidates and test statistics depend on Python's pseudo-random implementation and sample realization.
The benchmark lines must show order 100, `RP = 495.00`, expected-value order 105, `EEV = 498.75`, `VSS = 3.75`, `WS = 420.00`, and `EVPI = 75.00`, all in the units stated above.
The sampled estimates should be interpreted as estimates, not exact expected values.

```python
from math import sqrt
from random import Random
from statistics import mean, stdev

# Units: demand and order in units/week; all costs in USD/week.
SCENARIOS = [(80, 0.25), (100, 0.50), (140, 0.25)]
ORDERS = range(0, 201)  # Integer units; covers every relevant breakpoint.


def cost(order, demand):
    purchase = 4.0 * order
    shortage = 10.0 * max(demand - order, 0)
    salvage = 1.0 * max(order - demand, 0)
    return purchase + shortage - salvage


def expected_cost(order):
    return sum(probability * cost(order, demand)
               for demand, probability in SCENARIOS)


def best_order(cost_function):
    # Stable tie-breaking: choose the smaller order if costs are equal.
    return min(ORDERS, key=lambda order: (cost_function(order), order))


def sample_demands(rng, count):
    return rng.choices(
        [demand for demand, _ in SCENARIOS],
        weights=[probability for _, probability in SCENARIOS],
        k=count,
    )


def saa_order(sample):
    return best_order(lambda order: mean(cost(order, d) for d in sample))


# Exact finite-distribution calculations.
stochastic_order = best_order(expected_cost)
rp = expected_cost(stochastic_order)
mean_demand = sum(demand * probability for demand, probability in SCENARIOS)
ev_order = best_order(lambda order: cost(order, mean_demand))
eev = expected_cost(ev_order)
ws = sum(probability * min(cost(order, demand) for order in ORDERS)
         for demand, probability in SCENARIOS)
print(f"Stochastic order: {stochastic_order} units; RP: ${rp:.2f}/week")
print(f"Expected-value order: {ev_order} units; EEV: ${eev:.2f}/week")
print(f"VSS: ${eev - rp:.2f}/week; WS: ${ws:.2f}/week; "
      f"EVPI: ${rp - ws:.2f}/week")

# Independent training sets: inspect solution stability, not a test result.
training_rng = Random(7301)
training_candidates = [saa_order(sample_demands(training_rng, 200))
                       for _ in range(5)]
print("SAA training orders, units:", training_candidates)

# Fix the policies before examining this independent test set.
# For illustration, the candidate is the first SAA policy.
candidate = training_candidates[0]
test_rng = Random(9117)
test_demands = sample_demands(test_rng, 10000)
differences = [cost(ev_order, d) - cost(candidate, d) for d in test_demands]
average = mean(differences)
se = stdev(differences) / sqrt(len(differences))
print(f"Fixed candidate: {candidate} units; baseline: {ev_order} units")
print(f"Paired saving: ${average:.2f}/week; approximate 95% interval: "
      f"[${average - 1.96 * se:.2f}, ${average + 1.96 * se:.2f}]")
```

The training RNG and test RNG use different seeds and are used for separate samples.
In a real study, record the generator, seed, sampling method, model version, and dataset snapshot.
Separate seeds are an implementation convenience, not proof that a flawed generator matches reality.
The order enumeration includes 0 through 200 units; that range is part of the modeled feasible set.
For a real business, use actual contractual bounds and a solver appropriate to the model.

The interval uses independent test draws and the paired-difference standard error.
Its nominal 95% interpretation is conditional on the simulation assumptions and the normal approximation.
The candidate was fixed before the test outcomes were inspected.
If the candidate is then changed because of this test, a fresh holdout is needed for a clean final evaluation.

### Reading the result

The exact benchmark is more informative than the random output when the complete three-point distribution is known.
SAA is shown to illustrate the workflow used when the expectation is approximated by sampled scenarios.
Different training sets can yield different candidate quantities because scenario frequencies fluctuate.
The test comparison tells you how the selected candidate compares with the mean-demand order under independent draws from the same assumed distribution.
It does not establish that either policy will achieve that saving in an actual future week.

## 8. Data, standards, and regulatory context

### Data needed for the decision

| Data element | Purpose | Check before use | Typical owner |
|---|---|---|---|
| Historical demand or orders | Calibrate future demand | Identify cancellations, stockout censoring, promotions, and missing periods | Commercial analytics |
| Sales and available inventory | Separate sales from potential demand | Reconcile dates, units, returns, and availability | Sales operations |
| Supplier lead time | Establish decision timing and feasible response | Compare promised with actual receipt dates | Procurement |
| Supplier capacity | Bound feasible first-stage and recourse quantities | Confirm allocation rules and effective dates | Supplier management |
| Purchase price | Price the order | Include relevant tiers, currency, taxes, and freight treatment | Finance and procurement |
| Salvage or disposal | Price surplus | Verify actual recoverability and timing | Inventory control |
| Shortage consequences | Value unmet demand | Avoid double counting and distinguish lost from delayed demand | Finance and commercial |
| Customer service terms | Define binding obligations and metrics | Check contract scope, exceptions, and measurement | Legal and account teams |
| Storage and handling capacity | Enforce operational feasibility | Check peak occupancy and product compatibility | Operations |
| Scenario versions | Reproduce decisions | Record source data, model version, weights, and seed | Analytics |

The appropriate demand history is determined by the decision.
Shipment history can reflect constrained supply rather than underlying demand.
A zero recorded sale during a stockout is not evidence that demand was zero.
Separate known promotions, product launches, and discontinuations from stable recurring demand where possible.
Document uncertainty in any demand reconstruction; invented point demand can create false precision.

### Identifiers and data exchange

GS1 identification and data standards can support consistent product and location identifiers across trading partners.
For example, a Global Trade Item Number (GTIN) identifies a trade item, while a Global Location Number (GLN) can identify a location or party.
Use the official GS1 General Specifications for the exact identifier and application rules relevant to the implementation.
These identifiers help align records; they do not establish a demand distribution, determine scenario probabilities, or certify an optimized order.
Product hierarchies, unit conversions, and effective dates still need local governance.

Electronic data interchange (EDI) or an application programming interface may transmit orders and confirmations.
The transaction format is not the same thing as a contractual guarantee of capacity or delivery.
Reconcile acknowledgments, cancellations, and actual receipts before estimating supplier reliability.

### Regulatory and contractual boundaries

This lesson does not assert a universal legal requirement to use stochastic optimization or a particular confidence level.
Applicable privacy, competition, product-safety, financial-reporting, and sector-specific rules depend on jurisdiction and data use.
Personal data in customer orders may require lawful processing, access controls, and retention limits under applicable law.
For example, the EU General Data Protection Regulation sets obligations for processing personal data within its scope; counsel should determine whether and how it applies to a particular implementation.
A model's expected-value objective does not override applicable law, contractual obligations, or product handling requirements.

A service-level clause may be legally or commercially binding according to the relevant contract.
The model must use the contract's actual definition, measurement period, exceptions, and remedies rather than inventing a generic fill-rate proxy.
An internal target, by contrast, is an organizational choice unless incorporated into an obligation.

### Reproducibility and access

Version inputs, scenario generation code, constraint settings, and outputs together.
Restrict access to supplier pricing and customer-sensitive information according to policy and contract.
Keep enough metadata to reproduce a decision without exposing more personal or commercially sensitive data than necessary.
Record whether probabilities came from historical estimation, expert judgment, or a deliberately altered stress distribution.

## 9. Industry practice and case evidence

### Evidence interpretation

The example in this topic is synthetic and should not be attributed to a company.
Industry examples can demonstrate the relevance of uncertainty and analytical methods without proving the return on a particular implementation.
A company press release, academic case study, vendor description, and independently audited outcome have different evidentiary strength.

### Hewlett-Packard and product variety

Lee and Billington's published Hewlett-Packard case describes supply-chain design and inventory issues associated with product variety and delayed differentiation.
The operational lesson is that the timing of product configuration and inventory positioning changes exposure to uncertain demand.
Postponing differentiation can pool some demand uncertainty, but it may add handling, lead time, or capacity requirements.
The published case should be consulted for its actual setting and reported results; this topic does not assign the synthetic weekly saving to Hewlett-Packard.

**Inference for this lesson:** a stochastic model can compare a common-component commitment with product-specific commitments if the scenario model includes demand correlation, conversion capacity, and postponement costs.
That inference is a modeling application, not a claim that the case used the exact formulation shown here.

### Zara and inventory decisions

Caro and Gallien's published work on inventory management at Zara studies allocation decisions in a fast-fashion setting.
The case illustrates why a forecast, inventory constraints, and allocation timing matter together.
Consult the paper for the specific formulation, implementation, and reported evidence.
Do not generalize its results to a different product, lead time, or service contract without testing.

**Inference for this lesson:** a two-stage allocation model could choose an initial inventory position before local demand is known, then model feasible transfers after a specified information update.
A transfer assumed after final demand is known would be too optimistic if the actual transfer deadline occurs earlier.

### Applying the evidence to a new business

Start from the decision's physical timing, not from a vendor's model category.
Assess whether demand information arrives early enough to change an order.
Check whether the organization can execute the modeled response at the required cost and speed.
Review documented results in their original context and separate reported outcomes from analyst inference.
Use a pilot with an agreed baseline where feasible, while recognizing that seasonality, demand shifts, and simultaneous operational changes complicate attribution.

### Practical adoption patterns

In planning practice, organizations often compare a forecast-based baseline with a distribution-aware plan and operational overrides.
The initial deployment may be a recommendation reviewed by a buyer rather than an automatic purchase order.
An override can encode information missing from the model, but frequent unexplained overrides undermine reproducibility.
Track the reason, approver, and timing of each override.
A model whose assumptions remain unowned may degrade even when the optimization code continues to run successfully.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Future information leakage | Different initial orders by scenario | Nonanticipativity omitted | Unimplementable apparent savings | Enforce common first-stage actions and audit the timeline |
| Censored demand treated as demand | Many low sales during stockouts | Lost customer requests are unobserved | Under-ordering and understated shortage | Reconstruct or bound demand and test sensitivity |
| Broken dependence | Disruptions coincide with implausibly cheap freight | Marginals sampled independently | Understated joint loss | Generate joint scenarios and inspect dependence |
| Tail events absent | No disruption in a small sample | Rare event frequency too low | Fragile chosen policy | Increase relevant samples and perform separately labeled stresses |
| Invalid weights | Probabilities do not sum to one | Data assembly or normalization error | Misstated expected cost | Validate weights and preserve weighting provenance |
| Unsupported penalties | One undocumented shortage value drives the result | A guess substitutes for economics | Incorrect trade-off | Obtain ranges and calculate break-even thresholds |
| Double counting | Both lost margin and lost revenue are charged | Cost dictionary overlaps | Inflated benefit | Reconcile the objective with finance |
| Unrealistic recourse | Emergency supply exceeds contractual capacity | Missing capacity or lead time | False resilience | Model physical and contractual bounds |
| Hidden infeasibility | Solver adds huge penalties in many cases | Violation is treated as an actual action | Unexecutable plan | Report violations and obtain an operational response |
| In-sample optimism | Reported saving disappears on holdout | Selection exploits sample noise | Overstated performance | Evaluate fixed candidates independently |
| Test-set reuse | Many candidate changes after seeing test results | Holdout becomes a tuning set | Biased final estimate | Reserve a fresh final set or use nested evaluation |
| Incorrect VSS | Deterministic objective is called EEV | Mean-demand cost is not scenario evaluation | Misleading value claim | Fix the mean-value decision and evaluate all scenarios |
| Incorrect EVPI | Scenario-specific first-stage actions called a policy | Perfect foresight is mistaken for a signal | Inflated investment case | Label wait-and-see as an unattainable benchmark |
| Solver gap confused with forecast error | Small optimization gap used as validation | Mathematical and statistical uncertainty conflated | False confidence | Report solver, sampling, and model risk separately |
| Misdefined service | Service percentage has no denominator | Unit fill and stockout probability mixed | Contract breach or wrong trade-off | Specify metric, aggregation, and horizon |
| Distribution shift | Persistent forecast bias after launch | Future behavior differs from training | Declining policy performance | Monitor, review, and update assumptions |
| False precision | Many decimals from uncertain inputs | Numerical precision mistaken for evidence | Overconfident decision | Report ranges and materiality thresholds |
| Cash constraint omitted | Recommended order exceeds budget | Economic cost and cash timing differ | Plan cannot be funded | Add period-specific cash constraints |
| Misleading ROI | Gross expected cost improvement called cash saving | Opportunity cost and investment mixed | Poor capital allocation | Build a documented cash-flow bridge |
| Silent override | Planner frequently changes recommendations without reason | Model or data mismatch remains hidden | No learning and unclear accountability | Log overrides and review patterns |

## 11. Assumptions and limits

| Assumption | Why it is needed here | What breaks it | How to test or respond |
|---|---|---|---|
| The scenario distribution represents future demand | Defines the expected-cost objective | Market change or biased historical data | Backtest distributional forecasts and stress alternatives |
| The three states are mutually exclusive and exhaustive | Allows their probabilities to sum to one | Other demand levels occur | Replace with a richer distribution or explain approximation |
| Demand is independent of the order | Lets the model treat demand as exogenous | Availability or price changes customer demand | Estimate the causal response or model demand conditional on decisions |
| One order precedes demand | Defines a common first stage | Replenishment can occur during the week | Model the actual decision and observation sequence |
| All surplus can be salvaged at USD 1 per unit | Gives surplus a known linear value | Limited salvage capacity or disposal cost | Validate contracts and test alternative recovery values |
| Shortage costs USD 10 per unit | Quantifies the lost-demand consequence | Different customers, penalties, or backorders | Segment costs and run sensitivity analysis |
| Purchase cost is USD 4 per unit | Makes cost linear in quantity | Quantity discounts or fixed fees | Add actual price tiers and fixed costs |
| No service or cash constraint binds | Keeps the example focused on expected cost | A contract or funding limit applies | Add the appropriate constraint and recompute benchmarks |
| Integer orders 0–200 cover relevant choices | Makes enumeration finite | Supplier minimums or larger possible orders | Use the true feasible set and appropriate solver |
| Test scenarios are independent draws | Supports the stated standard error | Serial or common-shock dependence | Use time-aware or block-based evaluation |
| The generator is valid | Gives synthetic test results relevance | Misspecified tail, trend, or dependence | Validate separately against data and expert evidence |
| Cost is a suitable risk criterion | Aligns optimization with the stated choice | Tail loss or equity concerns dominate | Add a justified constraint or alternative risk criterion |
| No discounting within a week | Simplifies the weekly objective | Material intra-period financing or long lead times | Use dated cash flows and a matching discount rate |
| Perfect information arrives before commitment | Defines the EVPI counterfactual | Actual signal is noisy or late | Model the real signal and its timing explicitly |

A precise optimum can be economically immaterial when nearby quantities have similar costs.
The USD 3.75 weekly VSS is small relative to the cost coefficients and may be overwhelmed by estimation error.
Sensitivity analysis should therefore report ranges and decision stability, not only a single optimum.
The synthetic example does not establish empirical service, ROI, or forecast accuracy for any organization.

## 12. Exercises

### Exercise 1 — Map the information timeline

A buyer signs a supplier capacity reservation on Monday.
A preliminary demand update arrives Wednesday, and final demand is known Friday.
An expedited order can be placed on Wednesday but must be decided before Friday.

**Deliverable:** draw a timeline and identify the information available for each action.

**Acceptance criteria:** the Monday reservation is common across all scenarios; Wednesday actions agree for scenarios sharing the same Wednesday information; no action depends on Friday demand before Friday.

**Time:** 15 minutes.

### Exercise 2 — Build and critique scenarios

Use demand states 80, 100, and 140 units with probabilities 0.25, 0.50, and 0.25.
Suppose a disruption simultaneously raises freight cost and reduces supplier availability.

**Deliverable:** propose three joint scenario records with units and probabilities, and name one missing-data risk.

**Acceptance criteria:** weights are nonnegative and sum to one; disruption variables occur together in at least one coherent scenario; the risk is linked to a check or mitigation.

**Time:** 20 minutes.

### Exercise 3 — Calculate expected cost

Using the worked-example cost function, calculate the cost of ordering 100 units in each demand scenario and its expected cost.

**Deliverable:** show the three scenario costs and the weighted sum in USD per week.

**Acceptance criteria:** use the correct shortage and salvage terms; obtain USD 380, USD 400, USD 800, and USD 495.

**Time:** 15 minutes.

### Exercise 4 — Compute VSS and EVPI

Use the same assumptions and the mean-demand order of 105 units.

**Deliverable:** calculate EEV, RP, WS, VSS, and EVPI, with a sentence interpreting each value-of-information measure.

**Acceptance criteria:** obtain USD 498.75, USD 495, USD 420, USD 3.75, and USD 75 per week, respectively; distinguish the expected-value decision from perfect foresight.

**Time:** 25 minutes.

### Exercise 5 — Design an SAA experiment

You have a sampler for weekly demand and freight cost and a solver with a 2% reported relative gap.
You can run five independent training sets and one reserved test set.

**Deliverable:** specify the training, candidate-selection, and test protocol, including which data remain untouched until the final evaluation.

**Acceptance criteria:** candidate policies are fixed before test inspection; competing policies are evaluated on matched test scenarios; the solver gap is distinguished from sampling and model error.

**Time:** 25 minutes.

### Exercise 6 — Compute paired uncertainty

A fixed policy's five test-scenario savings relative to the baseline are 2, 4, 6, 8, and 10 USD per week.
Assume independent draws and use the normal approximation described in this topic.

**Deliverable:** calculate the mean saving, sample standard deviation, standard error, and approximate 95% interval.

**Acceptance criteria:** show the sample variance denominator, the square-root calculation, and the interval; explain why five observations do not establish a reliable normal approximation.

**Time:** 20 minutes.

### Exercise 7 — Build the investment bridge

Use a gross modeled saving of USD 3.75 per week for 50 weeks per year, an initial cost of USD 1,000, annual operating cost of USD 50, three year-end benefit periods, and a 10% annual discount rate.

**Deliverable:** calculate annual net benefit, NPV, and break-even gross annual benefit; recommend whether this one decision supports the investment.

**Acceptance criteria:** match cash-flow timing to the annual discount rate; state that modeled opportunity cost is not automatically realized cash; show the negative NPV.

**Time:** 25 minutes.

### Exercise 8 — Diagnose a proposal

A consultant reports a 0.1% solver gap, a 12% training-sample saving, and zero stockouts in ten synthetic test weeks.
The underlying demand generator was fitted to sales during a six-month period with several stockouts.
The model allows emergency supply to arrive instantly without a capacity limit.

**Deliverable:** write a one-page decision memo with at least four risks, evidence needed, and a release recommendation.

**Acceptance criteria:** address censored demand, recourse feasibility, sampling uncertainty, and the meaning of solver gap; name an owner and a monitoring trigger; do not claim that the test establishes a 100% service rate.

**Time:** 30 minutes.

## 13. Self-check answers

### 1. Timeline

The Monday reservation is a single first-stage decision across all branches.
The Wednesday expedited order may depend on the Wednesday update but not on final Friday demand.
Any two scenarios with the same information observed by Wednesday must share the same Wednesday action.
The Friday outcome can determine accounting, shortage, and salvage after demand is known.
A diagram with separate Monday decisions for each Friday outcome fails the information test.

### 2. Scenario design rubric

A valid answer may use a low-demand, normal-demand, and disruption scenario, provided the joint values and weights are explicitly stated.
The example does not prescribe actual freight or availability estimates; these need evidence.
A plausible disruption record couples lower availability with higher freight cost rather than sampling them independently by default.
A strong answer explains whether disruption changes demand and how historical records or expert elicitation support the joint assumption.
A missing-data risk is stockout-censored sales; inspect availability records and estimate or bound unmet demand rather than treating sales as demand.

### 3. Scenario costs

At demand 80, 20 units are salvaged: `400 - 20 = USD 380`.
At demand 100, there is neither surplus nor shortage: `USD 400`.
At demand 140, 40 units are short: `400 + 400 = USD 800`.
Expected cost is `0.25 × 380 + 0.50 × 400 + 0.25 × 800 = USD 495` per week.

### 4. VSS and EVPI

The 105-unit mean-demand decision has costs USD 395, USD 415, and USD 770.
Its expected cost is `EEV = USD 498.75` per week.
The stochastic decision has `RP = USD 495` per week.
Thus `VSS = 498.75 - 495 = USD 3.75` per week.
The scenario-specific perfect-foresight orders cost USD 320, USD 400, and USD 560.
Their weighted average is `WS = USD 420` per week, so `EVPI = 495 - 420 = USD 75` per week.
VSS compares two implementable decision procedures under the assumed distribution.
EVPI compares an implementable stochastic decision with a hypothetical decision made after perfect advance revelation.

### 5. SAA protocol rubric

Generate independent training sets under the stated sampler and solve each with recorded settings and gaps.
Select a candidate using a pre-specified rule that does not inspect the final test outcomes.
If a separate validation sample is used to choose among candidates, it is not the final test sample.
Evaluate the fixed candidate and baseline on identical independent test scenarios and summarize paired differences.
Report the solver gap as an optimization bound for each sampled model, not as a confidence interval or validation of the demand generator.
A strong answer also states that synthetic test data do not establish real-world distributional validity.

### 6. Paired uncertainty

The mean is `(2 + 4 + 6 + 8 + 10) / 5 = USD 6` per week.
Squared deviations sum to `16 + 4 + 0 + 4 + 16 = 40` USD² per week².
The sample variance is `40 / (5 - 1) = 10` USD² per week².
The sample standard deviation is `sqrt(10) ≈ USD 3.162` per week.
The standard error is `sqrt(10) / sqrt(5) = sqrt(2) ≈ USD 1.414` per week.
The approximate normal interval is `6 ± 1.96 × 1.414`, or approximately `[USD 3.23, USD 8.77]` per week.
With only five observations the normal approximation may be poor, especially for skewed or heavy-tailed savings.
The interval is a calculation under assumptions, not proof of out-of-sample business value.

### 7. Investment bridge

Gross annual modeled benefit is `3.75 × 50 = USD 187.50`.
Annual benefit after operating cost is `187.50 - 50 = USD 137.50`.
The three-year NPV at 10% is approximately negative USD 658.06.
The three-year annuity factor is `1/1.1 + 1/1.1² + 1/1.1³ ≈ 2.48685`.
Break-even net annual benefit is `1000 / 2.48685 ≈ USD 402.11`.
Adding the annual operating cost gives a break-even gross benefit of approximately USD 452.11 per year.
On these assumptions, this one decision does not support the investment.
The estimate still requires a bridge from modeled opportunity costs to realized financial outcomes.

### 8. Proposal rubric

A strong memo rejects automatic release pending evidence rather than treating a small solver gap as proof of real-world performance.
It identifies stockout censoring, obtains availability data, and bounds or reconstructs demand.
It checks whether emergency supply can actually arrive on time and at the assumed capacity and cost.
It explains that ten synthetic weeks provide weak tail evidence and are conditional on the generator.
It requires independent evaluation, scenario sensitivity, and an accountable business owner.
A useful trigger might be persistent forecast bias or a service breach, with a named owner authorized to review and suspend the policy.

## 14. Interview questions

| Question | Strong answer contains | Weak answer reveals |
|---|---|---|
| What makes a decision first-stage? | It is committed before the relevant uncertainty is observed. | It is simply the first variable in a spreadsheet. |
| What is nonanticipativity? | Actions sharing an information history must agree. | All scenario decisions can use their final outcomes. |
| Why optimize over scenarios rather than the mean? | Nonlinear or asymmetric consequences can change the preferred decision. | The mean is always the worst case. |
| What is recourse? | A feasible later action conditional on information actually available. | Unlimited emergency supply is assumed. |
| What does SAA approximate? | The expectation under an assumed distribution by a sample average. | It proves the distribution is correct. |
| Why is training performance optimistic? | Selection can exploit favorable sample noise. | Every individual sample must overstate performance. |
| How do you compute VSS for minimization? | Evaluate the mean-value decision across scenarios and subtract RP from EEV. | Subtract the deterministic mean-input objective from RP. |
| What does EVPI bound? | Gross value of perfect advance information under the specified model. | Guaranteed value of an imperfect forecast system. |
| Why pair out-of-sample comparisons? | Both policies face the same outcomes, isolating their cost difference. | Independent scenarios automatically give a paired estimate. |
| What does a solver gap measure? | An optimization bound conditional on the mathematical model. | The accuracy of the demand forecast. |
| How would you represent a service obligation? | Specify its actual measure, horizon, aggregation, and contractual constraints. | Use any generic service percentage. |
| When would you reject the investment? | When incremental risk-adjusted benefits do not cover adoption and operating costs or the plan is infeasible. | Whenever a more sophisticated model exists. |

## 15. Key terms

- [Baseline](#step-1-frame-the-business-choice): the comparator whose outcomes define incremental benefit.
- [Commitment](#forecast-plan-target-and-commitment): an action or obligation that is difficult or costly to reverse.
- [Complete recourse](#step-4-define-decisions-and-recourse): feasibility of later response for every feasible initial decision and represented state.
- [Conditional value at risk](#step-6-state-constraints-and-risk-preferences): an explicitly defined adverse-tail loss measure.
- [Decision](#4-definitions-and-vocabulary): a choice under the decision maker's control.
- [Distribution shift](#11-assumptions-and-limits): a difference between future and modeled data-generating behavior.
- [EEV](#expected-value-decision-and-vss): expected cost of the expected-value decision over the original distribution.
- [EVPI](#perfect-information-and-evpi): expected cost reduction from perfect advance information, under the model.
- [Expected cost](#3-the-core-mental-model): probability-weighted cost across outcomes.
- [Expected-value decision](#expected-value-decision-and-vss): the decision optimized with uncertain inputs replaced by their means.
- [First stage](#step-4-define-decisions-and-recourse): actions chosen before the represented uncertainty is observed.
- [Forecast](#forecast-plan-target-and-commitment): an estimate of future demand and uncertainty.
- [Holdout](#step-10-evaluate-selected-policies-out-of-sample): data or scenarios not used to select or tune a policy.
- [Joint scenario](#step-3-define-uncertain-quantities-and-scenarios): a coherent combined realization of uncertain inputs.
- [Nonanticipativity](#step-2-draw-the-information-timeline): consistency of actions across scenarios with identical known histories.
- [Opportunity cost](#step-5-build-a-consistent-economic-objective): value forgone by choosing an action or experiencing an outcome.
- [Out-of-sample evaluation](#step-10-evaluate-selected-policies-out-of-sample): evaluation of a fixed policy on independent data or scenarios.
- [Paired difference](#step-10-evaluate-selected-policies-out-of-sample): the difference between policies on the same test outcome.
- [Plan](#forecast-plan-target-and-commitment): intended action conditional on available information.
- [Recourse](#step-4-define-decisions-and-recourse): feasible response after an observation.
- [RP](#step-9-compute-information-benchmarks): optimal expected cost of the stochastic recourse problem.
- [SAA](#step-8-use-sample-average-approximation-when-the-expectation-is-not-tractable): sample average approximation of an expected objective.
- [Scenario probability](#step-3-define-uncertain-quantities-and-scenarios): the assumed weight of a possible state.
- [Service target](#step-6-state-constraints-and-risk-preferences): an explicitly defined desired performance level.
- [Solver gap](#step-7-solve-the-finite-scenario-problem): a bound on remaining optimization error under the stated model.
- [Stress test](#step-3-define-uncertain-quantities-and-scenarios): evaluation under selected adverse assumptions.
- [VSS](#expected-value-decision-and-vss): EEV minus RP for a cost-minimization model.
- [Wait-and-see value](#perfect-information-and-evpi): expected cost with perfect advance scenario information.

## 16. Sources

1. John R. Birge and François Louveaux, *Introduction to Stochastic Programming*, 2nd ed., Springer, 2011. A foundational treatment of recourse, scenario models, and information values. [Publisher](https://link.springer.com/book/10.1007/978-1-4614-0237-4).
2. Alexander Shapiro, Darinka Dentcheva, and Andrzej Ruszczyński, *Lectures on Stochastic Programming: Modeling and Theory*, 3rd ed., SIAM, 2021. Covers stochastic programming and sample-based methods. [Publisher](https://epubs.siam.org/doi/book/10.1137/1.9781611976595).
3. Anton J. Kleywegt, Alexander Shapiro, and Tito Homem-de-Mello, “The Sample Average Approximation Method for Stochastic Discrete Optimization,” *SIAM Journal on Optimization*, 12(2), 2002, pp. 479–502. [DOI](https://doi.org/10.1137/S1052623499363220).
4. Hau L. Lee and Corey Billington, “Managing Supply Chain Inventory: Pitfalls and Opportunities,” *Sloan Management Review*, 33(3), 1992, pp. 65–73. Industry discussion of supply-chain inventory and Hewlett-Packard experience. [MIT Sloan Management Review](https://sloanreview.mit.edu/article/managing-supply-chain-inventory-pitfalls-and-opportunities/).
5. Felipe Caro and Jérémie Gallien, “Inventory Management of a Fast-Fashion Retail Network,” *Operations Research*, 58(2), 2010, pp. 257–273. [DOI](https://doi.org/10.1287/opre.1090.0698).
6. GS1, *GS1 General Specifications*. Official identifier and application guidance; the version applicable to a deployment should be checked. [GS1 standards](https://www.gs1.org/standards/barcodes-epcrfid-id-keys/gs1-general-specifications).
7. European Parliament and Council, Regulation (EU) 2016/679, General Data Protection Regulation. Applicability depends on the processing and jurisdiction. [EUR-Lex](https://eur-lex.europa.eu/eli/reg/2016/679/oj).
8. Python Software Foundation, *Python Standard Library*: [`random`](https://docs.python.org/3/library/random.html) and [`statistics`](https://docs.python.org/3/library/statistics.html). Reference for the example's sampling and sample standard deviation.

## 17. Links

- [Module 20 index](_index.md)
- [Module 20 survey and lab sections](../../../course/20_digital_twins_simulation_optimization_uncertainty.md)
- [Input modeling](03_input-modeling.md)
- [Validation](04_validation.md)
- [Discrete-event simulation](02_discrete-event-simulation.md)
- [Twin, shadow, and simulation](01_twin-shadow-simulation.md)
