# Simulation-Optimization

> Module 20 — Digital Twins, Simulation, and Optimization under Uncertainty  
> Topic 9 of 10  
> Estimated study time: 4 hours  
> Prerequisites: simulation, input modelling, validation, experiment design, and basic optimization  
> Version 2026.5

## 1. Why this topic exists

A distribution center must choose staffing rules before the next peak season.
The operations manager cannot test hundreds of rules on live customer orders.
A static spreadsheet does not represent queues, breaks, absenteeism, or the interaction of several order waves.
A validated simulation can represent those mechanisms, but it does not automatically identify a good policy.

Simulation-optimization combines a simulator with a search procedure.
The simulator estimates consequences of a proposed decision under uncertain inputs.
The search procedure chooses which decisions to evaluate next.
The resulting recommendation is conditional on the model, input distributions, budget, and decision criteria.
It is not a guarantee that the best possible real-world policy has been found.

The expensive error is often not choosing a slightly suboptimal mean cost.
It is committing to a policy that looks cheap because of statistical noise, violates a service commitment, or cannot be operated.
A manager needs the estimated savings, service risk, uncertainty, implementation effort, and a rollback condition together.
A planner also needs to know whether the difference is large enough to justify changing a stable process.

A simulation run is an experiment, not a crystal ball.
Two runs of the same policy can produce different outcomes because demand, arrivals, processing times, and failures differ.
A search over many policies tends to select one that benefited from favorable noise.
Independent evaluation after selection is therefore a central part of the method, not an optional presentation detail.

Consider an inventory policy that sets a reorder point and order quantity.
A low reorder point can reduce holding cost while increasing lost sales.
A high order quantity can exploit transport economics while consuming space.
The policy is chosen before future demand is known, and the simulator represents demand and lead-time uncertainty.
The business decision is the rule to approve, not the random demand observed in any single simulated run.

This topic presents a practical workflow for a finite policy search and its extension to larger spaces.
The illustrative numbers are synthetic and are not evidence of any named company's results.
Financial figures are in US dollars (USD), and time buckets are stated where used.
The examples distinguish a demand forecast from a plan, a service target, and a commitment to implement.

### The stakeholders and their decisions

The operations owner decides whether a rule can be staffed and followed.
Finance decides which cost components count and how savings will be recognized.
The customer-service owner sets the service measure and acceptable risk.
The modelling team defines, tests, and documents the experiment.
Data owners explain how historical inputs were collected and censored.
An accountable approver decides whether to pilot or reject the recommendation.

A search can be useful without a digital twin.
A standalone, periodically calibrated simulation may be sufficient for an annual staffing choice.
Conversely, a connected twin with stale or unreliable logic is not automatically fit for optimization.
Choose the simplest representation that captures the decision's important consequences.

## 2. Learning outcomes

After this topic, you should be able to:

1. Formulate a feasible policy space, a simulation outcome, and a business objective with units.
2. Explain why repeated evaluations and independent final data are needed for a noisy objective.
3. Design a replication and compute budget that includes search, confirmation, and stress testing.
4. Compare candidates using paired differences and common random numbers where appropriate.
5. Distinguish deterministic feasibility, simulated service constraints, and confidence-qualified acceptance.
6. Calculate a cost comparison and a conservative service bound for a small example.
7. Implement reproducible simulation-based search with separate random streams and a baseline.
8. Recommend a controlled rollout with limits, monitoring, and a decision to stop or replan.

Exercises 1–8 address these outcomes in order.
The exercise criteria require both calculations and an explanation of what the numbers do not establish.
The code is intentionally small enough to inspect rather than a production inventory engine.

## 3. The core mental model

```text
Business decision and approval criteria
             |
             v
   feasible policy generator <---------------------+
             |                                      |
             v                                      |
   simulator + input scenarios                      |
             |                                      |
             v                                      |
   noisy cost and service observations              |
             |                                      |
             v                                      |
   search allocation and candidate comparison ------+
             |
             v
   freeze selected candidate and baseline
             |
             v
   independent confirmation + stress tests
             |
             v
   approve pilot / retain baseline / collect evidence
             |
             v
   monitor real outcomes and recalibrate
```

The loop on top is adaptive: the next simulated decision can depend on results seen so far.
That adaptivity creates selection bias if the same observations are also used as final proof.
The lower confirmation step uses untouched random scenarios and a frozen selection rule.
It estimates performance of the selected policy, not the minimum observed search estimate.

Let a policy be `x` and a complete simulated scenario be `W`.
The simulator returns a cost `C(x, W)` and service outcome `S(x, W)`.
A common formulation is to minimize expected cost `E[C(x, W)]` over feasible `x`, subject to a service requirement.
The expectation describes the assumed scenario distribution, not a promise about every future week.
In practice an average of a finite number of simulated replications estimates it.

A policy is a rule that maps available information into an action.
For example, order a fixed quantity if inventory position is at or below a reorder point.
It is not a list of actions that depends on future demand already known to the simulator.
The simulation can reveal future events to its engine, but the policy must not observe them early.

A scenario must include all relevant uncertainty for a fair comparison.
That may include daily demand, supplier delays, processing times, and outages.
Its components can be correlated; a demand surge and a supplier delay may have a common cause.
An independent replication is an independently generated scenario path, not merely a second observation from the same path.

### Noise, error, and uncertainty are different

Simulation noise arises because only finitely many scenario paths are run.
Input uncertainty arises because the distribution parameters and structure are not known exactly.
Structural error arises when important physical or behavioral mechanisms are missing.
More replications reduce Monte Carlo error under the same model but do not repair the latter two problems.
An independent confirmation set addresses selection bias and sampling noise, not an invalid simulator.

### A search result is a proposal

A finite grid search can identify the best estimated feasible member of its grid.
It does not establish global optimality over decisions not in the grid.
A heuristic can locate a strong candidate in a large space without proving a global optimum.
A business owner may reasonably retain the baseline if the improvement is too small or the risk is too uncertain.
The output should identify what was searched and how much evidence was used.

## 4. Definitions and vocabulary

| Term | Definition and unit | Common misuse |
|---|---|---|
| Policy | Action rule based on available state; parameters have business units | Treating a future-aware schedule as an implementable rule |
| Decision variable | Controllable policy parameter, such as reorder point in units | Treating realized demand as controllable |
| State | Information observable when the decision is made, such as inventory position in units | Including future arrivals or demand |
| Scenario | Joint realization of uncertain inputs over a defined horizon | Combining incompatible independently sampled events |
| Replication | One independent simulated trajectory for an evaluation unit | Counting correlated days in one trajectory as independent runs |
| Response | Measured output per replication, such as USD per year | Assuming every observed response is its expectation |
| Objective | Scalar criterion to minimize or maximize, such as expected USD per year | Quietly mixing service penalties with an explicit service constraint |
| Feasible set | Policies meeting known decision restrictions | Assuming every simulated policy is executable |
| Deterministic constraint | Restriction known from policy inputs, such as a maximum order quantity | Estimating a known capacity rule with simulation |
| Stochastic constraint | Requirement involving uncertain outcomes or their distribution | Treating a sampled pass as a certain pass |
| Chance constraint | Requirement that an event occur with at least a specified probability | Confusing probability of a weekly violation with average fill rate |
| Fill rate | Units filled immediately divided by units requested, with aggregation specified | Averaging ratios and claiming the pooled ratio |
| Cycle service level | Fraction of replenishment cycles without a stockout | Using it interchangeably with unit fill rate |
| Standard error | Estimated sampling variation of an estimator, in the estimator's units | Calling the standard deviation across runs the standard error |
| Confidence interval | Procedure-based interval with a stated long-run coverage property | Claiming a fixed parameter has the stated posterior probability |
| Common random numbers | Matching underlying scenario inputs across compared policies | Merely using the same seed when policy-dependent random draws desynchronize |
| Paired difference | Difference between candidate and baseline on the same scenario | Treating the two outcome samples as independent |
| Selection bias | Optimistic bias caused by selecting on noisy estimates | Reporting the search winner's observed value as unbiased confirmation |
| Search budget | Permitted simulator evaluations, runtime, or monetary compute cost | Spending all runs on exploration and none on confirmation |
| Search set | Policies eligible for evaluation by the algorithm | Reporting only the winner and hiding rejected alternatives |
| Incumbent | Best policy under the current selection rule | Claiming an intermediate incumbent is the true optimum |
| Replication allocation | Assignment of runs to policies | Assuming every policy needs equal runs at every stage |
| Baseline | Current or otherwise approved comparator evaluated under the same assumptions | Comparing against an outdated or differently simulated policy |
| Holdout | Scenario paths withheld from tuning and selection | Repeatedly inspecting and tuning against the holdout |
| Stress scenario | Deliberately challenging input condition, not necessarily probabilistic | Averaging unweighted stress cases into a nominal expectation |
| Warm-up | Initial period discarded to reduce initialization effects in a steady-state study | Automatically discarding early days in a finite-horizon decision |
| Terminating simulation | Simulation with a meaningful start and end, such as one peak season | Assuming its initial state is irrelevant |
| Penalty | Objective adjustment for an undesirable outcome, in compatible units | Assuming an arbitrary penalty guarantees constraint satisfaction |
| Tail risk | Severity or frequency of unfavorable outcomes beyond the mean | Reporting mean cost as evidence about catastrophic loss |
| Model validation | Assessment of fitness for the intended decision using independent evidence | Equating executable code with a valid model |
| Optimization gap | Difference from an optimum or bound, when legitimately available | Inferring a proven gap from a heuristic's stalled search |
| Regret | Performance difference from a comparator, defined for a scenario or distribution | Comparing with a clairvoyant policy without labelling it |
| Pilot | Limited real-world implementation with control and stopping rules | Presenting simulation-only results as realized savings |

## 5. The mechanics, step by step

### Step 1 — Specify the business decision

Write the decision in operational terms before selecting a search algorithm.
Name the owner, decision deadline, action frequency, and planning horizon.
State which information is available when the policy acts.
Document which decisions are already fixed by contracts or operating rules.
Separate forecast demand from the proposed plan, the service target, and the final commitment.

**Input:** a business question, current operating process, and decision calendar.
**Output:** a decision statement and a list of controllable and uncontrollable quantities.

For example, choose an inventory reorder point and order quantity for the next quarter.
Daily demand is uncertain, the supplier lead time is random, and a single policy applies throughout the quarter.
The policy is approved before the quarter starts; it cannot be revised in response to simulated future demand.
If a weekly review is allowed in reality, model its actual information and approval delay explicitly.

A policy can also include an exception action.
An overtime trigger, an expedite rule, or an alternative supplier can be feasible only if contractually available.
Such recourse must carry its cost and lead time.
Do not give the simulated decision maker a free emergency action that the real manager lacks.

### Step 2 — Define the policy space

Represent each decision as an implementable variable.
Specify units, ranges, step sizes, and legal combinations.
For integer workforce decisions, use integer shifts rather than fractional people.
For an inventory policy, define whether the reorder point uses on-hand stock or inventory position.
Inventory position generally includes on-hand plus outstanding orders minus backorders, if backorders are modelled.

**Input:** operating restrictions and controllable levers.
**Output:** a documented set `X` of feasible candidate policies.

Eliminate impossible combinations before paying for simulation.
A supplier minimum order quantity, truck capacity, or staffing agreement can define a hard restriction.
If a constraint is uncertain, distinguish the known rule from the random outcome.
For instance, maximum booked shifts are deterministic, but the number of employees who arrive is uncertain.

A coarse grid is often appropriate for the first pass.
The grid may contain every operationally meaningful policy or only a practical subset.
A finer grid around an apparent winner can improve resolution, but it is still part of search.
Do not describe a grid result as a proof about unsampled continuous choices.

### Step 3 — Specify the objective and constraints

Define the objective in a common unit and a stated horizon.
For inventory, relevant components may include order, carrying, shortage, and expedite cost in USD per quarter.
Use incremental cost where fixed costs are the same across choices.
Avoid counting lost margin twice if it is already included in a shortage penalty.
If the service requirement is explicit, explain why any remaining shortage cost is still economically appropriate.

**Input:** finance definitions, service obligations, and operational rules.
**Output:** an objective equation, service metrics, and feasibility tests.

One possible statement is:

```text
minimize    E[quarterly operating cost(x)]
subject to  x belongs to the implementable policy set
            P(quarter fill rate(x) >= 0.95) >= 0.90
```

The probability here is over quarters under the assumed distribution.
This is not the same requirement as expected fill rate of at least 95%.
It is also not the same as a contractual promise that every customer receives 95% of every order.
Write the unit of analysis and the denominator before computing any service statistic.

An alternative is a mean fill-rate constraint.
That may be appropriate, but it can hide a poor tail.
A pooled unit fill rate across runs weights high-demand runs more than an unweighted average of run-level fill rates.
Choose the definition with the business owner, then keep it unchanged through confirmation.

### Step 4 — Check that the simulator is decision-valid

Trace each policy variable to the mechanisms it can affect.
A staffing decision cannot be trusted if the model omits task queues and staffing coverage.
An inventory decision cannot be trusted if lead times are fixed when supplier variability drives shortages.
Validate against historical periods not used solely for calibration where data allow.
Compare not just means but distributions, correlations, bottleneck behavior, and critical tails.

**Input:** conceptual model, data lineage, calibration, and known historical outcomes.
**Output:** a fitness-for-purpose assessment and unresolved limitations.

An optimizer will search unusual combinations and may exploit a modelling defect.
Manually inspect extreme policies, event ordering, resource constraints, and mass balance.
Test whether sales data hide demand during stockouts.
Check whether a cost or service output changes in the expected direction when an input is deliberately changed.
These checks do not prove validity, but they can expose behavior that makes optimization unsafe.

If the model cannot reproduce a decision-relevant historical pattern, stop or narrow the decision scope explicitly.
More optimization runs cannot fix structural invalidity.
A policy comparison should be conditional on a clear model version and input period.
Any later model change requires re-evaluating the comparison rather than preserving an old winner by default.

### Step 5 — Build scenario paths and random streams

Define a replication as a complete decision-horizon path.
Preserve the dependence between demand, lead time, supplier reliability, and other drivers where evidence supports it.
Use empirical blocks or a fitted joint model when independent sampling would destroy meaningful temporal dependence.
Separate model-input sampling from simulator-internal randomness so experiments can be reproduced.

**Input:** fitted input models, dependence assumptions, and a seed plan.
**Output:** scenario generation code and independent search and confirmation streams.

A seed alone is not a scenario specification.
If one policy consumes an extra random draw when it places an order, subsequent draws can shift.
The same seed can then produce different demand sequences for the two policies.
Pre-generate exogenous demand and lead-time paths, or use separate named streams indexed by event and source.
For order-dependent lead times, specify how the matching is performed and whether matching remains meaningful.

With common random numbers, candidate and baseline are run on matched scenarios.
This usually reduces the variance of their difference when their responses move together.
It is not guaranteed to help: negative covariance can increase variance.
Estimate the variance of paired differences and compare it with an unpaired alternative in a pilot.
Keep different scenario paths independent unless the analysis explicitly handles dependence.

### Interpreting the value of pairing

Consider two policies exposed to the same weather, order volumes, and supplier delays.
When the environment is difficult, both may incur high costs.
Their separate cost levels can vary substantially even when their difference is stable.
Pairing removes some shared environmental variation from the comparison.
It is the distribution of the difference, not either policy's standalone cost distribution, that determines the precision of a paired estimate.

For illustration, suppose each policy's simulated quarterly cost has standard deviation USD 1,000.
If their within-path correlation is 0.8, the variance of their difference is:

```text
1,000^2 + 1,000^2 - 2 * 0.8 * 1,000 * 1,000
= 400,000 USD squared per quarter squared
```

The standard deviation of the paired difference is approximately USD 632 per quarter.
Without pairing, independent policy outcomes with those marginal standard deviations have difference standard deviation approximately USD 1,414 per quarter.
For 100 independent scenario pairs, the corresponding standard errors are approximately USD 63 and USD 141 per quarter.
The figures illustrate variance reduction; actual correlations and variances must be estimated for the application.
Negative correlation reverses the benefit.
Pairing does not make the scenario paths themselves more representative.

### Common random numbers when events do not match

In an inventory model, different policies place different numbers of orders on different dates.
Pairing the first order's lead time across policies makes an explicit order-index coupling.
That coupling preserves the specified marginal lead-time distribution if lead times are independent of order date and policy.
It may be inappropriate if a supplier disruption affects every order placed in a particular week.
For that situation, generate a calendar-indexed supplier state and let each policy's order experience the state on its own order date.
Both policies then encounter the same simulated disruption calendar even though their orders are not synchronized.

The appropriate coupling follows the physical cause of uncertainty.
A shared weather event can affect all deliveries on the same date.
An individual packing error may be tied to an order or worker rather than to the calendar.
Correlated order errors require their shared source to be represented explicitly.
There is no universal rule that the nth random number should represent the same event for all policies.

Record the mapping between random input and event type.
Check that each candidate sees the intended marginal input distribution.
If a policy changes the actual distribution of lead time through larger orders or priority service, encode that causal effect rather than forcing identical realized lead times.
In that case, shared underlying random variables may still enable a meaningful paired comparison.

### Estimating precision before the final run

A pilot can estimate the standard deviation of paired cost differences.
For a rough normal-approximation planning calculation, a desired 95% half-width `h` in USD per quarter and estimated paired standard deviation `s` imply:

```text
n approximately (1.96 * s / h)^2 independent scenario pairs
```

For `s = 800 USD per quarter` and `h = 100 USD per quarter`, the calculation gives approximately 246 pairs after rounding up.
Evaluating a candidate and baseline on each pair requires 492 policy-path calls.
This calculation concerns expected cost difference, not chance-constraint certification.
It assumes the pilot variance and the normal approximation are useful for planning.
Rare events, heavy tails, and uncertainty in the pilot variance can require more paths.

Service evidence has its own sample-size requirement.
Even with zero observed failures, the one-sided Hoeffding rule requires `1 - sqrt(log(1/alpha)/(2*n)) >= p_min`.
For `alpha = 0.05` and `p_min = 0.90`, at least 150 independent paths are needed for this conservative condition.
Any observed failures require more evidence than this all-success calculation.
Use the larger of the cost-precision and service-evidence requirements, subject to the decision budget and separately planned stress tests.
If the budget cannot support the required evidence, report that limitation before inspecting the final outcomes.

### Multiobjective and conflicting business goals

Some decisions have no single natural monetary objective.
A policy may lower cost but increase emissions, lateness, or workload variability.
A weighted score is meaningful only when weights and units have a defensible business interpretation.
An alternative is to impose hard limits on critical outcomes and compare cost within those limits.
Another is to present a set of nondominated candidates: each offers a tradeoff that no other evaluated candidate improves on every stated metric.

Simulation noise can make apparent dominance uncertain.
Use common scenarios and independent final evaluation for whichever candidate the business owner chooses.
Do not select a policy after viewing confirmation results and then claim those same results are independent evidence.
Where tradeoffs require human judgment, define who has authority to choose and which thresholds cannot be waived.
The resulting decision is accountable even when no single number captures every consequence.

### Step 6 — Plan the compute budget

Count simulator calls before running the search.
A call means one policy evaluated on one scenario path, unless another unit is explicitly used.
Budget model checks, initial exploration, refinement, independent confirmation, and stress tests separately.
Include runtime and parallel capacity, not just the number of nominal replications.

**Input:** candidate count, approximate runtime per call, decision deadline, and desired precision.
**Output:** a budget and an allocation rule.

For example, 20 policies times 30 initial paths uses 600 calls.
Three finalists times 100 additional search paths uses 300 calls.
One finalist and one baseline on 200 independent matched paths uses 400 calls.
A further 100 matched stress paths for those two policies uses 200 calls.
The total is 1,500 calls, excluding validation runs that should be budgeted separately.
If a call takes 2 seconds of serial compute, that is 3,000 seconds, or 50 minutes of compute time.
Parallel wall time can be lower but is not necessarily compute cost divided by worker count.

Protect the confirmation allocation from search overruns.
If the budget is exhausted during search, reduce search scope or postpone the decision rather than reuse search data as independent evidence.
Record failed runs and rerun them using a predetermined rule.
Dropping difficult scenarios because they time out can bias the result toward overly optimistic policies.

### Step 7 — Select a search strategy

A finite enumeration is transparent when the policy set is small.
A random or space-filling design explores a larger space without requiring gradients.
A local search can refine promising areas but can miss distant better regions.
Bayesian optimization fits a surrogate model and chooses informative evaluations; noisy outcomes require a noise-aware surrogate.
Evolutionary or other population methods can handle mixed and discontinuous decisions but need careful budgets and reproducibility.

**Input:** dimensionality, decision types, simulator runtime, and evaluation noise.
**Output:** a search algorithm with stopping and reporting rules.

For a costly simulator, spend initial runs on broad coverage and allocate later runs to plausible contenders.
For a fast simulator and a small grid, simple enumeration can be more defensible than a complex heuristic.
If the response surface is nonsmooth because of queues or threshold policies, ordinary finite-difference gradients may be unstable.
Gradient methods can still be appropriate when assumptions and estimators are justified; do not assume every simulator is differentiable.

A stopping rule can be a fixed call budget, deadline, or prespecified precision threshold.
A lack of visible improvement is not proof of global optimality.
Repeatedly checking uncertainty and stopping opportunistically can invalidate fixed-sample confidence statements.
If sequential inference is needed, use a method designed for that stopping rule or reserve a fixed independent confirmation sample.

### Step 8 — Screen and allocate replications

Run a common initial number of independent paths for each candidate.
Screen obvious infeasibilities and extreme costs using a rule fixed in advance.
Allocate more runs to contenders whose cost or feasibility is difficult to distinguish.
Do not eliminate a policy on the basis of a single favorable or unfavorable path.

**Input:** initial response samples and the allocation rule.
**Output:** a shortlist and a complete search history.

Adaptive allocation improves efficiency but complicates naive uncertainty statements.
Search-stage confidence intervals used repeatedly for elimination do not automatically retain their advertised joint coverage.
Treat them as screening diagnostics unless a valid sequential or multiple-comparison procedure is used.
Final evaluation uses fresh data after the candidate is frozen.

When candidates use common paths, preserve the pairing in every comparison.
A candidate with an apparent mean advantage may have a large standard error of paired differences.
If the cost difference is smaller than the minimum economically meaningful improvement, more simulation may not change the business decision.
Agree that threshold before seeing the results.

### Step 9 — Evaluate stochastic feasibility

Define a per-path success event before counting successes.
For a quarterly fill-rate requirement, one path is a success if its quarterly fill rate reaches the target.
With `k` successes in `n` independent paths, `k/n` estimates the success probability.
A high observed fraction alone does not establish the underlying probability is above the threshold.

**Input:** per-path outcomes and a specified confidence rule.
**Output:** an estimated probability, an uncertainty bound, and a feasibility decision.

A conservative one-sided Hoeffding lower bound is

```text
L = max(0, k/n - sqrt(log(1/alpha)/(2*n)))
```

under independent Bernoulli replications.
Here `alpha` is the one-sided error probability, not the required service probability.
For a 95% one-sided bound use `alpha = 0.05`.
Declare confirmation of `P(success) >= p_min` only when `L >= p_min`, using the prespecified method.
This bound can be conservative; an exact binomial bound is another defensible choice when its assumptions hold.

Even all successes may be insufficient with a small sample.
At 100 successes in 100 paths, the Hoeffding lower bound is approximately 0.878, below a 0.90 requirement.
The right action is not to relax the bound after seeing the data.
Increase a prespecified future sample, choose a different properly justified bound, or report insufficient evidence.
If several constraints or finalists are being certified, account for simultaneous error or use a final independent test for the single frozen candidate.

### Step 10 — Freeze and confirm

Freeze the winner, baseline, outcome definitions, sample size, and decision thresholds before opening the confirmation set.
Evaluate both policies on each independent confirmation scenario.
Calculate the paired cost difference and its uncertainty interval.
Evaluate service against its separately specified bound.
Record the selection process and all simulator versions.

**Input:** selected policy, baseline, and untouched confirmation scenario generator.
**Output:** independent cost and service evidence.

For cost, compute `D_i = C_candidate,i - C_baseline,i` for path `i`.
The mean `D_bar` is an estimate of incremental cost per horizon.
Its standard error is the sample standard deviation of the `D_i` values divided by the square root of the number of independent paths.
A negative difference favors the candidate on cost.
For a sufficiently well-behaved sample, a Student-t interval can summarize uncertainty in the expected paired difference.
A small sample with heavy tails warrants caution and potentially a more suitable preplanned method.

If confirmation fails, report the failure.
Selecting the next candidate using the same confirmation results turns those results into new search data.
A fresh confirmation set is needed for an unbiased final assessment of the newly chosen candidate.
No amount of reporting precision removes uncertainty about the underlying scenario distribution.

### Step 11 — Stress and sensitivity test

Run separate scenarios for plausible changes in input assumptions.
Examples include a longer supplier lead time, a promotion-driven demand shift, and a correlated disruption.
Report these as conditional stress results unless their probabilities are defensibly specified.
Do not mix arbitrary stress cases into a nominal expected cost without a justified weighting model.

**Input:** approved candidate, model uncertainties, and disruption narratives.
**Output:** a sensitivity table and operational trigger thresholds.

Vary cost parameters as well as physical inputs.
An apparent saving can disappear when expedited freight or handling cost is valued correctly.
Check whether the recommendation changes at plausible input parameter values.
If small changes reverse the decision, present it as fragile and consider collecting better data or choosing a more robust alternative.

### Step 12 — Decide, deploy, and monitor

The approval packet contains the policy, baseline, estimated incremental cost, service evidence, and limitations.
It also contains a pilot scope, required systems changes, owner, start date, and rollback authority.
The decision can be to implement, pilot, collect data, or retain the baseline.
The policy is a proposal until the designated business owner commits to it.

**Input:** confirmation and stress evidence, implementation cost, and governance requirements.
**Output:** an explicit approval decision and monitoring plan.

Monitor realized service, cost, demand distribution, lead time, and manual overrides against the model's definitions.
Compare comparable periods and account for seasonality or other interventions.
Realized performance is not automatically causal evidence of the policy's effect.
Where feasible, a controlled pilot or phased comparison can improve attribution.
Pause or roll back when a prespecified service or safety threshold is crossed.
Recalibrate if the environment materially differs from the validated range.

## 6. Worked example

### Decision and business context

A regional distributor replenishes a single fast-moving item weekly.
The inventory manager must choose a reorder point and a fixed order quantity for the coming quarter.
The supplier accepts the order sizes listed below and can deliver at most one outstanding order at a time.
Demand and supplier lead time vary across simulated quarters.
The example is synthetic; the figures illustrate arithmetic, not measured performance.

The demand forecast is a probability model for quarterly paths.
The plan is the candidate reorder rule selected by the search.
The target is a quarterly fill rate of at least 95% on at least 90% of modeled quarters.
The commitment would be the manager's approval to pilot the rule; simulation does not itself make that commitment.

The current baseline has reorder point 70 units and order quantity 100 units.
The feasible alternatives are as follows.
The reorder point refers to inventory position in units.
A new order is placed when the end-of-day position is at or below the point and no order is already outstanding.

| Policy | Reorder point, units | Order quantity, units | Operational status |
|---|---:|---:|---|
| A | 60 | 100 | Feasible candidate |
| B | 70 | 100 | Current baseline |
| C | 80 | 100 | Feasible candidate |
| D | 80 | 120 | Feasible candidate |
| E | 100 | 160 | Excluded: warehouse order limit is 120 units |

The warehouse limit is a deterministic restriction.
Policy E is not simulated as a possible recommendation.
The quarterly simulation charges ordering, inventory holding, and shortage costs in USD.
The exact event timing and input model must be documented in the real study.
The table below is an illustrative summary of a separate study, not output from the teaching code in section 7.

### Search and selection

Suppose 100 independent search paths are shared across policies A through D.
The same demand and supplier conditions are used for each policy on each matched path.
The search-stage results are:

| Policy | Mean quarterly cost, USD | Quarters meeting 95% fill, count of 100 | Search interpretation |
|---|---:|---:|---|
| A | 12,080 | 78 | Too few service successes for the screen |
| B | 12,560 | 92 | Baseline; retain as comparator |
| C | 12,210 | 98 | Lowest cost among screened candidates |
| D | 12,470 | 99 | Higher search cost than C |

The team prespecified a 95-of-100 search screen for service; it is a screening rule, not a certification.
A and B do not pass that screen, but B remains the operational baseline for comparison.
C is selected as the lowest-cost screened candidate and then frozen.
The search estimate suggests a USD 350 per quarter advantage over B.
It is optimistic to treat that selected estimate as final proof because C was chosen after inspecting search results.

The team reserves 400 fresh independent quarter paths for confirmation.
Each path is evaluated for both C and B, retaining scenario pairing.
The confirmation run reports a mean cost of USD 12,260 for C and USD 12,540 for B.
The paired differences have a sample standard deviation of USD 800 per quarter.
There are 400 independent paired differences.

The estimated incremental cost is:

```text
C minus B = 12,260 - 12,540 = -280 USD per quarter
standard error = 800 / sqrt(400) = 40 USD per quarter
```

Using an approximate 95% normal interval gives `-280 +/- 1.96 * 40`.
The interval is approximately `[-358.4, -201.6] USD per quarter`.
A Student-t critical value for 399 degrees of freedom gives a very similar interval.
The interval concerns the model's expected incremental quarterly cost under its assumed distribution.
It is not an interval for a particular quarter's cost or for savings realized after implementation.

### Service calculation

Suppose C meets the 95% quarterly fill target in 398 of the 400 confirmation paths.
Its observed success fraction is `398/400 = 0.995`.
For a one-sided 95% Hoeffding lower bound:

```text
margin = sqrt(log(20) / (2 * 400)) = approximately 0.0612
lower bound = 0.995 - 0.0612 = approximately 0.9338
```

The lower bound exceeds the required probability of 0.90.
Under the independent-path and correct-scenario-model assumptions, the prespecified statistical rule confirms the modeled chance constraint.
It does not establish a 95% fill rate on every quarter or for every customer.
It does not cover changes in the demand or supply distribution after the study.

Suppose implementation requires a one-time USD 500 system change.
The estimated first-year recurring saving is `4 * 280 = 1,120 USD`, if four comparable quarters occur.
Subtracting the one-time change gives USD 620 in estimated first-year net benefit before discounting.
That annualization assumes the quarterly model and operating conditions remain representative.
It is a business projection, not confirmed annual savings.

### Decision and stress response

The manager should consider a limited pilot of C if the implementation effort and risk tolerance support it.
The approval packet should retain the baseline and show the confirmation interval and service bound.
A stress test should separately increase supplier lead time and increase demand variability.
A stress result is not given here; no conclusion about C under those stresses follows from the nominal calculation.

Before launch, define a weekly monitoring rule with a minimum observation period and a clear escalation owner.
A severe customer-service failure can trigger immediate review even if the model's quarterly metric has not yet matured.
If the supplier changes its delivery process, re-estimate lead-time inputs and reconsider the policy.
A pilot should record stockouts and unfilled demand rather than only shipped orders.

### What the example proves and does not prove

The search chose C within the stated finite set and screen.
The independent evaluation supports a lower expected simulated cost than B under the model.
The conservative lower bound supports the chosen modeled chance constraint under the assumptions.
The calculation does not prove C is globally optimal, that the simulator is valid, or that savings will occur in production.
The cost and service evidence also do not eliminate exposure to unmodelled disruption.

## 7. Code, query, or calculation

The following standard-library Python program is a complete, small teaching experiment.
It searches a fixed grid of inventory policies and then confirms one frozen candidate against the baseline.
It models lost sales rather than backorders, one outstanding order, and a finite 90-day horizon.
All units and cost parameters are illustrative.
The code's results are separate from the hand-worked example above.

Python 3.9 or later is sufficient; no external package or input file is required.
Save the block as `simulation_search.py` and run `python simulation_search.py`.
Demand paths and order-specific lead times are generated before policy evaluations.
Each policy receives the same scenario object for a paired comparison.
The search and confirmation sets use different seeds.

```python
import math
import random
import statistics

DAYS = 90
POLICIES = [(40, 60), (50, 60), (60, 60), (70, 60)]
BASELINE = (50, 60)
SEARCH_N = 120
CONFIRM_N = 600
TARGET_FILL = 0.95
REQUIRED_PROB = 0.90
ALPHA = 0.05


def make_paths(seed, count):
    """Independent paths of daily demand and potential order lead times."""
    rng = random.Random(seed)
    paths = []
    for _ in range(count):
        # Demand is integer units/day; this simple distribution has mean 8.
        demand = [rng.randint(4, 12) for _ in range(DAYS)]
        # Assign lead times by order index, independently of the policy.
        leads = [rng.randint(2, 5) for _ in range(DAYS)]
        paths.append((demand, leads))
    return paths


def simulate(policy, path):
    """Return USD per 90 days and the 90-day immediate unit fill rate."""
    reorder_point, order_qty = policy
    demand, leads = path
    on_hand = 65
    pending_qty = 0
    arrival_day = None
    orders = 0
    requested = 0
    filled = 0
    holding_units_days = 0
    for day, need in enumerate(demand):
        if arrival_day == day:
            on_hand += pending_qty
            pending_qty = 0
            arrival_day = None
        served = min(on_hand, need)
        on_hand -= served
        requested += need
        filled += served
        holding_units_days += on_hand
        position = on_hand + pending_qty
        if position <= reorder_point and pending_qty == 0:
            pending_qty = order_qty
            arrival_day = day + leads[orders]
            orders += 1
    lost_units = requested - filled
    cost = 35.0 * orders + 0.04 * holding_units_days + 8.0 * lost_units
    return cost, filled / requested


def evaluate(policy, paths):
    return [simulate(policy, path) for path in paths]


def lower_bound(successes, n, alpha=ALPHA):
    return max(0.0, successes / n - math.sqrt(math.log(1 / alpha) / (2 * n)))


def main():
    search_paths = make_paths(202601, SEARCH_N)
    search = {p: evaluate(p, search_paths) for p in POLICIES}
    # Search screening is deliberately not a statistical certification.
    eligible = [p for p in POLICIES
                if sum(fill >= TARGET_FILL for _, fill in search[p]) / SEARCH_N >= 0.95]
    if not eligible:
        print("No candidate passed the search screen; retain baseline pending more study.")
        return
    winner = min(eligible, key=lambda p: statistics.mean(c for c, _ in search[p]))
    print("Selected policy:", winner, "baseline:", BASELINE)
    for p in POLICIES:
        results = search[p]
        mean_cost = statistics.mean(c for c, _ in results)
        success_rate = sum(fill >= TARGET_FILL for _, fill in results) / SEARCH_N
        print(f"Search {p}: mean USD {mean_cost:.2f}; success {success_rate:.3f}")

    # No tuning or re-selection takes place after these paths are generated.
    confirm_paths = make_paths(202602, CONFIRM_N)
    candidate = evaluate(winner, confirm_paths)
    baseline = evaluate(BASELINE, confirm_paths)
    differences = [c[0] - b[0] for c, b in zip(candidate, baseline)]
    mean_difference = statistics.mean(differences)
    se = statistics.stdev(differences) / math.sqrt(CONFIRM_N)
    successes = sum(fill >= TARGET_FILL for _, fill in candidate)
    bound = lower_bound(successes, CONFIRM_N)
    print(f"Confirmation paired difference USD: {mean_difference:.2f}")
    print(f"Approximate 95% normal interval USD: "
          f"[{mean_difference - 1.96 * se:.2f}, {mean_difference + 1.96 * se:.2f}]")
    print(f"Confirmation successes: {successes}/{CONFIRM_N}; "
          f"one-sided 95% lower bound: {bound:.4f}")
    print("Service evidence:", "meets rule" if bound >= REQUIRED_PROB else "inconclusive or fails rule")


if __name__ == "__main__":
    main()
```

Expected output is either the explicit no-candidate message or a selected-policy line, four search summaries, a paired cost difference and interval, and a service-evidence line.
Numerical values depend on the stated seeds and Python's random implementation; the example does not require a particular policy to win.
The program exits without confirmation if no policy passes the prespecified search screen.
If the winner happens to be the baseline, the paired difference is exactly zero and no improvement has been demonstrated.

The model places an order after daily demand and receives orders at the start of the arrival day.
Orders that would arrive after day 89 remain outstanding at the horizon; no salvage value or terminal penalty is included.
That terminal convention can bias the comparison and must be reviewed before business use.
The model initializes every path with 65 units and no order outstanding.
That is a finite-horizon assumption, not a claim of steady state.
The source of randomness is deliberately simple and ignores correlation and seasonality.

The 95% normal interval is an approximation for the expected paired difference.
It is not a guarantee when differences are heavily skewed or dependent.
The conservative service bound applies to the frozen candidate evaluated on independent confirmation paths.
The four search policies have not each been certified by their search results.
The independent holdout does not address distributional misspecification or structural model error.
Production software would also need validated data ingestion, failure handling, monitoring, and a documented release process.

## 8. Data, standards, and regulatory context

### Data required for a policy comparison

Collect transaction timestamps, item quantities, locations, order status, and inventory adjustments.
Retain both requested quantity and fulfilled quantity where possible.
A sales record alone may understate demand when items were unavailable.
Identify partial shipments, substitutions, cancellations, returns, and manual overrides.
Maintain a documented mapping from operational events to simulation events.

Supplier lead time should start and end at defined milestones.
Purchase-order creation, supplier acceptance, dispatch, receipt, and availability for picking are not interchangeable timestamps.
A nominal contractual lead time is not necessarily the distribution observed by operations.
Account for weekends, holidays, and cut-off times if they affect availability.
Preserve late, cancelled, and partially received orders rather than deleting inconvenient records.

Labor models need shift calendars, breaks, task times, absence, cross-training, and labor constraints.
Queue models need arrival times and service completion times at the relevant resource.
Capacity may be shared by products or customers; isolated item-level simulation can overstate available capacity.
Input models should represent known dependence and changing operating conditions.

For cost, distinguish financial ledger values from decision-relevant incremental costs.
Document the valuation of holding inventory, lost margin, expedite freight, labor, and service penalties.
Specify currency, price date, horizon, and whether tax or financing is included.
Reconcile unit costs with finance before interpreting simulated savings.
Include implementation cost separately when it is not incurred on every simulated quarter.

### Data quality and provenance

Record the source system, extraction date, transformation, owner, and validation status for every input.
Maintain immutable scenario-generation configurations, random seeds, simulator versions, and policy definitions.
A seed does not reproduce an experiment if the code, inputs, or random-number implementation changes.
Store sufficient artifacts to reconstruct the decisions and summary statistics.
Restrict access to sensitive transaction and employee data.
Use aggregated or synthetic data when detailed personal data are not needed.

Check missing timestamps, duplicate orders, impossible quantities, and inconsistent units.
Investigate stockouts before fitting demand from observed sales.
Hold out a historical period for validation where feasible, and document regime differences.
Data leakage occurs when future information enters a policy or when evaluation data influence selection.
Neither leakage is cured by adding more simulated runs.

### Standards and applicable obligations

GS1 identification standards provide conventions for identifying trade items and locations.
They can support consistent joins across systems, but identifiers do not guarantee accurate inventory records or validate a simulator.
The appropriate identifier and implementation depend on the trading process.
Consult the current GS1 specifications and trading-partner agreements for actual requirements.

ISO 31000:2018 provides risk-management guidance, not a prescribed simulation-optimization algorithm or a certification of a particular policy.
Its relevance is the discipline of identifying, evaluating, treating, and monitoring risk.
It does not determine the correct service probability, confidence level, or uncertainty set.

ISO/IEC 27001:2022 specifies requirements for an information-security management system when an organization adopts or is assessed against that standard.
It does not automatically apply to every simulator or make simulation results reliable.
An organization should follow its own security policies and applicable contractual obligations.

Where employee-level operational data identify people, privacy law and employment rules can apply.
For example, the EU General Data Protection Regulation (GDPR) applies to personal-data processing within its territorial and material scope.
Its principles include data minimization and purpose limitation; legal applicability depends on facts and jurisdiction.
A simulator's output is not automatically exempt from privacy obligations merely because it is analytical.
Obtain qualified advice for actual obligations, retention, and worker consultation.

Customer service targets and contractual service-level agreements are different concepts.
A model's probability-of-fill constraint is not necessarily the contract's measurement or remedy.
Map actual contract language to the correct customer, period, denominator, exceptions, and reporting rules.
Check whether the proposed policy could violate safety, labor, product, or procurement restrictions before rollout.
This educational topic does not provide legal or compliance advice.

### Model governance

Name an owner for the simulator, input models, policy selection, and operational approval.
Keep the search dataset separate from the final confirmation dataset in both storage and access controls.
Record any access to holdout results before a candidate is frozen.
Define who can change cost weights, service thresholds, policy ranges, or the simulation horizon.
An altered objective defines a new decision study, even if the code still runs.

Version the assumptions as carefully as the software.
An updated supplier lead-time distribution can change the preferred policy without a code change.
Retain failed and rejected candidates in the experiment record.
A dashboard showing only a winning policy is insufficient for audit or review.

## 9. Industry practice and case evidence

### How to read the evidence

Simulation-optimization is a method, not a single commercial product.
Public studies vary in whether they describe a real deployment, a retrospective simulation, or a synthetic demonstration.
An optimization result obtained in a model should not be described as measured operational savings without evidence of implementation.
A vendor's performance claim is not an independent controlled evaluation.
The examples below identify the relevant source and separate documented claims from instructional inference.

### Semiconductor manufacturing

The peer-reviewed review by Fu, Glover, and April describes simulation optimization methods and applications, including complex operations where analytical evaluation is difficult.
It explains the combination of simulation with search and discusses implementation issues.
The article is methodological evidence, not evidence that a particular warehouse achieved a stated saving.
Its practical lesson is that the choice of optimization procedure depends on the simulation problem and available information.

[INFERENCE] A semiconductor manufacturer with expensive, congested equipment could use a simulation-based search to compare dispatching rules.
A useful study would preserve machine failures, sequence-dependent processing, and work-in-process queues.
The operational decision would be a dispatching rule, not the simulated realization of future demand.
Any proposed saving would still need an independent evaluation and a controlled deployment.

### Logistics network design

The UPS ORION initiative is publicly described by UPS as a route-optimization system used to improve delivery routes.
UPS describes operational route planning and technology in its own materials; those statements are company evidence.
They do not establish that the inventory code in this topic, or any particular simulation-optimization method, was used.
They do show that deployment of an analytical recommendation depends on route data, operating constraints, and driver-facing execution.

[INFERENCE] A logistics operator could use simulation to compare routing policies under uncertain travel times and arrivals.
A candidate must be tested against service windows, driver hours, and feasible dispatch information.
A simulation-only estimated saving would remain distinct from a measured post-deployment UPS result.
This analogy illustrates the decision context, not a reconstruction of ORION's implementation.

### Inventory control research

Hong, Fan, and Luo's review surveys ranking and selection methods for comparing alternatives with noisy simulation outputs.
It supports the general methodological distinction between search and the evaluation of noisy responses.
It is not a company case or a guarantee that a specific heuristic will work in every inventory system.

[INFERENCE] A distributor facing variable lead times can search a reorder policy space when a closed-form inventory formula omits important interactions.
The search must include operational constraints and compare against the current policy on matched scenarios.
A statistically convincing simulated improvement can still disappear if demand was censored or the model omits supplier constraints.
A pilot and monitoring plan are therefore part of the business decision.

### Simulation evidence at scale

The Winter Simulation Conference proceedings contain peer-reviewed papers on simulation methodology and applications.
A published model can document a useful design without implying that its assumptions transfer to another site.
Read the original paper for the decision horizon, modeled system boundary, data period, validation, and observed deployment.
Do not copy a reported improvement percentage without confirming what the comparison and metric mean.

When using a public case to justify a local investment, distinguish four layers:

1. A documented business problem and site context.
2. A documented model or algorithm and its assumptions.
3. Reported simulated performance under specified scenarios.
4. Independently measured operational results after implementation, if available.

A source may substantiate only the first two or three layers.
The absence of a public post-deployment measurement is not evidence that the project failed or succeeded.
It is a reason to limit the claim made in the approval packet.

### Translating evidence into a local decision

Start with the current operating baseline, not a published case's comparator.
Estimate local cost components and service requirements.
Use local event data to validate the simulator and its input distributions.
Measure the incremental effect under a documented pilot if feasible.
Disclose differences in products, network, calendar, contracts, and disruption exposure.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Winner's curse | Search winner's holdout cost is much worse | Selection favors favorable noise | Overstated savings | Freeze then use independent confirmation |
| Leaked holdout | Team repeatedly checks final results while tuning | Evaluation data become search data | Misleading confidence | Restrict access and reserve fresh paths |
| Invalid simulator | Historical queues or stockouts are not reproduced | Missing decision-relevant mechanism | Wrong policy ranking | Validate structure and outputs before search |
| Infeasible policy | Candidate violates a contract or system limit | Feasible set was incomplete | Unimplementable recommendation | Encode hard constraints before simulation |
| Future information | Rule reads next week's demand | Look-ahead bias | Unrealistic service and cost | Enforce an information timeline |
| Mismatched random streams | Same seed gives different demand sequences | Policy-dependent random consumption | Lost pairing or misleading comparison | Pre-generate inputs or use named streams |
| Insufficient service evidence | Few failures in a tiny sample | Rare-event probability is poorly estimated | False confidence in service | Use a prespecified confidence bound and budget |
| Wrong service denominator | Fill rate differs between reports | Units, orders, and quarters are mixed | Wrong feasibility decision | Specify aggregation and numerator/denominator |
| Double-counted shortage | Cost increases unexpectedly when service penalties are added | Lost margin appears in two components | Distorted optimization | Reconcile financial definitions |
| Censored demand | Sales fall to zero during stockout | Unobserved unmet demand | Understated shortage risk | Capture lost demand or model censoring |
| Terminal artifact | Late orders arrive after the horizon | End effects favor some policies | Biased ranking | Model terminal value or extend the horizon |
| Bad initialization | Results depend strongly on starting stock | Transient dominates short horizon | Nonrepresentative costs | Use realistic initial-state distribution |
| Broken dependence | Stress disappears when drivers are sampled independently | Common disruptions removed | Underestimated tail | Model joint behavior and stress dependence |
| Adaptive inference | Many interim intervals are treated as final | Repeated selection and stopping | Coverage below nominal | Fixed independent confirmation or valid sequential method |
| False optimality | Heuristic stops improving | Local search or budget limit | Unjustified global claim | Report search space, budget, and lack of bound |
| Excessive precision | Costs reported to cents despite uncertain inputs | Monte Carlo precision mistaken for model accuracy | Overconfidence | Show input and structural sensitivity |
| Time-out bias | Difficult scenarios fail more often | Failed outputs are silently omitted | Optimistic performance | Track failures and use a predetermined remedy |
| Weak baseline | Current policy is misconfigured in the model | Comparator not representative | Exaggerated benefit | Validate baseline with operations owner |
| Distribution shift | Actual lead times drift outside validation range | Input model no longer represents operations | Service deterioration | Monitor drift and revalidate |
| No deployment owner | Recommended policy stays in a report | Decision rights and system changes absent | No realized benefit | Assign owner, pilot, and rollback |

A low Monte Carlo standard error is not a general quality certificate.
It can coexist with every structural, data, and governance failure in the table.
Escalate discrepancies between real operations and the model rather than treating them as inconvenient observations.

## 11. Assumptions and limits

| Assumption | Why it matters | What breaks it | How to investigate |
|---|---|---|---|
| Replication paths are independent | Standard errors and the binomial bound rely on independent sampling | Reused paths or correlated random streams | Inspect generation and seed allocation |
| Scenario distribution is relevant | Expected cost and chance constraint refer to that distribution | Seasonality, disruption, or process change | Compare holdout periods and monitor drift |
| Dependence is represented adequately | Joint events can drive shortages | Independent sampling of correlated drivers | Compare joint and conditional empirical behavior |
| Model captures causal mechanisms | Search ranking must respond realistically to decisions | Missing congestion, shared capacity, or behavior | Face validation and extreme-policy tests |
| Policy uses available information | A real operator must be able to apply the rule | Future demand or lead time leaks into decisions | Trace data availability at each event |
| Costs are complete and consistent | Objective must reflect the economic tradeoff | Missing expedite cost or double counting | Finance reconciliation and sensitivity analysis |
| The feasible set is implementable | Optimization must not propose impossible actions | Contracts, software, or physical limits omitted | Obtain operational and legal review |
| Service event matches the requirement | The statistical bound applies to the defined event | Wrong denominator, period, or customer population | Reconcile metric to contract and reports |
| Initial state is representative | A finite horizon includes start-up effects | Atypical initial inventory or backlog | Compare plausible initial conditions |
| End effects are acceptable | Terminal inventory and outstanding orders affect costs | Horizon truncates replenishment cycles | Extend horizon or add terminal valuation |
| Holdout is untouched until freeze | Confirmation must not influence selection | Repeated selection using holdout outcomes | Audit access and preserve a fresh test set |
| Cost-difference interval is appropriate | Normal or t approximation needs adequate behavior | Heavy tails or few independent paths | Inspect path-level differences and use suitable methods |
| Operational response remains stable | Simulated policy must be executed as modelled | Overrides, learning, or system constraints | Monitor compliance and pilot deviations |
| Stress cases are interpreted correctly | Conditional tests are not probability estimates | Arbitrary stresses averaged as if representative | Label scenario weights and provenance |

These assumptions are not all equally important in every application.
Prioritize those that can change the policy ranking or cause unacceptable service harm.
A bound on simulation noise cannot compensate for a misspecified input distribution.
A validated historical fit cannot ensure that an unprecedented future disruption is covered.

## 12. Exercises

### Exercise 1 — Formulate a policy search

A depot chooses the number of morning and evening shifts for a 13-week peak period.
Morning and evening demand are uncertain and share a common promotion effect.
The depot has at most 16 shifts per day and cannot assign fractional shifts.
Late orders incur a cost; a customer requires a defined on-time service target.

**Deliverable:** Write the decision variables, feasible set, random scenario, objective, and one clearly defined service constraint.
**Acceptance criteria:** Units and horizon are explicit; the policy uses no future information; shared demand dependence is retained.
**Time:** 15 minutes.

### Exercise 2 — Diagnose selection bias

Ten policies are evaluated on noisy search samples, and the lowest estimated cost is selected.
The analyst proposes to publish the winner's search average as its final expected cost.

**Deliverable:** Explain the likely bias and design a confirmation experiment.
**Acceptance criteria:** Freeze the candidate, use independent paths, retain a baseline, and distinguish Monte Carlo error from model error.
**Time:** 10 minutes.

### Exercise 3 — Allocate a budget

A simulator takes 3 seconds per policy-path evaluation.
You can spend at most 1,800 seconds of serial compute and must compare a winner with a baseline on 100 matched confirmation paths.
There are eight search candidates.
Ignore setup overhead and stress testing for this exercise only.

**Deliverable:** Calculate remaining search calls and a balanced initial allocation.
**Acceptance criteria:** Count both policies in confirmation; do not exceed the budget; state unused calls if any.
**Time:** 10 minutes.

### Exercise 4 — Calculate a paired comparison

For five independent matched paths, candidate-minus-baseline costs in USD per quarter are `[-100, -50, 0, -150, 50]`.

**Deliverable:** Compute the mean paired difference, sample standard deviation, and standard error.
**Acceptance criteria:** Use `n-1` in the sample variance and state which policy has lower estimated cost.
**Time:** 15 minutes.

### Exercise 5 — Evaluate a chance constraint

A candidate achieves at least 95% quarterly fill in 198 of 200 independent simulated quarters.
The requirement is a success probability of at least 0.90, assessed using the one-sided 95% Hoeffding bound in section 5.

**Deliverable:** Compute the observed fraction and lower bound, then apply the rule.
**Acceptance criteria:** Use the correct path count and distinguish the modeled probability from a guarantee for every quarter.
**Time:** 10 minutes.

### Exercise 6 — Critique a search report

A report states: “We tested 500 policies, found a 12% simulated saving, and therefore proved the globally optimal policy is safe.”
No baseline definition, holdout, constraints, or validation are described.

**Deliverable:** Identify at least five deficiencies and propose a remedy for each.
**Acceptance criteria:** Address selection, model validity, feasibility, uncertainty, and the global-optimality claim.
**Time:** 15 minutes.

### Exercise 7 — Extend the runnable model

The code in section 7 uses lost sales, a fixed starting inventory, and a zero terminal value.

**Deliverable:** Specify an implementable change to represent outstanding orders and terminal inventory economically; describe how you would compare it with the existing convention.
**Acceptance criteria:** Define timing and units, avoid future information, and propose independent confirmation after any re-selection.
**Time:** 20 minutes.

### Exercise 8 — Prepare an approval decision

A candidate meets the modeled service rule and has an estimated recurring saving of USD 280 per quarter.
Its supplier lead-time distribution was estimated before a recent supplier change.
Implementation costs USD 500 and the candidate has not been piloted.

**Deliverable:** Write a short approval recommendation with a pilot, monitoring, and rollback plan.
**Acceptance criteria:** Distinguish projection from realized benefit; identify the input-model uncertainty, owner, metric, and trigger.
**Time:** 15 minutes.

## 13. Self-check answers

### Answer 1

Let `m` and `e` be nonnegative integer shifts per day.
A simple fixed staffing policy satisfies `m + e <= 16` shifts per day.
A scenario contains the 13-week sequence of morning and evening demand, including their common promotion effect, and any relevant absence and service-time uncertainty.
Minimize expected labor plus late-order cost in USD per 13-week period.
One possible constraint is `P(on-time orders / eligible orders >= 0.95 in a 13-week period) >= 0.90`.
Define order eligibility, due times, and how zero eligible orders are handled before implementation.
The policy cannot use demand that will be observed only after staffing is committed.
A more detailed policy may react to observed orders if the actual planning process permits that reaction.

### Answer 2

The minimum of noisy estimates tends to be too favorable for a minimization problem.
Some of the apparent advantage may be random noise rather than a real difference in expected cost.
Freeze the selected policy and compare it with the actual baseline on independent paths not used in search.
Pair both policies on each path and report the mean and uncertainty of their differences.
Independently assess service feasibility using a prespecified rule.
This reduces selection-related optimism but does not validate the simulator or its assumed inputs.

### Answer 3

The total budget is `1,800 / 3 = 600` policy-path calls.
Confirmation requires `2 * 100 = 200` calls for the candidate and baseline.
That leaves `600 - 200 = 400` calls for search.
An equal initial allocation is `400 / 8 = 50` paths per candidate.
There are no unused calls in that allocation.
The baseline may also be a search candidate, but its confirmation evaluations remain separate and still count.
In a real study reserve additional resources for validation and stress testing.

### Answer 4

The sum of the five differences is `-250 USD` and the mean is `-50 USD per quarter`.
The deviations from the mean are `[-50, 0, 50, -100, 100] USD`.
Their squared sum is `25,000 USD squared`.
The sample variance is `25,000 / 4 = 6,250 USD squared`.
The sample standard deviation is approximately `79.06 USD`.
The standard error is `79.06 / sqrt(5)`, approximately `35.36 USD per quarter`.
The candidate has the lower estimated expected cost by USD 50 per quarter.
Five paths are too few to make strong distribution-free claims about the mean.

### Answer 5

The observed success fraction is `198 / 200 = 0.99`.
The margin is `sqrt(log(20) / 400)`, approximately `0.0865`.
The lower bound is approximately `0.9035`.
Because `0.9035 >= 0.90`, the candidate passes this prespecified one-sided rule under the independent Bernoulli-path assumptions.
That conclusion is conditional on the scenario model and is not a promise of no bad quarters.

### Answer 6

A strong critique identifies at least five distinct problems:

- The search winner is selected on noise; reserve independent confirmation.
- The baseline is undefined; compare with the current implementable policy.
- The simulator's validity is unknown; assess structure, historical outputs, and inputs.
- Service and operational feasibility are unspecified; state and test the constraints.
- The 12% estimate has no sampling uncertainty; report an appropriate interval and sample size.
- Testing 500 policies does not prove global optimality outside the tested set; describe the search space and any valid bound.
- Safety is not established by cost; define safety requirements and independent review.
- No deployment evidence is given; describe the figure as simulated rather than realized.

### Answer 7 rubric

A strong design values end-of-horizon on-hand inventory in USD per unit and treats outstanding orders consistently with ordering and purchase costs.
It states whether an order placed before the horizon incurs its cost immediately and whether goods arriving after the horizon retain value.
It defines the terminal accounting convention before comparing policies.
It uses the same exogenous paths for paired comparisons and tests sensitivity to the horizon.
If the change alters the selected policy, it reserves a fresh independent confirmation set.
It does not use simulated future demand to decide whether to place an order.

### Answer 8 rubric

A strong recommendation treats USD 280 per quarter as a modeled expectation rather than realized savings.
It asks the supplier-data owner to refresh and validate the lead-time distribution after the supplier change.
It names an operations owner and a limited pilot scope, plus customer-service and cost measures with denominators and review frequency.
It sets an actionable stop or escalation trigger before launch and identifies who can restore the baseline.
It accounts for the USD 500 implementation cost and does not assume four identical quarters without qualification.
It may recommend postponement if the new supplier regime cannot be represented credibly.

## 14. Interview questions

1. **Why is the lowest search estimate usually optimistic?**  
   Strong: explains selection among noisy estimates and independent confirmation.  
   Weak: claims the simulator's average is automatically unbiased after selection.

2. **What is a policy rather than a scenario?**  
   Strong: distinguishes a rule based on available state from uncertain future inputs.  
   Weak: treats realized future demand as a controllable parameter.

3. **When do common random numbers help?**  
   Strong: describes matched scenarios, covariance, and paired differences.  
   Weak: assumes the same seed guarantees aligned inputs or always reduces variance.

4. **How do you allocate a limited compute budget?**  
   Strong: reserves independent confirmation and considers exploration, refinement, and stress.  
   Weak: spends every run finding the minimum and reports it as verified.

5. **How do you certify a simulated chance constraint?**  
   Strong: defines the per-path event, independent sample, threshold, and one-sided uncertainty bound.  
   Weak: reports one high average fill rate as proof of every-quarter service.

6. **What can more replications fix?**  
   Strong: reduces Monte Carlo uncertainty but not invalid structure or input bias.  
   Weak: claims sufficiently many runs make a wrong model correct.

7. **What is the danger of optimizing a simulator?**  
   Strong: explains that search can exploit missing constraints or modelling defects.  
   Weak: assumes optimization itself validates the model.

8. **How would you handle a candidate that fails confirmation?**  
   Strong: report the result and use fresh confirmation after any new selection.  
   Weak: silently select the next policy on the same holdout and reuse its interval.

9. **When is a heuristic result globally optimal?**  
   Strong: only when a valid proof or bound supports that claim; report space and budget otherwise.  
   Weak: equates a long period without improvement with a proof.

10. **How do you move from simulated savings to a business commitment?**  
    Strong: includes implementation cost, ownership, controlled pilot, monitoring, and rollback.  
    Weak: calls modeled savings realized without operational evidence.

## 15. Key terms

- **Baseline:** Current or approved comparator; see [mechanics](#step-10--freeze-and-confirm).
- **Chance constraint:** Probability requirement for a defined event; see [service evaluation](#step-9--evaluate-stochastic-feasibility).
- **Common random numbers:** Matched random inputs across policies; see [scenario paths](#step-5--build-scenario-paths-and-random-streams).
- **Confidence interval:** Sampling-uncertainty procedure; see [confirmation](#step-10--freeze-and-confirm).
- **Confirmation set:** Untouched paths used after selection; see [freeze and confirm](#step-10--freeze-and-confirm).
- **Cost difference:** Candidate cost minus baseline cost; see [worked example](#6-worked-example).
- **Decision variable:** Controllable policy parameter; see [policy space](#step-2--define-the-policy-space).
- **Deterministic constraint:** Known restriction on decisions; see [policy space](#step-2--define-the-policy-space).
- **Distribution shift:** Change in the scenario-generating environment; see [assumptions](#11-assumptions-and-limits).
- **Feasible set:** Permitted policy combinations; see [policy space](#step-2--define-the-policy-space).
- **Fill rate:** Fraction of requested units filled under a stated aggregation; see [definitions](#4-definitions-and-vocabulary).
- **Holdout:** Data withheld from selection; see [confirmation](#step-10--freeze-and-confirm).
- **Incumbent:** Current best candidate under the selection rule; see [search strategy](#step-7--select-a-search-strategy).
- **Input uncertainty:** Uncertainty about scenario-model parameters and structure; see [mental model](#noise-error-and-uncertainty-are-different).
- **Monte Carlo error:** Sampling variation from finitely many simulated paths; see [mental model](#noise-error-and-uncertainty-are-different).
- **Objective:** Criterion optimized over policies; see [objective and constraints](#step-3--specify-the-objective-and-constraints).
- **Paired difference:** Difference in outcomes on a shared scenario; see [confirmation](#step-10--freeze-and-confirm).
- **Pilot:** Limited operational trial; see [deployment](#step-12--decide-deploy-and-monitor).
- **Policy:** An implementable action rule; see [mental model](#3-the-core-mental-model).
- **Replication:** One simulated trajectory; see [definitions](#4-definitions-and-vocabulary).
- **Scenario:** Joint uncertain path; see [scenario paths](#step-5--build-scenario-paths-and-random-streams).
- **Search budget:** Allowed simulation effort; see [budget](#step-6--plan-the-compute-budget).
- **Selection bias:** Optimism caused by choosing on noisy estimates; see [mental model](#3-the-core-mental-model).
- **Service event:** Defined per-path success condition; see [service evaluation](#step-9--evaluate-stochastic-feasibility).
- **Standard error:** Estimated sampling variation of an estimator; see [worked example](#6-worked-example).
- **Stress scenario:** A deliberately challenging conditional case; see [stress testing](#step-11--stress-and-sensitivity-test).
- **Structural error:** Missing or incorrect model mechanisms; see [model validation](#step-4--check-that-the-simulator-is-decision-valid).
- **Terminal value:** End-of-horizon economic treatment; see [code assumptions](#7-code-query-or-calculation).

## 16. Sources

1. Michael C. Fu, Fred W. Glover, and James April. “Simulation Optimization: A Review, New Developments, and Applications.” *Proceedings of the 2005 Winter Simulation Conference*. [DOI: 10.1109/WSC.2005.1574231](https://doi.org/10.1109/WSC.2005.1574231). Review of simulation-optimization methods and applications.
2. L. Jeff Hong, Weiwei Fan, and Jun Luo. “Review on Ranking and Selection: A New Perspective.” *Frontiers of Engineering Management* 8 (2021), 321–343. [DOI: 10.1007/s42524-021-0152-6](https://doi.org/10.1007/s42524-021-0152-6). Ranking and selection with noisy simulation outputs.
3. Wassily Hoeffding. “Probability Inequalities for Sums of Bounded Random Variables.” *Journal of the American Statistical Association* 58 (1963), 13–30. [DOI: 10.1080/01621459.1963.10500830](https://doi.org/10.1080/01621459.1963.10500830). Basis for the conservative bounded-variable inequality used here.
4. Barry L. Nelson. *Foundations and Methods of Stochastic Simulation: A First Course*. Springer, 2013. [Publisher page](https://link.springer.com/book/10.1007/978-1-4614-6160-9). Simulation experiment design and statistical analysis.
5. [Python standard library: `random`](https://docs.python.org/3/library/random.html). Pseudorandom generation and reproducibility notes.
6. [Python standard library: `statistics`](https://docs.python.org/3/library/statistics.html). Sample standard deviation and mean used in the code.
7. [GS1 standards](https://www.gs1.org/standards). Identification and data-sharing standards; consult current implementation guidance.
8. [ISO 31000:2018, Risk management — Guidelines](https://www.iso.org/standard/65694.html). Risk-management guidance, not an optimization mandate.
9. [ISO/IEC 27001:2022, Information security management systems](https://www.iso.org/standard/27001). Information-security management requirements where applicable.
10. [Regulation (EU) 2016/679, GDPR](https://eur-lex.europa.eu/eli/reg/2016/679/oj). Primary legal text; applicability depends on the processing and jurisdiction.
11. [UPS: ORION](https://about.ups.com/us/en/our-company/innovation-driven/orion.html). Company description of its route-optimization initiative; not evidence of the illustrative simulation's results.
12. [Winter Simulation Conference proceedings](https://www.informs-sim.org/). Searchable primary conference proceedings for simulation methods and applications.

## 17. Links

- [Module 20 index](_index.md): sequence and topic map.
- [Module 20 survey](../../../course/20_digital_twins_simulation_optimization_uncertainty.md): broader uncertainty and digital-twin context.
- [Discrete-event simulation](02_discrete-event-simulation.md): queues, resources, and event logic.
- [Input modelling](03_input-modeling.md): demand, lead time, and dependence.
- [Validation](04_validation.md): assessing fitness for a decision.
- [Experiment design](05_experiment-design.md): replications, pairing, and statistical comparisons.
- [Stochastic optimization](06_stochastic-optimization.md): scenario-based decision formulations.
- [Robust and distributionally robust optimization](07_robust-dro.md): sensitivity to uncertain distributions.
- [Simulation and optimization production labs](../../../course/30_simulation_optimization_production_labs.md): practical simulation-study material.
